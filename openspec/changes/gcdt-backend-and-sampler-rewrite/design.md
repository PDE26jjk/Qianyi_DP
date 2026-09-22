## Context

See `proposal.md` - Why. The existing pipeline normalises a panel by its
bounding box, places one jittered point per interior cell, relaxes them with a
capped repulsion, then runs a constrained Delaunay pass and drops triangles
outside the domain. The inside/outside classification used to come from a
Vulkan rasteriser, and the triangulation step only had gDel2D. The frontend
passes a boundary discretised at or below its `granularity`, so any change to
the cell-length semantics directly changes panel density and seam quality.

## Goals / Non-Goals

**Goals:**

- Remove the graphics-stack dependency from the geometry library.
- Make the requested resolution match the produced edge length.
- Guarantee that legal panels (boundary spacing at or below the requested
  length) have no zero-area triangles and no near-boundary sliver ring.
- Keep gDel2D as the safe default and make gCDT an opt-in backend.

**Non-Goals:**

- Replacing the jittered-grid sampler with a true Poisson-disc or CVT sampler.
- A sparse/unbounded sampler grid; the current grid stays proportional to the
  bounding box.
- Changing gDel2D, or claiming gCDT is universally faster or more robust.

## Decisions

### CUDA parity mask instead of the Vulkan rasteriser

Constraint edges are bucketed into fine rows (`cell / 4`) and each point
classifies itself with a parity ray cast over that row plus a clearance-aware
window. This keeps the mask exact per point, removes Vulkan and its shader
assets, and gives the triangle filter a mask it can reuse.

Alternatives: a BVH over the constraint edges (more code, less predictable cost
for the small per-point queries) and a coarse cell-resolution mask (the source
of the old concave-boundary leaks).

### Cell length equals the requested edge length

The old sampler used a cell of `radius / sqrt(2)` with a repulsion reach of one
cell, which produced edges around `0.73 x radius`. Setting the cell to `radius`
fixes the semantics, but the repulsion reach must then be expressed in cells,
not in radius: keeping `radius / sqrt(2)` left a reach of 0.707 cells, so
jittered neighbours between 0.707 and 1 cell were never pushed apart and the
mesh became Poisson-like. The reach is therefore one cell, and the measured
median interior edge is ~1.02 x radius.

Alternative: keep the old cell size and scale the reported radius. Rejected
because it couples density to a fudge factor instead of making the argument
mean what callers expect.

### Constraint clearance instead of a stronger area threshold

A pure area/height threshold cannot separate a collinear sliver from a legal
thin triangle near a long boundary: raising it removes real triangles and
leaves interior cracks (measured 19 stray edges on a coarse square, 158 on a
2 mm coarse-hole panel). The sampler instead keeps interior points at least
`boundary_margin_cells` (default 0.8) from every constraint edge. The margin
covers the outer loop, holes and internal lines, and is applied during
generation, relaxation and validation. The sliver filter is then only a
numerical safety net (normalised cross-product floor `1e-8`).

Alternatives: an offset ring of boundary points (better still, but requires
curve offsetting at corners) and post-filtering with re-triangulation (cannot
know which triangle is the artifact).

### gCDT as an opt-in backend

gCDT is vendored, validates each result and falls back to gDel2D on failure.
The default stays gDel2D because gCDT is only clearly faster on simple panels
and has shown a coverage gap its validation does not catch (see Risks).

### Additive tuning surface

`triangulator`, the four relaxation parameters and `boundary_margin_cells` are
optional keyword arguments with the shipped defaults; the five-argument form
keeps working. An env-gated (`QYDP_SAMPLE_PROFILE=1`) per-phase CUDA-event
breakdown is available for profiling and is off by default.

### Degenerate input is refused at the entry point, and never repaired

The frontend hands over the outline it sampled and de-duplicated: it merges
samples closer than `granularity * 0.02` and maps its edge list through that
merge, so a piece whose samples the merge collapses arrives as a run of edges
that run from a point to itself. A constrained triangulation is not defined
for a constraint of zero length, gDel2D's insertion of one never converges, and
the call then never returns: the device spins at 100% and the host blocks in a
synchronise. Two exactly coincident points, a non-finite coordinate and a
`curve_sizes` that does not describe the edge list are the same kind of input -
one the triangulator cannot be asked to work on. This is not new to the rewrite
(the pre-rewrite Vulkan-based sampler hung on the same inputs), but the sampler
entry point is the only place that sees the caller's own indexing, so that is
where the check belongs.

The sampler validates instead of repairing. It refuses coincident points naming
both indices, refuses a constraint edge that runs from a point to itself naming
the edge, refuses a non-finite point, and refuses edge counts that do not add up
to the list. It deliberately does not merge the points, drop the edges, clamp
the counts or remap the triangles: the returned point list is index-for-index
the caller's own list - the frontend maps its panel vertices, sewing ends and
cached per-vertex data through those indices - so a merge in here would shift
every index after the merged pair, and a dropped constraint would hand back a
mesh that quietly does not follow the outline the caller drew. Merging is what
the frontend's own de-duplication pass does; the engine's job is to say exactly
what is wrong with the input it was given.

The check costs one pass over the caller's points (a flat open-addressed cell
table, `coincidence_epsilon` = 1e-6 of the domain, far below one sampling cell
and far below any caller merge threshold): on a 160k-point stress panel the
boundary is a few hundred points, so the pass is under a millisecond of a
~300 ms call. A pairwise comparison of coincident points must cover every
caller point, not only the ones a constraint edge names: measured with the
check removed, a duplicated point that no edge touches still spins gDel2D.

### A failure on the triangulation route is thrown, never an exit

Validation covers the inputs the sampler can describe as degenerate, but the
triangulator has failure paths of its own, and the vendored gDel2D reported
them the way a command-line program does: print a message and call `exit(-1)`.
In a Python extension that ends the *interpreter*, and the frontend's
interpreter is Blender, so a panel whose outline collapsed onto one line (its
sampled points are all collinear, so gDel2D has no non-degenerate kernel
triangle) printed "Input too degenerate!!!" and took Blender down with it: no
exception for the frontend to catch, nothing left to recover. The same pattern
sat in gDel2D's CUDA error checks, its allocation failures and its counter
guard, and in the project's own `CUDA_CHECK` helper, so any CUDA failure
anywhere in the engine was fatal to the session.

All of those now throw `std::runtime_error` carrying the same information (the
file, the line and `cudaGetErrorString` where there is one), which pybind11
turns into a Python `RuntimeError`. The paths are marked as local modifications
in `src/gDel2D/README.txt` and in `src/common/cuda_utils.h`. Verified from
Blender: a collapsed outline raises a catchable error and the next
`sample_points` call in the same session returns a mesh.

## Risks / Trade-offs

- [Sampler memory grows with `(bounding-box extent / radius)^2`] -> Documented
  limitation; a sparse grid is future work. Stress tests at 2 mm on a 1 m panel
  allocate ~40 MB, at 0.125 mm ~2.5 GB, and a thin-strip case at 1.4 GB hit a
  driver-level fail-fast on a shared GPU.
- [gCDT can return a mesh with missing area that its edge-manifold validation
  accepts] -> gDel2D remains the default; the backend is opt-in and the bridge
  falls back when gCDT reports an invalid result. A constraint-coverage check
  is future work.
- [The boundary margin removes ~1-2% of near-boundary interior points] ->
  Median edge length and global quality are unchanged; the trade buys a large
  improvement in seam-adjacent minimum quality (0.16-0.49 before, 0.65-0.72
  after on legal scenes).
- [Timing on a shared desktop GPU is noisy] -> Benchmarks report the fastest of
  several repeats and phase deltas from CUDA events; repeat spread is recorded
  rather than hidden.
- [Removing Vulkan deletes the only rasteriser in the tree] -> The parity mask
  is covered by the sampling tests and by the concave/hole scenes; no other
  module used `src/graphics`.
- [A constraint list that crosses itself is still passed to the triangulator]
  -> Measured: a bow-tie outline returns a mesh in bounded time on both
  backends, so the sampler does not duplicate the frontend's own crossing test.
  The sampler's guarantee is "returns in bounded time for the degenerate lists
  callers produce", not "produces a meaningful mesh for a meaningless outline".
- [Callers that hand over a coalesced index list now get an error where they
  used to hang] -> That is the point: the frontend's corner command already
  refuses to leave a piece shorter than `max(granularity, 5 mm)`, and a caller
  that wants shorter pieces has to drop the edges its own merge collapsed
  (the message names the first one). The alternative - the engine repairing the
  list - was rejected because it changes the caller's vertex order or silently
  drops constraints from the outline.
- [A CUDA error leaves the device in a state the next call cannot use] ->
  Throwing keeps the process alive and the error text says what failed, but a
  context lost to an illegal access stays lost; the caller sees the failure on
  the following call too instead of losing the session. Recovering the device
  itself is out of scope here.

## Migration Plan

No data migration. The Python API is additive; the Blender frontend's existing
five-argument call takes the new defaults automatically. Rollback is a revert
of the change commit, which restores the Vulkan mask and the old sampler.
