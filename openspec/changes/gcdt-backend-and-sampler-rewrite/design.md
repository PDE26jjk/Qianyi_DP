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

## Migration Plan

No data migration. The Python API is additive; the Blender frontend's existing
five-argument call takes the new defaults automatically. Rollback is a revert
of the change commit, which restores the Vulkan mask and the old sampler.
