## 1. Sampler rewrite

- [x] 1.1 Replace the Vulkan inside/outside rasteriser with per-fine-row
  constraint edge buckets and a parity ray cast; verify `pytest -m quick`
  passes without the `src/graphics` module
- [x] 1.2 Make `is_holes` orientation independent (outer loop minus every hole
  loop) and verify clockwise and counter-clockwise inputs produce the same
  domain
- [x] 1.3 Set the sampling cell length to the requested edge length and the
  repulsion reach to one cell; verify the median interior edge is within 15%
  of the requested value on a densely discretised panel
- [x] 1.4 Filter triangles with the exact centroid-inside mask test and remove
  the old cell-resolution test and the "any vertex is interior" bypass; verify
  concave outlines and holes leak no triangle
- [x] 1.5 Remove `src/graphics`, its shader assets and the backup sampler;
  verify the geometry target builds and links without them

## 2. Constraint clearance and sliver guard

- [x] 2.1 Add `insideDomainWithMargin` and apply it to interior point
  generation, relaxation moves and interior point validation; verify no
  generated point is closer than the margin to a constraint edge
- [x] 2.2 Include internal (non-hole) lines in the clearance test, not only the
  outer loop and holes; verify a panel with a straight internal line keeps the
  same clearance and has no stray open edges
- [x] 2.3 Expose `boundary_margin_cells` on `sample_points` with default 0.8;
  verify the five-argument call still works and explicit values change the
  measured seam-adjacent quality
- [x] 2.4 Tune the default margin against legal scenes (boundary spacing at or
  below the requested length); verify the minimum quality of triangles touching
  a constraint edge is at least 0.5
- [x] 2.5 Restore the normalised cross-product sliver filter (`1e-8`) as a
  numerical safety net; verify a 2 mm straight edge produces no zero-area
  triangle and keeps the covered area within 0.1% of the panel

## 3. gCDT backend

- [x] 3.1 Vendor gCDT, remove its demo driver and diagnostic output, and keep
  the exact-size lazy memory pool and the `origIndex` vertex mapping; verify
  the vendored target builds
- [x] 3.2 Add the gCDT validation gate, the low-quantum perturbation retry and
  the gDel2D fallback in the sampler bridge; verify a rejected gCDT mesh
  returns the gDel2D result instead of failing
- [x] 3.3 Expose `triangulator` (0 = gDel2D, 1 = gCDT) on `sample_points`;
  verify both backends return valid meshes on the sampling test scenes
- [x] 3.4 Record the backend comparison for planning: gCDT is ~1.2-1.7x faster
  on simple panels but has no advantage on L-shaped/comb panels and can lose
  ~0.37% area on a 64-tooth comb while passing its own validation

## 4. API, docs and build

- [x] 4.1 Expose the relaxation gains and iteration counts as optional keyword
  arguments with the shipped defaults; verify existing callers are unaffected
- [x] 4.2 Add the `QYDP_SAMPLE_PROFILE` per-phase timing hook and document the
  sampling pipeline, the relaxation parameters, the margin and the backend
  selector in `README.md`
- [x] 4.3 Calibrate the test harness panel spacing to the new resolution
  semantics so harness mesh density is unchanged
- [x] 4.4 Update the CMake targets for the gCDT library, static CUDA runtime
  linkage and the removed graphics module

## 5. Verification

- [x] 5.1 Run `pytest tests/algo/test_sampling.py -q` and verify all sampling
  tests pass
- [x] 5.2 Run `pytest -m quick -q` and verify only the pre-existing
  `test_standard_scene_smoke` failure remains
- [x] 5.3 Measure the sampler/triangulator split on legal panels; verify the
  breakdown is recorded (triangulation dominates, sampling is 6-24%)
- [x] 5.4 Stress thin strips, deep notches, combs, stars, hole patterns,
  collinear constraints and duplicate vertices; verify no algorithmic crash and
  record the degenerate-input findings

## 6. Degenerate input is refused, not repaired

- [x] 6.1 Refuse coincident boundary points inside `sample_points` with an
  error naming both indices, and refuse a constraint edge that names the same
  point twice with an error naming the edge; verify a caller-produced list with
  thousands of zero-length edges reports the first one instead of spinning the
  device, and that no point or edge is merged, dropped or reordered
- [x] 6.2 Refuse a non-finite boundary point and a `curve_sizes` that does not
  describe the edge list, each with an error naming the offender; verify the
  NaN/Inf input and the under- and over-counted counts no longer hang or crash
  the process
- [x] 6.3 Verify a valid repeated constraint edge and a crossing outline still
  return in bounded time, then re-run the corner-radius sweep, the two-call
  repro and `pytest -m quick -q`; verify clean inputs return the same point and
  triangle counts as before the change and the pre-existing
  `test_standard_scene_smoke` failure is unchanged
- [x] 6.4 Replace the print-and-exit paths on the triangulation route with
  exceptions: gDel2D's non-degenerate-kernel check ("Input too degenerate!!!"),
  its CUDA error checks, its allocation failures and its counter guard, plus the
  project's own `CUDA_CHECK` helper; record the local modification in the
  vendored gDel2D README
- [x] 6.5 Verify from Blender that an outline collapsed onto one line raises a
  catchable `RuntimeError` and that the next `sample_points` call in the same
  session still returns a mesh
