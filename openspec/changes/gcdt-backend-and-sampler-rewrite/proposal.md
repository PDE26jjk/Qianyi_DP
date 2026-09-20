## Why

The pattern mesher that feeds every cloth panel was tied to a Vulkan rasteriser
for its inside/outside test, only offered gDel2D, and carried three defects
that reached simulation: hole loops whose winding matched the outer loop were
ignored, the requested resolution was off by a factor of sqrt(2), and
near-degenerate sliver triangles could survive at straight pattern edges and
seams. Removing the Vulkan dependency also removes a graphics-stack
requirement from a geometry-only consumer.

## What Changes

- Replace the Vulkan rasteriser with a CUDA inside/outside mask: constraint
  edges are bucketed per fine row and a parity ray cast classifies points.
  The test is orientation independent and driven by `is_holes`.
- Make the `radius` argument mean the target triangle edge length: one interior
  point per cell of that length, so the relaxed median interior edge lands at
  ~1.02x the requested value instead of ~0.73x.
- Add a constraint-edge clearance (`boundary_margin_cells`, default 0.8 edge
  lengths) that interior points must keep from every constraint edge (outer
  loop, holes, internal lines) during generation, relaxation and validation,
  so straight boundaries and seams cannot collect near-boundary slivers.
- Restore the numerical sliver filter (normalised cross product floor) and keep
  the exact centroid-inside triangle filter, replacing the old cell-resolution
  test and its "any vertex is an interior point" bypass.
- Add a triangulator selector: gDel2D stays the default; the vendored gCDT is
  an optional backend (`triangulator=1`) that validates each result and falls
  back to gDel2D when it cannot produce a valid mesh.
- Expose the two relaxation passes (`relax_gain1` / `relax_iters1` /
  `relax_gain2` / `relax_iters2`) and `boundary_margin_cells` as optional
  keyword arguments; existing five-argument calls keep their behaviour.
- Remove `src/graphics` (Vulkan rasteriser, shaders, SPIR-V assets) and the
  backup sampler; add an env-gated per-phase timing hook for the sampler.

## Capabilities

### New Capabilities

- `pattern-meshing`: the contract for turning a closed pattern outline plus
  internal lines into a triangle mesh - resolution semantics, hole handling,
  constraint-edge clearance, domain filtering, degenerate-triangle policy and
  the selectable triangulation backend.

### Modified Capabilities

None.

## Impact

- Geometry API: `qydp.geometry.sample_points` gains optional keyword arguments
  (`triangulator`, relaxation gains/iterations, `boundary_margin_cells`); the
  five required positional arguments are unchanged.
- Code: rewritten `src/geometry/sample_points.cu`, `src/geometry/sampler.h`,
  `src/geometry_interface.*`, `src/main.cpp`; new `src/gCDT/` and
  `src/geometry/gcdt_backend.cu`; removed `src/graphics/` and
  `src/geometry/sample_points_bak.cu`; build files no longer link the graphics
  module.
- Tests/docs: sampling tests, the harness panel spacing calibration and the
  README "Pattern meshing" section are updated.
- Dependencies: the geometry library no longer needs the Vulkan SDK; gCDT is
  vendored and built with static CUDA runtime linkage.
