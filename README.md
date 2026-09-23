# Qianyi_DP

这个项目是Qianyi的数据处理后端，使用了CUDA加速各种运算密集的程序。

## Engine input specification

[`docs/engine_input_spec.md`](docs/engine_input_spec.md) is the authoritative
description of the engine inputs: the `input_data` payload (mesh entries, sewing
entries, index spaces, ordering rules, units), the runtime parameter surface with
defaults, the driver order, and the diagnostics emitted for invalid input.



## 引用和参考

三角剖分引用了 [gDel2D](https://www.comp.nus.edu.sg/~tants/gdel3d.html)

lbvh参考了 [ppf-contact-solver](https://github.com/st-tech/ppf-contact-solver)

## Pattern meshing (`geometry.sample_points`)

The Blender frontend derives every cloth mesh from this entry point
(`pattern_mesh.generate_pattern_mesh` -> `qydp.geometry.sample_points`), so the
solver's triangle quality comes from the pipeline below. It is *not* Poisson
disc sampling:

1. normalise the domain by the bounding box (the cell grid is scale free);
2. build an inside/outside mask for the closed boundary curves (outer loop plus
   holes) on the device, with per-row edge buckets and a parity ray cast (no
   rasteriser / Vulkan dependency); the test is orientation independent and
   honours `is_holes`, so hole loops work with either winding;
3. place the input boundary points and the edge midpoints on the grid as fixed
   obstacles;
4. **stratified sampling** - one jittered point per interior cell; the cell
   length equals the requested `radius`, i.e. the argument is the target
   triangle edge length;
5. **repulsion relaxation** - neighbours come from a uniform grid holding at
   most four points per cell; interior points are pushed away from neighbours
   within `radius` while boundary points stay fixed (a move that would leave the
   domain is dropped, so relaxed points cannot drift outside at concavities).
   `sample_points` runs two passes with decreasing gain: `0.02 x 15`
   iterations, then `0.01 x 35`. Both passes are tunable through the optional
   `relax_gain1` / `relax_iters1` / `relax_gain2` / `relax_iters2` keyword
   arguments, which default to those values. Interior points also keep a
   `boundary_margin_cells` clearance from every constraint edge (outer loop,
   holes and internal lines); the default is 0.8 edge lengths, which keeps the
   seam-adjacent triangles well shaped, so a straight boundary cannot collect
   collinear points or a ring of near-boundary slivers;
6. validate the interior points, then constrained Delaunay (gDel2D) with all
   boundary and hole edges as constraints, then drop triangles whose centroid is
   outside the polygon (exact test - no cell-resolution approximation) and
   triangles whose cross product is below the numerical sliver floor.

`sample_points(..., triangulator=1)` selects the optional gCDT backend
(`src/gCDT`, vendored) instead of gDel2D; gCDT validates every result and the
sampler falls back to gDel2D when it cannot produce a valid mesh. Measured on
this machine, gCDT is ~3.5-4x faster on panels without holes.

The repulsion reach is one sampling cell (= `radius`) and every move is capped,
so the relaxed spacing approaches the requested edge length (the median interior
edge is ~1.02x `radius`); `tests/algo/test_sampling.py` asserts the conservative
`radius / sqrt(2)` lower bound and `tests/bench/test_sampling_bench.py` records
the cost.
Setting `QYDP_SAMPLE_PROFILE=1` prints a per-phase CUDA-event timing breakdown
(setup, mask, point generation, relaxation, triangulation, triangle filtering)
for each `sample_points` call.

## External forces (pressure and wind)

The PDNewton solver applies two additional external forces. Every other solver
ignores them.

PDNewton is also the only production-ready solver. `VBD`, `XPBD` and `Explicit`
are experimental: they are not validated, they are not part of the standard
configuration, and they are kept for research and comparison only.

- **Per-object pressure** is a constant load along the surface normal, scaled by
  face area: `F = p * A * n`, split across the face's three vertices. A positive
  value inflates, a negative value pulls inward, and `0` (the default) is inert.
  Pass `pressure` (Pa) in a mesh entry of `input_data`. No enclosed volume, gas
  state, or rest volume is involved, so the inflated size is an outcome of the
  pressure and the material stiffness, and a watertight shell is not required.
  Cloth triangles are re-oriented in the pattern plane at load and the force is
  evaluated on the simulated 3D positions, so the side a panel inflates towards
  is decided by how the panel is placed in 3D (`vertices_sim` and the object
  world matrix): two panels sewn back to back need opposite outward placements.
- **Wind** is a world-space velocity field: a base velocity plus an optional
  temporal gust and a divergence-free spatial turbulence. The aerodynamic force
  follows the separated quadratic lift/drag model from Disney's Frozen
  (SIGGRAPH 2014 talk, DOI 10.1145/2614106.2614120):
  `F = 0.5 * rho * A * [(C_D - C_L) (v . n) v + C_L |v|^2 n]`, where `v` is the
  air velocity relative to the surface (`wind - vertex_velocity`, so a positive
  normal projection means the air pushes the surface). A mesh entry may override
  the coefficients with `wind_drag` / `wind_lift`; a negative value means "use
  the global parameter".
- Both loads enter the solver as right-hand-side force terms only: they
  contribute no Hessian block and no diagonal entry to the linear system.

Global parameters (through `set_parameter` / `set_parameters`):

| Key | Meaning | Default |
| --- | --- | --- |
| `wind_x`, `wind_y`, `wind_z` | base wind velocity (m/s) | 0 |
| `wind_gust` | gust amplitude as a fraction of the base speed | 0 |
| `wind_gust_frequency` | gust noise frequency (Hz) | 0.5 |
| `wind_turbulence` | turbulence amplitude (m/s) | 0 |
| `wind_turbulence_scale` | turbulence spatial frequency (1/m) | 0.5 |
| `wind_turbulence_speed` | advection speed of the noise pattern (m/s) | 1 |
| `air_density` | air density (kg/m^3) | 1.225 |
| `wind_drag_coefficient` | global drag coefficient `C_D` | 1.0 |
| `wind_lift_coefficient` | global lift coefficient `C_L` | 0.0 |

Wind is inert while `wind_x = wind_y = wind_z = 0` and `wind_turbulence = 0`.

## Portability

Machine-specific paths, local build outputs, and local environment details
(for example the CMake output location, the Python environment, or the Blender
frontend checkout) MUST NOT appear in committed docs, tests, or OpenSpec
artifacts. They are recorded in `LOCAL_DEV.md` at the repository root, which is
gitignored and never committed.



