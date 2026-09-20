# Qianyi_DP

这个项目是Qianyi的数据处理后端，使用了CUDA加速各种运算密集的程序。



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

## Portability

Machine-specific paths, local build outputs, and local environment details
(for example the CMake output location, the Python environment, or the Blender
frontend checkout) MUST NOT appear in committed docs, tests, or OpenSpec
artifacts. They are recorded in `LOCAL_DEV.md` at the repository root, which is
gitignored and never committed.



