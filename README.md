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
2. rasterise the closed boundary curves and holes into an inside/outside mask;
3. place the input boundary points and the edge midpoints on the grid as fixed
   obstacles;
4. **stratified sampling** - one jittered point per interior cell, cell length
   `radius / sqrt(2)`;
5. **repulsion relaxation** - neighbours come from a uniform grid holding at
   most four points per cell; interior points are pushed away from neighbours
   within `radius` while boundary points stay fixed. `sample_points` runs two
   passes with decreasing gain: `0.02 x 15` iterations, then `0.01 x 35`;
6. validate the interior points, then constrained Delaunay (gDel2D) with all
   boundary and hole edges as constraints, then drop invalid triangles.

Because of the capped repulsion, the guaranteed minimum spacing is
`radius / sqrt(2)`, not `radius`; `tests/algo/test_sampling.py` asserts that
property and `tests/bench/test_sampling_bench.py` records its cost.

## Portability

Machine-specific paths, local build outputs, and local environment details
(for example the CMake output location, the Python environment, or the Blender
frontend checkout) MUST NOT appear in committed docs, tests, or OpenSpec
artifacts. They are recorded in `LOCAL_DEV.md` at the repository root, which is
gitignored and never committed.



