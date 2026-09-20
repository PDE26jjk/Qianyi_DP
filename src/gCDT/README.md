# gCDT (vendored)

GPU constrained Delaunay triangulation, used as an optional triangulation
backend for `geometry.sample_points` (the default stays gDel2D).

Upstream: https://github.com/yingtix/gCDT (revision `c5252b1`), by Peng Fan,
Min Tang, Ruofeng Tong, Lili He, Peng Du and Hailong Li; "gCDT: A Highly
Parallel GPU Algorithm for Large-Scale Constrained Delaunay Triangulation",
ACM SIGGRAPH 2026 (journal version: Computer-Aided Design, 2025).

Local changes on top of upstream:

* device memory: the eager ~3.2 GB bucket pool (which the demo initialised
  twice, i.e. ~6.4 GB) was replaced by a lazy, size-exact cache with a
  configurable limit, so the library also runs on low-end GPUs;
* CUB scratch is derived from the actual scene size instead of the hard-coded
  983743231 bytes of the upstream demo;
* the caller can carry a vertex-index map (`origIndex`) through the Morton
  reordering, so output vertices can be mapped back to the caller's ids;
* every result is validated on the device and invalid runs are retried with a
  deterministic sub-quantum perturbation of the quantised vertices; the reason
  is documented in `include/validate.h` (exactly collinear constraints drive
  the constraint stage into a degenerate branch);
* the demo entry point, the scene reader, the drawing helpers, commented-out
  code and the debug dumps were removed; the library entry point is
  `gcdtBuildMesh()` in `include/pipeline.h`.

Upstream has no license file; treat this copy as "all rights reserved by the
authors" until they grant one.
