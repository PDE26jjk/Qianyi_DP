# Proposal: real-time-broadphase

## Why

Real-time speed (RTS = simulated time / wall-clock time) is the main product
goal for the interactive frontend, and the current solver is roughly an order
of magnitude away from it. Measured with the study-1 scene configuration
(GarmentCodeData element `rand_00YONAPXZE`, PDNewton, `update(0.003)`, one
substep per frame) on an RTX 3070 Laptop GPU (sm_86, 8 GiB), one frame costs
**~27-40 ms** depending on GPU clock state, i.e. RTS ~0.08-0.11. Reaching
RTS 1.0 at this configuration needs ~3 ms/frame, about 9x less work.

A profile of that frame (Nsight Systems, 35 frames, GPU busy 25.7 ms/frame,
677 kernel launches/frame) attributes the cost as:

| Stage | ms/frame | share |
|---|---|---|
| Collision broad phase (`query_vf` + `query_ee` + `query_ef`) | 11.71 | **45.6%** |
| Bending (`compute_dihedral_bending_AOGS`) | 3.93 | 15.3% |
| Linear solve (PCG: `A_mul_x_offdiag`, `Jx_mult_x_diag`, cub reductions) | 3.23 | 12.6% |
| Contact forces (`compute_vf_force`, `compute_ee_force`) | 2.05 | 8.0% |
| Stiffness FEM (`compute_BW_FEM`) | 1.81 | 7.1% |
| Untangling (`solve_untangling_kernel`) | 1.39 | 5.4% |
| Everything else (sewing, smoothing, BVH maintenance, steps) | 1.55 | 6.0% |

The broad phase is therefore the single largest lever, and this change starts
there. It is also the stage whose cost depends most on how the scene is
configured, so the change records the measurements that decide where the next
round of work goes.

## What Changes

The change started at the broad phase and ended up covering the frame. What it
ships:

- A tight, non-swept broad phase (`tight_broad_phase`, default 1): the tree and
  the query boxes are inflated only by the contact radius instead of by the
  substep's motion, so the traversal covers far fewer nodes.
- The three broad-phase BVH queries overlapped on separate CUDA streams
  (`bvh_streams`, default 1), because they read the same BVHs but write
  disjoint candidate buffers and each one alone leaves the SMs under-utilized.
- The traversal-ordered query mapping (`bvh_query_order`, default 1), the
  project's earlier `nodes[i].x - 1` leaf mapping, now isolated and measured.
- The whole Projective-Dynamics loop replayed from a captured CUDA graph
  (`pd_cuda_graph`, default 1). The frame issues ~690 kernels and a dependent
  launch costs ~12 us of GPU idle time on this driver, so the capture removes
  several milliseconds of launch gaps.
- PCG with 2 iterations in place of Jacobi with 5 in the harness preset, from
  the linear-solver comparison.
- Caller-facing defaults that the measurements support: `dt = step_h = 0.0045`,
  `pd_iters = 5`, `linear_iters = 2`, applied to the debug window and the
  study-1 notebook.
- Record the measurement methodology, the per-kernel evidence and the
  rejected alternatives in `design.md`, so the next optimization round starts
  from data instead of intuition - including the experiments that were
  implemented, measured, and then removed again (Chebyshev, the VBD seam
  diagnostic, the lazy bending Jacobian, the warp-cooperative traversal, the
  half-precision Hessian).

Out of scope for this change: restructuring the broad phase itself beyond the
above (dual-tree traversal, packed node layout, SAH/leaf-batched tree), a
Schwarz-style preconditioner, and the seam-as-hard-constraint redesign. They
are recorded as follow-ups in `tasks.md`.
