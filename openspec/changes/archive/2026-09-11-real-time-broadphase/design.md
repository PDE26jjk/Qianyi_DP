# Design: real-time-broadphase

## What shipped, and what did not

Read this first: the rest of the document is the chronological evidence, most
of which is experiments that failed. This is the merged result.

Retained in the code, all on by default and each with a parameter that
restores the old behaviour for A/B measurement:

| change | parameter | measured gain |
|---|---|---|
| Tight, non-swept broad phase | `tight_broad_phase` (1) | x1.27-1.45 |
| Three broad-phase queries overlapped on streams | `bvh_streams` (1) | x1.04-1.09 |
| Query order = Morton leaf order (`nodes[i].x - 1`) | `bvh_query_order` (1) | x1.07-1.08 |
| Whole PD loop replayed from a captured CUDA graph | `pd_cuda_graph` (1) | x1.15-1.23 |
| PCG with 2 iterations instead of Jacobi with 5 (preset) | - | x1.10-1.31 |

Retained as caller-facing defaults (the window and the study-1 notebook):
`dt = step_h = 0.0045`, `pd_iters = 5`, `linear_iters = 2`, PCG. Together with
the code changes this takes the study-1 element from RTS 0.101 to 0.457 (x4.5),
of which about x1.7 is code and the rest is the iteration budget and the step
length. The quality envelope of that step is in "Recommended defaults after
round 1".

Implemented, measured, and **removed again** so the tree only carries
behaviour that earned its place - each one is documented below with the
measurement that killed it:

| experiment | measured result | where |
|---|---|---|
| Chebyshev semi-iteration as a linear solver | x1.23 faster, but tears seams open (350 mm) at the default budget - not converged enough | "Implemented: a reduction-free linear solver, and what it cost" |
| `vbd_project_stitches` (seam constraint for the VBD solver) | closes the seams but triples VBD's frame cost; VBD is not the shipped path | "Why VBD looked faster" |
| Lazy bending Jacobian (`bend_freeze_after`) | x1.08-1.11 at a heavy iteration budget, a wash (8.42 vs 8.30 ms) at the shipped one | "The lazy bending Jacobian" |
| `ee_coop_query` (warp-cooperative traversal) | x0.94-0.97 on the frame | "Tried: warp-cooperative traversal" |
| Half-precision assembly (diagonal float + couplings half) | diverges; the operator spans 9-12 decades | "Why the half-precision Hessian cannot be copied" |

## Context

The goal is RTS (simulated seconds per wall-clock second). The reference
configuration is the study-1 scene: GarmentCodeData element
`rand_00YONAPXZE` (18 cloth panels, 29,784 cloth vertices / 86,896 cloth
edges / 57,130 cloth triangles, plus a 23,752 vertex / 71,250 edge / 47,500
triangle body mesh, 975 stitches), PDNewton with the study-1 parameter
overlay (`pd_iters=10`, `linear_iters=5`, `bending_model=2`,
`linear_solver_type=0` -> PCG, `query_radius=0.001`, `max_vel=100`), and
`update(0.003)` once per frame, which is one substep since `step_h=0.003`.

`Simulator::update()` only queues CUDA work, so a frame must be measured as
`update(dt)` plus a synchronizing read (`get_simulation_data()`); the
read-back itself costs 0.14 ms and `update()+sync` and `update()+read` agree
within 1 ms.

## Baseline measurements

Nsight Systems over 35 frames (all kernels, GPU busy 25.67 ms/frame,
677 launches/frame):

| Kernel | calls/frame | ms/frame |
|---|---|---|
| `query_ee_pairs_capsule_kernel` | 1 | 6.303 |
| `compute_dihedral_bending_AOGS` | 10 | 3.934 |
| `query_vf_pairs_capsule_kernel` | 1 | 3.677 |
| `A_mul_x_offdiag_kernel` | 60 | 2.418 |
| `compute_BW_FEM` | 10 | 1.812 |
| `query_ef_pairs_kernel` | 1 | 1.728 |
| `solve_untangling_kernel` | 10 | 1.388 |
| `compute_ee_force` | 10 | 1.363 |
| `compute_vf_force` | 10 | 0.683 |
| `cub::DeviceReduceKernel` (PCG dot products) | 110 | 0.424 |

The broad phase is 11.71 ms/frame (45.6%) and runs once per substep from
`SolverPDNewton::step`, so its cost scales with the number of substeps while
the per-iteration kernels scale with `pd_iters`. `step_h` halves the frame
time almost exactly (`step_h=0.0015` -> x0.64, `step_h=0.001` -> x0.31 in a
sequential sweep), which is consistent with "fixed cost per substep".

## Why the broad phase is the way it is

`Contact::collision_detect_broad_phase` launches one thread per primitive and
each thread traverses the whole BVH on its own with `BVH_QUERY_LOOP`, a
64-entry `unsigned int stack` (confirmed in local memory:
`cuobjdump --dump-resource-usage` reports `STACK:256` for `query_vf` and
`query_ef`, `STACK:288` for `query_ee`).

Both the query volumes and the tree bounds are *swept*: the queries are
capsules from the step-start position to the inertial prediction, and
`refit_*_offset_bvh_kernel` rebuilds internal node AABBs as the union of
`pos_prev` and `pos_target` (plus 1.5x thickness). Measured for one substep:

| Quantity | Value |
|---|---|
| `abs(pos_pred - pos_prev)` median / mean | 4.1 mm / 10.3 mm |
| p90 / p99 / max | 21 mm / 118 mm / 614 mm |
| vertices moving > 10 mm per 3 ms substep | 25.7% |

Per-query cost measured with Nsight Compute:

| Metric | `query_vf` | `query_ee` | `query_ef` |
|---|---|---|---|
| L1 load requests per query | 529 | **907** | 71 |
| L1 sectors per query | 1143 | **3851** | 300 |
| Instructions per query | 4000 | **6790** | 402 |
| L1 throughput | 2.3% | 29.1% | 58.3% |
| Issue utilization | 2.0% | 23.0% | 34.2% |
| Achieved occupancy | 29.6% | 34.7% | 78.6% |
| Registers / theoretical occupancy | 55 / 66.7% | 74 / 50% | 33 / 100% |

Interpretation: these kernels are **L1-throughput and node-visit bound**,
not stack bound and not compute bound. Roughly 450 tree nodes are visited per
edge query, two loads per node, and the warp then waits on the L1 latency of
the next dependent node.

The candidate buffers (16 slots per primitive) are frequently full, which
also truncates the traversal and silently drops pairs:

| Query | mean candidates | nonzero | buffer full |
|---|---|---|---|
| vertex -> face | 7.7 / 15 | 90.7% | 27.0% |
| edge -> edge | 3.8 / 15 | 49.5% | 15.0% |
| edge -> face | 12.2 / 15 | 100% | **61.0%** |

97.6% of the vertex-face candidates are cloth-cloth (the body mesh is only
2.4%), so the dominant work is cloth self-collision. Removing the body mesh
from the scene entirely only gives x1.09 - the cloth self-overlap grows when
the body is gone - so the body is not the lever.

## Round 1 change: overlap the three queries

The three broad-phase queries are independent: they read the same BVHs and
write three disjoint candidate arrays. Each one alone leaves the SMs
under-occupied. The change records a fork event on the default stream, makes
three non-blocking streams wait on it, launches one query per stream, and
joins back to the default stream with three more events before the narrow
phase runs.

### Measurement methodology

Laptop GPUs throttle: the same binary measured 27.2 ms/frame cold and
36-50 ms/frame hot (SM clock 1110-1770 MHz against a 2100 MHz maximum,
83-88 C). Every conclusion below therefore comes from **interleaved A/B
rounds** (each variant once per round, round-robin, report min and median),
not from sequential sweeps.

To remove the toolchain as a variable, the unmodified sources were also
built into the same CMake tree (the only difference being the change), and
that build is the baseline in the table below.

### Results

40 frames x 3 rounds on the study-1 configuration:

| Variant | min ms | median ms | vs baseline |
|---|---|---|---|
| baseline (unmodified sources, same toolchain) | 38.70 | 41.93 | - |
| sequential (`bvh_streams=0`) | 40.23 | 42.81 | x0.96 |
| **overlapped (`bvh_streams=1`)** | **37.36** | **38.55** | **x1.04 / x1.09** |

Repeats of the same binary with only the switch changed: 40 frames x 5 ->
43.31 ms to 39.79 ms (-8.1%); 150 frames x 3 -> 38.47 ms to 37.88 ms (-1.5%).

Nsight Systems confirms the mechanism: the three queries now overlap 1.66x
(19.05 ms of kernel time inside an 11.47 ms window, against 11.71 ms of
strictly serial time), but each kernel runs ~60% slower while co-resident
because they compete for the same L1/latency budget. The union window only
moves from 11.72 ms to 11.47 ms per frame, which is why the end-to-end gain
is a few percent rather than a multiple.

Correctness: `pytest -m quick` passes (12 passed, 20 deselected) against the
new module; finiteness, displacement statistics and candidate distributions
match the sequential run. The engine is known to be non-bitwise-deterministic
run to run, so equality is asserted on statistics, not on hashes.

## Rejected: overlap-filtered, near-first traversal

The first attempt replaced the stack discipline: push a child only if its
AABB overlaps the query, and push the farther child first so the nearer
subtree is popped first. Both policies stayed in the same binary behind
`bvh_query_mode`.

Result: **11-23% slower** (43.71-45.14 ms against 37.36-40.23 ms for the
original traversal in the same binary). The ncu numbers above explain why:
the filter adds two child-AABB loads and a center-distance computation per
internal node, on a kernel that is already bound by L1 traffic and node
visits. Reducing stack operations cannot pay for that. The experiment was
reverted; the code is not part of this change.

## Rejected: clamping velocities, subspace acceleration

- Clamping `max_vel` (100 -> 20 / 5 / 1) makes the frame *slower*
  (x0.96 / x0.67 / x0.75): the cloth then stays longer in the tangled
  configuration that produces most of the contact candidates.
- Subspace acceleration is already known to be a negative optimization in
  this codebase (it is disabled in `SolverPDNewton::init`); it is not a
  candidate for reducing the iteration count. Better preconditioning or a
  direct reduction of `pd_iters` is.

## Rejected: tree rebuild interval and occupancy hints

Two cheaper hypotheses were tested in the same interleaved harness and
rejected:

- **Rebuild the BVH more often.** A full Morton rebuild every 20 frames
  (refit in between with swept bounds) was suspected of letting the tree
  degrade, since swept node AABBs overlap far more than the geometry they
  bound. Rebuilding every frame, every 5 frames and every 100 frames all
  measured the same as the default within noise (31.3-31.5 ms min frame
  time), so tree *structure* staleness is not the driver.
- **Raise occupancy with `__launch_bounds__(256, 4)`.** This drops
  `query_ee` from 74 to 64 registers and raises theoretical occupancy from
  37.5% to 50% with no spills, but measured slightly slower (34.94 ms vs
  33.49 ms min over four rounds). The kernels are bound by L1 latency and
  node-visit count, so extra resident warps do not pay for the tighter
  register budget.

Together with the traversal experiment these results narrow the problem
precisely: the cost is the *number of tree nodes visited per query* (times
the L1 latency of each visit), driven by the swept query volumes and by how
densely the cloth overlaps itself. Only changes that reduce visits -
dual-tree traversal with a compact pair list, better tree quality, smaller
query volumes - can move it.

## Rejected: interleaved (packed) node records

Each traversal step currently issues two dependent-free loads: the node's
bounds from `aabbs[node]` and its child indices from `nodes[node]`. Packing
them into one 32-byte record (`BVHNode { AABB3D aabb; int2 child; }`), filled
by a small pass after every refit, removes one load and one address
computation per step. Built and measured against the same binary's
split-array path, 40 frames x 4 rounds:

| Variant | min ms | median ms |
|---|---|---|
| split arrays | 34.86 | 36.99 |
| packed records | 35.14 | 36.16 |

The result is a wash. Combined with the L1 hit rate (86-95%) and the
long-scoreboard stall share, this says the two loads were already issued in
parallel and served from L1, so the critical path per node visit is one L1
round trip either way. **Merging loads does not shorten it; only visiting
fewer nodes does.** The change was reverted (it also adds a per-substep pass).

## Rejected: Morton-range pruning of the self-collision traversal

The one change that genuinely removes node visits is a range prune. Because
the LBVH is built in Morton order with the leaves first, every subtree covers
a contiguous range of that order; storing the largest order in each subtree
(one bottom-up pass per rebuild) lets the self-collision query skip subtrees
that can only contain pairs already owned by an earlier edge. The dedup key
has to become the Morton order instead of the current length rank; the *set*
of stored pairs is unchanged, only which of the two edges owns a pair.

Measured against its own off-path in the same binary (40 frames x 4 rounds):

| Variant | min ms | median ms |
|---|---|---|
| prune off | 36.61 | 38.97 |
| prune on | 36.77 | 37.99 |

The mechanism is right but the measured effect is ~2% on the median and nil
on the minimum, because the visited nodes are dominated by mid- and
high-level nodes whose swept AABBs span wide ranges of the Morton order -
the range bound cannot prune those. The change was reverted: a semantics
change (pair ownership, and therefore which pairs survive the 16-slot
truncation) is not worth a 2% median.

This closes the search inside the current broad-phase structure: **six
independent attempts (traversal, velocity clamp, rebuild interval, occupancy,
memory layout, range pruning) all land between -23% and +2.5%**. The visited
node count is set by the conservative swept bounds, and no re-ordering or
re-scheduling of the existing traversal changes it.

## Rejected: freezing the bending stiffness block across iterations

The bending kernel is the largest assembly kernel (3.93 ms/frame, 10 calls,
128 registers) and it re-derives its block of the system matrix on every
Projective-Dynamics iteration, although that block is a function of the
element geometry. An opt-in `freeze_bending_matrix` was implemented: the
first iteration assembles the bending block into per-frame buffers, later
iterations restore those buffers and ask the kernel for forces only (the
kernel already supports a forces-only path when the matrix pointers are
null). 40 frames x 4 rounds, same binary:

| Variant | min ms | median ms | max displacement/frame |
|---|---|---|---|
| freeze off | 33.61 | 36.13 | 0.067 m |
| freeze on | 36.34 | 40.55 | **0.300 m** |

It is slower *and* it changes the simulation: freezing the Jacobian makes the
iterations diverge (5x the per-frame motion, which feeds straight back into
the swept broad-phase volumes). Reverted. The lesson is worth recording: for
this scene, any change that alters the solver trajectory can cost more in the
collision pipeline than it saves in the stage it optimizes, so stage-level
timings must always be read together with the displacement statistics.

## Where the per-iteration cost actually goes: reduction traffic

The assembly kernels accumulate into shared matrix/force buffers with
`atomicAdd` on scattered addresses. Nsight Compute on
`compute_dihedral_bending_AOGS` (one call, 388 us, ~88k bend elements) shows
what that costs:

| Metric | Value |
|---|---|
| L1 load sectors | 1.18 M |
| L1 store sectors | 0 |
| **L1 reduction sectors** (`atomicAdd` compiles to `red.global`) | **8.54 M** |
| Instructions executed | 5.14 M |
| DRAM | 23.6 MB |

**88% of the L1 sector traffic of the largest assembly kernel is reduction
traffic** (ncu reports it under `l1tex__t_sectors_pipe_lsu_mem_global_op_red`,
not `..._op_atom`, which is why an earlier probe looked like there were no
atomics at all). Per element the kernel issues 4 `float3` and 10 `Mat3`
reductions - 102 float reductions - against roughly 13 load sectors of input.

The same pattern holds for `compute_BW_FEM`, `compute_ee_force`,
`compute_vf_force`, `solve_untangling_kernel` and `A_mul_x_offdiag_kernel`
(the last one does 6 float reductions per matrix row). Those kernels are
35.8% + 12.6% of the frame, so this is the largest single block of work left
in the profile.

This is the measured justification for the gather/CSR item (5.1): replacing
the scatter-with-reduction pattern with per-vertex gather removes those 8.5 M
reduction sectors per call in exchange for redundant element evaluation
(every element is evaluated once per incident vertex instead of once, about
1.6x the arithmetic) and coalesced reads. The arithmetic has room: this
kernel runs at 9% SM throughput.

### Why the reduction traffic is larger than it looks, and what a gather can

Two effects combine:

1. **A 4-byte reduction burns a 32-byte sector.** Each element issues 102
   scalar reductions against scattered addresses, so one element costs ~102
   sectors (3.3 KB) even though the payload is 408 bytes. That is why 88k
   elements produce 310 MB of L1 sector traffic per call.
2. **The gather does not remove all of it.** Per-vertex gather replaces the
   reductions with (a) reads of `q`, `bend_points`, `bend_rest_theta`,
   `bend_factor` and the row table once per *incident vertex* - four times
   more loads than the element-parallel version - and (b) plain stores for
   the force/diagonal (one writer per vertex) and for each row (written by
   the lower-index endpoint only). Estimated per-element traffic falls from
   ~9.7 M sectors to ~7 M, i.e. **a ~30% sector reduction, not 88%**, and the
   element evaluation is repeated up to four times.

So the honest expectation for 5.1 is **1.3-1.6x on the assembly kernels (not
2-3x)**, i.e. roughly 4-6% of the frame for the bending kernel alone and
perhaps 10-15% if the same shape is applied to FEM, contact and the PCG
matvec. The gain comes from removing reduction *latency and serialization*
(each reduction round-trips to L2), not from the bytes.

### Tried: the gather form is slower than the scatter it replaces

The bending gather was implemented as described: a vertex -> (element, slot)
CSR built once from the fixed topology, one thread per cloth vertex, the
element evaluated per incident vertex, the force and diagonal accumulated in
registers and written once per vertex, and each matrix row written by the
endpoint with the smaller vertex index (so every target has exactly one
writer and the kernel contains no atomics at all). `pytest -m quick` passes
and the simulation stays finite and in the same range as the scatter form
(the displacement statistics and candidate counts match).

40 frames x 4 rounds, same binary, same scene:

| Variant | min ms | median ms |
|---|---|---|
| scatter (current) | 28.09 | 31.42 |
| gather | 30.52 | 34.84 |

**The gather is ~10% slower**, which refutes the system-level reading of the
reduction-traffic measurement. Removing 88% of the kernel's sectors does not
pay because of what replaces them:

- the element data is ~50 bytes per element (`bend_points`, rest angle,
  factor, valid flag, the 6 row indices) and the gather reads all of it once
  per incident vertex - four times - from *scattered* addresses, where the
  element-parallel form reads it once, coalesced
  (thread i reads element i);
- the element evaluation (`get_theta_dpk_aogs` plus the seven outer products)
  is repeated up to four times;
- the work per vertex is irregular (2-6 incident elements), so the gather
  also loses the flat one-element-per-thread balance.

Reverted. Combined with 3.1-3.8 this is now a consistent picture: on this
scene the stage costs are dominated by *how much data each unit of work must
touch*, and the existing code is close to the best arrangement for that -
element-parallel, coalesced, with reductions. Ten attempts now span -23% to
+9%; only the broad-phase stream overlap landed.

## Consequences and constraints

- The scan of the broad-phase candidate buffers is a *correctness* issue as
  well as a cost: 27% (vf) and 61% (ef) of the sampled primitives fill all
  16 slots, and the traversal stops when the buffer is full, so further
  candidates are dropped. Any change that reduces traversal work must be
  checked against this, and the eventual fix is a compact pair list rather
  than fixed-size per-primitive slots.
- Any future "static body" optimization must assume the body is animated:
  the reference dataset happens not to move it, but the frontend animates
  bodies. The design that works for both is a per-object BVH built in the
  object's local frame, with the query AABB transformed by that object's
  inverse world matrix at query time (no refit per frame), falling back to a
  subtree refit when the object's own vertices are uploaded (skinning or
  other per-vertex deformation).

## Follow-up rounds (ordered by measured leverage)

| Priority | Work | Evidence | Expected |
|---|---|---|---|
| P0 | Dual-tree / BVH self-traversal emitting a compact pair list | 907 L1 requests and ~6800 instructions per edge query; most node pairs are rejected combinations | broad phase 1.5-3x (45.6% of the frame) |
| P1 | SAH or leaf-batched (leaf_size > 1) tree build | node visits dominate; the tree is a plain Morton LBVH | broad phase 20-40% (unmeasurable with refit-only, see 3.3) |
| P1 | Per-object BVH layering for animated bodies (local-space build + transformed queries) | the body is 45% of the tree leaves and edges, but decimating it 9.4x changed the frame time by only ~5% (see the scaling section), so treat this as unproven | unknown, likely small |
| P2 | Atomic-free assembly for bending / FEM / contact / matvec (35.8% of the frame) | 9 float atomics per matrix entry, 128 registers, 33% occupancy | assembly 1.3-1.6x |
| P2 | Reduce PCG work and kernel count (341 launches/frame) | `linear_iters` 5 -> 1 is x1.23; each iteration launches ~11 kernels | linear solve 1.2-1.5x |

Notes from the round-2 measurements on the two P0/P1 rows that were already
tried: packed node records (4.2) measured a wash, and the rebuild interval is
not the lever (3.3). Row 4.1 remains the only untried item that changes the
visited-node count by construction, and it is a substantial refactor. If it
also lands flat, the broad phase should be considered at its practical floor
for this scene, and effort should move to the P2 rows (35.8% + 12.6% of the
frame) and to reducing the work per unit of simulated time.

Expected combined effect of the P0/P1 items is 1.6-2.2x (RTS 0.11 -> 0.18-0.24).
Reaching RTS 1.0 at this configuration needs ~9x, which cannot come from
micro-optimization alone: it requires reducing the work per unit of simulated
time (fewer iterations with better convergence, cheaper contact queries, and
probably a coarser simulation mesh with render-mesh interpolation).

## Cost scaling with garment size (and the body-mesh hypothesis, refuted)

The study-1 configuration is one point; the dataset has 3,450 elements from
under 5k to over 100k triangles. Running the same parameter overlay on
elements of different sizes (40 frames each, same binary) gives:

| element | cloth tri | body tri | ms/frame | RTS |
|---|---|---|---|---|
| ELXRPKNR5Z | 3,997 | 47,500 | 10.53 | 0.285 |
| P9CM7RF9M5 | 16,256 | 47,500 | 16.37 | 0.183 |
| JAO8WE8XSM | 33,082 | 47,500 | 22.41 | 0.134 |
| 00YONAPXZE (study-1) | 57,130 | 47,500 | 29.99 | 0.100 |

The first four points fit `frame_ms ~= 9.1 + 0.366 us * cloth_triangles`, i.e.
a size-independent floor plus a cloth-proportional term. Nsight Systems on
the *smallest* element (4k cloth triangles, 2,238 vertices - 574 kernels per
frame, 9.28 ms of GPU time per frame) shows where the floor comes from:

| Stage | ms/frame | share |
|---|---|---|
| **Broad phase (`query_vf` + `query_ee` + `query_ef`)** | **6.28** | **67.6%** |
| untangling | 0.57 | 6.1% |
| bending | 0.44 | 4.7% |
| PCG (matvec, reductions, helpers) | ~0.87 | 9.4% |
| contact forces (vf + ee) | 0.51 | 5.5% |
| FEM | 0.16 | 1.8% |

`query_vf` alone costs 3.63 ms on that 4k-triangle garment, essentially the
same as the 3.68 ms it costs on the 57k-triangle study-1 garment. The reason
is that the query set is the whole scene, not the garment: every BVH query
runs for all vertices and all edges, and the neutral body (23,752 vertices,
71,250 edges, 47,500 triangles) is the same mesh in every element. For a
small garment the body *is* the scene.

The obvious reading - "the body sets the floor, so decimate the body" - was
then tested directly and **refuted**. Substituting a grid-clustered body proxy
(cell size 10 / 20 / 30 mm) into the same scene, same parameters, 60 frames:

| body cell | body triangles | ms/frame | RTS | mean cloth deviation | cloth verts < 3 mm from the body |
|---|---|---|---|---|---|
| original | 47,500 | 30.94 | 0.097 | - | 214 |
| 10 mm | 26,946 | 29.48 | 0.102 | 10 mm | 235 |
| 20 mm | 10,668 | 30.38 | 0.099 | 8 mm | 241 |
| 30 mm | 5,082 | 29.32 | 0.102 | 6 mm | 279 |

Cutting the body by **9.4x** changes the frame time by ~5%, inside the run-to-run
noise of this scene. So the broad phase is *not* driven by primitive counts.
The consistent explanation with the rest of this document is that it is bound
by the traversal a query must perform at the geometric scale of its own query
volume (the swept capsule), with a small number of waves: `query_vf` costs
3.63 ms on the 4k-cloth element and 3.68 ms on the 57k-cloth study-1 garment
*despite half the queries*, because the smaller scene launches half the blocks
(102 vs 210) and the kernel already runs at 1.31 waves and 29.6% achieved
occupancy. A coarser body has fewer triangles but larger ones, so the tree is
shallower and the per-query leaf overlap grows: the two effects cancel.

Corollary: **mesh LOD is not a lever for the broad phase.** The garment-size
trend in the table above is real across *different garments*, but it comes
from the shape and contact complexity of those garments, not from a
primitive-count cost that decimation can recover.

## References

## Why the fast implementations are fast, and what is transferable

Reference sources available on the development machine: Warp's own BVH
(`warp/native/bvh.h`, `bvh.cu`) and the Style3D solver port in the Newton
physics engine (`newton/_src/solvers/style3d/collision`), from which this
project's contact code is derived. Four design differences matter, and each
one lines up with a measurement from this document.

1. **Tight, non-swept broad-phase bounds.** Newton/Style3D's broad phase
   queries AABBs inflated only by `query_radius`
   (`bvh/kernels.py: aabb_vs_aabb_kernel`, `triangle_vs_point_kernel`,
   `edge_vs_edge_kernel`) against trees built from the *current* positions.
   They rely on small substeps to stay conservative. This codebase instead
   sweeps both the tree (`refit_*_offset_bvh_kernel` takes the union of
   `pos_prev`/`pos_target`) and the query (trajectory capsules), which
   measured median 4.1 mm / p99 118 mm of extra extent against a ~10 mm
   feature size. The cost is entirely explained by that: the *same*
   `query_ee_pairs_capsule_kernel` takes 1.07 ms on the cheapest frame and
   16.25 ms on the dearest frame of one 35-frame run, and 6.30 ms in the
   PDNewton configuration (43 mm/frame motion) versus 2.03 ms in the VBD
   configuration (15 mm/frame).
2. **Cheaper node representation.** Warp uses 16-byte packed node halves
   (`BVHPackedNodeHalf`), `USE_LOAD4` (128-bit loads), a SAH build
   (`SAH_NUM_BUCKETS 16`), a **32-entry stack in shared memory**
   (`BVH_SHARED_STACK 1`, strided by the block dimension), and an
   experimental block-cooperative "tile BVH query". This codebase uses an
   8-byte child array plus a 24-byte AABB array (two loads per visited node),
   a 64-entry stack in **local** memory, a Morton LBVH, and one thread per
   query - measured 1.31 waves and 29.6% achieved occupancy for `query_vf`.
3. **Local solvers instead of a global solve.** VBD/Style3D solve per-vertex
   local systems; there is no global linear solve. PDNewton runs 10
   iterations x 5 PCG iterations = 50 matvecs, 110 block reductions and 341
   kernel launches per frame.
4. **Stateful contact.** VBD's `*_stated_kernel` keeps the previous frame's
   candidate list (`kept_count`), filters it by distance and warm-starts the
   contact state (`ContactState` holds the contact multipliers). That is what
   makes a contact solve with tens of iterations affordable.

### Measured: the in-repo local solver is the fastest configuration

Same scene (study-1 element), each solver with its own preset, 40 frames:

| solver | substeps per frame | ms/frame | RTS | max motion per frame |
|---|---|---|---|---|
| PDNewton (current default) | 1 | 21.8 | 0.138 | 43 mm |
| **VBD** | 1 | **18.6** | **0.162** | 15 mm |
| XPBD | 3 | 64.3 | 0.047 | 40 mm |
| Explicit | 12 | 421.7 | 0.007 | 270 mm |

VBD iteration sweep (same scene, VBD preset):

| vbd_iters | ms/frame | RTS | max motion per frame |
|---|---|---|---|
| 1 | 8.7 | 0.346 | 7.6 mm |
| 2 | 11.2 | 0.267 | 8.4 mm |
| 3 | 13.7 | 0.220 | 13.9 mm |
| 5 | 18.9 | 0.159 | 21.1 mm |
| 10 | 33.5 | 0.090 | 29.9 mm |

So the fastest measured configuration for this scene is the local solver with
1-3 iterations: **2.5x / 1.6x the current PDNewton default**, with calmer
motion (7-14 mm per frame versus 43 mm). This is a solver and parameter
choice, not a kernel change, and it is what the fast production paths do.

Caveats before adopting it: VBD is marked experimental in this repository
(its smoke tests are `xfail`), its preset is not the study-1 parameter
overlay (so the study-1 configuration is not directly portable), and drape
fidelity against the current baseline has not been checked - it needs a
visual/quality comparison before being used as an interactive default.

### The visual check: VBD's speed comes from a missing constraint

Both configurations were run for 120 frames on the same scene and compared
frame by frame (`solver_compare.py`, snapshots at frames 0/40/80/120 plus the
invariant metrics):

| config | ms/frame | RTS | cloth area / rest | max seam gap | closest body approach | mean motion/frame |
|---|---|---|---|---|---|---|
| PDNewton (study-1 overlay) | 24.8 | 0.121 | 1.134 | **0.85 mm** | 0.31 mm | 43 mm |
| VBD, 3 iterations | 14.8 | 0.202 | 1.018 | **640 mm** | 0.24 mm | 8.7 mm |
| VBD, 1 iteration | 9.0 | 0.334 | 1.011 | **617 mm** | 0.36 mm | 4.1 mm |

The rendered comparison shows why: under VBD the garment never assembles -
the pattern panels stay separate and float around the body instead of closing
into a sewn garment, which is exactly the 0.6 m seam gap in the table.

So the VBD numbers are **not** a like-for-like speed comparison. Its calm
trajectory (4-9 mm per frame against PDNewton's 43 mm) is a consequence of
the missing seam constraint, and the collision pipeline is cheaper for the
same reason. The sewing path exists in the PDNewton route
(`accumulate_sewing_force` + `project_stitches`, ~0.6 ms/frame) but is not
driving the VBD route.

This closes the "just switch to the local solver" idea, and it identifies the
real piece of work behind it: VBD is ~2x cheaper per frame *with the same
collision pipeline*, so **making the seam/stitch constraint work in the VBD
route is the one change that could combine that speed with a correctly
assembled garment**. That is a solver feature, not a micro-optimization, and
it is the reason VBD is marked experimental today.

### Tried: giving VBD the stitch constraint (it closes the seams, then needs tuning)

The gap was concrete: `solver_VBD.cu` never referenced the sewing path at all,
so `accumulate_sewing_force` and `project_stitches` - both of which only need
positions, masses and the per-vertex Hessian that VBD already maintains - were
simply not called. Adding the projection alone changed nothing, which
diagnosed the mechanism: the projection is *gated* on `check_sewing` having
declared a seam closed, and closures only happen once the stitch spring has
pulled the panels close enough. Adding the spring (`accumulate_sewing_force`,
behind `vbd_project_stitches`) makes the seam gap collapse:

| VBD configuration | ms/frame | max seam gap | cloth area / rest | mean motion/frame |
|---|---|---|---|---|
| current (no seam path), 3 iterations | 14.6 | 605 mm | 1.03 | 9 mm |
| current (no seam path), 1 iteration | 8.8 | 588 mm | 1.03 | 8 mm |
| **with the stitch spring, 3 iterations** | **31.4** | **1.2 mm** | **1.29** | **92 mm** |
| with the stitch spring, 1 iteration | 32.7 | 265 mm | 1.46 | 229 mm |

So the constraint mechanism works - the garment assembles - but the current
VBD parameter block cannot absorb a 1e5 stitch stiffness with a handful of
local iterations: the cloth overshoots (29-46% area stretch, 92-229 mm of
motion per frame) and the frame cost more than doubles, because the violent
trajectory feeds straight back into the swept broad phase. The switch is
therefore not kept in the tree - it was a diagnostic, not an optimization, and
**it was removed again when the branch was cleaned up for merge**. The finding
stays: making VBD a usable fast path is not "add the call", it is
"add the constraint and re-tune the solver" (softer stitch stiffness, more
local iterations, or per-cluster projection consistent with VBD's
per-vertex solve).

### Tuning it: the local-solver path does not become competitive

The stitch stiffness and the iteration count were then swept with the seam
path enabled (120 frames, study-1 scene, VBD preset):

| sewing_k | vbd_iters | ms/frame | RTS | max seam gap | cloth area / rest | mean motion/frame |
|---|---|---|---|---|---|---|
| 1e2 | 3 | 24.1 | 0.125 | 21.3 mm | 1.124 | 14 mm |
| 1e3 | 3 | 24.9 | 0.121 | 8.9 mm | 1.168 | 17 mm |
| 1e4 | 3 | 22.1 | 0.136 | 3.8 mm | 1.344 | 22 mm |
| 1e5 | 3 | 31.9 | 0.094 | 0.64 mm | 1.298 | 92 mm |
| 1e3 | 10 | 59.8 | 0.050 | 14.2 mm | 1.168 | 28 mm |
| 1e3 | 30 | 212.4 | 0.014 | **19 m** | **24.9** | **3.7 m** (diverges) |

For reference the current default on the same scene is PDNewton at 24.8 ms
(RTS 0.121), cloth area 1.134, 0.85 mm maximum seam gap.

Reading:

- **No configuration beats the default meaningfully.** The best is 1e4 with 3
  iterations at 22.1 ms, ~1.1x, and it pays with 34% area stretch and a 4x
  larger seam gap.
- The settings that actually close the seams (1e5) are slower than the default
  and violent; the settings with sane stretch (1e2/1e3) leave 9-21 mm gaps.
- More local iterations do not help: they are slower *and* no more accurate
  (10 iterations: 59.8 ms, 14.2 mm gap), and 30 iterations diverge outright.

So "the local solver is the fast path" is closed by measurement: with the
constraint that makes the garment real, this VBD implementation is not
competitive with the current projective-dynamics route.

### Tried: the tight, non-swept broad phase from the reference implementations

`tight_broad_phase` (off by default) refits the tree and builds the query
boxes from the current position only, instead of the union of
`pos_step_prev`/`pos_pred`, which is the arrangement Warp and the Style3D port
use. It is only conservative when the per-substep motion stays inside the
contact radius, so it was measured together with the substep size (60 frames,
study-1 scene):

| mode | step_h | substeps | ms/frame | RTS | max seam gap | cloth area | closest body approach |
|---|---|---|---|---|---|---|---|
| swept (current) | 3.0 ms | 1 | 26.86 | 0.112 | 1.32 mm | 1.230 | 0.73 mm |
| **tight** | 3.0 ms | 1 | **21.11** | **0.142** | 0.84 mm | 1.264 | 0.87 mm |
| swept | 1.5 ms | 2 | 54.11 | 0.055 | 0.15 mm | 1.132 | 0.46 mm |
| tight | 1.5 ms | 2 | 49.37 | 0.061 | 0.21 mm | 1.173 | 0.56 mm |
| swept | 1.0 ms | 3 | 84.07 | 0.036 | 0.01 mm | 1.156 | 0.52 mm |
| tight | 1.0 ms | 3 | 78.77 | 0.038 | 0.03 mm | 1.187 | 0.98 mm |

Two things stand out:

- **Tight wins at every substep size** (x1.27 at the configured 3 ms step,
  x1.10 at 1.5 ms, x1.07 at 1 ms) - and the advantage shrinks as the sweep
  shrinks, which is exactly the mechanism: less sweep, less to gain.
- In this test the tight boxes do not cost quality: at 3 ms they give a
  *smaller* maximum seam gap (0.84 vs 1.32 mm) and more clearance from the
  body (0.87 vs 0.73 mm) than the conservative version, and at the smaller
  substeps the two are within noise of each other.

So this is the best remaining lever found in this change: **x1.27 at the
configured step size, with the invariants intact**. It stays behind the
parameter (default off) because the invariants are coarse and 60 frames is
short: the failure mode it risks is a *missed* contact during a fast frame,
which these metrics would not necessarily catch. Before enabling it by
default it needs (a) a long run (300+ frames) with the same invariants,
(b) the frame-to-frame trajectory compared against the swept baseline, and
(c) a visual check for interpenetration.

The first of those checks was run and it produced a methodological finding of
its own: over 300 frames the tight and swept runs diverge by a mean of
32.1 mm per cloth vertex (max 106 mm), but **two runs of the *same* swept
configuration diverge by a mean of 46.8 mm (max 141 mm)**. The engine's own
run-to-run non-determinism (already documented for PDNewton) is larger than
the difference being tested, so a trajectory comparison cannot validate this
change. Validation has to rest on the invariants (which are unchanged in the
tests run so far) and on a visual interpenetration check, not on matching
trajectories.

### Acceptance check at 300 frames

The state after a long run was compared directly, plus a rendered side-by-side
of the final frame with the body drawn semi-transparent:

| mode | cloth area / rest | max seam gap | min distance to body | 1st percentile | vertices < 1 mm | vertices < 3 mm |
|---|---|---|---|---|---|---|
| swept (current) | 1.097 | 2.30 mm | 0.625 mm | 1.754 mm | 3 | 1655 |
| **tight** | 1.179 | **0.00 mm** | 0.712 mm | 1.741 mm | 5 | 1700 |
| two swept runs (noise floor) | - | - | - | - | - | - |

- **No penetration regression**: the minimum distance, the 1st percentile and
  the counts of near-body vertices are the same within a few percent (5 vs 3
  vertices within 1 mm out of ~30k). The failure mode this change risks -
  a missed contact in a fast frame - does not show up.
- Seams are *better* under tight (0.00 mm vs 2.30 mm), and the render shows
  the same assembled garment in both cases.
- The one visible cost is stiffness: the cloth stretches ~8% more (area 1.179
  against 1.097), consistent with the shorter candidate lists.
- Meanwhile it is **x1.27 faster at the configured step size** (21.11 vs
  26.86 ms/frame, RTS 0.142 vs 0.112).

That is a favourable trade for a real-time target, so the recommendation is
to enable `tight_broad_phase` by default - a one-line default change in
`SolverPDNewton::step`, which is left for the maintainer because it changes
simulation results (note the `sewing`-like convention in this repository that
validated parameter blocks are not changed without re-validation).

### Enabled: final numbers for the two retained changes

`tight_broad_phase` now defaults to on; `tight_broad_phase=0` restores the
conservative-CCD behavior. Interleaved A/B in the same binary, 40 frames x 3
rounds, study-1 scene:

| variant | min ms | median ms | RTS |
|---|---|---|---|
| **tight broad phase (new default, streams on)** | **20.95** | **22.02** | **0.143** |
| swept broad phase (previous default, streams on) | 30.39 | 32.57 | 0.099 |
| swept broad phase + sequential queries (original behavior) | 31.96 | 36.08 | 0.094 |

So the two retained changes together give **x1.72 against the original
behavior** (36.08 -> 20.95 ms) and **x1.45 against the state before this
last change** (30.39 -> 20.95 ms), i.e. RTS 0.094 -> 0.143 on this scene.
Everything else in this document is either rejected or left as an opt-in
parameter.

### The new default across the dataset

Because the default changed, it was validated on five dataset elements
spanning 2.2k to 32k cloth vertices (150 frames each, study-1 parameter
overlay, both modes):

| element | cloth verts | mode | ms/frame | RTS | cloth area | max seam gap | min distance to body | all finite |
|---|---|---|---|---|---|---|---|---|
| 00YONAPXZE (study-1) | 29,784 | swept | 31.53 | 0.095 | 1.126 | 0.11 mm | 0.827 mm | yes |
| | | **tight** | 27.31 | 0.110 | 1.159 | 0.11 mm | 0.541 mm | yes |
| 6YGLO1BHYF | 32,202 | swept | 82.79 | 0.036 | 1.008 | 0.22 mm | 0.996 mm | yes |
| | | **tight** | **27.22** | **0.110** | 1.058 | 0.12 mm | 1.012 mm | yes |
| ELXRPKNR5Z | 2,238 | swept | 11.10 | 0.270 | 1.078 | 0.00 mm | 1.030 mm | yes |
| | | tight | 9.90 | 0.303 | 1.080 | 0.00 mm | 1.050 mm | yes |
| JAO8WE8XSM | 17,344 | swept | 22.61 | 0.133 | 1.118 | 0.09 mm | 1.005 mm | yes |
| | | tight | 19.64 | 0.153 | 1.129 | 0.10 mm | 0.998 mm | yes |
| P9CM7RF9M5 | 8,677 | swept | 15.94 | 0.188 | 1.101 | 0.11 mm | 0.926 mm | yes |
| | | tight | 14.12 | 0.213 | 1.107 | 0.06 mm | 0.912 mm | yes |

- Tight is faster on **every** element (x1.05 to x3.04; the pathological
  32k-vertex element, which was the outlier in the scaling section, benefits
  most: 82.79 -> 27.22 ms).
- No invariant degrades: area stays within 1.01-1.16 for both modes, seams are
  closed in both (<= 0.22 mm) and the minimum cloth-body distance stays
  positive everywhere (0.54-1.05 mm). The counts of near-body vertices are
  tiny in both modes (0-9 out of 2.2k-32k vertices).
- These runs are sequential rather than interleaved, so the *magnitude* is
  thermal-state dependent (the interleaved measurement for the study-1
  element gave x1.45, this table gives x1.15); the direction is consistent
  everywhere.

### With an animated body

The dataset's body never moves, which is the one case the tight broad phase
could get wrong: the tight query covers only the current pose, so a body that
travels further than the contact radius in one substep could pass through the
cloth. The body was therefore yawed +-25 degrees at 0.5 Hz by uploading
rotated local vertices every frame (the public animation path), 120 frames:

| mode | ms/frame | RTS | cloth area | max seam gap | min distance to body | vertices < 1 mm | vertices < 3 mm |
|---|---|---|---|---|---|---|---|
| swept | 34.20 | 0.088 | 1.147 | 0.29 mm | 0.345 mm | 11 | 283 |
| **tight** | **22.96** | **0.131** | 1.169 | 0.16 mm | **0.127 mm** | 7 | 322 |

- The speed advantage holds with a moving body (**x1.49**), and the simulation
  stays finite with seams closed and comparable stretch.
- The clearance to the body narrows from 0.345 mm to 0.127 mm and the count of
  near-body vertices rises slightly (283 -> 322 out of ~30k). There is no
  interpenetration in this test, but the margin is thinner - which is the
  expected signature of a non-swept query against a moving obstacle.

Conclusion: the default is validated for moderate body animation, and the
remaining risk is bounded and documented - a fast-moving body (or a longer
substep) should use `tight_broad_phase=0` until the body's own motion is
included in the query, which is a contained follow-up (inflate the query box
by the object's per-substep translation) rather than a redesign.

- The collision code follows the Style3D solver that is now part of the
  Newton physics engine (`newton/_src/solvers/style3d/collision`). That
  implementation uses Warp's built-in BVH with 16-byte packed nodes and
  non-swept query AABBs, which is a useful reference for the P0/P1 items
  above.
- `lbvh` follows the `ppf-contact-solver` variant noted in the repository
  README.

## Reproducing the measurement

## The one lever with measured headroom: iteration count vs drape

Everything else in this document is a kernel-level change that landed between
-23% and +9%. The remaining lever is how much solver work each frame does.
`pd_iters` was swept on the study-1 scene (120 frames, frames 20-120 timed,
one process per level so each starts from the same state):

| pd_iters | ms/frame | RTS | cloth area / rest area | seam gap avg / max | closest cloth-body approach | mean cloth deviation vs 10 iters |
|---|---|---|---|---|---|---|
| 1 | 14.5 | 0.207 | 1.193 | 0.02 / 0.89 mm | 0.10 mm | 19.8 mm |
| 2 | 12.2 | 0.245 | 1.181 | 0.01 / 0.27 mm | 0.55 mm | 27.8 mm |
| 3 | 16.3 | 0.184 | 1.149 | 0.01 / 0.16 mm | 0.69 mm | 11.9 mm |
| 5 | 24.0 | 0.125 | 1.146 | 0.00 / 0.13 mm | 0.54 mm | 8.9 mm |
| 7 | 29.5 | 0.102 | 1.141 | 0.01 / 0.13 mm | 0.54 mm | 7.5 mm |
| **10 (current)** | **31.8** | **0.094** | 1.135 | 0.01 / 0.15 mm | 0.39 mm | 0 |
| 15 | 48.5 | 0.062 | 1.110 | 0.00 / 0.14 mm | 0.48 mm | 10.7 mm |

Reading it:

- **10 -> 5 iterations is x1.33** (RTS 0.094 -> 0.125) for a ~9 mm mean
  change in the drape, with the enforced invariants untouched: seams stay
  closed (0.13 mm max) and the closest approach to the body is unchanged.
- **10 -> 3 is x1.95** (RTS 0.184) for a ~12 mm change, still with intact
  seams and contact distance.
- Below 3 the shape drifts much further (12 -> 28 mm) and the minimum
  cloth-body distance collapses (0.69 -> 0.10 mm at 1 iteration), i.e. the
  garment starts sinking into the body. 1 iteration is also *slower* than 2,
  so it is not even a Pareto point.
- Stretch is high at every level (area 1.11-1.19x the rest area), so the
  scene's parameters, not the iteration count, set the baseline stretch.
- 15 iterations deviates 10.7 mm from 10 iterations, i.e. **none of these
  settings is a converged solution**; the deviation columns measure change
  from the current default, not error against a ground truth.

This is the decision input for the RTS target: the fastest configuration that
keeps the pipeline's invariants intact is ~3 iterations, which buys about
2x (RTS 0.094 -> ~0.18) while changing the drape by roughly a centimetre.
It is a physics trade-off, not a free optimization, so it is left as a
parameter choice (`pd_iters` is already user-settable) rather than a code
change.

Builds must not disturb an in-use module output: configure a separate CMake
build tree, build the `Qianyi_DP` target in Release, and point the test
harness at the artifact through `QYDP_PYD`. Scene data comes from the
GarmentCodeData environment variables documented in the repository; the
study-1 parameter overlay is applied with `set_parameters` before
`input_data`, exactly as the interactive window does.

Timing must use `update(dt)` plus a synchronizing read, must interleave the
variants being compared, and must report min and median over at least three
rounds.

## Implemented: a reduction-free linear solver, and what it cost

`SolverChebyshev` was added next to `SolverJacobi` (`linear_solver_type=2`). It
generates its coefficient sequence on the device from a lower bound
`a = 1/h^2` and a row-sum upper bound of `M^-1 A`, so the iteration has **no
dot products, no scalar readback and no synchronisation** - the property that
made it worth trying (PCG spends 110 block reductions per frame).

Interleaved A/B in the same binary (40 frames x 3 rounds, study-1 scene):

| linear solver | min ms | median ms | RTS |
|---|---|---|---|
| PCG (current) | 19.28 | 20.33 | 0.156 |
| **Chebyshev (new)** | **15.69** | **16.19** | **0.191** |
| Jacobi (already in the repository, unused) | 16.39 | 16.88 | 0.183 |

Both reduction-free iterations beat PCG (x1.23 and x1.18). Over 150 frames they
differ in quality though:

| linear solver | max seam gap at 150 frames | cloth area | closest body approach |
|---|---|---|---|
| PCG | 0.08 mm | 1.155 | 0.61 mm |
| Chebyshev | **350 mm** | 1.133 | 0.71 mm |
| Jacobi | 0.07 mm | 1.172 | 0.83 mm |

Chebyshev at the default 5 iterations is **not converged enough**: its
extrapolation term, driven by a deliberately loose spectral bound, overshoots
and the garment's seams tear apart over a long run. It needs more iterations or
a tighter bound before it is usable, so **the solver and its kernels were
removed again when the branch was cleaned up for merge** - keeping an unusable
opt-in in the tree buys nothing that this section does not record.

**Jacobi is the immediately usable finding**: reduction-free, x1.18-1.20 faster
than PCG, with invariants identical to PCG's over 150 frames. Adopting it is a
one-value preset change (`linear_solver_type=1`) and needs the same validation
protocol `tight_broad_phase` went through before it becomes the default.

### Jacobi passed the full validation protocol

Five dataset elements (150 frames each, `linear_solver_type=1`, tight broad
phase) against the PCG numbers recorded earlier:

| element | cloth verts | PCG ms/frame | **Jacobi ms/frame** | seam gap (Jacobi) | area (Jacobi) | min body distance |
|---|---|---|---|---|---|---|
| 6YGLO1BHYF | 32,202 | 27.22 | 25.43 | 0.08 mm | 1.073 | 1.009 mm |
| 00YONAPXZE | 29,784 | 27.31 | 24.54 | 0.09 mm | 1.172 | 0.743 mm |
| JAO8WE8XSM | 17,344 | 19.64 | 14.88 | 0.09 mm | 1.205 | 0.993 mm |
| P9CM7RF9M5 | 8,677 | 14.12 | 10.31 | 0.03 mm | 1.149 | 0.956 mm |

And with the animated body (yaw +-25 degrees at 0.5 Hz, 120 frames):

| linear solver | ms/frame | seam gap | area | min body distance | vertices < 1 mm |
|---|---|---|---|---|---|
| PCG | 22.96 | 0.16 mm | 1.169 | 0.127 mm | 7 |
| Jacobi | 23.29 | 0.14 mm | 1.193 | 0.507 mm | 16 |

Every element is faster, seams stay closed (<= 0.09 mm), stretch is in the same
band and the cloth-body clearance stays positive - Jacobi is slightly softer
(areas up to 1.205 against 1.129) because it converges more slowly than PCG per
iteration, which is the trade for removing the reductions.

One deployment note: the value lives in the caller's parameter block, not in
the solver code - the study-1 notebook and `tests/harness/presets.py` both set
`linear_solver_type=0` explicitly, so enabling Jacobi is a one-value change in
those files (the harness preset is the repository's validated baseline block
and is left untouched here).

## Tried: aligning the bounds records for 128-bit loads (reverted)

The packed-record experiment above merged the two arrays but kept the record
at 24 bytes, so `ptxas` still emitted scalar loads (the disassembly of
`query_ee_pairs_capsule_kernel` shows `LDG.E.CONSTANT` x150, `LDG.E.128` x0).
This retry padded the record to 32 bytes with `alignas(16)` so `min` and `max`
sit at offsets 0 and 16, and forced the fetch with an explicit `float4`
reinterpretation:

```cpp
struct alignas(16) AABB3D { float3 min; float pad_min; float3 max; float pad_max; };
// per visit, instead of six scalar loads:
float4 lo = __ldg(reinterpret_cast<const float4*>(aabbs) + 2u * idx);
float4 hi = __ldg(reinterpret_cast<const float4*>(aabbs) + 2u * idx + 1);
```

The padding alone changed nothing at all (identical SASS, identical register
count): the compiler does not infer the alignment of a `float3` member through
a `const __restrict__` pointer parameter. Explicit vector loads are required
to get the pair into two 128-bit fetches.

Interleaved A/B against the unmodified module in the same toolchain
(study-1 scene, 40 frames x 4 rounds, Jacobi):

| Variant | min ms | median ms | RTS |
|---|---|---|---|
| split, scalar loads | 15.78 | 16.30 | 0.190 |
| 32-byte aligned, 2x LDG.128 | 15.84 | 16.06 | 0.189 |

A wash (x1.004 on the minimum, and the median moves the other way). This is
the third independent confirmation that the per-visit critical path is a
single L1 round trip and that neither the byte count nor the number of load
instructions in a visit is on it. Reverted.

## Reference implementation, kernel level

Warp's BVH and the Newton Style3D port - the public references this project's
collision code follows - keep the traversal's walk state in registers plus a
small shared stack, and their narrow phase reads a compact pair list instead
of re-walking the tree per candidate. Our kernels carry a 64-entry
local-memory stack (`STACK:256..288`, measured earlier in this document),
which is where the occupancy gap comes from.

The three record-layout experiments (split vs packed vs 32-byte aligned) all
measured a wash, so the remaining lever is removing the stack (section 13 of
`tasks.md`) and, separately, a multilevel domain-decomposition preconditioner
next to the PCG path.

## Superseded: the Jacobi linear solver (reverted to PCG at 2 iterations)

`linear_solver_type=1` was adopted as a speed change. Reading the code again
shows what it is: `SolverJacobi::solve` runs a fixed `linear_iters` loop of
`{Ax, r = b - Ax, x += M^-1 r}`, and its convergence check (`iter % 10 == 0 &&
iter > 0`) never fires at the 5 iterations the caller asks for. Inside 10 PD
outer iterations it is therefore a *smoother* with zero reductions and zero
host synchronizations, against PCG's two `vector_field_dot_sync` round trips
per iteration. That is why it was faster.

What the first measurement missed is that the same speed is available without
giving up PCG's convergence. Interleaved in one binary, one session, 40 frames
x 3 rounds:

| variant | min ms | median ms | RTS(min) |
|---|---|---|---|
| PCG, 5 iterations | 26.66 | 29.19 | 0.113 |
| PCG, 3 iterations | 25.19 | 26.26 | 0.119 |
| **PCG, 2 iterations** | **22.77** | **23.78** | **0.132** |
| Jacobi, 5 iterations | 23.47 | 23.84 | 0.128 |

and across five dataset elements (150 frames each, tight broad phase):

| element | cloth verts | PCG x5 | **PCG x2** | Jacobi x5 | area x5 / **x2** / Jacobi |
|---|---|---|---|---|---|
| 00YONAPXZE | 29,784 | 27.31 | **24.80** | 24.54 | 1.159 / **1.162** / 1.172 |
| 6YGLO1BHYF | 32,202 | 27.22 | **25.43** | 25.43 | 1.058 / **1.070** / 1.073 |
| JAO8WE8XSM | 17,344 | 19.64 | **16.28** | 14.88 | 1.129 / **1.132** / 1.205 |
| P9CM7RF9M5 | 8,677 | 14.12 | **11.50** | 10.31 | 1.107 / **1.109** / 1.149 |
| ELXRPKNR5Z | 2,238 | 9.90 | **7.58** | - | 1.080 / **1.085** / - |

PCG at 2 iterations is 1.10-1.31x faster than the validated PCG baseline *and*
keeps its drape (area within 0.005, seams below 0.2 mm, cloth-body clearance
positive); Jacobi is at best a few percent faster on the mid-size elements and
visibly stretchier. **The preset now uses `linear_solver_type=0` with
`linear_iters=2`.**

Adding Jacobi iterations does not buy the quality back - undamped Jacobi
diverges as the iteration matrix leaves the unit ball. Same scene, 120 frames,
120x:

| Jacobi iterations | ms/frame | max displacement per frame | area |
|---|---|---|---|
| 5 | 24.9 | 0.027 m | 1.192 |
| 10 | 29.9 | 0.067 m | 1.274 |
| 20 | 69.2 | 1.36 m (diverging) | - |

## The next measured bottleneck: the bending Jacobian assembly

Nsight Systems with the PCG-at-2 configuration attributes the frame as
follows (16.2 ms/frame of GPU work, 25 frames):

| kernel | ms/frame | share | calls/frame |
|---|---|---|---|
| `compute_dihedral_bending_AOGS` | 4.00 | 24.7% | 10 |
| `query_ee_pairs_capsule_kernel` | 2.45 | 15.2% | 1 |
| `compute_BW_FEM` | 1.90 | 11.8% | 10 |
| `query_ef_pairs_kernel` | 1.41 | 8.7% | 1 |
| `A_mul_x_offdiag_kernel` | 1.40 | 8.7% | 30 |
| `solve_untangling_kernel` | 1.37 | 8.5% | 10 |
| `query_vf_pairs_capsule_kernel` | 0.87 | 5.4% | 1 |

The two element-assembly kernels are 36% of the frame and both are called once
per PD iteration. Nsight Compute on the bending kernel:

| metric | value |
|---|---|
| registers / thread | 128 (occupancy limit: 2 blocks/SM) |
| achieved occupancy | 29.3% |
| issue active | 5.09% |
| global load sectors | 1.18 M |
| **global reduction sectors** | **8.54 M** |
| L1 sector hit rate | 9.7% |
| sectors per reduction request | 30.5 (of 32 lanes) |

It is a scattered atomic reduction with almost no L1 reuse: 14 `atomicAdd`
calls per element (10 `Mat3` blocks + 4 `float3` forces) expand to ~102 scalar
atomics per thread, and 8.5 M reduction sectors against 1.2 M load sectors.
The scatter is *inherently* one sector per lane - each lane writes a different
36-byte block, so reordering the elements cannot coalesce it (32 consecutive
vertices are 36 bytes apart, i.e. still one sector per block). The available
levers are therefore fewer or smaller entries:

1. the four diagonal blocks are symmetric (6 unique floats, not 9): ~12% fewer
   atomics;
2. half-precision blocks, which only pay off for a narrow-band operator (see
   the analysis below): a 3x3 block in
   `half2` atomics is 5 stores instead of 9, and halves the assembly footprint
   (the four Jacobian arrays total ~4.5 MB against 4 MB of L2, so they thrash);
3. raising occupancy by shortening the kernel's live range - the 7 cached
   outer products are 63 registers.

Forcing (3) with `__launch_bounds__(256, 4)` was tried and **rejected**: ptxas
cut the kernel to 64 registers and spilled 296 bytes to the stack, and the
interleaved A/B measured 18.87 -> 20.12 ms min (x1.066 *slower*). Occupancy
only pays if the register cut does not spill.

Its companion experiment - computing the seven weighted outer products inside
each block instead of materialising them once (63 live registers) - was also
run and **rejected**: ptxas went from 128 to 121 registers (still two blocks
per SM) and the interleaved A/B was x1.014, i.e. a wash. The register pressure
is not where the register-allocator wants it to be.

## Iteration count is still the largest lever (and the caller owns it)

Re-measured on the current configuration (tight broad phase, PCG x2, study-1
scene, 120 frames, one process per level):

| pd_iters | ms/frame | **RTS** | area | seam max | body min | deviation vs 10 |
|---|---|---|---|---|---|---|
| 3 | 9.09 | **0.330** | 1.177 | 0.17 mm | 0.58 mm | 12.2 mm |
| 5 | 12.84 | **0.234** | 1.172 | 0.17 mm | 0.46 mm | 8.8 mm |
| 7 | 17.65 | 0.170 | 1.170 | 0.16 mm | 0.88 mm | 6.3 mm |
| 10 | 23.63 | 0.127 | 1.167 | 0.13 mm | 0.78 mm | 0 |

The deviation column is against the 10-iteration run of the *same* build, but
the engine is not run-to-run deterministic (two identical 300-frame runs
already diverge by a mean of 46.8 mm per vertex), so at 120 frames the 3- and
5-iteration drapes are not distinguishable from the noise floor by this
metric. What does separate them is that 5 iterations keep every invariant
(area within 0.005 of the 10-iteration run, seams below 0.2 mm, cloth-body
clearance positive), which is why 5 is the repository's preset value.

The study-1 notebook block now matches the preset (`pd_iters=5`,
`linear_solver_type=0`, `linear_iters=2`); with it the scene measures 13.7 ms
per frame mean / 10.0-10.9 ms minimum, i.e. **RTS 0.219 (mean) and 0.28-0.30
(best frame)** against 0.094 at the start of the round.

## Where the remaining non-kernel time goes

Under Nsight Systems with PCG x2: 391 kernel launches and 16.15 ms of kernel
time per frame, against a 21.86 ms wall span - 26% of the frame is *not*
running kernels. The API trace attributes it as:

| item | per frame |
|---|---|
| Device-to-host copies | 13.4 calls, 0.078 ms of copy time |
| stream synchronizations (type 3) | 7.3 calls, 0.64 ms blocking |
| event synchronizations (type 4) | 1.2 calls, 0.38 ms blocking |

PCG itself is already device-side (`compute_alpha_kernel` / `compute_beta_kernel`
are one-thread kernels over a device scalar); its only host round trip is the
NaN check at the end of each solve, which is a blocking 4-byte D2H copy and
drains the stream. Removing it is a prerequisite for capturing the frame into
a CUDA graph, which is the next block: the graph is worth at most the 26%
gap, but only the part that survives the host-side synchronization that the
vertex readback needs anyway.

### Tried: removing the per-solve blocking copy (no effect, reverted)

The check was made sticky and amortized: `compute_alpha_kernel` and
`compute_beta_kernel` set a device flag when the coefficient they produce is
not finite, and the host reads that flag every eighth solve instead of doing a
blocking copy after every one. That removes ten stream drains per frame at
pd_iters=10 while keeping the safety net (the flag is sticky, so no
non-finite solve can be missed, only reported later).

Two interleaved A/Bs against the unmodified module in the same session
(study-1 at the current configuration, 40 frames per round):

| run | rounds | baseline min | with the change | ratio |
|---|---|---|---|---|
| 1 | 3 | 11.92 ms | 11.67 ms | x1.021 |
| 2 | 4 | 11.60 ms | 12.53 ms | x0.926 |

The two runs disagree in sign, so the effect is inside the noise: **the host
round trips are not on the critical path**, because the GPU is saturated and
the host's blocking time overlaps with work that has to happen anyway. The
change was reverted.

This also re-prices the CUDA graph block: the 26% of the frame that is not
running kernels is mostly launch overhead and the readback the engine needs
regardless, not synchronization that a graph would remove. A graph is now a
smaller and riskier proposition than the gap suggested, and it is *not* the
next thing worth doing.

## The configuration as it stands now, across the dataset

Five elements, 150 frames each, tight broad phase, PCG x2, `pd_iters=5`
(i.e. the preset block, not an experimental overlay):

| element | cloth verts | ms/frame | **RTS** | area | seam max | body min | verts < 1 mm |
|---|---|---|---|---|---|---|---|
| 6YGLO1BHYF | 32,202 | 15.68 | 0.191 | 1.065 | 0.15 mm | 0.997 mm | 1 |
| 00YONAPXZE | 29,784 | 15.16 | 0.198 | 1.160 | 0.10 mm | 0.650 mm | 2 |
| JAO8WE8XSM | 17,344 | 10.76 | 0.279 | 1.132 | 0.11 mm | 1.105 mm | 0 |
| P9CM7RF9M5 | 8,677 | 7.98 | 0.376 | 1.121 | 0.12 mm | 0.978 mm | 1 |
| ELXRPKNR5Z | 2,238 | 5.51 | 0.544 | 1.082 | 0.01 mm | 0.991 mm | 1 |

Every invariant holds at every size. Against the start of the round (swept
broad phase, sequential queries, PCG x5, `pd_iters=10`) the same elements go
from RTS 0.036-0.213 to 0.191-0.544; the study-1 element itself goes from
27.31 ms to a 10.0-15.2 ms range depending on the GPU's thermal state, i.e.
RTS 0.198 mean and up to 0.30 on a good frame.

Note that the last two rows of the gain are the caller's parameters
(`pd_iters` 10 -> 5 and `linear_iters` 5 -> 2), not code: the code-only gains
are the tight broad phase, the query overlap, and the solver change.

### The iteration budget, swept jointly

`pd_iters` and `linear_iters` are one budget spent in two places, so they were
swept together (study-1, tight broad phase, PCG, 120 frames per cell, one
process each):

| pd_iters | linear_iters | ms/frame | **RTS** | area | seam max | body min |
|---|---|---|---|---|---|---|
| 3 | 2 | 9.72 | **0.309** | 1.175 | 0.17 mm | 0.404 mm |
| 4 | 2 | 11.12 | **0.270** | 1.182 | 0.17 mm | 0.773 mm |
| 5 | 1 | 13.23 | 0.227 | 1.196 | 0.16 mm | 0.860 mm |
| 5 | 2 | 14.45 | 0.208 | 1.177 | 0.15 mm | 0.240 mm |
| 5 | 3 | 15.88 | 0.189 | 1.174 | 0.19 mm | 0.756 mm |
| 6 | 2 | 16.97 | 0.177 | 1.172 | 0.15 mm | 0.843 mm |

Every cell keeps the invariants (area 1.17-1.20, seams below 0.2 mm, no
penetration), so on this scene the quality metric does not separate the cells
at all - only the caller's tolerance for a drape that differs from a fully
converged one does. `pd_iters` dominates; `linear_iters=1` buys another 9%
over 2 but at visibly more stretch (area 1.196).

The study-1 notebook block uses `pd_iters=4`, `linear_iters=2` (RTS 0.270);
3 and 5 are one-value changes in the same block.

## Instrumented: what the self-collision traversal actually does

With the frame at `pd_iters=4` the broad phase is 47.5% of the GPU time
(`query_ee` 2.37 + `query_ef` 1.42 + `query_vf` 0.76 ms), so the traversal was
instrumented directly (temporary counters in the ee kernel, since reverted):

| quantity | measured |
|---|---|
| queries per launch | 158,146 (one per edge) |
| **nodes visited per query** | **91.8 - 102.9** (grows as the cloth drapes) |
| deepest stack occupancy | **21 entries** |
| pairs emitted per launch | 5,368 - 19,753 |

That is the number the layout work could never move: **~100 node visits per
query against ~0.1 emitted pairs**, i.e. 99.9% of the traversal is fruitless.
It also sizes the stack honestly - the 64-entry frame (256 bytes per thread)
is three times larger than the deepest traversal - so that was tried:

| variant | min ms | median ms |
|---|---|---|
| 64-entry stack (current) | 9.99 | 10.10 |
| 32-entry stack | 10.06 | 10.28 |

A wash (x1.008), and `query_ef` came out worse in the resource report (33
registers with a 256-byte local frame became 86 registers with the stack kept
in registers). Reverted.

The conclusion is that per-visit cost and per-visit storage are both
exhausted; the only remaining variable is the **visit count**, which is a
property of the tree and of the query volume:

- a good BVH descends a small box in ~2 tests per level, i.e. ~40 visits for
  an 18-level tree over these primitives; the measured 100 says the LBVH's
  node boxes overlap far more than they need to (a binned-SAH build is the
  standard fix, and it runs at rebuild time only - every 20 frames);
- the alternative structural fix is not a BVH at all: the primitives are
  uniform and small, so a spatial hash grid answers the same query in the 27
  cells around the box, and the repository already has a hash path for the
  point/triangle queries.

### Is the tree simply stale? No (measured)

The instrumented visit count grew from 91.8 to 102.9 over five frames, which
extrapolates to ~142 at the current 20-frame rebuild interval - so the tree
being deformed out of shape looked like the obvious cause. The rebuild
interval was made a parameter (`bvh_rebuild_interval`) and A/B-ed in one
binary, 40 frames x 3 rounds, interleaved:

| rebuild interval | min ms | median ms | RTS(min) |
|---|---|---|---|
| 20 (current) | 9.62 | 9.63 | 0.312 |
| 5 | 9.70 | 10.14 | 0.309 |
| **1 (every frame)** | **12.17** | **12.36** | 0.246 |

Rebuilding every frame is 25% *slower*: the rebuild (Morton sort + tree +
edge ranks over 158 k primitives) costs more than the clustering it recovers.
Five frames is indistinguishable from twenty. So the ~100 visits per query
are a property of the LBVH itself, not of how stale it is, and the parameter
was reverted (it has no value at either end of its range).

That closes the tuning-level work on the broad phase: the traversal's
per-visit cost, its storage, its stack and its rebuild cadence have all been
measured and none of them move the frame. What is left is a different tree -
a binned-SAH build, where the split is chosen by surface area rather than by
Morton order - or no tree at all (a spatial hash grid over these uniform,
small primitives).

## Why half-precision assembly does not transfer as-is

The plan was to assemble the Jacobian in
half: a `half2` atomic stores two entries per transaction, which would take a
bending element from 102 atomic adds to 42 and cut the reduction-sector count
by 59% in both assembly kernels. Before writing any of it, the arithmetic was
checked offline against the magnitudes this scene actually produces - a
diagonal 3x3 entry receives contributions from bending (`bending_k` 1e-2),
the membrane FEM term (~1e2), the mask stiffness (2e3) and the stitch
constraint (`sewing_k` 4e4), so a single entry spans more than six orders of
magnitude while `half` carries about three decimal digits.

Simulating the accumulation (200 trials, representative magnitudes, uniform
0.5-1.5 jitter per contribution):

| accumulator | relative error of the accumulated entry |
|---|---|
| float32 | 2.6e-8 median, 9.4e-8 max |
| **float16** | **overflow to inf** - two stitch contributions alone exceed 65504 |

And that is the good case: scale the whole matrix down enough to stop the
overflow (a factor of 16 or 64, which is free to undo in the matvec) and the
smallest group - the bending term, 3.5e-7 of the entry - falls below the
resolution of the accumulator entirely. Half-precision accumulation would not
be a rounding change to the operator, it would delete the bending coupling
wherever a stitch or mask term is present.

Half-precision coupling blocks only pay off when the accumulated values sit in
a narrow band, which is not the case here: the representation does not
transfer as-is. Either the matrix needs a per-array scale chosen so the useful
terms survive, or the block layout has to change for a different reason.

### Tried anyway, and it fails for a measurable reason

The diagonal/coupling split above was then implemented for real, because it is
the shape to try: keep the diagonal in float, move only the
off-diagonal coupling blocks to half (`Mat3h`, five `__half2` atoms per block
instead of nine float atoms, 20-byte records instead of 36). The build
succeeded - sm_86 emits a single `ATOM.E.ADD.F16x2` per half2, and the bending
kernel's disassembly shows exactly the intended 30 `ATOM.E.ADD.F16x2`
(6 coupling blocks x 5) plus 48 float reductions (4 diagonal blocks x 9 + 4
forces x 3) = 78 atomic instructions instead of 102, with the same 128
registers and no spills.

It also **diverges to NaN within three frames**, and the solver's own debug
dump says why:

| quantity in the first frame | value |
|---|---|
| diagonal block entries | ~4.5e4 (the `sewing_k` 4e4 penalty dominates) |
| coupling entries | up to +/-1.2e4 |
| smallest meaningful coupling | ~1e-4 |

Six couplings accumulating into one entry exceed half's 65504 ceiling, and the
range that must coexist inside a single accumulator is eight orders of
magnitude against half's ~5 orders of exponent (plus a 3-decimal mantissa).
No single scale fixes both ends: scaling down far enough to stop the overflow
pushes the 1e-4 couplings into subnormals, where they keep about three bits.
The change was reverted.

**The conclusion is not "half precision is impossible" - it is that half
precision only pays for an operator whose values sit in a narrow band, and
this one does not.**
Our system is dominated by a 4e4 *penalty* stiffness (the stitch spring, which
is there deliberately: a closed seam must still carry load for the tearing
model), so every diagonal block is ~4.5e4 and the couplings are ~1e4. Their
half blocks only make sense if their constraints are *projected* rather than
penalised, which keeps the operator O(1)-O(1e2). That is the same structural
difference that explains why their solver reaches convergence in fewer
iterations: a well-scaled operator. Matching their half precision therefore
means first matching their constraint formulation, not their storage type.

### Is the 4e4 penalty actually needed? Yes (measured)

If the penalty were only a numerical convenience it could be lowered, the
matrix would become half-friendly, and the whole half-precision path would
open up. Same scene, same iteration budget, only `sewing_k` changes, 150
frames:

| sewing_k | ms/frame | RTS | area | seam gap at the end | **worst seam gap in the run** | body min |
|---|---|---|---|---|---|---|
| 40000 (current) | 22.29 | 0.135 | 1.152 | 0.097 mm | **2.11 mm** | 0.833 mm |
| 4000 | 22.63 | 0.133 | 1.147 | 0.419 mm | **13.77 mm** | 0.591 mm |
| 400 | 23.92 | 0.125 | 1.123 | 1.173 mm | **149.76 mm** | 0.315 mm |

Relaxing the penalty tears the garment apart and buys no time at all. The
penalty carries the seam load, so it stays, and with it the penalty-dominated
matrix. **The half-precision route is therefore closed until the seam becomes
a hard constraint (projection or elimination) that removes the 4e4 term from
the global operator** - that is a solver-architecture change, and it is the
same change that would cut the iteration count.

**Update (measured later, see "The seam is not what makes the matrix
half-hostile"): this last inference is wrong.** The 4e4 stitch term was
estimated from an incomplete dump. Assembling every contribution separately
shows that contact reaches 7.6e6 and the AOGS bending diagonal 8.2e4 in the
same frame, and that removing the stitch term entirely still leaves 211
diagonal entries above half's ceiling. The hard-constraint seam is still worth
having on its own merits; it is *not* the key to half-precision storage.

## The query order: proof that the BVH-order mapping pays

`query_ee_pairs_capsule_kernel` maps thread i to `nodes[i].x - 1`, i.e. to the
i-th *leaf* of the tree, which is the i-th primitive in Morton order. That was
the project's own earlier optimisation, and it had never been isolated. It is
now a switch (`bvh_query_order`, default 1) so both orders can be measured in
one binary.

Nsight Compute, one launch each, same frame:

| order | L1 requests | L1 sectors | sectors/request | L1 hit rate | HMMA/inst |
|---|---|---|---|---|---|
| BVH/Morton (current) | 6,045,381 | 47,325,413 | **7.83** | **85.1%** | 53.62 M |
| natural edge order | 6,075,692 | 55,408,248 | 9.12 | 70.4% | 53.97 M |

Interleaved A/B, 40 frames x 4 rounds, whole frame:

| order | min ms | median ms | **RTS** |
|---|---|---|---|
| BVH/Morton | **21.42** | **23.10** | **0.140** |
| natural | 22.83 | 25.00 | 0.131 |

So the mapping is worth **x1.07-1.08 of the whole frame**, and the mechanism is
exactly the one it was added for: 14% fewer sectors per request and 15 points
more L1 hits. (A single-kernel Nsight Compute duration comparison is
misleading here - it reported the Morton order as slower, which the
end-to-end interleaved A/B contradicts; the counter values, not the replay
duration, are what the ordering changes.)

### Are there better orders? The spatial part is already at its limit

With 32 consecutive Morton leaves per warp, the bounding-box diagonal of the
warp's 32 edge centroids is 48 mm median / 59 mm mean (155,203 edges of the
whole cloth-plus-body mesh). The primitives are ~5 mm, so 32 adjacent cells of
a space-filling curve *cannot* span much less than that: the ordering is at
the limit of what a curve can do.

The residual spread is not spatial. A sectors-per-request of 7.83 over a
24-byte AABB record means a warp is touching about ten *distinct node records*
per load, not about two: the lanes are not reading different places in space,
they are reading different nodes because their traversal *stacks* diverge.
No space-filling curve fixes that - the fix is a warp-cooperative traversal
that walks one shared node list for the whole warp, which is a different
kernel structure rather than a different sort key.

## The stitch cannot leave the matrix either (measured)

The half-precision dead end traced back to one term: `sewing_k` (4e4). If the
seam were carried by the projection instead of by the penalty, the matrix would
stop being penalty-dominated and the half route would open. The stitch
constraint's matrix contribution is a `k*I` block on each of the two seam
vertices (the off-diagonal coupling is already omitted), so a switch was added
to drop it while keeping the force: `sewing_matrix=0`.

Result over 150 frames: **157 of 975 stitches ever close, area ratio 95.0
(the garment explodes), final seam gap 6.1 m, 56.8 ms/frame**. The reason is
structural, not numerical: the projection is *gated* on `check_sewing`, which
only declares the seam assembled once the stitches are within their distance
threshold - and it is the spring's matrix term that brings them there. The
penalty is therefore load-bearing twice over: it assembles the seam and it
carries its tension.

A hard-constraint seam is still the right long-term answer, but it is a
redesign of the seam assembly (unconditional projection or an elimination
formulation), not the removal of a term. Reverted.

## Tried: warp-cooperative traversal (fast kernel, slower frame)

The measurement above says the broad phase's residual cost is that a warp's
lanes read ~10 distinct node records per load because their stacks diverge, so
the obvious fix is a warp-cooperative traversal: one shared node list per warp,
every lane testing its own query against the node being visited, so the node
and bound loads become broadcasts and no lane idles while another descends.

It was implemented for the self-collision query behind `ee_coop_query`
(default off), with the per-lane kernel kept for A/B in the same binary. Two
things had to be right:

- the children of a visited node must be **deduplicated across the warp**
  (`__match_any_sync` + lowest-lane leaders). Without it the 32 lanes push 32
  copies of the same child, the shared stack grows by ~63 entries per step and
  overflows within eight levels - the first version did exactly that and lost
  every subtree below, which showed up as a runaway frame rather than a wrong
  answer;
- with the dedup, the measured deepest occupancy of the shared list is **21
  entries**, i.e. the union frontier is the traversal depth, not 32 stacks'
  worth of work.

With that fixed the kernel is genuinely faster in isolation (Nsight Compute,
one launch each; Nsight Systems, 25 frames, `pd_iters=10`):

| metric | per-lane | cooperative |
|---|---|---|
| ee kernel, ncu | 2.20 ms | **1.18 ms** |
| ee kernel, nsys steady state | 2.76 ms | **1.95 ms** |
| L1 sectors per request | 7.83 | **1.08** |
| L1 sectors total | 47.3 M | **14.5 M** |
| L1 hit rate | 85.1% | 77.0% |
| issue active | ~18% | **41.5%** |
| active lanes per warp | 14.15 | **25.5** |
| instructions executed | 53.6 M | 121.8 M |

**And the frame gets slower.** Three independent interleaved A/Bs, 40 frames
per round:

| configuration | per-lane | cooperative | ratio |
|---|---|---|---|
| overlapped queries (default) | 26.63 / 28.19 ms | 27.36 / 29.27 | x0.97 |
| queries sequential (`bvh_streams=0`) | 28.26 / 29.73 | 30.00 / 30.27 | x0.94 |
| overlapped, 2 KB shared stack instead of 16 KB | 26.55 / 27.59 | 27.57 / 29.34 | x0.96 |

So the technique trades instructions for latency: the cooperative kernel
executes 2.3x the instructions to save 3.3x the sector traffic and 0.8 ms of
its own time, while the frame is issue/throughput-bound and pays for the extra
instructions everywhere else - including when the three queries are run
sequentially, i.e. this is not only an interaction with the stream overlap.
The confound is that a different traversal order also changes which pairs
survive the 16-slot truncation, so the two variants do not simulate exactly
the same contacts; that is worth 1-2% at most and does not explain the sign.

Reverted. The general lesson for this codebase: an isolated kernel's duration
is not the optimisation target here - Nsight Compute reported the cooperative
kernel as 1.86x faster and the frame as 4% slower, the same trap the
query-order A/B sprung earlier.

## Where the traversal's visits actually go, and why the tree is not at fault

The next question is whether the tree is *badly* clustered or the query is
simply that big, so the visits were split (temporary counters again, one
study-1 frame at the current configuration):

| quantity | per query |
|---|---|
| internal-node tests | **51** |
| leaf visits | **14** |
| total | 65 |
| pairs emitted | 0.1 |

Fourteen leaf visits means the query box overlaps fourteen leaf AABBs. The
query box is the swept edge plus `query_radius` plus the object thickness,
i.e. roughly 10-20 mm; with ~5 mm mesh spacing over a ~1 m garment, a box that
size contains on the order of ten to forty edges. **The leaf count is what the
geometry dictates, not an artifact of the tree.** And 51 internal tests for 14
overlapping leaves is 3.6 tests per leaf, which is ordinary behaviour for a
binary tree over 158 k primitives (a perfect tree would be nearer 2 per level
of depth, i.e. ~34 for a 17-level tree, so the build is within 1.5x of ideal,
not 3x as the raw visit count suggested).

That also explains why leaf packing - the one thing the open-source reference's
in-tree BVH test (`src/test/bvh/bvh.cu` in the Newton checkout, a Warp port,
64-bit Morton keys plus
`mark_packed_leaf_nodes` with `leaf_size = 8`) does differently - is not
obviously a win here: packing eight primitives per leaf shortens the tree by
three levels, which removes ~9 of the 51 internal tests, but makes every one
of the 14 leaf visits evaluate up to eight primitives instead of one. The two
effects nearly cancel for a query whose cost is dominated by how many leaves
it overlaps rather than by tree depth.

The broad phase is therefore close to its structural floor for this query
volume. Getting below it needs either fewer queries (batching spatially
adjacent ones into a single traversal - the SIMT warp already shares the
instruction stream across 32 Morton-adjacent edges, hence the measured 7.8
sectors per warp load) or a different structure (a spatial hash grid, which
for these uniform small primitives answers the same query in the 27 cells
around the box).

## The code-only gain, at unchanged parameters

Because iteration counts are the caller's, the honest summary separates code
from parameters. Interleaved A/B of the unmodified sources against the current
ones, *both* at the original study-1 parameter block (`pd_iters=10`,
`linear_iters=5`, PCG), same toolchain, 40 frames x 3 rounds:

| build | min ms | median ms | **RTS** |
|---|---|---|---|
| original sources | 29.57 | 30.10 | 0.101 |
| current sources | 21.30 | 21.31 | **0.141** |

**The code changes are worth x1.39-1.41** at identical physics: the tight
non-swept broad phase and the three-stream query overlap. Everything beyond
that in the numbers reported earlier in this document came from the iteration
budget, which the end user owns - the study-1 notebook block is back at its
original `pd_iters=10`, `linear_iters=5`, and the repository preset keeps the
independently validated 5 / 2.

## The seam is not what makes the matrix half-hostile (measured)

The half-precision sections above end on an inference rather than a
measurement: the operator is penalty-dominated because of the `sewing_k` 4e4
stitch term, so turning the seam into a hard constraint should open the half
route. That was checked directly, and it is wrong.

Method: the solver was temporarily instrumented (reverted afterwards) to
assemble one contribution at a time into an empty operator - `contact`,
`sewing`, the FEM membrane and the AOGS bending element - on a chosen frame,
then copy both the diagonal blocks (29784 x 9 entries) and the coupling blocks
(172261 x 9 entries) to the host and write them raw. A replay script rounds
every partial sum to half after each add, which is what `ATOM.E.ADD.F16X2`
does, and compares against the float64 sum. Study-1 scene, frame 60, all 975
stitches closed:

| contribution | non-zero entries | log10 |entry| (min / median / p99 / max) | entries above 65504 |
|---|---|---|---|
| contact (vf + ee + ef + friction) | 13,305 | -12.8 / 2.90 / 5.03 / **6.88** | 190 |
| stitch penalty | 5,706 | 4.60 / 4.60 / 4.90 / 4.90 | 144 |
| FEM membrane | 268,056 | -2.57 / 2.88 / 3.85 / 3.99 | 0 |
| AOGS bending | 268,056 | -4.70 / 2.97 / 4.28 / 4.92 | 2 |
| **assembled diagonal** | 268,056 | -2.27 / 3.23 / 4.65 / **6.88** | **412** |
| assembled diagonal, stitch removed | 268,056 | -2.27 / 3.23 / 4.33 / 6.88 | **211** |

The stitch is not the ceiling, and it is not close. Contact reaches 7.6e6,
more than a hundred times half's 65504, and the AOGS bending diagonal reaches
8.2e4 on its own. Removing the stitch term from the operator - which is
exactly what a hard-constraint seam would do - leaves **211** diagonal entries
above the ceiling and the same 9.1-decade span. The stitch writes to 5,706 of
268,056 diagonal entries, 2.1% of them, and the largest value it can ever
produce is 8e4 (a vertex shared by two stitches).

The same frame with `friction_on=0` moves the ceiling instead of removing it:
the contact term falls from 7.6e6 to 4.6e4 while the AOGS bending term rises
from 8.2e4 to 2.6e7, because the cloth then reaches configurations where the
bending geometry is nearly degenerate. The ceiling is not a property of one
model; it is whichever term happens to be stressed.

### The replay: no global scale rescues it

| accumulator | scale | inf/nan | median rel-err | p99 | max | entries > 1% |
|---|---|---|---|---|---|---|
| diagonal, with stitch | x1 | **412** | 1.95e-4 | 2.0e-3 | 6.5e0 | 482 |
| diagonal, with stitch | x1/256 | 0 | 1.95e-4 | 2.0e-3 | 6.5e0 | 482 |
| diagonal, stitch removed | x1 | **211** | 1.94e-4 | 2.0e-3 | 6.5e0 | 482 |
| diagonal, stitch removed | x1/256 | 0 | 1.95e-4 | 2.0e-3 | 6.5e0 | 482 |
| coupling blocks | x1 | 0 | 1.83e-4 | 1.6e-3 | **2.4e1** | 2,481 |

Scaling down far enough to stop the overflow (x1/256) leaves the error
statistics completely unchanged, at every scale, with and without the stitch.
That is the signature of a mantissa problem rather than a range problem: in
one entry the largest contribution is a median 3.4x, p99 1373x and worst
2.5e6x the second-largest, so the smaller term does not survive the first
rounding. The coupling blocks do not overflow, but 2,481 of 1.55 M entries carry more than 1% error
and the worst is wrong by a factor of 24.

### What this changes

- Half-precision assembly is closed, and the reason is not the seam. The
  blocker is the operator's 9-12 decade dynamic range, produced by contact
  regularisation (the friction term is `mu * f_n / (friction_epsilon * h)`,
  i.e. a 3e-5 m slip-smoothing length at the study-1 substep) and by bending
  elements in near-degenerate geometry. A hard-constraint seam would fix
  neither.
- Half storage is only affordable for an operator whose accumulated values
  already sit in a narrow band and are reduced without atomics; it is not a
  storage swap that can be copied independently.
- The hard-constraint seam keeps its own reasons: the measured 2.11 mm worst
  seam gap under the 4e4 penalty, and the convergence argument. It is now
  decoupled from the half-precision question rather than a prerequisite for
  it.
- For RTS the finding points somewhere else. The same numbers say the
  preconditioner is fighting a 9-decade operator, and that the extremes come
  from contact/friction regularisation and degenerate bending geometry. Both
  are constructible quantities the code chooses, unlike the seam spring, which
  is load-bearing.

## Kept: the Projective-Dynamics iteration as a CUDA graph

The frame issues ~690 kernels. Nsight Systems put the kernel time at 16.97
ms/frame while `update()` plus a device synchronisation measured 21.87 ms, so
about 5 ms of every frame was *not* running kernels. Earlier this was filed
under "launch overhead that a graph would remove, but not the next thing worth
doing" (task 18.4) on the strength of 18.3, which showed that removing the
solver's blocking readback changed nothing. That reasoning was half right: the
readback was not on the critical path, but it was a *precondition* for
capturing the frame, and the gap was real.

Measured directly, with a micro-benchmark of the same shape (690 dependent
tiny kernels on one stream, 50 repetitions):

| sequence | ms/frame | per kernel |
|---|---|---|
| 690 eager launches | 8.09 | 11.7 us |
| one 690-node CUDA graph | 1.02 | 1.5 us |
| ten 69-node graph launches | 1.11 | 1.6 us |

**A dependent launch costs 11.7 us of GPU idle time on this machine.** That is
the whole gap, and it is worth ~20% of the frame.

### The legacy stream cannot be captured here

`cudaStreamBeginCapture(0, ...)` fails with `cudaErrorStreamCaptureUnsupported`
(900) on this driver even in an otherwise empty process; a stream the program
creates itself captures and replays normally. So the iteration was moved onto a
stream the engine owns (`sim_work_stream()`), with an event fork/join around
the iteration loop, and that made the iteration capturable.

### What had to change first

- The PCG solve ended with a blocking residual readback per solve (10 per
  frame), which invalidates a capture. It now raises a sticky device flag
  (`raise_failure_flag_if_nan_kernel`) that the frame consumes once, after the
  join; the safety net is unchanged, only later.
- The solver's mid-loop diagnostics read scalars back to the host; they are
  skipped while the stream is capturing.
- Every launch inside the iteration takes the iteration's stream: the two
  memsets' siblings in the assembly, the three contact kernels, the seam
  force, the FEM and bending element kernels, `prepare_linear_step_kernel`,
  `step_end_linear`, the cub reductions and the eight PCG kernels.
- The `.cuh` propagation in this build tree is not reliable for `*.cu` files,
  so a header-only change needs the sources touched (or a clean build).

### Result

Interleaved whole-frame A/B, same module, study-1, 40 frames per round:

| variant | min ms | median ms | RTS (min) |
|---|---|---|---|
| `pd_cuda_graph=0` | 26.27 | 27.39 | 0.114 |
| `pd_cuda_graph=1` | **21.88** | **22.28** | **0.137** |

Four rounds, every round in the same direction (x1.20 on the minimum, x1.23 on
the median). `update()` + device synchronisation, the same session:
26.62 ms -> 23.10 ms. Against the pre-round module in an interleaved A/B the
whole-frame ratio is x1.08-1.16 depending on how hot the GPU is (the absolute
milliseconds drift by 20% between sessions; the paired ratio does not).

Acceptance across five dataset elements, 150 frames each, `pd_iters=10`,
`linear_iters=5`, both variants in the same session:

| element | ms graph off | ms graph on | ratio | area (off/on) | seam max (off/on) | body min (off/on) |
|---|---|---|---|---|---|---|
| 00YONAPXZE (study-1) | 30.00 | 26.06 | x1.15 | 1.157 / 1.156 | 0.10 / 0.11 mm | 0.897 / 0.541 mm |
| 6YGLO1BHYF | 31.07 | 26.76 | x1.16 | 1.061 / 1.057 | 0.14 / 0.13 mm | 0.926 / 1.082 mm |
| JAO8WE8XSM | 22.65 | 16.93 | x1.34 | 1.130 / 1.129 | 0.12 / 0.11 mm | 0.847 / 0.849 mm |
| P9CM7RF9M5 | 18.38 | 11.98 | x1.53 | 1.108 / 1.105 | 0.11 / 0.05 mm | 0.820 / 0.995 mm |
| ELXRPKNR5Z | 12.57 | 6.53 | **x1.92** | 1.080 / 1.081 | 0.00 / 0.00 mm | 1.120 / 1.008 mm |

Every invariant holds, and the gain grows as the scene shrinks: the launch
overhead is a fixed ~4 ms per frame, so it is 15% of the study-1 element and
half of the smallest one. **This is the first change in the round that makes
the small elements real-time**: ELXRPKNR5Z goes from RTS 0.24 to 0.46.

### What the graph covers, and what it does not

The captured region is one Projective-Dynamics iteration (assembly, contact
force accumulation, linear solve, position update) - about 690 kernels - and
it is replayed `pd_iters` times per frame. The seam projection stays outside
it because its host-side ramp changes per frame, and the broad phase, refits
and frame bookkeeping are outside as well. Those are ~20 launches per frame,
so the remaining launch overhead is small; capturing the whole frame would
need the query streams and the per-frame host decisions inside the capture,
which is a different project.

Correctness of the replay is keyed on every host value baked into the captured
arguments: buffer pointers, the mesh sizes, the constitutive and bending
models, `h`, `mask_stiff`, `max_force_scale`, `bending_k`, `query_radius`,
`linear_iters`, and a parameter version counter bumped by every
`Simulator::set_parameter`. Any change rebuilds the capture; a capture failure
prints once, disables the path and falls back to direct launches.

## The lazy bending Jacobian (measured, then removed)

The bending block is the frame's largest kernel (3.73 ms, 22%) and, measured
earlier, its matrix half costs 2.82 ms/frame while the element evaluation and
the forces cost 0.97 ms. The first attempt at reusing it froze the block from
iteration 1 and lost: the operator then never saw the large initial motion, the
iterations diverged (0.30 m/frame against 0.067) and the divergence fed the
swept broad phase. Freezing only the *tail* is a different trade.

`bend_freeze_after = k` assembles the block normally up to iteration `k`,
saves it (one isolated matrix-only assembly into a saved pair of arrays), and
from iteration `k+1` adds the saved block back into the operator and asks the
element kernel for forces only. One frozen iteration replaces a 379 us
assembly with a 97 us forces-only pass plus the add.

Where the add goes matters, and so does what the save pass is allowed to
touch:

- the block has to be **added after `truncate_forces_kernel`**, because that
  kernel scales the diagonal by the force clamp. A device copy before it would
  scale the bending stiffness along with the contact term, which the exact path
  never does;
- the save pass must be **matrix only** (`forces=nullptr`). Running it with
  forces made the clamp see a contact-plus-bending magnitude on that one
  iteration and softened the operator enough to be unstable.

Interleaved whole-frame A/B, 40 frames x 4 rounds, one module, whole-loop graph
on, `pd_iters=10`, `linear_iters=5`:

| `bend_freeze_after` | min ms | median ms | RTS (min) |
|---|---|---|---|
| -1 (exact Jacobian) | 20.74 | 22.45 | 0.145 |
| 3 | **19.19** | **20.28** | **0.156** |

(x1.08 on the minimum, x1.11 on the median.) An earlier loop, measured before
the two ordering fixes and with the block copied in before the clamp, priced
the same change at x1.10 on its own and x1.02-1.15 across elements - the fixes
remove a stiffness error and a stability hazard, and cost about a third of the
raw win, which is the honest price of doing it correctly.

### The bug the ordering fix exposed

The save pass ran into a silent out-of-bounds write that took the smallest
dataset element to `PCG nan`. The add kernel was called with `n`, and `n` had
been reassigned to the bend-element count earlier in the iteration, so the
diagonal add walked 126,000 blocks (4.5 MB) past the end of `Jx_diag` on the
study-1 element: it corrupted whatever followed without any error, which is
why the timing A/B looked fine while one 40-frame round showed a 0.089 m
displacement spike that the exact path never produced. On `ELXRPKNR5Z` the
same walk is 4,500 blocks past the end and lands straight in the solver's
work arrays, which is how it was found. Both branches now size the diagonal
add by `nb_all_cloth_vertices` explicitly.

Acceptance across five elements, 150 frames each, graph on, exact against
lazy-at-3:

| element | exact ms | lazy ms | ratio | area (exact/lazy) | seam max (exact/lazy) | body min (exact/lazy) |
|---|---|---|---|---|---|---|
| 6YGLO1BHYF | 26.95 | 24.22 | x1.11 | 1.057 / 1.058 | 0.13 / 0.16 mm | 0.921 / 0.849 mm |
| JAO8WE8XSM | 17.09 | 15.71 | x1.09 | 1.130 / 1.131 | 0.11 / 0.12 mm | 1.036 / 0.997 mm |
| 00YONAPXZE | 25.12 | 23.29 | x1.08 | 1.154 / 1.156 | 0.10 / 0.06 mm | 0.783 / 0.373 mm |
| P9CM7RF9M5 | 11.72 | 11.07 | x1.06 | 1.103 / 1.106 | 0.06 / 0.10 mm | 0.869 / 0.916 mm |
| ELXRPKNR5Z | 6.50 | 7.06 | **x0.92** | 1.081 / 1.081 | 0.00 / 0.00 mm | 1.022 / 1.034 mm |

Every invariant holds and the differences are inside the engine's run-to-run
non-determinism. It is a numerical approximation - a chord/quasi-Newton tail -
and the quality signal available on this scene cannot resolve it, which is the
same reason the iteration budget stays the caller's decision. What it is *not*
is a rebadged iteration cut: every frozen iteration still evaluates the exact
forces and still solves the linear system; only the operator's bending block
is reused. On the smallest element it is a small *loss* (x0.92, and within
noise) - there the mesh is small enough that the extra save pass and the add
are not amortised.

**Removed again when the branch was cleaned up for merge.** At the shipped
operating point (`dt = 0.0045`, `pd_iters = 3`) it measures 8.42 ms against
8.30 ms: the frame is dominated by the fixed per-substep cost, so the win only
exists at the heavy end of the iteration budget, where the recommendation does
not sit. Carrying a whole approximation - plus its per-frame save buffers and
the ordering hazards documented above - for a case the default never uses is
not worth the review surface; the numbers stay here for whoever wants to
revisit it.

The per-iteration branch is why the graph has to record the whole loop rather
than a single iteration (see above).

## Compatibility with the cloth-plasticity change

`openspec/changes/cloth-plasticity` makes the rest shape mutable:
`bend_rest_theta` (and, behind a flag, `edge_lengths`) become live arrays that
a per-frame plastic update and `freeze_rest_shape()` rewrite, with
`refresh_rest_dependent_state()` re-deriving `areas`, `bend_valid`,
`bend_factor` and the PD diagonal. Checked against the two things this round
added:

- **The captured loop is unaffected.** The plastic update runs in
  `Geometry::update_for_frame()` (plasticity design D5), which is outside the
  captured region - the capture covers `SolverPDNewton`'s iteration loop only.
  Rest values are read from device memory when the graph replays, not baked
  into it, so a new `bend_rest_theta` takes effect on the next frame with no
  recapture. The same holds for `static_diags` / `Jx_diag_pd` (D7 item 4) and
  for `bend_valid`, whose per-element branching is device-side.
- **What invalidates the capture is a buffer moving or a size changing, not a
  value changing.** The replay key now covers every array the loop reads, by
  pointer *and* size, so a refresh that reallocates (the plasticity design
  requires it not to) forces a recapture instead of replaying against freed
  memory.
- **`freeze_rest_shape()` and `reset_plasticity()` need no graph handling**:
  they rewrite device arrays between frames, which is exactly the case the key
  is designed to leave alone.
- **Nothing in the captured loop caches a rest angle across frames**: the
  bending kernel reads `bend_rest_theta` on every iteration, and the plastic
  update only changes it between frames, so a recapture is never needed for a
  plastic step. If D5's open question is ever resolved the other way (update
  per substep, inside `update_for_step`), the update still runs before
  `solver->step()`, which is the loop the graph replays.
- The plasticity design's own constraint - "keep the per-frame cost
  O(edges + triangles), allocation-free, confined to one update site" - is what
  keeps the capture valid. If a later revision has to rebuild the unified bend
  table mid-run, the key catches it and the capture is rebuilt once.

## Recommended defaults after round 1 (measured)

The code changes above set the *cost per simulated second*; `dt` / `step_h`
and the iteration budget are the caller's, and they decide how much simulated
time one frame covers. The two are not interchangeable: swept over the same
0.45 s of physics, with 30 warm-up frames excluded (the seam projection ramps
its snap distance to 5 cm, so the assembly frames are not a quality signal),
medians of two runs each in one session:

| configuration | dt = step_h | pd | lin | ms/frame | **RTS** | area (mean) | area (max) | seam max | body min |
|---|---|---|---|---|---|---|---|---|---|
| study-1 notebook today | 0.003 | 10 | 5 | 22.74 | 0.132 | 1.210 | 1.391 | 1.47 mm | 0.269 mm |
| preset budget, current step | 0.003 | 5 | 2 | 11.02 | 0.272 | 1.218 | 1.401 | 1.27 mm | 0.216 mm |
| floor budget, current step | 0.003 | 3 | 2 | 8.54 | 0.351 | 1.217 | 1.396 | 1.31 mm | 0.186 mm |
| **recommended** | **0.0045** | **5** | **2** | 10.85 | **0.415** | 1.255 | 1.416 | 1.48 mm | 0.264 mm |
| speed tier | 0.006 | 3 | 2 | 8.34 | **0.719** | 1.282 | 1.407 | 1.14 mm | 0.212 mm |
| heavier budget, larger step | 0.0045 | 10 | 2 | 19.58 | 0.230 | 1.254 | 1.416 | 1.72 mm | 0.236 mm |

Three things follow from the table, and they are the reason the recommendation
is what it is:

1. **RTS is nearly linear in `dt` while the substep count stays at one**
   (`step_h = dt`). Every 1.5x of `dt` is 1.5x of RTS at almost the same
   ms/frame, on every element tested. `dt` with `step_h` left behind buys
   nothing: 0.006/0.003 measures 19.70 ms against 9.73 ms, RTS 0.305 against
   0.617, for the same physics.
2. **The iteration budget barely moves the quality metrics but moves the cost
   proportionally.** Area is 1.213 / 1.218 / 1.217 at `pd_iters` 10 / 5 / 3 on
   the current step, while the frame goes 19.05 / 11.02 / 8.54 ms. The stretch
   this scene shows is set by the time step, not by how long the solve is run.
3. **The price of the larger step is the area (stretch), and it is
   element-dependent.** Against the 0.003 reference: study-1 +3.2% at 0.0045
   and +5.7% at 0.006; across elements at 0.0045, 6YGLO1BHYF +2.0%,
   ELXRPKNR5Z +4.9%, and JAO8WE8XSM +10.5% (its trend is monotone: 1.305 /
   1.352 / 1.400 / 1.442 at 0.003 / 0.0035 / 0.004 / 0.0045, i.e. ~+3.5% per
   0.5 ms with no knee). Seam closure, body clearance, penetration counts and
   the per-frame displacement statistics all stay inside their existing bands.

**Recommendation for the shipped default: `dt = step_h = 0.0045 s`,
`pd_iters = 5`, `linear_iters = 2`, `linear_solver_type = 0` (PCG)** - RTS
0.415 on study-1 against 0.132 for the notebook's current block (x3.1), and
RTS 0.40 / 0.60 / 1.21 on the three spot-checked elements. It keeps the
repository's validated iteration budget, so the only quality axis that moves is
the one documented above (+2 to +10% stretch depending on the garment).

- Quality-first alternative: `dt = 0.003`, `pd_iters = 5`, `linear_iters = 2`
  (RTS 0.27, stretch within 0.4% of the reference).
- Speed-first alternative for preview/interactive use: `dt = 0.006`,
  `pd_iters = 3`, `linear_iters = 2` (RTS 0.72, stretch +5.7% on study-1 and
  up to +10% on the most sensitive element).

The recommended numbers are what the debug window and the study-1 notebook now
use; the harness preset
(`tests/harness/presets.py`) was deliberately left alone: it is the validated
baseline the tests compare against, and moving it is a separate decision from
recommending a shipped default. The harness preset does carry the PCG-with-2
iterations change, which is part of this round's measured work.
