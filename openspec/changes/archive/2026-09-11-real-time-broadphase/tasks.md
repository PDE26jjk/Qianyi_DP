## 1. Measurement baseline

- [x] 1.1 Reproduce the study-1 configuration headlessly (`gcd:rand_00YONAPXZE`,
      PDNewton + the study-1 parameter overlay, `update(0.003)`) and measure a
      frame as `update(dt)` plus a synchronizing read; report
      `RTS = dt / frame_seconds`
- [x] 1.2 Nsight Systems timeline (35 frames): attribute GPU time per kernel.
      Broad phase 45.6% of the frame, 677 kernel launches/frame, GPU busy
      25.7 ms/frame
- [x] 1.3 Nsight Compute on the three broad-phase queries: occupancy, L1
      traffic, instruction counts, stall reasons (latency bound, L1TEX
      long-scoreboard 50-75%, 29-36% achieved occupancy)
- [x] 1.4 `cuobjdump --dump-resource-usage`: the traversal stack is in local
      memory (`STACK:256/288` for the three queries)
- [x] 1.5 Candidate-buffer occupancy per primitive (vf 27%, ee 15%, ef 61% of
      the 16 slots full) and the distribution of the conservative-CCD sweep
      length (median 4.1 mm, p99 118 mm)
- [x] 1.6 Establish the interleaved A/B methodology (round-robin rounds,
      min/median over >= 3 rounds); the test GPU throttles by > 20% between
      runs, and the unmodified sources are also built into the same CMake tree
      so toolchain differences cannot masquerade as results

## 2. Round 1 - broad-phase query overlap (the only retained change)

- [x] 2.1 Overlap `query_vf` / `query_ee` / `query_ef` on three non-blocking
      streams with fork/join events against the default stream
- [x] 2.2 Keep the sequential launches behind `bvh_streams` (default 1)
- [x] 2.3 A/B against an unmodified build of the same toolchain: 38.70 ->
      37.36 ms min, 41.93 -> 38.55 ms median (x1.04 / x1.09)
- [x] 2.4 Repeatability: x1.08 on the mean over five 40-frame repeats, x1.015
      over three 150-frame repeats; 13 runs without an exception (one
      unreproduced process abort observed earlier, not seen since)
- [x] 2.5 `pytest -m quick` passes against the new module (12 passed)

## 3. Rejected kernel-level attempts (recorded so they are not retried)

- [x] 3.1 Overlap-filtered, near-first BVH traversal - 11-23% slower
- [x] 3.2 Velocity clamping (`max_vel` 20/5/1) - x0.96/x0.67/x0.75
- [x] 3.3 BVH rebuild interval (1/5/100 vs 20 frames) - identical within noise
- [x] 3.4 `__launch_bounds__(256,4)` occupancy raise - slightly slower
- [x] 3.5 Interleaved (packed) 32-byte node records - a wash
- [x] 3.6 Morton-range pruning with the dedup key switched to Morton order -
      ~2% on the median, nil on the minimum; reverted because it changes pair
      ownership for a gain inside the noise band
- [x] 3.7 Freezing the bending stiffness block across iterations - slower
      *and* it diverges the solver (0.30 vs 0.067 m per frame)
- [x] 3.8 Gather (CSR, atomic-free) assembly of the bending kernel - 10%
      slower: the element data is re-read four times from scattered addresses
      and the element evaluation is repeated, which costs more than the
      reduction latency it removes
- [x] 3.9 Conclusion: ten attempts span -23% to +9%. The stage costs track the
      geometric scale of the query volume and the kernels' wave counts, not
      primitive counts or memory layout

## 4. Structural work still untried

- [ ] 4.1 Dual-tree traversal emitting a compact pair list (the only
      structural change that reduces visited-node count by construction)
- [ ] 4.2 SAH or leaf-batched tree build
- [ ] 4.3 Per-object BVH layering for animated bodies (local-space build +
      transformed queries) - unproven, see 6.3

## 5. Per-iteration work (35.8% + 12.6% of the frame)

- [x] 5.1 Measure first: `compute_dihedral_bending_AOGS` spends 8.54 M L1
      reduction sectors against 1.18 M load sectors, i.e. 88% of its sector
      traffic is the `atomicAdd` scatter (ncu reports it under `op_red`)
- [x] 5.2 Gather rewrite of that kernel - **slower** (see 3.8); the
      scatter/gather question is closed for this scene
- [ ] 5.3 Reduce PCG cost and launch count (341 launches/frame, 110 block
      reductions)
- [ ] 5.4 Better preconditioning instead of the disabled subspace
      acceleration (subspace is a known negative optimization here)

## 6. Cost scaling and mesh LOD

- [x] 6.1 Four dataset elements of increasing size: `frame_ms ~= 9.1 + 0.366 us
      * cloth_triangles`; RTS 0.285 (4k cloth triangles) / 0.183 / 0.134 /
      0.100 (study-1)
- [x] 6.2 Nsight Systems on the smallest element (574 kernels/frame, 9.28 ms
      GPU/frame): broad phase 67.6% of the frame, and `query_vf` costs
      3.63 ms - the same as on the 14x larger study-1 garment, because the
      kernel launches half the blocks (102 vs 210) at 1.31 waves
- [x] 6.3 Tested the obvious lever directly - a grid-clustered kinematic body
      proxy: 47,500 -> 5,082 body triangles (9.4x) changes the frame time by
      ~5%, inside noise, with 6-10 mm of drape deviation. **Mesh LOD does not
      buy broad-phase time**; the body-mesh-floor hypothesis is refuted

## 7. Speed vs quality trade-offs (the remaining headroom)

- [x] 7.1 `pd_iters` sweep (120 frames, one process per level): 10 -> 5 is
      x1.33 (RTS 0.094 -> 0.125) and 10 -> 3 is x1.95 (RTS 0.184), both with
      seams closed and the cloth-body distance intact; below 3 the drape
      drifts 12 -> 28 mm and the minimum cloth-body distance collapses from
      0.69 mm to 0.10 mm
- [x] 7.2 15 iterations deviates 10.7 mm from 10, so none of these settings is
      converged; the deviation columns measure change from the default, not
      error
- [ ] 7.3 Decision needed: adopt a lower `pd_iters` (a parameter, already
      user-settable) or keep the current baseline

## 8. Learning from the reference implementations

- [x] 8.1 Compared against Warp's `native/bvh.*` and the Newton/Style3D
      collision port this contact code derives from. Transferable
      differences: tight non-swept broad-phase bounds with small substeps;
      16-byte packed nodes, SAH build, shared-memory 32-entry stack and
      block-cooperative tile queries; local per-vertex solvers instead of a
      50-matvec PCG chain; stateful contacts with warm-started multipliers
- [x] 8.2 Same-scene solver comparison with each solver's own preset:
      PDNewton 21.8 ms (RTS 0.138), VBD 18.6 ms (0.162), XPBD 64.3 ms
      (3 substeps), Explicit 421.7 ms (12 substeps)
- [x] 8.3 VBD iteration sweep: 1 -> 8.7 ms (RTS 0.346), 3 -> 13.7 ms (0.220),
      5 -> 18.9 ms (0.159), 10 -> 33.5 ms (0.090)
- [x] 8.4 Visual + invariant comparison against the PDNewton baseline (120
      frames, snapshot grid, area/seam/contact/motion metrics): VBD is faster
      but its **maximum seam gap is 617-640 mm against PDNewton's 0.85 mm** -
      the pattern panels never assemble (visible in the render). The speed
      advantage is entangled with a missing constraint and is not
      like-for-like
- [x] 8.5 Added the seam/stitch path to the VBD route behind
      `vbd_project_stitches` (the projection alone does nothing - it is gated
      on `check_sewing` having declared a seam closed, which needs the stitch
      spring first). With the spring the seam gap collapses from 605 mm to
      1.2 mm, so the garment assembles, **but** at the current parameter block
      the local iterations cannot absorb a 1e5 stitch stiffness: area stretch
      jumps to 1.29, motion to 92 mm/frame and the frame cost goes 14.6 ->
      31.4 ms. Left off by default; the finding is that this is "add the
      constraint and re-tune", not "add the call"
- [x] 8.6 Tuned the stitch constraint (`sewing_k` 1e2..1e5 x 3/10/30
      iterations, 120 frames each). **Closed: the local-solver path does not
      become competitive.** Best case is 1e4 / 3 iterations at 22.1 ms
      (RTS 0.136) against the default's 24.8 ms - only ~1.1x, and with 34%
      cloth stretch and a 4x larger seam gap; the settings that close the
      seams are slower than the default, more iterations are slower *and* no
      more accurate, and 30 iterations diverge
- [x] 8.7 Implemented the tight, non-swept broad phase behind
      `tight_broad_phase` (default off) and swept it against the substep size:
      it wins at every step size (x1.27 at the configured 3 ms step, x1.10 at
      1.5 ms, x1.07 at 1 ms) and does not lose quality in this test (smaller
      seam gap and more body clearance at 3 ms than the conservative
      version). This is the best remaining lever in the change
- [x] 8.8 First validation step: 300-frame runs. Trajectory comparison is
      *inconclusive by construction* - tight vs swept diverges by a mean of
      32.1 mm per vertex, while two runs of the same swept configuration
      diverge by a mean of 46.8 mm, i.e. the engine's run-to-run
      non-determinism exceeds the effect being tested
- [x] 8.9 Long-run acceptance for `tight_broad_phase` (300 frames, invariants
      + rendered side-by-side): no penetration regression (min distance 0.71
      vs 0.63 mm; 5 vs 3 vertices within 1 mm out of ~30k), seams better
      (0.00 vs 2.30 mm), garment assembles in both, at the cost of ~8% more
      cloth stretch (area 1.179 vs 1.097) - for x1.27 speed. Recommend
      enabling it
- [x] 8.10 Enabled `tight_broad_phase` by default (the switch still exists:
      `tight_broad_phase=0` restores the conservative-CCD behavior). Final
      interleaved A/B in the same binary (40 frames x 3 rounds): tight 20.95 ms
      min / 22.02 median (RTS 0.143) against swept 30.39 / 32.57 (0.099) and
      swept + sequential queries 31.96 / 36.08 (0.094). The two retained
      changes together are **x1.72 against the original behavior**
- [x] 8.11 Validated the new default across the dataset: five elements from
      2.2k to 32k cloth vertices, 150 frames each. Tight is faster on every
      element (x1.05 to x3.04; the 32k-vertex outlier goes 82.79 -> 27.22 ms)
      and no invariant degrades (areas 1.01-1.16, seams <= 0.22 mm, minimum
      cloth-body distance positive everywhere). Direction consistent, though
      the magnitude depends on GPU thermal state
- [x] 8.12 Validated the new default with an **animated** body (yawed +-25
      degrees at 0.5 Hz through `update_local_vertices`, 120 frames): still
      x1.49 faster, finite, seams closed, stretch comparable - but the
      clearance to the body narrows from 0.345 mm to 0.127 mm and near-body
      vertices rise slightly (283 -> 322 of ~30k). No interpenetration, but
      the margin is thinner, which is the expected signature of a non-swept
      query against a moving obstacle
- [ ] 8.13 Follow-up if faster body motion is needed: inflate the tight query
      box by the object's per-substep translation (contained change), or fall
      back to `tight_broad_phase=0` for scenes with fast-moving bodies

## 9. Status

Retained: the broad-phase query overlap (`bvh_streams`), the measurement
harness, and the recorded evidence for every rejected experiment.

## 10. A Chebyshev linear solver (designed, implemented, then removed)

Note added at merge: these tasks were carried out, measured, and the result
was not usable (see `design.md`, "Implemented: a reduction-free linear solver,
and what it cost"), so the solver and its kernels were removed again in the
merge cleanup. The tasks are kept as the design record.

Chebyshev is the interesting one here because it needs **no dot products**: the
PDNewton route currently pays 110 block reductions and 341 kernel launches per
frame in the PCG path, and Chebyshev removes the reductions entirely (the
coefficient sequence is data-independent, so it needs no scalar readback and
the whole loop is graph-capturable).

Work plan (one block):

- [x] 10.1 Add `SolverChebyshev : LinearSolver` next to `SolverJacobi` in
      `src/simulation/linear/solver_linear.cu` (the existing
      `src/simulation/solver_Chebyshev.cu` is a stale *standalone* solver, not
      a linear solver, and cannot be plugged in).
      -> DONE: added with a device-side coefficient sequence and no
      reductions, wired as `linear_solver_type=2`. Measured x1.23 faster than
      PCG (15.69 vs 19.28 ms min, interleaved) but its maximum seam gap reaches
      350 mm over 150 frames against PCG's 0.08 mm: at 5 iterations the
      extrapolation is under-converged and the seams tear. Kept as an opt-in
      that needs more iterations or a tighter spectral bound.
- [x] 10.0b The same A/B surfaced that the repository's existing, unused
      **Jacobi** linear solver is reduction-free, **x1.18-1.20 faster than
      PCG** (16.39/16.88 vs 19.28/20.33 ms) and quality-neutral over 150
      frames (seam 0.07 mm, area 1.172, no penetration). Adopting it is a
      preset change (`linear_solver_type=1`) pending the multi-element and
      animated-body validation
      -> DONE: fast on all five dataset elements (e.g. 14.12 -> 10.31 ms and
      27.22 -> 25.43 ms), seams closed (<= 0.09 mm), no penetration, and the
      animated-body check passes as well (0.14 mm seam, 0.507 mm clearance).
      It converges more slowly per iteration than PCG, so stretch is slightly
      higher (areas up to 1.205). **Adopted**: `tests/harness/presets.py` now
      sets `linear_solver_type=1` (the block's re-validation is the five
      elements + animated body + interleaved timing + invariants recorded
      above), and `pytest -m quick` / `-m sim` pass. Note that a caller that
      sets the value explicitly (the study-1 notebook does) overrides it.
- [x] 10.2 Spectral bounds for `M^-1 A`. With `A = M/h^2 + sum_k w_k S_k^T S_k`
      and the Jacobi preconditioner `M = diag(A)`, `M^-1 A >= I/h^2`, so use
      `a = 1/h^2`; for `b`, the diagonal blocks are PSD, so
      `|a_ij| <= (a_ii + a_jj)/2` bounds the off-diagonal row sums and
      `b = 2 * max_i ||M^-1 Jx_diag[i]||_inf` is a safe over-estimate. One
      reduction per frame (not per iteration).
- [x] 10.3 Iteration: `x_1 = x_0 + (1/d) M^-1 r_0`, then
      `beta_k = 1/(d - c^2 beta_{k-1}/4)` and
      `x_{k+1} = x_k + beta_k (M^-1 r_k + (c^2 beta_{k-1}/4)(x_k - x_{k-1}))`
      with `d = (b+a)/2`, `c = (b-a)/2`. One matvec per iteration, zero
      reductions, coefficients computed on the host (they do not depend on the
      solution).
- [x] 10.4 Wire it as `linear_solver_type = 2` in `SolverPDNewton::init` and
      keep PCG/Jacobi selectable.
- [x] 10.5 Validate with the same protocol used for the broad-phase change:
      interleaved A/B against PCG on the study-1 scene, then the invariant
      suite (finiteness, cloth area, seam closure, cloth-body distance) and
      the multi-element and animated-body checks. Keep only if it wins.

Secondary item from the same survey: packed **half-precision** BVH nodes with a
short stack. The earlier packing experiment in this change used 32-byte floats
and measured a wash; 16-byte `__half` bounds are what actually halves the bytes
per visited node, and a depth-bounded register stack replaces the 64-entry
local-memory stack.

## 11. Next structural item: half-precision BVH nodes (planned)

The design to match is half-precision bounds with a depth-bounded traversal;
the same shape appears in Warp (`BVHPackedNodeHalf`, `BVH_QUERY_STACK_SIZE
32` with a shared stack).
This codebase uses a 24-byte `AABB3D` array plus a separate 8-byte child array
and a 64-entry stack in local memory, which is why the earlier *float* packing
experiment (32-byte records) measured a wash: it co-located the data without
shrinking it.

Work plan, cheapest first:

- [ ] 11.1 Cut `BVH_QUERY_LOOP`'s `STACK_SIZE` from 64 to 32 (with the
      existing overflow guard). This halves the per-thread local-memory frame
      (256 -> 128 bytes) and is a one-line change per query kernel - a cheap
      occupancy experiment before anything else. Measure with the usual
      interleaved A/B plus the invariant suite.
- [ ] 11.2 Quantize the tree: bounds stored as `__half` over the scene box,
      rounded conservatively (floor the minima, ceil the maxima) so the tree
      can only over-estimate. Layout the two halves of a node adjacently
      (`{half3 lo; int child_lo}` + `{half3 hi; int child_hi}` = one 32-byte
      record) so a traversal step is one cache line and one to two loads,
      replacing the current two-array access.
- [ ] 11.3 Update the builder, the refit kernels (they currently write
      `AABB3D`), the three broad-phase query kernels and the debug accessors;
      keep the float path behind a switch so the two can be A/B-ed in one
      binary.
- [ ] 11.4 Validate with the established protocol: interleaved A/B, five
      dataset elements, animated body, invariants - and confirm the quantized
      bounds never *lose* a candidate pair (the conservative rounding makes
      that a correctness invariant, not a tuning knob).

## 12. Measured before implementing 11: alignment does not pay, the stack does

- [x] 12.1 Re-read the generated SASS before touching the layout again:
      `query_ee_pairs_capsule_kernel` issues `LDG.E.CONSTANT` x150 and
      `LDG.E.128` x0, i.e. the 24-byte `AABB3D` is fetched as six scalar
      32-bit loads, and the traversal stack is a `STACK:288` local-memory
      frame with 0.63 M `LDL` + 1.08 M `STL` instructions per frame.
- [x] 12.2 Pad the record to 32 bytes with `alignas(16)` and fetch it with
      explicit `float4` loads. The padding alone leaves the SASS byte-identical
      (the compiler will not infer `float3` alignment through a
      `const __restrict__` parameter); the explicit vector fetch is needed to
      get two 128-bit loads. Interleaved A/B, 40 frames x 4 rounds: 15.78 ->
      15.84 ms min, 16.30 -> 16.06 ms median. **A wash; reverted.**
- [x] 12.3 Conclusion: three independent layout experiments (split vs packed
      vs aligned-vector) now agree. Per visit the critical path is one L1
      round trip; neither the byte count nor the load-instruction count is on
      it. Stop tuning the record and remove the *stack* instead.

## 13. Next: a stackless, depth-first-threaded traversal

The target traversal is a single `while` loop with no stack and ~20
registers: a pre-order-threaded tree where descending is `index + 1` and
leaving a dead subtree is `skip[index]` (see the "Reference implementation,
kernel level" section of `design.md`). Our equivalent kernels carry a 64-entry
local-memory stack and
50-74 registers, which caps their occupancy at 41-56%.

Plan, in the established keep-only-what-wins order:

- [ ] 13.1 At rebuild time (every 20 frames, not per substep) compute, from
      the existing Karras topology: the subtree node count (the same bottom-up
      atomic merge the refit already uses), the depth-first pre-order index of
      every node, the permutation of `nodes`/`aabbs` into that order, and
      `skip[node]` = the pre-order index after the subtree (leaf: `+1`,
      interior: `+ subtree_size(left)`).
- [ ] 13.2 Add a second, packed 32-byte record `{float3 lo; int child;
      float3 hi; int skip;}` written by the refit kernels (the bounds are
      already computed there), and a stackless traversal used behind a switch
      so both paths can be A/B-ed in one binary.
- [ ] 13.3 Verify the candidate set is *identical* to the stack traversal's
      before timing: the skip pointer changes the visit order and the 16-slot
      truncation is order-sensitive, so the pair set has to be compared
      directly, not just the drape.
- [ ] 13.4 A/B with the interleaved harness, then the invariant protocol.
      Expect the win to come from occupancy (registers and the local-memory
      frame), not from the byte count - if the registers do not drop, the
      change is not doing what the reference does.

## 14. The linear solver: PCG at 2 iterations replaces Jacobi

- [x] 14.1 Re-read `SolverJacobi::solve`: at `linear_iters=5` its convergence
      check never fires, so it is a fixed-iteration, reduction-free smoother
      with no host synchronization. That is the entire source of its speed.
- [x] 14.2 Sweep the inner iteration count instead of switching solvers.
      Jacobi 5 / 10 / 20 iterations: 24.9 / 29.9 / 69.2 ms per frame with
      0.027 / 0.067 / 1.36 m of motion per frame - more Jacobi iterations do
      not converge, they diverge.
- [x] 14.3 Interleaved A/B, one binary, one session (40 frames x 3 rounds):
      PCG x5 26.66, PCG x3 25.19, **PCG x2 22.77**, Jacobi x5 23.47 (min ms).
- [x] 14.4 Five-element validation (150 frames each) of PCG x2 against the
      PCG x5 baseline and against Jacobi x5: x1.10-1.31 faster than PCG x5,
      area within 0.005 of it, seams <= 0.16 mm, cloth-body clearance positive.
      Jacobi is at best a few percent faster on the mid-size elements and
      visibly stretchier (1.205 vs 1.132 area on JAO8WE8XSM).
- [x] 14.5 **Adopted**: `tests/harness/presets.py` now uses
      `linear_solver_type=0` with `linear_iters=2`; the study-1 notebook block
      matches. `pytest -m quick` (12 passed) and `-m sim` (2 passed, 4 skipped,
      1 xfailed) pass.

## 15. Next: the element-assembly kernels (36% of the frame)

- [x] 15.1 Nsight Compute on `compute_dihedral_bending_AOGS`: 128 registers,
      2 blocks/SM, 29.3% achieved occupancy, **5.09% issue active**, 8.54 M
      global reduction sectors against 1.18 M load sectors, L1 hit rate 9.7%,
      30.5 sectors per reduction request. It is a scattered atomic reduction
      with no reuse, not a compute problem.
- [x] 15.2 Tried the cheap occupancy fix: `__launch_bounds__(256, 4)` cut the
      kernel from 128 to 64 registers and spilled 296 bytes; interleaved A/B
      measured x1.066 *slower* (18.87 -> 20.12 ms min). Reverted. Occupancy
      only pays if the register cut comes from the algorithm, not from ptxas.
- [ ] 15.3 Exploit the symmetry the kernel already relies on: the four
      diagonal `Mat3` blocks are symmetric (their two mixed terms carry equal
      weights), so they need 6 atomics instead of 9 - 102 -> 90 per element.
- [ ] 15.4 The larger item: half-precision assembly of the Jacobian blocks.
      A `half2` atomic stores two entries per
      transaction, so a block is 5 stores instead of 9, and the assembly
      working set (~4.5 MB of `Jx`/`Jx_diag` against 4 MB of L2) starts to fit
      in L2. This touches `Mat3` storage and every consumer (`A_mul_x_offdiag`,
      `Jx_mult_x_diag`, `prepare_linear_step`), so it needs the same
      one-change-per-A/B discipline and a precision check on the linear solve.
- [ ] 15.5 Shorten the kernel's live range so the register cut is free: the 7
      cached outer products are 63 registers. Computing them per block instead
      (pre-scaling the first vector) costs ~15% more FMA in a kernel that is
      95% idle on issue, and would let 4 blocks/SM fit without spilling.
      -> Tried as written: 128 -> 121 registers (still 2 blocks/SM), A/B x1.014.
      A wash; reverted. The allocator does not give the registers back, so this
      has to be attacked by splitting the kernel, not by re-associating the math.

## 17. Iteration count re-measured on the current configuration

- [x] 17.1 `pd_iters` sweep (study-1, tight broad phase, PCG x2, 120 frames per
      level): 3 -> 9.09 ms / RTS 0.330, 5 -> 12.84 / 0.234, 7 -> 17.65 / 0.170,
      10 -> 23.63 / 0.127. Area 1.167-1.177, seams <= 0.17 mm, cloth-body
      clearance positive at every level.
- [x] 17.2 The study-1 notebook block was set to the preset's `pd_iters=5` (it
      had been overriding it to 10). Measured with that block: 13.67-13.74 ms
      mean, 9.99-10.89 ms minimum, **RTS 0.219 mean / 0.28-0.30 best**.
- [ ] 17.3 Decision still open for the caller: `pd_iters=3` is another x1.4
      (RTS 0.33), at 12 mm of deviation from the 10-iteration drape - which is
      inside this engine's run-to-run non-determinism, so it can only be
      settled by a visual check, not by a metric.

## 18. Next: CUDA graph capture of the frame

- [x] 18.1 Measured the available headroom first: under Nsight Systems the
      frame is 16.15 ms of kernel time against a 21.86 ms wall span, so 26% of
      the frame is not running kernels. The API trace shows 13.4 D2H copies
      (0.078 ms), 7.3 stream syncs (0.64 ms blocking) and 1.2 event syncs
      (0.38 ms) per frame.
- [x] 18.2 Confirmed PCG is already device-side (alpha/beta are computed by
      one-thread kernels on device scalars) - its only host round trip is the
      blocking NaN check after each solve, which also drains the stream.
- [x] 18.3 Make the NaN check deferred: the coefficient kernels now set a
      sticky device flag and the host reads it every eighth solve, so the ten
      per-frame stream drains disappear. **Measured as no effect** - two
      interleaved A/Bs disagreed in sign (x1.021 and x0.926), which says the
      host round trips are not on the critical path while the GPU is
      saturated. Reverted; the finding is recorded in `design.md`.
- [x] 18.4 **Deprioritized, then re-opened and shipped.** With 18.3 showing that synchronization is not
      what the 26% non-kernel gap is made of, a CUDA graph would recover
      mostly launch overhead (~390 launches/frame). Still the largest single
      remaining structural item, but no longer ahead of the assembly work.
      **Superseded by 30: the gap was launch overhead after all (11.7 us per
      dependent launch), and the graph is implemented, kept and worth
      x1.20-1.23 on study-1 and up to x1.92 on the small elements.**

## 19. What is left, in order of measured size

| lever | measured size | cost |
|---|---|---|
| `pd_iters` 5 -> 3 (caller parameter) | x1.41, RTS 0.234 -> 0.330 | quality decision only |
| bending + FEM assembly (36% of the frame) | up to x1.5 on that 36% | half-precision blocks or a kernel split; touches `Mat3` storage and every consumer |
| the three broad-phase queries (28% of GPU time, ~16% of the wall) | x1.3-2 on the traversal | stackless/threaded traversal |
| CUDA graph capture of the frame | bounded by ~390 launches/frame | deprioritized by 18.3 |

## 20. Joint iteration-budget sweep, and why the assembly refactor is next

- [x] 20.1 Swept `pd_iters` x `linear_iters` together (six cells, 120 frames
      each, invariants per cell). The invariants are flat across the whole
      grid, so only the iteration budget moves the frame: (3,2) 9.72 ms /
      RTS 0.309, (4,2) 11.12 / 0.270, (5,2) 14.45 / 0.208, (5,1) 13.23 /
      0.227, (6,2) 16.97 / 0.177. The study-1 notebook block now uses
      `pd_iters=4`, `linear_iters=2`.
- [x] 20.2 Sized the assembly refactor with the same profile data. Both
      assembly kernels are scattered atomic reductions - `compute_BW_FEM` is
      the same shape as the bending one (80 registers, 3 blocks/SM, 40.7%
      occupancy, **3.04% issue active**, 3.54 M reduction sectors against
      0.69 M load sectors, 27.5 sectors per request).
- [x] 20.3 Recorded why element reordering cannot fix the scatter: at a
      36-byte block per vertex, 32 consecutive vertices are still 36 bytes
      apart, i.e. one sector per lane. The measured 30.5 sectors per request
      is already within 5% of the geometric best case for this layout.
- [ ] 20.4 The half-precision Hessian is therefore the only remaining
      double-digit lever: a `half2` atomic stores two entries per transaction,
      which takes the bending element from 102 atomics (4 blocks x 9 + 6 x 9 +
      4 x 3) to 42 (4 x 3 for the symmetric diagonal blocks + 6 x 5 + 4 x 2),
      a 59% cut in the sector count that dominates both kernels. It also cuts
      the matrix working set enough to fit the ~4 MB of L2. It touches the
      `Mat3` storage and every writer and reader of `Jx`/`Jx_diag`, and it
      needs a magnitude check (half saturates at 65504, and the stitch
      stiffness alone is 4e4), so it is a dedicated block rather than a
      drive-by change.
- [ ] 20.5 Cheaper intermediate worth trying first: store only the six unique
      entries of the *diagonal* blocks. They are exactly symmetric (the two
      mixed weights are the same product for a == b, so the transpose entries
      are bit-identical), which takes the bending element from 102 to 90
      atomics with no precision loss - at the cost of a layout change for
      `Jx_diag` and its three readers.

## 21. The traversal measured directly (and the stack sized honestly)

- [x] 21.1 Instrumented the self-collision traversal (temporary counters,
      reverted): 158,146 queries, **91.8-102.9 node visits per query**, deepest
      stack occupancy **21 entries**, 5.4k-19.8k pairs emitted per launch.
      99.9% of the traversal is fruitless, and the previous estimates of the
      per-visit cost were 4-8x too optimistic because they assumed ~20 visits.
- [x] 21.2 Sized the traversal stack from that measurement and cut it 64 -> 32
      (256 -> 160 bytes of local frame on `query_ee`, 256 -> 128 on
      `query_vf`): interleaved A/B x1.008, i.e. a wash, with `query_ef` going
      from 33 registers + a 256-byte frame to 86 registers and no frame.
      Reverted.
- [x] 21.3 Conclusion: per-visit cost and per-visit storage are both
      exhausted. The remaining variable is the visit count, which is a tree
      property: ~100 visits against ~40 for a well-clustered tree of this
      size. Two structural options, both rebuild-time:
      a binned-SAH build, or replacing the BVH with a spatial hash grid for
      these uniform small primitives (the repository already has a hash path
      for the point and triangle queries).

## 22. BVH, part two: the tree is not stale, the tree is the tree

- [x] 22.1 The instrumented visit count grows 91.8 -> 102.9 over five frames,
      which points at deformation and makes the 20-frame rebuild interval the
      obvious suspect. Made it a parameter (`bvh_rebuild_interval`) and A/B-ed
      it in one binary: interval 20 -> 9.62 ms min, 5 -> 9.70, **1 -> 12.17**
      (40 frames x 3 rounds, interleaved). Rebuilding every frame is 25%
      slower than every twenty; five is indistinguishable from twenty.
- [x] 22.2 Reverted the parameter (no value at either end of its range) and
      recorded the conclusion: the ~100 visits per query are what this LBVH
      costs for these query boxes. Every tuning-level knob of the broad phase
      - per-visit cost, record layout, alignment, packing, stack size, rebuild
      cadence, stream overlap - has now been measured, and only the stream
      overlap ever moved the frame.
- [ ] 22.3 The open item is a different tree, and it is an algorithm change
      rather than a tuning change: a binned-SAH top-down build (split chosen
      by surface area, ~12-16 bins, one work-queue pass per rebuild) or a
      spatial hash grid for these uniform small primitives. Both are
      rebuild-time work that does not touch the per-frame path.

## 23. Half-precision assembly: checked before writing, and rejected

- [x] 23.1 Measured the dynamic range this matrix actually has. One diagonal
      3x3 entry accumulates bending (`bending_k` 1e-2), membrane FEM (~1e2),
      mask stiffness (2e3) and stitch stiffness (`sewing_k` 4e4) - six orders
      of magnitude in a single accumulator, against `half`'s ~3 decimal digits.
- [x] 23.2 Simulated the accumulation offline (200 trials, 0.5-1.5 jitter per
      contribution): float32 gives 2.6e-8 median relative error; **float16
      overflows to inf** (two 4e4 contributions exceed its 65504 ceiling).
      Scaling the matrix down to avoid the overflow - free to undo in the
      matvec - puts the bending term (3.5e-7 of the entry) below the
      accumulator's resolution, i.e. the change would delete the bending
      coupling wherever a stitch or mask term is present.
- [x] 23.3 Rejected as a drop-in port. That shape only pays off when units or
      block scaling keep the entries in a much narrower band than this
      operator's do; copying the representation without that is a silent
      change to the operator.
- [ ] 23.4 If the assembly is to be attacked anyway, the safe lever is the one
      with no numeric risk: the four diagonal blocks are exactly symmetric
      (the two mixed weights are the same product when a == b, so the
      transpose entries are bit-identical), so storing six unique entries
      instead of nine takes a bending element from 102 atomic adds to 90 -
      about 4% of the frame - at the cost of a layout change for `Jx_diag`
      and its three readers.

## 24. BVH, part three: the tree is within 1.5x of ideal, the query is the cost

- [x] 24.1 Split the traversal's visits (temporary counters): **51 internal-node
      tests and 14 leaf visits per query**, 0.1 pairs emitted. Fourteen leaf
      visits is what a 10-20 mm query box over ~5 mm mesh spacing contains, so
      the leaf count is geometry, not tree quality; 51 tests for 14 leaves is
      3.6 per leaf, ordinary for a binary tree over 158 k primitives.
- [x] 24.2 Compared against the open-source reference's in-tree BVH test
      (`src/test/bvh/bvh.cu` in the Newton checkout: 64-bit Morton keys, Karras, and
      `mark_packed_leaf_nodes` with `leaf_size=8`). Leaf packing removes ~9 of
      the 51 internal tests by shortening the tree three levels but makes each
      of the 14 leaf visits evaluate up to eight primitives, so the two effects
      nearly cancel for a query whose cost tracks the number of overlapping
      leaves. Not implemented; the arithmetic does not support it.
- [x] 24.3 Recorded the code-only result instead: at the *unchanged* study-1
      parameter block, the current sources are x1.39-1.41 faster than the
      original ones (29.57 -> 21.30 ms min, RTS 0.101 -> 0.141). The study-1
      notebook block is restored to `pd_iters=10`, `linear_iters=5`.
- [ ] 24.4 The two ways left to beat the query volume, both larger work:
      batching spatially adjacent queries into one traversal (the SIMT warp
      already shares the instruction stream across 32 Morton-adjacent edges -
      measured 7.8 sectors per warp load - so the marginal gain is the node
      list, not the instruction stream), or a spatial hash grid.
      -> The hash grid is **withdrawn**: the repository already contains a
      point-hash broad phase, which is what the BVH replaced, so it is a
      known-losing design here and only worth revisiting if it can be shown
      faster than the tree, not assumed faster.

## 25. Half precision on the couplings: implemented, and it diverges

- [x] 25.1 Implemented the half-coupling shape for real - float diagonal, half
      couplings (`Mat3h`, five `__half2` atoms per block, 20-byte records).
      sm_86 emits one `ATOM.E.ADD.F16x2` per half2; the bending kernel's
      disassembly shows 30 F16x2 atoms (6 coupling blocks x 5) + 48 float
      reductions = 78 atomic instructions instead of 102, same 128 registers,
      no spills.
- [x] 25.2 It diverges to NaN in three frames. The solver's debug dump shows
      the reason: diagonal blocks ~4.5e4 (`sewing_k` 4e4 penalty), coupling
      entries up to +/-1.2e4, smallest meaningful coupling ~1e-4. Six
      couplings per entry exceed half's 65504 ceiling, and the range that has
      to coexist in one accumulator is eight orders of magnitude against
      half's ~5 exponent orders and 3-decimal mantissa. No single scale fixes
      both ends (scaling down pushes 1e-4 into subnormals). Reverted.
- [x] 25.3 The finding that matters: our matrix is **penalty-dominated**, theirs
      must be **projection-dominated**. The 4e4 stitch spring is deliberate
      (a closed seam must still carry load for the tearing model), so matching
      their half storage means first matching their constraint formulation -
      projected constraints keep the operator O(1)-O(1e2), which is also why
      their solver needs fewer iterations. Storage type is the last thing to
      copy, not the first.

## 26. The penalty cannot be relaxed, and the query order is proven

- [x] 26.1 Tested whether the 4e4 stitch penalty is only a numerical
      convenience (which would have made the matrix half-friendly). 150
      frames, same budget, only `sewing_k` changed: 40000 -> worst seam gap
      2.11 mm, 4000 -> 13.77 mm, **400 -> 149.76 mm**, with no time saved
      (22.29 / 22.63 / 23.92 ms). The penalty carries the seam load, so it
      stays, so the matrix stays penalty-dominated, so hard-constraint
      formulation (projection or elimination) is the prerequisite for both
      half precision *and* fewer iterations.
- [x] 26.2 Isolated the project's own query-ordering optimisation by making it
      a switch (`bvh_query_order`, default 1 = i-th leaf in BVH order, 0 =
      natural edge order). Nsight Compute counters: 7.83 vs 9.12 sectors per
      request, L1 hit 85.1% vs 70.4%, same instruction count. Interleaved A/B
      over the whole frame (40 frames x 4 rounds): 21.42 vs 22.83 ms min,
      23.10 vs 25.00 median, RTS 0.140 vs 0.131 - the mapping is worth
      **x1.07-1.08 of the frame**.
- [x] 26.3 Checked whether another curve would do better. 32 consecutive
      Morton leaves span 48 mm median (155,203 primitives of ~5 mm), which is
      the geometric limit for a space-filling curve; and 7.83 sectors per
      request over a 24-byte record means a warp touches ~10 *distinct nodes*,
      i.e. the residual cost is traversal-stack divergence, not spatial
      spread. A different sort key cannot fix that; a warp-cooperative
      traversal (one shared node list per warp) can, and that is a kernel
      restructure rather than an ordering change.

## 27. The stitch cannot leave the matrix either

- [x] 27.1 The half-precision dead end traces to `sewing_k` (4e4), so the
      obvious fix was to let the projection carry the seam and drop the
      stitch's `k*I` blocks from the matrix while keeping its force. Added
      `sewing_matrix` (default 1) and ran 150 frames with it at 0: **157 of
      975 stitches close, area ratio 95.0, seam gap 6.1 m, 56.8 ms/frame** -
      the garment explodes.
- [x] 27.2 The reason is structural: the projection is gated on
      `check_sewing`, which only declares the seam assembled once the stitches
      are within their threshold, and it is the spring's matrix term that
      brings them there. The penalty assembles the seam *and* carries its
      tension. Reverted.
- [ ] 27.3 A hard-constraint seam is still the right long-term answer, but it
      means redesigning the seam assembly (unconditional projection, or an
      elimination/Lagrange formulation), not removing a term. Both this and
      the half-precision storage depend on that one piece of work.
      **Corrected by 29: the two are independent. The hard-constraint seam
      stands on its own merits (seam closure, convergence); it is not the
      precondition for half-precision storage.**

## 29. The seam is not the half-precision blocker (measured)

28.3's protocol still applies, but this question is arithmetic rather than
throughput, so it is answered by dumping the operator rather than by an A/B.

- [x] 29.1 Temporarily instrumented the solver (reverted) to assemble one
      contribution at a time into an empty operator - contact, sewing, FEM
      membrane, AOGS bending - at a chosen frame and write the diagonal
      (29784 x 9) and coupling (172261 x 9) blocks raw to disk.
- [x] 29.2 At frame 60 with all 975 stitches closed: contact reaches 7.6e6,
      AOGS bending 8.2e4, the stitch penalty 8e4, the FEM membrane 9.8e3,
      against an inertia term of 2.8e2. The stitch writes 5,706 of 268,056
      diagonal entries (2.1%) and is nowhere near the ceiling.
- [x] 29.3 With the stitch term removed from the operator - what a
      hard-constraint seam would do - 211 diagonal entries still exceed
      half's 65504 and the span stays at 9.1 decades. Replaying the
      accumulation in float16 gives 412 overflowing entries with the stitch,
      211 without, and an identical 482 entries above 1% error at scale x1,
      x1/16 and x1/256: a mantissa problem, not a range problem, so no global
      scale fixes both ends. The coupling blocks do not overflow but carry up
      to 24x relative error on 2,481 of 1.55 M entries.
- [x] 29.4 The ceiling is not even stable across models: with `friction_on=0`
      the contact term falls to 4.6e4 and the AOGS bending term rises to
      2.6e7 on the same frame. Half storage would need the dynamic range
      bounded (contact regularisation, degenerate bending geometry), not the
      seam reformulated.
- [x] 29.5 Recorded the consequence for the report: half-precision assembly
      stays closed, and RTS work should look at the operator's conditioning,
      where the extremes are quantities the code chooses.

## 30. The Projective-Dynamics iteration as a CUDA graph (kept)

Re-opened 18.4. The measurement there was right about synchronisation and
wrong about what the 26% non-kernel gap was made of.

- [x] 30.1 Priced a launch directly: 690 dependent tiny kernels on one stream
      cost 8.09 ms/frame (11.7 us each); the same sequence as one 690-node
      CUDA graph costs 1.02 ms (1.5 us per node); as ten 69-node graphs,
      1.11 ms. The gap between kernel time (16.97 ms) and `update()` +
      synchronisation (21.87 ms) is exactly this overhead.
- [x] 30.2 Found the capture restriction before writing the plumbing: on this
      driver `cudaStreamBeginCapture(0, ...)` fails with
      `cudaErrorStreamCaptureUnsupported`, while a stream the program creates
      captures and replays normally. The iteration therefore runs on
      `sim_work_stream()`, forked and joined once per frame with events.
- [x] 30.3 Cleared the capture blockers: the PCG solve's per-solve blocking
      residual readback became a sticky device flag consumed once per frame,
      the solver's mid-loop diagnostics are skipped while capturing, and
      every launch inside the iteration carries the iteration's stream.
      Nothing here changes the arithmetic.
- [x] 30.4 Replay validity is keyed on the pointers, mesh sizes, models and
      host parameters baked into the captured arguments, including a
      `Simulator` parameter-version counter; a key change rebuilds the
      capture, and a capture failure disables the path and falls back to
      direct launches. `pd_cuda_graph=0` disables it explicitly.
- [x] 30.5 Interleaved whole-frame A/B, study-1, 40 frames x 4 rounds, one
      module: 26.27 ms min / RTS 0.114 without the graph against 21.88 ms /
      0.137 with it (x1.20 min, x1.23 median), every round the same sign.
      `update()` + synchronisation in the same session: 26.62 -> 23.10 ms.
- [x] 30.6 Accepted across five dataset elements, 150 frames each, at
      `pd_iters=10` / `linear_iters=5`: x1.15, x1.16, x1.34, x1.53, x1.92,
      with area within 0.004, seam gaps within 0.06 mm and body clearance
      positive in every pair. The smallest element goes from RTS 0.24 to
      0.46. The fixed ~4 ms of launch overhead is the reason the gain grows
      as the scene shrinks.
- [ ] 30.7 Not captured yet: the broad phase and the frame bookkeeping (~20
      launches/frame, so little left), and the seam projection, whose
      host-side ramp changes per frame. Capturing the whole frame would need
      those decisions moved onto the device.

## 31. Lazy bending Jacobian (measured, then removed)

- [x] 31.1 Measured the replaceable half first: the bending kernel costs
      379 us per call, of which the element evaluation and the forces are
      97 us and the matrix assembly 282 us; the restore path (two device
      copies of 7 MB) costs ~20 us, so a reused block turns 379 us into
      ~117 us.
- [x] 31.2 Implemented `bend_freeze_after` (default -1): the block is
      assembled once at iteration k into saved arrays and added to the
      operator; later iterations add it back and ask the element kernel for
      forces only. Verified the early divergence of the earlier freeze-all
      attempt does not return.
- [x] 31.3 Found and fixed two ordering hazards while doing it: the saved block
      must be added *after* `truncate_forces_kernel` (a clamp scales whatever
      diagonal is already there), and the save pass must be matrix-only, or
      the clamp sees a contact-plus-bending magnitude on that one iteration.
      Both are recorded in `design.md`.
- [x] 31.4 Interleaved whole-frame A/B, 40 frames x 4 rounds, whole-loop graph
      on: exact 20.74 ms min / 22.45 median (RTS 0.145) against
      freeze-after-3 **19.19 / 20.28 (RTS 0.156)** - x1.08 min, x1.11 median.
- [x] 31.5 Accepted across five elements, 150 frames, graph on, exact against
      lazy-at-3: x1.11, x1.09, x1.08, x1.06 on the four larger elements and
      x0.92 (within noise) on the smallest, with area within 0.005, seam gaps
      within 0.05 mm and body clearance positive in every pair.
- [x] 31.6 Recorded the bug this line produced: the diagonal add was once
      sized by `n` after `n` had become the bend-element count, so it walked
      4.5 MB past `Jx_diag` on study-1 (silent, showed up as a single 0.089 m
      spike) and 4,500 blocks past it on `ELXRPKNR5Z`, where it corrupted the
      solver's work arrays and produced `PCG nan`. Sized explicitly now, and
      the small element is the regression test for it.
- [x] 31.7 **Removed in the merge cleanup.** It is a chord / quasi-Newton
      approximation - it changes the trajectory inside the engine's own
      run-to-run noise floor - and at the shipped operating point
      (`dt = 0.0045`, `pd_iters = 3`) it measures 8.42 ms against 8.30 ms, so
      the win exists only at the heavy end of the iteration budget. Carrying
      the approximation, its per-frame save buffers and its ordering hazards
      for a case the default never uses is not worth the review surface; the
      numbers and the two ordering rules stay in `design.md` for whoever
      wants to revisit it.

## 32. Compatibility with the cloth-plasticity change

- [x] 32.1 Read `openspec/changes/cloth-plasticity` (design D1, D5, D7). The
      plastic update runs once per frame in `Geometry::update_for_frame()`,
      outside the captured region, and `refresh_rest_dependent_state()`
      re-derives `areas`, `bend_valid`, `bend_factor` and the PD diagonal.
- [x] 32.2 Confirmed the captured loop needs no change for it: rest values and
      `static_diags` are read from device memory at replay time, not baked
      into the graph, and the per-element `bend_valid` branching is
      device-side.
- [x] 32.3 Hardened the replay key against the one thing that would break it -
      a buffer moving or resizing - by keying every array the loop reads on
      both its pointer and its size. A rest-shape refresh that reallocated
      would now recapture rather than replay against freed memory.
- [x] 32.4 Recorded the one coupling: the lazy bending block is cached per
      frame, and the plastic state changes between frames, so the cache is
      rebuilt from the new rest angles every frame. If the update ever moves
      into `update_for_step` (D5's open question) it still runs before
      `solver->step()`, so the ordering holds.

## 33. Recommended defaults (final sweep)

Every configuration below ran the same 0.45 s of physics with 30 warm-up frames
excluded and two repeats per cell, medians reported, one session.

- [x] 33.1 Established that RTS is nearly linear in `dt` while `step_h = dt`
      keeps one substep per frame: 0.003 -> 0.0045 is x1.5 RTS at an unchanged
      ms/frame, on all four elements tested. `dt` raised without `step_h`
      buys nothing (0.006/0.003 is 19.70 ms against 9.73 ms for 0.006/0.006).
- [x] 33.2 Established that the iteration budget barely moves the quality
      metrics while moving the cost proportionally: area 1.213 / 1.218 / 1.217
      against 19.05 / 11.02 / 8.54 ms at `pd_iters` 10 / 5 / 3.
- [x] 33.3 Priced the quality the larger step costs: study-1 area +3.2% at
      `dt=0.0045` and +5.7% at 0.006 against the 0.003 reference; per element
      at 0.0045, +2.0% (6YGLO1BHYF), +4.9% (ELXRPKNR5Z), +10.5%
      (JAO8WE8XSM). Seams, clearance, penetration counts and per-frame
      displacement stay in their existing bands.
- [x] 33.4 **Recommendation: `dt = step_h = 0.0045`, `pd_iters = 5`,
      `linear_iters = 2`, PCG** - RTS 0.415 on study-1 against 0.132 for the
      notebook's current block (x3.1). Quality-first alternative 0.003 / 5 / 2
      (RTS 0.27), speed-first 0.006 / 3 / 2 (RTS 0.72).
- [x] 33.5 Noted that the lazy bending Jacobian is not part of the
      recommendation (8.42 ms against 8.30 ms at the recommended step), and
      that `tests/harness/presets.py` was left untouched: it is the validated
      harness baseline, which is a separate decision from a shipped default.

## 28. Next: warp-cooperative traversal

The measurement in 26.3 says the residual broad-phase cost is that a warp's
lanes read ~10 distinct node records per load, because their traversal stacks
diverge. The fix is the standard one: a warp walks *one* shared node list and
every lane tests its own query against the node being visited, so each node
load is a broadcast and no lane is idle while another still descends.

- [x] 28.1 Implemented for the self-collision query behind `ee_coop_query`
      (default off), per-lane kernel kept for A/B. Two things had to be right:
      the children must be deduplicated across the warp (`__match_any_sync` +
      lowest-lane leaders), and the shared stack must be sized from the
      measured frontier, which is **21 entries** once deduplicated.
- [x] 28.2 Counters move exactly as designed: sectors per request 7.83 -> 1.08,
      L1 sectors 47.3 M -> 14.5 M, issue active 18% -> 41.5%, active lanes
      14.15 -> 25.5, and the ee kernel goes 2.76 -> 1.95 ms in the steady-state
      timeline (2.20 -> 1.18 ms in Nsight Compute).
- [x] 28.3 **The frame gets slower anyway** - x0.97 overlapped, x0.94 with the
      queries sequential, x0.96 with a 2 KB shared stack. The kernel executes
      2.3x the instructions to save 3.3x the sectors and 0.8 ms of its own
      time, and the frame is throughput-bound, so the extra instructions cost
      more than the latency they hide. Reverted.
- [x] 28.4 Generalised the lesson into the change's measurement protocol: an
      isolated kernel's duration is not the target. Nsight Compute called this
      change 1.86x faster while the frame was 4% slower, and the same trap
      appeared when validating the query order. Every candidate from here on
      is judged by the interleaved whole-frame A/B first, counters second.

## 34. Merge cleanup

The tree now carries only behaviour that earned its place; every removal below
already had its measurement recorded in `design.md`.

- [x] 34.1 Removed the three implemented-but-ineffective pieces: the Chebyshev
      linear solver and its kernels (~150 lines), the `vbd_project_stitches`
      diagnostic, and the lazy bending Jacobian (`bend_freeze_after`) with its
      save buffers and add kernel. The retained optimizations keep their A/B
      parameters (`tight_broad_phase`, `bvh_streams`, `bvh_query_order`,
      `pd_cuda_graph`) because the measurement protocol is built on them.
- [x] 34.2 Re-verified that the removal changes nothing by default:
      interleaved A/B of the pre-cleanup and post-cleanup modules, 40 frames x
      3 rounds at the shipped configuration - 9.83 ms against 9.68 ms minimum
      (x0.985, inside noise), both finite with the same displacement
      statistics.
- [x] 34.3 Test gates after the cleanup: `pytest -m quick` 12 passed,
      `pytest -m sim` 2 passed / 4 skipped / 1 xfailed, and the frontend
      scripted runs (procedural scene and a dataset element) step normally.
- [x] 34.4 Reorganised the change for review: `proposal.md` describes what
      shipped instead of the original broad-phase-only scope, `design.md` now
      opens with the retained/removed summary, and the delta spec is the
      `real-time-performance` capability - the change outgrew
      `broadphase-performance` once the frame's scheduling and its defaults
      became part of it.
- [x] 34.5 Status of the remaining unchecked boxes in this file: they are
      future work, not unfinished parts of this change - dual-tree/SAH/packed
      BVH work, half-precision nodes, a Schwarz-style preconditioner, the
      seam-as-hard-constraint redesign, and the decisions that belong to the
      caller or the product (`pd_iters`, and the same question for the shipped
      defaults).

## 16. End-to-end check after the round

- [x] 16.1 Interleaved A/B against the unmodified sources built by the same
      toolchain, both sides at PCG x2: base 32.81 ms min / 33.21 median
      (RTS 0.091) against the current module 23.75 / 23.78 (RTS 0.126) -
      x1.38 on this (warm) session. Earlier cooler sessions measured x1.72 on
      the same comparison, so the ratio is the stable number and the absolute
      frame time is not.
