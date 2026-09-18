# Tasks: solver-damping

## 1. Pin the baseline

- [x] 1.1 Record the resting-stability baseline of the reported garment scene
      (tail-window median and max per-substep displacement, coherence, finite
      check) and of the undriven grid (RMS velocity decay), with the current
      binary and the shipped parameter block.
      Recorded (garment scene, 1200 substeps at 4.5 ms): window median
      0.21-0.24 mm/substep, per-substep maximum 1.1-1.8 mm, coherence 0.20-0.45,
      **chatter component (per-vertex deviation from the mean displacement)
      0.210-0.233 mm/substep and flat over the whole window**, no non-finite
      frame. Undriven grid, gravity off: decays (0.63 -> 0.12 mm/substep in
      2.7 s).
      Correction (maintainer report: the visible motion is centimetre scale, not
      sub-millimetre): the per-substep deltas above understate what a viewport
      shows. Re-measured over a 400-substep window: per-vertex excursion from
      the window mean 7.0 mm median / **23.1 mm maximum**, peak-to-peak span
      13.7 / **40.6 mm**, displacement over one ~16 ms display frame 0.88 mm
      median / 6.1 mm maximum, centroid span 7.3 mm. So the cloth wanders by
      centimetres while each individual substep moves sub-millimetre - a slow,
      driven wander, not a fast vibration.
- [x] 1.2 Record the frame budget baseline (solver time per update call) and
      the drape acceptance invariants (finite frames, seam closure, area,
      attached-vertex drift, penetration band) so the change can be compared
      against them.
      Recorded: 22-26 ms per 4.5 ms substep on the garment scene (GPU shared
      with an interactive session, so treat as an order of magnitude);
      `-m quick` 12 passed / 25 deselected in 4.4 s before and after the change.
- [x] 1.3 Record the cost of the frame-end velocity smoothing per pass, to
      confirm the launch-geometry fix (measure `smooth_times` 0 versus 20 on the
      same scene and divide by the iteration count).
      Recorded from three same-batch A/B pairs: 0.02, 0.14 and 0.24 ms per pass.
      The grid-geometry defect is arithmetic, not statistical: 26,211 blocks of
      256 threads were launched for 25,956 points.

## 2. Operator-level damping (Mechanism A)

- [x] 2.1 Decide the damping operator: assembled tangent (preferred) or the
      precomputed spring-diagonal basis. Record the decision and its accuracy
      consequence for `constitutive_model_planar=1` scenes.
      Decided: diagonal (Jacobi) form of the assembled proxy tangent - the
      dashpot coefficient is `alpha * m + beta * trace(Jx_diag + static_diags)`,
      evaluated where the diagonal is already assembled. The precomputed
      spring-diagonal basis is not reused, so the damping operator follows the
      material model actually in use (FEM_BW included). Only the diagonal part
      is used; the per-element (relative-motion) form is left as a follow-up
      because it needs the element kernels to see the previous positions.
- [x] 2.2 Add the stiffness-proportional term to the PDNewton iteration
      assembly (matrix and right-hand side consistent) behind
      `rayleigh_beta`, default `0`.
- [x] 2.3 Add the mass-proportional term behind `rayleigh_alpha`, default `0`.
      Measured inert on this scene (mass term is far below the stiffness term),
      as predicted; kept because it is the correct low-frequency knob.
- [x] 2.4 Add both scalars to the CUDA-graph capture key, and any new buffer to
      `mix_buffer`, so a parameter change cannot replay a stale graph.
      `mix_float(damping_alpha)` and `mix_float(damping_beta)` added to
      `iter_graph_key`; no new buffer is introduced by the diagonal form.
- [x] 2.5 Verify with the unit-level check that a zero value reproduces the
      pre-change result on a fixed scene within tolerance.
      Verified on the garment scene: damping off with the new binary gives
      1.128 mm/substep tail median against 1.132 / 1.158 for the two pre-change
      runs, i.e. within run-to-run noise.

## 3. Contact relative-velocity damping (Mechanism B)

- [x] 3.1 Add the normal dashpot to the vertex-face narrow phase behind
      `vf_damping_k`, semi-implicit on the same Hessian row, repulsive only.
      Implemented (`c/h` on the vertex and barycentric rows, force gated on the
      approaching sign, so the response stays repulsive).
- [x] 3.2 Add the same term to the edge-edge and edge-face narrow phases behind
      `ee_damping_k` and `ef_damping_k`.
      Edge-edge implemented. Edge-face is closed without implementation by
      maintainer decision: that path is `solve_untangling_kernel`, the
      untangling pass, and it is not a resting-contact response - adding
      viscosity there would fight the untangling it exists to do.
- [ ] 3.3 Calibrate `c = 2 * zeta * sqrt(k_contact * m_vertex)` per contact type
      against the resting-stability metric and the penetration band; record the
      chosen values and the sweep.
      Partial: `vf_damping_k = ee_damping_k = 1.0` was measured on the garment
      scene. Alone it changes nothing measurable there (tail median 1.214 mm
      against 1.128 mm with damping off), and combined with the operator damping
      it improves the tail (0.0253 -> 0.0221 mm/substep, chatter 0.0065 ->
      0.0063 mm). A proper per-type calibration with the penetration band is
      still to be done.
- [ ] 3.4 Check that friction is not required for the result: resting-stability
      must still pass with the friction coefficient set to zero.

## 4. Velocity-consistent seam merge (Mechanism C)

- [x] 4.1 Apply the momentum-preserving cluster velocity pass after the
      projection in the PDNewton step (the pass already exists; it is only
      referenced from a comment today), or equivalently make the velocity update
      ignore the projection displacement. Keep the merge envelope-free.
      Done and enabled by default behind `seam_merge_velocity` (1 = on, the
      pre-change behavior is `0`). The merge itself is untouched.
- [ ] 4.2 Add a check that a cluster projected from a known offset does not gain
      kinetic energy and keeps its momentum.
- [x] 4.3 Re-check the assembly window of the reported garment scene: the merge
      must still close the stitch pairs (world gap ~0) while the tail-window
      motion satisfies the resting-stability requirement.
      Verified: the stitch gap stays 0.000 mm through the whole run in every
      variant, and the tail-window motion improves rather than regresses.

## 5. Support changes

- [x] 5.1 Replace the hard-coded end-of-step velocity multiplier with the
      `velocity_damping` parameter, default 0.5, documented as a floor.
- [x] 5.2 Fix the frame-end Laplacian launch geometry to
      `ceil(num_vertices / block)` and confirm the anti-blow-up effect is
      unchanged at the same `smooth_times` value.
      Geometry fixed (`blocksPerGrid` instead of `n + block - 1`). The
      anti-blow-up confirmation is closed without a test by maintainer decision:
      the smoothing is the explicit-solver safety net, and both the explicit and
      XPBD solvers are incomplete, so there is no scene to reproduce it with.
      The smoothing itself is retained unchanged.
- [x] 5.3 Decide whether the frame-end smoothing also runs per substep, and if
      so whether `smooth_times` keeps its current meaning; record either
      decision.
      Decision: keep it frame-level and keep `smooth_times` as-is in this
      change. It stays the anti-blow-up safety net; the dissipation now comes
      from the solver operator.

## 6. Defaults and environment

- [x] 6.1 Choose the default values for `rayleigh_alpha`, `rayleigh_beta` and
      the contact damping parameters so the resting-stability requirement is
      satisfied by the defaults, and document units and safe ranges.
      Decided by maintainer: **ship inert defaults** (`rayleigh_alpha = 0`,
      `rayleigh_beta = 0`, `vf_damping_k = 0`, `ee_damping_k = 0`), and enable
      them from the caller's parameter block. Enabling them by default makes
      every drag interaction jitter, which is the report this change exists to
      fix. `seam_merge_velocity` stays on by default: it removes an injected
      velocity rather than adding a behavior.
      Working values for the reported scene: `rayleigh_alpha = 3`,
      `rayleigh_beta = 3e-5`, `vf_damping_k = ee_damping_k = 1`.
      Safe ranges: `rayleigh_alpha` 0-10 /s, `rayleigh_beta` 1e-6-1e-4 s
      (>=1e-3 makes the cloth read as viscous), contact damping 0-10.
- [x] 6.2 Decide the interaction with strain stiffening: it is measured to
      triple the resting chatter at its new defaults, so it must not ship as a
      silent default while damping is absent. Record the decision (opt-in, or
      a documented ordering constraint between the two changes).
      Decided by maintainer: ship it with the default **off** (`start <= 0`
      disables it). Measured here: with the engine defaults (`start = 0.05`,
      `rate = 2.0`) the reported scene goes from 0.23 mm to 3.30 mm per substep
      of chatter and damping does not recover it (3.03 mm); with the notebook's
      `rate = 0.1` it is a no-op. A strain-limiting projection (Provot) was
      already tried by the maintainer and blows up on scenes that must stretch
      past the limit, so that alternative is closed as well.
- [ ] 6.3 Update the shared parameter presets so a scene that does not set the
      new parameters still gets the documented defaults.

## 7. Verification

- [ ] 7.1 Resting stability: reported garment scene, tail window of at least
      1000 substeps, median per-substep displacement at or below 1 % of the mean
      edge length and not trending upward; gravity-off control decayed by at
      least one order of magnitude within 2 s of simulated time.
- [ ] 7.2 Ring-down: undriven grid, e-folding time within the documented
      maximum and consistent between 1 ms and 4.5 ms substeps.
- [ ] 7.3 Stability: 1 ms, 2 ms, 4.5 ms and 6 ms substeps all finite, no run
      needing a smaller substep than the undamped reference.
- [ ] 7.4 Equilibrium invariance: per-vertex rest-shape difference between
      damping off and defaults within tolerance, penetration band unchanged.
- [ ] 7.5 Contact pair behavior: approaching pair loses relative normal
      velocity without sign reversal and never becomes attractive.
- [ ] 7.6 Invariants and frame budget: drape acceptance set and self-contact
      scenes pass; solver time per update call within 5 % of the baseline.
      Partial: `-m quick` passes (12 passed) and the frame-budget delta is
      +3.5 % on the reported scene / +9 % on the harness grid, with no
      measurable kernel-level delta (kernel report of the local profiling run;
      `profile/` is a machine-local run directory and stays out of the
      repository - the numbers measured from it are recorded in tasks 2.16,
      2.20 and here).
      The drape/GCD invariant run and an idle-GPU timing confirmation are still
      owed, so this stays open.
- [ ] 7.7 Run the standard regression groups (`quick`, `sim`, `algo`, `api`,
      `data`/`drape` as available) and record the result.

## 8. Record the outcome

- [x] 8.1 Record the final parameter values, the measured before/after numbers,
      and the mechanisms that were implemented versus closed without
      implementation, in this file.

      Implemented in this pass:

      | mechanism | parameter (default) | state |
      |---|---|---|
      | operator damping, stiffness-proportional (diagonal) | `rayleigh_beta` (0) | implemented |
      | operator damping, mass-proportional | `rayleigh_alpha` (0) | implemented |
      | velocity floor (was hard-coded 0.5) | `velocity_damping` (0.5) | implemented |
      | contact normal dashpot, vertex-face | `vf_damping_k` (0) | implemented |
      | contact normal dashpot, edge-edge | `ee_damping_k` (0) | implemented |
      | contact normal dashpot, edge-face | `ef_damping_k` | not implemented yet |
      | seam cluster velocity consistency | `seam_merge_velocity` (1) | implemented, on by default |
      | frame-end smoothing launch geometry | - | fixed |

      Measured on the reported garment scene (1200 substeps, 4.5 ms substep,
      tail-window values):

      | configuration | median | max | chatter | coherence |
      |---|---|---|---|---|
      | damping off (pre-change behavior) | 0.231 mm | 0.273 mm | 0.233 mm | 0.196 |
      | `rayleigh_alpha=3, rayleigh_beta=1e-5` | 0.025 mm | 0.025 mm | ~0.006 mm | 0.914 |
      | `rayleigh_alpha=3, rayleigh_beta=3e-5` | 0.020 mm | 0.020 mm | ~0.006 mm | 0.898 |
      | `rayleigh_beta=3e-5` + `vf/ee_damping_k=1` | 0.022 mm | 0.022 mm | 0.0063 mm | 0.931 |

      ~~The chatter component drops by about 35x and decays over the window
      instead of holding flat.~~ **Retracted:** that measurement was taken with
      the per-vertex drag form of the stiffness term, which resisted rigid-body
      motion and froze the cloth rather than damping it (see task 2.6). The
      "settled" reading was a frozen garment.
      `-m quick` passes with the change (12 passed, 25 deselected).

## 2b. Defect found by maintainer testing (fixed)

- [x] 2.6 The first implementation accumulated the stiffness-proportional term
      as a per-vertex drag, `c_i = beta * trace(H_i)`, applied to the vertex's
      own velocity. Maintainer test: after dragging a panel the hem hangs in the
      air and never falls, no matter how large the fabric mass is; setting
      `rayleigh_beta = 0` restores the fall.
      Root cause: a nodal `c * v` resists *absolute* motion, so it damps free
      fall as well as the mesh modes. At this mass scale the inertia term is
      `m/h^2 ~ 0.02` against `trace(H) ~ 1e4`, so `c = beta * trace(H) ~ 0.3`
      N.s/m makes gravity's terminal velocity `m*g/c ~ 1.6e-5` m/s - frozen -
      and because that velocity is proportional to `m`, raising the mass does
      not rescue it.
      Fix: accumulate the term as a relative operator instead - `beta/h` times
      the projective-dynamics spring basis, force on a vertex
      `sum_edges w * (v_neighbour - v_self)`, lumped diagonal `sum_edges w`.
      That vanishes for rigid-body motion. Verified: the garment's centroid
      falls with `rayleigh_beta = 3e-5` exactly as it does with `beta = 0`
      (z 0.906 -> 0.891 over 1200 substeps in both).
      Consequence for the earlier numbers: with the corrected operator,
      `rayleigh_beta = 3e-5` has no measurable effect on the reported scene's
      chatter (0.21-0.24 mm/substep either way), and `rayleigh_alpha` is inert
      because the mass term is negligible. The working values quoted in task 6.1
      are therefore not yet demonstrated to fix the report.

- [x] 2.7 Identify the mechanism that actually sets the residual motion.
      **It is a steady creep, not an oscillation.** Per-vertex direction
      reversal rate over a 400-substep tail window: median **0.008**, p90 0.093
      - a vertex moves the same way for 100+ substeps before it may reverse. The
      reported "keeps shaking" is therefore a slow, persistent slide of the
      garment over the body (median per-vertex excursion 9.4 mm, maximum 25 mm,
      centroid drift 9.9 mm in 1.8 s), throttled in speed by the trajectory
      clamp.
      Evidence that rules out the solver-side mechanisms:

      | variant | display-frame motion (median) | excursion (median/max) |
      |---|---|---|
      | shipped | 0.94 mm | 9.4 / 25.0 mm |
      | stiffness (base_spring_stiffness x5) | no change | no change |
      | mass x1000 | no change | no change |
      | relative damping beta=3e-5 | no change | no change |
      | contact dashpot vf/ee=1 | no change | no change |
      | 625 vertices pinned (top 3 cm band) | 0.98 mm | 8.2 / 27.3 mm |
      | friction x20 | 0.75 mm | 5.1 / 22.8 mm |
      | seam projection disabled | 0.90 mm | 8.7 / 31.8 mm |
      | planar model spring-mass instead of FEM_BW | 0.90 mm | 12.6 / 45.4 mm |
      | bending off | 0.87 mm | 8.0 / 30.5 mm |
      | `pd_trajectory_margin` 3e-3 (clamp loosened) | 1.31 mm | 15.2 / 40.0 mm |
      | `pd_step_limit=1` (per-iteration |dx| bound instead of the tube) | 1.41 mm | 14.7 / 49.2 mm |
      | `query_radius` 1e-4 | 0.13 mm | 4.5 / 13.7 mm |

      So the residual motion is a force-balance/slip phenomenon: gravity plus
      contact tangential drive against a friction cap that only ramps up below
      `friction_epsilon * h` (45 um of slip per substep, against 200 um of
      observed slip). Velocity damping cannot remove a steady driving force, and
      the clamp only changes how fast the creep is allowed to run - which is why
      a small `query_radius` reads as "frozen" rather than "settled".
      Next step (not started): instrument the per-vertex tangential drive
      against the friction cap with `check_point_attributes`, then fix the
      friction/contact tangential term rather than adding damping.

- [x] 2.8 New parameters added while separating the two uses of `query_radius`:
      `pd_trajectory_margin` (0 = same as `query_radius`, i.e. shipped behavior)
      and `pd_step_limit` (0 = tube clamp, 1 = per-iteration correction bound).
      Measured above; both default to the shipped behavior.

- [x] 2.9 Creep-band damping added: `creep_damping` (1/s, default 0) applies an
      extra velocity decay only while `|v| < creep_speed` (m/s, default 0.03).
      Rationale: the measured residual is slow (0.05 m/s), and the shipped
      linear floor of 0.5/s is far too weak to remove it, while raising the
      global `velocity_damping` also removes the momentum a real motion needs.
      Measured on the reported scene: `creep_damping=50, creep_speed=0.03`
      takes the display-frame motion from 0.92 to 0.74 mm and keeps the
      post-drag return (mean z falls 18 mm in 5.4 s, versus 20 mm at the shipped
      defaults). `creep_damping=200, creep_speed=0.05` takes it to 0.086 mm
      (10x) but the cloth then climbs instead of returning.

- [ ] 2.10 Locate the generator of the residual wander. Current characterisation
      (all measured on the reported scene, 800 substeps):
      - the motion is smooth and direction-persistent (reversal rate 0.008 per
        substep) but diffusive over seconds: path length ~90 mm versus ~10 mm of
        net displacement over 1.8 s;
      - amplitude scales with the trajectory-envelope radius (linear);
      - insensitive to membrane stiffness (x5), mass (x1000), relative damping,
        contact dashpots, friction (x1 to x1000), 625 pinned vertices, the seam
        projection, the planar model, bending, contact stiffness (x10), contact
        band (1e-3 to 1e-4), rest-shape relief (up to x1.10) and initial-state
        relief (x1.05);
      - reduced by global velocity damping (12x at 50/s) and by more PD
        iterations (5x at 20/10), but both change the post-drag behaviour: with
        either the cloth climbs (contact push-out) instead of returning, i.e.
        the converged solution of this scene is not a static equilibrium.
      Next step: instrument the iteration itself - log per-iteration `|dx|` and
      the force residual for a handful of vertices across one substep, and check
      whether the iterate is converging or cycling. That distinguishes an
      unconverged fixed-point iteration from a genuinely moving equilibrium, and
      it is the last unexplained link.

- [x] 2.11 Convergence sweep against the drag scenario (drag a low triangle up
      20 cm, release, watch 1200 substeps). The result reframes the problem:

      | configuration | display-frame motion | after release (mean z) |
      |---|---|---|
      | shipped 5/2 iterations | 0.92 mm | falls 20 mm |
      | 10/5 iterations | 0.85 mm | climbs 26 mm |
      | 20/10 iterations | 0.18 mm | climbs 53 mm |
      | `velocity_damping=20` | - | climbs 22 mm |
      | `creep_damping=50, creep_speed=0.03` | 0.74 mm | falls 18 mm |
      | `creep_damping=200, creep_speed=0.05` | 0.086 mm | climbs 12 mm |

      Every setting that removes the wander also makes the cloth climb after a
      drag, and the climb grows with how well the solve converges. That is the
      signature of a scene without a static equilibrium: the garment is pressed
      on the body, the contact response keeps pushing it out, and the shipped
      loose solve leaves that push-out incomplete - which reads as a slow
      wander instead. The engine-side fix therefore belongs in the contact /
      equilibrium path (the tangential response of a pressed contact and the
      equilibrium it converges to), not in additional damping.
      The one setting that improves the wander without breaking the return is
      the creep band; it is a mitigation, not a fix.

- [x] 2.12 Solver convergence frontier on the reported scene (600 substeps,
      PCG, `seam_merge_velocity=1`). Area and mean strain are the observable the
      maintainer described as "the cloth does not shrink enough": a converged
      solve shrinks it further.

      | pd_iters / linear_iters | RTS (ms/substep) | area (cm^2) | mean strain | excursion (mm) | display-frame motion (mm) |
      |---|---|---|---|---|---|
      | 5 / 2 (shipped) | 22.7 | 4186.7 | 0.1431 | 10.7 | 0.96 |
      | 5 / 5 | 26.4 | 4016.4 | 0.1181 | 7.1 | 0.86 |
      | 5 / 10 | 28.6 | 3988.5 | 0.1141 | 7.2 | 0.93 |
      | 10 / 5 | 36.0 | 3869.7 | 0.0964 | 6.9 | 0.71 |
      | 10 / 10 | 46.3 | 3831.0 | 0.0905 | 6.9 | 0.72 |
      | 20 / 5 | 59.5 | 3778.9 | 0.0829 | 7.4 | 0.30 |

      Marginal convergence per millisecond: 5/2 -> 5/5 gives -0.025 strain for
      +3.7 ms (0.0068 per ms), 5/5 -> 10/5 gives -0.022 for +9.6 ms (0.0023),
      10/5 -> 20/5 gives -0.0135 for +23.5 ms (0.00057). The first step - raise
      `linear_iters` to 5 and keep `pd_iters` - is an order of magnitude more
      efficient than buying the same convergence with outer iterations, because
      one outer iteration pays the full assembly (~146 us, of which ~113 us is
      the membrane kernel) while one PCG iteration costs about 50 us.
      Recommendation: `linear_iters: 5` as the working value (+16 % RTS for
      -21 % strain, -4.8 % area, -25 % excursion).

- [x] 2.13 Alternative linear solvers measured with the same harness:
      `linear_solver_type = 1` (Jacobi) is unusable on this scene - with 2, 5 and
      10 sweeps the cloth blows up (area 82k / 186k / 139k cm^2, mean strain
      4.7-7.7) and it is slower than PCG (36-155 ms per substep). The system is
      far from diagonally dominant, so a block-Jacobi preconditioner cannot
      serve as the solver. Chebyshev exists in the tree but is not wired to
      `linear_solver_type`, so it needs work before it can be measured.

- [x] 2.14 Launch-count reduction in the PCG iteration: `compute_alpha_kernel`
      and `compute_beta_kernel` were `<<<1,1>>>` launches costing about 2 us each
      of pure launch latency; alpha and beta are now computed inside
      `ite_kernel1` / `ite_kernel2`, removing two launches per linear iteration.
      The measured wall-clock effect is inside run-to-run noise, so treat it as a
      launch-count saving for graph replay rather than a measured speedup.

- [ ] 2.15 Next converged-cost target: the per-outer-iteration assembly. With
      the linear solve at ~50 us per iteration against ~146 us of assembly, the
      affordable route to more nonlinear convergence is a cheaper assembly
      (profile: `compute_BW_FEM` 113 us, L1-throughput bound at 74 % of peak,
      83 registers, 16 % occupancy, nine atomic updates per triangle).
      Candidates to measure next: a chord / modified-Newton scheme that reuses
      the assembled Hessian for two or three outer iterations (skipping six of
      those nine atomics per triangle), and the disabled subspace acceleration in
      `subspace.cu`, which the maintainer suspects is implemented incorrectly
      rather than being a dead end.

- [x] 2.16 Full-frame kernel breakdown (nsys, harness grid, 20 substeps,
      `pd_cuda_graph=0`, per substep):

      | kernel | us/substep | share |
      |---|---|---|
      | `compute_dihedral_bending_AOGS` | 1643 | 32.0 % |
      | `query_ee_pairs_capsule_kernel` | 715 | 13.9 % |
      | `compute_BW_FEM` | 694 | 13.5 % |
      | `query_ef_pairs_kernel` | 415 | 8.1 % |
      | `A_mul_x_offdiag_kernel` (SpMV) | 330 | 6.4 % |
      | `query_vf_pairs_capsule_kernel` | 228 | 4.4 % |
      | `solve_untangling_kernel` | 200 | 3.9 % |
      | cub dot reductions | 187 | 3.6 % |
      | `prepare_linear_step_kernel` | 97 | 1.9 % |
      | `compute_vf_force` | 96 | 1.9 % |
      | BVH refits (edge + face) | 150 | 2.9 % |
      | `Jx_mult_x_diag_kernel` (SpMV) | 51 | 1.0 % |
      | `ite_kernel1` + `ite_kernel2` + `before_ite` | 92 | 1.8 % |
      | everything else | ~200 | ~4 % |

      Total GPU kernel time 4.97 ms against a 5.78 ms wall per substep: this
      scene is kernel-bound, not launch-bound.
      Reading: the **linear solver is about 10 % of the frame** (SpMV 7.4 %,
      dots 3.6 %, vector updates 1.8 %); the assembly is 73 % (bending 32 %,
      broad-phase queries 27 %, membrane 13.5 %). The queries and BVH refits run
      once per substep, so the per-outer-iteration cost is roughly
      `bending 330 us + membrane 139 us + linear solve 134 us`.
      Consequence for the library question: `cusparseSpMV` in BSR form targets
      the 17.6 us per `A_mul_x_offdiag` call (35 us per PCG iteration with the
      diagonal part), so it can buy only a few percent of the frame directly -
      its real value is making each PCG iteration cheap enough that more of them
      fit. A stronger preconditioner (IC(0), a polynomial sweep or an AMG cycle)
      pays off only if it removes *outer* iterations, because one outer iteration
      costs ~470 us of assembly against ~50 us for one PCG iteration.
      The single most expensive kernel in the loop is the AOGS bending assembly
      (330 us per outer iteration), so a chord / modified-Newton scheme reusing
      the bending and membrane Hessian across two or three outer iterations
      attacks the largest remaining term.

- [x] 2.17 Chord / modified Newton implemented as `pd_hessian_every` (default 1
      = the previous behavior). The assembled element/contact Hessian lives in
      its own buffer (`Jx_diag_assembled`) so a skipped assembly can be reused
      without re-adding the fixed projective diagonal or the per-iteration
      damping diagonal; every force term is still evaluated at the current
      positions, so the substep fixed point is unchanged and only the Newton
      curvature is stale.

      Measured on the reported scene (600 substeps, `linear_iters=5`):

      | config | RTS (ms) | area (cm^2) | mean strain | excursion (mm) | display-frame motion (mm) |
      |---|---|---|---|---|---|
      | 5 outer, assemble every iteration | 19.0 / 18.99 | 3998 | 0.1155 | 7.5 | 0.91 |
      | 5 outer, assemble every 2nd | 24.4 | 3884 | 0.0985 | 6.7 | 0.73 |
      | 5 outer, assemble every 3rd | 31.3 | 3840 | 0.0919 | 6.2 | 0.57 |
      | 10 outer, assemble every iteration | 32.6 (repeat 35.1) | 3871 (3866) | 0.0966 (0.0958) | 6.8 | 0.72 |
      | 10 outer, assemble every 2nd | 24.4 (repeat 32.1) | 3884 (3894) | 0.0985 (0.0999) | 6.3 | 0.74 |

      At equal outer-iteration count (10) the chord run reaches the same drape
      (area within 0.7 %, strain within 4 %) at 9-25 % lower cost across the two
      batches - the spread is GPU noise from the shared session, the direction is
      consistent. Read the other way: at equal cost the chord scheme affords
      about 1.4x more outer iterations, which is exactly what buys the shrink and
      the reduction in residual wander (display-frame motion 0.96 -> 0.74 mm,
      -23 %, against the shipped 5/2 default).
      Recommended working set for the frontend so far: `pd_iters: 10`,
      `linear_iters: 5`, `pd_hessian_every: 2`.

- [x] 2.18 Build-system trap found while landing 2.17: the MSBuild CUDA rules do
      not recompile every translation unit that includes a changed `.cuh`.
      `solver_PDNewton.cuh` gained a member, only `solver_PDNewton.obj` was
      rebuilt, and `simulator.obj` / `subspace.obj` kept the old class layout -
      an ABI mismatch that produced an access violation in `nvcuda64.dll` on the
      first update. Touching the includers (`simulator.cu`, `subspace.cu`)
      before rebuilding fixes it. Record this in the machine-local build notes;
      any change to a solver header must touch every `.cu` that includes it.
- [x] 2.19 Residual observability added and the measured-ineffective knobs
      removed (maintainer request: keep the code honest).

      New API: `qydp.simulator.get_residual_metrics()` returns
      `{newton_initial, newton_final, newton_relative, linear_initial,
      linear_final, linear_relative}` for the last substep - the squared norm of
      the force residual at the first and last outer iteration, and the
      preconditioned residual of the last linear solve with its ratio.

      Removed as measured-ineffective:

      | removed | evidence |
      |---|---|
      | `rayleigh_alpha`, `rayleigh_beta` (and `accumulate_relative_damping`) | once the nodal-drag form was replaced by the relative operator, beta was inert below the level that freezes free fall, and alpha is inert because the mass term is negligible |
      | `vf_damping_k`, `ee_damping_k` (contact normal dashpots) | on the reported scene they change nothing alone (tail median 1.214 vs 1.128 mm) and add ~0.4 % to the tail when combined |
      | `pd_step_limit` | per-iteration \|dx\| bound: display-frame motion 1.41 mm vs 0.96 mm shipped |
      | `pd_trajectory_margin` | loosening the clamp increases the wander (0.91 -> 1.21 mm) and does not improve the Newton residual |

      Kept: `pd_hessian_every` (chord, measured win), `velocity_damping` (it
      parameterizes what used to be hard-coded), `creep_damping` / `creep_speed`
      (measured win at the reported scene's scale), `seam_merge_velocity`
      (removes the snap velocity injection), the PCG launch-count reduction and
      the Laplacian grid fix.

      Verified after the removal: `-m quick` 12 passed; the shipped path
      reproduces the pre-cleanup numbers within run-to-run spread.

- [x] 2.20 Residual measurement decides where the convergence problem lives
      (reported scene, 600 substeps). RTS here is simulated time over wall time,
      the definition used elsewhere in the repository (LOCAL_DEV.md study-1
      notes: 0.042 with the viewport, 0.18-0.19 for the headless bench), i.e.
      `RTS = 0.0045 / (ms_per_substep / 1000)`:

      | configuration | RTS | wall (ms) | area (cm^2) | strain | newton_relative | linear_relative |
      |---|---|---|---|---|---|---|
      | shipped 5/2 | 0.253 | 17.8 | 4175 | 0.141 | 0.968 | 8.3e-3 |
      | 5/5 | 0.227 | 19.9 | 4002 | 0.116 | 0.983 | 1.6e-5 |
      | chord 10/5 (assemble every 2nd) | 0.166 | 27.1 | 3896 | 0.100 | 0.965 | 1.0e-5 |
      | full 10/5 | 0.138 | 32.6 | 3872 | 0.097 | 0.974 | 1.2e-5 |
      | clamp loosened to 1e-2, 5/5 | 0.173 | 26.0 | 4011 | 0.117 | 0.966 | 1.9e-5 |

      Two hard conclusions:
      1. The linear solve is **not** the bottleneck: 5 PCG iterations reach a
         relative residual of 1e-5, 2 iterations only 8e-3, and 10 iterations add
         nothing (matching the earlier convergence saturation).
      2. The Newton (outer) residual falls by only 1.5-3.6 % per substep, and
         neither more linear iterations (0.985 -> 0.983) nor a looser motion clamp
         (0.981) changes that. Each outer iteration buys about 3 %, which is what
         the area and strain improvements track.
      So the lever is the **accuracy of the operator** the outer iteration
      linearises: it is currently the assembled element/contact Hessian plus the
      *fixed spring-lattice projective diagonal* (`static_diags`, from
      `pd_precompute_spring_forces`), which is inconsistent with the FEM_BW
      material model the scene actually uses. Next experiment: make that fixed
      part consistent with the material model (or drop it and rely on the
      assembled tangent, with `fem_psd_clamp` keeping it definite) and watch
      `newton_relative` - that is the cheap route to more convergence, since it
      adds no iterations at all. Measured in 2.21: the factor is real but
      partial (`newton_relative` 0.965 -> 0.92), not the whole story.
- [x] 2.21 The operator-consistency experiment of 2.20, run behind a new
      parameter `pd_static_diag_scale` (default 1 = the shipped behavior) that
      scales the fixed spring-lattice diagonal `prepare_linear_step_kernel`
      adds on top of the assembled tangent.

      Reported scene, 600 substeps, `pd_iters`/`linear_iters` = 5/2 unless
      noted, two interleaved runs per configuration:

      | `pd_static_diag_scale` | area (cm^2) | strain | newton_relative | linear_relative | display motion (mm) | excursion (mm) |
      |---|---|---|---|---|---|---|
      | 1 (shipped) | 4164 | 0.140 | 0.94-0.97 | 0.008 | 0.95-1.00 | 10.0-10.2 |
      | 0.5 | 4063 | 0.125 | 0.97 | 0.011-0.013 | 0.93 | 7.9 |
      | 0.25 | 4017 | 0.117 | 0.90-0.95 | 0.018-0.020 | 0.92 | 7.5-7.8 |
      | 0 | 4033 | 0.119 | 0.91-0.93 | 0.042-0.051 | 0.95 | 7.3-7.6 |
      | 2 | 4492 | 0.187 | 0.97-0.99 | 0.003 | 0.97-0.98 | 15.7-17.4 |

      The response is monotone in the scale, so the fixed diagonal is not
      neutral: it is a regulariser of the same order as the assembled diagonal,
      and the shipped scale of 1 buys 3 % less strain, 25 % more wander and a
      worse Newton residual than a material-consistent matrix. Dropping it
      degrades the *linear* residual at 2 PCG iterations (8e-3 -> 4-5e-2) - the
      regulariser was also propping up the block-Jacobi preconditioner - so the
      scale-0 matrix needs an affordable `linear_iters`:

      | config (600 substeps) | area | strain | newton_relative | linear_relative | display motion | excursion | stitch gap | non-finite |
      |---|---|---|---|---|---|---|---|---|
      | shipped 5/2, scale 1 | 4099-4152 | 0.130-0.138 | 0.96-0.97 | 0.008-0.011 | 0.93-0.94 | 10.6-11.0 | 0.000 mm | 0 |
      | scale 0, 5/5 | 3801-3808 | 0.086-0.087 | 0.91-0.94 | 0.002-0.003 | 0.60-0.62 | 7.7 | 0.000 mm | 0 |
      | scale 0.25, 5/5 | 3819-3821 | 0.089 | 0.95 | 4-5e-4 | 0.63-0.64 | 6.6-6.7 | 0.000 mm | 0 |
      | scale 0, 10/5 | 3763-3764 | 0.081 | 0.92-0.96 | 0.0012 | 0.40-0.41 | 8.0 | 0.000 mm | 0 |

      The material-consistent matrix with 5 PCG iterations takes the mean
      membrane strain from 0.134 to 0.086 (-36 %), the visible per-display-frame
      motion from 0.94 to 0.60 mm (-35 %) and the wander from 10.8 to 7.7 mm
      (-30 %) with no non-finite frame and the seam still closed. The Newton
      residual improves only 0.965 -> 0.92: a real but partial contributor.
      The wall-clock column is deliberately absent - the machine drifted inside
      the batch (one configuration measured 19.3 ms/substep on its first run and
      28.7 ms on its last), so no RTS claim comes out of this batch.

      Drag check (400 substeps of assembly, a low triangle lifted 0.2 m,
      released, 400 substeps watched): the configurations are not comparable to
      each other - the pre-drag state already differs by 3-4 cm of mean height
      between them - so it serves only as a "the hem does not hang in the air"
      check, which every configuration passes (the mean height moves by
      2-10 mm over 1.8 s after release, in both directions).

- [x] 2.22 Where the residual stops: `query_radius` sweep at the
      material-consistent operator (reported scene, 400 substeps, scale 0,
      5/5, all rows from the same window so they are comparable to each other):

      | `query_radius` | newton_relative | linear_relative | area | strain | display motion | excursion |
      |---|---|---|---|---|---|---|
      | 1e-4 | 0.943 | 6.8e-4 | 3861 | 0.0946 | 0.20 mm | 14.2 mm |
      | 1e-3 | 0.918-0.926 | 0.0040 | 3848 | 0.0932 | 0.76 mm | 17.3 mm |
      | 1e-2 | 0.776 | 0.0120 | 3634 | 0.0613 | 0.70 mm | 30.1 mm |

      `clamp_to_trajectory_envelope` caps a vertex to a capsule of radius
      `query_radius` around the segment `pos_prev -> pos_target`, i.e. it bounds
      the *whole* displacement of a substep, not the per-iteration correction.
      With the material-consistent operator the residual is therefore
      clamp-limited: loosening the tube 10x takes `newton_relative` from 0.92 to
      0.78 and the strain from 0.093 to 0.061, while the wander grows from
      17 mm to 30 mm - the same tube both throttles the creep and blocks the
      solve. At the shipped scale of 1 the identical loosening changed nothing
      (task 2.20) because the regulariser had already shrunk the step below the
      tube.
      Consequence: more operator accuracy alone does not stop the motion. What
      still drives it is open - the next instrument is the per-vertex tangential
      drive against the friction cap (2.7's next step, still not built).
      Recommended working set for the reported scene:
      `pd_static_diag_scale: 0`, `linear_iters: 5`, `pd_iters: 10`,
      `pd_hessian_every: 2` (chord - the assembly, not the tube, is what makes
      extra outer iterations expensive). The default stays 1 (inert), per 6.1's
      rule that an existing scene keeps its behavior until its own parameter
      block enables a mechanism.
      **Correction (2.24):** the scale-0 recommendation above is wrong as a
      drop-in. It was read off per-vertex medians; the tail of the distribution
      moves the other way.
- [x] 2.23 The per-vertex force-balance instrument was attempted and does not yet
      answer the tangential-drive question; recording it so the next attempt
      starts from the right units.

      `check_point_attributes` returns `force` (the step's accumulated force
      buffer) and `force_elastic` (its elastic part), and the local probe
      normalises both by `mass * g`. At the shipped material scale that yardstick
      is meaningless: a vertex covers `0.1 kg/m^2 * (3.63 mm)^2 = 1.3e-6 kg`, so
      its weight is `1.3e-5 N` while the membrane forces are of order 1 N - the
      printed ratios are 1e4-1e8 and carry no information. **What does hold:** the
      garment's whole weight is `0.42 m^2 * 0.1 kg/m^2 * 9.8 = 0.41 N` against
      membrane forces four to five orders of magnitude larger, so gravity is
      negligible in this scene.

      That is the missing explanation for the maintainer's earlier "mass has no
      effect" observation (task 2.7's mass x1000 row): at this stiffness even a
      1000x mass leaves gravity 1-2 orders below the membrane force, so the mass
      is not a lever on this motion, and residual forces must be read in units of
      the local elastic force, not of the vertex weight.

      Reading the aggregate residual in those terms: `newton_final` is a squared
      norm over 25,956 vertices, so the RMS residual force after the substep is
      `sqrt(228319 / 25956) = 2.97 N` per vertex, against sampled elastic forces
      of order 0.1-1 N. The substep problem is therefore genuinely unsolved, not
      converged-but-jittering - which is the same statement as 2.22's
      `newton_relative` of 0.9, seen from the force side.

      Still owed: residual per vertex in units of the local elastic force, and
      the tangential contact force against the friction cap (the cap only ramps
      up below `friction_epsilon * h` of slip, 45 um per substep, against the
      observed 200 um).
- [x] 2.24 Maintainer report: "with `pd_static_diag_scale = 0` the cloth keeps
      shaking where it touches the body". Reproduced, and the hypothesis I put
      forward for it (the regulariser is what keeps `M_inv` from being zeroed on
      ill-conditioned rows) is **falsified**.

      Tried and removed (measured ineffective, so the code is gone): row
      counters for zeroed preconditioner rows, non-positive assembled diagonals
      and non-positive matrix diagonals, plus a
      `pd_precond_regularized` switch that built the block-Jacobi preconditioner
      from the regularised diagonal while the matrix used the scaled one. On the
      reported scene every configuration - scale 1, scale 0, scale 0 with the
      regularised preconditioner - reports **zero** degenerate rows, so there is
      no zeroed row to blame, and the switch only improved the linear residual
      (0.045 -> 0.024) without touching the tail chatter (reversal p99 0.621
      against 0.626). Both were deleted; this entry is their only record.

      The report is real, though, and it is a *tail* effect. Per-vertex
      direction-reversal rate over the last 600 substeps, sampled every 8th
      vertex (`QY_CONTACT_CORR` in the local probe), reported at the median and
      the 90th percentile of the sampled population:

      | configuration | strain | area | display motion (median) | reversal p90 | reversal p99 | reversal max |
      |---|---|---|---|---|---|---|
      | scale 1, 5/2 (shipped) | 0.137 | 4144 | 0.97 mm | 0.098 | 0.575 | 0.671 |
      | scale 1, 5/5 | 0.115 | 3997 | 0.93 mm | 0.101 | 0.585 | 0.673 |
      | scale 1, 10/5 | 0.096 | 3868 | 0.72 mm | 0.103 | 0.611 | 0.764 |
      | scale 0, 5/2 | 0.120 | 4039 | 0.94 mm | **0.250** | 0.631 | 0.726 |
      | scale 0, 5/5 | 0.087 | 3811 | 0.64 mm | **0.366** | 0.648 | 0.839 |

      Three readings:
      1. At scale 1, buying convergence with outer iterations (5/2 -> 5/5 ->
         10/5) leaves the tail untouched (0.098 -> 0.101 -> 0.103) while taking
         the strain from 0.137 to 0.096. That is the same drape quality the
         scale-0 configuration reaches (0.087 at 5/5) with **3.5x less tail
         chatter**. The drape improvement does not require the un-regularised
         operator at all.
      2. Scale 0 is what moves the tail: the reversal rate at the 90th percentile
         goes 0.098 -> 0.250 (5/2) and 0.103 -> 0.366 (5/5) against the matched
         scale-1 rows. The median gets quieter (0.97 -> 0.94 mm) while a small
         population starts flipping direction on most substeps - which is what a
         viewport shows as rubbery shaking at the contact.
      3. The trajectory tube is not the cause either: at scale 0, 5/5 the
         reversal p90 is 0.230 at `query_radius = 1e-4`, 0.366 at 1e-3 and 0.374
         at 1e-2 - the chatter survives a 10x looser tube and a 10x tighter one.
         (The same sweep does show that the *drape* depends on the limiter: mean
         strain 0.0875 / 0.0859 / 0.0365 at 1e-4 / 1e-3 / 1e-2, so the state the
         solve settles into is set by the clamp as much as by the material - a
         defect worth its own entry.)

      What is left as the mechanism: a small population of vertices in active
      contact whose force balance flips sign every substep. It is present in the
      shipped configuration too (reversal p99 0.58, max 0.67) and gets larger
      when the Newton step at those rows becomes sharper, which is what removing
      the regulariser does. Naming it needs the contact-state instrument that
      2.23 still owes (per-substep normal sign and the friction cap branch), not
      another operator knob.

      Corrected recommendation for the reported scene: keep
      `pd_static_diag_scale` at 1 and buy convergence with `pd_iters` (10/5, or
      the chord 10/5 with `pd_hessian_every: 2` for the cost). Scale 0 stays
      available for experiments but is not a drop-in improvement.
- [ ] 2.25 Maintainer decision: preconditioners first (S4). Audited
      `subspace.cu`, rebuilt it as a two-level preconditioner, and measured it.
      **It is not usable in this form** - recorded here with the numbers, and the
      code is left selectable (`linear_solver_type: 2`) but inert by default so
      the decision "fix it further or delete it" can be made without redoing the
      work.

      Audit of the disabled path (all three are real):
      1. `build_basis_kernel` left `basis_indices[vertex_offset + idx]`
         uninitialised for every stencil slot outside the panel grid, and
         `basis_to_new_index_kernel` then indexes `basis_new_indices` with that
         garbage value. Fixed (write 0; the weight is 0 so the slot is masked).
      2. The coarse solver was a `SolverPCG` initialised with
         `use_preconditioner = false` *and* its `M_inv` was never filled, so the
         coarse solve ran unpreconditioned.
      3. `solve_subspace` **replaced** the fine solution with the prolongated
         coarse one (`dx = P dy`) instead of adding a correction, and the coarse
         operator was the PD spring scalar rather than the operator being solved.
         Deleted together with its members (`subspace_rhs`, `subspace_dy`,
         `subspace_solver`) and the transfer kernels only it used
         (`restrict_kernel`, `prolongate_kernel`).

      Rebuilt as `SolverTwoLevel` (additive:
      `z = M_fine^-1 r + omega P (P^T A P)^-1 P^T r`, coarse solve by a short
      PCG with the coarse diagonal as its preconditioner, `subspace_omega`
      default 1, `subspace_coarse_iters` default 3). Two structural fixes were
      needed to stop the NaNs: the coarse Laplacian's kernel is the constant
      (rigid-translation) mode and the mass term only lifts it by ~10 against a
      stiffness of ~2.4e4, so a net force residual was answered by a large
      global translation - the mode is now projected out of both the coarse
      right-hand side and the coarse solution.

      Reported scene, 600 substeps, AOGS, scale 1 (`linear_plain_relative` is the
      unpreconditioned residual ratio, added to `get_residual_metrics` because the
      preconditioned ratio is not comparable between preconditioners):

      | config | ms | area | strain | newton_relative | display motion | reversal p90 |
      |---|---|---|---|---|---|---|
      | PCG 5/2 (shipped) | 23.2 | 4220 | 0.148 | 1.000 | 0.94 mm | 0.093 |
      | PCG 5/5 | 26.9 | 4019 | 0.118 | 0.968 | 0.85 mm | 0.113 |
      | two-level, omega 0.2, 5/2 | 31.1 | 3930 | 0.106 | **0.604** | 1.33 mm | **0.392** |
      | two-level, omega 0.2, 5/2, 5 coarse iterations | 32.7 | 3965 | 0.111 | **2.986** | 1.40 mm | 0.317 |
      | two-level, omega 0.4, 5/5 | - | - | - | PCG NaN | - | - |

      Reading: the correction does accelerate the *nonlinear* residual (0.604
      against 1.000 - the substep problem is genuinely solved ~20x further), but
      it is unstable (omega >= 0.4 NaNs, and omega 0.2 also NaNs on some runs -
      two of the four configurations of a separate 200-substep batch died), it
      makes the visible motion worse (1.33 mm and reversal p90 0.392 against
      0.94/0.093 shipped - the same tail-chatter signature as the un-regularised
      operator), and it costs 8-10 ms/substep more than the linear solve it is
      meant to accelerate.

      Why it cannot pay in this shape: (a) the coarse operator is reduced from
      the PD spring scalar, not from the assembled operator the fine solve uses;
      (b) the coarse solve is a truncated CG, so the preconditioner is an inexact
      non-symmetric operator - outside CG's assumptions; (c) the linear solve is
      only ~10 % of the frame (2.16), while the nested coarse solve with its
      reductions costs more than that. **Conclusion: the preconditioner work
      depends on the constant-matrix work (2.15/S1-S2) - on a rest-shape constant
      operator the coarse matrix is constant, can be factorised once instead of
      solved by a nested CG, and can be built from the operator actually being
      solved.** Recommended order: S1/S2 first, then revisit the preconditioner.
- [x] 2.26 The maintainer pointed at the source of this design: X. Li, Y. Fang,
      L. Lan, H. Wang, Y. Yang, M. Li, C. Jiang, "Subspace-Preconditioned GPU
      Projective Dynamics with Contact for Cloth Simulation", SIGGRAPH Asia 2023
      Conference Papers. Comparing the paper with `subspace.cu` shows the paper
      is not in question - our implementation is a different scheme. Point by
      point:

      | the paper | our engine |
      |---|---|
      | Basis is a per-patch 2D MPM quadratic B-spline grid, and it **satisfies partition of unity** (§4.2) | Same kernel, but the clipped panel-border stencil broke the sum - fixed in this pass (see 2.25) |
      | The reduced matrix is `P^T A P`, **prefactorised with Cholesky and reused** (§4.3), because in PD `A = M/h^2 + L` is the *rest-shape constant* matrix | Our `A` is the tangent assembled **every outer iteration**, so the reduction would have to be rebuilt and is the assembled tangent, not a constant operator |
      | The reduced solve is used as 1) the **exact initial guess** (one backsolve at the start of the substep) and 2) inside a **subspace L-BFGS** (2 iterations per global step, rest-shape reduced matrix as the initial Hessian) (§4.3, Algorithm 1) | Used as an additive, omega-damped correction inside **PCG** - neither an initial guess nor the paper's quasi-Newton scheme |
      | High frequencies come from **5 modified block-Jacobi iterations** per global step, with an analytically tuned step size that guarantees energy decrease (§4.3) | High frequencies come from PCG on the assembled tangent |
      | Contact enters the reduced matrix as `P^T (grad^2 E_contact) P` (§4.4, Eq. 11), tracked progressively by the BFGS updates | Contact is not in the reduced operator at all |
      | Overall: subspace integration coupled with Jacobi-PD inside PD; 23 s/frame for 120K nodes, 6.5x faster than a GPU CIPC solver (Fig. 1) | A Newton/PCG solver with an unrelated damped coarse correction bolted on |

      So the answer to "is the paper wrong?" is no. The measurements of 2.25
      actually support the paper's premise - the coarse correction does move the
      substep residual (0.604 against 1.000) - but the paper's structure is what
      makes it work: a *constant* prefactorised reduced operator, the reduced
      solve as an initial guess, a quasi-Newton history on top of it, and Jacobi
      smoothing for the high frequencies.

      What a faithful implementation needs, in order: (1) the constant PD global
      matrix (`M/h^2 + L_rest`) as the system operator, which is S1/S2; (2) the
      reduced factorisation built once from it; (3) the reduced solve as the
      substep's initial guess instead of a preconditioner term; (4) a subspace
      quasi-Newton history; (5) the contact proxy reduction; (6) a tuned
      block-Jacobi smoother in place of PCG. Item (1) is the prerequisite of all
      the others, which is why the order is S1/S2 before S4.

      Deleted in this pass (measured ineffective or unstable, per the maintainer's
      rule): the two-level experiment (`SolverTwoLevel`, the dense per-panel
      inverse, the `linear_solver_type: 2` wiring, the `apply_preconditioner_extra`
      / `on_assembly` hooks). Kept because they are independently correct: the
      basis-index initialisation and the partition-of-unity normalisation in
      `subspace.cu`, the removal of the dead `solve_subspace` / `SolverSubspace`
      path, and the new `linear_plain_relative` residual metric.
- [x] 2.27 The coarse initial guess, taken as far as it goes and then deleted by
      maintainer decision ("delete it, keep the documentation"). What the two
      follow-up rounds established:

      Implemented (all of it now removed from the tree): the precomputed subspace
      from the rest shape, a dense per-panel inverse of the reduced operator
      built once per substep, and the reduced solve used as the substep's initial
      guess - i.e. the semantics of the original `subspace.cu` and of the paper's
      Algorithm 1 ("Run a reduced-order global step w/o contact for an initial
      guess"). Two new diagnostics came out of it and are kept: the
      unpreconditioned residual triple `linear_plain_initial/final/relative`
      (`r . r`, comparable across preconditioners and across initial guesses) and,
      while it lasted, `coarse_solve_residual` = `|A_c dy - r_c|^2 / |r_c|^2`.

      Defects found on the way, all real and all in this path - the list is the
      useful part of the exercise for any future attempt:

      | defect | consequence |
      |---|---|
      | `build_basis_kernel` never wrote the index of a clipped stencil slot, and `basis_to_new_index_kernel` reads all nine | garbage index into the compaction table |
      | the clipped stencil also broke the partition of unity (`sum w != 1`) | a constant coarse vector no longer maps to a constant displacement, so the rigid mode is not representable |
      | the coarse reduction ran over `edges` only, while the fine operator's rows are `valid_pairs` (natural edges + deduped bending pairs) | the coarse operator was too soft, the coarse solution correspondingly too large |
      | a Gauss-Jordan written as "row-reduce A" returns the identity, not `A^-1` (it needs the augmented identity alongside) | the "inverse" was the identity, so the guess was a prolonged *force*, not a displacement - the numeric signature of the grid-shaped artifacts the path was abandoned for |
      | the diagonal floor was implemented as "add to every row" instead of "raise only missing rows" | the physically meaningful smooth modes were regularised away and the coarse solve became even softer |
      | `Geometry::step_h` is garbage before the first substep (-1.7e16 on this machine) | an inverse built at init time is meaningless |
      | the reduced solve had never been verified | an invalid coarse solve is invisible in the solution norm and only shows up as artifacts |

      Measured verdict (reported scene, 200 substeps, AOGS, notebook parameter
      block), with the coarse solve verified to 0.6-4.9 % relative residual:

      | configuration | `linear_plain_final` | `newton_relative` | `|x0|/|dx|` | area | strain | display motion |
      |---|---|---|---|---|---|---|
      | shipped 5/2, no guess | 1263 | 0.973 | - | 4165 | 0.140 | 0.88 mm |
      | 5/2 with guess | 63169 | 0.931 | 44 | 5750 | 0.357 | 3.07 mm |
      | 5/5, no guess | 4.54 | 0.959 | - | 4102 | 0.131 | 0.76 mm |
      | 5/5 with guess | 3.83 | 0.981 | 147 | 4012 | 0.117 | 1.24 mm |

      So the machinery is correct but the premise is not: block-Jacobi PCG already
      takes the residual down four orders of magnitude in 5 iterations (1263 ->
      4.5 relative to an initial ~1e5), so a coarse start has nothing left to buy
      at 5 iterations, and at the notebook's 2 iterations its overshoot - the
      coarse solution is 44-147x the truncated fine increment - makes the solve
      and the drape worse. This matches the frame profile: the cost is the
      per-iteration assembly (~73 %), not the linear solve (~10 %).

      Closed for now. The preconditioner/subspace direction only becomes
      meaningful on the paper's own premise - a rest-shape *constant* system
      operator with a reduced factorisation, a quasi-Newton history on the reduced
      variables and a tuned Jacobi smoother for the high frequencies - and that
      premise conflicts with the planned plastic rest-shape flow (task 2.24,
      `cloth-plasticity`). If it is revived, the table above is the list of traps
      to avoid, and `coarse_solve_residual` is the check to build first.
- [x] 2.28 Maintainer report: "a flat panel floating in air falls very slowly,
      like it is weightless; gravity is -9.8. Is it the mass unit change or a
      damping bug?" It is the mass unit change, but not a unit error - the
      smaller mass exposed a *pre-existing* operator defect. No damping is
      involved.

      Reproduced headless with a flat grid panel, `ground = 0`, nothing else in
      the scene, one substep per frame, and the closed form `z(t) = z0 - a t^2/2`
      as the reference: measured `a_eff = 0.197 m/s^2` against 9.8, i.e. the
      panel integrates 2 % of gravity. (A separate diagnostic probe, not part of
      the test suite: it builds the `input_data` contract directly and reads
      `check_point_attributes`, which reports the engine's own `pos_prev`,
      `pos_pred`, `pos_world`, `force`, `force_elastic` and `mass`.)

      | configuration | a_eff (m/s^2) |
      |---|---|
      | shipped (mass 0.1 kg/m^2) | 0.197 |
      | `velocity_damping = 0` | 0.624 |
      | `velocity_damping = 5` | 0.062 |
      | mass 0.01 / 0.1 / 1 / 10 / 100 / 1000 kg/m^2 | 0.000 / 0.197 / 1.387 / 5.795 / 8.443 / 8.594 |
      | `base_spring_stiffness = 10` | 8.412 |
      | `base_spring_stiffness = 0` | 8.594 |
      | `pd_static_diag_scale = 0` | 0.652 |

      Readings:
      1. Not damping. Turning `velocity_damping` off only moves 0.197 -> 0.624,
         and increasing it scales the fall down roughly linearly, which is what
         a velocity decay does - it is a second-order effect here.
      2. Mass, and in the direction the unit change went. The fall is correct
         (8.44, the damping-limited ballistic value) at 100 kg/m^2 - the
         pre-`b9d5cde` interpretation of the frontend's `mass = 100` - and it
         degrades smoothly as the cloth gets lighter, i.e. the engine behaves as
         if gravity were divided by the stiffness/mass ratio.
      3. The membrane stiffness is what resists, and it is not a convergence
         problem: with `pd_iters = 1, linear_iters = 20` (one exactly-solved
         Newton step; `linear_plain_relative` = 1e-15) the panel still moves
         0.03 % of the ballistic step, and raising `linear_iters` from 2 to 25
         changes nothing (0.197 -> 0.202).

      The mechanism, read out of the assembled system with a temporary dump of
      `Jx_diag`, `Jx_nondiag`, `valid_pairs`, `static_diags` and `f` (removed
      again; the numbers are for a single triangle, `base_spring_stiffness =
      4e3`, `h = 0.0045`):

      - `static_diags = 7994` at a vertex whose inertia term is
        `m/h^2 = 4.2e-5 / 2.025e-5 = 2.06`. `static_diags` is overwritten with
        `Jx_diag_pd` at the top of every substep (`solver_PDNewton.cu`), and
        `Jx_diag_pd` is the precomputed PD spring-lattice row sum
        `D = sum_e k_e` (~8000 for a 2-edge vertex at k = 4e3). `forward_step`
        only adds `m/h^2` on top of it, so the "fixed projective diagonal" is
        99.97 % lattice stiffness.
      - `D` is added **without its off-diagonal half**. The matching rows
        (`-k_e`, held in `linear->Jx_nondiag_identity` and never assembled; the
        identity-only branch of `A_mul_x` is disabled) would make it a graph
        Laplacian, which annihilates rigid translation. A bare diagonal does the
        opposite: it resists every vertex's own displacement, i.e. it anchors
        each vertex to where it was. Under gravity the fixed point becomes
        `dx = m g / D` per substep instead of `g h^2`.
      - That is exactly the measured signature: the drop is **independent of
        `h`** (3.0e-7 m at both `h = 1 ms` and `h = 4.5 ms`, where the ballistic
        answers differ by 20x), proportional to the mass, and mesh-independent
        (a single triangle and a 2x2 panel both measure 0.128).
      - It also explains why the engine looked healthy before `b9d5cde`: at
        `m/h^2 ~ 5e6 N/m` (the old 100 kg/m^2) the anchor is negligible next to
        the inertia term. At the correct areal density the inertia term is
        `m/h^2 ~ 2-5 N/m`, four orders below `D`.

      Second, independent limiter: the PD iteration clamps every vertex into a
      tube of radius `trajectory_margin = query_radius` around the inertia
      segment (`clamp_to_trajectory_envelope`), so the per-substep displacement
      saturates at ~`query_radius`. Measured terminal fall speed: 0.2230 m/s at
      `query_radius = 1e-3` and 2.246 m/s at 1e-2 against `radius/h` = 0.2222
      and 2.222. This is the mechanism behind the earlier report "`query_radius`
      small and it moves very slowly, like it is frozen": it is a *speed* cap of
      `query_radius / h`, not a damping term.

      Two repair routes were prototyped and measured (both behind parameters,
      both since deleted in favour of the decision below): start the linear
      solve from `q_inertia - q` (the rigid part of the step, exact for any
      iteration budget; free fall 0.197 -> 0.461), and assemble the regulariser
      with its off-diagonal half so it is a Laplacian instead of an anchor
      (0.586, and 8.594 with both). The second one *unmasked what the anchor had
      been hiding*: with the anchor gone the same 5/2 budget leaves
      `linear_plain_relative` = 0.29 and the t1 garment over-stretches (mean
      strain 0.094 -> 0.340, excursion 12 mm -> 225 mm).

      Maintainer decision: the lattice precompute is a leftover of the disabled
      `Jx_nondiag_identity_only` solver flavour, so it is simply not called any
      more - no new mechanism, no new parameter. `pd_precompute_spring_forces`
      (planar.cuh) and its two output buffers are gone from the PDNewton path,
      and `forward_step` now *writes* `static_diags` per substep (the inertia
      term `m/h^2`, or `mask_stiff` for a pinned vertex, 0 for a massless one)
      instead of accumulating into a buffer that the precompute used to
      overwrite. That is a smaller change than either prototype and it fixes the
      model rather than the symptom.

      **Reverted by 2.29.** The removal is not what shipped. It is exact for a
      small mesh but not at the shipping budget on a big one (its own table
      below: 0.606 m/s^2 on the 20x20 grid at 5 outer / 2 linear), so the
      precompute call and the paired off-diagonal are back behind
      `pd_static_diag_offdiag` (default on), and `static_diags` is again copied
      from `Jx_diag_pd` at the top of every substep.

      Measured after the removal (same probes; `a_eff` against 9.8, with the
      `velocity_damping = 0.5` ballistic value at 8.44):

      | free-fall mesh | before | after 5/2 | after, converged |
      |---|---|---|---|
      | single triangle (3 v) | 0.128 | **8.594** | - |
      | 2x2 quad (4 v) | 0.128 | 1.206 | - |
      | 20x20 grid (400 v) | 0.197 | 0.606 | 5.735 (`linear_iters` 10), 8.151 (25), 8.580 (20/20) |

      So the anchor is gone and gravity is correct wherever the substep problem
      is small enough for the truncated CG to resolve the rigid mode (the
      triangle is exact at the shipping 5 outer / 2 linear budget); a big stiff
      panel still lags until the linear solve converges, which is the same
      conclusion as 2.16 - at the new mass the diagonal block-Jacobi
      preconditioner is stiffness-dominated, so the gravity/inertia mode is what
      the iteration budget buys first. The trajectory tube still caps the
      per-substep motion at `query_radius` while the solve lags.

      Reported garment scene (t1, 300 substeps, notebook parameter block):

      | configuration | area | strain (mean) | reversal p90 | excursion (median) | substep | `linear_plain_relative` | tail coherence |
      |---|---|---|---|---|---|---|---|
      | before (anchor) | 3852 cm^2 | 0.094 | 0.070 | 11.9 mm | 24.7 ms | 0.005 | 0.337 |
      | after | 3226 cm^2 | -0.0006 | 0.003 | 134.9 mm | **19.7 ms** | 0.102 | 0.994 |

      The garment is no longer held stretched 9 % past its rest length (it now
      sits at rest length), the direction-reversal chatter at the 90th
      percentile drops 0.070 -> 0.003, the substep gets ~20 % faster, and the
      motion becomes coherent (0.337 -> 0.994) instead of the incoherent
      per-vertex jitter the anchor produced. The larger excursion is the garment
      actually draping/falling for 1.35 s instead of being frozen. The quick
      suite still passes (12 passed) and the standard smoke scene is unchanged
      (mean free displacement 5.5e-10 m, all vertices at the ground clamp): the
      resting sheet is already at its equilibrium, so it does not separate the
      two configurations.

      Defect the removal exposed and that had to be fixed with it: `ite_kernel1`
      computed `alpha = delta_old / d_dot_Ad` and only guarded
      `d_dot_Ad < 0`, so an already-converged residual
      (`delta_old == d_dot_Ad == 0`, which small stiff meshes reach inside the
      outer loop) gave 0/0 = NaN and the frame threw "PCG ended with NaN
      residual"; `ite_kernel2` had the same 0/0 in `beta = delta_new /
      delta_old`. Both directions are now guarded with `<= 0 -> 0`, which is the
      correct zero step for a converged solve.
- [ ] 8.2 Update the spec requirement that still needs a maintainer decision
      (defaults on/off) if the decision changes the requirement text.
- [x] 2.29 Where the initial value belongs, and the three position initial
      values measured.

      At the shipping 5 outer / 2 linear budget the gravity step is the part the
      truncated PCG delivers last, so the starting point is what the visible
      motion is made of: with every warm start off, a free-falling panel moves
      0.001 m/s^2 and a hanging one barely sags at all (2 mm in 1200 substeps
      against 598 mm when it is on). Two places can carry an initial value, and
      they are not the same quantity:

      - the position the iteration starts from, set by `forward_step`
        (`pos[i] = pos_v + accel_ext * a_factor * dt^2`, geometry.cu);
      - the increment `dx` the linear solve starts from, set in
        `prepare_linear_step_kernel` and consumed by PCG as its `x0`
        (`A_mult_x(Ax, x)` then `r = b - A*x`).

      An earlier pass put the inertia displacement into `dx`, which changed the
      linear solve's own logic (the maintainer: "don't touch my dx - I asked you
      to change the *pos* initial value"). That path is reverted; `dx` is back to
      the historic handling. The options now live on `forward_step`'s own
      `warm_start` argument, at the same level as the VBD predictor, exposed as
      the `warm_start` parameter: 0 = off, 1 = VBD predictor
      (`a_factor = clamp(dot(a_prev, g)/|g|^2, 0, 1)`, the shipped behaviour),
      2 = the inertia prediction `q_inertia = q_prev + v*h + g*h^2`, 3 = velocity
      only (`q_prev + v*h`, no gravity term).

      The VBD predictor is self-limiting: it follows the acceleration the body
      *already* has, so a body at rest gets `a_factor = 0` and it cannot
      bootstrap a fall - it settles where its contribution equals the part of
      gravity the truncated solve delivers (measured 0.62 m/s^2, with the
      velocity then sitting on the `query_radius/h` tube cap at 0.23 m/s).
      Mode 2 forces the full `g*h^2` and therefore does bootstrap it.

      Measured (same build; `a_eff` against the analytic damping-limited 8.44;
      the hang probe pins one edge of a 1 m panel and reports the final edge
      strain, whose analytic static value is 0.0245 %):

      | position initial value | free fall (20x20) | hang: strain at the pins | t1 strain | t1 tail motion |
      |---|---|---|---|---|
      | 0 = off | 0.001 | 0.0004 % (nothing loads) | - | - |
      | 1 = VBD `a_factor*g*h^2` | 0.652 | **0.022 %** | 0.074 | 1.09 mm |
      | 2 = inertia `v*h + g*h^2` | **8.594** (exact) | 6.9 % | 0.338 | 1.15 mm |

      Two readings:
      1. Only the inertia prediction makes a free fall exact, and it is also the
         one that over-stretches a constrained region: it moves every free
         vertex by the whole gravity step while pinned vertices stay put, and a
         two-iteration correction cannot relax that. 6.9 % against an analytic
         0.0245 % is the "pinned region pulled long" report. The same tradeoff
         is the reason the maintainer's scenes oscillate between the two modes.
      2. Putting the same information in `pos` instead of `dx` is measurably
         better: identical free fall, and on t1 the tail motion drops from
         7.53 mm (the reverted `dx` version) to 1.15 mm at the same strain, with
         the pin strain coming down from 10 % to 6.9 %. The initial value is a
         statement about where the iteration *starts*, so it belongs on the
         position.

      A `dx`-level variant of Newton's Style3D initial value (`dx0 = v*h` on the
      first nonlinear iteration, their `kernels.py init_step_kernel` +
      `self.dx if _iter == 0 else None`) was also measured while it existed: it
      was worse than no initial value on both probes (0.232 vs 0.580 free fall,
      2.3 % vs 0.022 % pin strain) and exploded the maintainer's production
      scene. That is consistent with their structure - Style3D leans on a
      rest-shape *constant* matrix and 10 CG iterations, where the initial value
      barely matters, while our per-iteration assembled tangent with 2
      iterations turns a non-uniform increment into noise.

      Decision: `warm_start` defaults to 2 (the maintainer's "the pos initial
      value is what I asked you to change"); 1 keeps the shipped VBD predictor
      available, and `pd_static_diag_offdiag` keeps its default of 1.
      Getting "small iteration budget + contact + gravity all correct"
      at once still needs either a constrained rigid-mode (coarse) initial value
      - the inertia displacement projected onto the rigid motions the pins
      allow, which degenerates to mode 1 when everything is pinned and to mode 2
      when nothing is - or the constant-matrix operator of S1/S2.

      Also kept from this round: the PCG `alpha`/`beta` guards for
      `delta_old <= 0` / `d_dot_Ad <= 0`. Without them a small stiff mesh whose
      residual is solved to zero inside the outer loop yields 0/0 = NaN and the
      frame throws "PCG ended with NaN residual" (reproduced on a single
      triangle).

      Quick-suite status of this build (`pytest -m quick`, 12 collected): 11
      passed, 1 failed - `sim/smoke`, on the ground resting tolerance only.
      Corner-pinned sheet, 60 frames at 24 fps, replayed headless with the same
      scene (probe_smoke_scene.py):

      | smoke scene configuration | mean free disp (m) | z_free max (mm) | vertices at the clamp | quick suite |
      |---|---|---|---|---|
      | `pd_static_diag_offdiag = 1` (default) | 2.3e-9 | 0.2253 | 89.6 % | 11 passed, 1 failed |
      | `pd_static_diag_offdiag = 0` (anchor) | 2.0e-10 | 0.1000 | 100 % | 12 passed |
      | `pd_static_diag_offdiag = 1, warm_start = 1` | 1.5e-5 | 0.3574 | 45.8 % | 1 failed |
      | `pd_static_diag_offdiag = 1, 20 outer / 20 linear` | 4.5e-4 | 3.1643 | 0 % | 1 failed |

      The sheet settles in every configuration (mean free displacement 1e-9 m
      or less on the two shipping budgets), so the anchor-free operator does
      not make the standard scene move; it moves its contact equilibrium. With
      the paired Laplacian 89.6 % of the free vertices still come to rest
      exactly at the clamp (`z = 0.1000 mm` = the sheet thickness) but the
      remaining ones settle up to 0.125 mm above it, which is outside the test's
      `GROUND_REST_TOL_M = 1e-5 m` "in ground contact" tolerance. Raising the
      budget to 20/20 does not close the gap - it lifts the sheet 2.4 mm off the
      clamp - so this is where the contact equilibrium sits, not a truncation
      artifact of the 5/2 budget. Whether to relax that tolerance, raise the
      contact stiffness, or keep the anchor as the smoke-scene default is the
      remaining decision (see 8.2).
