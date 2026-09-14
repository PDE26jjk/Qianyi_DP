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
      adds no iterations at all.
- [ ] 8.2 Update the spec requirement that still needs a maintainer decision
      (defaults on/off) if the decision changes the requirement text.
