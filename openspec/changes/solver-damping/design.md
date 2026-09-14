# Design: solver-damping

## Context

### Measured baseline (reported garment scene)

Scene: two sewn cloth panels draped over a body mesh, ~26k cloth vertices,
3.63 mm median edge, 2 mm thickness, 116 stitch pairs already closed (world gap
0.000 mm), no pinned and no attached vertices, PDNewton, `step_h` 4.5 ms,
`pd_iters` 5, `linear_iters` 2, FEM_BW planar, AOGS bending.

| quantity | value |
|---|---|
| tail-window median per-substep displacement | 0.21-0.25 mm, flat over 27 s |
| tail-window max per-substep displacement | 1.1-2.0 mm |
| displacement coherence (mean displacement / mean speed) | 0.1-0.5 (incoherent) |
| substep cost | 19-32 ms per 4.5 ms substep |
| with gravity off | decays to 0.02 mm per substep |
| membrane stiffness x5 (`base_spring_stiffness` 4000 -> 800) | no change |
| mass x1000 | no change |
| `pd_iters`/`linear_iters` 5/2 -> 10/10 | ~30 % lower, still sustained |
| contact stiffness (`ef_force_k` 50.5 -> 0.5, `vf_ground_k` 4 -> 20) | no change |
| seam merge disabled (`sewing_snap_max_dist` 1.0 -> 0) | no change (seams already closed) |
| `query_radius` 1 mm -> 5 mm | worse (max 2.8 mm) |
| `smooth_times` 0 -> 20 | median max-motion 1.22 -> 1.06 mm (~13 % lower) |
| `strain_stiffen_start` 0 (off) -> 0.05, rate 2.0 | 1.16 -> 3.32 mm median, max 7.37 mm |
| `strain_stiffen_start` 0.05, rate 0.1 | no change vs off (factor ~1) |

### Correction: the first damping implementation froze the cloth

The first cut accumulated the stiffness-proportional term as a per-vertex drag,
`c = beta * trace(H)` on the vertex's own velocity. Maintainer test: after
dragging a panel the hem hangs in the air and never falls, whatever the fabric
mass, and `rayleigh_beta = 0` restores the fall.

A nodal `c * v` resists absolute motion, so it damps free fall as well as the
mesh modes. Here the inertia term `m/h^2` is about 0.02 against `trace(H)` of
order 1e4, so `c ~ 0.3` N.s/m gives gravity a terminal velocity of
`m*g/c ~ 1.6e-5` m/s - visually frozen - and since that velocity is proportional
to `m`, raising the mass does not help. Stiffness-proportional damping has to be
a *relative* operator: `beta/h` times the element/spring Hessian, whose force on
a vertex is a difference of neighbour velocities and which is identically zero
for a rigid-body motion. The implementation now accumulates that operator on the
projective-dynamics spring basis (the same per-edge weight
`pd_precompute_spring_forces` uses) with a lumped diagonal for stability.

With the corrected operator the earlier "35x chatter reduction" is retracted: it
was measured against the freezing term. At a coefficient that still permits free
fall (`beta = 3e-5`) the corrected term does not measurably change the reported
scene's residual motion, and `alpha` is inert because the mass term is
negligible. First evidence for what does set it: the amplitude tracks
`query_radius`, the per-iteration trajectory-envelope radius (see task 2.7).

Two facts drive the design:

1. Gravity plus resting contact sustains the motion; every stiffness-like knob
   fails to change it. That is a missing-dissipation problem, not a
   configuration problem.
2. The mass/stiffness ratio makes the system elasticity dominated. The
   assembled operator is `H = K_assembled + Jx_diag_pd` and the inertia term
   enters only through the right-hand side (`(x_inertia - x_curr) * m / h^2`),
   so `m / h^2` is far below `K`. Multiplying the mass by 1000 changes nothing
   measurable. Mass-proportional damping alone will therefore also be almost
   inert here; the stiffness-proportional term is the effective lever.

### Why the existing mechanisms do not help

- End-of-step velocity multiplier (`v *= expf(-h * 0.5f)` in the PDNewton
  end-of-step kernel): a fixed 0.5/s floor, not exposed as a parameter, and
  applied to a velocity that the next substep re-derives from positions, so it
  only softens the inertia target.
- Frame-end velocity Laplacian (`Geometry::end_for_frame`): dissipative and
  useful as an anti-blow-up (measured: `smooth_times` 20 lowers the resting
  chatter ~13 %), but it acts once per frame on the velocity field, not on the
  modes the solve is resolving, and its launch geometry is wrong: the kernel is
  launched with `num_vertices + block - 1` blocks instead of
  `ceil(num_vertices / block)`: 26,211 blocks of 256 threads for a 25,956
  vertex mesh, i.e. 258x more threads than there is work, every one of them
  exiting on the bounds check. Measured cost of one pass, from three same-batch
  A/B pairs (`smooth_times` 0 vs 20, 400 substeps each): 0.02, 0.14 and 0.24 ms
  per pass. The measurement is noisy because the GPU is shared with an
  interactive session, so take the largest value as the bound: five passes cost
  at most about 1.2 ms of a 19-32 ms substep. Fixing the launch geometry is
  free and cannot be slower; it also lets the smoothing run more often, which is
  what makes keeping its anti-blow-up semantics affordable.
- Coulomb friction: tangential only (the slip is projected onto the contact
  tangent plane), capped by `mu * normal_load` where the normal load of a soft
  penalty contact is itself tiny, and driven by per-substep displacement with a
  smoothing band `friction_epsilon * h` (about 45 um at the shipped values,
  below the 0.23 mm residual motion, so it runs in its saturated branch and
  does not scale with velocity). The contact-stability investigation measured
  the same conclusion: turning friction off leaves the ringing unchanged, and
  sweeping `friction_epsilon` over four decades changes it by under 3 %.

## Mechanism A: implicit Rayleigh damping in the PDNewton operator

### Formulation

For `C = alpha * M + beta * K`, backward Euler gives

```
M/h^2 (x - x_inertia) + f_int(x) + (alpha * M + beta * K)/h * (x - x_prev) = 0
```

Two scalar effects:

- mass term: `M/h^2` becomes `M * (1/h^2 + alpha/h)`,
- stiffness term: the assembled tangent and the assembled elastic force are
  both scaled by `(1 + beta/h)`; equivalently `beta*K/h` is added to the matrix
  and `-(beta/h) * K * (x_curr - x_prev)` to the right-hand side.

Both are implicit, so they add no step-size stability limit, the matrix stays
SPD, and the equilibrium is unchanged (the scaling is homogeneous and the
damping term vanishes at rest). No extra kernel, no extra memory, and the PCG
iteration count is unaffected because the scaling is uniform.

### Implementation sites

- `SolverPDNewton::step` assembles `Jx_diag`, `Jx_nondiag`, `f` and `f_elastic`
  per iteration inside `run_pd_iteration`; the damping factors belong there, or
  preferably folded into the element kernels by passing a scalar factor so the
  extra arithmetic is shared with the existing accumulation.
- `static_diags` already carries the fixed part of the diagonal and
  `linear->Jx_nondiag_identity` the matching off-diagonal weights.
- `step_begin_pd` adds the inertia term; the `alpha` factor lands there.

### Which operator should carry `beta`

Two options, both SPD:

1. Assembled tangent (preferred). Scale what the iteration already assembled:
   membrane (FEM_BW or spring-mass), bending (IBM/AOGS/GN), stitch springs and
   the contact penalty blocks. This is the standard implicit Rayleigh form and
   it inherits damping of relative motion at contacts and seams at no extra
   cost.
2. Precomputed PD diagonal (cheap alternative). Reuse `Jx_diag_pd` and
   `Jx_nondiag_identity` as a fixed damping operator. Note what those arrays
   actually are: the spring-mass (Liu et al. 2013) projective diagonal,
   accumulated per edge from `base_spring_stiffness * mean(stretch)`, with the
   off-diagonal weight equal to the negated per-edge weight. They are not the
   FEM_BW tangent, and for `constitutive_model_planar=1` scenes they are only an
   approximation added to the assembled diagonal. Reusing them for damping is
   legitimate (damping needs an SPD operator, not the exact tangent) and it is
   essentially the existing frame-end velocity Laplacian made implicit and
   per-substep, which is attractive for anti-blow-up purposes. It damps a
   spring-lattice approximation rather than the material model actually in use,
   so record the choice in the implementation task.

### Parameter values

With `c = sqrt(E*t/rho)` (about 214 m/s at the shipped `base_spring_stiffness`
and 100 g/m^2) and `zeta(omega) = (alpha/omega + beta*omega)/2`:

| target mode | omega (rad/s) | alpha = 3, beta = 5e-5 |
|---|---|---|
| garment-scale swing / slide (period ~1.5 s) | ~4 | zeta ~ 0.38 |
| 5 cm fold | 1.3e4 | zeta ~ 0.32 |
| mesh-scale noise (3.6 mm) | 1.9e5 | zeta ~ 4.8 |

So `alpha` in 2-5 /s removes the garment-scale drift, and `beta` in 2e-5 to
1e-4 s removes the mesh-scale chatter while leaving folds alive. `beta` is the
primary knob on this mesh scale; `alpha` alone will look inert.

## Mechanism B: contact relative-velocity damping

The normal force of a resting contact is a positional penalty. It corrects
overlap but never removes the relative normal velocity that produced it, so a
penalty contact can sustain a force limit cycle when the substep is comparable
to its own time scale. On the reported scene, with a contact stiffness around
0.1 N/m and a per-vertex mass around 1e-6 kg, that time scale is
`sqrt(m/k) ~ 3 ms` against a 4.5 ms substep.

Add a normal dashpot to the vertex-face, edge-edge and edge-face narrow phases:

```
fn = -(k * pen + c * max(0, -v_rel . n)) * n
K_normal += c/h * I
```

- The `max(0, ...)` keeps the contact repulsive; the dashpot never pulls.
- `c/h` on the same row of `Jx_diag` makes it semi-implicit, so it stays stable
  for any substep.
- Calibrate `c = 2 * zeta * sqrt(k_contact * m_vertex)` with `zeta` about
  0.7-1.0 as the starting point, then tune against penetration and the
  resting-stability metric. Expose one parameter per contact type
  (`vf_damping_k`, `ee_damping_k`, `ef_damping_k`) so the existing three-way
  stiffness split can be A/B tested, with `0` meaning off.
- Relative velocity comes from `velocities` (already maintained by the solver)
  or from `(x - x_prev)/h`; the same quantity the friction term already uses as
  slip is available at the same code sites.

If Mechanism A already satisfies the resting-stability requirement, this
mechanism is still worth shipping for the pair-level case, but it becomes lower
priority and must be gated so it cannot regress penetration.

## Mechanism C: velocity-consistent seam merge

The cluster merge is a hard projection (`pos[v] = cluster_target`) and must stay
envelope-free: box-layout stitch pairs start about 181 mm apart on average
(552 mm maximum on the study element), and a 5 cm ramp ceiling made every such
cluster permanently ineligible, which left the stitch springs to drag the panels
together and stretch the fabric. The problem is not the projection, it is what
the next velocity update makes of it: the end-of-step kernel computes
`v = (x - x_prev)/h`, so a 20 cm merge over one substep reads as a 44 m/s
velocity, gets clamped by `max_vel`, and is injected as momentum.

Fix options, in order of preference:

1. Call the existing momentum-preserving pass
   (`average_stitch_cluster_velocities`, already implemented in the sewing unit
   and referenced but commented out at the PDNewton call site) after the
   projection, so a locked cluster keeps its mass-weighted mean velocity.
2. Update `pos_step_prev` for the projected vertices, so the projection is not
   seen as motion by the velocity update at all.

Both keep the merge semantics intact; option 1 is one call away.

## Support changes

- Replace the hard-coded `expf(-h * 0.5f)` with a `velocity_damping` parameter
  (default 0.5, so behavior is unchanged when unset). Document it as a floor.
- Keep the frame-end velocity Laplacian and fix its launch geometry to
  `ceil(num_vertices / block)`. Moving it to per-substep is a separate decision
  with its own measurement.
- Parameters added: `rayleigh_alpha` (1/s), `rayleigh_beta` (s),
  `vf_damping_k` / `ee_damping_k` / `ef_damping_k` (N s/m), `velocity_damping`
  (1/s). Names follow the existing map; all are floats set through
  `set_parameter`. New scalars and any new buffer must be added to the
  CUDA-graph capture key in `SolverPDNewton::step` (`mix_float` for scalars,
  `mix_buffer` for arrays) or a parameter change will silently replay the
  recorded graph.

## Alternatives considered

- Restore the trajectory envelope on the seam projection. Rejected: it
  prevents the closure of the box-layout stitch pairs entirely, which is the
  measured reason the ceiling and the hard merge exist.
- Scale up global velocity damping. Rejected by prior measurement in the
  contact-stability investigation: 100x global velocity damping leaves that
  ringing unchanged, and the mechanism cannot act on modes the solve is
  resolving.
- Smaller substeps. Rejected on frame budget, and measured ineffective for the
  same ringing (1 ms and 0.5 ms both keep it).
- Iteration counts. Measured here: 5/2 to 10/10 lowers the resting chatter by
  about 30 % only, so more iterations alone do not satisfy the requirement.
- Post-filtering the output positions on the `get_simulation_data` side.
  Rejected: the frontend reads the state asynchronously and feeds it back
  through shape keys, so filtering outside the solver breaks contact and seam
  consistency and hides the real state.
- XPBD-style constraint damping. Rejected as a drop-in: the XPBD path is
  documented as unstable at the shipped substeps.
- Provot strain-limiting projection as the stiffening mechanism. Already tried
  by the maintainer and rejected by measurement: scenes that must stretch past
  the limit (the box-layout assembly, where stitch pairs start tens of
  centimetres apart and have to be pulled together) blow up under it. Any
  stiffening work therefore has to stay a force-level mechanism with a bounded
  stiffness factor, defaulting to off.
- Damping the untangling pass. Rejected by maintainer decision: that pass exists
  to resolve crossings, and viscosity there fights the correction instead of
  dissipating a resting contact.
- More frame-end smoothing. It does help (~13 % at four times the cost), but it
  is a velocity-field low pass applied once per frame, not dissipation of the
  modes being solved. Keep it as anti-blow-up.

## Risks and trade-offs

- `beta` too large makes the cloth read as viscous (slow fold recovery, wet
  look). The sweep and the fold-recovery check bound it.
- `alpha` too large damps the garment-scale swing that reads as cloth motion.
- A contact dashpot can hurt PCG convergence if it is added to the diagonal
  without the matching right-hand-side term; keep the pair consistent.
- The hard-coded end-of-step multiplier currently hides some blow-ups; when it
  becomes a parameter, the `smooth_times` path must keep working for scenes
  that relied on it.
- Enabling strain stiffening at its new defaults is measured to triple the
  resting chatter (1.16 to 3.32 mm median, 7.37 mm max). It must not ship as a
  silent default before damping lands; a scene that wants it should opt in.

## Verification protocol

1. Resting stability. Reported garment scene, tail window of at least 1000
   substeps: median and max per-substep displacement per window, must decay and
   stay below 1 % of the mean edge length, finite frames throughout.
2. Ring-down. Undriven grid, gravity and support off: RMS free-vertex velocity,
   exponential fit for the e-folding time; must be at or below the documented
   maximum and consistent between 1 ms and 4.5 ms substeps.
3. Equilibrium invariance. Draped rest shape with damping off versus the default
   values: per-vertex difference and penetration statistics.
4. Contact pair behavior. Two layers with an initial normal approach: relative
   normal velocity decays without sign reversal, no attraction, penetration
   bounded.
5. Merge consistency. Cluster with a known offset and velocity: momentum
   preserved, kinetic energy not increased.
6. Invariants. Drape acceptance set and self-contact scenes: finite frames,
   bounding envelope, seam closure, area preservation, attached-vertex drift,
   penetration band.
7. Frame budget. Reference real-time scene, damping off versus defaults: solver
   time per update call within 5 %.

## Open questions

- Default values for `rayleigh_alpha` and `rayleigh_beta`: ship conservative
  defaults that satisfy the resting-stability requirement everywhere, or ship
  `0` and let the frontend opt in? The spec currently requires the defaults to
  satisfy the requirement.
- Mechanism A alone versus A+B: does operator-level damping suffice for the
  contact-driven limit cycle, or is the pair-level dashpot required?
- Whether the frame-end velocity smoothing should move to per-substep, and if so
  whether `smooth_times` keeps its current meaning.
