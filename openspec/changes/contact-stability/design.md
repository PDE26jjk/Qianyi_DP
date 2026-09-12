## Context

The solver is a Projective-Dynamics / Newton hybrid: per-element local
projections, a global block-sparse linear solve (block 3x3 PCG), a mass-spring
or planar-FEM membrane, a dihedral bending model, a stitch-cluster sewing
projection, and penalty contacts for vertex-face, edge-edge and the
edge-face untangling pass. The ground is a hard position clamp plus a penalty.

Contacts contribute *coupled forces* but a *diagonal-only operator*:
`accumulate_contact_force` receives `Jx_diag` and never `Jx_nondiag`, and the
friction Hessian that is added to the diagonal is the isotropic
`scale * (I - n n^T)` rather than the derivative of the friction force the
kernel applies. The contact stiffness also has no relation to the caller's
substep, so at the shipped `step_h` (0.0045 s) the contact's own time scale and
the step size are the same order of magnitude.

The measurements behind the proposal add three constraints to any solution:

* the limit cycle is in the force response, not in the penetration residual
  (7.9 um median penetration against 0.6-0.8 mm per-substep motion), so
  removing overlap or damping velocity cannot fix it;
* it scales with contact stiffness, so any mechanism that keeps a stiff
  penalty at this step size will keep ringing;
* it appears in both constitutive models, so it is not a membrane defect.

## Goals / Non-Goals

**Goals:**

- A resting contact configuration settles at the caller's `step_h`.
- Penetration stays below the combined shell thickness; exact non-penetration
  is not a goal.
- The frame budget is preserved (no substep increase, RTS within 10%).
- The mechanism is selectable at runtime so it can be A/B tested in one binary.
- The evidence and the rejected directions are recorded so they are not
  re-tried.

**Non-Goals:**

- Substeps as the fix, XPBD, VBD.
- Strict non-penetration (CCD + filtered line search) - not real-time here.
- Per-collision mass scaling and other corrective hacks.
- Changing the membrane, bending or sewing models.

## What is available publicly

The approaches below are the ones that are both published and compatible with
"large steps, no substeps, no hacks". Each entry says what it would change here.

| Source | Mechanism | Fit for this project |
|---|---|---|
| Ten Minute Physics #15, "Self-collisions" (notes: five tricks) | particles + hash instead of triangle/edge primitives; collision distance derived from the rest configuration; substepping instead of CCD; maximum velocity tied to the collision distance; time-step independent positional friction | The positional friction and the velocity clamp transfer at any step size. The rest-distance rule is already satisfied: the 1-ring is excluded and a flat 3 mm layer yields zero valid self-contacts. Substepping is out. |
| PhysX 3 cloth documentation | particle self-collision with `selfCollisionDistance` smaller than the smallest rest distance, separation applied as an impulse with a stiffness, a collision mass scale, simulation frequency 120-300 Hz, explicit warning against variable time steps | The *impulse* formulation (velocity-level separation instead of a position penalty) and the thickness buffer transfer. Mass scaling is excluded by maintainer decision. |
| Houdini Vellum documentation | the quality/speed knobs are Substeps x Constraint Iterations (XPBD/PBD) | Evidence that production answers stiff contacts with position constraints plus substepping; the positional part is what we can borrow, the substepping is out. |
| Unreal Chaos Cloth documentation | PBD with XPBD constraint parameters | Same family signal: shipped real-time cloth is position-based, not stiff-penalty based. |
| Macklin et al., "Small Steps in Physics Simulation" (SCA 2019) | n substeps of dt/n with a single XPBD iteration each is more stable and less damped than one large step with n iterations, even against Krylov solvers, and is insensitive to conditioning | Explains both our measurements and why XPBD blows up when given our large step: XPBD's stability argument assumes substeps. Out as a fix, but it tells us the *step size is the wrong knob* and the constraint formulation is the right one. |
| IPC / Codimensional IPC (Li et al., 2020/2021) | a C2 barrier potential for the contact and thickness constraints, optimized with a Newton solve whose Hessian is positive definite by construction; strict non-penetration needs CCD plus a filtered line search | The barrier is the principled answer to a force-response limit cycle: no stiffness-versus-step-size interaction at all. We may drop CCD and the filter because the target is approximately non-penetrating, which removes most of the cost. Our engine already carries an IPC-style barrier force option whose Hessian is currently a diagonal approximation - which is the same defect that makes the penalty ring. |
| ADMM / global-local contact (Overby et al. 2017; HOT 2019; "Efficient frictional contacts for soft body dynamics via ADMM", Vis. Comput. 2024) | local contact problems with dual variables (projected Gauss-Seidel with a Coulomb cone) driven by a global, constant, prefactorizable elastic solve | Projective Dynamics is a special case of ADMM, so this extends the solver we already have instead of replacing it: contacts keep their coupled response but stay out of the global operator, and the step size is not limited by their stiffness. |
| I-Cloth (Tang et al., SIGGRAPH Asia 2018) | a non-linear impact-zone solve that resolves penetrations (overlapping primitives are grouped and solved together instead of pairwise), with implicit integration at large steps; 2-8 fps at 2-3x10^5 vertices on a 2018 commodity GPU, with ten-layer stacks among its benchmarks | The impact-zone shape is the transferable idea: pairwise penalties are the wrong shape for a pile. Its companion technique, reusing one overlap set across the solver iterations of a substep, does not apply here - this engine is nearly stateless and the caller may pause and restart the simulation at any time (see the rejected directions). |
| Progressively Projected Newton (arXiv:2505.21013) | project only the element Hessians that need it, so the global Hessian stays close to the true one | Relevant to the indefinite-operator and PCG-NaN class we hit whenever contacts enter the operator. |
| Vertex Block Descent (arXiv:2403.06321) | per-vertex block coordinate descent with a closed-form 3x3 solve, stable under unconverged residuals | Ruled out by measurement: it does not converge for this geometry and problem sizes. |

## Decisions

**D1 (open, decided by spikes): the mechanism is not chosen in advance.** Four
candidates are on the table; they are ranked by (expected effect on the jitter)
per (cost and risk to the frame budget):

1. **ADMM / global-local contact with duals.** Give each active contact a
   dual variable and solve the contact sub-problem locally (projected
   Gauss-Seidel with a Coulomb cone) while the global elastic solve stays as it
   is. Contacts keep their coupling but never enter the global matrix, so the
   measured -33% cost of the row-based coupling is avoided and the step size is
   not limited by the contact stiffness. This is the smallest conceptual step
   from the current solver because PD is the special case of ADMM where the
   local step is a plain projection.
2. **C2 barrier contact with a consistent, positive-semidefinite Hessian and a
   capped line search.** Replace the penalty force with a barrier over the
   thickness constraint, contributing the derivative-consistent rank-one
   blocks. No CCD, no filter (approximately non-penetrating is enough), and the
   line search is capped so the frame cost stays bounded. The engine already
   has the barrier force; the missing piece is that its Hessian is
   diagonal-approximated, so the barrier never sees its own coupling.
3. **Impact-zone contact solve (I-Cloth-style).** Group overlapping primitives
   into zones and resolve each zone non-linearly instead of accumulating
   pairwise penalty forces. Attacks the shape of the contact model for piles
   and folds; the largest implementation of the four.
4. **Contact-aware preconditioner (multilevel additive Schwarz / aggregation).**
   Not a contact model by itself, but the published enabler for PCG with many
   contact rows. Adopt it only together with (2), and only if (1) and (3) fail.

**D2: cheap model corrections are prerequisites, not alternatives.** Regardless
of the mechanism: replace the explicit tangential friction force (whose ramp is
`1/|slip|`-like) with a time-step independent positional friction; define the
contact thickness as a two-sided shell thickness with an explicit buffer; and
tie the velocity clamp to the contact distance rather than to a global maximum.
Each is measurable on its own and each removes a way for the contact response
to fight the mesh.

**D3: the mechanism must be measured with the existing protocol.** Interleaved
A/B in one binary, minimum/median over at least three rounds, on: the two-layer
8 mm stack, the single 3 mm layer (both constitutive models), the baseline
garment scene for RTS, and the drape acceptance set for the invariants.

**D4: what has already been measured and rejected.**

| Attempt | Result |
|---|---|
| Contact rows in the matvec (coupled contact Hessian) | -33% RTS on the garment scene and unstable with friction: the diagonal tiles carried the friction stiffness while the emitted rows carried only the normal one, so the operator was not the derivative of its own force. Making the friction consistent removed the (small) benefit entirely. |
| Position-level projection (three formulations) | No formulation reduced the jitter; every setting that acted made it worse; the bounded version cost +20-35% per substep. |
| 100x global velocity damping | No change (0.795 mm per substep). |
| Iteration counts (`linear_iters` 2->10, `pd_iters` 5->20) | No change. |
| Smaller substeps (1 ms, 0.5 ms) | Worse (1.26 mm, 2.78 mm) and 5-9x the cost. |
| Lower contact stiffness (`ee_force_k`/`vf_force_k` 0.5 -> 0.1) | Reduces the jitter 5-6x but deepens the resting penetration, which the product target forbids |
| XPBD at the shipped step size | Blows up; consistent with the substep requirement above. |
| VBD | Does not converge for this geometry. |
| Incremental collision detection (reusing the previous iteration's overlap set) | Not applicable: the engine is nearly stateless by design and the caller may pause and restart the simulation at any time, so there is no reliable history to reuse. |

## Risks / Trade-offs

- **Friction is the awkward part of every candidate.** The current friction is
  an explicit force with a slip ramp; ADMM needs a Coulomb cone projection in
  the local step, the barrier needs a friction potential. If friction is left
  as it is, it can keep the limit cycle alive on its own, so the friction
  rewrite (D2) has to be part of the same change.
- **The barrier's Hessian is only affordable if the operator structure
  changes.** Consistent barrier rows cost what the rejected coupling cost
  unless the matvec becomes gather-based or the rows are packed; this is why
  candidate (2) is second and carries a preconditioner dependency.
- **ADMM convergence on a dense stack is not guaranteed by the literature.**
  The spikes must measure how many local iterations the two scenes need and
  whether the duals converge with a single global solve per substep.
- **Cost of the narrow phase.** Every mechanism here keeps re-running the
  narrow phase for each solver iteration; there is no stateful caching option
  (the engine is nearly stateless and the caller may pause and restart), so the
  contact-detection cost is a fixed part of the budget that candidates (1) and
  (2) cannot reduce.
- **Determinism.** The local contact solve is order dependent (Gauss-Seidel);
  the existing determinism expectation is already an xfail, but the new path
  must not make it worse without recording it.
- **Risk of a silent default change.** The mechanism ships behind a parameter
  and stays off until the acceptance protocol passes.
