# Proposal: real-time-contact-projection (investigated, not adopted)

## Why

The reported jitter in the Blender frontend is not one defect but two. The
planar-model half is fixed (`fem_shear_hessian` / `fem_psd_clamp`); the
remaining half is the contact path, and it is model independent: with the
frontend parameter block, two flat layers on the ground jitter 0.80 mm per
substep with `constitutive_model_planar=1` and 1.04 mm with `0`, and a single
3 mm cloth jitters 0.59 mm (planar 1) / 3.04 mm (planar 0).

The cause is structural. The contact forces are fully coupled (`compute_vf_force`
and `compute_ee_force` scatter into every participating vertex), but the
operator they contribute to is diagonal only: `accumulate_contact_force(f,
Jx_diag, h, stream)` never receives `Jx_nondiag`. The implicit solve therefore
overshoots on every contact-heavy step, the penetration is not removed, and the
configuration keeps cycling. Increasing the iteration count does not help - it
makes the overshoot larger - and the solver can end up reporting negative
curvature (`PCG nan`, reproducible with the spring model too, so it is not the
planar model's fault).

Two fixes were considered: feeding the contact blocks into the operator, and
removing the residual penetration geometrically with a position-level
projection. This change implemented and measured the projection. **It was not
adopted**: it does not reduce the jitter in any configuration, any setting that
acts makes the jitter worse, and it costs 2.5-3.5x the frame. The measurements
are in `design.md`; the engine code was reverted.

## What was measured

- Three formulations of the projection were implemented behind
  `contact_projection` and A/B tested on the reported scenes: a two-sided
  mass-weighted solve, a one-sided "deepest contact per primitive" push, and
  the same with the per-vertex correction bounded by the vertex's own
  penetration and a deep-penetration threshold.
- None of them reduced the jitter. Representative results (frontend mesh and
  parameter block, 900 substeps, last 300 measured, mm per substep):

| Scene | no projection | projection on |
|---|---|---|
| two flat layers, 8 mm | 0.78 | 6.2 - 6.7 (and 7.7 in the deep-penetration variant) |
| single 3 mm layer | 0.59 | 4.7 |
| flat single layer, 8 mm (no contacts) | 7e-6 | 1.5e-5 (no-op, as expected) |

- Cost when it acts: 9.1 ms/step against 3.7 ms (two-layer stack), 20.5 ms
  against 3.0 ms (3 mm layer).
- The first formulation reproduced exactly the failure the requirement below
  was written to prevent: with the per-vertex cap set to `query_radius`, many
  shallow contacts summed into a 2 mm push per sweep and the pile exploded.
  Bounding the correction by the vertex's own penetration fixed the explosion
  but not the jitter.

## Why it cannot work here

The limit cycle is not a penetration residual. Measured on the two-layer stack
(300 sampled vertices): the valid cross-layer contacts have a median
penetration of **7.9 um** and there are no spurious self-contacts (the 3 mm
flat layer has 13.8 candidates per vertex and **zero** valid ones), while the
per-substep motion is **0.6 - 0.8 mm** - two orders of magnitude larger. The
oscillation lives in the contact *force* response, so a correction bounded by
the penetration (micrometres) cannot counteract it, and anything larger simply
perturbs an already unstable loop.

Supporting measurements on the same scenes:

- Removing the contact forces makes the jitter vanish (3 mm layer 0.59 mm ->
  1.1e-4 mm; two layers 0.79 mm -> 2.5e-5 mm), so the jitter is contact driven.
- It scales with the contact stiffness: `ee_force_k`/`vf_force_k` 0.5 -> 0.1
  takes 0.81 mm -> 0.12 mm and 5208 -> 256 moving vertices, at the cost of
  deeper penetration.
- It is not a convergence or step-size problem: `linear_iters` 2 -> 10 (0.60 ->
  0.64 mm) and `pd_iters` 5 -> 20 (0.60 -> 0.60 mm) do not move it, smaller
  substeps make it worse (1.26 mm at 1 ms, 2.78 mm at 0.5 ms), and 100x global
  velocity damping does not change it (0.795 mm).

## Also measured: contact rows in the linear operator

The second candidate from the same investigation - emitting one compact row
per validated contact from the narrow phase and applying the rank-one term
`K J J^T x` in `A_mul_x` - was implemented behind `contact_operator` (default
0) and measured too. It removes only 6-21% of the jitter (0.79 -> 0.66 mm on
the two-layer stack, 1.55 -> 1.23 mm on the spring model, 0.61 -> 0.57 mm on
the 3 mm layer) while costing **RTS 0.466 -> 0.271 on the study-1 garment**,
and it fails with `PCG ended with NaN residual` in the default friction-on
configuration because the emitted row carries only the normal penalty
stiffness while the diagonal tiles also carry the friction stiffness. It was
not shipped either; details and numbers are at the end of `design.md`.

## Non-goals

- Making the simulation exactly non-penetrating. The product target is
  *approximately* non-penetrating real-time cloth: residual penetration inside
  the contact radius is expected and must degrade gracefully instead of
  exploding.
- Replacing the penalty contact forces. They keep their current role; the
  projection only removes what they leave behind.
- Shipping either candidate. Both were reverted; the measurements are recorded
  in `design.md` so the decision can be revisited if the contact set shrinks
  or the friction/diagonal inconsistency is resolved.
