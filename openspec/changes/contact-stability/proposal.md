## Why

Contact-driven jitter is the top product problem for real-time cloth editing:
a settled cloth has to stay settled at the step size the caller asks for. The
frontend reports two cloth layers resting on each other ringing indefinitely,
with occasional frames where many vertices jump and come back; the finer the
mesh, the worse it gets (worst around 3 mm, below which it blows up).

Reproduced headlessly with the shipped parameter block (`step_h` 0.0045 s,
`pd_iters` 5, `linear_iters` 2, PCG, `query_radius` 2 mm):

| scene | per-substep motion | moving vertices |
|---|---|---|
| two flat layers, 8 mm mesh | 0.78 mm | 5135 / 5706 |
| one flat layer, 3 mm mesh | 0.61 mm (`planar=1`) / 3.06 mm (`planar=0`) | 1777 / 4952 |

The planar-model half of this is already fixed (`fem_shear_hessian`,
`fem_psd_clamp`: the flat single-layer case went from 1.2 mm to 1.9e-6 mm per
substep). What remains is contact driven: disabling the contact forces drops
the same scenes to 1.1e-4 mm and 2.5e-5 mm, and the motion scales with the
contact stiffness (`ee_force_k`/`vf_force_k` 0.5 -> 0.1 takes the two-layer
stack from 0.81 mm to 0.12 mm, at the cost of deeper penetration).

The mechanism is not leftover overlap. In the two-layer stack the valid
cross-layer contacts have a median penetration of 7.9 um, two orders of
magnitude below the 0.6-0.8 mm per-substep motion, and no self-contacts are
generated at all (the 3 mm flat layer has 13.8 candidates per vertex and zero
valid ones). Iteration counts (`linear_iters` 2 -> 10, `pd_iters` 5 -> 20),
smaller substeps (1 ms, 0.5 ms) and 100x global velocity damping do not change
it. It is a limit cycle in the contact *force* response, so it cannot be fixed
by removing penetration residuals or by damping the velocity.

## What Changes

- Introduce a contact response that stays stable when the caller's substep is
  at or above the contact's own natural time scale, with penetration bounded
  rather than eliminated (the product target is *approximately*
  non-penetrating).
- Select the mechanism with the spikes ranked in `design.md` (ADMM /
  global-local contact with duals; C2 barrier contact with a consistent
  positive-semidefinite Hessian and a capped line search; an impact-zone
  contact solve; a contact-aware preconditioner), and ship the one that passes
  the measured protocol.
- Expose the new behaviour behind parameters so a scene can be A/B tested in
  one binary, and keep the current path as the default until the protocol
  passes.
- Record, in the change, the directions that were measured and rejected so
  they are not re-tried.

Out of scope, by maintainer decision and by measurement: substep increase as
the fix (the frame budget forbids it), XPBD (unstable at large steps; its
stability argument assumes substeps), VBD (does not converge for this
geometry), exact non-penetration via CCD plus filtered line search (not
real-time at this scale), and per-collision mass scaling.

## Capabilities

### New Capabilities

- `contact-stability`: a resting contact configuration settles instead of
  ringing, penetration stays bounded, the response holds across the contact
  stiffness and layer count, the frame budget is unchanged (no substep
  increase), and the existing quality invariants hold.

### Modified Capabilities

(none)

## Impact

- Engine: `src/simulation/collision.cu`, `src/simulation/contact/`,
  `src/simulation/solver_PDNewton.cu`, and possibly the linear solver's
  preconditioner; new solver parameters.
- Verification: existing harnesses only - the drape acceptance set for the
  invariants, the study-1 measurement path for RTS, and the two reproducible
  contact scenes above for the jitter metric.
- No public Python API change; no change to the shipped defaults until the
  protocol in `specs/contact-stability/spec.md` passes.
