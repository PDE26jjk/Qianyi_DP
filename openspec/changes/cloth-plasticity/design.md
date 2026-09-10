## Context

See `proposal.md` for motivation. Current state that shapes the design:

- Every rest quantity is derived once from the input pattern and never changes:
  `edge_lengths` (rest edge length from the 2D pattern), `Dms` / `areas` (per
  triangle rest shape), `bend_rest_theta` (unified bend entries: `0` for mesh
  edges, the sewing angle for seam hinges), `bend_factor` (dihedral factor) and
  the precomputed `IBM_q` used by the quadratic bending model.
- The dihedral convention in `dynamics/bending.cuh` is "flat = 0"; the discrete
  shell energy is `0.5 * k * (theta - theta_rest)^2`, so a rest angle of `0`
  makes any wrinkle an elastic deviation that always relaxes back to flat.
- Bending kernels read `bend_points` / `bend_rest_theta` / `bend_factor` /
  `bend_valid` / `bend_cross_rows` on every Newton iteration; the spring and
  FEM kernels read `edge_lengths` / `Dms` on every iteration. Changing a rest
  value therefore takes effect on the next iteration without touching the
  solver's assembly code.
- Some derived state is computed once per scene: `Jx_diag_pd` (the PD diagonal
  copied into `static_diags` every step) in `SolverPDNewton::init`, and `IBM_q`
  in the geometry bending setup.
- There is already a refresh precedent for rest-dependent state:
  `update_seam_state()` recomputes triangle areas and bend validity after a
  stitch-cluster rebuild, and `build_stitch_clusters()` is written to be
  allocation-free so it can run again during a simulation.
- The harness and the Blender frontend both drive `input_data` -> `set_solver`
  / `set_parameter` -> per-frame `update(dt)` with internal substeps of
  `step_h` (<= 0.003 s).

## References

The approach follows the published cloth-plasticity work; the first entry is
the closest match to the requirement (plasticity combined with internal
friction and time dependence) and the rest are the mechanisms this change
borrows or deliberately defers.

- D. Gong, Y. Yang, T. Shao, H. Wang. "Cloth Animation with Time-dependent
  Persistent Wrinkles." Eurographics 2025.
  [arXiv:2502.13491](https://arxiv.org/abs/2502.13491) - elasto-plastic bending
  with a yield strain, time-dependent hardening, and the internal-friction
  model that this change defers. Reference implementation:
  [github.com/realcrane/Cloth-Animation-with-Time-dependent-Persistent-Wrinkles](https://github.com/realcrane/Cloth-Animation-with-Time-dependent-Persistent-Wrinkles)
  (C++/CUDA).
- R. Narain, T. Pfaff, J. F. O'Brien. "Folding and Crumpling Adaptive Sheets."
  ACM Transactions on Graphics 32(4), SIGGRAPH 2013.
  [doi:10.1145/2461912.2461964](https://doi.org/10.1145/2461912.2461964) -
  plastic deformation of thin sheets through an evolving rest shape.
- T. H. Wong, G. Leach, F. Zambetta. "Modelling Bending Behaviour in Cloth
  Simulation Using Hysteresis." Computer Graphics Forum 32(6), 2013.
  [doi:10.1111/cgf.12137](https://doi.org/10.1111/cgf.12137) - bending
  hysteresis, i.e. the recoverable internal-friction half of the problem.
- E. Miguel et al. "Modeling and Estimation of Internal Friction in Cloth."
  ACM Transactions on Graphics 32(6), SIGGRAPH Asia 2013 - Dahl-style internal
  friction and its measurement; basis of the deferred friction model.
- B.-C. Kim, S. Oh, K. Wohn. "Persistent Wrinkles and Folds of Clothes."
  International Journal of Virtual Reality, 2011 - permanent wrinkles by
  changing the rest shape and material stiffness.
- Z. Wang, Y. Yang, H. Wang. "Stable Discrete Bending by Analytic Eigensystem
  and Adaptive Orthotropic Geometric Stiffness." ACM Transactions on Graphics
  42(6), Article 183, 2023 - the discrete bending model and the `theta`
  convention already used by this engine (`dynamics/bending.cuh`).
- Houdini Vellum, "Plasticity" (SideFX documentation):
  [sidefx.com/docs/houdini/vellum/plasticity.html](https://www.sidefx.com/docs/houdini/vellum/plasticity.html)
  - industry parameterization of the same idea (stretch/bend plastic flow
  gated by a threshold), used here as a naming reference for the API.
- Marvelous Designer, "Freeze/Unfreeze" (support article):
  [support.marvelousdesigner.com](https://support.marvelousdesigner.com/hc/en-us/articles/47358315135897-Freeze-Unfreeze)
  - the user-facing "bake the current garment shape" behavior this change
  reproduces with the freeze call.
- Style3D Studio help, "褶皱" (wrinkle tooling):
  [help.style3d.com](https://help.style3d.com/studio/zh/1113/c7491/ff6a6/5c9ba/4e55d)
  - shows the geometry-side alternative (pleats and gather tools that change
  the flat pattern); useful context for why the engine-side rest shape also
  has to change.

## Goals / Non-Goals

**Goals:**

- Persist wrinkles through rest-shape change, with a freeze/bake operation and a
  strain-driven plastic evolution behind one enable flag.
- Zero behavioral change when the feature is off, including for the existing
  baseline tests.
- Keep the per-frame cost O(edges + triangles), allocation-free, and confined to
  one update site so the Newton assembly path is untouched.
- Deterministic and resettable: loading a scene or resetting plasticity returns
  the exact elastic reference.

**Non-Goals:**

- Recoverable wrinkles (internal friction, bending hysteresis, Dahl-style
  friction, stick-slip anchors, dwell effects).
- Anisotropic plastic yield (warp/weft-direction dependence) and fabric
  calibration data.
- Changing the linear solver, the projector, collision handling or the seam
  cluster subsystem.
- Plasticity for the VBD / XPBD experimental solvers.

## Decisions

### D1. Plastic state lives next to the rest arrays it modifies

`Geometry` keeps two versions of each plastic-capable rest quantity:

- `bend_rest_theta_elastic` (immutable, built with the unified bend structure)
  and `bend_rest_theta` (live value consumed by the bending kernels);
- `edge_lengths_elastic` (immutable) and `edge_lengths` (live value consumed by
  the spring kernel);
- per-entry/per-edge plastic timers (`plastic_bend_time`,
  `plastic_stretch_time`) for time-dependent hardening.

Reset is then a device-to-device copy of the elastic arrays plus a timer
memset; no recomputation from the pattern is needed.

*Alternative:* mutate `pos_2D` instead of adding live copies. Rejected - `pos_2D`
is the pattern-space reference used by the bending factor, `IBM_q` and the
pick/2D mapping; overwriting it would destroy the elastic reference and break
reset.

### D2. Elasto-plastic bending with a yield threshold and bounded flow

Using the existing dihedral computation (`theta` with the flat convention):

```
eps_b   = theta - theta_rest                    # total bending strain
eps_e   = eps_b                                 # elastic (visible) part
eps_p   = plastic state, currently folded into theta_rest
d eps_p = sign(eps_e) * min(rate * h, max(|eps_e| - yield, 0))
theta_rest <- theta_rest + d eps_p
```

`theta_rest` moving toward `theta` is exactly a plastic rest-state update: the
energy minimum follows the deformation, so the wrinkle survives unloading.
Safety clamps: the per-step transfer is limited by `rate * h`, and `theta_rest`
is clamped to `+/- plastic_bend_max` so a wildly bent element cannot flip its
rest angle through `+/-pi` and lock in an inverted state.

Time-dependent hardening follows the published model: the effective yield
grows with how long the deformation has been held,

```
Kh      = Kh0 * (1 - g * (1 - exp(-t_p / tau)))      # g in (0,1)
t_p    += h   while plasticity is active, reset to 0 otherwise
yield_eff = yield / max(Kh, eps)
```

so a short press leaves a shallow crease and a long press a sharp one. `g = 0`
recovers the simple (time-independent) perfect-plastic model, which keeps the
feature tunable from "immediate permanent crease" to "only after a long hold".

*Alternative:* ideal plasticity (`eps_p = eps_e - yield`, instantaneous).
Rejected as the only mode - it produces an instant permanent set and cannot
express the "held for a while" behavior the proposal is after; it remains
available through `g = 0` plus a high rate.

### D3. Stretch plasticity is a separate, independently selectable mechanism

For the spring-mass model the same rule applies to the relative edge strain
`(L - L_rest) / L_rest`, updating `edge_lengths`. It is off by default because a
wrongly tuned stretch yield makes garments grow over time and it is not needed
for the primary crease use case.

For the FEM (BW) model the equivalent update means changing the per-triangle
rest metric `Dms` while keeping the material frame rotation-free. That metric
update is deliberately deferred: this change reports plastic stretch as
unsupported for that constitutive model (per the spec requirement) and only
implements the edge-length form. Plastic bending is unaffected by this and
works for both constitutive models.

### D4. Plastic bending requires a rest-angle bending model

`DiscreteShells_GN` and `DiscreteShells_AOGS` consume `bend_rest_theta` and get
plastic bending for free. The quadratic `IBM` model has no rest angle - its rest
state is the precomputed `IBM_q` derived from the pattern - so plastic bending
is reported as unsupported and the run continues elastically, as required by
the spec.

*Alternative:* recompute `IBM_q` from the deformed configuration (a different
plastic formulation for that model). Rejected for this change: it has different
energy semantics and would double the work of validating the feature.

### D5. One update site, once per frame

The plastic update runs in `Geometry::update_for_frame()`, before the substep
loop, with `h` equal to the frame's simulated time. Rationale: the frame is the
unit at which the frontends drive the engine, plastic time scales are seconds to
minutes, and keeping the update out of the substep loop leaves the Newton
assembly path unchanged. `rate * h` remains bounded because the same clamping
applies at frame scale.

*Alternative:* update per substep (as the reference implementation does).
Rejected for the first cut on cost and perturbation grounds; if the apply phase
measures instability (large per-frame strain on thin, heavily self-contacting
panels), moving the call into `update_for_step` is a one-line change and is
recorded as an open question.

### D6. Actions and state get real simulator calls, not parameter triggers

This capability is not a numeric knob: it adds verbs (freeze, reset) and state
(how much of the rest shape is currently plastic). Folding those into the
existing parameter map would mean expressing one-shot actions as sticky floats
that the engine has to consume and clear, and it does not scale to the surface
this feature needs. The change therefore adds a small set of simulator calls,
wired through `simulator_interface.*` in the same style as the existing
`input_data` / `update` / `pick_triangle*` bindings:

| Call | Semantics |
| --- | --- |
| `freeze_rest_shape()` | Adopt the current simulated configuration as the rest shape (one-shot action, vertex positions untouched). |
| `reset_plasticity()` | Drop the accumulated plastic state and return to the elastic reference of the current input. |
| `get_plasticity_state()` | Read the plastic state back (per bend entry, and per edge when stretch plasticity is enabled) for diagnostics, UI display and test verification. |

Numeric material constants (`plastic_enabled`, `plastic_bend_yield`,
`plastic_bend_rate`, `plastic_bend_max`, `plastic_hardening`, `plastic_tau`,
`plastic_stretch_enabled`, `plastic_stretch_yield`, `plastic_stretch_rate`)
stay in the existing parameter map, because that is where every other numeric
material or solver constant in this engine lives (`bending_k`, `sewing_k`,
`mask_stiff`, ...) and because the frontend already applies a parameter block
per scene. The split is: parameters carry numbers, calls carry actions and
state.

Freeze itself is the limit case of the plastic update: it transfers the entire
elastic deformation into the rest state in one pass (`theta_rest <- theta` for
valid bend entries, `L_rest <- L` for edges) and shares the cache-refresh path
with the continuous update, so there is still exactly one place where "the rest
shape changed" is handled.

The surface is expected to grow (per-region freeze, plastic presets, material
export/import for the frontend); new entries follow the existing `snake_case`
simulator-method convention, and the deferred phases (internal friction,
anisotropy) add their own calls and parameters rather than overloading these.

*Alternative:* edge-triggered parameter entries (`freeze_rest_shape = 1`), so no
interface changes are needed at all. Rejected - it hides an action behind a
float, makes it invisible to static analysis and to the frontend's parameter
UI, and it does not extend to the state readback and reset this feature needs.

### D7. A single refresh path for everything derived from the rest shape

`refresh_rest_dependent_state()` is the only entry point that runs after a rest
change, and it refreshes, in order:

1. `areas` (collapsed triangles stay zero) and `bend_valid` via the existing
   `update_seam_state()`;
2. `bend_factor`, which depends on rest areas and rest edge lengths;
3. `IBM_q` - skipped, since IBM is not a plastic-capable bending model (D4);
4. the PD diagonal (`Jx_diag_pd` / `static_diags`) used by the PDNewton
   iteration, which is otherwise computed once per scene.

Adding a new rest-derived quantity later means extending this one function,
which is the mitigation for the "forgot a cache" failure mode.

### D8. Cold state, resets and scene lifetime

The plastic state is part of the scene state: `Geometry::init` builds the
immutable elastic arrays and initializes the live arrays from them, so loading
input data always starts elastic. The `reset_plasticity()` call restores the
elastic arrays, zeroes the timers and refreshes derived state without reloading
the scene. Timers are ordinary floats on the device, so reset is a memset - no
host-side bookkeeping to keep in sync.

### D9. Code placement

A new `src/simulation/plasticity.cu` holds the device kernels
(`update_bend_plasticity`, `update_stretch_plasticity`, `commit_rest_shape`)
and the host-side `Geometry` methods; declarations go into `geometry.cuh` and
the file is added to `src/simulation/CMakeLists.txt`. The three public calls
are wired in the pybind interface layer next to the existing simulator
bindings, so the engine side stays free of Python types. Plasticity is
deliberately not folded into `sewing.cu` (which owns the unified bend table) or
`geometry.cu` (which already owns initialization), so the feature can be read
and removed in one place.

### D10. Verification is effect-based, like the rest of the project

Acceptance is measured the way this project verifies everything else - from
per-frame vertex data - with `get_plasticity_state()` available for assertions
that are about the state itself rather than the visible shape (for example
"rigid motion leaves the plastic state bit-identical"). Planned checks: a
residual deformation test (wring/compress, release, compare against the flat
pattern), a holding-time ordering test (same deformation held for different
durations), a rigid-motion invariance test (rotate/translate the whole cloth,
plastic state unchanged), a frozen-suspension test (freeze a wrinkled shape,
remove support, assert it holds), and a regression assertion that the
feature-off path is unchanged.

## Risks / Trade-offs

- [Plastic creep: a slow, unintended growth of the garment over long runs] ->
  yield thresholds are absolute and flow is rate-limited; below the yield
  nothing changes, and the hardening term saturates the effective yield.
- [Stale derived state silently producing wrong forces] -> D7's single refresh
  path plus a test that a frozen cloth stays at rest without residual jitter.
- [The plastic update perturbs the Newton solve] -> the update only runs between
  frames (D5), so no iteration sees the rest state change mid-solve.
- [Model coverage gaps read as bugs] -> unsupported combinations (IBM bending,
  FEM stretch) report explicitly instead of silently doing nothing.
- [Determinism] -> the project already has a known non-bitwise determinism
  issue; this change adds state but no new ordering nondeterminism (the kernels
  are per-entry and index-stable), and reset is a plain memset/copy.
- [Visual quality depends on tuning] -> defaults stay conservative and the
  feature is off by default; denim-like presets are calibrated during apply
  rather than guessed here.
- [Seam hinges carry a non-zero rest angle] -> plastic flow for seam entries is
  clamped per entry and uses the same `bend_valid` gate as the elastic path, so
  torn or collapsed entries never receive plastic updates.

## Migration Plan

Purely additive. The feature defaults to off, so no scene, test or frontend
configured today changes behavior. Rollback is either disabling the parameter
or reverting the change; no persisted data or saved state is involved. At
archive time the capability spec is copied into `openspec/specs/cloth-plasticity/`
and one paragraph is added to `AGENTS.md` describing the parameters and the
freeze entry point.

## Open Questions

- Default parameter values for a denim-like look (yield, rate, hardening, tau)
  - to be calibrated against a drape scene during apply, not fixed here.
- Whether per-substep plastic updates are needed for stability on thin panels
  with heavy self-contact; the call site (D5) is isolated so this can be
  revisited after measurement.
- Whether the FEM stretch metric update (D3) is worth a follow-up change
  (covering pressed/flattened cloth under tension) or whether spring-only
  stretch plasticity is enough for garment work.
