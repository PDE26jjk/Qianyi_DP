## 1. Plastic state and parameters

- [ ] 1.1 Add the plastic state to `Geometry` per design D1: immutable
  `bend_rest_theta_elastic` / `edge_lengths_elastic` plus live
  `bend_rest_theta` / `edge_lengths`, and the `plastic_bend_time` /
  `plastic_stretch_time` timers; verify with the existing quick group that a
  scene with plasticity off produces the same frame data as before the change
- [ ] 1.2 Initialize the live arrays from the elastic ones inside the geometry
  init path and rebuild them on `input_data`; verify that loading the same scene
  twice with plasticity enabled starts from the elastic reference both times
- [ ] 1.3 Add the parameter keys (`plastic_enabled`, `plastic_bend_yield`,
  `plastic_bend_rate`, `plastic_bend_max`, `plastic_hardening`, `plastic_tau`,
  `plastic_stretch_enabled`, `plastic_stretch_yield`, `plastic_stretch_rate`)
  with conservative defaults and read them in the update path; verify that an
  unset-thus-defaulted run and an explicitly-disabled run behave identically
- [ ] 1.4 Add `src/simulation/plasticity.cu` with its `geometry.cuh`
  declarations and the `src/simulation/CMakeLists.txt` entry; verify the module
  builds and imports
- [ ] 1.5 Add the public simulator calls (`freeze_rest_shape`,
  `reset_plasticity`, `get_plasticity_state`) to the pybind interface layer
  with their engine-side entry points declared, starting as stubs where the
  kernels do not exist yet; verify the module imports and each call is
  invocable on a loaded scene

## 2. Rest-shape refresh path

- [ ] 2.1 Implement `refresh_rest_dependent_state()` per design D7 (areas and
  bend validity through the existing `update_seam_state()`, then `bend_factor`,
  then the PD diagonal) and call it from scene init; verify the quick group is
  unchanged
- [ ] 2.2 Make the PDNewton precomputed diagonal refreshable outside `init()`
  (the `Jx_diag_pd` / `static_diags` path) and cover the explicit solver's use
  of the same rest quantities; verify the smoke scene still runs and steps
- [ ] 2.3 Verify the refresh is actually complete: change a rest value on
  purpose, refresh, and confirm no stale-force transient appears in the next
  frames (frame-by-frame inspection of `traces.json` / `frames.npz`)

## 3. Freeze (bake the current shape)

- [ ] 3.1 Implement the freeze kernel (`theta_rest <- theta` for valid bend
  entries, `L_rest <- L` for edges) and the host-side commit that calls the
  refresh path; verify vertex positions are unchanged on the freeze frame
- [ ] 3.2 Wire `freeze_rest_shape()` to the commit-and-refresh path and
  `reset_plasticity()` to the elastic-restore path (restore elastic arrays,
  zero timers, refresh); verify a second freeze with no step in between is a
  no-op and that reset restores elastic behavior
- [ ] 3.3 Report unsupported combinations per design D4: plastic bending with
  the quadratic IBM bending model and plastic stretch with the FEM constitutive
  model; verify the warning text appears once per run and the run continues
  elastically
- [ ] 3.4 Implement the `get_plasticity_state()` readback (bend entries always,
  edges when stretch plasticity is enabled); verify the returned state starts
  empty on load, matches the amount transferred by a known deformation, and is
  cleared by `reset_plasticity()`

## 4. Continuous plastic flow

- [ ] 4.1 Implement `update_bend_plasticity` with yield, bounded per-step flow,
  rest-angle clamping and the time-dependent hardening term (design D2); verify
  a deformation below the yield threshold releases back to the elastic result
- [ ] 4.2 Implement `update_stretch_plasticity` for the spring-mass model,
  enabled by its own flag (design D3); verify a stretched edge keeps a longer
  rest length after release and that the flag off changes nothing
- [ ] 4.3 Wire both updates at the documented single site (once per frame,
  before the substep loop) with the frame time as `h`, and accumulate/reset the
  plastic timers; verify the timer accumulates while the deformation is held and
  returns to zero after release
- [ ] 4.4 Call the refresh path after every plastic update; verify a plastic or
  frozen cloth suspended in mid-air holds its shape without jitter, drift or a
  pull back toward the flat pattern

## 5. Verification scenarios

- [ ] 5.1 Obtain explicit approval for the new test cases before writing them
  (repository policy: tests are never added speculatively), then add them under
  `tests/sim/`
- [ ] 5.2 Residual deformation: compress (or wring) a grid cloth past the yield
  threshold, release with the support removed, and assert the remaining wrinkle
  is above a measured threshold and far larger than the elastic-only residual
- [ ] 5.3 Holding-time ordering: hold the same deformation for a short and a
  long duration, then assert the longer hold leaves the larger residual
- [ ] 5.4 Rigid-motion invariance: translate and rotate the whole cloth with no
  internal deformation and assert the plastic state (observed through the
  residual shape after returning to rest) is unchanged
- [ ] 5.5 Frozen suspension: freeze a wrinkled configuration, remove the
  support, and assert the vertices stay within the configured tolerance of the
  frozen configuration over a full run
- [ ] 5.6 Regression: run `python -m pytest -m quick` with the feature off and
  confirm the result matches the pre-change baseline

## 6. Documentation and calibration

- [ ] 6.1 Add one `AGENTS.md` paragraph describing the feature, the parameter
  keys and the freeze entry point; verify the committed text contains no
  machine-specific paths
- [ ] 6.2 Manual pass in the debug window: load a GarmentCodeData scene, create
  wrinkles against the body, freeze, then suspend the garment and confirm the
  wrinkles persist; keep screenshots in the gitignored artifact tree and record
  the outcome in the change notes
- [ ] 6.3 Calibrate the default parameter values against a drape scene
  (denim-like target) and record the chosen values in the parameter block
  comments; verify the calibrated values keep the smoke scene finite over a full
  run
- [ ] 6.4 Run the full `quick` and `sim` groups with the feature on where
  applicable and summarize the outcome in the change notes
