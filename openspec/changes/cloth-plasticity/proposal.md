# Proposal: cloth-plasticity

## Why

Cloth in Qianyi_DP is purely elastic today: every rest quantity (`edge_lengths`,
`Dms`, `bend_rest_theta`, `IBM_q`) comes from the flat 2D pattern and never
changes. A wrinkle is therefore only a deviation from that flat rest shape, so
pressing a garment against a body or picking it up produces folds that
completely disappear once the support is removed - drape results look rubbery
rather than like denim, which keeps a crease after it is released. Production
tools (Marvelous Designer, Style3D, Houdini Vellum) all get this look from
plasticity: either an explicit "freeze"/bake of the current shape into the rest
state, or a strain-driven elasto-plastic evolution of the rest state. The
approach follows the published cloth-plasticity literature listed in
`design.md` (References).

## What Changes

- Add an opt-in plastic rest-shape state to `Geometry`, covering the bending
  rest angle of every unified bend entry (mesh edges and seam hinges) and,
  behind a separate flag, the rest length of every mesh edge.
- Add a **freeze** operation that commits the current configuration as the new
  rest shape, recomputing every derived rest quantity and cache in one step
  (the Marvelous Designer / Style3D "bake the wrinkles" behavior).
- Add continuous plastic flow: when the elastic part of a strain measure
  exceeds a yield threshold, part of it is transferred to the plastic state at
  a bounded rate, so wrinkles harden over the time a deformation is held and do
  not recover when the load is removed.
- Refresh all caches derived from the rest state when the plastic state
  changes, so force assembly and the linear system stay consistent inside a
  step.
- Add a public simulator API for the feature's actions and state - freeze the
  current shape into the rest shape, reset the plastic state, and read the
  plastic state back for diagnostics and verification - while numeric material
  constants keep flowing through the existing parameter map. The feature
  defaults to off, so every existing scene and test keeps its current behavior.
- Reset the plastic state on scene load, and provide an explicit reset so a
  run can be returned to the elastic reference without reloading the scene.

Non-goals (deferred to a later change, deliberately out of scope here):

- Internal friction / bending hysteresis (stick-slip anchor plus dwell effect)
  and any other recoverable-wrinkle model.
- Anisotropic plastic tensors (warp/weft-direction-dependent yield), plastic
  shear, and calibration against measured fabric data.
- Coupling between plasticity and tearing/seam failure.

## Capabilities

### New Capabilities

- `cloth-plasticity`: rest-shape plasticity for cloth - an explicit freeze that
  bakes the current shape into the rest state, plus strain-driven plastic flow
  with a yield threshold and time-dependent hardening, so wrinkles persist
  after the deforming load or support is removed.

### Modified Capabilities

(none - the existing `testing-harness`, `drape-debug-window`, and
`data-driven-drape-tests` specs describe test tooling and consume engine
behavior without touching rest-shape semantics)

## Impact

- Engine (`src/simulation/`): new plastic state and update kernels in
  `geometry.*` / a new plasticity translation unit; rest-angle consumption in
  `dynamics/bending.cuh`; rest-length consumption in `dynamics/planar.cuh`;
  cache refresh for `areas`, `bend_valid`, `IBM_q` and the PD diagonal in
  `solver_PDNewton.cu` / `solver_explicit.cu`; the unified bend entry table
  built in `sewing.cu` becomes a consumer of the plastic state.
- Public API: new simulator calls (`freeze_rest_shape`, `reset_plasticity`,
  `get_plasticity_state`) wired through `simulator_interface.*`, plus parameter
  keys for the numeric material constants (enable, yield thresholds, flow
  rates, hardening, time constant); no change to existing call signatures.
- Compatibility: default parameters leave the current elastic behavior bit-for-bit
  unchanged; only scenes that opt in are affected. Bending models that cannot
  represent a plastic rest angle (the quadratic IBM model) are skipped with a
  warning instead of silently doing nothing.
- Documentation: one paragraph in `AGENTS.md` describing the feature and its
  parameters, plus this capability spec as the behavioral contract.
- Verification: new test cases that suspend a wrinkled cloth and assert the
  wrinkle survives, that rigid-body motion does not create plasticity, and that
  the existing quick/GPU groups stay green with the feature off.
