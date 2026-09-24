# Proposal: cloth-plasticity

## Why

Cloth in Qianyi_DP is purely elastic: every rest quantity is derived from the
flat 2D pattern once and never changes, so a wrinkle is only a deviation from
that pattern and vanishes as soon as the deforming load or support is removed.
Real cloth keeps a crease, and how firm that crease is depends on how long the
deformation was held. The engine needs both halves of that behavior: a rest
shape a deformation can move, and the internal friction that makes holding time
matter.

## What Changes

- Add the bending-family wrinkle model of Gong et al. (2025) to PDNewton, in the
  dihedral-angle space the engine already uses:
  - an **internal-friction anchor angle** per bend entry with stick-slip and a
    dwell-dependent slip threshold, and
  - an **elasto-plastic rest angle** per bend entry with a yield threshold and
    time-dependent hardening.
- Make the rest shape an input instead of a constant: the mesh data carries two
  per-edge arrays, `angles` (rest dihedral angle, 0 = flat) and `compress`
  (relative rest-length change, 0 = unchanged), both defaulting to 0 when
  absent. The sewing input's own `angle` / `compress` fields are removed: a seam
  hinge and an internal line are expressed by the values on the edge they
  belong to, which is what the frontend's seam-angle, internal-line and painted
  expansion/shrinkage tools convert into. An immutable elastic copy of the
  resulting rest shape is kept for reset.
- Make the rest angle a piece of scene state rather than a constant: it is
  assembled at scene init from the per-edge input, and every bend entry of the
  unified table takes part - mesh edges and seam hinges alike - gated only by
  the existing bend validity.
- Add `plastic` to the per-object mesh input: only panels that set it take part
  in the model. Every other panel keeps a fixed rest angle and today's elastic
  bending behavior.
- Add a global `plasticity_time_scale` (default 0). At 0 the model is evaluated
  at t = 0, so no dwell grows and no time hardening happens; turning it up
  scales how fast the dwell and hardening clocks advance, which is how the
  animation mode compresses minutes of holding into seconds of simulated time.
- Add the verbs: `freeze_rest_shape()` (adopt the current shape as the rest
  shape), `reset_plasticity()`, and `get_plasticity_state()` for readback.
- Keep the feature PDNewton-only, like the external forces. Document that the
  quadratic IBM bending model has no rest angle and is therefore unaffected.

Non-goals (deferred, deliberately out of scope here):

- Stretch plasticity (the tensile half of the paper's model) and evolving rest
  edge lengths: `compress` is a static rest-shape input, not an evolving plastic
  state, so a rest length never changes by itself in this change.
- The frontend authoring tools themselves (drawing expansion/shrinkage,
  editing seam and internal-line angles) - the engine only consumes the two
  per-edge arrays they produce.
- Anisotropic plastic yield, plastic shear, and calibration against measured
  fabric data.
- Coupling to tearing/seam failure, per-region rest angles, and plastic
  deformation for VBD / XPBD / Explicit.

## Capabilities

### New Capabilities

- `cloth-plasticity`: time-dependent persistent wrinkles for PDNewton - internal
  friction (anchor angle, stick-slip, dwell) and elasto-plastic rest angles
  (yield, hardening), opt-in per panel, with freeze, reset and state readback.

### Modified Capabilities

(none - the existing `testing-harness`, `drape-debug-window` and
`data-driven-drape-tests` specs describe test tooling and consume engine
behavior without touching rest-shape semantics)

## Impact

- Engine (`src/simulation/`): a new `plasticity.cu` (state build, per-substep
  update, freeze/reset) with declarations in `geometry.cuh`; the new per-entry
  state and per-panel mask in `geometry.*`; the aggregated bending first and
  second derivative in `dynamics/bending.cuh` (the GN and AOGS kernels); one
  extra launch per substep in `solver_PDNewton.cu`; the `plastic` field in
  `simulator.h` / `simulator_interface.cpp`; the file added to CMake.
- Rest-shape input: the two per-edge arrays in the mesh contract, the rest
  length they produce (`edge_lengths`), the seam hinge rest angle that now reads
  the edge's value instead of the sewing input's, and the removal of the
  sewing `angle` / `compress` keys.
- Public API: `freeze_rest_shape()`, `reset_plasticity()`,
  `get_plasticity_state()`, one new per-object mesh field, and the numeric
  parameter keys listed in `design.md`. No existing call signature changes.
- Compatibility: with no panel flagged the simulation is unchanged, including
  the CUDA-graph path and the bending kernels; the feature is off by default.
- Documentation: the new field and parameter keys in `docs/engine_input_spec.md`,
  a short section in `README.md`, and this capability spec as the contract.
- Verification: frame-data scenarios (residual deformation, hold-time ordering,
  rigid motion, frozen suspension) plus a feature-off regression against the
  pre-change baseline.
