## Context

See `proposal.md` for motivation. The current state that shapes this design:

- `bend_rest_theta` is built once per scene in `Geometry::init_bend_structure()`:
  `0` for every mesh edge, `sewing_lines[..].angle` for every seam hinge slot
  (seam slots start at `nb_all_cloth_edges` in the unified table). The dihedral
  convention is "flat = 0" and the energy is `0.5 * k * (theta - rest)^2`, so a
  rest angle of 0 makes every wrinkle an elastic deviation that relaxes back.
- `edge_lengths` is the rest length of every edge: the 2D pattern length
  `|p2D(i) - p2D(j)|` for cloth edges, the input 3D length for the remaining
  (non-simulated) edges. It feeds the spring-mass force, the bending factor
  precompute (`geometric_scale = 3 l^2 / A`), the mean-edge-length the rest of
  the pipeline reads, and the collision broad phase.
- The FEM planar model's rest state is per *triangle*: `Dms[i]` is the 2x2 rest
  metric (its columns are the triangle's two edge vectors in the object's grain
  frame) and `areas[i] = 0.5 |det Dms|` is both the StVK energy weight and, at
  init, the source of the vertex masses. There is no per-edge rest length in
  that model. The default planar model is spring-mass (`constitutive_model_planar`
  defaults to 0 and the frontend does not send the key).
- The sewing input carries `angle` and `compress` per sewing line. `angle` is
  the seam hinge's rest angle; `compress` is parsed but consumed nowhere.
- The bending kernels (`dynamics/bending.cuh`) read `bend_rest_theta`,
  `bend_factor`, `bend_valid` and `bend_points` from the live arrays on every
  Newton iteration. Nothing derived from the rest angle is cached: the PD
  diagonal (`static_diags` / `Jx_diag_pd`) is built by
  `pd_precompute_spring_forces` from the spring lattice and the per-object
  stretch stiffness only, while `bend_factor` / `areas` / `IBM_q` come from the
  2D pattern and the assembled element diagonal is rebuilt per iteration.
  Changing a rest angle therefore needs no cache refresh at all.
- `SolverPDNewton::step(h)` runs `forward_step` first, which overwrites `q` with
  the warm-start prediction (`warm_start` defaults to 2), and then runs the
  Newton loop inside a captured CUDA graph. The substep-start positions survive
  in `pos_step_prev` (`Geometry::update_for_step` copies them at the top of every
  substep). External forces are already built once per substep just before the
  graph capture - the same slot this feature needs.
- The default bending model is AOGS (`bending_model` defaults to 2 and the
  frontend does not send the key). AOGS takes the first and second derivative of
  the bending energy at the current state (`g` and `p`) and derives an
  orthotropic geometric stiffness from them, so an extra quadratic energy term
  needs no new machinery.
- The reference implementation of the paper updates its friction and plastic
  state once per time step, at the current dihedral angle, before assembling the
  bending force and Jacobian, and it does so on every mesh edge. It also
  confirms the parameter units used below: the thresholds are dihedral angles in
  radians, not the curvature-scaled strain.

## References

- D. Gong, Y. Yang, T. Shao, H. Wang. "Cloth Animation with Time-dependent
  Persistent Wrinkles." Eurographics 2025.
  [arXiv:2502.13491](https://arxiv.org/abs/2502.13491) - the model this change
  implements: forces and Jacobians in SM Appendix E (Eq. 30-37), algorithms in
  Appendix F, parameter tables in Appendix C. Reference implementation:
  [github.com/realcrane/Cloth-Animation-with-Time-dependent-Persistent-Wrinkles](https://github.com/realcrane/Cloth-Animation-with-Time-dependent-Persistent-Wrinkles)
  (`scr/Cloth.cpp`: `cal_bend`, `dwell_friction`, `hardening_plastic`).
- Z. Wang, Y. Yang, H. Wang. "Stable Discrete Bending by Analytic Eigensystem
  and Adaptive Orthotropic Geometric Stiffness", ACM TOG 42(6), 2023 - the AOGS
  model whose `p` / `g` inputs this change aggregates.
- E. Grinspun, A. Hirani, M. Desbrun, P. Schroeder. "Discrete Shells", SCA 2003 -
  the energy the engine's dihedral bending already implements, and the reason
  the paper's `K_b` maps onto the existing `bending_k * bend_factor` without
  rescaling the geometry factor.
- Houdini Vellum "Plasticity" and Marvelous Designer "Freeze/Unfreeze" -
  industry naming references for the parameter surface and the freeze call.

## Goals / Non-Goals

**Goals:**

- Implement the paper's friction and plastic model faithfully in angle space,
  including its time-dependent parts, on the unified bend-entry table.
- Zero behavior change and zero added cost when no panel opts in, including the
  CUDA-graph capture and the bending kernels' existing path.
- State that is per entry, deterministic (no atomics), resettable and readable
  through the public API.
- Keep AOGS - the default and best-quality bending model - by folding the new
  terms into its existing `p` / `g` inputs instead of adding a second code path.

**Non-Goals:**

- Stretch plasticity, rest edge lengths, and the tensile half of the paper.
- The quadratic IBM bending model, which has no rest angle.
- VBD / XPBD / Explicit solvers.
- Exact Hessians: the anchor is treated as constant inside a step, so the
  tangent is the paper's Gauss-Newton form, not the true second derivative.
- A pattern-level interface for internal lines (their rest angle stays 0).

## Decisions

### D1. The plastic state lives next to the bend table it modifies

`Geometry` keeps, per bend entry (`nb_all_cloth_edges + nb_all_stitches` slots):

| Array | Meaning |
| --- | --- |
| `bend_rest_theta` | live rest angle (the plastic state, written by the update) |
| `bend_rest_theta_elastic` | immutable copy for reset |
| `bend_anchor_theta` | internal-friction anchor angle |
| `bend_stick_t` | stick timer for the dwell effect |
| `bend_plastic_t` | plastic timer for the hardening effect |
| `bend_plastic_hp` | accumulated hardening plastic strain |
| `bend_yield_theta` | live yield angle |
| `bend_plastic_enabled` | per-entry mask: owning panel flagged AND entry valid |

They are allocated and initialized in `Geometry::init_bend_structure()` right
after the bend table is built, so every scene load starts from the input:
rest = input angle, anchor = rest, timers = 0, hardening strain = 0,
yield = `plastic_bend_yield`. Sizing them with the bend table keeps them valid
across scene rebuilds, and only their contents change afterwards, so the
CUDA-graph capture key (which mixes buffer identity and size) stays valid.

*Alternative:* keep the state in the solver. Rejected - the bending kernels
consume the rest angle, and the state has to outlive a solver instance.

### D2. The model, in angle space, advanced once per substep

Per bend entry, with the entry's own stiffness `k_e = bending_k *
bend_factor[i]` and `k_f = plastic_bend_friction * k_e`:

```text
h_eff = plasticity_time_scale * h
theta = current dihedral angle of the entry (as computed today)

# internal friction: anchor + stick-slip + dwell   (paper Eq. 7-8)
thres = thres_inf - (thres_inf - thres_0) * exp(-t_stick / tau_f)
delta = theta - theta_anchor
if |delta| > thres:
    theta_anchor += sign(delta) * (|delta| - thres)
    t_stick = 0
else:
    t_stick += h_eff

# elasto-plastic rest angle: yield + time-dependent hardening (paper Eq. 9-14)
eps_e = theta - theta_rest
if |eps_e| > yield:
    t_plastic = (sign(eps_e) == sign(eps_p)) ? t_plastic + h_eff : 0
    K_h  = K_h0 * (1 - g * (1 - exp(-t_plastic / tau_p)))
    beta = k_e / (k_e + K_h)
    eps_hp      += beta * (|eps_e| - yield)
    theta_rest  += sign(eps_e) * beta * (|eps_e| - yield)
    yield        = yield_0 + eps_hp * (K_h / k_e)
```

Two properties matter for safety: `beta <= 1` and `|eps_e| - yield < |eps_e|`,
so the rest angle only ever moves toward the current angle, never past it - the
update cannot overshoot, cannot invert an entry, and needs no extra clamp.

*Alternative:* the paper's `eps_b = 3 (theta - rest) / H` variant with curvature
units. Rejected because the paper's own implementation and SM text use the
dihedral angle for both thresholds, because the engine's other bending
parameters are already in angle space, and because a curvature threshold would
require a second per-edge geometric quantity (`H` is not recoverable from
`bend_factor`, which folds in the per-object anisotropy).

### D3. One aggregate first/second derivative, so AOGS is unchanged

The friction and plastic terms only change *which* constant the bending energy
is quadratic around:

```text
force   = -(k_e * (theta - theta_rest) + k_f * (theta - theta_anchor)) * grad(theta)
p_total = k_e + k_f                                   # second derivative
g_total = k_e * (theta - theta_rest) + k_f * (theta - theta_anchor)   # first derivative
```

Both are exactly what the existing kernels already take as inputs: the GN kernel
uses the outer product with the second derivative, and AOGS is parameterized by
`(p, g)` at the current state. So the AOGS kernel keeps its single
`aogs_fp_diag` call and its rank decomposition, with `p = 1 +
plastic_bend_friction` and `g = (theta - theta_rest) + plastic_bend_friction *
(theta - theta_anchor)` (relative to `k_e`, which is applied afterwards as
today). The added cost is two multiply-adds per entry when the panel is flagged,
and a single branch when it is not.

*Alternative:* special-case the friction term with a separate geometric
stiffness. Rejected - the aggregate form is the same approximation the paper
uses for its own Jacobian and keeps one code path.

### D4. The update runs once per substep, outside the graph capture

`update_bend_plasticity` is launched from `SolverPDNewton::step(h)`, in the slot
where `accumulate_external_forces()` already runs: after `forward_step`, before
`run_pd_loop` and therefore outside the captured region. It reads
`pos_step_prev` (the substep-start configuration, i.e. the state the deformation
is measured on) and not `q`, which `forward_step` has already replaced with the
warm-start prediction.

*Alternatives:* inside the bending kernel (rejected: it would advance the state
once per Newton iteration, multiplying the plastic flow by the iteration count
and making the result depend on `pd_iters`), or once per frame in
`Geometry::update_for_frame` (rejected: `h` is the integration step, the
frontend's frame length varies, and the reference implementation also advances
per step).

### D5. The whole bend table takes part, gated only by validity

The update covers every entry of the unified table - mesh edges and seam hinges -
with no seam filter, matching the reference implementation's per-edge loop. The
only gate is `bend_valid` (dead, collapsed, torn or degenerately built slots
must be skipped: their `bend_points` are placeholders and the existing bending
kernels skip them too) plus the per-panel flag described in D7.

*Trade-off:* a seam hinge's rest angle interacts with the per-iteration seam
projection, and a plastic hinge angle can drift across frames. This is visible
in the verification scenarios (frozen suspension, seam closure) and the panel
flag is the escape hatch - a panel that should not drift simply does not set it.

### D6. `plasticity_time_scale` is the only clock

The dwell and hardening timers advance by `plasticity_time_scale * h`:

- `0` (default): the timers stay at 0, so the model is the t = 0 evaluation -
  a fixed slip threshold `thres_0` and a fixed hardening stiffness `K_h0`. This
  is the stateless drape case: hysteresis and yield are active, but nothing
  hardens over time.
- `1`: the paper's timing.
- `> 1`: the animation-mode compression the paper itself uses when it advances
  the timers by 10 s per step to reach a 500 s hold in 50 steps.

No additional mode flag is introduced; the animation mode is "time scale on",
the free-drape mode is "time scale 0".

### D7. The opt-in is per panel, the numbers are global

Each cloth object's mesh input carries `plastic` (integer/bool, default off).
The per-entry mask is derived at init from the owning object of the hinge edge
(`vertices_obj[bend_points[i].x]`, the same convention the bending factor
precompute uses) and from `bend_valid`.

The flag selects whether that panel's bending takes the plastic rest-angle
offset and the friction anchor offset at all. An unflagged panel keeps its
input rest angle, carries no anchor offset, and is otherwise untouched - same
force and same tangent as today, bit for bit. It is not a hardening mode: a
participating panel that should not harden is a parameter value
(`plastic_bend_hardening = 0`), not a second code path.

The numeric constants stay in the global parameter map, which is where every
other material constant lives and where the frontend already applies a
per-scene block:

| Key | Meaning | Default | Provenance |
| --- | --- | --- | --- |
| `plasticity_time_scale` | timer multiplier (D6) | `0` | this change |
| `plastic_bend_friction` | `K_friction / K_b` | `2.0` | cotton specimen column |
| `plastic_bend_thres0` | `thres_0`, slip threshold | `0.1` rad | cotton / denim |
| `plastic_bend_thres_inf` | `thres_inf`, dwell ceiling | `1.2` rad | cotton trousers |
| `plastic_bend_dwell_tau` | `tau_f` | `30` s | all materials |
| `plastic_bend_yield` | `yield_0` | `1.8` rad | cotton specimen |
| `plastic_bend_hardening` | `K_h0 / K_b` | `1.0` | "similar to the elastic parameters" |
| `plastic_bend_hardening_g` | `g`, hardening lower bound | `0.99` | all materials |
| `plastic_bend_hardening_tau` | `tau_p` | `30` s | all materials |

Default ratios come from the paper's tables so the first run is in the intended
order of magnitude; absolute stiffnesses stay relative to the scene's
`bending_k`, because the engine's `bending_k` is a scene-tuned value rather than
the paper's physical `K_b` (the engine's geometry factor is `3 l^2 / A`, whose
constant differs from the reference implementation's `l^2 / A`, so the
parameters are only meaningful as ratios of `bending_k`).

### D8. Rest angles keep their existing sources; freezing is the way to move them

The rest angle of an entry is established at init from the source that exists
today: `0` for a mesh edge, the sewing line's angle for a seam hinge, and (once
an input field exists) an internal line's angle. Nothing is measured from the
loaded 3D configuration, so a scene's initial drape behavior is exactly what it
is today. If a caller wants a non-flat loaded configuration to be stress-free,
that is `freeze_rest_shape()` after frame 0, not a different init rule.

### D9. The verbs and the readback

| Call | Semantics |
| --- | --- |
| `freeze_rest_shape()` | rest = current angle, anchor = current angle, timers cleared; vertex positions untouched on that frame |
| `reset_plasticity()` | restore the elastic rest angles, anchor = rest, timers and hardening strain cleared, yield = `plastic_bend_yield` |
| `get_plasticity_state()` | per bend entry: rest, anchor, yield, both timers (diagnostics, UI, tests) |

Freeze is the limit case of the plastic update (transfer the whole elastic part
into the rest shape) and shares the same state layout, so it needs no separate
model. Both calls only rewrite contents of already allocated buffers, so neither
invalidates the CUDA-graph capture.

### D10. No cache refresh, and no new kernels for the bending models

Because nothing derived from the rest angle is cached (Context), a plastic or
frozen update needs no rebuild of `areas`, `bend_valid`, `bend_factor`,
`IBM_q`, `static_diags` or the assembled diagonal. The previous revision of this
design specified a `refresh_rest_dependent_state()` path; it is dropped as
unnecessary for bending plasticity and would only come back with rest edge
lengths (stretch plasticity), which is out of scope.

### D11. Code placement

A new `src/simulation/plasticity.cu` holds the state initialization, the
per-substep update kernel, and the freeze/reset entry points, with declarations
in `geometry.cuh` and an entry in the CMake source list. The two dihedral bending
kernels in `dynamics/bending.cuh` are extended in place with the aggregate
`p` / `g` (D3). The per-object field is parsed in `simulator_interface.cpp`
beside `bending` / `stretch`, and the three public calls are wired in the pybind
layer, so the engine side stays free of Python types.

### D15. Prerequisite: the bend table's validity had to be filled at init

`Geometry::init_bend_structure()` called `update_seam_state()` *before* setting
`bend_structure_built`, and `update_seam_state()` returns immediately while that
flag is false - so the call (and the one inside `build_stitch_clusters()`)
no-oped for every scene, `bend_valid` stayed zeroed, and every bending kernel
returned before doing any work. Measured on the pre-change build: raising
`bending_k` from 0 to 1e6 changed a hanging panel by 0.4 mm (run-to-run noise is
~0.1-0.4 mm), while a build from before that guard was introduced (2026-09-12)
changed it by 135 mm. Bending was silently disabled engine-wide; the plastic
model cannot do anything while the entries it modifies are invalid, so setting
the flag before the call is a prerequisite of this change.

Consequence: scenes get their bending back, so existing frame data changes
(the bending-sensitive expectations in the experimental-solver group move).
This is a behavior *restoration*, and it is called out separately in the
implementation report because it is the one part of this change that is not
opt-in.

### D12. The rest shape is authored by two per-edge arrays in the mesh input

Each cloth mesh entry may carry `angles` and `compress`, both indexed by that
mesh's own edges and both defaulting to 0 when the key is absent (copied into
global per-edge arrays through the same offset-add path `edges` / `triangles`
use):

- `angles[i]` is the rest dihedral angle of the edge in the engine's convention
  (0 = flat, sign from the entry's fixed vertex order). It is written into the
  bend table at init: a mesh edge takes its own value and a seam hinge takes the
  value on its hinge edge. This replaces `SewingData.angle`, which is removed
  from the sewing input together with the unused `SewingData.compress`.
- `compress[i]` is the relative change of the rest length,
  `edge_lengths[i] = pattern_length(i) * (1 + compress[i])`: 0 keeps the pattern
  length and a negative value shrinks the edge.

Both are inputs, not state - they do not change while a simulation runs - and
they are what the frontend's seam and internal-line angle editing and painted
expansion/shrinkage convert into. Afterwards, only `freeze_rest_shape()` and the
plastic flow move a rest angle.

### D13. What the two arrays drive, and what they deliberately leave alone

`angles` feeds the bend table, so everything derived from the rest angle (the
bending force, the plastic state, the freeze path) follows it.

`compress` feeds `edge_lengths`, which is where the spring-mass model reads its
rest length directly - that is the in-plane path the shrinkage shows up on. The
geometry weights stay on the 2D pattern: `bend_factor`, `areas`, `Dms`, the
vertex masses and `cloth_edge_mean_length` are built from `pos_2D` and are not
rescaled. Rationale: those weights describe the discretization and the material,
while shrink/expand is in-plane rest-length authoring; scaling them per edge
would silently change the bending stiffness and the mass of an authored panel,
and a per-edge value cannot determine a per-triangle area scale without the
conversion of D14.

*Alternative:* scale `areas` and the bending weight by the compressed rest
lengths. Rejected for this change - it changes the bending stiffness and the
mass of every authored panel and needs the same conversion FEM needs.

### D14. The FEM planar model reads a per-triangle metric, so compress scales it

The FEM (BW / StVK) rest state is the per-triangle metric `Dms` plus the rest
area `areas[i]`, and its energy is `W = A_s * psi(E)` with `E` built from
`F = D_s * Dm^-1`. Shrink/expand is representable in that model - it is exactly
`Dm -> s * Dm` isotropically - but a per-edge scalar cannot drive it directly,
because one number per edge over-determines a triangle's rest shape unless the
three values are consistent.

Decision: an edge's `compress` therefore enters the FEM rest state as the mean
relative change of the three edges of each triangle,
`Dms[i] *= 1 + (c1 + c2 + c3) / 3`, clamped to a sane range so a degenerate
input cannot collapse the metric (`Dm^-1` would blow up). The metric carries the
authored rest shape; `areas` (the StVK energy weight *and* the init-time source
of the vertex masses) and the bending weight stay those of the pattern, so a
painted shrinkage changes the rest shape without silently changing the material
amount - consistent with D13. The frontend drives the FEM planar model, so this
path is not optional.

*Alternative:* rebuild each triangle's metric exactly from its three compressed
rest lengths (SSS in the same grain frame). Recorded as a refinement: it needs
the triangle-inequality guard and a separate rest area, and it behaves badly for
wildly non-uniform per-edge input, which is exactly what a painted field
produces at triangle scale.

## Risks / Trade-offs

- [Friction can destabilize the integration at large steps - the paper's own
  stability appendix reports an abrupt reaction-torque change at a 10 ms step
  that disappears at 1 ms] -> the substep stays the integration step (the
  frontend already subdivides, and the solver can cap its own substep); the
  frame-time and blow-up checks in the verification scenarios are the evidence.
- [The anchor starts at the rest angle, so a scene whose loaded configuration is
  already far from rest gets a one-time friction force proportional to that
  deviation] -> bounded by the slip threshold after a single substep, and the
  flagged-panel default keeps existing scenes out of the path; freeze at frame 0
  is the explicit way to adopt a loaded shape.
- [Thresholds are angles, so a plastic onset is mesh-density dependent] -> the
  paper makes the same choice for the same reason (its own text notes the
  curvature form is the mesh-independent one); the engine's bending stiffness
  already carries the same density dependence through `bend_factor`.
- [Seam hinges take part (D5), so a plastic seam angle can fight the seam
  projection] -> covered by the frozen-suspension and seam scenarios; the panel
  flag is the escape hatch.
- [Guidance parameters (nine keys) are a wide surface for a first cut] -> they
  are ordinary numeric material constants in the existing parameter map, all
  defaulted, and the only new *input field* is the per-panel flag.
- [Behavioral change when enabled: a flagged panel will not return to the
  pattern] -> that is the feature; the default keeps every existing scene and
  test on the elastic path.
- [`compress` is an authoring input applied once at init, so a large negative
  value makes a panel gather hard and can invert triangles at load] -> the
  degenerate-triangle handling the planar kernels already have skips zero-area
  faces, and the verification scenes read the frame data at load.
- [The sewing input loses two keys, so a caller still sending them is silently
  ignored] -> `compress` was unused, the seam angle moves to the hinge edge's
  `angles` entry, and the backend contract doc is updated in the same change.
- [The FEM planar model reads a per-triangle metric while the input is per edge
  (D14)] -> the metric is scaled by the triangle's mean relative change, which
  is exact for uniform input and smooth for painted fields; the exact per-edge
  rebuild stays recorded as the refinement.

## Migration Plan

Purely additive. With no panel flagged the parameter path is inert, so no scene,
test or frontend configured today changes behavior, and rollback is either
clearing the flag or reverting the change - no persisted data is involved.
Archive copies the capability spec into `openspec/specs/cloth-plasticity/` and
the input contract goes into `docs/engine_input_spec.md`.

## Open Questions

- Default numeric values for a denim-like look (friction ratio, yield, dwell
  ceiling) are to be calibrated in the apply phase against a drape scene, using
  the paper's denim column as the starting point.
- Whether the exact per-edge FEM metric rebuild (D14's alternative) is worth
  replacing the mean-scale form once a fabric-calibration pass exists.
- Whether stretch plasticity (rest edge lengths, and with it the cache refresh
  path) is worth a follow-up change.
