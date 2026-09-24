## 0. Prerequisite: the bend table's validity at scene init

- [x] 0.1 Set `bend_structure_built` before the `update_seam_state()` call at
  the end of `init_bend_structure()` (design D15); verified with a local probe
  that `bending_k` 0 vs 1e6 moves a hanging panel by ~135 mm again - it moved
  0.4 mm (run-to-run noise) while the flag was set afterwards
- [x] 0.2 Confirm the restored bending does not break the feature-off path
  structurally: the local verification script's feature-off scenario stays
  finite and the plastic state stays empty (see task 7.5 for the repository
  test groups, which move for the bending-sensitive expectations)

## 1. Input surface and parameters

- [x] 1.1 Add the per-object `plastic` flag to `ObjectDataInput` and parse it in
  the mesh-entry loop of `simulator_interface.cpp`, defaulting to off when the
  key is absent; verified with a scene that does not send the key (no plastic
  state, feature-off frames unchanged by the flag itself)
- [x] 1.2 Add the parameter keys of design D7 (`plasticity_time_scale`,
  `plastic_bend_friction`, `plastic_bend_thres0`, `plastic_bend_thres_inf`,
  `plastic_bend_dwell_tau`, `plastic_bend_yield`, `plastic_bend_hardening`,
  `plastic_bend_hardening_g`, `plastic_bend_hardening_tau`) and read them in the
  update path; verified that the defaults are inert and that each key moves the
  state or the held shape as designed

## 2. Rest-shape input

- [x] 2.1 Parse the per-mesh `angles` and `compress` arrays into the global
  per-edge arrays, treating a missing key or element as 0, and write `angles`
  into the bend table (a mesh edge from its own value, a seam slot from its
  hinge edge); verified that a scene without the arrays reports the input rest
  angles and runs as before
- [x] 2.2 Remove `angle` and `compress` from the sewing input contract; verified
  that a sewing entry's rest angle now comes from the hinge edge's `angles`
  value and that a leftover key is ignored
- [x] 2.3 Apply `compress` to the rest lengths at init
  (`edge_lengths[i] = pattern_length(i) * (1 + compress[i])`) after the
  pattern-derived weights exist; verified mean edge lengths 0.0247 m base,
  0.0181 m shrunk and 0.0331 m grown, with the vertex mass unchanged
  (2.362949e-05 kg in both runs)
- [x] 2.4 Verify the FEM planar model scales its per-triangle rest metric from
  the same input (design D14); verified the FEM in-plane response shrinks by the
  same ratio and that the planar x bending matrix stays finite in all four
  combinations

## 3. Plastic state and scene init

- [x] 3.1 Add the per-entry state of design D1 to `Geometry` and initialize it
  in `init_bend_structure()`'s caller path; verified the arrays are sized
  `nb_all_cloth_edges + nb_all_stitches` and that a scene reload re-initializes
  them (state readback empty again)
- [x] 3.2 Derive the per-entry enable mask from the owning panel's flag and
  `bend_valid`; verified a scene with an unflagged panel reports an unchanged
  rest angle and no timers
- [x] 3.3 Verify the seam slots carry the hinge edge's angle as their rest angle
  and that the anchor starts there (readback at frame 0)

## 4. Update kernel

- [x] 4.1 Implement `update_bend_plasticity` in a new
  `src/simulation/plasticity.cu`, launched once per substep from
  `SolverPDNewton::step` before the captured loop, reading `pos_step_prev`;
  verified the timers advance with simulated time (30 s accumulated over 60
  frames at time scale 60) and that the state does not depend on `pd_iters`
- [x] 4.2 Verify `plasticity_time_scale = 0` leaves both timers at zero and
  evaluates the model with the initial threshold and hardening stiffness
  (readback: timers exactly 0)
- [x] 4.3 Fold the aggregate first and second derivative into the GN and AOGS
  bending kernels (design D3); verified the friction term changes the held shape
  by 15.4 mm at `k_f = 2 k_e` and that the state kernel costs nothing
  measurable on a 100x100 grid (11.31 vs 11.35 ms/frame)
- [x] 4.4 Verify the rest angle never moves past the current angle and stays
  inside the dihedral range over a long run (no NaN anywhere in the matrix of
  planar x bending combinations)

## 5. Actions and readback

- [x] 5.1 Implement `freeze_rest_shape()`; verified vertex positions are
  unchanged on the freeze frame and that a frozen shape drifts 0.0000 m over
  0.5 s with gravity off
- [x] 5.2 Implement `reset_plasticity()`; verified the rest angles return to the
  input values exactly
- [x] 5.3 Implement `get_plasticity_state()`; verified the layout
  `(entries, 5)` = rest, anchor, yield, stick timer, plastic timer and that it
  is empty at load, populated after a hold, and cleared by reset
- [x] 5.4 Verify neither action reallocates a buffer covered by the CUDA-graph
  capture key (they only rewrite contents), and that the new state buffers are
  part of the key; verified by running freeze and reset inside a graph-captured
  run without a capture rebuild

## 6. Verification scenarios

- [x] 6.1 Approval: the maintainer authorized a small number of cases and asked
  for the non-PDNewton solver cases to be dropped (they were removed, and
  `AGENTS.md` records that those solvers are unmaintained)
- [x] 6.2 Rest-shape input (`tests/sim/test_cloth_plasticity.py`): a shrunk
  panel's mean edge length is below 95 % of the plain run under both planar
  models, the vertex mass is unchanged, the frames are finite, and a panel that
  did not opt in reports an empty plastic state
- [x] 6.3 Plastic flow and the timers (same file): a flagged pinned curtain
  develops a rest-angle change above 0.01 rad with the clock running, the stick
  timer accumulates simulated time, and reset returns the rest angles to the
  input
- [x] 6.4 Freeze contract (same file): freeze commits the current bend angle
  into the rest state on an isometric arc, moves no vertex, and a second freeze
  with no step in between is a no-op
- [ ] 6.5 Frozen suspension as a *visual* assertion: not asserted in the
  repository tests. On a procedural sheet the recovery after the load is removed
  is membrane-dominated while the freeze only changes the bending rest angle, so
  a hard threshold would measure the membrane. Covered as information by the
  local script and by the manual debug-window pass (task 7.4); a hard test needs
  stretch plasticity or a bend-dominated specimen
- [ ] 6.6 Residual deformation and hold-time ordering as *visual* assertions:
  covered as information locally (the hold-time effect is small in a
  self-limiting drape, larger for the hardening time constant); the state-level
  checks in 6.2-6.4 are the committed ones
- [ ] 6.7 Regression: `python -m pytest -m quick` reports 13 passed and one
  pre-existing failure (`sim/smoke`, the sheet hovering ~0.23 mm against a
  0.1 mm + 0.01 mm ground tolerance; it failed before this change too, at
  11.25 mm). Relaxing that tolerance is a maintainer decision, not part of this
  change

The wider scenario set (rigid-motion invariance, hold time, hardening time
constant, the planar x bending matrix, performance) is covered locally by the
gitignored `build/verify_plasticity.py` (19 checks, all passing).

## 7. Documentation and calibration

- [x] 7.1 Add the `plastic` field, the per-edge `angles` / `compress` arrays,
  the parameter table and the PDNewton / IBM limitations to
  `docs/engine_input_spec.md`; no machine-specific paths
- [x] 7.2 Add the README section (panel flag, authored rest shape, time scale,
  freeze call)
- [ ] 7.3 Calibrate the defaults against a drape scene, starting from the
  paper's denim column, and record the chosen values
- [ ] 7.4 Manual pass in the debug window: flag a garment, author a rest shape,
  fold or press it, freeze it and suspend it
- [x] 7.5 Run the `quick` group: 13 passed, 1 pre-existing `sim/smoke` failure
  (see 6.7), no failures from this change's cases; the local script covers the
  feature-on scenarios over longer runs
- [ ] 7.6 Run the `sim` group and the data-driven groups before archiving
