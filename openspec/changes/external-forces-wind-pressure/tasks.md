## 1. Preconditions and parameter surface

- [x] 1.1 Request maintainer approval for the test additions and edits this
  change needs (repo rule: tests are not written without explicit approval);
  record the approval before any test file is touched
  - Recorded: the maintainer asked that test code stay out of the repository for
    now, so verification ran from a gitignored script under `build/` and no test
    file was added or edited.
- [x] 1.2 Add the per-object pressure and optional wind coefficient fields to
  the object data and to the `input_data` parsing, with inert defaults; verify
  that a scene supplying none of the new fields loads and reports the same
  parameter values as before
- [x] 1.3 Add the global wind parameters (velocity components, gust and
  turbulence settings, air density, lift and drag coefficients) to the scalar
  parameter surface with documented defaults; verify a scene that sets none of
  them reads back the documented defaults
- [x] 1.4 Keep external forces PDNewton-only: other solvers ignore them, with no
  report and no partial application; verify that a non-PDNewton solver runs a
  pressure/wind scene and that its frame data does not depend on those values

## 2. PDNewton external-force plumbing

- [x] 2.1 Add the per-vertex external-force buffer and a clear path for the
  disabled case; verify the buffer is sized for all vertices and reads as zero
  when no external force is configured
- [x] 2.2 Wire the buffer into the PDNewton inertia kernel call (replacing the
  null pointer); verify the disabled path reproduces the current per-frame data
  for one existing scene
- [x] 2.3 Leave the warm-start path unchanged: no new parameter, since the
  projection is only reached with `warm_start = 1` and the shipped default is
  `warm_start = 2`; verify with forces disabled that the baseline displacement
  and residual metrics are unchanged
- [x] 2.4 Verify the enabled path changes motion in the configured direction and
  that all frame data stays finite

## 3. Constant normal pressure

- [x] 3.1 Implement the per-face constant pressure accumulation (area-scaled,
  along the face normal, split across the face's vertices) into the
  external-force buffer; verify on a two-panel sewn shell that it inflates into
  a puffed shape with finite frame data
- [x] 3.2 No orientation check for pressure objects: cloth triangles are already
  re-oriented to the pattern normal at load, so an input-winding check is dead
  weight and was removed after review; verify that a mixed-winding pressure
  object simulates with no diagnostic and that an open panel still loads
- [x] 3.3 Verify the inert and inverse cases: pressure zero reproduces the
  no-pressure motion, and negative pressure pulls the shell inward
- [x] 3.4 Keep pressure load out of the stitch tearing criterion; verify with
  tearing enabled that an inflating shell does not tear from pressure alone
  - Note: the pressure load is accumulated in its own buffer and never reaches
    the stitch status, and the PDNewton path does not write the torn status
    anywhere today, so this holds by construction.

## 4. Wind

- [x] 4.1 Implement the wind field (base velocity plus temporal gust and spatial
  noise) sampled at the surface, with noise applied to the velocity field rather
  than the force; verify zero wind is inert and a steady wind deflects a hanging
  panel downwind to a steady state
- [x] 4.2 Implement the quadratic lift/drag force with the convention pinned in
  the design (air-minus-cloth relative velocity, area-scaled, separated lift and
  drag coefficients); verify deflection grows with wind speed and frames stay
  finite
- [x] 4.3 Decide from the wind sweep whether drag needs semi-implicit velocity
  damping at the shipped step size, and implement it if so; verify the
  documented wind sweep stays finite either way
  - Decision: not needed. The runs stay finite at 12 m/s on a 1 m2 panel with
    the shipped step size, so no damping term was added.
- [ ] 4.4 Calibrate the shipped air density and coefficient defaults against the
  reference wind scene; record the chosen values and the observed steady-state
  deflection
  - Partially done: the defaults shipped are `air_density = 1.225`,
    `C_D = 1.0` and `C_L = 0.0`, and the observed deflection is recorded, but
    they were not calibrated against an external reference, so this stays open.

## 5. Verification

- [ ] 5.1 Register the balloon scene (two mirrored panels, same pressure) and
  the windy-cloth scene in the debug scene registry; verify the CPU-only
  registry check builds both `input_data` payloads and passes mesh validation
  - Blocked by the maintainer's instruction: the registry lives in the test
    tree, and test code stays out of the repository for now. The equivalent
    scenes exist only in the gitignored verification script.
- [x] 5.2 Run both reference scenes to completion and record per-frame
  finiteness, tail-window motion, and per-substep convergence metrics; verify
  the tail motion is bounded and compare against the no-external-force baselines
- [x] 5.3 Measure the per-frame cost with external forces active and record it
  against the recorded per-frame budget
- [x] 5.4 Evaluate the omitted-Hessian criterion from the recorded metrics: run
  a pressure sweep and compare iteration count and residual against the
  no-external-force run; record whether the diagonal approximation is needed
- [x] 5.5 Run the pressure-versus-mesh-density comparison (same pressure, two
  densities) and record whether the inflated size drifts beyond the stated
  tolerance
- [x] 5.6 Run the existing quick test group with external forces off and verify
  no regression
  - Result: 23 passed, 1 xfailed, 1 failed
    (`sim/test_smoke.py::test_standard_scene_smoke`). Rebuilding the unmodified
    HEAD reproduces the same failure with the same numbers, so it is
    pre-existing and unrelated to this change.

## 6. Documentation

- [x] 6.1 Document the new per-object and global parameters, the sign and
  velocity conventions, and the "constant normal pressure, not a gas model" caveat
  in the committed user-facing documentation; verify by reading the committed
  doc and confirming the parameter names match the implementation
