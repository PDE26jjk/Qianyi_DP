## Purpose

Gives the cloth solver a plastic rest shape so that wrinkles formed by contact,
compression or handling survive after the deforming support is removed, matching
the pressed-in creases that garment tools produce for denim-like materials.

## ADDED Requirements

### Requirement: Plasticity is opt-in and leaves the elastic behavior unchanged

The simulator SHALL keep its current purely elastic behavior unless plasticity
is enabled through the parameter API. With plasticity disabled (the default),
the rest shape SHALL be the one derived from the input pattern and simulation
results SHALL be unchanged.

#### Scenario: Default configuration is purely elastic

- **WHEN** a scene is loaded and simulated without setting any plasticity
  parameter
- **THEN** per-frame vertex data is identical to a run of the current
  implementation for the same scene, solver and parameter block

#### Scenario: Enabling plasticity

- **WHEN** plasticity is enabled through the parameter API before `input_data`
- **THEN** the simulation runs with the plastic rest shape active, and a
  subsequent run with plasticity disabled is again purely elastic

### Requirement: Freeze commits the current shape as the rest shape

The simulator SHALL provide an explicit operation that adopts the current
simulated configuration as the new rest shape. The operation SHALL NOT move any
vertex when it is applied.

#### Scenario: Wrinkled cloth stays wrinkled when suspended

- **WHEN** a cloth is deformed into a wrinkled configuration and the freeze
  operation is applied, and the deforming support is then removed
- **THEN** the vertices stay within a small tolerance of the frozen
  configuration instead of relaxing back to the flat input pattern

#### Scenario: Freeze preserves the visible configuration

- **WHEN** the freeze operation is applied at frame N
- **THEN** the vertex positions read at frame N are unchanged by the operation,
  and the following frames show no jump caused by the new rest shape

#### Scenario: Freeze is repeatable

- **WHEN** the freeze operation is applied twice with no simulation step in
  between
- **THEN** the resulting rest shape is the same as after a single application

### Requirement: Strain-driven plastic flow accumulates permanent deformation

With plasticity enabled, the simulation SHALL transfer the part of the elastic
strain that exceeds a yield threshold into the plastic state at a rate bounded
per step. Bending (rest dihedral angle) SHALL be the primary mechanism, and
stretch (rest edge length) SHALL be independently selectable. Plastic change
SHALL accumulate while the deformation is held and SHALL NOT be recovered when
the deforming load is removed.

#### Scenario: Below the yield threshold

- **WHEN** a cloth is deformed below the configured yield threshold and then
  released
- **THEN** it recovers to the configuration it would have reached without
  plasticity, within numerical tolerance

#### Scenario: Above the yield threshold

- **WHEN** a cloth is deformed beyond the yield threshold, held, and then
  released with the support removed
- **THEN** a measurable part of the deformation remains, and the remaining
  deformation is larger than for the same deformation released immediately

#### Scenario: Longer holding time hardens the deformation

- **WHEN** the same deformation is held for a longer time before being released
- **THEN** the residual deformation is larger (time-dependent hardening), and
  this ordering holds across the configured hardening range

### Requirement: Plastic update is bounded and numerically stable

The rest-shape update SHALL be rate-limited per step and SHALL keep the rest
shape a valid configuration. A run with plasticity enabled SHALL produce only
finite frame data, with no unbounded growth of the deformation or the plastic
state.

#### Scenario: Long run stays finite

- **WHEN** a reference scene is simulated for a full regression run with
  plasticity enabled
- **THEN** every frame is finite and no vertex leaves the scene bounding
  envelope beyond the tolerance used by the existing smoke test

#### Scenario: Per-step change is limited

- **WHEN** a large deformation is applied in a single step
- **THEN** the rest shape changes by at most the configured per-step limit and
  the simulation continues without NaN

### Requirement: Rigid-body motion does not induce plasticity

Plastic change SHALL be measured on the deformation of the cloth itself, so
that translating or rotating the whole garment does not create permanent
deformation.

#### Scenario: Rigid motion leaves the rest shape alone

- **WHEN** the whole cloth is moved rigidly (translation and rotation) with no
  internal deformation while plasticity is enabled
- **THEN** the plastic state is unchanged, and stopping the motion returns the
  cloth to its previous shape

### Requirement: Rest-shape changes propagate to derived quantities

Whenever the rest shape changes, every quantity derived from it SHALL be
updated before it is used again, so that elastic forces, area/bending terms and
the linear system agree within the same step.

#### Scenario: Frozen cloth hangs without residual forces

- **WHEN** a cloth is frozen into a wrinkled shape and then suspended
- **THEN** the cloth holds that shape without visible jitter, drift or a
  transient that pulls it back toward the flat input pattern

### Requirement: Plastic state resets with the scene

Loading a scene SHALL discard any previously accumulated plastic state, and an
explicit reset SHALL restore the purely elastic reference without reloading the
scene.

#### Scenario: Reloading a scene clears plasticity

- **WHEN** a scene with accumulated plastic deformation is loaded again through
  the input-data call
- **THEN** the simulation starts from the elastic reference of the new input

#### Scenario: Explicit reset

- **WHEN** the plastic state is reset without reloading
- **THEN** subsequent frames behave like a purely elastic run from the current
  configuration

### Requirement: Formulations that cannot express plasticity report the gap

When plasticity is enabled with a configuration that cannot represent a plastic
rest angle or a plastic rest length, the simulator SHALL report the unsupported
combination and SHALL fall back to elastic behavior instead of silently
producing no permanent deformation.

#### Scenario: Unsupported bending model

- **WHEN** bending plasticity is requested while the selected bending model has
  no rest-angle term
- **THEN** the simulator reports that plastic bending is unavailable for that
  model and the run continues with elastic behavior

### Requirement: Plastic state is readable through the public API

The simulator SHALL expose the current plastic state so callers, the frontend
and the test suite can inspect how much deformation has been made permanent,
without reaching into internal buffers.

#### Scenario: State after a deformation

- **WHEN** a deformation is applied with plasticity enabled and then released
- **THEN** the reported plastic state reflects the permanently deposited
  deformation, and it returns to empty after the explicit reset

#### Scenario: State while plasticity is disabled

- **WHEN** the feature is disabled
- **THEN** the reported plastic state is empty and no plastic time has been
  accumulated
