## Purpose

Keeps a cloth that has come to rest on another cloth or on itself at rest, so
real-time editing and near-rest scenes do not ring, while penetration stays
bounded and the frame budget is unchanged.

## ADDED Requirements

### Requirement: A resting contact configuration settles

For a configuration whose cloth has landed and is only held by contact, the
per-substep motion SHALL decay over the measurement window instead of staying
at a constant level, and at the end of the window the median per-substep
maximum displacement SHALL be at most 10% of the value measured for the same
scene and parameters before this change.

#### Scenario: Two flat layers resting on the ground

- **WHEN** two flat cloth layers are dropped onto the ground with the shipped
  parameter block and the default mesh density, and the run is measured over
  its tail window after both layers have landed
- **THEN** the median per-substep maximum displacement is at most 10% of the
  pre-change value for the same scene, and the motion does not increase from
  the first to the last quarter of the window

#### Scenario: A single layer at the finest supported density

- **WHEN** a single flat layer at the finest supported mesh density rests on
  the ground with the shipped parameter block
- **THEN** the median per-substep maximum displacement is at most 10% of the
  pre-change value, for both the spring-mass and the planar finite-element
  constitutive model

### Requirement: Stability does not come from smaller substeps

The contact behaviour SHALL be stable at the step size the caller asks for:
the number of internal substeps per `update(dt)` call SHALL NOT increase, and
the shipped default `step_h` SHALL NOT change as part of this capability.

#### Scenario: Substeps unchanged

- **WHEN** one `update(dt)` call is made with the shipped parameter block
- **THEN** the engine performs the same number of internal substeps as before
  the change, and the settling scenarios above still hold

### Requirement: Penetration stays bounded

The contact response SHALL keep the mid-surfaces of two resting layers from
crossing by more than the modelled cloth thickness, i.e. the maximum
penetration depth SHALL NOT exceed the combined thickness of the contacting
shells. Removing penetration completely is explicitly not required.

#### Scenario: Resting stack penetration

- **WHEN** two flat layers rest on each other with the shipped parameter block
- **THEN** the deepest valid penetration stays below the combined shell
  thickness throughout the tail window, and no frame reports non-finite
  vertices

#### Scenario: Deep penetration does not blow up

- **WHEN** two layers are initialised overlapping by several times the shell
  thickness
- **THEN** every vertex stays finite and the per-substep displacement stays
  bounded by the configured contact search radius

### Requirement: The frame budget is preserved

With the new contact behaviour enabled, the measured real-time speed of the
garment scene used for the round-1 baseline SHALL NOT fall more than 10% below
its pre-change value, using the established interleaved measurement protocol.

#### Scenario: Baseline scene cost

- **WHEN** the baseline garment scene is measured with the new behaviour
  enabled and disabled in the same binary
- **THEN** the enabled run is within 10% of the disabled run, or the change is
  rejected for that mechanism and the next candidate in the design is tried

### Requirement: Contact-free scenes are unchanged

A scene with no valid contact SHALL produce the same trajectory as before the
change, so the new behaviour cannot silently alter scenes that never touch
anything.

#### Scenario: No contact, no change

- **WHEN** a single flat layer rests on the ground at the default density with
  no valid contact pair
- **THEN** the vertex positions match the pre-change run within float
  tolerance

### Requirement: Quality invariants are preserved

The drape acceptance set SHALL keep its existing invariants (finite frames,
bounding envelope, seam closure, area preservation, attached-vertex drift)
within the tolerances that set already uses, with the new contact behaviour
enabled.

#### Scenario: Acceptance set

- **WHEN** the drape acceptance set runs with the new behaviour enabled
- **THEN** every element passes the same invariant checks it passes today

### Requirement: The new behaviour is selectable at runtime

The solver SHALL expose a parameter that selects the new contact behaviour or
the current one, so a scene can be A/B tested in one binary and the default
can be changed only after the requirements above pass.

#### Scenario: A/B in one binary

- **WHEN** the parameter is set to the current behaviour before the first
  update
- **THEN** the frame behaves exactly as it does today, including the measured
  jitter levels recorded in the proposal
