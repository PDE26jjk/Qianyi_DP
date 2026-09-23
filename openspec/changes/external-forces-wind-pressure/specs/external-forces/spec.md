## Purpose

Lets a scene drive cloth and closed shells with forces beyond gravity: a
per-object internal pressure that inflates sewn panel pairs into pillows,
plush bodies, and balloons, and a world-space wind field that makes garments
and accessories react to air movement.

## ADDED Requirements

### Requirement: External forces are opt-in and inert when unconfigured

Every new force input SHALL have an inert default, so a scene that does not
configure pressure or wind simulates exactly as it did before this capability
existed. Setting a force input back to its inert value SHALL restore that
behavior.

#### Scenario: Scene without external-force inputs

- **WHEN** a scene is loaded and simulated without any pressure or wind input
- **THEN** the resulting motion matches the pre-change behavior for the same
  scene, parameters, and solver

#### Scenario: Force inputs are reset to their inert value

- **WHEN** a scene that ran with pressure and wind is re-run with pressure
  zero and wind disabled
- **THEN** the motion returns to the no-external-force behavior for that scene

### Requirement: External forces are applied by the PDNewton solver only

The system SHALL apply pressure and wind forces when the active solver is
PDNewton. Any other solver SHALL ignore them: the scene still runs, and no
external-force term enters that solve.

#### Scenario: A non-PDNewton solver ignores external forces

- **WHEN** a scene that configures pressure or wind is run with a solver other
  than PDNewton
- **THEN** the run completes without external-force terms, and its frame data
  does not depend on the configured pressure or wind values

#### Scenario: PDNewton applies the configured external forces

- **WHEN** the same scene is run with PDNewton
- **THEN** the external forces are applied and the motion differs from the
  no-external-force run in the direction the configured forces describe

### Requirement: Per-object constant normal pressure

The system SHALL accept one pressure value per object, and SHALL apply it as a
distributed load along the object's surface normals with a magnitude
proportional to surface area. A positive value SHALL push the surface along its
normal (expanding a closed shell) and a negative value SHALL pull it inward. The
model SHALL NOT require or expose a target volume, rest volume, or gas state;
the inflated size of a shell is an outcome of the pressure value and the
material stiffness.

#### Scenario: Mirrored panel pair inflates into a closed shell

- **WHEN** two mirrored panels are sewn along their boundary into a closed
  shell and the same positive pressure is applied to both
- **THEN** the shell inflates away from the seam into a puffed shape, and all
  frame data stays finite

#### Scenario: Zero pressure has no effect

- **WHEN** an object's pressure is zero
- **THEN** no pressure force is applied to that object

#### Scenario: Negative pressure collapses the shell

- **WHEN** a closed shell is given a negative pressure
- **THEN** the surface is pulled inward instead of outward

#### Scenario: Inflated size is not driven by a volume target

- **WHEN** a user inspects or configures a pressure-enabled object
- **THEN** no target-volume or rest-volume input is available on that object,
  and changing only the pressure value changes the puffed size monotonically

### Requirement: Pressure needs no watertight shell

Pressure is a distributed load, not a gas state, so the system SHALL accept open
panels and shells with holes: no enclosed volume, no rest volume and no
watertightness check participates in the model.

#### Scenario: Open panels are accepted

- **WHEN** a pressure-enabled object is an open panel
- **THEN** the load completes without requiring a watertight shell, because the
  model does not need an enclosed volume

### Requirement: Wind is a world-space velocity field with quadratic lift and drag

The system SHALL expose a world-space wind velocity field expressed as a base
velocity plus temporal gust and spatial turbulence variation. The aerodynamic
force on a surface SHALL be quadratic in the relative air velocity, SHALL use
the separated lift and drag coefficients with an air-density scale, and SHALL
depend on the surface normal. Turbulence SHALL vary the wind velocity field and
SHALL NOT be added directly to the force.

#### Scenario: Steady wind deflects a hanging panel downwind

- **WHEN** a hanging cloth panel is simulated in a steady wind whose direction
  is horizontal
- **THEN** the free part of the panel deflects along the wind direction and
  reaches a steady state, with all frame data finite

#### Scenario: Zero wind has no aerodynamic effect

- **WHEN** the wind velocity and its turbulence are zero
- **THEN** no aerodynamic force is applied

#### Scenario: Stronger wind deflects the panel further

- **WHEN** the wind speed is increased while the panel and material parameters
  are unchanged
- **THEN** the steady-state deflection increases

#### Scenario: Turbulent wind stays continuous and finite

- **WHEN** a wind field with gusts and turbulence enabled is simulated over
  multiple frames
- **THEN** the per-frame motion remains finite and does not show discontinuous
  jumps between adjacent frames

### Requirement: External-force parameters have documented defaults and overrides

The system SHALL accept global wind parameters and per-object pressure and wind
coefficient values, SHALL apply documented defaults for every key that a scene
omits, and SHALL NOT produce non-finite simulation data for any accepted input
combination.

#### Scenario: Scene omits the new keys

- **WHEN** a scene supplies only the pre-existing mesh fields
- **THEN** documented defaults are used and the scene loads and simulates
  successfully

#### Scenario: Per-object override changes that object only

- **WHEN** one object in a multi-object scene overrides a wind coefficient or
  sets its own pressure
- **THEN** the other objects keep their previous behavior under the same global
  wind

### Requirement: External forces stay finite and report convergence

At the documented reference magnitudes, scenes using pressure and wind SHALL
remain finite and SHALL reach a steady state, and the solver's per-substep
convergence metrics SHALL remain observable while external forces are active so
the force treatment can be evaluated from recorded data.

#### Scenario: Reference scenes remain stable

- **WHEN** the reference balloon scene and the reference wind scene run to
  completion at the documented magnitudes
- **THEN** no frame contains non-finite values and the tail of the run shows a
  bounded, non-growing motion

#### Scenario: Convergence metrics are available with external forces active

- **WHEN** a substep is solved with pressure or wind active
- **THEN** the convergence metrics for that substep can be read from the
  existing metrics interface without changing the scene
