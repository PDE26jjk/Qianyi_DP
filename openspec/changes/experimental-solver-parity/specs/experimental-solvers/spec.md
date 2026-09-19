## Purpose

Defines what the non-PDNewton solvers (VBD / XPBD / Explicit) must model to be
usable as test solvers: the same shared physics pipeline as PDNewton, parameter
driven damping, collision data that matches the substep it is used in, and a
validated parameter block - without claiming production drape parity.

## ADDED Requirements

### Requirement: Experimental solvers model the shared pipeline

Each experimental solver (VBD, XPBD, Explicit) SHALL simulate the same
`input_data` contract as PDNewton, covering the membrane constitutive models
offered by `constitutive_model_planar`, the bending models offered by
`bending_model`, the stitch constraints built from `sewings`, and
contact/ground response. Selecting a model or turning a term off SHALL have the
effect the parameter documents.

#### Scenario: Membrane constitutive model is honoured

- **WHEN** the same scene is simulated by an experimental solver with
  `constitutive_model_planar` set to the spring lattice and then to the FEM
  membrane
- **THEN** both runs complete with finite data and their final positions differ,
  i.e. the switch is not ignored

#### Scenario: Bending model is honoured

- **WHEN** the same scene is simulated by an experimental solver with bending
  disabled (`bending_k` = 0) and then with a bending model and a non-zero
  stiffness
- **THEN** the bending stiffness measurably changes the final shape

#### Scenario: Stitch chains hold a seam

- **WHEN** a scene whose two panels are joined only by stitch chains, with the
  upper panel pinned and the lower one free, is simulated by an experimental
  solver
- **THEN** the maximum distance between the paired seam vertices stays within
  the seam tolerance for the whole run, and the free panel remains attached
  rather than falling away

#### Scenario: Contact and ground are respected

- **WHEN** cloth falls onto an obstacle or onto the ground plane in an
  experimental solver
- **THEN** the frame data stays finite and the penetration depth stays bounded
  instead of the cloth passing through

### Requirement: Damping is parameter driven

Each experimental solver SHALL apply velocity damping only through its damping
parameter, and SHALL NOT apply an unparameterized fixed velocity decay. With
damping set to zero the solver SHALL NOT decay velocities, so gravity integrates
as the solver's time integration prescribes.

#### Scenario: Zero damping removes the decay

- **WHEN** a free-falling scene is simulated with the damping parameter set to
  zero
- **THEN** the measured downward acceleration matches the solver's integration
  of gravity within tolerance, i.e. no fixed `exp(-h*k)` decay is applied

#### Scenario: Damping parameter changes the motion

- **WHEN** the same scene is simulated with the damping parameter set to a
  large value
- **THEN** the motion is measurably slower than with damping off

### Requirement: The seam projection is shared by every solver

Every solver SHALL run the engine's geometry-side seam projection (and the
velocity pass that removes the snap's velocity kick) once per substep, so a
stitched garment is welded by the same mechanism whichever solver is selected.
The soft stitch constraint SHALL remain active as well: it holds the seam before
the projection's activation gate opens.

#### Scenario: A stitched seam is welded once the projection gate opens

- **WHEN** a scene whose panels are joined only by stitch chains is simulated
  with the projection activation frame set inside the run
- **THEN** the paired seam vertices end within 0.1 mm of each other for every
  solver, i.e. the projection ran

#### Scenario: The stitch constraint is still what holds the seam early

- **WHEN** the same scene is simulated with `sewing_k` set to zero, so the soft
  constraint is gone
- **THEN** the panels separate by metres even with the projection enabled, so
  the projection is not standing in for a missing stitch constraint

### Requirement: Collision data matches the substep it is used in

Each experimental solver SHALL refresh the collision structure it queries at
least once per substep, so the broad phase and contact response reflect the
positions of the substep being solved rather than the start of the frame.

#### Scenario: Penetration stays bounded as substeps shrink

- **WHEN** the same cloth-on-obstacle scene is simulated with a large substep
  and with a smaller substep of the same frame
- **THEN** the maximum penetration depth stays within the contact radius in both
  runs

### Requirement: Shipped parameter block is a working configuration

The parameter block shipped for each experimental solver SHALL be a
configuration that solver actually implements: a run with its block on the
standard procedural scene completes with finite data, pinned vertices within
tolerance, free vertices moving above the motion threshold, and no parameter
that the solver ignores.

#### Scenario: Preset run completes

- **WHEN** the standard procedural scene is driven with an experimental
  solver's shipped parameter block
- **THEN** the run completes with finite data, pinned drift within tolerance and
  free-vertex motion above the threshold
