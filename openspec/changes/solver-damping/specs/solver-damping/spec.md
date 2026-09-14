## Purpose

Define the dissipation guarantees of the cloth solvers: a resting configuration
comes to rest, an undriven configuration loses kinetic energy at a bounded and
time-step-consistent rate, the static equilibrium is unaffected by damping, and
every dissipation mechanism is controllable through the parameter map.

## ADDED Requirements

### Requirement: A resting configuration settles

The solver SHALL dissipate the motion of a cloth configuration that rests on a
support under gravity with no external drive, so that the per-substep
displacement decays instead of holding a constant level. Reported per-substep
displacement SHALL be the median and the maximum over the free vertices of one
`update(dt)` call, measured in world space.

#### Scenario: Settled garment stops moving

- **WHEN** a garment that is already sewn and draped on a body is stepped for a
  tail window of at least 1000 substeps at the shipped substep size
- **THEN** the tail-window median per-substep displacement is below 1 % of the
  mean mesh edge length, the tail window does not trend upward against the first
  quarter of the window, and every frame is finite

#### Scenario: Gravity is the drive

- **WHEN** the same configuration is stepped with gravity set to zero and the
  same parameters
- **THEN** the median per-substep displacement decays by at least one order of
  magnitude within 2 s of simulated time

### Requirement: Undriven kinetic energy decays at a bounded rate

With every external load removed, the solver SHALL remove kinetic energy at a
rate controlled by the damping parameters, not only by the fixed end-of-step
velocity multiplier.

#### Scenario: Ring-down after a perturbation

- **WHEN** a cloth grid is released from a perturbed configuration with gravity
  and support forces disabled
- **THEN** the root-mean-square free-vertex velocity decays exponentially with an
  e-folding time no longer than the documented maximum, and the measured
  e-folding time changes by less than 20 % when the substep size is varied
  between 1 ms and the shipped substep size with the same parameters

### Requirement: Damping is applied inside the solve

Damping SHALL be part of the system the linear solve sees, so that it dissipates
the elastic modes themselves and does not introduce a new step-size stability
limit.

#### Scenario: No new stability limit at large substeps

- **WHEN** an undriven cloth grid with damping enabled is stepped at 1 ms, 2 ms,
  4.5 ms and 6 ms substeps
- **THEN** every run stays finite, the ring-down time follows the same trend as
  the undamped solver, and no run needs a smaller substep to remain finite than
  the undamped reference

#### Scenario: Post-processing alone is not sufficient

- **WHEN** the damping parameters are set to zero and only the end-of-step
  velocity multiplier and the frame-end velocity smoothing remain active
- **THEN** the resting-stability scenario above fails, documenting that the
  existing mechanisms alone do not satisfy this capability

### Requirement: Damping does not move the static equilibrium

Damping SHALL only dissipate; it SHALL NOT introduce a bias force. With the
drive held constant, the steady-state configuration reached with damping enabled
SHALL match the undamped steady state within numerical tolerance.

#### Scenario: Draped rest shape is unchanged

- **WHEN** the same scene is run to rest with damping disabled and with damping
  enabled at the documented default values
- **THEN** the per-vertex difference between the two rest configurations is
  within the documented tolerance and the collision penetration statistics stay
  within the documented band

### Requirement: Contact response dissipates relative normal velocity

An active contact SHALL dissipate the relative normal velocity of the pair while
remaining repulsive. The contact response SHALL NOT rely on Coulomb friction for
this: friction acts in the tangent plane, is capped by the friction coefficient
times the normal load, and is driven by per-substep slip rather than by relative
velocity.

#### Scenario: Approaching pair loses relative normal velocity

- **WHEN** two cloth layers rest on each other and one is given an initial
  normal velocity towards the other
- **THEN** the relative normal velocity decays monotonically with no sign
  reversal (no restitution), the pair never becomes attractive, and the
  penetration stays bounded by the documented maximum

#### Scenario: Friction is not the mechanism

- **WHEN** the friction coefficient is set to zero with contact damping enabled
- **THEN** the resting-stability scenario still passes

### Requirement: Seam cluster merge is velocity consistent

A hard seam cluster merge SHALL NOT inject kinetic energy. The merge is a
position projection by design and must stay envelope-free, but the velocity the
following substep observes SHALL be the momentum-consistent velocity of the
cluster, not the teleport displacement divided by the substep.

#### Scenario: Merge does not add energy

- **WHEN** a stitch cluster with a known offset and a known velocity is
  projected onto its target
- **THEN** the cluster's kinetic energy after the projection does not exceed the
  pre-projection energy (within tolerance) and the cluster's momentum is
  preserved

#### Scenario: Merged garment does not keep ringing

- **WHEN** the merge path runs for the whole assembly window of the reported
  garment scene
- **THEN** the tail-window median per-substep displacement satisfies the
  resting-stability requirement once the assembly window has passed

### Requirement: Damping controls are documented parameters

Every damping mechanism SHALL be selectable through the existing parameter map
with a documented name, unit, default value and safe range, and setting it to
zero SHALL restore the current behavior. The shipped defaults SHALL be inert
(zero) for the mechanisms that add a response, so an existing scene keeps its
behavior until its own parameter block enables them.

#### Scenario: Parameters are discoverable and neutral at zero

- **WHEN** a scene sets the damping parameters to zero explicitly
- **THEN** the run reproduces the pre-change solver behavior for that scene
  within tolerance

#### Scenario: Defaults do not change an existing scene

- **WHEN** the change is applied to a scene that does not set the new parameters
- **THEN** the scene reproduces its pre-change behavior within tolerance, and
  enabling the documented working values in the scene's own parameter block
  satisfies the resting-stability requirement

#### Scenario: Enabled values are documented and bounded

- **WHEN** a caller enables the mechanisms
- **THEN** the parameter names, units, working values for the reported scene and
  safe ranges are documented, and values outside the range are either clamped
  or rejected rather than silently producing an unusable cloth

### Requirement: Quality invariants and frame budget are preserved

The change SHALL NOT regress the existing acceptance invariants or the
interactive frame budget.

#### Scenario: Acceptance invariants hold

- **WHEN** the drape acceptance set and the self-contact scenes run with the
  documented damping defaults
- **THEN** the invariant tier passes (finite frames, bounding envelope, seam
  closure, area preservation, attached-vertex drift), resting penetration stays
  inside the documented band, and no new non-finite frame appears

#### Scenario: Frame budget holds

- **WHEN** the reference real-time scene is measured with damping disabled and
  with the documented defaults
- **THEN** the solver time per update call increases by no more than 5 %
