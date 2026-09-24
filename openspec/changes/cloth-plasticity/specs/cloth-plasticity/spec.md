## Purpose

Gives the PDNewton cloth solver time-dependent persistent wrinkles: a rest
dihedral angle that a held deformation can move, and the internal friction that
makes how long it was held decide how hard the resulting crease is.

## ADDED Requirements

### Requirement: Plasticity is opt-in per panel and leaves other panels unchanged

Each cloth object's input SHALL carry a plastic flag, off by default. Only bend
entries whose owning panel sets the flag SHALL take part in the model; every
other entry SHALL keep its input rest angle and the solver's existing elastic
bending behavior, with no measurable cost when nothing is flagged.

#### Scenario: No panel is flagged

- **WHEN** a scene is loaded and simulated without the plastic flag on any
  object
- **THEN** per-frame vertex data is identical to the same run before the
  feature existed, and no plastic state is reported

#### Scenario: One panel of several is flagged

- **WHEN** a scene contains two panels and only one sets the flag, and the
  garment is deformed and released
- **THEN** only the flagged panel's rest angles change, and the other panel's
  rest angles remain the values it was loaded with

### Requirement: Rest angles come from the per-edge input and are established at scene init

Every bend entry SHALL take its rest angle from the `angles` value of the edge
it belongs to at scene init - a mesh edge from its own value, a seam hinge from
its hinge edge - with 0 meaning flat. An immutable copy SHALL be kept so the
input rest shape can be restored.

#### Scenario: Initial state

- **WHEN** a scene is loaded with a plastic panel
- **THEN** at frame 0 every reported rest angle equals the value the input
  provides for that entry, the anchor equals the rest angle, and both timers and
  the accumulated hardening strain are zero

#### Scenario: Scene reload

- **WHEN** a scene with accumulated plastic deformation is loaded again through
  the input-data call
- **THEN** the plastic state starts from the input rest shape of the new scene,
  not from the previous run

### Requirement: The rest shape is authored per edge through the mesh input

Each cloth mesh entry MAY carry an `angles` array and a `compress` array indexed
by that mesh's edges. A missing key, a missing element or a zero value SHALL
mean "no change". `angles` SHALL set the rest angle of that edge's bend entry,
and `compress` SHALL set the edge's rest length to the pattern length times
`1 + compress`. Both SHALL be read at scene init and SHALL NOT change while the
simulation runs.

#### Scenario: Arrays absent

- **WHEN** a scene is loaded without the two arrays
- **THEN** every edge keeps its pattern rest length, every bend entry keeps the
  flat rest angle, and the frame data matches a run before the feature existed

#### Scenario: Shrinking a region

- **WHEN** a set of edges carries a negative `compress`
- **THEN** those edges rest at the shortened length, so the authored region
  gathers once the cloth is simulated, and the rest of the panel is unaffected

#### Scenario: Growing a region

- **WHEN** a set of edges carries a positive `compress`
- **THEN** those edges rest longer and the region expands under the solver until
  the in-plane force balances

#### Scenario: Non-flat rest angle

- **WHEN** an edge carries a non-zero `angles` value
- **THEN** that entry's bending force pulls toward that angle instead of flat,
  and the reported rest angle for the entry is the input value at frame 0

### Requirement: Seam and internal-line rest angles come from the mesh input

A seam hinge's rest angle SHALL be the `angles` value on its hinge edge, and the
sewing input SHALL NOT carry a rest angle of its own. An internal line SHALL be
expressed the same way, by the values on the edges it covers.

#### Scenario: Seam hinge rest angle

- **WHEN** a seam whose hinge edge carries an `angles` value is loaded
- **THEN** the seam entry's rest angle equals that value at frame 0, and the
  sewing input's own angle field is not read

### Requirement: Both planar models express the authored rest length

The spring-mass model SHALL use the compressed rest length directly. The FEM
(BW) planar model, whose rest state is a per-triangle metric, SHALL apply the
same input as a scale of that metric derived from the triangle's three edges, so
a shrunk region gathers there as well. The energy weight and the vertex masses
SHALL stay those of the input pattern in both models.

#### Scenario: FEM model selected

- **WHEN** a scene with a negative `compress` is run with the FEM planar model
- **THEN** the authored region contracts in-plane, the run stays finite, and a
  vertex's mass is the same as without the input

#### Scenario: Degenerate input

- **WHEN** `compress` would collapse a triangle's rest metric
- **THEN** the metric is clamped instead, and the run continues without
  non-finite frame data

### Requirement: Internal friction holds a deformation below the slip threshold

Each bend entry SHALL carry an anchor angle. While the deviation between the
current angle and the anchor stays below the slip threshold the anchor SHALL NOT
move; when the deviation exceeds it, the anchor SHALL move toward the current
angle until only the threshold remains, and the deviation SHALL contribute to
the bending force alongside the elastic term.

#### Scenario: Small deformation recovers

- **WHEN** a flagged panel is deformed by less than the slip threshold and then
  released
- **THEN** it relaxes back to its rest configuration, within the tolerance the
  same deformation reaches without the feature

#### Scenario: Deformation beyond the threshold leaves a shifted anchor

- **WHEN** a flagged panel is bent past the slip threshold and held
- **THEN** the recovery on release stops short of the input rest shape by an
  amount that grows with the amount of sliding, and the reported anchor has
  moved with the deformation

### Requirement: The dwell effect makes a long hold harder to recover

With the time scale above zero the slip threshold SHALL grow with the time the
entry has spent sticking, from its initial value toward the dwell ceiling at the
configured time constant, and SHALL reset to the initial value when the entry
slips.

#### Scenario: Longer stick state resists recovery more

- **WHEN** the same moderate deformation is held twice with different time
  scales (a short and a long effective hold) and then released
- **THEN** the longer effective hold leaves the larger residual deformation

#### Scenario: Slip resets the dwell

- **WHEN** an entry slips after having accumulated stick time
- **THEN** its stick timer returns to zero and its slip threshold returns to the
  initial value

### Requirement: Time scale zero evaluates the model without time evolution

The time scale SHALL default to zero, in which case the dwell and hardening
timers SHALL NOT advance and both thresholds SHALL be evaluated at t = 0. Values
above zero SHALL scale how fast the timers advance, without changing the model
itself.

#### Scenario: Default time scale

- **WHEN** a flagged panel is simulated with the default time scale
- **THEN** both timers stay at zero and the run reproduces the model evaluated
  with the initial slip threshold and the initial hardening stiffness

#### Scenario: Scaled time

- **WHEN** the same scene is run with the time scale increased by a factor
- **THEN** the timers advance by that factor per substep and the residual
  deformation moves toward the long-hold result of the unscaled run

### Requirement: Plastic flow with a yield threshold leaves a residual deformation

The rest angle of a flagged entry SHALL change only while the elastic part of
its bending strain exceeds the yield threshold, and then only toward the current
angle, with the share of the excess that becomes plastic set by the current
hardening stiffness. A rest angle SHALL never move past the current angle.

#### Scenario: Below the yield threshold

- **WHEN** a flagged panel is deformed below the yield threshold and released
- **THEN** it returns to its rest shape, and no plastic deformation is reported

#### Scenario: Above the yield threshold

- **WHEN** a flagged panel is deformed beyond the yield threshold and released
  with the deforming support removed
- **THEN** a measurable part of the deformation remains and it is far larger
  than for the same deformation on an unflagged panel

#### Scenario: No overshoot

- **WHEN** a flagged panel is held far beyond the yield threshold for a long run
- **THEN** every rest angle stays between its initial value and the current
  angle, and the frame data stays finite

### Requirement: Time-dependent hardening makes a longer hold leave a larger residual

While an entry yields, the accumulated hardening plastic strain SHALL raise its
yield threshold and the hardening stiffness SHALL decay from its initial value
toward its lower bound at the configured time constant, so that a longer hold
produces a firmer crease. The hardening timer SHALL reset when the plastic
direction reverses.

#### Scenario: Hold-time ordering

- **WHEN** the same deformation is held for a short and for a long effective
  duration and then released
- **THEN** the longer hold leaves the larger residual deformation, and this
  ordering holds across the configured hardening range

#### Scenario: Direction reversal resets the timer

- **WHEN** a plastically deformed entry is bent back the other way past its
  yield threshold
- **THEN** its plastic timer restarts from zero and its plastic direction
  changes sign

### Requirement: Freeze commits the current shape as the rest shape

The simulator SHALL expose a freeze operation that adopts the current
configuration as the rest shape for flagged entries and moves no vertex when it
is applied.

#### Scenario: Wrinkled cloth stays wrinkled when suspended

- **WHEN** a flagged panel is deformed into a wrinkled configuration, frozen,
  and then released from the deforming support
- **THEN** the vertices stay within a small tolerance of the frozen
  configuration instead of relaxing back to the flat input pattern

#### Scenario: Freeze is repeatable and does not move vertices

- **WHEN** the freeze operation is applied, and applied again with no simulation
  step in between
- **THEN** the vertex positions read on the freeze frames are unchanged and the
  resulting rest shape is the same as after a single application

### Requirement: Reset and scene load return to the input rest shape

The simulator SHALL expose a reset that restores the input rest angles, clears
the anchors, timers and accumulated hardening, and returns the simulation to the
elastic reference without reloading the scene.

#### Scenario: Explicit reset

- **WHEN** a run with accumulated plastic deformation and a frozen shape is
  reset
- **THEN** subsequent frames behave like a purely elastic run of the same scene
  from the current configuration, and the reported plastic state is empty

### Requirement: The model runs only on PDNewton and only where a rest angle exists

The plastic and friction state SHALL advance only under the PDNewton solver. The
other solvers SHALL keep their current behavior. The bending model SHALL carry a
rest angle for the feature to have any effect; the documentation SHALL state
that the quadratic IBM bending model has none.

#### Scenario: Another solver is selected

- **WHEN** a scene is run with VBD, XPBD or Explicit even though panels are
  flagged
- **THEN** the run behaves as it does today and no rest angle changes

#### Scenario: IBM bending model

- **WHEN** a flagged panel is run with the quadratic IBM bending model
- **THEN** the simulation continues on the elastic path and the documented
  limitation is the only statement about the combination

### Requirement: The plastic state is readable through the public API

The simulator SHALL expose the current plastic state - per bend entry, the rest
angle, the anchor, the yield angle and both timers - so callers, the frontend
and the test suite can inspect it without reaching into internal buffers.

#### Scenario: State after a deformation

- **WHEN** a deformation is applied to a flagged panel with plasticity enabled
  and then released
- **THEN** the reported state reflects the deformation that became permanent,
  and it returns to the input values after a reset

#### Scenario: State while nothing is flagged

- **WHEN** no panel is flagged
- **THEN** the reported state equals the input rest shape for every entry and no
  time has been accumulated

### Requirement: Rigid-body motion does not induce plasticity

The model SHALL be driven by the dihedral angles of the entries, so moving the
whole garment without deforming it SHALL NOT change the plastic state.

#### Scenario: Rigid translation and rotation

- **WHEN** a flagged panel is translated and rotated rigidly while the timers
  are advancing
- **THEN** the plastic state is unchanged and the panel returns to its previous
  shape when the motion stops
