## MODIFIED Requirements

### Requirement: Experimental solver governance

The system SHALL run one smoke test per experimental solver (VBD / XPBD /
Explicit) as a normal test whenever the solver passes it: finite frame data,
pinned vertices within tolerance, free motion above the threshold, and the
stitch chains of a seamed scene held within the seam tolerance. A solver that
does not pass SHALL be recorded as a known failure together with its measured
failure mode, and a state change in either direction SHALL be surfaced
explicitly rather than passing silently.

#### Scenario: Experimental solver smoke test runs by default

- **WHEN** the smoke suite runs for VBD, XPBD or Explicit on the standard
  procedural scene
- **THEN** the case executes (it is not skipped) and reports pass or fail with
  its per-frame artifacts written

#### Scenario: Seam closure is checked for an experimental solver

- **WHEN** a scene whose panels are joined only by stitch chains is driven by an
  experimental solver
- **THEN** the paired seam vertices stay within the seam tolerance for the whole
  run

#### Scenario: The seam projection is covered for every solver

- **WHEN** the same scene is driven with the projection's activation frame set
  inside the run
- **THEN** the case asserts that the clusters merge (seam gap within 0.1 mm), so
  a solver that stops calling the geometry-side projection fails loudly

#### Scenario: State flip of an experimental solver is surfaced explicitly

- **WHEN** a known-failing test for an experimental solver starts passing, or a
  passing one starts failing
- **THEN** the test run reports the state change as a failure, requiring manual
  confirmation before the marker is adjusted
