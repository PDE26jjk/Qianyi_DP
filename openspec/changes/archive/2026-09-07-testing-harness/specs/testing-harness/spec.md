## Purpose

Provide Qianyi_DP with an automated test system for developers and AI Agents: batch-verify simulation invariants, geometry-algorithm correctness, and API consistency using procedural scenes and the standard solver configuration, emitting machine-readable per-frame data and optional animation visualization so that every CUDA change can be verified quickly and repeatably.

## ADDED Requirements

### Requirement: Procedural scenes and standard simulation driver

The system SHALL support driving simulation tests through procedural scenes (no Blender dependency), using the standard solver configuration (PDNewton plus the commonly used parameter block) and update step sizes not exceeding 0.001s.

#### Scenario: Standard scene runs without failure and data is finite

- **WHEN** a default scene (grid cloth with pinned vertices) is driven for a configured number of frames with the standard solver configuration
- **THEN** all frame data is finite and the test passes

#### Scenario: Pinned vertices stay fixed while the cloth moves

- **WHEN** a corner-pinned cloth scene is driven over multiple frames
- **THEN** the world-space displacement of pinned vertices stays within numerical tolerance and free vertices move beyond a threshold

### Requirement: Machine-readable test results by default

The system SHALL emit per-frame simulation data traces and machine-readable result files (per-frame data, summary JSON, JUnit-compatible report) for every test run, so scripts and AI Agents can inspect them directly.

#### Scenario: All artifacts present after a run

- **WHEN** a simulation test run completes
- **THEN** the artifact directory contains the per-frame data file, the summary JSON, and the JUnit report, recording the test name, parameters, and status

#### Scenario: Failures are locatable to a frame

- **WHEN** a per-frame assertion fails (e.g. non-finite values appear)
- **THEN** the result file marks the failure and records the actual frame index plus per-frame statistics of the failing data (e.g. max displacement / non-finite count) so an Agent can read them directly for debugging

### Requirement: Geometry algorithm correctness verification

The system SHALL verify geometry API results via properties and reference checks, including the minimum-spacing property of sampled points as defined by the sampling algorithm in the CUDA source (grid-based; NOT Poisson-disc), the topological validity of triangulation, and cross-checking against an independent reference implementation.

#### Scenario: Sampled points satisfy the source-defined spacing rule

- **WHEN** sampling is executed for any procedural polygon
- **THEN** the distance between any two sampled points (including boundary points) satisfies the minimum-spacing rule derived from the sampling algorithm implementation, within the configured tolerance

#### Scenario: Triangulation covers the polygon and conserves area

- **WHEN** a procedural polygon is sampled and triangulated
- **THEN** the area sum of the output triangles equals the polygon area within tolerance, with no out-of-bounds or self-intersecting triangles

#### Scenario: Cross-check against an independent reference implementation

- **WHEN** triangulation is executed on the same random point set
- **THEN** the output is topologically consistent with the independent reference implementation (e.g. scipy spatial structures), in an equivalence sense

### Requirement: Experimental solver governance

The system SHALL mark tests for solvers that are not yet production-ready (VBD / XPBD / Explicit) as known failures; when such a solver becomes usable, the system SHALL surface that state change explicitly instead of passing silently.

#### Scenario: State flip of an experimental solver is surfaced explicitly

- **WHEN** a known-failing test for an experimental solver starts passing
- **THEN** the test run reports the state change as a failure (strict xfail semantics), requiring manual confirmation before the marker is adjusted

### Requirement: API consistency check

The system SHALL check self-consistency of the public simulator API, including that the solver enumeration interface and the solver-setting interface accept the same set of names.

#### Scenario: Inconsistency is reported as a known issue

- **WHEN** the set returned by the solver enumeration interface differs from the set accepted by the solver-setting interface
- **THEN** the test is identified as a known issue (currently a known failure) and the difference between the two sets is recorded in the results

### Requirement: Optional visualization artifacts

The system SHALL NOT generate animation visualization by default; GIFs SHALL only be generated when explicitly enabled, written to a gitignored artifact directory, and accompanied by a summary view of several test results.

#### Scenario: No GIF by default

- **WHEN** tests are run with default options
- **THEN** the artifact directory contains no GIF files - only data traces and result files

#### Scenario: GIF and gallery generated when explicitly enabled

- **WHEN** simulation tests are run with visualization enabled
- **THEN** each frame is rendered and a GIF is written to the gitignored directory, and a summary index (gallery) is generated for browsing multiple test results

### Requirement: Agent-friendly entry point and documentation

The system SHALL provide a quick single-command test group and Agent-facing documentation covering how to locate/build the module under test, which tests to run, where artifacts land, and how to debug failures; when the module under test is unavailable, the system SHALL skip explicitly with a reason.

#### Scenario: Explicit skip when the module under test is unavailable

- **WHEN** a test run cannot find a usable Qianyi_DP module (not built or unknown location)
- **THEN** the GPU-dependent tests are skipped and the report clearly states the missing reason and build instructions, rather than failing silently

#### Scenario: Agent runs the quick group per documentation

- **WHEN** an Agent runs the standard test group via the documented single command
- **THEN** the run outputs a pass / known-failure summary plus artifact paths, sufficient to judge whether a change broke existing behavior

### Requirement: Interactive exploration scenes

The system SHALL provide an interactive Warp scene notebook runnable cell by cell for human inspection of simulation behavior.

#### Scenario: Scenes are runnable cell by cell

- **WHEN** the interactive scene notebook is executed cell by cell
- **THEN** each cell runs independently, scene data is procedurally generated, and camera/state inspection capabilities are provided
