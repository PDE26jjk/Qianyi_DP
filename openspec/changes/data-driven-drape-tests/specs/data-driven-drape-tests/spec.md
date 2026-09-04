## Purpose

Adds real-garment regression testing to the simulation harness: loading
GarmentCodeData elements into the simulator, sampling and running drape cases,
asserting invariants and comparing against reference drapes, and providing an
interactive Warp viewer for inspecting results.

## ADDED Requirements

### Requirement: GarmentCodeData element loading
The system SHALL load a GarmentCodeData element (box mesh, per-vertex panel and
stitch segmentation, semantic vertex labels, specification JSON) into the
simulation `input_data` contract: one cloth mesh per panel, vertex-pair sewings
derived from stitch labels, attachment constraints derived from semantic labels,
and (when configured) a body obstacle mesh.

The loader SHALL convert dataset units to meters, SHALL map UV-duplicate mesh
vertices back to their original IDs, and SHALL produce edges and triangles that
satisfy the existing mesh validation contract. The dataset location and body mesh
path SHALL come from environment configuration, never from committed paths.

#### Scenario: Load a valid element
- **WHEN** a complete element directory is provided with a body mesh configured
- **THEN** the loader produces a mesh list whose per-panel triangle and edge sets
  match the source mesh, sewings that pair every stitch-labeled vertex exactly
  once, and a body obstacle with valid normals.

#### Scenario: Load an element without a body
- **WHEN** no body mesh is configured
- **THEN** the loader still produces the cloth meshes and sewings, and the test
  run falls back to the invariant tier with gravity disabled.

#### Scenario: Malformed element
- **WHEN** an element is missing required files, has mismatched vertex counts, or
  contains stitch labels that cannot be paired
- **THEN** the loader raises a classified error (loader error) that is recorded in
  the case result and does not abort the rest of the batch.

### Requirement: Stratified sampling and batch execution
The system SHALL select elements for testing by stratifying on garment type and
mesh-size bucket, using a fixed random seed so a given configuration reproduces
the same selection. Each selected element SHALL run in an isolated subprocess with
a per-case timeout, and SHALL produce the existing per-case artifacts (results,
traces) plus a failure classification: loader error, simulation blow-up,
reference mismatch, or timeout.

#### Scenario: Reproducible selection
- **WHEN** the sampler runs twice with the same seed, type/size strata, and counts
- **THEN** both runs select the identical element list in the same order.

#### Scenario: Isolated batch run
- **WHEN** one element in a batch crashes or hangs
- **THEN** the batch continues, the case is marked with the corresponding failure
  class, and its artifacts are retained.

### Requirement: Drape invariant assertions
The system SHALL assert, per simulated drape, that all frames are finite, that no
vertex escapes a bounding envelope, that stitched vertex pairs remain within a
closure tolerance after sewing, that total cloth area stays within a preservation
tolerance, and that attached vertices stay within a drift tolerance.

#### Scenario: Invariant tier without a body
- **WHEN** a drape runs without a body mesh
- **THEN** gravity is disabled and the invariant assertions above are evaluated.

#### Scenario: Seam-closure check
- **WHEN** a drape has completed the forced-connect phase
- **THEN** every stitched vertex pair has distance below the closure tolerance,
  otherwise the case fails with class "simulation blow-up" or "reference
  mismatch" as configured.

### Requirement: Reference drape comparison
When a body mesh is configured, the system SHALL compare the simulated final
drape against the dataset reference drape (`sim.ply`) using rotation-insensitive
loose metrics: per-panel area distribution, z-extent, horizontal cross-section
quantiles, and mean surface distance. Comparison tolerances SHALL be configurable
per metric, and results SHALL be recorded even when below the failure threshold.

#### Scenario: Reference comparison passes
- **WHEN** the simulated drape is within all configured tolerances of the
  reference drape
- **THEN** the case passes and the metric values are recorded in the result file.

#### Scenario: Reference comparison fails
- **WHEN** any configured metric exceeds its tolerance
- **THEN** the case fails with class "reference mismatch" and the per-metric
  values are recorded for triage.

### Requirement: Performance regression records
The system SHALL record per-substep wall-clock time for every simulated element,
grouped by size bucket, in the case results. These records SHALL be gated behind
the benchmark marker and SHALL NOT hard-fail a run in this phase.

#### Scenario: Timing recorded
- **WHEN** a drape completes
- **THEN** its per-substep time and size bucket are written to the result file.

### Requirement: Interactive Warp viewer
The system SHALL provide an interactive notebook-based viewer that loads a
selected element's box mesh, simulated frames, reference drape, and body, and
lets the user scrub through frames, toggle overlays (initial / simulated /
reference / body), and open failing cases from a batch report. The viewer SHALL
assume a CUDA-capable Warp runtime, matching the project's CUDA-based test
environment.

#### Scenario: Inspect a completed drape
- **WHEN** the viewer opens a completed case
- **THEN** the user can step through frames and toggle the initial, simulated,
  reference, and body overlays.
