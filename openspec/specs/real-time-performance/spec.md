# real-time-performance Specification

## Purpose
Keeps the solver's frame budget measurable and tunable so that real-time speed
(simulated time per wall-clock second, RTS) can be improved without silently
changing simulation results or dropping contacts. The capability covers how the
frame's work is scheduled - the broad phase's query order and overlap, the
replay of the solver loop from a captured CUDA graph - and the defaults that
ship with it.

## Requirements

### Requirement: The solver loop is replayed from a captured CUDA graph

The Projective-Dynamics solver SHALL replay one whole iteration loop per
substep from a captured CUDA graph by default, and SHALL expose a parameter
that restores direct kernel launches so the two can be compared in the same
binary. The capture SHALL be rebuilt whenever a value baked into the recorded
launch arguments changes, and a capture failure SHALL fall back to direct
launches instead of failing the simulation.

#### Scenario: Default is the captured path

- **WHEN** a scene is driven without setting `pd_cuda_graph`
- **THEN** the solver issues one graph launch per substep instead of one
  launch per kernel, and the simulation produces finite vertices with the same
  per-frame displacement statistics as the direct-launch path

#### Scenario: Direct launches on request

- **WHEN** `pd_cuda_graph` is set to 0 before the first `update`
- **THEN** every kernel inside the iteration loop is launched directly

#### Scenario: A stale capture is never replayed

- **WHEN** any buffer identity or size, iteration count, linear-solver
  selection, or numeric parameter that the captured launches depend on
  changes
- **THEN** the captured graph is destroyed and rebuilt before it is replayed

### Requirement: Broad-phase query scheduling is switchable

The simulator SHALL run the three broad-phase queries (vertex-face,
edge-edge, edge-face) concurrently by default, and SHALL expose a parameter
that restores the sequential launch order so the two can be compared in the
same binary.

#### Scenario: Default is the overlapped path

- **WHEN** a scene is driven without setting `bvh_streams`
- **THEN** the three broad-phase queries run on separate streams and the
  narrow phase observes their candidate buffers only after all three
  complete

#### Scenario: Sequential fallback

- **WHEN** `bvh_streams` is set to 0 before the first `update`
- **THEN** the three queries are launched in order on the default stream and
  the simulation produces the same candidate set as the overlapped path

### Requirement: Broad-phase conservativeness and query order are switchable

The simulator SHALL inflate the tree and the query boxes only by the contact
radius by default (the tight, non-swept mode), SHALL map query `i` to the i-th
leaf in traversal order by default, and SHALL expose parameters that restore
the swept boxes and the natural primitive order.

#### Scenario: Tight mode and swept mode

- **WHEN** `tight_broad_phase` is 0
- **THEN** the query boxes are swept to the predicted position, and the
  simulation is conservative over the whole substep

#### Scenario: Query order mapping

- **WHEN** `bvh_query_order` is 0
- **THEN** query `i` addresses primitive `i` in mesh order instead of the
  i-th leaf of the tree, and the candidate set is unchanged

### Requirement: Scheduling must not change simulation results

Overlapping the queries, changing the query order, or replaying the iteration
loop from a captured graph SHALL NOT change which candidate pairs are found or
what the solver computes: each query writes a disjoint buffer, the narrow
phase SHALL only read a buffer after the query that fills it has completed,
and the captured graph SHALL record the same kernel sequence with the same
arguments.

#### Scenario: Statistics match the reference path

- **WHEN** the same scene and parameter block are simulated with and without
  the scheduling changes
- **THEN** finiteness, per-frame displacement statistics, cloth area, seam
  closure and body clearance agree within the engine's documented run-to-run
  variation

### Requirement: The shipped step length and iteration budget are the measured ones

The frame's step length and the solver's iteration budget SHALL ship at the
values the project measured, and the measurement SHALL record the quality
envelope of that choice - the step length is what sets the cloth's stretch in
this solver, while the iteration budget moves cost almost proportionally.
`dt` and `step_h` SHALL be set together so that a frame stays one substep;
raising `dt` alone scales the frame cost instead of the simulated time.

#### Scenario: Defaults are the validated ones

- **WHEN** the debug window or the study-1 notebook drives a scene without
  overriding the step length or the iteration counts
- **THEN** `dt = step_h = 0.0045` and `pd_iters = 5`, `linear_iters = 2` are
  used, and the recorded quality envelope matches the measurements in
  `design.md`

### Requirement: Performance changes are measured, not assumed

Changes to the frame's scheduling SHALL be evaluated with an interleaved A/B
measurement (both variants in the same binary or the same toolchain build),
reporting at least the min and median frame time over three or more rounds,
because the test GPU's clock state changes frame time by more than 20%
between runs. A stage-level timing SHALL always be read together with the
per-frame displacement statistics, because a change that alters the solver
trajectory can move cost into the collision pipeline.

#### Scenario: Evaluating a scheduling change

- **WHEN** a scheduling change is proposed
- **THEN** the change's record includes per-kernel Nsight Systems time, the
  frame-time A/B result, and the statistics used to confirm that no contacts
  were dropped and the trajectory did not diverge
