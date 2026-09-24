## Purpose

Lets a caller see where one simulator `update` call spends its time, as a flat
dict of milliseconds per frame-level stage, without slowing the frame down.

## ADDED Requirements

### Requirement: Timing is opt-in and inert when disabled

The simulator SHALL record no timings unless the `profile_timing` parameter is
set to a non-zero value. With the feature off, an `update` call SHALL take the
same code path it takes today apart from one branch per stage, and the timing
readback SHALL report that it is disabled.

#### Scenario: Default configuration

- **WHEN** a scene is simulated without setting `profile_timing`
- **THEN** the timing readback reports `enabled` false and every reported
  duration is zero

#### Scenario: Enabled

- **WHEN** `profile_timing` is set before the scene is loaded
- **THEN** the timing readback reports `enabled` true and the durations of the
  frames that follow

### Requirement: The stages partition one frame

The readback SHALL report, for the last completed frame, the total time of the
`update` call and the time of each frame-level stage: the frame bookkeeping
(normals, pick, pin and sewing state), the collision phase, the rest of the
substep loop (external forces, plastic state, the Newton iteration, the seam
projection), and the end-of-frame work. The stages SHALL partition the frame -
they SHALL add up to the total - so the collision phase SHALL include the
contact work a solver performs inside its substep loop (the BVH refit and the
broad phase of every substep), not only the frame-level preparation. The stages
SHALL be measured on the stream the work runs on, so a stage that only queues
work asynchronously is still measured to completion.

#### Scenario: Stage breakdown of a frame

- **WHEN** a frame is simulated with the feature on and the timings are read
  back
- **THEN** the reported frame index matches the frame that was run, every stage
  duration is finite and non-negative, the stages add up to the total, and the
  collision stage is non-zero whenever the solver queried contacts in that frame

### Requirement: Measurement never blocks the frame

Recording a stage SHALL be asynchronous and SHALL NOT synchronize the device.
The readback SHALL resolve the events only when it is called, and SHALL return
the most recent frame whose sample is complete; if that sample is not ready yet
it SHALL report the previous one instead of waiting.

#### Scenario: Readback does not stall the simulation

- **WHEN** the timings are read back after every frame of a run
- **THEN** the run completes with the same frame data as the same run without
  the readback, and no stage sample is reported before its events have
  completed

#### Scenario: Sample not ready

- **WHEN** the readback is called before the most recent frame's events have
  completed
- **THEN** the previous completed sample is returned and the readback reports
  that the value is stale

### Requirement: The engine reports one frame and nothing else

The readback SHALL describe exactly one frame - the last completed one - as a
flat mapping of stage name to milliseconds, with keys that carry no unit
suffix. It SHALL NOT report averages, history or a per-substep or per-kernel
breakdown beyond the collision total, and it SHALL be available for every
solver: the stages are boundaries of the shared update path, and a solver that
brackets no contact work of its own reports a zero collision stage.

#### Scenario: Flat single-frame mapping

- **WHEN** the timings are read back
- **THEN** the result is a flat mapping whose values are numbers, with no nested
  structure, and calling it repeatedly without stepping the simulation returns
  the same sample

#### Scenario: Another solver

- **WHEN** a scene is run with a solver other than PDNewton and the feature is on
- **THEN** the same stages are reported, with the substep loop covering whatever
  that solver does inside it
