# Proposal: frame-stage-timing

## Why

The simulator reports a frame as one opaque `update(dt)` call, so neither the
frontend nor a profiling session can say where the frame time goes. The only
existing facility is `PerfTiming`, a host-side nested timer that nothing in the
simulation path uses and that cannot see asynchronous kernel time at all, while
the mesh sampler already shows the useful pattern for this codebase
(`QYDP_SAMPLE_PROFILE=1` prints CUDA-event phase timings for `sample_points`).
Without a per-stage breakdown, "the simulation is slow" cannot be attributed to
the collision phase, the substep loop, or the frame bookkeeping.

## What Changes

- Add an opt-in, per-frame stage timer to the simulator built on persistent CUDA
  events: one event pair per stage, recorded on the stream the stage runs on,
  never synchronized on the hot path, and resolved only when the caller asks.
- Measure four frame-level stages plus the total: `frame_update` (normals, pick,
  pin, sewing bookkeeping), `collision` (broad and narrow phase), `substeps`
  (the whole substep loop, i.e. external forces, plastic state, Newton
  iteration and seam projection) and `end_frame`.
- Expose it to Python as a flat dict of milliseconds for the last completed
  frame: `qydp.simulator.get_timing()` ->
  `{"frame": int, "stale": bool, "total_ms": float, "frame_update_ms": float,
  "collision_ms": float, "substeps_ms": float, "end_frame_ms": float}`.
  Averaging, history and presentation stay in the frontend.
- Enable it with the `profile_timing` parameter (default off), so a run that
  does not ask for timings pays nothing.

Non-goals:

- No per-substep or per-kernel breakdown. With the shipped `pd_cuda_graph = 1`
  the whole Newton loop is one captured graph (an event recorded inside a
  capture becomes a graph node), so the loop is reported as `substeps_ms` as a
  whole. A finer split would require disabling the graph, which costs more than
  the measurement is worth.
- No averaging, rolling window or history in the engine: one frame, one dict.
- No CPU-side timing of the Python call path or of `get_simulation_data()`; the
  frontend can time those itself with `time.perf_counter()`.

## Capabilities

### New Capabilities

- `frame-stage-timing`: an opt-in CUDA-event timer that reports the per-frame
  breakdown of one simulator `update` call as a flat dict of milliseconds.

### Modified Capabilities

(none)

## Impact

- Engine: a new `src/simulation/frame_timing.{cuh,cu}` (event pool, per-frame
  recording, on-demand resolution), a new parameter key, hooks at the four
  stage boundaries in `Simulator::update`, and a new binding in
  `simulator_interface.*` / `main.cpp`. No existing call signature changes.
- Performance: off by default (one branch per stage). On, it costs one
  `cudaEventRecord` per stage per frame (~1 µs each) and no synchronization;
  the `cudaEventElapsedTime` calls happen only inside `get_timing()`.
- Documentation: the parameter and the API in `docs/engine_input_spec.md`, a
  short README paragraph.
- Verification: one API test (dict shape, monotonic frame index, positive
  stage values that add up to the total, inert when the parameter is off).
