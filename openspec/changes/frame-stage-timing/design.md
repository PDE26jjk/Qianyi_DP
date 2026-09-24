## Context

See `proposal.md` for motivation. What shapes the approach:

- `Simulator::update` is a short host function: `Geometry::update_for_frame`,
  `Geometry::collision_detect`, `SolverBase::begin_frame`, the substep loop
  (`update_for_step` + `solver->step`), `Geometry::end_for_frame`. That is
  exactly the stage set the proposal fixes.
- Most of the work is queued asynchronously. A host-side timer (the existing
  `PerfTiming`) would measure kernel launches, not kernels.
- The PDNewton substep loop runs on `sim_work_stream()`, and the solver records
  its own join event back to the legacy stream at the end of the loop
  (`cudaEventRecord(stream_join, work_stream)` + `cudaStreamWaitEvent(0,
  stream_join, 0)`), so an event recorded on stream 0 after `step()` completes
  is correctly ordered behind the whole substep loop.
- With `pd_cuda_graph = 1` (the default) the entire loop is a captured graph, so
  individual kernels inside it cannot be timed with events - an event recorded
  inside a capture becomes a graph node.
- `cudaEventElapsedTime` needs both events complete; `cudaEventQuery` answers
  that without blocking.

## Goals / Non-Goals

**Goals:**

- A per-frame, per-stage breakdown that is accurate for asynchronous GPU work,
  costs nothing when disabled, and never synchronizes on the frame path.
- A flat, boring Python contract: a dict of milliseconds for one frame.

**Non-Goals:**

- Per-substep, per-kernel or per-iteration detail (it would require disabling
  the CUDA graph).
- Averaging, history, or an event timeline the engine keeps for the caller.
- CPU-side timing of the Python call path (the frontend can do that itself).

## Decisions

### D1. One event per stage boundary, with a two-slot ring

The stages are consecutive, so `StageCount + 1` boundaries describe them all:
`boundary_[i]` is the instant stage `i-1` ends and stage `i` starts, `boundary_[0]`
opens the frame and `boundary_[StageCount]` closes it. Stage durations and the
total are elapsed times between neighbouring boundaries, which also makes the
stages add up to the total exactly. The events are created once (event creation
costs tens of microseconds - exactly the overhead this feature must not add) and
re-recorded every frame into a two-slot ring, so the slot being recorded is
never the one a readback may still be resolving.

Measured: bracketing every stage with its own start *and* stop event (11 records
per frame) cost the same as the shared-boundary form (5 records per frame), so
the per-frame cost is dominated by having timing-enabled events on the frame
path at all, not by the record count. The cheaper form is kept because it is
also the one whose stages cannot disagree with the total.

### D2. Stages are bracketed on the legacy stream

Each stage is `begin(stage)` / `end(stage)`: `begin` records the start event,
`end` records the stop event. Everything is recorded on stream 0, which is
correct for all four stages because the solver hands control back to stream 0
at the end of its substep loop (Context), so `substeps_ms` covers the whole
loop including the work-stream work and seam projection. The alternative -
recording the substep events on `sim_work_stream()` - would need fork/join
events around every stage and would mis-measure the graph replay.

Measured cost of the enabled path: `cudaEventRecord` costs tens of microseconds
on this driver, so the five boundary records of the stage view are about
+0.3 ms per frame, and once the per-substep contact intervals are accumulated
(two records per substep) the cost is about +45 us per substep - roughly +3 ms
on a 26 ms frame that carries 14 substeps, and proportionally less on a
frontend frame, which subdivides into 3-4 substeps. The disabled path records
nothing and only pays one branch per stage.

### D2b. Repeating work inside a stage is accumulated, not nested by hand

The contact work a solver does inside its substep loop repeats, so it cannot be
one boundary interval. `begin_accum(label)` / `end_accum(label)` record one
interval per repetition, remember which stage was open around it, and at resolve
charge the sum to `label` and take it back out of the surrounding stage. That is
how `collision` (per-substep BVH refit plus broad phase, plus the frame-level
preparation) and `substeps` (the rest of the loop) stay disjoint and the four
stages keep partitioning the frame.

The pool holds `kMaxIntervals = 64` intervals per frame and slot (128 events per
slot, created lazily the first time a solver accumulates), which covers a frame
with up to 64 substeps; beyond that the frame reports the intervals that fit.
The alternative - one boundary pair around the whole loop and subtracting a
frame-level collision prepass - is what the maintainer rejected as meaningless.

### D3. Resolve on demand, never wait

`get_timing()` checks `cudaEventQuery(last_stop)`:

- complete: `cudaEventElapsedTime` per stage, store the sample as the current
  one, return it with `stale=False`;
- incomplete: return the previously resolved sample unchanged with
  `stale=True`.

Because the frontend reads the vertex array every frame (a synchronous copy),
the events of the previous frame are complete in practice; the stale path only
matters for a caller that polls faster than the GPU drains.

### D4. One parameter, read once per update

`profile_timing` (default `0`) is read at the top of `Simulator::update` through
the existing parameter map. When it is zero, `begin`/`end` are no-ops and the
only cost is the branch. It does not enter the PDNewton CUDA-graph capture key:
the timer brackets the stages on stream 0 *outside* the captured region, so
turning it on or off cannot change what the capture replays.

### D5. The Python contract

`qydp.simulator.get_timing()` returns a flat dict:

```text
{"frame": int,            # the frame the sample belongs to (-1 before any)
 "enabled": bool,         # whether profile_timing was on for that frame
 "stale": bool,           # the sample predates the most recent frame
 "total": float,
 "frame_update": float,
 "collision": float,
 "substeps": float,
 "end_frame": float}
```

Milliseconds, single frame, no nesting, no averaging, and no unit suffix in the
keys (the maintainer asked for both). `enabled` is part of the contract because
"the numbers are all zero" would otherwise be ambiguous between "the feature is
off" and "the sample is not ready".

*Alternative:* a nested dict with per-stage `last_ms` / `mean_ms` / `calls`. The
maintainer asked for the flat form and for averaging to live in the frontend.

### D6. Code placement

`src/simulation/frame_timing.cuh` declares the class and
`src/simulation/frame_timing.cu` implements it, next to the other simulation
translation units, so `simulator.cu` only gains the stage brackets. The class
holds no reference to `Geometry` or `SolverBase`, which keeps it reusable for
the pattern-side timer if that is ever unified with this one.

## Risks / Trade-offs

- [The event API adds host calls to the frame path] -> five records per frame at
  ~1 us each against a frame of milliseconds, and only when enabled; the test
  records the measured overhead as information rather than asserting a
  threshold.
- [`substeps_ms` hides the split between assembly, the linear solve and the
  seam projection] -> accepted by design: the graph makes that split
  unmeasurable without giving up more than the measurement is worth; the
  `pd_cuda_graph = 0` path is the escape hatch for a detailed study.
- [A stale sample could be read as a fresh one] -> the dict carries `frame` and
  `stale`, so a caller can always tell.
- [Event reuse across frames could race a slow readback] -> two slots, and the
  slot is only rewritten after the same slot's previous events have completed
  (the query in D3 is per slot).

## Migration Plan

Purely additive and off by default: no scene, test or frontend changes behavior
unless it sets `profile_timing`. Rollback is unsetting the parameter or
reverting the change.
