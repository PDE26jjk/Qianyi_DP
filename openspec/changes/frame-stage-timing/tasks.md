## 1. Timer

- [x] 1.1 Add `src/simulation/frame_timing.cuh` and `frame_timing.cu` with the
  event pool, the two-slot ring and the on-demand resolution of design D1-D3,
  and list the file in `src/simulation/CMakeLists.txt`; verify the module still
  builds and imports
- [x] 1.2 Verify the timer is inert while disabled: a run without
  `profile_timing` records no events and reports `enabled` false with zero
  durations

## 2. Stages

- [x] 2.1 Bracket `frame_update`, `collision`, `substeps` and `end_frame` inside
  `Simulator::update` with the parameter read of design D4, and accumulate the
  solver's per-substep contact work (BVH refit plus broad phase) into
  `collision` per design D2b; verify the four stages add up to the total exactly
  and that the contact work is non-zero whenever contacts were queried
- [x] 2.2 Verify the substep stage is measured to completion (the solver joins
  the work stream back to stream 0) by comparing `substeps_ms` against a
  `pd_cuda_graph = 0` run of the same scene: both have to be positive and of the
  same order
- [x] 2.3 Record the measured overhead as information: a frame with the timer
  on against the same frame with it off

## 3. Interface and documentation

- [x] 3.1 Add `get_timing()` to `SimulatorInterface` and bind it in `main.cpp`,
  returning the flat dict of design D5; verify the module exposes it and that
  the keys match the contract
- [x] 3.2 Document `profile_timing` and the readback in
  `docs/engine_input_spec.md` and add a short README paragraph; verify no
  machine-specific paths

## 4. Verification

- [x] 4.1 Add an API case under `tests/api/` that asserts the dict shape, a
  monotonic `frame`, finite non-negative stage values summing to at most the
  total, `stale` false after a readback that follows a completed frame, and the
  disabled contract of task 1.2
- [x] 4.2 Verify the timing does not change the simulation: the frame data of a
  run with `profile_timing = 1` matches the same run with it off
- [x] 4.3 Run `python -m pytest -m quick` and confirm the result matches the
  pre-change baseline apart from the known `sim/smoke` failure

Measured on this machine: the stages add up to the total exactly (boundary
events are shared and the accumulated intervals are moved out of the stage that
encloses them); on a 40x40 sheet resting on the ground at 24 fps the frame
breaks down as total 26.30 ms = 0.09 frame_update + 7.70 collision +
18.19 substeps + 0.32 end_frame, where `collision` is the per-substep contact
work the solver runs itself. `cudaEventRecord` costs tens of microseconds on
this driver, so the enabled path is +0.3 ms with the boundary stages alone and
about +45 us per substep once the contact intervals are accumulated (~+3 ms on
that 26 ms frame, which carries 14 substeps; a frontend frame with 3-4 substeps
adds a fraction of that). The disabled path records nothing. `pytest -m quick`:
15 passed, 1 pre-existing `sim/smoke` failure.
