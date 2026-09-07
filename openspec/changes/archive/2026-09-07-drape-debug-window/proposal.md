# Proposal: drape-debug-window

## Why

Testers currently verify drape-simulation health only through headless batch
runs and post-mortem artifacts (`frames.npz`, `traces.json`). There is no way
to watch a simulation live or to provoke failures by hand, so reproducing a
blow-up requires editing test code and re-running pytest. The Blender frontend
offers this interactivity but is heavy, GPU-tied to a Blender install, and not
part of this repo's test workflow.

## What Changes

- Add a standalone interactive debug window (`tests/frontend/`) built on the
  already-required Warp runtime (`warp.render.OpenGLRenderer`), driven by the
  engine's existing interactive bindings (`update`, `pick_triangle*`,
  `get_simulation_data`, `set_solver`, `set_parameters`). No C++ changes.
- Provide an extensible scene registry with two initial sources: procedural
  harness scenes (`tests/harness/meshspec.py`) and GarmentCodeData elements
  (`tests/harness/gcd/loader.py`); new debug scenes are added by registering a
  loader.
- Cloth dragging with the mouse (engine `pick_triangle` constraint semantics,
  as used by the Blender frontend), alongside Warp's built-in camera controls
  (left-drag orbit, scroll zoom, WASD pan).
- Blow-up detection rendered in the window: non-finite vertices or per-frame
  displacement beyond a configured threshold auto-pauses the simulation,
  flags the cloth red, and prints the frame index and stats to the console.
- Screenshot capture: a hotkey saves the current frame to a PNG under the
  gitignored test-artifacts tree, and a scripted non-interactive mode runs a
  configured number of frames (optionally performing a scripted drag), writes
  screenshot(s), prints the written paths, and exits — so a multimodal agent
  can consume the visual result of a run.
- Programmatic pick testing: the cursor-ray/triangle-hit/drag-plane math and
  the pick lifecycle live in a renderer-free module and are covered by a
  GPU-marked pytest that drives a full press/drag/release cycle without
  opening a window.
- All settings (solver, parameter block, fps/dt/substeps, scene, thresholds)
  are code constants at the top of the entry script; no GUI settings panels.
- Drive semantics default to the test harness (24 fps, `dt=0.001`,
  `ceil((1/24)/0.001)` substeps per rendered frame) so what the tester sees
  matches what the batch tests execute.

## Capabilities

### New Capabilities

- `drape-debug-window`: an interactive, code-configured Warp debug window in
  `tests/frontend/` that loads registered debug scenes, steps the simulation
  live, supports cloth dragging, and auto-flags blow-ups.

### Modified Capabilities

(none — no existing spec-level behavior changes; the window only consumes
existing engine and harness APIs)

## Impact

- New code: `tests/frontend/` package (entry script, scene registry,
  renderer-free picking module), plus tests: a CPU-only pytest that every
  registered scene builds a valid `input_data`, renderer-free unit tests for
  the ray math, and a GPU-marked headless pick-cycle integration test.
- New artifacts: screenshot PNGs under `tests/artifacts/frontend/<scene>/`
  (covered by the existing gitignored artifacts rule).
- Dependencies: `warp-lang` and its bundled `pyglet` become explicit
  development dependencies (`requirements-dev.txt`); the window requires a
  CUDA-capable Warp runtime and skips nothing at runtime (it is a manual tool,
  not part of CI).
- Reused as-is: `tests/conftest.py::_resolve_qydp` (module resolution),
  `tests/harness/presets.py` (solver blocks), `tests/harness/meshspec.py`,
  `tests/harness/gcd/loader.py`, `QYDP_GCD_ROOT`/`QYDP_GCD_BODY` env vars.
- The `warp_drape_viewer.ipynb` notebook stays (offline frame replay); the
  window covers the live/interactive use case.
