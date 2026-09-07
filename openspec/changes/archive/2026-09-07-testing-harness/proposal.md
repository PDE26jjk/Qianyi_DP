## Why

The only automated verification Qianyi_DP has today is a handful of timing-only benchmarks (sort / BVH / SDF / graph coloring) with no correctness assertions. Simulation results are only eyeballed manually in Blender, and the scenes, solver parameters, and verification logic used for that live in the maintainer's local test notebooks (kept out of this repository) - which have already drifted out of sync with the current API (legacy scene data is missing fields; the step sizes and parameter blocks no longer match the current solver). Neither developers nor AI Agents can quickly and repeatably answer "is the simulation still correct, are the geometry algorithms still correct" after a CUDA change.

This change establishes a testing framework for developers and AI Agents: batch-run correctness / invariant / API-consistency / performance-regression cases via pytest (Python tooling and environment details are recorded in the project's Python packaging files, not in OpenSpec artifacts), emitting machine-readable per-frame data traces and result files by default, with GIF as an optional visualization artifact.

## What Changes

- Add a `tests/` pytest suite plus a `tests/harness/` framework package (scene construction, driver, reference implementations, trace/rendering utilities).
- Add a procedural scene builder (`MeshSpec` / scene builder): defaults taken from the Blender frontend (fabric.py / pattern.py), no Blender dependency.
- Add machine-readable results only; machine-specific paths and local environment details are recorded in `LOCAL_DEV.md` (repo root, gitignored) and MUST NOT appear in artifacts or committed docs.
- Add a "Blender-equivalent semantics" simulation driver: PDNewton + the standard parameter block (locally validated canonical values; provenance in `LOCAL_DEV.md`) + `update(dt)` substeps (default dt=0.001).
- Add simulation tests: smoke (finiteness / pinned vertices / displacement / no fly-away), determinism, per-frame data trace export; VBD / XPBD / Explicit are marked as experimental solvers with `xfail` (the framework must surface explicitly when they become usable again).
- Add geometry correctness tests: `sample_points` (minimum-spacing property as defined by the sampling algorithm in the CUDA source, boundary containment, triangulation area conservation) and `delaunay_2d` (cross-check against scipy), plus a performance baseline for `sample_points` (it is scheduled for optimization).
- Add an API consistency test: `get_all_solver()` vs the accepted `set_solver()` names (currently a known failure, `xfail(strict)`).
- Default test artifacts: per-frame simulation data traces (npz/JSON) + `results.json` + `junit.xml`; `*.gif` only as an opt-in artifact written to a gitignored directory.
- Optional artifacts: a gallery summary and an interactive Warp scene notebook (`tests/notebooks/warp_scenes.ipynb`) runnable cell by cell.
- Add `AGENTS.md`, Agent-facing documentation: how to build/locate the pyd, which test group to run, where artifacts land, how to debug failures (machine-specific details point to `LOCAL_DEV.md`).

## Capabilities

### New Capabilities

- `testing-harness`: The developer- and Agent-facing test system for Qianyi_DP - scene construction, simulation driving, correctness / invariant / API-consistency / performance tests, machine-readable results, and optional visualization artifacts.

### Modified Capabilities

- None (this repository has no formal specs yet).

## Impact

- Added: `tests/` (pytest suite and framework package), `tests/notebooks/` (interactive Warp scenes), `AGENTS.md` (repo root), `.gitignore` entries for artifact directories, `openspec/` (this proposal).
- Unchanged: `src/` CUDA/C++ code, the public `Qianyi_DP` Python API, the Blender frontend.
- Compatibility: non-breaking; all tests are additive and do not change the existing `qydp.test` timing interface usage.
- Dependencies: test-tooling requirements are recorded in the project's Python packaging files (not in OpenSpec artifacts); optional soft dependency `shapely` (pattern_helper reference, deferred).
- CI: none yet; the framework is designed to be CI-ready (pytest / JUnit / marker layering) but runs locally for now.
