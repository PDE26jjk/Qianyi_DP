# AGENTS.md

Guidance for developers and AI agents working on Qianyi_DP (a CUDA-accelerated
cloth simulation and geometry library with a Python API via pybind11).

## Building and locating the module under test

The module under test is the CMake-built `Qianyi_DP` extension (a `.pyd` on
Windows). The test suite resolves it automatically in this order:

1. `QYDP_PYD` environment variable (path to a `.pyd` file or a directory
   containing one);
2. a scan of the repo's build output directories for a `.pyd` matching the
   running interpreter's ABI tag;
3. a plain `import Qianyi_DP` (probed in a subprocess so a stale installed
   copy cannot crash the test session).

The module must expose the `test` submodule and a readable `__version__`; a
found-but-stale module raises with a rebuild hint, and a missing module skips
the GPU-dependent tests with build instructions. Machine-specific build
commands, output paths, and environment details are recorded in
`LOCAL_DEV.md` (gitignored) - never in committed files.

## Running the tests

Quick group (the standard single command):

```bash
python -m pytest -m quick
```

Other useful invocations:

```bash
# Collect tests without running them
python -m pytest --collect-only

# Run one class
python -m pytest tests/sim/test_smoke.py -k standard

# Run a single marker group (simulation / geometry / api / bench)
python -m pytest -m sim
python -m pytest -m algo
python -m pytest -m bench

# Generate GIF artifacts (opt-in; visual marker)
python -m pytest -m visual --gif

# Export a JUnit report (plus per-case results.json)
python -m pytest -m quick --junitxml=tests/artifacts/junit.xml
```

Markers: `sim`, `algo`, `api`, `bench`, `visual`, `slow`, `quick`.
`quick` = simulation smoke + geometry correctness + API consistency.

## Artifact locations

- Per-case data: `tests/artifacts/<case_id>/` (gitignored):
  - `frames.npz` - per-frame local and world vertices plus timestamps;
  - `traces.json` - per-frame stats (max displacement, non-finite count,
    pinned drift, z-range);
  - `results.json` - case/params/status/artifacts, plus failure details
    (frame index, per-frame stats) when a case fails;
  - `sim.log` - C-level output (noise discarded, `Error`/`PCG`/`ERROR` lines
    kept).
- GIF artifacts: `build/artifacts/visual/` (gitignored), with `gallery.md`.

## Debugging a failure

1. Read `tests/artifacts/<case_id>/results.json` for the status and failure
   summary.
2. Read `traces.json` to locate the failing frame index and its statistics.
3. Load `frames.npz` and inspect the frame data directly (e.g. check
   non-finite values, displacements, or pinned drift).
4. Check `sim.log` for the kept C-level lines around the failure.

## Known xfails and strict-xfail semantics

`xfail(strict=True)` means: while the underlying issue exists the test is a
known failure (green run), but if the issue is fixed the run reports an XPASS
failure that requires manual confirmation before the marker is removed. The
current known issues:

- `sim/determinism`: PDNewton runs are not bitwise deterministic run-to-run
  (reproduced in fresh subprocesses). Fix direction: C++ side - fully reset
  per-vertex device buffers in `Geometry::init` and rule out kernel-level
  nondeterminism.
- `sim/experimental/*`: VBD/XPBD/Explicit smoke tests are deferred (skipped)
  by maintainer decision; re-enable after the solvers are re-validated.

## Portability

Machine-specific paths, local build outputs, and local environment details
MUST NOT appear in committed docs, tests, or OpenSpec artifacts. They are
recorded in `LOCAL_DEV.md` at the repository root, which is gitignored and
never committed. Code comments and committed public files MUST be in English.
