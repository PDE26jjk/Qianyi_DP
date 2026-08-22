## 1. Environment and skeleton

- [x] 1.1 Install pytest in the project's Python dev environment (the only new dependency; local env details in `LOCAL_DEV.md`) and confirm `python -m pytest --version` works - DONE by maintainer (Python 3.11, kept for Blender compatibility)
- [ ] 1.2 Create the `tests/` tree: `tests/harness/`, `tests/sim/`, `tests/algo/`, `tests/api/`, `tests/bench/`, `tests/notebooks/`, plus `tests/harness/__init__.py`
- [ ] 1.3 Register markers in the pytest config (root `pyproject.toml` or `pytest.ini`): `sim` / `algo` / `api` / `bench` / `visual` / `slow` / `quick`
- [ ] 1.4 Implement the `qydp` fixture in `tests/conftest.py`: resolution order `QYDP_PYD` env var -> scan the repo's build output directories for the interpreter ABI tag -> plain import; fingerprint check (`test` submodule present), mismatch raises with rebuild hint, not found skips GPU tests with build instructions
- [ ] 1.5 Append artifact directories to `.gitignore` (`build/artifacts/`, `tests/artifacts/`) and verify artifacts are never versioned
- [ ] 1.6 Dry-run validation: `pytest --collect-only` passes; skip reasons are correct when GPU/pyd is unavailable
- [ ] 1.7 Create `LOCAL_DEV.md` at the repo root (gitignored) recording machine-specific paths (module output location, Python environment, Blender frontend checkout, verified baseline facts) and add a portability note to `AGENTS.md`/README that OpenSpec artifacts and committed docs MUST NOT contain machine-specific paths
- [ ] 1.8 Create `requirements-dev.txt` at the repo root as the single place for test-tooling dependencies (initially `pytest`), with a header note that Python 3.11 is used for Blender compatibility; all future Python dependency changes are recorded there, never in OpenSpec artifacts

## 2. Harness package

- [ ] 2.1 `harness/meshspec.py`: `MeshSpec` data structure and procedural mesh generation (grid cloth: vertices/edges/triangles derivation, triangle-edge integrity completion), defaults from the Blender frontend (mass=100, granularity=20mm, thickness=0.1mm, friction=0.03, stretch/bending=(1,1,1), collision_layer=0, identity world matrix)
- [ ] 2.2 `harness/meshspec.py`: input validation and unit conversion (granularity/thickness mm->m; every triangle edge must be in edges; cloth objects precede obstacles)
- [ ] 2.3 `harness/presets.py`: PDNewton standard parameter block (28 entries - canonical locally validated values, provenance in `LOCAL_DEV.md`) plus solver registry (PDNewton=standard; VBD/XPBD/Explicit=experimental with reasons)
- [ ] 2.4 `harness/driver.py` `SimDriver`: `input_data` -> `set_solver` -> apply preset -> per frame `ceil(frame_time/0.001)` x `update(0.001)` -> collect local and world coordinates; configurable frames/fps/dt
- [ ] 2.5 `harness/traces.py`: per-frame trace writing (npy/npz: per-frame local + world vertices, timestamps) plus per-frame stats (traces.json: max |disp|, non-finite count, pinned drift, z-range)
- [ ] 2.6 `harness/output_redirect.py`: fd-level capture of C-level stdout/stderr (os.dup2 -> per-case log), discarding noise while keeping keyword lines (Error/PCG/ERROR) by default, pass-through with `--verbose`
- [ ] 2.7 `harness/report.py`: per-case `results.json` writer (case/params/status/artifact paths/trace summary), coexisting with pytest `--junitxml`

## 3. Simulation tests

- [ ] 3.1 `tests/sim/test_smoke.py`: standard scene (corner-pinned grid cloth) 60 frames @24fps smoke - all frames finite, pinned drift within tolerance, free-vertex total displacement above threshold, no vertex flying outside the scene bounding box
- [ ] 3.2 `tests/sim/test_determinism.py`: same scene and parameters run twice, point-by-point comparison across all frames; if unstable in-process, subprocess re-verification with a hint
- [ ] 3.3 `tests/sim/test_experimental_solvers.py`: VBD/XPBD/Explicit smoke `xfail(strict=True)` (explicit prompt for manual confirmation when they become usable)
- [ ] 3.4 Failure enhancement: on assertion failure, traces/frame index/stats land in `results.json` and a "view <artifact path>" hint is printed

## 4. Geometry correctness

- [ ] 4.1 `tests/algo/test_sampling.py`: sampled-point spacing property with the oracle derived from `src/geometry/sample_points.cu` (grid-based, grid spacing = radius/sqrt(2); NOT Poisson-disc), checked via scipy KDTree, including boundary point pairs
- [ ] 4.2 `tests/algo/test_sampling.py`: triangulation validity - output triangle area sum equals polygon area (shoelace reference), no out-of-bounds triangles; includes a polygon-with-holes case
- [ ] 4.3 `tests/algo/test_delaunay.py`: `delaunay_2d` on random point sets topologically consistent with `scipy.spatial.Delaunay`
- [ ] 4.4 `tests/bench/test_sampling_bench.py`: `sample_points` scale sweep (points x radius), baseline JSON of time/point counts (marked bench, not a default gate)

## 5. API consistency

- [ ] 5.1 `tests/api/test_api_consistency.py`: assert `get_all_solver()` matches the set accepted by `set_solver()`; currently `xfail(strict=True)` with the reason recording the difference and fix direction (sync the hardcoded list in simulator.cu)

## 6. Visualization and interaction

- [ ] 6.1 `harness/render.py`: per-frame matplotlib (tripcolor/plot_trisurf) -> Pillow GIF assembly; output to `build/artifacts/` (gitignored)
- [ ] 6.2 `tests/sim/test_visual.py` with `--gif` option (marker `visual`, skipped by default): generate GIF plus `gallery.md` summary index
- [ ] 6.3 `tests/notebooks/warp_scenes.ipynb`: runnable cell by cell (environment setup and pyd location -> procedural scene construction -> driver collection -> frame data checks -> Warp interactive scene with a graceful degradation notice when GPU is unavailable)

## 7. Agent interface and documentation

- [ ] 7.1 Root `AGENTS.md` (maintainer-owned - user has taken this over; this change only specifies the required content): pyd build/location guide (machine-specific paths point to `LOCAL_DEV.md`), `pytest -m quick` quick group, artifact locations, failure-debug flow (read traces.json -> locate frame -> inspect npz), known xfail list and semantics
- [ ] 7.2 `AGENTS.md` common command block (collect tests, run one class, generate GIF, export traces) plus local environment activation notes (maintainer-owned)
- [ ] 7.3 Verify `AGENTS.md` is not swallowed by the repo `.gitignore` - currently line 82 (`AGENTS.md`) still matches per `git check-ignore`; needs a `!AGENTS.md` negation or removing that entry before `AGENTS.md` can be committed
- [ ] 7.3 Full validation: `pytest -m quick` fully green (expected xfails included); `--junitxml` report and `results.json` generate correctly; skip info is correct without the pyd

## 8. Wrap-up

- [ ] 8.1 Run `openspec validate` to check change consistency
- [ ] 8.2 Clean up this phase's temporary files (smoke scripts live outside the repo; confirm nothing remains); report artifact paths and the known issue list to the user
- [ ] 8.3 Portability audit: scan all committed files (docs, tests, `openspec/`) for machine-specific paths and local environment details - they MUST only exist in `LOCAL_DEV.md` (gitignored)
