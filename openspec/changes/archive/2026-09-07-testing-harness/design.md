## Context

Current repository state: only the `qydp.test` timing benchmarks (sort / BVH / SDF / color_graph) exist, with no correctness assertions; simulation verification is manual in Blender; legacy local scene data (pickles from the maintainer's test notebooks - not part of this repository) has drifted out of sync with the current interface (missing fields); the development Python environment already provides the scientific stack needed by the tests (tooling requirements and environment details are recorded in the project's Python packaging files and in `LOCAL_DEV.md`, not in OpenSpec artifacts); the module under test is the CMake-built `Qianyi_DP` extension, which may exist in several ABI-tagged output directories of different ages (machine-specific paths and environment details live in `LOCAL_DEV.md` at the repo root and MUST NOT be repeated here).

Verified facts (smoke-tested):
- Blender-equivalent driver: **PDNewton defaults / the canonical parameter block + `update(0.001)`** runs 100 frames fully finite with real cloth motion; the legacy local scene data's arbitrary scaling factors or large dt produce PCG NaN.
- `update(h)` internally subdivides by `step_h` (default 0.001); the canonical block sets `step_h=0.003`.
- The frontend is a single runtime: the study notebook (`study02_clean.ipynb` in
  the frontend repo, run inside Blender via a notebook plugin) applies the canonical
  PDNewton block via `set_solver`/`set_parameter`, then the addon's
  `simulation_manager.py` drives `input_data`/`update` on the same `qydp.simulator`
  singleton; the addon runtime itself does not call `set_parameter` but inherits
  the notebook-set parameters. Parameter configuration is planned to move into the UI.
- `sample_points` is grid-based sampling (grid spacing = radius/sqrt(2), acceptance/refinement rules in `src/geometry/sample_points.cu`) - it is NOT Poisson-disc sampling; the test's spacing oracle MUST be derived from that source, not assumed.
- Scene contract: every triangle edge must appear in `edges`; `granularity` / `thickness` are in mm (converted internally, x0.001); cloth objects must precede obstacles.
- Known API issue (fixed during implementation): `get_all_solver()` hardcoded
  `('Explicit','PCG','Chebyshev','PNCG')`, inconsistent with the names actually
  accepted by `set_solver()` (`PDNewton` / `XPBD` / `VBD` / `Explicit`); the list
  is now synced (see Confirmed Findings).
- C-level debug prints are noisy (full Jx matrix dumps, `[Qianyi Error]` per edge, etc.); pybind11's ostream_redirect does not capture C printf.

## Goals / Non-Goals

**Goals:**
- A pytest suite with a single-command standard group, emitting machine-readable results (per-frame data traces, results.json, JUnit).
- Framework-provided procedural scene construction and a "Blender-equivalent semantics" driver; standard configuration = PDNewton + the commonly used parameter block.
- Coverage: simulation smoke / invariants / determinism, `sample_points` / `delaunay_2d` correctness, API consistency (known failure).
- Optional GIF artifacts (off by default) and an interactive Warp notebook; Agent-facing `AGENTS.md`.
- A performance-baseline mechanism for upcoming optimizations (e.g. `sample_points`).

**Non-Goals:**
- No changes to `src/` CUDA code (no test-only accessors; separate proposal if needed).
- Not testing SDF (not yet integrated into the solver), not testing pattern_helper (this phase), not testing VBD / XPBD / Explicit correctness (xfail smoke only).
- No CI integration yet (local first, but output is CI-ready).
- The `get_all_solver()` inconsistency was not an implementation task of this
  change (the test layer locks in the check); the enumeration fix landed
  separately during implementation (commit d8fe410).

## Decisions

### D1. Framework location: `tests/` + `tests/harness/` inside the qmyidp repo

Co-evolves with the code under test in the same repo/PR, avoiding the old "local test scripts drift out of sync" lesson. Alternatives: separate repo (sync overhead), separate local notes checkout (separated from CUDA code, rejected).

### D2. Test entry: pytest with marker layering

`pytest` (tooling requirement recorded in the project's Python packaging files) plus markers: `sim` (simulation), `algo` (geometry), `api`, `bench` (performance), `visual` (GIF rendering), `slow`, `quick` (standard group, `-m "quick"`, = sim smoke + algo + api). JUnit is natively supported via `--junitxml`. Alternative: custom runner (would rebuild ecosystem and reporting, rejected).

### D3. Module-under-test location and self-check (`qydp` fixture)

Resolution order: `QYDP_PYD` env var -> scan the repo's build output directories for the current interpreter's ABI tag (e.g. `Debug\`, pip/CMake `lib.win-amd64-cpython-<abi>\` output dirs; the exact local location is in `LOCAL_DEV.md`) -> plain import. Fingerprint check: `test` submodule present, `__version__` readable; mismatch -> error with a rebuild hint pointing to the documented CMake build (see `AGENTS.md`; machine paths in `LOCAL_DEV.md`); not found -> **skip GPU tests with build instructions** (spec: "explicit skip when the module is unavailable").

### D4. Scene builder: MeshSpec (with defaults) + derived construction

Defaults follow the Blender frontend: `mass=100`, `granularity=20` (mm), `thickness=0.1` (mm), `friction=0.03`, `stretch=(1,1,1)`, `bending=(1,1,1)`, `collision_layer=0`, identity `world_matrix`; derived from mesh data: `vertices` / `vertices_sim` (local), `edges` (with automatic triangle-edge integrity validation and completion), `triangles`, `fixed_vertices` / `attached_vertices` (weight semantics; procedural scenes use 0/1 weight equivalents). The builder emits a dict strictly identical to `simulator.input_data`'s contract. Alternative: replicate pkl scenes directly (proven out of sync, rejected).

### D5. Driver semantics (source of truth = simulation_manager.py)

`input_data({mesh_list, sewings})` -> `set_solver('PDNewton')` -> apply the standard parameter block (27 entries, canonical locally validated values - see `LOCAL_DEV.md`) -> per frame `update(0.001)` x 42 (ceil substeps at 24fps, matching the frontend's `max_update_time_step=0.001`) -> `get_simulation_data()` (local) plus `world_space=True` for traces/rendering. Frames, fps, and dt are configurable; default 60 frames @ 24fps.

### D6. Default artifacts: per-frame data traces (not GIF)

`artifacts/<test_id>/` (gitignored): `frames.npz` (per-frame local + world vertices, timestamps), `traces.json` (per-frame stats: max |disp|, non-finite count, pinned drift, z-range), `results.json` (case/params/status), `junit.xml`. **Default verification = scripts inspect traces directly, Agent does the debugging**; GIF is not a default artifact (D8). Alternative: store only the final frame (loses debugging information, rejected).

### D7. Warp's role: interactive notebook primary, reference implementation as spike

- Interactive: `tests/notebooks/warp_scenes.ipynb` runnable cell by cell (procedural scene -> driver -> frame data -> Warp scene/inspection), satisfying "cell-by-cell interactive viewing"; warp 1.14 is already installed.
- Reference: Warp component-level oracles (e.g. sampling/SDF-equivalent kernel cross-validation) are a follow-up spike, **not blocking this phase** - this phase uses numpy/scipy references (scipy.spatial.Delaunay cross-check, shoelace area, KDTree checks against the source-derived sampling spacing rule).
- Risk: warp currently discovers only a CPU device in this environment; GPU discovery is an open question (see Open Questions); the notebook needs a degradation notice.

### D8. GIF as opt-in artifact (off by default)

`--gif` (`-m visual`) explicitly enables: per-frame `matplotlib` tripcolor/plot_trisurf -> Pillow GIF assembly -> written to **`build/artifacts/` (gitignored temp directory)**; a `gallery.md` summary index is generated too. Alternatives: always generate (large files, conflicts with "scripts check data by default", rejected); Warp renderer (prettier, but GPU discovery risk and heavier dependency - listed as a later enhancement).

### D9. Experimental solvers and known issues governance

- Solver registry: PDNewton = `standard`; VBD/XPBD/Explicit = `experimental` (smoke tests `xfail(strict=True, reason=...)`; strict ensures that when they become usable the test **fails and prompts manual confirmation**, instead of passing silently).
- `get_all_solver()` inconsistency: `api/test_api_consistency.py` locks the
  check; the enumeration was synced during implementation (commit d8fe410), so
  the test is now a plain assertion.

### D10. Output noise control

conftest performs fd-level stdout redirection (os.dup2 -> per-case log file) around simulation calls, discarding C printf noise by default while keeping key lines (errors / `PCG`); `--verbose` passes through. Alternative: pybind11 ostream_redirect (only covers Python level, proven insufficient, rejected).

### D11. State isolation

Simulator is a process-wide singleton; each simulation case re-initializes via its own `input_data`. Determinism cases run the same scene twice and compare point-by-point; if same-process state residue causes instability, the failing case re-runs in a subprocess as a diagnostic (recorded as a risk now; subprocess-by-default not introduced in this phase).

### D12. Agent interface

- `AGENTS.md` (repo root): build/locate pyd commands (with a pointer to the gitignored `LOCAL_DEV.md` for machine-specific paths), `pytest -m quick` quick group, artifact locations, failure-debug flow (read traces.json -> locate frame -> inspect npz), known xfail list.
- Stable artifact naming + machine-readable `results.json`; each run ends with "artifact paths + pass/fail/xfail summary".

## Risks / Trade-offs

- [pyd out of sync with source] -> fingerprint check in conftest + build instructions in AGENTS.md; fingerprint mismatch raises an error instead of silently passing.
- [Simulator singleton state residue -> nondeterminism] -> per-case `input_data` re-initialization; determinism cases run twice and compare; subprocess re-verification as fallback if still unstable.
- [C-level print noise drowning useful info] -> fd redirection by default, keeping error lines; `--verbose` restores output.
- [warp finds no GPU] -> notebook degrades gracefully (CPU/assertions instead); GIF path uses matplotlib and is unaffected; a spike decides whether to enable Warp rendering.
- [Large scenes -> trace/render volume] -> small default scenes (~1k-vertex level) with frame caps; npz compression; GIF off by default.
- [PCG NaN may reappear (parameter drift)] -> the smoke case asserts finiteness as a hard check; the parameter block is frozen in the registry, so any parameter change is caught by the smoke test.

## Migration Plan

Additive only, no migration: landing order per tasks.md; rollback = remove `tests/`, `tests/notebooks/`, `AGENTS.md`, and the `.gitignore` additions - no impact on existing code or notebook assets.

## Open Questions

- Warp discovers only a CPU device in this environment (driver 13.0 but no GPU listed) - whether this relates to CUDA_PATH/runtime libs is pending a spike; does not block this phase (matplotlib rendering + scipy references).
- Performance baseline threshold policy (hard thresholds vs trend-only recording) will be decided with data during the bench phase.

## Confirmed Findings During Implementation (2026-08-24)

- **set_solver ordering**: `Simulator::init` (called by `input_data`) creates
  the solver object from the current solver name, so `set_solver` after
  `input_data` has no effect on the active solver. The driver now calls
  `set_solver`/`set_parameters` before `input_data` (matching the frontend
  notebook usage). The earlier "VBD runs PCG nan" probe was actually PDNewton
  running with the VBD parameter block (large `step_h`), not VBD itself.
- **PDNewton determinism (known issue)**: identical PDNewton runs differ by up
  to ~0.03-0.18 m (already at frame 0), reproduced in fresh subprocesses;
  `on_exit()` does not restore equality. The determinism test is therefore
  implemented as specified but marked `xfail(strict=True)` with the fix
  direction (C++ side: full buffer reset in `Geometry::init` / rule out
  kernel-level nondeterminism). It will enforce once the C++ side is fixed.
- **Experimental solvers deferred**: VBD/XPBD/Explicit smoke tests are skipped
  per maintainer decision (only PDNewton is tested this phase); the strict-xfail
  governance machinery remains in place for re-enabling later.
- **PDNewton init out-of-bounds (fixed)**: `Jx_nondiag_identity` was sized by
  vertex count but written per edge during the spring precompute, causing
  size-dependent crashes (e.g. a 20x20 grid) or silent memory corruption; the
  sizing was corrected in a separate commit (d8fe410).
