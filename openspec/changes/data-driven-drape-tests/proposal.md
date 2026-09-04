## Why

The pytest harness validates procedural toy scenes, but the real failure modes of
this CUDA solver (buffer sizing, scale/unit conventions, seam handling, large-mesh
stability) only surface on real garment meshes. GarmentCodeData provides thousands of
sewing-pattern garments with a reference drape, letting one developer replace
manual per-drape eyeballing with a repeatable, data-driven regression suite and an
interactive viewer for inspecting results.

## What Changes

- Add a GarmentCodeData loader that maps one dataset element (box mesh, panel/stitch
  segmentation, vertex labels, specification JSON) to the existing `input_data`
  contract: per-panel cloth meshes, vertex-pair sewings, attachment constraints, and
  the neutral-body obstacle mesh.
- Add a stratified sampler that selects elements by garment type and mesh-size bucket
  (fixed seed, reproducible), runs each element in an isolated subprocess, and emits
  the existing per-case artifacts (`results.json`, traces) plus a failure-class
  taxonomy (loader error / simulation blow-up / reference mismatch / timeout).
- Add a two-tier assertion set:
  - invariant tier (runs with or without the body): finite frames, seam closure,
    area preservation, pinned/attached drift, no explosion; when no body mesh is
    configured this tier runs with gravity disabled;
  - reference tier (neutral body configured): loose geometric comparison against the
    dataset's `sim.ply` (per-panel area, z-extent, horizontal cross-section
    quantiles, mean surface distance).
- Add a performance-regression tier (bench marker) recording per-substep wall time
  per size bucket; no hard threshold in this phase.
- Replace the current `warp_scenes.ipynb` with a Warp-based interactive viewer (a
  mini frontend) that loads an element's box mesh, simulated frames, reference
  drape, and body, and lets the user scrub frames, toggle overlays, and inspect
  failures; rendering pattern follows the `t3bvh_test.ipynb`.
- Extend `requirements-dev.txt` and `AGENTS.md` with the new test entry points and
  dataset configuration (dataset location is machine-specific and stays in
  `LOCAL_DEV.md`; tests read it from an environment variable).

## Capabilities

### New Capabilities

- `data-driven-drape-tests`: loading GarmentCodeData elements into the simulation
  harness, stratified sampling and execution of drape tests, two-tier drape
  assertions (invariants + reference comparison), performance regression records,
  and an interactive Warp viewer for inspecting results.

### Modified Capabilities

- None (the repository has no archived main specs yet; this change introduces the
  first new capability spec alongside the pending `testing-harness` capability).

## Impact

- Added: `tests/harness/gcd/` loader and sampling modules, `tests/data/` pytest
  cases (new markers `data` / `drape` / `slow`), a replacement
  `tests/notebooks/warp_drape_viewer.ipynb`, `requirements-dev.txt` entries
  (`trimesh` for PLY parsing and reference metrics), `AGENTS.md` updates.
- Unchanged: `src/` CUDA/C++ code and the public `Qianyi_DP` Python API.
- Compatibility: additive and non-breaking; existing markers and harness APIs stay
  intact; the old `warp_scenes.ipynb` is replaced (documented removal).
- Dependencies: `trimesh` (soft, used only by the data-driven path); dataset
  location is configured via `QYDP_GCD_ROOT` (and `QYDP_BODY_OBJ` for the neutral
  body), recorded in `LOCAL_DEV.md`.
- Not in scope: reproducing the dataset's exact drapes (different solver and
  materials), tuning fabric parameters to match the dataset materials, fixing
  solver determinism, or integrating SDF.
