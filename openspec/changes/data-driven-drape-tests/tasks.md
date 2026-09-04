## 1. Environment and configuration

- [x] 1.1 Add `trimesh` to `requirements-dev.txt` and register new pytest markers (`data`, `drape`) in `pyproject.toml`
- [x] 1.2 Add a `qydp_data` pytest fixture that resolves `QYDP_GCD_ROOT` / `QYDP_GCD_BODY`, skips data tests with setup instructions when missing, and records the local dataset paths in `LOCAL_DEV.md` only

## 2. GarmentCodeData loader

- [x] 2.1 Implement box-mesh reading: trimesh load (`process=False`), UV-duplicate mapping (`v_id_map`), centimetre-to-metre conversion
- [x] 2.2 Implement panel splitting: segmentation labels to per-panel vertex/triangle subsets, derive per-panel edges, emit one mesh dict per panel
- [x] 2.3 Implement stitch pairing: split each `stitch_N` seam chain into two boundary chains via edge adjacency and pair vertices in order; cross-check against the specification JSON's panel-edge stitches; classify unpaired seams as loader errors
- [x] 2.4 Map semantic vertex labels to `attached_vertices` (exposed but optional, see design D6)
- [x] 2.5 Implement the body obstacle loader: neutral-body OBJ as `object_type=1` with per-face normals, unit/scale conversion, bounding-box alignment check against the garment
- [x] 2.6 Loader dry-run validation: load one element end-to-end into `input_data`, run 1-2 frames, and emit a loader report (panel count, stitch count, scale/bbox stats, mismatch list)

## 3. Sampling and batch execution

- [x] 3.1 Implement stratified sampling: garment type from `design_params.yaml`, size buckets from face counts (S < 5k, M 5-25k, L 25-50k, XL > 50k), fixed seed, reproducible manifest with selected element ids
- [x] 3.2 Implement the isolated batch runner: fresh subprocess per element, per-case timeout, fd-level output capture, standard per-case artifacts, failure taxonomy (loader error / simulation blow-up / reference mismatch / timeout)
- [x] 3.3 Calibration pass: verify body/garment alignment and neutral-body applicability on a small hand-checked subset; confirm default frame budget and VRAM/time fit for the default manifest (XL excluded by default)

## 4. Assertions and records

- [x] 4.1 Implement the invariant tier: finite frames, bounding envelope, seam-closure tolerance, area preservation, attached-vertex drift; gravity disabled when no body is configured
- [x] 4.2 Implement reference metrics: per-panel area ratio, z-extent, horizontal cross-section quantiles, mean surface distance; record-only default until tolerances are calibrated, with a configuration path to hard-fail
- [x] 4.3 Implement per-substep performance records grouped by size bucket (benchmark marker, no hard threshold)

## 5. Data-driven tests

- [x] 5.1 Add loader unit tests with a synthetic fixture element (duplicate mapping, units, panel split, stitch pairing, error classification)
- [x] 5.2 Add `tests/data/` drape test cases (markers `data`/`drape`/`slow`) that run a small sample batch and assert per-case `results.json` contents
- [x] 5.3 Update `AGENTS.md` with the data-test entry points and artifact locations; run a portability scan over committed files

## 6. Warp interactive viewer

- [x] 6.1 Create `tests/notebooks/warp_drape_viewer.ipynb`: `OpenGLRenderer` frame loop rendering box mesh, simulated frames, reference drape, and body, with frame scrubbing and overlay toggles (pattern per `t3bvh_test.ipynb`); assumes a CUDA-capable Warp runtime, no non-GPU fallback
- [x] 6.2 Integrate batch results: list failing cases from `results.json` and open a selected case in the viewer
- [x] 6.3 Replace the old `warp_scenes.ipynb` with the new viewer (delete the old notebook)

## 7. Validation and wrap-up

- [x] 7.1 Run `openspec validate`; confirm `pytest -m quick` is unaffected and a small data batch runs end-to-end
- [x] 7.2 Portability audit and wrap-up summary (artifact paths, failure taxonomy, calibration results)
