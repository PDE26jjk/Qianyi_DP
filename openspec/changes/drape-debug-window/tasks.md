## 1. Foundation

- [x] 1.1 Create `tests/frontend/__init__.py` and a minimal
  `drape_window.py` that bootstraps sys.path, resolves `Qianyi_DP` via
  `conftest._resolve_qydp`, and prints the module version; verify with
  `python tests/frontend/drape_window.py --help` in the dev environment
- [x] 1.2 Add `warp-lang` to `requirements-dev.txt` and verify
  `pip install -r requirements-dev.txt` succeeds; confirm
  `import warp; warp.init()` runs on the CUDA environment
- [x] 1.3 Document the debug window in `AGENTS.md` (launch command, scene
  registry, keyboard/mouse map, code-only configuration rule); verify the
  committed text contains no machine-specific paths

## 2. Scene registry

- [x] 2.1 Implement `tests/frontend/scenes.py`: `SceneSpec`/`SceneData`
  contract per design D2, the `cloth-grid` procedural loader (MeshSpec grid
  mirroring `tests/sim/test_smoke.py`), and the `gcd:<element_id>` template
  loader failing fast with the data-test setup message when
  `QYDP_GCD_ROOT`/`QYDP_GCD_BODY` are unset; verify `--list-scenes` prints
  both entries and a bad scene name errors with the available names
- [x] 2.2 Add `tests/frontend/test_scenes.py` (CPU-only, no GPU/window):
  procedural scene builds a `validate_mesh_list`-passing `input_data`;
  GCD template raises the documented setup error with env unset and is
  skipped-with-instructions style when the dataset is absent; verify with
  `python -m pytest tests/frontend/test_scenes.py -m quick`

## 3. Simulation window core

- [x] 3.1 Implement the CONFIG block (scene, solver, parameter overrides,
  fps/dt/substeps, blow-up threshold) and the init sequence
  `apply_preset` -> `input_data` in `drape_window.py`; verify startup
  prints the applied solver/parameter block and the scene panel summary
- [x] 3.2 Implement the single-threaded main loop (design D3): harness
  pacing `ceil((1/24)/0.001)` substeps of `update(dt)` per frame,
  local-space `get_simulation_data()` readback, per-panel registration
  plus direct CUDA-GL vertex-buffer writes (positions + smooth normals),
  single RGB colors, obstacle rendering when present; verify the window shows the cloth-grid scene falling under
  gravity and updates smoothly
- [x] 3.3 Implement keyboard controls Space (pause/resume), N (one frame
  while paused), R (reset via `apply_preset` -> `input_data`, drop picks,
  zero frame counter), Esc (close with guarded `renderer.close()`),
  consuming the owned keys before Warp's built-in bindings; verify each key
  behaves as specified in a manual run
- [x] 3.4 Implement seam-aware rendering (design D10): filter
  seam-band faces (any face with a stitch-only vertex, exposed by the
  loader as per-panel seam_face_mask per the dataset's segmentation
  labeling) out of the render meshes and draw no sewing-line overlays
  (maintainer decision); verify on a GarmentCodeData element that panel
  boundaries appear as gaps and hoods (vertical panels) still render

## 4. Cloth picking

- [x] 4.1 Implement `tests/frontend/picking.py` (renderer-free, numpy only):
  cursor-ray unprojection through `inv(view @ proj)`, vectorized
  Moller-Trumbore over cloth triangles, camera-facing drag-plane
  intersection, and a `PickController` owning the press/drag/release ->
  `pick_triangle*` lifecycle (simulator injected); verify the ray-math unit
  tests in `test_picking.py` pass on CPU against analytic synthetic cases
- [x] 4.2 Wire the right-button mouse lifecycle in the window via pyglet
  `push_handlers` onto `PickController` (design D4), printing the picked
  mesh/triangle indices on click; verify the cloth follows the cursor,
  release resumes free simulation, and left-drag orbit/scroll zoom/WASD pan
  remain Warp defaults while no pick is active
- [x] 4.3 Add the GPU-marked headless pick-cycle integration test in
  `test_picking.py` (`sim` marker, no window): press at a known triangle,
  drag the target, step frames, assert the picked vertices displace toward
  the target beyond an unpicked baseline and relax after release; verify it
  passes on the dev GPU

## 5. Blow-up detection

- [x] 5.1 Implement per-frame checks (design D6): non-finite local
  vertices and max per-frame displacement vs `CONFIG.blowup_displacement_m`
  -> auto-pause, panels render red, console prints frame index, non-finite
  count, displacement, and sim ms/frame; verify by temporarily setting the
  threshold near zero (trips on the first frame) and restoring it
- [x] 5.2 Calibrate the default threshold against one known-good batch
  case (no trip over a full run) and one known blow-up configuration (trips
  within seconds); record the chosen default in the CONFIG comment

## 6. Screenshot capture

- [x] 6.1 Implement framebuffer capture (design D8: `get_pixels`
  rgb/uint8 -> vertical flip -> pyglet PNG save) behind hotkey `K`, writing
  `tests/artifacts/frontend/<scene>/frame_NNNNN.png`; verify the file is
  written, viewable, and its path printed to the console
- [x] 6.2 Implement the scripted mode `--frames N --screenshot [--drag
  px,py,dx,dy]`: fixed camera at the scene hint, optional scripted drag
  through `PickController`, write screenshot(s), print all written paths,
  exit without user input; verify an unattended invocation from a clean
  shell produces the expected PNGs and exits by itself

## 7. End-to-end verification

- [ ] 7.1 Manual pass on `cloth-grid`: run, pause/step/reset, drag cloth,
  orbit camera, hotkey screenshot, confirm no console errors on clean exit
  and that screenshot artifacts stay out of `git status`; record outcome in
  the change notes
- [ ] 7.2 Manual pass on a GarmentCodeData element (dataset env set): scene
  loads with body obstacle, drag works across panels, performance reported
  per frame; confirm graceful failure text when env vars are unset
