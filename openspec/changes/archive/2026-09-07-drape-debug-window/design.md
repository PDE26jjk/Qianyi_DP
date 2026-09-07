## Context

The engine already exposes the full interactive surface the window needs
(`update`, `pick_triangle`/`pick_triangle_update`/`pick_triangle_remove`,
`get_simulation_data`, `set_solver`, `set_parameters`) — the Blender frontend
drives exactly these bindings. The test tree already has module resolution
(`tests/conftest.py::_resolve_qydp`), solver presets
(`tests/harness/presets.py`), procedural scene building
(`tests/harness/meshspec.py`), and the GarmentCodeData loader
(`tests/harness/gcd/loader.py`). Constraints discovered during exploration:

- Warp 1.14 `render_mesh(colors=...)` is a per-instance single RGB color, not
  per-vertex colors; re-calling `render_mesh` with the same name updates
  vertices on a fixed-topology fast path.
- Warp's renderer owns left-drag orbit / scroll zoom / WASD pan; the right
  mouse button is free. Mouse handlers can be stacked via pyglet's
  `push_handlers` without replacing Warp's camera callbacks.
- Engine positions consumed by the harness are Z-up and use identity world
  matrices; `tests/harness/gcd/assertions.py` documents that world-space
  output leaves stale positions for proxy-merged seam vertices, so consumers
  must use **local-space** simulation data (`get_simulation_data()` default).
- Solver/parameters must be applied **before** `input_data` (Simulator::init
  creates the solver from the name at `input_data` time — see
  `tests/harness/driver.py`).

## Goals / Non-Goals

**Goals:**

- A single-command debug window that testers run, watch, and drag — close in
  feel to the Newton example demos (`while running: step; render`).
- A scene registry that makes "add another debug scene" a one-function change.
- Behavior comparable to the batch tests by default (same drive semantics and
  preset block), so a blow-up seen in CI can be reproduced interactively.

**Non-Goals:**

- No runtime settings UI, no parameter panels, no per-vertex debug coloring,
  no reference-mesh overlay, no GIF/recording (notebook and batch artifacts
  keep those jobs).
- No new engine (C++) API and no changes to existing harness behavior.
- No cross-platform or remote work: Windows + local CUDA, like the rest of
  the GPU-side test tooling.

## Decisions

### D1. Layout: `tests/frontend/` package with an entry script and a scene registry

```
tests/frontend/
  __init__.py
  drape_window.py   # CONFIG block at top + main loop + input handling
  picking.py        # renderer-free: ray math + PickController (design D4/D9)
  scenes.py         # SceneSpec registry (name -> loader)
  test_scenes.py    # CPU-only registry regression test (no GPU, no window)
  test_picking.py   # ray-math unit tests + GPU-marked headless pick-cycle test
```

Run as `python tests/frontend/drape_window.py [--scene NAME |
--list-scenes]`. The script bootstraps `sys.path` with the repo root and
`tests/` (same pattern as the notebook) to import `conftest._resolve_qydp`,
`harness.presets`, `harness.meshspec`, and `harness.gcd.loader`. A package
rather than one file because debug scenes are expected to multiply; the
registry file absorbs the growth while the window file stays stable.

*Alternative:* keep everything in one script. Rejected — scene additions would
churn the main loop file and the registry test would have no import seam.

### D2. Scene registry contract

`SceneSpec`: `name`, `kind` (`"procedural"` | `"gcd"`), `requires_gcd_env`
flag, `loader()` returning `SceneData`:

- `input_data` — passed to the engine verbatim (validated by
  `validate_mesh_list` inside loaders that build it);
- cloth render blocks — per mesh-list entry: flat points (N,3), flat triangle
  indices, mesh-list index (cloth panels first, obstacles after — the loader
  convention the engine's `pick_triangle(mesh_index, ...)` indexes into);
- optional obstacle render block (points/indices, per-face normals are not
  needed for display);
- camera hint (position/front) from the scene bounding box.

Initial registry: `cloth-grid` (MeshSpec grid, like `tests/sim/test_smoke.py`)
and `gcd:<element_id>` resolved through `QYDP_GCD_ROOT`/`QYDP_GCD_BODY` with
the same skip-style error text as the data tests. The GCD entry is a template
(scene name carries the element id) rather than one entry per element.

### D3. Single-threaded main loop, interactive pacing

```
while window open:
    handle pause/step/reset keys and pick mouse events
    if running: substeps x simulator.update(dt)          # default 1 x 0.01
    local = simulator.get_simulation_data()               # LOCAL space (D5)
    blow-up check on local                                 # isfinite + max |dx|
    render cloth panels + obstacle                        # render_mesh
```

One thread, no locks — the render loop *is* the clock, like the Newton demos.
Default pacing is interactive (2 x `update(0.003)` per frame; `0.01` steps
tunnel through the thin collision layer and drop the garment off the body):
harness pacing (42 x `update(0.001)`) is a CONFIG option for reproducing
batch behavior. Panels register ONCE and each frame the interleaved CUDA-GL
vertex buffer is rewritten directly (position + self-computed smooth normals
+ uv): Warp's position-only fast path leaves flat initial normals (solid-
color cloth), and its update_topology path re-registers shapes every frame -
deregister_shape pops the shape list and shifts later shapes' ids, which
made the obstacle mesh vanish after a few frames.

Reset (R key) re-runs the full init sequence `apply_preset` → `input_data`
(ordering constraint above), drops pick handles, and zeroes the frame counter
— mirroring the Blender frontend's `need_to_set_data` path.

### D4. Picking: right button + engine `pick_triangle`

All drag math lives in `tests/frontend/picking.py`, which imports only
numpy and receives the simulator as an argument — no renderer imports (D9).

- pyglet `window.push_handlers(on_mouse_press/on_mouse_drag/on_mouse_release)`
  stacked above Warp's handlers; the handlers act only on the right button and
  return not-handled otherwise, so Warp's camera behavior is untouched.
- Press: unproject the cursor through `inv(view @ proj)` (near/far world
  points → ray), vectorized Möller–Trumbore against all cloth panel triangles
  in world (== local, identity world matrices) space →
  `pick_triangle(mesh_index, tri_index, hit_point)`. Click-time only; numpy
  over ~50k triangles is milliseconds.
- Drag: intersect the cursor ray with the camera-facing plane through the
  grab point → `pick_triangle_update(handle, plane_hit)`.
- Release / window focus loss: `pick_triangle_remove(handle)`.

`PickController` encapsulates the lifecycle state (active handle, grab
point, mesh/triangle ids) behind `press(ray)` / `drag(ray)` / `release()`;
the window's mouse handlers and the headless integration test call the same
three methods.

*Alternative:* `add_picker` (vertex spring). Rejected — the Blender frontend's
grab semantics are `pick_triangle`; matching them keeps behavior comparable
across frontends.

### D5. Local-space data everywhere

Rendering, blow-up checks, and picking all use `get_simulation_data()` local
positions (world matrices are identity in every harness scene), per the
stale-world-positions caveat documented in `tests/harness/gcd/assertions.py`.

### D6. Blow-up detection is inline and cheap

Per rendered frame: `np.isfinite(local).all()` plus
`max |local - prev_local|` against `CONFIG.blowup_displacement_m`. On trip:
set `failed=True` (panels switch to red), auto-pause, print frame index,
non-finite count, and measured displacement. The check is two numpy
expressions over an existing array — deliberately not shared with the batch
invariant tier, whose helpers (seam closure, areas) assume batch data and
local-space frames of a full run.

### D7. Dependencies

Add `warp-lang` to `requirements-dev.txt` (pulls pyglet). The window itself
is never imported by pytest collection except `test_scenes.py`, which is
CPU-only and does not import the renderer modules.

### D8. Screenshot capture via `get_pixels` + scripted mode

Warp 1.14 `OpenGLRenderer.get_pixels(mode="rgb", use_uint8=True)` reads the
framebuffer into a CUDA `wp.array`; `.numpy()` plus a vertical flip yields an
HxWx3 buffer saved as PNG through pyglet's built-in image codecs (no Pillow
dependency). Naming trap: Warp's `OpenGLRenderer.save()` is *not* a
screenshot — it blocks keeping the window open.

- Hotkey `K` (avoids Warp's built-in V/C/G/I/X/T/B bindings): writes
  `tests/artifacts/frontend/<scene>/frame_NNNNN.png` (gitignored via the
  existing `tests/artifacts` rule) and prints the path.
- Scripted mode `--frames N --screenshot [--drag px,py,dx,dy]`: runs from
  the initial state with the camera fixed at the scene hint (no user camera
  input, so pixel-to-ray is reproducible), optionally performs one scripted
  drag through `PickController`, writes the screenshot(s), prints all
  written paths, and exits. This is the consumption path for multimodal
  agents and for automated verification of the window itself.

### D9. Pick testability without a window

`picking.py` keeps the geometry pure (cursor unprojection, Möller–Trumbore
hit, drag-plane intersection over numpy arrays) so `test_picking.py` covers
it with synthetic analytic cases on any machine. The headless integration
test (marked like the `sim` GPU tests) builds the `cloth-grid` scene, applies
the preset, steps to a settled state, presses at a ray aimed at a known
triangle, drags the target for N frames, and asserts: picked-triangle
vertices displace toward the target significantly more than an unpicked
baseline, and the displacement relaxes after `release()`. One deterministic
scene, no window, no screenshots — the visual path is exercised by the
scripted mode in D8 and the manual pass.

### D10. Seam-aware panel rendering (display only)

The box mesh's panels are split by UV islands and stay as separate render
meshes with their full topology. The window:

- Renders each panel's full triangles in a distinct color (cyclic palette),
  so seam boundaries are distinguishable by color alone; seam-bridging faces
  are kept and drawn.
- Builds the seam-chain geometry from the loader's `input_data["sewings"]`
  (panel-local stitch index pairs) and draws the chains as thin lines,
  recomputed every frame from the current panel vertices so they follow the
  moving cloth.
- Keeps everything display-only: `input_data`, the engine meshes, and picking
  use the full (unculled) topology.

## Risks / Trade-offs

- [Warp renderer `close()` can raise on abnormal shutdown] → wrap close in
  try/except and print; a debug tool may be killed from the console.
- [Repeated `input_data` resets could leak device memory] → same path the
  Blender frontend uses per start; if it ever misbehaves, R can fall back to
  "restart the process" (documented in the CONFIG comment).
- [Slow-motion on large garments hides timing bugs] → console prints sim
  ms/frame every N frames so testers can distinguish slow-motion from
  stall; CONFIG exposes substeps.
- [pyglet handler stacking vs Warp's own mouse assignment] → verified
  mechanism (Warp assigns defaults; pushed handlers run first and only
  consume right-button events); the apply phase smoke-tests both camera
  orbit and a pick cycle manually.
- [`get_pixels` requires a CUDA-array target and returns bottom-up rows] →
  allocate once, flip vertically on save; verified in the screenshot task.
- [Scripted drag depends on camera setup] → scripted mode pins the camera to
  the scene hint and accepts drag input as pixels relative to that fixed
  view, keeping pixel-to-ray reproducible across runs.
- [GCD element with missing `frames.npz` or body OBJ] → registry loader
  fails fast with the data-test setup text; the window never silently drops
  the obstacle.

## Migration Plan

Purely additive (`tests/frontend/`, one requirements line, one AGENTS.md
paragraph). Rollback = delete the directory and revert the two lines.

## Open Questions

- Default `blowup_displacement_m` value (proposed 0.5 m per rendered frame,
  i.e. 12 m/s) — calibrate against known-good and known-bad batch runs in
  the apply phase; it is a CONFIG constant either way.
