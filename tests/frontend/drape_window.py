"""Interactive drape debug window.

A single-window live view of a drape simulation with mouse cloth dragging,
blow-up auto-pause, and screenshot capture — the Blender frontend's free
simulation mode without Blender (see OpenSpec change ``drape-debug-window``).

Launch::

    python tests/frontend/drape_window.py                       # CONFIG scene
    python tests/frontend/drape_window.py --scene cloth-grid
    python tests/frontend/drape_window.py --list-scenes
    python tests/frontend/drape_window.py --scene gcd:<id> --frames 120 \
        --screenshot out.png --drag 640,450,120,-80

All settings live in the CONFIG block below (design D3/D7): edit and
relaunch. Controls (owned keys are consumed before Warp's built-ins):

    Right-drag   grab/pull the cloth (engine pick_triangle constraint)
    Left-drag    orbit camera (Warp built-in)   Scroll: zoom   WASD: pan
    Space        pause / resume                 N: one frame while paused
    R            reset to the initial state     K: screenshot
    Esc          close
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (REPO_ROOT, REPO_ROOT / "tests"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from conftest import _resolve_qydp  # noqa: E402
from frontend import picking, scenes  # noqa: E402
from harness.presets import apply_preset  # noqa: E402

# ---------------------------------------------------------------------------
# CONFIG - every tunable of the debug window (code-only configuration).
# ---------------------------------------------------------------------------
CONFIG = {
    # Scene selector; see `--list-scenes`.
    "scene": "cloth-grid",
    # Solver + parameter block from tests/harness/presets.py; overrides are
    # applied on top of the preset before input_data.
    "solver": "PDNewton",
    "param_overrides": {},
    "dt": 0.003,
    # Blow-up detection (design D6): auto-pause when any vertex is
    # non-finite or moves further than this in one rendered frame.
    "blowup_displacement_m": 0.5,
    # Display.
    "window_width": 1280,
    "window_height": 900,
    "camera_fov": 45.0,
    # Orange: the Warp sky is (0.53, 0.8, 0.92) blue, so a blue cloth
    # vanishes against the background.
    "cloth_color": (0.95, 0.60, 0.15),
    # Multi-panel scenes cycle this palette per panel so seam boundaries are
    # unmistakable even where the display gap is thin.
    "panel_palette": [
        (0.95, 0.60, 0.15), (0.95, 0.85, 0.30), (0.55, 0.80, 0.30),
        (0.30, 0.80, 0.75), (0.40, 0.65, 0.95), (0.70, 0.45, 0.90),
        (0.90, 0.45, 0.60), (0.75, 0.55, 0.40),
    ],
    "failed_color": (0.90, 0.15, 0.15),
    "seam_color": (0.10, 0.05, 0.02),
    "seam_radius": 0.0015,
    # Steel stays distinct from the orange cloth and the pale grid.
    "obstacle_color": (0.55, 0.60, 0.65),
    "report_every_frames": 120,
}

SCREENSHOT_DIR = REPO_ROOT / "tests" / "artifacts" / "frontend"


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

class SimState:
    def __init__(self) -> None:
        self.frame = 0
        self.prev_local: np.ndarray | None = None
        self.paused = False
        self.failed = False
        self.failure_text: str | None = None


def init_sim(simulator, scene: scenes.SceneData, sim_params_overlay=None) -> SimState:
    """apply_preset -> overrides -> input_data (order matters, see driver)."""
    apply_preset(simulator, CONFIG["solver"])
    for key, value in CONFIG["param_overrides"].items():
        simulator.set_parameter(key, float(value))

    if sim_params_overlay:
        simulator.set_parameters(sim_params_overlay)
    simulator.input_data(scene.input_data)
    return SimState()


def advance_frame(simulator, state: SimState) -> None:
    simulator.update(CONFIG["dt"])
    state.frame += 1


def read_local(simulator) -> np.ndarray:
    return np.asarray(simulator.get_simulation_data(), dtype=np.float32)


def check_blowup(state: SimState, local: np.ndarray) -> str | None:
    finite = np.isfinite(local)
    if not finite.all():
        bad = int((~finite).any(axis=1).sum())
        return f"non-finite vertices: {bad}/{len(local)}"
    if state.prev_local is not None and len(state.prev_local) == len(local):
        max_disp = float(np.linalg.norm(local - state.prev_local, axis=1).max())
        if max_disp > CONFIG["blowup_displacement_m"]:
            return (
                f"max displacement {max_disp:.3f} m/frame "
                f"> {CONFIG['blowup_displacement_m']}"
            )
    return None


def on_blowup(state: SimState, reason: str) -> None:
    state.failed = True
    state.paused = True
    state.failure_text = reason
    print(f"[BLOW-UP] frame {state.frame}: {reason} - simulation paused")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
# Warp's up_axis="Z" path renders the vertical axis flipped (verified with a
# beacon probe: world +Z appears BELOW world -Z on screen), so the window
# uses the renderer's default Y-up frame and rotates axes at the boundary.
# The map (x, y, z) -> (x, z, -y) is a proper rotation (det=+1): it keeps
# triangle winding and lighting correct, unlike an axis swap (a mirror).
# All engine/simulation/picking math stays in the Z-up world.
_TO_RENDER = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
_FROM_RENDER = _TO_RENDER.T


def to_render(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ _TO_RENDER.T


def from_render(points) -> np.ndarray:
    return np.asarray(points, dtype=np.float32) @ _FROM_RENDER.T


def make_renderer(scene: scenes.SceneData):
    import warp as wp  # noqa: PLC0415 - lazy so --list-scenes works without a GPU

    wp.init()
    import warp.render  # noqa: PLC0415

    return warp.render.OpenGLRenderer(
        title=f"Drape Debug - {scene.name}",
        screen_width=CONFIG["window_width"],
        screen_height=CONFIG["window_height"],
        camera_pos=tuple(to_render(scene.camera_pos)),
        camera_front=tuple(to_render(scene.camera_front)),
        camera_fov=CONFIG["camera_fov"],
        # Warp's default near plane (1.0 m) clips the cloth as soon as the
        # camera zooms in on a panel.
        near_plane=0.01,
        far_plane=50.0,
        # Cloth must be visible from both sides (Warp defaults to culling).
        enable_backface_culling=False,
        vsync=True,
    )


def vertex_normals(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Smooth area-weighted per-vertex normals."""
    v0, v1, v2 = (points[triangles[:, i]] for i in range(3))
    face = np.cross(v1 - v0, v2 - v0)
    normals = np.zeros_like(points)
    for col in range(3):
        np.add.at(normals, triangles[:, col], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0.0] = 1.0
    return normals / lengths


def write_panel_vertices(renderer, name: str, points: np.ndarray, triangles: np.ndarray) -> None:
    """Rewrite a registered panel's interleaved VBO (pos/normal/uv).

    Warp's fast path only updates positions (normals stay flat: the cloth
    renders as a solid color), and its update_topology path re-registers
    the shape every frame - deregister_shape pops the shape LIST, shifting
    the ids of every later-registered mesh (the obstacle vanishes after a
    few frames). Registering once and writing the mapped CUDA-GL buffer
    avoids both problems.
    """
    import warp as wp  # noqa: PLC0415

    shape = renderer._instances[name][2]
    cuda_buffer = renderer._shape_gl_buffers[shape][4]
    gfx = np.zeros((len(points), 8), dtype=np.float32)
    gfx[:, 0:3] = points
    gfx[:, 3:6] = vertex_normals(points, triangles)
    gfx[:, 6] = 0.5
    gfx[:, 7] = 0.5
    vbo = cuda_buffer.map(dtype=wp.float32, shape=gfx.shape)
    wp.copy(vbo, wp.array(gfx, dtype=wp.float32, device=renderer._device))
    cuda_buffer.unmap()


def seam_line_geometry(scene: scenes.SceneData, local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build thin lines between the two endpoints of every stitch pair."""
    line_points: list[np.ndarray] = []
    for sewing in scene.input_data["sewings"]:
        panel_a, panel_b = (int(index) for index in sewing["patterns"])
        stitches = np.asarray(sewing["stitches"], dtype=np.int64).reshape(-1, 2)
        first = scene.panels[panel_a]
        second = scene.panels[panel_b]
        points_a = local[first.vertex_offset + stitches[:, 0]].astype(np.float32)
        points_b = local[second.vertex_offset + stitches[:, 1]].astype(np.float32)
        for point_a, point_b in zip(points_a, points_b):
            line_points.extend((point_a, point_b))
    if not line_points:
        return np.empty((0, 3), dtype=np.float32), np.empty(0, dtype=np.int32)
    points = np.ascontiguousarray(to_render(np.asarray(line_points, dtype=np.float32)))
    indices = np.arange(len(points), dtype=np.int32)
    return points, indices


def render_scene(renderer, scene: scenes.SceneData, local: np.ndarray, failed: bool) -> None:
    # Re-assert the obstacle every frame: if anything drops its instance it
    # is re-added on the next frame instead of vanishing for the whole run.
    render_obstacle(renderer, scene)
    palette = CONFIG["panel_palette"]
    for index, panel in enumerate(scene.panels):
        if failed:
            base = CONFIG["failed_color"]
        elif len(scene.panels) > 1:
            base = palette[index % len(palette)]
        else:
            base = CONFIG["cloth_color"]
        color = np.asarray(base, dtype=np.float32)
        name = f"panel_{index}"
        points = local[panel.vertex_offset:panel.vertex_offset + len(panel.vertices)]
        points = np.ascontiguousarray(to_render(points))
        if name in renderer._instances:
            renderer.update_shape_instance(name, color1=color)
            write_panel_vertices(renderer, name, points, panel.triangles)
        else:
            renderer.render_mesh(
                name=name,
                points=points,
                indices=panel.triangles.reshape(-1),
                colors=tuple(color),
            )

    seam_points, seam_indices = seam_line_geometry(scene, local)
    if len(seam_points):
        renderer.render_line_list(
            name="seams",
            vertices=seam_points,
            indices=seam_indices,
            color=CONFIG["failed_color"] if failed else CONFIG["seam_color"],
            radius=CONFIG["seam_radius"],
        )


def render_obstacle(renderer, scene: scenes.SceneData) -> None:
    if scene.obstacle is not None:
        renderer.render_mesh(
            name="obstacle",
            points=np.ascontiguousarray(to_render(scene.obstacle.vertices)),
            indices=scene.obstacle.triangles.reshape(-1),
            colors=CONFIG["obstacle_color"],
        )


def capture_png(renderer, path: Path) -> Path:
    """Save the framebuffer as PNG (Warp get_pixels + pyglet codec)."""
    import warp as wp  # noqa: PLC0415
    import pyglet.image as pyglet_image  # noqa: PLC0415

    width, height = renderer.window.get_framebuffer_size()
    buffer = wp.zeros((height, width, 3), dtype=wp.uint8)
    ok = renderer.get_pixels(buffer, split_up_tiles=False, mode="rgb", use_uint8=True)
    if not ok:
        raise RuntimeError("get_pixels failed; framebuffer not readable")
    pixels = np.ascontiguousarray(buffer.numpy()[::-1])  # GL bottom-up -> top-down
    image = pyglet_image.ImageData(width, height, "RGB", pixels.tobytes(), pitch=width * 3)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
    return path


def screenshot_path(scene_name: str, frame: int) -> Path:
    safe = scene_name.replace(":", "_")
    return SCREENSHOT_DIR / safe / f"frame_{frame:05d}.png"


# ---------------------------------------------------------------------------
# Interactive loop
# ---------------------------------------------------------------------------

def run_interactive(renderer, simulator, scene: scenes.SceneData,
                    sim_params_overlay: dict[str, float] = None) -> None:
    import pyglet  # noqa: PLC0415

    holder = {"state": init_sim(simulator, scene, sim_params_overlay)}
    controller = picking.PickController(simulator, scene.panels)
    single_step = {"pending": False}
    exit_requested = {"flag": False}

    def current_ray(x: float, y: float):
        # The renderer stores camera state in its Y-up frame; rotate back to
        # the Z-up world the picking math and the engine work in.
        return picking.screen_ray(
            camera_pos=from_render(renderer._camera_pos),
            camera_front=from_render(renderer._camera_front),
            camera_up=from_render(renderer._camera_up),
            fov_deg=renderer.camera_fov,
            width=renderer.window.width,
            height=renderer.window.height,
            px=x,
            py=picking.window_to_screen_y(renderer.window.height, y),
        )

    def on_key(symbol, modifiers):
        owned = True
        if symbol == pyglet.window.key.SPACE:
            holder["state"].paused = not holder["state"].paused
            paused = holder["state"].paused
            print(f"{'paused' if paused else 'resumed'} at frame {holder['state'].frame}")
        elif symbol == pyglet.window.key.N:
            single_step["pending"] = True
        elif symbol == pyglet.window.key.R:
            controller.release()
            holder["state"] = init_sim(simulator, scene, sim_params_overlay)
            print("reset to initial state")
        elif symbol == pyglet.window.key.K:
            state = holder["state"]
            try:
                print("screenshot:", capture_png(renderer, screenshot_path(scene.name, state.frame)))
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                print("screenshot failed:", exc)
        elif symbol == pyglet.window.key.ESCAPE:
            exit_requested["flag"] = True
        else:
            owned = False
        return pyglet.event.EVENT_HANDLED if owned else None

    def on_mouse_press(x, y, buttons, modifiers):
        if buttons & pyglet.window.mouse.RIGHT and not holder["state"].failed:
            origin, direction = current_ray(x, y)
            if controller.press(origin, direction):
                panel_index, triangle_index, _ = controller.last_pick
                print(f"pick: panel {panel_index} triangle {triangle_index}")
        return None

    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        if buttons & pyglet.window.mouse.RIGHT:
            origin, direction = current_ray(x, y)
            controller.drag(origin, direction, from_render(renderer._camera_front))
            return pyglet.event.EVENT_HANDLED
        return None

    def on_mouse_release(x, y, buttons, modifiers):
        if buttons & pyglet.window.mouse.RIGHT:
            controller.release()
            return pyglet.event.EVENT_HANDLED
        return None

    renderer.register_key_press_callback(on_key)
    renderer.window.push_handlers(
        on_mouse_press=on_mouse_press,
        on_mouse_drag=on_mouse_drag,
        on_mouse_release=on_mouse_release,
    )

    render_obstacle(renderer, scene)
    last_report = time.perf_counter()
    report_frames = 0

    try:
        while not renderer.window.has_exit and not exit_requested["flag"]:
            state = holder["state"]
            if not state.paused:
                start = time.perf_counter()
                advance_frame(simulator, state)
                sim_ms = (time.perf_counter() - start) * 1e3
                report_frames += 1
                if report_frames >= CONFIG["report_every_frames"]:
                    elapsed = time.perf_counter() - last_report
                    print(
                        f"frame {state.frame}: {sim_ms:.1f} ms/frame sim, "
                        f"{report_frames / max(elapsed, 1e-9):.1f} fps effective"
                    )
                    last_report = time.perf_counter()
                    report_frames = 0
            elif single_step["pending"]:
                single_step["pending"] = False
                advance_frame(simulator, state)

            local = read_local(simulator)
            if not state.failed:
                reason = check_blowup(state, local)
                if reason is not None:
                    on_blowup(state, reason)
            state.prev_local = local

            renderer.begin_frame(state.frame)
            render_scene(renderer, scene, local, state.failed)
            renderer.end_frame()
    finally:
        try:
            controller.release()
        except Exception:  # noqa: BLE001 - best effort during teardown
            pass


# ---------------------------------------------------------------------------
# Scripted (non-interactive) loop: for automated consumers and agents
# ---------------------------------------------------------------------------

def run_scripted(
        renderer,
        simulator,
        scene: scenes.SceneData,
        frames: int,
        drag: tuple[float, float, float, float] | None,
        screenshot: Path,
) -> None:
    state = init_sim(simulator, scene)
    controller = picking.PickController(simulator, scene.panels)
    width, height = renderer.window.width, renderer.window.height
    # The camera stays at the scene hint, so pixel coordinates are
    # reproducible across runs (design D8).
    camera = dict(
        camera_pos=scene.camera_pos,
        camera_front=scene.camera_front,
        camera_up=(0.0, 0.0, 1.0),
        fov_deg=CONFIG["camera_fov"],
    )

    def ray_at(px: float, py: float):
        return picking.screen_ray(width=width, height=height, px=px, py=py, **camera)

    drag_frames = (frames // 4, frames // 2) if drag else (0, -1)
    px, py, dx, dy = drag if drag else (0.0, 0.0, 0.0, 0.0)

    start = time.perf_counter()
    for frame in range(frames):
        advance_frame(simulator, state)
        if drag and drag_frames[0] <= frame <= drag_frames[1]:
            span = max(drag_frames[1] - drag_frames[0], 1)
            a = (frame - drag_frames[0]) / span
            if frame == drag_frames[0]:
                controller.press(*ray_at(px, py))
            controller.drag(*ray_at(px + dx * a, py + dy * a), scene.camera_front)
        elif drag and frame == drag_frames[1] + 1:
            controller.release()

        local = read_local(simulator)
        reason = check_blowup(state, local) if not state.failed else None
        if reason is not None:
            on_blowup(state, reason)
            break  # capture the failure frame instead of its aftermath
        state.prev_local = local

        renderer.begin_frame(state.frame)
        render_scene(renderer, scene, local, state.failed)
        renderer.end_frame()

    controller.release()
    elapsed = time.perf_counter() - start
    sim_seconds = state.frame * CONFIG["dt"]
    print(
        f"scripted run: {state.frame} frames in {elapsed:.1f}s "
        f"({state.frame / max(elapsed, 1e-9):.1f} fps effective), "
        f"sim time {sim_seconds:.2f}s"
    )
    if state.failed:
        print(f"BLOW-UP at frame {state.frame}: {state.failure_text}")
    print("screenshot:", capture_png(renderer, screenshot))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene", help="scene selector (default: CONFIG['scene'])")
    parser.add_argument("--list-scenes", action="store_true", help="list registered scenes and exit")
    parser.add_argument("--frames", type=int, default=None, help="scripted mode: run N frames and exit")
    parser.add_argument("--screenshot", type=Path, default=None, help="scripted mode: PNG output path")
    parser.add_argument(
        "--drag",
        default=None,
        help="scripted mode: 'px,py,dx,dy' - press at (px,py), drag to (px+dx,py+dy)",
    )
    return parser.parse_args(argv)


def parse_drag(text: str) -> tuple[float, float, float, float]:
    values = [float(v) for v in text.split(",")]
    if len(values) != 4:
        raise ValueError(f"--drag expects px,py,dx,dy; got {text!r}")
    return values[0], values[1], values[2], values[3]


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_scenes:
        print("Registered debug scenes:")
        for name, kind in scenes.list_scenes():
            print(f"  {name:<24} [{kind}]")
        return 0

    try:
        scene = scenes.load_scene(args.scene or CONFIG["scene"])
    except scenes.SceneError as exc:
        print(f"error: {exc}")
        return 2
    except scenes.SceneSetupError as exc:
        print(f"error: {exc}")
        return 3

    qydp = _resolve_qydp()
    if qydp is None:
        print(
            "error: no loadable Qianyi_DP module found; set QYDP_PYD or build "
            "the extension (see AGENTS.md)"
        )
        return 4

    print(
        f"Qianyi_DP {getattr(qydp, '__version__', 'dev')} | scene {scene.name} | "
        f"solver {CONFIG['solver']} | dt={CONFIG['dt']}"
    )
    renderer = make_renderer(scene)
    simulator = qydp.simulator
    try:
        if args.frames is not None:
            if args.frames <= 0:
                print("error: --frames must be positive")
                return 2
            drag = parse_drag(args.drag) if args.drag else None
            out = args.screenshot or screenshot_path(scene.name, args.frames)
            run_scripted(renderer, simulator, scene, args.frames, drag, out)
        else:
            run_interactive(renderer, simulator, scene)
    finally:
        try:
            renderer.close()
        except Exception as exc:  # noqa: BLE001 - Warp teardown can be brittle
            print(f"renderer close failed: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
