"""Picking tests.

The ray-math tests are renderer-free and CPU-only (no GPU, no Warp). The
pick-cycle integration test drives the engine headlessly through the same
PickController the window uses - no window is created.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from frontend import scenes
from frontend.picking import (
    PickController,
    normalize,
    ray_plane,
    ray_triangle_hit,
    screen_ray,
    window_to_screen_y,
)


# ---------------------------------------------------------------------------
# Renderer-free unit tests (CPU only)
# ---------------------------------------------------------------------------

def test_screen_ray_center_points_forward():
    origin, direction = screen_ray(
        camera_pos=(0.0, 0.0, 5.0),
        camera_front=(0.0, 0.0, -1.0),
        camera_up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
        width=800,
        height=600,
        px=399.5,
        py=299.5,
    )
    assert np.allclose(origin, (0.0, 0.0, 5.0))
    assert np.allclose(direction, (0.0, 0.0, -1.0), atol=1e-9)


def test_screen_ray_pixel_quadrants():
    kwargs = dict(
        camera_pos=(0.0, 0.0, 5.0),
        camera_front=(0.0, 0.0, -1.0),
        camera_up=(0.0, 1.0, 0.0),
        fov_deg=60.0,
        width=800,
        height=600,
    )
    _, left = screen_ray(px=0, py=300, **kwargs)
    assert left[0] < 0.0, "left pixel must deviate towards -x (camera right = +x)"
    _, top = screen_ray(px=400, py=0, **kwargs)
    assert top[1] > 0.0, "top pixel must deviate towards +y"


def test_window_y_is_converted_from_bottom_left_to_top_left():
    assert window_to_screen_y(900, 0) == 900.0
    assert window_to_screen_y(900, 900) == 0.0


def test_ray_triangle_hit_and_miss():
    triangles = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    hit = ray_triangle_hit(np.array([0.2, 0.2, 1.0]), np.array([0.0, 0.0, -1.0]), triangles)
    assert hit is not None
    distance, index = hit
    assert index == 0
    assert math.isclose(distance, 1.0, rel_tol=1e-9)
    miss = ray_triangle_hit(np.array([2.0, 2.0, 1.0]), np.array([0.0, 0.0, -1.0]), triangles)
    assert miss is None


def test_ray_triangle_hit_picks_nearest():
    triangles = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0]],
        ]
    )
    hit = ray_triangle_hit(np.array([0.2, 0.2, 1.0]), np.array([0.0, 0.0, -1.0]), triangles)
    assert hit is not None and hit[1] == 0


def test_ray_plane_hit_parallel_and_behind():
    point = np.array([1.0, 2.0, 3.0])
    normal = np.array([0.0, 0.0, 1.0])
    hit = ray_plane(np.array([1.0, 2.0, 10.0]), np.array([0.0, 0.0, -1.0]), point, normal)
    assert hit is not None and np.allclose(hit, point)
    assert ray_plane(np.array([1.0, 2.0, 10.0]), np.array([1.0, 0.0, 0.0]), point, normal) is None
    assert ray_plane(np.array([1.0, 2.0, -10.0]), np.array([0.0, 0.0, -1.0]), point, normal) is None


# ---------------------------------------------------------------------------
# Headless pick-cycle integration test (GPU; same engine path as the window)
# ---------------------------------------------------------------------------

def _advance(simulator, frames: int) -> None:
    """Harness pacing, inlined to keep this module renderer-free on import."""
    substeps = math.ceil((1.0 / 24.0) / 0.001)
    for _ in range(frames):
        for _ in range(substeps):
            simulator.update(0.001)


@pytest.mark.sim
def test_pick_cycle_displaces_and_releases(qydp):
    simulator = qydp.simulator
    from harness.presets import apply_preset

    scene = scenes.load_scene("cloth-grid")

    def run(settle_frames: int, held_frames: int, relax_frames: int, drag_target_offset):
        apply_preset(simulator, "PDNewton")
        simulator.input_data(scene.input_data)
        _advance(simulator, settle_frames)
        settled = np.asarray(simulator.get_simulation_data(), dtype=np.float32)

        center = settled.mean(axis=0)
        origin = center + np.array([0.0, 0.0, 2.0])
        down = normalize(np.array([0.0, 0.0, -1.0]))

        controller = PickController(simulator, scene.panels)
        assert controller.press(origin, down), "ray through the cloth center must pick"
        _, _, global_ids = controller.last_pick

        target = center + drag_target_offset
        drag_dir = normalize(target - origin)
        for _ in range(held_frames):
            controller.drag(origin, drag_dir, np.array([0.0, 0.0, 1.0]))
            _advance(simulator, 1)
        held = np.asarray(simulator.get_simulation_data(), dtype=np.float32)
        controller.release()
        _advance(simulator, relax_frames)
        relaxed = np.asarray(simulator.get_simulation_data(), dtype=np.float32)
        return settled, held, relaxed, global_ids

    # Baseline: identical run without any pick (determinism tolerance below).
    def run_baseline(settle_frames: int, held_frames: int, relax_frames: int):
        apply_preset(simulator, "PDNewton")
        simulator.input_data(scene.input_data)
        _advance(simulator, settle_frames)
        _advance(simulator, held_frames)
        held = np.asarray(simulator.get_simulation_data(), dtype=np.float32)
        _advance(simulator, relax_frames)
        return held

    settle, held_frames, relax_frames = 30, 5, 40
    target_offset = np.array([0.35, 0.0, 0.15])

    settled, held, relaxed, ids = run(settle, held_frames, relax_frames, target_offset)
    baseline_held = run_baseline(settle, held_frames, relax_frames)

    held_pull = float(np.linalg.norm(held[ids] - baseline_held[ids], axis=1).max())
    relaxed_pull = float(np.linalg.norm(relaxed[ids] - baseline_held[ids], axis=1).max())
    toward = float(np.dot(held[ids].mean(axis=0) - baseline_held[ids].mean(axis=0), target_offset))

    assert held_pull > 0.05, f"picked vertices must follow the drag (pull={held_pull:.4f})"
    assert toward > 0.02, f"drag must pull towards the target (toward={toward:.4f})"
    assert relaxed_pull < held_pull, (
        f"release must relax the pick (relaxed={relaxed_pull:.4f} vs held={held_pull:.4f})"
    )
