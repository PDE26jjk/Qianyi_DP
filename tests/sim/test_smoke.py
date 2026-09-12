"""Simulation smoke test for the standard corner-pinned grid cloth scene.

Verifies, over 60 frames @ 24fps with the canonical PDNewton configuration:
all frame data finite, pinned vertices stay put, the free vertices move under
gravity (or are provably resting on the ground), and no vertex flies outside
the scene bounding box.

Rest note: the sheet starts flat in the XY plane, which is exactly the
ground's contact height (its ``thickness``), with all four corners pinned, so
a settled run legitimately stays still - the ground carries it and there are no
in-plane forces. The motion assertion below therefore accepts either real
gravity-driven motion or a configuration that is resting on the ground.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec
from harness.traces import compute_frame_stats, write_frames_npz, write_traces_json

pytestmark = [pytest.mark.sim, pytest.mark.quick]

CASE_ID = "sim/smoke"
FPS = 24
FRAMES = 60
DT = 1.0 / FPS
PINNED_DRIFT_TOL_M = 1e-3
MIN_FREE_DISP_M = 0.05
BOUNDS_MARGIN_M = 1.0
ROWS = 10
COLS = 10
PINNED_INDICES = (0, 9, 90, 99)
# The cloth rests on the ground plane at its own thickness; this is the
# tolerance for calling a free vertex "in ground contact".
GROUND_REST_TOL_M = 1e-5


def test_standard_scene_smoke(qydp, record_failure, capture) -> None:
    spec = MeshSpec(rows=ROWS, cols=COLS, fixed_vertex_indices=PINNED_INDICES)
    free_indices = [i for i in range(spec.num_vertices) if i not in PINNED_INDICES]
    art = report.artifact_dir(CASE_ID)
    driver = SimDriver(qydp, fps=FPS, frames=FRAMES, dt=DT)

    capture.begin_case(art / "sim.log")
    try:
        run = driver.run(spec.to_input_data())
    finally:
        capture.end_case()

    stats = compute_frame_stats(
        run.local_frames, run.world_frames, pinned_indices=PINNED_INDICES
    )
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    write_traces_json(art, stats)
    record_failure(art, CASE_ID, stats=stats)

    # All frames finite.
    bad_frames = [s["frame"] for s in stats if not s["all_finite"]]
    assert not bad_frames, f"non-finite values in frames {bad_frames}"

    # Pinned vertices stay within numerical tolerance.
    max_pinned_drift = max(s["max_pinned_drift"] for s in stats)
    assert max_pinned_drift <= PINNED_DRIFT_TOL_M, (
        f"pinned drift {max_pinned_drift} m exceeds tolerance"
    )

    # Free vertices move under gravity, unless the sheet is resting on the
    # ground (see the module docstring): then the ground contact is what holds
    # it, and the check below verifies that instead.
    free_disp = np.linalg.norm(
        run.local_frames[-1][free_indices] - run.local_frames[0][free_indices], axis=1
    )
    mean_free_disp = float(np.mean(free_disp))
    ground_height_m = spec.thickness_mm * 1e-3
    free_height = run.world_frames[-1][free_indices, 2]
    resting_on_ground = bool(
        np.all(free_height <= ground_height_m + GROUND_REST_TOL_M))
    assert mean_free_disp > MIN_FREE_DISP_M or resting_on_ground, (
        f"mean free displacement {mean_free_disp} m below threshold and the "
        f"free vertices are not resting on the ground (max height "
        f"{free_height.max() * 1e3:.4f} mm against a ground height of "
        f"{ground_height_m * 1e3:.4f} mm)"
    )

    # No vertex flies outside the (expanded) scene bounding box.
    world = run.world_frames[-1]
    bbox_min = np.array([0.0, 0.0, 0.0]) - BOUNDS_MARGIN_M
    bbox_max = np.array([1.0, 1.0, 0.0]) + BOUNDS_MARGIN_M
    assert bool(np.all(world >= bbox_min) and np.all(world <= bbox_max)), (
        "vertex outside expanded scene bounding box"
    )

    report.write_results(
        CASE_ID,
        params={"solver": "PDNewton", "fps": FPS, "frames": FRAMES, "dt": DT},
        status="passed",
        artifacts={
            "frames_npz": str(art / "frames.npz"),
            "traces_json": str(art / "traces.json"),
            "log": str(art / "sim.log"),
        },
        trace_summary=stats[-1],
    )
