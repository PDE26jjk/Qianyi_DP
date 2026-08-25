"""Simulation smoke test for the standard corner-pinned grid cloth scene.

Verifies, over 60 frames @ 24fps with the canonical PDNewton configuration:
all frame data finite, pinned vertices stay put, free vertices move, and no
vertex flies outside the scene bounding box.
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
DT = 0.001
PINNED_DRIFT_TOL_M = 1e-3
MIN_FREE_DISP_M = 0.05
BOUNDS_MARGIN_M = 1.0
ROWS = 10
COLS = 10
PINNED_INDICES = (0, 9, 90, 99)


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

    # Free vertices actually move (gravity-driven cloth motion).
    free_disp = np.linalg.norm(
        run.local_frames[-1][free_indices] - run.local_frames[0][free_indices], axis=1
    )
    mean_free_disp = float(np.mean(free_disp))
    assert mean_free_disp > MIN_FREE_DISP_M, (
        f"mean free displacement {mean_free_disp} m below threshold"
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
