"""Optional GIF visualization test (opt-in via ``--gif``, marker ``visual``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec
from harness.render import render_gif, write_gallery

pytestmark = [pytest.mark.sim, pytest.mark.visual]

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_render_gif(qydp, capture, pytestconfig) -> None:
    if not pytestconfig.getoption("gif"):
        pytest.skip("GIF rendering is opt-in; run pytest with --gif")

    spec = MeshSpec(rows=8, cols=8, fixed_vertex_indices=(0, 7, 56, 63))
    art = report.artifact_dir("sim/visual")
    driver = SimDriver(qydp, frames=12)
    capture.begin_case(art / "sim.log")
    try:
        run = driver.run(spec.to_input_data())
    finally:
        capture.end_case()

    gif_dir = REPO_ROOT / "build" / "artifacts" / "visual"
    gif_path = render_gif(
        run.world_frames,
        spec.triangles(),
        gif_dir / "grid_cloth.gif",
        frame_skip=2,
    )
    gallery_path = write_gallery(
        gif_dir / "gallery.md",
        [
            {
                "name": "grid_cloth",
                "gif": str(gif_path),
                "params": {"solver": "PDNewton", "frames": 12},
            }
        ],
    )

    assert gif_path.exists() and gif_path.stat().st_size > 0
    assert gallery_path.exists()
