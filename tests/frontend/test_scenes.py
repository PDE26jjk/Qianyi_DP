"""CPU-only scene registry tests (no GPU, no window, no Warp import).

Guards the registry contract: every registered scene selector resolves to a
loadable SceneData whose input_data passes the mesh-list validation, and
environment preconditions fail with the documented setup message.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from frontend import scenes


def test_scene_listing_contains_both_sources():
    listing = scenes.list_scenes()
    names = [name for name, _ in listing]
    kinds = {kind.split(",")[0] for _, kind in listing}
    assert "cloth-grid" in names
    assert "procedural" in kinds
    assert "GarmentCodeData" in kinds


def test_cloth_grid_builds_valid_input_data():
    from harness.meshspec import validate_mesh_list

    scene = scenes.load_scene("cloth-grid")
    validate_mesh_list(scene.input_data["mesh_list"])
    assert len(scene.panels) == 1
    assert scene.seams == []
    assert scene.obstacle is None
    panel = scene.panels[0]
    assert len(panel.triangles) > 0
    # Camera hint points at the cloth.
    front = np.asarray(scene.camera_front)
    center = scene.cloth_vertices.mean(axis=0)
    to_center = center - np.asarray(scene.camera_pos)
    assert float(np.dot(front, to_center / np.linalg.norm(to_center))) > 0.9


def test_unknown_scene_lists_candidates():
    with pytest.raises(scenes.SceneError, match="cloth-grid"):
        scenes.load_scene("does-not-exist")


def test_gcd_scene_without_env_reports_setup(monkeypatch):
    monkeypatch.delenv("QYDP_GCD_ROOT", raising=False)
    monkeypatch.delenv("QYDP_GCD_BODY", raising=False)
    with pytest.raises(scenes.SceneSetupError, match="QYDP_GCD_ROOT"):
        scenes.load_scene("gcd:whatever")


@pytest.mark.data
@pytest.mark.skipif(
    not os.environ.get("QYDP_GCD_ROOT") or not os.environ.get("QYDP_GCD_BODY"),
    reason="GarmentCodeData dataset not configured (set QYDP_GCD_ROOT/QYDP_GCD_BODY; see LOCAL_DEV.md)",
)
def test_gcd_scene_loads_raw_panel_topology():
    from harness.meshspec import validate_mesh_list

    root = Path(os.environ["QYDP_GCD_ROOT"])
    element_id = sorted(d.name for d in root.iterdir() if d.is_dir())[0]
    scene = scenes.load_scene(f"gcd:{element_id}")
    validate_mesh_list(scene.input_data["mesh_list"])

    assert len(scene.panels) >= 2, "a garment must decompose into panels"
    assert scene.obstacle is not None, "the body obstacle must be present"
    assert scene.seams, "sewing chains must be registered for rendering"
    assert scene.diagonal > 0.0

    for chain in scene.seams:
        panel = scene.panels[chain.panel_index]
        assert chain.vertex_ids.max() < len(panel.vertices)
