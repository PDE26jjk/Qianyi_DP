"""Loader unit tests with a synthetic GarmentCodeData-style fixture."""

from __future__ import annotations

import json
import io
from pathlib import Path

import numpy as np
import pytest
import trimesh

from harness.gcd.errors import InvalidElementError, SeamPairingError
from harness.gcd.loader import load_element


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a PLY via trimesh (round-trips vertex/face order with process=False)."""
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices, dtype=np.float64),
                           faces=np.asarray(faces, dtype=np.int64), process=False)
    with open(path, "wb") as handle:
        mesh.export(handle, file_type="ply")


def _write_segmentation(path: Path, labels: list[str]) -> None:
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def _write_spec(path: Path, panels: list[str], stitches: list[list[dict]]) -> None:
    spec = {
        "pattern": {
            "panels": {name: {"vertices": [], "edges": []} for name in panels},
            "stitches": stitches,
            "panel_order": panels,
        }
    }
    path.write_text(json.dumps(spec), encoding="utf-8")


def build_element(
    element_dir: Path,
    *,
    with_body: bool = True,
) -> Path:
    """Build a synthetic two-panel element with a seamed shared edge.

    Geometry (cm, Y-up like the dataset):
      panel A: 2x2 grid at x in [0, 10], y in [0, 10]
      panel B: 2x2 grid at x in [10, 20], y in [0, 10]
    The shared seam column (x=10) is UV-duplicated: A1==B0 and A3==B2 are
    stored as consecutive duplicate positions.
    """
    element_dir = Path(element_dir)
    element_dir.mkdir(parents=True, exist_ok=True)
    element_id = element_dir.name

    a0 = np.array([0.0, 0.0, 0.0])
    a1 = np.array([10.0, 0.0, 0.0])
    a2 = np.array([0.0, 10.0, 0.0])
    a3 = np.array([10.0, 10.0, 0.0])
    b4 = np.array([20.0, 0.0, 0.0])
    b5 = np.array([20.0, 10.0, 0.0])
    vertices = np.asarray([a0, a1, a1, a2, a3, a3, b4, b5], dtype=np.float64)
    faces = np.asarray(
        [[0, 1, 3], [1, 4, 3], [1, 6, 4], [6, 7, 4]], dtype=np.int32
    )
    # Original ids: 0:a0, 1:a1/b0, 2:a2, 3:a3/b2, 4:b4, 5:b5.
    # Panel A owns faces (0,1,2),(1,3,2); panel B owns (1,4,3),(4,5,3), so
    # both panels contain both seam vertices (ids 1 and 3).
    _write_ply(element_dir / f"{element_id}_boxmesh.ply", vertices, faces)
    _write_segmentation(
        element_dir / f"{element_id}_sim_segmentation.txt",
        ["panel_a", "stitch_0", "panel_a", "stitch_0", "panel_b", "panel_b"],
    )
    _write_spec(
        element_dir / f"{element_id}_specification.json",
        ["panel_a", "panel_b"],
        [[{"panel": "panel_a", "edge": 0}, {"panel": "panel_b", "edge": 1}]],
    )
    (element_dir / f"{element_id}_vertex_labels.yaml").write_text(
        "waist:\n- 0\n- 5\n", encoding="utf-8"
    )
    (element_dir / f"{element_id}_design_params.yaml").write_text(
        "design:\n  meta:\n    upper:\n      v: TestShirt\n", encoding="utf-8"
    )
    # Reference drape: same topology, slight z deformation (cm).
    ref_vertices = vertices.copy()
    ref_vertices[:, 2] = np.array([0.1, 0.2, 0.2, 0.3, 0.3, 0.3, 0.4, 0.5])
    _write_ply(element_dir / f"{element_id}_sim.ply", ref_vertices, faces)

    if with_body:
        body = element_dir / "body.obj"
        body.write_text(
            "v 0 0 0\nv 0.2 0 0\nv 0 0.2 0\nv 0 0 0.1\n"
            "f 1 2 3\n",
            encoding="ascii",
        )
    return element_dir


def test_duplicate_mapping_units_and_panels(scratch_dir: Path) -> None:
    element = build_element(scratch_dir / "el")
    loaded = load_element(element, None)
    report = loaded.report

    assert report["mesh_vertices"] == 8
    assert report["original_vertices"] == 6
    assert report["panel_count"] == 2
    assert report["stitch_count"] == 1
    assert report["mismatches"] == []

    cloth = [m for m in loaded.input_data["mesh_list"] if m["object_type"] == 0]
    assert len(cloth) == 2
    for mesh in cloth:
        vertices = np.asarray(mesh["vertices"]).reshape(-1, 3)
        # Dataset is centimetres; loader converts to metres.
        assert vertices.max() < 0.21
        triangles = np.asarray(mesh["triangles"]).reshape(-1, 3)
        edges = set(map(tuple, np.asarray(mesh["edges"]).reshape(-1, 2).tolist()))
        for triangle in triangles:
            for a, b in (
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ):
                assert tuple(sorted((int(a), int(b)))) in edges


def test_sewings_and_attachments(scratch_dir: Path) -> None:
    element = build_element(scratch_dir / "el")
    loaded = load_element(element, None)
    assert len(loaded.input_data["sewings"]) == 1
    sewing = loaded.input_data["sewings"][0]
    assert sewing["patterns"] == [0, 1]
    assert sewing["stitches"].shape == (2, 2)
    # Identity-position pairs: (local a of seam id 1, local b of seam id 1),
    # (local a of seam id 3, local b of seam id 3).
    rows = sewing["stitches"].tolist()
    cloth = [m for m in loaded.input_data["mesh_list"] if m["object_type"] == 0]
    for (la, lb) in rows:
        pa = cloth[0]["vertices"].reshape(-1, 3)[la]
        pb = cloth[1]["vertices"].reshape(-1, 3)[lb]
        assert np.allclose(pa, pb, atol=1e-6)

    attached = [
        np.asarray(m["attached_vertices"], dtype=np.float32) for m in cloth
    ]
    assert float(attached[0].sum()) == 1.0  # original id 0 (waist)
    assert float(attached[1].sum()) == 1.0  # original id 5 (waist)


def test_body_obstacle(scratch_dir: Path) -> None:
    element = build_element(scratch_dir / "el", with_body=True)
    loaded = load_element(element, element / "body.obj")
    body = loaded.input_data["mesh_list"][-1]
    assert body["object_type"] == 1
    assert len(body["normals"]) == 3  # one triangle face
    assert loaded.report["body"]["aligned"] is True


def test_malformed_segmentation_raises(scratch_dir: Path) -> None:
    element = build_element(scratch_dir / "el")
    _write_segmentation(
        element / f"{element.name}_sim_segmentation.txt",
        ["panel_a"],  # wrong count
    )
    with pytest.raises(InvalidElementError):
        load_element(element, None)


def test_branched_seam_raises(scratch_dir: Path) -> None:
    element = scratch_dir / "el"
    element.mkdir(parents=True, exist_ok=True)
    element_id = element.name
    # A star seam: center 0 connected to leaves 1,2,3 -> degree 3 branch.
    # Vertices 4,5,6 carry the panel label so the mesh still splits.
    vertices = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [-10.0, 0.0, 0.0],
            [0.0, 0.0, 10.0],
            [20.0, 0.0, 10.0],
            [-20.0, 0.0, 10.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 4], [0, 2, 5], [0, 3, 6]], dtype=np.int32)
    _write_ply(element / f"{element_id}_boxmesh.ply", vertices, faces)
    _write_segmentation(
        element / f"{element_id}_sim_segmentation.txt",
        ["stitch_0", "stitch_0", "stitch_0", "stitch_0", "panel_a", "panel_a", "panel_a"],
    )
    _write_spec(
        element / f"{element_id}_specification.json",
        ["panel_a"],
        [[{"panel": "panel_a", "edge": 0}, {"panel": "panel_a", "edge": 1}]],
    )
    (element / f"{element_id}_vertex_labels.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(SeamPairingError):
        load_element(element, None)
