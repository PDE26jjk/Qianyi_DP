"""Debug scene registry for the drape debug window.

A scene bundles the engine ``input_data`` with everything the window needs
to display it: per-panel render blocks (initial positions and full topology),
sewing chains as ordered panel-local vertex ids, an optional obstacle, and a
camera hint.
Adding a debug scene = adding a builder here and wiring it in
:func:`load_scene`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harness.meshspec import MeshSpec, validate_mesh_list

# Grid resolution of the default procedural scene (24x24 hangs well and is
# cheap enough to step at harness pacing).
_GRID_COLS = 24
_GRID_ROWS = 24

# Scene names that select a whole family (the element id is part of the name).
_GCD_PREFIX = "gcd:"

class SceneError(Exception):
    """Unknown scene name; the message lists the available scenes."""


class SceneSetupError(Exception):
    """A scene's environment preconditions are not met."""


@dataclass
class PanelRender:
    """One cloth panel with render-only topology annotations."""

    mesh_index: int  # index into input_data["mesh_list"]
    vertex_offset: int  # offset into the concatenated cloth vertex buffer
    vertices: np.ndarray  # (N, 3) float32 initial local positions
    triangles: np.ndarray  # (T, 3) int32 full (engine) topology


@dataclass
class SeamChain:
    """One side of a sewing entry, as an ordered chain of stitch vertices."""

    panel_index: int  # index into SceneData.panels
    vertex_ids: np.ndarray  # (K,) int64 panel-local, chain order


@dataclass
class ObstacleRender:
    vertices: np.ndarray  # (N, 3) float32 local positions
    triangles: np.ndarray  # (T, 3) int32


@dataclass
class SceneData:
    name: str
    input_data: dict
    panels: list[PanelRender]
    seams: list[SeamChain]
    obstacle: ObstacleRender | None
    camera_pos: tuple[float, float, float]
    camera_front: tuple[float, float, float]

    @property
    def cloth_vertices(self) -> np.ndarray:
        """All cloth vertices concatenated in panel order, (N, 3)."""
        return np.vstack([p.vertices for p in self.panels])

    @property
    def diagonal(self) -> float:
        """Scene bounding-box diagonal (used to size lines and the camera)."""
        parts = [self.cloth_vertices]
        if self.obstacle is not None:
            parts.append(self.obstacle.vertices)
        pts = np.vstack(parts)
        return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))


def _camera_hint(points: np.ndarray) -> tuple[tuple, tuple]:
    """A fixed third-person camera looking at the scene bounds.

    The vantage is high and pulled back, aiming slightly below the bbox
    center: drape content settles below the initial plane, and a generous
    framing keeps the settled cloth on screen without user camera moves.
    """
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.linalg.norm(points.max(axis=0) - points.min(axis=0))), 1e-6)
    pos = center + radius * np.array([1.0, -1.8, 1.2])
    look_at = center - np.array([0.0, 0.0, 0.15 * radius])
    front = look_at - pos
    front /= np.linalg.norm(front)
    return tuple(float(v) for v in pos), tuple(float(v) for v in front)


def _build_cloth_grid() -> SceneData:
    """A pinned hanging grid cloth (the smoke-test scene, no sewings).

    The grid is lifted above the ground so the pinned edge hangs like a
    curtain instead of crumpling onto the ground plane at pin height.
    """
    lift = np.array([0.0, 0.0, 1.5], dtype=np.float32)
    spec = MeshSpec(
        width_m=1.0,
        height_m=1.0,
        cols=_GRID_COLS,
        rows=_GRID_ROWS,
        fixed_vertex_indices=tuple(range(_GRID_COLS)),  # pin one edge row
    )
    input_data = spec.to_input_data()
    mesh = input_data["mesh_list"][0]
    mesh["vertices"] = (mesh["vertices"].reshape(-1, 3) + lift).reshape(-1)
    mesh["vertices_sim"] = (mesh["vertices_sim"].reshape(-1, 3) + lift).reshape(-1)
    validate_mesh_list(input_data["mesh_list"])
    vertices = spec.grid_vertices() + lift
    triangles = spec.triangles()
    panel = PanelRender(
        mesh_index=0,
        vertex_offset=0,
        vertices=vertices,
        triangles=triangles,
    )
    camera_pos, camera_front = _camera_hint(vertices)
    return SceneData(
        name="cloth-grid",
        input_data=input_data,
        panels=[panel],
        seams=[],
        obstacle=None,
        camera_pos=camera_pos,
        camera_front=camera_front,
    )


def _build_gcd(element_id: str) -> SceneData:
    """A GarmentCodeData element (panels + sewing chains + body obstacle).

    Simulates the dataset's closed box mesh with identity sewings - exactly
    the batch harness starting state - and renders panels with their full
    topology plus a distinct color.
    """
    root = os.environ.get("QYDP_GCD_ROOT")
    body = os.environ.get("QYDP_GCD_BODY")
    missing = [n for n, v in (("QYDP_GCD_ROOT", root), ("QYDP_GCD_BODY", body)) if not v]
    if missing:
        raise SceneSetupError(
            f"scene 'gcd:{element_id}' requires the GarmentCodeData dataset; "
            f"set {missing} (machine-specific paths live in LOCAL_DEV.md, gitignored)"
        )
    from harness.gcd.loader import load_element

    element = load_element(Path(root) / element_id, Path(body))
    mesh_list = element.input_data["mesh_list"]

    panels: list[PanelRender] = []
    obstacle: ObstacleRender | None = None
    vertex_offset = 0
    for mesh_index, mesh in enumerate(mesh_list):
        vertices = np.asarray(mesh["vertices"], dtype=np.float32).reshape(-1, 3)
        triangles = np.asarray(mesh["triangles"], dtype=np.int32).reshape(-1, 3)
        if int(mesh["object_type"]) == 0:
            panels.append(
                PanelRender(
                    mesh_index=mesh_index,
                    vertex_offset=vertex_offset,
                    vertices=vertices,
                    triangles=triangles,
                )
            )
            vertex_offset += len(vertices)
        else:
            obstacle = ObstacleRender(vertices=vertices, triangles=triangles)

    # Ordered chains from the sewing entries (design D10): stitches are
    # (K, 2) panel-local index pairs, column 0 on the first pattern, column 1
    # on the second.
    seams: list[SeamChain] = []
    for sewing in element.input_data["sewings"]:
        pa, pb = (int(i) for i in sewing["patterns"])
        stitches = np.asarray(sewing["stitches"], dtype=np.int64).reshape(-1, 2)
        for panel_index, column in ((pa, 0), (pb, 1)):
            ids = stitches[:, column]
            seams.append(SeamChain(panel_index=panel_index, vertex_ids=ids))

    camera_pos, camera_front = _camera_hint(
        np.vstack([p.vertices for p in panels]
                  + ([obstacle.vertices] if obstacle is not None else []))
    )
    return SceneData(
        name=f"gcd:{element_id}",
        input_data=element.input_data,
        panels=panels,
        seams=seams,
        obstacle=obstacle,
        camera_pos=camera_pos,
        camera_front=camera_front,
    )


def list_scenes() -> list[tuple[str, str]]:
    """Registered scene selectors with their source kind."""
    return [
        ("cloth-grid", "procedural"),
        ("gcd:<element_id>", "GarmentCodeData, closed garment (identity sewings)"),
    ]


def load_scene(name: str | None) -> SceneData:
    """Build a registered scene by name (see :func:`list_scenes`)."""
    name = name or "cloth-grid"
    if name == "cloth-grid":
        return _build_cloth_grid()
    if name.startswith(_GCD_PREFIX):
        return _build_gcd(name[len(_GCD_PREFIX):])
    available = ", ".join(n for n, _ in list_scenes())
    raise SceneError(f"unknown scene {name!r}; available: {available}")
