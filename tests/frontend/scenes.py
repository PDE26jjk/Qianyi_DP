"""Debug scene registry for the drape debug window.

A scene bundles the engine ``input_data`` with everything the window needs
to display it: per-panel render blocks (initial positions, full topology,
seam-vertex masks), sewing chains as
ordered panel-local vertex ids, an optional obstacle, and a camera hint.
Adding a debug scene = adding a builder here and wiring it in
:func:`load_scene`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harness.meshspec import MeshSpec, validate_mesh_list

# Grid resolution of the default procedural scene (24x24 hangs well and is
# cheap enough to step at harness pacing).
_GRID_COLS = 24
_GRID_ROWS = 24

# Scene names that select a whole family (the element id is part of the name).
_GCD_PREFIX = "gcd:"
_GCD_SEW_PREFIX = "gcdsew:"

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
    render_triangles: np.ndarray  # (T, 3) int32 display topology
    seam_mask: np.ndarray  # (N,) bool, vertex lies on a sewing chain


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
        render_triangles=triangles,
        seam_mask=np.zeros(len(vertices), dtype=bool),
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


# Pattern-to-garment sewing view: the dataset's box mesh is an already-closed
# garment whose sewings are identity pairs (zero distance) - the engine welds
# them instantly and nothing is ever pulled. The window variant instead
# places each panel on a SAFE RING around the body (bounding spheres clear of
# the obstacle and the floor), keeping panels rigid, so the sewing springs
# have a real gap to close. Radial scaling alone is not enough: large panels
# translated radially still interpenetrate the mannequin, and the collision
# response destabilizes the linear solve (PCG NaN, force magnitudes ~1e3).
_RING_MARGIN_M = 0.15
_FLOOR_CLEARANCE_M = 0.05


def _explode_panels(panels: list[PanelRender], mesh_list: list[dict], obstacle: ObstacleRender | None) -> None:
    """Rigidly move cloth panels onto a body-clearing ring (in place).

    Attachment/fixed weights are zeroed: pinned vertices would stay frozen
    at the separated positions and hold the garment off the body forever.
    """
    if obstacle is not None:
        body_center_xy = obstacle.vertices[:, :2].mean(axis=0)
        body_radius = float(np.linalg.norm(
            obstacle.vertices[:, :2] - body_center_xy, axis=1
        ).max())
    else:
        body_center_xy = np.zeros(2)
        body_radius = 0.0

    for panel in panels:
        center = panel.vertices.mean(axis=0)
        radius = float(np.linalg.norm(panel.vertices - center, axis=1).max())
        direction = center[:2] - body_center_xy
        norm = float(np.linalg.norm(direction))
        direction = direction / norm if norm > 1e-6 else np.array([1.0, 0.0])
        ring_distance = body_radius + radius + _RING_MARGIN_M
        target_xy = body_center_xy + direction * ring_distance
        moved = panel.vertices.copy()
        moved[:, :2] += (target_xy - center[:2]).astype(np.float32)
        moved[:, 2] += max(0.0, _FLOOR_CLEARANCE_M - moved[:, 2].min())
        panel.vertices = moved
        mesh = mesh_list[panel.mesh_index]
        flat = np.ascontiguousarray(moved.reshape(-1))
        mesh["vertices"] = flat
        mesh["vertices_sim"] = flat.copy()
        mesh["fixed_vertices"] = np.zeros(len(moved), dtype=np.float32)
        mesh["attached_vertices"] = np.zeros(len(moved), dtype=np.float32)


def _build_gcd(element_id: str, sewing_view: bool = False) -> SceneData:
    """A GarmentCodeData element (panels + sewing chains + body obstacle).

    ``sewing_view=False`` (default) simulates the dataset's closed box mesh
    with identity sewings - exactly the batch harness starting state - and
    renders panels with a small display-only separation so seams are visible.
    ``sewing_view=True`` rigidly moves the panels onto a body-clearing ring
    in the ENGINE INPUT as well (for future pull-sewing experiments; PDNewton
    currently has no sewing spring forces, so panels there just fall).
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
                    render_triangles=triangles,
                    seam_mask=np.zeros(len(vertices), dtype=bool),
                )
            )
            vertex_offset += len(vertices)
        else:
            obstacle = ObstacleRender(vertices=vertices, triangles=triangles)

    # Seam-vertex masks and ordered chains from the sewing entries (design
    # D10): stitches are (K, 2) panel-local index pairs, column 0 on the
    # first pattern, column 1 on the second.
    seams: list[SeamChain] = []
    for sewing in element.input_data["sewings"]:
        pa, pb = (int(i) for i in sewing["patterns"])
        stitches = np.asarray(sewing["stitches"], dtype=np.int64).reshape(-1, 2)
        for panel_index, column in ((pa, 0), (pb, 1)):
            ids = stitches[:, column]
            panels[panel_index].seam_mask[ids] = True
            seams.append(SeamChain(panel_index=panel_index, vertex_ids=ids))

    if sewing_view:
        _explode_panels(panels, mesh_list, obstacle)

    camera_pos, camera_front = _camera_hint(
        np.vstack([p.vertices for p in panels]
                  + ([obstacle.vertices] if obstacle is not None else []))
    )
    variant = "gcdsew" if sewing_view else "gcd"
    return SceneData(
        name=f"{variant}:{element_id}",
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
        ("gcd:<element_id>", "GarmentCodeData, closed garment (seams gapped for display)"),
        ("gcdsew:<element_id>", "GarmentCodeData, panels ring-separated (sewing experiments)"),
    ]


def load_scene(name: str | None) -> SceneData:
    """Build a registered scene by name (see :func:`list_scenes`)."""
    name = name or "cloth-grid"
    if name == "cloth-grid":
        return _build_cloth_grid()
    for prefix, sewing_view in ((_GCD_SEW_PREFIX, True), (_GCD_PREFIX, False)):
        if name.startswith(prefix):
            return _build_gcd(name[len(prefix):], sewing_view=sewing_view)
    available = ", ".join(n for n, _ in list_scenes())
    raise SceneError(f"unknown scene {name!r}; available: {available}")
