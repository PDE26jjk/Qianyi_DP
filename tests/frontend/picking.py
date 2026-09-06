"""Renderer-free picking math and pick lifecycle (design D4/D9).

Only numpy is imported here so the unit tests and the headless pick-cycle
integration test can run without Warp or a window. The window's mouse
handlers convert cursor coordinates to rays and forward them to
:class:`PickController`, which owns the engine ``pick_triangle*`` lifecycle.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def window_to_screen_y(height: int, y: float) -> float:
    """Convert a bottom-left window Y coordinate to a top-left pixel Y."""
    return float(height) - float(y)


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / max(float(np.linalg.norm(v)), _EPS)


def screen_ray(
    camera_pos,
    camera_front,
    camera_up,
    fov_deg: float,
    width: int,
    height: int,
    px: float,
    py: float,
) -> tuple[np.ndarray, np.ndarray]:
    """World-space ray through pixel (px, py) with a top-left pixel origin.

    ``fov_deg`` is the vertical field of view, matching the renderer's
    projection. Returns (origin, unit_direction).
    """
    origin = np.asarray(camera_pos, dtype=np.float64)
    forward = normalize(camera_front)
    right = normalize(np.cross(forward, np.asarray(camera_up, dtype=np.float64)))
    up = np.cross(right, forward)

    ndc_x = 2.0 * (px + 0.5) / width - 1.0
    ndc_y = 1.0 - 2.0 * (py + 0.5) / height
    tan_half = np.tan(np.deg2rad(fov_deg) / 2.0)
    aspect = width / height

    direction = forward + ndc_x * tan_half * aspect * right + ndc_y * tan_half * up
    return origin, normalize(direction)


def ray_triangle_hit(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
) -> tuple[float, int] | None:
    """Nearest hit of a ray against triangles of shape (T, 3, 3).

    Returns (distance, triangle_index) of the closest forward hit, or None.
    Vectorized Moller-Trumbore.
    """
    tris = np.asarray(triangles, dtype=np.float64)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    edge1 = v1 - v0
    edge2 = v2 - v0
    pvec = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, pvec)
    non_degenerate = np.abs(det) > _EPS
    inv_det = np.where(non_degenerate, 1.0 / np.where(non_degenerate, det, 1.0), 0.0)

    tvec = origin[None, :] - v0
    u = np.einsum("ij,ij->i", tvec, pvec) * inv_det
    qvec = np.cross(tvec, edge1)
    v = np.einsum("j,ij->i", direction, qvec) * inv_det
    t = np.einsum("ij,ij->i", edge2, qvec) * inv_det

    hit = (
        non_degenerate
        & (u >= -1e-9)
        & (v >= -1e-9)
        & (u + v <= 1.0 + 1e-9)
        & (t > 1e-6)
    )
    if not hit.any():
        return None
    indices = np.flatnonzero(hit)
    best = indices[np.argmin(t[indices])]
    return float(t[best]), int(best)


def ray_plane(
    origin: np.ndarray,
    direction: np.ndarray,
    point: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray | None:
    """Intersection of a ray with the plane through ``point`` with ``normal``."""
    normal = normalize(normal)
    denom = float(np.dot(direction, normal))
    if abs(denom) < _EPS:
        return None
    t = float(np.dot(np.asarray(point, dtype=np.float64) - origin, normal)) / denom
    if t <= 0.0:
        return None
    return origin + direction * t


class PickController:
    """Drives the engine's triangle pick constraint from world-space rays.

    The controller tests the FULL panel topology (including seam-bridging
    faces) and maps hits to (mesh_index, triangle_index) in ``input_data``
    coordinates. ``panels`` are ``PanelRender`` objects.
    """

    def __init__(self, simulator, panels) -> None:
        self._simulator = simulator
        self._panels = list(panels)
        self._triangles = [
            panel.vertices[panel.triangles].astype(np.float64) for panel in self._panels
        ]
        self.active: bool = False
        self.handle: int | None = None
        self.grab_point: np.ndarray | None = None
        # Last pick, for diagnostics and tests: (panel_index, triangle_index,
        # global cloth vertex ids).
        self.last_pick: tuple[int, int, np.ndarray] | None = None

    def press(
            self,
            origin: np.ndarray,
            direction: np.ndarray,
            positions: np.ndarray | None = None,
    ) -> bool:
        """Pick the closest cloth triangle under the ray, if any.

        The ray is tested against the cloth's *current* positions: when
        ``positions`` is omitted the controller reads them from the engine
        (``get_simulation_data``), so picking works after the cloth has moved.
        ``positions`` is the concatenated cloth vertex buffer in panel order.
        """
        if positions is None:
            positions = np.asarray(
                self._simulator.get_simulation_data(), dtype=np.float32
            )
        self._triangles = [
            positions[
                panel.vertex_offset: panel.vertex_offset + len(panel.vertices)
            ][panel.triangles].astype(np.float64)
            for panel in self._panels
        ]
        best: tuple[float, int, int] | None = None  # (t, panel, triangle)
        for panel_index, tris in enumerate(self._triangles):
            hit = ray_triangle_hit(origin, direction, tris)
            if hit is not None and (best is None or hit[0] < best[0]):
                best = (hit[0], panel_index, hit[1])
        if best is None:
            return False
        _, panel_index, triangle_index = best
        grab = origin + direction * best[0]
        mesh_index = self._panels[panel_index].mesh_index
        self.handle = int(
            self._simulator.pick_triangle(
                mesh_index, triangle_index, grab.astype(np.float32)
            )
        )
        self.grab_point = grab
        local_ids = self._panels[panel_index].triangles[triangle_index]
        global_ids = self._panels[panel_index].vertex_offset + local_ids
        self.last_pick = (panel_index, triangle_index, global_ids)
        self.active = True
        return True

    def drag(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        plane_normal: np.ndarray,
    ) -> bool:
        """Move the active pick target along its camera-facing drag plane."""
        if not self.active:
            return False
        target = ray_plane(origin, direction, self.grab_point, plane_normal)
        if target is None:
            return False
        self._simulator.pick_triangle_update(self.handle, target.astype(np.float32))
        return True

    def release(self) -> None:
        """Drop the active pick (safe to call when inactive)."""
        if self.active:
            self._simulator.pick_triangle_remove(self.handle)
        self.active = False
        self.handle = None
        self.grab_point = None
