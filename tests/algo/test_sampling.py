"""Geometry correctness tests for ``sample_points``.

The spacing oracle is derived from ``src/geometry/sample_points.cu``: the
sampler is grid-based with ``one_grid_length = radius / sqrt(2)`` (NOT
Poisson-disc), boundary points are placed on the grid as obstacles, interior
points are generated one per interior cell and separated by a capped repulsion
force. The minimum-spacing assertion therefore uses ``radius/sqrt(2)`` minus a
relative tolerance that accounts for the capped repulsion iterations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import cKDTree

from harness import report

pytestmark = [pytest.mark.algo, pytest.mark.quick]

MIN_SPACING_REL_TOL = 0.15
AREA_RTOL = 0.03
HOLE_SIZE = 0.5


def _square_scene(size: float = 1.0, with_hole: bool = False):
    """Build a square polygon (CCW), optionally with a CW square hole."""
    outer = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    curve_sizes = np.array([4], dtype=np.int32)
    is_holes = np.array([0], dtype=np.int32)
    if not with_hole:
        return outer, edges, curve_sizes, is_holes, []

    h = HOLE_SIZE * size
    off = (size - h) / 2.0
    hole = np.array(
        [[off, off], [off, off + h], [off + h, off + h], [off + h, off]],
        dtype=np.float32,
    )
    boundary = np.vstack([outer, hole])
    hole_edges = np.array([[4, 5], [5, 6], [6, 7], [7, 4]], dtype=np.int32)
    return (
        boundary,
        np.vstack([edges, hole_edges]),
        np.array([4, 4], dtype=np.int32),
        np.array([0, 1], dtype=np.int32),
        [hole.astype(np.float64)],
    )


def _ray_cast(point, polygon) -> bool:
    """Even-odd ray casting point-in-polygon test."""
    x, y = point
    inside = False
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside


def _point_segment_distance(p, a, b) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _min_distance_to_polygon(point, polygon) -> float:
    n = len(polygon)
    return min(
        _point_segment_distance(point, polygon[i], polygon[(i + 1) % n])
        for i in range(n)
    )


def _point_in_polygon(point, outer, holes, tol: float) -> bool:
    if not _ray_cast(point, outer):
        return _min_distance_to_polygon(point, outer) <= tol
    for hole in holes:
        if _ray_cast(point, hole) and _min_distance_to_polygon(point, hole) > tol:
            return False
    return True


def _triangle_area_sum(points, triangles) -> float:
    a = points[triangles[:, 0]]
    b = points[triangles[:, 1]]
    c = points[triangles[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (
        b[:, 1] - a[:, 1]
    ) * (c[:, 0] - a[:, 0])
    return float(0.5 * np.abs(cross).sum())


@pytest.mark.parametrize("radius", [0.05, 0.1, 0.2])
def test_sampled_point_spacing(qydp, capture, radius) -> None:
    boundary, edges, curve_sizes, is_holes, _ = _square_scene()
    art = report.artifact_dir(f"algo/sampling/spacing_r{radius}")
    capture.begin_case(art / "sim.log")
    try:
        points, _ = qydp.geometry.sample_points(
            boundary, edges, curve_sizes, is_holes, radius
        )
    finally:
        capture.end_case()
    points = np.asarray(points, dtype=np.float64)
    assert len(points) >= len(boundary)

    min_allowed = radius / np.sqrt(2.0) - MIN_SPACING_REL_TOL * radius
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    min_pair = float(distances[:, 1].min())
    assert min_pair >= min_allowed, (
        f"min pair distance {min_pair} below oracle {min_allowed} "
        f"(grid spacing radius/sqrt(2) = {radius / np.sqrt(2.0)})"
    )

    # The oracle applies to boundary point pairs as well (first N output points
    # are the input boundary points, order preserved).
    boundary_count = len(boundary)
    boundary_tree = cKDTree(points[:boundary_count])
    boundary_distances, _ = boundary_tree.query(points[:boundary_count], k=2)
    min_boundary = float(boundary_distances[:, 1].min())
    assert min_boundary >= min_allowed, (
        f"min boundary pair distance {min_boundary} below oracle {min_allowed}"
    )


@pytest.mark.parametrize("with_hole", [False, True])
def test_triangulation_area_and_bounds(qydp, capture, with_hole) -> None:
    boundary, edges, curve_sizes, is_holes, holes = _square_scene(with_hole=with_hole)
    art = report.artifact_dir(f"algo/sampling/triangulation_hole_{int(with_hole)}")
    capture.begin_case(art / "sim.log")
    try:
        points, triangles = qydp.geometry.sample_points(
            boundary, edges, curve_sizes, is_holes, 0.1
        )
    finally:
        capture.end_case()
    points = np.asarray(points, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    containment_tol = 0.1 / np.sqrt(2.0)

    assert triangles.size == 0 or int(triangles.max()) < len(points)
    outer = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    expected_area = 1.0 - (HOLE_SIZE * HOLE_SIZE if with_hole else 0.0)
    area = _triangle_area_sum(points, triangles)
    assert area == pytest.approx(expected_area, rel=AREA_RTOL), (
        f"triangle area sum {area} != polygon area {expected_area}"
    )

    # No out-of-bounds triangles: centroids stay inside the polygon.
    for tri in triangles:
        centroid = points[tri].mean(axis=0)
        assert _point_in_polygon(centroid, outer, holes, containment_tol), (
            f"triangle centroid {centroid} outside polygon"
        )


def test_sampled_points_contained(qydp, capture) -> None:
    boundary, edges, curve_sizes, is_holes, holes = _square_scene(with_hole=True)
    art = report.artifact_dir("algo/sampling/containment")
    capture.begin_case(art / "sim.log")
    try:
        points, _ = qydp.geometry.sample_points(boundary, edges, curve_sizes, is_holes, 0.1)
    finally:
        capture.end_case()
    points = np.asarray(points, dtype=np.float64)
    outer = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    containment_tol = 0.1 / np.sqrt(2.0)
    outside = [
        p.tolist()
        for p in points
        if not _point_in_polygon(p, outer, holes, containment_tol)
    ]
    assert not outside, f"{len(outside)} sampled points outside the polygon: {outside[:5]}"
