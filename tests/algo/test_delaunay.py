"""Cross-check ``delaunay_2d`` against an independent reference (scipy)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial import Delaunay

from harness import report

pytestmark = [pytest.mark.algo, pytest.mark.quick]


def _triangle_set(simplices: np.ndarray) -> set[tuple[int, int, int]]:
    return {tuple(sorted(map(int, tri))) for tri in simplices}


@pytest.mark.parametrize("seed", [42, 7, 123, 2026])
def test_delaunay_matches_scipy(qydp, capture, seed) -> None:
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.0, 1.0, (80, 2)).astype(np.float32)
    art = report.artifact_dir(f"algo/delaunay/seed_{seed}")
    capture.begin_case(art / "sim.log")
    try:
        triangles = qydp.geometry.delaunay_2d(
            points, np.empty((0, 2), dtype=np.int32)
        )
    finally:
        capture.end_case()
    triangles = np.asarray(triangles, dtype=np.int64)

    reference = Delaunay(np.asarray(points, dtype=np.float64))
    ours = _triangle_set(triangles)
    theirs = _triangle_set(reference.simplices)
    assert ours == theirs, (
        f"topology mismatch: {len(ours - theirs)} triangles only in qydp, "
        f"{len(theirs - ours)} only in scipy"
    )

