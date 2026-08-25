"""Performance baseline for ``sample_points`` (scale sweep, record-only).

Marked ``bench`` and not part of the default quick group; a hard threshold
policy is deferred until enough baseline data is collected.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from harness import report

pytestmark = pytest.mark.bench


def _square_scene(size: float):
    outer = np.array([[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]], dtype=np.float32)
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=np.int32)
    return outer, edges, np.array([4], dtype=np.int32), np.array([0], dtype=np.int32)


def test_sampling_scale_sweep(qydp, capture) -> None:
    sweeps = [
        {"size": 1.0, "radius": 0.2},
        {"size": 1.0, "radius": 0.1},
        {"size": 1.0, "radius": 0.05},
        {"size": 2.0, "radius": 0.1},
        {"size": 2.0, "radius": 0.05},
    ]
    results = []
    for case in sweeps:
        boundary, edges, curve_sizes, is_holes = _square_scene(case["size"])
        art = report.artifact_dir(f"bench/sampling/size{case['size']}_r{case['radius']}")
        capture.begin_case(art / "sim.log")
        try:
            start = time.perf_counter()
            points, triangles = qydp.geometry.sample_points(
                boundary, edges, curve_sizes, is_holes, case["radius"]
            )
            elapsed = time.perf_counter() - start
        finally:
            capture.end_case()
        results.append(
            {
                "size": case["size"],
                "radius": case["radius"],
                "n_boundary_points": len(boundary),
                "n_output_points": len(points),
                "n_triangles": len(triangles),
                "elapsed_s": round(elapsed, 6),
            }
        )

    baseline_path = report.artifact_dir("bench/sampling") / "baseline.json"
    baseline_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_path, "w", encoding="utf-8") as handle:
        json.dump({"sweeps": results}, handle, indent=2)

    assert len(results) == len(sweeps)
    for entry in results:
        assert entry["n_output_points"] >= entry["n_boundary_points"]
        assert entry["elapsed_s"] > 0

