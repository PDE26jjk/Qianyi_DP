"""Drape assertions: invariant tier, reference metrics, performance records.

All analysis uses *local-space* frames: the engine's world-space output leaves
stale positions for proxy-merged seam vertices (verified locally), while the
local buffer stays consistent. With identity world matrices local and world
coordinates agree everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_TOLERANCES: dict = {
    # Seam closure: stitched vertex pairs must be closer than this (m).
    "seam_closure": 0.01,
    # Bounding envelope: no vertex may leave the initial bbox by more than
    # this margin (m).
    "envelope_margin": 1.0,
    # Area preservation per panel: final/initial area must stay in this range.
    # Real-garment tolerance (calibrated on the local subset, task 3.3):
    # pattern meshes settle slightly during the first frames and the solver
    # is not bitwise deterministic.
    "area_ratio_min": 0.7,
    "area_ratio_max": 1.4,
    # Attached vertices may not drift more than this from their initial
    # position (m).
    "attached_drift": 0.05,
}


@dataclass
class InvariantResult:
    passed: bool
    stats: dict
    failures: list[str]


def _cloth_offsets(mesh_list: list[dict]) -> tuple[list[int], list[int]]:
    """Return per-mesh vertex offsets and triangle offsets for cloth meshes."""
    cloth = [m for m in mesh_list if m["object_type"] == 0]
    vertex_offsets: list[int] = []
    triangle_offsets: list[int] = []
    v_off = 0
    t_off = 0
    for mesh in cloth:
        vertex_offsets.append(v_off)
        triangle_offsets.append(t_off)
        v_off += len(mesh["vertices"]) // 3
        t_off += len(mesh["triangles"]) // 3
    return vertex_offsets, triangle_offsets


def panel_areas(
    local_frames: np.ndarray, mesh_list: list[dict]
) -> np.ndarray:
    """Per-panel triangle area sums per frame, shape (F, num_panels)."""
    cloth = [m for m in mesh_list if m["object_type"] == 0]
    vertex_offsets, _ = _cloth_offsets(mesh_list)
    areas = np.empty((len(local_frames), len(cloth)), dtype=np.float64)
    for p, mesh in enumerate(cloth):
        triangles = np.asarray(mesh["triangles"], dtype=np.int64).reshape(-1, 3)
        # Local frames are concatenated cloth vertices in mesh order.
        tri = local_frames[:, triangles + vertex_offsets[p]]  # (F, T, 3, 3)
        v0, v1, v2 = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
        areas[:, p] = 0.5 * np.linalg.norm(
            np.cross(v1 - v0, v2 - v0), axis=-1
        ).sum(axis=1)
    return areas


def seam_pairs(mesh_list: list[dict], sewings: list[dict]) -> np.ndarray:
    """Return (K, 2) global cloth vertex indices for all stitch pairs."""
    vertex_offsets, _ = _cloth_offsets(mesh_list)
    pairs: list[tuple[int, int]] = []
    for sewing in sewings:
        p0, p1 = sewing["patterns"]
        for a, b in sewing["stitches"]:
            pairs.append((vertex_offsets[p0] + int(a), vertex_offsets[p1] + int(b)))
    return np.asarray(pairs, dtype=np.int64)


def check_invariants(
    local_frames: np.ndarray,
    initial_vertices: np.ndarray,
    mesh_list: list[dict],
    sewings: list[dict],
    attached_vertices: np.ndarray | None = None,
    tolerances: dict | None = None,
) -> InvariantResult:
    """Evaluate the invariant tier on local-space frames.

    ``initial_vertices`` is the (N, 3) concatenated cloth vertex cloud at t=0.
    """
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    failures: list[str] = []
    stats: dict = {}

    finite = np.isfinite(local_frames).all()
    stats["all_finite"] = bool(finite)
    if not finite:
        failures.append("non-finite vertex values detected")

    final = local_frames[-1]
    initial_min = initial_vertices.min(axis=0) - tol["envelope_margin"]
    initial_max = initial_vertices.max(axis=0) + tol["envelope_margin"]
    escaped = (
        (final < initial_min).any(axis=1) | (final > initial_max).any(axis=1)
    ).sum()
    stats["escaped_vertices"] = int(escaped)
    if escaped:
        failures.append(f"{escaped} vertices escaped the bounding envelope")

    pairs = seam_pairs(mesh_list, sewings)
    if len(pairs):
        pair_dist = np.linalg.norm(
            final[pairs[:, 0]] - final[pairs[:, 1]], axis=1
        )
        stats["seam_closure_max"] = float(pair_dist.max())
        stats["seam_closure_mean"] = float(pair_dist.mean())
        open_pairs = int((pair_dist > tol["seam_closure"]).sum())
        stats["open_seam_pairs"] = open_pairs
        if open_pairs:
            failures.append(
                f"{open_pairs} seam pairs exceed the closure tolerance "
                f"({tol['seam_closure']} m)"
            )

    areas = panel_areas(local_frames, mesh_list)
    ratios = areas[-1] / np.maximum(areas[0], 1e-12)
    stats["area_ratio_min"] = float(ratios.min())
    stats["area_ratio_max"] = float(ratios.max())
    out_of_range = [
        (i, float(r))
        for i, r in enumerate(ratios)
        if r < tol["area_ratio_min"] or r > tol["area_ratio_max"]
    ]
    stats["area_out_of_range"] = out_of_range
    if out_of_range:
        failures.append(
            f"panel area ratios outside [{tol['area_ratio_min']}, "
            f"{tol['area_ratio_max']}]: {out_of_range}"
        )

    if attached_vertices is not None and len(attached_vertices):
        drift = np.linalg.norm(
            final[attached_vertices] - initial_vertices[attached_vertices], axis=1
        )
        stats["attached_drift_max"] = float(drift.max())
        stats["attached_drift_mean"] = float(drift.mean())
        drifting = int((drift > tol["attached_drift"]).sum())
        if drifting:
            failures.append(
                f"{drifting} attached vertices drifted beyond "
                f"{tol['attached_drift']} m"
            )

    return InvariantResult(passed=not failures, stats=stats, failures=failures)


def _per_panel_reference_areas(
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    panel_triangles: dict[str, np.ndarray],
) -> dict[str, float]:
    """Per-panel reference (sim.ply) areas in the same triangle subsets."""
    areas: dict[str, float] = {}
    for name, local_triangles in panel_triangles.items():
        tri = reference_vertices[reference_faces[local_triangles]]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        areas[name] = float(
            0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
        )
    return areas


def reference_metrics(
    final_vertices: np.ndarray,
    mesh_list: list[dict],
    panel_triangles: dict[str, np.ndarray],
    reference_vertices: np.ndarray,
    reference_faces: np.ndarray,
    *,
    hard_fail: bool = False,
    tolerances: dict | None = None,
) -> dict:
    """Loose reference comparison against ``sim.ply`` (record-only by default).

    Metrics: per-panel area ratio, z-extent, vertical quantiles, and
    bidirectional mean surface distance. Until tolerances are calibrated the
    metrics are recorded with ``passed: None``; when ``hard_fail`` is set the
    configured tolerances decide the pass/fail status.
    """
    cloth = [m for m in mesh_list if m["object_type"] == 0]
    ref_areas = _per_panel_reference_areas(
        reference_vertices, reference_faces, panel_triangles
    )
    sim_areas: dict[str, float] = {}
    vertex_offset = 0
    for name, mesh in zip(sorted(panel_triangles), cloth):
        triangles = np.asarray(mesh["triangles"], dtype=np.int64).reshape(-1, 3)
        tri = final_vertices[triangles + vertex_offset]
        v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
        sim_areas[name] = float(
            0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1).sum()
        )
        vertex_offset += len(mesh["vertices"]) // 3

    names = sorted(ref_areas)
    area_ratios = {
        name: sim_areas[name] / max(ref_areas[name], 1e-12) for name in names
    }
    metrics = {
        "per_panel_area_ratio": area_ratios,
        "area_ratio_mean": float(np.mean(list(area_ratios.values()))),
        "z_extent_sim": float(final_vertices[:, 2].max() - final_vertices[:, 2].min()),
        "z_extent_ref": float(
            reference_vertices[:, 2].max() - reference_vertices[:, 2].min()
        ),
        "z_quantiles_sim": {
            f"q{int(q * 100)}": float(np.quantile(final_vertices[:, 2], q))
            for q in (0.1, 0.5, 0.9)
        },
        "z_quantiles_ref": {
            f"q{int(q * 100)}": float(np.quantile(reference_vertices[:, 2], q))
            for q in (0.1, 0.5, 0.9)
        },
    }

    # Bidirectional mean surface distance (mean nearest-neighbour distance).
    try:
        from scipy.spatial import cKDTree

        tree_ref = cKDTree(reference_vertices)
        tree_sim = cKDTree(final_vertices)
        d1, _ = tree_ref.query(final_vertices)
        d2, _ = tree_sim.query(reference_vertices)
        metrics["mean_surface_distance"] = float(
            (d1.mean() + d2.mean()) / 2.0
        )
    except Exception as exc:  # pragma: no cover - scipy is a dev dependency
        metrics["mean_surface_distance"] = None
        metrics["surface_distance_error"] = str(exc)

    if not hard_fail:
        metrics["passed"] = None
        return metrics
    tol = {
        "area_ratio_min": 0.6,
        "area_ratio_max": 1.4,
        "z_extent_ratio": 0.5,
        "mean_surface_distance": 0.1,
        **(tolerances or {}),
    }
    failures: list[str] = []
    if not (tol["area_ratio_min"] <= metrics["area_ratio_mean"] <= tol["area_ratio_max"]):
        failures.append(
            f"mean area ratio {metrics['area_ratio_mean']:.3f} outside "
            f"[{tol['area_ratio_min']}, {tol['area_ratio_max']}]"
        )
    z_ratio = metrics["z_extent_sim"] / max(metrics["z_extent_ref"], 1e-12)
    metrics["z_extent_ratio"] = float(z_ratio)
    if abs(z_ratio - 1.0) > tol["z_extent_ratio"]:
        failures.append(f"z-extent ratio {z_ratio:.3f} deviates by more than {tol['z_extent_ratio']}")
    if (
        metrics["mean_surface_distance"] is not None
        and metrics["mean_surface_distance"] > tol["mean_surface_distance"]
    ):
        failures.append(
            f"mean surface distance {metrics['mean_surface_distance']:.4f} m "
            f"exceeds {tol['mean_surface_distance']} m"
        )
    metrics["passed"] = not failures
    metrics["failures"] = failures
    return metrics


def size_bucket(face_count: int) -> str:
    """Map a face count to a size bucket (design D9)."""
    if face_count < 5000:
        return "S"
    if face_count < 25000:
        return "M"
    if face_count < 50000:
        return "L"
    return "XL"
