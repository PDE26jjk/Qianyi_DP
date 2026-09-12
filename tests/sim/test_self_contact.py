"""Self-contact drape tests: a suspended cloth must not pass through itself.

Scene
-----
A square sheet lying flat in the XY plane with no ground under it, held by two
pins and released. Gravity folds the sheet in two: the two halves swing down
and land on each other, so the contact solver is the only thing keeping them
apart. With a very soft bending model the fold has almost no resistance and
the halves press together hard, which is the regime where a broken contact
response shows up as the two halves passing through each other.

The panel is triangulated by the engine's own pattern-mesh sampler
(``qydp.geometry.sample_points``: stratified jittered-grid sampling with
repulsion relaxation, then constrained Delaunay via gDel2D - see README.md,
"Pattern meshing"), which is what the Blender frontend feeds the simulator;
the two pins are the vertices nearest to the requested points.

Pacing
------
The engine's tight broad phase is only conservative while the per-substep
motion stays inside ``query_radius`` (see the comment in
``SolverPDNewton::step``), so the fold is driven with 0.5 ms substeps. At the
shipped 3 ms substep the same scene whips at ~10 mm per call and tunnels
straight through the broad phase; that case is recorded as a known issue in
``test_no_penetration_at_the_shipped_substep``.

Assertions
----------
Geometric and independent of the engine's own contact queries:

* ``_edge_triangle_crossings`` counts strict edge/triangle intersections
  between parts of the mesh that share no vertex. A crossing means the sheet
  has passed through itself. The detector is checked against hand-built cases
  in ``test_penetration_checker_known_cases``.
* ``_min_face_separation`` reports the smallest distance between a vertex and
  a non-incident triangle whose projection falls inside it.

The positive test asserts the state the engine can hold today: the fold must
settle with no crossing in the last quarter of the run and with the halves
resting apart by a fraction of the cloth thickness. Transient interpenetration
during the impact is measured and recorded as a metric (it is bounded, not
zero, because the contact model is a penalty).
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import (
    nearest_vertex,
    panel_mesh_dict,
    sample_panel,
    square_boundary,
    validate_mesh_list,
)
from harness.traces import compute_frame_stats, write_frames_npz, write_traces_json

# 0.5 ms substeps keep the per-substep motion inside the 2 mm query radius.
FOLD_DT = 0.005
FOLD_FRAMES = 600
FOLD_STEP_H = 0.0005
CHECK_STRIDE = 5
SIZE_M = 0.3
SPACING_M = 0.025
THICKNESS_MM = 0.1
# The frontend's own contact search radius.
QUERY_RADIUS_M = 0.002
BENDING_K = 0.01

PIN_TARGETS = {
    "midline": ((0.0, 0.5 * SIZE_M), (SIZE_M, 0.5 * SIZE_M)),
    "diagonal": ((0.0, 0.0), (SIZE_M, SIZE_M)),
}

# The fold is only meaningful if the sheet actually folded.
MIN_FOLD_EXTENT_M = 0.05

# Two shells in contact rest one thickness apart; half a thickness is a loose
# floor that only a real pass-through can violate.
MIN_SEPARATION_FRACTION = 0.5

CONTACT_OVERLAY = {
    "ground": 0.0,
    "bending_k": BENDING_K,
    "step_h": FOLD_STEP_H,
    "query_radius": QUERY_RADIUS_M,
}
NO_CONTACT_OVERLAY = dict(CONTACT_OVERLAY, vf_force_k=0.0, ee_force_k=0.0,
                          ef_force_k=0.0)


def _edges_of(triangles: np.ndarray) -> np.ndarray:
    """Unique undirected edges of a triangle list."""
    pairs = np.vstack([
        triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]],
    ])
    return np.unique(np.sort(pairs, axis=1), axis=0).astype(np.int32)


def _signed_plane_distances(
    positions: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Signed distance of every vertex to every triangle plane.

    Returns ``(distance, triangle_origin, unit_normal)`` with ``distance`` of
    shape (T, V).
    """
    origin = positions[triangles[:, 0]]
    edge1 = positions[triangles[:, 1]] - origin
    edge2 = positions[triangles[:, 2]] - origin
    normal = np.cross(edge1, edge2)
    length = np.linalg.norm(normal, axis=1)
    valid = length > 1e-14
    normal[valid] /= length[valid, None]
    relative = positions[None, :, :] - origin[:, None, :]
    distance = np.einsum("tvd,td->tv", relative, normal)
    distance[~valid] = 0.0
    return distance, origin, normal


def _crossing_pairs(
    positions: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
    epsilon_m: float = 1e-7,
) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays ``(triangle, edge)`` of every strict crossing.

    A pair counts only when the edge's endpoints sit strictly on opposite
    sides of the triangle's plane (more than ``epsilon_m`` away from it), the
    crossing point falls inside the triangle, and the triangle shares no
    vertex with the edge. Coplanar and grazing configurations - the flat
    starting sheet, or two halves resting against each other - are therefore
    never reported.
    """
    distance, origin, _ = _signed_plane_distances(positions, triangles)
    edge_a = edges[:, 0]
    edge_b = edges[:, 1]
    distance_a = distance[:, edge_a]
    distance_b = distance[:, edge_b]
    straddles = (
        ((distance_a > epsilon_m) & (distance_b < -epsilon_m))
        | ((distance_a < -epsilon_m) & (distance_b > epsilon_m))
    )
    incident = np.zeros((triangles.shape[0], positions.shape[0]), dtype=bool)
    incident[np.arange(triangles.shape[0])[:, None], triangles] = True
    shares_vertex = incident[:, edge_a] | incident[:, edge_b]
    candidates = np.nonzero(straddles & ~shares_vertex)
    if candidates[0].size == 0:
        return candidates

    tri_idx, edge_idx = candidates
    start = positions[edge_a[edge_idx]]
    end = positions[edge_b[edge_idx]]
    d_start = distance_a[tri_idx, edge_idx]
    d_end = distance_b[tri_idx, edge_idx]
    parameter = d_start / (d_start - d_end)
    point = start + (end - start) * parameter[:, None]

    v0 = origin[tri_idx]
    v1 = positions[triangles[tri_idx, 1]]
    v2 = positions[triangles[tri_idx, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    relative = point - v0
    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    d20 = np.einsum("ij,ij->i", relative, e1)
    d21 = np.einsum("ij,ij->i", relative, e2)
    denominator = d00 * d11 - d01 * d01
    safe = np.abs(denominator) > 1e-20
    denominator_safe = np.where(safe, denominator, 1.0)
    bary_v = np.where(safe, (d11 * d20 - d01 * d21) / denominator_safe, 0.0)
    bary_w = np.where(safe, (d00 * d21 - d01 * d20) / denominator_safe, 0.0)
    inside = (bary_v > 1e-9) & (bary_w > 1e-9) & (bary_v + bary_w < 1.0 - 1e-9)
    keep = np.nonzero(inside)[0]
    return tri_idx[keep], edge_idx[keep]


def _edge_triangle_crossings(
    positions: np.ndarray,
    triangles: np.ndarray,
    edges: np.ndarray,
    epsilon_m: float = 1e-7,
) -> int:
    """Number of strict edge/triangle crossings between non-adjacent parts."""
    return int(_crossing_pairs(positions, triangles, edges, epsilon_m)[0].size)


def _inside_plane_pairs(
    positions: np.ndarray,
    triangles: np.ndarray,
    origin: np.ndarray,
    inside_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pairs (triangle, vertex) whose projection lies inside the triangle."""
    v0 = origin
    v1 = positions[triangles[:, 1]]
    v2 = positions[triangles[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    relative = positions[None, :, :] - v0[:, None, :]
    d00 = np.einsum("ij,ij->i", e1, e1)
    d01 = np.einsum("ij,ij->i", e1, e2)
    d11 = np.einsum("ij,ij->i", e2, e2)
    # (T, V) dot products: contract the xyz axis of every (triangle, vertex)
    # offset with that triangle's own edge vector.
    d20 = np.einsum("tvj,tj->tv", relative, e1)
    d21 = np.einsum("tvj,tj->tv", relative, e2)
    denominator = d00 * d11 - d01 * d01
    safe = np.abs(denominator) > 1e-20
    denominator_safe = np.where(safe, denominator, 1.0)[:, None]
    bary_v = np.where(
        safe[:, None],
        (d11[:, None] * d20 - d01[:, None] * d21) / denominator_safe,
        0.0,
    )
    bary_w = np.where(
        safe[:, None],
        (d00[:, None] * d21 - d01[:, None] * d20) / denominator_safe,
        0.0,
    )
    inside = (
        (bary_v > inside_tolerance)
        & (bary_w > inside_tolerance)
        & (bary_v + bary_w < 1.0 - inside_tolerance)
    )
    incident = np.zeros_like(inside)
    incident[np.arange(triangles.shape[0])[:, None], triangles] = True
    inside &= ~incident
    return np.nonzero(inside)


def _min_face_separation(
    positions: np.ndarray,
    triangles: np.ndarray,
    inside_tolerance: float = 1e-3,
) -> float:
    """Smallest vertex-to-triangle distance for non-incident pairs (metres).

    Only pairs whose projected vertex falls strictly inside the triangle are
    considered, so the ordinary flat sheet - where vertices sit on the
    boundary of their neighbours' planes - contributes nothing. Returns
    ``nan`` when no such pair exists (unfolded sheet, or halves far apart).
    """
    distance, origin, _ = _signed_plane_distances(positions, triangles)
    tri_idx, vertex_idx = _inside_plane_pairs(
        positions, triangles, origin, inside_tolerance)
    if tri_idx.size == 0:
        return float("nan")
    return float(np.abs(distance[tri_idx, vertex_idx]).min())


def test_penetration_checker_known_cases() -> None:
    """The crossing detector on hand-built geometry with known answers."""
    triangles = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int32)
    edges = _edges_of(triangles)
    # A vertical triangle driven through a horizontal one: two of its edges
    # pierce the horizontal triangle, so two crossings are expected.
    piercing = np.asarray([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.2, 0.2, -0.5), (0.2, 0.2, 0.5), (0.8, 0.5, 0.0),
    ])
    assert _edge_triangle_crossings(piercing, triangles, edges) == 2
    separated = np.asarray([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.2, 0.2, 0.001), (0.2, 0.2, 0.5), (0.8, 0.5, 0.001),
    ])
    assert _edge_triangle_crossings(separated, triangles, edges) == 0
    assert _min_face_separation(separated, triangles) == pytest.approx(0.001)
    parallel = np.asarray([
        (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
        (0.0, 0.0, 0.001), (1.0, 0.0, 0.001), (0.0, 1.0, 0.001),
    ])
    assert _edge_triangle_crossings(parallel, triangles, edges) == 0


def _sample_fold_panel(qydp, pin_targets):
    boundary, segments = square_boundary(SIZE_M, SPACING_M)
    vertices, triangles = sample_panel(qydp, boundary, segments, SPACING_M)
    pins = tuple(nearest_vertex(vertices, target) for target in pin_targets)
    mesh = panel_mesh_dict(
        vertices,
        triangles,
        spacing_m=SPACING_M,
        thickness_mm=THICKNESS_MM,
        fixed_vertex_indices=pins,
    )
    validate_mesh_list([mesh])
    return vertices, triangles, pins, mesh


def _run_fold(qydp, pin_targets, overlay, frames: int = FOLD_FRAMES):
    vertices, triangles, pins, mesh = _sample_fold_panel(qydp, pin_targets)
    driver = SimDriver(qydp, fps=24, frames=frames, dt=FOLD_DT)
    run = driver.run({"mesh_list": [mesh], "sewings": []},
                     params_overlay=overlay)
    return vertices, triangles, pins, run


def _crossing_counts(frames, triangles, edges, stride: int = CHECK_STRIDE) -> dict:
    """Crossings per sampled frame, split into the whole run and its tail."""
    count = frames.shape[0]
    tail_start = int(count * 0.75)
    peak = 0
    tail_peak = 0
    tail_total = 0
    tail_sampled = 0
    for index in range(0, count, stride):
        found = _edge_triangle_crossings(
            np.asarray(frames[index], dtype=np.float64), triangles, edges)
        peak = max(peak, found)
        if index >= tail_start:
            tail_total += found
            tail_sampled += 1
            tail_peak = max(tail_peak, found)
    return {"peak_any_frame": peak, "tail_total": tail_total,
            "tail_peak": tail_peak, "tail_sampled_frames": tail_sampled}


@pytest.mark.sim
@pytest.mark.parametrize(
    "pin_name,case_id",
    [
        ("midline", "sim/self_contact_midline"),
        ("diagonal", "sim/self_contact_diagonal"),
    ],
    ids=["pinned-midline", "pinned-diagonal"],
)
def test_suspended_cloth_folds_without_passing_through(
    qydp, record_failure, capture, pin_name, case_id
) -> None:
    """The two halves collide and stay on their own side of each other."""
    art = report.artifact_dir(case_id)
    capture.begin_case(art / "sim.log")
    try:
        rest, triangles, pins, run = _run_fold(
            qydp, PIN_TARGETS[pin_name], CONTACT_OVERLAY)
    finally:
        capture.end_case()

    stats = compute_frame_stats(
        run.local_frames, run.world_frames, pinned_indices=pins)
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    write_traces_json(art, stats)
    record_failure(art, case_id, stats=stats)

    edges = _edges_of(triangles)
    crossings = _crossing_counts(run.world_frames, triangles, edges)
    final = np.asarray(run.world_frames[-1], dtype=np.float64)
    separation = _min_face_separation(final, triangles)
    fold_extent = float(np.ptp(final[:, 2]) - np.ptp(rest[:, 2]))

    report.write_results(
        case_id,
        params={"solver": "PDNewton", "frames": FOLD_FRAMES, "dt": FOLD_DT,
                "pin_configuration": pin_name, **CONTACT_OVERLAY},
        status="passed" if crossings["tail_total"] == 0 else "failed",
        trace_summary={**stats[-1], "fold_extent_m": fold_extent,
                       "min_face_separation_m": separation,
                       "self_contact": crossings},
    )

    assert all(s["all_finite"] for s in stats), "non-finite frame"
    assert fold_extent > MIN_FOLD_EXTENT_M, (
        f"the sheet only folded {fold_extent:.4f} m; the scene is not "
        "exercising self-contact"
    )
    assert crossings["tail_total"] == 0, (
        f"the folded halves are interpenetrating at the end of the run "
        f"({crossings['tail_total']} crossing(s) over "
        f"{crossings['tail_sampled_frames']} sampled frames in the last "
        f"quarter, peak {crossings['tail_peak']} on one frame)"
    )
    assert separation > MIN_SEPARATION_FRACTION * THICKNESS_MM * 1e-3, (
        f"folded halves came within {separation * 1e3:.4f} mm, below the "
        f"{MIN_SEPARATION_FRACTION * THICKNESS_MM:.3f} mm floor"
    )


@pytest.mark.sim
def test_contacts_disabled_passes_through(qydp, record_failure, capture) -> None:
    """Control: with the contact stiffnesses zeroed the fold passes through.

    Proves the geometric checks in the test above can see a penetration; if
    this ever stopped penetrating, the positive test would be vacuous.
    """
    case_id = "sim/self_contact_control"
    art = report.artifact_dir(case_id)
    capture.begin_case(art / "sim.log")
    try:
        _, triangles, pins, run = _run_fold(
            qydp, PIN_TARGETS["midline"], NO_CONTACT_OVERLAY)
    finally:
        capture.end_case()

    stats = compute_frame_stats(
        run.local_frames, run.world_frames, pinned_indices=pins)
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    write_traces_json(art, stats)
    record_failure(art, case_id, stats=stats)

    edges = _edges_of(triangles)
    crossings = _crossing_counts(run.world_frames, triangles, edges)
    report.write_results(
        case_id,
        params={"solver": "PDNewton", "frames": FOLD_FRAMES, "dt": FOLD_DT,
                "contacts": "disabled"},
        status="passed" if crossings["tail_total"] > 0 else "failed",
        trace_summary={**stats[-1], "self_contact": crossings},
    )
    assert crossings["tail_total"] > 0, (
        "the contact-free control did not pass through itself during the "
        "last quarter; the geometric check is not sensitive to penetration"
    )


@pytest.mark.sim
@pytest.mark.xfail(
    strict=True,
    reason=(
        "known issue: at the shipped 3 ms substep the ~1.2 m/s whip moves "
        "~10 mm per update() call, far past the 1-2 mm tight broad-phase "
        "radius, so the halves tunnel through each other transiently "
        "(~100-300 crossings per frame around the impact, against 0 at "
        "0.2-0.5 ms substeps). Remove this marker once contact detection is "
        "conservative with respect to the per-substep motion."
    ),
)
def test_no_penetration_at_the_shipped_substep(qydp, capture) -> None:
    """No self-penetration anywhere at the shipped substep (known issue)."""
    art = report.artifact_dir("sim/self_contact_shipped_substep")
    overlay = dict(CONTACT_OVERLAY, step_h=0.003)
    capture.begin_case(art / "sim.log")
    try:
        _, triangles, _, run = _run_fold(
            qydp, PIN_TARGETS["midline"], overlay, frames=120)
    finally:
        capture.end_case()

    edges = _edges_of(triangles)
    worst = 0
    for frame in run.world_frames:
        worst = max(worst, _edge_triangle_crossings(
            np.asarray(frame, dtype=np.float64), triangles, edges))
    assert worst == 0, (
        f"the sheet passed through itself {worst} times per frame at the "
        "shipped substep"
    )
