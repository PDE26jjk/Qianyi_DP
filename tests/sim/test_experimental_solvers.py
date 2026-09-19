"""Experimental solver governance (VBD / XPBD / Explicit).

Re-enabled by the `experimental-solver-parity` change: each non-PDNewton solver
runs a normal smoke case on the standard procedural scene and a seam case on a
two-panel scene whose panels are joined only by the engine's stitch
constraints. A solver that fails a case is marked `xfail(strict=True)` with its
measured failure mode, so a state flip in either direction surfaces explicitly
instead of passing silently (see spec: Experimental solver governance).

PDNewton runs the same seam case as the control: removing `sewing_k` from it
opens the seam by orders of magnitude, so the metric is known to detect a
missing stitch term rather than just a stiff drape.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec, seamed_panels_input_data
from harness.traces import compute_frame_stats, write_frames_npz, write_traces_json

pytestmark = [pytest.mark.sim, pytest.mark.quick]

FRAMES = 60
FPS = 24
DT = 1.0 / FPS
PINNED_INDICES = (0, 9, 90, 99)
# The standard smoke case uses a 50 mm bar for PDNewton's canonical block. The
# experimental presets take far coarser steps (VBD 10 ms, Explicit 0.25 ms) and
# are not tuned for this 1 m sheet, so the bar here only has to catch a solver
# that does not move at all (VBD measures 0.8 mm in 60 frames); the seam case
# below is where the quality of the physics is checked.
MIN_FREE_DISP_M = 5e-3
PINNED_DRIFT_TOL_M = 1e-3
GROUND_REST_TOL_M = 1e-5

SEAM_FRAMES = 30
# PDNewton and Explicit hold the seam within 1 mm, VBD within 1.5 mm; 5 mm
# leaves room for the solvers' coarser time steps while still failing loudly
# when a solver never applies the stitch constraint (the measured gap there is
# metres, not millimetres).
SEAM_TOL_M = 5e-3
# After the projection gate opens the clusters merge to a single point
# (measured 0.000 mm); the tolerance only has to catch a solver that never
# projects.
SEAM_PROJECTION_TOL_M = 1e-4

_EXPERIMENTAL_SOLVERS = ("VBD", "XPBD", "Explicit")

# Measured modes of the cases that do not pass yet, per case. Strict xfail: if a
# case starts passing, the run reports XPASS as a failure and the marker has to
# be adjusted by hand (see the spec's Experimental solver governance).
_SMOKE_KNOWN_FAILURES = {
}

_SEAM_KNOWN_FAILURES = {
    "Explicit": (
        "seam gap 44.2 mm at the shipped step_h = 0.25 ms. The stitch "
        "constraint is applied (the gap is step limited, not missing): a "
        "1e5 N/m stitch on this vertex mass needs h <= ~2e-5 s, and the "
        "measured gap falls to 5.5 mm at 5e-5 s and 0.005 mm at 2e-5 s, i.e. "
        "~12x the shipped substep count. Recorded in experimental-solver-parity"
        " task 7.1"
    ),
}


def _seam_case(qydp, solver: str, params_overlay: dict | None = None):
    input_data, upper_ids, lower_ids = seamed_panels_input_data()
    driver = SimDriver(qydp, solver=solver, fps=FPS, frames=SEAM_FRAMES, dt=DT)
    # Ground off: the panels lie in the XY plane at z = 0, which is exactly the
    # ground clamp height, so with the ground on the sheet rests on it and the
    # seam never carries the lower panel. The stitch constraint has to be the
    # only thing holding it for the metric to mean anything.
    overlay = {"ground": 0.0}
    overlay.update(params_overlay or {})
    run = driver.run(input_data, params_overlay=overlay)
    gap = np.linalg.norm(
        run.world_frames[-1][upper_ids] - run.world_frames[-1][lower_ids], axis=1
    )
    return run, gap


@pytest.mark.parametrize("solver", _EXPERIMENTAL_SOLVERS)
def test_experimental_solver_smoke(qydp, solver, capture, request) -> None:
    known = _SMOKE_KNOWN_FAILURES.get(solver)
    if known:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=known))
    spec = MeshSpec(rows=10, cols=10, fixed_vertex_indices=PINNED_INDICES)
    free_indices = [i for i in range(spec.num_vertices) if i not in PINNED_INDICES]
    art = report.artifact_dir(f"sim/experimental/{solver}")
    driver = SimDriver(qydp, solver=solver, fps=FPS, frames=FRAMES, dt=DT)

    capture.begin_case(art / "sim.log")
    try:
        run = driver.run(spec.to_input_data())
    finally:
        capture.end_case()

    stats = compute_frame_stats(
        run.local_frames, run.world_frames, pinned_indices=PINNED_INDICES
    )
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    write_traces_json(art, stats)

    assert all(s["all_finite"] for s in stats), "non-finite values in experimental solver run"
    assert max(s["max_pinned_drift"] for s in stats) <= PINNED_DRIFT_TOL_M, (
        "pinned drift out of tolerance"
    )
    # The flat sheet starts at the ground clamp height, so a settled run is
    # legitimate: accept motion under gravity or a sheet resting on the ground,
    # the same arrangement as the standard smoke case.
    free_disp = np.linalg.norm(
        run.local_frames[-1][free_indices] - run.local_frames[0][free_indices], axis=1
    )
    mean_free_disp = float(np.mean(free_disp))
    ground_height_m = spec.thickness_mm * 1e-3
    resting_on_ground = bool(
        np.all(run.world_frames[-1][free_indices, 2] <= ground_height_m + GROUND_REST_TOL_M)
    )
    assert mean_free_disp > MIN_FREE_DISP_M or resting_on_ground, (
        f"mean free displacement {mean_free_disp} m below threshold and the free "
        f"vertices are not resting on the ground"
    )

    report.write_results(
        f"sim/experimental/{solver}",
        params={"solver": solver, "fps": FPS, "frames": FRAMES, "dt": DT},
        status="passed",
        artifacts={"frames_npz": str(art / "frames.npz"),
                   "traces_json": str(art / "traces.json")},
        trace_summary=stats[-1],
    )


@pytest.mark.parametrize("solver", (*_EXPERIMENTAL_SOLVERS, "PDNewton"))
def test_seam_is_held(qydp, solver, capture, request) -> None:
    """A seam whose panels are joined only by stitches must stay closed."""
    known = _SEAM_KNOWN_FAILURES.get(solver)
    if known:
        request.node.add_marker(pytest.mark.xfail(strict=True, reason=known))

    art = report.artifact_dir(f"sim/experimental/{solver}/seam")
    capture.begin_case(art / "sim.log")
    try:
        run, gap = _seam_case(qydp, solver)
    finally:
        capture.end_case()

    stats = compute_frame_stats(run.local_frames, run.world_frames)
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    write_traces_json(art, stats)
    report.write_results(
        f"sim/experimental/{solver}/seam",
        params={"solver": solver, "frames": SEAM_FRAMES, "dt": DT,
                "seam_gap_max_m": float(gap.max())},
        status="failed" if float(gap.max()) > SEAM_TOL_M else "passed",
        artifacts={"frames_npz": str(art / "frames.npz"),
                   "traces_json": str(art / "traces.json")},
        trace_summary=stats[-1],
    )
    assert bool(np.isfinite(run.world_frames).all()), "non-finite seam run"
    assert float(gap.max()) <= SEAM_TOL_M, (
        f"seam gap {float(gap.max()) * 1e3:.2f} mm exceeds "
        f"{SEAM_TOL_M * 1e3:.1f} mm: the solver is not holding the stitch constraint"
    )


def test_seam_metric_is_sensitive_to_the_stitch_stiffness(qydp, capture) -> None:
    """Control: without the stitch stiffness the same scene opens the seam.

    Guards the metric itself - a solver that never applies the constraint fails
    this case by construction, which is what makes `test_seam_is_held` meaningful.
    """
    art = report.artifact_dir("sim/experimental/PDNewton/seam_no_stitch_k")
    capture.begin_case(art / "sim.log")
    try:
        run, gap = _seam_case(qydp, "PDNewton", {"sewing_k": 0.0})
    finally:
        capture.end_case()
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    report.write_results(
        "sim/experimental/PDNewton/seam_no_stitch_k",
        params={"solver": "PDNewton", "frames": SEAM_FRAMES, "dt": DT,
                "sewing_k": 0.0, "seam_gap_max_m": float(gap.max())},
        status="passed",
        artifacts={"frames_npz": str(art / "frames.npz")},
        trace_summary={},
    )
    assert float(gap.max()) > 0.1, (
        f"seam gap {float(gap.max()) * 1e3:.2f} mm with sewing_k = 0: the metric "
        "no longer detects a missing stitch constraint"
    )


@pytest.mark.parametrize("solver", (*_EXPERIMENTAL_SOLVERS, "PDNewton"))
def test_seam_projection_merges_clusters(qydp, solver, capture) -> None:
    """The geometry-side seam projection welds a stitched seam shut.

    `Geometry::project_stitches` is solver-independent (PDNewton's path called
    it all along; VBD / XPBD / Explicit now do too). The activation frame is
    lowered so the gate opens inside the test window: once it does, the cluster
    projection puts every free stitch member on the cluster centroid, which is
    the behaviour the frontend relies on for an assembled garment.
    """
    art = report.artifact_dir(f"sim/experimental/{solver}/seam_projection")
    capture.begin_case(art / "sim.log")
    try:
        run, gap = _seam_case(
            qydp, solver, {"sewing_forced_connect_frame": 10}
        )
    finally:
        capture.end_case()
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    report.write_results(
        f"sim/experimental/{solver}/seam_projection",
        params={"solver": solver, "frames": SEAM_FRAMES, "dt": DT,
                "sewing_forced_connect_frame": 10,
                "seam_gap_max_m": float(gap.max())},
        status="passed" if float(gap.max()) <= SEAM_PROJECTION_TOL_M else "failed",
        artifacts={"frames_npz": str(art / "frames.npz")},
        trace_summary={},
    )
    assert bool(np.isfinite(run.world_frames).all()), "non-finite seam run"
    assert float(gap.max()) <= SEAM_PROJECTION_TOL_M, (
        f"seam gap {float(gap.max()) * 1e3:.3f} mm after the projection gate: the "
        "solver is not running the seam projection"
    )


def test_explicit_substep_cap_prevents_the_contact_blowup(qydp, capture) -> None:
    """A coarse requested substep must not blow the explicit solver up.

    The frontend sends one shared `step_h` (4.5 ms by default) whatever solver
    is selected. Measured on this scene with the ground contact active: the
    explicit solver moves 1.74 km with a 10 ms substep when the guard is off
    (`explicit_max_step_h = 0`) and rests at the clamp with it on, because the
    engine subdivides the frame down to the solver's declared limit.
    """
    spec = MeshSpec(rows=10, cols=10, fixed_vertex_indices=PINNED_INDICES)
    free = [i for i in range(spec.num_vertices) if i not in PINNED_INDICES]
    driver = SimDriver(qydp, solver="Explicit", fps=FPS, frames=SEAM_FRAMES, dt=DT)
    run = driver.run(spec.to_input_data(), params_overlay={"step_h": 0.01})
    assert bool(np.isfinite(run.world_frames).all()), "non-finite capped run"
    disp = np.linalg.norm(run.local_frames[-1] - run.local_frames[0], axis=1)
    assert float(disp[free].max()) < 0.1, (
        f"the explicit solver moved {float(disp[free].max()):.3f} m with a 10 ms "
        "requested substep: the substep cap is not being applied"
    )
