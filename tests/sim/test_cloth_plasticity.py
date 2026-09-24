"""Cloth plasticity: the authored rest shape and the plastic bending model.

Two cases on procedural scenes, driven through the standard harness:

1. the per-edge rest-shape input (`angles` / `compress`) reaches both planar
   models, the pattern-derived quantities (vertex mass) stay put, and a panel
   that did not opt in reports an empty plastic state;
2. a panel that did opt in develops a plastic rest-angle change while it hangs
   pinned under gravity, `freeze_rest_shape` commits the current bend angle into
   the rest state without moving a vertex (and repeatably), and
   `reset_plasticity` returns to the input rest shape.

The harness preset selects the IBM bending model, which has no rest angle, so
the cases overlay AOGS - the model the frontend's panel drives. The visual
"frozen garment stays wrinkled" scenario is not asserted here: on a procedural
sheet the recovery after the load is removed is membrane-dominated while the
freeze only changes the bending rest angle, so the check would measure the
membrane. The state-level contract is what this case pins down.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec
from harness.traces import compute_frame_stats, write_frames_npz, write_traces_json

pytestmark = [pytest.mark.sim, pytest.mark.quick]

FPS = 24
DT = 1.0 / FPS

REST_CASE = "sim/plasticity-rest-shape"
STATE_CASE = "sim/plasticity-state"

# The paper's garment-scale yield angles sit far above what a 0.4 m procedural
# sample folds to, so the state case uses a yield this sample can reach.
REACHABLE_YIELD_RAD = 0.05
HOLD_TIME_SCALE = 60.0
HOLD_FRAMES = 180
ARC_RADIUS_M = 0.2
LIFT_M = 0.9


def _grid(cols: int = 20, rows: int = 20, width: float = 0.4,
          height: float = 0.4, fixed: tuple[int, ...] = ()) -> MeshSpec:
    return MeshSpec(width_m=width, height_m=height, cols=cols, rows=rows,
                    fixed_vertex_indices=fixed)


def _with_edges(data: dict, angles: float | None = None,
                compress: float | None = None, plastic: bool = False) -> dict:
    """Attach the per-edge rest-shape arrays to a single-panel scene."""
    mesh = data["mesh_list"][0]
    n_edges = len(mesh["edges"]) // 2  # `edges` is the flat int array
    mesh["angles"] = np.full(n_edges, angles if angles is not None else 0.0,
                             np.float32)
    mesh["compress"] = np.full(n_edges, compress if compress is not None else 0.0,
                               np.float32)
    mesh["plastic"] = plastic
    return data


def _place(data: dict, z) -> dict:
    """Place the panel: `z` is either a height or a (N,) height field."""
    mesh = data["mesh_list"][0]
    moved = mesh["vertices"].reshape(-1, 3).copy()
    moved[:, 2] = z if np.isscalar(z) else np.asarray(z, np.float32)
    mesh["vertices"] = moved.reshape(-1).copy()
    mesh["vertices_sim"] = moved.reshape(-1).copy()
    return data


def _lift(data: dict, dz: float) -> dict:
    """Raise the panel so the case is about bending, not about the ground."""
    mesh = data["mesh_list"][0]
    moved = mesh["vertices"].reshape(-1, 3) + np.array([0.0, 0.0, dz], np.float32)
    mesh["vertices"] = moved.reshape(-1).copy()
    mesh["vertices_sim"] = moved.reshape(-1).copy()
    return data


def _arc(data: dict) -> dict:
    """An isometric arc: a curved placement with a flat rest shape."""
    mesh = data["mesh_list"][0]
    flat = mesh["vertices"].reshape(-1, 3).copy()
    angle = flat[:, 0] / ARC_RADIUS_M
    curved = flat.copy()
    curved[:, 0] = ARC_RADIUS_M * np.sin(angle)
    curved[:, 2] = ARC_RADIUS_M * (1.0 - np.cos(angle)) + 0.3
    mesh["vertices"] = curved.reshape(-1).copy()
    mesh["vertices_sim"] = curved.reshape(-1).copy()
    return data


def _mean_edge_length(vertices: np.ndarray, edges: np.ndarray) -> float:
    pairs = np.asarray(edges).reshape(-1, 2)
    return float(np.mean(np.linalg.norm(vertices[pairs[:, 0]] - vertices[pairs[:, 1]],
                                        axis=1)))


def test_rest_shape_input(qydp, record_failure, capture) -> None:
    spec = _grid(fixed=tuple(range(20)))
    edges = spec.to_input_data()["mesh_list"][0]["edges"]
    cases = [
        ("plain", _lift(_with_edges(spec.to_input_data()), 0.6), None),
        ("shrink", _lift(_with_edges(spec.to_input_data(), compress=-0.35), 0.6),
         None),
        ("shrink_fem",
         _lift(_with_edges(spec.to_input_data(), compress=-0.35), 0.6),
         {"constitutive_model_planar": 1}),
    ]
    art = report.artifact_dir(REST_CASE)
    driver = SimDriver(qydp, fps=FPS, frames=30, dt=DT)
    runs = {}
    capture.begin_case(art / "sim.log")
    try:
        for name, data, overlay in cases:
            runs[name] = driver.run(data, params_overlay=overlay)
        # Vertex mass comes from the pattern area, so it must not follow the
        # compress input (only init runs; no stepping needed to read it).
        qydp.simulator.input_data(cases[0][1])
        mass_plain = qydp.simulator.check_point_attributes(240)["mass"]
        qydp.simulator.input_data(cases[1][1])
        mass_shrink = qydp.simulator.check_point_attributes(240)["mass"]
    finally:
        capture.end_case()

    stats = compute_frame_stats(runs["plain"].local_frames,
                               runs["plain"].world_frames)
    write_frames_npz(art, runs["plain"].local_frames, runs["plain"].world_frames,
                     runs["plain"].timestamps)
    write_traces_json(art, stats)
    record_failure(art, REST_CASE, stats=stats)

    lengths = {name: _mean_edge_length(run.local_frames[-1], edges)
               for name, run in runs.items()}
    for name, run in runs.items():
        assert np.isfinite(run.world_frames).all(), f"{name} produced non-finite data"
    assert lengths["shrink"] < lengths["plain"] * 0.95, (
        f"the shrunk rest length did not reach the spring-mass model: "
        f"{lengths['shrink']:.4f} m against {lengths['plain']:.4f} m")
    assert lengths["shrink_fem"] < lengths["plain"] * 0.95, (
        f"the shrunk rest length did not reach the FEM rest metric: "
        f"{lengths['shrink_fem']:.4f} m against {lengths['plain']:.4f} m")
    assert abs(mass_plain - mass_shrink) < 1e-9, (
        f"compress changed the pattern mass ({mass_plain} vs {mass_shrink})")

    # No panel set `plastic`, so nothing may accumulate plastic state.
    state = qydp.simulator.get_plasticity_state()
    assert state.ndim == 2 and state.shape[1] == 5, (
        f"unexpected plasticity state layout {state.shape}")
    assert float(np.max(np.abs(state[:, 0]))) == 0.0, (
        "an unflagged panel changed its rest angle")
    assert float(np.max(state[:, 3:5])) == 0.0, (
        "an unflagged panel accumulated time in the plastic timers")

    report.write_results(
        REST_CASE,
        params={"solver": "PDNewton", "fps": FPS, "frames": 30, "dt": DT},
        status="passed",
        artifacts={"frames_npz": str(art / "frames.npz"),
                   "traces_json": str(art / "traces.json")},
        trace_summary=stats[-1],
    )


def test_plastic_state_freeze_and_reset(qydp, record_failure, capture) -> None:
    # A pinned curtain lifted well above the ground: gravity folds it, contact
    # plays no part, and the plastic model runs with a reachable yield.
    data = _lift(_with_edges(_grid(fixed=tuple(range(20))).to_input_data(),
                             plastic=True), LIFT_M)
    overlay = {"bending_model": 2, "plastic_bend_yield": REACHABLE_YIELD_RAD,
               "plasticity_time_scale": HOLD_TIME_SCALE}
    # The freeze case needs a configuration that already bends while the rest
    # shape is still flat; the plastic model is inert there so the freeze is the
    # only thing changing the rest state.
    arc = _arc(_with_edges(_grid().to_input_data(), plastic=True))
    freeze_overlay = {"bending_model": 2, "plastic_bend_yield": 10.0,
                      "plasticity_time_scale": 0.0, "gravity": 0.0}

    art = report.artifact_dir(STATE_CASE)
    driver = SimDriver(qydp, fps=FPS, frames=HOLD_FRAMES, dt=DT)
    capture.begin_case(art / "sim.log")
    try:
        run = driver.run(data, params_overlay=overlay)
        state = np.array(qydp.simulator.get_plasticity_state(), copy=True)
        qydp.simulator.reset_plasticity()
        reset = np.array(qydp.simulator.get_plasticity_state(), copy=True)

        SimDriver(qydp, fps=FPS, frames=1, dt=DT).run(
            arc, params_overlay=freeze_overlay)
        before_freeze = np.array(qydp.simulator.get_plasticity_state(), copy=True)
        before_shape = np.array(qydp.simulator.get_simulation_data(True), copy=True)
        qydp.simulator.freeze_rest_shape()
        frozen_state = np.array(qydp.simulator.get_plasticity_state(), copy=True)
        after_shape = np.array(qydp.simulator.get_simulation_data(True), copy=True)
        qydp.simulator.freeze_rest_shape()
        repeated_state = np.array(qydp.simulator.get_plasticity_state(), copy=True)
    finally:
        capture.end_case()

    stats = compute_frame_stats(run.local_frames, run.world_frames)
    write_frames_npz(art, run.local_frames, run.world_frames, run.timestamps)
    write_traces_json(art, stats)
    record_failure(art, STATE_CASE, stats=stats)

    assert np.isfinite(run.world_frames).all(), "the held run produced non-finite data"
    rest_drift = float(np.max(np.abs(state[:, 0])))
    assert rest_drift > 0.01, (
        f"the flagged panel barely changed its rest angle ({rest_drift:.5f} rad)")
    assert float(np.max(state[:, 3])) > 1.0, (
        "the stick timer did not accumulate simulated time")

    assert float(np.max(np.abs(before_freeze[:, 0]))) == 0.0, (
        "the inert model changed the rest angle before the freeze")
    committed = float(np.max(np.abs(frozen_state[:, 0])))
    assert committed > 0.05, (
        f"freeze did not commit the current bend angle into the rest state "
        f"({committed:.5f} rad)")
    freeze_move = float(np.max(np.abs(after_shape - before_shape)))
    assert freeze_move < 1e-6, (
        f"freeze moved vertices on the freeze frame ({freeze_move:.2e} m)")
    assert np.array_equal(frozen_state, repeated_state), (
        "a second freeze with no step in between was not a no-op")

    assert float(np.max(np.abs(reset[:, 0]))) == 0.0, (
        "reset did not restore the input rest shape")

    report.write_results(
        STATE_CASE,
        params={"solver": "PDNewton", "fps": FPS, "frames": HOLD_FRAMES, "dt": DT,
                **overlay},
        status="passed",
        artifacts={"frames_npz": str(art / "frames.npz"),
                   "traces_json": str(art / "traces.json")},
        trace_summary=stats[-1],
    )
