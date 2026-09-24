"""Frame-stage timing: the flat per-frame readback of `profile_timing`.

The timer brackets four stages of one `Simulator::update` call with CUDA events
and resolves them on demand. This case pins the contract: the key set, the
disabled baseline, one sample per frame, and that switching the timer on does
not change the simulation.
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec

pytestmark = [pytest.mark.api, pytest.mark.quick]

CASE_ID = "api/frame_timing"
FPS = 24
FRAMES = 20
DT = 1.0 / FPS
STAGE_KEYS = ("frame_update", "collision", "substeps", "end_frame")
CONTRACT_KEYS = {"frame", "enabled", "stale", "total", *STAGE_KEYS}
PINNED_INDICES = (0, 9, 90, 99)
# PDNewton is not bitwise deterministic run-to-run (the `sim/determinism`
# xfail), so the non-interference check compares within a tolerance instead of
# demanding identical frames.
NON_INTERFERENCE_TOL_M = 1e-3


def _scene() -> dict:
    """The standard corner-pinned sheet, lifted off the ground plane."""
    spec = MeshSpec(rows=10, cols=10, fixed_vertex_indices=PINNED_INDICES)
    data = spec.to_input_data()
    mesh = data["mesh_list"][0]
    lifted = mesh["vertices"].reshape(-1, 3) + np.array([0.0, 0.0, 1.5], np.float32)
    mesh["vertices"] = lifted.reshape(-1).copy()
    mesh["vertices_sim"] = lifted.reshape(-1).copy()
    return data


def test_frame_timing_contract(qydp, capture) -> None:
    driver = SimDriver(qydp, fps=FPS, frames=FRAMES, dt=DT)
    art = report.artifact_dir(CASE_ID)
    capture.begin_case(art / "sim.log")
    try:
        # Disabled: nothing is recorded, and the readback says so.
        driver.run(_scene())
        off = dict(qydp.simulator.get_timing())

        # Enabled: the frame that was just run, with a stage breakdown.
        driver.run(_scene(), params_overlay={"profile_timing": 1})
        on = dict(qydp.simulator.get_timing())
        repeat = dict(qydp.simulator.get_timing())

        # Turned off again: the readback must not keep reporting the sample from
        # the enabled stretch as if it were live. (Parameters are sticky for the
        # life of the simulator, so switching off means setting the key to 0.)
        driver.run(_scene(), params_overlay={"profile_timing": 0})
        after_off = dict(qydp.simulator.get_timing())
    finally:
        capture.end_case()

    report.write_results(
        CASE_ID,
        params={"solver": "PDNewton", "fps": FPS, "frames": FRAMES, "dt": DT},
        status="passed",
        artifacts={"dir": str(art)},
        trace_summary={"disabled": off, "enabled": on},
    )

    assert set(off) == CONTRACT_KEYS, f"unexpected keys {sorted(off)}"
    assert off["enabled"] is False, "timing is on without profile_timing"
    assert off["frame"] == -1, f"a disabled timer reported frame {off['frame']}"
    assert all(off[key] == 0.0 for key in ("total", *STAGE_KEYS)), (
        f"a disabled timer reported durations: {off}")

    assert set(on) == CONTRACT_KEYS, f"unexpected keys {sorted(on)}"
    assert on["enabled"] is True, "profile_timing did not enable the timer"
    assert on["stale"] is False, f"the sample lagged behind: {on}"
    assert on["frame"] >= FRAMES - 2, (
        f"the sample belongs to frame {on['frame']}, not the run that ended at "
        f"frame {FRAMES - 1}")
    for key in ("total", *STAGE_KEYS):
        assert np.isfinite(on[key]) and on[key] >= 0.0, f"{key} = {on[key]}"
    stages = sum(on[key] for key in STAGE_KEYS)
    assert abs(stages - on["total"]) <= 1e-3, (
        f"the stages ({stages:.4f} ms) do not add up to the total "
        f"({on['total']:.4f} ms)")
    assert on["substeps"] > 0.0, "the substep loop was not measured"
    assert on["collision"] > 0.0, (
        "the per-substep contact work was not measured")

    assert repeat == on, "a second readback without a step changed the sample"

    assert after_off["enabled"] is False, (
        f"the timer still reports itself on: {after_off}")
    assert after_off["frame"] == -1, (
        f"a disabled timer reported frame {after_off['frame']}")
    assert all(after_off[key] == 0.0 for key in ("total", *STAGE_KEYS)), (
        f"a disabled timer reported durations: {after_off}")


def test_frame_timing_does_not_change_the_simulation(qydp, capture) -> None:
    driver = SimDriver(qydp, fps=FPS, frames=FRAMES, dt=DT)
    art = report.artifact_dir(f"{CASE_ID}_interference")
    capture.begin_case(art / "sim.log")
    try:
        without = driver.run(_scene())
        with_timing = driver.run(_scene(), params_overlay={"profile_timing": 1})
    finally:
        capture.end_case()

    drift = float(np.max(np.abs(with_timing.world_frames - without.world_frames)))
    report.write_results(
        f"{CASE_ID}_interference",
        params={"solver": "PDNewton", "fps": FPS, "frames": FRAMES, "dt": DT},
        status="passed",
        artifacts={"dir": str(art)},
        trace_summary={"max_frame_delta_m": drift},
    )
    assert np.isfinite(with_timing.world_frames).all(), "the timed run produced non-finite data"
    assert drift <= NON_INTERFERENCE_TOL_M, (
        f"enabling the timer changed the frame data by {drift:.5f} m")
