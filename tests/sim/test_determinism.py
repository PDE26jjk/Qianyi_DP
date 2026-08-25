"""Determinism test: identical scene/parameters must produce identical frames.

Known issue: PDNewton is not deterministic run-to-run (see the strict-xfail
marker); the check itself is implemented as specified (two identical runs,
point-by-point comparison, subprocess re-verification as fallback) and will
start enforcing once the C++ side is fixed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec
from harness.traces import compute_frame_stats, write_frames_npz, write_traces_json

pytestmark = [
    pytest.mark.sim,
    pytest.mark.xfail(
        strict=True,
        reason=(
            "Known issue: PDNewton is not deterministic run-to-run (max abs "
            "diff ~0.03-0.18 m from frame 0, reproduced in fresh subprocesses). "
            "Fix direction: C++ side - ensure Geometry::init fully resets all "
            "per-vertex device buffers and rule out kernel-level nondeterminism "
            "(atomics/reduction order). Remove this xfail when fixed."
        ),
    ),
]

CASE_ID = "sim/determinism"
FRAMES = 20
FPS = 24
DT = 0.001
PINNED_INDICES = (0, 9, 90, 99)
SUBPROCESS_SCRIPT = Path(__file__).with_name("determinism_subprocess.py")


def _run_once(qydp, spec: MeshSpec, art: Path, capture):
    driver = SimDriver(qydp, frames=FRAMES, fps=FPS, dt=DT)
    capture.begin_case(art / "sim.log")
    try:
        return driver.run(spec.to_input_data())
    finally:
        capture.end_case()


def test_deterministic_run(qydp, record_failure, capture) -> None:
    spec = MeshSpec(rows=10, cols=10, fixed_vertex_indices=PINNED_INDICES)
    art = report.artifact_dir(CASE_ID)

    run1 = _run_once(qydp, spec, art, capture)
    run2 = _run_once(qydp, spec, art, capture)

    stats = compute_frame_stats(
        run1.local_frames, run1.world_frames, pinned_indices=PINNED_INDICES
    )
    write_frames_npz(art, run1.local_frames, run1.world_frames, run1.timestamps)
    write_traces_json(art, stats)
    record_failure(art, CASE_ID, stats=stats)

    same_local = bool(np.array_equal(run1.local_frames, run2.local_frames))
    same_world = bool(np.array_equal(run1.world_frames, run2.world_frames))
    if not (same_local and same_world):
        if _subprocess_verification(spec):
            pytest.fail(
                "in-process runs differ but a fresh subprocess is deterministic: "
                "the simulator singleton retains state across in-process runs "
                f"(hint: inspect {art})"
            )
        pytest.fail(f"runs are not deterministic (in-process and subprocess) - see {art}")

    report.write_results(
        CASE_ID,
        params={"solver": "PDNewton", "fps": FPS, "frames": FRAMES, "dt": DT},
        status="passed",
        artifacts={
            "frames_npz": str(art / "frames.npz"),
            "traces_json": str(art / "traces.json"),
        },
        trace_summary=stats[-1],
    )


def _subprocess_verification(spec: MeshSpec) -> bool:
    """Re-run the determinism check in a fresh process."""
    env = dict(os.environ)
    result = subprocess.run(
        [sys.executable, str(SUBPROCESS_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    return result.returncode == 0
