"""Experimental solver governance (VBD / XPBD / Explicit).

Deferred by the maintainer (2026-08-24): only PDNewton is tested in this
phase. The original ``xfail(strict=True)`` smoke is kept (skipped) so that
when experimental solvers are re-enabled the strict-xfail semantics surface
state flips explicitly instead of passing silently (see spec: Experimental
solver governance).
"""

from __future__ import annotations

import numpy as np
import pytest

from harness import report
from harness.driver import SimDriver
from harness.meshspec import MeshSpec
from harness.traces import compute_frame_stats

pytestmark = pytest.mark.sim

FRAMES = 60
FPS = 24
DT = 0.001
PINNED_INDICES = (0, 9, 90, 99)
MIN_FREE_DISP_M = 0.05
PINNED_DRIFT_TOL_M = 1e-3

_EXPERIMENTAL_SOLVERS = ("VBD", "XPBD", "Explicit")

_DEFERRED_REASON = (
    "Deferred by maintainer: only PDNewton is tested this phase; the "
    "experimental solvers need re-validation after the driver order fix "
    "(set_solver before input_data). See design.md."
)


@pytest.mark.parametrize("solver", _EXPERIMENTAL_SOLVERS)
@pytest.mark.skip(reason=_DEFERRED_REASON)
def test_experimental_solver_smoke(qydp, solver, capture) -> None:
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
    assert all(s["all_finite"] for s in stats), "non-finite values in experimental solver run"
    assert max(s["max_pinned_drift"] for s in stats) <= PINNED_DRIFT_TOL_M, (
        "pinned drift out of tolerance"
    )
    free_disp = np.linalg.norm(
        run.local_frames[-1][free_indices] - run.local_frames[0][free_indices], axis=1
    )
    assert float(np.mean(free_disp)) > MIN_FREE_DISP_M, (
        "experimental solver smoke failed to move the cloth"
    )
