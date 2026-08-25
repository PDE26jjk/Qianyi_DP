"""API consistency: ``get_all_solver()`` vs the names ``set_solver()`` accepts.

``Simulator::get_all_solver`` must enumerate exactly the solver names that
``create_solver`` accepts (``PDNewton`` / ``XPBD`` / ``VBD`` / ``Explicit``).
"""

from __future__ import annotations

import pytest

from harness import report
from harness.presets import SOLVER_REGISTRY

pytestmark = [pytest.mark.api, pytest.mark.quick]

CASE_ID = "api/solver_consistency"

def test_solver_enumeration_matches_set(qydp) -> None:
    enumerated = set(qydp.simulator.get_all_solver())
    accepted = set(SOLVER_REGISTRY.keys())

    mismatch = bool(enumerated != accepted)
    failure = None
    if mismatch:
        failure = {
            "enumerated": sorted(enumerated),
            "accepted": sorted(accepted),
            "difference": sorted(enumerated.symmetric_difference(accepted)),
        }
    report.write_results(
        CASE_ID,
        status="failed" if mismatch else "passed",
        artifacts={"dir": str(report.artifact_dir(CASE_ID))},
        failure=failure,
    )
    assert enumerated == accepted, (
        f"get_all_solver()={sorted(enumerated)} does not match the set accepted "
        f"by set_solver()={sorted(accepted)}"
    )
