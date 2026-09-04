"""End-to-end data-driven drape batch tests (real dataset, small manifest)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.gcd.batch import run_batch

pytestmark = [pytest.mark.data, pytest.mark.drape, pytest.mark.slow]

# Small deterministic manifest: one tiny garment and one small torso, so the
# default run stays in the minute range (calibrated in task 3.3; XL excluded).
SMALL_MANIFEST = [
    {"id": "rand_07JYAR0G1P", "face_count": 1470, "bucket": "S"},
    {"id": "rand_1TDTWJLQZ9", "face_count": 2080, "bucket": "S"},
]


def test_small_drape_batch(qydp, qydp_data, scratch_dir: Path) -> None:
    artifact_root = scratch_dir / "batch"
    summary = run_batch(
        qydp_data.root,
        SMALL_MANIFEST,
        artifact_root,
        body_obj=qydp_data.body_obj,
        frames=5,
        timeout=300,
    )

    assert summary["total"] == len(SMALL_MANIFEST)
    assert (artifact_root / "batch_summary.json").is_file()
    for entry in SMALL_MANIFEST:
        case_dir = artifact_root / entry["id"]
        results_path = case_dir / "results.json"
        assert results_path.is_file(), f"missing results.json for {entry['id']}"
        results = json.loads(results_path.read_text(encoding="utf-8"))
        assert results["test_id"] == entry["id"]
        assert results["status"] in {"passed", "failed"}
        assert (case_dir / "frames.npz").is_file()
        assert (case_dir / "traces.json").is_file()
        assert (case_dir / "loader_report.json").is_file()
        # Failure classes must be from the documented taxonomy.
        failure = results.get("failure") or {}
        assert failure.get("class") in {
            None,
            "loader error",
            "simulation blow-up",
            "reference mismatch",
            "timeout",
        }


def test_no_body_tier_disables_gravity(qydp_data, scratch_dir: Path) -> None:
    """The no-body tier runs with gravity disabled (design D7)."""
    from harness.gcd.run_case import run_case

    result = run_case(
        qydp_data.root / "rand_07JYAR0G1P",
        scratch_dir / "nobody",
        body_obj=None,
        frames=3,
        run_reference=False,
    )
    results_path = scratch_dir / "nobody" / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8"))
    assert results["params"]["gravity"] == 0.0
    assert results["status"] == result["status"]
