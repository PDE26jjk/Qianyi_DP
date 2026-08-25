"""Per-case machine-readable result writing (results.json)."""

from __future__ import annotations

import json
from pathlib import Path


def default_artifact_root() -> Path:
    """Default artifact root: ``<repo>/tests/artifacts`` (gitignored)."""
    return Path(__file__).resolve().parents[2] / "tests" / "artifacts"


def artifact_dir(test_id: str, root: str | Path | None = None) -> Path:
    """Return the artifact directory for a test case."""
    root = Path(root) if root is not None else default_artifact_root()
    return root / test_id


def write_results(
    test_id: str,
    *,
    params: dict | None = None,
    status: str = "passed",
    artifacts: dict | None = None,
    trace_summary: dict | None = None,
    failure: dict | None = None,
    root: str | Path | None = None,
) -> Path:
    """Write (or overwrite) the per-case results.json file."""
    out_dir = artifact_dir(test_id, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "test_id": test_id,
        "params": params or {},
        "status": status,
        "artifacts": artifacts or {},
        "trace_summary": trace_summary,
    }
    if failure is not None:
        payload["failure"] = failure
    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return out_path

