"""Isolated batch execution of drape cases."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _case_command(
    element_id: str,
    root: Path,
    artifact_dir: Path,
    *,
    body_obj: Path | None,
    frames: int,
    fps: int,
    gravity: float | None,
    run_reference: bool,
    hard_fail: bool,
    solver: str,
) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "harness.gcd.run_case",
        "--element-dir",
        str(root / element_id),
        "--artifact-dir",
        str(artifact_dir),
        "--frames",
        str(frames),
        "--fps",
        str(fps),
    ]
    if body_obj is not None:
        cmd += ["--body-obj", str(body_obj)]
    if gravity is not None:
        cmd += ["--gravity", str(gravity)]
    if not run_reference:
        cmd += ["--no-reference"]
    if hard_fail:
        cmd += ["--hard-fail"]
    cmd += ["--solver", solver]
    return cmd


def _run_one(
    element_id: str,
    root: Path,
    artifact_dir: Path,
    *,
    body_obj: Path | None,
    frames: int,
    fps: int,
    gravity: float | None,
    run_reference: bool,
    hard_fail: bool,
    solver: str,
    timeout: float,
) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "sim.log"
    env = os.environ.copy()
    tests_dir = REPO_ROOT / "tests"
    env["PYTHONPATH"] = str(tests_dir) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = _case_command(
        element_id,
        root,
        artifact_dir,
        body_obj=body_obj,
        frames=frames,
        fps=fps,
        gravity=gravity,
        run_reference=run_reference,
        hard_fail=hard_fail,
        solver=solver,
    )
    with open(log_path, "wb") as log_handle:
        try:
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "id": element_id,
                "status": "failed",
                "failure_class": "timeout",
            }
    results_path = artifact_dir / "results.json"
    if not results_path.is_file():
        return {
            "id": element_id,
            "status": "failed",
            "failure_class": "crash",
            "returncode": result.returncode,
        }
    results = json.loads(results_path.read_text(encoding="utf-8"))
    failure = results.get("failure") or {}
    return {
        "id": element_id,
        "status": results.get("status", "unknown"),
        "failure_class": failure.get("class"),
        "returncode": result.returncode,
    }


def run_batch(
    root: Path,
    manifest: list[dict],
    artifact_root: Path,
    *,
    body_obj: Path | None = None,
    frames: int = 30,
    fps: int = 24,
    gravity: float | None = None,
    run_reference: bool = True,
    hard_fail: bool = False,
    solver: str = "PDNewton",
    timeout: float = 600.0,
) -> dict:
    """Run every manifest element in an isolated subprocess."""
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    cases: list[dict] = []
    for entry in manifest:
        element_id = entry["id"]
        cases.append(
            _run_one(
                element_id,
                root,
                artifact_root / element_id,
                body_obj=body_obj,
                frames=frames,
                fps=fps,
                gravity=gravity,
                run_reference=run_reference,
                hard_fail=hard_fail,
                solver=solver,
                timeout=timeout,
            )
        )
    summary = {
        "total": len(cases),
        "passed": sum(1 for c in cases if c["status"] == "passed"),
        "failed": sum(1 for c in cases if c["status"] == "failed"),
        "failure_classes": {
            name: sum(1 for c in cases if c["failure_class"] == name)
            for name in ("loader error", "simulation blow-up",
                         "reference mismatch", "timeout", "crash")
        },
        "cases": cases,
        "config": {
            "frames": frames,
            "fps": fps,
            "gravity": gravity,
            "run_reference": run_reference,
            "hard_fail": hard_fail,
            "solver": solver,
            "timeout_s": timeout,
        },
    }
    (artifact_root / "batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
