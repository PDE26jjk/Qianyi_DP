"""Per-case drape execution entry point.

Loads one GarmentCodeData element, runs a short simulation, evaluates the
invariant tier (and, when a body and reference drape are configured, the
reference metrics), and writes the standard per-case artifacts plus a
failure-classified ``results.json``. It is executed in a fresh subprocess by
the batch runner so a crash in one element cannot affect the rest of the
batch; the process exits 0 even for classified failures (crashes and timeouts
are detected by the batch runner instead).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from conftest import _resolve_qydp
from ..driver import SimDriver
from ..presets import apply_preset
from ..traces import compute_frame_stats, write_frames_npz
from .assertions import (
    check_invariants,
    reference_metrics,
    seam_pairs,
)
from .errors import LoaderError
from .loader import load_element


def _cloth_offsets(mesh_list: list[dict]) -> list[int]:
    offsets: list[int] = []
    total = 0
    for mesh in mesh_list:
        if mesh["object_type"] == 0:
            offsets.append(total)
            total += len(mesh["vertices"]) // 3
    return offsets


def _attached_global_indices(mesh_list: list[dict]) -> np.ndarray:
    offsets = _cloth_offsets(mesh_list)
    indices: list[int] = []
    for mesh_index, mesh in enumerate(mesh_list):
        if mesh["object_type"] != 0:
            continue
        weights = np.asarray(mesh["attached_vertices"], dtype=np.float32)
        indices.extend(
            int(offsets[mesh_index]) + i for i in range(len(weights)) if weights[i] > 0.5
        )
    return np.asarray(indices, dtype=np.int64)


def _write_results_flat(
    artifact_dir: Path,
    case_id: str,
    *,
    params: dict,
    status: str,
    artifacts: dict,
    trace_summary: dict,
    failure: dict | None = None,
) -> Path:
    """Write results.json directly inside ``artifact_dir`` (flat layout)."""
    payload: dict = {
        "test_id": case_id,
        "params": params,
        "status": status,
        "artifacts": artifacts,
        "trace_summary": trace_summary,
    }
    if failure is not None:
        payload["failure"] = failure
    out_path = Path(artifact_dir) / "results.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def run_case(
    element_dir: Path,
    artifact_dir: Path,
    *,
    body_obj: Path | None = None,
    frames: int = 30,
    fps: int = 24,
    gravity: float | None = None,
    run_reference: bool = True,
    hard_fail: bool = False,
    solver: str = "PDNewton",
) -> dict:
    """Run one element and return its case result summary."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    case_id = Path(element_dir).name
    # Design D7: the no-body tier runs with gravity disabled.
    if body_obj is None and gravity is None:
        gravity = 0.0

    # Phase 1: loader.
    try:
        element = load_element(Path(element_dir), body_obj)
    except LoaderError as exc:
        _write_results_flat(
            artifact_dir,
            case_id,
            params={"element": case_id, "frames": frames, "fps": fps},
            status="failed",
            artifacts={"dir": str(artifact_dir)},
            trace_summary={},
            failure={"class": "loader error", "message": str(exc)},
        )
        return {"status": "failed", "failure_class": "loader error"}

    loader_report_path = artifact_dir / "loader_report.json"
    loader_report_path.write_text(
        json.dumps(element.report, indent=2), encoding="utf-8"
    )

    # Phase 2: simulation.
    qydp = _resolve_qydp()
    sim = qydp.simulator
    params = apply_preset(sim, solver)
    if gravity is not None:
        params = dict(params)
        params["gravity"] = gravity
        sim.set_parameter("gravity", float(gravity))

    driver = SimDriver(qydp, fps=fps, frames=frames, dt=0.001)
    wall_start = time.perf_counter()
    run = driver.run(element.input_data, gravity=gravity)
    wall_time = time.perf_counter() - wall_start

    # Phase 3: analysis (local space; see assertions module docstring).
    cloth = [m for m in element.input_data["mesh_list"] if m["object_type"] == 0]
    initial = np.concatenate([m["vertices"] for m in cloth]).reshape(-1, 3)
    attached = _attached_global_indices(element.input_data["mesh_list"])
    invariant = check_invariants(
        run.local_frames,
        initial,
        element.input_data["mesh_list"],
        element.input_data["sewings"],
        attached_vertices=attached,
    )

    metrics: dict | None = None
    if (
        run_reference
        and body_obj is not None
        and element.reference_vertices is not None
        and element.reference_faces is not None
    ):
        metrics = reference_metrics(
            run.local_frames[-1],
            element.input_data["mesh_list"],
            element.panel_triangle_assignment,
            element.reference_vertices,
            element.reference_faces,
            hard_fail=hard_fail,
        )

    substeps = frames * driver.substeps_per_frame()
    perf = {
        "wall_time_s": float(wall_time),
        "substeps": int(substeps),
        "per_substep_s": float(wall_time / max(substeps, 1)),
        "faces": int(element.report["faces"]),
        "size_bucket": (
            "S" if element.report["faces"] < 5000
            else "M" if element.report["faces"] < 25000
            else "L" if element.report["faces"] < 50000
            else "XL"
        ),
    }

    failure_class: str | None = None
    failure_detail: list[str] = []
    if not invariant.passed:
        failure_class = "simulation blow-up"
        failure_detail = invariant.failures
    elif metrics is not None and metrics.get("passed") is False:
        failure_class = "reference mismatch"
        failure_detail = metrics.get("failures", [])

    frames_path = write_frames_npz(
        artifact_dir, run.local_frames, run.world_frames, run.timestamps
    )
    trace_stats = compute_frame_stats(run.local_frames, run.world_frames)
    (artifact_dir / "traces.json").write_text(
        json.dumps(trace_stats, indent=2), encoding="utf-8"
    )
    (artifact_dir / "sim.log").write_text(
        "C-level output is captured by the batch runner (fd-level); "
        "per-case log lives next to this file.\n",
        encoding="utf-8",
    )

    trace_summary = {
        "frames": len(trace_stats),
        "last_frame": trace_stats[-1],
        "all_finite": bool(np.isfinite(run.local_frames).all()),
    }
    payload_artifacts = {
        "dir": str(artifact_dir),
        "frames": str(frames_path.name),
        "loader_report": "loader_report.json",
    }
    payload_params = {
        "element": case_id,
        "frames": frames,
        "fps": fps,
        "solver": solver,
        "gravity": gravity if gravity is not None else params.get("gravity"),
        "invariant_stats": invariant.stats,
        "reference_metrics": metrics,
        "perf": perf,
    }
    if failure_class is not None:
        _write_results_flat(
            artifact_dir,
            case_id,
            params=payload_params,
            status="failed",
            artifacts=payload_artifacts,
            trace_summary=trace_summary,
            failure={"class": failure_class, "failures": failure_detail},
        )
    else:
        _write_results_flat(
            artifact_dir,
            case_id,
            params=payload_params,
            status="passed",
            artifacts=payload_artifacts,
            trace_summary=trace_summary,
        )

    return {
        "status": "failed" if failure_class else "passed",
        "failure_class": failure_class,
        "wall_time_s": perf["wall_time_s"],
        "size_bucket": perf["size_bucket"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one GCD drape case")
    parser.add_argument("--element-dir", required=True, type=Path)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--body-obj", type=Path, default=None)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gravity", type=float, default=None)
    parser.add_argument("--no-reference", action="store_true")
    parser.add_argument("--hard-fail", action="store_true")
    parser.add_argument("--solver", default="PDNewton")
    args = parser.parse_args(argv)
    run_case(
        args.element_dir,
        args.artifact_dir,
        body_obj=args.body_obj,
        frames=args.frames,
        fps=args.fps,
        gravity=args.gravity,
        run_reference=not args.no_reference,
        hard_fail=args.hard_fail,
        solver=args.solver,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
