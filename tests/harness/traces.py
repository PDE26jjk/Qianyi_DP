"""Per-frame data trace writing and statistics for simulation tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_frames_npz(
    artifact_dir: Path,
    local_frames: np.ndarray,
    world_frames: np.ndarray,
    timestamps: np.ndarray,
) -> Path:
    """Write per-frame local/world vertices and timestamps to frames.npz."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "frames.npz"
    np.savez_compressed(
        out_path,
        local=local_frames,
        world=world_frames,
        timestamps=timestamps,
    )
    return out_path


def compute_frame_stats(
    local_frames: np.ndarray,
    world_frames: np.ndarray,
    pinned_indices: tuple[int, ...] = (),
) -> list[dict]:
    """Per-frame stats: max displacement, finiteness, pinned drift, z-range."""
    initial_local = local_frames[0]
    initial_world = world_frames[0]
    stats: list[dict] = []
    for frame_index, (local, world) in enumerate(zip(local_frames, world_frames)):
        displacement = np.linalg.norm(local - initial_local, axis=1)
        pinned = (
            np.linalg.norm(world[list(pinned_indices)] - initial_world[list(pinned_indices)], axis=1)
            if pinned_indices
            else np.zeros(0)
        )
        non_finite = int(np.count_nonzero(~np.isfinite(local)))
        stats.append(
            {
                "frame": frame_index,
                "max_disp": float(np.max(displacement)) if displacement.size else 0.0,
                "mean_disp": float(np.mean(displacement)) if displacement.size else 0.0,
                "non_finite_count": non_finite,
                "all_finite": bool(non_finite == 0),
                "max_pinned_drift": float(np.max(pinned)) if pinned.size else 0.0,
                "z_min": float(np.min(local[:, 2])) if local.size else 0.0,
                "z_max": float(np.max(local[:, 2])) if local.size else 0.0,
                "world_max_abs": float(np.max(np.abs(world))) if world.size else 0.0,
            }
        )
    return stats


def write_traces_json(artifact_dir: Path, stats: list[dict]) -> Path:
    """Write per-frame statistics to traces.json."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifact_dir / "traces.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"frames": stats}, handle, indent=2)
    return out_path

