"""Subprocess re-verification for the determinism test.

Runs the same scene and parameters twice in a fresh process and reports
whether the frames match exactly. Used only when the in-process determinism
check fails, to distinguish real nondeterminism from simulator singleton
state residue.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tests"))

from harness.driver import SimDriver
from harness.meshspec import MeshSpec
from harness.output_redirect import FdOutputRedirect
from harness.traces import compute_frame_stats, write_frames_npz, write_traces_json

FRAMES = 20
FPS = 24
DT = 0.001
PINNED_INDICES = (0, 9, 90, 99)


def main() -> int:
    spec = MeshSpec(rows=10, cols=10, fixed_vertex_indices=PINNED_INDICES)
    art = REPO_ROOT / "tests" / "artifacts" / "sim" / "determinism_subprocess"
    from conftest import _resolve_qydp

    qydp = _resolve_qydp()
    if qydp is None:
        print("Qianyi_DP not found")
        return 2
    driver = SimDriver(qydp, frames=FRAMES, fps=FPS, dt=DT)
    with FdOutputRedirect(art / "sim.log", verbose=False):
        run1 = driver.run(spec.to_input_data())
        run2 = driver.run(spec.to_input_data())
    stats = compute_frame_stats(
        run1.local_frames, run1.world_frames, pinned_indices=PINNED_INDICES
    )
    write_frames_npz(art, run1.local_frames, run1.world_frames, run1.timestamps)
    write_traces_json(art, stats)
    same_local = bool(np.array_equal(run1.local_frames, run2.local_frames))
    same_world = bool(np.array_equal(run1.world_frames, run2.world_frames))
    return 0 if (same_local and same_world) else 1


if __name__ == "__main__":
    sys.exit(main())

