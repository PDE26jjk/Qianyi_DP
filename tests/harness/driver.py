"""Blender-equivalent simulation driver.

Driver semantics follow the Blender frontend (``simulation_manager.py``):
``input_data`` -> ``set_solver`` -> apply the solver parameter block -> for
each frame run ``ceil(frame_time / dt)`` substeps of ``update(dt)`` and
collect local- and world-space vertex data per frame.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .presets import SOLVER_REGISTRY, apply_preset


@dataclass
class SimRun:
    """Collected per-frame simulation data."""

    local_frames: np.ndarray  # (F, N, 3) float32, local space
    world_frames: np.ndarray  # (F, N, 3) float32, world space
    timestamps: np.ndarray  # (F,) float64, simulated time at each frame
    solver: str
    params: dict[str, float]
    fps: int
    dt: float


class SimDriver:
    """Drive a Qianyi_DP simulation with Blender-equivalent semantics."""

    def __init__(
        self,
        qydp,
        solver: str = "PDNewton",
        fps: int = 24,
        frames: int = 60,
        dt: float = 0.001,
    ) -> None:
        self.qydp = qydp
        self.solver = solver
        self.fps = fps
        self.frames = frames
        self.dt = dt

    def substeps_per_frame(self) -> int:
        """Substeps per frame: ceil(frame_time / dt), minimum 1."""
        frame_time = 1.0 / self.fps
        return max(1, math.ceil(frame_time / self.dt))

    def run(self, input_data: dict, solver: str | None = None) -> SimRun:
        """Run the configured simulation and return per-frame data."""
        solver = solver or self.solver
        if solver not in SOLVER_REGISTRY:
            raise ValueError(
                f"unknown solver {solver!r}; known: {sorted(SOLVER_REGISTRY)}"
            )

        sim = self.qydp.simulator
        # set_solver / set_parameters MUST precede input_data: Simulator::init
        # creates the solver object from the solver name at input_data time, so
        # a set_solver call after input_data would keep the previous solver.
        params = apply_preset(sim, solver)
        sim.input_data(input_data)

        substeps = self.substeps_per_frame()
        local_frames: list[np.ndarray] = []
        world_frames: list[np.ndarray] = []
        timestamps: list[float] = []
        elapsed = 0.0
        for _ in range(self.frames):
            for _ in range(substeps):
                sim.update(self.dt)
                elapsed += self.dt
            local_frames.append(np.array(sim.get_simulation_data(), copy=True))
            world_frames.append(np.array(sim.get_simulation_data(world_space=True), copy=True))
            timestamps.append(elapsed)

        return SimRun(
            local_frames=np.asarray(local_frames, dtype=np.float32),
            world_frames=np.asarray(world_frames, dtype=np.float32),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            solver=solver,
            params=params,
            fps=self.fps,
            dt=self.dt,
        )
