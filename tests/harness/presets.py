"""Canonical solver parameter blocks and the solver registry.

The PDNewton block is the locally validated baseline used by the Blender
frontend; its provenance is recorded in LOCAL_DEV.md (gitignored). Do not
change these values without re-validating the simulation baseline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SolverInfo:
    name: str
    status: str  # "standard" | "experimental"
    parameters: dict[str, float]
    reason: str


PDNEWTON_PARAMETERS: dict[str, float] = {
    "smooth_times": 5,
    "step_h": 0.003,
    "sewing_k": 2e3,
    "query_radius": 1e-3,
    "sewing_forced_connect_frame": 80,
    "vf_force_k": 0.5,
    "vf_ground_k": 20,
    "ee_force_k": 0.2,
    "ef_force_k": 0.5,
    "ground": 1,
    "vf_force_type": 0,
    "ee_force_type": 0,
    "max_vel": 100.0,
    "gravity": -9.8,
    "linear_solver_type": 0,
    "pd_iters": 5,
    "pc_iters": 0,
    "subspace_iters": 0,
    "linear_iters": 5,
    "mask_stiff": 2000,
    "max_force_scale": 0.1,
    "average_mass_by_cloth": 0,
    "debug_e_id": -7,
    "debug_v_id": -6,
    "constitutive_model_planar": 0,
    "bending_model": 0,
    "bending_k": 1.0,
}

# Experimental blocks are taken from the same notebook; the solvers are not
# part of the validated baseline. Their smoke tests use strict xfail so a
# state flip (becoming usable) is surfaced explicitly instead of passing
# silently (see spec: Experimental solver governance).
VBD_PARAMETERS: dict[str, float] = {
    "smooth_times": 5,
    "step_h": 0.01,
    "sewing_k": 2e3,
    "sewing_forced_connect_frame": 80,
    "query_radius": 0.5e-3,
    "avbd_contact_beta": 1e3,
    "ground": 1,
    "max_vel": 10.0,
    "gravity": -9.8,
    "vbd_iters": 5,
    "mask_stiff": 2000,
    "parallel_eps": 1e-6,
    "gamma_r": 1,
    "gamma_min": -1,
    "debug_e_id": -7,
    "debug_v_id": -6,
    "constitutive_model_planar": 1,
}

XPBD_PARAMETERS: dict[str, float] = {
    "smooth_times": 5,
    "step_h": 0.001,
    "sewing_forced_connect_frame": 80,
    "vf_force_k": 1e2,
    "vf_ground_k": 1e6,
    "ee_force_k": 0.1,
    "ef_force_k": 0.2,
    "ground": 1,
    "vf_force_type": 0,
    "ee_force_type": 0,
    "max_vel": 100.0,
    "gravity": -9.8,
    "xpbd_damping": 0.1,
    "xpbd_relaxation": 0.1,
    "xpbd_iters": 20,
    "average_mass_by_cloth": 0,
    "constitutive_model_planar": 1,
    "xpbd_dynamics_iters": 1,
    "xpbd_use_lambdas": 0,
}

EXPLICIT_PARAMETERS: dict[str, float] = {
    "smooth_times": 5,
    "step_h": 0.00025,
    "sewing_k": 2e3,
    "sewing_forced_connect_frame": 80,
    "vf_force_k": 0.02,
    "ee_force_k": 0.002,
    "ef_force_k": 0.001,
    "ground": 1,
    "vf_force_type": 0,
    "ee_force_type": 0,
    "max_vel": 1.0,
    "gravity": -9.8,
}

SOLVER_REGISTRY: dict[str, SolverInfo] = {
    "PDNewton": SolverInfo(
        "PDNewton",
        "standard",
        PDNEWTON_PARAMETERS,
        "canonical locally validated configuration used by the Blender frontend",
    ),
    "VBD": SolverInfo(
        "VBD",
        "experimental",
        VBD_PARAMETERS,
        "not yet production-ready; smoke test xfails until it becomes usable",
    ),
    "XPBD": SolverInfo(
        "XPBD",
        "experimental",
        XPBD_PARAMETERS,
        "not yet production-ready; smoke test xfails until it becomes usable",
    ),
    "Explicit": SolverInfo(
        "Explicit",
        "experimental",
        EXPLICIT_PARAMETERS,
        "not yet production-ready; smoke test xfails until it becomes usable",
    ),
}


def solver_status(solver_name: str) -> str:
    try:
        return SOLVER_REGISTRY[solver_name].status
    except KeyError as exc:
        raise ValueError(
            f"unknown solver {solver_name!r}; known: {sorted(SOLVER_REGISTRY)}"
        ) from exc


def apply_preset(simulator, solver_name: str) -> dict[str, float]:
    """Select the solver and apply its parameter block; returns the params."""
    try:
        info = SOLVER_REGISTRY[solver_name]
    except KeyError as exc:
        raise ValueError(
            f"unknown solver {solver_name!r}; known: {sorted(SOLVER_REGISTRY)}"
        ) from exc
    simulator.set_solver(solver_name)
    simulator.set_parameters(dict(info.parameters))
    return info.parameters

