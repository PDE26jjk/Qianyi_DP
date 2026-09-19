#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverExplicit : SolverBase {
    // ~SolverExplicit() = default;
    explicit SolverExplicit(Simulator* simulator):SolverBase(simulator){}

    // void init() override;
    void step(float h) override;
    // The explicit integrator is only conditionally stable
    // (`h < 2 sqrt(m/k)`); on a garment with contact the measured limit is
    // ~0.25 ms, so the engine subdivides a coarser request down to this. Set
    // `explicit_max_step_h = 0` to switch the guard off.
    float max_stable_step_h() const override;
};
