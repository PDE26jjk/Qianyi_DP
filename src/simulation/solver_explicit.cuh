#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverExplicit : SolverBase {
    // ~SolverExplicit() = default;
    explicit SolverExplicit(Simulator* simulator):SolverBase(simulator){}

    // void init() override;
    void step(float h) override;
    
};
