#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverXPBD : SolverBase {
    // ~SolverExplicit() = default;
    explicit SolverXPBD(Simulator* simulator):SolverBase(simulator){}

    // void init() override;
    void step(float h) override;
    
};
