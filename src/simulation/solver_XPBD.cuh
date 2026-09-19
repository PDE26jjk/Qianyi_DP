#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverXPBD : SolverBase {
    // ~SolverExplicit() = default;
    explicit SolverXPBD(Simulator* simulator):SolverBase(simulator){}

    void init() override;
    void step(float h) override;
private:
    thrust::device_vector<float3> delta;
    thrust::device_vector<float> lambdas;
    // Size of the membrane's range inside `lambdas`; bending hinges and
    // stitches accumulate into the ranges that follow it.
    int membrane_lambda_slots = 0;
};
