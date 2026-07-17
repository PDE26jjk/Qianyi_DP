#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverVBD : SolverBase {
    // ~SolverVBD() = default;
    explicit SolverVBD(Simulator* simulator):SolverBase(simulator){}

    void init() override;
    void step(float h) override;
    thrust::device_vector<float3> displacement;
    thrust::device_vector<Mat3> hessians;
};
