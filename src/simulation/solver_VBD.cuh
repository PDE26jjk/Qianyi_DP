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

    thrust::device_vector<float3> vf_lambdas;
    thrust::device_vector<float3> vf_penalties;
    thrust::device_vector<float3> ee_lambdas;
    thrust::device_vector<float3> ee_penalties;
    thrust::device_vector<float> ground_lambdas;
};
