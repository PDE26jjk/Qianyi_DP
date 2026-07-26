#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>


struct ContactState {
    union {
        float3 lambda;
        struct {
            float lambda_x;
            int type;
            float pen;
        };
    };
    float3 penalty;
    __forceinline__ __device__ void reset();
    __forceinline__ __device__ void new_step();
};

struct SolverVBD : SolverBase {
    // ~SolverVBD() = default;
    explicit SolverVBD(Simulator* simulator): SolverBase(simulator) {}

    void init() override;
    void step(float h) override;
    thrust::device_vector<float3> displacement;
    thrust::device_vector<Mat3> hessians;

    thrust::device_vector<ContactState> vf_states;
    thrust::device_vector<ContactState> ee_states;
    thrust::device_vector<ContactState> ground_states;
};
