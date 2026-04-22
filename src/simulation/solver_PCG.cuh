#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

struct SolverPCG : SolverBase {
    // ~SolverPCG() = default;
    SolverPCG(Simulator* simulator):SolverBase(simulator){}
    // Jacobian matrix of forces or negative second derivative (Hessian matrix) of constraints/energy, stored per edge, excluding diagonal. It should be symmetrical, so only half of matrix is stored.
    thrust::device_vector<Mat3> Jx;
    // Diagonal part of Jx, stored by vertices -- some simplified algorithms only use this part.
    thrust::device_vector<Mat3> Jx_diag;
    // for bending opposite point pairs
    thrust::device_vector<Mat3> Jx_bend_cross;

    thrust::device_vector<float3> Ad;
    thrust::device_vector<float3> r;
    thrust::device_vector<float3> d;
    thrust::device_vector<float3> z;

    thrust::device_vector<float> temp1;
    void init() override;
    void update(float dt) override;
    void contact_handle() override;
    
protected:
    void compute_constraint(float dt);
    // float vector_field_dot_cub(const float3* a, const float3* b);
    void A_mult_x(
        float3* __restrict__ dst,
        const float3* __restrict__ src);
};
