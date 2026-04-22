#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

struct SolverPNCG : SolverBase {
    // ~SolverPNCG() = default;
    SolverPNCG(Simulator* simulator):SolverBase(simulator){}
    thrust::device_vector<Mat3> Jx;
    thrust::device_vector<Mat3> Jx_bend_cross;
    // Diagonal part of Jx, stored by vertices, see .cuh of PCG solver.
    thrust::device_vector<Mat3> Jx_diag; // -H in the paper. Calculate for P = diag(H)^-1
    
    thrust::device_vector<float3> p;
    float d_hat;
    // thrust::device_vector<float3> p_prev;
    
    void init() override;
    void P_mult_x(float3* dst, const float3* src);
    void H_mult_x_neg(float3* dst, const float3* src);
    void update(float dt) override;

    void compute_elastic_constraint(float3*, float3* x_tilde, float3* forces);
    void compute_collision_constraint(float3*, float3* forces, bool change_hash);
};
