#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

struct SolverChebyshev : SolverBase {
    // ~SolverChebyshev() = default;
    SolverChebyshev(Simulator* simulator):SolverBase(simulator){}
    // thrust::device_vector<Mat3> Jx;
    // thrust::device_vector<Mat3> Jx_bend_cross;
    // Diagonal part of Jx, stored by vertices, see .cuh of PCG solver.
    thrust::device_vector<Mat3> Jx_diag;
    
    thrust::device_vector<float> energys;

    thrust::device_vector<float3> velocities_prev;
    thrust::device_vector<float3> q_prev;
    thrust::device_vector<float3> q_next;
    thrust::device_vector<float3> q_backup;
    
    void init() override;
    void update(float dt) override;
    void contact_handle() override;

    float stepping; // alpha
    std::vector<int> profile_iter;
    std::vector<float> profile_value;
    
protected:
    void compute_constraint(float3* q,float3*, bool update_Jx=false);
};
