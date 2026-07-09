#pragma once
#include "solver_base.cuh"

#include <thrust/device_vector.h>

#include "linear/solver_linear.cuh"

struct SolverSubspace;
struct SolverPDNewton : SolverBase {
    explicit SolverPDNewton(Simulator* simulator):SolverBase(simulator){}

    void init() override;
    // void compute_constraint();
    void step(float h) override;
    void solve_subspace(float3* dx, const float3* rhs);

private:
    LinearSolver* linear = nullptr;
    std::string m_last_linear_solver_name;
    std::string m_linear_solver_name = "PCG";
    thrust::device_vector<float3> dx;
    thrust::device_vector<float> Jx_diag_pd;
    thrust::device_vector<float> Jx_nondiag_pd;
    thrust::device_vector<float3> subspace_rhs;
    thrust::device_vector<float3> subspace_dy;
    SolverSubspace* subspace_solver = nullptr;
};
struct SolverSubspace : SolverPCG {
    SolverSubspace(Simulator* simulator): SolverPCG(simulator) {}
    void init(int diag_size, int edge_size) override;
    void A_mult_x(float3* __restrict__ dst,
        const float3* __restrict__ src) override;
};