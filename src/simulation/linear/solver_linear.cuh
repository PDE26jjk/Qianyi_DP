#pragma once

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/universal_vector.h>

#include "common/vec_math.h"
#include "simulation/simulator.h"

struct LinearSolver {
    LinearSolver(const LinearSolver& other) = delete;
    LinearSolver(LinearSolver&& other) noexcept = delete;
    LinearSolver& operator=(const LinearSolver& other) = delete;
    LinearSolver& operator=(LinearSolver&& other) noexcept = delete;
    virtual ~LinearSolver() = default;
    virtual void solve(float3* dx, const float3* rhs, int max_iters = 1000) = 0;
    LinearSolver(Simulator* simulator): simulator(simulator) {}
    virtual void init(int diag_size, int edge_size);
    void vector_field_dot(const float3* a, const float3* b, float* result);
    float vector_field_dot_sync(const float3* a, const float3* b);

    Simulator* simulator;
    // second derivative (Hessian matrix) of constraints/energy or negative Jacobian matrix of forces, stored per edge, excluding diagonal. It should be symmetrical, so only half of matrix is stored.
    thrust::device_vector<Mat3> Jx_nondiag;
    // Diagonal part of Jx, stored by vertices -- some simplified algorithms only use this part.
    thrust::device_vector<Mat3> Jx_diag;
    // 
    thrust::device_vector<Mat3> M_inv;
    // for bending opposite point pairs
    thrust::device_vector<Mat3> Jx_bend_cross;

protected:
    int m_edge_size;
    int m_diag_size;
    virtual void A_mult_x(
        float3* __restrict__ dst,
        const float3* __restrict__ src);
    thrust::device_vector<char> d_cub_temp;
    // size: 1
    thrust::device_vector<float> d_sum_result;
    thrust::device_vector<float> d_temp;
private:

};
struct SolverPCG : LinearSolver {
    // ~SolverPCG() = default;
    SolverPCG(Simulator* simulator): LinearSolver(simulator) {}
    void init(int diag_size, int edge_size) override {
        init(diag_size, edge_size, true);
    }
    void init(int diag_size, int edge_size, bool use_preconditioner);
    void solve(float3* dx, const float3* rhs, int max_iters) override {
        if ( use_preconditioner ) {
            solve_impl<true>(dx, rhs,max_iters);
        }
        else {
            solve_impl<false>(dx, rhs,max_iters);
        }
    }

    // float vector_field_dot_cub(const float3* a, const float3* b);
    thrust::device_vector<float3> Ad;
    thrust::device_vector<float3> r;
    thrust::device_vector<float3> d;
    thrust::device_vector<float3> z;

    thrust::device_vector<float> temp1;
    float extra_stiff;
    float projection_limit;

private:
    bool use_preconditioner;
    template<bool UsePreprocessingDiag>
    void solve_impl(float3* dx, const float3* rhs,int max_iters);
    virtual void try_make_A_PD();
    void print_debug(const float3* rhs);
};

struct SolverJacobi : LinearSolver {
    SolverJacobi(Simulator* simulator) : LinearSolver(simulator) {}

    void init(int diag_size, int edge_size) override;
    void solve(float3* dx, const float3* rhs,int max_iters) override;

    thrust::device_vector<float3> r;
    thrust::device_vector<float3> Ax;
};
