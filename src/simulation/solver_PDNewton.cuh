#pragma once
#include "solver_base.cuh"

#include <cstdint>
#include <cuda_runtime.h>
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
    // Observability: { Newton residual of the first outer iteration, of the
    // last one } in this substep, i.e. the norm of the force residual the
    // linear solve is asked to remove.
    thrust::device_vector<float> newton_residual;
    void fill_residual_metrics(std::vector<float>& out) override;
    // Assembled (element + contact) diagonal of the tangent, without the
    // fixed projective part `static_diags`. Kept separate from the solver's
    // `Jx_diag` so a chord / modified-Newton run can reuse the assembly for
    // several outer iterations without accumulating the fixed part or the
    // per-iteration damping diagonal on top of it.
    thrust::device_vector<Mat3> Jx_diag_assembled;
    thrust::device_vector<float> Jx_diag_pd;
    // thrust::device_vector<float> Jx_nondiag_pd;
    thrust::device_vector<float3> subspace_rhs;
    thrust::device_vector<float3> subspace_dy;
    SolverSubspace* subspace_solver = nullptr;
    // Captured Projective-Dynamics iteration (`pd_cuda_graph`), plus the key
    // that says when the capture still matches the buffers and parameters it
    // was recorded with. See SolverPDNewton::step.
    cudaGraphExec_t iter_graph_exec = nullptr;
    uint64_t iter_graph_key = 0;
    bool iter_graph_broken = false;
    // Fork/join between the legacy stream and the iteration's own stream.
    cudaEvent_t stream_fork = nullptr;
    cudaEvent_t stream_join = nullptr;
};
struct SolverSubspace : SolverPCG {
    SolverSubspace(Simulator* simulator): SolverPCG(simulator) {}
    void init(int diag_size, int edge_size, bool) override;
    void A_mult_x(float3* __restrict__ dst,
        const float3* __restrict__ src) override;
};
