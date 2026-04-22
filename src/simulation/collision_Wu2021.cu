#include "contact/collision.cuh"
#include "contact/collision_Wu2021.cuh"

#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>


static __global__ void collision_Init_Kernel(
    float3* X_init,
    const float3* X_prev,
    const float3* Y_target,
    float D_LIMIT,
    const int number) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= number ) return;

    float3 diff = Y_target[i] - X_prev[i];
    float dist = norm(diff);
    // 限制最大位移为 D_LIMIT
    float scale = 1.0f;
    if ( dist > 1e-6f ) {
        scale = fminf(D_LIMIT / dist, 1.0f);
    }
    X_init[i] = X_prev[i] + diff * scale;
}
//L. Wu, B. Wu, Y. Yang, and H. Wang, "A Safe and Fast Repulsion Method for GPU-based Cloth Self Collisions," ACM Trans. Graph., vol. 40, no. 1, pp. 1–18, Feb. 2021, doi: 10.1145/3430025.
// Very difficult to tuning parameter, do not use it.
float3* SolverBase::collision_Wu2021_step(float3* points_y, bool first, bool& collision_done) {
    int block = 256;
    int vertex_size = params.nb_all_cloth_vertices;
    int edge_size = params.nb_all_cloth_edges;
    float3* grads = temp_vertices2_f3.data().get();
    float3* points_collision = temp_vertices_f3.data().get();
    float3* points_safe = this->points_safe.data().get();
    if ( first ) {
        cudaMemcpyAsync(points_safe, vertices_old.data().get(),
            vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    }
    float max_dist = params.cloth_edge_mean_length;
    float max_dist_edge = params.cloth_edge_mean_length * 2.f;
    collision_collect_near_pairs(points_y, max_dist, true, true, false,false);
    collision_Init_Kernel<<<vertex_size + block - 1, block>>>
        (points_collision, points_safe, points_y, max_dist, vertex_size);
    if ( pp_result_size_h == 0 ) {
        collision_done = true;
        return nullptr;
    }
    // cudaMemcpyAsync(points_collision, points_y, vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    int n = vertex_size;
    int* num_violations = sort_result_size.data().get();
    int num_violations_h;
    int max_soft_phase_step = 6;
    int soft_phase_step = 0;
    for ( ; soft_phase_step < max_soft_phase_step; ++soft_phase_step ) {
        cudaMemsetAsync(num_violations, 0, sizeof(int));
        cudaMemsetAsync(grads, 0, vertex_size * sizeof(float3));
        soft_phase_constraint_edge<<<(edge_size + block - 1) / block, block>>>(
            grads, num_violations,
            points_collision,
            edges.data().get(),
            max_dist_edge, edge_size);
        soft_phase_constraint_pp<<<(pp_result_size_h + block - 1) / block, block>>>(
            grads, num_violations,
            pp_collision_result.data().get(),
            points_collision,
            max_dist, pp_result_size_h);
        soft_phase_update<<<(n + block - 1) / block, block>>>(
            points_collision,
            points_y,
            grads,
            max_dist * 0.0035f,
            n);
        cudaMemcpy(&num_violations_h, num_violations, sizeof(int), cudaMemcpyDeviceToHost);
        if ( num_violations_h == 0 ) break;
    }
    if ( num_violations_h == 0 ) {
        collision_done = true;
        if ( soft_phase_step == 0 ) {
            // std::cout << "No need to soft_phase" << std::endl;
        }
        else
            std::cout << "soft_phase succeed, soft_phase_step: " << soft_phase_step << std::endl;
    }
    else {
        std::cout << "soft_phase failed, soft_phase_step: " << soft_phase_step <<
            ", num_violations: " << num_violations_h << std::endl;
        collision_done = false;
        float eps_slack = 21.75f * 1e-6f; // 论文参数转换到米级别的大致量级 (根据你的尺度调整)
        float mu = 1e-4f; // Barrier 强度系数
        alpha_hard /= 0.7f; // Hard phase 步长
        alpha_hard = max(0.001f, alpha_hard);
        int max_hard_phase_step = 8;
        float3* points_collision_backup = temp_vertices3_f3.data().get();
        cudaMemcpyAsync(points_collision_backup, points_collision,
            vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
        for ( int i = 0; i < 4; i++ ) {
            cudaMemcpyAsync(points_collision, points_safe,
                vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
            int hard_phase_step = 0;
            for ( ; hard_phase_step < max_hard_phase_step; ++hard_phase_step ) {
                cudaMemsetAsync(num_violations, 0, sizeof(int));
                cudaMemsetAsync(grads, 0, vertex_size * sizeof(float3));
                // hard_phase_constraint_edge();
                hard_phase_constraint_pp_grad<<<(pp_result_size_h + block - 1) / block, block>>>(
                    grads, num_violations,
                    pp_collision_result.data().get(),
                    points_collision,
                    max_dist,
                    eps_slack, mu, pp_result_size_h);
                hard_phase_update<<<(n + block - 1) / block, block>>>(points_collision,
                    points_y, grads, max_dist, alpha_hard, n);
            }
            cudaMemcpy(&num_violations_h, num_violations, sizeof(int), cudaMemcpyDeviceToHost);
            if ( num_violations_h == 0 ) {
                std::cout << "hard_phase succeed, alpha_hard: " << alpha_hard <<
                    ", hard_phase_step: " << i << std::endl;
                collision_done = true;
                break;
            }
            std::cout << "hard_phase failed, alpha_hard: " << alpha_hard <<
                ", num_violations: " << num_violations_h << std::endl;
            alpha_hard *= 0.7f;
            if ( i == 3 ) {
                // ...No other options
                std::cout << "No other options " << std::endl;
                return points_collision_backup;
            }
        }
    }
    // backup
    cudaMemcpyAsync(points_safe, points_collision,
        vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    return points_collision;
}

