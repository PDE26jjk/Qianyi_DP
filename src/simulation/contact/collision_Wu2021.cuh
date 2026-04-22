#pragma once
#include "common/cuda_utils.h"
#include "common/atomic_utils.cuh"
#include "collision_type.cuh"

static __global__ void soft_phase_constraint_edge(
    float3* __restrict__ grads,
    int* __restrict__ num_violations,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    float max_dist,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_edges ) return;
    constexpr float rho = 10.f;
    auto [p0_i,p1_i] = edges[idx];
    float3 p0 = vertices[p0_i], p1 = vertices[p1_i];

    float dist = norm((p0 - p1));
    if ( dist > max_dist ) {
        float3 dir = (p0 - p1) / (dist + 1e-8f);
        float violation = dist - max_dist;
        float3 grad = dir * violation * rho;

        atomicAddFloat3(&grads[p0_i], grad);
        atomicAddFloat3(&grads[p1_i], -grad);

        atomicAdd(num_violations, 1);
    }
}
static __global__ void soft_phase_constraint_pp(
    float3* __restrict__ grads,
    int* __restrict__ num_violations,
    const CollisionResult_PP* __restrict__ near_point_pairs,
    const float3* __restrict__ vertices,
    float min_dist,
    int num_point_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_point_pairs ) return;
    constexpr float rho = 2.f;

    int p1_i = near_point_pairs[idx].p1, p2_i = near_point_pairs[idx].p2;
    float3 p1 = vertices[p1_i], p2 = vertices[p2_i];

    float dist = norm((p1 - p2));
    if ( dist < min_dist + 1e-5f ) {
        // L1
        float3 dir = (p2 - p1) / (dist + 1e-8f);
        float3 grad = dir * rho;
        // L2
        // float violation = max((min_dist - dist) * rho, min_dist * 0.05f);
        // float3 grad = dir * (violation);

        atomicAddFloat3(&grads[p1_i], grad);
        atomicAddFloat3(&grads[p2_i], -grad);

        atomicAdd(num_violations, 1);
        // printf("%d,%d, %f, %f\n", p1_i, p2_i, dist, min_dist);
    }
}

// Cloth point vs other triangles of objects, only update cloth points
static __global__ void soft_phase_constraint_tp(
    CollisionResult_TP* near_point_tri_pairs,
    float3* __restrict__ grads,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_indices,
    float min_dist,
    int num_point_tri_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_point_tri_pairs ) return;
    constexpr float rho = 1.f;

    auto near_point_tri_pair = near_point_tri_pairs[idx];
    int p_i = near_point_tri_pair.vert_idx, t_i = near_point_tri_pair.tri_idx;
    float3 p = vertices[p_i];
    float3 v0 = vertices[triangle_indices[t_i].x];
    float3& normal = near_point_tri_pair.normal;
    if ( dot(p - v0, normal) < min_dist ) {
        float3 grad = normal * rho;
        atomicAddFloat3(&grads[p_i], -grad);
    }
}
static __global__ void soft_phase_update(
    float3* __restrict__ points_collision,
    const float3* __restrict__ points_y,
    const float3* __restrict__ grads,
    float step_length,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;
    float3 v = points_y[idx];
    float3 coll_v = points_collision[idx];
    float3 grad = (coll_v - v) * 2.f;
    grad = grad + grads[idx];
    points_collision[idx] = coll_v - grad * step_length;

}

__device__ __forceinline__ void calc_barrier_f_and_prime(
    float x, float eps, float& f_val, float& f_prime) {
    if ( x >= eps ) {
        f_val = eps;
        f_prime = 0.0f;
    }
    else if ( x <= 1e-6f ) {
        x = 1e-6f;
        f_val = x;
        f_prime = 1.0f;
    }
    else {
        float a3 = -1.0f / (eps * eps);
        float a2 = 1.0f / eps;
        float a1 = 1.0f;
        float a0 = 1e-6f;

        float xx = x * x;
        f_val = a3 * xx * x + a2 * xx + +a1 * x + a0;
        f_prime = 3.0f * a3 * xx + 2.0f * a2 * x + a1;
    }
}

// Hard Phase: 计算 Log-Barrier 的梯度
static __global__ void hard_phase_constraint_pp_grad(
    float3* __restrict__ barrier_grads,
    int* __restrict__ num_violations,
    const CollisionResult_PP* __restrict__ near_point_pairs,
    const float3* __restrict__ vertices,
    float min_dist,
    float eps_slack,
    float mu,
    int num_point_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_point_pairs ) return;

    int p1_i = near_point_pairs[idx].p1;
    int p2_i = near_point_pairs[idx].p2;
    float3 p1 = vertices[p1_i];
    float3 p2 = vertices[p2_i];

    float dist_sq = len_sq(p1 - p2);
    float min_dist_sq = min_dist * min_dist;

    // 约束函数 c_ij(x) = ||x_ij||^2 - B^2
    float x = dist_sq - min_dist_sq;

    if ( x < eps_slack ) {
        float f_val, f_prime;
        calc_barrier_f_and_prime(x, eps_slack, f_val, f_prime);

        // 计算障碍函数能量 -mu * log(f(x)) 针对 p1 的梯度
        // Grad = -mu * (1 / f_val) * f_prime * \nabla c_ij(p1)
        // \nabla c_ij(p1) = 2 * (p1 - p2)
        float coeff = -mu * (f_prime / f_val) * 2.0f;
        float3 grad_p1 = (p1 - p2) * coeff;

        atomicAddFloat3(&barrier_grads[p1_i], grad_p1);
        atomicAddFloat3(&barrier_grads[p2_i], -grad_p1);
    }
    if (x < 0)
        atomicAdd(num_violations, 1);
}

// Hard Phase: 综合数据保真项和障碍梯度进行位置更新
static __global__ void hard_phase_update(
    float3* __restrict__ points_curr, // x^(l)
    const float3* __restrict__ points_init, // x^{init} (Soft Phase 之前的目标预测位置)
    const float3* __restrict__ barrier_grads,
    float min_dist,
    float step_length,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;

    float3 v_curr = points_curr[idx];
    float3 v_init = points_init[idx];

    // 目标函数：||x - x^{init}||^2 + Barrier
    // 对 x 求导：2 * (x - x^{init}) + \nabla Barrier
    float3 total_grad = (v_curr - v_init) * 2.0f + barrier_grads[idx];
    float3 dx = -total_grad * step_length;
    float dx_norm = norm(dx);
    // 最大位移不能超过碰撞厚度的一小部分（例如 20%），防止一次更新直接穿透
    float max_safe_dx = min_dist * 0.2f; 

    if (dx_norm > max_safe_dx) {
        // 等比例缩放位移，限制其最大步长
        dx = dx * (max_safe_dx / dx_norm);
    }
    points_curr[idx] = v_curr - dx;
}
