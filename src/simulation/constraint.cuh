#pragma once
#include "common/vec_math.h"
#include "common/atomic_utils.cuh"
#include "solver_base.cuh"


static __global__ void compute_collision_constraint_triangle_point_plane(
    // Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const CollisionResult_TP* __restrict__ triangle_point_pairs,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_indices,
    const int points_size,
    const int n // pairs size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    auto tp = triangle_point_pairs[idx];
    auto idx_t = tp.tri_idx, idx_p = tp.vert_idx;
    auto [v0_i, v1_i, v2_i] = triangle_indices[idx_t];
    auto v0 = vertices[v0_i];
    auto p = vertices[idx_p];
    float dist = dot((p - v0), tp.normal);
    float min_dist = sqrtf(tp.min_dist_sq);
    if ( dist < min_dist ) {
        float k = 1e3f;
        float violation = min_dist - dist;
        if ( forces ) {
            float3 force = tp.normal * violation * k;
            atomicAddFloat3(&forces[idx_p], force);
            if ( v0_i < points_size ) {
                force = -force * (1.f/3.f);
                atomicAddFloat3(&forces[v0_i], force);
                atomicAddFloat3(&forces[v1_i], force);
                atomicAddFloat3(&forces[v2_i], force);
            }
        }
        if ( Jx_diag ) {
            auto grad = Mat3::outer_product(tp.normal, tp.normal) * -k;
            atomicAddMat3(&Jx_diag[idx_p], grad);
            if ( v0_i < points_size ) {
                grad = -grad * (1.f/3.f);
                atomicAddMat3(&Jx_diag[v0_i], grad);
                atomicAddMat3(&Jx_diag[v1_i], grad);
                atomicAddMat3(&Jx_diag[v2_i], grad);
            }
        }
    }
}
static __global__ void compute_collision_constraint_point_point(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const float3* __restrict__ velocities,
    const CollisionResult_PP* __restrict__ point_pairs,
    const float3* __restrict__ vertices,
    float min_dist,
    float dt,
    int n // pairs size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    auto pp = point_pairs[idx];
    int p0_i = pp.p1, p1_i = pp.p2;
    float3 p0 = vertices[p0_i], p1 = vertices[p1_i];
    float3 e = p0 - p1;
    float length = norm(e);
    if ( length > 1e-7f && length < min_dist ) {

        float3 rel_v = velocities[p0_i] - velocities[p1_i];
        float3 normal = e / length;
        // float force_max_mag = dot(rel_v, normal) / dt * 0.45f;
        float force_max_mag = 500.f / dt;

        float spring_length = max(min_dist, 1e-6f);
        float k = min(1e3f, abs(force_max_mag / (length - spring_length)));
        float3 force = normal * ((length - spring_length) * k);
        atomicAddFloat3(&forces[p0_i], -force);
        atomicAddFloat3(&forces[p1_i], force);
        if ( Jx || Jx_diag ) {
            auto dxtdx = Mat3::outer_product(e, e);
            auto I = Mat3::identity();
            float l_inv = 1.0f / length;
            Mat3 K = -(I - (I - dxtdx * l_inv * l_inv) * spring_length * l_inv) * k;
            if ( Jx_diag ) {
                atomicAddMat3(&Jx_diag[p0_i], K);
                atomicAddMat3(&Jx_diag[p1_i], K);
            }
            if ( Jx ) {
                atomicAddMat3(&Jx[idx], -K);
            }
        }
    }
}

static __global__ void compute_stitch_constraint(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const float3* __restrict__ vertices,
    const char* __restrict__ mask,
    const int2* __restrict__ stitches,
    float min_dist,
    int n // stitches size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    auto s = stitches[idx];
    auto [p0_i, p1_i] = s;
    if (mask[p0_i] && mask[p1_i]) return;
    float3 p0 = vertices[p0_i], p1 = vertices[p1_i];
    float3 e = p0 - p1;
    float length = norm(e);
    if ( length > 1e-7f) {

        float3 normal = e / length;
        // float force_max_mag = dot(rel_v, normal) / dt * 0.45f;
        float force_max_mag = 10000.f;

        float spring_length = max(min_dist, 1e-6f);
        float k = min(4e9f, abs(force_max_mag / (length - spring_length)));
        float3 force = normal * ((length - spring_length) * k);
        atomicAddFloat3(&forces[p0_i], -force);
        atomicAddFloat3(&forces[p1_i], force);
        if ( Jx || Jx_diag ) {
            auto dxtdx = Mat3::outer_product(e, e);
            auto I = Mat3::identity();
            float l_inv = 1.0f / length;
            Mat3 K = -(I - (I - dxtdx * l_inv * l_inv) * spring_length * l_inv) * k;
            if ( Jx_diag ) {
                atomicAddMat3(&Jx_diag[p0_i], K);
                atomicAddMat3(&Jx_diag[p1_i], K);
            }
            if ( Jx ) {
                atomicAddMat3(&Jx[idx], -K);
            }
        }
    }
}
