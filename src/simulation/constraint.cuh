#pragma once
#include "common/vec_math.h"
#include "common/atomic_utils.cuh"
#include "solver_base.cuh"



static __global__ void compute_stitch_constraint(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices,
    const int* __restrict__ vertices_obj,
    const ObjectDataInput* __restrict__ obj_data,
    const char* __restrict__ mask,
    const int2* __restrict__ stitches,
    float min_dist,
    float k_input,
    int n // stitches size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    auto s = stitches[idx];
    auto [p0_i, p1_i] = s;
    if ( mask[p0_i] && mask[p1_i] ) return;
    float3 p0 = vertices[p0_i], p1 = vertices[p1_i];
    float3 e = p0 - p1;
    float length = norm(e);
    if ( length > 1e-7f ) {
        //
        float3 normal = e / length;
        // float force_max_mag = dot(rel_v, normal) / dt * 0.45f;
        float force_max_mag = 100000.f;

        float spring_length = max(min_dist, 1e-6f);
        float length_diff = length - spring_length;
        float factor = min(obj_data[vertices_obj[p0_i]].granularity, obj_data[vertices_obj[p1_i]].granularity) * 0.05f;
        float k = k_input * factor;
        k = min(k, abs(force_max_mag / length_diff));
        float3 force = normal * (length_diff * k);
        if ( enerys ) {
            atomicAdd(&enerys[p0_i], 0.5f * k * length_diff * length_diff);
        }
        // zero spring_length
        // float k = 1e4f;
        // float3 force = e * k;
        atomicAddFloat3(&forces[p0_i], -force);
        atomicAddFloat3(&forces[p1_i], force);
        if ( Jx || Jx_diag ) {
            // auto dxtdx = Mat3::outer_product(e, e);
            // auto I = Mat3::identity();
            // float l_inv = 1.0f / length;
            // Mat3 K = -(I - (I - dxtdx * l_inv * l_inv) * spring_length * l_inv) * k;
            // if ( Jx_diag ) {
            //     atomicAddMat3(&Jx_diag[p0_i], K);
            //     atomicAddMat3(&Jx_diag[p1_i], K);
            // }
            // if ( Jx ) {
            //     atomicAddMat3(&Jx[idx], -K);
            // }
            // zero spring_length
            Mat3 K = Mat3::identity() * (-k);

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
static __global__ void compute_q_safe_constraint(
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices,
    const char* __restrict__ mask,
    const float3* __restrict__ q_safe,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    if ( mask[idx] ) return;
    float3 p0 = vertices[idx], p1 = q_safe[idx];
    float3 e = p0 - p1;
    float length = norm(e);
    if ( length > 1e-7f ) {
        //
        float3 normal = e / length;
        // float force_max_mag = dot(rel_v, normal) / dt * 0.45f;

        constexpr float k = 8e4f;
        float3 force = normal * (length * k);
        if ( enerys ) {
            atomicAdd(&enerys[idx], 0.5f * k * length * length);
        }
        // zero spring_length
        atomicAddFloat3(&forces[idx], -force);
        // atomicAddFloat3(&forces[p1_i], force);
        Mat3 K = Mat3::identity() * -k;
        if ( Jx_diag )
            atomicAddMat3(&Jx_diag[idx], K);
        // atomicAddMat3(&Jx_diag[p1_i], K);

    }
}

static __global__ void compute_collision_penalty_force_triangle_point_plane(
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
    float min_dist = 1e-3f;
    if ( dist < min_dist ) {
        float k = 1e3f;
        float violation = min_dist - dist;
        if ( forces ) {
            float3 force = tp.normal * violation * k;
            atomicAddFloat3(&forces[idx_p], force);
            if ( v0_i < points_size ) {
                force = -force * (1.f / 3.f);
                atomicAddFloat3(&forces[v0_i], force);
                atomicAddFloat3(&forces[v1_i], force);
                atomicAddFloat3(&forces[v2_i], force);
            }
        }
        if ( Jx_diag ) {
            auto grad = Mat3::outer_product(tp.normal, tp.normal) * -k;
            atomicAddMat3(&Jx_diag[idx_p], grad);
            if ( v0_i < points_size ) {
                grad = -grad * (1.f / 3.f);
                atomicAddMat3(&Jx_diag[v0_i], grad);
                atomicAddMat3(&Jx_diag[v1_i], grad);
                atomicAddMat3(&Jx_diag[v2_i], grad);
            }
        }
    }
}

static __global__ void compute_collision_penalty_force_point_point(
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
        // float force_max_mag = 100000.f / dt;

        float spring_length = max(min_dist, 1e-6f);
        // float k = min(1e7f, abs(force_max_mag / (length - spring_length)));
        float k = 8e5f;
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
static __global__ void compute_normal_constraint_IPC_energy(
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    UnifiedNormalConstraint* __restrict__ constraints,
    const float3* __restrict__ x,
    const float* __restrict__ invMass,
    // const float d_hat,
    const ObjectDataInput* obj_data,
    const int* vertices_obj,
    float k,
    int num_constraints
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_constraints ) return;
    auto& c = constraints[tid];
    float3 t = make_float3(0.f, 0.f, 0.f);
    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        t += c.w[i] * x[c.v[i]];
    }
    float d = norm(t);
    c.lambda = d;
    float d_hat = obj_data[vertices_obj[c.v[0]]].granularity * 0.001f;
    if ( d >= d_hat || d <= 1e-6f ) return;
    float3 u = t / d;
    // c.normal = u;

    float d_ratio = d / d_hat;
    float log_term = logf(d_ratio);
    float diff = d - d_hat;

    float E_prime = -k * diff * (2.0f * log_term + 1.0f - 1.0f / d_ratio);
    float E_double_prime = -k * (2.0f * log_term + 3.0f - 2.0f / d_ratio - 1.0f / (d_ratio * d_ratio));

    float lambda_1 = fmaxf(0.0f, E_double_prime); // 法向刚度截断 (通常为正)
    float lambda_2 = fmaxf(0.0f, E_prime / d);    // 切向刚度截断 (E_prime < 0，因此这里必被截断为 0)

    Mat3 uuT = Mat3::outer_product(u, u);

    Mat3 H_local = uuT * fmaxf(0.0f, E_double_prime - E_prime / d) + Mat3::identity(lambda_2);

    c.H = H_local;
    for ( int i = 0; i < 4; ++i ) {
        int vi = c.v[i];
        if ( invMass[vi] > 0.f ) {
            float wi = c.w[i];
            // 累加负梯度 (Force)
            float3 f_i = u * (wi * -E_prime);
            atomicAddFloat3(&forces[vi], f_i);
            // Jx_diag 存储的是 -H，所以需要累加 -H_local * w_i^2
            Mat3 neg_delta_H_diag = H_local * (-wi * wi);
            atomicAddMat3(&Jx_diag[vi], neg_delta_H_diag);
        }
    }
}

static __global__ void compute_normal_constraint_IPC_force(
    float3* __restrict__ weight_forces,
    float* __restrict__ weight,
    UnifiedNormalConstraint* __restrict__ constraints,
    const float3* __restrict__ x,
    const float* __restrict__ invMass,
    // const float d_hat,
    const ObjectDataInput* obj_data,
    const int* vertices_obj,
    const float k,
    int num_constraints
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_constraints ) return;
    auto& c = constraints[tid];
    float3 t = make_float3(0.f, 0.f, 0.f);
    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        t += c.w[i] * x[c.v[i]];
    }
    float d = norm(t);
    c.lambda = d;
    float d_hat = obj_data[vertices_obj[c.v[0]]].granularity * 0.001f;
    if ( d >= d_hat || d <= 1e-8f ) return;
    float3 u;
    if ( len_sq(c.normal) > 0.5 ) {
        u = c.normal;
    }
    else {
        u = t / d;
    }
    // c.normal = u;

    float d_ratio = d / d_hat;
    float log_term = logf(d_ratio);
    log_term = min(log_term, 1e8f);
    float diff = d - d_hat;

    float E_prime = -k * diff * (2.0f * log_term + 1.0f - 1.0f / d_ratio);

    for ( int i = 0; i < 4; ++i ) {
        int vi = c.v[i];
        if ( invMass[vi] > 0.f && c.w[i] != 0.f ) {
            float w = -log_term;
            atomicAdd(&weight[vi], w);
            float wi = c.w[i];
            float3 f_i = u * (wi * -E_prime);
            atomicAddFloat3(&weight_forces[vi], f_i * w);
        }
    }
}
static __global__ void apply_weight_force(
    float3* __restrict__ weight_forces,
    const float* __restrict__ weight,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    if ( weight[idx] > 0.f )
        weight_forces[idx] *= 1.f / weight[idx];
}
