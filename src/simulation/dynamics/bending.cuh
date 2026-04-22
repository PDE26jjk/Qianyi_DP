#pragma once

#include "common/vec_math.h"
#include "common/atomic_utils.cuh"

static __device__ void get_theta_dpk(
    float3 p0, float3 p1, float3 p2, float3 p3,
    float3& theta_dp0, float3& theta_dp1, float3& theta_dp2, float3& theta_dp3, float& theta
) {
    float3 p01 = p1 - p0, p02 = p2 - p0;
    float3 p32 = p2 - p3, p31 = p1 - p3;
    float3 e = p2 - p1;

    float3 n0 = cross(p01, p02);
    float3 n1 = cross(p32, p31);
    float n0_norm = norm(n0);
    float n1_norm = norm(n1);

    float3 n0_ = (n0_norm > 1e-6f) ? (n0 / n0_norm) : make_float3(0, 0, 0);
    float3 n1_ = (n1_norm > 1e-6f) ? (n1 / n1_norm) : make_float3(0, 0, 0);
    float e_norm = norm(e);
    float3 e_ = (e_norm > 1e-6f) ? (e / e_norm) : make_float3(0, 0, 0);

    float cos_theta = dot(n0_, n1_);
    float sin_theta = dot(cross(n0_, n1_), e_);
    theta = atan2(sin_theta, cos_theta);

    Mat3 I = Mat3::identity();
    Mat3 n0_dn0 = (I - Mat3::outer_product(n0_, n0_)) * (n0_norm > 1e-6f ? 1.0f / n0_norm : 0.0f);
    Mat3 n1_dn1 = (I - Mat3::outer_product(n1_, n1_)) * (n1_norm > 1e-6f ? 1.0f / n1_norm : 0.0f);

    float3 e_Tn0x = -cross(n0_, e_);
    float3 e_Tn1x = -cross(n1_, e_);

    Mat3 n0_dpk, n1_dpk;
    float3 sin_dpk, cos_dpk;

    // --- p1 ---
    n0_dpk = n0_dn0 * Mat3::cross_mat(-p02);
    n1_dpk = n1_dn1 * Mat3::cross_mat(p32);
    sin_dpk = e_Tn0x * n1_dpk - e_Tn1x * n0_dpk;
    cos_dpk = n1_ * n0_dpk + n0_ * n1_dpk;
    theta_dp1 = sin_dpk * cos_theta - cos_dpk * sin_theta;

    // --- p2 ---
    n0_dpk = n0_dn0 * Mat3::cross_mat(p01);
    n1_dpk = n1_dn1 * Mat3::cross_mat(-p31);
    sin_dpk = e_Tn0x * n1_dpk - e_Tn1x * n0_dpk;
    cos_dpk = n1_ * n0_dpk + n0_ * n1_dpk;
    theta_dp2 = sin_dpk * cos_theta - cos_dpk * sin_theta;

    // --- p0 ---
    n0_dpk = n0_dn0 * Mat3::cross_mat(e);
    sin_dpk = -e_Tn1x * n0_dpk;
    cos_dpk = n1_ * n0_dpk;
    theta_dp0 = sin_dpk * cos_theta - cos_dpk * sin_theta;

    // --- p3 ---
    n1_dpk = n1_dn1 * Mat3::cross_mat(-e);
    sin_dpk = e_Tn0x * n1_dpk;
    cos_dpk = n0_ * n1_dpk;
    theta_dp3 = sin_dpk * cos_theta - cos_dpk * sin_theta;
}


//T. Kim and D. Eberle, "Dynamic deformables: implementation and production practicalities (now with code!)," in ACM SIGGRAPH 2022 Courses  (Chapter 10)
static __global__ void compute_dihedral_bending_Fizt(
    Mat3* Jx,
    Mat3* Jx_diag,
    Mat3* Jx_bend_cross,
    float3* forces,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const int3* __restrict__ triangles,
    const int2* __restrict__ edge_opposite_points,
    int num_edges, float kb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;

    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) return; // No need to calculate bending force at the boundary.
    // int2 t_adj = e2t[i];
    // if ( t_adj.x == -1 || t_adj.y == -1 ) return; 

    int2 e_i = edges[i];
    int p1_idx = e_i.x, p2_idx = e_i.y;
    int p0_idx = p_op.x, p3_idx = p_op.y;

    float3 theta_dp0, theta_dp1, theta_dp2, theta_dp3;
    float theta;
    get_theta_dpk(vertices[p0_idx], vertices[p1_idx], vertices[p2_idx], vertices[p3_idx],
        theta_dp0, theta_dp1, theta_dp2, theta_dp3, theta);

    float coef = -kb;
    if ( Jx_diag != nullptr ) {
        atomicAddMat3(&Jx_diag[p0_idx], Mat3::outer_product(theta_dp0, theta_dp0) * coef);
        atomicAddMat3(&Jx_diag[p1_idx], Mat3::outer_product(theta_dp1, theta_dp1) * coef);
        atomicAddMat3(&Jx_diag[p2_idx], Mat3::outer_product(theta_dp2, theta_dp2) * coef);
        atomicAddMat3(&Jx_diag[p3_idx], Mat3::outer_product(theta_dp3, theta_dp3) * coef);
        if ( Jx != nullptr ) {
            auto [t1_i, t2_i] = e2t[i];
            auto t1 = triangles[t1_i];
            auto t2 = triangles[t2_i];
            auto f1d2 = Mat3::outer_product(theta_dp1, theta_dp2) * coef;
            atomicAddMat3(&Jx[i], f1d2);

            auto f0d1 = Mat3::outer_product(theta_dp0, theta_dp1) * coef;
            auto f0d2 = Mat3::outer_product(theta_dp0, theta_dp2) * coef;
            if ( p0_idx < p1_idx ) {
                atomicAddMat3(&Jx[t1.x], f0d1);
                atomicAddMat3(&Jx[t1.y], f0d2);
            }
            else {
                atomicAddMat3(&Jx[t1.y], f0d1.transpose());
                atomicAddMat3(&Jx[t1.z], p0_idx < p2_idx ? f0d2 : f0d2.transpose());
            }
            auto f3d1 = Mat3::outer_product(theta_dp3, theta_dp1) * coef;
            auto f3d2 = Mat3::outer_product(theta_dp3, theta_dp2) * coef;
            if ( p3_idx < p1_idx ) {
                atomicAddMat3(&Jx[t2.y], f3d1);
                atomicAddMat3(&Jx[t2.x], f3d2);
            }
            else {
                atomicAddMat3(&Jx[t2.x], f3d1.transpose());
                atomicAddMat3(&Jx[t2.z], p3_idx < p2_idx ? f3d2 : f3d2.transpose());
            }
            auto f0d3 = Mat3::outer_product(theta_dp0, theta_dp3) * coef;
            // atomicAddMat3(&Jx_bend_cross[i],p0_idx < p3_idx ? f0d3 : f0d3.transpose());
            atomicAddMat3(&Jx_bend_cross[i], f0d3);
        }
    }
    coef *= theta;
    if ( forces != nullptr ) {
        atomicAddFloat3(&forces[p0_idx], theta_dp0 * coef);
        atomicAddFloat3(&forces[p1_idx], theta_dp1 * coef);
        atomicAddFloat3(&forces[p2_idx], theta_dp2 * coef);
        atomicAddFloat3(&forces[p3_idx], theta_dp3 * coef);
    }
}

static __device__ float compute_cotangent(float3 p0, float3 p1, float3 p2) {
    float3 e1 = normalized(p1 - p0), e2 = normalized(p2 - p0);
    float dot_ = dot(e1, e2);
    float cross_ = norm(cross(e1, e2));
    const float eps = 1e-6f;
    return dot_ / (cross_ + eps);
}
static __global__ void precompute_IBM_Q(
    float4* q,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const float3* __restrict__ vertices, // Material space
    const int2* __restrict__ edge_opposite_points,
    const Mat2* __restrict__ Dms,
    int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;
    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) {
        return;
    }
    int2 e_i = edges[i];
    int p0_idx = e_i.x, p1_idx = e_i.y;
    int p2_idx = p_op.x, p3_idx = p_op.y;
    auto [t1_i, t2_i] = e2t[i];
    // auto t1 = triangles[t1_i];
    // auto t2 = triangles[t2_i];
    float area1 = fabs(Dms[t1_i].det()) * 0.5f;
    float area2 = fabs(Dms[t2_i].det()) * 0.5f;
    float k = 3.0f / (area1 + area2 + 1e-8f);
    float3 x0 = vertices[p0_idx], x1 = vertices[p1_idx], x2 = vertices[p2_idx], x3 = vertices[p3_idx];
    float c01 = compute_cotangent(x0, x1, x2); // cot(x1-x0, x2-x0) 
    float c02 = compute_cotangent(x0, x3, x1); // cot(x1-x0, x3-x0)
    float c03 = compute_cotangent(x1, x2, x0); // cot(x0-x1, x2-x1)
    float c04 = compute_cotangent(x1, x0, x3); // cot(x0-x1, x3-x1)
    float4 _q = make_float4(c03 + c04, c01 + c02, -c01 - c03, -c02 - c04);
    q[i] = _q * sqrtf(k);
    // Q[i] = Mat4::outer_product(q,q) * k;
}

// M. Bergou, M. Wardetzky, D. Harmon, D. Zorin, and E. Grinspun, "A quadratic bending model for inextensible surfaces"
static __global__ void compute_quadratic_Bending_IBM(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    Mat3* __restrict__ Jx_bend_cross,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float4* __restrict__ _q,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const int3* __restrict__ triangles,
    const int2* __restrict__ edge_opposite_points,
    int num_edges, float kb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;
    int2 p_op = edge_opposite_points[i];
    if ( p_op.x == -1 || p_op.y == -1 ) return; // No need to calculate bending force at the boundary.
    int2 e_i = edges[i];

    int p0_idx = e_i.x, p1_idx = e_i.y;
    int p2_idx = p_op.x, p3_idx = p_op.y;
    float3 x0 = vertices[p0_idx], x1 = vertices[p1_idx], x2 = vertices[p2_idx], x3 = vertices[p3_idx];
    float4 q = _q[i];
    float3 qtX = x0 * q.x + x1 * q.y + x2 * q.z + x3 * q.w;
    if ( enerys ) {
        atomicAdd(&enerys[p0_idx],0.5f * kb * len_sq(qtX));
    }
    qtX = qtX * kb;

    if ( forces ) {
        atomicAddFloat3(&forces[p0_idx], -qtX * q.x);
        atomicAddFloat3(&forces[p1_idx], -qtX * q.y);
        atomicAddFloat3(&forces[p2_idx], -qtX * q.z);
        atomicAddFloat3(&forces[p3_idx], -qtX * q.w);
    }
    kb = -kb;
    if ( Jx_diag ) {
        atomicAddMat3(&Jx_diag[p0_idx], Mat3::identity(q.x * q.x * kb));
        atomicAddMat3(&Jx_diag[p1_idx], Mat3::identity(q.y * q.y * kb));
        atomicAddMat3(&Jx_diag[p2_idx], Mat3::identity(q.z * q.z * kb));
        atomicAddMat3(&Jx_diag[p3_idx], Mat3::identity(q.w * q.w * kb));
    }
    if ( Jx ) {
        auto [t1_i, t2_i] = e2t[i];
        auto t1 = triangles[t1_i];
        auto t2 = triangles[t2_i];
        auto f0d1 = Mat3::identity(q.x * q.y * kb);
        atomicAddMat3(&Jx[i], f0d1);

        auto f2d3 = Mat3::identity(q.z * q.w * kb);
        atomicAddMat3(&Jx_bend_cross[i], f2d3);

        auto f0d2 = Mat3::identity(q.x * q.z * kb);
        auto f1d2 = Mat3::identity(q.y * q.z * kb);
        auto f0d3 = Mat3::identity(q.x * q.w * kb);
        auto f1d3 = Mat3::identity(q.y * q.w * kb);
        if ( p0_idx > p2_idx ) {
            atomicAddMat3(&Jx[t1.x], f0d2);
            atomicAddMat3(&Jx[t1.y], f1d2);
        }
        else {
            atomicAddMat3(&Jx[t1.y], f0d2);
            atomicAddMat3(&Jx[t1.z], f1d2);
        }
        if ( p3_idx < p0_idx ) {
            atomicAddMat3(&Jx[t2.y], f0d3);
            atomicAddMat3(&Jx[t2.x], f1d3);
        }
        else {
            atomicAddMat3(&Jx[t2.x], f0d3);
            atomicAddMat3(&Jx[t2.z], f1d3);
        }
        // auto assign_jacobian = [&](int3 t, int p_opp, Mat3 f0_opp, Mat3 f1_opp) {
        //     [cite_start]int t_edges[3] = {t.x, t.y, t.z};
        //     for(int e = 0; e < 3; ++e) {
        //         int e_idx = t_edges[e];
        //         if (e_idx == i) continue; // Skip the central edge
        //         
        //         int2 v = edges[e_idx];
        //         // Check if this edge connects p0 and p_opp
        //         if ((v.x == p0_idx && v.y == p_opp) || (v.y == p0_idx && v.x == p_opp)) {
        //             atomicAddMat3(&Jx[e_idx], f0_opp);
        //         } 
        //         // Check if this edge connects p1 and p_opp
        //         else if ((v.x == p1_idx && v.y == p_opp) || (v.y == p1_idx && v.x == p_opp)) {
        //             atomicAddMat3(&Jx[e_idx], f1_opp);
        //         }
        //     }
        // };
        //
        // assign_jacobian(triangles[t1_i], p2_idx, f0d2, f1d2);
        // assign_jacobian(triangles[t2_i], p3_idx, f0d3, f1d3);

    }
}
