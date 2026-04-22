#pragma once
#include "common/vec_math.h"
#include "common/atomic_utils.cuh"


static __global__ void compute_spring_forces(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices, // world space
    const int2* __restrict__ edges,
    const float* __restrict__ edge_lengths,
    const int n // edge size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0,v1] = edges[i];
        float3 p0 = vertices[v0], p1 = vertices[v1];
        const float youngs = 4e3;
        float3 e = p0 - p1;
        float length = norm(e);
        if ( length > 1e-8 ) {
            float rest_length = edge_lengths[i];
            float k = youngs / rest_length;
            float length_diff = length - rest_length;
            float3 force = e * (length_diff * k / length);
            atomicAddFloat3(&forces[v0], -force);
            atomicAddFloat3(&forces[v1], force);
            if ( enerys ) {
                float enery = 0.5f * length_diff * length_diff * k;
                atomicAdd(&enerys[v0], enery);
            }
            if ( Jx || Jx_diag ) {

                auto I = Mat3::identity();
                // float l_inv = 1.0f / length;
                // auto dxtdx =  Mat3::outer_product(e,e);
                // Mat3 K = -(I - (I - dxtdx * l_inv * l_inv ) * rest_length * l_inv) * k;

                // Hessian Filter for SPD
                float val1 = max(1.0f - rest_length / length, -0.001f);
                float3 dir = (p1 - p0) / length;
                Mat3 dir_dirT = Mat3::outer_product(dir, dir);
                Mat3 K = -((I - dir_dirT) * val1 + dir_dirT) * k;
                if ( Jx_diag ) {
                    atomicAddMat3(&Jx_diag[v0], K);
                    atomicAddMat3(&Jx_diag[v1], K);
                }
                if ( Jx ) {
                    atomicAddMat3(&Jx[i], -K);
                }
            }
        }
    }
}

// triangular finite element. The formula derivation comes from
//T. Kim and D. Eberle, "Dynamic deformables: implementation and production practicalities (now with code!)," in ACM SIGGRAPH 2022 Courses  (Chapter 10)
// very expensive
static __global__ void compute_BW_FEM(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_edges,
    const int2* __restrict__ edges,
    const int* __restrict__ vertices_obj,
    const float* __restrict__ YoungsModulus,
    const Mat2* __restrict__ Dms,
    int num_triangles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_triangles )
        return;

    Mat2 Dm = Dms[i];
    float Dm_det = Dm.det();
    float area = fabs(Dm_det) * 0.5f;
    Mat2 Dm_inv = Dm.inverse();

    auto tri_edges = triangle_edges[i];
    // auto [v0_idx,v1_idx,v2_idx] = indeices[i];
    auto [e1_ii,e2_ii,e3_ii] = tri_edges;
    int2 e1_i = edges[tri_edges.x];
    int2 e2_i = edges[tri_edges.y];
    int2 e3_i = edges[tri_edges.z];

    int v0_idx = e1_i.x;
    int v1_idx = e1_i.y;
    int v2_idx = e2_i.y;

    float3 v0 = vertices[v0_idx];
    float3 e1 = vertices[v1_idx] - v0;
    float3 e2 = vertices[v2_idx] - v0;

    float wudp1 = Dm_inv.r[0].x; // Dm_inv[0, 0]
    float wvdp1 = Dm_inv.r[0].y; // Dm_inv[0, 1]
    float wudp2 = Dm_inv.r[1].x; // Dm_inv[1, 0]
    float wvdp2 = Dm_inv.r[1].y; // Dm_inv[1, 1]
    float3 wu = e1 * wudp1 + e2 * wudp2;
    float3 wv = e1 * wvdp1 + e2 * wvdp2;
    float wu_norm = norm(wu);
    float wv_norm = norm(wv);
    float3 wu_ = wu_norm > 1e-6f ? wu * (1.0f / wu_norm) : make_float3(0.0f, 0.0f, 0.0f);
    float3 wv_ = wv_norm > 1e-6f ? wv * (1.0f / wv_norm) : make_float3(0.0f, 0.0f, 0.0f);

    float Cu = wu_norm - 1.0f;
    float Cv = wv_norm - 1.0f;
    float3 Cudp1 = wu_ * wudp1;
    float3 Cvdp1 = wv_ * wvdp1;
    float3 Cudp2 = wu_ * wudp2;
    float3 Cvdp2 = wv_ * wvdp2;

    // int obj = vertices_obj[v0_idx];
    // float k = YoungsModulus[obj];
    float k = 8e3f;

    //  Projector Matrices
    Mat3 I = Mat3::identity();
    Mat3 wu_proj_mat = (I - Mat3::outer_product(wu_, wu_)) * (wu_norm > 1e-6f ? 1.0f / wu_norm : 0.0f);
    Mat3 wv_proj_mat = (I - Mat3::outer_product(wv_, wv_)) * (wv_norm > 1e-6f ? 1.0f / wv_norm : 0.0f);
    float3 f1 = -area * (Cu * Cudp1 + Cv * Cvdp1) * k;
    float3 f2 = -area * (Cu * Cudp2 + Cv * Cvdp2) * k;

    float coef = -area * k;

    Mat3 f1d1 = (Mat3::outer_product(Cudp1, Cudp1) + Mat3::outer_product(Cvdp1, Cvdp1) +
        wu_proj_mat * (Cu * wudp1 * wudp1) + wv_proj_mat * (Cv * wvdp1 * wvdp1)) * coef;

    Mat3 f2d2 = (Mat3::outer_product(Cudp2, Cudp2) + Mat3::outer_product(Cvdp2, Cvdp2) +
        wu_proj_mat * (Cu * wudp2 * wudp2) + wv_proj_mat * (Cv * wvdp2 * wvdp2)) * coef;

    Mat3 f1d2 = (Mat3::outer_product(Cudp1, Cudp2) + Mat3::outer_product(Cvdp1, Cvdp2) +
        wu_proj_mat * (Cu * wudp1 * wudp2) + wv_proj_mat * (Cv * wvdp1 * wvdp2)) * coef;
    if ( Jx_diag ) {
        atomicAddMat3(&Jx_diag[v1_idx], f1d1);
        atomicAddMat3(&Jx_diag[v2_idx], f2d2);

        Mat3 f0d2 = (f1d2 + f2d2) * -1.0f;
        Mat3 f0d1 = (f1d1 + f1d2.transpose()) * -1.0f;
        Mat3 f0d0 = (f0d1 + f0d2) * -1.0f;

        atomicAddMat3(&Jx_diag[v0_idx], f0d0);
    }
    if ( Jx ) {
        if ( v1_idx < v2_idx )
            atomicAddMat3(&Jx[e3_ii], f1d2);
        else
            atomicAddMat3(&Jx[e3_ii], f1d2.transpose());

        auto f0d2 = -(f1d2 + f2d2);
        atomicAddMat3(&Jx[e2_ii], f0d2);
        auto f0d1 = -(f1d1 + f1d2.transpose());
        atomicAddMat3(&Jx[e1_ii], f0d1);
    }

    // shear 
    float wu_dot_wv = dot(wu_, wv_);
    float Cshear = wu_dot_wv; // cos angle

    float3 wv_proj_ = (wv_ - wu_ * wu_dot_wv) * (wu_norm > 1e-6f ? 1.0f / wu_norm : 0.0f);
    float3 wu_proj_ = (wu_ - wv_ * wu_dot_wv) * (wv_norm > 1e-6f ? 1.0f / wv_norm : 0.0f);

    float k_shear = k;
    float shear_coef = -area * k_shear * Cshear;

    float3 f1_s = (wv_proj_ * wudp1 + wu_proj_ * wvdp1) * shear_coef;
    float3 f2_s = (wv_proj_ * wudp2 + wu_proj_ * wvdp2) * shear_coef;
    float3 f0 = (f1_s + f2_s + f1 + f2) * -1.0f;
    atomicAddFloat3(&forces[v0_idx], f0);
    atomicAddFloat3(&forces[v1_idx], f1 + f1_s);
    atomicAddFloat3(&forces[v2_idx], f2 + f2_s);
    if ( enerys ) {
        atomicAdd(&enerys[v0_idx], 0.5f * area * (k * (Cu * Cu + Cv * Cv) + k * wu_dot_wv * wu_dot_wv));
    }
    #if 0 // Hessian of shear, only SPD part.
    float3 dCs_dx1 = wv_proj_ * wudp1 + wu_proj_ * wvdp1;
    float3 dCs_dx2 = wv_proj_ * wudp2 + wu_proj_ * wvdp2;

    float s_coef = -area * k_shear; // 注意：Hessian = -coef, 此处 coef 对应 Jx

    Mat3 f1d1_s = Mat3::outer_product(dCs_dx1, dCs_dx1) * s_coef;
    Mat3 f2d2_s = Mat3::outer_product(dCs_dx2, dCs_dx2) * s_coef;
    Mat3 f1d2_s = Mat3::outer_product(dCs_dx1, dCs_dx2) * s_coef;

    // --- 累加到 Jx ---
    if (Jx_diag) {
        atomicAddMat3(&Jx_diag[v1_idx], f1d1_s);
        atomicAddMat3(&Jx_diag[v2_idx], f2d2_s);
        
        Mat3 f0d1_s = (f1d1_s + f1d2_s.transpose()) * -1.0f;
        Mat3 f0d2_s = (f1d2_s + f2d2_s) * -1.0f;
        Mat3 f0d0_s = (f0d1_s + f0d2_s) * -1.0f;
        atomicAddMat3(&Jx_diag[v0_idx], f0d0_s);
    }

    if (Jx) {
        // e3: v1-v2, e2: v0-v2, e1: v0-v1
        if (v1_idx < v2_idx) atomicAddMat3(&Jx[e3_ii], f1d2_s);
        else atomicAddMat3(&Jx[e3_ii], f1d2_s.transpose());

        atomicAddMat3(&Jx[e2_ii], -(f1d2_s + f2d2_s));
        atomicAddMat3(&Jx[e1_ii], -(f1d1_s + f1d2_s.transpose()));
    }
    #endif
}


// === 1. 解析 SVD 计算 U(3x3), S(2), V(2x2) ===
static __device__ void svd3x2_analytic(const float3& wu, const float3& wv,
    float3& U0, float3& U1, float3& U2,
    float& s0, float& s1,
    float& v00, float& v01, float& v10, float& v11) {
    // F^T * F
    float c00 = dot(wu, wu);
    float c11 = dot(wv, wv);
    float c01 = dot(wu, wv);

    float delta = c00 - c11;
    float disc = sqrtf(delta * delta + 4.0f * c01 * c01);

    // 特征值 (奇异值的平方)
    float l0 = (c00 + c11 + disc) * 0.5f;
    float l1 = (c00 + c11 - disc) * 0.5f;
    s0 = sqrtf(max(l0, 1e-8f));
    s1 = sqrtf(max(l1, 1e-8f));

    // 计算 V (2x2)
    if ( abs(c01) > 1e-6f ) {
        float angle = 0.5f * atan2f(2.0f * c01, delta);
        v00 = cosf(angle);
        v10 = sinf(angle);
        v01 = -v10;
        v11 = v00;
    }
    else {
        v00 = 1.0f;
        v01 = 0.0f;
        v10 = 0.0f;
        v11 = 1.0f;
    }

    // 计算 U 的前两列 (3x1)
    U0 = (wu * v00 + wv * v10) * (1.0f / s0);
    U1 = (wu * v01 + wv * v11) * (1.0f / s1);

    // 计算 U 的第三列 (法线，用于面外 Twist)
    U2 = cross(U0, U1);
    float n_len = norm(U2);
    U2 = (n_len > 1e-6f) ? (U2 * (1.0f / n_len)) : make_float3(0.0f, 0.0f, 1.0f);
}

template<bool FixedR = true>
static __global__ void compute_ARAP_FEM(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_edges,
    const int2* __restrict__ edges,
    const int* __restrict__ vertices_obj,
    const float* __restrict__ YoungsModulus,
    const Mat2* __restrict__ Dms,
    int num_triangles
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_triangles ) return;

    Mat2 Dm_inv = Dms[i].inverse();
    float area = fabs(Dms[i].det()) * 0.5f;

    auto tri_edges = triangle_edges[i];
    int v0_idx = edges[tri_edges.x].x;
    int v1_idx = edges[tri_edges.x].y;
    int v2_idx = edges[tri_edges.y].y;

    float3 v0 = vertices[v0_idx], v1 = vertices[v1_idx], v2 = vertices[v2_idx];
    float3 e1 = v1 - v0, e2 = v2 - v0;

    float m00 = Dm_inv.r[0].x, m10 = Dm_inv.r[0].y;
    float m01 = Dm_inv.r[1].x, m11 = Dm_inv.r[1].y;

    // F = [wu, wv]
    float3 wu = e1 * m00 + e2 * m10;
    float3 wv = e1 * m01 + e2 * m11;

    // 1. SVD 分析
    float3 U0, U1, U2;
    float s0, s1, v00, v01, v10, v11;
    svd3x2_analytic(wu, wv, U0, U1, U2, s0, s1, v00, v01, v10, v11);

    // int obj = vertices_obj[v0_idx];
    // float mu = YoungsModulus[obj]; TODO
    float mu = 8e3;
    float coef = -mu * area;

    // --- 力的计算 (First Piola-Kirchhoff) ---
    // R = U * I_{3x2} * V^T
    float3 r_col0 = U0 * v00 + U1 * v01;
    float3 r_col1 = U0 * v10 + U1 * v11;

    float3 p0 = wu - r_col0;
    float3 p1 = wv - r_col1;

    float3 f1 = (p0 * m00 + p1 * m01) * coef;
    float3 f2 = (p0 * m10 + p1 * m11) * coef;
    // float3 f0 = -(f1 + f2);

    atomicAddFloat3(&forces[v0_idx], -(f1 + f2));
    atomicAddFloat3(&forces[v1_idx], f1);
    atomicAddFloat3(&forces[v2_idx], f2);

    if ( enerys ) {
        atomicAdd(&enerys[v0_idx], 0.5f * coef * (dot(p0, p0) + dot(p1, p1)));
    }

    // --- Hessian 的特征值分析与截断 (Exact Eigensystem) ---
    if ( Jx || Jx_diag ) {
        // 计算 Dm_inv_T_inv 矩阵 (2x2)
        Mat3 I = Mat3::identity();
        Mat3 f1d1, f2d2, f1d2;
        if constexpr ( FixedR ) {
            f1d1 = I * (m00 * m00 + m01 * m01) * coef;
            f2d2 = I * (m10 * m10 + m11 * m11) * coef;
            f1d2 = I * (m00 * m10 + m01 * m11) * coef;
        }
        else {
            // 构造 Twist 模式 (严格对应 MATLAB 中的 T0, T1, T2 降维到 3x2)
            const float inv_sqrt2 = 0.707106781f;

            // 面内 Twist (T0)
            float3 t0_u = (U1 * v00 - U0 * v01) * inv_sqrt2;
            float3 t0_v = (U1 * v10 - U0 * v11) * inv_sqrt2;

            // 面外 Twist 1 (T1), s2 = 0
            float3 t1_u = -U2 * v01 * inv_sqrt2;
            float3 t1_v = -U2 * v11 * inv_sqrt2;

            // 面外 Twist 2 (T2), s2 = 0
            float3 t2_u = -U2 * v00 * inv_sqrt2;
            float3 t2_v = -U2 * v10 * inv_sqrt2;

            // 特征值计算与截断 (Clamping)
            // 原始 Hessian 特征值为 lambda_i = 2 - 4/(s_i + s_j)
            // 截断后的权重 weight_i = 2 - max(0, lambda_i)
            auto clamp_weight = [](float s_a, float s_b) {
                float lambda = 2.0f - 4.0f / (s_a + s_b + 1e-8f);
                return 2.0f - max(0.0f, lambda);
            };

            float w0 = clamp_weight(s0, s1);
            float w1 = clamp_weight(s1, 0.0f); // s2 = 0
            float w2 = clamp_weight(s0, 0.0f); // s2 = 0

            // 构造 6x6 d2E/dF2 的四个 3x3 子块
            // H_F = 2 * I - w0 * (t0*t0^T) - w1 * (t1*t1^T) - w2 * (t2*t2^T)
            Mat3 H_uu = I * 2.0f;
            Mat3 H_uv = Mat3::zero();
            Mat3 H_vu = Mat3::zero();
            Mat3 H_vv = I * 2.0f;

            auto rank1_update = [&](float weight, const float3& tu, const float3& tv) {
                H_uu = H_uu - Mat3::outer_product(tu, tu) * weight;
                H_vv = H_vv - Mat3::outer_product(tv, tv) * weight;
                Mat3 uv = Mat3::outer_product(tu, tv) * weight;
                H_uv = H_uv - uv;
                H_vu = H_vu - uv.transpose();
            };

            rank1_update(w0, t0_u, t0_v);
            rank1_update(w1, t1_u, t1_v);
            rank1_update(w2, t2_u, t2_v);

            // 应用材质坐标链式法则 (映射到 9x9 Node Hessian)
            f1d1 = (H_uu * (m00 * m00) + H_uv * (m00 * m01) + H_vu * (m01 * m00) + H_vv * (m01 * m01)) * coef;
            f2d2 = (H_uu * (m10 * m10) + H_uv * (m10 * m11) + H_vu * (m11 * m10) + H_vv * (m11 * m11)) * coef;
            f1d2 = (H_uu * (m00 * m10) + H_uv * (m00 * m11) + H_vu * (m01 * m10) + H_vv * (m01 * m11)) * coef;
        }
        if ( Jx_diag ) {
            atomicAddMat3(&Jx_diag[v1_idx], f1d1);
            atomicAddMat3(&Jx_diag[v2_idx], f2d2);
            Mat3 f0d0 = f1d1 + f2d2 + f1d2 + f1d2.transpose();
            atomicAddMat3(&Jx_diag[v0_idx], f0d0);
        }
        if ( Jx ) {
            if ( v1_idx < v2_idx ) atomicAddMat3(&Jx[tri_edges.z], f1d2);
            else atomicAddMat3(&Jx[tri_edges.z], f1d2.transpose());

            atomicAddMat3(&Jx[tri_edges.y], -(f1d2 + f2d2));    // f0d2
            atomicAddMat3(&Jx[tri_edges.x], -(f1d1 + f1d2.transpose())); // f0d1
        }

    }
}
