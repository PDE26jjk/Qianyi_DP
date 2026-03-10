#pragma once
#include "common/vec_math.h"
#include "common/atomic_utils.cuh"


static __global__ void compute_spring_forces(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const float3* __restrict__ vertices, // world space
    const int2* __restrict__ edges,
    const float* __restrict__ edge_lengths,
    const int n // edge size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0,v1] = edges[i];
        float3 p0 = vertices[v0], p1 = vertices[v1];
        float youngs = 8e4;
        float3 e = p0 - p1;
        float length = norm(e);
        if ( length > 1e-7 ) {
            float3 force = e * ((length - edge_lengths[i]) * youngs / length);
            atomicAddFloat3(&forces[v0], -force);
            atomicAddFloat3(&forces[v1], force);
            if (Jx || Jx_diag) {
                auto dxtdx =  Mat3::outer_product(e,e);
                auto I = Mat3::identity();
                float l_inv = 1.0f / length;
                Mat3 K = -(I - (I - dxtdx * l_inv * l_inv ) * edge_lengths[i] * l_inv) * youngs;
                if (Jx_diag) {
                    atomicAddMat3(&Jx_diag[v0],K);
                    atomicAddMat3(&Jx_diag[v1],K);
                }
                if (Jx) {
                    atomicAddMat3(&Jx[i],-K);
                }
            }
        }
    }
}

// triangular finite element
static __global__ void k_compute_stretch_gradient(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    const float3* __restrict__ q1,
    const int3* __restrict__ indeices,
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

    auto [v0_idx,v1_idx,v2_idx] = indeices[i];
    // int2 e1_i = edges[tri_edges.x];
    // int2 e2_i = edges[tri_edges.y];

    // int v0_idx = e1_i.x;
    // int v1_idx = e1_i.y;
    // int v2_idx = e2_i.y;

    float3 v0 = q1[v0_idx];
    float3 e1 = q1[v1_idx] - v0;
    float3 e2 = q1[v2_idx] - v0;

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

    int obj = vertices_obj[v0_idx];
    float k = YoungsModulus[obj];

    //  Projector Matrices
    Mat3 I = Mat3::identity();
    Mat3 wu_proj_mat = (I - Mat3::outer_product(wu_, wu_)) * (wu_norm > 1e-6f ? 1.0f / wu_norm : 0.0f);
    Mat3 wv_proj_mat = (I - Mat3::outer_product(wv_, wv_)) * (wv_norm > 1e-6f ? 1.0f / wv_norm : 0.0f);

    float coef = -area * k;

    Mat3 f1d1 = (Mat3::outer_product(Cudp1, Cudp1) + Mat3::outer_product(Cvdp1, Cvdp1) +
        wu_proj_mat * (Cu * wudp1 * wudp1) + wv_proj_mat * (Cv * wvdp1 * wvdp1)) * coef;

    Mat3 f2d2 = (Mat3::outer_product(Cudp2, Cudp2) + Mat3::outer_product(Cvdp2, Cvdp2) +
        wu_proj_mat * (Cu * wudp2 * wudp2) + wv_proj_mat * (Cv * wvdp2 * wvdp2)) * coef;

    Mat3 f1d2 = (Mat3::outer_product(Cudp1, Cudp2) + Mat3::outer_product(Cvdp1, Cvdp2) +
        wu_proj_mat * (Cu * wudp1 * wudp2) + wv_proj_mat * (Cv * wvdp1 * wvdp2)) * coef;

    atomicAddMat3(&Jx_diag[v1_idx], f1d1);
    atomicAddMat3(&Jx_diag[v2_idx], f2d2);

    Mat3 f0d2 = (f1d2 + f2d2) * -1.0f;
    Mat3 f0d1 = (f1d1 + f1d2.transpose()) * -1.0f;
    Mat3 f0d0 = (f0d1 + f0d2) * -1.0f;

    atomicAddMat3(&Jx_diag[v0_idx], f0d0);
}
