// This file contains code adapted from:
// NVIDIA Warp warp\native\solid_angle.h
// Original license:
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// This file contains code adapted from:
// https://github.com/alecjacobson/WindingNumber/tree/1e6081e52905575d8e98fb8b7c0921274a18752f
// Original license:
/*
MIT License
Copyright (c) 2018 Side Effects Software Inc.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

#include "common/cuda_utils.h"
#include "common/vec_math.h"
#include <thrust/device_vector.h>
#include "../contact/lbvh.cuh"
#include <cuda_runtime.h>


__device__ inline float3 cw_mul(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

namespace lbvh3d {

struct SolidAngleProps {
    float3 average_p;
    float3 normal;

    float3 n_ij_diag;
    float3 n_ijk_diag;

    float sum_permute_n_xyz;
    float two_n_xxy_n_yxx;
    float two_n_xxz_n_zxx;
    float two_n_yyz_n_zyy;
    float two_n_yyx_n_xyy;
    float two_n_zzx_n_xzz;
    float two_n_zzy_n_yzz;

    float n_xy;
    float n_yx;
    float n_yz;
    float n_zy;
    float n_zx;
    float n_xz;

    AABB3D box;
    float3 area_P;
    float area;
    float max_p_dist_sq;
};



__device__ inline bool evaluate_node_solid_angle(
    const float3& query_point, const SolidAngleProps& data, float& solid_angle, float accuracy_scale_sq)
{
    float max_p_sq = data.max_p_dist_sq;
    float3 q = query_point - data.average_p;
    float qlength2 = len_sq(q);

    if (qlength2 <= max_p_sq * accuracy_scale_sq) {
        solid_angle = 0.0f;
        return true; // need to descend
    }

    float omega_approx = 0.0f;
    float qlength_m2 = 1.0f / qlength2;
    float qlength_m1 = sqrtf(qlength_m2);
    q = q * qlength_m1;
 
    omega_approx = -qlength_m2 * dot(q, data.normal);
    float3 q2 = cw_mul(q, q);
    float qlength_m3 = qlength_m2 * qlength_m1;

    float omega_1 = qlength_m3
        * (data.n_ij_diag.x + data.n_ij_diag.y + data.n_ij_diag.z
           - 3.0f
               * (dot(q2, data.n_ij_diag) + q.x * q.y * (data.n_xy + data.n_yx)
                  + q.x * q.z * (data.n_zx + data.n_xz) + q.y * q.z * (data.n_yz + data.n_zy)));
    omega_approx += omega_1;

    float3 q3 = cw_mul(q2, q);
    float qlength_m4 = qlength_m2 * qlength_m2;

    float3 temp0 = make_float3(
        data.two_n_yyx_n_xyy + data.two_n_zzx_n_xzz,
        data.two_n_zzy_n_yzz + data.two_n_xxy_n_yxx,
        data.two_n_xxz_n_zxx + data.two_n_yyz_n_zyy);
    float3 temp1 = make_float3(
        q.y * data.two_n_xxy_n_yxx + q.z * data.two_n_xxz_n_zxx,
        q.z * data.two_n_yyz_n_zyy + q.x * data.two_n_yyx_n_xyy,
        q.x * data.two_n_zzx_n_xzz + q.y * data.two_n_zzy_n_yzz);

    float omega_2 = qlength_m4
        * (1.5f * dot(q, 3.0f * data.n_ijk_diag + temp0)
           - 7.5f * (dot(q3, data.n_ijk_diag) + q.x * q.y * q.z * data.sum_permute_n_xyz + dot(q2, temp1)));
    omega_approx += omega_2;

    if (!isfinite(omega_approx)) {
        solid_angle = 0.0f;
        return true;
    }

    solid_angle = omega_approx;
    return false; // approximation is good, no need to descend
}
__device__ inline float robust_solid_angle(const float3& a, const float3& b, const float3& c, const float3& p)
{
    float3 qa = a - p;
    float3 qb = b - p;
    float3 qc = c - p;

    float a_len = norm(qa);
    float b_len = norm(qb);
    float c_len = norm(qc);

    if (a_len == 0.0f || b_len == 0.0f || c_len == 0.0f) return 0.0f;

    qa = qa / a_len;
    qb = qb / b_len;
    qc = qc / c_len;

    float numerator = dot(qa, cross(qb - qa, qc - qa));
    if (numerator == 0.0f) return 0.0f;

    float denominator = 1.0f + dot(qa, qb) + dot(qa, qc) + dot(qb, qc);
    return 2.0f * atan2(numerator, denominator);
}

__global__ void refit_face_bvh_with_solid_angle_kernel(
    const float3* vertices,
    const int3* faces,
    unsigned int n,
    const int2* __restrict__ nodes,
    const unsigned int* __restrict__ parent,
    unsigned int* __restrict__ child_count,
    SolidAngleProps* __restrict__ solid_angle_props);

void build_face_bvh_with_solid_angle(const thrust::device_vector<float3>& vertices,
                    const thrust::device_vector<int3>& faces, BVH3D& bvh,
                    thrust::device_vector<SolidAngleProps>& solid_angle_props);
} // namespace lbvh3d