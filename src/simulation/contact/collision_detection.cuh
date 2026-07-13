#pragma once

#include "common/cuda_utils.h"
#include "common/atomic_utils.cuh"
#include "common/geometric_algorithms.h"
#include "lbvh.cuh"

static __global__ void query_vf_pairs_simple_kernel(
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const float3* __restrict__ pos,
    const int3* __restrict__ faces,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float min_radius,
    const float max_dist,
    int* __restrict__ query_results,
    const int active_vertices_size,
    int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    if ( sorted_indices ) i = sorted_indices[i];
    float3 qp = pos[i];
    const auto& od = &obj_data[vertices_obj[i]];
    float radius = max(min_radius, max(od->granularity, od->thickness)) * 0.5f;
    AABB q_aabb = {
        .min = qp - radius,
        .max = qp + radius,
    };
    float max_dist_i = max(max_dist, radius * 1.5f);
    float max_dist_sq = max_dist_i * max_dist_i;
    bool is_active = i < active_vertices_size;
    BVH_QUERY_LOOP(q_aabb, 64,
        {
        int3 f = faces[prim_idx];
        if ( f.x == i || f.y == i || f.z == i ) continue;
        if (!is_active && f.x >= active_vertices_size) continue;
        float3 v0 = pos[f.x];
        float3 v1 = pos[f.y];
        float3 v2 = pos[f.z];

        float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);

        if ( dist_sq < max_dist_sq ) {

        float3 force_dir = cross(v1 - v0, v2 - v0);
        int sign = dot(force_dir, qp-v0) >= 0.0f ? 1 : -1;
        query_result[++query_count] = sign * prim_idx;
        }}

        );
}
static __global__ void query_ee_pairs_simple_kernel(
    const float3*__restrict__ pos,
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2*__restrict__ nodes,
    const AABB*__restrict__ aabbs,
    unsigned int root_idx,
    const int2*__restrict__ edges,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float min_radius,
    const float max_dist,
    const int active_vertices_size,
    int*__restrict__ query_results, int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    AABB q_aabb = aabbs[i];
    if ( sorted_indices ) i = sorted_indices[i];
    int2 edge = edges[i];
    float3 p0 = pos[edge.x];
    float3 p1 = pos[edge.y];
    float3 E = p1 - p0;
    const auto& od = &obj_data[vertices_obj[edge.x]];
    float radius = max(min_radius, max(od->granularity, od->thickness)) * 0.5f;
    float max_dist_i = max(max_dist, radius * 1.5f);
    float max_dist_sq = max_dist_i * max_dist_i;
    q_aabb.min = q_aabb.min - radius;
    q_aabb.max = q_aabb.max + radius;
    bool is_active = edge.x < active_vertices_size;
    // @formatter:off
    BVH_QUERY_LOOP(q_aabb, 64,
        if ( prim_idx <= i ) continue;
        int2 e = edges[prim_idx];
        if (edge.x == e.x || edge.x == e.y || edge.y == e.x || edge.y == e.y) continue;
        if (!is_active && e.x >= active_vertices_size) continue;
        float3 v0 = pos[e.x];
        float3 v1 = pos[e.y];

        float dist_sq = segment_segment_dist_sq_robust(p0, p1, v0, v1);

        if ( dist_sq < max_dist_sq ) {
            int sign = dot(v0 - p0, cross(v1 - p0, E)) < 0.0f ? 1 : -1; // signed area
            query_result[++query_count] = prim_idx * sign;
        }
    );
    // @formatter:on
}
static __global__ void query_vf_pairs_capsule_kernel(
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const float3* __restrict__ pos,
    const int3* __restrict__ faces,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float min_radius,
    const float3* __restrict__ inertial_offset,
    int* __restrict__ query_results,
    const int active_vertices_size,
    int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    if ( sorted_indices ) i = sorted_indices[i];

    float3 P0 = pos[i];
    float3 P1 = P0 + inertial_offset[i];
    const auto& od = &obj_data[vertices_obj[i]];
    float r_p = fmaxf(min_radius, od->thickness);
    AABB q_aabb = {
        .min = fmin3(P0,P1) - r_p,
        .max = fmax3(P0,P1) + r_p,
    };

    bool is_active = i < active_vertices_size;
    // @formatter:off
    BVH_QUERY_LOOP(q_aabb, 64, {
        int3 f = faces[prim_idx];
        if ( f.x == i || f.y == i || f.z == i ) continue;
        if (!is_active && f.x >= active_vertices_size) continue;

        float3 A0 = pos[f.x], A1 = A0 + inertial_offset[f.x];
        float3 B0 = pos[f.y], B1 = B0 + inertial_offset[f.y];
        float3 C0 = pos[f.z], C1 = C0 + inertial_offset[f.z];

        float3 T0_c = (A0 + B0 + C0) * (1.0f / 3.0f);
        float3 T1_c = (A1 + B1 + C1) * (1.0f / 3.0f);

        float r_T0_sq = fmaxf(len_sq(A0 - T0_c), fmaxf(len_sq(B0 - T0_c), len_sq(C0 - T0_c)));

        float r_T1_sq = fmaxf(len_sq(A1 - T1_c), fmaxf(len_sq(B1 - T1_c), len_sq(C1 - T1_c)));

        const auto& od_f = &obj_data[vertices_obj[f.x]];
        float r_tri = fmaxf(min_radius,  od_f->thickness);

        float r_T = sqrtf(fmaxf(r_T0_sq, r_T1_sq)) + r_tri;
        float r_total = r_p + r_T;

        float dist_sq = segment_segment_dist_sq_robust(P0, P1, T0_c, T1_c);

        if ( dist_sq <= r_total * r_total ) {
            float3 force_dir = cross(B0 - A0, C0 - A0);
            int sign = dot(force_dir, P0 - A0) >= 0.0f ? 1 : -1;
            query_result[++query_count] = sign * prim_idx;
        }
    });
    // @formatter:on

}

static __global__ void query_ee_pairs_capsule_kernel(
    const float3* __restrict__ pos,
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const int2* __restrict__ edges,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float min_radius,
    const float3* __restrict__ inertial_offset,
    const float3* __restrict__ edge_normals,
    const int active_vertices_size,
    int* __restrict__ query_results,
    int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    AABB q_aabb = aabbs[i];
    if ( sorted_indices ) i = sorted_indices[i]; // or nodes[i].x - 1

    int2 edge = edges[i];
    float3 p0 = pos[edge.x];
    float3 p1 = pos[edge.y];
    // float3 E = p1 - p0;
    float3 N = edge_normals[i]; 

    const auto& od = &obj_data[vertices_obj[edge.x]];
    float r_e1_thick = fmaxf(min_radius, od->thickness);

    bool is_active = edge.x < active_vertices_size;
    float3 A0 = p0, A1 = A0 + inertial_offset[edge.x];
    float3 B0 = p1, B1 = B0 + inertial_offset[edge.y];
    float3 E1_0_c = (A0 + B0) * 0.5f;
    float3 E1_1_c = (A1 + B1) * 0.5f;
    
    float r_E1 = sqrtf(fmaxf(len_sq(A0 - E1_0_c), len_sq(A1 - E1_1_c))) + r_e1_thick;
    q_aabb.min = q_aabb.min - r_e1_thick;
    q_aabb.max = q_aabb.max + r_e1_thick;

    // @formatter:off
    BVH_QUERY_LOOP(q_aabb, 64, {
        if ( prim_idx <= i ) continue;
        int2 e = edges[prim_idx];
        if (edge.x == e.x || edge.x == e.y || edge.y == e.x || edge.y == e.y) continue;
        if (!is_active && e.x >= active_vertices_size) continue;

        float3 C0 = pos[e.x], C1 = C0 + inertial_offset[e.x];
        float3 D0 = pos[e.y], D1 = D0 + inertial_offset[e.y];
        float3 E2_0_c = (C0 + D0) * 0.5f;
        float3 E2_1_c = (C1 + D1) * 0.5f;

        const auto& od_e2 = &obj_data[vertices_obj[e.x]];
        float r_e2_thick = fmaxf(min_radius,  od_e2->thickness);
        float r_E2 = sqrtf(fmaxf(len_sq(C0 - E2_0_c), len_sq(C1 - E2_1_c))) + r_e2_thick;

        float r_total = r_E1 + r_E2;
        float dist_sq = segment_segment_dist_sq_robust(E1_0_c, E1_1_c, E2_0_c, E2_1_c);

        if ( dist_sq <= r_total * r_total ) {
            // int sign = dot(C0 - A0, cross(D0 - A0, E)) < 0.0f ? 1 : -1;
            int sign = dot(E2_0_c - E1_0_c, N) < 0.0f ? 1 : -1;
            query_result[++query_count] = prim_idx * sign;
        }
    });
    // @formatter:on
}
static __global__ void query_ef_pairs_kernel(
    const AABB*__restrict__ edge_aabbs,
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2*__restrict__ nodes,
    const AABB*__restrict__ aabbs,
    const int2*__restrict__ edges,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int2* __restrict__ e2t,
    unsigned int root_idx,
    float min_radius,
    int*__restrict__ query_results,
    int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    AABB q_aabb = edge_aabbs[i];
    if ( sorted_indices ) i = sorted_indices[i];
    int2 edge = edges[i];
    const auto& od = &obj_data[vertices_obj[edge.x]];
    float radius = max(min_radius, max(od->granularity, od->thickness)) * 0.2f;
    q_aabb.min = q_aabb.min - radius;
    q_aabb.max = q_aabb.max + radius;
    int2 adj_tris = e2t[i];

    BVH_QUERY_LOOP(q_aabb, 64,
        if (prim_idx == adj_tris.x || prim_idx == adj_tris.y ) continue;
        query_result[++query_count] = prim_idx;
        );
}
