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
        .min = fmin3(P0, P1) - r_p,
        .max = fmax3(P0, P1) + r_p,
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

/**
 * Evaluates point-triangle contact geometry.
 *
 * @param x0                   vertex position
 * @param x1, x2, x3           triangle vertex positions
 * @param combined_thickness   sum of vertex and triangle thickness (precomputed outside)
 * @param layer_diff           layer0 - layer1 (0: same layer, negative: vertex layer is smaller)
 * @param contact_side_sign    sign stored during broad phase, used only when layers are equal
 *                             (+1 or -1, indicating which side was originally detected)
 * @param vertex_normal        smoothed normal of the vertex (used when layer_diff < 0)
 * @param normal               output: unit normal pointing towards the vertex
 * @param u, v, w              output: barycentric coordinates of the closest point on the triangle
 * @param penetration          output: positive value when penetrating (= combined_thickness - signed distance)
 * @return                     true if a valid contact exists (barycentric coords inside triangle)
 */
__device__ inline bool compute_point_triangle_contact(
    const float3 x0,
    const float3 x1, const float3 x2, const float3 x3,
    const float combined_thickness,
    const int layer_diff,
    const float contact_side_sign,
    const float3 vertex_normal,
    float3& normal,
    float& u, float& v, float& w,
    float& penetration
) {
    // Triangle normal
    float3 tri_normal = cross(x2 - x1, x3 - x1);
    float len = norm(tri_normal);
    if ( len < 1e-8f )
        return false;
    tri_normal = tri_normal / len;

    // Orient the normal to point towards the vertex
    if ( layer_diff == 0 ) {
        // Same layer: use the sign stored during broad phase detection
        tri_normal = tri_normal * contact_side_sign;
    }
    else if ( layer_diff < 0 ) {
        // Vertex layer is smaller: flip normal if it points away from the vertex
        if ( dot(tri_normal, vertex_normal) > 0.0f ) {
            tri_normal = -tri_normal;
        }
    }
    // If layer_diff > 0, no adjustment is applied (preserving original behavior)

    float dist = dot(x0 - x1, tri_normal);
    float pen = combined_thickness - dist;
    if ( pen <= 0.0f )
        return false;

    // Closest point and barycentric test
    float3 closest = x0 - dist * tri_normal;
    barycentric(x1, x2, x3, closest, u, v, w);
    if ( u < 0.0f || v < 0.0f || w < 0.0f )
        return false;

    normal = tri_normal;
    penetration = pen;
    return true;
}

/**
 * Evaluates edge-edge contact geometry.
 *
 * @param p0, p1               endpoints of the first edge
 * @param q0, q1               endpoints of the second edge
 * @param combined_thickness   sum of the thickness values of the two edges (precomputed)
 * @param layer_diff           layer0 - layer1 (0: same, negative: edge0's layer < edge1's, positive: edge0's layer > edge1's)
 * @param contact_side_sign    sign stored during broad phase (+1 or -1), used only when layers are equal
 * @param edge_normal0         smoothed normal of the first edge
 * @param edge_normal1         smoothed normal of the second edge (used only when layer_diff > 0)
 * @param s, t                 output: closest-point parameters along the two edges
 * @param normal               output: unit normal pointing towards the first edge (after sign/layer correction)
 * @param penetration          output: positive value when penetrating (= combined_thickness - corrected distance)
 * @return                     true if the contact lies strictly inside both segments and penetration > 0
 */
__device__ inline bool compute_edge_edge_contact(
    const float3 p0, const float3 p1,
    const float3 q0, const float3 q1,
    const float combined_thickness,
    const int layer_diff,
    const float contact_side_sign,
    const float3 edge_normal0,
    const float3 edge_normal1,
    float& s, float& t,
    float3& normal,
    float& penetration
) {
    float3 ab;
    segment_segment_closest_robust(p0, p1, q0, q1, s, t, ab);

    // Only interior contacts are valid
    if ( s <= 0.0f || s >= 1.0f || t <= 0.0f || t >= 1.0f )
        return false;

    ab = -ab;  // vector from closest point on edge2 to closest point on edge1
    float dist = norm(ab);

    // Degenerate case: fall back to edge_normal0
    if ( dist < 1e-16f ) {
        normal = edge_normal0;
        ab = normal;
    }
    else {
        normal = ab / dist;
    }

    // Direction correction based on layer difference and broad-phase sign
    if ( layer_diff == 0 ) {
        // Same layer: use the sign stored during broad phase
        float sign_new = (dot(ab, edge_normal0) < 0.0f) ? 1.0f : -1.0f;
        sign_new *= contact_side_sign;
        if ( sign_new < 0.0f ) {
            dist = -dist;
            normal = -normal;
        }
    }
    else if ( layer_diff < 0 ) {
        // edge0's layer is smaller
        if ( dot(normal, edge_normal0) > 0.0f ) {
            normal = -normal;
            dist = -dist;
        }
    }
    else { // layer_diff > 0
        // edge0's layer is larger
        if ( dot(normal, edge_normal1) < 0.0f ) {
            normal = -normal;
            dist = -dist;
        }
    }

    float pen = combined_thickness - dist;
    if ( pen <= 0.0f )
        return false;

    penetration = pen;
    return true;
}
