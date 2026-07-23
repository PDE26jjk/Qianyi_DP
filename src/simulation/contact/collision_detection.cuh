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

// ------------------------------------------------------------------
// capsule–capsule collision culling
// Returns true if the capsules MAY intersect (i.e., NOT culled).
// A capsule is defined by a start point, an end point, and a radius.
// ------------------------------------------------------------------
__device__ __forceinline__ bool capsule_capsule_intersects(
    float3 A0, float3 A1, float rA,
    float3 B0, float3 B1, float rB) {
    float dist_sq = segment_segment_dist_sq_robust(A0, A1, B0, B1);
    float r = rA + rB;
    return dist_sq <= r * r;
}
// ------------------------------------------------------------------
// trajectory capsule for an edge.
// The capsule axis connects the edge midpoints at start and end of the
// time step. The radius bounds the maximum distance from the axis to any
// deformed edge point, plus the given thickness.
// ------------------------------------------------------------------
__device__ __forceinline__ void edge_trajectory_capsule(
    float3 A0, float3 B0,          // edge endpoints at start
    float3 A1, float3 B1,          // edge endpoints at end   (start + displacement)
    float thickness,              // per-edge thickness (max of vertex thicknesses)
    float3& cap_start,             // capsule start point (midpoint at t=0)
    float3& cap_end,               // capsule end point   (midpoint at t=1)
    float& cap_radius)            // capsule radius
{
    cap_start = (A0 + B0) * 0.5f;
    cap_end = (A1 + B1) * 0.5f;

    float r0_sq = len_sq(A0 - cap_start);
    float r1_sq = len_sq(A1 - cap_end);
    cap_radius = sqrtf(fmaxf(r0_sq, r1_sq)) + thickness;
}
// ------------------------------------------------------------------
// Compute a bounding capsule for a triangle over a time step.
// The capsule axis goes through the triangle centroids at start and end.
// The radius conservatively encloses the deformed triangle.
// ------------------------------------------------------------------
__device__ __forceinline__ void triangle_trajectory_capsule(
    float3 A0, float3 B0, float3 C0,
    float3 A1, float3 B1, float3 C1,
    float tri_thickness,
    float3& cap_start, float3& cap_end, float& cap_radius) {
    cap_start = (A0 + B0 + C0) * (1.0f / 3.0f);
    cap_end = (A1 + B1 + C1) * (1.0f / 3.0f);

    float r0_sq = fmaxf(len_sq(A0 - cap_start),
        fmaxf(len_sq(B0 - cap_start), len_sq(C0 - cap_start)));
    float r1_sq = fmaxf(len_sq(A1 - cap_end),
        fmaxf(len_sq(B1 - cap_end), len_sq(C1 - cap_end)));
    cap_radius = sqrtf(fmaxf(r0_sq, r1_sq)) + tri_thickness;
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

        const auto& od_f = &obj_data[vertices_obj[f.x]];
        float r_tri = fmaxf(min_radius,  od_f->thickness);

        float3 tri_cap_start, tri_cap_end;
        float  tri_cap_radius;
        triangle_trajectory_capsule(A0, B0, C0, A1, B1, C1,
                                    r_tri, 
                                    tri_cap_start, tri_cap_end, tri_cap_radius);

        if ( !capsule_capsule_intersects(P0, P1, r_p,
                                         tri_cap_start, tri_cap_end, tri_cap_radius) )
            continue;
        float3 force_dir = cross(B0 - A0, C0 - A0);
        int sign = dot(force_dir, P0 - A0) >= 0.0f ? 1 : -1;
        query_result[++query_count] = sign * prim_idx;
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
    i = nodes[i].x - 1;

    int2 edge = edges[i];
    float3 A0 = pos[edge.x];
    float3 B0 = pos[edge.y];
    float3 A1 = A0 + inertial_offset[edge.x];
    float3 B1 = B0 + inertial_offset[edge.y];
    float3 N = edge_normals[i];

    // thickness for edge i
    const auto& od_i = obj_data[vertices_obj[edge.x]];
    float r_e1_thick = fmaxf(min_radius, od_i.thickness);

    // Query edge trajectory capsule
    float3 cap1_start, cap1_end;
    float cap1_radius;
    edge_trajectory_capsule(A0, B0, A1, B1, r_e1_thick,
        cap1_start, cap1_end, cap1_radius);

    // conservative AABB for BVH query (original extended by thickness)
    AABB q_aabb;
    q_aabb.min = fmin3(fmin3(A0, B0), fmin3(A1, B1)) - r_e1_thick;
    q_aabb.max = fmax3(fmax3(A0, B0), fmax3(A1, B1)) + r_e1_thick;

    bool is_active = (edge.x < active_vertices_size);

    // @formatter:off
    BVH_QUERY_LOOP(q_aabb, 64, {
        if ( prim_idx <= i ) continue;
        int2 e = edges[prim_idx];
        if (!is_active && e.x >= active_vertices_size) continue;
        if (edge.x == e.x || edge.x == e.y || edge.y == e.x || edge.y == e.y) continue;

        float3 C0 = pos[e.x];
        float3 D0 = pos[e.y];
        float3 C1 = C0 + inertial_offset[e.x];
        float3 D1 = D0 + inertial_offset[e.y];

        const auto& od_e2 = obj_data[vertices_obj[e.x]];
        float r_e2_thick = fmaxf(min_radius, od_e2.thickness);

        float3 cap2_start, cap2_end;
        float  cap2_radius;
        edge_trajectory_capsule(C0, D0, C1, D1, r_e2_thick,
                                cap2_start, cap2_end, cap2_radius);

        if (!capsule_capsule_intersects(cap1_start, cap1_end, cap1_radius,
                                        cap2_start, cap2_end, cap2_radius))
            continue;
        // compute signed result: sign based on relative position to edge i's normal
        int sign = dot(cap1_start - cap2_start, N) < 0.0f ? 1 : -1;
       //  if (i == 7) {
       //      if (query_count == 0) {
       //          printf("i=%d: A0=(%e,%e,%e), B0=(%e,%e,%e), "
       //     "A1=(%e,%e,%e), B1=(%e,%e,%e)\n",
       //     i, A0.x, A0.y, A0.z, B0.x, B0.y, B0.z,
       //     A1.x, A1.y, A1.z, B1.x, B1.y, B1.z);
       //
       //          printf("N=(%e,%e,%e), r_e1_thick=%e\n", N.x, N.y, N.z, r_e1_thick);
       //      }
       //      printf("e1:%d (v%d,v%d), e2:%d (v%d,v%d), \n", i, edge.x, edge.y, prim_idx, e.x, e.y);
       //      printf("prim_idx=%d: C0=(%e,%e,%e), D0=(%e,%e,%e), "
       // "C1=(%e,%e,%e), D1=(%e,%e,%e)\n",
       // prim_idx,
       // C0.x, C0.y, C0.z, D0.x, D0.y, D0.z,
       // C1.x, C1.y, C1.z, D1.x, D1.y, D1.z);
       //
       //  }
        query_result[++query_count] = prim_idx * sign;
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
    float3 ba;
    segment_segment_closest_robust(p0, p1, q0, q1, s, t, ba);

    // Only interior contacts are valid
    if ( s <= 0.0f || s >= 1.0f || t <= 0.0f || t >= 1.0f )
        return false;

    float dist = norm(ba);

    // Degenerate case: fall back to edge_normal0
    if ( dist < 1e-16f ) {
        ba = -edge_normal0;
        normal = ba;
    }
    else {
        normal = ba / dist;
    }

    // Direction correction based on layer difference and broad-phase sign
    if ( layer_diff == 0 ) {
        // Same layer: use the sign stored during broad phase
        float sign_new = (dot(ba, edge_normal0) < 0.0f) ? 1.0f : -1.0f;
        sign_new *= contact_side_sign;
        if ( sign_new < 0.0f ) {
            normal = -normal;
            dist = -dist;
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
__device__ __forceinline__ float3 robust_edge_pair_normal(
    float3 A, float3 B,
    float3 C, float3 D) {
    const float3 u = B - A;
    const float3 v = D - C;

    // primary cross product
    float3 n = cross(u, v);
    float len_sq = dot(n, n);
    if ( len_sq > 1e-16f ) return n * rsqrtf(len_sq);

    // fallback use A-C
    n = A - C;
    len_sq = dot(n, n);
    if ( len_sq > 1e-16f ) return n * rsqrtf(len_sq);

    // if everything fails, return an arbitrary direction (non-zero)
    return make_float3(0.0f, 0.0f, 1.0f);
}

__device__ __forceinline__ float planar_truncation_t(
    float3 p, float proj_delta, float3 n, float3 d, float gamma_min
) {
    if ( fabsf(proj_delta) < 1e-16f ) return 1.0f;
    float t = dot(n, d - p) / proj_delta;               // intersection parameter
    if ( t < 0.0f ) return 1.0f;                     // plane behind starting point

    // conservative relaxation
    t = fmaxf(0.f, fminf(1.0f, t - gamma_min));
    return t;
}
__device__ inline bool vertex_triangle_planar_truncation(
    float3 v, float3 delta_v,
    float3 t0, float3 delta_t0,
    float3 t1, float3 delta_t1,
    float3 t2, float3 delta_t2,
    float thickness,
    int layer_diff,               // L_vertex - L_triangle
    float3 vertex_normal,
    float parallel_eps, float gamma_r, float gamma_min,
    float& t_v, float& t_t0, float& t_t1, float& t_t2
) {
    // closest point on triangle to current vertex
    float3 closest_pt;
    float len_sq = point_triangle_sq_dist(v, t0, t1, t2, closest_pt);

    // degenerate: too close – skip truncation (returns all 1.0)
    if ( len_sq < 1e-18f ) {
        printf("vf too close! %e \n", len_sq);
        return false;
    }

    float3 n_hat = v - closest_pt;
    if ( layer_diff != 0 ) {
        float3 low_normal;
        if ( layer_diff < 0 ) {
            low_normal = vertex_normal;                // vertex is low
        }
        else {
            // triangle is low → compute face normal from start positions
            low_normal = cross(t1 - t0, t2 - t0);
        }
        if ( dot(low_normal, n_hat) < 0.0f ) {
            return false; // Penetrated, no truncation, collision force will handle it
        }
    }

    float3 n;             // unit normal (points to vertex side)
    float current_dist = sqrtf(len_sq);
    if ( current_dist < thickness ) {
        n = normalized(cross(t1 - t0, t2 - t0));
        if ( dot(n, n_hat) < 0.0f ) n = -n;
    }
    else {
        n = n_hat * rsqrtf(len_sq);
    }

    // projected displacements towards the opposing side
    float proj_v = dot(n, delta_v);
    float proj_t0 = dot(n, delta_t0);
    float proj_t1 = dot(n, delta_t1);
    float proj_t2 = dot(n, delta_t2);

    float delta_v_n = fmaxf(-proj_v, 0.0f);
    float delta_t_n = fmaxf(fmaxf(proj_t0, proj_t1), fmaxf(proj_t2, 0.0f));

    // early exit if no collision or are moving away
    // float len_diff = sqrtf(len_sq);
    // if ( delta_v_n + delta_t_n + thickness < len_diff || delta_v_n + delta_t_n == 0.f )
    //     return false;

    // plane point d on the line connecting closest point to vertex
    // float lmbd = delta_t_n / (delta_v_n + delta_t_n);
    float lmbd;
    // if ( delta_v_n + delta_t_n == 0.0f )
    if ( delta_v_n == 0.0f || delta_t_n == 0.0f )
        lmbd = 0.5f;
    else
        lmbd = delta_t_n / (delta_v_n + delta_t_n);
    lmbd = fmaxf(0.2f, fminf(0.8f, lmbd));
    thickness *= gamma_r;
    float3 d = closest_pt + (lmbd * current_dist + thickness * 0.5f) * n;

    // compute truncation factors for all four vertices
    // t_v = proj_v < -parallel_eps ?
    //           planar_truncation_t(v, proj_v, n, d, gamma_min) : 1.0f;
    // d -= thickness * n_hat;
    // t_t0 = proj_t0 > parallel_eps ?
    //            planar_truncation_t(t0, proj_t0, n, d, gamma_min) : 1.0f;
    // t_t1 = proj_t1 > parallel_eps ?
    //            planar_truncation_t(t1, proj_t1, n, d, gamma_min) : 1.0f;
    // t_t2 = proj_t2 > parallel_eps ?
    //            planar_truncation_t(t2, proj_t2, n, d, gamma_min) : 1.0f;
    t_v = planar_truncation_t(v, proj_v, n, d, gamma_min);
    d -= thickness * n;
    t_t0 = planar_truncation_t(t0, proj_t0, n, d, gamma_min);
    t_t1 = planar_truncation_t(t1, proj_t1, n, d, gamma_min);
    t_t2 = planar_truncation_t(t2, proj_t2, n, d, gamma_min);
    // if (delta_v_n == 0.0f) {
    //     if (dot(n, d - t0) < -1e-7f) t_t0 = fminf(t_t0, 1e-6f);
    //     if (dot(n, d - t1) < -1e-7f) t_t1 = fminf(t_t1, 1e-6f);
    //     if (dot(n, d - t2) < -1e-7f) t_t2 = fminf(t_t2, 1e-6f);
    // }
    return (t_v < 1.0f) || (t_t0 < 1.0f) || (t_t1 < 1.0f) || (t_t2 < 1.0f);
}
__device__ inline bool edge_edge_planar_truncation(
    float3 e0v0, float3 delta_e0v0,
    float3 e0v1, float3 delta_e0v1,
    float3 e1v0, float3 delta_e1v0,
    float3 e1v1, float3 delta_e1v1,
    float thickness,
    int layer_diff,               // L_edge0 - L_edge1
    float3 edge0_normal,
    float3 edge1_normal,
    float parallel_eps, float gamma_r, float gamma_min,
    float& t_e0v0, float& t_e0v1, float& t_e1v0, float& t_e1v1
) {
    // ---- closest points on the two edges (at start configuration) ----
    float s, t;
    float3 dP;   // dP = closest_pt_on_edge0 - closest_pt_on_edge1
    segment_segment_closest_robust(e0v0, e0v1, e1v0, e1v1, s, t,
        dP);

    // float3 u0 = e0v1 - e0v0;
    float3 v1 = e1v1 - e1v0;
    // float3 c1 = e0v0 + s * u0;          // point on edge 0
    float3 c2 = e1v0 + t * v1;          // point on edge 1

    float3 n_hat = dP;             // identical to c1-c2
    float len_sq = dot(n_hat, n_hat);

    float scaled_thickness = thickness * gamma_r;

    if ( len_sq < 1e-18f ) {
        // --- degenerate case: edges almost intersect at start ---
        // printf("ee too close! %e \n", len_sq);
        return false;
    }
    if ( layer_diff != 0 ) {
        float3 low_normal = (layer_diff < 0) ? edge0_normal : edge1_normal;
        if ( dot(low_normal, n_hat) < 0.0f ) {
            return false; // Penetrated, no truncation, collision force will handle it
        }
    }
    float3 n = n_hat * rsqrtf(len_sq);     // unit normal pointing towards edge 0

    // projected displacements towards the opposing edge
    float proj_e0v0 = dot(n, delta_e0v0);
    float proj_e0v1 = dot(n, delta_e0v1);
    float proj_e1v0 = dot(n, delta_e1v0);
    float proj_e1v1 = dot(n, delta_e1v1);

    // maximum approach of each edge towards the other
    float delta_e0 = fmaxf(fmaxf(-proj_e0v0, -proj_e0v1), 0.0f); // edge 0 moving towards edge 1
    float delta_e1 = fmaxf(fmaxf(proj_e1v0, proj_e1v1), 0.0f); // edge 1 moving towards edge 0

    float current_dist = sqrtf(len_sq);

    // early exit if the gap cannot be closed
    if ( delta_e0 + delta_e1 + thickness < current_dist || delta_e0 + delta_e1 == 0.0f ) {
        // if ( current_dist > thickness * 3.0f && delta_e0 + delta_e1 == 0.0f ) {
        return false;
    }

    // interpolation factor – plane lies closer to the side that moves less
    float lmbd;
    if ( delta_e0 + delta_e1 + thickness < current_dist ||
        delta_e0 + delta_e1 == 0.0f )
        lmbd = 0.5f;
    else
        lmbd = delta_e1 / (delta_e0 + delta_e1);

    // lmbd = fmaxf(parallel_eps, fminf(1.f-parallel_eps, lmbd));
    lmbd = fmaxf(0.05f, fminf(0.995f, lmbd));

    // separation planes
    float3 d0 = c2 + (current_dist * lmbd + scaled_thickness * 0.5f) * n; // for edge 0 vertices
    float3 d1 = c2 + (current_dist * lmbd - scaled_thickness * 0.5f) * n; // for edge 1 vertices

    // ---- per-vertex truncation ----
    // Edge 0 vertices approach when dot(n, delta) < -parallel_eps
    // Edge 1 vertices approach when dot(n, delta) >  parallel_eps
    // bool force_trunc = current_dist < thickness * 0.5f;

    // t_e0v0 = proj_e0v0 < -parallel_eps ?
    //              planar_truncation_t(e0v0, proj_e0v0, n, d0, gamma_min) : 1.f;
    // t_e0v1 = proj_e0v1 < -parallel_eps ?
    //              planar_truncation_t(e0v1, proj_e0v1, n, d0, gamma_min) : 1.f;
    // t_e1v0 = proj_e1v0 > parallel_eps ?
    //              planar_truncation_t(e1v0, proj_e1v0, n, d1, gamma_min) : 1.f;
    // t_e1v1 = proj_e1v1 > parallel_eps ?
    //              planar_truncation_t(e1v1, proj_e1v1, n, d1, gamma_min) : 1.f;
    t_e0v0 =
        planar_truncation_t(e0v0, proj_e0v0, n, d0, gamma_min);
    t_e0v1 =
        planar_truncation_t(e0v1, proj_e0v1, n, d0, gamma_min);
    t_e1v0 =
        planar_truncation_t(e1v0, proj_e1v0, n, d1, gamma_min);
    t_e1v1 =
        planar_truncation_t(e1v1, proj_e1v1, n, d1, gamma_min);
    return (t_e0v0 < 1.0f) || (t_e0v1 < 1.0f) || (t_e1v0 < 1.0f) || (t_e1v1 < 1.0f);
}

// ------------------------------------------------------------------
// Vertex-face planar truncation kernel (BVH traversal).
// ------------------------------------------------------------------
static __global__ void vf_collision_planar_truncation_bvh_kernel(
    float* __restrict__ truncation_t,
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const float3* __restrict__ pos_prev,
    const float3* __restrict__ pos_target,
    const float3* __restrict__ vertex_normals,
    const int3* __restrict__ faces,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int active_vertices_size,
    const float parallel_eps,
    const float gamma_r,
    const float gamma_min
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    if ( sorted_indices ) i = sorted_indices[i];

    float3 P0 = pos_prev[i];
    float3 P1 = pos_target[i];
    const auto& od = obj_data[vertices_obj[i]];
    float r_p = od.thickness;
    int layer_v = od.collision_layer;
    AABB q_aabb = {
        .min = fmin3(P0, P1) - r_p,
        .max = fmax3(P0, P1) + r_p,
    };

    bool is_active = i < active_vertices_size;
    float min_t_i = 1.0f;
    float3 vertex_normal = vertex_normals[i];
    // @formatter:off
    BVH_TRAVERSE_LOOP(q_aabb, 64, {
        int3 f = faces[prim_idx];
        if ( f.x == i || f.y == i || f.z == i ) continue;
        if (!is_active && f.x >= active_vertices_size) continue;

        float3 A0 = pos_prev[f.x], A1 = pos_target[f.x];
        float3 B0 = pos_prev[f.y], B1 = pos_target[f.y];
        float3 C0 = pos_prev[f.z], C1 = pos_target[f.z];
        const auto& od2 = obj_data[vertices_obj[f.x]];
        float r_tri = od2.thickness;

        float3 tri_cap_start, tri_cap_end;
        float  tri_cap_radius;
        triangle_trajectory_capsule(A0, B0, C0, A1, B1, C1,
                                    r_tri,
                                    tri_cap_start, tri_cap_end, tri_cap_radius);

        if ( !capsule_capsule_intersects(P0, P1, r_p,
                                         tri_cap_start, tri_cap_end, tri_cap_radius) )
            continue;
        // Planar truncation for this pair
        // Pairwise thickness: sum of vertex and representative triangle thickness
        float pair_thickness = r_p + r_tri;

        float t_v, t_a, t_b, t_c;
        bool truncated = vertex_triangle_planar_truncation(
            P0, P1 - P0,
            A0,  A1 - A0,
            B0, B1 - B0,
            C0, C1 - C0,
            pair_thickness,
            layer_v - od2.collision_layer, vertex_normal,
            parallel_eps, gamma_r, gamma_min,
            t_v, t_a, t_b, t_c);

        if ( truncated ) {
            // printf("v:%d, f:%d, t: %e,%e,%e,%e\n",i, prim_idx, t_v, t_a, t_b, t_c);
            // Update vertex i
            if ( t_v < min_t_i ) min_t_i = t_v;

            // Update triangle vertices (atomic because shared with other threads)
            if ( f.x < active_vertices_size ) {
                if (t_a < 1.f) atomicMinFloat( truncation_t + f.x, t_a );
                if (t_b < 1.f) atomicMinFloat( truncation_t + f.y, t_b );
                if (t_c < 1.f) atomicMinFloat( truncation_t + f.z, t_c );
            }
        }
    })
    // @formatter:on
    if ( is_active && min_t_i < 1.f ) {
        atomicMinFloat(truncation_t + i, min_t_i);
    }
}
// ------------------------------------------------------------------
// Edge-edge planar truncation kernel (BVH traversal).
// ------------------------------------------------------------------
static __global__ void ee_collision_planar_truncation_bvh_kernel(
    float* __restrict__ truncation_t,          // in/out: per-vertex truncation factors (initialized to 1)
    unsigned int num_queries,                  // number of active query edges
    const int2* __restrict__ nodes,            // BVH nodes
    const AABB* __restrict__ aabbs,            // BVH leaf aabbs
    unsigned int root_idx,                     // BVH root node index
    const float3* __restrict__ pos_prev,       // vertex positions at start of step
    const float3* __restrict__ pos_target,     // vertex positions at end of step (pos_prev + displacement)
    const float3* __restrict__ edge_normals,
    const int2* __restrict__ edges,            // edge indices (vertex pairs)
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int active_vertices_size,            // only vertices with index < this are active
    const float parallel_eps,                  // parallel motion tolerance
    const float gamma_r,
    const float gamma_min)                       // relaxation factor for thickness gap
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    AABB q_aabb = aabbs[i];
    i = nodes[i].x - 1; // prim_idx

    // --- query edge data ---
    int2 edge = edges[i];
    float3 e0v0 = pos_prev[edge.x];
    float3 e0v1 = pos_prev[edge.y];
    float3 e0v0_targ = pos_target[edge.x];
    float3 e0v1_targ = pos_target[edge.y];
    const auto& od = obj_data[vertices_obj[edge.x]];
    float3 e0_n = edge_normals[i];
    // edge thickness as max of its two vertices
    float thick0 = od.thickness;
    int layer_e0 = od.collision_layer;

    // query edge trajectory capsule
    float3 cap0_start, cap0_end;
    float cap0_radius;
    edge_trajectory_capsule(e0v0, e0v1, e0v0_targ, e0v1_targ,
        thick0, cap0_start, cap0_end, cap0_radius);

    // conservative AABB for BVH traversal
    // AABB q_aabb;
    // q_aabb.min = fmin3(fmin3(e0v0, e0v1), fmin3(e0v0_targ, e0v1_targ)) - thick0;
    // q_aabb.max = fmax3(fmax3(e0v0, e0v1), fmax3(e0v0_targ, e0v1_targ)) + thick0;

    bool is_active = edge.x < active_vertices_size;

    // local accumulation for edge vertices
    float min_t_v0 = 1.0f;
    float min_t_v1 = 1.0f;

    // @formatter:off
    BVH_TRAVERSE_LOOP(q_aabb, 64, {
        // skip self
        if (prim_idx == i) continue;

        int2 e = edges[prim_idx];
        // skip if edges share any vertex
        if (edge.x == e.x || edge.x == e.y || edge.y == e.x || edge.y == e.y) continue;
        // if both edges are entirely static, skip (optional)
        if (!is_active && e.x >= active_vertices_size) continue;

        float3 e1v0 = pos_prev[e.x];
        float3 e1v1 = pos_prev[e.y];
        float3 e1v0_targ = pos_target[e.x];
        float3 e1v1_targ = pos_target[e.y];
        const auto& od2 = obj_data[vertices_obj[e.x]];
        float thick1 = od2.thickness;

        // trajectory capsule for candidate edge
        float3 cap1_start, cap1_end;
        float  cap1_radius;
        edge_trajectory_capsule(e1v0, e1v1, e1v0_targ, e1v1_targ,
                                thick1, cap1_start, cap1_end, cap1_radius);

        // capsule-capsule cull
        if (!capsule_capsule_intersects(cap0_start, cap0_end, cap0_radius,
                                        cap1_start, cap1_end, cap1_radius))
            continue;

        // --- detailed planar truncation for this pair ---
        float3 delta_e0v0 = e0v0_targ - e0v0;
        float3 delta_e0v1 = e0v1_targ - e0v1;
        float3 delta_e1v0 = e1v0_targ - e1v0;
        float3 delta_e1v1 = e1v1_targ - e1v1;

        float pair_thickness = thick0 + thick1;

        float t0, t1, t2, t3;  // per-vertex factors
        bool truncated = edge_edge_planar_truncation(
            e0v0, delta_e0v0, e0v1, delta_e0v1,
            e1v0, delta_e1v0, e1v1, delta_e1v1,pair_thickness,
            layer_e0 - od2.collision_layer,
            e0_n,edge_normals[prim_idx],
            parallel_eps, gamma_r, gamma_min,
            t0, t1, t2, t3);

        if (truncated) {
            // printf("e1:%d (v%d,v%d), e2:%d (v%d,v%d), t: %e,%e,%e,%e\n",i,edge.x, edge.y, prim_idx,e.x, e.y,t0,t1,t2,t3);
            if (t0 < min_t_v0) min_t_v0 = t0;
            if (t1 < min_t_v1) min_t_v1 = t1;

            if (e.x < active_vertices_size) {
                if (t2 < 1.f) atomicMinFloat(truncation_t + e.x, t2);
                if (t3 < 1.f) atomicMinFloat(truncation_t + e.y, t3);
            }
        }
    });
    // @formatter:on

    // write back accumulated minima for the query edge
    if ( is_active ) {
        if ( min_t_v0 < 1.f ) atomicMinFloat(truncation_t + edge.x, min_t_v0);
        if ( min_t_v1 < 1.f ) atomicMinFloat(truncation_t + edge.y, min_t_v1);
    }
}
