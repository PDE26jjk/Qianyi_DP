#pragma once

#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/sort.h>

#include "geometry.cuh"
#include "common/math_utils.h"
#include "contact/hash.cuh"
#include "contact/collision_detection.cuh"

__device__ __forceinline__ float planar_truncation_t_debug(
    float3 p, float proj, float3 n, float3 d, float gamma_min, const char* vertex_name) {
    if ( fabsf(proj) < 1e-16f ) {
        printf("    %s proj=%e (parallel) -> t=1.0\n", vertex_name, proj);
        return 1.0f;
    }
    float t_raw = dot(n, d - p) / proj;
    float t;
    if ( t_raw < 0.0f ) {
        t = 1.0f;
    }
    else {
        t = fmaxf(0.f, fminf(1.0f, t_raw - gamma_min));
    }
    printf("    %s : p=(%f %f %f) proj=%e | d=(%f %f %f) | n=(%f %f %f) | t_raw=%e t=%e\n",
        vertex_name,
        p.x, p.y, p.z, proj,
        d.x, d.y, d.z,
        n.x, n.y, n.z,
        t_raw, t);
    return t;
}

__device__ inline bool edge_edge_planar_truncation_debug(
    float3 e0v0, float3 delta_e0v0,
    float3 e0v1, float3 delta_e0v1,
    float3 e1v0, float3 delta_e1v0,
    float3 e1v1, float3 delta_e1v1,
    float thickness,
    int layer_diff,
    float3 edge0_normal,
    float3 edge1_normal,
    float parallel_eps, float gamma_r, float gamma_min,
    float& t_e0v0, float& t_e0v1, float& t_e1v0, float& t_e1v1) {
    float s, t;
    float3 dP;
    segment_segment_closest_robust(e0v0, e0v1, e1v0, e1v1, s, t, dP);
    float3 u0 = e0v1 - e0v0;
    float3 v1 = e1v1 - e1v0;
    float3 c1 = e0v0 + s * u0;          // point on edge 0
    float3 c2 = e1v0 + t * v1;          // point on edge 1

    float3 n_hat = dP;             // identical to c1-c2
    float len_sq = dot(n_hat, n_hat);

    float scaled_thickness = thickness * gamma_r;

    printf("  closest: s=%e t=%e | c1=(%f %f %f) c2=(%f %f %f)\n",
        s, t, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z);

    if ( len_sq < 1e-18f ) {
        printf("  degenerate (len_sq=%e) -> skip\n", len_sq);
        return false;
    }

    // layer check only if layer_diff != 0
    if ( layer_diff != 0 ) {
        float3 low_normal = (layer_diff < 0) ? edge0_normal : edge1_normal;
        if ( dot(low_normal, n_hat) < 0.0f ) {
            printf("  layer skip (dot=%e)\n", dot(low_normal, n_hat));
            return false;
        }
    }

    float3 n = n_hat * rsqrtf(len_sq);

    float proj_e0v0 = dot(n, delta_e0v0);
    float proj_e0v1 = dot(n, delta_e0v1);
    float proj_e1v0 = dot(n, delta_e1v0);
    float proj_e1v1 = dot(n, delta_e1v1);

    float delta_e0 = fmaxf(fmaxf(-proj_e0v0, -proj_e0v1), 0.0f);
    float delta_e1 = fmaxf(fmaxf(proj_e1v0, proj_e1v1), 0.0f);

    float current_dist = sqrtf(len_sq);

    printf("  proj: e0v0=%e e0v1=%e e1v0=%e e1v1=%e\n", proj_e0v0, proj_e0v1, proj_e1v0, proj_e1v1);
    printf("  delta_e0=%e delta_e1=%e current_dist=%e\n", delta_e0, delta_e1, current_dist);


    // early exit if the gap cannot be closed
    // if ( delta_e0 + delta_e1 + thickness < current_dist ||  delta_e0 + delta_e1 == 0.0f ) {
    if ( current_dist > thickness * 3.0f && delta_e0 + delta_e1 == 0.0f ) {
        printf("  early exit (delta_e0 + delta_e1=%e, current_dist=%e)\n",
            delta_e0 + delta_e1, current_dist);
        return false;
    }
    float lmbd;
    if ( delta_e0 + delta_e1 == 0.0f )
        lmbd = 0.5f;
    else
        lmbd = delta_e1 / (delta_e0 + delta_e1);
    lmbd = fmaxf(0.05f, fminf(0.95f, lmbd));
    printf("  lmbd=%e\n", lmbd);

    float3 d0 = c2 + lmbd * n_hat + (scaled_thickness * 0.5f) * n;
    float3 d1 = c2 + lmbd * n_hat - (scaled_thickness * 0.5f) * n;

    printf("  d0=(%f %f %f)  d1=(%f %f %f)\n", d0.x, d0.y, d0.z, d1.x, d1.y, d1.z);
    // t_e0v0 = proj_e0v0 < -parallel_eps ?
    //              planar_truncation_t_debug(e0v0, proj_e0v0, n, d0, gamma_min, "e0v0") : 1.f;
    // t_e0v1 = proj_e0v1 < -parallel_eps ?
    //              planar_truncation_t_debug(e0v1, proj_e0v1, n, d0, gamma_min, "e0v1") : 1.f;
    // t_e1v0 = proj_e1v0 > parallel_eps ?
    //              planar_truncation_t_debug(e1v0, proj_e1v0, n, d1, gamma_min, "e1v0") : 1.f;
    // t_e1v1 = proj_e1v1 > parallel_eps ?
    //              planar_truncation_t_debug(e1v1, proj_e1v1, n, d1, gamma_min, "e1v1") : 1.f;
    t_e0v0 =
        planar_truncation_t_debug(e0v0, proj_e0v0, n, d0, gamma_min, "e0v0");
    t_e0v1 =
        planar_truncation_t_debug(e0v1, proj_e0v1, n, d0, gamma_min, "e0v1");
    t_e1v0 =
        planar_truncation_t_debug(e1v0, proj_e1v0, n, d1, gamma_min, "e1v0");
    t_e1v1 =
        planar_truncation_t_debug(e1v1, proj_e1v1, n, d1, gamma_min, "e1v1");

    bool any = (t_e0v0 < 1.0f) || (t_e0v1 < 1.0f) || (t_e1v0 < 1.0f) || (t_e1v1 < 1.0f);
    printf("  results: t_e0v0=%e t_e0v1=%e t_e1v0=%e t_e1v1=%e  any=%d\n",
        t_e0v0, t_e0v1, t_e1v0, t_e1v1, any);
    return any;
}

__device__ inline bool vertex_triangle_planar_truncation_debug(
    float3 v, float3 delta_v,
    float3 t0, float3 delta_t0,
    float3 t1, float3 delta_t1,
    float3 t2, float3 delta_t2,
    float thickness,
    int layer_diff,
    float3 vertex_normal,
    float parallel_eps, float gamma_r, float gamma_min,
    float& t_v, float& t_t0, float& t_t1, float& t_t2) {
    float3 closest_pt;
    float len_sq = point_triangle_sq_dist(v, t0, t1, t2, closest_pt);
    if ( len_sq < 1e-18f ) {
        printf("  vf degenerate (len_sq=%e) -> skip\n", len_sq);
        return false;
    }

    float3 n_hat = v - closest_pt;
    if ( layer_diff != 0 ) {
        float3 low_normal;
        if ( layer_diff < 0 ) {
            low_normal = vertex_normal;
        }
        else {
            low_normal = cross(t1 - t0, t2 - t0);
        }
        if ( dot(low_normal, n_hat) < 0.0f ) {
            printf("  vf layer skip (dot=%e)\n", dot(low_normal, n_hat));
            return false;
        }
    }

    float3 n = n_hat * rsqrtf(len_sq);

    float proj_v = dot(n, delta_v);
    float proj_t0 = dot(n, delta_t0);
    float proj_t1 = dot(n, delta_t1);
    float proj_t2 = dot(n, delta_t2);

    float delta_v_n = fmaxf(-proj_v, 0.0f);
    float delta_t_n = fmaxf(fmaxf(proj_t0, proj_t1), fmaxf(proj_t2, 0.0f));

    printf("  vf proj: v=%e t0=%e t1=%e t2=%e | delta_v_n=%e delta_t_n=%e\n",
        proj_v, proj_t0, proj_t1, proj_t2, delta_v_n, delta_t_n);

    float lmbd;
    // if ( delta_v_n + delta_t_n == 0.0f )
    if ( delta_v_n == 0.0f || delta_t_n == 0.0f )
        lmbd = 0.5f;
    else
        lmbd = delta_t_n / (delta_v_n + delta_t_n);
    lmbd = fmaxf(0.2f, fminf(0.8f, lmbd));
    float current_dist = sqrtf(len_sq);
    printf("  vf lmbd=%e | current_dist=%e\n", lmbd, current_dist);
    thickness *= gamma_r;
    float3 d = closest_pt + lmbd * n_hat + (thickness * 0.5f) * n;
    printf("  vf d_plane=(%f %f %f)\n", d.x, d.y, d.z);

    t_v = planar_truncation_t_debug(v, proj_v, n, d, gamma_min, "v");
    d -= thickness * n;
    t_t0 = planar_truncation_t_debug(t0, proj_t0, n, d, gamma_min, "t0");
    t_t1 = planar_truncation_t_debug(t1, proj_t1, n, d, gamma_min, "t1");
    t_t2 = planar_truncation_t_debug(t2, proj_t2, n, d, gamma_min, "t2");
    if ( delta_v_n == 0.0f ) {
        if ( dot(n, d - t0) < -1e-7f ) t_t0 = fminf(t_t0, 1e-6f);
        if ( dot(n, d - t1) < -1e-7f ) t_t1 = fminf(t_t1, 1e-6f);
        if ( dot(n, d - t2) < -1e-7f ) t_t2 = fminf(t_t2, 1e-6f);
    }

    bool any = (t_v < 1.0f) || (t_t0 < 1.0f) || (t_t1 < 1.0f) || (t_t2 < 1.0f);
    printf("  vf results: t_v=%e t_t0=%e t_t1=%e t_t2=%e any=%d\n", t_v, t_t0, t_t1, t_t2, any);
    return any;
}

static __global__ void check_ef_pairs_kernel(
    float3* __restrict__ debug_colors,
    float* __restrict__ truncation_t,
    const unsigned int* __restrict__ sorted_indices,
    const unsigned int* __restrict__ edge_ranks,
    const AABB*__restrict__ edge_aabbs,
    unsigned int num_queries,
    const int2*__restrict__ nodes,
    const AABB*__restrict__ aabbs,
    const int2*__restrict__ edges,
    const int3*__restrict__ tri_edges,
    const float3* __restrict__ pos_prev,
    const float3* __restrict__ pos_target,          // 已截断位置
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int3* __restrict__ tri_indices,
    unsigned int root_idx,
    const float parallel_eps,
    const float gamma_r,
    const float gamma_min,
    int* __restrict__ debug_lock
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    AABB q_aabb = edge_aabbs[i];
    i = sorted_indices[i];
    int2 edge = edges[i];
    float3 v0 = pos_target[edge.x], v1 = pos_target[edge.y];
    float3 v0_prev = pos_prev[edge.x], v1_prev = pos_prev[edge.y];
    // @formatter:off
    BVH_TRAVERSE_LOOP(q_aabb, 64,
        int3 tri = tri_indices[prim_idx];
        if (edge.x == tri.x || edge.x == tri.y || edge.x == tri.z ||
            edge.y == tri.x || edge.y == tri.y || edge.y == tri.z) continue;
        float3 x0 = pos_target[tri.x], x1 = pos_target[tri.y], x2 = pos_target[tri.z];
        float3 x0_prev = pos_prev[tri.x], x1_prev = pos_prev[tri.y], x2_prev = pos_prev[tri.z];

        if (ef_intersect(v0,v1,x0,x1,x2) &&
            !ef_intersect(v0_prev,v1_prev,x0_prev,x1_prev,x2_prev)) {
        debug_colors[edge.x] = make_float3(1.f, 1.f, 0.f);
        debug_colors[edge.y] = make_float3(1.f, 1.f, 0.f);
        debug_colors[ tri.x] = make_float3(1.f, 1.f, 0.f);
        debug_colors[ tri.y] = make_float3(1.f, 1.f, 0.f);
        debug_colors[ tri.z] = make_float3(1.f, 1.f, 0.f);

        // 原子锁：仅打印一次
        if (atomicCAS(debug_lock, 0, 1) == 0) {
        debug_colors[edge.x] = make_float3(1.f, 0.f, 0.f);
        debug_colors[edge.y] = make_float3(1.f, 0.f, 0.f);
        printf("=== EF INTERSECTION DETECTED ===\n");
        printf("Edge %d (v%d,v%d): curr=(%f %f %f) - (%f %f %f)\n",
            i, edge.x, edge.y, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
        printf("Triangle %d (v%d,v%d,v%d): curr=(%f %f %f) (%f %f %f) (%f %f %f)\n",
            prim_idx, tri.x, tri.y, tri.z, x0.x, x0.y, x0.z, x1.x, x1.y, x1.z, x2.x, x2.y, x2.z);

        // 恢复原始目标位置
        auto recover = [&](int vid, float3 prev) -> float3 {
        float tr = truncation_t[vid];
        if (tr > 1e-10f) {
        return prev + (pos_target[vid] - prev) / tr;
        } else {
        return prev; // 位移为零
        }
        };

        float3 e0v0_orig = recover(edge.x, pos_prev[edge.x]);
        float3 e0v1_orig = recover(edge.y, pos_prev[edge.y]);
        float3 t0_orig = recover(tri.x, pos_prev[tri.x]);
        float3 t1_orig = recover(tri.y, pos_prev[tri.y]);
        float3 t2_orig = recover(tri.z, pos_prev[tri.z]);

        float3 delta_e0v0 = e0v0_orig - pos_prev[edge.x];
        float3 delta_e0v1 = e0v1_orig - pos_prev[edge.y];
        float3 delta_t0 = t0_orig - pos_prev[tri.x];
        float3 delta_t1 = t1_orig - pos_prev[tri.y];
        float3 delta_t2 = t2_orig - pos_prev[tri.z];

        printf("Original targets (pre-trunc):\n");
        printf("  e0v0: (%f %f %f) t=%e\n", e0v0_orig.x, e0v0_orig.y, e0v0_orig.z, truncation_t[edge.x]);
        printf("  e0v1: (%f %f %f) t=%e\n", e0v1_orig.x, e0v1_orig.y, e0v1_orig.z, truncation_t[edge.y]);
        printf("  tri0: (%f %f %f) t=%e\n", t0_orig.x, t0_orig.y, t0_orig.z, truncation_t[tri.x]);
        printf("  tri1: (%f %f %f) t=%e\n", t1_orig.x, t1_orig.y, t1_orig.z, truncation_t[tri.y]);
        printf("  tri2: (%f %f %f) t=%e\n", t2_orig.x, t2_orig.y, t2_orig.z, truncation_t[tri.z]);

        // 只检查最接近的那条三角形边
        int3 t2e = tri_edges[prim_idx];
        int2 tri_edges_arr[3] = { edges[t2e.x], edges[t2e.y], edges[t2e.z] };
        float min_dist_sq = FLT_MAX;
        int closest_k = -1;
        for (int k = 0; k < 3; k++) {
        int vA = tri_edges_arr[k].x, vB = tri_edges_arr[k].y;
        float dist_sq = segment_segment_dist_sq_robust(
            pos_prev[edge.x], pos_prev[edge.y], pos_prev[vA], pos_prev[vB]);
        if (dist_sq < min_dist_sq) {
        min_dist_sq = dist_sq;
        closest_k = k;
        }
        }

            // 对最接近的边进行 debug 截断
            int vA = tri_edges_arr[closest_k].x, vB = tri_edges_arr[closest_k].y;
            int E;
            if (closest_k == 0)
                E = t2e.x;
            else if (closest_k == 1)
                E = t2e.y;
            else
                E = t2e.z;
            printf("E %d vA: %d,vB: %d\n", E,vA,vB);
            // if (edge_nodes[edge_ranks[E]].x-1 == E) {
            //     printf("indices right\n");
            // }else {
            //     printf("indices not right!!!!!!\n");
            // }
            if (aabb_overlap_3d(q_aabb, edge_aabbs[edge_ranks[E]])) {
                printf("AABB overlaps pass\n");
            }else {
                printf("AABB overlaps not pass!!!!!!\n");
            }
            
            
        float3 delta_A = (vA == tri.x) ? delta_t0 : ((vA == tri.y) ? delta_t1 : delta_t2);
        float3 delta_B = (vB == tri.x) ? delta_t0 : ((vB == tri.y) ? delta_t1 : delta_t2);

        // 厚度：取两个顶点厚度的最大值（简化）或和
        float thick_edge0 = fmaxf(obj_data[vertices_obj[edge.x]].thickness,
            obj_data[vertices_obj[edge.y]].thickness);
        float thick_edge1 = fmaxf(obj_data[vertices_obj[vA]].thickness,
            obj_data[vertices_obj[vB]].thickness);
        float pair_thickness = thick_edge0 + thick_edge1;

        float t_v0 = 1.f, t_v1 = 1.f, t_A = 1.f, t_B = 1.f;
        bool tructed = edge_edge_planar_truncation_debug(
            pos_prev[edge.x], delta_e0v0,
            pos_prev[edge.y], delta_e0v1,
            pos_prev[vA], delta_A,
            pos_prev[vB], delta_B,
            pair_thickness,
            0,                            // layer_diff = 0
            make_float3(0,0,0), make_float3(0,0,0),
            parallel_eps, gamma_r, gamma_min,
            t_v0, t_v1, t_A, t_B);


        float new_t_ev0 = fminf(t_v0, truncation_t[edge.x]);
        float new_t_ev1 = fminf(t_v1, truncation_t[edge.y]);
        float new_t_A = fminf(t_A, truncation_t[vA]);
        float new_t_B = fminf(t_B, truncation_t[vB]);

        // 三角形第三个顶点（非 vA, vB）保持原截断因子
        int third_vertex = -1;
        for (int k = 0; k < 3; k++) {
            if (tri_edges[k].x != vA || tri_edges[k].y != vB) {
                // 找到那个不在这条边上的顶点
                int cand = (tri.x != vA && tri.x != vB) ? tri.x :
                (tri.y != vA && tri.y != vB) ? tri.y : tri.z;
                third_vertex = cand;
                break;
            }
        }
        float new_t_third = truncation_t[third_vertex]; // 保持原值

        // 计算新目标位置
        float3 new_e0v0 = pos_prev[edge.x] + delta_e0v0 * fminf(new_t_ev0, 1.0f);
        float3 new_e0v1 = pos_prev[edge.y] + delta_e0v1 * fminf(new_t_ev1, 1.0f);
        float3 new_triA = pos_prev[vA] + delta_A * fminf(new_t_A, 1.0f);
        float3 new_triB = pos_prev[vB] + delta_B * fminf(new_t_B, 1.0f);
        // 第三个顶点：获取其原始位移
        float3 delta_third = (third_vertex == tri.x) ? delta_t0 :
        (third_vertex == tri.y) ? delta_t1 : delta_t2;
        float3 new_triThird = pos_prev[third_vertex] + delta_third * fminf(new_t_third, 1.0f);

        printf("\nAfter applying EE truncation factors:\n");
        printf("  new edge: (%f %f %f) - (%f %f %f)\n",
            new_e0v0.x, new_e0v0.y, new_e0v0.z, new_e0v1.x, new_e0v1.y, new_e0v1.z);
        printf("  new tri:  (%f %f %f) (%f %f %f) (%f %f %f)\n",
            new_triA.x, new_triA.y, new_triA.z,
            new_triB.x, new_triB.y, new_triB.z,
            new_triThird.x, new_triThird.y, new_triThird.z);

        // 重新检测边-面相交
        bool still_intersect = ef_intersect(new_e0v0, new_e0v1, new_triA, new_triB, new_triThird);
        printf("Edge-face intersection after EE truncation: %s\n\n",
            still_intersect ? "STILL INTERSECTING" : "ELIMINATED");
        if (still_intersect) {
            // ---------- 增加点‑面截断调试 ----------
            float3 tri_normal = normalized( cross(pos_prev[tri.y] - pos_prev[tri.x],
                           pos_prev[tri.z] - pos_prev[tri.x]));

            float d0 = fabsf(dot(tri_normal, pos_prev[edge.x] - pos_prev[tri.x]));
            float d1 = fabsf(dot(tri_normal, pos_prev[edge.y] - pos_prev[tri.x]));

            float3 closest_pt_on_edge;
            float3 delta_vf;
            int edge_vid_for_vf;
            if (d0 <= d1) {
                closest_pt_on_edge = pos_prev[edge.x];
                delta_vf = delta_e0v0;
                edge_vid_for_vf = edge.x;
            } else {
                closest_pt_on_edge = pos_prev[edge.y];
                delta_vf = delta_e0v1;
                edge_vid_for_vf = edge.y;
            }
            float3 P0 = closest_pt_on_edge;
            float3 P1 = closest_pt_on_edge + delta_vf;
            float r_p = obj_data[vertices_obj[edge.x]].thickness;
            AABB v_aabb = {
                .min = fmin3(P0, P1) - r_p,
                .max = fmax3(P0, P1) + r_p,
            };
            if (aabb_overlap_3d(v_aabb, aabbs[node_idx])) {
                printf("vf AABB overlaps pass\n");
            }else {
                printf("vf AABB overlaps not pass!!!!!!\n");
            }
            float3 tri_cap_start, tri_cap_end;
            float  tri_cap_radius;
            float3 A0 = pos_prev[tri.x],B0 = pos_prev[tri.y],C0 = pos_prev[tri.z];
            triangle_trajectory_capsule(A0, B0, C0,
                A0 + delta_t0, B0 + delta_t1, C0 + delta_t2,
                thick_edge1,
                tri_cap_start, tri_cap_end, tri_cap_radius);

            if ( capsule_capsule_intersects(P0, P1, r_p,
                                             tri_cap_start, tri_cap_end, tri_cap_radius) )
                printf("vf capsule overlaps pass\n");
            else
                printf("vf capsule overlaps not pass!!!!!!\n");

            // 点‑面厚度：取边端点厚度与三角形厚度的和
            float vf_thickness = r_p + thick_edge1;

            float t_vf =1.f, t_tri0_vf=1.f, t_tri1_vf=1.f, t_tri2_vf=1.f;
            printf("\n=== Debug VF truncation (point on edge vs triangle) ===\n");
            bool vf_trunc = vertex_triangle_planar_truncation_debug(
                P0, delta_vf,
                A0, delta_t0,
                B0, delta_t1,
                C0, delta_t2,
                vf_thickness,
                0,                               // layer_diff = 0
                make_float3(0,0,0),              // vertex_normal unused
                parallel_eps, gamma_r, gamma_min,
                t_vf, t_tri0_vf, t_tri1_vf, t_tri2_vf);

            // 合并截断因子：对于每个顶点，取 min(边‑边截断, 点‑面截断, 全局)
            float eff_t_v0, eff_t_v1;
            if (edge_vid_for_vf == edge.x) {
                eff_t_v0 = fminf(fminf(t_v0, t_vf), truncation_t[edge.x]);  
                eff_t_v1 = fminf(t_v1, truncation_t[edge.y]);             
            } else {
                eff_t_v0 = fminf(t_v0, truncation_t[edge.x]);
                eff_t_v1 = fminf(fminf(t_v1, t_vf), truncation_t[edge.y]);
            }
            float eff_t_tri0 = fminf( t_tri0_vf, truncation_t[tri.x]);
            float eff_t_tri1 = fminf(t_tri1_vf, truncation_t[tri.y]);
            float eff_t_tri2 = fminf(t_tri2_vf, truncation_t[tri.z]);

            // 重新计算新位置
            float3 new_e0v0_vf = pos_prev[edge.x] + delta_e0v0 * eff_t_v0;
            float3 new_e0v1_vf = pos_prev[edge.y] + delta_e0v1 * eff_t_v1;
            float3 new_t0_vf    = pos_prev[tri.x] + delta_t0 * eff_t_tri0;
            float3 new_t1_vf    = pos_prev[tri.y] + delta_t1 * eff_t_tri1;
            float3 new_t2_vf    = pos_prev[tri.z] + delta_t2 * eff_t_tri2;

            bool still_vf = ef_intersect(new_e0v0_vf, new_e0v1_vf, new_t0_vf, new_t1_vf, new_t2_vf);
            printf("Edge-face intersection after VF+EE truncation: %s\n\n",
                   still_vf ? "STILL INTERSECTING" : "ELIMINATED");
                        }
                    }
        }
        );
    // @formatter:on
}
