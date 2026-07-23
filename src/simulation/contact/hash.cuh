#pragma once
// Deprecated, just a backup

#include "common/cuda_utils.h"
#include <device_launch_parameters.h>


#include "collision.cuh"
#include "common/atomic_utils.cuh"
#include "common/geometric_algorithms.h"
#include "collision_type.cuh"

static __global__ void record_edge_hashes(
    int2* out_pairs,
    int* sort_keys,
    int* out_count,
    const int2* edges,
    const float3* vertices,
    float cell_size,
    int max_results_size,
    int hash_table_size,
    int num_edges
) {
    int e_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( e_idx >= num_edges ) return;
    if ( *out_count >= max_results_size ) return;
    int2 e = edges[e_idx];
    float3 v0 = vertices[e.x];
    float3 v1 = vertices[e.y];
    float3 dir = normalized(v1 - v0);

    float3 min_p = fmin3(v0, v1);
    float3 max_p = fmax3(v0, v1);
    const float cell_size_inv = 1.f / cell_size;
    int3 min_g = make_int3(floorf(min_p.x * cell_size_inv), floorf(min_p.y * cell_size_inv), floorf(min_p.z * cell_size_inv));
    int3 max_g = make_int3(floorf(max_p.x * cell_size_inv), floorf(max_p.y * cell_size_inv),
        floorf(max_p.z * cell_size_inv));
    const float center2corner = cell_size * 0.866025f; // 3^0.5 / 2
    const float padding = cell_size * 0.2f;
    int3 range = (max_g - min_g);
    int range_val = range.x * range.y * range.z;
    if ( range_val > 200 || range_val < 0 ) {
        printf("ERROR: Very long edge! edge:%d, "
            "cells = %d\n, [(%f,%f,%f),(%f,%f,%f)]\n",
            e_idx, range_val, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z);
        return;
    }
    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {
                float3 center = make_float3(x + 0.5f, y + 0.5f, z + 0.5f) * cell_size;
                float dist = norm(center - v0 - dir * dot(center - v0, dir));
                if ( dist > center2corner + padding ) continue;

                int idx = atomicAdd(out_count, 1);
                if ( idx < max_results_size ) {
                    int h = get_hash(make_int3(x, y, z), hash_table_size);
                    out_pairs[idx] = make_int2(h, e_idx);
                    sort_keys[idx] = h;
                }
                else return;
            }
        }
    }
}

__device__ __forceinline__ float segment_segment_sq_dist(
    const float3 p1, const float3 q1,
    const float3 p2, const float3 q2,
    float& s, float& t, float3& t_ee) {
    float3 d1 = make_float3(q1.x - p1.x, q1.y - p1.y, q1.z - p1.z); // q1 - p1
    float3 d2 = make_float3(q2.x - p2.x, q2.y - p2.y, q2.z - p2.z); // q2 - p2
    float3 r = make_float3(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z); // p1 - p2

    float a = dot(d1, d1); // 线段 A 长度的平方
    float e = dot(d2, d2); // 线段 B 长度的平方
    float f = dot(d2, r);

    const float EPSILON = 1e-8f;

    // 检查线段是否退化为点
    if ( a <= EPSILON && e <= EPSILON ) {
        // 两条线段都退化成了点
        s = t = 0.0f;
        t_ee = r;
        return dot(t_ee, t_ee);
    }

    if ( a <= EPSILON ) {
        // 第一条线段退化为点
        s = 0.0f;
        t = f / e;
        t = fmaxf(0.0f, fminf(1.0f, t)); // 截断到 [0, 1]
    }
    else {
        float c = dot(d1, r);
        if ( e <= EPSILON ) {
            // 第二条线段退化为点
            t = 0.0f;
            s = fmaxf(0.0f, fminf(1.0f, -c / a));
        }
        else {
            // 一般情况（两条线段都不退化）
            float b = dot(d1, d2);
            float denom = a * e - b * b; // 必然 >= 0

            // 如果线段不平行，计算 L1 到 L2 的最近点参数 s，并截断
            if ( denom != 0.0f ) {
                s = fmaxf(0.0f, fminf(1.0f, (b * f - c * e) / denom));
            }
            else {
                // 两线段平行，任选一个 s 即可（这里选 0）
                s = 0.0f;
            }
            // 根据计算出的 s，求 L2 上的参数 t
            t = (b * s + f) / e;
            // 如果 t 落在 [0, 1] 之外，需要将 t 截断，然后反过来重新计算 s
            if ( t < 0.0f ) {
                t = 0.0f;
                s = fmaxf(0.0f, fminf(1.0f, -c / a));
            }
            else if ( t > 1.0f ) {
                t = 1.0f;
                s = fmaxf(0.0f, fminf(1.0f, (b - c) / a));
            }
        }
    }

    // 计算两条线段上的最近点
    float3 closest_A = make_float3(p1.x + d1.x * s, p1.y + d1.y * s, p1.z + d1.z * s);
    float3 closest_B = make_float3(p2.x + d2.x * t, p2.y + d2.y * t, p2.z + d2.z * t);

    // 计算连接向量 t_ee 和 平方距离
    t_ee = make_float3(closest_A.x - closest_B.x,
        closest_A.y - closest_B.y,
        closest_A.z - closest_B.z);

    return dot(t_ee, t_ee);
}

static __global__ void detect_edge_edge_constraints(
    UnifiedNormalConstraint* results,
    int* results_size,
    const int2* edges,
    const float3* vertices,
    const int2* edge_hashes,
    const int* hash_lookup,
    const char* mask,
    float cell_size,
    float dHat,
    int max_results,
    int hash_table_size,
    int hashes_size,
    int cloth_vertices_size,
    int num_edges
) {
    int edgeA_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( edgeA_idx >= num_edges ) return;

    int2 eA = edges[edgeA_idx];
    int a0 = eA.x;
    int a1 = eA.y;
    bool is_cloth = a0 < cloth_vertices_size;
    if ( is_cloth ) return; // TODO
    float3 x_a0 = vertices[a0];
    float3 x_a1 = vertices[a1];
    float3 dir = normalized(x_a1 - x_a0);

    float cell_size_inv = 1.0f / cell_size;
    float dHat_sq = dHat * dHat;
    const float SMALL_NUM = 1e-5f;
    bool fixed = a0 >= cloth_vertices_size || (mask[a0] && mask[a1]);
    // float3 min_p = fmin3(x_a0, x_a1);
    // float3 max_p = fmax3(x_a0, x_a1);

    // int3 min_g = make_int3(floorf(min_p.x * cell_size_inv), floorf(min_p.y * cell_size_inv), floorf(min_p.z * cell_size_inv));
    // int3 max_g = make_int3(floorf(max_p.x * cell_size_inv) + 1, floorf(max_p.y * cell_size_inv) + 1,
    //     floorf(max_p.z * cell_size_inv) + 1);
    const float padding = dHat * 2;
    float3 min_p = fmin3(x_a0, x_a1) - make_float3(padding, padding, padding);
    float3 max_p = fmax3(x_a0, x_a1) + make_float3(padding, padding, padding);

    int3 min_g = make_int3(floorf(min_p.x * cell_size_inv), floorf(min_p.y * cell_size_inv), floorf(min_p.z * cell_size_inv));
    int3 max_g = make_int3(floorf(max_p.x * cell_size_inv), floorf(max_p.y * cell_size_inv), floorf(max_p.z * cell_size_inv));
    const float center2corner = cell_size * 0.866025f; // 3^0.5 / 2
    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {
                float3 center = make_float3(x + 0.5f, y + 0.5f, z + 0.5f) * cell_size;
                float dist = norm(center - x_a0 - dir * dot(center - x_a0, dir));
                if ( dist > center2corner + padding ) continue;

                int h = get_hash(make_int3(x, y, z), hash_table_size);
                for ( int k = hash_lookup[h];
                      0 <= k && k < hashes_size && edge_hashes[k].x == h; k++ ) {
                    int edgeB_idx = edge_hashes[k].y;

                    // 1. 去重规则1：只判断单向，防止 A测B 和 B测A 重复
                    if ( edgeB_idx >= edgeA_idx ) continue;

                    int2 eB = edges[edgeB_idx];
                    int b0 = eB.x;
                    int b1 = eB.y;
                    // printf("hash_lookup[h]: %d\n", hash_lookup[h]);
                    if ( fixed && mask[b0] && mask[b1] ) continue;

                    // 2. 拓扑排除：舍弃共用顶点的边（相邻边）
                    if ( a0 == b0 || a0 == b1 || a1 == b0 || a1 == b1 ) continue;

                    float3 x_b0 = vertices[b0];
                    float3 x_b1 = vertices[b1];

                    float3 t_ee; // 连接两最近点的向量
                    float sc, tc; //  [0, 1] Interpolation coefficients instead of weights
                    // segment_segment_sq_dist 计算返回 sc, tc 和方向向量 t_ee
                    float dist_sq = segment_segment_sq_dist(x_a0, x_a1, x_b0, x_b1, sc, tc, t_ee);
                    // printf("dist_sq: %f, %f\n",dist_sq, dHat_sq);
                    if ( dist_sq < dHat_sq && dist_sq > SMALL_NUM * SMALL_NUM ) {

                        // 计算接触中心点
                        float3 ptA = x_a0 + (x_a1 - x_a0) * sc;
                        float3 ptB = x_b0 + (x_b1 - x_b0) * tc;
                        float3 contact_center = (ptA + ptB) * 0.5f;

                        // 如果接触点不在当前正在遍历的格子内，说明它在别的格子里已经被/会被处理，跳过！
                        int contact_hash = get_hash(make_int3(
                            floorf(contact_center.x * cell_size_inv),
                            floorf(contact_center.y * cell_size_inv),
                            floorf(contact_center.z * cell_size_inv)), hash_table_size);

                        if ( contact_hash != h ) continue;

                        // int type_index = 3; //  EE
                        //
                        // if ( sc < SMALL_NUM ) {
                        //     if ( tc < SMALL_NUM ) type_index = 0; // a0 - b0 (PP)
                        //     else if ( tc > 1.0f - SMALL_NUM ) type_index = 0; // a0 - b1 (PP)
                        //     else type_index = 1; // a0 - eB (PE)
                        // }
                        // else if ( sc > 1.0f - SMALL_NUM ) {
                        //     if ( tc < SMALL_NUM ) type_index = 0; // a1 - b0 (PP)
                        //     else if ( tc > 1.0f - SMALL_NUM ) type_index = 0; // a1 - b1 (PP)
                        //     else type_index = 1; // a1 - eB (PE)
                        // }
                        // else {
                        //     if ( tc < SMALL_NUM ) type_index = 1; // b0 - eA (PE)
                        //     else if ( tc > 1.0f - SMALL_NUM ) type_index = 1; // b1 - eA (PE)
                        // }

                        int res_idx = atomicAdd(results_size, 1);
                        if ( res_idx < max_results ) {
                            UnifiedNormalConstraint res{
                                {b0, b1, a0, a1, },
                                // { sc - 1.0f, -sc, 1.0f - tc, tc }
                                // normal is A <-- B
                                { tc - 1.f, -tc , 1.f - sc, sc,}
                            };
                            results[res_idx] = res;
                        }
                    }
                }
            }
        }
    }
}

static __global__ void triangles_query_points_by_point_hash(
    UnifiedNormalConstraint* results,
    float3* debug_color,
    int* results_size,
    const int3* __restrict__ indices,
    const float3* __restrict__ vertices,
    const int2* __restrict__ points_hashes,
    const int* __restrict__ hash_lookup,
    const char* __restrict__ mask,
    float cell_size,
    float query_dist_sq,
    int hash_table_size,
    int points_hashes_size,
    int max_results_size,
    int cloth_vertices_size,
    int num_triangles
) {
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tri_idx >= num_triangles ) return;

    int3 idx3 = indices[tri_idx];
    bool is_cloth = idx3.x < cloth_vertices_size;
    if ( is_cloth ) return; // TODO
    // if (idx3.x > num || idx3.y > num || idx3.z > num || idx3.x <cloth_vertices_size || idx3.y < cloth_vertices_size || idx3.z < cloth_vertices_size ) {
    //     printf("ERROR: [%d,%d,%d]\n",idx3.x,idx3.y,idx3.z);
    //     return;
    // }
    float3 v0 = vertices[idx3.x], v1 = vertices[idx3.y], v2 = vertices[idx3.z];
    float3 n_curr_vec = cross(v1 - v0, v2 - v0);
    float area_sq = len_sq(n_curr_vec);
    if ( area_sq < 1e-10f ) {
        // printf("ERROR: tri:%d, n_curr = %f\n, [(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)]\n",
        //     tri_idx, norm(n_curr), v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
        return;
    }
    float3 n_curr = n_curr_vec * rsqrtf(area_sq);

    float cell_size_inv = 1.0f / cell_size;
    float padding = sqrtf(query_dist_sq);
    float3 min_p = fmin3(v0, fmin3(v1, v2)) - make_float3(padding, padding, padding);
    float3 max_p = fmax3(v0, fmax3(v1, v2)) + make_float3(padding, padding, padding);

    int3 min_g = make_int3(floorf(min_p.x * cell_size_inv), floorf(min_p.y * cell_size_inv),
        floorf(min_p.z * cell_size_inv));
    int3 max_g = make_int3(floorf(max_p.x * cell_size_inv), floorf(max_p.y * cell_size_inv),
        floorf(max_p.z * cell_size_inv));
    int3 range = (max_g - min_g);
    int range_val = range.x * range.y * range.z;
    if ( range_val > 200 || range_val < 0 ) {
        printf("ERROR: Very big triangle! tri:%d, "
            "cells = %d\n, [(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)]\n",
            tri_idx, range_val, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
        return;
    }
    const float center2corner = cell_size * 0.866025f; // 3^0.5 / 2
    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {
                float3 center = make_float3(x + 0.5f, y + 0.5f, z + 0.5f) * cell_size;
                float dist = abs(dot(center - v0, n_curr));
                // printf("dist = %f / %f \n", dist, center2corner + padding);
                if ( dist > center2corner + padding ) continue;

                int h = get_hash(make_int3(x, y, z), hash_table_size);
                // if (h < 0 || h >= hash_table_size) {
                //     printf("ERROR: hash:%d\n", h);
                //     continue;
                // }
                // continue;//err
                if ( hash_lookup[h] == -1 ) continue;

                for ( int k = hash_lookup[h];
                      0 <= k && k < points_hashes_size && points_hashes[k].x == h; k++ ) {
                    int p_idx = points_hashes[k].y;
                    if ( p_idx == idx3.x || p_idx == idx3.y || p_idx == idx3.z ) continue;
                    if ( !is_cloth && mask[p_idx] ) continue; // Both are fixed.
                    float3 pos = vertices[p_idx];
                    // int grid_h = get_hash(make_int3(floorf(pos.x * cell_size_inv), floorf(pos.y * cell_size_inv),
                    //     floorf(pos.z * cell_size_inv)), hash_table_size);
                    // if ( grid_h != h ) continue;

                    dist = dot(pos - v0, n_curr);
                    float dist_sq = dist * dist;
                    // printf("dist_sq = %f / %f \n", dist_sq, query_dist_sq);

                    if ( dist_sq < query_dist_sq ) {
                        float3 closest_pt = pos - dist * n_curr;
                        float u, v, w;
                        barycentric(v0, v1, v2, closest_pt, u, v, w); 
                        // if (isnan(u) || isnan(v) || isnan(w) || isinf(u) || isinf(v) || isinf(w)) {
                        //     printf("nan in barycentric!!!\n");
                        //     continue;
                        // }
                        const float eps = -1e-2f;
                        const float one_eps = 1.f - eps;
                        if ( u < eps || v < eps || w < eps
                            || u > one_eps || v > one_eps || w > one_eps )
                            continue;
                        int tp_idx = atomicAdd(results_size, 1);
                        // printf("%f,%f [%d,%d]\n",dist_sq , query_dist_sq,p_idx,tri_idx);
                        // printf("tri:%d,%f,  [(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)],[(%f,%f,%f)][(%f,%f,%f)]\n",
                        //         tri_idx,dist_sq,  v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z,pos.x,pos.y,pos.z,u, v, w);
                        if ( tp_idx < max_results_size ) {
                            if (debug_color)
                                debug_color[p_idx] = make_float3(0.0f, 0.0f, 1.0f);
                            auto tp = UnifiedNormalConstraint{
                                { p_idx, idx3.x, idx3.y, idx3.z },
                                { 1.f, -u, -v, -w },
                                n_curr
                            };
                            results[tp_idx] = tp;
                        }
                    }
                }
            }
        }
    }
}

static __global__ void points_query_points_by_point_hash(
    UnifiedNormalConstraint* results,
    int* results_size,
    float3* debug_colors,
    const int2* __restrict__ points_hashes,
    const float3* __restrict__ points,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    const int* __restrict__ hash_lookup,
    const int* __restrict__ proxy,
    const char* __restrict__ mask,
    float cell_size,
    float max_dist,
    int hash_table_size,
    int points_hashes_size,
    int max_results_size,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= n ) return;
    if ( proxy[idx] != idx || mask[idx] ) return;

    float cell_size_inv = 1.0f / cell_size;
    float3 p = points[idx];
    int3 grid_idx = make_int3(floorf(p.x * cell_size_inv), floorf(p.y * cell_size_inv),
        floorf(p.z * cell_size_inv));
    // int centroid_hash = get_hash(grid_idx, hash_table_size);
    // if ( centroid_hash != grid_hash ) return;// not centroid
    float dist_sq = max_dist * max_dist;
    for ( int x = -1; x <= 1; x++ )
        for ( int y = -1; y <= 1; y++ )
            for ( int z = -1; z <= 1; z++ ) {
                int h = get_hash(grid_idx + make_int3(x, y, z), hash_table_size);
                if ( hash_lookup[h] == -1 ) continue;
                // printf("hash: %d,%d",h,hash_lookup[h]);

                for ( int k = hash_lookup[h];
                      0 <= k && k < points_hashes_size && points_hashes[k].x == h; k++ ) {
                    int p_idx = points_hashes[k].y;
                    if ( idx <= p_idx ) continue;
                    float3 p2 = points[p_idx];
                    // printf("dist_sq: %f, %f\n",len_sq(p - p2), dist_sq);
                    int grid_h = get_hash(make_int3(floorf(p2.x * cell_size_inv), floorf(p2.y * cell_size_inv),
                        floorf(p2.z * cell_size_inv)), hash_table_size);
                    if ( grid_h != h ) continue;
                    if ( len_sq(p - p2) < dist_sq ) {
                        if ( has_edge(idx, p_idx, edge_lookup, dir_edges) ) continue;
                        int res_id = atomicAdd(results_size, 1);
                        if ( res_id < max_results_size ) {
                            results[res_id] = UnifiedNormalConstraint{
                                { idx, p_idx, p_idx, p_idx },
                                { 1.f, -1.f, 0.f, 0.f },
                            };
                            if ( debug_colors ) {
                                debug_colors[idx] = make_float3(1.f, 0.f, 0.f);
                                debug_colors[p_idx] = make_float3(1.f, 0.f, 0.f);
                            }
                        }
                    }
                }
            }
}

// void Contact::contact_handle() {
//     int block = 256;
//     int n = (int)point_hash_table_size;
//     clear_hash_table<<<(n + block - 1) / block, block>>>(
//         point_hash_table.data().get(), n);
//     n = params.nb_all_cloth_vertices;
//     // 1. collect pp
//     float cell_size = params.cloth_edge_mean_length * 1.414f;
//     // const float cell_size = 5.f * 0.001f;
//     insert_points_to_grid<<<(n + block - 1) / block, block>>>(
//         vertices_world.data().get(),
//         point_hash_table.data().get(),
//         cell_size, point_hash_table_size,
//         n);
//
//     const float dist = 5.f * 0.001f;
//     pp_result_size.assign(1, 0);
//     collect_pp<<<(n + block - 1) / block, block>>>(
//         pp_collision_result.data().get(),
//         pp_result_size.data().get(),
//         // sort_key_temp.data().get(),
//         // sort_value_temp.data().get(),
//         // sort_result_size.data().get(),
//         vertices_world.data().get(),
//         point_hash_table.data().get(),
//         dist * dist, n,
//         max_pp_result_size,
//         point_hash_table_size,
//         cell_size);
//     // 2. graph coloring
//     int result_size;
//     cudaMemcpy(&result_size, sort_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//     result_size = min(result_size, max_pp_result_size);
// }
//

// void Contact::collision_LCP_postprocess(float3* points_y) {
//     START_TIMER;
//     int block = 256;
//     int n = (int)point_hash_table_size;
//     clear_hash_table<<<(n + block - 1) / block, block>>>(
//         point_hash_table.data().get(), n);
//     int cloth_vertex_size = params.nb_all_cloth_vertices;
//     n = cloth_vertex_size;
//     // 1. collect pp
//     float max_dist = params.cloth_edge_mean_length;
//
//     float3* points_x = vertices_old.data().get();
//     collision_collect_near_pairs(points_x, max_dist, true, false, true, true);
//     // tp_result_size_h = 0;
//     RECORD_TIME("collision_collect_near_pairs");
//
//
//     debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
//     int num_constraints = tp_result_size_h + ee_result_size_h;
//     if ( num_constraints > 0 ) {
//         int all_vertex_size = params.nb_all_vertices;
//         float3* points_collision = temp_vertices_f3.data().get();
//         cudaMemcpyAsync(points_collision, points_y, all_vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
//         // std::cout <<  result_size << " triangles" << std::endl; 
//         // compute_collision_penalty_force_triangle_point_plane<<<(result_size + block - 1) / block, block>>>(
//         //     // Jx.data().get(),
//         //     Jx_diag.data().get(),
//         //     forces.data().get(),
//         //     tp_collision_result.data().get(),
//         //     points_y, triangle_indices.data().get(),
//         //     params.nb_all_cloth_vertices,
//         //     result_size);
//         if ( tp_result_size_h > 0 )
//             collision_tp_to_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
//                 collision_constraints.data().get(),
//                 tp_collision_result.data().get(),
//                 triangle_indices.data().get(),
//                 tp_result_size_h);
//         if ( ee_result_size_h > 0 )
//             collision_ee_to_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
//                 collision_constraints.data().get(),
//                 ee_collision_result.data().get(),
//                 edges.data().get(),
//                 tp_result_size_h,
//                 ee_result_size_h);
//         // coloring
//         // 1. 构建邻接关系并着色 (对应 Vivace 算法)
//         int num_colors = color_constraints(num_constraints);
//         std::cout << "palette_size: " << num_colors << std::endl;
//         RECORD_TIME("color_constraints");
//         n = params.nb_all_vertices;
//         // fill_inv_mass<<<(n + block - 1) / block, block>>>(
//         //     mass_inv.data().get(),
//         //     vertices_obj.data().get(),
//         //     object_types.data().get(),
//         //     masses.data().get(),
//         //     vertices_mask.data().get(), n);
//         // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
//         // std::vector<type> __##v(_##v.begin(), _##v.end())
//         // CHECK(mass_inv, float);
//         // #undef CHECK
//         // solve LCP using PGS
//         collision_tp_to_normal_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             debug_colors.data().get(),
//             tp_collision_result.data().get(),
//             constraint_colors.data().get(),
//             triangle_indices.data().get(),
//             mass_inv.data().get(),
//             tp_result_size_h);
//         collision_ee_to_normal_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             debug_colors.data().get(),
//             ee_collision_result.data().get(),
//             constraint_colors.data().get(),
//             edges.data().get(),
//             mass_inv.data().get(),
//             tp_result_size_h,
//             ee_result_size_h);
//         // 2. 将约束按颜色进行排序/分组
//         // 这样在 GPU 上读取时是连续的，利用内存合并访问（Coalesced Memory Access）
//         thrust::sort_by_key(thrust::device, constraint_colors.begin(),
//             constraint_colors.begin() + num_constraints, normal_constraints.begin());
//
//         CUDA_CHECK(cudaDeviceSynchronize());
//         int* lookup = point_hash_table_lookup.data().get();
//         record_color_offsets<<<(num_constraints + block - 1) / block, block>>>(
//             lookup, normal_constraints.data().get(), num_constraints);
//         CUDA_CHECK(cudaDeviceSynchronize());
//
//         cudaMemcpy(constraint_color_offsets.data(), lookup, (num_colors + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//
//         int* d_needs_more_iters = sort_result_size.data().get();
//         int h_needs_more_iters;
//         // 3. Multi-Color PGS 求解主循环
//         int num_iterations = 100;
//         RECORD_TIME("sort constraints");
//         if ( current_graph_exec != nullptr ) {
//             cudaGraphExecDestroy(current_graph_exec);
//             cudaGraphDestroy(current_graph);
//         }
//         cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
//         cudaMemsetAsync(d_needs_more_iters, 0, sizeof(int), capture_stream);
//         // 把单次 PGS 迭代 (包含所有颜色的 Kernel 发射) 录制下来
//         int c = num_colors == MAX_COLORS ? -1 : 0;
//         for ( ; c < num_colors; ++c ) {
//             int start_idx = constraint_color_offsets[c + 1];
//             int end_idx = constraint_color_offsets[c + 2];
//             if ( start_idx < 0 ) continue;
//             if ( end_idx <= start_idx ) end_idx = num_colors;
//             int num_constraints_in_color = end_idx - start_idx;
//             solvePGS_UnifiedColorBatchKernel<<<(num_constraints_in_color + block - 1) / block, block,0,capture_stream>>>(
//                 normal_constraints.data().get(), d_needs_more_iters,
//                 points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
//                 num_constraints_in_color);
//         }
//
//         cudaStreamEndCapture(capture_stream, &current_graph);
//         cudaGraphInstantiate(&current_graph_exec, current_graph, NULL, NULL, 0);
//         for ( int iter = 0; iter < num_iterations; ++iter ) {
//             // 按颜色逐个批次启动 GPU Kernel
//             // int c = num_colors == MAX_COLORS ? -1 : 0;
//             // for ( ; c < num_colors; ++c ) {
//             //     int start_idx = constraint_color_offsets[c + 1];
//             //     int end_idx = constraint_color_offsets[c + 2];
//             //     if ( start_idx < 0 ) continue;
//             //     if ( end_idx <= start_idx ) end_idx = num_colors;
//             //     int num_constraints_in_color = end_idx - start_idx;
//             //
//             //     // int gridSize = (num_constraints_in_color + blockSize - 1) / blockSize;
//             //
//             //     // 启动 Kernel，仅处理当前颜色的约束
//             //     solvePGS_UnifiedColorBatchKernel
//             //         <<<(num_constraints_in_color + block - 1) / block, block>>>(
//             //             normal_constraints.data().get(), d_needs_more_iters,
//             //             points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
//             //             num_constraints_in_color);
//             //
//             // }
//             cudaGraphLaunch(current_graph_exec, nullptr);
//             if ( iter % 10 == 0 ) {
//                 cudaMemcpy(&h_needs_more_iters, d_needs_more_iters, sizeof(int), cudaMemcpyDeviceToHost);
//                 std::cout << "constraints: " << h_needs_more_iters << std::endl;
//                 if ( h_needs_more_iters == 0 ) {
//                     break;
//                 }
//             }
//         }
//         RECORD_TIME("Multi-Color PGS");
//         update_end_collision<<<(n + block - 1) / block, block>>>(
//             points_y,
//             velocities.data().get(),
//             points_x,
//             points_collision,
//             vertices_mask.data().get(),
//             dt, n);
//     }
// }
// void Contact::collision_LCP_postprocess_unified(float3* points_y) {
//     START_TIMER;
//     int block = 256;
//     int n = (int)point_hash_table_size;
//     clear_hash_table<<<(n + block - 1) / block, block>>>(
//         point_hash_table.data().get(), n);
//     int cloth_vertex_size = params.nb_all_cloth_vertices;
//     n = cloth_vertex_size;
//
//     // float3* points_x = vertices_old.data().get();
//
//     debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
//     int num_constraints = tp_result_size_h + ee_result_size_h;
//     if ( num_constraints == 0 ) return;
//     int all_vertex_size = params.nb_all_vertices;
//     // float3* points_collision = temp_vertices_f3.data().get();
//     // cudaMemcpyAsync(points_collision, points_y, all_vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
//     float3* points_collision = points_y;
//     // coloring
//     // 1. 构建邻接关系并着色 (对应 Vivace 算法)
//     int num_colors = color_constraints(num_constraints);
//     std::cout << "palette_size: " << num_colors << std::endl;
//     RECORD_TIME("color_constraints");
//     n = params.nb_all_vertices;
//     // solve LCP using PGS
//     // 2. 将约束按颜色进行排序/分组
//     // 这样在 GPU 上读取时是连续的，利用内存合并访问（Coalesced Memory Access）
//     thrust::sort_by_key(thrust::device, constraint_colors.begin(),
//         constraint_colors.begin() + num_constraints, normal_constraints.begin());
//
//     CUDA_CHECK(cudaDeviceSynchronize());
//     int* lookup = point_hash_table_lookup.data().get();
//     record_color_offsets<<<(num_constraints + block - 1) / block, block>>>(
//         lookup, normal_constraints.data().get(), num_constraints);
//     CUDA_CHECK(cudaDeviceSynchronize());
//
//     cudaMemcpy(constraint_color_offsets.data(), lookup, (num_colors + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//
//     int* d_needs_more_iters = sort_result_size.data().get();
//     int h_needs_more_iters;
//     // 3. Multi-Color PGS 求解主循环
//     int num_iterations = 10;
//     RECORD_TIME("sort constraints");
//     if ( current_graph_exec != nullptr ) {
//         cudaGraphExecDestroy(current_graph_exec);
//         cudaGraphDestroy(current_graph);
//     }
//     cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
//     cudaMemsetAsync(d_needs_more_iters, 0, sizeof(int), capture_stream);
//     int c = num_colors == MAX_COLORS ? -1 : 0;
//     for ( ; c < num_colors; ++c ) {
//         int start_idx = constraint_color_offsets[c + 1];
//         int end_idx = constraint_color_offsets[c + 2];
//         if ( start_idx < 0 ) continue;
//         if ( end_idx <= start_idx ) end_idx = num_colors;
//         int num_constraints_in_color = end_idx - start_idx;
//         solvePGS_UnifiedColorBatchKernel<<<(num_constraints_in_color + block - 1) / block, block,0,capture_stream>>>(
//             normal_constraints.data().get(), d_needs_more_iters,
//             points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
//             num_constraints_in_color);
//     }
//
//     cudaStreamEndCapture(capture_stream, &current_graph);
//     cudaGraphInstantiate(&current_graph_exec, current_graph, NULL, NULL, 0);
//     for ( int iter = 0; iter < num_iterations; ++iter ) {
//         cudaGraphLaunch(current_graph_exec, nullptr);
//         if ( iter % 10 == 0 ) {
//             cudaMemcpy(&h_needs_more_iters, d_needs_more_iters, sizeof(int), cudaMemcpyDeviceToHost);
//             std::cout << "constraints: " << h_needs_more_iters << std::endl;
//             if ( h_needs_more_iters == 0 ) {
//                 break;
//             }
//         }
//     }
//     RECORD_TIME("Multi-Color PGS");
//     // update_end_collision<<<(n + block - 1) / block, block>>>(
//     //     points_y,
//     //     velocities.data().get(),
//     //     points_x,
//     //     points_collision,
//     //     vertices_mask.data().get(),
//     //     dt, n);
// }
//
// int Geometry::color_constraints(int num_constraints) {
//     int blockSize = 256;
//     int gridSize = (num_constraints + blockSize - 1) / blockSize;
//     int* d_constraint_colors = this->constraint_colors.data().get();
//     // 初始化着色数组为 -1
//     cudaMemsetAsync(d_constraint_colors, -1, num_constraints * sizeof(int));
//
//     int num_vertices = params.nb_all_vertices;
//     // 分配黑板：记录每个顶点每种颜色被谁占用了
//     int* d_vertex_color_claimer = this->vertex_color_claimer.data().get();
//     cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int));
//
//     int* d_uncolored_count = this->uncolored_count.data().get();
//     uint64_t* d_vertex_forbidden_masks = this->vertex_forbidden_masks.data().get();
//
//     int h_uncolored_count = num_constraints;
//     // int last_uncolored_count = h_uncolored_count;
//     int iteration = 0;
//     // int h_current_palette_size = 4;
//
//     // CollisionConstraint* d_constraints = this->collision_constraints.data().get();
//     auto* d_constraints = this->normal_constraints.data().get();
//
//     auto d_current_palette_size = alloc_pool();
//     auto d_iteration = alloc_pool();
//     auto d_last_uncolored_count = alloc_pool();
//     cudaMemsetAsync(d_last_uncolored_count.ptr, num_constraints, sizeof(int));
//     cudaMemsetAsync(d_iteration.ptr, 0, sizeof(int));
//     // 一开始给出少量颜色，增加冲突几率但节约颜色
//     int h_current_palette_size = 4;
//     cudaMemcpyAsync(d_current_palette_size.ptr, &h_current_palette_size, sizeof(int), cudaMemcpyHostToDevice);
//
//     // if ( current_graph_exec != nullptr ) {
//     //     cudaGraphExecDestroy(current_graph_exec);
//     //     cudaGraphDestroy(current_graph);
//     // }
//     cudaGraph_t graph = nullptr;
//     cudaGraphExec_t graph_exec = nullptr;
//     cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
//
//     for ( int i = 0; i < 20; i++ ) {
//         cudaMemsetAsync(d_vertex_forbidden_masks, 0, num_vertices * sizeof(uint64_t), capture_stream);
//         cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int), capture_stream);
//         // cudaMemsetAsync(d_uncolored_count, 0, sizeof(int));
//         // 步骤 1：去抢占颜色
//         k_mark_forbidden_bits<<<gridSize, blockSize,0,capture_stream>>>(
//             d_vertex_forbidden_masks, d_uncolored_count, d_constraints, d_constraint_colors,
//             num_constraints);
//         k_claim_color_bitmask<<<gridSize, blockSize,0,capture_stream>>>(
//             d_vertex_color_claimer, d_constraints, d_constraint_colors, d_vertex_forbidden_masks,
//             d_current_palette_size.ptr, d_iteration.ptr, num_constraints);
//
//         // 步骤 2：验证是否成功
//         k_verify_colors<<<gridSize, blockSize,0,capture_stream>>>(
//             d_constraint_colors, d_uncolored_count, d_constraints,
//             d_vertex_color_claimer, num_constraints);
//         k_update_colors<<<1, 1,0,capture_stream>>>(
//             d_current_palette_size.ptr,
//             d_uncolored_count,
//             d_last_uncolored_count.ptr,
//             d_iteration.ptr);
//     }
//     cudaStreamEndCapture(capture_stream, &graph);
//     cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
//     // d_current_palette_size.ptr
//     while ( iteration < 10 ) {
//         cudaGraphLaunch(graph_exec, nullptr);
//         /*cudaMemsetAsync(d_vertex_forbidden_masks, 0, num_vertices * sizeof(uint64_t));
//         cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int));
//         // cudaMemsetAsync(d_uncolored_count, 0, sizeof(int));
//         // 步骤 1：去抢占颜色
//         k_mark_forbidden_bits<<<gridSize, blockSize>>>(
//             d_vertex_forbidden_masks, d_uncolored_count, d_constraints, d_constraint_colors,
//             num_constraints);
//         k_claim_color_bitmask<<<gridSize, blockSize>>>(
//             d_vertex_color_claimer, d_constraints, d_constraint_colors, d_vertex_forbidden_masks,
//             d_current_palette_size.ptr, d_iteration.ptr, num_constraints);
//
//         // 步骤 2：验证是否成功
//         k_verify_colors<<<gridSize, blockSize>>>(
//             d_constraint_colors, d_uncolored_count, d_constraints,
//             d_vertex_color_claimer, num_constraints);
//         k_update_colors<<<1, 1>>>(
//             d_current_palette_size.ptr,
//             d_uncolored_count,
//             d_last_uncolored_count.ptr,
//             d_iteration.ptr);*/
//         // if ( iteration % 3 == 0 ) {
//         cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
//         if ( h_uncolored_count == 0 ) break;
//         // }
//         iteration++;
//
//         // 如果还有没涂上的，慢慢增加可选颜色的种类
//         // cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
//         // if ( last_uncolored_count == h_uncolored_count && h_uncolored_count > 0 ) {
//         //     if ( current_palette_size < MAX_COLORS )
//         //         current_palette_size++;
//         // }
//         // last_uncolored_count = h_uncolored_count;
//         // std::cout << "uncolored_count: " << h_uncolored_count << std::endl;
//     }
//     // k_print_colors<<<gridSize, blockSize>>>(d_constraint_colors, d_constraints, num_constraints);
//     cudaGraphDestroy(graph);
//     cudaGraphExecDestroy(graph_exec);
//     cudaMemcpy(&h_current_palette_size, d_current_palette_size.ptr,
//         sizeof(int), cudaMemcpyDeviceToHost);
//     return h_current_palette_size;
//
// }
//
// void Contact::collision_collect_near_pairs(float3* points, float max_dist,
//     bool update_hash, bool collect_pp, bool collect_tp, bool collect_ee) {
//     START_TIMER;
//     int block = 256;
//     // int n = (int)point_hash_table_size;
//     // clear_hash_table<<<(n + block - 1) / block, block>>>(
//     //     point_hash_table.data().get(), n);
//     int vertex_size = params.nb_all_cloth_vertices;
//     int n = vertex_size;
//     float cell_size = max_dist * 2.f;
//     auto constraint_size_p = alloc_pool();
//     if ( collect_tp || collect_ee || collect_pp )
//         cudaMemsetAsync(constraint_size_p.ptr, 0, sizeof(int));
//     // float max_dist_edge = params.cloth_edge_mean_length * 2.f;
//     if ( update_hash ) {
//         auto point_hashes_size_p = alloc_pool();
//         auto edge_hashes_size_p = alloc_pool();
//         cudaMemsetAsync(point_hashes_size_p.ptr, 0, sizeof(int));
//         record_point_hash<false, false><<<(n + block - 1) / block, block>>>(
//             point_hashes.data().get(),
//             sort_key_temp.data().get(),
//             point_hashes_size_p.ptr,
//             nullptr,
//             points,
//             vertex_proxy.data().get(),
//             cell_size,
//             max_dist,
//             max_point_hashes_size,
//             point_hash_table_size,
//             n);
//
//         CUDA_CHECK(cudaMemcpy(&point_hashes_size_h, point_hashes_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost));
//         point_hashes_size_h = min(point_hashes_size_h, max_point_hashes_size);
//         RECORD_TIME("record_point_hash");
//         thrust::sort_by_key(thrust::device, sort_key_temp.begin(), sort_key_temp.begin() + point_hashes_size_h,
//             point_hashes.begin());
//
//         RECORD_TIME("sort_by_key");
//         // build hash_table lookup
//         cudaMemsetAsync(point_hash_table_lookup.data().get(), -1, sizeof(int) * (point_hash_table_size + 1));
//         record_hash_table_lookup<<<(point_hashes_size_h + block - 1) / block, block>>>(
//             point_hash_table_lookup.data().get(),
//             point_hashes.data().get(),
//             point_hash_table_size, point_hashes_size_h);
//         // edges
//         if ( collect_ee ) {
//             n = params.nb_all_cloth_edges;
//             CUDA_CHECK(cudaMemsetAsync(edge_hashes_size_p.ptr, 0, sizeof(int)));
//             record_edge_hashes<<<(n + block - 1) / block, block>>>(
//                 edge_hashes.data().get(),
//                 sort_key_temp.data().get(),
//                 edge_hashes_size_p.ptr,
//                 edges.data().get(),
//                 points,
//                 cell_size,
//                 max_edge_hashes_size,
//                 edge_hash_table_size,
//                 n);
//             CUDA_CHECK(cudaMemcpy(&edge_hashes_size_h, edge_hashes_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost));
//             edge_hashes_size_h = min(edge_hashes_size_h, max_edge_hashes_size);
//             RECORD_TIME("record_edge_hash");
//             thrust::sort_by_key(thrust::device,
//                 sort_key_temp.begin(), sort_key_temp.begin() + edge_hashes_size_h, edge_hashes.begin());
//             RECORD_TIME("sort_by_key");
//             // build hash_table lookup
//             cudaMemsetAsync(edge_hash_table_lookup.data().get(), -1, sizeof(int) * (edge_hash_table_size + 1));
//             record_hash_table_lookup<<<(edge_hashes_size_h + block - 1) / block, block>>>(
//                 edge_hash_table_lookup.data().get(),
//                 edge_hashes.data().get(),
//                 edge_hash_table_size, edge_hashes_size_h);
//         }
//     }
//     tp_result_size_h = pp_result_size_h = ee_result_size_h = 0;
//     if ( collect_pp ) {
//         cudaMemsetAsync(pp_result_size.data().get(), 0, sizeof(int));
//         n = params.nb_all_cloth_vertices;
//         // debug_colors.assign(vertex_size, make_float3(0.5f, 0.5f, 0.5f));
//         // collect_pp_sorted<<<(n + block - 1) / block, block>>>(
//         //     pp_collision_result.data().get(),
//         //     pp_result_size.data().get(),
//         //     debug_colors.data().get(),
//         //     point_hashes.data().get(),
//         //     points,
//         //     edge_lookup.data().get(),
//         //     dir_edges.data().get(),
//         //     point_hash_table_lookup.data().get(),
//         //     vertex_proxy.data().get(),
//         //     vertices_mask.data().get(),
//         //     cell_size,
//         //     max_dist,
//         //     point_hash_table_size,
//         //     point_hashes_size_h,
//         //     max_pp_result_size,
//         //     n);
//         points_query_points_by_point_hash<<<(n + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             pp_result_size.data().get(),
//             nullptr,
//             point_hashes.data().get(),
//             points,
//             edge_lookup.data().get(),
//             dir_edges.data().get(),
//             point_hash_table_lookup.data().get(),
//             vertex_proxy.data().get(),
//             vertices_mask.data().get(),
//             cell_size,
//             max_dist,
//             point_hash_table_size,
//             point_hashes_size_h,
//             max_collision_constraints_size,
//             n);
//         cudaMemcpy(&pp_result_size_h, pp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//         // pp_result_size_h = min(pp_result_size_h, max_pp_result_size);
//         pp_result_size_h = min(pp_result_size_h, max_collision_constraints_size);
//         std::cout << "pp_result_size_h = " << pp_result_size_h << std::endl;
//     }
//     if ( collect_tp ) {
//         n = params.nb_all_triangles;
//         // cudaMemsetAsync(tp_result_size.data().get(), 0, sizeof(int));
//         // triangles_query_points<<<(n + block - 1) / block, block>>>(
//         //     tp_collision_result.data().get(),
//         //     tp_result_size.data().get(),
//         //     triangle_indices.data().get(),
//         //     points,
//         //     vertices_old.data().get(),
//         //     point_hashes.data().get(),
//         //     point_hash_table_lookup.data().get(),
//         //     vertices_mask.data().get(), cell_size,
//         //     max_dist * max_dist, point_hash_table_size,
//         //     point_hashes_size_h, max_tp_result_size,
//         //     params.nb_all_cloth_vertices, n);
//         // cudaMemcpy(&tp_result_size_h, tp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//         // tp_result_size_h = min(tp_result_size_h, max_tp_result_size);
//         debug_colors.assign(params.nb_all_cloth_edges, make_float3(0.5f, 0.5f, 0.5f));
//         triangles_query_points_by_point_hash<<<(n + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             nullptr,
//             constraint_size_p.ptr,
//             triangle_indices.data().get(),
//             points,
//             // vertices_old.data().get(),
//             point_hashes.data().get(),
//             point_hash_table_lookup.data().get(),
//             vertices_mask.data().get(), cell_size,
//             max_dist * max_dist, point_hash_table_size,
//             point_hashes_size_h, max_collision_constraints_size,
//             params.nb_all_cloth_vertices, n);
//         cudaMemcpy(&tp_result_size_h, constraint_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost);
//         tp_result_size_h = min(tp_result_size_h, max_collision_constraints_size);
//         RECORD_TIME("triangles_query_points");
//         std::cout << "tp_result_size_h = " << tp_result_size_h << std::endl;
//     }
//     if ( collect_ee ) {
//         n = params.nb_all_edges;
//         // max_dist *= 0.5f;
//         // cudaMemsetAsync(ee_result_size.data().get(), 0, sizeof(int));
//         // edges_query_edges_via_point_hash<<<(n + block - 1) / block, block>>>(
//         //     ee_collision_result.data().get(),
//         //     ee_result_size.data().get(),
//         //     edges.data().get(),
//         //     points,
//         //     // vertices_old.data().get(),
//         //     point_hashes.data().get(),
//         //     point_hash_table_lookup.data().get(),
//         //     dir_edges.data().get(),
//         //     edge_lookup.data().get(),
//         //     vertices_mask.data().get(), cell_size,
//         //     max_dist * max_dist, max_dist * 0.5, point_hash_table_size,
//         //     point_hashes_size_h, max_ee_result_size,
//         //     params.nb_all_cloth_vertices, n);
//         // cudaMemcpy(&ee_result_size_h, ee_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//         // ee_result_size_h = min(ee_result_size_h, max_ee_result_size);
//         // RECORD_TIME("edges_query_edges_via_point_hash");
//         block = 64;
//         detect_edge_edge_constraints<<<(n + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             constraint_size_p.ptr,
//             edges.data().get(),
//             points,
//             edge_hashes.data().get(),
//             edge_hash_table_lookup.data().get(),
//             vertices_mask.data().get(),
//             cell_size, max_dist,
//             max_collision_constraints_size,
//             edge_hash_table_size,
//             edge_hashes_size_h,
//             params.nb_all_cloth_vertices,
//             n);
//         cudaMemcpy(&ee_result_size_h, constraint_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost);
//         RECORD_TIME("detect_edge_edge_constraints");
//         ee_result_size_h = min(ee_result_size_h, max_collision_constraints_size);
//         ee_result_size_h -= tp_result_size_h;
//         std::cout << "ee_result_size_h = " << ee_result_size_h << std::endl;
//     }
// }
