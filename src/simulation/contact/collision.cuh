#pragma once

#include "common/cuda_utils.h"
#include <device_launch_parameters.h>


#include "common/atomic_utils.cuh"
#include "common/geometric_algorithms.h"
#include "collision_type.cuh"

static __global__ void insert_points_to_grid(
    const float3* __restrict__ points,
    PointHashCell* table,
    float cell_size,
    int hash_table_size,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;

    float cell_size_inv = 1.0f / cell_size;
    float3 p = points[idx];
    int3 grid_idx = make_int3(floorf(p.x * cell_size_inv), floorf(p.y * cell_size_inv),
        floorf(p.z * cell_size_inv));

    int h = get_hash(grid_idx, hash_table_size);

    int old_count = atomicAdd(&table[h].count, 1);
    if ( old_count < MAX_POINTS_PER_CELL ) {
        table[h].point_indices[old_count] = idx;
    }
}
template<bool InsertToGrid, bool RecordAround = true>
static __global__ void record_point_hash(
    int2* __restrict__ results,
    int* __restrict__ results_sort_key,
    int* __restrict__ results_size,
    PointHashCell* hash_table,
    const float3* __restrict__ points,
    const int* __restrict__ points_proxy,
    float cell_size,
    float max_dist,
    int max_results_size,
    int hash_table_size,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;
    if ( points_proxy[idx] != idx )return;
    float cell_size_inv = 1.0f / cell_size;
    float3 p = points[idx];
    int3 grid_idx = make_int3(floorf(p.x * cell_size_inv), floorf(p.y * cell_size_inv),
        floorf(p.z * cell_size_inv));
    if constexpr ( InsertToGrid ) {
        int h = get_hash(grid_idx, hash_table_size);
        int old_count = atomicAdd(&hash_table[h].count, 1);
        if ( old_count < MAX_POINTS_PER_CELL ) {
            hash_table[h].point_indices[old_count] = idx;
        }
    }
    if constexpr ( RecordAround ) {
        int x_min = (int)floorf((p.x - max_dist) * cell_size_inv) != grid_idx.x ? -1 : 0;
        int x_max = (int)floorf((p.x + max_dist) * cell_size_inv) != grid_idx.x ? 1 : 0;
        int y_min = (int)floorf((p.y - max_dist) * cell_size_inv) != grid_idx.y ? -1 : 0;
        int y_max = (int)floorf((p.y + max_dist) * cell_size_inv) != grid_idx.y ? 1 : 0;
        int z_min = (int)floorf((p.z - max_dist) * cell_size_inv) != grid_idx.z ? -1 : 0;
        int z_max = (int)floorf((p.z + max_dist) * cell_size_inv) != grid_idx.z ? 1 : 0;
        for ( int x = x_min; x <= x_max; x++ )
            for ( int y = y_min; y <= y_max; y++ )
                for ( int z = z_min; z <= z_max; z++ ) {
                    int h = get_hash(grid_idx + make_int3(x, y, z), hash_table_size);
                    int res_id = atomicAdd(results_size, 1);
                    if ( res_id < max_results_size ) {
                        results[res_id] = make_int2(h, idx);
                        results_sort_key[res_id] = h;
                    }
                }
    }
    else {
        int h = get_hash(grid_idx, hash_table_size);
        int res_id = atomicAdd(results_size, 1);
        if ( res_id < max_results_size ) {
            results[res_id] = make_int2(h, idx);
            results_sort_key[res_id] = h;
        }
    }
}

static __global__ void record_hash_table_lookup(
    int* __restrict__ lookup,
    const int2* __restrict__ sorted_results,
    int hash_table_size,
    int sorted_results_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= sorted_results_size ) return;
    auto pp_hash = sorted_results[idx].x;
    auto pp_hash_prev = idx > 0 ? sorted_results[idx - 1].x : -1;
    if ( pp_hash != pp_hash_prev && pp_hash < hash_table_size ) {
        lookup[pp_hash] = idx;
    }
}
static __device__ bool has_edge(int v0, int v1, const int2* lookup, const int2* dir_edges) {
    // if (lookup[v0].y > lookup[v1].y) {int temp = v0; v0 = v1; v1 = temp; }
    auto [offset, degree] = lookup[v0];
    if ( dir_edges[offset].x <= v1 && v1 <= dir_edges[offset + degree - 1].x ) {
        for ( int d = 0; d < degree; d++ ) {
            if ( dir_edges[offset + d].x == v1 ) return true;
        }
    }
    return false;
}
static __global__ void collect_pp_sorted(
    CollisionResult_PP* results,
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
                            results[res_id] = CollisionResult_PP(idx, p_idx);
                            if ( debug_colors ) {
                                debug_colors[idx] = make_float3(1.f, 0.f, 0.f);
                                debug_colors[p_idx] = make_float3(1.f, 0.f, 0.f);
                            }
                        }
                    }
                }
            }
}
static __global__ void collect_pp(
    CollisionResult_PP* results,
    int* results_size,
    // unsigned long long* sort_key_ptr,
    // int* sort_value_ptr,
    // int* sort_results_size,
    const float3* __restrict__ vertices,
    const PointHashCell* table,
    float query_dist_sq,
    int num_vertices,
    int max_results_size,
    // int max_sort_results_size,
    int hash_table_size,
    float cell_size
) {
    int vert_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vert_idx >= num_vertices ) return;
    float3 p = vertices[vert_idx];
    float cell_size_inv = 1.0f / cell_size;
    int3 grid_idx = make_int3(floorf(p.x * cell_size_inv), floorf(p.y * cell_size_inv),
        floorf(p.z * cell_size_inv));
    for ( int x = -1; x <= 1; x++ )
        for ( int y = -1; y <= 1; y++ )
            for ( int z = -1; z <= 1; z++ ) {
                int h = get_hash(grid_idx + make_int3(x, y, z), hash_table_size);
                int count = table[h].count;
                if ( count == 0 ) continue;
                for ( int k = 0; k < count; k++ ) {
                    int p_idx = table[h].point_indices[k];
                    if ( vert_idx >= p_idx ) continue;
                    float3 p2 = vertices[p_idx];
                    if ( len_sq(p - p2) <= query_dist_sq ) {
                        int res_id = atomicAdd(results_size, 1);
                        if ( res_id < max_results_size ) {
                            results[res_id] = CollisionResult_PP(vert_idx, p_idx);
                            // int key_id = atomicAdd(sort_results_size, 1);
                            // auto sort_key = ((unsigned long long)vert_idx << 32) | (unsigned int)p_idx;
                            // sort_key_ptr[key_id] = sort_key;
                            // sort_value_ptr[key_id] = res_id;
                            // key_id = atomicAdd(sort_results_size, 1);
                            // sort_key = ((unsigned long long)p_idx << 32) | (unsigned int)vert_idx;
                            // sort_key_ptr[key_id] = sort_key;
                            // sort_value_ptr[key_id] = res_id;
                        }
                    }
                }
            }
}
static __device__ bool solve_point_triangle_ccd_simplified(
    float3 p_o, float3 p_curr,
    float3 v0_o, float3 n_o,
    float3 v0_c, float3 n_curr,
    float thickness,
    float3& out_normal
) {
    // 1. 计算上一帧点到面的有符号距离
    float dist_old = dot(p_o - v0_o, n_o);

    // 2. 计算当前帧点到面的有符号距离
    float dist_curr = dot(p_curr - v0_c, n_curr);

    // 3. 判断是否穿透（符号改变）
    // 或者虽然没改变，但当前距离已经小于厚度
    if ( dist_old > 0 && dist_curr < thickness ) {
        // 说明从正面撞向背面，法线必须保持为正面方向
        out_normal = n_o;
        return true;
    }
    else if ( dist_old < 0 && dist_curr > -thickness ) {
        // 说明从背面撞向正面，法线必须保持为背面方向
        out_normal = -n_o;
        return true;
    }

    return false;
}

static __global__ void triangles_query_points(
    CollisionResult_TP* results,
    int* results_size,
    const int3* __restrict__ indices,
    const float3* __restrict__ vertices,
    const float3* __restrict__ vertices_old,
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
    // if ( is_cloth ) return; // TODO
    // if (idx3.x > num || idx3.y > num || idx3.z > num || idx3.x <cloth_vertices_size || idx3.y < cloth_vertices_size || idx3.z < cloth_vertices_size ) {
    //     printf("ERROR: [%d,%d,%d]\n",idx3.x,idx3.y,idx3.z);
    //     return;
    // }
    float3 v0 = vertices[idx3.x], v1 = vertices[idx3.y], v2 = vertices[idx3.z];
    float3 v0_o = vertices_old[idx3.x], v1_o = vertices_old[idx3.y], v2_o = vertices_old[idx3.z];
    float3 n_o = normalized(cross(v1_o - v0_o, v2_o - v0_o));
    float3 n_curr = normalized(cross(v1 - v0, v2 - v0));
    if ( norm(n_curr) < 0.5 ) {
        // printf("ERROR: tri:%d, n_curr = %f\n, [(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)]\n",
        //     tri_idx, norm(n_curr), v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
        return;
    }

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
    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {
                // continue;//err
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
                    // continue;//err
                    if ( p_idx == idx3.x || p_idx == idx3.y || p_idx == idx3.z ) continue;
                    // continue;//noerr
                    // if ( p_idx < 0 || p_idx >= cloth_vertices_size ) {
                    //     printf("ERROR: p_idx:%d\n", p_idx);
                    //     continue;
                    // }
                    // continue;//err
                    if ( !is_cloth && mask[p_idx] ) continue; // Both are fixed.
                    // continue;//err
                    float3 pos = vertices[p_idx];
                    int grid_h = get_hash(make_int3(floorf(pos.x * cell_size_inv), floorf(pos.y * cell_size_inv),
                        floorf(pos.z * cell_size_inv)), hash_table_size);
                    if ( grid_h != h ) continue;

                    float dist = dot(pos - v0, n_curr);
                    // float3 closest_pt;
                    // float dist_sq = point_triangle_sq_dist(pos, v0, v1, v2, &closest_pt);
                    float dist_sq = dist * dist;

                    if ( dist_sq < query_dist_sq ) {
                        float3 closest_pt = pos - dist * n_curr;
                        float u, v, w;
                        barycentric(v0, v1, v2, closest_pt, u, v, w);
                        const float eps = -1e-2f;
                        const float one_eps = 1.f - eps;
                        if ( u < eps || v < eps || w < eps
                            || u > one_eps || v > one_eps || w > one_eps )
                            continue;
                        float3 normal;
                        if ( is_cloth ) {
                            // float3 diff = pos - closest_pt;
                            // float d = sqrtf(dist_sq);
                            // tp.normal = (d > 1e-6f) ? diff * (1.0f / d) : make_float3(0, 0, 1);
                            if ( !solve_point_triangle_ccd_simplified(vertices_old[p_idx], pos,
                                v0_o, n_o, v0, n_curr, padding, normal) )
                                continue;
                        }
                        else
                            normal = n_curr;
                        int tp_idx = atomicAdd(results_size, 1);
                        // printf("%f,%f [%d,%d]\n",dist_sq , query_dist_sq,p_idx,tri_idx);
                        if ( tp_idx < max_results_size ) {
                            auto tp = CollisionResult_TP();
                            tp.vert_idx = p_idx;
                            tp.tri_idx = tri_idx;
                            tp.weights = make_float3(u, v, w);
                            tp.normal = normal;
                            // tp.closest_pt = closest_pt;
                            results[tp_idx] = tp;
                        }
                    }
                }
            }
        }
    }
}

__device__ __host__ __forceinline__ float clamp(float x, float a, float b) {
    // return fminf(fmaxf(x, a), b);
    return (x < a) ? a : ((x > b) ? b : x);
}

static __device__ float edge_edge_dist_sq(float3 p1, float3 p2, float3 p3, float3 p4, float3& closest_a, float3& closest_b,
    float& s,
    float& t) {
    float3 d1 = p2 - p1; // Edge A direction
    float3 d2 = p4 - p3; // Edge B direction
    float3 r = p1 - p3;

    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);

    float eps = 1e-8f;

    if ( a > eps && e > eps ) {
        float b = dot(d1, d2);
        float denom = a * e - b * b;

        // 如果近似平行，denom 接近 0，这里简单处理为不相交
        // 如果需要处理平行边，此处需额外逻辑，但为了速度通常跳过或简化
        if ( fabsf(denom) < eps ) return false;

        float c = dot(d1, r);
        s = (b * f - c * e) / denom;
        t = (b * s + f) / e;
    }
    else {
        s = -1.f;
    }

    closest_a = p1 + s * d1;
    closest_b = p3 + t * d2;
    return dot(closest_a - closest_b, closest_a - closest_b);
}


static __global__ void edges_query_edges_via_point_hash(
    CollisionResult_EE* results,
    int* results_size,
    const int2* __restrict__ edges,
    const float3* __restrict__ vertices,
    // const float3* __restrict__ vertices_old,
    const int2* __restrict__ points_hashes,
    const int* __restrict__ hash_lookup,
    const int2* __restrict__ dir_edges,
    const int2* __restrict__ edge_lookup,
    const char* __restrict__ mask,
    float cell_size,
    float query_dist_sq,
    float max_edge_length,// 必须传入网格中的最大边长
    int hash_table_size,
    int points_hashes_size,
    int max_results_size,
    int cloth_vertices_size,
    int num_edges
) {
    int edgeA_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( edgeA_idx >= num_edges ) return;

    int2 eA = edges[edgeA_idx];
    bool is_cloth_A = eA.x < cloth_vertices_size;

    float3 v0 = vertices[eA.x];
    float3 v1 = vertices[eA.y];
    float3 dir = normalized(v1 - v0);
    bool mask_all = mask[eA.x] && mask[eA.y];

    float cell_size_inv = 1.0f / cell_size;

    float padding = sqrtf(query_dist_sq) + max_edge_length;

    float3 min_p = fmin3(v0, v1) - make_float3(padding, padding, padding);
    float3 max_p = fmax3(v0, v1) + make_float3(padding, padding, padding);

    int3 min_g = make_int3(floorf(min_p.x * cell_size_inv), floorf(min_p.y * cell_size_inv), floorf(min_p.z * cell_size_inv));
    int3 max_g = make_int3(floorf(max_p.x * cell_size_inv), floorf(max_p.y * cell_size_inv), floorf(max_p.z * cell_size_inv));
    const float center2corner = cell_size * 0.866025f; // 3^0.5 / 2
    // const float padding = dHat;
    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {
                float3 center = make_float3(x + 0.5f, y + 0.5f, z + 0.5f) * cell_size;
                float dist = norm(center - v0 - dir * dot(center - v0, dir));
                if ( dist > center2corner + padding *0.5f ) continue;

                int h = get_hash(make_int3(x, y, z), hash_table_size);
                if ( hash_lookup[h] == -1 ) continue;

                for ( int k = hash_lookup[h]; k < points_hashes_size && points_hashes[k].x == h; k++ ) {
                    int p_idx = points_hashes[k].y;
                    if ( eA.x == p_idx || eA.y == p_idx ) continue;
                    // float3 pos = vertices[p_idx];
                    // int grid_h = get_hash(make_int3(floorf(pos.x * cell_size_inv), floorf(pos.y * cell_size_inv),
                    //     floorf(pos.z * cell_size_inv)), hash_table_size);
                    // if ( grid_h != h ) continue;
                    // 找到了 AABB 内的点 p_idx，现在查找包含这个点的所有边 E_B
                    int start_edge = edge_lookup[p_idx].x;
                    int end_edge = start_edge + edge_lookup[p_idx].y;

                    for ( int i = start_edge; i < end_edge; ++i ) {
                        int edgeB_idx = dir_edges[i].y;

                        // 1. 去重：强制 edgeA_idx < edgeB_idx，保证每对只检测一次
                        if ( edgeA_idx >= edgeB_idx ) continue;

                        int2 eB = edges[edgeB_idx];

                        // 2. 拓扑排除：如果两条边共享顶点，则跳过
                        if ( eA.x == eB.x || eA.x == eB.y || eA.y == eB.x || eA.y == eB.y ) continue;

                        bool is_cloth_B = eB.x < cloth_vertices_size;
                        if ( !is_cloth_A && !is_cloth_B ) continue; // 忽略静态物体内部的边
                        if ( mask_all && mask[eB.x] && mask[eB.y] ) continue;

                        // 3. 几何去重：由于 E_B 的两个端点都可能在 AABB 内，会导致同一个 E_B 被处理两次。
                        //    如果另一个端点也在 AABB 内，我们只在 p_idx 是较小的那个端点时才处理。
                        int other_v_idx = (eB.x == p_idx) ? eB.y : eB.x;
                        float3 other_pos = vertices[other_v_idx];
                        bool other_in_aabb = (
                            other_pos.x >= min_p.x && other_pos.x <= max_p.x &&
                            other_pos.y >= min_p.y && other_pos.y <= max_p.y &&
                            other_pos.z >= min_p.z && other_pos.z <= max_p.z
                        );
                        if ( other_in_aabb && p_idx > other_v_idx ) continue;

                        // 取出坐标
                        float3 v2 = vertices[eB.x];
                        float3 v3 = vertices[eB.y];

                        // 4. 计算 边-边 最短距离
                        float s, t; // 线段参数
                        float3 closest_A, closest_B;
                        
                        float dist_sq = edge_edge_dist_sq(v0, v1, v2, v3, closest_A, closest_B, s, t);
                        if ( s < 0.0f || s > 1.0f || t < 0.0f || t > 1.0f ) {
                            continue;
                        }
                        if ( dist_sq < query_dist_sq ) {

                            // 5. 连续碰撞检测 (CCD) 或获取 Normal（如果需要）
                            float3 normal = make_float3(0, 0, 0);
                            // CCD 逻辑：类似 solve_edge_edge_ccd(...)
                            // 如果只是静态干涉：
                            if ( dist_sq > 1e-8f ) {
                                float d = sqrtf(dist_sq);
                                normal = (closest_A - closest_B) * (1.0f / d);
                            }
                            else continue;

                            int ee_idx = atomicAdd(results_size, 1);
                            if ( ee_idx < max_results_size ) {
                                auto ee = CollisionResult_EE();
                                ee.edgeA_idx = edgeA_idx;
                                ee.edgeB_idx = edgeB_idx;
                                ee.coefs = make_float2(s, t);
                                ee.normal = normal;
                                results[ee_idx] = ee;
                            }
                        }
                    }
                }
            }
        }
    }
}

static __global__ void clear_hash_table(PointHashCell* table, int hash_table_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < hash_table_size ) {
        table[idx].count = 0;
    }
}

static __global__ void update_end_collision(
    float3* __restrict__ points_y,
    float3* __restrict__ velocities,
    const float3* __restrict__ points_x,
    const float3* __restrict__ points_collision,
    const char* __restrict__ vertices_mask,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto x = points_collision[i];
        if ( !vertices_mask[i] ) {
            float h_inv = 1.f / h;
            auto new_vel = (x - points_x[i]) * h_inv;
            // auto old_vel_dir = normalized(points_y[i] - points_x[i]);
            // float proj_vel_len = max(0.f, dot(old_vel_dir, new_vel));
            // new_vel = old_vel_dir * proj_vel_len;
            velocities[i] = new_vel;

            points_y[i] = x;
        }
        else {
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
    }
}



static __global__ void collision_tp_to_constraints(
    CollisionConstraint* __restrict__ constraints,
    const CollisionResult_TP* __restrict__ tps,
    const int3* __restrict__ triangle_indices,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;
    auto tp = tps[cid];
    auto [v0_i, v1_i, v2_i] = triangle_indices[tp.tri_idx];
    CollisionConstraint c;
    // c.type = char(0);
    c.p0 = tp.vert_idx;
    c.p1 = v0_i;
    c.p2 = v1_i;
    c.p3 = v2_i;
    constraints[cid] = c;
}
static __global__ void collision_ee_to_constraints(
    CollisionConstraint* __restrict__ constraints,
    const CollisionResult_EE* __restrict__ ees,
    const int2* __restrict__ edges,
    int offset,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;
    auto ee = ees[cid];
    auto [v0_i, v1_i] = edges[ee.edgeA_idx];
    auto [v2_i, v3_i] = edges[ee.edgeB_idx];
    CollisionConstraint c;
    // c.type = char(0);
    c.p0 = v0_i;
    c.p1 = v1_i;
    c.p2 = v2_i;
    c.p3 = v3_i;
    constraints[cid + offset] = c;
}
static __global__ void collision_tp_to_normal_constraints(
    UnifiedNormalConstraint* __restrict__ constraints,
    float3* debug_colors,
    const CollisionResult_TP* __restrict__ tps,
    const int* __restrict__ constraint_colors,
    const int3* __restrict__ triangle_indices,
    const float* __restrict__ invMass,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;
    auto tp = tps[cid];
    auto [v1_i, v2_i, v3_i] = triangle_indices[tp.tri_idx];
    int v0_i = tp.vert_idx;
    UnifiedNormalConstraint c;
    c.color = constraint_colors[cid];
    c.v[0] = v0_i;
    c.v[1] = v1_i;
    c.v[2] = v2_i;
    c.v[3] = v3_i;
    c.w[0] = 1.f;
    // float3 p0 = vertices[v0_i], p1 = vertices[v1_i], p2 = vertices[v2_i], p3 = vertices[v3_i];
    // float3 p_project = p0 - dot(p0 - p1, tp.normal) * tp.normal;
    // float u, v, w;
    // barycentric(p1, p2, p3, p_project, u, v, w);
    c.w[1] = -tp.weights.x;
    c.w[2] = -tp.weights.y;
    c.w[3] = -tp.weights.z;
    // c.w[1] = -u;
    // c.w[2] = -v;
    // c.w[3] = -w;
    c.normal = tp.normal;
    c.A_ii = invMass[v0_i] + c.w[1] * c.w[1] * invMass[v1_i]
        + c.w[2] * c.w[2] * invMass[v2_i] + c.w[3] * c.w[3] * invMass[v3_i] + 1e-8f;
    // printf("c.A_ii:%f\n", c.A_ii);
    c.lambda = 0.f;
    constraints[cid] = c;
    if ( debug_colors ) {
        for ( int i : c.v ) debug_colors[i] = make_float3(1.f, 0.f, 0.f);
        // printf("color: %d\n",c.v[0]);
    }
}
static __global__ void collision_ee_to_normal_constraints(
    UnifiedNormalConstraint* __restrict__ constraints,
    float3* debug_colors,
    const CollisionResult_EE* __restrict__ ees,
    const int* __restrict__ constraint_colors,
    const int2* __restrict__ edges,
    const float* __restrict__ invMass,
    int offset,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;
    auto ee = ees[cid];
    auto [v0_i, v1_i] = edges[ee.edgeA_idx];
    auto [v2_i, v3_i] = edges[ee.edgeB_idx];
    UnifiedNormalConstraint c;
    c.color = constraint_colors[cid + offset];
    c.v[0] = v0_i;
    c.v[1] = v1_i;
    c.v[2] = v2_i;
    c.v[3] = v3_i;
    c.w[0] = 1.f - ee.coefs.x;
    c.w[1] = ee.coefs.x;
    c.w[2] = ee.coefs.y - 1.f;
    c.w[3] = -ee.coefs.y;

    c.normal = ee.normal;
    c.A_ii = invMass[v0_i] + c.w[1] * c.w[1] * invMass[v1_i]
        + c.w[2] * c.w[2] * invMass[v2_i] + c.w[3] * c.w[3] * invMass[v3_i] + 1e-8f;
    // printf("c.A_ii:%f\n", c.A_ii);
    c.lambda = 0.f;
    constraints[cid + offset] = c;
    if ( debug_colors ) {
        for ( int i : c.v ) debug_colors[i] = make_float3(0.f, 1.f, 0.f);
        // printf("color: %d\n",c.v[0]);
    }
}
#define MAX_COLORS 64

static __device__ int hash_color(int constraint_id, int iteration, int palette_size) {
    unsigned int x = (constraint_id ^ iteration) * 1103515245 + 12345;
    x = (x ^ (x >> 16)) * 2654435769;
    return x % palette_size;
}

static __global__ void k_mark_forbidden_bits(
    uint64_t* __restrict__ vertex_forbidden_masks,
    int* d_uncolored_count,
    const UnifiedNormalConstraint* __restrict__ constraints,
    const int* __restrict__ constraint_colors,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;
    if ( cid == 0 ) { *d_uncolored_count = 0; }

    int c = constraint_colors[cid];
    if ( c != -1 ) {
        const auto& cons = constraints[cid];
        uint64_t bit = (1ull << c);
        for (int v : cons.v) {
            atomicOr(&vertex_forbidden_masks[v], bit);
        }
        // atomicOr(&vertex_forbidden_masks[cons.p0], bit);
        // atomicOr(&vertex_forbidden_masks[cons.p1], bit);
        // atomicOr(&vertex_forbidden_masks[cons.p2], bit);
        // atomicOr(&vertex_forbidden_masks[cons.p3], bit);
    }
}

static __global__ void k_claim_color_bitmask(
    int* __restrict__ vertex_color_claimer,
    UnifiedNormalConstraint* __restrict__ constraints,
    const int* __restrict__ constraint_colors,
    const uint64_t* __restrict__ vertex_forbidden_masks,
    const int* current_palette_size,
    const int* iteration,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;

    if ( constraint_colors[cid] == -1 ) {
        auto& cons = constraints[cid];

        uint64_t forbidden = vertex_forbidden_masks[cons.v[0]] |
            vertex_forbidden_masks[cons.v[1]] |
            vertex_forbidden_masks[cons.v[2]] |
            vertex_forbidden_masks[cons.v[3]];

        // 计算当前调色板范围内的可用位
        uint64_t palette_range_mask = (1ull << *current_palette_size) - 1;
        uint64_t available_mask = (~forbidden) & palette_range_mask;

        if ( available_mask == 0ull ) {
            cons.color = -1;
            return;
        }

        int available_count = __popcll(available_mask);
        int r = hash_color(cid, *iteration, available_count);

        uint64_t temp_mask = available_mask;
        for ( int i = 0; i < r; i++ ) {  // 跳过前 r 个颜色
            int pos = __ffsll(temp_mask) - 1;  // 64 位版 __ffs
            temp_mask &= ~(1ull << pos);
        }
        int chosen_color = __ffsll(temp_mask) - 1;  // 取第 r 个颜色
        cons.color = chosen_color;
        for (int v : cons.v) {
        atomicMax(&vertex_color_claimer[v * MAX_COLORS + chosen_color], cid);
        }
        // atomicMax(&vertex_color_claimer[cons.p0 * MAX_COLORS + chosen_color], cid);
        // atomicMax(&vertex_color_claimer[cons.p1 * MAX_COLORS + chosen_color], cid);
        // atomicMax(&vertex_color_claimer[cons.p2 * MAX_COLORS + chosen_color], cid);
        // atomicMax(&vertex_color_claimer[cons.p3 * MAX_COLORS + chosen_color], cid);
    }
}

static __global__ void k_verify_colors(
    int* __restrict__ constraint_colors,
    int* __restrict__ uncolored_count,
    const UnifiedNormalConstraint* __restrict__ constraints,
    const int* __restrict__ vertex_color_claimer,
    int num_constraints
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid >= num_constraints ) return;

    if ( constraint_colors[cid] == -1 ) {
        const auto& cons = constraints[cid];
        int c = cons.color;
        if ( c == -1 ) {
            atomicAdd(uncolored_count, 1);
            return;
        }

        bool success = true;
        // 如果我的槽位里不是我的 ID（可能是比我大的 ID）
        // 那我就认输
        for (int v : cons.v) {
            success &= (vertex_color_claimer[v * MAX_COLORS + c] == cid);
        }
        // if ( vertex_color_claimer[cons.p0 * MAX_COLORS + c] != cid ) success = false;
        // if ( vertex_color_claimer[cons.p1 * MAX_COLORS + c] != cid ) success = false;
        // if ( vertex_color_claimer[cons.p2 * MAX_COLORS + c] != cid ) success = false;
        // if ( vertex_color_claimer[cons.p3 * MAX_COLORS + c] != cid ) success = false;

        if ( success ) {
            constraint_colors[cid] = c;
        }
        else {
            atomicAdd(uncolored_count, 1);
        }
    }
}
static __global__ void k_update_colors(
    int* __restrict__ current_palette_size,
    int* __restrict__ uncolored_count,
    int* __restrict__ last_uncolored_count,
    int* __restrict__ iteration
) {
    int cid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( cid != 0 ) return;
    if ( *last_uncolored_count == *uncolored_count && *uncolored_count > 0 ) {
        if ( *current_palette_size < MAX_COLORS )
            (*current_palette_size)++;
    }
    *last_uncolored_count = *uncolored_count;
    (*iteration)++;
}

// static __global__ void k_print_colors(
//     const int* __restrict__ constraint_colors,
//     const CollisionConstraint* __restrict__ constraints,
//     int num_constraints
// ) {
//     int cid = blockIdx.x * blockDim.x + threadIdx.x;
//     if ( cid >= num_constraints ) return;
//
//     const CollisionConstraint& cons = constraints[cid];
//     printf("%d,[%d,%d,%d,%d],%d\n", cid, cons.p0, cons.p1,
//         cons.p2, cons.p3, constraint_colors[cid]);
//
// }
static __global__ void fill_inv_mass(
    float* __restrict__ invMass,
    const int* __restrict__ vertex_obj,
    const int* __restrict__ object_types,
    float* masses,
    char* mask,
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    if ( mask[vid] ) {
        invMass[vid] = 0.f;
        return;
    }
    int obj = vertex_obj[vid];
    if ( object_types[obj] > 0 ) {
        invMass[vid] = 0.f;
        return;
    }
    invMass[vid] = 1.0f / masses[vid];
}
static __global__ void solvePGS_UnifiedColorBatchKernel(
    UnifiedNormalConstraint* __restrict__ constraints,
    int* __restrict__ needs_more_iters,
    float3* __restrict__ vertices,
    const int* __restrict__ constraint_colors,
    const float* __restrict__ invMass,
    int color_offset,
    int num_constraints
) {
    // 只处理当前批次颜色的约束
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_constraints ) return;
    tid += color_offset;
    // if ( constraint_colors[tid] != color ) return;

    auto& c = constraints[tid];

    // 1. 计算当前的加权位置和 C_vec = \sum (w_i * x_i)
    float3 C_vec = make_float3(0.0f, 0.0f, 0.0f);

    #pragma unroll
    for ( int i = 0; i < 4; ++i ) {
        if ( c.w[i] != 0.0f ) { // 权重为0的顶点直接跳过（比如点-点约束）
            float3 pos = vertices[c.v[i]];
            C_vec = C_vec + c.w[i] * pos;
        }
    }

    // 2. 计算穿透量（违规量）
    // 约束函数 C(x) = C_vec · normal - delta >= 0
    float delta = 0.015f; // 
    float C_val = dot(C_vec, c.normal) - delta;
    if (c.A_ii == 0.f) {
        c.A_ii = invMass[c.v[0]] + c.w[1] * c.w[1] * invMass[c.v[1]]
    + c.w[2] * c.w[2] * invMass[c.v[2]] + c.w[3] * c.w[3] * invMass[c.v[3]] + 1e-8f;
    }

    // 3. 计算 LCP 乘子增量
    float delta_lambda = -C_val / c.A_ii;

    // 4. 投影 (保证碰撞力只能推开，不能拉近)
    float old_lambda = c.lambda;
    c.lambda = fmaxf(0.0f, old_lambda + delta_lambda);
    float actual_delta_lambda = c.lambda - old_lambda;

    // 5. 将冲量转换回位置更新 (x_i += invMass_i * J_i^T * delta_lambda)
    // 根据数学推导，J_i^T 就是 w_i * normal
    // printf("c.A_ii: %f\n",c.A_ii);
    if ( actual_delta_lambda != 0.0f ) {
        float3 impulse = c.normal * actual_delta_lambda;
        if ( actual_delta_lambda > 1e-4f ) {
            *needs_more_iters = 1;
            // atomicAdd(needs_more_iters, 1);
        }
        #pragma unroll
        for ( int i = 0; i < 4; ++i ) {
            if ( c.w[i] != 0.0f ) {
                int v_idx = c.v[i];
                float invM = invMass[v_idx];

                vertices[v_idx] = vertices[v_idx] + impulse * (c.w[i] * invM);
                // vertices[v_idx].x += invM * c.w[i] * impulse.x;
                // vertices[v_idx].y += invM * c.w[i] * impulse.y;
                // vertices[v_idx].z += invM * c.w[i] * impulse.z;
            }
        }
    }
}

static __global__ void record_color_offsets(
    int* __restrict__ lookup,
    const UnifiedNormalConstraint* __restrict__ constraints,
    int num_constraints
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_constraints ) return;
    auto c = constraints[idx].color;
    auto c_prev = idx > 0 ? constraints[idx - 1].color : -2;
    if ( c != c_prev ) {
        lookup[c + 1] = idx;// start from -1
    }
}
