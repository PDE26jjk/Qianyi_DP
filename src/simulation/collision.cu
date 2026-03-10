#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "constraint.cuh"
#include "solver_PCG.cuh"
#include "common/atomic_utils.cuh"
#include "common/geometric_algorithms.h"
#include "common/math_utils.h"


static __device__ __forceinline__ int get_hash(int3 p, int table_size) {
    int h = (p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791);
    return abs(h) % table_size;
}

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
template<bool InsertToGrid>
static __global__ void record_point_hash(
    int2* __restrict__ results,
    int* __restrict__ results_sort_key,
    int* __restrict__ results_size,
    PointHashCell* hash_table,
    const float3* __restrict__ points,
    float cell_size,
    float max_dist,
    int max_results_size,
    int hash_table_size,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;

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

static __global__ void record_point_hash_table_lookup(
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
    const int2* __restrict__ sorted_results,
    const float3* __restrict__ points,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    float cell_size,
    float max_dist,
    int hash_table_size,
    int max_results_size,
    int sorted_results_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= sorted_results_size ) return;

    float cell_size_inv = 1.0f / cell_size;
    int p_i = sorted_results[idx].y;
    float3 p = points[p_i];
    int grid_hash = sorted_results[idx].x;
    int3 grid_idx = make_int3(floorf(p.x * cell_size_inv), floorf(p.y * cell_size_inv),
        floorf(p.z * cell_size_inv));
    int centroid_hash = get_hash(grid_idx, hash_table_size);
    if ( centroid_hash != grid_hash ) return;// not centroid
    float dist_sq = max_dist * max_dist;
    for ( int i = idx + 1; i < sorted_results_size && sorted_results[i].x == grid_hash; ++i ) {
        int p2_i = sorted_results[i].y;
        if ( p2_i <= p_i ) continue;
        float3 p2 = points[p2_i];
        if ( len_sq(p - p2) < dist_sq ) {
            if ( has_edge(p_i, p2_i, edge_lookup, dir_edges) ) continue;
            int res_id = atomicAdd(results_size, 1);
            if ( res_id < max_results_size )
                results[res_id] = CollisionResult_PP(p_i, p2_i);
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
static __global__ void triangles_query_points(
    CollisionResult_TP* results,
    int* results_size,
    const int3* __restrict__ indices,
    const float3* __restrict__ vertices,
    const int2* __restrict__ points_hashes,
    const int* __restrict__ hash_lookup,
    float cell_size,
    float query_dist_sq,
    int hash_table_size,
    int points_hashes_size,
    int max_results_size,
    int num_triangles
) {
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tri_idx >= num_triangles ) return;

    int3 idx3 = indices[tri_idx];
    float3 v0 = vertices[idx3.x];
    float3 v1 = vertices[idx3.y];
    float3 v2 = vertices[idx3.z];

    float cell_size_inv = 1.0f / cell_size;
    float padding = sqrtf(query_dist_sq);
    float3 min_p = fmin3(v0, fmin3(v1, v2)) - make_float3(padding, padding, padding);
    float3 max_p = fmax3(v0, fmax3(v1, v2)) + make_float3(padding, padding, padding);

    int3 min_g = make_int3(floorf(min_p.x * cell_size_inv), floorf(min_p.y * cell_size_inv),
        floorf(min_p.z * cell_size_inv));
    int3 max_g = make_int3(floorf(max_p.x * cell_size_inv), floorf(max_p.y * cell_size_inv),
        floorf(max_p.z * cell_size_inv));

    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {

                int h = get_hash(make_int3(x, y, z), hash_table_size);

                if ( hash_lookup[h] == -1 ) continue;

                for ( int k = hash_lookup[h]; k < points_hashes_size && points_hashes[k].x == h; k++ ) {
                    int p_idx = points_hashes[k].y;
                    if ( p_idx == idx3.x || p_idx == idx3.y || p_idx == idx3.z ) continue;

                    float3 pos = vertices[p_idx];
                    int grid_h = get_hash(make_int3(floorf(pos.x * cell_size_inv), floorf(pos.y * cell_size_inv),
                        floorf(pos.z * cell_size_inv)), hash_table_size);
                    if ( grid_h != h ) continue;

                    float3 closest_pt;
                    float dist_sq = point_triangle_sq_dist(pos, v0, v1, v2, &closest_pt);

                    if ( dist_sq < query_dist_sq ) {
                        int tp_idx = atomicAdd(results_size, 1);
                        if ( tp_idx < max_results_size ) {
                            auto tp = CollisionResult_TP();
                            tp.vert_idx = p_idx;
                            tp.tri_idx = tri_idx;
                            float3 diff = pos - closest_pt;
                            float d = sqrtf(dist_sq);
                            tp.normal = (d > 1e-6f) ? diff * (1.0f / d) : make_float3(0, 0, 1);
                            tp.min_dist_sq = query_dist_sq;
                            results[tp_idx] = tp;
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



void SolverBase::contact_handle() {
    int block = 256;
    int n = (int)point_hash_table_size;
    clear_hash_table<<<(n + block - 1) / block, block>>>(
        point_hash_table.data().get(),
        n
        );
    n = params.nb_all_cloth_vertices;
    // 1. collect pp
    float cell_size = params.cloth_edge_mean_length * 1.414f;
    // const float cell_size = 5.f * 0.001f;
    insert_points_to_grid<<<(n + block - 1) / block, block>>>(
        vertices_world.data().get(),
        point_hash_table.data().get(),
        cell_size, point_hash_table_size,
        n);

    const float dist = 5.f * 0.001f;
    pp_result_size.assign(1, 0);
    collect_pp<<<(n + block - 1) / block, block>>>(
        pp_collision_result.data().get(),
        pp_result_size.data().get(),
        // sort_key_temp.data().get(),
        // sort_value_temp.data().get(),
        // sort_result_size.data().get(),
        vertices_world.data().get(),
        point_hash_table.data().get(),
        dist * dist, n,
        max_pp_result_size,
        point_hash_table_size,
        cell_size);
    // 2. graph coloring
    int result_size;
    cudaMemcpy(&result_size, sort_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    result_size = min(result_size, max_pp_result_size);
    // if ( result_size != 0 ) {
    //     thrust::stable_sort_by_key(thrust::device, sort_key_temp.begin(), sort_key_temp.begin() + result_size,
    //         sort_value_temp.begin());
    // }

    // 3. sort graph

    // 4. GS iter
}

static __global__ void soft_phase_constraint_edge(
    float3* __restrict__ grads,
    int* __restrict__ num_violations,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edges,
    float max_dist,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_edges ) return;
    constexpr float rho = 10.f;
    auto [p0_i,p1_i] = edges[idx];
    float3 p0 = vertices[p0_i], p1 = vertices[p1_i];

    float dist = norm((p0 - p1));
    if ( dist > max_dist ) {
        float3 dir = (p0 - p1) / (dist + 1e-8f);
        float violation = dist - max_dist;
        float3 grad = dir * violation * rho;

        atomicAddFloat3(&grads[p0_i], grad);
        atomicAddFloat3(&grads[p1_i], -grad);

        atomicAdd(num_violations, 1);
    }
}
static __global__ void soft_phase_constraint_pp(
    float3* __restrict__ grads,
    int* __restrict__ num_violations,
    const CollisionResult_PP* __restrict__ near_point_pairs,
    const float3* __restrict__ vertices,
    float min_dist,
    int num_point_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_point_pairs ) return;
    constexpr float rho = 2.f;

    int p1_i = near_point_pairs[idx].p1, p2_i = near_point_pairs[idx].p2;
    float3 p1 = vertices[p1_i], p2 = vertices[p2_i];

    float dist = norm((p1 - p2));
    if ( dist < min_dist + 1e-5f ) {
        // L1
        float3 dir = (p2 - p1) / (dist + 1e-8f);
        float3 grad = dir * rho;
        // L2
        // float violation = max((min_dist - dist) * rho, min_dist * 0.05f);
        // float3 grad = dir * (violation);

        atomicAddFloat3(&grads[p1_i], grad);
        atomicAddFloat3(&grads[p2_i], -grad);

        atomicAdd(num_violations, 1);
        printf("%d,%d, %f, %f\n", p1_i, p2_i, dist, min_dist);
    }
}

// Cloth point vs other triangles of objects, only update cloth points
static __global__ void soft_phase_constraint_tp(
    CollisionResult_TP* near_point_tri_pairs,
    float3* __restrict__ grads,
    const float3* __restrict__ vertices,
    const int3* __restrict__ triangle_indices,
    float min_dist,
    int num_point_tri_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_point_tri_pairs ) return;
    constexpr float rho = 1.f;

    auto near_point_tri_pair = near_point_tri_pairs[idx];
    int p_i = near_point_tri_pair.vert_idx, t_i = near_point_tri_pair.tri_idx;
    float3 p = vertices[p_i];
    float3 v0 = vertices[triangle_indices[t_i].x];
    float3& normal = near_point_tri_pair.normal;
    if ( dot(p - v0, normal) < min_dist ) {
        float3 grad = normal * rho;
        atomicAddFloat3(&grads[p_i], -grad);
    }
}
static __global__ void soft_phase_update(
    float3* __restrict__ points_collision,
    const float3* __restrict__ points_y,
    const float3* __restrict__ grads,
    float step_length,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;
    float3 v = points_y[idx];
    float3 coll_v = points_collision[idx];
    float3 grad = (coll_v - v) * 2.f;
    grad = grad + grads[idx];
    points_collision[idx] = coll_v - grad * step_length;

}

__device__ __forceinline__ void calc_barrier_f_and_prime(
    float x, float eps, float& f_val, float& f_prime) {
    if ( x >= eps ) {
        f_val = eps;
        f_prime = 0.0f;
    }
    else if ( x <= 1e-6f ) {
        x = 1e-6f;
        f_val = x;
        f_prime = 1.0f;
    }
    else {
        float a3 = -1.0f / (eps * eps);
        float a2 = 1.0f / eps;
        float a1 = 1.0f;
        float a0 = 1e-6f;

        float xx = x * x;
        f_val = a3 * xx * x + a2 * xx + +a1 * x + a0;
        f_prime = 3.0f * a3 * xx + 2.0f * a2 * x + a1;
    }
}

// Hard Phase: 计算 Log-Barrier 的梯度
static __global__ void hard_phase_constraint_pp_grad(
    float3* __restrict__ barrier_grads,
    const CollisionResult_PP* __restrict__ near_point_pairs,
    const float3* __restrict__ vertices,
    float min_dist,
    float eps_slack,
    float mu,
    int num_point_pairs
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_point_pairs ) return;

    int p1_i = near_point_pairs[idx].p1;
    int p2_i = near_point_pairs[idx].p2;
    float3 p1 = vertices[p1_i];
    float3 p2 = vertices[p2_i];

    float dist_sq = len_sq(p1 - p2);
    float min_dist_sq = min_dist * min_dist;

    // 约束函数 c_ij(x) = ||x_ij||^2 - B^2
    float x = dist_sq - min_dist_sq;

    if ( x < eps_slack ) {
        float f_val, f_prime;
        calc_barrier_f_and_prime(x, eps_slack, f_val, f_prime);

        // 计算障碍函数能量 -mu * log(f(x)) 针对 p1 的梯度
        // Grad = -mu * (1 / f_val) * f_prime * \nabla c_ij(p1)
        // \nabla c_ij(p1) = 2 * (p1 - p2)
        float coeff = -mu * (f_prime / f_val) * 2.0f;
        float3 grad_p1 = (p1 - p2) * coeff;

        atomicAddFloat3(&barrier_grads[p1_i], grad_p1);
        atomicAddFloat3(&barrier_grads[p2_i], -grad_p1);
    }
}

// Hard Phase: 综合数据保真项和障碍梯度进行位置更新
static __global__ void hard_phase_update(
    float3* __restrict__ points_curr, // x^(l)
    const float3* __restrict__ points_init, // x^{init} (Soft Phase 之前的目标预测位置)
    const float3* __restrict__ barrier_grads,
    float step_length,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;

    float3 v_curr = points_curr[idx];
    float3 v_init = points_init[idx];

    // 目标函数：||x - x^{init}||^2 + Barrier
    // 对 x 求导：2 * (x - x^{init}) + \nabla Barrier
    float3 total_grad = (v_curr - v_init) * 2.0f + barrier_grads[idx];

    points_curr[idx] = v_curr - total_grad * step_length;
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
            auto old_vel_dir = normalized(points_y[i] - points_x[i]);
            auto new_vel = (x - points_x[i]) * h_inv;
            float proj_vel_len = max(0.f, dot(old_vel_dir, new_vel));
            new_vel = old_vel_dir * proj_vel_len;
            velocities[i] = new_vel;

            points_y[i] = x;
        }
        else {
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
    }
}


//L. Wu, B. Wu, Y. Yang, and H. Wang, "A Safe and Fast Repulsion Method for GPU-based Cloth Self Collisions," ACM Trans. Graph., vol. 40, no. 1, pp. 1–18, Feb. 2021, doi: 10.1145/3430025.
void SolverBase::collision_Wu2021() {
    int block = 256;
    int n = (int)point_hash_table_size;
    clear_hash_table<<<(n + block - 1) / block, block>>>(
        point_hash_table.data().get(),
        n
        );
    int vertex_size = params.nb_all_cloth_vertices;
    n = vertex_size;
    // 1. collect pp
    float max_dist = params.cloth_edge_mean_length;
    float cell_size = max_dist * 3.f;
    float max_dist_edge = params.cloth_edge_mean_length * 2.f;

    float3* points_x = vertices_old.data().get();
    float3* points_y = vertices_world.data().get();
    cudaMemsetAsync(sort_result_size.data().get(), 0, sizeof(int));
    record_point_hash<true><<<(n + block - 1) / block, block>>>(
        point_hashes.data().get(),
        sort_key_temp2.data().get(),
        sort_result_size.data().get(),
        point_hash_table.data().get(),
        points_y,
        cell_size,
        max_dist,
        max_sort_result_size,
        point_hash_table_size,
        n);
    int result_size;
    cudaMemcpy(&result_size, sort_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    result_size = min(result_size, max_sort_result_size);
    thrust::sort_by_key(thrust::device, sort_key_temp2.begin(), sort_key_temp2.begin() + result_size, point_hashes.begin());
    cudaMemsetAsync(pp_result_size.data().get(), 0, sizeof(int));
    collect_pp_sorted<<<(result_size + block - 1) / block, block>>>(
        pp_collision_result.data().get(),
        pp_result_size.data().get(),
        point_hashes.data().get(),
        points_y,
        edge_lookup.data().get(),
        dir_edges.data().get(),
        cell_size,
        max_dist,
        point_hash_table_size,
        max_pp_result_size,
        result_size
        );
    cudaMemcpy(&result_size, pp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    result_size = min(result_size, max_pp_result_size);
    int edge_size = params.nb_all_cloth_edges;

    // 2. soft phase
    float3* grads = velocities.data().get();
    float3* points_collision = temp_vertices_f3.data().get();
    cudaMemcpyAsync(points_collision, points_y, vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    n = vertex_size;
    int* num_violations = sort_result_size.data().get();
    int num_violations_h;
    int max_soft_phase_step = 15;
    int soft_phase_step = 0;
    for ( ; soft_phase_step < max_soft_phase_step; ++soft_phase_step ) {
        cudaMemsetAsync(num_violations, 0, sizeof(int));
        cudaMemsetAsync(grads, 0, vertex_size * sizeof(float3));
        soft_phase_constraint_edge<<<(edge_size + block - 1) / block, block>>>(
            grads, num_violations,
            points_collision,
            edges.data().get(),
            max_dist_edge, edge_size);
        soft_phase_constraint_pp<<<(result_size + block - 1) / block, block>>>(
            grads, num_violations,
            pp_collision_result.data().get(),
            points_collision,
            max_dist, result_size);
        soft_phase_update<<<(n + block - 1) / block, block>>>(
            points_collision,
            points_y,
            grads,
            max_dist * 0.0035f,
            n);
        cudaMemcpy(&num_violations_h, num_violations, sizeof(int), cudaMemcpyDeviceToHost);
        if ( num_violations_h == 0 ) break;
    }
    if ( num_violations_h == 0 ) {
        if ( soft_phase_step == 0 ) {
            // std::cout << "No need to soft_phase" << std::endl;
        }
        else
            std::cout << "soft_phase succeed, soft_phase_step: " << soft_phase_step << std::endl;
    }
    else
        std::cout << "soft_phase failed, soft_phase_step: " << soft_phase_step <<
            ", num_violations: " << num_violations_h << std::endl;
    update_end_collision<<<(n + block - 1) / block, block>>>(
        points_y,
        velocities.data().get(),
        points_x,
        points_collision,
        vertices_mask.data().get(),
        dt, n);
}

void SolverPCG::contact_handle() {
    int block = 256;
    int n = (int)point_hash_table_size;
    clear_hash_table<<<(n + block - 1) / block, block>>>(
        point_hash_table.data().get(),
        n
        );
    int vertex_size = params.nb_all_cloth_vertices;
    n = vertex_size;
    // 1. collect pp
    float max_dist = params.cloth_edge_mean_length;
    float cell_size = max_dist * 3.f;
    float max_dist_edge = params.cloth_edge_mean_length * 2.f;

    float3* points_x = vertices_old.data().get();
    float3* points_y = vertices_world.data().get();
    cudaMemsetAsync(sort_result_size.data().get(), 0, sizeof(int));
    record_point_hash<false><<<(n + block - 1) / block, block>>>(
        point_hashes.data().get(),
        sort_key_temp2.data().get(),
        sort_result_size.data().get(),
        point_hash_table.data().get(),
        points_y,
        cell_size,
        max_dist,
        max_sort_result_size,
        point_hash_table_size,
        n);
    int point_hashes_size_h;
    cudaMemcpy(&point_hashes_size_h, sort_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    point_hashes_size_h = min(point_hashes_size_h, max_sort_result_size);
    thrust::sort_by_key(thrust::device, sort_key_temp2.begin(), sort_key_temp2.begin() + point_hashes_size_h,
        point_hashes.begin());

    // build hash_table lookup
    cudaMemsetAsync(hash_table_lookup.data().get(), -1, sizeof(int) * point_hash_table_size);
    record_point_hash_table_lookup<<<(point_hashes_size_h + block - 1) / block, block>>>(
        hash_table_lookup.data().get(),
        point_hashes.data().get(),
        point_hash_table_size, point_hashes_size_h
        );
    cudaMemsetAsync(pp_result_size.data().get(), 0, sizeof(int));
    CollisionResult_PP* point_pairs = pp_collision_result.data().get();
    collect_pp_sorted<<<(point_hashes_size_h + block - 1) / block, block>>>(
        point_pairs,
        pp_result_size.data().get(),
        point_hashes.data().get(),
        points_y,
        edge_lookup.data().get(),
        dir_edges.data().get(),
        cell_size,
        max_dist,
        point_hash_table_size,
        max_pp_result_size,
        point_hashes_size_h
        );
    int result_size;
    cudaMemcpy(&result_size, pp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    result_size = min(result_size, max_pp_result_size);
    
    if ( result_size > 0 ) {
        compute_collision_constraint_point_point<<<(result_size + block - 1) / block, block>>>(
            Jx.data().get(),
            Jx_diag.data().get(),
            forces.data().get(),
            velocities.data().get(),
            point_pairs, points_y, max_dist, dt, result_size);
    }
    max_dist = params.cloth_edge_mean_length * 0.5f;
    n = params.nb_all_triangles;
    cudaMemsetAsync(tp_result_size.data().get(), 0, sizeof(int));
    triangles_query_points<<<(n + block - 1) / block, block>>>(
        tp_collision_result.data().get(),
        tp_result_size.data().get(),
        triangle_indices.data().get(),
        points_y,
        point_hashes.data().get(),
        hash_table_lookup.data().get(), cell_size,
        max_dist * max_dist, point_hash_table_size,
        point_hashes_size_h, max_tp_result_size, n);

    cudaMemcpy(&result_size, tp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    result_size = min(result_size, max_tp_result_size);
    if ( result_size > 0 ) {
        // std::cout <<  result_size << " triangles" << std::endl; 
        compute_collision_constraint_triangle_point_plane<<<(result_size + block - 1) / block, block>>>(
            // Jx.data().get(),
            Jx_diag.data().get(),
            forces.data().get(),
            tp_collision_result.data().get(),
            points_y, triangle_indices.data().get(),
            params.nb_all_cloth_vertices,
            result_size);
    }
}

void SolverBase::init_collision() {
    point_hash_table_size = next_prime((uint32_t)params.nb_all_cloth_vertices);
    point_hash_table.resize(point_hash_table_size);
    max_pp_result_size = params.nb_all_cloth_vertices;
    max_tp_result_size = params.nb_all_triangles;
    max_sort_result_size = max_pp_result_size * 8;
    point_hashes.resize(max_sort_result_size);
    sort_key_temp2.resize(max_sort_result_size);
    sort_result_size.resize(1);
    pp_result_size.resize(1);
    tp_result_size.resize(1);

    pp_collision_result.resize(max_pp_result_size);
    tp_collision_result.resize(max_tp_result_size);
    temp_vertices_f3.resize(params.nb_all_cloth_vertices);
    hash_table_lookup.resize(point_hash_table_size);
    // temp_edge_f3.resize(params.nb_all_cloth_edges);
    // vertex_colors.resize(params.nb_all_cloth_vertices);
    // sort_key_temp.resize(max_pp_result_size);
    // sort_value_temp.resize(max_pp_result_size);
    // pp_result_size.resize(1);
}
