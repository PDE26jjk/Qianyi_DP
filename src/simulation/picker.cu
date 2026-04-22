#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <thrust/execution_policy.h>

#include "common/atomic_utils.cuh"
#include "common/geometric_algorithms.h"


static __device__ __forceinline__ int get_picker_hash(int3 p, int table_size) {
    int h = (p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791);
    return abs(h) % table_size;
}

struct CollisionParams {
    float cell_size;
    float query_dist_sq; // 距离阈值的平方
    int num_triangles;
    int num_points;
};


static __global__ void insert_points_to_grid(
    const thrust::pair<int, float3>* __restrict__ points,
    PickerHashCell* table,
    CollisionResult_TP_Picker* results,
    float cell_size,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;

    if ( points[idx].first == -2 ) {

        results[idx].min_dist_sq = 1e30f;
        results[idx].tri_idx = -1;

        float3 p = points[idx].second;
        int3 grid_idx = make_int3(floorf(p.x / cell_size), floorf(p.y / cell_size),
            floorf(p.z / cell_size));

        int h = get_picker_hash(grid_idx, HASH_TABLE_SIZE);

        int old_count = atomicAdd(&table[h].count, 1);
        if ( old_count < MAX_PICKER_POINTS_PER_CELL ) {
            table[h].point_indices[old_count] = idx;
        }
    }
}
static __global__ void triangles_query_points(
    const float3* __restrict__ vertices,
    const int3* __restrict__ indices,
    thrust::pair<int, float3>* __restrict__ points,
    Mat3* __restrict__ p2t_offsets,
    PickerHashCell* table,
    CollisionResult_TP_Picker* results,
    int num_triangles,
    float query_dist_sq,
    float cell_size
) {
    int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tri_idx >= num_triangles ) return;

    // 1. 获取三角形顶点
    int3 idx3 = indices[tri_idx];
    float3 v0 = vertices[idx3.x];
    float3 v1 = vertices[idx3.y];
    float3 v2 = vertices[idx3.z];

    // 2. 计算三角形 AABB (扩充 query_dist 以防止漏检)
    float padding = sqrtf(query_dist_sq);
    float3 min_p = fmin3(v0, fmin3(v1, v2)) - make_float3(padding, padding, padding);
    float3 max_p = fmax3(v0, fmax3(v1, v2)) + make_float3(padding, padding, padding);

    int3 min_g = make_int3(floorf(min_p.x / cell_size), floorf(min_p.y / cell_size),
        floorf(min_p.z / cell_size));
    int3 max_g = make_int3(floorf(max_p.x / cell_size), floorf(max_p.y / cell_size),
        floorf(max_p.z / cell_size));

    // 4. 遍历 AABB 覆盖的所有格子
    for ( int x = min_g.x; x <= max_g.x; x++ ) {
        for ( int y = min_g.y; y <= max_g.y; y++ ) {
            for ( int z = min_g.z; z <= max_g.z; z++ ) {

                int h = get_picker_hash(make_int3(x, y, z), HASH_TABLE_SIZE);

                int count = table[h].count;
                if ( count == 0 ) continue;

                count = min(count, MAX_POINTS_PER_CELL);

                for ( int k = 0; k < count; k++ ) {
                    int p_idx = table[h].point_indices[k];
                    float3 pos = points[p_idx].second;

                    float3 closest_pt;
                    float dist_sq = point_triangle_sq_dist(pos, v0, v1, v2, &closest_pt);

                    if ( dist_sq < query_dist_sq ) {
                        if ( dist_sq < results[p_idx].min_dist_sq ) {
                            atomicMinFloat(&results[p_idx].min_dist_sq, dist_sq);

                            if ( results[p_idx].min_dist_sq == dist_sq ) {
                                results[p_idx].tri_idx = tri_idx;
                                points[p_idx].first = tri_idx;
                                Mat3 offsets{ v0 - pos, v1 - pos, v2 - pos };
                                p2t_offsets[p_idx] = offsets;
                            }
                        }
                    }
                }
            }
        }
    }
}
static __global__ void clear_hash_table(PickerHashCell* table) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < HASH_TABLE_SIZE ) {
        table[idx].count = 0;
    }
}

void SolverBase::init_picker() {
    picker_hash_table.resize(HASH_TABLE_SIZE);
    picker_collision_result.resize(HASH_TABLE_SIZE);
    clear_pick_triangle();
}

void SolverBase::check_picker() {
    if ( picker_collision_result.size() < pick_triangles.size() ) {
        picker_collision_result.resize(pick_triangles.size());
    }
    int block = 256;
    clear_hash_table<<<(HASH_TABLE_SIZE + block - 1) / block, block>>>(thrust::raw_pointer_cast(picker_hash_table.data()));
    int n = pick_size;
    float cell_size = params.cloth_edge_mean_length * 2;
    insert_points_to_grid<<<(n + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(pick_triangles.data()),
        thrust::raw_pointer_cast(picker_hash_table.data()),
        thrust::raw_pointer_cast(picker_collision_result.data()),
        cell_size, n
        );
    int num_triangles = params.nb_all_cloth_triangles;
    constexpr float query_dist = 0.5f * 0.01f;
    // 每个三角形去查它 AABB 内有没有点
    triangles_query_points<<<(num_triangles + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(vertices_world.data()),
        thrust::raw_pointer_cast(triangle_indices.data()),
        thrust::raw_pointer_cast(pick_triangles.data()),
        thrust::raw_pointer_cast(pick_triangle_offsets.data()),
        thrust::raw_pointer_cast(picker_hash_table.data()),
        thrust::raw_pointer_cast(picker_collision_result.data()),
        params.nb_all_cloth_triangles, query_dist * query_dist, cell_size
        );
}

// constexpr char pick_mesh_mask = 0b0010; // 2
__global__ void update_pick_triangles(
    char* __restrict__ vertices_mask,
    float3* __restrict__ vertices,
    const int3* __restrict__ indices,
    const thrust::pair<int, float3>* __restrict__ pick_triangles,
    const Mat3* __restrict__ pick_triangle_offsets,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [tri_index,pos] = pick_triangles[i];
        if ( tri_index < 0 ) continue;
        auto [o0,o1,o2] = pick_triangle_offsets[i].r;
        auto [v0,v1,v2] = indices[tri_index];
        vertices[v0] = o0 + pos;
        vertices[v1] = o1 + pos;
        vertices[v2] = o2 + pos;
        constexpr char mask = static_cast<char>(MaskBit::pick_mesh_mask);
        vertices_mask[v0] |= mask;
        vertices_mask[v1] |= mask;
        vertices_mask[v2] |= mask;
    }
}
__global__ void reset_pick_triangles(
    char* __restrict__ vertices_mask,
    const int3* __restrict__ indices,
    const thrust::pair<int, float3>* __restrict__ pick_triangles,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [tri_index,pos] = pick_triangles[i];
        if ( tri_index < 0 ) continue;
        auto [v0,v1,v2] = indices[tri_index];
        constexpr char mask = ~static_cast<char>(MaskBit::pick_mesh_mask);
        vertices_mask[v0] &= mask;
        vertices_mask[v1] &= mask;
        vertices_mask[v2] &= mask;
    }
}
__global__ void reset_pick_mask_kernel(
    char* __restrict__ vertices_mask, const int n) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        constexpr char mask = ~static_cast<char>(MaskBit::pick_mesh_mask);
        vertices_mask[i] &= mask;
    }
}
void SolverBase::check_update_pick() {
    std::lock_guard<std::mutex> lock(picker_mutex);
    std::lock_guard<std::mutex> lock2(pick_mutex);
    int threadsPerBlock = 256;
    int n = pick_size;
    bool has_pick_triangles = n > 0;
    bool has_picker = picker_size > 0;
    if ( has_picker )
        check_picker();

    if ( has_pick_triangles ) {
        update_pick_triangles<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            thrust::raw_pointer_cast(vertices_mask.data()),
            thrust::raw_pointer_cast(vertices_world.data()),
            thrust::raw_pointer_cast(triangle_indices.data()),
            thrust::raw_pointer_cast(pick_triangles.data()),
            thrust::raw_pointer_cast(pick_triangle_offsets.data()),
            n
            );
    }
}

void SolverBase::reset_pick_mask() {
    // std::lock_guard<std::mutex> lock(pick_mutex);
    int threadsPerBlock = 256;
    // int n = pick_size;
    // bool has_pick_triangles = n > 0;
    // if ( has_pick_triangles ) {
    // reset_pick_triangles<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
    //     thrust::raw_pointer_cast(vertices_mask.data()),
    //     thrust::raw_pointer_cast(triangle_indices.data()),
    //     thrust::raw_pointer_cast(pick_triangles.data()),
    //     n);
    // }
    int n = params.nb_all_cloth_vertices;
    reset_pick_mask_kernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        vertices_mask.data().get(), n);
}


__global__ void record_pick_triangle(
    int i,
    int mesh_idx, int tri_idx, float3 pos,
    Mat3* __restrict__ pick_triangle_offsets,
    thrust::pair<int, float3>* __restrict__ pick_triangles,
    const float3* __restrict__ vertices,
    const int3* __restrict__ indices,
    const int* __restrict__ triangle_index_offsets
) {
    int tri_index = tri_idx + triangle_index_offsets[mesh_idx];
    pick_triangles[i] = thrust::make_pair(tri_index, pos);
    auto [v0,v1,v2] = indices[tri_index];
    Mat3 offsets{ vertices[v0] - pos, vertices[v1] - pos, vertices[v2] - pos };
    pick_triangle_offsets[i] = offsets;
}
int SolverBase::add_pick_triangle(int mesh_index, int tri_index, float3 position) {
    // cudaDeviceSynchronize();
    std::lock_guard<std::mutex> lock(pick_mutex);
    if ( pick_size + 1 > max_pick_size ) return -1;
    int index = pick_size;
    pick_size++;
    record_pick_triangle<<<1,1>>>(index,
        mesh_index, tri_index, position,
        thrust::raw_pointer_cast(pick_triangle_offsets.data()),
        thrust::raw_pointer_cast(pick_triangles.data()),
        thrust::raw_pointer_cast(vertices_world.data()),
        thrust::raw_pointer_cast(triangle_indices.data()),
        thrust::raw_pointer_cast(triangle_index_offsets.data())
        );
    // cudaDeviceSynchronize();
    return index;
}
static __global__ void update_pick_position_kernel(thrust::pair<int, float3>* pick_triangles,
    int index, float3 pos) {
    pick_triangles[index].second = pos;
}
void SolverBase::update_pick_triangle(int index, float3 position) {
    std::lock_guard<std::mutex> lock(pick_mutex);
    // cudaDeviceSynchronize();
    // pick_triangles[index].second = position;
    if ( index >= 0 && index < max_pick_size )
        update_pick_position_kernel<<<1,1>>>(pick_triangles.data().get(), index, position);
    // cudaDeviceSynchronize();
}
static __global__ void invalidate_pick_triangle_kernel(thrust::pair<int, float3>* pick_triangles,
    int index) {
    pick_triangles[index].first = -1;
}
void SolverBase::remove_pick_triangle(int index) {
    std::unique_lock<std::mutex> lock(pick_mutex);
    // cudaDeviceSynchronize();
    if ( index >= max_pick_size || index < 0 ) return;
    // pick_triangles[index].first = -1;
    invalidate_pick_triangle_kernel<<<1,1>>>(pick_triangles.data().get(), index);
    // cudaDeviceSynchronize();
    // for ( const auto& pick_triangle : pick_triangles ) {
    //     if ( pick_triangle.first != -1 ) {
    //         return;
    //     }
    // }
    auto end_iter = pick_triangles.begin() + pick_size;
    auto iter = thrust::find_if(thrust::device, pick_triangles.begin(), end_iter,
        [] __device__ (const thrust::pair<int, float3>& p) {
            return p.first != -1;
        });
    lock.unlock();
    if ( iter == end_iter ) { clear_pick_triangle(); }
    // clear_pick_triangle();
}
void SolverBase::clear_pick_triangle() {
    std::lock_guard<std::mutex> lock(pick_mutex);
    cudaDeviceSynchronize();
    picker_size = pick_size = 0;
    pickers.resize(max_pick_size);
    pick_triangles.resize(max_pick_size);
    pick_triangle_offsets.resize(max_pick_size);
    cudaDeviceSynchronize();
}

int SolverBase::add_picker(float3 position) {
    std::lock_guard<std::mutex> lock(picker_mutex);
    int ptindex = add_pick_triangle(0, -2, position);
    if ( picker_size <= max_pick_size ) {
        pickers[picker_size] = ptindex;
        picker_size++;
        return picker_size - 1;
    }
    return -1;
}
void SolverBase::update_picker(int index, float3 position) {
    std::lock_guard<std::mutex> lock(picker_mutex);
    if ( pickers[index] != -1 ) {
        update_pick_triangle(pickers[index], position);
    }
}
void SolverBase::remove_picker(int index) {
    std::unique_lock<std::mutex> lock(picker_mutex);
    cudaDeviceSynchronize();
    if ( index >= max_pick_size || index < 0 ) return;
    remove_pick_triangle(pickers[index]);
    pickers[index] = -1;
    cudaDeviceSynchronize();
    // for ( int ptindex : pickers ) {
    //     if ( ptindex != -1 ) return;
    // }
    auto end_iter = pickers.begin() + picker_size;
    auto iter = thrust::find_if(
        thrust::device, pickers.begin(), end_iter,
        [] __device__ (int p) { return p != -1; });
    lock.unlock();
    if ( iter == end_iter ) {
        clear_picker();
    }
}
void SolverBase::clear_picker() {
    std::lock_guard<std::mutex> lock(picker_mutex);
    cudaDeviceSynchronize();
    // pickers.clear();
    picker_size = 0;
    cudaDeviceSynchronize();
}
