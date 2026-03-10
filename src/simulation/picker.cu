#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>

#include "common/atomic_utils.cuh"
#include "common/geometric_algorithms.h"


static __device__ __forceinline__ int get_hash(int3 p, int table_size) {
	int h = (p.x * 73856093) ^ (p.y * 19349663) ^ (p.z * 83492791);
	return abs(h) % table_size;
}

struct CollisionParams
{
	float cell_size;
	float query_dist_sq; // 距离阈值的平方
	int num_triangles;
	int num_points;
};


static __global__ void insert_points_to_grid(
	const thrust::pair<int, float3>* __restrict__ points,
	PickerHashCell* table,
	CollisionResult_TP* results,
	float cell_size,
	int num_points
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= num_points) return;

	if (points[idx].first == -2) {
		
		results[idx].min_dist_sq = 1e30f;
		results[idx].tri_idx = -1;
		results[idx].normal = make_float3(0, 0, 1);

		float3 p = points[idx].second;
		int3 grid_idx = make_int3(floorf(p.x / cell_size), floorf(p.y / cell_size),
			floorf(p.z / cell_size));

		int h = get_hash(grid_idx, HASH_TABLE_SIZE);

		int old_count = atomicAdd(&table[h].count, 1);
		if (old_count < MAX_PICKER_POINTS_PER_CELL) {
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
	CollisionResult_TP* results,
	int num_triangles,
	float query_dist_sq,
	float cell_size
) {
	int tri_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (tri_idx >= num_triangles) return;

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
	for (int x = min_g.x; x <= max_g.x; x++) {
		for (int y = min_g.y; y <= max_g.y; y++) {
			for (int z = min_g.z; z <= max_g.z; z++) {

				int h = get_hash(make_int3(x, y, z), HASH_TABLE_SIZE);

				int count = table[h].count;
				if (count == 0) continue;

				count = min(count, MAX_POINTS_PER_CELL);

				for (int k = 0; k < count; k++) {
					int p_idx = table[h].point_indices[k];
					float3 pos = points[p_idx].second;

					float3 closest_pt;
					float dist_sq = point_triangle_sq_dist(pos, v0, v1, v2, &closest_pt);

					if (dist_sq < query_dist_sq) {
						if (dist_sq < results[p_idx].min_dist_sq) {
							atomicMinFloat(&results[p_idx].min_dist_sq, dist_sq);

							if (results[p_idx].min_dist_sq == dist_sq) {
								results[p_idx].tri_idx = tri_idx;
								// 计算法线
								// float3 diff = p - closest_pt;
								// float d = sqrtf(dist_sq);
								// results[p_idx].normal = (d > 1e-6f) ? diff * (1.0f / d) : make_float3(0, 0, 1);
								// Make a lambda
								points[p_idx].first = tri_idx;
								Mat3 offsets{v0 - pos, v1 - pos, v2 - pos};
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
	if (idx < HASH_TABLE_SIZE) {
		table[idx].count = 0;
	}
}

void SolverBase::init_picker() {
	picker_hash_table.resize(HASH_TABLE_SIZE);
	picker_collision_result.resize(HASH_TABLE_SIZE);
	clear_pick_triangle();
}

void SolverBase::check_picker() {
	if (picker_collision_result.size() < pick_triangles.size()) {
		picker_collision_result.resize(pick_triangles.size());
	} 
	int block = 256;
	clear_hash_table<<<(HASH_TABLE_SIZE + block - 1) / block, block>>>(thrust::raw_pointer_cast(picker_hash_table.data()));
	int n = (int)pick_triangles.size();
	float cell_size = params.cloth_edge_mean_length * 2;
	insert_points_to_grid<<<(n + block - 1) / block, block>>>(
		thrust::raw_pointer_cast(pick_triangles.data()),
		thrust::raw_pointer_cast(picker_hash_table.data()),
		thrust::raw_pointer_cast(picker_collision_result.data()),
		cell_size, pick_triangles.size()
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

constexpr char pick_mesh_mask = 0b0010; // 2
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
        constexpr char mask = pick_mesh_mask;
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
        constexpr char mask = ~pick_mesh_mask;
        vertices_mask[v0] &= mask;
        vertices_mask[v1] &= mask;
        vertices_mask[v2] &= mask;
    }
}
void SolverBase::check_update_pick() {
    int threadsPerBlock = 256;
	int n = (int)pick_triangles.size();
    bool has_pick_triangles = n > 0;
    bool has_picker = (int)pickers.size() > 0;
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
	int threadsPerBlock = 256;
	int n = (int)pick_triangles.size();
	bool has_pick_triangles = n > 0;
    if ( has_pick_triangles ) {
        n = (int)pick_triangles.size();
        reset_pick_triangles<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            thrust::raw_pointer_cast(vertices_mask.data()),
            thrust::raw_pointer_cast(triangle_indices.data()),
            thrust::raw_pointer_cast(pick_triangles.data()),
            n);
    }
}
