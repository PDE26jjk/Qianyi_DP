// lbvh3d/bvh.h
// CUDA LBVH builder, refitter and rebuilder.
// Depends on: common/vec_math.h (float3 operators), common/cuda_utils.h (check_cuda)

#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include "common/vec_math.h"

#define BVH_QUERY_STACK_SIZE (32)

namespace lbvh3dtest {

struct BVHPackedNodeHalf {
    float x, y, z;
    unsigned int i: 31;          // index or range start
    unsigned int b: 1; // leaf flag (1 = leaf, 0 = internal)
};

struct BVH {
    BVHPackedNodeHalf* node_lowers = nullptr;
    BVHPackedNodeHalf* node_uppers = nullptr;
    int* node_parents = nullptr;
    int* node_counts = nullptr;       // for refit (atomic counter)
    int* primitive_indices = nullptr;
    int* root = nullptr;              // single device int
    int num_items = 0;
    int max_nodes = 0;
    int num_leaf_nodes = 0;
    int num_nodes = 0;
    int max_depth = 0;
    int leaf_size = 8;
    // externally owned item bounds
    const float3* item_lowers = nullptr;
    const float3* item_uppers = nullptr;
};

// Helper: create a packed node from bounds and index
inline __host__ __device__ BVHPackedNodeHalf make_node(const float3& bound, int index, bool isLeaf) {
    BVHPackedNodeHalf n;
    n.x = bound.x; n.y = bound.y; n.z = bound.z;
    n.i = index;
    n.b = isLeaf ? 1u : 0u;
    return n;
}
inline __host__ __device__ void make_node(volatile BVHPackedNodeHalf* n, const float3& bound, int index, bool isLeaf) {
    n->x = bound.x;
    n->y = bound.y;
    n->z = bound.z;
    n->i = index;
    n->b = isLeaf ? 1 : 0;
}

// Simple AABB helper
struct Bounds3 {
    float3 lower, upper;
    __host__ __device__ void add_bounds(const float3& lo, const float3& up) {
        lower = fmin3(lower, lo);
        upper = fmax3(upper, up);
    }
};

// Main API
void bvh_build(cudaStream_t stream, const float3* d_lowers, const float3* d_uppers, int num_items, int leaf_size, BVH& bvh);
void bvh_destroy(cudaStream_t stream, BVH& bvh);
void bvh_refit(cudaStream_t stream, BVH& bvh);
void bvh_rebuild(cudaStream_t stream, BVH& bvh);

} // namespace lbvh3d