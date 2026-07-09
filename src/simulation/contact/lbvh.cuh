#pragma once
#include <thrust/device_vector.h>

namespace lbvh3d {
struct AABB3D {
    float3 min;
    float3 max;
};
struct BVH3D {
    thrust::device_vector<int2> nodes;
    thrust::device_vector<AABB3D> aabbs;
    // thrust::device_vector<unsigned int> level_offsets;
    thrust::device_vector<unsigned int> parent;
    unsigned int root_idx;
    unsigned int num_primitives;
};

struct CollisionPair {
    unsigned int prim_a;
    unsigned int prim_b;
};


__device__ __forceinline__ bool aabb_overlap_3d(const AABB3D& a, const AABB3D& b) {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
        (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
        (a.min.z <= b.max.z && a.max.z >= b.min.z);
}
__device__ __forceinline__ float dist_sq_point_aabb_3d(float3 p, const AABB3D& b) {
    float dx = fmaxf(b.min.x - p.x, 0.0f) + fmaxf(p.x - b.max.x, 0.0f);
    float dy = fmaxf(b.min.y - p.y, 0.0f) + fmaxf(p.y - b.max.y, 0.0f);
    float dz = fmaxf(b.min.z - p.z, 0.0f) + fmaxf(p.z - b.max.z, 0.0f);
    return dx * dx + dy * dy + dz * dz;
}
__global__ void compute_morton_codes_kernel(const float3* centroids, unsigned int n,
    const float* bounds, unsigned int* codes, unsigned int* indices);
void compute_bounds(const float3* points,unsigned int n, float3& min_res, float3& max_res);
void compute_and_sort_by_morton_codes(float3* points,unsigned int n, unsigned int* sorted_indices);
unsigned int* get_sorted_indices();

void initialize(unsigned int max_primitives);
void cleanup();
void build_face_bvh(const thrust::device_vector<float3>& vertices,
    const thrust::device_vector<int3>& faces, BVH3D& bvh,const float3* additional_offset=nullptr);
void build_edge_bvh(const thrust::device_vector<float3>& vertices,
    const thrust::device_vector<int2>& edges, BVH3D& bvh,const float3* additional_offset=nullptr);
void refit_face_bvh(const float3* vertices,
    const thrust::device_vector<int3>& faces, BVH3D& bvh,const float3* additional_offset=nullptr);
void refit_edge_bvh(const float3* vertices,
    const thrust::device_vector<int2>& edges, BVH3D& bvh,const float3* additional_offset=nullptr);
// void build_point_bvh(const thrust::device_vector<float3>& vertices, BVH3D& bvh);
__global__ void query_nearest_face_kernel(
    const float3* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB3D* aabbs, unsigned int root_idx,
    const float3* vertices, const int3* faces,
    int* out_nearest_idx);
} // namespace lbvh3d

typedef lbvh3d::AABB3D AABB;
#define BVH_QUERY_LOOP(Q_AABB, STACK_SIZE, ...) \
    int* query_result = &query_results[i * result_size];\
    int query_count = 0; \
    unsigned int stack[STACK_SIZE]; \
    int sp = 0; \
    stack[sp++] = root_idx; \
    while (sp > 0 && query_count < result_size - 1) { \
        unsigned int node_idx = stack[--sp]; \
        if (!aabb_overlap_3d(Q_AABB, aabbs[node_idx])) continue; \
        int2 node = nodes[node_idx]; \
        if (node.y == 0) { \
            int prim_idx = node.x - 1; \
            __VA_ARGS__ \
        } else if (sp < STACK_SIZE - 2) { \
            stack[sp++] = node.x - 1; \
            stack[sp++] = node.y - 1; \
        } \
    } \
    query_result[0] = query_count
