#pragma once
#include <thrust/device_vector.h>
//==============================================================================
// BVH Structure
//==============================================================================

// 2D AABB Structure
struct AABB2D {
    float2 min; // .x = min_x, .y = min_y
    float2 max; // .x = max_x, .y = max_y

    __host__ __device__ AABB2D() {
        min = { FLT_MAX, FLT_MAX };
        max = { -FLT_MAX, -FLT_MAX };
    }
};

struct BVH2D {
    thrust::device_vector<int2> nodes;         // Tree structure (x: left, y: right)
    thrust::device_vector<AABB2D> aabbs;        // Bounding boxes
    thrust::device_vector<unsigned int> level_offsets; // Level offsets for traversal
    unsigned int num_primitives;
    unsigned int root_idx;
};



namespace lbvh2d {
struct NearestResult {
    unsigned int prim_idx;
    float dist_sq;
};

struct CollisionPair {
    unsigned int prim_a;
    unsigned int prim_b;
};
struct LocationResult {
    int prim_idx;
    float u, v, w;
    float dist_sq;
    int is_inside;
};

struct NearestEdgeResult {
    int idx;
    float dist_sq;
    float t;
};

struct IntersectionResult {
    int idx;
    float t;
    int found;
};
void initialize(unsigned int max_primitives);
void build_point_bvh(const thrust::device_vector<float2>& vertices, BVH2D& bvh);
void build_edge_bvh(const thrust::device_vector<float2>& vertices, const thrust::device_vector<int2>& edges, BVH2D& bvh);
void build_face_bvh(const thrust::device_vector<float2>& vertices, const thrust::device_vector<int3>& faces, BVH2D& bvh);
__global__ void query_nearest_kernel(
    const float2* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB2D* aabbs,
    unsigned int root_idx, NearestResult* results,
    const float2* primitive_data);

__global__ void query_location_kernel(
    const float2* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx,
    const float2* vertices, const int3* faces,
    LocationResult* results);

__global__ void query_nearest_edge_kernel(
    float2 query,
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx,
    const float2* vertices, const int2* edges,
    NearestEdgeResult* result);

__global__ void self_intersect_kernel(
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx, unsigned int num_edges,
    const float2* vertices, const int2* edges,
    IntersectionResult* result);

}
