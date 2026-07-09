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
void calc_bounds(const thrust::device_vector<float2>& q, float (&h_bounds)[4]);
thrust::device_vector<float>& get_scene_bounds();
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

struct FullIntersectionResult {
    int curve_a;        // 曲线 A 的索引
    int section_a;      // 新增：曲线 A 内的局部 section 索引（从0算起）
    float t_a;          // 修改：交点在 section A 上的长度百分比 [0.0, 1.0]
    int curve_b;        // 曲线 B 的索引
    int section_b;      // 新增：曲线 B 内的局部 section 索引（从0算起）
    float t_b;          // 修改：交点在 section B 上的长度百分比 [0.0, 1.0]
    int state;          // 状态：1=第二根曲线来路在里，2=第二根曲线来路在外，0=其他
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
// __global__ void all_intersections_kernel(
//     const int2* nodes, const AABB2D* aabbs, unsigned int root_idx, unsigned int num_edges,
//     const float2* vertices, const int2* edges,
//     const int* edge_to_curve, const int* edge_local_idx,
//     const float* edge_lengths, const float* edge_prefix_sums,
//     const float* curve_total_lengths, const int* curve_num_edges, const int8_t* is_loops,
//     FullIntersectionResult* results, unsigned int max_results, unsigned int* out_count);
__global__ void all_intersections_kernel(
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx, unsigned int num_edges,
    const float2* vertices, const int2* edges,
    const int* edge_to_curve, const int* edge_local_idx,
    const float* edge_lengths, const float* edge_prefix_sums,
    const float* curve_total_lengths, const int* curve_num_edges, const int8_t* is_loops,
    const int* edge_to_section, const float* section_prefix_sums,
    const float* section_total_lengths, const int* curve_to_section_offset,
    FullIntersectionResult* results, unsigned int max_results, unsigned int* out_count);
__global__ void merge_overlaps_kernel(
    const float2* points, const int2* nodes, const AABB2D* aabbs, unsigned int root_idx,
    float threshold_sq, unsigned int* map, unsigned int n);
}
