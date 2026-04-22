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
    thrust::device_vector<unsigned int> level_offsets;
    unsigned int root_idx;
    unsigned int num_primitives;
};

struct CollisionPair {
    unsigned int prim_a;
    unsigned int prim_b;
};



void initialize(unsigned int max_primitives);
void build_face_bvh(const thrust::device_vector<float3>& vertices, const thrust::device_vector<int3>& faces, BVH3D& bvh);
void build_edge_bvh(const thrust::device_vector<float3>& vertices, const thrust::device_vector<int2>& edges, BVH3D& bvh);
void build_point_bvh(const thrust::device_vector<float3>& vertices, BVH3D& bvh);
__global__ void query_nearest_face_kernel(
    const float3* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB3D* aabbs, unsigned int root_idx,
    const float3* vertices, const int3* faces,
    int* out_nearest_idx);
} // namespace lbvh3d
