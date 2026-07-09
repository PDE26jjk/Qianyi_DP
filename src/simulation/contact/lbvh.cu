// Adapted from ZOZO ppf-contact-solver and NVIDIA Warp
// code: GLM-5, PDE26jjk

#include "common/cuda_utils.h"
#include "common/vec_math.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
// #include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include "lbvh.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cassert>

#include "common/geometric_algorithms.h"
#include "../cuda_tools/sort.h"

namespace lbvh3d {
// Internal storage
namespace storage {
struct Storage {
    unsigned int max_primitives = 0;
    thrust::device_vector<unsigned int> morton_codes;
    thrust::device_vector<unsigned int> sorted_indices;
    // Combined storage for centroids (float3)
    thrust::device_vector<float3> centroids;
    // thrust::device_vector<unsigned int> parent;
    // thrust::device_vector<unsigned int> depth;
    thrust::device_vector<unsigned int> node_indices_scratch;
    thrust::device_vector<unsigned int> child_count;
    // Scene bounds: [min_x, min_y, min_z, max_x, max_y, max_z]
    thrust::device_vector<float> scene_bounds;
};
static Storage* inst = nullptr;
inline Storage& instance() {
    if ( inst == nullptr ) {
        inst = new Storage();
    }
    return *inst;
}
}

//==============================================================================
// Helper Functors for Thrust
//==============================================================================
struct float3_min {
    __host__ __device__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
    }
};
struct float3_max {
    __host__ __device__ float3 operator()(const float3& a, const float3& b) const {
        return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
    }
};
//==============================================================================
// Morton Code 3D (10 bits per axis → 30-bit code, shifted left by 2 to fill 32 bits)
//==============================================================================
__device__ __host__ unsigned int expand_bits_3d(unsigned int v) {
    v &= 0x000003FF;
    v = (v | (v << 16)) & 0x030000FF;
    v = (v | (v << 8)) & 0x0300F00F;
    v = (v | (v << 4)) & 0x030C30C3;
    v = (v | (v << 2)) & 0x09249249;
    return v;
}
__device__ __host__ __forceinline__ unsigned int morton_code_3d(unsigned int x, unsigned int y, unsigned int z) {
    return (expand_bits_3d(z) << 2) | (expand_bits_3d(y) << 1) | expand_bits_3d(x);
}
//==============================================================================
// Kernels
//==============================================================================
// Centroid Calculations (Writing to float3)
__global__ void compute_face_centroids_kernel(const float3* vertices, const int3* faces,
    unsigned int n, float3* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int3 f = faces[i];
    float3 v0 = vertices[f.x];
    float3 v1 = vertices[f.y];
    float3 v2 = vertices[f.z];
    centroids[i] = (v0 + v1 + v2) / 3.0f;
}
__global__ void compute_edge_centroids_kernel(const float3* vertices, const int2* edges,
    unsigned int n, float3* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int2 e = edges[i];
    float3 v0 = vertices[e.x];
    float3 v1 = vertices[e.y];
    centroids[i] = (v0 + v1) / 2.0f;
}
__global__ void compute_point_centroids_kernel(const float3* vertices, unsigned int n, float3* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    centroids[i] = vertices[i];
}

__global__ void compute_morton_codes_kernel(const float3* centroids, unsigned int n,
    const float* bounds, unsigned int* codes, unsigned int* indices) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    float min_x = bounds[0], min_y = bounds[1], min_z = bounds[2];
    float max_x = bounds[3], max_y = bounds[4], max_z = bounds[5];
    float3 c = centroids[i];
    float sx = (max_x - min_x) > 1e-10f ? 1023.0f / (max_x - min_x) : 0.0f;
    float sy = (max_y - min_y) > 1e-10f ? 1023.0f / (max_y - min_y) : 0.0f;
    float sz = (max_z - min_z) > 1e-10f ? 1023.0f / (max_z - min_z) : 0.0f;
    unsigned int ix = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (c.x - min_x) * sx));
    unsigned int iy = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (c.y - min_y) * sy));
    unsigned int iz = (unsigned int)fminf(1023.0f, fmaxf(0.0f, (c.z - min_z) * sz));

    codes[i] = morton_code_3d(ix, iy, iz);
    indices[i] = i;
}
// Tree Building (Karras Algorithm)
// Tried the Apetrei algorithm in bvh2_test, but couldn't get better performance. TODO Need to try it again in here.
__device__ int clz_3d(unsigned int x) { return x == 0 ? 32 : __clz(x); }
__device__ int longest_common_prefix(const unsigned int* codes, int n, int i, int j) {
    if ( j < 0 || j >= n ) return -1;
    unsigned int ki = codes[i];
    unsigned int kj = codes[j];
    if ( ki == kj ) return 32 + clz_3d(i ^ j);
    return clz_3d(ki ^ kj);
}
__device__ void find_split(const unsigned int* codes, int n, int i, int& left, int& right, int& split) {
    int lcp_prev = longest_common_prefix(codes, n, i, i - 1);
    int lcp_next = longest_common_prefix(codes, n, i, i + 1);
    int d = (lcp_next > lcp_prev) ? 1 : -1;
    // int lcp_min = longest_common_prefix(codes, n, i, i - d);
    int lcp_min = min(lcp_prev, lcp_next);
    int l_max = 2;
    while ( longest_common_prefix(codes, n, i, i + l_max * d) > lcp_min ) l_max *= 2;
    int l = 0;
    for ( int t = l_max / 2; t >= 1; t /= 2 ) {
        if ( longest_common_prefix(codes, n, i, i + (l + t) * d) > lcp_min ) l += t;
    }
    int j = i + l * d;
    left = min(i, j);
    right = max(i, j);
    int lcp_node = longest_common_prefix(codes, n, left, right);
    int s = 0;
    int t = right - left;
    do {
        t = (t + 1) / 2;
        if ( longest_common_prefix(codes, n, left, left + s + t) > lcp_node ) s += t;
    } while ( t > 1 );
    split = left + s;
}
__global__ void build_tree_kernel(const unsigned int* codes, const unsigned int* sorted_indices,
    unsigned int n, int2* nodes) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Leaves
    if ( i < n ) nodes[i] = make_int2(sorted_indices[i] + 1, 0);
    // Internal Nodes
    if ( i < n - 1 ) {
        unsigned int internal_idx = n + i;
        int left, right, split;
        find_split(codes, n, i, left, right, split);
        // Left child: if range [left, split] is single element, it's a leaf;
        // else internal
        unsigned int lc = (left == split) ? left : (n + split);
        // Right child: if range [split+1, right] is single element, it's a
        // leaf; else internal
        unsigned int rc = (split + 1 == right) ? right : (n + split + 1);
        nodes[internal_idx] = make_int2(lc + 1, rc + 1);
    }
}
// Depth & Parent Logic
__global__ void init_parent_kernel(unsigned int* parent, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) parent[i] = UINT_MAX;
}
__global__ void set_parent_kernel(const int2* nodes, unsigned int* parent, unsigned int num_nodes) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_nodes ) return;
    int2 node = nodes[i];
    if ( node.y != 0 ) {
        parent[node.x - 1] = i;
        parent[node.y - 1] = i;
    }
}
__global__ void compute_depths_kernel(unsigned int num_nodes, const unsigned int* parent, unsigned int* depth) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_nodes ) return;
    unsigned int d = 0;
    unsigned int curr = i;
    for ( int k = 0; k < 64; ++k ) {
        unsigned int p = parent[curr];
        if ( p == curr ) break;
        d++;
        curr = p;
    }
    depth[i] = d;
}
// AABB Calculations (3D)
__global__ void compute_leaf_aabbs_face_kernel(const float3* vertices, const int3* faces,
    unsigned int n, const int2* nodes, AABB3D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    unsigned int prim_idx = nodes[i].x - 1;
    int3 f = faces[prim_idx];
    float3 v0 = vertices[f.x];
    float3 v1 = vertices[f.y];
    float3 v2 = vertices[f.z];
    aabbs[i].min.x = fminf(v0.x, fminf(v1.x, v2.x));
    aabbs[i].min.y = fminf(v0.y, fminf(v1.y, v2.y));
    aabbs[i].min.z = fminf(v0.z, fminf(v1.z, v2.z));
    aabbs[i].max.x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    aabbs[i].max.y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
    aabbs[i].max.z = fmaxf(v0.z, fmaxf(v1.z, v2.z));
}
__global__ void compute_leaf_aabbs_edge_kernel(const float3* vertices, const int2* edges,
    unsigned int n, const int2* nodes, AABB3D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    unsigned int prim_idx = nodes[i].x - 1;
    int2 e = edges[prim_idx];
    float3 v0 = vertices[e.x];
    float3 v1 = vertices[e.y];
    aabbs[i].min.x = fminf(v0.x, v1.x);
    aabbs[i].min.y = fminf(v0.y, v1.y);
    aabbs[i].min.z = fminf(v0.z, v1.z);
    aabbs[i].max.x = fmaxf(v0.x, v1.x);
    aabbs[i].max.y = fmaxf(v0.y, v1.y);
    aabbs[i].max.z = fmaxf(v0.z, v1.z);
}
__global__ void compute_leaf_aabbs_point_kernel(const float3* vertices, unsigned int n,
    const int2* nodes, AABB3D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    unsigned int prim_idx = nodes[i].x - 1;
    float3 v = vertices[prim_idx];
    const float eps = 1e-4f;
    aabbs[i].min = make_float3(v.x - eps, v.y - eps, v.z - eps);
    aabbs[i].max = make_float3(v.x + eps, v.y + eps, v.z + eps);
}
__global__ void merge_aabbs_kernel(const unsigned int* level_data, unsigned int count,
    const int2* nodes, AABB3D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= count ) return;
    unsigned int node_idx = level_data[i];
    int2 children = nodes[node_idx];
    if ( children.y == 0 ) return;
    unsigned int lc = children.x - 1;
    unsigned int rc = children.y - 1;
    AABB3D a = aabbs[lc];
    AABB3D b = aabbs[rc];
    aabbs[node_idx].min.x = fminf(a.min.x, b.min.x);
    aabbs[node_idx].min.y = fminf(a.min.y, b.min.y);
    aabbs[node_idx].min.z = fminf(a.min.z, b.min.z);
    aabbs[node_idx].max.x = fmaxf(a.max.x, b.max.x);
    aabbs[node_idx].max.y = fmaxf(a.max.y, b.max.y);
    aabbs[node_idx].max.z = fmaxf(a.max.z, b.max.z);
}
//==============================================================================
// Query Kernels
//==============================================================================


// Collision Query Kernel
// __global__ void query_collisions_kernel(const int2* nodes_a, const AABB3D* aabbs_a,
//     unsigned int root_a, unsigned int num_leaves_a,
//     const int2* nodes_b, const AABB3D* aabbs_b,
//     unsigned int root_b,
//     CollisionPair* pairs, unsigned int max_pairs, unsigned int* out_count) {
//     unsigned int leaf_a_idx = blockIdx.x * blockDim.x + threadIdx.x;
//     if ( leaf_a_idx >= num_leaves_a ) return;
//     int2 node_a = nodes_a[leaf_a_idx];
//     if ( node_a.y != 0 ) return;
//     unsigned int prim_a = node_a.x - 1;
//     AABB3D box_a = aabbs_a[leaf_a_idx];
//     unsigned int stack[32];
//     int sp = 0;
//     stack[sp++] = root_b;
//     while ( sp > 0 ) {
//         unsigned int node_b_idx = stack[--sp];
//         AABB3D box_b = aabbs_b[node_b_idx];
//         if ( !aabb_overlap_3d(box_a, box_b) ) continue;
//         int2 node_b = nodes_b[node_b_idx];
//         if ( node_b.y == 0 ) {
//             unsigned int prim_b = node_b.x - 1;
//             unsigned int idx = atomicAdd(out_count, 1);
//             if ( idx < max_pairs ) {
//                 pairs[idx] = { prim_a, prim_b };
//             }
//         }
//         else {
//             stack[sp++] = node_b.y - 1;
//             stack[sp++] = node_b.x - 1;
//         }
//     }
// }
//==============================================================================
// 3D Geometry Helpers
//==============================================================================


__global__ void query_nearest_face_kernel(
    const float3* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB3D* aabbs, unsigned int root_idx,
    const float3* vertices, const int3* faces,
    int* out_nearest_idx) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;

    float3 qp = query_pts[i];
    float best_dist = FLT_MAX;
    unsigned int best_prim = UINT_MAX;

    unsigned int stack[64];
    int sp = 0;
    stack[sp++] = root_idx;

    while ( sp > 0 ) {
        if ( sp > 60 ) {
            printf("query_nearest_face_kernel: %d\n", sp);
        }
        unsigned int node_idx = stack[--sp];

        if ( dist_sq_point_aabb_3d(qp, aabbs[node_idx]) >= best_dist ) continue;

        int2 node = nodes[node_idx];

        if ( node.y == 0 ) { // 叶子节点
            unsigned int prim_idx = node.x - 1;
            int3 f = faces[prim_idx];

            float3 v0 = vertices[f.x];
            float3 v1 = vertices[f.y];
            float3 v2 = vertices[f.z];

            float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);

            if ( dist_sq < best_dist ) {
                best_dist = dist_sq;
                best_prim = prim_idx;

                if ( best_dist < 1e-8f ) break;
            }
        }
        else {
            unsigned int left_child = node.x - 1;
            unsigned int right_child = node.y - 1;

            float dist_left = dist_sq_point_aabb_3d(qp, aabbs[left_child]);
            float dist_right = dist_sq_point_aabb_3d(qp, aabbs[right_child]);

            if ( dist_left > dist_right ) {
                if ( dist_left < best_dist ) stack[sp++] = left_child;
                if ( dist_right < best_dist ) stack[sp++] = right_child;
            }
            else {
                if ( dist_right < best_dist ) stack[sp++] = right_child;
                if ( dist_left < best_dist ) stack[sp++] = left_child;
            }
        }
    }

    out_nearest_idx[i] = best_prim;
}

//==============================================================================
// Host Interface Implementation
//==============================================================================
void initialize(unsigned int max_primitives) {
    auto& s = storage::instance();
    if ( max_primitives <= s.max_primitives ) return;
    s.max_primitives = max_primitives;
    s.morton_codes.resize(max_primitives * 2); // *2 for sort
    s.sorted_indices.resize(max_primitives * 2);
    s.centroids.resize(max_primitives);
    unsigned int max_nodes = (max_primitives > 0) ? 2 * max_primitives - 1 : 0;
    // s.parent.resize(max_nodes);
    // s.depth.resize(max_nodes);
    s.node_indices_scratch.resize(max_nodes);
    s.child_count.resize(max_nodes);
    s.scene_bounds.resize(6);
}
void cleanup() {
    auto& s = storage::instance();
    s.max_primitives = 0;
    s.morton_codes.clear();
    s.sorted_indices.clear();
    s.centroids.clear();
    // s.parent.clear();
    // s.depth.clear();
    s.node_indices_scratch.clear();
    s.child_count.clear();
    s.scene_bounds.clear();
}
void compute_bounds(const float3* points, unsigned int n, float3& min_res, float3& max_res) {
    float3 init_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 init_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    min_res = thrust::reduce(thrust::device, points,
        points + n, init_min, float3_min());
    max_res = thrust::reduce(thrust::device, points,
        points + n, init_max, float3_max());
}
void compute_and_sort_by_morton_codes(float3* points, unsigned int n, unsigned int* sorted_indices) {
    auto& s = storage::instance();
    float3 min_res, max_res;
    compute_bounds(points, n, min_res, max_res);
    float h_bounds[6] = { min_res.x, min_res.y, min_res.z, max_res.x, max_res.y, max_res.z };
    thrust::copy(h_bounds, h_bounds + 6, s.scene_bounds.begin());

    unsigned int* d_codes = thrust::raw_pointer_cast(s.morton_codes.data());
    {
        const float* d_bounds = thrust::raw_pointer_cast(s.scene_bounds.data());
        int blocks = (n + 255) / 256;
        compute_morton_codes_kernel<<<blocks, 256>>>(points, n, d_bounds, d_codes, sorted_indices);
    }
    radix_sort_pairs(nullptr, d_codes, sorted_indices, n);
}
unsigned int* get_sorted_indices() {
    return thrust::raw_pointer_cast(storage::instance().sorted_indices.data());
}
void build_bvh_internal(BVH3D& bvh, unsigned int n) {
    if ( n == 0 ) return;
    bvh.num_primitives = n;
    unsigned int num_nodes = 2 * n - 1;
    auto& s = storage::instance();
    // 1. Compute Scene Bounds using Thrust Reduce on float3 centroids
    // float3 init_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    // float3 init_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float3* centroids_ptr = thrust::raw_pointer_cast(s.centroids.data());
    // size_t centroids_n = s.centroids.size();
    // float3 min_res = thrust::reduce(thrust::device, centroids_ptr,
    //     centroids_ptr + centroids_n, init_min, float3_min());
    // float3 max_res = thrust::reduce(thrust::device, centroids_ptr,
    //     centroids_ptr + centroids_n, init_max, float3_max());
    // float3 min_res, max_res;
    // compute_bounds(thrust::raw_pointer_cast(s.centroids.data()), s.centroids.size(), min_res, max_res);
    // float h_bounds[6] = { min_res.x, min_res.y, min_res.z, max_res.x, max_res.y, max_res.z };
    // thrust::copy(h_bounds, h_bounds + 6, s.scene_bounds.begin());
    // // 2. Compute Morton Codes
    // unsigned int* d_codes = thrust::raw_pointer_cast(s.morton_codes.data());
    // unsigned int* d_indices = thrust::raw_pointer_cast(s.sorted_indices.data());
    // {
    //     const float* d_bounds = thrust::raw_pointer_cast(s.scene_bounds.data());
    //     const float3* d_centroids = thrust::raw_pointer_cast(s.centroids.data());
    //     int blocks = (n + 255) / 256;
    //     compute_morton_codes_kernel<<<blocks, 256>>>(d_centroids, n, d_bounds, d_codes, d_indices);
    // }
    // // 3. Sort by Morton Code
    // // thrust::sort_by_key(s.morton_codes.begin(), s.morton_codes.begin() + n,
    // //     s.sorted_indices.begin());
    // radix_sort_pairs(nullptr, d_codes, d_indices, n);
    compute_and_sort_by_morton_codes(centroids_ptr, n, thrust::raw_pointer_cast(s.sorted_indices.data()));
    bvh.nodes.resize(num_nodes);
    bvh.parent.resize(num_nodes);
    // 4. Build Tree Structure
    {
        const unsigned int* d_codes = thrust::raw_pointer_cast(s.morton_codes.data());
        const unsigned int* d_indices = thrust::raw_pointer_cast(s.sorted_indices.data());
        int2* d_nodes = thrust::raw_pointer_cast(bvh.nodes.data());
        int blocks = (n + 255) / 256;
        build_tree_kernel<<<blocks, 256>>>(d_codes, d_indices, n, d_nodes);
        unsigned int* d_parent = thrust::raw_pointer_cast(bvh.parent.data());
        init_parent_kernel<<<(num_nodes + 255) / 256, 256>>>(d_parent, num_nodes);
        set_parent_kernel<<<(num_nodes + 255) / 256, 256>>>(d_nodes, d_parent, num_nodes);

        // thrust::device_vector<unsigned int>::iterator root_iter;
        // root_iter = thrust::find(s.parent.begin() + n, s.parent.begin() + num_nodes, UINT_MAX);
        // bvh.root_idx = root_iter - s.parent.begin();
        // unsigned int h_root = bvh.root_idx;
        // thrust::fill(s.parent.begin() + h_root, s.parent.begin() + h_root + 1, h_root);
        // unsigned int* d_depth = thrust::raw_pointer_cast(s.depth.data());
        // compute_depths_kernel<<<(num_nodes + 255) / 256, 256>>>(num_nodes, d_parent, d_depth);
        // CUDA_CHECK(cudaDeviceSynchronize());
        // Find root: the internal node with parent == UINT_MAX
        if ( n == 1 ) {
            // Single leaf is the root
            bvh.root_idx = 0;
            thrust::fill(bvh.parent.begin(), bvh.parent.begin() + 1, 0u);
        }
        else {
            auto root_iter = thrust::find(bvh.parent.begin() + n, bvh.parent.begin() + num_nodes, UINT_MAX);
            bvh.root_idx = root_iter - bvh.parent.begin();
            unsigned int h_root = bvh.root_idx;
            // Make root's parent point to itself (termination condition for refit)
            thrust::fill(bvh.parent.begin() + h_root, bvh.parent.begin() + h_root + 1, h_root);
        }
    }
    // // 5. Build Level Offsets via Sorting
    // thrust::sequence(s.node_indices_scratch.begin(), s.node_indices_scratch.begin() + num_nodes);
    // thrust::sort_by_key(s.depth.begin(), s.depth.begin() + num_nodes,
    //     s.node_indices_scratch.begin());
    // thrust::device_vector<unsigned int> unique_depths(num_nodes);
    // thrust::device_vector<unsigned int> counts(num_nodes);
    // auto new_end = thrust::reduce_by_key(
    //     s.depth.begin(), s.depth.begin() + num_nodes,
    //     thrust::constant_iterator<unsigned int>(1),
    //     unique_depths.begin(),
    //     counts.begin()
    //     );
    // int num_levels = new_end.first - unique_depths.begin();
    // bvh.level_offsets.resize(num_levels + 1);
    // bvh.level_offsets[0] = 0;
    // thrust::inclusive_scan(counts.begin(), counts.begin() + num_levels, bvh.level_offsets.begin() + 1);
}

__global__ void refit_face_bvh_kernel(
    const float3* vertices, const float3* additional_offset,
    const int3* faces,
    unsigned int n,
    const int2* __restrict__ nodes,
    const unsigned int* __restrict__ parent,
    unsigned int* __restrict__ child_count,
    AABB3D* __restrict__ aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    // Step 1: Compute leaf AABB from triangle primitive
    unsigned int prim_idx = nodes[i].x - 1;
    int3 f = faces[prim_idx];
    float3 v0 = vertices[f.x], v1 = vertices[f.y], v2 = vertices[f.z];
    aabbs[i].min = fmin3(v0, fmin3(v1, v2));
    aabbs[i].max = fmax3(v0, fmax3(v1, v2));;

    if ( additional_offset ) {
        v0 += additional_offset[f.x];
        v1 += additional_offset[f.y];
        v2 += additional_offset[f.z];
        aabbs[i].min = fmin3(aabbs[i].min, fmin3(v0, fmin3(v1, v2)));
        aabbs[i].max = fmax3(aabbs[i].max, fmax3(v0, fmax3(v1, v2)));
    }

    // Step 2: Bottom-up refit — walk toward root
    unsigned int index = i;
    for ( ;; ) {
        unsigned int p = parent[index];
        // Reached root (parent points to self) or no parent
        if ( p == index || p == UINT_MAX ) return;

        // Ensure our AABB write is visible before signaling completion
        __threadfence();

        unsigned int finished = atomicAdd(&child_count[p], 1);

        if ( finished == 1 ) {
            // Both children complete — merge their AABBs into parent
            int2 node = nodes[p];
            unsigned int lc = node.x - 1;
            unsigned int rc = node.y - 1;
            AABB3D a = aabbs[lc];
            AABB3D b = aabbs[rc];
            aabbs[p].min = make_float3(fminf(a.min.x, b.min.x),
                fminf(a.min.y, b.min.y),
                fminf(a.min.z, b.min.z));
            aabbs[p].max = make_float3(fmaxf(a.max.x, b.max.x),
                fmaxf(a.max.y, b.max.y),
                fmaxf(a.max.z, b.max.z));
            // Continue up the tree
            index = p;
        }
        else {
            // First child done — sibling will merge, terminate this thread
            break;
        }
    }
}

__global__ void refit_edge_bvh_kernel(
    const float3* vertices,
    const float3* additional_offset,
    const int2* edges,
    unsigned int n,
    const int2* __restrict__ nodes,
    const unsigned int* __restrict__ parent,
    unsigned int* __restrict__ child_count,
    AABB3D* __restrict__ aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    // Compute leaf AABB from edge primitive
    unsigned int prim_idx = nodes[i].x - 1;
    int2 e = edges[prim_idx];
    float3 v0 = vertices[e.x], v1 = vertices[e.y];
    aabbs[i].min = fmin3(v0, v1);
    aabbs[i].max = fmax3(v0, v1);
    if ( additional_offset ) {
        v0 += additional_offset[e.x];
        v1 += additional_offset[e.y];
        aabbs[i].min = fmin3(v0, aabbs[i].min);
        aabbs[i].max = fmax3(v0, aabbs[i].max);
    }

    // Bottom-up refit
    unsigned int index = i;
    for ( ;; ) {
        unsigned int p = parent[index];
        if ( p == index || p == UINT_MAX ) return;
        __threadfence();
        unsigned int finished = atomicAdd(&child_count[p], 1);
        if ( finished == 1 ) {
            int2 node = nodes[p];
            unsigned int lc = node.x - 1;
            unsigned int rc = node.y - 1;
            AABB3D a = aabbs[lc];
            AABB3D b = aabbs[rc];
            aabbs[p].min = make_float3(fminf(a.min.x, b.min.x),
                fminf(a.min.y, b.min.y),
                fminf(a.min.z, b.min.z));
            aabbs[p].max = make_float3(fmaxf(a.max.x, b.max.x),
                fmaxf(a.max.y, b.max.y),
                fmaxf(a.max.z, b.max.z));
            index = p;
        }
        else {
            break;
        }
    }
}
// Public Build Functions
void build_face_bvh(const thrust::device_vector<float3>& vertices,
    const thrust::device_vector<int3>& faces, BVH3D& bvh, const float3* additional_offset) {
    unsigned int n = faces.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    const float3* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int3* d_faces = thrust::raw_pointer_cast(faces.data());
    float3* d_centroids = thrust::raw_pointer_cast(s.centroids.data());
    int blocks = (n + 255) / 256;
    compute_face_centroids_kernel<<<blocks, 256>>>(d_verts, d_faces, n, d_centroids);
    build_bvh_internal(bvh, n);
    bvh.aabbs.resize(bvh.nodes.size());
    // compute_leaf_aabbs_face_kernel<<<blocks, 256>>>(d_verts, d_faces, n,
    //     thrust::raw_pointer_cast(bvh.nodes.data()),
    //     thrust::raw_pointer_cast(bvh.aabbs.data()));
    // thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
    // unsigned int num_levels = h_offsets.size() - 1;
    // const int2* d_nodes = thrust::raw_pointer_cast(bvh.nodes.data());
    // AABB3D* d_aabbs = thrust::raw_pointer_cast(bvh.aabbs.data());
    // const unsigned int* d_level_data = thrust::raw_pointer_cast(s.node_indices_scratch.data());
    // for ( int l = num_levels - 2; l >= 0; --l ) {
    //     unsigned int start = h_offsets[l];
    //     unsigned int end = h_offsets[l + 1];
    //     unsigned int count = end - start;
    //     int blks = (count + 255) / 256;
    //     merge_aabbs_kernel<<<blks, 256>>>(d_level_data + start, count, d_nodes, d_aabbs);
    // }
    unsigned int num_nodes = 2 * n - 1;
    // Zero child_count for refit synchronization
    thrust::fill(s.child_count.begin(), s.child_count.begin() + num_nodes, 0u);

    // Single-kernel bottom-up refit (replaces leaf AABB + level-by-level merge)
    refit_face_bvh_kernel<<<blocks, 256>>>(
        d_verts, additional_offset, d_faces, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.parent.data()),
        thrust::raw_pointer_cast(s.child_count.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
}
void refit_face_bvh(const float3* vertices, const thrust::device_vector<int3>& faces, BVH3D& bvh,
    const float3* additional_offset) {
    unsigned int n = faces.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    int blocks = (n + 255) / 256;

    unsigned int num_nodes = 2 * n - 1;
    // Zero child_count for refit synchronization
    thrust::fill(s.child_count.begin(), s.child_count.begin() + num_nodes, 0u);

    // Single-kernel bottom-up refit (replaces leaf AABB + level-by-level merge)
    refit_face_bvh_kernel<<<blocks, 256>>>(
        vertices, additional_offset,
        thrust::raw_pointer_cast(faces.data()), n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.parent.data()),
        thrust::raw_pointer_cast(s.child_count.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
}
void build_edge_bvh(const thrust::device_vector<float3>& vertices, const thrust::device_vector<int2>& edges, BVH3D& bvh,
    const float3* additional_offset) {
    unsigned int n = edges.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    const float3* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int2* d_edges = thrust::raw_pointer_cast(edges.data());
    float3* d_centroids = thrust::raw_pointer_cast(s.centroids.data());
    int blocks = (n + 255) / 256;
    compute_edge_centroids_kernel<<<blocks, 256>>>(d_verts, d_edges, n, d_centroids);
    build_bvh_internal(bvh, n);
    bvh.aabbs.resize(bvh.nodes.size());
    // compute_leaf_aabbs_edge_kernel<<<blocks, 256>>>(d_verts, d_edges, n,
    //     thrust::raw_pointer_cast(bvh.nodes.data()),
    //     thrust::raw_pointer_cast(bvh.aabbs.data()));
    // thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
    // unsigned int num_levels = h_offsets.size() - 1;
    // for ( int l = num_levels - 2; l >= 0; --l ) {
    //     unsigned int start = h_offsets[l];
    //     unsigned int count = h_offsets[l + 1] - start;
    //     merge_aabbs_kernel<<<(count + 255) / 256, 256>>>(
    //         thrust::raw_pointer_cast(s.node_indices_scratch.data()) + start,
    //         count, thrust::raw_pointer_cast(bvh.nodes.data()),
    //         thrust::raw_pointer_cast(bvh.aabbs.data()));
    // }
    unsigned int num_nodes = 2 * n - 1;
    thrust::fill(s.child_count.begin(), s.child_count.begin() + num_nodes, 0u);

    refit_edge_bvh_kernel<<<blocks, 256>>>(
        d_verts, additional_offset,
        d_edges, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.parent.data()),
        thrust::raw_pointer_cast(s.child_count.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
}
void refit_edge_bvh(const float3* vertices, const thrust::device_vector<int2>& edges, BVH3D& bvh,
    const float3* additional_offset) {
    unsigned int n = edges.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    int blocks = (n + 255) / 256;
    unsigned int num_nodes = 2 * n - 1;
    thrust::fill(s.child_count.begin(), s.child_count.begin() + num_nodes, 0u);

    refit_edge_bvh_kernel<<<blocks, 256>>>(
        vertices, additional_offset,
        thrust::raw_pointer_cast(edges.data()), n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.parent.data()),
        thrust::raw_pointer_cast(s.child_count.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
}
// void build_point_bvh(const thrust::device_vector<float3>& vertices, BVH3D& bvh) {
//     unsigned int n = vertices.size();
//     if ( n == 0 ) return;
//     auto& s = storage::instance();
//     s.centroids = vertices;
//     build_bvh_internal(bvh, n);
//     bvh.aabbs.resize(bvh.nodes.size());
//     compute_leaf_aabbs_point_kernel<<<(n + 255) / 256, 256>>>(
//         thrust::raw_pointer_cast(vertices.data()), n,
//         thrust::raw_pointer_cast(bvh.nodes.data()),
//         thrust::raw_pointer_cast(bvh.aabbs.data()));
//     thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
//     unsigned int num_levels = h_offsets.size() - 1;
//     for ( int l = num_levels - 2; l >= 0; --l ) {
//         unsigned int start = h_offsets[l];
//         unsigned int count = h_offsets[l + 1] - start;
//         merge_aabbs_kernel<<<(count + 255) / 256, 256>>>(
//             thrust::raw_pointer_cast(s.node_indices_scratch.data()) + start,
//             count, thrust::raw_pointer_cast(bvh.nodes.data()),
//             thrust::raw_pointer_cast(bvh.aabbs.data()));
//     }
// }

} // namespace lbvh3d
