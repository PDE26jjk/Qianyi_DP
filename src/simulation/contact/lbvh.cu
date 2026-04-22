// File: lbvh.cu
// code: GLM-5
// Review: PDE26jjk


#include "common/cuda_utils.h"
#include "common/vec_math.h"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/functional.h>
#include "lbvh.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cassert>

namespace lbvh3d {
// Internal storage
namespace storage {
struct Storage {
    thrust::device_vector<unsigned int> morton_codes;
    thrust::device_vector<unsigned int> sorted_indices;
    // Combined storage for centroids (float3)
    thrust::device_vector<float3> centroids;
    thrust::device_vector<unsigned int> parent;
    thrust::device_vector<unsigned int> depth;
    thrust::device_vector<unsigned int> node_indices_scratch;
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
__device__ __host__ unsigned int morton_code_3d(unsigned int x, unsigned int y, unsigned int z) {
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
    centroids[i].x = (v0.x + v1.x + v2.x) / 3.0f;
    centroids[i].y = (v0.y + v1.y + v2.y) / 3.0f;
    centroids[i].z = (v0.z + v1.z + v2.z) / 3.0f;
}
__global__ void compute_edge_centroids_kernel(const float3* vertices, const int2* edges,
    unsigned int n, float3* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    int2 e = edges[i];
    float3 v0 = vertices[e.x];
    float3 v1 = vertices[e.y];
    centroids[i].x = (v0.x + v1.x) / 2.0f;
    centroids[i].y = (v0.y + v1.y) / 2.0f;
    centroids[i].z = (v0.z + v1.z) / 2.0f;
}
__global__ void compute_point_centroids_kernel(const float3* vertices, unsigned int n, float3* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    centroids[i] = vertices[i];
}
// Morton Code Generation (Reading from float3)
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
// Tree Building (Karras Algorithm) — identical structure to 2D
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
        unsigned int lc = (left == split) ? left : (n + split);
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
__device__ bool aabb_overlap_3d(const AABB3D& a, const AABB3D& b) {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
        (a.min.y <= b.max.y && a.max.y >= b.min.y) &&
        (a.min.z <= b.max.z && a.max.z >= b.min.z);
}
__device__ float dist_sq_point_aabb_3d(float3 p, const AABB3D& b) {
    float dx = fmaxf(b.min.x - p.x, 0.0f) + fmaxf(p.x - b.max.x, 0.0f);
    float dy = fmaxf(b.min.y - p.y, 0.0f) + fmaxf(p.y - b.max.y, 0.0f);
    float dz = fmaxf(b.min.z - p.z, 0.0f) + fmaxf(p.z - b.max.z, 0.0f);
    return dx * dx + dy * dy + dz * dz;
}

// Collision Query Kernel
__global__ void query_collisions_kernel(const int2* nodes_a, const AABB3D* aabbs_a,
    unsigned int root_a, unsigned int num_leaves_a,
    const int2* nodes_b, const AABB3D* aabbs_b,
    unsigned int root_b,
    CollisionPair* pairs, unsigned int max_pairs, unsigned int* out_count) {
    unsigned int leaf_a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( leaf_a_idx >= num_leaves_a ) return;
    int2 node_a = nodes_a[leaf_a_idx];
    if ( node_a.y != 0 ) return;
    unsigned int prim_a = node_a.x - 1;
    AABB3D box_a = aabbs_a[leaf_a_idx];
    unsigned int stack[32];
    int sp = 0;
    stack[sp++] = root_b;
    while ( sp > 0 ) {
        unsigned int node_b_idx = stack[--sp];
        AABB3D box_b = aabbs_b[node_b_idx];
        if ( !aabb_overlap_3d(box_a, box_b) ) continue;
        int2 node_b = nodes_b[node_b_idx];
        if ( node_b.y == 0 ) {
            unsigned int prim_b = node_b.x - 1;
            unsigned int idx = atomicAdd(out_count, 1);
            if ( idx < max_pairs ) {
                pairs[idx] = { prim_a, prim_b };
            }
        }
        else {
            stack[sp++] = node_b.y - 1;
            stack[sp++] = node_b.x - 1;
        }
    }
}
//==============================================================================
// 3D Geometry Helpers
//==============================================================================

// Point-triangle interaction in 3D (Ericson's algorithm)
__device__ float dist_sq_point_triangle_3d(float3 p, float3 a, float3 b, float3 c) {
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;
 
    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return dot(ap, ap); // 顶点 A 最近
 
    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return dot(bp, bp); // 顶点 B 最近
 
    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        float3 closest = a + v * ab;
        float3 diff = p - closest;
        return dot(diff, diff); // 边 AB 最近
    }
 
    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return dot(cp, cp); // 顶点 C 最近
 
    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        float3 closest = a + w * ac;
        float3 diff = p - closest;
        return dot(diff, diff); // 边 AC 最近
    }
 
    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float3 bc = c - b;
        float3 closest = b + w * bc;
        float3 diff = p - closest;
        return dot(diff, diff); // 边 BC 最近
    }
 
    // 点在三角形内部或投影在三角形面上
    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float3 closest = a + v * ab + w * ac;
    float3 diff = p - closest;
    return dot(diff, diff);
}
 
__global__ void query_nearest_face_kernel(
    const float3* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB3D* aabbs, unsigned int root_idx,
    const float3* vertices, const int3* faces,
    int* out_nearest_idx) 
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_queries) return;
 
    float3 qp = query_pts[i];
    float best_dist = FLT_MAX;
    unsigned int best_prim = UINT_MAX;
 
    unsigned int stack[64]; 
    int sp = 0;
    stack[sp++] = root_idx;
 
    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
 
        if (dist_sq_point_aabb_3d(qp, aabbs[node_idx]) >= best_dist) continue;
 
        int2 node = nodes[node_idx];
 
        if (node.y == 0) { // 叶子节点
            unsigned int prim_idx = node.x - 1;
            int3 f = faces[prim_idx];
 
            float3 v0 = vertices[f.x];
            float3 v1 = vertices[f.y];
            float3 v2 = vertices[f.z];
 
            float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);
 
            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_prim = prim_idx;
 
                if (best_dist < 1e-8f) break; 
            }
        }
        else {
            unsigned int left_child = node.x - 1;
            unsigned int right_child = node.y - 1;
 
            float dist_left = dist_sq_point_aabb_3d(qp, aabbs[left_child]);
            float dist_right = dist_sq_point_aabb_3d(qp, aabbs[right_child]);
 
            if (dist_left > dist_right) {
                if (dist_left < best_dist) stack[sp++] = left_child;
                if (dist_right < best_dist) stack[sp++] = right_child;
            } else {
                if (dist_right < best_dist) stack[sp++] = right_child;
                if (dist_left < best_dist) stack[sp++] = left_child;
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
    s.morton_codes.resize(max_primitives);
    s.sorted_indices.resize(max_primitives);
    s.centroids.resize(max_primitives);
    unsigned int max_nodes = (max_primitives > 0) ? 2 * max_primitives - 1 : 0;
    s.parent.resize(max_nodes);
    s.depth.resize(max_nodes);
    s.node_indices_scratch.resize(max_nodes);
    s.scene_bounds.resize(6);
}
void build_bvh_internal(BVH3D& bvh, unsigned int n) {
    if ( n == 0 ) return;
    bvh.num_primitives = n;
    unsigned int num_nodes = 2 * n - 1;
    auto& s = storage::instance();
    // 1. Compute Scene Bounds using Thrust Reduce on float3 centroids
    float3 init_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 init_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    float3* centroids_ptr = thrust::raw_pointer_cast(s.centroids.data());
    size_t centroids_n = s.centroids.size();
    float3 min_res = thrust::reduce(thrust::device, centroids_ptr,
        centroids_ptr + centroids_n, init_min, float3_min());
    float3 max_res = thrust::reduce(thrust::device, centroids_ptr,
        centroids_ptr + centroids_n, init_max, float3_max());
    float h_bounds[6] = { min_res.x, min_res.y, min_res.z, max_res.x, max_res.y, max_res.z };
    thrust::copy(h_bounds, h_bounds + 6, s.scene_bounds.begin());
    // 2. Compute Morton Codes
    bvh.nodes.resize(num_nodes);
    {
        const float* d_bounds = thrust::raw_pointer_cast(s.scene_bounds.data());
        unsigned int* d_codes = thrust::raw_pointer_cast(s.morton_codes.data());
        unsigned int* d_indices = thrust::raw_pointer_cast(s.sorted_indices.data());
        const float3* d_centroids = thrust::raw_pointer_cast(s.centroids.data());
        int blocks = (n + 255) / 256;
        compute_morton_codes_kernel<<<blocks, 256>>>(d_centroids, n, d_bounds, d_codes, d_indices);
    }
    // 3. Sort by Morton Code
    thrust::sort_by_key(s.morton_codes.begin(), s.morton_codes.begin() + n,
        s.sorted_indices.begin());
    // 4. Build Tree Structure
    {
        const unsigned int* d_codes = thrust::raw_pointer_cast(s.morton_codes.data());
        const unsigned int* d_indices = thrust::raw_pointer_cast(s.sorted_indices.data());
        int2* d_nodes = thrust::raw_pointer_cast(bvh.nodes.data());
        int blocks = (n + 255) / 256;
        build_tree_kernel<<<blocks, 256>>>(d_codes, d_indices, n, d_nodes);
        unsigned int* d_parent = thrust::raw_pointer_cast(s.parent.data());
        init_parent_kernel<<<(num_nodes + 255) / 256, 256>>>(d_parent, num_nodes);
        set_parent_kernel<<<(num_nodes + 255) / 256, 256>>>(d_nodes, d_parent, num_nodes);
        thrust::device_vector<unsigned int>::iterator root_iter;
        root_iter = thrust::find(s.parent.begin() + n, s.parent.begin() + num_nodes, UINT_MAX);
        bvh.root_idx = root_iter - s.parent.begin();
        unsigned int h_root = bvh.root_idx;
        thrust::fill(s.parent.begin() + h_root, s.parent.begin() + h_root + 1, h_root);
        unsigned int* d_depth = thrust::raw_pointer_cast(s.depth.data());
        compute_depths_kernel<<<(num_nodes + 255) / 256, 256>>>(num_nodes, d_parent, d_depth);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    // 5. Build Level Offsets via Sorting
    thrust::sequence(s.node_indices_scratch.begin(), s.node_indices_scratch.begin() + num_nodes);
    thrust::sort_by_key(s.depth.begin(), s.depth.begin() + num_nodes,
        s.node_indices_scratch.begin());
    thrust::device_vector<unsigned int> unique_depths(num_nodes);
    thrust::device_vector<unsigned int> counts(num_nodes);
    auto new_end = thrust::reduce_by_key(
        s.depth.begin(), s.depth.begin() + num_nodes,
        thrust::constant_iterator<unsigned int>(1),
        unique_depths.begin(),
        counts.begin()
        );
    int num_levels = new_end.first - unique_depths.begin();
    bvh.level_offsets.resize(num_levels + 1);
    bvh.level_offsets[0] = 0;
    thrust::inclusive_scan(counts.begin(), counts.begin() + num_levels, bvh.level_offsets.begin() + 1);
}
// Public Build Functions
void build_face_bvh(const thrust::device_vector<float3>& vertices, const thrust::device_vector<int3>& faces, BVH3D& bvh) {
    unsigned int n = faces.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    s.centroids.resize(n);
    const float3* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int3* d_faces = thrust::raw_pointer_cast(faces.data());
    float3* d_centroids = thrust::raw_pointer_cast(s.centroids.data());
    int blocks = (n + 255) / 256;
    compute_face_centroids_kernel<<<blocks, 256>>>(d_verts, d_faces, n, d_centroids);
    build_bvh_internal(bvh, n);
    bvh.aabbs.resize(bvh.nodes.size());
    compute_leaf_aabbs_face_kernel<<<blocks, 256>>>(d_verts, d_faces, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
    thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
    unsigned int num_levels = h_offsets.size() - 1;
    const int2* d_nodes = thrust::raw_pointer_cast(bvh.nodes.data());
    AABB3D* d_aabbs = thrust::raw_pointer_cast(bvh.aabbs.data());
    const unsigned int* d_level_data = thrust::raw_pointer_cast(s.node_indices_scratch.data());
    for ( int l = num_levels - 2; l >= 0; --l ) {
        unsigned int start = h_offsets[l];
        unsigned int end = h_offsets[l + 1];
        unsigned int count = end - start;
        int blks = (count + 255) / 256;
        merge_aabbs_kernel<<<blks, 256>>>(d_level_data + start, count, d_nodes, d_aabbs);
    }
}
void build_edge_bvh(const thrust::device_vector<float3>& vertices, const thrust::device_vector<int2>& edges, BVH3D& bvh) {
    unsigned int n = edges.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    s.centroids.resize(n);
    const float3* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int2* d_edges = thrust::raw_pointer_cast(edges.data());
    float3* d_centroids = thrust::raw_pointer_cast(s.centroids.data());
    int blocks = (n + 255) / 256;
    compute_edge_centroids_kernel<<<blocks, 256>>>(d_verts, d_edges, n, d_centroids);
    build_bvh_internal(bvh, n);
    bvh.aabbs.resize(bvh.nodes.size());
    compute_leaf_aabbs_edge_kernel<<<blocks, 256>>>(d_verts, d_edges, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
    thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
    unsigned int num_levels = h_offsets.size() - 1;
    for ( int l = num_levels - 2; l >= 0; --l ) {
        unsigned int start = h_offsets[l];
        unsigned int count = h_offsets[l + 1] - start;
        merge_aabbs_kernel<<<(count + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(s.node_indices_scratch.data()) + start,
            count, thrust::raw_pointer_cast(bvh.nodes.data()),
            thrust::raw_pointer_cast(bvh.aabbs.data()));
    }
}
void build_point_bvh(const thrust::device_vector<float3>& vertices, BVH3D& bvh) {
    unsigned int n = vertices.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    s.centroids = vertices;
    build_bvh_internal(bvh, n);
    bvh.aabbs.resize(bvh.nodes.size());
    compute_leaf_aabbs_point_kernel<<<(n + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(vertices.data()), n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));
    thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
    unsigned int num_levels = h_offsets.size() - 1;
    for ( int l = num_levels - 2; l >= 0; --l ) {
        unsigned int start = h_offsets[l];
        unsigned int count = h_offsets[l + 1] - start;
        merge_aabbs_kernel<<<(count + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(s.node_indices_scratch.data()) + start,
            count, thrust::raw_pointer_cast(bvh.nodes.data()),
            thrust::raw_pointer_cast(bvh.aabbs.data()));
    }
}
} // namespace lbvh3d
