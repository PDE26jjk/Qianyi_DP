// File: lbvh_2d.cu
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
// #include <thrust/transform_reduce.h>

#include "lbvh_2d.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>
#include <cassert>

#include "sampler.h"

namespace lbvh2d {

// Internal storage
namespace storage { 
struct Storage {
    thrust::device_vector<unsigned int> morton_codes;
    thrust::device_vector<unsigned int> sorted_indices;
 
    // Combined storage for centroids (float2)
    thrust::device_vector<float2> centroids;
 
    thrust::device_vector<unsigned int> parent;
    thrust::device_vector<unsigned int> depth;
    thrust::device_vector<unsigned int> node_indices_scratch; // For sorting nodes by depth
 
    // Scene bounds: [min_x, min_y, max_x, max_y]
    // We can also use a float4 or AABB2D here, but let's keep raw array for clarity
    thrust::device_vector<float> scene_bounds;

};
static Storage* inst = nullptr; // will not delete to avoid cudaFree being called after the CUDA context is destroyed by the host.
inline Storage& instance() {
    if (inst == nullptr) {
        inst = new Storage();
    }
    return *inst;
}

}

//==============================================================================
// Helper Functors for Thrust
//==============================================================================

// Min functor for float2
struct float2_min {
    __host__ __device__ float2 operator()(const float2& a, const float2& b) const {
        return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
    }
};

// Max functor for float2
struct float2_max {
    __host__ __device__ float2 operator()(const float2& a, const float2& b) const {
        return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
    }
};

//==============================================================================
// Morton Code 2D
//==============================================================================

__device__ __host__ unsigned int expand_bits_2d(unsigned int v) {
    v &= 0x0000FFFF;
    v = (v | (v << 8)) & 0x00FF00FF;
    v = (v | (v << 4)) & 0x0F0F0F0F;
    v = (v | (v << 2)) & 0x33333333;
    v = (v | (v << 1)) & 0x55555555;
    return v;
}

__device__ __host__ unsigned int morton_code_2d(unsigned int x, unsigned int y) {
    return (expand_bits_2d(y) << 1) | expand_bits_2d(x);
}

//==============================================================================
// Kernels
//==============================================================================

// Centroid Calculations (Writing to float2)
__global__ void compute_face_centroids_kernel(const float2* vertices, const int3* faces,
    unsigned int n, float2* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    int3 f = faces[i];
    float2 v0 = vertices[f.x];
    float2 v1 = vertices[f.y];
    float2 v2 = vertices[f.z];

    centroids[i].x = (v0.x + v1.x + v2.x) / 3.0f;
    centroids[i].y = (v0.y + v1.y + v2.y) / 3.0f;
}

__global__ void compute_edge_centroids_kernel(const float2* vertices, const int2* edges,
    unsigned int n, float2* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    int2 e = edges[i];
    float2 v0 = vertices[e.x];
    float2 v1 = vertices[e.y];

    centroids[i].x = (v0.x + v1.x) / 2.0f;
    centroids[i].y = (v0.y + v1.y) / 2.0f;
}

__global__ void compute_point_centroids_kernel(const float2* vertices, unsigned int n, float2* centroids) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    centroids[i] = vertices[i];
}

// Morton Code Generation (Reading from float2)
__global__ void compute_morton_codes_kernel(const float2* centroids, unsigned int n,
    const float* bounds, unsigned int* codes, unsigned int* indices) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    float min_x = bounds[0], min_y = bounds[1];
    float max_x = bounds[2], max_y = bounds[3];

    float2 c = centroids[i];

    float sx = (max_x - min_x) > 1e-10f ? 65535.0f / (max_x - min_x) : 0.0f;
    float sy = (max_y - min_y) > 1e-10f ? 65535.0f / (max_y - min_y) : 0.0f;

    unsigned int ix = (unsigned int)fminf(65535.0f, fmaxf(0.0f, (c.x - min_x) * sx));
    unsigned int iy = (unsigned int)fminf(65535.0f, fmaxf(0.0f, (c.y - min_y) * sy));

    codes[i] = morton_code_2d(ix, iy);
    indices[i] = i;
}

// Tree Building (Karras Algorithm)
__device__ int clz_2d(unsigned int x) { return x == 0 ? 32 : __clz(x); }

__device__ int longest_common_prefix(const unsigned int* codes, int n, int i, int j) {
    if ( j < 0 || j >= n ) return -1;
    unsigned int ki = codes[i];
    unsigned int kj = codes[j];
    if ( ki == kj ) return 32 + clz_2d(i ^ j);
    return clz_2d(ki ^ kj);
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
    if ( node.y != 0 ) { // Internal node check
        parent[node.x - 1] = i;
        parent[node.y - 1] = i;
    }
}

__global__ void compute_depths_kernel(unsigned int num_nodes, const unsigned int* parent, unsigned int* depth) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_nodes ) return;

    unsigned int d = 0;
    unsigned int curr = i;
    // Walk up to root
    for ( int k = 0; k < 64; ++k ) {
        unsigned int p = parent[curr];
        if ( p == curr ) break; // Root reached
        d++;
        curr = p;
    }
    depth[i] = d;
}

// AABB Calculations
__global__ void compute_leaf_aabbs_face_kernel(const float2* vertices, const int3* faces,
    unsigned int n, const int2* nodes, AABB2D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    unsigned int prim_idx = nodes[i].x - 1;
    int3 f = faces[prim_idx];

    float2 v0 = vertices[f.x];
    float2 v1 = vertices[f.y];
    float2 v2 = vertices[f.z];

    aabbs[i].min.x = fminf(v0.x, fminf(v1.x, v2.x));
    aabbs[i].min.y = fminf(v0.y, fminf(v1.y, v2.y));
    aabbs[i].max.x = fmaxf(v0.x, fmaxf(v1.x, v2.x));
    aabbs[i].max.y = fmaxf(v0.y, fmaxf(v1.y, v2.y));
}

__global__ void compute_leaf_aabbs_edge_kernel(const float2* vertices, const int2* edges,
    unsigned int n, const int2* nodes, AABB2D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    unsigned int prim_idx = nodes[i].x - 1;
    int2 e = edges[prim_idx];

    float2 v0 = vertices[e.x];
    float2 v1 = vertices[e.y];

    aabbs[i].min.x = fminf(v0.x, v1.x);
    aabbs[i].min.y = fminf(v0.y, v1.y);
    aabbs[i].max.x = fmaxf(v0.x, v1.x);
    aabbs[i].max.y = fmaxf(v0.y, v1.y);
}

__global__ void compute_leaf_aabbs_point_kernel(const float2* vertices, unsigned int n,
    const int2* nodes, AABB2D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    unsigned int prim_idx = nodes[i].x - 1;
    float2 v = vertices[prim_idx];

    const float eps = 1e-4f;
    aabbs[i].min = make_float2(v.x - eps, v.y - eps);
    aabbs[i].max = make_float2(v.x + eps, v.y + eps);
}

__global__ void merge_aabbs_kernel(const unsigned int* level_data, unsigned int count,
    const int2* nodes, AABB2D* aabbs) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= count ) return;

    unsigned int node_idx = level_data[i];
    int2 children = nodes[node_idx];

    // Check if internal node (y != 0)
    if ( children.y == 0 ) return;

    unsigned int lc = children.x - 1;
    unsigned int rc = children.y - 1;

    AABB2D a = aabbs[lc];
    AABB2D b = aabbs[rc];

    aabbs[node_idx].min.x = fminf(a.min.x, b.min.x);
    aabbs[node_idx].min.y = fminf(a.min.y, b.min.y);
    aabbs[node_idx].max.x = fmaxf(a.max.x, b.max.x);
    aabbs[node_idx].max.y = fmaxf(a.max.y, b.max.y);
}

//==============================================================================
// Query Kernels
//==============================================================================

__device__ bool aabb_overlap_2d(const AABB2D& a, const AABB2D& b) {
    return (a.min.x <= b.max.x && a.max.x >= b.min.x) &&
        (a.min.y <= b.max.y && a.max.y >= b.min.y);
}

__device__ float dist_sq_point_aabb_2d(float2 p, const AABB2D& b) {
    float dx = fmaxf(b.min.x - p.x, 0.0f) + fmaxf(p.x - b.max.x, 0.0f);
    float dy = fmaxf(b.min.y - p.y, 0.0f) + fmaxf(p.y - b.max.y, 0.0f);
    return dx * dx + dy * dy;
}

// Nearest Query Kernel
// Assumes we are querying against 'primitive_data' which could be points, edges, etc.
// Generic distance functor logic inside
__global__ void query_nearest_kernel(const float2* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB2D* aabbs,
    unsigned int root_idx, NearestResult* results,
    const float2* primitive_data) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;

    float2 qp = query_pts[i];
    float best_dist = FLT_MAX;
    unsigned int best_prim = UINT_MAX;

    unsigned int stack[32];
    int sp = 0;
    stack[sp++] = root_idx;

    while ( sp > 0 ) {
        unsigned int node_idx = stack[--sp];

        // Pruning
        if ( dist_sq_point_aabb_2d(qp, aabbs[node_idx]) > best_dist ) continue;

        int2 node = nodes[node_idx];
        if ( node.y == 0 ) { // Leaf
            unsigned int prim_idx = node.x - 1;
            // Calculate distance to the primitive
            // Here we assume primitive_data represents points for simplicity.
            // For edges/triangles, pass specific distance kernels.
            float2 prim = primitive_data[prim_idx];
            float dx = qp.x - prim.x;
            float dy = qp.y - prim.y;
            float d = dx * dx + dy * dy;

            if ( d < best_dist ) {
                best_dist = d;
                best_prim = prim_idx;
            }
        }
        else {
            // Internal node: push children
            // Order by distance heuristic could be added here
            stack[sp++] = node.y - 1;
            stack[sp++] = node.x - 1;
        }
    }
    results[i] = { best_prim, best_dist };
}

// Collision Query Kernel
__global__ void query_collisions_kernel(const int2* nodes_a, const AABB2D* aabbs_a,
    unsigned int root_a, unsigned int num_leaves_a,
    const int2* nodes_b, const AABB2D* aabbs_b,
    unsigned int root_b,
    CollisionPair* pairs, unsigned int max_pairs, unsigned int* out_count) {
    unsigned int leaf_a_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( leaf_a_idx >= num_leaves_a ) return;

    // Check if it is a leaf (Indices 0..N-1 are leaves in LBVH)
    // Safety check: internal nodes have y != 0
    int2 node_a = nodes_a[leaf_a_idx];
    if ( node_a.y != 0 ) return;

    unsigned int prim_a = node_a.x - 1;
    AABB2D box_a = aabbs_a[leaf_a_idx];

    unsigned int stack[32];
    int sp = 0;
    stack[sp++] = root_b;

    while ( sp > 0 ) {
        unsigned int node_b_idx = stack[--sp];
        AABB2D box_b = aabbs_b[node_b_idx];

        if ( !aabb_overlap_2d(box_a, box_b) ) continue;

        int2 node_b = nodes_b[node_b_idx];
        if ( node_b.y == 0 ) { // Leaf in B
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
    s.scene_bounds.resize(4);
}

// Internal helper to build tree structure and levels
void build_bvh_internal(BVH2D& bvh, unsigned int n) {
    if ( n == 0 ) return;
    bvh.num_primitives = n;
    unsigned int num_nodes = 2 * n - 1;
    auto& s = storage::instance();
    // 1. Compute Scene Bounds using Thrust Reduce on float2 centroids
    // Init with first element or identity
    float2 init_min = make_float2(FLT_MAX, FLT_MAX);
    float2 init_max = make_float2(-FLT_MAX, -FLT_MAX);
    float2* centroids_ptr = thrust::raw_pointer_cast(s.centroids.data());
    size_t centroids_n = s.centroids.size();
    float2 min_res = thrust::reduce(thrust::device, centroids_ptr,
        centroids_ptr + centroids_n, init_min, float2_min());
    float2 max_res = thrust::reduce(thrust::device, centroids_ptr,
        centroids_ptr + centroids_n, init_max, float2_max());
    float h_bounds[4] = { min_res.x, min_res.y, max_res.x, max_res.y };
    thrust::copy(h_bounds, h_bounds + 4, s.scene_bounds.begin());

    // 2. Compute Morton Codes
    bvh.nodes.resize(num_nodes);
    {
        const float* d_bounds = thrust::raw_pointer_cast(s.scene_bounds.data());
        unsigned int* d_codes = thrust::raw_pointer_cast(s.morton_codes.data());
        unsigned int* d_indices = thrust::raw_pointer_cast(s.sorted_indices.data());
        const float2* d_centroids = thrust::raw_pointer_cast(s.centroids.data());

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
        // Setup parents
        unsigned int* d_parent = thrust::raw_pointer_cast(s.parent.data());
        init_parent_kernel<<<(num_nodes + 255) / 256, 256>>>(d_parent, num_nodes);
        set_parent_kernel<<<(num_nodes + 255) / 256, 256>>>(d_nodes, d_parent, num_nodes);
        // Find root (node with parent == UINT_MAX)
        thrust::device_vector<unsigned int>::iterator root_iter;
        root_iter = thrust::find(s.parent.begin() + n, s.parent.begin() + num_nodes, UINT_MAX);
        bvh.root_idx = root_iter - s.parent.begin();

        // Set root parent to itself
        unsigned int h_root = bvh.root_idx;
        thrust::fill(s.parent.begin() + h_root, s.parent.begin() + h_root + 1, h_root);

        // Compute Depths
        unsigned int* d_depth = thrust::raw_pointer_cast(s.depth.data());
        compute_depths_kernel<<<(num_nodes + 255) / 256, 256>>>(num_nodes, d_parent,
            d_depth);
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
void build_face_bvh(const thrust::device_vector<float2>& vertices, const thrust::device_vector<int3>& faces, BVH2D& bvh) {
    unsigned int n = faces.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    s.centroids.resize(n);

    const float2* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int3* d_faces = thrust::raw_pointer_cast(faces.data());
    float2* d_centroids = thrust::raw_pointer_cast(s.centroids.data());

    int blocks = (n + 255) / 256;
    compute_face_centroids_kernel<<<blocks, 256>>>(d_verts, d_faces, n, d_centroids);

    build_bvh_internal(bvh, n);

    // AABB Calculation
    bvh.aabbs.resize(bvh.nodes.size());
    compute_leaf_aabbs_face_kernel<<<blocks, 256>>>(d_verts, d_faces, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));

    // Propagate AABBs bottom-up
    thrust::host_vector<unsigned int> h_offsets = bvh.level_offsets;
    unsigned int num_levels = h_offsets.size() - 1;

    const int2* d_nodes = thrust::raw_pointer_cast(bvh.nodes.data());
    AABB2D* d_aabbs = thrust::raw_pointer_cast(bvh.aabbs.data());
    const unsigned int* d_level_data = thrust::raw_pointer_cast(s.node_indices_scratch.data());

    for ( int l = num_levels - 2; l >= 0; --l ) { // Skip leaf level (num_levels-1)
        unsigned int start = h_offsets[l];
        unsigned int end = h_offsets[l + 1];
        unsigned int count = end - start;

        int blks = (count + 255) / 256;
        merge_aabbs_kernel<<<blks, 256>>>(d_level_data + start, count, d_nodes, d_aabbs);
    }
}

void build_edge_bvh(const thrust::device_vector<float2>& vertices, const thrust::device_vector<int2>& edges, BVH2D& bvh) {
    unsigned int n = edges.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    s.centroids.resize(n);
    const float2* d_verts = thrust::raw_pointer_cast(vertices.data());
    const int2* d_edges = thrust::raw_pointer_cast(edges.data());
    float2* d_centroids = thrust::raw_pointer_cast(s.centroids.data());

    int blocks = (n + 255) / 256;
    compute_edge_centroids_kernel<<<blocks, 256>>>(d_verts, d_edges, n, d_centroids);

    build_bvh_internal(bvh, n);

    bvh.aabbs.resize(bvh.nodes.size());
    compute_leaf_aabbs_edge_kernel<<<blocks, 256>>>(d_verts, d_edges, n,
        thrust::raw_pointer_cast(bvh.nodes.data()),
        thrust::raw_pointer_cast(bvh.aabbs.data()));

    // Propagate (Same logic as face, can be refactored into a helper)
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

void build_point_bvh(const thrust::device_vector<float2>& vertices, BVH2D& bvh) {
    unsigned int n = vertices.size();
    if ( n == 0 ) return;
    auto& s = storage::instance();
    // For points, centroids are the vertices themselves
    s.centroids = vertices;
    // Note: Thrust assignment operator handles deep copy if sizes match or resize

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
__device__ float closest_point_on_segment(float2 p, float2 a, float2 b, float& t) {
    float2 ab = b - a;
    float2 ap = p - a;
    
    float len_sq = ab.x * ab.x + ab.y * ab.y;
    if (len_sq < 1e-12f) { // 退化情况
        t = 0.0f;
        return ap.x * ap.x + ap.y * ap.y;
    }
    
    t = (ap.x * ab.x + ap.y * ab.y) / len_sq;
    t = fmaxf(0.0f, fminf(1.0f, t)); // 限制在 [0, 1]
    
    float2 closest = a + t * ab;
    float2 diff = p - closest;
    return diff.x * diff.x + diff.y * diff.y;
}
 
// 核心几何逻辑：计算点与三角形的关系
// 返回距离平方，更新权重 和 is_inside
__device__ void point_triangle_interaction(
    float2 p, float2 a, float2 b, float2 c,
    float& out_dist_sq, float3& out_weights, int& out_is_inside) 
{
    // 1. 计算重心坐标
    float2 v0 = b - a;
    float2 v1 = c - a;
    float2 v2 = p - a;
 
    float d00 = v0.x * v0.x + v0.y * v0.y;
    float d01 = v0.x * v1.x + v0.y * v1.y;
    float d11 = v1.x * v1.x + v1.y * v1.y;
    float d20 = v2.x * v0.x + v2.y * v0.y;
    float d21 = v2.x * v1.x + v2.y * v1.y;
 
    float denom = d00 * d11 - d01 * d01;
    
    // 处理退化三角形
    float max_term = fmaxf(d00 * d11, d01 * d01);
    if (fabsf(denom) <= max_term * 1e-6f) {
        out_dist_sq = FLT_MAX;
        out_weights = make_float3(0, 0, 0);
        out_is_inside = 0;
        return;
    }
 
    float invDenom = 1.0f / denom;
    float v = (d11 * d20 - d01 * d21) * invDenom;
    float w = (d00 * d21 - d01 * d20) * invDenom;
    float u = 1.0f - v - w;
 
    out_weights = make_float3(u, v, w);
 
    // 2. 判断是否在内部
    // 使用稍微放宽的阈值处理数值误差
    if (u >= -1e-6f && v >= -1e-6f && w >= -1e-6f) {
        // 严格内部 (>=0) 或 边界/顶点 (接近0)
        // 如果三个值都严格大于0，则认为在内部
        if (u > 1e-6f && v > 1e-6f && w > 1e-6f) {
            out_is_inside = 1;
            out_dist_sq = 0.0f;
        } else {
            // 在边上或顶点上，距离视为0，但 is_inside 标记为 0
            // 这里可以根据需求调整，如果“在边上”也算“在三角形里”，可设为 1
            out_is_inside = 0; 
            out_dist_sq = 0.0f;
        }
        return;
    }
 
    // 3. 如果在外部，计算到三条边的最近距离
    out_is_inside = 0;
    
    float t_ab, t_bc, t_ca;
    float d_ab = closest_point_on_segment(p, a, b, t_ab);
    float d_bc = closest_point_on_segment(p, b, c, t_bc);
    float d_ca = closest_point_on_segment(p, c, a, t_ca);
 
    // 找出最近的边
    out_dist_sq = fminf(d_ab, fminf(d_bc, d_ca));
 
    // // 根据最近的边更新权重
    // // 权重定义：P = u*A + v*B + w*C
    // if (out_dist_sq == d_ab) {
    //     // 在 AB 边上
    //     out_weights.x = 1.0f - t_ab;
    //     out_weights.y = t_ab;
    //     out_weights.z = 0.0f;
    // } else if (out_dist_sq == d_bc) {
    //     // 在 BC 边上
    //     out_weights.x = 0.0f;
    //     out_weights.y = 1.0f - t_bc;
    //     out_weights.z = t_bc;
    // } else {
    //     // 在 CA 边上
    //     // 注意 CA 向量是从 C 指向 A，还是 A 指向 C?
    //     // closest_point_on_segment(p, c, a, t_ca) 中 p = c + t*(a-c)
    //     // 即 p = (1-t)*c + t*a
    //     // 所以 w = 1-t, u = t
    //     out_weights.x = t_ca;
    //     out_weights.y = 0.0f;
    //     out_weights.z = 1.0f - t_ca;
    // }
}
 
//------------------------------------------------------------------------------
// Query Kernel
//------------------------------------------------------------------------------
 
__global__ void query_location_kernel(
    const float2* query_pts, unsigned int num_queries,
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx,
    const float2* vertices, const int3* faces,
    LocationResult* results)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_queries) return;
 
    float2 qp = query_pts[i];
    
    // 初始化结果
    results[i].prim_idx = -1;
    results[i].dist_sq = FLT_MAX;
    results[i].u = results[i].v = results[i].w = 0.0f;
    results[i].is_inside = 0;
 
    unsigned int stack[128];
    int sp = 0;
    stack[sp++] = root_idx;
    // int times = 0;
    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        
        // 剪枝：如果 AABB 距离大于当前已找到的最佳距离，跳过
        // 如果我们已经找到了点所在的三角形 (dist_sq == 0)，
        // 则 dist_sq_point_aabb_2d 只有在点位于 AABB 内部时才返回 0。
        // 这意味着一旦找到包含点，我们会跳过所有不包含该点的 AABB。
        float d_aabb = dist_sq_point_aabb_2d(qp, aabbs[node_idx]);
        if (d_aabb > results[i].dist_sq) continue;
 
        int2 node = nodes[node_idx];
        
        // Leaf
        if (node.y == 0) {
            // times++;
            unsigned int prim_idx = node.x - 1; // Triangle index
            int3 f = faces[prim_idx];
            
            float2 v0 = vertices[f.x];
            float2 v1 = vertices[f.y];
            float2 v2 = vertices[f.z];
 
            float dist_sq;
            float3 weights;
            int is_inside;
 
            point_triangle_interaction(qp, v0, v1, v2, dist_sq, weights, is_inside);
 
            // 更新结果
            // 优先级：内部 (dist=0) > 距离更近
            // 如果 dist_sq 为 0，说明找到了包含点或点在边上，这是最高优先级
            if (dist_sq < results[i].dist_sq) {
                results[i].dist_sq = dist_sq;
                results[i].prim_idx = prim_idx;
                results[i].u = weights.x;
                results[i].v = weights.y;
                results[i].w = weights.z;
                results[i].is_inside = is_inside;
                if (is_inside) break;
            }
        } 
        else {
            // Internal node: Push children
            // stack[sp++] = node.y - 1;
            // stack[sp++] = node.x - 1;
            // 3. 内部节点处理：优化栈顺序
            
            // 获取子节点索引
            unsigned int left_child = node.x - 1;
            unsigned int right_child = node.y - 1;
 
            // 计算两个子节点的 AABB 距离
            float dist_left = dist_sq_point_aabb_2d(qp, aabbs[left_child]);
            float dist_right = dist_sq_point_aabb_2d(qp, aabbs[right_child]);
 
            // 策略：距离远的先压栈，距离近的后压栈（先处理）
            
            // 如果左边更远，先压左边
            if (dist_left > dist_right) {
                // 剪枝优化：只有当距离有可能比当前最佳小时才压栈
                if (dist_left <= results[i].dist_sq) 
                    stack[sp++] = left_child;
                if (dist_right <= results[i].dist_sq) 
                    stack[sp++] = right_child;
            } 
            else {
                // 右边更远（或相等），先压右边
                if (dist_right <= results[i].dist_sq) 
                    stack[sp++] = right_child;
                if (dist_left <= results[i].dist_sq) 
                    stack[sp++] = left_child;
            }
        }
    }
    // if (!results[i].is_inside) {
    //     printf("outside times: %d\n",times);
    // }else {
    //     printf("inside times: %d\n",times);
    // }
}

// Kernel for finding the nearest edge to a single point
__global__ void query_nearest_edge_kernel( 
    float2 query, 
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx, 
    const float2* vertices, const int2* edges, 
    NearestEdgeResult* result) 
{
    // Standard stack-based traversal for a single thread
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
 
    float best_dist = FLT_MAX;
    int best_idx = -1;
    float best_t = 0.0f;
 
    unsigned int stack[64];
    int sp = 0;
    stack[sp++] = root_idx;
 
    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        
        // Pruning
        float d_aabb = dist_sq_point_aabb_2d(query, aabbs[node_idx]);
        if (d_aabb > best_dist) continue;
 
        int2 node = nodes[node_idx];
        if (node.y == 0) { // Leaf
            unsigned int prim_idx = node.x - 1;
            int2 edge = edges[prim_idx];
            float2 a = vertices[edge.x];
            float2 b = vertices[edge.y];
            
            float2 ab = b - a;
            float2 ap = query - a;
            float len_sq = dot(ab, ab);
            float t = 0.0f;
            float dist_sq;
 
            if (len_sq > 1e-12f) {
                t = dot(ap, ab) / len_sq;
                t = fmaxf(0.0f, fminf(1.0f, t));
                float2 closest = a + t * ab;
                float2 diff = query - closest;
                dist_sq = dot(diff, diff);
            } else {
                dist_sq = dot(ap, ap);
            }
 
            if (dist_sq < best_dist) {
                best_dist = dist_sq;
                best_idx = prim_idx;
                best_t = t;
            }
        } else {
            unsigned int left_child = node.x - 1;
            unsigned int right_child = node.y - 1;
 
            float dist_left = dist_sq_point_aabb_2d(query, aabbs[left_child]);
            float dist_right = dist_sq_point_aabb_2d(query, aabbs[right_child]);
 
            if (dist_left > dist_right) {
                if (dist_left <= best_dist) 
                    stack[sp++] = left_child;
                if (dist_right <= best_dist) 
                    stack[sp++] = right_child;
            } 
            else {
                if (dist_right <= best_dist) 
                    stack[sp++] = right_child;
                if (dist_left <= best_dist) 
                    stack[sp++] = left_child;
            }
        }
    }
    result->idx = best_idx;
    result->dist_sq = best_dist;
    result->t = best_t;
}
// Kernel for self-intersection check within a loop
__global__ void self_intersect_kernel( 
    const int2* nodes, const AABB2D* aabbs, unsigned int root_idx, unsigned int num_edges,
    const float2* vertices, const int2* edges,
    IntersectionResult* result) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_edges) return;
    
    // Early exit if already found
    if (result->found) return;
 
    // AABB for edge i
    int2 edge_i = edges[i];
    float2 p1 = vertices[edge_i.x];
    float2 p2 = vertices[edge_i.y];
    AABB2D box_i;
    box_i.min = make_float2(fminf(p1.x, p2.x), fminf(p1.y, p2.y));
    box_i.max = make_float2(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y));
 
    unsigned int stack[64];
    int sp = 0;
    stack[sp++] = root_idx;
 
    while (sp > 0) {
        unsigned int node_idx = stack[--sp];
        
        if (result->found) return;
        if (!aabb_overlap_2d(box_i, aabbs[node_idx])) continue;
 
        int2 node = nodes[node_idx];
        if (node.y == 0) { // Leaf
            unsigned int prim_idx = node.x - 1;
            
            // Filter neighbors (adjacent edges in loop)
            int diff = abs((int)i - (int)prim_idx);
            if (diff <= 1 || diff == (int)num_edges - 1) continue;
 
            int2 edge_j = edges[prim_idx];
            float2 p3 = vertices[edge_j.x];
            float2 p4 = vertices[edge_j.y];
 
            // Segment-Segment Intersection
            float2 d1 = p2 - p1;
            float2 d2 = p4 - p3;
            float2 diff_st = p1 - p3;
            
            float cross_d = cross(d1, d2);
            float cross_diff_d2 = cross(diff_st, d2);
            float cross_diff_d1 = cross(diff_st, d1);
 
            // Check parallel
            if (fabsf(cross_d) < 1e-8f) continue;
 
            // float t = cross_diff_d2 / cross_d;
            // float u = cross_diff_d1 / cross_d;
            float t = -cross_diff_d2 / cross_d;
            float u = -cross_diff_d1 / cross_d;
 
            if (t >= 0.0f && t <= 1.0f && u >= 0.0f && u <= 1.0f) {
                // Found intersection
                int old = atomicCAS(&result->found, 0, 1);
                if (old == 0) {
                    result->idx = i;
                    result->t = t;
                }
                return;
            }
        } else {
            stack[sp++] = node.y - 1;
            stack[sp++] = node.x - 1;
        }
    }
}
} // namespace lbvh2d
