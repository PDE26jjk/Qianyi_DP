#include "pattern_helper.h"

#include "lbvh_2d.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "common/cuda_utils.h"
#include "common/device.h"


PatternHelper& PatternHelper::Instance() {
    static PatternHelper instance;
    init_device();
    return instance;
}

struct PatternHelper::Impl {
    BVH2D bvh;
    thrust::device_vector<float2> d_vertices;
    thrust::device_vector<int2> d_edges;
    thrust::device_vector<int> d_loop_offsets;
    bool initialized = false;
};

PatternHelper::PatternHelper() : impl(new Impl()) {}
PatternHelper::~PatternHelper() { delete impl; }
__device__ static void apply_transform(float2 p_in, const float* m, float2& p_out) {
    // Column-Major storage (Matrix * Column vector):
    // x' = m[0]*x + m[4]*y + m[12]
    // y' = m[1]*x + m[5]*y + m[13]
    p_out.x = p_in.x * m[0] + p_in.y * m[1] + m[3];
    p_out.y = p_in.x * m[4] + p_in.y * m[5] + m[7];
}

__global__ static void transform_points_kernel(
    const float2* raw_pts, int num_pts,
    const int* loop_offsets, const float* transforms, int num_loops,
    float2* out_pts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_pts ) return;

    // Binary search for loop index
    // We need to find 'k' such that loop_offsets[k] <= idx < loop_offsets[k+1]
    // loop_offsets has size num_loops + 1
    int lo = 0;
    int hi = num_loops;
    while ( lo < hi ) {
        int mid = (lo + hi + 1) / 2;
        if ( loop_offsets[mid] <= idx ) lo = mid;
        else hi = mid - 1;
    }
    int loop_id = lo;

    // Load matrix pointer
    const float* m = &transforms[loop_id * 16];

    float2 p_in = raw_pts[idx];
    float2 p_out;
    apply_transform(p_in, m, p_out);
    out_pts[idx] = p_out;
}
void PatternHelper::update_edges(std::vector<float>& edge_points, std::vector<int>& loop_sizes,
    std::vector<float>& loop_transforms) {
    if ( edge_points.empty() ) return;
    size_t num_vertices = edge_points.size() / 2;
    size_t num_loops = loop_sizes.size();

    // 1. Compute Offsets on Host
    std::vector<int> h_loop_offsets(num_loops + 1);
    h_loop_offsets[0] = 0;
    for ( size_t i = 0; i < num_loops; ++i ) {
        h_loop_offsets[i + 1] = h_loop_offsets[i] + loop_sizes[i];
    }

    // 2. Upload Raw Data
    thrust::device_vector<float2> d_raw_vertices(num_vertices);
    thrust::copy(reinterpret_cast<float2*>(edge_points.data()),
        reinterpret_cast<float2*>(edge_points.data()) + num_vertices,
        d_raw_vertices.begin());

    impl->d_loop_offsets = h_loop_offsets; // Upload offsets

    thrust::device_vector<float> d_transforms = loop_transforms; // Upload matrices

    // 3. Transform Points
    impl->d_vertices.resize(num_vertices);
    {
        transform_points_kernel<<<(num_vertices + 255) / 256, 256>>>(
            thrust::raw_pointer_cast(d_raw_vertices.data()), num_vertices,
            thrust::raw_pointer_cast(impl->d_loop_offsets.data()),
            thrust::raw_pointer_cast(d_transforms.data()), num_loops,
            thrust::raw_pointer_cast(impl->d_vertices.data())
            );
    }

    // 4. Build Edges (Host logic -> Device copy)
    // Since we have loop_offsets, we can parallelize edge creation or just keep simple host loop
    std::vector<int2> h_edges;
    h_edges.reserve(num_vertices); // Estimate
    for ( size_t i = 0; i < num_loops; ++i ) {
        int start = h_loop_offsets[i];
        int size = loop_sizes[i];
        for ( int k = 0; k < size; ++k ) {
            int next = (k + 1) % size;
            h_edges.push_back(make_int2(start + k, start + next));
        }
    }
    impl->d_edges = h_edges;
    size_t num_edges = h_edges.size();

    // 5. Initialize & Build BVH
    if ( !impl->initialized || num_edges > impl->bvh.nodes.size() ) {
        lbvh2d::initialize(num_edges);
        impl->initialized = true;
    }
    lbvh2d::build_edge_bvh(impl->d_vertices, impl->d_edges, impl->bvh);

}
void PatternHelper::find_nearest_edge(float query_point[2], int& res_index, float& res_weight) {
    if ( impl->d_edges.empty() ) {
        res_index = -1;
        res_weight = 0.0f;
        return;
    }

    float2 query = make_float2(query_point[0], query_point[1]);
    thrust::device_vector<lbvh2d::NearestEdgeResult> d_res(1);

    // Launch 1 thread to traverse tree
    lbvh2d::query_nearest_edge_kernel<<<1, 1>>>(
        query,
        thrust::raw_pointer_cast(impl->bvh.nodes.data()),
        thrust::raw_pointer_cast(impl->bvh.aabbs.data()),
        impl->bvh.root_idx,
        thrust::raw_pointer_cast(impl->d_vertices.data()),
        thrust::raw_pointer_cast(impl->d_edges.data()),
        thrust::raw_pointer_cast(d_res.data())
        );
    cudaDeviceSynchronize();

    lbvh2d::NearestEdgeResult h_res = d_res[0];
    res_index = h_res.idx;
    res_weight = h_res.t;
}
bool PatternHelper::check_edge_intersection(std::vector<float>& edge_points, int& res_index, float& res_weight) {
    if ( edge_points.empty() ) return false; // nothing cannot intersect.
    if ( edge_points.size() <= 4 ) return true; // one point or one edge.
    // 1. Prepare Temp Data (Single Loop assumed)
    size_t num_vertices = edge_points.size() / 2;
    thrust::device_vector<float2> d_temp_verts(num_vertices);
    thrust::copy(reinterpret_cast<float2*>(edge_points.data()),
        reinterpret_cast<float2*>(edge_points.data()) + num_vertices,
        d_temp_verts.begin());

    // Build edges for the loop
    size_t num_edges = num_vertices; // Loop implies N vertices -> N edges
    thrust::device_vector<int2> d_temp_edges(num_edges);
    // Generate indices 0-1, 1-2, ..., N-1-0
    thrust::host_vector<int2> h_temp_edges(num_edges);
    for ( int i = 0; i < num_edges; ++i ) {
        h_temp_edges[i] = make_int2(i, (i + 1) % num_edges);
    }
    d_temp_edges = h_temp_edges;

    // 2. Build Temp BVH
    // Ensure global storage is large enough
    lbvh2d::initialize(num_edges);

    BVH2D temp_bvh;
    lbvh2d::build_edge_bvh(d_temp_verts, d_temp_edges, temp_bvh);

    // 3. Query Self-Intersection
    thrust::device_vector<lbvh2d::IntersectionResult> d_int_res(1);
    // Initialize result
    lbvh2d::IntersectionResult init_res;
    init_res.found = 0;
    init_res.idx = -1;
    init_res.t = 0;
    d_int_res[0] = init_res;

    lbvh2d::self_intersect_kernel<<<(num_edges + 255) / 256, 256>>>(
        thrust::raw_pointer_cast(temp_bvh.nodes.data()),
        thrust::raw_pointer_cast(temp_bvh.aabbs.data()),
        temp_bvh.root_idx,
        num_edges,
        thrust::raw_pointer_cast(d_temp_verts.data()),
        thrust::raw_pointer_cast(d_temp_edges.data()),
        thrust::raw_pointer_cast(d_int_res.data())
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    lbvh2d::IntersectionResult h_res = d_int_res[0];
    res_index = h_res.idx;
    res_weight = h_res.t;

    return h_res.found;
}
