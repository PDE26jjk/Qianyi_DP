#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm>

#include "geometric_operator.cuh"
#include "solver_base.cuh"
#include "common/atomic_utils.cuh"

static __device__ bool find_edge(int v0, int v1, const int2* lookup, const int2* dir_edges, int& edge) {
    auto [offset, degree] = lookup[v0];
    if ( dir_edges[offset].x <= v1 && v1 <= dir_edges[offset + degree - 1].x ) {
        for ( int d = 0; d < degree; d++ ) {
            if ( dir_edges[offset + d].x == v1 ) {
                edge = dir_edges[offset + d].y;
                return true;
            }
        }
    }
    return false;
}
constexpr char stitch_status_running = (char)(0);
constexpr char stitch_status_done = (char)(1 << 1);
constexpr char stitch_status_suspend = (char)(1 << 2);
void SolverBase::init_sewing() {
    stitch_sewing.resize(params.nb_all_stitches);
    sewing_edges.resize(params.nb_all_stitches);
    sewing_e2t.resize(params.nb_all_stitches);
    sewing_edge_opposite_points.resize(params.nb_all_stitches);
    stitches_done_count.assign((size_t)1, 0);
    stitches_status.assign(params.nb_all_stitches, stitch_status_running);

    vertex_proxy.assign(thrust::make_counting_iterator(0), thrust::make_counting_iterator(params.nb_all_vertices));
    // generate stitch_sewing
    auto source_iter_begin = thrust::make_transform_iterator(
        sewing_lines.begin(),
        [] __host__ __device__ (const SewingData& key) { return key.start_idx; }
        );
    thrust::upper_bound(thrust::device,
        source_iter_begin, source_iter_begin + (int)sewing_lines.size(),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(params.nb_all_stitches),
        stitch_sewing.begin()
        );
    thrust::transform(
        stitch_sewing.begin(), stitch_sewing.end(), stitch_sewing.begin(),
        thrust::placeholders::_1 - 1
        );
    // 
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_stitches,
        [
            vertices = thrust::raw_pointer_cast(vertices_2D.data()), // float3
            triangles = thrust::raw_pointer_cast(triangles.data()), // int3
            edges = thrust::raw_pointer_cast(edges.data()), // int2
            indices = thrust::raw_pointer_cast(triangle_indices.data()), // int3
            dir_edges = thrust::raw_pointer_cast(dir_edges.data()), // int2
            edge_lookup = thrust::raw_pointer_cast(edge_lookup.data()), // int2
            e2t = thrust::raw_pointer_cast(e2t.data()), // int2
            edge_opposite_points = thrust::raw_pointer_cast(edge_opposite_points.data()), // int2
            sewing_lines = thrust::raw_pointer_cast(sewing_lines.data()), // 
            sewing_e2t = thrust::raw_pointer_cast(sewing_e2t.data()), // int2
            sewing_edges = thrust::raw_pointer_cast(sewing_edges.data()), // int2
            sewing_edge_opposite_points = thrust::raw_pointer_cast(sewing_edge_opposite_points.data()), // int2
            stitches = thrust::raw_pointer_cast(stitches.data()), // int2
            stitch_sewing = thrust::raw_pointer_cast(stitch_sewing.data()), // int
            nb_all_stitches=params.nb_all_stitches
        ] __device__ (const int i) {
            if ( i >= nb_all_stitches - 1 ) return;
            int sewing_idx = stitch_sewing[i];
            // only handle stitch in same sewing
            if ( sewing_idx != stitch_sewing[i + 1] ) return;
            int2 s0 = stitches[i], s1 = stitches[i + 1];
            int v0 = s0.x, v1 = s1.x, v2 = s0.y, v3 = s1.y;
            int edge1_idx, edge2_idx;
            auto sewing = sewing_lines[sewing_idx];
            int sewing_e_idx = i - sewing_idx;
            sewing_edges[sewing_e_idx] = make_int2(v0, v1);
            if ( find_edge(v0, v1, edge_lookup, dir_edges, edge1_idx) &&
                find_edge(v2, v3, edge_lookup, dir_edges, edge2_idx) ) {
                bool reverse = (edge1_idx > 0) ^ (edge2_idx > 0); // TODO how to use it?
                edge1_idx = abs(edge1_idx);
                auto tris1 = e2t[edge1_idx];
                auto eop1 = edge_opposite_points[edge1_idx];
                // Edge vertices could be in reverse order.
                if ( tris1.x == -1 || tris1.y == -1 ) {
                    sewing_e2t[sewing_e_idx].x = tris1.x != -1 ? tris1.x : tris1.y;
                    sewing_edge_opposite_points[sewing_e_idx].x = eop1.x != -1 ? eop1.x : eop1.y;
                }
                else { // Internal lines vertices should always be in same order.
                    sewing_e2t[sewing_e_idx].x = tris1.x;
                    sewing_edge_opposite_points[sewing_e_idx].x = eop1.x;
                }
                edge2_idx = abs(edge2_idx);
                auto tris2 = e2t[edge2_idx];
                auto eop2 = edge_opposite_points[edge2_idx];
                if ( tris2.x == -1 || tris2.y == -1 ) { // Edge
                    sewing_e2t[sewing_e_idx].y = tris2.x != -1 ? tris2.x : tris2.y;
                    sewing_edge_opposite_points[sewing_e_idx].y = eop2.x != -1 ? eop2.x : eop2.y;
                }
                else { // Internal lines
                    sewing_e2t[sewing_e_idx].y = tris1.y;
                    sewing_edge_opposite_points[sewing_e_idx].y = eop2.y;
                }
            }
            else {
                printf("Error: edge not found for sewing!\n");
            }
        });
    // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
    // std::vector<type> __##v(_##v.begin(), _##v.end())
    //
    // CHECK(sewing_lines,SewingData);
    // CHECK(stitches,int2);
    // CHECK(stitch_sewing,int);
    // CHECK(sewing_edges,int2);
    // CHECK(sewing_e2t,int2);
    // CHECK(sewing_edge_opposite_points,int2);
    //
    // #undef CHECK
}


static __global__ void check_sewing_kernel(
    int* vertex_proxy,
    int* stitches_done_count,
    char* stitches_status,
    const float3* __restrict__ vertices,
    const char* __restrict__ mask,
    const int2* __restrict__ stitches,
    float min_dist_sq,
    bool forced_connect,
    int num_stitches
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_stitches ) return;
    auto stitch = stitches[idx];
    if ( stitches_status[idx] == stitch_status_done ) return;// Already done
    auto [v0,v1] = stitch;
    if ( mask[v0] && mask[v1] ) { // No need to move
        // stitches_status[idx] = stitch_status_suspend;
        stitches_status[idx] = stitch_status_done;
        atomicAdd(stitches_done_count, 1);       
        return;
    }
    v0 = min(vertex_proxy[v0], v0);
    v1 = min(vertex_proxy[v1], v1);
    auto p0 = vertices[v0], p1 = vertices[v1];
    if ( forced_connect || len_sq(p0 - p1) < min_dist_sq ) {
        if ( mask[v0] ) {
            vertex_proxy[v1] = v0;
        }
        else if ( mask[v1] ) {
            vertex_proxy[v0] = v1;
        }
        else {
            atomicMin(&vertex_proxy[v1], v0);
            atomicMin(&vertex_proxy[v0], v1);
        }
        stitches_status[idx] = stitch_status_done;
        atomicAdd(stitches_done_count, 1);
    }
}
static __global__ void update_proxy_mask(
    int* vertex_proxy,
    char* __restrict__ mask,
    int num_vertices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_vertices ) return;
    int proxy_idx = idx;
    while ( vertex_proxy[proxy_idx] != proxy_idx )
        proxy_idx = vertex_proxy[proxy_idx];
    if ( proxy_idx != idx ) {
        vertex_proxy[idx] = proxy_idx;
        mask[idx] |= static_cast<char>(MaskBit::proxy_mask);
    }
}
static __global__ void update_proxy_edge(
    int2* __restrict__ edges,
    int2* __restrict__ opposite_points,
    const int* vertex_proxy,
    int num_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_edges ) return;
    auto op = opposite_points[idx];
    if ( op.x != -1 && vertex_proxy[op.x] != op.x ) op.x = vertex_proxy[op.x];
    if ( op.y != -1 && vertex_proxy[op.y] != op.y ) op.y = vertex_proxy[op.y];

    auto e = edges[idx];
    if ( vertex_proxy[e.x] != e.x ) e.x = vertex_proxy[e.x];
    if ( vertex_proxy[e.y] != e.y ) e.y = vertex_proxy[e.y];
    if ( e.y < e.x ) {
        int temp = e.y;
        e.y = e.x;
        e.x = temp;

        temp = op.y;
        op.y = op.x;
        op.x = temp;

    }
    edges[idx] = e;
    opposite_points[idx] = op;

}

static __global__ void update_proxy_dir_edges(
    int2* __restrict__ dir_edges,
    const int* vertex_proxy,
    int num_dir_edges
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_dir_edges ) return;
    int2 e = dir_edges[idx];
    if ( vertex_proxy[e.x] != e.x ) dir_edges[idx].x = vertex_proxy[e.x];
}
static __device__ void reorder_triangle(
    int i, int3* triangles, int3* indices, int2* e2t,
    const float3* vertices, const int2* edge_lookup, const int2* dir_edges
) {
    // 1. Load vertices of the triangle
    int3 tri_v = indices[i];
    int v0 = tri_v.x, v1 = tri_v.y, v2 = tri_v.z;

    // 2. Reorder vertices for consistent orientation
    if ( v1 > v2 ) {
        int tmp = v1;
        v1 = v2;
        v2 = tmp;
    }
    if ( v0 > v1 ) {
        int tmp = v0;
        v0 = v1;
        v1 = tmp;
    }
    float3 n_in = make_float3(0.0f, 0.0f, 1.0f);
    float3 p0 = vertices[v0], p1 = vertices[v1], p2 = vertices[v2];
    // Check orientation: (p1-p0) cross (p2-p0) dot normal
    if ( dot(cross(p1 - p0, p2 - p0), n_in) < 0.0f ) {
        int tmp = v1;
        v1 = v2;
        v2 = tmp; // Swap to keep CCW
        p1 = vertices[v1];
        p2 = vertices[v2];
    }
    indices[i] = make_int3(v0, v1, v2);
    // 3. Find global edge indices using the lookup table
    int e1_i = v2e_include_stitches(v0, v1, edge_lookup, dir_edges);
    int e2_i = v2e_include_stitches(v0, v2, edge_lookup, dir_edges);
    int e3_i = v2e_include_stitches(v1, v2, edge_lookup, dir_edges);
    if (e1_i == -1 || e2_i == -1 || e3_i == -1 ) {
        printf("ERROR in reorder_triangle!\n");
        return;
    }
    triangles[i] = make_int3(e1_i, e2_i, e3_i);

    // 4. Update Edge-to-Triangle (e2t) mapping
    // Caution: Multiple triangles share one edge. 
    // If not handled by specific logic, use atomicExch or similar if needed.
    e2t[e1_i].x = i; // Simplified assignment
    e2t[e2_i].y = i;

    // Logic for e3's slot based on cross product
    int v3 = v2;
    float3 e3_vec = p2 - p1;
    if ( v2 > v1 ) {
        e3_vec = -e3_vec;
        v3 = v1;
    }

    if ( dot(cross(p0 - vertices[v3], e3_vec), n_in) > 0.0f ) {
        e2t[e3_i].y = i;
    }
    else {
        e2t[e3_i].x = i;
    }
}
static __global__ void update_proxy_triangles(
    float3* __restrict__ debug_colors,
    int3* __restrict__ triangles,
    int3* __restrict__ triangle_indices,
    int2* __restrict__ e2t,
    const int* vertex_proxy,
    const float3* __restrict__ vertices,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    int num_triangles
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_triangles ) return;
    auto t = triangle_indices[idx];
    bool need_to_reorder = false;
    if ( vertex_proxy[t.x] != t.x ) {
        triangle_indices[idx].x = vertex_proxy[t.x];
        need_to_reorder = true;
    }
    if ( vertex_proxy[t.y] != t.y ) {
        triangle_indices[idx].y = vertex_proxy[t.y];
        need_to_reorder = true;
    }
    if ( vertex_proxy[t.z] != t.z ) {
        triangle_indices[idx].z = vertex_proxy[t.z];
        need_to_reorder = true;
    }
    if ( need_to_reorder ) {
        reorder_triangle(idx, triangles, triangle_indices, e2t, vertices, edge_lookup, dir_edges);
        t = triangle_indices[idx];
        debug_colors[t.x] = make_float3(1.0f, 0.0f, 0.0f);
        debug_colors[t.y] = make_float3(1.0f, 0.0f, 0.0f);
        debug_colors[t.z] = make_float3(1.0f, 0.0f, 0.0f);
    }
}

void SolverBase::check_sewing(bool forced_connect) {
    int block = 256;

    int n = params.nb_all_stitches;
    int stitches_done_count_old;
    cudaMemcpy(&stitches_done_count_old, stitches_done_count.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    sewing_done = true;
    if ( stitches_done_count_old >= n ) return;
    float min_dist = 1e-2f;
    check_sewing_kernel<<<(n + block - 1) / block, block>>>(
        vertex_proxy.data().get(),
        stitches_done_count.data().get(),
        stitches_status.data().get(),
        vertices_world.data().get(),
        vertices_mask.data().get(),
        stitches.data().get(),
        min_dist * min_dist, forced_connect, n);
    int stitches_done_count_new;
    cudaMemcpy(&stitches_done_count_new, stitches_done_count.data().get(), sizeof(int), cudaMemcpyDeviceToHost);

    std::cout << "stitches: " << stitches_done_count_new << "/" << n << std::endl;
    if ( stitches_done_count_new > stitches_done_count_old ) {
        // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
        // std::vector<type> __##v(_##v.begin(), _##v.end())
        n = params.nb_all_cloth_vertices;
        // CHECK(stitches_status, char);
        update_proxy_mask<<<(n + block - 1) / block, block>>>(
            vertex_proxy.data().get(),
            vertices_mask.data().get(),
            n);
        n = params.nb_all_cloth_edges;

        update_proxy_edge<<<(n + block - 1) / block, block>>>(
            edges.data().get(),
            edge_opposite_points.data().get(),
            vertex_proxy.data().get(),
            n);
        n = (int)dir_edges.size();


        update_proxy_dir_edges<<<(n + block - 1) / block, block>>>(
            dir_edges.data().get(),
            vertex_proxy.data().get(),
            n);
        n = params.nb_all_cloth_triangles;
        // CHECK(vertex_proxy, int);
        update_proxy_triangles<<<(n + block - 1) / block, block>>>(
            debug_colors.data().get(),
            triangles.data().get(),
            triangle_indices.data().get(),
            e2t.data().get(),
            vertex_proxy.data().get(),
            vertices_2D.data().get(),
            edge_lookup.data().get(),
            dir_edges.data().get(),
            n);
        // CHECK(sewing_lines, SewingData);
        // #undef CHECK
    }

    sewing_done = false;
}
