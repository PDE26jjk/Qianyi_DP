#include "pattern_helper.h"

#include <algorithm>

#include "lbvh_2d.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

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


// std::vector<lbvh2d::FullIntersectionResult> PatternHelper::get_all_intersections(
//     std::vector<float>& edge_points,
//     std::vector<int>& curve_sizes,
//     std::vector<int8_t>& is_loops) // 建议使用 int8_t 代替 bool 以避免 std::vector<bool> 的位压缩问题
// {
//     std::vector<lbvh2d::FullIntersectionResult> h_results;
//     size_t num_vertices = edge_points.size() / 2;
//     if ( num_vertices == 0 ) return h_results;
//     // 1. 在 Host 端构建拓扑与长度辅助数据
//     std::vector<int2> h_edges;
//     std::vector<int> h_edge_to_curve;
//     std::vector<int> h_edge_local_idx;
//     std::vector<float> h_edge_lengths;
//     std::vector<float> h_edge_prefix_sums;
//     std::vector<float> h_curve_total_lengths(curve_sizes.size(), 0.0f);
//     std::vector<int> h_curve_num_edges(curve_sizes.size(), 0);
//     int vert_offset = 0;
//     for ( int c = 0; c < curve_sizes.size(); ++c ) {
//         int n_v = curve_sizes[c];
//         bool is_loop = (is_loops[c] != 0);
//         int n_e = is_loop ? n_v : (n_v > 0 ? n_v - 1 : 0);
//         h_curve_num_edges[c] = n_e;
//         float prefix_sum = 0.0f;
//         for ( int k = 0; k < n_e; ++k ) {
//             int v0 = vert_offset + k;
//             int v1 = vert_offset + ((k + 1) % n_v); // 如果是开曲线，k+1最大为n_v-1，不会越界
//             float2 p0 = *reinterpret_cast<float2*>(&edge_points[v0 * 2]);
//             float2 p1 = *reinterpret_cast<float2*>(&edge_points[v1 * 2]);
//             float len = sqrtf((p1.x - p0.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p1.y - p0.y));
//             h_edges.push_back(make_int2(v0, v1));
//             h_edge_to_curve.push_back(c);
//             h_edge_local_idx.push_back(k);
//             h_edge_lengths.push_back(len);
//             h_edge_prefix_sums.push_back(prefix_sum);
//             prefix_sum += len;
//         }
//         h_curve_total_lengths[c] = prefix_sum;
//         vert_offset += n_v;
//     }
//     size_t num_edges = h_edges.size();
//     if ( num_edges == 0 ) return h_results;
//     // 2. 拷贝数据到 Device
//     thrust::device_vector<float2> d_vertices(num_vertices);
//     thrust::copy(reinterpret_cast<float2*>(edge_points.data()),
//         reinterpret_cast<float2*>(edge_points.data()) + num_vertices,
//         d_vertices.begin());
//     thrust::device_vector<int2> d_edges = h_edges;
//     thrust::device_vector<int> d_edge_to_curve = h_edge_to_curve;
//     thrust::device_vector<int> d_edge_local_idx = h_edge_local_idx;
//     thrust::device_vector<float> d_edge_lengths = h_edge_lengths;
//     thrust::device_vector<float> d_edge_prefix_sums = h_edge_prefix_sums;
//     thrust::device_vector<float> d_curve_total_lengths = h_curve_total_lengths;
//     thrust::device_vector<int> d_curve_num_edges = h_curve_num_edges;
//     thrust::device_vector<int8_t> d_is_loops = is_loops;
//     // 3. 构建 LBVH
//     lbvh2d::initialize(num_edges);
//     BVH2D temp_bvh;
//     lbvh2d::build_edge_bvh(d_vertices, d_edges, temp_bvh);
//     // 4. 分配结果缓冲区并执行 Kernel
//     // 假设最大交点数量为边数的 4 倍（可根据业务场景调整）
//     unsigned int max_results = num_edges * 4;
//     thrust::device_vector<lbvh2d::FullIntersectionResult> d_results(max_results);
//     thrust::device_vector<unsigned int> d_out_count(1);
//     thrust::fill(d_out_count.begin(), d_out_count.end(), 0);
//     int blocks = (num_edges + 255) / 256;
//     lbvh2d::all_intersections_kernel<<<blocks, 256>>>(
//         thrust::raw_pointer_cast(temp_bvh.nodes.data()),
//         thrust::raw_pointer_cast(temp_bvh.aabbs.data()),
//         temp_bvh.root_idx,
//         num_edges,
//         thrust::raw_pointer_cast(d_vertices.data()),
//         thrust::raw_pointer_cast(d_edges.data()),
//         thrust::raw_pointer_cast(d_edge_to_curve.data()),
//         thrust::raw_pointer_cast(d_edge_local_idx.data()),
//         thrust::raw_pointer_cast(d_edge_lengths.data()),
//         thrust::raw_pointer_cast(d_edge_prefix_sums.data()),
//         thrust::raw_pointer_cast(d_curve_total_lengths.data()),
//         thrust::raw_pointer_cast(d_curve_num_edges.data()),
//         thrust::raw_pointer_cast(d_is_loops.data()),
//         thrust::raw_pointer_cast(d_results.data()),
//         max_results,
//         thrust::raw_pointer_cast(d_out_count.data())
//         );
//     CUDA_CHECK(cudaDeviceSynchronize());
//     // 5. 取回结果
//     unsigned int h_out_count = d_out_count[0];
//     if ( h_out_count > max_results ) {
//         std::cout << "h_out_count > max_results)" << std::endl;
//         h_out_count = max_results;
//     } // 截断保护
//     h_results.resize(h_out_count);
//     thrust::copy(d_results.begin(), d_results.begin() + h_out_count, h_results.begin());
//     std::ranges::sort(h_results, 
//         [](const auto& a, const auto& b) {
//             if (a.curve_a != b.curve_a) 
//                 return a.curve_a < b.curve_a;
//             if (a.curve_b != b.curve_b) 
//                 return a.curve_b < b.curve_b;
//             return a.t_a < b.t_a;
//         }
//     );
//     return h_results;
// }
std::vector<lbvh2d::FullIntersectionResult> PatternHelper::get_all_intersections(
    std::vector<float>& edge_points,
    std::vector<int>& curve_sizes,
    std::vector<int8_t>& is_loops,
    std::vector<int>& num_sections_per_curve,  // 新增：每个曲线的 section 数量
    std::vector<int>& section_point_sizes)           // 新增：所有 section 的顶点数量
{
    std::vector<lbvh2d::FullIntersectionResult> h_results;
    size_t num_vertices = edge_points.size() / 2;
    if ( num_vertices == 0 ) return h_results;

    std::vector<int2> h_edges;
    std::vector<int> h_edge_to_curve;
    std::vector<int> h_edge_local_idx;
    std::vector<float> h_edge_lengths;
    std::vector<float> h_edge_prefix_sums;
    std::vector<float> h_curve_total_lengths(curve_sizes.size(), 0.0f);
    std::vector<int> h_curve_num_edges(curve_sizes.size(), 0);

    // --- 新增：Section 相关预处理数组 ---
    std::vector<int> h_edge_to_section;
    std::vector<float> h_section_prefix_sums;
    std::vector<float> h_section_total_lengths;
    std::vector<int> h_curve_to_section_offset(curve_sizes.size(), 0);

    int vert_offset = 0;
    int global_edge_idx = 0;
    int global_section_idx = 0;

    for ( int c = 0; c < curve_sizes.size(); ++c ) {
        int n_v = curve_sizes[c];
        bool is_loop = (is_loops[c] != 0);
        int n_e = is_loop ? n_v : (n_v > 0 ? n_v - 1 : 0);
        h_curve_num_edges[c] = n_e;
        h_curve_to_section_offset[c] = global_section_idx;

        float prefix_sum = 0.0f;
        for ( int k = 0; k < n_e; ++k ) {
            int v0 = vert_offset + k;
            int v1 = vert_offset + ((k + 1) % n_v);

            float2 p0 = *reinterpret_cast<float2*>(&edge_points[v0 * 2]);
            float2 p1 = *reinterpret_cast<float2*>(&edge_points[v1 * 2]);
            float len = sqrtf((p1.x - p0.x) * (p1.x - p0.x) + (p1.y - p0.y) * (p1.y - p0.y));

            h_edges.push_back(make_int2(v0, v1));
            h_edge_to_curve.push_back(c);
            h_edge_local_idx.push_back(k);
            h_edge_lengths.push_back(len);
            h_edge_prefix_sums.push_back(prefix_sum);

            prefix_sum += len;
            global_edge_idx++;
        }
        h_curve_total_lengths[c] = prefix_sum;
        vert_offset += n_v;

        // --- 新增：按 Section 划分计算 ---
        int num_sec = num_sections_per_curve[c];
        float sec_prefix_sum = 0.0f;
        int edge_offset_in_curve = 0;

        for ( int s = 0; s < num_sec; ++s ) {
            int sv = section_point_sizes[global_section_idx];
            // 计算当前 section 包含的边数
            int sec_edges = (!is_loop && s == num_sec - 1) ? sv - 1 : sv;

            h_section_prefix_sums.push_back(sec_prefix_sum);
            float sec_len = 0.0f;

            // 为该 section 内的边打上标签并累加 section 长度
            for ( int e = 0; e < sec_edges; ++e ) {
                int current_edge_idx = (global_edge_idx - n_e) + edge_offset_in_curve + e;
                h_edge_to_section.push_back(global_section_idx); // 记录边属于哪个全局 section
                sec_len += h_edge_lengths[current_edge_idx];
            }

            h_section_total_lengths.push_back(sec_len);
            sec_prefix_sum += sec_len;
            edge_offset_in_curve += sec_edges;
            global_section_idx++;
        }
    }

    size_t num_edges = h_edges.size();
    if ( num_edges == 0 ) return h_results;

    // 拷贝数据到 Device
    thrust::device_vector<float2> d_vertices(num_vertices);
    thrust::copy(reinterpret_cast<float2*>(edge_points.data()),
        reinterpret_cast<float2*>(edge_points.data()) + num_vertices,
        d_vertices.begin());

    thrust::device_vector<int2> d_edges = h_edges;
    thrust::device_vector<int> d_edge_to_curve = h_edge_to_curve;
    thrust::device_vector<int> d_edge_local_idx = h_edge_local_idx;
    thrust::device_vector<float> d_edge_lengths = h_edge_lengths;
    thrust::device_vector<float> d_edge_prefix_sums = h_edge_prefix_sums;
    thrust::device_vector<float> d_curve_total_lengths = h_curve_total_lengths;
    thrust::device_vector<int> d_curve_num_edges = h_curve_num_edges;
    thrust::device_vector<int8_t> d_is_loops = is_loops;

    // 拷贝 Section 相关数据到 Device
    thrust::device_vector<int> d_edge_to_section = h_edge_to_section;
    thrust::device_vector<float> d_section_prefix_sums = h_section_prefix_sums;
    thrust::device_vector<float> d_section_total_lengths = h_section_total_lengths;
    thrust::device_vector<int> d_curve_to_section_offset = h_curve_to_section_offset;

    // 构建 LBVH
    lbvh2d::initialize(num_edges);
    BVH2D temp_bvh;
    lbvh2d::build_edge_bvh(d_vertices, d_edges, temp_bvh);

    unsigned int max_results = num_edges * 4;
    thrust::device_vector<lbvh2d::FullIntersectionResult> d_results(max_results);
    thrust::device_vector<unsigned int> d_out_count(1);
    thrust::fill(d_out_count.begin(), d_out_count.end(), 0);

    int blocks = (num_edges + 255) / 256;
    all_intersections_kernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(temp_bvh.nodes.data()),
        thrust::raw_pointer_cast(temp_bvh.aabbs.data()),
        temp_bvh.root_idx,
        num_edges,
        thrust::raw_pointer_cast(d_vertices.data()),
        thrust::raw_pointer_cast(d_edges.data()),
        thrust::raw_pointer_cast(d_edge_to_curve.data()),
        thrust::raw_pointer_cast(d_edge_local_idx.data()),
        thrust::raw_pointer_cast(d_edge_lengths.data()),
        thrust::raw_pointer_cast(d_edge_prefix_sums.data()),
        thrust::raw_pointer_cast(d_curve_total_lengths.data()),
        thrust::raw_pointer_cast(d_curve_num_edges.data()),
        thrust::raw_pointer_cast(d_is_loops.data()),
        thrust::raw_pointer_cast(d_edge_to_section.data()),
        thrust::raw_pointer_cast(d_section_prefix_sums.data()),
        thrust::raw_pointer_cast(d_section_total_lengths.data()),
        thrust::raw_pointer_cast(d_curve_to_section_offset.data()),
        thrust::raw_pointer_cast(d_results.data()),
        max_results,
        thrust::raw_pointer_cast(d_out_count.data())
        );
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned int h_out_count = d_out_count[0];
    if ( h_out_count > max_results ) h_out_count = max_results;

    h_results.resize(h_out_count);
    thrust::copy(d_results.begin(), d_results.begin() + h_out_count, h_results.begin());

    // 排序逻辑更新：由于 t_a 现在是相对 section 的，按原 t_a 排序无意义。
    // 必须按 curve_a, curve_b, section_a, t_a 的优先级排序，才能保持有序性
    std::ranges::sort(h_results,
        [](const auto& a, const auto& b) {
            if ( a.curve_a != b.curve_a )
                return a.curve_a < b.curve_a;
            if ( a.curve_b != b.curve_b )
                return a.curve_b < b.curve_b;
            if ( a.section_a != b.section_a )
                return a.section_a < b.section_a;
            return a.t_a < b.t_a;
        }
        );

    return h_results;
}

// 2. 路径压缩 Kernel (确保所有点直接指向其所在重叠类的最小索引代表)
__global__ void compress_path_kernel(unsigned int* map, unsigned int n) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    // 追踪到最终代表
    unsigned int root = map[i];
    while ( root != map[root] ) {
        root = map[root];
    }

    // 路径压缩：直接指向代表
    if ( map[i] != root ) {
        map[i] = root;
    }
}

// 3. 标记保留点并生成最终映射和位置
__global__ void generate_dedup_results_kernel(
    const float2* input_points, const unsigned int* map, const unsigned int* new_indices,
    float2* out_points, unsigned int* out_map, unsigned int n) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    unsigned int root = map[i];

    // 计算原点在新数组中的索引
    out_map[i] = new_indices[root];

    // 只有代表点才写入新的点位置数组
    if ( root == i ) {
        out_points[new_indices[i]] = input_points[i];
    }
}

void PatternHelper::deduplicate_points(
    std::vector<float>& point_data,
    float threshold,
    std::vector<float>& out_point_data,
    std::vector<unsigned int>& out_map) {
    size_t num_vertices = point_data.size() / 2;
    out_point_data.clear();
    out_map.clear();

    if ( num_vertices == 0 ) return;

    // 1. 上传点数据到 Device
    thrust::device_vector<float2> d_vertices(num_vertices);
    thrust::copy(reinterpret_cast<float2*>(point_data.data()),
        reinterpret_cast<float2*>(point_data.data()) + num_vertices,
        d_vertices.begin());

    lbvh2d::initialize(num_vertices);
    BVH2D temp_bvh;
    lbvh2d::build_point_bvh(d_vertices, temp_bvh);

    thrust::device_vector<unsigned int> d_map(num_vertices);
    thrust::sequence(d_map.begin(), d_map.end(), 0);

    float threshold_sq = threshold * threshold;

    // 4. 执行合并 Kernel
    int blocks = (num_vertices + 255) / 256;
    lbvh2d::merge_overlaps_kernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(d_vertices.data()),
        thrust::raw_pointer_cast(temp_bvh.nodes.data()),
        thrust::raw_pointer_cast(temp_bvh.aabbs.data()),
        temp_bvh.root_idx,
        threshold_sq,
        thrust::raw_pointer_cast(d_map.data()),
        num_vertices);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. 执行路径压缩 (运行2次确保长链式重叠也能完全压缩到根节点)
    compress_path_kernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(d_map.data()), num_vertices);
    compress_path_kernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(d_map.data()), num_vertices);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. 计算保留点的新索引
    // is_kept[i] = (map[i] == i) ? 1 : 0
    thrust::device_vector<unsigned int> d_is_kept(num_vertices);
    thrust::transform(d_map.begin(), d_map.end(),
        thrust::make_counting_iterator(0u), d_is_kept.begin(),
        thrust::equal_to<unsigned int>());

    // 独占扫描求和，得到每个保留点的新索引
    thrust::device_vector<unsigned int> d_new_indices(num_vertices);
    thrust::exclusive_scan(d_is_kept.begin(), d_is_kept.end(), d_new_indices.begin());

    // 获取去重后的总点数
    unsigned int total_kept = 0;
    if ( num_vertices > 0 ) {
        total_kept = d_is_kept.back() + d_new_indices.back();
    }

    // 7. 生成去重后的点集和映射表
    thrust::device_vector<float2> d_out_vertices(total_kept);
    thrust::device_vector<unsigned int> d_out_map(num_vertices);

    generate_dedup_results_kernel<<<blocks, 256>>>(
        thrust::raw_pointer_cast(d_vertices.data()),
        thrust::raw_pointer_cast(d_map.data()),
        thrust::raw_pointer_cast(d_new_indices.data()),
        thrust::raw_pointer_cast(d_out_vertices.data()),
        thrust::raw_pointer_cast(d_out_map.data()),
        num_vertices);
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. 拷贝结果回 Host
    out_point_data.resize(total_kept * 2);
    thrust::copy(d_out_vertices.begin(), d_out_vertices.end(),
        reinterpret_cast<float2*>(out_point_data.data()));

    out_map.resize(num_vertices);
    thrust::copy(d_out_map.begin(), d_out_map.end(), out_map.begin());
}
