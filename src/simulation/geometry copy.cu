#include "geometry.cuh"
#include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>


// #include <filesystem>

#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
// #include <thrust/transform_reduce.h>

#include <cub/device/device_segmented_reduce.cuh>

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "common/cuda_utils.h"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"

    

template<typename T, typename otherT=T>
static void copy_to_device(const std::vector<T>& data, thrust::device_vector<otherT>& dst) {
    const size_t size = data.size() * sizeof(T) / sizeof(otherT);
    auto first = reinterpret_cast<const otherT*>(data.data());
    if ( size > 0 ) {
        dst.resize(size);
        CUDA_CHECK(cudaMemcpy(
            thrust::raw_pointer_cast(dst.data()),
            first,
            size * sizeof(otherT),
            cudaMemcpyHostToDevice
        ));
    }
    else {
        dst.resize(0);
    }
}

void Geometry::init(const GeoDataInput& geo) {

    params.nb_all_objects = (int)geo.obj_data_input.size();
    params.nb_all_vertices = (int)geo.vertices.size() / 3;
    params.nb_all_edges = (int)geo.edges.size() / 2;
    params.nb_all_triangles = (int)geo.triangles.size() / 3;
    params.nb_all_cloth_triangles = geo.nb_all_cloth_f / 3;
    params.nb_all_cloth_edges = geo.nb_all_cloth_e / 2;
    params.nb_all_cloth_vertices = geo.nb_all_cloth_v / 3;
    params.nb_all_stitches = (int)geo.stitches.size();

    copy_to_device<float, float3>(geo.vertices, pos_2D);
    copy_to_device<float, float3>(geo.normals, normals_input);
    copy_to_device<int, int2>(geo.edges, edges);
    copy_to_device<int, int3>(geo.triangles, triangles);
    copy_to_device(geo.object_types, object_types);
    copy_to_device(geo.obj_data_input, obj_data);
    copy_to_device(geo.world_matrices, world_matrices);
    copy_to_device(geo.sewings, sewing_lines);
    copy_to_device(geo.stitches, stitches);
    copy_to_device(geo.vertex_index_offsets, vertex_index_offsets);
    copy_to_device(geo.edge_index_offsets, edge_index_offsets);
    copy_to_device(geo.triangle_index_offsets, triangle_index_offsets);
    copy_to_device(geo.pin_fixed, pin_fixed);
    copy_to_device(geo.pin_attached, pin_attached);
    // vertices data for simulation,
    copy_to_device<float, float3>(geo.vertices_sim, pos_local);
    check_cuda(cudaDeviceSynchronize());

    pos_world.resize(params.nb_all_vertices);
    pos_inertia.resize(params.nb_all_vertices);
    edge_lengths.resize(params.nb_all_edges);
    static_diags.resize(params.nb_all_vertices);

    edge_normals.resize(params.nb_all_edges);
    vertex_normals.resize(params.nb_all_vertices);

    pos_old.resize(params.nb_all_vertices);
    pos_local_new_frame.resize(params.nb_all_vertices);
    pos_new.resize(params.nb_all_vertices);
    debug_colors.resize(params.nb_all_vertices);
    velocities.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    forces.resize(params.nb_all_vertices);
    vertices_mask.assign(params.nb_all_vertices, static_cast<char>(0));
    mass_inv.resize(params.nb_all_vertices);

    temp_vertices_f3.resize(params.nb_all_vertices);

    init_vertex_data();
    // sort_and_generate_edge_lookup
    init_edge_data();

    init_triangle_data();

    // precompute bending
    IBM_q.assign(params.nb_all_cloth_edges, make_float4(0.0f, 0.0f, 0.0f, 0.f));
    int threadsPerBlock = 256;
    int n = params.nb_all_cloth_edges;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    precompute_IBM_Q<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(IBM_q.data()),
        thrust::raw_pointer_cast(edges.data()),
        thrust::raw_pointer_cast(e2t.data()),
        thrust::raw_pointer_cast(pos_2D.data()),
        thrust::raw_pointer_cast(edge_opposite_points.data()),
        thrust::raw_pointer_cast(Dms.data()),
        n
        );

    need_update_inv_mass = true;
    has_upload_vertices_this_frame = false;
    has_upload_world_matrices_this_frame = true;
    has_pick_triangles_this_frame = false;

    m_contact.init();
    
    init_pin();
    init_picker();
    init_sewing();
}

static __global__ void transform_to_world(
    float3* __restrict__ vertices_world,
    const float3* __restrict__ vertices_local,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        vertices_world[i] = mul_homo(world_matrices[obj], vertices_local[i]);
    }
}
void Geometry::init_vertex_data() {
    // generate_vertex_object;
    vertices_obj.resize(params.nb_all_vertices);
    thrust::upper_bound(thrust::device,
        vertex_index_offsets.begin(), vertex_index_offsets.end(),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(params.nb_all_vertices),
        vertices_obj.begin()
        );
    thrust::transform(
        vertices_obj.begin(),
        vertices_obj.end(),
        vertices_obj.begin(),
        thrust::placeholders::_1 - 1
        );

    int block = 256;
    transform_to_world<<<(params.nb_all_vertices + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(pos_world.data()),
        thrust::raw_pointer_cast(pos_local.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        thrust::raw_pointer_cast(world_matrices.data()),
        params.nb_all_vertices
        );

    // generate_matrices_inv
    world_matrices_inv.resize(world_matrices.size());
    thrust::transform(world_matrices.begin(), world_matrices.end(), world_matrices_inv.begin(),
        [] __device__ (const Mat4 mat) {
            return mat.inverse();
        });

}
void Geometry::init_edge_data() {
    int nb_all_v = params.nb_all_vertices;
    size_t num_edges = edges.size();
    size_t num_dir_edges = num_edges * 2;

    // -------------------------------------------------------
    // 步骤 1: 准备排序键 (Keys) 和 值 (Values)
    // -------------------------------------------------------
    // Keys: 存储打包后的 (Source << 32 | Target)
    // Values: 存储 EdgeID (对应 Python 的 np.arange)
    thrust::device_vector<unsigned long long> sort_keys(num_dir_edges);
    thrust::device_vector<int> sort_values(num_dir_edges); // 存 EdgeID

    // 使用 transform 填充数据
    // 线程 i 处理一条输入边，同时生成“正向”和“反向”两条数据
    thrust::for_each(thrust::device,
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_edges),
        [
            in_ptr = thrust::raw_pointer_cast(edges.data()),
            keys_ptr = thrust::raw_pointer_cast(sort_keys.data()),
            vals_ptr = thrust::raw_pointer_cast(sort_values.data()),
            num_edges
        ] __device__ (size_t i) {
            int2 e = in_ptr[i];
            if ( e.x > e.y ) {
                e = make_int2(e.y, e.x);
                in_ptr[i] = e;
            }
            int u = e.x;
            int v = e.y;

            // 正向边: u -> v, ID = i
            // Key 高位是 u (Primary sort), 低位是 v (Secondary sort)
            keys_ptr[i] = ((unsigned long long)u << 32) | (unsigned int)v;
            vals_ptr[i] = (int)i;

            // 反向边: v -> u, ID = i
            keys_ptr[i + num_edges] = ((unsigned long long)v << 32) | (unsigned int)u;
            vals_ptr[i + num_edges] = (int)i;
        }
        );

    // -------------------------------------------------------
    // 步骤 2: 排序 (完全等价于 np.lexsort)
    // -------------------------------------------------------
    // 对 64位整数排序比对 struct 排序快得多
    thrust::stable_sort_by_key(thrust::device, sort_keys.begin(), sort_keys.end(), sort_values.begin());

    // -------------------------------------------------------
    // 步骤 3: 生成 dir_edges (解包)
    // -------------------------------------------------------
    // Python 返回的是 dir_edges[:, 1:]，即 [Target, EdgeID]
    // 我们的 sort_keys 低 32 位正是 Target，sort_values 正是 EdgeID
    dir_edges.resize(num_dir_edges);

    thrust::transform(
        thrust::make_zip_iterator(thrust::make_tuple(sort_keys.begin(), sort_values.begin())),
        thrust::make_zip_iterator(thrust::make_tuple(sort_keys.end(), sort_values.end())),
        dir_edges.begin(),
        [] __device__ (const thrust::tuple<unsigned long long, int>& t) {
            // 解包：Target 在低32位
            int target = (int)(thrust::get<0>(t) & 0xFFFFFFFF);
            int edge_id = thrust::get<1>(t);
            return make_int2(target, edge_id);
        }
        );

    // -------------------------------------------------------
    // 步骤 4: 生成 Lookup Table (CSR 格式)
    // -------------------------------------------------------
    // 我们需要 Source 数组来计算 offset。Source 就在 sort_keys 的高32位。

    // 提取 Source 序列 (为了 lower_bound)
    // 注意：这里不用显式分配大数组，用 transform_iterator 包装即可，节省显存
    auto source_iter_begin = thrust::make_transform_iterator(
        sort_keys.begin(),
        [] __host__ __device__ (unsigned long long key) { return (int)(key >> 32); }
        );
    auto source_iter_end = source_iter_begin + (int)num_dir_edges;

    // 准备查询序列 [0, 1, ..., nb_all_v - 1]
    thrust::device_vector<int> query_vertices(nb_all_v);
    thrust::sequence(thrust::device, query_vertices.begin(), query_vertices.end());

    // 计算 Offsets (Lower Bound) 和 Ends (Upper Bound)
    thrust::device_vector<int> offsets(nb_all_v);
    thrust::device_vector<int> counts(nb_all_v); // 暂存 counts，最后合并

    thrust::lower_bound(thrust::device,
        source_iter_begin, source_iter_end,
        query_vertices.begin(), query_vertices.end(),
        offsets.begin()
        );

    thrust::upper_bound(thrust::device,
        source_iter_begin, source_iter_end,
        query_vertices.begin(), query_vertices.end(),
        counts.begin() // 这里先存 end index，下一步做减法
        );

    // 合并结果到 lookup_table: .x = offset, .y = count
    edge_lookup.resize(nb_all_v);
    thrust::transform(
        offsets.begin(), offsets.end(),
        counts.begin(), // 此时里面存的是 upper_bound 的结果
        edge_lookup.begin(),
        [] __device__ (int start, int end) {
            return make_int2(start, end - start);
        }
        );
    calc_edge_length();
}

static __device__ int get_opposite_point(const int2& edge,const int3& tri, const int2* edges) {
    int v0 = edges[tri.x].x, v1 = edges[tri.x].y, v2 = edges[tri.y].y;
    if ( v0 != edge.x && v0 != edge.y ) return v0;
    if ( v1 != edge.x && v1 != edge.y ) return v1;
    return v2;
}
__global__ void mean_kernel(
    const float* __restrict__ d_sums,
    const int* __restrict__ d_offsets,
    ObjectDataInput* __restrict__ d_objects,
    int num_objects
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_objects ) return;

    int count = d_offsets[i + 1] - d_offsets[i];
    if ( count > 0 && d_objects[i].granularity < 1e-6f ) {
        d_objects[i].granularity = d_sums[i] / (float)count;
    }
}
void Geometry::calc_edge_length() {
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_edges,
        [
            vertices2D = thrust::raw_pointer_cast(pos_2D.data()),
            vertices3D = thrust::raw_pointer_cast(pos_world.data()),
            edges = thrust::raw_pointer_cast(edges.data()),
            edge_lengths = thrust::raw_pointer_cast(edge_lengths.data()),
            nb_all_cloth_edges = params.nb_all_cloth_edges
        ] __device__ (const int edge_index) {
            int2 e_i = edges[edge_index];
            if ( edge_index < nb_all_cloth_edges )
                edge_lengths[edge_index] = norm(vertices2D[e_i.x] - vertices2D[e_i.y]);
            else
                edge_lengths[edge_index] = norm(vertices3D[e_i.x] - vertices3D[e_i.y]);
        });

    params.cloth_edge_mean_length = thrust::reduce(edge_lengths.begin(),
        edge_lengths.begin() + params.nb_all_cloth_edges, 0.0, thrust::plus<double>()) / params.nb_all_cloth_edges;
    int num_objects = params.nb_all_objects;
    thrust::device_vector<int> d_offsets(num_objects + 1);
    thrust::copy(edge_index_offsets.begin(), edge_index_offsets.end(), d_offsets.begin());
    d_offsets[num_objects] = params.nb_all_edges;
    float* d_sums;
    cudaMalloc(&d_sums, num_objects * sizeof(float));
    float* d_edge_len = thrust::raw_pointer_cast(edge_lengths.data());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceSegmentedReduce::Sum(
        d_temp_storage, temp_storage_bytes,
        d_edge_len, d_sums, num_objects,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    cub::DeviceSegmentedReduce::Sum(
        d_temp_storage, temp_storage_bytes,
        d_edge_len, d_sums, num_objects,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1);

    int threads = 256;
    int blocks = (num_objects + threads - 1) / threads;
    mean_kernel<<<blocks, threads>>>(
        d_sums,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(obj_data.data()),
        num_objects);

    cudaFree(d_sums);
    cudaFree(d_temp_storage);
}
void Geometry::init_triangle_data() {
    e2t.assign(params.nb_all_edges, make_int2(-1, -1));
    edge_opposite_points.assign(params.nb_all_edges, make_int2(-1, -1));
    Dms.resize(params.nb_all_triangles);
    // areas.assign(params.nb_all_objects,0.f);
    masses.assign(params.nb_all_vertices, 0.f);
    triangle_indices.resize(params.nb_all_triangles);
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_triangles,
        [
            vertices = thrust::raw_pointer_cast(pos_2D.data()), // float3
            normals = thrust::raw_pointer_cast(normals_input.data()), // float3
            triangles = thrust::raw_pointer_cast(triangles.data()), // int3
            edges = thrust::raw_pointer_cast(edges.data()), // int2
            dir_edges = thrust::raw_pointer_cast(dir_edges.data()), // int2
            edge_lookup = thrust::raw_pointer_cast(edge_lookup.data()), // int2
            e2t = thrust::raw_pointer_cast(e2t.data()), // int2
            indices = thrust::raw_pointer_cast(triangle_indices.data()), // int3
            Dms = thrust::raw_pointer_cast(Dms.data()), // Mat2
            nb_all_edges=params.nb_all_edges,
            nb_all_cloth_triangles=params.nb_all_cloth_triangles,
            masses = masses.data().get(),
            vertices_obj = vertices_obj.data().get(),
            // density = mass_densitys.data().get(),
            obj_data = obj_data.data().get()
        ] __device__ (const int i) {
            // 1. Load vertices of the triangle
            int3 tri_v = triangles[i];
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
            // float3 n_in = make_float3(0.0f, 0.0f, 1.0f);
            float3 n_in = normals[i];
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
            int e1_i = v2e(v0, v1, edge_lookup, dir_edges);
            int e2_i = v2e(v0, v2, edge_lookup, dir_edges);
            int e3_i = v2e(v1, v2, edge_lookup, dir_edges);
            if (e1_i == -1 || e2_i == -1 || e3_i == -1) {
                printf("[Qianyi Error] There are edges that do not exist in the triangle!\n");
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

            // 5. Calculate Material Space Matrix Dm (for cloth simulation)
            if ( i < nb_all_cloth_triangles ) {
                // Here we use the reordered vertex positions
                int2 edge1 = edges[e1_i]; // Re-fetch vertex IDs of edge
                int2 edge2 = edges[e2_i];

                float3 x0 = vertices[edge1.x];
                float3 x1 = vertices[edge1.y];
                float3 x2 = vertices[edge2.y];

                float3 e1 = x1 - x0;
                float3 e2 = x2 - x0;

                float3 normal = normalized(cross(e1, e2));
                float3 u_dir = normalized(e1);
                float3 v_dir = cross(normal, u_dir);

                // Fill Dm matrix: columns are [e1_uv, e2_uv]
                // Dm = [ l1,  e2.dot(u) ]
                //      [ 0,   e2.dot(v) ]
                Dms[i].r[0].x = norm(e1);
                Dms[i].r[0].y = dot(e2, u_dir);
                Dms[i].r[1].x = 0.0f;
                Dms[i].r[1].y = dot(e2, v_dir);
                float area = 0.5f * fabsf(Dms[i].det());
                // atomicAdd(&areas[vertices_obj[edge1.x]], area);
                float mass_per_v = area * obj_data[vertices_obj[v0]].mass_densitys / 3.f;
                atomicAdd(&masses[v0], mass_per_v);
                atomicAdd(&masses[v1], mass_per_v);
                atomicAdd(&masses[v2], mass_per_v);
            }
            else {
                masses[v0] = 1.f;
                masses[v1] = 1.f;
                masses[v2] = 1.f;
            }
        });

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_edges,
        [
            triangles = thrust::raw_pointer_cast(triangles.data()), // int3
            edges = thrust::raw_pointer_cast(edges.data()), // int2
            e2t = thrust::raw_pointer_cast(e2t.data()), // int2
            edge_opposite_points = thrust::raw_pointer_cast(edge_opposite_points.data()) // int2
        ] __device__ (const int i) {
            int2 t_adj = e2t[i];
            int2 e_i = edges[i];
            int p0_idx = t_adj.x != -1 ? get_opposite_point(e_i, triangles[t_adj.x], edges) : -1;
            int p3_idx = t_adj.y != -1 ? get_opposite_point(e_i, triangles[t_adj.y], edges) : -1;
            edge_opposite_points[i] = make_int2(p0_idx, p3_idx);
        });
}

float Geometry::get_global_parameter(const std::string& key, float default_value) const {
    return simulator->get_parameter(key, default_value);
}
void Geometry::copy_vertices(float* ptr, bool world_space = false) {
    CUDA_CHECK(cudaMemcpy(
        ptr,
        thrust::raw_pointer_cast(world_space ? pos_world.data(): pos_local.data()),
        params.nb_all_cloth_vertices * sizeof(float3),
        cudaMemcpyDeviceToHost
    ));
    // thrust::host_vector<float3> vertices_ = vertices;
    // std::vector vertices__(vertices_.begin(), vertices_.end());
}
void Geometry::copy_debug_colors(float* ptr) {
    CUDA_CHECK(cudaMemcpy(
        ptr,
        thrust::raw_pointer_cast(debug_colors.data()),
        params.nb_all_cloth_vertices * sizeof(float3),
        cudaMemcpyDeviceToHost
    ));
}

// static __device__ __forceinline__
// float3 mul_homo_vec(const Mat4 m, const float3 v) {
// 	const float4 v_ = m * make_float4(v.x, v.y, v.z, 0.f);
// 	return make_float3(v_.x, v_.y, v_.z);
// }

static __global__ void update_and_sign_world_matrix_kernel(
    ObjectDataInput* object_data, const Mat4 world_matrix, int index) {
    object_data[index].new_matrix = world_matrix;
    object_data[index].matrix_updated = true;
}
void Geometry::upload_world_matrix(int obj_index, const std::vector<float>& matrix) {
    Mat4 world_matrix;
    memcpy(&world_matrix, matrix.data(), sizeof(float) * 16);
    update_and_sign_world_matrix_kernel<<<1,1>>>(
        obj_data.data().get(), world_matrix, obj_index);
    CUDA_CHECK(cudaDeviceSynchronize());
    has_upload_world_matrices_this_frame = true;
}
static __global__ void sign_updated_local_vertices_kernel(
    ObjectDataInput* object_data, int index) {
    object_data[index].vertices_updated = true;
}
void Geometry::upload_local_vertices(int obj_index, const std::vector<float>& vertices) {
    int offset;
    CUDA_CHECK(cudaMemcpy(&offset, vertex_index_offsets.data().get() + obj_index,
        sizeof(int), cudaMemcpyDeviceToHost));
    auto* ptr = pos_local_new_frame.data().get() + offset;
    CUDA_CHECK(cudaMemcpy(ptr,vertices.data(),vertices.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    sign_updated_local_vertices_kernel<<<1,1>>>(
        obj_data.data().get(), obj_index);
    CUDA_CHECK(cudaDeviceSynchronize());
    has_upload_vertices_this_frame = true;
}
__global__ void forward_step(
    const float3* __restrict__ pos,
    const float3* __restrict__ vel,
    const float* __restrict__ inv_mass,
    const float3* __restrict__ external_force,
    const char* __restrict__ mask,
    float3* __restrict__ inertia_out,
    float3* __restrict__ pos_prev,
    float3* __restrict__ dx,
    float* __restrict__ static_diags,
    float dt,
    float3 gravity,
    int num_vertices
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i >= num_vertices ) return;
    if (pos_prev) pos_prev[i] = pos[i];

    if ( mask[i] || inv_mass[i] == 0.0f ) {
        inertia_out[i] = pos[i];
        static_diags[i] = 0.0f;
        return;
    }
    float im = inv_mass[i];
    float force_factor = dt * dt;
    static_diags[i] += 1.f / (im * force_factor);

    float3 p = pos[i];
    float3 v = vel[i];
    float3 f_ext = gravity;
    if ( external_force ) {
        f_ext += external_force[i] * im;
    }
    inertia_out[i] = p + v * dt + f_ext * force_factor;
    if (dx) {
        dx[i] = v * dt;
        // dx[i] = inertia_out[i] - p; 
    }
}
static __global__ void fill_inv_mass_kernel(
    float* __restrict__ invMass,
    const int* __restrict__ vertex_obj,
    const int* __restrict__ object_types,
    float* masses,
    char* mask,
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    if ( mask[vid] ) { // not updating points
        invMass[vid] = 0.f;
        return;
    }
    int obj = vertex_obj[vid];
    if ( object_types[obj] > 0 ) { // TODO only update cloth now
        invMass[vid] = 0.f;
        return;
    }
    invMass[vid] = 1.0f / masses[vid];
}

static __global__ void update_begin_obj(
    ObjectDataInput*__restrict__ obj_data,
    Mat4* __restrict__ world_matrices,
    Mat4* __restrict__ world_matrices_inv,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    if ( obj_data[i].matrix_updated ) {
        world_matrices[i] = obj_data[i].new_matrix;
        world_matrices_inv[i] = world_matrices[i].inverse();
    }
    obj_data[i].matrix_updated = false;
    obj_data[i].vertices_updated = false;
}

static __global__ void update_world_pos(
    float3* __restrict__ vertices_world,
    float3* __restrict__ vertices_world_old,
    float3* __restrict__ vertices,
    const ObjectDataInput*__restrict__ obj_data,
    const float3* __restrict__ vertices_new,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        auto& obj_data_input = obj_data[obj];
        vertices_world_old[i] = mul_homo(world_matrices[obj], vertices[i]);
        auto wm = obj_data_input.matrix_updated ? obj_data_input.new_matrix : world_matrices[obj];
        if ( obj_data_input.vertices_updated ) {
            vertices[i] = vertices_new[i];
        }
        vertices_world[i] = mul_homo(wm, vertices[i]);
    }
}

__global__ void compute_normals_kernel(
    const int3* __restrict__ triangles,
    const int3* __restrict__ triangle_indices,
    const float3* __restrict__ pos_world,
    float3* __restrict__ vertex_normals,
    float3* __restrict__ edge_normals,
    int numTriangles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= numTriangles ) return;

    int3 tri_verts = triangle_indices[tid];
    int3 tri_edges = triangles[tid];

    float3 v0 = pos_world[tri_verts.x];
    float3 v1 = pos_world[tri_verts.y];
    float3 v2 = pos_world[tri_verts.z];
    float3 n = cross(v1 - v0, v2 - v0); // Length = 2 × Area

    atomicAddFloat3(&vertex_normals[tri_verts.x], n);
    atomicAddFloat3(&vertex_normals[tri_verts.y], n);
    atomicAddFloat3(&vertex_normals[tri_verts.z], n);

    atomicAddFloat3(&edge_normals[tri_edges.x], n);
    atomicAddFloat3(&edge_normals[tri_edges.y], n);
    atomicAddFloat3(&edge_normals[tri_edges.z], n);
}

__global__ void normalize_vectors_kernel(float3* __restrict__ vectors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= N ) return;

    float3 v = vectors[idx];
    float len = norm(v);
    if ( len > 1e-8f ) {
        vectors[idx] = v / len;
    }
    else {
        vectors[idx] = make_float3(0.0f, 0.0f, 1.0f);
    }
}

void Geometry::compute_normals() {
    int block = 256;
    cudaMemsetAsync(edge_normals.data().get(), 0, sizeof(float3) * params.nb_all_edges);
    cudaMemsetAsync(vertex_normals.data().get(), 0, sizeof(float3) * params.nb_all_vertices);
    int tri_num = params.nb_all_triangles;
    compute_normals_kernel<<<(tri_num + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(triangles.data()),
        thrust::raw_pointer_cast(triangle_indices.data()),
        thrust::raw_pointer_cast(pos_world.data()),
        thrust::raw_pointer_cast(vertex_normals.data()),
        thrust::raw_pointer_cast(edge_normals.data()),
        tri_num
        );
    int vertex_num = params.nb_all_vertices;
    normalize_vectors_kernel<<<(vertex_num + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(vertex_normals.data()), vertex_num);
    int edge_num = params.nb_all_edges;
    normalize_vectors_kernel<<<(edge_num + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(edge_normals.data()), edge_num);
}
void Geometry::update_for_frame() {
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;
    float dt = simulator->dt;
    int block = 256;
    if ( need_update_inv_mass ) {
        fill_inv_mass_kernel<<<(n + block - 1) / block, block>>>(
            mass_inv.data().get(),
            vertices_obj.data().get(),
            object_types.data().get(),
            masses.data().get(),
            vertices_mask.data().get(), n);
        need_update_inv_mass = false;
    }
    int obj_num = params.nb_all_objects;
    if ( has_upload_world_matrices_this_frame ) {
        update_world_pos<<<(n + block - 1) / block, block>>>(
            pos_world.data().get(),
            pos_old.data().get(),
            pos_local.data().get(),
            obj_data.data().get(),
            pos_local_new_frame.data().get(),
            vertices_obj.data().get(),
            world_matrices.data().get(),
            n);
        update_begin_obj<<<(obj_num + block - 1) / block, block>>>(
            obj_data.data().get(),
            world_matrices.data().get(),
            world_matrices_inv.data().get(),
            obj_num);
    }
    compute_normals();
    has_upload_world_matrices_this_frame = false;
    check_update_pick();
    update_pin(pos_world.data().get());
    int sewing_forced_connect_frame = max(0, (int)get_global_parameter("sewing_forced_connect_frame", 80));
    check_sewing(simulator->frame > sewing_forced_connect_frame);
}

static __global__ void update_interpolated_position(
    float3* __restrict__ vertices_interpolated,
    const float3* __restrict__ vertices_world_new,
    const float3* __restrict__ vertices_world_old,
    const float factor,
    const int num_cloth_vertices,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( i < num_cloth_vertices ) return;
        vertices_interpolated[i] =
            vertices_world_old[i] * (1.f - factor) + vertices_world_new[i] * factor;
    }
}
void Geometry::update_for_step(float h, float time_factor) {
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;
    int block = 256;

    if ( need_update_inv_mass ) {
        fill_inv_mass_kernel<<<(n + block - 1) / block, block>>>(
            mass_inv.data().get(),
            vertices_obj.data().get(),
            object_types.data().get(),
            masses.data().get(),
            vertices_mask.data().get(), n);
        need_update_inv_mass = false;
    }
    if ( has_upload_vertices_this_frame ) {
        update_interpolated_position<<<(n + block - 1) / block, block>>>(
            pos_world.data().get(), pos_new.data().get(),
            pos_old.data().get(),
            time_factor, params.nb_all_cloth_vertices, n);
    }

    float gravity_z = 0.f;
    if ( sewing_done ) // TODO
        gravity_z = get_global_parameter("gravity", -9.8f);
    gravity = make_float3(0.0f, 0.0f, gravity_z);

}
static __global__ void update_local_pos_kernel(
    float3* __restrict__ vertices,
    float3* __restrict__ vertices_world,
    const int* proxy,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices_inv,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {

        int p_id = proxy[i];
        int obj = vertices_obj[i];

        vertices[i] = mul_homo(world_matrices_inv[obj], vertices_world[p_id]);
    }
}
void Geometry::end_for_frame() {
    has_upload_vertices_this_frame = false;

    int n = params.nb_all_cloth_vertices;
    if ( n <= 0 ) return;
    int block = 256;
    int blocksPerGrid = (n + block - 1) / block;

    update_local_pos_kernel<<<blocksPerGrid, block>>>(
        pos_local.data().get(),
        pos_world.data().get(),
        vertex_proxy.data().get(),
        vertices_obj.data().get(),
        world_matrices_inv.data().get(),
        n);

    int smooth_times = max(0, (int)get_global_parameter("smooth_times", 5));
    if ( !sewing_done ) { smooth_times *= 2; } // TODO 
    for ( int i = 0; i < smooth_times; ++i ) {
        laplacian_smoothing<<<n + block - 1, block>>>(
            temp_vertices_f3.data().get(),
            velocities.data().get(),
            vertices_mask.data().get(),
            edge_lookup.data().get(),
            dir_edges.data().get(),
            0.02f, n
            );
        thrust::swap(temp_vertices_f3, velocities);
    }
    reset_pick_mask();
    CUDA_CHECK(cudaDeviceSynchronize());
}
