#include "solver_base.cuh"

#include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>


// #include <filesystem>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "common/cuda_utils.h"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"


void SolverBase::init() {
    sort_and_generate_edge_lookup();
    generate_inverse_matrix();
    generate_vertex_object();
    init_triangle_data();

    vertices_world.resize(params.nb_all_vertices);
    edge_lengths.resize(edges.size());
    thrust::transform(thrust::device, edges.begin(), edges.end(), edge_lengths.begin(), [
            vertices = thrust::raw_pointer_cast(vertices_2D.data())
        ]__device__ (const int2 edge_index) {
            return norm((vertices[edge_index.x] - vertices[edge_index.y]));
        });
    params.cloth_edge_mean_length = thrust::reduce(edge_lengths.begin(),
        edge_lengths.begin() + params.nb_all_cloth_edges, 0.0, thrust::plus<double>()) / params.nb_all_cloth_edges;


    vertices_old.resize(params.nb_all_vertices);
    vertices_local_new_frame.resize(params.nb_all_vertices);
    vertices_new.resize(params.nb_all_vertices);
    debug_colors.resize(params.nb_all_vertices);
    velocities.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    forces.resize(params.nb_all_vertices);
    vertices_mask.assign(params.nb_all_vertices, static_cast<char>(0));
    init_pin();


    IBM_q.assign(params.nb_all_cloth_edges, make_float4(0.0f, 0.0f, 0.0f, 0.f));
    int threadsPerBlock = 256;
    int n = params.nb_all_cloth_edges;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    precompute_IBM_Q<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(IBM_q.data()),
        thrust::raw_pointer_cast(edges.data()),
        thrust::raw_pointer_cast(e2t.data()),
        thrust::raw_pointer_cast(vertices_2D.data()),
        thrust::raw_pointer_cast(edge_opposite_points.data()),
        thrust::raw_pointer_cast(Dms.data()),
        n
        );
    init_picker();
    init_collision();
    init_sewing();
    if ( capture_stream != nullptr ) {
        cudaStreamDestroy(capture_stream);
        capture_stream = nullptr;
    }
    cudaStreamCreate(&capture_stream);
    pool.resize(1024);
    pool_used.assign(1024, false);
    frame = 0;
}
void SolverBase::sort_and_generate_edge_lookup() {
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
}

void SolverBase::generate_inverse_matrix() {
    world_matrices_inv.resize(world_matrices.size());
    thrust::transform(world_matrices.begin(), world_matrices.end(), world_matrices_inv.begin(),
        [] __device__ (const Mat4 mat) {
            return mat.inverse();
        });
}

void SolverBase::generate_vertex_object() {
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
}

static __device__ int get_opposite_point(int2 edge, int3 tri, const int2* edges) {
    int v0 = edges[tri.x].x, v1 = edges[tri.x].y, v2 = edges[tri.y].y;
    if ( v0 != edge.x && v0 != edge.y ) return v0;
    if ( v1 != edge.x && v1 != edge.y ) return v1;
    return v2;
}


void SolverBase::init_triangle_data() {
    e2t.assign(params.nb_all_edges, make_int2(-1, -1));
    edge_opposite_points.assign(params.nb_all_cloth_edges, make_int2(-1, -1));
    Dms.resize(params.nb_all_triangles);
    // areas.assign(params.nb_all_objects,0.f);
    masses.assign(params.nb_all_cloth_vertices, 0.f);
    triangle_indices.resize(params.nb_all_triangles);
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_triangles,
        [
            vertices = thrust::raw_pointer_cast(vertices_2D.data()), // float3
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
        });

    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_cloth_edges,
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
void SolverBase::copy_vertices(float* ptr, bool world_space = false) {
    CUDA_CHECK(cudaMemcpy(
        ptr,
        thrust::raw_pointer_cast(world_space ? vertices_world.data(): vertices_local.data()),
        params.nb_all_cloth_vertices * sizeof(float3),
        cudaMemcpyDeviceToHost
    ));
    // thrust::host_vector<float3> vertices_ = vertices;
    // std::vector vertices__(vertices_.begin(), vertices_.end());
}
void SolverBase::copy_debug_colors(float* ptr) {
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


static __global__ void update_begin(
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

static __global__ void update_substep_end(
    float3* __restrict__ vertices_world,
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const char* __restrict__ vertices_mask,
    const float* __restrict__ masses,
    const float h,
    const float max_velocity,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto x = vertices_world[i];
        if ( !vertices_mask[i] ) {
            float mass = masses[i];
            auto v = velocities[i];
            float3 force = forces[i] + make_float3(0.f, 0.f, -9.8f * mass);
            v = v + force / mass * h;
            // float max_velocity = 40.f;
            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            v = v * expf(-h * 0.5f);
            x = x + v * h;
            velocities[i] = v;
            // printf("%f mask:%d\n",x.z,i);
        }
        else {
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
        vertices_world[i] = x;
    }
}
static __global__ void update_end(
    float3* __restrict__ vertices,
    const float3* __restrict__ vertices_world,
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

#include "dynamics/planar.cuh"
#include "dynamics/bending.cuh"



void SolverBase::update(float dt) {
    // Explicit Euler
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;

    this->dt = dt;
    int block = 256;

    int blocksPerGrid = (n + block - 1) / block;
    update_begin<<<blocksPerGrid, block>>>(
        vertices_world.data().get(),
        vertices_old.data().get(),
        vertices_local.data().get(),
        obj_data.data().get(),
        vertices_local_new_frame.data().get(),
        vertices_obj.data().get(),
        world_matrices.data().get(),
        n);
    update_pin(vertices_world.data().get());
    cudaMemcpyAsync(vertices_new.data().get(), vertices_world.data().get(),
        params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    int obj_num = params.nb_all_objects;
    update_begin_obj<<<(obj_num + block - 1) / block, block>>>(
        obj_data.data().get(),
        world_matrices.data().get(),
        world_matrices_inv.data().get(),
        obj_num);

    int sewing_forced_connect_frame = max(1, (int)get_global_parameter("sewing_forced_connect_frame",80));
    check_sewing(frame > sewing_forced_connect_frame);
    fill_inv_mass<<<(n + block - 1) / block, block>>>(
        mass_inv.data().get(),
        vertices_obj.data().get(),
        object_types.data().get(),
        masses.data().get(),
        vertices_mask.data().get(), n);
    float step_h = max(0.000001f, get_global_parameter("step_h",0.0001f));
    float3* q = vertices_world.data().get();
    float max_dist = params.cloth_edge_mean_length;
    int update_pick_substeps = max(1, (int)get_global_parameter("update_pick_substeps",10));
    float IPC_k = max(0.f, get_global_parameter("IPC_k",1500.f));
    float sewing_k = max(0.f, get_global_parameter("sewing_k",2e3));
    float max_vel = max(0.f, get_global_parameter("max_vel",1000));
    int update_collision_substeps = max(1, (int)get_global_parameter("update_collision_substeps",20));
    // int LCP_substeps = max(1, (int)get_global_parameter("LCP_substeps",20));
    bool collision_collect_ee = get_global_parameter("collision_collect_ee", 1.f) > 0;
    bool collision_collect_tp = get_global_parameter("collision_collect_tp", 1.f) > 0;
    float dt_rest = dt;

    for ( int substep = 0; dt_rest > 0.f; substep++ ) {
        if ( substep > 10000 ) break;
        float h = step_h > dt_rest ? dt_rest : step_h;
        if ( substep % update_collision_substeps == 0 ) {
            float factor = clamp(1.f - (dt_rest / dt) + 0.1f, 0., 1.f);
            update_interpolated_position<<<blocksPerGrid, block>>>(
                q, vertices_new.data().get(),
                vertices_old.data().get(),
                factor, params.nb_all_cloth_vertices, n);
            collision_collect_near_pairs(q, max_dist, true, true, collision_collect_tp, collision_collect_ee);
        }
        // update substep...
        // const auto gravity = make_float3(0.f, 0.f, -9.8f);
        // forces.assign(params.nb_all_cloth_vertices, make_float3(0.f, 0.f, 0.f));
        cudaMemsetAsync(forces.data().get(), 0, params.nb_all_cloth_vertices * sizeof(float3));

        n = pp_result_size_h;
        // compute_collision_penalty_force_point_point<<<(n + block - 1) / block, block>>>(
        //     nullptr, nullptr,
        //     forces.data().get(),
        //     velocities.data().get(),
        //     pp_collision_result.data().get(),
        //     q, max_dist, h, n);

        int num_constraints = pp_result_size_h + tp_result_size_h + ee_result_size_h;
        if ( num_constraints > 0 ) {
            cudaMemsetAsync(weight.data().get(), 0, params.nb_all_cloth_vertices * sizeof(float));
            compute_normal_constraint_IPC_force<<<(num_constraints + block - 1) / block, block>>>(
                forces.data().get(), weight.data().get(), normal_constraints.data().get(),
                q, mass_inv.data().get(), obj_data.data().get(), vertices_obj.data().get(),
                IPC_k, num_constraints);
            n = params.nb_all_cloth_edges;
            apply_weight_force<<<(n + block - 1) / block, block>>>(
                forces.data().get(), weight.data().get(), n);
        }

        n = params.nb_all_cloth_edges;
        // compute_spring_forces<<<(n + block - 1) / block, block>>>(nullptr, nullptr,
        //     forces.data().get(), nullptr,
        //     q,
        //     thrust::raw_pointer_cast(edges.data()),
        //     thrust::raw_pointer_cast(edge_lengths.data()),
        //     n);
        n = params.nb_all_cloth_triangles;
        // compute_ARAP_FEM<<<(n + block - 1) / block, block>>>(
        //     nullptr, nullptr,
        //     forces.data().get(), nullptr,
        //     vertices_world.data().get(),
        //     triangles.data().get(),
        //     edges.data().get(),
        //     vertices_obj.data().get(),
        //     nullptr,
        //     Dms.data().get(),
        //     n);
        compute_BW_FEM<<<(n + block - 1) / block, block>>>(
            nullptr, nullptr,
            forces.data().get(), nullptr,
            vertices_world.data().get(),
            triangles.data().get(),
            edges.data().get(),
            vertices_obj.data().get(),
            nullptr,
            Dms.data().get(),
            n);
        // compute_dihedral_bending_Fizt<<<blocksPerGrid, block>>>(
        //     nullptr, nullptr, nullptr,
        //     thrust::raw_pointer_cast(forces.data()),
        //     thrust::raw_pointer_cast(vertices_world.data()),
        //     thrust::raw_pointer_cast(edges.data()),
        //     nullptr,
        //     nullptr,
        //     thrust::raw_pointer_cast(edge_opposite_points.data()),
        //     n, 0.2);
        n = params.nb_all_cloth_edges;
        compute_quadratic_Bending_IBM<<< (n + block - 1) / block, block>>>(
            nullptr, nullptr, nullptr,
            forces.data().get(), nullptr,
            IBM_q.data().get(),
            q,
            edges.data().get(),
            e2t.data().get(),
            triangles.data().get(),
            edge_opposite_points.data().get(),
            n, 0.2f);
        if ( !sewing_done ) {
            float min_dist = 2e-3f;
            n = params.nb_all_stitches;
            compute_stitch_constraint<<<(n + block - 1) / block, block>>>(
                nullptr, nullptr, forces.data().get(),
                nullptr,
                q, vertices_obj.data().get(), obj_data.data().get(),
                vertices_mask.data().get(), stitches.data().get(), min_dist, sewing_k, n);
        }
        // update substep end

        n = params.nb_all_cloth_vertices;
        if ( substep % update_pick_substeps == 0 ) {
            check_update_pick();
        }
        update_substep_end<<<(n + block - 1) / block, block>>>(
            thrust::raw_pointer_cast(vertices_world.data()),
            thrust::raw_pointer_cast(velocities.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(vertices_mask.data()),
            thrust::raw_pointer_cast(masses.data()),
            h, max_vel, n);
        // if ( substep % LCP_substeps == 0 ) {
        //     collision_LCP_postprocess_unified(vertices_world.data().get());
        // }
        dt_rest -= step_h;
    }
    n = params.nb_all_cloth_vertices;
    blocksPerGrid = (n + block - 1) / block;
    update_end<<<blocksPerGrid, block>>>(
        vertices_local.data().get(),
        vertices_world.data().get(),
        vertex_proxy.data().get(),
        vertices_obj.data().get(),
        world_matrices_inv.data().get(),
        n);
    int smooth_times = max(0, (int)get_global_parameter("smooth_times",5));
    if ( !sewing_done ) { smooth_times *= 2; }
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
    frame++;
}
static __global__ void update_world_matrix_kernel(
    ObjectDataInput* object_data, const Mat4 world_matrix, int index) {
    object_data[index].new_matrix = world_matrix;
    object_data[index].matrix_updated = true;
}
void SolverBase::update_world_matrix(int obj_index, const std::vector<float>& matrix) {
    Mat4 world_matrix;
    memcpy(&world_matrix, matrix.data(), sizeof(float) * 16);
    update_world_matrix_kernel<<<1,1>>>(
        obj_data.data().get(), world_matrix, obj_index);
    CUDA_CHECK(cudaDeviceSynchronize());
}
static __global__ void update_local_vertices_kernel(
    ObjectDataInput* object_data, int index) {
    object_data[index].vertices_updated = true;
}
void SolverBase::update_local_vertices(int obj_index, const std::vector<float>& vertices) {
    int offset;
    CUDA_CHECK(cudaMemcpy(&offset, vertex_index_offsets.data().get() + obj_index,
        sizeof(int), cudaMemcpyDeviceToHost));
    auto* ptr = vertices_local_new_frame.data().get() + offset;
    CUDA_CHECK(cudaMemcpy(ptr,vertices.data(),vertices.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    update_local_vertices_kernel<<<1,1>>>(
        obj_data.data().get(), obj_index);
    CUDA_CHECK(cudaDeviceSynchronize());
}


AutoGPUmem SolverBase::alloc_pool() {
    for ( size_t i = 0; i < pool_used.size(); ++i )
        if ( !pool_used[i] ) {
            pool_used[i] = true;
            return AutoGPUmem{ this,
                thrust::raw_pointer_cast(&pool[i]) };
        }
    throw std::exception("No pool available");
    // pool.resize(pool.size() * 2);
    // pool_used.resize(pool.size(), false);
    // pool_used[pool.size() / 2] = true;
    // return AutoGPUmem{ this,
    //     thrust::raw_pointer_cast(&pool[pool.size() / 2]) };
}
void SolverBase::dealloc_pool(void* p) {
    size_t idx = reinterpret_cast<int*>(p) - thrust::raw_pointer_cast(&pool[0]);
    if ( idx < pool.size() ) pool_used[idx] = false;
}

AutoGPUmem::~AutoGPUmem() { pool->dealloc_pool(ptr); }
struct DotProductFunctor {
    __device__ __forceinline__
    float operator()(const thrust::tuple<const float3&, const float3&>& t) const {
        const float3& va = thrust::get<0>(t);
        const float3& vb = thrust::get<1>(t);
        return va.x * vb.x + va.y * vb.y + va.z * vb.z;
    }
};

float SolverBase::vector_field_dot(const float3* a, const float3* b) {
    int n = params.nb_all_cloth_vertices;
    // int threadsPerBlock = 256;
    // int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    //
    // vector_field_dot_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //     thrust::raw_pointer_cast(temp1.data()),
    //     a, b, n);
    // float res = thrust::reduce(temp1.begin(), temp1.begin() + n, 0.f);
    // return res;
    auto a_begin = thrust::device_pointer_cast(a);
    auto a_end = thrust::device_pointer_cast(a + n);
    auto b_begin = thrust::device_pointer_cast(b);

    return thrust::transform_reduce(
        thrust::make_zip_iterator(thrust::make_tuple(a_begin, b_begin)),
        thrust::make_zip_iterator(thrust::make_tuple(a_end, b_begin)),
        DotProductFunctor{},
        0.0f,
        thrust::plus<float>());
}

float SolverBase::get_global_parameter(const std::string& key, float default_value) const {
    return simulator->get_parameter(key, default_value);
}
