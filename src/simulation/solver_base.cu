#include "solver_base.cuh"

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>


#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "geometric_operator.cuh"
#include "common/cuda_utils.h"
#include "dynamics/bending.cuh"

void SolverBase::init() {
    sort_and_generate_edge_lookup();
    generate_inverse_matrix();
    generate_vertex_object();
    init_triangle_index();

    vertices_world.resize(vertices_2D.size());
    edge_lengths.resize(edges.size());
    thrust::transform(thrust::device, edges.begin(), edges.end(), edge_lengths.begin(), [
            vertices = thrust::raw_pointer_cast(vertices_2D.data())
        ]__device__ (const int2 edge_index) {
            return norm((vertices[edge_index.x] - vertices[edge_index.y]));
        });
    params.cloth_edge_mean_length = thrust::reduce(edge_lengths.begin(),
        edge_lengths.begin() + params.nb_all_cloth_edges, 0.0, thrust::plus<double>()) / params.nb_all_cloth_edges;


    vertices_old.resize(params.nb_all_cloth_vertices);
    velocities.assign(params.nb_all_cloth_vertices, make_float3(0.0f, 0.0f, 0.0f));
    vertices_mask.assign(params.nb_all_cloth_vertices, static_cast<char>(0));
    // for test 
    thrust::transform(thrust::device, thrust::make_counting_iterator(0), thrust::make_counting_iterator(1),
        vertices_mask.begin(), []
        __device__ (int i) {
            if ( i == 0 ) {
                return char(1);
            }
            return char(0);
        });
    // thrust::host_vector<char> vertices_mask_ =vertices_mask;
    // std::vector vertices_mask__(vertices_mask_.begin(), vertices_mask_.end());
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


void SolverBase::init_triangle_index() {
    e2t.assign(params.nb_all_edges, make_int2(-1, -1));
    edge_opposite_points.assign(params.nb_all_cloth_edges, make_int2(-1, -1));
    Dms.resize(params.nb_all_triangles);
    triangle_indices.resize(params.nb_all_triangles);
    thrust::for_each_n(thrust::device, thrust::make_counting_iterator(0), params.nb_all_triangles,
        [
            vertices = thrust::raw_pointer_cast(vertices_2D.data()), // float3
            triangles = thrust::raw_pointer_cast(triangles.data()), // int3
            edges = thrust::raw_pointer_cast(edges.data()), // int2
            dir_edges = thrust::raw_pointer_cast(dir_edges.data()), // int2
            edge_lookup = thrust::raw_pointer_cast(edge_lookup.data()), // int2
            e2t = thrust::raw_pointer_cast(e2t.data()), // int2
            indices = thrust::raw_pointer_cast(triangle_indices.data()), // int3
            Dms = thrust::raw_pointer_cast(Dms.data()), // Mat2
            nb_all_edges=params.nb_all_edges,
            nb_all_cloth_triangles=params.nb_all_cloth_triangles
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
__global__ void record_pick_triangle(
    int i,
    int mesh_idx, int tri_idx, float3 pos,
    Mat3* __restrict__ pick_triangle_offsets,
    thrust::pair<int, float3>* __restrict__ pick_triangles,
    const float3* __restrict__ vertices,
    const int3* __restrict__ indices,
    const int* __restrict__ triangle_index_offsets
) {
    int tri_index = tri_idx + triangle_index_offsets[mesh_idx];
    pick_triangles[i] = thrust::make_pair(tri_index, pos);
    auto [v0,v1,v2] = indices[tri_index];
    Mat3 offsets{ vertices[v0] - pos, vertices[v1] - pos, vertices[v2] - pos };
    pick_triangle_offsets[i] = offsets;
}
int SolverBase::add_pick_triangle(int mesh_index, int tri_index, float3 position) {
    cudaDeviceSynchronize();
    pick_triangles.push_back({});
    pick_triangle_offsets.push_back({});
    int index = (int)pick_triangles.size() - 1;
    record_pick_triangle<<<1,1>>>(index,
        mesh_index, tri_index, position,
        thrust::raw_pointer_cast(pick_triangle_offsets.data()),
        thrust::raw_pointer_cast(pick_triangles.data()),
        thrust::raw_pointer_cast(vertices_world.data()),
        thrust::raw_pointer_cast(triangle_indices.data()),
        thrust::raw_pointer_cast(triangle_index_offsets.data())
        );
    cudaDeviceSynchronize();
    return index;
}
void SolverBase::update_pick_triangle(int index, float3 position) {
    cudaDeviceSynchronize();
    pick_triangles[index].second = position;
    cudaDeviceSynchronize();
}
void SolverBase::remove_pick_triangle(int index) {
    cudaDeviceSynchronize();
    if ( index >= (int)pick_triangles.size() || index < 0 ) return;
    pick_triangles[index].first = -1;
    cudaDeviceSynchronize();
    for ( const auto& pick_triangle : pick_triangles ) {
        if ( pick_triangle.first != -1 ) {
            return;
        }
    }
    clear_pick_triangle();
}
void SolverBase::clear_pick_triangle() {
    cudaDeviceSynchronize();
    pickers.clear();
    pick_triangles.clear();
    pick_triangle_offsets.clear();
    cudaDeviceSynchronize();
}

// static __device__ __forceinline__
// float3 mul_homo_vec(const Mat4 m, const float3 v) {
// 	const float4 v_ = m * make_float4(v.x, v.y, v.z, 0.f);
// 	return make_float3(v_.x, v_.y, v_.z);
// }


static __global__ void update_begin(
    float3* __restrict__ vertices_world,
    const float3* __restrict__ vertices,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        vertices_world[i] = mul_homo(world_matrices[obj], vertices[i]);
    }
}

static __global__ void update_substep_end(
    float3* __restrict__ vertices_world,
    float3* __restrict__ velocities,
    const float3* __restrict__ forces,
    const char* __restrict__ vertices_mask,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto x = vertices_world[i];
        if ( !vertices_mask[i] ) {
            float mass = 1.f;
            auto v = velocities[i];
            v = v + forces[i] / mass * h;
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
    const float3* __restrict__ vertices_world,
    float3* __restrict__ vertices,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices_inv,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        vertices[i] = mul_homo(world_matrices_inv[obj], vertices_world[i]);
    }
}

#include "dynamics/planar.cuh"
#include "dynamics/bending.cuh"



void SolverBase::update(float dt) {
    // Explicit Euler
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;

    this->dt = dt;
    int threadsPerBlock = 256;
    bool has_pick_triangles = (int)pick_triangles.size() > 0;


    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    update_begin<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(vertices_world.data()),
        thrust::raw_pointer_cast(vertices_local.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        thrust::raw_pointer_cast(world_matrices.data()),
        n
        );

    float step_h = 0.0001f;
    for ( int substep = 0; dt > 0.f; substep++, dt -= step_h ) {
        if ( substep > 1000 ) break;
        float h = step_h > dt ? dt : step_h;
        // update substep...
        const auto gravity = make_float3(0.f, 0.f, -9.8f);
        forces.assign(params.nb_all_cloth_vertices, gravity);
        n = params.nb_all_cloth_edges;
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        compute_spring_forces<<<blocksPerGrid, threadsPerBlock>>>(nullptr, nullptr,
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(vertices_world.data()),
            // thrust::raw_pointer_cast(vertices_obj.data()),
            thrust::raw_pointer_cast(edges.data()),
            thrust::raw_pointer_cast(edge_lengths.data()),
            n);
        compute_dihedral_bending_Fizt<<<blocksPerGrid, threadsPerBlock>>>(
            nullptr, nullptr, nullptr,
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(vertices_world.data()),
            thrust::raw_pointer_cast(edges.data()),
            nullptr,
            nullptr,
            thrust::raw_pointer_cast(edge_opposite_points.data()),
            n, 0.5);
        // update substep end
        check_update_pick();
        n = params.nb_all_cloth_vertices;
        blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
        update_substep_end<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(vertices_world.data()),
            thrust::raw_pointer_cast(velocities.data()),
            thrust::raw_pointer_cast(forces.data()),
            thrust::raw_pointer_cast(vertices_mask.data()),
            h, n);

    }
    n = params.nb_all_cloth_vertices;
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    update_end<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(vertices_world.data()),
        thrust::raw_pointer_cast(vertices_local.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        thrust::raw_pointer_cast(world_matrices_inv.data()),
        n
        );
    reset_pick_mask();
    cudaDeviceSynchronize();
}
int SolverBase::add_picker(float3 position) {
    cudaDeviceSynchronize();
    int ptindex = add_pick_triangle(0, -2, position);
    pickers.push_back(ptindex);
    return (int)pickers.size() - 1;
}
void SolverBase::update_picker(int index, float3 position) {
    cudaDeviceSynchronize();
    if ( pickers[index] != -1 ) {
        update_pick_triangle(pickers[index], position);
    }
    cudaDeviceSynchronize();
}
void SolverBase::remove_picker(int index) {
    cudaDeviceSynchronize();
    if ( index >= (int)pickers.size() || index < 0 ) return;
    remove_pick_triangle(pickers[index]);
    pickers[index] = -1;
    cudaDeviceSynchronize();
    for ( int ptindex : pickers ) {
        if ( ptindex != -1 ) return;
    }
    clear_picker();
}
void SolverBase::clear_picker() {
    cudaDeviceSynchronize();
    pickers.clear();
    cudaDeviceSynchronize();
}
