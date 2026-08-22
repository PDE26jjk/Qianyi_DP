#include "geometry.cuh"
#include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>


// #include <filesystem>

#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
// #include <thrust/transform_reduce.h>

#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_segmented_reduce.cuh>

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "color_graph/coloring.h"
#include "common/cuda_utils.h"
#include "contact/collision.cuh"
#include "cuda_tools/cub_tools.cuh"
#include "dynamics/bending.cuh"

// Define utility macro used to call cub functions that use dynamic temporary storage
// #ifndef CALL_CUBS
// #ifdef _WIN32
// #define CALL_CUBS(func, ...) \
// CUDA_CHECK(cub::func(nullptr, temp_mem_size, __VA_ARGS__)); \
// CUDA_CHECK(cub::func(get_device_temp_memory(), temp_mem_size, __VA_ARGS__))
// #else// fdef _WIN32
// #define CALL_CUBS(func, args...) \
// CUDA_CHECK(cub::func(nullptr, temp_mem_size, args)); \
// CUDA_CHECK(cub::func(get_device_temp_memory(), temp_mem_size, args))
// #endif// ifdef _WIN32
// #endif// ifndef CALL_CUBS

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
    params.nb_all_cloth_objects = geo.nb_all_cloth_o;
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
    pos_pred.resize(params.nb_all_vertices);
    pos_inertia.resize(params.nb_all_vertices);
    edge_lengths.resize(params.nb_all_edges);
    static_diags.resize(params.nb_all_vertices);

    edge_normals.resize(params.nb_all_edges);
    vertex_normals.resize(params.nb_all_vertices);

    pos_interpolation_old.resize(params.nb_all_vertices);
    pos_step_prev.resize(params.nb_all_vertices);
    pos_local_new_frame.resize(params.nb_all_vertices);
    pos_interpolation_new.resize(params.nb_all_vertices);
    debug_colors.resize(params.nb_all_vertices);
    velocities.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    vel_prev.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    forces.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    elastic_forces.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    vertices_mask.assign(params.nb_all_vertices, static_cast<char>(0));
    mass_inv.resize(params.nb_all_vertices);

    temp_vertices_f3.resize(params.nb_all_vertices);
    temp_mem.clear();

    inertial_offset.resize(params.nb_all_vertices);

    init_vertex_data();
    // sort_and_generate_edge_lookup
    init_edge_data();

    init_triangle_data();

    // check and set physical model
    switch ( int constitutive_model_planar = (int)get_global_parameter("constitutive_model_planar", 0) ) {
    case 0:
        constitutive_model = ConstitutiveModel::SpringMass;
        break;
    case 1:
        constitutive_model = ConstitutiveModel::FEM_BW;
        break;
    default:
        std::cout << "Unsupported constitutive model " << constitutive_model_planar << ", using FEM_BW." << std::endl;
        constitutive_model = ConstitutiveModel::FEM_BW;
        break;
    }
    switch ( int bending_model_ = (int)get_global_parameter("bending_model", 0) ) {
    case 0:
        bending_model = BendingModel::IBM_quadratic;
        break;
    case 1:
        bending_model = BendingModel::DiscreteShells_GN;
        break;
    case 2:
        bending_model = BendingModel::DiscreteShells_AOGS;
        break;
    default:
        std::cout << "Unsupported bending model " << bending_model_ << ", using DiscreteShells_GN." << std::endl;
        bending_model = BendingModel::DiscreteShells_GN;
        break;
    }
    precompute_bending_factor();
    if ( bending_model == BendingModel::IBM_quadratic ) {
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
    }
    need_update_inv_mass = true;
    need_update_interpolation_vertices_this_frame = false;
    need_record_interpolation_this_frame = true;
    has_pick_triangles_this_frame = false;

    m_contact.init();

    init_pin();
    init_picker();
    init_sewing();
    CUDA_CHECK(cudaDeviceSynchronize());
}

static __global__ void precompute_dihedral_bending_factor_kernel(
    float* __restrict__ bending_factor,
    const float3* __restrict__ pos_2D,
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const float* __restrict__ edge_lengths,
    const float* __restrict__ areas,
    const int* __restrict__ vertices_obj,
    const ObjectDataInput* __restrict__ obj_data,
    const int num_edges
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;

    int2 adjacent_triangles = e2t[i];
    float area_sum = 0.0f;
    if ( adjacent_triangles.x >= 0 ) area_sum += areas[adjacent_triangles.x];
    if ( adjacent_triangles.y >= 0 ) area_sum += areas[adjacent_triangles.y];

    float edge_length = edge_lengths[i];
    if ( area_sum <= 1e-12f || edge_length <= 1e-12f ) {
        bending_factor[i] = 1.0f;
        return;
    }

    int2 edge = edges[i];
    int object_index = vertices_obj[edge.x];
    ObjectDataInput object = obj_data[object_index];

    float3 edge_vector = pos_2D[edge.y] - pos_2D[edge.x];
    float edge_norm = norm(edge_vector);
    if ( edge_norm <= 1e-12f ) {
        bending_factor[i] = 1.0f;
        return;
    }
    edge_vector /= edge_norm;

    float grain_dir = object.grain_dir;
    float3 grain_axis = make_float3(cosf(grain_dir), sinf(grain_dir), 0.0f);
    float3 cross_grain_axis = make_float3(-sinf(grain_dir), cosf(grain_dir), 0.0f);
    float longitude_projection = dot(edge_vector, grain_axis);
    float latitude_projection = dot(edge_vector, cross_grain_axis);

    float longitude_bending = object.bending.x + object.bending.z;
    float latitude_bending = object.bending.y + object.bending.z;
    float anisotropic_scale =
        longitude_projection * longitude_projection * longitude_bending +
        latitude_projection * latitude_projection * latitude_bending;
    float geometric_scale = 3.0f * edge_length * edge_length / area_sum;
    bending_factor[i] = geometric_scale * anisotropic_scale;
}

void Geometry::precompute_bending_factor() {
    bending_factor.resize(params.nb_all_edges);
    int block = 256;
    int blocks = (params.nb_all_edges + block - 1) / block;
    precompute_dihedral_bending_factor_kernel<<<blocks, block>>>(
        bending_factor.data().get(),
        pos_2D.data().get(),
        edges.data().get(),
        e2t.data().get(),
        edge_lengths.data().get(),
        areas.data().get(),
        vertices_obj.data().get(),
        obj_data.data().get(),
        params.nb_all_edges);
}

// void* Geometry::get_device_temp_memory() {
//     if ( temp_mem.size() < temp_mem_size )
//         temp_mem.resize(temp_mem_size);
//     return thrust::raw_pointer_cast(temp_mem.data());
// }

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
    thrust::upper_bound(thrust::cuda::par_nosync,
        vertex_index_offsets.begin(), vertex_index_offsets.end(),
        thrust::make_counting_iterator(0), thrust::make_counting_iterator(params.nb_all_vertices),
        vertices_obj.begin()
        );
    thrust::transform(thrust::cuda::par_nosync,
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
    thrust::transform(thrust::cuda::par_nosync, world_matrices.begin(), world_matrices.end(), world_matrices_inv.begin(),
        [] __device__ (const Mat4 mat) {
            return mat.inverse();
        });

}
void Geometry::init_edge_data() {
    int nb_all_v = params.nb_all_vertices;
    size_t num_edges = edges.size();
    size_t num_dir_edges = num_edges * 2;
    dir_edges.resize(num_dir_edges);
    edge_lookup.resize(nb_all_v);
    edges_to_csr(nb_all_v, num_edges, edges.data().get(),
        dir_edges.data().get(), edge_lookup.data().get());

    calc_edge_length();
    rest_thetas.assign(num_edges,0.f); // TODO
}

static __device__ int get_opposite_point(const int2& edge, const int3& tri, const int2* edges) {
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
    thrust::for_each_n(thrust::cuda::par_nosync, thrust::make_counting_iterator(0), params.nb_all_edges,
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

    params.cloth_edge_mean_length = thrust::reduce(thrust::device, edge_lengths.begin(),
        edge_lengths.begin() + params.nb_all_cloth_edges, 0.0, thrust::plus<double>()) / params.nb_all_cloth_edges;
    int num_objects = params.nb_all_objects;
    thrust::device_vector<int>& d_offsets = edge_index_offsets;
    thrust::device_vector<float> d_sums(num_objects);
    float* d_edge_len = thrust::raw_pointer_cast(edge_lengths.data());

    CALL_CUBS(DeviceSegmentedReduce::Sum, d_edge_len, d_sums.data().get(), num_objects,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1);

    int threads = 256;
    int blocks = (num_objects + threads - 1) / threads;
    mean_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_sums.data()),
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(obj_data.data()),
        num_objects);
    // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
    // std::vector<type> __##v(_##v.begin(), _##v.end())
    // CHECK(d_sums,float);
    // #undef CHECK
    // cudaFree(d_temp_storage);
}

void Geometry::average_mass_by_cloth() {
    int num_cloth_objects = params.nb_all_cloth_objects;
    const thrust::device_vector<int>& d_offsets = vertex_index_offsets;
    thrust::device_vector<float> d_mass_sums(num_cloth_objects);
    CALL_CUBS(DeviceSegmentedReduce::Sum, masses.data().get(), d_mass_sums.data().get(), num_cloth_objects,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1);

    thrust::for_each_n(thrust::cuda::par_nosync,
        thrust::make_counting_iterator(0), params.nb_all_cloth_vertices,
        [
            d_offsets = thrust::raw_pointer_cast(d_offsets.data()),
            mass_sums = thrust::raw_pointer_cast(d_mass_sums.data()),
            mass = thrust::raw_pointer_cast(masses.data()),
            vertices_obj = thrust::raw_pointer_cast(vertices_obj.data())
        ] __device__ (const int i) {
            int obj = vertices_obj[i];
            int count = d_offsets[obj + 1] - d_offsets[obj];
            if ( count > 0 )
                mass[i] = mass_sums[obj] / (float)count;
        });
}
void Geometry::init_triangle_data() {
    e2t.assign(params.nb_all_edges, make_int2(-1, -1));
    edge_opposite_points.assign(params.nb_all_edges, make_int2(-1, -1));
    Dms.resize(params.nb_all_triangles);
    // areas.assign(params.nb_all_objects,0.f);
    masses.assign(params.nb_all_vertices, 0.f);
    triangle_indices.resize(params.nb_all_triangles);
    areas.resize(params.nb_all_triangles);
    thrust::for_each_n(thrust::cuda::par_nosync, thrust::make_counting_iterator(0), params.nb_all_triangles,
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
            // nb_all_edges=params.nb_all_edges,
            nb_all_cloth_triangles=params.nb_all_cloth_triangles,
            masses = masses.data().get(),
            areas = areas.data().get(),
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
            if ( e1_i == -1 || e2_i == -1 || e3_i == -1 ) {
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
            // int v3 = v2;
            // float3 e3_vec = p2 - p1;
            // if ( v2 > v1 ) {
            //     e3_vec = -e3_vec;
            //     v3 = v1;
            // }
            //
            // if ( dot(cross(p0 - vertices[v3], e3_vec), n_in) > 0.0f ) {
            //     e2t[e3_i].y = i;
            // }
            // else {
            //     e2t[e3_i].x = i;
            // }
            if ( v1 < v2 ) {
                e2t[e3_i].x = i;
            }
            else {
                e2t[e3_i].y = i;
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
                areas[i] = area;
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

    thrust::for_each_n(thrust::cuda::par_nosync,
        thrust::make_counting_iterator(0), params.nb_all_edges,
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
    if ( (bool)get_global_parameter("average_mass_by_cloth", 0.f) )
        average_mass_by_cloth();

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
    need_record_interpolation_this_frame = true;
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
    need_update_interpolation_vertices_this_frame = true;
}

__global__ void forward_step(
    const float3* __restrict__ vel,
    const float3* __restrict__ vel_prev,
    const float* __restrict__ inv_mass,
    const float3* __restrict__ external_force,
    const float3* __restrict__ elastic_force,
    const char* __restrict__ mask,
    float3* __restrict__ pos,
    float3* __restrict__ pos_preds,
    float3* __restrict__ inertia_out,
    float3* __restrict__ dx,
    float* __restrict__ static_diags,
    float dt,
    float mask_stiff,
    float3 gravity,
    bool warm_start,
    int num_vertices
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i >= num_vertices ) return;

    float3 p = pos[i];
    float3 v = vel[i];
    float3 accel_ext = gravity;
    float im = inv_mass[i];
    if ( external_force ) {
        accel_ext += external_force[i] * im;
    }
    if ( mask[i] ) {
        inertia_out[i] = p;
        static_diags[i] += mask_stiff;
    }
    float3 pos_pred = pos[i];
    if ( inv_mass[i] == 0.0f ) {
        inertia_out[i] = p;
    }
    else {
        float dt2 = dt * dt;
        float3 pos_v = p + v * dt;
        inertia_out[i] = pos_v + accel_ext * dt2;
        static_diags[i] += 1.f / (im * dt2);
        if ( warm_start ) { // Warm starting from VBD paper. 
            float3 accel_prev = (v - vel_prev[i]) / dt;
            float a_factor = 0;
            float a_ext_len_sq = len_sq(accel_ext);
            if ( a_ext_len_sq > 1e-16f ) {
                a_factor = dot(accel_prev, accel_ext) / a_ext_len_sq;
                a_factor = clamp(a_factor, 0.0f, 1.0f);
            }
            pos[i] = pos_v + accel_ext * (a_factor * dt2);
        }
        pos_pred = pos_v + (accel_ext + elastic_force[i] * im) * dt2;
    }
    if ( pos_preds )
        pos_preds[i] = pos_pred;

    if ( dx ) {
        dx[i] = inertia_out[i] - p;
    }

}
static __global__ void fill_inv_mass_kernel(
    float* __restrict__ invMass,
    const int* __restrict__ vertex_obj,
    const int* __restrict__ object_types,
    const float* __restrict__ masses,
    const char* __restrict__ mask,
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
    float3* __restrict__ vertices_world_new,
    float3* __restrict__ vertices_local,
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
        auto wm = obj_data_input.matrix_updated ? obj_data_input.new_matrix : world_matrices[obj];
        if ( obj_data_input.vertices_updated ) {
            vertices_local[i] = vertices_new[i];
        }
        vertices_world_new[i] = mul_homo(wm, vertices_local[i]);
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
    float3 n_normalized = normalized(n);

    atomicAddFloat3(&vertex_normals[tri_verts.x], n);
    atomicAddFloat3(&vertex_normals[tri_verts.y], n);
    atomicAddFloat3(&vertex_normals[tri_verts.z], n);

    atomicAddFloat3(&edge_normals[tri_edges.x], n_normalized);
    atomicAddFloat3(&edge_normals[tri_edges.y], n_normalized);
    atomicAddFloat3(&edge_normals[tri_edges.z], n_normalized);
}
__global__ void compute_normals_single_edge_kernel(
    const int2* __restrict__ edges,
    const int2* __restrict__ e2t,
    const float3* __restrict__ pos_world,
    float3* __restrict__ edge_normals,
    int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_edges ) return;

    int2 tris = e2t[tid];
    if ( tris.x != -1 && tris.y != -1 ) return;
    int2 e = edges[tid];
    float3 e_ = pos_world[e.y] - pos_world[e.x];
    if ( tris.x >= 0 ) {
        edge_normals[tid] = cross(e_, edge_normals[tid]);
    }
    else {
        edge_normals[tid] = cross(edge_normals[tid], e_);
    }
}

__global__ void normalize_vectors_kernel(float3* __restrict__ vectors, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= N ) return;

    float3 v = vectors[idx];
    float len = len_sq(v);
    if ( len > 1e-16f ) {
        vectors[idx] = v * rsqrtf(len);
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
    compute_normals_single_edge_kernel<<<(edge_num + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(edges.data()),
        thrust::raw_pointer_cast(e2t.data()),
        thrust::raw_pointer_cast(pos_world.data()),
        thrust::raw_pointer_cast(edge_normals.data()),
        edge_num);
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
    if ( need_record_interpolation_this_frame ) {
        update_world_pos<<<(n + block - 1) / block, block>>>(
            pos_interpolation_new.data().get(),
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
        need_update_interpolation_vertices_this_frame = true;
    }
    need_record_interpolation_this_frame = false;

    compute_normals();
    check_update_pick();
    update_pin(pos_world.data().get());// TODO 
    if ( need_update_interpolation_vertices_this_frame ) {
        // record old pos
        cudaMemcpyAsync(pos_interpolation_old.data().get(), pos_world.data().get(),
            n * sizeof(float3), cudaMemcpyDeviceToDevice);
    }

    int sewing_forced_connect_frame = max(0, (int)get_global_parameter("sewing_forced_connect_frame", 80));
    check_sewing(simulator->frame > sewing_forced_connect_frame);
}

static __global__ void update_interpolated_position(
    float3* __restrict__ vertices_interpolated,
    const float3* __restrict__ vertices_world_new,
    const float3* __restrict__ vertices_world_old,
    const float factor,
    const char* mask,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        // constexpr char mask = static_cast<char>(MaskBit::pick_mesh_mask);
        if ( mask[i] ) {

            vertices_interpolated[i] =
                vertices_world_old[i] * (1.f - factor) + vertices_world_new[i] * factor;
            // if (mask[i] | static_cast<char>(MaskBit::pick_mesh_mask)) {
            //     
            //     printf("v_update: %f, %f, %f\n", vertices_interpolated[i].x,vertices_interpolated[i].y,vertices_interpolated[i].z);
            // }
        }
    }
}
void Geometry::update_for_step(float h, float time_factor) {
    step_h = h;
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
    cudaMemcpyAsync(pos_step_prev.data().get(), pos_world.data().get(),
        n * sizeof(float3), cudaMemcpyDeviceToDevice);
    if ( need_update_interpolation_vertices_this_frame ) {
        update_interpolated_position<<<(n + block - 1) / block, block>>>(
            pos_world.data().get(), pos_interpolation_new.data().get(),
            pos_interpolation_old.data().get(),
            time_factor, vertices_mask.data().get(), n);
    }

    float gravity_z = 0.f;
    if ( sewing_done ) // TODO
        gravity_z = get_global_parameter("gravity", -9.8f);
    gravity = make_float3(0.0f, 0.0f, gravity_z);
    ground = bool(get_global_parameter("ground", 1));
}
static __global__ void update_local_pos_kernel(
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
void Geometry::end_for_frame() {
    need_update_interpolation_vertices_this_frame = false;

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
__global__ void compute_color_offsets_kernel(
    const int* __restrict__ d_sorted_colors,
    int num_vertices,
    int num_colors,
    int* __restrict__ d_offsets) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if ( c > num_colors ) return;   // extra threads in the last block

    if ( c == num_colors ) {
        d_offsets[c] = num_vertices;
        return;
    }

    // binary search for lower_bound(c) = first index with d_sorted_colors[i] >= c
    int low = 0, high = num_vertices;
    while ( low < high ) {
        int mid = (low + high) >> 1;
        if ( d_sorted_colors[mid] < c )
            low = mid + 1;
        else
            high = mid;
    }
    d_offsets[c] = low;
}

__global__ void calc_offsets_kernel(
    const int* __restrict__ sorted_keys, // size: num_pairs + 1, last key should be num_values
    int num_pairs,
    int* __restrict__ offsets
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i > num_pairs ) return;

    int prev_key = i == 0 ? -1 : sorted_keys[i - 1];
    int key = sorted_keys[i];
    for ( int j = prev_key + 1; j <= key; j++ ) {
        offsets[j] = i;
    }
}

void Geometry::color_graph() {
    // Collect valid edges in graph
    int num_edges = params.nb_all_cloth_edges;
    int num_nodes = params.nb_all_cloth_vertices;
    int num_stitches = params.nb_all_stitches;
    // edges and bending
    thrust::device_vector<int2> valid_edges(num_edges * 2 + num_stitches);
    node_colors.assign(params.nb_all_vertices, -1);
    int2* d_valid_edges = thrust::raw_pointer_cast(valid_edges.data());
    cudaMemcpyAsync(d_valid_edges, edges.data().get(), num_edges * sizeof(int2), cudaMemcpyDeviceToDevice);
    auto end = thrust::copy_if(thrust::device, edge_opposite_points.begin(), edge_opposite_points.begin() + num_edges,
        valid_edges.begin() + num_edges, [] __device__ (int2 e) {
            return e.x != -1 && e.y != -1;
        });
    int offset = end - valid_edges.begin();
    // stitches
    if ( num_stitches > 0 )
        cudaMemcpyAsync(d_valid_edges + offset, stitches.data().get(), num_stitches * sizeof(int2), cudaMemcpyDeviceToDevice);

    int num_colors = graph_coloring_cuda(num_nodes, offset + num_stitches,
        d_valid_edges, node_colors.data().get(), true);

    // Build color‑based partition of vertices (sorted order + CSR‑style offsets)
    color_groups.resize(num_nodes);
    thrust::sequence(color_groups.begin(), color_groups.end());
    thrust::device_vector<int> d_keys(num_nodes + 1, num_colors);
    CALL_CUBS(DeviceRadixSort::SortPairs, node_colors.data().get(), d_keys.data().get(),
        color_groups.data().get(), color_groups.data().get(), num_nodes);

    colors_index_offsets.resize(num_colors + 1);
    int threads = 256;
    int blocks = (num_nodes + 1 + threads - 1) / threads;
    calc_offsets_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_keys.data()), num_nodes,
        thrust::raw_pointer_cast(colors_index_offsets.data()));
    h_colors_index_offsets.resize(colors_index_offsets.size());
    cudaMemcpy(h_colors_index_offsets.data(), colors_index_offsets.data().get(), colors_index_offsets.size() * sizeof(int),
        cudaMemcpyDeviceToHost);

}
__global__ void bending_fill_keys_and_values(
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    int num_edges,
    int* __restrict__ d_keys,
    int2* __restrict__ d_values
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_edges ) return;

    int2 op = edge_opposite_points[tid];
    if ( op.x == -1 || op.y == -1 ) {
        return;
    }
    int2 e = edges[tid];

    int base = tid * 4;
    d_keys[base] = e.x;
    d_keys[base + 1] = e.y;
    d_keys[base + 2] = op.x;
    d_keys[base + 3] = op.y;
    d_values[base] = make_int2(tid, 0);
    d_values[base + 1] = make_int2(tid, 1);
    d_values[base + 2] = make_int2(tid, 2);
    d_values[base + 3] = make_int2(tid, 3);
}
__global__ void tris_fill_keys_and_values(
    const int3* __restrict__ tris,
    int num_tris,
    int* __restrict__ d_keys,
    int2* __restrict__ d_values
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= num_tris ) return;

    int3 tri = tris[tid];
    int base = tid * 3;

    d_keys[base] = tri.x;
    d_keys[base + 1] = tri.y;
    d_keys[base + 2] = tri.z;
    d_values[base] = make_int2(tid, 0);
    d_values[base + 1] = make_int2(tid, 1);
    d_values[base + 2] = make_int2(tid, 2);
}
void Geometry::build_adj_data() {
    // bending 
    int num_edges = params.nb_all_cloth_edges;
    int num_vertices = params.nb_all_cloth_vertices;
    int num_pairs = num_edges * 4;
    thrust::device_vector<int> keys(num_pairs + 1, num_vertices);
    v_adj_bending.resize(num_pairs);
    v_adj_bending_offsets.resize(num_vertices + 1);
    int* d_keys = keys.data().get();
    int2* d_values = v_adj_bending.data().get();
    int blockSize = 256;
    int gridSize = (num_edges + blockSize - 1) / blockSize;
    bending_fill_keys_and_values<<<gridSize, blockSize>>>(
        edges.data().get(), edge_opposite_points.data().get(),
        num_edges, d_keys, d_values);
    CALL_CUBS(DeviceRadixSort::SortPairs, d_keys, d_keys, d_values, d_values, num_pairs);
    calc_offsets_kernel<<< (num_pairs + 1 + blockSize - 1) / blockSize, blockSize>>>(
        d_keys, num_pairs, v_adj_bending_offsets.data().get());

    // triangle FEM
    int num_faces = params.nb_all_cloth_triangles;
    num_pairs = num_faces * 3;
    keys.assign(num_pairs + 1, num_vertices);
    v_adj_tris.resize(num_pairs);
    v_adj_tris_offsets.resize(num_vertices + 1);
    d_values = v_adj_tris.data().get();
    tris_fill_keys_and_values<<<gridSize, blockSize>>>(
        triangle_indices.data().get(), num_faces, d_keys, d_values);
    CALL_CUBS(DeviceRadixSort::SortPairs, d_keys, d_keys, d_values, d_values, num_pairs);
    calc_offsets_kernel<<< (num_pairs + 1 + blockSize - 1) / blockSize, blockSize>>>(
        d_keys, num_pairs, v_adj_tris_offsets.data().get());

}
