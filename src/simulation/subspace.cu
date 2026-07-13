#include "geometry.cuh"
#include <thrust/execution_policy.h>
// #include <thrust/host_vector.h>


#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/binary_search.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "solver_PDNewton.cuh"
#include "common/atomic_utils.cuh"
#include "cuda_tools/cub_tools.cuh"
// #include <thrust/transform_reduce.h>
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

struct f3Min {
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return fmin3(a, b);
    }
};
struct f3Max {
    __device__ __forceinline__
    float3 operator()(const float3& a, const float3& b) const {
        return fmax3(a, b);
    }
};

static __device__ float N_bspline(float x) {
    float absx = fabsf(x);
    if ( absx < 0.5f ) return 0.75f - x * x;
    else if ( absx < 1.5f ) {
        float t = 1.5f - absx;
        return 0.5f * t * t;
    }
    else return 0.0f;
}
static __global__ void calc_pattern_basis_size_kernel(
    int2* grid_res,
    int* basis_counts,
    const float3* bound_mins,
    const float3* bound_maxs,
    const float grid_spacing,
    int num_objs
) {
    int oid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( oid >= num_objs ) return;
    auto range = bound_maxs[oid] - bound_mins[oid] + grid_spacing * 3.f;
    int nu = (int)ceilf(range.x / grid_spacing);
    int nv = (int)ceilf(range.y / grid_spacing);
    grid_res[oid] = make_int2(nu, nv);
    basis_counts[oid] = nu * nv;
}
static __global__ void build_basis_kernel(
    int* __restrict__ basis_indices,
    int* __restrict__ basis_exist,
    float*__restrict__ basis_weights,
    const float3*__restrict__ pos_2D,
    const int*__restrict__ vertices_obj,
    const float3*__restrict__ bound_mins,
    const int2* __restrict__ grid_res,
    const int* __restrict__ basis_offsets,
    int num_vertices,
    float grid_spacing
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    int obj = vertices_obj[vid];
    auto bmin = bound_mins[obj] - 1.5f * grid_spacing;
    auto pos = pos_2D[vid];
    float u = pos.x;
    float v = pos.y;
    float fu = (u - bmin.x) / grid_spacing;
    float fv = (v - bmin.y) / grid_spacing;
    int iu = (int)floorf(fu);
    int iv = (int)floorf(fv);
    int2 res = grid_res[obj];
    int nu = res.x, nv = res.y;
    int max_size = nu * nv;

    int vertex_offset = 9 * vid;
    int basis_offset = basis_offsets[obj];
    int idx = 0;
    for ( int du = -1; du <= 1; ++du ) {
        for ( int dv = -1; dv <= 1; ++dv ) {
            int ci = iu + du;
            int cj = iv + dv;
            int base_idx = cj * nu + ci;
            if (base_idx < 0 || base_idx >= max_size ) {
                // printf("%d/%d base_idx is out of bounds!!!!!!!!!!!!!\n",base_idx,max_size);
                basis_weights[vertex_offset + idx] = 0.f;
            }
            else {
                float wu = N_bspline(fu - ci);
                float wv = N_bspline(fv - cj);
                float w = wu * wv;

                base_idx += basis_offset;
                basis_exist[base_idx] = 1;
                basis_indices[vertex_offset + idx] = base_idx;
                basis_weights[vertex_offset + idx] = w;
            }
            idx++;
        }
    }
}
static __global__ void to_square_kernel(int* __restrict__ y, const int* __restrict__ x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    y[i] = x[i] * x[i];
}
static __global__ void basis_to_new_index_kernel(
    int* __restrict__ basis_indices,
    int* __restrict__ basis_new_indices,
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    int vertex_offset = 9 * vid;
    for ( int i = 0; i < 9; ++i ) {
        basis_indices[vertex_offset + i] =
            basis_new_indices[basis_indices[vertex_offset + i]];
    }
}

static __global__ void find_H2y_index_kernel(
    int2*__restrict__ H2y,
    const int*__restrict__ basis_index_offsets,
    const int*__restrict__ H_red_offsets,
    int num_objects,
    int H_red_total_sizes
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= H_red_total_sizes ) return;
    int lo = 0, hi = num_objects - 1;
    int obj = -1;
    while ( lo <= hi ) {
        int mid = (lo + hi) >> 1;
        if ( i >= H_red_offsets[mid] && i < H_red_offsets[mid + 1] ) {
            obj = mid;
            break;
        }
        else if ( i < H_red_offsets[mid] ) {
            hi = mid - 1;
        }
        else {
            lo = mid + 1;
        }
    }
    if ( obj == -1 ) {
        printf("Something bad happened in find_H2y_index_kernel!!!\n");
        return;
    }

    int M = basis_index_offsets[obj + 1] - basis_index_offsets[obj];
    int offset = H_red_offsets[obj];
    int local_id = i - offset;
    int local_i = local_id / M;
    int local_j = local_id % M;
    H2y[i].x = basis_index_offsets[obj] + local_i;
    H2y[i].y = basis_index_offsets[obj] + local_j;
}
void Geometry::init_subspace() {

    int num_objects = params.nb_all_cloth_objects;
    int num_vertices = params.nb_all_cloth_vertices;
    thrust::device_vector<int>& d_offsets = vertex_index_offsets;
    thrust::device_vector<float3> d_bound_mins(num_objects);
    thrust::device_vector<float3> d_bound_maxs(num_objects);
    float3* pos = thrust::raw_pointer_cast(pos_2D.data());

    CALL_CUBS(DeviceSegmentedReduce::Reduce, pos, d_bound_mins.data().get(), num_objects,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1, f3Min(), make_float3(FLT_MAX,FLT_MAX,FLT_MAX));
    CALL_CUBS(DeviceSegmentedReduce::Reduce, pos, d_bound_maxs.data().get(), num_objects,
        thrust::raw_pointer_cast(d_offsets.data()),
        thrust::raw_pointer_cast(d_offsets.data()) + 1, f3Max(), make_float3(-FLT_MAX,-FLT_MAX,-FLT_MAX));

    // TODO Convex set segmentation?
    int block = 32;
    float grid_spacing = 0.08f;
    thrust::device_vector<int> d_basis_counts(num_objects);
    thrust::device_vector<int2> d_grid_res(num_objects);
    calc_pattern_basis_size_kernel<<<(num_objects + block - 1) / block, block>>>(
        d_grid_res.data().get(), d_basis_counts.data().get(),
        d_bound_mins.data().get(), d_bound_maxs.data().get(), grid_spacing, num_objects
        );
    thrust::device_vector<int> d_basis_offsets(num_objects + 1);
    CALL_CUBS(DeviceScan::ExclusiveScan, d_basis_counts.data().get(),
        d_basis_offsets.data().get(), thrust::plus<int>(), 0, num_objects);
    int total_bases = d_basis_offsets[num_objects - 1] + d_basis_counts.back();
    d_basis_offsets[num_objects] = total_bases;

    basis_indices.resize(9 * num_vertices);
    basis_weights.assign(9 * num_vertices, 0.f);
    thrust::device_vector<int> d_basis_exist(total_bases);
    d_basis_exist.assign(total_bases, 0);

    block = 256;
    build_basis_kernel<<<(num_vertices + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(basis_indices.data()),
        thrust::raw_pointer_cast(d_basis_exist.data()),
        thrust::raw_pointer_cast(basis_weights.data()),
        thrust::raw_pointer_cast(pos_2D.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        thrust::raw_pointer_cast(d_bound_mins.data()),
        thrust::raw_pointer_cast(d_grid_res.data()),
        thrust::raw_pointer_cast(d_basis_offsets.data()),
        num_vertices,
        grid_spacing
        );

    thrust::device_vector<int> d_new_indices(total_bases);
    CALL_CUBS(DeviceScan::ExclusiveScan, d_basis_exist.data().get(),
        d_new_indices.data().get(), thrust::plus<int>(), 0, total_bases);

    basis_size = d_new_indices.back() + d_basis_exist.back();
    basis_to_new_index_kernel<<<(num_vertices + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(basis_indices.data()),
        thrust::raw_pointer_cast(d_new_indices.data()),
        num_vertices);

    // for H_red
    thrust::device_vector<int> H_red_sizes; // save by obj
    H_red_sizes.resize(num_objects);
    basis_index_offsets.resize(num_objects + 1);
    CALL_CUBS(DeviceSegmentedReduce::Sum, d_basis_exist.data().get(),
        H_red_sizes.data().get(), num_objects,
        thrust::raw_pointer_cast(d_basis_offsets.data()),
        thrust::raw_pointer_cast(d_basis_offsets.data()) + 1);
    CALL_CUBS(DeviceScan::ExclusiveScan, H_red_sizes.data().get(),
        basis_index_offsets.data().get(), thrust::plus<int>(), 0, num_objects);
    basis_index_offsets[num_objects] = basis_size;

    thrust::device_vector<int> H_red_sizes_sq(num_objects);
    block = 16;
    to_square_kernel<<<(num_objects + block - 1) / block, block>>>(
        H_red_sizes_sq.data().get(), H_red_sizes.data().get(), num_objects);
    H_red_offsets.resize(num_objects + 1);
    CALL_CUBS(DeviceScan::ExclusiveScan, H_red_sizes_sq.data().get(),
        H_red_offsets.data().get(), thrust::plus<int>(), 0, num_objects);
    H_red_total_sizes = H_red_offsets[num_objects - 1] + H_red_sizes_sq.back();
    H_red_offsets[num_objects] = H_red_total_sizes;
    H2y.resize(H_red_total_sizes);
    block = 256;
    find_H2y_index_kernel<<<(H_red_total_sizes + block - 1) / block, block>>>(
        H2y.data().get(), basis_index_offsets.data().get(), H_red_offsets.data().get(), num_objects, H_red_total_sizes);

    // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
    // std::vector<type> __##v(_##v.begin(), _##v.end())
    // CHECK(H2y,int2);
    // CHECK(basis_index_offsets,int);
    // CHECK(H_red_offsets,int);
    // #undef CHECK
}
__global__ void prolongate_kernel(
    float3*__restrict__ x,
    const float3*__restrict__ y,
    const int*__restrict__ basis_indices,
    const float*__restrict__ basis_weights,
    int num_vertices) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    float3 d = make_float3(0, 0, 0);
    int vertex_offset = 9 * vid;
    for ( int k = 0; k < 9; ++k ) {
        float w = basis_weights[vertex_offset + k];
        if ( w <= 0.f ) continue;
        int bidx = basis_indices[vertex_offset + k];
        float3 val = y[bidx];
        d += val * w;
    }
    x[vid] = d;
}
__global__ void restrict_kernel(
    float3*__restrict__ y,
    const float3*__restrict__ x,
    const int*__restrict__ basis_indices,
    const float*__restrict__ basis_weights,
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    int vertex_offset = 9 * vid;
    float3 data = x[vid];
    for ( int k = 0; k < 9; ++k ) {
        float w = basis_weights[vertex_offset + k];
        if ( w <= 0.f ) continue;
        int bidx = basis_indices[vertex_offset + k];
        atomicAddFloat3(&y[bidx], data * w);
    }
}
__global__ void reduce_diag_kernel_per_obj(
    float* H_red,
    const int* H_red_offsets,
    const float* H_diag,
    const int* basis_index_offsets,
    const int* basis_indices,
    const float* basis_weights,
    const int* vertices_obj,
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;

    int obj = vertices_obj[vid];
    int offset = H_red_offsets[obj];
    int basis_offset = basis_index_offsets[obj];
    int M_obj = basis_index_offsets[obj + 1] - basis_offset;

    auto D = H_diag[vid];
    int off = 9 * vid;

    int idx[9];
    float w[9];
    for ( int k = 0; k < 9; ++k ) {
        idx[k] = basis_indices[off + k] - basis_offset;
        w[k] = basis_weights[off + k];
    }
    for ( int a = 0; a < 9; ++a ) {
        int ia = idx[a];
        float wa = w[a];
        if ( wa <= 0.0f ) continue;
        for ( int b = 0; b < 9; ++b ) {
            int ib = idx[b];
            float wb = w[b];
            if ( wb <= 0.0f ) continue;
            atomicAdd(&H_red[offset + ia * M_obj + ib], D * wa * wb);
        }
    }
}
__global__ void reduce_edge_kernel_per_obj(
    float* H_red,
    const int* H_red_offsets,
    const int2* edges,
    const float* H_edge,
    const int* basis_index_offsets,
    const int* basis_indices,
    const float* basis_weights,
    const int* vertices_obj,
    int num_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( eid >= num_edges ) return;

    int i = edges[eid].x;
    int j = edges[eid].y;
    int obj = vertices_obj[i];
    int offset = H_red_offsets[obj];
    int basis_offset = basis_index_offsets[obj];
    int M_obj = basis_index_offsets[obj + 1] - basis_offset;

    auto C = H_edge[eid];
    auto CT = C;

    int off_i = 9 * i;
    int off_j = 9 * j;
    int idx_i[9], idx_j[9];
    float w_i[9], w_j[9];
    for ( int k = 0; k < 9; ++k ) {
        idx_i[k] = basis_indices[off_i + k] - basis_offset;
        w_i[k] = basis_weights[off_i + k];
        idx_j[k] = basis_indices[off_j + k] - basis_offset;
        w_j[k] = basis_weights[off_j + k];
    }

    for ( int a = 0; a < 9; ++a ) {
        int ia = idx_i[a];
        float wa = w_i[a];
        if ( wa == 0.0f ) continue;
        for ( int b = 0; b < 9; ++b ) {
            int ib = idx_j[b];
            float wb = w_j[b];
            if ( wb == 0.0f ) continue;
            float scale = wa * wb;
            atomicAdd(&H_red[offset + ia * M_obj + ib], C * scale);
            atomicAdd(&H_red[offset + ib * M_obj + ia], CT * scale);
        }
    }
}
void Geometry::precompute_subspace_H(const float* Jx_diag_pd, const float* Jx_nondiag_pd) {
    int block = 256;
    int num_vertices = params.nb_all_cloth_vertices;
    int num_edges = params.nb_all_cloth_edges;
    H_red.assign(H_red_total_sizes, 0.0f);
    reduce_diag_kernel_per_obj<<<(num_vertices + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(H_red.data()),
        thrust::raw_pointer_cast(H_red_offsets.data()),
        Jx_diag_pd,
        thrust::raw_pointer_cast(basis_index_offsets.data()),
        thrust::raw_pointer_cast(basis_indices.data()),
        thrust::raw_pointer_cast(basis_weights.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        num_vertices
        );

    reduce_edge_kernel_per_obj<<<(num_edges + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(H_red.data()),
        thrust::raw_pointer_cast(H_red_offsets.data()),
        thrust::raw_pointer_cast(edges.data()),
        Jx_nondiag_pd,
        thrust::raw_pointer_cast(basis_index_offsets.data()),
        thrust::raw_pointer_cast(basis_indices.data()),
        thrust::raw_pointer_cast(basis_weights.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        num_edges
        );
    M_red.assign(H_red_total_sizes, 0.0f);
    reduce_diag_kernel_per_obj<<<(num_vertices + block - 1) / block, block>>>(
        thrust::raw_pointer_cast(M_red.data()),
        thrust::raw_pointer_cast(H_red_offsets.data()),
        masses.data().get(),
        thrust::raw_pointer_cast(basis_index_offsets.data()),
        thrust::raw_pointer_cast(basis_indices.data()),
        thrust::raw_pointer_cast(basis_weights.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        num_vertices
        );
}

__global__ void apply_H_red_kernel(
    float3* res,
    const float3* y,
    const float* H_red,
    const float* M_red,
    const int2* H2y,
    const float h,
    int total_elements
) {
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( bid >= total_elements ) return;
    auto index = H2y[bid];
    auto H = H_red[bid] + M_red[bid] / (h * h);
    if (H != 0.0f) {
        float3 v = y[index.x] * H;
        atomicAddFloat3(&res[index.y], v);
    }
}

void SolverPDNewton::solve_subspace(float3* dx, const float3* rhs) {
    auto* geo = simulator->get_geo();
    int num_bases = geo->basis_size;
    int num_vertices = geo->params.nb_all_cloth_vertices;
    auto* rhs_red = subspace_rhs.data().get();
    cudaMemset(rhs_red, 0, num_bases * sizeof(float3));
    auto* basis_indices = geo->basis_indices.data().get();
    auto* basis_weights = geo->basis_weights.data().get();
    int block = 256;
    // rhs_red = P^T * rhs
    restrict_kernel<<<(num_vertices + block - 1) / block, block>>>(
        rhs_red, rhs, basis_indices, basis_weights, num_vertices);
    // dy = P^T * dx_input
    auto* dy = subspace_dy.data().get();
    cudaMemset(dy, 0, num_bases * sizeof(float3));
    restrict_kernel<<<(num_vertices + block - 1) / block, block>>>(
        dy, dx, basis_indices, basis_weights, num_vertices);

    // solve H_red * dy = rhs_red 
    subspace_solver->solve(dy, rhs_red,10);

    // dx = P * dy
    prolongate_kernel<<<(num_vertices + block - 1) / block, block>>>(
        dx, dy, basis_indices, basis_weights, num_vertices);
}
void SolverSubspace::init(int diag_size, int edge_size) {
    SolverPCG::init(diag_size, edge_size, false);
}
// #include <cusolverDn.h>
void SolverSubspace::A_mult_x(float3* dst, const float3* src) {
    int block = 256;
    auto* geo = simulator->get_geo();
    int num_H_red_total = geo->H_red_total_sizes;
    cudaMemsetAsync(dst, 0, sizeof(float3) * m_diag_size);
    apply_H_red_kernel<<<(num_H_red_total + block - 1) / block, block>>>(
        dst, src,
        thrust::raw_pointer_cast(geo->H_red.data()),
        thrust::raw_pointer_cast(geo->M_red.data()),
        thrust::raw_pointer_cast(geo->H2y.data()),
        geo->step_h,
        num_H_red_total);
    // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
    // std::vector<type> __##v(_##v.begin(), _##v.end())
    // {
    //     CHECK(Ad, float3);
    // }
    // #undef CHECK
    // cusolverDnDpotrf 
}
