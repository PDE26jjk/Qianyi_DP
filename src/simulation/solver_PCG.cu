#include "solver_PCG.cuh"

#include "common/atomic_utils.cuh"

#include <cub/cub.cuh>
#include <thrust/transform_reduce.h>
#include <iostream>

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "contact/collision.cuh"

void SolverPCG::init() {
    SolverBase::init();
    Jx.resize(params.nb_all_cloth_edges);
    Jx_diag.resize(params.nb_all_cloth_vertices);
    Jx_bend_cross.resize(params.nb_all_cloth_edges);
    Ad.resize(params.nb_all_cloth_vertices);
    d.resize(params.nb_all_cloth_vertices);
    r.resize(params.nb_all_cloth_vertices);
    // temp1.resize(params.nb_all_cloth_vertices);
    z.resize(params.nb_all_cloth_vertices);
}

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

__global__ void Jx_mult_x_diag_kernel(
    float3* __restrict__ res,
    const Mat3* __restrict__ Jx_diag,
    const float3* __restrict__ x,
    int n // vertices size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        res[i] = Jx_diag[i] * x[i];
    }
}


__global__ void A_mul_x_offdiag_kernel(
    float3* __restrict__ res,
    const Mat3* __restrict__ Jx,
    const Mat3* __restrict__ Jx_diag,
    const Mat3* __restrict__ Jx_bend_cross,
    const float3* __restrict__ x,
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    int n // edge size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto [v0_i,v1_i] = edges[i];

        atomicAddFloat3(&res[v0_i], Jx[i] * x[v1_i]);
        atomicAddFloat3(&res[v1_i], Jx[i].transpose() * x[v0_i]);

        auto p_op = edge_opposite_points[i];
        if ( p_op.x != -1 && p_op.y != -1 ) {
            atomicAddFloat3(&res[p_op.x], Jx_bend_cross[i] * x[p_op.y]);
            atomicAddFloat3(&res[p_op.y], Jx_bend_cross[i].transpose() * x[p_op.x]);
        }
    }
}

void SolverPCG::A_mult_x(
    float3* __restrict__ dst,
    const float3* __restrict__ src
) {
    int threadsPerBlock = 256;
    int n = params.nb_all_cloth_vertices;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    Jx_mult_x_diag_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        dst,
        Jx_diag.data().get(),
        src,
        n);

    n = params.nb_all_cloth_edges;
    blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    A_mul_x_offdiag_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        dst,
        Jx.data().get(),
        Jx_diag.data().get(),
        Jx_bend_cross.data().get(),
        src,
        edges.data().get(),
        edge_opposite_points.data().get(),
        n);

}



__global__ void vector_field_dot_kernel(
    float* __restrict__ res,
    const float3* a,
    const float3* b,
    int n // size of res
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        res[i] = dot(a[i], b[i]);
    }
}



// float SolverPCG::vector_field_dot_cub(const float3* a, const float3* b) {
//     // int n = params.nb_all_cloth_vertices;
//
//     // void* d_temp_storage = nullptr;
//     // size_t temp_storage_bytes = 0;
//     float result = 0.f;
//     //
//     // cub::DeviceReduce::Sum(
//     //     d_temp_storage, temp_storage_bytes,
//     //     nullptr, nullptr, n);
//     //
//     // cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     //
//     // struct DotProduct {
//     //     const float3* a;
//     //     const float3* b;
//     //
//     //     __host__ __device__ __forceinline__
//     //     float operator()(const int& i) const {
//     //         return a[i].x * b[i].x + a[i].y * b[i].y + a[i].z * b[i].z;
//     //     }
//     // };
//     //
//     // cub::CountingInputIterator<int> counting_iter(0);
//     // DotProduct dot_op{ a, b };
//     //
//     // cub::TransformInputIterator<float, DotProduct,
//     //     cub::CountingInputIterator<int>> input_itr(counting_iter, dot_op);
//     //
//     // cub::DeviceReduce::Sum(
//     //     d_temp_storage, temp_storage_bytes,
//     //     input_itr, &result, n);
//     //
//     // cudaFree(d_temp_storage);
//     return result;
// }

template<bool UsePreprocessingDiag>
__global__ void before_ite_kernel(
    float3* r,
    float3* b,
    float3* __restrict__ d,
    float3* __restrict__ z,
    const Mat3* __restrict__ Jx_diag,
    const float3* __restrict__ Ax,
    int n // size of r
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        r[i] = b[i] - Ax[i];
        if constexpr ( UsePreprocessingDiag ) {
            z[i] = Jx_diag[i].inverse() * r[i];
            d[i] = z[i];
        }
        else {
            d[i] = r[i];
        }
    }
}

__global__ void preprocessing(
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ b,
    const float3* __restrict__ forces,
    const float3* __restrict__ velocities,
    const char* __restrict__ mask,
    const float* __restrict__ masses,
    float h,
    int n // vertex size
) {
    const float coeff = -h * h;
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        float m = mask[i] ? 1e6f : masses[i];
        float3 force = forces[i] + make_float3(0.f, 0.f, -9.8f * masses[i]);
        b[i] = h * force + m * velocities[i];
        auto diag = Mat3::identity(m) + Jx_diag[i] * coeff;
        Jx_diag[i] = diag;
    }
}
__global__ void preprocessing_offdiag(
    Mat3* __restrict__ Jx,
    Mat3* __restrict__ Jx_bend_cross,
    float h,
    int n // edge size
) {
    const float coeff = -h * h;
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        Jx[i] = Jx[i] * coeff;
        Jx_bend_cross[i] = Jx_bend_cross[i] * coeff;
    }
}

template<bool UsePreprocessingDiag>
__global__ void ite_kernel1(
    float3* __restrict__ r,
    float3* __restrict__ x,
    float3* __restrict__ z,
    const float3* __restrict__ d,
    const float3* __restrict__ Ad,
    const Mat3* __restrict__ Jx_diag,
    float alpha,
    int n // size of r
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        x[i] = x[i] + alpha * d[i];
        r[i] = r[i] - alpha * Ad[i];
        if constexpr ( UsePreprocessingDiag ) {
            z[i] = Jx_diag[i].inverse() * r[i];
        }
    }
}
__global__ void ite_kernel2(
    float3* __restrict__ d,
    const float3* __restrict__ r,
    float beta,
    int n // size of r
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        d[i] = r[i] + beta * d[i];
    }
}

static __global__ void update_dynamics_end(
    float3* __restrict__ vertices_world,
    float3* __restrict__ velocities,
    const char* __restrict__ vertices_mask,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        auto x = vertices_world[i];
        if ( !vertices_mask[i] ) {
            auto v = velocities[i];
            v = v * expf(-h * 0.5f);
            constexpr float max_velocity = 20.f;
            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            x = x + v * h;
            velocities[i] = v;
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
    const int* __restrict__ vertices_proxy,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices_inv,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int idx = vertices_proxy[i];
        float3 x = vertices_world[idx];
        int obj = vertices_obj[i];
        vertices[i] = mul_homo(world_matrices_inv[obj], x);
    }
}
#include "dynamics/planar.cuh"
#include "dynamics/bending.cuh"



void SolverPCG::compute_constraint(float dt) {
    int block = 256;
    // const auto gravity = make_float3(0.f, 0.f, -9.8f);
    forces.assign(params.nb_all_cloth_vertices, make_float3(0.f, 0.f, 0.f));
    Jx_diag.assign(params.nb_all_cloth_vertices, Mat3::zero());
    Jx.assign(params.nb_all_cloth_edges, Mat3::zero());
    Jx_bend_cross.assign(params.nb_all_cloth_edges, Mat3::zero());

    Mat3* Jx_diag = this->Jx_diag.data().get();
    Mat3* Jx = this->Jx.data().get();
    Mat3* Jx_bend_cross = this->Jx_bend_cross.data().get();

    int n = params.nb_all_cloth_edges;
    int blocksPerGrid = (n + block - 1) / block;
    // compute_spring_forces<<<blocksPerGrid, block>>>(
    //     Jx, Jx_diag,
    //     forces.data().get(), nullptr,
    //     vertices_world.data().get(),
    //     edges.data().get(),
    //     edge_lengths.data().get(),
    //     n);
    n = params.nb_all_cloth_triangles;
    float3* y = vertices_world.data().get();
    compute_BW_FEM<<<(n + block - 1) / block, block>>>(
        // compute_ARAP_FEM<<<(n + block - 1) / block, block>>>(
        Jx, Jx_diag,
        forces.data().get(), nullptr,
        y,
        triangles.data().get(),
        edges.data().get(),
        vertices_obj.data().get(),
        nullptr,
        Dms.data().get(),
        n);
    n = params.nb_all_cloth_edges;
    float bending_k = max(0.f, get_global_parameter("bending_k", 1.f));
    compute_dihedral_bending_Fizt<<<(n + block - 1) / block, block>>>(
        Jx,Jx_diag,Jx_bend_cross,
        forces.data().get(),
        vertices_world.data().get(),
        edges.data().get(),
        e2t.data().get(),
        triangles.data().get(),
        edge_opposite_points.data().get(),
        n, bending_k);
    // compute_quadratic_Bending_IBM<<<(n + block - 1) / block, block>>>(
    //     Jx, Jx_diag, Jx_bend_cross,
    //     forces.data().get(), nullptr,
    //     IBM_q.data().get(),
    //     y,
    //     edges.data().get(),
    //     e2t.data().get(),
    //     triangles.data().get(),
    //     edge_opposite_points.data().get(),
    //     n, 0.2f);
    if ( !sewing_done ) {
        float sewing_k = max(0.f, get_global_parameter("sewing_k", 1e6f));
        float min_dist = 2e-3f;
        n = params.nb_all_stitches;
        compute_stitch_constraint<<<(n + block - 1) / block, block>>>(
            nullptr, Jx_diag, forces.data().get(), nullptr, y, vertices_obj.data().get(),
            obj_data.data().get(),
            vertices_mask.data().get(), stitches.data().get(), min_dist, sewing_k, n);
    }
    int num_constraints = pp_result_size_h + tp_result_size_h + ee_result_size_h;
    if ( num_constraints > 0 ) {
        float IPC_k = max(0.f, get_global_parameter("IPC_k", 1500.f));
        if ( IPC_k > 0.f )
            compute_normal_constraint_IPC_energy<<<(num_constraints + block - 1) / block, block>>>(
                Jx_diag, forces.data().get(),
                normal_constraints.data().get(),
                y, mass_inv.data().get(), obj_data.data().get(),
                vertices_obj.data().get(), IPC_k,
                num_constraints);
    }
}

// Linear conjugate gradient
void SolverPCG::update(float dt) {
    this->dt = dt;
    int block = 256;
    int n = params.nb_all_vertices;
    int blocksPerGrid = (n + block - 1) / block;

    update_begin<<<blocksPerGrid, block>>>(
        vertices_world.data().get(),
        vertices_local.data().get(),
        vertices_obj.data().get(),
        world_matrices.data().get(),
        n);
    // Record old value for collision handle.
    // cudaMemcpyAsync(vertices_old.data().get(), vertices_world.data().get(),
    // params.nb_all_vertices * sizeof(float3),   cudaMemcpyDeviceToDevice);
    int sewing_forced_connect_frame = max(1, (int)get_global_parameter("sewing_forced_connect_frame", 20));
    check_sewing(frame > sewing_forced_connect_frame);
    float3* q = vertices_world.data().get();
    float max_dist = params.cloth_edge_mean_length;
    check_update_pick();
    fill_inv_mass<<<(n + block - 1) / block, block>>>(
        mass_inv.data().get(),
        vertices_obj.data().get(),
        object_types.data().get(),
        masses.data().get(),
        vertices_mask.data().get(), n);
    bool collision_collect_ee = get_global_parameter("collision_collect_ee", 1.f) > 0;
    bool collision_collect_tp = get_global_parameter("collision_collect_tp", 1.f) > 0;
    check_sewing(frame > sewing_forced_connect_frame);
    collision_collect_near_pairs(q, max_dist, true, true, collision_collect_tp, collision_collect_ee);
    compute_constraint(dt);
    constexpr bool UsePreprocessingDiag = true;
    float3* x = velocities.data().get();
    float3* r = this->r.data().get();
    float3* z = this->z.data().get();
    float3* b = r;
    float3* d = this->d.data().get();
    float3* Ad = this->Ad.data().get();
    float3* Ax = Ad;
    Mat3* Jx_diag = this->Jx_diag.data().get();
    Mat3* Jx = this->Jx.data().get();
    Mat3* Jx_bend_cross = this->Jx_bend_cross.data().get();

    n = params.nb_all_cloth_vertices;
    preprocessing<<<(n + block - 1) / block, block>>>(
        Jx_diag, b,
        forces.data().get(),
        velocities.data().get(),
        vertices_mask.data().get(),
        masses.data().get(),
        dt, n
        );
    n = params.nb_all_cloth_edges;
    preprocessing_offdiag<<<(n + block - 1) / block, block>>>(
        Jx, Jx_bend_cross, dt, n);

    n = params.nb_all_cloth_vertices;
    blocksPerGrid = (n + block - 1) / block;

    A_mult_x(Ax, x);
    // r = b - A @ x; d = r || r = b - A @ x; z = M^{-1} @ r; d = z;
    before_ite_kernel<UsePreprocessingDiag><<<blocksPerGrid, block>>>(
        r, b, d, z, Jx_diag, Ax, n);

    float delta_new = vector_field_dot(r, UsePreprocessingDiag ? z : r);
    int max_iter = 1000;
    int iter = 0;
    for ( ; iter < max_iter; ++iter ) {
        if ( delta_new < 1e-6f )break;
        float delta_old = delta_new;
        // Ad = A @ d
        A_mult_x(Ad, d);
        // alpha = (r^T * r) / dot(d, Ad)
        float alpha = delta_old / vector_field_dot(d, Ad);
        // x^{i+1} = x^{i} + alpha * d
        // r^{i+1} = r^{i} + alpha * Ad
        // || z^{i+1} = M^{-1} @ r^{i+1}
        ite_kernel1<UsePreprocessingDiag><<<blocksPerGrid, block>>>(
            r, x, z, d, Ad, Jx_diag, alpha, n);

        delta_new = vector_field_dot(r, UsePreprocessingDiag ? z : r);
        float beta = delta_new / delta_old;
        // d^{i+1} = r^{i+1} + beta * d^{i} || d^{i+1} = z^{i+1} + beta * d^{i}
        ite_kernel2<<<blocksPerGrid, block>>>(
            d, UsePreprocessingDiag ? z : r, beta, n);
    }
    std::cout << "iter = " << iter << ", delta=" << delta_new << std::endl;
    update_dynamics_end<<<blocksPerGrid, block>>>(
        vertices_world.data().get(),
        x,
        vertices_mask.data().get(),
        dt, n);
    int smooth_times = max(0, (int)get_global_parameter("smooth_times", 1));
    if ( !sewing_done ) { smooth_times *= 2; }
    if ( delta_new < -1e2f ) { smooth_times = 20; }
    for ( int i = 0; i < smooth_times; ++i ) {
        laplacian_smoothing<<<blocksPerGrid, block>>>(
            temp_vertices_f3.data().get(),
            velocities.data().get(),
            vertices_mask.data().get(),
            edge_lookup.data().get(),
            dir_edges.data().get(),
            0.02f, n
            );
        thrust::swap(temp_vertices_f3, velocities);
    }
    // contact_handle();
    update_end<<<blocksPerGrid, block>>>(
        vertices_world.data().get(),
        vertices_local.data().get(),
        vertex_proxy.data().get(),
        vertices_obj.data().get(),
        world_matrices_inv.data().get(),
        n);
    reset_pick_mask();
    CUDA_CHECK(cudaDeviceSynchronize());
    frame++;
}
