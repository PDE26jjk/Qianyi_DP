#include "solver_PDNewton.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"

// Adapted from Newton's style3d solver

static __global__ void prepare_linear_step_kernel(
    float3* __restrict__ dx,
    float3* __restrict__ rhs,
    Mat3* __restrict__ Jx_diags,
    Mat3* __restrict__ M_inv,
    const float3* __restrict__ f_elastic,
    const float*__restrict__ static_diags,
    const char*__restrict__ mask,
    const float3* __restrict__ pos_world,
    const float3* __restrict__ pos_prev,
    const float mask_stiff,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= n ) return;

    Mat3 diag = Jx_diags[tid];
    diag.add_diag(static_diags[tid]);
    Jx_diags[tid] = diag;
    // prepare_jacobi_preconditioner_kernel
    if ( diag.r[0].x > 0.0f ) {
        M_inv[tid] = diag.inverse();
    }
    else {
        M_inv[tid] = Mat3::zero();
    }
    if ( mask[tid] ) {
        dx[tid] = pos_world[tid] - pos_prev[tid];
        rhs[tid] += (pos_world[tid] - pos_prev[tid]) * mask_stiff;
    }
    rhs[tid] += f_elastic[tid];
}
static __global__ void step_begin_pd(
    float3* __restrict__ rhs,
    const float3* __restrict__ x_inertia,
    const float3* __restrict__ x_curr,
    const float* __restrict__ mass,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        rhs[i] += (x_inertia[i] - x_curr[i]) * mass[i] / (h * h);
    }
}
static __global__ void step_begin_pc(
    float3* __restrict__ rhs,
    float*__restrict__ static_diags,
    const float3* __restrict__ x_tr,
    const float3* __restrict__ x_curr,
    const float* __restrict__ mass,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        rhs[i] = (x_tr[i] - x_curr[i]) * mass[i] / (h * h);
        static_diags[i] = mass[i] / (h * h);
    }
}
static __global__ void solve_diag(
    float3* __restrict__ dx,
    const float3* __restrict__ rhs,
    const Mat3* __restrict__ Jx_diags,
    const float*__restrict__ static_diags,
    const char*__restrict__ mask,
    const float3* __restrict__ pos_world,
    const float3* __restrict__ pos_prev,
    const float mask_stiff,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= n ) return;

    Mat3 diag = Jx_diags[tid];
    diag.add_diag(static_diags[tid]);
    float3 f = rhs[tid];

    if ( mask[tid] ) {
        f += (pos_world[tid] - pos_prev[tid]) * mask_stiff;
    }
    if ( len_sq(f) > 1e-16f ) {
        dx[tid] = diag.inverse() * f;
    }
    else {
        dx[tid] = make_float3(0.f, 0.f, 0.f);
    }
}
static __global__ void step_end_linear(
    float3* __restrict__ pos_world,
    float3* __restrict__ dx,
    const float3* __restrict__ pos_target,
    const float3* __restrict__ pos_prev,
    float max_displacement,
    const char* __restrict__ mask,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !mask[i] ) {
            float3 x = pos_world[i] + dx[i];
            if ( isinf(x.x) || isinf(x.y) || isinf(x.z) ||
                isnan(x.x) || isnan(x.y) || isnan(x.z) || len_sq(x) > 1e6 ) {
                // printf("step_end_linear ERROR!!! %f \n", len_sq(x));
            }
            else {
                pos_world[i] = clamp_to_trajectory_envelope(
                    pos_prev[i], pos_target[i], x, max_displacement);
            }
        }
        dx[i] = make_float3(0.0f, 0.f, 0.f);
    }
}
static __global__ void prepare_pc_step_kernel(
    float3* __restrict__ pos_world,
    float3* __restrict__ dx,
    const float3* __restrict__ pos_prev,
    const char* __restrict__ vertices_mask,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const bool ground,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !vertices_mask[i] ) {
            float3 x = pos_world[i];
            if ( ground ) {
                float min_z = obj_data[vertices_obj[i]].thickness;
                if ( x.z <= min_z ) {
                    x.z = min_z;
                    pos_world[i] = x;
                }
            }
        }
        dx[i] = (pos_world[i] - pos_prev[i]) * 1.2f;
    }
}
static __global__ void step_end_kernel(
    float3* __restrict__ pos_world,
    float3* __restrict__ velocities,
    const float3* __restrict__ pos_prev,
    const char* __restrict__ vertices_mask,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float h,
    const float max_velocity,
    const bool ground,
    const float ground_f,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !vertices_mask[i] ) {
            auto x_old = pos_prev[i];
            float3 x = pos_world[i];
            float3 v = (x - x_old) / h;
            if ( ground ) {
                float min_z = obj_data[vertices_obj[i]].thickness;
                if ( x.z <= min_z ) {
                    x.z = min_z;
                    v.z = 0.f;
                    // v = v * expf(-h * ground_f);
                    pos_world[i] = x;
                }
            }

            if ( norm(v) > max_velocity ) {
                v = normalized(v) * max_velocity;
                x = x_old + v * h;
                pos_world[i] = x;
            }
            v = v * expf(-h * 0.5f);
            velocities[i] = v;
        }
        else {
            // pos_world[i] = pos_prev[i];
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
    }
}

void SolverPDNewton::init() {
    SolverBase::init();
    int linear_solver_type = (int)get_global_parameter("linear_solver_type", 0);
    if ( linear_solver_type == 0 ) {
        m_linear_solver_name = "PCG";
    }
    else if ( linear_solver_type == 1 ) {
        m_linear_solver_name = "Jacobi";
    }
    else {
        m_linear_solver_name = "???";
    }
    if ( linear == nullptr || m_linear_solver_name != m_last_linear_solver_name ) {
        delete linear;
        if ( m_linear_solver_name == "PCG" ) {
            linear = new SolverPCG(simulator);
        }
        else if ( m_linear_solver_name == "Jacobi" ) {
            linear = new SolverJacobi(simulator);
        }
        else {
            throw std::runtime_error("Unknown linear solver type: " + m_linear_solver_name);
        }
        m_last_linear_solver_name = m_linear_solver_name;
    }
    auto& params = *simulator->get_geo_params();
    auto* geo = simulator->get_geo();
    linear->init(params.nb_all_cloth_vertices, params.nb_all_cloth_edges, false);

    dx.resize(params.nb_all_vertices);

    geo->init_subspace();
    Jx_diag_pd.assign(params.nb_all_vertices, 0.f);
    // Jx_nondiag_pd.assign(params.nb_all_edges, 0.f);
    linear->Jx_nondiag_identity.assign(params.nb_all_vertices, 0.f);
    linear->Jx_bend_cross_identity.assign(params.nb_all_edges, 0.f);
    int block = 256;
    int n = params.nb_all_cloth_edges;
    pd_precompute_spring_forces<<<(n + block - 1) / block, block>>>(
        Jx_diag_pd.data().get(),
        linear->Jx_nondiag_identity.data().get(),
        geo->edges.data().get(),
        geo->obj_data.data().get(), geo->vertices_obj.data().get(),
        n);
    geo->precompute_subspace_H(Jx_diag_pd.data().get(), linear->Jx_nondiag_identity.data().get());
    subspace_rhs.resize(geo->basis_size);
    subspace_dy.resize(geo->basis_size);
    if ( subspace_solver == nullptr ) {
        subspace_solver = new SolverSubspace(simulator);
    }
    subspace_solver->init(geo->basis_size, 0, false);

}

static __global__ void truncate_forces_kernel(
    float3* __restrict__ forces,
    Mat3* __restrict__ Jx_diag,
    const float* __restrict__ static_diags,
    float max_force_scale,
    int num_vertices
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_vertices ) return;

    float3 f = forces[i];
    float mag = norm(f);
    if ( mag < 1e-12f ) return;

    float max_force = static_diags[i] * max_force_scale;
    // if (max_force <= 0.0f) {
    //     max_force = 1e-6f;
    // }
    if ( mag > max_force ) {
        float scale = max_force / mag;
        forces[i] *= scale;
        Jx_diag[i] *= scale;
    }
}

__global__ void preprocessing_nondiag(
    Mat3* __restrict__ Jx_nondiag,
    const float* __restrict__ Jx_nondiag_pd,
    const int n // edge size
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        Jx_nondiag[i] = Mat3::identity(Jx_nondiag_pd[i]);
    }
}

void SolverPDNewton::step(float h) {
    // Some kind of Newton method with a greatly simplified Jacobian matrix,
    // including but not limited to Projective Dynamics (PD), may ignore second-order
    // derivatives and only retain the diagonal elements in collisions, among other simplifications.
    auto& params = *simulator->get_geo_params();
    auto* geo = simulator->get_geo();
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;

    int block = 256;
    int blocksPerGrid = (n + block - 1) / block;

    float3* q = geo->pos_world.data().get();
    float3* q_pred = geo->pos_pred.data().get();
    const float3* q_prev = geo->pos_step_prev.data().get();
    float3* q_inertia = geo->pos_inertia.data().get();
    float3* q_tr = geo->pos_inertia.data().get();
    float3* v = geo->velocities.data().get();
    float3* v_prev = geo->vel_prev.data().get();
    float3* f = geo->forces.data().get();
    float3* f_elastic = geo->elastic_forces.data().get();
    float3* dx = this->dx.data().get();
    int2* edges = geo->edges.data().get();
    int3* tri_edges = geo->triangles.data().get();
    int3* tris = geo->triangle_indices.data().get();
    int2* e2t = geo->e2t.data().get();
    int2* eop = geo->edge_opposite_points.data().get();
    float* rest_thetas = geo->rest_thetas.data().get();
    char* mask = geo->vertices_mask.data().get();
    float* mass = geo->masses.data().get();
    float* mass_inv = geo->mass_inv.data().get();
    auto* obj_data = geo->obj_data.data().get();
    int* vertices_obj = geo->vertices_obj.data().get();
    const float* Jx_diag_pd = this->Jx_diag_pd.data().get();
    // const float* Jx_nondiag_pd = this->Jx_nondiag_pd.data().get();
    Mat3* Jx_diag = linear->Jx_diag.data().get();
    Mat3* M_inv = linear->M_inv.data().get();
    Mat3* Jx_nondiag = linear->Jx_nondiag.data().get();
    Mat3* Jx_bending_cross = linear->Jx_bend_cross.data().get();
    float* static_diags = geo->static_diags.data().get();
    cudaMemcpyAsync(static_diags, Jx_diag_pd, n * sizeof(float), cudaMemcpyDeviceToDevice);
    float mask_stiff = max(0.f, get_global_parameter("mask_stiff", 1e2f));
    auto& contact = geo->get_contact();
    float query_radius = max(0.f, get_global_parameter("query_radius", 0.001f));
    forward_step<<<(n + block - 1) / block, block>>>(
        v, v_prev, mass_inv,
        nullptr, f_elastic,
        mask, q, q_pred, q_inertia, nullptr,
        static_diags,
        h, mask_stiff, geo->gravity, true, n);
    contact.refit_bvh_with_target(q_prev, q_pred);
    contact.collision_detect_broad_phase(q_prev, q_pred, query_radius, true);

    int iters = max(1, (int)get_global_parameter("pd_iters", 10));
    int linear_iters = max(1, (int)get_global_parameter("linear_iters", 10));
    int subspace_iters = max(0, (int)get_global_parameter("subspace_iters", 1));
    float max_force_scale = max(0.f, get_global_parameter("max_force_scale", 100.f));
    float bending_k = max(0.f, get_global_parameter("bending_k", 0.2f));
    for ( int i = 0; i < iters; i++ ) {
        n = params.nb_all_cloth_vertices;
        cudaMemsetAsync(f, 0, sizeof(float3) * n);
        cudaMemsetAsync(f_elastic, 0, sizeof(float3) * n);
        cudaMemsetAsync(Jx_diag, 0, sizeof(Mat3) * n);
        contact.accumulate_contact_force(f, Jx_diag);
        truncate_forces_kernel<<<(n + block - 1), block>>>(
            f, Jx_diag, static_diags, max_force_scale, n);

        step_begin_pd<<<(n + block - 1) / block, block>>>(f, q_inertia, q, mass, h, n);
        n = params.nb_all_cloth_edges;
        cudaMemsetAsync(Jx_nondiag, 0, sizeof(Mat3) * n);
        cudaMemsetAsync(Jx_bending_cross, 0, sizeof(Mat3) * n);
        // preprocessing_nondiag<<<(n + block - 1) / block, block>>>(
        //     Jx_nondiag, Jx_nondiag_pd, n);
        // cudaMemsetAsync(Jx_bend_cross, 0, sizeof(Mat3) * n);
        // compute_constraint();
        // geo->accumulate_sewing_force();
        if ( geo->constitutive_model == ConstitutiveModel::SpringMass ) {
            accumulate_spring_forces<<<(n + block - 1) / block, block>>>(
                Jx_nondiag, Jx_diag, f_elastic, nullptr, q, edges,
                geo->edge_lengths.data().get(),
                geo->obj_data.data().get(), geo->vertices_obj.data().get(),
                n);
        }
        else if ( geo->constitutive_model == ConstitutiveModel::FEM_BW ) {}
        
        n = params.nb_all_cloth_edges;
        if ( geo->bending_model == BendingModel::IBM_quadratic )
            compute_quadratic_bending_IBM<<< (n + block - 1) / block, block>>>(
                Jx_nondiag, Jx_diag, Jx_bending_cross,
                f, nullptr,
                geo->IBM_q.data().get(),
                q, edges, e2t, tri_edges, eop,
                n, bending_k);
        else if ( geo->bending_model == BendingModel::DiscreteShells_GN )
            compute_dihedral_bending_GN<<<(n + block - 1), block>>>(
                Jx_nondiag, Jx_diag, Jx_bending_cross,
                f, q, edges, e2t, rest_thetas,
                tri_edges, eop,
                n, bending_k);
        else if ( geo->bending_model == BendingModel::DiscreteShells_AOGS )
            compute_dihedral_bending_AOGS<<<(n + block - 1), block>>>(
                Jx_nondiag, Jx_diag, Jx_bending_cross,
                f, q, edges, e2t, rest_thetas,
                tri_edges, eop,
                n, bending_k);

        n = params.nb_all_cloth_vertices;

        prepare_linear_step_kernel<<<(n + block - 1) / block, block>>>(
            dx, f, Jx_diag, M_inv, f_elastic, static_diags, mask, q, q_prev, mask_stiff, n);

        if ( i < subspace_iters ) solve_subspace(dx, f);

        linear->solve(dx, f, linear_iters);
        step_end_linear<<<(n + block - 1) / block, block>>>(
            q, dx, q_pred, q_prev, query_radius, mask, n);
        // CUDA_CHECK(cudaDeviceSynchronize());
    }
    // try to do penetration correction
    iters = max(0, (int)get_global_parameter("pc_iters", 2));
    // cudaMemsetAsync(Jx_bend_cross, 0, sizeof(Mat3) * params.nb_all_cloth_edges);
    // cudaMemsetAsync(Jx_nondiag, 0, sizeof(Mat3) * params.nb_all_cloth_edges);
    float ground_f = max(0.f, (get_global_parameter("ground_f", 1e3)));
    n = params.nb_all_vertices;
    float max_vel = max(0.f, get_global_parameter("max_vel", 1000));
    // prepare_pc_step_kernel<<<(n + block - 1) / block, block>>>(
    // q, dx, q_prev, mask, obj_data, vertices_obj, geo->ground, n);
    // contact.refit_bvh(q_prev, dx);
    // contact.collision_detect_broad_phase(q_prev, dx);
    // cudaMemcpyAsync(q_tr, q,
    //     sizeof(float3) * n, cudaMemcpyDeviceToDevice);
    // for ( int i = 0; i < iters; i++ ) {
    //     n = params.nb_all_cloth_vertices;
    //     step_begin_pc<<<(n + block - 1) / block, block>>>(f, static_diags, q_tr, q, mass, h, n);
    //     cudaMemcpyAsync(q_tr, q,
    //         sizeof(float3) * n, cudaMemcpyDeviceToDevice);
    //     cudaMemsetAsync(Jx_diag, 0, sizeof(Mat3) * n);
    //     contact.accumulate_contact_force(f, Jx_diag);
    //
    //     truncate_forces_kernel<<<(n + block - 1), block>>>(
    //         f, Jx_diag, static_diags, max_force_scale, n);
    //
    //     solve_diag<<<(n + block - 1) / block, block>>>(
    //         dx, f, Jx_diag, static_diags, mask, q, q_prev, mask_stiff, n);
    //
    //     step_end_linear<<<(n + block - 1) / block, block>>>(
    //         q, dx, mask, n);
    //     // CUDA_CHECK(cudaDeviceSynchronize());
    // }
    n = params.nb_all_vertices;
    cudaMemcpyAsync(v_prev, v, n * sizeof(float3), cudaMemcpyDeviceToDevice);
    step_end_kernel<<<(n + block - 1) / block, block>>>(
        q, v, q_prev, mask, obj_data, vertices_obj, h, max_vel, geo->ground, ground_f, n);
    // CUDA_CHECK(cudaDeviceSynchronize());
}
