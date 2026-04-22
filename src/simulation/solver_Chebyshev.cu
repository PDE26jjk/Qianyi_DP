#include "solver_Chebyshev.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"

void SolverChebyshev::init() {
    SolverBase::init();
    velocities_prev.assign(params.nb_all_vertices, make_float3(0.0f, 0.0f, 0.0f));
    Jx_diag.resize(params.nb_all_cloth_vertices);
    q_next.resize(params.nb_all_vertices);
    q_prev.resize(params.nb_all_vertices);
    q_backup.resize(params.nb_all_vertices);
    profile_iter = { 1, 7, 10 };
    profile_value = { .96f, .991f, .999f };
    stepping = 0.7;
    energys.resize(params.nb_all_cloth_vertices);
    debug_colors.assign(params.nb_all_cloth_vertices, make_float3(0.5f, 0.5f, 0.5f));
}
static __global__ void update_begin(
    float3* __restrict__ vertices_world,
    float3* __restrict__ q1,
    float3* __restrict__ x_tilde,
    const float3* __restrict__ vertices_local,
    const float3* __restrict__ velocities,
    const float3* __restrict__ velocities_prev,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        float3 x = mul_homo(world_matrices[obj], vertices_local[i]);
        vertices_world[i] = x;
        constexpr float factor = 0.2f;
        x_tilde[i] = x + velocities[i] * h;
        q1[i] = x + (velocities[i] * (1 + factor) - velocities_prev[i] * factor) * h;
    }
}

__global__ void perform_chebyshev_iterate(
    const float3* __restrict__ q_prev,
    float3* __restrict__ q_next,
    float omega, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= n ) return;

    q_next[i] = omega * (q_next[i] - q_prev[i]) + q_prev[i];
}
__global__ void perform_descent_iterate( // and update energys
    float3* __restrict__ q1,
    float3* __restrict__ q2,
    float* __restrict__ energys,
    const float3* __restrict__ x_tilde,
    const float3* __restrict__ x_fixed,
    const float3* __restrict__ forces,
    const Mat3* __restrict__ Jx_diag,
    const int* __restrict__ proxy,
    const char* __restrict__ mask,
    const float* __restrict__ masses,
    float stepping,
    float h, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if ( i >= n ) return;
    if ( proxy[i] != i ) return;

    float mass = masses[i]; 
    float coeff = mass / (h * h);
    float3 dq = q1[i] - x_tilde[i];
    // Kinetic energy and gravitational potential energy
    energys[i] += 0.5f * coeff * len_sq(dq) + 9.8f * mass * q1[i].z;

    float3 grad = coeff * dq - forces[i];
    Mat3 hessian = Mat3::identity(coeff) - Jx_diag[i];
    if ( mask[i] ) {
        constexpr float fixed_factor = 1.f;
        auto fixed_diff = q1[i] - x_fixed[i];
        // grad = grad + fixed_diff * fixed_factor;
        energys[i] += 0.5f * fixed_factor * len_sq(fixed_diff);
        // hessian = hessian + Mat3::identity(fixed_factor);
    }
    else {
        float3 descent_dir = hessian.inverse() * -grad;
        q2[i] = q1[i] + stepping * descent_dir;

    }
}



static __global__ void update_end(
    float3* __restrict__ vertices_local,
    float3* __restrict__ velocities,
    const float3* __restrict__ vertices_prev,
    const float3* __restrict__ vertices_world,
    const char* __restrict__ vertices_mask,
    const int* __restrict__ vertices_proxy,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices_inv,
    const float h,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int idx = vertices_proxy[i];
        auto x = vertices_world[idx];
        auto v = (x - vertices_prev[idx]) / h;
        if ( !vertices_mask[idx] ) {
            v = v * expf(-h * 0.5f);
            constexpr float max_velocity = 20.f;
            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            velocities[i] = v;
            x = vertices_prev[idx] + v * h;
        }
        else {
            x = vertices_prev[idx];
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }

        int obj = vertices_obj[i];
        vertices_local[i] = mul_homo(world_matrices_inv[obj], x);
    }
}
void SolverChebyshev::contact_handle() {
    SolverBase::contact_handle();
}
static __global__ void clear_data(
    Mat3* __restrict__ Jx_diag,
    float3* __restrict__ forces,
    float* __restrict__ enerys,
    const float* __restrict__ masses,
    const int n
) {
    const auto gravity = make_float3(0.f, 0.f, -9.8f);
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx < n ) {
        enerys[idx] = 0.f;
        forces[idx] = gravity * masses[idx];
        if ( Jx_diag ) Jx_diag[idx] = Mat3::zero();
    }
}
void SolverChebyshev::compute_constraint(float3* q, float3* q_safe, bool update_Jx) {
    int block = 256;
    int n = params.nb_all_cloth_vertices;
    Mat3* Jx_diag = update_Jx ? this->Jx_diag.data().get() : nullptr;
    float3* forces = this->forces.data().get();
    float* enerys = this->energys.data().get();
    clear_data<<<(n + block - 1) / block, block>>>(Jx_diag, forces, enerys,
        masses.data().get(), n);
    n = params.nb_all_cloth_edges;
    int blocksPerGrid = (n + block - 1) / block;
    compute_spring_forces<<<blocksPerGrid, block>>>(
        nullptr, Jx_diag,
        forces, enerys, q,
        edges.data().get(),
        edge_lengths.data().get(),
        n);

    // compute_dihedral_bending_Fizt<<<blocksPerGrid, threadsPerBlock>>>(
    //     Jx,Jx_diag,Jx_bend_cross,
    //     forces.data().get(),
    //     vertices_world.data().get(),
    //     edges.data().get(),
    //     e2t.data().get(),
    //     triangles.data().get(),
    //     edge_opposite_points.data().get(),
    //     n, 10.f);
    compute_quadratic_Bending_IBM<<<blocksPerGrid, block>>>(
        nullptr, Jx_diag, nullptr,
        forces, enerys,
        IBM_q.data().get(), q,
        edges.data().get(),
        e2t.data().get(),
        triangles.data().get(),
        edge_opposite_points.data().get(),
        n, 0.2f);
    if ( q_safe ) {
        n = params.nb_all_cloth_vertices;
        compute_q_safe_constraint<<<(n + block - 1) / block, block>>>(
            Jx_diag, forces, enerys, q, vertices_mask.data().get(), q_safe, n);
    }
    if ( !sewing_done ) {
        float min_dist = 2e-3f;
        n = params.nb_all_stitches;
        compute_stitch_constraint<<<(n + block - 1) / block, block>>>(
            nullptr, nullptr, forces,
            nullptr,
            q,vertices_obj.data().get(), obj_data.data().get(),
            vertices_mask.data().get(), stitches.data().get(), min_dist,1e6f, n);
    }

}

// H. Wang, "A chebyshev semi-iterative approach for accelerating projective and position-based dynamics," ACM Trans. Graph.
void SolverChebyshev::update(float dt) {
    START_TIMER;
    this->dt = dt;
    int block = 256;
    int n = params.nb_all_vertices;
    float3* q = vertices_world.data().get();
    float3* q_next = this->q_next.data().get();
    float3* q_prev = this->q_prev.data().get();
    float3* x_tilde = temp_vertices_f3.data().get(); // S 
    float3* q_backup = this->q_backup.data().get();
    update_begin<<<n + block - 1, block>>>(
        q, q_next, x_tilde,
        vertices_local.data().get(),
        velocities.data().get(),
        velocities_prev.data().get(),
        vertices_obj.data().get(),
        world_matrices.data().get(),
        dt, n);
    check_sewing(frame > 20);
    check_update_pick();
    cudaMemcpyAsync(vertices_old.data().get(), q, params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    std::swap(q, q_next);


    const float min_stepping = 0.1f;
    stepping /= 0.7f;
    stepping = max(min_stepping, stepping);
    int iterate_steps = 97;

    // Follow the logical in the code of the paper.
    float rho = 0.9992f;
    // float theta = 1.f;
    float omega = 1.f;
    bool first = true;
    float last_energy = 0.f;
    n = params.nb_all_cloth_vertices;
    float3* collision_x = nullptr;
    bool collision_updated = false;
    bool collision_done = true;
    for ( int i = 0; i < iterate_steps; ++i ) {
        if ( stepping < 0.01 ) break;
        bool update_energy_Hessian_diag = i % 32 == 0;
        compute_constraint(q, collision_x, update_energy_Hessian_diag);
        perform_descent_iterate<<<n + block - 1, block>>>(
            q, q_next, energys.data().get(), x_tilde,
            vertices_old.data().get(),
            forces.data().get(),
            Jx_diag.data().get(),
            vertex_proxy.data().get(),
            vertices_mask.data().get(),
            masses.data().get(),
            stepping, dt, n);
        // if ( i % 4 == 0 && i > 0 ) {
        //     collision_x = collision_Wu2021_step(q, !collision_updated, collision_done);
        //     collision_updated = true;
        // }

        if ( i % 8 == 0 ) {
            float energy = thrust::reduce(energys.begin(),
                energys.begin() + params.nb_all_cloth_vertices);
            // std::cout << "energy = " << energy << std::endl;
            if ( i == 0 && !first && last_energy > 0.f && energy > 100 * last_energy ) {
                std::cout << "energy > 100*last_energy " << energy << "\\" << last_energy << std::endl;

            }
            if ( i != 0 && (isnan(energy) || energy > last_energy) ) { //need to go back
                cudaMemcpyAsync(q, q_backup, params.nb_all_vertices * sizeof(float3),
                    cudaMemcpyDeviceToDevice);
                iterate_steps += 8 - i;
                i = -1;
                stepping *= 0.7f;
                first = false;
                continue;
            }
            else {
                if ( i > 7 && collision_done && ((energy - last_energy) / energy < 0.01) ) {
                    std::cout << "early break: " << i << std::endl;
                    break;
                }
                cudaMemcpyAsync(q_backup, q, params.nb_all_vertices * sizeof(float3),
                    cudaMemcpyDeviceToDevice);
                last_energy = energy;
            }

        }
        if ( i == 0 ) {
            rho = 0.f;
            omega = 1.f;
        }
        omega = 4 / (4 - rho * rho * omega);
        for ( size_t j = 0; j < profile_iter.size(); ++j ) {
            if ( i == profile_iter[j] - 1 ) {
                rho = 0.f;
                omega = 1.f;
            }
            if ( i == profile_iter[j] ) {
                rho = profile_value[j];
                omega = 2.f / (2.f - rho * rho);
                break;
            }
        }
        if ( omega != 1.f )
            perform_chebyshev_iterate<<<n + block - 1, block>>>(q_prev, q_next, omega, n);
        std::swap(q_prev, q);
        std::swap(q, q_next);
    }

    cudaMemcpyAsync(velocities_prev.data().get(), velocities.data().get(),
        params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    RECORD_TIME("dynamic");
    // collision_LCP_postprocess(q);
    // RECORD_TIME("collision_LCP_postprocess");
    update_end<<<n + block - 1, block>>>(
        vertices_local.data().get(),
        velocities.data().get(),
        vertices_old.data().get(),
        collision_x ? collision_x : q,
        vertices_mask.data().get(),
        vertex_proxy.data().get(),
        vertices_obj.data().get(),
        world_matrices_inv.data().get(),
        dt, n);
    // int smooth_times = 1;
    // if ( !sewing_done ) { smooth_times = 10; }
    // for ( int i = 0; i < smooth_times; ++i ) {
    //     laplacian_smoothing<<<n + block - 1, block>>>(
    //         temp_vertices_f3.data().get(),
    //         velocities.data().get(),
    //         vertices_mask.data().get(),
    //         edge_lookup.data().get(),
    //         dir_edges.data().get(),
    //         0.02f, n
    //         );
    //     thrust::swap(temp_vertices_f3, velocities);
    // }
    reset_pick_mask();
    CUDA_CHECK(cudaDeviceSynchronize());
    ++frame;
}
