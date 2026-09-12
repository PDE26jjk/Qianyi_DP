#include "solver_PDNewton.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>

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
    linear->init(params.nb_all_cloth_vertices, (int)geo->valid_pairs.size(), false);

    dx.resize(params.nb_all_vertices);

    // Subspace (reduced-basis) acceleration is disabled: it measured slower
    // than the plain PCG path. The implementation stays in subspace.cu for
    // future use; restore the commented calls here and in step() to re-enable.
    // geo->init_subspace();
    Jx_diag_pd.assign(params.nb_all_vertices, 0.f);
    // Jx_nondiag_pd.assign(params.nb_all_edges, 0.f);
    linear->Jx_nondiag_identity.assign(params.nb_all_edges, 0.f);
    linear->Jx_bend_cross_identity.assign(params.nb_all_edges, 0.f);
    int block = 256;
    int n = params.nb_all_cloth_edges;
    pd_precompute_spring_forces<<<(n + block - 1) / block, block>>>(
        Jx_diag_pd.data().get(),
        linear->Jx_nondiag_identity.data().get(),
        geo->edges.data().get(),
        geo->obj_data.data().get(), geo->vertices_obj.data().get(),
        n);
    // Disabled together with the subspace path above: H_red / M_red are only
    // consumed by SolverSubspace::A_mult_x, so the reduction is dead work now.
    // geo->precompute_subspace_H(Jx_diag_pd.data().get(), linear->Jx_nondiag_identity.data().get());
    // subspace_rhs.resize(geo->basis_size);
    // subspace_dy.resize(geo->basis_size);
    // if ( subspace_solver == nullptr ) {
    //     subspace_solver = new SolverSubspace(simulator);
    // }
    // subspace_solver->init(geo->basis_size, 0, false);
    auto& contact = geo->get_contact();
    contact.do_collision_detect_broad_phase_before_step = false;
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
    // Tight, non-swept broad phase (the Warp / Style3D arrangement): the tree
    // and the query boxes are inflated only by the contact radius, so the
    // traversal covers far fewer nodes. It is only conservative when the
    // per-substep motion stays inside that radius, which is why it is meant
    // to be paired with smaller substeps. `tight_broad_phase=0` keeps the
    // conservative-CCD behavior.
    // Default on: measured x1.27 at the configured substep size (21.11 vs
    // 26.86 ms/frame on the study-1 scene) with no penetration regression over
    // 300 frames, better seam closure and ~8% more cloth stretch. Set
    // `tight_broad_phase=0` for the conservative-CCD behavior.
    const bool tight_broad_phase = get_global_parameter("tight_broad_phase", 1.f) > 0.5f;
    if ( tight_broad_phase ) {
        contact.refit_bvh_with_target(q_prev, q_prev);
        contact.collision_detect_broad_phase(q_prev, q_prev, query_radius, true);
    }
    else {
        contact.refit_bvh_with_target(q_prev, q_pred);
        contact.collision_detect_broad_phase(q_prev, q_pred, query_radius, true);
    }
    int iters = max(1, (int)get_global_parameter("pd_iters", 10));
    int linear_iters = max(1, (int)get_global_parameter("linear_iters", 10));
    // Subspace acceleration disabled (see SolverPDNewton::init).
    // int subspace_iters = max(0, (int)get_global_parameter("subspace_iters", 1));
    float bending_k = max(0.f, get_global_parameter("bending_k", 0.2f));
    // Planar FEM operator fixes (see compute_BW_FEM): clamping the lateral
    // eigenvalue of the stretch Hessian keeps the assembled matrix positive
    // definite when a fold compresses the membrane, and the shear Hessian is
    // the rank-one term the operator was missing entirely.
    const float fem_psd_clamp = get_global_parameter("fem_psd_clamp", 1.f) > 0.5f ? 1.f : 0.f;
    const float fem_shear_hessian = get_global_parameter("fem_shear_hessian", 1.f) > 0.5f ? 1.f : 0.f;
    // The iteration runs on the engine's own stream so the loop can be
    // captured: the legacy stream cannot be captured on the drivers this
    // project targets (cudaErrorStreamCaptureUnsupported). The fork below
    // orders that stream after everything the frame has queued so far on the
    // legacy stream - the refits and the three broad-phase queries - and the
    // join after the loop hands the result back.
    cudaStream_t work_stream = sim_work_stream();
    if ( stream_fork == nullptr ) {
        cudaEventCreateWithFlags(&stream_fork, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&stream_join, cudaEventDisableTiming);
    }
    cudaEventRecord(stream_fork, 0);
    cudaStreamWaitEvent(work_stream, stream_fork, 0);
    linear->set_work_stream(work_stream);
    // One Projective-Dynamics iteration, minus the seam projection: the
    // projection's host-side ramp changes from frame to frame, so it stays
    // outside the captured region and runs after every replayed iteration.
    // Bending model dispatch.
    const int n_bend = params.nb_all_cloth_edges + params.nb_all_stitches;
    auto launch_bending = [&](Mat3* Jx, Mat3* Jx_diag) {
        if ( geo->bending_model == BendingModel::IBM_quadratic )
            compute_quadratic_bending_IBM<<< (n_bend + block - 1) / block, block, 0, work_stream>>>(
                Jx, Jx_diag,
                f, nullptr,
                geo->IBM_q.data().get(),
                q,
                geo->bend_points.data().get(),
                geo->bend_valid.data().get(),
                geo->bend_cross_rows.data().get(),
                n_bend, bending_k);
        else if ( geo->bending_model == BendingModel::DiscreteShells_GN )
            compute_dihedral_bending_GN<<<(n_bend + block - 1) / block, block, 0, work_stream>>>(
                Jx, Jx_diag,
                f, q,
                geo->bend_points.data().get(),
                geo->bend_rest_theta.data().get(),
                geo->bend_factor.data().get(),
                geo->bend_valid.data().get(),
                geo->bend_cross_rows.data().get(),
                n_bend, bending_k);
        else if ( geo->bending_model == BendingModel::DiscreteShells_AOGS )
            compute_dihedral_bending_AOGS<<<(n_bend + block - 1) / block, block, 0, work_stream>>>(
                Jx, Jx_diag,
                f, q,
                geo->bend_points.data().get(),
                geo->bend_rest_theta.data().get(),
                geo->bend_factor.data().get(),
                geo->bend_valid.data().get(),
                geo->bend_cross_rows.data().get(),
                n_bend, bending_k);
    };

    auto run_pd_iteration = [&]() {
        n = params.nb_all_cloth_vertices;
        cudaMemsetAsync(f, 0, sizeof(float3) * n, work_stream);
        cudaMemsetAsync(f_elastic, 0, sizeof(float3) * n, work_stream);
        cudaMemsetAsync(Jx_diag, 0, sizeof(Mat3) * n, work_stream);
        // Row space = valid_pairs (natural edges + deduped bend pairs); clear
        // the whole table every iteration, not just the edge part.
        cudaMemsetAsync(Jx_nondiag, 0,
            sizeof(Mat3) * geo->valid_pairs.size(), work_stream);
        contact.accumulate_contact_force(f, Jx_diag, h, work_stream);
        step_begin_pd<<<(n + block - 1) / block, block, 0, work_stream>>>(f, q_inertia, q, mass, h, n);
        n = params.nb_all_cloth_edges;
        geo->accumulate_sewing_force(Jx_diag, work_stream);
        if ( geo->constitutive_model == ConstitutiveModel::SpringMass ) {
            accumulate_spring_forces<<<(n + block - 1) / block, block, 0, work_stream>>>(
                Jx_nondiag, Jx_diag, f_elastic, nullptr, q, edges,
                geo->edge_lengths.data().get(),
                obj_data, vertices_obj,
                n);
        }
        else if ( geo->constitutive_model == ConstitutiveModel::FEM_BW ) {
            n = params.nb_all_cloth_triangles;
            compute_BW_FEM<<<(n + block - 1) / block, block, 0, work_stream>>>(
                Jx_nondiag, Jx_diag, f_elastic, nullptr, q, tri_edges,
                edges, geo->Dms.data().get(), geo->areas.data().get(),
                obj_data, vertices_obj,
                n, fem_psd_clamp, fem_shear_hessian);
        }

        n = n_bend;
        launch_bending(Jx_nondiag, Jx_diag);

        n = params.nb_all_cloth_vertices;

        prepare_linear_step_kernel<<<(n + block - 1) / block, block, 0, work_stream>>>(
            dx, f, Jx_diag, M_inv, f_elastic, static_diags, mask, q, q_prev, mask_stiff, n);

        // Subspace acceleration disabled (see SolverPDNewton::init).
        // if ( i < subspace_iters ) solve_subspace(dx, f);

        linear->solve(dx, f, linear_iters);
        step_end_linear<<<(n + block - 1) / block, block, 0, work_stream>>>(
            q, dx, q_pred, q_prev, query_radius, mask, n);
    };

    // A frame issues ~690 kernels, and on this driver a launch that has to
    // follow another one costs ~12 us of GPU idle time, so several
    // milliseconds of the frame are launch gaps rather than work. Replaying
    // the whole iteration loop from a captured graph removes them - one graph
    // launch per frame instead of ~690 kernel launches. The capture is only
    // valid while every host value baked into the recorded launch arguments
    // is unchanged, which is what the key covers; `pd_cuda_graph=0` disables
    // the whole path, and any capture failure falls back to direct launches.
    //
    // The loop, not a single iteration, is what gets recorded: the seam
    // projection runs between iterations, and its arguments change from frame
    // to frame.
    auto run_pd_loop = [&]() {
        for ( int i = 0; i < iters; i++ ) {
            run_pd_iteration();
            geo->project_stitches(work_stream); // seam projection, once per iter
        }
    };

    // The seam projection's host-side ramp is a captured argument, so the key
    // has to carry it: the gate can be off in the first frames, and snap_dist
    // grows until it saturates.
    const int sewing_activation =
        max(0, (int)get_global_parameter("sewing_forced_connect_frame", 80.f));
    const bool projection_active = simulator->frame > sewing_activation;
    const float projection_snap_dist = projection_active
        ? min(max(0.f, get_global_parameter("sewing_snap_dist", 3e-3f))
              * powf(1.5f, (float)(simulator->frame - sewing_activation)), 5e-2f)
        : 0.f;
    const bool graph_enabled = get_global_parameter("pd_cuda_graph", 1.f) > 0.5f;
    if ( graph_enabled && !iter_graph_broken && linear->graph_capture_safe() ) {
        uint64_t key = 1469598103934665603ull;
        auto mix = [&key](uint64_t v) { key = (key ^ v) * 1099511628211ull; };
        mix((uint64_t)(uintptr_t)Jx_diag);
        mix((uint64_t)(uintptr_t)Jx_nondiag);
        mix((uint64_t)(uintptr_t)M_inv);
        mix((uint64_t)(uintptr_t)f);
        mix((uint64_t)(uintptr_t)f_elastic);
        mix((uint64_t)(uintptr_t)q);
        mix((uint64_t)(uintptr_t)dx);
        mix((uint64_t)(uintptr_t)static_diags);
        mix((uint64_t)(uintptr_t)mask);
        mix((uint64_t)params.nb_all_cloth_vertices);
        mix((uint64_t)params.nb_all_cloth_edges);
        mix((uint64_t)params.nb_all_cloth_triangles);
        mix((uint64_t)params.nb_all_stitches);
        mix((uint64_t)params.nb_all_vertices);
        mix((uint64_t)geo->valid_pairs.size());
        mix((uint64_t)(int)geo->constitutive_model);
        mix((uint64_t)(int)geo->bending_model);
        mix((uint64_t)linear_iters);
        mix((uint64_t)simulator->parameter_version());
        auto mix_float = [&mix](float v) {
            float f = v;
            uint32_t bits = 0;
            memcpy(&bits, &f, sizeof(bits));
            mix((uint64_t)bits);
        };
        mix_float(h);
        mix_float(mask_stiff);
        mix_float(bending_k);
        mix_float(query_radius);
        mix_float(projection_snap_dist);
        mix((uint64_t)projection_active);
        // Every array the captured loop reads, by identity and size: a
        // rest-shape refresh (cloth plasticity) or any other rebuild that
        // moves or resizes one of them has to invalidate the capture even if
        // its values are the only thing that changed.
        auto mix_buffer = [&mix](const void* ptr, size_t count) {
            mix((uint64_t)(uintptr_t)ptr);
            mix((uint64_t)count);
        };
        mix_buffer(geo->bend_points.data().get(), geo->bend_points.size());
        mix_buffer(geo->bend_rest_theta.data().get(), geo->bend_rest_theta.size());
        mix_buffer(geo->bend_factor.data().get(), geo->bend_factor.size());
        mix_buffer(geo->bend_valid.data().get(), geo->bend_valid.size());
        mix_buffer(geo->bend_cross_rows.data().get(), geo->bend_cross_rows.size());
        mix_buffer(geo->IBM_q.data().get(), geo->IBM_q.size());
        mix_buffer(geo->edges.data().get(), geo->edges.size());
        mix_buffer(geo->triangles.data().get(), geo->triangles.size());
        mix_buffer(geo->triangle_indices.data().get(), geo->triangle_indices.size());
        mix_buffer(geo->edge_opposite_points.data().get(),
            geo->edge_opposite_points.size());
        mix_buffer(geo->Dms.data().get(), geo->Dms.size());
        mix_buffer(geo->areas.data().get(), geo->areas.size());
        mix_buffer(geo->edge_lengths.data().get(), geo->edge_lengths.size());
        mix_buffer(geo->obj_data.data().get(), geo->obj_data.size());
        mix_buffer(geo->vertices_obj.data().get(), geo->vertices_obj.size());
        mix_buffer(geo->pos_world.data().get(), geo->pos_world.size());
        mix_buffer(geo->pos_pred.data().get(), geo->pos_pred.size());
        mix_buffer(geo->pos_step_prev.data().get(), geo->pos_step_prev.size());
        mix_buffer(geo->pos_inertia.data().get(), geo->pos_inertia.size());
        mix_buffer(geo->velocities.data().get(), geo->velocities.size());
        mix_buffer(geo->masses.data().get(), geo->masses.size());
        mix_buffer(geo->mass_inv.data().get(), geo->mass_inv.size());
        mix_buffer(geo->forces.data().get(), geo->forces.size());
        mix_buffer(geo->elastic_forces.data().get(), geo->elastic_forces.size());
        mix_buffer(geo->static_diags.data().get(), geo->static_diags.size());
        mix_buffer(geo->vertices_mask.data().get(), geo->vertices_mask.size());
        mix_buffer(geo->vertex_normals.data().get(), geo->vertex_normals.size());
        mix_buffer(geo->edge_normals.data().get(), geo->edge_normals.size());
        mix_buffer(geo->stitches.data().get(), geo->stitches.size());
        mix_buffer(geo->stitches_status.data().get(), geo->stitches_status.size());
        mix_buffer(geo->stitch_cluster_lookup.data().get(),
            geo->stitch_cluster_lookup.size());
        mix_buffer(geo->stitch_cluster_members.data().get(),
            geo->stitch_cluster_members.size());
        mix_buffer(geo->stitch_cluster_locked.data().get(),
            geo->stitch_cluster_locked.size());
        mix_buffer(geo->valid_pairs.data().get(), geo->valid_pairs.size());
        if ( iter_graph_exec != nullptr && key != iter_graph_key ) {
            cudaGraphExecDestroy(iter_graph_exec);
            iter_graph_exec = nullptr;
        }
        if ( iter_graph_exec == nullptr ) {
            cudaGraph_t graph = nullptr;
            cudaError_t begin_err = cudaStreamBeginCapture(
                work_stream, cudaStreamCaptureModeThreadLocal);
            run_pd_loop();
            cudaError_t err = cudaStreamEndCapture(work_stream, &graph);
            if ( begin_err != cudaSuccess ) err = begin_err;
            if ( err == cudaSuccess && graph != nullptr ) {
                err = cudaGraphInstantiate(&iter_graph_exec, graph, nullptr,
                    nullptr, 0);
            }
            if ( graph != nullptr ) cudaGraphDestroy(graph);
            if ( err != cudaSuccess ) {
                std::cout << "PD iteration capture failed (" << cudaGetErrorString(err)
                    << ", code " << (int)err
                    << "); falling back to direct launches." << std::endl;
                iter_graph_exec = nullptr;
                iter_graph_broken = true;
                cudaGetLastError();
            }
            else {
                iter_graph_key = key;
            }
        }
    }

    if ( iter_graph_exec != nullptr ) cudaGraphLaunch(iter_graph_exec, work_stream);
    else run_pd_loop();
    cudaEventRecord(stream_join, work_stream);
    cudaStreamWaitEvent(0, stream_join, 0);
    // The linear solve reports a non-finite residual through a sticky device
    // flag instead of blocking once per solve; consume it once per frame.
    if ( linear->consume_failure_flag() ) {
        std::cout << "PCG ended with NaN residual." << std::endl;
        throw std::exception("PCG nan");
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
    // geo->average_stitch_cluster_velocities(); // drop the snap velocity kick
}
