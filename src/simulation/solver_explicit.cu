#include "solver_explicit.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"



static __global__ void step_end_kernel(
    float3* __restrict__ vertices_world,
    float3* __restrict__ velocities,
    const float3* __restrict__ pos_prev,
    const float3* __restrict__ other_forces,
    const float3* __restrict__ elastic_forces,
    const char* __restrict__ vertices_mask,
    const float* __restrict__ mass_inv,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float3 gravity,
    const float h,
    const float max_velocity,
    const bool ground,
    const float ground_f,
    // Velocity decay rate (1/s), see SolverBase::velocity_damping. The previous
    // fixed `exp(-h * 0.5)` ignored every parameter.
    const float damping_rate,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !vertices_mask[i] ) {
            // Symplectic (semi-implicit) Euler: the velocity advances with the
            // force sampled by this substep's assembly, the position with the
            // new velocity. The previous form advanced the position by
            // `f/m * h^2` from the *old* velocity, which is explicit Euler on
            // the positions and is unconditionally unstable for the elastic
            // oscillator - and it read that force from `pos_ine`, a buffer the
            // explicit path never wrote, so the instability was masked by the
            // force term being effectively dropped (measured: 2.7 m of motion
            // and a collapse onto the ground on a skirt drape, then NaN once
            // the term was correctly applied).
            const float3 x_old = pos_prev[i];
            const float mi = mass_inv[i];
            float3 v = velocities[i] + (gravity + (other_forces[i] + elastic_forces[i]) * mi) * h;
            float3 x = x_old + v * h;
            if ( ground ) {
                float min_z = obj_data[vertices_obj[i]].thickness;
                if ( x.z <= min_z ) {
                    x.z = min_z;
                    if ( v.z < 0.f ) v.z = 0.f;
                    v = v * expf(-h * ground_f);
                }
            }
            if ( norm(v) > max_velocity ) {
                v = normalized(v) * max_velocity;
                x = x_old + v * h;
            }
            v = v * expf(-h * damping_rate);
            velocities[i] = v;
            vertices_world[i] = x;
        }
        else {
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
    }
}

void SolverExplicit::step(float h) {
    // Explicit Euler
    auto& params = *simulator->get_geo_params();
    auto* geo = simulator->get_geo();
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;

    int block = 256;
    // update_begin<<<blocksPerGrid, block>>>(
    //     vertices_world.data().get(),
    //     vertices_old.data().get(),
    //     vertices_local.data().get(),
    //     obj_data.data().get(),
    //     vertices_local_new_frame.data().get(),
    //     vertices_obj.data().get(),
    //     world_matrices.data().get(),
    //     n);
    // update_pin(vertices_world.data().get());
    // cudaMemcpyAsync(vertices_new.data().get(), vertices_world.data().get(),
    //     params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    // int obj_num = params.nb_all_objects;
    // update_begin_obj<<<(obj_num + block - 1) / block, block>>>(
    //     obj_data.data().get(),
    //     world_matrices.data().get(),
    //     world_matrices_inv.data().get(),
    //     obj_num);
    //
    // int sewing_forced_connect_frame = max(1, (int)get_global_parameter("sewing_forced_connect_frame",80));
    // check_sewing(frame > sewing_forced_connect_frame);
    // fill_inv_mass<<<(n + block - 1) / block, block>>>(
    //     mass_inv.data().get(),
    //     vertices_obj.data().get(),
    //     object_types.data().get(),
    //     masses.data().get(),
    //     vertices_mask.data().get(), n);
    float3* q = geo->pos_world.data().get();
    const float3* q_prev = geo->pos_step_prev.data().get();
    float3* q_inertia = geo->pos_inertia.data().get();
    float3* v = geo->velocities.data().get();
    float3* f = geo->forces.data().get();
    float3* f_elastic = geo->elastic_forces.data().get();
    int2* edges = geo->edges.data().get();
    int2* e2t = geo->e2t.data().get();
    int2* eop = geo->edge_opposite_points.data().get();
    int3* tri_edges = geo->triangles.data().get();
    int3* tris = geo->triangle_indices.data().get();
    char* mask = geo->vertices_mask.data().get();
    float* mass = geo->masses.data().get();
    float* mass_inv = geo->mass_inv.data().get();
    auto* obj_data = geo->obj_data.data().get();
    int* vertices_obj = geo->vertices_obj.data().get();
    float* static_diags = geo->static_diags.data().get();
    // float max_dist = params.cloth_edge_mean_length;
    // int update_pick_substeps = max(1, (int)get_global_parameter("update_pick_substeps",10));
    // float IPC_k = max(0.f, get_global_parameter("IPC_k",1500.f));
    float max_vel = max(0.f, get_global_parameter("max_vel", 1000));
    // int update_collision_substeps = max(1, (int)get_global_parameter("update_collision_substeps",20));
    // // int LCP_substeps = max(1, (int)get_global_parameter("LCP_substeps",20));
    // bool collision_collect_ee = get_global_parameter("collision_collect_ee", 1.f) > 0;
    // bool collision_collect_tp = get_global_parameter("collision_collect_tp", 1.f) > 0;


    // if ( substep % update_collision_substeps == 0 ) {
    //     float factor = clamp(1.f - (dt_rest / dt) + 0.1f, 0., 1.f);
    //     update_interpolated_position<<<blocksPerGrid, block>>>(
    //         q, vertices_new.data().get(),
    //         vertices_old.data().get(),
    //         factor, params.nb_all_cloth_vertices, n);
    //     collision_collect_near_pairs(q, max_dist, true, true, collision_collect_tp, collision_collect_ee);
    // }
    cudaMemsetAsync(static_diags, 0, params.nb_all_vertices * sizeof(float));
    cudaMemsetAsync(f, 0, params.nb_all_cloth_vertices * sizeof(float3));
    cudaMemsetAsync(f_elastic, 0, params.nb_all_cloth_vertices * sizeof(float3));

    // `pos_world` stays at the substep start while the forces are assembled:
    // the explicit scheme samples the force at the position it integrates
    // from (`f(x_n)`), which is what keeps it stable. The inertia prediction
    // goes into `pos_inertia` and is used only as the swept target of the
    // broad phase below.
    forward_step<<<(n + block - 1) / block, block>>>(
        v, nullptr, mass_inv,
        nullptr, f_elastic,
        mask, q, nullptr, q_inertia, nullptr,
        static_diags,
        h, 1e2, geo->gravity, 0, n);
    // Per-substep collision refresh, so the contact penalty in this substep
    // uses candidate pairs and a swept BVH built from this substep's inertia
    // prediction instead of the frame start (`Simulator::update` detects
    // collisions once per frame, before the substep loop).
    {
        auto& contact = geo->get_contact();
        const float query_radius = max(1e-5f, get_global_parameter("query_radius", 1e-3f));
        contact.refit_bvh_with_target(q_prev, q_inertia);
        contact.collision_detect_broad_phase(q_prev, q_inertia, query_radius, true);
    }
    // n = pp_result_size_h;
    // compute_collision_penalty_force_point_point<<<(n + block - 1) / block, block>>>(
    //     nullptr, nullptr,
    //     forces.data().get(),
    //     velocities.data().get(),
    //     pp_collision_result.data().get(),
    //     q, max_dist, h, n);

    // int num_constraints = pp_result_size_h + tp_result_size_h + ee_result_size_h;
    // if ( num_constraints > 0 ) {
    //     cudaMemsetAsync(weight.data().get(), 0, params.nb_all_cloth_vertices * sizeof(float));
    //     compute_normal_constraint_IPC_force<<<(num_constraints + block - 1) / block, block>>>(
    //         forces.data().get(), weight.data().get(), normal_constraints.data().get(),
    //         q, mass_inv.data().get(), obj_data.data().get(), vertices_obj.data().get(),
    //         IPC_k, num_constraints);
    //     n = params.nb_all_cloth_edges;
    //     apply_weight_force<<<(n + block - 1) / block, block>>>(
    //         forces.data().get(), weight.data().get(), n);
    // }

    n = params.nb_all_cloth_edges;
    const float base_spring_k =
        geo->get_global_parameter("base_spring_stiffness", default_base_spring_stiffness);
    const float stiffen_start =
        geo->get_global_parameter("strain_stiffen_start", default_strain_stiffen_start);
    const float stiffen_rate =
        geo->get_global_parameter("strain_stiffen_rate", default_strain_stiffen_rate);
    if ( geo->constitutive_model == ConstitutiveModel::SpringMass )
        accumulate_spring_forces<<<(n + block - 1) / block, block>>>(nullptr, nullptr,
            f_elastic, nullptr, q, edges,
            geo->edge_lengths.data().get(),
            geo->obj_data.data().get(), geo->vertices_obj.data().get(),
            n, base_spring_k, stiffen_start, stiffen_rate);
    else if ( geo->constitutive_model == ConstitutiveModel::FEM_BW ) {
        // The planar FEM membrane path, previously commented out here: the
        // explicit solver only ever integrated the spring lattice, so
        // `constitutive_model_planar` had no effect on it. Forces only (both
        // Hessian arguments null) - the explicit integrator needs no tangent.
        n = params.nb_all_cloth_triangles;
        compute_BW_FEM<<<(n + block - 1) / block, block>>>(nullptr, nullptr,
            f_elastic, nullptr, q, tri_edges, edges,
            geo->Dms.data().get(), geo->areas.data().get(),
            geo->obj_data.data().get(), geo->vertices_obj.data().get(),
            n, 1.f, 1.f, base_spring_k, stiffen_start, stiffen_rate);
    }
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
    // compute_BW_FEM<<<(n + block - 1) / block, block>>>(
    //     nullptr, nullptr,
    //     forces.data().get(), nullptr,
    //     geo->pos_world.data().get(),
    //     geo->triangles.data().get(),
    //     geo->edges.data().get(),
    //     geo->vertices_obj.data().get(),
    //     nullptr,
    //     geo->Dms.data().get(),
    //     n);

    n = params.nb_all_cloth_edges + params.nb_all_stitches;
    // Bending stiffness: this path passed a literal 0.2f, so `bending_k` had no
    // effect on the explicit solver.
    const float bending_k = max(0.f, get_global_parameter("bending_k", 0.2f));
    if ( geo->bending_model == BendingModel::IBM_quadratic )
        compute_quadratic_bending_IBM<<< (n + block - 1) / block, block>>>(
            nullptr, nullptr,
            f, nullptr,
            geo->IBM_q.data().get(),
            q,
            geo->bend_points.data().get(),
            geo->bend_valid.data().get(),
            geo->bend_cross_rows.data().get(),
            n, bending_k);
    else if ( geo->bending_model == BendingModel::DiscreteShells_GN )
        compute_dihedral_bending_GN<<<(n + block - 1) / block, block>>>(
            nullptr, nullptr,
            f, q,
            geo->bend_points.data().get(),
            geo->bend_rest_theta.data().get(),
            geo->bend_factor.data().get(),
            geo->bend_valid.data().get(),
            geo->bend_cross_rows.data().get(),
            n, bending_k);
    else if ( geo->bending_model == BendingModel::DiscreteShells_AOGS )
        // AOGS was previously folded into the GN branch ("the forces are the
        // same"), which is not the case: it is a different curvature measure.
        // Both now take their own kernel, as PDNewton already did.
        compute_dihedral_bending_AOGS<<<(n + block - 1) / block, block>>>(
            nullptr, nullptr,
            f, q,
            geo->bend_points.data().get(),
            geo->bend_rest_theta.data().get(),
            geo->bend_factor.data().get(),
            geo->bend_valid.data().get(),
            geo->bend_cross_rows.data().get(),
            n, bending_k);
    geo->accumulate_sewing_force(nullptr);
    geo->get_contact().accumulate_contact_force(f, nullptr, h);
    // update substep end

    n = params.nb_all_cloth_vertices;
    // if ( substep % update_pick_substeps == 0 ) {
    //     check_update_pick();
    // }

    bool ground = geo->ground;
    float ground_f = max(0.f, (get_global_parameter("ground_f", 1e3)));
    // Velocity decay, parameter driven (was a fixed exp(-h * 0.5)).
    const float damping = max(0.f, velocity_damping(nullptr, 0.f));
    step_end_kernel<<<(n + block - 1) / block, block>>>(
        q, v, q_prev, f, f_elastic, mask, mass_inv, obj_data, vertices_obj,
        geo->gravity, h, max_vel, ground, ground_f, damping, n);
    // Seam projection and its matching velocity pass (geometry-side, see
    // SolverPDNewton::step). It has to run *after* `step_end_kernel` here: the
    // explicit integrator overwrites the position from the velocity, so a
    // projection before it would be discarded.
    geo->project_stitches();
    geo->average_stitch_cluster_velocities();

    // if ( substep % LCP_substeps == 0 ) {
    //     collision_LCP_postprocess_unified(vertices_world.data().get());
    // }

}
