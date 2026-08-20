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
    const float3* __restrict__ pos_ine,
    const float3* __restrict__ other_forces,
    const float3* __restrict__ elastic_forces,
    const char* __restrict__ vertices_mask,
    const float* __restrict__ mass_inv,
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
            auto x_old = vertices_world[i];
            float mi = mass_inv[i];
            float3 x = pos_ine[i] + (other_forces[i] + elastic_forces[i]) * mi * h * h;
            float3 v = (x - x_old) / h;
            if ( ground ) {
                float min_z = obj_data[vertices_obj[i]].thickness;
                if ( x.z <= min_z ) {
                    x.z = min_z;
                    v.z = 0.f;
                    v = v * expf(-h * ground_f);
                }
            }
            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            v = v * expf(-h * 0.5f);
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
    float3* q_inertia = geo->pos_inertia.data().get();
    float3* v = geo->velocities.data().get();
    float3* f = geo->forces.data().get();
    float3* f_elastic = geo->elastic_forces.data().get();
    int2* edges = geo->edges.data().get();
    int2* e2t = geo->e2t.data().get();
    int2* eop = geo->edge_opposite_points.data().get();
    float* rest_thetas = geo->rest_thetas.data().get();
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

    forward_step<<<(n + block - 1) / block, block>>>(
        v, nullptr, mass_inv,
        nullptr, f_elastic,
        mask, q_inertia, nullptr, q, nullptr,
        static_diags,
        h, 1e2, geo->gravity, false, n);
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
    accumulate_spring_forces<<<(n + block - 1) / block, block>>>(nullptr, nullptr,
        f_elastic, nullptr, q, edges,
        geo->edge_lengths.data().get(),
        geo->obj_data.data().get(), geo->vertices_obj.data().get(),
        n);
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

    n = params.nb_all_cloth_edges;
    if ( geo->bending_model == BendingModel::IBM_quadratic )
        compute_quadratic_bending_IBM<<< (n + block - 1) / block, block>>>(
            nullptr, nullptr, nullptr,
            f, nullptr,
            geo->IBM_q.data().get(),
            q, edges, e2t, tri_edges, eop,
            n, 0.2f);
    else if ( geo->bending_model == BendingModel::DiscreteShells_GN
        || geo->bending_model == BendingModel::DiscreteShells_AOGS )
        // The forces are the same
        compute_dihedral_bending_GN<<<(n + block - 1), block>>>(
            nullptr, nullptr, nullptr,
            f, q, edges, e2t, rest_thetas,
            tri_edges, eop,
            n, 0.2);
    geo->accumulate_sewing_force();
    geo->get_contact().accumulate_contact_force(f, nullptr);
    // update substep end

    n = params.nb_all_cloth_vertices;
    // if ( substep % update_pick_substeps == 0 ) {
    //     check_update_pick();
    // }

    bool ground = geo->ground;
    float ground_f = max(0.f, (get_global_parameter("ground_f", 1e3)));
    step_end_kernel<<<(n + block - 1) / block, block>>>(
        q, v, q_inertia, f, f_elastic, mask, mass_inv, obj_data, vertices_obj, h, max_vel, ground, ground_f, n);

    // if ( substep % LCP_substeps == 0 ) {
    //     collision_LCP_postprocess_unified(vertices_world.data().get());
    // }

}
