#include "solver_XPBD.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"


static __global__ void step_end_kernel(
    float3* __restrict__ pos_world,
    float3* __restrict__ velocities,
    const float3* __restrict__ pos_prev,
    const float3* __restrict__ dx,
    const char* __restrict__ vertices_mask,
    const float h,
    const float max_velocity,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !vertices_mask[i] ) {
            float3 x = pos_world[i] + dx[i];
            float3 v = (x - pos_prev[i]) / h;

            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            v = v * expf(-h * 0.5f);
            velocities[i] = v;
            pos_world[i] = x;
            // if ( ground ) {
            //     float min_z = obj_data[vertices_obj[i]].thickness;
            //     if ( x.z <= min_z ) {
            //         x.z = min_z;
            //         v.z = 0.f;
            //         pos_world[i] = x;
            //     }
            // }
        }
        else {
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
    }
}

void SolverXPBD::init() {
    SolverBase::init();
    auto& params = *simulator->get_geo_params();
    auto* geo = simulator->get_geo();

    delta.resize(params.nb_all_vertices);

}
__global__ void xpbd_forward_step(
    const float3* __restrict__ pos,
    const float3* __restrict__ vel,
    const float* __restrict__ inv_mass,
    const float3* __restrict__ external_force,
    const char* __restrict__ mask,
    float3* __restrict__ inertia_out,
    float dt,
    float3 gravity,
    int num_vertices
) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i >= num_vertices ) return;

    float3 p = pos[i];
    float3 v = vel[i];
    float3 g_ext = gravity;
    float im = inv_mass[i];
    if ( external_force ) {
        g_ext += external_force[i] * im;
    }
    if ( mask[i] || im == 0.0f ) {
        inertia_out[i] = p;
    }
    else {
        inertia_out[i] = p + v * dt + g_ext * dt * dt;
    }
}

__global__ void xpbd_solve_springs_kernel(
    float* __restrict__ lambdas,                   // accumulated multipliers
    float3* __restrict__ delta,                    // position correction accumulator
    const float3* __restrict__ pos_pred,           // predicted positions
    const float3* __restrict__ velocities,         // current velocities
    const float* __restrict__ inv_mass,            // inverse mass per vertex
    const int2* __restrict__ edges,                // edge vertex pairs
    const float* __restrict__ edge_lengths,        // rest length per edge
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float damping,             // kd
    const float h,                                 // time step
    const int num_edges                            // number of distance constraints
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x;
          i < num_edges;
          i += blockDim.x * gridDim.x ) {

        auto [v0, v1] = edges[i];

        // Current predicted positions and velocities
        float3 p0 = pos_pred[v0];
        float3 p1 = pos_pred[v1];
        float3 vv0 = velocities[v0];
        float3 vv1 = velocities[v1];

        float3 e = p0 - p1;
        float len = norm(e);
        if ( len < 1e-9f ) return;

        float3 n = e / len;   // direction from p1 to p0

        // Constraint: C = |p0 - p1| - rest_length
        float C = len - edge_lengths[i];

        // Gradients
        float3 grad0 = n;
        float3 grad1 = -n;

        float w0 = inv_mass[v0];
        float w1 = inv_mass[v1];
        float denom = w0 + w1;

        if ( denom <= 0.0f ) return;

        float3 ks = obj_data[vertices_obj[v0]].stretch;
        const float youngs = 4e2;
        const float ke = youngs * (ks.x + ks.y + ks.z) * 0.333f;
        const float kd = damping;
        if ( ke <= 0.0f ) return;

        float alpha = 1.0f / (ke * h * h);
        float gamma = kd / (ke * h);

        // Velocity projection onto gradient
        float grad_dot_v = h * dot(grad0, vv0 - vv1);
        float delta_lambda;
        if ( lambdas ) {
            delta_lambda = -(C + alpha * lambdas[i] + gamma * grad_dot_v)
                / ((1.0f + gamma) * denom + alpha);
            lambdas[i] += delta_lambda;
        }
        else { // only one-step history
            delta_lambda = -(C + gamma * grad_dot_v) / ((1.0f + gamma) * denom + alpha);
        }

        // Apply weighted correction
        atomicAddFloat3(&delta[v0], w0 * delta_lambda * grad0);
        atomicAddFloat3(&delta[v1], w1 * delta_lambda * grad1);
    }
}

__global__ void xpbd_solve_vf_contacts_kernel(
    const float3* __restrict__ pos,                // current predicted positions
    const float3* __restrict__ vel,                // current velocities
    const float* __restrict__ inv_mass,            // inverse mass per vertex
    const int* __restrict__ broad_phase_pairs,     // per‑vertex contact list: [count, face0, face1, …]
    const int broad_phase_size,                    // max contacts per vertex
    const int3* __restrict__ tri_indices,          // vertex indices of each triangle
    const float3* __restrict__ vertex_normals,     // smoothed vertex normals for direction
    const ObjectDataInput* __restrict__ obj_data,   // object data (thickness, layer, etc.)
    const int* __restrict__ vertices_obj,          // object index per vertex
    const float dt,                                // time step
    const float compliance,                        // 1 / stiffness
    const float relaxation,                        // global relaxation factor
    const bool ground,                             // enable ground plane at z = thickness
    const float ground_f,
    float3* __restrict__ delta,                    // position correction accumulator
    const int num_vertices                         // total vertices to process
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;

    const float3 x0 = pos[vid];
    const float3 v0 = vel[vid];
    const float w0 = inv_mass[vid];                 // may be 0 for fixed vertices
    const float thickness0 = obj_data[vertices_obj[vid]].thickness;
    const float friction_mu0 = obj_data[vertices_obj[vid]].friction;
    const float3 normal_v = vertex_normals[vid];
    const int layer0 = obj_data[vertices_obj[vid]].collision_layer;

    float3 dx0 = make_float3(0.0f, 0.0f, 0.0f);
    bool is_collided = false;

    // --- Ground contact ---
    if ( ground && w0 > 0.0f ) {
        const float d_ground = x0.z - thickness0;
        if ( d_ground < 0.0f ) {
            const float3 n = make_float3(0.0f, 0.0f, 1.0f);

            // Normal constraint: C = z - thickness
            float C_n = d_ground;
            float denom_n = w0;
            float alpha = 1.f / (ground_f * dt * dt);
            float dlambda_n = -C_n / (denom_n + alpha);

            dx0 += n * dlambda_n;
            is_collided = true;

            // Tangential friction
            float3 v_rel = v0;
            float3 vt = v_rel - n * dot(v_rel, n);
            float speed = norm(vt);
            if ( speed > 1e-9f ) {
                float3 t_dir = vt / speed;
                float C_t = speed * dt;
                float denom_t = w0;
                float dlambda_t = -C_t / (denom_t + alpha);
                float max_friction = friction_mu0 * fabsf(dlambda_n);
                dlambda_t = clamp(dlambda_t, -max_friction, max_friction);

                dx0 += t_dir * dlambda_t;
            }
        }
    }

    // --- Vertex‑triangle contacts ---
    const int* pairs = &broad_phase_pairs[vid * broad_phase_size];
    const int count = pairs[0];

    for ( int j = 1; j <= count; ++j ) {
        int fid = pairs[j];
        float sign = (fid > 0) ? 1.0f : -1.0f;
        fid = abs(fid);
        const int3 tri = tri_indices[fid];
        const int i1 = tri.x, i2 = tri.y, i3 = tri.z;

        const float3 x1 = pos[i1], x2 = pos[i2], x3 = pos[i3];
        const float3 v1 = vel[i1], v2 = vel[i2], v3 = vel[i3];
        const float w1 = inv_mass[i1], w2 = inv_mass[i2], w3 = inv_mass[i3];

        float3 normal = cross(x2 - x1, x3 - x1);
        float normal_len = norm(normal);
        if ( normal_len < 1e-8f ) continue;
        normal = normal / normal_len;

        int layer1 = obj_data[vertices_obj[i1]].collision_layer;
        if ( layer1 == layer0 ) {
            normal = normal * sign;
        }
        else if ( layer0 < layer1 ) {
            if ( dot(normal, normal_v) > 0.0f ) normal = -normal;
        }

        float dist = dot(x0 - x1, normal);
        float thickness = thickness0 + obj_data[vertices_obj[i1]].thickness;
        if ( dist > thickness ) continue;

        float3 closest_pt = x0 - dist * normal;
        float u, v, w;
        barycentric(x1, x2, x3, closest_pt, u, v, w);
        if ( u < 0.0f || v < 0.0f || w < 0.0f ) continue;

        // --- Normal constraint ---
        float C_n = dist - thickness;                // negative when penetrating
        float3 grad0_n = normal;
        float3 grad1_n = -u * normal;
        float3 grad2_n = -v * normal;
        float3 grad3_n = -w * normal;

        float denom_n = w0 + w1 * u * u + w2 * v * v + w3 * w * w;
        if ( denom_n <= 0.0f ) continue;

        float alpha = compliance / (dt * dt);
        float dlambda_n = -C_n / (denom_n + alpha);

        float3 dx1 = dlambda_n * grad1_n;
        float3 dx2 = dlambda_n * grad2_n;
        float3 dx3 = dlambda_n * grad3_n;
        dx0 += dlambda_n * grad0_n;
        is_collided = true;

        // --- Tangential friction ---
        float friction_mu = (friction_mu0 + obj_data[vertices_obj[i1]].friction) * 0.5f;
        float3 v_contact = v0 - (v1 * u + v2 * v + v3 * w);
        float3 vt = v_contact - normal * dot(v_contact, normal);
        float speed = norm(vt);
        if ( speed > 1e-9f ) {
            float3 t_dir = vt / speed;
            float C_t = speed * dt;
            float3 grad0_t = t_dir;
            float3 grad1_t = -u * t_dir;
            float3 grad2_t = -v * t_dir;
            float3 grad3_t = -w * t_dir;

            float denom_t = w0 + w1 * u * u + w2 * v * v + w3 * w * w;
            if ( denom_t > 0.0f ) {
                float dlambda_t = -C_t / (denom_t + alpha);
                float max_friction = friction_mu * fabsf(dlambda_n);
                dlambda_t = clamp(dlambda_t, -max_friction, max_friction);

                dx0 += dlambda_t * grad0_t;
                dx1 += dlambda_t * grad1_t;
                dx2 += dlambda_t * grad2_t;
                dx3 += dlambda_t * grad3_t;
            }
        }

        atomicAddFloat3(&delta[i1], dx1 * w1 * relaxation);
        atomicAddFloat3(&delta[i2], dx2 * w2 * relaxation);
        atomicAddFloat3(&delta[i3], dx3 * w3 * relaxation);
    }

    if ( is_collided && w0 > 0.0f ) {
        atomicAddFloat3(&delta[vid], dx0 * w0 * relaxation);
    }
}

__global__ void xpbd_solve_ee_contacts_kernel(
    const float3* __restrict__ pos,                // current predicted positions
    const float3* __restrict__ vel,                // current velocities
    const float* __restrict__ inv_mass,            // inverse mass per vertex
    const int* __restrict__ broad_phase_pairs,     // per‑edge contact list
    const int broad_phase_size,                    // max contacts per edge
    const int2* __restrict__ edges,                // edge vertex pairs
    const float3* __restrict__ edge_normals,        // smoothed edge normals
    const ObjectDataInput* __restrict__ obj_data,   // object data (thickness, layer, etc.)
    const int* __restrict__ vertices_obj,          // object index per vertex
    const float dt,                                // time step
    const float compliance,                        // 1 / stiffness
    const float relaxation,                        // global relaxation factor
    float3* __restrict__ delta,                    // position correction accumulator
    const int num_edges                            // total edges to process
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( eid >= num_edges ) return;

    const int2 edge = edges[eid];
    const int ia = edge.x, ib = edge.y;
    const float wa = inv_mass[ia], wb = inv_mass[ib];

    const float3 p0 = pos[ia], p1 = pos[ib];
    const float3 v0 = vel[ia], v1 = vel[ib];
    const float thickness0 = obj_data[vertices_obj[ia]].thickness;
    const float friction_mu0 = obj_data[vertices_obj[ia]].friction;
    const int layer0 = obj_data[vertices_obj[ia]].collision_layer;
    const float3 edge_normal0 = edge_normals[eid];

    float3 dx_a = make_float3(0.0f, 0.0f, 0.0f);
    float3 dx_b = make_float3(0.0f, 0.0f, 0.0f);
    bool is_collided = false;

    const int* pairs = &broad_phase_pairs[eid * broad_phase_size];
    const int count = pairs[0];

    for ( int j = 1; j <= count; ++j ) {
        int eid2 = pairs[j];
        float sign = (eid2 > 0) ? 1.0f : -1.0f;
        eid2 = abs(eid2);
        const int2 edge2 = edges[eid2];
        const int ic = edge2.x, id = edge2.y;

        const float3 q0 = pos[ic], q1 = pos[id];
        const float3 vq0 = vel[ic], vq1 = vel[id];
        const float wc = inv_mass[ic], wd = inv_mass[id];

        float s, t;
        float3 ab;
        segment_segment_closest_robust(p0, p1, q0, q1, s, t, ab);
        if ( s <= 0.0f || s >= 1.0f || t <= 0.0f || t >= 1.0f ) continue;

        ab = -ab;
        float dist = norm(ab);
        float3 normal;
        if ( dist < 1e-16f ) {
            normal = edge_normal0;
            ab = normal;
        }
        else {
            normal = ab / dist;
        }

        // Original direction logic (exactly as in compute_ee_force)
        int layer1 = obj_data[vertices_obj[ic]].collision_layer;
        if ( layer1 == layer0 ) {
            float sign_new = (dot(ab, edge_normal0) < 0.0f) ? 1.0f : -1.0f;
            sign *= sign_new;
            if ( sign < 0.0f ) {
                dist = -dist;
                normal = -normal;
            }
        }
        else {
            bool reverse = false;
            if ( layer0 < layer1 ) {
                reverse = (dot(normal, edge_normal0) > 0.0f);
            }
            else {
                const float3 edge_normal1 = edge_normals[eid2];
                reverse = (dot(normal, edge_normal1) < 0.0f);
            }
            if ( reverse ) {
                normal = -normal;
                dist = -dist;
            }
        }

        float thickness = thickness0 + obj_data[vertices_obj[ic]].thickness;
        if ( dist > thickness ) continue;

        // --- Normal constraint ---
        float C_n = dist - thickness;
        float3 grad_a = normal * (1.0f - s);
        float3 grad_b = normal * s;
        float3 grad_c = -normal * (1.0f - t);
        float3 grad_d = -normal * t;

        float denom_n = wa * (1.0f - s) * (1.0f - s) +
            wb * s * s +
            wc * (1.0f - t) * (1.0f - t) +
            wd * t * t;
        if ( denom_n <= 0.0f ) continue;

        float alpha = compliance / (dt * dt);
        float dlambda_n = -C_n / (denom_n + alpha);

        float3 dx_c = dlambda_n * grad_c;
        float3 dx_d = dlambda_n * grad_d;
        dx_a += dlambda_n * grad_a;
        dx_b += dlambda_n * grad_b;
        is_collided = true;

        // --- Tangential friction ---
        float friction_mu = (friction_mu0 + obj_data[vertices_obj[ic]].friction) * 0.5f;
        float3 v_contact_a = v0 * (1.0f - s) + v1 * s;
        float3 v_contact_b = vq0 * (1.0f - t) + vq1 * t;
        float3 v_rel = v_contact_a - v_contact_b;
        float3 vt = v_rel - normal * dot(v_rel, normal);
        float speed = norm(vt);
        if ( speed > 1e-9f ) {
            float3 t_dir = vt / speed;
            float C_t = speed * dt;
            float3 grad_a_t = t_dir * (1.0f - s);
            float3 grad_b_t = t_dir * s;
            float3 grad_c_t = -t_dir * (1.0f - t);
            float3 grad_d_t = -t_dir * t;

            float denom_t = wa * (1.0f - s) * (1.0f - s) +
                wb * s * s +
                wc * (1.0f - t) * (1.0f - t) +
                wd * t * t;
            if ( denom_t > 0.0f ) {
                float dlambda_t = -C_t / (denom_t + alpha);
                float max_friction = friction_mu * fabsf(dlambda_n);
                dlambda_t = clamp(dlambda_t, -max_friction, max_friction);

                dx_a += dlambda_t * grad_a_t;
                dx_b += dlambda_t * grad_b_t;
                dx_c += dlambda_t * grad_c_t;
                dx_d += dlambda_t * grad_d_t;
            }
        }

        atomicAddFloat3(&delta[ic], dx_c * wc * relaxation);
        atomicAddFloat3(&delta[id], dx_d * wd * relaxation);
    }

    if ( is_collided ) {
        if ( wa > 0.0f ) atomicAddFloat3(&delta[ia], dx_a * wa * relaxation);
        if ( wb > 0.0f ) atomicAddFloat3(&delta[ib], dx_b * wb * relaxation);
    }
}


void SolverXPBD::step(float h) {
    // XPBD using Jacobi iteration
    auto& params = *simulator->get_geo_params();
    auto* geo = simulator->get_geo();
    int n = params.nb_all_vertices;
    if ( n <= 0 ) return;

    int block = 256;

    float3* q = geo->pos_world.data().get();
    const float3* q_prev = geo->pos_step_prev.data().get();
    float3* q_inertia = geo->pos_inertia.data().get();
    float3* dx = this->delta.data().get();
    float3* v = geo->velocities.data().get();
    float3* f = geo->forces.data().get();
    int2* edges = geo->edges.data().get();
    int3* tri_edges = geo->triangles.data().get();
    int3* tris = geo->triangle_indices.data().get();
    char* mask = geo->vertices_mask.data().get();
    float* mass = geo->masses.data().get();
    float* mass_inv = geo->mass_inv.data().get();
    auto* obj_data = geo->obj_data.data().get();
    int* vertices_obj = geo->vertices_obj.data().get();

    float max_vel = max(0.f, get_global_parameter("max_vel", 1000));

    cudaMemsetAsync(dx, 0, params.nb_all_vertices * sizeof(float3));

    xpbd_forward_step<<<(n + block - 1) / block, block>>>(
        q, v, mass_inv,
        nullptr, mask, q_inertia, 
        h, geo->gravity, n);
    cudaMemcpyAsync(q, q_inertia,
        params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
    n = params.nb_all_cloth_edges;
    float damping = max(0.f, get_global_parameter("xpbd_damping", 0.f));
    xpbd_solve_springs_kernel<<<(n + block - 1) / block, block>>>(nullptr, dx,
        q, v, mass_inv, edges,
        geo->edge_lengths.data().get(),
        geo->obj_data.data().get(), geo->vertices_obj.data().get(),
        damping, h, n);
    // geo->accumulate_sewing_force();

    // contact
    float vf_force_k = max(0.f, geo->get_global_parameter("vf_force_k", 0.2f));
    float vf_ground_k = max(0.f, geo->get_global_parameter("vf_ground_k", 0.2f));
    float ee_force_k = max(0.f, geo->get_global_parameter("ee_force_k", 0.2f));
    float ef_force_k = max(0.f, geo->get_global_parameter("ef_force_k", 0.2f));

    float relaxation = max(0.f, geo->get_global_parameter("xpbd_relaxation", 0.9f));
    n = params.nb_all_vertices;
    auto& contact = geo->get_contact();
    xpbd_solve_vf_contacts_kernel<<<(n + block - 1) / block, block>>>(
        q, v, mass_inv,
        contact.broad_phase_ef.data().get(), broad_phase_size,
        tris, geo->vertex_normals.data().get(),
        obj_data, vertices_obj, h, 1.f / vf_force_k, relaxation,
        geo->ground, vf_ground_k, dx, n);

    n = params.nb_all_edges;
    xpbd_solve_ee_contacts_kernel<<<(n + block - 1) / block, block>>>(
        q, v, mass_inv,
        contact.broad_phase_ee.data().get(), broad_phase_size,
        edges, geo->edge_normals.data().get(),
        obj_data, vertices_obj, h, 1.f / ee_force_k, relaxation,
        dx, n);
    // update substep end

    step_end_kernel<<<(n + block - 1) / block, block>>>(
        q, v, q_prev, dx, mask, h, max_vel, n);

}
