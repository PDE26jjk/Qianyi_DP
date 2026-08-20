#include "solver_XPBD.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "contact/collision_detection.cuh"
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
    lambdas.resize(max(params.nb_all_edges, params.nb_all_cloth_triangles * 3));
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
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if ( i >= num_edges ) return;


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
    const float ke = base_spring_stiffness * (ks.x + ks.y + ks.z) * 0.333f;
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
__global__ void xpbd_solve_triangle_fem_kernel(
    float* __restrict__ lambdas,                   // accumulated multipliers
    float3* __restrict__ delta,                    // position correction accumulator
    const float3* __restrict__ pos,                // current predicted positions
    const float3* __restrict__ vel,                // current velocities (needed if damping > 0)
    const float* __restrict__ inv_mass,            // inverse mass per vertex
    const int3* __restrict__ triangles,            // [numTri] triple of edge indices
    const Mat2* __restrict__ Dms,                  // material matrices (Dm, NOT inverse)
    const ObjectDataInput* __restrict__ obj_data,   // for per‑object stretch parameters
    const int* __restrict__ vertices_obj,          // object index per vertex
    const float damping,                           // optional global material damping factor (0 = no damping)
    const float relaxation,                        // global relaxation factor
    const float dt,                                // time step
    const int num_triangles
) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( tid >= num_triangles ) return;

    // ----- reconstruct triangle vertices -----
    const int3 tri = triangles[tid];
    const int v0 = tri.x;
    const int v1 = tri.y;
    const int v2 = tri.z;

    // ----- material and geometric data -----
    const Mat2 Dm = Dms[tid];
    const float Dm_det = Dm.det();
    // if ( Dm_det <= 0.0f ) return;             // degenerate rest triangle
    const float area = fabsf(Dm_det) * 0.5f;
    const Mat2 Dm_inv = Dm.inverse();

    // stiffnesses from object data (stretch.x = u, .y = v, .z = shear)
    const float3 stretch = obj_data[vertices_obj[v0]].stretch * 1.38e3f;
    const float ku = stretch.x;
    const float kv = stretch.y;
    const float ks = stretch.z;

    // ----- positions, velocities and inverse masses -----
    const float3 x0 = pos[v0], x1 = pos[v1], x2 = pos[v2];
    const float3 v0_vec = vel[v0], v1_vec = vel[v1], v2_vec = vel[v2];
    const float w0 = inv_mass[v0], w1 = inv_mass[v1], w2 = inv_mass[v2];
    if ( w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f ) return;

    // ----- deformation gradients -----
    const float3 e1 = x1 - x0;
    const float3 e2 = x2 - x0;

    const float m00 = Dm_inv.r[0].x; // Dm_inv[0,0]
    const float m10 = Dm_inv.r[0].y; // Dm_inv[0,1]
    const float m01 = Dm_inv.r[1].x; // Dm_inv[1,0]
    const float m11 = Dm_inv.r[1].y; // Dm_inv[1,1]

    const float3 wu = e1 * m00 + e2 * m01;
    const float3 wv = e1 * m10 + e2 * m11;

    const float wu_norm = norm(wu);
    const float wv_norm = norm(wv);
    if ( wu_norm < 1e-8f || wv_norm < 1e-8f ) return;

    const float3 wu_hat = wu / wu_norm;
    const float3 wv_hat = wv / wv_norm;
    const float wu_dot_wv = dot(wu_hat, wv_hat);

    // accumulated vertex corrections
    float3 dx0 = make_float3(0.0f, 0.0f, 0.0f);
    float3 dx1 = make_float3(0.0f, 0.0f, 0.0f);
    float3 dx2 = make_float3(0.0f, 0.0f, 0.0f);

    // common constants
    const float dt2 = dt * dt;

    // ----- stretch in u -----
    if ( ku > 0.0f ) {
        const float Cu = wu_norm - 1.0f;
        const float alpha = 1.0f / (ku * area * dt2);
        const float gamma = (damping > 0.0f) ? (damping / (ku * dt)) : 0.0f;

        // gradients
        const float3 g1 = wu_hat * m00;
        const float3 g2 = wu_hat * m01;
        const float3 g0 = -(g1 + g2);

        const float denom = w0 * (m00 + m01) * (m00 + m01) +   // |g0|^2 = (m00+m01)^2
            w1 * m00 * m00 +                    // |g1|^2 = m00^2
            w2 * m01 * m01;                     // |g2|^2 = m01^2
        if ( denom > 0.0f ) {
            float lambda = 0.0f;
            if ( lambdas ) lambda = lambdas[tid * 3 + 0];   // u constraint
            float grad_dot_v = 0.f;
            if ( gamma > 0.0f ) {
                grad_dot_v = dt * (dot(g0, v0_vec) + dot(g1, v1_vec) + dot(g2, v2_vec));
            }
            float delta_lambda = -(Cu + alpha * lambda + gamma * grad_dot_v)
                / ((1.0f + gamma) * denom + alpha);
            if ( lambdas ) lambdas[tid * 3 + 0] += delta_lambda;

            dx0 += delta_lambda * g0;
            dx1 += delta_lambda * g1;
            dx2 += delta_lambda * g2;
        }
    }

    // ----- stretch in v -----
    if ( kv > 0.0f ) {
        const float Cv = wv_norm - 1.0f;
        const float alpha = 1.0f / (kv * area * dt2);
        const float gamma = (damping > 0.0f) ? (damping / (kv * dt)) : 0.0f;

        const float3 g1 = wv_hat * m10;
        const float3 g2 = wv_hat * m11;
        const float3 g0 = -(g1 + g2);

        const float denom = w0 * (m10 + m11) * (m10 + m11) +
            w1 * m10 * m10 +
            w2 * m11 * m11;
        if ( denom > 0.0f ) {
            float lambda = 0.0f;
            if ( lambdas ) lambda = lambdas[tid * 3 + 1];   // v constraint
            float grad_dot_v = 0.0f;
            if ( gamma > 0.0f ) {
                grad_dot_v = dt * (dot(g0, v0_vec) + dot(g1, v1_vec) + dot(g2, v2_vec));
            }
            float delta_lambda = -(Cv + alpha * lambda + gamma * grad_dot_v)
                / ((1.0f + gamma) * denom + alpha);
            if ( lambdas ) lambdas[tid * 3 + 1] += delta_lambda;

            dx0 += delta_lambda * g0;
            dx1 += delta_lambda * g1;
            dx2 += delta_lambda * g2;
        }
    }

    // ----- shear -----
    if ( ks > 0.0f ) {
        const float3 wv_proj = (wv_hat - wu_hat * wu_dot_wv) / wu_norm;
        const float3 wu_proj = (wu_hat - wv_hat * wu_dot_wv) / wv_norm;

        const float Cs = wu_dot_wv;
        const float alpha = 1.0f / (ks * area * dt2);
        const float gamma = (damping > 0.0f) ? (damping / (ks * dt)) : 0.0f;

        const float3 g1 = wv_proj * m00 + wu_proj * m10;
        const float3 g2 = wv_proj * m01 + wu_proj * m11;
        const float3 g0 = -(g1 + g2);

        const float denom = w0 * dot(g0, g0) +
            w1 * dot(g1, g1) +
            w2 * dot(g2, g2);
        if ( denom > 0.0f ) {
            float lambda = 0.0f;
            if ( lambdas ) lambda = lambdas[tid * 3 + 2];   // shear constraint
            float grad_dot_v = 0.0f;
            if ( gamma > 0.0f ) {
                grad_dot_v = dt * (dot(g0, v0_vec) + dot(g1, v1_vec) + dot(g2, v2_vec));
            }
            float delta_lambda = -(Cs + alpha * lambda + gamma * grad_dot_v)
                / ((1.0f + gamma) * denom + alpha);
            if ( lambdas ) lambdas[tid * 3 + 2] = lambda + delta_lambda;

            dx0 += delta_lambda * g0;
            dx1 += delta_lambda * g1;
            dx2 += delta_lambda * g2;
        }
    }

    // ----- apply corrections (single atomic per vertex) -----
    if ( w0 > 0.0f ) atomicAddFloat3(&delta[v0], dx0 * w0 * relaxation);
    if ( w1 > 0.0f ) atomicAddFloat3(&delta[v1], dx1 * w1 * relaxation);
    if ( w2 > 0.0f ) atomicAddFloat3(&delta[v2], dx2 * w2 * relaxation);
}

__global__ void xpbd_solve_vf_contacts_kernel(
    const float3* __restrict__ pos_prev,                
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
    const float stiffness,                        // 1 / compliance
    const float relaxation,                        // global relaxation factor
    const bool ground,                             // enable ground plane at z = thickness
    const float ground_f,
    float3* __restrict__ delta,                    // position correction accumulator
    // const int active_vertices_size,                       
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

    // float3 dx0_dyn = x0 - pos_prev[vid];
    float3 dx0 = make_float3(0.0f, 0.0f, 0.0f);
    bool is_collided = false;
    // bool is_active = vid < active_vertices_size && w0 > 0.0f;
    // --- Ground contact ---
    if ( ground && ground_f > 0.f && w0 > 0.0f ) {
        const float d_ground = x0.z - thickness0;
        if ( d_ground < 0.0f ) {
            const float3 n = make_float3(0.0f, 0.0f, 1.0f);

            // Normal constraint: C = z - thickness
            float C_n = d_ground;
            float denom_n = w0;
            float alpha = 1.f / (ground_f * dt * dt);
            float dlambda_n = -C_n / (denom_n + alpha);

            dx0 = n * dlambda_n;
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
        if ( stiffness < 1e-12f ) break;
        int fid = pairs[j];
        float sign = (fid > 0) ? 1.0f : -1.0f;
        fid = abs(fid);
        const int3 tri = tri_indices[fid];
        const int i1 = tri.x, i2 = tri.y, i3 = tri.z;

        const float3 x1 = pos[i1], x2 = pos[i2], x3 = pos[i3];

        float3 normal;
        float u, v, w, pen;
        float thickness = thickness0 + obj_data[vertices_obj[i1]].thickness;
        if ( !compute_point_triangle_contact(
            x0, x1, x2, x3,
            thickness,
            layer0 - obj_data[vertices_obj[i1]].collision_layer,
            sign, normal_v,
            normal, u, v, w, pen)
        ) {
            continue;
        }

        // --- Normal constraint ---
        float C_n = -pen;                // negative when penetrating
        float3 grad0_n = normal;
        float3 grad1_n = -u * normal;
        float3 grad2_n = -v * normal;
        float3 grad3_n = -w * normal;

        const float w1 = inv_mass[i1], w2 = inv_mass[i2], w3 = inv_mass[i3];
        float denom_n = w0 + w1 * u * u + w2 * v * v + w3 * w * w;
        if ( denom_n <= 0.0f ) continue;

        float alpha = 1.f / (stiffness * dt * dt);
        // float alpha = 0.f;
        float dlambda_n = -C_n / (denom_n + alpha);

        float3 dx1 = dlambda_n * grad1_n;
        float3 dx2 = dlambda_n * grad2_n;
        float3 dx3 = dlambda_n * grad3_n;
        dx0 += dlambda_n * grad0_n;
        is_collided = true;

        // --- Tangential friction ---
        const float3 v1 = vel[i1], v2 = vel[i2], v3 = vel[i3];
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

        float thickness = thickness0 + obj_data[vertices_obj[ic]].thickness;
        // if ( dist > thickness ) continue;
        float s, t;
        float3 normal;
        float pen;
        if ( !compute_edge_edge_contact(
            p0, p1, q0, q1,
            thickness,
            layer0 - obj_data[vertices_obj[ic]].collision_layer,
            sign, edge_normal0, edge_normals[eid2],
            s, t, normal, pen)
        ) {
            continue;
        }

        // --- Normal constraint ---
        float C_n = -pen;
        float3 grad_a = normal * (1.0f - s);
        float3 grad_b = normal * s;
        float3 grad_c = -normal * (1.0f - t);
        float3 grad_d = -normal * t;

        const float wc = inv_mass[ic], wd = inv_mass[id];
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
        const float3 vq0 = vel[ic], vq1 = vel[id];
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
static __global__ void applay_delta_xpbd(
    float3* __restrict__ pos_world,
    float3* __restrict__ dx,
    const char* __restrict__ mask,
    const float max_dx,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !mask[i] ) {
            float3 delta = dx[i];
            float3 x = pos_world[i] + delta;
            if ( isinf(x.x) || isinf(x.y) || isinf(x.z) ||
                isnan(x.x) || isnan(x.y) || isnan(x.z) ) {
                if ( i % 100 == 0 ) {
                    printf("step_end_xpbd ERROR!!! %f \n", len_sq(x));
                }
                return;
            }
            float len = norm(delta);
            if ( len > max_dx ) {
                delta = delta * (max_dx / len);
                x = pos_world[i] + delta;
            }
            pos_world[i] = x;
        }
        dx[i] = make_float3(0.0f, 0.f, 0.f);
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
    int dynamics_iters = max(1, (int)get_global_parameter("xpbd_dynamics_iters", 1));
    int use_lambdas = (int)geo->get_global_parameter("xpbd_use_lambdas", 1);

    float* lambdas = (use_lambdas && dynamics_iters > 1) ? this->lambdas.data().get() : nullptr;

    xpbd_forward_step<<<(n + block - 1) / block, block>>>(
        q, v, mass_inv,
        nullptr, mask, q_inertia,
        h, geo->gravity, n);

    cudaMemsetAsync(dx, 0, params.nb_all_vertices * sizeof(float3));
    int iters = max(1, (int)get_global_parameter("xpbd_iters", 10));
    cudaMemcpyAsync(q, q_inertia,
        params.nb_all_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);

    float max_vel = max(0.f, get_global_parameter("max_vel", 1000));
    float max_dx = max_vel * h;
    float vf_force_k = max(0.f, geo->get_global_parameter("vf_force_k", 0.2f));
    float vf_ground_k = max(0.f, geo->get_global_parameter("vf_ground_k", 0.2f));
    float ee_force_k = max(0.f, geo->get_global_parameter("ee_force_k", 0.2f));
    float ef_force_k = max(0.f, geo->get_global_parameter("ef_force_k", 0.2f));
    float damping = max(0.f, get_global_parameter("xpbd_damping", 0.f));
    float relaxation = max(0.f, geo->get_global_parameter("xpbd_relaxation", 0.9f));


    auto& contact = geo->get_contact();
    for ( int i = 0; i < iters; i++ ) {
        if ( lambdas )
            cudaMemsetAsync(lambdas, 0, this->lambdas.size() * sizeof(float));
        for ( int j = 0; j < dynamics_iters; j++ ) {
            if ( geo->constitutive_model == ConstitutiveModel::SpringMass ) {
                n = params.nb_all_cloth_edges;
                xpbd_solve_springs_kernel<<<(n + block - 1) / block, block>>>(lambdas, dx,
                    q, v, mass_inv, edges,
                    geo->edge_lengths.data().get(),
                    obj_data, vertices_obj,
                    damping, h, n);
            }
            else if ( geo->constitutive_model == ConstitutiveModel::FEM_BW ) {
                n = params.nb_all_cloth_triangles;
                xpbd_solve_triangle_fem_kernel<<<(n + block - 1) / block, block>>>(
                    lambdas, dx,
                    q, v, mass_inv, tris,
                    geo->Dms.data().get(),
                    obj_data, vertices_obj,
                    damping, relaxation, h, n);
            }
        }
        // contact.ccd_truncation_traverse_bvh(q_prev, q);
        // geo->accumulate_sewing_force();
        applay_delta_xpbd<<<(n + block - 1) / block, block>>>(
            q, dx, mask, max_dx, n);
        // contact
        {
            n = params.nb_all_vertices;
            xpbd_solve_vf_contacts_kernel<<<(n + block - 1) / block, block>>>(
                q_prev, q, v, mass_inv,
                contact.broad_phase_vf.data().get(), broad_phase_size,
                tris, geo->vertex_normals.data().get(),
                obj_data, vertices_obj, h, vf_force_k, relaxation,
                geo->ground, vf_ground_k, dx, n);

            n = params.nb_all_edges;
            if ( ee_force_k > 1e-12f )
                xpbd_solve_ee_contacts_kernel<<<(n + block - 1) / block, block>>>(
                    q, v, mass_inv,
                    contact.broad_phase_ee.data().get(), broad_phase_size,
                    edges, geo->edge_normals.data().get(),
                    obj_data, vertices_obj, h, 1.f / ee_force_k, relaxation,
                    dx, n);

            n = params.nb_all_cloth_vertices;
            applay_delta_xpbd<<<(n + block - 1) / block, block>>>(
                q, dx, mask, max_dx, n);
        }

    }
    // update substep end

    step_end_kernel<<<(n + block - 1) / block, block>>>(
        q, v, q_prev, dx, mask, h, max_vel, n);

}
