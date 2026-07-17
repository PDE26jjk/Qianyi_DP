#include "solver_VBD.cuh"

#include "solver_explicit.cuh"

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "contact/collision_detection.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"


void SolverVBD::init() {
    SolverBase::init();
    auto* geo = simulator->get_geo();
    geo->color_graph();
    geo->build_adj_data();
    auto& params = *simulator->get_geo_params();
    displacement.resize(params.nb_all_vertices);
    hessians.resize(params.nb_all_vertices);
}


static __global__ void step_end_kernel(
    float3* __restrict__ velocities,
    const float3* __restrict__ vertices_world,
    const float3* __restrict__ pos_prev,
    const char* __restrict__ vertices_mask,
    const float h,
    const float max_velocity,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !vertices_mask[i] ) {
            float3 v = (vertices_world[i] - pos_prev[i]) / h;
            if ( norm(v) > max_velocity )
                v = normalized(v) * max_velocity;
            v = v * expf(-h * 0.5f);
            velocities[i] = v;
        }
        else {
            velocities[i] = make_float3(0.f, 0.f, 0.f);
        }
    }
}
template<int BLOCK_SIZE, typename TYPE>
__device__ void block_reduce_sum(TYPE& val) {
    __shared__ TYPE smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    smem[tid] = val;
    __syncthreads();
    for ( int s = BLOCK_SIZE / 2; s > 0; s >>= 1 ) {
        if ( tid < s ) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    val = smem[0];
}

template<int BLOCK_SIZE>
__global__ void solve_elasticity_springs_kernel(
    float3* __restrict__ dx,
    const float3* __restrict__ pos_prev,
    const float3* __restrict__ pos,
    const float* __restrict__ inv_mass,
    const float* __restrict__ static_diags,
    const float3* __restrict__ inertia,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int2* __restrict__ edge_indices,
    const float* __restrict__ edge_rest_lengths,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    const float3* __restrict__ particle_forces,
    const Mat3* __restrict__ particle_hessians,
    const int* __restrict__ color_groups,
    const float kd,// damping
    float dt,
    int color_groups_size
) {
    int block_idx = blockIdx.x;
    int block_size = blockDim.x;
    int tid = threadIdx.x;
    if ( block_idx >= color_groups_size ) return;

    int pid = color_groups[block_idx];

    if ( inv_mass[pid] == 0.0f ) {
        return;
    }

    float3 f_total = make_float3(0, 0, 0);
    Mat3 H_total = Mat3::zero();
    float3 p0 = pos[pid];

    auto [offset ,count] = edge_lookup[pid];
    // ---------- accumulate contributions from all incident springs ----------
    float3 ks = obj_data[vertices_obj[pid]].stretch;
    float k = base_spring_stiffness * (ks.x + ks.y + ks.z) * 0.333f;
    for ( int i = tid; i < count; i += block_size ) {
        int2 elem = dir_edges[offset + i];

        float3 p1 = pos[elem.x];
        float rest_length = edge_rest_lengths[elem.y];
        float3 f_elastic;
        Mat3 H_elastic;
        calc_spring_elastic(p0, p1, rest_length, k,
            f_elastic, &H_elastic, HessianRegularization::SPD_CLAMP);

        f_total += f_elastic;
        H_total += H_elastic;
        // Damping contributions (original VBD paper)
        if ( kd > 0.0f ) {
            float coeff = kd / dt;
            // Damping Hessian: coeff * H_elastic
            H_total += H_elastic * coeff;

            // Damping force: coeff * H_elastic * (x_self - x_self_prev)
            f_total += (H_elastic * (p0 - pos_prev[pid])) * coeff;
        }
    }

    block_reduce_sum<BLOCK_SIZE>(H_total);
    block_reduce_sum<BLOCK_SIZE>(f_total);

    if ( tid == 0 ) {
        float sd = static_diags[pid];
        // ---------- add mass, inertia and external contributions ----------
        H_total.add_diag(sd);               // H += m/dt² I
        H_total += particle_hessians[pid];

        float3 rhs = f_total + (inertia[pid] - pos[pid]) * sd
            + particle_forces[pid];
        float3 delta = H_total.inverse() * rhs;
        dx[pid] = delta;
    }
}
constexpr int kNumThreadsPerPrimitive = 4;

__global__ void vbd_self_contact_kernel(
    float3* __restrict__ particle_forces,
    Mat3* __restrict__ particle_hessians,
    const float3* __restrict__ pos,
    const int* __restrict__ particle_colors,     // partition ID (for thread‑safe accumulation)
    const ObjectDataInput* __restrict__ obj_data,  // per‑object properties
    const int* __restrict__ vertices_obj,        // object index per vertex
    const int3* __restrict__ tri_indices,
    const int2* __restrict__ edges,
    const float3* __restrict__ vertex_normals,
    const float3* __restrict__ edge_normals,       // per‑edge smoothed normal
    const int* __restrict__ vf_broad_phase_pairs,// per‑vertex: [count, tri0, tri1, …]
    int broad_phase_size_vf,
    const int* __restrict__ ee_broad_phase_pairs,// per‑edge: [count, edge0, edge1, …]
    int broad_phase_size_ee,
    float vf_force_k,
    float ee_force_k,
    int current_color,
    float dt,
    int active_vertices_size,
    int num_vertices,
    int num_edges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int num_primitives = max(num_edges, num_vertices);
    int max_tid = kNumThreadsPerPrimitive * num_primitives;
    if ( tid >= max_tid ) return;

    int prim_id = tid / kNumThreadsPerPrimitive;
    int sub_id = tid % kNumThreadsPerPrimitive;

    // ---------- Edge-Edge collision for edge prim_id ----------
    if ( prim_id < num_edges ) {
        int e1 = prim_id;
        int2 edge = edges[e1];
        int v1 = edge.x, v2 = edge.y;

        // primary edge material
        int obj0 = vertices_obj[v1];
        const ObjectDataInput& od0 = obj_data[obj0];
        float thickness0 = od0.thickness;
        int layer0 = od0.collision_layer;
        int c1 = particle_colors[v1];
        int c2 = particle_colors[v2];

        float3 p0 = pos[v1], p1 = pos[v2];
        float3 edge_normal0 = edge_normals[e1];

        const int* base = &ee_broad_phase_pairs[e1 * broad_phase_size_ee];
        int count = base[0];
        const int* data = base + 1;

        for ( int j = sub_id; j < count; j += kNumThreadsPerPrimitive ) {
            int e2_raw = data[j];
            float side_sign = (e2_raw > 0) ? 1.0f : -1.0f;
            int e2 = abs(e2_raw);

            int2 edge2 = edges[e2];
            int u1 = edge2.x, u2 = edge2.y;
            float3 q0 = pos[u1], q1 = pos[u2];

            int obj1 = vertices_obj[u1];
            const ObjectDataInput& od1 = obj_data[obj1];
            float thickness1 = od1.thickness;
            int layer1 = od1.collision_layer;
            int cu1 = particle_colors[u1];
            int cu2 = particle_colors[u2];

            float comb_thick = thickness0 + thickness1;
            int layer_diff = layer0 - layer1;
            float3 edge_normal1 = edge_normals[e2];

            float s, t;
            float3 normal;
            float pen;
            if ( !compute_edge_edge_contact(p0, p1, q0, q1,
                comb_thick, layer_diff, side_sign,
                edge_normal0, edge_normal1,
                s, t, normal, pen) )
                continue;

            float fmag = ee_force_k * pen;
            float3 force = normal * fmag;
            Mat3 hess = Mat3::outer_product(normal, normal * ee_force_k);
            if ( v1 < active_vertices_size ) {
                if ( c1 == current_color ) {
                    float w = 1.0f - s;
                    atomicAddFloat3(&particle_forces[v1], force * w);
                    atomicAddMat3(&particle_hessians[v1], hess * (w * w));
                }
                if ( c2 == current_color ) {
                    float w = s;
                    atomicAddFloat3(&particle_forces[v2], force * w);
                    atomicAddMat3(&particle_hessians[v2], hess * (w * w));
                }
            }
            if ( u1 < active_vertices_size ) {
                if ( cu1 == current_color ) {
                    float w = 1.0f - t;
                    atomicAddFloat3(&particle_forces[u1], force * (-w));
                    atomicAddMat3(&particle_hessians[u1], hess * (w * w));
                }
                if ( cu2 == current_color ) {
                    float w = t;
                    atomicAddFloat3(&particle_forces[u2], force * (-w));
                    atomicAddMat3(&particle_hessians[u2], hess * (w * w));
                }
            }
        }
    }

    // ---------- Vertex-Triangle collision for vertex prim_id ----------
    if ( prim_id < num_vertices ) {
        int v_idx = prim_id;
        int cv = particle_colors[v_idx];
        // Material properties of the vertex
        int obj0 = vertices_obj[v_idx];
        const auto& od0 = obj_data[obj0];
        float thickness0 = od0.thickness;
        int layer0 = od0.collision_layer;

        float3 x0 = pos[v_idx];

        float3 vertex_normal = vertex_normals[v_idx];

        const int* pairs_base = &vf_broad_phase_pairs[v_idx * broad_phase_size_vf];
        int count = pairs_base[0];
        const int* pair_data = pairs_base + 1;

        for ( int j = sub_id; j < count; j += kNumThreadsPerPrimitive ) {
            int tri_raw = pair_data[j];
            float side_sign = (tri_raw > 0) ? 1.0f : -1.0f;
            int tri_idx = abs(tri_raw);

            int3 tri = tri_indices[tri_idx];
            int i1 = tri.x, i2 = tri.y, i3 = tri.z;
            float3 x1 = pos[i1], x2 = pos[i2], x3 = pos[i3];

            // Material properties of the triangle (all vertices share the same object)
            int obj1 = vertices_obj[i1];
            const auto& od1 = obj_data[obj1];
            float thickness1 = od1.thickness;
            int layer1 = od1.collision_layer;

            float combined_thickness = thickness0 + thickness1;
            int layer_diff = layer0 - layer1;

            float u, v, w, pen;
            float3 normal;
            if ( !compute_point_triangle_contact(x0, x1, x2, x3,
                combined_thickness, layer_diff, side_sign,
                vertex_normal,
                normal, u, v, w, pen) )
                continue;

            // Linear spring force and Hessian
            float fmag = vf_force_k * pen;
            float3 force = normal * fmag;
            Mat3 hess = Mat3::outer_product(normal, normal * vf_force_k);

            // Accumulate per vertex, only for those belonging to current_color
            if ( v_idx < active_vertices_size && cv == current_color ) {
                atomicAddFloat3(&particle_forces[v_idx], force);
                atomicAddMat3(&particle_hessians[v_idx], hess);
            }
            if ( i1 < active_vertices_size ) {
                if ( particle_colors[i1] == current_color ) {
                    atomicAddFloat3(&particle_forces[i1], force * (-u));
                    atomicAddMat3(&particle_hessians[i1], hess * (u * u));
                }
                if ( particle_colors[i2] == current_color ) {
                    atomicAddFloat3(&particle_forces[i2], force * (-v));
                    atomicAddMat3(&particle_hessians[i2], hess * (v * v));
                }
                if ( particle_colors[i3] == current_color ) {
                    atomicAddFloat3(&particle_forces[i3], force * (-w));
                    atomicAddMat3(&particle_hessians[i3], hess * (w * w));
                }
            }
        }
    }
}

__global__ void add_ground_contact(
    float3* __restrict__ particle_forces,
    Mat3* __restrict__ particle_hessians,
    const float3* __restrict__ pos,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int* __restrict__ color_groups,
    float ground_k,
    int color_groups_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tid >= color_groups_size ) return;
    int vid = color_groups[tid];
    float thickness = obj_data[vertices_obj[vid]].thickness;

    float z = pos[vid].z;
    if ( z < thickness ) {
        float penetration = thickness - z;
        float force_z = penetration * ground_k;
        particle_forces[vid].z += force_z;
        // outer product of (0,0,1) with itself scaled by ground_k
        particle_hessians[vid].r[2].z += ground_k;
    }
}

__global__ void apply_truncation_ts_kernel(
    float3* __restrict__ pos,
    const float3* __restrict__ dx,
    const float* __restrict__ truncation_ts,
    float max_displacement,
    const int* __restrict__ color_groups,
    int color_groups_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= color_groups_size ) return;
    int vid = color_groups[i];
    float t = truncation_ts ? truncation_ts[vid] : 1.f;
    float3 disp = dx[vid] * t;

    float len2 = dot(disp, disp);
    if ( len2 > max_displacement * max_displacement ) {
        disp = disp * (max_displacement / sqrtf(len2));
    }
    pos[vid] = pos[vid] + disp;
}

// A. H. Chen, Z. Liu, Y. Yang, and C. Yuksel, "Vertex Block Descent," ACM Trans. Graph. 43, 4, Article 116 (July 2024), 16 pages.
// https://doi.org/10.1145/3658179
void SolverVBD::step(float h) {
    // Vertex Block Descent
    auto& params = *simulator->get_geo_params();
    auto* geo = simulator->get_geo();
    int n = params.nb_all_vertices;
    int num_edges = params.nb_all_edges;
    int num_vertices = params.nb_all_vertices;
    int active_vertices_size = params.nb_all_cloth_vertices;
    if ( n <= 0 ) return;

    int block = 256;

    float3* q = geo->pos_world.data().get();
    const float3* q_prev = geo->pos_step_prev.data().get();
    float3* q_inertia = geo->pos_inertia.data().get();
    float3* dx = this->displacement.data().get();
    float3* v = geo->velocities.data().get();
    float3* f = geo->forces.data().get();
    Mat3* Jx_diag = this->hessians.data().get();
    int2* edges = geo->edges.data().get();
    int3* tri_edges = geo->triangles.data().get();
    int3* tris = geo->triangle_indices.data().get();
    char* mask = geo->vertices_mask.data().get();
    float* mass = geo->masses.data().get();
    float* mass_inv = geo->mass_inv.data().get();
    auto* obj_data = geo->obj_data.data().get();
    int* vertices_obj = geo->vertices_obj.data().get();
    float* static_diags = geo->static_diags.data().get();
    float max_vel = max(0.f, get_global_parameter("max_vel", 1000));

    cudaMemsetAsync(static_diags, 0, params.nb_all_vertices * sizeof(float));

    forward_step<<<(n + block - 1) / block, block>>>(
        q, v, mass_inv,
        nullptr,
        mask, q_inertia, nullptr,
        static_diags,
        h, 1e2, geo->gravity, n);

    int iters = max(1, (int)get_global_parameter("vbd_iters", 10));
    float damping = max(0.f, get_global_parameter("vbd_damping", 0.f));
    float vf_ground_k = max(0.f, geo->get_global_parameter("vf_ground_k", 0.2f));
    float vf_force_k = max(0.f, geo->get_global_parameter("vf_force_k", 0.2f));
    float ee_force_k = max(0.f, geo->get_global_parameter("ee_force_k", 0.2f));
    bool ground = geo->ground;
    // float ground_f = max(0.f, (get_global_parameter("ground_f", 1e3)));

    int num_colors = geo->h_colors_index_offsets.size() - 1;
    int* color_groups = geo->color_groups.data().get();
    constexpr int dynamics_block_size = 8;
    auto& contact = geo->get_contact();
    cudaMemcpyAsync(q, q_inertia,
        active_vertices_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    for ( int i = 0; i < iters; i++ ) {
        cudaMemsetAsync(f, 0, active_vertices_size * sizeof(float3));
        cudaMemsetAsync(Jx_diag, 0, active_vertices_size * sizeof(Mat3));
        cudaMemsetAsync(dx, 0, active_vertices_size * sizeof(float3));
        for ( int c = 0; c < num_colors; c++ ) {
            int color_index = geo->h_colors_index_offsets[c];
            int color_size = geo->h_colors_index_offsets[c + 1] - color_index;
            int total_threads = kNumThreadsPerPrimitive * max(num_vertices, num_edges);
            int* color_group_begin = color_groups + color_index;
            vbd_self_contact_kernel<<<(total_threads + block - 1) / block, block>>>(
                f, Jx_diag,
                q, geo->node_colors.data().get(),
                obj_data, vertices_obj,
                tris, edges,
                geo->vertex_normals.data().get(), geo->edge_normals.data().get(),
                contact.broad_phase_vf.data().get(), broad_phase_size,
                contact.broad_phase_ee.data().get(), broad_phase_size,
                vf_force_k, ee_force_k,
                c, h, active_vertices_size, num_vertices, num_edges);
            if ( ground )
                add_ground_contact<<<(color_size + block - 1) / block, block>>>(
                    f, Jx_diag,
                    q, obj_data, vertices_obj, color_group_begin, vf_ground_k, color_size);

            solve_elasticity_springs_kernel<dynamics_block_size><<<color_size, dynamics_block_size>>>
                (dx, q_prev, q, mass_inv, static_diags, q_inertia,
                obj_data, vertices_obj,
                edges, geo->edge_lengths.data().get(),
                geo->edge_lookup.data().get(),
                geo->dir_edges.data().get(), f, Jx_diag,
                color_group_begin, damping, h,
                color_size);

            apply_truncation_ts_kernel<<<(color_size + block - 1) / block, block>>>(
                q, dx, nullptr, 1000.f, color_group_begin, color_size);
        }

    }

    n = params.nb_all_cloth_edges;
    // update substep end

    n = params.nb_all_cloth_vertices;

    step_end_kernel<<<(n + block - 1) / block, block>>>(
        v, q, q_prev, mask, h, max_vel, n);

}
