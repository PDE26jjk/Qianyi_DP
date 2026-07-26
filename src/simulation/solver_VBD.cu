#include "solver_VBD.cuh"

#include "solver_explicit.cuh"

// #include "constraint.cuh"
#include "geometric_operator.cuh"
#include "geometry.cuh"
#include "contact/collision.cuh"
#include "contact/collision_detection.cuh"
#include "dynamics/bending.cuh"
#include "dynamics/planar.cuh"

__device__ int debug_e_id;
__device__ int debug_v_id;

constexpr float PENALTY_MIN = 0.1f;
constexpr float PENALTY_MAX = 1e2f;

void ContactState::reset() {
    lambda = make_float3(0, 0, 0);
    penalty = make_float3(PENALTY_MIN, PENALTY_MIN, PENALTY_MIN);
}
void ContactState::new_step() {
    // from avbd-demo3d
    constexpr float alpha = 0.99f;
    constexpr float gamma = 0.999f;
    lambda_x *= alpha * gamma;
    // lambda_y *= alpha * gamma;
    penalty = clamp(penalty * gamma, PENALTY_MIN, PENALTY_MAX);
}
void SolverVBD::init() {
    SolverBase::init();
    auto* geo = simulator->get_geo();
    geo->color_graph();
    geo->build_adj_data();
    auto& params = *simulator->get_geo_params();
    displacement.resize(params.nb_all_vertices);
    hessians.resize(params.nb_all_vertices);
    geo->pos_temp.resize(params.nb_all_vertices);

    auto& contact = geo->get_contact();
    contact.do_collision_detect_broad_phase_before_step = false;
    vf_states.resize(contact.broad_phase_vf.size());
    ee_states.resize(contact.broad_phase_ee.size());
    cudaMemsetAsync(contact.broad_phase_vf.data().get(), 0, contact.broad_phase_vf.size() * sizeof(int));
    cudaMemsetAsync(contact.broad_phase_ee.data().get(), 0, contact.broad_phase_ee.size() * sizeof(int));
    cudaMemsetAsync(vf_states.data().get(), 0, contact.broad_phase_vf.size() * sizeof(ContactState));
    cudaMemsetAsync(ee_states.data().get(), 0, contact.broad_phase_ee.size() * sizeof(ContactState));

    ground_states.resize(params.nb_all_vertices);
}
__device__ inline void get_face_normals(int2 eop, const float3* pos, float3 A0, float3 D, float3& Na, float3& Nb) {
    if ( eop.x >= 0 )
        Na = normalized(cross(D, pos[eop.x] - A0));
    if ( eop.y >= 0 )
        Nb = normalized(cross(pos[eop.y] - A0, D));
    else
        Nb = -Na;
    if ( eop.x < 0 )
        Na = -Nb;

}
static __global__ void query_ee_pairs_capsule_stated_kernel(
    const float3* __restrict__ pos,
    const float3* __restrict__ pos_target,
    unsigned int num_queries,
    const int2* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    float query_radius,
    const float3* __restrict__ edge_normals,
    const int active_vertices_size,
    ContactState* __restrict__ states,
    int* __restrict__ query_results,
    int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    i = nodes[i].x - 1;

    int2 edge = edges[i];
    float3 A0 = pos[edge.x];
    float3 B0 = pos[edge.y];
    float3 A1 = pos_target[edge.x];
    float3 B1 = pos_target[edge.y];
    const float3 N = edge_normals[i];

    // thickness for edge i
    const auto& od_i = obj_data[vertices_obj[edge.x]];
    int layer0 = od_i.collision_layer;
    float r_e1_thick = query_radius + od_i.thickness;

    // Query edge trajectory capsule
    float3 cap1_start, cap1_end;
    float cap1_radius;
    edge_trajectory_capsule(A0, B0, A1, B1, r_e1_thick,
        cap1_start, cap1_end, cap1_radius);

    // Check state in last frame
    int* base = &query_results[i * result_size];
    int old_count = base[0];
    ContactState* state_base = &states[i * result_size];

    int kept_count = 0;
    for ( int j = 1; j <= old_count; j++ ) {
        int e2_raw = base[j];
        int e2 = abs(e2_raw);
        int2 e2_ = edges[e2];
        float pen = state_base[j].pen;
        float dist2 = segment_segment_dist_sq_robust(A0, B0, pos[e2_.x], pos[e2_.y]);
        if ( pen > 0 && pen < r_e1_thick * 10 && sqrt(dist2) < r_e1_thick * 10 ) {
            ++kept_count;
            base[kept_count] = e2_raw;
            state_base[kept_count] = state_base[j];
            state_base[kept_count].new_step();
            if ( i == debug_e_id ) {
                printf("[BVH old] e2=%d,id=%d,pen=%e,k=%e\n", e2_raw, j, pen, state_base[kept_count].penalty.x);
            }
        }
        else {
            if ( i == debug_e_id ) {
                printf("[BVH discard] e2=%d,id=%d,pen=%e,k=%e\n", e2_raw, j, pen, state_base[j].penalty.x);
            }
        }
    }

    bool is_active = (edge.x < active_vertices_size);

    AABB q_aabb;
    q_aabb.min = fmin3(fmin3(A0, B0), fmin3(A1, B1)) - r_e1_thick * 2;
    q_aabb.max = fmax3(fmax3(A0, B0), fmax3(A1, B1)) + r_e1_thick * 2;
    float3 dir0 = B0 - A0;
    float len2 = dot(dir0, dir0);
    if ( len2 < 1e-12f ) {
        query_results[i * result_size] = 0;
        return;
    }
    float3 D = dir0 * rsqrtf(len2);
    float3 Na1, Nb1;
    get_face_normals(edge_opposite_points[i], pos, A0, D, Na1, Nb1);
    float angle1 = dot(Na1, Nb1);
    // @formatter:off
    BVH_QUERY_LOOP(q_aabb, 64, kept_count, {
        if ( prim_idx <= i ) continue;
        int2 e = edges[prim_idx];
        if (!is_active && e.x >= active_vertices_size) continue;
        if (edge.x == e.x || edge.x == e.y || edge.y == e.x || edge.y == e.y) continue;
        bool is_duplicate = false;
        for (int k = 1; k <= kept_count; ++k) {
            if (abs(query_result[k]) == prim_idx) {
                is_duplicate = true;
                break;
            }
        }

        if (is_duplicate) continue;
        float3 C0 = pos[e.x];
        float3 D0 = pos[e.y];
        float3 C1 = pos_target[e.x];
        float3 D1 = pos_target[e.y];
        float3 dir1 = D0 - C0;
        len2 = dot(dir1, dir1);
        if ( len2 < 1e-12f ) {
            continue;
        }
        float3 D2 = dir1 * rsqrtf(len2);
        
        const auto& od_e2 = obj_data[vertices_obj[e.x]];
        float r_e2_thick = query_radius+ od_e2.thickness;

        float3 cap2_start, cap2_end;
        float  cap2_radius;
        edge_trajectory_capsule(C0, D0, C1, D1, r_e2_thick,
                                cap2_start, cap2_end, cap2_radius);

        if (!capsule_capsule_intersects(cap1_start, cap1_end, cap1_radius,
                                        cap2_start, cap2_end, cap2_radius))
            continue;
        // compute signed result: sign based on relative position to edge i's normal
        // int sign = dot(cap1_start - cap2_start, N) < 0.0f ? 1 : -1;
        
        float3 Na2, Nb2;
        get_face_normals(edge_opposite_points[prim_idx],pos,C0,D2,Na2,Nb2);
        float angle2 = dot(Na2,Nb2);
        
        // int sign = compute_ee_min_signed_dist(A0, B0, C0, D0, N) < 0.0f ? 1 : -1;
        int sign = 1;

        
       //  if (i == 7) {
       //      if (query_count == 0) {
       //          printf("i=%d: A0=(%e,%e,%e), B0=(%e,%e,%e), "
       //     "A1=(%e,%e,%e), B1=(%e,%e,%e)\n",
       //     i, A0.x, A0.y, A0.z, B0.x, B0.y, B0.z,
       //     A1.x, A1.y, A1.z, B1.x, B1.y, B1.z);
       //
       //          printf("N=(%e,%e,%e), r_e1_thick=%e\n", N.x, N.y, N.z, r_e1_thick);
       //      }
       //      printf("e1:%d (v%d,v%d), e2:%d (v%d,v%d), \n", i, edge.x, edge.y, prim_idx, e.x, e.y);
       //      printf("prim_idx=%d: C0=(%e,%e,%e), D0=(%e,%e,%e), "
       // "C1=(%e,%e,%e), D1=(%e,%e,%e)\n",
       // prim_idx,
       // C0.x, C0.y, C0.z, D0.x, D0.y, D0.z,
       // C1.x, C1.y, C1.z, D1.x, D1.y, D1.z);
       //
       //  }
        ++query_count;
        state_base[query_count].reset();
        if (angle1 > angle2) {
            state_base[query_count].type = 0;
            sign = is_edge_wedge_outside(A0, D, C0, D0,Na1,Nb1) ? 1 : -1;
        }
        else {
            state_base[query_count].type = 1;
            sign = is_edge_wedge_outside(C0, D2, A0, B0,Na2,Nb2) ? 1 : -1;
        }
        query_result[query_count] = prim_idx * sign;
        if ( i == debug_e_id ) {
            int2 eop = edge_opposite_points[prim_idx];
        // printf("[face_normals] eop=(%d,%d),Na=(%e,%e,%e),Nb=(%e,%e,%e)\n", eop.x,eop.y ,Na2.x, Na2.y, Na2.z,Nb2.x,Nb2.y,Nb2.z);
            float3 ba = cap1_start - cap2_start;
            printf("[BVH new] e=%d,id=%d,angle1:%e,angle2:%e ba=(%e,%e,%e),N=(%e,%e,%e), k=%e\n", prim_idx * sign,
                query_count,angle1,angle2, ba.x, ba.y, ba.z,N.x,N.y,N.z, state_base[kept_count].penalty.x);
        }
    });
    // @formatter:on
}

static __global__ void query_vf_pairs_capsule_stated_kernel(
    const unsigned int* __restrict__ sorted_indices,
    unsigned int num_queries,
    const int2* __restrict__ nodes,
    const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const float3* __restrict__ pos,
    const float3* __restrict__ pos_target,
    const int3* __restrict__ faces,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const float query_radius,
    const float3* __restrict__ vertex_normals,
    const int active_vertices_size,
    ContactState* __restrict__ states,
    int* __restrict__ query_results,
    int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    if ( sorted_indices ) i = sorted_indices[i];

    float3 P0 = pos[i];
    float3 P1 = pos_target[i];
    const auto& od = obj_data[vertices_obj[i]];
    float r_p = od.thickness + query_radius;
    int layer0 = od.collision_layer;

    int* base = &query_results[i * result_size];
    int old_count = base[0];
    ContactState* state_base = &states[i * result_size];

    // Check state in last frame
    if ( P1.z > od.thickness * 1.5f ) {
        // ground collision state
        state_base[0].reset();
    }
    else {
        state_base[0].new_step();
    }
    int kept_count = 0;
    for ( int j = 1; j <= old_count; j++ ) {
        int tri_raw = base[j];
        float pen = state_base[j].pen;

        if ( pen > -od.thickness * 0.5f && pen < r_p * 10 ) {
            ++kept_count;
            base[kept_count] = tri_raw;
            state_base[kept_count] = state_base[j];
            state_base[kept_count].new_step();
            if ( i == debug_v_id ) {
                printf("[BVH old] t=%d,id=%d,pen=%e,k=%e\n", tri_raw, j, pen, state_base[kept_count].penalty.x);
            }
        }
        else {
            if ( i == debug_v_id ) {
                printf("[BVH discard] t=%d,id=%d,pen=%e\n", tri_raw, j, pen);
            }
        }
    }

    AABB q_aabb = {
        .min = fmin3(P0, P1) - r_p * 2,
        .max = fmax3(P0, P1) + r_p * 2,
    };
    bool is_active = i < active_vertices_size;

    // @formatter:off
    BVH_QUERY_LOOP(q_aabb, 64, kept_count, {
        int3 f = faces[prim_idx];
        if ( f.x == i || f.y == i || f.z == i ) continue;
        if (!is_active && f.x >= active_vertices_size) continue;
        bool is_duplicate = false;
        for (int k = 1; k <= kept_count; ++k) {
            if (abs(query_result[k]) == prim_idx) {
                is_duplicate = true;
                break;
            }
        }
        if (is_duplicate) continue;
        
        float3 A0 = pos[f.x], A1 = pos_target[f.x];
        float3 B0 = pos[f.y], B1 = pos_target[f.y];
        float3 C0 = pos[f.z], C1 = pos_target[f.z];

        const auto& od_f = obj_data[vertices_obj[f.x]];
        float r_tri = od_f.thickness + query_radius;

        float3 tri_cap_start, tri_cap_end;
        float  tri_cap_radius;
        triangle_trajectory_capsule(A0, B0, C0, A1, B1, C1,
                                    r_tri, 
                                    tri_cap_start, tri_cap_end, tri_cap_radius);

        if ( !capsule_capsule_intersects(P0, P1, r_p,
                                         tri_cap_start, tri_cap_end, tri_cap_radius) )
            continue;
        float3 face_dir = cross(B0 - A0, C0 - A0);
        int sign = dot(face_dir, P0 - A0) >= 0.0f ? 1 : -1;
        ++query_count;
        query_result[query_count] = prim_idx * sign;
        state_base[query_count].reset();
        if ( i == debug_v_id ) {
            printf("[BVH new] t=%d,id=%d,sign_factor=%e)\n", prim_idx * sign, query_count,dot(face_dir, P0 - A0));
        }
    });
    // @formatter:on

}
void Contact::collision_detect_broad_phase_stated(const float3* pos, const float3* pos_target,
    ContactState* vf_states, ContactState* ee_states, float query_radius
) {
    refit_bvh_with_target(pos, pos_target);
    auto& params = geo->params;
    int block = 256;
    int num_queries = params.nb_all_edges;

    query_ee_pairs_capsule_stated_kernel<<<(num_queries + block - 1) / block, block>>>(
        pos,
        pos_target,
        num_queries,
        thrust::raw_pointer_cast(edge_bvh.nodes.data()),
        thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
        edge_bvh.root_idx,
        thrust::raw_pointer_cast(geo->edges.data()),
        thrust::raw_pointer_cast(geo->edge_opposite_points.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        query_radius,
        thrust::raw_pointer_cast(geo->edge_normals.data()),
        params.nb_all_cloth_vertices,
        ee_states,
        thrust::raw_pointer_cast(broad_phase_ee.data()),
        broad_phase_size
        );


    num_queries = params.nb_all_vertices;
    query_vf_pairs_capsule_stated_kernel<<<(num_queries + block - 1) / block, block>>>(
        point_sorted_indices.data().get(),
        num_queries,
        thrust::raw_pointer_cast(tri_bvh.nodes.data()),
        thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
        tri_bvh.root_idx,
        pos,
        pos_target,
        thrust::raw_pointer_cast(geo->triangle_indices.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        query_radius,
        thrust::raw_pointer_cast(geo->vertex_normals.data()),
        params.nb_all_cloth_vertices,
        vf_states,
        thrust::raw_pointer_cast(broad_phase_vf.data()),
        broad_phase_size
        );
    // return;
}


static __global__ void step_end_kernel(
    float3* __restrict__ velocities,
    float3* __restrict__ pos_world,
    const float* __restrict__ truncation_ts,
    const float3* __restrict__ pos_prev,
    const char* __restrict__ vertices_mask,
    const float h,
    const float max_velocity,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        if ( !vertices_mask[i] ) {
            float3 dx = pos_world[i] - pos_prev[i];
            if ( truncation_ts )
                dx *= truncation_ts[i];
            pos_world[i] = pos_prev[i] + dx;
            float3 v = dx / h;
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
    // float3* __restrict__ dx,
    const float3* __restrict__ pos,
    float3* __restrict__ particle_forces,
    Mat3* __restrict__ particle_hessians,
    const float3* __restrict__ pos_prev,
    const float* __restrict__ inv_mass,
    const float* __restrict__ static_diags,
    const float3* __restrict__ inertia,
    const ObjectDataInput* __restrict__ obj_data,
    const int* __restrict__ vertices_obj,
    const int2* __restrict__ edge_indices,
    const float* __restrict__ edge_rest_lengths,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
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
            Mat3 H_damping = H_elastic * coeff;
            H_total += H_damping;

            // Damping force: - coeff * H_elastic * (x_self - x_self_prev)
            f_total -= H_damping * (p0 - pos_prev[pid]);
        }
    }

    block_reduce_sum<BLOCK_SIZE>(H_total);
    block_reduce_sum<BLOCK_SIZE>(f_total);

    if ( tid == 0 ) {
        float sd = static_diags[pid];
        // ---------- add mass, inertia and external contributions ----------
        H_total.add_diag(sd);               // H += m/dt² I
        // // H_total += particle_hessians[pid];
        //
        float3 rhs = f_total + (inertia[pid] - pos[pid]) * sd;
        // // + particle_forces[pid];
        // float3 delta = H_total.inverse() * rhs;
        // pos[pid] += delta;
        // dx[pid] = pos[pid] - pos_prev[pid];
        particle_hessians[pid] += H_total;
        particle_forces[pid] += rhs;
    }
}

__device__ void evaluate_self_contact_force_norm(
    float penetration_depth, float collision_radius, float k, float& dEdD, float& d2E_dDdD) {
    // Penetration depth (not used in all branches but kept for clarity)
    float dis = collision_radius - penetration_depth;

    // C2 continuous barrier parameters
    float tau = collision_radius * 0.5f;
    dis = max(dis, tau);
    const float d_min = 1.0e-5f;

    if ( tau > dis && dis > d_min ) {
        // Log-barrier region: E ∝ -ln(dis)
        float k2 = tau * tau * k;
        dEdD = -k2 / dis;
        d2E_dDdD = k2 / (dis * dis);
    }
    else if ( dis <= d_min ) {
        // Quadratic extension below d_min (Taylor expansion of the log-barrier at d_min)
        // Preserves C2 continuity: constant Hessian, linear gradient
        float k2 = tau * tau * k;
        float d_min_sq = d_min * d_min;
        dEdD = k2 * (dis - 2.0f * d_min) / d_min_sq;
        d2E_dDdD = k2 / d_min_sq;
    }
    else {
        // Outside barrier region: standard penalty force
        dEdD = -k * penetration_depth;
        d2E_dDdD = k;
    }
}
constexpr int kNumThreadsPerPrimitive = 4;

__global__ void vbd_self_contact_kernel(
    float3* __restrict__ forces,
    Mat3* __restrict__ hessians,
    const float3* __restrict__ pos,
    const int* __restrict__ particle_colors,     // partition ID (for thread‑safe accumulation)
    const ObjectDataInput* __restrict__ obj_data,  // per‑object properties
    const int* __restrict__ vertices_obj,        // object index per vertex
    const int3* __restrict__ tri_indices,
    const int2* __restrict__ edges,
    const int2* __restrict__ edge_opposite_points,
    const float3* __restrict__ vertex_normals,
    const float3* __restrict__ edge_normals,       // per‑edge smoothed normal
    const int* __restrict__ vf_broad_phase_pairs,// per‑vertex: [count, tri0, tri1, …]
    int broad_phase_size_vf,
    const int* __restrict__ ee_broad_phase_pairs,// per‑edge: [count, edge0, edge1, …]
    int broad_phase_size_ee,
    ContactState* __restrict__ vf_states,
    ContactState* __restrict__ ee_states,
    int current_color,
    float dt,
    bool ground,
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
        bool e1_active = (v1 < active_vertices_size)
            && (current_color == c1 || current_color == c2);
        float3 p0 = pos[v1], p1 = pos[v2];
        float3 edge_normal0 = edge_normals[e1];

        const int* base = &ee_broad_phase_pairs[e1 * broad_phase_size_ee];
        int count = base[0];
        const int* data = base + 1;
        auto* states = &ee_states[e1 * broad_phase_size_ee + 1];
        float3 D = normalized(p1 - p0);
        for ( int j = sub_id; j < count; j += kNumThreadsPerPrimitive ) {
            int e2_raw = data[j];
            float side_sign = (e2_raw > 0) ? 1.0f : -1.0f;
            int e2 = abs(e2_raw);

            int2 edge2 = edges[e2];
            int u1 = edge2.x, u2 = edge2.y;
            int cu1 = particle_colors[u1];
            int cu2 = particle_colors[u2];
            if ( !e1_active && !(u1 < active_vertices_size &&
                (u1 == current_color || u2 == current_color)) )
                continue;
            float3 q0 = pos[u1], q1 = pos[u2];

            int obj1 = vertices_obj[u1];
            const ObjectDataInput& od1 = obj_data[obj1];
            float thickness1 = od1.thickness;
            int layer1 = od1.collision_layer;

            float comb_thick = thickness0 + thickness1;
            int layer_diff = layer0 - layer1;
            float3 edge_normal1 = edge_normals[e2];

            float s, t;
            float3 normal = make_float3(0.0f, 0.0f, 0.0f);
            float pen = 0;
            float3 Na, Nb;
            float3 A0 = p0, B0 = p1, C0 = q0, D0 = q1, Dir = D;
            float dist2 = segment_segment_dist_sq_robust(A0, B0, C0, D0);
            if ( sqrt(dist2) > comb_thick * 3 ) continue;

            int e_id = e1;
            bool use_e2 = states[j].type == 1;
            if ( use_e2 ) {
                A0 = q0;
                B0 = q1;
                C0 = p0;
                D0 = p1;
                e_id = e2;
                Dir = normalized(B0 - A0);
            }
            get_face_normals(edge_opposite_points[e_id], pos, A0, Dir, Na, Nb);
            float tip_angle_cos = 0.9659f;
            if ( !compute_edge_edge_wedge_contact(A0, B0, C0, D0, Na, Nb,
                tip_angle_cos, comb_thick,
                side_sign > 0, pen, normal, s, t) ) {
                states[j].pen = pen;
                continue;
            }
            if ( use_e2 ) {
                normal = -normal;
                float temp = s;
                s = t;
                t = temp;
            }
            if ( e1 == debug_e_id ) {
                float3 dir0 = B0 - A0;
                float len2 = dot(dir0, dir0);
                D = dir0 * rsqrtf(len2);
                bool is_outside = side_sign > 0;
                int sign_result = is_outside ? 1 : -1;
                tip_angle_cos = max(tip_angle_cos, dot(Na, Nb));

                if ( !is_outside ) {
                    Na = -Na;
                    Nb = -Nb;
                }

                float3 Ne = Na + Nb;
                float ne_len2 = dot(Ne, Ne);
                if ( ne_len2 < 1e-6f ) Ne = normalized(cross(D, Na));
                else Ne = Ne * rsqrtf(ne_len2);

                float3 T = cross(D, Ne);
                if ( dot(T, T) < 1e-8f ) T = normalized(orthogonal_vector(D));
                else T = normalized(T);

                // ==========================================
                // 严格边-楔形体距离计算：寻找 E1 上最深穿透点
                // ==========================================
                float3 pC = C0 - A0;
                float3 V = D0 - C0;
                float v_dot_d = dot(V, D);
                float3 V2d = V - v_dot_d * D;
                float v2d_len2 = dot(V2d, V2d);

                float t_cross = 0.5f;
                if ( v2d_len2 > 1e-12f ) {
                    float3 pC2d = pC - dot(pC, D) * D;
                    t_cross = -dot(pC2d, V2d) / v2d_len2;
                    t_cross = fmaxf(0.0f, fminf(1.0f, t_cross));
                }

                float min_F = 1e38f;
                float3 best_norm = Ne;
                float best_t = 0.5f;

                auto eval_wedge_point = [&](float t_val) {
                    float3 P = C0 + t_val * V;
                    float3 p_vec = P - A0;
                    float p_x = dot(p_vec, T);
                    float p_y = dot(p_vec, Ne);
                    float L2 = p_x * p_x + p_y * p_y;
                    float L = sqrtf(L2);

                    float cos_theta = (L > 1e-6f) ? (p_y / L) : 1.0f;
                    float dist;
                    float3 norm;

                    if ( cos_theta > tip_angle_cos ) {
                        dist = L;
                        if ( L > 1e-6f ) { norm = normalized(p_x * T + p_y * Ne); }
                        else { norm = Ne; }
                    }
                    else {
                        float d_a = dot(p_vec, Na);
                        float d_b = dot(p_vec, Nb);
                        if ( d_a > d_b ) {
                            dist = d_a;
                            norm = Na;
                        }
                        else {
                            dist = d_b;
                            norm = Nb;
                        }
                    }

                    float F = dist;
                    // float F = dist * sign_result;
                    if ( F < min_F ) {
                        min_F = F;
                        best_norm = norm;
                        best_t = t_val; // 记录产生最深穿透的 t
                    }
                    printf("p_x: %e,p_y: %e,cos_theta=%e,normal=(%f, %f, %f),F: %e \n", p_x, p_y, cos_theta, norm.x, norm.y,
                        norm.z, F);
                };

                // 3. 评估三个极限候选点
                eval_wedge_point(0.0f);        // 端点 C0
                eval_wedge_point(1.0f);        // 端点 D0
                eval_wedge_point(t_cross);     // 与 E0 最近驻点

                float out_penetration = comb_thick - min_F;
                float out_t = best_t;

                // 6. 计算 s: 将最优的碰撞点 P1 正交投影回 E0 线段上
                float3 best_P1 = C0 + best_t * V;
                float s_unclamped = dot(best_P1 - A0, dir0) / len2; // len2 之前已计算: dot(B0-A0, B0-A0)
                float out_s = clamp(s_unclamped, 0.0f, 1.0f);      // 截断到 [0,1] 保证重心权重稳定

                printf("[contact data] e2=%d,e_id=%d,is_outside=%d,tip_angle_cos=%e, pen=%f,"
                    "normal=(%f, %f, %f),Ne=(%f, %f, %f),Na=(%f, %f, %f), Nb=(%f, %f, %f), t: %e,s: %e\n",
                    e2_raw, e_id, is_outside, tip_angle_cos, out_penetration,
                    normal.x, normal.y, normal.z, Ne.x, Ne.y, Ne.z, Na.x, Na.y, Na.z, Nb.x, Nb.y, Nb.z, out_t, out_s);
            }

            // if ( !compute_edge_edge_contact(p0, p1, q0, q1,
            //     comb_thick, layer_diff, side_sign,
            //     edge_normal0, edge_normal1,
            //     s, t, normal, pen) ) {
            //     if ( e1 == debug_e_id ) {
            // float3 ba; // b->a = a-b
            // segment_segment_closest_robust(p0, p1, q0, q1, s, t, ba);
            //
            // float dist = norm(ba);
            //
            // // Degenerate case: fall back to edge_normal0
            // if ( dist < 1e-16f ) {
            //     ba = -edge_normal0;
            //     normal = ba;
            // }
            // else {
            //     normal = ba / dist;
            // }
            //
            // // Direction correction based on layer difference and broad-phase sign
            // // Same layer: use the sign stored during broad phase
            // float sign_new_raw = (dot(ba, edge_normal0) < 0.0f) ? 1.0f : -1.0f;
            // float sign_new = sign_new_raw * side_sign;
            // if ( sign_new < 0.0f ) {
            //     normal = -normal;
            //     dist = -dist;
            // }
            // printf("[no contact] e2=%d, pen=%f,ba=(%f, %f, %f),"
            //     "edge_normal0=(%f, %f, %f),"
            //     "sign_new_raw=%f,sign_new=%f, normal=(%f, %f, %f)\n",
            //     e2_raw, pen, ba.x, ba.y, ba.z, edge_normal0.x, edge_normal0.y, edge_normal0.z, sign_new_raw, sign_new,
            //     normal.x, normal.y, normal.z);
            // }
            // continue;
            // }
            states[j].pen = pen;
            float fmag = states[j].lambda.x + states[j].penalty.x * pen;
            fmag = min(fmag, PENALTY_MAX);
            float3 force = normal * fmag;
            Mat3 hess = Mat3::outer_product(normal, normal * states[j].penalty.x);
            // if ( e1 == debug_e_id ) {
            //     printf("[contact] e2=%d, pen=%f, fmag=%e,k=%e, normal=(%f, %f, %f), force=(%f, %f, %f)\n",
            //         e2_raw, pen, fmag, states[j].penalty.x, normal.x, normal.y, normal.z,
            //         force.x, force.y, force.z);
            // }

            if ( v1 < active_vertices_size ) {
                if ( c1 == current_color ) {
                    float w = 1.0f - s;
                    atomicAddFloat3(&forces[v1], force * w);
                    atomicAddMat3(&hessians[v1], hess * (w * w));
                }
                if ( c2 == current_color ) {
                    float w = s;
                    atomicAddFloat3(&forces[v2], force * w);
                    atomicAddMat3(&hessians[v2], hess * (w * w));
                }
            }
            if ( u1 < active_vertices_size ) {
                if ( cu1 == current_color ) {
                    float w = 1.0f - t;
                    atomicAddFloat3(&forces[u1], force * (-w));
                    atomicAddMat3(&hessians[u1], hess * (w * w));
                }
                if ( cu2 == current_color ) {
                    float w = t;
                    atomicAddFloat3(&forces[u2], force * (-w));
                    atomicAddMat3(&hessians[u2], hess * (w * w));
                }
            }
        }
    }

    // ---------- Vertex-Triangle collision for vertex prim_id ----------
    if ( prim_id < num_vertices ) {
        int v_idx = prim_id;
        int cv = particle_colors[v_idx];
        bool v_active = v_idx < active_vertices_size && current_color == cv;

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
        auto* states = &vf_states[v_idx * broad_phase_size_vf];
        if ( sub_id == 0 && v_active && ground ) {
            float penetration = thickness0 - x0.z;
            states[0].pen = penetration;
            if ( penetration > 0.0f ) {
                float fmag = states[0].lambda.x + states[0].penalty.x * penetration;
                atomicAdd(&forces[v_idx].z, fmag);
                // outer product of (0,0,1) with itself scaled by ground_k
                atomicAdd(&hessians[v_idx].r[2].z, states[0].penalty.x);
            }
        }
        states += 1;
        for ( int j = sub_id; j < count; j += kNumThreadsPerPrimitive ) {
            int tri_raw = pair_data[j];
            float side_sign = (tri_raw > 0) ? 1.0f : -1.0f;
            int tri_idx = abs(tri_raw);

            int3 tri = tri_indices[tri_idx];
            int i1 = tri.x, i2 = tri.y, i3 = tri.z;
            int ct1 = particle_colors[i1], ct2 = particle_colors[i2], ct3 = particle_colors[i3];
            if ( !v_active && !(i1 < active_vertices_size && (ct1 == current_color
                || ct2 == current_color || ct3 == current_color)) )
                continue;
            if ( v_idx >= active_vertices_size && i1 >= active_vertices_size ) continue;
            float3 x1 = pos[i1], x2 = pos[i2], x3 = pos[i3];

            int obj1 = vertices_obj[i1];
            const auto& od1 = obj_data[obj1];
            float thickness1 = od1.thickness;
            int layer1 = od1.collision_layer;

            float combined_thickness = thickness0 + thickness1;
            int layer_diff = layer0 - layer1;

            float u, v, w, pen = 0;
            float3 normal;
            if ( !compute_point_triangle_contact(x0, x1, x2, x3,
                combined_thickness, layer_diff, side_sign,
                vertex_normal,
                normal, u, v, w, pen) ) {
                states[j].pen = pen;
                continue;
            }
            states[j].pen = pen;

            // Linear spring force and Hessian
            float fmag = states[j].lambda.x + states[j].penalty.x * pen;
            fmag = min(fmag, PENALTY_MAX);
            float3 force = normal * fmag;
            Mat3 hess = Mat3::outer_product(normal, normal * states[j].penalty.x);
            // Accumulate per vertex, only for those belonging to current_color
            if ( v_idx < active_vertices_size && cv == current_color ) {
                atomicAddFloat3(&forces[v_idx], force);
                atomicAddMat3(&hessians[v_idx], hess);
            }
            if ( i1 < active_vertices_size ) {
                if ( ct1 == current_color ) {
                    atomicAddFloat3(&forces[i1], force * (-u));
                    atomicAddMat3(&hessians[i1], hess * (u * u));
                }
                if ( ct2 == current_color ) {
                    atomicAddFloat3(&forces[i2], force * (-v));
                    atomicAddMat3(&hessians[i2], hess * (v * v));
                }
                if ( ct3 == current_color ) {
                    atomicAddFloat3(&forces[i3], force * (-w));
                    atomicAddMat3(&hessians[i3], hess * (w * w));
                }
            }
        }
    }
}
__device__ __forceinline__ void update_contact_state(ContactState& state, float pen, float beta) {
    float fmag = state.lambda.x + state.penalty.x * pen;
    fmag = max(0.f, fmag);
    state.lambda.x = fmag;
    if ( fmag > 0 ) {
        state.penalty.x = min(state.penalty.x + beta * pen, PENALTY_MAX);
    }
}
__global__ void vbd_contact_dual_update_kernel(
    const float3* __restrict__ pos,
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
    ContactState* __restrict__ vf_states,
    ContactState* __restrict__ ee_states,
    float beta,
    bool ground,
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

        float3 p0 = pos[v1], p1 = pos[v2];
        float3 edge_normal0 = edge_normals[e1];

        const int* base = &ee_broad_phase_pairs[e1 * broad_phase_size_ee];
        int count = base[0];
        const int* data = base + 1;
        auto* states = &ee_states[e1 * broad_phase_size_ee + 1];

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

            float comb_thick = thickness0 + thickness1;
            int layer_diff = layer0 - layer1;
            float3 edge_normal1 = edge_normals[e2];

            float s, t;
            float3 normal;
            float pen = 0;
            compute_edge_edge_contact(p0, p1, q0, q1,
                comb_thick, layer_diff, side_sign,
                edge_normal0, edge_normal1,
                s, t, normal, pen);

            update_contact_state(states[j], pen, beta * 0.1f);
        }
    }

    // ---------- Vertex-Triangle collision for vertex prim_id ----------
    if ( prim_id < num_vertices ) {
        int v_idx = prim_id;
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
        auto* states = &vf_states[v_idx * broad_phase_size_vf];
        if ( sub_id == 0 && ground ) {
            float pen = thickness0 - x0.z;
            update_contact_state(states[0], pen, beta);
        }
        states += 1;
        for ( int j = sub_id; j < count; j += kNumThreadsPerPrimitive ) {
            int tri_raw = pair_data[j];
            float side_sign = (tri_raw > 0) ? 1.0f : -1.0f;
            int tri_idx = abs(tri_raw);

            int3 tri = tri_indices[tri_idx];
            int i1 = tri.x, i2 = tri.y, i3 = tri.z;
            float3 x1 = pos[i1], x2 = pos[i2], x3 = pos[i3];

            int obj1 = vertices_obj[i1];
            const auto& od1 = obj_data[obj1];
            float thickness1 = od1.thickness;
            int layer1 = od1.collision_layer;

            float combined_thickness = thickness0 + thickness1;
            int layer_diff = layer0 - layer1;

            float u, v, w, pen = 0.f;
            float3 normal;
            compute_point_triangle_contact(x0, x1, x2, x3,
                combined_thickness, layer_diff, side_sign,
                vertex_normal,
                normal, u, v, w, pen);

            update_contact_state(states[j], pen, beta);
        }
    }
}
__device__ inline float3 clamp_to_trajectory_envelope(
    const float3& A,        // pos_prev
    const float3& B,        // pos_target_initial
    const float3& P,        // pos_new
    const float margin)     // Broad-phase query R
{
    float3 AB = B - A;
    float AB_len_sq = dot(AB, AB);

    float t = 0.0f;
    if ( AB_len_sq > 1e-12f ) {
        t = dot(P - A, AB) / AB_len_sq;
        t = clamp(t, 0.0f, 1.0f);
    }

    float3 C = A + t * AB;
    float3 diff = P - C;
    float dist2 = len_sq(diff);

    //  forcibly project back to the tube wall
    if ( dist2 > margin * margin ) {
        return C + diff * (margin * rsqrtf(dist2));
    }

    return P;
}
__global__ void apply_force_color_kernel(
    float3* __restrict__ pos,
    const float3* __restrict__ pos_target,
    const float3* __restrict__ pos_prev,
    const float* __restrict__ static_diags,
    const float3* __restrict__ f_collision,
    const float3* __restrict__ f_elastic,
    const Mat3* __restrict__ particle_hessians,
    const char* __restrict__ vertices_mask,
    float max_displacement,
    const int* __restrict__ color_groups,
    int truncation_type,
    int color_groups_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= color_groups_size ) return;
    int vid = color_groups[i];
    if ( vertices_mask[vid] ) return;
    Mat3 hess = particle_hessians[vid];
    // hess.add_diag(static_diags[vid]);
    float3 f = f_collision[vid] + f_elastic[vid];
    float3 disp = hess.inverse() * f;
    float3 ori_disp = pos[vid] - pos_prev[vid];
    if ( truncation_type == 1 ) {
        float3 pos_new = pos[vid] + disp;
        pos[vid] = clamp_to_trajectory_envelope(pos_prev[vid], pos_target[vid],
            pos_new, max_displacement);
    }
    else {
        float len2 = dot(disp, disp);
        if ( len2 > max_displacement * max_displacement ) {
            disp = disp * (max_displacement * rsqrtf(len2));
        }
        pos[vid] += disp;
    }

    if ( vid == debug_v_id ) {
        if ( len_sq(f) < 1e-6f ) {
            printf("[vid=%d] disp=0............\n", debug_v_id);
            return;
        }
        printf("====================================================\n");
        printf("[vid=%d] disp=(%e,%e,%e), ori_disp=(%e,%e,%e)\n", debug_v_id,
            disp.x, disp.y, disp.z,
            ori_disp.x, ori_disp.y, ori_disp.z);
        printf("  force=(%e,%e,%e)\n",
            f.x, f.y, f.z);
        printf("  Hessian:\n");
        printf("    (%e,%e,%e)\n", hess.r[0].x, hess.r[0].y, hess.r[0].z);
        printf("    (%e,%e,%e)\n", hess.r[1].x, hess.r[1].y, hess.r[1].z);
        printf("    (%e,%e,%e)\n", hess.r[2].x, hess.r[2].y, hess.r[2].z);
        printf("  static_diags=%e, max_displacement=%e\n",
            static_diags[vid], max_displacement);
        printf("  pos_prev=(%e,%e,%e), new_pos=(%e,%e,%e)\n",
            pos_prev[vid].x, pos_prev[vid].y, pos_prev[vid].z,
            pos[vid].x, pos[vid].y, pos[vid].z);
    }
}
__global__ void apply_truncation_ts_color_kernel(
    float3* __restrict__ pos,
    const float3* __restrict__ pos_saved,
    const float3* __restrict__ dx,
    const float* __restrict__ truncation_ts,
    const char* __restrict__ vertices_mask,
    float max_displacement,
    const int* __restrict__ color_groups,
    int color_groups_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= color_groups_size ) return;
    int vid = color_groups[i];
    if ( vertices_mask[vid] ) return;
    float t = truncation_ts ? truncation_ts[vid] : 1.f;
    float3 disp = dx[vid] * t;

    float len2 = dot(disp, disp);
    if ( len2 > max_displacement * max_displacement ) {
        disp = disp * (max_displacement / sqrtf(len2));
    }
    pos[vid] = pos_saved[vid] + disp;
    // pos_saved[vid] = pos[vid];
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
    float3* q_pred = geo->pos_temp.data().get();
    const float3* q_prev = geo->pos_step_prev.data().get();
    float3* q_inertia = geo->pos_inertia.data().get();
    float3* dx = this->displacement.data().get();
    float3* v = geo->velocities.data().get();
    float3* v_prev = geo->vel_prev.data().get();
    float3* f = geo->forces.data().get();
    float3* f_elastic = geo->elastic_forces.data().get();
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
    float query_radius = max(0.001f, get_global_parameter("query_radius", 0.01f));
    auto& contact = geo->get_contact();
    float* truncation_t = contact.truncation_t.data().get();
    auto* vf_states = this->vf_states.data().get();
    auto* ee_states = this->ee_states.data().get();
    cudaMemsetAsync(static_diags, 0, params.nb_all_vertices * sizeof(float));

    forward_step<<<(n + block - 1) / block, block>>>(
        v, v_prev, mass_inv,
        nullptr, f_elastic,
        mask, q, q_pred, q_inertia, nullptr,
        static_diags,
        h, 1e2, geo->gravity, true, n);
    contact.collision_detect_broad_phase_stated(q_prev, q_pred,
        vf_states, ee_states, query_radius);

    int iters = max(1, (int)get_global_parameter("vbd_iters", 10));
    float damping = max(0.f, get_global_parameter("vbd_damping", 0.f));
    float vf_ground_k = max(0.f, geo->get_global_parameter("vf_ground_k", 0.2f));
    float vf_force_k = max(0.f, geo->get_global_parameter("vf_force_k", 0.2f));
    float ee_force_k = max(0.f, geo->get_global_parameter("ee_force_k", 0.2f));
    float contact_beta = max(0.f, geo->get_global_parameter("avbd_contact_beta", 10000.f));
    int h_debug_e_id = (int)get_global_parameter("debug_e_id", -1);
    int h_debug_v_id = (int)get_global_parameter("debug_v_id", -1);
    cudaMemcpyToSymbol((const void*)&debug_e_id, &h_debug_e_id, sizeof(int));
    cudaMemcpyToSymbol((const void*)&debug_v_id, &h_debug_v_id, sizeof(int));
    bool ground = geo->ground;
    // float ground_f = max(0.f, (get_global_parameter("ground_f", 1e3)));

    int num_colors = geo->h_colors_index_offsets.size() - 1;
    int* color_groups = geo->color_groups.data().get();
    constexpr int dynamics_block_size = 8;
    // cudaMemcpyAsync(q, q_inertia,
    //     active_vertices_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    int total_threads = kNumThreadsPerPrimitive * max(num_vertices, num_edges);
    for ( int i = 0; i < iters; i++ ) {
        // cudaMemcpyAsync(q_saved, q,
        //     num_vertices * sizeof(float3), cudaMemcpyDeviceToDevice);
        cudaMemsetAsync(f, 0, active_vertices_size * sizeof(float3));
        cudaMemsetAsync(f_elastic, 0, active_vertices_size * sizeof(float3));
        cudaMemsetAsync(Jx_diag, 0, active_vertices_size * sizeof(Mat3));
        // cudaMemsetAsync(dx, 0, active_vertices_size * sizeof(float3));
        for ( int c = 0; c < num_colors; c++ ) {
            int color_index = geo->h_colors_index_offsets[c];
            int color_size = geo->h_colors_index_offsets[c + 1] - color_index;
            int* color_group_begin = color_groups + color_index;

            solve_elasticity_springs_kernel<dynamics_block_size><<<color_size, dynamics_block_size>>>
                (q, f_elastic, Jx_diag, q_prev, mass_inv, static_diags, q_inertia,
                obj_data, vertices_obj,
                edges, geo->edge_lengths.data().get(),
                geo->edge_lookup.data().get(),
                geo->dir_edges.data().get(),
                color_group_begin, damping, h,
                color_size);

            vbd_self_contact_kernel<<<(total_threads + block - 1) / block, block>>>(
                f, Jx_diag,
                q, geo->node_colors.data().get(),
                obj_data, vertices_obj,
                tris, edges, geo->edge_opposite_points.data().get(),
                geo->vertex_normals.data().get(), geo->edge_normals.data().get(),
                contact.broad_phase_vf.data().get(), broad_phase_size,
                contact.broad_phase_ee.data().get(), broad_phase_size,
                vf_states, ee_states,
                c, h, ground, active_vertices_size, num_vertices, num_edges);

            apply_force_color_kernel<<<(color_size + block - 1) / block, block>>>(
                q, q_pred, q_prev, static_diags, f, f_elastic, Jx_diag, mask, query_radius, color_group_begin, 1, color_size);
        }
        vbd_contact_dual_update_kernel<<<(total_threads + block - 1) / block, block>>>(
            q, obj_data, vertices_obj,
            tris, edges,
            geo->vertex_normals.data().get(), geo->edge_normals.data().get(),
            contact.broad_phase_vf.data().get(), broad_phase_size,
            contact.broad_phase_ee.data().get(), broad_phase_size,
            vf_states, ee_states, contact_beta,
            ground, num_vertices, num_edges);
    }
    // Update collision state using new position for next frame.
    for ( int c = 0; c < num_colors; c++ ) {
        vbd_self_contact_kernel<<<(total_threads + block - 1) / block, block>>>(
            f, Jx_diag,
            q, geo->node_colors.data().get(),
            obj_data, vertices_obj,
            tris, edges, geo->edge_opposite_points.data().get(),
            geo->vertex_normals.data().get(), geo->edge_normals.data().get(),
            contact.broad_phase_vf.data().get(), broad_phase_size,
            contact.broad_phase_ee.data().get(), broad_phase_size,
            vf_states, ee_states,
            c, h, ground, active_vertices_size, num_vertices, num_edges);
    }
    // contact.ccd_truncation_traverse_bvh(q_prev, q);
    // thrust::fill(thrust::cuda::par_nosync, contact.truncation_t.begin(), contact.truncation_t.end(), 1.f);
    n = active_vertices_size;
    cudaMemcpyAsync(v_prev, v, n * sizeof(float3), cudaMemcpyDeviceToDevice);
    step_end_kernel<<<(n + block - 1) / block, block>>>(
        v, q, nullptr,
        q_prev, mask, h, max_vel, n);
    // contact.check_truncation_traverse_bvh(q_prev, q);
}
