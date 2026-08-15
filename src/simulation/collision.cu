// #include "contact/collision.cuh"

#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "collision_debug.cuh"
#include "geometry.cuh"
#include "common/math_utils.h"
#include "contact/hash.cuh"
#include "contact/contact.cuh"
#include "contact/lbvh.cuh"
#include "contact/collision_detection.cuh"
#include "cuda_tools/cub_tools.cuh"


void Contact::compute_edge_ranks() {
    const int n = geo->edge_lengths.size();

    thrust::device_vector<int> indices(n);
    thrust::sequence(thrust::cuda::par_nosync, indices.begin(), indices.end());
    thrust::device_vector<float> lengths_copy = geo->edge_lengths;

    thrust::sort_by_key(thrust::cuda::par_nosync,
        lengths_copy.begin(), lengths_copy.end(),
        indices.begin());
    thrust::device_vector<int> ranks(n);
    thrust::sequence(thrust::cuda::par_nosync, ranks.begin(), ranks.end());
    thrust::scatter(thrust::cuda::par_nosync,
        ranks.begin(), ranks.end(),
        indices.begin(),
        edge_sorted_rank.begin());
}
void Contact::init() {
    auto& params = geo->params;
    // point_hash_table_size = max(67, next_prime((uint32_t)params.nb_all_cloth_vertices));
    // edge_hash_table_size = max(67, next_prime((uint32_t)params.nb_all_cloth_edges));
    // point_hash_table.resize(point_hash_table_size);
    // max_pp_result_size = params.nb_all_cloth_vertices * 2;
    // max_tp_result_size = params.nb_all_triangles * 2;
    // max_ee_result_size = params.nb_all_edges * 2;
    // max_point_hashes_size = max_pp_result_size * 8;
    // max_edge_hashes_size = params.nb_all_cloth_edges * 8;
    // point_hashes.resize(max_point_hashes_size);
    // edge_hashes.resize(max_edge_hashes_size);
    // max_sort_result_size = max(max_point_hashes_size, max_edge_hashes_size);
    // sort_key_temp.resize(max_sort_result_size);
    //
    // sort_result_size.resize(1);
    // pp_result_size.resize(1);
    // tp_result_size.resize(1);
    // ee_result_size.resize(1);
    // uncolored_count.resize(1);
    // vertex_color_claimer.resize(params.nb_all_vertices * MAX_COLORS);
    // vertex_forbidden_masks.resize(params.nb_all_vertices);
    //
    // pp_collision_result.resize(max_pp_result_size);
    // tp_collision_result.resize(max_tp_result_size);
    // ee_collision_result.resize(max_ee_result_size);
    // max_collision_constraints_size = max_pp_result_size + max_tp_result_size + max_ee_result_size;
    // normal_constraints.resize(max_collision_constraints_size);
    // collision_constraints.resize(max_collision_constraints_size);
    // constraint_colors.resize(max_collision_constraints_size);
    // point_hash_table_lookup.resize(point_hash_table_size + 1);
    // edge_hash_table_lookup.resize(edge_hash_table_size + 1);
    // weight.resize(params.nb_all_vertices);
    // constraint_color_offsets.resize(MAX_COLORS + 2);
    // points_safe.reserve(params.nb_all_vertices);
    //
    // alpha_hard = 0.005f;

    // if ( capture_stream != nullptr ) {
    //     cudaStreamDestroy(capture_stream);
    //     capture_stream = nullptr;
    // }
    // cudaStreamCreate(&capture_stream);

    point_sorted_indices.resize(params.nb_all_vertices * 2);
    edge_sorted_indices.resize(params.nb_all_edges);
    edge_sorted_rank.resize(params.nb_all_edges);

    broad_phase_vf.resize(broad_phase_size * params.nb_all_vertices);
    broad_phase_ee.resize(broad_phase_size * params.nb_all_edges);
    broad_phase_ef.resize(broad_phase_size * params.nb_all_edges);
    truncation_t.resize(params.nb_all_vertices);
    tri_bvh = lbvh3d::BVH3D();
    edge_bvh = lbvh3d::BVH3D();
    lbvh3d::initialize(max(params.nb_all_triangles, params.nb_all_edges));
    // rebuild_bvh();
    do_collision_detect_broad_phase_before_step = true;
}
// static __device__ __forceinline__ float3 tri_normal(const float3& x0, const float3& x1, const float3& x2) {
//     return normalized(cross(x1 - x0, x2 - x0));
// }

void Contact::collision_detect_broad_phase(const float3* pos, const float3* pos_target, float query_radius, bool ef) {
    auto& params = geo->params;
    int num_queries = params.nb_all_vertices;
    int threadsPerBlock = 256;

    query_vf_pairs_capsule_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        point_sorted_indices.data().get(),
        num_queries,
        thrust::raw_pointer_cast(tri_bvh.nodes.data()),
        thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
        tri_bvh.root_idx,
        pos,
        thrust::raw_pointer_cast(geo->triangle_indices.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        query_radius,
        pos_target,
        thrust::raw_pointer_cast(broad_phase_vf.data()),
        params.nb_all_cloth_vertices,
        broad_phase_size
        );
    num_queries = params.nb_all_edges;

    query_ee_pairs_capsule_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        pos,
        thrust::raw_pointer_cast(geo->pos_2D.data()),
        edge_sorted_rank.data().get(),
        num_queries,
        thrust::raw_pointer_cast(edge_bvh.nodes.data()),
        thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
        edge_bvh.root_idx,
        thrust::raw_pointer_cast(geo->edges.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        query_radius,
        pos_target,
        thrust::raw_pointer_cast(geo->edge_normals.data()),
        params.nb_all_cloth_vertices,
        thrust::raw_pointer_cast(broad_phase_ee.data()),
        broad_phase_size
        );
    // edges vs faces
    if ( ef )
        query_ef_pairs_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
            thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
            thrust::raw_pointer_cast(edge_bvh.nodes.data()),
            num_queries,
            thrust::raw_pointer_cast(tri_bvh.nodes.data()),
            thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
            thrust::raw_pointer_cast(geo->edges.data()),
            thrust::raw_pointer_cast(geo->obj_data.data()),
            thrust::raw_pointer_cast(geo->vertices_obj.data()),
            thrust::raw_pointer_cast(geo->triangle_indices.data()),
            tri_bvh.root_idx,
            query_radius,
            thrust::raw_pointer_cast(broad_phase_ef.data()),
            broad_phase_size
            );
}
void Contact::collision_detect_prepare() {
    if ( geo->simulator->frame % 20 == 0 ) {
        rebuild_bvh();
    }
    if ( do_collision_detect_broad_phase_before_step ) {
        refit_bvh();
        // broad phase
        collision_detect_broad_phase(geo->pos_world.data().get(), geo->inertial_offset.data().get(), point_radius, true);
    }
    int h_debug_e_id = (int)geo->get_global_parameter("debug_e_id", -1);
    int h_debug_v_id = (int)geo->get_global_parameter("debug_v_id", -1);
    cudaMemcpyToSymbol((const void*)&debug_e_id, &h_debug_e_id, sizeof(int));
    cudaMemcpyToSymbol((const void*)&debug_v_id, &h_debug_v_id, sizeof(int));

}

__global__ void refit_face_offset_bvh_kernel(
    const float3* pos_prev, const float3* pos_target,
    const int3* faces,
    const int2* __restrict__ nodes,
    const unsigned int* __restrict__ parent,
    unsigned int* __restrict__ child_count,
    lbvh3d::AABB3D* __restrict__ aabbs,
    const ObjectDataInput* __restrict__ obj_data,
    const int* vertices_obj,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    unsigned int prim_idx = nodes[i].x - 1;
    int3 f = faces[prim_idx];
    float3 v0 = pos_prev[f.x], v1 = pos_prev[f.y], v2 = pos_prev[f.z];
    aabbs[i].min = fmin3(v0, fmin3(v1, v2));
    aabbs[i].max = fmax3(v0, fmax3(v1, v2));
    v0 = pos_target[f.x];
    v1 = pos_target[f.y];
    v2 = pos_target[f.z];
    aabbs[i].min = fmin3(aabbs[i].min, fmin3(v0, fmin3(v1, v2)));
    aabbs[i].max = fmax3(aabbs[i].max, fmax3(v0, fmax3(v1, v2)));
    float thickness = obj_data[vertices_obj[f.x]].thickness * 1.5f + 1e-5f;
    aabbs[i].min = aabbs[i].min - thickness;
    aabbs[i].max = aabbs[i].max + thickness;

    lbvh3d::bottom_up_refit(i, nodes, parent, child_count, aabbs);
}

__global__ void refit_edge_offset_bvh_kernel(
    const float3* pos_prev, const float3* pos_target,
    const int2* edges,
    const int2* __restrict__ nodes,
    const unsigned int* __restrict__ parent,
    unsigned int* __restrict__ child_count,
    lbvh3d::AABB3D* __restrict__ aabbs,
    const ObjectDataInput* __restrict__ obj_data,
    const int* vertices_obj,
    unsigned int n
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;

    // Compute leaf AABB from edge primitive
    unsigned int prim_idx = nodes[i].x - 1;
    int2 e = edges[prim_idx];
    float3 v0 = pos_prev[e.x], v1 = pos_prev[e.y];
    aabbs[i].min = fmin3(v0, v1);
    aabbs[i].max = fmax3(v0, v1);
    v0 = pos_target[e.x];
    v1 = pos_target[e.y];
    aabbs[i].min = fmin3(v1, fmin3(v0, aabbs[i].min));
    aabbs[i].max = fmax3(v1, fmax3(v0, aabbs[i].max));
    float thickness = obj_data[vertices_obj[e.x]].thickness * 1.5f + 1e-5f;
    aabbs[i].min = aabbs[i].min - thickness;
    aabbs[i].max = aabbs[i].max + thickness;

    lbvh3d::bottom_up_refit(i, nodes, parent, child_count, aabbs);
}
__global__ void compute_rank_kernel(const unsigned int* indices_sorted, unsigned int* rank, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i < n ) {
        rank[indices_sorted[i]] = i;
    }
}
void Contact::rebuild_bvh() {
    lbvh3d::build_face_bvh_wo_refit(geo->pos_world, geo->triangle_indices, tri_bvh);
    lbvh3d::build_edge_bvh_wo_refit(geo->pos_world, geo->edges, edge_bvh);
    cudaMemcpyAsync(edge_sorted_indices.data().get(), lbvh3d::get_sorted_indices(),
        sizeof(unsigned int) * edge_sorted_indices.size(), cudaMemcpyDeviceToDevice);
    // int block = 256;
    // int n = geo->params.nb_all_edges;
    // compute_rank_kernel<<<(n + block - 1) / block,block>>>(
    //     edge_sorted_indices.data().get(), edge_sorted_rank.data().get(),
    //     n
    //     );
    compute_edge_ranks(); // sorted by length
    lbvh3d::compute_and_sort_by_morton_codes(geo->pos_world.data().get(),
        geo->pos_world.size(), point_sorted_indices.data().get());
}
void Contact::refit_bvh() {
    lbvh3d::refit_face_bvh(geo->pos_world.data().get(), geo->triangle_indices, tri_bvh, nullptr);
    lbvh3d::refit_edge_bvh(geo->pos_world.data().get(), geo->edges, edge_bvh, nullptr);
}

__global__ void compute_vf_force(
    float3* __restrict__ forces,
    Mat3* __restrict__ Jx,
    // float* __restrict__ weight,
    const int* __restrict__ broad_phase_pairs,
    const int broad_phase_size,
    const float3*__restrict__ pos,
    const float3*__restrict__ vertex_normals,
    const int3* tri_indices,
    const float* __restrict__ static_diags,
    const ObjectDataInput* __restrict__ obj_data,
    const int* vertices_obj,
    const int force_type, // 0: spring, 1: IPC
    const int active_vertices_size,
    const bool ground,
    const float k,
    const float ground_k,
    int num_vertices
) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= num_vertices ) return;
    // if ( static_diags[vid] <= 0.f ) return;
    float vert_stiff = static_diags[vid];
    float3 x0 = pos[vid];
    const auto& od = obj_data[vertices_obj[vid]];
    float thickness0 = od.thickness;
    const int* pairs = &broad_phase_pairs[vid * broad_phase_size];
    float3 normal_v = vertex_normals[vid];
    int layer0 = od.collision_layer;

    bool is_collided = false;
    float3 force0 = { 0.0f, 0.0f, 0.0f };
    Mat3 hess0 = Mat3::zero();
    if ( ground && x0.z < thickness0 ) {
        force0.z = (thickness0 - x0.z) * vert_stiff * ground_k;
        // outer_product of (0,0,1)
        hess0.r[2].z = vert_stiff * ground_k;
        is_collided = true;
    }
    int count = pairs[0];
    for ( int i = 1; i <= count; ++i ) {// [1, count]
        int fid = pairs[i];
        float sign = fid > 0 ? 1.0f : -1.0f;
        fid = abs(fid);
        int3 tri = tri_indices[fid];
        int i1 = tri.x, i2 = tri.y, i3 = tri.z;
        float3 x1 = pos[i1], x2 = pos[i2], x3 = pos[i3];

        float3 normal;
        float thickness = thickness0 + obj_data[vertices_obj[i1]].thickness;
        float u, v, w, pen;
        if ( !compute_point_triangle_contact(
            x0, x1, x2, x3,
            thickness,
            layer0 - obj_data[vertices_obj[i1]].collision_layer,
            sign, normal_v,
            normal, u, v, w, pen)
        ) {

            continue;
        }
        float stiff = k * vert_stiff;
        if ( static_diags[i1] > 0.f && static_diags[i2] > 0.f && static_diags[i3] > 0.f ) {
            float face_stiff = (static_diags[i1] + static_diags[i2] + static_diags[i3]) * 0.333333f;
            stiff = k * (vert_stiff * face_stiff) / (vert_stiff + face_stiff);
        }
        else if ( vert_stiff <= 0.f ) continue;

        float3 force;
        if ( force_type == 0 ) {
            float force_mag = stiff * pen;
            force = normal * force_mag;
            if ( Jx ) {
                Mat3 hess = Mat3::outer_product(normal, normal * stiff);
                hess0 += hess;
                if ( i1 < active_vertices_size ) {
                    atomicAddMat3(&Jx[i1], hess * (u * u));
                    atomicAddMat3(&Jx[i2], hess * (v * v));
                    atomicAddMat3(&Jx[i3], hess * (w * w));
                }
            }
        }
        else {
            float d = max(thickness - pen, thickness * 0.1f);
            float d_ratio = d / thickness;
            float diff = thickness - d;
            float log_term = logf(d_ratio);
            log_term = min(log_term, 1e8f);

            float E_prime = stiff * diff * (2.0f * log_term + 1.0f - 1.0f / d_ratio);
            force = log_term * E_prime * normal;
        }
        force0 = force0 + force;
        atomicAddFloat3(&forces[i1], force * -u);
        atomicAddFloat3(&forces[i2], force * -v);
        atomicAddFloat3(&forces[i3], force * -w);

        is_collided = true;
    }
    if ( is_collided ) {
        atomicAddFloat3(&forces[vid], force0);
        if ( Jx && vid < active_vertices_size ) atomicAddMat3(&Jx[vid], hess0);
    }
}
static __global__ void compute_ee_force(
    float3* __restrict__ forces,
    Mat3* __restrict__ Jx,
    const int* __restrict__ broad_phase_pairs,
    const int broad_phase_size,
    const float3*__restrict__ pos,
    const int2* edges,
    const float* __restrict__ static_diags,
    const ObjectDataInput* obj_data,
    const int* vertices_obj,
    const float3* __restrict__ edge_normals,
    const int force_type, // 0: spring, 1: IPC
    const int active_vertices_size,
    const float k,
    int num_edges
) {
    int eid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( eid >= num_edges ) return;
    // if ( static_diags[vid] <= 0.f ) return;
    int2 edge = edges[eid];
    float e1_stiff = (static_diags[edge.x] + static_diags[edge.y]) * 0.5f;
    float3 p0 = pos[edge.x];
    float3 p1 = pos[edge.y];
    // float3 E = p1 - p0;
    float thickness0 = obj_data[vertices_obj[edge.x]].thickness;
    int layer0 = obj_data[vertices_obj[edge.x]].collision_layer;
    float3 edge_normal0 = edge_normals[eid];
    float3 force0 = { 0.0f, 0.0f, 0.0f };
    float3 force1 = { 0.0f, 0.0f, 0.0f };
    Mat3 hess0 = Mat3::zero();
    Mat3 hess1 = Mat3::zero();

    const int* pairs = &broad_phase_pairs[eid * broad_phase_size];
    int count = pairs[0];
    bool is_collided = false;
    for ( int i = 1; i <= count; ++i ) {// [1, count]
        int eid2 = pairs[i];
        float sign = eid2 > 0 ? 1.0f : -1.0f;
        eid2 = abs(eid2);
        int2 e = edges[eid2];
        float3 q0 = pos[e.x], q1 = pos[e.y];

        float thickness = thickness0 + obj_data[vertices_obj[e.x]].thickness;
        // if ( dist > thickness ) continue;
        float s, t;
        float3 normal;
        float pen;
        if ( !compute_edge_edge_contact(
            p0, p1, q0, q1,
            thickness,
            layer0 - obj_data[vertices_obj[e.x]].collision_layer,
            sign, edge_normal0, edge_normals[eid2],
            s, t, normal, pen)
        ) {
            continue;
        }
        float stiff = k * e1_stiff;
        if ( static_diags[e.x] > 0.f || static_diags[e.y] > 0.f ) {
            float e2_stiff = (static_diags[e.x] + static_diags[e.y]) * 0.5f;
            if ( e1_stiff > 0.f )
                stiff = k * (e1_stiff * e2_stiff) / (e1_stiff + e2_stiff);
            else
                stiff = k * e2_stiff;
        }
        if ( stiff <= 0.f ) continue;

        float3 force;
        if ( force_type == 0 ) {
            float force_mag = stiff * pen;
            force = normal * force_mag;
            if ( Jx ) {
                Mat3 hess = Mat3::outer_product(normal, normal) * stiff;
                hess0 += hess * ((1.0f - s) * (1.0f - s));
                hess1 += hess * (s * s);
                if ( e.x < active_vertices_size ) {
                    atomicAddMat3(&Jx[e.x], hess * ((1.0f - t) * (1.0f - t)));
                    atomicAddMat3(&Jx[e.y], hess * (t * t));
                }
            }
        }
        else {
            float d = max(thickness - pen, thickness * 0.05f);
            float d_ratio = d / thickness;
            float log_term = logf(d_ratio);
            log_term = min(log_term, 1e8f);
            float diff = thickness - d;

            float E_prime = stiff * diff * (2.0f * log_term + 1.0f - 1.0f / d_ratio);
            force = log_term * E_prime * normal;
        }
        force0 = force0 + force * (1.0f - s);
        force1 = force1 + force * s;
        atomicAddFloat3(&forces[e.x], force * (t - 1.f));
        atomicAddFloat3(&forces[e.y], force * -t);

        is_collided = true;
    }
    if ( is_collided && edge.x < active_vertices_size ) {
        atomicAddFloat3(&forces[edge.x], force0);
        atomicAddFloat3(&forces[edge.y], force1);
        if ( Jx ) {
            atomicAddMat3(&Jx[edge.x], hess0);
            atomicAddMat3(&Jx[edge.y], hess1);
        }
    }
}

__device__ float3 intersection_gradient_vector(const float3& R, const float3& E, const float3& N) {
    float dot_EN = dot(E, N);
    if ( fabsf(dot_EN) > 1e-6f ) {
        return R - 2.0f * N * (dot(E, R) / dot_EN);
    }
    else {
        return R;
    }
}
__global__ void solve_untangling_kernel(
    const float3* pos,
    const int3* tri_indices,
    const int2* edges,
    const int2* edge_opposite_points,
    const int* broad_phase_pairs,
    const float* static_diags,
    const ObjectDataInput* obj_data,
    const int* vertices_obj,
    float3* forces,
    Mat3* Jx,
    const int active_vertices_size,
    float k,
    int broad_phase_size,
    int num_edges
) {
    int eid = threadIdx.x + blockIdx.x * blockDim.x;
    if ( eid >= num_edges ) return;

    int2 edge = edges[eid];

    int2 eop = edge_opposite_points[eid];
    int e0_aux0 = eop.x, e0_aux1 = eop.y;
    float3 v0 = pos[edge.x], v1 = pos[edge.y];
    float thickness = obj_data[vertices_obj[edge.x]].thickness;

    float len0 = norm(v0 - v1);
    if ( len0 < 5e-4f ) return;

    float3 force0 = { 0.0f, 0.0f, 0.0f };
    float3 force1 = { 0.0f, 0.0f, 0.0f };
    Mat3 hess0 = Mat3::zero();
    Mat3 hess1 = Mat3::zero();
    float stiff_0 = (static_diags[edge.x] + static_diags[edge.y]) * 0.5f;
    int is_collided = 0;

    float3 E = normalized(v1 - v0);
    float3 N2 = { 0.0f, 0.0f, 0.0f };
    float3 N3 = { 0.0f, 0.0f, 0.0f };
    if ( e0_aux0 >= 0 ) N2 = normalized(cross(E, pos[e0_aux0] - v0));
    if ( e0_aux1 >= 0 ) N3 = normalized(cross(pos[e0_aux1] - v0, E));

    const int* pairs = &broad_phase_pairs[eid * broad_phase_size];

    int count = pairs[0];

    for ( int i = 1; i <= count; ++i ) {
        int fid = pairs[i];

        int3 tri = tri_indices[fid];
        int f0 = tri.x, f1 = tri.y, f2 = tri.z;
        float3 x0 = pos[f0], x1 = pos[f1], x2 = pos[f2];

        float3 face_normal = cross(x1 - x0, x2 - x0);
        float normal_len = norm(face_normal);

        if ( normal_len < 1e-8f ) continue;
        face_normal = face_normal / normal_len;

        float d1 = dot(face_normal, v0 - x0);
        float d2 = dot(face_normal, v1 - x0);

        if ( d1 * d2 >= 0.0f ) continue;

        float abs_d1 = fabsf(d1);
        float abs_d2 = fabsf(d2);
        float3 hit_point = (v0 * abs_d2 + v1 * abs_d1) / (abs_d2 + abs_d1);
        float u, v, w;
        barycentric(x0, x1, x2, hit_point, u, v, w);
        if ( u < 1e-2f || v < 1e-2f || w < 1e-2f ) continue;

        float3 G = { 0.0f, 0.0f, 0.0f };

        if ( e0_aux0 >= 0 ) {
            float3 R = normalized(cross(face_normal, N2));
            if ( dot(cross(E, R), cross(E, pos[e0_aux0] - hit_point)) < 0.0f ) R = -R;
            G = G + intersection_gradient_vector(R, E, face_normal);
        }

        if ( e0_aux1 >= 0 ) {
            float3 R = normalized(cross(face_normal, N3));
            if ( dot(cross(E, R), cross(E, pos[e0_aux1] - hit_point)) < 0.0f ) R = -R;
            G = G + intersection_gradient_vector(R, E, face_normal);
        }

        if ( len_sq(G) < 1e-16f ) continue;
        G = normalized(G);

        float stiff_1 = (static_diags[f0] + static_diags[f1] + static_diags[f2]) / 3.0f;
        if ( stiff_0 + stiff_1 <= 0.f ) continue;
        float stiff = k * (stiff_0 * stiff_1) / (stiff_0 + stiff_1);
        // float disp = 2.0f * thickness;
        // float3 force = G * (stiff * disp);
        float penetration_depth = fabsf(d1) + fabsf(d2);
        float max_disp = 2.0f * thickness;
        float effective_disp = fminf(penetration_depth, max_disp);
        float3 force = G * (stiff * effective_disp);
        float edge_bary0 = abs_d2 / (abs_d1 + abs_d2);
        float edge_bary1 = 1 - edge_bary0;
        force0 = force0 + force * edge_bary0;
        force1 = force1 + force * edge_bary1;

        atomicAddFloat3(&forces[f0], force * -u);
        atomicAddFloat3(&forces[f1], force * -v);
        atomicAddFloat3(&forces[f2], force * -w);

        if ( Jx ) {
            Mat3 hess = Mat3::outer_product(G, G) * stiff;
            hess0 += hess * (edge_bary0 * edge_bary0);
            hess1 += hess * (edge_bary1 * edge_bary1);
            if ( f0 < active_vertices_size ) {
                atomicAddMat3(&Jx[f0], hess * (u * u));
                atomicAddMat3(&Jx[f1], hess * (v * v));
                atomicAddMat3(&Jx[f2], hess * (w * w));
            }
        }
        is_collided = 1;
    }

    if ( is_collided && edge.x < active_vertices_size ) {
        atomicAddFloat3(&forces[edge.x], force0);
        atomicAddFloat3(&forces[edge.y], force1);
        if ( Jx ) {
            atomicAddMat3(&Jx[edge.x], hess0);
            atomicAddMat3(&Jx[edge.y], hess1);
        }
    }
}
void Contact::accumulate_contact_force(float3* forces, Mat3* Jx_diag) {
    auto& params = geo->params;
    int num_vertices = params.nb_all_vertices;
    int num_edges = params.nb_all_edges;
    int threadsPerBlock = 256;
    float vf_force_k = max(0.f, geo->get_global_parameter("vf_force_k", 0.2f));
    float vf_ground_k = max(0.f, geo->get_global_parameter("vf_ground_k", 0.2f));
    float ee_force_k = max(0.f, geo->get_global_parameter("ee_force_k", 0.2f));
    float ef_force_k = max(0.f, geo->get_global_parameter("ef_force_k", 0.2f));
    int vf_force_type = max(0, (int)geo->get_global_parameter("vf_force_type", 1));
    int ee_force_type = max(0, (int)geo->get_global_parameter("ee_force_type", 1));
    // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
    // std::vector<type> __##v(_##v.begin(), _##v.end())
    // auto& edge_opposite_points = geo->edge_opposite_points;
    // CHECK(edge_opposite_points, int2);
    // #undef CHECK
    int cloth_vertices = params.nb_all_cloth_vertices;
    compute_vf_force<<<(num_vertices + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        forces,
        Jx_diag,
        broad_phase_vf.data().get(),
        broad_phase_size,
        geo->pos_world.data().get(),
        geo->vertex_normals.data().get(),
        geo->triangle_indices.data().get(),
        geo->static_diags.data().get(),
        geo->obj_data.data().get(),
        geo->vertices_obj.data().get(),
        vf_force_type,
        cloth_vertices,
        geo->ground,
        vf_force_k, vf_ground_k,
        num_vertices
        );
    compute_ee_force<<<(num_edges + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        forces,
        Jx_diag,
        broad_phase_ee.data().get(),
        broad_phase_size,
        geo->pos_world.data().get(),
        geo->edges.data().get(),
        geo->static_diags.data().get(),
        geo->obj_data.data().get(),
        geo->vertices_obj.data().get(),
        geo->edge_normals.data().get(),
        ee_force_type, cloth_vertices,
        ee_force_k, num_edges
        );

    solve_untangling_kernel<<<(num_edges + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        geo->pos_world.data().get(),
        geo->triangle_indices.data().get(),
        geo->edges.data().get(),
        geo->edge_opposite_points.data().get(),
        broad_phase_ef.data().get(),
        geo->static_diags.data().get(),
        geo->obj_data.data().get(),
        geo->vertices_obj.data().get(),
        forces,
        Jx_diag,
        cloth_vertices,
        ef_force_k,
        broad_phase_size, num_edges
        );
}

void Contact::refit_bvh_with_target(const float3* pos_prev, const float3* pos_target) {
    auto& params = geo->params;
    int block = 256;
    int n = params.nb_all_edges;
    unsigned int num_nodes = 2 * n - 1;
    auto child_count = (unsigned int*)get_device_temp_memory(nullptr, num_nodes * sizeof(unsigned int));
    cudaMemsetAsync(child_count, 0, num_nodes * sizeof(unsigned int));
    refit_edge_offset_bvh_kernel<<<(n + block - 1) / block, block>>>(
        pos_prev, pos_target,
        thrust::raw_pointer_cast(geo->edges.data()),
        thrust::raw_pointer_cast(edge_bvh.nodes.data()),
        thrust::raw_pointer_cast(edge_bvh.parent.data()),
        child_count,
        thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        n);
    n = params.nb_all_triangles;
    num_nodes = 2 * n - 1;
    child_count = (unsigned int*)get_device_temp_memory(nullptr, num_nodes * sizeof(unsigned int));
    cudaMemsetAsync(child_count, 0, num_nodes * sizeof(unsigned int));
    refit_face_offset_bvh_kernel<<<(n + block - 1) / block, block>>>(
        pos_prev, pos_target,
        thrust::raw_pointer_cast(geo->triangle_indices.data()),
        thrust::raw_pointer_cast(tri_bvh.nodes.data()),
        thrust::raw_pointer_cast(tri_bvh.parent.data()),
        child_count,
        thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        n);


}
void Contact::ccd_truncation_traverse_bvh(const float3* pos_prev, const float3* pos_target) {
    thrust::fill(thrust::cuda::par_nosync,
        truncation_t.begin(), truncation_t.end(), 1.f);
    float gamma_min = max(-1.f, geo->get_global_parameter("gamma_min", 1e-9f));
    if ( gamma_min == -1.f ) return;

    refit_bvh_with_target(pos_prev, pos_target);

    auto& params = geo->params;
    int block = 256;
    int n = params.nb_all_vertices;
    float parallel_eps = max(0.f, geo->get_global_parameter("parallel_eps", 1e-6f));
    float gamma_r = max(0.f, geo->get_global_parameter("gamma_r", 0.9f));

    vf_collision_planar_truncation_bvh_kernel<<<(n + block - 1) / block, block>>>(
        truncation_t.data().get(), point_sorted_indices.data().get(), n,
        tri_bvh.nodes.data().get(), tri_bvh.aabbs.data().get(), tri_bvh.root_idx,
        pos_prev, pos_target,
        geo->vertex_normals.data().get(), geo->triangle_indices.data().get(),
        geo->obj_data.data().get(), geo->vertices_obj.data().get(),
        params.nb_all_cloth_vertices,
        parallel_eps, gamma_r, gamma_min);
    n = params.nb_all_edges;
    ee_collision_planar_truncation_bvh_kernel<<<(n + block - 1) / block, block>>>(
        truncation_t.data().get(), n, edge_bvh.nodes.data().get(),
        edge_bvh.aabbs.data().get(), edge_bvh.root_idx,
        pos_prev, pos_target,
        geo->edge_normals.data().get(), geo->edges.data().get(),
        geo->obj_data.data().get(), geo->vertices_obj.data().get(),
        params.nb_all_cloth_vertices,
        parallel_eps, gamma_r, gamma_min);
    // CUDA_CHECK(cudaDeviceSynchronize());

}


// static __global__ void apply_truncation_t0(
//     float3* __restrict__ pos_target,
//     const float* __restrict__ truncation_ts,
//     const float3* __restrict__ pos_prev,
//     const int n
// ) {
//     for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
//           i += blockDim.x * gridDim.x ) {
//         if ( truncation_ts[i] == 0.f )
//             pos_target[i] = pos_prev[i];
//     }
// }
void Contact::check_truncation_traverse_bvh(const float3* pos_prev, float3* pos_target) {
    auto& params = geo->params;
    int block = 256;
    int n = params.nb_all_edges;
    geo->debug_colors.assign(params.nb_all_cloth_vertices, make_float3(1.0f, 1.0f, 1.0f));
    float parallel_eps = max(0.f, geo->get_global_parameter("parallel_eps", 1e-6f));
    float gamma_r = max(0.f, geo->get_global_parameter("gamma_r", 0.9f));
    float gamma_min = max(0.f, geo->get_global_parameter("gamma_min", 1e-9f));
    auto debug_lock = (int*)get_device_temp_memory(nullptr, sizeof(int));
    cudaMemsetAsync(debug_lock, 0, sizeof(int));
    check_ef_pairs_kernel<<<(n + block - 1) / block, block>>>(
        geo->debug_colors.data().get(),
        truncation_t.data().get(),
        edge_sorted_indices.data().get(),
        edge_sorted_rank.data().get(),
        thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
        n,
        thrust::raw_pointer_cast(tri_bvh.nodes.data()),
        thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
        thrust::raw_pointer_cast(geo->edges.data()),
        thrust::raw_pointer_cast(geo->triangles.data()),
        pos_prev, pos_target,
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        thrust::raw_pointer_cast(geo->triangle_indices.data()),
        tri_bvh.root_idx, parallel_eps, gamma_r, gamma_min, debug_lock
        );
    // int h_debug_lock;
    // cudaMemcpy(&h_debug_lock,debug_lock,sizeof(int),cudaMemcpyDeviceToHost);
    // if(h_debug_lock) {
    //     throw std::runtime_error("check_ef_pairs_kernel faild");
    // }
    // n = params.nb_all_cloth_vertices;
    // apply_truncation_t0<<<(n + block - 1) / block, block>>>(
    //     pos_target,
    //     truncation_t.data().get(),
    //     pos_prev, n
    //     );
}
__global__ void collect_all_edge_collisions_debug(
    int target_eid,                     // edge index to debug
    const int* __restrict__ broad_phase_pairs,
    int num_edges,
    int broad_phase_size,               // max neighbors per edge
    const float3* __restrict__ pos,
    const int2* edges,
    const float* __restrict__ static_diags,
    const ObjectDataInput* obj_data,
    const int* vertices_obj,
    const float3* __restrict__ edge_normals,
    const int force_type,               // 0: spring, 1: IPC
    const float k,
    int* valid_out,                     // 1 if collision contributes, 0 otherwise
    float3* force_out,                  // force on target edge contact point
    float2* st_out,                     // barycentric coords on target edge
    int* other_edge_out,                // index of the other edge in the pair
    int* counter                        // atomic output index
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = num_edges * broad_phase_size;
    if ( idx >= total_pairs ) return;

    int eid = idx / broad_phase_size;           // owner edge of this slot
    int slot = idx % broad_phase_size;

    // Locate the neighbor list for edge eid
    const int* pairs = &broad_phase_pairs[eid * broad_phase_size];
    int count = pairs[0];                       // number of valid neighbors

    // slot 0 holds count; only process slots within valid range
    if ( slot == 0 || slot > count ) return;

    int eid2_raw = pairs[slot];
    float sign = eid2_raw > 0 ? 1.0f : -1.0f;       // sign indicates orientation
    int eid2 = abs(eid2_raw);

    // Only process pairs where the target edge is involved
    bool is_target_e1 = (eid == target_eid);
    bool is_target_e2 = (eid2 == target_eid);
    if ( !is_target_e1 && !is_target_e2 ) return;

    // Gather geometry data for edge eid (e1)
    int2 edge = edges[eid];
    float3 p0 = pos[edge.x];
    float3 p1 = pos[edge.y];
    float thickness0 = obj_data[vertices_obj[edge.x]].thickness;
    int layer0 = obj_data[vertices_obj[edge.x]].collision_layer;
    float3 edge_normal0 = edge_normals[eid];
    float e1_stiff = (static_diags[edge.x] + static_diags[edge.y]) * 0.5f;

    // Gather geometry data for edge eid2 (e2)
    int2 e = edges[eid2];
    float3 q0 = pos[e.x];
    float3 q1 = pos[e.y];
    float3 edge_normal1 = edge_normals[eid2];

    float thickness = thickness0 + obj_data[vertices_obj[e.x]].thickness;
    float s, t;
    float3 normal;
    float pen = -1.f;

    // Try to compute a valid contact
    if ( !compute_edge_edge_contact(
        p0, p1, q0, q1,
        thickness,
        layer0 - obj_data[vertices_obj[e.x]].collision_layer,
        sign, edge_normal0, edge_normal1,
        s, t, normal, pen) ) {
        // Contact rejected by compute_edge_edge_contact
        int out_idx = atomicAdd(counter, 1);
        valid_out[out_idx] = 0;
        force_out[out_idx] = make_float3(0.0f, 0.0f, 0.0f);
        st_out[out_idx] = make_float2(0.0f, 0.0f);
        other_edge_out[out_idx] = is_target_e1 ? eid2 : eid;
        printf(
            "id:%d, contact failed, pen: %e, st:(%e, %e) A:(%e, %e, %e), B:(%e, %e, %e), C:(%e, %e, %e), D:(%e, %e, %e), edge_normal1: (%e, %e, %e)\n",
            is_target_e1 ? eid2_raw : eid * (int)sign, pen, s, t,
            p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, q0.x, q0.y, q0.z, q1.x, q1.y, q1.z,
            edge_normal1.x, edge_normal1.y, edge_normal1.z
            );
        return;
    }

    // Compute stiffness
    float stiff = k * e1_stiff;
    if ( static_diags[e.x] > 0.f || static_diags[e.y] > 0.f ) {
        float e2_stiff = (static_diags[e.x] + static_diags[e.y]) * 0.5f;
        if ( e1_stiff > 0.f )
            stiff = k * (e1_stiff * e2_stiff) / (e1_stiff + e2_stiff);
        else
            stiff = k * e2_stiff;
    }
    if ( stiff <= 0.f ) {
        // Stiffness invalid; record as non-contributing
        int out_idx = atomicAdd(counter, 1);
        valid_out[out_idx] = 0;
        force_out[out_idx] = make_float3(0.0f, 0.0f, 0.0f);
        st_out[out_idx] = make_float2(0.0f, 0.0f);
        other_edge_out[out_idx] = is_target_e1 ? eid2 : eid;
        return;
    }

    // Compute force
    float3 force;
    if ( force_type == 0 ) {
        float force_mag = stiff * pen;
        force = normal * force_mag;
        printf("id:%d, force_mag: %e, stiff: %e, pen:%e, st:(%e, %e), A0:(%e, %e, %e), B0:(%e, %e, %e), C0:(%e, %e, %e), D0:(%e, %e, %e)\n", is_target_e1 ? eid2_raw : eid * (int)sign, force_mag,
            stiff, pen, s, t,
            p0.x, p0.y, p0.z, p1.x, p1.y, p1.z, q0.x, q0.y, q0.z, q1.x, q1.y, q1.z);
    }
    else {
        float d = fmaxf(thickness - pen, thickness * 0.05f);
        float d_ratio = d / thickness;
        float log_term = logf(d_ratio);
        log_term = fminf(log_term, 1e8f);
        float diff = thickness - d;
        float E_prime = stiff * diff * (2.0f * log_term + 1.0f - 1.0f / d_ratio);
        force = log_term * E_prime * normal;
    }

    // Write results, adjusting based on whether the target is e1 or e2
    int out_idx = atomicAdd(counter, 1);
    valid_out[out_idx] = 1;   // this pair contributes forces
    if ( is_target_e1 ) {
        force_out[out_idx] = force;
        st_out[out_idx] = make_float2(s, t);
        other_edge_out[out_idx] = eid2_raw;
    }
    else { // target is e2
        // When target is the secondary edge, swap s,t and negate force
        force_out[out_idx] = -force;
        st_out[out_idx] = make_float2(t, s);
        other_edge_out[out_idx] = eid * (int)sign;
    }
}

void Contact::get_check_edge_collision_data(int eid, CheckEdgeCollisionData& res) {

    // Broad-phase data dimensions
    int num_edges = geo->edges.size();

    // Maximum possible output entries (one per broad-phase slot)
    int max_entries = num_edges * broad_phase_size;

    // Allocate device output arrays
    int *d_valid, *d_other_edge, *d_counter;
    float3* d_forces;
    float2* d_st;
    cudaMalloc(&d_valid, max_entries * sizeof(int));
    cudaMalloc(&d_forces, max_entries * sizeof(float3));
    cudaMalloc(&d_st, max_entries * sizeof(float2));
    cudaMalloc(&d_other_edge, max_entries * sizeof(int));
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(d_counter, 0, sizeof(int));

    float ee_force_k = max(0.f, geo->get_global_parameter("ee_force_k", 0.2f));
    int ee_force_type = max(0, (int)geo->get_global_parameter("ee_force_type", 1));
    // Launch one thread per broad-phase slot
    int threadsPerBlock = 256;
    int blocks = (max_entries + threadsPerBlock - 1) / threadsPerBlock;
    collect_all_edge_collisions_debug<<<blocks, threadsPerBlock>>>(
        eid,
        broad_phase_ee.data().get(),
        num_edges,
        broad_phase_size,
        geo->pos_world.data().get(),
        geo->edges.data().get(),
        geo->static_diags.data().get(),
        geo->obj_data.data().get(),
        geo->vertices_obj.data().get(),
        geo->edge_normals.data().get(),
        ee_force_type,
        ee_force_k,
        d_valid,
        d_forces,
        d_st,
        d_other_edge,
        d_counter);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Read back number of written entries
    int actual_count = 0;
    cudaMemcpy(&actual_count, d_counter, sizeof(int), cudaMemcpyDeviceToHost);

    // Copy results to host vectors
    res.nearby_edges.resize(actual_count);
    res.valid.resize(actual_count);
    res.forces.resize(actual_count);
    res.st.resize(actual_count);
    cudaMemcpy(res.nearby_edges.data(), d_other_edge, actual_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(res.valid.data(), d_valid, actual_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(res.forces.data(), d_forces, actual_count * sizeof(float3), cudaMemcpyDeviceToHost);
    cudaMemcpy(res.st.data(), d_st, actual_count * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_valid);
    cudaFree(d_forces);
    cudaFree(d_st);
    cudaFree(d_other_edge);
    cudaFree(d_counter);
}
