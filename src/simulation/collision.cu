#include "contact/collision.cuh"

#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "geometry.cuh"
#include "common/math_utils.h"
#include "contact/hash.cuh"
#include "contact/contact.cuh"
#include "contact/lbvh.cuh"
#include "contact/collision_detection.cuh"

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

    broad_phase_vf.resize(broad_phase_size * params.nb_all_vertices);
    broad_phase_ee.resize(broad_phase_size * params.nb_all_edges);
    broad_phase_ef.resize(broad_phase_size * params.nb_all_edges);
    tri_bvh = lbvh3d::BVH3D();
    edge_bvh = lbvh3d::BVH3D();
    lbvh3d::initialize(max(params.nb_all_triangles, params.nb_all_edges));
    rebuild_bvh();
}
// static __device__ __forceinline__ float3 tri_normal(const float3& x0, const float3& x1, const float3& x2) {
//     return normalized(cross(x1 - x0, x2 - x0));
// }

void Contact::collision_detect_broad_phase(const float3* pos, const float3* offset) {
    auto& params = geo->params;
    int num_queries = params.nb_all_vertices;
    int threadsPerBlock = 256;
    // query_vf_pairs_simple_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
    //     point_sorted_indices.data().get(),
    //     num_queries,
    //     thrust::raw_pointer_cast(tri_bvh.nodes.data()),
    //     thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
    //     tri_bvh.root_idx,
    //     thrust::raw_pointer_cast(geo->pos_world.data()),
    //     thrust::raw_pointer_cast(geo->triangle_indices.data()),
    //     thrust::raw_pointer_cast(geo->obj_data.data()),
    //     thrust::raw_pointer_cast(geo->vertices_obj.data()),
    //     point_radius,
    //     max_dist,
    //     thrust::raw_pointer_cast(broad_phase_vf.data()),
    //     params.nb_all_cloth_vertices,
    //     broad_phase_size
    //     );
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
        point_radius,
        offset,
        thrust::raw_pointer_cast(broad_phase_vf.data()),
        params.nb_all_cloth_vertices,
        broad_phase_size
        );
    num_queries = params.nb_all_edges;
    // query_ee_pairs_simple_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
    //     thrust::raw_pointer_cast(geo->pos_world.data()),
    //     edge_sorted_indices.data().get(),
    //     num_queries,
    //     thrust::raw_pointer_cast(edge_bvh.nodes.data()),
    //     thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
    //     edge_bvh.root_idx,
    //     thrust::raw_pointer_cast(geo->edges.data()),
    //     thrust::raw_pointer_cast(geo->obj_data.data()),
    //     thrust::raw_pointer_cast(geo->vertices_obj.data()),
    //     point_radius,
    //     max_dist,
    //     params.nb_all_cloth_vertices,
    //     thrust::raw_pointer_cast(broad_phase_ee.data()),
    //     broad_phase_size
    //     );
    query_ee_pairs_capsule_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        pos,
        edge_sorted_indices.data().get(),
        num_queries,
        thrust::raw_pointer_cast(edge_bvh.nodes.data()),
        thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
        edge_bvh.root_idx,
        thrust::raw_pointer_cast(geo->edges.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        point_radius,
        offset,
        thrust::raw_pointer_cast(geo->edge_normals.data()),
        params.nb_all_cloth_vertices,
        thrust::raw_pointer_cast(broad_phase_ee.data()),
        broad_phase_size
        );
    // edges vs faces
    query_ef_pairs_kernel<<<(num_queries + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        thrust::raw_pointer_cast(edge_bvh.aabbs.data()),
        edge_sorted_indices.data().get(),
        num_queries,
        thrust::raw_pointer_cast(tri_bvh.nodes.data()),
        thrust::raw_pointer_cast(tri_bvh.aabbs.data()),
        thrust::raw_pointer_cast(geo->edges.data()),
        thrust::raw_pointer_cast(geo->obj_data.data()),
        thrust::raw_pointer_cast(geo->vertices_obj.data()),
        thrust::raw_pointer_cast(geo->e2t.data()),
        tri_bvh.root_idx,
        point_radius,
        thrust::raw_pointer_cast(broad_phase_ef.data()),
        broad_phase_size
        );
}
void Contact::collision_detect() {
    compute_inertial_offset();
    if ( geo->simulator->frame % 20 == 0 ) {
        rebuild_bvh();
    }
    else {
        refit_bvh();
    }
    // broad phase
    collision_detect_broad_phase(geo->pos_world.data().get(), geo->inertial_offset.data().get());
}
static __global__ void compute_inertial_offset_kernel(
    float3* __restrict__ inertial_offset,
    const float3* __restrict__ pos,
    const char* __restrict__ mask,
    const float3* __restrict__ velocities,
    float3 gravity,
    bool ground,
    float dt,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= n ) return;
    if ( mask[i] ) {
        inertial_offset[i] = make_float3(0, 0, 0);
        return;
    }
    float3 dx = velocities[i] * dt;
    if ( !ground || (pos[i] + dx).z > 0.0f ) {
        dx += gravity * dt * dt;
    }
    inertial_offset[i] = dx;
}
void Contact::compute_inertial_offset() {
    int num_vertices = geo->params.nb_all_vertices;
    int threadsPerBlock = 256;
    compute_inertial_offset_kernel<<<(num_vertices + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(
        geo->inertial_offset.data().get(),
        geo->pos_world.data().get(),
        geo->vertices_mask.data().get(),
        geo->velocities.data().get(),
        geo->gravity, geo->ground,
        geo->simulator->dt, num_vertices);
}
void Contact::rebuild_bvh() {
    lbvh3d::build_face_bvh(geo->pos_world, geo->triangle_indices, tri_bvh, geo->inertial_offset.data().get());
    lbvh3d::build_edge_bvh(geo->pos_world, geo->edges, edge_bvh, geo->inertial_offset.data().get());
    cudaMemcpyAsync(edge_sorted_indices.data().get(), lbvh3d::get_sorted_indices(),
        sizeof(unsigned int) * edge_sorted_indices.size(), cudaMemcpyDeviceToDevice);
    lbvh3d::compute_and_sort_by_morton_codes(geo->pos_world.data().get(),
        geo->pos_world.size(), point_sorted_indices.data().get());
}
void Contact::refit_bvh() {
    lbvh3d::refit_face_bvh(geo->pos_world.data().get(), geo->triangle_indices, tri_bvh, geo->inertial_offset.data().get());
    lbvh3d::refit_edge_bvh(geo->pos_world.data().get(), geo->edges, edge_bvh, geo->inertial_offset.data().get());
}
void Contact::refit_bvh(const float3* pos, const float3* offset) {
    lbvh3d::refit_face_bvh(pos, geo->triangle_indices, tri_bvh, offset);
    lbvh3d::refit_edge_bvh(pos, geo->edges, edge_bvh, offset);
}

static __global__ void compute_vf_force(
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
        int i1 = tri.x;
        int i2 = tri.y;
        int i3 = tri.z;
        float3 x1 = pos[i1];
        float3 x2 = pos[i2];
        float3 x3 = pos[i3];
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
        float stiff = k * vert_stiff;
        if ( static_diags[i1] > 0.f && static_diags[i2] > 0.f && static_diags[i3] > 0.f ) {
            float face_stiff = (static_diags[i1] + static_diags[i2] + static_diags[i3]) * 0.333333f;
            stiff = k * (vert_stiff * face_stiff) / (vert_stiff + face_stiff);
        }
        else if ( vert_stiff <= 0.f ) continue;
        // float sign_dist = (dist > 0.0f) ? 1.0f : -1.0f;
        float3 force;
        float diff = thickness - dist;
        if ( force_type == 0 ) {
            float force_mag = stiff * diff;
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
            float d = max(dist, thickness * 0.1f);
            float d_ratio = d / thickness;
            float log_term = logf(d_ratio);
            log_term = min(log_term, 1e8f);
            diff = thickness - d;

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
    // float* __restrict__ weight,
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
        float3 v0 = pos[e.x];
        float3 v1 = pos[e.y];

        float s, t;
        float3 ab;
        segment_segment_closest_robust(p0, p1, v0, v1, s, t, ab);
        ab = -ab;
        // float s, t;
        // float3 closest_A, closest_B;
        // if ( !edge_edge_closest_points(p0, p1, v0, v1, closest_A, closest_B, s, t) ) {
        //     continue;
        // }
        if ( s <= 0.0f || s >= 1.0f || t <= 0.0f || t >= 1.0f ) {
            continue;
        }
        // float3 ab = closest_B - closest_A;
        float dist = norm(ab);
        float3 normal;
        if ( dist < 1e-16f ) {
            normal = edge_normal0;
            ab = normal;
        }
        else {
            normal = ab / -dist;
        }

        // layer
        int layer1 = obj_data[vertices_obj[e.x]].collision_layer;
        if ( layer1 == layer0 ) {
            // float sign_new = dot(ab, cross(v1 - p0, E)) < 0.0f ? 1.0f : -1.0f;
            float sign_new = dot(ab, edge_normal0) < 0.0f ? 1.0f : -1.0f;
            sign *= sign_new;
            if ( sign < 0.0f ) {
                dist = -dist;
                normal = -normal;
            }
        }
        else {
            bool reverse = false;
            if ( layer0 < layer1 ) {
                reverse = dot(normal, edge_normal0) > 0.0f;
            }
            else {
                float3 edge_normal1 = edge_normals[eid2];
                reverse = dot(normal, edge_normal1) < 0.0f;
            }
            if ( reverse ) {
                normal = -normal;
                dist = -dist;
            }
        }

        float thickness = thickness0 + obj_data[vertices_obj[e.x]].thickness;
        if ( dist > thickness ) continue;

        float stiff = k * e1_stiff;
        if ( static_diags[e.x] > 0.f && static_diags[e.y] > 0.f ) {
            float e2_stiff = (static_diags[e.x] + static_diags[e.y]) * 0.5f;
            stiff = k * (e1_stiff * e2_stiff) / (e1_stiff + e2_stiff);
        }
        else if ( e1_stiff <= 0.f ) continue;

        float3 force;
        float diff = thickness - dist;
        if ( force_type == 0 ) {
            float force_mag = stiff * diff;
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
            float d = max(dist, thickness * 0.05f);
            float d_ratio = d / thickness;
            float log_term = logf(d_ratio);
            log_term = min(log_term, 1e8f);
            diff = thickness - d;

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

        if ( d1 * d2 >= 0.0f ) continue;  // 同侧

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

// void Contact::contact_handle() {
//     int block = 256;
//     int n = (int)point_hash_table_size;
//     clear_hash_table<<<(n + block - 1) / block, block>>>(
//         point_hash_table.data().get(), n);
//     n = params.nb_all_cloth_vertices;
//     // 1. collect pp
//     float cell_size = params.cloth_edge_mean_length * 1.414f;
//     // const float cell_size = 5.f * 0.001f;
//     insert_points_to_grid<<<(n + block - 1) / block, block>>>(
//         vertices_world.data().get(),
//         point_hash_table.data().get(),
//         cell_size, point_hash_table_size,
//         n);
//
//     const float dist = 5.f * 0.001f;
//     pp_result_size.assign(1, 0);
//     collect_pp<<<(n + block - 1) / block, block>>>(
//         pp_collision_result.data().get(),
//         pp_result_size.data().get(),
//         // sort_key_temp.data().get(),
//         // sort_value_temp.data().get(),
//         // sort_result_size.data().get(),
//         vertices_world.data().get(),
//         point_hash_table.data().get(),
//         dist * dist, n,
//         max_pp_result_size,
//         point_hash_table_size,
//         cell_size);
//     // 2. graph coloring
//     int result_size;
//     cudaMemcpy(&result_size, sort_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//     result_size = min(result_size, max_pp_result_size);
// }
//

// void Contact::collision_LCP_postprocess(float3* points_y) {
//     START_TIMER;
//     int block = 256;
//     int n = (int)point_hash_table_size;
//     clear_hash_table<<<(n + block - 1) / block, block>>>(
//         point_hash_table.data().get(), n);
//     int cloth_vertex_size = params.nb_all_cloth_vertices;
//     n = cloth_vertex_size;
//     // 1. collect pp
//     float max_dist = params.cloth_edge_mean_length;
//
//     float3* points_x = vertices_old.data().get();
//     collision_collect_near_pairs(points_x, max_dist, true, false, true, true);
//     // tp_result_size_h = 0;
//     RECORD_TIME("collision_collect_near_pairs");
//
//
//     debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
//     int num_constraints = tp_result_size_h + ee_result_size_h;
//     if ( num_constraints > 0 ) {
//         int all_vertex_size = params.nb_all_vertices;
//         float3* points_collision = temp_vertices_f3.data().get();
//         cudaMemcpyAsync(points_collision, points_y, all_vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
//         // std::cout <<  result_size << " triangles" << std::endl; 
//         // compute_collision_penalty_force_triangle_point_plane<<<(result_size + block - 1) / block, block>>>(
//         //     // Jx.data().get(),
//         //     Jx_diag.data().get(),
//         //     forces.data().get(),
//         //     tp_collision_result.data().get(),
//         //     points_y, triangle_indices.data().get(),
//         //     params.nb_all_cloth_vertices,
//         //     result_size);
//         if ( tp_result_size_h > 0 )
//             collision_tp_to_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
//                 collision_constraints.data().get(),
//                 tp_collision_result.data().get(),
//                 triangle_indices.data().get(),
//                 tp_result_size_h);
//         if ( ee_result_size_h > 0 )
//             collision_ee_to_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
//                 collision_constraints.data().get(),
//                 ee_collision_result.data().get(),
//                 edges.data().get(),
//                 tp_result_size_h,
//                 ee_result_size_h);
//         // coloring
//         // 1. 构建邻接关系并着色 (对应 Vivace 算法)
//         int num_colors = color_constraints(num_constraints);
//         std::cout << "palette_size: " << num_colors << std::endl;
//         RECORD_TIME("color_constraints");
//         n = params.nb_all_vertices;
//         // fill_inv_mass<<<(n + block - 1) / block, block>>>(
//         //     mass_inv.data().get(),
//         //     vertices_obj.data().get(),
//         //     object_types.data().get(),
//         //     masses.data().get(),
//         //     vertices_mask.data().get(), n);
//         // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
//         // std::vector<type> __##v(_##v.begin(), _##v.end())
//         // CHECK(mass_inv, float);
//         // #undef CHECK
//         // solve LCP using PGS
//         collision_tp_to_normal_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             debug_colors.data().get(),
//             tp_collision_result.data().get(),
//             constraint_colors.data().get(),
//             triangle_indices.data().get(),
//             mass_inv.data().get(),
//             tp_result_size_h);
//         collision_ee_to_normal_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             debug_colors.data().get(),
//             ee_collision_result.data().get(),
//             constraint_colors.data().get(),
//             edges.data().get(),
//             mass_inv.data().get(),
//             tp_result_size_h,
//             ee_result_size_h);
//         // 2. 将约束按颜色进行排序/分组
//         // 这样在 GPU 上读取时是连续的，利用内存合并访问（Coalesced Memory Access）
//         thrust::sort_by_key(thrust::device, constraint_colors.begin(),
//             constraint_colors.begin() + num_constraints, normal_constraints.begin());
//
//         CUDA_CHECK(cudaDeviceSynchronize());
//         int* lookup = point_hash_table_lookup.data().get();
//         record_color_offsets<<<(num_constraints + block - 1) / block, block>>>(
//             lookup, normal_constraints.data().get(), num_constraints);
//         CUDA_CHECK(cudaDeviceSynchronize());
//
//         cudaMemcpy(constraint_color_offsets.data(), lookup, (num_colors + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//
//         int* d_needs_more_iters = sort_result_size.data().get();
//         int h_needs_more_iters;
//         // 3. Multi-Color PGS 求解主循环
//         int num_iterations = 100;
//         RECORD_TIME("sort constraints");
//         if ( current_graph_exec != nullptr ) {
//             cudaGraphExecDestroy(current_graph_exec);
//             cudaGraphDestroy(current_graph);
//         }
//         cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
//         cudaMemsetAsync(d_needs_more_iters, 0, sizeof(int), capture_stream);
//         // 把单次 PGS 迭代 (包含所有颜色的 Kernel 发射) 录制下来
//         int c = num_colors == MAX_COLORS ? -1 : 0;
//         for ( ; c < num_colors; ++c ) {
//             int start_idx = constraint_color_offsets[c + 1];
//             int end_idx = constraint_color_offsets[c + 2];
//             if ( start_idx < 0 ) continue;
//             if ( end_idx <= start_idx ) end_idx = num_colors;
//             int num_constraints_in_color = end_idx - start_idx;
//             solvePGS_UnifiedColorBatchKernel<<<(num_constraints_in_color + block - 1) / block, block,0,capture_stream>>>(
//                 normal_constraints.data().get(), d_needs_more_iters,
//                 points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
//                 num_constraints_in_color);
//         }
//
//         cudaStreamEndCapture(capture_stream, &current_graph);
//         cudaGraphInstantiate(&current_graph_exec, current_graph, NULL, NULL, 0);
//         for ( int iter = 0; iter < num_iterations; ++iter ) {
//             // 按颜色逐个批次启动 GPU Kernel
//             // int c = num_colors == MAX_COLORS ? -1 : 0;
//             // for ( ; c < num_colors; ++c ) {
//             //     int start_idx = constraint_color_offsets[c + 1];
//             //     int end_idx = constraint_color_offsets[c + 2];
//             //     if ( start_idx < 0 ) continue;
//             //     if ( end_idx <= start_idx ) end_idx = num_colors;
//             //     int num_constraints_in_color = end_idx - start_idx;
//             //
//             //     // int gridSize = (num_constraints_in_color + blockSize - 1) / blockSize;
//             //
//             //     // 启动 Kernel，仅处理当前颜色的约束
//             //     solvePGS_UnifiedColorBatchKernel
//             //         <<<(num_constraints_in_color + block - 1) / block, block>>>(
//             //             normal_constraints.data().get(), d_needs_more_iters,
//             //             points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
//             //             num_constraints_in_color);
//             //
//             // }
//             cudaGraphLaunch(current_graph_exec, nullptr);
//             if ( iter % 10 == 0 ) {
//                 cudaMemcpy(&h_needs_more_iters, d_needs_more_iters, sizeof(int), cudaMemcpyDeviceToHost);
//                 std::cout << "constraints: " << h_needs_more_iters << std::endl;
//                 if ( h_needs_more_iters == 0 ) {
//                     break;
//                 }
//             }
//         }
//         RECORD_TIME("Multi-Color PGS");
//         update_end_collision<<<(n + block - 1) / block, block>>>(
//             points_y,
//             velocities.data().get(),
//             points_x,
//             points_collision,
//             vertices_mask.data().get(),
//             dt, n);
//     }
// }
// void Contact::collision_LCP_postprocess_unified(float3* points_y) {
//     START_TIMER;
//     int block = 256;
//     int n = (int)point_hash_table_size;
//     clear_hash_table<<<(n + block - 1) / block, block>>>(
//         point_hash_table.data().get(), n);
//     int cloth_vertex_size = params.nb_all_cloth_vertices;
//     n = cloth_vertex_size;
//
//     // float3* points_x = vertices_old.data().get();
//
//     debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
//     int num_constraints = tp_result_size_h + ee_result_size_h;
//     if ( num_constraints == 0 ) return;
//     int all_vertex_size = params.nb_all_vertices;
//     // float3* points_collision = temp_vertices_f3.data().get();
//     // cudaMemcpyAsync(points_collision, points_y, all_vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
//     float3* points_collision = points_y;
//     // coloring
//     // 1. 构建邻接关系并着色 (对应 Vivace 算法)
//     int num_colors = color_constraints(num_constraints);
//     std::cout << "palette_size: " << num_colors << std::endl;
//     RECORD_TIME("color_constraints");
//     n = params.nb_all_vertices;
//     // solve LCP using PGS
//     // 2. 将约束按颜色进行排序/分组
//     // 这样在 GPU 上读取时是连续的，利用内存合并访问（Coalesced Memory Access）
//     thrust::sort_by_key(thrust::device, constraint_colors.begin(),
//         constraint_colors.begin() + num_constraints, normal_constraints.begin());
//
//     CUDA_CHECK(cudaDeviceSynchronize());
//     int* lookup = point_hash_table_lookup.data().get();
//     record_color_offsets<<<(num_constraints + block - 1) / block, block>>>(
//         lookup, normal_constraints.data().get(), num_constraints);
//     CUDA_CHECK(cudaDeviceSynchronize());
//
//     cudaMemcpy(constraint_color_offsets.data(), lookup, (num_colors + 2) * sizeof(int), cudaMemcpyDeviceToHost);
//
//     int* d_needs_more_iters = sort_result_size.data().get();
//     int h_needs_more_iters;
//     // 3. Multi-Color PGS 求解主循环
//     int num_iterations = 10;
//     RECORD_TIME("sort constraints");
//     if ( current_graph_exec != nullptr ) {
//         cudaGraphExecDestroy(current_graph_exec);
//         cudaGraphDestroy(current_graph);
//     }
//     cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
//     cudaMemsetAsync(d_needs_more_iters, 0, sizeof(int), capture_stream);
//     int c = num_colors == MAX_COLORS ? -1 : 0;
//     for ( ; c < num_colors; ++c ) {
//         int start_idx = constraint_color_offsets[c + 1];
//         int end_idx = constraint_color_offsets[c + 2];
//         if ( start_idx < 0 ) continue;
//         if ( end_idx <= start_idx ) end_idx = num_colors;
//         int num_constraints_in_color = end_idx - start_idx;
//         solvePGS_UnifiedColorBatchKernel<<<(num_constraints_in_color + block - 1) / block, block,0,capture_stream>>>(
//             normal_constraints.data().get(), d_needs_more_iters,
//             points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
//             num_constraints_in_color);
//     }
//
//     cudaStreamEndCapture(capture_stream, &current_graph);
//     cudaGraphInstantiate(&current_graph_exec, current_graph, NULL, NULL, 0);
//     for ( int iter = 0; iter < num_iterations; ++iter ) {
//         cudaGraphLaunch(current_graph_exec, nullptr);
//         if ( iter % 10 == 0 ) {
//             cudaMemcpy(&h_needs_more_iters, d_needs_more_iters, sizeof(int), cudaMemcpyDeviceToHost);
//             std::cout << "constraints: " << h_needs_more_iters << std::endl;
//             if ( h_needs_more_iters == 0 ) {
//                 break;
//             }
//         }
//     }
//     RECORD_TIME("Multi-Color PGS");
//     // update_end_collision<<<(n + block - 1) / block, block>>>(
//     //     points_y,
//     //     velocities.data().get(),
//     //     points_x,
//     //     points_collision,
//     //     vertices_mask.data().get(),
//     //     dt, n);
// }
//
// int Geometry::color_constraints(int num_constraints) {
//     int blockSize = 256;
//     int gridSize = (num_constraints + blockSize - 1) / blockSize;
//     int* d_constraint_colors = this->constraint_colors.data().get();
//     // 初始化着色数组为 -1
//     cudaMemsetAsync(d_constraint_colors, -1, num_constraints * sizeof(int));
//
//     int num_vertices = params.nb_all_vertices;
//     // 分配黑板：记录每个顶点每种颜色被谁占用了
//     int* d_vertex_color_claimer = this->vertex_color_claimer.data().get();
//     cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int));
//
//     int* d_uncolored_count = this->uncolored_count.data().get();
//     uint64_t* d_vertex_forbidden_masks = this->vertex_forbidden_masks.data().get();
//
//     int h_uncolored_count = num_constraints;
//     // int last_uncolored_count = h_uncolored_count;
//     int iteration = 0;
//     // int h_current_palette_size = 4;
//
//     // CollisionConstraint* d_constraints = this->collision_constraints.data().get();
//     auto* d_constraints = this->normal_constraints.data().get();
//
//     auto d_current_palette_size = alloc_pool();
//     auto d_iteration = alloc_pool();
//     auto d_last_uncolored_count = alloc_pool();
//     cudaMemsetAsync(d_last_uncolored_count.ptr, num_constraints, sizeof(int));
//     cudaMemsetAsync(d_iteration.ptr, 0, sizeof(int));
//     // 一开始给出少量颜色，增加冲突几率但节约颜色
//     int h_current_palette_size = 4;
//     cudaMemcpyAsync(d_current_palette_size.ptr, &h_current_palette_size, sizeof(int), cudaMemcpyHostToDevice);
//
//     // if ( current_graph_exec != nullptr ) {
//     //     cudaGraphExecDestroy(current_graph_exec);
//     //     cudaGraphDestroy(current_graph);
//     // }
//     cudaGraph_t graph = nullptr;
//     cudaGraphExec_t graph_exec = nullptr;
//     cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
//
//     for ( int i = 0; i < 20; i++ ) {
//         cudaMemsetAsync(d_vertex_forbidden_masks, 0, num_vertices * sizeof(uint64_t), capture_stream);
//         cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int), capture_stream);
//         // cudaMemsetAsync(d_uncolored_count, 0, sizeof(int));
//         // 步骤 1：去抢占颜色
//         k_mark_forbidden_bits<<<gridSize, blockSize,0,capture_stream>>>(
//             d_vertex_forbidden_masks, d_uncolored_count, d_constraints, d_constraint_colors,
//             num_constraints);
//         k_claim_color_bitmask<<<gridSize, blockSize,0,capture_stream>>>(
//             d_vertex_color_claimer, d_constraints, d_constraint_colors, d_vertex_forbidden_masks,
//             d_current_palette_size.ptr, d_iteration.ptr, num_constraints);
//
//         // 步骤 2：验证是否成功
//         k_verify_colors<<<gridSize, blockSize,0,capture_stream>>>(
//             d_constraint_colors, d_uncolored_count, d_constraints,
//             d_vertex_color_claimer, num_constraints);
//         k_update_colors<<<1, 1,0,capture_stream>>>(
//             d_current_palette_size.ptr,
//             d_uncolored_count,
//             d_last_uncolored_count.ptr,
//             d_iteration.ptr);
//     }
//     cudaStreamEndCapture(capture_stream, &graph);
//     cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
//     // d_current_palette_size.ptr
//     while ( iteration < 10 ) {
//         cudaGraphLaunch(graph_exec, nullptr);
//         /*cudaMemsetAsync(d_vertex_forbidden_masks, 0, num_vertices * sizeof(uint64_t));
//         cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int));
//         // cudaMemsetAsync(d_uncolored_count, 0, sizeof(int));
//         // 步骤 1：去抢占颜色
//         k_mark_forbidden_bits<<<gridSize, blockSize>>>(
//             d_vertex_forbidden_masks, d_uncolored_count, d_constraints, d_constraint_colors,
//             num_constraints);
//         k_claim_color_bitmask<<<gridSize, blockSize>>>(
//             d_vertex_color_claimer, d_constraints, d_constraint_colors, d_vertex_forbidden_masks,
//             d_current_palette_size.ptr, d_iteration.ptr, num_constraints);
//
//         // 步骤 2：验证是否成功
//         k_verify_colors<<<gridSize, blockSize>>>(
//             d_constraint_colors, d_uncolored_count, d_constraints,
//             d_vertex_color_claimer, num_constraints);
//         k_update_colors<<<1, 1>>>(
//             d_current_palette_size.ptr,
//             d_uncolored_count,
//             d_last_uncolored_count.ptr,
//             d_iteration.ptr);*/
//         // if ( iteration % 3 == 0 ) {
//         cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
//         if ( h_uncolored_count == 0 ) break;
//         // }
//         iteration++;
//
//         // 如果还有没涂上的，慢慢增加可选颜色的种类
//         // cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
//         // if ( last_uncolored_count == h_uncolored_count && h_uncolored_count > 0 ) {
//         //     if ( current_palette_size < MAX_COLORS )
//         //         current_palette_size++;
//         // }
//         // last_uncolored_count = h_uncolored_count;
//         // std::cout << "uncolored_count: " << h_uncolored_count << std::endl;
//     }
//     // k_print_colors<<<gridSize, blockSize>>>(d_constraint_colors, d_constraints, num_constraints);
//     cudaGraphDestroy(graph);
//     cudaGraphExecDestroy(graph_exec);
//     cudaMemcpy(&h_current_palette_size, d_current_palette_size.ptr,
//         sizeof(int), cudaMemcpyDeviceToHost);
//     return h_current_palette_size;
//
// }
//
// void Contact::collision_collect_near_pairs(float3* points, float max_dist,
//     bool update_hash, bool collect_pp, bool collect_tp, bool collect_ee) {
//     START_TIMER;
//     int block = 256;
//     // int n = (int)point_hash_table_size;
//     // clear_hash_table<<<(n + block - 1) / block, block>>>(
//     //     point_hash_table.data().get(), n);
//     int vertex_size = params.nb_all_cloth_vertices;
//     int n = vertex_size;
//     float cell_size = max_dist * 2.f;
//     auto constraint_size_p = alloc_pool();
//     if ( collect_tp || collect_ee || collect_pp )
//         cudaMemsetAsync(constraint_size_p.ptr, 0, sizeof(int));
//     // float max_dist_edge = params.cloth_edge_mean_length * 2.f;
//     if ( update_hash ) {
//         auto point_hashes_size_p = alloc_pool();
//         auto edge_hashes_size_p = alloc_pool();
//         cudaMemsetAsync(point_hashes_size_p.ptr, 0, sizeof(int));
//         record_point_hash<false, false><<<(n + block - 1) / block, block>>>(
//             point_hashes.data().get(),
//             sort_key_temp.data().get(),
//             point_hashes_size_p.ptr,
//             nullptr,
//             points,
//             vertex_proxy.data().get(),
//             cell_size,
//             max_dist,
//             max_point_hashes_size,
//             point_hash_table_size,
//             n);
//
//         CUDA_CHECK(cudaMemcpy(&point_hashes_size_h, point_hashes_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost));
//         point_hashes_size_h = min(point_hashes_size_h, max_point_hashes_size);
//         RECORD_TIME("record_point_hash");
//         thrust::sort_by_key(thrust::device, sort_key_temp.begin(), sort_key_temp.begin() + point_hashes_size_h,
//             point_hashes.begin());
//
//         RECORD_TIME("sort_by_key");
//         // build hash_table lookup
//         cudaMemsetAsync(point_hash_table_lookup.data().get(), -1, sizeof(int) * (point_hash_table_size + 1));
//         record_hash_table_lookup<<<(point_hashes_size_h + block - 1) / block, block>>>(
//             point_hash_table_lookup.data().get(),
//             point_hashes.data().get(),
//             point_hash_table_size, point_hashes_size_h);
//         // edges
//         if ( collect_ee ) {
//             n = params.nb_all_cloth_edges;
//             CUDA_CHECK(cudaMemsetAsync(edge_hashes_size_p.ptr, 0, sizeof(int)));
//             record_edge_hashes<<<(n + block - 1) / block, block>>>(
//                 edge_hashes.data().get(),
//                 sort_key_temp.data().get(),
//                 edge_hashes_size_p.ptr,
//                 edges.data().get(),
//                 points,
//                 cell_size,
//                 max_edge_hashes_size,
//                 edge_hash_table_size,
//                 n);
//             CUDA_CHECK(cudaMemcpy(&edge_hashes_size_h, edge_hashes_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost));
//             edge_hashes_size_h = min(edge_hashes_size_h, max_edge_hashes_size);
//             RECORD_TIME("record_edge_hash");
//             thrust::sort_by_key(thrust::device,
//                 sort_key_temp.begin(), sort_key_temp.begin() + edge_hashes_size_h, edge_hashes.begin());
//             RECORD_TIME("sort_by_key");
//             // build hash_table lookup
//             cudaMemsetAsync(edge_hash_table_lookup.data().get(), -1, sizeof(int) * (edge_hash_table_size + 1));
//             record_hash_table_lookup<<<(edge_hashes_size_h + block - 1) / block, block>>>(
//                 edge_hash_table_lookup.data().get(),
//                 edge_hashes.data().get(),
//                 edge_hash_table_size, edge_hashes_size_h);
//         }
//     }
//     tp_result_size_h = pp_result_size_h = ee_result_size_h = 0;
//     if ( collect_pp ) {
//         cudaMemsetAsync(pp_result_size.data().get(), 0, sizeof(int));
//         n = params.nb_all_cloth_vertices;
//         // debug_colors.assign(vertex_size, make_float3(0.5f, 0.5f, 0.5f));
//         // collect_pp_sorted<<<(n + block - 1) / block, block>>>(
//         //     pp_collision_result.data().get(),
//         //     pp_result_size.data().get(),
//         //     debug_colors.data().get(),
//         //     point_hashes.data().get(),
//         //     points,
//         //     edge_lookup.data().get(),
//         //     dir_edges.data().get(),
//         //     point_hash_table_lookup.data().get(),
//         //     vertex_proxy.data().get(),
//         //     vertices_mask.data().get(),
//         //     cell_size,
//         //     max_dist,
//         //     point_hash_table_size,
//         //     point_hashes_size_h,
//         //     max_pp_result_size,
//         //     n);
//         points_query_points_by_point_hash<<<(n + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             pp_result_size.data().get(),
//             nullptr,
//             point_hashes.data().get(),
//             points,
//             edge_lookup.data().get(),
//             dir_edges.data().get(),
//             point_hash_table_lookup.data().get(),
//             vertex_proxy.data().get(),
//             vertices_mask.data().get(),
//             cell_size,
//             max_dist,
//             point_hash_table_size,
//             point_hashes_size_h,
//             max_collision_constraints_size,
//             n);
//         cudaMemcpy(&pp_result_size_h, pp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//         // pp_result_size_h = min(pp_result_size_h, max_pp_result_size);
//         pp_result_size_h = min(pp_result_size_h, max_collision_constraints_size);
//         std::cout << "pp_result_size_h = " << pp_result_size_h << std::endl;
//     }
//     if ( collect_tp ) {
//         n = params.nb_all_triangles;
//         // cudaMemsetAsync(tp_result_size.data().get(), 0, sizeof(int));
//         // triangles_query_points<<<(n + block - 1) / block, block>>>(
//         //     tp_collision_result.data().get(),
//         //     tp_result_size.data().get(),
//         //     triangle_indices.data().get(),
//         //     points,
//         //     vertices_old.data().get(),
//         //     point_hashes.data().get(),
//         //     point_hash_table_lookup.data().get(),
//         //     vertices_mask.data().get(), cell_size,
//         //     max_dist * max_dist, point_hash_table_size,
//         //     point_hashes_size_h, max_tp_result_size,
//         //     params.nb_all_cloth_vertices, n);
//         // cudaMemcpy(&tp_result_size_h, tp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//         // tp_result_size_h = min(tp_result_size_h, max_tp_result_size);
//         debug_colors.assign(params.nb_all_cloth_edges, make_float3(0.5f, 0.5f, 0.5f));
//         triangles_query_points_by_point_hash<<<(n + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             nullptr,
//             constraint_size_p.ptr,
//             triangle_indices.data().get(),
//             points,
//             // vertices_old.data().get(),
//             point_hashes.data().get(),
//             point_hash_table_lookup.data().get(),
//             vertices_mask.data().get(), cell_size,
//             max_dist * max_dist, point_hash_table_size,
//             point_hashes_size_h, max_collision_constraints_size,
//             params.nb_all_cloth_vertices, n);
//         cudaMemcpy(&tp_result_size_h, constraint_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost);
//         tp_result_size_h = min(tp_result_size_h, max_collision_constraints_size);
//         RECORD_TIME("triangles_query_points");
//         std::cout << "tp_result_size_h = " << tp_result_size_h << std::endl;
//     }
//     if ( collect_ee ) {
//         n = params.nb_all_edges;
//         // max_dist *= 0.5f;
//         // cudaMemsetAsync(ee_result_size.data().get(), 0, sizeof(int));
//         // edges_query_edges_via_point_hash<<<(n + block - 1) / block, block>>>(
//         //     ee_collision_result.data().get(),
//         //     ee_result_size.data().get(),
//         //     edges.data().get(),
//         //     points,
//         //     // vertices_old.data().get(),
//         //     point_hashes.data().get(),
//         //     point_hash_table_lookup.data().get(),
//         //     dir_edges.data().get(),
//         //     edge_lookup.data().get(),
//         //     vertices_mask.data().get(), cell_size,
//         //     max_dist * max_dist, max_dist * 0.5, point_hash_table_size,
//         //     point_hashes_size_h, max_ee_result_size,
//         //     params.nb_all_cloth_vertices, n);
//         // cudaMemcpy(&ee_result_size_h, ee_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
//         // ee_result_size_h = min(ee_result_size_h, max_ee_result_size);
//         // RECORD_TIME("edges_query_edges_via_point_hash");
//         block = 64;
//         detect_edge_edge_constraints<<<(n + block - 1) / block, block>>>(
//             normal_constraints.data().get(),
//             constraint_size_p.ptr,
//             edges.data().get(),
//             points,
//             edge_hashes.data().get(),
//             edge_hash_table_lookup.data().get(),
//             vertices_mask.data().get(),
//             cell_size, max_dist,
//             max_collision_constraints_size,
//             edge_hash_table_size,
//             edge_hashes_size_h,
//             params.nb_all_cloth_vertices,
//             n);
//         cudaMemcpy(&ee_result_size_h, constraint_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost);
//         RECORD_TIME("detect_edge_edge_constraints");
//         ee_result_size_h = min(ee_result_size_h, max_collision_constraints_size);
//         ee_result_size_h -= tp_result_size_h;
//         std::cout << "ee_result_size_h = " << ee_result_size_h << std::endl;
//     }
// }
