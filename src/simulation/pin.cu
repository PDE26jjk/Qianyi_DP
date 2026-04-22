#include "solver_base.cuh"

#include <thrust/execution_policy.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

#include "constraint.cuh"
#include "geometric_operator.cuh"
#include "contact/collision.cuh"


#include "contact/lbvh.cuh"

static __global__ void transform_to_world(
    float3* __restrict__ vertices_world,
    const float3* __restrict__ vertices_local,
    const int* __restrict__ vertices_obj,
    const Mat4* __restrict__ world_matrices,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
          i += blockDim.x * gridDim.x ) {
        int obj = vertices_obj[i];
        vertices_world[i] = mul_homo(world_matrices[obj], vertices_local[i]);
    }
}
static __global__ void compute_attach_info_kernel(
    AttachInfo* __restrict__ attach_data,
    const int* __restrict__ attached_indices,
    const int* __restrict__ nearest_global_faces,
    const float3* __restrict__ vertices_world,
    const int3* __restrict__ faces,
    const int num_attached
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_attached ) return;
    int vert_idx = attached_indices[i];
    int face_idx = nearest_global_faces[i];
    float3 qp = vertices_world[vert_idx];
    int3 f = faces[face_idx];
    float3 v0 = vertices_world[f.x];
    float3 v1 = vertices_world[f.y];
    float3 v2 = vertices_world[f.z];
    AttachInfo info;
    info.face_idx = face_idx;

    float3 e1 = v1 - v0;
    float3 e2 = v2 - v0;
    float3 N = cross(e1, e2);
    float len_sq = dot(N, N);
    if ( len_sq > 1e-20f ) {
        float inv_len = rsqrtf(len_sq);
        float3 N_hat = N * inv_len;
        float3 P_rel = qp - v0;

        info.d = dot(P_rel, N_hat);
        float3 P_proj = P_rel - info.d * N_hat;

        float d00 = dot(e1, e1);
        float d01 = dot(e1, e2);
        float d11 = dot(e2, e2);
        float d20 = dot(P_proj, e1);
        float d21 = dot(P_proj, e2);
        float denom = d00 * d11 - d01 * d01;
        float inv_denom = 1.0f / denom;
        info.u = (d11 * d20 - d01 * d21) * inv_denom;
        info.v = (d00 * d21 - d01 * d20) * inv_denom;
    }
    else {
        info.u = 0.0f;
        info.v = 0.0f;
        info.d = 0.0f;
        // printf("len_sq: %f\n",len_sq);
        printf("ERROR: tri:%d,  [(%f,%f,%f),(%f,%f,%f),(%f,%f,%f)]\n",
            face_idx, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);
    }
    printf("vert_idx:%d,tri_id:%d,u:%f,v:%f,d:%f\n",vert_idx, face_idx,info.u,info.v,info.d);
    attach_data[vert_idx] = info;
}
void SolverBase::init_pin() {
    int threadsPerBlock = 256;
    int blocksPerGrid = (params.nb_all_vertices + threadsPerBlock - 1) / threadsPerBlock;
    transform_to_world<<<blocksPerGrid, threadsPerBlock>>>(
        thrust::raw_pointer_cast(vertices_world.data()),
        thrust::raw_pointer_cast(vertices_local.data()),
        thrust::raw_pointer_cast(vertices_obj.data()),
        thrust::raw_pointer_cast(world_matrices.data()),
        params.nb_all_vertices
        );
    thrust::transform(thrust::device, thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(params.nb_all_cloth_vertices),
        vertices_mask.begin(), [
            pin_attached = thrust::raw_pointer_cast(pin_attached.data()),
            pin_fixed = thrust::raw_pointer_cast(pin_fixed.data())]
        __device__ (int i) {
            if ( pin_attached[i] > 0.01f ) {
                return char(MaskBit::attach_mask);
            }
            if ( pin_fixed[i] > 0.01f ) {
                return char(MaskBit::fix_mask);
            }
            return char(0);
        });
    thrust::device_vector<int> attached_indices(params.nb_all_cloth_vertices);
    auto attach_end = thrust::copy_if(
        thrust::device,
        thrust::make_counting_iterator(0),
        thrust::make_counting_iterator(params.nb_all_cloth_vertices),
        vertices_mask.begin(),
        attached_indices.begin(),
        [] __device__ (char mask) {
            return mask & static_cast<char>(MaskBit::attach_mask);
        }
        );
    int num_attached = attach_end - attached_indices.begin();
    attached_indices.resize(num_attached);
    attach_data.resize(params.nb_all_cloth_vertices);
    if ( num_attached > 0 && params.nb_all_triangles > params.nb_all_cloth_triangles ) {
        lbvh3d::BVH3D obstacle_bvh;
        thrust::device_vector<int3> obstacle_faces(triangle_indices.begin() + params.nb_all_cloth_triangles,
            triangle_indices.begin() + params.nb_all_triangles);
        int num_obstacle_triangles = obstacle_faces.size();
        lbvh3d::initialize(num_obstacle_triangles);
        lbvh3d::build_face_bvh(vertices_world, obstacle_faces, obstacle_bvh);
        thrust::device_vector<float3> attached_verts_world(num_attached);
        thrust::gather(thrust::device, attached_indices.begin(), attached_indices.end(), vertices_world.begin(),
            attached_verts_world.begin());

        thrust::device_vector<int> nearest_obstacle_faces(num_attached);
        int threadsPerBlock = 256;
        int blocksPerGrid = (num_attached + threadsPerBlock - 1) / threadsPerBlock;

        query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(attached_verts_world.data()),
            num_attached,
            thrust::raw_pointer_cast(obstacle_bvh.nodes.data()),
            thrust::raw_pointer_cast(obstacle_bvh.aabbs.data()),
            obstacle_bvh.root_idx,
            thrust::raw_pointer_cast(vertices_world.data()),
            thrust::raw_pointer_cast(obstacle_faces.data()),
            thrust::raw_pointer_cast(nearest_obstacle_faces.data())
            );
        thrust::transform(thrust::device, nearest_obstacle_faces.begin(), nearest_obstacle_faces.end(),
            nearest_obstacle_faces.begin(),
            [offset = params.nb_all_cloth_triangles] __device__ (unsigned int local_idx) {
                return local_idx + offset;
            });
        compute_attach_info_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(attach_data.data()),
            thrust::raw_pointer_cast(attached_indices.data()),
            thrust::raw_pointer_cast(nearest_obstacle_faces.data()),
            thrust::raw_pointer_cast(vertices_world.data()),
            thrust::raw_pointer_cast(triangle_indices.data()),
            num_attached
            );
    }
    thrust::device_vector<float>().swap(pin_fixed);// clear
    thrust::device_vector<float>().swap(pin_attached);
}

static __global__ void update_attached_vertices_kernel(
    float3* vertices_world_out,
    const AttachInfo* __restrict__ attach_data,
    const char* __restrict__ vertices_mask,
    const int3* __restrict__ faces,
    const float3* vertices_world,
    const int n
) {
    for ( int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x ) {
        if ( vertices_mask[i] & static_cast<char>(MaskBit::attach_mask) ) {
            AttachInfo info = attach_data[i];

            // 1. 取出当前帧该三角形的三个顶点
            int3 f = faces[info.face_idx];
            float3 v0 = vertices_world[f.x];
            float3 v1 = vertices_world[f.y];
            float3 v2 = vertices_world[f.z];

            // 2. 计算当前帧的局部坐标系轴
            float3 e1 = v1 - v0;
            float3 e2 = v2 - v0;
            float3 N = cross(e1, e2);
            float len_sq = dot(N, N);

            if ( len_sq > 1e-20f ) {
                float inv_len = rsqrtf(len_sq);
                float3 N_hat = N * inv_len; // 当前帧的单位法线

                // 3. 逆向重构世界坐标: P = v0 + c1*e1 + c2*e2 + c3*N_hat
                // 注意：如果碰撞体发生了缩放或剪切变形，e1和e2的长度/方向会自动反映在偏移中
                float3 new_pos = v0 + info.u * e1 + info.v * e2 + info.d * N_hat;

                vertices_world_out[i] = new_pos;
            }else {
                printf("Error in update_attached_vertices_kernel\n");
            }
        }
    }
}
void SolverBase::update_pin(float3* q) {
    int block = 256;
    int n = params.nb_all_cloth_vertices;
    update_attached_vertices_kernel<<<(n + block - 1) / block,block>>>(
        q, attach_data.data().get(), vertices_mask.data().get(),
        triangle_indices.data().get(), vertices_world.data().get(),
        n);
}
