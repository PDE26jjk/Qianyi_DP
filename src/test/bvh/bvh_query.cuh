// Builds face BVH (LBVH) and provides nearest-face query.
#pragma once

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include "bvh.h"             // lbvh3d::BVH, bvh_build, bvh_destroy
#include "common/vec_math.h" // float3, int3, operators
#include "common/cuda_utils.h"

namespace lbvh3dtest {

// ------------------------------------------------------------
// Compute per-triangle AABBs
// ------------------------------------------------------------
__global__ void compute_tri_aabbs(
    const float3* vertices,
    const int3*   faces,
    int           num_tris,
    float3*       d_lowers,
    float3*       d_uppers)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_tris) return;

    int3 f = faces[idx];
    float3 v0 = vertices[f.x];
    float3 v1 = vertices[f.y];
    float3 v2 = vertices[f.z];

    float3 lo = fmin3(v0, fmin3(v1, v2));
    float3 up = fmax3(v0, fmax3(v1, v2));

    d_lowers[idx] = lo;
    d_uppers[idx] = up;
}

// ------------------------------------------------------------
// Closest point on triangle (GPU)
// ------------------------------------------------------------
__device__ inline float3 closest_point_on_triangle(
    const float3& p,
    const float3& a, const float3& b, const float3& c)
{
    float3 ab = b - a;
    float3 ac = c - a;
    float3 ap = p - a;

    float d1 = dot(ab, ap);
    float d2 = dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    float3 bp = p - b;
    float d3 = dot(ab, bp);
    float d4 = dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return a + v * ab;
    }

    float3 cp = p - c;
    float d5 = dot(ab, cp);
    float d6 = dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return a + w * ac;
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return a + ab * v + ac * w;
}

// ------------------------------------------------------------
// Nearest face query kernel (uses raw BVH nodes)
// ------------------------------------------------------------
// 快速点到 AABB 距离平方
__device__ inline float dist_sq_point_aabb(
    const float3& p,
    const float3& bb_min,
    const float3& bb_max)
{
    float3 t = fmin3(fmax3(p, bb_min), bb_max);
    float3 d = p - t;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

// 快速点到三角形距离平方（调用已有的 closest_point_on_triangle）
__device__ inline float dist_sq_point_triangle(
    const float3& p,
    const float3& a, const float3& b, const float3& c)
{
    float3 cp = closest_point_on_triangle(p, a, b, c);
    float3 d = p - cp;
    return d.x * d.x + d.y * d.y + d.z * d.z;
}

__global__ void query_nearest_face_kernel(
    const float3* __restrict__ points,
    int                        num_points,
    const BVHPackedNodeHalf* __restrict__ node_lowers,
    const BVHPackedNodeHalf* __restrict__ node_uppers,
    const int*    __restrict__ primitive_indices,
    int                        root_idx,
    const float3* __restrict__ vertices,
    const int3*   __restrict__ faces,
    int*          __restrict__ nearest_faces)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= num_points) return;

    float3 p = points[tid];
    float best_dist2 = FLT_MAX;
    int best_face = -1;

    constexpr int STACK_SIZE = 64;
    int stack[STACK_SIZE];
    int sp = 0;
    stack[sp++] = root_idx;

    while (sp > 0) {
        int node_idx = stack[--sp];

        // 使用 __ldg 缓存只读数据（可选）
        BVHPackedNodeHalf lower = node_lowers[node_idx];
        BVHPackedNodeHalf upper = node_uppers[node_idx];
        
        float3 bb_min = make_float3(lower.x, lower.y, lower.z);
        float3 bb_max = make_float3(upper.x, upper.y, upper.z);
        // BVHPackedNodeHalf lower_f4 = (const BVHPackedNodeHalf&)__ldg((const float4*)(node_lowers + node_idx));
        // float4 upper_f4 = __ldg((const float4*)(node_uppers + node_idx));

        // 直接取出 AABB 的 min 和 max
        // float3 bb_min = make_float3(lower_f4.x, lower_f4.y, lower_f4.z);
        // float3 bb_max = make_float3(upper_f4.x, upper_f4.y, upper_f4.z);

        if (dist_sq_point_aabb(p, bb_min, bb_max) >= best_dist2)
            continue;

        int left_val  = lower.i;
        int right_val = upper.i;
        // int left_val  = __float_as_int(lower_f4.w);
        // int right_val = __float_as_int(upper_f4.w);
        // bool is_leaf  = (left_val & 1);

        if (lower.b) { // 叶子节点
            int start = left_val;
            int end   = right_val;   // exclusive
            for (int i = start; i < end; ++i) {
                int face_idx = primitive_indices[i];
                int3 f = faces[face_idx];
                float3 v0 = vertices[f.x];
                float3 v1 = vertices[f.y];
                float3 v2 = vertices[f.z];

                float d2 = dist_sq_point_triangle(p, v0, v1, v2);
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_face = face_idx;
                    if (best_dist2 < 1e-8f) {   // 早停阈值
                        nearest_faces[tid] = best_face;
                        return;
                    }
                }
            }
        } else { // 内部节点：按子节点距离排序压栈（近的后压，先遍历）
            int left_child  = left_val;
            int right_child = right_val;

            // 加载子节点包围盒
            BVHPackedNodeHalf l_low = node_lowers[left_child];
            BVHPackedNodeHalf l_up  = node_uppers[left_child];
            float3 l_min = make_float3(l_low.x, l_low.y, l_low.z);
            float3 l_max = make_float3(l_up.x, l_up.y, l_up.z);
            float dist_l = dist_sq_point_aabb(p, l_min, l_max);

            BVHPackedNodeHalf r_low = node_lowers[right_child];
            BVHPackedNodeHalf r_up  = node_uppers[right_child];
            float3 r_min = make_float3(r_low.x, r_low.y, r_low.z);
            float3 r_max = make_float3(r_up.x, r_up.y, r_up.z);
            float dist_r = dist_sq_point_aabb(p, r_min, r_max);

            // 较远的先入栈，较近的后入栈（下次循环先处理近的）
            if (dist_l > dist_r) {
                if (dist_l < best_dist2) stack[sp++] = left_child;
                if (dist_r < best_dist2) stack[sp++] = right_child;
            } else {
                if (dist_r < best_dist2) stack[sp++] = right_child;
                if (dist_l < best_dist2) stack[sp++] = left_child;
            }
        }
    }
    nearest_faces[tid] = best_face;
}


// ------------------------------------------------------------
// Build face BVH (returns compact BVH object, no flattening)
// ------------------------------------------------------------
inline void build_face_bvh_test(
    const thrust::device_vector<float3>& d_vertices,
    const thrust::device_vector<int3>&   d_faces,
    BVH&                                 bvh_out)   // use raw lbvh3d::BVH
{
    int num_faces = (int)d_faces.size();
    if (num_faces == 0) {
        bvh_out = BVH();
        return;
    }

    // 1. Compute per-triangle AABBs
    thrust::device_vector<float3> d_lowers(num_faces);
    thrust::device_vector<float3> d_uppers(num_faces);
    {
        int block = 256;
        int grid  = (num_faces + block - 1) / block;
        compute_tri_aabbs<<<grid, block>>>(
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_faces.data()),
            num_faces,
            thrust::raw_pointer_cast(d_lowers.data()),
            thrust::raw_pointer_cast(d_uppers.data()));
    }
    cudaDeviceSynchronize();

    // 2. Build internal BVH (LBVH)
    const int leaf_size = 1;
    bvh_build(0,
              thrust::raw_pointer_cast(d_lowers.data()),
              thrust::raw_pointer_cast(d_uppers.data()),
              num_faces,
              leaf_size,
              bvh_out);
}

// Optional helper: copy root index to host
inline int get_root_index(const BVH& bvh) {
    int root;
    cudaMemcpy(&root, bvh.root, sizeof(int), cudaMemcpyDeviceToHost);
    return root;
}

} // namespace lbvh3d