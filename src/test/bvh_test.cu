#include <cuda_runtime.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cub/cub.cuh>
#include <vector>
#include <random>
#include <string>
#include <iostream>
#include "benchmarks.h"
#include "simulation/contact/collision.cuh"


#include "simulation/contact/lbvh.cuh"
// 子树展开：从 node_idx 下探 num_expansion_steps 层
// 每个 thread 根据 lane_id 的 bit 走不同路径，返回到达的节点索引
// -1 表示该线程无需处理（空闲或重复叶子被去重）
__device__ inline int bvh_expand_node(
    const int2* __restrict__ nodes,
    int node_idx,
    int lane_id,
    int num_expansion_steps
) {
    if ( num_expansion_steps == 0 ) return node_idx;

    int max_lanes = 1 << num_expansion_steps;
    if ( lane_id >= max_lanes ) return -1;

    // 起始节点本身就是叶子
    int2 node = nodes[node_idx];
    if ( node.y == 0 ) {
        return (lane_id == 0) ? node_idx : -1;
    }

    for ( int i = 0; i < num_expansion_steps; ++i ) {
        int bit_pos = num_expansion_steps - 1 - i;
        int select = (lane_id >> bit_pos) & 1;

        // 读取当前内部节点的子节点
        node = nodes[node_idx];
        int child_idx = (select == 0) ? (node.x - 1) : (node.y - 1);

        // 检查子节点是否为叶子
        int2 child = nodes[child_idx];
        if ( child.y == 0 ) {
            // 只有"规范线程"（剩余低位全零）返回此叶子，避免重复
            int mask = (1 << bit_pos) - 1;
            return ((lane_id & mask) == 0) ? child_idx : -1;
        }
        node_idx = child_idx;
    }
    return node_idx;
}
#define COOP_STACK_CAPACITY 2048

static __global__ void query_vf_pairs_kernel_cooperative(
    const float3* __restrict__ query_pts, unsigned int num_queries,
    const int2* __restrict__ nodes, const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const float3* __restrict__ vertices, const int3* __restrict__ faces,
    const float radius, const float max_dist,
    int* __restrict__ query_results, int result_size
) {
    const int lane_id = threadIdx.x;
    const int block_size = blockDim.x;

    // 展开深度 = ceil(log2(block_size))
    // 128 → 7, 64 → 6, 32 → 5
    int num_expansion_steps = 0;
    {
        int p = 1;
        while ( p < block_size ) {
            p <<= 1;
            ++num_expansion_steps;
        }
    }

    // ---------- 共享内存 ----------
    __shared__ int stack[COOP_STACK_CAPACITY];
    __shared__ int stack_count;      // 栈中元素数
    __shared__ int result_count;     // 已找到的结果数
    __shared__ int top_node;         // 展开模式下弹出的节点

    // 每个线程块顺序处理多个查询
    for ( int query_idx = blockIdx.x;
          query_idx < num_queries;
          query_idx += gridDim.x ) {
        // ---- 初始化 ----
        if ( lane_id == 0 ) {
            stack[0] = root_idx;
            stack_count = 1;
            result_count = 0;
        }
        __syncthreads();

        float3 qp = query_pts[query_idx];
        AABB q_aabb;
        q_aabb.min = qp - radius;
        q_aabb.max = qp + radius;
        float max_dist_sq = max_dist * max_dist;
        int max_results = result_size - 1;

        // ---- 协作遍历 ----
        while ( stack_count > 0 && result_count < max_results ) {
            __syncthreads();

            int node_idx = -1;
            int current_count = stack_count;

            if ( current_count >= block_size ) {
                // ======== 直接模式 ========
                // 栈中节点充足，每线程取一个
                if ( lane_id == 0 )
                    stack_count = current_count - block_size;
                __syncthreads();
                node_idx = stack[stack_count + lane_id];
            }
            else {
                // ======== 展开模式 ========
                // 栈中节点不足，弹出一个并展开
                if ( lane_id == 0 ) {
                    stack_count = current_count - 1;
                    top_node = stack[current_count - 1];
                }
                __syncthreads();

                node_idx = bvh_expand_node(nodes, top_node,
                    lane_id, num_expansion_steps);

                // 展开未分配到节点的线程，尝试从剩余栈中取
                if ( node_idx < 0 && current_count > 1 ) {
                    int id = atomicAdd(&stack_count, -1) - 1;
                    if ( id >= 0 )
                        node_idx = stack[id];
                    else
                        atomicAdd(&stack_count, 1);   // 回退下溢
                }
            }

            __syncthreads();

            // ---- 每个线程处理分配到的节点 ----
            if ( node_idx >= 0 && aabb_overlap_3d(q_aabb, aabbs[node_idx]) ) {
                int2 node = nodes[node_idx];

                if ( node.y == 0 ) {
                    // 叶子：做精确距离测试
                    int prim_idx = node.x - 1;
                    int3 f = faces[prim_idx];
                    if ( f.x != query_idx &&
                        f.y != query_idx &&
                        f.z != query_idx ) {
                        float3 v0 = vertices[f.x];
                        float3 v1 = vertices[f.y];
                        float3 v2 = vertices[f.z];
                        float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);
                        if ( dist_sq < max_dist_sq ) {
                            int pos = atomicAdd(&result_count, 1);
                            if ( pos < max_results )
                                query_results[query_idx * result_size + pos + 1]
                                    = prim_idx;
                        }
                    }
                }
                else {
                    // 内部节点：子节点入栈
                    int pos = atomicAdd(&stack_count, 2);
                    if ( pos + 1 < COOP_STACK_CAPACITY ) {
                        stack[pos] = node.x - 1;
                        stack[pos + 1] = node.y - 1;
                    }
                }
            }
            __syncthreads();
        }

        // ---- 写回结果数量 ----
        if ( lane_id == 0 )
            query_results[query_idx * result_size] =
                min(result_count, max_results);
        __syncthreads();
    }
}
static __global__ void
query_vf_pairs_kernel2(
    const float3* query_pts, const unsigned int* sorted_indices, unsigned int num_queries,
    const int2* __restrict__ nodes, const AABB* __restrict__ aabbs,
    unsigned int root_idx,
    const float3* vertices, const int3* __restrict__ faces,
    const float radius, const float max_dist,
    int* __restrict__ query_results, int result_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_queries ) return;
    if ( sorted_indices )
        i = sorted_indices[i];
    float3 qp = query_pts[i];

    // float max_dist_sq = max_dist * max_dist;
    //
    // AABB q_aabb = {
    //     .min = qp - radius,
    //     .max = qp + radius,
    // };
    // BVH_QUERY_LOOP(q_aabb, 32,
    //     int3 f = faces[prim_idx];
    //     if ( f.x == i || f.y == i || f.z == i ) continue; // itself
    //     float3 v0 = vertices[f.x];
    //     float3 v1 = vertices[f.y];
    //     float3 v2 = vertices[f.z];
    //
    //     float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);
    //
    //     if ( dist_sq < max_dist_sq ) {
    //     query_result[++query_count] = prim_idx;
    //     }
    //     );
    float3 q_min = qp - radius;
    float3 q_max = qp + radius;
    float max_dist_sq = max_dist * max_dist;

    int* query_result = &query_results[i * result_size];
    int query_count = 0;

    unsigned int stack[32];
    int sp = 0;
    stack[sp++] = root_idx;
    while ( sp > 0 && query_count < result_size - 1 ) {
        unsigned int node_idx = stack[--sp];

        // 优化：展开 AABB overlap 测试，尽早退出
        const AABB node_aabb = aabbs[node_idx];
        if ( q_min.x > node_aabb.max.x || q_max.x < node_aabb.min.x ||
            q_min.y > node_aabb.max.y || q_max.y < node_aabb.min.y ||
            q_min.z > node_aabb.max.z || q_max.z < node_aabb.min.z ) {
            continue;
        }

        int2 node = nodes[node_idx];
        if ( node.y == 0 ) {
            int prim_idx = node.x - 1;
            int3 f = faces[prim_idx];

            // 本身检测
            if ( f.x == i || f.y == i || f.z == i ) continue;

            float3 v0 = vertices[f.x];
            float3 v1 = vertices[f.y];
            float3 v2 = vertices[f.z];

            float dist_sq = dist_sq_point_triangle_3d(qp, v0, v1, v2);

            if ( dist_sq < max_dist_sq ) {
                query_result[++query_count] = prim_idx;
            }
        }
        else if ( sp < 30 ) {
            stack[sp++] = node.x - 1;
            stack[sp++] = node.y - 1;
        }
    }
    query_result[0] = query_count;
}

std::vector<int> bvh_benchmark(const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    const std::vector<float>& query_points, int warmup, int runs, bool verify) {
    auto& timer = globalTimer();
    thrust::device_vector<float3> d_points(query_points.size() / 3);
    cudaMemcpy(d_points.data().get(), query_points.data(), d_points.size() * sizeof(float3), cudaMemcpyHostToDevice);
    thrust::device_vector<float3> d_vertices(vertices_in.size() / 3);
    cudaMemcpy(d_vertices.data().get(), vertices_in.data(), d_vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    thrust::device_vector<int3> d_faces(faces_in.size() / 3);
    cudaMemcpy(d_faces.data().get(), faces_in.data(), d_faces.size() * sizeof(int3), cudaMemcpyHostToDevice);

    int num_triangles = (int)d_faces.size();
    lbvh3d::initialize(num_triangles);

    lbvh3d::BVH3D bvh;
    for ( int i = 0; i < warmup; i++ ) {
        bvh = lbvh3d::BVH3D();
        lbvh3d::build_face_bvh(d_vertices, d_faces, bvh);
        cudaDeviceSynchronize();
    }
    for ( int i = 0; i < runs; i++ ) {
        timer.start("build_bvh");
        bvh = lbvh3d::BVH3D();
        lbvh3d::build_face_bvh(d_vertices, d_faces, bvh);
        cudaDeviceSynchronize();
        timer.stop();
    }
    for ( int i = 0; i < warmup; i++ ) {
        lbvh3d::refit_face_bvh(d_vertices.data().get(), d_faces, bvh);
        cudaDeviceSynchronize();
    }
    for ( int i = 0; i < runs; i++ ) {
        timer.start("refit_bvh");
        lbvh3d::refit_face_bvh(d_vertices.data().get(), d_faces, bvh);
        cudaDeviceSynchronize();
        timer.stop();
    }

    int num_queries = query_points.size() / 3;
    printf("Number of queries = %d\n", num_queries);
    thrust::device_vector<int> nearest_obstacle_faces(num_queries);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
    // for(int i = 0; i < warmup; i++) {
    //     cudaMemset(nearest_obstacle_faces.data().get(), -1, num_queries * sizeof(int));
    //     lbvh3d::query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //         thrust::raw_pointer_cast(d_points.data()),
    //         num_queries,
    //         thrust::raw_pointer_cast(bvh.nodes.data()),
    //         thrust::raw_pointer_cast(bvh.aabbs.data()),
    //         bvh.root_idx,
    //         thrust::raw_pointer_cast(d_vertices.data()),
    //         thrust::raw_pointer_cast(d_faces.data()),
    //         thrust::raw_pointer_cast(nearest_obstacle_faces.data())
    //         );
    //     cudaDeviceSynchronize();
    // }
    // for ( int i = 0; i < runs; i++ ) {
    //     cudaMemset(nearest_obstacle_faces.data().get(), -1, num_queries * sizeof(int));
    //     timer.start("bvh_query_nearest");
    //     lbvh3d::query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //         thrust::raw_pointer_cast(d_points.data()),
    //         num_queries,
    //         thrust::raw_pointer_cast(bvh.nodes.data()),
    //         thrust::raw_pointer_cast(bvh.aabbs.data()),
    //         bvh.root_idx,
    //         thrust::raw_pointer_cast(d_vertices.data()),
    //         thrust::raw_pointer_cast(d_faces.data()),
    //         thrust::raw_pointer_cast(nearest_obstacle_faces.data())
    //         );
    //     cudaDeviceSynchronize();
    //     timer.stop();
    // }

    const float radius = 0.1f;
    const float max_dist = 0.2f;
    const int max_pairs_per_query = 8;

    thrust::device_vector<int> d_query_results(num_queries * max_pairs_per_query);

    // for ( int i = 0; i < warmup; i++ ) {
    //     cudaMemset(d_query_results.data().get(), 0, d_query_results.size() * sizeof(int));
    //     query_vf_pairs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //         thrust::raw_pointer_cast(d_points.data()),
    //         num_queries,
    //         thrust::raw_pointer_cast(bvh.nodes.data()),
    //         thrust::raw_pointer_cast(bvh.aabbs.data()),
    //         bvh.root_idx,
    //         thrust::raw_pointer_cast(d_vertices.data()),
    //         thrust::raw_pointer_cast(d_faces.data()),
    //         radius,
    //         max_dist,
    //         thrust::raw_pointer_cast(d_query_results.data()),
    //         max_pairs_per_query
    //         );
    //
    //     cudaDeviceSynchronize();
    // }
    // thrust::device_vector<float> scene_bounds(6);
    // thrust::device_vector<unsigned int> morton_codes(num_queries);
    thrust::device_vector<unsigned int> sorted_indices(num_queries * 2);

    lbvh3d::compute_and_sort_by_morton_codes(thrust::raw_pointer_cast(d_points.data()),
        num_queries, sorted_indices.data().get(), false);

    for ( int i = 0; i < runs; i++ ) {
        cudaMemset(d_query_results.data().get(), 0, d_query_results.size() * sizeof(int));
        timer.start("bvh_query_pairs");

        query_vf_pairs_kernel2<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            thrust::raw_pointer_cast(sorted_indices.data()),
            num_queries,
            thrust::raw_pointer_cast(bvh.nodes.data()),
            thrust::raw_pointer_cast(bvh.aabbs.data()),
            bvh.root_idx,
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_faces.data()),
            radius,
            max_dist,
            thrust::raw_pointer_cast(d_query_results.data()),
            max_pairs_per_query
            );
        cudaDeviceSynchronize();
        timer.stop();
    }
    auto result = std::vector<int>();
    if ( verify ) {
        result.resize(num_queries * max_pairs_per_query);
        cudaMemcpy(result.data(), d_query_results.data().get(), result.size() * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return result;
}
std::vector<int> bvh_edge_benchmark(const std::vector<float>& vertices_in,
    const std::vector<int>& edge_in,
    const std::vector<float>& query_points, int warmup, int runs, bool verify) {
    auto& timer = globalTimer();
    thrust::device_vector<float3> d_points(query_points.size() / 3);
    cudaMemcpy(d_points.data().get(), query_points.data(), d_points.size() * sizeof(float3), cudaMemcpyHostToDevice);
    thrust::device_vector<float3> d_vertices(vertices_in.size() / 3);
    cudaMemcpy(d_vertices.data().get(), vertices_in.data(), d_vertices.size() * sizeof(float3), cudaMemcpyHostToDevice);
    thrust::device_vector<int2> d_edges(edge_in.size() / 2);
    cudaMemcpy(d_edges.data().get(), edge_in.data(), d_edges.size() * sizeof(int2), cudaMemcpyHostToDevice);

    int num_edges = (int)d_edges.size();
    lbvh3d::initialize(num_edges);

    lbvh3d::BVH3D bvh;
    for ( int i = 0; i < warmup; i++ ) {
        bvh = lbvh3d::BVH3D();
        lbvh3d::build_edge_bvh(d_vertices, d_edges, bvh);
    }
    for ( int i = 0; i < runs; i++ ) {
        timer.start("build_bvh");
        bvh = lbvh3d::BVH3D();
        lbvh3d::build_edge_bvh(d_vertices, d_edges, bvh);
        timer.stop();
    }
    for ( int i = 0; i < warmup; i++ ) {
        lbvh3d::refit_edge_bvh(d_vertices.data().get(), d_edges, bvh);
        cudaDeviceSynchronize();
    }
    for ( int i = 0; i < runs; i++ ) {
        timer.start("refit_bvh");
        lbvh3d::refit_edge_bvh(d_vertices.data().get(), d_edges, bvh);
        cudaDeviceSynchronize();
        timer.stop();
    }

    int num_queries = edge_in.size() / 2;
    printf("Number of queries = %d\n", num_queries);
    thrust::device_vector<int> nearest_obstacle_faces(num_queries);
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_queries + threadsPerBlock - 1) / threadsPerBlock;
    // for(int i = 0; i < warmup; i++) {
    //     cudaMemset(nearest_obstacle_faces.data().get(), -1, num_queries * sizeof(int));
    //     lbvh3d::query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //         thrust::raw_pointer_cast(d_points.data()),
    //         num_queries,
    //         thrust::raw_pointer_cast(bvh.nodes.data()),
    //         thrust::raw_pointer_cast(bvh.aabbs.data()),
    //         bvh.root_idx,
    //         thrust::raw_pointer_cast(d_vertices.data()),
    //         thrust::raw_pointer_cast(d_faces.data()),
    //         thrust::raw_pointer_cast(nearest_obstacle_faces.data())
    //         );
    //     cudaDeviceSynchronize();
    // }
    // for ( int i = 0; i < runs; i++ ) {
    //     cudaMemset(nearest_obstacle_faces.data().get(), -1, num_queries * sizeof(int));
    //     timer.start("bvh_query_nearest");
    //     lbvh3d::query_nearest_face_kernel<<<blocksPerGrid, threadsPerBlock>>>(
    //         thrust::raw_pointer_cast(d_points.data()),
    //         num_queries,
    //         thrust::raw_pointer_cast(bvh.nodes.data()),
    //         thrust::raw_pointer_cast(bvh.aabbs.data()),
    //         bvh.root_idx,
    //         thrust::raw_pointer_cast(d_vertices.data()),
    //         thrust::raw_pointer_cast(d_faces.data()),
    //         thrust::raw_pointer_cast(nearest_obstacle_faces.data())
    //         );
    //     cudaDeviceSynchronize();
    //     timer.stop();
    // }

    const float radius = 0.1f;
    const float max_dist = 0.2f;
    const int max_pairs_per_query = 8;

    thrust::device_vector<int> d_query_results(num_queries * max_pairs_per_query);

    // thrust::device_vector<unsigned int> sorted_indices(num_queries);
    // cudaMemcpyAsync(sorted_indices.data().get(), lbvh3d::get_sorted_indices(),
    // sizeof(unsigned int) * sorted_indices.size(), cudaMemcpyDeviceToDevice);

    // lbvh3d::compute_and_sort_by_morton_codes(thrust::raw_pointer_cast(d_points.data()),
    //     num_queries, sorted_indices.data().get());

    for ( int i = 0; i < warmup; i++ ) {
        cudaMemset(d_query_results.data().get(), 0, d_query_results.size() * sizeof(int));
        query_ee_pairs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            num_queries,
            thrust::raw_pointer_cast(bvh.nodes.data()),
            thrust::raw_pointer_cast(bvh.aabbs.data()),
            bvh.root_idx,
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_edges.data()),
            radius,
            max_dist,
            thrust::raw_pointer_cast(d_query_results.data()),
            max_pairs_per_query
            );


        cudaDeviceSynchronize();
    }

    for ( int i = 0; i < runs; i++ ) {
        cudaMemset(d_query_results.data().get(), 0, d_query_results.size() * sizeof(int));
        timer.start("bvh_query_pairs");

        query_ee_pairs_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            thrust::raw_pointer_cast(d_points.data()),
            num_queries,
            thrust::raw_pointer_cast(bvh.nodes.data()),
            thrust::raw_pointer_cast(bvh.aabbs.data()),
            bvh.root_idx,
            thrust::raw_pointer_cast(d_vertices.data()),
            thrust::raw_pointer_cast(d_edges.data()),
            radius,
            max_dist,
            thrust::raw_pointer_cast(d_query_results.data()),
            max_pairs_per_query
            );
        cudaDeviceSynchronize();
        timer.stop();
    }
    auto result = std::vector<int>();
    if ( verify ) {
        result.resize(num_queries * max_pairs_per_query);
        cudaMemcpy(result.data(), d_query_results.data().get(), result.size() * sizeof(int), cudaMemcpyDeviceToHost);
    }
    return result;
}
