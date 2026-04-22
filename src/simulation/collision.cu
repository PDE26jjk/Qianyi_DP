#include "contact/collision.cuh"

#include "solver_base.cuh"
#include "common/cuda_utils.h"
#include <cuda_runtime.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include "solver_PCG.cuh"
#include "common/math_utils.h"
#include "contact/hash.cuh"


void SolverBase::contact_handle() {
    int block = 256;
    int n = (int)point_hash_table_size;
    clear_hash_table<<<(n + block - 1) / block, block>>>(
        point_hash_table.data().get(), n);
    n = params.nb_all_cloth_vertices;
    // 1. collect pp
    float cell_size = params.cloth_edge_mean_length * 1.414f;
    // const float cell_size = 5.f * 0.001f;
    insert_points_to_grid<<<(n + block - 1) / block, block>>>(
        vertices_world.data().get(),
        point_hash_table.data().get(),
        cell_size, point_hash_table_size,
        n);

    const float dist = 5.f * 0.001f;
    pp_result_size.assign(1, 0);
    collect_pp<<<(n + block - 1) / block, block>>>(
        pp_collision_result.data().get(),
        pp_result_size.data().get(),
        // sort_key_temp.data().get(),
        // sort_value_temp.data().get(),
        // sort_result_size.data().get(),
        vertices_world.data().get(),
        point_hash_table.data().get(),
        dist * dist, n,
        max_pp_result_size,
        point_hash_table_size,
        cell_size);
    // 2. graph coloring
    int result_size;
    cudaMemcpy(&result_size, sort_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
    result_size = min(result_size, max_pp_result_size);
    // if ( result_size != 0 ) {
    //     thrust::stable_sort_by_key(thrust::device, sort_key_temp.begin(), sort_key_temp.begin() + result_size,
    //         sort_value_temp.begin());
    // }

}


void SolverBase::collision_LCP_postprocess(float3* points_y) {
    START_TIMER;
    int block = 256;
    int n = (int)point_hash_table_size;
    clear_hash_table<<<(n + block - 1) / block, block>>>(
        point_hash_table.data().get(), n);
    int cloth_vertex_size = params.nb_all_cloth_vertices;
    n = cloth_vertex_size;
    // 1. collect pp
    float max_dist = params.cloth_edge_mean_length;

    float3* points_x = vertices_old.data().get();
    collision_collect_near_pairs(points_x, max_dist, true, false, true, true);
    // tp_result_size_h = 0;
    RECORD_TIME("collision_collect_near_pairs");


    debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
    int num_constraints = tp_result_size_h + ee_result_size_h;
    if ( num_constraints > 0 ) {
        int all_vertex_size = params.nb_all_vertices;
        float3* points_collision = temp_vertices_f3.data().get();
        cudaMemcpyAsync(points_collision, points_y, all_vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
        // std::cout <<  result_size << " triangles" << std::endl; 
        // compute_collision_penalty_force_triangle_point_plane<<<(result_size + block - 1) / block, block>>>(
        //     // Jx.data().get(),
        //     Jx_diag.data().get(),
        //     forces.data().get(),
        //     tp_collision_result.data().get(),
        //     points_y, triangle_indices.data().get(),
        //     params.nb_all_cloth_vertices,
        //     result_size);
        if ( tp_result_size_h > 0 )
            collision_tp_to_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
                collision_constraints.data().get(),
                tp_collision_result.data().get(),
                triangle_indices.data().get(),
                tp_result_size_h);
        if ( ee_result_size_h > 0 )
            collision_ee_to_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
                collision_constraints.data().get(),
                ee_collision_result.data().get(),
                edges.data().get(),
                tp_result_size_h,
                ee_result_size_h);
        // coloring
        // 1. 构建邻接关系并着色 (对应 Vivace 算法)
        int num_colors = color_constraints(num_constraints);
        std::cout << "palette_size: " << num_colors << std::endl;
        RECORD_TIME("color_constraints");
        n = params.nb_all_vertices;
        // fill_inv_mass<<<(n + block - 1) / block, block>>>(
        //     mass_inv.data().get(),
        //     vertices_obj.data().get(),
        //     object_types.data().get(),
        //     masses.data().get(),
        //     vertices_mask.data().get(), n);
        // #define CHECK(v,type) thrust::host_vector<type> _##v = v;\
        // std::vector<type> __##v(_##v.begin(), _##v.end())
        // CHECK(mass_inv, float);
        // #undef CHECK
        // solve LCP using PGS
        collision_tp_to_normal_constraints<<<(tp_result_size_h + block - 1) / block, block>>>(
            normal_constraints.data().get(),
            debug_colors.data().get(),
            tp_collision_result.data().get(),
            constraint_colors.data().get(),
            triangle_indices.data().get(),
            mass_inv.data().get(),
            tp_result_size_h);
        collision_ee_to_normal_constraints<<<(ee_result_size_h + block - 1) / block, block>>>(
            normal_constraints.data().get(),
            debug_colors.data().get(),
            ee_collision_result.data().get(),
            constraint_colors.data().get(),
            edges.data().get(),
            mass_inv.data().get(),
            tp_result_size_h,
            ee_result_size_h);
        // 2. 将约束按颜色进行排序/分组
        // 这样在 GPU 上读取时是连续的，利用内存合并访问（Coalesced Memory Access）
        thrust::sort_by_key(thrust::device, constraint_colors.begin(),
            constraint_colors.begin() + num_constraints, normal_constraints.begin());

        CUDA_CHECK(cudaDeviceSynchronize());
        int* lookup = point_hash_table_lookup.data().get();
        record_color_offsets<<<(num_constraints + block - 1) / block, block>>>(
            lookup, normal_constraints.data().get(), num_constraints);
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaMemcpy(constraint_color_offsets.data(), lookup, (num_colors + 2) * sizeof(int), cudaMemcpyDeviceToHost);

        int* d_needs_more_iters = sort_result_size.data().get();
        int h_needs_more_iters;
        // 3. Multi-Color PGS 求解主循环
        int num_iterations = 100;
        RECORD_TIME("sort constraints");
        if ( current_graph_exec != nullptr ) {
            cudaGraphExecDestroy(current_graph_exec);
            cudaGraphDestroy(current_graph);
        }
        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
        cudaMemsetAsync(d_needs_more_iters, 0, sizeof(int), capture_stream);
        // 把单次 PGS 迭代 (包含所有颜色的 Kernel 发射) 录制下来
        int c = num_colors == MAX_COLORS ? -1 : 0;
        for ( ; c < num_colors; ++c ) {
            int start_idx = constraint_color_offsets[c + 1];
            int end_idx = constraint_color_offsets[c + 2];
            if ( start_idx < 0 ) continue;
            if ( end_idx <= start_idx ) end_idx = num_colors;
            int num_constraints_in_color = end_idx - start_idx;
            solvePGS_UnifiedColorBatchKernel<<<(num_constraints_in_color + block - 1) / block, block,0,capture_stream>>>(
                normal_constraints.data().get(), d_needs_more_iters,
                points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
                num_constraints_in_color);
        }

        cudaStreamEndCapture(capture_stream, &current_graph);
        cudaGraphInstantiate(&current_graph_exec, current_graph, NULL, NULL, 0);
        for ( int iter = 0; iter < num_iterations; ++iter ) {
            // 按颜色逐个批次启动 GPU Kernel
            // int c = num_colors == MAX_COLORS ? -1 : 0;
            // for ( ; c < num_colors; ++c ) {
            //     int start_idx = constraint_color_offsets[c + 1];
            //     int end_idx = constraint_color_offsets[c + 2];
            //     if ( start_idx < 0 ) continue;
            //     if ( end_idx <= start_idx ) end_idx = num_colors;
            //     int num_constraints_in_color = end_idx - start_idx;
            //
            //     // int gridSize = (num_constraints_in_color + blockSize - 1) / blockSize;
            //
            //     // 启动 Kernel，仅处理当前颜色的约束
            //     solvePGS_UnifiedColorBatchKernel
            //         <<<(num_constraints_in_color + block - 1) / block, block>>>(
            //             normal_constraints.data().get(), d_needs_more_iters,
            //             points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
            //             num_constraints_in_color);
            //
            // }
            cudaGraphLaunch(current_graph_exec, nullptr);
            if ( iter % 10 == 0 ) {
                cudaMemcpy(&h_needs_more_iters, d_needs_more_iters, sizeof(int), cudaMemcpyDeviceToHost);
                std::cout << "constraints: " << h_needs_more_iters << std::endl;
                if ( h_needs_more_iters == 0 ) {
                    break;
                }
            }
        }
        RECORD_TIME("Multi-Color PGS");
        update_end_collision<<<(n + block - 1) / block, block>>>(
            points_y,
            velocities.data().get(),
            points_x,
            points_collision,
            vertices_mask.data().get(),
            dt, n);
    }
}
void SolverBase::collision_LCP_postprocess_unified(float3* points_y) {
    START_TIMER;
    int block = 256;
    int n = (int)point_hash_table_size;
    clear_hash_table<<<(n + block - 1) / block, block>>>(
        point_hash_table.data().get(), n);
    int cloth_vertex_size = params.nb_all_cloth_vertices;
    n = cloth_vertex_size;

    // float3* points_x = vertices_old.data().get();

    debug_colors.assign(cloth_vertex_size, make_float3(0.5f, 0.5f, 0.5f));
    int num_constraints = tp_result_size_h + ee_result_size_h;
    if ( num_constraints == 0 ) return;
    int all_vertex_size = params.nb_all_vertices;
    // float3* points_collision = temp_vertices_f3.data().get();
    // cudaMemcpyAsync(points_collision, points_y, all_vertex_size * sizeof(float3), cudaMemcpyDeviceToDevice);
    float3* points_collision = points_y;
    // coloring
    // 1. 构建邻接关系并着色 (对应 Vivace 算法)
    int num_colors = color_constraints(num_constraints);
    std::cout << "palette_size: " << num_colors << std::endl;
    RECORD_TIME("color_constraints");
    n = params.nb_all_vertices;
    // solve LCP using PGS
    // 2. 将约束按颜色进行排序/分组
    // 这样在 GPU 上读取时是连续的，利用内存合并访问（Coalesced Memory Access）
    thrust::sort_by_key(thrust::device, constraint_colors.begin(),
        constraint_colors.begin() + num_constraints, normal_constraints.begin());

    CUDA_CHECK(cudaDeviceSynchronize());
    int* lookup = point_hash_table_lookup.data().get();
    record_color_offsets<<<(num_constraints + block - 1) / block, block>>>(
        lookup, normal_constraints.data().get(), num_constraints);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaMemcpy(constraint_color_offsets.data(), lookup, (num_colors + 2) * sizeof(int), cudaMemcpyDeviceToHost);

    int* d_needs_more_iters = sort_result_size.data().get();
    int h_needs_more_iters;
    // 3. Multi-Color PGS 求解主循环
    int num_iterations = 10;
    RECORD_TIME("sort constraints");
    if ( current_graph_exec != nullptr ) {
        cudaGraphExecDestroy(current_graph_exec);
        cudaGraphDestroy(current_graph);
    }
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
    cudaMemsetAsync(d_needs_more_iters, 0, sizeof(int), capture_stream);
    int c = num_colors == MAX_COLORS ? -1 : 0;
    for ( ; c < num_colors; ++c ) {
        int start_idx = constraint_color_offsets[c + 1];
        int end_idx = constraint_color_offsets[c + 2];
        if ( start_idx < 0 ) continue;
        if ( end_idx <= start_idx ) end_idx = num_colors;
        int num_constraints_in_color = end_idx - start_idx;
        solvePGS_UnifiedColorBatchKernel<<<(num_constraints_in_color + block - 1) / block, block,0,capture_stream>>>(
            normal_constraints.data().get(), d_needs_more_iters,
            points_collision, constraint_colors.data().get(), mass_inv.data().get(), start_idx,
            num_constraints_in_color);
    }

    cudaStreamEndCapture(capture_stream, &current_graph);
    cudaGraphInstantiate(&current_graph_exec, current_graph, NULL, NULL, 0);
    for ( int iter = 0; iter < num_iterations; ++iter ) {
        cudaGraphLaunch(current_graph_exec, nullptr);
        if ( iter % 10 == 0 ) {
            cudaMemcpy(&h_needs_more_iters, d_needs_more_iters, sizeof(int), cudaMemcpyDeviceToHost);
            std::cout << "constraints: " << h_needs_more_iters << std::endl;
            if ( h_needs_more_iters == 0 ) {
                break;
            }
        }
    }
    RECORD_TIME("Multi-Color PGS");
    // update_end_collision<<<(n + block - 1) / block, block>>>(
    //     points_y,
    //     velocities.data().get(),
    //     points_x,
    //     points_collision,
    //     vertices_mask.data().get(),
    //     dt, n);
}
void SolverPCG::contact_handle() {
    collision_LCP_postprocess(vertices_world.data().get());
    // CUDA_CHECK(cudaDeviceSynchronize());
}

int SolverBase::color_constraints(int num_constraints) {
    int blockSize = 256;
    int gridSize = (num_constraints + blockSize - 1) / blockSize;
    int* d_constraint_colors = this->constraint_colors.data().get();
    // 初始化着色数组为 -1
    cudaMemsetAsync(d_constraint_colors, -1, num_constraints * sizeof(int));

    int num_vertices = params.nb_all_vertices;
    // 分配黑板：记录每个顶点每种颜色被谁占用了
    int* d_vertex_color_claimer = this->vertex_color_claimer.data().get();
    cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int));

    int* d_uncolored_count = this->uncolored_count.data().get();
    uint64_t* d_vertex_forbidden_masks = this->vertex_forbidden_masks.data().get();

    int h_uncolored_count = num_constraints;
    // int last_uncolored_count = h_uncolored_count;
    int iteration = 0;
    // int h_current_palette_size = 4;

    // CollisionConstraint* d_constraints = this->collision_constraints.data().get();
    auto* d_constraints = this->normal_constraints.data().get();

    auto d_current_palette_size = alloc_pool();
    auto d_iteration = alloc_pool();
    auto d_last_uncolored_count = alloc_pool();
    cudaMemsetAsync(d_last_uncolored_count.ptr, num_constraints, sizeof(int));
    cudaMemsetAsync(d_iteration.ptr, 0, sizeof(int));
    // 一开始给出少量颜色，增加冲突几率但节约颜色
    int h_current_palette_size = 4;
    cudaMemcpyAsync(d_current_palette_size.ptr, &h_current_palette_size, sizeof(int), cudaMemcpyHostToDevice);

    // if ( current_graph_exec != nullptr ) {
    //     cudaGraphExecDestroy(current_graph_exec);
    //     cudaGraphDestroy(current_graph);
    // }
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);

    for ( int i = 0; i < 20; i++ ) {
        cudaMemsetAsync(d_vertex_forbidden_masks, 0, num_vertices * sizeof(uint64_t), capture_stream);
        cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int), capture_stream);
        // cudaMemsetAsync(d_uncolored_count, 0, sizeof(int));
        // 步骤 1：去抢占颜色
        k_mark_forbidden_bits<<<gridSize, blockSize,0,capture_stream>>>(
            d_vertex_forbidden_masks, d_uncolored_count, d_constraints, d_constraint_colors,
            num_constraints);
        k_claim_color_bitmask<<<gridSize, blockSize,0,capture_stream>>>(
            d_vertex_color_claimer, d_constraints, d_constraint_colors, d_vertex_forbidden_masks,
            d_current_palette_size.ptr, d_iteration.ptr, num_constraints);

        // 步骤 2：验证是否成功
        k_verify_colors<<<gridSize, blockSize,0,capture_stream>>>(
            d_constraint_colors, d_uncolored_count, d_constraints,
            d_vertex_color_claimer, num_constraints);
        k_update_colors<<<1, 1,0,capture_stream>>>(
            d_current_palette_size.ptr,
            d_uncolored_count,
            d_last_uncolored_count.ptr,
            d_iteration.ptr);
    }
    cudaStreamEndCapture(capture_stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0);
    // d_current_palette_size.ptr
    while ( iteration < 10 ) {
        cudaGraphLaunch(graph_exec, nullptr);
        /*cudaMemsetAsync(d_vertex_forbidden_masks, 0, num_vertices * sizeof(uint64_t));
        cudaMemsetAsync(d_vertex_color_claimer, -1, num_vertices * MAX_COLORS * sizeof(int));
        // cudaMemsetAsync(d_uncolored_count, 0, sizeof(int));
        // 步骤 1：去抢占颜色
        k_mark_forbidden_bits<<<gridSize, blockSize>>>(
            d_vertex_forbidden_masks, d_uncolored_count, d_constraints, d_constraint_colors,
            num_constraints);
        k_claim_color_bitmask<<<gridSize, blockSize>>>(
            d_vertex_color_claimer, d_constraints, d_constraint_colors, d_vertex_forbidden_masks,
            d_current_palette_size.ptr, d_iteration.ptr, num_constraints);

        // 步骤 2：验证是否成功
        k_verify_colors<<<gridSize, blockSize>>>(
            d_constraint_colors, d_uncolored_count, d_constraints,
            d_vertex_color_claimer, num_constraints);
        k_update_colors<<<1, 1>>>(
            d_current_palette_size.ptr,
            d_uncolored_count,
            d_last_uncolored_count.ptr,
            d_iteration.ptr);*/
        // if ( iteration % 3 == 0 ) {
        cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
        if ( h_uncolored_count == 0 ) break;
        // }
        iteration++;

        // 如果还有没涂上的，慢慢增加可选颜色的种类
        // cudaMemcpy(&h_uncolored_count, d_uncolored_count, sizeof(int), cudaMemcpyDeviceToHost);
        // if ( last_uncolored_count == h_uncolored_count && h_uncolored_count > 0 ) {
        //     if ( current_palette_size < MAX_COLORS )
        //         current_palette_size++;
        // }
        // last_uncolored_count = h_uncolored_count;
        // std::cout << "uncolored_count: " << h_uncolored_count << std::endl;
    }
    // k_print_colors<<<gridSize, blockSize>>>(d_constraint_colors, d_constraints, num_constraints);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
    cudaMemcpy(&h_current_palette_size, d_current_palette_size.ptr,
        sizeof(int), cudaMemcpyDeviceToHost);
    return h_current_palette_size;

}

void SolverBase::collision_collect_near_pairs(float3* points, float max_dist,
    bool update_hash, bool collect_pp, bool collect_tp, bool collect_ee) {
    START_TIMER;
    int block = 256;
    // int n = (int)point_hash_table_size;
    // clear_hash_table<<<(n + block - 1) / block, block>>>(
    //     point_hash_table.data().get(), n);
    int vertex_size = params.nb_all_cloth_vertices;
    int n = vertex_size;
    float cell_size = max_dist * 2.f;
    auto constraint_size_p = alloc_pool();
    if ( collect_tp || collect_ee || collect_pp )
        cudaMemsetAsync(constraint_size_p.ptr, 0, sizeof(int));
    // float max_dist_edge = params.cloth_edge_mean_length * 2.f;
    if ( update_hash ) {
        auto point_hashes_size_p = alloc_pool();
        auto edge_hashes_size_p = alloc_pool();
        cudaMemsetAsync(point_hashes_size_p.ptr, 0, sizeof(int));
        record_point_hash<false, false><<<(n + block - 1) / block, block>>>(
            point_hashes.data().get(),
            sort_key_temp.data().get(),
            point_hashes_size_p.ptr,
            nullptr,
            points,
            vertex_proxy.data().get(),
            cell_size,
            max_dist,
            max_point_hashes_size,
            point_hash_table_size,
            n);

        CUDA_CHECK(cudaMemcpy(&point_hashes_size_h, point_hashes_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost));
        point_hashes_size_h = min(point_hashes_size_h, max_point_hashes_size);
        RECORD_TIME("record_point_hash");
        thrust::sort_by_key(thrust::device, sort_key_temp.begin(), sort_key_temp.begin() + point_hashes_size_h,
            point_hashes.begin());

        RECORD_TIME("sort_by_key");
        // build hash_table lookup
        cudaMemsetAsync(point_hash_table_lookup.data().get(), -1, sizeof(int) * (point_hash_table_size + 1));
        record_hash_table_lookup<<<(point_hashes_size_h + block - 1) / block, block>>>(
            point_hash_table_lookup.data().get(),
            point_hashes.data().get(),
            point_hash_table_size, point_hashes_size_h);
        // edges
        if ( collect_ee ) {
            n = params.nb_all_cloth_edges;
            CUDA_CHECK(cudaMemsetAsync(edge_hashes_size_p.ptr, 0, sizeof(int)));
            record_edge_hashes<<<(n + block - 1) / block, block>>>(
                edge_hashes.data().get(),
                sort_key_temp.data().get(),
                edge_hashes_size_p.ptr,
                edges.data().get(),
                points,
                cell_size,
                max_edge_hashes_size,
                edge_hash_table_size,
                n);
            CUDA_CHECK(cudaMemcpy(&edge_hashes_size_h, edge_hashes_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost));
            edge_hashes_size_h = min(edge_hashes_size_h, max_edge_hashes_size);
            RECORD_TIME("record_edge_hash");
            thrust::sort_by_key(thrust::device,
                sort_key_temp.begin(), sort_key_temp.begin() + edge_hashes_size_h, edge_hashes.begin());
            RECORD_TIME("sort_by_key");
            // build hash_table lookup
            cudaMemsetAsync(edge_hash_table_lookup.data().get(), -1, sizeof(int) * (edge_hash_table_size + 1));
            record_hash_table_lookup<<<(edge_hashes_size_h + block - 1) / block, block>>>(
                edge_hash_table_lookup.data().get(),
                edge_hashes.data().get(),
                edge_hash_table_size, edge_hashes_size_h);
        }
    }
    tp_result_size_h = pp_result_size_h = ee_result_size_h = 0;
    if ( collect_pp ) {
        cudaMemsetAsync(pp_result_size.data().get(), 0, sizeof(int));
        n = params.nb_all_cloth_vertices;
        // debug_colors.assign(vertex_size, make_float3(0.5f, 0.5f, 0.5f));
        // collect_pp_sorted<<<(n + block - 1) / block, block>>>(
        //     pp_collision_result.data().get(),
        //     pp_result_size.data().get(),
        //     debug_colors.data().get(),
        //     point_hashes.data().get(),
        //     points,
        //     edge_lookup.data().get(),
        //     dir_edges.data().get(),
        //     point_hash_table_lookup.data().get(),
        //     vertex_proxy.data().get(),
        //     vertices_mask.data().get(),
        //     cell_size,
        //     max_dist,
        //     point_hash_table_size,
        //     point_hashes_size_h,
        //     max_pp_result_size,
        //     n);
        points_query_points_by_point_hash<<<(n + block - 1) / block, block>>>(
            normal_constraints.data().get(),
            pp_result_size.data().get(),
            nullptr,
            point_hashes.data().get(),
            points,
            edge_lookup.data().get(),
            dir_edges.data().get(),
            point_hash_table_lookup.data().get(),
            vertex_proxy.data().get(),
            vertices_mask.data().get(),
            cell_size,
            max_dist,
            point_hash_table_size,
            point_hashes_size_h,
            max_collision_constraints_size,
            n);
        cudaMemcpy(&pp_result_size_h, pp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
        // pp_result_size_h = min(pp_result_size_h, max_pp_result_size);
        pp_result_size_h = min(pp_result_size_h, max_collision_constraints_size);
        std::cout << "pp_result_size_h = " << pp_result_size_h << std::endl;
    }
    if ( collect_tp ) {
        n = params.nb_all_triangles;
        // cudaMemsetAsync(tp_result_size.data().get(), 0, sizeof(int));
        // triangles_query_points<<<(n + block - 1) / block, block>>>(
        //     tp_collision_result.data().get(),
        //     tp_result_size.data().get(),
        //     triangle_indices.data().get(),
        //     points,
        //     vertices_old.data().get(),
        //     point_hashes.data().get(),
        //     point_hash_table_lookup.data().get(),
        //     vertices_mask.data().get(), cell_size,
        //     max_dist * max_dist, point_hash_table_size,
        //     point_hashes_size_h, max_tp_result_size,
        //     params.nb_all_cloth_vertices, n);
        // cudaMemcpy(&tp_result_size_h, tp_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
        // tp_result_size_h = min(tp_result_size_h, max_tp_result_size);
        debug_colors.assign(params.nb_all_cloth_edges, make_float3(0.5f, 0.5f, 0.5f));
        triangles_query_points_by_point_hash<<<(n + block - 1) / block, block>>>(
            normal_constraints.data().get(),
            nullptr,
            constraint_size_p.ptr,
            triangle_indices.data().get(),
            points,
            // vertices_old.data().get(),
            point_hashes.data().get(),
            point_hash_table_lookup.data().get(),
            vertices_mask.data().get(), cell_size,
            max_dist * max_dist, point_hash_table_size,
            point_hashes_size_h, max_collision_constraints_size,
            params.nb_all_cloth_vertices, n);
        cudaMemcpy(&tp_result_size_h, constraint_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost);
        tp_result_size_h = min(tp_result_size_h, max_collision_constraints_size);
        RECORD_TIME("triangles_query_points");
        std::cout << "tp_result_size_h = " << tp_result_size_h << std::endl;
    }
    if ( collect_ee ) {
        n = params.nb_all_edges;
        // max_dist *= 0.5f;
        // cudaMemsetAsync(ee_result_size.data().get(), 0, sizeof(int));
        // edges_query_edges_via_point_hash<<<(n + block - 1) / block, block>>>(
        //     ee_collision_result.data().get(),
        //     ee_result_size.data().get(),
        //     edges.data().get(),
        //     points,
        //     // vertices_old.data().get(),
        //     point_hashes.data().get(),
        //     point_hash_table_lookup.data().get(),
        //     dir_edges.data().get(),
        //     edge_lookup.data().get(),
        //     vertices_mask.data().get(), cell_size,
        //     max_dist * max_dist, max_dist * 0.5, point_hash_table_size,
        //     point_hashes_size_h, max_ee_result_size,
        //     params.nb_all_cloth_vertices, n);
        // cudaMemcpy(&ee_result_size_h, ee_result_size.data().get(), sizeof(int), cudaMemcpyDeviceToHost);
        // ee_result_size_h = min(ee_result_size_h, max_ee_result_size);
        // RECORD_TIME("edges_query_edges_via_point_hash");
        block = 64;
        detect_edge_edge_constraints<<<(n + block - 1) / block, block>>>(
            normal_constraints.data().get(),
            constraint_size_p.ptr,
            edges.data().get(),
            points,
            edge_hashes.data().get(),
            edge_hash_table_lookup.data().get(),
            vertices_mask.data().get(),
            cell_size, max_dist,
            max_collision_constraints_size,
            edge_hash_table_size,
            edge_hashes_size_h,
            params.nb_all_cloth_vertices,
            n);
        cudaMemcpy(&ee_result_size_h, constraint_size_p.ptr, sizeof(int), cudaMemcpyDeviceToHost);
        RECORD_TIME("detect_edge_edge_constraints");
        ee_result_size_h = min(ee_result_size_h, max_collision_constraints_size);
        ee_result_size_h -= tp_result_size_h;
        std::cout << "ee_result_size_h = " << ee_result_size_h << std::endl;
    }
}
void SolverBase::init_collision() {
    point_hash_table_size = max(67, next_prime((uint32_t)params.nb_all_cloth_vertices));
    edge_hash_table_size = max(67, next_prime((uint32_t)params.nb_all_cloth_edges));
    point_hash_table.resize(point_hash_table_size);
    max_pp_result_size = params.nb_all_cloth_vertices * 2;
    max_tp_result_size = params.nb_all_triangles * 2;
    max_ee_result_size = params.nb_all_edges * 2;
    max_point_hashes_size = max_pp_result_size * 8;
    max_edge_hashes_size = params.nb_all_cloth_edges * 8;
    point_hashes.resize(max_point_hashes_size);
    edge_hashes.resize(max_edge_hashes_size);
    max_sort_result_size = max(max_point_hashes_size, max_edge_hashes_size);
    sort_key_temp.resize(max_sort_result_size);

    sort_result_size.resize(1);
    pp_result_size.resize(1);
    tp_result_size.resize(1);
    ee_result_size.resize(1);
    uncolored_count.resize(1);
    vertex_color_claimer.resize(params.nb_all_vertices * MAX_COLORS);
    vertex_forbidden_masks.resize(params.nb_all_vertices);

    pp_collision_result.resize(max_pp_result_size);
    tp_collision_result.resize(max_tp_result_size);
    ee_collision_result.resize(max_ee_result_size);
    max_collision_constraints_size = max_pp_result_size + max_tp_result_size + max_ee_result_size;
    normal_constraints.resize(max_collision_constraints_size);
    collision_constraints.resize(max_collision_constraints_size);
    constraint_colors.resize(max_collision_constraints_size);
    temp_vertices_f3.resize(params.nb_all_vertices);
    point_hash_table_lookup.resize(point_hash_table_size + 1);
    edge_hash_table_lookup.resize(edge_hash_table_size + 1);
    mass_inv.resize(params.nb_all_vertices);
    weight.resize(params.nb_all_vertices);
    constraint_color_offsets.resize(MAX_COLORS + 2);
    points_safe.reserve(params.nb_all_vertices);
    temp_vertices2_f3.reserve(params.nb_all_vertices);
    temp_vertices3_f3.reserve(params.nb_all_vertices);
    // temp_edge_f3.resize(params.nb_all_cloth_edges);
    // vertex_colors.resize(params.nb_all_cloth_vertices);
    // sort_key_temp.resize(max_pp_result_size);
    // sort_value_temp.resize(max_pp_result_size);
    // pp_result_size.resize(1);
    alpha_hard = 0.005f;
}
