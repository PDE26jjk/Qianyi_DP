// Copyright (c) 2024 Graph Coloring Benchmark Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_vector.h>

#include "benchmarks.h"

#include "common/vec_math.h"
#include "simulation/color_graph/coloring.h"



std::vector<int> graph_coloring_benchmark(
    int num_nodes, const std::vector<int>& edges,
    int algorithm, bool balance, float target_max_min_ratio,
    int warmup, int runs, bool verify
) {
    auto& timer = globalTimer();

    int num_edges = edges.size() / 2;
    std::vector<int2> work_edges(num_edges);
    memcpy(work_edges.data(), edges.data(), work_edges.size() * sizeof(int2));

    // Color output buffer – resized each call to guarantee num_nodes elements.
    std::vector<int> node_colors(num_nodes);
    int num_colors = 0;

    const char* algo_name = (algorithm == 0) ? "graph_coloring_mcs" :
                                (algorithm == 1) ? "graph_coloring_greedy" : "graph_coloring_cuda";
    thrust::device_vector<int2> d_edges(work_edges);
    thrust::device_vector<int> d_node_colors(num_nodes);
    for ( int i = 0; i < warmup; ++i ) {
        node_colors.assign(num_nodes, -1);
        if ( algorithm < 2 )
            num_colors = graph_coloring_cpu(num_nodes, work_edges, algorithm, node_colors, balance, target_max_min_ratio);
        else {
            thrust::fill(d_node_colors.begin(), d_node_colors.end(), -1);
            num_colors = graph_coloring_cuda(
                num_nodes, num_edges,
                thrust::raw_pointer_cast(d_edges.data()),
                thrust::raw_pointer_cast(d_node_colors.data()),
                balance,
                target_max_min_ratio
                );
        }
    }

    for ( int i = 0; i < runs; ++i ) {
        timer.start(algo_name);
        if ( algorithm < 2 ) {

            node_colors.assign(num_nodes, -1);
            num_colors = graph_coloring_cpu(num_nodes, work_edges, algorithm, node_colors, balance, target_max_min_ratio);
        }
        else {
            thrust::fill(d_node_colors.begin(), d_node_colors.end(), -1);
            num_colors = graph_coloring_cuda(
                num_nodes, num_edges,
                thrust::raw_pointer_cast(d_edges.data()),
                thrust::raw_pointer_cast(d_node_colors.data()),
                balance,
                target_max_min_ratio
                );
            if ( i == runs - 1 )
                thrust::copy(d_node_colors.begin(), d_node_colors.end(),
                    node_colors.begin());
        }
        timer.stop();
    }

    // printf("Number of colors = %d\n", num_colors);
    if ( verify )
        return node_colors;

    return {};
}
