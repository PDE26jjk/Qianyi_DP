#pragma once

#include "common/perf_timing.h"
static PerfTiming& globalTimer() {
    return PerfTiming::global_timer();
}
void sort_benchmark(const std::vector<size_t>& sizes, int warmup = 2, int runs = 10, bool verify = false);

std::vector<int> bvh_benchmark(const std::vector<float>& vertices_in, const std::vector<int>& faces_in,
    const std::vector<float>& query_points, int warmup, int runs, bool verify);
std::vector<int> bvh_edge_benchmark(const std::vector<float>& vertices_in,
    const std::vector<int>& edge_in,
    const std::vector<float>& query_points, int warmup, int runs, bool verify);
std::vector<int> bvh2_benchmark(
    const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    const std::vector<float>& query_points,
    int warmup, int runs, bool verify);

std::vector<float> sdf_benchmark(
    const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    bool use_parity,
    int warmup, int runs);

std::vector<float> sdf_check_inside(
    const std::vector<float>& vertices_in,
    const std::vector<int>& faces_in,
    const std::vector<float>& query_points,
    int warmup, int runs
);

std::vector<int> graph_coloring_benchmark( 
    int num_nodes,
    const std::vector<int>& edges,
    int algorithm,
    bool balance,
    float target_max_min_ratio,
    int warmup, int runs, bool verify);
