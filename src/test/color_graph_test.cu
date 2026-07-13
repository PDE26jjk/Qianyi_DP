// Copyright (c) 2024 Graph Coloring Benchmark Authors. All rights reserved.
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

#include <vector>
#include <cub/device/device_radix_sort.cuh>
#include <thrust/device_vector.h>

#include "benchmarks.h"
#include "geometry/color_graph/coloring.h"

#include "common/vec_math.h"

__global__ void prepare_keys_and_values(
    int2* __restrict__ d_edges,
    unsigned long long* __restrict__ d_keys,
    int* __restrict__ d_vals,
    size_t num_edges) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= num_edges ) return;

    int2 e = d_edges[i];
    if ( e.x > e.y ) {
        e = make_int2(e.y, e.x);
        d_edges[i] = e;          // enforce u <= v in-place
    }
    int u = e.x;
    int v = e.y;

    d_keys[i] = ((unsigned long long)u << 32) | (unsigned int)v;
    d_vals[i] = (int)i;
    d_keys[i + num_edges] = ((unsigned long long)v << 32) | (unsigned int)u;
    d_vals[i + num_edges] = (int)i;
}

__global__ void unpack_dir_edges(
    const unsigned long long* __restrict__ d_keys,
    const int* __restrict__ d_vals,
    int2* __restrict__ d_dir_edges,
    size_t num_dir_edges) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_dir_edges ) return;

    int target = (int)(d_keys[idx] & 0xFFFFFFFFULL);
    int edge_id = d_vals[idx];
    d_dir_edges[idx] = make_int2(target, edge_id);
}

__global__ void compute_lookup(
    const unsigned long long* __restrict__ d_sorted_keys,
    int num_dir_edges,
    int nb_all_v,
    int2* __restrict__ d_lookup) {
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if ( vid >= nb_all_v ) return;

    // Lower bound: first index where source >= vid
    int low = 0, high = num_dir_edges;
    while ( low < high ) {
        int mid = (low + high) >> 1;
        int src = (int)(d_sorted_keys[mid] >> 32);
        if ( src < vid )
            low = mid + 1;
        else
            high = mid;
    }
    int lower = low;

    // Upper bound: first index where source > vid
    low = 0;
    high = num_dir_edges;
    while ( low < high ) {
        int mid = (low + high) >> 1;
        int src = (int)(d_sorted_keys[mid] >> 32);
        if ( src <= vid )
            low = mid + 1;
        else
            high = mid;
    }
    int upper = low;

    d_lookup[vid] = make_int2(lower, upper - lower);
}

void edges_to_csr(
    int nb_all_vertices,
    size_t num_edges,
    int2* d_edges,              // in/out: edges, will be normalized (u <= v) in-place
    int2* d_dir_edges,          // out:  size = 2 * num_edges,  [target, edge_id] per element
    int2* d_edge_lookup)        // out:  size = nb_all_vertices, [offset, count] per vertex
{
    size_t num_dir_edges = num_edges * 2;

    // Temporary buffers for sort keys and values
    unsigned long long* d_sort_keys = nullptr;
    int* d_sort_values = nullptr;
    cudaMalloc(&d_sort_keys, num_dir_edges * sizeof(unsigned long long));
    cudaMalloc(&d_sort_values, num_dir_edges * sizeof(int));

    int threads = 256;
    int blocks = (num_edges + threads - 1) / threads;
    prepare_keys_and_values<<<blocks, threads>>>(d_edges, d_sort_keys, d_sort_values, num_edges);
    CUDA_CHECK(cudaGetLastError());

    // Stable sort by key using CUB DeviceRadixSort (in-place)
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_sort_keys, d_sort_keys,
        d_sort_values, d_sort_values, num_dir_edges);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, temp_storage_bytes, d_sort_keys, d_sort_keys,
        d_sort_values, d_sort_values, num_dir_edges);
    cudaFree(d_temp_storage);

    blocks = (num_dir_edges + threads - 1) / threads;
    unpack_dir_edges<<<blocks, threads>>>(d_sort_keys, d_sort_values, d_dir_edges, num_dir_edges);
    CUDA_CHECK(cudaGetLastError());

    blocks = (nb_all_vertices + threads - 1) / threads;
    compute_lookup<<<blocks, threads>>>(d_sort_keys, (int)num_dir_edges, nb_all_vertices, d_edge_lookup);
    CUDA_CHECK(cudaGetLastError());

    cudaFree(d_sort_keys);
    cudaFree(d_sort_values);
}

#define MAX_COLORS 64            // Colours are encoded as bitmask bits
#define MAX_ITER   200           // Safety iteration bound
#define MAX_BALANCE_ITER 50      // 


__global__ void k_mark_forbidden_bits_vertex(
    const int* __restrict__ node_colors,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    uint64_t* __restrict__ vertex_forbidden,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int c = node_colors[v];
    if (c == -1) return;

    uint64_t bit = 1ull << c;
    int2 lookup = edge_lookup[v];
    int off = lookup.x;
    int cnt = lookup.y;

    for (int i = 0; i < cnt; ++i) {
        int nb = dir_edges[off + i].x;
        atomicOr(&vertex_forbidden[nb], bit);
    }
}

__global__ void k_claim_color_bitmask_vertex(
    const int* __restrict__ node_colors,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    const uint64_t* __restrict__ vertex_forbidden,
    int* __restrict__ vertex_color_claimer,
    int* __restrict__ d_candidate_colors,
    const int* current_palette_size,
    const int* iteration,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (node_colors[v] != -1) return;

    // Bug 1 fixed: forbidden = direct neighbours' colours only.
    uint64_t forbidden = vertex_forbidden[v];

    int palette = *current_palette_size;
    // Guard against undefined behaviour when palette == MAX_COLORS.
    uint64_t palette_mask = (palette >= MAX_COLORS) ? ~0ull
                                                    : ((1ull << palette) - 1);
    uint64_t available = (~forbidden) & palette_mask;

    if (available == 0) {
        d_candidate_colors[v] = -1;
        return;
    }

    int pop = __popcll(available);
    unsigned int hash = (v ^ (*iteration)) * 1103515245u + 12345u;
    hash = (hash ^ (hash >> 16)) * 2654435769u;
    int r = hash % pop;

    uint64_t temp = available;
    for (int i = 0; i < r; ++i) {
        int pos = __ffsll(temp) - 1;
        temp &= ~(1ull << pos);
    }
    int chosen = __ffsll(temp) - 1;

    d_candidate_colors[v] = chosen;

    // Claim this colour for myself and every direct neighbour.
    int* base = vertex_color_claimer + v * MAX_COLORS;
    atomicMax(&base[chosen], v);
    int2 lookup = edge_lookup[v];
    int off = lookup.x;
    int cnt = lookup.y;
    for (int i = 0; i < cnt; ++i) {
        int nb = dir_edges[off + i].x;
        atomicMax(&vertex_color_claimer[nb * MAX_COLORS + chosen], v);
    }
}

__global__ void k_verify_colors_vertex(
    int* __restrict__ node_colors,
    const int* __restrict__ vertex_color_claimer,
    const int* __restrict__ d_candidate_colors,
    int* __restrict__ uncolored_count,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    if (node_colors[v] != -1) return;

    int c = d_candidate_colors[v];
    if (c == -1) {
        atomicAdd(uncolored_count, 1);
        return;
    }

    // Bug 2 fixed: only check that v is the maximum ID that claimed colour c
    // within its own closed neighbourhood.
    if (vertex_color_claimer[v * MAX_COLORS + c] == v) {
        node_colors[v] = c;
    } else {
        atomicAdd(uncolored_count, 1);
    }
}

__global__ void k_update_colors(
    int* __restrict__ current_palette_size,
    int* __restrict__ uncolored_count,
    int* __restrict__ last_uncolored_count,
    int* __restrict__ iteration)
{
    if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;

    if (*last_uncolored_count == *uncolored_count && *uncolored_count > 0) {
        if (*current_palette_size < MAX_COLORS)
            (*current_palette_size)++;
    }
    *last_uncolored_count = *uncolored_count;
    (*iteration)++;
}

// ===========================================================================
//                          COLOUR BALANCING KERNELS
// ===========================================================================

__global__ void k_count_colors_kernel(
    const int* __restrict__ node_colors,
    int* __restrict__ color_counts,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    int c = node_colors[v];
    if (c >= 0 && c < MAX_COLORS) {
        atomicAdd(&color_counts[c], 1);
    }
}

__global__ void k_generate_priorities_kernel(
    int* __restrict__ d_priorities,
    int iteration,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;
    unsigned int hash = (v ^ (iteration * 2654435761u)) * 1103515245u + 12345u;
    hash = (hash ^ (hash >> 16)) * 2654435769u;
    d_priorities[v] = (int)hash;
}

__global__ void k_mark_forbidden_balance_kernel(
    const int* __restrict__ node_colors,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    uint64_t* __restrict__ vertex_forbidden,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int c = node_colors[v];
    if (c < 0 || c >= MAX_COLORS) return;

    uint64_t bit = 1ull << c;
    int2 lookup = edge_lookup[v];
    int off = lookup.x;
    int cnt = lookup.y;

    for (int i = 0; i < cnt; ++i) {
        int nb = dir_edges[off + i].x;
        atomicOr(&vertex_forbidden[nb], bit);
    }
}

__global__ void k_claim_balanced_kernel(
    const int* __restrict__ node_colors,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    const uint64_t* __restrict__ vertex_forbidden,
    const int* __restrict__ color_counts,
    const int* __restrict__ d_priorities,
    int* __restrict__ vertex_color_claimer,
    int* __restrict__ d_candidate_colors,
    int num_vertices,
    int num_colors)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int current_c = node_colors[v];
    if (current_c < 0) return;

    uint64_t forbidden = vertex_forbidden[v];
    uint64_t available_mask = ~forbidden;
    if (num_colors < 64) {
        available_mask &= ((1ull << num_colors) - 1);
    }

    int best_c = current_c;
    int best_count = color_counts[current_c];

    uint64_t temp = available_mask;
    while (temp) {
        int c = __ffsll(temp) - 1;
        temp &= ~(1ull << c);
        if (color_counts[c] < best_count) {
            best_count = color_counts[c];
            best_c = c;
        }
    }

    if (best_c != current_c) {
        d_candidate_colors[v] = best_c;
        int prio = d_priorities[v];

        int* base = vertex_color_claimer + v * MAX_COLORS;
        atomicMax(&base[best_c], prio);

        int2 lookup = edge_lookup[v];
        int off = lookup.x;
        int cnt = lookup.y;
        for (int i = 0; i < cnt; ++i) {
            int nb = dir_edges[off + i].x;
            atomicMax(&vertex_color_claimer[nb * MAX_COLORS + best_c], prio);
        }
    } else {
        d_candidate_colors[v] = -1;
    }
}

__global__ void k_verify_balanced_kernel(
    int* __restrict__ node_colors,
    const int* __restrict__ d_candidate_colors,
    const int* __restrict__ vertex_color_claimer,
    const int* __restrict__ d_priorities,
    int* __restrict__ d_changed_count,
    int num_vertices)
{
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    int target_c = d_candidate_colors[v];
    if (target_c == -1) return;

    int prio = d_priorities[v];
    if (vertex_color_claimer[v * MAX_COLORS + target_c] == prio) {
        node_colors[v] = target_c;
        atomicAdd(d_changed_count, 1);
    }
}

int graph_coloring_cuda(
    int num_vertices,
    std::vector<int2>& edges,
    std::vector<int>& node_colors)
{
    size_t num_edges = edges.size();
    if (num_edges == 0) {
        node_colors.assign(num_vertices, 0);
        return 1;
    }

    // --- 1. Copy edges to device and build CSR ---
    thrust::device_vector<int2> d_edges(edges.begin(), edges.end());
    thrust::device_vector<int2> d_dir_edges(2 * num_edges);
    thrust::device_vector<int2> d_edge_lookup(num_vertices);

    edges_to_csr(num_vertices, num_edges,
                 thrust::raw_pointer_cast(d_edges.data()),
                 thrust::raw_pointer_cast(d_dir_edges.data()),
                 thrust::raw_pointer_cast(d_edge_lookup.data()));

    // --- 2. Allocate colouring state ---
    thrust::device_vector<int> d_node_colors(num_vertices, -1);
    thrust::device_vector<uint64_t> d_forbidden(num_vertices, 0);
    thrust::device_vector<int> d_claimer(num_vertices * MAX_COLORS, -1);
    thrust::device_vector<int> d_candidate(num_vertices, -1);
    thrust::device_vector<int> d_uncolored(1, 0);
    thrust::device_vector<int> d_last_uncolored(1, num_vertices); // large number
    thrust::device_vector<int> d_palette(1, 4);   // initial palette size
    thrust::device_vector<int> d_iteration(1, 0);

    // --- 3. Iterative colouring loop ---
    int blockSize = 256;
    int gridV = (num_vertices + blockSize - 1) / blockSize;

    for (int iter = 0; iter < MAX_ITER; ++iter) {
        // Reset per‑iteration arrays
        // thrust::fill(d_forbidden.begin(), d_forbidden.end(), 0);
        // thrust::fill(d_claimer.begin(), d_claimer.end(), -1);
        // thrust::fill(d_candidate.begin(), d_candidate.end(), -1);
        // thrust::fill(d_uncolored.begin(), d_uncolored.end(), 0);
        cudaMemset(d_forbidden.data().get(), 0, num_vertices * sizeof(uint64_t));
        cudaMemset(d_claimer.data().get(), -1, num_vertices * MAX_COLORS * sizeof(int));
        cudaMemset(d_candidate.data().get(), -1, num_vertices * sizeof(int));
        cudaMemset(d_uncolored.data().get(), 0, sizeof(int));
        
        // Mark forbidden bits
        k_mark_forbidden_bits_vertex<<<gridV, blockSize>>>(
            thrust::raw_pointer_cast(d_node_colors.data()),
            thrust::raw_pointer_cast(d_edge_lookup.data()),
            thrust::raw_pointer_cast(d_dir_edges.data()),
            thrust::raw_pointer_cast(d_forbidden.data()),
            num_vertices);

        // Claim colours
        k_claim_color_bitmask_vertex<<<gridV, blockSize>>>(
            thrust::raw_pointer_cast(d_node_colors.data()),
            thrust::raw_pointer_cast(d_edge_lookup.data()),
            thrust::raw_pointer_cast(d_dir_edges.data()),
            thrust::raw_pointer_cast(d_forbidden.data()),
            thrust::raw_pointer_cast(d_claimer.data()),
            thrust::raw_pointer_cast(d_candidate.data()),
            thrust::raw_pointer_cast(d_palette.data()),
            thrust::raw_pointer_cast(d_iteration.data()),
            num_vertices);

        // Verify and finalise
        k_verify_colors_vertex<<<gridV, blockSize>>>(
            thrust::raw_pointer_cast(d_node_colors.data()),
            thrust::raw_pointer_cast(d_claimer.data()),
            thrust::raw_pointer_cast(d_candidate.data()),
            thrust::raw_pointer_cast(d_uncolored.data()),
            num_vertices);

        // Update palette and iteration
        k_update_colors<<<1, 1>>>(
            thrust::raw_pointer_cast(d_palette.data()),
            thrust::raw_pointer_cast(d_uncolored.data()),
            thrust::raw_pointer_cast(d_last_uncolored.data()),
            thrust::raw_pointer_cast(d_iteration.data()));

        if (d_uncolored[0] == 0)
            break;
    }

    // --- 4. Copy results back to host ---
    node_colors.resize(num_vertices);
    thrust::copy(d_node_colors.begin(), d_node_colors.end(), node_colors.begin());

    int num_colors = 0;
    if (num_vertices > 0) {
        num_colors = 1 + *std::ranges::max_element(node_colors);
    }
    // --- 5. Colour balancing (only if we have at least 2 colours) ---
    if (num_colors > 1 && num_colors <= MAX_COLORS) {
        thrust::device_vector<int> d_color_counts(MAX_COLORS, 0);
        thrust::device_vector<int> d_priorities(num_vertices);
        thrust::device_vector<int> d_changed_count(1, 0);

        for (int bal_iter = 0; bal_iter < MAX_BALANCE_ITER; ++bal_iter) {
            // Reset arrays for this iteration
            thrust::fill(d_color_counts.begin(), d_color_counts.end(), 0);
            thrust::fill(d_changed_count.begin(), d_changed_count.end(), 0);
            thrust::fill(d_forbidden.begin(), d_forbidden.end(), 0);
            // Reset claimer to minimum int value (priorities can be negative)
            thrust::fill(d_claimer.begin(), d_claimer.end(),
                         std::numeric_limits<int>::lowest());
            thrust::fill(d_candidate.begin(), d_candidate.end(), -1);

            // Count colours, generate priorities, mark forbidden
            k_count_colors_kernel<<<gridV, blockSize>>>(
                thrust::raw_pointer_cast(d_node_colors.data()),
                thrust::raw_pointer_cast(d_color_counts.data()),
                num_vertices);

            k_generate_priorities_kernel<<<gridV, blockSize>>>(
                thrust::raw_pointer_cast(d_priorities.data()),
                bal_iter,
                num_vertices);

            k_mark_forbidden_balance_kernel<<<gridV, blockSize>>>(
                thrust::raw_pointer_cast(d_node_colors.data()),
                thrust::raw_pointer_cast(d_edge_lookup.data()),
                thrust::raw_pointer_cast(d_dir_edges.data()),
                thrust::raw_pointer_cast(d_forbidden.data()),
                num_vertices);

            // Propose migrations
            k_claim_balanced_kernel<<<gridV, blockSize>>>(
                thrust::raw_pointer_cast(d_node_colors.data()),
                thrust::raw_pointer_cast(d_edge_lookup.data()),
                thrust::raw_pointer_cast(d_dir_edges.data()),
                thrust::raw_pointer_cast(d_forbidden.data()),
                thrust::raw_pointer_cast(d_color_counts.data()),
                thrust::raw_pointer_cast(d_priorities.data()),
                thrust::raw_pointer_cast(d_claimer.data()),
                thrust::raw_pointer_cast(d_candidate.data()),
                num_vertices,
                num_colors);

            // Verify and apply migrations
            k_verify_balanced_kernel<<<gridV, blockSize>>>(
                thrust::raw_pointer_cast(d_node_colors.data()),
                thrust::raw_pointer_cast(d_candidate.data()),
                thrust::raw_pointer_cast(d_claimer.data()),
                thrust::raw_pointer_cast(d_priorities.data()),
                thrust::raw_pointer_cast(d_changed_count.data()),
                num_vertices);

            if (d_changed_count[0] == 0) break;
        }

        // Copy back final colours and update number of colours
        thrust::copy(d_node_colors.begin(), d_node_colors.end(), node_colors.begin());
        if (num_vertices > 0) {
            num_colors = 1 + *std::max_element(node_colors.begin(), node_colors.end());
        }
    }
    // All device vectors are freed automatically – no explicit cudaFree calls.
    return num_colors;
}



std::vector<int> graph_coloring_benchmark(
    int num_nodes, const std::vector<int>& edges,
    int algorithm, bool balance, float target_max_min_ratio,
    int warmup, int runs, bool verify
) {
    auto& timer = globalTimer();

    std::vector<int2> work_edges(edges.size() / 2);
    memcpy(work_edges.data(), edges.data(), work_edges.size() * sizeof(int2));

    // Color output buffer – resized each call to guarantee num_nodes elements.
    std::vector<int> node_colors(num_nodes);
    int num_colors = 0;

    const char* algo_name = (algorithm == 0) ? "graph_coloring_mcs" : "graph_coloring_greedy";

    for ( int i = 0; i < warmup; ++i ) {
        node_colors.assign(num_nodes, -1);
        if ( algorithm < 2 )
            num_colors = graph_coloring(num_nodes, work_edges, algorithm, node_colors, balance, target_max_min_ratio);
        else {
            graph_coloring_cuda(num_nodes, work_edges, node_colors);
        }
    }

    for ( int i = 0; i < runs; ++i ) {
        node_colors.assign(num_nodes, -1);
        timer.start(algo_name);
        if ( algorithm < 2 )
            num_colors = graph_coloring(num_nodes, work_edges, algorithm, node_colors, balance, target_max_min_ratio);
        else {
            graph_coloring_cuda(num_nodes, work_edges, node_colors);
        }
        timer.stop();
    } 

    // printf("Number of colors = %d\n", num_colors);
    if ( verify )
        return node_colors;

    return {};
}
