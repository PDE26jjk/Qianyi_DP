#include "coloring.h"

// [1] D. A. Grable and A. Panconesi, "Fast Distributed Algorithms for Brooks-Vizing Colorings,"
// J. Algorithms, vol. 37, no. 1, pp. 85–120, Oct. 2000, doi: 10.1006/jagm.2000.1097.
// [2] M. Fratarcangeli, V. Tibaldo, and F. Pellacini, "Vivace: a practical gauss-seidel method for
// stable soft body dynamics" , ACM Trans. Graph., vol. 35, no. 6, pp. 1–9, Nov. 2016,
// doi: 10.1145/2980179.2982437.

// Building on Grable and Panconesi's algorithm with the addition of a balancing operation,
// in each round, each node attempts to recolor itself to the least frequent color among
// its available options, followed by conflict resolution.

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <thrust/device_vector.h>

#include "simulation/geometric_operator.cuh"
#include "simulation/cuda_tools/cub_tools.cuh"

#define MAX_COLORS 64            // Colours are encoded as bitmask bits
#define MAX_ITER   200           // Safety iteration bound
#define MAX_BALANCE_ITER 8       // 


__global__ void k_mark_forbidden_bits_vertex(
    const int* __restrict__ node_colors,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    uint64_t* __restrict__ vertex_forbidden,
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;

    int c = node_colors[v];
    if ( c == -1 ) return;

    uint64_t bit = 1ull << c;
    int2 lookup = edge_lookup[v];
    int off = lookup.x;
    int cnt = lookup.y;

    for ( int i = 0; i < cnt; ++i ) {
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
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;
    if ( node_colors[v] != -1 ) return;

    uint64_t forbidden = vertex_forbidden[v];

    int palette = *current_palette_size;
    // Guard against undefined behaviour when palette == MAX_COLORS.
    uint64_t palette_mask = (palette >= MAX_COLORS) ? ~0ull
                                : ((1ull << palette) - 1);
    uint64_t available = (~forbidden) & palette_mask;

    if ( available == 0 ) {
        d_candidate_colors[v] = -1;
        return;
    }

    int pop = __popcll(available);
    unsigned int hash = (v ^ (*iteration)) * 1103515245u + 12345u;
    hash = (hash ^ (hash >> 16)) * 2654435769u;
    int r = hash % pop;

    uint64_t temp = available;
    for ( int i = 0; i < r; ++i ) {
        int pos = __ffsll(temp) - 1;
        temp &= ~(1ull << pos);
    }
    int chosen = __ffsll(temp) - 1;

    // Claim this colour for myself and every direct neighbour.
    if ( atomicMax(&vertex_color_claimer[v * MAX_COLORS + chosen], v) > v ) {
        d_candidate_colors[v] = -1;
    }
    else {
        int2 lookup = edge_lookup[v];
        int off = lookup.x;
        int cnt = lookup.y;
        for ( int i = 0; i < cnt; ++i ) {
            int nb = dir_edges[off + i].x;
            atomicMax(&vertex_color_claimer[nb * MAX_COLORS + chosen], v);
        }
        d_candidate_colors[v] = chosen;
    }
}

__global__ void k_verify_colors_vertex(
    int* __restrict__ node_colors,
    const int* __restrict__ vertex_color_claimer,
    const int* __restrict__ d_candidate_colors,
    int* __restrict__ uncolored_count,
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;
    if ( node_colors[v] != -1 ) return;

    int c = d_candidate_colors[v];
    if ( c == -1 ) {
        atomicAdd(uncolored_count, 1);
        return;
    }

    if ( vertex_color_claimer[v * MAX_COLORS + c] == v ) {
        node_colors[v] = c;
    }
    else {
        atomicAdd(uncolored_count, 1);
    }
}

__global__ void k_update_colors(
    int* __restrict__ current_palette_size,
    int* __restrict__ uncolored_count,
    int* __restrict__ last_uncolored_count,
    int* __restrict__ iteration) {
    if ( blockIdx.x * blockDim.x + threadIdx.x != 0 ) return;

    if ( *last_uncolored_count == *uncolored_count && *uncolored_count > 0 ) {
        if ( *current_palette_size < MAX_COLORS )
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
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;
    int c = node_colors[v];
    if ( c >= 0 && c < MAX_COLORS ) {
        atomicAdd(&color_counts[c], 1);
    }
}

__global__ void k_generate_priorities_kernel(
    int* __restrict__ d_priorities,
    int iteration,
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;
    unsigned int hash = (v ^ (iteration * 2654435761u)) * 1103515245u + 12345u;
    hash = (hash ^ (hash >> 16)) * 2654435769u;
    d_priorities[v] = (int)hash;
}

__global__ void k_mark_forbidden_balance_kernel(
    const int* __restrict__ node_colors,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    uint64_t* __restrict__ vertex_forbidden,
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;

    int c = node_colors[v];
    if ( c < 0 || c >= MAX_COLORS ) return;

    uint64_t bit = 1ull << c;
    int2 lookup = edge_lookup[v];
    int off = lookup.x;
    int cnt = lookup.y;

    for ( int i = 0; i < cnt; ++i ) {
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
    int num_colors
) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;

    int current_c = node_colors[v];
    if ( current_c < 0 ) return;

    uint64_t forbidden = vertex_forbidden[v];
    uint64_t available_mask = ~forbidden;
    if ( num_colors < 64 ) {
        available_mask &= ((1ull << num_colors) - 1);
    }

    int best_c = current_c;
    int best_count = color_counts[current_c];

    uint64_t temp = available_mask;
    while ( temp ) {
        int c = __ffsll(temp) - 1;
        temp &= ~(1ull << c);
        if ( color_counts[c] < best_count ) {
            best_count = color_counts[c];
            best_c = c;
        }
    }

    if ( best_c != current_c ) {
        int prio = d_priorities[v];

        // Early exit: if someone with higher priority already claimed it,
        // we give up immediately and do NOT write to neighbours.
        if ( atomicMax(&vertex_color_claimer[v * MAX_COLORS + best_c], prio) >= prio ) {
            d_candidate_colors[v] = -1;
            return;
        }

        int2 lookup = edge_lookup[v];
        int off = lookup.x;
        int cnt = lookup.y;
        for ( int i = 0; i < cnt; ++i ) {
            int nb = dir_edges[off + i].x;
            if ( atomicMax(&vertex_color_claimer[nb * MAX_COLORS + best_c], prio) >= prio ) {
                d_candidate_colors[v] = -1;
                return;
            }
        }
        d_candidate_colors[v] = best_c;
    }
    else {
        d_candidate_colors[v] = -1;
    }
}

__global__ void k_verify_balanced_kernel(
    int* __restrict__ node_colors,
    const int* __restrict__ d_candidate_colors,
    const int* __restrict__ vertex_color_claimer,
    const int* __restrict__ d_priorities,
    int* __restrict__ d_changed_count,
    int num_vertices) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if ( v >= num_vertices ) return;

    int target_c = d_candidate_colors[v];
    if ( target_c == -1 ) return;

    int prio = d_priorities[v];
    if ( vertex_color_claimer[v * MAX_COLORS + target_c] == prio ) {
        node_colors[v] = target_c;
        atomicAdd(d_changed_count, 1);
    }
}

inline int max_color(int* d_node_colors, int num_vertices) {
    int* d_max_color;
    cudaMalloc(&d_max_color, sizeof(int));
    CALL_CUBS(DeviceReduce::Max, d_node_colors, d_max_color, num_vertices);
    int h_max_color;
    cudaMemcpy(&h_max_color, d_max_color, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_max_color);
    return h_max_color;
}
int graph_coloring_cuda(
    int num_vertices, int num_edges, int2* d_edges, int* d_node_colors,
    bool balance, float target_max_min_ratio) {
    // --- 1. Copy edges to device and build CSR ---
    thrust::device_vector<int2> d_dir_edges(2 * num_edges);
    thrust::device_vector<int2> d_edge_lookup(num_vertices);

    edges_to_csr(num_vertices, num_edges,
        d_edges,
        thrust::raw_pointer_cast(d_dir_edges.data()),
        thrust::raw_pointer_cast(d_edge_lookup.data()));

    // --- 2. Allocate colouring state ---
    cudaMemsetAsync(d_node_colors, -1, sizeof(int) * num_vertices);
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

    for ( int iter = 0; iter < MAX_ITER; ++iter ) {
        // Reset per‑iteration arrays
        // thrust::fill(d_forbidden.begin(), d_forbidden.end(), 0);
        // thrust::fill(d_claimer.begin(), d_claimer.end(), -1);
        // thrust::fill(d_candidate.begin(), d_candidate.end(), -1);
        // thrust::fill(d_uncolored.begin(), d_uncolored.end(), 0);
        cudaMemsetAsync(d_forbidden.data().get(), 0, num_vertices * sizeof(uint64_t));
        cudaMemsetAsync(d_claimer.data().get(), -1, num_vertices * MAX_COLORS * sizeof(int));
        cudaMemsetAsync(d_candidate.data().get(), -1, num_vertices * sizeof(int));
        cudaMemsetAsync(d_uncolored.data().get(), 0, sizeof(int));

        // Mark forbidden bits
        k_mark_forbidden_bits_vertex<<<gridV, blockSize>>>(
            d_node_colors,
            thrust::raw_pointer_cast(d_edge_lookup.data()),
            thrust::raw_pointer_cast(d_dir_edges.data()),
            thrust::raw_pointer_cast(d_forbidden.data()),
            num_vertices);

        // Claim colours
        k_claim_color_bitmask_vertex<<<gridV, blockSize>>>(
            d_node_colors,
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
            d_node_colors,
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

        if ( d_uncolored[0] == 0 )
            break;
    }

    // --- 4. Copy results back to host ---

    int num_colors = 0;
    if ( num_vertices > 0 ) {
        num_colors = 1 + max_color(d_node_colors, num_vertices);
    }
    // --- 5. Colour balancing (only if we have at least 2 colours) ---
    if ( balance && num_colors > 1 && num_colors <= MAX_COLORS ) {
        thrust::device_vector<int> d_color_counts(MAX_COLORS, 0);
        thrust::device_vector<int> d_priorities(num_vertices);
        thrust::device_vector<int> d_changed_count(1, 0);

        for ( int bal_iter = 0; bal_iter < MAX_BALANCE_ITER; ++bal_iter ) {
            // Reset arrays for this iteration
            // thrust::fill(d_color_counts.begin(), d_color_counts.end(), 0);
            // thrust::fill(d_changed_count.begin(), d_changed_count.end(), 0);
            // thrust::fill(d_forbidden.begin(), d_forbidden.end(), 0);
            // // Reset claimer to minimum int value (priorities can be negative)
            // thrust::fill(d_claimer.begin(), d_claimer.end(),
            //     std::numeric_limits<int>::lowest());
            // thrust::fill(d_candidate.begin(), d_candidate.end(), -1);
            cudaMemsetAsync(thrust::raw_pointer_cast(d_color_counts.data()), 0,
                MAX_COLORS * sizeof(int));
            cudaMemsetAsync(thrust::raw_pointer_cast(d_changed_count.data()), 0,
                sizeof(int));
            cudaMemsetAsync(thrust::raw_pointer_cast(d_forbidden.data()), 0,
                num_vertices * sizeof(uint64_t));

            // claimer: set to -1 (all bytes 0xFF) – our priorities are non‑negative
            cudaMemsetAsync(thrust::raw_pointer_cast(d_claimer.data()), 0xFF,
                num_vertices * MAX_COLORS * sizeof(int));

            // candidate: set to -1 (no migration proposed yet)
            cudaMemsetAsync(thrust::raw_pointer_cast(d_candidate.data()), 0xFF,
                num_vertices * sizeof(int));
            // Count colours, generate priorities, mark forbidden
            // k_count_colors_kernel<<<gridV, blockSize>>>(
            //     d_node_colors,
            //     thrust::raw_pointer_cast(d_color_counts.data()),
            //     num_vertices);
            CALL_CUBS(DeviceHistogram::HistogramEven,
                d_node_colors, // d_samples
                thrust::raw_pointer_cast(d_color_counts.data()), // d_histogram
                MAX_COLORS + 1,   // num_levels = bins + 1
                0,                // lower_level
                MAX_COLORS,       // upper_level
                num_vertices);

            k_generate_priorities_kernel<<<gridV, blockSize>>>(
                thrust::raw_pointer_cast(d_priorities.data()),
                bal_iter,
                num_vertices);

            k_mark_forbidden_balance_kernel<<<gridV, blockSize>>>(
                d_node_colors,
                thrust::raw_pointer_cast(d_edge_lookup.data()),
                thrust::raw_pointer_cast(d_dir_edges.data()),
                thrust::raw_pointer_cast(d_forbidden.data()),
                num_vertices);

            // Propose migrations
            k_claim_balanced_kernel<<<gridV, blockSize>>>(
                d_node_colors,
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
                d_node_colors,
                thrust::raw_pointer_cast(d_candidate.data()),
                thrust::raw_pointer_cast(d_claimer.data()),
                thrust::raw_pointer_cast(d_priorities.data()),
                thrust::raw_pointer_cast(d_changed_count.data()),
                num_vertices);

            if ( d_changed_count[0] == 0 ) break;
        }

        // Update number of colours
        if ( num_vertices > 0 ) {
            num_colors = 1 + max_color(d_node_colors, num_vertices);
        }
    }
    // All device vectors are freed automatically – no explicit cudaFree calls.
    return num_colors;
}
