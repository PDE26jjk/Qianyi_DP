#pragma once
#include <cub/device/device_radix_sort.cuh>

#include "common/vec_math.h"
#include "common/atomic_utils.cuh"
#include "cuda_tools/cub_tools.cuh"

template<typename T>
__global__ void laplacian_smoothing(
    T* __restrict__ result,
    const T* __restrict__ points_data,
    const char* __restrict__ mask,
    const int2* __restrict__ edge_lookup,
    const int2* __restrict__ dir_edges,
    float rate,
    int num_points
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if ( idx >= num_points ) return;
    if ( mask && mask[idx] ) return;
    auto [offset, degree] = edge_lookup[idx];
    T res{};
    T data_center = points_data[idx];
    for ( int d = 0; d < degree; d++ ) {
        int idx_adj = dir_edges[offset + d].x;
        res = res + points_data[idx_adj] - data_center;
    }
    result[idx] = data_center + res * rate;
}
// Function to find edge index by vertices
static __device__ int v2e(int v0, int v1, const int2* lookup, const int2* dir_edges) {
    int2 range = lookup[v0]; // x: offset, y: degree
    for ( int i = 0; i < range.y; ++i ) {
        int2 entry = dir_edges[range.x + i]; // x: target_v, y: original_edge_id
        if ( entry.x == v1 ) return entry.y;
    }
    return -1;
}
static __device__ int v2e_include_stitches(int v0, int v1, const int2* lookup, const int2* dir_edges) {
    int2 range = lookup[v1]; // x: offset, y: degree
    for ( int i = 0; i < range.y; ++i ) {
        int2 entry = dir_edges[range.x + i]; // x: target_v, y: original_edge_id
        if ( entry.x == v0 ) return entry.y;
    }
    range = lookup[v0];
    for ( int i = 0; i < range.y; ++i ) {
        int2 entry = dir_edges[range.x + i];
        if ( entry.x == v1 ) return entry.y;
    }
    return -1;
}

static __device__ bool find_edge(int v0, int v1, const int2* lookup, const int2* dir_edges, int& edge) {
    auto [offset, degree] = lookup[v0];
    if ( dir_edges[offset].x <= v1 && v1 <= dir_edges[offset + degree - 1].x ) {
        for ( int d = 0; d < degree; d++ ) {
            if ( dir_edges[offset + d].x == v1 ) {
                edge = dir_edges[offset + d].y;
                return true;
            }
        }
    }
    return false;
}

static __global__ void prepare_keys_and_values(
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

static __global__ void unpack_dir_edges(
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

static __global__ void compute_lookup(
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

inline void edges_to_csr(
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

    // Stable sort by key using CUB DeviceRadixSort (in-place)
    CALL_CUBS(DeviceRadixSort::SortPairs, d_sort_keys, d_sort_keys,
        d_sort_values, d_sort_values, num_dir_edges);

    blocks = (num_dir_edges + threads - 1) / threads;
    unpack_dir_edges<<<blocks, threads>>>(d_sort_keys, d_sort_values, d_dir_edges, num_dir_edges);

    blocks = (nb_all_vertices + threads - 1) / threads;
    compute_lookup<<<blocks, threads>>>(d_sort_keys, (int)num_dir_edges, nb_all_vertices, d_edge_lookup);

    cudaFree(d_sort_keys);
    cudaFree(d_sort_values);
}
