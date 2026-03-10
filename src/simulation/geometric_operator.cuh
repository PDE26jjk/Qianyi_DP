#pragma once
#include "common/vec_math.h"
#include "common/atomic_utils.cuh"

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

