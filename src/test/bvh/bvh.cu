// lbvh3d/bvh.cu
#include "bvh.h"
#include "common/cuda_utils.h"
#include "common/vec_math.h"
#include "simulation/cuda_tools/sort.h"   

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace lbvh3dtest {

// -----------------------------------------------------------------------------
// Device atomic min/max for float
// -----------------------------------------------------------------------------
__device__ inline void atomicMaxFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) >= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

__device__ inline void atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) break;
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
}

// -----------------------------------------------------------------------------
// 30-bit Morton code generator (10 bits per component, 1024^3 grid)
// -----------------------------------------------------------------------------
__device__ inline uint32_t expand_bits(uint32_t v) {
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__device__ inline uint32_t morton3(float x, float y, float z) {
    x = fminf(fmaxf(x * 1024.0f, 0.0f), 1024.0f - 1.0f);
    y = fminf(fmaxf(y * 1024.0f, 0.0f), 1024.0f - 1.0f);
    z = fminf(fmaxf(z * 1024.0f, 0.0f), 1024.0f - 1.0f);
    uint32_t ix = expand_bits((uint32_t)x);
    uint32_t iy = expand_bits((uint32_t)y);
    uint32_t iz = expand_bits((uint32_t)z);
    return ix * 4 + iy * 2 + iz;
}



// -----------------------------------------------------------------------------
// Kernels
// -----------------------------------------------------------------------------
__global__ void memset_kernel(int* dest, int value, size_t n) {
    size_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < n) dest[tid] = value;
}

__global__ void compute_morton_codes(
    const float3* __restrict__ item_lowers,
    const float3* __restrict__ item_uppers,
    int n,
    const float3* grid_lower,
    const float3* grid_inv_edges,
    int* __restrict__ indices,
    uint64_t* __restrict__ keys)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        float3 lower = item_lowers[index];
        float3 upper = item_uppers[index];
        float3 center = (lower + upper) * 0.5f;
        float3 local = (center - grid_lower[0]) * grid_inv_edges[0];
        uint64_t code = static_cast<uint64_t>(morton3(local.x, local.y, local.z));
        indices[index] = index;
        keys[index] = code;
    }
}

__global__ void compute_key_deltas(const uint64_t* __restrict__ keys, int* __restrict__ deltas, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        uint64_t diff = keys[index] ^ keys[index + 1];
        deltas[index] = (diff == 0) ? 64 : __clzll(diff);
    }
}

__global__ void build_leaves(
    const float3* __restrict__ item_lowers,
    const float3* __restrict__ item_uppers,
    int n,
    const int* __restrict__ indices,
    int* __restrict__ range_lefts,
    int* __restrict__ range_rights,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < n) {
        int item = indices[index];
        lowers[index] = make_node(item_lowers[item], index, true);
        uppers[index] = make_node(item_uppers[item], index, false);
        range_lefts[index] = index;
        range_rights[index] = index;
    }
}

__global__ void build_hierarchy(
    int n,
    int* root,
    const int* __restrict__ deltas,
    const uint64_t* __restrict__ keys,
    int* __restrict__ num_children,
    const int* __restrict__ primitive_indices,
    volatile int* __restrict__ range_lefts,
    volatile int* __restrict__ range_rights,
    volatile int* __restrict__ parents,
    volatile BVHPackedNodeHalf* __restrict__ lowers,
    volatile BVHPackedNodeHalf* __restrict__ uppers)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;

    const int internal_offset = n;
    for (;;) {
        int left = range_lefts[index];
        int right = range_rights[index];

        if (left == 0 && right == n - 1) {
            *root = index;
            parents[index] = -1;
            break;
        }

        int childCount = 0;
        int parent;
        bool parent_right = false;

        if (left == 0) {
            parent_right = true;
        } else if (right == n - 1) {
            parent_right = false;
        } else {
            if (deltas[right] > deltas[left - 1]) {
                parent_right = true;
            } else if (deltas[right] < deltas[left - 1]) {
                parent_right = false;
            } else {
                // parity tie-break to avoid ladders
                parent_right = (primitive_indices[left - 1] % 2) ^ (primitive_indices[right] % 2);
            }
        }

        if (parent_right) {
            parent = right + internal_offset;
            parents[index] = parent;
            lowers[parent].i = index;
            range_lefts[parent] = left;
            __threadfence();
            childCount = atomicAdd(&num_children[parent], 1);
        } else {
            parent = left + internal_offset - 1;
            parents[index] = parent;
            uppers[parent].i = index;
            range_rights[parent] = right;
            __threadfence();
            childCount = atomicAdd(&num_children[parent], 1);
        }

        if (childCount == 1) {
            int left_child = lowers[parent].i;
            int right_child = uppers[parent].i;

            float3 left_lower = make_float3(lowers[left_child].x, lowers[left_child].y, lowers[left_child].z);
            float3 left_upper = make_float3(uppers[left_child].x, uppers[left_child].y, uppers[left_child].z);
            float3 right_lower = make_float3(lowers[right_child].x, lowers[right_child].y, lowers[right_child].z);
            float3 right_upper = make_float3(uppers[right_child].x, uppers[right_child].y, uppers[right_child].z);

            float3 lower = fmin3(left_lower, right_lower);
            float3 upper = fmax3(left_upper, right_upper);

            make_node(&lowers[parent], lower, left_child, false);
            make_node(&uppers[parent], upper, right_child, false);
            index = parent;
        } else {
            break;
        }
    }
}

__global__ void mark_packed_leaf_nodes(
    int max_nodes,
    const int* __restrict__ range_lefts,
    const int* __restrict__ range_rights,
    const int* __restrict__ parents,
    BVHPackedNodeHalf* __restrict__ lowers,
    BVHPackedNodeHalf* __restrict__ uppers,
    int leaf_size)
{
    int node_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (node_index >= max_nodes) return;

    int left = range_lefts[node_index];
    int right = range_rights[node_index] + 1;  // convert inclusive right to exclusive
    int range_size = right - left;

    if (range_size > leaf_size && range_size > 0) return;

    // depth check
    int depth = 1;
    int parent = parents[node_index];
    while (parent != -1) {
        parent = parents[parent];
        depth++;
    }

    if (range_size <= leaf_size || depth >= BVH_QUERY_STACK_SIZE) {
        lowers[node_index].b = 1;
        lowers[node_index].i = left;
        uppers[node_index].i = right;
    }
}

__global__ void compute_total_bounds(
    const float3* item_lowers, const float3* item_uppers, float3* total_lower, float3* total_upper, int num_items)
{
    typedef cub::BlockReduce<float3, 256> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int numValid = min(num_items - blockIdx.x * blockDim.x, 256);

    float3 lower = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 upper = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    if (tid < num_items) {
        lower = item_lowers[tid];
        upper = item_uppers[tid];
    }

    float3 block_upper = BlockReduce(temp_storage).Reduce(upper, fmax3, numValid);
    __syncthreads();
    float3 block_lower = BlockReduce(temp_storage).Reduce(lower, fmin3, numValid);

    if (threadIdx.x == 0) {
        atomicMinFloat(&total_lower->x, block_lower.x);
        atomicMinFloat(&total_lower->y, block_lower.y);
        atomicMinFloat(&total_lower->z, block_lower.z);
        atomicMaxFloat(&total_upper->x, block_upper.x);
        atomicMaxFloat(&total_upper->y, block_upper.y);
        atomicMaxFloat(&total_upper->z, block_upper.z);
    }
}

__global__ void compute_total_inv_edges(const float3* total_lower, const float3* total_upper, float3* total_inv_edges) {
    float3 edges = *total_upper - *total_lower;
    edges += make_float3(0.0001f, 0.0001f, 0.0001f);
    *total_inv_edges = make_float3(1.0f / edges.x, 1.0f / edges.y, 1.0f / edges.z);
}

// -----------------------------------------------------------------------------
// LinearBVHBuilderGPU (internal)
// -----------------------------------------------------------------------------
class LinearBVHBuilderGPU {
public:
    void build(cudaStream_t stream, BVH& bvh, const float3* item_lowers, const float3* item_uppers, int num_items) {
        int max_nodes = 2 * num_items - 1;

        // Allocate temporary buffers
        int* indices;
        uint64_t* keys;
        int* deltas;
        int* range_lefts;
        int* range_rights;
        int* num_children;
        float3* total_lower;
        float3* total_upper;
        float3* total_inv_edges;

        check_cuda(cudaMalloc(&indices, sizeof(int) * num_items * 2)); // *2 for radix sort double buffer
        check_cuda(cudaMalloc(&keys, sizeof(uint64_t) * num_items * 2));
        check_cuda(cudaMalloc(&deltas, sizeof(int) * num_items));
        check_cuda(cudaMalloc(&range_lefts, sizeof(int) * max_nodes));
        check_cuda(cudaMalloc(&range_rights, sizeof(int) * max_nodes));
        check_cuda(cudaMalloc(&num_children, sizeof(int) * max_nodes));
        check_cuda(cudaMalloc(&total_lower, sizeof(float3)));
        check_cuda(cudaMalloc(&total_upper, sizeof(float3)));
        check_cuda(cudaMalloc(&total_inv_edges, sizeof(float3)));
        
        check_cuda(cudaMemsetAsync(range_lefts, 0, sizeof(int) * max_nodes, stream));
        check_cuda(cudaMemsetAsync(range_rights, 0, sizeof(int) * max_nodes, stream));

        // Initialize total bounds: lower = +FLT_MAX, upper = -FLT_MAX
        // float init_max = FLT_MAX;
        // float init_neg_max = -FLT_MAX;
        check_cuda(cudaMemsetAsync(total_lower, 0x7f, sizeof(float3), stream)); // will set to positive large (approx)
        // Actually memset with 0x7f7f7f7f gives FLT_MAX, for all bytes.
        // We can use kernel memset_kernel, or simple cudaMemset.
        // cudaMemset sets byte pattern, not float directly. Better use kernel:
        {
            // int block = (sizeof(float3) + 127) / 128;
            memset_kernel<<<1, 128, 0, stream>>>((int*)total_lower, 0x7f7fffff, 3);
            memset_kernel<<<1, 128, 0, stream>>>((int*)total_upper, 0xff7fffff, 3);
        }

        // Compute total bounds
        {
            int threads = 256;
            int blocks = (num_items + threads - 1) / threads;
            compute_total_bounds<<<blocks, threads, 0, stream>>>(
                item_lowers, item_uppers, total_lower, total_upper, num_items);
        }

        // Compute inverse edges
        compute_total_inv_edges<<<1, 1, 0, stream>>>(total_lower, total_upper, total_inv_edges);

        // Morton codes
        {
            int threads = 256;
            int blocks = (num_items + threads - 1) / threads;
            compute_morton_codes<<<blocks, threads, 0, stream>>>(
                item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys);
        }

        // Sort
        radix_sort_pairs(stream, keys, indices, num_items);
        // Copy sorted indices to primitive_indices
        check_cuda(cudaMemcpyAsync(bvh.primitive_indices, indices, sizeof(int) * num_items,
                                   cudaMemcpyDeviceToDevice, stream));

        // Deltas
        {
            int threads = 256;
            int blocks = (num_items + threads - 1) / threads;
            compute_key_deltas<<<blocks, threads, 0, stream>>>(keys, deltas, num_items - 1);
        }

        // Build leaves
        {
            int threads = 256;
            int blocks = (num_items + threads - 1) / threads;
            build_leaves<<<blocks, threads, 0, stream>>>(
                item_lowers, item_uppers, num_items, indices, range_lefts, range_rights,
                bvh.node_lowers, bvh.node_uppers);
        }

        // Reset child counters
        check_cuda(cudaMemsetAsync(num_children, 0, sizeof(int) * max_nodes, stream));

        // Build hierarchy
        {
            int threads = 256;
            int blocks = (num_items + threads - 1) / threads;
            build_hierarchy<<<blocks, threads, 0, stream>>>(
                num_items, bvh.root, deltas, keys, num_children, bvh.primitive_indices,
                range_lefts, range_rights, bvh.node_parents, bvh.node_lowers, bvh.node_uppers);
        }

        // Mark packed leaf nodes
        {
            int threads = 256;
            int blocks = (max_nodes + threads - 1) / threads;
            mark_packed_leaf_nodes<<<blocks, threads, 0, stream>>>(
                max_nodes, range_lefts, range_rights, bvh.node_parents, bvh.node_lowers, bvh.node_uppers,
                bvh.leaf_size);
        }

        // Clean up temporary memory
        check_cuda(cudaFree(indices));
        check_cuda(cudaFree(keys));
        check_cuda(cudaFree(deltas));
        check_cuda(cudaFree(range_lefts));
        check_cuda(cudaFree(range_rights));
        check_cuda(cudaFree(num_children));
        check_cuda(cudaFree(total_lower));
        check_cuda(cudaFree(total_upper));
        check_cuda(cudaFree(total_inv_edges));
    }
};

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
void bvh_build(cudaStream_t stream, const float3* d_lowers, const float3* d_uppers, int num_items, int leaf_size, BVH& bvh) {
    // Initialize BVH descriptor
    bvh.num_items = num_items;
    bvh.leaf_size = leaf_size;
    bvh.max_nodes = 2 * num_items - 1;
    bvh.num_leaf_nodes = num_items; // approximate, will be updated after mark_packed_leaf_nodes

    // Allocate BVH internal storage
    check_cuda(cudaMalloc(&bvh.node_lowers, sizeof(BVHPackedNodeHalf) * bvh.max_nodes));
    check_cuda(cudaMalloc(&bvh.node_uppers, sizeof(BVHPackedNodeHalf) * bvh.max_nodes));
    check_cuda(cudaMalloc(&bvh.node_parents, sizeof(int) * bvh.max_nodes));
    check_cuda(cudaMalloc(&bvh.node_counts, sizeof(int) * bvh.max_nodes));
    check_cuda(cudaMalloc(&bvh.root, sizeof(int)));
    check_cuda(cudaMalloc(&bvh.primitive_indices, sizeof(int) * num_items));

    // Zero out nodes
    check_cuda(cudaMemsetAsync(bvh.node_lowers, 0, sizeof(BVHPackedNodeHalf) * bvh.max_nodes, stream));
    check_cuda(cudaMemsetAsync(bvh.node_uppers, 0, sizeof(BVHPackedNodeHalf) * bvh.max_nodes, stream));

    bvh.item_lowers = d_lowers;
    bvh.item_uppers = d_uppers;

    LinearBVHBuilderGPU builder;
    builder.build(stream, bvh, d_lowers, d_uppers, num_items);
}

void bvh_destroy(cudaStream_t stream, BVH& bvh) {
    if (bvh.node_lowers)      { check_cuda(cudaFree(bvh.node_lowers)); }
    if (bvh.node_uppers)      { check_cuda(cudaFree(bvh.node_uppers)); }
    if (bvh.node_parents)     { check_cuda(cudaFree(bvh.node_parents)); }
    if (bvh.node_counts)      { check_cuda(cudaFree(bvh.node_counts)); }
    if (bvh.primitive_indices){ check_cuda(cudaFree(bvh.primitive_indices)); }
    if (bvh.root)             { check_cuda(cudaFree(bvh.root)); }
    bvh = BVH(); // reset to default
}

__global__ void bvh_refit_kernel(
    int n,
    const int* __restrict__ parents,
    int* __restrict__ child_count,
    const int* __restrict__ primitive_indices,
    BVHPackedNodeHalf* __restrict__ node_lowers,
    BVHPackedNodeHalf* __restrict__ node_uppers,
    const float3* __restrict__ item_lowers,
    const float3* __restrict__ item_uppers)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= n) return;

    bool leaf = node_lowers[index].b;
    int parent = parents[index];

    if (leaf) {
        BVHPackedNodeHalf& lower = node_lowers[index];
        BVHPackedNodeHalf& upper = node_uppers[index];
        if (parent == -1 || !node_lowers[parent].b) {
            int start = lower.i;
            int end = upper.i;
            Bounds3 bound;
            bound.lower = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
            bound.upper = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for (int i = start; i < end; ++i) {
                int prim = primitive_indices[i];
                bound.add_bounds(item_lowers[prim], item_uppers[prim]);
            }
            lower.x = bound.lower.x; lower.y = bound.lower.y; lower.z = bound.lower.z;
            upper.x = bound.upper.x; upper.y = bound.upper.y; upper.z = bound.upper.z;
        }
    } else {
        return;
    }

    for (;;) {
        parent = parents[index];
        if (parent == -1) return;
        __threadfence();
        int finished = atomicAdd(&child_count[parent], 1);

        if (finished == 1) {
            BVHPackedNodeHalf& parent_lower = node_lowers[parent];
            BVHPackedNodeHalf& parent_upper = node_uppers[parent];
            if (parent_lower.b) {
                int pp = parents[parent];
                if (pp == -1 || !node_lowers[pp].b) {
                    int start = parent_lower.i;
                    int end = parent_upper.i;
                    Bounds3 bound;
                    bound.lower = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
                    bound.upper = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
                    for (int i = start; i < end; ++i) {
                        int prim = primitive_indices[i];
                        bound.add_bounds(item_lowers[prim], item_uppers[prim]);
                    }
                    parent_lower.x = bound.lower.x; parent_lower.y = bound.lower.y; parent_lower.z = bound.lower.z;
                    parent_upper.x = bound.upper.x; parent_upper.y = bound.upper.y; parent_upper.z = bound.upper.z;
                }
            } else {
                int left = parent_lower.i;
                int right = parent_upper.i;
                float3 ll = make_float3(node_lowers[left].x, node_lowers[left].y, node_lowers[left].z);
                float3 lu = make_float3(node_uppers[left].x, node_uppers[left].y, node_uppers[left].z);
                float3 rl = make_float3(node_lowers[right].x, node_lowers[right].y, node_lowers[right].z);
                float3 ru = make_float3(node_uppers[right].x, node_uppers[right].y, node_uppers[right].z);
                float3 lower = fmin3(ll, rl);
                float3 upper = fmax3(lu, ru);
                parent_lower.x = lower.x; parent_lower.y = lower.y; parent_lower.z = lower.z;
                parent_upper.x = upper.x; parent_upper.y = upper.y; parent_upper.z = upper.z;
            }
            index = parent;
        } else {
            break;
        }
    }
}

void bvh_refit(cudaStream_t stream, BVH& bvh) {
    check_cuda(cudaMemsetAsync(bvh.node_counts, 0, sizeof(int) * bvh.max_nodes, stream));
    int threads = 256;
    int blocks = (bvh.num_leaf_nodes + threads - 1) / threads;
    bvh_refit_kernel<<<blocks, threads, 0, stream>>>(
        bvh.num_leaf_nodes, bvh.node_parents, bvh.node_counts, bvh.primitive_indices,
        bvh.node_lowers, bvh.node_uppers, bvh.item_lowers, bvh.item_uppers);
}

void bvh_rebuild(cudaStream_t stream, BVH& bvh) {
    // reuse existing BVH memory but rebuild structure
    // we need to keep the item bounds pointers
    LinearBVHBuilderGPU builder;
    builder.build(stream, bvh, bvh.item_lowers, bvh.item_uppers, bvh.num_items);
    // leaf_size remains unchanged
}

} // namespace lbvh3d