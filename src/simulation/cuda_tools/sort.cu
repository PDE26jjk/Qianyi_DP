// Modifications Copyright (c) 2026 PDE26jjk
// Adapted from NVIDIA Warp — original license below
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "common/cuda_utils.h"
#include "sort.h"

#define THRUST_IGNORE_CUB_VERSION_CHECK

#include <unordered_map>

#include <cub/cub.cuh>

// temporary buffer for radix sort
struct RadixSortTemp {
    void* mem = NULL;
    size_t size = 0;
};

// use unique temp buffers per CUDA stream to avoid race conditions
static std::unordered_map<void*, RadixSortTemp> g_radix_sort_temp_map;


template <typename KeyType>
void radix_sort_reserve_internal(void* stream, int n, void** mem_out, size_t* size_out)
{
    cub::DoubleBuffer<KeyType> d_keys;
    cub::DoubleBuffer<int> d_values;

    // compute temporary memory required
    size_t sort_temp_size;
    CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            NULL, sort_temp_size, d_keys, d_values, n, 0, sizeof(KeyType) * 8, (cudaStream_t)stream)
    );

    RadixSortTemp& temp = g_radix_sort_temp_map[stream];

    if (sort_temp_size > temp.size) {
        if (temp.mem) {
            CUDA_CHECK(cudaFree(temp.mem));
        }
        CUDA_CHECK(cudaMalloc(&temp.mem, sort_temp_size));
        temp.size = sort_temp_size;
    }

    if (mem_out)
        *mem_out = temp.mem;
    if (size_out)
        *size_out = temp.size;
}

void radix_sort_reserve(void* stream, int n, void** mem_out, size_t* size_out)
{
    radix_sort_reserve_internal<int>(stream, n, mem_out, size_out);
}

void radix_sort_release(void* stream)
{
    // release temporary buffer for the given stream, if it exists
    auto it = g_radix_sort_temp_map.find(stream);
    if (it != g_radix_sort_temp_map.end()) {
        CUDA_CHECK(cudaFree(it->second.mem));
        g_radix_sort_temp_map.erase(it);
    }
}

template <typename KeyType, class ValueType>
void radix_sort_pairs(void* stream, KeyType* keys, ValueType* values, int n)
{
    cub::DoubleBuffer<KeyType> d_keys(keys, keys + n);
    cub::DoubleBuffer<ValueType> d_values(values, values + n);

    RadixSortTemp temp;
    radix_sort_reserve_internal<KeyType>(stream, n, &temp.mem, &temp.size);

    // sort
    CUDA_CHECK(
        cub::DeviceRadixSort::SortPairs(
            temp.mem, temp.size, d_keys, d_values, n, 0, sizeof(KeyType) * 8, (cudaStream_t)stream)
    );

    if (d_keys.Current() != keys)
        CUDA_CHECK(cudaMemcpyAsync(keys, d_keys.Current(), sizeof(KeyType) * n, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));

    if (d_values.Current() != values)
        CUDA_CHECK(cudaMemcpyAsync(values, d_values.Current(), sizeof(ValueType) * n, cudaMemcpyDeviceToDevice, (cudaStream_t)stream));
}

void radix_sort_pairs(void* stream, int* keys, int* values, int n)
{
    radix_sort_pairs<int>(stream, keys, values, n);
}

void radix_sort_pairs(void* stream,unsigned int* keys,unsigned int* values, int n)
{
    radix_sort_pairs<unsigned int>(stream, keys, values, n);
}

void radix_sort_pairs(void* stream, float* keys, int* values, int n)
{
    radix_sort_pairs<float>(stream, keys, values, n);
}

void radix_sort_pairs(void* stream, int64_t* keys, int* values, int n)
{
    radix_sort_pairs<int64_t>(stream, keys, values, n);
}

void radix_sort_pairs(void* stream, uint64_t* keys, int* values, int n)
{
    radix_sort_pairs<uint64_t>(stream, keys, values, n);
}