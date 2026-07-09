// Modifications Copyright (c) 2026 PDE26jjk
// Adapted from NVIDIA Warp — original license below
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda/std/cstdint>

void radix_sort_reserve(void* stream, int n, void** mem_out = NULL, size_t* size_out = NULL);
void radix_sort_release(void* stream);

void radix_sort_pairs(void* stream, int* keys, int* values, int n);
void radix_sort_pairs(void* stream, float* keys, int* values, int n);
void radix_sort_pairs(void* stream, int64_t* keys, int* values, int n);
void radix_sort_pairs(void* stream, uint64_t* keys, int* values, int n);

void radix_sort_pairs(void* stream, unsigned int* keys,unsigned int* values, int n);
