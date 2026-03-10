#pragma once

#if defined(__INTELLISENSE__) || defined(__RESHARPER__) || defined(__JETBRAINS_IDE__)

#if !defined(__CUDACC__)
#define __CUDACC__
#endif

#include<device_launch_parameters.h>
#include<device_functions.h>
#include<math_functions.h>
#include<device_atomic_functions.h>
#include<cuda/atomic>
#endif // __INTELLISENSE__ || __RESHARPER__ || __JETBRAINS_IDE__

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

inline void __cudaCheckError(cudaError_t err, const char* file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); //
        fprintf(stderr, "File: %s\n", file);                          //
        fprintf(stderr, "Line: %d\n", line);                          //

        exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) do { \
    __cudaCheckError((call), __FILE__, __LINE__); \
} while (0)
 
