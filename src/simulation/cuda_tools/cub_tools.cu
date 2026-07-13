#include "cub_tools.cuh"

#include "common/cuda_utils.h"

void* CubTempCache::allocate(cudaStream_t stream, size_t need_size) {
    // std::lock_guard<std::mutex> lock(mutex_); // Currently, there is no demand for multithreading.
    Buffer& buf = cache_[stream];
    if (need_size > buf.size) {
        if (buf.ptr) cudaFree(buf.ptr);
        cudaMalloc(&buf.ptr, need_size);
        buf.size = need_size;
    }
    return buf.ptr;
}

CubTempCache::~CubTempCache() {
    if ( !cuda_device_valid() ) return;
    for (auto& [stream, buf] : cache_) {
        if (buf.ptr) cudaFree(buf.ptr);
    }
}

CubTempCache g_cub_temp_cache;