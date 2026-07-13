#pragma once

#include <unordered_map>
#include <mutex>
#include <cuda_runtime.h>

class CubTempCache {
public:
    struct Buffer {
        void* ptr = nullptr;
        size_t size = 0;
    };

    void* allocate(cudaStream_t stream, size_t need_size);

    ~CubTempCache();

private:
    std::unordered_map<cudaStream_t, Buffer> cache_;
    std::mutex mutex_;
};
extern CubTempCache g_cub_temp_cache;

inline void* get_device_temp_memory(cudaStream_t stream, size_t size) {
    return g_cub_temp_cache.allocate(stream, size);
}

#ifndef CALL_CUBS
#ifdef _WIN32
#define CALL_CUBS(func, ...)                                               \
    do {                                                                   \
        size_t temp_mem_size = 0;                                          \
        CUDA_CHECK(cub::func(nullptr, temp_mem_size, __VA_ARGS__));        \
        void* temp_mem = get_device_temp_memory((cudaStream_t)0,           \
                                                temp_mem_size);            \
        CUDA_CHECK(cub::func(temp_mem, temp_mem_size, __VA_ARGS__));       \
    } while(0)
#else
#define CALL_CUBS(func, args...)                                           \
    do {                                                                   \
        size_t temp_mem_size = 0;                                          \
        CUDA_CHECK(cub::func(nullptr, temp_mem_size, args));               \
        void* temp_mem = get_device_temp_memory((cudaStream_t)0,           \
                                                temp_mem_size);            \
        CUDA_CHECK(cub::func(temp_mem, temp_mem_size, args));              \
    } while(0)
#endif
#endif

#ifndef CALL_CUBS_STREAM
#ifdef _WIN32
#define CALL_CUBS_STREAM(stream, func, ...)                                \
    do {                                                                   \
        size_t temp_mem_size = 0;                                          \
        CUDA_CHECK(cub::func(nullptr, temp_mem_size, __VA_ARGS__));        \
        void* temp_mem = get_device_temp_memory((cudaStream_t)(stream),    \
                                                temp_mem_size);            \
        CUDA_CHECK(cub::func(temp_mem, temp_mem_size, __VA_ARGS__));       \
    } while(0)
#else
#define CALL_CUBS_STREAM(stream, func, args...)                            \
    do {                                                                   \
        size_t temp_mem_size = 0;                                          \
        CUDA_CHECK(cub::func(nullptr, temp_mem_size, args));               \
        void* temp_mem = get_device_temp_memory((cudaStream_t)(stream),    \
                                                temp_mem_size);            \
        CUDA_CHECK(cub::func(temp_mem, temp_mem_size, args));              \
    } while(0)
#endif
#endif