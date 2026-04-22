// #include <cuda_runtime.h>
//
#include "common/cuda_utils.h"

// #include <cuda_runtime.h>
#include <vector>
#include "common/device.h"
#include "graphics/graphics.h"

// inline int cutGetMaxGflopsDeviceId() {
//     int current_device = 0, sm_per_multiproc = 0;
//     int max_compute_perf = 0, max_perf_device = 0;
//     int device_count = 0, best_SM_arch = 0;
//     int arch_cores_sm[3] = { 1, 8, 32 };
//     cudaDeviceProp deviceProp;
//
//     cudaGetDeviceCount(&device_count);
//     // Find the best major SM Architecture GPU device
//     while ( current_device < device_count ) {
//         cudaGetDeviceProperties(&deviceProp, current_device);
//         if ( deviceProp.major > 0 && deviceProp.major < 9999 ) {
//             if ( deviceProp.major > best_SM_arch )
//                 best_SM_arch = deviceProp.major;
//         }
//         current_device++;
//     }
//
//     // Find the best CUDA capable GPU device
//     current_device = 0;
//     while ( current_device < device_count ) {
//         cudaGetDeviceProperties(&deviceProp, current_device);
//         if ( deviceProp.major == 9999 && deviceProp.minor == 9999 ) {
//             sm_per_multiproc = 1;
//         }
//         else if ( deviceProp.major <= 2 ) {
//             sm_per_multiproc = arch_cores_sm[deviceProp.major];
//         }
//         else {
//             sm_per_multiproc = arch_cores_sm[2];
//         }
//
//         int compute_perf = deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;
//         if ( compute_perf > max_compute_perf ) {
//             // If we find GPU with SM major > 2, search only these
//             if ( best_SM_arch > 2 ) {
//                 // If our device==best_SM_arch, choose this, or else pass
//                 if ( deviceProp.major == best_SM_arch ) {
//                     max_compute_perf = compute_perf;
//                     max_perf_device = current_device;
//                 }
//             }
//             else {
//                 max_compute_perf = compute_perf;
//                 max_perf_device = current_device;
//             }
//         }
//         ++current_device;
//     }
//     return max_perf_device;
// }

inline int ConvertSMVer2Cores(int major, int minor) {
    // 基于 CUDA 编程指南的 CUDA 核心/SM 对照表
    switch ( major ) {
    case 1:
        return 8;
    case 2:
        return (minor == 0) ? 32 : 48;
    case 3:
        return 192;
    case 5:
        return 128;
    case 6:
        return (minor == 0) ? 64 : 128;
    case 7:
        return (minor == 0) ? 64 : 128;
    case 8:
        return (minor == 0) ? 64 : 128;
    case 9:
        return 128;
    default:
        return 128; // 未知架构，保守估计
    }
}
inline int cutGetMaxGflopsDeviceId() {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if ( err != cudaSuccess || device_count == 0 ) {
        return -1;
    }

    int best_SM_arch = 0;
    int max_perf_device = 0;
    int max_compute_perf = 0;

    // 第一轮：找到最高 SM 架构
    for ( int i = 0; i < device_count; ++i ) {
        cudaDeviceProp prop;
        if ( cudaGetDeviceProperties(&prop, i) != cudaSuccess ) continue;
        if ( prop.major > 0 && prop.major < 9999 && prop.major > best_SM_arch ) {
            best_SM_arch = prop.major;
        }
    }

    // 第二轮：在最高架构（或所有架构）中选算力最高的
    for ( int i = 0; i < device_count; ++i ) {
        cudaDeviceProp prop;
        if ( cudaGetDeviceProperties(&prop, i) != cudaSuccess ) continue;
        if ( prop.major == 9999 && prop.minor == 9999 ) continue; // 模拟设备跳过

        int sm_per_multiproc = ConvertSMVer2Cores(prop.major, prop.minor);
        int compute_perf = prop.multiProcessorCount * sm_per_multiproc * prop.clockRate;

        if ( best_SM_arch >= 3 ) {
            // 优先选择最高架构的设备
            if ( prop.major == best_SM_arch && compute_perf > max_compute_perf ) {
                max_compute_perf = compute_perf;
                max_perf_device = i;
            }
        }
        else {
            if ( compute_perf > max_compute_perf ) {
                max_compute_perf = compute_perf;
                max_perf_device = i;
            }
        }
    }
    return max_perf_device;
}


inline bool is_cuda_context_valid(int device_id) {
    cudaError_t err = cudaSetDevice(device_id);
    if ( err != cudaSuccess ) return false;

    err = cudaFree(nullptr);
    return (err == cudaSuccess);
}
static int init_cuda_device() {
    int existing_device = -1;
    cudaError_t err = cudaGetDevice(&existing_device);
    if ( err == cudaSuccess && existing_device >= 0 ) {
        if ( is_cuda_context_valid(existing_device) ) {
            printf("[Qianyi] Reusing active CUDA device: %d\n", existing_device);
        }
    }
    const int deviceIdx = cutGetMaxGflopsDeviceId();
    CUDA_CHECK(cudaSetDevice( deviceIdx ));
    // CUDA_CHECK(cudaDeviceReset());

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, deviceIdx));
    return deviceIdx;
}
bool device_initialized = false;
int active_device_id = -1;
void init_device() {
    if ( device_initialized ) return;
    auto device_id = init_cuda_device();
    active_device_id = device_id;
    graphics::VulkanCudaRasterizer::Init(device_id);
    graphics::VulkanCudaRasterizer::SetResourceDirectory(module_dir);
    device_initialized = true;
}
