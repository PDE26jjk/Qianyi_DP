#pragma once

#include <cuda_runtime.h>
#include <string>
#include <vector>

namespace graphics {

struct Point2D {
    float x, y;
};
struct Edge {
    Point2D v1, v2;
};

float* render_edges_to_cuda(const std::vector<Edge>& edges, cudaStream_t stream = 0);
class VulkanCudaRasterizer {
    VulkanCudaRasterizer();
    ~VulkanCudaRasterizer();
public:
    // 初始化 Vulkan 全局环境 (只需调用一次)
    // cudaDeviceId: 对应的 CUDA 设备索引，用于 Vulkan-CUDA 互操作
    static void Init(int cudaDeviceId);
    static VulkanCudaRasterizer& Instance();
    static void SetResourceDirectory(const std::string& path);

    // 清理全局环境
    static void Cleanup();


    // 渲染一组边并返回 CUDA 可用的显存指针
    // width, height: 动态分辨率，函数内部会自动处理资源的销毁与重建
    // stream: CUDA 流，用于异步同步
    // 返回: 指向 R32_SFLOAT 格式图像数据的 GPU 指针
    float* render(const std::vector<Edge>& edges, int width, int height, cudaStream_t stream = 0);

private:
    
    struct Impl;
    Impl* pImpl;
};
}
