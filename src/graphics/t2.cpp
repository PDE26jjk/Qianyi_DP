#include <vector>
#include <array>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <span>
#include <vulkan/vulkan.h>
#include <cuda_runtime.h>

#include "graphics.h"
// --- 跨平台宏定义 ---
#ifdef _WIN32
	#include <windows.h>
	#include <vulkan/vulkan_win32.h>
	#define VK_EXT_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
	#define VK_EXT_SEM_HANDLE_TYPE VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
typedef HANDLE NativeHandle;
#else
	#include <unistd.h>
	#include <vulkan/vulkan_linux.h> // 或对应平台头文件
	#define VK_EXT_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
	#define VK_EXT_SEM_HANDLE_TYPE VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
	typedef int NativeHandle;
#endif

#define VK_CHECK(call) \
	do { \
	    VkResult result = call; \
	    if (result != VK_SUCCESS) { \
	        throw std::runtime_error("Vulkan error: " #call); \
	    } \
	} while(0)

namespace graphics {
// 全局句柄
VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkDevice device = VK_NULL_HANDLE;
VkQueue graphicsQueue = VK_NULL_HANDLE;
uint32_t queueFamilyIndex = 0;
void init(const std::vector<uint8_t>& device_uuid) { // same as cuda
    // 1. 创建 Instance (Vulkan 1.3)
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_3 // 使用 1.3 基线
    };
    std::vector<const char*> instanceExtensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
    };
    VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size()),
        .ppEnabledExtensionNames = instanceExtensions.data()
    };
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
    // 2. 查找匹配 UUID 的物理设备
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    for ( const auto& dev : devices ) {
        VkPhysicalDeviceIDProperties idProps{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &idProps };
        vkGetPhysicalDeviceProperties2(dev, &props2);
        if ( memcmp(idProps.deviceUUID, device_uuid.data(), VK_UUID_SIZE) == 0 ) {
            physicalDevice = dev;
            break;
        }
    }
    if ( !physicalDevice ) throw std::runtime_error("Failed to find Vulkan device matching UUID");
    // 3. 查找 Graphics Queue
    uint32_t queueCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueCount, queueFamilies.data());
    bool found = false;
    for ( uint32_t i = 0; i < queueCount; ++i ) {
        if ( queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ) {
            queueFamilyIndex = i;
            found = true;
            break;
        }
    }
    if ( !found ) throw std::runtime_error("No graphics queue found");
    // 4. 创建 Device (启用 Vulkan 1.3 特性)
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = queueFamilyIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority
    };
    // 启用动态渲染和同步2
    VkPhysicalDeviceVulkan13Features vk13Features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .dynamicRendering = VK_TRUE,
        .synchronization2 = VK_TRUE
    };
    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
	    #ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
	    #endif
    };
    VkDeviceCreateInfo deviceInfo{
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = &vk13Features, // 链接特性
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queueInfo,
        .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
        .ppEnabledExtensionNames = deviceExtensions.data()
    };
    VK_CHECK(vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device));
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &graphicsQueue);
}
}


struct WindingVertex {
    float x, y;
    int32_t weight;
};
class VulkanCudaRasterizer {
private:
    VkDevice device;
    VkPhysicalDevice physicalDevice;
    VkQueue graphicsQueue;
    uint32_t queueFamilyIndex;
    int width, height;
    // Vulkan 资源
    VkImage sharedImage = VK_NULL_HANDLE;
    VkDeviceMemory sharedImageMemory = VK_NULL_HANDLE;
    VkImageView sharedImageView = VK_NULL_HANDLE;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkDeviceMemory vertexBufferMemory = VK_NULL_HANDLE;
    void* vertexMappedData = nullptr;
    size_t maxVertexCount = 100000;
    // 管线
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline graphicsPipeline = VK_NULL_HANDLE;
    // 命令与同步
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence renderFence = VK_NULL_HANDLE; // 用于 CPU 等待 GPU
    VkSemaphore vkRenderCompleteSem = VK_NULL_HANDLE;
    NativeHandle extMemHandle = {};
    NativeHandle extSemHandle = {};
    // CUDA 资源
    cudaExternalMemory_t cudaExtMem;
    cudaExternalSemaphore_t cudaExtSem;
    float* d_cuda_grid_ptr = nullptr; // 注意：改为 float* 以匹配 R32_SFLOAT
public:
    VulkanCudaRasterizer(VkPhysicalDevice pDev, VkDevice dev, VkQueue queue, uint32_t qFam, int w, int h)
    : physicalDevice(pDev), device(dev), graphicsQueue(queue), queueFamilyIndex(qFam), width(w), height(h) {
        create_shared_image();
        create_vertex_buffer();
        create_pipeline_dynamic();
        create_sync_objects();
        map_resources_to_cuda();
    }
    ~VulkanCudaRasterizer() {
        if ( d_cuda_grid_ptr ) cudaFree(d_cuda_grid_ptr);
        cudaDestroyExternalMemory(cudaExtMem);
        cudaDestroyExternalSemaphore(cudaExtSem);
        vkDestroySemaphore(device, vkRenderCompleteSem, nullptr);
        vkDestroyFence(device, renderFence, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyImageView(device, sharedImageView, nullptr);
        vkDestroyImage(device, sharedImage, nullptr);
        vkFreeMemory(device, sharedImageMemory, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    // 返回 float* 指针，CUDA 可读取累加后的浮点值
    float* render_edges_to_cuda(const std::vector<graphics::Edge>& edges, cudaStream_t stream = 0) {
        if ( edges.empty() ) return d_cuda_grid_ptr;
        // 1. 准备顶点数据
        std::vector<WindingVertex> vertices;
        vertices.reserve(edges.size() * 3);
        graphics::Point2D anchor = { 0.0f, 0.0f }; // 假设锚点逻辑
        for ( const auto& edge : edges ) {
            int32_t weight = 1; // 根据您的逻辑调整
            vertices.push_back({ anchor.x, anchor.y, weight });
            vertices.push_back({ edge.v1.x, edge.v1.y, weight });
            vertices.push_back({ edge.v2.x, edge.v2.y, weight });
        }
        size_t dataSize = vertices.size() * sizeof(WindingVertex);
        if ( vertices.size() > maxVertexCount ) throw std::runtime_error("Overflow");
        // 2. CPU 等待 GPU 空闲 (通过 Fence)
        // 比较粗暴的同步，但确保安全。生产环境建议使用 Ring Buffer。
        vkWaitForFences(device, 1, &renderFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &renderFence);
        // 3. 更新顶点
        memcpy(vertexMappedData, vertices.data(), dataSize);
        // 4. 录制并提交命令
        record_command_buffer_dynamic(vertices.size());
        VkSubmitInfo submitInfo{
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &commandBuffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &vkRenderCompleteSem
        };
        VK_CHECK(vkQueueSubmit(graphicsQueue, 1, &submitInfo, renderFence));
        // 5. CUDA 等待 Vulkan 信号量
        cudaExternalSemaphoreWaitParams waitParams{};
        cudaWaitExternalSemaphoresAsync(&cudaExtSem, &waitParams, 1, stream);
        return d_cuda_grid_ptr;
    }

private:
    uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProps);
        for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
            if ((typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type!");
    }
    void create_sync_objects() {
        // 1. Fence: 用于 CPU 等待 GPU 完成
        VkFenceCreateInfo fenceInfo {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT // 初始为信号状态，避免第一次等待死锁
        };
        VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &renderFence));
 
        // 2. Semaphore: 用于 Vulkan 通知 CUDA，需要导出
        VkExportSemaphoreCreateInfo exportInfo {
            .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXT_SEM_HANDLE_TYPE
        };
        VkSemaphoreCreateInfo semInfo {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
            .pNext = &exportInfo
        };
        VK_CHECK(vkCreateSemaphore(device, &semInfo, nullptr, &vkRenderCompleteSem));
 
        // 3. 获取 Semaphore 的原生句柄给 CUDA
#ifdef _WIN32
        auto pfnGetHandle = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
        VkSemaphoreGetWin32HandleInfoKHR handleInfo { 
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR, 
            .semaphore = vkRenderCompleteSem, 
            .handleType = VK_EXT_SEM_HANDLE_TYPE 
        };
        pfnGetHandle(device, &handleInfo, &extSemHandle);
#else
        auto pfnGetHandle = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
        VkSemaphoreGetFdInfoKHR handleInfo { 
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, 
            .semaphore = vkRenderCompleteSem, 
            .handleType = VK_EXT_SEM_HANDLE_TYPE 
        };
        pfnGetHandle(device, &handleInfo, &extSemHandle);
#endif
    }
    void create_vertex_buffer() {
        VkDeviceSize bufferSize = maxVertexCount * sizeof(WindingVertex);
        
        VkBufferCreateInfo bufferInfo {
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        VK_CHECK(vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer));
 
        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memReqs);
 
        VkMemoryAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, 
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        };
        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory));
        VK_CHECK(vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0));
        VK_CHECK(vkMapMemory(device, vertexBufferMemory, 0, bufferSize, 0, &vertexMappedData));
    }
 
    // --- 创建命令缓冲池 ---
    void create_command_buffer() {
        VkCommandPoolCreateInfo poolInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queueFamilyIndex
        };
        VK_CHECK(vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool));
 
        VkCommandBufferAllocateInfo allocInfo {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = commandPool,
            .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1
        };
        VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer));
    }
 
    // --- 映射资源到 CUDA ---
    void map_resources_to_cuda() {
        // 1. 映射 Image 内存
        cudaExternalMemoryHandleDesc memDesc {};
#ifdef _WIN32
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memDesc.handle.win32.handle = extMemHandle;
#else
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memDesc.handle.fd = extMemHandle;
#endif
        memDesc.size = width * height * sizeof(float); // R32_SFLOAT = 4 bytes
 
        cudaImportExternalMemory(&cudaExtMem, &memDesc);
 
        cudaExternalMemoryBufferDesc bufferDesc {};
        bufferDesc.offset = 0;
        bufferDesc.size = memDesc.size;
        cudaExternalMemoryGetMappedBuffer((void**)&d_cuda_grid_ptr, cudaExtMem, &bufferDesc);
 
        // 2. 映射 Semaphore
        cudaExternalSemaphoreHandleDesc semDesc {};
#ifdef _WIN32
        semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        semDesc.handle.win32.handle = extSemHandle;
#else
        semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        semDesc.handle.fd = extSemHandle;
#endif
        cudaImportExternalSemaphore(&cudaExtSem, &semDesc);
    }
 
    // --- 文件读取辅助 ---
    static std::vector<char> read_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open shader file: " + filename);
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }
 
    // --- Shader Module 辅助 ---
    VkShaderModule create_shader_module(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.size(),
            .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };
        VkShaderModule module;
        VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &module));
        return module;
    }
    void create_shared_image() {
        // 外部内存信息
        VkExternalMemoryImageCreateInfo extInfo{
            .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .handleTypes = VK_EXT_MEM_HANDLE_TYPE
        };
        // 修正：使用 R32_SFLOAT 以支持混合
        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &extInfo,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R32_SFLOAT,
            .extent = { (uint32_t)width, (uint32_t)height, 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_STORAGE_BIT, // Storage 用于 CUDA 通用访问
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };
        VK_CHECK(vkCreateImage(device, &imageInfo, nullptr, &sharedImage));
        // 内存分配与导出
        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, sharedImage, &memReqs);
        VkExportMemoryAllocateInfo exportAlloc{
            .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXT_MEM_HANDLE_TYPE
        };
        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &exportAlloc,
            .allocationSize = memReqs.size,
            .memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        };
        VK_CHECK(vkAllocateMemory(device, &allocInfo, nullptr, &sharedImageMemory));
        VK_CHECK(vkBindImageMemory(device, sharedImage, sharedImageMemory, 0));
        // 获取句柄
	#ifdef _WIN32
        auto pfnGetHandle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
        VkMemoryGetWin32HandleInfoKHR handleInfo{ .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory = sharedImageMemory, .handleType = VK_EXT_MEM_HANDLE_TYPE };
        pfnGetHandle(device, &handleInfo, &extMemHandle);
	#else
	        auto pfnGetHandle = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
	        VkMemoryGetFdInfoKHR handleInfo { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = sharedImageMemory, .handleType = VK_EXT_MEM_HANDLE_TYPE };
	        pfnGetHandle(device, &handleInfo, &extMemHandle);
	#endif
        // 创建 ImageView
        VkImageViewCreateInfo viewInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = sharedImage,
            .viewType = VK_IMAGE_VIEW_TYPE_2D,
            .format = VK_FORMAT_R32_SFLOAT,
            .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 }
        };
        VK_CHECK(vkCreateImageView(device, &viewInfo, nullptr, &sharedImageView));
    }

    void create_pipeline_dynamic() {
        // 加载着色器
        auto vertCode = read_file("vert.spv");
        auto fragCode = read_file("frag.spv");
        VkShaderModule vertMod = create_shader_module(vertCode);
        VkShaderModule fragMod = create_shader_module(fragCode);
        std::array stages = {
            VkPipelineShaderStageCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertMod, .pName = "main" },
            VkPipelineShaderStageCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragMod, .pName = "main" }
        };
        // 顶点输入
        VkVertexInputBindingDescription binding{ .binding = 0, .stride = sizeof(WindingVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
        std::array attributes = {
            VkVertexInputAttributeDescription{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(WindingVertex, x) },
            VkVertexInputAttributeDescription{ .location = 1, .binding = 0, .format = VK_FORMAT_R32_SINT,
                .offset = offsetof(WindingVertex, weight) }
        };
        VkPipelineVertexInputStateCreateInfo vertexInput{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &binding,
            .vertexAttributeDescriptionCount = (uint32_t)attributes.size(), .pVertexAttributeDescriptions = attributes.data() };
        VkPipelineInputAssemblyStateCreateInfo inputAssembly{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        // 动态状态
        std::array dynamicStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dynamicState{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = (uint32_t)dynamicStates.size(), .pDynamicStates = dynamicStates.data() };
        VkPipelineViewportStateCreateInfo viewportState{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1, .scissorCount = 1 };
        VkPipelineRasterizationStateCreateInfo rasterizer{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f };
        VkPipelineMultisampleStateCreateInfo multisampling{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        // 混合 (修正：浮点加法)
        VkPipelineColorBlendAttachmentState blendAttachment{
            .blendEnable = VK_TRUE,
            .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, // Src: weight (float)
            .dstColorBlendFactor = VK_BLEND_FACTOR_ONE, // Dst: current sum
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD,
            .colorWriteMask = VK_COLOR_COMPONENT_R_BIT
        };
        VkPipelineColorBlendStateCreateInfo colorBlending{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &blendAttachment };
        // Push Constants
        VkPushConstantRange pushRange{ .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .size = sizeof(float) * 2 };
        VkPipelineLayoutCreateInfo layoutInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pushConstantRangeCount = 1, .pPushConstantRanges = &pushRange };
        VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &pipelineLayout));
        // 动态渲染附件信息 (关键：替代 RenderPass)
        VkFormat imgFormat = VK_FORMAT_R32_SFLOAT;
        VkPipelineRenderingCreateInfo renderingInfo{
            .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &imgFormat // 假设 sharedImage 变量名复用了，这里应该用一个变量存储 format
        };

        VkGraphicsPipelineCreateInfo pipelineInfo{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .pNext = &renderingInfo,
            .stageCount = (uint32_t)stages.size(),
            .pStages = stages.data(),
            .pVertexInputState = &vertexInput,
            .pInputAssemblyState = &inputAssembly,
            .pViewportState = &viewportState,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &colorBlending,
            .pDynamicState = &dynamicState,
            .layout = pipelineLayout
        };
        VK_CHECK(vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline));
        vkDestroyShaderModule(device, vertMod, nullptr);
        vkDestroyShaderModule(device, fragMod, nullptr);
    }
    void record_command_buffer_dynamic(size_t vertexCount) {
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);
        // 1. 屏障：UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
        VkImageMemoryBarrier2 barrier1{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
            .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = sharedImage,
            .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 }
        };
        VkDependencyInfo depInfo1{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier1 };
        vkCmdPipelineBarrier2(commandBuffer, &depInfo1);
        // 2. 动态渲染
        VkClearValue clearVal{ .color = { { 0.0f, 0.0f, 0.0f, 0.0f } } };
        VkRenderingAttachmentInfo colorAtt{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
            .imageView = sharedImageView,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = clearVal
        };
        VkRenderingInfo renderingInfo{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { .extent = { (uint32_t)width, (uint32_t)height } },
            .layerCount = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments = &colorAtt
        };
        vkCmdBeginRendering(commandBuffer, &renderingInfo);
        VkViewport vp{ .width = (float)width, .height = (float)height, .maxDepth = 1.0f };
        VkRect2D scissor{ .extent = { (uint32_t)width, (uint32_t)height } };
        vkCmdSetViewport(commandBuffer, 0, 1, &vp);
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        VkDeviceSize offset = 0;
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, &offset);
        float pushData[2] = { (float)width, (float)height };
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(pushData), pushData);
        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertexCount), 1, 0, 0);
        vkCmdEndRendering(commandBuffer);
        // 3. 屏障：COLOR_ATTACHMENT_OPTIMAL -> GENERAL (供 CUDA 使用)
        VkImageMemoryBarrier2 barrier2{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, // 或 ALL_COMMANDS
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_GENERAL, // CUDA 互操作通常期望 GENERAL
            .image = sharedImage,
            .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 }
        };
        VkDependencyInfo depInfo2{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrier2 };
        vkCmdPipelineBarrier2(commandBuffer, &depInfo2);
        vkEndCommandBuffer(commandBuffer);
    }

    // 注意：map_resources_to_cuda 中 d_cuda_grid_ptr 类型应为 float*
};

// int main() {
//     // 1. 获取 CUDA 设备 UUID
//     int cudaDevice = 0;
//     cudaSetDevice(cudaDevice);
//     cudaDeviceProp prop;
//     cudaGetDeviceProperties(&prop, cudaDevice);
//     // CUDA UUID 是 16 字节
//     std::vector<uint8_t> uuid(prop.uuid.bytes, prop.uuid.bytes + 16);
//     // 2. 初始化 Vulkan
//     graphics::init(uuid);
//     // 3. 创建光栅化器 (Width=1024, Height=1024)
//     VulkanCudaRasterizer rasterizer(
//         graphics::physicalDevice,
//         graphics::device,
//         graphics::graphicsQueue,
//         graphics::queueFamilyIndex,
//         1024, 1024
//         );
//     // 4. 准备几何数据
//     std::vector<Edge> edges = {
//         { { 100, 100 }, { 200, 100 } }, // 示例边
//         { { 200, 100 }, { 200, 200 } },
//         { { 200, 200 }, { 100, 100 } }
//     };
//     // 5. 渲染并获取 CUDA 指针
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     // 返回的是 float* 指针，指向 GPU 显存
//     float* d_result = rasterizer.render_edges_to_cuda(edges, stream);
//     // 6. CUDA 后续处理
//     // 此时 d_result 已经在 stream 上就绪，可以运行 kernel
//     // my_cuda_kernel<<<..., stream>>>(d_result);
//     cudaStreamSynchronize(stream);
//     cudaStreamDestroy(stream);
//     return 0;
// }
