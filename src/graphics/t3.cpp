#include "graphics.h"
#include <memory>
#include <stdexcept>
#include <iostream>
#include <fstream>
#include <array>
#include <vulkan/vulkan.h>
#include <filesystem>
// ================== 平台宏定义 ==================
#ifdef _WIN32
    #include <windows.h>
    #include <vulkan/vulkan_win32.h>
    #define VK_EXT_MEM_HANDLE_TYPE VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
    #define VK_EXT_SEM_HANDLE_TYPE VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
typedef HANDLE NativeHandle;
#else
    #include <unistd.h>
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

// ================== 内部数据结构 ==================
namespace graphics {
struct WindingVertex {
    float x, y;
    int32_t weight;
};

// ================== 全局上下文 (单例) ==================
namespace GlobalContext {
VkInstance instance = VK_NULL_HANDLE;
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkDevice device = VK_NULL_HANDLE;
VkQueue graphicsQueue = VK_NULL_HANDLE;
uint32_t queueFamilyIndex = 0;
std::filesystem::path shaderDir;

// 新增：设置路径的接口
void setResourceDirectory(const std::string& path) {
    shaderDir = std::filesystem::path(path);
}
void init(int cudaDeviceId) {
    // 1. 获取 CUDA device UUID
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDeviceId);
    std::vector<uint8_t> uuid(prop.uuid.bytes, prop.uuid.bytes + 16);
    // 2. 创建 Instance (Vulkan 1.3)
    VkApplicationInfo appInfo{
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .apiVersion = VK_API_VERSION_1_3
    };
    std::vector<const char*> instExt = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
    };
    VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
        .enabledExtensionCount = (uint32_t)instExt.size(),
        .ppEnabledExtensionNames = instExt.data()
    };
    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
    // 3. 匹配设备 UUID
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    std::vector<VkPhysicalDevice> devs(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, devs.data());
    for ( auto d : devs ) {
        VkPhysicalDeviceIDProperties idProps{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES };
        VkPhysicalDeviceProperties2 props2{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &idProps };
        vkGetPhysicalDeviceProperties2(d, &props2);
        if ( memcmp(idProps.deviceUUID, uuid.data(), 16) == 0 ) {
            physicalDevice = d;
            break;
        }
    }
    if ( !physicalDevice ) throw std::runtime_error("No matching Vulkan device found for CUDA UUID");
    // 4. 查找队列
    uint32_t qCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qCount, nullptr);
    std::vector<VkQueueFamilyProperties> qProps(qCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qCount, qProps.data());
    for ( uint32_t i = 0; i < qCount; ++i ) {
        if ( qProps[i].queueFlags & VK_QUEUE_GRAPHICS_BIT ) {
            queueFamilyIndex = i;
            break;
        }
    }
    // 5. 创建 Device
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = queueFamilyIndex,
        .queueCount = 1, .pQueuePriorities = &prio };
    VkPhysicalDeviceVulkan13Features feat13{ .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
        .synchronization2 = VK_TRUE, .dynamicRendering = VK_TRUE };
    std::vector<const char*> devExt = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
            #ifdef _WIN32
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME
            #endif
    };
    VkDeviceCreateInfo devInfo{ .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &feat13, .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &qInfo, .enabledExtensionCount = (uint32_t)devExt.size(),
        .ppEnabledExtensionNames = devExt.data() };
    VK_CHECK(vkCreateDevice(physicalDevice, &devInfo, nullptr, &device));
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &graphicsQueue);
}
void cleanup() {
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
}
}

// ================== 实现类 ==================

struct VulkanCudaRasterizer::Impl {
    // 动态资源
    int currentWidth = 0, currentHeight = 0;
    // Vulkan 资源
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory imageMem = VK_NULL_HANDLE;
    VkImageView imageView = VK_NULL_HANDLE;
    VkBuffer vertexBuffer = VK_NULL_HANDLE;
    VkBuffer readBuffer = VK_NULL_HANDLE;
    VkDeviceMemory readBufferMem = VK_NULL_HANDLE;
    NativeHandle readBufferHandle = {};
    VkDeviceMemory vertexMem = VK_NULL_HANDLE;
    void* vertexMapped = nullptr;
    size_t vertexCapacity = 0; // 当前顶点缓冲容量
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    VkCommandBuffer cmdBuf = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    VkSemaphore sem = VK_NULL_HANDLE;
    // CUDA 互操作
    NativeHandle extMemHandle = {};
    NativeHandle extSemHandle = {};
    cudaExternalMemory_t cudaExtMem = nullptr;
    cudaExternalSemaphore_t cudaExtSem = nullptr;
    float* d_result = nullptr;
    Impl() {
        create_pipeline(); // Pipeline 只需创建一次
        create_command_pool();
        create_sync_objects();
        create_vertex_buffer(1024); // 初始容量
    }
    ~Impl() {
        vkDeviceWaitIdle(GlobalContext::device);
        // 清理 CUDA
        if ( d_result ) cudaFree(d_result);
        cudaDestroyExternalMemory(cudaExtMem);
        cudaDestroyExternalSemaphore(cudaExtSem);
        // 清理 Vulkan
        vkDestroySemaphore(GlobalContext::device, sem, nullptr);
        vkDestroyFence(GlobalContext::device, fence, nullptr);
        vkDestroyCommandPool(GlobalContext::device, cmdPool, nullptr);
        vkDestroyPipeline(GlobalContext::device, pipeline, nullptr);
        vkDestroyPipelineLayout(GlobalContext::device, layout, nullptr);
        cleanup_image_resources();
        vkDestroyBuffer(GlobalContext::device, vertexBuffer, nullptr);
        vkFreeMemory(GlobalContext::device, vertexMem, nullptr);
    }
    void cleanup_image_resources() {
        if ( image ) {
            vkDestroyImageView(GlobalContext::device, imageView, nullptr);
            vkDestroyImage(GlobalContext::device, image, nullptr);
            vkFreeMemory(GlobalContext::device, imageMem, nullptr);
            image = VK_NULL_HANDLE; // 重置句柄
        }
        if ( readBuffer ) {
            vkDestroyBuffer(GlobalContext::device, readBuffer, nullptr);
            vkFreeMemory(GlobalContext::device, readBufferMem, nullptr);
        }
    }

    // 核心渲染逻辑
    float* render(const std::vector<Edge>& edges, int w, int h, cudaStream_t stream) {
        // 1. 检查并重建分辨率资源
        if ( w != currentWidth || h != currentHeight || image == VK_NULL_HANDLE ) {
            rebuild_image_resources(w, h);
        }
        // 准备顶点数据
        std::vector<WindingVertex> verts;
        // verts.reserve(edges.size() * 3);
        Point2D anchor{ 0, 0 }; // 根据需求设定
        for (const auto& edge : edges) {
            Point2D v1 = edge.v1;
            Point2D v2 = edge.v2;
 
            // 2.1 计算向量
            // 向量 A: Anchor -> V1
            float ax = v1.x - anchor.x;
            float ay = v1.y - anchor.y;
            // 向量 B: Anchor -> V2
            float bx = v2.x - anchor.x;
            float by = v2.y - anchor.y;
 
            // 2.2 计算叉积 (确定方向)
            // cross = A.x * B.y - A.y * B.x
            // 结果 > 0 表示 V1 -> V2 相对于 Anchor 是逆时针 (CCW)
            // 结果 < 0 表示 V1 -> V2 相对于 Anchor 是顺时针 (CW)
            float cross = ax * by - ay * bx;
 
            // 2.3 确定权重
            int weight = 0;
            if (cross > 0) weight = 1;   // CCW: 累加 +1
            else if (cross < 0) weight = -1; // CW: 累加 -1
        
            verts.push_back({anchor.x, anchor.y, weight}); 
            verts.push_back({v1.x, v1.y, weight});
            verts.push_back({v2.x, v2.y, weight});
        }

        // 2. 检查并扩容顶点缓冲
        size_t reqSize = verts.size() * sizeof(WindingVertex);
        if ( reqSize > vertexCapacity ) {
            resize_vertex_buffer(reqSize * 2); // 扩容 2 倍
        }
        // 3. 同步：等待上一帧 GPU 结束
        vkWaitForFences(GlobalContext::device, 1, &fence, VK_TRUE, UINT64_MAX);
        vkResetFences(GlobalContext::device, 1, &fence);
        // 4. 上传数据
        memcpy(vertexMapped, verts.data(), reqSize);
        // 5. 记录并提交命令
        record_and_submit(verts.size());
        // 6. CUDA 同步
        cudaExternalSemaphoreWaitParams waitParams{};
        cudaWaitExternalSemaphoresAsync(&cudaExtSem, &waitParams, 1, stream);
        cudaStreamSynchronize(stream);
        return d_result;
    }

private:
    uint32_t find_memory(uint32_t typeFilter, VkMemoryPropertyFlags props) {
        VkPhysicalDeviceMemoryProperties memProps;
        vkGetPhysicalDeviceMemoryProperties(GlobalContext::physicalDevice, &memProps);
        for ( uint32_t i = 0; i < memProps.memoryTypeCount; ++i ) {
            if ( (typeFilter & (1 << i)) && (memProps.memoryTypes[i].propertyFlags & props) == props ) return i;
        }
        return 0;
    }
    void create_vertex_buffer(size_t size) {
        if ( vertexBuffer ) {
            vkDestroyBuffer(GlobalContext::device, vertexBuffer, nullptr);
            vkFreeMemory(GlobalContext::device, vertexMem, nullptr);
        }
        VkBufferCreateInfo info{ .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
            .usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT };
        VK_CHECK(vkCreateBuffer(GlobalContext::device, &info, nullptr, &vertexBuffer));
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(GlobalContext::device, vertexBuffer, &req);
        VkMemoryAllocateInfo alloc{ .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
            .memoryTypeIndex = find_memory(req.memoryTypeBits,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) };
        VK_CHECK(vkAllocateMemory(GlobalContext::device, &alloc, nullptr, &vertexMem));
        VK_CHECK(vkBindBufferMemory(GlobalContext::device, vertexBuffer, vertexMem, 0));
        VK_CHECK(vkMapMemory(GlobalContext::device, vertexMem, 0, size, 0, &vertexMapped));
        vertexCapacity = size;
    }
    void resize_vertex_buffer(size_t newSize) {
        create_vertex_buffer(newSize);
    }
    void rebuild_image_resources(int w, int h) {
        cleanup_image_resources();
        // 销毁旧的 CUDA 资源
        if ( d_result ) {
            cudaFree(d_result);
            d_result = nullptr;
        }
        if ( cudaExtMem ) cudaDestroyExternalMemory(cudaExtMem);
        currentWidth = w;
        currentHeight = h;
        // 创建 Vulkan Image (可导出)
        VkExternalMemoryImageCreateInfo extInfo{ .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO,
            .handleTypes = VK_EXT_MEM_HANDLE_TYPE };
        VkImageCreateInfo imgInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = &extInfo,
            .imageType = VK_IMAGE_TYPE_2D,
            .format = VK_FORMAT_R32_SFLOAT, // 支持混合的浮点格式
            .extent = { (uint32_t)w, (uint32_t)h, 1 },
            .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
            .tiling = VK_IMAGE_TILING_OPTIMAL,
            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED
        };
        VK_CHECK(vkCreateImage(GlobalContext::device, &imgInfo, nullptr, &image));
        VkMemoryRequirements imgReqs;
        vkGetImageMemoryRequirements(GlobalContext::device, image, &imgReqs);

        VkMemoryAllocateInfo imgAllocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .allocationSize = imgReqs.size,
            // Image 必须在 Device Local 显存上
            .memoryTypeIndex = find_memory(imgReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        };
        VK_CHECK(vkAllocateMemory(GlobalContext::device, &imgAllocInfo, nullptr, &imageMem));
        VK_CHECK(vkBindImageMemory(GlobalContext::device, image, imageMem, 0));
        // 创建 View
        VkImageViewCreateInfo viewInfo{ .sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = image,
            .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32_SFLOAT,
            .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 } };
        VK_CHECK(vkCreateImageView(GlobalContext::device, &viewInfo, nullptr, &imageView));

        // 创建 Vulkan Buffer (用于导出给 CUDA)
        VkDeviceSize bufferSize = w * h * sizeof(float);
        VkExternalMemoryBufferCreateInfo extBufInfo{ .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
            .handleTypes = VK_EXT_MEM_HANDLE_TYPE };

        VkBufferCreateInfo bufInfo{
            .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = &extBufInfo,
            .size = bufferSize,
            .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT, // 作为拷贝目标
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE
        };
        VK_CHECK(vkCreateBuffer(GlobalContext::device, &bufInfo, nullptr, &readBuffer));
        // 分配内存 (带导出信息)
        VkExportMemoryAllocateInfo exportAlloc{ .sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO,
            .handleTypes = VK_EXT_MEM_HANDLE_TYPE };
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(GlobalContext::device, readBuffer, &req);


        VkMemoryAllocateInfo allocInfo{
            .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = &exportAlloc,
            .allocationSize = req.size,
            .memoryTypeIndex = find_memory(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
        };
        VK_CHECK(vkAllocateMemory(GlobalContext::device, &allocInfo, nullptr, &readBufferMem));
        VK_CHECK(vkBindBufferMemory(GlobalContext::device, readBuffer, readBufferMem, 0));
        // 获取句柄
#ifdef _WIN32
        auto pfn = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(GlobalContext::device, "vkGetMemoryWin32HandleKHR");
        VkMemoryGetWin32HandleInfoKHR hInfo{ .sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR,
            .memory = readBufferMem,
            .handleType = VK_EXT_MEM_HANDLE_TYPE };
        pfn(GlobalContext::device, &hInfo, &extMemHandle);
#else
        auto pfn = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(GlobalContext::device, "vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR hInfo { .sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR, .memory = readBufferMem, .handleType = VK_EXT_MEM_HANDLE_TYPE };
        pfn(GlobalContext::device, &hInfo, &extMemHandle);
#endif
        // 映射到 CUDA
        cudaExternalMemoryHandleDesc memDesc{};
#ifdef _WIN32
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memDesc.handle.win32.handle = extMemHandle;
#else
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memDesc.handle.fd = extMemHandle;
#endif
        memDesc.size = req.size;
        cudaImportExternalMemory(&cudaExtMem, &memDesc);
        cudaExternalMemoryBufferDesc bufDesc{ .offset = 0, .size = bufferSize };
        cudaExternalMemoryGetMappedBuffer((void**)&d_result, cudaExtMem, &bufDesc);
    }
    void create_pipeline() {
        // 加载着色器 (假设已编译为 SPIR-V)
        // 此处省略文件读取代码，与之前版本一致
        std::vector<char> vertCode = read_file("assets/vert.spv");
        std::vector<char> fragCode = read_file("assets/frag.spv");
        VkShaderModule vertMod = create_module(vertCode);
        VkShaderModule fragMod = create_module(fragCode);
        std::array stages = {
            VkPipelineShaderStageCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vertMod, .pName = "main" },
            VkPipelineShaderStageCreateInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fragMod, .pName = "main" }
        };
        // 顶点输入布局
        VkVertexInputBindingDescription bind{ .binding = 0, .stride = sizeof(WindingVertex),
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX };
        std::array attrs = {
            VkVertexInputAttributeDescription{ .location = 0, .binding = 0, .format = VK_FORMAT_R32G32_SFLOAT,
                .offset = offsetof(WindingVertex, x) },
            VkVertexInputAttributeDescription{ .location = 1, .binding = 0, .format = VK_FORMAT_R32_SINT,
                .offset = offsetof(WindingVertex, weight) }
        };
        VkPipelineVertexInputStateCreateInfo vi{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1, .pVertexBindingDescriptions = &bind,
            .vertexAttributeDescriptionCount = (uint32_t)attrs.size(), .pVertexAttributeDescriptions = attrs.data() };
        VkPipelineInputAssemblyStateCreateInfo ia{ .sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST };
        // 动态状态
        std::array dynStates = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
        VkPipelineDynamicStateCreateInfo dyn{ .sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = (uint32_t)dynStates.size(), .pDynamicStates = dynStates.data() };
        VkPipelineViewportStateCreateInfo vp{ .sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .viewportCount = 1,
            .scissorCount = 1 };
        VkPipelineRasterizationStateCreateInfo rs{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE, .lineWidth = 1.0f };
        VkPipelineMultisampleStateCreateInfo ms{ .sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT };
        // 混合: 加法累加
        VkPipelineColorBlendAttachmentState att{
            .blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_ONE, .dstColorBlendFactor = VK_BLEND_FACTOR_ONE,
            .colorBlendOp = VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = VK_COLOR_COMPONENT_R_BIT
        };
        VkPipelineColorBlendStateCreateInfo cb{ .sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .attachmentCount = 1, .pAttachments = &att };
        VkPushConstantRange push{ .stageFlags = VK_SHADER_STAGE_VERTEX_BIT, .size = sizeof(float) * 2 };
        VkPipelineLayoutCreateInfo layInfo{ .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push };
        VK_CHECK(vkCreatePipelineLayout(GlobalContext::device, &layInfo, nullptr, &layout));
        VkFormat fmt = VK_FORMAT_R32_SFLOAT;
        VkPipelineRenderingCreateInfo ri{ .sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &fmt };
        VkGraphicsPipelineCreateInfo pi{
            .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
            .stageCount = (uint32_t)stages.size(), .pStages = stages.data(),
            .pVertexInputState = &vi, .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
            .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &dyn,
            .layout = layout
        };
        VK_CHECK(vkCreateGraphicsPipelines(GlobalContext::device, VK_NULL_HANDLE, 1, &pi, nullptr, &pipeline));
        vkDestroyShaderModule(GlobalContext::device, vertMod, nullptr);
        vkDestroyShaderModule(GlobalContext::device, fragMod, nullptr);
    }
    void create_command_pool() {
        VkCommandPoolCreateInfo info{ .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = GlobalContext::queueFamilyIndex };
        VK_CHECK(vkCreateCommandPool(GlobalContext::device, &info, nullptr, &cmdPool));
        VkCommandBufferAllocateInfo alloc{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmdPool,
            .commandBufferCount = 1 };
        VK_CHECK(vkAllocateCommandBuffers(GlobalContext::device, &alloc, &cmdBuf));
    }
    void create_sync_objects() {
        VkFenceCreateInfo fInfo{ .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT };
        VK_CHECK(vkCreateFence(GlobalContext::device, &fInfo, nullptr, &fence));
        VkExportSemaphoreCreateInfo expInfo{ .sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO,
            .handleTypes = VK_EXT_SEM_HANDLE_TYPE };
        VkSemaphoreCreateInfo sInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &expInfo };
        VK_CHECK(vkCreateSemaphore(GlobalContext::device, &sInfo, nullptr, &sem));
#ifdef _WIN32
        auto pfn = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(GlobalContext::device, "vkGetSemaphoreWin32HandleKHR");
        VkSemaphoreGetWin32HandleInfoKHR hInfo{ .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR, .semaphore = sem,
            .handleType = VK_EXT_SEM_HANDLE_TYPE };
        pfn(GlobalContext::device, &hInfo, &extSemHandle);
#else
        auto pfn = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(GlobalContext::device, "vkGetSemaphoreFdKHR");
        VkSemaphoreGetFdInfoKHR hInfo { .sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR, .semaphore = sem, .handleType = VK_EXT_SEM_HANDLE_TYPE };
        pfn(GlobalContext::device, &hInfo, &extSemHandle);
#endif
        cudaExternalSemaphoreHandleDesc semDesc{};
#ifdef _WIN32
        semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        semDesc.handle.win32.handle = extSemHandle;
#else
        semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        semDesc.handle.fd = extSemHandle;
#endif
        cudaImportExternalSemaphore(&cudaExtSem, &semDesc);
    }
    void record_and_submit(size_t vertCount) {
        vkResetCommandBuffer(cmdBuf, 0);
        VkCommandBufferBeginInfo begin{ .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmdBuf, &begin);
        // Barrier: Undefined -> Color Attachment
        VkImageMemoryBarrier2 b1{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .image = image, .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 }
        };
        VkDependencyInfo d1{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &b1 };
        vkCmdPipelineBarrier2(cmdBuf, &d1);
        // Dynamic Rendering
        VkRenderingAttachmentInfo att{
            .sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = imageView,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE,
            .clearValue = { .color = { { 0, 0, 0, 0 } } }
        };
        VkRenderingInfo ri{
            .sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
            .renderArea = { .extent = { (uint32_t)currentWidth, (uint32_t)currentHeight } }, .layerCount = 1,
            .colorAttachmentCount = 1, .pColorAttachments = &att
        };
        vkCmdBeginRendering(cmdBuf, &ri);
        VkViewport vp{ .width = (float)currentWidth, .height = (float)currentHeight, .maxDepth = 1.0f };
        VkRect2D sc{ .extent = { (uint32_t)currentWidth, (uint32_t)currentHeight } };
        vkCmdSetViewport(cmdBuf, 0, 1, &vp);
        vkCmdSetScissor(cmdBuf, 0, 1, &sc);
        vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize off = 0;
        vkCmdBindVertexBuffers(cmdBuf, 0, 1, &vertexBuffer, &off);
        float push[2] = { (float)currentWidth, (float)currentHeight };
        vkCmdPushConstants(cmdBuf, layout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(push), push);
        vkCmdDraw(cmdBuf, (uint32_t)vertCount, 1, 0, 0);
        vkCmdEndRendering(cmdBuf);
        VkImageMemoryBarrier2 barrierTrans{
            .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
            .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            .newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            .image = image,
            .subresourceRange = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1 }
        };
        VkDependencyInfo depTrans{ .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers = &barrierTrans };
        vkCmdPipelineBarrier2(cmdBuf, &depTrans);

        // Image -> Buffer
        VkBufferImageCopy copyRegion{
            .bufferOffset = 0,
            .bufferRowLength = (uint32_t)currentWidth, // 指定宽度
            .bufferImageHeight = (uint32_t)currentHeight,
            .imageSubresource = { .aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .mipLevel = 0, .baseArrayLayer = 0, .layerCount = 1 },
            .imageOffset = { 0, 0, 0 },
            .imageExtent = { (uint32_t)currentWidth, (uint32_t)currentHeight, 1 }
        };
        vkCmdCopyImageToBuffer(cmdBuf, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readBuffer, 1, &copyRegion);

        vkEndCommandBuffer(cmdBuf);
        VkSubmitInfo sub{ .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmdBuf,
            .signalSemaphoreCount = 1, .pSignalSemaphores = &sem };
        vkQueueSubmit(GlobalContext::graphicsQueue, 1, &sub, fence);
    }
    std::vector<char> read_file(const std::string& filename) {
        std::filesystem::path fullPath = GlobalContext::shaderDir / filename;
        std::ifstream file(fullPath, std::ios::ate | std::ios::binary);
        if ( !file.is_open() ) throw std::runtime_error("Shader not found");
        size_t size = file.tellg();
        std::vector<char> buf(size);
        file.seekg(0);
        file.read(buf.data(), size);
        file.close();
        return buf;
    }
    VkShaderModule create_module(const std::vector<char>& code) {
        VkShaderModuleCreateInfo info{ .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = code.size(),
            .pCode = (uint32_t*)code.data() };
        VkShaderModule mod;
        VK_CHECK(vkCreateShaderModule(GlobalContext::device, &info, nullptr, &mod));
        return mod;
    }
};
// ================== 公共接口实现 ==================
void VulkanCudaRasterizer::Init(int cudaDeviceId) { GlobalContext::init(cudaDeviceId); }
void VulkanCudaRasterizer::SetResourceDirectory(
    const std::string& path) {
    GlobalContext::setResourceDirectory(path);
}
void VulkanCudaRasterizer::Cleanup() { GlobalContext::cleanup(); }
VulkanCudaRasterizer::VulkanCudaRasterizer() : pImpl(new Impl()) {}
VulkanCudaRasterizer::~VulkanCudaRasterizer() { delete pImpl; }
float* VulkanCudaRasterizer::render(const std::vector<Edge>& edges, int width, int height, cudaStream_t stream) {
    return pImpl->render(edges, width, height, stream);
}

VulkanCudaRasterizer& VulkanCudaRasterizer::Instance() {
    static VulkanCudaRasterizer instance{};
    return instance;
}
}
