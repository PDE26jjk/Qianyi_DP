#include <vector>
#include <vulkan/vulkan.h>
#include <stdexcept>
#include "graphics.h"
namespace graphics {
VkInstance instance;
VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
VkDevice device;
VkQueue graphicsQueue;
uint32_t queueFamilyIndex = 0;

VkPhysicalDevice find_matching_vulkan_device(VkInstance instance, const std::vector<uint8_t>& target_uuid) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& device : devices) {
        // 准备查询 ID 属性的结构体链
        VkPhysicalDeviceIDProperties deviceIDProps = {};
        deviceIDProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;

        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &deviceIDProps; // 挂载 ID 属性查询

        vkGetPhysicalDeviceProperties2(device, &props2);

        // 比较 UUID (VK_UUID_SIZE 通常为 16)
        if (memcmp(deviceIDProps.deviceUUID, target_uuid.data(), 16) == 0) {
            return device; // 找到了匹配的 GPU
        }
    }
    return VK_NULL_HANDLE;
}

void init(const std::vector<uint8_t>& device_uuid) {// same as cuda
    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.apiVersion = VK_API_VERSION_1_2; // 推荐至少 1.1，1.2 更好

    std::vector<const char*> instanceExtensions = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME
    };

    VkInstanceCreateInfo createInfo = {};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
    createInfo.ppEnabledExtensionNames = instanceExtensions.data();

    if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan instance!");
    }

    // 2. 匹配与 CUDA 相同的 Physical Device
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    for (const auto& dev : devices) {
        VkPhysicalDeviceIDProperties idProps = {};
        idProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
        VkPhysicalDeviceProperties2 props2 = {};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &idProps;
        vkGetPhysicalDeviceProperties2(dev, &props2);

        if (memcmp(idProps.deviceUUID, device_uuid.data(), 16) == 0) {
            physicalDevice = dev;
            break;
        }
    }

    if (physicalDevice == VK_NULL_HANDLE) {
        throw std::runtime_error("Failed to find a Vulkan device matching the CUDA UUID!");
    }

    // 3. 查找支持 Graphics 的 Queue Family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    bool found = false;
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            queueFamilyIndex = i;
            found = true; break;
        }
    }
    if (!found) throw std::runtime_error("No graphics queue family found!");

    // 4. 创建 Logical Device (开启 Win32 外部内存和信号量扩展)
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
    };

    VkDeviceCreateInfo deviceInfo = {};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceInfo.ppEnabledExtensionNames = deviceExtensions.data();

    if (vkCreateDevice(physicalDevice, &deviceInfo, nullptr, &device) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(device, queueFamilyIndex, 0, &graphicsQueue);

    
}
}

#include <vulkan/vulkan.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <fstream>
#include <iostream>

// --- 跨平台宏定义 ---
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

// 几何数据结构
struct Point2D { float x, y; };
struct Edge { Point2D v1, v2; };
struct WindingVertex {
    float x, y;
    int32_t weight;
};

class VulkanCudaRasterizer {
private:
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    uint32_t queueFamilyIndex;
    int width, height;

    // Vulkan 渲染资源
    VkRenderPass renderPass;
    VkPipelineLayout pipelineLayout;
    VkPipeline graphicsPipeline;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;

    // 共享 Image 资源
    VkImage sharedImage;
    VkDeviceMemory sharedImageMemory;
    VkImageView sharedImageView;
    VkFramebuffer framebuffer;
    NativeHandle extMemHandle;

    // 顶点缓冲 (Host Visible，用于每帧更新多边形数据)
    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    void* vertexMappedData = nullptr;
    size_t maxVertexCount = 100000; // 预分配足够大的空间

    // 同步对象
    VkSemaphore vkRenderCompleteSem;
    NativeHandle extSemHandle;

    // CUDA 互操作对象
    cudaExternalMemory_t cudaExtMem;
    cudaExternalSemaphore_t cudaExtSem;
    int8_t* d_cuda_grid_ptr = nullptr;

public:
    VulkanCudaRasterizer(VkPhysicalDevice pDev, VkDevice dev, VkQueue queue, uint32_t qFamIdx, int w, int h)
        : physicalDevice(pDev), device(dev), graphicsQueue(queue), queueFamilyIndex(qFamIdx), width(w), height(h) 
    {
        create_shared_image();
        create_render_pass();
        create_framebuffer();
        create_graphics_pipeline();
        create_vertex_buffer();
        create_shared_semaphore();
        create_command_buffer();
        
        map_resources_to_cuda();
    }

    ~VulkanCudaRasterizer() {
        // 清理 CUDA 资源
        if (d_cuda_grid_ptr) cudaFree(d_cuda_grid_ptr);
        cudaDestroyExternalMemory(cudaExtMem);
        cudaDestroyExternalSemaphore(cudaExtSem);

        // 清理 Vulkan 资源
        vkDestroySemaphore(device, vkRenderCompleteSem, nullptr);
        vkDestroyBuffer(device, vertexBuffer, nullptr);
        vkFreeMemory(device, vertexBufferMemory, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyRenderPass(device, renderPass, nullptr);
        vkDestroyFramebuffer(device, framebuffer, nullptr);
        vkDestroyImageView(device, sharedImageView, nullptr);
        vkDestroyImage(device, sharedImage, nullptr);
        vkFreeMemory(device, sharedImageMemory, nullptr);
        vkDestroyCommandPool(device, commandPool, nullptr);
    }

    // --- 核心工作接口 ---
    int8_t* render_edges_to_cuda(const std::vector<Edge>& edges, cudaStream_t stream = 0) {
        if (edges.empty()) return d_cuda_grid_ptr;

        std::vector<WindingVertex> vertices;
        vertices.reserve(edges.size() * 3);
        Point2D anchor = {0.0f, 0.0f};

        for (const auto& edge : edges) {
            // 这里假定传入的边已经是逆时针为内部边界。
            // 逆时针边向目标累加 1。如果是孔洞（顺时针），请在此处传入 -1。
            int32_t weight = 1; 
            vertices.push_back({anchor.x, anchor.y, weight});
            vertices.push_back({edge.v1.x, edge.v1.y, weight});
            vertices.push_back({edge.v2.x, edge.v2.y, weight});
        }

        size_t dataSize = vertices.size() * sizeof(WindingVertex);
        if (vertices.size() > maxVertexCount) throw std::runtime_error("Vertex buffer overflow!");

        // 1. 更新顶点数据 (Host to Device)
        memcpy(vertexMappedData, vertices.data(), dataSize);

        // 2. 录制命令缓冲 (因为顶点数量变了，需重新录制)
        record_command_buffer(vertices.size());

        // 3. 提交到 Vulkan 队列
        VkSubmitInfo submitInfo = {};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &vkRenderCompleteSem; // 渲染完发出信号

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);

        // 4. CUDA 异步等待 Vulkan 渲染完成
        cudaExternalSemaphoreWaitParams waitParams = {};
        cudaWaitExternalSemaphoresAsync(&cudaExtSem, &waitParams, 1, stream);

        // 此时，你可以安全地在给定的 CUDA Stream 中启动 Kernel 处理 d_cuda_grid_ptr 了
        return d_cuda_grid_ptr;
    }

private:
    uint32_t find_memory_type(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        throw std::runtime_error("Failed to find suitable memory type!");
    }

    void create_shared_image() {
        VkExternalMemoryImageCreateInfo extImageInfo = {};
        extImageInfo.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
        extImageInfo.handleTypes = VK_EXT_MEM_HANDLE_TYPE;

        VkImageCreateInfo imageInfo = {};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.pNext = &extImageInfo;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.format = VK_FORMAT_R8_SINT; // 8位有符号整型
        imageInfo.extent = { (uint32_t)width, (uint32_t)height, 1 };
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
        imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

        vkCreateImage(device, &imageInfo, nullptr, &sharedImage);

        VkMemoryRequirements memReqs;
        vkGetImageMemoryRequirements(device, sharedImage, &memReqs);

        VkExportMemoryAllocateInfo exportAllocInfo = {};
        exportAllocInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
        exportAllocInfo.handleTypes = VK_EXT_MEM_HANDLE_TYPE;

        VkMemoryAllocateInfo allocInfo = {};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.pNext = &exportAllocInfo;
        allocInfo.allocationSize = memReqs.size;
        allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        vkAllocateMemory(device, &allocInfo, nullptr, &sharedImageMemory);
        vkBindImageMemory(device, sharedImage, sharedImageMemory, 0);

        // 提取句柄
#ifdef _WIN32
        auto pfnGetMemHandle = (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetMemoryWin32HandleKHR");
        VkMemoryGetWin32HandleInfoKHR handleInfo = { VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR };
        handleInfo.memory = sharedImageMemory;
        handleInfo.handleType = VK_EXT_MEM_HANDLE_TYPE;
        pfnGetMemHandle(device, &handleInfo, &extMemHandle);
#else
        auto pfnGetMemHandle = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(device, "vkGetMemoryFdKHR");
        VkMemoryGetFdInfoKHR handleInfo = { VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
        handleInfo.memory = sharedImageMemory;
        handleInfo.handleType = VK_EXT_MEM_HANDLE_TYPE;
        pfnGetMemHandle(device, &handleInfo, &extMemHandle);
#endif

        VkImageViewCreateInfo viewInfo = {};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = sharedImage;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = VK_FORMAT_R8_SINT;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        vkCreateImageView(device, &viewInfo, nullptr, &sharedImageView);
    }

    void create_shared_semaphore() {
        VkExportSemaphoreCreateInfo exportSemInfo = {};
        exportSemInfo.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
        exportSemInfo.handleTypes = VK_EXT_SEM_HANDLE_TYPE;

        VkSemaphoreCreateInfo semInfo = {};
        semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
        semInfo.pNext = &exportSemInfo;
        vkCreateSemaphore(device, &semInfo, nullptr, &vkRenderCompleteSem);

#ifdef _WIN32
        auto pfnGetSemHandle = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreWin32HandleKHR");
        VkSemaphoreGetWin32HandleInfoKHR handleInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR };
        handleInfo.semaphore = vkRenderCompleteSem;
        handleInfo.handleType = VK_EXT_SEM_HANDLE_TYPE;
        pfnGetSemHandle(device, &handleInfo, &extSemHandle);
#else
        auto pfnGetSemHandle = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
        VkSemaphoreGetFdInfoKHR handleInfo = { VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
        handleInfo.semaphore = vkRenderCompleteSem;
        handleInfo.handleType = VK_EXT_SEM_HANDLE_TYPE;
        pfnGetSemHandle(device, &handleInfo, &extSemHandle);
#endif
    }

    void map_resources_to_cuda() {
        // 映射内存
        cudaExternalMemoryHandleDesc memDesc = {};
        memDesc.size = width * height * sizeof(int8_t);
#ifdef _WIN32
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
        memDesc.handle.win32.handle = extMemHandle;
#else
        memDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
        memDesc.handle.fd = extMemHandle;
#endif
        cudaImportExternalMemory(&cudaExtMem, &memDesc);

        cudaExternalMemoryBufferDesc bufferDesc = {};
        bufferDesc.offset = 0;
        bufferDesc.size = memDesc.size;
        cudaExternalMemoryGetMappedBuffer((void**)&d_cuda_grid_ptr, cudaExtMem, &bufferDesc);

        // 映射信号量
        cudaExternalSemaphoreHandleDesc semDesc = {};
#ifdef _WIN32
        semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
        semDesc.handle.win32.handle = extSemHandle;
#else
        semDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
        semDesc.handle.fd = extSemHandle;
#endif
        cudaImportExternalSemaphore(&cudaExtSem, &semDesc);
    }

    void create_render_pass() {
        VkAttachmentDescription colorAttachment = {};
        colorAttachment.format = VK_FORMAT_R8_SINT;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        // 关键：每次渲染前必须清零，否则上一帧的累加结果会污染当前帧
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR; 
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_GENERAL; // General 布局方便 CUDA 读写

        VkAttachmentReference colorAttachmentRef = {};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;

        VkRenderPassCreateInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO };
        renderPassInfo.attachmentCount = 1;
        renderPassInfo.pAttachments = &colorAttachment;
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass);
    }

    void create_framebuffer() {
        VkFramebufferCreateInfo fbInfo = { VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO };
        fbInfo.renderPass = renderPass;
        fbInfo.attachmentCount = 1;
        fbInfo.pAttachments = &sharedImageView;
        fbInfo.width = width;
        fbInfo.height = height;
        fbInfo.layers = 1;
        vkCreateFramebuffer(device, &fbInfo, nullptr, &framebuffer);
    }

    // 辅助函数：读取 SPIR-V
    std::vector<char> read_file(const std::string& filename) {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);
        if (!file.is_open()) throw std::runtime_error("Failed to open shader file!");
        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();
        return buffer;
    }

    VkShaderModule create_shader_module(const std::vector<char>& code) {
        VkShaderModuleCreateInfo createInfo = { VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module;
        vkCreateShaderModule(device, &createInfo, nullptr, &module);
        return module;
    }

    void create_graphics_pipeline() {
        auto vertShaderCode = read_file("vert.spv");
        auto fragShaderCode = read_file("frag.spv");
        VkShaderModule vertModule = create_shader_module(vertShaderCode);
        VkShaderModule fragModule = create_shader_module(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertStageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        vertStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertStageInfo.module = vertModule;
        vertStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo fragStageInfo = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO };
        fragStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragStageInfo.module = fragModule;
        fragStageInfo.pName = "main";

        VkPipelineShaderStageCreateInfo shaderStages[] = { vertStageInfo, fragStageInfo };

        // 顶点输入布局匹配 WindingVertex
        VkVertexInputBindingDescription bindingDescription = {};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(WindingVertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        std::vector<VkVertexInputAttributeDescription> attributeDescriptions(2);
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(WindingVertex, x);
        
        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32_SINT; // 对应 inWeight
        attributeDescriptions[1].offset = offsetof(WindingVertex, weight);

        VkPipelineVertexInputStateCreateInfo vertexInputInfo = { VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly = { VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST; // 三角形扇被拆分为了独立的三角形
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport = { 0.0f, 0.0f, (float)width, (float)height, 0.0f, 1.0f };
        VkRect2D scissor = { {0, 0}, {(uint32_t)width, (uint32_t)height} };
        VkPipelineViewportStateCreateInfo viewportState = { VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer = { VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        // 极其重要：关闭背面剔除，无论边是怎么转的，虚拟三角形都要画出来参与累加
        rasterizer.cullMode = VK_CULL_MODE_NONE; 
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;

        VkPipelineMultisampleStateCreateInfo multisampling = { VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        // 核心混合逻辑：加法混合
        VkPipelineColorBlendAttachmentState colorBlendAttachment = {};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT;
        colorBlendAttachment.blendEnable = VK_TRUE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // + inWeight
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE; // + 原来的值
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

        VkPipelineColorBlendStateCreateInfo colorBlending = { VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;

        VkPushConstantRange pushConstant = {};
        pushConstant.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(float) * 2; // width, height

        VkPipelineLayoutCreateInfo pipelineLayoutInfo = { VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipelineLayoutInfo.pushConstantRangeCount = 1;
        pipelineLayoutInfo.pPushConstantRanges = &pushConstant;
        vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout);

        VkGraphicsPipelineCreateInfo pipelineInfo = { VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;

        vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline);

        vkDestroyShaderModule(device, fragModule, nullptr);
        vkDestroyShaderModule(device, vertModule, nullptr);
    }

    void create_vertex_buffer() {
        VkBufferCreateInfo bufferInfo = { VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufferInfo.size = maxVertexCount * sizeof(WindingVertex);
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        vkCreateBuffer(device, &bufferInfo, nullptr, &vertexBuffer);

        VkMemoryRequirements memReqs;
        vkGetBufferMemoryRequirements(device, vertexBuffer, &memReqs);

        VkMemoryAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo.allocationSize = memReqs.size;
        // 使用 Host Visible 方便 CPU 每帧更新几何体
        allocInfo.memoryTypeIndex = find_memory_type(memReqs.memoryTypeBits, 
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        vkAllocateMemory(device, &allocInfo, nullptr, &vertexBufferMemory);
        vkBindBufferMemory(device, vertexBuffer, vertexBufferMemory, 0);
        vkMapMemory(device, vertexBufferMemory, 0, bufferInfo.size, 0, &vertexMappedData);
    }

    void create_command_buffer() {
        VkCommandPoolCreateInfo poolInfo = { VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
        poolInfo.queueFamilyIndex = queueFamilyIndex;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

        VkCommandBufferAllocateInfo allocInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
    }

    void record_command_buffer(size_t vertexCount) {
        vkResetCommandBuffer(commandBuffer, 0);

        VkCommandBufferBeginInfo beginInfo = { VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        // 转换 Image Layout 以便渲染
        // (略微简化：RenderPass 已经配置了 initialLayout = UNDEFINED -> COLOR_ATTACHMENT)

        VkRenderPassBeginInfo renderPassInfo = { VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffer;
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = { (uint32_t)width, (uint32_t)height };

        VkClearValue clearColor = {}; // 初始化背景为 0（即外部）
        clearColor.color = { {0, 0, 0, 0} };
        renderPassInfo.clearValueCount = 1;
        renderPassInfo.pClearValues = &clearColor;

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkBuffer vertexBuffers[] = { vertexBuffer };
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, vertexBuffers, offsets);

        float pushData[2] = { (float)width, (float)height };
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(float)*2, pushData);

        vkCmdDraw(commandBuffer, static_cast<uint32_t>(vertexCount), 1, 0, 0);
        vkCmdEndRenderPass(commandBuffer);
        
        // 此处不需要显式的 Image Memory Barrier 到 General，因为 RenderPass 的 finalLayout 已经配置为 GENERAL
        
        vkEndCommandBuffer(commandBuffer);
    }
};