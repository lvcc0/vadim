#include <iostream>
#include <fstream>

#include <array>
#include <vector>
#include <map>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cstdint>

#include <vulkan/vulkan_raii.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

constexpr uint32_t WIDTH = 600;
constexpr uint32_t HEIGHT = 400;

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const*> validationLayers =
{
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex
{
    glm::vec2 pos;
    glm::vec3 color;

    static vk::VertexInputBindingDescription getBindingDescription()
    {
        return
        {
            .binding   = 0,
            .stride    = sizeof(Vertex),
            .inputRate = vk::VertexInputRate::eVertex
        };
    }

    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions()
    {
        return 
        {{
            { .location = 0, .binding = 0, .format = vk::Format::eR32G32Sfloat,    .offset = offsetof(Vertex, pos)   },
            { .location = 1, .binding = 0, .format = vk::Format::eR32G32B32Sfloat, .offset = offsetof(Vertex, color) }
        }};
    }
};

const std::vector<Vertex> vertices =
{
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}},
    {{ 0.5f,  0.5f}, {0.0f, 0.0f, 1.0f}},
    {{-0.5f,  0.5f}, {1.0f, 1.0f, 1.0f}}
};

const std::vector<uint16_t> indices =
{
    0, 1, 2, 2, 3, 0
};

class VadimApp
{
public:
    void run()
    {
        this->initWindow();
        this->initVulkan();
        this->mainLoop();
        this->cleanup();
    }

private:
    GLFWwindow* window = nullptr;

    vk::raii::Context                context;
    vk::raii::Instance               instance       = nullptr;
    vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
    vk::raii::SurfaceKHR             surface        = nullptr;

    vk::raii::PhysicalDevice physicalDevice = nullptr;
    vk::raii::Device         device         = nullptr;
    vk::raii::Queue          queue          = nullptr;
    uint32_t                 queueIndex     = ~0;

    vk::raii::SwapchainKHR           swapChain = nullptr;
    std::vector<vk::Image>           swapChainImages;
    vk::SurfaceFormatKHR             swapChainSurfaceFormat;
    vk::Extent2D                     swapChainExtent;
    std::vector<vk::raii::ImageView> swapChainImageViews;

    vk::raii::PipelineLayout pipelineLayout   = nullptr;
    vk::raii::Pipeline       graphicsPipeline = nullptr;

    vk::raii::Buffer       vertexBuffer       = nullptr;
    vk::raii::DeviceMemory vertexBufferMemory = nullptr;
    vk::raii::Buffer       indexBuffer        = nullptr;
    vk::raii::DeviceMemory indexBufferMemory  = nullptr;

    vk::raii::CommandPool                commandPool = nullptr;
    std::vector<vk::raii::CommandBuffer> commandBuffers;

    std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
    std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
    std::vector<vk::raii::Fence>     inFlightFences;
    uint32_t                         frameIndex = 0;

    bool framebufferResized = false;

    std::vector<const char*> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};

    void initWindow()
    {
        glfwInit();

        // don't create opengl context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        this->window = glfwCreateWindow(640, 480, "VADIM", nullptr, nullptr);
        
        glfwSetWindowUserPointer(this->window, this);
        glfwSetFramebufferSizeCallback(this->window, this->framebufferResizedCallback);
    }

    void initVulkan()
    {
        this->createInstance();
        this->setupDebugMessenger();
        this->createSurface();
        this->pickPhysicalDevice();
        this->createLogicalDevice();
        this->createSwapChain();
        this->createImageViews();
        this->createGraphicsPipeline();
        this->createCommandPool();
        this->createVertexBuffer();
        this->createIndexBuffer();
        this->createCommandBuffers();
        this->createSyncObjects();
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(this->window))
        {
            glfwPollEvents();
            this->drawFrame();
        }

        this->device.waitIdle();
    }

    void drawFrame()
    {
        vk::Result fenceResult = this->device.waitForFences(*this->inFlightFences[frameIndex], vk::True, UINT64_MAX);
    
        if (fenceResult != vk::Result::eSuccess)
            throw std::runtime_error("failed to wait for fence");
        
        this->device.resetFences(*this->inFlightFences[frameIndex]);

        auto [result, imageIndex] = this->swapChain.acquireNextImage(UINT64_MAX, *this->presentCompleteSemaphores[frameIndex], nullptr);

        if (result == vk::Result::eErrorOutOfDateKHR)
        {
            this->recreateSwapChain();
            return;
        }

        if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR)
            throw std::runtime_error("failed to acquire swap chain image");

        this->device.resetFences(*this->inFlightFences[frameIndex]);

        this->commandBuffers[frameIndex].reset();
        this->recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitDestionationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
        const vk::SubmitInfo submitInfo{
            .waitSemaphoreCount   = 1,
            .pWaitSemaphores      = &*this->presentCompleteSemaphores[frameIndex],
            .pWaitDstStageMask    = &waitDestionationStageMask,
            .commandBufferCount   = 1,
            .pCommandBuffers      = &*this->commandBuffers[frameIndex],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores    = &*this->renderFinishedSemaphores[imageIndex]
        };

        this->queue.submit(submitInfo, *this->inFlightFences[frameIndex]);

        const vk::PresentInfoKHR presentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores    = &*this->renderFinishedSemaphores[imageIndex],
            .swapchainCount     = 1,
            .pSwapchains        = &*this->swapChain,
            .pImageIndices      = &imageIndex
        };

        result = this->queue.presentKHR(presentInfoKHR);
        
        switch (result)
        {
        case vk::Result::eSuccess:
            break;
        
        case vk::Result::eErrorOutOfDateKHR:
        case vk::Result::eSuboptimalKHR:
            this->framebufferResized = false;
            this->recreateSwapChain();
            break;
        
        default:
            break;
        }
        
        this->frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void cleanup()
    {
        this->cleanupSwapChain();

        glfwDestroyWindow(this->window);
        glfwTerminate();
    }

    static void framebufferResizedCallback(GLFWwindow* window, int width, int height)
    {
        auto app = reinterpret_cast<VadimApp*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void createInstance()
    {
        constexpr vk::ApplicationInfo appInfo{
            .pApplicationName   = "VADIM",
            .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
            .pEngineName        = "None, baremetal!",
            .engineVersion      = VK_MAKE_VERSION(1, 0, 0),
            .apiVersion         = vk::ApiVersion14
        };

        // check for validation layers (DEBUG)

        std::vector<const char*> requiredLayers;
        if (enableValidationLayers) requiredLayers.assign(validationLayers.begin(), validationLayers.end());

        std::vector<vk::LayerProperties> layerProperties = this->context.enumerateInstanceLayerProperties();
        auto unsupportedLayerIt = std::ranges::find_if(requiredLayers, [&layerProperties](auto const &requiredLayer) {
            return std::ranges::none_of(layerProperties, [requiredLayer](auto const &layerProperty) {
                return strcmp(layerProperty.layerName, requiredLayer) == 0;
            });
        });

        if (unsupportedLayerIt != requiredLayers.end())
            throw std::runtime_error("required layer not supported: " + std::string(*unsupportedLayerIt));

        // check for extensions

        std::vector<const char*> requiredExtensions = this->getRequiredInstanceExtensions();

        std::vector<vk::ExtensionProperties> extensionProperties = this->context.enumerateInstanceExtensionProperties();
        auto unsupportedPropertyIt = std::ranges::find_if(requiredExtensions, [&extensionProperties](auto const &requiredExtension) {
            return std::ranges::none_of(extensionProperties, [requiredExtension](auto const &extensionProperty) {
                return strcmp(extensionProperty.extensionName, requiredExtension) == 0;
            });
        });

        if (unsupportedPropertyIt != requiredExtensions.end())
            throw std::runtime_error("required extension not supported: " + std::string(*unsupportedPropertyIt));

        // finally create the instance after all the checks

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo        = &appInfo,
            .enabledLayerCount       = static_cast<uint32_t>(requiredLayers.size()),
            .ppEnabledLayerNames     = requiredLayers.data(),
            .enabledExtensionCount   = static_cast<uint32_t>(requiredExtensions.size()),
            .ppEnabledExtensionNames = requiredExtensions.data()
        };

        this->instance = vk::raii::Instance(this->context, createInfo);
    }

    void setupDebugMessenger()
    {
        if (!enableValidationLayers) return;

        vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError
        );

        vk::DebugUtilsMessageTypeFlagsEXT typeFlags(
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
            vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
            vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation
        );

        vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
            .messageSeverity = severityFlags,
            .messageType     = typeFlags,
            .pfnUserCallback = &this->debugCallback
        };

        this->debugMessenger = this->instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
    }

    void createSurface()
    {
        VkSurfaceKHR _surface;

        if (glfwCreateWindowSurface(*this->instance, this->window, nullptr, &_surface) != 0)
            throw std::runtime_error("failed to create window surface");

        this->surface = vk::raii::SurfaceKHR(this->instance, _surface);
    }

    bool isPhysicalDeviceSuitable(const vk::raii::PhysicalDevice& pd)
    {
        auto extensions = pd.enumerateDeviceExtensionProperties();
        auto features = pd.template getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

        bool supportsVulkan13 = pd.getProperties().apiVersion >= vk::ApiVersion13;

        bool supportsGraphics = std::ranges::any_of(pd.getQueueFamilyProperties(), [](const auto& qfp) {
            return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

        bool supportsAllRequiredExtensions = std::ranges::all_of(this->requiredDeviceExtensions, [&extensions](const auto& requiredDeviceExtension) {
            return std::ranges::any_of(extensions, [requiredDeviceExtension](const auto& availableDeviceExtension) {
                return strcmp(availableDeviceExtension.extensionName, requiredDeviceExtension) == 0;
            });
        });

        bool supportsRequiredFeatures = features.template get<vk::PhysicalDeviceVulkan13Features>().dynamicRendering &&
                                        features.template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>().extendedDynamicState;

        return supportsVulkan13 && supportsGraphics && supportsAllRequiredExtensions && supportsRequiredFeatures;
    }

    void pickPhysicalDevice()
    {
        std::vector<vk::raii::PhysicalDevice> physicalDevices = this->instance.enumeratePhysicalDevices();
    
        if (physicalDevices.empty())
            throw std::runtime_error("failed to find GPUs with Vulkan support");

        // scoring available GPUs
        std::multimap<int, vk::raii::PhysicalDevice> candidates;

        for (const vk::raii::PhysicalDevice& pd : physicalDevices)
        {
            vk::PhysicalDeviceProperties deviceProperties = pd.getProperties();
            vk::PhysicalDeviceFeatures   deviceFeatures   = pd.getFeatures();
            uint32_t score = 0;

            if (!this->isPhysicalDeviceSuitable(pd))
                continue;

            if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
                score += 1000;

            score += deviceProperties.limits.maxImageDimension2D;

            candidates.insert(std::make_pair(score, pd));

            if (enableValidationLayers)
                std::cout << deviceProperties.deviceName << ": " << score << std::endl;
        }

        if (!candidates.empty() && candidates.rbegin()->first > 0)
        {
            std::cout << "using " << candidates.rbegin()->second.getProperties().deviceName << std::endl;

            this->physicalDevice = candidates.rbegin()->second;
            return;
        }
        
        throw std::runtime_error("failed to find a suitable GPU");
    }

    void createLogicalDevice()
    {
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = this->physicalDevice.getQueueFamilyProperties();

        // todo: rewrite with find_if, if possible
        for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size(); qfpIndex++)
        {
            if ((queueFamilyProperties[qfpIndex].queueFlags & vk::QueueFlagBits::eGraphics) && this->physicalDevice.getSurfaceSupportKHR(qfpIndex, *this->surface))
            {
                this->queueIndex = qfpIndex;
                break;
            }
        }

        if (this->queueIndex == ~0)
            throw std::runtime_error("could not find a queue for graphics and presenting");

        vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan11Features, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT> featureChain =
        {
            { // vk::PhysicalDeviceFeatures2

            },
            { // vk::PhysicalDeviceVulkan11Features
                .shaderDrawParameters = vk::True
            },
            { // vk::PhysicalDeviceVulkan13Features
                .synchronization2 = vk::True,
                .dynamicRendering = vk::True
            },
            { // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
               .extendedDynamicState = vk::True
            }
        };

        float queuePriority = 1.0f;

        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{
            .queueFamilyIndex = this->queueIndex,
            .queueCount       = 1,
            .pQueuePriorities = &queuePriority
        };

        vk::DeviceCreateInfo deviceCreateInfo{
            .pNext                   = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
            .queueCreateInfoCount    = 1,
            .pQueueCreateInfos       = &deviceQueueCreateInfo,
            .enabledExtensionCount   = static_cast<uint32_t>(requiredDeviceExtensions.size()),
            .ppEnabledExtensionNames = requiredDeviceExtensions.data()
        };

        this->device = vk::raii::Device(this->physicalDevice, deviceCreateInfo);
        this->queue  = vk::raii::Queue(this->device, this->queueIndex, 0);
    }

    void createSwapChain()
    {
        vk::SurfaceCapabilitiesKHR surfaceCapabilities = this->physicalDevice.getSurfaceCapabilitiesKHR(*this->surface);
        vk::PresentModeKHR presentMode = this->chooseSwapPresentMode(this->physicalDevice.getSurfacePresentModesKHR(*this->surface));
        uint32_t minImageCount = this->chooseSwapMinImageCount(surfaceCapabilities);
        
        this->swapChainExtent = this->chooseSwapExtent(surfaceCapabilities);
        this->swapChainSurfaceFormat = this->chooseSwapSurfaceFormat(this->physicalDevice.getSurfaceFormatsKHR(*this->surface));
        
        vk::SwapchainCreateInfoKHR swapChainCreateInfo{
            .surface          = *this->surface,
            .minImageCount    = minImageCount,
            .imageFormat      = this->swapChainSurfaceFormat.format,
            .imageColorSpace  = this->swapChainSurfaceFormat.colorSpace,
            .imageExtent      = this->swapChainExtent,
            .imageArrayLayers = 1,
            .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform     = surfaceCapabilities.currentTransform,
            .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode      = presentMode,
            .clipped          = true
        };

        this->swapChain = vk::raii::SwapchainKHR(this->device, swapChainCreateInfo);
        this->swapChainImages = this->swapChain.getImages();
    }

    static uint32_t chooseSwapMinImageCount(const vk::SurfaceCapabilitiesKHR& surfaceCapabilities)
    {
        auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);

        if ((0 < surfaceCapabilities.maxImageCount) && (surfaceCapabilities.maxImageCount < minImageCount))
            minImageCount = surfaceCapabilities.maxImageCount;

        return minImageCount;
    }

    static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats)
    {
        if (availableFormats.empty())
            throw std::runtime_error("no available formats for swap chain");
    
        const auto formatIt = std::ranges::find_if(availableFormats, [](const auto& format) {
            return format.format == vk::Format::eB8G8R8A8Srgb && format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });

        return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
    }

    static vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes)
    {
        if (std::ranges::all_of(availablePresentModes, [](auto presentMode) {
            return presentMode != vk::PresentModeKHR::eFifo;
        }))
            throw std::runtime_error("no available present mode for swap chain");
        
        return std::ranges::any_of(availablePresentModes, [](const auto value) {
            return vk::PresentModeKHR::eMailbox == value;
        }) ? vk::PresentModeKHR::eMailbox : vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities)
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;

        int width, height;
        glfwGetFramebufferSize(this->window, &width, &height);

        return {
            std::clamp<uint32_t>(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)
        };
    }

    void recreateSwapChain()
    {
        int width = 0, height = 0;
        glfwGetFramebufferSize(this->window, &width, &height);

        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(this->window, &width, &height);
            glfwWaitEvents();
        }

        this->device.waitIdle();

        this->cleanupSwapChain();
        this->createSwapChain();
        this->createImageViews();
    }

    void cleanupSwapChain()
    {
        this->swapChainImageViews.clear();
        this->swapChain = nullptr;
    }

    void createImageViews()
    {
        if (!this->swapChainImageViews.empty())
            throw std::runtime_error("swap chain image views are not empty");
    
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .viewType = vk::ImageViewType::e2D,
            .format = this->swapChainSurfaceFormat.format,
            .subresourceRange = { vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1 }
        };

        for (auto &image : this->swapChainImages)
        {
            imageViewCreateInfo.image = image;
            this->swapChainImageViews.emplace_back(this->device, imageViewCreateInfo);
        }
    }

    void createGraphicsPipeline()
    {
        // note: path relative to main.cpp
        vk::raii::ShaderModule shaderModule = createShaderModule(readFile("../res/shaders/slang.spv"));
    
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = shaderModule,
            .pName = "vertMain"
        };

        vk::PipelineShaderStageCreateInfo fragShaderStageInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = shaderModule,
            .pName = "fragMain"
        };

        vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};
    
        vk::VertexInputBindingDescription                  bindingDescription    = Vertex::getBindingDescription();
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions = Vertex::getAttributeDescriptions();
        
        vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
            .vertexBindingDescriptionCount   = 1,
            .pVertexBindingDescriptions      = &bindingDescription,
            .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
            .pVertexAttributeDescriptions    = attributeDescriptions.data()
        };

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{
            .topology = vk::PrimitiveTopology::eTriangleList
        };

        vk::PipelineViewportStateCreateInfo viewportState{
            .viewportCount = 1,
            .scissorCount = 1
        };
    
        vk::PipelineRasterizationStateCreateInfo rasterizer{
            .depthClampEnable        = vk::False,
            .rasterizerDiscardEnable = vk::False,
            .polygonMode             = vk::PolygonMode::eFill,
            .cullMode                = vk::CullModeFlagBits::eBack,
            .frontFace               = vk::FrontFace::eClockwise,
            .depthBiasEnable         = vk::False,
            .lineWidth               = 1.0f
        };

        vk::PipelineMultisampleStateCreateInfo multisampling{
            .rasterizationSamples = vk::SampleCountFlagBits::e1,
            .sampleShadingEnable = vk::False
        };

        vk::PipelineColorBlendAttachmentState colorBlendAttachment{
		    .blendEnable    = vk::False,
		    .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA
        };

        vk::PipelineColorBlendStateCreateInfo colorBlending{
		    .logicOpEnable = vk::False,
            .logicOp = vk::LogicOp::eCopy,
            .attachmentCount = 1,
            .pAttachments = &colorBlendAttachment
        };

        std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
		vk::PipelineDynamicStateCreateInfo dynamicState{
            .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
            .pDynamicStates = dynamicStates.data()
        };

        vk::PipelineLayoutCreateInfo pipelineLayoutInfo{
            .setLayoutCount = 0,
            .pushConstantRangeCount = 0
        };
		
        this->pipelineLayout = vk::raii::PipelineLayout(this->device, pipelineLayoutInfo);
    
        vk::StructureChain<vk::GraphicsPipelineCreateInfo, vk::PipelineRenderingCreateInfo> pipelineCreateInfoChain = {
            {
                .stageCount          = 2,
                .pStages             = shaderStages,
                .pVertexInputState   = &vertexInputInfo,
                .pInputAssemblyState = &inputAssembly,
                .pViewportState      = &viewportState,
                .pRasterizationState = &rasterizer,
                .pMultisampleState   = &multisampling,
                .pColorBlendState    = &colorBlending,
                .pDynamicState       = &dynamicState,
                .layout              = this->pipelineLayout,
                .renderPass          = nullptr // dynamic rendering!
            },
            {
                .colorAttachmentCount    = 1,
                .pColorAttachmentFormats = &this->swapChainSurfaceFormat.format
            }
        };

        this->graphicsPipeline = vk::raii::Pipeline(this->device, nullptr, pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
    }

    [[nodiscard]] vk::raii::ShaderModule createShaderModule(const std::vector<char>& code) const
    {
        vk::ShaderModuleCreateInfo shaderModuleCreateInfo{
            .codeSize = code.size() * sizeof(char),
            .pCode = reinterpret_cast<const uint32_t*>(code.data())
        };

        vk::raii::ShaderModule shaderModule{
            this->device,
            shaderModuleCreateInfo
        };

        return shaderModule;
    }

    void createCommandPool()
    {
        vk::CommandPoolCreateInfo poolInfo
        {
            .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            .queueFamilyIndex = this->queueIndex
        };

        this->commandPool = vk::raii::CommandPool(this->device, poolInfo);
    }
    
    void createVertexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        auto [stagingBuffer, stagingBufferMemory] = this->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, vertices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        std::tie(this->vertexBuffer, this->vertexBufferMemory) = this->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        this->copyBuffer(stagingBuffer, this->vertexBuffer, bufferSize);
    }

    void createIndexBuffer()
    {
        vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        auto [stagingBuffer, stagingBufferMemory] = this->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eTransferSrc,
            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
        );

        void* data = stagingBufferMemory.mapMemory(0, bufferSize);
        memcpy(data, indices.data(), bufferSize);
        stagingBufferMemory.unmapMemory();

        std::tie(this->indexBuffer, this->indexBufferMemory) = this->createBuffer(
            bufferSize,
            vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
            vk::MemoryPropertyFlagBits::eDeviceLocal
        );

        this->copyBuffer(stagingBuffer, this->indexBuffer, bufferSize);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties memoryProperties = this->physicalDevice.getMemoryProperties();

        for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++)
            if ((typeFilter & (1 << i)) && (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                return i;
        
        throw std::runtime_error("failed to find suitable memory");
    }

    std::pair<vk::raii::Buffer, vk::raii::DeviceMemory> createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
    {
        // todo: custom allocator that splits up a single allocation among many different objects by using the offset parameters

        vk::BufferCreateInfo bufferInfo{
            .size        = size,
            .usage       = usage,
            .sharingMode = vk::SharingMode::eExclusive
        };

        vk::raii::Buffer buffer = vk::raii::Buffer(this->device, bufferInfo);

        vk::MemoryRequirements memoryRequirements = buffer.getMemoryRequirements();
    
        vk::MemoryAllocateInfo memoryAllocateInfo{
            .allocationSize = memoryRequirements.size,
            .memoryTypeIndex = this->findMemoryType(memoryRequirements.memoryTypeBits, properties)
        };

        vk::raii::DeviceMemory bufferMemory = vk::raii::DeviceMemory(this->device, memoryAllocateInfo);
        buffer.bindMemory(*bufferMemory, 0);

        return {std::move(buffer), std::move(bufferMemory)};
    }

    void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size)
    {
        vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = this->commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1
        };

        vk::raii::CommandBuffer commandCopyBuffer = std::move(device.allocateCommandBuffers(allocateInfo).front());

        commandCopyBuffer.begin({ .flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit });
        commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, vk::BufferCopy(0, 0, size));
        commandCopyBuffer.end();

        queue.submit(vk::SubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &*commandCopyBuffer}, nullptr);
        queue.waitIdle();
    }

    void createCommandBuffers()
    {
        vk::CommandBufferAllocateInfo allocateInfo{
            .commandPool = this->commandPool,
            .level = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = MAX_FRAMES_IN_FLIGHT
        };

        this->commandBuffers = vk::raii::CommandBuffers(this->device, allocateInfo);
    }

    void recordCommandBuffer(uint32_t imageIndex)
    {
        vk::raii::CommandBuffer &commandBuffer = this->commandBuffers[frameIndex];
        
        commandBuffer.begin({});

        this->transitionImageLayout(
            imageIndex,
            vk::ImageLayout::eUndefined,
            vk::ImageLayout::eColorAttachmentOptimal,
            {},
            vk::AccessFlagBits2::eColorAttachmentWrite,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eColorAttachmentOutput
        );

        vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
        
        vk::RenderingAttachmentInfo attachmentInfo = 
        {
            .imageView   = this->swapChainImageViews[imageIndex],
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp      = vk::AttachmentLoadOp::eClear,
            .storeOp     = vk::AttachmentStoreOp::eStore,
            .clearValue  = clearColor
        };

        vk::RenderingInfo renderingInfo = 
        {
            .renderArea           =
            {
                .offset = {0, 0},
                .extent = this->swapChainExtent
            },
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &attachmentInfo
        };

        // todo: store multiple buffers, like the vertex and index buffer, into a single vk::raii::Buffer and use offsets in commands

        commandBuffer.beginRendering(renderingInfo);
        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, *this->graphicsPipeline);
        commandBuffer.setViewport(0, vk::Viewport(0.0f, 0.0f, static_cast<float>(this->swapChainExtent.width), static_cast<float>(this->swapChainExtent.height), 0.0f, 1.0f));
        commandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), this->swapChainExtent));
        commandBuffer.bindVertexBuffers(0, *this->vertexBuffer, {0});
        commandBuffer.bindIndexBuffer(*this->indexBuffer, 0, vk::IndexTypeValue<decltype(indices)::value_type>::value);
        commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        commandBuffer.endRendering();

        this->transitionImageLayout(
            imageIndex,
            vk::ImageLayout::eColorAttachmentOptimal,
            vk::ImageLayout::ePresentSrcKHR,
            vk::AccessFlagBits2::eColorAttachmentWrite,
            {},
            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
            vk::PipelineStageFlagBits2::eBottomOfPipe
        );

        commandBuffer.end();
    }

    void transitionImageLayout(
        uint32_t                imageIndex,
        vk::ImageLayout         oldLayout,
        vk::ImageLayout         newLayout,
        vk::AccessFlags2        srcAccessMask,
        vk::AccessFlags2        dstAccessMask,
        vk::PipelineStageFlags2 srcStageMask,
        vk::PipelineStageFlags2 dstStageMask
    )
    {
        vk::ImageMemoryBarrier2 barrier =
        {
            .srcStageMask        = srcStageMask,
            .srcAccessMask       = srcAccessMask,
            .dstStageMask        = dstStageMask,
            .dstAccessMask       = dstAccessMask,
            .oldLayout           = oldLayout,
            .newLayout           = newLayout,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image               = this->swapChainImages[imageIndex],
            .subresourceRange    =
            {
                .aspectMask     = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel   = 0,
                .levelCount     = 1,
                .baseArrayLayer = 0,
                .layerCount     = 1
            }
        };

        vk::DependencyInfo dependencyInfo =
        {
            .dependencyFlags         = {},
            .imageMemoryBarrierCount = 1,
            .pImageMemoryBarriers    = &barrier
        };

        this->commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
    }

    void createSyncObjects()
    {
        if (!(presentCompleteSemaphores.empty() && renderFinishedSemaphores.empty() && inFlightFences.empty()))
            throw std::runtime_error("sync objects non-empty");

        for (size_t i = 0; i < this->swapChainImages.size(); i++)
            this->renderFinishedSemaphores.emplace_back(this->device, vk::SemaphoreCreateInfo());

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            this->presentCompleteSemaphores.emplace_back(this->device, vk::SemaphoreCreateInfo());
            this->inFlightFences.emplace_back(this->device, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
        }
    }

    std::vector<const char*> getRequiredInstanceExtensions()
    {
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // GLFW extensions are always required, so always push them
        std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

        // debug messenger
        if (enableValidationLayers) extensions.push_back(vk::EXTDebugUtilsExtensionName);

        return extensions;
    }

    static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
        vk::DebugUtilsMessageSeverityFlagBitsEXT      severity,
        vk::DebugUtilsMessageTypeFlagsEXT             type,
        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void*                                         pUserData
    )
    {
        // warning or error
        if (severity >= vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning)
            std::cerr << "validation layer: type " << to_string(type) << " msg: " << pCallbackData->pMessage << std::endl;

        return vk::False;
    }

    static std::vector<char> readFile(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open())
            throw std::runtime_error("failed to open file: " + filename);

        std::vector<char> buffer(file.tellg());

        file.seekg(0, std::ios::beg);
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        file.close();

        return buffer;
    }
};

int main(int argc, char* argv[]) {
    try
    {
        std::cout << "hello vadim" << std::endl;

        VadimApp app;
        app.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }

    return 0;
}

