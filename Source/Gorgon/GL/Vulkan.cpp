#include "Vulkan.h"
#include "FrameBuffer.h"
#include "Shader.h"

#include "../Window.h"
#include "../Containers/Image.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>

#define VULKAN_UTIL_CORE_ONLY
#include "VulkanUtil.hpp"


namespace Gorgon :: GL {

	Gorgon::Utils::Logger log;

	const char *GetNativeSurfaceExtension();
	VkSurfaceKHR CreateNativeSurface(VkInstance instance, void *display, unsigned long window);

	namespace VF = VulkanAbstraction::VulkanF;
	namespace SC = VulkanAbstraction::SingleTimeCommands;
	namespace RC = VulkanAbstraction::VulkanF::RecordCommand;

	enum BlendMode { DefaultBlend, MaskBlend, BlendModeCount };

	constexpr uint32_t FramesInFlight = 2;
	constexpr uint32_t DescriptorsPerPool = 1024;
	constexpr int TextureUnits = 2;

	/// Uniforms of a quad, sent as push constants. This should match Graphics/Shaders/Vulkan/Uniforms.glsl
	struct Uniforms {
		float vertex_coords[4][4];
		float tex_coords[4][2];
		float tint[4];
	};
	static_assert(sizeof(Uniforms) == 112, "Uniforms does not match its shader counterpart");

	struct DescriptorPool {
		VkDescriptorPool pool;
		uint32_t used;
	};

	/// GL::Texture is an index to these slots. Texture name 0 is reserved as it denotes no texture.
	struct TextureSlot {
		VF::Image image{};
		VkImageView view = VK_NULL_HANDLE;
		VkFramebuffer framebuffer = VK_NULL_HANDLE; //only frame buffer textures have this
		VkDescriptorSet descriptor = VK_NULL_HANDLE;
		size_t pool = 0;
		Graphics::ColorMode mode = Graphics::ColorMode::Invalid;
		Geometry::Size size = {0, 0};
		VkFilter filter = VK_FILTER_LINEAR;
		bool inuse = false;
	};

	/// Pipelines depend on blending and are created when they are first used
	struct Program {
		std::vector<char> vertex, fragment;
		std::unique_ptr<VF::VulkanGraphicsPipeline> pipelines[BlendModeCount];
	};

	struct Frame {
		VkCommandBuffer commandbuffer = VK_NULL_HANDLE;
		VF::FrameSyncObjects sync{};
		uint64_t serial = 0;
	};

	struct WindowContext {
		VkSurfaceKHR surface = VK_NULL_HANDLE;
		VF::VulkanPresenter presenter{};
		std::vector<VkFramebuffer> framebuffers;

		//Presenting has no fence, thus a semaphore waited by it can only be reused with the same image
		std::vector<VkSemaphore> renderfinished;

		Frame frames[FramesInFlight];
		uint32_t currentframe = 0;
		uint32_t imageindex = 0;

		Geometry::Size size = {0, 0};
		bool outofdate = false;

		//command recording state, command buffer state does not outlive these
		bool recording = false;
		bool inpass = false;
		VF::VulkanGraphicsPipeline *pipeline = nullptr;
		VkDescriptorSet boundsets[TextureUnits] = {};
	};

	/// Resources are retired when they are no longer needed by Gorgon, but they could still be used by
	/// the commands in flight. They are destroyed after the commands up to their serial are completed.
	struct RetiredResource {
		uint64_t serial;
		std::function<void()> destroy;
	};

	struct Backend {
		VkInstance instance = VK_NULL_HANDLE;
		VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
		VF::VulkanDevice device{};
		VmaAllocator allocator = VK_NULL_HANDLE;
		VkCommandPool commandpool = VK_NULL_HANDLE;

		VkFormat targetformat = VK_FORMAT_UNDEFINED;
		VkRenderPass screenpass = VK_NULL_HANDLE;
		VkRenderPass texturepass = VK_NULL_HANDLE;

		VkSampler linearsampler = VK_NULL_HANDLE;
		VkSampler nearestsampler = VK_NULL_HANDLE;
		VkDescriptorSetLayout texturelayout = VK_NULL_HANDLE;
		VkPipelineLayout pipelinelayout = VK_NULL_HANDLE;

		std::vector<DescriptorPool> pools;
		std::vector<TextureSlot> textures;
		std::vector<Texture> freetextures;
		std::deque<Program> programs;
		std::vector<WindowContext *> contexts;

		std::deque<RetiredResource> retired;
		uint64_t lastsubmit = 0;
		uint64_t completed = 0;

		//The state that OpenGL would keep
		WindowContext *current = nullptr;
		Uniforms uniforms{};
		Program *program = nullptr;
		BlendMode blend = DefaultBlend;
		Texture activetextures[TextureUnits] = {};
		bool activated[TextureUnits] = {};
		Texture target = 0; //0 denotes the screen
		VkClearColorValue clearcolor{};
		bool clipenabled = false;
		Geometry::Rectangle cliprectangle = {0, 0, 0, 0};
	};

	/// This is not an object as textures and frame buffers could be destroyed after static destructors of
	/// this file are run. When this is null, all functions are ignored.
	static Backend *vk = nullptr;

	namespace {

		// All formats are linear: Gorgon blends in gamma space and its colors would be washed out with sRGB.
		const std::vector<VkFormat> surfaceformats = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};

#ifndef NDEBUG
		VKAPI_ATTR VkBool32 VKAPI_CALL debugproc(
			VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
			const VkDebugUtilsMessengerCallbackDataEXT *data, void *
		) {
			log << data->pMessage;

			if(severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
				std::cerr << "Vulkan: " << data->pMessage << std::endl;
			}

			return VK_FALSE;
		}

		bool hasvalidationlayer() {
			uint32_t count = 0;
			vkEnumerateInstanceLayerProperties(&count, nullptr);

			std::vector<VkLayerProperties> layers(count);
			vkEnumerateInstanceLayerProperties(&count, layers.data());

			for(auto &layer : layers) {
				if(VF::validationLayers[0] == std::string(layer.layerName)) {
					return true;
				}
			}

			return false;
		}
#endif

		void createinstance() {
			std::vector<const char *> extensions = {VK_KHR_SURFACE_EXTENSION_NAME, GetNativeSurfaceExtension()};
			std::vector<const char *> layers;

#ifndef NDEBUG
			bool validation = hasvalidationlayer();
			if(validation) {
				layers = VF::validationLayers;
				extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			}
#endif

			//Vulkan 1.1 is required for flipping the viewport
			vk->instance = VF::CreateInstance(extensions, layers, VK_API_VERSION_1_1, "Gorgon", "Gorgon");

#ifndef NDEBUG
			if(validation) {
				VkDebugUtilsMessengerCreateInfoEXT createinfo{};
				createinfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
				createinfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
				createinfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT;
				createinfo.pfnUserCallback = &debugproc;

				auto create = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk->instance, "vkCreateDebugUtilsMessengerEXT"));
				if(create) {
					create(vk->instance, &createinfo, nullptr, &vk->messenger);
				}
			}
#endif
		}

		/// Creates the objects that are shared between windows. Needs a surface as it decides the device.
		void createdevice(VkSurfaceKHR surface) {
			auto &device = vk->device;

			device = VF::SetupDevice(vk->instance, surface);
			vk->allocator = VF::CreateAllocator(vk->instance, device);
			vk->commandpool = VF::CreateCommandPool(device, VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

			//Textures that are rendered to and the screen use the same format so that the same
			//pipelines can be used for both. Compatible render passes can only differ in layout.
			vk->targetformat = VF::SwapchainHelpers::ChooseSurfaceFormats(device.physicalDevice, surface, surfaceformats).format;
			vk->screenpass = VF::CreateColorRenderpass(device.logicalDevice, vk->targetformat, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
			vk->texturepass = VF::CreateColorRenderpass(device.logicalDevice, vk->targetformat, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

			vk->linearsampler = VF::CreateTextureSampler(device, VK_FILTER_LINEAR);
			vk->nearestsampler = VF::CreateTextureSampler(device, VK_FILTER_NEAREST);

			VF::VulkanDescriptorSetLayout texturelayout{};
			texturelayout.AddBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
			texturelayout.Build(device);
			vk->texturelayout = texturelayout.layout;

			//Every texture unit is a descriptor set using the same layout. As all pipelines share this
			//layout, bound textures and uniforms stay valid while switching shaders, just like in OpenGL.
			VkDescriptorSetLayout setlayouts[TextureUnits] = {vk->texturelayout, vk->texturelayout};

			VkPushConstantRange pushconstants{};
			pushconstants.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
			pushconstants.offset = 0;
			pushconstants.size = sizeof(Uniforms);

			VkPipelineLayoutCreateInfo layoutinfo{};
			layoutinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
			layoutinfo.setLayoutCount = TextureUnits;
			layoutinfo.pSetLayouts = setlayouts;
			layoutinfo.pushConstantRangeCount = 1;
			layoutinfo.pPushConstantRanges = &pushconstants;

			if(vkCreatePipelineLayout(device.logicalDevice, &layoutinfo, nullptr, &vk->pipelinelayout) != VK_SUCCESS) {
				throw std::runtime_error("Cannot create pipeline layout");
			}
		}

		void retire(std::function<void()> destroy) {
			vk->retired.push_back({vk->lastsubmit + 1, std::move(destroy)});
		}

		void destroyretired() {
			while(!vk->retired.empty() && vk->retired.front().serial <= vk->completed) {
				vk->retired.front().destroy();
				vk->retired.pop_front();
			}
		}

		void waitidle() {
			vkDeviceWaitIdle(vk->device.logicalDevice);
			vk->completed = vk->lastsubmit;
		}

		VkDescriptorSet allocatedescriptor(size_t &poolindex) {
			auto &device = vk->device;

			poolindex = 0;
			while(poolindex < vk->pools.size() && vk->pools[poolindex].used == DescriptorsPerPool) {
				poolindex++;
			}

			if(poolindex == vk->pools.size()) {
				vk->pools.push_back({
					VF::CreateDescriptorPool(device, {VF::CreatePoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, DescriptorsPerPool)}, DescriptorsPerPool),
					0
				});
			}

			vk->pools[poolindex].used++;

			return VF::CreateDescriptorSet(1, device, vk->pools[poolindex].pool, {vk->texturelayout})[0];
		}

		/// A descriptor that is bound to a command buffer cannot be updated, thus it is replaced instead.
		void createdescriptor(TextureSlot &slot) {
			VkDescriptorImageInfo imageinfo{};
			imageinfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			imageinfo.imageView = slot.view;
			imageinfo.sampler = slot.filter == VK_FILTER_NEAREST ? vk->nearestsampler : vk->linearsampler;

			slot.descriptor = allocatedescriptor(slot.pool);

			VF::UpdateDescriptorSets(vk->device, {
				VF::CreateWriteDescriptorSet(slot.descriptor, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, 0, 1, nullptr, &imageinfo)
			});
		}

		void retiredescriptor(TextureSlot &slot) {
			if(!slot.descriptor) return;

			retire([descriptor = slot.descriptor, pool = slot.pool]() {
				vkFreeDescriptorSets(vk->device.logicalDevice, vk->pools[pool].pool, 1, &descriptor);
				vk->pools[pool].used--;
			});

			slot.descriptor = VK_NULL_HANDLE;
		}

		/// Texture formats are the same as the colors they store, with the exception of 24bit colors which
		/// are not widely supported.
		VkFormat getformat(Graphics::ColorMode mode) {
			switch(mode) {
			case Graphics::ColorMode::Alpha:
			case Graphics::ColorMode::Grayscale:
				return VK_FORMAT_R8_UNORM;
			case Graphics::ColorMode::Grayscale_Alpha:
				return VK_FORMAT_R8G8_UNORM;
			case Graphics::ColorMode::BGR:
			case Graphics::ColorMode::BGRA:
				return VK_FORMAT_B8G8R8A8_UNORM;
			default:
				return VK_FORMAT_R8G8B8A8_UNORM;
			}
		}

		Graphics::ColorMode getstoragemode(Graphics::ColorMode mode) {
			switch(mode) {
			case Graphics::ColorMode::RGB:
			case Graphics::ColorMode::Invalid:
				return Graphics::ColorMode::RGBA;
			case Graphics::ColorMode::BGR:
				return Graphics::ColorMode::BGRA;
			default:
				return mode;
			}
		}

		/// Maps the channels stored in textures to what OpenGL would return for the same color mode.
		VkComponentMapping getswizzle(Graphics::ColorMode mode) {
			const auto r = VK_COMPONENT_SWIZZLE_R, g = VK_COMPONENT_SWIZZLE_G;
			const auto zero = VK_COMPONENT_SWIZZLE_ZERO, one = VK_COMPONENT_SWIZZLE_ONE;

			switch(mode) {
			case Graphics::ColorMode::Alpha:
				return {zero, zero, zero, r};
			case Graphics::ColorMode::Grayscale:
				return {r, r, r, one};
			case Graphics::ColorMode::Grayscale_Alpha:
				return {r, r, r, g};
			default:
				return {};
			}
		}

		void createimage(TextureSlot &slot, const Geometry::Size &size, Graphics::ColorMode mode, VkFilter filter, bool rendertarget) {
			//Vulkan does not support empty images, OpenGL does
			VkExtent2D extent = {(uint32_t)std::max(size.Width, 1), (uint32_t)std::max(size.Height, 1)};

			slot.size = size;
			slot.mode = mode;
			slot.filter = filter;

			if(rendertarget) {
				auto target = VF::CreateOffscreenTarget(vk->device, vk->allocator, vk->commandpool, vk->texturepass, vk->targetformat, extent);

				slot.image = target.image;
				slot.view = target.imageView;
				slot.framebuffer = target.frameBuffer;
			}
			else {
				auto format = getformat(mode);

				slot.image = VF::CreateEmptyTextureImage(vk->device, vk->allocator, vk->commandpool, format, extent);
				slot.view = VF::SwapchainHelpers::CreateImageView(vk->device.logicalDevice, slot.image.image, format, VK_IMAGE_ASPECT_COLOR_BIT, getswizzle(mode));
				slot.framebuffer = VK_NULL_HANDLE;
			}

			createdescriptor(slot);
		}

		void retireimage(TextureSlot &slot) {
			retiredescriptor(slot);

			if(!slot.image.image) return;

			retire([image = slot.image, view = slot.view, framebuffer = slot.framebuffer]() {
				if(framebuffer) {
					vkDestroyFramebuffer(vk->device.logicalDevice, framebuffer, nullptr);
				}
				vkDestroyImageView(vk->device.logicalDevice, view, nullptr);
				vmaDestroyImage(vk->allocator, image.image, image.deviceMemory);
			});

			slot.image = {};
			slot.view = VK_NULL_HANDLE;
			slot.framebuffer = VK_NULL_HANDLE;
		}

		Texture allocatetexture() {
			Texture tex;

			if(!vk->freetextures.empty()) {
				tex = vk->freetextures.back();
				vk->freetextures.pop_back();
			}
			else {
				tex = (Texture)vk->textures.size();
				vk->textures.emplace_back();
			}

			vk->textures[tex].inuse = true;

			return tex;
		}

		/// Texture 0 is sampled when there is no texture. OpenGL returns opaque black in this case.
		void createnulltexture() {
			const Byte black[4] = {0, 0, 0, 255};

			vk->textures.emplace_back();
			auto &slot = vk->textures[0];
			slot.inuse = true;
			createimage(slot, {1, 1}, Graphics::ColorMode::RGBA, VK_FILTER_NEAREST, false);

			VF::UpdateTextureImage(vk->device, vk->allocator, vk->commandpool, slot.image.image, black, sizeof(black), {0, 0}, {1, 1});
		}

		bool isvalidtexture(Texture tex) {
			return tex < vk->textures.size() && vk->textures[tex].inuse;
		}

		void convertpixel(const Byte *source, Graphics::ColorMode sourcemode, Byte *dest, Graphics::ColorMode destmode) {
			Byte r = 0, g = 0, b = 0, a = 255;

			switch(sourcemode) {
			case Graphics::ColorMode::Alpha:
				a = source[0];
				break;
			case Graphics::ColorMode::Grayscale:
				r = g = b = source[0];
				break;
			case Graphics::ColorMode::Grayscale_Alpha:
				r = g = b = source[0];
				a = source[1];
				break;
			case Graphics::ColorMode::RGB:
			case Graphics::ColorMode::RGBA:
				r = source[0]; g = source[1]; b = source[2];
				if(sourcemode == Graphics::ColorMode::RGBA) a = source[3];
				break;
			case Graphics::ColorMode::BGR:
			case Graphics::ColorMode::BGRA:
				b = source[0]; g = source[1]; r = source[2];
				if(sourcemode == Graphics::ColorMode::BGRA) a = source[3];
				break;
			default:
				break;
			}

			switch(destmode) {
			case Graphics::ColorMode::Alpha:
				dest[0] = a;
				break;
			case Graphics::ColorMode::Grayscale:
				dest[0] = r;
				break;
			case Graphics::ColorMode::Grayscale_Alpha:
				dest[0] = r;
				dest[1] = a;
				break;
			case Graphics::ColorMode::BGRA:
				dest[0] = b; dest[1] = g; dest[2] = r; dest[3] = a;
				break;
			default:
				dest[0] = r; dest[1] = g; dest[2] = b; dest[3] = a;
				break;
			}
		}

		/// Copies the given part of the image to the given part of the texture.
		void uploadpixels(TextureSlot &slot, const Containers::Image &data, const Geometry::Bounds &source, const Geometry::Point &target) {
			int w = source.Width(), h = source.Height();

			if(w <= 0 || h <= 0) return;

			if(target.X < 0 || target.Y < 0 || target.X + w > slot.size.Width || target.Y + h > slot.size.Height ||
			   source.Left < 0 || source.Top < 0 || source.Right > data.GetWidth() || source.Bottom > data.GetHeight()) {
				log << "Texture copy is out of bounds and it is ignored.";
				return;
			}

			auto sourcemode = data.GetMode();
			auto destmode = getstoragemode(slot.mode);

			//frame buffer textures use the format of the screen instead of their color mode
			if(slot.framebuffer && vk->targetformat == VK_FORMAT_B8G8R8A8_UNORM) {
				destmode = Graphics::ColorMode::BGRA;
			}

			size_t sourcebpp = Graphics::GetChannelsPerPixel(sourcemode);
			size_t destbpp = Graphics::GetChannelsPerPixel(destmode);
			size_t sourcestride = sourcebpp * data.GetWidth();

			std::vector<Byte> pixels(destbpp * w * h);

			const Byte *sourcerow = data.RawData() + sourcebpp * source.Left + sourcestride * source.Top;
			Byte *destrow = pixels.data();

			for(int y = 0; y < h; y++) {
				if(sourcemode == destmode) {
					std::memcpy(destrow, sourcerow, destbpp * w);
				}
				else {
					for(int x = 0; x < w; x++) {
						convertpixel(sourcerow + x * sourcebpp, sourcemode, destrow + x * destbpp, destmode);
					}
				}

				sourcerow += sourcestride;
				destrow += destbpp * w;
			}

			VF::UpdateTextureImage(
				vk->device, vk->allocator, vk->commandpool, slot.image.image, pixels.data(), pixels.size(),
				{target.X, target.Y}, {(uint32_t)w, (uint32_t)h}
			);

			//all submitted commands are completed at this point
			vk->completed = vk->lastsubmit;
		}

		void settexturedata(Texture tex, const Containers::Image &data) {
			auto &slot = vk->textures[tex];
			auto mode = data.GetMode();

			if(!slot.image.image || slot.size != data.GetSize() || slot.mode != mode) {
				bool rendertarget = slot.framebuffer != VK_NULL_HANDLE;

				retireimage(slot);
				createimage(slot, data.GetSize(), mode, VK_FILTER_LINEAR, rendertarget);
			}
			else if(slot.filter != VK_FILTER_LINEAR) {
				retiredescriptor(slot);
				slot.filter = VK_FILTER_LINEAR;
				createdescriptor(slot);
			}

			uploadpixels(slot, data, {0, 0, data.GetWidth(), data.GetHeight()}, {0, 0});
		}

		void destroyswapchain(WindowContext &ctx, bool destroyhandle) {
			auto &device = vk->device.logicalDevice;

			for(auto framebuffer : ctx.framebuffers) {
				vkDestroyFramebuffer(device, framebuffer, nullptr);
			}
			ctx.framebuffers.clear();

			for(auto semaphore : ctx.renderfinished) {
				vkDestroySemaphore(device, semaphore, nullptr);
			}
			ctx.renderfinished.clear();

			for(auto view : ctx.presenter.swapChainImageViews) {
				vkDestroyImageView(device, view, nullptr);
			}
			ctx.presenter.swapChainImageViews.clear();

			if(destroyhandle && ctx.presenter.swapChain) {
				vkDestroySwapchainKHR(device, ctx.presenter.swapChain, nullptr);
				ctx.presenter.swapChain = VK_NULL_HANDLE;
			}
		}

		/// Creates or recreates the swapchain. A window that has no area cannot have a swapchain, if that is
		/// the case, this function will be tried again on the next frame.
		void createswapchain(WindowContext &ctx) {
			waitidle();

			auto caps = VF::SwapchainHelpers::GetSurfaceCapabilities(vk->device.physicalDevice, ctx.surface);
			auto extent = VF::SwapchainHelpers::GetImageExtent(VkExtent2D{(uint32_t)ctx.size.Width, (uint32_t)ctx.size.Height}, caps);

			if(extent.width == 0 || extent.height == 0) {
				destroyswapchain(ctx, true);
				ctx.outofdate = true;
				return;
			}

			auto old = ctx.presenter.swapChain;
			destroyswapchain(ctx, false);

			ctx.presenter = VF::SetupPresenter(vk->device, extent, ctx.surface, surfaceformats, old);

			if(old) {
				vkDestroySwapchainKHR(vk->device.logicalDevice, old, nullptr);
			}

			if(ctx.presenter.swapchainProperties.surfaceFormat.format != vk->targetformat) {
				throw std::runtime_error("Windows with different surface formats are not supported");
			}

			extent = ctx.presenter.swapchainProperties.imageExtent;
			for(auto view : ctx.presenter.swapChainImageViews) {
				ctx.framebuffers.push_back(VF::CreateColorFramebuffer(vk->device.logicalDevice, vk->screenpass, view, extent));
				ctx.renderfinished.push_back(VF::CreateBinarySemaphore(vk->device));
			}

			ctx.outofdate = false;

			log << "Swapchain created: " << extent.width << "x" << extent.height << ", " << ctx.framebuffers.size() << " images";
		}

		bool acquireimage(WindowContext &ctx, Frame &frame) {
			auto result = VF::TryAcquireNextImage(vk->device, ctx.presenter, frame.sync.imageAvailableSemaphore, &ctx.imageindex);

			//suboptimal image can still be used, swapchain will be recreated after presenting it
			return result == VK_SUCCESS || result == VK_SUBOPTIMAL_KHR;
		}

		/// Returns the command buffer of the current frame, starts the frame if it is not already started.
		/// Returns nullptr if the current window cannot be drawn to.
		VkCommandBuffer beginframe() {
			if(!vk || !vk->current) return VK_NULL_HANDLE;

			auto &ctx = *vk->current;
			auto &frame = ctx.frames[ctx.currentframe];

			if(ctx.recording) return frame.commandbuffer;

			vkWaitForFences(vk->device.logicalDevice, 1, &frame.sync.inFlightFence, VK_TRUE, UINT64_MAX);
			vk->completed = std::max(vk->completed, frame.serial);
			destroyretired();

			if(ctx.outofdate || !ctx.presenter.swapChain) {
				createswapchain(ctx);

				if(!ctx.presenter.swapChain) return VK_NULL_HANDLE;
			}

			if(!acquireimage(ctx, frame)) {
				createswapchain(ctx);

				if(!ctx.presenter.swapChain || !acquireimage(ctx, frame)) return VK_NULL_HANDLE;
			}

			//Fence should only be reset if there is going to be a submit, otherwise next wait will never end.
			vkResetFences(vk->device.logicalDevice, 1, &frame.sync.inFlightFence);

			vkResetCommandBuffer(frame.commandbuffer, 0);
			RC::BeginCommandBuffer(frame.commandbuffer);

			SC::recordTransitionImageLayout(
				frame.commandbuffer, ctx.presenter.swapChainImages[ctx.imageindex],
				VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
			);

			ctx.recording = true;
			ctx.inpass = false;
			ctx.pipeline = nullptr;
			std::fill(std::begin(ctx.boundsets), std::end(ctx.boundsets), (VkDescriptorSet)VK_NULL_HANDLE);

			return frame.commandbuffer;
		}

		bool istargetvalid() {
			return vk->target && isvalidtexture(vk->target) && vk->textures[vk->target].framebuffer;
		}

		VkExtent2D gettargetextent(WindowContext &ctx) {
			if(istargetvalid()) {
				auto &size = vk->textures[vk->target].size;

				return {(uint32_t)std::max(size.Width, 1), (uint32_t)std::max(size.Height, 1)};
			}

			return ctx.presenter.swapchainProperties.imageExtent;
		}

		/// Vulkan requires the scissor to be inside the target, OpenGL does not. Additionally, clip rectangle
		/// uses the bottom left of the target as its origin while the screen is flipped. Frame buffers are 
		/// not flipped.
		VkRect2D getscissor(WindowContext &ctx) {
			auto extent = gettargetextent(ctx);
			int left = 0, top = 0, right = extent.width, bottom = extent.height;

			if(vk->clipenabled) {
				auto &clip = vk->cliprectangle;
				int y = istargetvalid() ? clip.Y : (int)extent.height - (clip.Y + clip.Height);

				left = std::max(left, clip.X);
				top = std::max(top, y);
				right = std::min(right, clip.X + clip.Width);
				bottom = std::min(bottom, y + clip.Height);
			}

			return {{left, top}, {(uint32_t)std::max(right - left, 0), (uint32_t)std::max(bottom - top, 0)}};
		}

		/// Gorgon uses OpenGL clip space and texture coordinates, where the first row of a frame buffer is at
		/// the bottom. Keeping frame buffers unflipped retains this, making their texture coordinates and mask
		/// coordinates correct. Viewport covers the window even if the frame buffer is larger, as in OpenGL.
		VkViewport getviewport(WindowContext &ctx) {
			float w = (float)std::max(ctx.size.Width, 1), h = (float)std::max(ctx.size.Height, 1);

			if(istargetvalid()) {
				return {0, 0, w, h, 0, 1};
			}

			//Origin is the bottom left of the screen, window size could be different for a short time while resizing
			return {0, (float)ctx.presenter.swapchainProperties.imageExtent.height, w, -h, 0, 1};
		}

		void beginpass(WindowContext &ctx, VkCommandBuffer commandbuffer) {
			if(ctx.inpass) return;

			if(istargetvalid()) {
				RC::BeginColorRenderPass(commandbuffer, vk->texturepass, vk->textures[vk->target].framebuffer, gettargetextent(ctx));
			}
			else {
				RC::BeginColorRenderPass(commandbuffer, vk->screenpass, ctx.framebuffers[ctx.imageindex], ctx.presenter.swapchainProperties.imageExtent);
			}

			ctx.inpass = true;

			RC::SetViewport(commandbuffer, getviewport(ctx));
		}

		void endpass(WindowContext &ctx) {
			if(!ctx.inpass) return;

			RC::EndRenderPass(ctx.frames[ctx.currentframe].commandbuffer);
			ctx.inpass = false;
		}

		void settarget(Texture target) {
			if(!vk || vk->target == target) return;

			if(vk->current) {
				endpass(*vk->current);
			}

			vk->target = target;
		}

		VF::VulkanGraphicsPipeline &getpipeline(Program &program, BlendMode blend) {
			auto &pipeline = program.pipelines[blend];

			if(pipeline) return *pipeline;

			pipeline = std::make_unique<VF::VulkanGraphicsPipeline>();

			VF::VulkanRenderPass renderpass{};
			renderpass.renderPass = vk->screenpass;
			pipeline->InitDefaultPipeline(vk->current->presenter, renderpass);

			//Quads are flat and drawn in order, they can also be rotated to face any direction
			auto &states = pipeline->__States;
			states.DepthStencilState.depthTestEnable = VK_FALSE;
			states.DepthStencilState.depthWriteEnable = VK_FALSE;
			states.RasterizationState.cullMode = VK_CULL_MODE_NONE;

			auto &colorblend = states.colorBlendAttachement;
			colorblend.colorBlendOp = VK_BLEND_OP_ADD;
			colorblend.alphaBlendOp = VK_BLEND_OP_ADD;
			if(blend == MaskBlend) {
				colorblend.srcColorBlendFactor = VK_BLEND_FACTOR_DST_COLOR;
				colorblend.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
				colorblend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
				colorblend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
			}
			else {
				colorblend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
				colorblend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
				colorblend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
				colorblend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
			}

			pipeline->UseSharedPipelineLayout(vk->pipelinelayout);
			pipeline->AddShaderStageFromSPIRV(vk->device, program.vertex, VK_SHADER_STAGE_VERTEX_BIT);
			pipeline->AddShaderStageFromSPIRV(vk->device, program.fragment, VK_SHADER_STAGE_FRAGMENT_BIT);
			pipeline->BuildPipeline(vk->device, VK_SAMPLE_COUNT_1_BIT);

			return *pipeline;
		}

		void destroycontextframes(WindowContext &ctx) {
			auto &device = vk->device.logicalDevice;

			for(auto &frame : ctx.frames) {
				vkDestroySemaphore(device, frame.sync.imageAvailableSemaphore, nullptr);
				vkDestroySemaphore(device, frame.sync.renderFinishedSemaphore, nullptr);
				vkDestroyFence(device, frame.sync.inFlightFence, nullptr);
				vkFreeCommandBuffers(device, vk->commandpool, 1, &frame.commandbuffer);
			}
		}

		void destroycontext(WindowContext *ctx) {
			waitidle();

			destroyswapchain(*ctx, true);
			destroycontextframes(*ctx);
			vkDestroySurfaceKHR(vk->instance, ctx->surface, nullptr);

			delete ctx;
		}

		void destroyprogram(Program &program) {
			for(auto &pipeline : program.pipelines) {
				if(pipeline) {
					vkDestroyPipeline(vk->device.logicalDevice, pipeline->pipeline, nullptr);
					pipeline.reset();
				}
			}
		}

		/// Runs at exit: static data of the drivers and layers could already be destroyed by this point, so
		/// it is not safe to use Vulkan anymore. Just like the OpenGL contexts of Gorgon, everything is left to
		/// the operating system. Global windows and textures that are destroyed afterwards will have no effect.
		void abandon() {
			vk = nullptr;
		}

	}

	void Shutdown() {
		if(!vk) return;

		auto &device = vk->device.logicalDevice;

		if(device) {
			waitidle();

			for(auto ctx : vk->contexts) {
				destroycontext(ctx);
			}

			for(auto &program : vk->programs) {
				destroyprogram(program);
			}

			while(!vk->retired.empty()) {
				vk->retired.front().destroy();
				vk->retired.pop_front();
			}

			for(auto &slot : vk->textures) {
				if(slot.image.image) {
					if(slot.framebuffer) {
						vkDestroyFramebuffer(device, slot.framebuffer, nullptr);
					}
					vkDestroyImageView(device, slot.view, nullptr);
					vmaDestroyImage(vk->allocator, slot.image.image, slot.image.deviceMemory);
				}
			}

			for(auto &pool : vk->pools) {
				vkDestroyDescriptorPool(device, pool.pool, nullptr);
			}

			vkDestroyPipelineLayout(device, vk->pipelinelayout, nullptr);
			vkDestroyDescriptorSetLayout(device, vk->texturelayout, nullptr);
			vkDestroySampler(device, vk->linearsampler, nullptr);
			vkDestroySampler(device, vk->nearestsampler, nullptr);
			vkDestroyRenderPass(device, vk->screenpass, nullptr);
			vkDestroyRenderPass(device, vk->texturepass, nullptr);
			vkDestroyCommandPool(device, vk->commandpool, nullptr);
			vmaDestroyAllocator(vk->allocator);
			vkDestroyDevice(device, nullptr);
		}

		if(vk->messenger) {
			auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(vkGetInstanceProcAddr(vk->instance, "vkDestroyDebugUtilsMessengerEXT"));
			if(destroy) {
				destroy(vk->instance, vk->messenger, nullptr);
			}
		}

		vkDestroyInstance(vk->instance, nullptr);

		delete vk;
		vk = nullptr;
	}

	WindowContext *CreateWindowContext(void *display, unsigned long window, const Geometry::Size &size) {
		bool first = !vk;

		if(first) {
			vk = new Backend;
			createinstance();
		}

		auto ctx = new WindowContext;
		ctx->size = size;
		ctx->surface = CreateNativeSurface(vk->instance, display, window);

		if(first) {
			createdevice(ctx->surface);
			createnulltexture();

			//should be registered after drivers and layers are loaded so that it runs before their cleanup
			std::atexit(&abandon);
		}

		VkBool32 presentsupport = VK_FALSE;
		vkGetPhysicalDeviceSurfaceSupportKHR(vk->device.physicalDevice, vk->device.familyIndices.presentationFamily.value(), ctx->surface, &presentsupport);
		if(!presentsupport) {
			throw std::runtime_error("Selected graphics device cannot present to this window");
		}

		auto commandbuffers = VF::CreateCommandBuffers(FramesInFlight, vk->device, vk->commandpool);
		for(uint32_t i = 0; i < FramesInFlight; i++) {
			ctx->frames[i].commandbuffer = commandbuffers[i];
			ctx->frames[i].sync = VF::CreateSyncObjects(vk->device);
		}

		createswapchain(*ctx);

		vk->contexts.push_back(ctx);

		return ctx;
	}

	void DestroyWindowContext(WindowContext *context) {
		if(!vk || !context) return;

		if(vk->current == context) {
			vk->current = nullptr;
		}

		vk->contexts.erase(std::remove(vk->contexts.begin(), vk->contexts.end(), context), vk->contexts.end());

		destroycontext(context);
	}

	void MakeCurrent(WindowContext *context) {
		if(vk) {
			vk->current = context;
		}
	}

	void Present() {
		if(!vk || !vk->current || !vk->current->recording) return;

		auto &ctx = *vk->current;
		auto &frame = ctx.frames[ctx.currentframe];

		endpass(ctx);

		SC::recordTransitionImageLayout(
			frame.commandbuffer, ctx.presenter.swapChainImages[ctx.imageindex],
			VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
		);

		if(RC::EndCommandBuffer(frame.commandbuffer) != VK_SUCCESS) {
			throw std::runtime_error("Cannot record the commands of the frame");
		}

		VF::GraphicsQueueSubmit(
			vk->device, frame.sync, {frame.commandbuffer},
			{frame.sync.imageAvailableSemaphore}, {ctx.renderfinished[ctx.imageindex]}
		);

		frame.serial = ++vk->lastsubmit;
		ctx.recording = false;

		auto result = VF::PresentQueueSubmit(vk->device, ctx.presenter, ctx.imageindex, {ctx.renderfinished[ctx.imageindex]});

		if(result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
			ctx.outofdate = true;
		}
		else if(result != VK_SUCCESS) {
			log << "Cannot present the frame: " << (int)result;
		}

		ctx.currentframe = (ctx.currentframe + 1) % FramesInFlight;
	}

	void Initialize() {
		FrameBuffer::HardwareSupport = true;

		log << "Vulkan initialized";
	}

	Texture GenerateTexture(const Containers::Image &data) {
		if(!vk) return 0;

		auto tex = allocatetexture();

		settexturedata(tex, data);

		return tex;
	}

	Texture GenerateEmptyTexture(const Geometry::Size &size, Graphics::ColorMode mode) {
		if(!vk) return 0;

		auto tex = allocatetexture();

		createimage(vk->textures[tex], size, mode, VK_FILTER_NEAREST, false);

		return tex;
	}

	void ResizeTexture(Texture tex, const Geometry::Size &size, Graphics::ColorMode mode) {
		if(!vk || tex == 0 || !isvalidtexture(tex)) return;

		auto &slot = vk->textures[tex];

		if(slot.size == size && slot.mode == mode) return;

		bool rendertarget = slot.framebuffer != VK_NULL_HANDLE;
		auto filter = slot.filter;

		retireimage(slot);
		createimage(slot, size, mode, filter, rendertarget);
	}

	void UpdateTexture(Texture tex, const Containers::Image &data) {
		if(!vk || tex == 0 || !isvalidtexture(tex)) return;

		settexturedata(tex, data);
	}

	void CopyToTexture(Texture tex, const Containers::Image &data, Geometry::Point target) {
		if(!vk || tex == 0 || !isvalidtexture(tex)) return;

		uploadpixels(vk->textures[tex], data, {0, 0, data.GetWidth(), data.GetHeight()}, target);
	}

	void CopyToTexture(Texture tex, const Containers::Image &data, Geometry::Bounds source, Geometry::Point target) {
		if(!vk || tex == 0 || !isvalidtexture(tex)) return;

		uploadpixels(vk->textures[tex], data, source, target);
	}

	void DestroyTexture(Texture tex) {
		if(!vk || tex == 0 || !isvalidtexture(tex)) return;

		auto &slot = vk->textures[tex];

		retireimage(slot);
		slot = TextureSlot{};
		vk->freetextures.push_back(tex);
	}

	void ActivateTexture(Texture tex, int unit) {
		if(!vk) return;

		vk->activetextures[unit] = tex;
		vk->activated[unit] = true;
	}

	void RenderToTexture(FrameBuffer &buffer) {
		buffer.Use();
	}

	void RenderToScreen() {
		settarget(0);
	}

	void SetupContext(const Geometry::Size &size) {
		SetDefaultClear();
		SetDefaultBlending();

		Resize(size);
	}

	void Resize(const Geometry::Size &size) {
		if(!vk || !vk->current) return;

		auto &ctx = *vk->current;
		auto &extent = ctx.presenter.swapchainProperties.imageExtent;

		ctx.size = size;

		if(extent.width != (uint32_t)size.Width || extent.height != (uint32_t)size.Height) {
			ctx.outofdate = true;
		}
	}

	void Clear() {
		auto commandbuffer = beginframe();

		if(!commandbuffer) return;

		auto &ctx = *vk->current;

		beginpass(ctx, commandbuffer);

		auto scissor = getscissor(ctx);

		if(scissor.extent.width && scissor.extent.height) {
			RC::ClearColorAttachment(commandbuffer, vk->clearcolor, scissor);
		}
	}

	void SetDefaultClear() {
		SetClearColor(0.4f, 0.2f, 0.0f, 0.0f);
	}

	void SetClearColor(float r, float g, float b, float a) {
		if(!vk) return;

		vk->clearcolor.float32[0] = r;
		vk->clearcolor.float32[1] = g;
		vk->clearcolor.float32[2] = b;
		vk->clearcolor.float32[3] = a;
	}

	void SetDefaultBlending() {
		if(vk) {
			vk->blend = DefaultBlend;
		}
	}

	void SetMaskBlending() {
		if(vk) {
			vk->blend = MaskBlend;
		}
	}

	bool IsClipEnabled() {
		return vk && vk->clipenabled;
	}

	void SetClipEnabled(bool enabled) {
		if(vk) {
			vk->clipenabled = enabled;
		}
	}

	void SetClipRectangle(const Geometry::Rectangle &rectangle) {
		if(vk) {
			vk->cliprectangle = rectangle;
		}
	}

	void SetupQuadVertices() {
		//there are no vertex buffers, vertices are selected by the shaders
	}

	void ActivateQuadVertices() {
	}

	void DrawQuadVertices() {
		if(!vk || !vk->program) return;

		auto commandbuffer = beginframe();

		if(!commandbuffer) return;

		auto &ctx = *vk->current;

		beginpass(ctx, commandbuffer);

		auto &pipeline = getpipeline(*vk->program, vk->blend);

		if(ctx.pipeline != &pipeline) {
			RC::BindPipeline(pipeline, commandbuffer);
			ctx.pipeline = &pipeline;
		}

		//only the textures selected for this draw are bound, others could be in use as a render target
		for(int unit = 0; unit < TextureUnits; unit++) {
			if(!vk->activated[unit]) continue;

			vk->activated[unit] = false;

			auto tex = vk->activetextures[unit];
			auto descriptor = vk->textures[isvalidtexture(tex) ? tex : 0].descriptor;

			if(ctx.boundsets[unit] != descriptor) {
				RC::BindDescriptorSets(commandbuffer, pipeline, {descriptor}, unit);
				ctx.boundsets[unit] = descriptor;
			}
		}

		RC::SetScissor(commandbuffer, getscissor(ctx));

		RC::PushConstants(
			commandbuffer, vk->pipelinelayout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
			0, sizeof(Uniforms), &vk->uniforms
		);

		RC::Draw(commandbuffer, 6);
	}


	void FrameBuffer::Generate(bool gendepth) {
		if(!vk) return;

		buffers.Add(this);

		texture = allocatetexture();
		createimage(vk->textures[texture], Window::GetMinimumRequiredSize(), Graphics::ColorMode::RGBA, VK_FILTER_NEAREST, true);

		//Texture name is enough to identify the render target. Depth is not created: it has no effect as all
		//surfaces are coplanar, Gorgon draws them in order.
		buffer = texture;

		log << "Frame buffer texture name " << texture << ", size " << vk->textures[texture].size;
	}

	void FrameBuffer::UpdateSizes() {
		auto sz = Window::GetMinimumRequiredSize();

		log << "Frame buffers are resized to " << sz;

		for(FrameBuffer &buffer : buffers) {
			ResizeTexture(buffer.GetTexture(), sz, Graphics::ColorMode::RGBA);
		}
	}

	void FrameBuffer::Destroy() {
		if(!buffer) return;

		if(vk && vk->target == buffer) {
			settarget(0);
		}

		DestroyTexture(texture);

		buffer = 0;
		texture = 0;
	}

	bool FrameBuffer::IsFunctional() const {
		return vk != nullptr;
	}

	void FrameBuffer::Use() {
		if(!buffer) return;

		settarget(buffer);
	}

	void FrameBuffer::RenderToScreen() {
		GL::RenderToScreen();
	}

	bool FrameBuffer::HardwareSupport = false;
	Containers::Collection<FrameBuffer> FrameBuffer::buffers;


	Shader::~Shader() {
		//Pipelines belong to the backend. Shaders are static objects, Vulkan may not be available at this point.
	}

	void Shader::Use() {
		if(vk && program) {
			vk->program = &vk->programs[program - 1];
		}
	}

	int Shader::LocateUniform(const std::string &name) {
		if(name == "vertex_coords") return offsetof(Uniforms, vertex_coords);
		if(name == "tex_coords") return offsetof(Uniforms, tex_coords);
		if(name == "tint") return offsetof(Uniforms, tint);

		return -1;
	}

	int Shader::BindTexture(const std::string &name, int location) {
		//texture units are fixed in the shaders
		return location;
	}

	namespace {
		void updateuniform(int offset, const void *data, size_t size) {
			if(!vk || offset < 0 || offset + size > sizeof(Uniforms)) return;

			std::memcpy(reinterpret_cast<Byte *>(&vk->uniforms) + offset, data, size);
		}
	}

	void Shader::UpdateUniform(int name, float value) {
		updateuniform(name, &value, sizeof(value));
	}

	void Shader::UpdateUniform(int name, int value) {
		updateuniform(name, &value, sizeof(value));
	}

	void Shader::UpdateUniform(int name, const QuadVertices &value) {
		if(!vk || name != (int)offsetof(Uniforms, vertex_coords)) return;

		//push constants have 16 byte stride for vec3 arrays
		for(int i = 0; i < 4; i++) {
			std::memcpy(vk->uniforms.vertex_coords[i], value.Data + i * 3, 3 * sizeof(float));
		}
	}

	void Shader::UpdateUniform(int name, const QuadTextureCoords &value) {
		updateuniform(name, value.Data, sizeof(value.Data));
	}

	void Shader::UpdateUniform(int name, const Graphics::RGBAf &value) {
		updateuniform(name, value.Vector, sizeof(value.Vector));
	}

	void Shader::UpdateUniform(int name, const Geometry::Point3D &value) {
		updateuniform(name, value.Vector, 3 * sizeof(float));
	}

	void Shader::InitializeFromFiles(const std::string &vertexfile, const std::string &fragmentfile,
		const std::string &geometryfile, std::map<std::string, std::string> defines) {

		auto read = [](const std::string &filename) -> std::string {
			if(filename == "") return "";

			std::ifstream file(filename, std::ios::binary);

			if(!file.is_open()) {
				throw std::runtime_error("Cannot open shader file: " + filename);
			}

			return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
		};

		InitializeWithSource(read(vertexfile), read(fragmentfile), read(geometryfile), defines);
	}

	/// Sources are in compiled SPIR-V form as they are compiled during the build.
	void Shader::InitializeWithSource(std::string vertexsrc, std::string fragmentsrc,
		std::string geometrysrc, std::map<std::string, std::string> defines) {

		if(!vk) {
			throw std::logic_error("Shaders cannot be initialized without a window");
		}

		if(geometrysrc != "" || !defines.empty()) {
			log << "Shader " << name << ": geometry shaders and defines are not supported.";
		}

		vk->programs.emplace_back();

		auto &prog = vk->programs.back();
		prog.vertex.assign(vertexsrc.begin(), vertexsrc.end());
		prog.fragment.assign(fragmentsrc.begin(), fragmentsrc.end());

		program = (unsigned int)vk->programs.size();

		log << "Shader program " << name << " is ready.";
	}

}
