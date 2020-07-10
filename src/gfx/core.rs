use vulkano::device::Queue;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, FramebufferBuilder, Subpass};
use vulkano::image::{SwapchainImage, ImmutableImage, Dimensions};
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::sync::{GpuFuture, JoinFuture};
use vulkano::{sync, swapchain};
use vulkano_win::VkSurfaceBuild;
use winit::event_loop::{EventLoop, ControlFlow};
use winit::window::{WindowBuilder, Window};
use winit::event::Event;

use std::sync::Arc;
use std::time::Instant;
use std::fmt::{Display, Formatter};
use stable_eyre::Report;
use vulkano::swapchain::{Swapchain, SurfaceTransform, ColorSpace, PresentMode, FullscreenExclusive, SwapchainAcquireFuture, Surface, SwapchainCreationError, AcquireError};
use vulkano::pipeline::viewport::Viewport;
use vulkano::format::Format;
use time::Duration;
use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
use std::vec::IntoIter;
use std::io::Cursor;
use vulkano::memory::pool::{StdMemoryPoolAlloc, PotentialDedicatedAllocation};
use std::ops::{Deref, DerefMut};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract, GraphicsPipelineBuilder};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, PersistentDescriptorSetBuilder};
use crate::gfx::ext::ThreeBuffersDefinition;
use crate::gfx::util::{VkVertex, VkColour, VkTexCoord};
use vulkano::pipeline::vertex::{TwoBuffersDefinition, BufferlessDefinition};
use vulkano::pipeline::shader::EmptyEntryPointDummy;

#[derive(Debug)]
pub enum VkError {
    InvalidDevice,
    UnsupportedDevice,
    QueueCreationFailed,
    CommandAccessFailed {
        name: String,
        param: String,
        offset: usize,
    },
    OptionMissing,
    DescriptorSetLayoutMissing,
    SubpassMissing,
}

impl Display for VkError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for VkError {}

pub struct VulkanFrontend {
    events_loop: EventLoop<()>,
    pub vk: VulkanBackend,
}

#[derive(Clone)]
pub struct VkPipeline {
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
}

impl VkPipeline {
    pub fn inner(&self) -> Arc<dyn GraphicsPipelineAbstract + Send + Sync> {
        self.pipeline.clone()
    }

    pub fn descriptor_set(&self, index: usize) -> Result<PersistentDescriptorSetBuilder<()>, Report> {
        Ok(PersistentDescriptorSet::start(self.pipeline.descriptor_set_layout(index)
            .ok_or(VkError::DescriptorSetLayoutMissing)?.clone()))
    }
}

pub struct VulkanBackend {
    show_fps: bool,
    recreate_swapchain: bool,
    device: Arc<Device>,
    queue: Arc<Queue>,
    swapchain: Arc<Swapchain<Window>>,
    dynamic_state: DynamicState,
    surface: Arc<Surface<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,
    image_num: usize,
    future: Option<VulkanFuture>,
    t: Instant,
    default_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    vc_pipeline: VkPipeline,
    vct_pipeline: VkPipeline,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanDevice {
    pub name: String,
    pub id: usize,
}

pub struct VulkanInitialiser {
    instance: Arc<Instance>,
    devices: Vec<VulkanDevice>,
}

enum VulkanFuture {
    Joined(JoinFuture<Box<dyn GpuFuture>, Box<dyn GpuFuture>>),
    Lone(Box<dyn GpuFuture>)
}

impl From<Box<dyn GpuFuture>> for VulkanFuture {
    fn from(future: Box<dyn GpuFuture>) -> Self {
        Self::Lone(future)
    }
}

impl From<JoinFuture<Box<dyn GpuFuture>, Box<dyn GpuFuture>>> for VulkanFuture {
    fn from(future: JoinFuture<Box<dyn GpuFuture>, Box<dyn GpuFuture>>) -> Self {
        Self::Joined(future)
    }
}

impl From<JoinFuture<JoinFuture<Box<dyn GpuFuture>, Box<dyn GpuFuture>>, Box<dyn GpuFuture>>> for VulkanFuture {
    fn from(future: JoinFuture<JoinFuture<Box<dyn GpuFuture>, Box<dyn GpuFuture>>, Box<dyn GpuFuture>>) -> Self {
        Self::Lone(future.boxed())
    }
}

impl VulkanFuture {
    fn cleanup_finished(&mut self) {
        match self {
            VulkanFuture::Joined(future) => future.cleanup_finished(),
            VulkanFuture::Lone(future) => future.cleanup_finished()
        }
    }

    fn join(self, device: Arc<Device>, other: Box<dyn GpuFuture>) -> JoinFuture<JoinFuture<Box<dyn GpuFuture>, Box<dyn GpuFuture>>, Box<dyn GpuFuture>>
            where Self: Sized {
        match self {
            VulkanFuture::Joined(future) => future.join(other),
            VulkanFuture::Lone(future) => {
                // this is a bad hack hack
                future.join(sync::now(device).boxed()).join(other)
            },
        }
    }
}

impl VulkanInitialiser {
    pub fn devices(&self) -> &[VulkanDevice] {
        &self.devices
    }

    pub fn new() -> Result<VulkanInitialiser, Report> {
        println!("Beginning Vulkan setup...");
        let instance = {
            let extensions = vulkano_win::required_extensions();
            Instance::new(None, &extensions, None)
        }?;

        let devices = PhysicalDevice::enumerate(&instance)
            .map(|device| VulkanDevice {
                name: device.name().to_string(),
                id: device.index(),
            })
            .collect();

        Ok(Self {
            instance, devices,
        })
    }

    pub fn finalise(self, device_id: usize) -> Result<VulkanFrontend, Report> {
        let physical = PhysicalDevice::from_index(&self.instance, device_id)
            .ok_or(VkError::InvalidDevice)?;

        let events_loop = EventLoop::new();
        let surface = WindowBuilder::new()
            .with_title("Svarog Window")
            .with_decorations(false)
            .build_vk_surface(&events_loop, self.instance.clone())?;

        let queue_family = physical.queue_families().find(|&q| {
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        }).ok_or(VkError::UnsupportedDevice)?;

        let device_ext = DeviceExtensions { khr_swapchain: true, ..DeviceExtensions::none() };
        let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                               [(queue_family, 0.5)].iter().cloned())?;
        let queue = queues.next().ok_or(VkError::QueueCreationFailed)?;

        let caps = surface.capabilities(physical)?;
        let usage = caps.supported_usage_flags;
        let dims = caps.current_extent.unwrap_or([1280, 1024]);
        let alpha = caps.supported_composite_alpha.iter().next().ok_or(VkError::OptionMissing)?;
        let format = caps.supported_formats[0].0;

        let (swapchain, images) = Swapchain::new(device.clone(), surface.clone(),
                                                 caps.min_image_count, format, dims, 1, usage,
                                                 &queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo, FullscreenExclusive::Default,
                                                 true, ColorSpace::SrgbNonLinear)?;

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dims[0] as f32, dims[1] as f32],
            depth_range: 0.0..1.0,
        };
        let dynamic_state = DynamicState {
            viewports: Some(vec![viewport]),
            .. DynamicState::none()
        };

        // Generate default pipelines
        let vs = vc_vs::Shader::load(device.clone())?;
        let fs = vc_fs::Shader::load(device.clone())?;
        let default_pass = Arc::new(vulkano::single_pass_renderpass!(
            device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: swapchain.format(),
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {}
            }
        )?);
        let vc_pipeline = VkPipeline {
            pipeline: Arc::new(GraphicsPipeline::start()
                .vertex_input(TwoBuffersDefinition::<VkVertex, VkColour>::new())
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_alpha_blending()
                .render_pass(Subpass::from(default_pass.clone(), 0).ok_or(VkError::SubpassMissing)?)
                .build(device.clone())?),
        };

        let vs = vct_vs::Shader::load(device.clone())?;
        let fs = vct_fs::Shader::load(device.clone())?;
        let vct_pipeline = VkPipeline {
            pipeline: Arc::new(GraphicsPipeline::start()
                .vertex_input(ThreeBuffersDefinition::<VkVertex, VkColour, VkTexCoord>::new())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .vertex_shader(vs.main_entry_point(), ())
                .fragment_shader(fs.main_entry_point(), ())
                .blend_alpha_blending()
                .render_pass(Subpass::from(default_pass.clone(), 0).ok_or(VkError::SubpassMissing)?)
                .build(device.clone())?),
        };

        let backend = VulkanBackend {
            show_fps: true,
            recreate_swapchain: false,
            device,
            queue,
            swapchain,
            surface,
            images,
            image_num: 0,
            dynamic_state,
            future: None,
            t: Instant::now(),
            default_pass,
            vc_pipeline,
            vct_pipeline,
        };

        Ok(VulkanFrontend {
            events_loop,
            vk: backend
        })
    }
}

pub trait VulkanInterface {
    fn iter_buffer<T>(&self, iter: IntoIter<T>) -> Result<Arc<CpuAccessibleBuffer<[T]>>, Report>
        where
            T: 'static;

    fn slice_buffer<T>(&self, buffer: &[T]) -> Result<Arc<CpuAccessibleBuffer<[T]>>, Report>
        where
            T: 'static + Clone;

    fn load_texture(&mut self, data: &[u8]) -> Result<Arc<ImmutableImage<Format>>, Report>;
}

impl VulkanFrontend {
    pub fn begin<F>(self, mut step: F) where F: FnMut(&mut VulkanBackend) -> Result<AutoCommandBufferBuilder, Report> + 'static {
        let mut backend = self.vk;
        backend.future = Some(sync::now(backend.device()).boxed().into());
        let mut updates = 0.0;

        self.events_loop.run( move |event, _, control_flow| {
            match event {
                Event::WindowEvent { event: winit::event::WindowEvent::CloseRequested, .. } => {
                    *control_flow = ControlFlow::Exit;
                },
                Event::WindowEvent { event: winit::event::WindowEvent::Resized(_), .. } => {
                    println!("window resized");
                    backend.recreate_swapchain = true;
                },
                Event::RedrawEventsCleared => {
                    let future = backend.future.take().unwrap();

                    let fps_resolution = Duration::seconds(1);

                    if backend.show_fps {
                        updates += 1.0;
                        if backend.t.elapsed() > fps_resolution {
                            let elapsed = backend.t.elapsed().as_millis() as f64;
                            let fps = ((updates / elapsed) * 1e3).round();
                            println!("{} fps", fps.round());
                            backend.t = Instant::now();
                            updates = 0.0;
                        }
                    }

                    backend.future = Some(backend.full_redraw(future, &mut step).into());
                },
                _ => (),
            }
        });
    }
}

impl VulkanInterface for VulkanFrontend {
    fn iter_buffer<T>(&self, iter: IntoIter<T>) -> Result<Arc<CpuAccessibleBuffer<[T]>>, Report>
        where
            T: 'static {
        self.vk.iter_buffer(iter)
    }

    fn slice_buffer<T>(&self, buffer: &[T]) -> Result<Arc<CpuAccessibleBuffer<[T]>>, Report>
        where
            T: 'static + Clone {
        self.vk.slice_buffer(buffer)
    }

    fn load_texture(&mut self, data: &[u8]) -> Result<Arc<ImmutableImage<Format>>, Report> {
        self.vk.load_texture(data)
    }
}

impl VulkanBackend {
    pub fn device(&self) -> Arc<Device> {
        self.device.clone()
    }

    fn full_redraw<F>(&mut self, mut future: VulkanFuture, step: &mut F) -> VulkanFuture
        where F: FnMut(&mut Self) -> Result<AutoCommandBufferBuilder, Report> {
        future.cleanup_finished();

        if self.recreate_swapchain {
            println!("recreating swapchain");
            let dims = self.surface.window().inner_size().into();
            let (new_swap, new_images) = match self.swapchain.recreate_with_dimensions(dims) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return future,
                Err(e) => panic!("Failed to recreate swapchain: {}", e),
            };
            self.swapchain = new_swap;
            self.images = new_images;
            self.recreate_swapchain = false;
        }

        let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(self.swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                println!("swapchain outdated");
                self.recreate_swapchain = true;
                return future;
            },
            Err(e) => panic!("Failed to acquire swapchain: {}", e),
        };

        if suboptimal {
            println!("suboptimal swapchain created");
            self.recreate_swapchain = true;
        }
        self.image_num = image_num;

        let command_buffer = step(self).unwrap();
        match self.render_step(future, acquire_future, command_buffer) {
            Ok(future) => {
                future.into()
            },
            Err(e) => {
                if e.to_string() == "the swapchain needs to be recreated" {
                    self.recreate_swapchain = true;
                    sync::now(self.device()).boxed().into()
                } else {
                    println!("Failed to flush future: {:?}", e);
                    sync::now(self.device()).boxed().into()
                }
            }
        }
    }

    fn render_step(&mut self,
                   future: VulkanFuture,
                   acquire_future: SwapchainAcquireFuture<Window>,
                   command_buffer: AutoCommandBufferBuilder) -> Result<Box<dyn GpuFuture>, Report> {
        let command_buffer = command_buffer.build()?;

        Ok(future.join(self.device(), acquire_future.boxed())
            .then_execute(self.queue.clone(), command_buffer)?
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), self.image_num)
            .then_signal_fence_and_flush()?
            .boxed())
    }

    pub fn new_command_buffer(&self) -> Result<AutoCommandBufferBuilder, Report> {
        Ok(AutoCommandBufferBuilder::new(self.device.clone(), self.queue.family())?)
    }

    // TODO: cache render passes & framebuffers
    pub fn next_framebuffer(&mut self, render_pass: Arc<dyn RenderPassAbstract + Send + Sync>)
                            -> Result<Arc<dyn FramebufferAbstract + Send + Sync>, Report> {
        let image = &self.images[self.image_num];
        let dims = image.dimensions();
        self.dynamic_state.viewports = Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dims[0] as f32, dims[1] as f32],
            depth_range: 0.0..1.0,
        }]);
        Ok(Framebuffer::start(render_pass.clone())
            .add(image.clone())
            .and_then(FramebufferBuilder::build)
            .map(Arc::new)?)
    }

    pub fn dynamic(&self) -> &DynamicState {
        &self.dynamic_state
    }

    pub fn window_dims(&self) -> [f32; 2] {
        self.dynamic().viewports.as_ref().unwrap()[0].dimensions
    }

    pub fn default_pass(&self) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        self.default_pass.clone()
    }

    pub fn default_subpass(&self) -> Result<Subpass<Arc<dyn RenderPassAbstract + Send + Sync>>, Report> {
        Ok(Subpass::from(self.default_pass.clone(), 0).ok_or(VkError::SubpassMissing)?)
    }

    pub fn vert_col(&self) -> VkPipeline {
        self.vc_pipeline.clone()
    }

    pub fn vert_col_tex(&self) -> VkPipeline {
        self.vct_pipeline.clone()
    }

    pub fn window_to_vk(&self, x: f32, y: f32) -> (f32, f32) {
        let (x, y) = self.size_to_vk(x, y);
        ((x - 0.5) * 2.0, (y - 0.5) * 2.0)
    }

    pub fn size_to_vk(&self, x: f32, y: f32) -> (f32, f32) {
        (x / self.window_dims()[0], y / self.window_dims()[1])
    }

    pub fn new_vert_col(&self) -> Result<GraphicsPipelineBuilder<TwoBuffersDefinition<VkVertex, VkColour>, EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), Arc<dyn RenderPassAbstract + Send + Sync>>, Report> {
        Ok(GraphicsPipeline::start()
            .vertex_input(TwoBuffersDefinition::<VkVertex, VkColour>::new())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .blend_alpha_blending()
            .render_pass(self.default_subpass()?))
    }

    pub fn new_vert_col_tex(&self) -> Result<GraphicsPipelineBuilder<ThreeBuffersDefinition<VkVertex, VkColour, VkTexCoord>, EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), Arc<dyn RenderPassAbstract + Send + Sync>>, Report> {
        Ok(GraphicsPipeline::start()
            .vertex_input(ThreeBuffersDefinition::<VkVertex, VkColour, VkTexCoord>::new())
            .triangle_list()
            .viewports_dynamic_scissors_irrelevant(1)
            .blend_alpha_blending()
            .render_pass(self.default_subpass()?))
    }

    pub fn basic_pipeline(&self) -> Result<GraphicsPipelineBuilder<BufferlessDefinition, EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), EmptyEntryPointDummy, (), Arc<dyn RenderPassAbstract + Send + Sync>>, Report> {
        Ok(GraphicsPipeline::start()
                    .triangle_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .blend_alpha_blending()
                    .render_pass(self.default_subpass()?))
    }
}

impl VulkanInterface for VulkanBackend {
    fn iter_buffer<T>(&self, iter: IntoIter<T>) -> Result<Arc<CpuAccessibleBuffer<[T]>>, Report>
        where
            T: 'static {
        Ok(CpuAccessibleBuffer::from_iter(self.device(), BufferUsage::all(), false,
                                          iter)?)
    }

    fn slice_buffer<T>(&self, buffer: &[T]) -> Result<Arc<CpuAccessibleBuffer<[T]>>, Report>
        where
            T: 'static + Clone {
        Ok(CpuAccessibleBuffer::from_iter(self.device(), BufferUsage::all(), false,
                                          buffer.iter().cloned())?)
    }

    fn load_texture(&mut self, data: &[u8]) -> Result<Arc<ImmutableImage<Format>>, Report> {
        // see https://github.com/vulkano-rs/vulkano-examples/blob/master/src/bin/image/main.rs#L150
        let (texture, tex_future) = {
            // let png_bytes = include_bytes!(filename).to_vec();
            let cursor = Cursor::new(data);
            let decoder = png::Decoder::new(cursor);
            let (info, mut reader) = decoder.read_info()?;
            let dimensions = Dimensions::Dim2d {
                width: info.width,
                height: info.height,
            };

            let mut image_data = Vec::new();
            image_data.resize((info.width * info.height * 4) as usize, 0);
            reader.next_frame(&mut image_data)?;

            ImmutableImage::from_iter(
                image_data.iter().cloned(),
                dimensions,
                Format::R8G8B8A8Srgb,
                self.queue.clone(),
            )?
        };

        let future = self.future.take();
        self.future = future.map(|f| f.join(self.device(), tex_future.boxed()).into());

        Ok(texture)
    }
}

impl VulkanInterface for &mut VulkanBackend {
    fn iter_buffer<T>(&self, iter: IntoIter<T>) -> Result<Arc<CpuAccessibleBuffer<[T], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>, Report> where
        T: 'static {
        self.deref().iter_buffer(iter)
    }

    fn slice_buffer<T>(&self, buffer: &[T]) -> Result<Arc<CpuAccessibleBuffer<[T], PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>, Report> where
        T: 'static + Clone {
        self.deref().slice_buffer(buffer)
    }

    fn load_texture(&mut self, data: &[u8]) -> Result<Arc<ImmutableImage<Format, PotentialDedicatedAllocation<StdMemoryPoolAlloc>>>, Report> {
        self.deref_mut().load_texture(data)
    }
}

mod vc_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_col/vert.shader",
    }
}

mod vc_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/vert_col/frag.shader",
    }
}

mod vct_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "shaders/vert_col_tex/vert.shader",
    }
}

mod vct_fs {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "shaders/vert_col_tex/frag.shader",
    }
}