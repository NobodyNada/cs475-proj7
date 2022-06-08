use std::collections::VecDeque;
use std::time::Duration;
use std::{sync::Arc, time::Instant};

use anyhow::{anyhow, bail, Result};
use cgmath::{Matrix4, Point3, Vector3};
use rustfft::num_complex::Complex;
use vulkano::buffer::{BufferAccess, CpuAccessibleBuffer, DeviceLocalBuffer};
use vulkano::command_buffer::PrimaryCommandBuffer;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::pipeline::layout::PipelineLayoutCreateInfo;
use vulkano::pipeline::{ComputePipeline, Pipeline, PipelineLayout};
use vulkano::query::{QueryPool, QueryPoolCreateInfo, QueryResultFlags, QueryType};
use vulkano::sync::PipelineStage;
use vulkano::{
    buffer::{BufferUsage, CpuBufferPool, TypedBufferAccess},
    command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents},
    descriptor_set::{
        layout::{
            DescriptorSetLayout, DescriptorSetLayoutBinding, DescriptorSetLayoutCreateInfo,
            DescriptorType,
        },
        SingleLayoutDescSetPool,
    },
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, Queue,
    },
    format::Format,
    image::{view::ImageView, AttachmentImage, ImageUsage},
    instance::{Instance, InstanceCreateInfo},
    pipeline::{
        graphics::{
            input_assembly::{InputAssemblyState, PrimitiveTopology},
            vertex_input::BuffersDefinition,
            viewport::{Viewport, ViewportState},
        },
        GraphicsPipeline, PipelineBindPoint,
    },
    render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass},
    shader::ShaderStages,
    single_pass_renderpass,
    swapchain::{
        acquire_next_image, AcquireError, ColorSpace, Surface, Swapchain, SwapchainCreateInfo,
    },
    sync::{FlushError, GpuFuture},
};
use vulkano_win::VkSurfaceBuild;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent},
    event_loop::ControlFlow,
    window::Window,
};

mod audio;
mod shader;

const NUM_PARTICLES_PERAXIS: usize = 100;
const VIEW_DISTANCE: f32 = 10.0;

/// Our renderer, in charge of managing Vulkan resources and drawing stuff.
struct Renderer {
    /// The Vulkan device.
    device: Arc<Device>,
    queue: Arc<Queue>,
    surface: Arc<Surface<Window>>,
    color_format: Format,
    depth_format: Format,

    render_pass: Arc<RenderPass>,
    framebuffers: Framebuffers,
    compute_pipeline: Arc<ComputePipeline>,
    graphics_pipeline: Arc<GraphicsPipeline>,

    /// The particle positions, stored on device memory..
    points: Arc<DeviceLocalBuffer<[shader::Point]>>,
    /// The particle velocities, stored on device memory.
    _velocities: Arc<DeviceLocalBuffer<[shader::Velocity]>>,
    /// The pressure of each cell, stored on device memory.
    /// This buffer is twice the size you'd expect: it stores
    /// one copy of the pressures from the previous frame, and
    /// a second copy to accumulate pressures being computed this frame.
    pressures: Arc<DeviceLocalBuffer<[u32]>>,

    /// Which part of the pressure buffers is currently being written to.
    which_pressure_buffer: bool,

    compute_storage_descriptors: Arc<PersistentDescriptorSet>,
    compute_uniforms: CpuBufferPool<shader::compute::ty::Uniforms>,
    compute_uniform_descriptor_pool: SingleLayoutDescSetPool,

    vertex_uniforms: CpuBufferPool<shader::vertex::ty::Uniforms>,
    /// The buffer storing the amplitudes of each frequency band.
    /// Note that each band is stored as a vec4, with only the first component used, because
    /// apparently SPIR-V does not allow arrays of floats (?)
    bands_uniform: CpuBufferPool<[shader::Point; shader::compute::EQ_BANDS as usize]>,
    vertex_uniform_descriptor_pool: SingleLayoutDescSetPool,

    /// The projection & view matrix.
    matrix: cgmath::Matrix4<f32>,

    /// The time we started the last frame.
    last_frame: Instant,
    /// The time we last printed performance.
    last_fps_print: Instant,
    /// The query pool we use to measure timestamps of GPU operations.
    query_pool: Arc<QueryPool>,
    /// The number of frames since we last printed performance.
    frames: u32,
    /// The number of particles computed since we last printed performance.
    particles: f32,
    /// The amount of time we've spent computing particles since we last printed performance.
    seconds: f32,

    /// The audio player.
    player: Option<audio::Player>,

    /// Incoming audio samples we have yet to analyze.
    sample_buffer: VecDeque<f32>,
}

enum Framebuffers {
    // We have not yet initalized framebuffers.
    NotCreated,

    // We have a swapchain, but it's invalid.
    Invalid {
        swapchain: Arc<Swapchain<Window>>,
    },

    // We have a valid swapchain.
    Valid {
        swapchain: Arc<Swapchain<Window>>,
        framebuffers: Vec<Arc<Framebuffer>>,
    },
}
impl Default for Framebuffers {
    fn default() -> Self {
        Self::NotCreated
    }
}
impl Framebuffers {
    fn invalidate(&mut self) {
        *self = match std::mem::take(self) {
            Framebuffers::Valid {
                swapchain,
                framebuffers: _,
            } => Framebuffers::Invalid { swapchain },
            x => x,
        }
    }
}

impl Renderer {
    /// Selects and initializes a Vulkan device.
    fn create_device(
        instance: &Arc<Instance>,
        surface: &Surface<Window>,
    ) -> Result<(Arc<Device>, Arc<Queue>)> {
        // Look for a graphics card that meets our requirements.
        let required_extensions = vulkano::device::DeviceExtensions {
            khr_swapchain: true,
            ..vulkano::device::DeviceExtensions::none()
        };

        let (best_device, queue_family) = PhysicalDevice::enumerate(instance)
            .filter(|d| {
                d.supported_extensions()
                    .is_superset_of(&required_extensions)
            })
            .flat_map(|d| {
                d.queue_families()
                    .filter(|q| {
                        q.supports_graphics()
                            && q.supports_compute()
                            && q.supports_surface(surface).unwrap_or(false)
                    })
                    .map(move |q| (d, q))
            })
            .min_by_key(|(d, _)| match d.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
            })
            .expect("No Vulkan device!");

        println!("Using device: {:#?}", best_device);

        // Initialize the Vulkan device
        let (device, mut queues) = Device::new(
            best_device,
            vulkano::device::DeviceCreateInfo {
                enabled_extensions: required_extensions.union(best_device.required_extensions()),
                queue_create_infos: vec![vulkano::device::QueueCreateInfo::family(queue_family)],
                ..Default::default()
            },
        )?;
        assert!(queues.len() == 1);
        Ok((device, queues.next().unwrap()))
    }

    /// Creates the renderer.
    fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        surface: Arc<Surface<Window>>,
        player: Option<audio::Player>,
    ) -> Result<Self> {
        // Choose a color and depth format.
        let color_format = device
            .physical_device()
            .surface_formats(&surface, Default::default())?
            .iter()
            .find(|(f, c)| {
                *c == ColorSpace::SrgbNonLinear
                    && [Format::R8G8B8A8_UNORM, Format::B8G8R8A8_UNORM].contains(f)
            })
            .ok_or_else(|| anyhow!("no suitable color formats"))?
            .0;
        let depth_format = Format::D16_UNORM;

        // Uniformly distribute particles inside of the unit cube.
        let points: Vec<shader::Point> = (0..NUM_PARTICLES_PERAXIS)
            .flat_map(|x| (0..NUM_PARTICLES_PERAXIS).map(move |y| (x, y)))
            .flat_map(|(x, y)| {
                (0..NUM_PARTICLES_PERAXIS).map(move |z| (x as f32, y as f32, z as f32))
            })
            .map(|(x, y, z)| shader::Point {
                position: [
                    (x - NUM_PARTICLES_PERAXIS as f32 / 2.0) / (NUM_PARTICLES_PERAXIS as f32 / 2.0),
                    (y - NUM_PARTICLES_PERAXIS as f32 / 2.0) / (NUM_PARTICLES_PERAXIS as f32 / 2.0),
                    (z - NUM_PARTICLES_PERAXIS as f32 / 2.0) / (NUM_PARTICLES_PERAXIS as f32 / 2.0),
                    1.0,
                ],
            })
            .collect();

        // Allocate device memory for our buffers.
        let points_buffer = DeviceLocalBuffer::array(
            device.clone(),
            points.len() as u64,
            BufferUsage {
                transfer_destination: true,
                vertex_buffer: true,
                storage_buffer: true,
                ..Default::default()
            },
            [queue.family()],
        )?;
        let velocities_buffer = DeviceLocalBuffer::array(
            device.clone(),
            points.len() as u64,
            BufferUsage {
                transfer_destination: true,
                storage_buffer: true,
                ..Default::default()
            },
            [queue.family()],
        )?;
        let pressures_buffer = DeviceLocalBuffer::array(
            device.clone(),
            shader::compute::NUM_CELLS_TOTAL as u64 * 2,
            BufferUsage {
                transfer_destination: true,
                storage_buffer: true,
                ..Default::default()
            },
            [queue.family()],
        )?;

        // Upload the initial values to the GPU.
        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            device.clone(),
            queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )?;
        cmd_builder
            .fill_buffer(velocities_buffer.clone(), 0)?
            .fill_buffer(pressures_buffer.clone(), 0)?
            .copy_buffer(
                CpuAccessibleBuffer::from_iter(
                    device.clone(),
                    BufferUsage::transfer_source(),
                    false,
                    points.iter().copied(),
                )?,
                points_buffer.clone(),
            )?;
        let inflight = cmd_builder
            .build()?
            .execute(queue.clone())?
            .then_signal_fence_and_flush()?;

        // Create our render pass:
        let render_pass = single_pass_renderpass!(device.clone(),
            attachments: {
                // Clear the color buffer on load, and store it to memory so we can see it.
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                // Clear the depth buffer on load, but we don't need to save the results.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: depth_format,
                    samples: 1,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        )?;

        // Load our shaders.
        let cs = shader::compute::load(device.clone())?;
        let vs = shader::vertex::load(device.clone())?;
        let fs = shader::fragment::load(device.clone())?;

        // Define our shader binding layouts.
        let compute_storage_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: std::collections::BTreeMap::from([
                    (
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages {
                                compute: true,
                                ..Default::default()
                            },
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        1,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages {
                                compute: true,
                                ..Default::default()
                            },
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                    (
                        2,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages {
                                compute: true,
                                ..Default::default()
                            },
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                ]),
                ..Default::default()
            },
        )?;

        let compute_uniform_layout = DescriptorSetLayout::new(
            device.clone(),
            DescriptorSetLayoutCreateInfo {
                bindings: std::collections::BTreeMap::from([
                    (
                        0,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages {
                                compute: true,
                                ..Default::default()
                            },
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::UniformBuffer,
                            )
                        },
                    ),
                    (
                        1,
                        DescriptorSetLayoutBinding {
                            stages: ShaderStages {
                                compute: true,
                                ..Default::default()
                            },
                            ..DescriptorSetLayoutBinding::descriptor_type(
                                DescriptorType::StorageBuffer,
                            )
                        },
                    ),
                ]),
                ..Default::default()
            },
        )?;

        // Create the compute pipeline, which defines the shader program along with its inputs and
        // outputs.
        let compute_pipeline = ComputePipeline::with_pipeline_layout(
            device.clone(),
            cs.entry_point("main").unwrap(),
            &shader::compute::SpecializationConstants::new(),
            PipelineLayout::new(
                device.clone(),
                PipelineLayoutCreateInfo {
                    set_layouts: vec![
                        compute_storage_layout.clone(),
                        compute_uniform_layout.clone(),
                    ],
                    ..Default::default()
                },
            )?,
            None,
        )?;

        // Define our persistent bindings -- the buffers that will stay bound and never change.
        let compute_storage_descriptors = PersistentDescriptorSet::new(
            compute_storage_layout,
            [
                WriteDescriptorSet::buffer(0, points_buffer.clone()),
                WriteDescriptorSet::buffer(1, velocities_buffer.clone()),
                WriteDescriptorSet::buffer(2, pressures_buffer.clone()),
            ],
        )?;

        // Create a buffer pool for our uniforms (these buffers will change every frame as we
        // upload new values).
        let compute_uniforms = CpuBufferPool::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
        );
        // Create a descriptor pool that we can use to instantiate descriptors to bind uniform
        // buffers to our compute shader.
        let compute_uniform_descriptor_pool = SingleLayoutDescSetPool::new(compute_uniform_layout);

        // Now do all that again, but for the graphics pipelie with our vertex and fragment shader.
        let graphics_pipeline = GraphicsPipeline::start()
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .vertex_input_state(BuffersDefinition::new().vertex::<shader::Point>())
            .input_assembly_state(InputAssemblyState::new().topology(PrimitiveTopology::PointList))
            .vertex_shader(
                vs.entry_point("main").unwrap(),
                shader::vertex::SpecializationConstants::new(),
            )
            .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
            .fragment_shader(fs.entry_point("main").unwrap(), ())
            .build(device.clone())?;

        let vertex_uniforms = CpuBufferPool::new(
            device.clone(),
            BufferUsage::uniform_buffer_transfer_destination(),
        );
        let bands_uniform = CpuBufferPool::new(
            device.clone(),
            BufferUsage {
                storage_buffer: true,
                transfer_destination: true,
                ..Default::default()
            },
        );

        let matrix = Self::create_matrix(surface.window().inner_size());

        let vertex_uniform_descriptor_pool =
            SingleLayoutDescSetPool::new(DescriptorSetLayout::new(
                device.clone(),
                DescriptorSetLayoutCreateInfo {
                    bindings: std::collections::BTreeMap::from([
                        (
                            0,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages {
                                    vertex: true,
                                    ..Default::default()
                                },
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::UniformBuffer,
                                )
                            },
                        ),
                        (
                            1,
                            DescriptorSetLayoutBinding {
                                stages: ShaderStages {
                                    vertex: true,
                                    ..Default::default()
                                },
                                ..DescriptorSetLayoutBinding::descriptor_type(
                                    DescriptorType::StorageBuffer,
                                )
                            },
                        ),
                    ]),
                    ..Default::default()
                },
            )?);

        let query_pool = QueryPool::new(
            device.clone(),
            QueryPoolCreateInfo {
                query_count: 2,
                ..QueryPoolCreateInfo::query_type(QueryType::Timestamp)
            },
        )?;

        // Wait for the buffer uploads we submitted earlier to complete.
        inflight.wait(None)?;

        Ok(Self {
            device,
            queue,
            surface,
            color_format,
            depth_format,
            render_pass,
            framebuffers: Framebuffers::NotCreated,
            compute_pipeline,
            graphics_pipeline,
            points: points_buffer,
            _velocities: velocities_buffer,
            pressures: pressures_buffer,
            compute_storage_descriptors,
            compute_uniforms,
            compute_uniform_descriptor_pool,
            vertex_uniforms,
            bands_uniform,
            vertex_uniform_descriptor_pool,
            matrix,
            last_frame: Instant::now(),
            last_fps_print: Instant::now(),
            frames: 0,
            which_pressure_buffer: false,
            player,
            sample_buffer: VecDeque::new(),
            query_pool,
            particles: 0.0,
            seconds: 0.0,
        })
    }

    /// Creates a projection/view matrix.
    fn create_matrix(dimensions: PhysicalSize<u32>) -> Matrix4<f32> {
        let aspect = dimensions.width as f32 / dimensions.height as f32;
        let proj = cgmath::perspective(cgmath::Deg(90.0), aspect, 1.0, 100.0);
        let view = Matrix4::look_at_rh(
            Point3::new(0.0, -VIEW_DISTANCE, -VIEW_DISTANCE),
            Point3::new(0.0, 0.0, 0.0),
            Vector3::new(0.0, 1.0, 0.0),
        );
        proj * view
    }

    /// Indicates that the window has been resized and the framebuffers should be recreated.
    fn resize(&mut self) {
        self.framebuffers.invalidate()
    }

    /// Called every frame to update the simulation and render the scene.
    fn render(&mut self) -> Result<()> {
        let dimensions = self.surface.window().inner_size();

        let elapsed: Duration = self.last_frame.elapsed();
        self.last_frame = Instant::now();

        // First, ensure we have framebuffers set up.
        let (swapchain, framebuffers) = match std::mem::take(&mut self.framebuffers) {
            // We do, good!
            Framebuffers::Valid {
                swapchain,
                framebuffers,
            } => (swapchain, framebuffers),

            // We don't.
            fbs => {
                // Create a swapchain of color buffers, and a shared depth buffer.
                let info = SwapchainCreateInfo {
                    image_usage: ImageUsage::color_attachment(),
                    image_extent: dimensions.into(),
                    image_format: Some(self.color_format),
                    ..Default::default()
                };

                let (swapchain, images) = if let Framebuffers::Invalid { swapchain } = fbs {
                    // We have an existing swapchain, recreate it.
                    swapchain.recreate(info)
                } else {
                    Swapchain::new(self.device.clone(), self.surface.clone(), info)
                }?;

                let depth_buffer = ImageView::new_default(AttachmentImage::transient(
                    self.device.clone(),
                    dimensions.into(),
                    self.depth_format,
                )?)?;

                // Bind the color and depth buffers to framebuffers.
                let framebuffers = images
                    .into_iter()
                    .map(|i| {
                        let color_buffer = ImageView::new_default(i)?;

                        Framebuffer::new(
                            self.render_pass.clone(),
                            FramebufferCreateInfo {
                                attachments: vec![color_buffer, depth_buffer.clone()],
                                ..Default::default()
                            },
                        )
                        .map_err(anyhow::Error::new)
                    })
                    .collect::<Result<Vec<_>>>()?;

                // Recreate the projection matrix, since the window size may have changed.
                self.matrix = Self::create_matrix(dimensions);

                (swapchain, framebuffers)
            }
        };

        // Acquire a framebuffer.
        let (fb_idx, mut suboptimal, acquired) = match acquire_next_image(swapchain.clone(), None) {
            Ok(result) => result,
            Err(AcquireError::OutOfDate) => {
                // Recreate the swapchain and try again next frame.
                self.framebuffers = Framebuffers::Invalid { swapchain };
                return Ok(());
            }
            Err(e) => bail!(e),
        };

        // Deal with audio samples.
        const EQ_BANDS: usize = shader::compute::EQ_BANDS as usize;
        let mut bands = [shader::Point {
            position: [1.0, 0.0, 0.0, 0.0],
        }; EQ_BANDS];
        let mut amplitude = 0.5;
        if let Some(ref p) = self.player {
            // Take whatever samples the player has generated in the meantime.
            self.sample_buffer
                .extend(p.sample_buffer.lock().unwrap().drain(..));

            // Take one frames' worth of audio out of the buffer.
            let num_samples = (p.sample_rate.0 as f32 * elapsed.as_secs_f32()).round() as usize;
            let num_samples = std::cmp::min(num_samples, self.sample_buffer.len());
            let samples = self.sample_buffer.drain(0..num_samples).collect::<Vec<_>>();

            let mut fft_len = samples.len();
            // round to the next lowest power of 2
            if fft_len != 0 {
                fft_len = samples.len() & (1 << (usize::BITS - 1 - samples.len().leading_zeros()));
            }

            // Do a fast Fourier transform to convert our sound samples into the frequency domain.
            let fft = rustfft::FftPlanner::new().plan_fft(fft_len, rustfft::FftDirection::Forward);
            let mut fft_buffer = samples
                .iter()
                .take(fft_len)
                .map(|&s| Complex::new(s, 0.0))
                .collect::<Vec<Complex<f32>>>();
            fft.process(&mut fft_buffer);

            // Divide the FFT results into bands.
            for (i, chunk) in fft_buffer.chunks(fft_buffer.len() / EQ_BANDS).enumerate() {
                let average = chunk.iter().map(|x| x.norm()).sum::<f32>() / chunk.len() as f32;
                bands[i].position[0] = average;
            }

            if samples.is_empty() {
                // The audio engine has fallen behind.
                println!("NO SAMPLES");
            } else {
                // Set our amplitude to the average amplitude of the samples.
                amplitude = (samples.iter().copied().map(|x| x.abs() as f64).sum::<f64>()
                    / samples.len() as f64) as f32;
            }
        }

        // Upload our compute shader uniforms.
        let compute_uniforms = shader::compute::ty::Uniforms {
            which_pressure_buffer: self.which_pressure_buffer as u32,
            amplitude,
            elapsed: elapsed.as_secs_f32(),
        };
        let compute_uniform_buffer = self.compute_uniforms.next(compute_uniforms)?;

        // Upload our vertex shader uniforms.
        let vertex_uniforms = shader::vertex::ty::Uniforms {
            matrix: self.matrix.into(),
            num_points: self.points.len() as u32,
        };
        let vertex_uniform_buffer = self.vertex_uniforms.next(vertex_uniforms)?;

        // Upload the frequency bands.
        let bands_uniform_buffer = self.bands_uniform.next(bands)?;

        // We're ready to issue some rendering commands!
        let mut cmd_builder = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .expect("Could not create command buffer builder");

        // Are we going to write to the first half or the second half of the pressure buffer?
        let pressure_dst_range = if self.which_pressure_buffer {
            0..shader::compute::NUM_CELLS_TOTAL as u64
        } else {
            shader::compute::NUM_CELLS_TOTAL as u64..(shader::compute::NUM_CELLS_TOTAL as u64 * 2)
        };

        // First, execute the compute program.
        unsafe {
            // write_timestamp is marked as unsafe because writing a timestamp without first
            // resetting the query pool is undefined behavior. We use an `unsafe` block to indicate
            // to the compiler that we know what we're doing and we've checked for that.
            cmd_builder
                // Measure the time at which the GPU begins running the compute pipeline.
                .reset_query_pool(self.query_pool.clone(), 0..2)?
                .write_timestamp(self.query_pool.clone(), 0, PipelineStage::TopOfPipe)?
                // Zero out the destination area of the pressure buffer.
                .fill_buffer(self.pressures.slice(pressure_dst_range).unwrap(), 0)?
                // Bind our uniform buffers to our compute program.
                .bind_pipeline_compute(self.compute_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Compute,
                    self.compute_pipeline.layout().clone(),
                    0,
                    (
                        self.compute_storage_descriptors.clone(),
                        self.compute_uniform_descriptor_pool.next([
                            WriteDescriptorSet::buffer(0, compute_uniform_buffer),
                            WriteDescriptorSet::buffer(1, bands_uniform_buffer.clone()),
                        ])?,
                    ),
                )
                // Run our compute program over all the particles.
                .dispatch([self.points.len() as u32 / 32, 1, 1])?
                // Measure the time at which we finish running the compute pipeline.
                .write_timestamp(self.query_pool.clone(), 1, PipelineStage::BottomOfPipe)?;
        }

        // We have now updated the particle simulation, so render the results.
        cmd_builder
            // Begin a render pass, clearing the color and depth buffer.
            .begin_render_pass(
                framebuffers[fb_idx].clone(),
                SubpassContents::Inline,
                [
                    [0.0, 0.0, 0.0, 0.0].into(), // color clear value
                    1.0.into(),                  // depth clear value
                ],
            )?
            .set_viewport(
                0,
                [Viewport {
                    origin: [0.0, 0.0],
                    dimensions: dimensions.into(),
                    depth_range: 0.0..1.0,
                }],
            )
            .bind_pipeline_graphics(self.graphics_pipeline.clone())
            // Attach our vertex and uniform buffers to our shaders.
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.graphics_pipeline.layout().clone(),
                0,
                self.vertex_uniform_descriptor_pool.next([
                    WriteDescriptorSet::buffer(0, vertex_uniform_buffer),
                    WriteDescriptorSet::buffer(1, bands_uniform_buffer),
                ])?,
            )
            .bind_vertex_buffers(0, self.points.clone())
            // Run our graphics program over all our particles.
            .draw(self.points.len() as u32, 1, 0, 0)?
            .end_render_pass()?;

        // Actually generate the command buffer and upload it to the GPU.
        let inflight = cmd_builder
            .build()?
            // Don't run until the framebuffer is ready.
            .execute_after(acquired, self.queue.clone())?
            // Once we're done rendering, present the buffer to the screen.
            .then_swapchain_present(self.queue.clone(), swapchain.clone(), fb_idx)
            .then_signal_fence_and_flush()?;
        match inflight.flush() {
            Ok(_) => {}
            Err(FlushError::OutOfDate) => suboptimal = true,
            Err(e) => bail!(e),
        }

        self.framebuffers = if suboptimal {
            Framebuffers::Invalid { swapchain }
        } else {
            Framebuffers::Valid {
                swapchain,
                framebuffers,
            }
        };

        // Copy the query results back.
        let mut timestamps = [0u64; 2];
        self.query_pool.queries_range(0..2).unwrap().get_results(
            &mut timestamps,
            QueryResultFlags {
                wait: true,
                partial: false,
                with_availability: false,
            },
        )?;

        // Update our statistics.
        let period_ns = self.device.physical_device().properties().timestamp_period;
        let seconds = period_ns * (timestamps[1] - timestamps[0]) as f32 / 1e9;
        self.seconds += seconds;
        self.particles += NUM_PARTICLES_PERAXIS.pow(3) as f32;
        self.frames += 1;

        if self.last_fps_print.elapsed().as_secs() >= 1 {
            // It's been a second since the last time we printed our statistics, so do that.
            let particles_per_second = self.particles / self.seconds;
            let mps = particles_per_second / 1e6;
            println!("FPS: {}\tMegaParticles/s: {mps}", self.frames);
            self.last_fps_print = Instant::now();
            self.frames = 0;
            self.seconds = 0.0;
            self.particles = 0.0;
        }

        Ok(())
    }
}

fn main() {
    let event_loop = winit::event_loop::EventLoop::new();

    // Create a Vulkan instance
    let instance = Instance::new(InstanceCreateInfo {
        enabled_extensions: vulkano_win::required_extensions(),
        ..Default::default()
    })
    .expect("failed to create Vulkan instance");

    // Create a window with a Vulkan surface
    let surface = winit::window::WindowBuilder::new()
        .with_title("CS 475 Project 7A")
        .build_vk_surface(&event_loop, instance.clone())
        .expect("failed to create window");

    // Start audio.
    let player = audio::play();
    if let Err(ref e) = player {
        eprintln!("Failed to initialize audio: {e:#?}");
    }

    // Create our renderer.
    let (device, queue) =
        Renderer::create_device(&instance, &surface).expect("failed to create device");
    let mut renderer =
        Renderer::new(device, queue, surface, player.ok()).expect("failed to create renderer");

    // Run the window's event loop.
    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            renderer.resize();
        }
        Event::RedrawEventsCleared => renderer.render().expect("render failed"),
        _ => (),
    })
}
