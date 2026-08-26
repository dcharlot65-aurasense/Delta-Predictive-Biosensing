//! GPU context management using wgpu.

use crate::config::GpuConfig;
use crate::error::{DpbError, Result};
use std::sync::Arc;
use wgpu;

/// GPU context wrapper for wgpu.
#[derive(Clone)]
pub struct GpuContext {
    /// wgpu device
    pub device: Arc<wgpu::Device>,
    /// wgpu queue
    pub queue: Arc<wgpu::Queue>,
    /// Adapter info
    pub adapter_info: wgpu::AdapterInfo,
}

impl GpuContext {
    /// Creates a new GPU context.
    pub async fn new(config: &GpuConfig) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Self::backends_from_config(config),
            flags: Default::default(),
            backend_options: Default::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: config.power_preference.into(),
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| DpbError::Gpu(format!("Failed to find suitable GPU adapter: {e}")))?;

        let adapter_info = adapter.get_info();
        tracing::info!(
            "Selected GPU adapter: {} ({:?})",
            adapter_info.name,
            adapter_info.backend
        );

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("DPB GPU Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                    experimental_features: Default::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .map_err(|e| DpbError::Gpu(format!("Failed to create device: {}", e)))?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter_info,
        })
    }

    /// Creates a new GPU context with default configuration.
    pub async fn new_default() -> Result<Self> {
        Self::new(&GpuConfig::default()).await
    }

    /// Returns the backend being used.
    pub fn backend(&self) -> wgpu::Backend {
        self.adapter_info.backend
    }

    /// Returns the device name.
    pub fn device_name(&self) -> &str {
        &self.adapter_info.name
    }

    /// Returns the device type.
    pub fn device_type(&self) -> wgpu::DeviceType {
        self.adapter_info.device_type
    }

    /// Creates a compute pipeline from shader source.
    pub fn create_compute_pipeline(
        &self,
        shader_source: &str,
        entry_point: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> Result<wgpu::ComputePipeline> {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(pipeline)
    }

    /// Creates a bind group layout.
    pub fn create_bind_group_layout(
        &self,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries,
        })
    }

    /// Creates a bind group.
    pub fn create_bind_group(
        &self,
        layout: &wgpu::BindGroupLayout,
        entries: &[wgpu::BindGroupEntry],
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout,
            entries,
        })
    }

    /// Submits a command buffer and waits for completion.
    pub async fn submit_and_wait(&self, command_buffer: wgpu::CommandBuffer) -> Result<()> {
        self.queue.submit(Some(command_buffer));
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .map_err(|e| DpbError::Gpu(format!("Device poll failed: {e}")))?;
        Ok(())
    }

    /// Submits multiple command buffers.
    pub fn submit(&self, command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.queue.submit(command_buffers);
    }

    fn backends_from_config(config: &GpuConfig) -> wgpu::Backends {
        if let Some(ref backend) = config.backend {
            match backend.to_lowercase().as_str() {
                "vulkan" => wgpu::Backends::VULKAN,
                "metal" => wgpu::Backends::METAL,
                "dx12" | "directx12" => wgpu::Backends::DX12,
                "webgpu" => wgpu::Backends::BROWSER_WEBGPU,
                "gl" | "opengl" => wgpu::Backends::GL,
                _ => wgpu::Backends::all(),
            }
        } else {
            wgpu::Backends::all()
        }
    }
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("adapter_info", &self.adapter_info)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bringing up a real device is the only thing that actually exercises the
    /// wgpu descriptor structs; they change shape between major versions and a
    /// clean compile does not prove they are filled in correctly.
    ///
    /// Not every environment has a GPU, so this distinguishes the two failures:
    /// no adapter at all is a skip, but an adapter that refuses to yield a
    /// device is a bug in this file.
    #[tokio::test]
    async fn test_gpu_context_creation() {
        let adapters =
            wgpu::Instance::default().enumerate_adapters(wgpu::Backends::all()).await;
        if adapters.is_empty() {
            eprintln!("no GPU adapter present; skipping device bring-up");
            return;
        }

        let ctx = GpuContext::new_default()
            .await
            .expect("an adapter is present, so device creation must succeed");
        assert!(!ctx.device_name().is_empty(), "adapter reported an empty name");
        println!("GPU: {} ({:?})", ctx.device_name(), ctx.backend());
    }

    /// Upload and read back. This covers the three pieces of surface wgpu 30
    /// changed and a compile check cannot reach: PollType replacing Maintain,
    /// the now-fallible get_mapped_range, and map_async completion.
    #[tokio::test]
    async fn test_gpu_roundtrip_preserves_data() {
        let adapters =
            wgpu::Instance::default().enumerate_adapters(wgpu::Backends::all()).await;
        if adapters.is_empty() {
            eprintln!("no GPU adapter present; skipping buffer round-trip");
            return;
        }

        let ctx = GpuContext::new_default().await.expect("device creation");
        let input: Vec<f32> = (0..256).map(|i| i as f32 * 0.5).collect();

        // MAP_READ is only valid alongside COPY_DST, so this is the usage pair
        // that lets one buffer be both written at creation and mapped back.
        let buffer = crate::gpu::GpuBuffer::from_slice(
            &ctx.device,
            &input,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        );
        let read_back: Vec<f32> = buffer
            .read(&ctx.device)
            .await
            .expect("reading a buffer we just wrote must succeed");

        assert_eq!(read_back, input, "data did not survive the GPU round-trip");
    }
}
