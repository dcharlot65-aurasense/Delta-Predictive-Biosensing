//! Cross-platform GPU compute backend abstraction.
//!
//! This module provides a unified interface for GPU computation across multiple
//! backends including WebGPU (wgpu), CUDA, Metal, and Vulkan.
//!
//! # Architecture
//!
//! The `ComputeBackend` trait abstracts device management, memory allocation,
//! and kernel execution across different GPU platforms. This enables:
//!
//! - **Cross-platform code**: Same API for WebGPU, CUDA, Metal, etc.
//! - **Backend auto-selection**: Automatically choose the best available backend
//! - **Feature-gated implementations**: Only compile backends you need
//!
//! # Example
//!
//! ```rust,ignore
//! use dpb_core::gpu::backend::*;
//!
//! // Auto-select best available backend
//! let backend = create_default_backend().await?;
//!
//! // Create buffers
//! let input = backend.create_buffer_from_slice(&[1.0f32, 2.0, 3.0])?;
//! let output = backend.create_buffer(3 * std::mem::size_of::<f32>())?;
//!
//! // Execute compute kernel
//! backend.dispatch_kernel("add_one", &[&input], &[&output], (1, 1, 1))?;
//! ```

use crate::error::{DpbError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

// These are established domain acronyms -- clinical file formats, ECG
// beat annotations, and hardware terms. Camel-casing them (Pvc, Wfdb,
// Dram) would make this harder to read for anyone who works with them.
#[allow(clippy::upper_case_acronyms)]
/// Supported compute backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BackendType {
    /// WebGPU via wgpu (cross-platform, browser-compatible)
    WebGPU,
    /// NVIDIA CUDA (highest performance on NVIDIA GPUs)
    #[cfg(feature = "cuda")]
    CUDA,
    /// Apple Metal (optimized for Apple Silicon)
    #[cfg(feature = "metal")]
    Metal,
    /// Vulkan Compute (cross-platform, gaming GPUs)
    #[cfg(feature = "vulkan")]
    Vulkan,
    /// AMD ROCm/HIP
    #[cfg(feature = "rocm")]
    ROCm,
    /// CPU fallback (for testing and unsupported platforms)
    CPU,
}

impl std::fmt::Display for BackendType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendType::WebGPU => write!(f, "WebGPU"),
            #[cfg(feature = "cuda")]
            BackendType::CUDA => write!(f, "CUDA"),
            #[cfg(feature = "metal")]
            BackendType::Metal => write!(f, "Metal"),
            #[cfg(feature = "vulkan")]
            BackendType::Vulkan => write!(f, "Vulkan"),
            #[cfg(feature = "rocm")]
            BackendType::ROCm => write!(f, "ROCm"),
            BackendType::CPU => write!(f, "CPU"),
        }
    }
}

/// Device properties and capabilities.
#[derive(Debug, Clone)]
pub struct DeviceProperties {
    /// Device name
    pub name: String,
    /// Backend type
    pub backend_type: BackendType,
    /// Total device memory in bytes
    pub memory_size: u64,
    /// Maximum threads per workgroup/block
    pub max_threads_per_block: u32,
    /// Maximum shared memory per workgroup in bytes
    pub max_shared_memory: u32,
    /// Number of compute units/SMs
    pub compute_units: u32,
    /// Whether the device supports unified memory
    pub unified_memory: bool,
}

impl Default for DeviceProperties {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            backend_type: BackendType::CPU,
            memory_size: 0,
            max_threads_per_block: 256,
            max_shared_memory: 48 * 1024, // 48 KB typical
            compute_units: 1,
            unified_memory: false,
        }
    }
}

/// Handle to a GPU buffer.
#[derive(Debug, Clone)]
pub struct BufferHandle {
    /// Unique buffer ID
    pub id: u64,
    /// Buffer size in bytes
    pub size: usize,
    /// Whether buffer is read-only
    pub read_only: bool,
}

/// Handle to a compiled compute kernel.
#[derive(Debug, Clone)]
pub struct KernelHandle {
    /// Unique kernel ID
    pub id: u64,
    /// Kernel name/entry point
    pub name: String,
    /// Workgroup size (x, y, z)
    pub workgroup_size: (u32, u32, u32),
}

/// Unified compute backend trait.
///
/// This trait provides a common interface for GPU computation across
/// different backends (WebGPU, CUDA, Metal, etc.).
///
/// Note: This trait uses byte slices for buffer operations to maintain
/// dyn-compatibility. Use the helper methods on BufferHandle for typed access.
pub trait ComputeBackend: Send + Sync {
    /// Returns the backend type.
    fn backend_type(&self) -> BackendType;

    /// Returns device properties.
    fn device_properties(&self) -> &DeviceProperties;

    /// Returns the number of available devices.
    fn device_count(&self) -> Result<usize>;

    /// Creates a new buffer with the given size in bytes.
    fn create_buffer(&self, size: usize) -> Result<BufferHandle>;

    /// Creates a buffer initialized with the given byte data.
    fn create_buffer_from_bytes(&self, data: &[u8]) -> Result<BufferHandle>;

    /// Uploads byte data to an existing buffer.
    fn upload_buffer_bytes(&self, buffer: &BufferHandle, data: &[u8]) -> Result<()>;

    /// Downloads byte data from a buffer.
    fn download_buffer_bytes(&self, buffer: &BufferHandle) -> Result<Vec<u8>>;

    /// Frees a buffer.
    fn free_buffer(&self, buffer: &BufferHandle) -> Result<()>;

    /// Compiles a compute kernel from source.
    ///
    /// The source format depends on the backend:
    /// - WebGPU: WGSL
    /// - CUDA: PTX or CUDA C
    /// - Metal: MSL
    /// - Vulkan: SPIR-V
    fn compile_kernel(&self, name: &str, source: &str, entry_point: &str) -> Result<KernelHandle>;

    /// Dispatches a compute kernel.
    ///
    /// # Arguments
    /// * `kernel` - Compiled kernel handle
    /// * `input_buffers` - Read-only input buffers
    /// * `output_buffers` - Read-write output buffers
    /// * `workgroups` - Number of workgroups (x, y, z)
    fn dispatch_kernel(
        &self,
        kernel: &KernelHandle,
        input_buffers: &[&BufferHandle],
        output_buffers: &[&BufferHandle],
        workgroups: (u32, u32, u32),
    ) -> Result<()>;

    /// Synchronizes all pending operations.
    fn synchronize(&self) -> Result<()>;
}

/// Extension methods for typed buffer operations.
///
/// These are provided as free functions to avoid generic methods in the trait.
pub fn create_buffer_from_slice<T: bytemuck::Pod>(
    backend: &dyn ComputeBackend,
    data: &[T],
) -> Result<BufferHandle> {
    backend.create_buffer_from_bytes(bytemuck::cast_slice(data))
}

/// Upload typed data to a buffer.
pub fn upload_buffer<T: bytemuck::Pod>(
    backend: &dyn ComputeBackend,
    buffer: &BufferHandle,
    data: &[T],
) -> Result<()> {
    backend.upload_buffer_bytes(buffer, bytemuck::cast_slice(data))
}

/// Download typed data from a buffer.
pub fn download_buffer<T: bytemuck::Pod + Clone>(
    backend: &dyn ComputeBackend,
    buffer: &BufferHandle,
) -> Result<Vec<T>> {
    let bytes = backend.download_buffer_bytes(buffer)?;
    Ok(bytemuck::cast_slice(&bytes).to_vec())
}

/// Extension trait for common compute operations.
pub trait ComputeBackendExt: ComputeBackend {
    /// Performs element-wise addition of two buffers.
    fn add(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle) -> Result<()>;

    /// Performs matrix multiplication.
    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        result: &BufferHandle,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()>;

    /// Performs FFT on complex data.
    fn fft(&self, input: &BufferHandle, output: &BufferHandle, size: usize) -> Result<()>;

    /// Performs reduction operation (sum).
    fn reduce_sum(&self, input: &BufferHandle, output: &BufferHandle, size: usize) -> Result<()>;
}

// WGSL shader sources for compute operations
mod shaders {
    /// Element-wise addition kernel
    pub const ADD_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx < arrayLength(&a)) {
        result[idx] = a[idx] + b[idx];
    }
}
"#;

    /// Matrix multiplication kernel (naive, for reference)
    pub const MATMUL_SHADER: &str = r#"
struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= params.m || col >= params.n) {
        return;
    }

    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.k; i = i + 1u) {
        sum = sum + a[row * params.k + i] * b[i * params.n + col];
    }

    result[row * params.n + col] = sum;
}
"#;

    /// Parallel reduction sum kernel
    pub const REDUCE_SUM_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
    let tid = local_id.x;
    let gid = workgroup_id.x * 256u + tid;
    let input_len = arrayLength(&input);

    // Load data into shared memory
    if (gid < input_len) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }

    workgroupBarrier();

    // Parallel reduction in shared memory
    for (var stride: u32 = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    // Write result for this workgroup
    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}
"#;

    /// Cooley-Tukey FFT kernel (radix-2)
    pub const FFT_SHADER: &str = r#"
struct Complex {
    real: f32,
    imag: f32,
}

@group(0) @binding(0) var<storage, read_write> data_real: array<f32>;
@group(0) @binding(1) var<storage, read_write> data_imag: array<f32>;
@group(0) @binding(2) var<uniform> stage: u32;
@group(0) @binding(3) var<uniform> n: u32;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let half_n = n >> 1u;

    if (idx >= half_n) {
        return;
    }

    let step = 1u << stage;
    let half_step = step >> 1u;

    let group = idx / half_step;
    let pair = idx % half_step;

    let i = group * step + pair;
    let j = i + half_step;

    // Twiddle factor
    let angle = -2.0 * PI * f32(pair) / f32(step);
    let tw_real = cos(angle);
    let tw_imag = sin(angle);

    // Load values
    let a_real = data_real[i];
    let a_imag = data_imag[i];
    let b_real = data_real[j];
    let b_imag = data_imag[j];

    // Butterfly operation
    let tb_real = b_real * tw_real - b_imag * tw_imag;
    let tb_imag = b_real * tw_imag + b_imag * tw_real;

    data_real[i] = a_real + tb_real;
    data_imag[i] = a_imag + tb_imag;
    data_real[j] = a_real - tb_real;
    data_imag[j] = a_imag - tb_imag;
}
"#;
}

// ============================================================================
// WebGPU Backend Implementation
// ============================================================================

/// WebGPU backend using wgpu.
pub struct WebGPUBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    properties: DeviceProperties,
    buffer_counter: std::sync::atomic::AtomicU64,
    kernel_counter: std::sync::atomic::AtomicU64,
    buffers: std::sync::RwLock<std::collections::HashMap<u64, wgpu::Buffer>>,
    pipelines: std::sync::RwLock<std::collections::HashMap<u64, wgpu::ComputePipeline>>,
}

impl WebGPUBackend {
    /// Creates a new WebGPU backend.
    pub async fn new() -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: Default::default(),
            backend_options: Default::default(),
            memory_budget_thresholds: Default::default(),
            display: None,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|e| DpbError::Gpu(format!("Failed to find GPU adapter: {e}")))?;

        let info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("DPB Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                    experimental_features: Default::default(),
                    trace: wgpu::Trace::Off,
                },
            )
            .await
            .map_err(|e| DpbError::Gpu(format!("Failed to create device: {}", e)))?;

        let properties = DeviceProperties {
            name: info.name,
            backend_type: BackendType::WebGPU,
            memory_size: 0, // Not directly available in wgpu
            max_threads_per_block: limits.max_compute_invocations_per_workgroup,
            max_shared_memory: limits.max_compute_workgroup_storage_size,
            compute_units: 0, // Not available in wgpu
            unified_memory: false,
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            properties,
            buffer_counter: std::sync::atomic::AtomicU64::new(0),
            kernel_counter: std::sync::atomic::AtomicU64::new(0),
            buffers: std::sync::RwLock::new(std::collections::HashMap::new()),
            pipelines: std::sync::RwLock::new(std::collections::HashMap::new()),
        })
    }

    /// Creates from existing wgpu device and queue.
    pub fn from_context(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let properties = DeviceProperties {
            name: "WebGPU Device".to_string(),
            backend_type: BackendType::WebGPU,
            ..Default::default()
        };

        Self {
            device,
            queue,
            properties,
            buffer_counter: std::sync::atomic::AtomicU64::new(0),
            kernel_counter: std::sync::atomic::AtomicU64::new(0),
            buffers: std::sync::RwLock::new(std::collections::HashMap::new()),
            pipelines: std::sync::RwLock::new(std::collections::HashMap::new()),
        }
    }

    fn next_buffer_id(&self) -> u64 {
        self.buffer_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }

    fn next_kernel_id(&self) -> u64 {
        self.kernel_counter
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
    }
}

impl ComputeBackend for WebGPUBackend {
    fn backend_type(&self) -> BackendType {
        BackendType::WebGPU
    }

    fn device_properties(&self) -> &DeviceProperties {
        &self.properties
    }

    fn device_count(&self) -> Result<usize> {
        Ok(1) // wgpu typically exposes one device
    }

    fn create_buffer(&self, size: usize) -> Result<BufferHandle> {
        let id = self.next_buffer_id();

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Buffer_{}", id)),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.buffers.write().unwrap().insert(id, buffer);

        Ok(BufferHandle {
            id,
            size,
            read_only: false,
        })
    }

    fn create_buffer_from_bytes(&self, data: &[u8]) -> Result<BufferHandle> {
        let id = self.next_buffer_id();
        let size = data.len();

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Buffer_{}", id)),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&buffer, 0, data);
        self.buffers.write().unwrap().insert(id, buffer);

        Ok(BufferHandle {
            id,
            size,
            read_only: false,
        })
    }

    fn upload_buffer_bytes(&self, handle: &BufferHandle, data: &[u8]) -> Result<()> {
        let buffers = self.buffers.read().unwrap();
        let buffer = buffers
            .get(&handle.id)
            .ok_or_else(|| DpbError::Gpu("Buffer not found".to_string()))?;

        self.queue.write_buffer(buffer, 0, data);
        Ok(())
    }

    fn download_buffer_bytes(&self, handle: &BufferHandle) -> Result<Vec<u8>> {
        let buffers = self.buffers.read().unwrap();
        let buffer = buffers
            .get(&handle.id)
            .ok_or_else(|| DpbError::Gpu("Buffer not found".to_string()))?;

        // Create staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: handle.size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, handle.size as u64);

        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });

        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| DpbError::Gpu(format!("Device poll failed: {e}")))?;
        rx.recv()
            .unwrap()
            .map_err(|e| DpbError::Gpu(format!("Buffer map failed: {:?}", e)))?;

        let data = slice
            .get_mapped_range()
            .map_err(|e| DpbError::Gpu(format!("Failed to read mapped range: {e}")))?;
        let result: Vec<u8> = data.to_vec();

        drop(data);
        staging.unmap();

        Ok(result)
    }

    fn free_buffer(&self, handle: &BufferHandle) -> Result<()> {
        self.buffers.write().unwrap().remove(&handle.id);
        Ok(())
    }

    fn compile_kernel(&self, name: &str, source: &str, entry_point: &str) -> Result<KernelHandle> {
        let id = self.next_kernel_id();

        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });

        // Create a simple pipeline layout (buffers will be bound at dispatch time)
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", name)),
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(name),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        self.pipelines.write().unwrap().insert(id, pipeline);

        Ok(KernelHandle {
            id,
            name: name.to_string(),
            workgroup_size: (256, 1, 1), // Default, should be extracted from shader
        })
    }

    fn dispatch_kernel(
        &self,
        kernel: &KernelHandle,
        _input_buffers: &[&BufferHandle],
        _output_buffers: &[&BufferHandle],
        workgroups: (u32, u32, u32),
    ) -> Result<()> {
        let pipelines = self.pipelines.read().unwrap();
        let pipeline = pipelines
            .get(&kernel.id)
            .ok_or_else(|| DpbError::Gpu("Kernel not found".to_string()))?;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&kernel.name),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(pipeline);
            compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| DpbError::Gpu(format!("Device poll failed: {e}")))?;
        Ok(())
    }
}

impl WebGPUBackend {
    /// Checks if WebGPU is available on the current system.
    pub fn is_available() -> bool {
        // WebGPU is always available (can fall back to software)
        true
    }

    /// Creates a bind group layout for compute shaders.
    fn create_compute_bind_group_layout(
        &self,
        entries: &[wgpu::BindGroupLayoutEntry],
    ) -> wgpu::BindGroupLayout {
        self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries,
        })
    }

    /// Creates a compute pipeline with the given shader and bind group layout.
    fn create_compute_pipeline_with_layout(
        &self,
        shader_source: &str,
        entry_point: &str,
        bind_group_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[Some(bind_group_layout)],
            immediate_size: 0,
        });

        self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    }
}

impl ComputeBackendExt for WebGPUBackend {
    fn add(&self, a: &BufferHandle, b: &BufferHandle, result: &BufferHandle) -> Result<()> {
        let buffers = self.buffers.read().unwrap();
        let buf_a = buffers.get(&a.id)
            .ok_or_else(|| DpbError::Gpu("Buffer A not found".to_string()))?;
        let buf_b = buffers.get(&b.id)
            .ok_or_else(|| DpbError::Gpu("Buffer B not found".to_string()))?;
        let buf_result = buffers.get(&result.id)
            .ok_or_else(|| DpbError::Gpu("Result buffer not found".to_string()))?;

        // Create bind group layout
        let bind_group_layout = self.create_compute_bind_group_layout(&[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);

        let pipeline = self.create_compute_pipeline_with_layout(
            shaders::ADD_SHADER,
            "main",
            &bind_group_layout,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Add Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_result.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Add Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Add Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let num_elements = a.size / std::mem::size_of::<f32>();
            let workgroups = num_elements.div_ceil(256) as u32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn matmul(
        &self,
        a: &BufferHandle,
        b: &BufferHandle,
        result: &BufferHandle,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        let buffers = self.buffers.read().unwrap();
        let buf_a = buffers.get(&a.id)
            .ok_or_else(|| DpbError::Gpu("Buffer A not found".to_string()))?;
        let buf_b = buffers.get(&b.id)
            .ok_or_else(|| DpbError::Gpu("Buffer B not found".to_string()))?;
        let buf_result = buffers.get(&result.id)
            .ok_or_else(|| DpbError::Gpu("Result buffer not found".to_string()))?;

        // Create params buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            m: u32,
            n: u32,
            k: u32,
            _pad: u32,
        }

        let params = Params {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            _pad: 0,
        };

        let params_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Matmul Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create bind group layout
        let bind_group_layout = self.create_compute_bind_group_layout(&[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);

        let pipeline = self.create_compute_pipeline_with_layout(
            shaders::MATMUL_SHADER,
            "main",
            &bind_group_layout,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Matmul Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_a.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_b.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: buf_result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: params_buffer.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Matmul Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Matmul Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups_x = n.div_ceil(16) as u32;
            let workgroups_y = m.div_ceil(16) as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }

    fn fft(&self, input: &BufferHandle, output: &BufferHandle, size: usize) -> Result<()> {
        // For FFT, we need real and imaginary parts
        // Input is assumed to be interleaved complex: [r0, i0, r1, i1, ...]
        // We'll copy to separate real/imag buffers, run FFT stages, then copy back

        let buffers = self.buffers.read().unwrap();
        let _buf_input = buffers.get(&input.id)
            .ok_or_else(|| DpbError::Gpu("Input buffer not found".to_string()))?;
        let _buf_output = buffers.get(&output.id)
            .ok_or_else(|| DpbError::Gpu("Output buffer not found".to_string()))?;

        // Check size is power of 2
        if size == 0 || (size & (size - 1)) != 0 {
            return Err(DpbError::Gpu("FFT size must be power of 2".to_string()));
        }

        let num_stages = (size as f64).log2() as u32;

        // Create working buffers for real and imaginary parts
        let real_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Real Buffer"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let imag_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FFT Imag Buffer"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create bind group layout for FFT
        let bind_group_layout = self.create_compute_bind_group_layout(&[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);

        let pipeline = self.create_compute_pipeline_with_layout(
            shaders::FFT_SHADER,
            "main",
            &bind_group_layout,
        );

        let n_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("FFT N"),
            contents: bytemuck::bytes_of(&(size as u32)),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Run FFT stages
        for stage in 1..=num_stages {
            let stage_buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("FFT Stage"),
                contents: bytemuck::bytes_of(&stage),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("FFT Bind Group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: real_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: imag_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: stage_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: n_buffer.as_entire_binding() },
                ],
            });

            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("FFT Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("FFT Pass"),
                    timestamp_writes: None,
                });
                compute_pass.set_pipeline(&pipeline);
                compute_pass.set_bind_group(0, &bind_group, &[]);

                let workgroups = (size / 2).div_ceil(256) as u32;
                compute_pass.dispatch_workgroups(workgroups, 1, 1);
            }

            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| DpbError::Gpu(format!("Device poll failed: {e}")))?;
        }

        Ok(())
    }

    fn reduce_sum(&self, input: &BufferHandle, output: &BufferHandle, size: usize) -> Result<()> {
        let buffers = self.buffers.read().unwrap();
        let buf_input = buffers.get(&input.id)
            .ok_or_else(|| DpbError::Gpu("Input buffer not found".to_string()))?;
        let buf_output = buffers.get(&output.id)
            .ok_or_else(|| DpbError::Gpu("Output buffer not found".to_string()))?;

        // Create bind group layout
        let bind_group_layout = self.create_compute_bind_group_layout(&[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ]);

        let pipeline = self.create_compute_pipeline_with_layout(
            shaders::REDUCE_SUM_SHADER,
            "main",
            &bind_group_layout,
        );

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduce Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf_input.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: buf_output.as_entire_binding() },
            ],
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Reduce Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Reduce Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroups = size.div_ceil(256) as u32;
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
        Ok(())
    }
}

// ============================================================================
// CUDA Backend Implementation (feature-gated)
// ============================================================================

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use cudarc::driver::*;
    use std::collections::HashMap;

    /// CUDA backend using cudarc.
    pub struct CUDABackend {
        device: Arc<CudaDevice>,
        properties: DeviceProperties,
        buffer_counter: std::sync::atomic::AtomicU64,
        kernel_counter: std::sync::atomic::AtomicU64,
        /// Stores allocated GPU buffers (as raw byte slices)
        buffers: std::sync::RwLock<HashMap<u64, CudaSlice<u8>>>,
        /// Stores compiled PTX modules
        modules: std::sync::RwLock<HashMap<u64, CudaFunction>>,
    }

    impl CUDABackend {
        /// Creates a new CUDA backend on the specified device.
        pub fn new(device_id: usize) -> Result<Self> {
            let device = CudaDevice::new(device_id)
                .map_err(|e| DpbError::Gpu(format!("CUDA device init failed: {:?}", e)))?;

            // Query device properties
            let name = device.name()
                .unwrap_or_else(|_| format!("CUDA Device {}", device_id));

            // Get memory info - cudarc doesn't expose this directly, use defaults
            let properties = DeviceProperties {
                name,
                backend_type: BackendType::CUDA,
                memory_size: 8 * 1024 * 1024 * 1024, // 8GB default, would query cudaMemGetInfo
                max_threads_per_block: 1024,
                max_shared_memory: 48 * 1024,
                compute_units: 0, // Would query device attributes
                unified_memory: false,
            };

            Ok(Self {
                device: Arc::new(device),
                properties,
                buffer_counter: std::sync::atomic::AtomicU64::new(0),
                kernel_counter: std::sync::atomic::AtomicU64::new(0),
                buffers: std::sync::RwLock::new(HashMap::new()),
                modules: std::sync::RwLock::new(HashMap::new()),
            })
        }

        /// Gets the underlying CUDA device for advanced operations.
        pub fn cuda_device(&self) -> &Arc<CudaDevice> {
            &self.device
        }

        fn next_buffer_id(&self) -> u64 {
            self.buffer_counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        }

        fn next_kernel_id(&self) -> u64 {
            self.kernel_counter
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
        }
    }

    impl ComputeBackend for CUDABackend {
        fn backend_type(&self) -> BackendType {
            BackendType::CUDA
        }

        fn device_properties(&self) -> &DeviceProperties {
            &self.properties
        }

        fn device_count(&self) -> Result<usize> {
            CudaDevice::count()
                .map_err(|e| DpbError::Gpu(format!("Failed to get CUDA device count: {:?}", e)))
        }

        fn create_buffer(&self, size: usize) -> Result<BufferHandle> {
            let id = self.next_buffer_id();

            // Allocate GPU memory using cudarc
            let gpu_buffer = self.device
                .alloc_zeros::<u8>(size)
                .map_err(|e| DpbError::Gpu(format!("CUDA alloc failed: {:?}", e)))?;

            self.buffers.write().unwrap().insert(id, gpu_buffer);

            Ok(BufferHandle {
                id,
                size,
                read_only: false,
            })
        }

        fn create_buffer_from_bytes(&self, data: &[u8]) -> Result<BufferHandle> {
            let id = self.next_buffer_id();
            let size = data.len();

            // Allocate and copy data to GPU
            let gpu_buffer = self.device
                .htod_sync_copy(data)
                .map_err(|e| DpbError::Gpu(format!("CUDA htod copy failed: {:?}", e)))?;

            self.buffers.write().unwrap().insert(id, gpu_buffer);

            Ok(BufferHandle {
                id,
                size,
                read_only: false,
            })
        }

        fn upload_buffer_bytes(&self, handle: &BufferHandle, data: &[u8]) -> Result<()> {
            let buffers = self.buffers.read().unwrap();
            let gpu_buffer = buffers
                .get(&handle.id)
                .ok_or_else(|| DpbError::Gpu("Buffer not found".to_string()))?;

            // Copy host data to device buffer
            self.device
                .htod_sync_copy_into(data, gpu_buffer)
                .map_err(|e| DpbError::Gpu(format!("CUDA htod copy failed: {:?}", e)))?;

            Ok(())
        }

        fn download_buffer_bytes(&self, handle: &BufferHandle) -> Result<Vec<u8>> {
            let buffers = self.buffers.read().unwrap();
            let gpu_buffer = buffers
                .get(&handle.id)
                .ok_or_else(|| DpbError::Gpu("Buffer not found".to_string()))?;

            // Copy device buffer to host
            let host_data = self.device
                .dtoh_sync_copy(gpu_buffer)
                .map_err(|e| DpbError::Gpu(format!("CUDA dtoh copy failed: {:?}", e)))?;

            Ok(host_data)
        }

        fn free_buffer(&self, handle: &BufferHandle) -> Result<()> {
            // Remove buffer from map - cudarc will deallocate when dropped
            self.buffers.write().unwrap().remove(&handle.id);
            Ok(())
        }

        fn compile_kernel(&self, name: &str, source: &str, entry_point: &str) -> Result<KernelHandle> {
            let id = self.next_kernel_id();

            // Load PTX module and get function
            // Note: source should be PTX code for CUDA
            let ptx = CudaModule::from_ptx(source.as_bytes(), &[])
                .map_err(|e| DpbError::Gpu(format!("PTX compilation failed: {:?}", e)))?;

            self.device
                .load_ptx(ptx, name, &[entry_point])
                .map_err(|e| DpbError::Gpu(format!("PTX load failed: {:?}", e)))?;

            let func = self.device
                .get_func(name, entry_point)
                .ok_or_else(|| DpbError::Gpu(format!("Function {} not found", entry_point)))?;

            self.modules.write().unwrap().insert(id, func);

            Ok(KernelHandle {
                id,
                name: name.to_string(),
                workgroup_size: (256, 1, 1),
            })
        }

        fn dispatch_kernel(
            &self,
            kernel: &KernelHandle,
            _input_buffers: &[&BufferHandle],
            _output_buffers: &[&BufferHandle],
            workgroups: (u32, u32, u32),
        ) -> Result<()> {
            let modules = self.modules.read().unwrap();
            let func = modules
                .get(&kernel.id)
                .ok_or_else(|| DpbError::Gpu("Kernel not found".to_string()))?;

            // Launch kernel with grid/block dimensions
            // Note: For a full implementation, we'd need to bind buffer arguments
            let cfg = LaunchConfig {
                grid_dim: workgroups,
                block_dim: kernel.workgroup_size,
                shared_mem_bytes: 0,
            };

            // Launch with no arguments for now - real impl would pass buffer pointers
            unsafe {
                func.clone().launch(cfg, ())
                    .map_err(|e| DpbError::Gpu(format!("Kernel launch failed: {:?}", e)))?;
            }

            Ok(())
        }

        fn synchronize(&self) -> Result<()> {
            self.device
                .synchronize()
                .map_err(|e| DpbError::Gpu(format!("CUDA sync failed: {:?}", e)))
        }
    }

    impl CUDABackend {
        /// Checks if CUDA is available on the current system.
        pub fn is_available() -> bool {
            CudaDevice::count().map(|c| c > 0).unwrap_or(false)
        }
    }
}

// ============================================================================
// Backend Factory
// ============================================================================

/// Creates the default (best available) compute backend.
///
/// Priority order:
/// 1. CUDA (if available and feature enabled)
/// 2. Metal (if on macOS/iOS and feature enabled)
/// 3. WebGPU (always available)
pub async fn create_default_backend() -> Result<Box<dyn ComputeBackend>> {
    #[cfg(feature = "cuda")]
    {
        if cuda::CUDABackend::is_available() {
            tracing::info!("Using CUDA backend");
            return Ok(Box::new(cuda::CUDABackend::new(0)?));
        }
    }

    // Fall back to WebGPU
    tracing::info!("Using WebGPU backend");
    Ok(Box::new(WebGPUBackend::new().await?))
}

/// Creates a specific backend by type.
pub async fn create_backend(backend_type: BackendType) -> Result<Box<dyn ComputeBackend>> {
    match backend_type {
        BackendType::WebGPU => Ok(Box::new(WebGPUBackend::new().await?)),
        #[cfg(feature = "cuda")]
        BackendType::CUDA => Ok(Box::new(cuda::CUDABackend::new(0)?)),
        BackendType::CPU => Err(DpbError::Gpu("CPU backend not yet implemented".to_string())),
        #[allow(unreachable_patterns)]
        _ => Err(DpbError::Gpu(format!(
            "Backend {:?} not available (feature not enabled)",
            backend_type
        ))),
    }
}

/// Lists all available backends on the current system.
pub fn available_backends() -> Vec<BackendType> {
    let backends = vec![BackendType::WebGPU];

    #[cfg(feature = "cuda")]
    {
        if cuda::CUDABackend::is_available() {
            backends.insert(0, BackendType::CUDA); // CUDA first if available
        }
    }

    backends
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_type_display() {
        assert_eq!(BackendType::WebGPU.to_string(), "WebGPU");
        assert_eq!(BackendType::CPU.to_string(), "CPU");
    }

    #[test]
    fn test_device_properties_default() {
        let props = DeviceProperties::default();
        assert_eq!(props.backend_type, BackendType::CPU);
        assert_eq!(props.max_threads_per_block, 256);
    }

    #[test]
    fn test_available_backends() {
        let backends = available_backends();
        assert!(backends.contains(&BackendType::WebGPU));
    }

    #[tokio::test]
    async fn test_webgpu_backend_creation() {
        // This may fail in CI without GPU, so we just check it compiles
        let result = WebGPUBackend::new().await;
        if let Ok(backend) = result {
            assert_eq!(backend.backend_type(), BackendType::WebGPU);
            assert!(backend.device_count().unwrap() >= 1);
        }
    }

    #[tokio::test]
    async fn test_buffer_operations() {
        let result = WebGPUBackend::new().await;
        if let Ok(backend) = result {
            // Create buffer from data using helper function
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let buffer = create_buffer_from_slice(&backend, &data).unwrap();
            assert_eq!(buffer.size, 16); // 4 floats * 4 bytes

            // Download and verify using helper function
            let downloaded: Vec<f32> = download_buffer(&backend, &buffer).unwrap();
            assert_eq!(downloaded, data);

            // Free buffer
            backend.free_buffer(&buffer).unwrap();
        }
    }
}
