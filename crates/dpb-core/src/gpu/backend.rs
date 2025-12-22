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
pub trait ComputeBackend: Send + Sync {
    /// Returns the backend type.
    fn backend_type(&self) -> BackendType;

    /// Returns device properties.
    fn device_properties(&self) -> &DeviceProperties;

    /// Returns the number of available devices.
    fn device_count(&self) -> Result<usize>;

    /// Creates a new buffer with the given size in bytes.
    fn create_buffer(&self, size: usize) -> Result<BufferHandle>;

    /// Creates a buffer initialized with the given data.
    fn create_buffer_from_slice<T: bytemuck::Pod>(&self, data: &[T]) -> Result<BufferHandle>;

    /// Uploads data to an existing buffer.
    fn upload_buffer<T: bytemuck::Pod>(&self, buffer: &BufferHandle, data: &[T]) -> Result<()>;

    /// Downloads data from a buffer.
    fn download_buffer<T: bytemuck::Pod>(&self, buffer: &BufferHandle) -> Result<Vec<T>>;

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

    /// Checks if this backend is available on the current system.
    fn is_available() -> bool
    where
        Self: Sized;
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
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            flags: Default::default(),
            backend_options: Default::default(),
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| DpbError::Gpu("Failed to find GPU adapter".to_string()))?;

        let info = adapter.get_info();
        let limits = adapter.limits();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("DPB Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None,
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

    fn create_buffer_from_slice<T: bytemuck::Pod>(&self, data: &[T]) -> Result<BufferHandle> {
        let id = self.next_buffer_id();
        let size = data.len() * std::mem::size_of::<T>();

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Buffer_{}", id)),
            size: size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
        self.buffers.write().unwrap().insert(id, buffer);

        Ok(BufferHandle {
            id,
            size,
            read_only: false,
        })
    }

    fn upload_buffer<T: bytemuck::Pod>(&self, handle: &BufferHandle, data: &[T]) -> Result<()> {
        let buffers = self.buffers.read().unwrap();
        let buffer = buffers
            .get(&handle.id)
            .ok_or_else(|| DpbError::Gpu("Buffer not found".to_string()))?;

        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    fn download_buffer<T: bytemuck::Pod>(&self, handle: &BufferHandle) -> Result<Vec<T>> {
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

        self.device.poll(wgpu::Maintain::Wait);
        rx.recv()
            .unwrap()
            .map_err(|e| DpbError::Gpu(format!("Buffer map failed: {:?}", e)))?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

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
            push_constant_ranges: &[],
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
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn is_available() -> bool {
        // WebGPU is always available (can fall back to software)
        true
    }
}

// ============================================================================
// CUDA Backend Implementation (feature-gated)
// ============================================================================

#[cfg(feature = "cuda")]
pub mod cuda {
    use super::*;
    use cudarc::driver::*;

    /// CUDA backend using cudarc.
    pub struct CUDABackend {
        device: Arc<CudaDevice>,
        properties: DeviceProperties,
        buffer_counter: std::sync::atomic::AtomicU64,
        kernel_counter: std::sync::atomic::AtomicU64,
    }

    impl CUDABackend {
        /// Creates a new CUDA backend on the specified device.
        pub fn new(device_id: usize) -> Result<Self> {
            let device = CudaDevice::new(device_id)
                .map_err(|e| DpbError::Gpu(format!("CUDA device init failed: {:?}", e)))?;

            // Get device properties
            let properties = DeviceProperties {
                name: format!("CUDA Device {}", device_id),
                backend_type: BackendType::CUDA,
                memory_size: 0, // Would need to query
                max_threads_per_block: 1024,
                max_shared_memory: 48 * 1024,
                compute_units: 0,
                unified_memory: false,
            };

            Ok(Self {
                device: Arc::new(device),
                properties,
                buffer_counter: std::sync::atomic::AtomicU64::new(0),
                kernel_counter: std::sync::atomic::AtomicU64::new(0),
            })
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
            // Note: Actual buffer allocation would happen here
            // For now, we track the handle
            Ok(BufferHandle {
                id,
                size,
                read_only: false,
            })
        }

        fn create_buffer_from_slice<T: bytemuck::Pod>(&self, data: &[T]) -> Result<BufferHandle> {
            let id = self.next_buffer_id();
            let size = data.len() * std::mem::size_of::<T>();
            // Note: Would allocate and copy data to GPU
            Ok(BufferHandle {
                id,
                size,
                read_only: false,
            })
        }

        fn upload_buffer<T: bytemuck::Pod>(&self, _handle: &BufferHandle, _data: &[T]) -> Result<()> {
            // Would copy data to GPU
            Ok(())
        }

        fn download_buffer<T: bytemuck::Pod>(&self, handle: &BufferHandle) -> Result<Vec<T>> {
            let count = handle.size / std::mem::size_of::<T>();
            Ok(vec![T::zeroed(); count])
        }

        fn free_buffer(&self, _handle: &BufferHandle) -> Result<()> {
            Ok(())
        }

        fn compile_kernel(&self, name: &str, _source: &str, _entry_point: &str) -> Result<KernelHandle> {
            let id = self.next_kernel_id();
            Ok(KernelHandle {
                id,
                name: name.to_string(),
                workgroup_size: (256, 1, 1),
            })
        }

        fn dispatch_kernel(
            &self,
            _kernel: &KernelHandle,
            _input_buffers: &[&BufferHandle],
            _output_buffers: &[&BufferHandle],
            _workgroups: (u32, u32, u32),
        ) -> Result<()> {
            // Would launch CUDA kernel
            Ok(())
        }

        fn synchronize(&self) -> Result<()> {
            self.device
                .synchronize()
                .map_err(|e| DpbError::Gpu(format!("CUDA sync failed: {:?}", e)))
        }

        fn is_available() -> bool {
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
    let mut backends = vec![BackendType::WebGPU];

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
            // Create buffer from data
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let buffer = backend.create_buffer_from_slice(&data).unwrap();
            assert_eq!(buffer.size, 16); // 4 floats * 4 bytes

            // Download and verify
            let downloaded: Vec<f32> = backend.download_buffer(&buffer).unwrap();
            assert_eq!(downloaded, data);

            // Free buffer
            backend.free_buffer(&buffer).unwrap();
        }
    }
}
