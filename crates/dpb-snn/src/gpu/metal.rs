//! Metal Backend for GPU Acceleration on Apple Silicon
//!
//! This module provides Apple Metal support for GPU-accelerated SNN operations
//! on macOS and iOS devices.
//!
//! ## Features
//!
//! - **Unified Memory**: Efficient memory sharing between CPU and GPU
//! - **Compute Pipelines**: Optimized Metal shaders for SNN operations
//! - **Command Buffers**: Batched GPU operations for efficiency
//! - **Metal Performance Shaders**: Hardware-optimized primitives
//!
//! ## Architecture
//!
//! The Metal backend uses:
//! - MTLDevice for device management
//! - MTLBuffer for memory allocation
//! - MTLComputePipelineState for kernel execution
//! - MTLCommandQueue for command submission
//!
//! ## Example
//!
//! ```rust
//! use dpb_snn::gpu::metal::{MetalDevice, MetalCommandQueue};
//!
//! # #[cfg(feature = "metal")]
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = MetalDevice::new(0)?;
//! let command_queue = device.create_command_queue()?;
//!
//! // Allocate memory
//! let buffer = device.allocate(1024)?;
//!
//! // Create and submit command buffer
//! let command_buffer = command_queue.create_command_buffer()?;
//! command_buffer.commit()?;
//! # Ok(())
//! # }
//! ```

use super::{Backend, GpuBuffer, GpuDevice, GpuError, GpuResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "metal")]
use metal::{
    Buffer as MtlBuffer, CommandBuffer as MtlCommandBuffer, CommandQueue as MtlCommandQueue,
    ComputeCommandEncoder as MtlComputeEncoder, ComputePipelineState as MtlPipeline,
    Device as MtlDevice, Library as MtlLibrary, MTLResourceOptions,
};

/// Check if Metal is available on this system
///
/// Uses the metal crate to check for available Metal devices.
pub fn is_metal_available() -> bool {
    #[cfg(feature = "metal")]
    {
        MtlDevice::system_default().is_some()
    }
    #[cfg(not(feature = "metal"))]
    {
        false
    }
}

/// List all available Metal devices
pub fn list_metal_devices() -> GpuResult<Vec<(u32, String)>> {
    #[cfg(feature = "metal")]
    {
        let devices = MtlDevice::all();
        Ok(devices
            .into_iter()
            .enumerate()
            .map(|(i, d)| (i as u32, d.name().to_string()))
            .collect())
    }
    #[cfg(not(feature = "metal"))]
    {
        Err(GpuError::BackendNotAvailable(
            "Metal feature not enabled".to_string(),
        ))
    }
}

/// Metal device handle with metal-rs backend
pub struct MetalDevice {
    device_id: u32,
    #[cfg(feature = "metal")]
    mtl_device: MtlDevice,
    #[cfg(feature = "metal")]
    command_queue: MtlCommandQueue,
    memory_pool: Arc<Mutex<MetalMemoryPool>>,
    #[cfg(feature = "metal")]
    pipelines: Arc<Mutex<HashMap<String, MtlPipeline>>>,
    #[cfg(feature = "metal")]
    library: Option<MtlLibrary>,
}

impl MetalDevice {
    /// Create a new Metal device
    pub fn new(device_id: u32) -> GpuResult<Self> {
        #[cfg(feature = "metal")]
        {
            let devices = MtlDevice::all();
            let mtl_device = if device_id == 0 {
                MtlDevice::system_default().ok_or_else(|| {
                    GpuError::BackendNotAvailable("No Metal device found".to_string())
                })?
            } else {
                devices
                    .get(device_id as usize)
                    .cloned()
                    .ok_or_else(|| GpuError::DeviceNotFound(device_id))?
            };

            let command_queue = mtl_device.new_command_queue();
            let memory_pool = Arc::new(Mutex::new(MetalMemoryPool::new()));

            Ok(MetalDevice {
                device_id,
                mtl_device,
                command_queue,
                memory_pool,
                pipelines: Arc::new(Mutex::new(HashMap::new())),
                library: None,
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(GpuError::BackendNotAvailable(
                "Metal feature not enabled".to_string(),
            ))
        }
    }

    /// Create a new command queue
    #[cfg(feature = "metal")]
    pub fn create_command_queue(&self) -> GpuResult<Arc<MetalCommandQueue>> {
        Ok(Arc::new(MetalCommandQueue {
            queue: self.mtl_device.new_command_queue(),
        }))
    }

    #[cfg(not(feature = "metal"))]
    pub fn create_command_queue(&self) -> GpuResult<Arc<MetalCommandQueue>> {
        Err(GpuError::BackendNotAvailable(
            "Metal feature not enabled".to_string(),
        ))
    }

    /// Load Metal shader library from source
    #[cfg(feature = "metal")]
    pub fn load_library(&mut self, source: &str) -> GpuResult<()> {
        let library = self
            .mtl_device
            .new_library_with_source(source, &metal::CompileOptions::new())
            .map_err(|e| {
                GpuError::BackendError(format!("Failed to compile Metal library: {}", e))
            })?;
        self.library = Some(library);
        Ok(())
    }

    /// Get or create a compute pipeline for the specified function
    #[cfg(feature = "metal")]
    fn get_pipeline(&self, function_name: &str) -> GpuResult<MtlPipeline> {
        let mut cache = self.pipelines.lock().expect("GPU mutex poisoned");

        if let Some(pipeline) = cache.get(function_name) {
            return Ok(pipeline.clone());
        }

        let library = self
            .library
            .as_ref()
            .ok_or_else(|| GpuError::BackendError("No shader library loaded".to_string()))?;

        let function = library.get_function(function_name, None).map_err(|e| {
            GpuError::BackendError(format!("Function {} not found: {}", function_name, e))
        })?;

        let pipeline = self
            .mtl_device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| GpuError::BackendError(format!("Failed to create pipeline: {}", e)))?;

        cache.insert(function_name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Launch spike propagation kernel
    ///
    /// Uses Metal compute shader for parallel neuron updates
    #[cfg(feature = "metal")]
    pub fn launch_spike_propagation(
        &self,
        voltages: &Arc<dyn GpuBuffer>,
        spikes: &Arc<dyn GpuBuffer>,
        weights: &Arc<dyn GpuBuffer>,
        num_neurons: usize,
        dt: f32,
        threshold: f32,
        command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        let pipeline = self.get_pipeline("spike_propagation")?;

        if let Some(encoder) = command_buffer.buffer.new_compute_command_encoder() {
            encoder.set_compute_pipeline_state(&pipeline);

            // Set buffers
            if let Some(v_buf) = voltages.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = v_buf.mtl_buffer {
                    encoder.set_buffer(0, Some(buf), 0);
                }
            }
            if let Some(s_buf) = spikes.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = s_buf.mtl_buffer {
                    encoder.set_buffer(1, Some(buf), 0);
                }
            }
            if let Some(w_buf) = weights.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = w_buf.mtl_buffer {
                    encoder.set_buffer(2, Some(buf), 0);
                }
            }

            // Set constants
            encoder.set_bytes(
                3,
                std::mem::size_of::<f32>() as u64,
                &dt as *const f32 as *const _,
            );
            encoder.set_bytes(
                4,
                std::mem::size_of::<f32>() as u64,
                &threshold as *const f32 as *const _,
            );

            let threads_per_group = 256;
            let num_groups = (num_neurons + threads_per_group - 1) / threads_per_group;

            let thread_group_size = metal::MTLSize::new(threads_per_group as u64, 1, 1);
            let grid_size = metal::MTLSize::new(num_groups as u64, 1, 1);

            encoder.dispatch_thread_groups(grid_size, thread_group_size);
            encoder.end_encoding();
        }

        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    pub fn launch_spike_propagation(
        &self,
        _voltages: &Arc<dyn GpuBuffer>,
        _spikes: &Arc<dyn GpuBuffer>,
        _weights: &Arc<dyn GpuBuffer>,
        _num_neurons: usize,
        _dt: f32,
        _threshold: f32,
        _command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }

    /// Launch STDP weight update kernel
    #[cfg(feature = "metal")]
    pub fn launch_stdp_update(
        &self,
        weights: &Arc<dyn GpuBuffer>,
        pre_spikes: &Arc<dyn GpuBuffer>,
        post_spikes: &Arc<dyn GpuBuffer>,
        traces: &Arc<dyn GpuBuffer>,
        num_synapses: usize,
        learning_rate: f32,
        tau_plus: f32,
        tau_minus: f32,
        command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        let pipeline = self.get_pipeline("stdp_update")?;

        if let Some(encoder) = command_buffer.buffer.new_compute_command_encoder() {
            encoder.set_compute_pipeline_state(&pipeline);

            // Set buffers
            if let Some(w_buf) = weights.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = w_buf.mtl_buffer {
                    encoder.set_buffer(0, Some(buf), 0);
                }
            }
            if let Some(pre_buf) = pre_spikes.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = pre_buf.mtl_buffer {
                    encoder.set_buffer(1, Some(buf), 0);
                }
            }
            if let Some(post_buf) = post_spikes.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = post_buf.mtl_buffer {
                    encoder.set_buffer(2, Some(buf), 0);
                }
            }
            if let Some(t_buf) = traces.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = t_buf.mtl_buffer {
                    encoder.set_buffer(3, Some(buf), 0);
                }
            }

            // Set constants
            encoder.set_bytes(
                4,
                std::mem::size_of::<f32>() as u64,
                &learning_rate as *const f32 as *const _,
            );
            encoder.set_bytes(
                5,
                std::mem::size_of::<f32>() as u64,
                &tau_plus as *const f32 as *const _,
            );
            encoder.set_bytes(
                6,
                std::mem::size_of::<f32>() as u64,
                &tau_minus as *const f32 as *const _,
            );

            let threads_per_group = 256;
            let num_groups = (num_synapses + threads_per_group - 1) / threads_per_group;

            let thread_group_size = metal::MTLSize::new(threads_per_group as u64, 1, 1);
            let grid_size = metal::MTLSize::new(num_groups as u64, 1, 1);

            encoder.dispatch_thread_groups(grid_size, thread_group_size);
            encoder.end_encoding();
        }

        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    pub fn launch_stdp_update(
        &self,
        _weights: &Arc<dyn GpuBuffer>,
        _pre_spikes: &Arc<dyn GpuBuffer>,
        _post_spikes: &Arc<dyn GpuBuffer>,
        _traces: &Arc<dyn GpuBuffer>,
        _num_synapses: usize,
        _learning_rate: f32,
        _tau_plus: f32,
        _tau_minus: f32,
        _command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }

    /// Launch sparse matrix-vector multiplication
    #[cfg(feature = "metal")]
    pub fn launch_sparse_matmul(
        &self,
        output: &Arc<dyn GpuBuffer>,
        matrix_values: &Arc<dyn GpuBuffer>,
        matrix_indices: &Arc<dyn GpuBuffer>,
        matrix_indptr: &Arc<dyn GpuBuffer>,
        vector: &Arc<dyn GpuBuffer>,
        num_rows: usize,
        command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        let pipeline = self.get_pipeline("sparse_matmul_csr")?;

        if let Some(encoder) = command_buffer.buffer.new_compute_command_encoder() {
            encoder.set_compute_pipeline_state(&pipeline);

            // Set buffers
            let buffers = [output, matrix_values, matrix_indices, matrix_indptr, vector];
            for (i, buf) in buffers.iter().enumerate() {
                if let Some(m_buf) = buf.as_any().downcast_ref::<MetalBuffer>() {
                    if let Some(ref mtl_buf) = m_buf.mtl_buffer {
                        encoder.set_buffer(i as u64, Some(mtl_buf), 0);
                    }
                }
            }

            let threads_per_group = 256;
            let num_groups = (num_rows + threads_per_group - 1) / threads_per_group;

            let thread_group_size = metal::MTLSize::new(threads_per_group as u64, 1, 1);
            let grid_size = metal::MTLSize::new(num_groups as u64, 1, 1);

            encoder.dispatch_thread_groups(grid_size, thread_group_size);
            encoder.end_encoding();
        }

        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    pub fn launch_sparse_matmul(
        &self,
        _output: &Arc<dyn GpuBuffer>,
        _matrix_values: &Arc<dyn GpuBuffer>,
        _matrix_indices: &Arc<dyn GpuBuffer>,
        _matrix_indptr: &Arc<dyn GpuBuffer>,
        _vector: &Arc<dyn GpuBuffer>,
        _num_rows: usize,
        _command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }

    /// Launch reduction kernel
    #[cfg(feature = "metal")]
    pub fn launch_reduction(
        &self,
        input: &Arc<dyn GpuBuffer>,
        output: &Arc<dyn GpuBuffer>,
        num_elements: usize,
        reduction_op: ReductionOp,
        command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        let function_name = match reduction_op {
            ReductionOp::Sum => "reduce_sum",
            ReductionOp::Max => "reduce_max",
            ReductionOp::Min => "reduce_min",
        };

        let pipeline = self.get_pipeline(function_name)?;

        if let Some(encoder) = command_buffer.buffer.new_compute_command_encoder() {
            encoder.set_compute_pipeline_state(&pipeline);

            if let Some(i_buf) = input.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = i_buf.mtl_buffer {
                    encoder.set_buffer(0, Some(buf), 0);
                }
            }
            if let Some(o_buf) = output.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref buf) = o_buf.mtl_buffer {
                    encoder.set_buffer(1, Some(buf), 0);
                }
            }

            let n = num_elements as u32;
            encoder.set_bytes(
                2,
                std::mem::size_of::<u32>() as u64,
                &n as *const u32 as *const _,
            );

            let threads_per_group = 256;
            let num_groups = (num_elements + threads_per_group - 1) / threads_per_group;

            let thread_group_size = metal::MTLSize::new(threads_per_group as u64, 1, 1);
            let grid_size = metal::MTLSize::new(num_groups as u64, 1, 1);

            encoder.dispatch_thread_groups(grid_size, thread_group_size);
            encoder.end_encoding();
        }

        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    pub fn launch_reduction(
        &self,
        _input: &Arc<dyn GpuBuffer>,
        _output: &Arc<dyn GpuBuffer>,
        _num_elements: usize,
        _reduction_op: ReductionOp,
        _command_buffer: &MetalCommandBuffer,
    ) -> GpuResult<()> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }
}

impl GpuDevice for MetalDevice {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn backend(&self) -> Backend {
        Backend::Metal
    }

    fn allocate(&self, size_bytes: usize) -> GpuResult<Arc<dyn GpuBuffer>> {
        #[cfg(feature = "metal")]
        {
            let mut pool = self.memory_pool.lock().expect("GPU mutex poisoned");
            pool.allocate(size_bytes, &self.mtl_device)
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(GpuError::BackendNotAvailable(
                "Metal not available".to_string(),
            ))
        }
    }

    fn copy_to_device(&self, src: &[f32], dst: &Arc<dyn GpuBuffer>) -> GpuResult<()> {
        let required_size = src.len() * std::mem::size_of::<f32>();
        if dst.size() < required_size {
            return Err(GpuError::InvalidBufferSize {
                expected: required_size,
                actual: dst.size(),
            });
        }

        #[cfg(feature = "metal")]
        {
            if let Some(m_buf) = dst.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref mtl_buf) = m_buf.mtl_buffer {
                    let contents = mtl_buf.contents() as *mut f32;
                    unsafe {
                        std::ptr::copy_nonoverlapping(src.as_ptr(), contents, src.len());
                    }
                }
            }
        }
        Ok(())
    }

    fn copy_to_host(&self, src: &Arc<dyn GpuBuffer>, dst: &mut [f32]) -> GpuResult<()> {
        let required_size = dst.len() * std::mem::size_of::<f32>();
        if src.size() < required_size {
            return Err(GpuError::InvalidBufferSize {
                expected: required_size,
                actual: src.size(),
            });
        }

        #[cfg(feature = "metal")]
        {
            if let Some(m_buf) = src.as_any().downcast_ref::<MetalBuffer>() {
                if let Some(ref mtl_buf) = m_buf.mtl_buffer {
                    let contents = mtl_buf.contents() as *const f32;
                    unsafe {
                        std::ptr::copy_nonoverlapping(contents, dst.as_mut_ptr(), dst.len());
                    }
                }
            }
        }
        Ok(())
    }

    fn copy_device_to_device(
        &self,
        src: &Arc<dyn GpuBuffer>,
        dst: &Arc<dyn GpuBuffer>,
    ) -> GpuResult<()> {
        if src.size() != dst.size() {
            return Err(GpuError::InvalidBufferSize {
                expected: dst.size(),
                actual: src.size(),
            });
        }

        #[cfg(feature = "metal")]
        {
            if let (Some(s_buf), Some(d_buf)) = (
                src.as_any().downcast_ref::<MetalBuffer>(),
                dst.as_any().downcast_ref::<MetalBuffer>(),
            ) {
                if let (Some(ref src_mtl), Some(ref dst_mtl)) =
                    (&s_buf.mtl_buffer, &d_buf.mtl_buffer)
                {
                    // Use blit encoder for device-to-device copy
                    let cmd_buf = self.command_queue.new_command_buffer();
                    if let Some(encoder) = cmd_buf.new_blit_command_encoder() {
                        encoder.copy_from_buffer(src_mtl, 0, dst_mtl, 0, src.size() as u64);
                        encoder.end_encoding();
                    }
                    cmd_buf.commit();
                    cmd_buf.wait_until_completed();
                }
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> GpuResult<()> {
        #[cfg(feature = "metal")]
        {
            // Create a dummy command buffer and wait for it
            let cmd_buf = self.command_queue.new_command_buffer();
            cmd_buf.commit();
            cmd_buf.wait_until_completed();
        }
        Ok(())
    }

    fn memory_info(&self) -> GpuResult<(usize, usize)> {
        #[cfg(feature = "metal")]
        {
            let total = self.mtl_device.recommended_max_working_set_size() as usize;
            let current = self.mtl_device.current_allocated_size() as usize;
            Ok((total, total - current))
        }
        #[cfg(not(feature = "metal"))]
        {
            Ok((0, 0))
        }
    }

    fn name(&self) -> String {
        #[cfg(feature = "metal")]
        {
            self.mtl_device.name().to_string()
        }
        #[cfg(not(feature = "metal"))]
        {
            format!("Metal Device {}", self.device_id)
        }
    }

    fn compute_capability(&self) -> (u32, u32) {
        // Metal family versions (Apple GPU families)
        #[cfg(feature = "metal")]
        {
            // Return Metal family version (approximate)
            if self.mtl_device.supports_family(metal::MTLGPUFamily::Apple7) {
                (7, 0)
            } else if self.mtl_device.supports_family(metal::MTLGPUFamily::Apple6) {
                (6, 0)
            } else if self.mtl_device.supports_family(metal::MTLGPUFamily::Apple5) {
                (5, 0)
            } else {
                (4, 0)
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            (0, 0)
        }
    }
}

/// Metal command queue for command submission
pub struct MetalCommandQueue {
    #[cfg(feature = "metal")]
    queue: MtlCommandQueue,
}

impl MetalCommandQueue {
    /// Create a new command buffer
    #[cfg(feature = "metal")]
    pub fn create_command_buffer(&self) -> GpuResult<MetalCommandBuffer> {
        Ok(MetalCommandBuffer {
            buffer: self.queue.new_command_buffer().to_owned(),
        })
    }

    #[cfg(not(feature = "metal"))]
    pub fn create_command_buffer(&self) -> GpuResult<MetalCommandBuffer> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }
}

/// Metal command buffer for batching GPU operations
pub struct MetalCommandBuffer {
    #[cfg(feature = "metal")]
    buffer: MtlCommandBuffer,
}

impl MetalCommandBuffer {
    /// Commit the command buffer for execution
    #[cfg(feature = "metal")]
    pub fn commit(&self) -> GpuResult<()> {
        self.buffer.commit();
        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    pub fn commit(&self) -> GpuResult<()> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }

    /// Wait for command buffer to complete
    #[cfg(feature = "metal")]
    pub fn wait_until_completed(&self) -> GpuResult<()> {
        self.buffer.wait_until_completed();
        Ok(())
    }

    #[cfg(not(feature = "metal"))]
    pub fn wait_until_completed(&self) -> GpuResult<()> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }
}

/// Metal buffer backed by MTLBuffer
pub struct MetalBuffer {
    size: usize,
    device_id: u32,
    #[cfg(feature = "metal")]
    pub(crate) mtl_buffer: Option<metal::Buffer>,
}

impl std::fmt::Debug for MetalBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalBuffer")
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl MetalBuffer {
    #[cfg(feature = "metal")]
    fn new(device: &MtlDevice, size: usize, device_id: u32) -> GpuResult<Self> {
        // Use shared storage mode for unified memory on Apple Silicon
        let mtl_buffer = device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);

        Ok(MetalBuffer {
            size,
            device_id,
            mtl_buffer: Some(mtl_buffer),
        })
    }

    #[cfg(not(feature = "metal"))]
    fn new(_size: usize, _device_id: u32) -> GpuResult<Self> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }
}

impl GpuBuffer for MetalBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *mut u8 {
        #[cfg(feature = "metal")]
        {
            if let Some(ref buf) = self.mtl_buffer {
                buf.contents() as *mut u8
            } else {
                std::ptr::null_mut()
            }
        }
        #[cfg(not(feature = "metal"))]
        {
            std::ptr::null_mut()
        }
    }

    fn is_valid(&self) -> bool {
        #[cfg(feature = "metal")]
        {
            self.mtl_buffer.is_some()
        }
        #[cfg(not(feature = "metal"))]
        {
            false
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

unsafe impl Send for MetalBuffer {}
unsafe impl Sync for MetalBuffer {}

/// Memory pool for efficient buffer reuse
struct MetalMemoryPool {
    free_buffers: HashMap<usize, Vec<Arc<MetalBuffer>>>,
    allocated_bytes: usize,
}

impl MetalMemoryPool {
    fn new() -> Self {
        MetalMemoryPool {
            free_buffers: HashMap::new(),
            allocated_bytes: 0,
        }
    }

    #[cfg(feature = "metal")]
    fn allocate(&mut self, size: usize, device: &MtlDevice) -> GpuResult<Arc<dyn GpuBuffer>> {
        // Try to reuse
        if let Some(buffers) = self.free_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer as Arc<dyn GpuBuffer>);
            }
        }

        // Allocate new
        let buffer = Arc::new(MetalBuffer::new(device, size, 0)?);
        self.allocated_bytes += size;

        Ok(buffer as Arc<dyn GpuBuffer>)
    }

    #[cfg(not(feature = "metal"))]
    fn allocate(&mut self, _size: usize, _device_id: u32) -> GpuResult<Arc<dyn GpuBuffer>> {
        Err(GpuError::BackendNotAvailable(
            "Metal not available".to_string(),
        ))
    }

    #[allow(dead_code)]
    fn total_allocated(&self) -> usize {
        self.allocated_bytes
    }
}

/// Reduction operation types
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
}

/// Metal shader source for SNN kernels
#[cfg(feature = "metal")]
pub const METAL_SNN_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Spike propagation kernel
kernel void spike_propagation(
    device float* voltages [[buffer(0)]],
    device bool* spikes [[buffer(1)]],
    device const float* weights [[buffer(2)]],
    constant float& dt [[buffer(3)]],
    constant float& threshold [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float v = voltages[id];

    // Update voltage based on input weights
    // Simplified LIF model
    float tau = 20.0; // membrane time constant in ms
    v = v * exp(-dt / tau);

    // Check for spike
    if (v >= threshold) {
        spikes[id] = true;
        v = 0.0; // reset
    } else {
        spikes[id] = false;
    }

    voltages[id] = v;
}

// STDP weight update kernel
kernel void stdp_update(
    device float* weights [[buffer(0)]],
    device const bool* pre_spikes [[buffer(1)]],
    device const bool* post_spikes [[buffer(2)]],
    device float* traces [[buffer(3)]],
    constant float& learning_rate [[buffer(4)]],
    constant float& tau_plus [[buffer(5)]],
    constant float& tau_minus [[buffer(6)]],
    uint id [[thread_position_in_grid]]
) {
    float trace = traces[id];
    float weight = weights[id];

    // Update trace
    trace *= exp(-1.0 / tau_plus);

    if (pre_spikes[id]) {
        trace += 1.0;
    }

    // STDP update
    if (post_spikes[id] && trace > 0.0) {
        // LTP (potentiation)
        weight += learning_rate * trace;
    }

    // Clamp weight
    weights[id] = clamp(weight, 0.0f, 1.0f);
    traces[id] = trace;
}

// Sparse matrix-vector multiplication (CSR format)
kernel void sparse_matmul_csr(
    device float* output [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const uint* indices [[buffer(2)]],
    device const uint* indptr [[buffer(3)]],
    device const float* vector [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    float sum = 0.0;
    uint start = indptr[row];
    uint end = indptr[row + 1];

    for (uint i = start; i < end; i++) {
        sum += values[i] * vector[indices[i]];
    }

    output[row] = sum;
}

// Reduction kernels
kernel void reduce_sum(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float sum = 0.0;
    for (uint i = id; i < n; i += 256 * 256) {
        sum += input[i];
    }

    shared[local_id] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce within threadgroup
    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared[local_id] += shared[local_id + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        output[group_id] = shared[0];
    }
}

kernel void reduce_max(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float val = -INFINITY;
    for (uint i = id; i < n; i += 256 * 256) {
        val = max(val, input[i]);
    }

    shared[local_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared[local_id] = max(shared[local_id], shared[local_id + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        output[group_id] = shared[0];
    }
}

kernel void reduce_min(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint id [[thread_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint group_id [[threadgroup_position_in_grid]],
    threadgroup float* shared [[threadgroup(0)]]
) {
    float val = INFINITY;
    for (uint i = id; i < n; i += 256 * 256) {
        val = min(val, input[i]);
    }

    shared[local_id] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint s = 128; s > 0; s >>= 1) {
        if (local_id < s) {
            shared[local_id] = min(shared[local_id], shared[local_id + s]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (local_id == 0) {
        output[group_id] = shared[0];
    }
}
"#;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_metal_available() {
        // Should not crash
        let _ = is_metal_available();
    }

    #[test]
    fn test_list_metal_devices() {
        if is_metal_available() {
            let devices = list_metal_devices();
            assert!(devices.is_ok());
        }
    }

    #[test]
    fn test_reduction_op() {
        let op = ReductionOp::Max;
        assert!(matches!(op, ReductionOp::Max));
    }
}
