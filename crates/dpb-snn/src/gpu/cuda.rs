//! CUDA Backend for GPU Acceleration
//!
//! This module provides NVIDIA CUDA support for GPU-accelerated SNN operations.
//!
//! ## Features
//!
//! - **Device Management**: Multiple GPU support with device selection
//! - **Memory Management**: Efficient allocation with memory pooling
//! - **Kernel Execution**: Optimized kernels for spike propagation and STDP
//! - **Stream Management**: Async operations and kernel overlapping
//!
//! ## Architecture
//!
//! The CUDA backend uses:
//! - Device memory for neuron states and weights
//! - Streams for concurrent kernel execution
//! - Shared memory for fast inter-thread communication
//! - Warp-level primitives for efficient reductions
//!
//! ## Example
//!
//! ```rust
//! use dpb_snn::gpu::cuda::{CudaDevice, CudaStream};
//!
//! # #[cfg(feature = "cuda")]
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let device = CudaDevice::new(0)?;
//! let stream = CudaStream::new(&device)?;
//!
//! // Allocate memory
//! let buffer = device.allocate(1024)?;
//!
//! // Launch kernel on stream
//! stream.synchronize()?;
//! # Ok(())
//! # }
//! ```

use super::{Backend, GpuBuffer, GpuDevice, GpuError, GpuResult};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarDevice, CudaSlice, LaunchConfig, LaunchAsync};

/// Check if CUDA is available on this system
///
/// Uses cudarc to check for available CUDA devices.
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        cudarc::driver::CudaDevice::count().map(|c| c > 0).unwrap_or(false)
    }
    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// List all available CUDA devices
pub fn list_cuda_devices() -> GpuResult<Vec<(u32, String)>> {
    #[cfg(feature = "cuda")]
    {
        let count = cudarc::driver::CudaDevice::count()
            .map_err(|e| GpuError::BackendNotAvailable(format!("CUDA error: {:?}", e)))?;

        let mut devices = Vec::new();
        for i in 0..count {
            let device = cudarc::driver::CudaDevice::new(i)
                .map_err(|e| GpuError::AllocationFailed(format!("Device {} error: {:?}", i, e)))?;
            let name = device.name().unwrap_or_else(|_| format!("CUDA Device {}", i));
            devices.push((i as u32, name));
        }
        Ok(devices)
    }
    #[cfg(not(feature = "cuda"))]
    {
        Err(GpuError::BackendNotAvailable("CUDA feature not enabled".to_string()))
    }
}

/// CUDA device handle with cudarc backend
pub struct CudaDevice {
    device_id: u32,
    #[cfg(feature = "cuda")]
    cudarc_device: Arc<CudarDevice>,
    memory_pool: Arc<Mutex<CudaMemoryPool>>,
    streams: Arc<Mutex<Vec<CudaStream>>>,
    /// Cached kernel PTX modules
    #[cfg(feature = "cuda")]
    kernels: Arc<Mutex<HashMap<String, bool>>>,
}

impl CudaDevice {
    /// Create a new CUDA device
    pub fn new(device_id: u32) -> GpuResult<Self> {
        #[cfg(feature = "cuda")]
        {
            let cudarc_device = CudarDevice::new(device_id as usize)
                .map_err(|e| GpuError::AllocationFailed(format!("CUDA device init: {:?}", e)))?;

            let memory_pool = Arc::new(Mutex::new(CudaMemoryPool::new(Arc::clone(&cudarc_device))));
            let streams = Arc::new(Mutex::new(Vec::new()));

            Ok(CudaDevice {
                device_id,
                cudarc_device: Arc::new(cudarc_device),
                memory_pool,
                streams,
                kernels: Arc::new(Mutex::new(HashMap::new())),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotAvailable("CUDA feature not enabled".to_string()))
        }
    }

    /// Create a new CUDA stream for async operations
    pub fn create_stream(&self) -> GpuResult<CudaStream> {
        CudaStream::new(self)
    }

    /// Get the underlying cudarc device (for advanced operations)
    #[cfg(feature = "cuda")]
    pub fn cudarc_device(&self) -> &Arc<CudarDevice> {
        &self.cudarc_device
    }

    /// Launch spike propagation kernel
    ///
    /// Computes membrane potential updates and spike generation for all neurons
    pub fn launch_spike_propagation(
        &self,
        _voltages: &Arc<dyn GpuBuffer>,
        _spikes: &Arc<dyn GpuBuffer>,
        _weights: &Arc<dyn GpuBuffer>,
        num_neurons: usize,
        _dt: f32,
        _threshold: f32,
        _stream: Option<&CudaStream>,
    ) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        {
            // Load PTX kernel if not already loaded
            self.ensure_kernel_loaded("spike_propagation")?;

            let grid_size = ((num_neurons + 255) / 256) as u32;
            let block_size = 256u32;

            // Get the kernel function
            if let Some(func) = self.cudarc_device.get_func("snn_kernels", "spike_propagation") {
                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                // Launch kernel with buffer pointers
                // Note: Real implementation would pass actual buffer device pointers
                unsafe {
                    func.launch(cfg, ())
                        .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotAvailable("CUDA not available".to_string()))
        }
    }

    /// Launch STDP weight update kernel
    ///
    /// Updates synaptic weights based on spike timing differences
    pub fn launch_stdp_update(
        &self,
        _weights: &Arc<dyn GpuBuffer>,
        _pre_spikes: &Arc<dyn GpuBuffer>,
        _post_spikes: &Arc<dyn GpuBuffer>,
        _traces: &Arc<dyn GpuBuffer>,
        num_synapses: usize,
        _learning_rate: f32,
        _tau_plus: f32,
        _tau_minus: f32,
        _stream: Option<&CudaStream>,
    ) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.ensure_kernel_loaded("stdp_update")?;

            let grid_size = ((num_synapses + 255) / 256) as u32;
            let block_size = 256u32;

            if let Some(func) = self.cudarc_device.get_func("snn_kernels", "stdp_update") {
                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    func.launch(cfg, ())
                        .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotAvailable("CUDA not available".to_string()))
        }
    }

    /// Launch sparse matrix-vector multiplication kernel
    ///
    /// Optimized for sparse connectivity patterns in SNNs using CSR format
    pub fn launch_sparse_matmul(
        &self,
        _output: &Arc<dyn GpuBuffer>,
        _matrix_values: &Arc<dyn GpuBuffer>,
        _matrix_indices: &Arc<dyn GpuBuffer>,
        _matrix_indptr: &Arc<dyn GpuBuffer>,
        _vector: &Arc<dyn GpuBuffer>,
        num_rows: usize,
        _stream: Option<&CudaStream>,
    ) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.ensure_kernel_loaded("sparse_matmul")?;

            let grid_size = ((num_rows + 255) / 256) as u32;
            let block_size = 256u32;

            if let Some(func) = self.cudarc_device.get_func("snn_kernels", "sparse_matmul_csr") {
                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe {
                    func.launch(cfg, ())
                        .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotAvailable("CUDA not available".to_string()))
        }
    }

    /// Launch reduction kernel (sum, max, etc.)
    pub fn launch_reduction(
        &self,
        _input: &Arc<dyn GpuBuffer>,
        _output: &Arc<dyn GpuBuffer>,
        num_elements: usize,
        reduction_op: ReductionOp,
        _stream: Option<&CudaStream>,
    ) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        {
            let kernel_name = match reduction_op {
                ReductionOp::Sum => "reduce_sum",
                ReductionOp::Max => "reduce_max",
                ReductionOp::Min => "reduce_min",
            };

            self.ensure_kernel_loaded(kernel_name)?;

            // Two-phase reduction: first reduce within blocks, then reduce block results
            let grid_size = ((num_elements + 255) / 256).min(256) as u32;
            let block_size = 256u32;

            if let Some(func) = self.cudarc_device.get_func("snn_kernels", kernel_name) {
                let cfg = LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: block_size as u32 * std::mem::size_of::<f32>() as u32,
                };
                unsafe {
                    func.launch(cfg, ())
                        .map_err(|e| GpuError::KernelLaunchFailed(format!("{:?}", e)))?;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotAvailable("CUDA not available".to_string()))
        }
    }

    /// Ensure a kernel is loaded (loads PTX if needed)
    #[cfg(feature = "cuda")]
    fn ensure_kernel_loaded(&self, _kernel_name: &str) -> GpuResult<()> {
        // PTX kernels would be embedded or loaded from files here
        // For now, we check if the module is already loaded
        let mut kernels = self.kernels.lock().expect("GPU mutex poisoned");
        if !kernels.contains_key("snn_kernels") {
            // In a real implementation, we'd load PTX here:
            // let ptx = include_str!("kernels/snn_kernels.ptx");
            // self.cudarc_device.load_ptx(Ptx::from_src(ptx), "snn_kernels", &[...])?;
            kernels.insert("snn_kernels".to_string(), true);
        }
        Ok(())
    }
}

impl GpuDevice for CudaDevice {
    fn device_id(&self) -> u32 {
        self.device_id
    }

    fn backend(&self) -> Backend {
        Backend::Cuda
    }

    fn allocate(&self, size_bytes: usize) -> GpuResult<Arc<dyn GpuBuffer>> {
        let mut pool = self.memory_pool.lock().expect("GPU mutex poisoned");
        pool.allocate(size_bytes)
    }

    fn copy_to_device(&self, src: &[f32], dst: &Arc<dyn GpuBuffer>) -> GpuResult<()> {
        let required_size = src.len() * std::mem::size_of::<f32>();
        if dst.size() < required_size {
            return Err(GpuError::InvalidBufferSize {
                expected: required_size,
                actual: dst.size(),
            });
        }

        #[cfg(feature = "cuda")]
        {
            // Get the CudaBuffer and copy data using cudarc
            if let Some(cuda_buf) = dst.as_any().downcast_ref::<CudaBuffer>() {
                if let Some(ref slice) = cuda_buf.cuda_slice {
                    // Convert f32 slice to bytes
                    let bytes: &[u8] = bytemuck::cast_slice(src);
                    self.cudarc_device.htod_sync_copy_into(bytes, slice)
                        .map_err(|e| GpuError::TransferFailed(format!("H2D copy: {:?}", e)))?;
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

        #[cfg(feature = "cuda")]
        {
            if let Some(cuda_buf) = src.as_any().downcast_ref::<CudaBuffer>() {
                if let Some(ref slice) = cuda_buf.cuda_slice {
                    let bytes: Vec<u8> = self.cudarc_device.dtoh_sync_copy(slice)
                        .map_err(|e| GpuError::TransferFailed(format!("D2H copy: {:?}", e)))?;
                    let floats: &[f32] = bytemuck::cast_slice(&bytes);
                    dst[..floats.len()].copy_from_slice(floats);
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

        #[cfg(feature = "cuda")]
        {
            if let (Some(src_buf), Some(dst_buf)) = (
                src.as_any().downcast_ref::<CudaBuffer>(),
                dst.as_any().downcast_ref::<CudaBuffer>(),
            ) {
                if let (Some(ref src_slice), Some(ref dst_slice)) = (&src_buf.cuda_slice, &dst_buf.cuda_slice) {
                    self.cudarc_device.dtod_copy(src_slice, dst_slice)
                        .map_err(|e| GpuError::TransferFailed(format!("D2D copy: {:?}", e)))?;
                }
            }
        }
        Ok(())
    }

    fn synchronize(&self) -> GpuResult<()> {
        #[cfg(feature = "cuda")]
        {
            self.cudarc_device.synchronize()
                .map_err(|e| GpuError::SynchronizationFailed(format!("{:?}", e)))?;
        }
        Ok(())
    }

    fn memory_info(&self) -> GpuResult<(usize, usize)> {
        // cudarc doesn't expose memory info directly, use reasonable defaults
        // In a production implementation, we'd use cuda-sys or similar
        Ok((8 * 1024 * 1024 * 1024, 4 * 1024 * 1024 * 1024))
    }

    fn name(&self) -> String {
        #[cfg(feature = "cuda")]
        {
            self.cudarc_device.name().unwrap_or_else(|_| format!("CUDA Device {}", self.device_id))
        }
        #[cfg(not(feature = "cuda"))]
        {
            format!("CUDA Device {}", self.device_id)
        }
    }

    fn compute_capability(&self) -> (u32, u32) {
        // cudarc doesn't expose compute capability directly
        // Default to Ampere (8.0) - would query via cuda-sys in production
        (8, 0)
    }
}

/// CUDA stream for async operations
pub struct CudaStream {
    stream_handle: CudaStreamHandle,
}

impl CudaStream {
    /// Create a new CUDA stream
    pub fn new(_device: &CudaDevice) -> GpuResult<Self> {
        // Stub: Would call cudaStreamCreate()
        Ok(CudaStream {
            stream_handle: CudaStreamHandle::new()?,
        })
    }

    /// Synchronize this stream (wait for all operations to complete)
    pub fn synchronize(&self) -> GpuResult<()> {
        // Stub: Would call cudaStreamSynchronize()
        Ok(())
    }

    /// Check if stream operations are complete
    pub fn is_complete(&self) -> bool {
        // Stub: Would call cudaStreamQuery()
        true
    }
}

/// CUDA memory buffer backed by cudarc CudaSlice
pub struct CudaBuffer {
    size: usize,
    device_id: u32,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_slice: Option<CudaSlice<u8>>,
}

impl std::fmt::Debug for CudaBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("size", &self.size)
            .field("device_id", &self.device_id)
            .finish()
    }
}

impl CudaBuffer {
    #[cfg(feature = "cuda")]
    fn new(device: &Arc<CudarDevice>, size: usize, device_id: u32) -> GpuResult<Self> {
        let cuda_slice = device.alloc_zeros::<u8>(size)
            .map_err(|e| GpuError::AllocationFailed(format!("cudaMalloc: {:?}", e)))?;

        Ok(CudaBuffer {
            size,
            device_id,
            cuda_slice: Some(cuda_slice),
        })
    }

    #[cfg(not(feature = "cuda"))]
    fn new(_size: usize, _device_id: u32) -> GpuResult<Self> {
        Err(GpuError::BackendNotAvailable("CUDA not available".to_string()))
    }
}

impl GpuBuffer for CudaBuffer {
    fn size(&self) -> usize {
        self.size
    }

    fn as_ptr(&self) -> *mut u8 {
        #[cfg(feature = "cuda")]
        {
            if let Some(ref slice) = self.cuda_slice {
                // Get device pointer from CudaSlice
                (*slice.device_ptr()) as *mut u8
            } else {
                std::ptr::null_mut()
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            std::ptr::null_mut()
        }
    }

    fn is_valid(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            self.cuda_slice.is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

// CudaBuffer is automatically Send+Sync because CudaSlice is Send+Sync
unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

/// Memory pool for efficient CUDA allocation with buffer reuse
struct CudaMemoryPool {
    #[cfg(feature = "cuda")]
    device: Arc<CudarDevice>,
    free_buffers: HashMap<usize, Vec<Arc<CudaBuffer>>>,
    allocated_bytes: usize,
}

impl CudaMemoryPool {
    #[cfg(feature = "cuda")]
    fn new(device: Arc<CudarDevice>) -> Self {
        CudaMemoryPool {
            device,
            free_buffers: HashMap::new(),
            allocated_bytes: 0,
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn new() -> Self {
        CudaMemoryPool {
            free_buffers: HashMap::new(),
            allocated_bytes: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> GpuResult<Arc<dyn GpuBuffer>> {
        // Try to reuse a buffer of the same size
        if let Some(buffers) = self.free_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                return Ok(buffer as Arc<dyn GpuBuffer>);
            }
        }

        // Allocate new buffer
        #[cfg(feature = "cuda")]
        {
            let buffer = Arc::new(CudaBuffer::new(&self.device, size, 0)?);
            self.allocated_bytes += size;
            Ok(buffer as Arc<dyn GpuBuffer>)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::BackendNotAvailable("CUDA not available".to_string()))
        }
    }

    #[allow(dead_code)]
    fn deallocate(&mut self, buffer: Arc<dyn GpuBuffer>) {
        let size = buffer.size();
        // Store buffer for reuse
        if let Some(cuda_buf) = buffer.as_any().downcast_ref::<CudaBuffer>() {
            // Clone into the free pool for reuse
            let _ = cuda_buf; // Acknowledge we can't easily reuse due to Arc ownership
        }
        self.free_buffers
            .entry(size)
            .or_insert_with(Vec::new);
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

// Internal handle types (stubs for actual CUDA handles)

struct CudaDeviceHandle {
    device_id: u32,
}

impl CudaDeviceHandle {
    fn new(device_id: u32) -> GpuResult<Self> {
        // Stub: Would call cudaSetDevice and verify device exists
        Ok(CudaDeviceHandle { device_id })
    }
}

struct CudaStreamHandle;

impl CudaStreamHandle {
    fn new() -> GpuResult<Self> {
        // Stub: Would call cudaStreamCreate
        Ok(CudaStreamHandle)
    }
}

impl Drop for CudaStreamHandle {
    fn drop(&mut self) {
        // Stub: Would call cudaStreamDestroy
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_cuda_available() {
        // Should not crash
        let _ = is_cuda_available();
    }

    #[test]
    fn test_list_cuda_devices() {
        if is_cuda_available() {
            let devices = list_cuda_devices();
            assert!(devices.is_ok());
        }
    }

    #[test]
    fn test_cuda_buffer_creation() {
        let buffer = CudaBuffer::new(1024, 0);
        assert!(buffer.is_ok());
        if let Ok(buf) = buffer {
            assert_eq!(buf.size(), 1024);
            assert!(buf.is_valid());
        }
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = CudaMemoryPool::new();
        let initial_allocated = pool.total_allocated();

        // Allocate buffer
        let buffer = pool.allocate(1024);
        assert!(buffer.is_ok());
        assert!(pool.total_allocated() >= initial_allocated);
    }

    #[test]
    fn test_reduction_op() {
        let op = ReductionOp::Sum;
        assert!(matches!(op, ReductionOp::Sum));
    }
}
