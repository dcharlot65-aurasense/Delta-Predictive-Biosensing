//! GPU Acceleration for Spiking Neural Networks
//!
//! This module provides GPU-accelerated implementations for SNN training and inference,
//! supporting multiple backends including CUDA and Metal.
//!
//! ## Features
//!
//! - **Multi-Backend Support**: CUDA for NVIDIA GPUs, Metal for Apple Silicon
//! - **Optimized Kernels**: Parallel spike propagation, STDP weight updates, sparse operations
//! - **Memory Management**: Efficient buffer pooling and async transfers
//! - **Stream/Queue Management**: Overlapped computation and data transfer
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_snn::gpu::{GpuDevice, Backend, auto_detect_backend};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Auto-detect the best available backend
//! let backend = auto_detect_backend();
//! let device = backend.create_device(0)?;
//!
//! // Allocate GPU memory
//! let buffer = device.allocate(1024)?;
//!
//! // Transfer data to GPU
//! let data = vec![0.0f32; 1024];
//! device.copy_to_device(&data, &buffer)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The GPU module is organized around traits for backend abstraction:
//!
//! - [`GpuDevice`]: Device management and memory allocation
//! - [`GpuBuffer`]: GPU memory buffers
//! - [`SpikeKernel`], [`WeightUpdateKernel`]: Kernel interfaces
//!
//! Backend-specific implementations are in:
//! - [`cuda`]: NVIDIA CUDA backend
//! - [`metal`]: Apple Metal backend

pub mod kernels;
pub mod memory;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "metal")]
pub mod metal;

use std::sync::Arc;
use thiserror::Error;

/// GPU-specific error types
#[derive(Debug, Error)]
pub enum GpuError {
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),

    #[error("Device not found: {0}")]
    DeviceNotFound(u32),

    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    #[error("Memory transfer failed: {0}")]
    TransferFailed(String),

    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize { expected: usize, actual: usize },

    #[error("Synchronization failed: {0}")]
    SyncFailed(String),

    #[error("Backend error: {0}")]
    BackendError(String),
}

pub type GpuResult<T> = Result<T, GpuError>;

/// GPU backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// NVIDIA CUDA backend
    Cuda,
    /// Apple Metal backend
    Metal,
    /// CPU fallback (no GPU acceleration)
    Cpu,
}

impl Backend {
    /// Check if this backend is available on the current system
    pub fn is_available(&self) -> bool {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda => cuda::is_cuda_available(),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => false,

            #[cfg(feature = "metal")]
            Backend::Metal => metal::is_metal_available(),
            #[cfg(not(feature = "metal"))]
            Backend::Metal => false,

            Backend::Cpu => true,
        }
    }

    /// Create a device for this backend
    pub fn create_device(&self, device_id: u32) -> GpuResult<Box<dyn GpuDevice>> {
        match self {
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                let device = cuda::CudaDevice::new(device_id)?;
                Ok(Box::new(device))
            }
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => Err(GpuError::BackendNotAvailable(
                "CUDA support not compiled".to_string(),
            )),

            #[cfg(feature = "metal")]
            Backend::Metal => {
                let device = metal::MetalDevice::new(device_id)?;
                Ok(Box::new(device))
            }
            #[cfg(not(feature = "metal"))]
            Backend::Metal => Err(GpuError::BackendNotAvailable(
                "Metal support not compiled".to_string(),
            )),

            Backend::Cpu => Err(GpuError::BackendNotAvailable(
                "CPU backend does not support GpuDevice interface".to_string(),
            )),
        }
    }

    /// Get the name of the backend
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::Metal => "Metal",
            Backend::Cpu => "CPU",
        }
    }
}

/// Trait for GPU device management
///
/// Provides device initialization, memory allocation, and kernel execution
pub trait GpuDevice: Send + Sync {
    /// Get the device ID
    fn device_id(&self) -> u32;

    /// Get the backend type
    fn backend(&self) -> Backend;

    /// Allocate device memory
    fn allocate(&self, size_bytes: usize) -> GpuResult<Arc<dyn GpuBuffer>>;

    /// Copy data from host to device
    fn copy_to_device(&self, src: &[f32], dst: &Arc<dyn GpuBuffer>) -> GpuResult<()>;

    /// Copy data from device to host
    fn copy_to_host(&self, src: &Arc<dyn GpuBuffer>, dst: &mut [f32]) -> GpuResult<()>;

    /// Copy data between device buffers
    fn copy_device_to_device(
        &self,
        src: &Arc<dyn GpuBuffer>,
        dst: &Arc<dyn GpuBuffer>,
    ) -> GpuResult<()>;

    /// Synchronize device (wait for all operations to complete)
    fn synchronize(&self) -> GpuResult<()>;

    /// Get device memory info (total, free) in bytes
    fn memory_info(&self) -> GpuResult<(usize, usize)>;

    /// Get device name
    fn name(&self) -> String;

    /// Get compute capability or equivalent
    fn compute_capability(&self) -> (u32, u32);
}

/// Trait for GPU memory buffers
pub trait GpuBuffer: Send + Sync + std::fmt::Debug {
    /// Get the buffer size in bytes
    fn size(&self) -> usize;

    /// Get the buffer pointer (for backend-specific operations)
    fn as_ptr(&self) -> *mut u8;

    /// Check if buffer is valid
    fn is_valid(&self) -> bool;
}

/// Auto-detect the best available GPU backend
///
/// Returns the first available backend in priority order:
/// 1. CUDA (if available)
/// 2. Metal (if available)
/// 3. CPU (fallback)
///
/// # Examples
///
/// ```
/// use dpb_snn::gpu::auto_detect_backend;
///
/// let backend = auto_detect_backend();
/// println!("Using backend: {}", backend.name());
/// ```
pub fn auto_detect_backend() -> Backend {
    if Backend::Cuda.is_available() {
        Backend::Cuda
    } else if Backend::Metal.is_available() {
        Backend::Metal
    } else {
        Backend::Cpu
    }
}

/// List all available GPU devices for a backend
///
/// # Examples
///
/// ```
/// use dpb_snn::gpu::{list_devices, Backend};
///
/// # fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let backend = Backend::Cuda;
/// if let Ok(devices) = list_devices(backend) {
///     for (id, name) in devices {
///         println!("Device {}: {}", id, name);
///     }
/// }
/// # Ok(())
/// # }
/// ```
pub fn list_devices(backend: Backend) -> GpuResult<Vec<(u32, String)>> {
    match backend {
        #[cfg(feature = "cuda")]
        Backend::Cuda => cuda::list_cuda_devices(),
        #[cfg(not(feature = "cuda"))]
        Backend::Cuda => Err(GpuError::BackendNotAvailable(
            "CUDA support not compiled".to_string(),
        )),

        #[cfg(feature = "metal")]
        Backend::Metal => metal::list_metal_devices(),
        #[cfg(not(feature = "metal"))]
        Backend::Metal => Err(GpuError::BackendNotAvailable(
            "Metal support not compiled".to_string(),
        )),

        Backend::Cpu => Ok(vec![(0, "CPU".to_string())]),
    }
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub id: u32,
    pub name: String,
    pub backend: Backend,
    pub total_memory: usize,
    pub compute_capability: (u32, u32),
}

impl DeviceInfo {
    /// Get information about a specific device
    pub fn query(backend: Backend, device_id: u32) -> GpuResult<Self> {
        let device = backend.create_device(device_id)?;
        let (total_memory, _) = device.memory_info()?;

        Ok(DeviceInfo {
            id: device_id,
            name: device.name(),
            backend,
            total_memory,
            compute_capability: device.compute_capability(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_names() {
        assert_eq!(Backend::Cuda.name(), "CUDA");
        assert_eq!(Backend::Metal.name(), "Metal");
        assert_eq!(Backend::Cpu.name(), "CPU");
    }

    #[test]
    fn test_cpu_backend_always_available() {
        assert!(Backend::Cpu.is_available());
    }

    #[test]
    fn test_auto_detect_backend() {
        let backend = auto_detect_backend();
        // Should at least return CPU
        assert!(backend == Backend::Cpu || backend.is_available());
    }

    #[test]
    fn test_backend_equality() {
        assert_eq!(Backend::Cuda, Backend::Cuda);
        assert_ne!(Backend::Cuda, Backend::Metal);
    }
}
