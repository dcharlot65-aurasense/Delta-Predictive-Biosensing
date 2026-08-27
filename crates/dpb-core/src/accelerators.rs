//! Hardware accelerator abstractions for DPB.
//!
//! This module provides traits and implementations for various hardware
//! accelerators including Intel Gaudi, Graphcore IPU, RISC-V, and standard GPUs.
//!
//! ## Supported Accelerators
//!
//! - **Intel Gaudi**: AI training accelerator with high memory bandwidth (Synapse AI SDK)
//! - **Graphcore IPU**: Massively parallel graph processor (Poplar SDK)
//! - **RISC-V**: Embedded targets with fixed-point arithmetic (ESP32-C3, SiFive, etc.)
//! - **NVIDIA GPU**: CUDA-based acceleration (via wgpu)
//! - **AMD GPU**: ROCm-based acceleration (via wgpu)
//! - **CPU SIMD**: AVX2/AVX-512 vectorization
//!
//! ## Submodules
//!
//! Feature-gated implementations provide full SDK integration:
//!
//! - `gaudi` (feature: `intel-gaudi`) - Full Synapse AI SDK integration
//! - `ipu` (feature: `graphcore-ipu`) - Full Poplar SDK integration
//! - `riscv` (feature: `riscv-hal`) - Hardware abstraction for embedded RISC-V targets
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_core::accelerators::{Accelerator, AcceleratorType, detect_accelerators};
//!
//! // Detect available accelerators
//! let accelerators = detect_accelerators();
//!
//! // Use the best available accelerator
//! if let Some(accel) = accelerators.first() {
//!     let result = accel.encode_signal(&signal, &config)?;
//! }
//! ```

// Feature-gated submodules
#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "intel-gaudi")]
pub mod gaudi;

#[cfg(feature = "graphcore-ipu")]
pub mod ipu;

#[cfg(feature = "riscv-hal")]
pub mod riscv;

use std::fmt;

/// Accelerator type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcceleratorType {
    /// Standard CPU execution.
    Cpu,
    /// CPU with SIMD (AVX2/AVX-512).
    CpuSimd,
    /// NVIDIA CUDA GPU.
    NvidiaCuda,
    /// AMD ROCm GPU.
    AmdRocm,
    /// Intel Gaudi AI accelerator.
    IntelGaudi,
    /// Graphcore IPU.
    GraphcoreIpu,
    /// Apple Metal GPU.
    AppleMetal,
    /// WebGPU (browser).
    WebGpu,
    /// RISC-V embedded processor.
    RiscV,
}

impl fmt::Display for AcceleratorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorType::Cpu => write!(f, "CPU"),
            AcceleratorType::CpuSimd => write!(f, "CPU (SIMD)"),
            AcceleratorType::NvidiaCuda => write!(f, "NVIDIA CUDA"),
            AcceleratorType::AmdRocm => write!(f, "AMD ROCm"),
            AcceleratorType::IntelGaudi => write!(f, "Intel Gaudi"),
            AcceleratorType::GraphcoreIpu => write!(f, "Graphcore IPU"),
            AcceleratorType::AppleMetal => write!(f, "Apple Metal"),
            AcceleratorType::WebGpu => write!(f, "WebGPU"),
            AcceleratorType::RiscV => write!(f, "RISC-V"),
        }
    }
}

/// Accelerator capabilities.
#[derive(Debug, Clone)]
pub struct AcceleratorCapabilities {
    /// Type of accelerator.
    pub accelerator_type: AcceleratorType,
    /// Device name.
    pub name: String,
    /// Available memory in bytes.
    pub memory_bytes: u64,
    /// Compute units (cores, processing elements, etc.).
    pub compute_units: u32,
    /// Maximum workgroup/thread block size.
    pub max_workgroup_size: u32,
    /// Supports float16 computation.
    pub supports_fp16: bool,
    /// Supports bfloat16 computation.
    pub supports_bf16: bool,
    /// Supports int8 computation.
    pub supports_int8: bool,
    /// Theoretical peak TFLOPS (FP32).
    pub peak_tflops: f32,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbps: f32,
}

impl Default for AcceleratorCapabilities {
    fn default() -> Self {
        Self {
            accelerator_type: AcceleratorType::Cpu,
            name: "CPU".to_string(),
            memory_bytes: 0,
            compute_units: 1,
            max_workgroup_size: 1,
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: true,
            peak_tflops: 0.1,
            memory_bandwidth_gbps: 50.0,
        }
    }
}

/// Trait for hardware accelerators.
pub trait Accelerator: Send + Sync {
    /// Get accelerator type.
    fn accelerator_type(&self) -> AcceleratorType;

    /// Get capabilities.
    fn capabilities(&self) -> &AcceleratorCapabilities;

    /// Check if accelerator is available.
    fn is_available(&self) -> bool;

    /// Allocate memory on the accelerator.
    fn allocate(&self, size_bytes: usize) -> Result<AcceleratorBuffer, AcceleratorError>;

    /// Copy data to accelerator.
    fn copy_to_device(
        &self,
        buffer: &mut AcceleratorBuffer,
        data: &[f32],
    ) -> Result<(), AcceleratorError>;

    /// Copy data from accelerator.
    fn copy_from_device(
        &self,
        buffer: &AcceleratorBuffer,
        data: &mut [f32],
    ) -> Result<(), AcceleratorError>;

    /// Execute a compute operation.
    fn execute(&self, operation: &AcceleratorOperation) -> Result<(), AcceleratorError>;

    /// Synchronize (wait for all operations to complete).
    fn synchronize(&self) -> Result<(), AcceleratorError>;
}

/// Buffer allocated on an accelerator.
pub struct AcceleratorBuffer {
    /// Unique buffer ID.
    pub id: u64,
    /// Size in bytes.
    pub size_bytes: usize,
    /// Accelerator type.
    pub accelerator: AcceleratorType,
    /// Opaque handle (implementation-specific).
    handle: Option<Box<dyn std::any::Any + Send + Sync>>,
}

impl AcceleratorBuffer {
    /// Create a new buffer.
    pub fn new(id: u64, size_bytes: usize, accelerator: AcceleratorType) -> Self {
        Self {
            id,
            size_bytes,
            accelerator,
            handle: None,
        }
    }

    /// Set the internal handle.
    pub fn set_handle<T: 'static + Send + Sync>(&mut self, handle: T) {
        self.handle = Some(Box::new(handle));
    }

    /// Get handle reference.
    pub fn handle<T: 'static>(&self) -> Option<&T> {
        self.handle.as_ref()?.downcast_ref()
    }
}

/// Operation to execute on accelerator.
#[derive(Debug, Clone)]
pub struct AcceleratorOperation {
    /// Operation type.
    pub op_type: OperationType,
    /// Input buffers.
    pub inputs: Vec<u64>,
    /// Output buffers.
    pub outputs: Vec<u64>,
    /// Parameters.
    pub params: OperationParams,
}

/// Type of accelerator operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationType {
    /// Level crossing detection.
    LevelCrossing,
    /// Delta modulation.
    DeltaModulation,
    /// Temporal contrast.
    TemporalContrast,
    /// Matrix multiplication.
    MatMul,
    /// Element-wise operation.
    ElementWise,
    /// Reduction operation.
    Reduce,
    /// Convolution.
    Convolution,
    /// Custom operation.
    Custom,
}

/// Parameters for accelerator operations.
#[derive(Debug, Clone)]
pub struct OperationParams {
    /// Number of channels.
    pub num_channels: usize,
    /// Number of samples.
    pub num_samples: usize,
    /// Thresholds.
    pub thresholds: Vec<f32>,
    /// Extra parameters.
    pub extra: std::collections::HashMap<String, f32>,
}

impl Default for OperationParams {
    fn default() -> Self {
        Self {
            num_channels: 1,
            num_samples: 0,
            thresholds: Vec::new(),
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Accelerator error.
#[derive(Debug)]
pub enum AcceleratorError {
    /// Accelerator not available.
    NotAvailable(String),
    /// Out of memory.
    OutOfMemory {
        /// Bytes the allocation asked for.
        requested: usize,
        /// Bytes the device had free.
        available: usize,
    },
    /// Invalid operation.
    InvalidOperation(String),
    /// Driver error.
    DriverError(String),
    /// Compilation error (for JIT kernels).
    CompilationError(String),
    /// Execution error.
    ExecutionError(String),
    /// Unsupported operation.
    Unsupported(String),
}

impl fmt::Display for AcceleratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AcceleratorError::NotAvailable(msg) => write!(f, "Accelerator not available: {}", msg),
            AcceleratorError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of memory: requested {} bytes, {} available",
                    requested, available
                )
            }
            AcceleratorError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
            AcceleratorError::DriverError(msg) => write!(f, "Driver error: {}", msg),
            AcceleratorError::CompilationError(msg) => write!(f, "Compilation error: {}", msg),
            AcceleratorError::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            AcceleratorError::Unsupported(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for AcceleratorError {}

/// CPU accelerator (fallback implementation).
pub struct CpuAccelerator {
    capabilities: AcceleratorCapabilities,
    buffers: std::sync::Mutex<std::collections::HashMap<u64, Vec<f32>>>,
    next_buffer_id: std::sync::atomic::AtomicU64,
}

impl CpuAccelerator {
    /// Create new CPU accelerator.
    pub fn new() -> Self {
        let has_avx2 = cfg!(target_feature = "avx2");
        let has_avx512 = cfg!(target_feature = "avx512f");

        let accel_type = if has_avx512 || has_avx2 {
            AcceleratorType::CpuSimd
        } else {
            AcceleratorType::Cpu
        };

        Self {
            capabilities: AcceleratorCapabilities {
                accelerator_type: accel_type,
                name: format!(
                    "CPU {}",
                    if has_avx512 {
                        "(AVX-512)"
                    } else if has_avx2 {
                        "(AVX2)"
                    } else {
                        ""
                    }
                ),
                memory_bytes: get_system_memory(),
                compute_units: num_cpus(),
                max_workgroup_size: 1,
                supports_fp16: false,
                supports_bf16: false,
                supports_int8: true,
                peak_tflops: estimate_cpu_tflops(),
                memory_bandwidth_gbps: 50.0,
            },
            buffers: std::sync::Mutex::new(std::collections::HashMap::new()),
            next_buffer_id: std::sync::atomic::AtomicU64::new(1),
        }
    }
}

impl Default for CpuAccelerator {
    fn default() -> Self {
        Self::new()
    }
}

impl Accelerator for CpuAccelerator {
    fn accelerator_type(&self) -> AcceleratorType {
        self.capabilities.accelerator_type
    }

    fn capabilities(&self) -> &AcceleratorCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        true
    }

    fn allocate(&self, size_bytes: usize) -> Result<AcceleratorBuffer, AcceleratorError> {
        let id = self
            .next_buffer_id
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let num_floats = size_bytes / 4;

        let mut buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        buffers.insert(id, vec![0.0; num_floats]);

        Ok(AcceleratorBuffer::new(
            id,
            size_bytes,
            self.capabilities.accelerator_type,
        ))
    }

    fn copy_to_device(
        &self,
        buffer: &mut AcceleratorBuffer,
        data: &[f32],
    ) -> Result<(), AcceleratorError> {
        let mut buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        if let Some(buf) = buffers.get_mut(&buffer.id) {
            let len = data.len().min(buf.len());
            buf[..len].copy_from_slice(&data[..len]);
            Ok(())
        } else {
            Err(AcceleratorError::InvalidOperation(
                "Buffer not found".to_string(),
            ))
        }
    }

    fn copy_from_device(
        &self,
        buffer: &AcceleratorBuffer,
        data: &mut [f32],
    ) -> Result<(), AcceleratorError> {
        let buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        if let Some(buf) = buffers.get(&buffer.id) {
            let len = data.len().min(buf.len());
            data[..len].copy_from_slice(&buf[..len]);
            Ok(())
        } else {
            Err(AcceleratorError::InvalidOperation(
                "Buffer not found".to_string(),
            ))
        }
    }

    fn execute(&self, operation: &AcceleratorOperation) -> Result<(), AcceleratorError> {
        let mut buffers = self.buffers.lock().expect("accelerator mutex poisoned");

        match operation.op_type {
            OperationType::LevelCrossing => {
                // Get input and output buffers
                let input_id = operation.inputs.first().ok_or_else(|| {
                    AcceleratorError::InvalidOperation("No input buffer".to_string())
                })?;
                let output_id = operation.outputs.first().ok_or_else(|| {
                    AcceleratorError::InvalidOperation("No output buffer".to_string())
                })?;

                let input = buffers
                    .get(input_id)
                    .ok_or_else(|| {
                        AcceleratorError::InvalidOperation("Input buffer not found".to_string())
                    })?
                    .clone();

                let num_channels = operation.params.num_channels;
                let thresholds = &operation.params.thresholds;

                // Simple level crossing detection
                let mut spikes = Vec::new();
                let mut prev = vec![0.0f32; num_channels];

                for (i, chunk) in input.chunks(num_channels).enumerate() {
                    for (ch, &value) in chunk.iter().enumerate() {
                        let threshold = thresholds.get(ch).copied().unwrap_or(0.1);
                        if prev[ch] < threshold && value >= threshold {
                            spikes.push((i * num_channels + ch) as f32);
                        }
                        prev[ch] = value;
                    }
                }

                if let Some(output) = buffers.get_mut(output_id) {
                    let len = spikes.len().min(output.len());
                    output[..len].copy_from_slice(&spikes[..len]);
                }
            }
            _ => {
                return Err(AcceleratorError::Unsupported(format!(
                    "Operation {:?} not implemented",
                    operation.op_type
                )));
            }
        }

        Ok(())
    }

    fn synchronize(&self) -> Result<(), AcceleratorError> {
        Ok(()) // CPU is always synchronous
    }
}

/// Detect available accelerators.
pub fn detect_accelerators() -> Vec<Box<dyn Accelerator>> {
    // Always have CPU fallback. `mut` is only exercised when an accelerator
    // feature is on -- in a default build nothing below ever pushes.
    #[allow(unused_mut)]
    let mut accelerators: Vec<Box<dyn Accelerator>> = vec![Box::new(CpuAccelerator::new())];

    // Check for Intel Gaudi
    #[cfg(feature = "intel-gaudi")]
    if let Ok(gaudi) = gaudi::GaudiAccelerator::new() {
        accelerators.push(Box::new(gaudi));
    }

    // Check for Graphcore IPU
    #[cfg(feature = "graphcore-ipu")]
    if let Ok(ipu) = ipu::IpuAccelerator::new() {
        accelerators.push(Box::new(ipu));
    }

    // Check for NVIDIA CUDA
    #[cfg(feature = "cuda")]
    if let Ok(gpu) = cuda::CudaAccelerator::new(cuda::CudaConfig::default()) {
        accelerators.push(Box::new(gpu));
    }

    // Check for a RISC-V vector unit
    #[cfg(feature = "riscv-hal")]
    if let Ok(hal) = riscv::RiscVHal::new(riscv::RiscVConfig::default()) {
        accelerators.push(Box::new(hal));
    }

    accelerators
}

/// Get the best available accelerator.
pub fn best_accelerator() -> Box<dyn Accelerator> {
    let accelerators = detect_accelerators();

    // Priority: IPU > Gaudi > GPU > CPU SIMD > RISC-V > CPU
    let priority = |a: &dyn Accelerator| match a.accelerator_type() {
        AcceleratorType::GraphcoreIpu => 7,
        AcceleratorType::IntelGaudi => 6,
        AcceleratorType::NvidiaCuda => 5,
        AcceleratorType::AmdRocm => 4,
        AcceleratorType::AppleMetal => 3,
        AcceleratorType::CpuSimd => 2,
        AcceleratorType::RiscV => 1, // Specialized embedded target
        AcceleratorType::Cpu => 0,
        AcceleratorType::WebGpu => 0,
    };

    accelerators
        .into_iter()
        .filter(|a| a.is_available())
        .max_by_key(|a| priority(a.as_ref()))
        .unwrap_or_else(|| Box::new(CpuAccelerator::new()))
}

// Helper functions

fn get_system_memory() -> u64 {
    // Platform-specific memory detection would go here
    // For now, return a reasonable default
    16 * 1024 * 1024 * 1024 // 16 GB
}

fn num_cpus() -> u32 {
    std::thread::available_parallelism()
        .map(|p| p.get() as u32)
        .unwrap_or(4)
}

fn estimate_cpu_tflops() -> f32 {
    // Rough estimate based on core count
    // Assumes ~100 GFLOPS per core for modern CPUs
    (num_cpus() as f32) * 0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_accelerator() {
        let accel = CpuAccelerator::new();
        assert!(accel.is_available());
        assert!(matches!(
            accel.accelerator_type(),
            AcceleratorType::Cpu | AcceleratorType::CpuSimd
        ));
    }

    #[test]
    fn test_buffer_allocation() {
        let accel = CpuAccelerator::new();
        let buffer = accel.allocate(1024).unwrap();
        assert_eq!(buffer.size_bytes, 1024);
    }

    #[test]
    fn test_copy_to_from_device() {
        let accel = CpuAccelerator::new();
        let mut buffer = accel.allocate(16).unwrap();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        accel.copy_to_device(&mut buffer, &data).unwrap();

        let mut result = vec![0.0; 4];
        accel.copy_from_device(&buffer, &mut result).unwrap();

        assert_eq!(data, result);
    }

    #[test]
    fn test_detect_accelerators() {
        let accelerators = detect_accelerators();
        assert!(!accelerators.is_empty());
        // Should at least have CPU
        assert!(accelerators.iter().any(|a| matches!(
            a.accelerator_type(),
            AcceleratorType::Cpu | AcceleratorType::CpuSimd
        )));
    }

    #[test]
    fn test_best_accelerator() {
        let accel = best_accelerator();
        assert!(accel.is_available());
    }

    #[test]
    fn test_accelerator_type_display() {
        assert_eq!(format!("{}", AcceleratorType::IntelGaudi), "Intel Gaudi");
        assert_eq!(
            format!("{}", AcceleratorType::GraphcoreIpu),
            "Graphcore IPU"
        );
    }

    #[test]
    fn test_level_crossing_operation() {
        let accel = CpuAccelerator::new();

        let mut input_buf = accel.allocate(20).unwrap(); // 5 samples
        let output_buf = accel.allocate(40).unwrap();

        let input = vec![0.0, 0.5, 0.9, 1.1, 0.8]; // Single channel, crossing at sample 3
        accel.copy_to_device(&mut input_buf, &input).unwrap();

        let operation = AcceleratorOperation {
            op_type: OperationType::LevelCrossing,
            inputs: vec![input_buf.id],
            outputs: vec![output_buf.id],
            params: OperationParams {
                num_channels: 1,
                num_samples: 5,
                thresholds: vec![1.0],
                ..Default::default()
            },
        };

        accel.execute(&operation).unwrap();
    }
}
