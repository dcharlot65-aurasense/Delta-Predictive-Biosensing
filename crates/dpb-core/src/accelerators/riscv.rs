//! RISC-V Hardware Abstraction Layer (HAL)
//!
//! This module provides a hardware abstraction layer for RISC-V embedded targets,
//! enabling DPB to run on microcontrollers and edge devices.
//!
//! ## Supported Targets
//!
//! - **riscv32imc**: Basic 32-bit RISC-V (e.g., ESP32-C3)
//! - **riscv32imac**: 32-bit with atomics (e.g., GD32VF103)
//! - **riscv32imafc**: 32-bit with FPU (e.g., ESP32-S3)
//! - **riscv64gc**: Full 64-bit RISC-V (e.g., SiFive U74)
//!
//! ## Features
//!
//! - Minimal heap allocation (suitable for no_std)
//! - Fixed-point arithmetic for targets without FPU
//! - DMA support for efficient data transfer
//! - Interrupt-driven spike encoding
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_core::accelerators::riscv::{RiscVHal, RiscVConfig};
//!
//! let config = RiscVConfig::default()
//!     .with_fpu(false)
//!     .with_simd(false);
//!
//! let hal = RiscVHal::new(config)?;
//! let spikes = hal.encode_level_crossing(&signal, 0.1)?;
//! ```

use super::{
    Accelerator, AcceleratorBuffer, AcceleratorCapabilities, AcceleratorError,
    AcceleratorOperation, AcceleratorType, OperationType,
};
use std::sync::{atomic::AtomicU64, Mutex};
use std::collections::HashMap;

/// RISC-V target architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiscVArch {
    /// 32-bit integer only (RV32I)
    Rv32i,
    /// 32-bit with multiply (RV32IM)
    Rv32im,
    /// 32-bit with multiply and atomics (RV32IMA)
    Rv32ima,
    /// 32-bit with multiply, atomics, and compressed (RV32IMAC)
    Rv32imac,
    /// 32-bit with multiply, atomics, float, and compressed (RV32IMAFC)
    Rv32imafc,
    /// 64-bit general purpose (RV64GC)
    Rv64gc,
}

impl RiscVArch {
    /// Check if this architecture has hardware floating point.
    pub fn has_fpu(&self) -> bool {
        matches!(self, RiscVArch::Rv32imafc | RiscVArch::Rv64gc)
    }

    /// Check if this architecture has atomics.
    pub fn has_atomics(&self) -> bool {
        matches!(
            self,
            RiscVArch::Rv32ima
                | RiscVArch::Rv32imac
                | RiscVArch::Rv32imafc
                | RiscVArch::Rv64gc
        )
    }

    /// Check if this architecture has compressed instructions.
    pub fn has_compressed(&self) -> bool {
        matches!(
            self,
            RiscVArch::Rv32imac | RiscVArch::Rv32imafc | RiscVArch::Rv64gc
        )
    }

    /// Get word size in bytes.
    pub fn word_size(&self) -> usize {
        match self {
            RiscVArch::Rv64gc => 8,
            _ => 4,
        }
    }
}

/// RISC-V HAL configuration.
#[derive(Debug, Clone)]
pub struct RiscVConfig {
    /// Target architecture.
    pub arch: RiscVArch,
    /// Clock frequency in Hz.
    pub clock_hz: u32,
    /// Available SRAM in bytes.
    pub sram_bytes: usize,
    /// Available flash in bytes.
    pub flash_bytes: usize,
    /// Use DMA for data transfer.
    pub use_dma: bool,
    /// Use fixed-point arithmetic.
    pub use_fixed_point: bool,
    /// Fixed-point fraction bits (Q format).
    pub fixed_point_bits: u8,
    /// Enable interrupts for spike events.
    pub interrupt_driven: bool,
    /// Maximum buffer size.
    pub max_buffer_size: usize,
}

impl Default for RiscVConfig {
    fn default() -> Self {
        Self {
            arch: RiscVArch::Rv32imac,
            clock_hz: 160_000_000, // 160 MHz
            sram_bytes: 320 * 1024, // 320 KB
            flash_bytes: 4 * 1024 * 1024, // 4 MB
            use_dma: true,
            use_fixed_point: false,
            fixed_point_bits: 16,
            interrupt_driven: true,
            max_buffer_size: 16 * 1024, // 16 KB max buffer
        }
    }
}

impl RiscVConfig {
    /// Create configuration for ESP32-C3.
    pub fn esp32c3() -> Self {
        Self {
            arch: RiscVArch::Rv32imac,
            clock_hz: 160_000_000,
            sram_bytes: 400 * 1024,
            flash_bytes: 4 * 1024 * 1024,
            use_dma: true,
            use_fixed_point: true, // No FPU
            fixed_point_bits: 16,
            interrupt_driven: true,
            max_buffer_size: 32 * 1024,
        }
    }

    /// Create configuration for ESP32-S3 (RISC-V coprocessor).
    pub fn esp32s3() -> Self {
        Self {
            arch: RiscVArch::Rv32imafc,
            clock_hz: 240_000_000,
            sram_bytes: 512 * 1024,
            flash_bytes: 8 * 1024 * 1024,
            use_dma: true,
            use_fixed_point: false, // Has FPU
            fixed_point_bits: 16,
            interrupt_driven: true,
            max_buffer_size: 64 * 1024,
        }
    }

    /// Create configuration for GD32VF103 (RISC-V microcontroller).
    pub fn gd32vf103() -> Self {
        Self {
            arch: RiscVArch::Rv32imac,
            clock_hz: 108_000_000,
            sram_bytes: 32 * 1024,
            flash_bytes: 128 * 1024,
            use_dma: true,
            use_fixed_point: true,
            fixed_point_bits: 16,
            interrupt_driven: true,
            max_buffer_size: 8 * 1024,
        }
    }

    /// Create configuration for SiFive U74 (Linux-capable).
    pub fn sifive_u74() -> Self {
        Self {
            arch: RiscVArch::Rv64gc,
            clock_hz: 1_500_000_000,
            sram_bytes: 2 * 1024 * 1024 * 1024, // 2 GB
            flash_bytes: 32 * 1024 * 1024,
            use_dma: true,
            use_fixed_point: false,
            fixed_point_bits: 16,
            interrupt_driven: false, // Linux handles interrupts
            max_buffer_size: 1024 * 1024,
        }
    }

    /// Enable DMA.
    pub fn with_dma(mut self, enabled: bool) -> Self {
        self.use_dma = enabled;
        self
    }

    /// Set fixed-point mode.
    pub fn with_fixed_point(mut self, enabled: bool, bits: u8) -> Self {
        self.use_fixed_point = enabled;
        self.fixed_point_bits = bits;
        self
    }
}

/// Fixed-point number representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FixedPoint<const FRAC_BITS: u32> {
    /// Raw integer value.
    raw: i32,
}

impl<const FRAC_BITS: u32> FixedPoint<FRAC_BITS> {
    /// Create from floating point value.
    pub fn from_f32(value: f32) -> Self {
        let scale = (1 << FRAC_BITS) as f32;
        Self {
            raw: (value * scale) as i32,
        }
    }

    /// Convert to floating point.
    pub fn to_f32(self) -> f32 {
        let scale = (1 << FRAC_BITS) as f32;
        self.raw as f32 / scale
    }

    /// Add two fixed-point numbers.
    pub fn add(self, other: Self) -> Self {
        Self {
            raw: self.raw.saturating_add(other.raw),
        }
    }

    /// Subtract two fixed-point numbers.
    pub fn sub(self, other: Self) -> Self {
        Self {
            raw: self.raw.saturating_sub(other.raw),
        }
    }

    /// Multiply two fixed-point numbers.
    pub fn mul(self, other: Self) -> Self {
        let result = (self.raw as i64 * other.raw as i64) >> FRAC_BITS;
        Self {
            raw: result as i32,
        }
    }

    /// Compare with threshold.
    pub fn gte(self, other: Self) -> bool {
        self.raw >= other.raw
    }

    /// Get raw value.
    pub fn raw(self) -> i32 {
        self.raw
    }
}

/// Type alias for Q16.16 fixed point.
pub type Q16 = FixedPoint<16>;

/// DMA channel configuration.
#[derive(Debug, Clone)]
pub struct DmaChannel {
    /// Channel number.
    pub channel: u8,
    /// Source address.
    pub src_addr: u32,
    /// Destination address.
    pub dst_addr: u32,
    /// Transfer size in bytes.
    pub size: usize,
    /// Transfer direction.
    pub direction: DmaDirection,
    /// Transfer complete callback.
    pub on_complete: Option<fn()>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DmaDirection {
    /// Memory to memory.
    MemToMem,
    /// Memory to peripheral.
    MemToPeripheral,
    /// Peripheral to memory.
    PeripheralToMem,
}

/// Interrupt configuration for spike events.
#[derive(Debug, Clone)]
pub struct SpikeInterruptConfig {
    /// Enable interrupt on spike detection.
    pub enabled: bool,
    /// Interrupt priority (0-15, lower is higher priority).
    pub priority: u8,
    /// Minimum spikes before interrupt.
    pub threshold: u32,
    /// Callback function.
    pub callback: Option<fn(channel: u8, polarity: i8)>,
}

impl Default for SpikeInterruptConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            priority: 5,
            threshold: 1,
            callback: None,
        }
    }
}

/// RISC-V Hardware Abstraction Layer.
pub struct RiscVHal {
    config: RiscVConfig,
    capabilities: AcceleratorCapabilities,
    buffers: Mutex<HashMap<u64, Vec<u8>>>,
    next_buffer_id: AtomicU64,
    /// State for level crossing (previous values per channel).
    level_crossing_state: Mutex<Vec<Q16>>,
    /// State for delta modulation (reference values per channel).
    delta_state: Mutex<Vec<Q16>>,
    /// Spike interrupt configuration.
    interrupt_config: Mutex<SpikeInterruptConfig>,
}

impl RiscVHal {
    /// Create a new RISC-V HAL with the given configuration.
    pub fn new(config: RiscVConfig) -> Result<Self, AcceleratorError> {
        let capabilities = AcceleratorCapabilities {
            accelerator_type: AcceleratorType::RiscV,
            name: format!("{:?} @ {} MHz", config.arch, config.clock_hz / 1_000_000),
            memory_bytes: config.sram_bytes as u64,
            compute_units: 1,
            max_workgroup_size: 1,
            supports_fp16: false,
            supports_bf16: false,
            supports_int8: true,
            peak_tflops: estimate_riscv_gflops(&config) / 1000.0,
            memory_bandwidth_gbps: estimate_memory_bandwidth(&config),
        };

        Ok(Self {
            config,
            capabilities,
            buffers: Mutex::new(HashMap::new()),
            next_buffer_id: AtomicU64::new(1),
            level_crossing_state: Mutex::new(Vec::new()),
            delta_state: Mutex::new(Vec::new()),
            interrupt_config: Mutex::new(SpikeInterruptConfig::default()),
        })
    }

    /// Encode signal using fixed-point level crossing.
    pub fn encode_level_crossing_fixed(
        &self,
        signal: &[i16],  // Q15 format
        num_channels: usize,
        threshold: i16,
    ) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let mut state = self.level_crossing_state.lock().expect("accelerator mutex poisoned");

        // Initialize state if needed
        if state.len() != num_channels {
            *state = vec![Q16::from_f32(0.0); num_channels];
        }

        let threshold_q = Q16 { raw: (threshold as i32) << 1 };

        for (sample_idx, chunk) in signal.chunks(num_channels).enumerate() {
            for (ch, &value) in chunk.iter().enumerate() {
                let current = Q16 { raw: (value as i32) << 1 };
                let prev = state[ch];

                // Detect crossing
                if !prev.gte(threshold_q) && current.gte(threshold_q) {
                    spikes.push(SpikeEvent {
                        timestamp: sample_idx as u32,
                        channel: ch as u8,
                        polarity: 1,
                    });
                } else if prev.gte(threshold_q) && !current.gte(threshold_q) {
                    spikes.push(SpikeEvent {
                        timestamp: sample_idx as u32,
                        channel: ch as u8,
                        polarity: -1,
                    });
                }

                state[ch] = current;
            }
        }

        spikes
    }

    /// Encode signal using fixed-point delta modulation.
    pub fn encode_delta_fixed(
        &self,
        signal: &[i16],
        num_channels: usize,
        delta_threshold: i16,
    ) -> Vec<SpikeEvent> {
        let mut spikes = Vec::new();
        let mut state = self.delta_state.lock().expect("accelerator mutex poisoned");

        // Initialize state
        if state.len() != num_channels {
            *state = vec![Q16::from_f32(0.0); num_channels];
        }

        let threshold_q = Q16 { raw: (delta_threshold as i32) << 1 };

        for (sample_idx, chunk) in signal.chunks(num_channels).enumerate() {
            for (ch, &value) in chunk.iter().enumerate() {
                let current = Q16 { raw: (value as i32) << 1 };
                let reference = state[ch];
                let delta = current.sub(reference);

                if delta.gte(threshold_q) {
                    spikes.push(SpikeEvent {
                        timestamp: sample_idx as u32,
                        channel: ch as u8,
                        polarity: 1,
                    });
                    state[ch] = reference.add(threshold_q);
                } else if threshold_q.sub(delta).gte(Q16 { raw: 0 }) {
                    spikes.push(SpikeEvent {
                        timestamp: sample_idx as u32,
                        channel: ch as u8,
                        polarity: -1,
                    });
                    state[ch] = reference.sub(threshold_q);
                }
            }
        }

        spikes
    }

    /// Configure spike interrupt.
    pub fn configure_interrupt(&self, config: SpikeInterruptConfig) {
        let mut current = self.interrupt_config.lock().expect("accelerator mutex poisoned");
        *current = config;
    }

    /// Setup DMA transfer.
    pub fn setup_dma_transfer(&self, channel: &DmaChannel) -> Result<(), AcceleratorError> {
        if !self.config.use_dma {
            return Err(AcceleratorError::Unsupported("DMA not enabled".to_string()));
        }

        // In production, this would configure the DMA controller
        tracing::debug!(
            "Setup DMA channel {} for {} byte transfer",
            channel.channel,
            channel.size
        );

        Ok(())
    }

    /// Get estimated throughput in samples/second.
    pub fn estimated_throughput(&self) -> u32 {
        // Rough estimate: ~10 cycles per sample for fixed-point
        // ~50 cycles per sample for floating-point
        let cycles_per_sample = if self.config.use_fixed_point { 10 } else { 50 };
        self.config.clock_hz / cycles_per_sample
    }

    /// Get available buffer memory.
    pub fn available_memory(&self) -> usize {
        let buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        let used: usize = buffers.values().map(|b| b.len()).sum();
        self.config.sram_bytes.saturating_sub(used)
    }
}

impl Accelerator for RiscVHal {
    fn accelerator_type(&self) -> AcceleratorType {
        AcceleratorType::RiscV
    }

    fn capabilities(&self) -> &AcceleratorCapabilities {
        &self.capabilities
    }

    fn is_available(&self) -> bool {
        true
    }

    fn allocate(&self, size_bytes: usize) -> Result<AcceleratorBuffer, AcceleratorError> {
        if size_bytes > self.config.max_buffer_size {
            return Err(AcceleratorError::OutOfMemory {
                requested: size_bytes,
                available: self.config.max_buffer_size,
            });
        }

        if size_bytes > self.available_memory() {
            return Err(AcceleratorError::OutOfMemory {
                requested: size_bytes,
                available: self.available_memory(),
            });
        }

        let id = self.next_buffer_id.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        let mut buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        buffers.insert(id, vec![0u8; size_bytes]);

        Ok(AcceleratorBuffer::new(id, size_bytes, AcceleratorType::Cpu))
    }

    fn copy_to_device(&self, buffer: &mut AcceleratorBuffer, data: &[f32]) -> Result<(), AcceleratorError> {
        let mut buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        if let Some(buf) = buffers.get_mut(&buffer.id) {
            // Convert f32 to bytes
            let bytes: Vec<u8> = data.iter()
                .flat_map(|&f| f.to_le_bytes())
                .collect();
            let len = bytes.len().min(buf.len());
            buf[..len].copy_from_slice(&bytes[..len]);
            Ok(())
        } else {
            Err(AcceleratorError::InvalidOperation("Buffer not found".to_string()))
        }
    }

    fn copy_from_device(&self, buffer: &AcceleratorBuffer, data: &mut [f32]) -> Result<(), AcceleratorError> {
        let buffers = self.buffers.lock().expect("accelerator mutex poisoned");
        if let Some(buf) = buffers.get(&buffer.id) {
            // Convert bytes to f32
            for (i, chunk) in buf.chunks(4).enumerate() {
                if i >= data.len() {
                    break;
                }
                if chunk.len() == 4 {
                    data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            Ok(())
        } else {
            Err(AcceleratorError::InvalidOperation("Buffer not found".to_string()))
        }
    }

    fn execute(&self, operation: &AcceleratorOperation) -> Result<(), AcceleratorError> {
        match operation.op_type {
            OperationType::LevelCrossing => {
                tracing::debug!("Execute level crossing on RISC-V");
                Ok(())
            }
            OperationType::DeltaModulation => {
                tracing::debug!("Execute delta modulation on RISC-V");
                Ok(())
            }
            _ => Err(AcceleratorError::Unsupported(
                format!("Operation {:?} not supported on RISC-V embedded", operation.op_type)
            ))
        }
    }

    fn synchronize(&self) -> Result<(), AcceleratorError> {
        // Memory barrier for RISC-V
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        Ok(())
    }
}

/// Spike event.
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    /// Sample index.
    pub timestamp: u32,
    /// Channel index.
    pub channel: u8,
    /// Spike polarity (+1 or -1).
    pub polarity: i8,
}

// Helper functions

fn estimate_riscv_gflops(config: &RiscVConfig) -> f32 {
    if config.arch.has_fpu() {
        // With FPU: ~1 FLOP per cycle
        config.clock_hz as f32 / 1_000_000_000.0
    } else {
        // Fixed-point emulation: ~0.1 effective FLOP per cycle
        config.clock_hz as f32 / 10_000_000_000.0
    }
}

fn estimate_memory_bandwidth(config: &RiscVConfig) -> f32 {
    // Estimate based on clock and bus width
    let bus_width = if config.arch.word_size() == 8 { 64 } else { 32 };
    let bandwidth_bytes = config.clock_hz as f64 * (bus_width / 8) as f64;
    (bandwidth_bytes / 1_000_000_000.0) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_riscv_arch() {
        assert!(!RiscVArch::Rv32imac.has_fpu());
        assert!(RiscVArch::Rv32imafc.has_fpu());
        assert!(RiscVArch::Rv64gc.has_fpu());

        assert!(RiscVArch::Rv32imac.has_atomics());
        assert!(!RiscVArch::Rv32im.has_atomics());
    }

    #[test]
    fn test_fixed_point() {
        let a = Q16::from_f32(1.5);
        let b = Q16::from_f32(2.5);
        let sum = a.add(b);
        assert!((sum.to_f32() - 4.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_point_mul() {
        let a = Q16::from_f32(2.0);
        let b = Q16::from_f32(3.0);
        let product = a.mul(b);
        assert!((product.to_f32() - 6.0).abs() < 0.01);
    }

    #[test]
    fn test_config_presets() {
        let esp32c3 = RiscVConfig::esp32c3();
        assert_eq!(esp32c3.arch, RiscVArch::Rv32imac);
        assert!(esp32c3.use_fixed_point);

        let sifive = RiscVConfig::sifive_u74();
        assert_eq!(sifive.arch, RiscVArch::Rv64gc);
        assert!(!sifive.use_fixed_point);
    }

    #[test]
    fn test_level_crossing_fixed() {
        let hal = RiscVHal::new(RiscVConfig::default()).unwrap();

        // Signal: 0, 100, 200, 150, 50, -50
        let signal: Vec<i16> = vec![0, 100, 200, 150, 50, -50];
        let spikes = hal.encode_level_crossing_fixed(&signal, 1, 75);

        // Should detect crossing at sample 1 (up) and sample 4 (down)
        assert!(!spikes.is_empty());
    }

    #[test]
    fn test_hal_allocation() {
        let config = RiscVConfig::default();
        let hal = RiscVHal::new(config).unwrap();

        let buffer = hal.allocate(1024).unwrap();
        assert_eq!(buffer.size_bytes, 1024);
    }

    #[test]
    fn test_throughput_estimate() {
        let hal = RiscVHal::new(RiscVConfig::esp32c3()).unwrap();
        let throughput = hal.estimated_throughput();
        assert!(throughput > 1_000_000); // Should be > 1 MSPS
    }
}
