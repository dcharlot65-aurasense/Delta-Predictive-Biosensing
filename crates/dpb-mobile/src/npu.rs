//! Mobile Neural Processing Unit (NPU) support for DPB.
//!
//! This module provides abstractions for mobile NPU accelerators:
//! - **Qualcomm Hexagon DSP/HVX** (Snapdragon devices)
//! - **ARM Ethos-U** (Cortex-M devices with NPU)
//! - **Apple Neural Engine** (via CoreML)
//! - **Samsung NPU** (Exynos devices)
//!
//! ## Hexagon DSP
//!
//! Qualcomm Hexagon is available on Snapdragon SoCs and provides:
//! - HVX (Hexagon Vector Extensions) for SIMD operations
//! - HTP (Hexagon Tensor Processor) for ML acceleration
//! - Low-power always-on sensing
//!
//! ## ARM Ethos-U
//!
//! ARM Ethos-U NPUs are designed for Cortex-M microcontrollers:
//! - Ethos-U55: 32-256 MAC/cycle, optimized for always-on
//! - Ethos-U65: 256-512 MAC/cycle, higher throughput
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_mobile::npu::{NpuBackend, NpuEncoder, NpuConfig};
//!
//! // Detect available NPU
//! let backend = NpuBackend::detect();
//!
//! // Create encoder with NPU acceleration
//! let config = NpuConfig::default();
//! let encoder = NpuEncoder::new(backend, config)?;
//!
//! // Encode signal
//! let spikes = encoder.encode(&signal)?;
//! ```

use std::fmt;

/// NPU backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuBackend {
    /// No NPU available, use CPU fallback.
    Cpu,
    /// Qualcomm Hexagon DSP.
    QualcommHexagon,
    /// Qualcomm Hexagon HTP (Tensor Processor).
    QualcommHtp,
    /// ARM Ethos-U55 NPU.
    ArmEthosU55,
    /// ARM Ethos-U65 NPU.
    ArmEthosU65,
    /// Apple Neural Engine.
    AppleAne,
    /// Samsung NPU (Exynos).
    SamsungNpu,
    /// MediaTek APU.
    MediaTekApu,
}

// Not consulted yet; kept so a caller's configuration is not silently
// discarded.
#[allow(dead_code)]
impl NpuBackend {
    /// Detect available NPU on current device.
    pub fn detect() -> Self {
        // Platform-specific detection
        #[cfg(target_os = "android")]
        {
            // Check for Qualcomm Hexagon
            if Self::check_hexagon() {
                return NpuBackend::QualcommHexagon;
            }
            // Check for Samsung NPU
            if Self::check_samsung_npu() {
                return NpuBackend::SamsungNpu;
            }
            // Check for MediaTek APU
            if Self::check_mediatek_apu() {
                return NpuBackend::MediaTekApu;
            }
        }

        #[cfg(target_os = "ios")]
        {
            return NpuBackend::AppleAne;
        }

        // Fallback to CPU
        NpuBackend::Cpu
    }

    /// Check if Hexagon DSP is available.
    #[cfg(target_os = "android")]
    fn check_hexagon() -> bool {
        // Would check /dev/adsprpc-smd or similar
        std::path::Path::new("/dev/adsprpc-smd").exists()
    }

    #[cfg(not(target_os = "android"))]
    fn check_hexagon() -> bool {
        false
    }

    /// Check if Samsung NPU is available.
    #[cfg(target_os = "android")]
    fn check_samsung_npu() -> bool {
        // Would check for Samsung Eden runtime
        false
    }

    #[cfg(not(target_os = "android"))]
    fn check_samsung_npu() -> bool {
        false
    }

    /// Check if MediaTek APU is available.
    #[cfg(target_os = "android")]
    fn check_mediatek_apu() -> bool {
        // Would check for NeuroPilot runtime
        false
    }

    #[cfg(not(target_os = "android"))]
    fn check_mediatek_apu() -> bool {
        false
    }

    /// Get backend name.
    pub fn name(&self) -> &'static str {
        match self {
            NpuBackend::Cpu => "CPU",
            NpuBackend::QualcommHexagon => "Qualcomm Hexagon DSP",
            NpuBackend::QualcommHtp => "Qualcomm Hexagon HTP",
            NpuBackend::ArmEthosU55 => "ARM Ethos-U55",
            NpuBackend::ArmEthosU65 => "ARM Ethos-U65",
            NpuBackend::AppleAne => "Apple Neural Engine",
            NpuBackend::SamsungNpu => "Samsung NPU",
            NpuBackend::MediaTekApu => "MediaTek APU",
        }
    }

    /// Get theoretical peak performance in TOPS (Tera Operations Per Second).
    pub fn peak_tops(&self) -> f32 {
        match self {
            NpuBackend::Cpu => 0.1,
            NpuBackend::QualcommHexagon => 4.0, // HVX
            NpuBackend::QualcommHtp => 15.0,    // Snapdragon 8 Gen 2
            NpuBackend::ArmEthosU55 => 0.5,     // 256 MAC @ 500MHz
            NpuBackend::ArmEthosU65 => 1.0,     // 512 MAC @ 500MHz
            NpuBackend::AppleAne => 17.0,       // M1/M2 Neural Engine
            NpuBackend::SamsungNpu => 10.0,     // Exynos 2200
            NpuBackend::MediaTekApu => 6.0,     // Dimensity 9000
        }
    }
}

impl fmt::Display for NpuBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// NPU configuration.
#[derive(Debug, Clone)]
pub struct NpuConfig {
    /// Power mode.
    pub power_mode: NpuPowerMode,
    /// Use INT8 quantization.
    pub use_int8: bool,
    /// Use INT16 quantization.
    pub use_int16: bool,
    /// Batch size for inference.
    pub batch_size: usize,
    /// Number of channels.
    pub num_channels: usize,
    /// Cache model in NPU memory.
    pub cache_model: bool,
}

impl Default for NpuConfig {
    fn default() -> Self {
        Self {
            power_mode: NpuPowerMode::Balanced,
            use_int8: true,
            use_int16: false,
            batch_size: 1,
            num_channels: 32,
            cache_model: true,
        }
    }
}

/// NPU power mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NpuPowerMode {
    /// Low power mode for battery savings.
    LowPower,
    /// Balanced performance and power.
    Balanced,
    /// High performance mode.
    HighPerformance,
    /// Burst mode (temporary high performance).
    Burst,
}

/// Hexagon DSP configuration.
#[derive(Debug, Clone)]
pub struct HexagonConfig {
    /// DSP clock frequency in MHz.
    pub clock_mhz: u32,
    /// Enable HVX (vector extensions).
    pub enable_hvx: bool,
    /// HVX mode (64-byte or 128-byte vectors).
    pub hvx_mode: HvxMode,
    /// VTCM (Vector TCM) size in KB.
    pub vtcm_size_kb: u32,
    /// Enable unsigned PD (Protection Domain).
    pub unsigned_pd: bool,
}

impl Default for HexagonConfig {
    fn default() -> Self {
        Self {
            clock_mhz: 1000,
            enable_hvx: true,
            hvx_mode: HvxMode::Mode128,
            vtcm_size_kb: 256,
            unsigned_pd: false,
        }
    }
}

/// Hexagon HVX mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HvxMode {
    /// 64-byte vectors.
    Mode64,
    /// 128-byte vectors.
    Mode128,
}

/// ARM Ethos-U configuration.
#[derive(Debug, Clone)]
pub struct EthosUConfig {
    /// NPU variant.
    pub variant: EthosUVariant,
    /// MAC units enabled (affects power).
    pub mac_units: u32,
    /// SRAM size in KB.
    pub sram_size_kb: u32,
    /// Enable PMU (Performance Monitor Unit).
    pub enable_pmu: bool,
}

impl Default for EthosUConfig {
    fn default() -> Self {
        Self {
            variant: EthosUVariant::U55_256,
            mac_units: 256,
            sram_size_kb: 512,
            enable_pmu: false,
        }
    }
}

/// ARM Ethos-U variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EthosUVariant {
    /// Ethos-U55 with 32 MACs.
    U55_32,
    /// Ethos-U55 with 64 MACs.
    U55_64,
    /// Ethos-U55 with 128 MACs.
    U55_128,
    /// Ethos-U55 with 256 MACs.
    U55_256,
    /// Ethos-U65 with 256 MACs.
    U65_256,
    /// Ethos-U65 with 512 MACs.
    U65_512,
}

impl EthosUVariant {
    /// Get MAC count.
    pub fn mac_count(&self) -> u32 {
        match self {
            EthosUVariant::U55_32 => 32,
            EthosUVariant::U55_64 => 64,
            EthosUVariant::U55_128 => 128,
            EthosUVariant::U55_256 => 256,
            EthosUVariant::U65_256 => 256,
            EthosUVariant::U65_512 => 512,
        }
    }
}

/// NPU-accelerated spike encoder.
pub struct NpuEncoder {
    /// NPU backend.
    backend: NpuBackend,
    /// Configuration.
    config: NpuConfig,
    /// Encoding threshold.
    threshold: f32,
    /// Previous sample values (for stateful encoding).
    prev_values: Vec<f32>,
    /// Quantization scale (for INT8).
    quant_scale: f32,
    /// Quantization zero point.
    quant_zero_point: i32,
}

impl NpuEncoder {
    /// Create a new NPU encoder.
    pub fn new(backend: NpuBackend, config: NpuConfig, threshold: f32) -> Result<Self, NpuError> {
        let prev_values = vec![0.0; config.num_channels];

        // Calculate quantization parameters for INT8
        let (quant_scale, quant_zero_point) = if config.use_int8 {
            // Signal range [-1, 1] into a SIGNED i8, so quantization is
            // symmetric: zero point 0, scale 1/127.
            //
            // This previously used the UINT8 convention -- zero point 128 with
            // scale 2/255 -- in an i8 container whose range is [-128, 127].
            // Every positive input then exceeded 127 and saturated: 0.5
            // quantized to 191.75, clamped to 127, and dequantized back to
            // -0.0078. The entire positive half of the range collapsed to
            // approximately zero, so any encoder configured for int8 emitted
            // meaningless values.
            let scale = 1.0 / 127.0;
            let zero_point = 0;
            (scale, zero_point)
        } else {
            (1.0, 0)
        };

        Ok(Self {
            backend,
            config,
            threshold,
            prev_values,
            quant_scale,
            quant_zero_point,
        })
    }

    /// Get backend info.
    pub fn backend(&self) -> NpuBackend {
        self.backend
    }

    /// Encode signal to spikes.
    pub fn encode(&mut self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        match self.backend {
            NpuBackend::Cpu => self.encode_cpu(signal),
            NpuBackend::QualcommHexagon | NpuBackend::QualcommHtp => self.encode_hexagon(signal),
            NpuBackend::ArmEthosU55 | NpuBackend::ArmEthosU65 => self.encode_ethos(signal),
            NpuBackend::AppleAne => self.encode_ane(signal),
            _ => self.encode_cpu(signal),
        }
    }

    /// CPU fallback encoding.
    fn encode_cpu(&mut self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        // The signal is channel-interleaved, so its length must be a whole
        // number of frames. Integer division alone silently discarded a partial
        // frame -- and returned an EMPTY result for any signal shorter than
        // `num_channels`, which looks like "no spikes" rather than "your data
        // did not match the configured channel count".
        if self.config.num_channels == 0 {
            return Err(NpuError::InvalidConfig(
                "num_channels must be non-zero".to_string(),
            ));
        }
        if !signal.is_empty() && !signal.len().is_multiple_of(self.config.num_channels) {
            return Err(NpuError::InvalidConfig(format!(
                "signal length {} is not a multiple of num_channels {}",
                signal.len(),
                self.config.num_channels
            )));
        }

        let num_samples = signal.len() / self.config.num_channels;
        let mut spikes = Vec::with_capacity(signal.len());

        for sample_idx in 0..num_samples {
            for ch in 0..self.config.num_channels {
                let idx = sample_idx * self.config.num_channels + ch;
                let curr = signal[idx];
                let prev = self.prev_values[ch];

                let spike = if prev < self.threshold && curr >= self.threshold {
                    1i8
                } else if prev >= self.threshold && curr < self.threshold {
                    -1i8
                } else {
                    0i8
                };

                spikes.push(spike);
                self.prev_values[ch] = curr;
            }
        }

        Ok(spikes)
    }

    /// Hexagon DSP encoding.
    fn encode_hexagon(&mut self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        // In production, would use Hexagon SDK:
        // 1. rpcmem_alloc for shared memory
        // 2. Load stub library
        // 3. Call remote procedure
        //
        // For now, fall back to CPU
        self.encode_cpu(signal)
    }

    /// ARM Ethos-U encoding.
    fn encode_ethos(&mut self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        // In production, would use:
        // 1. TensorFlow Lite Micro with Ethos-U delegate
        // 2. Load .tflite model compiled for Ethos-U
        // 3. Run inference
        //
        // For now, fall back to CPU
        self.encode_cpu(signal)
    }

    /// Apple Neural Engine encoding.
    fn encode_ane(&mut self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        // In production, would use CoreML:
        // 1. Load .mlmodel
        // 2. Create prediction request
        // 3. Run inference
        //
        // For now, fall back to CPU
        self.encode_cpu(signal)
    }

    /// Quantize float to INT8.
    #[inline]
    pub fn quantize(&self, value: f32) -> i8 {
        // `.round()` before the cast: `as i8` truncates toward zero, which
        // doubles the worst-case error to a full LSB and biases every magnitude
        // downward. Round-to-nearest keeps it at half an LSB and unbiased.
        let scaled = value / self.quant_scale + self.quant_zero_point as f32;
        scaled.round().clamp(-128.0, 127.0) as i8
    }

    /// Dequantize INT8 to float.
    #[inline]
    pub fn dequantize(&self, value: i8) -> f32 {
        (value as f32 - self.quant_zero_point as f32) * self.quant_scale
    }
}

/// Hexagon DSP encoder (Qualcomm).
// Not consulted yet; kept so a caller's configuration is not silently
// discarded.
#[allow(dead_code)]
pub struct HexagonEncoder {
    /// Configuration.
    config: HexagonConfig,
    /// Threshold.
    threshold: f32,
    /// Number of channels.
    num_channels: usize,
    /// Is initialized.
    initialized: bool,
}

impl HexagonEncoder {
    /// Create new Hexagon encoder.
    pub fn new(
        config: HexagonConfig,
        threshold: f32,
        num_channels: usize,
    ) -> Result<Self, NpuError> {
        Ok(Self {
            config,
            threshold,
            num_channels,
            initialized: false,
        })
    }

    /// Initialize Hexagon runtime.
    pub fn initialize(&mut self) -> Result<(), NpuError> {
        // In production:
        // 1. Open FastRPC channel
        // 2. Load DSP skel library
        // 3. Allocate shared memory
        // 4. Initialize HVX context

        self.initialized = true;
        Ok(())
    }

    /// Check if initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Encode using Hexagon DSP.
    pub fn encode(&self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        if !self.initialized {
            return Err(NpuError::NotInitialized);
        }

        // Would call into DSP via FastRPC
        // For now, CPU fallback
        let num_samples = signal.len() / self.num_channels;
        let mut spikes = vec![0i8; signal.len()];
        let mut prev = vec![0.0f32; self.num_channels];

        for sample_idx in 0..num_samples {
            for (ch, prev_val) in prev.iter_mut().enumerate().take(self.num_channels) {
                let idx = sample_idx * self.num_channels + ch;
                let curr = signal[idx];

                if *prev_val < self.threshold && curr >= self.threshold {
                    spikes[idx] = 1;
                } else if *prev_val >= self.threshold && curr < self.threshold {
                    spikes[idx] = -1;
                }

                *prev_val = curr;
            }
        }

        Ok(spikes)
    }
}

/// ARM Ethos-U encoder.
pub struct EthosUEncoder {
    /// Configuration.
    config: EthosUConfig,
    /// Threshold.
    threshold: f32,
    /// Number of channels.
    num_channels: usize,
    /// Model buffer (compiled .tflite).
    model_buffer: Option<Vec<u8>>,
}

impl EthosUEncoder {
    /// Create new Ethos-U encoder.
    pub fn new(
        config: EthosUConfig,
        threshold: f32,
        num_channels: usize,
    ) -> Result<Self, NpuError> {
        Ok(Self {
            config,
            threshold,
            num_channels,
            model_buffer: None,
        })
    }

    /// Load compiled TFLite model.
    pub fn load_model(&mut self, model_bytes: &[u8]) -> Result<(), NpuError> {
        self.model_buffer = Some(model_bytes.to_vec());
        Ok(())
    }

    /// Check if model is loaded.
    pub fn is_model_loaded(&self) -> bool {
        self.model_buffer.is_some()
    }

    /// Encode using Ethos-U NPU.
    pub fn encode(&self, signal: &[f32]) -> Result<Vec<i8>, NpuError> {
        if self.model_buffer.is_none() {
            return Err(NpuError::ModelNotLoaded);
        }

        // Would use TFLite Micro interpreter with Ethos-U delegate
        // For now, CPU fallback
        let num_samples = signal.len() / self.num_channels;
        let mut spikes = vec![0i8; signal.len()];
        let mut prev = vec![0.0f32; self.num_channels];

        for sample_idx in 0..num_samples {
            for (ch, prev_val) in prev.iter_mut().enumerate().take(self.num_channels) {
                let idx = sample_idx * self.num_channels + ch;
                let curr = signal[idx];

                if *prev_val < self.threshold && curr >= self.threshold {
                    spikes[idx] = 1;
                } else if *prev_val >= self.threshold && curr < self.threshold {
                    spikes[idx] = -1;
                }

                *prev_val = curr;
            }
        }

        Ok(spikes)
    }

    /// Get MAC utilization estimate.
    pub fn estimate_mac_utilization(&self, num_samples: usize) -> f32 {
        let ops_per_sample = self.num_channels * 4; // Approx ops for level crossing
        let total_ops = ops_per_sample * num_samples;
        let available_macs = self.config.variant.mac_count() as usize;

        // Simple utilization estimate
        (total_ops as f32 / available_macs as f32).min(1.0)
    }
}

/// NPU error types.
#[derive(Debug)]
pub enum NpuError {
    /// NPU not available.
    NotAvailable(String),
    /// Not initialized.
    NotInitialized,
    /// Model not loaded.
    ModelNotLoaded,
    /// Invalid configuration.
    InvalidConfig(String),
    /// Runtime error.
    RuntimeError(String),
    /// Memory allocation error.
    MemoryError(String),
}

impl fmt::Display for NpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NpuError::NotAvailable(msg) => write!(f, "NPU not available: {}", msg),
            NpuError::NotInitialized => write!(f, "NPU not initialized"),
            NpuError::ModelNotLoaded => write!(f, "Model not loaded"),
            NpuError::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            NpuError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
            NpuError::MemoryError(msg) => write!(f, "Memory error: {}", msg),
        }
    }
}

impl std::error::Error for NpuError {}

/// Detect all available NPU backends on this device.
pub fn detect_npus() -> Vec<NpuBackend> {
    let backends = vec![NpuBackend::Cpu]; // CPU always available

    // Platform-specific detection
    #[cfg(target_os = "android")]
    {
        if NpuBackend::check_hexagon() {
            backends.push(NpuBackend::QualcommHexagon);
        }
    }

    #[cfg(target_os = "ios")]
    {
        backends.push(NpuBackend::AppleAne);
    }

    backends
}

/// Get the best available NPU backend.
pub fn best_npu() -> NpuBackend {
    NpuBackend::detect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_npu_backend_detection() {
        let backend = NpuBackend::detect();
        // On non-mobile platforms, should be CPU
        #[cfg(not(any(target_os = "android", target_os = "ios")))]
        assert_eq!(backend, NpuBackend::Cpu);
    }

    #[test]
    fn test_npu_encoder_cpu_fallback() {
        // The signal below is single-channel, so the config must say so; the
        // default is 32 channels, under which five samples are not even one
        // complete frame.
        let config = NpuConfig {
            num_channels: 1,
            ..NpuConfig::default()
        };
        let mut encoder = NpuEncoder::new(NpuBackend::Cpu, config, 0.1).unwrap();

        let signal = vec![0.0, 0.05, 0.15, 0.1, 0.05]; // 5 samples, 1 channel
        let spikes = encoder.encode(&signal).unwrap();

        assert_eq!(spikes.len(), 5);
        assert_eq!(spikes[2], 1); // Crossing at sample 2
    }

    #[test]
    fn test_hexagon_config() {
        let config = HexagonConfig::default();
        assert!(config.enable_hvx);
        assert_eq!(config.hvx_mode, HvxMode::Mode128);
    }

    #[test]
    fn test_ethos_variant_mac_count() {
        assert_eq!(EthosUVariant::U55_256.mac_count(), 256);
        assert_eq!(EthosUVariant::U65_512.mac_count(), 512);
    }

    #[test]
    fn test_quantization() {
        let config = NpuConfig::default();
        let encoder = NpuEncoder::new(NpuBackend::Cpu, config, 0.1).unwrap();

        let original = 0.5f32;
        let quantized = encoder.quantize(original);
        let dequantized = encoder.dequantize(quantized);

        // Should be approximately equal (quantization error)
        assert!((original - dequantized).abs() < 0.01);
    }
    /// INT8 quantization must round-trip across the whole declared range.
    ///
    /// Regression: the scale and zero point followed the UINT8 convention in a
    /// signed container, so everything above zero saturated to 127 and came
    /// back as approximately zero. A single-value check near the top of the
    /// range would have caught it; one near the bottom would not.
    #[test]
    fn test_quantization_round_trip_over_range() {
        let config = NpuConfig::default();
        let encoder = NpuEncoder::new(NpuBackend::Cpu, config, 0.1).unwrap();

        for step in 0..=40 {
            let original = -1.0 + (step as f32) * 0.05;
            let round_tripped = encoder.dequantize(encoder.quantize(original));
            assert!(
                (original - round_tripped).abs() <= 0.5 / 127.0 + 1e-6,
                "{original} -> {round_tripped} exceeds half an LSB"
            );
        }
    }

    /// A signal that is not a whole number of frames is an error, not silence.
    #[test]
    fn test_encode_rejects_partial_frame() {
        let config = NpuConfig {
            num_channels: 4,
            ..NpuConfig::default()
        };
        let mut encoder = NpuEncoder::new(NpuBackend::Cpu, config, 0.1).unwrap();

        // Six samples across four channels is one frame and a half.
        assert!(encoder.encode(&[0.0; 6]).is_err());
        assert!(encoder.encode(&[0.0; 8]).is_ok());
    }
}
