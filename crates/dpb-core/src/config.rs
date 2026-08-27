//! Configuration structures for the DPB framework.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main framework configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DpbConfig {
    /// GPU configuration
    pub gpu: GpuConfig,
    /// Signal processing configuration
    pub signal: SignalConfig,
    /// Neural network configuration
    pub network: NetworkConfig,
    /// Training configuration
    pub training: TrainingConfig,
    /// Logging and debugging
    pub logging: LoggingConfig,
}

/// GPU-related configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Enable GPU acceleration
    pub enabled: bool,
    /// Backend preference (Vulkan, Metal, DX12, WebGPU)
    pub backend: Option<String>,
    /// Power preference (low, high)
    pub power_preference: PowerPreference,
    /// Maximum batch size for GPU operations
    pub max_batch_size: usize,
    /// Device selection strategy
    pub device_selection: DeviceSelection,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backend: None,
            power_preference: PowerPreference::HighPerformance,
            max_batch_size: 256,
            device_selection: DeviceSelection::Auto,
        }
    }
}

/// Power preference for GPU selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PowerPreference {
    /// Low power consumption
    LowPower,
    /// High performance
    HighPerformance,
}

impl From<PowerPreference> for wgpu::PowerPreference {
    fn from(pref: PowerPreference) -> Self {
        match pref {
            PowerPreference::LowPower => wgpu::PowerPreference::LowPower,
            PowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        }
    }
}

/// Device selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceSelection {
    /// Automatic selection
    Auto,
    /// Prefer discrete GPU
    DiscreteGpu,
    /// Prefer integrated GPU
    IntegratedGpu,
    /// CPU fallback
    Cpu,
}

/// Signal processing configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalConfig {
    /// Default sample rate for resampling
    pub default_sample_rate: f64,
    /// Enable preprocessing
    pub preprocessing_enabled: bool,
    /// Filter configuration
    pub filters: FilterConfig,
    /// Normalization method
    pub normalization: NormalizationMethod,
    /// Window size for analysis (samples)
    pub window_size: usize,
    /// Window overlap (samples)
    pub window_overlap: usize,
}

impl Default for SignalConfig {
    fn default() -> Self {
        Self {
            default_sample_rate: 1000.0,
            preprocessing_enabled: true,
            filters: FilterConfig::default(),
            normalization: NormalizationMethod::ZScore,
            window_size: 1024,
            window_overlap: 512,
        }
    }
}

/// Filter configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterConfig {
    /// Highpass cutoff frequency (Hz)
    pub highpass_cutoff: Option<f64>,
    /// Lowpass cutoff frequency (Hz)
    pub lowpass_cutoff: Option<f64>,
    /// Notch filter frequencies (Hz)
    pub notch_frequencies: Vec<f64>,
    /// Filter order
    pub order: usize,
}

impl Default for FilterConfig {
    fn default() -> Self {
        Self {
            highpass_cutoff: Some(0.5),
            lowpass_cutoff: Some(100.0),
            notch_frequencies: vec![50.0, 60.0], // Power line noise
            order: 4,
        }
    }
}

/// Normalization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormalizationMethod {
    /// No normalization
    None,
    /// Z-score normalization
    ZScore,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Min-max normalization to [-1, 1]
    MinMaxSymmetric,
    /// Robust scaling (using median and IQR)
    Robust,
}

/// Neural network configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Network architecture
    pub architecture: Vec<LayerConfig>,
    /// Time step for simulation (seconds)
    pub dt: f64,
    /// Membrane time constant (seconds)
    pub tau_mem: f64,
    /// Synaptic time constant (seconds)
    pub tau_syn: f64,
    /// Spike threshold
    pub threshold: f64,
    /// Reset potential
    pub reset_potential: f64,
    /// Surrogate gradient configuration
    pub surrogate_gradient: SurrogateGradientConfig,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            architecture: vec![],
            dt: 0.001,      // 1ms
            tau_mem: 0.020, // 20ms
            tau_syn: 0.005, // 5ms
            threshold: 1.0,
            reset_potential: 0.0,
            surrogate_gradient: SurrogateGradientConfig::default(),
        }
    }
}

/// Layer configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Layer type
    pub layer_type: LayerType,
    /// Number of neurons
    pub neurons: usize,
    /// Activation function (if applicable)
    pub activation: Option<String>,
    /// Custom parameters
    pub params: HashMap<String, f64>,
}

/// Layer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Leaky Integrate-and-Fire
    Lif,
    /// Adaptive LIF
    Alif,
    /// Izhikevich
    Izhikevich,
    /// Hodgkin-Huxley
    HodgkinHuxley,
    /// Recurrent layer
    Recurrent,
}

/// Surrogate gradient configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurrogateGradientConfig {
    /// Gradient type
    pub gradient_type: SurrogateType,
    /// Scale parameter
    pub scale: f64,
    /// Additional parameters
    pub params: HashMap<String, f64>,
}

impl Default for SurrogateGradientConfig {
    fn default() -> Self {
        Self {
            gradient_type: SurrogateType::FastSigmoid,
            scale: 10.0,
            params: HashMap::new(),
        }
    }
}

/// Surrogate gradient types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SurrogateType {
    /// Fast sigmoid
    FastSigmoid,
    /// Arctangent
    ArcTan,
    /// Exponential
    Exponential,
    /// Piecewise linear
    PiecewiseLinear,
}

/// Training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size
    pub batch_size: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// Optimizer type
    pub optimizer: OptimizerType,
    /// Loss function
    pub loss_function: LossType,
    /// Learning rate schedule
    pub lr_schedule: Option<LrScheduleConfig>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// Validation split ratio
    pub validation_split: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 0.001,
            optimizer: OptimizerType::Adam,
            loss_function: LossType::SpikeCrossEntropy,
            lr_schedule: None,
            early_stopping: None,
            gradient_clip: Some(1.0),
            validation_split: 0.2,
        }
    }
}

/// Optimizer types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    Sgd,
    /// Adam optimizer
    Adam,
    /// AdamW optimizer
    AdamW,
    /// RMSprop
    RmsProp,
}

/// Loss function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossType {
    /// Mean squared error
    Mse,
    /// Cross entropy
    CrossEntropy,
    /// Spike-based cross entropy
    SpikeCrossEntropy,
    /// Spike count loss
    SpikeCount,
}

/// Learning rate schedule configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LrScheduleConfig {
    /// Schedule type
    pub schedule_type: LrScheduleType,
    /// Step size (for step decay)
    pub step_size: Option<usize>,
    /// Decay rate
    pub decay_rate: f64,
}

/// Learning rate schedule types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LrScheduleType {
    /// Constant learning rate
    Constant,
    /// Step decay
    StepDecay,
    /// Exponential decay
    ExponentialDecay,
    /// Cosine annealing
    CosineAnnealing,
}

/// Early stopping configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EarlyStoppingConfig {
    /// Patience (epochs to wait for improvement)
    pub patience: usize,
    /// Minimum delta for improvement
    pub min_delta: f64,
    /// Metric to monitor
    pub monitor: String,
}

/// Logging and debugging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable progress bars
    pub progress_bars: bool,
    /// Log directory
    pub log_dir: Option<String>,
    /// Checkpoint directory
    pub checkpoint_dir: Option<String>,
    /// Checkpoint frequency (epochs)
    pub checkpoint_frequency: usize,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            progress_bars: true,
            log_dir: None,
            checkpoint_dir: None,
            checkpoint_frequency: 10,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DpbConfig::default();
        assert!(config.gpu.enabled);
        assert_eq!(config.training.batch_size, 32);
    }

    #[test]
    fn test_serialization() {
        let config = DpbConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DpbConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.training.epochs, config.training.epochs);
    }

    #[test]
    fn test_power_preference_conversion() {
        let pref = PowerPreference::HighPerformance;
        let wgpu_pref: wgpu::PowerPreference = pref.into();
        assert!(matches!(wgpu_pref, wgpu::PowerPreference::HighPerformance));
    }
}
