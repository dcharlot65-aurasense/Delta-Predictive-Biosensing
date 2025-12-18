//! # dpb-snn - Spiking Neural Networks
//!
//! Spiking Neural Network architectures, training, and deployment.
//!
//! ## Features
//!
//! - **Architectures**: Feedforward, convolutional, recurrent SNNs
//! - **Training**: BPTT, surrogate gradients, ANN-to-SNN conversion
//! - **Calibration**: Temperature scaling, isotonic regression, uncertainty quantification
//! - **Explainability**: Spike importance, attention maps, feature attribution
//! - **Fusion**: Multi-modal sensor fusion architectures
//! - **Baselines**: 44 ANN architectures for comparison
//! - **Export**: ONNX format, weight serialization
//!
//! ## Quick Start: Building an SNN
//!
//! ```rust
//! use dpb_snn::*;
//! use dpb_neurons::prelude::*;
//!
//! # fn example() -> SNNResult<()> {
//! // Build a feedforward SNN
//! let config = SNNConfig::default();
//! let mut network = FeedforwardSNN::new(
//!     vec![128, 64, 10],  // layer sizes
//!     config,
//! );
//!
//! // Configure training
//! let mut trainer = BPTT::new(
//!     FastSigmoid::default(),  // surrogate gradient
//!     0.001,                    // learning rate
//! );
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Model Calibration
//!
//! ```rust
//! use dpb_snn::calibration::*;
//!
//! # fn example() -> dpb_snn::SNNResult<()> {
//! # let logits = vec![vec![1.0, 2.0, 3.0]];
//! # let labels = vec![2];
//! # let predictions = vec![vec![0.1, 0.3, 0.6]];
//! // Calibrate model confidence scores
//! let mut calibrator = TemperatureScaling::new();
//! calibrator.fit(&logits, &labels)?;
//! let calibrated = calibrator.calibrate(&predictions);
//!
//! // Evaluate calibration quality
//! let ece = expected_calibration_error(&calibrated, &labels, 10);
//! println!("Expected Calibration Error: {:.4}", ece);
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Explainability
//!
//! ```rust
//! use dpb_snn::explain::*;
//! use ndarray::Array2;
//!
//! # fn example() -> dpb_snn::SNNResult<()> {
//! # let spikes = Array2::from_shape_vec((10, 100), vec![false; 1000]).unwrap();
//! # let gradients = Array2::from_shape_vec((10, 100), vec![0.1; 1000]).unwrap();
//! # let weights = Array2::from_shape_vec((10, 100), vec![0.5; 1000]).unwrap();
//! // Compute spike importance for interpretability
//! let importance = compute_spike_importance(
//!     &spikes,
//!     &gradients,
//!     &weights,
//! );
//!
//! // Aggregate to neuron-level importance
//! let neuron_importance = aggregate_to_neurons(&importance);
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Multi-Modal Fusion
//!
//! ```rust
//! use dpb_snn::fusion::*;
//!
//! # fn example() -> dpb_snn::SNNResult<()> {
//! // Create a cross-modal attention fusion network
//! let config = FusionConfig {
//!     ecg_channels: 12,
//!     imu_channels: 6,
//!     video_channels: 3,
//!     num_classes: 5,
//! };
//!
//! let network = CrossModalAttentionSNN::new(config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`architectures`] | SNN architectures (feedforward, CNN, RNN, transformer) |
//! | [`layers`] | Spiking layers (linear, conv, pooling, attention) |
//! | [`training`] | Training algorithms (BPTT, OTTT, SLTT) |
//! | [`decoders`] | Output decoders for spike trains |
//! | [`calibration`] | Model calibration and uncertainty |
//! | [`explain`] | Explainability and interpretability |
//! | [`fusion`] | Multi-modal fusion architectures |
//! | [`baselines`] | ANN baseline architectures |
//! | [`conversion`] | ANN-to-SNN conversion |
//! | [`analysis`] | Training analysis and convergence |
//! | [`learning`] | Unsupervised learning rules (STDP, Hebbian) |
//! | [`optimization`] | Network pruning and compression |
//! | [`export`] | Model export utilities |

pub mod tensor;
pub mod layers;
pub mod architectures;
pub mod training;
pub mod conversion;
pub mod decoders;
pub mod export;
pub mod fusion;
pub mod analysis;
pub mod baselines;
pub mod learning;
pub mod optimization;
pub mod calibration;
pub mod explain;

// Re-export commonly used types
pub use tensor::{SpikeTensor, SpikeRepresentation};
pub use layers::{
    SpikingLinear, SpikingConv2d, SpikingConv1d,
    SpikingSumPool2d, SpikingMaxPool2d,
    SpikingRNN, SpikingLSTM,
    SpikingAttention,
};
pub use architectures::{
    FeedforwardSNN, ConvolutionalSNN, RecurrentSNN,
    SpikingGCN, SpikingTransformer,
};
pub use training::{
    SurrogateGradient, BPTT, OTTT, SLTT,
    SpikingCrossEntropy, SpikeCountLoss, SpikeTimingLoss,
    AdamOptimizer, SGDOptimizer,
};
pub use conversion::{ANNToSNNConverter, WeightNormalization, ThresholdBalancing};
pub use decoders::{
    // Rate-based decoders
    SpikeRateDecoder, FirstSpikeDecoder, PopulationDecoder, MaxSpikeDecoder,
    WindowedRateDecoder, ExponentialRateDecoder, AdaptiveRateDecoder,
    NormalizedRateDecoder, WeightedRateDecoder,
    // Temporal decoders
    TemporalPatternDecoder, LatencyDecoder, ISIDecoder, BurstDecoder,
    LastSpikeDecoder, PhaseDecoder, RankOrderDecoder,
    // Clinical score decoders
    UPDRSDecoder, TremorSeverityDecoder, GaitScoreDecoder,
    UPDRSMotorDecoder, UPDRSTremorDecoder, UPDRSBradykinesiaDecoder,
    UPDRSRigidityDecoder, UPDRSGaitDecoder, TUGDecoder,
    BergBalanceDecoder, MoCADecoder, VoiceHDDecoder,
    PDQ39Decoder, HoehnYahrDecoder, SEADLDecoder,
    // Regression decoders
    HeartRateDecoder, HRVDecoder, TremorFrequencyDecoder,
    TremorAmplitudeDecoder, GaitVelocityDecoder, StrideTimeDecoder,
    TappingFrequencyDecoder, ReactionTimeDecoder, SpeechRateDecoder,
    PupilDiameterDecoder,
    // Classification decoders
    BinaryClassDecoder, MultiClassDecoder, TremorTypeDecoder,
    GaitPhaseDecoder, SleepStageDecoder, ActivityDecoder,
    EmotionDecoder, FatigueDecoder, MedicationStateDecoder,
    DyskinesiasDecoder,
    // Base decoder trait
    Decoder,
};
pub use fusion::{
    FusionNetwork, FusionConfig, Modality,
    EarlyFusionSNN, LateFusionSNN, CrossModalAttentionSNN,
    HierarchicalFusionSNN, TemporalAlignmentSNN, GatedFusionSNN,
    NeuroPlaySNN, CognitiveMotorFusionSNN,
    CognitiveMotorFusion, CognitiveMotorFusionConfig,
    CognitiveProfile, MotorProfile, IntegratedAssessment,
    DissociationPattern, DissociationType, ChangeMetrics, ChangeDirection,
    RiskCategory, DomainZScores,
};
pub use analysis::{
    ConvergenceAnalyzer, TrainingMetrics, AnalysisReport,
    LossPlateauDetector, AccuracyPlateauDetector, EarlyStoppingAnalyzer,
    ConvergenceRateAnalyzer, OscillationDetector, DivergenceDetector,
    LearningCurveSmoothed, GeneralizationGapAnalyzer, OverfittingDetector,
    LearningRateAnalyzer, BatchSizeAnalyzer, EpochEfficiencyAnalyzer,
    GradientNormTracker, GradientFlowAnalyzer, VanishingGradientDetector,
    ExplodingGradientDetector, SurrogateGradientAnalyzer,
    SpikeRateTracker, SparsityTracker, SilentNeuronDetector,
    SaturatedNeuronDetector, TemporalDynamicsAnalyzer,
    WeightDistributionTracker, WeightMagnitudeTracker, WeightSparsityTracker,
    WeightUpdateTracker, MethodComparisonAnalyzer, HyperparameterSensitivityAnalyzer,
};
pub use baselines::{
    ANNBaseline, Tensor as BaselineTensor,
    // MLP architectures
    MLP2Layer, MLP3Layer, MLP4Layer, MLPDropout, MLPBatchNorm, MLPResidual,
    MLPWideSingle, MLPDeep,
    // CNN architectures
    CNN1DSmall, CNN1DMedium, CNN1DLarge, CNN1DResidual, CNN1DDilated,
    CNN2DLeNet, CNN2DVGG, CNN2DResNet, CNN2DMobileNet, TCN,
    // RNN architectures
    SimpleRNN, LSTM, BiLSTM, StackedLSTM, GRU, BiGRU, StackedGRU,
    PeepholeLSTM, AttentionLSTM, IndRNN,
    // Transformer architectures
    TransformerEncoder, TransformerSmall, TransformerMedium, TransformerLarge,
    LinearTransformer, Performer, Informer, Autoformer,
    // Specialized architectures
    ECGNet, DeepGait, TremorNet, VoiceNet,
    MultimodalFusion, AttentionFusion, GraphNN, HybridCNNRNN,
    // Conversion utilities
    ANNToSNNConverter as BaselineConverter,
    ConversionConfig as BaselineConversionConfig,
    WeightNormalizationMethod, ThresholdBalancingStrategy,
    convert_model_to_snn,
};

// Re-export learning types
pub use learning::{
    BCMRule, CovarianceRule, HebbianLayer, HebbianRule, OjasRule, SynapticTrace, STDP,
};

// Re-export optimization types
pub use optimization::{NetworkPruner, PruningMask, PruningSchedule, PruningStats, PruningStrategy};

// Re-export calibration types
pub use calibration::{
    TemperatureScaling, PlattScaling, IsotonicCalibration,
    UncertaintyEstimator, MCDropout, EnsembleUncertainty, ConfidenceInterval,
    expected_calibration_error, maximum_calibration_error, reliability_diagram,
    brier_score, negative_log_likelihood, ReliabilityBin, bootstrap_ci,
};

// Re-export explainability types
pub use explain::{
    SpikeImportance, NeuronImportance, LayerImportance,
    compute_spike_importance, compute_importance_by_perturbation, aggregate_to_neurons,
    AttentionMap, TemporalAttention, SpatialAttention,
    FeatureAttribution, GradientAttribution, IntegratedGradients, SpikeSHAP,
    ExplanationVisualizer, HeatmapData, export_explanation_json,
};

use dpb_core::error::{DpbError, Result};
use ndarray::{Array, ArrayD};
use serde::{Deserialize, Serialize};

/// SNN-specific error type
#[derive(Debug, thiserror::Error)]
pub enum SNNError {
    #[error("Invalid network configuration: {0}")]
    InvalidConfig(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("Layer error: {0}")]
    Layer(String),

    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    #[error("Conversion error: {0}")]
    Conversion(String),

    #[error("Export error: {0}")]
    Export(String),

    #[error(transparent)]
    DpbError(#[from] DpbError),
}

pub type SNNResult<T> = std::result::Result<T, SNNError>;

/// Neuron model types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuronModel {
    /// Leaky Integrate-and-Fire
    LIF,
    /// Adaptive LIF
    AdaptiveLIF,
    /// Izhikevich
    Izhikevich,
    /// Hodgkin-Huxley
    HodgkinHuxley,
    /// Spike Response Model
    SRM,
}

/// Common neuron parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronParams {
    /// Membrane time constant (ms)
    pub tau_mem: f32,
    /// Synaptic time constant (ms)
    pub tau_syn: f32,
    /// Threshold voltage
    pub v_threshold: f32,
    /// Reset voltage
    pub v_reset: f32,
    /// Refractory period (ms)
    pub t_refrac: f32,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_syn: 5.0,
            v_threshold: 1.0,
            v_reset: 0.0,
            t_refrac: 2.0,
        }
    }
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SNNConfig {
    /// Time step size (ms)
    pub dt: f32,
    /// Number of time steps
    pub num_steps: usize,
    /// Neuron model
    pub neuron_model: NeuronModel,
    /// Neuron parameters
    pub neuron_params: NeuronParams,
    /// Enable GPU acceleration
    pub use_gpu: bool,
}

impl Default for SNNConfig {
    fn default() -> Self {
        Self {
            dt: 1.0,
            num_steps: 100,
            neuron_model: NeuronModel::LIF,
            neuron_params: NeuronParams::default(),
            use_gpu: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuron_params_default() {
        let params = NeuronParams::default();
        assert_eq!(params.tau_mem, 20.0);
        assert_eq!(params.v_threshold, 1.0);
    }

    #[test]
    fn test_snn_config_default() {
        let config = SNNConfig::default();
        assert_eq!(config.dt, 1.0);
        assert_eq!(config.num_steps, 100);
        assert_eq!(config.neuron_model, NeuronModel::LIF);
    }
}
