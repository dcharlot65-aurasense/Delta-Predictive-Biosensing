//! # DPB-SNN: Spiking Neural Network Implementation
//!
//! This crate provides a comprehensive implementation of Spiking Neural Networks (SNNs)
//! for the Delta-Predictive Biosensing Framework. It includes:
//!
//! - Spike tensor operations and representations
//! - Various layer types (linear, convolutional, recurrent, attention)
//! - Multiple SNN architectures
//! - Training algorithms (BPTT, OTTT, SLTT)
//! - ANN-to-SNN conversion utilities
//! - Output decoders for spike trains
//! - Model export capabilities
//! - Training analysis and convergence detection
//! - 44 ANN baseline architectures for fair SNN comparison

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
