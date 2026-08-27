//! # dpb-snn - Spiking Neural Networks
//!
//! Spiking Neural Network architectures, training, and deployment.
//!
//! ## Features
//!
//! - **Architectures**: Feedforward, convolutional, recurrent SNNs
//! - **Training**: BPTT, surrogate gradients, ANN-to-SNN conversion
//! - **Neuromodulation**: Dopamine, acetylcholine, reward-modulated learning, homeostasis
//! - **Distillation**: Knowledge distillation for SNN compression
//! - **Calibration**: Temperature scaling, isotonic regression, uncertainty quantification
//! - **Explainability**: Spike importance, attention maps, feature attribution
//! - **Fusion**: Multi-modal sensor fusion architectures
//! - **Baselines**: 44 ANN architectures for comparison
//! - **Export**: ONNX format, weight serialization, neuromorphic hardware
//!
//! ## Quick Start: Building an SNN
//!
//! ```rust
//! use dpb_snn::*;
//!
//! # fn example() -> SNNResult<()> {
//! // Build a feedforward SNN
//! let config = SNNConfig::default();
//! let mut network = FeedforwardSNN::new(
//!     vec![128, 64, 10],  // layer sizes
//!     config,
//!     true,               // use bias
//! )?;
//!
//! // Configure training. `BPTT` selects a surrogate gradient by kind; the
//! // second argument caps how many timesteps to unroll (None = all of them).
//! let mut trainer = BPTT::new(SurrogateType::FastSigmoid, None);
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Model Calibration
//!
//! ```rust
//! use dpb_snn::calibration::*;
//!
//! # fn example() {
//! let logits = vec![vec![1.0, 2.0, 3.0], vec![0.5, 0.2, 3.5]];
//! let labels = vec![2usize, 2];
//!
//! // Calibrate model confidence scores
//! let mut calibrator = TemperatureScaling::new();
//! calibrator.fit(&logits, &labels).expect("fit");
//! let calibrated = calibrator.calibrate_batch(&logits);
//!
//! // `expected_calibration_error` is binary: it takes the confidence assigned
//! // to each prediction and whether that prediction was correct.
//! let (confidences, correct): (Vec<f64>, Vec<bool>) = calibrated
//!     .iter()
//!     .zip(labels.iter())
//!     .map(|(probs, &label)| {
//!         let (predicted, confidence) = probs
//!             .iter()
//!             .enumerate()
//!             .max_by(|(_, a), (_, b)| a.total_cmp(b))
//!             .map(|(i, &p)| (i, p))
//!             .unwrap();
//!         (confidence, predicted == label)
//!     })
//!     .unzip();
//!
//! let ece = expected_calibration_error(&confidences, &correct, 10);
//! println!("Expected Calibration Error: {:.4}", ece);
//! # }
//! ```
//!
//! ## Example: Explainability
//!
//! ```rust
//! use dpb_snn::explain::*;
//!
//! # fn example() {
//! // Spike TIMES per neuron, the output gradients, and the layer weights.
//! let spike_times = vec![
//!     vec![2.0, 11.5, 30.0],  // neuron 0
//!     vec![5.0, 18.0],        // neuron 1
//! ];
//! let output_gradients = vec![0.8, -0.3];
//! let layer_weights = vec![vec![0.5, 0.2], vec![-0.1, 0.7]];
//!
//! // Compute spike importance for interpretability
//! let importance = compute_spike_importance(
//!     &spike_times,
//!     &output_gradients,
//!     &layer_weights,
//! );
//!
//! // Aggregate to neuron-level importance
//! let neuron_importance = aggregate_to_neurons(&importance);
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
//!     modalities: vec![Modality::Contact, Modality::Pose, Modality::Voice],
//!     hidden_size: 128,
//!     output_size: 5,
//!     ..FusionConfig::default()
//! };
//!
//! let network = CrossModalAttentionSNN::new(config)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Knowledge Distillation
//!
//! ```rust
//! use dpb_snn::distillation::*;
//!
//! # fn example() -> dpb_snn::SNNResult<()> {
//! // Configure distillation for model compression
//! let config = DistillationConfig {
//!     temperature: 4.0,
//!     alpha: 0.7,  // Weight for soft targets
//!     beta: 0.3,   // Weight for hard targets
//!     mode: DistillationMode::ResponseBased,
//!     feature_layers: vec![],
//!     scale_temperature: true,
//! };
//!
//! // Find optimal student architecture
//! let teacher_layers = vec![128, 256, 128, 64, 10];
//! let search = ArchitectureSearch::new(
//!     teacher_layers,
//!     SearchStrategy::Hybrid,
//!     0.3  // 30% compression
//! )?;
//!
//! let student_layers = search.get_best_candidate()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Neuromodulation
//!
//! ```rust
//! use dpb_snn::neuromodulation::*;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create integrated modulatory network
//! let config = ModulatoryNetworkConfig::default();
//! let mut network = ModulatoryNetwork::new(config);
//!
//! // Update with reward signal (dopamine)
//! network.update_with_reward(1.0, 1.0, 0.5)?;
//!
//! // Update with attention signal (acetylcholine)
//! network.update_with_attention(1.0, 0.8)?;
//!
//! // Get learning rate modulation
//! let lr_modulation = network.get_learning_rate_modulation();
//!
//! // Apply modulatory effects to plasticity
//! let base_weight_change = 0.01;
//! let modulated_change = network.modulate_weight_change(base_weight_change);
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
//! | [`neuromodulation`] | Neuromodulation systems (dopamine, acetylcholine, reward learning) |
//! | [`optimization`] | Network pruning and compression |
//! | [`distillation`] | Knowledge distillation for model compression |
//! | [`export`] | Model export utilities |
//! | [`neuromorphic`] | Neuromorphic hardware export (Loihi, SpiNNaker, BrainScaleS) |
//! | [`gpu`] | GPU acceleration (CUDA, Metal) - requires `gpu` feature |
//! | [`distributed`] | Distributed training infrastructure - requires `distributed` feature |

pub mod analysis;
pub mod architectures;
pub mod baselines;
pub mod calibration;
pub mod conversion;
pub mod decoders;
pub mod distillation;
pub mod explain;
pub mod export;
pub mod fusion;
pub mod layers;
pub mod learning;
pub mod neuromodulation;
pub mod neuromorphic;
pub mod optimization;
pub mod tensor;
pub mod training;

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "gpu")]
pub mod gpu;

// Re-export commonly used types
pub use analysis::{
    AccuracyPlateauDetector, AnalysisReport, BatchSizeAnalyzer, ConvergenceAnalyzer,
    ConvergenceRateAnalyzer, DivergenceDetector, EarlyStoppingAnalyzer, EpochEfficiencyAnalyzer,
    ExplodingGradientDetector, GeneralizationGapAnalyzer, GradientFlowAnalyzer,
    GradientNormTracker, HyperparameterSensitivityAnalyzer, LearningCurveSmoothed,
    LearningRateAnalyzer, LossPlateauDetector, MethodComparisonAnalyzer, OscillationDetector,
    OverfittingDetector, SaturatedNeuronDetector, SilentNeuronDetector, SparsityTracker,
    SpikeRateTracker, SurrogateGradientAnalyzer, TemporalDynamicsAnalyzer, TrainingMetrics,
    VanishingGradientDetector, WeightDistributionTracker, WeightMagnitudeTracker,
    WeightSparsityTracker, WeightUpdateTracker,
};
pub use architectures::{
    ConvolutionalSNN, FeedforwardSNN, RecurrentSNN, SpikingGCN, SpikingTransformer,
};
pub use baselines::{
    ANNBaseline,
    // Conversion utilities
    ANNToSNNConverter as BaselineConverter,
    AttentionFusion,
    AttentionLSTM,
    Autoformer,
    BiGRU,
    BiLSTM,
    CNN1DDilated,
    CNN1DLarge,
    CNN1DMedium,
    CNN1DResidual,
    // CNN architectures
    CNN1DSmall,
    CNN2DLeNet,
    CNN2DMobileNet,
    CNN2DResNet,
    CNN2DVGG,
    ConversionConfig as BaselineConversionConfig,
    DeepGait,
    // Specialized architectures
    ECGNet,
    GRU,
    GraphNN,
    HybridCNNRNN,
    IndRNN,
    Informer,
    LSTM,
    LinearTransformer,
    // MLP architectures
    MLP2Layer,
    MLP3Layer,
    MLP4Layer,
    MLPBatchNorm,
    MLPDeep,
    MLPDropout,
    MLPResidual,
    MLPWideSingle,
    MultimodalFusion,
    PeepholeLSTM,
    Performer,
    // RNN architectures
    SimpleRNN,
    StackedGRU,
    StackedLSTM,
    TCN,
    Tensor as BaselineTensor,
    ThresholdBalancingStrategy,
    // Transformer architectures
    TransformerEncoder,
    TransformerLarge,
    TransformerMedium,
    TransformerSmall,
    TremorNet,
    VoiceNet,
    WeightNormalizationMethod,
    convert_model_to_snn,
};
pub use conversion::{ANNToSNNConverter, ThresholdBalancing, WeightNormalization};
pub use decoders::{
    ActivityDecoder,
    AdaptiveRateDecoder,
    BergBalanceDecoder,
    // Classification decoders
    BinaryClassDecoder,
    BurstDecoder,
    // Base decoder trait
    Decoder,
    DyskinesiasDecoder,
    EmotionDecoder,
    ExponentialRateDecoder,
    FatigueDecoder,
    FirstSpikeDecoder,
    GaitPhaseDecoder,
    GaitScoreDecoder,
    GaitVelocityDecoder,
    HRVDecoder,
    // Regression decoders
    HeartRateDecoder,
    HoehnYahrDecoder,
    ISIDecoder,
    LastSpikeDecoder,
    LatencyDecoder,
    MaxSpikeDecoder,
    MedicationStateDecoder,
    MoCADecoder,
    MultiClassDecoder,
    NormalizedRateDecoder,
    PDQ39Decoder,
    PhaseDecoder,
    PopulationDecoder,
    PupilDiameterDecoder,
    RankOrderDecoder,
    ReactionTimeDecoder,
    SEADLDecoder,
    SleepStageDecoder,
    SpeechRateDecoder,
    // Rate-based decoders
    SpikeRateDecoder,
    StrideTimeDecoder,
    TUGDecoder,
    TappingFrequencyDecoder,
    // Temporal decoders
    TemporalPatternDecoder,
    TremorAmplitudeDecoder,
    TremorFrequencyDecoder,
    TremorSeverityDecoder,
    TremorTypeDecoder,
    UPDRSBradykinesiaDecoder,
    // Clinical score decoders
    UPDRSDecoder,
    UPDRSGaitDecoder,
    UPDRSMotorDecoder,
    UPDRSRigidityDecoder,
    UPDRSTremorDecoder,
    VoiceHDDecoder,
    WeightedRateDecoder,
    WindowedRateDecoder,
};
pub use fusion::{
    ChangeDirection, ChangeMetrics, CognitiveMotorFusion, CognitiveMotorFusionConfig,
    CognitiveMotorFusionSNN, CognitiveProfile, CrossModalAttentionSNN, DissociationPattern,
    DissociationType, DomainZScores, EarlyFusionSNN, FusionConfig, FusionNetwork, GatedFusionSNN,
    HierarchicalFusionSNN, IntegratedAssessment, LateFusionSNN, Modality, MotorProfile,
    NeuroPlaySNN, RiskCategory, TemporalAlignmentSNN,
};
pub use layers::{
    SpikingAttention, SpikingConv1d, SpikingConv2d, SpikingLSTM, SpikingLinear, SpikingMaxPool2d,
    SpikingRNN, SpikingSumPool2d,
};
pub use tensor::{SpikeRepresentation, SpikeTensor};
pub use training::{
    AdamOptimizer, BPTT, OTTT, SGDOptimizer, SLTT, SpikeCountLoss, SpikeTimingLoss,
    SpikingCrossEntropy, SurrogateGradient, SurrogateType,
};

// Re-export learning types
pub use learning::{
    BCMRule, CovarianceRule, HebbianLayer, HebbianRule, OjasRule, STDP, SynapticTrace,
};

// Re-export optimization types
pub use optimization::{
    NetworkPruner, PruningMask, PruningSchedule, PruningStats, PruningStrategy,
};

// Re-export calibration types
pub use calibration::{
    ConfidenceInterval, EnsembleUncertainty, IsotonicCalibration, MCDropout, PlattScaling,
    ReliabilityBin, TemperatureScaling, UncertaintyEstimator, bootstrap_ci, brier_score,
    expected_calibration_error, maximum_calibration_error, negative_log_likelihood,
    reliability_diagram,
};

// Re-export explainability types
pub use explain::{
    AttentionMap, ExplanationVisualizer, FeatureAttribution, GradientAttribution, HeatmapData,
    IntegratedGradients, LayerImportance, NeuronImportance, SpatialAttention, SpikeImportance,
    SpikeSHAP, TemporalAttention, aggregate_to_neurons, compute_importance_by_perturbation,
    compute_spike_importance, export_explanation_json,
};

// Re-export distillation types
pub use distillation::{
    // Compression utilities
    ArchitectureSearch,
    AttentionTransferLoss,
    BornAgainNetworks,
    ChannelPruningGuided,
    CombinedDistillationLoss,
    CompressionConfig,
    CompressionMetrics,
    CosineSimLoss,
    DistillationConfig,
    // Loss functions
    DistillationLoss,
    DistillationMode,
    HintLoss,
    KLDivergenceLoss,
    KnowledgeTransfer,
    LayerMerging,
    LossWeights,
    MSELoss,
    MembranePotentialDistillation,
    ProgressiveConfig,
    ProgressiveDistillation,
    QuantizationAwareDistillation,
    SearchStrategy,
    // Self-distillation
    SelfDistillation,
    SelfDistillationConfig,
    SpikeDistillationConfig,
    // SNN-specific distillation
    SpikePatternDistillation,
    SpikeRateDistillation,
    StudentModel,
    SynapticWeightTransfer,
    TeacherModel,
    // Core distillation
    TeacherStudentFramework,
    TemporalCreditAssignment,
};

// Re-export neuromodulation types
pub use neuromodulation::{
    AChReceptorType,
    Acetylcholine,
    AcetylcholineConfig,
    // Acetylcholine system
    AcetylcholineSystem,
    AttentionState,
    BasalForebrainRegion,
    BrainState,
    ChAT,
    DiffusionModel,
    Dopamine,
    DopamineConfig,
    DopamineMode,
    DopamineReceptorType,
    // Dopamine system
    DopamineSystem,
    EligibilityTrace,
    FiringRateHomeostasis,
    // Gating
    GainModulation,
    GainModulationConfig,
    HomeostaticConfig,
    // Homeostasis
    HomeostaticPlasticity,
    InputGating,
    IntrinsicMotivation,
    IntrinsicPlasticity,
    Metaplasticity,
    ModulationType,
    ModulatorConcentration,
    ModulatorInteraction,
    // Integration
    ModulatoryNetwork,
    ModulatoryNetworkConfig,
    ModulatorySystem,
    // Core types
    Neuromodulator,
    NeuromodulatorType,
    Norepinephrine,
    OutputGating,
    ReceptorBinding,
    // Reward-modulated learning
    RewardModulatedSTDP,
    RewardPredictionError,
    RewardShaping,
    RewardSignal,
    Serotonin,
    SleepConsolidation,
    SpatialScope,
    StateDependent,
    StrialRegion,
    SynapticScaling,
    TemporalCoordination,
    TemporalCreditAssignment as NeuromodTemporalCreditAssignment,
    ThreeFactorRule,
    ThresholdModulation,
};

use dpb_core::error::DpbError;
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

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
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
    /// Enable GPU acceleration (requires `gpu` feature)
    #[cfg(feature = "gpu")]
    pub use_gpu: bool,
}

impl Default for SNNConfig {
    fn default() -> Self {
        Self {
            dt: 1.0,
            num_steps: 100,
            neuron_model: NeuronModel::LIF,
            neuron_params: NeuronParams::default(),
            #[cfg(feature = "gpu")]
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
