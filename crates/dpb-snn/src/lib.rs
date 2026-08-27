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
pub mod distillation;
pub mod neuromorphic;
pub mod neuromodulation;

#[cfg(feature = "distributed")]
pub mod distributed;

#[cfg(feature = "gpu")]
pub mod gpu;

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
    SurrogateGradient, SurrogateType, BPTT, OTTT, SLTT,
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

// Re-export distillation types
pub use distillation::{
    // Core distillation
    TeacherStudentFramework, TeacherModel, StudentModel,
    DistillationConfig, DistillationMode, KnowledgeTransfer,
    // Loss functions
    DistillationLoss, KLDivergenceLoss, MSELoss, CosineSimLoss,
    HintLoss, AttentionTransferLoss, CombinedDistillationLoss, LossWeights,
    // SNN-specific distillation
    SpikePatternDistillation, SpikeRateDistillation,
    MembranePotentialDistillation, SynapticWeightTransfer,
    TemporalCreditAssignment, SpikeDistillationConfig,
    // Compression utilities
    ArchitectureSearch, LayerMerging, ChannelPruningGuided,
    QuantizationAwareDistillation, CompressionMetrics,
    CompressionConfig, SearchStrategy,
    // Self-distillation
    SelfDistillation, BornAgainNetworks, ProgressiveDistillation,
    SelfDistillationConfig, ProgressiveConfig,
};

// Re-export neuromodulation types
pub use neuromodulation::{
    // Core types
    Neuromodulator, NeuromodulatorType, ModulatorySystem,
    ModulatorConcentration, DiffusionModel, ReceptorBinding,
    Dopamine, Acetylcholine, Serotonin, Norepinephrine,
    // Dopamine system
    DopamineSystem, DopamineConfig, RewardPredictionError,
    DopamineMode, DopamineReceptorType, StrialRegion,
    // Acetylcholine system
    AcetylcholineSystem, AcetylcholineConfig,
    AChReceptorType, AttentionState, BasalForebrainRegion, ChAT,
    // Reward-modulated learning
    RewardModulatedSTDP, RewardSignal, EligibilityTrace,
    TemporalCreditAssignment as NeuromodTemporalCreditAssignment,
    IntrinsicMotivation, RewardShaping, ThreeFactorRule,
    // Gating
    GainModulation, GainModulationConfig, ModulationType,
    InputGating, OutputGating, ThresholdModulation,
    // Homeostasis
    HomeostaticPlasticity, HomeostaticConfig, FiringRateHomeostasis,
    SynapticScaling, IntrinsicPlasticity, Metaplasticity,
    SleepConsolidation,
    // Integration
    ModulatoryNetwork, ModulatoryNetworkConfig,
    ModulatorInteraction, SpatialScope, TemporalCoordination,
    StateDependent, BrainState,
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
