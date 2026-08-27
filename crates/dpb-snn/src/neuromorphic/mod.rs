//! Neuromorphic hardware export
//!
//! Provides export capabilities for deploying SNNs to neuromorphic hardware platforms
//! including Intel Loihi, SpiNNaker, and BrainScaleS.
//!
//! ## Supported Hardware Platforms
//!
//! - **Intel Loihi/Loihi2**: NxSDK and Lava framework export
//! - **SpiNNaker/SpiNNaker2**: PyNN and sPyNNaker export
//! - **BrainScaleS/BrainScaleS2**: PyHMF configuration export
//!
//! ## Features
//!
//! - Hardware-specific neuron model mapping
//! - Synapse format conversion (weights, delays)
//! - Network partitioning for multi-core/chip systems
//! - Hardware constraint validation
//! - Quantization for fixed-point hardware
//! - Learning rule mapping (STDP, etc.)
//!
//! ## Example: Export to Loihi
//!
//! ```rust
//! use dpb_snn::neuromorphic::*;
//! # use dpb_snn::*;
//!
//! # fn example() -> SNNResult<()> {
//! // Configure Loihi export
//! let config = LoihiConfig {
//!     target: LoihiVersion::Loihi2,
//!     framework: LoihiFramework::Lava,
//!     num_chips: 1,
//!     weight_precision: 8,
//!     enable_learning: true,
//!     // Spread the default so a new field does not silently break this.
//!     ..LoihiConfig::default()
//! };
//!
//! let mut exporter = LoihiExporter::new(config);
//!
//! // Export network (placeholder - would take actual network)
//! // let result = exporter.export(&network)?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Export to SpiNNaker
//!
//! ```rust
//! use dpb_snn::neuromorphic::*;
//! # use dpb_snn::*;
//!
//! # fn example() -> SNNResult<()> {
//! let config = SpiNNakerConfig {
//!     version: SpiNNakerVersion::SpiNNaker2,
//!     board: "SpiNNaker2-48".to_string(),
//!     timestep_ms: 1.0,
//!     enable_live_io: true,
//!     ..SpiNNakerConfig::default()
//! };
//!
//! let mut exporter = SpiNNakerExporter::new(config);
//! # Ok(())
//! # }
//! ```

pub mod loihi;
pub mod spinnaker;
pub mod brainscales;
pub mod constraints;
pub mod partitioning;
pub mod quantization;

pub use loihi::{
    LoihiExporter, LoihiConfig, LoihiVersion, LoihiFramework,
    LoihiCompartment, LoihiSynapse, LoihiCore, LoihiNetwork,
};
pub use spinnaker::{
    SpiNNakerExporter, SpiNNakerConfig, SpiNNakerVersion,
    SpiNNakerPopulation, SpiNNakerProjection, SpiNNakerNetwork,
};
pub use brainscales::{
    BrainScaleSExporter, BrainScaleSConfig, BrainScaleSVersion,
    BrainScaleSNeuron, BrainScaleSNetwork,
};
pub use constraints::{
    HardwareConstraints, NeuronConstraints, SynapseConstraints,
    WeightBitDepth, DelayRange, SupportedNeuronModels,
};
pub use partitioning::{
    NetworkPartitioner, PartitionStrategy, CoreAssignment,
    PartitionResult, PartitionMetrics,
};
pub use quantization::{
    WeightQuantizer, QuantizationScheme, QuantizationResult,
    ThresholdQuantizer, TimeConstantQuantizer,
};

use crate::SNNResult;
use serde::{Deserialize, Serialize};

/// Neuromorphic hardware target platform
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NeuromorphicTarget {
    /// Intel Loihi (1st generation)
    Loihi,
    /// Intel Loihi 2 (2nd generation)
    Loihi2,
    /// SpiNNaker (1st generation)
    SpiNNaker,
    /// SpiNNaker 2 (2nd generation)
    SpiNNaker2,
    /// BrainScaleS (1st generation)
    BrainScaleS,
    /// BrainScaleS 2 (2nd generation)
    BrainScaleS2,
}

impl NeuromorphicTarget {
    /// Get hardware constraints for this target
    pub fn constraints(&self) -> HardwareConstraints {
        match self {
            Self::Loihi => HardwareConstraints::loihi_constraints(),
            Self::Loihi2 => HardwareConstraints::loihi2_constraints(),
            Self::SpiNNaker => HardwareConstraints::spinnaker_constraints(),
            Self::SpiNNaker2 => HardwareConstraints::spinnaker2_constraints(),
            Self::BrainScaleS => HardwareConstraints::brainscales_constraints(),
            Self::BrainScaleS2 => HardwareConstraints::brainscales2_constraints(),
        }
    }

    /// Check if target supports on-chip learning
    pub fn supports_learning(&self) -> bool {
        matches!(self, Self::Loihi | Self::Loihi2)
    }

    /// Check if target is digital (vs analog)
    pub fn is_digital(&self) -> bool {
        !matches!(self, Self::BrainScaleS | Self::BrainScaleS2)
    }
}

/// Base trait for neuromorphic hardware exporters
pub trait NeuromorphicExporter {
    /// Export network to target hardware format
    fn export(&mut self, network: &NetworkDescription) -> SNNResult<ExportResult>;

    /// Validate network against hardware constraints
    fn validate(&self, network: &NetworkDescription) -> SNNResult<Vec<String>>;

    /// Get hardware constraints
    fn constraints(&self) -> &HardwareConstraints;

    /// Optimize network for target hardware
    fn optimize(&mut self, network: &NetworkDescription) -> SNNResult<NetworkDescription>;
}

/// Network description for export (hardware-agnostic)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkDescription {
    /// Network name
    pub name: String,
    /// Neuron populations
    pub populations: Vec<Population>,
    /// Synaptic connections
    pub connections: Vec<Connection>,
    /// Network-level parameters
    pub parameters: NetworkParameters,
}

/// Neuron population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population {
    /// Population ID
    pub id: String,
    /// Number of neurons
    pub size: usize,
    /// Neuron model type
    pub neuron_model: String,
    /// Neuron parameters
    pub parameters: NeuronParameters,
}

/// Connection between populations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Connection {
    /// Connection ID
    pub id: String,
    /// Source population ID
    pub source: String,
    /// Target population ID
    pub target: String,
    /// Connection weights (flattened matrix)
    pub weights: Vec<f32>,
    /// Synaptic delays (ms)
    pub delays: Vec<f32>,
    /// Weight shape (source_size, target_size)
    pub shape: (usize, usize),
    /// Connection type
    pub connection_type: ConnectionType,
}

/// Connection topology type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionType {
    /// All-to-all connectivity
    AllToAll,
    /// One-to-one connectivity
    OneToOne,
    /// Fixed probability
    FixedProbability,
    /// Sparse connectivity
    Sparse,
}

/// Neuron model parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuronParameters {
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
    /// Additional model-specific parameters
    pub extra: std::collections::HashMap<String, f32>,
}

impl Default for NeuronParameters {
    fn default() -> Self {
        Self {
            tau_mem: 20.0,
            tau_syn: 5.0,
            v_threshold: 1.0,
            v_reset: 0.0,
            t_refrac: 2.0,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Network-level parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkParameters {
    /// Simulation timestep (ms)
    pub dt: f32,
    /// Total simulation time (ms)
    pub duration: f32,
    /// Enable plasticity
    pub enable_plasticity: bool,
    /// Additional parameters
    pub extra: std::collections::HashMap<String, String>,
}

impl Default for NetworkParameters {
    fn default() -> Self {
        Self {
            dt: 1.0,
            duration: 1000.0,
            enable_plasticity: false,
            extra: std::collections::HashMap::new(),
        }
    }
}

/// Export result containing hardware-specific artifacts
#[derive(Debug)]
pub struct ExportResult {
    /// Target hardware platform
    pub target: NeuromorphicTarget,
    /// Generated code/configuration files
    pub files: Vec<ExportFile>,
    /// Export metadata
    pub metadata: ExportMetadata,
    /// Warnings during export
    pub warnings: Vec<String>,
    /// Partition information
    pub partition_info: Option<PartitionResult>,
}

/// Exported file (code or configuration)
#[derive(Debug, Clone)]
pub struct ExportFile {
    /// File name
    pub name: String,
    /// File content
    pub content: Vec<u8>,
    /// File type/format
    pub file_type: String,
}

/// Export metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportMetadata {
    /// Export timestamp
    pub timestamp: String,
    /// Network statistics
    pub stats: NetworkStats,
    /// Hardware utilization
    pub utilization: HardwareUtilization,
}

/// Network statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkStats {
    /// Total neurons
    pub num_neurons: usize,
    /// Total synapses
    pub num_synapses: usize,
    /// Number of populations
    pub num_populations: usize,
    /// Number of connections
    pub num_connections: usize,
}

/// Hardware utilization metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareUtilization {
    /// Number of cores used
    pub cores_used: usize,
    /// Number of chips used
    pub chips_used: usize,
    /// Percentage of total neurons used
    pub neuron_utilization: f32,
    /// Percentage of total synapses used
    pub synapse_utilization: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromorphic_target_constraints() {
        let loihi = NeuromorphicTarget::Loihi;
        let constraints = loihi.constraints();

        assert!(constraints.neuron.max_neurons_per_core > 0);
        assert!(loihi.supports_learning());
        assert!(loihi.is_digital());
    }

    #[test]
    fn test_brainscales_is_analog() {
        let bs = NeuromorphicTarget::BrainScaleS;
        assert!(!bs.is_digital());
        assert!(!bs.supports_learning());
    }

    #[test]
    fn test_network_description_creation() {
        let net = NetworkDescription {
            name: "test_network".to_string(),
            populations: vec![],
            connections: vec![],
            parameters: NetworkParameters::default(),
        };

        assert_eq!(net.name, "test_network");
        assert_eq!(net.parameters.dt, 1.0);
    }

    #[test]
    fn test_neuron_parameters_default() {
        let params = NeuronParameters::default();
        assert_eq!(params.tau_mem, 20.0);
        assert_eq!(params.v_threshold, 1.0);
        assert!(params.extra.is_empty());
    }

    #[test]
    fn test_connection_type() {
        let conn_type = ConnectionType::AllToAll;
        assert_eq!(conn_type, ConnectionType::AllToAll);
        assert_ne!(conn_type, ConnectionType::Sparse);
    }
}
