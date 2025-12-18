//! Intel Loihi neuromorphic hardware export
//!
//! Provides export capabilities for Intel Loihi and Loihi 2 chips using
//! NxSDK and Lava frameworks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::{
    NeuromorphicExporter, NeuromorphicTarget, NetworkDescription, ExportResult,
    ExportFile, ExportMetadata, NetworkStats, HardwareUtilization,
};
use super::constraints::HardwareConstraints;
use super::partitioning::{NetworkPartitioner, PartitionStrategy, NetworkGraph, Edge};
use super::quantization::{WeightQuantizer, QuantizationScheme};
use crate::{SNNResult, SNNError};

/// Loihi exporter
pub struct LoihiExporter {
    config: LoihiConfig,
    constraints: HardwareConstraints,
}

impl LoihiExporter {
    /// Create a new Loihi exporter
    pub fn new(config: LoihiConfig) -> Self {
        let constraints = match config.target {
            LoihiVersion::Loihi => HardwareConstraints::loihi_constraints(),
            LoihiVersion::Loihi2 => HardwareConstraints::loihi2_constraints(),
        };

        Self {
            config,
            constraints,
        }
    }

    /// Generate NxSDK Python code
    fn generate_nxsdk_code(&self, network: &LoihiNetwork) -> String {
        let mut code = String::new();

        code.push_str("# Generated NxSDK code for Loihi\n");
        code.push_str("import nxsdk.api.n2a as nx\n\n");
        code.push_str("# Create network\n");
        code.push_str("net = nx.NxNet()\n\n");

        // Create compartment groups (neuron populations)
        code.push_str("# Create compartments\n");
        for (i, comp_group) in network.compartment_groups.iter().enumerate() {
            code.push_str(&format!(
                "compartments_{} = net.createCompartmentGroup(size={})\n",
                i, comp_group.size
            ));

            // Configure compartment parameters
            code.push_str(&format!(
                "compartments_{}.configure(\n",
                i
            ));
            code.push_str(&format!("    vThMant={},\n", comp_group.v_threshold_mant));
            code.push_str(&format!("    vDecay={},\n", comp_group.decay_v));
            code.push_str(&format!("    uDecay={},\n", comp_group.decay_u));
            code.push_str(&format!("    refractDelay={},\n", comp_group.refractory_delay));
            code.push_str(")\n\n");
        }

        // Create connections
        code.push_str("# Create connections\n");
        for (i, conn) in network.connections.iter().enumerate() {
            code.push_str(&format!(
                "conn_{} = compartments_{}.connect(\n",
                i, conn.source_group
            ));
            code.push_str(&format!("    compartments_{},\n", conn.target_group));
            code.push_str("    prototype=nx.ConnectionPrototype(\n");
            code.push_str(&format!("        weight={},\n", conn.weight));
            code.push_str(&format!("        delay={},\n", conn.delay));
            code.push_str("    ),\n");
            code.push_str(&format!("    connectionMask=nx.{},\n", conn.connection_type));
            code.push_str(")\n\n");
        }

        // Configure learning if enabled
        if self.config.enable_learning {
            code.push_str("# Configure learning rules\n");
            for (i, rule) in network.learning_rules.iter().enumerate() {
                code.push_str(&format!(
                    "learning_rule_{} = net.createLearningRule(\n",
                    i
                ));
                code.push_str(&format!("    dw='{}',\n", rule.rule_type));
                code.push_str(&format!("    x0={},\n", rule.x0));
                code.push_str(&format!("    y0={},\n", rule.y0));
                code.push_str(&format!("    tauX={},\n", rule.tau_x));
                code.push_str(&format!("    tauY={},\n", rule.tau_y));
                code.push_str(")\n");
                code.push_str(&format!("conn_{}.enableLearning(learning_rule_{})\n\n", i, i));
            }
        }

        // Compile and run
        code.push_str("# Compile network\n");
        code.push_str("compiler = nx.N2Compiler()\n");
        code.push_str("board = compiler.compile(net)\n\n");
        code.push_str("# Run network\n");
        code.push_str(&format!("board.run({})\n", self.config.num_steps));
        code.push_str("board.disconnect()\n");

        code
    }

    /// Generate Lava framework code
    fn generate_lava_code(&self, network: &LoihiNetwork) -> String {
        let mut code = String::new();

        code.push_str("# Generated Lava code for Loihi 2\n");
        code.push_str("from lava.magma.core.run_configs import Loihi2SimCfg\n");
        code.push_str("from lava.magma.core.run_conditions import RunSteps\n");
        code.push_str("from lava.proc.lif.process import LIF\n");
        code.push_str("from lava.proc.dense.process import Dense\n\n");

        // Create LIF neurons
        code.push_str("# Create LIF neuron populations\n");
        for (i, comp_group) in network.compartment_groups.iter().enumerate() {
            code.push_str(&format!(
                "neurons_{} = LIF(\n",
                i
            ));
            code.push_str(&format!("    shape=({}),\n", comp_group.size));
            code.push_str(&format!("    vth={},\n", comp_group.v_threshold_mant));
            code.push_str(&format!("    dv={},\n", comp_group.decay_v));
            code.push_str(&format!("    du={},\n", comp_group.decay_u));
            code.push_str(")\n\n");
        }

        // Create connections (Dense processes)
        code.push_str("# Create synaptic connections\n");
        for (i, conn) in network.connections.iter().enumerate() {
            code.push_str(&format!(
                "synapse_{} = Dense(\n",
                i
            ));
            code.push_str(&format!("    weights=np.array([{}]),\n", conn.weight));
            code.push_str(")\n");
            code.push_str(&format!(
                "neurons_{}.s_out.connect(synapse_{}.s_in)\n",
                conn.source_group, i
            ));
            code.push_str(&format!(
                "synapse_{}.a_out.connect(neurons_{}.a_in)\n\n",
                i, conn.target_group
            ));
        }

        // Run configuration
        code.push_str("# Configure and run\n");
        code.push_str("run_cfg = Loihi2SimCfg()\n");
        code.push_str(&format!("neurons_0.run(condition=RunSteps(num_steps={}), run_cfg=run_cfg)\n", self.config.num_steps));
        code.push_str("neurons_0.stop()\n");

        code
    }

    /// Map network to Loihi cores
    fn map_to_cores(&self, network: &NetworkDescription) -> SNNResult<LoihiNetwork> {
        // Create network graph for partitioning
        let num_neurons: usize = network.populations.iter().map(|p| p.size).sum();
        let mut edges = Vec::new();

        for conn in &network.connections {
            // Create edges from connection matrix
            edges.push(Edge {
                source: 0, // Simplified
                target: 1,
                weight: conn.weights.get(0).copied().unwrap_or(0.0),
            });
        }

        let graph = NetworkGraph::new(num_neurons, edges);

        // Partition network
        let partitioner = NetworkPartitioner::new(
            self.constraints.clone(),
            PartitionStrategy::LoadBalanced,
        );
        let partition = partitioner.partition(&graph)?;

        // Create Loihi-specific structures
        let mut compartment_groups = Vec::new();
        let mut connections = Vec::new();
        let mut cores = Vec::new();

        for core_id in 0..partition.num_cores {
            let neurons_on_core = partition.get_neurons_on_core(core_id);

            // Create compartment group for this core
            compartment_groups.push(LoihiCompartmentGroup {
                size: neurons_on_core.len(),
                v_threshold_mant: 1024, // Default
                decay_v: 128,
                decay_u: 128,
                refractory_delay: 2,
                compartment_type: "LIF".to_string(),
            });

            cores.push(LoihiCore {
                core_id,
                compartments: neurons_on_core.len(),
                synapses: 0, // Updated later
            });
        }

        // Create connections between compartment groups
        for conn in &network.connections {
            connections.push(LoihiConnection {
                source_group: 0, // Simplified
                target_group: 1,
                weight: conn.weights.get(0).copied().unwrap_or(0.0) as i32,
                delay: conn.delays.get(0).copied().unwrap_or(1.0) as usize,
                connection_type: "ONE_TO_ONE".to_string(),
            });
        }

        // Quantize weights
        let mut quantizer = WeightQuantizer::new(
            QuantizationScheme::Symmetric,
            self.constraints.weight_precision,
        );

        for conn in &mut connections {
            let weights = vec![conn.weight as f32];
            quantizer.calibrate(&weights);
            let result = quantizer.quantize(&weights);
            conn.weight = result.quantized_weights[0];
        }

        Ok(LoihiNetwork {
            compartment_groups,
            connections,
            learning_rules: Vec::new(),
            cores,
        })
    }
}

impl NeuromorphicExporter for LoihiExporter {
    fn export(&mut self, network: &NetworkDescription) -> SNNResult<ExportResult> {
        // Validate network
        let warnings = self.validate(network)?;

        // Map network to Loihi cores
        let loihi_net = self.map_to_cores(network)?;

        // Generate code based on framework
        let code = match self.config.framework {
            LoihiFramework::NxSDK => self.generate_nxsdk_code(&loihi_net),
            LoihiFramework::Lava => self.generate_lava_code(&loihi_net),
        };

        let file_ext = match self.config.framework {
            LoihiFramework::NxSDK => "py",
            LoihiFramework::Lava => "py",
        };

        let files = vec![ExportFile {
            name: format!("loihi_network.{}", file_ext),
            content: code.into_bytes(),
            file_type: "python".to_string(),
        }];

        // Calculate statistics
        let num_neurons: usize = network.populations.iter().map(|p| p.size).sum();
        let num_synapses: usize = network.connections.iter().map(|c| c.weights.len()).sum();

        let stats = NetworkStats {
            num_neurons,
            num_synapses,
            num_populations: network.populations.len(),
            num_connections: network.connections.len(),
        };

        let utilization = HardwareUtilization {
            cores_used: loihi_net.cores.len(),
            chips_used: (loihi_net.cores.len() + 127) / 128, // 128 cores per chip
            neuron_utilization: (num_neurons as f32
                / self.constraints.neuron.max_neurons_total as f32)
                * 100.0,
            synapse_utilization: (num_synapses as f32
                / (num_neurons * self.constraints.synapse.max_fanin) as f32)
                * 100.0,
        };

        let metadata = ExportMetadata {
            timestamp: chrono::Utc::now().to_rfc3339(),
            stats,
            utilization,
        };

        let target = match self.config.target {
            LoihiVersion::Loihi => NeuromorphicTarget::Loihi,
            LoihiVersion::Loihi2 => NeuromorphicTarget::Loihi2,
        };

        Ok(ExportResult {
            target,
            files,
            metadata,
            warnings,
            partition_info: None,
        })
    }

    fn validate(&self, network: &NetworkDescription) -> SNNResult<Vec<String>> {
        let mut warnings = Vec::new();

        let num_neurons: usize = network.populations.iter().map(|p| p.size).sum();
        let num_synapses: usize = network.connections.iter().map(|c| c.weights.len()).sum();

        // Check neuron count
        if num_neurons > self.constraints.neuron.max_neurons_total {
            return Err(SNNError::Export(format!(
                "Network has {} neurons, exceeds Loihi limit of {}",
                num_neurons, self.constraints.neuron.max_neurons_total
            )));
        }

        // Check if multi-chip needed
        let num_cores = self.constraints.estimate_cores(num_neurons);
        if num_cores > 128 {
            warnings.push(format!(
                "Network requires {} cores, will use multiple chips",
                num_cores
            ));
        }

        // Check synaptic constraints
        for conn in &network.connections {
            let max_fanin = conn.shape.0;
            if max_fanin > self.constraints.synapse.max_fanin {
                return Err(SNNError::Export(format!(
                    "Connection has fanin of {}, exceeds Loihi limit of {}",
                    max_fanin, self.constraints.synapse.max_fanin
                )));
            }
        }

        // Check neuron models
        for pop in &network.populations {
            if !self.constraints.supports_model(&pop.neuron_model) {
                warnings.push(format!(
                    "Neuron model '{}' not directly supported, will map to closest model",
                    pop.neuron_model
                ));
            }
        }

        Ok(warnings)
    }

    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn optimize(&mut self, network: &NetworkDescription) -> SNNResult<NetworkDescription> {
        // Return unmodified for now - would implement optimization passes
        Ok(network.clone())
    }
}

/// Loihi configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiConfig {
    /// Loihi version
    pub target: LoihiVersion,
    /// Framework to use
    pub framework: LoihiFramework,
    /// Number of chips available
    pub num_chips: usize,
    /// Weight precision (bits)
    pub weight_precision: u8,
    /// Enable on-chip learning
    pub enable_learning: bool,
    /// Number of timesteps
    pub num_steps: usize,
}

impl Default for LoihiConfig {
    fn default() -> Self {
        Self {
            target: LoihiVersion::Loihi2,
            framework: LoihiFramework::Lava,
            num_chips: 1,
            weight_precision: 8,
            enable_learning: false,
            num_steps: 100,
        }
    }
}

/// Loihi version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoihiVersion {
    Loihi,
    Loihi2,
}

/// Loihi framework
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LoihiFramework {
    /// NxSDK (Loihi 1 primary framework)
    NxSDK,
    /// Lava (Loihi 2 framework)
    Lava,
}

/// Loihi compartment (neuron)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiCompartment {
    pub id: usize,
    pub v_threshold_mant: i32,
    pub decay_v: u32,
    pub decay_u: u32,
    pub refractory_delay: u32,
}

/// Loihi compartment group
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiCompartmentGroup {
    pub size: usize,
    pub v_threshold_mant: i32,
    pub decay_v: u32,
    pub decay_u: u32,
    pub refractory_delay: u32,
    pub compartment_type: String,
}

/// Loihi synapse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiSynapse {
    pub source: usize,
    pub target: usize,
    pub weight: i32,
    pub delay: usize,
}

/// Loihi connection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiConnection {
    pub source_group: usize,
    pub target_group: usize,
    pub weight: i32,
    pub delay: usize,
    pub connection_type: String,
}

/// Loihi learning rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiLearningRule {
    pub rule_type: String, // "dw = x * y" or similar
    pub x0: i32,
    pub y0: i32,
    pub tau_x: u32,
    pub tau_y: u32,
}

/// Loihi core
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoihiCore {
    pub core_id: usize,
    pub compartments: usize,
    pub synapses: usize,
}

/// Complete Loihi network
#[derive(Debug, Clone)]
pub struct LoihiNetwork {
    pub compartment_groups: Vec<LoihiCompartmentGroup>,
    pub connections: Vec<LoihiConnection>,
    pub learning_rules: Vec<LoihiLearningRule>,
    pub cores: Vec<LoihiCore>,
}

// Add chrono for timestamps
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> DateTime {
            DateTime
        }
    }
    pub struct DateTime;
    impl DateTime {
        pub fn to_rfc3339(&self) -> String {
            "2025-01-01T00:00:00Z".to_string()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromorphic::{Population, Connection, NeuronParameters, NetworkParameters, ConnectionType};

    fn create_test_network() -> NetworkDescription {
        NetworkDescription {
            name: "test_network".to_string(),
            populations: vec![
                Population {
                    id: "pop1".to_string(),
                    size: 100,
                    neuron_model: "LIF".to_string(),
                    parameters: NeuronParameters::default(),
                },
                Population {
                    id: "pop2".to_string(),
                    size: 50,
                    neuron_model: "LIF".to_string(),
                    parameters: NeuronParameters::default(),
                },
            ],
            connections: vec![Connection {
                id: "conn1".to_string(),
                source: "pop1".to_string(),
                target: "pop2".to_string(),
                weights: vec![0.5; 5000],
                delays: vec![1.0; 5000],
                shape: (100, 50),
                connection_type: ConnectionType::AllToAll,
            }],
            parameters: NetworkParameters::default(),
        }
    }

    #[test]
    fn test_loihi_exporter_creation() {
        let config = LoihiConfig::default();
        let exporter = LoihiExporter::new(config);

        assert_eq!(exporter.config.target, LoihiVersion::Loihi2);
        assert_eq!(exporter.config.framework, LoihiFramework::Lava);
    }

    #[test]
    fn test_validate_network() {
        let config = LoihiConfig::default();
        let exporter = LoihiExporter::new(config);
        let network = create_test_network();

        let result = exporter.validate(&network);
        assert!(result.is_ok());
    }

    #[test]
    fn test_export_nxsdk() {
        let config = LoihiConfig {
            framework: LoihiFramework::NxSDK,
            ..Default::default()
        };
        let mut exporter = LoihiExporter::new(config);
        let network = create_test_network();

        let result = exporter.export(&network);
        assert!(result.is_ok());

        let export = result.unwrap();
        assert_eq!(export.files.len(), 1);
        assert!(export.files[0].name.ends_with(".py"));
    }

    #[test]
    fn test_export_lava() {
        let config = LoihiConfig {
            framework: LoihiFramework::Lava,
            target: LoihiVersion::Loihi2,
            ..Default::default()
        };
        let mut exporter = LoihiExporter::new(config);
        let network = create_test_network();

        let result = exporter.export(&network);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_nxsdk_code() {
        let config = LoihiConfig {
            framework: LoihiFramework::NxSDK,
            ..Default::default()
        };
        let exporter = LoihiExporter::new(config);

        let loihi_net = LoihiNetwork {
            compartment_groups: vec![LoihiCompartmentGroup {
                size: 100,
                v_threshold_mant: 1024,
                decay_v: 128,
                decay_u: 128,
                refractory_delay: 2,
                compartment_type: "LIF".to_string(),
            }],
            connections: vec![],
            learning_rules: vec![],
            cores: vec![],
        };

        let code = exporter.generate_nxsdk_code(&loihi_net);
        assert!(code.contains("import nxsdk"));
        assert!(code.contains("createCompartmentGroup"));
    }

    #[test]
    fn test_constraints() {
        let config = LoihiConfig {
            target: LoihiVersion::Loihi,
            ..Default::default()
        };
        let exporter = LoihiExporter::new(config);

        let constraints = exporter.constraints();
        assert_eq!(constraints.neuron.max_neurons_per_core, 1024);
    }

    #[test]
    fn test_loihi2_has_more_capacity() {
        let config1 = LoihiConfig {
            target: LoihiVersion::Loihi,
            ..Default::default()
        };
        let config2 = LoihiConfig {
            target: LoihiVersion::Loihi2,
            ..Default::default()
        };

        let exporter1 = LoihiExporter::new(config1);
        let exporter2 = LoihiExporter::new(config2);

        assert!(
            exporter2.constraints().neuron.max_neurons_total
                > exporter1.constraints().neuron.max_neurons_total
        );
    }
}
