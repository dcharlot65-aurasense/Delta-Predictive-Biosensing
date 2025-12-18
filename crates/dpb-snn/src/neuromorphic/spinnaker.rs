//! SpiNNaker neuromorphic hardware export
//!
//! Provides export capabilities for SpiNNaker and SpiNNaker 2 systems using
//! PyNN and sPyNNaker frameworks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use super::{
    NeuromorphicExporter, NeuromorphicTarget, NetworkDescription, ExportResult,
    ExportFile, ExportMetadata, NetworkStats, HardwareUtilization,
};
use super::constraints::HardwareConstraints;
use super::partitioning::{NetworkPartitioner, PartitionStrategy, NetworkGraph, Edge};
use crate::{SNNResult, SNNError};

/// SpiNNaker exporter
pub struct SpiNNakerExporter {
    config: SpiNNakerConfig,
    constraints: HardwareConstraints,
}

impl SpiNNakerExporter {
    /// Create a new SpiNNaker exporter
    pub fn new(config: SpiNNakerConfig) -> Self {
        let constraints = match config.version {
            SpiNNakerVersion::SpiNNaker => HardwareConstraints::spinnaker_constraints(),
            SpiNNakerVersion::SpiNNaker2 => HardwareConstraints::spinnaker2_constraints(),
        };

        Self {
            config,
            constraints,
        }
    }

    /// Generate PyNN code
    fn generate_pynn_code(&self, network: &SpiNNakerNetwork) -> String {
        let mut code = String::new();

        code.push_str("# Generated PyNN code for SpiNNaker\n");
        code.push_str("import pyNN.spiNNaker as sim\n");
        code.push_str("from pyNN.utility.plotting import Figure, Panel\n");
        code.push_str("import matplotlib.pyplot as plt\n\n");

        // Setup
        code.push_str("# Setup simulation\n");
        code.push_str(&format!("sim.setup(timestep={})\n\n", self.config.timestep_ms));

        // Create populations
        code.push_str("# Create neuron populations\n");
        for (i, pop) in network.populations.iter().enumerate() {
            code.push_str(&format!(
                "pop_{} = sim.Population(\n",
                i
            ));
            code.push_str(&format!("    {},\n", pop.size));
            code.push_str(&format!("    sim.{}(**{{}}),\n", pop.cell_type));
            code.push_str(&format!("    label='{}'\n", pop.label));
            code.push_str(")\n");

            // Set neuron parameters
            if !pop.parameters.is_empty() {
                code.push_str(&format!("pop_{}.set(**{{\n", i));
                for (key, value) in &pop.parameters {
                    code.push_str(&format!("    '{}': {},\n", key, value));
                }
                code.push_str("})\n");
            }

            // Record spikes
            code.push_str(&format!("pop_{}.record(['spikes'])\n\n", i));
        }

        // Create projections (connections)
        code.push_str("# Create projections\n");
        for (i, proj) in network.projections.iter().enumerate() {
            code.push_str(&format!(
                "proj_{} = sim.Projection(\n",
                i
            ));
            code.push_str(&format!("    pop_{},\n", proj.source));
            code.push_str(&format!("    pop_{},\n", proj.target));
            code.push_str(&format!("    sim.{}(\n", proj.connector_type));

            // Connector-specific parameters
            match proj.connector_type.as_str() {
                "AllToAllConnector" => {
                    code.push_str(&format!("        weight={},\n", proj.weight));
                    code.push_str(&format!("        delay={}\n", proj.delay));
                }
                "OneToOneConnector" => {
                    code.push_str(&format!("        weight={},\n", proj.weight));
                    code.push_str(&format!("        delay={}\n", proj.delay));
                }
                "FixedProbabilityConnector" => {
                    code.push_str(&format!("        p_connect={},\n", proj.probability));
                    code.push_str(&format!("        weight={},\n", proj.weight));
                    code.push_str(&format!("        delay={}\n", proj.delay));
                }
                _ => {
                    code.push_str(&format!("        weight={},\n", proj.weight));
                    code.push_str(&format!("        delay={}\n", proj.delay));
                }
            }

            code.push_str("    ),\n");
            code.push_str(&format!("    receptor_type='{}'\n", proj.receptor_type));
            code.push_str(")\n\n");
        }

        // Setup live I/O if enabled
        if self.config.enable_live_io {
            code.push_str("# Setup live I/O\n");
            code.push_str("live_output = sim.external_devices.SpynnakerLiveSpikesConnection(\n");
            code.push_str("    send_labels=['pop_0'],\n");
            code.push_str("    receive_labels=['pop_0']\n");
            code.push_str(")\n\n");
        }

        // Run simulation
        code.push_str("# Run simulation\n");
        code.push_str(&format!("sim.run({})\n\n", self.config.run_time_ms));

        // Get data
        code.push_str("# Get spike data\n");
        for i in 0..network.populations.len() {
            code.push_str(&format!("spikes_{} = pop_{}.get_data('spikes').segments[0].spiketrains\n", i, i));
        }

        // End simulation
        code.push_str("\n# End simulation\n");
        code.push_str("sim.end()\n\n");

        // Plotting
        code.push_str("# Plot results\n");
        code.push_str("# plt.figure()\n");
        code.push_str("# ... add plotting code ...\n");
        code.push_str("# plt.show()\n");

        code
    }

    /// Generate routing table
    fn generate_routing_table(&self, network: &SpiNNakerNetwork) -> String {
        let mut routing = String::new();

        routing.push_str("# Routing table configuration\n");
        routing.push_str("# Format: (source_chip, source_core) -> [(target_chip, target_core)]\n\n");

        routing.push_str("routing_table = {\n");
        for entry in &network.routing_entries {
            routing.push_str(&format!(
                "    ({}, {}): [",
                entry.source_chip, entry.source_core
            ));
            for (i, target) in entry.targets.iter().enumerate() {
                if i > 0 {
                    routing.push_str(", ");
                }
                routing.push_str(&format!("({}, {})", target.chip, target.core));
            }
            routing.push_str("],\n");
        }
        routing.push_str("}\n");

        routing
    }

    /// Generate core allocation map
    fn generate_core_allocation(&self, network: &SpiNNakerNetwork) -> String {
        let mut allocation = String::new();

        allocation.push_str("# Core allocation map\n");
        allocation.push_str("# Population -> (chip_id, core_id, neuron_count)\n\n");

        allocation.push_str("core_allocation = {\n");
        for (i, alloc) in network.core_allocations.iter().enumerate() {
            allocation.push_str(&format!(
                "    'pop_{}': ({}, {}, {}),\n",
                i, alloc.chip_id, alloc.core_id, alloc.neuron_count
            ));
        }
        allocation.push_str("}\n");

        allocation
    }

    /// Map network to SpiNNaker structure
    fn map_to_spinnaker(&self, network: &NetworkDescription) -> SNNResult<SpiNNakerNetwork> {
        let mut populations = Vec::new();
        let mut projections = Vec::new();
        let mut core_allocations = Vec::new();
        let mut routing_entries = Vec::new();

        // Map populations
        for (i, pop) in network.populations.iter().enumerate() {
            // Map neuron model to PyNN cell type
            let cell_type = self.map_neuron_model(&pop.neuron_model);

            let mut parameters = HashMap::new();
            parameters.insert("tau_m".to_string(), pop.parameters.tau_mem);
            parameters.insert("tau_syn_E".to_string(), pop.parameters.tau_syn);
            parameters.insert("v_thresh".to_string(), pop.parameters.v_threshold);
            parameters.insert("v_reset".to_string(), pop.parameters.v_reset);
            parameters.insert("tau_refrac".to_string(), pop.parameters.t_refrac);

            populations.push(SpiNNakerPopulation {
                label: pop.id.clone(),
                size: pop.size,
                cell_type,
                parameters,
            });

            // Allocate to cores (simplified)
            let neurons_per_core = self.constraints.neuron.max_neurons_per_core;
            let num_cores = (pop.size + neurons_per_core - 1) / neurons_per_core;

            for core in 0..num_cores {
                let neurons_in_core = if core == num_cores - 1 {
                    pop.size - (core * neurons_per_core)
                } else {
                    neurons_per_core
                };

                core_allocations.push(CoreAllocation {
                    chip_id: core / 16, // Simplified chip allocation
                    core_id: core % 16,
                    neuron_count: neurons_in_core,
                });
            }
        }

        // Map connections
        for conn in &network.connections {
            let source_idx = network
                .populations
                .iter()
                .position(|p| p.id == conn.source)
                .unwrap_or(0);
            let target_idx = network
                .populations
                .iter()
                .position(|p| p.id == conn.target)
                .unwrap_or(0);

            let (connector_type, probability) = match conn.connection_type {
                super::ConnectionType::AllToAll => ("AllToAllConnector".to_string(), 1.0),
                super::ConnectionType::OneToOne => ("OneToOneConnector".to_string(), 1.0),
                super::ConnectionType::FixedProbability => {
                    ("FixedProbabilityConnector".to_string(), 0.1)
                }
                super::ConnectionType::Sparse => ("FromListConnector".to_string(), 1.0),
            };

            let avg_weight = if !conn.weights.is_empty() {
                conn.weights.iter().sum::<f32>() / conn.weights.len() as f32
            } else {
                0.5
            };

            let avg_delay = if !conn.delays.is_empty() {
                conn.delays.iter().sum::<f32>() / conn.delays.len() as f32
            } else {
                1.0
            };

            projections.push(SpiNNakerProjection {
                source: source_idx,
                target: target_idx,
                connector_type,
                weight: avg_weight,
                delay: avg_delay,
                receptor_type: "excitatory".to_string(),
                probability,
            });

            // Generate routing entry
            routing_entries.push(RoutingEntry {
                source_chip: 0,
                source_core: source_idx % 16,
                targets: vec![RoutingTarget {
                    chip: 0,
                    core: target_idx % 16,
                }],
            });
        }

        Ok(SpiNNakerNetwork {
            populations,
            projections,
            core_allocations,
            routing_entries,
        })
    }

    /// Map generic neuron model to PyNN cell type
    fn map_neuron_model(&self, model: &str) -> String {
        match model {
            "LIF" => "IF_curr_exp".to_string(),
            "AdaptiveLIF" | "ALIF" => "IF_cond_exp".to_string(),
            "Izhikevich" => "Izhikevich".to_string(),
            _ => "IF_curr_exp".to_string(), // Default
        }
    }
}

impl NeuromorphicExporter for SpiNNakerExporter {
    fn export(&mut self, network: &NetworkDescription) -> SNNResult<ExportResult> {
        // Validate network
        let warnings = self.validate(network)?;

        // Map to SpiNNaker
        let spinnaker_net = self.map_to_spinnaker(network)?;

        // Generate PyNN code
        let pynn_code = self.generate_pynn_code(&spinnaker_net);
        let routing_table = self.generate_routing_table(&spinnaker_net);
        let core_allocation = self.generate_core_allocation(&spinnaker_net);

        let mut files = vec![
            ExportFile {
                name: "spinnaker_network.py".to_string(),
                content: pynn_code.into_bytes(),
                file_type: "python".to_string(),
            },
            ExportFile {
                name: "routing_table.py".to_string(),
                content: routing_table.into_bytes(),
                file_type: "python".to_string(),
            },
            ExportFile {
                name: "core_allocation.py".to_string(),
                content: core_allocation.into_bytes(),
                file_type: "python".to_string(),
            },
        ];

        // Calculate statistics
        let num_neurons: usize = network.populations.iter().map(|p| p.size).sum();
        let num_synapses: usize = network.connections.iter().map(|c| c.weights.len()).sum();

        let stats = NetworkStats {
            num_neurons,
            num_synapses,
            num_populations: network.populations.len(),
            num_connections: network.connections.len(),
        };

        let num_cores: usize = spinnaker_net.core_allocations.len();
        let num_chips = (num_cores + 15) / 16; // 16 cores per chip

        let utilization = HardwareUtilization {
            cores_used: num_cores,
            chips_used: num_chips,
            neuron_utilization: (num_neurons as f32
                / self.constraints.neuron.max_neurons_total as f32)
                * 100.0,
            synapse_utilization: (num_synapses as f32
                / (num_neurons * self.constraints.synapse.max_fanin) as f32)
                * 100.0,
        };

        let metadata = ExportMetadata {
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            stats,
            utilization,
        };

        let target = match self.config.version {
            SpiNNakerVersion::SpiNNaker => NeuromorphicTarget::SpiNNaker,
            SpiNNakerVersion::SpiNNaker2 => NeuromorphicTarget::SpiNNaker2,
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

        // Check neuron count
        if num_neurons > self.constraints.neuron.max_neurons_total {
            return Err(SNNError::Export(format!(
                "Network has {} neurons, exceeds SpiNNaker limit of {}",
                num_neurons, self.constraints.neuron.max_neurons_total
            )));
        }

        // Check board availability
        if num_neurons > 1_000_000 && self.config.version == SpiNNakerVersion::SpiNNaker {
            warnings.push(
                "Network requires large SpiNNaker system (48+ chip board)".to_string()
            );
        }

        // Check delays
        for conn in &network.connections {
            for &delay in &conn.delays {
                let delay_steps = (delay / self.config.timestep_ms).round() as usize;
                if let Err(e) = self.constraints.delay_range.validate(delay_steps) {
                    warnings.push(format!("Delay validation warning: {}", e));
                }
            }
        }

        Ok(warnings)
    }

    fn constraints(&self) -> &HardwareConstraints {
        &self.constraints
    }

    fn optimize(&mut self, network: &NetworkDescription) -> SNNResult<NetworkDescription> {
        // Return unmodified for now
        Ok(network.clone())
    }
}

/// SpiNNaker configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiNNakerConfig {
    /// SpiNNaker version
    pub version: SpiNNakerVersion,
    /// Board identifier
    pub board: String,
    /// Simulation timestep (ms)
    pub timestep_ms: f32,
    /// Total run time (ms)
    pub run_time_ms: f32,
    /// Enable live I/O
    pub enable_live_io: bool,
}

impl Default for SpiNNakerConfig {
    fn default() -> Self {
        Self {
            version: SpiNNakerVersion::SpiNNaker2,
            board: "SpiNNaker2-48".to_string(),
            timestep_ms: 1.0,
            run_time_ms: 1000.0,
            enable_live_io: false,
        }
    }
}

/// SpiNNaker version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpiNNakerVersion {
    SpiNNaker,
    SpiNNaker2,
}

/// SpiNNaker population (neuron group)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiNNakerPopulation {
    pub label: String,
    pub size: usize,
    pub cell_type: String,
    pub parameters: HashMap<String, f32>,
}

/// SpiNNaker projection (connection)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpiNNakerProjection {
    pub source: usize,
    pub target: usize,
    pub connector_type: String,
    pub weight: f32,
    pub delay: f32,
    pub receptor_type: String,
    pub probability: f32,
}

/// Core allocation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreAllocation {
    pub chip_id: usize,
    pub core_id: usize,
    pub neuron_count: usize,
}

/// Routing entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingEntry {
    pub source_chip: usize,
    pub source_core: usize,
    pub targets: Vec<RoutingTarget>,
}

/// Routing target
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTarget {
    pub chip: usize,
    pub core: usize,
}

/// Complete SpiNNaker network
#[derive(Debug, Clone)]
pub struct SpiNNakerNetwork {
    pub populations: Vec<SpiNNakerPopulation>,
    pub projections: Vec<SpiNNakerProjection>,
    pub core_allocations: Vec<CoreAllocation>,
    pub routing_entries: Vec<RoutingEntry>,
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
                    id: "input".to_string(),
                    size: 100,
                    neuron_model: "LIF".to_string(),
                    parameters: NeuronParameters::default(),
                },
                Population {
                    id: "hidden".to_string(),
                    size: 50,
                    neuron_model: "LIF".to_string(),
                    parameters: NeuronParameters::default(),
                },
            ],
            connections: vec![Connection {
                id: "input_hidden".to_string(),
                source: "input".to_string(),
                target: "hidden".to_string(),
                weights: vec![0.5; 5000],
                delays: vec![1.0; 5000],
                shape: (100, 50),
                connection_type: ConnectionType::AllToAll,
            }],
            parameters: NetworkParameters::default(),
        }
    }

    #[test]
    fn test_spinnaker_exporter_creation() {
        let config = SpiNNakerConfig::default();
        let exporter = SpiNNakerExporter::new(config);

        assert_eq!(exporter.config.version, SpiNNakerVersion::SpiNNaker2);
    }

    #[test]
    fn test_validate_network() {
        let config = SpiNNakerConfig::default();
        let exporter = SpiNNakerExporter::new(config);
        let network = create_test_network();

        let result = exporter.validate(&network);
        assert!(result.is_ok());
    }

    #[test]
    fn test_export() {
        let config = SpiNNakerConfig::default();
        let mut exporter = SpiNNakerExporter::new(config);
        let network = create_test_network();

        let result = exporter.export(&network);
        assert!(result.is_ok());

        let export = result.unwrap();
        assert_eq!(export.files.len(), 3); // PyNN + routing + allocation
    }

    #[test]
    fn test_generate_pynn_code() {
        let config = SpiNNakerConfig::default();
        let exporter = SpiNNakerExporter::new(config);

        let spinnaker_net = SpiNNakerNetwork {
            populations: vec![SpiNNakerPopulation {
                label: "test".to_string(),
                size: 100,
                cell_type: "IF_curr_exp".to_string(),
                parameters: HashMap::new(),
            }],
            projections: vec![],
            core_allocations: vec![],
            routing_entries: vec![],
        };

        let code = exporter.generate_pynn_code(&spinnaker_net);
        assert!(code.contains("import pyNN.spiNNaker"));
        assert!(code.contains("sim.Population"));
        assert!(code.contains("sim.run"));
    }

    #[test]
    fn test_map_neuron_model() {
        let config = SpiNNakerConfig::default();
        let exporter = SpiNNakerExporter::new(config);

        assert_eq!(exporter.map_neuron_model("LIF"), "IF_curr_exp");
        assert_eq!(exporter.map_neuron_model("Izhikevich"), "Izhikevich");
        assert_eq!(exporter.map_neuron_model("Unknown"), "IF_curr_exp");
    }

    #[test]
    fn test_spinnaker2_has_more_capacity() {
        let config1 = SpiNNakerConfig {
            version: SpiNNakerVersion::SpiNNaker,
            ..Default::default()
        };
        let config2 = SpiNNakerConfig {
            version: SpiNNakerVersion::SpiNNaker2,
            ..Default::default()
        };

        let exporter1 = SpiNNakerExporter::new(config1);
        let exporter2 = SpiNNakerExporter::new(config2);

        assert!(
            exporter2.constraints().neuron.max_neurons_total
                > exporter1.constraints().neuron.max_neurons_total
        );
    }
}
