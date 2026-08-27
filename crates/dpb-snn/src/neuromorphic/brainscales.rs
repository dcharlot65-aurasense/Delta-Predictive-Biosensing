//! BrainScaleS neuromorphic hardware export
//!
//! Provides export capabilities for BrainScaleS and BrainScaleS-2 analog
//! neuromorphic hardware using PyHMF framework.

use super::constraints::HardwareConstraints;
use super::quantization::{QuantizationScheme, WeightQuantizer};
use super::{
    ExportFile, ExportMetadata, ExportResult, HardwareUtilization, NetworkDescription,
    NetworkStats, NeuromorphicExporter, NeuromorphicTarget,
};
use crate::{SNNError, SNNResult};
use serde::{Deserialize, Serialize};

/// BrainScaleS exporter
pub struct BrainScaleSExporter {
    config: BrainScaleSConfig,
    constraints: HardwareConstraints,
}

impl BrainScaleSExporter {
    /// Create a new BrainScaleS exporter
    pub fn new(config: BrainScaleSConfig) -> Self {
        let constraints = match config.version {
            BrainScaleSVersion::BrainScaleS => HardwareConstraints::brainscales_constraints(),
            BrainScaleSVersion::BrainScaleS2 => HardwareConstraints::brainscales2_constraints(),
        };

        Self {
            config,
            constraints,
        }
    }

    /// Generate PyHMF configuration
    fn generate_pyhmf_code(&self, network: &BrainScaleSNetwork) -> String {
        let mut code = String::new();

        code.push_str("# Generated PyHMF code for BrainScaleS\n");
        code.push_str("import pynn_brainscales.brainscales2 as pynn\n");
        code.push_str("from dlens_vx_v3 import hal, sta\n\n");

        // Setup
        code.push_str("# Initialize hardware\n");
        code.push_str("pynn.setup(\n");
        code.push_str(&format!("    timestep={},\n", self.config.timestep_us));
        code.push_str("    hardware_time=True,  # Use accelerated hardware time\n");
        code.push_str(")\n\n");

        // Calibration
        if self.config.enable_calibration {
            code.push_str("# Apply calibration\n");
            code.push_str("pynn.preprocess(lambda: pynn.calib.calibrate())\n\n");
        }

        // Create neurons
        code.push_str("# Create neuron populations\n");
        for (i, neuron_group) in network.neurons.iter().enumerate() {
            code.push_str(&format!("neurons_{} = pynn.Population(\n", i));
            code.push_str(&format!("    {},\n", neuron_group.size));
            code.push_str("    pynn.cells.HXNeuron(\n");

            // Analog neuron parameters (calibrated)
            code.push_str(&format!("        threshold={},\n", neuron_group.threshold));
            code.push_str(&format!("        leak={},\n", neuron_group.leak));
            code.push_str(&format!("        reset={},\n", neuron_group.reset));
            code.push_str(&format!("        tau_mem={},\n", neuron_group.tau_mem));
            code.push_str(&format!("        tau_syn={},\n", neuron_group.tau_syn));

            // Refractory period
            code.push_str(&format!(
                "        refractory_period_hw={},\n",
                neuron_group.refractory_period_hw
            ));

            code.push_str("    ),\n");
            code.push_str(&format!("    label='neurons_{}'\n", i));
            code.push_str(")\n");

            // Record spikes
            code.push_str(&format!("neurons_{}.record(['spikes'])\n\n", i));
        }

        // Create synapses
        code.push_str("# Create synaptic connections\n");
        for (i, synapse) in network.synapses.iter().enumerate() {
            code.push_str(&format!("projection_{} = pynn.Projection(\n", i));
            code.push_str(&format!("    neurons_{},\n", synapse.source));
            code.push_str(&format!("    neurons_{},\n", synapse.target));

            // Connector type
            match synapse.connector_type.as_str() {
                "all_to_all" => {
                    code.push_str("    pynn.AllToAllConnector(),\n");
                    code.push_str(&format!(
                        "    synapse_type=pynn.StaticSynapse(weight={}),\n",
                        synapse.weight
                    ));
                }
                "one_to_one" => {
                    code.push_str("    pynn.OneToOneConnector(),\n");
                    code.push_str(&format!(
                        "    synapse_type=pynn.StaticSynapse(weight={}),\n",
                        synapse.weight
                    ));
                }
                _ => {
                    code.push_str("    pynn.AllToAllConnector(),\n");
                    code.push_str(&format!(
                        "    synapse_type=pynn.StaticSynapse(weight={}),\n",
                        synapse.weight
                    ));
                }
            }

            code.push_str("    receptor_type='excitatory',\n");
            code.push_str(&format!("    label='projection_{}'\n", i));
            code.push_str(")\n\n");
        }

        // Run emulation
        code.push_str("# Run hardware emulation\n");
        code.push_str(&format!("pynn.run({})\n\n", self.config.run_time_us));

        // Retrieve data
        code.push_str("# Get spike data\n");
        for i in 0..network.neurons.len() {
            code.push_str(&format!(
                "spikes_{} = neurons_{}.get_data('spikes').segments[0].spiketrains\n",
                i, i
            ));
        }

        // Cleanup
        code.push_str("\n# End emulation\n");
        code.push_str("pynn.end()\n");

        code
    }

    /// Generate calibration data export
    fn generate_calibration_data(&self, network: &BrainScaleSNetwork) -> String {
        let mut calib = String::new();

        calib.push_str("# BrainScaleS calibration data\n");
        calib.push_str("# This file contains hardware-specific calibration parameters\n\n");

        calib.push_str("calibration_data = {\n");

        // Neuron calibration
        calib.push_str("    'neurons': {\n");
        for (i, neuron_group) in network.neurons.iter().enumerate() {
            calib.push_str(&format!("        'group_{}': {{\n", i));
            calib.push_str(&format!(
                "            'threshold_calib': {},\n",
                neuron_group.threshold_calib
            ));
            calib.push_str(&format!(
                "            'leak_calib': {},\n",
                neuron_group.leak_calib
            ));
            calib.push_str(&format!(
                "            'reset_calib': {},\n",
                neuron_group.reset_calib
            ));
            calib.push_str("        },\n");
        }
        calib.push_str("    },\n");

        // Synapse calibration
        calib.push_str("    'synapses': {\n");
        for (i, synapse) in network.synapses.iter().enumerate() {
            calib.push_str(&format!("        'synapse_{}': {{\n", i));
            calib.push_str(&format!(
                "            'weight_calib': {},\n",
                synapse.weight_calib
            ));
            calib.push_str(&format!(
                "            'dac_value': {},\n",
                synapse.dac_value
            ));
            calib.push_str("        },\n");
        }
        calib.push_str("    },\n");

        calib.push_str("}\n");

        calib
    }

    /// Map network to BrainScaleS structure
    fn map_to_brainscales(&self, network: &NetworkDescription) -> SNNResult<BrainScaleSNetwork> {
        let mut neurons = Vec::new();
        let mut synapses = Vec::new();

        // Map populations to analog neurons
        for pop in &network.populations {
            // Quantize neuron parameters for analog hardware
            let threshold = self.quantize_voltage(pop.parameters.v_threshold);
            let reset = self.quantize_voltage(pop.parameters.v_reset);
            let leak = self.calculate_leak_conductance(pop.parameters.tau_mem);

            // Calculate hardware time constants (accelerated by 1000x)
            let tau_mem_hw = pop.parameters.tau_mem / 1000.0;
            let tau_syn_hw = pop.parameters.tau_syn / 1000.0;

            neurons.push(BrainScaleSNeuron {
                size: pop.size,
                threshold,
                leak,
                reset,
                tau_mem: tau_mem_hw,
                tau_syn: tau_syn_hw,
                refractory_period_hw: (pop.parameters.t_refrac / 1000.0) as u32,
                // Calibration values (would be measured from hardware)
                threshold_calib: threshold * 1.1,
                leak_calib: leak * 0.95,
                reset_calib: reset * 1.05,
            });
        }

        // Map connections to synapses
        let mut quantizer = WeightQuantizer::new(
            QuantizationScheme::Asymmetric,
            self.constraints.weight_precision,
        );

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

            let connector_type = match conn.connection_type {
                super::ConnectionType::AllToAll => "all_to_all".to_string(),
                super::ConnectionType::OneToOne => "one_to_one".to_string(),
                _ => "all_to_all".to_string(),
            };

            // Quantize weights
            quantizer.calibrate(&conn.weights);
            let quantized = quantizer.quantize(&conn.weights);

            let avg_weight = if !quantized.quantized_weights.is_empty() {
                quantized.quantized_weights.iter().sum::<i32>() as f32
                    / quantized.quantized_weights.len() as f32
            } else {
                0.0
            };

            // Calculate DAC value for analog hardware
            let dac_value = self.weight_to_dac(avg_weight);

            synapses.push(BrainScaleSynapse {
                source: source_idx,
                target: target_idx,
                weight: avg_weight,
                connector_type,
                weight_calib: avg_weight * 1.05, // Calibration correction
                dac_value,
            });
        }

        Ok(BrainScaleSNetwork { neurons, synapses })
    }

    /// Quantize voltage for analog hardware
    fn quantize_voltage(&self, voltage: f32) -> f32 {
        // BrainScaleS uses limited DAC resolution
        let bits = self.constraints.weight_precision.bits() as u32;
        let levels = (1 << bits) - 1;

        let normalized = (voltage + 2.0) / 4.0; // Assume -2V to +2V range
        let quantized_level = (normalized * levels as f32).round();

        (quantized_level / levels as f32) * 4.0 - 2.0
    }

    /// Calculate leak conductance from time constant
    fn calculate_leak_conductance(&self, tau_mem: f32) -> f32 {
        // g_leak = C_mem / tau_mem (simplified)
        let c_mem = 1.0; // Normalized capacitance
        c_mem / tau_mem
    }

    /// Convert weight to DAC value
    fn weight_to_dac(&self, weight: f32) -> u32 {
        let bits = self.constraints.weight_precision.bits() as u32;
        let max_dac = (1 << bits) - 1;

        let normalized = (weight + 1.0) / 2.0; // Map [-1, 1] to [0, 1]
        (normalized * max_dac as f32).round() as u32
    }
}

impl NeuromorphicExporter for BrainScaleSExporter {
    fn export(&mut self, network: &NetworkDescription) -> SNNResult<ExportResult> {
        // Validate network
        let warnings = self.validate(network)?;

        // Map to BrainScaleS
        let bs_network = self.map_to_brainscales(network)?;

        // Generate PyHMF code
        let pyhmf_code = self.generate_pyhmf_code(&bs_network);
        let calibration_data = self.generate_calibration_data(&bs_network);

        let files = vec![
            ExportFile {
                name: "brainscales_network.py".to_string(),
                content: pyhmf_code.into_bytes(),
                file_type: "python".to_string(),
            },
            ExportFile {
                name: "calibration_data.py".to_string(),
                content: calibration_data.into_bytes(),
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

        let utilization = HardwareUtilization {
            cores_used: 1, // Single wafer module
            chips_used: 1,
            neuron_utilization: (num_neurons as f32
                / self.constraints.neuron.max_neurons_total as f32)
                * 100.0,
            synapse_utilization: (num_synapses as f32
                / (num_neurons * self.constraints.synapse.max_fanin) as f32)
                * 100.0,
        };

        let metadata = ExportMetadata {
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            stats,
            utilization,
        };

        let target = match self.config.version {
            BrainScaleSVersion::BrainScaleS => NeuromorphicTarget::BrainScaleS,
            BrainScaleSVersion::BrainScaleS2 => NeuromorphicTarget::BrainScaleS2,
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

        // Check neuron count (very limited for BrainScaleS)
        if num_neurons > self.constraints.neuron.max_neurons_total {
            return Err(SNNError::Export(format!(
                "Network has {} neurons, exceeds BrainScaleS limit of {}",
                num_neurons, self.constraints.neuron.max_neurons_total
            )));
        }

        // Warn about analog limitations
        warnings.push(
            "BrainScaleS uses analog neurons - expect device variation and calibration needs"
                .to_string(),
        );

        // Check for delays (not supported)
        for conn in &network.connections {
            if conn.delays.iter().any(|&d| d > 0.0) {
                warnings.push(
                    "BrainScaleS does not support explicit delays - delays will be ignored"
                        .to_string(),
                );
                break;
            }
        }

        // Check synapse limits
        for conn in &network.connections {
            let fanin = conn.shape.0;
            if fanin > self.constraints.synapse.max_fanin {
                return Err(SNNError::Export(format!(
                    "Connection fanin {} exceeds BrainScaleS limit of {}",
                    fanin, self.constraints.synapse.max_fanin
                )));
            }
        }

        // Warn about time acceleration
        warnings.push(format!(
            "BrainScaleS runs at {}x biological time - adjust time constants accordingly",
            self.config.speedup_factor
        ));

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

/// BrainScaleS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainScaleSConfig {
    /// BrainScaleS version
    pub version: BrainScaleSVersion,
    /// Timestep (microseconds in biological time)
    pub timestep_us: f32,
    /// Run time (microseconds in biological time)
    pub run_time_us: f32,
    /// Enable hardware calibration
    pub enable_calibration: bool,
    /// Hardware speedup factor (typically 1000x)
    pub speedup_factor: u32,
}

impl Default for BrainScaleSConfig {
    fn default() -> Self {
        Self {
            version: BrainScaleSVersion::BrainScaleS2,
            timestep_us: 1.0,
            run_time_us: 1000.0,
            enable_calibration: true,
            speedup_factor: 1000,
        }
    }
}

/// BrainScaleS version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BrainScaleSVersion {
    BrainScaleS,
    BrainScaleS2,
}

/// BrainScaleS neuron (analog)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainScaleSNeuron {
    pub size: usize,
    pub threshold: f32,
    pub leak: f32,
    pub reset: f32,
    pub tau_mem: f32,
    pub tau_syn: f32,
    pub refractory_period_hw: u32,
    // Calibration parameters
    pub threshold_calib: f32,
    pub leak_calib: f32,
    pub reset_calib: f32,
}

/// BrainScaleS synapse
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrainScaleSynapse {
    pub source: usize,
    pub target: usize,
    pub weight: f32,
    pub connector_type: String,
    // Hardware-specific
    pub weight_calib: f32,
    pub dac_value: u32,
}

/// Complete BrainScaleS network
#[derive(Debug, Clone)]
pub struct BrainScaleSNetwork {
    pub neurons: Vec<BrainScaleSNeuron>,
    pub synapses: Vec<BrainScaleSynapse>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuromorphic::{
        Connection, ConnectionType, NetworkParameters, NeuronParameters, Population,
    };

    fn create_test_network() -> NetworkDescription {
        NetworkDescription {
            name: "test_network".to_string(),
            populations: vec![
                Population {
                    id: "neurons1".to_string(),
                    size: 100,
                    neuron_model: "AdEx".to_string(),
                    parameters: NeuronParameters::default(),
                },
                Population {
                    id: "neurons2".to_string(),
                    size: 50,
                    neuron_model: "AdEx".to_string(),
                    parameters: NeuronParameters::default(),
                },
            ],
            connections: vec![Connection {
                id: "conn1".to_string(),
                source: "neurons1".to_string(),
                target: "neurons2".to_string(),
                weights: vec![0.5; 5000],
                delays: vec![0.0; 5000], // No delays
                shape: (100, 50),
                connection_type: ConnectionType::AllToAll,
            }],
            parameters: NetworkParameters::default(),
        }
    }

    #[test]
    fn test_brainscales_exporter_creation() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);

        assert_eq!(exporter.config.version, BrainScaleSVersion::BrainScaleS2);
        assert_eq!(exporter.config.speedup_factor, 1000);
    }

    #[test]
    fn test_validate_network() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);
        let network = create_test_network();

        let result = exporter.validate(&network);
        assert!(result.is_ok());

        let warnings = result.unwrap();
        assert!(!warnings.is_empty()); // Should have warnings about analog hardware
    }

    #[test]
    fn test_export() {
        let config = BrainScaleSConfig::default();
        let mut exporter = BrainScaleSExporter::new(config);
        let network = create_test_network();

        let result = exporter.export(&network);
        assert!(result.is_ok());

        let export = result.unwrap();
        assert_eq!(export.files.len(), 2); // PyHMF + calibration
    }

    #[test]
    fn test_generate_pyhmf_code() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);

        let bs_net = BrainScaleSNetwork {
            neurons: vec![BrainScaleSNeuron {
                size: 100,
                threshold: 1.0,
                leak: 0.1,
                reset: 0.0,
                tau_mem: 20.0,
                tau_syn: 5.0,
                refractory_period_hw: 2,
                threshold_calib: 1.1,
                leak_calib: 0.095,
                reset_calib: 0.05,
            }],
            synapses: vec![],
        };

        let code = exporter.generate_pyhmf_code(&bs_net);
        assert!(code.contains("import pynn_brainscales"));
        assert!(code.contains("HXNeuron"));
        assert!(code.contains("hardware_time"));
    }

    #[test]
    fn test_quantize_voltage() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);

        let quantized = exporter.quantize_voltage(1.0);
        assert!((-2.0..=2.0).contains(&quantized));
    }

    #[test]
    fn test_weight_to_dac() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);

        let dac = exporter.weight_to_dac(0.0);
        assert!(dac > 0); // Should be mid-range

        let dac_max = exporter.weight_to_dac(1.0);
        let dac_min = exporter.weight_to_dac(-1.0);
        assert!(dac_max > dac_min);
    }

    #[test]
    fn test_limited_neuron_capacity() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);

        // BrainScaleS has very limited neurons
        assert_eq!(exporter.constraints().neuron.max_neurons_total, 512);
    }

    #[test]
    fn test_low_weight_precision() {
        let config = BrainScaleSConfig::default();
        let exporter = BrainScaleSExporter::new(config);

        // BrainScaleS has low weight precision
        let bits = exporter.constraints().weight_precision.bits();
        assert!(bits <= 6);
    }
}
