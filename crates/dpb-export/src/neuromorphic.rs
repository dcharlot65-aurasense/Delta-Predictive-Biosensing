//! Neuromorphic hardware export for DPB spike encoders and SNNs.
//!
//! This module provides export capabilities for major neuromorphic platforms:
//!
//! - **Intel Loihi 2** (via Lava framework) — see the status note below
//!
//! # Status of the Lava target (verified 2026-08-01)
//!
//! Intel **archived every `lava-nc` repository on 2026-05-13**. The framework is
//! read-only and unsupported; there is no announced successor. Code emitted for
//! the `Loihi2` target therefore targets a dead SDK.
//!
//! The target is retained rather than deleted because existing Loihi 2 hardware
//! and existing Lava installations still run, and removing the exporter would
//! strand them. It should be treated as **legacy**: do not build new work on it.
//!
//! For portable neuromorphic interchange prefer **NIR** (Neuromorphic
//! Intermediate Representation), which is actively maintained — v1.0.8 released
//! 2026-07-06 — and is supported across multiple simulators and hardware
//! backends rather than a single vendor's stack.
//! - **SpiNNaker 2** (via PyNN/sPyNNaker)
//! - **BrainScaleS-2** (via PyNN/hxtorch)
//!
//! ## Supported Export Formats
//!
//! | Platform | Format | Framework |
//! |----------|--------|-----------|
//! | Loihi 2 | Python/Lava | lava-nc |
//! | SpiNNaker | PyNN | sPyNNaker |
//! | BrainScaleS | PyNN/hxtorch | hxtorch |
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_export::neuromorphic::{NeuromorphicExporter, NeuromorphicTarget, NetworkConfig};
//!
//! let config = NetworkConfig::new(128, 64, 32); // input, hidden, output
//! let exporter = NeuromorphicExporter::new(NeuromorphicTarget::Loihi2);
//!
//! let lava_code = exporter.export(&config)?;
//! std::fs::write("network.py", lava_code)?;
//! ```

use std::fmt::Write;

/// Neuromorphic hardware target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuromorphicTarget {
    /// Intel Loihi 2, via the Lava framework.
    ///
    /// **Legacy.** Intel archived all `lava-nc` repositories on 2026-05-13;
    /// the SDK this emits for is unsupported. Retained so existing Loihi 2
    /// deployments are not stranded. Prefer NIR for new work.
    #[deprecated(
        since = "0.1.0",
        note = "Lava was archived by Intel on 2026-05-13 and is unsupported; \
               prefer NIR for portable neuromorphic export"
    )]
    Loihi2,
    /// SpiNNaker 2 (via sPyNNaker).
    SpiNNaker2,
    /// BrainScaleS-2 (via hxtorch).
    BrainScaleS2,
    /// Generic PyNN (portable).
    GenericPyNN,
}

impl NeuromorphicTarget {
    /// Get target name.
    pub fn name(&self) -> &'static str {
        match self {
            #[allow(deprecated)]
            NeuromorphicTarget::Loihi2 => "Intel Loihi 2",
            NeuromorphicTarget::SpiNNaker2 => "SpiNNaker 2",
            NeuromorphicTarget::BrainScaleS2 => "BrainScaleS-2",
            NeuromorphicTarget::GenericPyNN => "Generic PyNN",
        }
    }

    /// Get framework name.
    pub fn framework(&self) -> &'static str {
        match self {
            #[allow(deprecated)]
            NeuromorphicTarget::Loihi2 => "Lava",
            NeuromorphicTarget::SpiNNaker2 => "sPyNNaker",
            NeuromorphicTarget::BrainScaleS2 => "hxtorch",
            NeuromorphicTarget::GenericPyNN => "PyNN",
        }
    }

    /// Get hardware specifications.
    pub fn specs(&self) -> HardwareSpecs {
        match self {
            #[allow(deprecated)]
            NeuromorphicTarget::Loihi2 => HardwareSpecs {
                neurons_per_core: 8192,
                cores_per_chip: 128,
                synapses_per_neuron: 4096,
                weight_bits: 8,
                supports_learning: true,
                supports_dendrites: true,
                time_resolution_us: 1.0,
            },
            NeuromorphicTarget::SpiNNaker2 => HardwareSpecs {
                neurons_per_core: 1000,
                cores_per_chip: 152,
                synapses_per_neuron: 16000,
                weight_bits: 16,
                supports_learning: true,
                supports_dendrites: false,
                time_resolution_us: 1000.0, // 1ms
            },
            NeuromorphicTarget::BrainScaleS2 => HardwareSpecs {
                neurons_per_core: 512,
                cores_per_chip: 1, // HICANN-X
                synapses_per_neuron: 256,
                weight_bits: 6,
                supports_learning: true,
                supports_dendrites: true,
                time_resolution_us: 0.001, // Analog, ~1000x speedup
            },
            NeuromorphicTarget::GenericPyNN => HardwareSpecs {
                neurons_per_core: 10000,
                cores_per_chip: 1,
                synapses_per_neuron: 10000,
                weight_bits: 32,
                supports_learning: true,
                supports_dendrites: false,
                time_resolution_us: 100.0,
            },
        }
    }
}

/// Hardware specifications for neuromorphic platforms.
#[derive(Debug, Clone)]
pub struct HardwareSpecs {
    /// Maximum neurons per core.
    pub neurons_per_core: u32,
    /// Cores per chip.
    pub cores_per_chip: u32,
    /// Maximum synapses per neuron.
    pub synapses_per_neuron: u32,
    /// Weight precision in bits.
    pub weight_bits: u8,
    /// Supports on-chip learning.
    pub supports_learning: bool,
    /// Supports dendritic computation.
    pub supports_dendrites: bool,
    /// Time resolution in microseconds.
    pub time_resolution_us: f32,
}

/// Neuron model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuronModel {
    /// Leaky Integrate-and-Fire.
    LIF,
    /// Current-based LIF.
    CUBA,
    /// Conductance-based LIF.
    COBA,
    /// Adaptive LIF.
    ALIF,
    /// Izhikevich model.
    Izhikevich,
    /// Adaptive Exponential.
    AdEx,
}

impl NeuronModel {
    /// Get PyNN cell type name.
    pub fn pynn_name(&self) -> &'static str {
        match self {
            NeuronModel::LIF => "IF_curr_exp",
            NeuronModel::CUBA => "IF_curr_exp",
            NeuronModel::COBA => "IF_cond_exp",
            NeuronModel::ALIF => "IF_curr_exp", // Approximation
            NeuronModel::Izhikevich => "Izhikevich",
            NeuronModel::AdEx => "EIF_cond_exp_isfa_ista",
        }
    }

    /// Get Lava process name.
    pub fn lava_name(&self) -> &'static str {
        match self {
            NeuronModel::LIF => "LIF",
            NeuronModel::CUBA => "LIF",
            NeuronModel::COBA => "LIFCond", // Hypothetical
            NeuronModel::ALIF => "ALIF",
            NeuronModel::Izhikevich => "Izhikevich",
            NeuronModel::AdEx => "AdEx",
        }
    }
}

/// Synapse model type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseModel {
    /// Static weights.
    Static,
    /// STDP learning.
    STDP,
    /// Short-term plasticity.
    STP,
    /// Reward-modulated STDP.
    RSTDP,
}

/// Network layer configuration.
#[derive(Debug, Clone)]
pub struct LayerConfig {
    /// Layer name.
    pub name: String,
    /// Number of neurons.
    pub size: usize,
    /// Neuron model.
    pub neuron_model: NeuronModel,
    /// Neuron parameters.
    pub params: NeuronParams,
}

/// Neuron parameters.
#[derive(Debug, Clone)]
pub struct NeuronParams {
    /// Membrane time constant (ms).
    pub tau_m: f32,
    /// Synaptic time constant (ms).
    pub tau_syn: f32,
    /// Threshold voltage (mV).
    pub v_thresh: f32,
    /// Reset voltage (mV).
    pub v_reset: f32,
    /// Resting potential (mV).
    pub v_rest: f32,
    /// Refractory period (ms).
    pub t_refrac: f32,
    /// Membrane capacitance (nF).
    pub cm: f32,
}

impl Default for NeuronParams {
    fn default() -> Self {
        Self {
            tau_m: 10.0,
            tau_syn: 5.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            t_refrac: 2.0,
            cm: 1.0,
        }
    }
}

/// Connection configuration.
#[derive(Debug, Clone)]
pub struct ConnectionConfig {
    /// Source layer name.
    pub source: String,
    /// Target layer name.
    pub target: String,
    /// Connection type.
    pub conn_type: ConnectionType,
    /// Weight matrix (if dense).
    pub weights: Option<Vec<Vec<f32>>>,
    /// Synapse model.
    pub synapse_model: SynapseModel,
    /// Delay (ms).
    pub delay: f32,
}

/// Connection type.
#[derive(Debug, Clone)]
pub enum ConnectionType {
    /// Full connectivity.
    AllToAll,
    /// One-to-one connectivity.
    OneToOne,
    /// Random with probability.
    Random { probability: f32 },
    /// From explicit weight matrix.
    FromMatrix,
}

/// Complete network configuration.
#[derive(Debug, Clone)]
pub struct NetworkConfig {
    /// Network name.
    pub name: String,
    /// Layers.
    pub layers: Vec<LayerConfig>,
    /// Connections.
    pub connections: Vec<ConnectionConfig>,
    /// Simulation duration (ms).
    pub duration_ms: f32,
    /// Time step (ms).
    pub dt: f32,
}

impl NetworkConfig {
    /// Create a simple feedforward network.
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let params = NeuronParams::default();

        Self {
            name: "dpb_network".to_string(),
            layers: vec![
                LayerConfig {
                    name: "input".to_string(),
                    size: input_size,
                    neuron_model: NeuronModel::LIF,
                    params: params.clone(),
                },
                LayerConfig {
                    name: "hidden".to_string(),
                    size: hidden_size,
                    neuron_model: NeuronModel::LIF,
                    params: params.clone(),
                },
                LayerConfig {
                    name: "output".to_string(),
                    size: output_size,
                    neuron_model: NeuronModel::LIF,
                    params: params,
                },
            ],
            connections: vec![
                ConnectionConfig {
                    source: "input".to_string(),
                    target: "hidden".to_string(),
                    conn_type: ConnectionType::AllToAll,
                    weights: None,
                    synapse_model: SynapseModel::Static,
                    delay: 1.0,
                },
                ConnectionConfig {
                    source: "hidden".to_string(),
                    target: "output".to_string(),
                    conn_type: ConnectionType::AllToAll,
                    weights: None,
                    synapse_model: SynapseModel::Static,
                    delay: 1.0,
                },
            ],
            duration_ms: 100.0,
            dt: 1.0,
        }
    }

    /// Add a layer.
    pub fn add_layer(&mut self, config: LayerConfig) {
        self.layers.push(config);
    }

    /// Add a connection.
    pub fn add_connection(&mut self, config: ConnectionConfig) {
        self.connections.push(config);
    }

    /// Set weight matrix for a connection.
    pub fn set_weights(&mut self, source: &str, target: &str, weights: Vec<Vec<f32>>) {
        for conn in &mut self.connections {
            if conn.source == source && conn.target == target {
                conn.weights = Some(weights.clone());
                conn.conn_type = ConnectionType::FromMatrix;
            }
        }
    }
}

/// Neuromorphic hardware exporter.
pub struct NeuromorphicExporter {
    /// Target platform.
    target: NeuromorphicTarget,
    /// Include spike input generator.
    include_input_gen: bool,
    /// Include spike recorder.
    include_recorder: bool,
    /// Include visualization code.
    include_viz: bool,
}

impl NeuromorphicExporter {
    /// Create a new exporter for the specified target.
    pub fn new(target: NeuromorphicTarget) -> Self {
        Self {
            target,
            include_input_gen: true,
            include_recorder: true,
            include_viz: true,
        }
    }

    /// Set whether to include input generator.
    pub fn with_input_gen(mut self, include: bool) -> Self {
        self.include_input_gen = include;
        self
    }

    /// Set whether to include spike recorder.
    pub fn with_recorder(mut self, include: bool) -> Self {
        self.include_recorder = include;
        self
    }

    /// Export network configuration.
    pub fn export(&self, config: &NetworkConfig) -> Result<String, NeuromorphicExportError> {
        match self.target {
            #[allow(deprecated)]
            NeuromorphicTarget::Loihi2 => self.export_lava(config),
            NeuromorphicTarget::SpiNNaker2 => self.export_spinnaker(config),
            NeuromorphicTarget::BrainScaleS2 => self.export_brainscales(config),
            NeuromorphicTarget::GenericPyNN => self.export_pynn(config),
        }
    }

    /// Export for Intel Loihi 2 (Lava framework).
    fn export_lava(&self, config: &NetworkConfig) -> Result<String, NeuromorphicExportError> {
        let mut code = String::new();

        // Header
        writeln!(code, "# DPB Network Export for Intel Loihi 2")?;
        writeln!(code, "# Generated by dpb-export")?;
        writeln!(code, "# Framework: Lava-NC")?;
        writeln!(code, "#")?;
        writeln!(code, "# Network: {}", config.name)?;
        writeln!(code, "# Layers: {}", config.layers.len())?;
        writeln!(code, "# Connections: {}", config.connections.len())?;
        writeln!(code)?;

        // Imports
        writeln!(code, "import numpy as np")?;
        writeln!(code, "from lava.magma.core.process.process import AbstractProcess")?;
        writeln!(code, "from lava.magma.core.process.ports.ports import InPort, OutPort")?;
        writeln!(code, "from lava.magma.core.model.py.model import PyLoihiProcessModel")?;
        writeln!(code, "from lava.magma.core.model.py.type import LavaPyType")?;
        writeln!(code, "from lava.magma.core.model.py.ports import PyInPort, PyOutPort")?;
        writeln!(code, "from lava.magma.core.resources import CPU, Loihi2NeuroCore")?;
        writeln!(code, "from lava.magma.core.decorator import implements, requires")?;
        writeln!(code, "from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol")?;
        writeln!(code, "from lava.proc.lif.process import LIF")?;
        writeln!(code, "from lava.proc.dense.process import Dense")?;
        writeln!(code)?;

        // Network class
        writeln!(code, "class {}Network:", config.name)?;
        writeln!(code, "    \"\"\"DPB-generated Lava network for Loihi 2.\"\"\"")?;
        writeln!(code)?;
        writeln!(code, "    def __init__(self):")?;
        writeln!(code, "        # Create layers")?;

        // Create layers
        for layer in &config.layers {
            let params = &layer.params;
            writeln!(code, "        self.{} = LIF(", layer.name)?;
            writeln!(code, "            shape=({},),", layer.size)?;
            writeln!(code, "            du={:.4},  # decay constant", 1.0 / params.tau_m)?;
            writeln!(code, "            dv={:.4},  # voltage decay", 1.0 / params.tau_syn)?;
            writeln!(code, "            vth={:.1},  # threshold", params.v_thresh.abs() as i32)?;
            writeln!(code, "        )")?;
        }

        writeln!(code)?;
        writeln!(code, "        # Create connections")?;

        // Create connections
        for (i, conn) in config.connections.iter().enumerate() {
            let src_layer = config.layers.iter().find(|l| l.name == conn.source);
            let tgt_layer = config.layers.iter().find(|l| l.name == conn.target);

            if let (Some(src), Some(tgt)) = (src_layer, tgt_layer) {
                writeln!(code, "        # Connection: {} -> {}", conn.source, conn.target)?;

                if let Some(ref weights) = conn.weights {
                    writeln!(code, "        weights_{} = np.array([", i)?;
                    for row in weights {
                        let row_str: Vec<String> = row.iter().map(|w| format!("{:.4}", w)).collect();
                        writeln!(code, "            [{}],", row_str.join(", "))?;
                    }
                    writeln!(code, "        ])")?;
                } else {
                    writeln!(code, "        weights_{} = np.random.randn({}, {}) * 0.1", i, tgt.size, src.size)?;
                }

                writeln!(code, "        self.dense_{} = Dense(weights=weights_{})", i, i)?;
                writeln!(code, "        self.{}.s_out.connect(self.dense_{}.s_in)", conn.source, i)?;
                writeln!(code, "        self.dense_{}.a_out.connect(self.{}.a_in)", i, conn.target)?;
            }
        }

        writeln!(code)?;
        writeln!(code, "    def run(self, num_steps={}):", config.duration_ms as i32)?;
        writeln!(code, "        \"\"\"Run the network for specified timesteps.\"\"\"")?;
        writeln!(code, "        from lava.magma.core.run_configs import Loihi2HwCfg, Loihi2SimCfg")?;
        writeln!(code, "        from lava.magma.core.run_conditions import RunSteps")?;
        writeln!(code)?;
        writeln!(code, "        # Try hardware, fall back to simulation")?;
        writeln!(code, "        try:")?;
        writeln!(code, "            run_cfg = Loihi2HwCfg()")?;
        writeln!(code, "        except:")?;
        writeln!(code, "            run_cfg = Loihi2SimCfg()")?;
        writeln!(code)?;
        writeln!(code, "        self.{}.run(", config.layers[0].name)?;
        writeln!(code, "            condition=RunSteps(num_steps=num_steps),")?;
        writeln!(code, "            run_cfg=run_cfg")?;
        writeln!(code, "        )")?;
        writeln!(code)?;
        writeln!(code, "    def stop(self):")?;
        writeln!(code, "        \"\"\"Stop the network.\"\"\"")?;
        writeln!(code, "        self.{}.stop()", config.layers[0].name)?;
        writeln!(code)?;

        // Main block
        writeln!(code)?;
        writeln!(code, "if __name__ == '__main__':")?;
        writeln!(code, "    # Create and run network")?;
        writeln!(code, "    network = {}Network()", config.name)?;
        writeln!(code, "    network.run(num_steps={})", config.duration_ms as i32)?;
        writeln!(code, "    network.stop()")?;
        writeln!(code, "    print('Network execution complete')")?;

        Ok(code)
    }

    /// Export for SpiNNaker 2 (sPyNNaker).
    fn export_spinnaker(&self, config: &NetworkConfig) -> Result<String, NeuromorphicExportError> {
        let mut code = String::new();

        // Header
        writeln!(code, "# DPB Network Export for SpiNNaker 2")?;
        writeln!(code, "# Generated by dpb-export")?;
        writeln!(code, "# Framework: sPyNNaker")?;
        writeln!(code)?;

        // Imports
        writeln!(code, "import pyNN.spiNNaker as sim")?;
        writeln!(code, "import numpy as np")?;
        writeln!(code)?;

        // Setup
        writeln!(code, "# Initialize simulator")?;
        writeln!(code, "sim.setup(timestep={:.1})", config.dt)?;
        writeln!(code)?;

        // Create populations
        writeln!(code, "# Create populations")?;
        for layer in &config.layers {
            let params = &layer.params;
            let cell_type = layer.neuron_model.pynn_name();

            writeln!(code, "{} = sim.Population(", layer.name)?;
            writeln!(code, "    {},", layer.size)?;
            writeln!(code, "    sim.{}(", cell_type)?;
            writeln!(code, "        tau_m={:.1},", params.tau_m)?;
            writeln!(code, "        tau_syn_E={:.1},", params.tau_syn)?;
            writeln!(code, "        tau_syn_I={:.1},", params.tau_syn)?;
            writeln!(code, "        v_thresh={:.1},", params.v_thresh)?;
            writeln!(code, "        v_reset={:.1},", params.v_reset)?;
            writeln!(code, "        v_rest={:.1},", params.v_rest)?;
            writeln!(code, "        tau_refrac={:.1},", params.t_refrac)?;
            writeln!(code, "        cm={:.1},", params.cm)?;
            writeln!(code, "    ),")?;
            writeln!(code, "    label='{}'", layer.name)?;
            writeln!(code, ")")?;
            writeln!(code)?;
        }

        // Create projections
        writeln!(code, "# Create projections")?;
        for conn in &config.connections {
            let connector = match &conn.conn_type {
                ConnectionType::AllToAll => "sim.AllToAllConnector()".to_string(),
                ConnectionType::OneToOne => "sim.OneToOneConnector()".to_string(),
                ConnectionType::Random { probability } => {
                    format!("sim.FixedProbabilityConnector({})", probability)
                }
                ConnectionType::FromMatrix => "sim.FromListConnector(conn_list)".to_string(),
            };

            if let ConnectionType::FromMatrix = conn.conn_type {
                if let Some(ref weights) = conn.weights {
                    writeln!(code, "# Connection list for {} -> {}", conn.source, conn.target)?;
                    writeln!(code, "conn_list = [")?;
                    for (i, row) in weights.iter().enumerate() {
                        for (j, &w) in row.iter().enumerate() {
                            if w.abs() > 1e-6 {
                                writeln!(code, "    ({}, {}, {:.4}, {:.1}),", j, i, w, conn.delay)?;
                            }
                        }
                    }
                    writeln!(code, "]")?;
                }
            }

            writeln!(code, "proj_{}_{} = sim.Projection(", conn.source, conn.target)?;
            writeln!(code, "    {},", conn.source)?;
            writeln!(code, "    {},", conn.target)?;
            writeln!(code, "    {},"  , connector)?;
            writeln!(code, "    synapse_type=sim.StaticSynapse(weight=1.0, delay={:.1}),", conn.delay)?;
            writeln!(code, "    receptor_type='excitatory'")?;
            writeln!(code, ")")?;
            writeln!(code)?;
        }

        // Recording
        if self.include_recorder {
            writeln!(code, "# Set up recording")?;
            for layer in &config.layers {
                writeln!(code, "{}.record(['spikes', 'v'])", layer.name)?;
            }
            writeln!(code)?;
        }

        // Run simulation
        writeln!(code, "# Run simulation")?;
        writeln!(code, "sim.run({:.1})", config.duration_ms)?;
        writeln!(code)?;

        // Extract data
        if self.include_recorder {
            writeln!(code, "# Extract spike data")?;
            for layer in &config.layers {
                writeln!(code, "{}_spikes = {}.get_data('spikes').segments[0].spiketrains", layer.name, layer.name)?;
            }
            writeln!(code)?;
        }

        // End simulation
        writeln!(code, "# End simulation")?;
        writeln!(code, "sim.end()")?;
        writeln!(code)?;

        // Visualization
        if self.include_viz {
            writeln!(code, "# Visualization")?;
            writeln!(code, "import matplotlib.pyplot as plt")?;
            writeln!(code)?;
            writeln!(code, "fig, axes = plt.subplots({}, 1, figsize=(12, {}))", config.layers.len(), config.layers.len() * 3)?;
            for (i, layer) in config.layers.iter().enumerate() {
                writeln!(code, "for idx, train in enumerate({}_spikes):", layer.name)?;
                writeln!(code, "    axes[{}].scatter(train, [idx]*len(train), s=1)", i)?;
                writeln!(code, "axes[{}].set_ylabel('{}')", i, layer.name)?;
                writeln!(code, "axes[{}].set_xlim(0, {:.1})", i, config.duration_ms)?;
            }
            writeln!(code, "axes[-1].set_xlabel('Time (ms)')")?;
            writeln!(code, "plt.tight_layout()")?;
            writeln!(code, "plt.savefig('{}_raster.png')", config.name)?;
            writeln!(code, "plt.show()")?;
        }

        Ok(code)
    }

    /// Export for BrainScaleS-2 (hxtorch).
    fn export_brainscales(&self, config: &NetworkConfig) -> Result<String, NeuromorphicExportError> {
        let mut code = String::new();

        // Header
        writeln!(code, "# DPB Network Export for BrainScaleS-2")?;
        writeln!(code, "# Generated by dpb-export")?;
        writeln!(code, "# Framework: hxtorch")?;
        writeln!(code, "#")?;
        writeln!(code, "# Note: BrainScaleS-2 operates in accelerated analog mode")?;
        writeln!(code, "# (approximately 1000x faster than biological real-time)")?;
        writeln!(code)?;

        // Imports
        writeln!(code, "import torch")?;
        writeln!(code, "import hxtorch")?;
        writeln!(code, "import hxtorch.snn as snn")?;
        writeln!(code, "from hxtorch.snn import Synapse, Neuron")?;
        writeln!(code, "import numpy as np")?;
        writeln!(code)?;

        // Initialize hardware
        writeln!(code, "# Initialize BrainScaleS-2 hardware")?;
        writeln!(code, "hxtorch.init_hardware()")?;
        writeln!(code)?;

        // Network class
        writeln!(code, "class {}Network(torch.nn.Module):", config.name)?;
        writeln!(code, "    \"\"\"DPB-generated network for BrainScaleS-2.\"\"\"")?;
        writeln!(code)?;
        writeln!(code, "    def __init__(self):")?;
        writeln!(code, "        super().__init__()")?;
        writeln!(code)?;

        // Create layers
        for (i, layer) in config.layers.iter().enumerate() {
            if i == 0 {
                continue; // Input layer is handled separately
            }
            let prev_layer = &config.layers[i - 1];
            writeln!(code, "        # Layer: {}", layer.name)?;
            writeln!(code, "        self.synapse_{} = snn.Synapse(", layer.name)?;
            writeln!(code, "            in_features={},", prev_layer.size)?;
            writeln!(code, "            out_features={},", layer.size)?;
            writeln!(code, "        )")?;
            writeln!(code, "        self.neuron_{} = snn.Neuron(", layer.name)?;
            writeln!(code, "            size={},", layer.size)?;
            writeln!(code, "            leak={:.4},", 1.0 / layer.params.tau_m)?;
            writeln!(code, "            threshold={:.1},", layer.params.v_thresh.abs())?;
            writeln!(code, "        )")?;
            writeln!(code)?;
        }

        // Forward pass
        writeln!(code, "    def forward(self, spikes):")?;
        writeln!(code, "        \"\"\"Forward pass through the network.\"\"\"")?;
        writeln!(code, "        x = spikes")?;
        for (i, layer) in config.layers.iter().enumerate() {
            if i == 0 {
                continue;
            }
            writeln!(code, "        x = self.synapse_{}(x)", layer.name)?;
            writeln!(code, "        x = self.neuron_{}(x)", layer.name)?;
        }
        writeln!(code, "        return x")?;
        writeln!(code)?;

        // Main block
        writeln!(code)?;
        writeln!(code, "if __name__ == '__main__':")?;
        writeln!(code, "    # Create network")?;
        writeln!(code, "    network = {}Network()", config.name)?;
        writeln!(code)?;
        writeln!(code, "    # Create input spikes (example)")?;
        writeln!(code, "    num_timesteps = {}", (config.duration_ms / config.dt) as i32)?;
        writeln!(code, "    input_spikes = torch.rand(num_timesteps, {}) > 0.9", config.layers[0].size)?;
        writeln!(code, "    input_spikes = input_spikes.float()")?;
        writeln!(code)?;
        writeln!(code, "    # Run inference")?;
        writeln!(code, "    with torch.no_grad():")?;
        writeln!(code, "        output_spikes = network(input_spikes)")?;
        writeln!(code)?;
        writeln!(code, "    print(f'Output spike count: {{output_spikes.sum().item()}}')")?;
        writeln!(code)?;
        writeln!(code, "    # Release hardware")?;
        writeln!(code, "    hxtorch.release_hardware()")?;

        Ok(code)
    }

    /// Export generic PyNN code.
    fn export_pynn(&self, config: &NetworkConfig) -> Result<String, NeuromorphicExportError> {
        let mut code = String::new();

        writeln!(code, "# DPB Network Export - Generic PyNN")?;
        writeln!(code, "# Generated by dpb-export")?;
        writeln!(code, "#")?;
        writeln!(code, "# Compatible with: NEST, Brian2, Neuron (via PyNN)")?;
        writeln!(code)?;

        writeln!(code, "# Choose simulator backend:")?;
        writeln!(code, "# import pyNN.nest as sim")?;
        writeln!(code, "# import pyNN.brian2 as sim")?;
        writeln!(code, "# import pyNN.neuron as sim")?;
        writeln!(code, "import pyNN.nest as sim  # Default to NEST")?;
        writeln!(code, "import numpy as np")?;
        writeln!(code)?;

        writeln!(code, "sim.setup(timestep={:.2})", config.dt)?;
        writeln!(code)?;

        // Populations
        for layer in &config.layers {
            let params = &layer.params;
            writeln!(code, "{} = sim.Population({}, sim.IF_curr_exp(", layer.name, layer.size)?;
            writeln!(code, "    tau_m={:.1}, tau_syn_E={:.1}, tau_syn_I={:.1},", params.tau_m, params.tau_syn, params.tau_syn)?;
            writeln!(code, "    v_thresh={:.1}, v_reset={:.1}, v_rest={:.1}", params.v_thresh, params.v_reset, params.v_rest)?;
            writeln!(code, "), label='{}')", layer.name)?;
        }
        writeln!(code)?;

        // Projections
        for conn in &config.connections {
            writeln!(code, "sim.Projection({}, {}, sim.AllToAllConnector(),", conn.source, conn.target)?;
            writeln!(code, "    synapse_type=sim.StaticSynapse(weight=0.1, delay={:.1}))", conn.delay)?;
        }
        writeln!(code)?;

        // Record and run
        for layer in &config.layers {
            writeln!(code, "{}.record('spikes')", layer.name)?;
        }
        writeln!(code)?;
        writeln!(code, "sim.run({:.1})", config.duration_ms)?;
        writeln!(code, "sim.end()")?;

        Ok(code)
    }
}

/// Error type for neuromorphic export.
#[derive(Debug)]
pub enum NeuromorphicExportError {
    /// Write error.
    WriteError(std::fmt::Error),
    /// Invalid configuration.
    InvalidConfig(String),
    /// Unsupported feature for target.
    UnsupportedFeature(String),
}

impl From<std::fmt::Error> for NeuromorphicExportError {
    fn from(err: std::fmt::Error) -> Self {
        NeuromorphicExportError::WriteError(err)
    }
}

impl std::fmt::Display for NeuromorphicExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NeuromorphicExportError::WriteError(e) => write!(f, "Write error: {}", e),
            NeuromorphicExportError::InvalidConfig(msg) => write!(f, "Invalid config: {}", msg),
            NeuromorphicExportError::UnsupportedFeature(msg) => write!(f, "Unsupported: {}", msg),
        }
    }
}

impl std::error::Error for NeuromorphicExportError {}

#[cfg(test)]
#[allow(deprecated)] // exhaustive coverage of targets, including the legacy one
mod tests {
    use super::*;

    #[test]
    fn test_loihi_export() {
        let config = NetworkConfig::new(32, 64, 10);
        let exporter = NeuromorphicExporter::new(NeuromorphicTarget::Loihi2);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("from lava"));
        assert!(code.contains("LIF"));
    }

    #[test]
    fn test_spinnaker_export() {
        let config = NetworkConfig::new(32, 64, 10);
        let exporter = NeuromorphicExporter::new(NeuromorphicTarget::SpiNNaker2);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("pyNN.spiNNaker"));
        assert!(code.contains("sim.Population"));
    }

    #[test]
    fn test_brainscales_export() {
        let config = NetworkConfig::new(32, 64, 10);
        let exporter = NeuromorphicExporter::new(NeuromorphicTarget::BrainScaleS2);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("hxtorch"));
    }

    #[test]
    fn test_hardware_specs() {
        let specs = NeuromorphicTarget::Loihi2.specs();
        assert_eq!(specs.neurons_per_core, 8192);
        assert!(specs.supports_learning);
    }
}
