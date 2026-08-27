//! Tests for neuromorphic hardware export functionality.
//!
//! These tests verify the generation of code for Intel Loihi 2 (Lava),
//! SpiNNaker 2 (sPyNNaker), BrainScaleS-2 (hxtorch), and PyNN.

#[cfg(test)]
mod neuromorphic_export_tests {
    

    #[allow(clippy::upper_case_acronyms)] // domain notation
    /// Neuromorphic target platforms.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
    enum NeuromorphicTarget {
        Loihi2,
        SpiNNaker2,
        BrainScaleS2,
        PyNN,
    }

    #[allow(clippy::upper_case_acronyms)] // domain notation
    /// Neuron model types.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
    enum NeuronModel {
        LIF,   // Leaky Integrate-and-Fire
        CUBA,  // Current-Based
        COBA,  // Conductance-Based
        ALIF,  // Adaptive LIF
        AdEx,  // Adaptive Exponential
    }

    impl NeuronModel {
        fn pynn_name(&self) -> &'static str {
            match self {
                NeuronModel::LIF => "IF_curr_exp",
                NeuronModel::CUBA => "IF_curr_exp",
                NeuronModel::COBA => "IF_cond_exp",
                NeuronModel::ALIF => "IF_curr_exp_adapt",
                NeuronModel::AdEx => "EIF_cond_exp_isfa_ista",
            }
        }

        fn lava_class(&self) -> &'static str {
            match self {
                NeuronModel::LIF => "LIF",
                NeuronModel::CUBA => "CubaLIF",
                NeuronModel::COBA => "CobaLIF",
                NeuronModel::ALIF => "AdaptiveLIF",
                NeuronModel::AdEx => "AdExLIF",
            }
        }
    }

    #[test]
    fn test_neuron_model_names() {
        assert_eq!(NeuronModel::LIF.pynn_name(), "IF_curr_exp");
        assert_eq!(NeuronModel::COBA.pynn_name(), "IF_cond_exp");
        assert_eq!(NeuronModel::LIF.lava_class(), "LIF");
    }

    /// Network configuration.
    #[derive(Debug, Clone)]
    struct NetworkConfig {
        input_size: usize,
        hidden_sizes: Vec<usize>,
        output_size: usize,
        neuron_model: NeuronModel,
        time_step_ms: f32,
        simulation_time_ms: f32,
    }

    impl Default for NetworkConfig {
        fn default() -> Self {
            Self {
                input_size: 256,
                hidden_sizes: vec![128, 64],
                output_size: 10,
                neuron_model: NeuronModel::LIF,
                time_step_ms: 1.0,
                simulation_time_ms: 100.0,
            }
        }
    }

    #[test]
    fn test_network_config() {
        let config = NetworkConfig::default();
        assert_eq!(config.input_size, 256);
        assert_eq!(config.hidden_sizes.len(), 2);
        assert_eq!(config.neuron_model, NeuronModel::LIF);
    }

    /// LIF neuron parameters.
    #[derive(Debug, Clone)]
    struct LIFParams {
        tau_mem: f32,      // Membrane time constant (ms)
        tau_syn: f32,      // Synaptic time constant (ms)
        v_thresh: f32,     // Threshold voltage (mV)
        v_reset: f32,      // Reset voltage (mV)
        v_rest: f32,       // Resting potential (mV)
        refractory_ms: f32, // Refractory period (ms)
    }

    impl Default for LIFParams {
        fn default() -> Self {
            Self {
                tau_mem: 20.0,
                tau_syn: 5.0,
                v_thresh: -50.0,
                v_reset: -70.0,
                v_rest: -65.0,
                refractory_ms: 2.0,
            }
        }
    }

    #[test]
    fn test_lif_params() {
        let params = LIFParams::default();
        assert!(params.v_thresh > params.v_reset);
        assert!(params.v_rest > params.v_reset);
    }

    /// Simulated neuromorphic exporter.
    struct NeuromorphicExporter {
        target: NeuromorphicTarget,
        config: NetworkConfig,
        params: LIFParams,
    }

    impl NeuromorphicExporter {
        fn new(target: NeuromorphicTarget, config: NetworkConfig) -> Self {
            Self {
                target,
                config,
                params: LIFParams::default(),
            }
        }

        fn generate_lava_code(&self) -> String {
            let mut code = String::new();

            code.push_str("# Intel Loihi 2 - Lava Implementation\n");
            code.push_str("import numpy as np\n");
            code.push_str("from lava.proc.lif.process import LIF\n");
            code.push_str("from lava.proc.dense.process import Dense\n");
            code.push_str("from lava.proc.io.source import RingBuffer as Source\n");
            code.push_str("from lava.proc.io.sink import RingBuffer as Sink\n");
            code.push_str("from lava.magma.core.run_configs import Loihi2HwCfg\n");
            code.push_str("from lava.magma.core.run_conditions import RunSteps\n\n");

            // Network configuration
            code.push_str("# Network Configuration\n");
            code.push_str(&format!("INPUT_SIZE = {}\n", self.config.input_size));
            code.push_str(&format!("HIDDEN_SIZES = {:?}\n", self.config.hidden_sizes));
            code.push_str(&format!("OUTPUT_SIZE = {}\n", self.config.output_size));
            code.push_str(&format!("TIME_STEPS = {}\n\n",
                (self.config.simulation_time_ms / self.config.time_step_ms) as u32));

            // LIF parameters
            code.push_str("# LIF Neuron Parameters\n");
            code.push_str("lif_params = {\n");
            code.push_str(&format!("    'du': {},  # Decay constant\n",
                (1.0 / self.params.tau_mem * 4095.0) as u32));
            code.push_str(&format!("    'dv': {},  # Voltage decay\n",
                (1.0 / self.params.tau_syn * 4095.0) as u32));
            code.push_str(&format!("    'vth': {},  # Threshold\n",
                ((self.params.v_thresh - self.params.v_rest) * 10.0) as i32));
            code.push_str("}\n\n");

            // Build network
            code.push_str("# Build Network\n");
            code.push_str("source = Source(data=input_spikes)\n\n");

            let mut prev_size = self.config.input_size;
            for (i, &size) in self.config.hidden_sizes.iter().enumerate() {
                code.push_str(&format!(
                    "lif_{} = LIF(shape=({},), **lif_params)\n",
                    i, size
                ));
                code.push_str(&format!(
                    "dense_{} = Dense(weights=np.random.randn({}, {}) * 0.1)\n",
                    i, size, prev_size
                ));
                prev_size = size;
            }

            code.push_str(&format!(
                "\noutput = LIF(shape=({},), **lif_params)\n",
                self.config.output_size
            ));
            code.push_str("sink = Sink(shape=output.shape, buffer=TIME_STEPS)\n\n");

            // Connect layers
            code.push_str("# Connect Layers\n");
            code.push_str("source.out_ports.s_out.connect(dense_0.in_ports.s_in)\n");
            for i in 0..self.config.hidden_sizes.len() {
                code.push_str(&format!(
                    "dense_{}.out_ports.a_out.connect(lif_{}.in_ports.a_in)\n",
                    i, i
                ));
                if i < self.config.hidden_sizes.len() - 1 {
                    code.push_str(&format!(
                        "lif_{}.out_ports.s_out.connect(dense_{}.in_ports.s_in)\n",
                        i, i + 1
                    ));
                }
            }
            code.push_str(&format!(
                "lif_{}.out_ports.s_out.connect(output.in_ports.a_in)\n",
                self.config.hidden_sizes.len() - 1
            ));
            code.push_str("output.out_ports.s_out.connect(sink.in_ports.a_in)\n\n");

            // Run
            code.push_str("# Run on Loihi 2\n");
            code.push_str("run_cfg = Loihi2HwCfg()\n");
            code.push_str("source.run(condition=RunSteps(TIME_STEPS), run_cfg=run_cfg)\n");
            code.push_str("output_spikes = sink.data.get()\n");
            code.push_str("source.stop()\n");

            code
        }

        fn generate_pynn_code(&self) -> String {
            let mut code = String::new();

            code.push_str("# PyNN Implementation (Portable)\n");
            code.push_str("import pyNN.spiNNaker as sim  # or pyNN.nest, pyNN.brian2\n");
            code.push_str("import numpy as np\n\n");

            // Setup
            code.push_str(&format!("sim.setup(timestep={})\n\n", self.config.time_step_ms));

            // Parameters
            code.push_str("# LIF Parameters\n");
            code.push_str("cell_params = {\n");
            code.push_str(&format!("    'tau_m': {},\n", self.params.tau_mem));
            code.push_str(&format!("    'tau_syn_E': {},\n", self.params.tau_syn));
            code.push_str(&format!("    'tau_syn_I': {},\n", self.params.tau_syn));
            code.push_str(&format!("    'v_thresh': {},\n", self.params.v_thresh));
            code.push_str(&format!("    'v_reset': {},\n", self.params.v_reset));
            code.push_str(&format!("    'v_rest': {},\n", self.params.v_rest));
            code.push_str(&format!("    'tau_refrac': {},\n", self.params.refractory_ms));
            code.push_str("}\n\n");

            // Create populations
            code.push_str("# Create Populations\n");
            code.push_str(&format!(
                "input_pop = sim.Population({}, sim.SpikeSourceArray(spike_times=[]))\n",
                self.config.input_size
            ));

            // The PyNN generator emits Populations only; prev_size used to be
            // tracked for the Projections between them, which it does not emit.
            for (i, &size) in self.config.hidden_sizes.iter().enumerate() {
                code.push_str(&format!(
                    "hidden_{} = sim.Population({}, sim.{}, cell_params)\n",
                    i, size, self.config.neuron_model.pynn_name()
                ));
            }

            code.push_str(&format!(
                "output_pop = sim.Population({}, sim.{}, cell_params)\n\n",
                self.config.output_size,
                self.config.neuron_model.pynn_name()
            ));

            // Create projections
            code.push_str("# Create Projections\n");
            code.push_str("sim.Projection(input_pop, hidden_0,\n");
            code.push_str("    sim.AllToAllConnector(),\n");
            code.push_str("    sim.StaticSynapse(weight=0.1))\n");

            for i in 0..self.config.hidden_sizes.len() - 1 {
                code.push_str(&format!(
                    "sim.Projection(hidden_{}, hidden_{},\n",
                    i, i + 1
                ));
                code.push_str("    sim.AllToAllConnector(),\n");
                code.push_str("    sim.StaticSynapse(weight=0.1))\n");
            }

            code.push_str(&format!(
                "sim.Projection(hidden_{}, output_pop,\n",
                self.config.hidden_sizes.len() - 1
            ));
            code.push_str("    sim.AllToAllConnector(),\n");
            code.push_str("    sim.StaticSynapse(weight=0.1))\n\n");

            // Record and run
            code.push_str("# Record and Run\n");
            code.push_str("output_pop.record('spikes')\n");
            code.push_str(&format!("sim.run({})\n\n", self.config.simulation_time_ms));

            code.push_str("# Get Results\n");
            code.push_str("spikes = output_pop.get_data('spikes')\n");
            code.push_str("sim.end()\n");

            code
        }

        fn generate_hxtorch_code(&self) -> String {
            let mut code = String::new();

            code.push_str("# BrainScaleS-2 - hxtorch Implementation\n");
            code.push_str("import torch\n");
            code.push_str("import hxtorch\n");
            code.push_str("import hxtorch.snn as snn\n\n");

            // Hardware calibration
            code.push_str("# Initialize Hardware\n");
            code.push_str("hxtorch.init_hardware()\n\n");

            // Network class
            code.push_str("class SpikingNetwork(torch.nn.Module):\n");
            code.push_str("    def __init__(self):\n");
            code.push_str("        super().__init__()\n\n");

            code.push_str(&format!(
                "        self.fc1 = snn.Synapse({}, {})\n",
                self.config.input_size, self.config.hidden_sizes[0]
            ));
            code.push_str(&format!(
                "        self.lif1 = snn.LIF({})\n",
                self.config.hidden_sizes[0]
            ));

            for i in 0..self.config.hidden_sizes.len() - 1 {
                code.push_str(&format!(
                    "        self.fc{} = snn.Synapse({}, {})\n",
                    i + 2, self.config.hidden_sizes[i], self.config.hidden_sizes[i + 1]
                ));
                code.push_str(&format!(
                    "        self.lif{} = snn.LIF({})\n",
                    i + 2, self.config.hidden_sizes[i + 1]
                ));
            }

            code.push_str(&format!(
                "        self.fc_out = snn.Synapse({}, {})\n",
                self.config.hidden_sizes.last().unwrap(),
                self.config.output_size
            ));
            code.push_str(&format!(
                "        self.lif_out = snn.LIF({})\n\n",
                self.config.output_size
            ));

            // Forward method
            code.push_str("    def forward(self, x):\n");
            code.push_str("        x = self.lif1(self.fc1(x))\n");
            for i in 0..self.config.hidden_sizes.len() - 1 {
                code.push_str(&format!(
                    "        x = self.lif{}(self.fc{}(x))\n",
                    i + 2, i + 2
                ));
            }
            code.push_str("        x = self.lif_out(self.fc_out(x))\n");
            code.push_str("        return x\n\n");

            // Usage
            code.push_str("# Create and run network\n");
            code.push_str("model = SpikingNetwork()\n");
            code.push_str(&format!(
                "input_spikes = torch.zeros({}, 1, {})\n",
                (self.config.simulation_time_ms / self.config.time_step_ms) as u32,
                self.config.input_size
            ));
            code.push_str("with hxtorch.measure():\n");
            code.push_str("    output = model(input_spikes)\n");

            code
        }

        fn export(&self) -> String {
            match self.target {
                NeuromorphicTarget::Loihi2 => self.generate_lava_code(),
                NeuromorphicTarget::PyNN | NeuromorphicTarget::SpiNNaker2 => self.generate_pynn_code(),
                NeuromorphicTarget::BrainScaleS2 => self.generate_hxtorch_code(),
            }
        }
    }

    #[test]
    fn test_lava_export() {
        let config = NetworkConfig::default();
        let exporter = NeuromorphicExporter::new(NeuromorphicTarget::Loihi2, config);
        let code = exporter.export();

        assert!(code.contains("from lava.proc.lif.process import LIF"));
        assert!(code.contains("Loihi2HwCfg"));
        assert!(code.contains("Dense"));
    }

    #[test]
    fn test_pynn_export() {
        let config = NetworkConfig::default();
        let exporter = NeuromorphicExporter::new(NeuromorphicTarget::PyNN, config);
        let code = exporter.export();

        assert!(code.contains("import pyNN"));
        assert!(code.contains("sim.Population"));
        assert!(code.contains("sim.Projection"));
    }

    #[test]
    fn test_hxtorch_export() {
        let config = NetworkConfig::default();
        let exporter = NeuromorphicExporter::new(NeuromorphicTarget::BrainScaleS2, config);
        let code = exporter.export();

        assert!(code.contains("import hxtorch"));
        assert!(code.contains("snn.LIF"));
        assert!(code.contains("hxtorch.init_hardware"));
    }

    /// Test weight quantization for neuromorphic hardware.
    #[test]
    fn test_weight_quantization() {
        fn quantize_weights_loihi(weights: &[f32], bits: u32) -> Vec<i32> {
            let max_val = (1 << (bits - 1)) - 1;
            let min_val = -(1 << (bits - 1));

            let w_max = weights.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let w_min = weights.iter().cloned().fold(f32::INFINITY, f32::min);
            let scale = if w_max != w_min {
                (max_val as f32) / w_max.max(-w_min)
            } else {
                1.0
            };

            weights
                .iter()
                .map(|&w| {
                    let q = (w * scale).round() as i32;
                    q.clamp(min_val, max_val)
                })
                .collect()
        }

        let weights = vec![-1.0f32, -0.5, 0.0, 0.5, 1.0];
        let quantized = quantize_weights_loihi(&weights, 8);

        assert_eq!(quantized.len(), 5);
        assert!(quantized[0] < 0); // -1.0 should be negative
        assert_eq!(quantized[2], 0); // 0.0 should be 0
        assert!(quantized[4] > 0); // 1.0 should be positive
    }

    /// Test spike encoding format.
    #[test]
    fn test_spike_encoding() {
        #[derive(Debug, Clone)]
        #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
        struct SpikeEvent {
            time: f32,
            neuron_id: u32,
        }

        fn encode_rate_spikes(rates: &[f32], duration_ms: f32, dt_ms: f32) -> Vec<SpikeEvent> {
            let mut spikes = Vec::new();
            let steps = (duration_ms / dt_ms) as u32;

            for (neuron_id, &rate) in rates.iter().enumerate() {
                let spike_prob = rate * dt_ms / 1000.0;
                for step in 0..steps {
                    if rand_simplified() < spike_prob {
                        spikes.push(SpikeEvent {
                            time: step as f32 * dt_ms,
                            neuron_id: neuron_id as u32,
                        });
                    }
                }
            }

            spikes
        }

        fn rand_simplified() -> f32 {
            // Simplified random for testing
            0.01
        }

        let rates = vec![100.0, 50.0, 200.0]; // Hz
        let spikes = encode_rate_spikes(&rates, 100.0, 1.0);

        // Should generate some spikes
        assert!(!spikes.is_empty() || rates.iter().all(|&r| r < 10.0));
    }

    /// Test hardware constraints.
    #[test]
    fn test_hardware_constraints() {
        #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
        struct HardwareSpecs {
            max_neurons: u32,
            max_synapses: u32,
            weight_bits: u32,
            max_fan_in: u32,
            max_fan_out: u32,
        }

        fn get_specs(target: NeuromorphicTarget) -> HardwareSpecs {
            match target {
                NeuromorphicTarget::Loihi2 => HardwareSpecs {
                    max_neurons: 1_000_000,
                    max_synapses: 120_000_000,
                    weight_bits: 8,
                    max_fan_in: 8192,
                    max_fan_out: 8192,
                },
                NeuromorphicTarget::SpiNNaker2 => HardwareSpecs {
                    max_neurons: 200_000,
                    max_synapses: 1_000_000_000,
                    weight_bits: 16,
                    max_fan_in: 10000,
                    max_fan_out: 10000,
                },
                NeuromorphicTarget::BrainScaleS2 => HardwareSpecs {
                    max_neurons: 512,
                    max_synapses: 131_072,
                    weight_bits: 6,
                    max_fan_in: 256,
                    max_fan_out: 256,
                },
                NeuromorphicTarget::PyNN => HardwareSpecs {
                    max_neurons: u32::MAX,
                    max_synapses: u32::MAX,
                    weight_bits: 32,
                    max_fan_in: u32::MAX,
                    max_fan_out: u32::MAX,
                },
            }
        }

        let loihi = get_specs(NeuromorphicTarget::Loihi2);
        assert_eq!(loihi.max_neurons, 1_000_000);
        assert_eq!(loihi.weight_bits, 8);

        let brainscales = get_specs(NeuromorphicTarget::BrainScaleS2);
        assert_eq!(brainscales.max_neurons, 512);
    }
}
