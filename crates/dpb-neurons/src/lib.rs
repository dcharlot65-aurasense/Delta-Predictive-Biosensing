//! # DPB Neurons - Neuron Models for Spiking Neural Networks
//!
//! This crate implements 19 different neuron models for the Delta-Predictive
//! Biosensing Framework, ranging from simple integrate-and-fire to complex
//! biophysical models.
//!
//! ## Neuron Models
//!
//! ### Leaky Integrate-and-Fire (LIF) Variants (7 models)
//! - [`IfNeuron`](lif::IfNeuron) - Simple integrate-and-fire
//! - [`LifNeuron`](lif::LifNeuron) - Standard leaky integrate-and-fire
//! - [`ClifNeuron`](lif::ClifNeuron) - Current-based LIF with synaptic filtering
//! - [`AlifNeuron`](lif::AlifNeuron) - Adaptive LIF with threshold adaptation
//! - [`ElifNeuron`](lif::ElifNeuron) - Exponential LIF with smooth spike initiation
//! - [`QlifNeuron`](lif::QlifNeuron) - Quadratic LIF with parabolic nonlinearity
//! - [`GlifNeuron`](lif::GlifNeuron) - Generalized LIF with multiple adaptation variables
//!
//! ### Phenomenological Models (1 model)
//! - [`IzhikevichNeuron`](izhikevich::IzhikevichNeuron) - Efficient model with diverse dynamics
//!
//! ### Adaptive Models (2 models)
//! - [`AdExNeuron`](adex::AdExNeuron) - Adaptive exponential integrate-and-fire
//! - [`CalciumNeuron`](calcium::CalciumNeuron) - Calcium-based adaptation
//!
//! ### Biophysical Models (3 models)
//! - [`HodgkinHuxleyNeuron`](hodgkin_huxley::HodgkinHuxleyNeuron) - Full conductance-based model
//! - [`FitzHughNagumoNeuron`](hodgkin_huxley::FitzHughNagumoNeuron) - Simplified 2D model
//! - [`MorrisLecarNeuron`](hodgkin_huxley::MorrisLecarNeuron) - Ca²⁺ and K⁺ dynamics
//!
//! ### Specialized Models (4 models)
//! - [`SrmNeuron`](srm::SrmNeuron) - Spike response model with kernels
//! - [`StochasticLifNeuron`](stochastic::StochasticLifNeuron) - LIF with noise injection
//! - [`RecurrentNeuron`](recurrent::RecurrentNeuron) - Neuron with self-connection
//!
//! ### Hardware-Optimized Models (3 models)
//! - [`XyloLifNeuron`](hardware::XyloLifNeuron) - SynSense Xylo chip compatible
//! - [`PulsarLifNeuron`](hardware::PulsarLifNeuron) - Generic neuromorphic hardware
//! - [`QuantizedLifNeuron`](hardware::QuantizedLifNeuron) - Fixed-point for edge devices
//!
//! ## Surrogate Gradients (6 functions)
//!
//! For backpropagation through spiking neurons:
//! - [`FastSigmoid`](surrogate::FastSigmoid)
//! - [`Arctan`](surrogate::Arctan)
//! - [`Triangular`](surrogate::Triangular)
//! - [`SuperSpike`](surrogate::SuperSpike)
//! - [`MultiGaussian`](surrogate::MultiGaussian)
//! - [`StraightThroughEstimator`](surrogate::StraightThroughEstimator)
//!
//! ## Examples
//!
//! ### Basic LIF Neuron
//!
//! ```rust
//! use dpb_neurons::lif::{LifNeuron, LifConfig};
//! use dpb_neurons::traits::NeuronModel;
//!
//! let config = LifConfig::default();
//! let mut neuron = LifNeuron::new(config);
//!
//! // Simulate for 100ms
//! for _ in 0..100 {
//!     let spiked = neuron.update(10.0, 1.0); // 10nA input, 1ms timestep
//!     if spiked {
//!         println!("Spike!");
//!     }
//! }
//! ```
//!
//! ### Izhikevich with Presets
//!
//! ```rust
//! use dpb_neurons::izhikevich::IzhikevichNeuron;
//! use dpb_neurons::traits::{NeuronModel, NeuronPreset};
//!
//! let mut neuron = IzhikevichNeuron::from_preset(NeuronPreset::FastSpiking);
//! let spiked = neuron.update(15.0, 1.0);
//! ```
//!
//! ### Batch Processing
//!
//! ```rust
//! use dpb_neurons::batch::BatchLifLayer;
//! use dpb_neurons::lif::LifConfig;
//! use ndarray::Array1;
//!
//! let config = LifConfig::default();
//! let mut layer = BatchLifLayer::new(100, config);
//!
//! let inputs = Array1::from_elem(100, 10.0);
//! let spikes = layer.update(&inputs, 1.0);
//! ```
//!
//! ### GPU Acceleration
//!
//! ```rust
//! use dpb_neurons::gpu::{GpuNeuronKernel, LIF_SHADER};
//!
//! // Get WGSL shader for GPU compute
//! let kernel = GpuNeuronKernel::Lif;
//! let shader_source = kernel.shader_source();
//! ```

pub mod traits;

// Core neuron models
pub mod lif;
pub mod izhikevich;
pub mod adex;
pub mod hodgkin_huxley;
pub mod srm;
pub mod calcium;
pub mod stochastic;
pub mod recurrent;
pub mod hardware;

// Reservoir computing
pub mod reservoir;

// Training support
pub mod surrogate;

// Batch and GPU operations
pub mod batch;
pub mod gpu;

// Re-export commonly used types
pub use traits::{
    MembraneDynamics, NeuronConfig, NeuronModel, NeuronPreset, NeuronState, SpikeEvent,
    SynapticInput,
};

// Re-export LIF variants
pub use lif::{
    AlifNeuron, ClifNeuron, ElifNeuron, GlifNeuron, IfNeuron, LifNeuron, QlifNeuron,
};

// Re-export other models
pub use adex::AdExNeuron;
pub use calcium::CalciumNeuron;
pub use hodgkin_huxley::{FitzHughNagumoNeuron, HodgkinHuxleyNeuron, MorrisLecarNeuron};
pub use izhikevich::IzhikevichNeuron;
pub use recurrent::RecurrentNeuron;
pub use srm::SrmNeuron;
pub use stochastic::StochasticLifNeuron;

// Re-export hardware models
pub use hardware::{PulsarLifNeuron, QuantizedLifNeuron, XyloLifNeuron};

// Re-export surrogate functions
pub use surrogate::{
    Arctan, FastSigmoid, MultiGaussian, StraightThroughEstimator, SuperSpike, SurrogateFunction,
    SurrogateGradient, Triangular,
};

// Re-export batch types
pub use batch::{BatchLifLayer, BatchNetwork, BatchNeuronLayer, PopulationStats};

// Re-export GPU types
pub use gpu::{
    GpuAlifNeuron, GpuIzhikevichNeuron, GpuLifNeuron, GpuNeuronKernel, GpuSimParams,
    ALIF_SHADER, ELIF_SHADER, IZHIKEVICH_SHADER, LIF_SHADER,
};

// Re-export reservoir computing types
pub use reservoir::{EchoStateNetwork, LiquidStateMachine, SparsityPattern};

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::traits::{MembraneDynamics, NeuronConfig, NeuronModel, NeuronPreset};

    // Most common neuron types
    pub use crate::adex::AdExNeuron;
    pub use crate::izhikevich::IzhikevichNeuron;
    pub use crate::lif::{AlifNeuron, LifNeuron};

    // Batch processing
    pub use crate::batch::{BatchLifLayer, BatchNeuronLayer};

    // Surrogate gradients
    pub use crate::surrogate::{FastSigmoid, SurrogateGradient};
}

/// Get the version of this crate.
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Get the total number of neuron models implemented.
pub const NEURON_MODEL_COUNT: usize = 19;

/// Get the total number of surrogate gradient functions.
pub const SURROGATE_FUNCTION_COUNT: usize = 6;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }

    #[test]
    fn test_constants() {
        assert_eq!(NEURON_MODEL_COUNT, 19);
        assert_eq!(SURROGATE_FUNCTION_COUNT, 6);
    }

    #[test]
    fn test_all_models_importable() {
        // LIF variants (7)
        let _if = IfNeuron::new(lif::IfConfig::default());
        let _lif = LifNeuron::new(lif::LifConfig::default());
        let _clif = ClifNeuron::new(lif::ClifConfig::default());
        let _alif = AlifNeuron::new(lif::AlifConfig::default());
        let _elif = ElifNeuron::new(lif::ElifConfig::default());
        let _qlif = QlifNeuron::new(lif::QlifConfig::default());
        let _glif = GlifNeuron::new(lif::GlifConfig::default());

        // Izhikevich (1)
        let _izh = IzhikevichNeuron::new(izhikevich::IzhikevichConfig::default());

        // AdEx (1)
        let _adex = AdExNeuron::new(adex::AdExConfig::default());

        // Calcium (1)
        let _ca = CalciumNeuron::new(calcium::CalciumConfig::default());

        // Biophysical (3)
        let _hh = HodgkinHuxleyNeuron::new(hodgkin_huxley::HodgkinHuxleyConfig::default());
        let _fhn = FitzHughNagumoNeuron::new(hodgkin_huxley::FitzHughNagumoConfig::default());
        let _ml = MorrisLecarNeuron::new(hodgkin_huxley::MorrisLecarConfig::default());

        // SRM (1)
        let _srm = SrmNeuron::new(srm::SrmConfig::default());

        // Stochastic (1)
        let _stoch = StochasticLifNeuron::new(stochastic::StochasticLifConfig::default());

        // Recurrent (1)
        let _rec = RecurrentNeuron::new(recurrent::RecurrentNeuronConfig::default());

        // Hardware (3)
        let _xylo = XyloLifNeuron::new(hardware::XyloLifConfig::default());
        let _pulsar = PulsarLifNeuron::new(hardware::PulsarLifConfig::default());
        let _quant = QuantizedLifNeuron::new(hardware::QuantizedLifConfig::default());

        // Total: 19 models
    }

    #[test]
    fn test_all_surrogates_importable() {
        let _fast_sig = FastSigmoid::default();
        let _arctan = Arctan::default();
        let _tri = Triangular::default();
        let _super_spike = SuperSpike::default();
        let _gaussian = MultiGaussian::default();
        let _ste = StraightThroughEstimator::default();

        // Total: 6 surrogate functions
    }

    #[test]
    fn test_prelude() {
        use prelude::*;

        let _lif = LifNeuron::new(lif::LifConfig::default());
        let _adex = AdExNeuron::new(adex::AdExConfig::default());
        let _surrogate: Box<dyn SurrogateGradient> = Box::new(FastSigmoid::default());
    }
}
