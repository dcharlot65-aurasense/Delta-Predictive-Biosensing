//! # Dendritic Computation Module
//!
//! This module implements multi-compartment neuron models with detailed
//! dendritic computation capabilities, including:
//!
//! - Multi-compartment cable theory
//! - Detailed ion channel dynamics
//! - Dendritic morphology (SWC format support)
//! - Active and passive dendritic integration
//! - Synaptic placement and integration
//! - Dendritic plasticity mechanisms
//!
//! ## Features
//!
//! ### Biophysical Realism
//! - Hodgkin-Huxley ion channels (Na⁺, K⁺, Ca²⁺)
//! - Voltage-gated and ligand-gated channels
//! - NMDA receptors with Mg²⁺ block
//! - Realistic morphology support
//!
//! ### Dendritic Computation
//! - Passive cable theory
//! - Active dendritic spikes
//! - NMDA plateau potentials
//! - Calcium spikes
//! - Backpropagating action potentials
//!
//! ### Plasticity
//! - Spike-timing-dependent plasticity (STDP)
//! - Branch-specific plasticity
//! - Heterosynaptic plasticity
//! - Metaplasticity
//!
//! ## Example
//!
//! ```rust
//! use dpb_neurons::dendritic::{MultiCompartmentNeuron, DendriticTree};
//!
//! // Create a multi-compartment neuron
//! let mut neuron = MultiCompartmentNeuron::new(10); // 10 compartments
//!
//! // Simulate with synaptic input
//! let spike = neuron.update(1.0); // 1ms timestep
//! ```

pub mod compartment;
pub mod morphology;
pub mod channels;
pub mod synapse;
pub mod integration;
pub mod plasticity;
pub mod multi_compartment;

// Re-export main types
pub use compartment::{Compartment, CompartmentConfig, CableParams};
pub use morphology::{DendriticTree, BranchNode, MorphologyData, SwcPoint};
pub use channels::{
    IonChannel, HodgkinHuxleyChannel, CalciumChannel, PotassiumChannel,
    NmdaReceptor, AmpaReceptor, GabaAReceptor, GabaBReceptor,
    ChannelType, GatingVariable,
};
pub use synapse::{
    DendriticSynapse, SynapseConfig, SynapseType, SynapticConductance,
};
pub use integration::{
    DendriticIntegration, PassiveIntegration, ActiveIntegration,
    NonlinearDendrites, CoincidenceDetection,
};
pub use plasticity::{
    DendriticPlasticity, DendriticStdp, BranchSpecificPlasticity,
    CompartmentPlasticity, Heterosynaptic, Metaplasticity,
};
pub use multi_compartment::{
    MultiCompartmentNeuron, NeuronConfig, BackpropagationConfig,
    NumericalSolver,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compartment_creation() {
        let config = CompartmentConfig::default();
        let comp = Compartment::new(config);
        assert!(comp.voltage() < 0.0); // Resting potential is negative
    }

    #[test]
    fn test_multi_compartment_creation() {
        let neuron = MultiCompartmentNeuron::new(5);
        assert_eq!(neuron.num_compartments(), 5);
    }

    #[test]
    fn test_dendritic_tree_creation() {
        let tree = DendriticTree::new();
        assert_eq!(tree.num_branches(), 0);
    }
}
