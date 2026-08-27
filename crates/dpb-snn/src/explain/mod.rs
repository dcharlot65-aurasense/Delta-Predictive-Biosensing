//! Explainability and interpretability tools for spiking neural networks
//!
//! Provides methods for understanding model predictions including
//! spike importance, attention visualization, and feature attribution.

pub mod attention;
pub mod attribution;
pub mod importance;
pub mod visualization;

pub use attention::{AttentionMap, SpatialAttention, TemporalAttention};
pub use attribution::{FeatureAttribution, GradientAttribution, IntegratedGradients, SpikeSHAP};
pub use importance::{
    LayerImportance, NeuronImportance, SpikeImportance, aggregate_to_neurons,
    compute_importance_by_perturbation, compute_spike_importance,
};
pub use visualization::{ExplanationVisualizer, HeatmapData, export_explanation_json};
