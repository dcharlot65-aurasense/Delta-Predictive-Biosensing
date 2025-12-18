//! Explainability and interpretability tools for spiking neural networks
//!
//! Provides methods for understanding model predictions including
//! spike importance, attention visualization, and feature attribution.

pub mod importance;
pub mod attention;
pub mod attribution;
pub mod visualization;

pub use importance::{SpikeImportance, NeuronImportance, LayerImportance, compute_spike_importance, compute_importance_by_perturbation, aggregate_to_neurons};
pub use attention::{AttentionMap, TemporalAttention, SpatialAttention};
pub use attribution::{FeatureAttribution, GradientAttribution, IntegratedGradients, SpikeSHAP};
pub use visualization::{ExplanationVisualizer, HeatmapData, export_explanation_json};
