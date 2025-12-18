//! Learning algorithms for Spiking Neural Networks
//!
//! This module provides various learning rules and algorithms for training SNNs:
//!
//! - **Hebbian Learning** - Unsupervised learning based on correlated activity
//!   - STDP (Spike-Timing-Dependent Plasticity)
//!   - BCM (Bienenstock-Cooper-Munro) rule
//!   - Oja's rule
//!   - Covariance rule

pub mod hebbian;

// Re-export commonly used types
pub use hebbian::{
    BCMRule, CovarianceRule, HebbianLayer, HebbianRule, OjasRule, SynapticTrace, STDP,
};
