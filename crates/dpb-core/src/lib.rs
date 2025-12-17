//! # DPB Core Library
//!
//! Core types, traits, and infrastructure for the Delta-Predictive Biosensing Framework.
//!
//! This is the foundation crate that all other DPB crates depend on. It provides:
//!
//! - **Core Types**: Data structures for spike events, time series, ground truth, etc.
//! - **Traits**: Abstract interfaces for encoders, networks, datasets, and more
//! - **GPU Infrastructure**: WebGPU-based compute acceleration
//! - **Signal Processing**: Filters, FFT, resampling, and analysis tools
//! - **Math Utilities**: Statistical functions, interpolation, and numerical methods
//! - **Configuration**: Structured configuration for all framework components
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_core::{types::*, error::Result};
//!
//! fn example() -> Result<()> {
//!     // Create a spike event
//!     let event = SpikeEvent::new(0.001, 5, 1, 1.0);
//!
//!     // Build a spike train
//!     let mut train = SpikeTrain::new(10);
//!     train.add_event(event);
//!
//!     // Create time series data
//!     let data = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
//!     let ts = TimeSeries::new(data, 1000.0)?;
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Modules
//!
//! - [`types`] - Core data types (SpikeEvent, SpikeTrain, TimeSeries, etc.)
//! - [`traits`] - Framework trait definitions
//! - [`error`] - Error types and Result alias
//! - [`config`] - Configuration structures
//! - [`tensor`] - Batched spike tensor operations
//! - [`gpu`] - GPU infrastructure and utilities
//! - [`signal`] - Signal processing tools
//! - [`math`] - Mathematical and statistical utilities

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod config;
pub mod error;
pub mod gpu;
pub mod math;
pub mod metrics;
pub mod signal;
pub mod tensor;
pub mod traits;
pub mod types;

// Re-export commonly used items
pub use error::{DpbError, Result};

// Re-export core types at crate root for convenience
pub use types::{
    Context, GroundTruth, Modality, SignalBuffer, SignalQuality, SpikeEvent, SpikeTrain, TimeSeries,
};

// Re-export core traits at crate root
pub use traits::{
    Configurable, ConvergenceAnalyzer, Dataset, EventEncoder, FeatureExtractor, GpuKernel,
    HardwareExporter, LossFunction, MembraneDynamics, Metric, Optimizer, PopulationTemplate,
    SNNNetwork, Signal, SpikingLayer, SurrogateGradient, SynapticModel, SyntheticGenerator,
    PowerEstimator,
};

// Re-export tensor types
pub use tensor::SpikeTensor;

// Implement Signal trait for SignalBuffer
impl traits::Signal for types::SignalBuffer {
    fn samples(&self) -> &[f32] {
        &self.data
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn channels(&self) -> usize {
        self.num_channels
    }
}

/// Prelude module for convenient imports.
pub mod prelude {
    pub use crate::config::*;
    pub use crate::error::{DpbError, Result};
    pub use crate::metrics::MetricTrait;
    pub use crate::traits::*;
    pub use crate::types::*;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        use prelude::*;

        // Test that prelude imports work
        let _result: Result<()> = Ok(());
        let _event = SpikeEvent::new(1.0, 0, 1, 1.0);
    }

    #[test]
    fn test_crate_structure() {
        // Verify all modules are accessible
        let _error_module = error::DpbError::Other("test".to_string());
        let _config = config::DpbConfig::default();
    }
}
