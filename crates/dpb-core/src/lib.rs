//! # dpb-core - Core Types and Signal Processing
//!
//! The foundational crate for the Delta-Predictive Biosensing framework.
//!
//! ## Features
//!
//! - **Signal Processing**: FFT, filtering, resampling, normalization
//! - **Domain-Specific Analysis**: EEG, ECG, EMG, PPG, respiratory, voice
//! - **Real-Time Pipeline**: Streaming inference with latency tracking
//! - **Data Formats**: WFDB/PhysioNet and EDF/EDF+ support
//! - **GPU Acceleration**: WebGPU-based compute infrastructure
//! - **Visualization**: Plotting utilities for signals, spikes, and analysis
//! - **Power Modeling**: Energy estimation for neuromorphic and conventional hardware
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_core::{signal::*, pipeline::*};
//!
//! # fn example() -> dpb_core::Result<()> {
//! // Create a real-time pipeline
//! let config = PipelineConfig::new(256, 128, 1000.0)
//!     .with_max_latency(10.0);
//! let mut executor = PipelineExecutor::new(config)?;
//!
//! // Process streaming data
//! # let signal = vec![0.0f64; 100];
//! for sample in signal.iter() {
//!     if let Some(result) = executor.process_sample(*sample, |window| {
//!         // Your processing logic here
//!         vec![window.iter().sum::<f64>()]
//!     }) {
//!         // Handle result
//!         println!("Result: {:?}", result);
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: ECG Analysis
//!
//! ```rust
//! use dpb_core::signal::ecg::*;
//! use dpb_core::signal::hrv::*;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let ecg_signal = vec![0.0f64; 1000];
//! # let sample_rate = 250.0;
//! // Detect R-peaks
//! let detector = PanTompkinsDetector::new(sample_rate)?;
//! let peaks = detector.detect_r_peaks(&ecg_signal)?;
//!
//! // HRV is computed from RR INTERVALS in milliseconds, which each peak
//! // already carries relative to its predecessor.
//! let rr_intervals: Vec<f64> = peaks.iter().filter_map(|p| p.rr_interval_ms).collect();
//!
//! // Analyze HRV
//! let analyzer = HrvAnalyzer::new();
//! let metrics = analyzer.compute_time_domain(&rr_intervals)?;
//!
//! println!("Mean RR: {:.1} ms", metrics.mean_rr_ms);
//! println!("RMSSD: {:.1} ms", metrics.rmssd_ms);
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`signal`] | Signal processing primitives and domain-specific analyzers |
//! | [`pipeline`] | Real-time streaming pipeline infrastructure |
//! | [`io`] | Data format readers/writers (WFDB, EDF) |
//! | [`types`] | Core type definitions (SpikeEvent, TimeSeries, etc.) |
//! | [`traits`] | Framework trait definitions |
//! | [`gpu`] | GPU compute infrastructure |
//! | [`power`] | Power estimation models |
//! | [`viz`] | Visualization utilities |
//! | [`math`] | Mathematical and statistical utilities |
//! | [`metrics`] | Performance metrics |
//! | [`config`] | Configuration structures |
//! | [`tensor`] | Batched spike tensor operations |
//! | [`validation`] | Data validation utilities |
//!
//! ## Domain-Specific Modules
//!
//! | Module | Description |
//! |--------|-------------|
//! | [`cardiopulmonary`] | Cardiopulmonary analysis and modeling |
//! | [`biomechanics`] | Biomechanical analysis and gait |
//! | [`sleep`] | Sleep stage analysis |
//! | [`pain`] | Pain and sensory processing |
//! | [`somatosensory`] | Somatosensory signal analysis |
//! | [`vestibular`] | Vestibular system analysis |

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)]

pub mod accelerators;
pub mod biomechanics;
pub mod cardiopulmonary;
pub mod config;
pub mod error;
pub mod gpu;
pub mod io;
pub mod math;
pub mod metrics;
pub mod pain;
pub mod pipeline;
pub mod power;
pub mod signal;
pub mod sleep;
pub mod somatosensory;
pub mod tensor;
pub mod traits;
pub mod types;
pub mod validation;
pub mod vestibular;
pub mod viz;

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
    pub use crate::power::{ModelStats, PowerEstimator, PowerMetrics};
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
