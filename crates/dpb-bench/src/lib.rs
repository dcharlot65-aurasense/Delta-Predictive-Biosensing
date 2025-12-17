//! # DPB-Bench: Comprehensive Benchmark Suite
//!
//! This crate provides a comprehensive benchmarking and validation suite for the
//! Delta-Predictive Biosensing Framework. It includes:
//!
//! - **Standard Datasets**: Synthetic biosignal datasets with ground truth
//! - **Baseline Comparisons**: Compare SNNs against ANNs and traditional methods
//! - **Performance Profiling**: Time, memory, spike, and energy profiling
//! - **Report Generation**: Export results as JSON, CSV, and Markdown
//! - **Test Scenarios**: Pre-configured benchmarking scenarios
//!
//! ## Quick Start
//!
//! ```rust
//! use dpb_bench::prelude::*;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Load a standard dataset
//! let dataset = SyntheticECG::new(1000, 250.0, Some(42));
//! let (signal, ground_truth) = dataset.generate()?;
//!
//! // Profile encoding performance
//! let mut profiler = TimeProfiler::new("ECG Encoding");
//! profiler.start();
//!
//! // ... perform encoding ...
//!
//! profiler.stop();
//! println!("Encoding took: {:.2}ms", profiler.elapsed_ms());
//! # Ok(())
//! # }
//! ```
//!
//! ## Modules
//!
//! - [`datasets`] - Standard synthetic datasets
//! - [`baselines`] - Baseline comparison implementations
//! - [`profiling`] - Performance profiling tools
//! - [`reports`] - Report generation and export
//! - [`scenarios`] - Pre-configured test scenarios

#![warn(missing_docs)]
#![warn(clippy::all)]

pub mod datasets;
pub mod baselines;
pub mod profiling;
pub mod reports;
pub mod scenarios;

// Re-export commonly used types
pub use datasets::{
    SyntheticECG, SyntheticGait, SyntheticTremor, SyntheticVoice,
    BenchmarkDataset,
};

pub use baselines::{
    ANNBaseline, ConventionalBaseline, SNNBaseline,
    BaselineResult, ComparisonMetrics,
};

pub use profiling::{
    TimeProfiler, MemoryProfiler, SpikeProfiler, EnergyEstimator,
    ProfileResult,
};

pub use reports::{
    BenchmarkReport, ComparisonReport, ReportFormat,
};

pub use scenarios::{
    EncodingScenario, ClassificationScenario, RegressionScenario, LatencyScenario,
    ScenarioResult,
};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::datasets::*;
    pub use crate::baselines::*;
    pub use crate::profiling::*;
    pub use crate::reports::*;
    pub use crate::scenarios::*;
}

/// Version information
pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        let ver = version();
        assert!(!ver.is_empty());
    }

    #[test]
    fn test_module_structure() {
        // Verify all modules are accessible
        use prelude::*;

        // This is a compile-time check
        let _dataset: SyntheticECG;
        let _profiler: TimeProfiler;
        let _report: BenchmarkReport;
    }
}
