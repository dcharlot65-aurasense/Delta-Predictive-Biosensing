//! Model calibration and uncertainty quantification
//!
//! Provides methods for calibrating model confidence scores and
//! quantifying prediction uncertainty.

pub mod temperature;
pub mod isotonic;
pub mod uncertainty;
pub mod metrics;

pub use temperature::{TemperatureScaling, PlattScaling};
pub use isotonic::IsotonicCalibration;
pub use uncertainty::{
    UncertaintyEstimator, MCDropout, EnsembleUncertainty, ConfidenceInterval, bootstrap_ci,
};
pub use metrics::{
    expected_calibration_error, maximum_calibration_error, reliability_diagram, brier_score,
    negative_log_likelihood, ReliabilityBin,
};
