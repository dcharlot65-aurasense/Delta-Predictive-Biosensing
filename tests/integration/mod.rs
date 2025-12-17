//! Integration tests for the DPB framework
//!
//! These tests validate full pipelines from signal generation through encoding,
//! SNN processing, and decoding to final predictions.

pub mod ecg_pipeline;
pub mod gait_pipeline;
pub mod tremor_pipeline;
pub mod voice_pipeline;
pub mod multimodal_pipeline;
pub mod synthetic_validation;
pub mod encoder_accuracy;
pub mod training_loop;
pub mod inference_latency;
pub mod cross_crate;

/// Common test utilities
pub mod utils {
    /// Default test seed for reproducibility
    pub const TEST_SEED: u64 = 42;

    /// Tolerance for floating point comparisons (0.1%)
    pub const RELATIVE_TOL: f64 = 0.001;

    /// Absolute tolerance for small values
    pub const ABSOLUTE_TOL: f64 = 1e-6;

    /// Assert two values are approximately equal
    pub fn assert_approx_eq(actual: f64, expected: f64, context: &str) {
        assert!(
            (actual - expected).abs() <= ABSOLUTE_TOL ||
            ((actual - expected).abs() / expected.abs()) <= RELATIVE_TOL,
            "Failed: {} - expected {}, got {}, diff {}",
            context,
            expected,
            actual,
            (actual - expected).abs()
        );
    }

    /// Assert value is within range
    pub fn assert_in_range(value: f64, min: f64, max: f64, context: &str) {
        assert!(
            value >= min && value <= max,
            "Failed: {} - value {} not in range [{}, {}]",
            context,
            value,
            min,
            max
        );
    }

    /// Assert event timing is within tolerance (1ms)
    pub fn assert_event_timing(actual_time: f64, expected_time: f64, context: &str) {
        let tolerance = 0.001; // 1ms
        assert!(
            (actual_time - expected_time).abs() < tolerance,
            "Failed: {} - event timing {} not within 1ms of expected {}",
            context,
            actual_time,
            expected_time
        );
    }
}
