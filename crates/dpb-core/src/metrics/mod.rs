//! Evaluation metrics for the DPB framework.
//!
//! This module provides comprehensive evaluation metrics for:
//! - Classification tasks
//! - Regression tasks
//! - Signal quality assessment
//! - Clinical validity
//! - SNN efficiency analysis

use crate::error::Result;

pub mod classification;
pub mod clinical;
pub mod efficiency;
pub mod regression;
pub mod signal;

// Re-export all metrics
pub use classification::*;
pub use clinical::*;
pub use efficiency::*;
pub use regression::*;
pub use signal::*;

/// Core trait for all evaluation metrics.
///
/// Metrics can be used in two modes:
/// 1. Stateless: Use `compute()` directly with data
/// 2. Stateful: Use `update()` to accumulate data, then `result()` to get the metric
pub trait MetricTrait: Send + Sync {
    /// Returns the name of this metric.
    fn name(&self) -> &str;

    /// Computes the metric directly from predictions and targets (stateless).
    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64>;

    /// Updates the metric state with new predictions and targets (stateful).
    fn update(&mut self, predictions: &[f32], targets: &[f32]);

    /// Returns the current metric result based on accumulated state.
    fn result(&self) -> f64;

    /// Resets the metric state.
    fn reset(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metric_trait_bounds() {
        // Verify that MetricTrait has the required bounds
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Box<dyn MetricTrait>>();
    }
}
