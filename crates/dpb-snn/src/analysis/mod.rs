//! Training analysis and convergence detection for SNNs
//!
//! This module provides comprehensive tools for analyzing SNN training dynamics,
//! detecting convergence, and comparing different training methods.

mod comparison;
mod convergence;
mod gradient_analysis;
mod learning_curves;
mod spike_statistics;
mod weight_analysis;

pub use convergence::{
    AccuracyPlateauDetector, ConvergenceRateAnalyzer, DivergenceDetector, EarlyStoppingAnalyzer,
    LossPlateauDetector, OscillationDetector,
};

pub use learning_curves::{
    BatchSizeAnalyzer, EpochEfficiencyAnalyzer, GeneralizationGapAnalyzer, LearningCurveSmoothed,
    LearningRateAnalyzer, OverfittingDetector,
};

pub use gradient_analysis::{
    ExplodingGradientDetector, GradientFlowAnalyzer, GradientNormTracker,
    SurrogateGradientAnalyzer, VanishingGradientDetector,
};

pub use spike_statistics::{
    SaturatedNeuronDetector, SilentNeuronDetector, SparsityTracker, SpikeRateTracker,
    TemporalDynamicsAnalyzer,
};

pub use weight_analysis::{
    WeightDistributionTracker, WeightMagnitudeTracker, WeightSparsityTracker, WeightUpdateTracker,
};

pub use comparison::{HyperparameterSensitivityAnalyzer, MethodComparisonAnalyzer};

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Core trait for all convergence analyzers
pub trait ConvergenceAnalyzer: Send + Sync {
    /// Returns the name of this analyzer
    fn name(&self) -> &str;

    /// Updates the analyzer with new training metrics
    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics);

    /// Returns whether training has converged according to this analyzer
    fn is_converged(&self) -> bool;

    /// Returns the epoch at which convergence was detected, if any
    fn convergence_epoch(&self) -> Option<usize>;

    /// Generates a detailed analysis report
    fn analysis_report(&self) -> AnalysisReport;

    /// Resets the analyzer state
    fn reset(&mut self);
}

/// Detects when a tracked series has stopped moving.
///
/// The criterion is the one the analyzers in `convergence.rs` already use:
/// the trailing window's spread stays below a threshold for `patience`
/// consecutive epochs. The spread is measured relative to the window's mean,
/// so a single threshold is meaningful for series of different magnitudes --
/// a loss of 40 and a sparsity of 0.02 do not need separate tuning.
///
/// This exists because most `ConvergenceAnalyzer` implementations had a
/// `converged` field that was set to false in their constructor and reset, and
/// never anywhere set to true: `is_converged` could not return true however
/// the training went, and `convergence_epoch` returned a hardcoded None. An
/// analyzer that always answers "not converged" is indistinguishable from one
/// that is working and has nothing to report.
#[derive(Debug, Clone)]
pub struct StabilityDetector {
    window: usize,
    /// Maximum relative standard deviation still counted as stable.
    tolerance: f64,
    patience: usize,
    stable_epochs: usize,
    converged_at: Option<usize>,
}

impl StabilityDetector {
    /// A detector over `window` recent values, requiring `patience`
    /// consecutive stable windows.
    pub fn new(window: usize, tolerance: f64, patience: usize) -> Self {
        Self {
            window: window.max(2),
            tolerance,
            patience: patience.max(1),
            stable_epochs: 0,
            converged_at: None,
        }
    }

    /// Folds the latest state of `history` in, at `epoch`.
    ///
    /// Does nothing until the history is at least a window long: a series too
    /// short to have a trend cannot be said to have settled into one.
    pub fn observe(&mut self, epoch: usize, history: &[f64]) {
        if history.len() < self.window {
            return;
        }
        let recent = &history[history.len() - self.window..];
        if recent.iter().any(|v| !v.is_finite()) {
            self.stable_epochs = 0;
            return;
        }

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / recent.len() as f64;
        // Relative to the mean's magnitude, with a floor so a series sitting at
        // zero is judged on its absolute spread rather than dividing by nothing.
        let relative = variance.sqrt() / mean.abs().max(1e-6);

        if relative <= self.tolerance {
            self.stable_epochs += 1;
            if self.stable_epochs >= self.patience && self.converged_at.is_none() {
                self.converged_at = Some(epoch);
            }
        } else {
            self.stable_epochs = 0;
        }
    }

    /// Whether the series has settled.
    pub fn is_converged(&self) -> bool {
        self.converged_at.is_some()
    }

    /// The epoch at which it first settled.
    pub fn epoch(&self) -> Option<usize> {
        self.converged_at
    }

    pub fn reset(&mut self) {
        self.stable_epochs = 0;
        self.converged_at = None;
    }
}

impl Default for StabilityDetector {
    /// Ten-epoch window, 1% relative spread, sustained for three epochs.
    fn default() -> Self {
        Self::new(10, 0.01, 3)
    }
}

/// Training metrics collected during training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Current epoch number
    pub epoch: usize,

    /// Training loss
    pub train_loss: f64,

    /// Validation loss (optional)
    pub val_loss: Option<f64>,

    /// Training accuracy
    pub train_accuracy: f64,

    /// Validation accuracy (optional)
    pub val_accuracy: Option<f64>,

    /// Current learning rate
    pub learning_rate: f64,

    /// Gradient norm
    pub gradient_norm: f64,

    /// Average spike rate across network
    pub spike_rate: f64,

    /// Average weight norm
    pub weight_norm: f64,

    /// Additional custom metrics
    pub custom_metrics: HashMap<String, f64>,
}

impl TrainingMetrics {
    /// Creates a new TrainingMetrics instance
    pub fn new(epoch: usize, train_loss: f64, train_accuracy: f64) -> Self {
        Self {
            epoch,
            train_loss,
            val_loss: None,
            train_accuracy,
            val_accuracy: None,
            learning_rate: 0.001,
            gradient_norm: 0.0,
            spike_rate: 0.0,
            weight_norm: 0.0,
            custom_metrics: HashMap::new(),
        }
    }

    /// Builder method to set validation loss
    pub fn with_val_loss(mut self, val_loss: f64) -> Self {
        self.val_loss = Some(val_loss);
        self
    }

    /// Builder method to set validation accuracy
    pub fn with_val_accuracy(mut self, val_accuracy: f64) -> Self {
        self.val_accuracy = Some(val_accuracy);
        self
    }

    /// Builder method to set learning rate
    pub fn with_learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Builder method to set gradient norm
    pub fn with_gradient_norm(mut self, gradient_norm: f64) -> Self {
        self.gradient_norm = gradient_norm;
        self
    }

    /// Builder method to set spike rate
    pub fn with_spike_rate(mut self, spike_rate: f64) -> Self {
        self.spike_rate = spike_rate;
        self
    }

    /// Builder method to set weight norm
    pub fn with_weight_norm(mut self, weight_norm: f64) -> Self {
        self.weight_norm = weight_norm;
        self
    }
}

/// Analysis report generated by an analyzer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Name of the analyzer
    pub analyzer_name: String,

    /// Whether convergence has been detected
    pub converged: bool,

    /// Epoch at which convergence was detected
    pub convergence_epoch: Option<usize>,

    /// Analysis metrics
    pub metrics: HashMap<String, f64>,

    /// Recommendations for improving training
    pub recommendations: Vec<String>,

    /// Additional analysis details
    pub details: HashMap<String, String>,
}

impl AnalysisReport {
    /// Creates a new empty analysis report
    pub fn new(analyzer_name: String) -> Self {
        Self {
            analyzer_name,
            converged: false,
            convergence_epoch: None,
            metrics: HashMap::new(),
            recommendations: Vec::new(),
            details: HashMap::new(),
        }
    }

    /// Adds a metric to the report
    pub fn add_metric(&mut self, key: impl Into<String>, value: f64) {
        self.metrics.insert(key.into(), value);
    }

    /// Adds a recommendation to the report
    pub fn add_recommendation(&mut self, recommendation: impl Into<String>) {
        self.recommendations.push(recommendation.into());
    }

    /// Adds a detail to the report
    pub fn add_detail(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.details.insert(key.into(), value.into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_metrics_builder() {
        let metrics = TrainingMetrics::new(1, 0.5, 0.9)
            .with_val_loss(0.6)
            .with_val_accuracy(0.85)
            .with_learning_rate(0.001)
            .with_gradient_norm(0.1)
            .with_spike_rate(0.05)
            .with_weight_norm(1.5);

        assert_eq!(metrics.epoch, 1);
        assert_eq!(metrics.train_loss, 0.5);
        assert_eq!(metrics.val_loss, Some(0.6));
        assert_eq!(metrics.val_accuracy, Some(0.85));
        assert_eq!(metrics.learning_rate, 0.001);
    }

    #[test]
    fn test_analysis_report_builder() {
        let mut report = AnalysisReport::new("TestAnalyzer".to_string());
        report.add_metric("test_metric", 1.5);
        report.add_recommendation("Try lower learning rate");
        report.add_detail("note", "Test detail");

        assert_eq!(report.analyzer_name, "TestAnalyzer");
        assert_eq!(report.metrics.get("test_metric"), Some(&1.5));
        assert_eq!(report.recommendations.len(), 1);
        assert_eq!(report.details.get("note"), Some(&"Test detail".to_string()));
    }
}

#[cfg(test)]
mod stability_tests {
    use super::*;

    /// A settled series is detected, and the epoch recorded.
    #[test]
    fn a_settled_series_converges() {
        let mut d = StabilityDetector::new(5, 0.01, 2);
        let mut history = Vec::new();
        for epoch in 0..30 {
            history.push(1.0);
            d.observe(epoch, &history);
        }
        assert!(d.is_converged());
        assert!(d.epoch().is_some());
    }

    /// A series still moving does not.
    #[test]
    fn a_moving_series_does_not_converge() {
        let mut d = StabilityDetector::new(5, 0.01, 2);
        let mut history = Vec::new();
        for epoch in 0..30 {
            history.push(epoch as f64);
            d.observe(epoch, &history);
        }
        assert!(
            !d.is_converged(),
            "a series climbing by 1.0 each epoch was called converged"
        );
    }

    /// Too little history is not evidence of anything.
    #[test]
    fn a_short_series_does_not_converge() {
        let mut d = StabilityDetector::new(10, 0.01, 2);
        let history = vec![1.0, 1.0, 1.0];
        d.observe(2, &history);
        assert!(!d.is_converged());
    }

    /// The relative criterion works across magnitudes: a series jittering by
    /// the same fraction is judged the same way whether it sits at 0.02 or 40.
    #[test]
    fn the_criterion_is_scale_free() {
        for level in [0.02f64, 1.0, 40.0] {
            let mut d = StabilityDetector::new(6, 0.02, 2);
            let mut history = Vec::new();
            for epoch in 0..25 {
                // 0.5% jitter: well inside a 2% tolerance at every level.
                let jitter = if epoch % 2 == 0 { 1.005 } else { 0.995 };
                history.push(level * jitter);
                d.observe(epoch, &history);
            }
            assert!(d.is_converged(), "level {level} was not detected as stable");

            let mut d = StabilityDetector::new(6, 0.02, 2);
            let mut history = Vec::new();
            for epoch in 0..25 {
                // 20% jitter: outside the tolerance at every level.
                let jitter = if epoch % 2 == 0 { 1.2 } else { 0.8 };
                history.push(level * jitter);
                d.observe(epoch, &history);
            }
            assert!(!d.is_converged(), "level {level} was wrongly called stable");
        }
    }

    /// Non-finite values break the run rather than being averaged in.
    #[test]
    fn non_finite_values_reset_the_count() {
        let mut d = StabilityDetector::new(4, 0.01, 3);
        let mut history = Vec::new();
        for epoch in 0..10 {
            history.push(if epoch == 5 { f64::NAN } else { 1.0 });
            d.observe(epoch, &history);
        }
        // The NaN sits inside the trailing window for four epochs after it
        // appears, so convergence cannot be reached before then.
        assert!(d.epoch().is_none_or(|e| e >= 9), "converged across a NaN");
    }
}
