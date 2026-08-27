//! Weight distribution and analysis

use super::{AnalysisReport, ConvergenceAnalyzer, TrainingMetrics};
use std::collections::HashMap;

/// Tracks weight distribution evolution during training
pub struct WeightDistributionTracker {
    weight_snapshots: Vec<Vec<f64>>,
    snapshot_epochs: Vec<usize>,
    snapshot_interval: usize,
    last_snapshot_epoch: Option<usize>,
    converged: bool,
}

impl WeightDistributionTracker {
    pub fn new(snapshot_interval: usize) -> Self {
        Self {
            weight_snapshots: Vec::new(),
            snapshot_epochs: Vec::new(),
            snapshot_interval,
            last_snapshot_epoch: None,
            converged: false,
        }
    }

    /// Records a snapshot if `snapshot_interval` epochs have passed since the
    /// last one. The first snapshot is always recorded.
    pub fn add_weight_snapshot(&mut self, epoch: usize, weights: Vec<f64>) {
        // `last_snapshot_epoch` is an Option because 0 is a legitimate epoch.
        // Initialising it to 0 instead made "nothing recorded yet" and
        // "recorded at epoch 0" the same state, so the gate below always
        // rejected the FIRST call -- discarding the initial weight distribution
        // that every later snapshot is compared against. At interval 1 it
        // discarded epoch 0 and then recorded everything after, and any caller
        // passing a single snapshot at epoch 0 got an empty tracker and an
        // empty report.
        let due = match self.last_snapshot_epoch {
            None => true,
            Some(last) => epoch >= last + self.snapshot_interval,
        };
        if due {
            self.weight_snapshots.push(weights);
            self.snapshot_epochs.push(epoch);
            self.last_snapshot_epoch = Some(epoch);
        }
    }
}

impl Default for WeightDistributionTracker {
    fn default() -> Self {
        Self::new(10)
    }
}

impl ConvergenceAnalyzer for WeightDistributionTracker {
    fn name(&self) -> &str {
        "WeightDistributionTracker"
    }

    fn update(&mut self, _epoch: usize, _metrics: &TrainingMetrics) {
        // Weight snapshots are added via add_weight_snapshot
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric("num_snapshots", self.weight_snapshots.len() as f64);

        if let Some(latest_weights) = self.weight_snapshots.last() {
            // Compute statistics of latest weight distribution
            let mean = latest_weights.iter().sum::<f64>() / latest_weights.len() as f64;
            let variance = latest_weights
                .iter()
                .map(|&w| (w - mean).powi(2))
                .sum::<f64>()
                / latest_weights.len() as f64;
            let std_dev = variance.sqrt();

            report.add_metric("weight_mean", mean);
            report.add_metric("weight_std", std_dev);
            report.add_metric("weight_variance", variance);

            // Check for dead weights (very close to zero)
            let dead_weight_count = latest_weights.iter().filter(|&&w| w.abs() < 1e-6).count();
            let dead_weight_ratio = dead_weight_count as f64 / latest_weights.len() as f64;
            report.add_metric("dead_weight_ratio", dead_weight_ratio);

            if dead_weight_ratio > 0.5 {
                report.add_recommendation("More than 50% of weights are near zero. Consider weight initialization or regularization adjustments.");
            }

            // Check for distribution changes over time
            if self.weight_snapshots.len() >= 2 {
                let first = &self.weight_snapshots[0];
                let first_mean = first.iter().sum::<f64>() / first.len() as f64;
                let mean_change = (mean - first_mean).abs();

                report.add_metric("weight_mean_change", mean_change);

                if mean_change < 1e-6 {
                    report.add_recommendation(
                        "Weight distribution is not changing. Training may have stalled.",
                    );
                }
            }
        }

        report
    }

    fn reset(&mut self) {
        self.weight_snapshots.clear();
        self.snapshot_epochs.clear();
        self.last_snapshot_epoch = None;
        self.converged = false;
    }
}

/// Tracks weight magnitudes (norms) during training
pub struct WeightMagnitudeTracker {
    weight_norm_history: Vec<f64>,
    layer_norms: HashMap<String, Vec<f64>>,
    converged: bool,
}

impl WeightMagnitudeTracker {
    pub fn new() -> Self {
        Self {
            weight_norm_history: Vec::new(),
            layer_norms: HashMap::new(),
            converged: false,
        }
    }

    pub fn update_layer_norm(&mut self, layer_name: String, norm: f64) {
        self.layer_norms.entry(layer_name).or_default().push(norm);
    }
}

impl Default for WeightMagnitudeTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for WeightMagnitudeTracker {
    fn name(&self) -> &str {
        "WeightMagnitudeTracker"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        self.weight_norm_history.push(metrics.weight_norm);
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if !self.weight_norm_history.is_empty() {
            let mean_norm = self.weight_norm_history.iter().sum::<f64>()
                / self.weight_norm_history.len() as f64;
            let max_norm = self
                .weight_norm_history
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let min_norm = self
                .weight_norm_history
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));

            report.add_metric("mean_weight_norm", mean_norm);
            report.add_metric("max_weight_norm", max_norm);
            report.add_metric("min_weight_norm", min_norm);

            if let Some(&current_norm) = self.weight_norm_history.last() {
                report.add_metric("current_weight_norm", current_norm);

                if current_norm > 10.0 {
                    report.add_recommendation(
                        "Weight norms are very large. Consider weight regularization or clipping.",
                    );
                } else if current_norm < 0.01 {
                    report.add_recommendation(
                        "Weight norms are very small. Weights may be decaying too much.",
                    );
                }
            }

            // Check norm growth
            if self.weight_norm_history.len() >= 2 {
                let first = self.weight_norm_history[0];
                let last = *self.weight_norm_history.last().unwrap();
                let growth_ratio = last / first;

                report.add_metric("weight_norm_growth_ratio", growth_ratio);

                if growth_ratio > 2.0 {
                    report.add_recommendation(
                        "Weight norms are growing significantly. May lead to instability.",
                    );
                } else if growth_ratio < 0.5 {
                    report.add_recommendation("Weight norms are shrinking significantly. May indicate over-regularization.");
                }
            }
        }

        // Per-layer statistics
        for (layer_name, norms) in &self.layer_norms {
            if !norms.is_empty() {
                let mean = norms.iter().sum::<f64>() / norms.len() as f64;
                report.add_metric(format!("{}_mean_norm", layer_name), mean);
            }
        }

        report
    }

    fn reset(&mut self) {
        self.weight_norm_history.clear();
        self.layer_norms.clear();
        self.converged = false;
    }
}

/// Tracks weight sparsity (pruning) during training
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
pub struct WeightSparsityTracker {
    sparsity_history: Vec<f64>,
    target_sparsity: f64,
    sparsity_threshold: f64,
    converged: bool,
}

impl WeightSparsityTracker {
    pub fn new(target_sparsity: f64, sparsity_threshold: f64) -> Self {
        Self {
            sparsity_history: Vec::new(),
            target_sparsity,
            sparsity_threshold,
            converged: false,
        }
    }

    pub fn update_sparsity(&mut self, sparsity: f64) {
        self.sparsity_history.push(sparsity);
    }
}

impl Default for WeightSparsityTracker {
    fn default() -> Self {
        Self::new(0.5, 1e-6)
    }
}

impl ConvergenceAnalyzer for WeightSparsityTracker {
    fn name(&self) -> &str {
        "WeightSparsityTracker"
    }

    fn update(&mut self, _epoch: usize, _metrics: &TrainingMetrics) {
        // Sparsity is updated via update_sparsity method
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric("target_sparsity", self.target_sparsity);

        if !self.sparsity_history.is_empty() {
            let mean_sparsity =
                self.sparsity_history.iter().sum::<f64>() / self.sparsity_history.len() as f64;
            report.add_metric("mean_sparsity", mean_sparsity);

            if let Some(&current_sparsity) = self.sparsity_history.last() {
                report.add_metric("current_sparsity", current_sparsity);

                let deviation = (current_sparsity - self.target_sparsity).abs();
                if deviation > 0.2 {
                    report.add_recommendation(format!(
                        "Weight sparsity ({:.2}) deviates from target ({:.2}). Consider adjusting pruning strategy.",
                        current_sparsity, self.target_sparsity
                    ));
                }

                if current_sparsity > 0.9 {
                    report.add_recommendation(
                        "Very high weight sparsity. Network capacity may be limited.",
                    );
                } else if current_sparsity < 0.1 {
                    report.add_recommendation(
                        "Low weight sparsity. Network is dense, may benefit from pruning.",
                    );
                }
            }

            // Track sparsity trend
            if self.sparsity_history.len() >= 10 {
                let recent_mean = self.sparsity_history[self.sparsity_history.len() - 5..]
                    .iter()
                    .sum::<f64>()
                    / 5.0;
                let early_mean = self.sparsity_history[..5].iter().sum::<f64>() / 5.0;
                let trend = recent_mean - early_mean;

                report.add_metric("sparsity_trend", trend);

                if trend > 0.1 {
                    report.add_recommendation("Weight sparsity is increasing over time.");
                } else if trend < -0.1 {
                    report.add_recommendation("Weight sparsity is decreasing over time.");
                }
            }
        }

        report
    }

    fn reset(&mut self) {
        self.sparsity_history.clear();
        self.converged = false;
    }
}

/// Tracks magnitudes of weight updates during training
pub struct WeightUpdateTracker {
    update_magnitude_history: Vec<f64>,
    layer_update_magnitudes: HashMap<String, Vec<f64>>,
    converged: bool,
}

impl WeightUpdateTracker {
    pub fn new() -> Self {
        Self {
            update_magnitude_history: Vec::new(),
            layer_update_magnitudes: HashMap::new(),
            converged: false,
        }
    }

    pub fn update_layer_magnitude(&mut self, layer_name: String, magnitude: f64) {
        self.layer_update_magnitudes
            .entry(layer_name)
            .or_default()
            .push(magnitude);
    }
}

impl Default for WeightUpdateTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for WeightUpdateTracker {
    fn name(&self) -> &str {
        "WeightUpdateTracker"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Use gradient norm as proxy for update magnitude
        let update_magnitude = metrics.gradient_norm * metrics.learning_rate;
        self.update_magnitude_history.push(update_magnitude);
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if !self.update_magnitude_history.is_empty() {
            let mean_magnitude = self.update_magnitude_history.iter().sum::<f64>()
                / self.update_magnitude_history.len() as f64;
            let max_magnitude = self
                .update_magnitude_history
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            report.add_metric("mean_update_magnitude", mean_magnitude);
            report.add_metric("max_update_magnitude", max_magnitude);

            if let Some(&current_magnitude) = self.update_magnitude_history.last() {
                report.add_metric("current_update_magnitude", current_magnitude);

                if current_magnitude < 1e-8 {
                    report.add_recommendation(
                        "Weight updates are very small. Learning may have stopped.",
                    );
                } else if current_magnitude > 1.0 {
                    report.add_recommendation(
                        "Weight updates are very large. May cause instability.",
                    );
                }
            }

            // Check if updates are decreasing (normal in convergence)
            if self.update_magnitude_history.len() >= 10 {
                let recent_mean = self.update_magnitude_history
                    [self.update_magnitude_history.len() - 5..]
                    .iter()
                    .sum::<f64>()
                    / 5.0;
                let early_mean = self.update_magnitude_history[..5].iter().sum::<f64>() / 5.0;

                report.add_metric(
                    "update_magnitude_ratio",
                    recent_mean / early_mean.max(1e-10),
                );

                if recent_mean < early_mean * 0.1 {
                    report.add_recommendation(
                        "Weight updates have decreased significantly. Nearing convergence.",
                    );
                }
            }
        }

        // Per-layer statistics
        for (layer_name, magnitudes) in &self.layer_update_magnitudes {
            if !magnitudes.is_empty() {
                let mean = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
                report.add_metric(format!("{}_mean_update", layer_name), mean);
            }
        }

        report
    }

    fn reset(&mut self) {
        self.update_magnitude_history.clear();
        self.layer_update_magnitudes.clear();
        self.converged = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_distribution_tracker() {
        let mut tracker = WeightDistributionTracker::new(5);

        tracker.add_weight_snapshot(0, vec![0.1, 0.2, 0.3, 0.4]);
        tracker.add_weight_snapshot(5, vec![0.15, 0.25, 0.35, 0.45]);
        tracker.add_weight_snapshot(10, vec![0.2, 0.3, 0.4, 0.5]);

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("num_snapshots"));
        assert_eq!(report.metrics.get("num_snapshots"), Some(&3.0));
    }

    #[test]
    fn test_weight_magnitude_tracker() {
        let mut tracker = WeightMagnitudeTracker::new();

        for i in 0..10 {
            let weight_norm = 1.0 + i as f64 * 0.1;
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_weight_norm(weight_norm);
            tracker.update(i, &metrics);
        }

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("mean_weight_norm"));
        assert!(report.metrics.contains_key("current_weight_norm"));
    }

    #[test]
    fn test_weight_sparsity_tracker() {
        let mut tracker = WeightSparsityTracker::new(0.5, 1e-6);

        for i in 0..10 {
            tracker.update_sparsity(0.4 + i as f64 * 0.01);
        }

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("mean_sparsity"));
        assert!(report.metrics.contains_key("current_sparsity"));
    }

    #[test]
    fn test_weight_update_tracker() {
        let mut tracker = WeightUpdateTracker::new();

        for i in 0..10 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.9)
                .with_gradient_norm(0.1)
                .with_learning_rate(0.001);
            tracker.update(i, &metrics);
        }

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("mean_update_magnitude"));
        assert!(report.metrics.contains_key("current_update_magnitude"));
    }

    #[test]
    fn test_layer_specific_tracking() {
        let mut tracker = WeightMagnitudeTracker::new();

        tracker.update_layer_norm("layer1".to_string(), 1.0);
        tracker.update_layer_norm("layer1".to_string(), 1.1);
        tracker.update_layer_norm("layer2".to_string(), 2.0);

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("layer1_mean_norm"));
        assert!(report.metrics.contains_key("layer2_mean_norm"));
    }

    #[test]
    fn test_dead_weights_detection() {
        let mut tracker = WeightDistributionTracker::new(1);

        let weights = vec![0.0, 0.0, 0.0, 0.1, 0.2]; // 60% dead weights
        tracker.add_weight_snapshot(0, weights);

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("dead_weight_ratio"));
        assert!(!report.recommendations.is_empty());
    }
}
