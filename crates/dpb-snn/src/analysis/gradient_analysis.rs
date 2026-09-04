//! Gradient flow analysis for SNNs

use super::{AnalysisReport, ConvergenceAnalyzer, StabilityDetector, TrainingMetrics};
use std::collections::HashMap;

/// Tracks gradient norms throughout training
pub struct GradientNormTracker {
    gradient_history: Vec<f64>,
    mean_gradient: f64,
    max_gradient: f64,
    min_gradient: f64,
    stability: StabilityDetector,
}

impl GradientNormTracker {
    pub fn new() -> Self {
        Self {
            gradient_history: Vec::new(),
            mean_gradient: 0.0,
            max_gradient: 0.0,
            min_gradient: f64::INFINITY,
            stability: StabilityDetector::default(),
        }
    }
}

impl Default for GradientNormTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for GradientNormTracker {
    fn name(&self) -> &str {
        "GradientNormTracker"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        let grad_norm = metrics.gradient_norm;
        self.gradient_history.push(grad_norm);

        self.max_gradient = self.max_gradient.max(grad_norm);
        self.min_gradient = self.min_gradient.min(grad_norm);

        if !self.gradient_history.is_empty() {
            self.mean_gradient =
                self.gradient_history.iter().sum::<f64>() / self.gradient_history.len() as f64;
        }

        // Converged when gradient history has stopped moving.
        self.stability.observe(epoch, &self.gradient_history);
    }

    fn is_converged(&self) -> bool {
        self.stability.is_converged()
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.stability.epoch()
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric("mean_gradient_norm", self.mean_gradient);
        report.add_metric("max_gradient_norm", self.max_gradient);
        report.add_metric("min_gradient_norm", self.min_gradient);

        if let Some(&current_grad) = self.gradient_history.last() {
            report.add_metric("current_gradient_norm", current_grad);

            if current_grad < 1e-7 {
                report.add_recommendation(
                    "Gradients are very small. May indicate vanishing gradients.",
                );
            } else if current_grad > 10.0 {
                report.add_recommendation("Gradients are large. May indicate exploding gradients.");
            }
        }

        report
    }

    fn reset(&mut self) {
        self.gradient_history.clear();
        self.mean_gradient = 0.0;
        self.max_gradient = 0.0;
        self.min_gradient = f64::INFINITY;
        self.stability.reset();
    }
}

/// Analyzes gradient flow through network layers
pub struct GradientFlowAnalyzer {
    layer_gradients: HashMap<String, Vec<f64>>,
    stability: StabilityDetector,
}

impl GradientFlowAnalyzer {
    pub fn new() -> Self {
        Self {
            layer_gradients: HashMap::new(),
            stability: StabilityDetector::default(),
        }
    }

    pub fn update_layer_gradient(&mut self, layer_name: String, gradient_norm: f64) {
        self.layer_gradients
            .entry(layer_name)
            .or_default()
            .push(gradient_norm);
    }
}

impl Default for GradientFlowAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for GradientFlowAnalyzer {
    fn name(&self) -> &str {
        "GradientFlowAnalyzer"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Store overall gradient norm
        self.update_layer_gradient("overall".to_string(), metrics.gradient_norm);
    }

    fn is_converged(&self) -> bool {
        self.stability.is_converged()
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.stability.epoch()
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        // Compute statistics per layer
        for (layer_name, gradients) in &self.layer_gradients {
            if !gradients.is_empty() {
                let mean = gradients.iter().sum::<f64>() / gradients.len() as f64;
                let max = gradients.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min = gradients.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                report.add_metric(format!("{}_mean_grad", layer_name), mean);
                report.add_metric(format!("{}_max_grad", layer_name), max);
                report.add_metric(format!("{}_min_grad", layer_name), min);

                if mean < 1e-7 {
                    report.add_recommendation(format!(
                        "Layer {} has very small gradients. May indicate vanishing gradient problem.",
                        layer_name
                    ));
                }
            }
        }

        report
    }

    fn reset(&mut self) {
        self.layer_gradients.clear();
        self.stability.reset();
    }
}

/// Detects vanishing gradient problem
pub struct VanishingGradientDetector {
    gradient_history: Vec<f64>,
    threshold: f64,
    window_size: usize,
    detected: bool,
    detection_epoch: Option<usize>,
}

impl VanishingGradientDetector {
    pub fn new(threshold: f64, window_size: usize) -> Self {
        Self {
            gradient_history: Vec::new(),
            threshold,
            window_size,
            detected: false,
            detection_epoch: None,
        }
    }
}

impl Default for VanishingGradientDetector {
    fn default() -> Self {
        Self::new(1e-7, 5)
    }
}

impl ConvergenceAnalyzer for VanishingGradientDetector {
    fn name(&self) -> &str {
        "VanishingGradientDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        self.gradient_history.push(metrics.gradient_norm);

        if self.gradient_history.len() >= self.window_size {
            let recent = &self.gradient_history[self.gradient_history.len() - self.window_size..];
            let mean_recent = recent.iter().sum::<f64>() / self.window_size as f64;

            if mean_recent < self.threshold && !self.detected {
                self.detected = true;
                self.detection_epoch = Some(epoch);
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.detected
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.detection_epoch
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());
        report.converged = self.detected;
        report.convergence_epoch = self.detection_epoch;

        if let Some(&current_grad) = self.gradient_history.last() {
            report.add_metric("current_gradient", current_grad);
        }

        report.add_metric("threshold", self.threshold);

        if self.detected {
            report.add_recommendation("Vanishing gradients detected! Consider:");
            report.add_recommendation("- Using skip connections or residual connections");
            report.add_recommendation("- Adjusting neuron parameters");
            report.add_recommendation("- Using gradient clipping");
            report.add_recommendation("- Reducing network depth");
        }

        report
    }

    fn reset(&mut self) {
        self.gradient_history.clear();
        self.detected = false;
        self.detection_epoch = None;
    }
}

/// Detects exploding gradient problem
pub struct ExplodingGradientDetector {
    gradient_history: Vec<f64>,
    threshold: f64,
    detected: bool,
    detection_epoch: Option<usize>,
}

impl ExplodingGradientDetector {
    pub fn new(threshold: f64) -> Self {
        Self {
            gradient_history: Vec::new(),
            threshold,
            detected: false,
            detection_epoch: None,
        }
    }
}

impl Default for ExplodingGradientDetector {
    fn default() -> Self {
        Self::new(10.0)
    }
}

impl ConvergenceAnalyzer for ExplodingGradientDetector {
    fn name(&self) -> &str {
        "ExplodingGradientDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        let grad_norm = metrics.gradient_norm;
        self.gradient_history.push(grad_norm);

        if (grad_norm > self.threshold || grad_norm.is_nan() || grad_norm.is_infinite())
            && !self.detected
        {
            self.detected = true;
            self.detection_epoch = Some(epoch);
        }
    }

    fn is_converged(&self) -> bool {
        self.detected
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.detection_epoch
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());
        report.converged = self.detected;
        report.convergence_epoch = self.detection_epoch;

        if let Some(&max_grad) = self.gradient_history.iter().max_by(|a, b| a.total_cmp(b)) {
            report.add_metric("max_gradient", max_grad);
        }

        report.add_metric("threshold", self.threshold);

        if self.detected {
            report.add_recommendation("Exploding gradients detected! Consider:");
            report.add_recommendation("- Using gradient clipping");
            report.add_recommendation("- Reducing learning rate");
            report.add_recommendation("- Using batch normalization");
            report.add_recommendation("- Adjusting weight initialization");
        }

        report
    }

    fn reset(&mut self) {
        self.gradient_history.clear();
        self.detected = false;
        self.detection_epoch = None;
    }
}

/// Analyzes the quality of surrogate gradients in SNNs
pub struct SurrogateGradientAnalyzer {
    surrogate_stats: Vec<(f64, f64)>, // (mean, std)
    stability: StabilityDetector,
}

impl SurrogateGradientAnalyzer {
    pub fn new() -> Self {
        Self {
            surrogate_stats: Vec::new(),
            stability: StabilityDetector::default(),
        }
    }

    pub fn update_surrogate_stats(&mut self, mean: f64, std: f64) {
        self.surrogate_stats.push((mean, std));
    }
}

impl Default for SurrogateGradientAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for SurrogateGradientAnalyzer {
    fn name(&self) -> &str {
        "SurrogateGradientAnalyzer"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Use gradient norm as a proxy for surrogate gradient quality
        let grad_norm = metrics.gradient_norm;
        self.surrogate_stats.push((grad_norm, grad_norm * 0.1));
    }

    fn is_converged(&self) -> bool {
        self.stability.is_converged()
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.stability.epoch()
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if !self.surrogate_stats.is_empty() {
            let mean_of_means = self.surrogate_stats.iter().map(|(m, _)| m).sum::<f64>()
                / self.surrogate_stats.len() as f64;
            let mean_of_stds = self.surrogate_stats.iter().map(|(_, s)| s).sum::<f64>()
                / self.surrogate_stats.len() as f64;

            report.add_metric("mean_surrogate_gradient", mean_of_means);
            report.add_metric("mean_surrogate_std", mean_of_stds);

            if mean_of_means < 1e-6 {
                report.add_recommendation("Surrogate gradients are very small. Consider adjusting surrogate function parameters.");
            }

            if mean_of_stds > mean_of_means * 2.0 {
                report.add_recommendation(
                    "High variance in surrogate gradients. Training may be unstable.",
                );
            }
        }

        report
    }

    fn reset(&mut self) {
        self.surrogate_stats.clear();
        self.stability.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_norm_tracker() {
        let mut tracker = GradientNormTracker::new();

        for i in 0..10 {
            let grad_norm = 0.1 + i as f64 * 0.05;
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_gradient_norm(grad_norm);
            tracker.update(i, &metrics);
        }

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("mean_gradient_norm"));
        assert!(report.metrics.contains_key("max_gradient_norm"));
    }

    #[test]
    fn test_vanishing_gradient_detector() {
        let mut detector = VanishingGradientDetector::new(1e-6, 3);

        for i in 0..10 {
            let grad_norm = 1e-7;
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_gradient_norm(grad_norm);
            detector.update(i, &metrics);
        }

        assert!(detector.is_converged());
    }

    #[test]
    fn test_exploding_gradient_detector() {
        let mut detector = ExplodingGradientDetector::new(5.0);

        let metrics = TrainingMetrics::new(0, 0.5, 0.9).with_gradient_norm(20.0);
        detector.update(0, &metrics);

        assert!(detector.is_converged());
    }

    #[test]
    fn test_gradient_flow_analyzer() {
        let mut analyzer = GradientFlowAnalyzer::new();

        analyzer.update_layer_gradient("layer1".to_string(), 0.5);
        analyzer.update_layer_gradient("layer1".to_string(), 0.6);
        analyzer.update_layer_gradient("layer2".to_string(), 0.3);

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("layer1_mean_grad"));
        assert!(report.metrics.contains_key("layer2_mean_grad"));
    }

    #[test]
    fn test_surrogate_gradient_analyzer() {
        let mut analyzer = SurrogateGradientAnalyzer::new();

        for i in 0..10 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_gradient_norm(0.1);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("mean_surrogate_gradient"));
    }

    #[test]
    fn test_gradient_norm_reset() {
        let mut tracker = GradientNormTracker::new();
        let metrics = TrainingMetrics::new(0, 0.5, 0.9).with_gradient_norm(1.0);
        tracker.update(0, &metrics);

        tracker.reset();
        assert_eq!(tracker.gradient_history.len(), 0);
    }
}
