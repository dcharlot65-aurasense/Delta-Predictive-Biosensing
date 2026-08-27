//! Spike pattern and statistics analysis

use super::{AnalysisReport, ConvergenceAnalyzer, TrainingMetrics};
use std::collections::HashMap;

/// Tracks firing rates across the network during training
pub struct SpikeRateTracker {
    spike_rate_history: Vec<f64>,
    mean_spike_rate: f64,
    target_spike_rate: f64,
    converged: bool,
}

impl SpikeRateTracker {
    pub fn new(target_spike_rate: f64) -> Self {
        Self {
            spike_rate_history: Vec::new(),
            mean_spike_rate: 0.0,
            target_spike_rate,
            converged: false,
        }
    }
}

impl Default for SpikeRateTracker {
    fn default() -> Self {
        Self::new(0.05) // 5% target firing rate
    }
}

impl ConvergenceAnalyzer for SpikeRateTracker {
    fn name(&self) -> &str {
        "SpikeRateTracker"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        self.spike_rate_history.push(metrics.spike_rate);

        if !self.spike_rate_history.is_empty() {
            self.mean_spike_rate =
                self.spike_rate_history.iter().sum::<f64>() / self.spike_rate_history.len() as f64;
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric("mean_spike_rate", self.mean_spike_rate);
        report.add_metric("target_spike_rate", self.target_spike_rate);

        if let Some(&current_rate) = self.spike_rate_history.last() {
            report.add_metric("current_spike_rate", current_rate);

            let deviation = (current_rate - self.target_spike_rate).abs() / self.target_spike_rate;

            if current_rate < 0.001 {
                report.add_recommendation("Spike rate is very low. Network may be inactive.");
            } else if current_rate > 0.5 {
                report.add_recommendation("Spike rate is very high. Network may be over-active.");
            } else if deviation > 0.5 {
                report.add_recommendation(format!(
                    "Spike rate deviates significantly from target. Current: {:.4}, Target: {:.4}",
                    current_rate, self.target_spike_rate
                ));
            }
        }

        report
    }

    fn reset(&mut self) {
        self.spike_rate_history.clear();
        self.mean_spike_rate = 0.0;
        self.converged = false;
    }
}

/// Tracks activation sparsity during training
pub struct SparsityTracker {
    sparsity_history: Vec<f64>,
    target_sparsity: f64,
    converged: bool,
}

impl SparsityTracker {
    pub fn new(target_sparsity: f64) -> Self {
        Self {
            sparsity_history: Vec::new(),
            target_sparsity,
            converged: false,
        }
    }
}

impl Default for SparsityTracker {
    fn default() -> Self {
        Self::new(0.9) // 90% sparsity target
    }
}

impl ConvergenceAnalyzer for SparsityTracker {
    fn name(&self) -> &str {
        "SparsityTracker"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Sparsity is 1 - spike_rate
        let sparsity = 1.0 - metrics.spike_rate;
        self.sparsity_history.push(sparsity);
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if !self.sparsity_history.is_empty() {
            let mean_sparsity =
                self.sparsity_history.iter().sum::<f64>() / self.sparsity_history.len() as f64;
            report.add_metric("mean_sparsity", mean_sparsity);

            if let Some(&current_sparsity) = self.sparsity_history.last() {
                report.add_metric("current_sparsity", current_sparsity);

                if current_sparsity < 0.5 {
                    report.add_recommendation(
                        "Low sparsity. Network is not sparse enough for efficient SNN computation.",
                    );
                } else if current_sparsity > 0.99 {
                    report.add_recommendation("Very high sparsity. Network may be too inactive.");
                }
            }
        }

        report.add_metric("target_sparsity", self.target_sparsity);
        report
    }

    fn reset(&mut self) {
        self.sparsity_history.clear();
        self.converged = false;
    }
}

/// Detects silent (never firing) neurons
pub struct SilentNeuronDetector {
    neuron_spike_counts: HashMap<usize, usize>,
    total_updates: usize,
    silence_threshold: usize,
    detected_silent: Vec<usize>,
    converged: bool,
}

impl SilentNeuronDetector {
    pub fn new(silence_threshold: usize) -> Self {
        Self {
            neuron_spike_counts: HashMap::new(),
            total_updates: 0,
            silence_threshold,
            detected_silent: Vec::new(),
            converged: false,
        }
    }

    pub fn update_neuron_activity(&mut self, neuron_id: usize, spike_count: usize) {
        *self.neuron_spike_counts.entry(neuron_id).or_insert(0) += spike_count;
    }
}

impl Default for SilentNeuronDetector {
    fn default() -> Self {
        Self::new(10) // Silent if no spikes for 10 epochs
    }
}

impl ConvergenceAnalyzer for SilentNeuronDetector {
    fn name(&self) -> &str {
        "SilentNeuronDetector"
    }

    fn update(&mut self, _epoch: usize, _metrics: &TrainingMetrics) {
        self.total_updates += 1;

        if self.total_updates >= self.silence_threshold {
            self.detected_silent.clear();
            for (&neuron_id, &count) in &self.neuron_spike_counts {
                if count == 0 {
                    self.detected_silent.push(neuron_id);
                }
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric(
            "total_neurons_tracked",
            self.neuron_spike_counts.len() as f64,
        );
        report.add_metric("silent_neurons_detected", self.detected_silent.len() as f64);

        if !self.detected_silent.is_empty() {
            let silent_ratio =
                self.detected_silent.len() as f64 / self.neuron_spike_counts.len() as f64;
            report.add_metric("silent_neuron_ratio", silent_ratio);

            if silent_ratio > 0.1 {
                report.add_recommendation(format!(
                    "{} silent neurons detected ({:.1}% of network). Consider:",
                    self.detected_silent.len(),
                    silent_ratio * 100.0
                ));
                report.add_recommendation("- Adjusting neuron thresholds");
                report.add_recommendation("- Checking weight initialization");
                report.add_recommendation("- Increasing input stimulus");
            }
        }

        report
    }

    fn reset(&mut self) {
        self.neuron_spike_counts.clear();
        self.total_updates = 0;
        self.detected_silent.clear();
        self.converged = false;
    }
}

/// Detects saturated (always firing) neurons
pub struct SaturatedNeuronDetector {
    neuron_activity_ratios: HashMap<usize, f64>,
    saturation_threshold: f64,
    detected_saturated: Vec<usize>,
    converged: bool,
}

impl SaturatedNeuronDetector {
    pub fn new(saturation_threshold: f64) -> Self {
        Self {
            neuron_activity_ratios: HashMap::new(),
            saturation_threshold,
            detected_saturated: Vec::new(),
            converged: false,
        }
    }

    pub fn update_neuron_ratio(&mut self, neuron_id: usize, activity_ratio: f64) {
        self.neuron_activity_ratios
            .insert(neuron_id, activity_ratio);
    }
}

impl Default for SaturatedNeuronDetector {
    fn default() -> Self {
        Self::new(0.95) // Saturated if firing >95% of the time
    }
}

impl ConvergenceAnalyzer for SaturatedNeuronDetector {
    fn name(&self) -> &str {
        "SaturatedNeuronDetector"
    }

    fn update(&mut self, _epoch: usize, _metrics: &TrainingMetrics) {
        self.detected_saturated.clear();

        for (&neuron_id, &ratio) in &self.neuron_activity_ratios {
            if ratio >= self.saturation_threshold {
                self.detected_saturated.push(neuron_id);
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric(
            "total_neurons_tracked",
            self.neuron_activity_ratios.len() as f64,
        );
        report.add_metric(
            "saturated_neurons_detected",
            self.detected_saturated.len() as f64,
        );

        if !self.detected_saturated.is_empty() {
            let saturated_ratio =
                self.detected_saturated.len() as f64 / self.neuron_activity_ratios.len() as f64;
            report.add_metric("saturated_neuron_ratio", saturated_ratio);

            if saturated_ratio > 0.1 {
                report.add_recommendation(format!(
                    "{} saturated neurons detected ({:.1}% of network). Consider:",
                    self.detected_saturated.len(),
                    saturated_ratio * 100.0
                ));
                report.add_recommendation("- Increasing neuron thresholds");
                report.add_recommendation("- Reducing input strength");
                report.add_recommendation("- Adding inhibitory connections");
            }
        }

        report
    }

    fn reset(&mut self) {
        self.neuron_activity_ratios.clear();
        self.detected_saturated.clear();
        self.converged = false;
    }
}

/// Analyzes temporal dynamics of spike patterns
pub struct TemporalDynamicsAnalyzer {
    spike_timing_variance: Vec<f64>,
    synchrony_measures: Vec<f64>,
    converged: bool,
}

impl TemporalDynamicsAnalyzer {
    pub fn new() -> Self {
        Self {
            spike_timing_variance: Vec::new(),
            synchrony_measures: Vec::new(),
            converged: false,
        }
    }

    pub fn update_timing_variance(&mut self, variance: f64) {
        self.spike_timing_variance.push(variance);
    }

    pub fn update_synchrony(&mut self, synchrony: f64) {
        self.synchrony_measures.push(synchrony);
    }
}

impl Default for TemporalDynamicsAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for TemporalDynamicsAnalyzer {
    fn name(&self) -> &str {
        "TemporalDynamicsAnalyzer"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Use spike rate as a proxy for temporal dynamics
        // Higher variance in spike rate indicates more dynamic temporal patterns
        self.spike_timing_variance.push(metrics.spike_rate);
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if !self.spike_timing_variance.is_empty() {
            let mean_variance = self.spike_timing_variance.iter().sum::<f64>()
                / self.spike_timing_variance.len() as f64;
            report.add_metric("mean_timing_variance", mean_variance);

            // Calculate actual variance of spike rates
            let variance: f64 = self
                .spike_timing_variance
                .iter()
                .map(|&x| (x - mean_variance).powi(2))
                .sum::<f64>()
                / self.spike_timing_variance.len() as f64;
            report.add_metric("spike_rate_variance", variance);

            if variance < 1e-6 {
                report.add_recommendation(
                    "Very low temporal variance. Spike patterns may be too regular.",
                );
            }
        }

        if !self.synchrony_measures.is_empty() {
            let mean_synchrony =
                self.synchrony_measures.iter().sum::<f64>() / self.synchrony_measures.len() as f64;
            report.add_metric("mean_synchrony", mean_synchrony);

            if mean_synchrony > 0.9 {
                report.add_recommendation(
                    "High synchrony detected. Network may lack diversity in spike patterns.",
                );
            }
        }

        report
    }

    fn reset(&mut self) {
        self.spike_timing_variance.clear();
        self.synchrony_measures.clear();
        self.converged = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_rate_tracker() {
        let mut tracker = SpikeRateTracker::new(0.05);

        for i in 0..10 {
            let spike_rate = 0.05 + i as f64 * 0.01;
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_spike_rate(spike_rate);
            tracker.update(i, &metrics);
        }

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("mean_spike_rate"));
        assert!(report.metrics.contains_key("current_spike_rate"));
    }

    #[test]
    fn test_sparsity_tracker() {
        let mut tracker = SparsityTracker::new(0.9);

        for i in 0..10 {
            let spike_rate = 0.1; // 90% sparsity
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_spike_rate(spike_rate);
            tracker.update(i, &metrics);
        }

        let report = tracker.analysis_report();
        assert!(report.metrics.contains_key("mean_sparsity"));
        assert!(report.metrics.contains_key("current_sparsity"));
    }

    #[test]
    fn test_silent_neuron_detector() {
        let mut detector = SilentNeuronDetector::new(5);

        detector.update_neuron_activity(0, 10);
        detector.update_neuron_activity(1, 0);
        detector.update_neuron_activity(2, 5);
        detector.update_neuron_activity(3, 0);

        for i in 0..10 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.9);
            detector.update(i, &metrics);
        }

        let report = detector.analysis_report();
        assert!(report.metrics.contains_key("silent_neurons_detected"));
        assert_eq!(report.metrics.get("silent_neurons_detected"), Some(&2.0));
    }

    #[test]
    fn test_saturated_neuron_detector() {
        let mut detector = SaturatedNeuronDetector::new(0.95);

        detector.update_neuron_ratio(0, 0.5);
        detector.update_neuron_ratio(1, 0.98);
        detector.update_neuron_ratio(2, 0.96);

        let metrics = TrainingMetrics::new(0, 0.5, 0.9);
        detector.update(0, &metrics);

        let report = detector.analysis_report();
        assert!(report.metrics.contains_key("saturated_neurons_detected"));
        assert_eq!(report.metrics.get("saturated_neurons_detected"), Some(&2.0));
    }

    #[test]
    fn test_temporal_dynamics_analyzer() {
        let mut analyzer = TemporalDynamicsAnalyzer::new();

        for i in 0..10 {
            let spike_rate = 0.05 + (i as f64 * 0.01);
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_spike_rate(spike_rate);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("mean_timing_variance"));
        assert!(report.metrics.contains_key("spike_rate_variance"));
    }

    #[test]
    fn test_spike_rate_recommendations() {
        let mut tracker = SpikeRateTracker::new(0.05);

        let metrics = TrainingMetrics::new(0, 0.5, 0.9).with_spike_rate(0.6);
        tracker.update(0, &metrics);

        let report = tracker.analysis_report();
        assert!(!report.recommendations.is_empty());
    }
}
