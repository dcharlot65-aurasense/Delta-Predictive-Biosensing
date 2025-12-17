//! Convergence detection analyzers

use super::{AnalysisReport, ConvergenceAnalyzer, TrainingMetrics};
use std::collections::VecDeque;

/// Detects when training loss stops decreasing (plateaus)
pub struct LossPlateauDetector {
    window_size: usize,
    threshold: f64,
    patience: usize,
    loss_history: VecDeque<f64>,
    converged: bool,
    convergence_epoch: Option<usize>,
    plateau_count: usize,
}

impl LossPlateauDetector {
    pub fn new(window_size: usize, threshold: f64, patience: usize) -> Self {
        Self {
            window_size,
            threshold,
            patience,
            loss_history: VecDeque::new(),
            converged: false,
            convergence_epoch: None,
            plateau_count: 0,
        }
    }
}

impl Default for LossPlateauDetector {
    fn default() -> Self {
        Self::new(10, 1e-4, 5)
    }
}

impl ConvergenceAnalyzer for LossPlateauDetector {
    fn name(&self) -> &str {
        "LossPlateauDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        self.loss_history.push_back(metrics.train_loss);

        if self.loss_history.len() > self.window_size {
            self.loss_history.pop_front();
        }

        if self.loss_history.len() == self.window_size {
            let mean: f64 = self.loss_history.iter().sum::<f64>() / self.window_size as f64;
            let variance: f64 = self.loss_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / self.window_size as f64;

            if variance < self.threshold {
                self.plateau_count += 1;
                if self.plateau_count >= self.patience && !self.converged {
                    self.converged = true;
                    self.convergence_epoch = Some(epoch);
                }
            } else {
                self.plateau_count = 0;
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.convergence_epoch
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());
        report.converged = self.converged;
        report.convergence_epoch = self.convergence_epoch;

        if let Some(&last_loss) = self.loss_history.back() {
            report.add_metric("final_loss", last_loss);
        }

        if self.loss_history.len() == self.window_size {
            let mean: f64 = self.loss_history.iter().sum::<f64>() / self.window_size as f64;
            let variance: f64 = self.loss_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / self.window_size as f64;
            report.add_metric("loss_variance", variance);
        }

        report.add_metric("plateau_count", self.plateau_count as f64);

        if self.converged {
            report.add_recommendation("Training has plateaued. Consider stopping or adjusting hyperparameters.");
        }

        report
    }

    fn reset(&mut self) {
        self.loss_history.clear();
        self.converged = false;
        self.convergence_epoch = None;
        self.plateau_count = 0;
    }
}

/// Detects when accuracy stops improving
pub struct AccuracyPlateauDetector {
    window_size: usize,
    threshold: f64,
    patience: usize,
    accuracy_history: VecDeque<f64>,
    converged: bool,
    convergence_epoch: Option<usize>,
    plateau_count: usize,
}

impl AccuracyPlateauDetector {
    pub fn new(window_size: usize, threshold: f64, patience: usize) -> Self {
        Self {
            window_size,
            threshold,
            patience,
            accuracy_history: VecDeque::new(),
            converged: false,
            convergence_epoch: None,
            plateau_count: 0,
        }
    }
}

impl Default for AccuracyPlateauDetector {
    fn default() -> Self {
        Self::new(10, 1e-3, 5)
    }
}

impl ConvergenceAnalyzer for AccuracyPlateauDetector {
    fn name(&self) -> &str {
        "AccuracyPlateauDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        let accuracy = metrics.val_accuracy.unwrap_or(metrics.train_accuracy);
        self.accuracy_history.push_back(accuracy);

        if self.accuracy_history.len() > self.window_size {
            self.accuracy_history.pop_front();
        }

        if self.accuracy_history.len() == self.window_size {
            let mean: f64 = self.accuracy_history.iter().sum::<f64>() / self.window_size as f64;
            let variance: f64 = self.accuracy_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / self.window_size as f64;

            if variance < self.threshold {
                self.plateau_count += 1;
                if self.plateau_count >= self.patience && !self.converged {
                    self.converged = true;
                    self.convergence_epoch = Some(epoch);
                }
            } else {
                self.plateau_count = 0;
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.convergence_epoch
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());
        report.converged = self.converged;
        report.convergence_epoch = self.convergence_epoch;

        if let Some(&last_acc) = self.accuracy_history.back() {
            report.add_metric("final_accuracy", last_acc);
        }

        if self.accuracy_history.len() == self.window_size {
            let mean: f64 = self.accuracy_history.iter().sum::<f64>() / self.window_size as f64;
            let variance: f64 = self.accuracy_history
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>()
                / self.window_size as f64;
            report.add_metric("accuracy_variance", variance);
        }

        if self.converged {
            report.add_recommendation("Accuracy has plateaued. Model may have reached its capacity.");
        }

        report
    }

    fn reset(&mut self) {
        self.accuracy_history.clear();
        self.converged = false;
        self.convergence_epoch = None;
        self.plateau_count = 0;
    }
}

/// Early stopping analyzer with validation-based detection
pub struct EarlyStoppingAnalyzer {
    patience: usize,
    min_delta: f64,
    best_val_loss: f64,
    best_epoch: usize,
    wait: usize,
    converged: bool,
    convergence_epoch: Option<usize>,
}

impl EarlyStoppingAnalyzer {
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_val_loss: f64::INFINITY,
            best_epoch: 0,
            wait: 0,
            converged: false,
            convergence_epoch: None,
        }
    }
}

impl Default for EarlyStoppingAnalyzer {
    fn default() -> Self {
        Self::new(10, 1e-4)
    }
}

impl ConvergenceAnalyzer for EarlyStoppingAnalyzer {
    fn name(&self) -> &str {
        "EarlyStoppingAnalyzer"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        let val_loss = metrics.val_loss.unwrap_or(metrics.train_loss);

        if val_loss < self.best_val_loss - self.min_delta {
            self.best_val_loss = val_loss;
            self.best_epoch = epoch;
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience && !self.converged {
                self.converged = true;
                self.convergence_epoch = Some(epoch);
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.convergence_epoch
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());
        report.converged = self.converged;
        report.convergence_epoch = self.convergence_epoch;

        report.add_metric("best_val_loss", self.best_val_loss);
        report.add_metric("best_epoch", self.best_epoch as f64);
        report.add_metric("epochs_without_improvement", self.wait as f64);

        if self.converged {
            report.add_recommendation(format!(
                "Early stopping triggered. Best model was at epoch {}.",
                self.best_epoch
            ));
        }

        report
    }

    fn reset(&mut self) {
        self.best_val_loss = f64::INFINITY;
        self.best_epoch = 0;
        self.wait = 0;
        self.converged = false;
        self.convergence_epoch = None;
    }
}

/// Analyzes the rate of convergence
pub struct ConvergenceRateAnalyzer {
    loss_history: Vec<f64>,
    rate_window: usize,
    converged: bool,
    convergence_epoch: Option<usize>,
    convergence_rate: f64,
}

impl ConvergenceRateAnalyzer {
    pub fn new(rate_window: usize) -> Self {
        Self {
            loss_history: Vec::new(),
            rate_window,
            converged: false,
            convergence_epoch: None,
            convergence_rate: 0.0,
        }
    }
}

impl Default for ConvergenceRateAnalyzer {
    fn default() -> Self {
        Self::new(10)
    }
}

impl ConvergenceAnalyzer for ConvergenceRateAnalyzer {
    fn name(&self) -> &str {
        "ConvergenceRateAnalyzer"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        self.loss_history.push(metrics.train_loss);

        if self.loss_history.len() >= self.rate_window {
            let recent = &self.loss_history[self.loss_history.len() - self.rate_window..];
            let rate = (recent[0] - recent[recent.len() - 1]) / self.rate_window as f64;
            self.convergence_rate = rate;

            // Consider converged if improvement rate is very small
            if rate.abs() < 1e-5 && !self.converged {
                self.converged = true;
                self.convergence_epoch = Some(epoch);
            }
        }
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        self.convergence_epoch
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());
        report.converged = self.converged;
        report.convergence_epoch = self.convergence_epoch;

        report.add_metric("convergence_rate", self.convergence_rate);

        if self.loss_history.len() >= 2 {
            let total_improvement = self.loss_history[0] - self.loss_history.last().unwrap();
            report.add_metric("total_improvement", total_improvement);
        }

        if self.convergence_rate < 1e-6 {
            report.add_recommendation("Convergence rate is very slow. Consider increasing learning rate.");
        } else if self.convergence_rate > 0.1 {
            report.add_recommendation("Convergence rate is fast. Training is progressing well.");
        }

        report
    }

    fn reset(&mut self) {
        self.loss_history.clear();
        self.converged = false;
        self.convergence_epoch = None;
        self.convergence_rate = 0.0;
    }
}

/// Detects oscillations in training metrics
pub struct OscillationDetector {
    loss_history: VecDeque<f64>,
    window_size: usize,
    oscillation_threshold: f64,
    detected: bool,
    detection_epoch: Option<usize>,
    oscillation_count: usize,
}

impl OscillationDetector {
    pub fn new(window_size: usize, oscillation_threshold: f64) -> Self {
        Self {
            loss_history: VecDeque::new(),
            window_size,
            oscillation_threshold,
            detected: false,
            detection_epoch: None,
            oscillation_count: 0,
        }
    }
}

impl Default for OscillationDetector {
    fn default() -> Self {
        Self::new(20, 0.1)
    }
}

impl ConvergenceAnalyzer for OscillationDetector {
    fn name(&self) -> &str {
        "OscillationDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        self.loss_history.push_back(metrics.train_loss);

        if self.loss_history.len() > self.window_size {
            self.loss_history.pop_front();
        }

        if self.loss_history.len() >= 4 {
            // Count sign changes in loss differences
            let mut sign_changes = 0;
            let losses: Vec<f64> = self.loss_history.iter().copied().collect();

            for i in 1..losses.len() - 1 {
                let diff1 = losses[i] - losses[i - 1];
                let diff2 = losses[i + 1] - losses[i];

                if diff1.signum() != diff2.signum() && diff1.abs() > self.oscillation_threshold {
                    sign_changes += 1;
                }
            }

            let oscillation_ratio = sign_changes as f64 / (losses.len() - 2) as f64;

            if oscillation_ratio > 0.5 && !self.detected {
                self.detected = true;
                self.detection_epoch = Some(epoch);
            }

            self.oscillation_count = sign_changes;
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

        report.add_metric("oscillation_count", self.oscillation_count as f64);

        if self.detected {
            report.add_recommendation("Training is oscillating. Consider reducing learning rate or adjusting optimizer.");
        }

        report
    }

    fn reset(&mut self) {
        self.loss_history.clear();
        self.detected = false;
        self.detection_epoch = None;
        self.oscillation_count = 0;
    }
}

/// Detects divergence in training (loss increasing)
pub struct DivergenceDetector {
    loss_history: VecDeque<f64>,
    window_size: usize,
    divergence_threshold: f64,
    detected: bool,
    detection_epoch: Option<usize>,
}

impl DivergenceDetector {
    pub fn new(window_size: usize, divergence_threshold: f64) -> Self {
        Self {
            loss_history: VecDeque::new(),
            window_size,
            divergence_threshold,
            detected: false,
            detection_epoch: None,
        }
    }
}

impl Default for DivergenceDetector {
    fn default() -> Self {
        Self::new(5, 2.0)
    }
}

impl ConvergenceAnalyzer for DivergenceDetector {
    fn name(&self) -> &str {
        "DivergenceDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        self.loss_history.push_back(metrics.train_loss);

        if self.loss_history.len() > self.window_size {
            self.loss_history.pop_front();
        }

        if self.loss_history.len() == self.window_size {
            let first = self.loss_history[0];
            let last = *self.loss_history.back().unwrap();

            // Check if loss has increased beyond threshold or is NaN/infinite
            if last.is_nan() || last.is_infinite() || (last / first > self.divergence_threshold) {
                if !self.detected {
                    self.detected = true;
                    self.detection_epoch = Some(epoch);
                }
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

        if let (Some(&first), Some(&last)) = (self.loss_history.front(), self.loss_history.back()) {
            report.add_metric("loss_ratio", last / first);
            report.add_metric("first_loss", first);
            report.add_metric("last_loss", last);
        }

        if self.detected {
            report.add_recommendation("Training is diverging! Reduce learning rate significantly or restart training.");
        }

        report
    }

    fn reset(&mut self) {
        self.loss_history.clear();
        self.detected = false;
        self.detection_epoch = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_loss_plateau_detector() {
        let mut detector = LossPlateauDetector::new(3, 1e-4, 2);

        for i in 0..10 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.9);
            detector.update(i, &metrics);
        }

        assert!(detector.is_converged());
        assert!(detector.convergence_epoch().is_some());
    }

    #[test]
    fn test_early_stopping() {
        let mut analyzer = EarlyStoppingAnalyzer::new(3, 1e-4);

        // Improving loss
        for i in 0..5 {
            let metrics = TrainingMetrics::new(i, 1.0 - i as f64 * 0.1, 0.9)
                .with_val_loss(1.0 - i as f64 * 0.1);
            analyzer.update(i, &metrics);
        }

        assert!(!analyzer.is_converged());

        // Plateau
        for i in 5..10 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_val_loss(0.5);
            analyzer.update(i, &metrics);
        }

        assert!(analyzer.is_converged());
    }

    #[test]
    fn test_divergence_detector() {
        let mut detector = DivergenceDetector::new(3, 2.0);

        let metrics1 = TrainingMetrics::new(0, 1.0, 0.9);
        detector.update(0, &metrics1);

        let metrics2 = TrainingMetrics::new(1, 1.5, 0.9);
        detector.update(1, &metrics2);

        let metrics3 = TrainingMetrics::new(2, 3.0, 0.9);
        detector.update(2, &metrics3);

        assert!(detector.is_converged());
    }

    #[test]
    fn test_oscillation_detector() {
        let mut detector = OscillationDetector::new(10, 0.05);

        let losses = vec![1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 1.0, 0.8];
        for (i, &loss) in losses.iter().enumerate() {
            let metrics = TrainingMetrics::new(i, loss, 0.9);
            detector.update(i, &metrics);
        }

        assert!(detector.is_converged());
    }

    #[test]
    fn test_convergence_rate_analyzer() {
        let mut analyzer = ConvergenceRateAnalyzer::new(5);

        for i in 0..20 {
            let loss = 1.0 / (i as f64 + 1.0);
            let metrics = TrainingMetrics::new(i, loss, 0.9);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("convergence_rate"));
    }

    #[test]
    fn test_accuracy_plateau_detector() {
        let mut detector = AccuracyPlateauDetector::new(3, 1e-4, 2);

        for i in 0..10 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.95);
            detector.update(i, &metrics);
        }

        assert!(detector.is_converged());
    }
}
