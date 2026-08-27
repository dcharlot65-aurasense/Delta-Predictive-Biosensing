//! Learning curve analysis

use super::{AnalysisReport, ConvergenceAnalyzer, TrainingMetrics};

/// Provides smoothed learning curves with exponential moving average
pub struct LearningCurveSmoothed {
    alpha: f64,
    smoothed_loss: Option<f64>,
    smoothed_accuracy: Option<f64>,
    loss_history: Vec<f64>,
    accuracy_history: Vec<f64>,
    converged: bool,
}

impl LearningCurveSmoothed {
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            smoothed_loss: None,
            smoothed_accuracy: None,
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
            converged: false,
        }
    }
}

impl Default for LearningCurveSmoothed {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl ConvergenceAnalyzer for LearningCurveSmoothed {
    fn name(&self) -> &str {
        "LearningCurveSmoothed"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Update smoothed loss
        self.smoothed_loss = Some(match self.smoothed_loss {
            None => metrics.train_loss,
            Some(prev) => self.alpha * metrics.train_loss + (1.0 - self.alpha) * prev,
        });

        // Update smoothed accuracy
        self.smoothed_accuracy = Some(match self.smoothed_accuracy {
            None => metrics.train_accuracy,
            Some(prev) => self.alpha * metrics.train_accuracy + (1.0 - self.alpha) * prev,
        });

        self.loss_history.push(metrics.train_loss);
        self.accuracy_history.push(metrics.train_accuracy);
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if let Some(smoothed_loss) = self.smoothed_loss {
            report.add_metric("smoothed_loss", smoothed_loss);
        }

        if let Some(smoothed_acc) = self.smoothed_accuracy {
            report.add_metric("smoothed_accuracy", smoothed_acc);
        }

        // Calculate noise in raw curves
        if self.loss_history.len() > 1 {
            let loss_variance: f64 = self
                .loss_history
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .sum::<f64>()
                / (self.loss_history.len() - 1) as f64;
            report.add_metric("loss_variance", loss_variance);
        }

        report
    }

    fn reset(&mut self) {
        self.smoothed_loss = None;
        self.smoothed_accuracy = None;
        self.loss_history.clear();
        self.accuracy_history.clear();
        self.converged = false;
    }
}

/// Analyzes the generalization gap between training and validation
pub struct GeneralizationGapAnalyzer {
    gap_history: Vec<f64>,
    gap_threshold: f64,
    converged: bool,
    convergence_epoch: Option<usize>,
}

impl GeneralizationGapAnalyzer {
    pub fn new(gap_threshold: f64) -> Self {
        Self {
            gap_history: Vec::new(),
            gap_threshold,
            converged: false,
            convergence_epoch: None,
        }
    }
}

impl Default for GeneralizationGapAnalyzer {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl ConvergenceAnalyzer for GeneralizationGapAnalyzer {
    fn name(&self) -> &str {
        "GeneralizationGapAnalyzer"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        if let Some(val_loss) = metrics.val_loss {
            let gap = val_loss - metrics.train_loss;
            self.gap_history.push(gap);

            // Detect if gap is growing (potential overfitting)
            if gap > self.gap_threshold && !self.converged {
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

        if let Some(&current_gap) = self.gap_history.last() {
            report.add_metric("current_gap", current_gap);
        }

        if self.gap_history.len() > 1 {
            let mean_gap = self.gap_history.iter().sum::<f64>() / self.gap_history.len() as f64;
            report.add_metric("mean_gap", mean_gap);

            // Check if gap is increasing
            let recent_gap = self.gap_history[self.gap_history.len() - 1];
            let early_gap = self.gap_history[0];
            if recent_gap > early_gap * 1.5 {
                report.add_recommendation(
                    "Generalization gap is increasing. Consider regularization.",
                );
            }
        }

        report
    }

    fn reset(&mut self) {
        self.gap_history.clear();
        self.converged = false;
        self.convergence_epoch = None;
    }
}

/// Detects overfitting by monitoring validation performance
pub struct OverfittingDetector {
    patience: usize,
    best_val_loss: f64,
    epochs_worse: usize,
    converged: bool,
    convergence_epoch: Option<usize>,
}

impl OverfittingDetector {
    pub fn new(patience: usize) -> Self {
        Self {
            patience,
            best_val_loss: f64::INFINITY,
            epochs_worse: 0,
            converged: false,
            convergence_epoch: None,
        }
    }
}

impl Default for OverfittingDetector {
    fn default() -> Self {
        Self::new(5)
    }
}

impl ConvergenceAnalyzer for OverfittingDetector {
    fn name(&self) -> &str {
        "OverfittingDetector"
    }

    fn update(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        let val_loss = metrics.val_loss.unwrap_or(metrics.train_loss);

        if val_loss < self.best_val_loss {
            self.best_val_loss = val_loss;
            self.epochs_worse = 0;
        } else {
            self.epochs_worse += 1;
            if self.epochs_worse >= self.patience && !self.converged {
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
        report.add_metric("epochs_without_improvement", self.epochs_worse as f64);

        if self.converged {
            report.add_recommendation(
                "Overfitting detected. Consider early stopping or regularization.",
            );
        }

        report
    }

    fn reset(&mut self) {
        self.best_val_loss = f64::INFINITY;
        self.epochs_worse = 0;
        self.converged = false;
        self.convergence_epoch = None;
    }
}

/// Analyzes optimal learning rate based on loss trajectory
pub struct LearningRateAnalyzer {
    lr_history: Vec<f64>,
    loss_history: Vec<f64>,
    optimal_lr: Option<f64>,
    converged: bool,
}

impl LearningRateAnalyzer {
    pub fn new() -> Self {
        Self {
            lr_history: Vec::new(),
            loss_history: Vec::new(),
            optimal_lr: None,
            converged: false,
        }
    }
}

impl Default for LearningRateAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for LearningRateAnalyzer {
    fn name(&self) -> &str {
        "LearningRateAnalyzer"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        self.lr_history.push(metrics.learning_rate);
        self.loss_history.push(metrics.train_loss);

        // Find LR with steepest loss decrease
        if self.loss_history.len() > 2 {
            let mut best_rate = 0.0;
            let mut best_idx = 0;

            for i in 1..self.loss_history.len() {
                let rate = (self.loss_history[i - 1] - self.loss_history[i])
                    / self.loss_history[i - 1].max(1e-10);
                if rate > best_rate {
                    best_rate = rate;
                    best_idx = i;
                }
            }

            if best_idx > 0 && best_idx < self.lr_history.len() {
                self.optimal_lr = Some(self.lr_history[best_idx]);
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

        if let Some(optimal_lr) = self.optimal_lr {
            report.add_metric("optimal_learning_rate", optimal_lr);
        }

        if let Some(&current_lr) = self.lr_history.last() {
            report.add_metric("current_learning_rate", current_lr);

            if let Some(optimal_lr) = self.optimal_lr {
                if current_lr > optimal_lr * 2.0 {
                    report
                        .add_recommendation("Learning rate may be too high. Consider reducing it.");
                } else if current_lr < optimal_lr * 0.5 {
                    report.add_recommendation(
                        "Learning rate may be too low. Consider increasing it.",
                    );
                }
            }
        }

        report
    }

    fn reset(&mut self) {
        self.lr_history.clear();
        self.loss_history.clear();
        self.optimal_lr = None;
        self.converged = false;
    }
}

/// Analyzes the effect of batch size on training
pub struct BatchSizeAnalyzer {
    loss_per_batch: Vec<(usize, f64)>,
    converged: bool,
}

impl BatchSizeAnalyzer {
    pub fn new() -> Self {
        Self {
            loss_per_batch: Vec::new(),
            converged: false,
        }
    }

    pub fn update_with_batch_size(&mut self, batch_size: usize, loss: f64) {
        self.loss_per_batch.push((batch_size, loss));
    }
}

impl Default for BatchSizeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for BatchSizeAnalyzer {
    fn name(&self) -> &str {
        "BatchSizeAnalyzer"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        // Store metrics with default batch size indicator
        self.loss_per_batch.push((0, metrics.train_loss));
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        if !self.loss_per_batch.is_empty() {
            let mean_loss = self.loss_per_batch.iter().map(|(_, l)| l).sum::<f64>()
                / self.loss_per_batch.len() as f64;
            report.add_metric("mean_loss", mean_loss);
        }

        report.add_recommendation("Experiment with different batch sizes for optimal performance.");

        report
    }

    fn reset(&mut self) {
        self.loss_per_batch.clear();
        self.converged = false;
    }
}

/// Analyzes training efficiency per epoch
pub struct EpochEfficiencyAnalyzer {
    epoch_losses: Vec<f64>,
    efficiency_scores: Vec<f64>,
    converged: bool,
}

impl EpochEfficiencyAnalyzer {
    pub fn new() -> Self {
        Self {
            epoch_losses: Vec::new(),
            efficiency_scores: Vec::new(),
            converged: false,
        }
    }
}

impl Default for EpochEfficiencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for EpochEfficiencyAnalyzer {
    fn name(&self) -> &str {
        "EpochEfficiencyAnalyzer"
    }

    fn update(&mut self, _epoch: usize, metrics: &TrainingMetrics) {
        self.epoch_losses.push(metrics.train_loss);

        if self.epoch_losses.len() > 1 {
            let improvement = self.epoch_losses[self.epoch_losses.len() - 2]
                - self.epoch_losses[self.epoch_losses.len() - 1];
            let efficiency =
                improvement / self.epoch_losses[self.epoch_losses.len() - 2].max(1e-10);
            self.efficiency_scores.push(efficiency);
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

        if !self.efficiency_scores.is_empty() {
            let mean_efficiency =
                self.efficiency_scores.iter().sum::<f64>() / self.efficiency_scores.len() as f64;
            report.add_metric("mean_epoch_efficiency", mean_efficiency);

            if let Some(&recent_efficiency) = self.efficiency_scores.last() {
                report.add_metric("recent_epoch_efficiency", recent_efficiency);

                if recent_efficiency < 0.01 {
                    report.add_recommendation(
                        "Epoch efficiency is low. Consider stopping or adjusting hyperparameters.",
                    );
                }
            }
        }

        report
    }

    fn reset(&mut self) {
        self.epoch_losses.clear();
        self.efficiency_scores.clear();
        self.converged = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_curve_smoothed() {
        let mut analyzer = LearningCurveSmoothed::new(0.1);

        for i in 0..10 {
            let loss = 1.0 - i as f64 * 0.05;
            let metrics = TrainingMetrics::new(i, loss, 0.9);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("smoothed_loss"));
        assert!(report.metrics.contains_key("smoothed_accuracy"));
    }

    #[test]
    fn test_generalization_gap() {
        let mut analyzer = GeneralizationGapAnalyzer::new(0.2);

        for i in 0..10 {
            let train_loss = 0.5 - i as f64 * 0.01;
            let val_loss = train_loss + 0.3;
            let metrics = TrainingMetrics::new(i, train_loss, 0.9).with_val_loss(val_loss);
            analyzer.update(i, &metrics);
        }

        assert!(analyzer.is_converged());
    }

    #[test]
    fn test_overfitting_detector() {
        let mut detector = OverfittingDetector::new(3);

        // Decreasing val loss
        for i in 0..5 {
            let metrics = TrainingMetrics::new(i, 0.5, 0.9).with_val_loss(1.0 - i as f64 * 0.1);
            detector.update(i, &metrics);
        }

        // Increasing val loss
        for i in 5..10 {
            let metrics =
                TrainingMetrics::new(i, 0.5, 0.9).with_val_loss(0.5 + (i - 5) as f64 * 0.1);
            detector.update(i, &metrics);
        }

        assert!(detector.is_converged());
    }

    #[test]
    fn test_learning_rate_analyzer() {
        let mut analyzer = LearningRateAnalyzer::new();

        for i in 0..10 {
            let loss = 1.0 / (i as f64 + 1.0);
            let lr = 0.001 * (1.0 + i as f64 * 0.1);
            let metrics = TrainingMetrics::new(i, loss, 0.9).with_learning_rate(lr);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("current_learning_rate"));
    }

    #[test]
    fn test_epoch_efficiency() {
        let mut analyzer = EpochEfficiencyAnalyzer::new();

        for i in 0..10 {
            let loss = 1.0 - i as f64 * 0.05;
            let metrics = TrainingMetrics::new(i, loss, 0.9);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("mean_epoch_efficiency"));
    }

    #[test]
    fn test_batch_size_analyzer() {
        let mut analyzer = BatchSizeAnalyzer::new();

        for i in 0..5 {
            let metrics = TrainingMetrics::new(i, 0.5 - i as f64 * 0.05, 0.9);
            analyzer.update(i, &metrics);
        }

        let report = analyzer.analysis_report();
        assert!(report.metrics.contains_key("mean_loss"));
    }
}
