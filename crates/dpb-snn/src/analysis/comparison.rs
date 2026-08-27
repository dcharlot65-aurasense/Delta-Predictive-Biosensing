//! Method comparison and hyperparameter sensitivity analysis

use super::{AnalysisReport, ConvergenceAnalyzer, TrainingMetrics};
use std::collections::HashMap;

/// Compares different training methods/runs
pub struct MethodComparisonAnalyzer {
    method_histories: HashMap<String, Vec<TrainingMetrics>>,
    converged: bool,
}

impl MethodComparisonAnalyzer {
    pub fn new() -> Self {
        Self {
            method_histories: HashMap::new(),
            converged: false,
        }
    }

    /// Adds a training run for a specific method
    pub fn add_method_run(&mut self, method_name: String, metrics: Vec<TrainingMetrics>) {
        self.method_histories.insert(method_name, metrics);
    }

    /// Updates a method with new metrics
    pub fn update_method(&mut self, method_name: String, metrics: TrainingMetrics) {
        self.method_histories
            .entry(method_name)
            .or_default()
            .push(metrics);
    }

    /// Compares final performance of all methods
    pub fn compare_final_performance(&self) -> HashMap<String, (f64, f64)> {
        let mut results = HashMap::new();

        for (method_name, history) in &self.method_histories {
            if let Some(last_metrics) = history.last() {
                results.insert(
                    method_name.clone(),
                    (last_metrics.train_loss, last_metrics.train_accuracy),
                );
            }
        }

        results
    }

    /// Finds the best performing method based on validation loss
    pub fn best_method(&self) -> Option<(String, f64)> {
        let mut best_method = None;
        let mut best_loss = f64::INFINITY;

        for (method_name, history) in &self.method_histories {
            if let Some(last_metrics) = history.last() {
                let loss = last_metrics.val_loss.unwrap_or(last_metrics.train_loss);
                if loss < best_loss {
                    best_loss = loss;
                    best_method = Some((method_name.clone(), loss));
                }
            }
        }

        best_method
    }

    /// Compares convergence speed (epochs to reach target loss)
    pub fn compare_convergence_speed(&self, target_loss: f64) -> HashMap<String, Option<usize>> {
        let mut results = HashMap::new();

        for (method_name, history) in &self.method_histories {
            let convergence_epoch = history
                .iter()
                .position(|m| m.train_loss <= target_loss)
                .map(|idx| history[idx].epoch);

            results.insert(method_name.clone(), convergence_epoch);
        }

        results
    }
}

impl Default for MethodComparisonAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for MethodComparisonAnalyzer {
    fn name(&self) -> &str {
        "MethodComparisonAnalyzer"
    }

    fn update(&mut self, _epoch: usize, _metrics: &TrainingMetrics) {
        // Methods are added via add_method_run or update_method
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric("num_methods_compared", self.method_histories.len() as f64);

        // Compare final performance
        let final_performance = self.compare_final_performance();
        for (method, (loss, accuracy)) in &final_performance {
            report.add_metric(format!("{}_final_loss", method), *loss);
            report.add_metric(format!("{}_final_accuracy", method), *accuracy);
        }

        // Find best method
        if let Some((best_method, best_loss)) = self.best_method() {
            report.add_detail("best_method", best_method.clone());
            report.add_metric("best_method_loss", best_loss);

            report.add_recommendation(format!(
                "Best performing method: {} with loss {:.4}",
                best_method, best_loss
            ));
        }

        // Compare convergence speeds
        let convergence_speeds = self.compare_convergence_speed(0.1);
        for (method, epoch_opt) in &convergence_speeds {
            if let Some(epoch) = epoch_opt {
                report.add_metric(format!("{}_convergence_epoch", method), *epoch as f64);
            }
        }

        // Provide comparative recommendations
        if self.method_histories.len() >= 2 {
            report.add_recommendation(
                "Multiple methods compared. Review final losses and convergence speeds above.",
            );

            // Find fastest converging method
            let mut fastest_method = None;
            let mut fastest_epoch = usize::MAX;

            for (method, epoch_opt) in &convergence_speeds {
                if let Some(epoch) = epoch_opt
                    && *epoch < fastest_epoch
                {
                    fastest_epoch = *epoch;
                    fastest_method = Some(method.clone());
                }
            }

            if let Some(fastest) = fastest_method {
                report.add_recommendation(format!(
                    "Fastest converging method: {} (epoch {})",
                    fastest, fastest_epoch
                ));
            }
        }

        report
    }

    fn reset(&mut self) {
        self.method_histories.clear();
        self.converged = false;
    }
}

/// Analyzes sensitivity to hyperparameter changes
pub struct HyperparameterSensitivityAnalyzer {
    hp_configurations: HashMap<String, (HashMap<String, f64>, Vec<TrainingMetrics>)>,
    converged: bool,
}

impl HyperparameterSensitivityAnalyzer {
    pub fn new() -> Self {
        Self {
            hp_configurations: HashMap::new(),
            converged: false,
        }
    }

    /// Adds a hyperparameter configuration with its training history
    pub fn add_configuration(
        &mut self,
        config_name: String,
        hyperparameters: HashMap<String, f64>,
        metrics_history: Vec<TrainingMetrics>,
    ) {
        self.hp_configurations
            .insert(config_name, (hyperparameters, metrics_history));
    }

    /// Updates a configuration with new metrics
    pub fn update_configuration(&mut self, config_name: String, metrics: TrainingMetrics) {
        if let Some((_, history)) = self.hp_configurations.get_mut(&config_name) {
            history.push(metrics);
        }
    }

    /// Analyzes which hyperparameter has the most impact
    pub fn analyze_sensitivity(&self, hp_name: &str) -> Option<(f64, f64)> {
        let mut hp_values = Vec::new();
        let mut final_losses = Vec::new();

        for (hps, history) in self.hp_configurations.values() {
            if let Some(&hp_value) = hps.get(hp_name)
                && let Some(last_metrics) = history.last()
            {
                hp_values.push(hp_value);
                final_losses.push(last_metrics.train_loss);
            }
        }

        if hp_values.len() >= 2 {
            // Calculate correlation between HP value and loss
            let mean_hp = hp_values.iter().sum::<f64>() / hp_values.len() as f64;
            let mean_loss = final_losses.iter().sum::<f64>() / final_losses.len() as f64;

            let covariance: f64 = hp_values
                .iter()
                .zip(final_losses.iter())
                .map(|(&hp, &loss)| (hp - mean_hp) * (loss - mean_loss))
                .sum::<f64>()
                / hp_values.len() as f64;

            let hp_variance: f64 = hp_values
                .iter()
                .map(|&hp| (hp - mean_hp).powi(2))
                .sum::<f64>()
                / hp_values.len() as f64;

            let loss_variance: f64 = final_losses
                .iter()
                .map(|&loss| (loss - mean_loss).powi(2))
                .sum::<f64>()
                / final_losses.len() as f64;

            if hp_variance > 0.0 && loss_variance > 0.0 {
                let correlation = covariance / (hp_variance.sqrt() * loss_variance.sqrt());
                return Some((correlation, mean_hp));
            }
        }

        None
    }

    /// Finds the best hyperparameter configuration
    pub fn best_configuration(&self) -> Option<(String, f64, HashMap<String, f64>)> {
        let mut best_config = None;
        let mut best_loss = f64::INFINITY;

        for (config_name, (hps, history)) in &self.hp_configurations {
            if let Some(last_metrics) = history.last() {
                let loss = last_metrics.val_loss.unwrap_or(last_metrics.train_loss);
                if loss < best_loss {
                    best_loss = loss;
                    best_config = Some((config_name.clone(), loss, hps.clone()));
                }
            }
        }

        best_config
    }
}

impl Default for HyperparameterSensitivityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer for HyperparameterSensitivityAnalyzer {
    fn name(&self) -> &str {
        "HyperparameterSensitivityAnalyzer"
    }

    fn update(&mut self, _epoch: usize, _metrics: &TrainingMetrics) {
        // Configurations are added via add_configuration or update_configuration
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn convergence_epoch(&self) -> Option<usize> {
        None
    }

    fn analysis_report(&self) -> AnalysisReport {
        let mut report = AnalysisReport::new(self.name().to_string());

        report.add_metric("num_configurations", self.hp_configurations.len() as f64);

        // Find best configuration
        if let Some((best_config, best_loss, best_hps)) = self.best_configuration() {
            report.add_detail("best_configuration", best_config.clone());
            report.add_metric("best_configuration_loss", best_loss);

            for (hp_name, hp_value) in &best_hps {
                report.add_metric(format!("best_{}", hp_name), *hp_value);
            }

            report.add_recommendation(format!(
                "Best configuration: {} with loss {:.4}",
                best_config, best_loss
            ));
        }

        // Analyze sensitivity for common hyperparameters
        let common_hps = vec!["learning_rate", "batch_size", "weight_decay", "dropout"];

        for hp_name in &common_hps {
            if let Some((correlation, mean_value)) = self.analyze_sensitivity(hp_name) {
                report.add_metric(format!("{}_correlation", hp_name), correlation);
                report.add_metric(format!("{}_mean_value", hp_name), mean_value);

                if correlation.abs() > 0.7 {
                    let direction = if correlation > 0.0 {
                        "increasing"
                    } else {
                        "decreasing"
                    };
                    report.add_recommendation(format!(
                        "{} is highly sensitive (correlation: {:.3}). {} it improves performance.",
                        hp_name, correlation, direction
                    ));
                }
            }
        }

        // Performance range analysis
        let mut all_final_losses = Vec::new();
        for (_hps, history) in self.hp_configurations.values() {
            if let Some(last_metrics) = history.last() {
                all_final_losses.push(last_metrics.train_loss);
            }
        }

        if !all_final_losses.is_empty() {
            let min_loss = all_final_losses
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let max_loss = all_final_losses
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let mean_loss = all_final_losses.iter().sum::<f64>() / all_final_losses.len() as f64;

            report.add_metric("performance_range_min", min_loss);
            report.add_metric("performance_range_max", max_loss);
            report.add_metric("performance_range_mean", mean_loss);
            report.add_metric("performance_range_span", max_loss - min_loss);

            if max_loss - min_loss > mean_loss {
                report.add_recommendation(
                    "Large performance variance across configurations. Hyperparameter tuning is critical."
                );
            } else {
                report.add_recommendation(
                    "Relatively stable performance across configurations. Model is robust to hyperparameter changes."
                );
            }
        }

        report
    }

    fn reset(&mut self) {
        self.hp_configurations.clear();
        self.converged = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_comparison() {
        let mut analyzer = MethodComparisonAnalyzer::new();

        let metrics1 = vec![
            TrainingMetrics::new(0, 1.0, 0.5),
            TrainingMetrics::new(1, 0.8, 0.6),
            TrainingMetrics::new(2, 0.6, 0.7),
        ];

        let metrics2 = vec![
            TrainingMetrics::new(0, 1.0, 0.5),
            TrainingMetrics::new(1, 0.7, 0.65),
            TrainingMetrics::new(2, 0.5, 0.75),
        ];

        analyzer.add_method_run("BPTT".to_string(), metrics1);
        analyzer.add_method_run("OTTT".to_string(), metrics2);

        let report = analyzer.analysis_report();
        assert_eq!(report.metrics.get("num_methods_compared"), Some(&2.0));
        assert!(report.details.contains_key("best_method"));
    }

    #[test]
    fn test_best_method_selection() {
        let mut analyzer = MethodComparisonAnalyzer::new();

        analyzer.update_method("method1".to_string(), TrainingMetrics::new(0, 0.5, 0.9));
        analyzer.update_method("method2".to_string(), TrainingMetrics::new(0, 0.3, 0.95));

        let (best_method, best_loss) = analyzer.best_method().unwrap();
        assert_eq!(best_method, "method2");
        assert_eq!(best_loss, 0.3);
    }

    #[test]
    fn test_convergence_speed_comparison() {
        let mut analyzer = MethodComparisonAnalyzer::new();

        let fast_method = vec![
            TrainingMetrics::new(0, 1.0, 0.5),
            TrainingMetrics::new(1, 0.5, 0.7),
            TrainingMetrics::new(2, 0.05, 0.95),
        ];

        let slow_method = vec![
            TrainingMetrics::new(0, 1.0, 0.5),
            TrainingMetrics::new(1, 0.8, 0.6),
            TrainingMetrics::new(2, 0.7, 0.65),
            TrainingMetrics::new(3, 0.5, 0.7),
            TrainingMetrics::new(4, 0.05, 0.95),
        ];

        analyzer.add_method_run("fast".to_string(), fast_method);
        analyzer.add_method_run("slow".to_string(), slow_method);

        let speeds = analyzer.compare_convergence_speed(0.1);
        assert_eq!(speeds.get("fast"), Some(&Some(2)));
        assert_eq!(speeds.get("slow"), Some(&Some(4)));
    }

    #[test]
    fn test_hyperparameter_sensitivity() {
        let mut analyzer = HyperparameterSensitivityAnalyzer::new();

        let mut hp1 = HashMap::new();
        hp1.insert("learning_rate".to_string(), 0.001);
        analyzer.add_configuration(
            "config1".to_string(),
            hp1,
            vec![TrainingMetrics::new(0, 0.5, 0.9)],
        );

        let mut hp2 = HashMap::new();
        hp2.insert("learning_rate".to_string(), 0.01);
        analyzer.add_configuration(
            "config2".to_string(),
            hp2,
            vec![TrainingMetrics::new(0, 0.3, 0.95)],
        );

        let report = analyzer.analysis_report();
        assert_eq!(report.metrics.get("num_configurations"), Some(&2.0));
    }

    #[test]
    fn test_best_configuration() {
        let mut analyzer = HyperparameterSensitivityAnalyzer::new();

        let mut hp1 = HashMap::new();
        hp1.insert("learning_rate".to_string(), 0.001);

        let mut hp2 = HashMap::new();
        hp2.insert("learning_rate".to_string(), 0.01);

        analyzer.add_configuration(
            "config1".to_string(),
            hp1.clone(),
            vec![TrainingMetrics::new(0, 0.5, 0.9)],
        );

        analyzer.add_configuration(
            "config2".to_string(),
            hp2.clone(),
            vec![TrainingMetrics::new(0, 0.3, 0.95)],
        );

        let (best_config, best_loss, _) = analyzer.best_configuration().unwrap();
        assert_eq!(best_config, "config2");
        assert_eq!(best_loss, 0.3);
    }

    #[test]
    fn test_sensitivity_analysis() {
        let mut analyzer = HyperparameterSensitivityAnalyzer::new();

        for i in 0..5 {
            let mut hp = HashMap::new();
            let lr = 0.001 * (i + 1) as f64;
            hp.insert("learning_rate".to_string(), lr);

            // Simulate that higher LR gives lower loss (negative correlation)
            let loss = 1.0 - lr * 100.0;

            analyzer.add_configuration(
                format!("config{}", i),
                hp,
                vec![TrainingMetrics::new(0, loss, 0.9)],
            );
        }

        let (correlation, _mean) = analyzer.analyze_sensitivity("learning_rate").unwrap();
        // Should have negative correlation
        assert!(correlation < 0.0);
    }

    #[test]
    fn test_update_methods() {
        let mut analyzer = MethodComparisonAnalyzer::new();

        analyzer.update_method("method1".to_string(), TrainingMetrics::new(0, 1.0, 0.5));
        analyzer.update_method("method1".to_string(), TrainingMetrics::new(1, 0.8, 0.6));

        let final_perf = analyzer.compare_final_performance();
        assert!(final_perf.contains_key("method1"));
        assert_eq!(final_perf.get("method1").unwrap().0, 0.8);
    }
}
