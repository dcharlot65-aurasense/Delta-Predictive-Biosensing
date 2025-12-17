//! Pre-configured test scenarios
//!
//! This module provides pre-configured benchmarking scenarios for common
//! use cases like encoding, classification, regression, and latency testing.

use crate::datasets::BenchmarkDataset;
use crate::profiling::TimeProfiler;
use dpb_core::Signal;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Result from running a scenario
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    /// Scenario name
    pub name: String,
    /// Success status
    pub success: bool,
    /// Metrics
    pub metrics: HashMap<String, f64>,
    /// Timing information
    pub timing: TimingInfo,
    /// Error message (if failed)
    pub error: Option<String>,
}

/// Timing information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    /// Total time (ms)
    pub total_ms: f64,
    /// Encoding time (ms)
    pub encoding_ms: f64,
    /// Inference time (ms)
    pub inference_ms: f64,
    /// Decoding time (ms)
    pub decoding_ms: f64,
}

/// Encoding scenario - Test encoder accuracy vs ground truth
pub struct EncodingScenario {
    /// Scenario name
    pub name: String,
    /// Number of trials
    pub num_trials: usize,
    /// Expected encoding rate (events per second)
    pub target_rate: Option<f64>,
}

impl EncodingScenario {
    /// Create a new encoding scenario
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            num_trials: 100,
            target_rate: None,
        }
    }

    /// Set number of trials
    pub fn with_trials(mut self, num_trials: usize) -> Self {
        self.num_trials = num_trials;
        self
    }

    /// Set target encoding rate
    pub fn with_target_rate(mut self, rate: f64) -> Self {
        self.target_rate = Some(rate);
        self
    }

    /// Run the scenario
    pub fn run<D, F>(&self, dataset: &D, encoder_fn: F) -> anyhow::Result<ScenarioResult>
    where
        D: BenchmarkDataset,
        F: Fn(&[f32]) -> anyhow::Result<Vec<(f64, usize)>>,
    {
        let mut profiler = TimeProfiler::new(&self.name);
        let mut metrics = HashMap::new();

        let mut total_events = 0;
        let mut total_samples = 0;

        profiler.start();

        for _trial in 0..self.num_trials {
            let (signal, ground_truth) = dataset.generate()?;

            // Encode signal
            let events = encoder_fn(signal.samples())?;
            total_events += events.len();
            total_samples += signal.samples().len();

            // Compare with ground truth
            // Extract timestamps from temporal annotations
            let gt_times: Vec<f64> = ground_truth.temporal
                .as_ref()
                .map(|t| t.iter().map(|(time, _, _)| *time).collect())
                .unwrap_or_default();

            let precision = self.calculate_precision(&events, &gt_times);
            let recall = self.calculate_recall(&events, &gt_times);

            metrics.insert("precision".to_string(), precision);
            metrics.insert("recall".to_string(), recall);
            metrics.insert("f1_score".to_string(), 2.0 * precision * recall / (precision + recall));
        }

        profiler.stop();

        // Calculate encoding rate
        let encoding_rate = total_events as f64 / (total_samples as f64 / dataset.sample_rate());
        metrics.insert("encoding_rate".to_string(), encoding_rate);
        metrics.insert("avg_events_per_trial".to_string(), total_events as f64 / self.num_trials as f64);

        let timing = TimingInfo {
            total_ms: profiler.elapsed_ms(),
            encoding_ms: profiler.elapsed_ms(),
            inference_ms: 0.0,
            decoding_ms: 0.0,
        };

        Ok(ScenarioResult {
            name: self.name.clone(),
            success: true,
            metrics,
            timing,
            error: None,
        })
    }

    fn calculate_precision(&self, events: &[(f64, usize)], ground_truth: &[f64]) -> f64 {
        if events.is_empty() {
            return 0.0;
        }

        let tolerance = 0.05; // 50ms tolerance
        let mut true_positives = 0;

        for &(event_time, _) in events {
            if ground_truth.iter().any(|&gt_time| (event_time - gt_time).abs() < tolerance) {
                true_positives += 1;
            }
        }

        true_positives as f64 / events.len() as f64
    }

    fn calculate_recall(&self, events: &[(f64, usize)], ground_truth: &[f64]) -> f64 {
        if ground_truth.is_empty() {
            return 0.0;
        }

        let tolerance = 0.05; // 50ms tolerance
        let mut true_positives = 0;

        for &gt_time in ground_truth {
            if events.iter().any(|&(event_time, _)| (event_time - gt_time).abs() < tolerance) {
                true_positives += 1;
            }
        }

        true_positives as f64 / ground_truth.len() as f64
    }
}

/// Classification scenario - Test classification accuracy
pub struct ClassificationScenario {
    /// Scenario name
    pub name: String,
    /// Number of classes
    pub num_classes: usize,
    /// Number of samples per class
    pub samples_per_class: usize,
}

impl ClassificationScenario {
    /// Create a new classification scenario
    pub fn new(name: &str, num_classes: usize) -> Self {
        Self {
            name: name.to_string(),
            num_classes,
            samples_per_class: 50,
        }
    }

    /// Set samples per class
    pub fn with_samples(mut self, samples_per_class: usize) -> Self {
        self.samples_per_class = samples_per_class;
        self
    }

    /// Run the scenario
    pub fn run<F>(&self, classifier_fn: F) -> anyhow::Result<ScenarioResult>
    where
        F: Fn(&[f32]) -> anyhow::Result<usize>,
    {
        let mut profiler = TimeProfiler::new(&self.name);
        let mut metrics = HashMap::new();

        let mut correct = 0;
        let mut total = 0;

        // Confusion matrix
        let mut confusion = vec![vec![0; self.num_classes]; self.num_classes];

        profiler.start();

        for true_class in 0..self.num_classes {
            for _ in 0..self.samples_per_class {
                // Generate synthetic sample for this class
                let sample = self.generate_sample(true_class);

                // Classify
                let predicted_class = classifier_fn(&sample)?;

                if predicted_class < self.num_classes {
                    confusion[true_class][predicted_class] += 1;
                    total += 1;

                    if predicted_class == true_class {
                        correct += 1;
                    }
                }
            }
        }

        profiler.stop();

        // Calculate metrics
        let accuracy = correct as f64 / total as f64;
        metrics.insert("accuracy".to_string(), accuracy);

        // Per-class precision and recall
        let mut avg_precision = 0.0;
        let mut avg_recall = 0.0;

        for class in 0..self.num_classes {
            let tp = confusion[class][class] as f64;
            let fp: f64 = (0..self.num_classes)
                .filter(|&i| i != class)
                .map(|i| confusion[i][class] as f64)
                .sum();
            let fn_: f64 = (0..self.num_classes)
                .filter(|&i| i != class)
                .map(|i| confusion[class][i] as f64)
                .sum();

            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
            let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };

            avg_precision += precision;
            avg_recall += recall;
        }

        avg_precision /= self.num_classes as f64;
        avg_recall /= self.num_classes as f64;

        metrics.insert("precision".to_string(), avg_precision);
        metrics.insert("recall".to_string(), avg_recall);
        metrics.insert("f1_score".to_string(), 2.0 * avg_precision * avg_recall / (avg_precision + avg_recall));

        let timing = TimingInfo {
            total_ms: profiler.elapsed_ms(),
            encoding_ms: 0.0,
            inference_ms: profiler.elapsed_ms(),
            decoding_ms: 0.0,
        };

        Ok(ScenarioResult {
            name: self.name.clone(),
            success: true,
            metrics,
            timing,
            error: None,
        })
    }

    fn generate_sample(&self, class: usize) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Generate a simple synthetic sample
        let mut sample = vec![0.0; 100];
        for i in 0..100 {
            sample[i] = rng.r#gen::<f32>() * 0.1 + class as f32 * 0.3;
        }
        sample
    }
}

/// Regression scenario - Test regression performance
pub struct RegressionScenario {
    /// Scenario name
    pub name: String,
    /// Number of samples
    pub num_samples: usize,
    /// Target RMSE
    pub target_rmse: Option<f64>,
}

impl RegressionScenario {
    /// Create a new regression scenario
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            num_samples: 100,
            target_rmse: None,
        }
    }

    /// Set number of samples
    pub fn with_samples(mut self, num_samples: usize) -> Self {
        self.num_samples = num_samples;
        self
    }

    /// Set target RMSE
    pub fn with_target_rmse(mut self, rmse: f64) -> Self {
        self.target_rmse = Some(rmse);
        self
    }

    /// Run the scenario
    pub fn run<F>(&self, regressor_fn: F) -> anyhow::Result<ScenarioResult>
    where
        F: Fn(&[f32]) -> anyhow::Result<f64>,
    {
        let mut profiler = TimeProfiler::new(&self.name);
        let mut metrics = HashMap::new();

        let mut squared_errors = Vec::new();
        let mut absolute_errors = Vec::new();

        profiler.start();

        for _ in 0..self.num_samples {
            // Generate synthetic sample
            let (input, target) = self.generate_sample();

            // Predict
            let prediction = regressor_fn(&input)?;

            let error = prediction - target;
            squared_errors.push(error * error);
            absolute_errors.push(error.abs());
        }

        profiler.stop();

        // Calculate metrics
        let mse = squared_errors.iter().sum::<f64>() / self.num_samples as f64;
        let rmse = mse.sqrt();
        let mae = absolute_errors.iter().sum::<f64>() / self.num_samples as f64;

        metrics.insert("rmse".to_string(), rmse);
        metrics.insert("mae".to_string(), mae);
        metrics.insert("mse".to_string(), mse);

        let timing = TimingInfo {
            total_ms: profiler.elapsed_ms(),
            encoding_ms: 0.0,
            inference_ms: profiler.elapsed_ms(),
            decoding_ms: 0.0,
        };

        Ok(ScenarioResult {
            name: self.name.clone(),
            success: true,
            metrics,
            timing,
            error: None,
        })
    }

    fn generate_sample(&self) -> (Vec<f32>, f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let input: Vec<f32> = (0..50).map(|_| rng.r#gen::<f32>()).collect();
        let target = input.iter().sum::<f32>() as f64 / input.len() as f64; // Mean

        (input, target)
    }
}

/// Latency scenario - Measure end-to-end latency
pub struct LatencyScenario {
    /// Scenario name
    pub name: String,
    /// Number of iterations
    pub num_iterations: usize,
    /// Target latency (ms)
    pub target_latency_ms: Option<f64>,
}

impl LatencyScenario {
    /// Create a new latency scenario
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            num_iterations: 1000,
            target_latency_ms: None,
        }
    }

    /// Set number of iterations
    pub fn with_iterations(mut self, num_iterations: usize) -> Self {
        self.num_iterations = num_iterations;
        self
    }

    /// Set target latency
    pub fn with_target_latency(mut self, latency_ms: f64) -> Self {
        self.target_latency_ms = Some(latency_ms);
        self
    }

    /// Run the scenario
    pub fn run<F>(&self, pipeline_fn: F) -> anyhow::Result<ScenarioResult>
    where
        F: Fn() -> anyhow::Result<()>,
    {
        let mut profiler = TimeProfiler::new(&self.name);
        let mut metrics = HashMap::new();

        // Warmup
        for _ in 0..10 {
            pipeline_fn()?;
        }

        // Measure
        for _ in 0..self.num_iterations {
            profiler.start();
            pipeline_fn()?;
            profiler.stop();
        }

        let avg_latency = profiler.average_ms();
        let min_latency = profiler.min_ms();
        let max_latency = profiler.max_ms();
        let std_dev = profiler.std_dev_ms();

        metrics.insert("avg_latency_ms".to_string(), avg_latency);
        metrics.insert("min_latency_ms".to_string(), min_latency);
        metrics.insert("max_latency_ms".to_string(), max_latency);
        metrics.insert("std_dev_ms".to_string(), std_dev);
        metrics.insert("throughput_per_sec".to_string(), 1000.0 / avg_latency);

        // Check if target was met
        let success = if let Some(target) = self.target_latency_ms {
            avg_latency <= target
        } else {
            true
        };

        let timing = TimingInfo {
            total_ms: profiler.elapsed_ms(),
            encoding_ms: 0.0,
            inference_ms: avg_latency,
            decoding_ms: 0.0,
        };

        Ok(ScenarioResult {
            name: self.name.clone(),
            success,
            metrics,
            timing,
            error: if !success {
                Some(format!("Latency {:.2}ms exceeds target {:.2}ms", avg_latency, self.target_latency_ms.unwrap()))
            } else {
                None
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasets::SyntheticECG;

    #[test]
    fn test_encoding_scenario() {
        let dataset = SyntheticECG::new(1000, 250.0, Some(42));
        let scenario = EncodingScenario::new("test").with_trials(5);

        let result = scenario.run(&dataset, |_signal| {
            // Mock encoder that returns some events
            Ok(vec![(0.1, 0), (0.5, 1), (1.0, 2)])
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.metrics.contains_key("encoding_rate"));
    }

    #[test]
    fn test_classification_scenario() {
        let scenario = ClassificationScenario::new("test", 3).with_samples(10);

        let result = scenario.run(|sample| {
            // Mock classifier
            Ok(if sample[0] < 0.3 { 0 } else if sample[0] < 0.6 { 1 } else { 2 })
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.metrics.contains_key("accuracy"));
    }

    #[test]
    fn test_regression_scenario() {
        let scenario = RegressionScenario::new("test").with_samples(50);

        let result = scenario.run(|sample| {
            // Mock regressor - just return mean
            Ok(sample.iter().sum::<f32>() as f64 / sample.len() as f64)
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.success);
        assert!(result.metrics.contains_key("rmse"));
    }

    #[test]
    fn test_latency_scenario() {
        let scenario = LatencyScenario::new("test")
            .with_iterations(100)
            .with_target_latency(10.0);

        let result = scenario.run(|| {
            // Mock pipeline
            Ok(())
        });

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.metrics.contains_key("avg_latency_ms"));
    }
}
