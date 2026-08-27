//! Uncertainty estimation methods

use rand::RngExt;

/// Confidence interval representation
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
}

impl ConfidenceInterval {
    /// Create a new confidence interval
    pub fn new(lower: f64, upper: f64, confidence_level: f64) -> Self {
        Self {
            lower,
            upper,
            confidence_level,
        }
    }

    /// Get the width of the interval
    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    /// Get the midpoint of the interval
    pub fn midpoint(&self) -> f64 {
        (self.lower + self.upper) / 2.0
    }

    /// Check if a value is within the interval
    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Trait for uncertainty estimation methods
pub trait UncertaintyEstimator {
    /// Estimate uncertainty from predictions
    fn estimate_uncertainty(&self, predictions: &[f64]) -> f64;

    /// Compute confidence interval
    fn confidence_interval(&self, predictions: &[f64], level: f64) -> ConfidenceInterval;
}

/// Monte Carlo Dropout uncertainty estimation
#[derive(Debug, Clone)]
pub struct MCDropout {
    n_samples: usize,
    dropout_rate: f64,
}

impl MCDropout {
    /// Create a new MC Dropout estimator
    pub fn new(n_samples: usize, dropout_rate: f64) -> Self {
        assert!(n_samples > 0, "Number of samples must be positive");
        assert!(
            (0.0..1.0).contains(&dropout_rate),
            "Dropout rate must be in [0, 1)"
        );

        Self {
            n_samples,
            dropout_rate,
        }
    }

    /// Compute predictive mean and variance from MC samples
    pub fn compute_statistics(&self, samples: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        if samples.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let n_samples = samples.len();
        let n_outputs = samples[0].len();

        let mut means = vec![0.0; n_outputs];
        let mut variances = vec![0.0; n_outputs];

        // Compute means
        for sample in samples {
            for (i, &val) in sample.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= n_samples as f64;
        }

        // Compute variances
        for sample in samples {
            for (i, &val) in sample.iter().enumerate() {
                let diff = val - means[i];
                variances[i] += diff * diff;
            }
        }
        for var in &mut variances {
            *var /= n_samples as f64;
        }

        (means, variances)
    }

    /// Compute epistemic uncertainty (model uncertainty)
    /// This is the variance in predictions across different dropout masks
    pub fn epistemic_uncertainty(&self, samples: &[Vec<f64>]) -> Vec<f64> {
        let (_, variances) = self.compute_statistics(samples);
        variances
    }

    /// Compute aleatoric uncertainty (data uncertainty)
    /// For classification, this can be approximated by the entropy of the mean prediction
    pub fn aleatoric_uncertainty(&self, samples: &[Vec<f64>]) -> Vec<f64> {
        let (means, _) = self.compute_statistics(samples);

        // Compute entropy: -sum(p * log(p))
        means
            .iter()
            .map(|&p| {
                if p > 1e-10 && p < 1.0 - 1e-10 {
                    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// Get number of samples
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get dropout rate
    pub fn dropout_rate(&self) -> f64 {
        self.dropout_rate
    }
}

impl UncertaintyEstimator for MCDropout {
    fn estimate_uncertainty(&self, predictions: &[f64]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        // Compute variance as uncertainty measure
        let mean: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance: f64 = predictions
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / predictions.len() as f64;

        variance.sqrt()
    }

    fn confidence_interval(&self, predictions: &[f64], level: f64) -> ConfidenceInterval {
        if predictions.is_empty() {
            return ConfidenceInterval::new(0.0, 0.0, level);
        }

        let mut sorted = predictions.to_vec();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let alpha = 1.0 - level;
        let lower_idx = ((alpha / 2.0) * sorted.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * sorted.len() as f64) as usize;

        let lower_idx = lower_idx.min(sorted.len() - 1);
        let upper_idx = upper_idx.min(sorted.len() - 1);

        ConfidenceInterval::new(sorted[lower_idx], sorted[upper_idx], level)
    }
}

/// Ensemble-based uncertainty estimation
#[derive(Debug, Clone)]
pub struct EnsembleUncertainty {
    n_models: usize,
}

impl EnsembleUncertainty {
    /// Create a new ensemble uncertainty estimator
    pub fn new(n_models: usize) -> Self {
        assert!(n_models > 0, "Number of models must be positive");
        Self { n_models }
    }

    /// Compute ensemble mean and variance
    pub fn compute_statistics(&self, predictions: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        if predictions.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let n_models = predictions.len();
        let n_classes = predictions[0].len();

        let mut means = vec![0.0; n_classes];
        let mut variances = vec![0.0; n_classes];

        // Compute means
        for pred in predictions {
            for (i, &val) in pred.iter().enumerate() {
                means[i] += val;
            }
        }
        for mean in &mut means {
            *mean /= n_models as f64;
        }

        // Compute variances
        for pred in predictions {
            for (i, &val) in pred.iter().enumerate() {
                let diff = val - means[i];
                variances[i] += diff * diff;
            }
        }
        for var in &mut variances {
            *var /= n_models as f64;
        }

        (means, variances)
    }

    /// Compute prediction entropy
    /// H(y|x) = -sum_c p(y=c|x) log p(y=c|x)
    pub fn entropy(&self, predictions: &[Vec<f64>]) -> Vec<f64> {
        let (means, _) = self.compute_statistics(predictions);

        // Compute entropy for each output dimension
        vec![compute_entropy(&means)]
    }

    /// Compute mutual information (epistemic uncertainty)
    /// I(y; theta | x) = H(E[p(y|x, theta)]) - E[H(p(y|x, theta))]
    pub fn mutual_information(&self, predictions: &[Vec<f64>]) -> Vec<f64> {
        if predictions.is_empty() {
            return Vec::new();
        }

        let (means, _) = self.compute_statistics(predictions);

        // Entropy of mean predictions
        let entropy_mean = compute_entropy(&means);

        // Mean entropy of individual predictions
        let mean_entropy: f64 = predictions
            .iter()
            .map(|pred| compute_entropy(pred))
            .sum::<f64>()
            / predictions.len() as f64;

        // Mutual information
        vec![entropy_mean - mean_entropy]
    }

    /// Get number of models
    pub fn n_models(&self) -> usize {
        self.n_models
    }
}

impl UncertaintyEstimator for EnsembleUncertainty {
    fn estimate_uncertainty(&self, predictions: &[f64]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        // Compute standard deviation as uncertainty measure
        let mean: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let variance: f64 = predictions
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>()
            / predictions.len() as f64;

        variance.sqrt()
    }

    fn confidence_interval(&self, predictions: &[f64], level: f64) -> ConfidenceInterval {
        if predictions.is_empty() {
            return ConfidenceInterval::new(0.0, 0.0, level);
        }

        let mean: f64 = predictions.iter().sum::<f64>() / predictions.len() as f64;
        let std: f64 = self.estimate_uncertainty(predictions);

        // Use normal approximation
        let z = match level {
            l if (l - 0.90).abs() < 0.01 => 1.645,
            l if (l - 0.95).abs() < 0.01 => 1.96,
            l if (l - 0.99).abs() < 0.01 => 2.576,
            _ => 1.96, // Default to 95%
        };

        let margin = z * std / (predictions.len() as f64).sqrt();

        ConfidenceInterval::new(mean - margin, mean + margin, level)
    }
}

/// Bootstrap confidence intervals
pub fn bootstrap_ci(data: &[f64], n_bootstrap: usize, confidence: f64) -> ConfidenceInterval {
    if data.is_empty() {
        return ConfidenceInterval::new(0.0, 0.0, confidence);
    }

    let mut rng = rand::rng();
    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample with replacement
        let mut sample = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            let idx = rng.random_range(0..data.len());
            sample.push(data[idx]);
        }

        // Compute mean
        let mean: f64 = sample.iter().sum::<f64>() / sample.len() as f64;
        bootstrap_means.push(mean);
    }

    // Sort bootstrap means
    bootstrap_means.sort_by(|a, b| a.total_cmp(b));

    // Compute percentiles
    let alpha = 1.0 - confidence;
    let lower_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    let lower_idx = lower_idx.min(bootstrap_means.len() - 1);
    let upper_idx = upper_idx.min(bootstrap_means.len() - 1);

    ConfidenceInterval::new(
        bootstrap_means[lower_idx],
        bootstrap_means[upper_idx],
        confidence,
    )
}

/// Compute entropy of a probability distribution
fn compute_entropy(probs: &[f64]) -> f64 {
    probs
        .iter()
        .filter(|&&p| p > 1e-10)
        .map(|&p| -p * p.ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_interval() {
        let ci = ConfidenceInterval::new(1.0, 3.0, 0.95);
        assert_eq!(ci.lower, 1.0);
        assert_eq!(ci.upper, 3.0);
        assert_eq!(ci.confidence_level, 0.95);
        assert_eq!(ci.width(), 2.0);
        assert_eq!(ci.midpoint(), 2.0);
        assert!(ci.contains(2.0));
        assert!(!ci.contains(0.5));
        assert!(!ci.contains(3.5));
    }

    #[test]
    fn test_mc_dropout_new() {
        let mc = MCDropout::new(10, 0.5);
        assert_eq!(mc.n_samples(), 10);
        assert_eq!(mc.dropout_rate(), 0.5);
    }

    #[test]
    #[should_panic(expected = "Number of samples must be positive")]
    fn test_mc_dropout_invalid_samples() {
        MCDropout::new(0, 0.5);
    }

    #[test]
    #[should_panic(expected = "Dropout rate must be in [0, 1)")]
    fn test_mc_dropout_invalid_rate() {
        MCDropout::new(10, 1.5);
    }

    #[test]
    fn test_mc_dropout_compute_statistics() {
        let mc = MCDropout::new(5, 0.5);

        let samples = vec![
            vec![0.8, 0.2],
            vec![0.7, 0.3],
            vec![0.9, 0.1],
            vec![0.75, 0.25],
            vec![0.85, 0.15],
        ];

        let (means, variances) = mc.compute_statistics(&samples);

        assert_eq!(means.len(), 2);
        assert_eq!(variances.len(), 2);

        // Check mean calculation
        let expected_mean_0 = (0.8 + 0.7 + 0.9 + 0.75 + 0.85) / 5.0;
        assert!((means[0] - expected_mean_0).abs() < 1e-6);

        // Variances should be positive
        assert!(variances[0] >= 0.0);
        assert!(variances[1] >= 0.0);
    }

    #[test]
    fn test_mc_dropout_epistemic_uncertainty() {
        let mc = MCDropout::new(3, 0.5);

        let samples = vec![vec![0.8, 0.2], vec![0.7, 0.3], vec![0.9, 0.1]];

        let uncertainty = mc.epistemic_uncertainty(&samples);

        assert_eq!(uncertainty.len(), 2);
        assert!(uncertainty[0] >= 0.0);
        assert!(uncertainty[1] >= 0.0);
    }

    #[test]
    fn test_mc_dropout_estimate_uncertainty() {
        let mc = MCDropout::new(5, 0.5);

        let predictions = vec![0.8, 0.85, 0.75, 0.9, 0.8];
        let uncertainty = mc.estimate_uncertainty(&predictions);

        assert!(uncertainty >= 0.0);
        assert!(uncertainty < 1.0);
    }

    #[test]
    fn test_mc_dropout_confidence_interval() {
        let mc = MCDropout::new(5, 0.5);

        let predictions = vec![0.8, 0.85, 0.75, 0.9, 0.8];
        let ci = mc.confidence_interval(&predictions, 0.95);

        assert_eq!(ci.confidence_level, 0.95);
        assert!(ci.lower <= ci.upper);
        assert!(ci.lower >= 0.0);
        assert!(ci.upper <= 1.0);
    }

    #[test]
    fn test_ensemble_uncertainty_new() {
        let ensemble = EnsembleUncertainty::new(5);
        assert_eq!(ensemble.n_models(), 5);
    }

    #[test]
    #[should_panic(expected = "Number of models must be positive")]
    fn test_ensemble_uncertainty_invalid() {
        EnsembleUncertainty::new(0);
    }

    #[test]
    fn test_ensemble_compute_statistics() {
        let ensemble = EnsembleUncertainty::new(3);

        let predictions = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.8, 0.1, 0.1],
            vec![0.75, 0.15, 0.1],
        ];

        let (means, variances) = ensemble.compute_statistics(&predictions);

        assert_eq!(means.len(), 3);
        assert_eq!(variances.len(), 3);

        // Check mean
        let expected_mean_0 = (0.7 + 0.8 + 0.75) / 3.0;
        assert!((means[0] - expected_mean_0).abs() < 1e-6);
    }

    #[test]
    fn test_ensemble_entropy() {
        let ensemble = EnsembleUncertainty::new(2);

        let predictions = vec![vec![0.6, 0.3, 0.1], vec![0.7, 0.2, 0.1]];

        let entropy = ensemble.entropy(&predictions);

        assert_eq!(entropy.len(), 1);
        assert!(entropy[0] > 0.0); // Should have positive entropy
    }

    #[test]
    fn test_ensemble_mutual_information() {
        let ensemble = EnsembleUncertainty::new(3);

        let predictions = vec![
            vec![0.7, 0.2, 0.1],
            vec![0.8, 0.1, 0.1],
            vec![0.6, 0.3, 0.1],
        ];

        let mi = ensemble.mutual_information(&predictions);

        assert_eq!(mi.len(), 1);
        assert!(mi[0] >= 0.0); // MI should be non-negative
    }

    #[test]
    fn test_ensemble_estimate_uncertainty() {
        let ensemble = EnsembleUncertainty::new(5);

        let predictions = vec![0.7, 0.8, 0.75, 0.85, 0.8];
        let uncertainty = ensemble.estimate_uncertainty(&predictions);

        assert!(uncertainty >= 0.0);
    }

    #[test]
    fn test_bootstrap_ci() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let ci = bootstrap_ci(&data, 1000, 0.95);

        assert_eq!(ci.confidence_level, 0.95);
        assert!(ci.lower <= ci.upper);

        // Mean should be around 5.5
        let mean = 5.5;
        assert!(ci.lower < mean);
        assert!(ci.upper > mean);
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let data: Vec<f64> = vec![];
        let ci = bootstrap_ci(&data, 100, 0.95);

        assert_eq!(ci.lower, 0.0);
        assert_eq!(ci.upper, 0.0);
    }

    #[test]
    fn test_compute_entropy() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy_uniform = compute_entropy(&uniform);

        // Deterministic distribution has zero entropy
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let entropy_deterministic = compute_entropy(&deterministic);

        assert!(entropy_uniform > entropy_deterministic);
        assert!((entropy_deterministic - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mc_dropout_aleatoric_uncertainty() {
        let mc = MCDropout::new(5, 0.5);

        // High confidence predictions (low aleatoric uncertainty)
        let samples_confident = vec![
            vec![0.95, 0.05],
            vec![0.96, 0.04],
            vec![0.94, 0.06],
        ];

        let aleatoric_confident = mc.aleatoric_uncertainty(&samples_confident);

        // Low confidence predictions (high aleatoric uncertainty)
        let samples_uncertain = vec![
            vec![0.5, 0.5],
            vec![0.51, 0.49],
            vec![0.49, 0.51],
        ];

        let aleatoric_uncertain = mc.aleatoric_uncertainty(&samples_uncertain);

        // Uncertain predictions should have higher aleatoric uncertainty
        assert!(aleatoric_uncertain[0] > aleatoric_confident[0]);
    }

    #[test]
    fn test_ensemble_high_disagreement() {
        let ensemble = EnsembleUncertainty::new(3);

        // Models with high disagreement
        let predictions = vec![
            vec![0.9, 0.1],
            vec![0.1, 0.9],
            vec![0.5, 0.5],
        ];

        let (_, variances) = ensemble.compute_statistics(&predictions);

        // Should have high variance due to disagreement
        assert!(variances[0] > 0.1);
    }

    #[test]
    fn test_ensemble_low_disagreement() {
        let ensemble = EnsembleUncertainty::new(3);

        // Models with low disagreement
        let predictions = vec![
            vec![0.9, 0.1],
            vec![0.91, 0.09],
            vec![0.89, 0.11],
        ];

        let (_, variances) = ensemble.compute_statistics(&predictions);

        // Should have low variance due to agreement
        assert!(variances[0] < 0.01);
    }
}
