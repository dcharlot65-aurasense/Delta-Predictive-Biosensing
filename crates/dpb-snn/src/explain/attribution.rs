//! Feature attribution methods for explainability

use rand::Rng;

/// Feature attribution result
#[derive(Debug, Clone)]
pub struct FeatureAttribution {
    pub feature_names: Vec<String>,
    pub attribution_values: Vec<f64>,
    pub baseline_output: f64,
    pub actual_output: f64,
}

impl FeatureAttribution {
    /// Create a new feature attribution result
    pub fn new(
        feature_names: Vec<String>,
        attribution_values: Vec<f64>,
        baseline_output: f64,
        actual_output: f64,
    ) -> Self {
        assert_eq!(
            feature_names.len(),
            attribution_values.len(),
            "Feature names and attribution values must have same length"
        );

        Self {
            feature_names,
            attribution_values,
            baseline_output,
            actual_output,
        }
    }

    /// Get top contributing features
    pub fn top_features(&self, k: usize) -> Vec<(String, f64)> {
        let mut indexed: Vec<_> = self
            .feature_names
            .iter()
            .zip(self.attribution_values.iter())
            .map(|(name, &val)| (name.clone(), val))
            .collect();

        indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());

        indexed.into_iter().take(k).collect()
    }

    /// Sum of attributions (should equal actual - baseline for complete methods)
    pub fn attribution_sum(&self) -> f64 {
        self.attribution_values.iter().sum()
    }

    /// Normalize attributions to sum to 1
    pub fn normalize(&mut self) {
        let sum: f64 = self.attribution_values.iter().map(|x| x.abs()).sum();
        if sum > 0.0 {
            for val in &mut self.attribution_values {
                *val /= sum;
            }
        }
    }

    /// Check completeness property (sum equals output difference)
    pub fn check_completeness(&self, tolerance: f64) -> bool {
        let expected = self.actual_output - self.baseline_output;
        let actual = self.attribution_sum();
        (expected - actual).abs() < tolerance
    }

    /// Get positive and negative contributions separately
    pub fn split_contributions(&self) -> (Vec<(String, f64)>, Vec<(String, f64)>) {
        let mut positive = Vec::new();
        let mut negative = Vec::new();

        for (name, &val) in self.feature_names.iter().zip(&self.attribution_values) {
            if val >= 0.0 {
                positive.push((name.clone(), val));
            } else {
                negative.push((name.clone(), val));
            }
        }

        (positive, negative)
    }
}

/// Gradient-based attribution
pub struct GradientAttribution;

impl GradientAttribution {
    /// Compute simple gradient * input attribution
    pub fn compute(inputs: &[f64], gradients: &[f64]) -> Vec<f64> {
        assert_eq!(
            inputs.len(),
            gradients.len(),
            "Inputs and gradients must have same length"
        );

        inputs
            .iter()
            .zip(gradients.iter())
            .map(|(&input, &grad)| input * grad)
            .collect()
    }

    /// Compute gradient * (input - baseline)
    pub fn compute_with_baseline(
        inputs: &[f64],
        baseline: &[f64],
        gradients: &[f64],
    ) -> Vec<f64> {
        assert_eq!(inputs.len(), baseline.len());
        assert_eq!(inputs.len(), gradients.len());

        inputs
            .iter()
            .zip(baseline.iter())
            .zip(gradients.iter())
            .map(|((&input, &base), &grad)| (input - base) * grad)
            .collect()
    }

    /// Compute SmoothGrad by averaging over noisy inputs
    pub fn smooth_grad<F>(
        input: &[f64],
        gradient_fn: F,
        n_samples: usize,
        noise_level: f64,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut rng = rand::thread_rng();
        let n_features = input.len();
        let mut grad_sum = vec![0.0; n_features];

        for _ in 0..n_samples {
            // Add Gaussian noise to input (using uniform noise for simplicity)
            let noisy_input: Vec<f64> = input
                .iter()
                .map(|&x| {
                    let noise = rng.gen_range(-noise_level..noise_level);
                    x + noise
                })
                .collect();

            // Compute gradient at noisy input
            let grad = gradient_fn(&noisy_input);

            // Accumulate
            for (sum, g) in grad_sum.iter_mut().zip(&grad) {
                *sum += g;
            }
        }

        // Average
        grad_sum.iter().map(|&sum| sum / n_samples as f64).collect()
    }
}

/// Integrated Gradients attribution
pub struct IntegratedGradients {
    n_steps: usize,
}

impl IntegratedGradients {
    /// Create a new integrated gradients calculator
    pub fn new(n_steps: usize) -> Self {
        assert!(n_steps > 0, "Number of steps must be positive");
        Self { n_steps }
    }

    /// Compute integrated gradients
    pub fn compute<F>(&self, input: &[f64], baseline: &[f64], gradient_fn: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        assert_eq!(
            input.len(),
            baseline.len(),
            "Input and baseline must have same length"
        );

        let n_features = input.len();
        let mut integrated_grads = vec![0.0; n_features];

        // Integrate gradients along path from baseline to input
        for step in 0..self.n_steps {
            let alpha = (step as f64 + 0.5) / self.n_steps as f64;

            // Interpolate between baseline and input
            let interpolated: Vec<f64> = baseline
                .iter()
                .zip(input.iter())
                .map(|(&b, &i)| b + alpha * (i - b))
                .collect();

            // Compute gradient at interpolated point
            let grads = gradient_fn(&interpolated);

            // Accumulate
            for (ig, g) in integrated_grads.iter_mut().zip(&grads) {
                *ig += g;
            }
        }

        // Scale by (input - baseline) and average over steps
        integrated_grads
            .iter()
            .zip(input.iter().zip(baseline.iter()))
            .map(|(&ig, (&i, &b))| (i - b) * ig / self.n_steps as f64)
            .collect()
    }

    /// Verify completeness axiom
    pub fn verify_completeness<F>(
        &self,
        attribution: &[f64],
        input: &[f64],
        baseline: &[f64],
        output_fn: F,
    ) -> f64
    where
        F: Fn(&[f64]) -> f64,
    {
        let attribution_sum: f64 = attribution.iter().sum();
        let output_diff = output_fn(input) - output_fn(baseline);

        (attribution_sum - output_diff).abs()
    }

    /// Compute with multiple baselines and average
    pub fn compute_multi_baseline<F>(
        &self,
        input: &[f64],
        baselines: &[Vec<f64>],
        gradient_fn: F,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let n_features = input.len();
        let mut avg_attribution = vec![0.0; n_features];

        for baseline in baselines {
            let attribution = self.compute(input, baseline, &gradient_fn);
            for (avg, attr) in avg_attribution.iter_mut().zip(&attribution) {
                *avg += attr;
            }
        }

        // Average over baselines
        avg_attribution
            .iter()
            .map(|&sum| sum / baselines.len() as f64)
            .collect()
    }
}

/// SHAP-like importance for spike features
pub struct SpikeSHAP {
    n_samples: usize,
}

impl SpikeSHAP {
    /// Create a new SpikeSHAP calculator
    pub fn new(n_samples: usize) -> Self {
        assert!(n_samples > 0, "Number of samples must be positive");
        Self { n_samples }
    }

    /// Compute approximate Shapley values for spike features
    ///
    /// This uses sampling to approximate the exact Shapley values which
    /// would be computationally expensive for many features.
    pub fn compute<F>(&self, spike_features: &[f64], output_fn: F) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = spike_features.len();
        let mut shapley_values = vec![0.0; n_features];
        let mut rng = rand::thread_rng();

        // Sample random permutations and compute marginal contributions
        for _ in 0..self.n_samples {
            // Generate random permutation
            let mut perm: Vec<usize> = (0..n_features).collect();
            for i in 0..n_features {
                let j = rng.gen_range(i..n_features);
                perm.swap(i, j);
            }

            // Compute marginal contribution for each feature in this permutation
            let mut current_features = vec![0.0; n_features];
            let mut prev_output = output_fn(&current_features);

            for &feature_idx in &perm {
                // Add this feature
                current_features[feature_idx] = spike_features[feature_idx];
                let new_output = output_fn(&current_features);

                // Marginal contribution
                shapley_values[feature_idx] += new_output - prev_output;
                prev_output = new_output;
            }
        }

        // Average over samples
        shapley_values
            .iter()
            .map(|&sum| sum / self.n_samples as f64)
            .collect()
    }

    /// Compute with baseline (zero features)
    pub fn compute_with_baseline<F>(
        &self,
        spike_features: &[f64],
        baseline: &[f64],
        output_fn: F,
    ) -> Vec<f64>
    where
        F: Fn(&[f64]) -> f64,
    {
        let n_features = spike_features.len();
        let mut shapley_values = vec![0.0; n_features];
        let mut rng = rand::thread_rng();

        for _ in 0..self.n_samples {
            let mut perm: Vec<usize> = (0..n_features).collect();
            for i in 0..n_features {
                let j = rng.gen_range(i..n_features);
                perm.swap(i, j);
            }

            let mut current_features = baseline.to_vec();
            let mut prev_output = output_fn(&current_features);

            for &feature_idx in &perm {
                current_features[feature_idx] = spike_features[feature_idx];
                let new_output = output_fn(&current_features);

                shapley_values[feature_idx] += new_output - prev_output;
                prev_output = new_output;
            }
        }

        shapley_values
            .iter()
            .map(|&sum| sum / self.n_samples as f64)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_attribution_creation() {
        let names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let values = vec![0.5, 0.3, -0.2];
        let attribution = FeatureAttribution::new(names, values, 0.0, 0.6);

        assert_eq!(attribution.feature_names.len(), 3);
        assert_eq!(attribution.attribution_values.len(), 3);
        assert_eq!(attribution.baseline_output, 0.0);
        assert_eq!(attribution.actual_output, 0.6);
    }

    #[test]
    fn test_top_features() {
        let names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let values = vec![0.5, 0.8, -0.3];
        let attribution = FeatureAttribution::new(names, values, 0.0, 1.0);

        let top = attribution.top_features(2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "f2"); // Highest absolute value
        assert_eq!(top[1].0, "f1"); // Second highest
    }

    #[test]
    fn test_attribution_sum() {
        let names = vec!["f1".to_string(), "f2".to_string()];
        let values = vec![0.5, 0.3];
        let attribution = FeatureAttribution::new(names, values, 0.0, 0.8);

        assert_eq!(attribution.attribution_sum(), 0.8);
    }

    #[test]
    fn test_split_contributions() {
        let names = vec!["f1".to_string(), "f2".to_string(), "f3".to_string()];
        let values = vec![0.5, -0.3, 0.2];
        let attribution = FeatureAttribution::new(names, values, 0.0, 0.4);

        let (positive, negative) = attribution.split_contributions();

        assert_eq!(positive.len(), 2);
        assert_eq!(negative.len(), 1);
        assert_eq!(negative[0].0, "f2");
    }

    #[test]
    fn test_gradient_attribution() {
        let inputs = vec![1.0, 2.0, 3.0];
        let gradients = vec![0.5, 0.3, 0.2];

        let attribution = GradientAttribution::compute(&inputs, &gradients);

        assert_eq!(attribution.len(), 3);
        assert!((attribution[0] - 0.5).abs() < 1e-10);
        assert!((attribution[1] - 0.6).abs() < 1e-10);
        assert!((attribution[2] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_gradient_attribution_with_baseline() {
        let inputs = vec![1.0, 2.0, 3.0];
        let baseline = vec![0.0, 0.0, 0.0];
        let gradients = vec![0.5, 0.3, 0.2];

        let attribution = GradientAttribution::compute_with_baseline(&inputs, &baseline, &gradients);

        assert_eq!(attribution.len(), 3);
        assert!((attribution[0] - 0.5).abs() < 1e-10);
        assert!((attribution[1] - 0.6).abs() < 1e-10);
        assert!((attribution[2] - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_integrated_gradients() {
        let ig = IntegratedGradients::new(50);

        let input = vec![1.0, 2.0, 3.0];
        let baseline = vec![0.0, 0.0, 0.0];

        // Simple linear gradient function
        let gradient_fn = |_x: &[f64]| vec![0.5, 0.3, 0.2];

        let attribution = ig.compute(&input, &baseline, gradient_fn);

        assert_eq!(attribution.len(), 3);

        // For constant gradients, IG should equal (input - baseline) * gradient
        assert!((attribution[0] - 0.5).abs() < 1e-6);
        assert!((attribution[1] - 0.6).abs() < 1e-6);
        assert!((attribution[2] - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_integrated_gradients_completeness() {
        let ig = IntegratedGradients::new(100);

        let input = vec![1.0, 2.0];
        let baseline = vec![0.0, 0.0];

        // Linear output function: y = 0.5*x1 + 0.3*x2
        let output_fn = |x: &[f64]| 0.5 * x[0] + 0.3 * x[1];
        let gradient_fn = |_x: &[f64]| vec![0.5, 0.3];

        let attribution = ig.compute(&input, &baseline, gradient_fn);

        let error = ig.verify_completeness(&attribution, &input, &baseline, output_fn);

        // Should be very small error for linear function
        assert!(error < 1e-6);
    }

    #[test]
    fn test_spike_shap() {
        let shap = SpikeSHAP::new(1000);

        let features = vec![1.0, 2.0, 3.0];

        // Simple additive output function
        let output_fn = |x: &[f64]| x.iter().sum::<f64>();

        let shapley = shap.compute(&features, output_fn);

        assert_eq!(shapley.len(), 3);

        // For additive function, each feature should have value equal to its input
        assert!((shapley[0] - 1.0).abs() < 0.1);
        assert!((shapley[1] - 2.0).abs() < 0.1);
        assert!((shapley[2] - 3.0).abs() < 0.1);
    }

    #[test]
    fn test_spike_shap_with_baseline() {
        let shap = SpikeSHAP::new(1000);

        let features = vec![1.0, 2.0, 3.0];
        let baseline = vec![0.5, 0.5, 0.5];

        // Simple additive output function
        let output_fn = |x: &[f64]| x.iter().sum::<f64>();

        let shapley = shap.compute_with_baseline(&features, &baseline, output_fn);

        assert_eq!(shapley.len(), 3);

        // Shapley values should reflect change from baseline
        assert!((shapley[0] - 0.5).abs() < 0.1);
        assert!((shapley[1] - 1.5).abs() < 0.1);
        assert!((shapley[2] - 2.5).abs() < 0.1);
    }

    #[test]
    fn test_feature_attribution_normalize() {
        let names = vec!["f1".to_string(), "f2".to_string()];
        let values = vec![3.0, 7.0];
        let mut attribution = FeatureAttribution::new(names, values, 0.0, 10.0);

        attribution.normalize();

        let sum: f64 = attribution.attribution_values.iter().map(|x| x.abs()).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_check_completeness() {
        let names = vec!["f1".to_string(), "f2".to_string()];
        let values = vec![0.5, 0.5];
        let attribution = FeatureAttribution::new(names, values, 0.0, 1.0);

        assert!(attribution.check_completeness(0.01));
    }
}
