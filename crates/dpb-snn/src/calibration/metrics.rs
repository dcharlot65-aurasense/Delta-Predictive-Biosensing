//! Calibration metrics

/// Expected Calibration Error
/// Measures the difference between predicted confidence and actual accuracy
pub fn expected_calibration_error(
    probabilities: &[f64],
    labels: &[bool],
    n_bins: usize,
) -> f64 {
    if probabilities.len() != labels.len() || probabilities.is_empty() {
        return 0.0;
    }

    let bins = create_bins(probabilities, labels, n_bins);
    let total_samples = probabilities.len() as f64;

    let mut ece = 0.0;
    for bin in bins {
        if bin.count > 0 {
            let weight = bin.count as f64 / total_samples;
            let diff = (bin.predicted_prob - bin.actual_freq).abs();
            ece += weight * diff;
        }
    }

    ece
}

/// Maximum Calibration Error
/// The maximum difference between confidence and accuracy across all bins
pub fn maximum_calibration_error(
    probabilities: &[f64],
    labels: &[bool],
    n_bins: usize,
) -> f64 {
    if probabilities.len() != labels.len() || probabilities.is_empty() {
        return 0.0;
    }

    let bins = create_bins(probabilities, labels, n_bins);

    let mut mce: f64 = 0.0;
    for bin in bins {
        if bin.count > 0 {
            let diff = (bin.predicted_prob - bin.actual_freq).abs();
            mce = mce.max(diff);
        }
    }

    mce
}

/// Brier Score (mean squared error of probability estimates)
/// Lower is better, range [0, 1]
pub fn brier_score(probabilities: &[f64], labels: &[bool]) -> f64 {
    if probabilities.len() != labels.len() || probabilities.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    for (&prob, &label) in probabilities.iter().zip(labels.iter()) {
        let target = if label { 1.0 } else { 0.0 };
        let diff = prob - target;
        sum += diff * diff;
    }

    sum / probabilities.len() as f64
}

/// Negative Log-Likelihood
/// Measures how well the predicted probabilities match the true labels
pub fn negative_log_likelihood(probabilities: &[f64], labels: &[bool]) -> f64 {
    if probabilities.len() != labels.len() || probabilities.is_empty() {
        return 0.0;
    }

    let mut nll = 0.0;
    for (&prob, &label) in probabilities.iter().zip(labels.iter()) {
        let p = if label {
            prob.clamp(1e-10, 1.0 - 1e-10)
        } else {
            (1.0 - prob).clamp(1e-10, 1.0 - 1e-10)
        };
        nll -= p.ln();
    }

    nll / probabilities.len() as f64
}

/// Data for reliability diagram
#[derive(Debug, Clone)]
pub struct ReliabilityBin {
    /// Mean predicted probability in bin
    pub predicted_prob: f64,
    /// Actual positive frequency in bin
    pub actual_freq: f64,
    /// Number of samples in bin
    pub count: usize,
}

impl ReliabilityBin {
    /// Create a new reliability bin
    pub fn new(predicted_prob: f64, actual_freq: f64, count: usize) -> Self {
        Self {
            predicted_prob,
            actual_freq,
            count,
        }
    }

    /// Calculate calibration error for this bin
    pub fn calibration_error(&self) -> f64 {
        (self.predicted_prob - self.actual_freq).abs()
    }
}

/// Compute reliability diagram bins
/// Returns bins with predicted probability vs actual frequency
pub fn reliability_diagram(
    probabilities: &[f64],
    labels: &[bool],
    n_bins: usize,
) -> Vec<ReliabilityBin> {
    if probabilities.len() != labels.len() || probabilities.is_empty() {
        return Vec::new();
    }

    create_bins(probabilities, labels, n_bins)
}

/// Helper function to create bins for calibration metrics
fn create_bins(probabilities: &[f64], labels: &[bool], n_bins: usize) -> Vec<ReliabilityBin> {
    let mut bins = vec![
        ReliabilityBin {
            predicted_prob: 0.0,
            actual_freq: 0.0,
            count: 0,
        };
        n_bins
    ];

    let bin_width = 1.0 / n_bins as f64;

    // Accumulate statistics for each bin
    let mut bin_sums = vec![0.0; n_bins];
    let mut bin_counts = vec![0; n_bins];
    let mut bin_positives = vec![0; n_bins];

    for (&prob, &label) in probabilities.iter().zip(labels.iter()) {
        // Determine which bin this sample belongs to
        let bin_idx = ((prob / bin_width).floor() as usize).min(n_bins - 1);

        bin_sums[bin_idx] += prob;
        bin_counts[bin_idx] += 1;
        if label {
            bin_positives[bin_idx] += 1;
        }
    }

    // Compute bin statistics
    for i in 0..n_bins {
        if bin_counts[i] > 0 {
            bins[i].predicted_prob = bin_sums[i] / bin_counts[i] as f64;
            bins[i].actual_freq = bin_positives[i] as f64 / bin_counts[i] as f64;
            bins[i].count = bin_counts[i];
        }
    }

    bins
}

/// Compute adaptive calibration error with variable-width bins
pub fn adaptive_calibration_error(
    probabilities: &[f64],
    labels: &[bool],
    n_bins: usize,
) -> f64 {
    if probabilities.len() != labels.len() || probabilities.is_empty() {
        return 0.0;
    }

    // Sort by probability
    let mut data: Vec<(f64, bool)> = probabilities
        .iter()
        .zip(labels.iter())
        .map(|(&p, &l)| (p, l))
        .collect();
    data.sort_by(|a, b| a.0.total_cmp(&b.0));

    let samples_per_bin = probabilities.len() / n_bins;
    let mut ace = 0.0;

    for bin_idx in 0..n_bins {
        let start = bin_idx * samples_per_bin;
        let end = if bin_idx == n_bins - 1 {
            data.len()
        } else {
            (bin_idx + 1) * samples_per_bin
        };

        if start >= end {
            continue;
        }

        let bin_data = &data[start..end];
        let count = bin_data.len();

        let mean_prob: f64 = bin_data.iter().map(|(p, _)| p).sum::<f64>() / count as f64;
        let actual_freq = bin_data.iter().filter(|(_, l)| *l).count() as f64 / count as f64;

        let weight = count as f64 / probabilities.len() as f64;
        ace += weight * (mean_prob - actual_freq).abs();
    }

    ace
}

/// Compute classwise calibration error
/// Useful for multi-class calibration
pub fn classwise_calibration_error(
    probabilities: &[Vec<f64>],
    labels: &[usize],
    n_bins: usize,
) -> Vec<f64> {
    if probabilities.is_empty() {
        return Vec::new();
    }

    let n_classes = probabilities[0].len();
    let mut class_errors = Vec::with_capacity(n_classes);

    for class_idx in 0..n_classes {
        let class_probs: Vec<f64> = probabilities.iter().map(|p| p[class_idx]).collect();
        let class_labels: Vec<bool> = labels.iter().map(|&l| l == class_idx).collect();

        let ece = expected_calibration_error(&class_probs, &class_labels, n_bins);
        class_errors.push(ece);
    }

    class_errors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expected_calibration_error_perfect() {
        // Perfect calibration: confidence matches accuracy
        let probabilities = vec![0.9, 0.8, 0.7, 0.6, 0.1, 0.2, 0.3, 0.4];
        let labels = vec![true, true, true, true, false, false, false, false];

        // For perfect calibration, ECE should be close to 0
        // (may not be exactly 0 due to binning)
        let ece = expected_calibration_error(&probabilities, &labels, 10);
        println!("Perfect calibration ECE: {}", ece);
        assert!(ece < 0.3); // Should be relatively small
    }

    #[test]
    fn test_expected_calibration_error_overconfident() {
        // Overconfident predictions
        let probabilities = vec![0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1];
        let labels = vec![true, false, true, false, false, true, false, true];

        let ece = expected_calibration_error(&probabilities, &labels, 10);
        println!("Overconfident ECE: {}", ece);
        assert!(ece > 0.0); // Should have calibration error
    }

    #[test]
    fn test_maximum_calibration_error() {
        let probabilities = vec![0.9, 0.8, 0.7, 0.2, 0.1];
        let labels = vec![true, true, false, false, false];

        let mce = maximum_calibration_error(&probabilities, &labels, 5);
        println!("MCE: {}", mce);
        assert!((0.0..=1.0).contains(&mce));
    }

    #[test]
    fn test_brier_score_perfect() {
        // Perfect predictions
        let probabilities = vec![1.0, 1.0, 0.0, 0.0];
        let labels = vec![true, true, false, false];

        let bs = brier_score(&probabilities, &labels);
        assert!((bs - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_brier_score_worst() {
        // Worst predictions
        let probabilities = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![true, true, false, false];

        let bs = brier_score(&probabilities, &labels);
        assert!((bs - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_brier_score_uncertain() {
        // Uncertain predictions
        let probabilities = vec![0.5, 0.5, 0.5, 0.5];
        let labels = vec![true, true, false, false];

        let bs = brier_score(&probabilities, &labels);
        assert!((bs - 0.25).abs() < 1e-6);
    }

    #[test]
    fn test_negative_log_likelihood() {
        let probabilities = vec![0.9, 0.8, 0.2, 0.1];
        let labels = vec![true, true, false, false];

        let nll = negative_log_likelihood(&probabilities, &labels);
        assert!(nll > 0.0);
        assert!(nll.is_finite());
    }

    #[test]
    fn test_negative_log_likelihood_perfect() {
        // Near-perfect predictions (use 0.999 instead of 1.0 to avoid log(0))
        let probabilities = vec![0.999, 0.999, 0.001, 0.001];
        let labels = vec![true, true, false, false];

        let nll = negative_log_likelihood(&probabilities, &labels);
        assert!(nll < 0.01); // Should be very small
    }

    #[test]
    fn test_reliability_diagram() {
        let probabilities = vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.1];
        let labels = vec![true, true, false, false, false, false];

        let bins = reliability_diagram(&probabilities, &labels, 5);

        assert_eq!(bins.len(), 5);

        // Check that bins are properly structured
        for bin in &bins {
            assert!(bin.predicted_prob >= 0.0 && bin.predicted_prob <= 1.0);
            assert!(bin.actual_freq >= 0.0 && bin.actual_freq <= 1.0);
            assert!(bin.count <= probabilities.len());
        }
    }

    #[test]
    fn test_reliability_bin() {
        let bin = ReliabilityBin::new(0.8, 0.6, 10);

        assert_eq!(bin.predicted_prob, 0.8);
        assert_eq!(bin.actual_freq, 0.6);
        assert_eq!(bin.count, 10);
        assert!((bin.calibration_error() - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_calibration_error() {
        let probabilities = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![false, false, false, true, true, true, true, true, true];

        let ace = adaptive_calibration_error(&probabilities, &labels, 3);
        println!("Adaptive CE: {}", ace);
        assert!((0.0..=1.0).contains(&ace));
    }

    #[test]
    fn test_classwise_calibration_error() {
        let probabilities = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.7, 0.2, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.7, 0.2],
            vec![0.1, 0.1, 0.8],
            vec![0.2, 0.1, 0.7],
        ];
        let labels = vec![0, 0, 1, 1, 2, 2];

        let class_errors = classwise_calibration_error(&probabilities, &labels, 5);

        assert_eq!(class_errors.len(), 3);
        for &error in &class_errors {
            assert!((0.0..=1.0).contains(&error));
        }
    }

    #[test]
    fn test_empty_input() {
        let probabilities: Vec<f64> = vec![];
        let labels: Vec<bool> = vec![];

        let ece = expected_calibration_error(&probabilities, &labels, 10);
        assert_eq!(ece, 0.0);

        let mce = maximum_calibration_error(&probabilities, &labels, 10);
        assert_eq!(mce, 0.0);

        let bs = brier_score(&probabilities, &labels);
        assert_eq!(bs, 0.0);

        let nll = negative_log_likelihood(&probabilities, &labels);
        assert_eq!(nll, 0.0);

        let bins = reliability_diagram(&probabilities, &labels, 10);
        assert!(bins.is_empty());
    }

    #[test]
    fn test_mismatched_lengths() {
        let probabilities = vec![0.5, 0.6];
        let labels = vec![true];

        let ece = expected_calibration_error(&probabilities, &labels, 10);
        assert_eq!(ece, 0.0);
    }

    #[test]
    fn test_single_bin() {
        let probabilities = vec![0.7, 0.8, 0.6, 0.9];
        let labels = vec![true, true, false, true];

        let ece = expected_calibration_error(&probabilities, &labels, 1);
        assert!(ece >= 0.0);

        let bins = reliability_diagram(&probabilities, &labels, 1);
        assert_eq!(bins.len(), 1);
        assert_eq!(bins[0].count, 4);
    }

    #[test]
    fn test_many_bins() {
        let probabilities = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![false, false, false, true, true, true, true, true, true];

        let ece = expected_calibration_error(&probabilities, &labels, 20);
        println!("ECE with many bins: {}", ece);
        assert!((0.0..=1.0).contains(&ece));
    }

    #[test]
    fn test_calibration_metrics_consistency() {
        // Well-calibrated model
        let probabilities = vec![0.2, 0.2, 0.5, 0.5, 0.8, 0.8];
        let labels = vec![false, true, false, true, false, true];

        let ece = expected_calibration_error(&probabilities, &labels, 3);
        let mce = maximum_calibration_error(&probabilities, &labels, 3);

        // MCE should be >= ECE
        assert!(mce >= ece);
    }

    #[test]
    fn test_reliability_diagram_structure() {
        let n = 100;
        let mut probabilities = Vec::with_capacity(n);
        let mut labels = Vec::with_capacity(n);

        // Create synthetic data
        for i in 0..n {
            let p = (i as f64) / (n as f64);
            probabilities.push(p);
            labels.push(i % 2 == 0);
        }

        let bins = reliability_diagram(&probabilities, &labels, 10);

        // Should have samples distributed across bins
        let non_empty_bins = bins.iter().filter(|b| b.count > 0).count();
        assert!(non_empty_bins > 0);
    }

    #[test]
    fn test_extreme_confidence() {
        // Test with extreme confidence values
        let probabilities = vec![0.0, 0.0, 1.0, 1.0];
        let labels = vec![false, false, true, true];

        let ece = expected_calibration_error(&probabilities, &labels, 10);
        assert!(ece < 0.1); // Should be well-calibrated

        let bs = brier_score(&probabilities, &labels);
        assert!(bs < 0.01); // Should have low Brier score
    }
}
