//! Isotonic regression for calibration

/// Isotonic regression for calibration
/// Fits a non-decreasing function to map scores to calibrated probabilities
#[derive(Debug, Clone)]
pub struct IsotonicCalibration {
    x_values: Vec<f64>,
    y_values: Vec<f64>,
    fitted: bool,
}

impl IsotonicCalibration {
    /// Create a new isotonic calibration instance
    pub fn new() -> Self {
        Self {
            x_values: Vec::new(),
            y_values: Vec::new(),
            fitted: false,
        }
    }

    /// Fit isotonic regression using pool adjacent violators algorithm (PAVA)
    pub fn fit(&mut self, scores: &[f64], labels: &[f64]) -> Result<(), String> {
        if scores.len() != labels.len() {
            return Err("Number of scores must match number of labels".to_string());
        }

        if scores.is_empty() {
            return Err("Cannot fit on empty dataset".to_string());
        }

        // Sort by scores
        let mut data: Vec<(f64, f64)> = scores.iter().zip(labels.iter()).map(|(&s, &l)| (s, l)).collect();
        data.sort_by(|a, b| a.0.total_cmp(&b.0));

        let sorted_scores: Vec<f64> = data.iter().map(|(s, _)| *s).collect();
        let sorted_labels: Vec<f64> = data.iter().map(|(_, l)| *l).collect();

        // Apply PAVA
        let weights = vec![1.0; sorted_labels.len()];
        let isotonic_values = pava(&sorted_labels, &weights);

        // Store unique x and y values (merge duplicates)
        let mut x_vals = Vec::new();
        let mut y_vals = Vec::new();

        let mut i = 0;
        while i < sorted_scores.len() {
            let x = sorted_scores[i];
            let mut y_sum = isotonic_values[i];
            let mut count = 1;

            // Merge duplicates
            while i + count < sorted_scores.len() && (sorted_scores[i + count] - x).abs() < 1e-10 {
                y_sum += isotonic_values[i + count];
                count += 1;
            }

            x_vals.push(x);
            y_vals.push(y_sum / count as f64);
            i += count;
        }

        self.x_values = x_vals;
        self.y_values = y_vals;
        self.fitted = true;

        Ok(())
    }

    /// Calibrate a single score using linear interpolation
    pub fn calibrate(&self, score: f64) -> f64 {
        if !self.fitted || self.x_values.is_empty() {
            return score;
        }

        // Handle boundary cases
        if score <= self.x_values[0] {
            return self.y_values[0];
        }
        if score >= self.x_values[self.x_values.len() - 1] {
            return self.y_values[self.y_values.len() - 1];
        }

        // Binary search for the interval
        let mut left = 0;
        let mut right = self.x_values.len() - 1;

        while right - left > 1 {
            let mid = (left + right) / 2;
            if self.x_values[mid] <= score {
                left = mid;
            } else {
                right = mid;
            }
        }

        // Linear interpolation
        let x0 = self.x_values[left];
        let x1 = self.x_values[right];
        let y0 = self.y_values[left];
        let y1 = self.y_values[right];

        let t = (score - x0) / (x1 - x0);
        y0 + t * (y1 - y0)
    }

    /// Calibrate batch of scores
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }

    /// Get the fitted curve points
    pub fn curve_points(&self) -> (Vec<f64>, Vec<f64>) {
        (self.x_values.clone(), self.y_values.clone())
    }
}

impl Default for IsotonicCalibration {
    fn default() -> Self {
        Self::new()
    }
}

/// Pool Adjacent Violators Algorithm implementation
/// Fits a non-decreasing function to data
fn pava(y: &[f64], weights: &[f64]) -> Vec<f64> {
    if y.is_empty() {
        return Vec::new();
    }

    let n = y.len();
    let mut result = y.to_vec();
    let mut w = weights.to_vec();

    let mut i = 0;
    while i < n - 1 {
        // If monotonicity is violated
        if result[i] > result[i + 1] {
            // Pool violating blocks
            let mut j = i;
            let mut pooled_sum = result[i] * w[i] + result[i + 1] * w[i + 1];
            let mut pooled_weight = w[i] + w[i + 1];

            // Keep pooling backwards while violation exists
            while j > 0 && result[j - 1] > pooled_sum / pooled_weight {
                j -= 1;
                pooled_sum += result[j] * w[j];
                pooled_weight += w[j];
            }

            // Keep pooling forwards while violation exists
            let mut k = i + 1;
            while k < n - 1 && result[k + 1] < pooled_sum / pooled_weight {
                k += 1;
                pooled_sum += result[k] * w[k];
                pooled_weight += w[k];
            }

            // Set pooled value
            let pooled_value = pooled_sum / pooled_weight;
            for idx in j..=k {
                result[idx] = pooled_value;
                w[idx] = pooled_weight / (k - j + 1) as f64;
            }

            // Continue checking from the pooled region
            i = j;
        } else {
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isotonic_calibration_new() {
        let ic = IsotonicCalibration::new();
        assert!(!ic.fitted);
        assert!(ic.x_values.is_empty());
        assert!(ic.y_values.is_empty());
    }

    #[test]
    fn test_isotonic_calibration_fit() {
        let mut ic = IsotonicCalibration::new();

        // Scores that need calibration
        let scores = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let labels = vec![0.0, 0.0, 0.1, 0.2, 0.5, 0.6, 0.8, 0.9, 1.0];

        let result = ic.fit(&scores, &labels);
        assert!(result.is_ok());
        assert!(ic.fitted);

        let (x, y) = ic.curve_points();
        assert!(!x.is_empty());
        assert_eq!(x.len(), y.len());

        // Check monotonicity
        for i in 1..y.len() {
            assert!(y[i] >= y[i - 1], "Isotonic regression should be non-decreasing");
        }
    }

    #[test]
    fn test_isotonic_calibration_calibrate() {
        let mut ic = IsotonicCalibration::new();

        let scores = vec![0.1, 0.5, 0.9];
        let labels = vec![0.1, 0.5, 0.9];

        ic.fit(&scores, &labels).unwrap();

        // Test interpolation
        let calibrated = ic.calibrate(0.3);
        assert!((0.1..=0.5).contains(&calibrated));

        let calibrated = ic.calibrate(0.7);
        assert!((0.5..=0.9).contains(&calibrated));
    }

    #[test]
    fn test_isotonic_calibration_batch() {
        let mut ic = IsotonicCalibration::new();

        let scores = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let labels = vec![0.0, 0.25, 0.5, 0.75, 1.0];

        ic.fit(&scores, &labels).unwrap();

        let test_scores = vec![0.1, 0.3, 0.6, 0.8];
        let calibrated = ic.calibrate_batch(&test_scores);

        assert_eq!(calibrated.len(), test_scores.len());

        // All calibrated values should be in [0, 1]
        for &val in &calibrated {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_isotonic_calibration_boundary() {
        let mut ic = IsotonicCalibration::new();

        let scores = vec![0.2, 0.5, 0.8];
        let labels = vec![0.2, 0.5, 0.8];

        ic.fit(&scores, &labels).unwrap();

        // Below minimum
        let calibrated = ic.calibrate(0.1);
        assert_eq!(calibrated, 0.2);

        // Above maximum
        let calibrated = ic.calibrate(0.9);
        assert_eq!(calibrated, 0.8);
    }

    #[test]
    fn test_pava_simple() {
        let y = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![1.0; 4];
        let result = pava(&y, &weights);

        // Already monotonic, should be unchanged
        assert_eq!(result, y);
    }

    #[test]
    fn test_pava_violation() {
        let y = vec![1.0, 3.0, 2.0, 4.0];
        let weights = vec![1.0; 4];
        let result = pava(&y, &weights);

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }

        // Middle values should be pooled to 2.5
        assert!((result[1] - 2.5).abs() < 1e-10);
        assert!((result[2] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_pava_multiple_violations() {
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let weights = vec![1.0; 5];
        let result = pava(&y, &weights);

        // All should be pooled to mean = 3.0
        for &val in &result {
            assert!((val - 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pava_weighted() {
        let y = vec![1.0, 4.0, 2.0];
        let weights = vec![1.0, 1.0, 1.0];
        let result = pava(&y, &weights);

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }

    #[test]
    fn test_pava_empty() {
        let y: Vec<f64> = vec![];
        let weights: Vec<f64> = vec![];
        let result = pava(&y, &weights);
        assert!(result.is_empty());
    }

    #[test]
    fn test_isotonic_calibration_invalid_input() {
        let mut ic = IsotonicCalibration::new();

        // Mismatched lengths
        let scores = vec![0.5];
        let labels = vec![0.5, 0.6];
        assert!(ic.fit(&scores, &labels).is_err());

        // Empty input
        let scores: Vec<f64> = vec![];
        let labels: Vec<f64> = vec![];
        assert!(ic.fit(&scores, &labels).is_err());
    }

    #[test]
    fn test_isotonic_calibration_not_fitted() {
        let ic = IsotonicCalibration::new();
        let score = 0.5;
        let calibrated = ic.calibrate(score);

        // Should return original score when not fitted
        assert_eq!(calibrated, score);
    }

    #[test]
    fn test_isotonic_real_calibration_scenario() {
        let mut ic = IsotonicCalibration::new();

        // Overconfident model: high scores but low actual frequencies
        let scores = vec![0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1];
        let labels = vec![0.7, 0.65, 0.6, 0.55, 0.5, 0.4, 0.3, 0.25, 0.15, 0.1];

        ic.fit(&scores, &labels).unwrap();

        // High confidence should be calibrated down
        let calibrated_high = ic.calibrate(0.9);
        assert!(calibrated_high < 0.9);
        assert!(calibrated_high >= 0.7);

        // Low confidence might stay similar
        let calibrated_low = ic.calibrate(0.1);
        assert!(calibrated_low <= 0.2);
    }
}
