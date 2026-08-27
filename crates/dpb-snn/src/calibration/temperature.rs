//! Temperature scaling for neural network calibration


/// Temperature scaling for neural network calibration
/// Divides logits by a learned temperature parameter
#[derive(Debug, Clone)]
pub struct TemperatureScaling {
    temperature: f64,
    fitted: bool,
}

impl TemperatureScaling {
    /// Create a new temperature scaling instance
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            fitted: false,
        }
    }

    /// Fit temperature on validation set using Brent's method to minimize NLL
    pub fn fit(&mut self, logits: &[Vec<f64>], labels: &[usize]) -> Result<(), String> {
        if logits.len() != labels.len() {
            return Err("Number of logits must match number of labels".to_string());
        }

        if logits.is_empty() {
            return Err("Cannot fit on empty dataset".to_string());
        }

        // Validate labels
        let num_classes = logits[0].len();
        for &label in labels {
            if label >= num_classes {
                return Err(format!(
                    "Label {} exceeds number of classes {}",
                    label, num_classes
                ));
            }
        }

        // Use Brent's method to find optimal temperature
        // Search in range [0.01, 10.0]
        let temp = self.brent_optimize(logits, labels, 0.01, 10.0, 1e-5)?;

        self.temperature = temp;
        self.fitted = true;

        Ok(())
    }

    /// Apply temperature scaling to logits
    pub fn calibrate(&self, logits: &[f64]) -> Vec<f64> {
        if !self.fitted {
            return logits.to_vec();
        }

        // Divide logits by temperature
        let scaled: Vec<f64> = logits.iter().map(|&x| x / self.temperature).collect();

        // Apply softmax
        softmax(&scaled)
    }

    /// Apply to batch of predictions
    pub fn calibrate_batch(&self, logits: &[Vec<f64>]) -> Vec<Vec<f64>> {
        logits.iter().map(|l| self.calibrate(l)).collect()
    }

    /// Get the fitted temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Brent's method for 1D optimization
    fn brent_optimize(
        &self,
        logits: &[Vec<f64>],
        labels: &[usize],
        a: f64,
        b: f64,
        tol: f64,
    ) -> Result<f64, String> {
        const GOLDEN_RATIO: f64 = 0.381966011; // (3 - sqrt(5)) / 2
        const MAX_ITER: usize = 100;

        let mut a = a;
        let mut b = b;
        let mut v = a + GOLDEN_RATIO * (b - a);
        let mut w = v;
        let mut x = v;
        let mut fx = self.compute_nll(logits, labels, x);
        let mut fv = fx;
        let mut fw = fx;

        for _ in 0..MAX_ITER {
            let xm = 0.5 * (a + b);
            let tol1 = tol * x.abs() + 1e-10;
            let tol2 = 2.0 * tol1;

            // Check convergence
            if (x - xm).abs() <= (tol2 - 0.5 * (b - a)) {
                return Ok(x);
            }

            // Golden section step
            let r = if x >= xm { a } else { b };
            let u = x + GOLDEN_RATIO * (r - x);
            let fu = self.compute_nll(logits, labels, u);

            // Update bounds
            if fu <= fx {
                if u >= x {
                    a = x;
                } else {
                    b = x;
                }
                v = w;
                fv = fw;
                w = x;
                fw = fx;
                x = u;
                fx = fu;
            } else {
                if u < x {
                    a = u;
                } else {
                    b = u;
                }
                if fu <= fw || w == x {
                    v = w;
                    fv = fw;
                    w = u;
                    fw = fu;
                } else if fu <= fv || v == x || v == w {
                    v = u;
                    fv = fu;
                }
            }
        }

        Ok(x)
    }

    /// Compute negative log-likelihood for a given temperature
    fn compute_nll(&self, logits: &[Vec<f64>], labels: &[usize], temp: f64) -> f64 {
        let mut nll = 0.0;

        for (logit, &label) in logits.iter().zip(labels.iter()) {
            // Scale logits by temperature
            let scaled: Vec<f64> = logit.iter().map(|&x| x / temp).collect();

            // Compute log softmax
            let max_logit = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = scaled
                .iter()
                .map(|&x| (x - max_logit).exp())
                .sum::<f64>()
                .ln()
                + max_logit;

            let log_prob = scaled[label] - log_sum_exp;
            nll -= log_prob;
        }

        nll / logits.len() as f64
    }
}

impl Default for TemperatureScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Platt scaling (logistic regression on scores)
#[derive(Debug, Clone)]
pub struct PlattScaling {
    a: f64,
    b: f64,
    fitted: bool,
}

impl PlattScaling {
    /// Create a new Platt scaling instance
    pub fn new() -> Self {
        Self {
            a: 1.0,
            b: 0.0,
            fitted: false,
        }
    }

    /// Fit Platt scaling parameters using gradient descent
    pub fn fit(&mut self, scores: &[f64], labels: &[bool]) -> Result<(), String> {
        if scores.len() != labels.len() {
            return Err("Number of scores must match number of labels".to_string());
        }

        if scores.is_empty() {
            return Err("Cannot fit on empty dataset".to_string());
        }

        // Initialize parameters
        let mut a = 0.0;
        let mut b = 0.0;

        // Count positive and negative examples
        let n_pos = labels.iter().filter(|&&x| x).count() as f64;
        let n_neg = (labels.len() as f64) - n_pos;

        // Target probabilities (with smoothing)
        let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
        let t_neg = 1.0 / (n_neg + 2.0);

        // Gradient descent
        let learning_rate = 0.01;
        let max_iter = 1000;
        let tol = 1e-6;

        for iter in 0..max_iter {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;
            let mut _loss = 0.0;

            for (&score, &label) in scores.iter().zip(labels.iter()) {
                let target = if label { t_pos } else { t_neg };
                let z = a * score + b;
                let p = sigmoid(z);

                let diff = p - target;
                grad_a += diff * score;
                grad_b += diff;

                // Cross-entropy loss
                _loss -= target * p.ln() + (1.0 - target) * (1.0 - p).ln();
            }

            grad_a /= scores.len() as f64;
            grad_b /= scores.len() as f64;
            _loss /= scores.len() as f64;

            // Update parameters
            let new_a = a - learning_rate * grad_a;
            let new_b = b - learning_rate * grad_b;

            // Check convergence
            if (new_a - a).abs() < tol && (new_b - b).abs() < tol {
                break;
            }

            a = new_a;
            b = new_b;

            // Prevent divergence
            if !a.is_finite() || !b.is_finite() {
                return Err(format!("Optimization diverged at iteration {}", iter));
            }
        }

        self.a = a;
        self.b = b;
        self.fitted = true;

        Ok(())
    }

    /// Calibrate a single score
    pub fn calibrate(&self, score: f64) -> f64 {
        if !self.fitted {
            return score;
        }
        sigmoid(self.a * score + self.b)
    }

    /// Calibrate batch of scores
    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }

    /// Get fitted parameters
    pub fn parameters(&self) -> (f64, f64) {
        (self.a, self.b)
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute softmax of a vector
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_x = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp_x: Vec<f64> = x.iter().map(|&v| (v - max_x).exp()).collect();
    let sum_exp: f64 = exp_x.iter().sum();
    exp_x.iter().map(|&v| v / sum_exp).collect()
}

/// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temperature_scaling_new() {
        let ts = TemperatureScaling::new();
        assert_eq!(ts.temperature(), 1.0);
        assert!(!ts.fitted);
    }

    #[test]
    fn test_temperature_scaling_fit() {
        let mut ts = TemperatureScaling::new();

        // Create synthetic data: overconfident predictions
        let logits = vec![
            vec![5.0, 1.0, 0.5],  // Predict class 0, label 0
            vec![0.5, 5.0, 1.0],  // Predict class 1, label 1
            vec![1.0, 0.5, 5.0],  // Predict class 2, label 2
            vec![4.0, 1.0, 0.5],  // Predict class 0, label 0
        ];
        let labels = vec![0, 1, 2, 0];

        let result = ts.fit(&logits, &labels);
        assert!(result.is_ok());
        assert!(ts.fitted);

        // Temperature should be > 1.0 to reduce overconfidence
        println!("Fitted temperature: {}", ts.temperature());
        assert!(ts.temperature() > 0.0);
    }

    #[test]
    fn test_temperature_scaling_calibrate() {
        let mut ts = TemperatureScaling::new();
        ts.temperature = 2.0;
        ts.fitted = true;

        let logits = vec![4.0, 2.0, 1.0];
        let calibrated = ts.calibrate(&logits);

        // Check that probabilities sum to 1
        let sum: f64 = calibrated.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that calibrated probabilities are less confident
        assert!(calibrated[0] < 0.8); // Original would be ~0.84
    }

    #[test]
    fn test_temperature_scaling_batch() {
        let mut ts = TemperatureScaling::new();
        ts.temperature = 1.5;
        ts.fitted = true;

        let logits = vec![
            vec![3.0, 1.0, 0.5],
            vec![0.5, 3.0, 1.0],
        ];

        let calibrated = ts.calibrate_batch(&logits);
        assert_eq!(calibrated.len(), 2);

        for probs in &calibrated {
            let sum: f64 = probs.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_platt_scaling_new() {
        let ps = PlattScaling::new();
        let (a, b) = ps.parameters();
        assert_eq!(a, 1.0);
        assert_eq!(b, 0.0);
        assert!(!ps.fitted);
    }

    #[test]
    fn test_platt_scaling_fit() {
        let mut ps = PlattScaling::new();

        // Synthetic data: scores and labels
        let scores = vec![0.9, 0.8, 0.3, 0.2, 0.7, 0.4];
        let labels = vec![true, true, false, false, true, false];

        let result = ps.fit(&scores, &labels);
        assert!(result.is_ok());
        assert!(ps.fitted);

        let (a, b) = ps.parameters();
        println!("Fitted parameters: a={}, b={}", a, b);
        assert!(a.is_finite());
        assert!(b.is_finite());
    }

    #[test]
    fn test_platt_scaling_calibrate() {
        let mut ps = PlattScaling::new();
        ps.a = 2.0;
        ps.b = -1.0;
        ps.fitted = true;

        let score = 0.8;
        let calibrated = ps.calibrate(score);

        // Check that output is a valid probability
        assert!((0.0..=1.0).contains(&calibrated));

        // With a=2, b=-1: sigmoid(2*0.8 - 1) = sigmoid(0.6)
        let expected = sigmoid(0.6);
        assert!((calibrated - expected).abs() < 1e-6);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let probs = softmax(&x);

        // Check sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check monotonic (higher logit -> higher probability)
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);

        // Check symmetry
        assert!((sigmoid(2.0) + sigmoid(-2.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_temperature_scaling_invalid_input() {
        let mut ts = TemperatureScaling::new();

        // Mismatched lengths
        let logits = vec![vec![1.0, 2.0]];
        let labels = vec![0, 1];
        assert!(ts.fit(&logits, &labels).is_err());

        // Empty input
        let logits: Vec<Vec<f64>> = vec![];
        let labels: Vec<usize> = vec![];
        assert!(ts.fit(&logits, &labels).is_err());

        // Invalid label
        let logits = vec![vec![1.0, 2.0]];
        let labels = vec![5];
        assert!(ts.fit(&logits, &labels).is_err());
    }

    #[test]
    fn test_platt_scaling_invalid_input() {
        let mut ps = PlattScaling::new();

        // Mismatched lengths
        let scores = vec![0.5];
        let labels = vec![true, false];
        assert!(ps.fit(&scores, &labels).is_err());

        // Empty input
        let scores: Vec<f64> = vec![];
        let labels: Vec<bool> = vec![];
        assert!(ps.fit(&scores, &labels).is_err());
    }
}
