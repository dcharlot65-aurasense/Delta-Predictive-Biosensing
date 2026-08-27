//! Clinical validity metrics for medical device validation.

use super::MetricTrait;
use crate::error::{DpbError, Result};

/// Intraclass Correlation Coefficient (ICC).
#[derive(Debug, Clone)]
pub struct IntraclassCorrelation {
    sum_x: f64,
    sum_y: f64,
    sum_x_sq: f64,
    sum_y_sq: f64,
    sum_xy: f64,
    count: f64,
}

impl IntraclassCorrelation {
    /// Creates a new [`IntraclassCorrelation`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            sum_x: 0.0,
            sum_y: 0.0,
            sum_x_sq: 0.0,
            sum_y_sq: 0.0,
            sum_xy: 0.0,
            count: 0.0,
        }
    }

    fn compute_icc(x: &[f32], y: &[f32]) -> f64 {
        let n = x.len() as f64;
        if n == 0.0 {
            return 0.0;
        }

        // Calculate means
        let mean_x: f64 = x.iter().map(|&v| v as f64).sum::<f64>() / n;
        let mean_y: f64 = y.iter().map(|&v| v as f64).sum::<f64>() / n;
        let grand_mean = (mean_x + mean_y) / 2.0;

        // Between-subject variance
        let mut bs_var = 0.0;
        for i in 0..x.len() {
            let pair_mean = ((x[i] as f64) + (y[i] as f64)) / 2.0;
            let diff = pair_mean - grand_mean;
            bs_var += diff * diff;
        }
        bs_var /= n - 1.0;

        // Within-subject variance
        let mut ws_var = 0.0;
        for i in 0..x.len() {
            let diff = (x[i] as f64) - (y[i] as f64);
            ws_var += diff * diff;
        }
        ws_var /= 2.0 * n;

        // ICC calculation
        if bs_var + ws_var == 0.0 {
            0.0
        } else {
            (bs_var - ws_var) / (bs_var + ws_var)
        }
    }
}

impl Default for IntraclassCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for IntraclassCorrelation {
    fn name(&self) -> &str {
        "icc"
    }

    fn compute(&self, x: &[f32], y: &[f32]) -> Result<f64> {
        if x.len() != y.len() {
            return Err(DpbError::Other(
                "Arrays must have the same length".to_string(),
            ));
        }
        Ok(Self::compute_icc(x, y))
    }

    fn update(&mut self, x: &[f32], y: &[f32]) {
        for (x_val, y_val) in x.iter().zip(y.iter()) {
            let x_f64 = *x_val as f64;
            let y_f64 = *y_val as f64;

            self.sum_x += x_f64;
            self.sum_y += y_f64;
            self.sum_x_sq += x_f64 * x_f64;
            self.sum_y_sq += y_f64 * y_f64;
            self.sum_xy += x_f64 * y_f64;
            self.count += 1.0;
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            return 0.0;
        }

        let mean_x = self.sum_x / self.count;
        let mean_y = self.sum_y / self.count;

        // Approximation: use variance decomposition
        let var_x = (self.sum_x_sq / self.count) - (mean_x * mean_x);
        let var_y = (self.sum_y_sq / self.count) - (mean_y * mean_y);
        let cov_xy = (self.sum_xy / self.count) - (mean_x * mean_y);

        let total_var = var_x + var_y;
        if total_var == 0.0 {
            0.0
        } else {
            (2.0 * cov_xy) / total_var
        }
    }

    fn reset(&mut self) {
        self.sum_x = 0.0;
        self.sum_y = 0.0;
        self.sum_x_sq = 0.0;
        self.sum_y_sq = 0.0;
        self.sum_xy = 0.0;
        self.count = 0.0;
    }
}

/// Bland-Altman Bias (mean difference).
#[derive(Debug, Clone)]
pub struct BlandAltmanBias {
    sum_diff: f64,
    count: f64,
}

impl BlandAltmanBias {
    /// Creates a new [`BlandAltmanBias`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            sum_diff: 0.0,
            count: 0.0,
        }
    }
}

impl Default for BlandAltmanBias {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for BlandAltmanBias {
    fn name(&self) -> &str {
        "bland_altman_bias"
    }

    fn compute(&self, method1: &[f32], method2: &[f32]) -> Result<f64> {
        if method1.len() != method2.len() {
            return Err(DpbError::Other(
                "Methods must have the same length".to_string(),
            ));
        }
        if method1.is_empty() {
            return Ok(0.0);
        }

        let diff_sum: f64 = method1
            .iter()
            .zip(method2.iter())
            .map(|(m1, m2)| (*m1 as f64) - (*m2 as f64))
            .sum();

        Ok(diff_sum / method1.len() as f64)
    }

    fn update(&mut self, method1: &[f32], method2: &[f32]) {
        let diff: f64 = method1
            .iter()
            .zip(method2.iter())
            .map(|(m1, m2)| (*m1 as f64) - (*m2 as f64))
            .sum();

        self.sum_diff += diff;
        self.count += method1.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.sum_diff / self.count
        }
    }

    fn reset(&mut self) {
        self.sum_diff = 0.0;
        self.count = 0.0;
    }
}

/// Bland-Altman Limits of Agreement.
#[derive(Debug, Clone)]
pub struct BlandAltmanLimits {
    differences: Vec<f64>,
}

impl BlandAltmanLimits {
    /// Creates a new [`BlandAltmanLimits`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            differences: Vec::new(),
        }
    }
}

impl Default for BlandAltmanLimits {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for BlandAltmanLimits {
    fn name(&self) -> &str {
        "bland_altman_limits"
    }

    fn compute(&self, method1: &[f32], method2: &[f32]) -> Result<f64> {
        if method1.len() != method2.len() {
            return Err(DpbError::Other(
                "Methods must have the same length".to_string(),
            ));
        }
        if method1.is_empty() {
            return Ok(0.0);
        }

        let diffs: Vec<f64> = method1
            .iter()
            .zip(method2.iter())
            .map(|(m1, m2)| (*m1 as f64) - (*m2 as f64))
            .collect();

        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;

        let variance = diffs
            .iter()
            .map(|&d| {
                let diff = d - mean_diff;
                diff * diff
            })
            .sum::<f64>()
            / (diffs.len() - 1) as f64;

        let sd = variance.sqrt();

        // Return width of limits (upper - lower = 2 * 1.96 * SD)
        Ok(2.0 * 1.96 * sd)
    }

    fn update(&mut self, method1: &[f32], method2: &[f32]) {
        for (m1, m2) in method1.iter().zip(method2.iter()) {
            self.differences.push((*m1 as f64) - (*m2 as f64));
        }
    }

    fn result(&self) -> f64 {
        if self.differences.len() < 2 {
            return 0.0;
        }

        let mean_diff = self.differences.iter().sum::<f64>() / self.differences.len() as f64;

        let variance = self
            .differences
            .iter()
            .map(|&d| {
                let diff = d - mean_diff;
                diff * diff
            })
            .sum::<f64>()
            / (self.differences.len() - 1) as f64;

        let sd = variance.sqrt();

        // Return width of limits
        2.0 * 1.96 * sd
    }

    fn reset(&mut self) {
        self.differences.clear();
    }
}

/// Clinical Sensitivity (True Positive Rate).
#[derive(Debug, Clone)]
pub struct ClinicalSensitivity {
    true_positives: f64,
    false_negatives: f64,
}

impl ClinicalSensitivity {
    /// Creates a new [`ClinicalSensitivity`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            true_positives: 0.0,
            false_negatives: 0.0,
        }
    }
}

impl Default for ClinicalSensitivity {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for ClinicalSensitivity {
    fn name(&self) -> &str {
        "clinical_sensitivity"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other(
                "Predictions and targets must have the same length".to_string(),
            ));
        }

        let mut tp = 0.0;
        let mut fn_count = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                tp += 1.0;
            } else if pred < 0.5 && target > 0.5 {
                fn_count += 1.0;
            }
        }

        if tp + fn_count == 0.0 {
            Ok(0.0)
        } else {
            Ok(tp / (tp + fn_count))
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                self.true_positives += 1.0;
            } else if pred < 0.5 && target > 0.5 {
                self.false_negatives += 1.0;
            }
        }
    }

    fn result(&self) -> f64 {
        if self.true_positives + self.false_negatives == 0.0 {
            0.0
        } else {
            self.true_positives / (self.true_positives + self.false_negatives)
        }
    }

    fn reset(&mut self) {
        self.true_positives = 0.0;
        self.false_negatives = 0.0;
    }
}

/// Clinical Specificity (True Negative Rate).
#[derive(Debug, Clone)]
pub struct ClinicalSpecificity {
    true_negatives: f64,
    false_positives: f64,
}

impl ClinicalSpecificity {
    /// Creates a new [`ClinicalSpecificity`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            true_negatives: 0.0,
            false_positives: 0.0,
        }
    }
}

impl Default for ClinicalSpecificity {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for ClinicalSpecificity {
    fn name(&self) -> &str {
        "clinical_specificity"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other(
                "Predictions and targets must have the same length".to_string(),
            ));
        }

        let mut tn = 0.0;
        let mut fp = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred < 0.5 && target < 0.5 {
                tn += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                fp += 1.0;
            }
        }

        if tn + fp == 0.0 {
            Ok(0.0)
        } else {
            Ok(tn / (tn + fp))
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred < 0.5 && target < 0.5 {
                self.true_negatives += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                self.false_positives += 1.0;
            }
        }
    }

    fn result(&self) -> f64 {
        if self.true_negatives + self.false_positives == 0.0 {
            0.0
        } else {
            self.true_negatives / (self.true_negatives + self.false_positives)
        }
    }

    fn reset(&mut self) {
        self.true_negatives = 0.0;
        self.false_positives = 0.0;
    }
}

/// Positive and Negative Predictive Value (combined metric).
#[derive(Debug, Clone)]
pub struct PredictiveValue {
    tp: f64,
    tn: f64,
    fp: f64,
    fn_: f64,
}

impl PredictiveValue {
    /// Creates a new [`PredictiveValue`] accumulator with no observations recorded.
    pub fn new() -> Self {
        Self {
            tp: 0.0,
            tn: 0.0,
            fp: 0.0,
            fn_: 0.0,
        }
    }

    /// Returns (PPV, NPV)
    pub fn ppv_npv(&self) -> (f64, f64) {
        let ppv = if self.tp + self.fp == 0.0 {
            0.0
        } else {
            self.tp / (self.tp + self.fp)
        };

        let npv = if self.tn + self.fn_ == 0.0 {
            0.0
        } else {
            self.tn / (self.tn + self.fn_)
        };

        (ppv, npv)
    }
}

impl Default for PredictiveValue {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for PredictiveValue {
    fn name(&self) -> &str {
        "predictive_value"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other(
                "Predictions and targets must have the same length".to_string(),
            ));
        }

        let mut tp = 0.0;
        let mut _tn = 0.0;
        let mut fp = 0.0;
        let mut _fn_ = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                tp += 1.0;
            } else if pred < 0.5 && target < 0.5 {
                _tn += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                fp += 1.0;
            } else {
                _fn_ += 1.0;
            }
        }

        // Return PPV as the primary metric
        if tp + fp == 0.0 {
            Ok(0.0)
        } else {
            Ok(tp / (tp + fp))
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                self.tp += 1.0;
            } else if pred < 0.5 && target < 0.5 {
                self.tn += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                self.fp += 1.0;
            } else {
                self.fn_ += 1.0;
            }
        }
    }

    fn result(&self) -> f64 {
        // Return PPV as the primary metric
        if self.tp + self.fp == 0.0 {
            0.0
        } else {
            self.tp / (self.tp + self.fp)
        }
    }

    fn reset(&mut self) {
        self.tp = 0.0;
        self.tn = 0.0;
        self.fp = 0.0;
        self.fn_ = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_icc() {
        let icc = IntraclassCorrelation::new();
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.1, 2.1, 3.1, 4.1];

        let result = icc.compute(&x, &y).unwrap();
        assert!(result > 0.9); // High agreement
    }

    #[test]
    fn test_bland_altman_bias() {
        let mut bias = BlandAltmanBias::new();
        let method1 = vec![10.0, 20.0, 30.0, 40.0];
        let method2 = vec![11.0, 21.0, 31.0, 41.0];

        let result = bias.compute(&method1, &method2).unwrap();
        assert_eq!(result, -1.0); // Consistent bias of -1

        bias.update(&method1, &method2);
        assert_eq!(bias.result(), -1.0);
    }

    #[test]
    fn test_bland_altman_limits() {
        let limits = BlandAltmanLimits::new();
        let method1 = vec![10.0, 20.0, 30.0, 40.0];
        let method2 = vec![10.0, 20.0, 30.0, 40.0];

        let result = limits.compute(&method1, &method2).unwrap();
        assert_eq!(result, 0.0); // Identical methods
    }

    #[test]
    fn test_clinical_sensitivity() {
        let sens = ClinicalSensitivity::new();
        let preds = vec![1.0, 1.0, 0.0, 1.0];
        let targets = vec![1.0, 1.0, 0.0, 1.0];

        let result = sens.compute(&preds, &targets).unwrap();
        assert_eq!(result, 1.0); // Perfect sensitivity
    }

    #[test]
    fn test_clinical_specificity() {
        let spec = ClinicalSpecificity::new();
        let preds = vec![0.0, 0.0, 1.0, 0.0];
        let targets = vec![0.0, 0.0, 1.0, 0.0];

        let result = spec.compute(&preds, &targets).unwrap();
        assert_eq!(result, 1.0); // Perfect specificity
    }

    #[test]
    fn test_predictive_value() {
        let mut pv = PredictiveValue::new();
        let preds = vec![1.0, 1.0, 0.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];

        pv.update(&preds, &targets);
        let (ppv, npv) = pv.ppv_npv();

        assert_eq!(ppv, 0.5); // 1 TP out of 2 positive predictions
        assert_eq!(npv, 0.5); // 1 TN out of 2 negative predictions
    }
}
