//! Regression metrics for continuous prediction tasks.

use super::MetricTrait;
use crate::error::{DpbError, Result};

/// Mean Absolute Error (MAE).
#[derive(Debug, Clone)]
pub struct MeanAbsoluteError {
    sum_abs_error: f64,
    count: f64,
}

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Self {
            sum_abs_error: 0.0,
            count: 0.0,
        }
    }
}

impl Default for MeanAbsoluteError {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for MeanAbsoluteError {
    fn name(&self) -> &str {
        "mae"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let mae = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs() as f64)
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mae)
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        let sum: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).abs() as f64)
            .sum();

        self.sum_abs_error += sum;
        self.count += predictions.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.sum_abs_error / self.count
        }
    }

    fn reset(&mut self) {
        self.sum_abs_error = 0.0;
        self.count = 0.0;
    }
}

/// Mean Squared Error (MSE).
#[derive(Debug, Clone)]
pub struct MeanSquaredError {
    sum_squared_error: f64,
    count: f64,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            count: 0.0,
        }
    }
}

impl Default for MeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for MeanSquaredError {
    fn name(&self) -> &str {
        "mse"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let diff = (p - t) as f64;
                diff * diff
            })
            .sum::<f64>()
            / predictions.len() as f64;

        Ok(mse)
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        let sum: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let diff = (p - t) as f64;
                diff * diff
            })
            .sum();

        self.sum_squared_error += sum;
        self.count += predictions.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            self.sum_squared_error / self.count
        }
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.count = 0.0;
    }
}

/// Root Mean Squared Error (RMSE).
#[derive(Debug, Clone)]
pub struct RootMeanSquaredError {
    mse: MeanSquaredError,
}

impl RootMeanSquaredError {
    pub fn new() -> Self {
        Self {
            mse: MeanSquaredError::new(),
        }
    }
}

impl Default for RootMeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for RootMeanSquaredError {
    fn name(&self) -> &str {
        "rmse"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        Ok(self.mse.compute(predictions, targets)?.sqrt())
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        self.mse.update(predictions, targets);
    }

    fn result(&self) -> f64 {
        self.mse.result().sqrt()
    }

    fn reset(&mut self) {
        self.mse.reset();
    }
}

/// R² (Coefficient of Determination).
#[derive(Debug, Clone)]
pub struct RSquared {
    sum_squared_residuals: f64,
    sum_targets: f64,
    sum_squared_targets: f64,
    count: f64,
}

impl RSquared {
    pub fn new() -> Self {
        Self {
            sum_squared_residuals: 0.0,
            sum_targets: 0.0,
            sum_squared_targets: 0.0,
            count: 0.0,
        }
    }
}

impl Default for RSquared {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for RSquared {
    fn name(&self) -> &str {
        "r_squared"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let mean_target = targets.iter().map(|&t| t as f64).sum::<f64>() / targets.len() as f64;

        let ss_res: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let diff = (*t as f64) - (*p as f64);
                diff * diff
            })
            .sum();

        let ss_tot: f64 = targets
            .iter()
            .map(|t| {
                let diff = (*t as f64) - mean_target;
                diff * diff
            })
            .sum();

        if ss_tot == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - (ss_res / ss_tot))
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let diff = (*t as f64) - (*p as f64);
            self.sum_squared_residuals += diff * diff;
            self.sum_targets += *t as f64;
            self.sum_squared_targets += (*t as f64) * (*t as f64);
            self.count += 1.0;
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            return 0.0;
        }

        let mean_target = self.sum_targets / self.count;
        let ss_tot = self.sum_squared_targets - self.count * mean_target * mean_target;

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (self.sum_squared_residuals / ss_tot)
        }
    }

    fn reset(&mut self) {
        self.sum_squared_residuals = 0.0;
        self.sum_targets = 0.0;
        self.sum_squared_targets = 0.0;
        self.count = 0.0;
    }
}

/// Pearson Correlation Coefficient.
#[derive(Debug, Clone)]
pub struct PearsonCorrelation {
    sum_pred: f64,
    sum_target: f64,
    sum_pred_sq: f64,
    sum_target_sq: f64,
    sum_prod: f64,
    count: f64,
}

impl PearsonCorrelation {
    pub fn new() -> Self {
        Self {
            sum_pred: 0.0,
            sum_target: 0.0,
            sum_pred_sq: 0.0,
            sum_target_sq: 0.0,
            sum_prod: 0.0,
            count: 0.0,
        }
    }
}

impl Default for PearsonCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for PearsonCorrelation {
    fn name(&self) -> &str {
        "pearson_correlation"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let n = predictions.len() as f64;
        let sum_p: f64 = predictions.iter().map(|&x| x as f64).sum();
        let sum_t: f64 = targets.iter().map(|&x| x as f64).sum();
        let sum_p_sq: f64 = predictions.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let sum_t_sq: f64 = targets.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let sum_prod: f64 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (*p as f64) * (*t as f64))
            .sum();

        let numerator = n * sum_prod - sum_p * sum_t;
        let denominator = ((n * sum_p_sq - sum_p * sum_p) * (n * sum_t_sq - sum_t * sum_t)).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let p_f64 = *p as f64;
            let t_f64 = *t as f64;

            self.sum_pred += p_f64;
            self.sum_target += t_f64;
            self.sum_pred_sq += p_f64 * p_f64;
            self.sum_target_sq += t_f64 * t_f64;
            self.sum_prod += p_f64 * t_f64;
            self.count += 1.0;
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            return 0.0;
        }

        let n = self.count;
        let numerator = n * self.sum_prod - self.sum_pred * self.sum_target;
        let denominator = ((n * self.sum_pred_sq - self.sum_pred * self.sum_pred)
            * (n * self.sum_target_sq - self.sum_target * self.sum_target))
            .sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn reset(&mut self) {
        self.sum_pred = 0.0;
        self.sum_target = 0.0;
        self.sum_pred_sq = 0.0;
        self.sum_target_sq = 0.0;
        self.sum_prod = 0.0;
        self.count = 0.0;
    }
}

/// Spearman Correlation Coefficient (rank-based).
#[derive(Debug, Clone)]
pub struct SpearmanCorrelation {
    predictions: Vec<f32>,
    targets: Vec<f32>,
}

impl SpearmanCorrelation {
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            targets: Vec::new(),
        }
    }

    fn rank(values: &[f32]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f32)> = values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

        let mut ranks = vec![0.0; values.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-10 {
                j += 1;
            }

            let avg_rank = ((i + j - 1) as f64) / 2.0 + 1.0;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }

            i = j;
        }

        ranks
    }

    fn compute_correlation(predictions: &[f32], targets: &[f32]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        let pred_ranks = Self::rank(predictions);
        let target_ranks = Self::rank(targets);

        let n = predictions.len() as f64;
        let sum_d_sq: f64 = pred_ranks
            .iter()
            .zip(target_ranks.iter())
            .map(|(p, t)| {
                let diff = p - t;
                diff * diff
            })
            .sum();

        1.0 - (6.0 * sum_d_sq) / (n * (n * n - 1.0))
    }
}

impl Default for SpearmanCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for SpearmanCorrelation {
    fn name(&self) -> &str {
        "spearman_correlation"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        Ok(Self::compute_correlation(predictions, targets))
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        self.predictions.extend_from_slice(predictions);
        self.targets.extend_from_slice(targets);
    }

    fn result(&self) -> f64 {
        Self::compute_correlation(&self.predictions, &self.targets)
    }

    fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }
}

/// Mean Absolute Percentage Error (MAPE).
#[derive(Debug, Clone)]
pub struct MeanAbsolutePercentageError {
    sum_percentage_error: f64,
    count: f64,
}

impl MeanAbsolutePercentageError {
    pub fn new() -> Self {
        Self {
            sum_percentage_error: 0.0,
            count: 0.0,
        }
    }
}

impl Default for MeanAbsolutePercentageError {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for MeanAbsolutePercentageError {
    fn name(&self) -> &str {
        "mape"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let mut sum = 0.0;
        let mut valid_count = 0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            if t.abs() > 1e-10 {  // Avoid division by zero
                sum += ((t - p) / t).abs() as f64;
                valid_count += 1;
            }
        }

        if valid_count == 0 {
            Ok(0.0)
        } else {
            Ok(100.0 * sum / valid_count as f64)
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            if t.abs() > 1e-10 {  // Avoid division by zero
                self.sum_percentage_error += ((t - p) / t).abs() as f64;
                self.count += 1.0;
            }
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            0.0
        } else {
            100.0 * self.sum_percentage_error / self.count
        }
    }

    fn reset(&mut self) {
        self.sum_percentage_error = 0.0;
        self.count = 0.0;
    }
}

/// Explained Variance Score.
#[derive(Debug, Clone)]
pub struct ExplainedVariance {
    sum_diff: f64,
    sum_diff_sq: f64,
    sum_target: f64,
    sum_target_sq: f64,
    count: f64,
}

impl ExplainedVariance {
    pub fn new() -> Self {
        Self {
            sum_diff: 0.0,
            sum_diff_sq: 0.0,
            sum_target: 0.0,
            sum_target_sq: 0.0,
            count: 0.0,
        }
    }
}

impl Default for ExplainedVariance {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for ExplainedVariance {
    fn name(&self) -> &str {
        "explained_variance"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let n = predictions.len() as f64;
        let mean_target: f64 = targets.iter().map(|&t| t as f64).sum::<f64>() / n;

        let var_target: f64 = targets
            .iter()
            .map(|&t| {
                let diff = t as f64 - mean_target;
                diff * diff
            })
            .sum::<f64>()
            / n;

        let residuals: Vec<f64> = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (*t as f64) - (*p as f64))
            .collect();

        let mean_residual = residuals.iter().sum::<f64>() / n;

        let var_residual: f64 = residuals
            .iter()
            .map(|&r| {
                let diff = r - mean_residual;
                diff * diff
            })
            .sum::<f64>()
            / n;

        if var_target == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - var_residual / var_target)
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        for (p, t) in predictions.iter().zip(targets.iter()) {
            let diff = (*t as f64) - (*p as f64);
            self.sum_diff += diff;
            self.sum_diff_sq += diff * diff;
            self.sum_target += *t as f64;
            self.sum_target_sq += (*t as f64) * (*t as f64);
            self.count += 1.0;
        }
    }

    fn result(&self) -> f64 {
        if self.count == 0.0 {
            return 0.0;
        }

        let mean_target = self.sum_target / self.count;
        let var_target = (self.sum_target_sq / self.count) - (mean_target * mean_target);

        let mean_diff = self.sum_diff / self.count;
        let var_diff = (self.sum_diff_sq / self.count) - (mean_diff * mean_diff);

        if var_target == 0.0 {
            0.0
        } else {
            1.0 - var_diff / var_target
        }
    }

    fn reset(&mut self) {
        self.sum_diff = 0.0;
        self.sum_diff_sq = 0.0;
        self.sum_target = 0.0;
        self.sum_target_sq = 0.0;
        self.count = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mae() {
        let mut mae = MeanAbsoluteError::new();
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.5, 2.5, 2.5, 3.5];

        let result = mae.compute(&preds, &targets).unwrap();
        assert_eq!(result, 0.5);

        mae.update(&preds, &targets);
        assert_eq!(mae.result(), 0.5);
    }

    #[test]
    fn test_mse() {
        let mse = MeanSquaredError::new();
        let preds = vec![1.0, 2.0, 3.0];
        let targets = vec![1.0, 2.0, 3.0];

        let result = mse.compute(&preds, &targets).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_rmse() {
        let rmse = RootMeanSquaredError::new();
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let result = rmse.compute(&preds, &targets).unwrap();
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_r_squared() {
        let r2 = RSquared::new();
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let result = r2.compute(&preds, &targets).unwrap();
        assert_eq!(result, 1.0);  // Perfect prediction
    }

    #[test]
    fn test_pearson_correlation() {
        let pearson = PearsonCorrelation::new();
        let preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let result = pearson.compute(&preds, &targets).unwrap();
        assert!((result - 1.0).abs() < 1e-6);  // Perfect linear correlation
    }

    #[test]
    fn test_spearman_correlation() {
        let spearman = SpearmanCorrelation::new();
        let preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = spearman.compute(&preds, &targets).unwrap();
        assert!((result - 1.0).abs() < 1e-6);  // Perfect rank correlation
    }

    #[test]
    fn test_mape() {
        let mape = MeanAbsolutePercentageError::new();
        let preds = vec![90.0, 110.0];
        let targets = vec![100.0, 100.0];

        let result = mape.compute(&preds, &targets).unwrap();
        assert!((result - 10.0).abs() < 0.01);  // 10% average error (with tolerance)
    }

    #[test]
    fn test_explained_variance() {
        let ev = ExplainedVariance::new();
        let preds = vec![1.0, 2.0, 3.0, 4.0];
        let targets = vec![1.0, 2.0, 3.0, 4.0];

        let result = ev.compute(&preds, &targets).unwrap();
        assert!((result - 1.0).abs() < 1e-6);  // Perfect prediction
    }
}
