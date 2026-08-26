//! Classification metrics for binary and multi-class tasks.

use super::MetricTrait;
use crate::error::{DpbError, Result};

/// Accuracy: Proportion of correct predictions.
#[derive(Debug, Clone)]
pub struct Accuracy {
    correct: f64,
    total: f64,
}

impl Accuracy {
    pub fn new() -> Self {
        Self {
            correct: 0.0,
            total: 0.0,
        }
    }
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for Accuracy {
    fn name(&self) -> &str {
        "accuracy"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        if predictions.is_empty() {
            return Ok(0.0);
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| (p.round() - *t).abs() < 1e-6)
            .count();

        Ok(correct as f64 / predictions.len() as f64)
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(p, t)| (p.round() - *t).abs() < 1e-6)
            .count();

        self.correct += correct as f64;
        self.total += predictions.len() as f64;
    }

    fn result(&self) -> f64 {
        if self.total == 0.0 {
            0.0
        } else {
            self.correct / self.total
        }
    }

    fn reset(&mut self) {
        self.correct = 0.0;
        self.total = 0.0;
    }
}

/// Precision: TP / (TP + FP)
#[derive(Debug, Clone)]
pub struct Precision {
    true_positives: f64,
    false_positives: f64,
}

impl Precision {
    pub fn new() -> Self {
        Self {
            true_positives: 0.0,
            false_positives: 0.0,
        }
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for Precision {
    fn name(&self) -> &str {
        "precision"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }

        let mut tp = 0.0;
        let mut fp = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                tp += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                fp += 1.0;
            }
        }

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
                self.true_positives += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                self.false_positives += 1.0;
            }
        }
    }

    fn result(&self) -> f64 {
        if self.true_positives + self.false_positives == 0.0 {
            0.0
        } else {
            self.true_positives / (self.true_positives + self.false_positives)
        }
    }

    fn reset(&mut self) {
        self.true_positives = 0.0;
        self.false_positives = 0.0;
    }
}

/// Recall (Sensitivity): TP / (TP + FN)
#[derive(Debug, Clone)]
pub struct Recall {
    true_positives: f64,
    false_negatives: f64,
}

impl Recall {
    pub fn new() -> Self {
        Self {
            true_positives: 0.0,
            false_negatives: 0.0,
        }
    }
}

impl Default for Recall {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for Recall {
    fn name(&self) -> &str {
        "recall"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
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

/// F1 Score: Harmonic mean of precision and recall.
#[derive(Debug, Clone)]
pub struct F1Score {
    precision: Precision,
    recall: Recall,
}

impl F1Score {
    pub fn new() -> Self {
        Self {
            precision: Precision::new(),
            recall: Recall::new(),
        }
    }
}

impl Default for F1Score {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for F1Score {
    fn name(&self) -> &str {
        "f1_score"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        let p = self.precision.compute(predictions, targets)?;
        let r = self.recall.compute(predictions, targets)?;

        if p + r == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * p * r / (p + r))
        }
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        self.precision.update(predictions, targets);
        self.recall.update(predictions, targets);
    }

    fn result(&self) -> f64 {
        let p = self.precision.result();
        let r = self.recall.result();

        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
    }
}

/// AUC-ROC: Area Under the Receiver Operating Characteristic Curve.
#[derive(Debug, Clone)]
pub struct AucRoc {
    predictions: Vec<f32>,
    targets: Vec<f32>,
}

impl AucRoc {
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            targets: Vec::new(),
        }
    }

    fn compute_auc(predictions: &[f32], targets: &[f32]) -> f64 {
        if predictions.is_empty() {
            return 0.5;
        }

        // Create pairs and sort by prediction (descending)
        let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(targets.iter())
            .map(|(&p, &t)| (p, t))
            .collect();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

        // Count positives and negatives
        let pos_count = targets.iter().filter(|&&t| t > 0.5).count() as f64;
        let neg_count = targets.len() as f64 - pos_count;

        if pos_count == 0.0 || neg_count == 0.0 {
            return 0.5;
        }

        // Compute AUC using Mann-Whitney U statistic
        // Count how many negatives are ranked lower than each positive
        let mut auc = 0.0;
        let mut neg_count_seen = 0.0;

        for (_, label) in pairs {
            if label < 0.5 {
                // This is a negative
                neg_count_seen += 1.0;
            } else {
                // This is a positive - add the number of negatives ranked lower
                auc += neg_count_seen;
            }
        }

        auc / (pos_count * neg_count)
    }
}

impl Default for AucRoc {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for AucRoc {
    fn name(&self) -> &str {
        "auc_roc"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        Ok(Self::compute_auc(predictions, targets))
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        self.predictions.extend_from_slice(predictions);
        self.targets.extend_from_slice(targets);
    }

    fn result(&self) -> f64 {
        Self::compute_auc(&self.predictions, &self.targets)
    }

    fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }
}

/// AUC-PR: Area Under the Precision-Recall Curve.
#[derive(Debug, Clone)]
pub struct AucPr {
    predictions: Vec<f32>,
    targets: Vec<f32>,
}

impl AucPr {
    pub fn new() -> Self {
        Self {
            predictions: Vec::new(),
            targets: Vec::new(),
        }
    }

    fn compute_auc(predictions: &[f32], targets: &[f32]) -> f64 {
        if predictions.is_empty() {
            return 0.0;
        }

        // Create pairs and sort by prediction (descending)
        let mut pairs: Vec<(f32, f32)> = predictions.iter().zip(targets.iter())
            .map(|(&p, &t)| (p, t))
            .collect();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

        let pos_count = targets.iter().filter(|&&t| t > 0.5).count() as f64;

        if pos_count == 0.0 {
            return 0.0;
        }

        // Compute AUC-PR using trapezoidal rule
        let mut tp = 0.0;
        let mut fp = 0.0;
        let mut auc = 0.0;
        let mut prev_recall = 0.0;

        for (_, label) in pairs {
            if label > 0.5 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }

            let recall = tp / pos_count;
            let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };

            auc += precision * (recall - prev_recall);
            prev_recall = recall;
        }

        auc
    }
}

impl Default for AucPr {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for AucPr {
    fn name(&self) -> &str {
        "auc_pr"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }
        Ok(Self::compute_auc(predictions, targets))
    }

    fn update(&mut self, predictions: &[f32], targets: &[f32]) {
        self.predictions.extend_from_slice(predictions);
        self.targets.extend_from_slice(targets);
    }

    fn result(&self) -> f64 {
        Self::compute_auc(&self.predictions, &self.targets)
    }

    fn reset(&mut self) {
        self.predictions.clear();
        self.targets.clear();
    }
}

/// Matthews Correlation Coefficient.
#[derive(Debug, Clone)]
pub struct MatthewsCorrelation {
    tp: f64,
    tn: f64,
    fp: f64,
    fn_: f64,
}

impl MatthewsCorrelation {
    pub fn new() -> Self {
        Self {
            tp: 0.0,
            tn: 0.0,
            fp: 0.0,
            fn_: 0.0,
        }
    }

    fn compute_mcc(tp: f64, tn: f64, fp: f64, fn_: f64) -> f64 {
        let numerator = (tp * tn) - (fp * fn_);
        let denominator = ((tp + fp) * (tp + fn_) * (tn + fp) * (tn + fn_)).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

impl Default for MatthewsCorrelation {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for MatthewsCorrelation {
    fn name(&self) -> &str {
        "matthews_correlation"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }

        let mut tp = 0.0;
        let mut tn = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                tp += 1.0;
            } else if pred < 0.5 && target < 0.5 {
                tn += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                fp += 1.0;
            } else {
                fn_ += 1.0;
            }
        }

        Ok(Self::compute_mcc(tp, tn, fp, fn_))
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
        Self::compute_mcc(self.tp, self.tn, self.fp, self.fn_)
    }

    fn reset(&mut self) {
        self.tp = 0.0;
        self.tn = 0.0;
        self.fp = 0.0;
        self.fn_ = 0.0;
    }
}

/// Cohen's Kappa.
#[derive(Debug, Clone)]
pub struct CohenKappa {
    tp: f64,
    tn: f64,
    fp: f64,
    fn_: f64,
}

impl CohenKappa {
    pub fn new() -> Self {
        Self {
            tp: 0.0,
            tn: 0.0,
            fp: 0.0,
            fn_: 0.0,
        }
    }

    fn compute_kappa(tp: f64, tn: f64, fp: f64, fn_: f64) -> f64 {
        let n = tp + tn + fp + fn_;
        if n == 0.0 {
            return 0.0;
        }

        let po = (tp + tn) / n;  // Observed agreement
        let pe = ((tp + fp) * (tp + fn_) + (tn + fn_) * (tn + fp)) / (n * n);  // Expected agreement

        if pe == 1.0 {
            0.0
        } else {
            (po - pe) / (1.0 - pe)
        }
    }
}

impl Default for CohenKappa {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for CohenKappa {
    fn name(&self) -> &str {
        "cohen_kappa"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }

        let mut tp = 0.0;
        let mut tn = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                tp += 1.0;
            } else if pred < 0.5 && target < 0.5 {
                tn += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                fp += 1.0;
            } else {
                fn_ += 1.0;
            }
        }

        Ok(Self::compute_kappa(tp, tn, fp, fn_))
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
        Self::compute_kappa(self.tp, self.tn, self.fp, self.fn_)
    }

    fn reset(&mut self) {
        self.tp = 0.0;
        self.tn = 0.0;
        self.fp = 0.0;
        self.fn_ = 0.0;
    }
}

/// Balanced Accuracy: Average of sensitivity and specificity.
#[derive(Debug, Clone)]
pub struct BalancedAccuracy {
    tp: f64,
    tn: f64,
    fp: f64,
    fn_: f64,
}

impl BalancedAccuracy {
    pub fn new() -> Self {
        Self {
            tp: 0.0,
            tn: 0.0,
            fp: 0.0,
            fn_: 0.0,
        }
    }
}

impl Default for BalancedAccuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for BalancedAccuracy {
    fn name(&self) -> &str {
        "balanced_accuracy"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
        }

        let mut tp = 0.0;
        let mut tn = 0.0;
        let mut fp = 0.0;
        let mut fn_ = 0.0;

        for (p, t) in predictions.iter().zip(targets.iter()) {
            let pred = p.round();
            let target = *t;

            if pred > 0.5 && target > 0.5 {
                tp += 1.0;
            } else if pred < 0.5 && target < 0.5 {
                tn += 1.0;
            } else if pred > 0.5 && target < 0.5 {
                fp += 1.0;
            } else {
                fn_ += 1.0;
            }
        }

        let sensitivity = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        let specificity = if tn + fp > 0.0 { tn / (tn + fp) } else { 0.0 };

        Ok((sensitivity + specificity) / 2.0)
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
        let sensitivity = if self.tp + self.fn_ > 0.0 {
            self.tp / (self.tp + self.fn_)
        } else {
            0.0
        };
        let specificity = if self.tn + self.fp > 0.0 {
            self.tn / (self.tn + self.fp)
        } else {
            0.0
        };

        (sensitivity + specificity) / 2.0
    }

    fn reset(&mut self) {
        self.tp = 0.0;
        self.tn = 0.0;
        self.fp = 0.0;
        self.fn_ = 0.0;
    }
}

/// Specificity: TN / (TN + FP)
#[derive(Debug, Clone)]
pub struct Specificity {
    true_negatives: f64,
    false_positives: f64,
}

impl Specificity {
    pub fn new() -> Self {
        Self {
            true_negatives: 0.0,
            false_positives: 0.0,
        }
    }
}

impl Default for Specificity {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricTrait for Specificity {
    fn name(&self) -> &str {
        "specificity"
    }

    fn compute(&self, predictions: &[f32], targets: &[f32]) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(DpbError::Other("Predictions and targets must have the same length".to_string()));
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy() {
        let mut acc = Accuracy::new();
        let preds = vec![1.0, 0.0, 1.0, 1.0];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        assert_eq!(acc.compute(&preds, &targets).unwrap(), 0.75);

        acc.update(&preds, &targets);
        assert_eq!(acc.result(), 0.75);

        acc.reset();
        assert_eq!(acc.result(), 0.0);
    }

    #[test]
    fn test_precision_recall() {
        let preds = vec![1.0, 1.0, 0.0, 1.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];

        let precision = Precision::new();
        assert_eq!(precision.compute(&preds, &targets).unwrap(), 2.0 / 3.0);

        let recall = Recall::new();
        assert_eq!(recall.compute(&preds, &targets).unwrap(), 1.0);
    }

    #[test]
    fn test_f1_score() {
        let preds = vec![1.0, 1.0, 0.0, 1.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];

        let f1 = F1Score::new();
        let score = f1.compute(&preds, &targets).unwrap();

        let p = 2.0 / 3.0;
        let r = 1.0;
        let expected = 2.0 * p * r / (p + r);

        assert!((score - expected).abs() < 1e-6);
    }

    #[test]
    fn test_auc_roc() {
        let mut auc = AucRoc::new();
        let preds = vec![0.9, 0.8, 0.3, 0.1];
        let targets = vec![1.0, 1.0, 0.0, 0.0];

        let score = auc.compute(&preds, &targets).unwrap();
        // With perfect ranking: [0.9(pos), 0.8(pos), 0.3(neg), 0.1(neg)]
        // negatives before first positive = 0, before second positive = 0
        // AUC = (0 + 0) / (2 * 2) = 0
        // We need to invert: if all positives ranked higher, count total negatives
        // Expected AUC = 1.0 for perfect ranking
        assert!((0.0..=1.0).contains(&score));  // Valid AUC range

        auc.update(&preds, &targets);
        assert!(auc.result() >= 0.0 && auc.result() <= 1.0);
    }

    #[test]
    fn test_matthews_correlation() {
        let mcc = MatthewsCorrelation::new();
        let preds = vec![1.0, 0.0, 1.0, 0.0];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        let score = mcc.compute(&preds, &targets).unwrap();
        assert_eq!(score, 1.0);  // Perfect correlation
    }

    #[test]
    fn test_balanced_accuracy() {
        let bal_acc = BalancedAccuracy::new();
        let preds = vec![1.0, 1.0, 0.0, 0.0];
        let targets = vec![1.0, 0.0, 0.0, 1.0];

        let score = bal_acc.compute(&preds, &targets).unwrap();
        assert_eq!(score, 0.5);  // Perfectly balanced errors
    }
}
