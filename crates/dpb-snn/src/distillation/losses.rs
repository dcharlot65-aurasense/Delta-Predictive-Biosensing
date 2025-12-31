//! Distillation Loss Functions
//!
//! This module implements various loss functions for knowledge distillation,
//! including response-based, feature-based, and attention-based losses.

use crate::{SNNError, SNNResult};
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Deserialize, Serialize};

/// Base trait for distillation loss functions
pub trait DistillationLoss: Send + Sync {
    /// Compute distillation loss
    fn compute(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<f32>;

    /// Compute gradient with respect to student output
    fn gradient(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<Array2<f32>>;

    /// Get loss name
    fn name(&self) -> &str;
}

/// KL Divergence Loss for soft target matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KLDivergenceLoss {
    /// Temperature for softening distributions
    pub temperature: f32,
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl KLDivergenceLoss {
    pub fn new(temperature: f32) -> Self {
        Self {
            temperature,
            eps: 1e-8,
        }
    }
}

impl Default for KLDivergenceLoss {
    fn default() -> Self {
        Self::new(4.0)
    }
}

impl DistillationLoss for KLDivergenceLoss {
    fn compute(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<f32> {
        let (batch_size, num_classes) = (teacher_output.shape()[0], teacher_output.shape()[1]);

        if student_output.shape() != [batch_size, num_classes] {
            return Err(SNNError::DimensionMismatch {
                expected: format!("({}, {})", batch_size, num_classes),
                actual: format!("{:?}", student_output.shape()),
            });
        }

        // Apply temperature-scaled softmax
        let teacher_probs = self.apply_softmax(teacher_output);
        let student_probs = self.apply_softmax(student_output);

        // Compute KL divergence: KL(P||Q) = sum(P * log(P/Q))
        let mut loss = 0.0;
        for b in 0..batch_size {
            for c in 0..num_classes {
                let p = teacher_probs[[b, c]].max(self.eps);
                let q = student_probs[[b, c]].max(self.eps);
                loss += p * (p / q).ln();
            }
        }

        // Scale by temperature squared
        Ok(loss * self.temperature * self.temperature / batch_size as f32)
    }

    fn gradient(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<Array2<f32>> {
        let teacher_probs = self.apply_softmax(teacher_output);
        let student_probs = self.apply_softmax(student_output);

        let (batch_size, num_classes) = (teacher_probs.shape()[0], teacher_probs.shape()[1]);
        let mut gradient = Array2::zeros((batch_size, num_classes));

        // Gradient of KL divergence w.r.t. student logits
        for b in 0..batch_size {
            for c in 0..num_classes {
                let p = teacher_probs[[b, c]];
                let q = student_probs[[b, c]].max(self.eps);
                gradient[[b, c]] = self.temperature * (q - p) / batch_size as f32;
            }
        }

        Ok(gradient)
    }

    fn name(&self) -> &str {
        "KLDivergence"
    }
}

impl KLDivergenceLoss {
    fn apply_softmax(&self, logits: &Array2<f32>) -> Array2<f32> {
        let (batch_size, num_classes) = (logits.shape()[0], logits.shape()[1]);
        let mut softmax = Array2::zeros((batch_size, num_classes));

        for b in 0..batch_size {
            // Scale by temperature
            let scaled = logits.row(b).mapv(|x| x / self.temperature);
            let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let exp_vals: Array1<f32> = scaled.mapv(|x| (x - max_val).exp());
            let sum_exp = exp_vals.sum();

            for c in 0..num_classes {
                softmax[[b, c]] = exp_vals[c] / sum_exp;
            }
        }

        softmax
    }
}

/// Mean Squared Error Loss for feature matching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MSELoss {
    /// Normalization factor
    pub normalize: bool,
}

impl MSELoss {
    pub fn new(normalize: bool) -> Self {
        Self { normalize }
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new(true)
    }
}

impl DistillationLoss for MSELoss {
    fn compute(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<f32> {
        if teacher_output.shape() != student_output.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", teacher_output.shape()),
                actual: format!("{:?}", student_output.shape()),
            });
        }

        let diff = teacher_output - student_output;
        let mse = diff.mapv(|x| x * x).sum();

        if self.normalize {
            Ok(mse / teacher_output.len() as f32)
        } else {
            Ok(mse)
        }
    }

    fn gradient(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<Array2<f32>> {
        let diff = student_output - teacher_output;
        let scale = if self.normalize {
            2.0 / teacher_output.len() as f32
        } else {
            2.0
        };

        Ok(diff * scale)
    }

    fn name(&self) -> &str {
        "MSE"
    }
}

/// Cosine Similarity Loss for representation alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CosineSimLoss {
    /// Epsilon for numerical stability
    pub eps: f32,
}

impl CosineSimLoss {
    pub fn new() -> Self {
        Self { eps: 1e-8 }
    }
}

impl Default for CosineSimLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillationLoss for CosineSimLoss {
    fn compute(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<f32> {
        if teacher_output.shape() != student_output.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", teacher_output.shape()),
                actual: format!("{:?}", student_output.shape()),
            });
        }

        let batch_size = teacher_output.shape()[0];
        let mut total_loss = 0.0;

        for b in 0..batch_size {
            let teacher_vec = teacher_output.row(b);
            let student_vec = student_output.row(b);

            // Compute cosine similarity
            let dot_product = teacher_vec.iter().zip(student_vec.iter())
                .map(|(t, s)| t * s)
                .sum::<f32>();

            let teacher_norm = teacher_vec.mapv(|x| x * x).sum().sqrt().max(self.eps);
            let student_norm = student_vec.mapv(|x| x * x).sum().sqrt().max(self.eps);

            let cosine_sim = dot_product / (teacher_norm * student_norm);

            // Loss is 1 - cosine similarity (0 when identical)
            total_loss += 1.0 - cosine_sim;
        }

        Ok(total_loss / batch_size as f32)
    }

    fn gradient(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<Array2<f32>> {
        let (batch_size, dim) = (teacher_output.shape()[0], teacher_output.shape()[1]);
        let mut gradient = Array2::zeros((batch_size, dim));

        for b in 0..batch_size {
            let t = teacher_output.row(b);
            let s = student_output.row(b);

            let dot = t.iter().zip(s.iter()).map(|(a, b)| a * b).sum::<f32>();
            let t_norm = t.mapv(|x| x * x).sum().sqrt().max(self.eps);
            let s_norm = s.mapv(|x| x * x).sum().sqrt().max(self.eps);

            // Gradient of cosine similarity
            for i in 0..dim {
                gradient[[b, i]] = -(t[i] / (t_norm * s_norm) - dot * s[i] / (t_norm * s_norm.powi(3)));
            }
        }

        Ok(gradient / batch_size as f32)
    }

    fn name(&self) -> &str {
        "CosineSimilarity"
    }
}

/// Hint Loss for intermediate layer matching (FitNets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HintLoss {
    /// Weight for hint loss
    pub weight: f32,
    /// Use L2 distance (else L1)
    pub use_l2: bool,
}

impl HintLoss {
    pub fn new(weight: f32, use_l2: bool) -> Self {
        Self { weight, use_l2 }
    }
}

impl Default for HintLoss {
    fn default() -> Self {
        Self::new(1.0, true)
    }
}

impl DistillationLoss for HintLoss {
    fn compute(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<f32> {
        if teacher_output.shape() != student_output.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", teacher_output.shape()),
                actual: format!("{:?}", student_output.shape()),
            });
        }

        let diff = teacher_output - student_output;
        let loss = if self.use_l2 {
            diff.mapv(|x| x * x).sum()
        } else {
            diff.mapv(|x| x.abs()).sum()
        };

        Ok(self.weight * loss / teacher_output.len() as f32)
    }

    fn gradient(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<Array2<f32>> {
        let diff = student_output - teacher_output;
        let scale = self.weight / teacher_output.len() as f32;

        let gradient = if self.use_l2 {
            diff * (2.0 * scale)
        } else {
            diff.mapv(|x| if x > 0.0 { scale } else if x < 0.0 { -scale } else { 0.0 })
        };

        Ok(gradient)
    }

    fn name(&self) -> &str {
        "Hint"
    }
}

/// Attention Transfer Loss (Zagoruyko & Komodakis, 2017)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionTransferLoss {
    /// Power for attention map computation
    pub p: f32,
    /// Weight for loss
    pub weight: f32,
}

impl AttentionTransferLoss {
    pub fn new(p: f32, weight: f32) -> Self {
        Self { p, weight }
    }

    /// Compute attention map from features
    fn compute_attention_map(&self, features: &Array2<f32>) -> Array2<f32> {
        // Attention is sum of absolute values raised to power p
        features.mapv(|x| x.abs().powf(self.p))
    }
}

impl Default for AttentionTransferLoss {
    fn default() -> Self {
        Self::new(2.0, 1.0)
    }
}

impl DistillationLoss for AttentionTransferLoss {
    fn compute(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<f32> {
        // Compute attention maps
        let teacher_attn = self.compute_attention_map(teacher_output);
        let student_attn = self.compute_attention_map(student_output);

        // Normalize attention maps
        let teacher_norm = teacher_attn.sum().max(1e-8);
        let student_norm = student_attn.sum().max(1e-8);

        let teacher_attn_norm = &teacher_attn / teacher_norm;
        let student_attn_norm = &student_attn / student_norm;

        // Compute L2 distance
        let diff = &teacher_attn_norm - &student_attn_norm;
        let loss = diff.mapv(|x| x * x).sum();

        Ok(self.weight * loss / teacher_attn_norm.len() as f32)
    }

    fn gradient(
        &self,
        teacher_output: &Array2<f32>,
        student_output: &Array2<f32>,
    ) -> SNNResult<Array2<f32>> {
        // Simplified gradient (approximate)
        let teacher_attn = self.compute_attention_map(teacher_output);
        let student_attn = self.compute_attention_map(student_output);

        let teacher_norm = teacher_attn.sum().max(1e-8);
        let student_norm = student_attn.sum().max(1e-8);

        let diff = &student_attn / student_norm - &teacher_attn / teacher_norm;
        let scale = 2.0 * self.weight / diff.len() as f32;

        Ok(diff * scale)
    }

    fn name(&self) -> &str {
        "AttentionTransfer"
    }
}

/// Combined weighted loss
pub struct CombinedDistillationLoss {
    /// Individual losses with weights
    pub losses: Vec<(Box<dyn DistillationLoss>, f32)>,
}

impl std::fmt::Debug for CombinedDistillationLoss {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CombinedDistillationLoss")
            .field("losses", &format!("{} loss function(s)", self.losses.len()))
            .finish()
    }
}

/// Loss weights configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossWeights {
    pub kl_weight: f32,
    pub mse_weight: f32,
    pub cosine_weight: f32,
    pub hint_weight: f32,
    pub attention_weight: f32,
}

impl Default for LossWeights {
    fn default() -> Self {
        Self {
            kl_weight: 1.0,
            mse_weight: 0.0,
            cosine_weight: 0.0,
            hint_weight: 0.0,
            attention_weight: 0.0,
        }
    }
}

impl LossWeights {
    /// Create a balanced combination
    pub fn balanced() -> Self {
        Self {
            kl_weight: 0.4,
            mse_weight: 0.2,
            cosine_weight: 0.2,
            hint_weight: 0.1,
            attention_weight: 0.1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kl_divergence_loss() {
        let loss_fn = KLDivergenceLoss::new(1.0);

        let teacher = Array2::from_shape_vec((2, 3), vec![
            2.0, 1.0, 0.5,
            1.0, 3.0, 0.5,
        ]).unwrap();

        let student = Array2::from_shape_vec((2, 3), vec![
            1.8, 1.2, 0.5,
            1.2, 2.8, 0.6,
        ]).unwrap();

        let loss = loss_fn.compute(&teacher, &student).unwrap();
        assert!(loss >= 0.0);
        assert!(loss.is_finite());

        let grad = loss_fn.gradient(&teacher, &student).unwrap();
        assert_eq!(grad.shape(), [2, 3]);
    }

    #[test]
    fn test_mse_loss() {
        let loss_fn = MSELoss::new(true);

        let teacher = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]).unwrap();

        let student = Array2::from_shape_vec((2, 4), vec![
            1.1, 2.1, 2.9, 4.1,
            4.9, 6.1, 7.1, 7.9,
        ]).unwrap();

        let loss = loss_fn.compute(&teacher, &student).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());

        let grad = loss_fn.gradient(&teacher, &student).unwrap();
        assert_eq!(grad.shape(), [2, 4]);
    }

    #[test]
    fn test_cosine_sim_loss() {
        let loss_fn = CosineSimLoss::new();

        let teacher = Array2::from_shape_vec((2, 4), vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ]).unwrap();

        let student = Array2::from_shape_vec((2, 4), vec![
            0.9, 0.1, 0.0, 0.0,
            0.1, 0.9, 0.0, 0.0,
        ]).unwrap();

        let loss = loss_fn.compute(&teacher, &student).unwrap();
        assert!(loss > 0.0);
        assert!(loss < 1.0); // Should be small since vectors are similar
        assert!(loss.is_finite());
    }

    #[test]
    fn test_hint_loss() {
        let loss_fn = HintLoss::new(1.0, true);

        let teacher = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]).unwrap();

        let student = Array2::from_shape_vec((2, 4), vec![
            1.1, 2.1, 2.9, 4.1,
            4.9, 6.1, 7.1, 7.9,
        ]).unwrap();

        let loss = loss_fn.compute(&teacher, &student).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_attention_transfer_loss() {
        let loss_fn = AttentionTransferLoss::new(2.0, 1.0);

        let teacher = Array2::from_shape_vec((2, 4), vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
        ]).unwrap();

        let student = Array2::from_shape_vec((2, 4), vec![
            1.1, 2.1, 2.9, 4.1,
            4.9, 6.1, 7.1, 7.9,
        ]).unwrap();

        let loss = loss_fn.compute(&teacher, &student).unwrap();
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_loss_weights() {
        let weights = LossWeights::default();
        assert_eq!(weights.kl_weight, 1.0);

        let balanced = LossWeights::balanced();
        let total: f32 = balanced.kl_weight + balanced.mse_weight +
                        balanced.cosine_weight + balanced.hint_weight +
                        balanced.attention_weight;
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let loss_fn = MSELoss::new(true);

        let teacher = Array2::from_shape_vec((2, 4), vec![1.0; 8]).unwrap();
        let student = Array2::from_shape_vec((2, 3), vec![1.0; 6]).unwrap();

        assert!(loss_fn.compute(&teacher, &student).is_err());
    }
}
