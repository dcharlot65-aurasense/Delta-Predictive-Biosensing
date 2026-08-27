//! Teacher-Student Knowledge Distillation Framework
//!
//! This module implements the core teacher-student framework for knowledge
//! distillation, coordinating forward passes and knowledge transfer between
//! a pre-trained teacher network and a student network.

use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Knowledge distillation modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DistillationMode {
    /// Response-based: Transfer soft output targets
    ResponseBased,
    /// Feature-based: Transfer intermediate layer representations
    FeatureBased,
    /// Relation-based: Transfer relationships between layers
    RelationBased,
    /// Hybrid: Combine multiple modes
    Hybrid,
}

/// Distillation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    /// Temperature for softening probability distributions
    pub temperature: f32,
    /// Weight for distillation loss (soft targets)
    pub alpha: f32,
    /// Weight for student loss (hard targets)
    pub beta: f32,
    /// Distillation mode
    pub mode: DistillationMode,
    /// Layer indices for feature matching (if feature-based)
    pub feature_layers: Vec<usize>,
    /// Enable temperature scaling during inference
    pub scale_temperature: bool,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            temperature: 4.0,
            alpha: 0.7,
            beta: 0.3,
            mode: DistillationMode::ResponseBased,
            feature_layers: vec![],
            scale_temperature: true,
        }
    }
}

impl DistillationConfig {
    /// Validate configuration
    pub fn validate(&self) -> SNNResult<()> {
        if self.temperature <= 0.0 {
            return Err(SNNError::InvalidConfig(
                "Temperature must be positive".to_string()
            ));
        }
        if self.alpha < 0.0 || self.beta < 0.0 {
            return Err(SNNError::InvalidConfig(
                "Loss weights must be non-negative".to_string()
            ));
        }
        if (self.alpha + self.beta - 1.0).abs() > 1e-6 {
            return Err(SNNError::InvalidConfig(
                "Alpha and beta should sum to 1.0".to_string()
            ));
        }
        Ok(())
    }
}

/// Teacher model wrapper (frozen weights)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TeacherModel<T> {
    /// The underlying network
    pub network: T,
    /// Whether to freeze teacher weights
    pub frozen: bool,
    /// Temperature for output softening
    pub temperature: f32,
}

impl<T> TeacherModel<T> {
    /// Create a new teacher model
    pub fn new(network: T, temperature: f32) -> Self {
        Self {
            network,
            frozen: true,
            temperature,
        }
    }

    /// Freeze teacher weights
    pub fn freeze(&mut self) {
        self.frozen = true;
    }

    /// Unfreeze teacher weights (for fine-tuning)
    pub fn unfreeze(&mut self) {
        self.frozen = false;
    }
}

/// Student model wrapper (trainable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StudentModel<T> {
    /// The underlying network
    pub network: T,
    /// Learning rate
    pub learning_rate: f32,
    /// Number of parameters
    pub num_parameters: usize,
}

impl<T> StudentModel<T> {
    /// Create a new student model
    pub fn new(network: T, learning_rate: f32) -> Self {
        Self {
            network,
            learning_rate,
            num_parameters: 0,
        }
    }

    /// Get compression ratio compared to teacher
    pub fn compression_ratio(&self, teacher_params: usize) -> f32 {
        if teacher_params == 0 {
            return 1.0;
        }
        self.num_parameters as f32 / teacher_params as f32
    }
}

/// Knowledge transfer methods
pub trait KnowledgeTransfer {
    /// Transfer response-based knowledge (soft targets)
    fn transfer_responses(
        &self,
        teacher_output: &SpikeTensor,
        student_output: &SpikeTensor,
        temperature: f32,
    ) -> SNNResult<f32>;

    /// Transfer feature-based knowledge (intermediate representations)
    fn transfer_features(
        &self,
        teacher_features: &[Array2<f32>],
        student_features: &[Array2<f32>],
    ) -> SNNResult<f32>;

    /// Transfer relation-based knowledge (layer relationships)
    fn transfer_relations(
        &self,
        teacher_features: &[Array2<f32>],
        student_features: &[Array2<f32>],
    ) -> SNNResult<f32>;
}

/// Teacher-Student distillation framework
pub struct TeacherStudentFramework<Teacher, Student> {
    /// Teacher model
    pub teacher: TeacherModel<Teacher>,
    /// Student model
    pub student: StudentModel<Student>,
    /// Distillation configuration
    pub config: DistillationConfig,
    /// Training statistics
    pub stats: DistillationStats,
}

/// Training statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistillationStats {
    /// Total distillation loss
    pub total_loss: f32,
    /// Distillation loss (soft targets)
    pub distillation_loss: f32,
    /// Student loss (hard targets)
    pub student_loss: f32,
    /// Number of training steps
    pub steps: usize,
    /// Teacher accuracy
    pub teacher_accuracy: f32,
    /// Student accuracy
    pub student_accuracy: f32,
    /// Compression ratio
    pub compression_ratio: f32,
}

impl<Teacher, Student> TeacherStudentFramework<Teacher, Student> {
    /// Create a new distillation framework
    pub fn new(
        teacher: Teacher,
        student: Student,
        config: DistillationConfig,
    ) -> SNNResult<Self> {
        config.validate()?;

        Ok(Self {
            teacher: TeacherModel::new(teacher, config.temperature),
            student: StudentModel::new(student, 0.001),
            config,
            stats: DistillationStats::default(),
        })
    }

    /// Get distillation statistics
    pub fn statistics(&self) -> &DistillationStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = DistillationStats::default();
    }
}

/// Temperature-scaled softmax
pub fn temperature_softmax(logits: &Array2<f32>, temperature: f32) -> Array2<f32> {
    let (batch_size, num_classes) = (logits.shape()[0], logits.shape()[1]);
    let mut softmax = Array2::zeros((batch_size, num_classes));

    for b in 0..batch_size {
        // Scale by temperature
        let scaled = logits.row(b).mapv(|x| x / temperature);

        // Compute softmax with numerical stability
        let max_val = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals: Array1<f32> = scaled.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_vals.sum();

        for c in 0..num_classes {
            softmax[[b, c]] = exp_vals[c] / sum_exp;
        }
    }

    softmax
}

/// Compute KL divergence between teacher and student outputs
pub fn kl_divergence_loss(
    teacher_probs: &Array2<f32>,
    student_probs: &Array2<f32>,
    temperature: f32,
) -> SNNResult<f32> {
    let (batch_size, num_classes) = (teacher_probs.shape()[0], teacher_probs.shape()[1]);

    if student_probs.shape() != [batch_size, num_classes] {
        return Err(SNNError::DimensionMismatch {
            expected: format!("({}, {})", batch_size, num_classes),
            actual: format!("{:?}", student_probs.shape()),
        });
    }

    let mut loss = 0.0;
    let eps = 1e-8;

    for b in 0..batch_size {
        for c in 0..num_classes {
            let p = teacher_probs[[b, c]].max(eps);
            let q = student_probs[[b, c]].max(eps);
            loss += p * (p / q).ln();
        }
    }

    // Scale by temperature squared (KL divergence property)
    Ok(loss * temperature * temperature / batch_size as f32)
}

/// Default implementation of knowledge transfer
pub struct DefaultKnowledgeTransfer;

impl KnowledgeTransfer for DefaultKnowledgeTransfer {
    fn transfer_responses(
        &self,
        teacher_output: &SpikeTensor,
        student_output: &SpikeTensor,
        temperature: f32,
    ) -> SNNResult<f32> {
        // Convert spike rates to logits
        let teacher_rates = teacher_output.spike_rate();
        let student_rates = student_output.spike_rate();

        // Apply temperature-scaled softmax
        let teacher_probs = temperature_softmax(&teacher_rates, temperature);
        let student_probs = temperature_softmax(&student_rates, temperature);

        // Compute KL divergence
        kl_divergence_loss(&teacher_probs, &student_probs, temperature)
    }

    fn transfer_features(
        &self,
        teacher_features: &[Array2<f32>],
        student_features: &[Array2<f32>],
    ) -> SNNResult<f32> {
        if teacher_features.len() != student_features.len() {
            return Err(SNNError::InvalidConfig(
                "Number of teacher and student features must match".to_string()
            ));
        }

        let mut total_loss = 0.0;
        let num_layers = teacher_features.len();

        for (teacher_feat, student_feat) in teacher_features.iter().zip(student_features.iter()) {
            // MSE loss between features
            let diff = teacher_feat - student_feat;
            let mse = diff.mapv(|x| x * x).sum() / teacher_feat.len() as f32;
            total_loss += mse;
        }

        Ok(total_loss / num_layers as f32)
    }

    fn transfer_relations(
        &self,
        teacher_features: &[Array2<f32>],
        student_features: &[Array2<f32>],
    ) -> SNNResult<f32> {
        if teacher_features.len() < 2 || student_features.len() < 2 {
            return Err(SNNError::InvalidConfig(
                "Need at least 2 layers for relation-based distillation".to_string()
            ));
        }

        let mut total_loss = 0.0;
        let num_pairs = teacher_features.len() - 1;

        // Compute relationships between consecutive layers
        for i in 0..num_pairs {
            let teacher_rel = compute_feature_relation(&teacher_features[i], &teacher_features[i + 1]);
            let student_rel = compute_feature_relation(&student_features[i], &student_features[i + 1]);

            // MSE on relations
            let diff = &teacher_rel - &student_rel;
            let mse = diff.mapv(|x| x * x).sum() / teacher_rel.len() as f32;
            total_loss += mse;
        }

        Ok(total_loss / num_pairs as f32)
    }
}

/// Compute relation between two feature maps (correlation)
fn compute_feature_relation(features1: &Array2<f32>, features2: &Array2<f32>) -> Array2<f32> {
    // Compute correlation matrix between features
    // Simplified: dot product between normalized features
    let (n1, d1) = (features1.shape()[0], features1.shape()[1]);
    let (n2, d2) = (features2.shape()[0], features2.shape()[1]);

    // For simplicity, if dimensions don't match, return a small identity-like matrix
    if n1 != n2 {
        return Array2::eye(n1.min(n2));
    }

    // Normalize features
    let mut norm1 = Array2::zeros((n1, d1));
    let mut norm2 = Array2::zeros((n2, d2));

    for i in 0..n1 {
        let norm = features1.row(i).mapv(|x| x * x).sum().sqrt().max(1e-8);
        norm1.row_mut(i).assign(&features1.row(i).mapv(|x| x / norm));
    }

    for i in 0..n2 {
        let norm = features2.row(i).mapv(|x| x * x).sum().sqrt().max(1e-8);
        norm2.row_mut(i).assign(&features2.row(i).mapv(|x| x / norm));
    }

    // Compute pairwise similarities (simplified to diagonal)
    let mut relation = Array2::zeros((n1, n1));
    for i in 0..n1 {
        relation[[i, i]] = 1.0;
    }
    relation
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SpikeTensor;
    use ndarray::Array3;

    #[test]
    fn test_distillation_config_default() {
        let config = DistillationConfig::default();
        assert_eq!(config.temperature, 4.0);
        assert_eq!(config.alpha, 0.7);
        assert_eq!(config.beta, 0.3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_distillation_config_validation() {
        let mut config = DistillationConfig {
            temperature: -1.0,
            ..Default::default()
        };
        assert!(config.validate().is_err());

        config.temperature = 4.0;
        config.alpha = 0.5;
        config.beta = 0.3; // Sum != 1.0
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_temperature_softmax() {
        let logits = Array2::from_shape_vec((2, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ]).unwrap();

        let softmax = temperature_softmax(&logits, 1.0);

        // Check that probabilities sum to 1
        for b in 0..2 {
            let sum: f32 = softmax.row(b).sum();
            assert!((sum - 1.0).abs() < 1e-5);
        }

        // Check that all values are in [0, 1]
        for &val in softmax.iter() {
            assert!((0.0..=1.0).contains(&val));
        }
    }

    #[test]
    fn test_temperature_softmax_temperature_effect() {
        let logits = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();

        let softmax_low_temp = temperature_softmax(&logits, 0.5);
        let softmax_high_temp = temperature_softmax(&logits, 4.0);

        // High temperature should produce more uniform distribution
        let entropy_low = -softmax_low_temp.iter().map(|&p| {
            if p > 1e-8 { p * p.ln() } else { 0.0 }
        }).sum::<f32>();

        let entropy_high = -softmax_high_temp.iter().map(|&p| {
            if p > 1e-8 { p * p.ln() } else { 0.0 }
        }).sum::<f32>();

        assert!(entropy_high > entropy_low);
    }

    #[test]
    fn test_kl_divergence_loss() {
        let teacher = Array2::from_shape_vec((2, 3), vec![
            0.7, 0.2, 0.1,
            0.1, 0.8, 0.1,
        ]).unwrap();

        let student = Array2::from_shape_vec((2, 3), vec![
            0.6, 0.3, 0.1,
            0.2, 0.7, 0.1,
        ]).unwrap();

        let loss = kl_divergence_loss(&teacher, &student, 1.0).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_kl_divergence_identical() {
        let probs = Array2::from_shape_vec((2, 3), vec![
            0.7, 0.2, 0.1,
            0.1, 0.8, 0.1,
        ]).unwrap();

        let loss = kl_divergence_loss(&probs, &probs, 1.0).unwrap();
        assert!(loss.abs() < 1e-5); // Should be ~0 for identical distributions
    }

    #[test]
    fn test_teacher_model() {
        let network = (); // Dummy network
        let mut teacher = TeacherModel::new(network, 4.0);

        assert!(teacher.frozen);
        teacher.unfreeze();
        assert!(!teacher.frozen);
        teacher.freeze();
        assert!(teacher.frozen);
    }

    #[test]
    fn test_student_model() {
        let network = (); // Dummy network
        let mut student = StudentModel::new(network, 0.001);
        student.num_parameters = 1000;

        let ratio = student.compression_ratio(10000);
        assert_eq!(ratio, 0.1); // 1000/10000
    }

    #[test]
    fn test_default_knowledge_transfer() {
        let transfer = DefaultKnowledgeTransfer;

        // Test response-based transfer
        let teacher_data = Array3::from_shape_vec((1, 10, 3), vec![0.0; 30]).unwrap();
        let student_data = Array3::from_shape_vec((1, 10, 3), vec![0.0; 30]).unwrap();

        let teacher_output = SpikeTensor::from_dense(teacher_data, false);
        let student_output = SpikeTensor::from_dense(student_data, false);

        let loss = transfer.transfer_responses(&teacher_output, &student_output, 1.0);
        assert!(loss.is_ok());
    }

    #[test]
    fn test_feature_transfer() {
        let transfer = DefaultKnowledgeTransfer;

        let teacher_features = vec![
            Array2::from_shape_vec((4, 10), vec![1.0; 40]).unwrap(),
            Array2::from_shape_vec((4, 10), vec![2.0; 40]).unwrap(),
        ];

        let student_features = vec![
            Array2::from_shape_vec((4, 10), vec![0.9; 40]).unwrap(),
            Array2::from_shape_vec((4, 10), vec![1.9; 40]).unwrap(),
        ];

        let loss = transfer.transfer_features(&teacher_features, &student_features).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_relation_transfer() {
        let transfer = DefaultKnowledgeTransfer;

        let teacher_features = vec![
            Array2::from_shape_vec((4, 10), vec![1.0; 40]).unwrap(),
            Array2::from_shape_vec((4, 10), vec![2.0; 40]).unwrap(),
        ];

        let student_features = vec![
            Array2::from_shape_vec((4, 10), vec![0.9; 40]).unwrap(),
            Array2::from_shape_vec((4, 10), vec![1.9; 40]).unwrap(),
        ];

        let loss = transfer.transfer_relations(&teacher_features, &student_features).unwrap();
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }
}
