//! Cognitive-Motor Fusion for comprehensive integrated assessment
//!
//! This module provides fusion of cognitive and motor assessments to enable:
//! - Cross-modal integration for comprehensive health evaluation
//! - Cognitive-motor dissociation detection (important for neurological conditions)
//! - Longitudinal change tracking across both domains
//!
//! Clinical applications include Parkinson's disease, stroke recovery,
//! traumatic brain injury, and age-related decline assessment.

use super::{FusionConfig, FusionNetwork, Modality};
use crate::{SNNError, SNNResult, SpikeTensor, SpikingLinear, NeuronParams};
use crate::layers::SpikingLayer;
use ndarray::Array3;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cognitive profile from cognitive assessments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveProfile {
    /// Attention metrics (0-1 scale, higher is better)
    pub attention_score: f64,
    /// Working memory capacity (0-1 scale)
    pub working_memory_score: f64,
    /// Executive function (0-1 scale)
    pub executive_function_score: f64,
    /// Processing speed (0-1 scale)
    pub processing_speed_score: f64,
    /// Reaction time variability (coefficient of variation)
    pub reaction_time_cv: f64,
    /// D-prime from signal detection tasks
    pub d_prime: f64,
    /// Response bias (criterion c)
    pub response_bias: f64,
    /// Composite cognitive score
    pub composite_score: f64,
    /// Raw feature vector for neural network input
    pub feature_vector: Vec<f64>,
}

impl CognitiveProfile {
    /// Create a new cognitive profile from scores
    pub fn new(
        attention: f64,
        working_memory: f64,
        executive_function: f64,
        processing_speed: f64,
        rt_cv: f64,
        d_prime: f64,
        response_bias: f64,
    ) -> Self {
        // Compute composite as weighted average
        let composite = 0.25 * attention
            + 0.25 * working_memory
            + 0.20 * executive_function
            + 0.15 * processing_speed
            + 0.15 * (1.0 - rt_cv.min(1.0)); // Lower CV is better

        let feature_vector = vec![
            attention,
            working_memory,
            executive_function,
            processing_speed,
            rt_cv,
            d_prime,
            response_bias,
            composite,
        ];

        Self {
            attention_score: attention,
            working_memory_score: working_memory,
            executive_function_score: executive_function,
            processing_speed_score: processing_speed,
            reaction_time_cv: rt_cv,
            d_prime,
            response_bias,
            composite_score: composite,
            feature_vector,
        }
    }

    /// Create from reaction time metrics
    pub fn from_reaction_time(
        mean_rt: f64,
        std_rt: f64,
        accuracy: f64,
        anticipations: usize,
        lapses: usize,
        total_trials: usize,
    ) -> Self {
        // Normalize RT (assume 200-800ms is normal range)
        let processing_speed = 1.0 - ((mean_rt - 200.0) / 600.0).clamp(0.0, 1.0);

        // CV as RT variability
        let rt_cv = if mean_rt > 0.0 { std_rt / mean_rt } else { 1.0 };

        // Attention based on lapses
        let lapse_rate = lapses as f64 / total_trials as f64;
        let attention = 1.0 - lapse_rate;

        // Impulsivity/executive function based on anticipations
        let anticipation_rate = anticipations as f64 / total_trials as f64;
        let executive_function = 1.0 - anticipation_rate;

        // Working memory placeholder (would need N-back data)
        let working_memory = accuracy;

        // D-prime (simplified - assumes equal target/non-target)
        let hit_rate = accuracy.clamp(0.01, 0.99);
        let fa_rate = anticipation_rate.clamp(0.01, 0.99);
        let d_prime = Self::calculate_d_prime(hit_rate, fa_rate);

        // Response bias
        let response_bias = Self::calculate_criterion(hit_rate, fa_rate);

        Self::new(
            attention,
            working_memory,
            executive_function,
            processing_speed,
            rt_cv,
            d_prime,
            response_bias,
        )
    }

    /// Calculate d-prime (sensitivity) from hit rate and false alarm rate
    fn calculate_d_prime(hit_rate: f64, fa_rate: f64) -> f64 {
        use std::f64::consts::SQRT_2;

        // Inverse normal (z-score) approximation
        let z_hit = Self::inverse_normal(hit_rate);
        let z_fa = Self::inverse_normal(fa_rate);

        z_hit - z_fa
    }

    /// Calculate criterion c (response bias)
    fn calculate_criterion(hit_rate: f64, fa_rate: f64) -> f64 {
        let z_hit = Self::inverse_normal(hit_rate);
        let z_fa = Self::inverse_normal(fa_rate);

        -0.5 * (z_hit + z_fa)
    }

    /// Approximate inverse normal distribution
    fn inverse_normal(p: f64) -> f64 {
        // Rational approximation for probit function
        let p = p.clamp(0.0001, 0.9999);

        if p < 0.5 {
            -Self::inverse_normal_positive(1.0 - p)
        } else {
            Self::inverse_normal_positive(p)
        }
    }

    fn inverse_normal_positive(p: f64) -> f64 {
        // Approximation constants
        let a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383_577_518_672_69e2,
            -3.066479806614716e1,
            2.506628277459239e0,
        ];
        let b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ];

        let q = p - 0.5;
        if q.abs() <= 0.425 {
            let r = 0.180625 - q * q;
            q * (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        } else {
            let r = if q < 0.0 { p } else { 1.0 - p };
            let r = (-r.ln()).sqrt();

            let c = [
                -7.784894002430293e-3,
                -3.223964580411365e-1,
                -2.400758277161838,
                -2.549732539343734,
                4.374664141464968,
                2.938163982698783,
            ];
            let d = [
                7.784695709041462e-3,
                3.224671290700398e-1,
                2.445134137142996,
                3.754408661907416,
            ];

            let result = (((((c[0] * r + c[1]) * r + c[2]) * r + c[3]) * r + c[4]) * r + c[5])
                / ((((d[0] * r + d[1]) * r + d[2]) * r + d[3]) * r + 1.0);

            if q < 0.0 { -result } else { result }
        }
    }

    /// Convert to feature tensor for neural network
    pub fn to_tensor(&self, batch_size: usize, time_steps: usize) -> SpikeTensor {
        let features = self.feature_vector.len();
        let mut data = Array3::<f32>::zeros((batch_size, time_steps, features));

        for b in 0..batch_size {
            for t in 0..time_steps {
                for (f, &val) in self.feature_vector.iter().enumerate() {
                    data[[b, t, f]] = val as f32;
                }
            }
        }

        SpikeTensor::from_dense(data, false)
    }
}

/// Motor profile from motor assessments
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotorProfile {
    /// Gait velocity (normalized 0-1, higher is better)
    pub gait_velocity_score: f64,
    /// Gait variability (CV, lower is better)
    pub gait_variability: f64,
    /// Balance score (0-1)
    pub balance_score: f64,
    /// Fine motor control score (0-1)
    pub fine_motor_score: f64,
    /// Grip strength (normalized 0-1)
    pub grip_strength_score: f64,
    /// Tremor severity (0-1, 0 = no tremor)
    pub tremor_severity: f64,
    /// Bradykinesia score (0-1, 0 = normal speed)
    pub bradykinesia_score: f64,
    /// Composite motor score
    pub composite_score: f64,
    /// Raw feature vector for neural network input
    pub feature_vector: Vec<f64>,
}

impl MotorProfile {
    /// Create a new motor profile from scores
    pub fn new(
        gait_velocity: f64,
        gait_variability: f64,
        balance: f64,
        fine_motor: f64,
        grip_strength: f64,
        tremor: f64,
        bradykinesia: f64,
    ) -> Self {
        // Compute composite (impairment scores negatively affect composite)
        let composite = 0.20 * gait_velocity
            + 0.15 * (1.0 - gait_variability.min(1.0))
            + 0.20 * balance
            + 0.15 * fine_motor
            + 0.10 * grip_strength
            + 0.10 * (1.0 - tremor)
            + 0.10 * (1.0 - bradykinesia);

        let feature_vector = vec![
            gait_velocity,
            gait_variability,
            balance,
            fine_motor,
            grip_strength,
            tremor,
            bradykinesia,
            composite,
        ];

        Self {
            gait_velocity_score: gait_velocity,
            gait_variability,
            balance_score: balance,
            fine_motor_score: fine_motor,
            grip_strength_score: grip_strength,
            tremor_severity: tremor,
            bradykinesia_score: bradykinesia,
            composite_score: composite,
            feature_vector,
        }
    }

    /// Create from gait analysis results
    pub fn from_gait(
        velocity_ms: f64,
        stride_time_cv: f64,
        double_support_percent: f64,
        cadence: f64,
    ) -> Self {
        // Normalize velocity (assume 0.8-1.4 m/s is normal)
        let gait_velocity = ((velocity_ms - 0.4) / 1.0).clamp(0.0, 1.0);

        // Gait variability
        let gait_variability = stride_time_cv.min(0.2) / 0.2;

        // Balance from double support (higher = worse balance)
        let balance = 1.0 - ((double_support_percent - 20.0) / 30.0).clamp(0.0, 1.0);

        // Use defaults for other metrics
        Self::new(
            gait_velocity,
            gait_variability,
            balance,
            0.5, // fine motor - unknown
            0.5, // grip strength - unknown
            0.0, // tremor - unknown
            0.0, // bradykinesia - unknown
        )
    }

    /// Convert to feature tensor for neural network
    pub fn to_tensor(&self, batch_size: usize, time_steps: usize) -> SpikeTensor {
        let features = self.feature_vector.len();
        let mut data = Array3::<f32>::zeros((batch_size, time_steps, features));

        for b in 0..batch_size {
            for t in 0..time_steps {
                for (f, &val) in self.feature_vector.iter().enumerate() {
                    data[[b, t, f]] = val as f32;
                }
            }
        }

        SpikeTensor::from_dense(data, false)
    }
}

/// Integrated assessment combining cognitive and motor profiles
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedAssessment {
    /// Cognitive profile
    pub cognitive: CognitiveProfile,
    /// Motor profile
    pub motor: MotorProfile,
    /// Overall composite score (0-1)
    pub composite_score: f64,
    /// Cognitive-motor coupling coefficient (-1 to 1)
    pub coupling_coefficient: f64,
    /// Risk category
    pub risk_category: RiskCategory,
    /// Domain-specific z-scores
    pub domain_z_scores: DomainZScores,
    /// Detected dissociation pattern (if any)
    pub dissociation: Option<DissociationPattern>,
}

/// Risk category based on integrated assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskCategory {
    /// Normal function
    Normal,
    /// Mild impairment in one or both domains
    MildImpairment,
    /// Moderate impairment
    ModerateImpairment,
    /// Severe impairment
    SevereImpairment,
    /// Cognitive-motor dissociation present
    Dissociated,
}

impl RiskCategory {
    /// Get numeric severity (0-4)
    pub fn severity(&self) -> u8 {
        match self {
            RiskCategory::Normal => 0,
            RiskCategory::MildImpairment => 1,
            RiskCategory::ModerateImpairment => 2,
            RiskCategory::SevereImpairment => 3,
            RiskCategory::Dissociated => 2, // Dissociation is moderate concern
        }
    }
}

/// Domain-specific z-scores for normative comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainZScores {
    pub cognitive_z: f64,
    pub motor_z: f64,
    pub attention_z: f64,
    pub memory_z: f64,
    pub gait_z: f64,
    pub balance_z: f64,
}

impl Default for DomainZScores {
    fn default() -> Self {
        Self {
            cognitive_z: 0.0,
            motor_z: 0.0,
            attention_z: 0.0,
            memory_z: 0.0,
            gait_z: 0.0,
            balance_z: 0.0,
        }
    }
}

/// Pattern of cognitive-motor dissociation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DissociationPattern {
    /// Type of dissociation
    pub pattern_type: DissociationType,
    /// Magnitude of dissociation (z-score difference)
    pub magnitude: f64,
    /// Confidence in detection (0-1)
    pub confidence: f64,
    /// Clinical interpretation
    pub interpretation: String,
}

/// Types of cognitive-motor dissociation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DissociationType {
    /// Motor preserved, cognitive impaired
    MotorPreservedCognitiveImpaired,
    /// Cognitive preserved, motor impaired
    CognitivePreservedMotorImpaired,
    /// Double dissociation (specific functions affected differently)
    DoubleDissociation,
    /// No significant dissociation
    None,
}

/// Metrics for tracking longitudinal change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeMetrics {
    /// Change in cognitive composite
    pub cognitive_change: f64,
    /// Change in motor composite
    pub motor_change: f64,
    /// Change in overall composite
    pub overall_change: f64,
    /// Reliable change index for cognitive
    pub cognitive_rci: f64,
    /// Reliable change index for motor
    pub motor_rci: f64,
    /// Whether cognitive change is clinically significant
    pub cognitive_significant: bool,
    /// Whether motor change is clinically significant
    pub motor_significant: bool,
    /// Direction of change
    pub change_direction: ChangeDirection,
}

/// Direction of longitudinal change
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeDirection {
    /// Improvement in both domains
    ImprovedBoth,
    /// Improvement in cognitive only
    ImprovedCognitive,
    /// Improvement in motor only
    ImprovedMotor,
    /// Stable in both domains
    Stable,
    /// Decline in cognitive only
    DeclinedCognitive,
    /// Decline in motor only
    DeclinedMotor,
    /// Decline in both domains
    DeclinedBoth,
    /// Mixed pattern
    Mixed,
}

/// Cognitive-motor fusion network
pub struct CognitiveMotorFusion {
    /// Configuration
    config: CognitiveMotorFusionConfig,
    /// Cognitive encoder layers
    cognitive_encoder: Vec<SpikingLinear>,
    /// Motor encoder layers
    motor_encoder: Vec<SpikingLinear>,
    /// Cross-modal attention layer
    cross_attention: SpikingLinear,
    /// Fusion layers
    fusion_layers: Vec<SpikingLinear>,
    /// Output layer
    output_layer: SpikingLinear,
    /// Neuron parameters
    neuron_params: NeuronParams,
}

/// Configuration for cognitive-motor fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveMotorFusionConfig {
    /// Cognitive input dimension
    pub cognitive_dim: usize,
    /// Motor input dimension
    pub motor_dim: usize,
    /// Hidden layer size
    pub hidden_size: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of fusion layers
    pub num_fusion_layers: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Weight for cognitive domain in fusion
    pub cognitive_weight: f64,
    /// Weight for motor domain in fusion
    pub motor_weight: f64,
    /// Standard error of measurement for RCI calculation
    pub sem_cognitive: f64,
    /// Standard error of measurement for motor
    pub sem_motor: f64,
}

impl Default for CognitiveMotorFusionConfig {
    fn default() -> Self {
        Self {
            cognitive_dim: 8,
            motor_dim: 8,
            hidden_size: 64,
            num_encoder_layers: 2,
            num_fusion_layers: 2,
            output_dim: 1,
            cognitive_weight: 0.5,
            motor_weight: 0.5,
            sem_cognitive: 0.1,
            sem_motor: 0.1,
        }
    }
}

impl CognitiveMotorFusion {
    /// Create a new cognitive-motor fusion network
    pub fn new(config: CognitiveMotorFusionConfig) -> SNNResult<Self> {
        let neuron_params = NeuronParams::default();
        let dt = 1.0;
        let adaptive = false;

        // Build cognitive encoder
        let mut cognitive_encoder = Vec::new();
        let mut prev_size = config.cognitive_dim;
        for _ in 0..config.num_encoder_layers {
            cognitive_encoder.push(SpikingLinear::new(
                prev_size,
                config.hidden_size,
                true,
                neuron_params.clone(),
                dt,
                adaptive,
            ));
            prev_size = config.hidden_size;
        }

        // Build motor encoder
        let mut motor_encoder = Vec::new();
        prev_size = config.motor_dim;
        for _ in 0..config.num_encoder_layers {
            motor_encoder.push(SpikingLinear::new(
                prev_size,
                config.hidden_size,
                true,
                neuron_params.clone(),
                dt,
                adaptive,
            ));
            prev_size = config.hidden_size;
        }

        // Cross-modal attention (simplified as linear projection)
        let cross_attention = SpikingLinear::new(
            config.hidden_size * 2,
            config.hidden_size,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        // Fusion layers
        let mut fusion_layers = Vec::new();
        prev_size = config.hidden_size;
        for i in 0..config.num_fusion_layers {
            let next_size = if i == config.num_fusion_layers - 1 {
                config.hidden_size / 2
            } else {
                config.hidden_size
            };
            fusion_layers.push(SpikingLinear::new(
                prev_size,
                next_size,
                true,
                neuron_params.clone(),
                dt,
                adaptive,
            ));
            prev_size = next_size;
        }

        // Output layer
        let output_layer = SpikingLinear::new(
            prev_size,
            config.output_dim,
            true,
            neuron_params.clone(),
            dt,
            adaptive,
        );

        Ok(Self {
            config,
            cognitive_encoder,
            motor_encoder,
            cross_attention,
            fusion_layers,
            output_layer,
            neuron_params,
        })
    }

    /// Fuse cognitive and motor profiles into integrated assessment
    pub fn fuse(
        &self,
        cognitive: &CognitiveProfile,
        motor: &MotorProfile,
    ) -> IntegratedAssessment {
        // Calculate weighted composite
        let composite = self.config.cognitive_weight * cognitive.composite_score
            + self.config.motor_weight * motor.composite_score;

        // Calculate coupling coefficient (correlation between domains)
        let coupling = self.calculate_coupling(cognitive, motor);

        // Detect dissociation
        let dissociation = self.detect_dissociation(cognitive, motor);

        // Determine risk category
        let risk_category = self.categorize_risk(cognitive, motor, &dissociation);

        // Calculate domain z-scores (placeholder - would use normative data)
        let domain_z_scores = DomainZScores {
            cognitive_z: (cognitive.composite_score - 0.5) / 0.15,
            motor_z: (motor.composite_score - 0.5) / 0.15,
            attention_z: (cognitive.attention_score - 0.5) / 0.15,
            memory_z: (cognitive.working_memory_score - 0.5) / 0.15,
            gait_z: (motor.gait_velocity_score - 0.5) / 0.15,
            balance_z: (motor.balance_score - 0.5) / 0.15,
        };

        IntegratedAssessment {
            cognitive: cognitive.clone(),
            motor: motor.clone(),
            composite_score: composite,
            coupling_coefficient: coupling,
            risk_category,
            domain_z_scores,
            dissociation,
        }
    }

    /// Calculate cognitive-motor coupling coefficient
    fn calculate_coupling(&self, cognitive: &CognitiveProfile, motor: &MotorProfile) -> f64 {
        // Simple correlation proxy based on composite scores
        // In practice, would use correlation across multiple assessments
        let cog_norm = (cognitive.composite_score - 0.5) * 2.0;
        let mot_norm = (motor.composite_score - 0.5) * 2.0;

        // Coupling is high when both move together
        1.0 - (cog_norm - mot_norm).abs()
    }

    /// Detect cognitive-motor dissociation
    pub fn detect_dissociation(
        &self,
        cognitive: &CognitiveProfile,
        motor: &MotorProfile,
    ) -> Option<DissociationPattern> {
        // Convert to z-scores (assuming mean=0.5, sd=0.15 for normalized scores)
        let cog_z = (cognitive.composite_score - 0.5) / 0.15;
        let mot_z = (motor.composite_score - 0.5) / 0.15;

        let diff = (cog_z - mot_z).abs();

        // Dissociation threshold: 1.5 SD difference
        if diff < 1.5 {
            return None;
        }

        let (pattern_type, interpretation) = if cog_z < mot_z - 1.5 {
            (
                DissociationType::MotorPreservedCognitiveImpaired,
                "Motor function is relatively preserved while cognitive function shows impairment. \
                 This pattern may be seen in early Alzheimer's disease or other primarily cognitive conditions."
                    .to_string(),
            )
        } else if mot_z < cog_z - 1.5 {
            (
                DissociationType::CognitivePreservedMotorImpaired,
                "Cognitive function is relatively preserved while motor function shows impairment. \
                 This pattern is common in Parkinson's disease and other movement disorders."
                    .to_string(),
            )
        } else {
            return None;
        };

        // Confidence based on magnitude
        let confidence = (diff / 3.0).min(1.0);

        Some(DissociationPattern {
            pattern_type,
            magnitude: diff,
            confidence,
            interpretation,
        })
    }

    /// Categorize risk based on profiles
    fn categorize_risk(
        &self,
        cognitive: &CognitiveProfile,
        motor: &MotorProfile,
        dissociation: &Option<DissociationPattern>,
    ) -> RiskCategory {
        if dissociation.is_some() {
            return RiskCategory::Dissociated;
        }

        let cog_impaired = cognitive.composite_score < 0.35;
        let mot_impaired = motor.composite_score < 0.35;
        let cog_mild = cognitive.composite_score < 0.5;
        let mot_mild = motor.composite_score < 0.5;

        if cog_impaired && mot_impaired {
            RiskCategory::SevereImpairment
        } else if cog_impaired || mot_impaired {
            RiskCategory::ModerateImpairment
        } else if cog_mild || mot_mild {
            RiskCategory::MildImpairment
        } else {
            RiskCategory::Normal
        }
    }

    /// Detect longitudinal change between assessments
    pub fn detect_change(
        &self,
        baseline: &IntegratedAssessment,
        followup: &IntegratedAssessment,
    ) -> ChangeMetrics {
        let cognitive_change = followup.cognitive.composite_score - baseline.cognitive.composite_score;
        let motor_change = followup.motor.composite_score - baseline.motor.composite_score;
        let overall_change = followup.composite_score - baseline.composite_score;

        // Reliable Change Index: change / (SEM * sqrt(2))
        let se_diff_cog = self.config.sem_cognitive * std::f64::consts::SQRT_2;
        let se_diff_mot = self.config.sem_motor * std::f64::consts::SQRT_2;

        let cognitive_rci = cognitive_change / se_diff_cog;
        let motor_rci = motor_change / se_diff_mot;

        // Significant if |RCI| > 1.96 (95% CI)
        let cognitive_significant = cognitive_rci.abs() > 1.96;
        let motor_significant = motor_rci.abs() > 1.96;

        let change_direction = match (
            cognitive_significant,
            motor_significant,
            cognitive_change > 0.0,
            motor_change > 0.0,
        ) {
            (true, true, true, true) => ChangeDirection::ImprovedBoth,
            (true, true, false, false) => ChangeDirection::DeclinedBoth,
            (true, false, true, _) => ChangeDirection::ImprovedCognitive,
            (true, false, false, _) => ChangeDirection::DeclinedCognitive,
            (false, true, _, true) => ChangeDirection::ImprovedMotor,
            (false, true, _, false) => ChangeDirection::DeclinedMotor,
            (false, false, _, _) => ChangeDirection::Stable,
            _ => ChangeDirection::Mixed,
        };

        ChangeMetrics {
            cognitive_change,
            motor_change,
            overall_change,
            cognitive_rci,
            motor_rci,
            cognitive_significant,
            motor_significant,
            change_direction,
        }
    }

    /// Forward pass through fusion network
    pub fn forward_profiles(
        &mut self,
        cognitive: &CognitiveProfile,
        motor: &MotorProfile,
        time_steps: usize,
    ) -> SNNResult<SpikeTensor> {
        let batch_size = 1;

        // Convert profiles to tensors
        let cog_tensor = cognitive.to_tensor(batch_size, time_steps);
        let mot_tensor = motor.to_tensor(batch_size, time_steps);

        // Encode cognitive
        let mut cog_encoded = cog_tensor;
        for layer in &mut self.cognitive_encoder {
            cog_encoded = layer.forward(&cog_encoded)?;
        }

        // Encode motor
        let mut mot_encoded = mot_tensor;
        for layer in &mut self.motor_encoder {
            mot_encoded = layer.forward(&mot_encoded)?;
        }

        // Concatenate for cross-attention
        let cog_dense = cog_encoded.to_dense();
        let mot_dense = mot_encoded.to_dense();

        let (batch, time, cog_feat) = cog_dense.dim();
        let mot_feat = mot_dense.dim().2;

        let mut concat = Array3::<f32>::zeros((batch, time, cog_feat + mot_feat));
        concat.slice_mut(ndarray::s![.., .., 0..cog_feat]).assign(&cog_dense);
        concat.slice_mut(ndarray::s![.., .., cog_feat..]).assign(&mot_dense);

        let concat_tensor = SpikeTensor::from_dense(concat, false);

        // Cross-modal attention
        let mut fused = self.cross_attention.forward(&concat_tensor)?;

        // Fusion layers
        for layer in &mut self.fusion_layers {
            fused = layer.forward(&fused)?;
        }

        // Output
        self.output_layer.forward(&fused)
    }

    /// Get configuration
    pub fn config(&self) -> &CognitiveMotorFusionConfig {
        &self.config
    }

    /// Reset network state
    pub fn reset_state(&mut self) {
        for layer in &mut self.cognitive_encoder {
            layer.reset_state();
        }
        for layer in &mut self.motor_encoder {
            layer.reset_state();
        }
        self.cross_attention.reset_state();
        for layer in &mut self.fusion_layers {
            layer.reset_state();
        }
        self.output_layer.reset_state();
    }

    /// Get total parameter count
    pub fn num_parameters(&self) -> usize {
        let mut total = 0;

        for layer in &self.cognitive_encoder {
            total += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }
        for layer in &self.motor_encoder {
            total += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }
        total += self.cross_attention.parameters().iter().map(|p| p.len()).sum::<usize>();
        for layer in &self.fusion_layers {
            total += layer.parameters().iter().map(|p| p.len()).sum::<usize>();
        }
        total += self.output_layer.parameters().iter().map(|p| p.len()).sum::<usize>();

        total
    }
}

/// Wrapper to implement FusionNetwork trait
pub struct CognitiveMotorFusionSNN {
    inner: CognitiveMotorFusion,
    config: FusionConfig,
}

impl CognitiveMotorFusionSNN {
    /// Create new fusion SNN
    pub fn new(config: FusionConfig) -> SNNResult<Self> {
        let inner_config = CognitiveMotorFusionConfig {
            cognitive_dim: 8,
            motor_dim: 8,
            hidden_size: config.hidden_size,
            num_encoder_layers: config.num_layers,
            num_fusion_layers: 2,
            output_dim: config.output_size,
            ..Default::default()
        };

        Ok(Self {
            inner: CognitiveMotorFusion::new(inner_config)?,
            config,
        })
    }

    /// Get inner fusion module
    pub fn inner(&self) -> &CognitiveMotorFusion {
        &self.inner
    }

    /// Get mutable inner fusion module
    pub fn inner_mut(&mut self) -> &mut CognitiveMotorFusion {
        &mut self.inner
    }
}

impl FusionNetwork for CognitiveMotorFusionSNN {
    fn name(&self) -> &str {
        "CognitiveMotorFusionSNN"
    }

    fn modalities(&self) -> Vec<Modality> {
        // Cognitive-motor fusion uses Contact (for motor) and Eye (for cognitive) as proxies
        vec![Modality::Contact, Modality::Eye, Modality::Pose]
    }

    fn forward(&mut self, inputs: &HashMap<Modality, SpikeTensor>) -> SNNResult<SpikeTensor> {
        // Extract cognitive and motor inputs
        let cognitive_input = inputs.get(&Modality::Eye)
            .or_else(|| inputs.get(&Modality::Contact))
            .ok_or_else(|| SNNError::InvalidConfig("Missing cognitive/motor input".to_string()))?;

        let motor_input = inputs.get(&Modality::Pose)
            .or_else(|| inputs.get(&Modality::Contact))
            .ok_or_else(|| SNNError::InvalidConfig("Missing motor input".to_string()))?;

        // Forward through encoders
        let mut cog_encoded = cognitive_input.clone();
        for layer in &mut self.inner.cognitive_encoder {
            cog_encoded = layer.forward(&cog_encoded)?;
        }

        let mut mot_encoded = motor_input.clone();
        for layer in &mut self.inner.motor_encoder {
            mot_encoded = layer.forward(&mot_encoded)?;
        }

        // Concatenate
        let cog_dense = cog_encoded.to_dense();
        let mot_dense = mot_encoded.to_dense();

        let (batch, time, cog_feat) = cog_dense.dim();
        let mot_feat = mot_dense.dim().2;

        let mut concat = Array3::<f32>::zeros((batch, time, cog_feat + mot_feat));
        concat.slice_mut(ndarray::s![.., .., 0..cog_feat]).assign(&cog_dense);
        concat.slice_mut(ndarray::s![.., .., cog_feat..]).assign(&mot_dense);

        let concat_tensor = SpikeTensor::from_dense(concat, false);

        // Cross-attention and fusion
        let mut fused = self.inner.cross_attention.forward(&concat_tensor)?;
        for layer in &mut self.inner.fusion_layers {
            fused = layer.forward(&fused)?;
        }

        self.inner.output_layer.forward(&fused)
    }

    fn num_parameters(&self) -> usize {
        self.inner.num_parameters()
    }

    fn reset_state(&mut self) {
        self.inner.reset_state();
    }

    fn handles_missing_modalities(&self) -> bool {
        false // Requires both cognitive and motor inputs
    }

    fn required_modalities(&self) -> Vec<Modality> {
        vec![Modality::Contact, Modality::Pose]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_profile_creation() {
        let profile = CognitiveProfile::new(0.8, 0.7, 0.75, 0.85, 0.15, 2.5, 0.1);

        assert!(profile.composite_score > 0.0);
        assert!(profile.composite_score <= 1.0);
        assert_eq!(profile.feature_vector.len(), 8);
    }

    #[test]
    fn test_motor_profile_creation() {
        let profile = MotorProfile::new(0.9, 0.08, 0.85, 0.7, 0.8, 0.1, 0.2);

        assert!(profile.composite_score > 0.0);
        assert!(profile.composite_score <= 1.0);
        assert_eq!(profile.feature_vector.len(), 8);
    }

    #[test]
    fn test_cognitive_motor_fusion() {
        let config = CognitiveMotorFusionConfig::default();
        let fusion = CognitiveMotorFusion::new(config).unwrap();

        let cognitive = CognitiveProfile::new(0.8, 0.7, 0.75, 0.85, 0.15, 2.5, 0.1);
        let motor = MotorProfile::new(0.9, 0.08, 0.85, 0.7, 0.8, 0.1, 0.2);

        let assessment = fusion.fuse(&cognitive, &motor);

        assert!(assessment.composite_score > 0.0);
        assert_eq!(assessment.risk_category, RiskCategory::Normal);
        assert!(assessment.dissociation.is_none());
    }

    #[test]
    fn test_dissociation_detection() {
        let config = CognitiveMotorFusionConfig::default();
        let fusion = CognitiveMotorFusion::new(config).unwrap();

        // Strong dissociation: good motor, poor cognitive
        let cognitive = CognitiveProfile::new(0.2, 0.25, 0.2, 0.3, 0.4, 1.0, 0.3);
        let motor = MotorProfile::new(0.9, 0.05, 0.95, 0.85, 0.9, 0.0, 0.0);

        let dissociation = fusion.detect_dissociation(&cognitive, &motor);

        assert!(dissociation.is_some());
        let d = dissociation.unwrap();
        assert_eq!(d.pattern_type, DissociationType::MotorPreservedCognitiveImpaired);
        assert!(d.magnitude > 1.5);
    }

    #[test]
    fn test_change_detection() {
        let config = CognitiveMotorFusionConfig::default();
        let fusion = CognitiveMotorFusion::new(config).unwrap();

        let cog1 = CognitiveProfile::new(0.5, 0.5, 0.5, 0.5, 0.2, 2.0, 0.0);
        let mot1 = MotorProfile::new(0.5, 0.1, 0.5, 0.5, 0.5, 0.2, 0.2);
        let baseline = fusion.fuse(&cog1, &mot1);

        // Improvement in cognitive, decline in motor
        let cog2 = CognitiveProfile::new(0.8, 0.8, 0.8, 0.8, 0.1, 3.0, 0.0);
        let mot2 = MotorProfile::new(0.3, 0.2, 0.3, 0.3, 0.3, 0.4, 0.4);
        let followup = fusion.fuse(&cog2, &mot2);

        let change = fusion.detect_change(&baseline, &followup);

        assert!(change.cognitive_change > 0.0);
        assert!(change.motor_change < 0.0);
    }

    #[test]
    fn test_risk_categorization() {
        let config = CognitiveMotorFusionConfig::default();
        let fusion = CognitiveMotorFusion::new(config).unwrap();

        // Normal
        let cog_normal = CognitiveProfile::new(0.8, 0.8, 0.8, 0.8, 0.1, 3.0, 0.0);
        let mot_normal = MotorProfile::new(0.9, 0.05, 0.9, 0.85, 0.9, 0.0, 0.0);
        let normal = fusion.fuse(&cog_normal, &mot_normal);
        assert_eq!(normal.risk_category, RiskCategory::Normal);

        // Mild
        let cog_mild = CognitiveProfile::new(0.45, 0.45, 0.45, 0.45, 0.25, 1.5, 0.1);
        let mot_mild = MotorProfile::new(0.45, 0.12, 0.45, 0.45, 0.45, 0.15, 0.15);
        let mild = fusion.fuse(&cog_mild, &mot_mild);
        assert_eq!(mild.risk_category, RiskCategory::MildImpairment);

        // Severe
        let cog_severe = CognitiveProfile::new(0.2, 0.2, 0.2, 0.2, 0.5, 0.5, 0.3);
        let mot_severe = MotorProfile::new(0.2, 0.3, 0.2, 0.2, 0.2, 0.5, 0.5);
        let severe = fusion.fuse(&cog_severe, &mot_severe);
        assert_eq!(severe.risk_category, RiskCategory::SevereImpairment);
    }

    #[test]
    fn test_profile_from_reaction_time() {
        let profile = CognitiveProfile::from_reaction_time(
            350.0, // mean RT
            50.0,  // std RT
            0.95,  // accuracy
            2,     // anticipations
            3,     // lapses
            100,   // total trials
        );

        assert!(profile.processing_speed_score > 0.5);
        assert!(profile.attention_score > 0.9);
        // D-prime should be finite (sign depends on hit/fa rates)
        assert!(profile.d_prime.is_finite(), "d_prime should be finite, got {}", profile.d_prime);
    }

    #[test]
    fn test_fusion_network_forward() {
        let mut fusion = CognitiveMotorFusion::new(CognitiveMotorFusionConfig::default()).unwrap();

        let cognitive = CognitiveProfile::new(0.8, 0.7, 0.75, 0.85, 0.15, 2.5, 0.1);
        let motor = MotorProfile::new(0.9, 0.08, 0.85, 0.7, 0.8, 0.1, 0.2);

        let output = fusion.forward_profiles(&cognitive, &motor, 10);
        assert!(output.is_ok());

        let tensor = output.unwrap();
        let (batch, time, features) = tensor.shape();
        assert_eq!(batch, 1);
        assert_eq!(time, 10);
    }

    #[test]
    fn test_cognitive_motor_fusion_snn() {
        let config = FusionConfig {
            modalities: vec![Modality::Contact, Modality::Pose],
            hidden_size: 64,
            num_layers: 2,
            output_size: 1,
            ..Default::default()
        };

        let mut network = CognitiveMotorFusionSNN::new(config).unwrap();
        assert_eq!(network.name(), "CognitiveMotorFusionSNN");
        assert!(!network.handles_missing_modalities());

        // Create input tensors
        let mut inputs = HashMap::new();
        inputs.insert(
            Modality::Contact,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 8)), false),
        );
        inputs.insert(
            Modality::Pose,
            SpikeTensor::from_dense(Array3::<f32>::zeros((1, 10, 8)), false),
        );

        let output = network.forward(&inputs);
        assert!(output.is_ok());
    }
}
