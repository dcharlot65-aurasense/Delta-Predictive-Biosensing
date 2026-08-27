//! SNN-Specific Knowledge Distillation
//!
//! This module implements distillation techniques specific to Spiking Neural
//! Networks, including temporal spike pattern matching, firing rate alignment,
//! and membrane potential distillation.

use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{s, Array1, Array2, Array3};
use serde::{Deserialize, Serialize};

/// Configuration for spike-based distillation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeDistillationConfig {
    /// Match spike patterns (temporal structure)
    pub match_spike_patterns: bool,
    /// Match firing rates (average activity)
    pub match_firing_rates: bool,
    /// Match membrane potentials
    pub match_membrane_potentials: bool,
    /// Temporal alignment tolerance (time steps)
    pub temporal_alignment: bool,
    /// Weights for different components
    pub pattern_weight: f32,
    pub rate_weight: f32,
    pub membrane_weight: f32,
}

impl Default for SpikeDistillationConfig {
    fn default() -> Self {
        Self {
            match_spike_patterns: true,
            match_firing_rates: true,
            match_membrane_potentials: false,
            temporal_alignment: true,
            pattern_weight: 0.4,
            rate_weight: 0.4,
            membrane_weight: 0.2,
        }
    }
}

impl SpikeDistillationConfig {
    /// Validate configuration
    pub fn validate(&self) -> SNNResult<()> {
        let total_weight = self.pattern_weight + self.rate_weight + self.membrane_weight;
        if (total_weight - 1.0).abs() > 1e-5 {
            return Err(SNNError::InvalidConfig(
                "Distillation weights should sum to 1.0".to_string()
            ));
        }
        Ok(())
    }
}

/// Spike pattern distillation (temporal structure matching)
#[derive(Debug, Clone)]
pub struct SpikePatternDistillation {
    /// Configuration
    pub config: SpikeDistillationConfig,
    /// Window size for pattern matching
    pub window_size: usize,
    /// Temporal tolerance for alignment
    pub tolerance: usize,
}

// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
impl SpikePatternDistillation {
    pub fn new(config: SpikeDistillationConfig) -> Self {
        Self {
            config,
            window_size: 10,
            tolerance: 2,
        }
    }

    /// Compute loss between teacher and student spike patterns
    pub fn compute_loss(
        &self,
        teacher_spikes: &SpikeTensor,
        student_spikes: &SpikeTensor,
    ) -> SNNResult<f32> {
        let teacher_dense = teacher_spikes.to_dense();
        let student_dense = student_spikes.to_dense();

        if teacher_dense.shape() != student_dense.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", teacher_dense.shape()),
                actual: format!("{:?}", student_dense.shape()),
            });
        }

        let (batch_size, _num_steps, num_neurons) = (
            teacher_dense.shape()[0],
            teacher_dense.shape()[1],
            teacher_dense.shape()[2],
        );

        let mut total_loss = 0.0;

        // Compute pattern matching loss for each neuron
        for b in 0..batch_size {
            for n in 0..num_neurons {
                let teacher_pattern = teacher_dense.slice(s![b, .., n]);
                let student_pattern = student_dense.slice(s![b, .., n]);

                // Compute temporal cross-correlation
                let pattern_loss = self.pattern_distance(&teacher_pattern, &student_pattern);
                total_loss += pattern_loss;
            }
        }

        Ok(total_loss / (batch_size * num_neurons) as f32)
    }

    /// Compute distance between two spike patterns
    fn pattern_distance(&self, teacher: &ndarray::ArrayView1<f32>, student: &ndarray::ArrayView1<f32>) -> f32 {
        let len = teacher.len();
        let mut min_distance = f32::MAX;

        // Try different time shifts within tolerance
        for shift in 0..=self.tolerance {
            let mut distance = 0.0;

            for i in 0..(len - shift) {
                let diff = teacher[i] - student[i + shift];
                distance += diff * diff;
            }

            min_distance = min_distance.min(distance);

            // Try negative shift
            if shift > 0 {
                let mut distance = 0.0;
                for i in shift..len {
                    let diff = teacher[i] - student[i - shift];
                    distance += diff * diff;
                }
                min_distance = min_distance.min(distance);
            }
        }

        min_distance / len as f32
    }

    /// Extract spike intervals for a pattern
    fn extract_spike_intervals(&self, pattern: &ndarray::ArrayView1<f32>) -> Vec<usize> {
        let mut intervals = Vec::new();
        let mut last_spike_time = None;

        for (t, &spike) in pattern.iter().enumerate() {
            if spike > 0.5 {
                if let Some(last_t) = last_spike_time {
                    intervals.push(t - last_t);
                }
                last_spike_time = Some(t);
            }
        }

        intervals
    }
}

/// Spike rate distillation (firing rate matching)
#[derive(Debug, Clone)]
pub struct SpikeRateDistillation {
    /// Temporal window for rate computation
    pub window_size: usize,
    /// Smoothing kernel
    pub smoothing: bool,
}

impl SpikeRateDistillation {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            smoothing: true,
        }
    }

    /// Compute firing rate loss
    pub fn compute_loss(
        &self,
        teacher_spikes: &SpikeTensor,
        student_spikes: &SpikeTensor,
    ) -> SNNResult<f32> {
        let teacher_rates = teacher_spikes.spike_rate();
        let student_rates = student_spikes.spike_rate();

        if teacher_rates.shape() != student_rates.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", teacher_rates.shape()),
                actual: format!("{:?}", student_rates.shape()),
            });
        }

        // MSE between firing rates
        let diff = &teacher_rates - &student_rates;
        let mse = diff.mapv(|x| x * x).sum() / teacher_rates.len() as f32;

        Ok(mse)
    }

    /// Compute instantaneous rates with temporal windowing
    pub fn compute_windowed_rates(
        &self,
        spikes: &SpikeTensor,
    ) -> SNNResult<Array3<f32>> {
        let dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            dense.shape()[0],
            dense.shape()[1],
            dense.shape()[2],
        );

        let num_windows = num_steps.div_ceil(self.window_size);
        let mut rates = Array3::zeros((batch_size, num_windows, num_neurons));

        for b in 0..batch_size {
            for w in 0..num_windows {
                let start = w * self.window_size;
                let end = ((w + 1) * self.window_size).min(num_steps);

                for n in 0..num_neurons {
                    let window_spikes: f32 = dense.slice(s![b, start..end, n]).sum();
                    rates[[b, w, n]] = window_spikes / (end - start) as f32;
                }
            }
        }

        Ok(rates)
    }
}

/// Membrane potential distillation
#[derive(Debug, Clone)]
pub struct MembranePotentialDistillation {
    /// Weight for sub-threshold dynamics
    pub subthreshold_weight: f32,
    /// Weight for threshold proximity
    pub threshold_weight: f32,
}

impl MembranePotentialDistillation {
    pub fn new(subthreshold_weight: f32, threshold_weight: f32) -> Self {
        Self {
            subthreshold_weight,
            threshold_weight,
        }
    }

    /// Compute membrane potential loss
    pub fn compute_loss(
        &self,
        teacher_potentials: &Array3<f32>,
        student_potentials: &Array3<f32>,
    ) -> SNNResult<f32> {
        if teacher_potentials.shape() != student_potentials.shape() {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{:?}", teacher_potentials.shape()),
                actual: format!("{:?}", student_potentials.shape()),
            });
        }

        // MSE between membrane potentials
        let diff = teacher_potentials - student_potentials;
        let mse = diff.mapv(|x| x * x).sum() / teacher_potentials.len() as f32;

        Ok(mse)
    }

    /// Compute loss focusing on near-threshold dynamics
    pub fn compute_threshold_proximity_loss(
        &self,
        teacher_potentials: &Array3<f32>,
        student_potentials: &Array3<f32>,
        threshold: f32,
    ) -> SNNResult<f32> {
        let (batch_size, num_steps, num_neurons) = (
            teacher_potentials.shape()[0],
            teacher_potentials.shape()[1],
            teacher_potentials.shape()[2],
        );

        let mut total_loss = 0.0;
        let mut count = 0;

        for b in 0..batch_size {
            for t in 0..num_steps {
                for n in 0..num_neurons {
                    let teacher_v = teacher_potentials[[b, t, n]];
                    let student_v = student_potentials[[b, t, n]];

                    // Only consider potentials near threshold
                    if teacher_v > 0.7 * threshold || student_v > 0.7 * threshold {
                        let diff = teacher_v - student_v;
                        total_loss += diff * diff;
                        count += 1;
                    }
                }
            }
        }

        if count > 0 {
            Ok(total_loss / count as f32)
        } else {
            Ok(0.0)
        }
    }
}

/// Synaptic weight transfer
#[derive(Debug, Clone)]
pub struct SynapticWeightTransfer {
    /// Transfer mode
    pub mode: WeightTransferMode,
    /// Scale factor for initialization
    pub scale_factor: f32,
}

/// Weight transfer modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightTransferMode {
    /// Direct copy
    Direct,
    /// Scaled copy
    Scaled,
    /// Subset transfer (for different architectures)
    Subset,
    /// Initialize with noise around teacher weights
    NoisyInitialization,
}

impl SynapticWeightTransfer {
    pub fn new(mode: WeightTransferMode) -> Self {
        Self {
            mode,
            scale_factor: 1.0,
        }
    }

    /// Transfer weights from teacher to student
    pub fn transfer_weights(
        &self,
        teacher_weights: &Array2<f32>,
        student_weights: &mut Array2<f32>,
    ) -> SNNResult<()> {
        let (teacher_rows, teacher_cols) = (teacher_weights.shape()[0], teacher_weights.shape()[1]);
        let (student_rows, student_cols) = (student_weights.shape()[0], student_weights.shape()[1]);

        match self.mode {
            WeightTransferMode::Direct => {
                if teacher_rows != student_rows || teacher_cols != student_cols {
                    return Err(SNNError::DimensionMismatch {
                        expected: format!("({}, {})", teacher_rows, teacher_cols),
                        actual: format!("({}, {})", student_rows, student_cols),
                    });
                }
                student_weights.assign(teacher_weights);
            }
            WeightTransferMode::Scaled => {
                if teacher_rows != student_rows || teacher_cols != student_cols {
                    return Err(SNNError::DimensionMismatch {
                        expected: format!("({}, {})", teacher_rows, teacher_cols),
                        actual: format!("({}, {})", student_rows, student_cols),
                    });
                }
                student_weights.assign(&(teacher_weights * self.scale_factor));
            }
            WeightTransferMode::Subset => {
                // Transfer a subset of weights (e.g., for smaller student)
                let rows_to_copy = teacher_rows.min(student_rows);
                let cols_to_copy = teacher_cols.min(student_cols);

                for i in 0..rows_to_copy {
                    for j in 0..cols_to_copy {
                        student_weights[[i, j]] = teacher_weights[[i, j]];
                    }
                }
            }
            WeightTransferMode::NoisyInitialization => {
                // Initialize with noise (not a true transfer, handled elsewhere)
                return Ok(());
            }
        }

        Ok(())
    }
}

/// Temporal credit assignment for distillation
#[derive(Debug, Clone)]
pub struct TemporalCreditAssignment {
    /// Decay factor for temporal importance
    pub decay_factor: f32,
    /// Window for credit assignment
    pub window_size: usize,
}

impl TemporalCreditAssignment {
    pub fn new(decay_factor: f32, window_size: usize) -> Self {
        Self {
            decay_factor,
            window_size,
        }
    }

    /// Compute temporal importance weights
    pub fn compute_temporal_weights(&self, num_steps: usize) -> Array1<f32> {
        let mut weights = Array1::zeros(num_steps);

        for t in 0..num_steps {
            // Exponential decay from present to past
            weights[t] = (self.decay_factor * (num_steps - t - 1) as f32).exp();
        }

        // Normalize
        let sum = weights.sum();
        if sum > 1e-8 {
            weights /= sum;
        }

        weights
    }

    /// Compute weighted loss over time
    pub fn compute_weighted_loss(
        &self,
        teacher_spikes: &SpikeTensor,
        student_spikes: &SpikeTensor,
    ) -> SNNResult<f32> {
        let teacher_dense = teacher_spikes.to_dense();
        let student_dense = student_spikes.to_dense();

        let num_steps = teacher_dense.shape()[1];
        let temporal_weights = self.compute_temporal_weights(num_steps);

        let mut total_loss = 0.0;
        let (batch_size, _, num_neurons) = (
            teacher_dense.shape()[0],
            teacher_dense.shape()[1],
            teacher_dense.shape()[2],
        );

        for b in 0..batch_size {
            for t in 0..num_steps {
                for n in 0..num_neurons {
                    let diff = teacher_dense[[b, t, n]] - student_dense[[b, t, n]];
                    total_loss += temporal_weights[t] * diff * diff;
                }
            }
        }

        Ok(total_loss / (batch_size * num_neurons) as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_spike_distillation_config() {
        let config = SpikeDistillationConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = config.clone();
        invalid_config.pattern_weight = 0.5;
        invalid_config.rate_weight = 0.3;
        invalid_config.membrane_weight = 0.3; // Sum > 1.0
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_spike_pattern_distillation() {
        let config = SpikeDistillationConfig::default();
        let distiller = SpikePatternDistillation::new(config);

        let mut teacher_data = Array3::zeros((1, 20, 2));
        teacher_data[[0, 5, 0]] = 1.0;
        teacher_data[[0, 10, 0]] = 1.0;
        teacher_data[[0, 15, 0]] = 1.0;

        let mut student_data = Array3::zeros((1, 20, 2));
        student_data[[0, 6, 0]] = 1.0;  // Slightly shifted
        student_data[[0, 11, 0]] = 1.0;
        student_data[[0, 16, 0]] = 1.0;

        let teacher_spikes = SpikeTensor::from_dense(teacher_data, false);
        let student_spikes = SpikeTensor::from_dense(student_data, false);

        let loss = distiller.compute_loss(&teacher_spikes, &student_spikes).unwrap();
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_spike_rate_distillation() {
        let distiller = SpikeRateDistillation::new(5);

        let mut teacher_data = Array3::zeros((1, 10, 2));
        teacher_data[[0, 2, 0]] = 1.0;
        teacher_data[[0, 5, 0]] = 1.0;

        let mut student_data = Array3::zeros((1, 10, 2));
        student_data[[0, 3, 0]] = 1.0;
        student_data[[0, 6, 0]] = 1.0;

        let teacher_spikes = SpikeTensor::from_dense(teacher_data, false);
        let student_spikes = SpikeTensor::from_dense(student_data, false);

        let loss = distiller.compute_loss(&teacher_spikes, &student_spikes).unwrap();
        assert!(loss >= 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_windowed_rates() {
        let distiller = SpikeRateDistillation::new(5);

        let mut data = Array3::zeros((1, 10, 2));
        data[[0, 2, 0]] = 1.0;
        data[[0, 7, 0]] = 1.0;

        let spikes = SpikeTensor::from_dense(data, false);
        let rates = distiller.compute_windowed_rates(&spikes).unwrap();

        assert_eq!(rates.shape(), &[1, 2, 2]); // 2 windows of size 5
        assert!(rates[[0, 0, 0]] > 0.0); // First window has a spike
        assert!(rates[[0, 1, 0]] > 0.0); // Second window has a spike
    }

    #[test]
    fn test_membrane_potential_distillation() {
        let distiller = MembranePotentialDistillation::new(0.5, 0.5);

        let teacher = Array3::from_shape_vec((1, 10, 2), vec![0.5; 20]).unwrap();
        let student = Array3::from_shape_vec((1, 10, 2), vec![0.4; 20]).unwrap();

        let loss = distiller.compute_loss(&teacher, &student).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_threshold_proximity_loss() {
        let distiller = MembranePotentialDistillation::new(0.5, 0.5);

        let mut teacher = Array3::zeros((1, 10, 2));
        teacher[[0, 5, 0]] = 0.9; // Near threshold

        let mut student = Array3::zeros((1, 10, 2));
        student[[0, 5, 0]] = 0.85;

        let loss = distiller.compute_threshold_proximity_loss(&teacher, &student, 1.0).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_weight_transfer_direct() {
        let transfer = SynapticWeightTransfer::new(WeightTransferMode::Direct);

        let teacher_weights = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();
        let mut student_weights = Array2::zeros((3, 4));

        transfer.transfer_weights(&teacher_weights, &mut student_weights).unwrap();

        assert_eq!(student_weights, teacher_weights);
    }

    #[test]
    fn test_weight_transfer_scaled() {
        let mut transfer = SynapticWeightTransfer::new(WeightTransferMode::Scaled);
        transfer.scale_factor = 0.5;

        let teacher_weights = Array2::from_shape_vec((3, 4), vec![2.0; 12]).unwrap();
        let mut student_weights = Array2::zeros((3, 4));

        transfer.transfer_weights(&teacher_weights, &mut student_weights).unwrap();

        for &val in student_weights.iter() {
            assert_eq!(val, 1.0); // 2.0 * 0.5
        }
    }

    #[test]
    fn test_weight_transfer_subset() {
        let transfer = SynapticWeightTransfer::new(WeightTransferMode::Subset);

        let teacher_weights = Array2::from_shape_vec((4, 5), vec![1.0; 20]).unwrap();
        let mut student_weights = Array2::zeros((2, 3));

        transfer.transfer_weights(&teacher_weights, &mut student_weights).unwrap();

        // Check that subset is transferred
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(student_weights[[i, j]], 1.0);
            }
        }
    }

    #[test]
    fn test_temporal_credit_assignment() {
        let tca = TemporalCreditAssignment::new(-0.1, 10);

        let weights = tca.compute_temporal_weights(10);
        assert_eq!(weights.len(), 10);

        // Check normalization
        let sum: f32 = weights.sum();
        assert!((sum - 1.0).abs() < 1e-5);

        // Check that recent timesteps have higher weight
        assert!(weights[9] > weights[0]);
    }

    #[test]
    fn test_weighted_temporal_loss() {
        let tca = TemporalCreditAssignment::new(-0.1, 10);

        let teacher_data = Array3::from_shape_vec((1, 10, 2), vec![0.5; 20]).unwrap();
        let student_data = Array3::from_shape_vec((1, 10, 2), vec![0.4; 20]).unwrap();

        let teacher_spikes = SpikeTensor::from_dense(teacher_data, false);
        let student_spikes = SpikeTensor::from_dense(student_data, false);

        let loss = tca.compute_weighted_loss(&teacher_spikes, &student_spikes).unwrap();
        assert!(loss > 0.0);
        assert!(loss.is_finite());
    }
}
