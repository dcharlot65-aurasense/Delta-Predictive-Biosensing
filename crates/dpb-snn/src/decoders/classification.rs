//! Classification decoders for categorical outputs

use super::Decoder;
use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array2, s};
use serde::{Deserialize, Serialize};

/// Binary classification decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryClassDecoder {
    pub num_neurons: usize,
    pub threshold: f32,
}

impl BinaryClassDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Decoder for BinaryClassDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 2)); // [class_0, class_1]

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);

            if mean_rate > self.threshold {
                output[[b, 1]] = 1.0; // Class 1
            } else {
                output[[b, 0]] = 1.0; // Class 0
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        2
    }
}

/// Multi-class classification decoder with softmax
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiClassDecoder {
    pub num_classes: usize,
    pub neurons_per_class: usize,
}

impl MultiClassDecoder {
    pub fn new(num_classes: usize, neurons_per_class: usize) -> Self {
        Self {
            num_classes,
            neurons_per_class,
        }
    }

    fn softmax(&self, logits: &mut [f32]) {
        let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_val).exp()).sum();

        for val in logits.iter_mut() {
            *val = (*val - max_val).exp() / exp_sum;
        }
    }
}

impl Decoder for MultiClassDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        let expected_neurons = self.num_classes * self.neurons_per_class;
        if num_neurons != expected_neurons {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} neurons", expected_neurons),
                actual: format!("{} neurons", num_neurons),
            });
        }

        let mut output = Array2::zeros((batch_size, self.num_classes));

        for b in 0..batch_size {
            let mut class_scores = vec![0.0f32; self.num_classes];

            for (c, slot) in class_scores.iter_mut().enumerate().take(self.num_classes) {
                let start = c * self.neurons_per_class;
                let end = start + self.neurons_per_class;

                let class_rate: f32 = rates.slice(s![b, start..end]).mean().unwrap_or(0.0);
                *slot = class_rate;
            }

            self.softmax(&mut class_scores);

            for c in 0..self.num_classes {
                output[[b, c]] = class_scores[c];
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_classes
    }
}

/// Tremor type decoder (rest, action, postural)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorTypeDecoder {
    pub num_neurons: usize,
}

impl TremorTypeDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }

    fn classify_tremor_type(&self, variance: f32, mean_rate: f32) -> usize {
        // 0: Rest tremor (high variance, lower rate)
        // 1: Action tremor (high variance, higher rate)
        // 2: Postural tremor (moderate variance)

        if variance > 0.15 {
            if mean_rate > 0.4 {
                1 // Action
            } else {
                0 // Rest
            }
        } else {
            2 // Postural
        }
    }
}

impl Decoder for TremorTypeDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 3)); // [rest, action, postural]

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            let variance = rates.row(b).var(0.0);

            let tremor_type = self.classify_tremor_type(variance, mean_rate);
            output[[b, tremor_type]] = 1.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        3
    }
}

/// Gait phase decoder (stance, swing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitPhaseDecoder {
    pub num_neurons: usize,
    pub phase_threshold: f32,
}

impl GaitPhaseDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            phase_threshold: 0.4,
        }
    }

    fn detect_phase(&self, activity_level: f32) -> usize {
        if activity_level > self.phase_threshold {
            1 // Swing phase (higher activity)
        } else {
            0 // Stance phase (lower activity)
        }
    }
}

impl Decoder for GaitPhaseDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 2)); // [stance, swing]

        for b in 0..batch_size {
            let mut stance_count = 0;
            let mut swing_count = 0;

            for t in 0..num_steps {
                let activity: f32 = spike_dense.slice(s![b, t, ..]).sum();
                let max_activity = self.num_neurons as f32;
                let normalized_activity = activity / max_activity;

                if self.detect_phase(normalized_activity) == 1 {
                    swing_count += 1;
                } else {
                    stance_count += 1;
                }
            }

            let total = (stance_count + swing_count) as f32;
            output[[b, 0]] = stance_count as f32 / total;
            output[[b, 1]] = swing_count as f32 / total;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        2
    }
}

/// Sleep stage decoder (Wake, N1, N2, N3, REM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepStageDecoder {
    pub num_neurons: usize,
}

impl SleepStageDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }

    fn classify_sleep_stage(&self, mean_rate: f32, variance: f32) -> usize {
        // Simplified classification based on activity patterns
        // 0: Wake (high rate, high variance)
        // 1: N1 (moderate rate, moderate variance)
        // 2: N2 (lower rate, moderate variance)
        // 3: N3 (low rate, low variance)
        // 4: REM (moderate-high rate, high variance)

        if mean_rate > 0.6 && variance > 0.1 {
            0 // Wake
        } else if mean_rate > 0.5 && variance > 0.15 {
            4 // REM
        } else if mean_rate > 0.3 {
            1 // N1
        } else if variance > 0.05 {
            2 // N2
        } else {
            3 // N3
        }
    }
}

impl Decoder for SleepStageDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 5)); // [Wake, N1, N2, N3, REM]

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            let variance = rates.row(b).var(0.0);

            let stage = self.classify_sleep_stage(mean_rate, variance);
            output[[b, stage]] = 1.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5
    }
}

/// Activity recognition decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivityDecoder {
    pub num_neurons: usize,
    pub num_activities: usize,
}

impl ActivityDecoder {
    pub fn new(num_neurons: usize, num_activities: usize) -> Self {
        Self {
            num_neurons,
            num_activities,
        }
    }

    pub fn with_default_activities(num_neurons: usize) -> Self {
        // Default: [sitting, standing, walking, running, lying]
        Self {
            num_neurons,
            num_activities: 5,
        }
    }

    fn classify_activity(&self, mean_rate: f32, std_dev: f32) -> usize {
        // 0: Sitting (low rate, low variance)
        // 1: Standing (low-moderate rate, low variance)
        // 2: Walking (moderate rate, moderate variance)
        // 3: Running (high rate, high variance)
        // 4: Lying (very low rate, very low variance)

        if mean_rate < 0.15 {
            4 // Lying
        } else if mean_rate < 0.3 {
            if std_dev < 0.1 {
                0 // Sitting
            } else {
                1 // Standing
            }
        } else if mean_rate < 0.6 {
            2 // Walking
        } else {
            3 // Running
        }
    }
}

impl Decoder for ActivityDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, self.num_activities));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            let std_dev = rates.row(b).std(0.0);

            let activity = self.classify_activity(mean_rate, std_dev);
            if activity < self.num_activities {
                output[[b, activity]] = 1.0;
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_activities
    }
}

/// Emotion decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionDecoder {
    pub num_neurons: usize,
}

impl EmotionDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }

    fn classify_emotion(&self, arousal: f32, valence: f32) -> usize {
        // Circumplex model of emotion
        // 0: Happy (high arousal, positive valence)
        // 1: Sad (low arousal, negative valence)
        // 2: Angry (high arousal, negative valence)
        // 3: Calm (low arousal, positive valence)
        // 4: Neutral (moderate arousal, neutral valence)

        if arousal.abs() < 0.3 && valence.abs() < 0.3 {
            4 // Neutral
        } else if arousal > 0.3 {
            if valence > 0.0 {
                0 // Happy
            } else {
                2 // Angry
            }
        } else {
            if valence > 0.0 {
                3 // Calm
            } else {
                1 // Sad
            }
        }
    }
}

impl Decoder for EmotionDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let (batch_size, num_neurons) = (rates.shape()[0], rates.shape()[1]);

        let mut output = Array2::zeros((batch_size, 5)); // [happy, sad, angry, calm, neutral]

        for b in 0..batch_size {
            // Use first half for arousal, second half for valence
            let mid = num_neurons / 2;

            let arousal: f32 = rates.slice(s![b, 0..mid]).mean().unwrap_or(0.0);
            let valence: f32 = rates.slice(s![b, mid..]).mean().unwrap_or(0.0);

            // Normalize to [-1, 1]
            let arousal_norm = (arousal - 0.5) * 2.0;
            let valence_norm = (valence - 0.5) * 2.0;

            let emotion = self.classify_emotion(arousal_norm, valence_norm);
            output[[b, emotion]] = 1.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5
    }
}

/// Fatigue level decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FatigueDecoder {
    pub num_neurons: usize,
}

impl FatigueDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }

    fn classify_fatigue(&self, activity_decline: f32, variability: f32) -> usize {
        // 0: Not fatigued (no decline, normal variability)
        // 1: Mild fatigue (slight decline)
        // 2: Moderate fatigue (moderate decline, increased variability)
        // 3: Severe fatigue (significant decline, high variability)

        if activity_decline < 0.2 {
            0 // Not fatigued
        } else if activity_decline < 0.4 {
            1 // Mild
        } else if activity_decline < 0.6 || variability < 0.2 {
            2 // Moderate
        } else {
            3 // Severe
        }
    }
}

impl Decoder for FatigueDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 4)); // [none, mild, moderate, severe]

        for b in 0..batch_size {
            // Compare early vs late activity
            let first_third = num_steps / 3;
            let last_third = num_steps - first_third;

            let early_activity: f32 = spike_dense.slice(s![b, 0..first_third, ..]).sum();
            let late_activity: f32 = spike_dense.slice(s![b, last_third.., ..]).sum();

            let early_mean = early_activity / (first_third as f32);
            let late_mean = late_activity / (first_third as f32);

            let activity_decline = if early_mean > 0.0 {
                (early_mean - late_mean) / early_mean
            } else {
                0.0
            };

            let rates = spikes.spike_rate();
            let variability = rates.row(b).std(0.0);

            let fatigue_level = self.classify_fatigue(activity_decline.max(0.0), variability);
            output[[b, fatigue_level]] = 1.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4
    }
}

/// Medication state decoder (ON/OFF for Parkinson's)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MedicationStateDecoder {
    pub num_neurons: usize,
    pub on_threshold: f32,
}

impl MedicationStateDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            on_threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.on_threshold = threshold;
        self
    }
}

impl Decoder for MedicationStateDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 2)); // [OFF, ON]

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);

            // Higher activity suggests ON state (medication effective)
            if mean_rate > self.on_threshold {
                output[[b, 1]] = 1.0; // ON
            } else {
                output[[b, 0]] = 1.0; // OFF
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        2
    }
}

/// Dyskinesias detection decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DyskinesiasDecoder {
    pub num_neurons: usize,
    pub dyskinesia_threshold: f32,
}

impl DyskinesiasDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            dyskinesia_threshold: 0.15,
        }
    }

    fn detect_dyskinesias(&self, variance: f32, irregularity: f32) -> bool {
        // Dyskinesias characterized by high variance and irregularity
        variance > self.dyskinesia_threshold && irregularity > 0.2
    }
}

impl Decoder for DyskinesiasDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 2)); // [no_dyskinesia, dyskinesia]

        for b in 0..batch_size {
            let rates = spikes.spike_rate();
            let variance = rates.row(b).var(0.0);

            // Compute temporal irregularity
            let mut irregularity = 0.0;
            for n in 0..num_neurons {
                for t in 1..num_steps {
                    irregularity +=
                        (spike_dense[[b, t, n]] - spike_dense[[b, t - 1, n]]).abs();
                }
            }
            irregularity /= (num_steps * num_neurons) as f32;

            if self.detect_dyskinesias(variance, irregularity) {
                output[[b, 1]] = 1.0; // Dyskinesia present
            } else {
                output[[b, 0]] = 1.0; // No dyskinesia
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_binary_class_decoder() {
        let decoder = BinaryClassDecoder::new(10);

        let mut spike_data = Array3::zeros((2, 20, 10));
        // High activity for batch 0
        for t in 0..20 {
            for n in 0..10 {
                spike_data[[0, t, n]] = 0.8;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output[[0, 1]], 1.0); // Class 1 for high activity
        assert_eq!(output[[1, 0]], 1.0); // Class 0 for low activity
    }

    #[test]
    fn test_multi_class_decoder() {
        let decoder = MultiClassDecoder::new(3, 5);

        let mut spike_data = Array3::zeros((1, 20, 15));
        // High activity in class 1 neurons (5-9)
        for t in 0..20 {
            for n in 5..10 {
                spike_data[[0, t, n]] = 1.0;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        // Class 1 should have highest probability
        assert!(output[[0, 1]] > output[[0, 0]]);
        assert!(output[[0, 1]] > output[[0, 2]]);
    }

    #[test]
    fn test_tremor_type_decoder() {
        let decoder = TremorTypeDecoder::new(10);

        let mut spike_data = Array3::zeros((1, 50, 10));
        for t in (0..50).step_by(5) {
            spike_data[[0, t, 0]] = 1.0;
            spike_data[[0, t, 5]] = 0.5;
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 3]);
        let sum: f32 = output.row(0).sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_sleep_stage_decoder() {
        let decoder = SleepStageDecoder::new(10);

        let mut spike_data = Array3::zeros((1, 100, 10));
        // Low activity for deep sleep
        for t in (0..100).step_by(10) {
            spike_data[[0, t, 0]] = 0.2;
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 5]);
    }

    #[test]
    fn test_activity_decoder() {
        let decoder = ActivityDecoder::with_default_activities(10);

        let mut spike_data = Array3::zeros((1, 50, 10));
        // Moderate activity for walking
        for t in 0..50 {
            for n in 0..10 {
                spike_data[[0, t, n]] = 0.4;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 5]);
    }

    #[test]
    fn test_medication_state_decoder() {
        let decoder = MedicationStateDecoder::new(10);

        let mut spike_data = Array3::zeros((2, 30, 10));
        // High activity (ON state) for batch 0
        for t in 0..30 {
            for n in 0..10 {
                spike_data[[0, t, n]] = 0.7;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output[[0, 1]], 1.0); // ON
        assert_eq!(output[[1, 0]], 1.0); // OFF
    }
}
