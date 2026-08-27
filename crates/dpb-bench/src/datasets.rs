//! Standard benchmark datasets with ground truth
//!
//! This module provides synthetic biosignal datasets for benchmarking encoders
//! and neural networks. All datasets include ground truth annotations.

use dpb_core::{GroundTruth, SignalBuffer};
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Common trait for benchmark datasets
pub trait BenchmarkDataset {
    /// Generate signal and ground truth
    fn generate(&self) -> anyhow::Result<(SignalBuffer, GroundTruth)>;

    /// Get dataset name
    fn name(&self) -> &str;

    /// Get sample rate
    fn sample_rate(&self) -> f64;

    /// Get duration in seconds
    fn duration(&self) -> f64;
}

/// Synthetic ECG dataset with known R-peaks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticECG {
    /// Number of samples
    pub num_samples: usize,
    /// Sample rate (Hz)
    pub sample_rate: f64,
    /// Heart rate (bpm)
    pub heart_rate: f64,
    /// Noise level (standard deviation)
    pub noise_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl SyntheticECG {
    /// Create a new synthetic ECG dataset
    pub fn new(num_samples: usize, sample_rate: f64, seed: Option<u64>) -> Self {
        Self {
            num_samples,
            sample_rate,
            heart_rate: 75.0, // Normal resting heart rate
            noise_level: 0.05,
            seed,
        }
    }

    /// Create with custom heart rate
    pub fn with_heart_rate(mut self, heart_rate: f64) -> Self {
        self.heart_rate = heart_rate;
        self
    }

    /// Create with custom noise level
    pub fn with_noise(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }
}

impl BenchmarkDataset for SyntheticECG {
    fn generate(&self) -> anyhow::Result<(SignalBuffer, GroundTruth)> {
        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::make_rng::<rand::rngs::StdRng>()
        };

        let duration = self.num_samples as f64 / self.sample_rate;
        let rr_interval = 60.0 / self.heart_rate; // seconds between beats
        let num_beats = (duration / rr_interval) as usize;

        let mut signal = vec![0.0; self.num_samples];
        let mut r_peak_times = Vec::new();
        let mut r_peak_indices = Vec::new();

        // Generate R-peaks
        for beat in 0..num_beats {
            let peak_time = beat as f64 * rr_interval;
            let peak_idx = (peak_time * self.sample_rate) as usize;

            if peak_idx >= self.num_samples {
                break;
            }

            r_peak_times.push(peak_time);
            r_peak_indices.push(peak_idx);

            // Generate PQRST complex
            self.add_pqrst_complex(&mut signal, peak_idx, &mut rng);
        }

        // Add noise
        let noise = Normal::new(0.0, self.noise_level).unwrap();
        for sample in signal.iter_mut() {
            *sample += noise.sample(&mut rng) as f32;
        }

        // Create signal buffer
        let signal_buffer = SignalBuffer::single_channel(signal, self.sample_rate);

        // Create ground truth
        let mut metadata = HashMap::new();
        metadata.insert("heart_rate".to_string(), self.heart_rate.to_string());
        metadata.insert("num_beats".to_string(), num_beats.to_string());
        metadata.insert("rr_interval".to_string(), rr_interval.to_string());
        metadata.insert("modality".to_string(), "ECG".to_string());

        // Store R-peak times as temporal annotations
        let temporal: Vec<(f64, f64, u32)> = r_peak_times
            .iter()
            .map(|&time| (time, time, 0)) // (start, end, label=0 for R-peak)
            .collect();

        let ground_truth = GroundTruth {
            labels: vec![0; r_peak_indices.len()], // 0 = R-peak
            values: Some(r_peak_indices.iter().map(|&idx| idx as f64).collect()),
            temporal: Some(temporal),
            metadata,
        };

        Ok((signal_buffer, ground_truth))
    }

    fn name(&self) -> &str {
        "SyntheticECG"
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }
}

impl SyntheticECG {
    fn add_pqrst_complex(&self, signal: &mut [f32], peak_idx: usize, rng: &mut impl Rng) {
        let sample_rate = self.sample_rate;

        // PQRST timing (in seconds)
        let p_offset = -0.18;
        let q_offset = -0.04;
        let r_offset = 0.0;
        let s_offset = 0.04;
        let t_offset = 0.2;

        // Amplitudes
        let p_amp = 0.15 + rng.random::<f32>() * 0.05;
        let q_amp = -0.1;
        let r_amp = 1.0;
        let s_amp = -0.2;
        let t_amp = 0.3 + rng.random::<f32>() * 0.1;

        // Add each wave
        self.add_gaussian_wave(signal, peak_idx, p_offset, p_amp, 0.04, sample_rate);
        self.add_gaussian_wave(signal, peak_idx, q_offset, q_amp, 0.02, sample_rate);
        self.add_gaussian_wave(signal, peak_idx, r_offset, r_amp, 0.03, sample_rate);
        self.add_gaussian_wave(signal, peak_idx, s_offset, s_amp, 0.02, sample_rate);
        self.add_gaussian_wave(signal, peak_idx, t_offset, t_amp, 0.08, sample_rate);
    }

    fn add_gaussian_wave(
        &self,
        signal: &mut [f32],
        center_idx: usize,
        time_offset: f64,
        amplitude: f32,
        width: f64,
        sample_rate: f64,
    ) {
        let offset_samples = (time_offset * sample_rate) as isize;
        let width_samples = (width * sample_rate * 3.0) as usize; // 3 sigma

        for i in 0..width_samples {
            let signed_idx =
                center_idx as isize + offset_samples + i as isize - width_samples as isize / 2;
            if let Ok(idx) = usize::try_from(signed_idx)
                && idx < signal.len()
            {
                let t = i as f64 - width_samples as f64 / 2.0;
                let gaussian =
                    amplitude * ((-0.5 * (t / (width * sample_rate)).powi(2)).exp() as f32);
                signal[idx] += gaussian;
            }
        }
    }
}

/// Synthetic gait dataset with known heel strikes and toe-offs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticGait {
    /// Number of samples
    pub num_samples: usize,
    /// Sample rate (Hz)
    pub sample_rate: f64,
    /// Stride frequency (Hz)
    pub stride_frequency: f64,
    /// Stance phase ratio (0.0-1.0)
    pub stance_ratio: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl SyntheticGait {
    /// Create new synthetic gait dataset
    pub fn new(num_samples: usize, sample_rate: f64, seed: Option<u64>) -> Self {
        Self {
            num_samples,
            sample_rate,
            stride_frequency: 0.9, // ~54 steps/min
            stance_ratio: 0.6,     // 60% stance, 40% swing
            noise_level: 0.1,
            seed,
        }
    }
}

impl BenchmarkDataset for SyntheticGait {
    fn generate(&self) -> anyhow::Result<(SignalBuffer, GroundTruth)> {
        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::make_rng::<rand::rngs::StdRng>()
        };

        let duration = self.num_samples as f64 / self.sample_rate;
        let stride_period = 1.0 / self.stride_frequency;
        let num_strides = (duration / stride_period) as usize;

        // 3-axis accelerometer data
        let mut signal = vec![vec![0.0f32; self.num_samples]; 3];
        let mut heel_strikes = Vec::new();
        let mut toe_offs = Vec::new();

        for stride in 0..num_strides {
            let hs_time = stride as f64 * stride_period;
            let to_time = hs_time + stride_period * self.stance_ratio;

            let hs_idx = (hs_time * self.sample_rate) as usize;
            let to_idx = (to_time * self.sample_rate) as usize;

            if hs_idx >= self.num_samples {
                break;
            }

            heel_strikes.push(hs_time);

            if to_idx < self.num_samples {
                toe_offs.push(to_time);
            }

            // Add heel strike impact (vertical acceleration spike)
            for i in 0..20.min(self.num_samples - hs_idx) {
                let t = i as f64 / 20.0;
                let impact = 2.0 * (-10.0 * t).exp();
                signal[2][hs_idx + i] += impact as f32;
            }

            // Add toe-off (forward acceleration)
            if to_idx < self.num_samples {
                for i in 0..15.min(self.num_samples - to_idx) {
                    let t = i as f64 / 15.0;
                    let push = 1.5 * (-8.0 * t).exp();
                    signal[0][to_idx + i] += push as f32;
                }
            }
        }

        // Add noise and baseline
        let noise = Normal::new(0.0, self.noise_level).unwrap();
        for (axis, channel) in signal.iter_mut().enumerate().take(3) {
            for sample in channel.iter_mut() {
                *sample += noise.sample(&mut rng) as f32;
                if axis == 2 {
                    *sample += 9.81; // gravity baseline
                }
            }
        }

        // Flatten to single vector
        let flat_signal: Vec<f32> = signal.into_iter().flatten().collect();
        let signal_buffer = SignalBuffer::multi_channel(flat_signal, self.sample_rate, 3);

        // Create ground truth
        let mut metadata = HashMap::new();
        metadata.insert(
            "stride_frequency".to_string(),
            self.stride_frequency.to_string(),
        );
        metadata.insert("stance_ratio".to_string(), self.stance_ratio.to_string());
        metadata.insert("modality".to_string(), "Gait".to_string());

        // Create temporal annotations
        let mut temporal = Vec::new();
        for &hs_time in &heel_strikes {
            temporal.push((hs_time, hs_time, 0)); // 0 = heel_strike
        }
        for &to_time in &toe_offs {
            temporal.push((to_time, to_time, 1)); // 1 = toe_off
        }

        let mut labels = vec![0; heel_strikes.len()];
        labels.extend(vec![1; toe_offs.len()]);

        let mut values: Vec<f64> = heel_strikes.iter().map(|&t| t * self.sample_rate).collect();
        values.extend(toe_offs.iter().map(|&t| t * self.sample_rate));

        let ground_truth = GroundTruth {
            labels,
            values: Some(values),
            temporal: Some(temporal),
            metadata,
        };

        Ok((signal_buffer, ground_truth))
    }

    fn name(&self) -> &str {
        "SyntheticGait"
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }
}

/// Synthetic tremor dataset with known frequency
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticTremor {
    /// Number of samples
    pub num_samples: usize,
    /// Sample rate (Hz)
    pub sample_rate: f64,
    /// Tremor frequency (Hz)
    pub tremor_frequency: f64,
    /// Tremor amplitude
    pub amplitude: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl SyntheticTremor {
    /// Create new synthetic tremor dataset
    pub fn new(num_samples: usize, sample_rate: f64, seed: Option<u64>) -> Self {
        Self {
            num_samples,
            sample_rate,
            tremor_frequency: 5.0, // Parkinsonian tremor ~4-6 Hz
            amplitude: 1.0,
            noise_level: 0.1,
            seed,
        }
    }

    /// Create with custom frequency
    pub fn with_frequency(mut self, frequency: f64) -> Self {
        self.tremor_frequency = frequency;
        self
    }
}

impl BenchmarkDataset for SyntheticTremor {
    fn generate(&self) -> anyhow::Result<(SignalBuffer, GroundTruth)> {
        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::make_rng::<rand::rngs::StdRng>()
        };

        // 3-axis accelerometer/gyroscope data
        let mut signal = vec![vec![0.0f32; self.num_samples]; 3];

        let noise = Normal::new(0.0, self.noise_level).unwrap();
        let omega = 2.0 * std::f64::consts::PI * self.tremor_frequency;

        // The index reaches other collections too (a second axis, a confusion
        // matrix row, the previous layer's buffer), so a single iterator over one
        // of them will not do.
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.num_samples {
            let t = i as f64 / self.sample_rate;

            // Dominant tremor in one axis with harmonics
            signal[0][i] = (self.amplitude * (omega * t).sin()) as f32;
            signal[0][i] += (0.3 * self.amplitude * (2.0 * omega * t).sin()) as f32; // 2nd harmonic

            // Coupled tremor in other axes
            signal[1][i] = (0.6 * self.amplitude * (omega * t + 0.5).sin()) as f32;
            signal[2][i] = (0.4 * self.amplitude * (omega * t + 1.0).sin()) as f32;

            // Add noise to all axes
            // The index reaches other collections too (a second axis, a confusion
            // matrix row, the previous layer's buffer), so a single iterator over one
            // of them will not do.
            #[allow(clippy::needless_range_loop)]
            for axis in 0..3 {
                signal[axis][i] += noise.sample(&mut rng) as f32;
            }
        }

        // Flatten signal
        let flat_signal: Vec<f32> = signal.into_iter().flatten().collect();
        let signal_buffer = SignalBuffer::multi_channel(flat_signal, self.sample_rate, 3);

        // Create ground truth
        let mut metadata = HashMap::new();
        metadata.insert(
            "tremor_frequency".to_string(),
            self.tremor_frequency.to_string(),
        );
        metadata.insert("amplitude".to_string(), self.amplitude.to_string());
        metadata.insert("modality".to_string(), "Tremor".to_string());

        let ground_truth = GroundTruth {
            labels: vec![],
            values: Some(vec![self.tremor_frequency]),
            temporal: None,
            metadata,
        };

        Ok((signal_buffer, ground_truth))
    }

    fn name(&self) -> &str {
        "SyntheticTremor"
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }
}

/// Synthetic voice dataset with known F0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticVoice {
    /// Number of samples
    pub num_samples: usize,
    /// Sample rate (Hz)
    pub sample_rate: f64,
    /// Fundamental frequency (Hz)
    pub f0: f64,
    /// Number of harmonics
    pub num_harmonics: usize,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl SyntheticVoice {
    /// Create new synthetic voice dataset
    pub fn new(num_samples: usize, sample_rate: f64, seed: Option<u64>) -> Self {
        Self {
            num_samples,
            sample_rate,
            f0: 120.0, // Typical male voice
            num_harmonics: 10,
            noise_level: 0.05,
            seed,
        }
    }

    /// Create with custom F0
    pub fn with_f0(mut self, f0: f64) -> Self {
        self.f0 = f0;
        self
    }
}

impl BenchmarkDataset for SyntheticVoice {
    fn generate(&self) -> anyhow::Result<(SignalBuffer, GroundTruth)> {
        let mut rng = if let Some(seed) = self.seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::make_rng::<rand::rngs::StdRng>()
        };

        let mut signal = vec![0.0f32; self.num_samples];
        let noise = Normal::new(0.0, self.noise_level).unwrap();

        // Generate harmonics
        // The index reaches other collections too (a second axis, a confusion
        // matrix row, the previous layer's buffer), so a single iterator over one
        // of them will not do.
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.num_samples {
            let t = i as f64 / self.sample_rate;

            for h in 1..=self.num_harmonics {
                let freq = self.f0 * h as f64;
                let amplitude = 1.0 / h as f64; // Decreasing amplitude
                signal[i] += (amplitude * (2.0 * std::f64::consts::PI * freq * t).sin()) as f32;
            }

            signal[i] += noise.sample(&mut rng) as f32;
        }

        // Normalize
        let max_amp = signal.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        for sample in signal.iter_mut() {
            *sample /= max_amp;
        }

        let signal_buffer = SignalBuffer::single_channel(signal, self.sample_rate);

        // Create ground truth
        let mut metadata = HashMap::new();
        metadata.insert("f0".to_string(), self.f0.to_string());
        metadata.insert("num_harmonics".to_string(), self.num_harmonics.to_string());
        metadata.insert("modality".to_string(), "Voice".to_string());

        let ground_truth = GroundTruth {
            labels: vec![],
            values: Some(vec![self.f0]),
            temporal: None,
            metadata,
        };

        Ok((signal_buffer, ground_truth))
    }

    fn name(&self) -> &str {
        "SyntheticVoice"
    }

    fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::Signal;

    #[test]
    fn test_synthetic_ecg() {
        let dataset = SyntheticECG::new(1000, 250.0, Some(42));
        let (signal, gt) = dataset.generate().unwrap();

        assert_eq!(signal.sample_rate(), 250.0);
        assert_eq!(signal.samples().len(), 1000);
        assert!(gt.temporal.is_some());
        assert!(!gt.temporal.unwrap().is_empty());
    }

    #[test]
    fn test_synthetic_gait() {
        let dataset = SyntheticGait::new(1000, 100.0, Some(42));
        let (signal, gt) = dataset.generate().unwrap();

        assert_eq!(signal.sample_rate(), 100.0);
        assert_eq!(signal.channels(), 3);
        assert!(gt.temporal.is_some());
    }

    #[test]
    fn test_synthetic_tremor() {
        let dataset = SyntheticTremor::new(1000, 100.0, Some(42));
        let (signal, gt) = dataset.generate().unwrap();

        assert_eq!(signal.sample_rate(), 100.0);
        assert!(gt.values.is_some());
        let values = gt.values.unwrap();
        assert!(!values.is_empty());
        assert_eq!(values[0], 5.0); // Default tremor frequency
    }

    #[test]
    fn test_synthetic_voice() {
        let dataset = SyntheticVoice::new(1000, 16000.0, Some(42));
        let (signal, gt) = dataset.generate().unwrap();

        assert_eq!(signal.sample_rate(), 16000.0);
        assert!(gt.values.is_some());
        let values = gt.values.unwrap();
        assert!(!values.is_empty());
        assert_eq!(values[0], 120.0); // Default F0
    }
}
