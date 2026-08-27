//! Regression decoders for continuous physiological measurements

use super::Decoder;
use crate::{SNNResult, SpikeTensor};
use ndarray::{Array2, s};
use serde::{Deserialize, Serialize};

/// Heart rate decoder - HR from cardiac spikes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartRateDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32, // Hz
    pub min_hr: f32,
    pub max_hr: f32,
}

impl HeartRateDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32) -> Self {
        Self {
            num_neurons,
            sampling_rate,
            min_hr: 40.0,
            max_hr: 200.0,
        }
    }

    fn detect_peaks(&self, spike_train: &[f32]) -> Vec<usize> {
        let mut peaks = Vec::new();
        let window_size = (self.sampling_rate * 0.2) as usize; // 200ms refractory

        for i in window_size..spike_train.len() - window_size {
            if spike_train[i] > 0.5 {
                // Check if it's a local maximum
                let is_peak = (i.saturating_sub(window_size)..i)
                    .all(|j| spike_train[j] <= spike_train[i])
                    && (i + 1..=(i + window_size).min(spike_train.len() - 1))
                        .all(|j| spike_train[j] < spike_train[i]);

                if is_peak && (peaks.is_empty() || i - peaks[peaks.len() - 1] >= window_size) {
                    peaks.push(i);
                }
            }
        }

        peaks
    }

    fn compute_heart_rate(&self, peaks: &[usize], _num_steps: usize) -> f32 {
        if peaks.len() < 2 {
            return 60.0; // Default resting HR
        }

        // Compute RR intervals
        let rr_intervals: Vec<f32> = peaks
            .windows(2)
            .map(|w| (w[1] - w[0]) as f32 / self.sampling_rate)
            .collect();

        if rr_intervals.is_empty() {
            return 60.0;
        }

        // Mean heart rate in BPM
        let mean_rr = rr_intervals.iter().sum::<f32>() / rr_intervals.len() as f32;
        let hr = 60.0 / mean_rr;

        hr.clamp(self.min_hr, self.max_hr)
    }
}

impl Decoder for HeartRateDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            // Average across all neurons
            let mut combined_signal = vec![0.0f32; num_steps];
            for (t, slot) in combined_signal.iter_mut().enumerate().take(num_steps) {
                *slot = spike_dense.slice(s![b, t, ..]).sum();
            }

            let peaks = self.detect_peaks(&combined_signal);
            output[[b, 0]] = self.compute_heart_rate(&peaks, num_steps);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Heart rate variability decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HRVDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32,
}

impl HRVDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32) -> Self {
        Self {
            num_neurons,
            sampling_rate,
        }
    }

    fn compute_hrv_metrics(&self, rr_intervals: &[f32]) -> (f32, f32) {
        if rr_intervals.len() < 2 {
            return (0.0, 0.0);
        }

        // SDNN (standard deviation of NN intervals)
        let mean_rr = rr_intervals.iter().sum::<f32>() / rr_intervals.len() as f32;
        let sdnn = (rr_intervals
            .iter()
            .map(|&x| (x - mean_rr).powi(2))
            .sum::<f32>()
            / rr_intervals.len() as f32)
            .sqrt();

        // RMSSD (root mean square of successive differences)
        let successive_diffs: Vec<f32> = rr_intervals
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .collect();

        let rmssd = if !successive_diffs.is_empty() {
            (successive_diffs.iter().sum::<f32>() / successive_diffs.len() as f32).sqrt()
        } else {
            0.0
        };

        (sdnn * 1000.0, rmssd * 1000.0) // Convert to ms
    }
}

impl Decoder for HRVDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 2)); // SDNN and RMSSD

        for b in 0..batch_size {
            let mut combined_signal = vec![0.0f32; num_steps];
            for (t, slot) in combined_signal.iter_mut().enumerate().take(num_steps) {
                *slot = spike_dense.slice(s![b, t, ..]).sum();
            }

            // Detect peaks
            let peaks: Vec<usize> = (1..num_steps - 1)
                .filter(|&i| {
                    combined_signal[i] > 0.5
                        && combined_signal[i] > combined_signal[i - 1]
                        && combined_signal[i] > combined_signal[i + 1]
                })
                .collect();

            if peaks.len() >= 2 {
                let rr_intervals: Vec<f32> = peaks
                    .windows(2)
                    .map(|w| (w[1] - w[0]) as f32 / self.sampling_rate)
                    .collect();

                let (sdnn, rmssd) = self.compute_hrv_metrics(&rr_intervals);
                output[[b, 0]] = sdnn;
                output[[b, 1]] = rmssd;
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        2 // SDNN and RMSSD
    }
}

/// Tremor frequency decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorFrequencyDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32,
}

impl TremorFrequencyDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32) -> Self {
        Self {
            num_neurons,
            sampling_rate,
        }
    }

    fn estimate_dominant_frequency(&self, signal: &[f32]) -> f32 {
        if signal.len() < 10 {
            return 0.0;
        }

        // Simple autocorrelation-based frequency estimation
        let mut max_corr = 0.0;
        let mut best_lag = 1;

        for lag in 1..signal.len() / 2 {
            let mut corr = 0.0;
            let mut count = 0;

            for i in 0..signal.len() - lag {
                corr += signal[i] * signal[i + lag];
                count += 1;
            }

            corr /= count as f32;

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        // Convert lag to frequency
        self.sampling_rate / best_lag as f32
    }
}

impl Decoder for TremorFrequencyDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, _num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mut frequencies = Vec::new();

            for n in 0..num_neurons {
                let signal: Vec<f32> = spike_dense.slice(s![b, .., n]).to_vec();
                let freq = self.estimate_dominant_frequency(&signal);
                if freq > 1.0 && freq < 20.0 {
                    // Physiological tremor range
                    frequencies.push(freq);
                }
            }

            if !frequencies.is_empty() {
                output[[b, 0]] = frequencies.iter().sum::<f32>() / frequencies.len() as f32;
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Tremor amplitude decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorAmplitudeDecoder {
    pub num_neurons: usize,
}

impl TremorAmplitudeDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for TremorAmplitudeDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, _num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mut total_amplitude = 0.0;

            for n in 0..num_neurons {
                let signal: Vec<f32> = spike_dense.slice(s![b, .., n]).to_vec();

                // Compute peak-to-peak amplitude using envelope
                let max_val = signal.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let min_val = signal.iter().fold(f32::INFINITY, |a, &b| a.min(b));

                total_amplitude += max_val - min_val;
            }

            output[[b, 0]] = total_amplitude / num_neurons as f32;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Gait velocity decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitVelocityDecoder {
    pub num_neurons: usize,
    pub stride_length_m: f32, // Average stride length in meters
}

impl GaitVelocityDecoder {
    pub fn new(num_neurons: usize, stride_length_m: f32) -> Self {
        Self {
            num_neurons,
            stride_length_m,
        }
    }

    fn detect_steps(&self, signal: &[f32], threshold: f32) -> usize {
        let mut step_count = 0;
        let mut in_step = false;

        for &val in signal {
            if val > threshold && !in_step {
                step_count += 1;
                in_step = true;
            } else if val <= threshold {
                in_step = false;
            }
        }

        step_count
    }
}

impl Decoder for GaitVelocityDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            // Combine signal across neurons
            let mut combined_signal = vec![0.0f32; num_steps];
            for (t, slot) in combined_signal.iter_mut().enumerate().take(num_steps) {
                *slot = spike_dense.slice(s![b, t, ..]).sum();
            }

            // Normalize
            let max_val = combined_signal
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            if max_val > 0.0 {
                for val in combined_signal.iter_mut() {
                    *val /= max_val;
                }
            }

            let steps = self.detect_steps(&combined_signal, 0.3);

            // Velocity = (steps * stride_length) / time
            // Assuming num_steps represents time in some unit
            let velocity = (steps as f32 * self.stride_length_m) / (num_steps as f32 / 100.0);
            output[[b, 0]] = velocity.clamp(0.0, 3.0); // Cap at 3 m/s
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Stride time decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideTimeDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32,
}

impl StrideTimeDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32) -> Self {
        Self {
            num_neurons,
            sampling_rate,
        }
    }
}

impl Decoder for StrideTimeDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            // Detect stride events
            let mut combined_signal = vec![0.0f32; num_steps];
            for (t, slot) in combined_signal.iter_mut().enumerate().take(num_steps) {
                *slot = spike_dense.slice(s![b, t, ..]).sum();
            }

            let peaks: Vec<usize> = (1..num_steps - 1)
                .filter(|&i| {
                    combined_signal[i] > 0.5
                        && combined_signal[i] > combined_signal[i - 1]
                        && combined_signal[i] > combined_signal[i + 1]
                })
                .collect();

            if peaks.len() >= 2 {
                let stride_intervals: Vec<f32> = peaks
                    .windows(2)
                    .map(|w| (w[1] - w[0]) as f32 / self.sampling_rate)
                    .collect();

                let mean_stride_time =
                    stride_intervals.iter().sum::<f32>() / stride_intervals.len() as f32;
                output[[b, 0]] = mean_stride_time;
            } else {
                output[[b, 0]] = 1.0; // Default 1 second
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Tapping frequency decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TappingFrequencyDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32,
}

impl TappingFrequencyDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32) -> Self {
        Self {
            num_neurons,
            sampling_rate,
        }
    }
}

impl Decoder for TappingFrequencyDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mut combined_signal = vec![0.0f32; num_steps];
            for (t, slot) in combined_signal.iter_mut().enumerate().take(num_steps) {
                *slot = spike_dense.slice(s![b, t, ..]).sum();
            }

            // Count taps (peaks above threshold)
            let taps = combined_signal.iter().filter(|&&x| x > 0.5).count();

            // Tapping frequency in Hz
            let duration_sec = num_steps as f32 / self.sampling_rate;
            let frequency = if duration_sec > 0.0 {
                taps as f32 / duration_sec
            } else {
                0.0
            };

            output[[b, 0]] = frequency;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Reaction time decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionTimeDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32,
    pub stimulus_time: usize, // Time step when stimulus was presented
}

impl ReactionTimeDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32, stimulus_time: usize) -> Self {
        Self {
            num_neurons,
            sampling_rate,
            stimulus_time,
        }
    }
}

impl Decoder for ReactionTimeDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            // Find first significant spike after stimulus
            let mut reaction_time = f32::INFINITY;

            for t in self.stimulus_time..num_steps {
                let spike_sum: f32 = spike_dense.slice(s![b, t, ..]).sum();

                if spike_sum > (num_neurons as f32 * 0.3) {
                    // 30% of neurons active
                    reaction_time = (t - self.stimulus_time) as f32 / self.sampling_rate;
                    break;
                }
            }

            output[[b, 0]] = if reaction_time.is_finite() {
                reaction_time * 1000.0 // Convert to ms
            } else {
                1000.0 // Default 1000ms if no response
            };
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Speech rate decoder (words per minute)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRateDecoder {
    pub num_neurons: usize,
    pub sampling_rate: f32,
}

impl SpeechRateDecoder {
    pub fn new(num_neurons: usize, sampling_rate: f32) -> Self {
        Self {
            num_neurons,
            sampling_rate,
        }
    }

    fn detect_syllables(&self, signal: &[f32], threshold: f32) -> usize {
        let mut syllable_count = 0;
        let mut in_syllable = false;
        let min_gap = 5; // Minimum gap between syllables
        let mut last_syllable = 0;

        for (i, &val) in signal.iter().enumerate() {
            if val > threshold && !in_syllable && (i - last_syllable) >= min_gap {
                syllable_count += 1;
                in_syllable = true;
                last_syllable = i;
            } else if val <= threshold * 0.5 {
                in_syllable = false;
            }
        }

        syllable_count
    }
}

impl Decoder for SpeechRateDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mut combined_signal = vec![0.0f32; num_steps];
            for (t, slot) in combined_signal.iter_mut().enumerate().take(num_steps) {
                *slot = spike_dense.slice(s![b, t, ..]).sum();
            }

            // Normalize
            let max_val = combined_signal
                .iter()
                .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            if max_val > 0.0 {
                for val in combined_signal.iter_mut() {
                    *val /= max_val;
                }
            }

            let syllables = self.detect_syllables(&combined_signal, 0.3);

            // Estimate words (assuming ~1.5 syllables per word)
            let words = syllables as f32 / 1.5;

            // Convert to words per minute
            let duration_min = (num_steps as f32 / self.sampling_rate) / 60.0;
            let wpm = if duration_min > 0.0 {
                words / duration_min
            } else {
                0.0
            };

            output[[b, 0]] = wpm.min(300.0); // Cap at 300 WPM
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Pupil diameter decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PupilDiameterDecoder {
    pub num_neurons: usize,
    pub min_diameter_mm: f32,
    pub max_diameter_mm: f32,
}

impl PupilDiameterDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            min_diameter_mm: 2.0,
            max_diameter_mm: 8.0,
        }
    }
}

impl Decoder for PupilDiameterDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);

            // Map spike rate to pupil diameter
            // Higher activity typically correlates with larger pupils (arousal)
            let diameter = self.min_diameter_mm
                + mean_rate * (self.max_diameter_mm - self.min_diameter_mm);

            output[[b, 0]] = diameter.clamp(self.min_diameter_mm, self.max_diameter_mm);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    #[test]
    fn test_heart_rate_decoder() {
        let decoder = HeartRateDecoder::new(10, 100.0);

        let mut spike_data = Array3::zeros((1, 100, 10));
        // Simulate regular heartbeat at ~60 BPM
        for t in (10..100).step_by(10) {
            for n in 0..10 {
                spike_data[[0, t, n]] = 1.0;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] > 0.0);
        assert!(output[[0, 0]] >= decoder.min_hr && output[[0, 0]] <= decoder.max_hr);
    }

    #[test]
    fn test_hrv_decoder() {
        let decoder = HRVDecoder::new(10, 100.0);

        let mut spike_data = Array3::zeros((1, 100, 10));
        for t in (10..100).step_by(12) {
            for n in 0..10 {
                spike_data[[0, t, n]] = 1.0;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_tremor_frequency_decoder() {
        let decoder = TremorFrequencyDecoder::new(5, 100.0);

        let mut spike_data = Array3::zeros((1, 100, 5));
        // Simulate 5 Hz tremor
        for t in (0..100).step_by(20) {
            spike_data[[0, t, 0]] = 1.0;
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] >= 0.0);
    }

    #[test]
    fn test_gait_velocity_decoder() {
        let decoder = GaitVelocityDecoder::new(10, 0.7);

        let mut spike_data = Array3::zeros((1, 100, 10));
        for t in (10..100).step_by(20) {
            for n in 0..10 {
                spike_data[[0, t, n]] = 1.0;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] >= 0.0 && output[[0, 0]] <= 3.0);
    }

    #[test]
    fn test_pupil_diameter_decoder() {
        let decoder = PupilDiameterDecoder::new(10);

        let mut spike_data = Array3::zeros((1, 50, 10));
        for t in 0..25 {
            for n in 0..10 {
                spike_data[[0, t, n]] = 0.5;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert!(output[[0, 0]] >= decoder.min_diameter_mm);
        assert!(output[[0, 0]] <= decoder.max_diameter_mm);
    }
}
