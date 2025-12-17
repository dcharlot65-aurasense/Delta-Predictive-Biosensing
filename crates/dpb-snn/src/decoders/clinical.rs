//! Clinical score decoders for Parkinson's disease and movement disorders

use super::Decoder;
use crate::{SNNError, SNNResult, SpikeTensor};
use ndarray::{Array1, Array2, s};
use serde::{Deserialize, Serialize};

/// UPDRS (Unified Parkinson's Disease Rating Scale) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDRSDecoder {
    /// Number of UPDRS subscores
    pub num_subscores: usize,
    /// Neuron groups for each subscore
    pub neuron_groups: Vec<(usize, usize)>, // (start, end) indices
    /// Scaling factors for each subscore
    pub scale_factors: Vec<f32>,
    /// Offset values
    pub offsets: Vec<f32>,
}

impl UPDRSDecoder {
    /// Create decoder for UPDRS Part III (motor examination)
    pub fn for_part_iii(num_neurons: usize) -> Self {
        // UPDRS Part III has multiple subscores for different motor features
        // Simplified: 4 main categories
        let categories = vec![
            "tremor",       // 0-4 scale
            "rigidity",     // 0-4 scale
            "bradykinesia", // 0-4 scale
            "postural",     // 0-4 scale
        ];

        let num_subscores = categories.len();
        let neurons_per_score = num_neurons / num_subscores;

        let mut neuron_groups = Vec::new();
        for i in 0..num_subscores {
            let start = i * neurons_per_score;
            let end = if i == num_subscores - 1 {
                num_neurons
            } else {
                (i + 1) * neurons_per_score
            };
            neuron_groups.push((start, end));
        }

        Self {
            num_subscores,
            neuron_groups,
            scale_factors: vec![4.0; num_subscores], // Scale to 0-4
            offsets: vec![0.0; num_subscores],
        }
    }

    /// Create custom decoder
    pub fn new(
        num_subscores: usize,
        neuron_groups: Vec<(usize, usize)>,
        scale_factors: Vec<f32>,
        offsets: Vec<f32>,
    ) -> Self {
        Self {
            num_subscores,
            neuron_groups,
            scale_factors,
            offsets,
        }
    }

    fn compute_subscore(&self, spike_rates: &Array1<f32>, group_idx: usize) -> f32 {
        let (start, end) = self.neuron_groups[group_idx];

        // Average activity of neurons in this group
        let mean_activity: f32 = spike_rates.slice(s![start..end]).mean().unwrap_or(0.0);

        // Scale and offset to clinical score range
        mean_activity * self.scale_factors[group_idx] + self.offsets[group_idx]
    }
}

impl Decoder for UPDRSDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, self.num_subscores));

        for b in 0..batch_size {
            let batch_rates = rates.row(b);

            for i in 0..self.num_subscores {
                output[[b, i]] = self.compute_subscore(&batch_rates.to_owned(), i);
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.num_subscores
    }
}

/// Tremor severity decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorSeverityDecoder {
    /// Frequency bands for tremor analysis
    pub frequency_bands: Vec<(f32, f32)>, // (low_freq, high_freq) in Hz
    /// Sampling rate (Hz)
    pub sampling_rate: f32,
    /// Severity levels (0-4 scale)
    pub severity_levels: usize,
}

impl TremorSeverityDecoder {
    pub fn new(sampling_rate: f32) -> Self {
        // Parkinson's tremor is typically 4-6 Hz
        // Essential tremor is typically 4-12 Hz
        let frequency_bands = vec![
            (3.0, 7.0),   // Parkinsonian tremor band
            (7.0, 12.0),  // Essential tremor band
            (12.0, 20.0), // Physiological tremor band
        ];

        Self {
            frequency_bands,
            sampling_rate,
            severity_levels: 5, // 0-4 scale
        }
    }

    fn compute_tremor_power(&self, spike_train: &[f32]) -> Vec<f32> {
        // Simplified power spectral density computation
        // In practice, would use proper FFT
        let mut band_powers = vec![0.0; self.frequency_bands.len()];

        for (band_idx, &(low_freq, high_freq)) in self.frequency_bands.iter().enumerate() {
            // Simple oscillation detection in frequency band
            let samples_per_cycle = self.sampling_rate / ((low_freq + high_freq) / 2.0);
            let window_size = samples_per_cycle as usize;

            if window_size > spike_train.len() {
                continue;
            }

            // Compute autocorrelation at expected periods
            let mut power = 0.0;
            for lag in (window_size..window_size * 2).step_by(1) {
                if lag >= spike_train.len() {
                    break;
                }

                let mut corr = 0.0;
                for i in 0..(spike_train.len() - lag) {
                    corr += spike_train[i] * spike_train[i + lag];
                }
                power += corr.abs();
            }

            band_powers[band_idx] = power;
        }

        band_powers
    }

    fn classify_severity(&self, power: f32) -> f32 {
        // Map power to severity score (0-4)
        let normalized_power = power.min(1.0);
        normalized_power * (self.severity_levels - 1) as f32
    }
}

impl Decoder for TremorSeverityDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: batch x frequency_bands
        let num_bands = self.frequency_bands.len();
        let mut output = Array2::zeros((batch_size, num_bands + 1)); // +1 for overall severity

        for b in 0..batch_size {
            // Compute tremor power for each neuron and average
            let mut total_band_powers = vec![0.0; num_bands];

            for n in 0..num_neurons {
                let spike_train: Vec<f32> = spike_dense.slice(s![b, .., n]).to_vec();
                let band_powers = self.compute_tremor_power(&spike_train);

                for (i, &power) in band_powers.iter().enumerate() {
                    total_band_powers[i] += power;
                }
            }

            // Average across neurons
            for i in 0..num_bands {
                total_band_powers[i] /= num_neurons as f32;
                output[[b, i]] = self.classify_severity(total_band_powers[i]);
            }

            // Overall severity (max across bands)
            let max_power = total_band_powers
                .iter()
                .fold(0.0f32, |a, &b| a.max(b));
            output[[b, num_bands]] = self.classify_severity(max_power);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.frequency_bands.len() + 1 // bands + overall
    }
}

/// Gait score decoder for mobility assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitScoreDecoder {
    /// Gait features to assess
    pub features: Vec<GaitFeature>,
    /// Neurons per feature
    pub neurons_per_feature: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GaitFeature {
    /// Step frequency/cadence
    Cadence,
    /// Step length/stride
    StrideLength,
    /// Gait speed
    Speed,
    /// Balance/stability
    Balance,
    /// Freezing of gait
    Freezing,
}

impl GaitScoreDecoder {
    pub fn new(neurons_per_feature: usize) -> Self {
        let features = vec![
            GaitFeature::Cadence,
            GaitFeature::StrideLength,
            GaitFeature::Speed,
            GaitFeature::Balance,
            GaitFeature::Freezing,
        ];

        Self {
            features,
            neurons_per_feature,
        }
    }

    fn compute_feature_score(
        &self,
        feature: GaitFeature,
        spike_rates: &[f32],
    ) -> f32 {
        match feature {
            GaitFeature::Cadence => {
                // Higher spike rate = higher cadence
                let mean_rate = spike_rates.iter().sum::<f32>() / spike_rates.len() as f32;
                mean_rate * 10.0 // Scale to clinical range
            }
            GaitFeature::StrideLength => {
                // Variability in spike patterns indicates stride length
                let mean = spike_rates.iter().sum::<f32>() / spike_rates.len() as f32;
                let variance = spike_rates
                    .iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f32>()
                    / spike_rates.len() as f32;
                variance.sqrt() * 5.0
            }
            GaitFeature::Speed => {
                // Combined measure
                let mean_rate = spike_rates.iter().sum::<f32>() / spike_rates.len() as f32;
                mean_rate * 8.0
            }
            GaitFeature::Balance => {
                // Regularity of spike patterns
                let mut regularity = 0.0;
                for i in 1..spike_rates.len() {
                    regularity += (spike_rates[i] - spike_rates[i - 1]).abs();
                }
                regularity /= (spike_rates.len() - 1) as f32;
                (1.0 - regularity).max(0.0) * 4.0 // Inverse - more regular = better balance
            }
            GaitFeature::Freezing => {
                // Detect sudden drops in activity
                let mut freeze_score = 0.0f32;
                let threshold = 0.5f32;

                for window in spike_rates.windows(3) {
                    let before = window[0];
                    let during = window[1];
                    let after = window[2];

                    if before > threshold && during < threshold * 0.3 && after > threshold {
                        freeze_score += 1.0f32;
                    }
                }

                freeze_score.min(4.0f32) // Cap at 4 (severe)
            }
        }
    }
}

impl Decoder for GaitScoreDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];
        let num_features = self.features.len();

        let mut output = Array2::zeros((batch_size, num_features));

        for b in 0..batch_size {
            for (f_idx, &feature) in self.features.iter().enumerate() {
                let start = f_idx * self.neurons_per_feature;
                let end = (start + self.neurons_per_feature).min(rates.shape()[1]);

                let feature_rates: Vec<f32> = rates.slice(s![b, start..end]).to_vec();
                output[[b, f_idx]] = self.compute_feature_score(feature, &feature_rates);
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        self.features.len()
    }
}

/// UPDRS Motor decoder - comprehensive motor assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDRSMotorDecoder {
    pub num_neurons: usize,
}

impl UPDRSMotorDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for UPDRSMotorDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // UPDRS Part III total score (0-132 scale)
        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Scale to UPDRS range (inverse - lower activity = higher impairment)
            output[[b, 0]] = (1.0 - mean_rate).max(0.0) * 132.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// UPDRS Tremor decoder - tremor-specific assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDRSTremorDecoder {
    pub num_neurons: usize,
    pub tremor_band_low: f32,  // Hz
    pub tremor_band_high: f32, // Hz
}

impl UPDRSTremorDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            tremor_band_low: 4.0,
            tremor_band_high: 6.0,
        }
    }
}

impl Decoder for UPDRSTremorDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, _, _) = (spike_dense.shape()[0], spike_dense.shape()[1], spike_dense.shape()[2]);

        // Tremor score (0-4 for each limb, 0-20 total)
        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let rates = spikes.spike_rate();
            let variance = rates.row(b).var(0.0);
            // Higher variance suggests tremor
            output[[b, 0]] = (variance * 20.0).min(20.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// UPDRS Bradykinesia decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDRSBradykinesiaDecoder {
    pub num_neurons: usize,
}

impl UPDRSBradykinesiaDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for UPDRSBradykinesiaDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Lower rate suggests bradykinesia
            output[[b, 0]] = (1.0 - mean_rate).max(0.0) * 4.0; // 0-4 scale
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// UPDRS Rigidity decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDRSRigidityDecoder {
    pub num_neurons: usize,
}

impl UPDRSRigidityDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for UPDRSRigidityDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let batch_size = spike_dense.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let rates = spikes.spike_rate();
            let std_dev = rates.row(b).std(0.0);
            // Low variability suggests rigidity
            output[[b, 0]] = ((1.0 - std_dev) * 4.0).max(0.0).min(4.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// UPDRS Gait decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UPDRSGaitDecoder {
    pub num_neurons: usize,
}

impl UPDRSGaitDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for UPDRSGaitDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            // Detect rhythm regularity
            let mut regularity_score = 0.0;
            for n in 0..num_neurons {
                let spike_train: Vec<f32> = spike_dense.slice(s![b, .., n]).to_vec();
                let spike_times: Vec<usize> = spike_train
                    .iter()
                    .enumerate()
                    .filter(|&(_, s)| *s > 0.5)
                    .map(|(t, _)| t)
                    .collect();

                if spike_times.len() >= 2 {
                    let isis: Vec<f32> = spike_times
                        .windows(2)
                        .map(|w| (w[1] - w[0]) as f32)
                        .collect();
                    let mean_isi = isis.iter().sum::<f32>() / isis.len() as f32;
                    let isi_std = (isis.iter().map(|&x| (x - mean_isi).powi(2)).sum::<f32>()
                        / isis.len() as f32)
                        .sqrt();
                    regularity_score += isi_std / mean_isi.max(1.0);
                }
            }
            regularity_score /= num_neurons as f32;
            // Higher irregularity = higher gait impairment
            output[[b, 0]] = (regularity_score * 4.0).min(4.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Timed Up and Go (TUG) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TUGDecoder {
    pub num_neurons: usize,
    pub expected_duration: f32, // seconds
}

impl TUGDecoder {
    pub fn new(num_neurons: usize, expected_duration: f32) -> Self {
        Self {
            num_neurons,
            expected_duration,
        }
    }
}

impl Decoder for TUGDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Lower activity suggests longer TUG time
            let estimated_time = self.expected_duration / (mean_rate + 0.1);
            output[[b, 0]] = estimated_time.min(60.0); // Cap at 60 seconds
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Berg Balance Scale decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BergBalanceDecoder {
    pub num_neurons: usize,
}

impl BergBalanceDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for BergBalanceDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let std_dev = rates.row(b).std(0.0);
            // Higher stability (lower variance) = higher Berg score (0-56)
            output[[b, 0]] = ((1.0 - std_dev) * 56.0).max(0.0).min(56.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Montreal Cognitive Assessment (MoCA) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoCADecoder {
    pub num_neurons: usize,
}

impl MoCADecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for MoCADecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Higher cognitive function = higher rate (0-30 scale)
            output[[b, 0]] = (mean_rate * 30.0).min(30.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Voice Handicap Index decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceHDDecoder {
    pub num_neurons: usize,
}

impl VoiceHDDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for VoiceHDDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let batch_size = spike_dense.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let rates = spikes.spike_rate();
            let variance = rates.row(b).var(0.0);
            // Higher irregularity suggests voice impairment (0-120 scale)
            output[[b, 0]] = (variance * 120.0).min(120.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Parkinson's Disease Questionnaire-39 decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PDQ39Decoder {
    pub num_neurons: usize,
}

impl PDQ39Decoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for PDQ39Decoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Lower quality of life = lower rate (0-100 scale, higher = worse)
            output[[b, 0]] = ((1.0 - mean_rate) * 100.0).max(0.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Hoehn & Yahr stage decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HoehnYahrDecoder {
    pub num_neurons: usize,
}

impl HoehnYahrDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for HoehnYahrDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Stage 0-5 (lower activity = higher stage)
            let stage = ((1.0 - mean_rate) * 5.0).max(0.0).min(5.0);
            output[[b, 0]] = stage;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Schwab & England Activities of Daily Living decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEADLDecoder {
    pub num_neurons: usize,
}

impl SEADLDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for SEADLDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // 0-100% scale (higher activity = better function)
            output[[b, 0]] = (mean_rate * 100.0).min(100.0);
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
    fn test_updrs_decoder() {
        let decoder = UPDRSDecoder::for_part_iii(20);

        let mut spike_data = Array3::zeros((1, 50, 20));
        // Add some spikes to first group (tremor)
        for i in 0..5 {
            spike_data[[0, 10, i]] = 1.0;
            spike_data[[0, 20, i]] = 1.0;
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 4]);
        // First subscore (tremor) should be higher
        assert!(output[[0, 0]] > 0.0);
    }

    #[test]
    fn test_tremor_severity_decoder() {
        let decoder = TremorSeverityDecoder::new(100.0); // 100 Hz sampling

        let mut spike_data = Array3::zeros((1, 100, 5));
        // Create oscillatory pattern (tremor-like)
        for t in 0..100 {
            if t % 20 == 0 {
                // ~5 Hz oscillation
                spike_data[[0, t, 0]] = 1.0;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape()[1], 4); // 3 bands + overall
    }

    #[test]
    fn test_gait_score_decoder() {
        let decoder = GaitScoreDecoder::new(4);

        let mut spike_data = Array3::zeros((1, 50, 20));
        // Add regular spike pattern for good gait
        for t in (0..50).step_by(5) {
            for n in 0..4 {
                spike_data[[0, t, n]] = 1.0;
            }
        }

        let spikes = SpikeTensor::from_dense(spike_data, false);
        let output = decoder.decode(&spikes).unwrap();

        assert_eq!(output.shape(), &[1, 5]); // 5 gait features
        // Should have non-zero scores
        assert!(output.iter().any(|&x| x > 0.0));
    }
}
