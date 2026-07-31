//! Clinical score decoders for Parkinson's disease and movement disorders.
//!
//! # ⚠️ Intended use — research and education only
//!
//! **The values produced by these decoders are model estimates, not clinical
//! scores.** They are named after published rating scales because they are
//! modelled after those constructs — they are not equivalent to an administered
//! assessment, they have not been validated against one, and they must never be
//! recorded, reported, or interpreted as a clinical score.
//!
//! This software is not a medical device. It is not FDA-cleared or CE-marked and
//! has not been validated for diagnosis, treatment, monitoring, or any clinical
//! decision.
//!
//! MDS-UPDRS, Berg Balance Scale, Tinetti, TUG, PDQ-39 and other named
//! instruments are marks of their respective owners; use here is descriptive and
//! implies no endorsement, affiliation or license. No instrument item content or
//! official scoring form is reproduced.

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

// ============================================================================
// Additional Balance Decoders (Phase E)
// ============================================================================

/// Tinetti Performance-Oriented Mobility Assessment decoder
/// Assesses both gait (12 points) and balance (16 points) for total 0-28 score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TinettiDecoder {
    pub num_neurons: usize,
    /// Neurons allocated to gait assessment
    pub gait_neurons: usize,
    /// Neurons allocated to balance assessment
    pub balance_neurons: usize,
}

impl TinettiDecoder {
    pub fn new(num_neurons: usize) -> Self {
        let gait_neurons = num_neurons * 12 / 28; // Proportional to score range
        let balance_neurons = num_neurons - gait_neurons;
        Self {
            num_neurons,
            gait_neurons,
            balance_neurons,
        }
    }
}

impl Decoder for TinettiDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 3)); // gait, balance, total

        for b in 0..batch_size {
            // Gait score (0-12): regularity of spike patterns
            let gait_rates: Vec<f32> = rates.slice(s![b, 0..self.gait_neurons.min(rates.shape()[1])]).to_vec();
            let gait_regularity = if gait_rates.len() >= 2 {
                let mean = gait_rates.iter().sum::<f32>() / gait_rates.len() as f32;
                let std = (gait_rates.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / gait_rates.len() as f32).sqrt();
                1.0 - (std / (mean + 0.01)).min(1.0)
            } else {
                0.5
            };
            let gait_score = (gait_regularity * 12.0).max(0.0).min(12.0);

            // Balance score (0-16): stability of spike patterns
            let start = self.gait_neurons.min(rates.shape()[1]);
            let end = rates.shape()[1];
            let balance_rates: Vec<f32> = rates.slice(s![b, start..end]).to_vec();
            let balance_stability = if !balance_rates.is_empty() {
                let mean = balance_rates.iter().sum::<f32>() / balance_rates.len() as f32;
                let variance = balance_rates.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / balance_rates.len() as f32;
                1.0 - variance.sqrt().min(1.0)
            } else {
                0.5
            };
            let balance_score = (balance_stability * 16.0).max(0.0).min(16.0);

            output[[b, 0]] = gait_score;
            output[[b, 1]] = balance_score;
            output[[b, 2]] = gait_score + balance_score; // Total (0-28)
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        3 // gait, balance, total
    }
}

/// Mini-BESTest (Balance Evaluation Systems Test) decoder
/// Assesses 4 balance domains: anticipatory, reactive, sensory, dynamic gait
/// Score: 0-32 (each domain 0-8)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBESTDecoder {
    pub num_neurons: usize,
}

impl MiniBESTDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for MiniBESTDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 5)); // 4 domains + total

        let neurons_per_domain = (rates.shape()[1] / 4).max(1);

        for b in 0..batch_size {
            let mut total = 0.0;

            for domain in 0..4 {
                let start = domain * neurons_per_domain;
                let end = ((domain + 1) * neurons_per_domain).min(rates.shape()[1]);

                if start < end {
                    let domain_rates: Vec<f32> = rates.slice(s![b, start..end]).to_vec();
                    let mean = domain_rates.iter().sum::<f32>() / domain_rates.len() as f32;
                    let variance = domain_rates.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / domain_rates.len() as f32;

                    // Higher stability = better balance
                    let domain_score = match domain {
                        0 => mean * 8.0,                          // Anticipatory: activity level
                        1 => (1.0 - variance.sqrt()) * 8.0,       // Reactive: consistency
                        2 => ((mean + 1.0 - variance) * 4.0).max(0.0), // Sensory: combined
                        3 => {                                     // Dynamic gait: regularity
                            let regularity = 1.0 - variance.sqrt();
                            regularity * 8.0
                        }
                        _ => 0.0,
                    };

                    output[[b, domain]] = domain_score.max(0.0).min(8.0);
                    total += output[[b, domain]];
                }
            }

            output[[b, 4]] = total.min(32.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5 // 4 domains + total
    }
}

// ============================================================================
// Pain Decoders (Phase E)
// ============================================================================

/// Visual Analog Scale (VAS) decoder for pain intensity
/// Output: 0-100 continuous scale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VasDecoder {
    pub num_neurons: usize,
    /// Whether high spike activity indicates high pain
    pub high_activity_is_high_pain: bool,
}

impl VasDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            high_activity_is_high_pain: true,
        }
    }
}

impl Decoder for VasDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);

            let vas = if self.high_activity_is_high_pain {
                mean_rate * 100.0
            } else {
                (1.0 - mean_rate) * 100.0
            };

            output[[b, 0]] = vas.max(0.0).min(100.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// Numeric Rating Scale (NRS) decoder for pain intensity
/// Output: 0-10 integer scale
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NrsDecoder {
    pub num_neurons: usize,
}

impl NrsDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for NrsDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        let mut output = Array2::zeros((batch_size, 1));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            // Round to nearest integer 0-10
            let nrs = (mean_rate * 10.0).round().max(0.0).min(10.0);
            output[[b, 0]] = nrs;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        1
    }
}

/// QST Sensory Phenotype decoder
/// Classifies into: Normal, SensoryLoss, ThermalHyperalgesia, MechanicalHyperalgesia, Mixed
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QstPhenotypeDecoder {
    pub num_neurons: usize,
    /// Number of QST modalities (typically 8-13)
    pub num_modalities: usize,
}

impl QstPhenotypeDecoder {
    pub fn new(num_neurons: usize, num_modalities: usize) -> Self {
        Self {
            num_neurons,
            num_modalities,
        }
    }

    /// Default with standard 8 QST modalities
    pub fn standard(num_neurons: usize) -> Self {
        Self::new(num_neurons, 8)
    }
}

impl Decoder for QstPhenotypeDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: 5 phenotype probabilities + dominant phenotype index
        let mut output = Array2::zeros((batch_size, 6));

        let neurons_per_modality = (rates.shape()[1] / self.num_modalities).max(1);

        for b in 0..batch_size {
            // Calculate Z-scores for each modality group
            let mut thermal_loss = 0.0_f32;
            let mut thermal_gain = 0.0_f32;
            let mut mech_loss = 0.0_f32;
            let mut mech_gain = 0.0_f32;

            for m in 0..self.num_modalities.min(8) {
                let start = m * neurons_per_modality;
                let end = ((m + 1) * neurons_per_modality).min(rates.shape()[1]);

                if start < end {
                    let modality_rates: Vec<f32> = rates.slice(s![b, start..end]).to_vec();
                    let mean = modality_rates.iter().sum::<f32>() / modality_rates.len() as f32;

                    // Simplified Z-score interpretation
                    let z_score = (mean - 0.5) * 4.0; // Centered at 0.5, scaled

                    match m {
                        0..=3 => { // Thermal modalities
                            if z_score < -1.0 {
                                thermal_loss += z_score.abs();
                            } else if z_score > 1.0 {
                                thermal_gain += z_score;
                            }
                        }
                        4..=7 => { // Mechanical modalities
                            if z_score < -1.0 {
                                mech_loss += z_score.abs();
                            } else if z_score > 1.0 {
                                mech_gain += z_score;
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Calculate phenotype probabilities (softmax-like)
            let total = thermal_loss + thermal_gain + mech_loss + mech_gain + 1.0;
            output[[b, 0]] = 1.0 / total;                    // Normal
            output[[b, 1]] = thermal_loss / total;           // Sensory loss
            output[[b, 2]] = thermal_gain / total;           // Thermal hyperalgesia
            output[[b, 3]] = mech_gain / total;              // Mechanical hyperalgesia
            output[[b, 4]] = (thermal_gain + mech_gain) / total; // Mixed

            // Dominant phenotype (argmax)
            let phenotypes = [output[[b, 0]], output[[b, 1]], output[[b, 2]], output[[b, 3]], output[[b, 4]]];
            let max_idx = phenotypes.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap_or(0);
            output[[b, 5]] = max_idx as f32;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        6 // 5 phenotype probabilities + dominant index
    }
}

// ============================================================================
// Vestibular Decoders (Phase E)
// ============================================================================

/// VOR (Vestibulo-Ocular Reflex) Gain decoder
/// Output: VOR gain (normal ~0.8-1.0)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VorGainDecoder {
    pub num_neurons: usize,
    /// Neurons for head velocity encoding
    pub head_neurons: usize,
    /// Neurons for eye velocity encoding
    pub eye_neurons: usize,
}

impl VorGainDecoder {
    pub fn new(num_neurons: usize) -> Self {
        let head_neurons = num_neurons / 2;
        let eye_neurons = num_neurons - head_neurons;
        Self {
            num_neurons,
            head_neurons,
            eye_neurons,
        }
    }
}

impl Decoder for VorGainDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: VOR gain, asymmetry, left gain, right gain
        let mut output = Array2::zeros((batch_size, 4));

        for b in 0..batch_size {
            // Head velocity neurons (first half)
            let head_end = self.head_neurons.min(rates.shape()[1]);
            let head_rates: Vec<f32> = rates.slice(s![b, 0..head_end]).to_vec();
            let head_activity = if !head_rates.is_empty() {
                head_rates.iter().sum::<f32>() / head_rates.len() as f32
            } else {
                0.5
            };

            // Eye velocity neurons (second half)
            let eye_start = head_end;
            let eye_end = rates.shape()[1];
            let eye_rates: Vec<f32> = rates.slice(s![b, eye_start..eye_end]).to_vec();
            let eye_activity = if !eye_rates.is_empty() {
                eye_rates.iter().sum::<f32>() / eye_rates.len() as f32
            } else {
                0.5
            };

            // VOR gain = eye velocity / head velocity
            let vor_gain = if head_activity > 0.01 {
                eye_activity / head_activity
            } else {
                1.0
            };

            // Asymmetry (difference between left and right responses)
            let mid_eye = (eye_start + eye_end) / 2;
            let left_eye: f32 = rates.slice(s![b, eye_start..mid_eye]).mean().unwrap_or(0.5);
            let right_eye: f32 = rates.slice(s![b, mid_eye..eye_end]).mean().unwrap_or(0.5);

            let left_gain = if head_activity > 0.01 { left_eye / head_activity } else { 1.0 };
            let right_gain = if head_activity > 0.01 { right_eye / head_activity } else { 1.0 };
            let asymmetry = ((left_gain - right_gain) / (left_gain + right_gain + 0.01) * 100.0).abs();

            output[[b, 0]] = vor_gain.max(0.0).min(2.0);
            output[[b, 1]] = asymmetry.max(0.0).min(100.0);
            output[[b, 2]] = left_gain.max(0.0).min(2.0);
            output[[b, 3]] = right_gain.max(0.0).min(2.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4 // gain, asymmetry, left, right
    }
}

/// Canal Paresis decoder (caloric test asymmetry)
/// Output: Canal paresis percentage (normal <20-25%)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanalParesisDecoder {
    pub num_neurons: usize,
}

impl CanalParesisDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for CanalParesisDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: CP%, directional preponderance, left warm, right warm, left cool, right cool
        let mut output = Array2::zeros((batch_size, 6));

        let quarter = (rates.shape()[1] / 4).max(1);

        for b in 0..batch_size {
            // Jongkees formula: CP = (RW + RC - LW - LC) / (RW + RC + LW + LC) × 100
            // Where: RW=right warm, RC=right cool, LW=left warm, LC=left cool

            let lw: f32 = rates.slice(s![b, 0..quarter]).mean().unwrap_or(0.0);
            let lc: f32 = rates.slice(s![b, quarter..quarter*2]).mean().unwrap_or(0.0);
            let rw: f32 = rates.slice(s![b, quarter*2..quarter*3]).mean().unwrap_or(0.0);
            let rc: f32 = rates.slice(s![b, quarter*3..]).mean().unwrap_or(0.0);

            let total = lw + lc + rw + rc + 0.001; // Avoid division by zero
            let cp = ((rw + rc - lw - lc) / total * 100.0).abs();

            // Directional preponderance: (RW + LC - LW - RC) / total × 100
            let dp = ((rw + lc - lw - rc) / total * 100.0).abs();

            output[[b, 0]] = cp.min(100.0);
            output[[b, 1]] = dp.min(100.0);
            output[[b, 2]] = lw * 100.0; // Scale to SPV-like values
            output[[b, 3]] = rw * 100.0;
            output[[b, 4]] = lc * 100.0;
            output[[b, 5]] = rc * 100.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        6 // CP, DP, LW, RW, LC, RC
    }
}

/// BPPV (Benign Paroxysmal Positional Vertigo) decoder
/// Classifies BPPV presence and affected canal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BppvDecoder {
    pub num_neurons: usize,
}

impl BppvDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for BppvDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: BPPV probability, canal type (0-5 for 6 semicircular canals)
        let mut output = Array2::zeros((batch_size, 7));

        for b in 0..batch_size {
            // Detect characteristic BPPV patterns:
            // - Latency (1-5 seconds)
            // - Crescendo-decrescendo pattern
            // - Duration (<60 seconds)
            // - Fatigability

            let mut has_latency = false;
            let mut has_pattern = false;
            let mut onset_time = 0;

            // Find response onset (latency detection)
            for t in 0..num_steps {
                let time_activity: f32 = spike_dense.slice(s![b, t, ..]).sum();
                if time_activity > 0.5 && onset_time == 0 {
                    onset_time = t;
                    // Check if onset is delayed (BPPV typically has 1-5s latency)
                    if t > num_steps / 20 && t < num_steps / 4 {
                        has_latency = true;
                    }
                }
            }

            // Check for crescendo-decrescendo pattern
            if onset_time > 0 {
                let mut prev_activity = 0.0_f32;
                let mut increasing = true;
                let mut peak_found = false;

                for t in onset_time..num_steps {
                    let activity: f32 = spike_dense.slice(s![b, t, ..]).sum();
                    if increasing && activity < prev_activity {
                        peak_found = true;
                        increasing = false;
                    }
                    prev_activity = activity;
                }

                has_pattern = peak_found && !increasing;
            }

            // BPPV probability based on features
            let bppv_prob = if has_latency && has_pattern {
                0.9
            } else if has_latency || has_pattern {
                0.5
            } else {
                0.1
            };

            output[[b, 0]] = bppv_prob;

            // Canal classification based on neuron group activity
            let neurons_per_canal = (num_neurons / 6).max(1);
            for canal in 0..6 {
                let start = canal * neurons_per_canal;
                let end = ((canal + 1) * neurons_per_canal).min(num_neurons);
                let canal_activity: f32 = spike_dense.slice(s![b, .., start..end]).sum();
                output[[b, canal + 1]] = canal_activity / num_steps as f32;
            }
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        7 // BPPV prob + 6 canal activities
    }
}

// ============================================================================
// Force Decoders (Gap Fill)
// ============================================================================

/// Ground Reaction Force (GRF) decoder
/// Decodes vertical GRF, loading rate, and symmetry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrfDecoder {
    pub num_neurons: usize,
}

impl GrfDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for GrfDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: peak force (normalized), loading rate, unloading rate, symmetry index
        let mut output = Array2::zeros((batch_size, 4));

        let half = rates.shape()[1] / 2;

        for b in 0..batch_size {
            // Left leg neurons (first half)
            let left_mean: f32 = rates.slice(s![b, 0..half]).mean().unwrap_or(0.0);
            // Right leg neurons (second half)
            let right_mean: f32 = rates.slice(s![b, half..]).mean().unwrap_or(0.0);

            // Peak force estimation (normalized to body weight)
            let peak_force = (left_mean + right_mean) * 2.0; // Scale to ~1-2x body weight

            // Loading rate from spike rate variance
            let variance = rates.row(b).var(0.0);
            let loading_rate = variance * 100.0; // BW/s approximation

            // Symmetry index
            let symmetry = if left_mean + right_mean > 0.01 {
                (left_mean - right_mean).abs() / (left_mean + right_mean) * 100.0
            } else {
                0.0
            };

            output[[b, 0]] = peak_force.max(0.0).min(3.0);
            output[[b, 1]] = loading_rate.max(0.0).min(200.0);
            output[[b, 2]] = loading_rate * 0.8; // Unloading typically slower
            output[[b, 3]] = symmetry.max(0.0).min(100.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4 // peak force, loading rate, unloading rate, symmetry
    }
}

/// Grip Strength decoder
/// Decodes maximum grip force and fatigue metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GripStrengthDecoder {
    pub num_neurons: usize,
    /// Expected max grip (kg) for normalization
    pub normalization_max: f32,
}

impl GripStrengthDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            normalization_max: 50.0, // Typical adult max
        }
    }

    pub fn with_normalization(num_neurons: usize, max_grip: f32) -> Self {
        Self {
            num_neurons,
            normalization_max: max_grip,
        }
    }
}

impl Decoder for GripStrengthDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: max grip (kg), time to max, fatigue index, grip variability
        let mut output = Array2::zeros((batch_size, 4));

        for b in 0..batch_size {
            let mut max_activity = 0.0_f32;
            let mut time_to_max = 0;
            let mut activities = Vec::new();

            for t in 0..num_steps {
                let activity: f32 = spike_dense.slice(s![b, t, ..]).mean().unwrap_or(0.0);
                activities.push(activity);
                if activity > max_activity {
                    max_activity = activity;
                    time_to_max = t;
                }
            }

            // Max grip in kg
            let max_grip = max_activity * self.normalization_max;

            // Fatigue index: (initial - final) / initial
            let initial = activities.iter().take(num_steps / 4).sum::<f32>()
                / (num_steps / 4) as f32;
            let final_val = activities.iter().skip(3 * num_steps / 4).sum::<f32>()
                / (num_steps / 4) as f32;
            let fatigue = if initial > 0.01 {
                ((initial - final_val) / initial * 100.0).max(0.0)
            } else {
                0.0
            };

            // Variability (coefficient of variation)
            let mean_activity = activities.iter().sum::<f32>() / activities.len() as f32;
            let variance = activities.iter().map(|x| (x - mean_activity).powi(2)).sum::<f32>()
                / activities.len() as f32;
            let cv = if mean_activity > 0.01 {
                variance.sqrt() / mean_activity * 100.0
            } else {
                0.0
            };

            output[[b, 0]] = max_grip.max(0.0);
            output[[b, 1]] = (time_to_max as f32 / num_steps as f32) * 100.0; // % of duration
            output[[b, 2]] = fatigue.min(100.0);
            output[[b, 3]] = cv.min(100.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4 // max grip, time to max, fatigue, variability
    }
}

/// Rate of Force Development (RFD) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfdDecoder {
    pub num_neurons: usize,
    pub sample_rate: f32, // Hz
}

impl RfdDecoder {
    pub fn new(num_neurons: usize, sample_rate: f32) -> Self {
        Self {
            num_neurons,
            sample_rate,
        }
    }
}

impl Decoder for RfdDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: RFD at 50ms, 100ms, 200ms, peak RFD
        let mut output = Array2::zeros((batch_size, 4));

        let samples_50ms = (0.05 * self.sample_rate) as usize;
        let samples_100ms = (0.1 * self.sample_rate) as usize;
        let samples_200ms = (0.2 * self.sample_rate) as usize;

        for b in 0..batch_size {
            let mut activities: Vec<f32> = Vec::new();
            for t in 0..num_steps {
                activities.push(spike_dense.slice(s![b, t, ..]).mean().unwrap_or(0.0));
            }

            // Find onset (first significant activity)
            let threshold = 0.1_f32;
            let onset = activities.iter().position(|&x| x > threshold).unwrap_or(0);

            // Calculate RFD at different time points
            let rfd_50 = if onset + samples_50ms < num_steps {
                (activities[onset + samples_50ms] - activities[onset]) / 0.05
            } else {
                0.0
            };

            let rfd_100 = if onset + samples_100ms < num_steps {
                (activities[onset + samples_100ms] - activities[onset]) / 0.1
            } else {
                0.0
            };

            let rfd_200 = if onset + samples_200ms < num_steps {
                (activities[onset + samples_200ms] - activities[onset]) / 0.2
            } else {
                0.0
            };

            // Peak RFD (max instantaneous rate)
            let mut peak_rfd = 0.0_f32;
            for i in 1..activities.len() {
                let instant_rfd = (activities[i] - activities[i - 1]) * self.sample_rate;
                if instant_rfd > peak_rfd {
                    peak_rfd = instant_rfd;
                }
            }

            output[[b, 0]] = rfd_50.max(0.0) * 1000.0; // Scale to N/s
            output[[b, 1]] = rfd_100.max(0.0) * 1000.0;
            output[[b, 2]] = rfd_200.max(0.0) * 1000.0;
            output[[b, 3]] = peak_rfd.max(0.0) * 1000.0;
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4 // RFD at 50ms, 100ms, 200ms, peak
    }
}

// ============================================================================
// Cardiopulmonary Decoders (Gap Fill)
// ============================================================================

/// HRV (Heart Rate Variability) decoder
/// Decodes RMSSD, SDNN, and LF/HF ratio
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrvDecoder {
    pub num_neurons: usize,
}

impl HrvDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for HrvDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, num_neurons) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: mean HR, RMSSD, SDNN, pNN50, LF/HF ratio
        let mut output = Array2::zeros((batch_size, 5));

        for b in 0..batch_size {
            // Find spike times (representing R-peaks)
            let mut intervals = Vec::new();

            for n in 0..num_neurons {
                let mut last_spike = None;
                for t in 0..num_steps {
                    if spike_dense[[b, t, n]] > 0.5 {
                        if let Some(last) = last_spike {
                            intervals.push((t - last) as f32);
                        }
                        last_spike = Some(t);
                    }
                }
            }

            if intervals.len() < 2 {
                continue;
            }

            // Mean heart rate (assuming 1000Hz sampling -> intervals in ms)
            let mean_interval = intervals.iter().sum::<f32>() / intervals.len() as f32;
            let mean_hr = if mean_interval > 0.0 { 60000.0 / mean_interval } else { 0.0 };

            // SDNN (standard deviation of intervals)
            let variance = intervals.iter()
                .map(|&x| (x - mean_interval).powi(2))
                .sum::<f32>() / intervals.len() as f32;
            let sdnn = variance.sqrt();

            // RMSSD (root mean square of successive differences)
            let successive_diffs: Vec<f32> = intervals.windows(2)
                .map(|w| (w[1] - w[0]).powi(2))
                .collect();
            let rmssd = if !successive_diffs.is_empty() {
                (successive_diffs.iter().sum::<f32>() / successive_diffs.len() as f32).sqrt()
            } else {
                0.0
            };

            // pNN50 (percentage of successive differences > 50ms)
            let nn50_count = intervals.windows(2)
                .filter(|w| (w[1] - w[0]).abs() > 50.0)
                .count();
            let pnn50 = if intervals.len() > 1 {
                nn50_count as f32 / (intervals.len() - 1) as f32 * 100.0
            } else {
                0.0
            };

            // LF/HF ratio (simplified - based on interval variability)
            let lf_hf = if rmssd > 0.0 { sdnn / rmssd } else { 1.0 };

            output[[b, 0]] = mean_hr.max(30.0).min(200.0);
            output[[b, 1]] = rmssd.max(0.0).min(300.0);
            output[[b, 2]] = sdnn.max(0.0).min(300.0);
            output[[b, 3]] = pnn50.max(0.0).min(100.0);
            output[[b, 4]] = lf_hf.max(0.0).min(10.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5 // mean HR, RMSSD, SDNN, pNN50, LF/HF
    }
}

/// Respiratory decoder
/// Decodes respiratory rate, tidal volume, and variability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespiratoryDecoder {
    pub num_neurons: usize,
    pub sample_rate: f32,
}

impl RespiratoryDecoder {
    pub fn new(num_neurons: usize, sample_rate: f32) -> Self {
        Self { num_neurons, sample_rate }
    }
}

impl Decoder for RespiratoryDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: respiratory rate, inspiratory time, expiratory time, I:E ratio, variability
        let mut output = Array2::zeros((batch_size, 5));

        for b in 0..batch_size {
            // Sum activity over time to get respiratory pattern
            let mut pattern: Vec<f32> = Vec::new();
            for t in 0..num_steps {
                pattern.push(spike_dense.slice(s![b, t, ..]).sum());
            }

            // Find peaks (inspiration) and valleys (expiration)
            let mean_activity = pattern.iter().sum::<f32>() / pattern.len() as f32;
            let mut peaks = Vec::new();
            let mut valleys = Vec::new();

            for i in 1..pattern.len() - 1 {
                if pattern[i] > pattern[i - 1] && pattern[i] > pattern[i + 1] && pattern[i] > mean_activity {
                    peaks.push(i);
                }
                if pattern[i] < pattern[i - 1] && pattern[i] < pattern[i + 1] && pattern[i] < mean_activity {
                    valleys.push(i);
                }
            }

            // Respiratory rate from peak-to-peak intervals
            let breath_intervals: Vec<f32> = peaks.windows(2)
                .map(|w| (w[1] - w[0]) as f32 / self.sample_rate)
                .collect();

            let mean_breath_time = if !breath_intervals.is_empty() {
                breath_intervals.iter().sum::<f32>() / breath_intervals.len() as f32
            } else {
                4.0 // Default 15 breaths/min
            };

            let rr = if mean_breath_time > 0.0 { 60.0 / mean_breath_time } else { 0.0 };

            // Inspiratory and expiratory times (simplified)
            let ti = mean_breath_time * 0.4; // Typical I:E ratio ~1:1.5
            let te = mean_breath_time * 0.6;
            let ie_ratio = if te > 0.0 { ti / te } else { 0.0 };

            // Variability (CV of breath intervals)
            let variance = if !breath_intervals.is_empty() {
                breath_intervals.iter()
                    .map(|x| (x - mean_breath_time).powi(2))
                    .sum::<f32>() / breath_intervals.len() as f32
            } else {
                0.0
            };
            let cv = if mean_breath_time > 0.0 {
                variance.sqrt() / mean_breath_time * 100.0
            } else {
                0.0
            };

            output[[b, 0]] = rr.max(4.0).min(60.0);
            output[[b, 1]] = ti.max(0.0).min(5.0);
            output[[b, 2]] = te.max(0.0).min(10.0);
            output[[b, 3]] = ie_ratio.max(0.0).min(3.0);
            output[[b, 4]] = cv.max(0.0).min(100.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5 // RR, Ti, Te, I:E, variability
    }
}

/// VO2 (Oxygen Consumption) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vo2Decoder {
    pub num_neurons: usize,
    pub max_vo2: f32, // Expected max VO2 for normalization
}

impl Vo2Decoder {
    pub fn new(num_neurons: usize) -> Self {
        Self {
            num_neurons,
            max_vo2: 45.0, // ml/kg/min, typical adult
        }
    }
}

impl Decoder for Vo2Decoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: VO2 (ml/kg/min), VCO2, RER, VE
        let mut output = Array2::zeros((batch_size, 4));

        let quarter = (rates.shape()[1] / 4).max(1);

        for b in 0..batch_size {
            // VO2 from first quarter of neurons
            let vo2_activity: f32 = rates.slice(s![b, 0..quarter]).mean().unwrap_or(0.0);
            let vo2 = vo2_activity * self.max_vo2;

            // VCO2 from second quarter
            let vco2_activity: f32 = rates.slice(s![b, quarter..quarter*2]).mean().unwrap_or(0.0);
            let vco2 = vco2_activity * self.max_vo2 * 0.9; // Typically slightly less

            // RER = VCO2/VO2
            let rer = if vo2 > 0.1 { vco2 / vo2 } else { 0.85 };

            // VE from third quarter
            let ve_activity: f32 = rates.slice(s![b, quarter*2..quarter*3]).mean().unwrap_or(0.0);
            let ve = ve_activity * 150.0; // L/min

            output[[b, 0]] = vo2.max(0.0).min(80.0);
            output[[b, 1]] = vco2.max(0.0).min(80.0);
            output[[b, 2]] = rer.max(0.5).min(1.5);
            output[[b, 3]] = ve.max(0.0).min(200.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4 // VO2, VCO2, RER, VE
    }
}

// ============================================================================
// Cognitive Decoders (Gap Fill)
// ============================================================================

/// Cognitive Reaction Time decoder (with variability metrics)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveRtDecoder {
    pub num_neurons: usize,
    pub sample_rate: f32,
}

impl CognitiveRtDecoder {
    pub fn new(num_neurons: usize, sample_rate: f32) -> Self {
        Self { num_neurons, sample_rate }
    }
}

impl Decoder for CognitiveRtDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: mean RT, RT variability, fastest RT, slowest RT
        let mut output = Array2::zeros((batch_size, 4));

        for b in 0..batch_size {
            // Find response onsets
            let mut reaction_times = Vec::new();
            let threshold = 0.3_f32;
            let mut in_response = false;

            for t in 0..num_steps {
                let activity: f32 = spike_dense.slice(s![b, t, ..]).mean().unwrap_or(0.0);

                if activity > threshold && !in_response {
                    reaction_times.push(t as f32 / self.sample_rate * 1000.0); // Convert to ms
                    in_response = true;
                } else if activity < threshold * 0.5 {
                    in_response = false;
                }
            }

            if reaction_times.is_empty() {
                reaction_times.push(500.0); // Default if no response detected
            }

            let mean_rt = reaction_times.iter().sum::<f32>() / reaction_times.len() as f32;
            let variance = reaction_times.iter()
                .map(|x| (x - mean_rt).powi(2))
                .sum::<f32>() / reaction_times.len() as f32;
            let rt_sd = variance.sqrt();

            let min_rt = reaction_times.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_rt = reaction_times.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

            output[[b, 0]] = mean_rt.max(100.0).min(2000.0);
            output[[b, 1]] = rt_sd.max(0.0).min(500.0);
            output[[b, 2]] = min_rt.max(100.0).min(2000.0);
            output[[b, 3]] = max_rt.max(100.0).min(2000.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        4 // mean RT, SD, min, max
    }
}

/// Attention/CPT (Continuous Performance Test) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionDecoder {
    pub num_neurons: usize,
}

impl AttentionDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for AttentionDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: d-prime, omission rate, commission rate, mean RT, RT variability
        let mut output = Array2::zeros((batch_size, 5));

        let half = rates.shape()[1] / 2;

        for b in 0..batch_size {
            // Hit rate from first half of neurons (target detection)
            let hit_activity: f32 = rates.slice(s![b, 0..half]).mean().unwrap_or(0.0);
            let hit_rate = hit_activity.clamp(0.01, 0.99);

            // False alarm rate from second half (false positives)
            let fa_activity: f32 = rates.slice(s![b, half..]).mean().unwrap_or(0.0);
            let fa_rate = fa_activity.clamp(0.01, 0.99);

            // d-prime = Z(hit_rate) - Z(false_alarm_rate)
            // Using approximation: Z(p) ≈ 5.0 * (p - 0.5) for simple estimation
            let z_hit = 5.0 * (hit_rate - 0.5);
            let z_fa = 5.0 * (fa_rate - 0.5);
            let d_prime = (z_hit - z_fa).clamp(-4.0, 4.0);

            // Omission rate (1 - hit rate)
            let omission_rate = (1.0 - hit_rate) * 100.0;

            // Commission rate (false alarm rate)
            let commission_rate = fa_rate * 100.0;

            // RT from activity variance
            let variance = rates.row(b).var(0.0);
            let mean_rt = 300.0 + (1.0 - hit_activity) * 300.0; // 300-600ms range
            let rt_variability = variance * 100.0;

            output[[b, 0]] = d_prime;
            output[[b, 1]] = omission_rate.max(0.0).min(100.0);
            output[[b, 2]] = commission_rate.max(0.0).min(100.0);
            output[[b, 3]] = mean_rt.max(200.0).min(1000.0);
            output[[b, 4]] = rt_variability.max(0.0).min(200.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5 // d-prime, omission, commission, mean RT, RT variability
    }
}

/// Working Memory decoder (N-back performance)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkingMemoryDecoder {
    pub num_neurons: usize,
    pub n_back_level: usize,
}

impl WorkingMemoryDecoder {
    pub fn new(num_neurons: usize, n_back_level: usize) -> Self {
        Self { num_neurons, n_back_level }
    }
}

impl Decoder for WorkingMemoryDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: accuracy, d-prime, capacity estimate
        let mut output = Array2::zeros((batch_size, 3));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            let variance = rates.row(b).var(0.0);

            // Accuracy estimation
            let accuracy = mean_rate * 100.0;

            // d-prime from signal-to-noise
            let d_prime = if variance > 0.01 {
                mean_rate / variance.sqrt()
            } else {
                mean_rate * 3.0
            }.clamp(-4.0, 4.0);

            // Working memory capacity (Cowan's K approximation)
            // K = (hit_rate - false_alarm_rate) * set_size
            let capacity = mean_rate * (self.n_back_level as f32 + 2.0);

            output[[b, 0]] = accuracy.max(0.0).min(100.0);
            output[[b, 1]] = d_prime;
            output[[b, 2]] = capacity.max(0.0).min(7.0); // Miller's 7±2
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        3 // accuracy, d-prime, capacity
    }
}

// ============================================================================
// EDA (Electrodermal Activity) Decoders (Gap Fill)
// ============================================================================

/// SCR (Skin Conductance Response) decoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrDecoder {
    pub num_neurons: usize,
    pub sample_rate: f32,
}

impl ScrDecoder {
    pub fn new(num_neurons: usize, sample_rate: f32) -> Self {
        Self { num_neurons, sample_rate }
    }
}

impl Decoder for ScrDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let spike_dense = spikes.to_dense();
        let (batch_size, num_steps, _) = (
            spike_dense.shape()[0],
            spike_dense.shape()[1],
            spike_dense.shape()[2],
        );

        // Output: SCR count, mean amplitude, sum amplitude, mean rise time, latency
        let mut output = Array2::zeros((batch_size, 5));

        for b in 0..batch_size {
            // Sum activity to get EDA-like signal
            let mut signal: Vec<f32> = Vec::new();
            for t in 0..num_steps {
                signal.push(spike_dense.slice(s![b, t, ..]).sum());
            }

            // Detect SCR peaks
            let mean_level = signal.iter().sum::<f32>() / signal.len() as f32;
            let threshold = mean_level * 1.5;

            let mut scr_count = 0;
            let mut amplitudes = Vec::new();
            let mut in_scr = false;
            let mut scr_start = 0;
            let mut first_latency = None;

            for (t, &val) in signal.iter().enumerate() {
                if val > threshold && !in_scr {
                    in_scr = true;
                    scr_start = t;
                    if first_latency.is_none() {
                        first_latency = Some(t);
                    }
                } else if val <= threshold && in_scr {
                    in_scr = false;
                    scr_count += 1;

                    // Find peak in this SCR
                    let peak = signal[scr_start..t].iter().cloned().fold(0.0f32, f32::max);
                    amplitudes.push(peak - mean_level);
                }
            }

            let mean_amp = if !amplitudes.is_empty() {
                amplitudes.iter().sum::<f32>() / amplitudes.len() as f32
            } else {
                0.0
            };

            let sum_amp = amplitudes.iter().sum::<f32>();
            let latency = first_latency.map(|t| t as f32 / self.sample_rate * 1000.0).unwrap_or(0.0);

            output[[b, 0]] = scr_count as f32;
            output[[b, 1]] = mean_amp.max(0.0);
            output[[b, 2]] = sum_amp.max(0.0);
            output[[b, 3]] = 1500.0; // Typical rise time in ms (simplified)
            output[[b, 4]] = latency.max(0.0).min(5000.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        5 // count, mean amp, sum amp, rise time, latency
    }
}

/// SCL (Skin Conductance Level) decoder - tonic component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SclDecoder {
    pub num_neurons: usize,
}

impl SclDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for SclDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: mean SCL, SCL range, SCL slope
        let mut output = Array2::zeros((batch_size, 3));

        for b in 0..batch_size {
            let row = rates.row(b);
            let mean_scl = row.mean().unwrap_or(0.0) * 10.0; // Scale to microSiemens

            let min_scl = row.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_scl = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let scl_range = (max_scl - min_scl) * 10.0;

            // Slope (linear trend)
            let n = row.len() as f32;
            let x_mean = (n - 1.0) / 2.0;
            let y_mean = mean_scl / 10.0;

            let mut num = 0.0_f32;
            let mut den = 0.0_f32;
            for (i, &y) in row.iter().enumerate() {
                let x = i as f32;
                num += (x - x_mean) * (y - y_mean);
                den += (x - x_mean).powi(2);
            }

            let slope = if den > 0.0 { num / den * 1000.0 } else { 0.0 }; // microS/s

            output[[b, 0]] = mean_scl.max(0.0).min(30.0);
            output[[b, 1]] = scl_range.max(0.0).min(20.0);
            output[[b, 2]] = slope.clamp(-5.0, 5.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        3 // mean SCL, range, slope
    }
}

/// Stress Index decoder from EDA
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressIndexDecoder {
    pub num_neurons: usize,
}

impl StressIndexDecoder {
    pub fn new(num_neurons: usize) -> Self {
        Self { num_neurons }
    }
}

impl Decoder for StressIndexDecoder {
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>> {
        let rates = spikes.spike_rate();
        let batch_size = rates.shape()[0];

        // Output: stress index (0-100), arousal level, SNS activity
        let mut output = Array2::zeros((batch_size, 3));

        for b in 0..batch_size {
            let mean_rate: f32 = rates.row(b).mean().unwrap_or(0.0);
            let variance = rates.row(b).var(0.0);

            // Stress index based on activity level and variability
            // Higher activity + higher variability = higher stress
            let stress_index = ((mean_rate * 50.0) + (variance.sqrt() * 50.0)).min(100.0);

            // Arousal level (0-10 scale)
            let arousal = mean_rate * 10.0;

            // SNS activity estimate
            let sns_activity = (mean_rate * 0.6 + variance.sqrt() * 0.4) * 100.0;

            output[[b, 0]] = stress_index.max(0.0);
            output[[b, 1]] = arousal.max(0.0).min(10.0);
            output[[b, 2]] = sns_activity.max(0.0).min(100.0);
        }

        Ok(output)
    }

    fn output_dim(&self) -> usize {
        3 // stress index, arousal, SNS activity
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
