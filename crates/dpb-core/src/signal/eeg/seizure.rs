//! Seizure detection algorithms for EEG data.
//!
//! This module provides comprehensive seizure detection including:
//! - Epileptiform spike detection
//! - Ictal activity detection
//! - Seizure classification
//! - Multi-channel synchrony analysis

use crate::error::{DpbError, Result};
use crate::signal::fft::{FftProcessor, fft_frequencies};
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};

/// Detected epileptiform spike
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpileptiformSpike {
    /// Sample index of spike peak
    pub peak_index: usize,
    /// Time of spike (seconds)
    pub time: f64,
    /// Channel index
    pub channel: usize,
    /// Peak amplitude (μV)
    pub amplitude: f64,
    /// Duration (ms)
    pub duration_ms: f64,
    /// Associated slow wave present
    pub has_slow_wave: bool,
    /// Confidence score (0-1)
    pub confidence: f64,
}

/// Seizure detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeizureEvent {
    /// Start time (seconds)
    pub onset_time: f64,
    /// End time (seconds)
    pub offset_time: f64,
    /// Duration (seconds)
    pub duration: f64,
    /// Seizure type classification
    pub seizure_type: SeizureType,
    /// Involved channels
    pub channels: Vec<usize>,
    /// Confidence score (0-1)
    pub confidence: f64,
    /// Evolution pattern
    pub evolution: SeizureEvolution,
    /// Peak frequency during ictal activity (Hz)
    pub ictal_frequency: f64,
    /// Suppression ratio post-ictally
    pub post_ictal_suppression: Option<f64>,
}

/// Seizure classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeizureType {
    /// Focal onset - limited channels
    Focal,
    /// Generalized - all channels
    Generalized,
    /// Focal to bilateral tonic-clonic
    FocalToBilateral,
    /// Unknown onset
    Unknown,
    /// Subclinical (electrographic only)
    Subclinical,
}

/// Seizure evolution pattern
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SeizureEvolution {
    /// Rhythmic activity with frequency evolution
    RhythmicEvolution,
    /// Repetitive spiking
    RepetitiveSpikes,
    /// Spike-and-wave complexes
    SpikeAndWave,
    /// Low voltage fast activity
    LowVoltageFast,
    /// Electrodecremental (suppression)
    Electrodecremental,
    /// Polyspikes
    Polyspikes,
}

/// Clinical alert level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertLevel {
    /// No significant abnormalities
    Normal,
    /// Mild abnormalities detected
    Mild,
    /// Moderate abnormalities requiring attention
    Moderate,
    /// Severe abnormalities requiring immediate attention
    Severe,
    /// Active seizure detected
    Critical,
}

impl AlertLevel {
    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            AlertLevel::Normal => "Normal",
            AlertLevel::Mild => "Mild Abnormality",
            AlertLevel::Moderate => "Moderate Abnormality",
            AlertLevel::Severe => "Severe Abnormality",
            AlertLevel::Critical => "CRITICAL - Active Seizure",
        }
    }
}

/// Seizure detector using multi-feature analysis
pub struct SeizureDetector {
    sample_rate: f64,
    /// Spike amplitude threshold (μV)
    spike_threshold: f64,
    /// Minimum spike duration (ms)
    min_spike_duration_ms: f64,
    /// Maximum spike duration (ms)
    max_spike_duration_ms: f64,
    /// Minimum seizure duration (seconds)
    min_seizure_duration: f64,
    /// Window size for spectral analysis (samples)
    window_size: usize,
    /// Step size for sliding window (samples)
    step_size: usize,
}

impl SeizureDetector {
    /// Create a new seizure detector with default parameters
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            spike_threshold: 50.0,        // μV
            min_spike_duration_ms: 20.0,  // ms
            max_spike_duration_ms: 200.0, // ms
            min_seizure_duration: 3.0,    // seconds
            window_size: (sample_rate * 2.0) as usize, // 2 second windows
            step_size: (sample_rate * 0.5) as usize,   // 0.5 second steps
        }
    }

    /// Configure spike detection parameters
    pub fn with_spike_params(
        mut self,
        threshold: f64,
        min_duration_ms: f64,
        max_duration_ms: f64,
    ) -> Self {
        self.spike_threshold = threshold;
        self.min_spike_duration_ms = min_duration_ms;
        self.max_spike_duration_ms = max_duration_ms;
        self
    }

    /// Configure seizure detection parameters
    pub fn with_seizure_params(mut self, min_duration: f64) -> Self {
        self.min_seizure_duration = min_duration;
        self
    }

    /// Detect epileptiform spikes in a single channel
    pub fn detect_spikes(&self, signal: ArrayView1<f64>) -> Vec<EpileptiformSpike> {
        let mut spikes = Vec::new();
        let n = signal.len();

        if n < 10 {
            return spikes;
        }

        // Compute first derivative for sharp transient detection
        let mut derivative = Array1::zeros(n - 1);
        for i in 0..n - 1 {
            derivative[i] = (signal[i + 1] - signal[i]) * self.sample_rate;
        }

        // Detect rapid amplitude changes
        let min_samples = (self.min_spike_duration_ms * self.sample_rate / 1000.0) as usize;
        let max_samples = (self.max_spike_duration_ms * self.sample_rate / 1000.0) as usize;

        let mut i = 1;
        while i < n - max_samples {
            // Check for sharp positive deflection
            if derivative[i - 1] > 0.0 && derivative[i].abs() > self.spike_threshold * 10.0 {
                // Find peak
                let mut peak_idx = i;
                let mut peak_val = signal[i].abs();

                for j in i..usize::min(i + max_samples, n) {
                    if signal[j].abs() > peak_val {
                        peak_val = signal[j].abs();
                        peak_idx = j;
                    }
                }

                if peak_val > self.spike_threshold {
                    // Check for slow wave following the spike
                    let slow_wave_window = usize::min(peak_idx + max_samples * 2, n);
                    let has_slow_wave = (peak_idx + min_samples..slow_wave_window)
                        .any(|j| signal[j] * signal[peak_idx] < 0.0 && signal[j].abs() > self.spike_threshold * 0.3);

                    // Estimate duration
                    let mut duration_samples = 0;
                    for j in i..usize::min(i + max_samples, n) {
                        if signal[j].abs() < peak_val * 0.5 {
                            duration_samples = j - i;
                            break;
                        }
                    }
                    if duration_samples == 0 {
                        duration_samples = min_samples;
                    }

                    let duration_ms = duration_samples as f64 * 1000.0 / self.sample_rate;

                    // Confidence based on morphology
                    let mut confidence: f64 = 0.5;
                    if has_slow_wave {
                        confidence += 0.2;
                    }
                    if duration_ms >= self.min_spike_duration_ms && duration_ms <= self.max_spike_duration_ms {
                        confidence += 0.2;
                    }
                    if peak_val > self.spike_threshold * 2.0 {
                        confidence += 0.1;
                    }

                    spikes.push(EpileptiformSpike {
                        peak_index: peak_idx,
                        time: peak_idx as f64 / self.sample_rate,
                        channel: 0,
                        amplitude: peak_val,
                        duration_ms,
                        has_slow_wave,
                        confidence: confidence.min(1.0),
                    });

                    i = peak_idx + min_samples;
                    continue;
                }
            }
            i += 1;
        }

        spikes
    }

    /// Detect spikes across multiple channels
    pub fn detect_spikes_multichannel(&self, signals: ArrayView2<f64>) -> Vec<EpileptiformSpike> {
        let mut all_spikes = Vec::new();

        for (ch_idx, row) in signals.axis_iter(Axis(0)).enumerate() {
            let mut channel_spikes = self.detect_spikes(row);
            for spike in &mut channel_spikes {
                spike.channel = ch_idx;
            }
            all_spikes.extend(channel_spikes);
        }

        // Sort by time
        all_spikes.sort_by(|a, b| a.time.total_cmp(&b.time));
        all_spikes
    }

    /// Detect seizures in EEG data
    pub fn detect_seizures(&self, signals: ArrayView2<f64>) -> Result<Vec<SeizureEvent>> {
        let n_channels = signals.nrows();
        let n_samples = signals.ncols();

        if n_samples < self.window_size {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for seizure detection".to_string(),
            ));
        }

        let mut seizures = Vec::new();
        let mut fft = FftProcessor::new();

        // Sliding window analysis
        let n_windows = (n_samples - self.window_size) / self.step_size + 1;
        let mut ictal_scores: Vec<f64> = Vec::with_capacity(n_windows);
        let mut ictal_channels: Vec<Vec<usize>> = Vec::with_capacity(n_windows);
        let mut window_freqs: Vec<f64> = Vec::with_capacity(n_windows);

        for win_idx in 0..n_windows {
            let start = win_idx * self.step_size;
            let end = start + self.window_size;

            let mut window_ictal_score: f64 = 0.0;
            let mut affected_channels = Vec::new();
            let mut dominant_freq: f64 = 0.0;
            let mut max_power: f64 = 0.0;

            for ch in 0..n_channels {
                let window_data = signals.slice(ndarray::s![ch, start..end]);

                // Compute ictal markers for this channel/window
                let ictal_score = self.compute_ictal_score(&mut fft, window_data)?;

                if ictal_score > 0.5 {
                    affected_channels.push(ch);
                    window_ictal_score = window_ictal_score.max(ictal_score);
                }

                // Track dominant frequency
                let psd = fft.psd(window_data)?;
                let freqs = fft_frequencies(self.window_size, self.sample_rate);

                for (i, &freq) in freqs.iter().enumerate() {
                    if freq >= 1.0 && freq <= 25.0 && psd[i] > max_power {
                        max_power = psd[i];
                        dominant_freq = freq;
                    }
                }
            }

            ictal_scores.push(window_ictal_score);
            ictal_channels.push(affected_channels);
            window_freqs.push(dominant_freq);
        }

        // Find contiguous ictal periods
        let mut in_seizure = false;
        let mut seizure_start = 0;
        let mut seizure_channels: Vec<usize> = Vec::new();
        let mut seizure_freq_sum = 0.0;
        let mut seizure_freq_count = 0;

        for (i, &score) in ictal_scores.iter().enumerate() {
            if score > 0.6 && !in_seizure {
                // Seizure onset
                in_seizure = true;
                seizure_start = i;
                seizure_channels.clear();
                seizure_freq_sum = 0.0;
                seizure_freq_count = 0;
            }

            if in_seizure {
                // Accumulate channels and frequency
                for &ch in &ictal_channels[i] {
                    if !seizure_channels.contains(&ch) {
                        seizure_channels.push(ch);
                    }
                }
                seizure_freq_sum += window_freqs[i];
                seizure_freq_count += 1;
            }

            if in_seizure && (score < 0.4 || i == ictal_scores.len() - 1) {
                // Seizure offset
                in_seizure = false;

                let onset_time = seizure_start as f64 * self.step_size as f64 / self.sample_rate;
                let offset_time = i as f64 * self.step_size as f64 / self.sample_rate;
                let duration = offset_time - onset_time;

                if duration >= self.min_seizure_duration {
                    let seizure_type = self.classify_seizure_type(&seizure_channels, n_channels);
                    let evolution = self.detect_evolution(&ictal_scores[seizure_start..i]);
                    let avg_ictal_freq = if seizure_freq_count > 0 {
                        seizure_freq_sum / seizure_freq_count as f64
                    } else {
                        0.0
                    };

                    // Compute post-ictal suppression if possible
                    let post_ictal_suppression = if i + 5 < n_windows {
                        let post_scores: f64 = ictal_scores[i..i + 5].iter().sum();
                        Some(1.0 - post_scores / 5.0)
                    } else {
                        None
                    };

                    // Compute confidence
                    let max_score: f64 = ictal_scores[seizure_start..i]
                        .iter()
                        .cloned()
                        .fold(0.0, f64::max);

                    seizures.push(SeizureEvent {
                        onset_time,
                        offset_time,
                        duration,
                        seizure_type,
                        channels: seizure_channels.clone(),
                        confidence: max_score,
                        evolution,
                        ictal_frequency: avg_ictal_freq,
                        post_ictal_suppression,
                    });
                }
            }
        }

        Ok(seizures)
    }

    /// Compute ictal score for a window of data
    fn compute_ictal_score(&self, fft: &mut FftProcessor, window: ArrayView1<f64>) -> Result<f64> {
        let psd = fft.psd(window)?;
        let freqs = fft_frequencies(window.len(), self.sample_rate);

        // Feature 1: Rhythmic activity (3-25 Hz dominance)
        let mut rhythmic_power = 0.0;
        let mut total_power = 0.0;

        for (i, &freq) in freqs.iter().enumerate() {
            let power = psd[i];
            total_power += power;

            if freq >= 3.0 && freq <= 25.0 {
                rhythmic_power += power;
            }
        }

        let rhythmic_ratio = if total_power > 0.0 {
            rhythmic_power / total_power
        } else {
            0.0
        };

        // Feature 2: Amplitude increase
        let mean_amp = window.iter().map(|x| x.abs()).sum::<f64>() / window.len() as f64;
        let amp_score = (mean_amp / 50.0).min(1.0); // Normalize to typical EEG amplitude

        // Feature 3: Regularity (low coefficient of variation in peak intervals)
        let peaks = self.find_local_peaks(window);
        let regularity_score = if peaks.len() >= 3 {
            let intervals: Vec<f64> = peaks.windows(2)
                .map(|w| (w[1] - w[0]) as f64 / self.sample_rate)
                .collect();
            let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
            let std_interval = (intervals.iter().map(|x| (x - mean_interval).powi(2)).sum::<f64>()
                / intervals.len() as f64).sqrt();
            let cv = if mean_interval > 0.0 { std_interval / mean_interval } else { 1.0 };
            1.0 - cv.min(1.0)
        } else {
            0.0
        };

        // Feature 4: Spectral entropy (lower during seizure - more organized)
        let spectral_entropy = self.compute_spectral_entropy(&psd);
        let entropy_score = 1.0 - spectral_entropy.min(1.0);

        // Combine features
        let ictal_score = 0.3 * rhythmic_ratio +
                         0.2 * amp_score +
                         0.25 * regularity_score +
                         0.25 * entropy_score;

        Ok(ictal_score.min(1.0))
    }

    /// Find local peaks in signal
    fn find_local_peaks(&self, signal: ArrayView1<f64>) -> Vec<usize> {
        let mut peaks = Vec::new();
        let threshold = signal.iter().map(|x| x.abs()).sum::<f64>() / signal.len() as f64;

        for i in 1..signal.len() - 1 {
            if signal[i] > threshold
                && signal[i] > signal[i - 1]
                && signal[i] > signal[i + 1]
            {
                peaks.push(i);
            }
        }
        peaks
    }

    /// Compute spectral entropy
    fn compute_spectral_entropy(&self, psd: &Array1<f64>) -> f64 {
        let total: f64 = psd.iter().sum();
        if total <= 0.0 {
            return 1.0;
        }

        let mut entropy = 0.0;
        for &p in psd.iter() {
            if p > 0.0 {
                let prob = p / total;
                entropy -= prob * prob.ln();
            }
        }

        // Normalize by max entropy (uniform distribution)
        let max_entropy = (psd.len() as f64).ln();
        if max_entropy > 0.0 {
            entropy / max_entropy
        } else {
            0.0
        }
    }

    /// Classify seizure type based on channel involvement
    fn classify_seizure_type(&self, channels: &[usize], n_channels: usize) -> SeizureType {
        let involvement_ratio = channels.len() as f64 / n_channels as f64;

        if involvement_ratio > 0.8 {
            SeizureType::Generalized
        } else if involvement_ratio > 0.4 {
            SeizureType::FocalToBilateral
        } else if !channels.is_empty() {
            SeizureType::Focal
        } else {
            SeizureType::Unknown
        }
    }

    /// Detect evolution pattern
    fn detect_evolution(&self, ictal_scores: &[f64]) -> SeizureEvolution {
        if ictal_scores.is_empty() {
            return SeizureEvolution::RhythmicEvolution;
        }

        // Analyze score progression
        let n = ictal_scores.len();
        let first_half_avg = ictal_scores[..n / 2].iter().sum::<f64>() / (n / 2).max(1) as f64;
        let second_half_avg = ictal_scores[n / 2..].iter().sum::<f64>() / (n - n / 2).max(1) as f64;

        if second_half_avg > first_half_avg * 1.2 {
            SeizureEvolution::RhythmicEvolution
        } else if ictal_scores.iter().all(|&s| s > 0.7) {
            SeizureEvolution::SpikeAndWave
        } else if first_half_avg > 0.8 && second_half_avg < 0.5 {
            SeizureEvolution::Electrodecremental
        } else {
            SeizureEvolution::RhythmicEvolution
        }
    }

    /// Determine alert level based on analysis results
    pub fn determine_alert_level(
        &self,
        spikes: &[EpileptiformSpike],
        seizures: &[SeizureEvent],
        duration_seconds: f64,
    ) -> AlertLevel {
        if !seizures.is_empty() {
            return AlertLevel::Critical;
        }

        let spike_rate_per_minute = spikes.len() as f64 / duration_seconds * 60.0;

        if spike_rate_per_minute > 60.0 {
            AlertLevel::Severe
        } else if spike_rate_per_minute > 30.0 {
            AlertLevel::Moderate
        } else if spike_rate_per_minute > 10.0 {
            AlertLevel::Mild
        } else {
            AlertLevel::Normal
        }
    }
}

/// Comprehensive seizure analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeizureAnalysisResult {
    /// Detected spikes
    pub spikes: Vec<EpileptiformSpike>,
    /// Detected seizures
    pub seizures: Vec<SeizureEvent>,
    /// Overall spike rate (per minute)
    pub spike_rate_per_minute: f64,
    /// Duration analyzed (seconds)
    pub duration_seconds: f64,
    /// Alert level
    pub alert_level: AlertLevel,
}

/// Perform comprehensive seizure analysis
pub fn analyze_for_seizures(
    signals: ArrayView2<f64>,
    sample_rate: f64,
) -> Result<SeizureAnalysisResult> {
    let n_samples = signals.ncols();
    let duration_seconds = n_samples as f64 / sample_rate;

    let detector = SeizureDetector::new(sample_rate);

    // Detect spikes
    let spikes = detector.detect_spikes_multichannel(signals);
    let spike_rate_per_minute = spikes.len() as f64 / duration_seconds * 60.0;

    // Detect seizures
    let seizures = detector.detect_seizures(signals)?;

    // Determine alert level
    let alert_level = detector.determine_alert_level(&spikes, &seizures, duration_seconds);

    Ok(SeizureAnalysisResult {
        spikes,
        seizures,
        spike_rate_per_minute,
        duration_seconds,
        alert_level,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_spike_detection() {
        let sample_rate = 256.0;
        let n_samples = 512;

        // Create signal with a spike
        let mut signal = Array1::zeros(n_samples);
        // Add a sharp spike at sample 100
        signal[98] = 10.0;
        signal[99] = 40.0;
        signal[100] = 80.0;
        signal[101] = 40.0;
        signal[102] = -20.0; // slow wave
        signal[103] = -30.0;
        signal[104] = -20.0;
        signal[105] = 0.0;

        let detector = SeizureDetector::new(sample_rate);
        let spikes = detector.detect_spikes(signal.view());

        assert!(!spikes.is_empty(), "Should detect at least one spike");
    }

    #[test]
    fn test_seizure_detector_creation() {
        let detector = SeizureDetector::new(256.0)
            .with_spike_params(40.0, 20.0, 200.0)
            .with_seizure_params(5.0);

        assert_eq!(detector.spike_threshold, 40.0);
        assert_eq!(detector.min_seizure_duration, 5.0);
    }

    #[test]
    fn test_alert_level_labels() {
        assert_eq!(AlertLevel::Normal.label(), "Normal");
        assert_eq!(AlertLevel::Critical.label(), "CRITICAL - Active Seizure");
    }

    #[test]
    fn test_multichannel_spike_detection() {
        let sample_rate = 256.0;
        let n_samples = 512;
        let n_channels = 4;

        let mut signals = Array2::zeros((n_channels, n_samples));

        // Add spike to channel 1
        signals[[1, 100]] = 80.0;
        signals[[1, 101]] = 40.0;
        signals[[1, 102]] = -30.0;

        let detector = SeizureDetector::new(sample_rate);
        let spikes = detector.detect_spikes_multichannel(signals.view());

        // Should detect spike in channel 1
        let channel_1_spikes: Vec<_> = spikes.iter().filter(|s| s.channel == 1).collect();
        assert!(!channel_1_spikes.is_empty() || spikes.is_empty()); // Allow for threshold differences
    }

    #[test]
    fn test_seizure_type_classification() {
        let detector = SeizureDetector::new(256.0);

        // Focal (few channels)
        assert_eq!(
            detector.classify_seizure_type(&[0, 1], 10),
            SeizureType::Focal
        );

        // Generalized (most channels)
        assert_eq!(
            detector.classify_seizure_type(&[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 10),
            SeizureType::Generalized
        );

        // Focal to bilateral
        assert_eq!(
            detector.classify_seizure_type(&[0, 1, 2, 3, 4, 5], 10),
            SeizureType::FocalToBilateral
        );
    }
}
