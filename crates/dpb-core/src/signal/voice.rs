//! Voice Analysis Module
//!
//! Provides comprehensive voice/speech signal analysis including:
//! - Fundamental frequency (F0) estimation
//! - Jitter and shimmer calculation
//! - Formant analysis
//! - Harmonic-to-noise ratio (HNR)
//! - Voice quality metrics
//! - Spectral analysis for speech

use crate::error::{DpbError, Result};
use ndarray::ArrayView1;
use serde::{Deserialize, Serialize};

/// Fundamental frequency (F0) metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Metrics {
    /// Mean F0 (Hz)
    pub mean_f0: f64,
    /// Standard deviation of F0 (Hz)
    pub std_f0: f64,
    /// Minimum F0 (Hz)
    pub min_f0: f64,
    /// Maximum F0 (Hz)
    pub max_f0: f64,
    /// F0 range (semitones)
    pub range_semitones: f64,
    /// F0 contour (frame-by-frame)
    pub contour: Vec<f64>,
    /// Voiced frame ratio
    pub voiced_ratio: f64,
}

/// Jitter metrics (frequency perturbation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterMetrics {
    /// Jitter (local) - average absolute difference between consecutive periods
    pub jitter_local: f64,
    /// Jitter (local, absolute) in microseconds
    pub jitter_local_abs: f64,
    /// Jitter (RAP) - relative average perturbation
    pub jitter_rap: f64,
    /// Jitter (PPQ5) - five-point period perturbation quotient
    pub jitter_ppq5: f64,
    /// Jitter (DDP) - difference of differences of periods
    pub jitter_ddp: f64,
}

/// Shimmer metrics (amplitude perturbation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShimmerMetrics {
    /// Shimmer (local) - average absolute difference between consecutive amplitudes
    pub shimmer_local: f64,
    /// Shimmer (local, dB)
    pub shimmer_local_db: f64,
    /// Shimmer (APQ3) - three-point amplitude perturbation quotient
    pub shimmer_apq3: f64,
    /// Shimmer (APQ5) - five-point amplitude perturbation quotient
    pub shimmer_apq5: f64,
    /// Shimmer (APQ11) - eleven-point amplitude perturbation quotient
    pub shimmer_apq11: f64,
    /// Shimmer (DDA) - difference of differences of amplitudes
    pub shimmer_dda: f64,
}

/// Formant frequencies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantFrequencies {
    /// First formant (F1) in Hz
    pub f1: f64,
    /// Second formant (F2) in Hz
    pub f2: f64,
    /// Third formant (F3) in Hz
    pub f3: f64,
    /// Fourth formant (F4) in Hz (optional)
    pub f4: Option<f64>,
    /// Bandwidths for each formant
    pub bandwidths: Vec<f64>,
}

/// Voice quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceQualityMetrics {
    /// Harmonic-to-noise ratio (dB)
    pub hnr: f64,
    /// Noise-to-harmonic ratio
    pub nhr: f64,
    /// Cepstral peak prominence (dB)
    pub cpp: f64,
    /// Smoothed cepstral peak prominence (dB)
    pub cpps: f64,
    /// Soft phonation index
    pub spi: f64,
}

/// Spectral voice features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpectralVoiceFeatures {
    /// Spectral centroid (Hz)
    pub centroid: f64,
    /// Spectral spread (Hz)
    pub spread: f64,
    /// Spectral skewness
    pub skewness: f64,
    /// Spectral kurtosis
    pub kurtosis: f64,
    /// Spectral slope
    pub slope: f64,
    /// Spectral flux
    pub flux: f64,
    /// Spectral rolloff (Hz)
    pub rolloff: f64,
    /// Alpha ratio (energy ratio 0-1kHz / 1-5kHz)
    pub alpha_ratio: f64,
    /// Hammarberg index
    pub hammarberg_index: f64,
}

/// Speech timing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechTimingMetrics {
    /// Speech rate (syllables per second)
    pub speech_rate: f64,
    /// Articulation rate (syllables per second, excluding pauses)
    pub articulation_rate: f64,
    /// Total speech time (seconds)
    pub total_speech_time: f64,
    /// Total pause time (seconds)
    pub total_pause_time: f64,
    /// Number of pauses
    pub pause_count: usize,
    /// Mean pause duration (seconds)
    pub mean_pause_duration: f64,
    /// Phonation time ratio
    pub phonation_ratio: f64,
}

/// Voice analyzer
pub struct VoiceAnalyzer {
    sample_rate: f64,
    /// Frame size for analysis (samples)
    frame_size: usize,
    /// Frame hop size (samples)
    hop_size: usize,
    /// Minimum F0 search range (Hz)
    min_f0: f64,
    /// Maximum F0 search range (Hz)
    max_f0: f64,
}

impl VoiceAnalyzer {
    /// Create a new voice analyzer
    pub fn new(sample_rate: f64) -> Self {
        let frame_size = (sample_rate * 0.025) as usize; // 25ms frames
        let hop_size = (sample_rate * 0.010) as usize; // 10ms hop

        Self {
            sample_rate,
            frame_size,
            hop_size,
            min_f0: 75.0,
            max_f0: 600.0,
        }
    }

    /// Configure F0 search range (e.g., for male vs female voices)
    pub fn set_f0_range(&mut self, min_f0: f64, max_f0: f64) {
        self.min_f0 = min_f0;
        self.max_f0 = max_f0;
    }

    /// Configure frame parameters
    pub fn set_frame_params(&mut self, frame_ms: f64, hop_ms: f64) {
        self.frame_size = (self.sample_rate * frame_ms / 1000.0) as usize;
        self.hop_size = (self.sample_rate * hop_ms / 1000.0) as usize;
    }

    /// Estimate fundamental frequency using autocorrelation
    pub fn estimate_f0(&self, signal: ArrayView1<f64>) -> Result<F0Metrics> {
        if signal.len() < self.frame_size {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for F0 analysis".to_string(),
            ));
        }

        let num_frames = (signal.len() - self.frame_size) / self.hop_size + 1;
        let mut f0_values = Vec::with_capacity(num_frames);
        let mut voiced_frames = 0;

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = start + self.frame_size;
            let frame = signal.slice(ndarray::s![start..end]);

            if let Some(f0) = self.estimate_frame_f0(&frame) {
                f0_values.push(f0);
                voiced_frames += 1;
            } else {
                f0_values.push(0.0); // Unvoiced
            }
        }

        let voiced_f0: Vec<f64> = f0_values.iter().filter(|&&f| f > 0.0).cloned().collect();

        if voiced_f0.is_empty() {
            return Ok(F0Metrics {
                mean_f0: 0.0,
                std_f0: 0.0,
                min_f0: 0.0,
                max_f0: 0.0,
                range_semitones: 0.0,
                contour: f0_values,
                voiced_ratio: 0.0,
            });
        }

        let mean_f0 = voiced_f0.iter().sum::<f64>() / voiced_f0.len() as f64;
        let variance = voiced_f0
            .iter()
            .map(|&f| (f - mean_f0).powi(2))
            .sum::<f64>()
            / voiced_f0.len() as f64;
        let std_f0 = variance.sqrt();

        let min_f0 = voiced_f0.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_f0 = voiced_f0.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let range_semitones = if min_f0 > 0.0 {
            12.0 * (max_f0 / min_f0).log2()
        } else {
            0.0
        };

        let voiced_ratio = voiced_frames as f64 / num_frames as f64;

        Ok(F0Metrics {
            mean_f0,
            std_f0,
            min_f0,
            max_f0,
            range_semitones,
            contour: f0_values,
            voiced_ratio,
        })
    }

    /// Estimate F0 for a single frame using autocorrelation
    fn estimate_frame_f0(&self, frame: &ArrayView1<f64>) -> Option<f64> {
        let n = frame.len();

        // Apply window
        let windowed: Vec<f64> = frame
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let w = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / (n - 1) as f64).cos();
                x * w
            })
            .collect();

        // Check if frame is voiced (sufficient energy)
        let energy: f64 = windowed.iter().map(|&x| x * x).sum();
        if energy < 1e-10 {
            return None;
        }

        // Autocorrelation
        let min_lag = (self.sample_rate / self.max_f0) as usize;
        let max_lag = (self.sample_rate / self.min_f0) as usize;
        let max_lag = max_lag.min(n - 1);

        let mut best_lag = 0;
        let mut best_corr = 0.0;

        for lag in min_lag..=max_lag {
            let mut corr = 0.0;
            for i in 0..(n - lag) {
                corr += windowed[i] * windowed[i + lag];
            }

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // Check if correlation is strong enough (voiced detection)
        let autocorr_0: f64 = windowed.iter().map(|&x| x * x).sum();
        let normalized_corr = if autocorr_0 > 0.0 {
            best_corr / autocorr_0
        } else {
            0.0
        };

        if normalized_corr > 0.3 && best_lag > 0 {
            Some(self.sample_rate / best_lag as f64)
        } else {
            None
        }
    }

    /// Calculate jitter metrics from period sequence
    pub fn calculate_jitter(&self, periods: &[f64]) -> Result<JitterMetrics> {
        if periods.len() < 3 {
            return Err(DpbError::DataValidation(
                "Need at least 3 periods for jitter calculation".to_string(),
            ));
        }

        let n = periods.len();
        let mean_period = periods.iter().sum::<f64>() / n as f64;

        // Jitter (local)
        let local_diffs: Vec<f64> = periods.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let jitter_local_abs = local_diffs.iter().sum::<f64>() / local_diffs.len() as f64;
        let jitter_local = jitter_local_abs / mean_period * 100.0;

        // Jitter (RAP) - 3-point smoothing
        let mut rap_sum = 0.0;
        for i in 1..n - 1 {
            let local_mean = (periods[i - 1] + periods[i] + periods[i + 1]) / 3.0;
            rap_sum += (periods[i] - local_mean).abs();
        }
        let jitter_rap = rap_sum / ((n - 2) as f64 * mean_period) * 100.0;

        // Jitter (PPQ5) - 5-point smoothing
        let mut ppq5_sum = 0.0;
        let mut ppq5_count = 0;
        for i in 2..n.saturating_sub(2) {
            let local_mean =
                (periods[i - 2] + periods[i - 1] + periods[i] + periods[i + 1] + periods[i + 2])
                    / 5.0;
            ppq5_sum += (periods[i] - local_mean).abs();
            ppq5_count += 1;
        }
        let jitter_ppq5 = if ppq5_count > 0 {
            ppq5_sum / (ppq5_count as f64 * mean_period) * 100.0
        } else {
            0.0
        };

        // Jitter (DDP)
        let jitter_ddp = jitter_rap * 3.0;

        Ok(JitterMetrics {
            jitter_local,
            jitter_local_abs: jitter_local_abs * 1_000_000.0, // Convert to microseconds
            jitter_rap,
            jitter_ppq5,
            jitter_ddp,
        })
    }

    /// Calculate shimmer metrics from amplitude sequence
    pub fn calculate_shimmer(&self, amplitudes: &[f64]) -> Result<ShimmerMetrics> {
        if amplitudes.len() < 3 {
            return Err(DpbError::DataValidation(
                "Need at least 3 amplitudes for shimmer calculation".to_string(),
            ));
        }

        let n = amplitudes.len();
        let mean_amplitude = amplitudes.iter().sum::<f64>() / n as f64;

        // Shimmer (local)
        let local_diffs: Vec<f64> = amplitudes.windows(2).map(|w| (w[1] - w[0]).abs()).collect();
        let shimmer_local =
            local_diffs.iter().sum::<f64>() / local_diffs.len() as f64 / mean_amplitude * 100.0;

        // Shimmer (local, dB)
        let db_diffs: Vec<f64> = amplitudes
            .windows(2)
            .filter(|w| w[0] > 0.0 && w[1] > 0.0)
            .map(|w| 20.0 * (w[1] / w[0]).log10().abs())
            .collect();
        let shimmer_local_db = if !db_diffs.is_empty() {
            db_diffs.iter().sum::<f64>() / db_diffs.len() as f64
        } else {
            0.0
        };

        // Shimmer (APQ3)
        let mut apq3_sum = 0.0;
        for i in 1..n - 1 {
            let local_mean = (amplitudes[i - 1] + amplitudes[i] + amplitudes[i + 1]) / 3.0;
            apq3_sum += (amplitudes[i] - local_mean).abs();
        }
        let shimmer_apq3 = apq3_sum / ((n - 2) as f64 * mean_amplitude) * 100.0;

        // Shimmer (APQ5)
        let mut apq5_sum = 0.0;
        let mut apq5_count = 0;
        for i in 2..n.saturating_sub(2) {
            let local_mean = (amplitudes[i - 2]
                + amplitudes[i - 1]
                + amplitudes[i]
                + amplitudes[i + 1]
                + amplitudes[i + 2])
                / 5.0;
            apq5_sum += (amplitudes[i] - local_mean).abs();
            apq5_count += 1;
        }
        let shimmer_apq5 = if apq5_count > 0 {
            apq5_sum / (apq5_count as f64 * mean_amplitude) * 100.0
        } else {
            0.0
        };

        // Shimmer (APQ11)
        let mut apq11_sum = 0.0;
        let mut apq11_count = 0;
        for i in 5..n.saturating_sub(5) {
            let local_mean: f64 = (-5..=5)
                .map(|j| amplitudes[(i as i64 + j) as usize])
                .sum::<f64>()
                / 11.0;
            apq11_sum += (amplitudes[i] - local_mean).abs();
            apq11_count += 1;
        }
        let shimmer_apq11 = if apq11_count > 0 {
            apq11_sum / (apq11_count as f64 * mean_amplitude) * 100.0
        } else {
            0.0
        };

        // Shimmer (DDA)
        let shimmer_dda = shimmer_apq3 * 3.0;

        Ok(ShimmerMetrics {
            shimmer_local,
            shimmer_local_db,
            shimmer_apq3,
            shimmer_apq5,
            shimmer_apq11,
            shimmer_dda,
        })
    }

    /// Estimate harmonic-to-noise ratio
    pub fn calculate_hnr(&self, signal: ArrayView1<f64>) -> Result<f64> {
        let f0_metrics = self.estimate_f0(signal)?;

        if f0_metrics.mean_f0 <= 0.0 {
            return Err(DpbError::DataValidation(
                "Cannot calculate HNR for unvoiced signal".to_string(),
            ));
        }

        // Simplified HNR using autocorrelation at F0
        let period_samples = (self.sample_rate / f0_metrics.mean_f0) as usize;

        if period_samples >= signal.len() {
            return Ok(0.0);
        }

        // Calculate autocorrelation at the period
        let mut r0 = 0.0;
        let mut r_period = 0.0;

        for i in 0..signal.len() - period_samples {
            r0 += signal[i] * signal[i];
            r_period += signal[i] * signal[i + period_samples];
        }

        if r0 <= 0.0 {
            return Ok(0.0);
        }

        let normalized_corr = r_period / r0;

        // HNR in dB
        if (0.0..1.0).contains(&normalized_corr) {
            Ok(10.0 * (normalized_corr / (1.0 - normalized_corr)).log10())
        } else if normalized_corr >= 1.0 {
            Ok(30.0) // Very high HNR
        } else {
            Ok(0.0)
        }
    }

    /// Calculate voice quality metrics
    pub fn analyze_voice_quality(&self, signal: ArrayView1<f64>) -> Result<VoiceQualityMetrics> {
        let hnr = self.calculate_hnr(signal).unwrap_or(0.0);
        let nhr = if hnr > 0.0 { 1.0 / hnr } else { 1.0 };

        // Cepstral Peak Prominence (simplified)
        let cpp = self.calculate_cpp(signal);
        let cpps = cpp * 0.9; // Smoothed version approximation

        // Soft Phonation Index (ratio of low to high frequency energy)
        let spi = self.calculate_spi(signal);

        Ok(VoiceQualityMetrics {
            hnr,
            nhr,
            cpp,
            cpps,
            spi,
        })
    }

    /// Calculate Cepstral Peak Prominence
    fn calculate_cpp(&self, signal: ArrayView1<f64>) -> f64 {
        // Simplified CPP using autocorrelation cepstrum
        let n = signal.len().min(self.frame_size * 4);
        let frame = signal.slice(ndarray::s![..n]);

        // Compute log power spectrum (simplified)
        let energy: f64 = frame.iter().map(|&x| x * x).sum();
        let log_energy = if energy > 0.0 { energy.log10() } else { -10.0 };

        // CPP typically ranges 0-30 dB for normal voices
        // This is a simplified approximation
        (log_energy + 10.0).clamp(0.0, 30.0)
    }

    /// Calculate Soft Phonation Index
    fn calculate_spi(&self, signal: ArrayView1<f64>) -> f64 {
        // Ratio of energy below 1kHz to energy above 1kHz
        let cutoff = (1000.0 / self.sample_rate * 2.0 * signal.len() as f64) as usize;

        let low_energy: f64 = signal.iter().take(cutoff).map(|&x| x * x).sum();
        let high_energy: f64 = signal.iter().skip(cutoff).map(|&x| x * x).sum();

        if high_energy > 0.0 {
            low_energy / high_energy
        } else {
            0.0
        }
    }

    /// Calculate spectral voice features
    pub fn analyze_spectrum(&self, signal: ArrayView1<f64>) -> Result<SpectralVoiceFeatures> {
        if signal.len() < 256 {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for spectral analysis".to_string(),
            ));
        }

        // Compute magnitude spectrum (simplified using DFT)
        let n = signal.len().min(2048);
        let mut magnitudes = Vec::with_capacity(n / 2);
        let mut freqs = Vec::with_capacity(n / 2);

        for k in 0..n / 2 {
            let freq = k as f64 * self.sample_rate / n as f64;
            let omega = 2.0 * std::f64::consts::PI * k as f64 / n as f64;

            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &sample) in signal.iter().take(n).enumerate() {
                real += sample * (omega * i as f64).cos();
                imag += sample * (omega * i as f64).sin();
            }

            magnitudes.push((real * real + imag * imag).sqrt());
            freqs.push(freq);
        }

        let total_energy: f64 = magnitudes.iter().map(|&m| m * m).sum();

        // Spectral centroid
        let centroid = if total_energy > 0.0 {
            magnitudes
                .iter()
                .zip(freqs.iter())
                .map(|(&m, &f)| m * m * f)
                .sum::<f64>()
                / total_energy
        } else {
            0.0
        };

        // Spectral spread
        let spread = if total_energy > 0.0 {
            (magnitudes
                .iter()
                .zip(freqs.iter())
                .map(|(&m, &f)| m * m * (f - centroid).powi(2))
                .sum::<f64>()
                / total_energy)
                .sqrt()
        } else {
            0.0
        };

        // Higher moments
        let skewness = if spread > 0.0 && total_energy > 0.0 {
            magnitudes
                .iter()
                .zip(freqs.iter())
                .map(|(&m, &f)| m * m * ((f - centroid) / spread).powi(3))
                .sum::<f64>()
                / total_energy
        } else {
            0.0
        };

        let kurtosis = if spread > 0.0 && total_energy > 0.0 {
            magnitudes
                .iter()
                .zip(freqs.iter())
                .map(|(&m, &f)| m * m * ((f - centroid) / spread).powi(4))
                .sum::<f64>()
                / total_energy
                - 3.0
        } else {
            0.0
        };

        // Spectral slope
        let mean_freq = freqs.iter().sum::<f64>() / freqs.len() as f64;
        let mean_mag = magnitudes.iter().sum::<f64>() / magnitudes.len() as f64;
        let slope = {
            let num: f64 = magnitudes
                .iter()
                .zip(freqs.iter())
                .map(|(&m, &f)| (f - mean_freq) * (m - mean_mag))
                .sum();
            let den: f64 = freqs.iter().map(|&f| (f - mean_freq).powi(2)).sum();
            if den > 0.0 { num / den } else { 0.0 }
        };

        // Spectral rolloff (85% of energy)
        let target_energy = total_energy * 0.85;
        let mut cumsum = 0.0;
        let rolloff = magnitudes
            .iter()
            .zip(freqs.iter())
            .find_map(|(&m, &f)| {
                cumsum += m * m;
                if cumsum >= target_energy {
                    Some(f)
                } else {
                    None
                }
            })
            .unwrap_or(0.0);

        // Alpha ratio (0-1kHz vs 1-5kHz)
        let low_energy: f64 = magnitudes
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f <= 1000.0)
            .map(|(&m, _)| m * m)
            .sum();
        let mid_energy: f64 = magnitudes
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f > 1000.0 && **f <= 5000.0)
            .map(|(&m, _)| m * m)
            .sum();
        let alpha_ratio = if mid_energy > 0.0 {
            10.0 * (low_energy / mid_energy).log10()
        } else {
            0.0
        };

        // Hammarberg index (0-2kHz vs 2-5kHz peak ratio)
        let low_peak = magnitudes
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f <= 2000.0)
            .map(|(&m, _)| m)
            .fold(0.0f64, f64::max);
        let high_peak = magnitudes
            .iter()
            .zip(freqs.iter())
            .filter(|(_, f)| **f > 2000.0 && **f <= 5000.0)
            .map(|(&m, _)| m)
            .fold(0.0f64, f64::max);
        let hammarberg_index = if high_peak > 0.0 {
            20.0 * (low_peak / high_peak).log10()
        } else {
            0.0
        };

        Ok(SpectralVoiceFeatures {
            centroid,
            spread,
            skewness,
            kurtosis,
            slope,
            flux: 0.0, // Would need previous frame
            rolloff,
            alpha_ratio,
            hammarberg_index,
        })
    }

    /// Detect voiced/unvoiced segments and pauses
    pub fn analyze_timing(
        &self,
        signal: ArrayView1<f64>,
        total_syllables: Option<usize>,
    ) -> Result<SpeechTimingMetrics> {
        let duration_seconds = signal.len() as f64 / self.sample_rate;

        // Calculate frame energies
        let num_frames = (signal.len() - self.frame_size) / self.hop_size + 1;
        let mut frame_energies = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = start + self.frame_size;
            let frame = signal.slice(ndarray::s![start..end]);
            let energy: f64 = frame.iter().map(|&x| x * x).sum();
            frame_energies.push(energy);
        }

        // Determine energy threshold for silence
        let mean_energy = frame_energies.iter().sum::<f64>() / frame_energies.len() as f64;
        let silence_threshold = mean_energy * 0.1;

        // Find pauses
        let mut in_pause = false;
        let mut pause_start = 0;
        let mut pauses = Vec::new();

        for (i, &energy) in frame_energies.iter().enumerate() {
            if energy < silence_threshold && !in_pause {
                in_pause = true;
                pause_start = i;
            } else if energy >= silence_threshold && in_pause {
                in_pause = false;
                let pause_duration =
                    (i - pause_start) as f64 * self.hop_size as f64 / self.sample_rate;
                if pause_duration > 0.15 {
                    // Minimum 150ms for a pause
                    pauses.push(pause_duration);
                }
            }
        }

        let pause_count = pauses.len();
        let total_pause_time: f64 = pauses.iter().sum();
        let total_speech_time = duration_seconds - total_pause_time;
        let mean_pause_duration = if pause_count > 0 {
            total_pause_time / pause_count as f64
        } else {
            0.0
        };

        let phonation_ratio = total_speech_time / duration_seconds;

        // Speech rate estimation
        let syllables = total_syllables.unwrap_or_else(|| {
            // Rough estimation based on energy peaks
            let threshold = mean_energy * 0.5;
            frame_energies
                .windows(3)
                .filter(|w| w[1] > threshold && w[1] > w[0] && w[1] > w[2])
                .count()
        });

        let speech_rate = syllables as f64 / duration_seconds;
        let articulation_rate = if total_speech_time > 0.0 {
            syllables as f64 / total_speech_time
        } else {
            0.0
        };

        Ok(SpeechTimingMetrics {
            speech_rate,
            articulation_rate,
            total_speech_time,
            total_pause_time,
            pause_count,
            mean_pause_duration,
            phonation_ratio,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn generate_synthetic_vowel(sample_rate: f64, duration: f64, f0: f64) -> Array1<f64> {
        let n_samples = (sample_rate * duration) as usize;
        let mut signal = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let t = i as f64 / sample_rate;
            // Fundamental + harmonics
            signal[i] = (2.0 * std::f64::consts::PI * f0 * t).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * 2.0 * f0 * t).sin()
                + 0.25 * (2.0 * std::f64::consts::PI * 3.0 * f0 * t).sin()
                + 0.125 * (2.0 * std::f64::consts::PI * 4.0 * f0 * t).sin();
            // Add slight noise
            signal[i] += (i as f64 * 0.1).sin() * 0.01;
        }

        signal
    }

    #[test]
    fn test_f0_estimation() {
        let sample_rate = 16000.0;
        let signal = generate_synthetic_vowel(sample_rate, 0.5, 150.0);

        let analyzer = VoiceAnalyzer::new(sample_rate);
        let f0_metrics = analyzer.estimate_f0(signal.view()).unwrap();

        assert!((f0_metrics.mean_f0 - 150.0).abs() < 20.0);
        assert!(f0_metrics.voiced_ratio > 0.5);
    }

    #[test]
    fn test_jitter_calculation() {
        // Relatively stable periods with slight variation
        let periods = vec![6.67, 6.68, 6.66, 6.67, 6.69, 6.65, 6.67, 6.68, 6.66, 6.67];

        let analyzer = VoiceAnalyzer::new(16000.0);
        let jitter = analyzer.calculate_jitter(&periods).unwrap();

        assert!(jitter.jitter_local >= 0.0);
        assert!(jitter.jitter_local < 5.0); // Normal jitter < 1%
    }

    #[test]
    fn test_shimmer_calculation() {
        let amplitudes = vec![1.0, 1.02, 0.98, 1.01, 0.99, 1.03, 0.97, 1.0, 1.01, 0.99];

        let analyzer = VoiceAnalyzer::new(16000.0);
        let shimmer = analyzer.calculate_shimmer(&amplitudes).unwrap();

        assert!(shimmer.shimmer_local >= 0.0);
        assert!(shimmer.shimmer_local < 10.0); // Normal shimmer < 3%
    }

    #[test]
    fn test_hnr_calculation() {
        let sample_rate = 16000.0;
        let signal = generate_synthetic_vowel(sample_rate, 0.5, 150.0);

        let analyzer = VoiceAnalyzer::new(sample_rate);
        let hnr = analyzer.calculate_hnr(signal.view()).unwrap();

        assert!(hnr > 0.0);
    }

    #[test]
    fn test_spectral_analysis() {
        let sample_rate = 16000.0;
        let signal = generate_synthetic_vowel(sample_rate, 0.5, 150.0);

        let analyzer = VoiceAnalyzer::new(sample_rate);
        let spectral = analyzer.analyze_spectrum(signal.view()).unwrap();

        assert!(spectral.centroid > 0.0);
        assert!(spectral.spread > 0.0);
        assert!(spectral.rolloff > 0.0);
    }

    #[test]
    fn test_timing_analysis() {
        let sample_rate = 16000.0;
        let mut signal = generate_synthetic_vowel(sample_rate, 2.0, 150.0);

        // Add silence in the middle
        for i in 12000..16000 {
            signal[i] = 0.0;
        }

        let analyzer = VoiceAnalyzer::new(sample_rate);
        let timing = analyzer.analyze_timing(signal.view(), None).unwrap();

        assert!(timing.total_speech_time > 0.0);
        assert!(timing.phonation_ratio > 0.0 && timing.phonation_ratio < 1.0);
    }
}
