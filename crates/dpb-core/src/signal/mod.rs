//! # Signal Processing Module
//!
//! Comprehensive signal processing for biosignals.
//!
//! This module provides domain-specific analyzers for various biosignal modalities,
//! along with general-purpose signal processing utilities.
//!
//! ## Submodules
//!
//! ### Contact Modalities
//! - [`ecg`]: ECG analysis including R-peak detection, QRS morphology, and arrhythmia detection
//! - [`hrv`]: Heart rate variability analysis (time and frequency domain)
//! - [`ppg`]: PPG analysis including pulse detection and SpO2 estimation
//! - [`eda`]: Electrodermal activity decomposition and SCR detection
//! - [`emg`]: EMG burst detection and fatigue analysis
//!
//! ### Neural Signals
//! - [`eeg`]: EEG band power analysis, artifact detection, ERP analysis, and seizure detection
//!
//! ### Movement and Voice
//! - [`eye`]: Eye tracking analysis (saccades, fixations, pupil metrics)
//! - [`voice`]: Voice analysis (F0, jitter, shimmer, formants, prosody)
//! - [`respiratory`]: Respiratory analysis and sleep apnea detection
//!
//! ### Multi-Modal Analysis
//! - [`fatigue`]: Multi-modal fatigue detection (EMG, force, cognitive)
//!
//! ### Signal Processing Primitives
//! - [`fft`]: FFT and STFT with windowing
//! - [`filter`]: FIR and IIR filters
//! - [`resample`]: Upsampling, downsampling, and resampling
//! - [`wavelet`]: Continuous and discrete wavelet transforms
//! - [`hilbert`]: Hilbert transform and analytic signals
//! - [`ica`]: Independent Component Analysis
//!
//! ## Example: ECG Analysis
//!
//! ```rust
//! use dpb_core::signal::ecg::*;
//! use dpb_core::signal::hrv::*;
//! use ndarray::Array1;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let ecg_signal = Array1::from_vec(vec![0.0; 1000]);
//! # let sample_rate = 250.0;
//! // Detect R-peaks
//! let detector = PanTompkinsDetector::new(sample_rate);
//! let peaks = detector.detect(&ecg_signal)?;
//!
//! // Compute HRV metrics
//! let analyzer = HrvAnalyzer::new();
//! let time_domain = analyzer.compute_time_domain(&peaks, sample_rate)?;
//! let freq_domain = analyzer.compute_frequency_domain(&peaks, sample_rate)?;
//!
//! println!("RMSSD: {:.1} ms", time_domain.rmssd);
//! println!("LF/HF: {:.2}", freq_domain.lf_hf_ratio);
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: EEG Band Power Analysis
//!
//! ```rust
//! use dpb_core::signal::eeg::*;
//! use ndarray::Array1;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let eeg_signal = Array1::from_vec(vec![0.0; 1000]);
//! # let sample_rate = 250.0;
//! // Compute band powers
//! let band_powers = compute_band_powers(&eeg_signal, sample_rate)?;
//!
//! println!("Delta: {:.2}", band_powers.delta);
//! println!("Theta: {:.2}", band_powers.theta);
//! println!("Alpha: {:.2}", band_powers.alpha);
//! println!("Beta: {:.2}", band_powers.beta);
//! println!("Gamma: {:.2}", band_powers.gamma);
//!
//! // Compute theta/beta ratio (ADHD marker)
//! let ratio = theta_beta_ratio(&band_powers);
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: Filtering and Normalization
//!
//! ```rust
//! use dpb_core::signal::*;
//! use ndarray::Array1;
//!
//! # fn example() -> dpb_core::Result<()> {
//! # let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
//! // Apply bandpass filter
//! let mut filter = IirFilter::bandpass(0.5, 40.0, 250.0, 4)?;
//! let filtered = filter.apply(signal.view())?;
//!
//! // Normalize signal
//! let normalized = normalize(filtered.view(), NormalizationMethod::ZScore)?;
//! # Ok(())
//! # }
//! ```

pub mod ecg;
pub mod eeg;
pub mod eda;
pub mod emg;
pub mod eye;
pub mod fatigue;
pub mod fft;
pub mod filter;
pub mod hilbert;
pub mod hrv;
pub mod ica;
pub mod ppg;
pub mod resample;
pub mod respiratory;
pub mod voice;
pub mod wavelet;

pub use fft::{FftProcessor, Stft, WindowType, create_window, fft_frequencies, stft_times};
pub use filter::{FirFilter, IirFilter, FilterType, median_filter};
pub use resample::{downsample, resample_linear, resample_to_length, upsample};

// Wavelet transform exports
pub use wavelet::{
    WaveletFamily, ContinuousWaveletTransform, DiscreteWaveletTransform, DwtResult,
};

// Hilbert transform exports
pub use hilbert::{
    hilbert_transform, analytic_signal, instantaneous_phase,
    instantaneous_frequency, AnalyticSignal,
};

// ICA exports
pub use ica::{FastICA, ICAResult, NonlinearFunction};

// ECG analysis exports
pub use ecg::{
    PanTompkinsDetector, RPeak, QrsMorphology, QrsTemplate, BeatType,
    ArrhythmiaDetector, ArrhythmiaAnalysis,
};

// HRV analysis exports
pub use hrv::{HrvTimeDomain, HrvFrequencyDomain, HrvAnalyzer, HrvMetrics};

// PPG analysis exports
pub use ppg::{PpgAnalyzer, PulseWaveFeatures, SpO2Result, PrvMetrics};

// EDA analysis exports
pub use eda::{EdaAnalyzer, EdaDecomposition, EdaMetrics, ScrEvent};

// EMG analysis exports
pub use emg::{EmgAnalyzer, EmgBurst, EmgTimeMetrics, EmgFrequencyMetrics, FatigueMetrics};

// Eye tracking analysis exports
pub use eye::{EyeAnalyzer, Saccade, Fixation, Blink, SmoothPursuitMetrics, PupilMetrics, GazePatternSummary};

// Voice analysis exports
pub use voice::{VoiceAnalyzer, F0Metrics, JitterMetrics, ShimmerMetrics, VoiceQualityMetrics, SpectralVoiceFeatures, SpeechTimingMetrics};

// EEG analysis exports
pub use eeg::{
    // Band power analysis
    alpha_asymmetry, compute_band_powers, extract_band_power, relative_band_power,
    theta_beta_ratio, BandPowers, EegBands,
    // Artifact detection
    apply_notch_filter, detect_artifacts, ArtifactSegment, ArtifactType,
    // ERP analysis
    Erp, ErpAnalyzer, ErpComponent, ErpGenerator, OddballData,
    // Seizure detection
    SeizureDetector, SeizureEvent, SeizureType, SeizureEvolution,
    EpileptiformSpike, AlertLevel, SeizureAnalysisResult, analyze_for_seizures,
};

// Fatigue detection exports
pub use fatigue::{
    EmgFatigueAnalyzer, EmgFatigueMetrics,
    ForceFatigueAnalyzer, ForceFatigueMetrics,
    CognitiveFatigueAnalyzer, CognitiveFatigueMetrics,
    IntegratedFatigueMetrics, FatigueType, FatigueSeverity,
    integrate_fatigue,
};

// Respiratory analysis exports
pub use respiratory::{
    RespiratoryAnalyzer, RespiratoryMetrics, RespiratoryPattern, PatternType,
    BreathEvent, ApneaEvent, ApneaType, ApneaSeverity,
    SleepBreathingAnalysis, SleepApneaSeverity, analyze_sleep_breathing,
};

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};

/// Normalization methods for signals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationMethod {
    /// Z-score normalization (mean=0, std=1)
    ZScore,
    /// Min-max normalization to [0, 1]
    MinMax,
    /// Min-max normalization to [-1, 1]
    MinMaxSymmetric,
    /// Robust normalization using median and IQR
    Robust,
}

/// Normalizes a signal using the specified method.
pub fn normalize(signal: ArrayView1<f64>, method: NormalizationMethod) -> Result<Array1<f64>> {
    if signal.is_empty() {
        return Err(DpbError::InvalidDimensions(
            "Signal cannot be empty".to_string(),
        ));
    }

    match method {
        NormalizationMethod::ZScore => {
            let mean = signal.mean().unwrap_or(0.0);
            let std = signal.std(0.0);
            if std == 0.0 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok((signal.to_owned() - mean) / std)
        }
        NormalizationMethod::MinMax => {
            let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (max - min).abs() < 1e-10 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok((signal.to_owned() - min) / (max - min))
        }
        NormalizationMethod::MinMaxSymmetric => {
            let min = signal.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = signal.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if (max - min).abs() < 1e-10 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok(((signal.to_owned() - min) / (max - min)) * 2.0 - 1.0)
        }
        NormalizationMethod::Robust => {
            let mut sorted = signal.to_vec();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q25 = sorted[sorted.len() / 4];
            let q75 = sorted[3 * sorted.len() / 4];
            let median = sorted[sorted.len() / 2];
            let iqr = q75 - q25;
            if iqr == 0.0 {
                return Ok(Array1::zeros(signal.len()));
            }
            Ok((signal.to_owned() - median) / iqr)
        }
    }
}

/// Normalizes multiple channels independently.
pub fn normalize_channels(
    signals: ArrayView2<f64>,
    method: NormalizationMethod,
) -> Result<ndarray::Array2<f64>> {
    let num_channels = signals.nrows();
    let mut normalized = ndarray::Array2::zeros(signals.dim());

    for i in 0..num_channels {
        let channel = signals.row(i);
        let norm_channel = normalize(channel, method)?;
        normalized.row_mut(i).assign(&norm_channel);
    }

    Ok(normalized)
}

/// Removes DC offset from a signal.
pub fn remove_dc_offset(signal: ArrayView1<f64>) -> Array1<f64> {
    let mean = signal.mean().unwrap_or(0.0);
    signal.to_owned() - mean
}

/// Applies a gain to a signal.
pub fn apply_gain(signal: ArrayView1<f64>, gain: f64) -> Array1<f64> {
    signal.to_owned() * gain
}

/// Clips signal values to a range.
pub fn clip(signal: ArrayView1<f64>, min: f64, max: f64) -> Array1<f64> {
    signal.mapv(|x| x.clamp(min, max))
}

/// Detects zero crossings in a signal.
pub fn zero_crossings(signal: ArrayView1<f64>) -> Vec<usize> {
    let mut crossings = Vec::new();

    for i in 0..signal.len() - 1 {
        if (signal[i] < 0.0 && signal[i + 1] >= 0.0) || (signal[i] >= 0.0 && signal[i + 1] < 0.0) {
            crossings.push(i);
        }
    }

    crossings
}

/// Detects peaks in a signal.
pub fn find_peaks(signal: ArrayView1<f64>, threshold: f64) -> Vec<usize> {
    let mut peaks = Vec::new();

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

/// Detects valleys in a signal.
pub fn find_valleys(signal: ArrayView1<f64>, threshold: f64) -> Vec<usize> {
    let mut valleys = Vec::new();

    for i in 1..signal.len() - 1 {
        if signal[i] < threshold
            && signal[i] < signal[i - 1]
            && signal[i] < signal[i + 1]
        {
            valleys.push(i);
        }
    }

    valleys
}

/// Computes the envelope of a signal using Hilbert transform approximation.
pub fn envelope(signal: ArrayView1<f64>) -> Result<Array1<f64>> {
    // Simplified envelope using moving maximum
    let window_size = signal.len().min(20);
    let mut env = Array1::zeros(signal.len());

    for i in 0..signal.len() {
        let start = i.saturating_sub(window_size / 2);
        let end = (i + window_size / 2 + 1).min(signal.len());
        let max_val = signal
            .slice(ndarray::s![start..end])
            .iter()
            .map(|x| x.abs())
            .fold(0.0f64, f64::max);
        env[i] = max_val;
    }

    Ok(env)
}

/// Computes the root mean square (RMS) of a signal.
pub fn rms(signal: ArrayView1<f64>) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let sum_squares: f64 = signal.iter().map(|x| x * x).sum();
    (sum_squares / signal.len() as f64).sqrt()
}

/// Computes the energy of a signal.
pub fn energy(signal: ArrayView1<f64>) -> f64 {
    signal.iter().map(|x| x * x).sum()
}

/// Computes signal-to-noise ratio (SNR) in dB.
pub fn snr_db(signal: ArrayView1<f64>, noise: ArrayView1<f64>) -> Result<f64> {
    if signal.len() != noise.len() {
        return Err(DpbError::InvalidDimensions(
            "Signal and noise must have same length".to_string(),
        ));
    }

    let signal_power = energy(signal);
    let noise_power = energy(noise);

    if noise_power == 0.0 {
        return Ok(f64::INFINITY);
    }

    Ok(10.0 * (signal_power / noise_power).log10())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalize_zscore() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize(signal.view(), NormalizationMethod::ZScore).unwrap();

        let mean = normalized.mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_normalize_minmax() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let normalized = normalize(signal.view(), NormalizationMethod::MinMax).unwrap();

        assert_relative_eq!(normalized[0], 0.0);
        assert_relative_eq!(normalized[4], 1.0);
    }

    #[test]
    fn test_remove_dc_offset() {
        let signal = Array1::from_vec(vec![3.0, 4.0, 5.0, 6.0]);
        let detrended = remove_dc_offset(signal.view());

        let mean = detrended.mean().unwrap();
        assert_relative_eq!(mean, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zero_crossings() {
        let signal = Array1::from_vec(vec![-1.0, -0.5, 0.5, 1.0, 0.5, -0.5]);
        let crossings = zero_crossings(signal.view());

        assert_eq!(crossings.len(), 2);
        assert_eq!(crossings[0], 1); // Between index 1 and 2
        assert_eq!(crossings[1], 4); // Between index 4 and 5
    }

    #[test]
    fn test_find_peaks() {
        let signal = Array1::from_vec(vec![1.0, 3.0, 2.0, 4.0, 1.0]);
        let peaks = find_peaks(signal.view(), 2.0);

        assert_eq!(peaks.len(), 2);
        assert!(peaks.contains(&1));
        assert!(peaks.contains(&3));
    }

    #[test]
    fn test_rms() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let rms_val = rms(signal.view());

        let expected = ((1.0f64 + 4.0 + 9.0 + 16.0) / 4.0).sqrt();
        assert_relative_eq!(rms_val, expected);
    }

    #[test]
    fn test_energy() {
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let eng = energy(signal.view());

        assert_relative_eq!(eng, 14.0); // 1 + 4 + 9
    }

    #[test]
    fn test_clip() {
        let signal = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
        let clipped = clip(signal.view(), -1.0, 1.0);

        assert_eq!(clipped[0], -1.0);
        assert_eq!(clipped[4], 1.0);
        assert_eq!(clipped[2], 0.0);
    }
}
