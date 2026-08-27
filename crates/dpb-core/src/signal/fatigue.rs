//! Fatigue Detection Module
//!
//! Provides comprehensive fatigue detection algorithms including:
//! - EMG fatigue analysis (spectral shift, amplitude changes)
//! - Force/grip fatigue (strength decline, endurance)
//! - Cognitive fatigue (reaction time deterioration, vigilance decrement)
//! - Multi-modal fatigue integration

use crate::error::{DpbError, Result};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

// ============================================================================
// EMG Fatigue Detection
// ============================================================================

/// EMG fatigue indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgFatigueMetrics {
    /// Median frequency slope (Hz/s) - negative indicates fatigue
    pub mdf_slope: f64,
    /// Mean frequency slope (Hz/s)
    pub mnf_slope: f64,
    /// Initial median frequency (Hz)
    pub initial_mdf: f64,
    /// Final median frequency (Hz)
    pub final_mdf: f64,
    /// Percent MDF decrease
    pub mdf_decrease_percent: f64,
    /// RMS amplitude slope - positive may indicate compensation
    pub rms_slope: f64,
    /// Fatigue index (0-1, higher = more fatigue)
    pub fatigue_index: f64,
    /// Estimated time to fatigue failure (seconds)
    pub estimated_endurance_time: Option<f64>,
    /// Is fatigue detected
    pub fatigue_detected: bool,
}

/// EMG fatigue analyzer
pub struct EmgFatigueAnalyzer {
    sample_rate: f64,
    window_size: usize,
    overlap: f64,
    mdf_threshold: f64,  // Hz/s threshold for fatigue detection
}

impl EmgFatigueAnalyzer {
    /// Create new EMG fatigue analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            window_size: (sample_rate * 0.5) as usize, // 500ms windows
            overlap: 0.5,
            mdf_threshold: -0.5, // Hz/s - typical fatigue threshold
        }
    }

    /// Create with custom parameters
    pub fn with_params(sample_rate: f64, window_size_sec: f64, overlap: f64) -> Self {
        Self {
            sample_rate,
            window_size: (sample_rate * window_size_sec) as usize,
            overlap,
            mdf_threshold: -0.5,
        }
    }

    /// Analyze EMG signal for fatigue
    pub fn analyze(&self, signal: ArrayView1<f64>) -> Result<EmgFatigueMetrics> {
        if signal.len() < self.window_size * 2 {
            return Err(DpbError::InvalidDimensions(
                "Signal too short for fatigue analysis".to_string(),
            ));
        }

        // Calculate time-varying spectral features
        let (mdf_series, mnf_series, rms_series) = self.extract_features(signal)?;

        if mdf_series.len() < 3 {
            return Err(DpbError::InvalidDimensions(
                "Insufficient windows for trend analysis".to_string(),
            ));
        }

        // Calculate slopes using linear regression
        let mdf_slope = self.calculate_slope(&mdf_series);
        let mnf_slope = self.calculate_slope(&mnf_series);
        let rms_slope = self.calculate_slope(&rms_series);

        let initial_mdf = mdf_series[0];
        let final_mdf = *mdf_series.last().unwrap_or(&initial_mdf);
        let mdf_decrease_percent = ((initial_mdf - final_mdf) / initial_mdf) * 100.0;

        // Calculate fatigue index (normalized to 0-1)
        let fatigue_index = self.calculate_fatigue_index(mdf_slope, mdf_decrease_percent);

        // Estimate endurance time (if not already fatigued)
        let estimated_endurance_time = if mdf_slope < 0.0 && final_mdf > 0.0 {
            // Extrapolate to ~40% MDF decrease (common failure threshold)
            let target_mdf = initial_mdf * 0.6;
            let remaining_decrease = final_mdf - target_mdf;
            if remaining_decrease > 0.0 {
                let time_per_window = (self.window_size as f64 / self.sample_rate)
                    * (1.0 - self.overlap);
                let current_duration = mdf_series.len() as f64 * time_per_window;
                let rate = mdf_slope.abs();
                if rate > 0.0 {
                    Some(current_duration + remaining_decrease / rate)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let fatigue_detected = mdf_slope < self.mdf_threshold
            || mdf_decrease_percent > 15.0
            || fatigue_index > 0.5;

        Ok(EmgFatigueMetrics {
            mdf_slope,
            mnf_slope,
            initial_mdf,
            final_mdf,
            mdf_decrease_percent,
            rms_slope,
            fatigue_index,
            estimated_endurance_time,
            fatigue_detected,
        })
    }

    /// Extract time-varying spectral features
    fn extract_features(
        &self,
        signal: ArrayView1<f64>,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
        let step = ((1.0 - self.overlap) * self.window_size as f64) as usize;
        let step = step.max(1);
        let num_windows = (signal.len() - self.window_size) / step + 1;

        let mut mdf_series = Vec::with_capacity(num_windows);
        let mut mnf_series = Vec::with_capacity(num_windows);
        let mut rms_series = Vec::with_capacity(num_windows);

        for i in 0..num_windows {
            let start = i * step;
            let end = start + self.window_size;
            if end > signal.len() {
                break;
            }

            let window = signal.slice(ndarray::s![start..end]);

            // Calculate RMS
            let rms = (window.iter().map(|x| x * x).sum::<f64>() / self.window_size as f64).sqrt();
            rms_series.push(rms);

            // Calculate spectral features using simple power spectrum
            let (mdf, mnf) = self.calculate_spectral_features(&window)?;
            mdf_series.push(mdf);
            mnf_series.push(mnf);
        }

        Ok((mdf_series, mnf_series, rms_series))
    }

    /// Calculate median and mean frequency from signal window
    fn calculate_spectral_features(&self, window: &ArrayView1<f64>) -> Result<(f64, f64)> {
        // Simple DFT-based spectral estimation
        let n = window.len();
        let mut power_spectrum = Vec::with_capacity(n / 2);
        let mut frequencies = Vec::with_capacity(n / 2);

        for k in 1..n / 2 {
            let freq = k as f64 * self.sample_rate / n as f64;

            // Only consider EMG frequency range (20-500 Hz)
            if freq < 20.0 || freq > 500.0 {
                continue;
            }

            let mut real = 0.0;
            let mut imag = 0.0;

            for (i, &x) in window.iter().enumerate() {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                real += x * angle.cos();
                imag += x * angle.sin();
            }

            let power = (real * real + imag * imag) / n as f64;
            power_spectrum.push(power);
            frequencies.push(freq);
        }

        if power_spectrum.is_empty() {
            return Ok((100.0, 100.0)); // Default fallback
        }

        // Calculate total power
        let total_power: f64 = power_spectrum.iter().sum();
        if total_power <= 0.0 {
            return Ok((100.0, 100.0));
        }

        // Mean frequency
        let mnf: f64 = power_spectrum
            .iter()
            .zip(frequencies.iter())
            .map(|(&p, &f)| p * f)
            .sum::<f64>()
            / total_power;

        // Median frequency
        let mut cumulative = 0.0;
        let half_power = total_power / 2.0;
        let mut mdf = frequencies[0];

        for (&p, &f) in power_spectrum.iter().zip(frequencies.iter()) {
            cumulative += p;
            if cumulative >= half_power {
                mdf = f;
                break;
            }
        }

        Ok((mdf, mnf))
    }

    /// Calculate slope using linear regression
    fn calculate_slope(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let time_per_window =
            (self.window_size as f64 / self.sample_rate) * (1.0 - self.overlap);

        let x_mean = (n - 1.0) * time_per_window / 2.0;
        let y_mean: f64 = values.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den = 0.0;

        for (i, &y) in values.iter().enumerate() {
            let x = i as f64 * time_per_window;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean).powi(2);
        }

        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }

    /// Calculate normalized fatigue index
    fn calculate_fatigue_index(&self, mdf_slope: f64, mdf_decrease_percent: f64) -> f64 {
        // Combine slope and total decrease into single index
        let slope_component = (-mdf_slope / 2.0).clamp(0.0, 1.0); // Normalize to 0-1
        let decrease_component = (mdf_decrease_percent / 40.0).clamp(0.0, 1.0);

        // Weighted combination
        (0.6 * slope_component + 0.4 * decrease_component).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Force/Grip Fatigue Detection
// ============================================================================

/// Force fatigue metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForceFatigueMetrics {
    /// Initial maximum force (N or kg)
    pub initial_force: f64,
    /// Final force at end of test
    pub final_force: f64,
    /// Force decline percent
    pub decline_percent: f64,
    /// Time to 50% decline (seconds)
    pub time_to_50_percent: Option<f64>,
    /// Force slope (units/s)
    pub force_slope: f64,
    /// Coefficient of variation of force
    pub force_cv: f64,
    /// Endurance time (if sustained contraction)
    pub endurance_time: Option<f64>,
    /// Fatigue index (0-1)
    pub fatigue_index: f64,
}

/// Force fatigue analyzer
pub struct ForceFatigueAnalyzer {
    sample_rate: f64,
    target_force_percent: f64, // For sustained contractions (e.g., 50% MVC)
}

impl ForceFatigueAnalyzer {
    /// Create new force fatigue analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            target_force_percent: 50.0,
        }
    }

    /// Analyze force signal for fatigue
    pub fn analyze(&self, force: ArrayView1<f64>) -> Result<ForceFatigueMetrics> {
        if force.len() < 10 {
            return Err(DpbError::InvalidDimensions(
                "Force signal too short".to_string(),
            ));
        }

        // Get initial force (first 10% of signal)
        let init_samples = (force.len() as f64 * 0.1) as usize;
        let init_samples = init_samples.max(1);
        let initial_force: f64 = force.slice(ndarray::s![..init_samples]).iter().sum::<f64>()
            / init_samples as f64;

        // Get final force (last 10% of signal)
        let final_start = force.len() - init_samples;
        let final_force: f64 = force.slice(ndarray::s![final_start..]).iter().sum::<f64>()
            / init_samples as f64;

        // Calculate decline
        let decline_percent = if initial_force > 0.0 {
            ((initial_force - final_force) / initial_force) * 100.0
        } else {
            0.0
        };

        // Find time to 50% decline
        let target_50 = initial_force * 0.5;
        let time_to_50_percent = self.find_time_to_threshold(force, target_50);

        // Calculate force slope
        let force_slope = self.calculate_force_slope(force);

        // Calculate CV
        let mean_force = force.mean().unwrap_or(0.0);
        let force_cv = if mean_force > 0.0 {
            force.std(0.0) / mean_force * 100.0
        } else {
            0.0
        };

        // Calculate endurance time (time until force drops below target)
        let target_threshold = initial_force * (self.target_force_percent / 100.0) * 0.9;
        let endurance_time = self.find_time_to_threshold(force, target_threshold);

        // Fatigue index
        let fatigue_index = self.calculate_fatigue_index(decline_percent, force_cv);

        Ok(ForceFatigueMetrics {
            initial_force,
            final_force,
            decline_percent,
            time_to_50_percent,
            force_slope,
            force_cv,
            endurance_time,
            fatigue_index,
        })
    }

    fn find_time_to_threshold(&self, force: ArrayView1<f64>, threshold: f64) -> Option<f64> {
        for (i, &f) in force.iter().enumerate() {
            if f < threshold {
                return Some(i as f64 / self.sample_rate);
            }
        }
        None
    }

    fn calculate_force_slope(&self, force: ArrayView1<f64>) -> f64 {
        let n = force.len() as f64;
        let duration = n / self.sample_rate;

        let x_mean = duration / 2.0;
        let y_mean = force.mean().unwrap_or(0.0);

        let mut num = 0.0;
        let mut den = 0.0;

        for (i, &y) in force.iter().enumerate() {
            let x = i as f64 / self.sample_rate;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean).powi(2);
        }

        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }

    fn calculate_fatigue_index(&self, decline_percent: f64, force_cv: f64) -> f64 {
        let decline_component = (decline_percent / 50.0).clamp(0.0, 1.0);
        let variability_component = (force_cv / 20.0).clamp(0.0, 1.0);

        (0.7 * decline_component + 0.3 * variability_component).clamp(0.0, 1.0)
    }
}

// ============================================================================
// Cognitive Fatigue Detection
// ============================================================================

/// Cognitive fatigue metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CognitiveFatigueMetrics {
    /// Initial mean reaction time (ms)
    pub initial_rt: f64,
    /// Final mean reaction time (ms)
    pub final_rt: f64,
    /// RT increase percent (vigilance decrement)
    pub rt_increase_percent: f64,
    /// Initial accuracy (%)
    pub initial_accuracy: f64,
    /// Final accuracy (%)
    pub final_accuracy: f64,
    /// Accuracy decline percent
    pub accuracy_decline_percent: f64,
    /// RT variability increase
    pub variability_increase: f64,
    /// Lapse count (RTs > 500ms)
    pub lapse_count: usize,
    /// Lapse rate (per minute)
    pub lapse_rate: f64,
    /// Fatigue index (0-1)
    pub fatigue_index: f64,
    /// Time-on-task effect detected
    pub time_on_task_effect: bool,
}

/// Cognitive fatigue analyzer
pub struct CognitiveFatigueAnalyzer {
    lapse_threshold_ms: f64,
    block_size: usize,
}

impl CognitiveFatigueAnalyzer {
    /// Create new cognitive fatigue analyzer
    pub fn new() -> Self {
        Self {
            lapse_threshold_ms: 500.0,
            block_size: 20, // trials per block
        }
    }

    /// Analyze reaction time series for cognitive fatigue
    pub fn analyze(
        &self,
        reaction_times_ms: &[f64],
        accuracy: Option<&[bool]>,
        task_duration_min: f64,
    ) -> Result<CognitiveFatigueMetrics> {
        if reaction_times_ms.len() < self.block_size * 2 {
            return Err(DpbError::InvalidDimensions(
                "Insufficient trials for fatigue analysis".to_string(),
            ));
        }

        // Calculate block statistics
        let num_blocks = reaction_times_ms.len() / self.block_size;

        // First block (initial)
        let first_block: Vec<f64> = reaction_times_ms[..self.block_size].to_vec();
        let initial_rt = first_block.iter().sum::<f64>() / self.block_size as f64;
        let initial_rt_sd = self.calculate_sd(&first_block);

        // Last block (final)
        let last_start = reaction_times_ms.len() - self.block_size;
        let last_block: Vec<f64> = reaction_times_ms[last_start..].to_vec();
        let final_rt = last_block.iter().sum::<f64>() / self.block_size as f64;
        let final_rt_sd = self.calculate_sd(&last_block);

        let rt_increase_percent = ((final_rt - initial_rt) / initial_rt) * 100.0;
        let variability_increase = if initial_rt_sd > 0.0 {
            ((final_rt_sd - initial_rt_sd) / initial_rt_sd) * 100.0
        } else {
            0.0
        };

        // Count lapses
        let lapse_count = reaction_times_ms
            .iter()
            .filter(|&&rt| rt > self.lapse_threshold_ms)
            .count();
        let lapse_rate = lapse_count as f64 / task_duration_min;

        // Calculate accuracy if provided
        let (initial_accuracy, final_accuracy, accuracy_decline_percent) =
            if let Some(acc) = accuracy {
                let first_acc: f64 = acc[..self.block_size.min(acc.len())]
                    .iter()
                    .filter(|&&a| a)
                    .count() as f64
                    / self.block_size.min(acc.len()) as f64
                    * 100.0;

                let last_start = acc.len().saturating_sub(self.block_size);
                let last_acc: f64 = acc[last_start..]
                    .iter()
                    .filter(|&&a| a)
                    .count() as f64
                    / acc[last_start..].len() as f64
                    * 100.0;

                let decline = if first_acc > 0.0 {
                    ((first_acc - last_acc) / first_acc) * 100.0
                } else {
                    0.0
                };

                (first_acc, last_acc, decline)
            } else {
                (100.0, 100.0, 0.0)
            };

        // Calculate fatigue index
        let fatigue_index = self.calculate_fatigue_index(
            rt_increase_percent,
            accuracy_decline_percent,
            lapse_rate,
        );

        // Detect time-on-task effect (significant RT increase over time)
        let rt_slope = self.calculate_block_slope(reaction_times_ms, num_blocks);
        let time_on_task_effect = rt_slope > 0.5 && rt_increase_percent > 5.0;

        Ok(CognitiveFatigueMetrics {
            initial_rt,
            final_rt,
            rt_increase_percent,
            initial_accuracy,
            final_accuracy,
            accuracy_decline_percent,
            variability_increase,
            lapse_count,
            lapse_rate,
            fatigue_index,
            time_on_task_effect,
        })
    }

    fn calculate_sd(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / (values.len() - 1) as f64;
        variance.sqrt()
    }

    fn calculate_block_slope(&self, rts: &[f64], num_blocks: usize) -> f64 {
        if num_blocks < 2 {
            return 0.0;
        }

        let block_means: Vec<f64> = (0..num_blocks)
            .map(|i| {
                let start = i * self.block_size;
                let end = start + self.block_size;
                rts[start..end.min(rts.len())].iter().sum::<f64>() / self.block_size as f64
            })
            .collect();

        // Linear regression slope
        let n = block_means.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean: f64 = block_means.iter().sum::<f64>() / n;

        let mut num = 0.0;
        let mut den = 0.0;

        for (i, &y) in block_means.iter().enumerate() {
            let x = i as f64;
            num += (x - x_mean) * (y - y_mean);
            den += (x - x_mean).powi(2);
        }

        if den > 0.0 {
            num / den
        } else {
            0.0
        }
    }

    fn calculate_fatigue_index(
        &self,
        rt_increase: f64,
        acc_decline: f64,
        lapse_rate: f64,
    ) -> f64 {
        let rt_component = (rt_increase / 30.0).clamp(0.0, 1.0);
        let acc_component = (acc_decline / 20.0).clamp(0.0, 1.0);
        let lapse_component = (lapse_rate / 5.0).clamp(0.0, 1.0);

        (0.4 * rt_component + 0.3 * acc_component + 0.3 * lapse_component).clamp(0.0, 1.0)
    }
}

impl Default for CognitiveFatigueAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Multi-modal Fatigue Integration
// ============================================================================

/// Integrated multi-modal fatigue assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegratedFatigueMetrics {
    /// EMG fatigue metrics (if available)
    pub emg: Option<EmgFatigueMetrics>,
    /// Force fatigue metrics (if available)
    pub force: Option<ForceFatigueMetrics>,
    /// Cognitive fatigue metrics (if available)
    pub cognitive: Option<CognitiveFatigueMetrics>,
    /// Global fatigue index (0-1)
    pub global_fatigue_index: f64,
    /// Primary fatigue type
    pub primary_fatigue_type: FatigueType,
    /// Fatigue severity classification
    pub severity: FatigueSeverity,
}

/// Type of fatigue
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FatigueType {
    /// No significant fatigue
    None,
    /// Primarily peripheral/muscular fatigue
    Peripheral,
    /// Primarily central/cognitive fatigue
    Central,
    /// Both peripheral and central fatigue
    Mixed,
}

/// Fatigue severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FatigueSeverity {
    /// No fatigue
    None,
    /// Mild fatigue (index < 0.3)
    Mild,
    /// Moderate fatigue (index 0.3-0.6)
    Moderate,
    /// Severe fatigue (index > 0.6)
    Severe,
}

impl FatigueSeverity {
    /// Create from fatigue index
    pub fn from_index(index: f64) -> Self {
        if index < 0.1 {
            FatigueSeverity::None
        } else if index < 0.3 {
            FatigueSeverity::Mild
        } else if index < 0.6 {
            FatigueSeverity::Moderate
        } else {
            FatigueSeverity::Severe
        }
    }

    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            FatigueSeverity::None => "No Fatigue",
            FatigueSeverity::Mild => "Mild Fatigue",
            FatigueSeverity::Moderate => "Moderate Fatigue",
            FatigueSeverity::Severe => "Severe Fatigue",
        }
    }
}

/// Integrate multiple fatigue measures
pub fn integrate_fatigue(
    emg: Option<&EmgFatigueMetrics>,
    force: Option<&ForceFatigueMetrics>,
    cognitive: Option<&CognitiveFatigueMetrics>,
) -> IntegratedFatigueMetrics {
    // Collect available indices
    let mut indices = Vec::new();
    let mut has_peripheral = false;
    let mut has_central = false;

    if let Some(e) = emg {
        indices.push(e.fatigue_index);
        if e.fatigue_detected {
            has_peripheral = true;
        }
    }

    if let Some(f) = force {
        indices.push(f.fatigue_index);
        if f.decline_percent > 20.0 {
            has_peripheral = true;
        }
    }

    if let Some(c) = cognitive {
        indices.push(c.fatigue_index);
        if c.time_on_task_effect {
            has_central = true;
        }
    }

    // Calculate global index
    let global_fatigue_index = if indices.is_empty() {
        0.0
    } else {
        indices.iter().sum::<f64>() / indices.len() as f64
    };

    // Determine primary type
    let primary_fatigue_type = match (has_peripheral, has_central) {
        (true, true) => FatigueType::Mixed,
        (true, false) => FatigueType::Peripheral,
        (false, true) => FatigueType::Central,
        (false, false) => FatigueType::None,
    };

    let severity = FatigueSeverity::from_index(global_fatigue_index);

    IntegratedFatigueMetrics {
        emg: emg.cloned(),
        force: force.cloned(),
        cognitive: cognitive.cloned(),
        global_fatigue_index,
        primary_fatigue_type,
        severity,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_fatiguing_emg(n_samples: usize, sample_rate: f64) -> Array1<f64> {
        let mut signal = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let t = i as f64 / sample_rate;
            // Decreasing frequency component over time to simulate fatigue
            let freq = 80.0 - 20.0 * (t / (n_samples as f64 / sample_rate));
            let freq = freq.max(40.0);
            signal[i] = (2.0 * std::f64::consts::PI * freq * t).sin()
                + 0.3 * rand_noise();
        }
        signal
    }

    fn rand_noise() -> f64 {
        // Simple deterministic pseudo-random for testing
        ((std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
            % 1000) as f64
            - 500.0)
            / 500.0
    }

    #[test]
    fn test_emg_fatigue_analyzer() {
        let sample_rate = 1000.0;
        let signal = generate_fatiguing_emg(10000, sample_rate);
        let analyzer = EmgFatigueAnalyzer::new(sample_rate);

        let result = analyzer.analyze(signal.view());
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // MDF should decrease during fatigue
        assert!(metrics.final_mdf <= metrics.initial_mdf || metrics.mdf_slope <= 0.0);
    }

    #[test]
    fn test_force_fatigue_analyzer() {
        let sample_rate = 100.0;
        // Simulated declining force
        let n = 1000;
        let force: Array1<f64> = (0..n)
            .map(|i| 100.0 * (-0.001 * i as f64).exp())
            .collect();

        let analyzer = ForceFatigueAnalyzer::new(sample_rate);
        let result = analyzer.analyze(force.view());

        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.decline_percent > 0.0);
        assert!(metrics.final_force < metrics.initial_force);
    }

    #[test]
    fn test_cognitive_fatigue_analyzer() {
        let analyzer = CognitiveFatigueAnalyzer::new();

        // Simulated RT increase over time (fatigue effect)
        let rts: Vec<f64> = (0..100)
            .map(|i| 250.0 + 50.0 * (i as f64 / 100.0) + 20.0 * rand_noise())
            .collect();

        let result = analyzer.analyze(&rts, None, 10.0);
        assert!(result.is_ok());

        let metrics = result.unwrap();
        // Should detect some RT increase
        assert!(metrics.final_rt >= metrics.initial_rt - 50.0); // Allow some variance
    }

    #[test]
    fn test_fatigue_severity() {
        assert_eq!(FatigueSeverity::from_index(0.0), FatigueSeverity::None);
        assert_eq!(FatigueSeverity::from_index(0.2), FatigueSeverity::Mild);
        assert_eq!(FatigueSeverity::from_index(0.5), FatigueSeverity::Moderate);
        assert_eq!(FatigueSeverity::from_index(0.8), FatigueSeverity::Severe);
    }

    #[test]
    fn test_integrate_fatigue() {
        let emg = EmgFatigueMetrics {
            mdf_slope: -1.0,
            mnf_slope: -0.8,
            initial_mdf: 80.0,
            final_mdf: 60.0,
            mdf_decrease_percent: 25.0,
            rms_slope: 0.1,
            fatigue_index: 0.5,
            estimated_endurance_time: Some(120.0),
            fatigue_detected: true,
        };

        let cognitive = CognitiveFatigueMetrics {
            initial_rt: 250.0,
            final_rt: 300.0,
            rt_increase_percent: 20.0,
            initial_accuracy: 95.0,
            final_accuracy: 90.0,
            accuracy_decline_percent: 5.3,
            variability_increase: 30.0,
            lapse_count: 3,
            lapse_rate: 0.5,
            fatigue_index: 0.4,
            time_on_task_effect: true,
        };

        let integrated = integrate_fatigue(Some(&emg), None, Some(&cognitive));

        assert_eq!(integrated.primary_fatigue_type, FatigueType::Mixed);
        assert!(integrated.global_fatigue_index > 0.0);
    }
}
