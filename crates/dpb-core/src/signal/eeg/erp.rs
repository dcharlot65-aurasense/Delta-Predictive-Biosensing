//! Event-Related Potential (ERP) analysis
//!
//! Provides tools for extracting and analyzing ERPs from EEG data,
//! including detection of standard components like P300, N400, and MMN.

/// ERP component information
#[derive(Debug, Clone)]
pub struct ErpComponent {
    /// Component name (e.g., "P300", "N400")
    pub name: String,
    /// Latency from stimulus onset in milliseconds
    pub latency_ms: f64,
    /// Peak amplitude in microvolts
    pub amplitude_uv: f64,
    /// Channel where component was maximal
    pub peak_channel: Option<String>,
    /// Area under the curve
    pub area: f64,
}

/// Extracted ERP waveform
#[derive(Debug, Clone)]
pub struct Erp {
    /// Time points in milliseconds relative to stimulus
    pub times: Vec<f64>,
    /// Averaged waveform (microvolts)
    pub waveform: Vec<f64>,
    /// Number of epochs averaged
    pub n_epochs: usize,
    /// Standard error at each time point
    pub standard_error: Vec<f64>,
    /// Sample rate in Hz
    pub sample_rate: f64,
}

/// ERP Analyzer for extracting and analyzing event-related potentials
#[derive(Debug, Clone)]
pub struct ErpAnalyzer {
    /// Sample rate of EEG data
    pub sample_rate: f64,
    /// Baseline window (start_ms, end_ms) relative to stimulus
    pub baseline_window: (f64, f64),
}

impl Default for ErpAnalyzer {
    fn default() -> Self {
        Self {
            sample_rate: 256.0,
            baseline_window: (-200.0, 0.0),
        }
    }
}

impl ErpAnalyzer {
    /// Create new analyzer with specified sample rate
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            baseline_window: (-200.0, 0.0),
        }
    }

    /// Set baseline correction window
    pub fn with_baseline(mut self, start_ms: f64, end_ms: f64) -> Self {
        self.baseline_window = (start_ms, end_ms);
        self
    }

    /// Extract ERP by averaging time-locked epochs
    pub fn extract_erp(
        &self,
        eeg: &[f64],
        event_samples: &[usize],
        epoch_window: (f64, f64),
    ) -> Erp {
        let pre_samples = (epoch_window.0.abs() * self.sample_rate / 1000.0) as usize;
        let post_samples = (epoch_window.1 * self.sample_rate / 1000.0) as usize;
        let epoch_length = pre_samples + post_samples;

        let mut epochs: Vec<Vec<f64>> = Vec::new();

        for &event in event_samples {
            if event >= pre_samples && event + post_samples <= eeg.len() {
                let start = event - pre_samples;
                let end = event + post_samples;
                let epoch: Vec<f64> = eeg[start..end].to_vec();
                epochs.push(epoch);
            }
        }

        if epochs.is_empty() {
            return Erp {
                times: Vec::new(),
                waveform: Vec::new(),
                n_epochs: 0,
                standard_error: Vec::new(),
                sample_rate: self.sample_rate,
            };
        }

        // Apply baseline correction
        let baseline_start =
            ((self.baseline_window.0 - epoch_window.0) * self.sample_rate / 1000.0) as usize;
        let baseline_end =
            ((self.baseline_window.1 - epoch_window.0) * self.sample_rate / 1000.0) as usize;

        for epoch in epochs.iter_mut() {
            if baseline_end <= epoch.len() && baseline_start < baseline_end {
                let baseline_mean: f64 = epoch[baseline_start..baseline_end].iter().sum::<f64>()
                    / (baseline_end - baseline_start) as f64;
                for sample in epoch.iter_mut() {
                    *sample -= baseline_mean;
                }
            }
        }

        // Average epochs
        let n_epochs = epochs.len();
        let mut waveform = vec![0.0; epoch_length];

        for epoch in &epochs {
            for (i, &val) in epoch.iter().enumerate() {
                waveform[i] += val;
            }
        }
        for val in waveform.iter_mut() {
            *val /= n_epochs as f64;
        }

        // Calculate standard error
        let mut variance = vec![0.0; epoch_length];
        for epoch in &epochs {
            for (i, &val) in epoch.iter().enumerate() {
                variance[i] += (val - waveform[i]).powi(2);
            }
        }
        let standard_error: Vec<f64> = variance
            .iter()
            .map(|v| (v / (n_epochs as f64 - 1.0).max(1.0)).sqrt() / (n_epochs as f64).sqrt())
            .collect();

        let times: Vec<f64> = (0..epoch_length)
            .map(|i| epoch_window.0 + (i as f64 * 1000.0 / self.sample_rate))
            .collect();

        Erp {
            times,
            waveform,
            n_epochs,
            standard_error,
            sample_rate: self.sample_rate,
        }
    }

    /// Detect P300 component (positive peak 250-500ms post-stimulus)
    pub fn detect_p300(&self, erp: &Erp) -> Option<ErpComponent> {
        self.detect_component(erp, "P300", 250.0, 500.0, true)
    }

    /// Detect N400 component (negative peak 300-500ms, semantic processing)
    pub fn detect_n400(&self, erp: &Erp) -> Option<ErpComponent> {
        self.detect_component(erp, "N400", 300.0, 500.0, false)
    }

    /// Detect N170 component (face processing, 150-200ms)
    pub fn detect_n170(&self, erp: &Erp) -> Option<ErpComponent> {
        self.detect_component(erp, "N170", 150.0, 200.0, false)
    }

    /// Detect N100 component (auditory processing, 80-120ms)
    pub fn detect_n100(&self, erp: &Erp) -> Option<ErpComponent> {
        self.detect_component(erp, "N100", 80.0, 120.0, false)
    }

    /// Detect P100 component (visual processing, 80-130ms)
    pub fn detect_p100(&self, erp: &Erp) -> Option<ErpComponent> {
        self.detect_component(erp, "P100", 80.0, 130.0, true)
    }

    /// Generic component detection
    fn detect_component(
        &self,
        erp: &Erp,
        name: &str,
        start_ms: f64,
        end_ms: f64,
        positive: bool,
    ) -> Option<ErpComponent> {
        if erp.waveform.is_empty() {
            return None;
        }

        let start_idx = erp.times.iter().position(|&t| t >= start_ms)?;
        let end_idx = erp
            .times
            .iter()
            .position(|&t| t >= end_ms)
            .unwrap_or(erp.times.len());

        if start_idx >= end_idx {
            return None;
        }

        let window = &erp.waveform[start_idx..end_idx];
        let (peak_idx, amplitude) = if positive {
            window
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?
        } else {
            window
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())?
        };

        let global_idx = start_idx + peak_idx;
        let latency_ms = erp.times[global_idx];

        let dt = 1000.0 / erp.sample_rate;
        let area: f64 = window.iter().map(|v| v.abs() * dt).sum();

        Some(ErpComponent {
            name: name.to_string(),
            latency_ms,
            amplitude_uv: *amplitude,
            peak_channel: None,
            area,
        })
    }

    /// Calculate Mismatch Negativity (MMN) from standard and deviant ERPs
    pub fn calculate_mmn(&self, standard_erp: &Erp, deviant_erp: &Erp) -> Option<Erp> {
        if standard_erp.waveform.len() != deviant_erp.waveform.len() {
            return None;
        }

        let difference: Vec<f64> = deviant_erp
            .waveform
            .iter()
            .zip(standard_erp.waveform.iter())
            .map(|(d, s)| d - s)
            .collect();

        Some(Erp {
            times: standard_erp.times.clone(),
            waveform: difference,
            n_epochs: standard_erp.n_epochs.min(deviant_erp.n_epochs),
            standard_error: vec![0.0; standard_erp.waveform.len()],
            sample_rate: self.sample_rate,
        })
    }

    /// Detect MMN component from difference wave (100-250ms)
    pub fn detect_mmn(&self, mmn_wave: &Erp) -> Option<ErpComponent> {
        self.detect_component(mmn_wave, "MMN", 100.0, 250.0, false)
    }

    /// Calculate peak-to-peak amplitude in a window
    pub fn peak_to_peak(&self, erp: &Erp, window: (f64, f64)) -> f64 {
        let start_idx = erp.times.iter().position(|&t| t >= window.0).unwrap_or(0);
        let end_idx = erp
            .times
            .iter()
            .position(|&t| t >= window.1)
            .unwrap_or(erp.times.len());

        if start_idx >= end_idx {
            return 0.0;
        }

        let window_data = &erp.waveform[start_idx..end_idx];
        let max = window_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = window_data.iter().cloned().fold(f64::INFINITY, f64::min);

        max - min
    }

    /// Calculate mean amplitude in a time window
    pub fn mean_amplitude(&self, erp: &Erp, window: (f64, f64)) -> f64 {
        let start_idx = erp.times.iter().position(|&t| t >= window.0).unwrap_or(0);
        let end_idx = erp
            .times
            .iter()
            .position(|&t| t >= window.1)
            .unwrap_or(erp.times.len());

        if start_idx >= end_idx {
            return 0.0;
        }

        let window_data = &erp.waveform[start_idx..end_idx];
        window_data.iter().sum::<f64>() / window_data.len() as f64
    }

    /// Calculate 50% area latency
    pub fn fractional_area_latency(
        &self,
        erp: &Erp,
        window: (f64, f64),
        fraction: f64,
    ) -> Option<f64> {
        let start_idx = erp.times.iter().position(|&t| t >= window.0)?;
        let end_idx = erp
            .times
            .iter()
            .position(|&t| t >= window.1)
            .unwrap_or(erp.times.len());

        if start_idx >= end_idx {
            return None;
        }

        let window_data = &erp.waveform[start_idx..end_idx];
        let total_area: f64 = window_data.iter().map(|v| v.abs()).sum();
        let target_area = total_area * fraction;

        let mut cumulative = 0.0;
        for (i, &val) in window_data.iter().enumerate() {
            cumulative += val.abs();
            if cumulative >= target_area {
                return Some(erp.times[start_idx + i]);
            }
        }

        None
    }
}

/// ERP Generator for creating synthetic ERPs
#[derive(Debug, Clone)]
pub struct ErpGenerator {
    /// Sample rate in Hz
    pub sample_rate: f64,
}

impl Default for ErpGenerator {
    fn default() -> Self {
        Self { sample_rate: 256.0 }
    }
}

impl ErpGenerator {
    /// Generate synthetic P300 component
    pub fn generate_p300(&self, amplitude_uv: f64, latency_ms: f64, jitter_ms: f64) -> Vec<f64> {
        self.generate_component(amplitude_uv, latency_ms, 100.0, jitter_ms, true)
    }

    /// Generate synthetic N400 component
    pub fn generate_n400(&self, amplitude_uv: f64, latency_ms: f64, jitter_ms: f64) -> Vec<f64> {
        self.generate_component(amplitude_uv, latency_ms, 150.0, jitter_ms, false)
    }

    /// Generate a Gaussian-shaped ERP component
    fn generate_component(
        &self,
        amplitude: f64,
        latency_ms: f64,
        width_ms: f64,
        jitter_ms: f64,
        positive: bool,
    ) -> Vec<f64> {
        let total_ms = 1000.0;
        let n_samples = (total_ms * self.sample_rate / 1000.0) as usize;
        let mut signal = vec![0.0; n_samples];

        let actual_latency = latency_ms + rand::random::<f64>() * 2.0 * jitter_ms - jitter_ms;
        let sigma = width_ms / 2.355;

        for i in 0..n_samples {
            let t_ms = (i as f64 * 1000.0 / self.sample_rate) - 200.0;
            let gaussian = (-((t_ms - actual_latency).powi(2)) / (2.0 * sigma.powi(2))).exp();
            signal[i] = if positive {
                amplitude * gaussian
            } else {
                -amplitude * gaussian
            };
        }

        signal
    }

    /// Generate oddball paradigm data
    pub fn generate_oddball_paradigm(
        &self,
        n_trials: usize,
        target_probability: f64,
    ) -> OddballData {
        let mut standards = Vec::new();
        let mut targets = Vec::new();
        let mut is_target = Vec::new();

        for _ in 0..n_trials {
            let target = rand::random::<f64>() < target_probability;
            is_target.push(target);

            let n100 = self.generate_component(5.0, 100.0, 30.0, 10.0, false);
            let p200 = self.generate_component(3.0, 200.0, 40.0, 10.0, true);

            let mut trial: Vec<f64> = n100.iter().zip(p200.iter()).map(|(a, b)| a + b).collect();

            if target {
                let p300 = self.generate_p300(8.0, 350.0, 30.0);
                for (i, val) in p300.iter().enumerate() {
                    if i < trial.len() {
                        trial[i] += val;
                    }
                }
                targets.push(trial);
            } else {
                standards.push(trial);
            }
        }

        OddballData {
            standards,
            targets,
            is_target,
            sample_rate: self.sample_rate,
        }
    }
}

/// Data from oddball paradigm
#[derive(Debug, Clone)]
pub struct OddballData {
    /// Standard (non-target) trials
    pub standards: Vec<Vec<f64>>,
    /// Target trials
    pub targets: Vec<Vec<f64>>,
    /// Trial type indicators
    pub is_target: Vec<bool>,
    /// Sample rate
    pub sample_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erp_analyzer_creation() {
        let analyzer = ErpAnalyzer::new(256.0);
        assert_eq!(analyzer.sample_rate, 256.0);
    }

    #[test]
    fn test_p300_detection() {
        let generator = ErpGenerator::default();
        let analyzer = ErpAnalyzer::new(256.0);

        let p300_signal = generator.generate_p300(10.0, 350.0, 0.0);

        let erp = Erp {
            times: (0..p300_signal.len())
                .map(|i| -200.0 + i as f64 * 1000.0 / 256.0)
                .collect(),
            waveform: p300_signal,
            n_epochs: 1,
            standard_error: vec![],
            sample_rate: 256.0,
        };

        let component = analyzer.detect_p300(&erp);
        assert!(component.is_some());

        let p300 = component.unwrap();
        assert!(p300.latency_ms > 250.0 && p300.latency_ms < 500.0);
        assert!(p300.amplitude_uv > 0.0);
    }

    #[test]
    fn test_oddball_generation() {
        let generator = ErpGenerator::default();
        let data = generator.generate_oddball_paradigm(100, 0.2);

        assert!(!data.standards.is_empty());
        assert!(!data.targets.is_empty());

        let target_count = data.is_target.iter().filter(|&&t| t).count();
        assert!(target_count > 5 && target_count < 50);
    }
}
