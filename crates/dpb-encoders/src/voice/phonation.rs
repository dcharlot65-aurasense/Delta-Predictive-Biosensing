//! Phonation quality encoders (F0, jitter, shimmer, HNR)

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Fundamental frequency (F0) template
pub struct F0Template;

impl PopulationTemplate for F0Template {
    fn expected_value(&self, context: &Context) -> f64 {
        match context.sex.as_deref() {
            Some("M") | Some("Male") => 120.0, // Hz
            Some("F") | Some("Female") => 210.0,
            _ => 165.0, // Average
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        match context.sex.as_deref() {
            Some("M") | Some("Male") => 20.0,
            Some("F") | Some("Female") => 30.0,
            _ => 30.0,
        }
    }

    fn name(&self) -> &str {
        "F0Template"
    }
}

/// Jitter template (pitch perturbation)
pub struct JitterTemplate;

impl PopulationTemplate for JitterTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        0.5 // % - normal jitter < 1%
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.3
    }

    fn name(&self) -> &str {
        "JitterTemplate"
    }
}

/// Shimmer template (amplitude perturbation)
pub struct ShimmerTemplate;

impl PopulationTemplate for ShimmerTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        2.0 // % - normal shimmer < 3%
    }

    fn variance(&self, _context: &Context) -> f64 {
        1.0
    }

    fn name(&self) -> &str {
        "ShimmerTemplate"
    }
}

/// Harmonics-to-Noise Ratio template
pub struct HnrTemplate;

impl PopulationTemplate for HnrTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        20.0 // dB - normal HNR > 15 dB
    }

    fn variance(&self, _context: &Context) -> f64 {
        5.0
    }

    fn name(&self) -> &str {
        "HnrTemplate"
    }
}

// ============================================================================
// F0 (Fundamental Frequency) Encoder
// ============================================================================

/// Configuration for [`F0Encoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct F0Config {
    /// Analysis window length.
    pub window_size: usize,
    /// Detection threshold, in Hz.
    pub threshold: f32, // Hz deviation
}

impl Default for F0Config {
    fn default() -> Self {
        Self {
            window_size: 512,
            threshold: 20.0,
        }
    }
}

/// F0 encoder.
pub struct F0Encoder {
    template: F0Template,
}

impl F0Encoder {
    /// Creates a new [`F0Encoder`].
    pub fn new() -> Self {
        Self {
            template: F0Template,
        }
    }

    fn estimate_f0(&self, window: &[f32], sample_rate: f64) -> f32 {
        // Simplified autocorrelation-based F0 estimation
        let mut max_corr = 0.0;
        let mut best_lag = 0;

        let min_lag = (sample_rate / 500.0) as usize; // ~500 Hz max
        let max_lag = (sample_rate / 50.0) as usize; // ~50 Hz min

        for lag in min_lag..max_lag.min(window.len() / 2) {
            let mut corr = 0.0;
            for i in 0..(window.len() - lag) {
                corr += window[i] * window[i + lag];
            }

            if corr > max_corr {
                max_corr = corr;
                best_lag = lag;
            }
        }

        if best_lag > 0 {
            (sample_rate as f32) / (best_lag as f32)
        } else {
            0.0
        }
    }
}

impl Default for F0Encoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for F0Encoder {
    type Config = F0Config;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_f0 = self.template.expected_value(&context) as f32;

        for i in (config.window_size..samples.len()).step_by(config.window_size / 2) {
            let window = &samples[i - config.window_size..i];
            let f0 = self.estimate_f0(window, sample_rate);

            if f0 > 0.0 {
                let deviation = (f0 - expected_f0).abs();

                if deviation > config.threshold {
                    let time = i as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, deviation));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "F0Encoder"
    }
}

// ============================================================================
// Jitter Encoder
// ============================================================================

/// Configuration for [`JitterEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JitterConfig {
    /// Detection threshold.
    pub threshold: f32, // % jitter
}

impl Default for JitterConfig {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

/// Jitter encoder.
pub struct JitterEncoder;

impl JitterEncoder {
    /// Creates a new [`JitterEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn calculate_jitter(&self, periods: &[f32]) -> f32 {
        if periods.len() < 2 {
            return 0.0;
        }

        let mut abs_diffs = 0.0;
        for i in 1..periods.len() {
            abs_diffs += (periods[i] - periods[i - 1]).abs();
        }

        let mean_period = periods.iter().sum::<f32>() / periods.len() as f32;
        if mean_period > 0.0 {
            (abs_diffs / ((periods.len() - 1) as f32)) / mean_period * 100.0
        } else {
            0.0
        }
    }
}

impl Default for JitterEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for JitterEncoder {
    type Config = JitterConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Detect pitch periods (simplified zero-crossing)
        let mut periods = Vec::new();
        let mut last_crossing = 0;

        let mean = samples.iter().sum::<f32>() / samples.len() as f32;

        for i in 1..samples.len() {
            if (samples[i - 1] - mean) * (samples[i] - mean) < 0.0 && samples[i] > mean {
                let period = (i - last_crossing) as f32;
                if period > 10.0 {
                    // Filter out noise
                    periods.push(period);
                }
                last_crossing = i;
            }
        }

        let mut events = Vec::new();
        let window_size = 10;

        for i in window_size..periods.len() {
            let window = &periods[i - window_size..i];
            let jitter = self.calculate_jitter(window);

            if jitter > config.threshold {
                let time = (last_crossing as f64) * dt;
                events.push(SpikeEvent::new(time, 0, 1, jitter));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "JitterEncoder"
    }
}

// ============================================================================
// Shimmer Encoder
// ============================================================================

/// Configuration for [`ShimmerEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShimmerConfig {
    /// Detection threshold.
    pub threshold: f32, // % shimmer
}

impl Default for ShimmerConfig {
    fn default() -> Self {
        Self { threshold: 3.0 }
    }
}

/// Shimmer encoder.
pub struct ShimmerEncoder;

impl ShimmerEncoder {
    /// Creates a new [`ShimmerEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn calculate_shimmer(&self, amplitudes: &[f32]) -> f32 {
        if amplitudes.len() < 2 {
            return 0.0;
        }

        let mut abs_diffs = 0.0;
        for i in 1..amplitudes.len() {
            abs_diffs += (amplitudes[i] - amplitudes[i - 1]).abs();
        }

        let mean_amp = amplitudes.iter().sum::<f32>() / amplitudes.len() as f32;
        if mean_amp > 0.0 {
            (abs_diffs / ((amplitudes.len() - 1) as f32)) / mean_amp * 100.0
        } else {
            0.0
        }
    }
}

impl Default for ShimmerEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for ShimmerEncoder {
    type Config = ShimmerConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Extract peak amplitudes for each pitch period
        let mut amplitudes = Vec::new();
        let window_size = (0.01 * sample_rate) as usize; // 10 ms windows

        for i in (window_size..samples.len()).step_by(window_size) {
            let window = &samples[i - window_size..i];
            let peak = window
                .iter()
                .map(|&x| x.abs())
                .fold(f32::NEG_INFINITY, f32::max);
            amplitudes.push(peak);
        }

        let mut events = Vec::new();
        let shimmer_window = 10;

        for i in shimmer_window..amplitudes.len() {
            let window = &amplitudes[i - shimmer_window..i];
            let shimmer = self.calculate_shimmer(window);

            if shimmer > config.threshold {
                let time = (i * window_size) as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, shimmer));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "ShimmerEncoder"
    }
}

// ============================================================================
// HNR (Harmonics-to-Noise Ratio) Encoder
// ============================================================================

/// Configuration for [`HnrEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnrConfig {
    /// Analysis window length.
    pub window_size: usize,
    /// Detection threshold, in dB.
    pub threshold: f32, // dB
}

impl Default for HnrConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            threshold: 15.0,
        }
    }
}

/// Hnr encoder.
pub struct HnrEncoder;

impl HnrEncoder {
    /// Creates a new [`HnrEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn estimate_hnr(&self, _window: &[f32]) -> f32 {
        // Simplified HNR estimation (would use autocorrelation in production)
        20.0 // Placeholder
    }
}

impl Default for HnrEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for HnrEncoder {
    type Config = HnrConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in (config.window_size..samples.len()).step_by(config.window_size / 2) {
            let window = &samples[i - config.window_size..i];
            let hnr = self.estimate_hnr(window);

            if hnr < config.threshold {
                let time = i as f64 * dt;
                let deviation = config.threshold - hnr;
                events.push(SpikeEvent::new(time, 0, -1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "HnrEncoder"
    }
}
