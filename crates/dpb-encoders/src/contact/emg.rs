//! EMG (Electromyography) encoders

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// EMG amplitude template for muscle activation
pub struct EmgAmplitudeTemplate;

impl PopulationTemplate for EmgAmplitudeTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // RMS EMG during moderate contraction (μV)
        100.0
    }

    fn variance(&self, _context: &Context) -> f64 {
        50.0
    }

    fn name(&self) -> &str {
        "EmgAmplitudeTemplate"
    }
}

/// Muscle fatigue template (median frequency shift)
pub struct EmgFatigueTemplate;

impl PopulationTemplate for EmgFatigueTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Median frequency in Hz
        80.0
    }

    fn variance(&self, _context: &Context) -> f64 {
        15.0
    }

    fn name(&self) -> &str {
        "EmgFatigueTemplate"
    }
}

// ============================================================================
// EMG Burst Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgBurstConfig {
    pub threshold: f32,
    pub min_duration: f64,
    pub min_rest: f64,
}

impl Default for EmgBurstConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            min_duration: 0.05,  // 50 ms
            min_rest: 0.1,       // 100 ms
        }
    }
}

pub struct EmgBurstEncoder;

impl EmgBurstEncoder {
    pub fn new() -> Self {
        Self {
        }
    }

    fn calculate_rms(&self, window: &[f32]) -> f32 {
        let sum_squares: f32 = window.iter().map(|&x| x * x).sum();
        (sum_squares / window.len() as f32).sqrt()
    }
}

impl Default for EmgBurstEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EmgBurstEncoder {
    type Config = EmgBurstConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Calculate RMS envelope
        let window_size = (0.02 * sample_rate) as usize; // 20 ms window
        let mut envelope = Vec::with_capacity(samples.len());

        for i in 0..samples.len() {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2).min(samples.len());
            let window = &samples[start..end];
            envelope.push(self.calculate_rms(window));
        }

        // Detect bursts
        let mut events = Vec::new();
        let mut in_burst = false;
        let mut burst_start = 0;

        for i in 0..envelope.len() {
            if !in_burst && envelope[i] > config.threshold {
                in_burst = true;
                burst_start = i;
            } else if in_burst && envelope[i] <= config.threshold {
                in_burst = false;
                let burst_duration = (i - burst_start) as f64 * dt;

                if burst_duration >= config.min_duration {
                    let time = burst_start as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, envelope[burst_start]));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EmgBurstEncoder"
    }
}

// ============================================================================
// EMG Amplitude Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgAmplitudeConfig {
    pub window_size: usize,
    pub threshold: f32,
}

impl Default for EmgAmplitudeConfig {
    fn default() -> Self {
        Self {
            window_size: 50,
            threshold: 0.2,
        }
    }
}

pub struct EmgAmplitudeEncoder;

impl EmgAmplitudeEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for EmgAmplitudeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EmgAmplitudeEncoder {
    type Config = EmgAmplitudeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in config.window_size..samples.len() {
            let window = &samples[i - config.window_size..i];
            let rms: f32 = (window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32).sqrt();

            if rms > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, rms));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EmgAmplitudeEncoder"
    }
}

// ============================================================================
// EMG Fatigue Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmgFatigueConfig {
    pub window_size: usize,
    pub frequency_threshold: f32,
}

impl Default for EmgFatigueConfig {
    fn default() -> Self {
        Self {
            window_size: 256,
            frequency_threshold: 60.0, // Hz - below this indicates fatigue
        }
    }
}

pub struct EmgFatigueEncoder;

impl EmgFatigueEncoder {
    pub fn new() -> Self {
        Self {
        }
    }

    fn estimate_median_frequency(&self, _window: &[f32], _sample_rate: f64) -> f32 {
        // Simplified - would use FFT in production
        75.0
    }
}

impl Default for EmgFatigueEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EmgFatigueEncoder {
    type Config = EmgFatigueConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in (config.window_size..samples.len()).step_by(config.window_size / 2) {
            let window = &samples[i - config.window_size..i];
            let median_freq = self.estimate_median_frequency(window, sample_rate);

            if median_freq < config.frequency_threshold {
                let time = i as f64 * dt;
                let fatigue_index = 1.0 - (median_freq / config.frequency_threshold);
                events.push(SpikeEvent::new(time, 0, 1, fatigue_index));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EmgFatigueEncoder"
    }
}
