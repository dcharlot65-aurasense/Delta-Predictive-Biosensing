//! Fixation stability and microsaccade encoders

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Fixation stability template (dispersion)
pub struct FixationStabilityTemplate;

impl PopulationTemplate for FixationStabilityTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Fixation dispersion in degrees
        0.5
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.2
    }

    fn name(&self) -> &str {
        "FixationStabilityTemplate"
    }
}

/// Microsaccade rate template
pub struct MicrosaccadeRateTemplate;

impl PopulationTemplate for MicrosaccadeRateTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Microsaccades per second
        1.5
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.5
    }

    fn name(&self) -> &str {
        "MicrosaccadeRateTemplate"
    }
}

// ============================================================================
// Fixation Stability Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixationStabilityConfig {
    pub window_size: usize,
    pub threshold: f32,
}

impl Default for FixationStabilityConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            threshold: 1.0, // degrees
        }
    }
}

pub struct FixationStabilityEncoder;

impl FixationStabilityEncoder {
    pub fn new() -> Self {
        Self {
        }
    }

    fn calculate_dispersion(&self, window: &[f32]) -> f32 {
        if window.is_empty() {
            return 0.0;
        }

        let mean = window.iter().sum::<f32>() / window.len() as f32;
        let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>()
            / window.len() as f32;

        variance.sqrt()
    }
}

impl Default for FixationStabilityEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for FixationStabilityEncoder {
    type Config = FixationStabilityConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in config.window_size..samples.len() {
            let window = &samples[i - config.window_size..i];
            let dispersion = self.calculate_dispersion(window);

            if dispersion > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, dispersion));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "FixationStabilityEncoder"
    }
}

// ============================================================================
// Microsaccade Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicrosaccadeConfig {
    pub velocity_threshold: f32,   // deg/s (lower than regular saccades)
    pub amplitude_threshold: f32,  // degrees
    pub min_duration: f64,
    pub max_duration: f64,
}

impl Default for MicrosaccadeConfig {
    fn default() -> Self {
        Self {
            velocity_threshold: 8.0, // deg/s
            amplitude_threshold: 0.5, // degrees (< 2 degrees)
            min_duration: 0.006,      // 6 ms
            max_duration: 0.040,      // 40 ms
        }
    }
}

pub struct MicrosaccadeEncoder;

impl MicrosaccadeEncoder {
    pub fn new() -> Self {
        Self {
        }
    }
}

impl Default for MicrosaccadeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for MicrosaccadeEncoder {
    type Config = MicrosaccadeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Calculate velocity
        let mut velocity = Vec::with_capacity(samples.len() - 1);
        for i in 1..samples.len() {
            let vel = ((samples[i] - samples[i - 1]) / dt as f32).abs();
            velocity.push(vel);
        }

        let mut events = Vec::new();
        let mut in_microsaccade = false;
        let mut ms_start = 0;

        for i in 0..velocity.len() {
            if !in_microsaccade && velocity[i] > config.velocity_threshold {
                in_microsaccade = true;
                ms_start = i;
            } else if in_microsaccade && velocity[i] <= config.velocity_threshold {
                in_microsaccade = false;
                let duration = (i - ms_start) as f64 * dt;
                let amplitude = (samples[i] - samples[ms_start]).abs();

                if duration >= config.min_duration
                    && duration <= config.max_duration
                    && amplitude <= config.amplitude_threshold
                {
                    let time = ms_start as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, amplitude));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "MicrosaccadeEncoder"
    }
}
