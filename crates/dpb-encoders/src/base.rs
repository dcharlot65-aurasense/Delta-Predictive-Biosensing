//! Base encoder implementations
//!
//! This module provides the fundamental encoder types that other encoders build upon.

use dpb_core::{Context, DpbError, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Level Crossing Encoder
// ============================================================================

/// Configuration for level crossing encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelCrossingConfig {
    /// Threshold value for level crossing detection
    pub threshold: f32,
    /// Relative threshold (if true, threshold is relative to signal mean)
    pub relative: bool,
    /// Refractory period in seconds (minimum time between events)
    pub refractory_period: f64,
}

impl Default for LevelCrossingConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            relative: false,
            refractory_period: 0.0,
        }
    }
}

/// Level crossing encoder - detects when signal crosses a threshold
pub struct LevelCrossingEncoder {
    name: String,
}

impl LevelCrossingEncoder {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl EventEncoder for LevelCrossingEncoder {
    type Config = LevelCrossingConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Calculate threshold
        let threshold = if config.relative {
            let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
            mean + config.threshold
        } else {
            config.threshold
        };

        let mut events = Vec::new();
        let mut last_event_time = -config.refractory_period;

        for i in 1..samples.len() {
            let prev = samples[i - 1];
            let curr = samples[i];
            let time = i as f64 * dt;

            // Check refractory period
            if time - last_event_time < config.refractory_period {
                continue;
            }

            // Upward crossing
            if prev < threshold && curr >= threshold {
                events.push(SpikeEvent::new(
                    time,
                    0,
                    1,
                    (curr - threshold).abs(),
                ));
                last_event_time = time;
            }
            // Downward crossing
            else if prev >= threshold && curr < threshold {
                events.push(SpikeEvent::new(
                    time,
                    0,
                    -1,
                    (threshold - curr).abs(),
                ));
                last_event_time = time;
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Template Deviation Encoder
// ============================================================================

/// Configuration for template deviation encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateDeviationConfig {
    /// Template waveform
    pub template: Vec<f32>,
    /// Deviation threshold (in standard deviations)
    pub threshold: f32,
    /// Window size for template matching
    pub window_size: usize,
}

impl Default for TemplateDeviationConfig {
    fn default() -> Self {
        Self {
            template: vec![0.0; 10],
            threshold: 2.0,
            window_size: 100,
        }
    }
}

/// Template deviation encoder - detects deviations from expected pattern
pub struct TemplateDeviationEncoder {
    name: String,
    population_template: Option<Box<dyn PopulationTemplate>>,
}

impl TemplateDeviationEncoder {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            population_template: None,
        }
    }

    pub fn with_template(
        name: impl Into<String>,
        template: Box<dyn PopulationTemplate>,
    ) -> Self {
        Self {
            name: name.into(),
            population_template: Some(template),
        }
    }

    /// Calculate correlation between signal and template
    fn correlate(&self, signal: &[f32], template: &[f32]) -> f32 {
        if signal.len() != template.len() {
            return 0.0;
        }

        let sig_mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let tmpl_mean = template.iter().sum::<f32>() / template.len() as f32;

        let mut numerator = 0.0;
        let mut sig_var = 0.0;
        let mut tmpl_var = 0.0;

        for i in 0..signal.len() {
            let sig_dev = signal[i] - sig_mean;
            let tmpl_dev = template[i] - tmpl_mean;
            numerator += sig_dev * tmpl_dev;
            sig_var += sig_dev * sig_dev;
            tmpl_var += tmpl_dev * tmpl_dev;
        }

        if sig_var > 0.0 && tmpl_var > 0.0 {
            numerator / (sig_var * tmpl_var).sqrt()
        } else {
            0.0
        }
    }
}

impl EventEncoder for TemplateDeviationEncoder {
    type Config = TemplateDeviationConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let template = &config.template;
        let template_len = template.len();
        let mut events = Vec::new();

        if samples.len() < template_len {
            return Ok(events);
        }

        for i in 0..=samples.len() - template_len {
            let window = &samples[i..i + template_len];
            let correlation = self.correlate(window, template);

            // Low correlation indicates deviation
            let deviation = 1.0 - correlation;

            if deviation > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Derivative Encoder
// ============================================================================

/// Configuration for derivative encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeConfig {
    /// Threshold for derivative magnitude
    pub threshold: f32,
    /// Order of derivative (1 = velocity, 2 = acceleration)
    pub order: u32,
}

impl Default for DerivativeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            order: 1,
        }
    }
}

/// Derivative encoder - detects rapid changes in signal
pub struct DerivativeEncoder {
    name: String,
}

impl DerivativeEncoder {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    fn compute_derivative(&self, signal: &[f32], order: u32) -> Vec<f32> {
        if order == 0 || signal.len() < 2 {
            return signal.to_vec();
        }

        let mut result = Vec::with_capacity(signal.len() - 1);
        for i in 1..signal.len() {
            result.push(signal[i] - signal[i - 1]);
        }

        if order > 1 {
            self.compute_derivative(&result, order - 1)
        } else {
            result
        }
    }
}

impl EventEncoder for DerivativeEncoder {
    type Config = DerivativeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let derivative = self.compute_derivative(samples, config.order);
        let mut events = Vec::new();

        for (i, &value) in derivative.iter().enumerate() {
            let abs_value = value.abs();
            if abs_value > config.threshold {
                let time = (i + 1) as f64 * dt; // +1 because derivative shifts by one
                let polarity = if value > 0.0 { 1 } else { -1 };
                events.push(SpikeEvent::new(time, 0, polarity, abs_value));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Discrete Event Encoder
// ============================================================================

/// Configuration for discrete event encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteEventConfig {
    /// Event timestamps (in seconds)
    pub event_times: Vec<f64>,
    /// Event magnitudes
    pub magnitudes: Vec<f32>,
}

impl Default for DiscreteEventConfig {
    fn default() -> Self {
        Self {
            event_times: Vec::new(),
            magnitudes: Vec::new(),
        }
    }
}

/// Discrete event encoder - converts pre-detected events to spikes
pub struct DiscreteEventEncoder {
    name: String,
}

impl DiscreteEventEncoder {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl EventEncoder for DiscreteEventEncoder {
    type Config = DiscreteEventConfig;

    fn encode(&self, _signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let mut events = Vec::new();

        for (i, &time) in config.event_times.iter().enumerate() {
            let magnitude = config.magnitudes.get(i).copied().unwrap_or(1.0);
            events.push(SpikeEvent::new(time, 0, 1, magnitude));
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_level_crossing_encoder() {
        let data = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0];
        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = LevelCrossingEncoder::new("test");
        let config = LevelCrossingConfig {
            threshold: 0.3,
            relative: false,
            refractory_period: 0.0,
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_derivative_encoder() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = DerivativeEncoder::new("test");
        let config = DerivativeConfig {
            threshold: 0.05,
            order: 1,
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }
}
