//! EDA (Electrodermal Activity / Galvanic Skin Response) encoders

use dpb_core::{
    Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates for EDA
// ============================================================================

/// Tonic skin conductance level template
pub struct EdaTonicTemplate;

impl PopulationTemplate for EdaTonicTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Baseline SCL in microsiemens (μS)
        10.0 // Normal range: 5-20 μS
    }

    fn variance(&self, _context: &Context) -> f64 {
        5.0
    }

    fn name(&self) -> &str {
        "EdaTonicTemplate"
    }
}

/// Skin conductance response (SCR) template
pub struct EdaScrTemplate;

impl PopulationTemplate for EdaScrTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // SCR amplitude in μS
        0.2 // Typical: 0.05-1.0 μS
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.3
    }

    fn name(&self) -> &str {
        "EdaScrTemplate"
    }
}

// ============================================================================
// EDA Level Crossing Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdaLevelCrossingConfig {
    pub threshold: f32,
    pub refractory_period: f64,
}

impl Default for EdaLevelCrossingConfig {
    fn default() -> Self {
        Self {
            threshold: 0.05, // 0.05 μS
            refractory_period: 1.0, // 1 second
        }
    }
}

pub struct EdaLevelCrossingEncoder;

impl EdaLevelCrossingEncoder {
    pub fn new() -> Self {
        Self {
        }
    }
}

impl Default for EdaLevelCrossingEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EdaLevelCrossingEncoder {
    type Config = EdaLevelCrossingConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let mut last_event_time = -config.refractory_period;

        for i in 1..samples.len() {
            let change = samples[i] - samples[i - 1];
            let time = i as f64 * dt;

            if time - last_event_time < config.refractory_period {
                continue;
            }

            if change.abs() > config.threshold {
                let polarity = if change > 0.0 { 1 } else { -1 };
                events.push(SpikeEvent::new(time, 0, polarity, change.abs()));
                last_event_time = time;
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EdaLevelCrossingEncoder"
    }
}

// ============================================================================
// EDA SCR (Skin Conductance Response) Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdaScrConfig {
    pub min_amplitude: f32,
    pub rise_time_max: f64,
    pub window_size: usize,
}

impl Default for EdaScrConfig {
    fn default() -> Self {
        Self {
            min_amplitude: 0.05,
            rise_time_max: 3.0,
            window_size: 50,
        }
    }
}

pub struct EdaScrEncoder;

impl EdaScrEncoder {
    pub fn new() -> Self {
        Self {
        }
    }

    fn detect_scr(&self, signal: &[f32], config: &EdaScrConfig, sample_rate: f64) -> Vec<usize> {
        let mut scr_onsets = Vec::new();
        let _dt = 1.0 / sample_rate;

        for i in config.window_size..signal.len() - config.window_size {
            let window_before = &signal[i - config.window_size..i];
            let window_after = &signal[i..i + config.window_size];

            let baseline = window_before.iter().sum::<f32>() / window_before.len() as f32;
            let peak = window_after
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            let amplitude = peak - baseline;

            if amplitude > config.min_amplitude {
                scr_onsets.push(i);
            }
        }

        scr_onsets
    }
}

impl Default for EdaScrEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EdaScrEncoder {
    type Config = EdaScrConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let scr_onsets = self.detect_scr(samples, config, sample_rate);
        let mut events = Vec::new();

        for onset in scr_onsets {
            let time = onset as f64 * dt;
            let magnitude = samples[onset];
            events.push(SpikeEvent::new(time, 0, 1, magnitude));
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EdaScrEncoder"
    }
}

// ============================================================================
// EDA Tonic Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdaTonicConfig {
    pub window_size: usize,
    pub threshold: f32,
}

impl Default for EdaTonicConfig {
    fn default() -> Self {
        Self {
            window_size: 500, // ~5 seconds at 100 Hz
            threshold: 0.5,
        }
    }
}

pub struct EdaTonicEncoder {
    template: EdaTonicTemplate,
}

impl EdaTonicEncoder {
    pub fn new() -> Self {
        Self {
            template: EdaTonicTemplate,
        }
    }
}

impl Default for EdaTonicEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EdaTonicEncoder {
    type Config = EdaTonicConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_tonic = self.template.expected_value(&context);

        for i in config.window_size..samples.len() {
            let window = &samples[i - config.window_size..i];
            let tonic_level = window.iter().sum::<f32>() / window.len() as f32;
            let deviation = ((tonic_level as f64 - expected_tonic) / expected_tonic).abs();

            if deviation > config.threshold as f64 {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation as f32));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EdaTonicEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_eda_level_crossing() {
        // Create signal with sharp changes that exceed threshold (0.05)
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let signal = SignalBuffer::single_channel(data, 10.0);
        let encoder = EdaLevelCrossingEncoder::new();
        let config = EdaLevelCrossingConfig {
            threshold: 0.01, // Lower threshold to trigger on smaller changes
            refractory_period: 0.1, // Short refractory period
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }
}
