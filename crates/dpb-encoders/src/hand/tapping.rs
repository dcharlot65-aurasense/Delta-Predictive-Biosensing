//! Finger tapping encoders (for motor assessment)

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Tapping frequency template (taps per second)
pub struct TappingFrequencyTemplate;

impl PopulationTemplate for TappingFrequencyTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 30.0 => 6.5,
            Some(age) if age < 50.0 => 6.0,
            Some(age) if age < 70.0 => 5.5,
            Some(_) => 5.0,
            None => 6.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.8
    }

    fn name(&self) -> &str {
        "TappingFrequencyTemplate"
    }
}

/// Tapping amplitude template (finger aperture in cm)
pub struct TappingAmplitudeTemplate;

impl PopulationTemplate for TappingAmplitudeTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        3.0 // cm
    }

    fn variance(&self, _context: &Context) -> f64 {
        1.0
    }

    fn name(&self) -> &str {
        "TappingAmplitudeTemplate"
    }
}

// ============================================================================
// Tap Onset Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapOnsetConfig {
    pub threshold: f32,
    pub min_interval: f64,
}

impl Default for TapOnsetConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_interval: 0.1, // 100 ms
        }
    }
}

pub struct TapOnsetEncoder {
    template: TappingFrequencyTemplate,
}

impl TapOnsetEncoder {
    pub fn new() -> Self {
        Self {
            template: TappingFrequencyTemplate,
        }
    }
}

impl Default for TapOnsetEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TapOnsetEncoder {
    type Config = TapOnsetConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;
        let mut last_tap = 0;

        for i in 1..samples.len() {
            let velocity = samples[i] - samples[i - 1];

            if velocity > config.threshold && (i - last_tap) > min_samples {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, velocity));
                last_tap = i;
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TapOnsetEncoder"
    }
}

// ============================================================================
// Tap Aperture Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapApertureConfig {
    pub threshold: f32,
}

impl Default for TapApertureConfig {
    fn default() -> Self {
        Self { threshold: 1.0 }
    }
}

pub struct TapApertureEncoder {
    template: TappingAmplitudeTemplate,
}

impl TapApertureEncoder {
    pub fn new() -> Self {
        Self {
            template: TappingAmplitudeTemplate,
        }
    }
}

impl Default for TapApertureEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TapApertureEncoder {
    type Config = TapApertureConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_aperture = self.template.expected_value(&context) as f32;

        // Detect local maxima (maximum aperture)
        for i in 1..samples.len() - 1 {
            if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] {
                let deviation = (samples[i] - expected_aperture).abs();

                if deviation > config.threshold {
                    let time = i as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, deviation));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TapApertureEncoder"
    }
}

// ============================================================================
// Tap Frequency Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapFrequencyConfig {
    pub window_size: usize,
    pub threshold: f32,
}

impl Default for TapFrequencyConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            threshold: 1.0, // Hz deviation
        }
    }
}

pub struct TapFrequencyEncoder {
    template: TappingFrequencyTemplate,
}

impl TapFrequencyEncoder {
    pub fn new() -> Self {
        Self {
            template: TappingFrequencyTemplate,
        }
    }
}

impl Default for TapFrequencyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TapFrequencyEncoder {
    type Config = TapFrequencyConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        // First detect tap onsets
        let onset_config = TapOnsetConfig::default();
        let onset_encoder = TapOnsetEncoder::new();
        let taps = onset_encoder.encode(signal, &onset_config)?;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_freq = self.template.expected_value(&context);

        // Calculate frequency over windows
        for i in config.window_size..taps.len() {
            let window_taps = &taps[i - config.window_size..i];
            let duration = window_taps.last().unwrap().timestamp
                - window_taps.first().unwrap().timestamp;

            if duration > 0.0 {
                let frequency = window_taps.len() as f64 / duration;
                let deviation = (frequency - expected_freq).abs();

                if deviation > config.threshold as f64 {
                    let time = window_taps.last().unwrap().timestamp;
                    events.push(SpikeEvent::new(time, 0, 1, deviation as f32));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TapFrequencyEncoder"
    }
}

// ============================================================================
// Tap Decrement Encoder (detects amplitude decay over time)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TapDecrementConfig {
    pub threshold: f32, // Minimum decrement to detect
}

impl Default for TapDecrementConfig {
    fn default() -> Self {
        Self { threshold: 0.2 }
    }
}

pub struct TapDecrementEncoder;

impl TapDecrementEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TapDecrementEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TapDecrementEncoder {
    type Config = TapDecrementConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Detect peaks (tap amplitudes)
        let mut peaks = Vec::new();
        for i in 1..samples.len() - 1 {
            if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] {
                peaks.push((i, samples[i]));
            }
        }

        let mut events = Vec::new();

        // Compare consecutive peaks
        for i in 1..peaks.len() {
            let prev_amp = peaks[i - 1].1;
            let curr_amp = peaks[i].1;

            if prev_amp > 0.0 {
                let decrement = (prev_amp - curr_amp) / prev_amp;

                if decrement > config.threshold {
                    let time = peaks[i].0 as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, decrement));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TapDecrementEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_tapping_frequency_template() {
        let template = TappingFrequencyTemplate;
        let mut context = Context::default();

        context.age = Some(25.0);
        assert_eq!(template.expected_value(&context), 6.5);
    }
}
