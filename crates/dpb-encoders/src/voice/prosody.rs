//! Prosody encoders (speech rate, pauses, intonation)

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Speech rate template (syllables per second)
pub struct SpeechRateTemplate;

impl PopulationTemplate for SpeechRateTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Normal conversational speech
        match context.age {
            Some(age) if age < 30.0 => 5.0,
            Some(age) if age < 50.0 => 4.5,
            Some(age) if age < 70.0 => 4.0,
            Some(_) => 3.5,
            None => 4.5,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.8
    }

    fn name(&self) -> &str {
        "SpeechRateTemplate"
    }
}

/// Pause duration template
pub struct PauseDurationTemplate;

impl PopulationTemplate for PauseDurationTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Average pause duration in seconds
        0.5
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.3
    }

    fn name(&self) -> &str {
        "PauseDurationTemplate"
    }
}

// ============================================================================
// Speech Rate Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechRateConfig {
    pub window_size: f64, // seconds
    pub threshold: f32,    // syllables/second deviation
}

impl Default for SpeechRateConfig {
    fn default() -> Self {
        Self {
            window_size: 5.0,
            threshold: 1.0,
        }
    }
}

pub struct SpeechRateEncoder {
    template: SpeechRateTemplate,
}

impl SpeechRateEncoder {
    pub fn new() -> Self {
        Self {
            template: SpeechRateTemplate,
        }
    }

    fn estimate_syllable_rate(&self, window: &[f32], sample_rate: f64) -> f32 {
        // Simplified syllable rate estimation based on amplitude envelope peaks
        let mut peaks = 0;
        let threshold = 0.3;

        for i in 1..window.len() - 1 {
            if window[i] > window[i - 1]
                && window[i] > window[i + 1]
                && window[i] > threshold
            {
                peaks += 1;
            }
        }

        let duration = window.len() as f64 / sample_rate;
        peaks as f32 / duration as f32
    }
}

impl Default for SpeechRateEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for SpeechRateEncoder {
    type Config = SpeechRateConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let window_samples = (config.window_size * sample_rate) as usize;
        let mut events = Vec::new();

        let context = Context::default();
        let expected_rate = self.template.expected_value(&context) as f32;

        for i in (window_samples..samples.len()).step_by(window_samples / 2) {
            let window = &samples[i - window_samples..i];
            let rate = self.estimate_syllable_rate(window, sample_rate);

            let deviation = (rate - expected_rate).abs();

            if deviation > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "SpeechRateEncoder"
    }
}

// ============================================================================
// Pause Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauseConfig {
    pub min_pause_duration: f64, // seconds
    pub energy_threshold: f32,
}

impl Default for PauseConfig {
    fn default() -> Self {
        Self {
            min_pause_duration: 0.2,
            energy_threshold: 0.01,
        }
    }
}

pub struct PauseEncoder {
    template: PauseDurationTemplate,
}

impl PauseEncoder {
    pub fn new() -> Self {
        Self {
            template: PauseDurationTemplate,
        }
    }

    fn calculate_energy(&self, window: &[f32]) -> f32 {
        window.iter().map(|&x| x * x).sum::<f32>() / window.len() as f32
    }
}

impl Default for PauseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PauseEncoder {
    type Config = PauseConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let window_size = (0.01 * sample_rate) as usize; // 10 ms windows
        let mut events = Vec::new();

        let mut in_pause = false;
        let mut pause_start = 0;

        for i in (window_size..samples.len()).step_by(window_size) {
            let window = &samples[i - window_size..i];
            let energy = self.calculate_energy(window);

            if !in_pause && energy < config.energy_threshold {
                in_pause = true;
                pause_start = i;
            } else if in_pause && energy >= config.energy_threshold {
                in_pause = false;
                let pause_duration = (i - pause_start) as f64 * dt;

                if pause_duration >= config.min_pause_duration {
                    let time = pause_start as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, pause_duration as f32));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "PauseEncoder"
    }
}

// ============================================================================
// Intonation Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntonationConfig {
    pub window_size: usize,
    pub threshold: f32, // Hz/s pitch change rate
}

impl Default for IntonationConfig {
    fn default() -> Self {
        Self {
            window_size: 256,
            threshold: 50.0,
        }
    }
}

pub struct IntonationEncoder;

impl IntonationEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for IntonationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for IntonationEncoder {
    type Config = IntonationConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Simplified: track rapid F0 changes
        let mut events = Vec::new();
        let mut prev_f0 = 0.0f32;

        for i in (config.window_size..samples.len()).step_by(config.window_size) {
            // Simplified F0 estimation
            let _window = &samples[i - config.window_size..i];
            let f0 = 100.0f32; // Placeholder

            if prev_f0 > 0.0f32 {
                let f0_change_rate = (f0 - prev_f0).abs() / (config.window_size as f32 * dt as f32);

                if f0_change_rate > config.threshold {
                    let time = i as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, f0_change_rate));
                }
            }

            prev_f0 = f0;
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "IntonationEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_rate_template() {
        let template = SpeechRateTemplate;
        let mut context = Context::default();

        context.age = Some(25.0);
        assert_eq!(template.expected_value(&context), 5.0);

        context.age = Some(75.0);
        assert_eq!(template.expected_value(&context), 3.5);
    }
}
