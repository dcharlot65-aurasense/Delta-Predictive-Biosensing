//! Saccade detection and encoding

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Saccade peak velocity template (main sequence relationship)
pub struct SaccadeVelocityTemplate;

impl PopulationTemplate for SaccadeVelocityTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Peak velocity in degrees/second for 10-degree saccade
        400.0
    }

    fn variance(&self, _context: &Context) -> f64 {
        100.0
    }

    fn name(&self) -> &str {
        "SaccadeVelocityTemplate"
    }
}

/// Saccade latency template (time to initiate saccade)
pub struct SaccadeLatencyTemplate;

impl PopulationTemplate for SaccadeLatencyTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Latency in milliseconds
        match context.age {
            Some(age) if age < 30.0 => 200.0,
            Some(age) if age < 50.0 => 220.0,
            Some(age) if age < 70.0 => 250.0,
            Some(_) => 280.0,
            None => 220.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        40.0
    }

    fn name(&self) -> &str {
        "SaccadeLatencyTemplate"
    }
}

// ============================================================================
// Saccade Onset Encoder
// ============================================================================

/// Configuration for [`SaccadeOnsetEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaccadeOnsetConfig {
    /// Velocity above which an event is emitted, in deg/s.
    pub velocity_threshold: f32, // deg/s
    /// Shortest event accepted, in seconds.
    pub min_duration: f64,        // seconds
}

impl Default for SaccadeOnsetConfig {
    fn default() -> Self {
        Self {
            velocity_threshold: 30.0, // 30 deg/s
            min_duration: 0.02,        // 20 ms
        }
    }
}

/// Saccade onset encoder.
pub struct SaccadeOnsetEncoder;

impl SaccadeOnsetEncoder {
    /// Creates a new [`SaccadeOnsetEncoder`].
    pub fn new() -> Self {
        Self {
        }
    }
}

impl Default for SaccadeOnsetEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for SaccadeOnsetEncoder {
    type Config = SaccadeOnsetConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Calculate velocity
        let mut velocity = Vec::with_capacity(samples.len() - 1);
        for i in 1..samples.len() {
            let vel = (samples[i] - samples[i - 1]) / dt as f32;
            velocity.push(vel.abs());
        }

        let mut events = Vec::new();
        let mut in_saccade = false;
        let mut saccade_start = 0;

        for i in 0..velocity.len() {
            if !in_saccade && velocity[i] > config.velocity_threshold {
                in_saccade = true;
                saccade_start = i;
            } else if in_saccade && velocity[i] <= config.velocity_threshold {
                in_saccade = false;
                let duration = (i - saccade_start) as f64 * dt;

                if duration >= config.min_duration {
                    let time = saccade_start as f64 * dt;
                    let peak_vel = velocity[saccade_start..i]
                        .iter()
                        .copied()
                        .fold(f32::NEG_INFINITY, f32::max);
                    events.push(SpikeEvent::new(time, 0, 1, peak_vel));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "SaccadeOnsetEncoder"
    }
}

// ============================================================================
// Saccade Main Sequence Encoder
// ============================================================================

/// Configuration for [`SaccadeMainSequenceEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaccadeMainSequenceConfig {
    /// Deviation from the expected value that counts as an event, in % deviation.
    pub deviation_threshold: f32,
}

impl Default for SaccadeMainSequenceConfig {
    fn default() -> Self {
        Self {
            deviation_threshold: 0.3, // 30% deviation from main sequence
        }
    }
}

/// Saccade main sequence encoder.
pub struct SaccadeMainSequenceEncoder;

impl SaccadeMainSequenceEncoder {
    /// Creates a new [`SaccadeMainSequenceEncoder`].
    pub fn new() -> Self {
        Self {
        }
    }

    fn expected_peak_velocity(&self, amplitude: f32) -> f32 {
        // Main sequence: V_peak ≈ 20 * A^0.7 (empirical formula)
        20.0 * amplitude.powf(0.7)
    }
}

impl Default for SaccadeMainSequenceEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for SaccadeMainSequenceEncoder {
    type Config = SaccadeMainSequenceConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        // First detect saccades
        let onset_config = SaccadeOnsetConfig::default();
        let onset_encoder = SaccadeOnsetEncoder::new();
        let saccades = onset_encoder.encode(signal, &onset_config)?;

        let samples = signal.samples();
        let mut events = Vec::new();

        for saccade in saccades {
            let idx = (saccade.timestamp * signal.sample_rate()) as usize;
            if idx + 10 < samples.len() {
                // Estimate amplitude
                let amplitude = (samples[idx + 10] - samples[idx]).abs();
                let expected_vel = self.expected_peak_velocity(amplitude);
                let actual_vel = saccade.magnitude;

                let deviation = ((actual_vel - expected_vel) / expected_vel).abs();

                if deviation > config.deviation_threshold {
                    events.push(SpikeEvent::new(
                        saccade.timestamp,
                        0,
                        1,
                        deviation,
                    ));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "SaccadeMainSequenceEncoder"
    }
}

// ============================================================================
// Saccade Latency Encoder
// ============================================================================

/// Configuration for [`SaccadeLatencyEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaccadeLatencyConfig {
    /// Stimulus onset times.
    pub stimulus_times: Vec<f64>,
    /// Longest response latency accepted, in ms.
    pub max_latency: f64,
}

impl Default for SaccadeLatencyConfig {
    fn default() -> Self {
        Self {
            stimulus_times: Vec::new(),
            max_latency: 0.5, // 500 ms
        }
    }
}

/// Saccade latency encoder.
pub struct SaccadeLatencyEncoder {
    template: SaccadeLatencyTemplate,
}

impl SaccadeLatencyEncoder {
    /// Creates a new [`SaccadeLatencyEncoder`].
    pub fn new() -> Self {
        Self {
            template: SaccadeLatencyTemplate,
        }
    }
}

impl Default for SaccadeLatencyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for SaccadeLatencyEncoder {
    type Config = SaccadeLatencyConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        // First detect saccades
        let onset_config = SaccadeOnsetConfig::default();
        let onset_encoder = SaccadeOnsetEncoder::new();
        let saccades = onset_encoder.encode(signal, &onset_config)?;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_latency = self.template.expected_value(&context) / 1000.0; // Convert to seconds

        for &stimulus_time in &config.stimulus_times {
            // Find first saccade after stimulus
            if let Some(saccade) = saccades
                .iter()
                .find(|s| s.timestamp > stimulus_time && s.timestamp < stimulus_time + config.max_latency)
            {
                let latency = saccade.timestamp - stimulus_time;
                let deviation = (latency - expected_latency).abs();

                events.push(SpikeEvent::new(
                    saccade.timestamp,
                    0,
                    1,
                    (deviation * 1000.0) as f32, // Convert to ms
                ));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "SaccadeLatencyEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_saccade_velocity_template() {
        let template = SaccadeVelocityTemplate;
        let context = Context::default();
        assert_eq!(template.expected_value(&context), 400.0);
    }
}
