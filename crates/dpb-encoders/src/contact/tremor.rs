//! Tremor detection encoders (3-axis accelerometer)

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Normal physiological tremor template (8-12 Hz)
pub struct PhysiologicalTremorTemplate;

impl PopulationTemplate for PhysiologicalTremorTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        10.0 // Hz - normal physiological tremor frequency
    }

    fn variance(&self, _context: &Context) -> f64 {
        2.0
    }

    fn name(&self) -> &str {
        "PhysiologicalTremorTemplate"
    }
}

/// Pathological tremor template (3-7 Hz for Parkinson's)
pub struct PathologicalTremorTemplate;

impl PopulationTemplate for PathologicalTremorTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        5.0 // Hz - Parkinsonian rest tremor
    }

    fn variance(&self, _context: &Context) -> f64 {
        1.5
    }

    fn name(&self) -> &str {
        "PathologicalTremorTemplate"
    }
}

// ============================================================================
// Tremor Level Crossing Encoder (3-axis)
// ============================================================================

/// Configuration for [`TremorLevelCrossingEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorLevelCrossingConfig {
    /// Detection threshold.
    pub threshold: f32,
    /// Which axis to analyse.
    pub axis: usize, // 0=X, 1=Y, 2=Z, 3=magnitude
}

impl Default for TremorLevelCrossingConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            axis: 3, // Use magnitude by default
        }
    }
}

/// Tremor level crossing encoder.
pub struct TremorLevelCrossingEncoder;

impl TremorLevelCrossingEncoder {
    /// Creates a new [`TremorLevelCrossingEncoder`].
    pub fn new() -> Self {
        Self
    }

    fn calculate_magnitude(&self, x: f32, y: f32, z: f32) -> f32 {
        (x * x + y * y + z * z).sqrt()
    }
}

impl Default for TremorLevelCrossingEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TremorLevelCrossingEncoder {
    type Config = TremorLevelCrossingConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;
        let channels = signal.channel_count();

        if channels < 3 {
            return Ok(Vec::new());
        }

        let mut events = Vec::new();
        let samples_per_channel = samples.len() / channels;

        for i in 1..samples_per_channel {
            let x = samples[i];
            let y = samples[samples_per_channel + i];
            let z = samples[2 * samples_per_channel + i];

            let value = if config.axis < 3 {
                [x, y, z][config.axis]
            } else {
                self.calculate_magnitude(x, y, z)
            };

            if value > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, config.axis as u32, 1, value));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TremorLevelCrossingEncoder"
    }
}

// ============================================================================
// Tremor Frequency Encoder
// ============================================================================

/// Configuration for [`TremorFrequencyEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorFrequencyConfig {
    /// Analysis window length.
    pub window_size: usize,
    /// Frequency of interest, in Hz.
    pub target_frequency: f32, // Hz
    /// Half-width of the band around the target frequency.
    pub frequency_tolerance: f32,
}

impl Default for TremorFrequencyConfig {
    fn default() -> Self {
        Self {
            window_size: 256,
            target_frequency: 5.0,
            frequency_tolerance: 2.0,
        }
    }
}

/// Tremor frequency encoder.
pub struct TremorFrequencyEncoder;

impl TremorFrequencyEncoder {
    /// Creates a new [`TremorFrequencyEncoder`].
    pub fn new() -> Self {
        Self {
        }
    }

    fn estimate_frequency(&self, window: &[f32], sample_rate: f64) -> f32 {
        // Simple zero-crossing based frequency estimation
        let mut crossings = 0;
        let mean = window.iter().sum::<f32>() / window.len() as f32;

        for i in 1..window.len() {
            if (window[i - 1] - mean) * (window[i] - mean) < 0.0 {
                crossings += 1;
            }
        }

        let duration = window.len() as f64 / sample_rate;
        (crossings as f64 / (2.0 * duration)) as f32
    }
}

impl Default for TremorFrequencyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TremorFrequencyEncoder {
    type Config = TremorFrequencyConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in (config.window_size..samples.len()).step_by(config.window_size / 2) {
            let window = &samples[i - config.window_size..i];
            let freq = self.estimate_frequency(window, sample_rate);

            let deviation = (freq - config.target_frequency).abs();
            if deviation < config.frequency_tolerance {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, freq));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TremorFrequencyEncoder"
    }
}

// ============================================================================
// Tremor Amplitude Encoder
// ============================================================================

/// Configuration for [`TremorAmplitudeEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TremorAmplitudeConfig {
    /// Analysis window length.
    pub window_size: usize,
    /// Detection threshold.
    pub threshold: f32,
}

impl Default for TremorAmplitudeConfig {
    fn default() -> Self {
        Self {
            window_size: 100,
            threshold: 0.5,
        }
    }
}

/// Tremor amplitude encoder.
pub struct TremorAmplitudeEncoder;

impl TremorAmplitudeEncoder {
    /// Creates a new [`TremorAmplitudeEncoder`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for TremorAmplitudeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TremorAmplitudeEncoder {
    type Config = TremorAmplitudeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in config.window_size..samples.len() {
            let window = &samples[i - config.window_size..i];
            let max_val = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let min_val = window.iter().copied().fold(f32::INFINITY, f32::min);
            let amplitude = max_val - min_val;

            if amplitude > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, amplitude));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "TremorAmplitudeEncoder"
    }
}
