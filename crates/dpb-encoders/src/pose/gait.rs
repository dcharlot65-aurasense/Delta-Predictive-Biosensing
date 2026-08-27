//! Gait analysis encoders

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates for Gait
// ============================================================================

/// Cadence (steps per minute) template by age
pub struct CadenceTemplate;

impl PopulationTemplate for CadenceTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 20.0 => 120.0,
            Some(age) if age < 40.0 => 115.0,
            Some(age) if age < 60.0 => 110.0,
            Some(age) if age < 80.0 => 105.0,
            Some(_) => 95.0,
            None => 110.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        10.0
    }

    fn name(&self) -> &str {
        "CadenceTemplate"
    }
}

/// Stride length template (in meters)
pub struct StrideLengthTemplate;

impl PopulationTemplate for StrideLengthTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Based on height if available
        match context.height_cm {
            Some(height) => height * 0.007, // ~0.7% of height
            None => 1.4, // Average adult
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.2
    }

    fn name(&self) -> &str {
        "StrideLengthTemplate"
    }
}

/// Gait speed template (m/s)
pub struct GaitSpeedTemplate;

impl PopulationTemplate for GaitSpeedTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 30.0 => 1.4,
            Some(age) if age < 50.0 => 1.3,
            Some(age) if age < 70.0 => 1.2,
            Some(_) => 1.0,
            None => 1.3,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.2
    }

    fn name(&self) -> &str {
        "GaitSpeedTemplate"
    }
}

/// Stance phase duration template (% of gait cycle)
pub struct StancePhaseTemplate;

impl PopulationTemplate for StancePhaseTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        60.0 // 60% of gait cycle
    }

    fn variance(&self, _context: &Context) -> f64 {
        5.0
    }

    fn name(&self) -> &str {
        "StancePhaseTemplate"
    }
}

// ============================================================================
// Heel Strike Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeelStrikeConfig {
    pub threshold: f32,
    pub min_interval: f64,
}

impl Default for HeelStrikeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_interval: 0.3, // Minimum 0.3s between steps
        }
    }
}

pub struct HeelStrikeEncoder;

impl HeelStrikeEncoder {
    pub fn new() -> Self {
        Self {
        }
    }
}

impl Default for HeelStrikeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for HeelStrikeEncoder {
    type Config = HeelStrikeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;
        let mut last_strike = 0;

        // Detect heel strikes (peaks in vertical acceleration)
        for i in 2..samples.len() - 2 {
            if samples[i] > samples[i - 1]
                && samples[i] > samples[i + 1]
                && samples[i] > config.threshold
                && (i - last_strike) > min_samples
            {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, samples[i]));
                last_strike = i;
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "HeelStrikeEncoder"
    }
}

// ============================================================================
// Toe Off Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToeOffConfig {
    pub threshold: f32,
    pub min_interval: f64,
}

impl Default for ToeOffConfig {
    fn default() -> Self {
        Self {
            threshold: -0.3,
            min_interval: 0.3,
        }
    }
}

pub struct ToeOffEncoder;

impl ToeOffEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ToeOffEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for ToeOffEncoder {
    type Config = ToeOffConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;
        let mut last_toe_off = 0;

        // Detect toe-offs (troughs in vertical acceleration)
        for i in 2..samples.len() - 2 {
            if samples[i] < samples[i - 1]
                && samples[i] < samples[i + 1]
                && samples[i] < config.threshold
                && (i - last_toe_off) > min_samples
            {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, -1, samples[i].abs()));
                last_toe_off = i;
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "ToeOffEncoder"
    }
}

// ============================================================================
// Gait Phase Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct GaitPhaseConfig {
    pub heel_strikes: Vec<f64>,
    pub toe_offs: Vec<f64>,
}


pub struct GaitPhaseEncoder {
    template: StancePhaseTemplate,
}

impl GaitPhaseEncoder {
    pub fn new() -> Self {
        Self {
            template: StancePhaseTemplate,
        }
    }
}

impl Default for GaitPhaseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for GaitPhaseEncoder {
    type Config = GaitPhaseConfig;

    fn encode(&self, _signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let mut events = Vec::new();
        let context = Context::default();
        let expected_stance = self.template.expected_value(&context);

        // Calculate stance phase duration for each gait cycle
        for i in 0..config.heel_strikes.len() {
            let heel_strike = config.heel_strikes[i];

            // Find next toe-off after this heel strike
            if let Some(&toe_off) = config.toe_offs.iter().find(|&&t| t > heel_strike) {
                // Find next heel strike to get cycle duration
                if i + 1 < config.heel_strikes.len() {
                    let next_heel_strike = config.heel_strikes[i + 1];
                    let cycle_duration = next_heel_strike - heel_strike;
                    let stance_duration = toe_off - heel_strike;
                    let stance_percent = (stance_duration / cycle_duration) * 100.0;

                    let deviation = ((stance_percent - expected_stance) / expected_stance).abs();

                    if deviation > 0.1 {
                        // >10% deviation
                        events.push(SpikeEvent::new(
                            heel_strike,
                            0,
                            1,
                            deviation as f32,
                        ));
                    }
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "GaitPhaseEncoder"
    }
}

// ============================================================================
// Stride Time Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrideTimeConfig {
    pub threshold: f32,
}

impl Default for StrideTimeConfig {
    fn default() -> Self {
        Self { threshold: 0.15 }
    }
}

pub struct StrideTimeEncoder {
    template: CadenceTemplate,
}

impl StrideTimeEncoder {
    pub fn new() -> Self {
        Self {
            template: CadenceTemplate,
        }
    }
}

impl Default for StrideTimeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for StrideTimeEncoder {
    type Config = StrideTimeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        // First detect heel strikes
        let heel_config = HeelStrikeConfig::default();
        let heel_encoder = HeelStrikeEncoder::new();
        let heel_strikes = heel_encoder.encode(signal, &heel_config)?;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_cadence = self.template.expected_value(&context);
        let expected_stride_time = 60.0 / (expected_cadence / 2.0); // Two heel strikes per stride

        for i in 1..heel_strikes.len() {
            let stride_time = heel_strikes[i].timestamp - heel_strikes[i - 1].timestamp;
            let deviation = ((stride_time - expected_stride_time) / expected_stride_time).abs();

            if deviation > config.threshold as f64 {
                events.push(SpikeEvent::new(
                    heel_strikes[i].timestamp,
                    0,
                    1,
                    deviation as f32,
                ));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "StrideTimeEncoder"
    }
}

// ============================================================================
// Gait Asymmetry Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitAsymmetryConfig {
    pub threshold: f32,
    pub left_channel: usize,
    pub right_channel: usize,
}

impl Default for GaitAsymmetryConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            left_channel: 0,
            right_channel: 1,
        }
    }
}

pub struct GaitAsymmetryEncoder;

impl GaitAsymmetryEncoder {
    pub fn new() -> Self {
        Self
    }
}

impl Default for GaitAsymmetryEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for GaitAsymmetryEncoder {
    type Config = GaitAsymmetryConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let left_channel = signal.channel(config.left_channel);
        let right_channel = signal.channel(config.right_channel);

        if left_channel.is_none() || right_channel.is_none() {
            return Ok(Vec::new());
        }

        let left = left_channel.unwrap();
        let right = right_channel.unwrap();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        // Compare amplitudes between left and right
        let window_size = (sample_rate * 0.5) as usize; // 0.5 second windows

        for i in (window_size..left.len()).step_by(window_size / 2) {
            let left_window = &left[i - window_size..i];
            let right_window = &right[i - window_size..i];

            let left_energy: f32 = left_window.iter().map(|&x| x * x).sum();
            let right_energy: f32 = right_window.iter().map(|&x| x * x).sum();

            let asymmetry = ((left_energy - right_energy) / (left_energy + right_energy)).abs();

            if asymmetry > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, asymmetry));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "GaitAsymmetryEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_cadence_template() {
        let template = CadenceTemplate;
        let mut context = Context::default();

        context.age = Some(30.0);
        assert_eq!(template.expected_value(&context), 115.0);
    }

    #[test]
    fn test_heel_strike_encoder() {
        let mut data = vec![0.0; 1000];
        for i in (100..1000).step_by(100) {
            data[i] = 1.0;
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = HeelStrikeEncoder::new();
        let config = HeelStrikeConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }
}
