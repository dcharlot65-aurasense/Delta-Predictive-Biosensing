//! Pupil diameter and response encoders

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Pupil diameter template
pub struct PupilDiameterTemplate;

impl PopulationTemplate for PupilDiameterTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Baseline pupil diameter in mm (moderate light)
        match context.age {
            Some(age) if age < 20.0 => 5.5,
            Some(age) if age < 40.0 => 5.0,
            Some(age) if age < 60.0 => 4.5,
            Some(_) => 4.0,
            None => 4.5,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.8
    }

    fn name(&self) -> &str {
        "PupilDiameterTemplate"
    }
}

/// Pupillary light reflex template (constriction velocity)
pub struct PupilLightReflexTemplate;

impl PopulationTemplate for PupilLightReflexTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Constriction velocity in mm/s
        -2.5 // Negative indicates constriction
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.5
    }

    fn name(&self) -> &str {
        "PupilLightReflexTemplate"
    }
}

// ============================================================================
// Pupil Dilation Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PupilDilationConfig {
    pub threshold: f32, // mm/s
}

impl Default for PupilDilationConfig {
    fn default() -> Self {
        Self { threshold: 0.5 }
    }
}

pub struct PupilDilationEncoder {
    template: PupilDiameterTemplate,
}

impl PupilDilationEncoder {
    pub fn new() -> Self {
        Self {
            template: PupilDiameterTemplate,
        }
    }
}

impl Default for PupilDilationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PupilDilationEncoder {
    type Config = PupilDilationConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in 1..samples.len() {
            let velocity = (samples[i] - samples[i - 1]) / dt as f32;

            if velocity.abs() > config.threshold {
                let time = i as f64 * dt;
                let polarity = if velocity > 0.0 { 1 } else { -1 }; // 1=dilation, -1=constriction
                events.push(SpikeEvent::new(time, 0, polarity, velocity.abs()));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "PupilDilationEncoder"
    }
}

// ============================================================================
// Pupil Light Reflex Encoder
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PupilLightReflexConfig {
    pub light_onset_times: Vec<f64>,
    pub response_window: f64, // seconds
    pub threshold: f32,
}

impl Default for PupilLightReflexConfig {
    fn default() -> Self {
        Self {
            light_onset_times: Vec::new(),
            response_window: 1.0,
            threshold: 0.3, // mm constriction
        }
    }
}

pub struct PupilLightReflexEncoder {
    template: PupilLightReflexTemplate,
}

impl PupilLightReflexEncoder {
    pub fn new() -> Self {
        Self {
            template: PupilLightReflexTemplate,
        }
    }

    fn find_max_constriction(
        &self,
        samples: &[f32],
        start_idx: usize,
        end_idx: usize,
    ) -> (usize, f32) {
        if start_idx >= samples.len() || end_idx > samples.len() || start_idx >= end_idx {
            return (start_idx, 0.0);
        }

        let baseline = samples[start_idx];
        let mut max_constriction = 0.0;
        let mut max_idx = start_idx;

        for (offset, &sample) in samples[start_idx..end_idx].iter().enumerate() {
            let constriction = baseline - sample;
            if constriction > max_constriction {
                max_constriction = constriction;
                max_idx = start_idx + offset;
            }
        }

        (max_idx, max_constriction)
    }
}

impl Default for PupilLightReflexEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PupilLightReflexEncoder {
    type Config = PupilLightReflexConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_velocity = self.template.expected_value(&context).abs();

        for &light_time in &config.light_onset_times {
            let start_idx = (light_time * sample_rate) as usize;
            let end_idx = ((light_time + config.response_window) * sample_rate) as usize;

            let (max_idx, constriction) = self.find_max_constriction(samples, start_idx, end_idx);

            if constriction > config.threshold {
                let latency = (max_idx - start_idx) as f64 * dt;
                let velocity = constriction / latency as f32;

                // Check if velocity deviates from expected
                let deviation = (velocity - expected_velocity as f32).abs();

                let time = max_idx as f64 * dt;
                events.push(SpikeEvent::new(time, 0, -1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "PupilLightReflexEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_pupil_diameter_template() {
        let template = PupilDiameterTemplate;
        let mut context = Context::default();

        context.age = Some(25.0);
        assert_eq!(template.expected_value(&context), 5.0);
    }
}
