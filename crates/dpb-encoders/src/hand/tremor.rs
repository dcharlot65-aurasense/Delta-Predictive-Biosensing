//! Hand tremor encoders (similar to contact tremor but for hand tracking data)

use dpb_core::{EventEncoder, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// Re-use tremor templates from contact module
pub use crate::contact::tremor::{PathologicalTremorTemplate, PhysiologicalTremorTemplate};

// ============================================================================
// Hand Tremor Encoder
// ============================================================================

/// Configuration for [`HandTremorEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandTremorConfig {
    /// Detection threshold.
    pub threshold: f32,
    /// Analysis window length.
    pub window_size: usize,
}

impl Default for HandTremorConfig {
    fn default() -> Self {
        Self {
            threshold: 0.05,
            window_size: 50,
        }
    }
}

/// Hand tremor encoder.
pub struct HandTremorEncoder;

impl HandTremorEncoder {
    /// Creates a new [`HandTremorEncoder`].
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for HandTremorEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for HandTremorEncoder {
    type Config = HandTremorConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        for i in config.window_size..samples.len() {
            let window = &samples[i - config.window_size..i];

            // Calculate tremor amplitude (standard deviation)
            let mean = window.iter().sum::<f32>() / window.len() as f32;
            let variance =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / window.len() as f32;
            let tremor_amp = variance.sqrt();

            if tremor_amp > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, tremor_amp));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "HandTremorEncoder"
    }
}
