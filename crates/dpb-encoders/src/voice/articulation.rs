//! Articulation encoders (formants, vowel space)

use dpb_core::{Context, DpbError, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Formant frequency template (F1, F2, F3)
pub struct FormantTemplate;

impl PopulationTemplate for FormantTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // F1 for /a/ vowel (Hz)
        match context.sex.as_deref() {
            Some("M") | Some("Male") => 700.0,
            Some("F") | Some("Female") => 850.0,
            _ => 775.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        100.0
    }

    fn name(&self) -> &str {
        "FormantTemplate"
    }
}

/// Vowel space area template
pub struct VowelSpaceTemplate;

impl PopulationTemplate for VowelSpaceTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Vowel space area (Hz²)
        300000.0
    }

    fn variance(&self, _context: &Context) -> f64 {
        50000.0
    }

    fn name(&self) -> &str {
        "VowelSpaceTemplate"
    }
}

// ============================================================================
// Formant Encoder
// ============================================================================

/// Configuration for [`FormantEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormantConfig {
    /// Analysis window length.
    pub window_size: usize,
    /// How many formants to track.
    pub num_formants: usize,
    /// Detection threshold, in Hz.
    pub threshold: f32, // Hz deviation
}

impl Default for FormantConfig {
    fn default() -> Self {
        Self {
            window_size: 512,
            num_formants: 3,
            threshold: 150.0,
        }
    }
}

/// Formant encoder.
pub struct FormantEncoder {
    template: FormantTemplate,
}

impl FormantEncoder {
    /// Creates a new [`FormantEncoder`].
    pub fn new() -> Self {
        Self {
            template: FormantTemplate,
        }
    }

    fn estimate_formants(&self, _window: &[f32], _sample_rate: f64) -> Vec<f32> {
        // Simplified formant tracking (would use LPC in production)
        vec![700.0, 1220.0, 2600.0] // Example F1, F2, F3
    }
}

impl Default for FormantEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for FormantEncoder {
    type Config = FormantConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_f1 = self.template.expected_value(&context) as f32;

        for i in (config.window_size..samples.len()).step_by(config.window_size / 2) {
            let window = &samples[i - config.window_size..i];
            let formants = self.estimate_formants(window, sample_rate);

            if !formants.is_empty() {
                let deviation = (formants[0] - expected_f1).abs();

                if deviation > config.threshold {
                    let time = i as f64 * dt;
                    events.push(SpikeEvent::new(time, 0, 1, deviation));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "FormantEncoder"
    }
}

// ============================================================================
// Vowel Space Encoder
// ============================================================================

/// Configuration for [`VowelSpaceEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VowelSpaceConfig {
    /// Vowel phonemes to track (e.g., /a/, /i/, /u/)
    pub vowels: Vec<String>,
    /// Threshold for vowel space reduction
    pub threshold: f32,
}

impl Default for VowelSpaceConfig {
    fn default() -> Self {
        Self {
            vowels: vec!["a".to_string(), "i".to_string(), "u".to_string()],
            threshold: 0.3, // 30% reduction
        }
    }
}

/// Vowel space encoder.
pub struct VowelSpaceEncoder;

impl VowelSpaceEncoder {
    /// Creates a new [`VowelSpaceEncoder`].
    pub fn new() -> Self {
        Self {
        }
    }

}

impl Default for VowelSpaceEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for VowelSpaceEncoder {
    type Config = VowelSpaceConfig;

    /// Not implemented.
    ///
    /// Vowel space area is defined over pre-segmented vowel tokens, so this
    /// needs phoneme-level annotation that the raw signal does not carry.
    /// Returning an empty event list instead would be indistinguishable from
    /// "this recording contains no vowels".
    fn encode(&self, _signal: &dyn Signal, _config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        Err(DpbError::Other(
            "VowelSpaceEncoder requires phoneme-level vowel segmentation, which \
             is not implemented"
                .to_string(),
        ))
    }

    fn name(&self) -> &str {
        "VowelSpaceEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_formant_template() {
        let template = FormantTemplate;
        let mut context = Context::default();

        context.sex = Some("Male".to_string());
        assert_eq!(template.expected_value(&context), 700.0);

        context.sex = Some("Female".to_string());
        assert_eq!(template.expected_value(&context), 850.0);
    }
}
