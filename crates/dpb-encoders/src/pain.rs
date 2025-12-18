//! Pain and sensory testing encoders
//!
//! This module provides event-based encoders for pain perception and
//! quantitative sensory testing (QST) signals.
//!
//! ## Encoders
//!
//! - [`PainThresholdEncoder`]: Encodes pain threshold crossings
//! - [`TemporalSummationEncoder`]: Detects wind-up/temporal summation patterns
//! - [`CpmEncoder`]: Encodes conditioned pain modulation responses
//!
//! ## Population Templates
//!
//! - [`PressurePainThresholdTemplate`]: PPT norms by body site
//! - [`HeatPainThresholdTemplate`]: HPT norms
//! - [`WindUpRatioTemplate`]: Temporal summation ratio norms

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Pressure pain threshold population template (kPa)
pub struct PressurePainThresholdTemplate;

impl PopulationTemplate for PressurePainThresholdTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // PPT varies by site, age, and sex (trapezius reference site)
        let base = match context.sex.as_deref() {
            Some("Male") | Some("male") => 350.0,    // Males: ~350 kPa
            Some("Female") | Some("female") => 280.0, // Females: ~280 kPa
            _ => 315.0,
        };

        // PPT decreases with age slightly
        match context.age {
            Some(age) if age < 30.0 => base * 1.05,
            Some(age) if age < 50.0 => base,
            Some(age) if age < 70.0 => base * 0.95,
            Some(_) => base * 0.90,
            None => base,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        // High individual variability
        80.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "PressurePainThreshold"
    }
}

/// Heat pain threshold population template (°C)
pub struct HeatPainThresholdTemplate;

impl PopulationTemplate for HeatPainThresholdTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // HPT typically 40-46°C
        match context.sex.as_deref() {
            Some("Male") | Some("male") => 44.5,
            Some("Female") | Some("female") => 43.0,
            _ => 43.5,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        2.5_f64.powi(2)
    }

    fn name(&self) -> &str {
        "HeatPainThreshold"
    }
}

/// Wind-up ratio (temporal summation) population template
pub struct WindUpRatioTemplate;

impl PopulationTemplate for WindUpRatioTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Normal wind-up ratio is ~1.5-2.5
        // Higher values indicate central sensitization
        1.8
    }

    fn variance(&self, context: &Context) -> f64 {
        0.5_f64.powi(2)
    }

    fn name(&self) -> &str {
        "WindUpRatio"
    }
}

// ============================================================================
// Pain Threshold Encoder
// ============================================================================

/// Configuration for pain threshold encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainThresholdConfig {
    /// Pain rating threshold (0-10 VAS/NRS scale)
    pub pain_threshold: f32,
    /// Minimum stimulus intensity to consider (normalized 0-1)
    pub min_stimulus: f32,
    /// Minimum interval between events (seconds)
    pub min_interval: f64,
    /// Hysteresis for threshold crossing (prevents bouncing)
    pub hysteresis: f32,
}

impl Default for PainThresholdConfig {
    fn default() -> Self {
        Self {
            pain_threshold: 4.0,   // Moderate pain threshold (4/10)
            min_stimulus: 0.1,     // 10% stimulus minimum
            min_interval: 0.5,     // 500ms minimum between events
            hysteresis: 0.5,       // 0.5 point hysteresis
        }
    }
}

/// Encodes pain threshold crossings
///
/// Detects when pain ratings cross specified thresholds, useful for
/// identifying pain onset, tolerance, and intensity changes.
///
/// # Input Signal
/// - Channel 0: Pain rating (0-10 scale)
/// - Channel 1 (optional): Stimulus intensity (normalized)
///
/// # Output Events
/// - Channel 0: Threshold crossed upward (polarity +1)
/// - Channel 1: Threshold crossed downward (polarity -1)
/// - Magnitude is the stimulus intensity at crossing
pub struct PainThresholdEncoder {
    template: PressurePainThresholdTemplate,
}

impl PainThresholdEncoder {
    /// Create a new pain threshold encoder
    pub fn new() -> Self {
        Self {
            template: PressurePainThresholdTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &PressurePainThresholdTemplate {
        &self.template
    }
}

impl Default for PainThresholdEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PainThresholdEncoder {
    type Config = PainThresholdConfig;

    fn name(&self) -> &str {
        "PainThresholdEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;

        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        let mut above_threshold = samples[0] > config.pain_threshold;
        let mut last_event_idx = 0;

        for i in 1..num_frames {
            if (i - last_event_idx) < min_samples {
                continue;
            }

            let pain_rating = samples[i * samples_per_frame];
            let stimulus = if num_channels > 1 {
                samples[i * samples_per_frame + 1]
            } else {
                1.0
            };

            // Skip if stimulus too low
            if stimulus < config.min_stimulus {
                continue;
            }

            let time = i as f64 * dt;

            // Upward crossing with hysteresis
            if !above_threshold && pain_rating > config.pain_threshold + config.hysteresis {
                events.push(SpikeEvent::new(time, 0, 1, stimulus));
                above_threshold = true;
                last_event_idx = i;
            }
            // Downward crossing with hysteresis
            else if above_threshold && pain_rating < config.pain_threshold - config.hysteresis {
                events.push(SpikeEvent::new(time, 1, -1, stimulus));
                above_threshold = false;
                last_event_idx = i;
            }
        }

        Ok(events)
    }
}

// ============================================================================
// Temporal Summation Encoder
// ============================================================================

/// Configuration for temporal summation encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalSummationConfig {
    /// Number of stimuli in the train
    pub num_stimuli: usize,
    /// Inter-stimulus interval (seconds)
    pub isi: f64,
    /// Threshold for significant increase (rating points)
    pub increase_threshold: f32,
    /// Window around expected stimulus time (seconds)
    pub detection_window: f64,
}

impl Default for TemporalSummationConfig {
    fn default() -> Self {
        Self {
            num_stimuli: 10,          // Standard 10-pulse train
            isi: 1.0,                 // 1 Hz stimulation
            increase_threshold: 1.0,  // 1 point increase = significant
            detection_window: 0.2,    // ±200ms window
        }
    }
}

/// Encodes temporal summation (wind-up) patterns
///
/// Detects the progressive increase in pain ratings to repeated
/// identical stimuli, characteristic of central sensitization.
///
/// # Input Signal
/// - Channel 0: Pain rating (0-10 scale)
/// - Channel 1 (optional): Stimulus timing marker
///
/// # Output Events
/// - Channel indicates stimulus number (0 to N-1)
/// - Magnitude is the pain rating at that stimulus
/// - Final event has wind-up ratio in magnitude
pub struct TemporalSummationEncoder {
    template: WindUpRatioTemplate,
}

impl TemporalSummationEncoder {
    /// Create a new temporal summation encoder
    pub fn new() -> Self {
        Self {
            template: WindUpRatioTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &WindUpRatioTemplate {
        &self.template
    }
}

impl Default for TemporalSummationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for TemporalSummationEncoder {
    type Config = TemporalSummationConfig;

    fn name(&self) -> &str {
        "TemporalSummationEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        let window_samples = (config.detection_window * sample_rate) as usize;
        let isi_samples = (config.isi * sample_rate) as usize;

        let mut pain_ratings = Vec::new();
        let mut first_rating = 0.0_f32;

        // Extract pain ratings at each expected stimulus time
        for stim in 0..config.num_stimuli {
            let expected_sample = stim * isi_samples;

            if expected_sample >= num_frames {
                break;
            }

            // Find peak pain rating in window around expected time
            let start = expected_sample.saturating_sub(window_samples);
            let end = (expected_sample + window_samples).min(num_frames);

            let mut max_pain = 0.0_f32;
            let mut max_idx = expected_sample;

            for i in start..end {
                let pain = samples[i * samples_per_frame];
                if pain > max_pain {
                    max_pain = pain;
                    max_idx = i;
                }
            }

            let time = max_idx as f64 * dt;
            pain_ratings.push(max_pain);

            if stim == 0 {
                first_rating = max_pain;
            }

            // Emit event for each stimulus response
            events.push(SpikeEvent::new(time, stim as u32, 1, max_pain));
        }

        // Calculate and emit wind-up ratio if we have enough data
        if pain_ratings.len() >= 2 && first_rating > 0.0 {
            let last_rating = *pain_ratings.last().unwrap();
            let wind_up_ratio = last_rating / first_rating;
            let final_time = (config.num_stimuli - 1) as f64 * config.isi;

            // Emit wind-up ratio event
            events.push(SpikeEvent::new(
                final_time + 0.1,
                config.num_stimuli as u32,
                if wind_up_ratio > 1.0 { 1 } else { -1 },
                wind_up_ratio,
            ));
        }

        Ok(events)
    }
}

// ============================================================================
// CPM (Conditioned Pain Modulation) Encoder
// ============================================================================

/// Configuration for CPM encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpmConfig {
    /// Duration of baseline test stimulus (seconds)
    pub baseline_duration: f64,
    /// Duration of conditioning stimulus (seconds)
    pub conditioning_duration: f64,
    /// Duration of post-conditioning test (seconds)
    pub post_duration: f64,
    /// Minimum CPM effect to detect (% reduction)
    pub min_cpm_effect: f32,
}

impl Default for CpmConfig {
    fn default() -> Self {
        Self {
            baseline_duration: 30.0,
            conditioning_duration: 60.0,
            post_duration: 30.0,
            min_cpm_effect: 10.0,  // 10% minimum reduction
        }
    }
}

/// Encodes conditioned pain modulation (CPM) responses
///
/// Detects the inhibitory effect of a conditioning pain stimulus
/// on test pain perception, indicating descending pain modulation.
///
/// # Input Signal
/// - Channel 0: Test pain rating (0-10)
/// - Channel 1: Conditioning stimulus indicator (0/1)
///
/// # Output Events
/// - Channel 0: Baseline pain level
/// - Channel 1: During conditioning pain level
/// - Channel 2: CPM effect (magnitude = % change)
pub struct CpmEncoder {
    template: PressurePainThresholdTemplate,
}

impl CpmEncoder {
    /// Create a new CPM encoder
    pub fn new() -> Self {
        Self {
            template: PressurePainThresholdTemplate,
        }
    }
}

impl Default for CpmEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for CpmEncoder {
    type Config = CpmConfig;

    fn name(&self) -> &str {
        "CpmEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        let baseline_samples = (config.baseline_duration * sample_rate) as usize;
        let conditioning_samples = (config.conditioning_duration * sample_rate) as usize;

        // Calculate baseline pain (first period)
        let baseline_end = baseline_samples.min(num_frames);
        let mut baseline_sum = 0.0_f32;
        for i in 0..baseline_end {
            baseline_sum += samples[i * samples_per_frame];
        }
        let baseline_pain = if baseline_end > 0 {
            baseline_sum / baseline_end as f32
        } else {
            0.0
        };

        // Emit baseline event
        let baseline_time = baseline_end as f64 * dt / 2.0;
        events.push(SpikeEvent::new(baseline_time, 0, 1, baseline_pain));

        // Calculate conditioning period pain
        let conditioning_start = baseline_samples;
        let conditioning_end = (conditioning_start + conditioning_samples).min(num_frames);

        if conditioning_end > conditioning_start {
            let mut cond_sum = 0.0_f32;
            for i in conditioning_start..conditioning_end {
                cond_sum += samples[i * samples_per_frame];
            }
            let conditioning_pain = cond_sum / (conditioning_end - conditioning_start) as f32;

            // Emit conditioning event
            let cond_time = (conditioning_start + conditioning_end) as f64 * dt / 2.0;
            events.push(SpikeEvent::new(cond_time, 1, 1, conditioning_pain));

            // Calculate CPM effect (% change from baseline)
            if baseline_pain > 0.0 {
                let cpm_effect = (baseline_pain - conditioning_pain) / baseline_pain * 100.0;

                if cpm_effect.abs() >= config.min_cpm_effect {
                    let effect_time = conditioning_end as f64 * dt;
                    let polarity = if cpm_effect > 0.0 { 1 } else { -1 }; // Positive = inhibition
                    events.push(SpikeEvent::new(effect_time, 2, polarity, cpm_effect.abs()));
                }
            }
        }

        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_ppt_template() {
        let template = PressurePainThresholdTemplate;

        let male = Context {
            sex: Some("Male".to_string()),
            ..Default::default()
        };
        let female = Context {
            sex: Some("Female".to_string()),
            ..Default::default()
        };

        assert!(template.expected_value(&male) > template.expected_value(&female));
    }

    #[test]
    fn test_pain_threshold_encoder() {
        // Simulate ramping pain rating
        let mut data = Vec::new();
        for i in 0..200 {
            let pain = (i as f32 / 20.0).min(10.0);
            data.push(pain);
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = PainThresholdEncoder::new();
        let config = PainThresholdConfig {
            pain_threshold: 5.0,
            ..Default::default()
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect threshold crossing");
    }

    #[test]
    fn test_temporal_summation_encoder() {
        // Simulate increasing pain to repeated stimuli (wind-up)
        let mut data = Vec::new();
        let sample_rate = 100.0;

        for stim in 0..10 {
            // Pain increases with each stimulus (wind-up)
            let pain = 3.0 + stim as f32 * 0.4; // 3.0 to 6.6

            // 1 second per stimulus
            for _ in 0..100 {
                data.push(pain);
            }
        }

        let signal = SignalBuffer::single_channel(data, sample_rate);
        let encoder = TemporalSummationEncoder::new();
        let config = TemporalSummationConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(events.len() >= 10, "Should detect all stimulus responses");

        // Check for wind-up ratio event
        let wur_event = events.iter().find(|e| e.channel == 10);
        assert!(wur_event.is_some(), "Should emit wind-up ratio");
        if let Some(wur) = wur_event {
            assert!(wur.magnitude > 1.5, "Should show significant wind-up");
        }
    }

    #[test]
    fn test_cpm_encoder() {
        // Simulate CPM: baseline pain, then reduced pain during conditioning
        let mut data = Vec::new();
        let sample_rate = 10.0; // Low rate for longer duration

        // Baseline (30s at 6/10 pain)
        for _ in 0..300 {
            data.push(6.0);
        }

        // Conditioning period (60s at 4/10 - CPM effect)
        for _ in 0..600 {
            data.push(4.0);
        }

        let signal = SignalBuffer::single_channel(data, sample_rate);
        let encoder = CpmEncoder::new();
        let config = CpmConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(events.len() >= 2, "Should detect baseline and conditioning");

        // Check for CPM effect event
        let cpm_event = events.iter().find(|e| e.channel == 2);
        assert!(cpm_event.is_some(), "Should detect CPM effect");
    }
}
