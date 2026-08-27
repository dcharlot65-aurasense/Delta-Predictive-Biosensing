//! Cognitive task encoders
//!
//! This module provides event-based encoders for reaction time, accuracy,
//! and attention lapse signals from cognitive tasks.
//!
//! ## Encoders
//!
//! - [`ReactionTimeEncoder`]: Encodes reaction time events
//! - [`ErrorEncoder`]: Encodes response errors and types
//! - [`LapseEncoder`]: Detects attention lapses (slow/missed responses)
//!
//! ## Population Templates
//!
//! - [`SimpleRtTemplate`]: Simple reaction time norms
//! - [`ChoiceRtTemplate`]: Choice reaction time norms
//! - [`LapseRateTemplate`]: Attention lapse rate norms

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Simple reaction time population template (ms)
pub struct SimpleRtTemplate;

impl PopulationTemplate for SimpleRtTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Simple RT increases with age
        match context.age {
            Some(age) if age < 30.0 => 220.0,
            Some(age) if age < 40.0 => 230.0,
            Some(age) if age < 50.0 => 245.0,
            Some(age) if age < 60.0 => 265.0,
            Some(age) if age < 70.0 => 290.0,
            Some(age) if age < 80.0 => 320.0,
            Some(_) => 360.0,
            None => 260.0,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        // Variability also increases with age
        match context.age {
            Some(age) if age < 40.0 => 30.0_f64.powi(2),
            Some(age) if age < 60.0 => 40.0_f64.powi(2),
            Some(_) => 60.0_f64.powi(2),
            None => 40.0_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "SimpleReactionTime"
    }
}

/// Choice reaction time population template (ms)
pub struct ChoiceRtTemplate;

impl PopulationTemplate for ChoiceRtTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Choice RT is ~100ms longer than simple RT
        match context.age {
            Some(age) if age < 30.0 => 350.0,
            Some(age) if age < 40.0 => 370.0,
            Some(age) if age < 50.0 => 400.0,
            Some(age) if age < 60.0 => 440.0,
            Some(age) if age < 70.0 => 490.0,
            Some(age) if age < 80.0 => 550.0,
            Some(_) => 620.0,
            None => 420.0,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 40.0 => 50.0_f64.powi(2),
            Some(age) if age < 60.0 => 70.0_f64.powi(2),
            Some(_) => 100.0_f64.powi(2),
            None => 60.0_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "ChoiceReactionTime"
    }
}

/// Attention lapse rate population template (lapses per minute)
pub struct LapseRateTemplate;

impl PopulationTemplate for LapseRateTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Lapse rate increases with age and varies by task
        match context.age {
            Some(age) if age < 30.0 => 0.5,
            Some(age) if age < 50.0 => 1.0,
            Some(age) if age < 70.0 => 2.0,
            Some(_) => 4.0,
            None => 1.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        1.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "LapseRate"
    }
}

/// Response accuracy population template (proportion correct)
pub struct AccuracyTemplate;

impl PopulationTemplate for AccuracyTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Accuracy decreases slightly with age
        match context.age {
            Some(age) if age < 40.0 => 0.95,
            Some(age) if age < 60.0 => 0.93,
            Some(age) if age < 80.0 => 0.90,
            Some(_) => 0.85,
            None => 0.93,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        0.05_f64.powi(2)
    }

    fn name(&self) -> &str {
        "ResponseAccuracy"
    }
}

// ============================================================================
// Reaction Time Encoder
// ============================================================================

/// Configuration for reaction time encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactionTimeConfig {
    /// Fast RT threshold (ms) - responses below this are anticipatory/fast
    pub fast_threshold: f32,
    /// Slow RT threshold (ms) - responses above this are slow/lapse
    pub slow_threshold: f32,
    /// Very slow threshold for attention lapse (ms)
    pub lapse_threshold: f32,
    /// Whether to encode individual RTs or just outliers
    pub encode_all: bool,
}

impl Default for ReactionTimeConfig {
    fn default() -> Self {
        Self {
            fast_threshold: 150.0,   // <150ms is anticipatory
            slow_threshold: 500.0,   // >500ms is slow
            lapse_threshold: 1000.0, // >1000ms is attention lapse
            encode_all: true,
        }
    }
}

/// Encodes reaction time events from cognitive tasks
///
/// Generates spike events for each response, with different channels
/// indicating normal, fast, slow, or lapse responses.
///
/// # Input Signal
/// - Channel 0: Reaction times in milliseconds
/// - Channel 1 (optional): Stimulus onset times
/// - Channel 2 (optional): Accuracy (1 = correct, 0 = error)
///
/// # Output Events
/// - Channel 0: Normal RT (magnitude = RT in ms)
/// - Channel 1: Fast RT / anticipatory (magnitude = RT)
/// - Channel 2: Slow RT (magnitude = RT)
/// - Channel 3: Attention lapse (magnitude = RT)
pub struct ReactionTimeEncoder {
    template: SimpleRtTemplate,
}

impl ReactionTimeEncoder {
    /// Create a new reaction time encoder
    pub fn new() -> Self {
        Self {
            template: SimpleRtTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &SimpleRtTemplate {
        &self.template
    }
}

impl Default for ReactionTimeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for ReactionTimeEncoder {
    type Config = ReactionTimeConfig;

    fn name(&self) -> &str {
        "ReactionTimeEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        for i in 0..num_frames {
            let rt = samples[i * samples_per_frame];

            // Skip invalid RTs (0 or negative usually means no response)
            if rt <= 0.0 {
                continue;
            }

            let time = if num_channels > 1 {
                samples[i * samples_per_frame + 1] as f64
            } else {
                i as f64 * dt
            };

            // Classify RT and emit appropriate event
            let (channel, polarity) = if rt < config.fast_threshold {
                (1, 1) // Fast/anticipatory
            } else if rt > config.lapse_threshold {
                (3, -1) // Attention lapse
            } else if rt > config.slow_threshold {
                (2, -1) // Slow
            } else {
                (0, 1) // Normal
            };

            if config.encode_all || channel != 0 {
                events.push(SpikeEvent::new(time, channel, polarity, rt));
            }
        }

        Ok(events)
    }
}

// ============================================================================
// Error Encoder
// ============================================================================

/// Configuration for error encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorConfig {
    /// Whether to encode correct responses too
    pub encode_correct: bool,
    /// Window for detecting error bursts (number of trials)
    pub burst_window: usize,
    /// Threshold for error burst (errors in window)
    pub burst_threshold: usize,
}

impl Default for ErrorConfig {
    fn default() -> Self {
        Self {
            encode_correct: false,
            burst_window: 10,
            burst_threshold: 4,
        }
    }
}

/// Response type for cognitive tasks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ResponseType {
    /// Correct response
    Correct,
    /// Commission error (responded when shouldn't have)
    Commission,
    /// Omission error (didn't respond when should have)
    Omission,
    /// Wrong response (chose incorrect option)
    Wrong,
}

/// Encodes response errors from cognitive tasks
///
/// Detects errors, error types, and error patterns (bursts).
///
/// # Input Signal
/// - Channel 0: Response accuracy (1 = correct, 0 = error)
/// - Channel 1 (optional): Error type (1 = commission, 2 = omission, 3 = wrong)
/// - Channel 2 (optional): Trial time
///
/// # Output Events
/// - Channel 0: Correct response
/// - Channel 1: Commission error
/// - Channel 2: Omission error
/// - Channel 3: Wrong response
/// - Channel 4: Error burst detected
pub struct ErrorEncoder {
    template: AccuracyTemplate,
}

impl ErrorEncoder {
    /// Create a new error encoder
    pub fn new() -> Self {
        Self {
            template: AccuracyTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &AccuracyTemplate {
        &self.template
    }
}

impl Default for ErrorEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for ErrorEncoder {
    type Config = ErrorConfig;

    fn name(&self) -> &str {
        "ErrorEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        let mut recent_errors = Vec::new();

        for i in 0..num_frames {
            let accuracy = samples[i * samples_per_frame];
            let error_type = if num_channels > 1 {
                samples[i * samples_per_frame + 1] as i32
            } else {
                0
            };
            let time = if num_channels > 2 {
                samples[i * samples_per_frame + 2] as f64
            } else {
                i as f64 * dt
            };

            let is_correct = accuracy > 0.5;

            if is_correct {
                if config.encode_correct {
                    events.push(SpikeEvent::new(time, 0, 1, 1.0));
                }
            } else {
                // Determine error type
                let channel = match error_type {
                    1 => 1, // Commission
                    2 => 2, // Omission
                    3 => 3, // Wrong
                    _ => 1, // Default to commission
                };

                events.push(SpikeEvent::new(time, channel as u32, -1, 1.0));
                recent_errors.push(i);
            }

            // Check for error burst
            recent_errors.retain(|&idx| i - idx < config.burst_window);
            if recent_errors.len() >= config.burst_threshold {
                // Only emit burst event once per burst
                if recent_errors.len() == config.burst_threshold {
                    events.push(SpikeEvent::new(time, 4, -1, recent_errors.len() as f32));
                }
            }
        }

        Ok(events)
    }
}

// ============================================================================
// Lapse Encoder
// ============================================================================

/// Configuration for attention lapse encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LapseConfig {
    /// RT threshold for lapse detection (ms)
    pub rt_threshold: f32,
    /// Whether missed responses (RT = 0) count as lapses
    pub include_misses: bool,
    /// Window for calculating lapse rate (trials)
    pub rate_window: usize,
    /// High lapse rate threshold (lapses per window)
    pub high_lapse_threshold: usize,
}

impl Default for LapseConfig {
    fn default() -> Self {
        Self {
            rt_threshold: 500.0,
            include_misses: true,
            rate_window: 20,
            high_lapse_threshold: 5,
        }
    }
}

/// Encodes attention lapses from cognitive tasks
///
/// Detects slow responses and missed responses indicative of
/// attention lapses or vigilance decrements.
///
/// # Input Signal
/// - Channel 0: Reaction times in milliseconds (0 = miss)
/// - Channel 1 (optional): Trial time
///
/// # Output Events
/// - Channel 0: Individual lapse (magnitude = RT or 0 for miss)
/// - Channel 1: High lapse rate period
/// - Channel 2: Lapse rate value (magnitude = rate)
pub struct LapseEncoder {
    template: LapseRateTemplate,
}

impl LapseEncoder {
    /// Create a new lapse encoder
    pub fn new() -> Self {
        Self {
            template: LapseRateTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &LapseRateTemplate {
        &self.template
    }
}

impl Default for LapseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for LapseEncoder {
    type Config = LapseConfig;

    fn name(&self) -> &str {
        "LapseEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        let mut recent_lapses = Vec::new();

        for i in 0..num_frames {
            let rt = samples[i * samples_per_frame];
            let time = if num_channels > 1 {
                samples[i * samples_per_frame + 1] as f64
            } else {
                i as f64 * dt
            };

            // Check for lapse
            let is_lapse = if rt <= 0.0 {
                config.include_misses
            } else {
                rt > config.rt_threshold
            };

            if is_lapse {
                recent_lapses.push(i);
                events.push(SpikeEvent::new(time, 0, -1, rt.max(0.0)));
            }

            // Calculate running lapse rate
            recent_lapses.retain(|&idx| i - idx < config.rate_window);

            // Emit lapse rate event periodically
            if i > 0 && i % config.rate_window == 0 {
                let lapse_rate = recent_lapses.len() as f32;
                events.push(SpikeEvent::new(time, 2, 1, lapse_rate));

                // Check for high lapse rate
                if recent_lapses.len() >= config.high_lapse_threshold {
                    events.push(SpikeEvent::new(time, 1, -1, lapse_rate));
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
    fn test_simple_rt_template() {
        let template = SimpleRtTemplate;
        let young = Context {
            age: Some(25.0),
            ..Default::default()
        };
        let old = Context {
            age: Some(75.0),
            ..Default::default()
        };

        assert!(template.expected_value(&young) < template.expected_value(&old));
    }

    #[test]
    fn test_reaction_time_encoder() {
        // Simulate a series of RTs
        let data = vec![
            250.0,  // Normal
            120.0,  // Fast
            450.0,  // Normal
            800.0,  // Slow
            1200.0, // Lapse
            280.0,  // Normal
        ];

        let signal = SignalBuffer::single_channel(data, 1.0);
        let encoder = ReactionTimeEncoder::new();
        let config = ReactionTimeConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert_eq!(events.len(), 6, "Should encode all RTs");

        // Check categories
        let fast = events.iter().filter(|e| e.channel == 1).count();
        let slow = events.iter().filter(|e| e.channel == 2).count();
        let lapse = events.iter().filter(|e| e.channel == 3).count();

        assert_eq!(fast, 1, "Should detect 1 fast RT");
        assert_eq!(slow, 1, "Should detect 1 slow RT");
        assert_eq!(lapse, 1, "Should detect 1 lapse");
    }

    #[test]
    fn test_error_encoder() {
        // Simulate accuracy data with some errors
        let data = vec![
            1.0, // Correct
            1.0, // Correct
            0.0, // Error
            1.0, // Correct
            0.0, // Error
            0.0, // Error
            0.0, // Error
        ];

        let signal = SignalBuffer::single_channel(data, 1.0);
        let encoder = ErrorEncoder::new();
        let config = ErrorConfig {
            burst_window: 5,
            burst_threshold: 3,
            ..Default::default()
        };

        let events = encoder.encode(&signal, &config).unwrap();

        let errors = events
            .iter()
            .filter(|e| e.channel >= 1 && e.channel <= 3)
            .count();
        assert_eq!(errors, 4, "Should detect 4 errors");

        let bursts = events.iter().filter(|e| e.channel == 4).count();
        assert!(bursts >= 1, "Should detect error burst");
    }

    #[test]
    fn test_lapse_encoder() {
        // Simulate RTs with lapses
        let mut data = Vec::new();
        for i in 0..50 {
            let rt = if i % 10 == 0 {
                800.0 // Lapse every 10 trials
            } else {
                300.0 // Normal RT
            };
            data.push(rt);
        }

        let signal = SignalBuffer::single_channel(data, 1.0);
        let encoder = LapseEncoder::new();
        let config = LapseConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();

        let lapses = events.iter().filter(|e| e.channel == 0).count();
        assert_eq!(lapses, 5, "Should detect 5 lapses");
    }
}
