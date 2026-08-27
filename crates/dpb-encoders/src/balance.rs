//! Balance and postural control encoders
//!
//! This module provides event-based encoders for center of pressure (CoP) and
//! postural stability analysis from force plate or posturography data.
//!
//! ## Encoders
//!
//! - [`CopSwayEncoder`]: Detects CoP excursions beyond stability thresholds
//! - [`CopVelocityEncoder`]: Encodes rapid CoP velocity changes
//! - [`StabilityLimitEncoder`]: Detects approaches to limits of stability
//!
//! ## Population Templates
//!
//! - [`SwayAreaTemplate`]: Age-normed sway area values
//! - [`SwayVelocityTemplate`]: Age-normed sway velocity norms
//! - [`StabilityLimitTemplate`]: Limits of stability norms

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// Sway area population template (95% ellipse area in cm²)
pub struct SwayAreaTemplate;

impl PopulationTemplate for SwayAreaTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Sway area increases with age (published norms)
        match context.age {
            Some(age) if age < 30.0 => 1.5, // Young adults: ~1.5 cm²
            Some(age) if age < 50.0 => 2.0, // Middle-aged: ~2.0 cm²
            Some(age) if age < 70.0 => 3.0, // Older adults: ~3.0 cm²
            Some(_) => 4.5,                 // Elderly: ~4.5 cm²
            None => 2.5,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        // Variance also increases with age
        match context.age {
            Some(age) if age < 30.0 => 0.5_f64.powi(2),
            Some(age) if age < 50.0 => 0.8_f64.powi(2),
            Some(age) if age < 70.0 => 1.2_f64.powi(2),
            Some(_) => 2.0_f64.powi(2),
            None => 1.0_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "SwayArea"
    }
}

/// Sway velocity population template (mean velocity in cm/s)
pub struct SwayVelocityTemplate;

impl PopulationTemplate for SwayVelocityTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Mean sway velocity norms by age
        match context.age {
            Some(age) if age < 30.0 => 0.8, // Young: ~0.8 cm/s
            Some(age) if age < 50.0 => 1.0, // Middle: ~1.0 cm/s
            Some(age) if age < 70.0 => 1.4, // Older: ~1.4 cm/s
            Some(_) => 2.0,                 // Elderly: ~2.0 cm/s
            None => 1.2,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 30.0 => 0.2_f64.powi(2),
            Some(age) if age < 50.0 => 0.3_f64.powi(2),
            Some(age) if age < 70.0 => 0.4_f64.powi(2),
            Some(_) => 0.6_f64.powi(2),
            None => 0.3_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "SwayVelocity"
    }
}

/// Limits of stability (max excursion) population template (% of theoretical limit)
pub struct StabilityLimitTemplate;

impl PopulationTemplate for StabilityLimitTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Max excursion decreases with age
        match context.age {
            Some(age) if age < 30.0 => 95.0, // Young: ~95%
            Some(age) if age < 50.0 => 90.0, // Middle: ~90%
            Some(age) if age < 70.0 => 80.0, // Older: ~80%
            Some(_) => 65.0,                 // Elderly: ~65%
            None => 85.0,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 30.0 => 5.0_f64.powi(2),
            Some(age) if age < 50.0 => 8.0_f64.powi(2),
            Some(age) if age < 70.0 => 12.0_f64.powi(2),
            Some(_) => 15.0_f64.powi(2),
            None => 10.0_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "StabilityLimit"
    }
}

// ============================================================================
// CoP Sway Encoder
// ============================================================================

/// Configuration for CoP sway encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopSwayConfig {
    /// Threshold for significant sway excursion (cm from center)
    pub excursion_threshold: f32,
    /// Minimum time between events (seconds)
    pub min_interval: f64,
    /// Window size for calculating mean position (samples)
    pub window_size: usize,
    /// Whether to encode AP and ML separately (true) or combined (false)
    pub separate_axes: bool,
}

impl Default for CopSwayConfig {
    fn default() -> Self {
        Self {
            excursion_threshold: 1.0, // 1 cm threshold
            min_interval: 0.1,        // 100ms minimum interval
            window_size: 50,          // ~0.5s at 100Hz
            separate_axes: true,
        }
    }
}

/// Encodes center of pressure sway excursions as spike events
///
/// Generates events when CoP position exceeds threshold distance from
/// the mean position. Useful for detecting postural instability.
///
/// # Input Signal
/// - Channel 0: CoP anterior-posterior (AP) position in cm
/// - Channel 1: CoP medial-lateral (ML) position in cm
///
/// # Output Events
/// - Polarity +1: Anterior or lateral excursion
/// - Polarity -1: Posterior or medial excursion
/// - Channel 0: AP events, Channel 1: ML events (if separate_axes)
pub struct CopSwayEncoder {
    template: SwayAreaTemplate,
}

impl CopSwayEncoder {
    /// Create a new CoP sway encoder
    pub fn new() -> Self {
        Self {
            template: SwayAreaTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &SwayAreaTemplate {
        &self.template
    }

    fn calculate_mean(data: &[f32], start: usize, window: usize) -> f32 {
        let end = (start + window).min(data.len());
        let slice = &data[start.saturating_sub(window / 2)..end];
        if slice.is_empty() {
            0.0
        } else {
            slice.iter().sum::<f32>() / slice.len() as f32
        }
    }
}

impl Default for CopSwayEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for CopSwayEncoder {
    type Config = CopSwayConfig;

    fn name(&self) -> &str {
        "CopSwayEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;

        // Process each axis
        let channels_to_process = if config.separate_axes {
            num_channels.min(2)
        } else {
            1
        };

        for ch in 0..channels_to_process {
            let mut last_event_idx = 0;

            // Get channel data
            let channel_data: Vec<f32> = if num_channels > 1 {
                samples
                    .iter()
                    .skip(ch)
                    .step_by(num_channels)
                    .copied()
                    .collect()
            } else {
                samples.to_vec()
            };

            for i in config.window_size..channel_data.len() {
                if (i - last_event_idx) < min_samples {
                    continue;
                }

                let mean_pos = Self::calculate_mean(&channel_data, i, config.window_size);
                let deviation = channel_data[i] - mean_pos;

                if deviation.abs() > config.excursion_threshold {
                    let time = i as f64 * dt;
                    let polarity = if deviation > 0.0 { 1 } else { -1 };
                    let magnitude = deviation.abs();

                    events.push(SpikeEvent::new(time, ch as u32, polarity, magnitude));
                    last_event_idx = i;
                }
            }
        }

        // Sort by timestamp (NaN-safe comparison)
        events.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(events)
    }
}

// ============================================================================
// CoP Velocity Encoder
// ============================================================================

/// Configuration for CoP velocity encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopVelocityConfig {
    /// Threshold for significant velocity (cm/s)
    pub velocity_threshold: f32,
    /// Minimum time between events (seconds)
    pub min_interval: f64,
    /// Smoothing window for velocity calculation (samples)
    pub smoothing_window: usize,
}

impl Default for CopVelocityConfig {
    fn default() -> Self {
        Self {
            velocity_threshold: 3.0, // 3 cm/s threshold
            min_interval: 0.05,      // 50ms minimum interval
            smoothing_window: 5,     // 5-sample smoothing
        }
    }
}

/// Encodes rapid CoP velocity changes as spike events
///
/// Generates events when CoP velocity exceeds threshold. Higher velocity
/// indicates rapid postural corrections or instability.
///
/// # Input Signal
/// - Channel 0: CoP AP position (cm)
/// - Channel 1: CoP ML position (cm)
///
/// # Output Events
/// - Magnitude encodes velocity magnitude
/// - Channel indicates direction (0=AP, 1=ML)
pub struct CopVelocityEncoder {
    template: SwayVelocityTemplate,
}

impl CopVelocityEncoder {
    /// Create a new CoP velocity encoder
    pub fn new() -> Self {
        Self {
            template: SwayVelocityTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &SwayVelocityTemplate {
        &self.template
    }

    fn calculate_velocity(data: &[f32], idx: usize, sample_rate: f64, window: usize) -> f32 {
        if idx < window || idx >= data.len() {
            return 0.0;
        }

        let dt = window as f64 / sample_rate;
        let velocity = (data[idx] - data[idx - window]) / dt as f32;
        velocity.abs()
    }
}

impl Default for CopVelocityEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for CopVelocityEncoder {
    type Config = CopVelocityConfig;

    fn name(&self) -> &str {
        "CopVelocityEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;

        for ch in 0..num_channels.min(2) {
            let mut last_event_idx = 0;

            let channel_data: Vec<f32> = if num_channels > 1 {
                samples
                    .iter()
                    .skip(ch)
                    .step_by(num_channels)
                    .copied()
                    .collect()
            } else {
                samples.to_vec()
            };

            for i in config.smoothing_window..channel_data.len() {
                if (i - last_event_idx) < min_samples {
                    continue;
                }

                let velocity = Self::calculate_velocity(
                    &channel_data,
                    i,
                    sample_rate,
                    config.smoothing_window,
                );

                if velocity > config.velocity_threshold {
                    let time = i as f64 * dt;
                    let direction = if channel_data[i] > channel_data[i - config.smoothing_window] {
                        1
                    } else {
                        -1
                    };

                    events.push(SpikeEvent::new(time, ch as u32, direction, velocity));
                    last_event_idx = i;
                }
            }
        }

        // Sort by timestamp (NaN-safe comparison)
        events.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(events)
    }
}

// ============================================================================
// Stability Limit Encoder
// ============================================================================

/// Configuration for stability limit encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityLimitConfig {
    /// Base of support dimensions (anterior, posterior, left, right) in cm
    pub bos_limits: (f32, f32, f32, f32),
    /// Threshold as fraction of limit (0.0-1.0)
    pub threshold_fraction: f32,
    /// Minimum time between events (seconds)
    pub min_interval: f64,
}

impl Default for StabilityLimitConfig {
    fn default() -> Self {
        Self {
            bos_limits: (12.0, 8.0, 8.0, 8.0), // Typical BoS: 12cm ant, 8cm post/left/right
            threshold_fraction: 0.8,           // Fire at 80% of limit
            min_interval: 0.2,                 // 200ms minimum interval
        }
    }
}

/// Encodes approaches to limits of stability
///
/// Generates events when CoP approaches the boundary of the base of support,
/// indicating risk of balance loss.
///
/// # Input Signal
/// - Channel 0: CoP AP position (cm, positive = anterior)
/// - Channel 1: CoP ML position (cm, positive = right)
///
/// # Output Events
/// - Channel encodes direction (0=ant, 1=post, 2=left, 3=right)
/// - Magnitude encodes proximity to limit (0-1)
pub struct StabilityLimitEncoder {
    template: StabilityLimitTemplate,
}

impl StabilityLimitEncoder {
    /// Create a new stability limit encoder
    pub fn new() -> Self {
        Self {
            template: StabilityLimitTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &StabilityLimitTemplate {
        &self.template
    }
}

impl Default for StabilityLimitEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for StabilityLimitEncoder {
    type Config = StabilityLimitConfig;

    fn name(&self) -> &str {
        "StabilityLimitEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_samples = (config.min_interval * sample_rate) as usize;
        let mut last_event_times = [0usize; 4]; // One per direction

        let samples_per_frame = num_channels.max(1);
        let num_frames = samples.len() / samples_per_frame;

        for i in 0..num_frames {
            let ap = samples[i * samples_per_frame];
            let ml = if num_channels > 1 {
                samples[i * samples_per_frame + 1]
            } else {
                0.0
            };

            let time = i as f64 * dt;

            // Check each direction
            let (ant_lim, post_lim, left_lim, right_lim) = config.bos_limits;
            let threshold = config.threshold_fraction;

            // Anterior
            if ap > ant_lim * threshold && (i - last_event_times[0]) >= min_samples {
                let proximity = (ap / ant_lim).min(1.0);
                events.push(SpikeEvent::new(time, 0, 1, proximity));
                last_event_times[0] = i;
            }

            // Posterior
            if ap < -post_lim * threshold && (i - last_event_times[1]) >= min_samples {
                let proximity = (-ap / post_lim).min(1.0);
                events.push(SpikeEvent::new(time, 1, -1, proximity));
                last_event_times[1] = i;
            }

            // Right (lateral)
            if ml > right_lim * threshold && (i - last_event_times[2]) >= min_samples {
                let proximity = (ml / right_lim).min(1.0);
                events.push(SpikeEvent::new(time, 2, 1, proximity));
                last_event_times[2] = i;
            }

            // Left (medial)
            if ml < -left_lim * threshold && (i - last_event_times[3]) >= min_samples {
                let proximity = (-ml / left_lim).min(1.0);
                events.push(SpikeEvent::new(time, 3, -1, proximity));
                last_event_times[3] = i;
            }
        }

        // Sort by timestamp (NaN-safe comparison)
        events.sort_by(|a, b| {
            a.timestamp
                .partial_cmp(&b.timestamp)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_sway_area_template() {
        let template = SwayAreaTemplate;
        let context = Context::default();
        let expected = template.expected_value(&context);
        assert!(expected > 0.0);
    }

    #[test]
    fn test_cop_sway_encoder() {
        // Create CoP signal with excursion
        let mut data = vec![0.0; 200];
        // Add excursion at sample 100
        #[allow(clippy::needless_range_loop)] // the index carries meaning beyond the lookup
        for i in 100..150 {
            data[i] = 2.0; // 2cm excursion
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = CopSwayEncoder::new();
        let config = CopSwayConfig {
            excursion_threshold: 1.0,
            min_interval: 0.05,
            window_size: 20,
            separate_axes: false,
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect excursion");
    }

    #[test]
    fn test_cop_velocity_encoder() {
        // Create CoP signal with rapid movement
        let mut data = vec![0.0; 200];
        // Rapid movement from sample 50-60
        #[allow(clippy::needless_range_loop)] // the index carries meaning beyond the lookup
        for i in 50..60 {
            data[i] = (i - 50) as f32 * 0.5; // 0.5 cm per sample = 50 cm/s at 100Hz
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = CopVelocityEncoder::new();
        let config = CopVelocityConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect high velocity");
    }

    #[test]
    fn test_stability_limit_encoder() {
        // Create CoP signal approaching anterior limit
        let mut data = vec![0.0; 200];
        // Approach anterior limit
        #[allow(clippy::needless_range_loop)] // the index carries meaning beyond the lookup
        for i in 50..100 {
            data[i] = 10.0; // 10cm anterior (80% of 12cm limit)
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = StabilityLimitEncoder::new();
        let config = StabilityLimitConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect approach to limit");
    }
}
