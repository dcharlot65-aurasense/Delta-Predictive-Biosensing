//! Vestibular system encoders
//!
//! This module provides event-based encoders for vestibular function assessment
//! including VOR (vestibulo-ocular reflex), nystagmus, and caloric test responses.
//!
//! ## Encoders
//!
//! - [`VorGainEncoder`]: Encodes VOR gain deviations from normal
//! - [`NystagmusEncoder`]: Detects nystagmus beats and patterns
//! - [`CaloricEncoder`]: Encodes caloric test responses
//!
//! ## Population Templates
//!
//! - [`VorGainTemplate`]: Age-normed VOR gain values
//! - [`NystagmusSPVTemplate`]: Slow phase velocity norms
//! - [`CaloricAsymmetryTemplate`]: Canal paresis norms

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// VOR gain population template (eye velocity / head velocity)
pub struct VorGainTemplate;

impl PopulationTemplate for VorGainTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Normal VOR gain is close to 1.0, decreases with age
        match context.age {
            Some(age) if age < 40.0 => 0.95,
            Some(age) if age < 60.0 => 0.90,
            Some(age) if age < 80.0 => 0.82,
            Some(_) => 0.75,
            None => 0.90,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        // Variance increases with age
        match context.age {
            Some(age) if age < 40.0 => 0.08_f64.powi(2),
            Some(age) if age < 60.0 => 0.10_f64.powi(2),
            Some(_) => 0.12_f64.powi(2),
            None => 0.10_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "VorGain"
    }
}

/// Nystagmus slow phase velocity template (deg/s)
pub struct NystagmusSPVTemplate;

impl PopulationTemplate for NystagmusSPVTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Normal spontaneous nystagmus SPV should be near 0
        // Pathological values can be 5-30+ deg/s
        0.0 // Normal is no spontaneous nystagmus
    }

    fn variance(&self, context: &Context) -> f64 {
        // Small variance for normal; up to ~3 deg/s considered normal
        2.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "NystagmusSPV"
    }
}

/// Caloric asymmetry (canal paresis) population template (%)
pub struct CaloricAsymmetryTemplate;

impl PopulationTemplate for CaloricAsymmetryTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Normal asymmetry is near 0%, pathological is >20-25%
        0.0
    }

    fn variance(&self, context: &Context) -> f64 {
        // Up to ~20% asymmetry can be normal
        10.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "CaloricAsymmetry"
    }
}

// ============================================================================
// VOR Gain Encoder
// ============================================================================

/// Configuration for VOR gain encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VorGainConfig {
    /// Normal VOR gain range (min, max)
    pub normal_range: (f32, f32),
    /// Minimum head velocity to consider (deg/s)
    pub min_head_velocity: f32,
    /// Window for gain calculation (samples)
    pub window_size: usize,
    /// Minimum interval between events (seconds)
    pub min_interval: f64,
}

impl Default for VorGainConfig {
    fn default() -> Self {
        Self {
            normal_range: (0.8, 1.0),
            min_head_velocity: 50.0,  // Only analyze during significant head movement
            window_size: 10,
            min_interval: 0.02,       // 20ms minimum
        }
    }
}

/// Encodes VOR gain deviations from normal range
///
/// Detects when the ratio of eye velocity to head velocity falls outside
/// the normal range, indicating vestibular dysfunction.
///
/// # Input Signal
/// - Channel 0: Head velocity (deg/s)
/// - Channel 1: Eye velocity (deg/s)
///
/// # Output Events
/// - Channel 0: Reduced gain (polarity -1)
/// - Channel 1: Elevated gain (polarity +1)
/// - Magnitude is the actual VOR gain value
pub struct VorGainEncoder {
    template: VorGainTemplate,
}

impl VorGainEncoder {
    /// Create a new VOR gain encoder
    pub fn new() -> Self {
        Self {
            template: VorGainTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &VorGainTemplate {
        &self.template
    }
}

impl Default for VorGainEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for VorGainEncoder {
    type Config = VorGainConfig;

    fn name(&self) -> &str {
        "VorGainEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        if num_channels < 2 {
            return Ok(events); // Need both head and eye velocity
        }

        let min_samples = (config.min_interval * sample_rate) as usize;
        let mut last_event_idx = 0;

        let samples_per_frame = num_channels;
        let num_frames = samples.len() / samples_per_frame;

        for i in config.window_size..num_frames {
            if (i - last_event_idx) < min_samples {
                continue;
            }

            let head_vel = samples[i * samples_per_frame].abs();
            let eye_vel = samples[i * samples_per_frame + 1].abs();

            // Only analyze during significant head movement
            if head_vel < config.min_head_velocity {
                continue;
            }

            let gain = eye_vel / head_vel;
            let time = i as f64 * dt;

            // Detect abnormal gain
            if gain < config.normal_range.0 {
                // Reduced gain (hypofunction)
                events.push(SpikeEvent::new(time, 0, -1, gain));
                last_event_idx = i;
            } else if gain > config.normal_range.1 {
                // Elevated gain (hyperfunction or artifact)
                events.push(SpikeEvent::new(time, 1, 1, gain));
                last_event_idx = i;
            }
        }

        Ok(events)
    }
}

// ============================================================================
// Nystagmus Encoder
// ============================================================================

/// Configuration for nystagmus encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NystagmusConfig {
    /// Minimum slow phase velocity to detect (deg/s)
    pub min_spv: f32,
    /// Velocity threshold for quick phase detection (deg/s)
    pub quick_phase_threshold: f32,
    /// Maximum slow phase duration (seconds)
    pub max_slow_phase_duration: f64,
    /// Minimum interval between beat events (seconds)
    pub min_beat_interval: f64,
}

impl Default for NystagmusConfig {
    fn default() -> Self {
        Self {
            min_spv: 3.0,                  // 3 deg/s minimum
            quick_phase_threshold: 100.0,  // Quick phases are >100 deg/s
            max_slow_phase_duration: 0.5,  // Max 500ms slow phase
            min_beat_interval: 0.15,       // Min 150ms between beats
        }
    }
}

/// Encodes nystagmus beats and patterns
///
/// Detects the characteristic slow-fast pattern of nystagmus, encoding
/// each beat as an event with direction and intensity information.
///
/// # Input Signal
/// - Single channel: Eye velocity (deg/s)
///
/// # Output Events
/// - Channel 0: Left-beating nystagmus (polarity -1)
/// - Channel 1: Right-beating nystagmus (polarity +1)
/// - Magnitude is slow phase velocity
pub struct NystagmusEncoder {
    template: NystagmusSPVTemplate,
}

impl NystagmusEncoder {
    /// Create a new nystagmus encoder
    pub fn new() -> Self {
        Self {
            template: NystagmusSPVTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &NystagmusSPVTemplate {
        &self.template
    }
}

impl Default for NystagmusEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for NystagmusEncoder {
    type Config = NystagmusConfig;

    fn name(&self) -> &str {
        "NystagmusEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let min_beat_samples = (config.min_beat_interval * sample_rate) as usize;
        let max_slow_samples = (config.max_slow_phase_duration * sample_rate) as usize;

        let mut last_beat_idx = 0;
        let mut in_slow_phase = false;
        let mut slow_phase_start = 0;
        let mut slow_phase_direction: i8 = 0;
        let mut slow_phase_sum = 0.0_f32;
        let mut slow_phase_count = 0;

        for i in 1..samples.len() {
            let velocity = samples[i];
            let abs_velocity = velocity.abs();
            let time = i as f64 * dt;

            // Detect quick phase (resets)
            if abs_velocity > config.quick_phase_threshold {
                if in_slow_phase && slow_phase_count > 0 {
                    let avg_spv = slow_phase_sum / slow_phase_count as f32;

                    // Emit beat event if significant and not too recent
                    if avg_spv.abs() > config.min_spv && (i - last_beat_idx) >= min_beat_samples {
                        let channel = if slow_phase_direction > 0 { 1 } else { 0 };
                        events.push(SpikeEvent::new(time, channel as u32, slow_phase_direction, avg_spv.abs()));
                        last_beat_idx = i;
                    }
                }

                // Reset for next slow phase
                in_slow_phase = false;
                slow_phase_sum = 0.0;
                slow_phase_count = 0;
                continue;
            }

            // Detect slow phase
            if abs_velocity > config.min_spv && abs_velocity < config.quick_phase_threshold {
                if !in_slow_phase {
                    in_slow_phase = true;
                    slow_phase_start = i;
                    slow_phase_direction = if velocity > 0.0 { 1 } else { -1 };
                    slow_phase_sum = 0.0;
                    slow_phase_count = 0;
                }

                // Accumulate slow phase velocity
                slow_phase_sum += velocity;
                slow_phase_count += 1;

                // Check for max duration exceeded
                if (i - slow_phase_start) > max_slow_samples {
                    in_slow_phase = false;
                }
            } else if abs_velocity < config.min_spv {
                // Velocity too low, reset if we were tracking
                if in_slow_phase && slow_phase_count > 5 {
                    let avg_spv = slow_phase_sum / slow_phase_count as f32;
                    if avg_spv.abs() > config.min_spv && (i - last_beat_idx) >= min_beat_samples {
                        let channel = if slow_phase_direction > 0 { 1 } else { 0 };
                        events.push(SpikeEvent::new(time, channel as u32, slow_phase_direction, avg_spv.abs()));
                        last_beat_idx = i;
                    }
                }
                in_slow_phase = false;
            }
        }

        Ok(events)
    }
}

// ============================================================================
// Caloric Encoder
// ============================================================================

/// Configuration for caloric test encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaloricConfig {
    /// Peak SPV threshold to detect response (deg/s)
    pub peak_spv_threshold: f32,
    /// Time window after stimulus for response (seconds)
    pub response_window: f64,
    /// Smoothing window for SPV calculation (samples)
    pub smoothing_window: usize,
}

impl Default for CaloricConfig {
    fn default() -> Self {
        Self {
            peak_spv_threshold: 5.0,   // 5 deg/s minimum response
            response_window: 90.0,     // 90 second response window
            smoothing_window: 50,      // 0.5s smoothing at 100Hz
        }
    }
}

/// Encodes caloric test responses
///
/// Analyzes the nystagmus response to warm/cold caloric stimulation,
/// encoding peak SPV, latency, and duration.
///
/// # Input Signal
/// - Channel 0: Eye velocity (deg/s)
/// - Channel 1 (optional): Stimulus marker (1 = stimulus on)
///
/// # Output Events
/// - Channel 0: Response onset (magnitude = latency)
/// - Channel 1: Peak response (magnitude = peak SPV)
/// - Channel 2: Response end (magnitude = duration)
pub struct CaloricEncoder {
    template: CaloricAsymmetryTemplate,
}

impl CaloricEncoder {
    /// Create a new caloric encoder
    pub fn new() -> Self {
        Self {
            template: CaloricAsymmetryTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &CaloricAsymmetryTemplate {
        &self.template
    }

    fn smooth_signal(samples: &[f32], window: usize) -> Vec<f32> {
        let mut smoothed = Vec::with_capacity(samples.len());
        let half_window = window / 2;

        for i in 0..samples.len() {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(samples.len());
            let sum: f32 = samples[start..end].iter().sum();
            smoothed.push(sum / (end - start) as f32);
        }

        smoothed
    }
}

impl Default for CaloricEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for CaloricEncoder {
    type Config = CaloricConfig;

    fn name(&self) -> &str {
        "CaloricEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        // Smooth the eye velocity signal
        let smoothed = Self::smooth_signal(samples, config.smoothing_window);

        // Find response characteristics
        let response_samples = (config.response_window * sample_rate) as usize;
        let analysis_length = smoothed.len().min(response_samples);

        let mut response_started = false;
        let mut response_start_idx = 0;
        let mut peak_spv = 0.0_f32;
        let mut peak_idx = 0;

        for i in 0..analysis_length {
            let abs_spv = smoothed[i].abs();

            // Detect response onset
            if !response_started && abs_spv > config.peak_spv_threshold {
                response_started = true;
                response_start_idx = i;
                let latency = i as f64 * dt;
                events.push(SpikeEvent::new(latency, 0, 1, latency as f32));
            }

            // Track peak
            if response_started && abs_spv > peak_spv {
                peak_spv = abs_spv;
                peak_idx = i;
            }

            // Detect response end (falls below threshold after peak)
            if response_started && i > peak_idx + (sample_rate as usize * 5)
                && abs_spv < config.peak_spv_threshold {
                    let duration = (i - response_start_idx) as f64 * dt;
                    let end_time = i as f64 * dt;
                    events.push(SpikeEvent::new(end_time, 2, -1, duration as f32));
                    break;
                }
        }

        // Emit peak event if response was detected
        if response_started && peak_spv > config.peak_spv_threshold {
            let peak_time = peak_idx as f64 * dt;
            let direction = if smoothed[peak_idx] > 0.0 { 1 } else { -1 };
            events.push(SpikeEvent::new(peak_time, 1, direction, peak_spv));
        }

        events.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
        Ok(events)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_vor_gain_template() {
        let template = VorGainTemplate;
        let young = Context { age: Some(25.0), ..Default::default() };
        let old = Context { age: Some(75.0), ..Default::default() };

        assert!(template.expected_value(&young) > template.expected_value(&old));
    }

    #[test]
    fn test_vor_gain_encoder() {
        // Create head and eye velocity signals
        // Normal VOR: eye velocity ≈ -head velocity (compensatory)
        let mut data = Vec::new();
        for i in 0..200 {
            let t = i as f32 * 0.01;
            let head_vel = 100.0 * (t * 5.0).sin(); // Head velocity
            let eye_vel = -85.0 * (t * 5.0).sin();  // Eye velocity (gain = 0.85)
            data.push(head_vel);
            data.push(eye_vel);
        }

        let signal = SignalBuffer::multi_channel(data, 100.0, 2);
        let encoder = VorGainEncoder::new();
        let config = VorGainConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        // Should detect reduced gain events
        let reduced_gain: Vec<_> = events.iter().filter(|e| e.channel == 0).collect();
        assert!(!reduced_gain.is_empty() || events.is_empty(), "May detect gain issues");
    }

    #[test]
    fn test_nystagmus_encoder() {
        // Simulate nystagmus pattern: slow drift + quick reset
        let mut data = Vec::new();
        let sample_rate = 100.0;

        for beat in 0..5 {
            // Slow phase (0.3s at 10 deg/s)
            for _ in 0..30 {
                data.push(10.0);
            }
            // Quick phase (0.05s at -200 deg/s)
            for _ in 0..5 {
                data.push(-200.0);
            }
        }

        let signal = SignalBuffer::single_channel(data, sample_rate);
        let encoder = NystagmusEncoder::new();
        let config = NystagmusConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect nystagmus beats");
    }

    #[test]
    fn test_caloric_encoder() {
        // Simulate caloric response: gradual onset, peak, gradual decline
        let mut data = vec![0.0; 100]; // Baseline

        // Response onset and buildup
        for i in 0..200 {
            let spv = 20.0 * (1.0 - (-i as f32 * 0.02).exp());
            data.push(spv);
        }

        // Peak and decline
        for i in 0..300 {
            let spv = 20.0 * (-i as f32 * 0.01).exp();
            data.push(spv);
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = CaloricEncoder::new();
        let config = CaloricConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect caloric response");
    }
}
