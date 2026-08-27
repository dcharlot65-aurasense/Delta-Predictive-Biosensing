//! Cardiopulmonary encoders
//!
//! This module provides event-based encoders for heart rate variability (HRV)
//! and respiratory signals.
//!
//! ## Encoders
//!
//! - [`HrvEncoder`]: Encodes HRV metrics and anomalies
//! - [`RespiratoryPhaseEncoder`]: Encodes respiratory phases and rate changes
//! - [`RsaEncoder`]: Encodes respiratory sinus arrhythmia patterns
//!
//! ## Population Templates
//!
//! - [`RmssdTemplate`]: RMSSD norms by age
//! - [`RespiratoryRateTemplate`]: Breathing rate norms
//! - [`RsaTemplate`]: RSA amplitude norms

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates
// ============================================================================

/// RMSSD (root mean square of successive differences) population template (ms)
pub struct RmssdTemplate;

impl PopulationTemplate for RmssdTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // RMSSD decreases with age
        match context.age {
            Some(age) if age < 30.0 => 42.0,
            Some(age) if age < 40.0 => 35.0,
            Some(age) if age < 50.0 => 30.0,
            Some(age) if age < 60.0 => 25.0,
            Some(age) if age < 70.0 => 22.0,
            Some(_) => 18.0,
            None => 30.0,
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        // High individual variability
        match context.age {
            Some(age) if age < 40.0 => 15.0_f64.powi(2),
            Some(age) if age < 60.0 => 12.0_f64.powi(2),
            Some(_) => 10.0_f64.powi(2),
            None => 12.0_f64.powi(2),
        }
    }

    fn name(&self) -> &str {
        "RMSSD"
    }
}

/// SDNN (standard deviation of NN intervals) population template (ms)
pub struct SdnnTemplate;

impl PopulationTemplate for SdnnTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        match context.age {
            Some(age) if age < 30.0 => 150.0,
            Some(age) if age < 50.0 => 130.0,
            Some(age) if age < 70.0 => 110.0,
            Some(_) => 90.0,
            None => 120.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        40.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "SDNN"
    }
}

/// Respiratory rate population template (breaths/min)
pub struct RespiratoryRateTemplate;

impl PopulationTemplate for RespiratoryRateTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Normal adult respiratory rate is 12-20 breaths/min
        match context.age {
            Some(age) if age < 1.0 => 40.0,   // Infants
            Some(age) if age < 5.0 => 28.0,   // Toddlers
            Some(age) if age < 12.0 => 22.0,  // Children
            Some(_) => 15.0,                   // Adults
            None => 15.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        3.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "RespiratoryRate"
    }
}

/// RSA (respiratory sinus arrhythmia) amplitude template (ms)
pub struct RsaTemplate;

impl PopulationTemplate for RsaTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // RSA amplitude decreases with age
        match context.age {
            Some(age) if age < 30.0 => 80.0,
            Some(age) if age < 50.0 => 60.0,
            Some(age) if age < 70.0 => 40.0,
            Some(_) => 25.0,
            None => 50.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        25.0_f64.powi(2)
    }

    fn name(&self) -> &str {
        "RsaAmplitude"
    }
}

// ============================================================================
// HRV Encoder
// ============================================================================

/// Configuration for HRV encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HrvConfig {
    /// Window size for HRV calculation (number of beats)
    pub window_size: usize,
    /// RMSSD threshold for low HRV event (ms)
    pub low_rmssd_threshold: f32,
    /// RMSSD threshold for high HRV event (ms)
    pub high_rmssd_threshold: f32,
    /// Minimum interval between HRV events (seconds)
    pub min_interval: f64,
    /// Whether input is RR intervals (true) or R-peak times (false)
    pub input_is_intervals: bool,
}

impl Default for HrvConfig {
    fn default() -> Self {
        Self {
            window_size: 30,             // 30 beats (~30s at 60 bpm)
            low_rmssd_threshold: 15.0,   // Below 15ms is concerning
            high_rmssd_threshold: 80.0,  // Above 80ms is unusual
            min_interval: 5.0,           // 5 second minimum between events
            input_is_intervals: true,
        }
    }
}

/// Encodes HRV metrics and anomalies
///
/// Calculates running HRV metrics (RMSSD, SDNN) and generates events
/// when values fall outside normal ranges.
///
/// # Input Signal
/// - Single channel: RR intervals in ms (or R-peak times in seconds)
///
/// # Output Events
/// - Channel 0: Low HRV event (polarity -1)
/// - Channel 1: High HRV event (polarity +1)
/// - Channel 2: RMSSD value event (magnitude = RMSSD)
/// - Channel 3: SDNN value event (magnitude = SDNN)
pub struct HrvEncoder {
    template: RmssdTemplate,
}

impl HrvEncoder {
    /// Create a new HRV encoder
    pub fn new() -> Self {
        Self {
            template: RmssdTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &RmssdTemplate {
        &self.template
    }

    fn calculate_rmssd(intervals: &[f32]) -> f32 {
        if intervals.len() < 2 {
            return 0.0;
        }

        let mut sum_sq_diff = 0.0_f32;
        for i in 1..intervals.len() {
            let diff = intervals[i] - intervals[i - 1];
            sum_sq_diff += diff * diff;
        }

        (sum_sq_diff / (intervals.len() - 1) as f32).sqrt()
    }

    fn calculate_sdnn(intervals: &[f32]) -> f32 {
        if intervals.is_empty() {
            return 0.0;
        }

        let mean = intervals.iter().sum::<f32>() / intervals.len() as f32;
        let variance = intervals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / intervals.len() as f32;
        variance.sqrt()
    }
}

impl Default for HrvEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for HrvEncoder {
    type Config = HrvConfig;

    fn name(&self) -> &str {
        "HrvEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let _sample_rate = signal.sample_rate();

        let mut events = Vec::new();

        // Convert to RR intervals if input is R-peak times
        let intervals: Vec<f32> = if config.input_is_intervals {
            samples.to_vec()
        } else {
            // Input is R-peak times, convert to intervals
            let mut ivls = Vec::new();
            for i in 1..samples.len() {
                ivls.push((samples[i] - samples[i - 1]) * 1000.0); // Convert to ms
            }
            ivls
        };

        if intervals.len() < config.window_size {
            return Ok(events);
        }

        let min_beats = (config.min_interval * 1000.0 / 800.0) as usize; // Assuming ~75 bpm
        let mut last_event_beat = 0;

        // Sliding window HRV calculation
        for i in config.window_size..intervals.len() {
            if (i - last_event_beat) < min_beats {
                continue;
            }

            let window = &intervals[i - config.window_size..i];
            let rmssd = Self::calculate_rmssd(window);
            let sdnn = Self::calculate_sdnn(window);

            // Estimate time (sum of intervals up to this point)
            let time: f64 = intervals[0..i].iter().map(|x| *x as f64).sum::<f64>() / 1000.0;

            // Emit HRV metric events
            events.push(SpikeEvent::new(time, 2, 1, rmssd));
            events.push(SpikeEvent::new(time, 3, 1, sdnn));

            // Check for abnormal HRV
            if rmssd < config.low_rmssd_threshold {
                events.push(SpikeEvent::new(time, 0, -1, rmssd));
                last_event_beat = i;
            } else if rmssd > config.high_rmssd_threshold {
                events.push(SpikeEvent::new(time, 1, 1, rmssd));
                last_event_beat = i;
            }
        }

        Ok(events)
    }
}

// ============================================================================
// Respiratory Phase Encoder
// ============================================================================

/// Configuration for respiratory phase encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespiratoryPhaseConfig {
    /// Threshold for breath detection (fraction of signal range)
    pub breath_threshold: f32,
    /// Minimum breath duration (seconds)
    pub min_breath_duration: f64,
    /// Maximum breath duration (seconds)
    pub max_breath_duration: f64,
    /// Rate change threshold to emit event (breaths/min)
    pub rate_change_threshold: f32,
}

impl Default for RespiratoryPhaseConfig {
    fn default() -> Self {
        Self {
            breath_threshold: 0.3,        // 30% of range
            min_breath_duration: 1.5,     // Min 1.5s (40 breaths/min)
            max_breath_duration: 10.0,    // Max 10s (6 breaths/min)
            rate_change_threshold: 3.0,   // 3 breaths/min change
        }
    }
}

/// Encodes respiratory phases and rate changes
///
/// Detects inspiration and expiration phases, and generates events
/// for significant respiratory rate changes.
///
/// # Input Signal
/// - Single channel: Respiratory signal (chest expansion, flow, etc.)
///
/// # Output Events
/// - Channel 0: Inspiration onset (polarity +1)
/// - Channel 1: Expiration onset (polarity -1)
/// - Channel 2: Rate increase (polarity +1, magnitude = new rate)
/// - Channel 3: Rate decrease (polarity -1, magnitude = new rate)
pub struct RespiratoryPhaseEncoder {
    template: RespiratoryRateTemplate,
}

impl RespiratoryPhaseEncoder {
    /// Create a new respiratory phase encoder
    pub fn new() -> Self {
        Self {
            template: RespiratoryRateTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &RespiratoryRateTemplate {
        &self.template
    }
}

impl Default for RespiratoryPhaseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for RespiratoryPhaseEncoder {
    type Config = RespiratoryPhaseConfig;

    fn name(&self) -> &str {
        "RespiratoryPhaseEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        // Find signal range
        let min_val = samples.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = samples.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range < 0.001 {
            return Ok(events); // Signal too flat
        }

        let threshold_high = min_val + range * (1.0 - config.breath_threshold);
        let threshold_low = min_val + range * config.breath_threshold;

        let min_samples = (config.min_breath_duration * sample_rate) as usize;
        let max_samples = (config.max_breath_duration * sample_rate) as usize;

        let mut in_inspiration = samples[0] < (min_val + max_val) / 2.0;
        let mut phase_start = 0;
        let mut breath_times = Vec::new();
        let mut last_rate = 0.0_f32;

        for i in 1..samples.len() {
            let val = samples[i];
            let time = i as f64 * dt;
            let phase_duration = i - phase_start;

            // Detect inspiration onset (crossing low threshold upward)
            if !in_inspiration && val > threshold_high && samples[i - 1] <= threshold_high {
                if phase_duration >= min_samples && phase_duration <= max_samples {
                    events.push(SpikeEvent::new(time, 0, 1, val / max_val));
                    breath_times.push(time);

                    // Calculate and check respiratory rate
                    if breath_times.len() >= 3 {
                        let recent_breaths = &breath_times[breath_times.len() - 3..];
                        let breath_interval = (recent_breaths[2] - recent_breaths[0]) / 2.0;
                        let current_rate = (60.0 / breath_interval) as f32;

                        if last_rate > 0.0 {
                            let rate_change = current_rate - last_rate;
                            if rate_change > config.rate_change_threshold {
                                events.push(SpikeEvent::new(time, 2, 1, current_rate));
                            } else if rate_change < -config.rate_change_threshold {
                                events.push(SpikeEvent::new(time, 3, -1, current_rate));
                            }
                        }
                        last_rate = current_rate;
                    }
                }
                in_inspiration = true;
                phase_start = i;
            }

            // Detect expiration onset (crossing high threshold downward)
            if in_inspiration && val < threshold_low && samples[i - 1] >= threshold_low {
                if phase_duration >= min_samples && phase_duration <= max_samples {
                    events.push(SpikeEvent::new(time, 1, -1, val / max_val));
                }
                in_inspiration = false;
                phase_start = i;
            }
        }

        Ok(events)
    }
}

// ============================================================================
// RSA (Respiratory Sinus Arrhythmia) Encoder
// ============================================================================

/// Configuration for RSA encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RsaConfig {
    /// Minimum RSA amplitude to detect (ms)
    pub min_amplitude: f32,
    /// Window size for RSA calculation (breaths)
    pub window_breaths: usize,
    /// Low RSA threshold (ms)
    pub low_rsa_threshold: f32,
}

impl Default for RsaConfig {
    fn default() -> Self {
        Self {
            min_amplitude: 10.0,      // 10ms minimum
            window_breaths: 5,        // 5 breath window
            low_rsa_threshold: 20.0,  // Below 20ms is reduced
        }
    }
}

/// Encodes respiratory sinus arrhythmia patterns
///
/// Detects the heart rate variation coupled to breathing, a marker
/// of parasympathetic (vagal) function.
///
/// # Input Signal
/// - Channel 0: RR intervals (ms)
/// - Channel 1: Respiratory signal (for phase alignment)
///
/// # Output Events
/// - Channel 0: RSA amplitude per breath cycle (magnitude = amplitude in ms)
/// - Channel 1: Low RSA event (magnitude = amplitude)
pub struct RsaEncoder {
    template: RsaTemplate,
}

impl RsaEncoder {
    /// Create a new RSA encoder
    pub fn new() -> Self {
        Self {
            template: RsaTemplate,
        }
    }

    /// Get the associated population template
    pub fn template(&self) -> &RsaTemplate {
        &self.template
    }
}

impl Default for RsaEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for RsaEncoder {
    type Config = RsaConfig;

    fn name(&self) -> &str {
        "RsaEncoder"
    }

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let num_channels = signal.channels();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();

        if num_channels < 2 || samples.len() < 100 {
            return Ok(events);
        }

        let samples_per_frame = num_channels;
        let num_frames = samples.len() / samples_per_frame;

        // Estimate breath cycle length (~4 seconds at 15 breaths/min)
        let breath_samples = (4.0 * sample_rate) as usize;

        // Calculate RSA for each breath cycle
        for breath in 0..(num_frames / breath_samples) {
            let start = breath * breath_samples;
            let end = (start + breath_samples).min(num_frames);

            // Find max and min RR in this breath cycle
            let mut max_rr = f32::MIN;
            let mut min_rr = f32::MAX;

            for i in start..end {
                let rr = samples[i * samples_per_frame];
                if rr > max_rr {
                    max_rr = rr;
                }
                if rr < min_rr {
                    min_rr = rr;
                }
            }

            let rsa_amplitude = max_rr - min_rr;
            let time = (start + breath_samples / 2) as f64 * dt;

            if rsa_amplitude > config.min_amplitude {
                events.push(SpikeEvent::new(time, 0, 1, rsa_amplitude));

                // Check for low RSA
                if rsa_amplitude < config.low_rsa_threshold {
                    events.push(SpikeEvent::new(time, 1, -1, rsa_amplitude));
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
    fn test_rmssd_template() {
        let template = RmssdTemplate;
        let young = Context { age: Some(25.0), ..Default::default() };
        let old = Context { age: Some(65.0), ..Default::default() };

        assert!(template.expected_value(&young) > template.expected_value(&old));
    }

    #[test]
    fn test_hrv_encoder() {
        // Simulate RR intervals with some variability
        let mut data = Vec::new();
        for i in 0..100 {
            // ~800ms RR with ±30ms variation
            let rr = 800.0 + 30.0 * (i as f32 * 0.5).sin();
            data.push(rr);
        }

        let signal = SignalBuffer::single_channel(data, 1.0); // 1 sample per beat
        let encoder = HrvEncoder::new();
        let config = HrvConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should generate HRV events");
    }

    #[test]
    fn test_respiratory_phase_encoder() {
        // Simulate sinusoidal respiratory signal
        let mut data = Vec::new();
        let sample_rate = 100.0_f64;

        for i in 0..1000 {
            let t = i as f32 / sample_rate as f32;
            // 15 breaths/min = 0.25 Hz
            let resp = (t * 0.25 * 2.0 * std::f32::consts::PI).sin();
            data.push(resp);
        }

        let signal = SignalBuffer::single_channel(data, sample_rate);
        let encoder = RespiratoryPhaseEncoder::new();
        let config = RespiratoryPhaseConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect respiratory phases");
    }

    #[test]
    fn test_rsa_encoder() {
        // Simulate RR intervals with respiratory modulation
        let mut data = Vec::new();
        let sample_rate = 10.0_f64; // 10 Hz for RR data

        for i in 0..500 {
            let t = i as f32 / sample_rate as f32;
            // RR varies with breathing (~15 breaths/min)
            let rr = 800.0 + 50.0 * (t * 0.25 * 2.0 * std::f32::consts::PI).sin();
            // Respiratory signal
            let resp = (t * 0.25 * 2.0 * std::f32::consts::PI).sin();
            data.push(rr);
            data.push(resp);
        }

        let signal = SignalBuffer::multi_channel(data, sample_rate, 2);
        let encoder = RsaEncoder::new();
        let config = RsaConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty(), "Should detect RSA");
    }
}
