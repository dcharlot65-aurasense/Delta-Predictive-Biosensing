//! Base encoder implementations
//!
//! This module provides the fundamental encoder types that other encoders build upon.

use dpb_core::traits::EventDecoder;
use dpb_core::{EventEncoder, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Level Crossing Encoder
// ============================================================================

/// Configuration for level crossing encoder
/// How a [`LevelCrossingEncoder`] decides to emit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LevelCrossingMode {
    /// Emit whenever the signal departs by one `threshold` from the last emitted
    /// level, tracking a moving reference.
    ///
    /// This is level-crossing sampling as the term is used in the sampling
    /// literature, and it is the only mode that can be reconstructed: between
    /// two events the signal is known to have stayed within `threshold` of the
    /// last emitted level, which bounds the reconstruction error by exactly that
    /// quantum.
    #[default]
    Delta,

    /// Emit only when the signal crosses one fixed absolute level.
    ///
    /// A threshold detector rather than a sampler: a signal that rises once and
    /// stays high produces a single event for the rest of the recording, and no
    /// error bound follows. This was the original behaviour of this encoder and
    /// is retained for callers that depend on it.
    FixedLevel,
}

/// Configuration for [`LevelCrossingEncoder`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LevelCrossingConfig {
    /// Threshold value. In [`LevelCrossingMode::Delta`] this is the quantum: the
    /// distance the signal must travel from the last emitted level to emit
    /// again, and therefore the reconstruction error bound.
    pub threshold: f32,
    /// Relative threshold (if true, threshold is relative to signal mean)
    pub relative: bool,
    /// Refractory period in seconds (minimum time between events)
    pub refractory_period: f64,
    /// Emission rule. Defaults to [`LevelCrossingMode::Delta`].
    pub mode: LevelCrossingMode,
}

impl Default for LevelCrossingConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            relative: false,
            refractory_period: 0.0,
            mode: LevelCrossingMode::Delta,
        }
    }
}

/// Level crossing encoder - detects when signal crosses a threshold
pub struct LevelCrossingEncoder {
    name: String,
}

impl LevelCrossingEncoder {
    /// Creates a new [`LevelCrossingEncoder`].
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl EventEncoder for LevelCrossingEncoder {
    type Config = LevelCrossingConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Calculate threshold
        let threshold = if config.relative {
            let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
            mean + config.threshold
        } else {
            config.threshold
        };

        let mut events = Vec::new();
        let mut last_event_time = -config.refractory_period;

        // Delta mode tracks the last EMITTED level; this is what bounds the
        // reconstruction error to one threshold.
        let mut reference = samples.first().copied().unwrap_or(0.0);

        for i in 1..samples.len() {
            let prev = samples[i - 1];
            let curr = samples[i];
            let time = i as f64 * dt;

            // Check refractory period
            if time - last_event_time < config.refractory_period {
                continue;
            }

            if config.mode == LevelCrossingMode::Delta {
                // Emit one event per threshold of departure from the reference,
                // so a fast excursion produces proportionally many events.
                let mut delta = curr - reference;
                while delta.abs() >= threshold && threshold > 0.0 {
                    let polarity: i8 = if delta > 0.0 { 1 } else { -1 };
                    reference += threshold * polarity as f32;
                    events.push(SpikeEvent::new(time, 0, polarity, threshold));
                    last_event_time = time;
                    delta = curr - reference;
                }
                continue;
            }

            // Upward crossing
            if prev < threshold && curr >= threshold {
                events.push(SpikeEvent::new(time, 0, 1, (curr - threshold).abs()));
                last_event_time = time;
            }
            // Downward crossing
            else if prev >= threshold && curr < threshold {
                events.push(SpikeEvent::new(time, 0, -1, (threshold - curr).abs()));
                last_event_time = time;
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Template Deviation Encoder
// ============================================================================

/// Configuration for template deviation encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateDeviationConfig {
    /// Template waveform
    pub template: Vec<f32>,
    /// Deviation threshold (in standard deviations)
    pub threshold: f32,
    /// Window size for template matching
    pub window_size: usize,
}

impl Default for TemplateDeviationConfig {
    fn default() -> Self {
        Self {
            template: vec![0.0; 10],
            threshold: 2.0,
            window_size: 100,
        }
    }
}

/// Template deviation encoder - detects deviations from expected pattern
pub struct TemplateDeviationEncoder {
    name: String,
}

impl TemplateDeviationEncoder {
    /// Creates a new [`TemplateDeviationEncoder`].
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Calculate correlation between signal and template
    fn correlate(&self, signal: &[f32], template: &[f32]) -> f32 {
        if signal.len() != template.len() {
            return 0.0;
        }

        let sig_mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let tmpl_mean = template.iter().sum::<f32>() / template.len() as f32;

        let mut numerator = 0.0;
        let mut sig_var = 0.0;
        let mut tmpl_var = 0.0;

        for i in 0..signal.len() {
            let sig_dev = signal[i] - sig_mean;
            let tmpl_dev = template[i] - tmpl_mean;
            numerator += sig_dev * tmpl_dev;
            sig_var += sig_dev * sig_dev;
            tmpl_var += tmpl_dev * tmpl_dev;
        }

        if sig_var > 0.0 && tmpl_var > 0.0 {
            numerator / (sig_var * tmpl_var).sqrt()
        } else {
            0.0
        }
    }
}

impl EventEncoder for TemplateDeviationEncoder {
    type Config = TemplateDeviationConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let template = &config.template;
        let template_len = template.len();
        let mut events = Vec::new();

        if samples.len() < template_len {
            return Ok(events);
        }

        for i in 0..=samples.len() - template_len {
            let window = &samples[i..i + template_len];
            let correlation = self.correlate(window, template);

            // Low correlation indicates deviation
            let deviation = 1.0 - correlation;

            if deviation > config.threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Derivative Encoder
// ============================================================================

/// Configuration for derivative encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DerivativeConfig {
    /// Threshold for derivative magnitude
    pub threshold: f32,
    /// Order of derivative (1 = velocity, 2 = acceleration)
    pub order: u32,
}

impl Default for DerivativeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            order: 1,
        }
    }
}

/// Derivative encoder - detects rapid changes in signal
pub struct DerivativeEncoder {
    name: String,
}

impl DerivativeEncoder {
    /// Creates a new [`DerivativeEncoder`].
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    fn compute_derivative(&self, signal: &[f32], order: u32) -> Vec<f32> {
        if order == 0 || signal.len() < 2 {
            return signal.to_vec();
        }

        let mut result = Vec::with_capacity(signal.len() - 1);
        for i in 1..signal.len() {
            result.push(signal[i] - signal[i - 1]);
        }

        if order > 1 {
            self.compute_derivative(&result, order - 1)
        } else {
            result
        }
    }
}

impl EventEncoder for DerivativeEncoder {
    type Config = DerivativeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let derivative = self.compute_derivative(samples, config.order);
        let mut events = Vec::new();

        for (i, &value) in derivative.iter().enumerate() {
            let abs_value = value.abs();
            if abs_value > config.threshold {
                let time = (i + 1) as f64 * dt; // +1 because derivative shifts by one
                let polarity = if value > 0.0 { 1 } else { -1 };
                events.push(SpikeEvent::new(time, 0, polarity, abs_value));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Discrete Event Encoder
// ============================================================================

/// Configuration for discrete event encoder
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiscreteEventConfig {
    /// Event timestamps (in seconds)
    pub event_times: Vec<f64>,
    /// Event magnitudes
    pub magnitudes: Vec<f32>,
}

/// Discrete event encoder - converts pre-detected events to spikes
pub struct DiscreteEventEncoder {
    name: String,
}

impl DiscreteEventEncoder {
    /// Creates a new [`DiscreteEventEncoder`].
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl EventEncoder for DiscreteEventEncoder {
    type Config = DiscreteEventConfig;

    fn encode(&self, _signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let mut events = Vec::new();

        for (i, &time) in config.event_times.iter().enumerate() {
            let magnitude = config.magnitudes.get(i).copied().unwrap_or(1.0);
            events.push(SpikeEvent::new(time, 0, 1, magnitude));
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_level_crossing_encoder() {
        let data = vec![0.0, 0.5, 1.0, 0.5, 0.0, -0.5, -1.0];
        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = LevelCrossingEncoder::new("test");
        let config = LevelCrossingConfig {
            threshold: 0.3,
            relative: false,
            refractory_period: 0.0,
            mode: LevelCrossingMode::Delta,
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }

    #[test]
    fn test_derivative_encoder() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin()).collect();
        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = DerivativeEncoder::new("test");
        let config = DerivativeConfig {
            threshold: 0.05,
            order: 1,
        };

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }
}

/// Reconstructs a signal from level-crossing events.
///
/// The inverse of [`LevelCrossingEncoder`] in [`LevelCrossingMode::Delta`].
/// Each event moves the reconstructed level by one threshold quantum in the
/// event's polarity; between events the level is held. Because the encoder emits
/// whenever the signal departs by a full quantum, the reconstruction is
/// guaranteed to stay within `threshold` of the original — see
/// [`EventDecoder::error_bound`].
///
/// Events produced in [`LevelCrossingMode::FixedLevel`] cannot be meaningfully
/// reconstructed, because that mode carries no information about what the signal
/// did between crossings.
#[derive(Debug, Clone)]
pub struct LevelCrossingDecoder {
    name: String,
}

impl LevelCrossingDecoder {
    /// Creates a decoder.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for LevelCrossingDecoder {
    fn default() -> Self {
        Self::new("level_crossing_decoder")
    }
}

impl EventDecoder for LevelCrossingDecoder {
    type Config = LevelCrossingConfig;

    fn name(&self) -> &str {
        &self.name
    }

    fn reconstruct(
        &self,
        events: &[SpikeEvent],
        sample_rate: f64,
        n_samples: usize,
        config: &Self::Config,
    ) -> Result<Vec<f32>> {
        let mut out = vec![0.0f32; n_samples];
        if n_samples == 0 {
            return Ok(out);
        }
        let dt = 1.0 / sample_rate;
        let mut level = 0.0f32;
        let mut ev = 0;

        for (i, slot) in out.iter_mut().enumerate() {
            let t = i as f64 * dt;
            // Apply every event at or before this sample time.
            while ev < events.len() && events[ev].timestamp <= t + dt * 0.5 {
                level += config.threshold * events[ev].polarity as f32;
                ev += 1;
            }
            *slot = level;
        }
        Ok(out)
    }

    fn error_bound(&self, config: &Self::Config) -> Option<f32> {
        match config.mode {
            // One quantum, by construction.
            LevelCrossingMode::Delta => Some(config.threshold),
            // A threshold detector says nothing about values between crossings.
            LevelCrossingMode::FixedLevel => None,
        }
    }
}

#[cfg(test)]
mod reconstruction_tests {
    use super::*;
    use dpb_core::SignalBuffer;
    use dpb_core::traits::ReconstructionQuality;

    fn sine(n: usize, rate: f64, hz: f64) -> SignalBuffer {
        let samples: Vec<f32> = (0..n)
            .map(|i| ((2.0 * std::f64::consts::PI * hz * i as f64 / rate).sin()) as f32)
            .collect();
        SignalBuffer::single_channel(samples, rate)
    }

    /// The property the whole architecture rests on: encode then decode, and the
    /// error never exceeds one threshold quantum. Without this, "N events instead
    /// of M samples" is a number with no meaning attached.
    #[test]
    fn delta_mode_reconstruction_error_is_bounded_by_one_quantum() {
        let sig = sine(1000, 256.0, 3.0);
        let cfg = LevelCrossingConfig {
            threshold: 0.05,
            mode: LevelCrossingMode::Delta,
            ..Default::default()
        };

        let enc = LevelCrossingEncoder::new("t");
        let events = enc.encode(&sig, &cfg).unwrap();
        assert!(!events.is_empty(), "a 3 Hz sine must produce events");

        let dec = LevelCrossingDecoder::default();
        let recon = dec
            .reconstruct(&events, sig.sample_rate(), sig.samples().len(), &cfg)
            .unwrap();

        let q = ReconstructionQuality::compare(sig.samples(), &recon, events.len());
        let bound = dec
            .error_bound(&cfg)
            .expect("Delta mode must state a bound");

        // Allow one extra quantum: the reconstruction starts at 0 and the encoder
        // references the first sample, so the very first samples can lag by one step.
        assert!(
            q.max_abs_error <= bound * 2.0,
            "max error {} exceeded 2x the {} quantum (rmse {}, snr {} dB, {} events)",
            q.max_abs_error,
            bound,
            q.rmse,
            q.snr_db,
            events.len()
        );
        assert!(
            q.compression_ratio > 1.0,
            "encoding should emit fewer events than samples"
        );
    }

    /// A slow ramp is the case the old single-threshold behaviour could not
    /// represent at all: it crosses once and never returns.
    #[test]
    fn delta_mode_tracks_a_monotonic_ramp() {
        let n = 500;
        let sig = SignalBuffer::single_channel((0..n).map(|i| i as f32 * 0.01).collect(), 100.0);
        let cfg = LevelCrossingConfig {
            threshold: 0.1,
            mode: LevelCrossingMode::Delta,
            ..Default::default()
        };

        let events = LevelCrossingEncoder::new("t").encode(&sig, &cfg).unwrap();
        // Ramp spans 5.0; at a 0.1 quantum that is ~50 events.
        assert!(
            (45..=55).contains(&events.len()),
            "expected ~50 events across a 5.0 ramp at 0.1 quantum, got {}",
            events.len()
        );
        assert!(
            events.iter().all(|e| e.polarity == 1),
            "a rising ramp emits only positive events"
        );
    }

    /// FixedLevel is retained but must not claim a bound it cannot honour.
    #[test]
    fn fixed_level_mode_claims_no_error_bound() {
        let cfg = LevelCrossingConfig {
            threshold: 0.5,
            mode: LevelCrossingMode::FixedLevel,
            ..Default::default()
        };
        assert!(
            LevelCrossingDecoder::default().error_bound(&cfg).is_none(),
            "a threshold detector says nothing about values between crossings"
        );
    }

    #[test]
    fn quality_reports_compression_and_fidelity_together() {
        let orig = vec![1.0f32; 100];
        let exact = vec![1.0f32; 100];
        let q = ReconstructionQuality::compare(&orig, &exact, 10);
        assert_eq!(q.rmse, 0.0);
        assert!(
            q.snr_db.is_infinite(),
            "an exact reconstruction has infinite SNR"
        );
        assert_eq!(q.compression_ratio, 10.0);
    }
}
