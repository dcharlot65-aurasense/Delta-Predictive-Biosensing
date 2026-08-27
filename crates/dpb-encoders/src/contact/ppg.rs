//! PPG (Photoplethysmography) encoders and population templates

use dpb_core::{Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates for PPG
// ============================================================================

/// Pulse rate population template
pub struct PulseRateTemplate;

impl PopulationTemplate for PulseRateTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Same as heart rate
        match context.age {
            Some(age) if age < 1.0 => 140.0,
            Some(age) if age < 3.0 => 120.0,
            Some(age) if age < 12.0 => 100.0,
            Some(age) if age < 18.0 => 85.0,
            Some(age) if age < 65.0 => 72.0,
            Some(_) => 75.0,
            None => 72.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        10.0
    }

    fn name(&self) -> &str {
        "PulseRateTemplate"
    }
}

/// PPG amplitude template (perfusion index)
pub struct PpgAmplitudeNorms;

impl PopulationTemplate for PpgAmplitudeNorms {
    fn expected_value(&self, _context: &Context) -> f64 {
        // Perfusion index (PI) percentage
        2.0 // Normal: 1-5%
    }

    fn variance(&self, _context: &Context) -> f64 {
        1.0
    }

    fn name(&self) -> &str {
        "PpgAmplitudeNorms"
    }
}

/// Pulse transit time (PTT) template
pub struct PttTemplate;

impl PopulationTemplate for PttTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // PTT in milliseconds (ECG R-peak to PPG pulse)
        200.0 // Normal: 150-250 ms
    }

    fn variance(&self, _context: &Context) -> f64 {
        30.0
    }

    fn name(&self) -> &str {
        "PttTemplate"
    }
}

// ============================================================================
// PPG Pulse Encoder
// ============================================================================

/// Configuration for PPG pulse detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpgPulseConfig {
    /// Minimum pulse height
    pub min_height: f32,
    /// Minimum distance between pulses (in seconds)
    pub min_distance: f64,
    /// Use low-pass filter
    pub use_filter: bool,
}

impl Default for PpgPulseConfig {
    fn default() -> Self {
        Self {
            min_height: 0.3,
            min_distance: 0.3,
            use_filter: true,
        }
    }
}

/// PPG pulse encoder - detects systolic peaks
pub struct PpgPulseEncoder;

impl PpgPulseEncoder {
    /// Creates a new [`PpgPulseEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn detect_peaks(
        &self,
        signal: &[f32],
        config: &PpgPulseConfig,
        sample_rate: f64,
    ) -> Vec<usize> {
        let mut peaks = Vec::new();
        let min_samples = (config.min_distance * sample_rate) as usize;

        // Normalize signal
        let min_val = signal.iter().copied().fold(f32::INFINITY, f32::min);
        let max_val = signal.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max_val - min_val;

        if range <= 0.0 {
            return peaks;
        }

        let threshold = config.min_height;
        let mut last_peak = 0;

        for i in 1..signal.len() - 1 {
            let normalized = (signal[i] - min_val) / range;
            if signal[i] > signal[i - 1]
                && signal[i] > signal[i + 1]
                && normalized > threshold
                && (i - last_peak) > min_samples
            {
                peaks.push(i);
                last_peak = i;
            }
        }

        peaks
    }
}

impl Default for PpgPulseEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PpgPulseEncoder {
    type Config = PpgPulseConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let peaks = self.detect_peaks(samples, config, sample_rate);
        let mut events = Vec::new();

        for peak_idx in peaks {
            let time = peak_idx as f64 * dt;
            let magnitude = samples[peak_idx];
            events.push(SpikeEvent::new(time, 0, 1, magnitude));
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "PpgPulseEncoder"
    }
}

// ============================================================================
// PPG Amplitude Encoder
// ============================================================================

/// Configuration for PPG amplitude encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpgAmplitudeConfig {
    /// Amplitude change threshold
    pub threshold: f32,
    /// Window size for amplitude calculation (in samples)
    pub window_size: usize,
}

impl Default for PpgAmplitudeConfig {
    fn default() -> Self {
        Self {
            threshold: 0.2,
            window_size: 100,
        }
    }
}

/// PPG amplitude encoder - detects perfusion changes
pub struct PpgAmplitudeEncoder;

impl PpgAmplitudeEncoder {
    /// Creates a new [`PpgAmplitudeEncoder`].
    pub fn new() -> Self {
        Self {}
    }

    fn calculate_amplitude(&self, window: &[f32]) -> f32 {
        let max_val = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val = window.iter().copied().fold(f32::INFINITY, f32::min);
        max_val - min_val
    }
}

impl Default for PpgAmplitudeEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PpgAmplitudeEncoder {
    type Config = PpgAmplitudeConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let mut events = Vec::new();
        let mut prev_amplitude = 0.0;

        for i in config.window_size..samples.len() {
            let window = &samples[i - config.window_size..i];
            let amplitude = self.calculate_amplitude(window);
            let change = (amplitude - prev_amplitude).abs();

            if i > config.window_size && change > config.threshold {
                let time = i as f64 * dt;
                let polarity = if amplitude > prev_amplitude { 1 } else { -1 };
                events.push(SpikeEvent::new(time, 0, polarity, change));
            }

            prev_amplitude = amplitude;
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "PpgAmplitudeEncoder"
    }
}

// ============================================================================
// PPG Pulse Transit Time (PTT) Encoder
// ============================================================================

/// Configuration for PTT encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PpgPttConfig {
    /// ECG R-peak times (reference)
    pub ecg_peaks: Vec<f64>,
    /// PTT deviation threshold (in ms)
    pub threshold: f32,
}

impl Default for PpgPttConfig {
    fn default() -> Self {
        Self {
            ecg_peaks: Vec::new(),
            threshold: 20.0,
        }
    }
}

/// PTT encoder - measures pulse transit time
pub struct PpgPttEncoder {
    template: PttTemplate,
}

impl PpgPttEncoder {
    /// Creates a new [`PpgPttEncoder`].
    pub fn new() -> Self {
        Self {
            template: PttTemplate,
        }
    }
}

impl Default for PpgPttEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for PpgPttEncoder {
    type Config = PpgPttConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let _samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let _dt = 1.0 / sample_rate;

        // Detect PPG pulses
        let pulse_config = PpgPulseConfig::default();
        let pulse_encoder = PpgPulseEncoder::new();
        let ppg_peaks = pulse_encoder.encode(signal, &pulse_config)?;

        let mut events = Vec::new();
        let context = Context::default();
        let expected_ptt = self.template.expected_value(&context);

        // Match ECG peaks with PPG peaks
        for ecg_time in &config.ecg_peaks {
            // Find next PPG peak after ECG peak
            if let Some(ppg_event) = ppg_peaks
                .iter()
                .find(|e| e.timestamp > *ecg_time && e.timestamp < ecg_time + 0.5)
            {
                let ptt = (ppg_event.timestamp - ecg_time) * 1000.0; // Convert to ms
                let deviation = (ptt - expected_ptt).abs();

                if deviation > config.threshold as f64 {
                    events.push(SpikeEvent::new(ppg_event.timestamp, 0, 1, deviation as f32));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "PpgPttEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_pulse_rate_template() {
        let template = PulseRateTemplate;
        // Reassigned below, so it stays mut.
        let context = Context {
            age: Some(30.0),
            ..Default::default()
        };
        assert_eq!(template.expected_value(&context), 72.0);
    }

    #[test]
    fn test_ppg_pulse_encoder() {
        // Simulate PPG with periodic pulses
        let mut data = vec![0.0; 1000];
        for i in (100..1000).step_by(200) {
            // Simulate pulse waveform
            if i + 20 < data.len() {
                for j in 0..20 {
                    data[i + j] = (j as f32 / 10.0).sin();
                }
            }
        }

        let signal = SignalBuffer::single_channel(data, 100.0);
        let encoder = PpgPulseEncoder::new();
        let config = PpgPulseConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }
}
