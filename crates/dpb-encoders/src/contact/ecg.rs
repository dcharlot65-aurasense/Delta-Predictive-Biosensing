//! ECG (Electrocardiogram) encoders and population templates

use dpb_core::{
    Context, EventEncoder, PopulationTemplate, Result, Signal, SpikeEvent,
};
use serde::{Deserialize, Serialize};

// ============================================================================
// Population Templates for ECG
// ============================================================================

/// Heart rate population template by age
pub struct HeartRateTemplate;

impl PopulationTemplate for HeartRateTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // Age-based heart rate norms (beats per minute)
        match context.age {
            Some(age) if age < 1.0 => 140.0,     // Infant
            Some(age) if age < 3.0 => 120.0,     // Toddler
            Some(age) if age < 12.0 => 100.0,    // Child
            Some(age) if age < 18.0 => 85.0,     // Adolescent
            Some(age) if age < 65.0 => 72.0,     // Adult
            Some(_) => 75.0,                      // Senior
            None => 72.0,                         // Default adult
        }
    }

    fn variance(&self, context: &Context) -> f64 {
        // Standard deviation increases with age
        let base_std = 10.0;
        match context.age {
            Some(age) if age > 65.0 => base_std * 1.5,
            _ => base_std,
        }
    }

    fn name(&self) -> &str {
        "HeartRateTemplate"
    }
}

/// Heart rate variability (HRV) template
pub struct HrvTemplate;

impl PopulationTemplate for HrvTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // SDNN (standard deviation of NN intervals) in ms
        match context.age {
            Some(age) if age < 30.0 => 55.0,
            Some(age) if age < 50.0 => 45.0,
            Some(age) if age < 70.0 => 35.0,
            Some(_) => 30.0,
            None => 45.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        15.0 // Standard deviation of SDNN
    }

    fn name(&self) -> &str {
        "HrvTemplate"
    }
}

/// QRS duration template (normal QRS complex duration)
pub struct QrsDurationTemplate;

impl PopulationTemplate for QrsDurationTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // QRS duration in milliseconds
        90.0 // Normal: 80-100 ms
    }

    fn variance(&self, _context: &Context) -> f64 {
        10.0 // Standard deviation
    }

    fn name(&self) -> &str {
        "QrsDurationTemplate"
    }
}

/// PR interval template
pub struct PrIntervalTemplate;

impl PopulationTemplate for PrIntervalTemplate {
    fn expected_value(&self, _context: &Context) -> f64 {
        // PR interval in milliseconds
        160.0 // Normal: 120-200 ms
    }

    fn variance(&self, _context: &Context) -> f64 {
        20.0
    }

    fn name(&self) -> &str {
        "PrIntervalTemplate"
    }
}

/// QT interval template (corrected for heart rate)
pub struct QtIntervalTemplate;

impl PopulationTemplate for QtIntervalTemplate {
    fn expected_value(&self, context: &Context) -> f64 {
        // QTc (corrected QT) in milliseconds
        match context.sex.as_deref() {
            Some("M") | Some("Male") => 410.0, // Male: <430 ms
            Some("F") | Some("Female") => 420.0, // Female: <450 ms
            _ => 415.0,
        }
    }

    fn variance(&self, _context: &Context) -> f64 {
        20.0
    }

    fn name(&self) -> &str {
        "QtIntervalTemplate"
    }
}

// ============================================================================
// ECG R-Peak Encoder
// ============================================================================

/// Configuration for R-peak detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcgRPeakConfig {
    /// Minimum R-peak height (relative to mean)
    pub min_height: f32,
    /// Minimum distance between peaks (in seconds)
    pub min_distance: f64,
    /// Use bandpass filter (5-15 Hz)
    pub use_filter: bool,
}

impl Default for EcgRPeakConfig {
    fn default() -> Self {
        Self {
            min_height: 0.5,
            min_distance: 0.3, // 200 BPM max
            use_filter: true,
        }
    }
}

/// R-peak encoder - detects QRS complexes in ECG
pub struct EcgRPeakEncoder;

impl EcgRPeakEncoder {
    /// Creates a new [`EcgRPeakEncoder`].
    pub fn new() -> Self {
        Self {
        }
    }

    fn detect_peaks(&self, signal: &[f32], config: &EcgRPeakConfig) -> Vec<usize> {
        let mut peaks = Vec::new();

        // Simple peak detection
        let mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let threshold = mean + config.min_height;

        let min_samples = (config.min_distance * 250.0) as usize; // Assume 250 Hz

        let mut last_peak = 0;
        for i in 1..signal.len() - 1 {
            if signal[i] > signal[i - 1]
                && signal[i] > signal[i + 1]
                && signal[i] > threshold
                && (i - last_peak) > min_samples
            {
                peaks.push(i);
                last_peak = i;
            }
        }

        peaks
    }
}

impl Default for EcgRPeakEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EcgRPeakEncoder {
    type Config = EcgRPeakConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let peaks = self.detect_peaks(samples, config);
        let mut events = Vec::new();

        for peak_idx in peaks {
            let time = peak_idx as f64 * dt;
            let magnitude = samples[peak_idx];
            events.push(SpikeEvent::new(time, 0, 1, magnitude));
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EcgRPeakEncoder"
    }
}

// ============================================================================
// ECG Morphology Encoder
// ============================================================================

/// Configuration for ECG morphology encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcgMorphologyConfig {
    /// Expected QRS template
    pub qrs_template: Vec<f32>,
    /// Deviation threshold
    pub deviation_threshold: f32,
}

impl Default for EcgMorphologyConfig {
    fn default() -> Self {
        // Simplified QRS template
        Self {
            qrs_template: vec![0.0, 0.2, 1.0, 0.2, 0.0, -0.1, 0.0],
            deviation_threshold: 0.3,
        }
    }
}

/// ECG morphology encoder - detects abnormal QRS shapes
pub struct EcgMorphologyEncoder;

impl EcgMorphologyEncoder {
    /// Creates a new [`EcgMorphologyEncoder`].
    pub fn new() -> Self {
        Self {
        }
    }

    fn correlate(&self, signal: &[f32], template: &[f32]) -> f32 {
        if signal.len() != template.len() {
            return 0.0;
        }

        let sig_mean = signal.iter().sum::<f32>() / signal.len() as f32;
        let tmpl_mean = template.iter().sum::<f32>() / template.len() as f32;

        let mut num = 0.0;
        let mut sig_var = 0.0;
        let mut tmpl_var = 0.0;

        for i in 0..signal.len() {
            let sig_dev = signal[i] - sig_mean;
            let tmpl_dev = template[i] - tmpl_mean;
            num += sig_dev * tmpl_dev;
            sig_var += sig_dev * sig_dev;
            tmpl_var += tmpl_dev * tmpl_dev;
        }

        if sig_var > 0.0 && tmpl_var > 0.0 {
            num / (sig_var * tmpl_var).sqrt()
        } else {
            0.0
        }
    }
}

impl Default for EcgMorphologyEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EcgMorphologyEncoder {
    type Config = EcgMorphologyConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        let template = &config.qrs_template;
        let template_len = template.len();
        let mut events = Vec::new();

        if samples.len() < template_len {
            return Ok(events);
        }

        // Slide template across signal
        for i in 0..=samples.len() - template_len {
            let window = &samples[i..i + template_len];
            let correlation = self.correlate(window, template);
            let deviation = 1.0 - correlation;

            if deviation > config.deviation_threshold {
                let time = i as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EcgMorphologyEncoder"
    }
}

// ============================================================================
// ECG ST-Segment Deviation Encoder
// ============================================================================

/// Configuration for ST-segment analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcgStDeviationConfig {
    /// ST deviation threshold (in mV)
    pub threshold: f32,
    /// J-point offset from R-peak (in samples)
    pub j_point_offset: usize,
    /// ST segment measurement point (in samples after J-point)
    pub st_point_offset: usize,
}

impl Default for EcgStDeviationConfig {
    fn default() -> Self {
        Self {
            threshold: 0.1, // 100 μV
            j_point_offset: 20,
            st_point_offset: 20,
        }
    }
}

/// ST-segment deviation encoder - detects ischemia
pub struct EcgStDeviationEncoder;

impl EcgStDeviationEncoder {
    /// Creates a new [`EcgStDeviationEncoder`].
    pub fn new() -> Self {
        Self
    }
}

impl Default for EcgStDeviationEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EcgStDeviationEncoder {
    type Config = EcgStDeviationConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Simplified: detect R-peaks first, then measure ST segment
        let mut events = Vec::new();

        // Simple R-peak detection
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let mut r_peaks = Vec::new();

        for i in 1..samples.len() - 1 {
            if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] && samples[i] > mean {
                r_peaks.push(i);
            }
        }

        // Measure ST deviation at each R-peak
        for &r_peak in &r_peaks {
            let st_point = r_peak + config.j_point_offset + config.st_point_offset;
            if st_point < samples.len() {
                let baseline = samples[r_peak.saturating_sub(50)];
                let st_level = samples[st_point];
                let deviation = st_level - baseline;

                if deviation.abs() > config.threshold {
                    let time = st_point as f64 * dt;
                    let polarity = if deviation > 0.0 { 1 } else { -1 };
                    events.push(SpikeEvent::new(time, 0, polarity, deviation.abs()));
                }
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EcgStDeviationEncoder"
    }
}

// ============================================================================
// ECG HRV (Heart Rate Variability) Encoder
// ============================================================================

/// Configuration for HRV encoding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EcgHrvConfig {
    /// Window size for HRV calculation (in beats)
    pub window_size: usize,
    /// HRV deviation threshold (in ms)
    pub threshold: f32,
}

impl Default for EcgHrvConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            threshold: 10.0, // 10 ms deviation
        }
    }
}

/// HRV encoder - detects heart rate variability changes
pub struct EcgHrvEncoder {
    template: HrvTemplate,
}

impl EcgHrvEncoder {
    /// Creates a new [`EcgHrvEncoder`].
    pub fn new() -> Self {
        Self {
            template: HrvTemplate,
        }
    }

    fn calculate_rr_intervals(&self, peaks: &[usize], dt: f64) -> Vec<f64> {
        let mut rr_intervals = Vec::new();
        for i in 1..peaks.len() {
            let interval = (peaks[i] - peaks[i - 1]) as f64 * dt * 1000.0; // Convert to ms
            rr_intervals.push(interval);
        }
        rr_intervals
    }

    fn calculate_hrv(&self, rr_intervals: &[f64]) -> f64 {
        if rr_intervals.len() < 2 {
            return 0.0;
        }

        let mean = rr_intervals.iter().sum::<f64>() / rr_intervals.len() as f64;
        let variance = rr_intervals
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / rr_intervals.len() as f64;

        variance.sqrt() // SDNN
    }
}

impl Default for EcgHrvEncoder {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEncoder for EcgHrvEncoder {
    type Config = EcgHrvConfig;

    fn encode(&self, signal: &dyn Signal, config: &Self::Config) -> Result<Vec<SpikeEvent>> {
        let samples = signal.samples();
        let sample_rate = signal.sample_rate();
        let dt = 1.0 / sample_rate;

        // Detect R-peaks
        let mean = samples.iter().sum::<f32>() / samples.len() as f32;
        let mut r_peaks = Vec::new();

        for i in 1..samples.len() - 1 {
            if samples[i] > samples[i - 1] && samples[i] > samples[i + 1] && samples[i] > mean {
                r_peaks.push(i);
            }
        }

        let rr_intervals = self.calculate_rr_intervals(&r_peaks, dt);
        let mut events = Vec::new();

        // Calculate HRV in sliding windows
        for i in config.window_size..rr_intervals.len() {
            let window = &rr_intervals[i - config.window_size..i];
            let hrv = self.calculate_hrv(window);

            // Expected HRV from template (would use context in practice)
            let context = Context::default();
            let expected_hrv = self.template.expected_value(&context);
            let deviation = (hrv - expected_hrv).abs();

            if deviation > config.threshold as f64 {
                let time = r_peaks[i] as f64 * dt;
                events.push(SpikeEvent::new(time, 0, 1, deviation as f32));
            }
        }

        Ok(events)
    }

    fn name(&self) -> &str {
        "EcgHrvEncoder"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dpb_core::SignalBuffer;

    #[test]
    fn test_heart_rate_template() {
        let template = HeartRateTemplate;
        let mut context = Context::default();

        context.age = Some(30.0);
        assert_eq!(template.expected_value(&context), 72.0);

        context.age = Some(5.0);
        assert_eq!(template.expected_value(&context), 100.0);
    }

    #[test]
    fn test_ecg_rpeak_encoder() {
        // Simulate ECG with periodic peaks
        let mut data = vec![0.0; 1000];
        for i in (100..1000).step_by(200) {
            data[i] = 1.0; // R-peaks
        }

        let signal = SignalBuffer::single_channel(data, 250.0);
        let encoder = EcgRPeakEncoder::new();
        let config = EcgRPeakConfig::default();

        let events = encoder.encode(&signal, &config).unwrap();
        assert!(!events.is_empty());
    }
}
