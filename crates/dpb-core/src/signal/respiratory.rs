//! Respiratory signal analysis and apnea detection.
//!
//! This module provides comprehensive respiratory analysis including:
//! - Breath detection from various modalities (impedance, belts, flow)
//! - Respiratory rate calculation
//! - Apnea and hypopnea detection
//! - Respiratory pattern analysis
//! - Sleep-disordered breathing metrics

use crate::error::{DpbError, Result};
use crate::signal::fft::{FftProcessor, fft_frequencies};
use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};

/// Individual breath event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BreathEvent {
    /// Start time (seconds)
    pub start_time: f64,
    /// End time (seconds)
    pub end_time: f64,
    /// Duration (seconds)
    pub duration: f64,
    /// Peak inspiratory amplitude
    pub inspiratory_amplitude: f64,
    /// Peak expiratory amplitude
    pub expiratory_amplitude: f64,
    /// Tidal volume estimate (arbitrary units)
    pub tidal_volume: f64,
    /// Inspiratory time (Ti)
    pub ti: f64,
    /// Expiratory time (Te)
    pub te: f64,
    /// Ti/Ttot ratio
    pub ti_ratio: f64,
}

/// Apnea/Hypopnea event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApneaEvent {
    /// Start time (seconds)
    pub start_time: f64,
    /// End time (seconds)
    pub end_time: f64,
    /// Duration (seconds)
    pub duration: f64,
    /// Event type
    pub event_type: ApneaType,
    /// Severity based on duration and desaturation
    pub severity: ApneaSeverity,
    /// Associated oxygen desaturation (if available)
    pub desaturation_percent: Option<f64>,
    /// Confidence score (0-1)
    pub confidence: f64,
}

/// Type of apnea event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ApneaType {
    /// Complete cessation of airflow (obstructive)
    ObstructiveApnea,
    /// Complete cessation (central - no effort)
    CentralApnea,
    /// Mixed (starts central, ends obstructive)
    MixedApnea,
    /// Partial reduction (≥30% reduction with desaturation)
    Hypopnea,
    /// Respiratory effort-related arousal
    Rera,
}

/// Apnea severity classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApneaSeverity {
    /// Duration < 10 seconds (not clinically significant)
    Subclinical,
    /// Duration 10-20 seconds
    Mild,
    /// Duration 20-40 seconds
    Moderate,
    /// Duration > 40 seconds
    Severe,
}

impl ApneaSeverity {
    /// Create from duration in seconds
    pub fn from_duration(duration: f64) -> Self {
        if duration < 10.0 {
            ApneaSeverity::Subclinical
        } else if duration < 20.0 {
            ApneaSeverity::Mild
        } else if duration < 40.0 {
            ApneaSeverity::Moderate
        } else {
            ApneaSeverity::Severe
        }
    }
}

/// Respiratory pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespiratoryPattern {
    /// Pattern type
    pub pattern_type: PatternType,
    /// Regularity score (0-1, 1 = very regular)
    pub regularity: f64,
    /// Presence of periodic breathing
    pub periodic_breathing: bool,
    /// Cycle length if periodic (seconds)
    pub cycle_length: Option<f64>,
}

/// Respiratory pattern types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PatternType {
    /// Normal regular breathing
    Normal,
    /// Cheyne-Stokes respiration
    CheyneStokes,
    /// Biot's respiration (ataxic)
    Biots,
    /// Kussmaul respiration (deep, rapid)
    Kussmaul,
    /// Tachypnea (rapid shallow)
    Tachypnea,
    /// Bradypnea (slow)
    Bradypnea,
    /// Irregular/ataxic
    Irregular,
}

/// Comprehensive respiratory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RespiratoryMetrics {
    /// Mean respiratory rate (breaths per minute)
    pub respiratory_rate: f64,
    /// Respiratory rate variability (CV)
    pub rate_variability: f64,
    /// Mean inspiratory time (seconds)
    pub mean_ti: f64,
    /// Mean expiratory time (seconds)
    pub mean_te: f64,
    /// Mean Ti/Ttot ratio
    pub mean_ti_ratio: f64,
    /// Mean tidal volume (arbitrary units)
    pub mean_tidal_volume: f64,
    /// Tidal volume variability (CV)
    pub tidal_volume_variability: f64,
    /// Minute ventilation estimate
    pub minute_ventilation: f64,
    /// Apnea-Hypopnea Index (events per hour)
    pub ahi: f64,
    /// Respiratory pattern
    pub pattern: RespiratoryPattern,
    /// Duration analyzed (seconds)
    pub duration_analyzed: f64,
}

/// Respiratory signal analyzer
pub struct RespiratoryAnalyzer {
    sample_rate: f64,
    /// Minimum breath duration (seconds)
    min_breath_duration: f64,
    /// Maximum breath duration (seconds)
    max_breath_duration: f64,
    /// Apnea duration threshold (seconds)
    apnea_threshold: f64,
    /// Hypopnea amplitude reduction threshold (fraction)
    hypopnea_threshold: f64,
}

impl RespiratoryAnalyzer {
    /// Create a new respiratory analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            min_breath_duration: 1.0,    // seconds (60 breaths/min max)
            max_breath_duration: 10.0,   // seconds (6 breaths/min min)
            apnea_threshold: 10.0,       // seconds
            hypopnea_threshold: 0.3,     // 30% reduction
        }
    }

    /// Configure breath detection parameters
    pub fn with_breath_params(mut self, min_duration: f64, max_duration: f64) -> Self {
        self.min_breath_duration = min_duration;
        self.max_breath_duration = max_duration;
        self
    }

    /// Configure apnea detection parameters
    pub fn with_apnea_params(mut self, apnea_threshold: f64, hypopnea_threshold: f64) -> Self {
        self.apnea_threshold = apnea_threshold;
        self.hypopnea_threshold = hypopnea_threshold;
        self
    }

    /// Detect individual breaths from respiratory signal
    pub fn detect_breaths(&self, signal: ArrayView1<f64>) -> Vec<BreathEvent> {
        let mut breaths = Vec::new();
        let n = signal.len();

        if n < (self.min_breath_duration * self.sample_rate) as usize * 2 {
            return breaths;
        }

        // Find zero crossings (breath boundaries)
        let mean = signal.mean().unwrap_or(0.0);
        let mut crossings = Vec::new();

        for i in 1..n {
            if (signal[i - 1] < mean && signal[i] >= mean) ||
               (signal[i - 1] >= mean && signal[i] < mean) {
                crossings.push(i);
            }
        }

        // Find peaks between zero crossings
        let mut i = 0;
        while i < crossings.len().saturating_sub(2) {
            let breath_start = crossings[i];

            // Find next positive-going crossing (inspiration start)
            let mut insp_start = breath_start;
            if signal[breath_start] < mean {
                // We're at expiration end, find inspiration
                if i + 1 < crossings.len() {
                    insp_start = crossings[i + 1];
                    i += 1;
                }
            }

            // Find end of this breath cycle
            if i + 2 >= crossings.len() {
                break;
            }

            let peak_idx = crossings[i + 1]; // Approximate peak
            let breath_end = crossings[i + 2];

            let duration = (breath_end - insp_start) as f64 / self.sample_rate;

            if duration >= self.min_breath_duration && duration <= self.max_breath_duration {
                // Find actual peak (max) during inspiration
                let insp_region = &signal.as_slice().unwrap()[insp_start..peak_idx.min(n)];
                let insp_amp = insp_region.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - mean;

                // Find trough during expiration
                let exp_region = &signal.as_slice().unwrap()[peak_idx.min(n)..breath_end.min(n)];
                let exp_amp = mean - exp_region.iter().cloned().fold(f64::INFINITY, f64::min);

                let ti = (peak_idx - insp_start) as f64 / self.sample_rate;
                let te = (breath_end - peak_idx) as f64 / self.sample_rate;
                let ti_ratio = ti / duration;

                // Estimate tidal volume from peak-to-trough amplitude
                let tidal_volume = insp_amp + exp_amp;

                breaths.push(BreathEvent {
                    start_time: insp_start as f64 / self.sample_rate,
                    end_time: breath_end as f64 / self.sample_rate,
                    duration,
                    inspiratory_amplitude: insp_amp,
                    expiratory_amplitude: exp_amp,
                    tidal_volume,
                    ti,
                    te,
                    ti_ratio,
                });
            }

            i += 2;
        }

        breaths
    }

    /// Detect apnea and hypopnea events
    pub fn detect_apneas(
        &self,
        signal: ArrayView1<f64>,
        breaths: &[BreathEvent],
    ) -> Vec<ApneaEvent> {
        let mut apneas = Vec::new();

        if breaths.len() < 2 {
            return apneas;
        }

        // Calculate baseline breath amplitude
        let baseline_amplitude: f64 = breaths.iter()
            .map(|b| b.tidal_volume)
            .sum::<f64>() / breaths.len() as f64;

        // Look for gaps between breaths (apneas) and reduced breaths (hypopneas)
        for i in 0..breaths.len() - 1 {
            let gap_start = breaths[i].end_time;
            let gap_end = breaths[i + 1].start_time;
            let gap_duration = gap_end - gap_start;

            // Check for apnea (pause > threshold)
            if gap_duration >= self.apnea_threshold {
                let severity = ApneaSeverity::from_duration(gap_duration);

                // Try to classify type based on signal characteristics
                let gap_start_sample = (gap_start * self.sample_rate) as usize;
                let gap_end_sample = (gap_end * self.sample_rate) as usize;

                let apnea_type = if gap_end_sample < signal.len() {
                    self.classify_apnea_type(signal, gap_start_sample, gap_end_sample)
                } else {
                    ApneaType::ObstructiveApnea
                };

                apneas.push(ApneaEvent {
                    start_time: gap_start,
                    end_time: gap_end,
                    duration: gap_duration,
                    event_type: apnea_type,
                    severity,
                    desaturation_percent: None,
                    confidence: 0.8,
                });
            }

            // Check for hypopnea (reduced amplitude)
            let next_breath_amp = breaths[i + 1].tidal_volume;
            let reduction = 1.0 - (next_breath_amp / baseline_amplitude);

            if reduction >= self.hypopnea_threshold && breaths[i + 1].duration >= 10.0 {
                apneas.push(ApneaEvent {
                    start_time: breaths[i + 1].start_time,
                    end_time: breaths[i + 1].end_time,
                    duration: breaths[i + 1].duration,
                    event_type: ApneaType::Hypopnea,
                    severity: ApneaSeverity::Mild,
                    desaturation_percent: None,
                    confidence: 0.6 + reduction * 0.3,
                });
            }
        }

        apneas
    }

    /// Classify apnea type based on effort signal
    fn classify_apnea_type(
        &self,
        signal: ArrayView1<f64>,
        start: usize,
        end: usize,
    ) -> ApneaType {
        let segment = signal.slice(ndarray::s![start..end]);

        // Calculate variance during apnea - high variance suggests obstructive
        let mean = segment.mean().unwrap_or(0.0);
        let variance: f64 = segment.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / segment.len() as f64;

        // Low variance suggests central apnea (no respiratory effort)
        let baseline_var = signal.var(0.0);

        if variance < baseline_var * 0.1 {
            ApneaType::CentralApnea
        } else if variance < baseline_var * 0.5 {
            ApneaType::MixedApnea
        } else {
            ApneaType::ObstructiveApnea
        }
    }

    /// Analyze respiratory pattern
    pub fn analyze_pattern(&self, breaths: &[BreathEvent]) -> RespiratoryPattern {
        if breaths.len() < 5 {
            return RespiratoryPattern {
                pattern_type: PatternType::Normal,
                regularity: 0.0,
                periodic_breathing: false,
                cycle_length: None,
            };
        }

        // Calculate breath interval statistics
        let intervals: Vec<f64> = breaths.windows(2)
            .map(|w| w[1].start_time - w[0].start_time)
            .collect();

        let mean_interval = intervals.iter().sum::<f64>() / intervals.len() as f64;
        let std_interval = (intervals.iter()
            .map(|x| (x - mean_interval).powi(2))
            .sum::<f64>() / intervals.len() as f64)
            .sqrt();
        let cv = std_interval / mean_interval;

        let respiratory_rate = 60.0 / mean_interval;
        let regularity = 1.0 - cv.min(1.0);

        // Calculate tidal volume variations for pattern detection
        let amplitudes: Vec<f64> = breaths.iter().map(|b| b.tidal_volume).collect();
        let mean_amp = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;

        // Check for periodic breathing (Cheyne-Stokes pattern)
        let periodic = self.detect_periodic_breathing(&amplitudes);

        // Classify pattern
        let pattern_type = if periodic.0 {
            PatternType::CheyneStokes
        } else if cv > 0.5 {
            PatternType::Irregular
        } else if respiratory_rate > 25.0 {
            if mean_amp < amplitudes.iter().cloned().fold(f64::INFINITY, f64::min) * 1.5 {
                PatternType::Tachypnea
            } else {
                PatternType::Kussmaul
            }
        } else if respiratory_rate < 8.0 {
            PatternType::Bradypnea
        } else {
            PatternType::Normal
        };

        RespiratoryPattern {
            pattern_type,
            regularity,
            periodic_breathing: periodic.0,
            cycle_length: periodic.1,
        }
    }

    /// Detect periodic breathing patterns
    fn detect_periodic_breathing(&self, amplitudes: &[f64]) -> (bool, Option<f64>) {
        if amplitudes.len() < 10 {
            return (false, None);
        }

        // Look for oscillations in amplitude
        let mean_amp = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        let normalized: Vec<f64> = amplitudes.iter().map(|a| a - mean_amp).collect();

        // Count zero crossings in normalized amplitude
        let mut crossings = 0;
        for i in 1..normalized.len() {
            if normalized[i - 1] * normalized[i] < 0.0 {
                crossings += 1;
            }
        }

        // Periodic if we have regular oscillations
        let expected_crossings_per_cycle = 2.0;
        let n_cycles = crossings as f64 / expected_crossings_per_cycle;

        if n_cycles >= 2.0 {
            let cycle_length = amplitudes.len() as f64 / n_cycles;
            // Cheyne-Stokes typically has 45-90 second cycles
            if (5.0..=20.0).contains(&cycle_length) {
                return (true, Some(cycle_length));
            }
        }

        (false, None)
    }

    /// Calculate respiratory rate from signal using spectral analysis
    pub fn calculate_respiratory_rate(&self, signal: ArrayView1<f64>) -> Result<f64> {
        let mut fft = FftProcessor::new();
        let psd = fft.psd(signal)?;
        let freqs = fft_frequencies(signal.len(), self.sample_rate);

        // Find peak in respiratory frequency range (0.1-0.5 Hz = 6-30 breaths/min)
        let mut max_power: f64 = 0.0;
        let mut peak_freq: f64 = 0.0;

        for (i, &freq) in freqs.iter().enumerate() {
            if (0.1..=0.5).contains(&freq) && psd[i] > max_power {
                max_power = psd[i];
                peak_freq = freq;
            }
        }

        if peak_freq > 0.0 {
            Ok(peak_freq * 60.0) // Convert Hz to breaths per minute
        } else {
            Err(DpbError::InvalidParameter(
                "Could not detect respiratory rate".to_string(),
            ))
        }
    }

    /// Perform comprehensive respiratory analysis
    pub fn analyze(&self, signal: ArrayView1<f64>) -> Result<RespiratoryMetrics> {
        let duration = signal.len() as f64 / self.sample_rate;

        // Detect breaths
        let breaths = self.detect_breaths(signal);

        if breaths.is_empty() {
            return Err(DpbError::InvalidParameter(
                "No breaths detected in signal".to_string(),
            ));
        }

        // Calculate basic metrics
        let n_breaths = breaths.len();
        let respiratory_rate = n_breaths as f64 / duration * 60.0;

        // Calculate intervals and variability
        let intervals: Vec<f64> = breaths.windows(2)
            .map(|w| w[1].start_time - w[0].start_time)
            .collect();

        let mean_interval = if !intervals.is_empty() {
            intervals.iter().sum::<f64>() / intervals.len() as f64
        } else {
            60.0 / respiratory_rate
        };

        let rate_variability = if intervals.len() > 1 {
            let std: f64 = (intervals.iter()
                .map(|x| (x - mean_interval).powi(2))
                .sum::<f64>() / intervals.len() as f64)
                .sqrt();
            std / mean_interval
        } else {
            0.0
        };

        // Calculate timing metrics
        let mean_ti = breaths.iter().map(|b| b.ti).sum::<f64>() / n_breaths as f64;
        let mean_te = breaths.iter().map(|b| b.te).sum::<f64>() / n_breaths as f64;
        let mean_ti_ratio = breaths.iter().map(|b| b.ti_ratio).sum::<f64>() / n_breaths as f64;

        // Calculate volume metrics
        let tidal_volumes: Vec<f64> = breaths.iter().map(|b| b.tidal_volume).collect();
        let mean_tidal_volume = tidal_volumes.iter().sum::<f64>() / n_breaths as f64;

        let tidal_volume_variability = if n_breaths > 1 {
            let std = (tidal_volumes.iter()
                .map(|v| (v - mean_tidal_volume).powi(2))
                .sum::<f64>() / n_breaths as f64)
                .sqrt();
            std / mean_tidal_volume
        } else {
            0.0
        };

        let minute_ventilation = mean_tidal_volume * respiratory_rate;

        // Detect apneas
        let apneas = self.detect_apneas(signal, &breaths);
        let hours = duration / 3600.0;
        let ahi = if hours > 0.0 {
            apneas.len() as f64 / hours
        } else {
            0.0
        };

        // Analyze pattern
        let pattern = self.analyze_pattern(&breaths);

        Ok(RespiratoryMetrics {
            respiratory_rate,
            rate_variability,
            mean_ti,
            mean_te,
            mean_ti_ratio,
            mean_tidal_volume,
            tidal_volume_variability,
            minute_ventilation,
            ahi,
            pattern,
            duration_analyzed: duration,
        })
    }
}

/// Sleep-disordered breathing analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepBreathingAnalysis {
    /// Apnea-Hypopnea Index
    pub ahi: f64,
    /// Oxygen Desaturation Index (requires SpO2 data)
    pub odi: Option<f64>,
    /// Sleep apnea severity classification
    pub severity: SleepApneaSeverity,
    /// Apnea events
    pub apneas: Vec<ApneaEvent>,
    /// Dominant apnea type
    pub dominant_type: ApneaType,
    /// Mean apnea duration
    pub mean_apnea_duration: f64,
    /// Longest apnea duration
    pub max_apnea_duration: f64,
    /// Total sleep time analyzed (hours)
    pub total_time_hours: f64,
}

/// Sleep apnea severity based on AHI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SleepApneaSeverity {
    /// AHI < 5: Normal
    Normal,
    /// AHI 5-15: Mild
    Mild,
    /// AHI 15-30: Moderate
    Moderate,
    /// AHI > 30: Severe
    Severe,
}

impl SleepApneaSeverity {
    /// Classify from AHI
    pub fn from_ahi(ahi: f64) -> Self {
        if ahi < 5.0 {
            SleepApneaSeverity::Normal
        } else if ahi < 15.0 {
            SleepApneaSeverity::Mild
        } else if ahi < 30.0 {
            SleepApneaSeverity::Moderate
        } else {
            SleepApneaSeverity::Severe
        }
    }

    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            SleepApneaSeverity::Normal => "Normal (AHI < 5)",
            SleepApneaSeverity::Mild => "Mild (AHI 5-15)",
            SleepApneaSeverity::Moderate => "Moderate (AHI 15-30)",
            SleepApneaSeverity::Severe => "Severe (AHI > 30)",
        }
    }
}

/// Analyze sleep-disordered breathing from respiratory signal
pub fn analyze_sleep_breathing(
    signal: ArrayView1<f64>,
    sample_rate: f64,
    total_time_hours: f64,
) -> Result<SleepBreathingAnalysis> {
    let analyzer = RespiratoryAnalyzer::new(sample_rate);
    let breaths = analyzer.detect_breaths(signal);
    let apneas = analyzer.detect_apneas(signal, &breaths);

    let ahi = apneas.len() as f64 / total_time_hours;
    let severity = SleepApneaSeverity::from_ahi(ahi);

    // Calculate apnea statistics
    let mean_duration = if !apneas.is_empty() {
        apneas.iter().map(|a| a.duration).sum::<f64>() / apneas.len() as f64
    } else {
        0.0
    };

    let max_duration = apneas.iter()
        .map(|a| a.duration)
        .fold(0.0, f64::max);

    // Find dominant type
    let mut type_counts = std::collections::HashMap::new();
    for apnea in &apneas {
        *type_counts.entry(apnea.event_type).or_insert(0) += 1;
    }

    let dominant_type = type_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(t, _)| t)
        .unwrap_or(ApneaType::ObstructiveApnea);

    Ok(SleepBreathingAnalysis {
        ahi,
        odi: None,
        severity,
        apneas,
        dominant_type,
        mean_apnea_duration: mean_duration,
        max_apnea_duration: max_duration,
        total_time_hours,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_breathing_signal(sample_rate: f64, duration: f64, resp_rate: f64) -> Array1<f64> {
        let n_samples = (sample_rate * duration) as usize;
        let freq = resp_rate / 60.0; // Convert to Hz

        Array1::from_shape_fn(n_samples, |i| {
            let t = i as f64 / sample_rate;
            // Simulate breathing with inspiration being faster than expiration
            (2.0 * std::f64::consts::PI * freq * t).sin()
        })
    }

    #[test]
    fn test_breath_detection() {
        let sample_rate = 100.0;
        let duration = 60.0; // 1 minute
        let resp_rate = 15.0; // 15 breaths/min

        let signal = create_breathing_signal(sample_rate, duration, resp_rate);
        let analyzer = RespiratoryAnalyzer::new(sample_rate);
        let breaths = analyzer.detect_breaths(signal.view());

        // Should detect approximately 15 breaths
        assert!(breaths.len() >= 12 && breaths.len() <= 18,
            "Expected ~15 breaths, got {}", breaths.len());
    }

    #[test]
    fn test_respiratory_rate_spectral() {
        let sample_rate = 100.0;
        let duration = 60.0;
        let resp_rate = 12.0;

        let signal = create_breathing_signal(sample_rate, duration, resp_rate);
        let analyzer = RespiratoryAnalyzer::new(sample_rate);
        let rate = analyzer.calculate_respiratory_rate(signal.view()).unwrap();

        assert!((rate - resp_rate).abs() < 2.0,
            "Expected rate ~{}, got {}", resp_rate, rate);
    }

    #[test]
    fn test_apnea_severity() {
        assert_eq!(ApneaSeverity::from_duration(5.0), ApneaSeverity::Subclinical);
        assert_eq!(ApneaSeverity::from_duration(15.0), ApneaSeverity::Mild);
        assert_eq!(ApneaSeverity::from_duration(30.0), ApneaSeverity::Moderate);
        assert_eq!(ApneaSeverity::from_duration(50.0), ApneaSeverity::Severe);
    }

    #[test]
    fn test_sleep_apnea_severity() {
        assert_eq!(SleepApneaSeverity::from_ahi(3.0), SleepApneaSeverity::Normal);
        assert_eq!(SleepApneaSeverity::from_ahi(10.0), SleepApneaSeverity::Mild);
        assert_eq!(SleepApneaSeverity::from_ahi(20.0), SleepApneaSeverity::Moderate);
        assert_eq!(SleepApneaSeverity::from_ahi(40.0), SleepApneaSeverity::Severe);
    }

    #[test]
    fn test_respiratory_analysis() {
        let sample_rate = 100.0;
        let duration = 120.0; // 2 minutes
        let resp_rate = 14.0;

        let signal = create_breathing_signal(sample_rate, duration, resp_rate);
        let analyzer = RespiratoryAnalyzer::new(sample_rate);
        let metrics = analyzer.analyze(signal.view()).unwrap();

        assert!((metrics.respiratory_rate - resp_rate).abs() < 3.0);
        assert!(metrics.mean_ti > 0.0);
        assert!(metrics.mean_te > 0.0);
    }

    #[test]
    fn test_pattern_types() {
        assert_eq!(PatternType::Normal, PatternType::Normal);
        assert_ne!(PatternType::CheyneStokes, PatternType::Normal);
    }
}
