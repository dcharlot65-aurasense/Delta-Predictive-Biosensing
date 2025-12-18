//! Pain-related autonomic responses
//!
//! Analyzes physiological responses to pain stimuli:
//! - Skin conductance response (SCR)
//! - Heart rate response
//! - Pupil dilation response
//! - Respiratory response

/// Pain autonomic response analyzer
#[derive(Debug, Clone)]
pub struct PainAutonomicAnalyzer {
    /// Sample rate of autonomic signals (Hz)
    pub sample_rate: f64,
    /// Pre-stimulus baseline window (seconds)
    pub baseline_window_sec: f64,
    /// Post-stimulus response window (seconds)
    pub response_window_sec: f64,
}

impl Default for PainAutonomicAnalyzer {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            baseline_window_sec: 5.0,
            response_window_sec: 10.0,
        }
    }
}

impl PainAutonomicAnalyzer {
    /// Create new analyzer
    pub fn new(sample_rate: f64) -> Self {
        Self {
            sample_rate,
            ..Default::default()
        }
    }

    /// Analyze skin conductance response to pain
    pub fn scr_to_pain(&self, eda: &[f64], pain_onset_sec: f64) -> ScrMetrics {
        if eda.is_empty() {
            return ScrMetrics::default();
        }

        let onset_sample = (pain_onset_sec * self.sample_rate) as usize;
        let baseline_start = onset_sample.saturating_sub((self.baseline_window_sec * self.sample_rate) as usize);
        let response_end = (onset_sample + (self.response_window_sec * self.sample_rate) as usize).min(eda.len());

        // Calculate baseline
        let baseline_samples = &eda[baseline_start..onset_sample.min(eda.len())];
        let baseline_mean = if baseline_samples.is_empty() {
            0.0
        } else {
            baseline_samples.iter().sum::<f64>() / baseline_samples.len() as f64
        };

        // Find response in post-stimulus window
        let response_samples = if onset_sample < eda.len() {
            &eda[onset_sample..response_end]
        } else {
            &[]
        };

        if response_samples.is_empty() {
            return ScrMetrics::default();
        }

        // Find SCR onset (first point exceeding baseline + threshold)
        let threshold = baseline_mean * 1.1; // 10% above baseline
        let mut scr_onset_idx = None;
        for (i, &v) in response_samples.iter().enumerate() {
            if v > threshold {
                scr_onset_idx = Some(i);
                break;
            }
        }

        // Find peak amplitude
        let peak_amplitude = response_samples.iter().cloned().fold(f64::NAN, f64::max);
        let peak_idx = response_samples.iter().position(|&v| v == peak_amplitude).unwrap_or(0);

        let scr_onset_latency = scr_onset_idx.map(|i| i as f64 / self.sample_rate);
        let peak_latency = peak_idx as f64 / self.sample_rate;

        // Calculate rise time and half-recovery time
        let rise_time = scr_onset_idx.map(|onset| (peak_idx - onset) as f64 / self.sample_rate);

        let half_peak = baseline_mean + (peak_amplitude - baseline_mean) / 2.0;
        let mut half_recovery_time = None;
        for (i, &v) in response_samples.iter().enumerate().skip(peak_idx) {
            if v < half_peak {
                half_recovery_time = Some((i - peak_idx) as f64 / self.sample_rate);
                break;
            }
        }

        ScrMetrics {
            baseline_level: baseline_mean,
            peak_amplitude,
            amplitude_change: peak_amplitude - baseline_mean,
            onset_latency_sec: scr_onset_latency,
            peak_latency_sec: peak_latency,
            rise_time_sec: rise_time,
            half_recovery_time_sec: half_recovery_time,
        }
    }

    /// Analyze heart rate response to pain
    pub fn hr_response(&self, rr_intervals_ms: &[f64], pain_onset_sec: f64) -> HrResponse {
        if rr_intervals_ms.is_empty() {
            return HrResponse::default();
        }

        // Convert RR intervals to cumulative time to find pain onset
        let mut cumulative_time = vec![0.0];
        for rr in rr_intervals_ms {
            cumulative_time.push(cumulative_time.last().unwrap() + rr / 1000.0);
        }

        // Find index at pain onset
        let onset_idx = cumulative_time
            .iter()
            .position(|&t| t >= pain_onset_sec)
            .unwrap_or(0);

        // Calculate baseline HR (5 beats before onset)
        let baseline_start = onset_idx.saturating_sub(5);
        let baseline_rr: Vec<f64> = rr_intervals_ms[baseline_start..onset_idx].to_vec();
        let baseline_hr = if baseline_rr.is_empty() {
            0.0
        } else {
            60000.0 / (baseline_rr.iter().sum::<f64>() / baseline_rr.len() as f64)
        };

        // Calculate response HR (10 beats after onset)
        let response_end = (onset_idx + 10).min(rr_intervals_ms.len());
        let response_rr: Vec<f64> = rr_intervals_ms[onset_idx..response_end].to_vec();

        if response_rr.is_empty() {
            return HrResponse {
                baseline_hr_bpm: baseline_hr,
                ..Default::default()
            };
        }

        // Find peak HR in response window
        let min_rr = response_rr.iter().cloned().fold(f64::INFINITY, f64::min);
        let peak_hr = 60000.0 / min_rr;

        let peak_idx = response_rr.iter().position(|&v| v == min_rr).unwrap_or(0);
        let peak_latency = response_rr[..=peak_idx].iter().sum::<f64>() / 1000.0;

        // Calculate mean response HR
        let mean_response_hr = 60000.0 / (response_rr.iter().sum::<f64>() / response_rr.len() as f64);

        // Calculate HRV in response window (RMSSD)
        let rmssd = self.calculate_rmssd(&response_rr);

        HrResponse {
            baseline_hr_bpm: baseline_hr,
            peak_hr_bpm: peak_hr,
            hr_change_bpm: peak_hr - baseline_hr,
            peak_latency_sec: peak_latency,
            mean_response_hr_bpm: mean_response_hr,
            response_rmssd_ms: rmssd,
        }
    }

    /// Analyze pupil dilation response to pain
    pub fn pupil_response(&self, pupil_diameter_mm: &[f64], pain_onset_sec: f64) -> PupilResponse {
        if pupil_diameter_mm.is_empty() {
            return PupilResponse::default();
        }

        let onset_sample = (pain_onset_sec * self.sample_rate) as usize;
        let baseline_start = onset_sample.saturating_sub((self.baseline_window_sec * self.sample_rate) as usize);
        let response_end = (onset_sample + (self.response_window_sec * self.sample_rate) as usize).min(pupil_diameter_mm.len());

        // Calculate baseline
        let baseline_samples = &pupil_diameter_mm[baseline_start..onset_sample.min(pupil_diameter_mm.len())];
        let baseline_diameter = if baseline_samples.is_empty() {
            0.0
        } else {
            baseline_samples.iter().sum::<f64>() / baseline_samples.len() as f64
        };

        // Get response window
        let response_samples = if onset_sample < pupil_diameter_mm.len() {
            &pupil_diameter_mm[onset_sample..response_end]
        } else {
            &[]
        };

        if response_samples.is_empty() {
            return PupilResponse {
                baseline_diameter_mm: baseline_diameter,
                ..Default::default()
            };
        }

        // Find peak dilation
        let peak_diameter = response_samples.iter().cloned().fold(f64::NAN, f64::max);
        let peak_idx = response_samples.iter().position(|&v| v == peak_diameter).unwrap_or(0);
        let peak_latency = peak_idx as f64 / self.sample_rate;

        // Calculate dilation metrics
        let absolute_dilation = peak_diameter - baseline_diameter;
        let percent_dilation = if baseline_diameter > 0.0 {
            (absolute_dilation / baseline_diameter) * 100.0
        } else {
            0.0
        };

        // Calculate area under curve (integral of dilation)
        let dt = 1.0 / self.sample_rate;
        let auc: f64 = response_samples
            .iter()
            .map(|&d| (d - baseline_diameter).max(0.0) * dt)
            .sum();

        PupilResponse {
            baseline_diameter_mm: baseline_diameter,
            peak_diameter_mm: peak_diameter,
            absolute_dilation_mm: absolute_dilation,
            percent_dilation,
            peak_latency_sec: peak_latency,
            dilation_auc: auc,
        }
    }

    /// Analyze respiratory response to pain
    pub fn respiratory_response(
        &self,
        respiratory_signal: &[f64],
        pain_onset_sec: f64,
    ) -> RespiratoryResponse {
        if respiratory_signal.is_empty() {
            return RespiratoryResponse::default();
        }

        let onset_sample = (pain_onset_sec * self.sample_rate) as usize;
        let baseline_start = onset_sample.saturating_sub((self.baseline_window_sec * self.sample_rate) as usize);
        let response_end = (onset_sample + (self.response_window_sec * self.sample_rate) as usize).min(respiratory_signal.len());

        // Calculate baseline respiratory rate
        let baseline_samples = &respiratory_signal[baseline_start..onset_sample.min(respiratory_signal.len())];
        let baseline_rate = self.estimate_respiratory_rate(baseline_samples);

        // Calculate response respiratory rate
        let response_samples = if onset_sample < respiratory_signal.len() {
            &respiratory_signal[onset_sample..response_end]
        } else {
            &[]
        };
        let response_rate = self.estimate_respiratory_rate(response_samples);

        // Check for breath hold (reduced amplitude/rate)
        let baseline_amplitude = self.calculate_amplitude(baseline_samples);
        let response_amplitude = self.calculate_amplitude(response_samples);
        let breath_hold_detected = response_rate < baseline_rate * 0.5 || response_amplitude < baseline_amplitude * 0.3;

        RespiratoryResponse {
            baseline_rate_bpm: baseline_rate,
            response_rate_bpm: response_rate,
            rate_change_bpm: response_rate - baseline_rate,
            breath_hold_detected,
        }
    }

    /// Generate comprehensive autonomic profile for pain response
    pub fn analyze_pain_response(
        &self,
        eda: Option<&[f64]>,
        rr_intervals: Option<&[f64]>,
        pupil: Option<&[f64]>,
        respiratory: Option<&[f64]>,
        pain_onset_sec: f64,
    ) -> AutonomicPainProfile {
        let scr = eda.map(|e| self.scr_to_pain(e, pain_onset_sec));
        let hr = rr_intervals.map(|r| self.hr_response(r, pain_onset_sec));
        let pupil_response = pupil.map(|p| self.pupil_response(p, pain_onset_sec));
        let respiratory_response = respiratory.map(|r| self.respiratory_response(r, pain_onset_sec));

        // Calculate composite autonomic response score
        let composite = self.calculate_composite_score(&scr, &hr, &pupil_response);

        AutonomicPainProfile {
            scr_metrics: scr,
            hr_response: hr,
            pupil_response,
            respiratory_response,
            composite_autonomic_score: composite,
        }
    }

    /// Calculate composite autonomic response score
    fn calculate_composite_score(
        &self,
        scr: &Option<ScrMetrics>,
        hr: &Option<HrResponse>,
        pupil: &Option<PupilResponse>,
    ) -> f64 {
        let mut score = 0.0;
        let mut n_measures = 0.0;

        if let Some(s) = scr {
            // Normalize SCR amplitude change (typical range 0-2 µS)
            score += (s.amplitude_change / 2.0).min(1.0);
            n_measures += 1.0;
        }

        if let Some(h) = hr {
            // Normalize HR change (typical range 0-30 bpm)
            score += (h.hr_change_bpm / 30.0).min(1.0);
            n_measures += 1.0;
        }

        if let Some(p) = pupil {
            // Normalize pupil dilation (typical range 0-30%)
            score += (p.percent_dilation / 30.0).min(1.0);
            n_measures += 1.0;
        }

        if n_measures > 0.0 {
            score / n_measures * 100.0
        } else {
            0.0
        }
    }

    /// Calculate RMSSD from RR intervals
    fn calculate_rmssd(&self, rr_intervals: &[f64]) -> f64 {
        if rr_intervals.len() < 2 {
            return 0.0;
        }

        let successive_diffs: Vec<f64> = rr_intervals
            .windows(2)
            .map(|w| (w[1] - w[0]).powi(2))
            .collect();

        (successive_diffs.iter().sum::<f64>() / successive_diffs.len() as f64).sqrt()
    }

    /// Estimate respiratory rate from signal
    fn estimate_respiratory_rate(&self, signal: &[f64]) -> f64 {
        if signal.len() < 10 {
            return 0.0;
        }

        // Simple zero-crossing rate estimation
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let mut crossings = 0;

        for i in 1..signal.len() {
            if (signal[i - 1] < mean && signal[i] >= mean)
                || (signal[i - 1] >= mean && signal[i] < mean)
            {
                crossings += 1;
            }
        }

        // Each breath cycle has 2 zero crossings
        let duration_sec = signal.len() as f64 / self.sample_rate;
        (crossings as f64 / 2.0) / duration_sec * 60.0
    }

    /// Calculate signal amplitude (peak-to-peak)
    fn calculate_amplitude(&self, signal: &[f64]) -> f64 {
        if signal.is_empty() {
            return 0.0;
        }

        let max = signal.iter().cloned().fold(f64::NAN, f64::max);
        let min = signal.iter().cloned().fold(f64::NAN, f64::min);
        max - min
    }
}

/// Skin conductance response metrics
#[derive(Debug, Clone, Default)]
pub struct ScrMetrics {
    /// Baseline SCL (µS)
    pub baseline_level: f64,
    /// Peak amplitude (µS)
    pub peak_amplitude: f64,
    /// Change from baseline (µS)
    pub amplitude_change: f64,
    /// Onset latency (seconds)
    pub onset_latency_sec: Option<f64>,
    /// Peak latency (seconds)
    pub peak_latency_sec: f64,
    /// Rise time (seconds)
    pub rise_time_sec: Option<f64>,
    /// Half-recovery time (seconds)
    pub half_recovery_time_sec: Option<f64>,
}

/// Heart rate response metrics
#[derive(Debug, Clone, Default)]
pub struct HrResponse {
    /// Baseline heart rate (bpm)
    pub baseline_hr_bpm: f64,
    /// Peak heart rate (bpm)
    pub peak_hr_bpm: f64,
    /// HR change from baseline (bpm)
    pub hr_change_bpm: f64,
    /// Peak latency (seconds)
    pub peak_latency_sec: f64,
    /// Mean response HR (bpm)
    pub mean_response_hr_bpm: f64,
    /// RMSSD during response (ms)
    pub response_rmssd_ms: f64,
}

/// Pupil dilation response metrics
#[derive(Debug, Clone, Default)]
pub struct PupilResponse {
    /// Baseline pupil diameter (mm)
    pub baseline_diameter_mm: f64,
    /// Peak pupil diameter (mm)
    pub peak_diameter_mm: f64,
    /// Absolute dilation (mm)
    pub absolute_dilation_mm: f64,
    /// Percent dilation
    pub percent_dilation: f64,
    /// Peak latency (seconds)
    pub peak_latency_sec: f64,
    /// Area under dilation curve
    pub dilation_auc: f64,
}

/// Respiratory response metrics
#[derive(Debug, Clone, Default)]
pub struct RespiratoryResponse {
    /// Baseline respiratory rate (breaths/min)
    pub baseline_rate_bpm: f64,
    /// Response respiratory rate (breaths/min)
    pub response_rate_bpm: f64,
    /// Rate change from baseline (breaths/min)
    pub rate_change_bpm: f64,
    /// Whether breath hold was detected
    pub breath_hold_detected: bool,
}

/// Comprehensive autonomic pain profile
#[derive(Debug, Clone, Default)]
pub struct AutonomicPainProfile {
    /// SCR metrics
    pub scr_metrics: Option<ScrMetrics>,
    /// HR response metrics
    pub hr_response: Option<HrResponse>,
    /// Pupil response metrics
    pub pupil_response: Option<PupilResponse>,
    /// Respiratory response metrics
    pub respiratory_response: Option<RespiratoryResponse>,
    /// Composite autonomic response score (0-100)
    pub composite_autonomic_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scr_analysis() {
        let analyzer = PainAutonomicAnalyzer::default();

        // Generate synthetic EDA data with SCR
        let mut eda: Vec<f64> = vec![5.0; 500]; // Baseline
        for i in 500..800 {
            // Response
            let t = (i - 500) as f64 / 100.0;
            eda.push(5.0 + 1.5 * (1.0 - (-t).exp()) * (-t / 3.0).exp());
        }
        eda.extend(vec![5.0; 200]); // Recovery

        let scr = analyzer.scr_to_pain(&eda, 5.0);

        assert!(scr.baseline_level > 4.5 && scr.baseline_level < 5.5);
        assert!(scr.amplitude_change > 0.0);
    }

    #[test]
    fn test_hr_response() {
        let analyzer = PainAutonomicAnalyzer::default();

        // Generate RR intervals (baseline 800ms = 75bpm, response 700ms = 86bpm)
        let mut rr: Vec<f64> = vec![800.0; 10]; // Baseline
        rr.extend(vec![700.0; 10]); // Response
        rr.extend(vec![750.0; 10]); // Recovery

        let hr = analyzer.hr_response(&rr, 8.0);

        assert!(hr.baseline_hr_bpm > 70.0 && hr.baseline_hr_bpm < 80.0);
        assert!(hr.hr_change_bpm > 0.0);
    }

    #[test]
    fn test_pupil_response() {
        let analyzer = PainAutonomicAnalyzer::default();

        // Generate pupil data with dilation
        let mut pupil: Vec<f64> = vec![4.0; 500]; // Baseline
        for i in 500..700 {
            pupil.push(4.0 + 0.8 * ((i - 500) as f64 / 200.0).min(1.0));
        }
        pupil.extend(vec![4.5; 300]); // Sustained dilation

        let response = analyzer.pupil_response(&pupil, 5.0);

        assert!(response.baseline_diameter_mm > 3.8 && response.baseline_diameter_mm < 4.2);
        assert!(response.absolute_dilation_mm > 0.0);
    }
}
