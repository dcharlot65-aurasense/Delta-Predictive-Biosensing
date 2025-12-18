//! Pressure pain threshold and tolerance measurement
//!
//! Implements algorithms for:
//! - Pressure pain threshold (PPT) using algometry
//! - Pain tolerance measurement
//! - Conditioned Pain Modulation (CPM)

/// Pressure pain threshold analyzer
#[derive(Debug, Clone)]
pub struct PressurePainThreshold {
    /// Sample rate of pressure data (Hz)
    pub sample_rate: f64,
    /// Pressure application rate (kPa/s)
    pub application_rate: f64,
}

impl Default for PressurePainThreshold {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            application_rate: 30.0, // Standard 30 kPa/s
        }
    }
}

impl PressurePainThreshold {
    /// Create new PPT analyzer
    pub fn new(sample_rate: f64, application_rate: f64) -> Self {
        Self {
            sample_rate,
            application_rate,
        }
    }

    /// Calculate pain threshold from pressure ramp and pain onset time
    pub fn calculate_threshold(&self, pressure: &[f64], pain_onset_time: f64) -> f64 {
        if pressure.is_empty() {
            return 0.0;
        }

        let sample_index = (pain_onset_time * self.sample_rate) as usize;
        if sample_index < pressure.len() {
            pressure[sample_index]
        } else {
            *pressure.last().unwrap()
        }
    }

    /// Calculate pain tolerance level
    pub fn calculate_tolerance(&self, pressure: &[f64], tolerance_time: f64) -> f64 {
        if pressure.is_empty() {
            return 0.0;
        }

        let sample_index = (tolerance_time * self.sample_rate) as usize;
        if sample_index < pressure.len() {
            pressure[sample_index]
        } else {
            *pressure.last().unwrap()
        }
    }

    /// Calculate suprathreshold pain intensity
    /// Pressure at which specific pain intensity (e.g., 5/10) is reached
    pub fn suprathreshold_pressure(&self, pressure: &[f64], pain_ratings: &[f64], target_rating: f64) -> Option<f64> {
        if pressure.len() != pain_ratings.len() || pressure.is_empty() {
            return None;
        }

        // Find first point where rating exceeds target
        for (i, &rating) in pain_ratings.iter().enumerate() {
            if rating >= target_rating {
                return Some(pressure[i]);
            }
        }

        None
    }

    /// Analyze PPT from multiple trials
    pub fn analyze_trials(&self, trials: &[PptTrial]) -> PptMetrics {
        if trials.is_empty() {
            return PptMetrics::default();
        }

        let thresholds: Vec<f64> = trials.iter().map(|t| t.threshold_kpa).collect();

        let mean_threshold = thresholds.iter().sum::<f64>() / thresholds.len() as f64;

        let variance: f64 = thresholds
            .iter()
            .map(|t| (t - mean_threshold).powi(2))
            .sum::<f64>()
            / thresholds.len() as f64;
        let threshold_variability = variance.sqrt();

        // Calculate coefficient of variation
        let cv = if mean_threshold > 0.0 {
            threshold_variability / mean_threshold * 100.0
        } else {
            0.0
        };

        PptMetrics {
            mean_threshold_kpa: mean_threshold,
            threshold_variability_kpa: threshold_variability,
            coefficient_of_variation: cv,
            n_trials: trials.len(),
        }
    }

    /// Calculate side-to-side difference
    pub fn side_difference(&self, left: &[PptTrial], right: &[PptTrial]) -> SideDifference {
        let left_mean = if left.is_empty() {
            0.0
        } else {
            left.iter().map(|t| t.threshold_kpa).sum::<f64>() / left.len() as f64
        };

        let right_mean = if right.is_empty() {
            0.0
        } else {
            right.iter().map(|t| t.threshold_kpa).sum::<f64>() / right.len() as f64
        };

        let absolute_diff = (left_mean - right_mean).abs();
        let asymmetry = if (left_mean + right_mean) > 0.0 {
            (left_mean - right_mean) / ((left_mean + right_mean) / 2.0) * 100.0
        } else {
            0.0
        };

        SideDifference {
            left_mean_kpa: left_mean,
            right_mean_kpa: right_mean,
            absolute_difference_kpa: absolute_diff,
            asymmetry_percent: asymmetry,
        }
    }

    /// Get normative percentile for PPT
    pub fn normative_percentile(&self, threshold_kpa: f64, site: PptSite, sex: Sex) -> f64 {
        let (mean, sd) = self.normative_data(site, sex);
        let z = (threshold_kpa - mean) / sd;
        normal_cdf(z) * 100.0
    }

    /// Get normative data for site and sex
    fn normative_data(&self, site: PptSite, sex: Sex) -> (f64, f64) {
        // Normative data from literature (approximate values)
        match (site, sex) {
            (PptSite::Trapezius, Sex::Male) => (350.0, 100.0),
            (PptSite::Trapezius, Sex::Female) => (280.0, 90.0),
            (PptSite::Deltoid, Sex::Male) => (380.0, 110.0),
            (PptSite::Deltoid, Sex::Female) => (300.0, 95.0),
            (PptSite::Forearm, Sex::Male) => (320.0, 90.0),
            (PptSite::Forearm, Sex::Female) => (250.0, 80.0),
            (PptSite::Quadriceps, Sex::Male) => (450.0, 130.0),
            (PptSite::Quadriceps, Sex::Female) => (350.0, 110.0),
            (PptSite::TibialisAnterior, Sex::Male) => (400.0, 120.0),
            (PptSite::TibialisAnterior, Sex::Female) => (320.0, 100.0),
        }
    }
}

/// Single PPT trial result
#[derive(Debug, Clone)]
pub struct PptTrial {
    /// Threshold in kPa
    pub threshold_kpa: f64,
    /// Site tested
    pub site: PptSite,
    /// Side (if applicable)
    pub side: Option<Side>,
    /// Trial number
    pub trial_number: usize,
}

/// PPT measurement site
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PptSite {
    /// Upper trapezius muscle
    Trapezius,
    /// Deltoid muscle
    Deltoid,
    /// Extensor carpi radialis (forearm)
    Forearm,
    /// Quadriceps muscle
    Quadriceps,
    /// Tibialis anterior muscle
    TibialisAnterior,
}

/// Side of body
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Left,
    Right,
}

/// Sex for normative comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sex {
    Male,
    Female,
}

/// PPT metrics from multiple trials
#[derive(Debug, Clone, Default)]
pub struct PptMetrics {
    /// Mean threshold in kPa
    pub mean_threshold_kpa: f64,
    /// Variability of threshold (SD)
    pub threshold_variability_kpa: f64,
    /// Coefficient of variation (%)
    pub coefficient_of_variation: f64,
    /// Number of trials
    pub n_trials: usize,
}

/// Side-to-side difference metrics
#[derive(Debug, Clone, Default)]
pub struct SideDifference {
    /// Left side mean
    pub left_mean_kpa: f64,
    /// Right side mean
    pub right_mean_kpa: f64,
    /// Absolute difference
    pub absolute_difference_kpa: f64,
    /// Asymmetry percentage
    pub asymmetry_percent: f64,
}

/// Conditioned Pain Modulation analyzer
#[derive(Debug, Clone)]
pub struct ConditionedPainModulation {
    /// Type of conditioning stimulus
    pub conditioning_stimulus: StimulusType,
    /// Type of test stimulus
    pub test_stimulus: StimulusType,
}

impl ConditionedPainModulation {
    /// Create new CPM analyzer
    pub fn new(conditioning_stimulus: StimulusType, test_stimulus: StimulusType) -> Self {
        Self {
            conditioning_stimulus,
            test_stimulus,
        }
    }

    /// Calculate CPM effect (absolute change in pain threshold)
    pub fn cpm_effect(&self, baseline_threshold: f64, conditioned_threshold: f64) -> f64 {
        conditioned_threshold - baseline_threshold
    }

    /// Calculate CPM effect percentage
    pub fn cpm_effect_percent(&self, baseline_threshold: f64, conditioned_threshold: f64) -> f64 {
        if baseline_threshold <= 0.0 {
            return 0.0;
        }
        ((conditioned_threshold - baseline_threshold) / baseline_threshold) * 100.0
    }

    /// Calculate CPM efficiency (effect relative to conditioning pain)
    pub fn cpm_efficiency(&self, cpm_effect: f64, conditioning_pain: f64) -> f64 {
        if conditioning_pain <= 0.0 {
            return 0.0;
        }
        cpm_effect / conditioning_pain
    }

    /// Classify CPM response
    pub fn classify_response(&self, cpm_effect_percent: f64) -> CpmResponse {
        if cpm_effect_percent > 15.0 {
            CpmResponse::PronouncedInhibition
        } else if cpm_effect_percent > 5.0 {
            CpmResponse::ModerateInhibition
        } else if cpm_effect_percent > -5.0 {
            CpmResponse::NoEffect
        } else if cpm_effect_percent > -15.0 {
            CpmResponse::ModerateFacilitation
        } else {
            CpmResponse::PronouncedFacilitation
        }
    }

    /// Analyze CPM from multiple test points
    pub fn analyze_session(&self, session: &CpmSession) -> CpmMetrics {
        let cpm_effect = self.cpm_effect(session.baseline_threshold, session.conditioned_threshold);
        let cpm_effect_percent =
            self.cpm_effect_percent(session.baseline_threshold, session.conditioned_threshold);
        let efficiency = self.cpm_efficiency(cpm_effect, session.conditioning_pain_rating);
        let response = self.classify_response(cpm_effect_percent);

        CpmMetrics {
            baseline_threshold: session.baseline_threshold,
            conditioned_threshold: session.conditioned_threshold,
            cpm_effect,
            cpm_effect_percent,
            cpm_efficiency: efficiency,
            response,
            conditioning_pain_rating: session.conditioning_pain_rating,
        }
    }
}

/// Type of pain stimulus
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StimulusType {
    /// Pressure from algometer
    Pressure,
    /// Cold water or cold plate
    Cold,
    /// Heat stimulus
    Heat,
    /// Electrical stimulus
    Electrical,
    /// Ischemic (tourniquet)
    Ischemic,
}

/// CPM session data
#[derive(Debug, Clone)]
pub struct CpmSession {
    /// Baseline test threshold
    pub baseline_threshold: f64,
    /// Threshold during conditioning
    pub conditioned_threshold: f64,
    /// Pain rating during conditioning stimulus
    pub conditioning_pain_rating: f64,
    /// Duration of conditioning (seconds)
    pub conditioning_duration: f64,
}

/// CPM response classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpmResponse {
    /// Strong inhibition (>15% increase)
    PronouncedInhibition,
    /// Moderate inhibition (5-15% increase)
    ModerateInhibition,
    /// No significant effect (-5% to 5%)
    NoEffect,
    /// Moderate facilitation (-15% to -5%)
    ModerateFacilitation,
    /// Strong facilitation (<-15%)
    PronouncedFacilitation,
}

/// CPM analysis metrics
#[derive(Debug, Clone)]
pub struct CpmMetrics {
    /// Baseline threshold
    pub baseline_threshold: f64,
    /// Threshold during/after conditioning
    pub conditioned_threshold: f64,
    /// Absolute CPM effect
    pub cpm_effect: f64,
    /// Percent CPM effect
    pub cpm_effect_percent: f64,
    /// CPM efficiency
    pub cpm_efficiency: f64,
    /// Response classification
    pub response: CpmResponse,
    /// Conditioning stimulus pain rating
    pub conditioning_pain_rating: f64,
}

fn normal_cdf(z: f64) -> f64 {
    0.5 * (1.0 + erf(z / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppt_calculation() {
        let ppt = PressurePainThreshold::default();

        // Simulate linear pressure ramp at 30 kPa/s
        let pressure: Vec<f64> = (0..300).map(|i| i as f64 * 0.3).collect(); // 0-90 kPa over 3s

        // Pain onset at 1.5 seconds = 45 kPa
        let threshold = ppt.calculate_threshold(&pressure, 1.5);
        assert!((threshold - 45.0).abs() < 1.0);
    }

    #[test]
    fn test_cpm_effect() {
        let cpm = ConditionedPainModulation::new(StimulusType::Cold, StimulusType::Pressure);

        let effect = cpm.cpm_effect(200.0, 250.0);
        assert!((effect - 50.0).abs() < 0.01);

        let effect_pct = cpm.cpm_effect_percent(200.0, 250.0);
        assert!((effect_pct - 25.0).abs() < 0.01);
    }

    #[test]
    fn test_cpm_classification() {
        let cpm = ConditionedPainModulation::new(StimulusType::Cold, StimulusType::Pressure);

        assert_eq!(cpm.classify_response(20.0), CpmResponse::PronouncedInhibition);
        assert_eq!(cpm.classify_response(10.0), CpmResponse::ModerateInhibition);
        assert_eq!(cpm.classify_response(0.0), CpmResponse::NoEffect);
        assert_eq!(cpm.classify_response(-10.0), CpmResponse::ModerateFacilitation);
    }

    #[test]
    fn test_side_difference() {
        let ppt = PressurePainThreshold::default();

        let left = vec![
            PptTrial { threshold_kpa: 300.0, site: PptSite::Trapezius, side: Some(Side::Left), trial_number: 1 },
            PptTrial { threshold_kpa: 310.0, site: PptSite::Trapezius, side: Some(Side::Left), trial_number: 2 },
        ];
        let right = vec![
            PptTrial { threshold_kpa: 350.0, site: PptSite::Trapezius, side: Some(Side::Right), trial_number: 1 },
            PptTrial { threshold_kpa: 360.0, site: PptSite::Trapezius, side: Some(Side::Right), trial_number: 2 },
        ];

        let diff = ppt.side_difference(&left, &right);
        assert!((diff.left_mean_kpa - 305.0).abs() < 0.1);
        assert!((diff.right_mean_kpa - 355.0).abs() < 0.1);
    }
}
