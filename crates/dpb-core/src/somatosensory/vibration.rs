//! Vibration perception threshold (VPT) assessment
//!
//! Implements algorithms for measuring vibration sense using tuning forks
//! or vibrometers, with normative data for neuropathy screening.

/// Anatomical site for vibration testing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VibrationSite {
    /// Great toe (hallux) - most distal, most sensitive to neuropathy
    GreatToe,
    /// Medial malleolus (ankle)
    MedialMalleolus,
    /// Tibial tuberosity (knee)
    TibialTuberosity,
    /// Anterior superior iliac spine
    Asis,
    /// Ulnar styloid (wrist)
    UlnarStyloid,
    /// Index finger
    IndexFinger,
}

impl VibrationSite {
    /// Get the expected gradient factor relative to great toe
    /// Lower limb sites should have higher thresholds distally
    pub fn gradient_factor(&self) -> f64 {
        match self {
            VibrationSite::GreatToe => 1.0,
            VibrationSite::MedialMalleolus => 0.85,
            VibrationSite::TibialTuberosity => 0.70,
            VibrationSite::Asis => 0.55,
            VibrationSite::UlnarStyloid => 0.90,
            VibrationSite::IndexFinger => 0.95,
        }
    }

    /// Check if this is a lower limb site
    pub fn is_lower_limb(&self) -> bool {
        matches!(
            self,
            VibrationSite::GreatToe
                | VibrationSite::MedialMalleolus
                | VibrationSite::TibialTuberosity
                | VibrationSite::Asis
        )
    }
}

/// Neuropathy risk classification based on VPT
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuropathyRisk {
    /// Normal vibration sense
    Normal,
    /// Mildly reduced - monitor
    MildlyReduced,
    /// Moderately reduced - clinical concern
    ModeratelyReduced,
    /// Severely reduced - high neuropathy risk
    SeverelyReduced,
    /// Absent vibration sense
    Absent,
}

impl NeuropathyRisk {
    /// Get risk level as numeric score (0-4)
    pub fn score(&self) -> u8 {
        match self {
            NeuropathyRisk::Normal => 0,
            NeuropathyRisk::MildlyReduced => 1,
            NeuropathyRisk::ModeratelyReduced => 2,
            NeuropathyRisk::SeverelyReduced => 3,
            NeuropathyRisk::Absent => 4,
        }
    }

    /// Clinical recommendation based on risk level
    pub fn recommendation(&self) -> &'static str {
        match self {
            NeuropathyRisk::Normal => "Continue routine monitoring",
            NeuropathyRisk::MildlyReduced => "Retest in 3-6 months, optimize glucose control",
            NeuropathyRisk::ModeratelyReduced => {
                "Refer to neurology, initiate foot protection program"
            }
            NeuropathyRisk::SeverelyReduced => {
                "High fall risk, comprehensive foot care, protective footwear"
            }
            NeuropathyRisk::Absent => {
                "Very high ulceration risk, intensive foot surveillance required"
            }
        }
    }
}

/// Vibration sense analyzer
#[derive(Debug, Clone)]
pub struct VibrationSense {
    /// Testing site
    pub site: VibrationSite,
    /// Vibration frequency (Hz) - typically 128 Hz for tuning fork
    pub frequency: f64,
}

impl VibrationSense {
    /// Create new VibrationSense analyzer
    pub fn new(site: VibrationSite, frequency: f64) -> Self {
        Self { site, frequency }
    }

    /// Create analyzer for standard 128 Hz tuning fork
    pub fn tuning_fork_128hz(site: VibrationSite) -> Self {
        Self::new(site, 128.0)
    }

    /// Calculate vibration perception threshold from method of limits
    /// Uses ascending/descending trials to find threshold
    pub fn perception_threshold(&self, ascending_trials: &[f64], descending_trials: &[f64]) -> f64 {
        if ascending_trials.is_empty() && descending_trials.is_empty() {
            return 0.0;
        }

        let asc_mean = if ascending_trials.is_empty() {
            0.0
        } else {
            ascending_trials.iter().sum::<f64>() / ascending_trials.len() as f64
        };

        let desc_mean = if descending_trials.is_empty() {
            0.0
        } else {
            descending_trials.iter().sum::<f64>() / descending_trials.len() as f64
        };

        if ascending_trials.is_empty() {
            desc_mean
        } else if descending_trials.is_empty() {
            asc_mean
        } else {
            (asc_mean + desc_mean) / 2.0
        }
    }

    /// Calculate threshold using adaptive staircase method (e.g., PEST)
    pub fn adaptive_threshold(&self, reversals: &[f64], n_final: usize) -> f64 {
        if reversals.is_empty() {
            return 0.0;
        }

        let n_use = n_final.min(reversals.len());
        let start_idx = reversals.len() - n_use;

        reversals[start_idx..].iter().sum::<f64>() / n_use as f64
    }

    /// Get age-adjusted normative percentile for VPT
    /// Based on published normative data for 128 Hz vibration at great toe
    pub fn age_percentile(&self, threshold_volts: f64, age: u8) -> f64 {
        // Normative data: VPT increases with age
        // Values are approximate means and SDs for biothesiometer readings
        let (mean, sd) = self.normative_data(age);

        // Adjust for site-specific gradient
        let adjusted_threshold = threshold_volts / self.site.gradient_factor();

        let z = (adjusted_threshold - mean) / sd;
        normal_cdf(z) * 100.0
    }

    /// Get normative mean and SD for age group
    fn normative_data(&self, age: u8) -> (f64, f64) {
        // Biothesiometer normative data (volts) at great toe
        match age {
            0..=30 => (4.0, 2.0),
            31..=40 => (6.0, 3.0),
            41..=50 => (8.0, 4.0),
            51..=60 => (12.0, 5.0),
            61..=70 => (16.0, 6.0),
            _ => (20.0, 7.0),
        }
    }

    /// Screen for diabetic peripheral neuropathy
    /// Uses >25V threshold at great toe as standard cutoff
    pub fn neuropathy_screening(&self, threshold_volts: f64, age: u8) -> NeuropathyRisk {
        // Adjust threshold for testing site
        let adjusted = threshold_volts / self.site.gradient_factor();

        // Age-adjusted cutoffs based on clinical guidelines
        let (mild_cutoff, moderate_cutoff, severe_cutoff) = match age {
            0..=40 => (10.0, 15.0, 25.0),
            41..=60 => (15.0, 20.0, 30.0),
            _ => (20.0, 25.0, 35.0),
        };

        if adjusted < mild_cutoff {
            NeuropathyRisk::Normal
        } else if adjusted < moderate_cutoff {
            NeuropathyRisk::MildlyReduced
        } else if adjusted < severe_cutoff {
            NeuropathyRisk::ModeratelyReduced
        } else if adjusted < 50.0 {
            NeuropathyRisk::SeverelyReduced
        } else {
            NeuropathyRisk::Absent
        }
    }

    /// Calculate gradient ratio between two sites
    /// Abnormal gradient suggests length-dependent neuropathy
    pub fn gradient_ratio(distal_vpt: f64, proximal_vpt: f64) -> f64 {
        if proximal_vpt < 0.001 {
            return 0.0;
        }
        distal_vpt / proximal_vpt
    }

    /// Check if gradient is abnormal (suggests peripheral neuropathy)
    /// Normal gradient: distal VPT 1.0-1.5x proximal
    pub fn is_gradient_abnormal(distal_vpt: f64, proximal_vpt: f64) -> bool {
        let ratio = Self::gradient_ratio(distal_vpt, proximal_vpt);
        ratio > 2.0 || ratio < 0.8
    }

    /// Analyze multiple VPT trials
    pub fn analyze_trials(&self, trials: &[VptTrial]) -> VptMetrics {
        if trials.is_empty() {
            return VptMetrics::default();
        }

        let thresholds: Vec<f64> = trials.iter().map(|t| t.threshold).collect();

        let mean_threshold = thresholds.iter().sum::<f64>() / thresholds.len() as f64;

        let variance: f64 = thresholds
            .iter()
            .map(|t| (t - mean_threshold).powi(2))
            .sum::<f64>()
            / thresholds.len() as f64;
        let threshold_variability = variance.sqrt();

        // Calculate coefficient of variation
        let cv = if mean_threshold > 0.001 {
            threshold_variability / mean_threshold * 100.0
        } else {
            0.0
        };

        VptMetrics {
            mean_threshold,
            threshold_variability,
            coefficient_of_variation: cv,
            n_trials: trials.len(),
        }
    }

    /// Estimate time since vibration onset for tuning fork
    /// Useful for calculating duration of perception
    pub fn tuning_fork_decay(&self, initial_amplitude: f64, decay_time: f64) -> f64 {
        // Tuning fork amplitude decays exponentially
        // Time constant ~15-20 seconds for 128 Hz fork
        let tau = 17.5; // Time constant in seconds
        initial_amplitude * (-decay_time / tau).exp()
    }

    /// Convert tuning fork "on time" to threshold estimate
    /// Based on decay characteristics of standard tuning fork
    pub fn on_time_to_threshold(&self, on_time_seconds: f64, max_on_time: f64) -> f64 {
        // Longer on-time = lower threshold (better sensation)
        // Normalize to 0-50V scale typical of biothesiometer
        if max_on_time < 0.001 {
            return 50.0;
        }

        let ratio = on_time_seconds / max_on_time;
        50.0 * (1.0 - ratio)
    }
}

/// Result from a single VPT trial
#[derive(Debug, Clone)]
pub struct VptTrial {
    /// Threshold in volts (biothesiometer) or arbitrary units
    pub threshold: f64,
    /// Site tested
    pub site: VibrationSite,
    /// Method used (ascending, descending, adaptive)
    pub method: VptMethod,
}

/// VPT testing method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VptMethod {
    /// Ascending method of limits
    Ascending,
    /// Descending method of limits
    Descending,
    /// Combined ascending/descending
    Combined,
    /// Adaptive staircase (PEST, etc.)
    Adaptive,
    /// Tuning fork on-time
    TuningFork,
}

/// Aggregated VPT metrics
#[derive(Debug, Clone, Default)]
pub struct VptMetrics {
    /// Mean threshold across trials
    pub mean_threshold: f64,
    /// Standard deviation of thresholds
    pub threshold_variability: f64,
    /// Coefficient of variation (%)
    pub coefficient_of_variation: f64,
    /// Number of trials
    pub n_trials: usize,
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
    fn test_perception_threshold() {
        let vs = VibrationSense::tuning_fork_128hz(VibrationSite::GreatToe);

        let asc = vec![5.0, 6.0, 5.5];
        let desc = vec![7.0, 6.5, 7.0];

        let threshold = vs.perception_threshold(&asc, &desc);
        // Mean of ascending (5.5) and descending (6.83) = ~6.17
        assert!(threshold > 5.0 && threshold < 7.0);
    }

    #[test]
    fn test_neuropathy_screening() {
        let vs = VibrationSense::tuning_fork_128hz(VibrationSite::GreatToe);

        // Normal young person
        assert_eq!(vs.neuropathy_screening(5.0, 25), NeuropathyRisk::Normal);

        // High threshold = neuropathy risk
        assert_eq!(
            vs.neuropathy_screening(35.0, 50),
            NeuropathyRisk::SeverelyReduced
        );
    }

    #[test]
    fn test_gradient_ratio() {
        // Normal gradient: distal slightly higher than proximal
        assert!(!VibrationSense::is_gradient_abnormal(12.0, 10.0));

        // Abnormal gradient: distal much higher (neuropathy pattern)
        assert!(VibrationSense::is_gradient_abnormal(30.0, 10.0));
    }

    #[test]
    fn test_site_gradient() {
        assert!(
            VibrationSite::GreatToe.gradient_factor()
                > VibrationSite::TibialTuberosity.gradient_factor()
        );
    }
}
