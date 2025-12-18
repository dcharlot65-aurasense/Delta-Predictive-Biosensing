//! Quantitative Sensory Testing (QST)
//!
//! Implements the German Research Network on Neuropathic Pain (DFNS) QST protocol
//! and related sensory testing algorithms.

/// Sensory modality being tested
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModalityType {
    /// Cold detection threshold
    ColdDetection,
    /// Warm detection threshold
    WarmDetection,
    /// Cold pain threshold
    ColdPain,
    /// Heat pain threshold
    HeatPain,
    /// Mechanical detection threshold (von Frey)
    MechanicalDetection,
    /// Mechanical pain threshold
    MechanicalPain,
    /// Mechanical pain sensitivity (stimulus-response)
    MechanicalPainSensitivity,
    /// Pressure pain threshold (algometry)
    PressurePain,
    /// Vibration detection threshold
    VibrationDetection,
    /// Wind-up ratio (temporal summation)
    WindUpRatio,
    /// Dynamic mechanical allodynia
    DynamicAllodynia,
}

impl ModalityType {
    /// Get the unit for this modality
    pub fn unit(&self) -> &'static str {
        match self {
            ModalityType::ColdDetection
            | ModalityType::WarmDetection
            | ModalityType::ColdPain
            | ModalityType::HeatPain => "°C",
            ModalityType::MechanicalDetection => "mN",
            ModalityType::MechanicalPain | ModalityType::MechanicalPainSensitivity => "mN",
            ModalityType::PressurePain => "kPa",
            ModalityType::VibrationDetection => "µm",
            ModalityType::WindUpRatio => "ratio",
            ModalityType::DynamicAllodynia => "NRS",
        }
    }

    /// Check if this modality tests small fiber function
    pub fn is_small_fiber(&self) -> bool {
        matches!(
            self,
            ModalityType::ColdDetection
                | ModalityType::WarmDetection
                | ModalityType::ColdPain
                | ModalityType::HeatPain
                | ModalityType::MechanicalPain
        )
    }

    /// Check if this modality tests large fiber function
    pub fn is_large_fiber(&self) -> bool {
        matches!(
            self,
            ModalityType::MechanicalDetection | ModalityType::VibrationDetection
        )
    }
}

/// Sensory phenotype classification based on QST profile
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensoryPhenotype {
    /// Loss of function (hypoesthesia/hypoalgesia)
    SensoryLoss,
    /// Thermal hyperalgesia
    ThermalHyperalgesia,
    /// Mechanical hyperalgesia
    MechanicalHyperalgesia,
    /// Mixed pattern
    Mixed,
    /// Normal sensory function
    Normal,
}

/// QST analyzer for pain assessment
#[derive(Debug, Clone)]
pub struct QstAnalyzer {
    /// Reference body site for normalization
    pub reference_site: String,
    /// Baseline temperature for thermal testing (°C)
    pub baseline_temp: f64,
}

impl Default for QstAnalyzer {
    fn default() -> Self {
        Self {
            reference_site: "thenar_eminence".to_string(),
            baseline_temp: 32.0,
        }
    }
}

impl QstAnalyzer {
    /// Create new QST analyzer
    pub fn new(reference_site: &str, baseline_temp: f64) -> Self {
        Self {
            reference_site: reference_site.to_string(),
            baseline_temp,
        }
    }

    /// Calculate cold detection threshold (CDT)
    /// Returns temperature difference from baseline
    pub fn cold_detection_threshold(&self, detection_temps: &[f64]) -> f64 {
        if detection_temps.is_empty() {
            return 0.0;
        }

        let mean_temp = detection_temps.iter().sum::<f64>() / detection_temps.len() as f64;
        self.baseline_temp - mean_temp
    }

    /// Calculate warm detection threshold (WDT)
    /// Returns temperature difference from baseline
    pub fn warm_detection_threshold(&self, detection_temps: &[f64]) -> f64 {
        if detection_temps.is_empty() {
            return 0.0;
        }

        let mean_temp = detection_temps.iter().sum::<f64>() / detection_temps.len() as f64;
        mean_temp - self.baseline_temp
    }

    /// Calculate thermal sensory limen (TSL)
    /// Measures ability to detect alternating warm/cold stimuli
    pub fn thermal_sensory_limen(&self, alternations: &[(f64, f64)]) -> f64 {
        if alternations.is_empty() {
            return 0.0;
        }

        let total: f64 = alternations
            .iter()
            .map(|(cold, warm)| (self.baseline_temp - cold) + (warm - self.baseline_temp))
            .sum();

        total / alternations.len() as f64
    }

    /// Calculate cold pain threshold (CPT)
    pub fn cold_pain_threshold(&self, pain_temps: &[f64]) -> f64 {
        if pain_temps.is_empty() {
            return 0.0; // No pain detected
        }

        pain_temps.iter().sum::<f64>() / pain_temps.len() as f64
    }

    /// Calculate heat pain threshold (HPT)
    pub fn heat_pain_threshold(&self, pain_temps: &[f64]) -> f64 {
        if pain_temps.is_empty() {
            return 50.0; // Max temp reached without pain
        }

        pain_temps.iter().sum::<f64>() / pain_temps.len() as f64
    }

    /// Calculate mechanical detection threshold (MDT)
    /// Using von Frey filaments (forces in mN)
    pub fn mechanical_detection_threshold(&self, detection_forces: &[f64]) -> f64 {
        if detection_forces.is_empty() {
            return 0.0;
        }

        // Geometric mean is standard for von Frey thresholds
        let log_sum: f64 = detection_forces.iter().map(|f| f.ln()).sum();
        (log_sum / detection_forces.len() as f64).exp()
    }

    /// Calculate mechanical pain threshold (MPT)
    /// Using pinprick stimulators
    pub fn mechanical_pain_threshold(&self, pain_forces: &[f64]) -> f64 {
        if pain_forces.is_empty() {
            return 512.0; // Max force without pain
        }

        // Geometric mean
        let log_sum: f64 = pain_forces.iter().map(|f| f.ln()).sum();
        (log_sum / pain_forces.len() as f64).exp()
    }

    /// Calculate mechanical pain sensitivity (MPS)
    /// Stimulus-response function for pinprick
    pub fn mechanical_pain_sensitivity(&self, forces: &[f64], ratings: &[f64]) -> f64 {
        if forces.len() != ratings.len() || forces.is_empty() {
            return 0.0;
        }

        // Mean rating across all intensities
        ratings.iter().sum::<f64>() / ratings.len() as f64
    }

    /// Calculate wind-up ratio (WUR)
    /// Ratio of pain rating for repeated vs single stimulus
    pub fn wind_up_ratio(&self, single_ratings: &[f64], repeated_ratings: &[f64]) -> f64 {
        if single_ratings.is_empty() || repeated_ratings.is_empty() {
            return 1.0;
        }

        let single_mean = single_ratings.iter().sum::<f64>() / single_ratings.len() as f64;
        let repeated_mean = repeated_ratings.iter().sum::<f64>() / repeated_ratings.len() as f64;

        if single_mean < 0.1 {
            return 1.0;
        }

        repeated_mean / single_mean
    }

    /// Calculate pressure pain threshold (PPT) using algometry
    pub fn pressure_pain_threshold(&self, thresholds_kpa: &[f64]) -> f64 {
        if thresholds_kpa.is_empty() {
            return 0.0;
        }

        thresholds_kpa.iter().sum::<f64>() / thresholds_kpa.len() as f64
    }

    /// Calculate dynamic mechanical allodynia (DMA) score
    pub fn dynamic_allodynia(&self, brush_ratings: &[f64]) -> f64 {
        if brush_ratings.is_empty() {
            return 0.0;
        }

        // Mean pain rating from light brush stimuli
        ratings_mean(brush_ratings)
    }

    /// Convert raw QST value to Z-score using DFNS normative data
    pub fn to_z_score(&self, modality: ModalityType, value: f64, age: u8, site: &str) -> f64 {
        let (mean, sd) = self.normative_data(modality, age, site);
        (value - mean) / sd
    }

    /// Get normative mean and SD for modality
    fn normative_data(&self, modality: ModalityType, age: u8, _site: &str) -> (f64, f64) {
        // Simplified normative data based on DFNS reference values
        // In production, this would use full site-specific normative tables
        let age_factor = 1.0 + (age as f64 - 40.0).abs() * 0.01;

        match modality {
            ModalityType::ColdDetection => (1.5 * age_factor, 0.8),
            ModalityType::WarmDetection => (3.5 * age_factor, 1.5),
            ModalityType::ColdPain => (15.0, 8.0),
            ModalityType::HeatPain => (42.0, 3.0),
            ModalityType::MechanicalDetection => (0.5 * age_factor, 0.3),
            ModalityType::MechanicalPain => (50.0, 30.0),
            ModalityType::MechanicalPainSensitivity => (1.0, 1.0),
            ModalityType::PressurePain => (350.0, 100.0),
            ModalityType::VibrationDetection => (0.5 * age_factor, 0.3),
            ModalityType::WindUpRatio => (2.5, 1.0),
            ModalityType::DynamicAllodynia => (0.0, 0.1),
        }
    }

    /// Classify sensory phenotype from QST profile
    pub fn classify_phenotype(&self, results: &[QstResult]) -> SensoryPhenotype {
        let mut loss_count = 0;
        let mut thermal_hyper = 0;
        let mut mech_hyper = 0;

        for result in results {
            if result.z_score < -1.96 {
                // Loss of function
                if result.modality.is_small_fiber() || result.modality.is_large_fiber() {
                    loss_count += 1;
                }
            } else if result.z_score > 1.96 {
                // Gain of function
                match result.modality {
                    ModalityType::ColdPain | ModalityType::HeatPain => thermal_hyper += 1,
                    ModalityType::MechanicalPain
                    | ModalityType::MechanicalPainSensitivity
                    | ModalityType::PressurePain => mech_hyper += 1,
                    _ => {}
                }
            }
        }

        if loss_count >= 2 && thermal_hyper == 0 && mech_hyper == 0 {
            SensoryPhenotype::SensoryLoss
        } else if thermal_hyper >= 1 && mech_hyper == 0 {
            SensoryPhenotype::ThermalHyperalgesia
        } else if mech_hyper >= 1 && thermal_hyper == 0 {
            SensoryPhenotype::MechanicalHyperalgesia
        } else if (thermal_hyper + mech_hyper) >= 1 || loss_count >= 1 {
            SensoryPhenotype::Mixed
        } else {
            SensoryPhenotype::Normal
        }
    }

    /// Generate comprehensive QST metrics
    pub fn analyze(&self, results: &[QstResult]) -> QstMetrics {
        let phenotype = self.classify_phenotype(results);

        let loss_of_function: Vec<&QstResult> =
            results.iter().filter(|r| r.z_score < -1.96).collect();
        let gain_of_function: Vec<&QstResult> =
            results.iter().filter(|r| r.z_score > 1.96).collect();

        QstMetrics {
            n_modalities_tested: results.len(),
            n_abnormal: loss_of_function.len() + gain_of_function.len(),
            n_loss_of_function: loss_of_function.len(),
            n_gain_of_function: gain_of_function.len(),
            phenotype,
        }
    }
}

/// Result from a single QST modality test
#[derive(Debug, Clone)]
pub struct QstResult {
    /// Modality tested
    pub modality: ModalityType,
    /// Raw threshold/value
    pub raw_value: f64,
    /// Z-score relative to normative data
    pub z_score: f64,
    /// Test site
    pub site: String,
}

/// Aggregated QST metrics
#[derive(Debug, Clone)]
pub struct QstMetrics {
    /// Number of modalities tested
    pub n_modalities_tested: usize,
    /// Number of abnormal findings
    pub n_abnormal: usize,
    /// Number showing loss of function
    pub n_loss_of_function: usize,
    /// Number showing gain of function
    pub n_gain_of_function: usize,
    /// Overall sensory phenotype
    pub phenotype: SensoryPhenotype,
}

/// Temporal summation analyzer
#[derive(Debug, Clone)]
pub struct TemporalSummation {
    /// Stimulus frequency (Hz)
    pub frequency: f64,
    /// Number of stimuli in train
    pub n_stimuli: usize,
}

impl TemporalSummation {
    /// Create new temporal summation analyzer
    pub fn new(frequency: f64, n_stimuli: usize) -> Self {
        Self {
            frequency,
            n_stimuli,
        }
    }

    /// Calculate temporal summation index
    /// Compares final pain rating to initial rating
    pub fn summation_index(&self, pain_ratings: &[f64]) -> f64 {
        if pain_ratings.len() < 2 {
            return 1.0;
        }

        let initial = pain_ratings[0];
        let final_rating = pain_ratings[pain_ratings.len() - 1];

        if initial < 0.1 {
            return 1.0;
        }

        final_rating / initial
    }

    /// Calculate area under the pain curve
    pub fn pain_auc(&self, pain_ratings: &[f64]) -> f64 {
        if pain_ratings.len() < 2 {
            return 0.0;
        }

        let dt = 1.0 / self.frequency;
        let mut auc = 0.0;

        for i in 1..pain_ratings.len() {
            // Trapezoidal rule
            auc += (pain_ratings[i - 1] + pain_ratings[i]) * dt / 2.0;
        }

        auc
    }

    /// Detect wind-up (progressive increase in pain)
    pub fn detect_wind_up(&self, pain_ratings: &[f64]) -> bool {
        if pain_ratings.len() < 3 {
            return false;
        }

        // Check for consistent increase
        let mut increases = 0;
        for i in 1..pain_ratings.len() {
            if pain_ratings[i] > pain_ratings[i - 1] + 0.5 {
                increases += 1;
            }
        }

        // Wind-up if more than half of transitions show increase
        increases > pain_ratings.len() / 2
    }

    /// Calculate decay rate after stimulus train
    pub fn decay_rate(&self, post_stimulus_ratings: &[f64], time_points: &[f64]) -> f64 {
        if post_stimulus_ratings.len() < 2 || time_points.len() != post_stimulus_ratings.len() {
            return 0.0;
        }

        // Fit exponential decay: rating = A * exp(-t/tau)
        // Use log-linear regression
        let valid: Vec<(f64, f64)> = time_points
            .iter()
            .zip(post_stimulus_ratings.iter())
            .filter(|&(_, r)| *r > 0.1)
            .map(|(&t, &r)| (t, r.ln()))
            .collect();

        if valid.len() < 2 {
            return 0.0;
        }

        // Simple linear regression on log values
        let n = valid.len() as f64;
        let sum_x: f64 = valid.iter().map(|(t, _)| t).sum();
        let sum_y: f64 = valid.iter().map(|(_, lr)| lr).sum();
        let sum_xy: f64 = valid.iter().map(|(t, lr)| t * lr).sum();
        let sum_xx: f64 = valid.iter().map(|(t, _)| t * t).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);

        // Decay rate is negative of slope
        -slope
    }
}

fn ratings_mean(ratings: &[f64]) -> f64 {
    if ratings.is_empty() {
        return 0.0;
    }
    ratings.iter().sum::<f64>() / ratings.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cold_detection() {
        let qst = QstAnalyzer::default();

        let temps = vec![30.5, 30.0, 30.2];
        let cdt = qst.cold_detection_threshold(&temps);

        // Baseline 32°C - mean ~30.23 = ~1.77°C
        assert!(cdt > 1.5 && cdt < 2.0);
    }

    #[test]
    fn test_wind_up_ratio() {
        let qst = QstAnalyzer::default();

        let single = vec![3.0, 3.5, 3.0];
        let repeated = vec![6.0, 7.0, 6.5];

        let wur = qst.wind_up_ratio(&single, &repeated);
        // Repeated (~6.5) / Single (~3.17) = ~2.05
        assert!(wur > 1.5 && wur < 2.5);
    }

    #[test]
    fn test_temporal_summation() {
        let ts = TemporalSummation::new(1.0, 10);

        let ratings = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let si = ts.summation_index(&ratings);

        assert!((si - 3.0).abs() < 0.01);
        assert!(ts.detect_wind_up(&ratings));
    }

    #[test]
    fn test_phenotype_classification() {
        let qst = QstAnalyzer::default();

        // Sensory loss pattern
        let results = vec![
            QstResult {
                modality: ModalityType::ColdDetection,
                raw_value: 5.0,
                z_score: -2.5,
                site: "hand".to_string(),
            },
            QstResult {
                modality: ModalityType::WarmDetection,
                raw_value: 8.0,
                z_score: -2.2,
                site: "hand".to_string(),
            },
        ];

        let phenotype = qst.classify_phenotype(&results);
        assert_eq!(phenotype, SensoryPhenotype::SensoryLoss);
    }
}
