//! Pain and Sensory Signal Generators
//!
//! Generates pain-related physiological signals:
//! - Pain ratings and temporal summation
//! - Thermal detection thresholds
//! - Pressure pain thresholds
//! - Conditioned pain modulation
//! - Quantitative sensory testing (QST) profiles
//! - Pathological patterns (hyperalgesia, allodynia)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Configuration for pain signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainConfig {
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Base pain threshold (0-10 scale)
    pub base_threshold: f64,
    /// Pain sensitivity factor
    pub sensitivity: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for PainConfig {
    fn default() -> Self {
        Self {
            sample_rate: 10.0, // Low rate for pain ratings
            base_threshold: 5.0,
            sensitivity: 1.0,
            noise_level: 0.1,
            seed: None,
        }
    }
}

/// Output from pain signal generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainOutput {
    /// Time points (s)
    pub time: Vec<f64>,
    /// Pain rating (0-10 VAS)
    pub pain_rating: Vec<f64>,
    /// Stimulus intensity
    pub stimulus: Vec<f64>,
    /// Ground truth
    pub ground_truth: PainGroundTruth,
    /// Configuration
    pub config: PainConfig,
}

/// Ground truth for pain signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PainGroundTruth {
    /// Pain threshold
    pub threshold: f64,
    /// Pain tolerance
    pub tolerance: f64,
    /// Temporal summation ratio
    pub temporal_summation: f64,
    /// Wind-up present
    pub wind_up: bool,
    /// CPM effect (% reduction)
    pub cpm_effect: f64,
    /// Applied pathology
    pub pathology: Option<PainPathology>,
}

/// Pathological pain patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PainPathology {
    /// Primary hyperalgesia (increased sensitivity at injury site)
    PrimaryHyperalgesia { threshold_reduction: f64 },
    /// Secondary hyperalgesia (spread of sensitivity)
    SecondaryHyperalgesia { spread: f64, threshold_reduction: f64 },
    /// Allodynia (pain from non-painful stimuli)
    Allodynia { threshold: f64 },
    /// Central sensitization
    CentralSensitization { gain: f64, temporal_summation: f64 },
    /// Reduced CPM (chronic pain)
    ReducedCPM { cpm_reduction: f64 },
    /// Fibromyalgia profile
    Fibromyalgia,
    /// Neuropathic pain
    Neuropathic { abnormal_sensations: bool },
}

/// QST modality types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QstModality {
    /// Cold detection threshold
    ColdDetection,
    /// Warm detection threshold
    WarmDetection,
    /// Cold pain threshold
    ColdPain,
    /// Heat pain threshold
    HeatPain,
    /// Pressure pain threshold
    PressurePain,
    /// Mechanical detection
    MechanicalDetection,
    /// Vibration detection
    VibrationDetection,
    /// Pinprick pain
    PinprickPain,
}

/// QST output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QstOutput {
    /// Modality tested
    pub modality: QstModality,
    /// Threshold value
    pub threshold: f64,
    /// Z-score compared to norms
    pub z_score: f64,
    /// Classification
    pub classification: QstClassification,
}

/// QST classification
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum QstClassification {
    Normal,
    LossOfFunction,
    GainOfFunction,
}

/// Pain signal generator
pub struct PainGenerator {
    config: PainConfig,
    rng: StdRng,
}

impl PainGenerator {
    /// Create new pain generator
    pub fn new(config: PainConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Generate pain rating response to ramping stimulus
    pub fn generate_pain_ramp(&mut self, duration: f64, max_stimulus: f64) -> PainOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut stimulus = Vec::with_capacity(n_samples);
        let mut pain_rating = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level).unwrap();

        let threshold = self.config.base_threshold;
        let mut reached_threshold = false;
        let mut threshold_stimulus = 0.0;
        let mut tolerance_stimulus = max_stimulus;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Linear ramp
            let stim = max_stimulus * (t / duration);
            stimulus.push(stim);

            // Pain response (sigmoid function above threshold)
            let pain = if stim < threshold {
                0.0
            } else {
                if !reached_threshold {
                    reached_threshold = true;
                    threshold_stimulus = stim;
                }
                let above_threshold = stim - threshold;
                10.0 * (1.0 - (-above_threshold * self.config.sensitivity * 0.5).exp())
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let final_pain = (pain + noise).clamp(0.0, 10.0);
            pain_rating.push(final_pain);

            // Track tolerance (pain = 10)
            if final_pain >= 9.5 && tolerance_stimulus == max_stimulus {
                tolerance_stimulus = stim;
            }
        }

        let ground_truth = PainGroundTruth {
            threshold: threshold_stimulus,
            tolerance: tolerance_stimulus,
            temporal_summation: 1.0,
            wind_up: false,
            cpm_effect: 0.0,
            pathology: None,
        };

        PainOutput {
            time,
            pain_rating,
            stimulus,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate temporal summation protocol
    pub fn generate_temporal_summation(
        &mut self,
        n_stimuli: usize,
        isi: f64, // Inter-stimulus interval
        stimulus_intensity: f64,
    ) -> PainOutput {
        let duration = n_stimuli as f64 * isi + 2.0;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut stimulus = Vec::with_capacity(n_samples);
        let mut pain_rating = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level).unwrap();

        // Temporal summation increases pain over repeated stimuli
        let summation_factor = 0.15; // 15% increase per stimulus
        let mut cumulative_summation = 1.0;
        let mut last_stim_time = -isi;
        let mut first_pain = 0.0;
        let mut last_pain = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Pulsed stimuli
            let stim_num = ((t - 1.0) / isi).floor() as i32;
            let in_stimulus = stim_num >= 0
                && stim_num < n_stimuli as i32
                && (t - 1.0 - stim_num as f64 * isi) < 0.1;

            let stim = if in_stimulus { stimulus_intensity } else { 0.0 };
            stimulus.push(stim);

            if in_stimulus && t > last_stim_time + isi * 0.5 {
                cumulative_summation += summation_factor;
                last_stim_time = t;
            }

            let pain = if stim > 0.0 {
                let base_pain = (stim - self.config.base_threshold).max(0.0) * self.config.sensitivity;
                (base_pain * cumulative_summation).min(10.0)
            } else {
                // Decay between stimuli
                pain_rating.last().unwrap_or(&0.0) * 0.95
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let final_pain = (pain + noise).clamp(0.0, 10.0);

            if first_pain == 0.0 && final_pain > 0.1 {
                first_pain = final_pain;
            }
            if final_pain > 0.1 {
                last_pain = final_pain;
            }

            pain_rating.push(final_pain);
        }

        let temporal_summation = if first_pain > 0.0 {
            last_pain / first_pain
        } else {
            1.0
        };

        let ground_truth = PainGroundTruth {
            threshold: self.config.base_threshold,
            tolerance: 10.0,
            temporal_summation,
            wind_up: temporal_summation > 1.5,
            cpm_effect: 0.0,
            pathology: None,
        };

        PainOutput {
            time,
            pain_rating,
            stimulus,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate conditioned pain modulation (CPM) protocol
    pub fn generate_cpm(&mut self, conditioning_intensity: f64) -> (PainOutput, PainOutput) {
        // Baseline pain test
        let baseline = self.generate_pain_ramp(30.0, 10.0);

        // Test during conditioning (should be reduced)
        let cpm_effect = 0.3 * conditioning_intensity / 10.0; // 30% max reduction
        let original_threshold = self.config.base_threshold;
        self.config.base_threshold = original_threshold * (1.0 + cpm_effect);

        let mut conditioned = self.generate_pain_ramp(30.0, 10.0);
        conditioned.ground_truth.cpm_effect = cpm_effect * 100.0;

        self.config.base_threshold = original_threshold;

        (baseline, conditioned)
    }

    /// Generate QST battery
    pub fn generate_qst_battery(&mut self) -> Vec<QstOutput> {
        let modalities = [
            (QstModality::ColdDetection, 30.0, 2.0),      // °C, SD
            (QstModality::WarmDetection, 34.0, 1.5),
            (QstModality::ColdPain, 10.0, 5.0),
            (QstModality::HeatPain, 45.0, 3.0),
            (QstModality::PressurePain, 400.0, 100.0),    // kPa
            (QstModality::MechanicalDetection, 0.5, 0.3), // mN
            (QstModality::VibrationDetection, 0.5, 0.2),  // arbitrary
            (QstModality::PinprickPain, 3.0, 1.0),        // VAS rating
        ];

        let noise_dist = Normal::new(0.0, 1.0).unwrap();

        modalities.iter().map(|&(modality, mean, sd)| {
            let noise: f64 = self.rng.sample(noise_dist);
            let threshold = mean + noise * sd * 0.5 * self.config.sensitivity;
            let z_score = (threshold - mean) / sd;

            let classification = if z_score < -1.96 {
                QstClassification::LossOfFunction
            } else if z_score > 1.96 {
                QstClassification::GainOfFunction
            } else {
                QstClassification::Normal
            };

            QstOutput {
                modality,
                threshold,
                z_score,
                classification,
            }
        }).collect()
    }

    /// Generate pathological pain pattern
    pub fn generate_pathological(
        &mut self,
        pathology: PainPathology,
        duration: f64,
    ) -> PainOutput {
        match pathology {
            PainPathology::PrimaryHyperalgesia { threshold_reduction } => {
                let original = self.config.base_threshold;
                self.config.base_threshold = original * (1.0 - threshold_reduction);
                let mut output = self.generate_pain_ramp(duration, 10.0);
                output.ground_truth.pathology = Some(pathology);
                self.config.base_threshold = original;
                output
            }
            PainPathology::CentralSensitization { gain, temporal_summation: ts } => {
                let original_sens = self.config.sensitivity;
                self.config.sensitivity = original_sens * (1.0 + gain);
                let mut output = self.generate_temporal_summation(10, 1.0, 7.0);
                output.ground_truth.pathology = Some(pathology);
                output.ground_truth.temporal_summation *= 1.0 + ts;
                self.config.sensitivity = original_sens;
                output
            }
            PainPathology::Allodynia { threshold } => {
                let original = self.config.base_threshold;
                self.config.base_threshold = threshold;
                let mut output = self.generate_pain_ramp(duration, 5.0);
                output.ground_truth.pathology = Some(pathology);
                self.config.base_threshold = original;
                output
            }
            _ => {
                let mut output = self.generate_pain_ramp(duration, 10.0);
                output.ground_truth.pathology = Some(pathology);
                output
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pain_ramp() {
        let config = PainConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = PainGenerator::new(config);
        let output = gen.generate_pain_ramp(30.0, 10.0);

        assert!(!output.pain_rating.is_empty());
        assert!(output.ground_truth.threshold > 0.0);
    }

    #[test]
    fn test_temporal_summation() {
        let config = PainConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = PainGenerator::new(config);
        let output = gen.generate_temporal_summation(10, 1.0, 7.0);

        // Should show summation
        assert!(output.ground_truth.temporal_summation >= 1.0);
    }

    #[test]
    fn test_cpm() {
        let config = PainConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = PainGenerator::new(config);
        let (baseline, conditioned) = gen.generate_cpm(8.0);

        // Conditioned should have higher threshold (less pain)
        assert!(conditioned.ground_truth.cpm_effect > 0.0);
    }

    #[test]
    fn test_qst_battery() {
        let config = PainConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = PainGenerator::new(config);
        let qst = gen.generate_qst_battery();

        assert_eq!(qst.len(), 8);
    }

    #[test]
    fn test_hyperalgesia() {
        let config = PainConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = PainGenerator::new(config);

        let normal = gen.generate_pain_ramp(30.0, 10.0);
        let hyperalgesia = gen.generate_pathological(
            PainPathology::PrimaryHyperalgesia { threshold_reduction: 0.3 },
            30.0,
        );

        // Hyperalgesia should have lower threshold
        assert!(hyperalgesia.ground_truth.threshold < normal.ground_truth.threshold);
    }
}
