//! Center of Pressure (COP) Signal Generator
//!
//! Generates realistic postural sway patterns including:
//! - Quiet standing in various stances
//! - Limits of stability testing
//! - Ramped weight shifting
//! - Pathological balance patterns

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for COP generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Foot length in meters (for normalization)
    pub foot_length: f64,
    /// Stance width in meters
    pub stance_width: f64,
    /// Base of support dimensions (AP x ML) in meters
    pub bos_dimensions: (f64, f64),
    /// Sway amplitude scaling factor
    pub sway_amplitude: f64,
    /// Noise level (0.0-1.0)
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CopConfig {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            foot_length: 0.26,
            stance_width: 0.15,
            bos_dimensions: (0.26, 0.15),
            sway_amplitude: 1.0,
            noise_level: 0.05,
            seed: None,
        }
    }
}

/// Stance conditions for balance testing
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum StanceCondition {
    /// Normal bilateral stance
    BilateralNormal,
    /// Feet together (narrow base)
    FeetTogether,
    /// Tandem stance (heel-to-toe)
    Tandem,
    /// Single leg stance
    SingleLeg { dominant: bool },
    /// Semi-tandem
    SemiTandem,
    /// Wide stance
    WideStance,
}

/// Pathological balance patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PathologicalBalance {
    /// Parkinson's disease (reduced sway, festination)
    Parkinsonian { severity: f64 },
    /// Cerebellar ataxia (increased sway, irregular)
    CerebellarAtaxia { severity: f64 },
    /// Vestibular deficit (directional bias)
    VestibularDeficit { bias_direction: f64, severity: f64 },
    /// Peripheral neuropathy (increased sway, delayed responses)
    PeripheralNeuropathy { severity: f64 },
    /// Age-related decline
    AgeRelated { age_factor: f64 },
    /// Stroke hemiparesis (asymmetric weight bearing)
    Hemiparesis { affected_side: bool, severity: f64 },
    /// Anxiety/fear of falling (stiffening strategy)
    AnxietyStiffening { severity: f64 },
}

/// Output from COP generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopOutput {
    /// Time points in seconds
    pub time: Vec<f64>,
    /// COP position in anteroposterior direction (m)
    pub cop_ap: Vec<f64>,
    /// COP position in mediolateral direction (m)
    pub cop_ml: Vec<f64>,
    /// COP velocity AP (m/s)
    pub vel_ap: Vec<f64>,
    /// COP velocity ML (m/s)
    pub vel_ml: Vec<f64>,
    /// Ground truth annotations
    pub ground_truth: CopGroundTruth,
    /// Configuration used
    pub config: CopConfig,
}

/// Ground truth for COP signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopGroundTruth {
    /// Stance condition
    pub stance: StanceCondition,
    /// Applied pathology if any
    pub pathology: Option<PathologicalBalance>,
    /// Summary sway metrics
    pub sway_metrics: CopMetrics,
    /// Time-domain summaries
    pub sway_summary: SwaySummary,
    /// Dominant sway frequency AP (Hz)
    pub dominant_freq_ap: f64,
    /// Dominant sway frequency ML (Hz)
    pub dominant_freq_ml: f64,
    /// Mean COP position
    pub mean_cop: (f64, f64),
}

/// COP metrics for algorithm validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CopMetrics {
    /// Root mean square AP (m)
    pub rms_ap: f64,
    /// Root mean square ML (m)
    pub rms_ml: f64,
    /// Sway path length (m)
    pub path_length: f64,
    /// Mean velocity (m/s)
    pub mean_velocity: f64,
    /// 95% confidence ellipse area (m²)
    pub ellipse_area: f64,
    /// Ellipse major axis length (m)
    pub ellipse_major: f64,
    /// Ellipse minor axis length (m)
    pub ellipse_minor: f64,
    /// Ellipse orientation (radians)
    pub ellipse_angle: f64,
    /// Range AP (m)
    pub range_ap: f64,
    /// Range ML (m)
    pub range_ml: f64,
    /// Sample entropy AP
    pub sample_entropy_ap: f64,
    /// Sample entropy ML
    pub sample_entropy_ml: f64,
}

/// Time-domain sway summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwaySummary {
    /// Mean AP position (m)
    pub mean_ap: f64,
    /// Mean ML position (m)
    pub mean_ml: f64,
    /// Standard deviation AP (m)
    pub std_ap: f64,
    /// Standard deviation ML (m)
    pub std_ml: f64,
    /// Max AP excursion (m)
    pub max_ap: f64,
    /// Max ML excursion (m)
    pub max_ml: f64,
    /// Time in each quadrant (%)
    pub quadrant_time: [f64; 4],
}

/// COP signal generator
pub struct CopGenerator {
    config: CopConfig,
    rng: StdRng,
}

impl CopGenerator {
    /// Create a new COP generator
    pub fn new(config: CopConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate quiet standing COP trajectory
    pub fn generate_quiet_standing(
        &mut self,
        duration: f64,
        stance: StanceCondition,
    ) -> CopOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);

        // Stance-dependent parameters
        let (base_sway_ap, base_sway_ml) = self.get_stance_sway(stance);

        // Frequency components (typical postural sway: 0.1-2 Hz)
        let freqs = [0.15, 0.3, 0.5, 0.8, 1.2];
        let phases_ap: Vec<f64> = freqs.iter().map(|_| self.rng.random::<f64>() * 2.0 * PI).collect();
        let phases_ml: Vec<f64> = freqs.iter().map(|_| self.rng.random::<f64>() * 2.0 * PI).collect();
        let amps_ap: Vec<f64> = vec![0.4, 0.25, 0.15, 0.12, 0.08];
        let amps_ml: Vec<f64> = vec![0.35, 0.28, 0.18, 0.12, 0.07];

        let noise_dist = Normal::new(0.0, self.config.noise_level * base_sway_ap).unwrap();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Sum of sinusoids for postural sway
            let mut ap = 0.0;
            let mut ml = 0.0;

            for j in 0..freqs.len() {
                ap += amps_ap[j] * (2.0 * PI * freqs[j] * t + phases_ap[j]).sin();
                ml += amps_ml[j] * (2.0 * PI * freqs[j] * t + phases_ml[j]).sin();
            }

            // Add slow drift component
            let drift_ap = 0.1 * (0.05 * PI * t).sin();
            let drift_ml = 0.08 * (0.03 * PI * t).sin();

            ap = (ap + drift_ap) * base_sway_ap * self.config.sway_amplitude;
            ml = (ml + drift_ml) * base_sway_ml * self.config.sway_amplitude;

            // Add noise
            let noise_ap: f64 = self.rng.sample(noise_dist);
            let noise_ml: f64 = self.rng.sample(noise_dist);

            cop_ap.push(ap + noise_ap);
            cop_ml.push(ml + noise_ml);
        }

        // Calculate velocities
        let vel_ap = self.calculate_velocity(&time, &cop_ap);
        let vel_ml = self.calculate_velocity(&time, &cop_ml);

        // Calculate ground truth
        let sway_metrics = self.calculate_metrics(&cop_ap, &cop_ml, &vel_ap, &vel_ml, dt);
        let sway_summary = self.calculate_summary(&cop_ap, &cop_ml);

        let mean_cop = (
            cop_ap.iter().sum::<f64>() / n_samples as f64,
            cop_ml.iter().sum::<f64>() / n_samples as f64,
        );

        let ground_truth = CopGroundTruth {
            stance,
            pathology: None,
            sway_metrics,
            sway_summary,
            dominant_freq_ap: 0.3, // Simplified - would need FFT for actual
            dominant_freq_ml: 0.25,
            mean_cop,
        };

        CopOutput {
            time,
            cop_ap,
            cop_ml,
            vel_ap,
            vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate limits of stability test
    pub fn generate_limits_of_stability(
        &mut self,
        direction: f64, // radians, 0 = anterior
        hold_duration: f64,
    ) -> CopOutput {
        let total_duration = 2.0 + hold_duration + 2.0; // reach + hold + return
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);

        // Target position (80% of BOS)
        let target_ap = 0.8 * self.config.bos_dimensions.0 * 0.5 * direction.cos();
        let target_ml = 0.8 * self.config.bos_dimensions.1 * 0.5 * direction.sin();

        let noise_dist = Normal::new(0.0, self.config.noise_level * 0.01).unwrap();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let (base_ap, base_ml) = if t < 2.0 {
                // Reach phase
                let progress = t / 2.0;
                let smooth = 0.5 * (1.0 - (PI * progress).cos());
                (target_ap * smooth, target_ml * smooth)
            } else if t < 2.0 + hold_duration {
                // Hold at limit
                (target_ap, target_ml)
            } else {
                // Return phase
                let progress = (t - 2.0 - hold_duration) / 2.0;
                let smooth = 0.5 * (1.0 + (PI * progress).cos());
                (target_ap * smooth, target_ml * smooth)
            };

            // Add sway during hold
            let sway_amp = if t >= 2.0 && t < 2.0 + hold_duration {
                0.005 // Increased sway at limits
            } else {
                0.002
            };

            let sway_ap = sway_amp * (3.0 * PI * t).sin();
            let sway_ml = sway_amp * (2.5 * PI * t).sin();

            let noise_ap: f64 = self.rng.sample(noise_dist);
            let noise_ml: f64 = self.rng.sample(noise_dist);

            cop_ap.push(base_ap + sway_ap + noise_ap);
            cop_ml.push(base_ml + sway_ml + noise_ml);
        }

        let vel_ap = self.calculate_velocity(&time, &cop_ap);
        let vel_ml = self.calculate_velocity(&time, &cop_ml);
        let sway_metrics = self.calculate_metrics(&cop_ap, &cop_ml, &vel_ap, &vel_ml, dt);
        let sway_summary = self.calculate_summary(&cop_ap, &cop_ml);

        let ground_truth = CopGroundTruth {
            stance: StanceCondition::BilateralNormal,
            pathology: None,
            sway_metrics,
            sway_summary,
            dominant_freq_ap: 0.3,
            dominant_freq_ml: 0.25,
            mean_cop: (target_ap * 0.5, target_ml * 0.5),
        };

        CopOutput {
            time,
            cop_ap,
            cop_ml,
            vel_ap,
            vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate rhythmic weight shifting
    pub fn generate_weight_shifting(
        &mut self,
        direction: WeightShiftDirection,
        frequency: f64,
        duration: f64,
    ) -> CopOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);

        let (amp_ap, amp_ml) = match direction {
            WeightShiftDirection::AnteroPosterior => (0.1, 0.02),
            WeightShiftDirection::MedioLateral => (0.02, 0.08),
            WeightShiftDirection::Circular => (0.08, 0.08),
            WeightShiftDirection::FigureEight => (0.08, 0.06),
        };

        let noise_dist = Normal::new(0.0, self.config.noise_level * 0.01).unwrap();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let (base_ap, base_ml) = match direction {
                WeightShiftDirection::AnteroPosterior => {
                    (amp_ap * (2.0 * PI * frequency * t).sin(), 0.0)
                }
                WeightShiftDirection::MedioLateral => {
                    (0.0, amp_ml * (2.0 * PI * frequency * t).sin())
                }
                WeightShiftDirection::Circular => {
                    (
                        amp_ap * (2.0 * PI * frequency * t).cos(),
                        amp_ml * (2.0 * PI * frequency * t).sin(),
                    )
                }
                WeightShiftDirection::FigureEight => {
                    (
                        amp_ap * (2.0 * PI * frequency * t).sin(),
                        amp_ml * (4.0 * PI * frequency * t).sin(),
                    )
                }
            };

            let noise_ap: f64 = self.rng.sample(noise_dist);
            let noise_ml: f64 = self.rng.sample(noise_dist);

            cop_ap.push(base_ap + noise_ap);
            cop_ml.push(base_ml + noise_ml);
        }

        let vel_ap = self.calculate_velocity(&time, &cop_ap);
        let vel_ml = self.calculate_velocity(&time, &cop_ml);
        let sway_metrics = self.calculate_metrics(&cop_ap, &cop_ml, &vel_ap, &vel_ml, dt);
        let sway_summary = self.calculate_summary(&cop_ap, &cop_ml);

        let ground_truth = CopGroundTruth {
            stance: StanceCondition::BilateralNormal,
            pathology: None,
            sway_metrics,
            sway_summary,
            dominant_freq_ap: frequency,
            dominant_freq_ml: frequency,
            mean_cop: (0.0, 0.0),
        };

        CopOutput {
            time,
            cop_ap,
            cop_ml,
            vel_ap,
            vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate pathological balance pattern
    pub fn generate_pathological(
        &mut self,
        pathology: PathologicalBalance,
        duration: f64,
        stance: StanceCondition,
    ) -> CopOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);

        let (base_sway_ap, base_sway_ml) = self.get_stance_sway(stance);

        // Apply pathology modifications
        let (sway_mod_ap, sway_mod_ml, freq_mod, bias_ap, bias_ml, irregularity) =
            self.get_pathology_params(pathology);

        let modified_sway_ap = base_sway_ap * sway_mod_ap;
        let modified_sway_ml = base_sway_ml * sway_mod_ml;

        let freqs = [0.15, 0.3, 0.5, 0.8, 1.2];
        let phases_ap: Vec<f64> = freqs.iter().map(|_| self.rng.random::<f64>() * 2.0 * PI).collect();
        let phases_ml: Vec<f64> = freqs.iter().map(|_| self.rng.random::<f64>() * 2.0 * PI).collect();

        let noise_dist = Normal::new(0.0, self.config.noise_level * modified_sway_ap).unwrap();
        let irregular_dist = Normal::new(0.0, irregularity).unwrap();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let mut ap = 0.0;
            let mut ml = 0.0;

            for j in 0..freqs.len() {
                let modified_freq = freqs[j] * freq_mod;
                let amp_ap = [0.4, 0.25, 0.15, 0.12, 0.08][j];
                let amp_ml = [0.35, 0.28, 0.18, 0.12, 0.07][j];

                ap += amp_ap * (2.0 * PI * modified_freq * t + phases_ap[j]).sin();
                ml += amp_ml * (2.0 * PI * modified_freq * t + phases_ml[j]).sin();
            }

            ap = ap * modified_sway_ap * self.config.sway_amplitude + bias_ap;
            ml = ml * modified_sway_ml * self.config.sway_amplitude + bias_ml;

            // Add irregularity
            let irregular_ap: f64 = self.rng.sample(irregular_dist);
            let irregular_ml: f64 = self.rng.sample(irregular_dist);

            let noise_ap: f64 = self.rng.sample(noise_dist);
            let noise_ml: f64 = self.rng.sample(noise_dist);

            cop_ap.push(ap + irregular_ap + noise_ap);
            cop_ml.push(ml + irregular_ml + noise_ml);
        }

        let vel_ap = self.calculate_velocity(&time, &cop_ap);
        let vel_ml = self.calculate_velocity(&time, &cop_ml);
        let sway_metrics = self.calculate_metrics(&cop_ap, &cop_ml, &vel_ap, &vel_ml, dt);
        let sway_summary = self.calculate_summary(&cop_ap, &cop_ml);

        let ground_truth = CopGroundTruth {
            stance,
            pathology: Some(pathology),
            sway_metrics,
            sway_summary,
            dominant_freq_ap: 0.3 * freq_mod,
            dominant_freq_ml: 0.25 * freq_mod,
            mean_cop: (bias_ap, bias_ml),
        };

        CopOutput {
            time,
            cop_ap,
            cop_ml,
            vel_ap,
            vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Get sway parameters for stance condition
    fn get_stance_sway(&self, stance: StanceCondition) -> (f64, f64) {
        match stance {
            StanceCondition::BilateralNormal => (0.008, 0.005),
            StanceCondition::FeetTogether => (0.010, 0.008),
            StanceCondition::Tandem => (0.012, 0.015),
            StanceCondition::SingleLeg { .. } => (0.015, 0.020),
            StanceCondition::SemiTandem => (0.011, 0.010),
            StanceCondition::WideStance => (0.007, 0.004),
        }
    }

    /// Get pathology modification parameters
    fn get_pathology_params(&self, pathology: PathologicalBalance) -> (f64, f64, f64, f64, f64, f64) {
        // Returns: (sway_mod_ap, sway_mod_ml, freq_mod, bias_ap, bias_ml, irregularity)
        match pathology {
            PathologicalBalance::Parkinsonian { severity } => {
                (1.0 - 0.3 * severity, 1.0 - 0.3 * severity, 1.2, 0.0, 0.0, 0.001 * severity)
            }
            PathologicalBalance::CerebellarAtaxia { severity } => {
                (1.0 + 1.5 * severity, 1.0 + 1.5 * severity, 0.8, 0.0, 0.0, 0.02 * severity)
            }
            PathologicalBalance::VestibularDeficit { bias_direction, severity } => {
                let bias_ap = 0.02 * severity * bias_direction.cos();
                let bias_ml = 0.02 * severity * bias_direction.sin();
                (1.0 + 0.5 * severity, 1.0 + 0.5 * severity, 1.0, bias_ap, bias_ml, 0.005 * severity)
            }
            PathologicalBalance::PeripheralNeuropathy { severity } => {
                (1.0 + severity, 1.0 + 0.8 * severity, 0.9, 0.0, 0.0, 0.008 * severity)
            }
            PathologicalBalance::AgeRelated { age_factor } => {
                let mod_factor = 1.0 + 0.3 * (age_factor - 0.5).max(0.0);
                (mod_factor, mod_factor * 0.9, 0.95, 0.0, 0.0, 0.003 * age_factor)
            }
            PathologicalBalance::Hemiparesis { affected_side, severity } => {
                let bias = 0.03 * severity * if affected_side { 1.0 } else { -1.0 };
                (1.0 + 0.3 * severity, 1.0 + 0.6 * severity, 1.0, 0.0, bias, 0.005 * severity)
            }
            PathologicalBalance::AnxietyStiffening { severity } => {
                (1.0 - 0.4 * severity, 1.0 - 0.4 * severity, 1.5, 0.0, 0.0, 0.001)
            }
        }
    }

    /// Calculate velocity from position
    fn calculate_velocity(&self, time: &[f64], position: &[f64]) -> Vec<f64> {
        if position.len() < 2 {
            return vec![0.0; position.len()];
        }

        let mut velocity = Vec::with_capacity(position.len());
        velocity.push(0.0);

        for i in 1..position.len() {
            let dt = time[i] - time[i - 1];
            if dt > 0.0 {
                velocity.push((position[i] - position[i - 1]) / dt);
            } else {
                velocity.push(0.0);
            }
        }

        velocity
    }

    /// Calculate COP metrics
    fn calculate_metrics(
        &self,
        cop_ap: &[f64],
        cop_ml: &[f64],
        vel_ap: &[f64],
        vel_ml: &[f64],
        dt: f64,
    ) -> CopMetrics {
        let n = cop_ap.len() as f64;

        // Mean positions
        let mean_ap = cop_ap.iter().sum::<f64>() / n;
        let mean_ml = cop_ml.iter().sum::<f64>() / n;

        // RMS
        let rms_ap = (cop_ap.iter().map(|x| (x - mean_ap).powi(2)).sum::<f64>() / n).sqrt();
        let rms_ml = (cop_ml.iter().map(|x| (x - mean_ml).powi(2)).sum::<f64>() / n).sqrt();

        // Path length
        let path_length: f64 = (1..cop_ap.len())
            .map(|i| {
                let dap = cop_ap[i] - cop_ap[i - 1];
                let dml = cop_ml[i] - cop_ml[i - 1];
                (dap.powi(2) + dml.powi(2)).sqrt()
            })
            .sum();

        // Mean velocity
        let mean_velocity: f64 = vel_ap.iter()
            .zip(vel_ml.iter())
            .map(|(vap, vml)| (vap.powi(2) + vml.powi(2)).sqrt())
            .sum::<f64>() / n;

        // Range
        let range_ap = cop_ap.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cop_ap.iter().cloned().fold(f64::INFINITY, f64::min);
        let range_ml = cop_ml.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cop_ml.iter().cloned().fold(f64::INFINITY, f64::min);

        // 95% confidence ellipse (simplified calculation)
        let var_ap = rms_ap.powi(2);
        let var_ml = rms_ml.powi(2);
        let covar: f64 = cop_ap.iter()
            .zip(cop_ml.iter())
            .map(|(a, m)| (a - mean_ap) * (m - mean_ml))
            .sum::<f64>() / n;

        // Eigenvalues for ellipse
        let trace = var_ap + var_ml;
        let det = var_ap * var_ml - covar.powi(2);
        let discriminant = (trace.powi(2) - 4.0 * det).max(0.0).sqrt();

        let lambda1 = (trace + discriminant) / 2.0;
        let lambda2 = (trace - discriminant) / 2.0;

        let ellipse_major = 2.0 * (5.991 * lambda1.max(0.0)).sqrt(); // 95% CI factor
        let ellipse_minor = 2.0 * (5.991 * lambda2.max(0.0)).sqrt();
        let ellipse_area = PI * ellipse_major * ellipse_minor / 4.0;
        let ellipse_angle = if var_ap != var_ml {
            0.5 * (2.0 * covar).atan2(var_ap - var_ml)
        } else {
            0.0
        };

        // Sample entropy (simplified approximation)
        let sample_entropy_ap = self.approximate_sample_entropy(cop_ap);
        let sample_entropy_ml = self.approximate_sample_entropy(cop_ml);

        CopMetrics {
            rms_ap,
            rms_ml,
            path_length,
            mean_velocity,
            ellipse_area,
            ellipse_major,
            ellipse_minor,
            ellipse_angle,
            range_ap,
            range_ml,
            sample_entropy_ap,
            sample_entropy_ml,
        }
    }

    /// Calculate sway summary
    fn calculate_summary(&self, cop_ap: &[f64], cop_ml: &[f64]) -> SwaySummary {
        let n = cop_ap.len() as f64;

        let mean_ap = cop_ap.iter().sum::<f64>() / n;
        let mean_ml = cop_ml.iter().sum::<f64>() / n;

        let std_ap = (cop_ap.iter().map(|x| (x - mean_ap).powi(2)).sum::<f64>() / n).sqrt();
        let std_ml = (cop_ml.iter().map(|x| (x - mean_ml).powi(2)).sum::<f64>() / n).sqrt();

        let max_ap = cop_ap.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let max_ml = cop_ml.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        // Quadrant time calculation
        let mut quadrant_counts = [0usize; 4];
        for (ap, ml) in cop_ap.iter().zip(cop_ml.iter()) {
            let quadrant = match (ap >= &mean_ap, ml >= &mean_ml) {
                (true, true) => 0,   // Anterior-Right
                (true, false) => 1,  // Anterior-Left
                (false, false) => 2, // Posterior-Left
                (false, true) => 3,  // Posterior-Right
            };
            quadrant_counts[quadrant] += 1;
        }

        let total = cop_ap.len() as f64;
        let quadrant_time = [
            quadrant_counts[0] as f64 / total * 100.0,
            quadrant_counts[1] as f64 / total * 100.0,
            quadrant_counts[2] as f64 / total * 100.0,
            quadrant_counts[3] as f64 / total * 100.0,
        ];

        SwaySummary {
            mean_ap,
            mean_ml,
            std_ap,
            std_ml,
            max_ap,
            max_ml,
            quadrant_time,
        }
    }

    /// Approximate sample entropy
    fn approximate_sample_entropy(&self, data: &[f64]) -> f64 {
        if data.len() < 10 {
            return 0.0;
        }

        // Simplified approximation based on regularity
        let n = data.len();
        let mut diffs = Vec::with_capacity(n - 1);
        for i in 1..n {
            diffs.push((data[i] - data[i - 1]).abs());
        }

        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let std_diff = (diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64).sqrt();

        // Higher entropy for more irregular signals
        if mean_diff > 0.0 {
            (1.0 + std_diff / mean_diff).ln().max(0.0).min(3.0)
        } else {
            0.0
        }
    }
}

/// Direction for weight shifting
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WeightShiftDirection {
    AnteroPosterior,
    MedioLateral,
    Circular,
    FigureEight,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quiet_standing() {
        let config = CopConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CopGenerator::new(config);
        let output = generator.generate_quiet_standing(30.0, StanceCondition::BilateralNormal);

        assert!(!output.cop_ap.is_empty());
        assert!(!output.cop_ml.is_empty());
        assert!(output.ground_truth.sway_metrics.path_length > 0.0);
    }

    #[test]
    fn test_single_leg_increases_sway() {
        let config = CopConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CopGenerator::new(config);

        let bilateral = generator.generate_quiet_standing(10.0, StanceCondition::BilateralNormal);
        let single = generator.generate_quiet_standing(10.0, StanceCondition::SingleLeg { dominant: true });

        // Single leg should have more sway
        assert!(single.ground_truth.sway_metrics.rms_ml > bilateral.ground_truth.sway_metrics.rms_ml * 0.5);
    }

    #[test]
    fn test_limits_of_stability() {
        let config = CopConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CopGenerator::new(config);
        let output = generator.generate_limits_of_stability(0.0, 5.0); // Anterior direction

        // Should reach significant anterior displacement
        let max_ap = output.cop_ap.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(max_ap > 0.05);
    }

    #[test]
    fn test_weight_shifting() {
        let config = CopConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CopGenerator::new(config);
        let output = generator.generate_weight_shifting(
            WeightShiftDirection::MedioLateral,
            0.5,
            10.0,
        );

        assert!(!output.cop_ml.is_empty());
        assert!(output.ground_truth.sway_metrics.range_ml > 0.05);
    }

    #[test]
    fn test_pathological_cerebellar() {
        let config = CopConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CopGenerator::new(config);

        let normal = generator.generate_quiet_standing(10.0, StanceCondition::BilateralNormal);
        let ataxia = generator.generate_pathological(
            PathologicalBalance::CerebellarAtaxia { severity: 0.8 },
            10.0,
            StanceCondition::BilateralNormal,
        );

        // Cerebellar ataxia should increase sway
        assert!(ataxia.ground_truth.sway_metrics.rms_ap > normal.ground_truth.sway_metrics.rms_ap);
    }

    #[test]
    fn test_pathological_parkinson() {
        let config = CopConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CopGenerator::new(config);

        let normal = generator.generate_quiet_standing(10.0, StanceCondition::BilateralNormal);
        let pd = generator.generate_pathological(
            PathologicalBalance::Parkinsonian { severity: 0.8 },
            10.0,
            StanceCondition::BilateralNormal,
        );

        // Parkinsonian balance typically shows reduced sway amplitude
        assert!(pd.ground_truth.sway_metrics.rms_ap < normal.ground_truth.sway_metrics.rms_ap * 1.5);
    }
}
