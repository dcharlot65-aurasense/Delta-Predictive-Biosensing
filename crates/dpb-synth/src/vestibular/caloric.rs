//! Caloric Test Response Generator
//!
//! Generates vestibular responses to caloric stimulation:
//! - Warm and cold water irrigation
//! - Air caloric responses
//! - Monothermal screening
//! - Bithermal testing (Fitzgerald-Hallpike)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Configuration for caloric generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaloricConfig {
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Normal peak SPV (deg/s)
    pub normal_peak_spv: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for CaloricConfig {
    fn default() -> Self {
        Self {
            sample_rate: 100.0,
            normal_peak_spv: 25.0,
            noise_level: 0.1,
            seed: None,
        }
    }
}

/// Caloric stimulus parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CaloricStimulus {
    /// Temperature (°C)
    pub temperature: f64,
    /// Duration of irrigation (s)
    pub duration: f64,
    /// Side (true = right)
    pub right_ear: bool,
    /// Water vs air
    pub water: bool,
}

impl CaloricStimulus {
    /// Standard warm water (44°C)
    pub fn warm_right() -> Self {
        Self {
            temperature: 44.0,
            duration: 30.0,
            right_ear: true,
            water: true,
        }
    }

    pub fn warm_left() -> Self {
        Self {
            temperature: 44.0,
            duration: 30.0,
            right_ear: false,
            water: true,
        }
    }

    /// Standard cold water (30°C)
    pub fn cold_right() -> Self {
        Self {
            temperature: 30.0,
            duration: 30.0,
            right_ear: true,
            water: true,
        }
    }

    pub fn cold_left() -> Self {
        Self {
            temperature: 30.0,
            duration: 30.0,
            right_ear: false,
            water: true,
        }
    }

    /// Is this a warm stimulus?
    pub fn is_warm(&self) -> bool {
        self.temperature > 37.0
    }
}

/// Output from caloric generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaloricOutput {
    /// Time (s)
    pub time: Vec<f64>,
    /// Horizontal slow phase velocity (deg/s)
    pub slow_phase_velocity: Vec<f64>,
    /// Eye position (deg)
    pub eye_position: Vec<f64>,
    /// Ground truth
    pub ground_truth: CaloricGroundTruth,
    /// Configuration
    pub config: CaloricConfig,
}

/// Ground truth for caloric response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaloricGroundTruth {
    /// Stimulus used
    pub stimulus: CaloricStimulus,
    /// Response summary
    pub response: CaloricResponse,
    /// Latency to onset (s)
    pub latency: f64,
    /// Time to peak (s)
    pub time_to_peak: f64,
    /// Peak SPV (deg/s)
    pub peak_spv: f64,
    /// Duration of response (s)
    pub duration: f64,
    /// Nystagmus direction
    pub direction: NystagmusDirection,
}

/// Nystagmus direction
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NystagmusDirection {
    Right,
    Left,
}

/// Caloric response summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaloricResponse {
    /// Is response present?
    pub present: bool,
    /// Total response (area under SPV curve)
    pub total_response: f64,
    /// Response gain (compared to normal)
    pub gain: f64,
}

/// Caloric test generator
pub struct CaloricGenerator {
    config: CaloricConfig,
    rng: StdRng,
}

impl CaloricGenerator {
    /// Create new caloric generator
    pub fn new(config: CaloricConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate single caloric response
    pub fn generate_response(&mut self, stimulus: CaloricStimulus) -> CaloricOutput {
        let total_duration = 180.0; // 3 minutes to capture full response
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut slow_phase_velocity = Vec::with_capacity(n_samples);
        let mut eye_position = Vec::with_capacity(n_samples);

        // Caloric response characteristics
        let latency = 20.0 + self.rng.random::<f64>() * 10.0; // 20-30s latency
        let time_to_peak = 60.0 + self.rng.random::<f64>() * 20.0; // 60-80s to peak
        let response_duration = 120.0;

        // Temperature effect: deviation from body temp (37°C)
        let temp_deviation = (stimulus.temperature - 37.0).abs();
        let peak_spv = self.config.normal_peak_spv * (temp_deviation / 7.0);

        // Direction: COWS mnemonic (Cold Opposite, Warm Same)
        let direction = if stimulus.is_warm() == stimulus.right_ear {
            NystagmusDirection::Right
        } else {
            NystagmusDirection::Left
        };
        let direction_sign = match direction {
            NystagmusDirection::Right => 1.0,
            NystagmusDirection::Left => -1.0,
        };

        let noise_dist = Normal::new(0.0, self.config.noise_level * peak_spv).unwrap();

        let mut cumulative_pos = 0.0;
        let mut total_response = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Caloric response time course
            let spv = if t < latency {
                0.0
            } else {
                let active_time = t - latency;
                let rise_time = time_to_peak - latency;
                let fall_time = response_duration - time_to_peak;

                if active_time < rise_time {
                    // Rising phase
                    peak_spv * (1.0 - (-active_time / (rise_time * 0.4)).exp())
                } else if active_time < response_duration - latency {
                    // Falling phase
                    let decay_time = active_time - rise_time;
                    peak_spv * (-decay_time / (fall_time * 0.5)).exp()
                } else {
                    0.0
                }
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let final_spv = direction_sign * spv + noise;
            slow_phase_velocity.push(final_spv);

            total_response += spv.abs() * dt;

            // Integrate for position (sawtooth pattern)
            cumulative_pos -= final_spv * dt;
            // Quick phase resets
            if cumulative_pos.abs() > 10.0 {
                cumulative_pos *= 0.2; // Quick phase
            }
            eye_position.push(cumulative_pos);
        }

        let ground_truth = CaloricGroundTruth {
            stimulus,
            response: CaloricResponse {
                present: peak_spv > 5.0,
                total_response,
                gain: peak_spv / self.config.normal_peak_spv,
            },
            latency,
            time_to_peak,
            peak_spv,
            duration: response_duration,
            direction,
        };

        CaloricOutput {
            time,
            slow_phase_velocity,
            eye_position,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate bithermal caloric battery
    pub fn generate_bithermal(&mut self) -> BithermalResults {
        let warm_right = self.generate_response(CaloricStimulus::warm_right());
        let warm_left = self.generate_response(CaloricStimulus::warm_left());
        let cold_right = self.generate_response(CaloricStimulus::cold_right());
        let cold_left = self.generate_response(CaloricStimulus::cold_left());

        // Calculate Jongkees formulas
        let rw = warm_right.ground_truth.peak_spv;
        let lw = warm_left.ground_truth.peak_spv;
        let rc = cold_right.ground_truth.peak_spv;
        let lc = cold_left.ground_truth.peak_spv;

        // Unilateral weakness (canal paresis)
        let right_total = rw + rc;
        let left_total = lw + lc;
        let canal_paresis = if right_total + left_total > 0.0 {
            ((right_total - left_total) / (right_total + left_total)) * 100.0
        } else {
            0.0
        };

        // Directional preponderance
        let right_beating = rw + lc; // Right-beating responses
        let left_beating = lw + rc; // Left-beating responses
        let directional_preponderance = if right_beating + left_beating > 0.0 {
            ((right_beating - left_beating) / (right_beating + left_beating)) * 100.0
        } else {
            0.0
        };

        // Total response
        let total_response = rw + lw + rc + lc;

        BithermalResults {
            warm_right,
            warm_left,
            cold_right,
            cold_left,
            canal_paresis_percent: canal_paresis,
            directional_preponderance_percent: directional_preponderance,
            total_response,
            bilateral_weakness: total_response < 40.0,
            unilateral_weakness: canal_paresis.abs() > 25.0,
            affected_side: if canal_paresis > 25.0 {
                Some(false) // Left weakness
            } else if canal_paresis < -25.0 {
                Some(true) // Right weakness
            } else {
                None
            },
        }
    }

    /// Generate pathological caloric response
    pub fn generate_pathological(
        &mut self,
        stimulus: CaloricStimulus,
        pathology: CaloricPathology,
    ) -> CaloricOutput {
        let mut output = self.generate_response(stimulus);

        // Modify based on pathology
        let modifier = match pathology {
            CaloricPathology::UnilateralLoss { affected_side, severity } => {
                if affected_side == stimulus.right_ear {
                    1.0 - severity
                } else {
                    1.0
                }
            }
            CaloricPathology::BilateralLoss { severity } => 1.0 - severity,
            CaloricPathology::Hyperactive { gain } => gain,
        };

        // Apply modifier to SPV values
        for spv in &mut output.slow_phase_velocity {
            *spv *= modifier;
        }

        // Update ground truth
        output.ground_truth.peak_spv *= modifier;
        output.ground_truth.response.gain *= modifier;
        output.ground_truth.response.total_response *= modifier;
        output.ground_truth.response.present = output.ground_truth.peak_spv > 5.0;

        output
    }
}

/// Results from bithermal caloric testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BithermalResults {
    /// Warm right response
    pub warm_right: CaloricOutput,
    /// Warm left response
    pub warm_left: CaloricOutput,
    /// Cold right response
    pub cold_right: CaloricOutput,
    /// Cold left response
    pub cold_left: CaloricOutput,
    /// Unilateral weakness (Jongkees)
    pub canal_paresis_percent: f64,
    /// Directional preponderance
    pub directional_preponderance_percent: f64,
    /// Total caloric response
    pub total_response: f64,
    /// Bilateral weakness present
    pub bilateral_weakness: bool,
    /// Unilateral weakness present
    pub unilateral_weakness: bool,
    /// Affected side (if unilateral)
    pub affected_side: Option<bool>,
}

/// Caloric pathologies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum CaloricPathology {
    /// Unilateral vestibular loss
    UnilateralLoss { affected_side: bool, severity: f64 },
    /// Bilateral vestibular loss
    BilateralLoss { severity: f64 },
    /// Hyperactive response
    Hyperactive { gain: f64 },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warm_caloric() {
        let config = CaloricConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CaloricGenerator::new(config);
        let output = generator.generate_response(CaloricStimulus::warm_right());

        assert!(output.ground_truth.response.present);
        assert!(output.ground_truth.peak_spv > 0.0);
        assert_eq!(output.ground_truth.direction, NystagmusDirection::Right);
    }

    #[test]
    fn test_cold_caloric() {
        let config = CaloricConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CaloricGenerator::new(config);
        let output = generator.generate_response(CaloricStimulus::cold_right());

        // Cold right should produce left-beating nystagmus (COWS)
        assert_eq!(output.ground_truth.direction, NystagmusDirection::Left);
    }

    #[test]
    fn test_bithermal() {
        let config = CaloricConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CaloricGenerator::new(config);
        let results = generator.generate_bithermal();

        // Normal subject should have low canal paresis
        assert!(results.canal_paresis_percent.abs() < 25.0);
        assert!(!results.bilateral_weakness);
    }

    #[test]
    fn test_unilateral_loss() {
        let config = CaloricConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = CaloricGenerator::new(config);

        let normal = generator.generate_response(CaloricStimulus::warm_right());
        let pathological = generator.generate_pathological(
            CaloricStimulus::warm_right(),
            CaloricPathology::UnilateralLoss {
                affected_side: true,
                severity: 0.8,
            },
        );

        assert!(pathological.ground_truth.peak_spv < normal.ground_truth.peak_spv * 0.5);
    }
}
