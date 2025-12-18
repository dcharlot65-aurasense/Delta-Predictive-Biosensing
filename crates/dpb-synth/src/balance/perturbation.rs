//! Balance Perturbation Response Generator
//!
//! Generates postural responses to external perturbations:
//! - Platform translations and rotations
//! - Push/pull perturbations
//! - Slip and trip simulations
//! - Recovery strategies (ankle, hip, stepping)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for perturbation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Body mass in kg
    pub body_mass: f64,
    /// Standing height in m
    pub height: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for PerturbationConfig {
    fn default() -> Self {
        Self {
            sample_rate: 200.0,
            body_mass: 70.0,
            height: 1.7,
            noise_level: 0.02,
            seed: None,
        }
    }
}

/// Types of perturbations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PerturbationType {
    /// Platform translation
    Translation { velocity: f64, displacement: f64 },
    /// Platform rotation (toes up/down)
    Rotation { angle: f64, velocity: f64 },
    /// External push/pull force
    ExternalForce { magnitude: f64, duration: f64 },
    /// Simulated slip
    Slip { velocity: f64 },
    /// Simulated trip
    Trip { obstacle_height: f64 },
    /// Support surface compliance change
    ComplianceChange { stiffness_ratio: f64 },
}

/// Direction of perturbation
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PerturbationDirection {
    Anterior,
    Posterior,
    LeftLateral,
    RightLateral,
    AnteriorLeft,
    AnteriorRight,
    PosteriorLeft,
    PosteriorRight,
}

impl PerturbationDirection {
    fn to_angle(&self) -> f64 {
        match self {
            Self::Anterior => 0.0,
            Self::Posterior => PI,
            Self::LeftLateral => PI / 2.0,
            Self::RightLateral => -PI / 2.0,
            Self::AnteriorLeft => PI / 4.0,
            Self::AnteriorRight => -PI / 4.0,
            Self::PosteriorLeft => 3.0 * PI / 4.0,
            Self::PosteriorRight => -3.0 * PI / 4.0,
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    /// Ankle strategy (small perturbations)
    Ankle,
    /// Hip strategy (medium perturbations)
    Hip,
    /// Stepping strategy (large perturbations)
    Stepping,
    /// Mixed ankle-hip
    AnkleHip,
    /// Failed recovery (fall)
    Failed,
}

/// Output from perturbation generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationOutput {
    /// Time points in seconds
    pub time: Vec<f64>,
    /// COP AP position (m)
    pub cop_ap: Vec<f64>,
    /// COP ML position (m)
    pub cop_ml: Vec<f64>,
    /// COM AP position (m)
    pub com_ap: Vec<f64>,
    /// COM ML position (m)
    pub com_ml: Vec<f64>,
    /// COM AP velocity (m/s)
    pub com_vel_ap: Vec<f64>,
    /// COM ML velocity (m/s)
    pub com_vel_ml: Vec<f64>,
    /// Ground truth
    pub ground_truth: PerturbationGroundTruth,
    /// Configuration
    pub config: PerturbationConfig,
}

/// Ground truth for perturbation responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationGroundTruth {
    /// Type of perturbation
    pub perturbation_type: PerturbationType,
    /// Direction of perturbation
    pub direction: PerturbationDirection,
    /// Recovery strategy used
    pub recovery_strategy: RecoveryStrategy,
    /// Perturbation onset time (s)
    pub onset_time: f64,
    /// Response onset latency (ms)
    pub response_latency: f64,
    /// Time to peak displacement (s)
    pub time_to_peak: f64,
    /// Peak COM displacement (m)
    pub peak_displacement: f64,
    /// Recovery time (s)
    pub recovery_time: f64,
    /// Margin of stability at peak (m)
    pub min_margin_of_stability: f64,
    /// Whether recovery was successful
    pub recovery_successful: bool,
    /// Step taken (for stepping strategy)
    pub step_taken: Option<StepInfo>,
}

/// Information about protective step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepInfo {
    /// Step onset time (s)
    pub onset_time: f64,
    /// Step direction (radians)
    pub direction: f64,
    /// Step length (m)
    pub length: f64,
    /// Step duration (s)
    pub duration: f64,
}

/// Perturbation response generator
pub struct PerturbationGenerator {
    config: PerturbationConfig,
    rng: StdRng,
}

impl PerturbationGenerator {
    /// Create a new perturbation generator
    pub fn new(config: PerturbationConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Generate response to platform translation
    pub fn generate_translation_response(
        &mut self,
        direction: PerturbationDirection,
        velocity: f64,
        displacement: f64,
    ) -> PerturbationOutput {
        let total_duration = 3.0; // Pre + perturbation + recovery
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let onset_time = 0.5; // 500ms quiet standing before perturbation
        let perturbation_duration = displacement / velocity;

        let direction_angle = direction.to_angle();
        let pert_ap = displacement * direction_angle.cos();
        let pert_ml = displacement * direction_angle.sin();

        // Determine recovery strategy based on perturbation magnitude
        let recovery_strategy = if displacement < 0.05 {
            RecoveryStrategy::Ankle
        } else if displacement < 0.10 {
            RecoveryStrategy::Hip
        } else if displacement < 0.20 {
            RecoveryStrategy::AnkleHip
        } else {
            RecoveryStrategy::Stepping
        };

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);
        let mut com_ap = Vec::with_capacity(n_samples);
        let mut com_ml = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * 0.01).unwrap();

        // Response parameters
        let response_latency = 0.100 + self.rng.r#gen::<f64>() * 0.050; // 100-150ms
        let time_constant = 0.3; // Recovery time constant

        let mut peak_disp = 0.0_f64;
        let mut time_to_peak = 0.0;
        let mut recovery_time = total_duration;
        let baseline_threshold = 0.01;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let (base_cop_ap, base_cop_ml, base_com_ap, base_com_ml) = if t < onset_time {
                // Quiet standing
                let sway_ap = 0.005 * (2.0 * PI * 0.3 * t).sin();
                let sway_ml = 0.003 * (2.0 * PI * 0.25 * t).sin();
                (sway_ap, sway_ml, sway_ap * 0.8, sway_ml * 0.8)
            } else if t < onset_time + perturbation_duration {
                // During perturbation
                let pert_progress = (t - onset_time) / perturbation_duration;

                // Platform moves, COM lags behind
                let platform_ap = pert_ap * pert_progress;
                let platform_ml = pert_ml * pert_progress;

                // COM response delayed
                let com_response = if t > onset_time + response_latency / 1000.0 {
                    let response_time = t - onset_time - response_latency / 1000.0;
                    1.0 - (-response_time / 0.1).exp()
                } else {
                    0.0
                };

                let com_ap = -pert_ap * (1.0 - com_response) * pert_progress;
                let com_ml = -pert_ml * (1.0 - com_response) * pert_progress;

                // COP leads COM for recovery
                let cop_ap = com_ap - 0.02 * pert_ap.signum();
                let cop_ml = com_ml - 0.02 * pert_ml.signum();

                (cop_ap, cop_ml, com_ap, com_ml)
            } else {
                // Recovery phase
                let recovery_t = t - onset_time - perturbation_duration;

                // Exponential recovery
                let recovery_factor = (-recovery_t / time_constant).exp();

                let peak_com_ap = -pert_ap * 0.8;
                let peak_com_ml = -pert_ml * 0.8;

                let com_ap = peak_com_ap * recovery_factor;
                let com_ml = peak_com_ml * recovery_factor;

                // COP overshoots during recovery
                let overshoot = 0.3 * (PI * recovery_t / 0.5).sin() * recovery_factor;
                let cop_ap = com_ap + overshoot * pert_ap.signum();
                let cop_ml = com_ml + overshoot * pert_ml.signum();

                (cop_ap, cop_ml, com_ap, com_ml)
            };

            let noise: f64 = self.rng.sample(noise_dist);

            cop_ap.push(base_cop_ap + noise);
            cop_ml.push(base_cop_ml + noise);
            com_ap.push(base_com_ap + noise * 0.5);
            com_ml.push(base_com_ml + noise * 0.5);

            // Track metrics
            let disp = (base_com_ap.powi(2) + base_com_ml.powi(2)).sqrt();
            if disp > peak_disp {
                peak_disp = disp;
                time_to_peak = t - onset_time;
            }

            if t > onset_time + perturbation_duration + 0.2 && disp < baseline_threshold && recovery_time == total_duration {
                recovery_time = t - onset_time;
            }
        }

        // Calculate velocities
        let com_vel_ap = Self::calculate_velocity(&time, &com_ap);
        let com_vel_ml = Self::calculate_velocity(&time, &com_ml);

        // Calculate margin of stability (simplified)
        let bos_limit = 0.15; // Approximate BOS limit
        let min_mos = bos_limit - peak_disp;

        let ground_truth = PerturbationGroundTruth {
            perturbation_type: PerturbationType::Translation { velocity, displacement },
            direction,
            recovery_strategy,
            onset_time,
            response_latency: response_latency * 1000.0,
            time_to_peak,
            peak_displacement: peak_disp,
            recovery_time,
            min_margin_of_stability: min_mos,
            recovery_successful: min_mos > 0.0,
            step_taken: if recovery_strategy == RecoveryStrategy::Stepping {
                Some(StepInfo {
                    onset_time: onset_time + 0.3,
                    direction: direction_angle + PI,
                    length: displacement * 0.8,
                    duration: 0.3,
                })
            } else {
                None
            },
        };

        PerturbationOutput {
            time,
            cop_ap,
            cop_ml,
            com_ap,
            com_ml,
            com_vel_ap,
            com_vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate response to platform rotation
    pub fn generate_rotation_response(
        &mut self,
        toes_up: bool,
        angle_deg: f64,
        velocity_deg_s: f64,
    ) -> PerturbationOutput {
        let total_duration = 3.0;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let onset_time = 0.5;
        let angle_rad = angle_deg.to_radians();
        let velocity_rad = velocity_deg_s.to_radians();
        let rotation_duration = angle_rad / velocity_rad;

        let direction_sign = if toes_up { 1.0 } else { -1.0 };

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);
        let mut com_ap = Vec::with_capacity(n_samples);
        let mut com_ml = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * 0.01).unwrap();

        let response_latency = 0.080; // Earlier response to rotation
        let time_constant = 0.4;

        let mut peak_disp = 0.0_f64;
        let mut time_to_peak = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let (base_cop_ap, base_cop_ml, base_com_ap, base_com_ml) = if t < onset_time {
                (0.0, 0.0, 0.0, 0.0)
            } else if t < onset_time + rotation_duration {
                // During rotation
                let progress = (t - onset_time) / rotation_duration;
                let current_angle = angle_rad * progress;

                // Ankle stretch causes COM shift
                let com_shift = direction_sign * self.config.height * current_angle.sin() * 0.5;

                let com_ap = com_shift;
                let cop_ap = com_shift * 1.2; // COP leads

                (cop_ap, 0.0, com_ap, 0.0)
            } else {
                // Recovery
                let recovery_t = t - onset_time - rotation_duration;
                let peak_shift = direction_sign * self.config.height * angle_rad.sin() * 0.5;

                let recovery_factor = (-recovery_t / time_constant).exp();
                let com_ap = peak_shift * recovery_factor;
                let cop_ap = com_ap * (1.0 + 0.3 * (PI * recovery_t / 0.3).sin().abs());

                (cop_ap, 0.0, com_ap, 0.0)
            };

            let noise: f64 = self.rng.sample(noise_dist);

            cop_ap.push(base_cop_ap + noise);
            cop_ml.push(base_cop_ml + noise);
            com_ap.push(base_com_ap + noise * 0.5);
            com_ml.push(base_com_ml + noise * 0.5);

            if base_com_ap.abs() > peak_disp {
                peak_disp = base_com_ap.abs();
                time_to_peak = t - onset_time;
            }
        }

        let com_vel_ap = Self::calculate_velocity(&time, &com_ap);
        let com_vel_ml = Self::calculate_velocity(&time, &com_ml);

        let recovery_strategy = if angle_deg < 5.0 {
            RecoveryStrategy::Ankle
        } else if angle_deg < 10.0 {
            RecoveryStrategy::AnkleHip
        } else {
            RecoveryStrategy::Hip
        };

        let ground_truth = PerturbationGroundTruth {
            perturbation_type: PerturbationType::Rotation {
                angle: angle_deg,
                velocity: velocity_deg_s,
            },
            direction: if toes_up {
                PerturbationDirection::Posterior
            } else {
                PerturbationDirection::Anterior
            },
            recovery_strategy,
            onset_time,
            response_latency: response_latency * 1000.0,
            time_to_peak,
            peak_displacement: peak_disp,
            recovery_time: time_constant * 3.0,
            min_margin_of_stability: 0.15 - peak_disp,
            recovery_successful: peak_disp < 0.12,
            step_taken: None,
        };

        PerturbationOutput {
            time,
            cop_ap,
            cop_ml,
            com_ap,
            com_ml,
            com_vel_ap,
            com_vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate response to external push/pull
    pub fn generate_push_response(
        &mut self,
        direction: PerturbationDirection,
        force_magnitude: f64,
        force_duration: f64,
    ) -> PerturbationOutput {
        let total_duration = 3.0;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let onset_time = 0.5;
        let direction_angle = direction.to_angle();

        // Calculate impulse and resulting velocity change
        let impulse = force_magnitude * force_duration;
        let velocity_change = impulse / self.config.body_mass;

        let mut time = Vec::with_capacity(n_samples);
        let mut cop_ap = Vec::with_capacity(n_samples);
        let mut cop_ml = Vec::with_capacity(n_samples);
        let mut com_ap = Vec::with_capacity(n_samples);
        let mut com_ml = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * 0.01).unwrap();

        let mut current_vel_ap = 0.0;
        let mut current_vel_ml = 0.0;
        let mut current_pos_ap = 0.0;
        let mut current_pos_ml = 0.0;

        let damping = 3.0; // Recovery damping
        let stiffness = 20.0; // Postural stiffness

        let mut peak_disp = 0.0_f64;
        let mut time_to_peak = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            if t >= onset_time && t < onset_time + force_duration {
                // Apply force
                let accel = force_magnitude / self.config.body_mass;
                current_vel_ap += accel * direction_angle.cos() * dt;
                current_vel_ml += accel * direction_angle.sin() * dt;
            }

            // Spring-damper recovery
            let restore_ap = -stiffness * current_pos_ap - damping * current_vel_ap;
            let restore_ml = -stiffness * current_pos_ml - damping * current_vel_ml;

            current_vel_ap += restore_ap * dt;
            current_vel_ml += restore_ml * dt;
            current_pos_ap += current_vel_ap * dt;
            current_pos_ml += current_vel_ml * dt;

            let noise: f64 = self.rng.sample(noise_dist);

            // COP reflects recovery torque
            let cop_offset = 0.02;
            cop_ap.push(current_pos_ap - cop_offset * current_vel_ap.signum() + noise);
            cop_ml.push(current_pos_ml - cop_offset * current_vel_ml.signum() + noise);
            com_ap.push(current_pos_ap + noise * 0.5);
            com_ml.push(current_pos_ml + noise * 0.5);

            let disp = (current_pos_ap.powi(2) + current_pos_ml.powi(2)).sqrt();
            if disp > peak_disp {
                peak_disp = disp;
                time_to_peak = t - onset_time;
            }
        }

        let com_vel_ap = Self::calculate_velocity(&time, &com_ap);
        let com_vel_ml = Self::calculate_velocity(&time, &com_ml);

        let recovery_strategy = if peak_disp < 0.03 {
            RecoveryStrategy::Ankle
        } else if peak_disp < 0.08 {
            RecoveryStrategy::Hip
        } else {
            RecoveryStrategy::Stepping
        };

        let ground_truth = PerturbationGroundTruth {
            perturbation_type: PerturbationType::ExternalForce {
                magnitude: force_magnitude,
                duration: force_duration,
            },
            direction,
            recovery_strategy,
            onset_time,
            response_latency: 0.0, // Passive response, no latency
            time_to_peak,
            peak_displacement: peak_disp,
            recovery_time: 1.0,
            min_margin_of_stability: 0.15 - peak_disp,
            recovery_successful: peak_disp < 0.12,
            step_taken: None,
        };

        PerturbationOutput {
            time,
            cop_ap,
            cop_ml,
            com_ap,
            com_ml,
            com_vel_ap,
            com_vel_ml,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Calculate velocity from position
    fn calculate_velocity(time: &[f64], position: &[f64]) -> Vec<f64> {
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_translation_response() {
        let config = PerturbationConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = PerturbationGenerator::new(config);
        let output = generator.generate_translation_response(
            PerturbationDirection::Posterior,
            0.3,
            0.05,
        );

        assert!(!output.com_ap.is_empty());
        assert!(output.ground_truth.peak_displacement > 0.0);
        assert!(output.ground_truth.response_latency > 0.0);
    }

    #[test]
    fn test_rotation_response() {
        let config = PerturbationConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = PerturbationGenerator::new(config);
        let output = generator.generate_rotation_response(true, 8.0, 50.0);

        assert!(!output.com_ap.is_empty());
        assert!(output.ground_truth.recovery_successful);
    }

    #[test]
    fn test_push_response() {
        let config = PerturbationConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = PerturbationGenerator::new(config);
        let output = generator.generate_push_response(
            PerturbationDirection::Anterior,
            50.0,
            0.1,
        );

        assert!(!output.com_ap.is_empty());
        assert!(output.ground_truth.time_to_peak > 0.0);
    }

    #[test]
    fn test_stepping_strategy_for_large_perturbation() {
        let config = PerturbationConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = PerturbationGenerator::new(config);
        let output = generator.generate_translation_response(
            PerturbationDirection::Posterior,
            0.5,
            0.25, // Large perturbation
        );

        assert_eq!(output.ground_truth.recovery_strategy, RecoveryStrategy::Stepping);
        assert!(output.ground_truth.step_taken.is_some());
    }
}
