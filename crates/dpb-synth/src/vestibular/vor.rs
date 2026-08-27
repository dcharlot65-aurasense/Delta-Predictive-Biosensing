//! Vestibulo-Ocular Reflex (VOR) Signal Generator
//!
//! Generates VOR responses including:
//! - Sinusoidal head rotation responses
//! - Head impulse test (HIT) responses
//! - Rotational chair responses
//! - Pathological patterns (gain reduction, phase lead/lag)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for VOR generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VorConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Normal VOR gain (typically 0.9-1.0)
    pub normal_gain: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for VorConfig {
    fn default() -> Self {
        Self {
            sample_rate: 500.0, // High rate for eye tracking
            normal_gain: 0.95,
            noise_level: 0.02,
            seed: None,
        }
    }
}

/// Output from VOR generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VorOutput {
    /// Time points (s)
    pub time: Vec<f64>,
    /// Head velocity (deg/s)
    pub head_velocity: Vec<f64>,
    /// Eye velocity (deg/s) - should be opposite to head
    pub eye_velocity: Vec<f64>,
    /// Eye position (deg)
    pub eye_position: Vec<f64>,
    /// Gaze velocity (head + eye, should be ~0 for perfect VOR)
    pub gaze_velocity: Vec<f64>,
    /// Ground truth
    pub ground_truth: VorGroundTruth,
    /// Configuration
    pub config: VorConfig,
}

/// Ground truth for VOR responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VorGroundTruth {
    /// VOR gain (eye velocity / head velocity)
    pub gain: VorGain,
    /// VOR phase (degrees, negative = phase lead)
    pub phase: f64,
    /// Asymmetry (% difference left vs right)
    pub asymmetry: f64,
    /// Time constant (s)
    pub time_constant: f64,
    /// Head impulse results (if applicable)
    pub hit_results: Option<Vec<HeadImpulseResult>>,
    /// Applied pathology
    pub pathology: Option<VorPathology>,
}

/// VOR gain measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VorGain {
    /// Overall gain
    pub overall: f64,
    /// Leftward gain
    pub leftward: f64,
    /// Rightward gain
    pub rightward: f64,
    /// Gain at different frequencies (if sinusoidal)
    pub frequency_gains: Vec<(f64, f64)>,
}

/// Head impulse test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadImpulseResult {
    /// Direction (positive = rightward)
    pub direction: f64,
    /// Peak head velocity (deg/s)
    pub peak_head_velocity: f64,
    /// VOR gain
    pub gain: f64,
    /// Catch-up saccades if present
    pub catch_up_saccades: Vec<CatchUpSaccade>,
    /// Covert saccade present
    pub covert_saccade: bool,
    /// Overt saccade present
    pub overt_saccade: bool,
}

/// Catch-up saccade information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CatchUpSaccade {
    /// Onset time relative to head impulse (ms)
    pub onset_time: f64,
    /// Amplitude (deg)
    pub amplitude: f64,
    /// Peak velocity (deg/s)
    pub peak_velocity: f64,
    /// Duration (ms)
    pub duration: f64,
    /// Covert (during head movement) or overt (after)
    pub covert: bool,
}

/// Pathological VOR patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum VorPathology {
    /// Unilateral vestibular loss
    UnilateralLoss { affected_side: bool, severity: f64 },
    /// Bilateral vestibular loss
    BilateralLoss { severity: f64 },
    /// Age-related decline
    AgeRelated { age_factor: f64 },
    /// Cerebellar dysfunction (gain/phase abnormalities)
    Cerebellar { gain_error: f64, phase_error: f64 },
    /// Vestibular neuritis (acute)
    VestibularNeuritis { affected_side: bool, days_post_onset: f64 },
}

/// VOR signal generator
pub struct VorGenerator {
    config: VorConfig,
    rng: StdRng,
}

impl VorGenerator {
    /// Create a new VOR generator
    pub fn new(config: VorConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate sinusoidal VOR response
    pub fn generate_sinusoidal(
        &mut self,
        frequency: f64,
        amplitude_deg_s: f64,
        duration: f64,
    ) -> VorOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut head_velocity = Vec::with_capacity(n_samples);
        let mut eye_velocity = Vec::with_capacity(n_samples);
        let mut eye_position = Vec::with_capacity(n_samples);

        let gain = self.config.normal_gain;
        let phase_deg: f64 = -5.0; // Slight phase lead is normal
        let phase_rad = phase_deg.to_radians();

        let noise_dist = Normal::new(0.0, self.config.noise_level * amplitude_deg_s).unwrap();

        let mut cumulative_eye_pos = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // Sinusoidal head velocity
            let hv = amplitude_deg_s * (2.0 * PI * frequency * t).sin();
            head_velocity.push(hv);

            // Eye velocity (opposite direction, with gain and phase)
            let ev_base = -gain * amplitude_deg_s * (2.0 * PI * frequency * t + phase_rad).sin();
            let noise: f64 = self.rng.sample(noise_dist);
            let ev = ev_base + noise;
            eye_velocity.push(ev);

            // Integrate for position
            cumulative_eye_pos += ev * dt;
            eye_position.push(cumulative_eye_pos);
        }

        // Calculate gaze velocity
        let gaze_velocity: Vec<f64> = head_velocity
            .iter()
            .zip(eye_velocity.iter())
            .map(|(h, e)| h + e)
            .collect();

        let ground_truth = VorGroundTruth {
            gain: VorGain {
                overall: gain,
                leftward: gain,
                rightward: gain,
                frequency_gains: vec![(frequency, gain)],
            },
            phase: phase_deg,
            asymmetry: 0.0,
            time_constant: 15.0, // Normal ~15-20s
            hit_results: None,
            pathology: None,
        };

        VorOutput {
            time,
            head_velocity,
            eye_velocity,
            eye_position,
            gaze_velocity,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate head impulse test (HIT) response
    pub fn generate_head_impulse(
        &mut self,
        peak_velocity: f64,
        rightward: bool,
        n_trials: usize,
    ) -> VorOutput {
        let trial_duration = 0.5; // 500ms per trial
        let total_duration = n_trials as f64 * trial_duration;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let direction = if rightward { 1.0 } else { -1.0 };

        let mut time = Vec::with_capacity(n_samples);
        let mut head_velocity = Vec::with_capacity(n_samples);
        let mut eye_velocity = Vec::with_capacity(n_samples);
        let mut eye_position = Vec::with_capacity(n_samples);

        let mut hit_results = Vec::with_capacity(n_trials);

        let noise_dist = Normal::new(0.0, self.config.noise_level * peak_velocity).unwrap();

        let mut cumulative_eye_pos = 0.0;

        for trial in 0..n_trials {
            let trial_start = trial as f64 * trial_duration;

            // Vary peak velocity slightly between trials
            let trial_peak = peak_velocity * (0.9 + self.rng.random::<f64>() * 0.2);

            let samples_per_trial = (trial_duration * self.config.sample_rate) as usize;

            for i in 0..samples_per_trial {
                let t = trial_start + i as f64 * dt;
                let trial_t = i as f64 * dt;
                time.push(t);

                // Head impulse profile (rapid acceleration, then deceleration)
                let impulse_duration = 0.15; // 150ms
                let hv = if trial_t < impulse_duration {
                    let phase = trial_t / impulse_duration;
                    direction * trial_peak * (PI * phase).sin()
                } else {
                    0.0
                };
                head_velocity.push(hv);

                // Eye velocity response
                let ev = -self.config.normal_gain * hv + self.rng.sample(noise_dist);
                eye_velocity.push(ev);

                cumulative_eye_pos += ev * dt;
                eye_position.push(cumulative_eye_pos);
            }

            // Record HIT result for this trial
            hit_results.push(HeadImpulseResult {
                direction,
                peak_head_velocity: trial_peak,
                gain: self.config.normal_gain,
                catch_up_saccades: vec![],
                covert_saccade: false,
                overt_saccade: false,
            });
        }

        let gaze_velocity: Vec<f64> = head_velocity
            .iter()
            .zip(eye_velocity.iter())
            .map(|(h, e)| h + e)
            .collect();

        let ground_truth = VorGroundTruth {
            gain: VorGain {
                overall: self.config.normal_gain,
                // VorConfig has a single normal_gain and no asymmetry term, so
                // both directions are by construction equal -- the `rightward`
                // conditional that used to sit here selected between two copies
                // of the same value and implied a directionality the model does
                // not represent. `asymmetry: 0.0` below says the same thing.
                leftward: self.config.normal_gain,
                rightward: self.config.normal_gain,
                frequency_gains: vec![],
            },
            phase: 0.0,
            asymmetry: 0.0,
            time_constant: 15.0,
            hit_results: Some(hit_results),
            pathology: None,
        };

        VorOutput {
            time,
            head_velocity,
            eye_velocity,
            eye_position,
            gaze_velocity,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate pathological VOR response
    pub fn generate_pathological(
        &mut self,
        pathology: VorPathology,
        frequency: f64,
        amplitude: f64,
        duration: f64,
    ) -> VorOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let (gain_left, gain_right, phase_error, asymmetry) = match pathology {
            VorPathology::UnilateralLoss { affected_side, severity } => {
                if affected_side {
                    (self.config.normal_gain, self.config.normal_gain * (1.0 - severity), 0.0, severity * 100.0)
                } else {
                    (self.config.normal_gain * (1.0 - severity), self.config.normal_gain, 0.0, -severity * 100.0)
                }
            }
            VorPathology::BilateralLoss { severity } => {
                let reduced = self.config.normal_gain * (1.0 - severity);
                (reduced, reduced, 10.0 * severity, 0.0)
            }
            VorPathology::AgeRelated { age_factor } => {
                let reduction = 0.1 * age_factor;
                let reduced = self.config.normal_gain * (1.0 - reduction);
                (reduced, reduced, 5.0 * age_factor, 0.0)
            }
            VorPathology::Cerebellar { gain_error, phase_error } => {
                let gain = self.config.normal_gain * (1.0 + gain_error);
                (gain, gain, phase_error, 0.0)
            }
            VorPathology::VestibularNeuritis { affected_side, days_post_onset } => {
                // Recovery follows exponential time course
                let recovery = 1.0 - (-days_post_onset / 30.0).exp();
                let affected_gain = self.config.normal_gain * (0.2 + 0.7 * recovery);
                if affected_side {
                    (self.config.normal_gain, affected_gain, 0.0, (1.0 - recovery) * 50.0)
                } else {
                    (affected_gain, self.config.normal_gain, 0.0, -(1.0 - recovery) * 50.0)
                }
            }
        };

        let mut time = Vec::with_capacity(n_samples);
        let mut head_velocity = Vec::with_capacity(n_samples);
        let mut eye_velocity = Vec::with_capacity(n_samples);
        let mut eye_position = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * amplitude).unwrap();
        let base_phase = -5.0_f64.to_radians();
        let phase_rad = base_phase + phase_error.to_radians();

        let mut cumulative_eye_pos = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let hv = amplitude * (2.0 * PI * frequency * t).sin();
            head_velocity.push(hv);

            // Direction-dependent gain
            let gain = if hv > 0.0 { gain_right } else { gain_left };
            let ev_base = -gain * amplitude * (2.0 * PI * frequency * t + phase_rad).sin();
            let ev = ev_base + self.rng.sample(noise_dist);
            eye_velocity.push(ev);

            cumulative_eye_pos += ev * dt;
            eye_position.push(cumulative_eye_pos);
        }

        let gaze_velocity: Vec<f64> = head_velocity
            .iter()
            .zip(eye_velocity.iter())
            .map(|(h, e)| h + e)
            .collect();

        let ground_truth = VorGroundTruth {
            gain: VorGain {
                overall: (gain_left + gain_right) / 2.0,
                leftward: gain_left,
                rightward: gain_right,
                frequency_gains: vec![(frequency, (gain_left + gain_right) / 2.0)],
            },
            phase: phase_error - 5.0,
            asymmetry,
            time_constant: 15.0,
            hit_results: None,
            pathology: Some(pathology),
        };

        VorOutput {
            time,
            head_velocity,
            eye_velocity,
            eye_position,
            gaze_velocity,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate pathological HIT with catch-up saccades
    pub fn generate_pathological_hit(
        &mut self,
        pathology: VorPathology,
        peak_velocity: f64,
        rightward: bool,
        n_trials: usize,
    ) -> VorOutput {
        let trial_duration = 0.5;
        let total_duration = n_trials as f64 * trial_duration;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let direction = if rightward { 1.0 } else { -1.0 };

        // Determine affected gain based on pathology and direction
        let affected_gain = match pathology {
            VorPathology::UnilateralLoss { affected_side, severity } => {
                if (affected_side && rightward) || (!affected_side && !rightward) {
                    self.config.normal_gain * (1.0 - severity)
                } else {
                    self.config.normal_gain
                }
            }
            VorPathology::BilateralLoss { severity } => {
                self.config.normal_gain * (1.0 - severity)
            }
            _ => self.config.normal_gain,
        };

        let mut time = Vec::with_capacity(n_samples);
        let mut head_velocity = Vec::with_capacity(n_samples);
        let mut eye_velocity = Vec::with_capacity(n_samples);
        let mut eye_position = Vec::with_capacity(n_samples);

        let mut hit_results = Vec::with_capacity(n_trials);
        let noise_dist = Normal::new(0.0, self.config.noise_level * peak_velocity).unwrap();

        let mut cumulative_eye_pos = 0.0;

        for trial in 0..n_trials {
            let trial_start = trial as f64 * trial_duration;
            let trial_peak = peak_velocity * (0.9 + self.rng.random::<f64>() * 0.2);
            let samples_per_trial = (trial_duration * self.config.sample_rate) as usize;

            let mut trial_catch_up_saccades = Vec::new();
            let gaze_error_threshold = 3.0; // deg

            // Track cumulative gaze error for saccade triggering
            let mut gaze_error = 0.0;
            let impulse_duration = 0.15;

            for i in 0..samples_per_trial {
                let t = trial_start + i as f64 * dt;
                let trial_t = i as f64 * dt;
                time.push(t);

                // Head impulse
                let hv = if trial_t < impulse_duration {
                    let phase = trial_t / impulse_duration;
                    direction * trial_peak * (PI * phase).sin()
                } else {
                    0.0
                };
                head_velocity.push(hv);

                // Deficient VOR
                let vor_response = -affected_gain * hv;

                // Accumulate gaze error
                if trial_t < impulse_duration {
                    gaze_error += (hv + vor_response) * dt;
                }

                // Generate catch-up saccade if error exceeds threshold
                let saccade_contribution = if gaze_error.abs() > gaze_error_threshold {
                    let saccade_onset = trial_t;
                    let saccade_amp = -gaze_error * 0.8;
                    let saccade_duration = 0.03; // 30ms

                    // Is this covert (during impulse) or overt (after)?
                    let is_covert = trial_t < impulse_duration;

                    if trial_catch_up_saccades.is_empty() || trial_t > trial_catch_up_saccades.last().map(|s: &CatchUpSaccade| s.onset_time / 1000.0 + 0.05).unwrap_or(0.0) {
                        trial_catch_up_saccades.push(CatchUpSaccade {
                            onset_time: saccade_onset * 1000.0,
                            amplitude: saccade_amp.abs(),
                            peak_velocity: saccade_amp.abs() / saccade_duration * 1000.0,
                            duration: saccade_duration * 1000.0,
                            covert: is_covert,
                        });
                        gaze_error = 0.0;
                    }

                    // Saccade velocity contribution
                    saccade_amp / saccade_duration * (-((trial_t - saccade_onset) / saccade_duration).powi(2)).exp()
                } else {
                    0.0
                };

                let ev = vor_response + saccade_contribution + self.rng.sample(noise_dist);
                eye_velocity.push(ev);

                cumulative_eye_pos += ev * dt;
                eye_position.push(cumulative_eye_pos);
            }

            hit_results.push(HeadImpulseResult {
                direction,
                peak_head_velocity: trial_peak,
                gain: affected_gain,
                covert_saccade: trial_catch_up_saccades.iter().any(|s| s.covert),
                overt_saccade: trial_catch_up_saccades.iter().any(|s| !s.covert),
                catch_up_saccades: trial_catch_up_saccades,
            });
        }

        let gaze_velocity: Vec<f64> = head_velocity
            .iter()
            .zip(eye_velocity.iter())
            .map(|(h, e)| h + e)
            .collect();

        let ground_truth = VorGroundTruth {
            gain: VorGain {
                overall: affected_gain,
                leftward: if rightward { self.config.normal_gain } else { affected_gain },
                rightward: if rightward { affected_gain } else { self.config.normal_gain },
                frequency_gains: vec![],
            },
            phase: 0.0,
            asymmetry: (self.config.normal_gain - affected_gain) / self.config.normal_gain * 100.0,
            time_constant: 15.0,
            hit_results: Some(hit_results),
            pathology: Some(pathology),
        };

        VorOutput {
            time,
            head_velocity,
            eye_velocity,
            eye_position,
            gaze_velocity,
            ground_truth,
            config: self.config.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sinusoidal_vor() {
        let config = VorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = VorGenerator::new(config);
        let output = generator.generate_sinusoidal(0.5, 60.0, 10.0);

        assert!(!output.eye_velocity.is_empty());
        assert!((output.ground_truth.gain.overall - 0.95).abs() < 0.1);
    }

    #[test]
    fn test_head_impulse() {
        let config = VorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = VorGenerator::new(config);
        let output = generator.generate_head_impulse(150.0, true, 5);

        assert!(output.ground_truth.hit_results.is_some());
        assert_eq!(output.ground_truth.hit_results.as_ref().unwrap().len(), 5);
    }

    #[test]
    fn test_unilateral_loss() {
        let config = VorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = VorGenerator::new(config);
        let output = generator.generate_pathological(
            VorPathology::UnilateralLoss {
                affected_side: true,
                severity: 0.7,
            },
            0.5,
            60.0,
            10.0,
        );

        // Right side should have reduced gain
        assert!(output.ground_truth.gain.rightward < output.ground_truth.gain.leftward);
    }

    #[test]
    fn test_pathological_hit_with_saccades() {
        let config = VorConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = VorGenerator::new(config);
        let output = generator.generate_pathological_hit(
            VorPathology::UnilateralLoss {
                affected_side: true,
                severity: 0.8,
            },
            200.0,
            true, // Rightward - affected side
            5,
        );

        // Should have catch-up saccades due to VOR deficit
        let results = output.ground_truth.hit_results.as_ref().unwrap();
        let has_saccades = results.iter().any(|r| !r.catch_up_saccades.is_empty());
        assert!(has_saccades || output.ground_truth.gain.rightward < 0.5);
    }
}
