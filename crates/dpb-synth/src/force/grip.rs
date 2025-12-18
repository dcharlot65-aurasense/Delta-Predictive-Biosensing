//! Grip Strength Dynamometry Signal Generator
//!
//! Generates realistic grip strength signals including:
//! - Maximum voluntary contraction (MVC) trials
//! - Sustained grip with fatigue modeling
//! - Rapid grip-release sequences
//! - Bilateral comparison protocols
//! - Pathological patterns (weakness, tremor, fatigue)

use rand::prelude::*;
use rand_distr::{Normal, Uniform};
use serde::{Deserialize, Serialize};

/// Configuration for grip strength generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GripConfig {
    /// Sampling rate in Hz
    pub sample_rate: f64,
    /// Maximum voluntary contraction in Newtons (typical: 200-600N)
    pub mvc_newtons: f64,
    /// Hand dominance factor (dominant hand typically 10% stronger)
    pub dominance_factor: f64,
    /// Age-related strength factor (0.0-1.0)
    pub age_factor: f64,
    /// Sex-based strength factor (females ~60-70% of males)
    pub sex_factor: f64,
    /// Noise level (0.0-1.0)
    pub noise_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for GripConfig {
    fn default() -> Self {
        Self {
            sample_rate: 1000.0,
            mvc_newtons: 400.0,
            dominance_factor: 1.1,
            age_factor: 1.0,
            sex_factor: 1.0,
            noise_level: 0.02,
            seed: None,
        }
    }
}

/// Output from grip strength generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GripOutput {
    /// Time points in seconds
    pub time: Vec<f64>,
    /// Force in Newtons
    pub force: Vec<f64>,
    /// Force rate (dF/dt) in N/s
    pub force_rate: Vec<f64>,
    /// Ground truth annotations
    pub ground_truth: GripGroundTruth,
    /// Configuration used
    pub config: GripConfig,
}

/// Ground truth for grip strength signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GripGroundTruth {
    /// Peak force achieved (N)
    pub peak_force: f64,
    /// Time to peak force (s)
    pub time_to_peak: f64,
    /// Peak rate of force development (N/s)
    pub peak_rfd: f64,
    /// Mean force during hold phase (N)
    pub mean_sustained_force: f64,
    /// Coefficient of variation during hold (%)
    pub force_variability: f64,
    /// Fatigue index (% decline from peak)
    pub fatigue_index: f64,
    /// Force impulse (N·s)
    pub impulse: f64,
    /// Trial events with timestamps
    pub events: Vec<GripEvent>,
    /// Applied pathology if any
    pub pathology: Option<PathologicalGrip>,
}

/// Events during grip trial
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GripEvent {
    /// Time of event (s)
    pub time: f64,
    /// Event type
    pub event_type: GripEventType,
    /// Force at event (N)
    pub force: f64,
}

/// Types of grip events
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GripEventType {
    /// Trial start
    TrialStart,
    /// Force onset (>5% MVC)
    ForceOnset,
    /// Peak force reached
    PeakForce,
    /// Hold phase start
    HoldStart,
    /// Hold phase end
    HoldEnd,
    /// Force offset (<5% MVC)
    ForceOffset,
    /// Trial end
    TrialEnd,
}

/// Pathological grip patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PathologicalGrip {
    /// Generalized weakness (reduced MVC)
    Weakness { severity: f64 },
    /// Rapid fatigue during sustained grip
    RapidFatigue { fatigue_rate: f64 },
    /// Tremor superimposed on grip
    Tremor { frequency: f64, amplitude: f64 },
    /// Cogwheel rigidity (Parkinson's-like)
    CogwheelRigidity { frequency: f64 },
    /// Grip-release impairment (slow release)
    SlowRelease { time_constant: f64 },
    /// Myotonia (delayed relaxation)
    Myotonia { delay: f64 },
    /// Spasticity (velocity-dependent resistance)
    Spasticity { gain: f64 },
}

/// Grip strength signal generator
pub struct GripGenerator {
    config: GripConfig,
    rng: StdRng,
}

impl GripGenerator {
    /// Create a new grip generator with the given configuration
    pub fn new(config: GripConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Generate maximum voluntary contraction (MVC) trial
    ///
    /// Standard protocol: ramp up, brief hold at max, release
    pub fn generate_mvc_trial(&mut self, trial_duration: f64) -> GripOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (trial_duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        // Effective MVC considering all factors
        let effective_mvc = self.config.mvc_newtons
            * self.config.age_factor
            * self.config.sex_factor;

        // Phase timing (as fractions of total duration)
        let ramp_up_end = 0.3;
        let hold_end = 0.6;
        let ramp_down_end = 0.9;

        // Noise distribution
        let noise_dist = Normal::new(0.0, self.config.noise_level * effective_mvc).unwrap();

        events.push(GripEvent {
            time: 0.0,
            event_type: GripEventType::TrialStart,
            force: 0.0,
        });

        let mut peak_force = 0.0_f64;
        let mut peak_time = 0.0;
        let mut onset_recorded = false;
        let mut hold_forces = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let phase_fraction = t / trial_duration;
            let base_force = if phase_fraction < ramp_up_end {
                // Ramp up phase - sigmoid curve
                let ramp_progress = phase_fraction / ramp_up_end;
                let sigmoid = 1.0 / (1.0 + (-12.0 * (ramp_progress - 0.5)).exp());
                effective_mvc * sigmoid
            } else if phase_fraction < hold_end {
                // Hold phase - slight natural variation
                let hold_progress = (phase_fraction - ramp_up_end) / (hold_end - ramp_up_end);
                let fatigue = 1.0 - 0.05 * hold_progress; // 5% fatigue during hold
                effective_mvc * fatigue
            } else if phase_fraction < ramp_down_end {
                // Ramp down phase
                let ramp_progress = (phase_fraction - hold_end) / (ramp_down_end - hold_end);
                let sigmoid = 1.0 / (1.0 + (12.0 * (ramp_progress - 0.5)).exp());
                effective_mvc * sigmoid
            } else {
                // Rest phase
                0.0
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);

            // Track events
            if !onset_recorded && f > 0.05 * effective_mvc {
                onset_recorded = true;
                events.push(GripEvent {
                    time: t,
                    event_type: GripEventType::ForceOnset,
                    force: f,
                });
            }

            if f > peak_force {
                peak_force = f;
                peak_time = t;
            }

            // Collect hold phase forces
            if phase_fraction >= ramp_up_end && phase_fraction < hold_end {
                hold_forces.push(f);
            }
        }

        // Record peak event
        events.push(GripEvent {
            time: peak_time,
            event_type: GripEventType::PeakForce,
            force: peak_force,
        });

        events.push(GripEvent {
            time: trial_duration,
            event_type: GripEventType::TrialEnd,
            force: *force.last().unwrap_or(&0.0),
        });

        // Calculate force rate
        let force_rate = self.calculate_force_rate(&time, &force);

        // Calculate metrics
        let mean_sustained = if !hold_forces.is_empty() {
            hold_forces.iter().sum::<f64>() / hold_forces.len() as f64
        } else {
            0.0
        };

        let force_variability = if !hold_forces.is_empty() && mean_sustained > 0.0 {
            let variance = hold_forces.iter()
                .map(|f| (f - mean_sustained).powi(2))
                .sum::<f64>() / hold_forces.len() as f64;
            (variance.sqrt() / mean_sustained) * 100.0
        } else {
            0.0
        };

        let peak_rfd = force_rate.iter().cloned().fold(0.0_f64, f64::max);
        let impulse = force.iter().sum::<f64>() * dt;

        let ground_truth = GripGroundTruth {
            peak_force,
            time_to_peak: peak_time,
            peak_rfd,
            mean_sustained_force: mean_sustained,
            force_variability,
            fatigue_index: ((peak_force - mean_sustained) / peak_force * 100.0).max(0.0),
            impulse,
            events,
            pathology: None,
        };

        GripOutput {
            time,
            force,
            force_rate,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate sustained grip trial with fatigue
    ///
    /// Target force held for extended duration, showing fatigue effects
    pub fn generate_sustained_grip(
        &mut self,
        duration: f64,
        target_percent_mvc: f64,
    ) -> GripOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let effective_mvc = self.config.mvc_newtons
            * self.config.age_factor
            * self.config.sex_factor;
        let target_force = effective_mvc * target_percent_mvc;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        // Fatigue model parameters (exponential decay)
        let fatigue_time_constant = 60.0; // seconds
        let max_fatigue = 0.4; // 40% max decline

        let noise_dist = Normal::new(0.0, self.config.noise_level * target_force).unwrap();

        events.push(GripEvent {
            time: 0.0,
            event_type: GripEventType::TrialStart,
            force: 0.0,
        });

        let ramp_duration = 2.0; // 2 second ramp up
        let mut peak_force = 0.0_f64;
        let mut peak_time = 0.0;
        let mut hold_forces = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let base_force = if t < ramp_duration {
                // Ramp up
                let progress = t / ramp_duration;
                target_force * progress
            } else {
                // Sustained with fatigue
                let fatigue_time = t - ramp_duration;
                let fatigue_factor = 1.0 - max_fatigue * (1.0 - (-fatigue_time / fatigue_time_constant).exp());
                target_force * fatigue_factor
            };

            // Add physiological tremor (8-12 Hz)
            let tremor_freq = 10.0;
            let tremor_amp = 0.02 * target_force;
            let tremor = tremor_amp * (2.0 * std::f64::consts::PI * tremor_freq * t).sin();

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + tremor + noise).max(0.0);
            force.push(f);

            if f > peak_force {
                peak_force = f;
                peak_time = t;
            }

            if t >= ramp_duration {
                hold_forces.push(f);
            }
        }

        events.push(GripEvent {
            time: peak_time,
            event_type: GripEventType::PeakForce,
            force: peak_force,
        });

        events.push(GripEvent {
            time: duration,
            event_type: GripEventType::TrialEnd,
            force: *force.last().unwrap_or(&0.0),
        });

        let force_rate = self.calculate_force_rate(&time, &force);

        let mean_sustained = if !hold_forces.is_empty() {
            hold_forces.iter().sum::<f64>() / hold_forces.len() as f64
        } else {
            0.0
        };

        let force_variability = if !hold_forces.is_empty() && mean_sustained > 0.0 {
            let variance = hold_forces.iter()
                .map(|f| (f - mean_sustained).powi(2))
                .sum::<f64>() / hold_forces.len() as f64;
            (variance.sqrt() / mean_sustained) * 100.0
        } else {
            0.0
        };

        let final_force = *force.last().unwrap_or(&0.0);
        let fatigue_index = ((peak_force - final_force) / peak_force * 100.0).max(0.0);

        let ground_truth = GripGroundTruth {
            peak_force,
            time_to_peak: peak_time,
            peak_rfd: force_rate.iter().cloned().fold(0.0_f64, f64::max),
            mean_sustained_force: mean_sustained,
            force_variability,
            fatigue_index,
            impulse: force.iter().sum::<f64>() * dt,
            events,
            pathology: None,
        };

        GripOutput {
            time,
            force,
            force_rate,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate rapid grip-release sequence
    ///
    /// Tests motor control and coordination
    pub fn generate_rapid_sequence(
        &mut self,
        n_repetitions: usize,
        target_percent_mvc: f64,
    ) -> GripOutput {
        let cycle_duration = 1.0; // 1 second per grip-release cycle
        let total_duration = n_repetitions as f64 * cycle_duration;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let effective_mvc = self.config.mvc_newtons
            * self.config.age_factor
            * self.config.sex_factor;
        let target_force = effective_mvc * target_percent_mvc;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        let noise_dist = Normal::new(0.0, self.config.noise_level * target_force).unwrap();

        events.push(GripEvent {
            time: 0.0,
            event_type: GripEventType::TrialStart,
            force: 0.0,
        });

        let mut peak_forces = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let cycle_phase = (t % cycle_duration) / cycle_duration;

            // Bell-shaped force profile within each cycle
            let base_force = if cycle_phase < 0.5 {
                // Grip phase
                let grip_progress = cycle_phase / 0.5;
                target_force * (std::f64::consts::PI * grip_progress).sin()
            } else {
                // Release phase
                let release_progress = (cycle_phase - 0.5) / 0.5;
                target_force * (std::f64::consts::PI * (1.0 - release_progress)).sin().max(0.0)
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);

            // Track peak of each cycle
            let cycle_num = (t / cycle_duration) as usize;
            if cycle_num >= peak_forces.len() {
                peak_forces.push(0.0_f64);
            }
            if f > peak_forces[cycle_num] {
                peak_forces[cycle_num] = f;
            }
        }

        let peak_force = peak_forces.iter().cloned().fold(0.0_f64, f64::max);
        let mean_peak = peak_forces.iter().sum::<f64>() / peak_forces.len() as f64;

        events.push(GripEvent {
            time: total_duration,
            event_type: GripEventType::TrialEnd,
            force: *force.last().unwrap_or(&0.0),
        });

        let force_rate = self.calculate_force_rate(&time, &force);

        let ground_truth = GripGroundTruth {
            peak_force,
            time_to_peak: 0.25, // Peak at middle of grip phase
            peak_rfd: force_rate.iter().cloned().fold(0.0_f64, f64::max),
            mean_sustained_force: mean_peak,
            force_variability: if mean_peak > 0.0 {
                let variance = peak_forces.iter()
                    .map(|f| (f - mean_peak).powi(2))
                    .sum::<f64>() / peak_forces.len() as f64;
                (variance.sqrt() / mean_peak) * 100.0
            } else {
                0.0
            },
            fatigue_index: if !peak_forces.is_empty() && peak_forces[0] > 0.0 {
                ((peak_forces[0] - *peak_forces.last().unwrap()) / peak_forces[0] * 100.0).max(0.0)
            } else {
                0.0
            },
            impulse: force.iter().sum::<f64>() * dt,
            events,
            pathology: None,
        };

        GripOutput {
            time,
            force,
            force_rate,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate bilateral grip comparison
    ///
    /// Returns (dominant_hand, non_dominant_hand) outputs
    pub fn generate_bilateral_mvc(&mut self, trial_duration: f64) -> (GripOutput, GripOutput) {
        // Generate dominant hand
        let dominant = self.generate_mvc_trial(trial_duration);

        // Adjust for non-dominant hand (typically 10% weaker)
        let original_mvc = self.config.mvc_newtons;
        self.config.mvc_newtons = original_mvc / self.config.dominance_factor;

        let non_dominant = self.generate_mvc_trial(trial_duration);

        // Restore config
        self.config.mvc_newtons = original_mvc;

        (dominant, non_dominant)
    }

    /// Generate pathological grip pattern
    pub fn generate_pathological(
        &mut self,
        pathology: PathologicalGrip,
        trial_duration: f64,
    ) -> GripOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (trial_duration * self.config.sample_rate) as usize;

        let effective_mvc = self.config.mvc_newtons
            * self.config.age_factor
            * self.config.sex_factor;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        let noise_dist = Normal::new(0.0, self.config.noise_level * effective_mvc).unwrap();

        events.push(GripEvent {
            time: 0.0,
            event_type: GripEventType::TrialStart,
            force: 0.0,
        });

        let ramp_up_end = 0.3;
        let hold_end = 0.6;
        let ramp_down_end = 0.9;

        let mut peak_force = 0.0_f64;
        let mut peak_time = 0.0;
        let mut hold_forces = Vec::new();

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let phase_fraction = t / trial_duration;

            // Base force profile (same as MVC)
            let mut base_force = if phase_fraction < ramp_up_end {
                let ramp_progress = phase_fraction / ramp_up_end;
                let sigmoid = 1.0 / (1.0 + (-12.0 * (ramp_progress - 0.5)).exp());
                effective_mvc * sigmoid
            } else if phase_fraction < hold_end {
                effective_mvc
            } else if phase_fraction < ramp_down_end {
                let ramp_progress = (phase_fraction - hold_end) / (ramp_down_end - hold_end);
                let sigmoid = 1.0 / (1.0 + (12.0 * (ramp_progress - 0.5)).exp());
                effective_mvc * sigmoid
            } else {
                0.0
            };

            // Apply pathological modifications
            base_force = match pathology {
                PathologicalGrip::Weakness { severity } => {
                    base_force * (1.0 - severity)
                }
                PathologicalGrip::RapidFatigue { fatigue_rate } => {
                    if phase_fraction >= ramp_up_end && phase_fraction < hold_end {
                        let hold_time = (phase_fraction - ramp_up_end) * trial_duration;
                        base_force * (-fatigue_rate * hold_time).exp()
                    } else {
                        base_force
                    }
                }
                PathologicalGrip::Tremor { frequency, amplitude } => {
                    let tremor = amplitude * effective_mvc
                        * (2.0 * std::f64::consts::PI * frequency * t).sin();
                    base_force + tremor
                }
                PathologicalGrip::CogwheelRigidity { frequency } => {
                    // Sawtooth-like resistance pattern
                    let phase = (t * frequency) % 1.0;
                    let cogwheel = if phase < 0.5 { phase * 2.0 } else { 2.0 - phase * 2.0 };
                    base_force * (0.8 + 0.4 * cogwheel)
                }
                PathologicalGrip::SlowRelease { time_constant } => {
                    if phase_fraction >= hold_end {
                        let release_time = (phase_fraction - hold_end) * trial_duration;
                        effective_mvc * (-release_time / time_constant).exp()
                    } else {
                        base_force
                    }
                }
                PathologicalGrip::Myotonia { delay } => {
                    if phase_fraction >= hold_end {
                        let release_time = (phase_fraction - hold_end) * trial_duration;
                        if release_time < delay {
                            effective_mvc // Delayed relaxation
                        } else {
                            let adjusted_time = release_time - delay;
                            let release_duration = (ramp_down_end - hold_end) * trial_duration;
                            effective_mvc * (1.0 - adjusted_time / release_duration).max(0.0)
                        }
                    } else {
                        base_force
                    }
                }
                PathologicalGrip::Spasticity { gain } => {
                    // Higher resistance during movement phases
                    if phase_fraction < ramp_up_end ||
                       (phase_fraction >= hold_end && phase_fraction < ramp_down_end) {
                        base_force * (1.0 + gain)
                    } else {
                        base_force
                    }
                }
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);

            if f > peak_force {
                peak_force = f;
                peak_time = t;
            }

            if phase_fraction >= ramp_up_end && phase_fraction < hold_end {
                hold_forces.push(f);
            }
        }

        events.push(GripEvent {
            time: peak_time,
            event_type: GripEventType::PeakForce,
            force: peak_force,
        });

        events.push(GripEvent {
            time: trial_duration,
            event_type: GripEventType::TrialEnd,
            force: *force.last().unwrap_or(&0.0),
        });

        let force_rate = self.calculate_force_rate(&time, &force);

        let mean_sustained = if !hold_forces.is_empty() {
            hold_forces.iter().sum::<f64>() / hold_forces.len() as f64
        } else {
            0.0
        };

        let force_variability = if !hold_forces.is_empty() && mean_sustained > 0.0 {
            let variance = hold_forces.iter()
                .map(|f| (f - mean_sustained).powi(2))
                .sum::<f64>() / hold_forces.len() as f64;
            (variance.sqrt() / mean_sustained) * 100.0
        } else {
            0.0
        };

        let ground_truth = GripGroundTruth {
            peak_force,
            time_to_peak: peak_time,
            peak_rfd: force_rate.iter().cloned().fold(0.0_f64, f64::max),
            mean_sustained_force: mean_sustained,
            force_variability,
            fatigue_index: ((peak_force - mean_sustained) / peak_force * 100.0).max(0.0),
            impulse: force.iter().sum::<f64>() * dt,
            events,
            pathology: Some(pathology),
        };

        GripOutput {
            time,
            force,
            force_rate,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Calculate force rate (derivative)
    fn calculate_force_rate(&self, time: &[f64], force: &[f64]) -> Vec<f64> {
        if force.len() < 2 {
            return vec![0.0; force.len()];
        }

        let mut rate = Vec::with_capacity(force.len());
        rate.push(0.0); // First point

        for i in 1..force.len() {
            let dt = time[i] - time[i - 1];
            if dt > 0.0 {
                rate.push((force[i] - force[i - 1]) / dt);
            } else {
                rate.push(0.0);
            }
        }

        rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mvc_trial() {
        let config = GripConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = GripGenerator::new(config);
        let output = gen.generate_mvc_trial(5.0);

        assert!(!output.force.is_empty());
        assert!(output.ground_truth.peak_force > 0.0);
        assert!(output.ground_truth.time_to_peak > 0.0);
        assert!(output.ground_truth.peak_rfd > 0.0);
    }

    #[test]
    fn test_sustained_grip() {
        let config = GripConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = GripGenerator::new(config);
        let output = gen.generate_sustained_grip(30.0, 0.5);

        assert!(!output.force.is_empty());
        assert!(output.ground_truth.fatigue_index >= 0.0);
    }

    #[test]
    fn test_rapid_sequence() {
        let config = GripConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = GripGenerator::new(config);
        let output = gen.generate_rapid_sequence(10, 0.5);

        assert!(!output.force.is_empty());
        assert!(output.ground_truth.force_variability >= 0.0);
    }

    #[test]
    fn test_bilateral() {
        let config = GripConfig {
            seed: Some(42),
            dominance_factor: 1.1,
            ..Default::default()
        };
        let mut gen = GripGenerator::new(config);
        let (dominant, non_dominant) = gen.generate_bilateral_mvc(5.0);

        // Dominant hand should be stronger
        assert!(dominant.ground_truth.peak_force > non_dominant.ground_truth.peak_force * 0.95);
    }

    #[test]
    fn test_pathological_weakness() {
        let config = GripConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = GripGenerator::new(config);

        let normal = gen.generate_mvc_trial(5.0);
        let weak = gen.generate_pathological(
            PathologicalGrip::Weakness { severity: 0.5 },
            5.0,
        );

        // Weak grip should have lower peak force
        assert!(weak.ground_truth.peak_force < normal.ground_truth.peak_force);
    }

    #[test]
    fn test_pathological_tremor() {
        let config = GripConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut gen = GripGenerator::new(config);

        let tremor = gen.generate_pathological(
            PathologicalGrip::Tremor {
                frequency: 6.0,
                amplitude: 0.1,
            },
            5.0,
        );

        // Tremor should increase variability
        assert!(tremor.ground_truth.force_variability > 0.0);
    }
}
