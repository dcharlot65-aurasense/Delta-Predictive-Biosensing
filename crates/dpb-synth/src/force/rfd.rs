//! Rate of Force Development (RFD) Signal Generator
//!
//! Generates explosive force production signals for:
//! - Isometric rapid force tasks
//! - Countermovement jump force profiles
//! - Reactive strength assessments
//! - Neuromuscular function testing
//! - Pathological patterns (delayed onset, reduced RFD)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};

/// Configuration for RFD generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfdConfig {
    /// Sampling rate in Hz (high rate needed for RFD accuracy)
    pub sample_rate: f64,
    /// Maximum force capacity in Newtons
    pub max_force: f64,
    /// Time to peak force in seconds (typical: 0.1-0.4s)
    pub time_to_peak: f64,
    /// Early RFD window (0-50ms) scaling factor
    pub early_rfd_factor: f64,
    /// Late RFD window (100-200ms) scaling factor
    pub late_rfd_factor: f64,
    /// Noise level (0.0-1.0)
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for RfdConfig {
    fn default() -> Self {
        Self {
            sample_rate: 2000.0, // High rate for RFD
            max_force: 500.0,
            time_to_peak: 0.25,
            early_rfd_factor: 1.0,
            late_rfd_factor: 1.0,
            noise_level: 0.02,
            seed: None,
        }
    }
}

/// Output from RFD generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfdOutput {
    /// Time points in seconds
    pub time: Vec<f64>,
    /// Force in Newtons
    pub force: Vec<f64>,
    /// Instantaneous RFD (dF/dt) in N/s
    pub rfd: Vec<f64>,
    /// Ground truth annotations
    pub ground_truth: RfdGroundTruth,
    /// Configuration used
    pub config: RfdConfig,
}

/// Ground truth for RFD signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RfdGroundTruth {
    /// Peak force achieved (N)
    pub peak_force: f64,
    /// Time to peak force (s)
    pub time_to_peak: f64,
    /// Peak RFD (N/s)
    pub peak_rfd: f64,
    /// Time to peak RFD (s)
    pub time_to_peak_rfd: f64,
    /// RFD at 0-50ms window (N/s)
    pub rfd_0_50: f64,
    /// RFD at 0-100ms window (N/s)
    pub rfd_0_100: f64,
    /// RFD at 0-200ms window (N/s)
    pub rfd_0_200: f64,
    /// RFD at 100-200ms window (N/s)
    pub rfd_100_200: f64,
    /// Force at 50ms (N)
    pub force_50ms: f64,
    /// Force at 100ms (N)
    pub force_100ms: f64,
    /// Force at 200ms (N)
    pub force_200ms: f64,
    /// Electromechanical delay (EMD) in ms
    pub electromechanical_delay: f64,
    /// Force onset time (s)
    pub onset_time: f64,
    /// Impulse to peak (N·s)
    pub impulse_to_peak: f64,
    /// Applied pathology if any
    pub pathology: Option<PathologicalRfd>,
    /// Task type
    pub task_type: RfdTaskType,
}

/// Types of RFD tasks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RfdTaskType {
    /// Isometric rapid contraction
    IsometricRapid,
    /// Countermovement jump
    CountermovementJump,
    /// Squat jump (no countermovement)
    SquatJump,
    /// Drop jump reactive strength
    DropJump,
    /// Ballistic push
    BallisticPush,
}

/// Pathological RFD patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum PathologicalRfd {
    /// Delayed neural activation
    DelayedOnset { delay_ms: f64 },
    /// Reduced early RFD (neural deficit)
    ReducedEarlyRfd { reduction: f64 },
    /// Reduced late RFD (contractile deficit)
    ReducedLateRfd { reduction: f64 },
    /// Prolonged time to peak
    SlowContraction { factor: f64 },
    /// Force plateau before peak
    ForcePlateau { plateau_level: f64 },
    /// Inconsistent force development
    Inconsistent { variability: f64 },
    /// Fatigue-related RFD decline
    FatigueDecline { rate: f64 },
}

/// RFD signal generator
pub struct RfdGenerator {
    config: RfdConfig,
    rng: StdRng,
}

impl RfdGenerator {
    /// Create a new RFD generator
    pub fn new(config: RfdConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Generate isometric rapid force production
    ///
    /// Standard explosive isometric contraction task
    pub fn generate_isometric_rapid(&mut self, trial_duration: f64) -> RfdOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (trial_duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * self.config.max_force).unwrap();

        // Pre-stimulus baseline period
        let baseline_duration = 0.5;
        let onset_time = baseline_duration;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let base_force = if t < onset_time {
                // Baseline period
                0.0
            } else {
                // Explosive force rise - double exponential model
                let active_time = t - onset_time;
                let tau_rise = self.config.time_to_peak / 3.0; // Time constant

                // Early phase (neural drive)
                let early_component = self.config.early_rfd_factor
                    * (1.0 - (-active_time / (tau_rise * 0.5)).exp());

                // Late phase (contractile)
                let late_component = self.config.late_rfd_factor
                    * (1.0 - (-active_time / tau_rise).exp());

                // Combined force
                let force_fraction = (early_component * 0.4 + late_component * 0.6).min(1.0);
                self.config.max_force * force_fraction
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);
        }

        let rfd = self.calculate_rfd(&time, &force);
        let ground_truth = self.calculate_ground_truth(&time, &force, &rfd, onset_time, RfdTaskType::IsometricRapid, None);

        RfdOutput {
            time,
            force,
            rfd,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate countermovement jump force profile
    ///
    /// Includes unweighting, braking, and propulsion phases
    pub fn generate_cmj(&mut self, body_mass: f64) -> RfdOutput {
        let dt = 1.0 / self.config.sample_rate;
        let total_duration = 2.0; // 2 second trial
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let body_weight = body_mass * 9.81; // N

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * body_weight).unwrap();

        // Phase timing
        let quiet_stand_end = 0.5;
        let unweighting_end = 0.8;
        let braking_end = 1.1;
        let propulsion_end = 1.4;
        let flight_end = 1.7;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let base_force = if t < quiet_stand_end {
                // Quiet standing
                body_weight
            } else if t < unweighting_end {
                // Unweighting phase (countermovement down)
                let phase_progress = (t - quiet_stand_end) / (unweighting_end - quiet_stand_end);
                let unweight_factor = 1.0 - 0.6 * (std::f64::consts::PI * phase_progress).sin();
                body_weight * unweight_factor
            } else if t < braking_end {
                // Braking phase (deceleration at bottom)
                let phase_progress = (t - unweighting_end) / (braking_end - unweighting_end);
                let brake_factor = 0.4 + 1.6 * (std::f64::consts::PI * phase_progress * 0.5).sin();
                body_weight * brake_factor
            } else if t < propulsion_end {
                // Propulsion phase (push off)
                let phase_progress = (t - braking_end) / (propulsion_end - braking_end);
                // Peak force around 2.5x body weight
                let prop_factor = 2.0 + 0.5 * (std::f64::consts::PI * phase_progress).sin();
                let decay = 1.0 - phase_progress.powi(2);
                body_weight * prop_factor * decay
            } else if t < flight_end {
                // Flight phase
                0.0
            } else {
                // Landing
                let landing_progress = (t - flight_end) / (total_duration - flight_end);
                let landing_force = 3.0 * (1.0 - landing_progress).max(0.0);
                body_weight * (1.0 + landing_force)
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);
        }

        let rfd = self.calculate_rfd(&time, &force);

        // Find propulsion onset for ground truth
        let onset_idx = time.iter().position(|&t| t >= braking_end).unwrap_or(0);
        let onset_time = time[onset_idx];

        let ground_truth = self.calculate_ground_truth(&time, &force, &rfd, onset_time, RfdTaskType::CountermovementJump, None);

        RfdOutput {
            time,
            force,
            rfd,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate squat jump force profile (no countermovement)
    pub fn generate_squat_jump(&mut self, body_mass: f64) -> RfdOutput {
        let dt = 1.0 / self.config.sample_rate;
        let total_duration = 1.5;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let body_weight = body_mass * 9.81;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * body_weight).unwrap();

        let squat_hold_end = 0.5;
        let propulsion_end = 0.9;
        let flight_end = 1.2;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let base_force = if t < squat_hold_end {
                // Static squat hold
                body_weight
            } else if t < propulsion_end {
                // Propulsion (no countermovement, so starts from isometric)
                let phase_progress = (t - squat_hold_end) / (propulsion_end - squat_hold_end);
                let prop_factor = 1.0 + 1.5 * (std::f64::consts::PI * phase_progress * 0.5).sin();
                let decay = 1.0 - phase_progress.powi(3);
                body_weight * prop_factor * decay
            } else if t < flight_end {
                0.0
            } else {
                // Landing
                let landing_progress = (t - flight_end) / (total_duration - flight_end);
                body_weight * (1.0 + 2.5 * (1.0 - landing_progress).max(0.0))
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);
        }

        let rfd = self.calculate_rfd(&time, &force);
        let ground_truth = self.calculate_ground_truth(&time, &force, &rfd, squat_hold_end, RfdTaskType::SquatJump, None);

        RfdOutput {
            time,
            force,
            rfd,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate drop jump reactive strength profile
    pub fn generate_drop_jump(&mut self, body_mass: f64, drop_height: f64) -> RfdOutput {
        let dt = 1.0 / self.config.sample_rate;
        let total_duration = 1.5;
        let n_samples = (total_duration * self.config.sample_rate) as usize;

        let body_weight = body_mass * 9.81;
        // Impact velocity from drop height
        let impact_velocity = (2.0 * 9.81 * drop_height).sqrt();
        // Peak impact force scales with velocity
        let peak_impact_factor = 3.0 + impact_velocity * 0.3;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * body_weight).unwrap();

        let drop_phase_end = 0.3;
        let impact_end = 0.5;
        let propulsion_end = 0.8;
        let flight_end = 1.2;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let base_force = if t < drop_phase_end {
                // Free fall (no force)
                0.0
            } else if t < impact_end {
                // Impact absorption
                let phase_progress = (t - drop_phase_end) / (impact_end - drop_phase_end);
                let impact = peak_impact_factor * (std::f64::consts::PI * phase_progress).sin();
                body_weight * impact
            } else if t < propulsion_end {
                // Rapid propulsion (reactive)
                let phase_progress = (t - impact_end) / (propulsion_end - impact_end);
                let prop_factor = 2.5 * (1.0 - phase_progress.powi(2));
                body_weight * prop_factor
            } else if t < flight_end {
                0.0
            } else {
                // Final landing
                let landing_progress = (t - flight_end) / (total_duration - flight_end);
                body_weight * (1.0 + 2.0 * (1.0 - landing_progress).max(0.0))
            };

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);
        }

        let rfd = self.calculate_rfd(&time, &force);
        let ground_truth = self.calculate_ground_truth(&time, &force, &rfd, drop_phase_end, RfdTaskType::DropJump, None);

        RfdOutput {
            time,
            force,
            rfd,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate repeated RFD trials showing fatigue
    pub fn generate_fatigue_series(&mut self, n_trials: usize, rest_interval: f64) -> Vec<RfdOutput> {
        let mut outputs = Vec::with_capacity(n_trials);
        let trial_duration = 1.0;

        for trial_num in 0..n_trials {
            // Progressive fatigue
            let fatigue_factor = 1.0 - 0.3 * (trial_num as f64 / n_trials as f64);

            // Temporarily modify config for fatigue
            let original_max = self.config.max_force;
            let original_early = self.config.early_rfd_factor;
            let original_late = self.config.late_rfd_factor;

            self.config.max_force = original_max * fatigue_factor;
            self.config.early_rfd_factor = original_early * (fatigue_factor + 0.1).min(1.0);
            self.config.late_rfd_factor = original_late * fatigue_factor;

            let mut output = self.generate_isometric_rapid(trial_duration);

            // Adjust time for trial position
            let time_offset = trial_num as f64 * (trial_duration + rest_interval);
            for t in &mut output.time {
                *t += time_offset;
            }

            // Restore config
            self.config.max_force = original_max;
            self.config.early_rfd_factor = original_early;
            self.config.late_rfd_factor = original_late;

            outputs.push(output);
        }

        outputs
    }

    /// Generate pathological RFD pattern
    pub fn generate_pathological(
        &mut self,
        pathology: PathologicalRfd,
        trial_duration: f64,
    ) -> RfdOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (trial_duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut force = Vec::with_capacity(n_samples);

        let noise_dist = Normal::new(0.0, self.config.noise_level * self.config.max_force).unwrap();

        // Base onset time
        let mut onset_time = 0.5;

        // Adjust onset for delayed onset pathology
        if let PathologicalRfd::DelayedOnset { delay_ms } = pathology {
            onset_time += delay_ms / 1000.0;
        }

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            let mut base_force = if t < onset_time {
                0.0
            } else {
                let active_time = t - onset_time;
                let tau_rise = self.config.time_to_peak / 3.0;

                let early_component = self.config.early_rfd_factor
                    * (1.0 - (-active_time / (tau_rise * 0.5)).exp());
                let late_component = self.config.late_rfd_factor
                    * (1.0 - (-active_time / tau_rise).exp());

                let force_fraction = (early_component * 0.4 + late_component * 0.6).min(1.0);
                self.config.max_force * force_fraction
            };

            // Apply pathological modifications
            if t >= onset_time {
                let active_time = t - onset_time;

                base_force = match pathology {
                    PathologicalRfd::DelayedOnset { .. } => base_force, // Already handled

                    PathologicalRfd::ReducedEarlyRfd { reduction } => {
                        if active_time < 0.1 {
                            base_force * (1.0 - reduction)
                        } else {
                            base_force
                        }
                    }

                    PathologicalRfd::ReducedLateRfd { reduction } => {
                        if active_time >= 0.1 {
                            base_force * (1.0 - reduction * (active_time - 0.1).min(0.2) / 0.2)
                        } else {
                            base_force
                        }
                    }

                    PathologicalRfd::SlowContraction { factor } => {
                        let slowed_time = active_time / factor;
                        let tau_rise = self.config.time_to_peak / 3.0;
                        let force_fraction = (1.0 - (-slowed_time / tau_rise).exp()).min(1.0);
                        self.config.max_force * force_fraction
                    }

                    PathologicalRfd::ForcePlateau { plateau_level } => {
                        base_force.min(self.config.max_force * plateau_level)
                    }

                    PathologicalRfd::Inconsistent { variability } => {
                        let var_dist = Normal::new(1.0, variability).unwrap();
                        let var_factor: f64 = self.rng.sample(var_dist);
                        base_force * var_factor.max(0.5).min(1.5)
                    }

                    PathologicalRfd::FatigueDecline { rate } => {
                        base_force * (-rate * active_time).exp()
                    }
                };
            }

            let noise: f64 = self.rng.sample(noise_dist);
            let f = (base_force + noise).max(0.0);
            force.push(f);
        }

        let rfd = self.calculate_rfd(&time, &force);
        let ground_truth = self.calculate_ground_truth(
            &time, &force, &rfd, onset_time,
            RfdTaskType::IsometricRapid, Some(pathology)
        );

        RfdOutput {
            time,
            force,
            rfd,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Calculate instantaneous RFD
    fn calculate_rfd(&self, time: &[f64], force: &[f64]) -> Vec<f64> {
        if force.len() < 2 {
            return vec![0.0; force.len()];
        }

        let mut rfd = Vec::with_capacity(force.len());
        rfd.push(0.0);

        for i in 1..force.len() {
            let dt = time[i] - time[i - 1];
            if dt > 0.0 {
                rfd.push((force[i] - force[i - 1]) / dt);
            } else {
                rfd.push(0.0);
            }
        }

        rfd
    }

    /// Calculate ground truth metrics
    fn calculate_ground_truth(
        &self,
        time: &[f64],
        force: &[f64],
        rfd: &[f64],
        onset_time: f64,
        task_type: RfdTaskType,
        pathology: Option<PathologicalRfd>,
    ) -> RfdGroundTruth {
        let dt = 1.0 / self.config.sample_rate;

        // Find onset index (force > 5% of max)
        let threshold = self.config.max_force * 0.05;
        let onset_idx = force.iter()
            .position(|&f| f > threshold)
            .unwrap_or(0);
        let actual_onset = time.get(onset_idx).copied().unwrap_or(onset_time);

        // Peak force and time
        let (peak_idx, peak_force) = force.iter()
            .enumerate()
            .fold((0, 0.0_f64), |(max_i, max_f), (i, &f)| {
                if f > max_f { (i, f) } else { (max_i, max_f) }
            });
        let time_to_peak = time.get(peak_idx).copied().unwrap_or(0.0) - actual_onset;

        // Peak RFD
        let (peak_rfd_idx, peak_rfd) = rfd.iter()
            .enumerate()
            .fold((0, 0.0_f64), |(max_i, max_r), (i, &r)| {
                if r > max_r { (i, r) } else { (max_i, max_r) }
            });
        let time_to_peak_rfd = time.get(peak_rfd_idx).copied().unwrap_or(0.0) - actual_onset;

        // RFD windows (relative to onset)
        let get_force_at_time = |target_time: f64| -> f64 {
            let target_t = actual_onset + target_time;
            let idx = ((target_t * self.config.sample_rate) as usize).min(force.len() - 1);
            force.get(idx).copied().unwrap_or(0.0)
        };

        let force_at_onset = get_force_at_time(0.0);
        let force_50ms = get_force_at_time(0.05);
        let force_100ms = get_force_at_time(0.1);
        let force_200ms = get_force_at_time(0.2);

        let rfd_0_50 = if force_50ms > force_at_onset {
            (force_50ms - force_at_onset) / 0.05
        } else {
            0.0
        };

        let rfd_0_100 = if force_100ms > force_at_onset {
            (force_100ms - force_at_onset) / 0.1
        } else {
            0.0
        };

        let rfd_0_200 = if force_200ms > force_at_onset {
            (force_200ms - force_at_onset) / 0.2
        } else {
            0.0
        };

        let rfd_100_200 = if force_200ms > force_100ms {
            (force_200ms - force_100ms) / 0.1
        } else {
            0.0
        };

        // Impulse to peak
        let impulse_to_peak: f64 = force[onset_idx..=peak_idx.min(force.len() - 1)]
            .iter()
            .sum::<f64>() * dt;

        // Electromechanical delay (time from "stimulus" to force onset)
        let emd = (actual_onset - onset_time) * 1000.0; // Convert to ms

        RfdGroundTruth {
            peak_force,
            time_to_peak: time_to_peak.max(0.0),
            peak_rfd,
            time_to_peak_rfd: time_to_peak_rfd.max(0.0),
            rfd_0_50,
            rfd_0_100,
            rfd_0_200,
            rfd_100_200,
            force_50ms,
            force_100ms,
            force_200ms,
            electromechanical_delay: emd.max(0.0),
            onset_time: actual_onset,
            impulse_to_peak,
            pathology,
            task_type,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_isometric_rapid() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);
        let output = generator.generate_isometric_rapid(2.0);

        assert!(!output.force.is_empty());
        assert!(output.ground_truth.peak_force > 0.0);
        assert!(output.ground_truth.peak_rfd > 0.0);
        assert!(output.ground_truth.rfd_0_100 > 0.0);
    }

    #[test]
    fn test_cmj() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);
        let output = generator.generate_cmj(70.0);

        assert!(!output.force.is_empty());
        // CMJ should have flight phase (zero force)
        assert!(output.force.iter().any(|&f| f < 10.0));
    }

    #[test]
    fn test_squat_jump() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);
        let output = generator.generate_squat_jump(70.0);

        assert!(!output.force.is_empty());
        assert_eq!(output.ground_truth.task_type, RfdTaskType::SquatJump);
    }

    #[test]
    fn test_drop_jump() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);
        let output = generator.generate_drop_jump(70.0, 0.3);

        assert!(!output.force.is_empty());
        // Drop jump should have high impact forces
        let body_weight = 70.0 * 9.81;
        assert!(output.ground_truth.peak_force > body_weight * 2.0);
    }

    #[test]
    fn test_fatigue_series() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);
        let series = generator.generate_fatigue_series(5, 1.0);

        assert_eq!(series.len(), 5);
        // Later trials should show reduced peak force
        assert!(series[4].ground_truth.peak_force < series[0].ground_truth.peak_force);
    }

    #[test]
    fn test_pathological_delayed() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);

        let normal = generator.generate_isometric_rapid(2.0);
        let delayed = generator.generate_pathological(
            PathologicalRfd::DelayedOnset { delay_ms: 50.0 },
            2.0,
        );

        // Delayed onset should have later force onset
        assert!(delayed.ground_truth.onset_time > normal.ground_truth.onset_time);
    }

    #[test]
    fn test_pathological_reduced_early() {
        let config = RfdConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);

        let normal = generator.generate_isometric_rapid(2.0);
        let reduced = generator.generate_pathological(
            PathologicalRfd::ReducedEarlyRfd { reduction: 0.5 },
            2.0,
        );

        // Early RFD should be reduced
        assert!(reduced.ground_truth.rfd_0_50 < normal.ground_truth.rfd_0_50);
    }

    #[test]
    fn test_rfd_windows() {
        let config = RfdConfig {
            seed: Some(42),
            sample_rate: 2000.0,
            ..Default::default()
        };
        let mut generator = RfdGenerator::new(config);
        let output = generator.generate_isometric_rapid(2.0);

        // All RFD windows should be positive for normal trial
        assert!(output.ground_truth.rfd_0_50 >= 0.0);
        assert!(output.ground_truth.rfd_0_100 >= 0.0);
        assert!(output.ground_truth.rfd_0_200 >= 0.0);

        // Force should increase over time windows
        assert!(output.ground_truth.force_100ms >= output.ground_truth.force_50ms);
        assert!(output.ground_truth.force_200ms >= output.ground_truth.force_100ms);
    }
}
