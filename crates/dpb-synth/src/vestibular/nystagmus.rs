//! Nystagmus Signal Generator
//!
//! Generates various nystagmus patterns:
//! - Spontaneous nystagmus
//! - Gaze-evoked nystagmus
//! - Positional nystagmus (BPPV)
//! - Optokinetic nystagmus
//! - Downbeat/upbeat nystagmus

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for nystagmus generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NystagmusConfig {
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Noise level
    pub noise_level: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for NystagmusConfig {
    fn default() -> Self {
        Self {
            sample_rate: 500.0,
            noise_level: 0.05,
            seed: None,
        }
    }
}

/// Types of nystagmus
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NystagmusType {
    /// Spontaneous (vestibular imbalance)
    Spontaneous,
    /// Gaze-evoked (cerebellar/brainstem)
    GazeEvoked,
    /// Positional (BPPV)
    Positional,
    /// Optokinetic
    Optokinetic,
    /// Downbeat (craniocervical junction)
    Downbeat,
    /// Upbeat (medullary/pontine)
    Upbeat,
    /// Periodic alternating
    PeriodicAlternating,
    /// Congenital
    Congenital,
}

/// Direction of nystagmus (quick phase direction)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NystagmusDirection {
    Right,
    Left,
    Up,
    Down,
    Torsional { clockwise: bool },
    Mixed { horizontal: f64, vertical: f64, torsional: f64 },
}

/// Output from nystagmus generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NystagmusOutput {
    /// Time (s)
    pub time: Vec<f64>,
    /// Horizontal eye position (deg)
    pub horizontal_position: Vec<f64>,
    /// Vertical eye position (deg)
    pub vertical_position: Vec<f64>,
    /// Horizontal velocity (deg/s)
    pub horizontal_velocity: Vec<f64>,
    /// Vertical velocity (deg/s)
    pub vertical_velocity: Vec<f64>,
    /// Ground truth
    pub ground_truth: NystagmusGroundTruth,
    /// Configuration
    pub config: NystagmusConfig,
}

/// Ground truth for nystagmus
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NystagmusGroundTruth {
    /// Type of nystagmus
    pub nystagmus_type: NystagmusType,
    /// Direction
    pub direction: NystagmusDirection,
    /// Slow phase velocity (deg/s)
    pub slow_phase_velocity: f64,
    /// Beat frequency (Hz)
    pub beat_frequency: f64,
    /// Amplitude (deg)
    pub amplitude: f64,
    /// Individual beat information
    pub beats: Vec<BeatInfo>,
    /// Intensity grade (1-3)
    pub intensity: u8,
    /// Suppressed by fixation
    pub fixation_suppression: bool,
}

/// Information about individual nystagmus beat
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeatInfo {
    /// Beat onset time (s)
    pub onset_time: f64,
    /// Slow phase duration (ms)
    pub slow_phase_duration: f64,
    /// Quick phase duration (ms)
    pub quick_phase_duration: f64,
    /// Slow phase velocity (deg/s)
    pub slow_phase_velocity: f64,
    /// Quick phase velocity (deg/s)
    pub quick_phase_velocity: f64,
    /// Amplitude (deg)
    pub amplitude: f64,
}

/// Nystagmus generator
pub struct NystagmusGenerator {
    config: NystagmusConfig,
    rng: StdRng,
}

impl NystagmusGenerator {
    /// Create new nystagmus generator
    pub fn new(config: NystagmusConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Self { config, rng }
    }

    /// Generate spontaneous nystagmus
    pub fn generate_spontaneous(
        &mut self,
        direction: NystagmusDirection,
        slow_phase_velocity: f64,
        duration: f64,
    ) -> NystagmusOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut horizontal_position = Vec::with_capacity(n_samples);
        let mut vertical_position = Vec::with_capacity(n_samples);
        let mut beats = Vec::new();

        // Direction components
        let (h_component, v_component) = match direction {
            NystagmusDirection::Right => (1.0, 0.0),
            NystagmusDirection::Left => (-1.0, 0.0),
            NystagmusDirection::Up => (0.0, 1.0),
            NystagmusDirection::Down => (0.0, -1.0),
            NystagmusDirection::Torsional { .. } => (0.0, 0.0),
            NystagmusDirection::Mixed { horizontal, vertical, .. } => (horizontal, vertical),
        };

        let noise_dist = Normal::new(0.0, self.config.noise_level).unwrap();

        // Nystagmus parameters
        let beat_amplitude = 5.0 + slow_phase_velocity * 0.1; // deg
        let beat_period = beat_amplitude / slow_phase_velocity; // s

        let mut current_h = 0.0;
        let mut current_v = 0.0;
        let mut phase_in_beat = 0.0;
        let mut beat_start = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            phase_in_beat += dt;

            // Slow phase (drift away from center)
            let slow_phase_duration = beat_period * 0.85;
            let quick_phase_duration = beat_period * 0.15;

            if phase_in_beat < slow_phase_duration {
                // Slow phase - linear drift
                current_h -= h_component * slow_phase_velocity * dt;
                current_v -= v_component * slow_phase_velocity * dt;
            } else if phase_in_beat < beat_period {
                // Quick phase - rapid return
                let quick_progress = (phase_in_beat - slow_phase_duration) / quick_phase_duration;
                let quick_factor = (PI * quick_progress).sin();
                current_h += h_component * beat_amplitude * quick_factor * dt / quick_phase_duration;
                current_v += v_component * beat_amplitude * quick_factor * dt / quick_phase_duration;
            } else {
                // New beat
                beats.push(BeatInfo {
                    onset_time: beat_start,
                    slow_phase_duration: slow_phase_duration * 1000.0,
                    quick_phase_duration: quick_phase_duration * 1000.0,
                    slow_phase_velocity,
                    quick_phase_velocity: beat_amplitude / quick_phase_duration,
                    amplitude: beat_amplitude,
                });
                phase_in_beat = 0.0;
                beat_start = t;
            }

            let noise: f64 = self.rng.sample(noise_dist);
            horizontal_position.push(current_h + noise);
            vertical_position.push(current_v + noise * 0.5);
        }

        let horizontal_velocity = self.calculate_velocity(&time, &horizontal_position);
        let vertical_velocity = self.calculate_velocity(&time, &vertical_position);

        let intensity = if slow_phase_velocity < 5.0 {
            1
        } else if slow_phase_velocity < 15.0 {
            2
        } else {
            3
        };

        let ground_truth = NystagmusGroundTruth {
            nystagmus_type: NystagmusType::Spontaneous,
            direction,
            slow_phase_velocity,
            beat_frequency: 1.0 / beat_period,
            amplitude: beat_amplitude,
            beats,
            intensity,
            fixation_suppression: true, // Peripheral nystagmus suppresses with fixation
        };

        NystagmusOutput {
            time,
            horizontal_position,
            vertical_position,
            horizontal_velocity,
            vertical_velocity,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate gaze-evoked nystagmus
    pub fn generate_gaze_evoked(
        &mut self,
        gaze_position: f64, // Horizontal gaze angle (deg)
        duration: f64,
    ) -> NystagmusOutput {
        // Gaze-evoked nystagmus increases with eccentricity
        let eccentricity = gaze_position.abs();
        let threshold = 20.0; // deg

        let slow_phase_velocity = if eccentricity > threshold {
            (eccentricity - threshold) * 0.5
        } else {
            0.0
        };

        let direction = if gaze_position > 0.0 {
            NystagmusDirection::Right
        } else {
            NystagmusDirection::Left
        };

        let mut output = self.generate_spontaneous(direction, slow_phase_velocity, duration);
        output.ground_truth.nystagmus_type = NystagmusType::GazeEvoked;
        output.ground_truth.fixation_suppression = false; // Central origin

        // Add gaze position offset
        for pos in &mut output.horizontal_position {
            *pos += gaze_position;
        }

        output
    }

    /// Generate BPPV-type positional nystagmus
    pub fn generate_bppv(
        &mut self,
        canal: BppvCanal,
        duration: f64,
    ) -> NystagmusOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        // BPPV characteristics by canal
        let (direction, peak_spv, latency, duration_s, torsional) = match canal {
            BppvCanal::PosteriorRight => (NystagmusDirection::Up, 30.0, 1.0, 30.0, true),
            BppvCanal::PosteriorLeft => (NystagmusDirection::Up, 30.0, 1.0, 30.0, true),
            BppvCanal::HorizontalRightGeotropic => (NystagmusDirection::Right, 60.0, 0.5, 60.0, false),
            BppvCanal::HorizontalLeftGeotropic => (NystagmusDirection::Left, 60.0, 0.5, 60.0, false),
            BppvCanal::AnteriorRight => (NystagmusDirection::Down, 20.0, 1.0, 20.0, true),
            BppvCanal::AnteriorLeft => (NystagmusDirection::Down, 20.0, 1.0, 20.0, true),
        };

        let mut time = Vec::with_capacity(n_samples);
        let mut horizontal_position = Vec::with_capacity(n_samples);
        let mut vertical_position = Vec::with_capacity(n_samples);
        let mut beats = Vec::new();

        let noise_dist = Normal::new(0.0, self.config.noise_level).unwrap();

        let (h_comp, v_comp) = match direction {
            NystagmusDirection::Right => (1.0, 0.0),
            NystagmusDirection::Left => (-1.0, 0.0),
            NystagmusDirection::Up => (0.0, 1.0),
            NystagmusDirection::Down => (0.0, -1.0),
            _ => (0.0, 0.0),
        };

        let mut current_h = 0.0;
        let mut current_v = 0.0;

        for i in 0..n_samples {
            let t = i as f64 * dt;
            time.push(t);

            // BPPV time course: latency, crescendo, decrescendo
            let spv = if t < latency {
                0.0
            } else {
                let active_time = t - latency;
                let peak_time = duration_s * 0.3;
                if active_time < peak_time {
                    // Crescendo
                    peak_spv * (active_time / peak_time)
                } else if active_time < duration_s {
                    // Decrescendo
                    peak_spv * (1.0 - (active_time - peak_time) / (duration_s - peak_time))
                } else {
                    0.0
                }
            };

            // Generate sawtooth nystagmus pattern
            if spv > 1.0 {
                let beat_period = 5.0 / spv;
                let phase = (t % beat_period) / beat_period;

                if phase < 0.85 {
                    current_h -= h_comp * spv * dt;
                    current_v -= v_comp * spv * dt;
                } else {
                    let quick_fraction = (phase - 0.85) / 0.15;
                    current_h += h_comp * 5.0 * quick_fraction * dt / (beat_period * 0.15);
                    current_v += v_comp * 5.0 * quick_fraction * dt / (beat_period * 0.15);
                }
            }

            let noise: f64 = self.rng.sample(noise_dist);
            horizontal_position.push(current_h + noise);
            vertical_position.push(current_v + noise);
        }

        let horizontal_velocity = self.calculate_velocity(&time, &horizontal_position);
        let vertical_velocity = self.calculate_velocity(&time, &vertical_position);

        let ground_truth = NystagmusGroundTruth {
            nystagmus_type: NystagmusType::Positional,
            direction,
            slow_phase_velocity: peak_spv,
            beat_frequency: peak_spv / 5.0,
            amplitude: 5.0,
            beats,
            intensity: if peak_spv < 15.0 { 1 } else if peak_spv < 30.0 { 2 } else { 3 },
            fixation_suppression: true,
        };

        NystagmusOutput {
            time,
            horizontal_position,
            vertical_position,
            horizontal_velocity,
            vertical_velocity,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate optokinetic nystagmus
    pub fn generate_optokinetic(
        &mut self,
        stimulus_velocity: f64, // deg/s
        duration: f64,
    ) -> NystagmusOutput {
        // OKN gain typically 0.7-0.9
        let okn_gain = 0.8;
        let slow_phase_velocity = stimulus_velocity * okn_gain;

        let direction = if stimulus_velocity > 0.0 {
            NystagmusDirection::Right
        } else {
            NystagmusDirection::Left
        };

        let mut output = self.generate_spontaneous(direction, slow_phase_velocity.abs(), duration);
        output.ground_truth.nystagmus_type = NystagmusType::Optokinetic;
        output.ground_truth.fixation_suppression = false;

        output
    }

    /// Generate downbeat nystagmus
    pub fn generate_downbeat(
        &mut self,
        slow_phase_velocity: f64,
        duration: f64,
    ) -> NystagmusOutput {
        let mut output = self.generate_spontaneous(
            NystagmusDirection::Down,
            slow_phase_velocity,
            duration,
        );
        output.ground_truth.nystagmus_type = NystagmusType::Downbeat;
        output.ground_truth.fixation_suppression = false;
        output
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
}

/// BPPV canal types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BppvCanal {
    PosteriorRight,
    PosteriorLeft,
    HorizontalRightGeotropic,
    HorizontalLeftGeotropic,
    AnteriorRight,
    AnteriorLeft,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spontaneous_nystagmus() {
        let config = NystagmusConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = NystagmusGenerator::new(config);
        let output = generator.generate_spontaneous(NystagmusDirection::Right, 10.0, 10.0);

        assert!(!output.horizontal_position.is_empty());
        assert!(!output.ground_truth.beats.is_empty());
        assert_eq!(output.ground_truth.intensity, 2);
    }

    #[test]
    fn test_gaze_evoked() {
        let config = NystagmusConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = NystagmusGenerator::new(config);

        // Below threshold - no nystagmus
        let output_low = generator.generate_gaze_evoked(15.0, 5.0);
        assert!(output_low.ground_truth.slow_phase_velocity < 1.0);

        // Above threshold - nystagmus present
        let output_high = generator.generate_gaze_evoked(35.0, 5.0);
        assert!(output_high.ground_truth.slow_phase_velocity > 0.0);
    }

    #[test]
    fn test_bppv() {
        let config = NystagmusConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = NystagmusGenerator::new(config);
        let output = generator.generate_bppv(BppvCanal::PosteriorRight, 60.0);

        assert_eq!(output.ground_truth.nystagmus_type, NystagmusType::Positional);
        assert!(output.ground_truth.slow_phase_velocity > 0.0);
    }

    #[test]
    fn test_optokinetic() {
        let config = NystagmusConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = NystagmusGenerator::new(config);
        let output = generator.generate_optokinetic(30.0, 10.0);

        // OKN should have high SPV following stimulus
        assert!(output.ground_truth.slow_phase_velocity > 20.0);
    }
}
