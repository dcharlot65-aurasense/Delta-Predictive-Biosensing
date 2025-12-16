//! Finger tapping generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth, Event};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Normal finger tapping generator
pub struct NormalTappingGenerator;

#[derive(Debug, Clone)]
pub struct NormalTappingParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub frequency: f64,        // Hz (typically 3-5 Hz)
    pub amplitude: f64,        // cm
    pub regularity: f64,       // 0-1 (1 = perfect regularity)
}

impl SyntheticGenerator for NormalTappingGenerator {
    type Output = Vec<f64>; // finger separation distance over time
    type GroundTruth = SpatialGroundTruth;
    type Parameters = NormalTappingParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let timing_noise = Normal::new(0.0, (1.0 - params.regularity) * 0.1).unwrap();
        let amplitude_noise = Normal::new(1.0, (1.0 - params.regularity) * 0.1).unwrap();

        let mut separation = Vec::with_capacity(n_frames);
        let mut events = Vec::new();
        let mut cumulative_time = 0.0;

        for i in 0..n_frames {
            let t = i as f64 * dt;

            // Generate tapping motion (open/close cycle)
            let phase = (2.0 * PI * params.frequency * t + timing_noise.sample(&mut rng)).rem_euclid(2.0 * PI);
            let amp_mod = amplitude_noise.sample(&mut rng).max(0.1);

            // Distance: max amplitude when open, 0 when closed
            let distance = if phase < PI {
                // Opening phase
                params.amplitude * amp_mod * (phase / PI)
            } else {
                // Closing phase
                params.amplitude * amp_mod * (1.0 - (phase - PI) / PI)
            };

            separation.push(distance);

            // Detect tap events (contact)
            if distance < 0.1 && i > 0 && separation[i - 1] >= 0.1 {
                events.push(Event {
                    time: t,
                    event_type: "tap".to_string(),
                    amplitude: Some(params.amplitude * amp_mod),
                    attributes: HashMap::new(),
                });
            }
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("finger_flexion".to_string(), separation.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(separation, ground_truth, params.frame_rate)
            .with_metadata("tap_count".to_string(), events.len().to_string()))
    }

    fn default_params() -> Self::Parameters {
        NormalTappingParams {
            duration: 10.0,
            frame_rate: 60.0,
            frequency: 4.0,
            amplitude: 5.0,  // cm
            regularity: 0.9,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.regularity < 0.0 || params.regularity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("regularity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Bradykinetic tapping generator (slowed movements)
pub struct BradykineticTappingGenerator;

#[derive(Debug, Clone)]
pub struct BradykineticTappingParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub initial_frequency: f64,
    pub frequency_decay: f64,  // Hz per second
    pub initial_amplitude: f64,
    pub amplitude_decay: f64,  // cm per second
}

impl SyntheticGenerator for BradykineticTappingGenerator {
    type Output = Vec<f64>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = BradykineticTappingParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut separation = Vec::with_capacity(n_frames);
        let mut phase = 0.0;

        for i in 0..n_frames {
            let t = i as f64 * dt;

            // Decaying frequency and amplitude
            let freq = (params.initial_frequency - params.frequency_decay * t).max(0.5);
            let amp = (params.initial_amplitude - params.amplitude_decay * t).max(0.5);

            phase += 2.0 * PI * freq * dt;

            let distance = if phase.rem_euclid(2.0 * PI) < PI {
                amp * (phase.rem_euclid(PI) / PI)
            } else {
                amp * (1.0 - (phase.rem_euclid(2.0 * PI) - PI) / PI)
            };

            separation.push(distance);
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("finger_flexion".to_string(), separation.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(separation, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        BradykineticTappingParams {
            duration: 10.0,
            frame_rate: 60.0,
            initial_frequency: 4.0,
            frequency_decay: 0.2,
            initial_amplitude: 5.0,
            amplitude_decay: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

/// Amplitude decrement generator (progressive reduction)
pub struct AmplitudeDecrementGenerator;

#[derive(Debug, Clone)]
pub struct AmplitudeDecrementParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub frequency: f64,
    pub initial_amplitude: f64,
    pub final_amplitude_ratio: f64, // 0-1 (final/initial)
    pub decrement_profile: DecrementProfile,
}

#[derive(Debug, Clone)]
pub enum DecrementProfile {
    Linear,
    Exponential { tau: f64 },
    Stepwise { steps: usize },
}

impl SyntheticGenerator for AmplitudeDecrementGenerator {
    type Output = Vec<f64>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = AmplitudeDecrementParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;

        let mut separation = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 * dt;
            let progress = t / params.duration;

            // Calculate amplitude based on decrement profile
            let amplitude = match &params.decrement_profile {
                DecrementProfile::Linear => {
                    params.initial_amplitude *
                        (1.0 - progress * (1.0 - params.final_amplitude_ratio))
                }
                DecrementProfile::Exponential { tau } => {
                    let final_amp = params.initial_amplitude * params.final_amplitude_ratio;
                    final_amp + (params.initial_amplitude - final_amp) * (-t / tau).exp()
                }
                DecrementProfile::Stepwise { steps } => {
                    let step_idx = (progress * (*steps as f64)).floor() as usize;
                    let step_fraction = step_idx as f64 / *steps as f64;
                    params.initial_amplitude *
                        (1.0 - step_fraction * (1.0 - params.final_amplitude_ratio))
                }
            };

            // Generate tapping motion with current amplitude
            let phase = (2.0 * PI * params.frequency * t).rem_euclid(2.0 * PI);
            let distance = if phase < PI {
                amplitude * (phase / PI)
            } else {
                amplitude * (1.0 - (phase - PI) / PI)
            };

            separation.push(distance);
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("finger_flexion".to_string(), separation.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(separation, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        AmplitudeDecrementParams {
            duration: 10.0,
            frame_rate: 60.0,
            frequency: 4.0,
            initial_amplitude: 5.0,
            final_amplitude_ratio: 0.3,
            decrement_profile: DecrementProfile::Exponential { tau: 3.0 },
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.final_amplitude_ratio < 0.0 || params.final_amplitude_ratio > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("final_amplitude_ratio must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Frequency decrement generator
pub struct FrequencyDecrementGenerator;

#[derive(Debug, Clone)]
pub struct FrequencyDecrementParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub initial_frequency: f64,
    pub final_frequency_ratio: f64, // 0-1
    pub amplitude: f64,
}

impl SyntheticGenerator for FrequencyDecrementGenerator {
    type Output = Vec<f64>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = FrequencyDecrementParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;

        let mut separation = Vec::with_capacity(n_frames);
        let mut phase = 0.0;

        for i in 0..n_frames {
            let t = i as f64 * dt;
            let progress = t / params.duration;

            // Linearly decreasing frequency
            let freq = params.initial_frequency *
                (1.0 - progress * (1.0 - params.final_frequency_ratio));

            phase += 2.0 * PI * freq * dt;

            let distance = if phase.rem_euclid(2.0 * PI) < PI {
                params.amplitude * (phase.rem_euclid(PI) / PI)
            } else {
                params.amplitude * (1.0 - (phase.rem_euclid(2.0 * PI) - PI) / PI)
            };

            separation.push(distance);
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("finger_flexion".to_string(), separation.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(separation, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        FrequencyDecrementParams {
            duration: 10.0,
            frame_rate: 60.0,
            initial_frequency: 4.0,
            final_frequency_ratio: 0.4,
            amplitude: 5.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.final_frequency_ratio < 0.0 || params.final_frequency_ratio > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("final_frequency_ratio must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Hesitation/arrest generator (freezing episodes)
pub struct HesitationArrestGenerator;

#[derive(Debug, Clone)]
pub struct HesitationArrestParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub frequency: f64,
    pub amplitude: f64,
    pub arrest_probability: f64,  // Probability per tap of arrest
    pub arrest_duration_mean: f64, // Mean duration of arrest (seconds)
    pub arrest_duration_std: f64,  // Std deviation of arrest duration
}

impl SyntheticGenerator for HesitationArrestGenerator {
    type Output = Vec<f64>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HesitationArrestParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut separation = Vec::with_capacity(n_frames);
        let mut phase = 0.0;
        let mut arrest_remaining = 0.0;
        let mut events = Vec::new();

        for i in 0..n_frames {
            let t = i as f64 * dt;

            // Check if in arrest state
            if arrest_remaining > 0.0 {
                // During arrest, movement is frozen
                if let Some(&last_val) = separation.last() {
                    separation.push(last_val);
                } else {
                    separation.push(0.0);
                }
                arrest_remaining -= dt;
            } else {
                // Normal tapping
                phase += 2.0 * PI * params.frequency * dt;

                let distance = if phase.rem_euclid(2.0 * PI) < PI {
                    params.amplitude * (phase.rem_euclid(PI) / PI)
                } else {
                    params.amplitude * (1.0 - (phase.rem_euclid(2.0 * PI) - PI) / PI)
                };

                separation.push(distance);

                // Check for arrest at tap completion
                if distance < 0.1 && i > 0 && separation[i - 1] >= 0.1 {
                    if rng.r#gen::<f64>() < params.arrest_probability {
                        let duration_dist = Normal::new(params.arrest_duration_mean, params.arrest_duration_std).unwrap();
                        arrest_remaining = duration_dist.sample(&mut rng).max(0.1);

                        events.push(Event {
                            time: t,
                            event_type: "arrest".to_string(),
                            amplitude: Some(arrest_remaining),
                            attributes: HashMap::new(),
                        });
                    }
                }
            }
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("finger_flexion".to_string(), separation.clone());

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(separation, ground_truth, params.frame_rate)
            .with_metadata("arrest_count".to_string(), events.len().to_string()))
    }

    fn default_params() -> Self::Parameters {
        HesitationArrestParams {
            duration: 10.0,
            frame_rate: 60.0,
            frequency: 4.0,
            amplitude: 5.0,
            arrest_probability: 0.15,
            arrest_duration_mean: 1.0,
            arrest_duration_std: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.arrest_probability < 0.0 || params.arrest_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("arrest_probability must be 0-1".to_string()));
        }
        if params.arrest_duration_std < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("arrest_duration_std must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Tapping fatigue generator (progressive exhaustion)
pub struct TappingFatigueGenerator;

#[derive(Debug, Clone)]
pub struct TappingFatigueParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub initial_frequency: f64,
    pub initial_amplitude: f64,
    pub fatigue_time_constant: f64,  // Time constant for exponential fatigue
    pub rest_periods: Vec<(f64, f64)>, // Optional rest periods (start, duration)
    pub recovery_rate: f64,            // Recovery during rest (0-1)
}

impl SyntheticGenerator for TappingFatigueGenerator {
    type Output = Vec<f64>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = TappingFatigueParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut separation = Vec::with_capacity(n_frames);
        let mut phase = 0.0;
        let mut fatigue_level = 0.0; // 0 = no fatigue, 1 = complete fatigue

        for i in 0..n_frames {
            let t = i as f64 * dt;

            // Check if in rest period
            let in_rest = params.rest_periods.iter().any(|(start, duration)| {
                t >= *start && t < *start + *duration
            });

            if in_rest {
                // Recovery during rest
                fatigue_level = (fatigue_level - params.recovery_rate * dt).max(0.0);
                // Maintain current position during rest
                if let Some(&last_val) = separation.last() {
                    separation.push(last_val);
                } else {
                    separation.push(0.0);
                }
            } else {
                // Accumulate fatigue
                fatigue_level = (fatigue_level + dt / params.fatigue_time_constant).min(1.0);

                // Fatigue reduces both frequency and amplitude
                let current_frequency = params.initial_frequency * (1.0 - 0.5 * fatigue_level);
                let current_amplitude = params.initial_amplitude * (1.0 - 0.7 * fatigue_level);

                phase += 2.0 * PI * current_frequency * dt;

                let distance = if phase.rem_euclid(2.0 * PI) < PI {
                    current_amplitude * (phase.rem_euclid(PI) / PI)
                } else {
                    current_amplitude * (1.0 - (phase.rem_euclid(2.0 * PI) - PI) / PI)
                };

                separation.push(distance);
            }
        }

        let mut joint_angles = HashMap::new();
        joint_angles.insert("finger_flexion".to_string(), separation.clone());
        joint_angles.insert("fatigue_level".to_string(), vec![fatigue_level]);

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles,
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(separation, ground_truth, params.frame_rate)
            .with_metadata("final_fatigue".to_string(), fatigue_level.to_string()))
    }

    fn default_params() -> Self::Parameters {
        TappingFatigueParams {
            duration: 30.0,
            frame_rate: 60.0,
            initial_frequency: 4.0,
            initial_amplitude: 5.0,
            fatigue_time_constant: 15.0,
            rest_periods: vec![],
            recovery_rate: 0.1,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.fatigue_time_constant <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("fatigue_time_constant must be positive".to_string()));
        }
        if params.recovery_rate < 0.0 || params.recovery_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("recovery_rate must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_tapping() {
        let generator = NormalTappingGenerator;
        let params = NormalTappingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_bradykinetic_tapping() {
        let generator = BradykineticTappingGenerator;
        let params = BradykineticTappingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_amplitude_decrement() {
        let generator = AmplitudeDecrementGenerator;
        let params = AmplitudeDecrementGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        // Check amplitude decreases
        let first_quarter = &result.signal[0..result.signal.len() / 4];
        let last_quarter = &result.signal[3 * result.signal.len() / 4..];

        let first_max = first_quarter.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let last_max = last_quarter.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        assert!(first_max > last_max, "Amplitude should decrease over time");
    }

    #[test]
    fn test_hesitation_arrest() {
        let generator = HesitationArrestGenerator;
        let params = HesitationArrestGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_tapping_fatigue() {
        let generator = TappingFatigueGenerator;
        let params = TappingFatigueGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);

        // Check that fatigue increases over time
        let first_half_avg: f64 = result.signal[0..result.signal.len()/2].iter().sum::<f64>() / (result.signal.len()/2) as f64;
        let second_half_avg: f64 = result.signal[result.signal.len()/2..].iter().sum::<f64>() / (result.signal.len()/2) as f64;
        assert!(first_half_avg > second_half_avg, "Tapping should show fatigue over time");
    }
}
