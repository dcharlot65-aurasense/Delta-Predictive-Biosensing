//! Gait variability generators
//! Temporal and spatial variability patterns for gait analysis

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth, GaitPhase};
use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Stride time variability generator
pub struct StrideTimeVariabilityGenerator;

#[derive(Debug, Clone)]
pub struct StrideTimeVariabilityParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub cv_target: f64,             // Coefficient of Variation (CV) target (0.02-0.10)
    pub fractal_index: f64,         // DFA alpha (0.5-1.5, healthy ~0.8-1.0)
    pub mean_cadence: f64,          // steps per minute
    pub height: f64,
}

impl SyntheticGenerator for StrideTimeVariabilityGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = StrideTimeVariabilityParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let num_keypoints = 33;
        let mut keypoints = Vec::with_capacity(n_frames);
        let mut gait_phases = Vec::new();
        let mut joint_angles = HashMap::new();

        joint_angles.insert("stride_time".to_string(), Vec::new());
        joint_angles.insert("instantaneous_cadence".to_string(), Vec::new());

        // Generate stride times with target CV and fractal structure
        let mean_stride_time = 60.0 / params.mean_cadence;
        let stride_time_std = mean_stride_time * params.cv_target;

        let noise_dist = Normal::new(0.0, stride_time_std).unwrap();

        let mut cumulative_time = 0.0;
        let mut stride_count = 0;
        let mut prev_noise = 0.0;

        for frame in 0..n_frames {
            let t = frame as f64 * dt;

            // Fractal noise (simple first-order autoregressive approximation)
            let alpha = params.fractal_index.clamp(0.5, 1.5);
            let correlation = (alpha - 0.5) / 1.0; // maps 0.5-1.5 to 0-1
            let noise = correlation * prev_noise + (1.0 - correlation) * noise_dist.sample(&mut rng);
            prev_noise = noise;

            let current_stride_time = (mean_stride_time + noise).max(0.5);
            let gait_phase = (t - cumulative_time) / current_stride_time;

            if gait_phase >= 1.0 {
                cumulative_time = t;
                stride_count += 1;
                joint_angles.get_mut("stride_time").unwrap().push(current_stride_time);
            }

            let phase = gait_phase % 1.0;
            let instantaneous_cadence = 60.0 / current_stride_time;
            joint_angles.get_mut("instantaneous_cadence").unwrap().push(instantaneous_cadence);

            // Generate keypoints
            let mut frame_keypoints = vec![[0.0; 3]; num_keypoints];
            let pelvis_y = params.height * 0.55;
            let stride_length = 1.4;
            let pelvis_z = (cumulative_time + phase * current_stride_time) * stride_length / mean_stride_time;

            frame_keypoints[0] = [0.0, pelvis_y, pelvis_z];

            gait_phases.push(GaitPhase {
                frame,
                phase,
                phase_name: if phase < 0.6 { "stance" } else { "swing" }.to_string(),
            });

            keypoints.push(frame_keypoints);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: keypoints.clone(),
            joint_angles,
            gait_phases,
        };

        let mut result = GeneratedData::new(keypoints, ground_truth, params.frame_rate);
        result.metadata.insert("cv_target".to_string(), params.cv_target.to_string());
        result.metadata.insert("fractal_index".to_string(), params.fractal_index.to_string());
        result.metadata.insert("stride_count".to_string(), stride_count.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        StrideTimeVariabilityParams {
            duration: 60.0,  // need longer for variability analysis
            frame_rate: 30.0,
            cv_target: 0.03,        // 3% CV (healthy)
            fractal_index: 0.9,     // DFA alpha (healthy)
            mean_cadence: 110.0,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.cv_target < 0.0 || params.cv_target > 0.2 {
            return Err(crate::GeneratorError::InvalidParameter("cv_target must be 0-0.2".to_string()));
        }
        if params.fractal_index < 0.5 || params.fractal_index > 1.5 {
            return Err(crate::GeneratorError::InvalidParameter("fractal_index must be 0.5-1.5".to_string()));
        }
        Ok(())
    }
}

/// Stride length variability generator
pub struct StrideLengthVariabilityGenerator;

#[derive(Debug, Clone)]
pub struct StrideLengthVariabilityParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub cv_target: f64,             // CV target (0.02-0.10)
    pub mean_stride_length: f64,    // meters
    pub height: f64,
}

impl SyntheticGenerator for StrideLengthVariabilityGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = StrideLengthVariabilityParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let num_keypoints = 33;
        let mut keypoints = Vec::with_capacity(n_frames);
        let mut gait_phases = Vec::new();
        let mut joint_angles = HashMap::new();

        joint_angles.insert("stride_length".to_string(), Vec::new());

        let stride_length_std = params.mean_stride_length * params.cv_target;
        let noise_dist = Normal::new(0.0, stride_length_std).unwrap();

        let cadence = 110.0;
        let cycle_duration = 60.0 / cadence;
        let mut cumulative_distance = 0.0;

        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            let gait_phase = (t % cycle_duration) / cycle_duration;

            // Vary stride length each cycle
            let current_stride_length = if gait_phase < 0.1 {
                // New stride, sample new length
                let sl = (params.mean_stride_length + noise_dist.sample(&mut rng)).max(0.5);
                joint_angles.get_mut("stride_length").unwrap().push(sl);
                sl
            } else {
                // Use last sampled stride length
                *joint_angles.get("stride_length").unwrap().last().unwrap_or(&params.mean_stride_length)
            };

            // Update position based on variable stride length
            cumulative_distance += (current_stride_length / cycle_duration) * dt;

            let mut frame_keypoints = vec![[0.0; 3]; num_keypoints];
            let pelvis_y = params.height * 0.55;
            frame_keypoints[0] = [0.0, pelvis_y, cumulative_distance];

            gait_phases.push(GaitPhase {
                frame,
                phase: gait_phase,
                phase_name: if gait_phase < 0.6 { "stance" } else { "swing" }.to_string(),
            });

            keypoints.push(frame_keypoints);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: keypoints.clone(),
            joint_angles,
            gait_phases,
        };

        let mut result = GeneratedData::new(keypoints, ground_truth, params.frame_rate);
        result.metadata.insert("cv_target".to_string(), params.cv_target.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        StrideLengthVariabilityParams {
            duration: 60.0,
            frame_rate: 30.0,
            cv_target: 0.03,
            mean_stride_length: 1.4,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.cv_target < 0.0 || params.cv_target > 0.2 {
            return Err(crate::GeneratorError::InvalidParameter("cv_target must be 0-0.2".to_string()));
        }
        Ok(())
    }
}

/// Dual-task variability generator (cognitive load effect)
pub struct DualTaskVariabilityGenerator;

#[derive(Debug, Clone)]
pub struct DualTaskVariabilityParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub cognitive_load: f64,        // 0-1 (0 = single task, 1 = high dual-task)
    pub effect_on_speed: f64,       // speed reduction ratio
    pub effect_on_variability: f64, // CV increase multiplier
    pub height: f64,
}

impl SyntheticGenerator for DualTaskVariabilityGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = DualTaskVariabilityParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        // Dual-task effects:
        // - Reduced walking speed
        // - Increased stride time variability
        // - Potential stride length reduction

        let base_cadence = 110.0;
        let adjusted_cadence = base_cadence * (1.0 - params.effect_on_speed * params.cognitive_load);

        let base_cv = 0.02; // healthy baseline
        let adjusted_cv = base_cv * (1.0 + params.effect_on_variability * params.cognitive_load);

        let variability_params = StrideTimeVariabilityParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cv_target: adjusted_cv,
            fractal_index: 0.9 - params.cognitive_load * 0.2, // less fractal with cognitive load
            mean_cadence: adjusted_cadence,
            height: params.height,
        };

        let variability_gen = StrideTimeVariabilityGenerator;
        let mut result = variability_gen.generate(&variability_params, seed)?;

        result.metadata.insert("task_type".to_string(), "dual_task".to_string());
        result.metadata.insert("cognitive_load".to_string(), params.cognitive_load.to_string());
        result.metadata.insert("adjusted_cadence".to_string(), adjusted_cadence.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        DualTaskVariabilityParams {
            duration: 60.0,
            frame_rate: 30.0,
            cognitive_load: 0.5,        // moderate dual-task
            effect_on_speed: 0.15,      // 15% speed reduction
            effect_on_variability: 2.0, // 2x variability increase
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.cognitive_load < 0.0 || params.cognitive_load > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("cognitive_load must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Fatigue progression generator
pub struct FatigueProgressionGenerator;

#[derive(Debug, Clone)]
pub struct FatigueProgressionParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub fatigue_rate: f64,          // rate of fatigue development (0-1)
    pub fatigue_pattern: String,    // "linear", "exponential", "step"
    pub height: f64,
}

impl SyntheticGenerator for FatigueProgressionGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = FatigueProgressionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        let num_keypoints = 33;
        let mut keypoints = Vec::with_capacity(n_frames);
        let mut gait_phases = Vec::new();
        let mut joint_angles = HashMap::new();

        joint_angles.insert("fatigue_level".to_string(), Vec::new());
        joint_angles.insert("cadence".to_string(), Vec::new());
        joint_angles.insert("stride_length".to_string(), Vec::new());

        let base_cadence = 110.0;
        let base_stride = 1.4;
        let mut cumulative_distance = 0.0;

        for frame in 0..n_frames {
            let t = frame as f64 * dt;
            let progress = t / params.duration;

            // Calculate fatigue level based on pattern
            let fatigue_level = match params.fatigue_pattern.as_str() {
                "linear" => progress * params.fatigue_rate,
                "exponential" => (1.0 - (-3.0 * progress * params.fatigue_rate).exp()) * params.fatigue_rate,
                "step" => {
                    if progress < 0.33 { 0.0 }
                    else if progress < 0.67 { 0.5 * params.fatigue_rate }
                    else { params.fatigue_rate }
                },
                _ => progress * params.fatigue_rate,
            };

            joint_angles.get_mut("fatigue_level").unwrap().push(fatigue_level);

            // Fatigue effects:
            // - Decreased cadence
            // - Decreased stride length
            // - Increased variability (not shown in this simple version)
            let current_cadence = base_cadence * (1.0 - fatigue_level * 0.3);
            let current_stride = base_stride * (1.0 - fatigue_level * 0.2);

            joint_angles.get_mut("cadence").unwrap().push(current_cadence);
            joint_angles.get_mut("stride_length").unwrap().push(current_stride);

            let cycle_duration = 60.0 / current_cadence;
            let gait_phase = (t % cycle_duration) / cycle_duration;

            cumulative_distance += (current_stride / cycle_duration) * dt;

            let mut frame_keypoints = vec![[0.0; 3]; num_keypoints];
            let pelvis_y = params.height * 0.55 * (1.0 - fatigue_level * 0.05); // slight height loss
            frame_keypoints[0] = [0.0, pelvis_y, cumulative_distance];

            gait_phases.push(GaitPhase {
                frame,
                phase: gait_phase,
                phase_name: if gait_phase < 0.6 { "stance" } else { "swing" }.to_string(),
            });

            keypoints.push(frame_keypoints);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: keypoints.clone(),
            joint_angles,
            gait_phases,
        };

        let mut result = GeneratedData::new(keypoints, ground_truth, params.frame_rate);
        result.metadata.insert("fatigue_pattern".to_string(), params.fatigue_pattern.clone());
        result.metadata.insert("fatigue_rate".to_string(), params.fatigue_rate.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        FatigueProgressionParams {
            duration: 300.0,  // 5 minutes
            frame_rate: 30.0,
            fatigue_rate: 0.5,
            fatigue_pattern: "exponential".to_string(),
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.fatigue_rate < 0.0 || params.fatigue_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("fatigue_rate must be 0-1".to_string()));
        }
        if params.fatigue_pattern != "linear" &&
           params.fatigue_pattern != "exponential" &&
           params.fatigue_pattern != "step" {
            return Err(crate::GeneratorError::InvalidParameter(
                "fatigue_pattern must be 'linear', 'exponential', or 'step'".to_string()
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stride_time_variability() {
        let generator = StrideTimeVariabilityGenerator;
        let params = StrideTimeVariabilityGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
        assert!(!result.ground_truth.joint_angles.get("stride_time").unwrap().is_empty());
    }

    #[test]
    fn test_stride_length_variability() {
        let generator = StrideLengthVariabilityGenerator;
        let params = StrideLengthVariabilityGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_dual_task_variability() {
        let generator = DualTaskVariabilityGenerator;
        let params = DualTaskVariabilityGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_fatigue_progression() {
        let generator = FatigueProgressionGenerator;
        let params = FatigueProgressionParams {
            duration: 30.0,  // shorter for test
            frame_rate: 30.0,
            fatigue_rate: 0.5,
            fatigue_pattern: "linear".to_string(),
            height: 1.75,
        };
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);

        // Check fatigue progresses
        let fatigue_levels = result.ground_truth.joint_angles.get("fatigue_level").unwrap();
        assert!(fatigue_levels.last().unwrap() > fatigue_levels.first().unwrap());
    }
}
