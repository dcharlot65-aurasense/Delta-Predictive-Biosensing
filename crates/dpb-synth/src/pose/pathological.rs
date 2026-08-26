//! Pathological gait pattern generators
//! Specific clinical gait abnormalities with biomechanical accuracy

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth, GaitPhase};
use crate::pose::gait::{GaitCycleGenerator, GaitCycleParams};
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Shuffling gait generator (Parkinsonian)
pub struct ShufflingGaitGenerator;

#[derive(Debug, Clone)]
pub struct ShufflingGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub step_height_reduction: f64,  // 0-1 (0 = normal, 1 = no lift)
    pub stride_shortening: f64,      // 0-1 (0 = normal, 1 = minimal stride)
    pub height: f64,                 // meters
}

impl SyntheticGenerator for ShufflingGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = ShufflingGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        // Start with normal gait
        let base_stride = 1.4 * (1.0 - params.stride_shortening * 0.7);
        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0 * (1.0 + params.stride_shortening * 0.2), // faster, shorter steps
            stride_length: base_stride,
            step_width: 0.15,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Reduce vertical displacement (step height)
        for frame_kps in &mut result.signal {
            for kp in frame_kps.iter_mut() {
                // Reduce vertical movement
                let baseline_height = params.height * 0.55;
                kp[1] = baseline_height + (kp[1] - baseline_height) * (1.0 - params.step_height_reduction);
            }
        }

        result.metadata.insert("pathology".to_string(), "shuffling".to_string());
        result.metadata.insert("step_height_reduction".to_string(), params.step_height_reduction.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        ShufflingGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            step_height_reduction: 0.7,
            stride_shortening: 0.5,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.step_height_reduction < 0.0 || params.step_height_reduction > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("step_height_reduction must be 0-1".to_string()));
        }
        if params.stride_shortening < 0.0 || params.stride_shortening > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("stride_shortening must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Festination generator (progressive acceleration)
pub struct FestinationGenerator;

#[derive(Debug, Clone)]
pub struct FestinationParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub acceleration_rate: f64,     // cadence increase per second
    pub trigger_time: f64,          // when festination starts (seconds)
    pub height: f64,
}

impl SyntheticGenerator for FestinationGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = FestinationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let num_keypoints = 33;

        let mut keypoints = Vec::with_capacity(n_frames);
        let mut gait_phases = Vec::new();
        let mut joint_angles = HashMap::new();
        joint_angles.insert("cadence".to_string(), Vec::new());

        let base_cadence = 110.0;
        let mut cumulative_phase = 0.0;

        for frame in 0..n_frames {
            let t = frame as f64 * dt;

            // Progressive acceleration after trigger
            let current_cadence = if t < params.trigger_time {
                base_cadence
            } else {
                base_cadence + params.acceleration_rate * (t - params.trigger_time)
            };

            let cycle_duration = 60.0 / current_cadence;
            cumulative_phase += dt / cycle_duration;
            let gait_phase = cumulative_phase % 1.0;

            joint_angles.get_mut("cadence").unwrap().push(current_cadence);

            // Generate simplified keypoints
            let mut frame_keypoints = vec![[0.0; 3]; num_keypoints];

            // Pelvis
            let pelvis_y = params.height * 0.55;
            let stride_length = 1.4 * (1.0 - (current_cadence - base_cadence) / 100.0).max(0.5);
            let pelvis_z = cumulative_phase * stride_length;
            frame_keypoints[0] = [0.0, pelvis_y, pelvis_z];

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
        result.metadata.insert("pathology".to_string(), "festination".to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        FestinationParams {
            duration: 10.0,
            frame_rate: 30.0,
            acceleration_rate: 10.0,  // 10 steps/min increase per second
            trigger_time: 2.0,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

/// Freezing of Gait (FOG) generator
pub struct FreezingOfGaitGenerator;

#[derive(Debug, Clone)]
pub struct FreezingOfGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub fog_probability: f64,       // probability per second
    pub fog_duration_mean: f64,     // mean duration (seconds)
    pub fog_duration_std: f64,      // std deviation
    pub trigger_situations: Vec<String>, // "turning", "doorway", "initiation"
    pub height: f64,
}

impl SyntheticGenerator for FreezingOfGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = FreezingOfGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0,
            stride_length: 1.4,
            step_width: 0.15,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Inject FOG episodes
        let fog_duration_dist = Normal::new(params.fog_duration_mean, params.fog_duration_std).unwrap();
        let mut in_fog = false;
        let mut fog_end_frame = 0;
        let mut fog_episodes = Vec::new();

        for frame in 0..n_frames {
            if !in_fog && rng.random_range(0.0..1.0) < params.fog_probability * dt {
                // Start FOG episode
                in_fog = true;
                let duration = fog_duration_dist.sample(&mut rng).max(0.5);
                fog_end_frame = frame + (duration * params.frame_rate) as usize;
                fog_episodes.push((frame, fog_end_frame));
            }

            if in_fog {
                if frame < fog_end_frame {
                    // During FOG: freeze keypoints, add trembling
                    let prev_kps = if frame > 0 {
                        result.signal[frame - 1].clone()
                    } else {
                        result.signal[frame].clone()
                    };

                    let tremor = Normal::new(0.0, 0.002).unwrap();
                    for (kp, prev_kp) in result.signal[frame].iter_mut().zip(prev_kps.iter()) {
                        kp[0] = prev_kp[0] + tremor.sample(&mut rng);
                        kp[1] = prev_kp[1] + tremor.sample(&mut rng);
                        kp[2] = prev_kp[2]; // no forward progression
                    }
                } else {
                    in_fog = false;
                }
            }
        }

        result.metadata.insert("pathology".to_string(), "freezing_of_gait".to_string());
        result.metadata.insert("fog_episodes".to_string(), fog_episodes.len().to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        FreezingOfGaitParams {
            duration: 20.0,
            frame_rate: 30.0,
            fog_probability: 0.05,      // 5% per second
            fog_duration_mean: 2.0,     // 2 seconds mean
            fog_duration_std: 0.5,
            trigger_situations: vec!["turning".to_string()],
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.fog_probability < 0.0 || params.fog_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("fog_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Asymmetric gait generator
pub struct AsymmetricGaitGenerator;

#[derive(Debug, Clone)]
pub struct AsymmetricGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub asymmetry_ratio: f64,       // 0-1 (ratio of affected to unaffected)
    pub affected_side: String,      // "left" or "right"
    pub height: f64,
}

impl SyntheticGenerator for AsymmetricGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = AsymmetricGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0,
            stride_length: 1.4,
            step_width: 0.15,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Apply asymmetry
        let affected_indices = if params.affected_side == "right" {
            vec![24, 26, 28] // right hip, knee, ankle
        } else {
            vec![23, 25, 27] // left hip, knee, ankle
        };

        for frame_kps in &mut result.signal {
            for &idx in &affected_indices {
                // Reduce step length and height on affected side
                frame_kps[idx][1] *= 1.0 - (1.0 - params.asymmetry_ratio) * 0.3; // less vertical lift
                frame_kps[idx][2] *= params.asymmetry_ratio; // shorter stride
            }
        }

        result.metadata.insert("pathology".to_string(), "asymmetric".to_string());
        result.metadata.insert("affected_side".to_string(), params.affected_side.clone());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        AsymmetricGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            asymmetry_ratio: 0.7,       // affected side 70% of normal
            affected_side: "right".to_string(),
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.asymmetry_ratio < 0.0 || params.asymmetry_ratio > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("asymmetry_ratio must be 0-1".to_string()));
        }
        if params.affected_side != "left" && params.affected_side != "right" {
            return Err(crate::GeneratorError::InvalidParameter("affected_side must be 'left' or 'right'".to_string()));
        }
        Ok(())
    }
}

/// Ataxic gait generator (cerebellar dysfunction)
pub struct AtaxicGaitGenerator;

#[derive(Debug, Clone)]
pub struct AtaxicGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub variability_increase: f64,  // 0-1 (0 = normal, 1 = severe ataxia)
    pub base_widening: f64,         // meters (additional step width)
    pub height: f64,
}

impl SyntheticGenerator for AtaxicGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = AtaxicGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0,
            stride_length: 1.4,
            step_width: 0.15 + params.base_widening,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Add irregular, wide-based gait pattern
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, 0.05 * params.variability_increase).unwrap();

        for frame_kps in &mut result.signal {
            for kp in frame_kps.iter_mut() {
                // High spatial variability
                kp[0] += noise.sample(&mut rng);
                kp[1] += noise.sample(&mut rng);
                kp[2] += noise.sample(&mut rng);
            }
        }

        result.metadata.insert("pathology".to_string(), "ataxic".to_string());
        result.metadata.insert("base_widening".to_string(), params.base_widening.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        AtaxicGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            variability_increase: 0.7,
            base_widening: 0.10,  // 10 cm wider base
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.variability_increase < 0.0 || params.variability_increase > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("variability_increase must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Spastic gait generator (stiff, circumduction)
pub struct SpasticGaitGenerator;

#[derive(Debug, Clone)]
pub struct SpasticGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub stiffness: f64,             // 0-1 (reduced joint ROM)
    pub circumduction: f64,         // meters (lateral swing)
    pub height: f64,
}

impl SyntheticGenerator for SpasticGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = SpasticGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0 * (1.0 - params.stiffness * 0.3), // slower with stiffness
            stride_length: 1.4 * (1.0 - params.stiffness * 0.4),
            step_width: 0.15,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Apply stiffness and circumduction
        for (frame_idx, frame_kps) in result.signal.iter_mut().enumerate() {
            let t = frame_idx as f64 / params.frame_rate;
            let gait_phase = (t * 110.0 / 60.0) % 1.0;

            // Right leg circumduction during swing
            if gait_phase > 0.6 {
                let swing_progress = (gait_phase - 0.6) / 0.4;
                let circumduction_amount = params.circumduction * (PI * swing_progress).sin();
                frame_kps[24][0] += circumduction_amount; // right hip
                frame_kps[26][0] += circumduction_amount; // right knee
                frame_kps[28][0] += circumduction_amount; // right ankle
            }

            // Reduce vertical displacement (stiff knee)
            for kp in frame_kps.iter_mut() {
                let baseline = params.height * 0.55;
                kp[1] = baseline + (kp[1] - baseline) * (1.0 - params.stiffness * 0.6);
            }
        }

        result.metadata.insert("pathology".to_string(), "spastic".to_string());
        result.metadata.insert("stiffness".to_string(), params.stiffness.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        SpasticGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            stiffness: 0.6,
            circumduction: 0.08,  // 8 cm lateral swing
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.stiffness < 0.0 || params.stiffness > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("stiffness must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Antalgic gait generator (pain-avoiding)
pub struct AntalglicGaitGenerator;

#[derive(Debug, Clone)]
pub struct AntalglicGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub pain_side: String,          // "left" or "right"
    pub severity: f64,              // 0-1 (stance time reduction)
    pub height: f64,
}

impl SyntheticGenerator for AntalglicGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = AntalglicGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0,
            stride_length: 1.4,
            step_width: 0.15,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Reduce stance phase on painful side (quick weight transfer)
        let affected_indices = if params.pain_side == "right" {
            vec![24, 26, 28]
        } else {
            vec![23, 25, 27]
        };

        for frame_kps in &mut result.signal {
            for &idx in &affected_indices {
                // Reduce weight-bearing (less vertical displacement)
                frame_kps[idx][1] *= 1.0 + params.severity * 0.05;
            }
        }

        result.metadata.insert("pathology".to_string(), "antalgic".to_string());
        result.metadata.insert("pain_side".to_string(), params.pain_side.clone());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        AntalglicGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            pain_side: "right".to_string(),
            severity: 0.5,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        if params.pain_side != "left" && params.pain_side != "right" {
            return Err(crate::GeneratorError::InvalidParameter("pain_side must be 'left' or 'right'".to_string()));
        }
        Ok(())
    }
}

/// Hemiplegic gait generator (stroke-related)
pub struct HemiplegicGaitGenerator;

#[derive(Debug, Clone)]
pub struct HemiplegicGaitParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub affected_side: String,      // "left" or "right"
    pub severity: f64,              // 0-1
    pub height: f64,
}

impl SyntheticGenerator for HemiplegicGaitGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HemiplegicGaitParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let gait_params = GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: 110.0 * (1.0 - params.severity * 0.3),
            stride_length: 1.4 * (1.0 - params.severity * 0.4),
            step_width: 0.15,
            height: params.height,
        };

        let gait_gen = GaitCycleGenerator;
        let mut result = gait_gen.generate(&gait_params, seed)?;

        // Hemiplegic pattern: circumduction, foot drop, reduced arm swing
        let leg_indices = if params.affected_side == "right" {
            vec![24, 26, 28]
        } else {
            vec![23, 25, 27]
        };

        for (frame_idx, frame_kps) in result.signal.iter_mut().enumerate() {
            let t = frame_idx as f64 / params.frame_rate;
            let gait_phase = (t * 110.0 / 60.0) % 1.0;

            // Circumduction during swing phase
            if gait_phase > 0.6 {
                let swing_progress = (gait_phase - 0.6) / 0.4;
                let circumduction = 0.12 * params.severity * (PI * swing_progress).sin();

                for &idx in &leg_indices {
                    frame_kps[idx][0] += circumduction;
                }
            }

            // Foot drop (ankle plantarflexion)
            let ankle_idx = leg_indices[2];
            frame_kps[ankle_idx][1] *= 1.0 - params.severity * 0.1;
        }

        result.metadata.insert("pathology".to_string(), "hemiplegic".to_string());
        result.metadata.insert("affected_side".to_string(), params.affected_side.clone());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        HemiplegicGaitParams {
            duration: 10.0,
            frame_rate: 30.0,
            affected_side: "right".to_string(),
            severity: 0.6,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        if params.affected_side != "left" && params.affected_side != "right" {
            return Err(crate::GeneratorError::InvalidParameter("affected_side must be 'left' or 'right'".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shuffling_gait() {
        let generator = ShufflingGaitGenerator;
        let params = ShufflingGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_festination() {
        let generator = FestinationGenerator;
        let params = FestinationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_freezing_of_gait() {
        let generator = FreezingOfGaitGenerator;
        let params = FreezingOfGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_asymmetric_gait() {
        let generator = AsymmetricGaitGenerator;
        let params = AsymmetricGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_ataxic_gait() {
        let generator = AtaxicGaitGenerator;
        let params = AtaxicGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_spastic_gait() {
        let generator = SpasticGaitGenerator;
        let params = SpasticGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_antalgic_gait() {
        let generator = AntalglicGaitGenerator;
        let params = AntalglicGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_hemiplegic_gait() {
        let generator = HemiplegicGaitGenerator;
        let params = HemiplegicGaitGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }
}
