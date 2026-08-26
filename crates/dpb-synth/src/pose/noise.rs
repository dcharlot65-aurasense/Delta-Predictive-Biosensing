//! Pose noise and artifact generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;

/// Keypoint jitter generator
pub struct KeypointJitterGenerator;

#[derive(Debug, Clone)]
pub struct KeypointJitterParams {
    pub keypoints: Vec<Vec<[f64; 3]>>, // input keypoints
    pub jitter_std: f64,                // standard deviation of jitter
    pub frame_rate: f64,
}

impl SyntheticGenerator for KeypointJitterGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = KeypointJitterParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, params.jitter_std).unwrap();

        let noisy_keypoints: Vec<Vec<[f64; 3]>> = params.keypoints
            .iter()
            .map(|frame| {
                frame.iter()
                    .map(|kp| {
                        [
                            kp[0] + noise.sample(&mut rng),
                            kp[1] + noise.sample(&mut rng),
                            kp[2] + noise.sample(&mut rng),
                        ]
                    })
                    .collect()
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(noisy_keypoints, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        KeypointJitterParams {
            keypoints: vec![vec![[0.0; 3]; 33]; 10],
            jitter_std: 0.01,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.keypoints.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("keypoints cannot be empty".to_string()));
        }
        if params.jitter_std < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("jitter_std must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Occlusion generator
pub struct OcclusionGenerator;

#[derive(Debug, Clone)]
pub struct OcclusionParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub occlusion_probability: f64, // per frame per keypoint
    pub occlusion_value: f64,       // value to set (e.g., 0.0 or NaN)
    pub frame_rate: f64,
}

impl SyntheticGenerator for OcclusionGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = OcclusionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let occluded_keypoints: Vec<Vec<[f64; 3]>> = params.keypoints
            .iter()
            .map(|frame| {
                frame.iter()
                    .map(|kp| {
                        if rng.random::<f64>() < params.occlusion_probability {
                            [params.occlusion_value; 3]
                        } else {
                            *kp
                        }
                    })
                    .collect()
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(occluded_keypoints, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        OcclusionParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 10],
            occlusion_probability: 0.1,
            occlusion_value: 0.0,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.keypoints.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("keypoints cannot be empty".to_string()));
        }
        if params.occlusion_probability < 0.0 || params.occlusion_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("occlusion_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Tracking dropout generator (entire frames lost)
pub struct TrackingDropoutGenerator;

#[derive(Debug, Clone)]
pub struct TrackingDropoutParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub dropout_probability: f64,   // per frame
    pub frame_rate: f64,
}

impl SyntheticGenerator for TrackingDropoutGenerator {
    type Output = Vec<Option<Vec<[f64; 3]>>>; // None = dropped frame
    type GroundTruth = SpatialGroundTruth;
    type Parameters = TrackingDropoutParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let frames_with_dropout: Vec<Option<Vec<[f64; 3]>>> = params.keypoints
            .iter()
            .map(|frame| {
                if rng.random::<f64>() < params.dropout_probability {
                    None
                } else {
                    Some(frame.clone())
                }
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(frames_with_dropout, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        TrackingDropoutParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 100],
            dropout_probability: 0.05,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.keypoints.is_empty() {
            return Err(crate::GeneratorError::InvalidParameter("keypoints cannot be empty".to_string()));
        }
        if params.dropout_probability < 0.0 || params.dropout_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("dropout_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// ID switch generator (tracking errors)
pub struct IdSwitchGenerator;

#[derive(Debug, Clone)]
pub struct IdSwitchParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub switch_probability: f64,    // per frame
    pub num_persons: usize,          // number of tracked persons
    pub frame_rate: f64,
}

impl SyntheticGenerator for IdSwitchGenerator {
    type Output = Vec<Vec<(usize, Vec<[f64; 3]>)>>; // (person_id, keypoints)
    type GroundTruth = SpatialGroundTruth;
    type Parameters = IdSwitchParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut current_ids: Vec<usize> = (0..params.num_persons).collect();

        let frames_with_ids: Vec<Vec<(usize, Vec<[f64; 3]>)>> = params.keypoints
            .iter()
            .map(|frame| {
                // Randomly swap IDs
                if rng.random::<f64>() < params.switch_probability
                    && params.num_persons >= 2 {
                        let idx1 = rng.random_range(0..params.num_persons);
                        let idx2 = rng.random_range(0..params.num_persons);
                        current_ids.swap(idx1, idx2);
                    }

                // Assign keypoints to IDs (simplified - same keypoints with different IDs)
                current_ids.iter()
                    .map(|&id| (id, frame.clone()))
                    .collect()
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(frames_with_ids, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        IdSwitchParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 100],
            switch_probability: 0.01,
            num_persons: 2,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.switch_probability < 0.0 || params.switch_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("switch_probability must be 0-1".to_string()));
        }
        if params.num_persons < 1 {
            return Err(crate::GeneratorError::InvalidParameter("num_persons must be >= 1".to_string()));
        }
        Ok(())
    }
}

/// Depth ambiguity generator (2D projection issues)
pub struct DepthAmbiguityGenerator;

#[derive(Debug, Clone)]
pub struct DepthAmbiguityParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub ambiguity_strength: f64,    // 0-1 (depth compression)
    pub frame_rate: f64,
}

impl SyntheticGenerator for DepthAmbiguityGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = DepthAmbiguityParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0, 0.01 * params.ambiguity_strength).unwrap();

        // Compress depth (Z) dimension and add uncertainty
        let ambiguous_keypoints: Vec<Vec<[f64; 3]>> = params.keypoints
            .iter()
            .map(|frame| {
                frame.iter()
                    .map(|kp| {
                        [
                            kp[0] + noise.sample(&mut rng),
                            kp[1] + noise.sample(&mut rng),
                            kp[2] * (1.0 - params.ambiguity_strength * 0.5) + noise.sample(&mut rng),
                        ]
                    })
                    .collect()
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(ambiguous_keypoints, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        DepthAmbiguityParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 100],
            ambiguity_strength: 0.5,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.ambiguity_strength < 0.0 || params.ambiguity_strength > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("ambiguity_strength must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Camera motion generator (pan, tilt, zoom)
pub struct CameraMotionGenerator;

#[derive(Debug, Clone)]
pub struct CameraMotionParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub pan_amplitude: f64,         // meters (lateral movement)
    pub tilt_amplitude: f64,        // meters (vertical movement)
    pub zoom_range: (f64, f64),     // (min, max) scale factors
    pub motion_frequency: f64,      // Hz
    pub frame_rate: f64,
}

impl SyntheticGenerator for CameraMotionGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = CameraMotionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        use std::f64::consts::PI;

        let moved_keypoints: Vec<Vec<[f64; 3]>> = params.keypoints
            .iter()
            .enumerate()
            .map(|(frame_idx, frame)| {
                let t = frame_idx as f64 / params.frame_rate;

                // Pan (X-axis shift)
                let pan = params.pan_amplitude * (2.0 * PI * params.motion_frequency * t).sin();

                // Tilt (Y-axis shift)
                let tilt = params.tilt_amplitude * (2.0 * PI * params.motion_frequency * t * 0.7).cos();

                // Zoom (scale)
                let zoom_mid = (params.zoom_range.0 + params.zoom_range.1) / 2.0;
                let zoom_amp = (params.zoom_range.1 - params.zoom_range.0) / 2.0;
                let zoom = zoom_mid + zoom_amp * (2.0 * PI * params.motion_frequency * t * 0.3).sin();

                frame.iter()
                    .map(|kp| {
                        [
                            kp[0] * zoom + pan,
                            kp[1] * zoom + tilt,
                            kp[2] * zoom,
                        ]
                    })
                    .collect()
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(moved_keypoints, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        CameraMotionParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 100],
            pan_amplitude: 0.05,
            tilt_amplitude: 0.03,
            zoom_range: (0.95, 1.05),
            motion_frequency: 0.1,  // Hz (slow drift)
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.zoom_range.0 >= params.zoom_range.1 {
            return Err(crate::GeneratorError::InvalidParameter("zoom_range min must be < max".to_string()));
        }
        Ok(())
    }
}

/// Lighting variation generator
pub struct LightingVariationGenerator;

#[derive(Debug, Clone)]
pub struct LightingVariationParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub confidence_scores: Vec<Vec<f64>>, // per keypoint per frame
    pub illumination_variation: f64,      // 0-1 (affects confidence)
    pub shadow_probability: f64,          // per frame
    pub frame_rate: f64,
}

impl SyntheticGenerator for LightingVariationGenerator {
    type Output = Vec<Vec<([f64; 3], f64)>>; // (keypoint, confidence)
    type GroundTruth = SpatialGroundTruth;
    type Parameters = LightingVariationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let keypoints_with_confidence: Vec<Vec<([f64; 3], f64)>> = params.keypoints
            .iter()
            .enumerate()
            .map(|(frame_idx, frame)| {
                // Random illumination variation
                let illumination_factor = 1.0 - rng.random_range(0.0..1.0) * params.illumination_variation;

                // Shadow events
                let in_shadow = rng.random_range(0.0..1.0) < params.shadow_probability;
                let shadow_factor = if in_shadow { 0.5 } else { 1.0 };

                let base_confidences = if frame_idx < params.confidence_scores.len() {
                    &params.confidence_scores[frame_idx]
                } else {
                    // Default high confidence
                    &vec![0.9; frame.len()]
                };

                frame.iter()
                    .enumerate()
                    .map(|(kp_idx, kp)| {
                        let base_conf = if kp_idx < base_confidences.len() {
                            base_confidences[kp_idx]
                        } else {
                            0.9
                        };
                        let adjusted_conf = (base_conf * illumination_factor * shadow_factor).clamp(0.0, 1.0);
                        (*kp, adjusted_conf)
                    })
                    .collect()
            })
            .collect();

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(keypoints_with_confidence, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        let n_frames = 100;
        let n_keypoints = 33;
        LightingVariationParams {
            keypoints: vec![vec![[1.0; 3]; n_keypoints]; n_frames],
            confidence_scores: vec![vec![0.9; n_keypoints]; n_frames],
            illumination_variation: 0.3,
            shadow_probability: 0.05,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.illumination_variation < 0.0 || params.illumination_variation > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("illumination_variation must be 0-1".to_string()));
        }
        if params.shadow_probability < 0.0 || params.shadow_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("shadow_probability must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Frame rate variation generator
pub struct FrameRateVariationGenerator;

#[derive(Debug, Clone)]
pub struct FrameRateVariationParams {
    pub keypoints: Vec<Vec<[f64; 3]>>,
    pub source_fps: f64,
    pub target_fps: f64,            // 15, 30, 60 fps
    pub frame_rate: f64,
}

impl SyntheticGenerator for FrameRateVariationGenerator {
    type Output = Vec<Vec<[f64; 3]>>;
    type GroundTruth = SpatialGroundTruth;
    type Parameters = FrameRateVariationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        // Resample to target frame rate
        let source_duration = params.keypoints.len() as f64 / params.source_fps;
        let target_n_frames = (source_duration * params.target_fps) as usize;

        let mut resampled_keypoints = Vec::with_capacity(target_n_frames);

        for target_frame in 0..target_n_frames {
            let target_time = target_frame as f64 / params.target_fps;
            let source_frame = (target_time * params.source_fps) as usize;

            if source_frame < params.keypoints.len() {
                resampled_keypoints.push(params.keypoints[source_frame].clone());
            } else if !params.keypoints.is_empty() {
                resampled_keypoints.push(params.keypoints.last().unwrap().clone());
            }
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: params.keypoints.clone(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        let mut result = GeneratedData::new(resampled_keypoints, ground_truth, params.target_fps);
        result.metadata.insert("source_fps".to_string(), params.source_fps.to_string());
        result.metadata.insert("target_fps".to_string(), params.target_fps.to_string());

        Ok(result)
    }

    fn default_params() -> Self::Parameters {
        FrameRateVariationParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 300], // 10 seconds at 30fps
            source_fps: 30.0,
            target_fps: 15.0,
            frame_rate: 30.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.source_fps <= 0.0 || params.target_fps <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("fps values must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypoint_jitter() {
        let generator = KeypointJitterGenerator;
        let params = KeypointJitterGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_occlusion_generation() {
        let generator = OcclusionGenerator;
        let params = OcclusionGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_tracking_dropout() {
        let generator = TrackingDropoutGenerator;
        let params = TrackingDropoutGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_id_switch() {
        let generator = IdSwitchGenerator;
        let params = IdSwitchGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_depth_ambiguity() {
        let generator = DepthAmbiguityGenerator;
        let params = DepthAmbiguityGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_camera_motion() {
        let generator = CameraMotionGenerator;
        let params = CameraMotionGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_lighting_variation() {
        let generator = LightingVariationGenerator;
        let params = LightingVariationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.keypoints.len());
    }

    #[test]
    fn test_frame_rate_variation() {
        let generator = FrameRateVariationGenerator;
        let params = FrameRateVariationParams {
            keypoints: vec![vec![[1.0; 3]; 33]; 300],
            source_fps: 30.0,
            target_fps: 15.0,
            frame_rate: 30.0,
        };
        let result = generator.generate(&params, 42).unwrap();
        // Should have half the frames when going from 30 to 15 fps
        assert!(result.signal.len() < params.keypoints.len());
    }
}
