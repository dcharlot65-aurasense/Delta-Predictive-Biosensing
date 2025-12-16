//! Pose noise and artifact generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{Rng, SeedableRng};
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
                        if rng.r#gen::<f64>() < params.occlusion_probability {
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
                if rng.r#gen::<f64>() < params.dropout_probability {
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
}
