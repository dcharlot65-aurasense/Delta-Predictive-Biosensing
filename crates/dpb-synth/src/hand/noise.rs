//! Hand tracking noise and artifact generators

use crate::traits::{SyntheticGenerator, GeneratedData, SpatialGroundTruth};
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal, Bernoulli};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Landmark jitter generator (per-landmark tracking noise)
pub struct LandmarkJitterGenerator;

#[derive(Debug, Clone)]
pub struct LandmarkJitterParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub n_landmarks: usize,
    pub jitter_std_per_landmark: Vec<f64>, // Standard deviation per landmark (cm)
    pub temporal_correlation: f64,          // 0-1 (1 = perfectly correlated across time)
}

impl SyntheticGenerator for LandmarkJitterGenerator {
    type Output = Vec<Vec<[f64; 3]>>; // [frames][landmarks][xyz]
    type GroundTruth = SpatialGroundTruth;
    type Parameters = LandmarkJitterParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut positions = Vec::with_capacity(n_frames);
        let mut prev_jitter: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]; params.n_landmarks];

        for _ in 0..n_frames {
            let mut frame_landmarks = Vec::with_capacity(params.n_landmarks);

            // The body reads prev_jitter alongside writing it, so iterating
            // mutably over the same vector does not borrow-check.
            #[allow(clippy::needless_range_loop)]
            for landmark_idx in 0..params.n_landmarks {
                let std = if landmark_idx < params.jitter_std_per_landmark.len() {
                    params.jitter_std_per_landmark[landmark_idx]
                } else {
                    0.1 // default
                };

                let noise_dist = Normal::new(0.0, std).unwrap();

                // Generate new jitter with temporal correlation
                let new_jitter = [
                    noise_dist.sample(&mut rng),
                    noise_dist.sample(&mut rng),
                    noise_dist.sample(&mut rng),
                ];

                let jitter = [
                    params.temporal_correlation * prev_jitter[landmark_idx][0] +
                        (1.0 - params.temporal_correlation) * new_jitter[0],
                    params.temporal_correlation * prev_jitter[landmark_idx][1] +
                        (1.0 - params.temporal_correlation) * new_jitter[1],
                    params.temporal_correlation * prev_jitter[landmark_idx][2] +
                        (1.0 - params.temporal_correlation) * new_jitter[2],
                ];

                frame_landmarks.push(jitter);
                prev_jitter[landmark_idx] = jitter;
            }

            positions.push(frame_landmarks);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(positions, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        LandmarkJitterParams {
            duration: 10.0,
            frame_rate: 60.0,
            n_landmarks: 21, // MediaPipe hand landmarks
            jitter_std_per_landmark: vec![0.1; 21],
            temporal_correlation: 0.8,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.temporal_correlation < 0.0 || params.temporal_correlation > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("temporal_correlation must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Self-occlusion generator (finger occlusion patterns)
pub struct SelfOcclusionGenerator;

#[derive(Debug, Clone)]
pub struct SelfOcclusionParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub hand_orientation: Vec<f64>, // Hand orientation angles over time (degrees)
    pub occlusion_threshold_angle: f64, // Angle at which occlusion occurs
    pub n_landmarks: usize,
}

impl SyntheticGenerator for SelfOcclusionGenerator {
    type Output = Vec<Vec<bool>>; // [frames][landmarks] - true if visible
    type GroundTruth = SpatialGroundTruth;
    type Parameters = SelfOcclusionParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let _rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut visibility = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let angle = if i < params.hand_orientation.len() {
                params.hand_orientation[i]
            } else {
                0.0
            };

            let mut frame_visibility = Vec::with_capacity(params.n_landmarks);

            for landmark_idx in 0..params.n_landmarks {
                // Simplified occlusion model: back of hand landmarks occluded when hand rotated
                let is_back_landmark = landmark_idx % 4 == 3; // Simple heuristic
                let is_occluded = is_back_landmark &&
                    angle.abs() > params.occlusion_threshold_angle;

                frame_visibility.push(!is_occluded);
            }

            visibility.push(frame_visibility);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(visibility, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        SelfOcclusionParams {
            duration: 10.0,
            frame_rate: 60.0,
            hand_orientation: vec![0.0; 600], // 10s at 60fps
            occlusion_threshold_angle: 90.0,
            n_landmarks: 21,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        Ok(())
    }
}

/// Hand tracking loss generator
pub struct HandTrackingLossGenerator;

#[derive(Debug, Clone)]
pub struct HandTrackingLossParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub loss_probability: f64,         // Per-frame probability of tracking loss
    pub min_loss_duration: f64,        // Minimum loss duration (seconds)
    pub max_loss_duration: f64,        // Maximum loss duration (seconds)
}

impl SyntheticGenerator for HandTrackingLossGenerator {
    type Output = Vec<bool>; // true if tracking is active
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandTrackingLossParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let dt = 1.0 / params.frame_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let loss_dist = Bernoulli::new(params.loss_probability).unwrap();

        let mut tracking_active = Vec::with_capacity(n_frames);
        let mut loss_remaining = 0.0;

        for _ in 0..n_frames {
            if loss_remaining > 0.0 {
                tracking_active.push(false);
                loss_remaining -= dt;
            } else {
                // Check for new loss
                if loss_dist.sample(&mut rng) {
                    let loss_duration = rng.random_range(params.min_loss_duration..=params.max_loss_duration);
                    loss_remaining = loss_duration;
                    tracking_active.push(false);
                } else {
                    tracking_active.push(true);
                }
            }
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(tracking_active, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        HandTrackingLossParams {
            duration: 10.0,
            frame_rate: 60.0,
            loss_probability: 0.01,
            min_loss_duration: 0.1,
            max_loss_duration: 0.5,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.loss_probability < 0.0 || params.loss_probability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("loss_probability must be 0-1".to_string()));
        }
        if params.min_loss_duration > params.max_loss_duration {
            return Err(crate::GeneratorError::InvalidParameter("min_loss_duration must be <= max_loss_duration".to_string()));
        }
        Ok(())
    }
}

/// Hand depth error generator
pub struct HandDepthErrorGenerator;

#[derive(Debug, Clone)]
pub struct HandDepthErrorParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub depth_noise_std: f64,       // Depth noise standard deviation (cm)
    pub depth_bias: f64,             // Systematic depth bias (cm)
    pub distance_dependency: f64,    // Noise increases with distance (0-1)
}

impl SyntheticGenerator for HandDepthErrorGenerator {
    type Output = Vec<f64>; // depth error over time (cm)
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandDepthErrorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let base_noise = Normal::new(0.0, params.depth_noise_std).unwrap();

        let mut depth_errors = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            // Simulate varying distance (simple model)
            let t = i as f64 / n_frames as f64;
            let distance_factor = 1.0 + params.distance_dependency * (2.0 * PI * t).sin();

            let noise = base_noise.sample(&mut rng) * distance_factor;
            let error = params.depth_bias + noise;

            depth_errors.push(error);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(depth_errors, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        HandDepthErrorParams {
            duration: 10.0,
            frame_rate: 60.0,
            depth_noise_std: 0.5,
            depth_bias: 0.0,
            distance_dependency: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.distance_dependency < 0.0 || params.distance_dependency > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("distance_dependency must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Hand motion blur generator
pub struct HandMotionBlurGenerator;

#[derive(Debug, Clone)]
pub struct HandMotionBlurParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub velocities: Vec<f64>,        // Hand velocities over time (cm/s)
    pub blur_threshold_velocity: f64, // Velocity at which blur starts (cm/s)
    pub blur_kernel_size_max: f64,   // Maximum blur kernel size
}

impl SyntheticGenerator for HandMotionBlurGenerator {
    type Output = Vec<f64>; // blur kernel size over time
    type GroundTruth = SpatialGroundTruth;
    type Parameters = HandMotionBlurParams;

    fn generate(&self, params: &Self::Parameters, _seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;

        let mut blur_kernels = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let velocity = if i < params.velocities.len() {
                params.velocities[i]
            } else {
                0.0
            };

            // Blur increases linearly with velocity above threshold
            let blur = if velocity > params.blur_threshold_velocity {
                let excess_velocity = velocity - params.blur_threshold_velocity;
                let normalized = excess_velocity / params.blur_threshold_velocity;
                (normalized * params.blur_kernel_size_max).min(params.blur_kernel_size_max)
            } else {
                0.0
            };

            blur_kernels.push(blur);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(blur_kernels, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        HandMotionBlurParams {
            duration: 10.0,
            frame_rate: 60.0,
            velocities: vec![10.0; 600], // Constant velocity
            blur_threshold_velocity: 50.0,
            blur_kernel_size_max: 10.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.blur_threshold_velocity < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("blur_threshold_velocity must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Background clutter generator
pub struct BackgroundClutterGenerator;

#[derive(Debug, Clone)]
pub struct BackgroundClutterParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub n_distractor_objects: usize,  // Number of hand-like distractors
    pub false_positive_rate: f64,     // Rate of false hand detections
    pub clutter_density: f64,          // 0-1 (density of background clutter)
}

impl SyntheticGenerator for BackgroundClutterGenerator {
    type Output = Vec<Vec<[f64; 3]>>; // [frames][objects][xyz] - distractor positions
    type GroundTruth = SpatialGroundTruth;
    type Parameters = BackgroundClutterParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let position_dist = Normal::new(0.0, 20.0).unwrap(); // 20cm spread

        let mut clutter_data = Vec::with_capacity(n_frames);

        for _ in 0..n_frames {
            let mut frame_objects = Vec::new();

            for _ in 0..params.n_distractor_objects {
                // Random chance of distractor appearing based on clutter density
                if rng.random_range(0.0..1.0) < params.clutter_density {
                    let depth: f64 = position_dist.sample(&mut rng);
                    let object = [
                        position_dist.sample(&mut rng),
                        position_dist.sample(&mut rng),
                        depth.abs(), // positive depth
                    ];
                    frame_objects.push(object);
                }
            }

            clutter_data.push(frame_objects);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(clutter_data, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        BackgroundClutterParams {
            duration: 10.0,
            frame_rate: 60.0,
            n_distractor_objects: 3,
            false_positive_rate: 0.05,
            clutter_density: 0.3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.clutter_density < 0.0 || params.clutter_density > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("clutter_density must be 0-1".to_string()));
        }
        if params.false_positive_rate < 0.0 || params.false_positive_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("false_positive_rate must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Skin tone variation generator
pub struct SkinToneVariationGenerator;

#[derive(Debug, Clone)]
pub struct SkinToneVariationParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub lighting_variability: f64,   // 0-1 (lighting changes)
    pub skin_tone_base: [f64; 3],    // RGB base skin tone (0-1)
    pub ambient_light_color: [f64; 3], // RGB ambient light (0-1)
}

impl SyntheticGenerator for SkinToneVariationGenerator {
    type Output = Vec<[f64; 3]>; // RGB values over time
    type GroundTruth = SpatialGroundTruth;
    type Parameters = SkinToneVariationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_frames = (params.duration * params.frame_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let lighting_noise = Normal::new(1.0, params.lighting_variability * 0.2).unwrap();

        let mut skin_tones = Vec::with_capacity(n_frames);

        for i in 0..n_frames {
            let t = i as f64 / n_frames as f64;

            // Simulate lighting changes
            let lighting_factor = lighting_noise.sample(&mut rng).clamp(0.3, 1.5);

            // Ambient light influence
            let ambient_influence = 0.2 * (2.0 * PI * t).sin();

            let r = (params.skin_tone_base[0] * lighting_factor +
                    params.ambient_light_color[0] * ambient_influence).clamp(0.0, 1.0);
            let g = (params.skin_tone_base[1] * lighting_factor +
                    params.ambient_light_color[1] * ambient_influence).clamp(0.0, 1.0);
            let b = (params.skin_tone_base[2] * lighting_factor +
                    params.ambient_light_color[2] * ambient_influence).clamp(0.0, 1.0);

            skin_tones.push([r, g, b]);
        }

        let ground_truth = SpatialGroundTruth {
            keypoints: Vec::new(),
            joint_angles: HashMap::new(),
            gait_phases: Vec::new(),
        };

        Ok(GeneratedData::new(skin_tones, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        SkinToneVariationParams {
            duration: 10.0,
            frame_rate: 60.0,
            lighting_variability: 0.3,
            skin_tone_base: [0.8, 0.6, 0.5],
            ambient_light_color: [1.0, 1.0, 0.9],
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.lighting_variability < 0.0 || params.lighting_variability > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("lighting_variability must be 0-1".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landmark_jitter() {
        let generator = LandmarkJitterGenerator;
        let params = LandmarkJitterGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
        assert_eq!(result.signal[0].len(), params.n_landmarks);
    }

    #[test]
    fn test_self_occlusion() {
        let generator = SelfOcclusionGenerator;
        let params = SelfOcclusionGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_tracking_loss() {
        let generator = HandTrackingLossGenerator;
        let params = HandTrackingLossGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_depth_error() {
        let generator = HandDepthErrorGenerator;
        let params = HandDepthErrorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_motion_blur() {
        let generator = HandMotionBlurGenerator;
        let params = HandMotionBlurGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_background_clutter() {
        let generator = BackgroundClutterGenerator;
        let params = BackgroundClutterGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);
    }

    #[test]
    fn test_skin_tone_variation() {
        let generator = SkinToneVariationGenerator;
        let params = SkinToneVariationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.frame_rate) as usize);

        // Check RGB values are in valid range
        for &rgb in &result.signal {
            assert!(rgb[0] >= 0.0 && rgb[0] <= 1.0);
            assert!(rgb[1] >= 0.0 && rgb[1] <= 1.0);
            assert!(rgb[2] >= 0.0 && rgb[2] <= 1.0);
        }
    }
}
