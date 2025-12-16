//! Multi-modal synchronized generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth, SpatialGroundTruth};
use crate::contact::{ecg, eda, tremor as contact_tremor};
use crate::pose::gait;
use crate::hand::tapping;
use crate::voice::phonation;
use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-modal output structure
#[derive(Debug, Clone)]
pub struct MultiModalOutput {
    // Contact biosignals
    pub ecg: Option<Array1<f64>>,
    pub ppg: Option<Array1<f64>>,
    pub eda: Option<Array1<f64>>,

    // Movement signals
    pub gait_keypoints: Option<Vec<Vec<[f64; 3]>>>,
    pub tremor: Option<Array1<f64>>,
    pub finger_tapping: Option<Vec<f64>>,

    // Voice
    pub voice_f0: Option<Array1<f64>>,

    // Metadata
    pub modalities: Vec<String>,
}

/// Multi-modal ground truth
#[derive(Debug, Clone)]
pub struct MultiModalGroundTruth {
    pub timeseries_gt: TimeSeriesGroundTruth,
    pub spatial_gt: SpatialGroundTruth,
    pub syndrome_label: String,
    pub severity: f64,
}

/// Full Parkinson's disease simulator
pub struct FullPDSimulator;

#[derive(Debug, Clone)]
pub struct FullPDParams {
    pub duration: f64,
    pub sampling_rate: f64,      // for biosignals
    pub video_frame_rate: f64,   // for pose/gait
    pub severity: f64,           // 0-1 (UPDRS-based)
    pub height: f64,             // meters
}

impl SyntheticGenerator for FullPDSimulator {
    type Output = MultiModalOutput;
    type GroundTruth = MultiModalGroundTruth;
    type Parameters = FullPDParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut modalities = Vec::new();

        // 1. Generate ECG with slightly elevated resting HR
        let ecg_params = ecg::EcgMorphologyParams {
            duration: params.duration,
            sampling_rate: params.sampling_rate,
            heart_rate: 70.0 + params.severity * 10.0, // elevated in PD
            p_wave: ecg::WaveParams {
                amplitude: 0.25,
                width: 0.1,
                time_offset: -std::f64::consts::PI / 3.0,
            },
            qrs_complex: ecg::WaveParams {
                amplitude: 1.0,
                width: 0.1,
                time_offset: 0.0,
            },
            t_wave: ecg::WaveParams {
                amplitude: 0.35,
                width: 0.25,
                time_offset: std::f64::consts::PI / 2.0,
            },
        };
        let ecg_gen = ecg::EcgMorphologyGenerator;
        let ecg_result = ecg_gen.generate(&ecg_params, seed)?;
        modalities.push("ECG".to_string());

        // 2. Generate EDA with altered arousal (typically higher in PD)
        let eda_params = eda::EdaTonicParams {
            duration: params.duration,
            sampling_rate: 10.0,
            baseline_scl: 5.0 + params.severity * 2.0,
            drift_magnitude: 0.5,
            drift_frequency: 0.01,
        };
        let eda_gen = eda::EdaTonicGenerator;
        let eda_result = eda_gen.generate(&eda_params, seed + 1)?;
        modalities.push("EDA".to_string());

        // 3. Generate Parkinsonian gait
        let gait_params = gait::PathologicalGaitParams {
            duration: params.duration,
            frame_rate: params.video_frame_rate,
            pathology: gait::GaitPathology::Parkinsonian,
            severity: params.severity,
            baseline_cadence: 100.0 + params.severity * 20.0, // increased cadence in PD
            height: params.height,
        };
        let gait_gen = gait::PathologicalGaitGenerator;
        let gait_result = gait_gen.generate(&gait_params, seed + 2)?;
        modalities.push("Gait".to_string());

        // 4. Generate Parkinsonian rest tremor
        let tremor_params = contact_tremor::ParkinsonianTremorParams {
            duration: params.duration,
            sampling_rate: 100.0,
            frequency: 4.5 + params.severity * 0.5,
            amplitude: 2.0 * params.severity,
            pill_rolling: true,
            amplitude_modulation: 3.0,
        };
        let tremor_gen = contact_tremor::ParkinsonianTremorGenerator;
        let tremor_result = tremor_gen.generate(&tremor_params, seed + 3)?;
        modalities.push("Tremor".to_string());

        // 5. Generate bradykinetic finger tapping
        let tapping_params = tapping::BradykineticTappingParams {
            duration: params.duration.min(10.0), // tapping test typically 10s
            frame_rate: 60.0,
            initial_frequency: 4.0 - params.severity * 1.5,
            frequency_decay: 0.2 * params.severity,
            initial_amplitude: 5.0,
            amplitude_decay: 0.3 * params.severity,
        };
        let tapping_gen = tapping::BradykineticTappingGenerator;
        let tapping_result = tapping_gen.generate(&tapping_params, seed + 4)?;
        modalities.push("FingerTapping".to_string());

        // 6. Generate voice with reduced prosody
        let voice_params = phonation::VoiceTremorParams {
            duration: params.duration.min(5.0),
            sampling_rate: 100.0, // f0 sampling
            baseline_f0: 120.0 - params.severity * 10.0, // monotone in PD
            tremor_frequency: 5.0,
            tremor_extent: 5.0 * params.severity,
        };
        let voice_gen = phonation::VoiceTremorGenerator;
        let voice_result = voice_gen.generate(&voice_params, seed + 5)?;
        modalities.push("Voice".to_string());

        // Construct multi-modal output
        let output = MultiModalOutput {
            ecg: Some(ecg_result.signal),
            ppg: None,
            eda: Some(eda_result.signal),
            gait_keypoints: Some(gait_result.signal),
            tremor: Some(tremor_result.signal),
            finger_tapping: Some(tapping_result.signal),
            voice_f0: Some(voice_result.signal),
            modalities,
        };

        // Construct ground truth
        let mut gt_params = HashMap::new();
        gt_params.insert("severity".to_string(), params.severity);
        gt_params.insert("syndrome".to_string(), 1.0); // PD = 1

        let ground_truth = MultiModalGroundTruth {
            timeseries_gt: TimeSeriesGroundTruth {
                parameters: gt_params,
                events: Vec::new(),
                segments: Vec::new(),
            },
            spatial_gt: gait_result.ground_truth,
            syndrome_label: "Parkinson's Disease".to_string(),
            severity: params.severity,
        };

        Ok(GeneratedData::new(output, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        FullPDParams {
            duration: 30.0,
            sampling_rate: 1000.0,
            video_frame_rate: 30.0,
            severity: 0.5,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Healthy aging simulator
pub struct HealthyAgingSimulator;

#[derive(Debug, Clone)]
pub struct HealthyAgingParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub video_frame_rate: f64,
    pub age: f64,              // years
    pub height: f64,
}

impl SyntheticGenerator for HealthyAgingSimulator {
    type Output = MultiModalOutput;
    type GroundTruth = MultiModalGroundTruth;
    type Parameters = HealthyAgingParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut modalities = Vec::new();

        // Age factor (0-1, normalized by 100 years)
        let age_factor = (params.age - 20.0) / 80.0;
        let age_factor = age_factor.max(0.0).min(1.0);

        // 1. ECG with age-related changes
        let ecg_params = ecg::EcgMorphologyParams {
            duration: params.duration,
            sampling_rate: params.sampling_rate,
            heart_rate: 60.0 + age_factor * 5.0, // slightly elevated
            p_wave: ecg::WaveParams {
                amplitude: 0.25,
                width: 0.1,
                time_offset: -std::f64::consts::PI / 3.0,
            },
            qrs_complex: ecg::WaveParams {
                amplitude: 1.0 - age_factor * 0.1,
                width: 0.1,
                time_offset: 0.0,
            },
            t_wave: ecg::WaveParams {
                amplitude: 0.35,
                width: 0.25,
                time_offset: std::f64::consts::PI / 2.0,
            },
        };
        let ecg_gen = ecg::EcgMorphologyGenerator;
        let ecg_result = ecg_gen.generate(&ecg_params, seed)?;
        modalities.push("ECG".to_string());

        // 2. Normal gait with age-related slowing
        let gait_params = gait::GaitCycleParams {
            duration: params.duration,
            frame_rate: params.video_frame_rate,
            cadence: 110.0 - age_factor * 15.0, // slower with age
            stride_length: 1.4 - age_factor * 0.2,
            step_width: 0.15 + age_factor * 0.05, // wider with age
            height: params.height,
        };
        let gait_gen = gait::GaitCycleGenerator;
        let gait_result = gait_gen.generate(&gait_params, seed + 1)?;
        modalities.push("Gait".to_string());

        // 3. Physiological tremor (increases with age)
        let tremor_params = contact_tremor::PhysiologicalTremorParams {
            duration: params.duration,
            sampling_rate: 100.0,
            frequency: 10.0,
            amplitude: 0.3 + age_factor * 0.2,
            frequency_variability: 0.5,
        };
        let tremor_gen = contact_tremor::PhysiologicalTremorGenerator;
        let tremor_result = tremor_gen.generate(&tremor_params, seed + 2)?;
        modalities.push("Tremor".to_string());

        // 4. Normal finger tapping (slightly slower with age)
        let tapping_params = tapping::NormalTappingParams {
            duration: params.duration.min(10.0),
            frame_rate: 60.0,
            frequency: 4.0 - age_factor * 0.5,
            amplitude: 5.0,
            regularity: 0.9 - age_factor * 0.1,
        };
        let tapping_gen = tapping::NormalTappingGenerator;
        let tapping_result = tapping_gen.generate(&tapping_params, seed + 3)?;
        modalities.push("FingerTapping".to_string());

        let output = MultiModalOutput {
            ecg: Some(ecg_result.signal),
            ppg: None,
            eda: None,
            gait_keypoints: Some(gait_result.signal),
            tremor: Some(tremor_result.signal),
            finger_tapping: Some(tapping_result.signal),
            voice_f0: None,
            modalities,
        };

        let mut gt_params = HashMap::new();
        gt_params.insert("age".to_string(), params.age);
        gt_params.insert("syndrome".to_string(), 0.0); // healthy = 0

        let ground_truth = MultiModalGroundTruth {
            timeseries_gt: TimeSeriesGroundTruth {
                parameters: gt_params,
                events: Vec::new(),
                segments: Vec::new(),
            },
            spatial_gt: gait_result.ground_truth,
            syndrome_label: "Healthy Aging".to_string(),
            severity: 0.0,
        };

        Ok(GeneratedData::new(output, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        HealthyAgingParams {
            duration: 30.0,
            sampling_rate: 1000.0,
            video_frame_rate: 30.0,
            age: 65.0,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.age < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("age must be non-negative".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_full_pd_simulator() {
        let generator = FullPDSimulator;
        let params = FullPDSimulator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert!(result.signal.ecg.is_some());
        assert!(result.signal.eda.is_some());
        assert!(result.signal.gait_keypoints.is_some());
        assert!(result.signal.tremor.is_some());
        assert!(result.signal.finger_tapping.is_some());
        assert!(result.signal.voice_f0.is_some());

        assert_eq!(result.ground_truth.syndrome_label, "Parkinson's Disease");
        assert_eq!(result.ground_truth.severity, params.severity);
    }

    #[test]
    fn test_healthy_aging_simulator() {
        let generator = HealthyAgingSimulator;
        let params = HealthyAgingSimulator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert!(result.signal.ecg.is_some());
        assert!(result.signal.gait_keypoints.is_some());
        assert!(result.signal.tremor.is_some());
        assert!(result.signal.finger_tapping.is_some());

        assert_eq!(result.ground_truth.syndrome_label, "Healthy Aging");
        assert_eq!(result.ground_truth.severity, 0.0);
    }
}
