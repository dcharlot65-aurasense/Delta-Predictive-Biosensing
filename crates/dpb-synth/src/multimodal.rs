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
        let age_factor = age_factor.clamp(0.0, 1.0);

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

// ============================================================================
// COUPLED MULTI-MODAL GENERATORS
// ============================================================================

/// Hand-Voice Tremor Coupling Generator
/// Generates synchronized hand and voice tremor with specified correlation
#[derive(Debug, Clone)]
pub struct HandVoiceTremorOutput {
    pub hand_tremor: Array1<f64>,
    pub voice_tremor: Array1<f64>,
    pub correlation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandVoiceTremorGroundTruth {
    pub hand_frequency: f64,
    pub voice_frequency: f64,
    pub coupling_strength: f64,
    pub severity: f64,
}

impl crate::traits::GroundTruth for HandVoiceTremorGroundTruth {}

#[derive(Debug, Clone)]
pub struct HandVoiceTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub severity: f64,              // 0-1
    pub coupling_strength: f64,     // 0-1 (correlation coefficient)
    pub hand_tremor_freq: f64,      // Hz (typically 4-6 for PD)
    pub voice_tremor_freq: f64,     // Hz (typically 4-6 for PD)
}

pub struct HandVoiceTremorCouplingGenerator;

impl SyntheticGenerator for HandVoiceTremorCouplingGenerator {
    type Output = HandVoiceTremorOutput;
    type GroundTruth = HandVoiceTremorGroundTruth;
    type Parameters = HandVoiceTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        use rand::{RngExt, SeedableRng};
        use rand_distr::{Distribution, Normal};
        use std::f64::consts::PI;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate shared noise source for coupling
        let noise_dist = Normal::new(0.0, 1.0).unwrap();
        let shared_noise: Vec<f64> = (0..n_samples)
            .map(|_| noise_dist.sample(&mut rng))
            .collect();

        // Generate independent noise for each modality
        let hand_noise: Vec<f64> = (0..n_samples)
            .map(|_| noise_dist.sample(&mut rng))
            .collect();

        let voice_noise: Vec<f64> = (0..n_samples)
            .map(|_| noise_dist.sample(&mut rng))
            .collect();

        // Generate coupled tremor signals
        let hand_amplitude = 2.0 * params.severity;
        let voice_amplitude = 5.0 * params.severity; // voice in Hz

        let hand_tremor: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                let phase_noise = 0.1 * shared_noise[i];

                // Coupled component
                let coupled = params.coupling_strength * (2.0 * PI * params.hand_tremor_freq * t + phase_noise).sin();

                // Independent component
                let independent = (1.0 - params.coupling_strength) * (2.0 * PI * params.hand_tremor_freq * t).sin();

                hand_amplitude * (coupled + independent) + 0.1 * hand_noise[i]
            })
            .collect();

        let voice_tremor: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;
                let phase_noise = 0.1 * shared_noise[i];

                // Coupled component
                let coupled = params.coupling_strength * (2.0 * PI * params.voice_tremor_freq * t + phase_noise).sin();

                // Independent component
                let independent = (1.0 - params.coupling_strength) * (2.0 * PI * params.voice_tremor_freq * t).sin();

                voice_amplitude * (coupled + independent) + 0.2 * voice_noise[i]
            })
            .collect();

        // Calculate actual correlation
        let correlation = calculate_correlation(&hand_tremor, &voice_tremor);

        let output = HandVoiceTremorOutput {
            hand_tremor: Array1::from_vec(hand_tremor),
            voice_tremor: Array1::from_vec(voice_tremor),
            correlation,
        };

        let ground_truth = HandVoiceTremorGroundTruth {
            hand_frequency: params.hand_tremor_freq,
            voice_frequency: params.voice_tremor_freq,
            coupling_strength: params.coupling_strength,
            severity: params.severity,
        };

        Ok(GeneratedData::new(output, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        HandVoiceTremorParams {
            duration: 10.0,
            sampling_rate: 100.0,
            severity: 0.7,
            coupling_strength: 0.8,
            hand_tremor_freq: 4.5,
            voice_tremor_freq: 5.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        if params.coupling_strength < 0.0 || params.coupling_strength > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("coupling_strength must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Gait-Speech Rate Coupling Generator
/// Models the coupling between walking cadence and speech rate
#[derive(Debug, Clone)]
pub struct GaitSpeechRateOutput {
    pub gait_keypoints: Vec<Vec<[f64; 3]>>,
    pub syllable_times: Vec<f64>,
    pub gait_cadence: f64,  // steps/min
    pub speech_rate: f64,   // syllables/sec
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitSpeechRateGroundTruth {
    pub baseline_cadence: f64,
    pub baseline_speech_rate: f64,
    pub coupling_strength: f64,
    pub severity: f64,
}

impl crate::traits::GroundTruth for GaitSpeechRateGroundTruth {}

#[derive(Debug, Clone)]
pub struct GaitSpeechRateParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub severity: f64,              // 0-1 (bradykinesia severity)
    pub coupling_strength: f64,     // 0-1
    pub baseline_cadence: f64,      // steps/min
    pub baseline_speech_rate: f64,  // syllables/sec
    pub height: f64,
}

pub struct GaitSpeechRateCouplingGenerator;

impl SyntheticGenerator for GaitSpeechRateCouplingGenerator {
    type Output = GaitSpeechRateOutput;
    type GroundTruth = GaitSpeechRateGroundTruth;
    type Parameters = GaitSpeechRateParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        use rand::{RngExt, SeedableRng};

        // Apply severity to both modalities
        let actual_cadence = params.baseline_cadence * (1.0 - 0.3 * params.severity);
        let actual_speech_rate = params.baseline_speech_rate * (1.0 - 0.4 * params.severity);

        // Generate gait with pathological pattern
        let gait_params = gait::PathologicalGaitParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            pathology: gait::GaitPathology::Parkinsonian,
            severity: params.severity,
            baseline_cadence: actual_cadence,
            height: params.height,
        };
        let gait_gen = gait::PathologicalGaitGenerator;
        let gait_result = gait_gen.generate(&gait_params, seed)?;

        // Generate speech with coupling to gait
        // Speech rate should vary with gait rhythm when coupling is strong
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1);
        let step_duration = 60.0 / actual_cadence; // seconds per step

        let mut syllable_times = Vec::new();
        let mut t = 0.0;

        while t < params.duration {
            syllable_times.push(t);

            // Base interval from speech rate
            let base_interval = 1.0 / actual_speech_rate;

            // Modulation from gait coupling
            let gait_phase = (t / step_duration).fract();
            let gait_modulation = if params.coupling_strength > 0.0 {
                // Speech is slightly faster during stance phase
                1.0 + 0.2 * params.coupling_strength * (2.0 * std::f64::consts::PI * gait_phase).cos()
            } else {
                1.0
            };

            let interval = base_interval * gait_modulation;
            let noise = rng.random_range(-0.05..0.05);
            t += interval * (1.0 + noise);
        }

        let output = GaitSpeechRateOutput {
            gait_keypoints: gait_result.signal,
            syllable_times,
            gait_cadence: actual_cadence,
            speech_rate: actual_speech_rate,
        };

        let ground_truth = GaitSpeechRateGroundTruth {
            baseline_cadence: params.baseline_cadence,
            baseline_speech_rate: params.baseline_speech_rate,
            coupling_strength: params.coupling_strength,
            severity: params.severity,
        };

        Ok(GeneratedData::new(output, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        GaitSpeechRateParams {
            duration: 30.0,
            frame_rate: 30.0,
            severity: 0.6,
            coupling_strength: 0.7,
            baseline_cadence: 110.0,
            baseline_speech_rate: 4.5,
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
        if params.coupling_strength < 0.0 || params.coupling_strength > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("coupling_strength must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Saccade-Reaction Time Coupling Generator
/// Models cognitive-motor coupling between saccades and manual reaction times
#[derive(Debug, Clone)]
pub struct SaccadeReactionTimeOutput {
    pub saccade_positions: Vec<[f64; 2]>,
    pub reaction_times: Vec<f64>,  // ms
    pub stimulus_times: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaccadeReactionTimeGroundTruth {
    pub saccade_latency_mean: f64,
    pub manual_rt_mean: f64,
    pub coupling_strength: f64,
    pub severity: f64,
}

impl crate::traits::GroundTruth for SaccadeReactionTimeGroundTruth {}

#[derive(Debug, Clone)]
pub struct SaccadeReactionTimeParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub severity: f64,              // 0-1 (cognitive slowing)
    pub coupling_strength: f64,     // 0-1
    pub stimulus_times: Vec<f64>,
    pub saccade_amplitude: f64,
}

pub struct SaccadeReactionTimeCouplingGenerator;

impl SyntheticGenerator for SaccadeReactionTimeCouplingGenerator {
    type Output = SaccadeReactionTimeOutput;
    type GroundTruth = SaccadeReactionTimeGroundTruth;
    type Parameters = SaccadeReactionTimeParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        use rand::{RngExt, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Base latencies (increased by severity)
        let base_saccade_latency = 220.0 + 100.0 * params.severity; // ms
        let base_manual_rt = 300.0 + 150.0 * params.severity; // ms

        let saccade_dist = Normal::new(base_saccade_latency, 30.0).unwrap();
        let rt_dist = Normal::new(base_manual_rt, 50.0).unwrap();

        // Generate correlated latencies
        let shared_noise_dist = Normal::new(0.0, 1.0).unwrap();

        let mut saccade_positions = vec![[0.0, 0.0]; n_samples];
        let mut reaction_times = Vec::new();
        let mut current_position = [0.0, 0.0];

        for &stim_time in &params.stimulus_times {
            // Generate shared cognitive delay
            let shared_delay = shared_noise_dist.sample(&mut rng);

            // Saccade latency with coupling
            let saccade_latency = if params.coupling_strength > 0.0 {
                base_saccade_latency + params.coupling_strength * 30.0 * shared_delay
            } else {
                saccade_dist.sample(&mut rng).max(50.0)
            };

            // Manual RT with coupling to same cognitive delay
            let manual_rt = if params.coupling_strength > 0.0 {
                base_manual_rt + params.coupling_strength * 50.0 * shared_delay
            } else {
                rt_dist.sample(&mut rng).max(100.0)
            };

            reaction_times.push(manual_rt);

            // Generate saccade
            let saccade_time = stim_time + saccade_latency / 1000.0;
            let duration = (2.2 * params.saccade_amplitude + 21.0) / 1000.0;
            let start_idx = (saccade_time * params.sampling_rate) as usize;
            let end_idx = ((saccade_time + duration) * params.sampling_rate) as usize;

            let target_position = [params.saccade_amplitude, 0.0];

            for (i_off, i_slot) in saccade_positions[start_idx..end_idx.min(n_samples)].iter_mut().enumerate() {
                let i = start_idx + i_off;
                let progress = (i - start_idx) as f64 / (end_idx - start_idx) as f64;
                let s = 10.0 * (progress - 0.5);
                let position_progress = 1.0 / (1.0 + (-s).exp());

                *i_slot = [
                    current_position[0] + (target_position[0] - current_position[0]) * position_progress,
                    0.0,
                ];
            }

            if end_idx < n_samples {
                current_position = target_position;
                for (i_off, i_slot) in saccade_positions[end_idx..n_samples].iter_mut().enumerate() {
                    let i = end_idx + i_off;
                    *i_slot = current_position;
                }
            }
        }

        let output = SaccadeReactionTimeOutput {
            saccade_positions,
            reaction_times,
            stimulus_times: params.stimulus_times.clone(),
        };

        let ground_truth = SaccadeReactionTimeGroundTruth {
            saccade_latency_mean: base_saccade_latency,
            manual_rt_mean: base_manual_rt,
            coupling_strength: params.coupling_strength,
            severity: params.severity,
        };

        Ok(GeneratedData::new(output, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        SaccadeReactionTimeParams {
            duration: 20.0,
            sampling_rate: 500.0,
            severity: 0.5,
            coupling_strength: 0.75,
            stimulus_times: vec![2.0, 5.0, 8.0, 11.0, 14.0, 17.0],
            saccade_amplitude: 10.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.severity < 0.0 || params.severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("severity must be 0-1".to_string()));
        }
        if params.coupling_strength < 0.0 || params.coupling_strength > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("coupling_strength must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Pupil-Voice Affect Generator
/// Models arousal coherence between pupil dilation and voice affect
#[derive(Debug, Clone)]
pub struct PupilVoiceAffectOutput {
    pub pupil_diameter: Array1<f64>,
    pub voice_f0: Array1<f64>,
    pub arousal_level: Array1<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PupilVoiceAffectGroundTruth {
    pub baseline_pupil: f64,
    pub baseline_f0: f64,
    pub arousal_segments: Vec<(f64, f64, f64)>, // (start, end, arousal)
}

impl crate::traits::GroundTruth for PupilVoiceAffectGroundTruth {}

#[derive(Debug, Clone)]
pub struct PupilVoiceAffectParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub arousal_segments: Vec<(f64, f64, f64)>, // (start, end, arousal 0-1)
    pub baseline_pupil: f64,    // mm
    pub baseline_f0: f64,       // Hz
    pub pupil_dilation_per_arousal: f64,  // mm
    pub f0_increase_per_arousal: f64,     // Hz
}

pub struct PupilVoiceAffectGenerator;

impl SyntheticGenerator for PupilVoiceAffectGenerator {
    type Output = PupilVoiceAffectOutput;
    type GroundTruth = PupilVoiceAffectGroundTruth;
    type Parameters = PupilVoiceAffectParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        use rand::{RngExt, SeedableRng};
        use rand_distr::{Distribution, Normal};

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let pupil_noise = Normal::new(0.0, 0.02).unwrap();
        let f0_noise = Normal::new(0.0, 2.0).unwrap();

        let tau = 1.0; // time constant for exponential smoothing

        let mut arousal_level = Vec::with_capacity(n_samples);
        let mut target_arousal = 0.0;
        let mut current_arousal = 0.0;

        // Generate smooth arousal trajectory
        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Determine target arousal from segments
            target_arousal = 0.0;
            for (start, end, arousal) in &params.arousal_segments {
                if t >= *start && t <= *end {
                    target_arousal = *arousal;
                    break;
                }
            }

            // Exponential approach to target
            let alpha = 1.0 - (-dt / tau).exp();
            current_arousal += alpha * (target_arousal - current_arousal);
            arousal_level.push(current_arousal);
        }

        // Generate pupil and voice signals coupled to arousal
        let pupil_diameter: Vec<f64> = (0..n_samples)
            .map(|i| {
                params.baseline_pupil
                    + params.pupil_dilation_per_arousal * arousal_level[i]
                    + pupil_noise.sample(&mut rng)
            })
            .collect();

        let voice_f0: Vec<f64> = (0..n_samples)
            .map(|i| {
                params.baseline_f0
                    + params.f0_increase_per_arousal * arousal_level[i]
                    + f0_noise.sample(&mut rng)
            })
            .collect();

        let output = PupilVoiceAffectOutput {
            pupil_diameter: Array1::from_vec(pupil_diameter),
            voice_f0: Array1::from_vec(voice_f0),
            arousal_level: Array1::from_vec(arousal_level),
        };

        let ground_truth = PupilVoiceAffectGroundTruth {
            baseline_pupil: params.baseline_pupil,
            baseline_f0: params.baseline_f0,
            arousal_segments: params.arousal_segments.clone(),
        };

        Ok(GeneratedData::new(output, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PupilVoiceAffectParams {
            duration: 60.0,
            sampling_rate: 60.0,
            arousal_segments: vec![
                (10.0, 20.0, 0.5),
                (30.0, 45.0, 0.8),
                (50.0, 55.0, 0.3),
            ],
            baseline_pupil: 4.0,
            baseline_f0: 120.0,
            pupil_dilation_per_arousal: 0.8,
            f0_increase_per_arousal: 40.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.baseline_pupil <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_pupil must be positive".to_string()));
        }
        if params.baseline_f0 <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("baseline_f0 must be positive".to_string()));
        }
        Ok(())
    }
}

/// Bradykinesia-Hypomimia Generator
/// Models motor severity coupling across multiple systems (hand, gait, face)
#[derive(Debug, Clone)]
pub struct BradykinesiaHypomimiaOutput {
    pub finger_tapping: Vec<f64>,
    pub gait_keypoints: Vec<Vec<[f64; 3]>>,
    pub facial_movement_amplitude: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BradykinesiaHypomimiaGroundTruth {
    pub motor_severity: f64,
    pub tapping_frequency: f64,
    pub gait_cadence: f64,
    pub facial_amplitude_reduction: f64,
}

impl crate::traits::GroundTruth for BradykinesiaHypomimiaGroundTruth {}

#[derive(Debug, Clone)]
pub struct BradykinesiaHypomimiaParams {
    pub duration: f64,
    pub frame_rate: f64,
    pub motor_severity: f64,    // 0-1 (unified severity across systems)
    pub height: f64,
}

pub struct BradykinesiaHypomimiaGenerator;

impl SyntheticGenerator for BradykinesiaHypomimiaGenerator {
    type Output = BradykinesiaHypomimiaOutput;
    type GroundTruth = BradykinesiaHypomimiaGroundTruth;
    type Parameters = BradykinesiaHypomimiaParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        // All systems affected proportionally by motor severity

        // 1. Generate bradykinetic finger tapping
        let tapping_params = tapping::BradykineticTappingParams {
            duration: params.duration.min(10.0),
            frame_rate: params.frame_rate,
            initial_frequency: 4.0 - params.motor_severity * 1.5,
            frequency_decay: 0.2 * params.motor_severity,
            initial_amplitude: 5.0,
            amplitude_decay: 0.3 * params.motor_severity,
        };
        let tapping_gen = tapping::BradykineticTappingGenerator;
        let tapping_result = tapping_gen.generate(&tapping_params, seed)?;

        // 2. Generate Parkinsonian gait
        let gait_params = gait::PathologicalGaitParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            pathology: gait::GaitPathology::Parkinsonian,
            severity: params.motor_severity,
            baseline_cadence: 100.0 + params.motor_severity * 20.0,
            height: params.height,
        };
        let gait_gen = gait::PathologicalGaitGenerator;
        let gait_result = gait_gen.generate(&gait_params, seed + 1)?;

        // 3. Calculate facial movement reduction (hypomimia)
        // Normal facial movement amplitude is ~1.0, reduced proportionally
        let facial_amplitude = 1.0 * (1.0 - 0.7 * params.motor_severity);

        let actual_tapping_freq = tapping_params.initial_frequency;
        let actual_gait_cadence = gait_params.baseline_cadence;

        let output = BradykinesiaHypomimiaOutput {
            finger_tapping: tapping_result.signal,
            gait_keypoints: gait_result.signal,
            facial_movement_amplitude: facial_amplitude,
        };

        let ground_truth = BradykinesiaHypomimiaGroundTruth {
            motor_severity: params.motor_severity,
            tapping_frequency: actual_tapping_freq,
            gait_cadence: actual_gait_cadence,
            facial_amplitude_reduction: 1.0 - facial_amplitude,
        };

        Ok(GeneratedData::new(output, ground_truth, params.frame_rate))
    }

    fn default_params() -> Self::Parameters {
        BradykinesiaHypomimiaParams {
            duration: 30.0,
            frame_rate: 60.0,
            motor_severity: 0.7,
            height: 1.75,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.motor_severity < 0.0 || params.motor_severity > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("motor_severity must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Gait-Postural Tremor Generator
/// Models movement-tremor interaction during gait
#[derive(Debug, Clone)]
pub struct GaitPosturalTremorOutput {
    pub gait_keypoints: Vec<Vec<[f64; 3]>>,
    pub tremor: Array1<f64>,
    pub tremor_modulation: Vec<f64>, // tremor amplitude per gait phase
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GaitPosturalTremorGroundTruth {
    pub gait_cadence: f64,
    pub tremor_frequency: f64,
    pub tremor_amplitude: f64,
    pub severity: f64,
}

impl crate::traits::GroundTruth for GaitPosturalTremorGroundTruth {}

#[derive(Debug, Clone)]
pub struct GaitPosturalTremorParams {
    pub duration: f64,
    pub sampling_rate: f64,     // for tremor signal
    pub frame_rate: f64,        // for gait
    pub severity: f64,          // 0-1
    pub baseline_cadence: f64,
    pub tremor_frequency: f64,  // Hz
    pub tremor_amplitude: f64,
    pub height: f64,
}

pub struct GaitPosturalTremorGenerator;

impl SyntheticGenerator for GaitPosturalTremorGenerator {
    type Output = GaitPosturalTremorOutput;
    type GroundTruth = GaitPosturalTremorGroundTruth;
    type Parameters = GaitPosturalTremorParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        use rand::{RngExt, SeedableRng};
        use std::f64::consts::PI;

        // Generate gait pattern
        let gait_params = gait::GaitCycleParams {
            duration: params.duration,
            frame_rate: params.frame_rate,
            cadence: params.baseline_cadence,
            stride_length: 1.4 - params.severity * 0.2,
            step_width: 0.15,
            height: params.height,
        };
        let gait_gen = gait::GaitCycleGenerator;
        let gait_result = gait_gen.generate(&gait_params, seed)?;

        // Generate tremor modulated by gait phase
        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed + 1);

        let step_duration = 60.0 / params.baseline_cadence; // seconds per step

        let tremor: Vec<f64> = (0..n_samples)
            .map(|i| {
                let t = i as f64 * dt;

                // Gait phase (0-1)
                let gait_phase = (t / step_duration).fract();

                // Tremor increases during swing phase (reduced postural control)
                // Stance: phase 0-0.6, Swing: phase 0.6-1.0
                let phase_modulation = if gait_phase > 0.6 {
                    1.0 + 0.5 * params.severity // increased tremor in swing
                } else {
                    1.0 - 0.2 * params.severity // slightly reduced in stance
                };

                // Generate tremor signal
                let phase_noise = rng.random_range(-0.1..0.1);
                params.tremor_amplitude * phase_modulation
                    * (2.0 * PI * params.tremor_frequency * t + phase_noise).sin()
            })
            .collect();

        // Calculate tremor modulation per gait cycle
        let n_cycles = (params.duration / step_duration) as usize;
        let tremor_modulation: Vec<f64> = (0..n_cycles)
            .map(|i| {
                let start_idx = (i as f64 * step_duration * params.sampling_rate) as usize;
                let end_idx = std::cmp::min(
                    ((i + 1) as f64 * step_duration * params.sampling_rate) as usize,
                    tremor.len()
                );

                if start_idx < end_idx {
                    let cycle_tremor = &tremor[start_idx..end_idx];
                    cycle_tremor.iter().map(|x| x.abs()).sum::<f64>() / cycle_tremor.len() as f64
                } else {
                    0.0
                }
            })
            .collect();

        let output = GaitPosturalTremorOutput {
            gait_keypoints: gait_result.signal,
            tremor: Array1::from_vec(tremor),
            tremor_modulation,
        };

        let ground_truth = GaitPosturalTremorGroundTruth {
            gait_cadence: params.baseline_cadence,
            tremor_frequency: params.tremor_frequency,
            tremor_amplitude: params.tremor_amplitude,
            severity: params.severity,
        };

        Ok(GeneratedData::new(output, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        GaitPosturalTremorParams {
            duration: 30.0,
            sampling_rate: 100.0,
            frame_rate: 30.0,
            severity: 0.6,
            baseline_cadence: 110.0,
            tremor_frequency: 6.0,
            tremor_amplitude: 2.0,
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

// Helper function for correlation calculation
fn calculate_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.is_empty() {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut denom_x = 0.0;
    let mut denom_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        num += dx * dy;
        denom_x += dx * dx;
        denom_y += dy * dy;
    }

    if denom_x == 0.0 || denom_y == 0.0 {
        return 0.0;
    }

    num / (denom_x * denom_y).sqrt()
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

    // Tests for coupled generators
    #[test]
    fn test_hand_voice_tremor_coupling() {
        let generator = HandVoiceTremorCouplingGenerator;
        let params = HandVoiceTremorCouplingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert_eq!(result.signal.hand_tremor.len(), (params.duration * params.sampling_rate) as usize);
        assert_eq!(result.signal.voice_tremor.len(), (params.duration * params.sampling_rate) as usize);
        assert!(result.signal.correlation >= -1.0 && result.signal.correlation <= 1.0);
    }

    #[test]
    fn test_gait_speech_rate_coupling() {
        let generator = GaitSpeechRateCouplingGenerator;
        let params = GaitSpeechRateCouplingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert!(!result.signal.gait_keypoints.is_empty());
        assert!(!result.signal.syllable_times.is_empty());
    }

    #[test]
    fn test_saccade_reaction_time_coupling() {
        let generator = SaccadeReactionTimeCouplingGenerator;
        let params = SaccadeReactionTimeCouplingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert_eq!(result.signal.saccade_positions.len(), (params.duration * params.sampling_rate) as usize);
        assert!(!result.signal.reaction_times.is_empty());
    }

    #[test]
    fn test_pupil_voice_affect_generator() {
        let generator = PupilVoiceAffectGenerator;
        let params = PupilVoiceAffectGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert_eq!(result.signal.pupil_diameter.len(), (params.duration * params.sampling_rate) as usize);
        assert_eq!(result.signal.voice_f0.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_bradykinesia_hypomimia_generator() {
        let generator = BradykinesiaHypomimiaGenerator;
        let params = BradykinesiaHypomimiaGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert!(!result.signal.finger_tapping.is_empty());
        assert!(!result.signal.gait_keypoints.is_empty());
    }

    #[test]
    fn test_gait_postural_tremor_generator() {
        let generator = GaitPosturalTremorGenerator;
        let params = GaitPosturalTremorGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();

        assert!(!result.signal.gait_keypoints.is_empty());
        assert_eq!(result.signal.tremor.len(), (params.duration * params.sampling_rate) as usize);
    }
}
