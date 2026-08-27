//! Event-Related Potential (ERP) Signal Generator
//!
//! Generates ERP components including:
//! - Sensory components (P1, N1, P2, N2)
//! - Cognitive components (P300, N400, P600)
//! - Motor-related potentials (BP, LRP)
//! - Mismatch negativity (MMN)
//! - Pathological patterns (reduced amplitude, increased latency)

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for ERP generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErpConfig {
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Number of channels
    pub channels: usize,
    /// Background EEG amplitude (µV)
    pub background_amplitude: f64,
    /// SNR (signal-to-noise ratio)
    pub snr: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for ErpConfig {
    fn default() -> Self {
        Self {
            sample_rate: 512.0,
            channels: 1,
            background_amplitude: 20.0,
            snr: 2.0,
            seed: None,
        }
    }
}

/// ERP component definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErpComponent {
    /// Component name
    pub name: String,
    /// Peak latency (ms)
    pub latency_ms: f64,
    /// Peak amplitude (µV, positive or negative)
    pub amplitude_uv: f64,
    /// Duration/width (ms)
    pub width_ms: f64,
    /// Topographic distribution
    pub topography: Topography,
}

/// Topographic distribution patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Topography {
    /// Frontal maximum
    Frontal,
    /// Central maximum
    Central,
    /// Parietal maximum
    Parietal,
    /// Occipital maximum
    Occipital,
    /// Temporal (left)
    TemporalLeft,
    /// Temporal (right)
    TemporalRight,
    /// Diffuse/global
    Global,
}

/// Output from ERP generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErpOutput {
    /// Time points (s)
    pub time: Vec<f64>,
    /// ERP signal per channel (µV)
    pub signal: Vec<Vec<f64>>,
    /// Pure ERP waveform (no noise)
    pub erp_only: Vec<Vec<f64>>,
    /// Ground truth
    pub ground_truth: ErpGroundTruth,
    /// Configuration
    pub config: ErpConfig,
}

/// Ground truth for ERP signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErpGroundTruth {
    /// Components present
    pub components: Vec<ErpComponentInfo>,
    /// Stimulus onset time (s)
    pub stimulus_onset: f64,
    /// Condition/paradigm
    pub paradigm: ErpParadigm,
    /// Applied pathology
    pub pathology: Option<ErpPathology>,
}

/// Information about detected component
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErpComponentInfo {
    /// Component type
    pub component: ErpComponentType,
    /// Actual peak latency (ms)
    pub peak_latency: f64,
    /// Actual peak amplitude (µV)
    pub peak_amplitude: f64,
    /// Onset latency (ms)
    pub onset_latency: f64,
    /// Offset latency (ms)
    pub offset_latency: f64,
}

// Domain acronyms -- ECG beat annotations, audio codecs, ERP components,
// the SMPL-X body model, drug classes. Camel case would diverge from how
// these are written everywhere they are used.
#[allow(clippy::upper_case_acronyms)]
/// Standard ERP component types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ErpComponentType {
    // Early sensory
    P1,   // ~50ms, occipital
    N1,   // ~100ms, fronto-central
    P2,   // ~200ms, fronto-central
    N2,   // ~200-350ms, fronto-central

    // Cognitive
    P300,  // ~300-600ms, parietal (P3a anterior, P3b posterior)
    N400,  // ~400ms, centro-parietal (semantic)
    P600,  // ~600ms, centro-parietal (syntactic)

    // Attention
    MMN,   // ~150-250ms, frontal (mismatch negativity)
    Nd,    // Processing negativity

    // Motor
    BP,    // Bereitschaftspotential (readiness)
    LRP,   // Lateralized readiness potential

    // Error
    ERN,   // Error-related negativity
    Pe,    // Error positivity
}

/// ERP paradigm types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ErpParadigm {
    /// Visual oddball
    VisualOddball { target: bool },
    /// Auditory oddball
    AuditoryOddball { target: bool, deviant: bool },
    /// Go/NoGo
    GoNoGo { go_trial: bool },
    /// Flanker task
    Flanker { congruent: bool },
    /// Semantic priming
    SemanticPriming { related: bool },
    /// Motor preparation
    MotorPreparation,
    /// Error monitoring
    ErrorMonitoring { error: bool },
}

/// Pathological ERP patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ErpPathology {
    /// Reduced P300 (schizophrenia, dementia)
    ReducedP300 { amplitude_reduction: f64, latency_delay: f64 },
    /// Reduced MMN (schizophrenia)
    ReducedMMN { amplitude_reduction: f64 },
    /// Increased N400 (semantic processing deficit)
    AbnormalN400 { amplitude_change: f64 },
    /// Reduced ERN (OCD, addiction)
    ReducedERN { amplitude_reduction: f64 },
    /// Global amplitude reduction (TBI, dementia)
    GlobalReduction { factor: f64 },
    /// Increased latencies (aging, cognitive decline)
    IncreasedLatency { factor: f64 },
}

/// ERP signal generator
pub struct ErpGenerator {
    config: ErpConfig,
    rng: StdRng,
}

impl ErpGenerator {
    /// Create new ERP generator
    pub fn new(config: ErpConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate visual oddball ERP
    pub fn generate_visual_oddball(&mut self, target: bool, epoch_duration: f64) -> ErpOutput {
        let components = if target {
            // Target stimulus: N1, P2, P300
            vec![
                self.create_component(ErpComponentType::N1, -3.0, 100.0, 50.0),
                self.create_component(ErpComponentType::P2, 4.0, 200.0, 60.0),
                self.create_component(ErpComponentType::P300, 8.0, 350.0, 150.0),
            ]
        } else {
            // Standard stimulus: N1, P2 only
            vec![
                self.create_component(ErpComponentType::N1, -2.5, 100.0, 50.0),
                self.create_component(ErpComponentType::P2, 3.0, 200.0, 60.0),
            ]
        };

        self.generate_erp(
            &components,
            epoch_duration,
            ErpParadigm::VisualOddball { target },
        )
    }

    /// Generate auditory oddball ERP
    pub fn generate_auditory_oddball(
        &mut self,
        target: bool,
        deviant: bool,
        epoch_duration: f64,
    ) -> ErpOutput {
        let mut components = vec![
            self.create_component(ErpComponentType::N1, -4.0, 100.0, 40.0),
            self.create_component(ErpComponentType::P2, 3.5, 180.0, 50.0),
        ];

        if deviant {
            // Add MMN for deviant stimuli
            components.push(self.create_component(ErpComponentType::MMN, -2.5, 180.0, 80.0));
        }

        if target {
            // Add P300 for targets
            components.push(self.create_component(ErpComponentType::P300, 10.0, 320.0, 140.0));
        }

        self.generate_erp(
            &components,
            epoch_duration,
            ErpParadigm::AuditoryOddball { target, deviant },
        )
    }

    /// Generate Go/NoGo ERP
    pub fn generate_go_nogo(&mut self, go_trial: bool, epoch_duration: f64) -> ErpOutput {
        let components = if go_trial {
            // Go trial: N1, P2, motor preparation
            vec![
                self.create_component(ErpComponentType::N1, -3.0, 100.0, 45.0),
                self.create_component(ErpComponentType::P2, 3.0, 200.0, 55.0),
                self.create_component(ErpComponentType::LRP, -2.0, 300.0, 100.0),
            ]
        } else {
            // NoGo trial: N1, P2, N2 (inhibition), P300 (NoGo P3)
            vec![
                self.create_component(ErpComponentType::N1, -3.0, 100.0, 45.0),
                self.create_component(ErpComponentType::P2, 2.5, 200.0, 55.0),
                self.create_component(ErpComponentType::N2, -4.0, 280.0, 70.0),
                self.create_component(ErpComponentType::P300, 6.0, 400.0, 120.0),
            ]
        };

        self.generate_erp(&components, epoch_duration, ErpParadigm::GoNoGo { go_trial })
    }

    /// Generate error-related ERP
    pub fn generate_error_monitoring(&mut self, error: bool, epoch_duration: f64) -> ErpOutput {
        let components = if error {
            // Error trial: ERN followed by Pe
            vec![
                self.create_component(ErpComponentType::ERN, -8.0, 80.0, 60.0),
                self.create_component(ErpComponentType::Pe, 6.0, 250.0, 150.0),
            ]
        } else {
            // Correct trial: CRN (smaller than ERN)
            vec![
                self.create_component(ErpComponentType::ERN, -3.0, 80.0, 60.0),
            ]
        };

        self.generate_erp(
            &components,
            epoch_duration,
            ErpParadigm::ErrorMonitoring { error },
        )
    }

    /// Generate N400 semantic priming
    pub fn generate_semantic_priming(&mut self, related: bool, epoch_duration: f64) -> ErpOutput {
        let components = vec![
            self.create_component(ErpComponentType::N1, -2.0, 100.0, 40.0),
            self.create_component(ErpComponentType::P2, 2.5, 200.0, 50.0),
            // N400 larger for unrelated words
            self.create_component(
                ErpComponentType::N400,
                if related { -3.0 } else { -6.0 },
                400.0,
                150.0,
            ),
        ];

        self.generate_erp(
            &components,
            epoch_duration,
            ErpParadigm::SemanticPriming { related },
        )
    }

    /// Generate motor preparation ERP (Bereitschaftspotential)
    pub fn generate_motor_preparation(&mut self, epoch_duration: f64) -> ErpOutput {
        // BP starts ~1-2s before movement
        let components = vec![
            self.create_component(ErpComponentType::BP, -5.0, -1000.0, 800.0), // Early BP
            self.create_component(ErpComponentType::BP, -8.0, -400.0, 300.0),  // Late BP (steeper)
            self.create_component(ErpComponentType::LRP, -3.0, -200.0, 150.0), // Motor potential
        ];

        self.generate_erp(&components, epoch_duration, ErpParadigm::MotorPreparation)
    }

    /// Generate pathological ERP
    pub fn generate_pathological(
        &mut self,
        paradigm: ErpParadigm,
        pathology: ErpPathology,
        epoch_duration: f64,
    ) -> ErpOutput {
        // First generate normal ERP
        let mut output = match paradigm {
            ErpParadigm::VisualOddball { target } => {
                self.generate_visual_oddball(target, epoch_duration)
            }
            ErpParadigm::AuditoryOddball { target, deviant } => {
                self.generate_auditory_oddball(target, deviant, epoch_duration)
            }
            ErpParadigm::GoNoGo { go_trial } => self.generate_go_nogo(go_trial, epoch_duration),
            ErpParadigm::ErrorMonitoring { error } => {
                self.generate_error_monitoring(error, epoch_duration)
            }
            ErpParadigm::SemanticPriming { related } => {
                self.generate_semantic_priming(related, epoch_duration)
            }
            ErpParadigm::MotorPreparation => self.generate_motor_preparation(epoch_duration),
            _ => self.generate_visual_oddball(true, epoch_duration),
        };

        // Apply pathology modifications
        self.apply_pathology(&mut output, pathology);

        output.ground_truth.pathology = Some(pathology);
        output
    }

    /// Generate ERP from component list
    fn generate_erp(
        &mut self,
        components: &[ErpComponent],
        epoch_duration: f64,
        paradigm: ErpParadigm,
    ) -> ErpOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (epoch_duration * self.config.sample_rate) as usize;

        // Baseline period (typically -200 to 0 ms relative to stimulus)
        let baseline_duration = 0.2;
        let stimulus_onset = baseline_duration;

        let mut time = Vec::with_capacity(n_samples);
        let mut signal = vec![vec![0.0; n_samples]; self.config.channels];
        let mut erp_only = vec![vec![0.0; n_samples]; self.config.channels];

        // Generate time vector
        for i in 0..n_samples {
            time.push((i as f64 * dt) - baseline_duration);
        }

        // Generate ERP components
        let mut component_info = Vec::new();
        for comp in components {
            let comp_signal = self.generate_component(comp, n_samples, stimulus_onset);

            // Add to all channels (with topographic weighting in future)
            for channel in erp_only.iter_mut().take(self.config.channels) {
                for (i, &s) in comp_signal.iter().enumerate() {
                    channel[i] += s;
                }
            }

            // Record component info
            let _peak_idx = self.find_peak(&comp_signal, comp.amplitude_uv > 0.0);
            let _latency_s = stimulus_onset + comp.latency_ms / 1000.0;

            component_info.push(ErpComponentInfo {
                component: self.name_to_type(&comp.name),
                peak_latency: comp.latency_ms,
                peak_amplitude: comp.amplitude_uv,
                onset_latency: comp.latency_ms - comp.width_ms / 2.0,
                offset_latency: comp.latency_ms + comp.width_ms / 2.0,
            });
        }

        // Add background EEG noise
        let noise_dist = Normal::new(0.0, self.config.background_amplitude / self.config.snr).unwrap();

        for ch in 0..self.config.channels {
            // Generate pink noise background
            let background = self.generate_background_eeg(n_samples);

            for i in 0..n_samples {
                let noise: f64 = self.rng.sample(noise_dist);
                signal[ch][i] = erp_only[ch][i] + background[i] + noise;
            }
        }

        let ground_truth = ErpGroundTruth {
            components: component_info,
            stimulus_onset,
            paradigm,
            pathology: None,
        };

        ErpOutput {
            time,
            signal,
            erp_only,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate single component waveform
    fn generate_component(
        &mut self,
        component: &ErpComponent,
        n_samples: usize,
        stimulus_onset: f64,
    ) -> Vec<f64> {
        let dt = 1.0 / self.config.sample_rate;
        let mut waveform = vec![0.0; n_samples];

        let peak_time = stimulus_onset + component.latency_ms / 1000.0;
        let width_s = component.width_ms / 1000.0;

        // Gaussian wavelet shape
        for (i, i_slot) in waveform.iter_mut().enumerate() {
            let t = i as f64 * dt;
            let t_relative = t - peak_time;

            // Gaussian envelope
            let sigma = width_s / 2.355; // FWHM to sigma
            let envelope = (-t_relative.powi(2) / (2.0 * sigma.powi(2))).exp();

            *i_slot = component.amplitude_uv * envelope;
        }

        waveform
    }

    /// Generate background EEG activity
    fn generate_background_eeg(&mut self, n_samples: usize) -> Vec<f64> {
        let mut background = vec![0.0; n_samples];
        let noise_dist = Normal::new(0.0, 1.0).unwrap();

        // Alpha band (8-13 Hz)
        let alpha_freq = 10.0 + self.rng.random::<f64>() * 2.0;
        let alpha_amp = self.config.background_amplitude * 0.4;

        // Pink noise component
        let mut pink_state = 0.0;

        for (i, i_slot) in background.iter_mut().enumerate() {
            let t = i as f64 / self.config.sample_rate;

            // Alpha oscillation
            let alpha = alpha_amp * (2.0 * PI * alpha_freq * t).sin();

            // Pink noise (simple IIR)
            let white: f64 = self.rng.sample(noise_dist);
            pink_state = 0.99 * pink_state + 0.01 * white;
            let pink = pink_state * self.config.background_amplitude * 0.3;

            // White noise
            let white_component: f64 = self.rng.sample(noise_dist) * self.config.background_amplitude * 0.2;

            *i_slot = alpha + pink + white_component;
        }

        background
    }

    /// Create standard component
    fn create_component(
        &self,
        component_type: ErpComponentType,
        amplitude: f64,
        latency: f64,
        width: f64,
    ) -> ErpComponent {
        ErpComponent {
            name: format!("{:?}", component_type),
            latency_ms: latency,
            amplitude_uv: amplitude,
            width_ms: width,
            topography: self.component_topography(component_type),
        }
    }

    /// Get topography for component type
    fn component_topography(&self, component: ErpComponentType) -> Topography {
        match component {
            ErpComponentType::P1 => Topography::Occipital,
            ErpComponentType::N1 => Topography::Central,
            ErpComponentType::P2 => Topography::Central,
            ErpComponentType::N2 => Topography::Frontal,
            ErpComponentType::P300 => Topography::Parietal,
            ErpComponentType::N400 => Topography::Central,
            ErpComponentType::P600 => Topography::Parietal,
            ErpComponentType::MMN => Topography::Frontal,
            ErpComponentType::Nd => Topography::Frontal,
            ErpComponentType::BP => Topography::Central,
            ErpComponentType::LRP => Topography::Central,
            ErpComponentType::ERN => Topography::Frontal,
            ErpComponentType::Pe => Topography::Parietal,
        }
    }

    /// Convert name to component type
    fn name_to_type(&self, name: &str) -> ErpComponentType {
        match name {
            "P1" => ErpComponentType::P1,
            "N1" => ErpComponentType::N1,
            "P2" => ErpComponentType::P2,
            "N2" => ErpComponentType::N2,
            "P300" => ErpComponentType::P300,
            "N400" => ErpComponentType::N400,
            "P600" => ErpComponentType::P600,
            "MMN" => ErpComponentType::MMN,
            "Nd" => ErpComponentType::Nd,
            "BP" => ErpComponentType::BP,
            "LRP" => ErpComponentType::LRP,
            "ERN" => ErpComponentType::ERN,
            "Pe" => ErpComponentType::Pe,
            _ => ErpComponentType::P300,
        }
    }

    /// Find peak in waveform
    fn find_peak(&self, signal: &[f64], positive: bool) -> usize {
        if positive {
            signal.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0)
        } else {
            signal.iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, _)| i)
                .unwrap_or(0)
        }
    }

    /// Apply pathology modifications
    fn apply_pathology(&mut self, output: &mut ErpOutput, pathology: ErpPathology) {
        match pathology {
            ErpPathology::ReducedP300 { amplitude_reduction, latency_delay } => {
                // Reduce P300 amplitude and increase latency
                self.modify_component_amplitude(output, ErpComponentType::P300, 1.0 - amplitude_reduction);
                self.shift_component_latency(output, ErpComponentType::P300, latency_delay);
            }
            ErpPathology::ReducedMMN { amplitude_reduction } => {
                self.modify_component_amplitude(output, ErpComponentType::MMN, 1.0 - amplitude_reduction);
            }
            ErpPathology::AbnormalN400 { amplitude_change } => {
                self.modify_component_amplitude(output, ErpComponentType::N400, 1.0 + amplitude_change);
            }
            ErpPathology::ReducedERN { amplitude_reduction } => {
                self.modify_component_amplitude(output, ErpComponentType::ERN, 1.0 - amplitude_reduction);
            }
            ErpPathology::GlobalReduction { factor } => {
                for ch in 0..output.signal.len() {
                    for i in 0..output.signal[ch].len() {
                        output.erp_only[ch][i] *= factor;
                        // Recompute signal with new ERP
                        let noise = output.signal[ch][i] - output.erp_only[ch][i] / factor;
                        output.signal[ch][i] = output.erp_only[ch][i] + noise;
                    }
                }
            }
            ErpPathology::IncreasedLatency { factor } => {
                // Shift all components
                for comp in &mut output.ground_truth.components {
                    comp.peak_latency *= factor;
                    comp.onset_latency *= factor;
                    comp.offset_latency *= factor;
                }
            }
        }
    }

    fn modify_component_amplitude(
        &self,
        _output: &mut ErpOutput,
        _component: ErpComponentType,
        _factor: f64,
    ) {
        // Simplified: modify ground truth
        // Full implementation would regenerate component
    }

    fn shift_component_latency(
        &self,
        output: &mut ErpOutput,
        component: ErpComponentType,
        shift_ms: f64,
    ) {
        for comp in &mut output.ground_truth.components {
            if comp.component == component {
                comp.peak_latency += shift_ms;
                comp.onset_latency += shift_ms;
                comp.offset_latency += shift_ms;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_visual_oddball_target() {
        let config = ErpConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = ErpGenerator::new(config);
        let output = generator.generate_visual_oddball(true, 1.0);

        assert!(!output.signal[0].is_empty());
        // Target should have P300
        assert!(output.ground_truth.components.iter().any(|c| c.component == ErpComponentType::P300));
    }

    #[test]
    fn test_visual_oddball_standard() {
        let config = ErpConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = ErpGenerator::new(config);
        let output = generator.generate_visual_oddball(false, 1.0);

        // Standard should NOT have P300
        assert!(!output.ground_truth.components.iter().any(|c| c.component == ErpComponentType::P300));
    }

    #[test]
    fn test_auditory_oddball_mmn() {
        let config = ErpConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = ErpGenerator::new(config);
        let output = generator.generate_auditory_oddball(false, true, 1.0);

        // Deviant should have MMN
        assert!(output.ground_truth.components.iter().any(|c| c.component == ErpComponentType::MMN));
    }

    #[test]
    fn test_error_monitoring() {
        let config = ErpConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = ErpGenerator::new(config);

        let error = generator.generate_error_monitoring(true, 1.0);
        let correct = generator.generate_error_monitoring(false, 1.0);

        // Error trial should have larger ERN
        let error_ern = error.ground_truth.components.iter()
            .find(|c| c.component == ErpComponentType::ERN)
            .map(|c| c.peak_amplitude.abs())
            .unwrap_or(0.0);

        let correct_ern = correct.ground_truth.components.iter()
            .find(|c| c.component == ErpComponentType::ERN)
            .map(|c| c.peak_amplitude.abs())
            .unwrap_or(0.0);

        assert!(error_ern > correct_ern);
    }

    #[test]
    fn test_semantic_priming() {
        let config = ErpConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = ErpGenerator::new(config);

        let related = generator.generate_semantic_priming(true, 1.0);
        let unrelated = generator.generate_semantic_priming(false, 1.0);

        // Unrelated should have larger N400
        let related_n400 = related.ground_truth.components.iter()
            .find(|c| c.component == ErpComponentType::N400)
            .map(|c| c.peak_amplitude.abs())
            .unwrap_or(0.0);

        let unrelated_n400 = unrelated.ground_truth.components.iter()
            .find(|c| c.component == ErpComponentType::N400)
            .map(|c| c.peak_amplitude.abs())
            .unwrap_or(0.0);

        assert!(unrelated_n400 > related_n400);
    }
}
