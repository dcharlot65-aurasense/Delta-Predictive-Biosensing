//! Sleep Microstructure Signal Generator
//!
//! Generates detailed sleep microstructure elements:
//! - Sleep spindles (sigma band, 11-16 Hz)
//! - K-complexes (high amplitude biphasic waves)
//! - Slow oscillations (< 1 Hz)
//! - Sharp wave ripples
//! - Vertex waves
//! - Pathological patterns

use rand::prelude::*;
use rand_distr::Normal;
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// Configuration for sleep microstructure generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepMicroConfig {
    /// Sampling rate (Hz)
    pub sample_rate: f64,
    /// Number of channels
    pub channels: usize,
    /// Background amplitude (µV)
    pub background_amplitude: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for SleepMicroConfig {
    fn default() -> Self {
        Self {
            sample_rate: 256.0,
            channels: 1,
            background_amplitude: 30.0,
            seed: None,
        }
    }
}

/// Output from sleep microstructure generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepMicroOutput {
    /// Time points (s)
    pub time: Vec<f64>,
    /// EEG signal per channel (µV)
    pub signal: Vec<Vec<f64>>,
    /// Ground truth
    pub ground_truth: SleepMicroGroundTruth,
    /// Configuration
    pub config: SleepMicroConfig,
}

/// Ground truth for sleep microstructure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepMicroGroundTruth {
    /// Sleep spindles detected
    pub spindles: Vec<SpindleInfo>,
    /// K-complexes detected
    pub k_complexes: Vec<KComplexInfo>,
    /// Slow oscillations
    pub slow_oscillations: Vec<SlowOscillationInfo>,
    /// Applied pathology
    pub pathology: Option<SleepMicroPathology>,
    /// Spindle density (per minute)
    pub spindle_density: f64,
    /// K-complex density (per minute)
    pub kcomplex_density: f64,
    /// Slow wave activity (µV²)
    pub slow_wave_activity: f64,
}

/// Sleep spindle information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpindleInfo {
    /// Onset time (s)
    pub onset: f64,
    /// Offset time (s)
    pub offset: f64,
    /// Duration (s)
    pub duration: f64,
    /// Peak frequency (Hz)
    pub frequency: f64,
    /// Peak amplitude (µV)
    pub amplitude: f64,
    /// Spindle type
    pub spindle_type: SpindleType,
    /// Associated slow oscillation phase
    pub so_phase: Option<f64>,
}

/// Types of sleep spindles
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SpindleType {
    /// Slow spindles (11-13 Hz, frontal)
    Slow,
    /// Fast spindles (13-16 Hz, centroparietal)
    Fast,
}

/// K-complex information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KComplexInfo {
    /// Onset time (s)
    pub onset: f64,
    /// Peak negative time (s)
    pub negative_peak_time: f64,
    /// Peak positive time (s)
    pub positive_peak_time: f64,
    /// Negative amplitude (µV)
    pub negative_amplitude: f64,
    /// Positive amplitude (µV)
    pub positive_amplitude: f64,
    /// Total duration (s)
    pub duration: f64,
    /// Evoked vs spontaneous
    pub evoked: bool,
    /// Associated spindle
    pub associated_spindle: bool,
}

/// Slow oscillation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlowOscillationInfo {
    /// Onset time (s)
    pub onset: f64,
    /// Duration (s)
    pub duration: f64,
    /// Down state onset (s)
    pub down_state_onset: f64,
    /// Up state onset (s)
    pub up_state_onset: f64,
    /// Peak-to-peak amplitude (µV)
    pub amplitude: f64,
    /// Frequency (Hz)
    pub frequency: f64,
}

/// Pathological sleep patterns
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SleepMicroPathology {
    /// Reduced spindle density (schizophrenia, aging)
    ReducedSpindles { reduction: f64 },
    /// Reduced slow wave activity (depression, insomnia)
    ReducedSlowWaves { reduction: f64 },
    /// Abnormal spindle frequency
    AbnormalSpindleFrequency { shift: f64 },
    /// Reduced spindle-slow wave coupling
    ReducedCoupling { reduction: f64 },
    /// Fragmented sleep (OSA, periodic limb movements)
    Fragmented { arousal_rate: f64 },
    /// REM without atonia
    RemWithoutAtonia,
}

/// Sleep microstructure generator
pub struct SleepMicrostructureGenerator {
    config: SleepMicroConfig,
    rng: StdRng,
}

impl SleepMicrostructureGenerator {
    /// Create new sleep microstructure generator
    pub fn new(config: SleepMicroConfig) -> Self {
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => rand::make_rng::<StdRng>(),
        };
        Self { config, rng }
    }

    /// Generate N2 sleep with spindles and K-complexes
    pub fn generate_n2(&mut self, duration: f64) -> SleepMicroOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut signal = vec![vec![0.0; n_samples]; self.config.channels];

        // Generate time vector
        for i in 0..n_samples {
            time.push(i as f64 * dt);
        }

        // Add background EEG (N2 characteristics)
        self.add_n2_background(&mut signal, n_samples);

        // Add sleep spindles
        let spindles = self.add_spindles(&mut signal, duration, 6.0); // ~6 per minute

        // Add K-complexes
        let k_complexes = self.add_k_complexes(&mut signal, duration, 1.0); // ~1 per minute

        let ground_truth = SleepMicroGroundTruth {
            spindles,
            k_complexes,
            slow_oscillations: vec![],
            pathology: None,
            spindle_density: 6.0,
            kcomplex_density: 1.0,
            slow_wave_activity: 0.0,
        };

        SleepMicroOutput {
            time,
            signal,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate N3 sleep with slow waves
    pub fn generate_n3(&mut self, duration: f64) -> SleepMicroOutput {
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut signal = vec![vec![0.0; n_samples]; self.config.channels];

        for i in 0..n_samples {
            time.push(i as f64 * dt);
        }

        // Add slow oscillations
        let slow_oscillations = self.add_slow_oscillations(&mut signal, duration);

        // Add spindles coupled to slow oscillations
        let spindles = self.add_coupled_spindles(&mut signal, &slow_oscillations);

        // Add background delta
        self.add_n3_background(&mut signal, n_samples);

        // Calculate slow wave activity
        let swa = self.calculate_slow_wave_activity(&signal[0]);

        let ground_truth = SleepMicroGroundTruth {
            spindles,
            k_complexes: vec![],
            slow_oscillations,
            pathology: None,
            spindle_density: 4.0,
            kcomplex_density: 0.0,
            slow_wave_activity: swa,
        };

        SleepMicroOutput {
            time,
            signal,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate isolated spindles for detection testing
    pub fn generate_spindle_sequence(&mut self, n_spindles: usize) -> SleepMicroOutput {
        let spindle_spacing = 3.0; // 3 seconds between spindles
        let duration = n_spindles as f64 * spindle_spacing + 2.0;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut signal = vec![vec![0.0; n_samples]; self.config.channels];

        for i in 0..n_samples {
            time.push(i as f64 * dt);
        }

        // Add background
        self.add_n2_background(&mut signal, n_samples);

        // Add spindles at regular intervals
        let mut spindles = Vec::new();
        for i in 0..n_spindles {
            let onset = 1.0 + i as f64 * spindle_spacing;
            let spindle_type = if self.rng.random::<bool>() {
                SpindleType::Fast
            } else {
                SpindleType::Slow
            };
            let info = self.add_single_spindle(&mut signal, onset, spindle_type);
            spindles.push(info);
        }

        let ground_truth = SleepMicroGroundTruth {
            spindles,
            k_complexes: vec![],
            slow_oscillations: vec![],
            pathology: None,
            spindle_density: n_spindles as f64 / (duration / 60.0),
            kcomplex_density: 0.0,
            slow_wave_activity: 0.0,
        };

        SleepMicroOutput {
            time,
            signal,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate isolated K-complexes
    pub fn generate_kcomplex_sequence(&mut self, n_kcomplexes: usize) -> SleepMicroOutput {
        let kc_spacing = 5.0;
        let duration = n_kcomplexes as f64 * kc_spacing + 2.0;
        let dt = 1.0 / self.config.sample_rate;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let mut time = Vec::with_capacity(n_samples);
        let mut signal = vec![vec![0.0; n_samples]; self.config.channels];

        for i in 0..n_samples {
            time.push(i as f64 * dt);
        }

        self.add_n2_background(&mut signal, n_samples);

        let mut k_complexes = Vec::new();
        for i in 0..n_kcomplexes {
            let onset = 1.0 + i as f64 * kc_spacing;
            let info = self.add_single_k_complex(&mut signal, onset, false);
            k_complexes.push(info);
        }

        let ground_truth = SleepMicroGroundTruth {
            spindles: vec![],
            k_complexes,
            slow_oscillations: vec![],
            pathology: None,
            spindle_density: 0.0,
            kcomplex_density: n_kcomplexes as f64 / (duration / 60.0),
            slow_wave_activity: 0.0,
        };

        SleepMicroOutput {
            time,
            signal,
            ground_truth,
            config: self.config.clone(),
        }
    }

    /// Generate pathological sleep
    pub fn generate_pathological(
        &mut self,
        stage: SleepStage,
        pathology: SleepMicroPathology,
        duration: f64,
    ) -> SleepMicroOutput {
        let mut output = match stage {
            SleepStage::N2 => self.generate_n2(duration),
            SleepStage::N3 => self.generate_n3(duration),
            _ => self.generate_n2(duration),
        };

        // Apply pathology
        match pathology {
            SleepMicroPathology::ReducedSpindles { reduction } => {
                // Remove some spindles
                let n_to_remove = (output.ground_truth.spindles.len() as f64 * reduction) as usize;
                for _ in 0..n_to_remove {
                    if !output.ground_truth.spindles.is_empty() {
                        let idx = self.rng.random_range(0..output.ground_truth.spindles.len());
                        output.ground_truth.spindles.remove(idx);
                    }
                }
                output.ground_truth.spindle_density *= 1.0 - reduction;
            }
            SleepMicroPathology::ReducedSlowWaves { reduction } => {
                // Reduce slow wave amplitude
                for ch in &mut output.signal {
                    for s in ch.iter_mut() {
                        *s *= 1.0 - reduction * 0.5;
                    }
                }
                output.ground_truth.slow_wave_activity *= 1.0 - reduction;
            }
            SleepMicroPathology::AbnormalSpindleFrequency { shift } => {
                for spindle in &mut output.ground_truth.spindles {
                    spindle.frequency += shift;
                }
            }
            _ => {}
        }

        output.ground_truth.pathology = Some(pathology);
        output
    }

    // Helper methods

    fn add_n2_background(&mut self, signal: &mut [Vec<f64>], _n_samples: usize) {
        let noise_dist = Normal::new(0.0, self.config.background_amplitude * 0.3).unwrap();

        for ch in signal.iter_mut() {
            for (i, i_slot) in ch.iter_mut().enumerate() {
                let t = i as f64 / self.config.sample_rate;

                // Theta activity (4-8 Hz)
                let theta = 10.0 * (2.0 * PI * 6.0 * t).sin();

                // Some alpha (8-12 Hz)
                let alpha = 5.0 * (2.0 * PI * 10.0 * t).sin();

                // Low delta
                let delta = 8.0 * (2.0 * PI * 2.0 * t).sin();

                let noise: f64 = self.rng.sample(noise_dist);
                *i_slot += theta + alpha + delta + noise;
            }
        }
    }

    fn add_n3_background(&mut self, signal: &mut [Vec<f64>], _n_samples: usize) {
        let noise_dist = Normal::new(0.0, self.config.background_amplitude * 0.2).unwrap();

        for ch in signal.iter_mut() {
            for (i, i_slot) in ch.iter_mut().enumerate() {
                let t = i as f64 / self.config.sample_rate;

                // Strong delta (0.5-4 Hz)
                let delta = 40.0 * (2.0 * PI * 1.5 * t).sin();

                // Low theta
                let theta = 5.0 * (2.0 * PI * 5.0 * t).sin();

                let noise: f64 = self.rng.sample(noise_dist);
                *i_slot += delta + theta + noise;
            }
        }
    }

    fn add_spindles(
        &mut self,
        signal: &mut [Vec<f64>],
        duration: f64,
        density: f64,
    ) -> Vec<SpindleInfo> {
        let n_spindles = (duration / 60.0 * density) as usize;
        let mut spindles = Vec::new();

        for _ in 0..n_spindles {
            let onset = self.rng.random::<f64>() * (duration - 2.0) + 0.5;
            let spindle_type = if self.rng.random::<bool>() {
                SpindleType::Fast
            } else {
                SpindleType::Slow
            };
            let info = self.add_single_spindle(signal, onset, spindle_type);
            spindles.push(info);
        }

        spindles.sort_by(|a, b| a.onset.total_cmp(&b.onset));
        spindles
    }

    fn add_single_spindle(
        &mut self,
        signal: &mut [Vec<f64>],
        onset: f64,
        spindle_type: SpindleType,
    ) -> SpindleInfo {
        let (freq, duration) = match spindle_type {
            SpindleType::Slow => (
                12.0 + self.rng.random::<f64>(),
                0.5 + self.rng.random::<f64>() * 0.5,
            ),
            SpindleType::Fast => (
                14.0 + self.rng.random::<f64>() * 2.0,
                0.5 + self.rng.random::<f64>() * 1.0,
            ),
        };

        let amplitude = 30.0 + self.rng.random::<f64>() * 30.0;
        let n_samples = (duration * self.config.sample_rate) as usize;
        let onset_sample = (onset * self.config.sample_rate) as usize;

        for ch in signal.iter_mut() {
            for i in 0..n_samples {
                let idx = onset_sample + i;
                if idx < ch.len() {
                    let t = i as f64 / self.config.sample_rate;
                    let phase = t / duration;

                    // Waxing and waning envelope
                    let envelope = (PI * phase).sin();

                    // Sigma oscillation
                    let oscillation = (2.0 * PI * freq * t).sin();

                    ch[idx] += amplitude * envelope * oscillation;
                }
            }
        }

        SpindleInfo {
            onset,
            offset: onset + duration,
            duration,
            frequency: freq,
            amplitude,
            spindle_type,
            so_phase: None,
        }
    }

    fn add_k_complexes(
        &mut self,
        signal: &mut [Vec<f64>],
        duration: f64,
        density: f64,
    ) -> Vec<KComplexInfo> {
        let n_kcomplexes = (duration / 60.0 * density) as usize;
        let mut kcomplexes = Vec::new();

        for _ in 0..n_kcomplexes {
            let onset = self.rng.random::<f64>() * (duration - 2.0) + 0.5;
            let info = self.add_single_k_complex(signal, onset, false);
            kcomplexes.push(info);
        }

        kcomplexes.sort_by(|a, b| a.onset.total_cmp(&b.onset));
        kcomplexes
    }

    fn add_single_k_complex(
        &mut self,
        signal: &mut [Vec<f64>],
        onset: f64,
        evoked: bool,
    ) -> KComplexInfo {
        let duration = 0.5 + self.rng.random::<f64>() * 0.3; // 0.5-0.8s
        let neg_amp = -(100.0 + self.rng.random::<f64>() * 100.0); // -100 to -200 µV
        let pos_amp = 50.0 + self.rng.random::<f64>() * 50.0; // 50-100 µV

        let onset_sample = (onset * self.config.sample_rate) as usize;
        let n_samples = (duration * self.config.sample_rate) as usize;

        let neg_peak_time = onset + duration * 0.2;
        let pos_peak_time = onset + duration * 0.6;

        for ch in signal.iter_mut() {
            for i in 0..n_samples {
                let idx = onset_sample + i;
                if idx < ch.len() {
                    let t = i as f64 / self.config.sample_rate;
                    let phase = t / duration;

                    // Biphasic wave: sharp negative followed by broader positive
                    let wave = if phase < 0.3 {
                        // Sharp negative peak
                        neg_amp * (PI * phase / 0.3).sin()
                    } else {
                        // Broad positive wave
                        pos_amp * (PI * (phase - 0.3) / 0.7).sin()
                    };

                    ch[idx] += wave;
                }
            }
        }

        // Maybe add associated spindle
        let associated_spindle = self.rng.random::<f64>() < 0.3;
        if associated_spindle {
            self.add_single_spindle(signal, onset + duration, SpindleType::Fast);
        }

        KComplexInfo {
            onset,
            negative_peak_time: neg_peak_time,
            positive_peak_time: pos_peak_time,
            negative_amplitude: neg_amp,
            positive_amplitude: pos_amp,
            duration,
            evoked,
            associated_spindle,
        }
    }

    fn add_slow_oscillations(
        &mut self,
        signal: &mut [Vec<f64>],
        duration: f64,
    ) -> Vec<SlowOscillationInfo> {
        let so_freq = 0.8; // ~0.8 Hz
        let so_period = 1.0 / so_freq;
        let n_oscillations = (duration * so_freq) as usize;

        let mut slow_oscillations = Vec::new();
        let amplitude = 100.0 + self.rng.random::<f64>() * 50.0;

        for i in 0..n_oscillations {
            let onset = i as f64 * so_period;
            let down_state = onset + so_period * 0.25;
            let up_state = onset + so_period * 0.75;

            for ch in signal.iter_mut() {
                let onset_sample = (onset * self.config.sample_rate) as usize;
                let period_samples = (so_period * self.config.sample_rate) as usize;

                for j in 0..period_samples {
                    let idx = onset_sample + j;
                    if idx < ch.len() {
                        let t = j as f64 / self.config.sample_rate;
                        let _phase = t / so_period;

                        // Slow oscillation waveform
                        let wave = amplitude * (2.0 * PI * so_freq * t).sin();
                        ch[idx] += wave;
                    }
                }
            }

            slow_oscillations.push(SlowOscillationInfo {
                onset,
                duration: so_period,
                down_state_onset: down_state,
                up_state_onset: up_state,
                amplitude: amplitude * 2.0, // Peak-to-peak
                frequency: so_freq,
            });
        }

        slow_oscillations
    }

    fn add_coupled_spindles(
        &mut self,
        signal: &mut [Vec<f64>],
        slow_oscillations: &[SlowOscillationInfo],
    ) -> Vec<SpindleInfo> {
        let mut spindles = Vec::new();

        // Add spindles at up-state of slow oscillations (phase-locked)
        for so in slow_oscillations.iter() {
            if self.rng.random::<f64>() < 0.4 {
                // 40% chance of spindle at each SO
                let spindle_onset = so.up_state_onset - 0.1; // Just before up-state
                let info = self.add_single_spindle(signal, spindle_onset, SpindleType::Fast);

                let mut spindle_with_phase = info;
                spindle_with_phase.so_phase = Some(0.75 * 2.0 * PI); // Up-state phase
                spindles.push(spindle_with_phase);
            }
        }

        spindles
    }

    fn calculate_slow_wave_activity(&self, signal: &[f64]) -> f64 {
        // Simplified SWA calculation (would use bandpass filter in production)
        let mut swa = 0.0;
        for window in signal.windows(256) {
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            let variance =
                window.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            swa += variance;
        }
        swa / (signal.len() / 256) as f64
    }
}

/// Sleep stage enum (for consistency with existing EEG module)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SleepStage {
    Wake,
    N1,
    N2,
    N3,
    Rem,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n2_generation() {
        let config = SleepMicroConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SleepMicrostructureGenerator::new(config);
        let output = generator.generate_n2(60.0); // 1 minute

        assert!(!output.signal[0].is_empty());
        assert!(!output.ground_truth.spindles.is_empty());
    }

    #[test]
    fn test_n3_generation() {
        let config = SleepMicroConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SleepMicrostructureGenerator::new(config);
        let output = generator.generate_n3(60.0);

        assert!(!output.ground_truth.slow_oscillations.is_empty());
        assert!(output.ground_truth.slow_wave_activity > 0.0);
    }

    #[test]
    fn test_spindle_sequence() {
        let config = SleepMicroConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SleepMicrostructureGenerator::new(config);
        let output = generator.generate_spindle_sequence(5);

        assert_eq!(output.ground_truth.spindles.len(), 5);
    }

    #[test]
    fn test_kcomplex_sequence() {
        let config = SleepMicroConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SleepMicrostructureGenerator::new(config);
        let output = generator.generate_kcomplex_sequence(3);

        assert_eq!(output.ground_truth.k_complexes.len(), 3);
    }

    #[test]
    fn test_spindle_types() {
        let config = SleepMicroConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SleepMicrostructureGenerator::new(config);
        let output = generator.generate_n2(120.0);

        let slow_count = output
            .ground_truth
            .spindles
            .iter()
            .filter(|s| s.spindle_type == SpindleType::Slow)
            .count();
        let fast_count = output
            .ground_truth
            .spindles
            .iter()
            .filter(|s| s.spindle_type == SpindleType::Fast)
            .count();

        // Should have both types
        assert!(slow_count > 0 || fast_count > 0);
    }

    #[test]
    fn test_pathological_reduced_spindles() {
        let config = SleepMicroConfig {
            seed: Some(42),
            ..Default::default()
        };
        let mut generator = SleepMicrostructureGenerator::new(config);

        let normal = generator.generate_n2(60.0);
        let pathological = generator.generate_pathological(
            SleepStage::N2,
            SleepMicroPathology::ReducedSpindles { reduction: 0.5 },
            60.0,
        );

        assert!(pathological.ground_truth.spindle_density < normal.ground_truth.spindle_density);
    }
}
