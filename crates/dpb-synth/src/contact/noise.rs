//! Noise and artifact generators

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::{Rng, RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// White noise generator
pub struct WhiteNoiseGenerator;

#[derive(Debug, Clone)]
pub struct WhiteNoiseParams {
    pub duration: f64,      // seconds
    pub sampling_rate: f64, // Hz
    pub amplitude: f64,     // RMS amplitude
}

impl SyntheticGenerator for WhiteNoiseGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = WhiteNoiseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, params.amplitude).unwrap();

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|_| normal.sample(&mut rng))
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("amplitude".to_string(), params.amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        WhiteNoiseParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            amplitude: 1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.amplitude < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("amplitude must be non-negative".to_string()));
        }
        Ok(())
    }
}

/// Pink noise (1/f) generator using Voss-McCartney algorithm
pub struct PinkNoiseGenerator;

#[derive(Debug, Clone)]
pub struct PinkNoiseParams {
    pub duration: f64,      // seconds
    pub sampling_rate: f64, // Hz
    pub amplitude: f64,     // RMS amplitude
    pub num_octaves: usize, // typically 10-16
}

impl SyntheticGenerator for PinkNoiseGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PinkNoiseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        // Voss-McCartney algorithm
        let mut octaves = vec![0.0; params.num_octaves];
        let mut signal = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Update octaves at different rates
            for (j, octave) in octaves.iter_mut().enumerate() {
                if i % (1 << j) == 0 {
                    *octave = normal.sample(&mut rng);
                }
            }

            // Sum octaves
            let sample: f64 = octaves.iter().sum();
            signal.push(sample);
        }

        // Normalize to desired amplitude
        let rms = (signal.iter().map(|x| x * x).sum::<f64>() / n_samples as f64).sqrt();
        let scale = params.amplitude / rms;
        let signal = Array1::from_vec(signal.iter().map(|x| x * scale).collect());

        let mut gt_params = HashMap::new();
        gt_params.insert("amplitude".to_string(), params.amplitude);
        gt_params.insert("num_octaves".to_string(), params.num_octaves as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PinkNoiseParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            amplitude: 1.0,
            num_octaves: 12,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.amplitude < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("amplitude must be non-negative".to_string()));
        }
        if params.num_octaves == 0 {
            return Err(crate::GeneratorError::InvalidParameter("num_octaves must be positive".to_string()));
        }
        Ok(())
    }
}

/// Powerline interference generator (50/60 Hz)
pub struct PowerlineInterferenceGenerator;

#[derive(Debug, Clone)]
pub struct PowerlineParams {
    pub duration: f64,      // seconds
    pub sampling_rate: f64, // Hz
    pub frequency: f64,     // Hz (50 or 60)
    pub amplitude: f64,     // amplitude
    pub harmonics: Vec<f64>, // relative amplitudes of harmonics
}

impl SyntheticGenerator for PowerlineInterferenceGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = PowerlineParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Add phase jitter for realism
        let phase_jitter = Normal::new(0.0, 0.05).unwrap();
        let phase = phase_jitter.sample(&mut rng);

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 / params.sampling_rate;
                    let mut sample = params.amplitude * (2.0 * PI * params.frequency * t + phase).sin();

                    // Add harmonics
                    for (harm_idx, &harm_amp) in params.harmonics.iter().enumerate() {
                        let harm_freq = params.frequency * (harm_idx + 2) as f64;
                        sample += params.amplitude * harm_amp * (2.0 * PI * harm_freq * t + phase).sin();
                    }

                    sample
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("frequency".to_string(), params.frequency);
        gt_params.insert("amplitude".to_string(), params.amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        PowerlineParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            frequency: 60.0,
            amplitude: 0.1,
            harmonics: vec![0.3, 0.1], // 2nd and 3rd harmonics
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.frequency <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("frequency must be positive".to_string()));
        }
        Ok(())
    }
}

/// Motion artifact generator (low-frequency drift)
pub struct MotionArtifactGenerator;

#[derive(Debug, Clone)]
pub struct MotionArtifactParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub amplitude: f64,
    pub frequency_range: (f64, f64), // Hz
    pub num_components: usize,
}

impl SyntheticGenerator for MotionArtifactGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = MotionArtifactParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate random sinusoidal components
        let mut components = Vec::new();
        for _ in 0..params.num_components {
            let freq = rng.random_range(params.frequency_range.0..params.frequency_range.1);
            let phase = rng.random_range(0.0..2.0 * PI);
            let amp = rng.random_range(0.3..1.0);
            components.push((freq, phase, amp));
        }

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 / params.sampling_rate;
                    components.iter()
                        .map(|(freq, phase, amp)| {
                            params.amplitude * amp * (2.0 * PI * freq * t + phase).sin()
                        })
                        .sum()
                })
                .collect()
        );

        let ground_truth = TimeSeriesGroundTruth {
            parameters: HashMap::new(),
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        MotionArtifactParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            amplitude: 0.5,
            frequency_range: (0.1, 2.0),
            num_components: 3,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.frequency_range.0 >= params.frequency_range.1 {
            return Err(crate::GeneratorError::InvalidParameter("invalid frequency range".to_string()));
        }
        Ok(())
    }
}

/// Baseline wander generator
pub struct BaselineWanderGenerator;

#[derive(Debug, Clone)]
pub struct BaselineWanderParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub amplitude: f64,
    pub frequency: f64, // typical 0.15-0.3 Hz for respiration
}

impl SyntheticGenerator for BaselineWanderGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = BaselineWanderParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let phase = rng.random_range(0.0..2.0 * PI);

        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|i| {
                    let t = i as f64 / params.sampling_rate;
                    params.amplitude * (2.0 * PI * params.frequency * t + phase).sin()
                })
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("frequency".to_string(), params.frequency);
        gt_params.insert("amplitude".to_string(), params.amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        BaselineWanderParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            amplitude: 0.2,
            frequency: 0.25,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        Ok(())
    }
}

/// Quantization noise generator
pub struct QuantizationNoiseGenerator;

#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub bit_depth: u32, // ADC bit depth
}

impl SyntheticGenerator for QuantizationNoiseGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = QuantizationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Quantization step size
        let q_step = 2.0 / (2_f64.powi(params.bit_depth as i32));

        // Uniform distribution [-q/2, q/2]
        let signal = Array1::from_vec(
            (0..n_samples)
                .map(|_| rng.random_range(-q_step / 2.0..q_step / 2.0))
                .collect()
        );

        let mut gt_params = HashMap::new();
        gt_params.insert("bit_depth".to_string(), params.bit_depth as f64);
        gt_params.insert("q_step".to_string(), q_step);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        QuantizationParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            bit_depth: 12,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.bit_depth == 0 || params.bit_depth > 32 {
            return Err(crate::GeneratorError::InvalidParameter("bit_depth must be 1-32".to_string()));
        }
        Ok(())
    }
}

/// Electrode contact impedance noise generator
pub struct ElectrodeNoiseGenerator;

#[derive(Debug, Clone)]
pub struct ElectrodeNoiseParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub baseline_impedance: f64,      // kOhm
    pub impedance_variation: f64,     // kOhm (variation range)
    pub variation_frequency: f64,     // Hz (slow drift)
    pub spike_rate: f64,              // spikes per minute
    pub spike_amplitude: f64,         // relative to baseline
}

impl SyntheticGenerator for ElectrodeNoiseGenerator {
    type Output = Array1<f64>;
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ElectrodeNoiseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut signal = Vec::with_capacity(n_samples);
        let mut events = Vec::new();

        // Generate slow impedance drift
        let phase = rng.random_range(0.0..2.0 * PI);

        for i in 0..n_samples {
            let t = i as f64 * dt;

            // Slow variation
            let drift = params.impedance_variation *
                (2.0 * PI * params.variation_frequency * t + phase).sin();

            // Thermal noise
            let thermal_noise = Normal::new(0.0, params.baseline_impedance * 0.01).unwrap();
            let noise = thermal_noise.sample(&mut rng);

            signal.push(params.baseline_impedance + drift + noise);
        }

        // Add contact spikes
        let spike_exp = rand_distr::Exp::new(params.spike_rate / 60.0).unwrap();
        let mut t = spike_exp.sample(&mut rng);

        while t < params.duration {
            let spike_idx = (t * params.sampling_rate) as usize;
            if spike_idx < n_samples {
                let spike_amp = params.baseline_impedance * params.spike_amplitude;
                signal[spike_idx] += spike_amp;

                events.push(crate::traits::Event {
                    time: t,
                    event_type: "contact_spike".to_string(),
                    amplitude: Some(spike_amp),
                    attributes: HashMap::new(),
                });

                // Exponential decay
                for i in 1..50.min(n_samples - spike_idx) {
                    let decay = (-(i as f64) * 0.1).exp();
                    signal[spike_idx + i] += spike_amp * decay;
                }
            }

            t += spike_exp.sample(&mut rng);
        }

        let signal = Array1::from_vec(signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("baseline_impedance".to_string(), params.baseline_impedance);
        gt_params.insert("impedance_variation".to_string(), params.impedance_variation);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ElectrodeNoiseParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            baseline_impedance: 5.0,  // kOhm
            impedance_variation: 1.0,
            variation_frequency: 0.1,
            spike_rate: 2.0,          // 2 per minute
            spike_amplitude: 0.5,     // 50% spike
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        Ok(())
    }
}

/// Sampling jitter generator (timing uncertainty)
pub struct SamplingJitterGenerator;

#[derive(Debug, Clone)]
pub struct SamplingJitterParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub jitter_std: f64,  // standard deviation in samples
}

impl SyntheticGenerator for SamplingJitterGenerator {
    type Output = Vec<f64>; // Actual sample times
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = SamplingJitterParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let dt = 1.0 / params.sampling_rate;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let jitter_dist = Normal::new(0.0, params.jitter_std * dt).unwrap();

        let sample_times: Vec<f64> = (0..n_samples)
            .map(|i| {
                let ideal_time = i as f64 * dt;
                let jitter = jitter_dist.sample(&mut rng);
                (ideal_time + jitter).max(0.0)
            })
            .collect();

        let mut gt_params = HashMap::new();
        gt_params.insert("jitter_std".to_string(), params.jitter_std);
        gt_params.insert("jitter_std_seconds".to_string(), params.jitter_std * dt);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(sample_times, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        SamplingJitterParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            jitter_std: 0.1, // 0.1 samples = 100 microseconds at 1 kHz
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        Ok(())
    }
}

/// Signal clipping/saturation generator
pub struct ClippingGenerator;

#[derive(Debug, Clone)]
pub struct ClippingParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub positive_threshold: f64,
    pub negative_threshold: f64,
}

impl SyntheticGenerator for ClippingGenerator {
    type Output = Array1<f64>; // Clipping mask (0 = no clip, 1 = clipped)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = ClippingParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;

        // This generator is typically applied to existing signals
        // For standalone use, we'll return the threshold values
        let signal = Array1::from_elem(n_samples, 0.0);

        let mut gt_params = HashMap::new();
        gt_params.insert("positive_threshold".to_string(), params.positive_threshold);
        gt_params.insert("negative_threshold".to_string(), params.negative_threshold);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        ClippingParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            positive_threshold: 1.0,
            negative_threshold: -1.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.positive_threshold <= params.negative_threshold {
            return Err(crate::GeneratorError::InvalidParameter("positive_threshold must be > negative_threshold".to_string()));
        }
        Ok(())
    }
}

/// Missing data generator (dropouts, packet loss)
pub struct MissingDataGenerator;

#[derive(Debug, Clone)]
pub struct MissingDataParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub dropout_rate: f64,      // dropouts per minute
    pub burst_length_mean: f64, // mean dropout length in samples
    pub burst_length_std: f64,
}

impl SyntheticGenerator for MissingDataGenerator {
    type Output = Array1<bool>; // true = valid, false = missing
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = MissingDataParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let mut valid_mask = vec![true; n_samples];
        let mut events = Vec::new();

        // Generate dropout events
        let dropout_exp = rand_distr::Exp::new(params.dropout_rate / 60.0).unwrap();
        let burst_dist = Normal::new(params.burst_length_mean, params.burst_length_std).unwrap();

        let mut t = dropout_exp.sample(&mut rng);

        while t < params.duration {
            let dropout_idx = (t * params.sampling_rate) as usize;
            let burst_length = burst_dist.sample(&mut rng).max(1.0) as usize;

            if dropout_idx < n_samples {
                let end_idx = (dropout_idx + burst_length).min(n_samples);

                // Mark samples as missing
                for i in dropout_idx..end_idx {
                    valid_mask[i] = false;
                }

                events.push(crate::traits::Event {
                    time: t,
                    event_type: "dropout".to_string(),
                    amplitude: Some(burst_length as f64),
                    attributes: {
                        let mut attrs = HashMap::new();
                        attrs.insert("end_time".to_string(), (end_idx as f64) / params.sampling_rate);
                        attrs
                    },
                });
            }

            t += dropout_exp.sample(&mut rng);
        }

        let signal = Array1::from_vec(valid_mask);

        let num_missing = signal.iter().filter(|&&x| !x).count();
        let missing_fraction = num_missing as f64 / n_samples as f64;

        let mut gt_params = HashMap::new();
        gt_params.insert("dropout_rate".to_string(), params.dropout_rate);
        gt_params.insert("burst_length_mean".to_string(), params.burst_length_mean);
        gt_params.insert("missing_fraction".to_string(), missing_fraction);
        gt_params.insert("num_dropouts".to_string(), events.len() as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events,
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(signal, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        MissingDataParams {
            duration: 10.0,
            sampling_rate: 1000.0,
            dropout_rate: 3.0,       // 3 dropouts per minute
            burst_length_mean: 10.0, // 10 samples
            burst_length_std: 5.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.sampling_rate <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("sampling_rate must be positive".to_string()));
        }
        if params.burst_length_mean <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("burst_length_mean must be positive".to_string()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_white_noise_generation() {
        let generator = WhiteNoiseGenerator;
        let params = WhiteNoiseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_pink_noise_generation() {
        let generator = PinkNoiseGenerator;
        let params = PinkNoiseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_powerline_interference() {
        let generator = PowerlineInterferenceGenerator;
        let params = PowerlineInterferenceGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_electrode_noise() {
        let generator = ElectrodeNoiseGenerator;
        let params = ElectrodeNoiseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_sampling_jitter() {
        let generator = SamplingJitterGenerator;
        let params = SamplingJitterGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_clipping() {
        let generator = ClippingGenerator;
        let params = ClippingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_missing_data() {
        let generator = MissingDataGenerator;
        let mut params = MissingDataGenerator::default_params();
        params.dropout_rate = 10.0; // Increase rate to ensure at least one dropout in test
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
        // Check that we have at least some valid samples
        let num_valid = result.signal.iter().filter(|&&x| x).count();
        assert!(num_valid > 0);
    }
}
