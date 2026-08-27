//! Noise and degradation generators for audio signals

use crate::traits::{SyntheticGenerator, GeneratedData, TimeSeriesGroundTruth};
use ndarray::Array1;
use rand::{RngExt, SeedableRng};
use rand_distr::{Distribution, Normal};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Background noise generator
pub struct BackgroundNoiseGenerator;

#[derive(Debug, Clone)]
pub struct BackgroundNoiseParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub signal_amplitude: f64,
    pub snr_db: f64,           // Signal-to-noise ratio in dB
    pub noise_type: NoiseType,
}

#[derive(Debug, Clone)]
pub enum NoiseType {
    White,    // Equal power across all frequencies
    Pink,     // 1/f power spectrum
    Babble,   // Multi-talker background
    Brown,    // 1/f^2 power spectrum
}

impl SyntheticGenerator for BackgroundNoiseGenerator {
    type Output = Array1<f64>; // noise signal
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = BackgroundNoiseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Calculate noise amplitude from SNR
        // SNR = 20 * log10(signal_amplitude / noise_amplitude)
        let noise_amplitude = params.signal_amplitude / 10.0_f64.powf(params.snr_db / 20.0);

        let noise_signal = match params.noise_type {
            NoiseType::White => {
                let white_dist = Normal::new(0.0, noise_amplitude).unwrap();
                (0..n_samples)
                    .map(|_| white_dist.sample(&mut rng))
                    .collect::<Vec<f64>>()
            }
            NoiseType::Pink => {
                // Simple pink noise using weighted sum of octave bands
                let mut pink = vec![0.0; n_samples];
                let white_dist = Normal::new(0.0, 1.0).unwrap();

                for octave in 0..8 {
                    let weight = 1.0 / (octave + 1) as f64;
                    let step_size = 2_usize.pow(octave);

                    for i in (0..n_samples).step_by(step_size) {
                        let value = white_dist.sample(&mut rng) * weight;
                        for slot in pink[i..(i + step_size).min(n_samples)].iter_mut() {
                            *slot += value;
                        }
                    }
                }

                // Normalize to desired amplitude
                let max_val = pink.iter().cloned().fold(0.0, f64::max);
                pink.iter().map(|&v| v / max_val * noise_amplitude).collect()
            }
            NoiseType::Babble => {
                // Simulate multi-talker babble (3-5 modulated noise sources)
                let num_talkers = rng.random_range(3..=5);
                let mut babble = vec![0.0; n_samples];
                let white_dist = Normal::new(0.0, 1.0).unwrap();

                for _ in 0..num_talkers {
                    let f_mod = rng.random_range(2.0..6.0); // modulation frequency (Hz)
                    let phase = rng.random_range(0.0..2.0 * PI);

                    for (i, i_slot) in babble.iter_mut().enumerate() {
                        let t = i as f64 / params.sampling_rate;
                        let modulation = 0.5 + 0.5 * (2.0 * PI * f_mod * t + phase).sin();
                        *i_slot += white_dist.sample(&mut rng) * modulation;
                    }
                }

                // Normalize
                let max_val = babble.iter().cloned().fold(0.0, f64::max);
                babble.iter().map(|&v| v / max_val * noise_amplitude).collect()
            }
            NoiseType::Brown => {
                // Brownian noise (integrate white noise)
                let white_dist = Normal::new(0.0, 1.0).unwrap();
                let mut brown = Vec::with_capacity(n_samples);
                let mut accumulator = 0.0;

                for _ in 0..n_samples {
                    accumulator += white_dist.sample(&mut rng);
                    brown.push(accumulator);
                }

                // Normalize
                let max_val = brown.iter().cloned().fold(0.0, f64::max);
                brown.iter().map(|&v| v / max_val * noise_amplitude).collect()
            }
        };

        let noise_array = Array1::from_vec(noise_signal);

        let mut gt_params = HashMap::new();
        gt_params.insert("snr_db".to_string(), params.snr_db);
        gt_params.insert("noise_amplitude".to_string(), noise_amplitude);
        gt_params.insert("signal_amplitude".to_string(), params.signal_amplitude);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(noise_array, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        BackgroundNoiseParams {
            duration: 5.0,
            sampling_rate: 16000.0,
            signal_amplitude: 1.0,
            snr_db: 20.0,
            noise_type: NoiseType::White,
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

/// Room acoustics generator (reverberation)
pub struct RoomAcousticsGenerator;

#[derive(Debug, Clone)]
pub struct RoomAcousticsParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub rt60: f64,              // Reverberation time (seconds)
    pub direct_to_reverb_ratio: f64, // dB
    pub room_size: RoomSize,
}

#[derive(Debug, Clone)]
pub enum RoomSize {
    Small,   // RT60 ~ 0.2-0.4s
    Medium,  // RT60 ~ 0.4-0.8s
    Large,   // RT60 ~ 0.8-2.0s
}

impl SyntheticGenerator for RoomAcousticsGenerator {
    type Output = Array1<f64>; // impulse response
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = RoomAcousticsParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Generate impulse response
        let mut impulse_response = Vec::with_capacity(n_samples);

        // Direct sound (delta spike at t=0)
        let direct_amplitude = 10.0_f64.powf(params.direct_to_reverb_ratio / 20.0);

        // Early reflections (first 50ms)
        let early_reflection_time = 0.05;
        let num_early = (early_reflection_time * params.sampling_rate) as usize;

        // Calculate decay rate from RT60
        // RT60 is time for 60dB decay: A(t) = A0 * exp(-t / tau)
        // where tau = RT60 / 6.91
        let tau = params.rt60 / 6.91;

        let noise_dist = Normal::new(0.0, 1.0).unwrap();

        for i in 0..n_samples {
            let t = i as f64 / params.sampling_rate;

            let mut amplitude = 0.0;

            // Direct sound
            if i == 0 {
                amplitude += direct_amplitude;
            }

            // Early reflections (discrete reflections)
            if i < num_early && i > 0 {
                // Random sparse reflections
                if rng.random_bool(0.05) {
                    let reflection_amplitude = (-t / tau).exp() * rng.random_range(0.1..0.3);
                    amplitude += reflection_amplitude;
                }
            }

            // Late reverberation (diffuse tail)
            if i >= num_early {
                let reverb_amplitude = (-t / tau).exp() * noise_dist.sample(&mut rng) * 0.1;
                amplitude += reverb_amplitude;
            }

            impulse_response.push(amplitude);
        }

        let ir_array = Array1::from_vec(impulse_response);

        let mut gt_params = HashMap::new();
        gt_params.insert("rt60".to_string(), params.rt60);
        gt_params.insert("direct_to_reverb_ratio".to_string(), params.direct_to_reverb_ratio);
        gt_params.insert("decay_constant".to_string(), tau);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(ir_array, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        RoomAcousticsParams {
            duration: 2.0,
            sampling_rate: 16000.0,
            rt60: 0.5,
            direct_to_reverb_ratio: 10.0,
            room_size: RoomSize::Medium,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.rt60 <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("rt60 must be positive".to_string()));
        }
        Ok(())
    }
}

/// Microphone response generator (frequency-dependent distortion)
pub struct MicrophoneResponseGenerator;

#[derive(Debug, Clone)]
pub struct MicrophoneResponseParams {
    pub num_frequency_points: usize,
    pub frequency_range: (f64, f64), // Hz
    pub response_flatness: f64,      // 0-1 (1 = perfectly flat)
    pub distortion_level: f64,       // 0-1 (THD+N)
}

impl SyntheticGenerator for MicrophoneResponseGenerator {
    type Output = Vec<(f64, f64)>; // (frequency, magnitude) in dB
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = MicrophoneResponseParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let (f_min, f_max) = params.frequency_range;
        let log_f_min = f_min.ln();
        let log_f_max = f_max.ln();

        let mut frequency_response = Vec::new();

        // Generate response with deviations from flat
        let deviation_magnitude = 10.0 * (1.0 - params.response_flatness); // up to ±10 dB
        let noise_dist = Normal::new(0.0, deviation_magnitude / 3.0).unwrap();

        for i in 0..params.num_frequency_points {
            // Log-spaced frequencies
            let alpha = i as f64 / (params.num_frequency_points - 1) as f64;
            let log_f = log_f_min + alpha * (log_f_max - log_f_min);
            let frequency = log_f.exp();

            // Base response with roll-off at extremes
            let mut magnitude_db = 0.0;

            // Low-frequency roll-off
            if frequency < 100.0 {
                magnitude_db -= 6.0 * (100.0 / frequency).log2();
            }

            // High-frequency roll-off
            if frequency > 8000.0 {
                magnitude_db -= 3.0 * (frequency / 8000.0).log2();
            }

            // Add random deviations
            magnitude_db += noise_dist.sample(&mut rng);

            frequency_response.push((frequency, magnitude_db));
        }

        let flatness_deviation = frequency_response.iter()
            .map(|(_, mag)| mag.abs())
            .sum::<f64>() / params.num_frequency_points as f64;

        let mut gt_params = HashMap::new();
        gt_params.insert("response_flatness".to_string(), params.response_flatness);
        gt_params.insert("distortion_level".to_string(), params.distortion_level);
        gt_params.insert("mean_deviation_db".to_string(), flatness_deviation);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(frequency_response, ground_truth, 1.0))
    }

    fn default_params() -> Self::Parameters {
        MicrophoneResponseParams {
            num_frequency_points: 100,
            frequency_range: (20.0, 20000.0),
            response_flatness: 0.8,
            distortion_level: 0.01,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.num_frequency_points == 0 {
            return Err(crate::GeneratorError::InvalidParameter("num_frequency_points must be positive".to_string()));
        }
        if params.response_flatness < 0.0 || params.response_flatness > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("response_flatness must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Codec artifacts generator (compression distortion)
pub struct CodecArtifactsGenerator;

#[derive(Debug, Clone)]
pub struct CodecArtifactsParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub codec_type: CodecType,
    pub bitrate_kbps: f64,
}

// Domain acronyms -- ECG beat annotations, audio codecs, ERP components,
// the SMPL-X body model, drug classes. Camel case would diverge from how
// these are written everywhere they are used.
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug, Clone)]
pub enum CodecType {
    MP3,
    AAC,
    Opus,
    GSM,    // Mobile phone quality
}

impl SyntheticGenerator for CodecArtifactsGenerator {
    type Output = Array1<f64>; // artifact signal (to be added to clean signal)
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = CodecArtifactsParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Artifact characteristics vary by codec and bitrate
        let (quantization_noise_level, bandwidth_limit_hz) = match params.codec_type {
            CodecType::MP3 => {
                let noise = 0.01 * (128.0 / params.bitrate_kbps);
                let bw = (params.bitrate_kbps / 128.0 * 16000.0).min(20000.0);
                (noise, bw)
            }
            CodecType::AAC => {
                let noise = 0.008 * (128.0 / params.bitrate_kbps);
                let bw = (params.bitrate_kbps / 128.0 * 18000.0).min(20000.0);
                (noise, bw)
            }
            CodecType::Opus => {
                let noise = 0.005 * (64.0 / params.bitrate_kbps);
                let bw = (params.bitrate_kbps / 64.0 * 12000.0).min(20000.0);
                (noise, bw)
            }
            CodecType::GSM => {
                let noise = 0.03; // GSM has significant artifacts
                let bw = 3400.0;  // telephone bandwidth
                (noise, bw)
            }
        };

        let noise_dist = Normal::new(0.0, quantization_noise_level).unwrap();

        // Generate quantization noise
        let mut artifacts: Vec<f64> = (0..n_samples)
            .map(|_| noise_dist.sample(&mut rng))
            .collect();

        // Add bandwidth limitation artifacts (high-frequency noise reduction)
        let nyquist = params.sampling_rate / 2.0;
        if bandwidth_limit_hz < nyquist {
            // Simple high-frequency attenuation
            for (i, i_slot) in artifacts.iter_mut().enumerate() {
                let t = i as f64 / params.sampling_rate;
                // Add some aliasing artifacts near cutoff
                let alias_freq = bandwidth_limit_hz * 0.9;
                *i_slot += quantization_noise_level * 0.5 * (2.0 * PI * alias_freq * t).sin();
            }
        }

        let artifacts_array = Array1::from_vec(artifacts);

        let mut gt_params = HashMap::new();
        gt_params.insert("bitrate_kbps".to_string(), params.bitrate_kbps);
        gt_params.insert("quantization_noise_level".to_string(), quantization_noise_level);
        gt_params.insert("bandwidth_limit_hz".to_string(), bandwidth_limit_hz);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(artifacts_array, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        CodecArtifactsParams {
            duration: 5.0,
            sampling_rate: 16000.0,
            codec_type: CodecType::Opus,
            bitrate_kbps: 32.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.bitrate_kbps <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("bitrate_kbps must be positive".to_string()));
        }
        Ok(())
    }
}

/// Audio clipping generator (saturation/distortion)
pub struct AudioClippingGenerator;

#[derive(Debug, Clone)]
pub struct AudioClippingParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub clip_threshold: f64,    // 0-1 (amplitude level where clipping occurs)
    pub saturation_type: SaturationType,
}

#[derive(Debug, Clone)]
pub enum SaturationType {
    HardClip,   // Abrupt clipping
    SoftClip,   // Gradual saturation (tanh-like)
}

impl SyntheticGenerator for AudioClippingGenerator {
    type Output = Vec<f64>; // clipping function values
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = AudioClippingParams;

    fn generate(&self, params: &Self::Parameters, _seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let n_samples = (params.duration * params.sampling_rate) as usize;

        // Generate test amplitudes from -1 to 1
        let test_amplitudes: Vec<f64> = (0..n_samples)
            .map(|i| -1.0 + 2.0 * i as f64 / (n_samples - 1) as f64)
            .collect();

        let clipped_amplitudes: Vec<f64> = test_amplitudes.iter()
            .map(|&x| match params.saturation_type {
                SaturationType::HardClip => {
                    x.max(-params.clip_threshold).min(params.clip_threshold)
                }
                SaturationType::SoftClip => {
                    // tanh-based soft clipping
                    let scaled = x / params.clip_threshold;
                    params.clip_threshold * scaled.tanh()
                }
            })
            .collect();

        // Calculate total harmonic distortion
        let mut distortion_sum = 0.0;
        for i in 0..n_samples {
            distortion_sum += (clipped_amplitudes[i] - test_amplitudes[i]).powi(2);
        }
        let thd = (distortion_sum / n_samples as f64).sqrt();

        let mut gt_params = HashMap::new();
        gt_params.insert("clip_threshold".to_string(), params.clip_threshold);
        gt_params.insert("estimated_thd".to_string(), thd);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(clipped_amplitudes, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        AudioClippingParams {
            duration: 1.0,
            sampling_rate: 16000.0,
            clip_threshold: 0.8,
            saturation_type: SaturationType::SoftClip,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.clip_threshold <= 0.0 || params.clip_threshold > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("clip_threshold must be 0-1".to_string()));
        }
        Ok(())
    }
}

/// Telehealth degradation generator (network effects)
pub struct TelehealthDegradationGenerator;

#[derive(Debug, Clone)]
pub struct TelehealthDegradationParams {
    pub duration: f64,
    pub sampling_rate: f64,
    pub packet_loss_rate: f64,  // 0-1 (proportion of packets lost)
    pub jitter_ms: f64,          // milliseconds (timing variation)
    pub packet_size_ms: f64,     // milliseconds per packet
}

impl SyntheticGenerator for TelehealthDegradationGenerator {
    type Output = Vec<PacketEvent>; // packet delivery events
    type GroundTruth = TimeSeriesGroundTruth;
    type Parameters = TelehealthDegradationParams;

    fn generate(&self, params: &Self::Parameters, seed: u64) -> crate::Result<GeneratedData<Self::Output, Self::GroundTruth>> {
        Self::validate_params(params)?;

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let num_packets = (params.duration * 1000.0 / params.packet_size_ms) as usize;
        let jitter_dist = Normal::new(0.0, params.jitter_ms).unwrap();

        let mut packet_events = Vec::new();
        let mut lost_packets = 0;

        for i in 0..num_packets {
            let ideal_time = i as f64 * params.packet_size_ms / 1000.0;

            // Check if packet is lost
            if rng.random_bool(params.packet_loss_rate) {
                packet_events.push(PacketEvent {
                    packet_id: i,
                    ideal_time,
                    actual_time: None,
                    is_lost: true,
                });
                lost_packets += 1;
            } else {
                // Add jitter
                let jitter = jitter_dist.sample(&mut rng) / 1000.0; // convert to seconds
                let actual_time = (ideal_time + jitter).max(0.0);

                packet_events.push(PacketEvent {
                    packet_id: i,
                    ideal_time,
                    actual_time: Some(actual_time),
                    is_lost: false,
                });
            }
        }

        // Calculate actual packet loss rate
        let actual_loss_rate = lost_packets as f64 / num_packets as f64;

        // Calculate jitter statistics (for delivered packets)
        let jitter_values: Vec<f64> = packet_events.iter()
            .filter_map(|p| p.actual_time.map(|t| (t - p.ideal_time).abs() * 1000.0))
            .collect();

        let mean_jitter = if !jitter_values.is_empty() {
            jitter_values.iter().sum::<f64>() / jitter_values.len() as f64
        } else {
            0.0
        };

        let mut gt_params = HashMap::new();
        gt_params.insert("target_packet_loss_rate".to_string(), params.packet_loss_rate);
        gt_params.insert("actual_packet_loss_rate".to_string(), actual_loss_rate);
        gt_params.insert("target_jitter_ms".to_string(), params.jitter_ms);
        gt_params.insert("mean_jitter_ms".to_string(), mean_jitter);
        gt_params.insert("total_packets".to_string(), num_packets as f64);
        gt_params.insert("lost_packets".to_string(), lost_packets as f64);

        let ground_truth = TimeSeriesGroundTruth {
            parameters: gt_params,
            events: Vec::new(),
            segments: Vec::new(),
        };

        Ok(GeneratedData::new(packet_events, ground_truth, params.sampling_rate))
    }

    fn default_params() -> Self::Parameters {
        TelehealthDegradationParams {
            duration: 10.0,
            sampling_rate: 16000.0,
            packet_loss_rate: 0.02,  // 2% packet loss
            jitter_ms: 10.0,
            packet_size_ms: 20.0,
        }
    }

    fn validate_params(params: &Self::Parameters) -> crate::Result<()> {
        if params.duration <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("duration must be positive".to_string()));
        }
        if params.packet_loss_rate < 0.0 || params.packet_loss_rate > 1.0 {
            return Err(crate::GeneratorError::InvalidParameter("packet_loss_rate must be 0-1".to_string()));
        }
        if params.jitter_ms < 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("jitter_ms must be non-negative".to_string()));
        }
        if params.packet_size_ms <= 0.0 {
            return Err(crate::GeneratorError::InvalidParameter("packet_size_ms must be positive".to_string()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct PacketEvent {
    pub packet_id: usize,
    pub ideal_time: f64,      // seconds
    pub actual_time: Option<f64>, // None if lost
    pub is_lost: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_background_noise_white() {
        let generator = BackgroundNoiseGenerator;
        let params = BackgroundNoiseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_background_noise_pink() {
        let generator = BackgroundNoiseGenerator;
        let mut params = BackgroundNoiseGenerator::default_params();
        params.noise_type = NoiseType::Pink;
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_room_acoustics() {
        let generator = RoomAcousticsGenerator;
        let params = RoomAcousticsGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_microphone_response() {
        let generator = MicrophoneResponseGenerator;
        let params = MicrophoneResponseGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), params.num_frequency_points);
    }

    #[test]
    fn test_codec_artifacts() {
        let generator = CodecArtifactsGenerator;
        let params = CodecArtifactsGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_audio_clipping() {
        let generator = AudioClippingGenerator;
        let params = AudioClippingGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert_eq!(result.signal.len(), (params.duration * params.sampling_rate) as usize);
    }

    #[test]
    fn test_telehealth_degradation() {
        let generator = TelehealthDegradationGenerator;
        let params = TelehealthDegradationGenerator::default_params();
        let result = generator.generate(&params, 42).unwrap();
        assert!(!result.signal.is_empty());
    }
}
