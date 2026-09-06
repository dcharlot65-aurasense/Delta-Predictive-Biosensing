use rand::RngExt;
use rand_distr::{Distribution, Normal};
use std::f64::consts::PI;

/// Sleep stages for EEG generation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SleepStage {
    Wake,
    N1,
    N2,
    N3,
    Rem,
}

/// Seizure types for pathological EEG
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SeizureType {
    Absence,     // 3 Hz spike-wave
    TonicClonic, // Generalized
    Focal,       // Localized
}

/// Band power configuration for EEG generation
#[derive(Debug, Clone)]
pub struct BandPowerConfig {
    pub delta: f64, // 0.5-4 Hz - Relative power 0-1
    pub theta: f64, // 4-8 Hz
    pub alpha: f64, // 8-13 Hz
    pub beta: f64,  // 13-30 Hz
    pub gamma: f64, // 30-100 Hz
}

impl Default for BandPowerConfig {
    fn default() -> Self {
        // Normal relaxed wakefulness (eyes closed)
        Self {
            delta: 0.1,
            theta: 0.15,
            alpha: 0.45, // Dominant alpha
            beta: 0.2,
            gamma: 0.1,
        }
    }
}

/// EEG signal generator
#[derive(Debug, Clone)]
pub struct EegGenerator {
    pub sample_rate: f64,
    pub channels: usize,
    pub noise_level: f64,
}

impl Default for EegGenerator {
    fn default() -> Self {
        Self {
            sample_rate: 256.0,
            channels: 1,
            noise_level: 0.1,
        }
    }
}

impl EegGenerator {
    /// Create a new EEG generator with specified sample rate and channel count
    pub fn new(sample_rate: f64, channels: usize) -> Self {
        Self {
            sample_rate,
            channels,
            noise_level: 0.1,
        }
    }

    /// Generate resting-state EEG with specified band characteristics
    pub fn generate_resting_state(
        &self,
        duration_sec: f64,
        config: &BandPowerConfig,
    ) -> Vec<Vec<f64>> {
        (0..self.channels)
            .map(|_| self.generate_channel(duration_sec, config))
            .collect()
    }

    /// Generate single channel EEG
    pub fn generate_channel(&self, duration_sec: f64, config: &BandPowerConfig) -> Vec<f64> {
        let n_samples = (duration_sec * self.sample_rate) as usize;
        let mut signal = vec![0.0; n_samples];

        // Normalize band powers
        let total_power = config.delta + config.theta + config.alpha + config.beta + config.gamma;
        let norm_delta = config.delta / total_power;
        let norm_theta = config.theta / total_power;
        let norm_alpha = config.alpha / total_power;
        let norm_beta = config.beta / total_power;
        let norm_gamma = config.gamma / total_power;

        // One amplitude for every band, so `BandPowerConfig` alone decides the
        // relative power -- which is what its fields document themselves as.
        //
        // Each band used to carry its own constant (50, 30, 20, 10 microvolts,
        // and 40 for alpha), encoding EEG's 1/f amplitude falloff. That falloff
        // is already expressed in `BandPowerConfig::default`, so applying it
        // again here double-counted it and overrode the caller: a config asking
        // for dominant gamma still came out delta-dominant, because delta's
        // constant was five times gamma's.
        const BAND_AMPLITUDE: f64 = 40.0;

        // Generate band-limited components
        // Delta: 0.5-4 Hz
        if norm_delta > 0.01 {
            let delta = generate_band_noise(n_samples, self.sample_rate, 0.5, 4.0);
            for (s, d) in signal.iter_mut().zip(delta.iter()) {
                *s += d * norm_delta.sqrt() * BAND_AMPLITUDE; // Scale to microvolts
            }
        }

        // Theta: 4-8 Hz
        if norm_theta > 0.01 {
            let theta = generate_band_noise(n_samples, self.sample_rate, 4.0, 8.0);
            for (s, t) in signal.iter_mut().zip(theta.iter()) {
                *s += t * norm_theta.sqrt() * BAND_AMPLITUDE;
            }
        }

        // Alpha: 8-13 Hz (dominant oscillation in awake, relaxed state)
        if norm_alpha > 0.01 {
            let alpha_freq = 10.0; // Center at 10 Hz
            // A sinusoid of amplitude A carries A^2/2 of power where unit-RMS
            // noise at A carries A^2, so scale by sqrt(2) to put alpha on the
            // same power footing as the noise bands.
            let alpha = generate_oscillation(
                duration_sec,
                self.sample_rate,
                alpha_freq,
                norm_alpha.sqrt() * BAND_AMPLITUDE * std::f64::consts::SQRT_2,
            );
            for (s, a) in signal.iter_mut().zip(alpha.iter()) {
                *s += a;
            }
        }

        // Beta: 13-30 Hz
        if norm_beta > 0.01 {
            let beta = generate_band_noise(n_samples, self.sample_rate, 13.0, 30.0);
            for (s, b) in signal.iter_mut().zip(beta.iter()) {
                *s += b * norm_beta.sqrt() * BAND_AMPLITUDE;
            }
        }

        // Gamma: 30-100 Hz
        if norm_gamma > 0.01 {
            let gamma = generate_band_noise(n_samples, self.sample_rate, 30.0, 100.0);
            for (s, g) in signal.iter_mut().zip(gamma.iter()) {
                *s += g * norm_gamma.sqrt() * BAND_AMPLITUDE;
            }
        }

        // Add pink noise baseline
        let pink = generate_pink_noise(n_samples);
        for (s, p) in signal.iter_mut().zip(pink.iter()) {
            *s += p * self.noise_level * 5.0;
        }

        signal
    }

    /// Generate EEG for specific sleep stage
    pub fn generate_sleep_stage(&self, stage: SleepStage, duration_sec: f64) -> Vec<Vec<f64>> {
        let config = Self::sleep_stage_config(stage);
        let mut signal = self.generate_resting_state(duration_sec, &config);

        // Add stage-specific features
        match stage {
            SleepStage::N2 => {
                // Add sleep spindles (12-14 Hz bursts) and K-complexes
                self.add_sleep_spindles(&mut signal, duration_sec);
                self.add_k_complexes(&mut signal, duration_sec);
            }
            SleepStage::N3 => {
                // Enhance slow wave activity
                self.add_slow_waves(&mut signal, duration_sec);
            }
            SleepStage::Rem => {
                // Add sawtooth waves (2-6 Hz sharp waves)
                self.add_sawtooth_waves(&mut signal, duration_sec);
            }
            _ => {}
        }

        signal
    }

    /// Generate EEG with embedded artifacts
    pub fn generate_with_artifacts(&self, duration_sec: f64, artifact_rate: f64) -> Vec<Vec<f64>> {
        let mut signal = self.generate_resting_state(duration_sec, &BandPowerConfig::default());
        let n_samples = (duration_sec * self.sample_rate) as usize;
        let mut rng = rand::rng();

        // Add blink artifacts (rate per second)
        let n_blinks = (duration_sec * artifact_rate * 0.3) as usize;
        for _ in 0..n_blinks {
            let blink_time = rng.random_range(0..n_samples);
            for channel in &mut signal {
                self.add_blink_artifact(channel, blink_time);
            }
        }

        // Add muscle artifacts
        let n_muscle = (duration_sec * artifact_rate * 0.2) as usize;
        for _ in 0..n_muscle {
            let start = rng.random_range(0..n_samples.saturating_sub(100));
            let duration = rng.random_range(50..200);
            for channel in &mut signal {
                self.add_muscle_artifact(channel, start, duration);
            }
        }

        signal
    }

    /// Generate seizure-like activity
    pub fn generate_ictal_pattern(
        &self,
        seizure_type: SeizureType,
        duration_sec: f64,
    ) -> Vec<Vec<f64>> {
        let n_samples = (duration_sec * self.sample_rate) as usize;
        let mut signal = vec![vec![0.0; n_samples]; self.channels];

        match seizure_type {
            SeizureType::Absence => {
                // 3 Hz spike-wave pattern
                for channel in &mut signal {
                    let spike_wave = self.generate_spike_wave(duration_sec, 3.0);
                    *channel = spike_wave;
                }
            }
            SeizureType::TonicClonic => {
                // Generalized high amplitude rhythmic activity
                // Tonic phase: sustained high frequency
                let tonic_duration = duration_sec * 0.3;
                let clonic_duration = duration_sec * 0.7;

                for channel in &mut signal {
                    // Tonic phase
                    let tonic = generate_band_noise(
                        (tonic_duration * self.sample_rate) as usize,
                        self.sample_rate,
                        15.0,
                        25.0,
                    );

                    // Clonic phase: rhythmic spike-wave (2-4 Hz)
                    let clonic = self.generate_spike_wave(clonic_duration, 3.0);

                    // Combine
                    for (i, &t) in tonic.iter().enumerate() {
                        channel[i] = t * 200.0; // High amplitude
                    }
                    let offset = (tonic_duration * self.sample_rate) as usize;
                    for (i, &c) in clonic.iter().enumerate() {
                        if offset + i < channel.len() {
                            channel[offset + i] = c;
                        }
                    }
                }
            }
            SeizureType::Focal => {
                // Localized to first channel if multi-channel
                let focal_pattern = self.generate_spike_wave(duration_sec, 4.0);
                signal[0] = focal_pattern;

                // Other channels show normal activity with some spread
                for ch in 1..self.channels {
                    signal[ch] = self.generate_channel(duration_sec, &BandPowerConfig::default());
                    // Add attenuated focal activity
                    // Reads channel 0 while writing channel ch, so both
                    // borrows cannot be iterators over `signal`.
                    #[allow(clippy::needless_range_loop)]
                    for i in 0..n_samples {
                        signal[ch][i] += signal[0][i] * 0.3 / (ch as f64);
                    }
                }
            }
        }

        signal
    }

    /// Add eye blink artifact
    pub fn add_blink_artifact(&self, signal: &mut [f64], blink_time: usize) {
        let blink_duration = (0.2 * self.sample_rate) as usize; // 200ms blink
        let blink_amplitude = 100.0; // High amplitude frontal artifact

        for i in 0..blink_duration {
            let idx = blink_time + i;
            if idx < signal.len() {
                // Bell-shaped blink waveform
                let t = i as f64 / blink_duration as f64;
                let envelope = (-16.0 * (t - 0.5).powi(2)).exp();
                signal[idx] += blink_amplitude * envelope;
            }
        }
    }

    /// Add muscle artifact (EMG)
    pub fn add_muscle_artifact(&self, signal: &mut [f64], start: usize, duration: usize) {
        let mut rng = rand::rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..duration {
            let idx = start + i;
            if idx < signal.len() {
                // High frequency EMG noise (20-200 Hz dominant)
                let emg = normal.sample(&mut rng) * 30.0;
                signal[idx] += emg;
            }
        }
    }

    /// Get band power config for sleep stage
    pub fn sleep_stage_config(stage: SleepStage) -> BandPowerConfig {
        match stage {
            SleepStage::Wake => BandPowerConfig {
                delta: 0.1,
                theta: 0.15,
                alpha: 0.45, // High alpha
                beta: 0.2,
                gamma: 0.1,
            },
            SleepStage::N1 => BandPowerConfig {
                delta: 0.15,
                theta: 0.35, // Increased theta
                alpha: 0.25, // Reduced alpha
                beta: 0.15,
                gamma: 0.1,
            },
            SleepStage::N2 => BandPowerConfig {
                delta: 0.25,
                theta: 0.3,
                alpha: 0.15,
                beta: 0.2, // Spindles in beta range
                gamma: 0.1,
            },
            SleepStage::N3 => BandPowerConfig {
                delta: 0.6, // Dominant slow waves
                theta: 0.2,
                alpha: 0.05,
                beta: 0.1,
                gamma: 0.05,
            },
            SleepStage::Rem => BandPowerConfig {
                delta: 0.1,
                theta: 0.3, // Prominent theta
                alpha: 0.25,
                beta: 0.25,
                gamma: 0.1,
            },
        }
    }

    // Helper methods for stage-specific features

    fn add_sleep_spindles(&self, signal: &mut [Vec<f64>], duration_sec: f64) {
        let mut rng = rand::rng();
        let n_spindles = (duration_sec / 10.0) as usize; // ~6 per minute
        let n_samples = (duration_sec * self.sample_rate) as usize;

        for _ in 0..n_spindles {
            let start = rng.random_range(0..n_samples.saturating_sub(1000));
            let spindle_duration = 0.5 + rng.random_range(0.0..1.0); // 0.5-1.5 sec
            let spindle = generate_oscillation(
                spindle_duration,
                self.sample_rate,
                13.0, // 12-14 Hz
                50.0,
            );

            // Add to all channels with envelope
            for channel in signal.iter_mut() {
                for (i, &s) in spindle.iter().enumerate() {
                    let idx = start + i;
                    if idx < channel.len() {
                        let t = i as f64 / spindle.len() as f64;
                        let envelope = (PI * t).sin(); // Waxing and waning
                        channel[idx] += s * envelope;
                    }
                }
            }
        }
    }

    fn add_k_complexes(&self, signal: &mut [Vec<f64>], duration_sec: f64) {
        let mut rng = rand::rng();
        let n_complexes = (duration_sec / 60.0) as usize; // ~1 per minute
        let n_samples = (duration_sec * self.sample_rate) as usize;

        for _ in 0..n_complexes {
            let start = rng.random_range(0..n_samples.saturating_sub(500));
            let k_duration = (0.5 * self.sample_rate) as usize; // 500ms

            // K-complex: sharp negative wave followed by positive component
            for channel in signal.iter_mut() {
                for i in 0..k_duration {
                    let idx = start + i;
                    if idx < channel.len() {
                        let t = i as f64 / k_duration as f64;
                        // Biphasic wave
                        let wave = if t < 0.3 {
                            -150.0 * (PI * t / 0.3).sin()
                        } else {
                            75.0 * (PI * (t - 0.3) / 0.7).sin()
                        };
                        channel[idx] += wave;
                    }
                }
            }
        }
    }

    fn add_slow_waves(&self, signal: &mut [Vec<f64>], duration_sec: f64) {
        // Enhance delta band slow waves (high amplitude, 0.5-2 Hz)
        let _n_samples = (duration_sec * self.sample_rate) as usize;
        let slow_wave = generate_oscillation(duration_sec, self.sample_rate, 1.0, 100.0);

        for channel in signal.iter_mut() {
            for (i, &sw) in slow_wave.iter().enumerate() {
                if i < channel.len() {
                    channel[i] += sw;
                }
            }
        }
    }

    fn add_sawtooth_waves(&self, signal: &mut [Vec<f64>], duration_sec: f64) {
        let _rng = rand::rng();
        let n_samples = (duration_sec * self.sample_rate) as usize;

        // Sawtooth waves: 2-6 Hz sharp transients
        let wave_period = (self.sample_rate / 4.0) as usize; // ~4 Hz
        let mut phase = 0;

        for channel in signal.iter_mut() {
            for i in 0..n_samples {
                if phase == 0 && rand::random::<f64>() < 0.3 {
                    // Start sawtooth wave
                    for j in 0..wave_period {
                        if i + j < channel.len() {
                            let t = j as f64 / wave_period as f64;
                            channel[i + j] += 30.0 * (1.0 - t); // Sawtooth shape
                        }
                    }
                }
                phase = (phase + 1) % wave_period;
            }
        }
    }

    fn generate_spike_wave(&self, duration_sec: f64, frequency: f64) -> Vec<f64> {
        let n_samples = (duration_sec * self.sample_rate) as usize;
        let mut signal = vec![0.0; n_samples];
        let samples_per_cycle = (self.sample_rate / frequency) as usize;

        for (i, i_slot) in signal.iter_mut().enumerate() {
            let cycle_pos = i % samples_per_cycle;
            let t = cycle_pos as f64 / samples_per_cycle as f64;

            // Spike-wave: sharp spike followed by slow wave
            if t < 0.2 {
                // Sharp spike
                *i_slot = 200.0 * (PI * t / 0.2).sin();
            } else {
                // Slow wave
                *i_slot = -100.0 * (PI * (t - 0.2) / 0.8).sin();
            }
        }

        signal
    }
}

// Helper function to generate oscillation at specific frequency
fn generate_oscillation(
    duration_sec: f64,
    sample_rate: f64,
    frequency: f64,
    amplitude: f64,
) -> Vec<f64> {
    let n_samples = (duration_sec * sample_rate) as usize;
    let mut signal = vec![0.0; n_samples];
    let mut rng = rand::rng();

    // Add frequency jitter and amplitude modulation for realism
    let freq_jitter = Normal::new(0.0, frequency * 0.02).unwrap(); // 2% jitter
    let amp_mod_freq = 0.5; // Slow amplitude modulation

    let mut phase: f64 = 0.0;
    for (i, i_slot) in signal.iter_mut().enumerate() {
        let t = i as f64 / sample_rate;

        // Amplitude modulation
        let amp_envelope = 1.0 + 0.2 * (2.0 * PI * amp_mod_freq * t).sin();

        // Frequency with jitter
        let freq_inst = frequency + freq_jitter.sample(&mut rng);
        let phase_increment = 2.0 * PI * freq_inst / sample_rate;

        *i_slot = amplitude * amp_envelope * phase.sin();
        phase += phase_increment;

        // Keep phase in bounds
        if phase > 2.0 * PI {
            phase -= 2.0 * PI;
        }
    }

    signal
}

// Helper to generate pink noise (1/f)
fn generate_pink_noise(n_samples: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let mut signal = vec![0.0; n_samples];

    // Simple pink noise using running sum of white noise
    // (Voss-McCartney algorithm approximation)
    let mut generators = [0.0; 16];

    for (i, i_slot) in signal.iter_mut().enumerate() {
        // Update generators based on bit pattern
        let mut sum = 0.0;
        let mut bit = i;
        for generator in generators.iter_mut() {
            if bit & 1 == 1 {
                *generator = normal.sample(&mut rng);
            }
            sum += *generator;
            bit >>= 1;
        }

        *i_slot = sum / 16.0;
    }

    signal
}

// Helper to generate band-limited noise
/// Band-limited noise with unit RMS, over `[low_freq, high_freq)`.
///
/// A sum of sinusoids at frequencies drawn across the band with random phases,
/// which is band-limited by construction. The previous version multiplied the
/// *same* white-noise sample by five cosines and summed the products: that
/// modulates white noise rather than filtering it, so the result carried energy
/// across the whole spectrum and none of the bands were band-limited. The
/// consequence was that `BandPowerConfig` could not control anything -- the
/// widest band dominated the output whatever powers were requested.
///
/// Unit RMS so the caller's amplitude scaling means what it says.
fn generate_band_noise(
    n_samples: usize,
    sample_rate: f64,
    low_freq: f64,
    high_freq: f64,
) -> Vec<f64> {
    if n_samples == 0 || high_freq <= low_freq {
        return vec![0.0; n_samples];
    }

    // Nothing above Nyquist can be represented, and asking for it aliases back
    // into the band the caller was trying to avoid.
    let nyquist = sample_rate / 2.0;
    let high = high_freq.min(nyquist);
    let low = low_freq.min(high);
    if high <= low {
        return vec![0.0; n_samples];
    }

    let mut rng = rand::rng();
    // Enough components that the sum is noise-like rather than a chord.
    const COMPONENTS: usize = 48;

    let mut signal = vec![0.0; n_samples];
    for _ in 0..COMPONENTS {
        let freq = rng.random_range(low..high);
        let phase = rng.random_range(0.0..(2.0 * PI));
        let omega = 2.0 * PI * freq / sample_rate;
        for (i, s) in signal.iter_mut().enumerate() {
            *s += (omega * i as f64 + phase).sin();
        }
    }

    let mean_square = signal.iter().map(|v| v * v).sum::<f64>() / n_samples as f64;
    let rms = mean_square.sqrt();
    if rms > f64::EPSILON {
        for s in &mut signal {
            *s /= rms;
        }
    }
    signal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_eeg_generator_creation() {
        let eeg_gen = EegGenerator::default();
        assert_eq!(eeg_gen.sample_rate, 256.0);
        assert_eq!(eeg_gen.channels, 1);
    }

    #[test]
    fn test_generate_resting_state() {
        let eeg_gen = EegGenerator::new(256.0, 2);
        let config = BandPowerConfig::default();
        let signal = eeg_gen.generate_resting_state(1.0, &config);

        assert_eq!(signal.len(), 2);
        assert_eq!(signal[0].len(), 256);
    }

    #[test]
    fn test_sleep_stages() {
        let eeg_gen = EegGenerator::default();

        for stage in [
            SleepStage::Wake,
            SleepStage::N1,
            SleepStage::N2,
            SleepStage::N3,
            SleepStage::Rem,
        ] {
            let signal = eeg_gen.generate_sleep_stage(stage, 1.0);
            assert_eq!(signal.len(), 1);
            assert_eq!(signal[0].len(), 256);
        }
    }

    #[test]
    fn test_seizure_patterns() {
        let eeg_gen = EegGenerator::new(256.0, 2);

        for seizure_type in [
            SeizureType::Absence,
            SeizureType::TonicClonic,
            SeizureType::Focal,
        ] {
            let signal = eeg_gen.generate_ictal_pattern(seizure_type, 1.0);
            assert_eq!(signal.len(), 2);
            assert_eq!(signal[0].len(), 256);
        }
    }

    #[test]
    fn test_artifacts() {
        let eeg_gen = EegGenerator::default();
        let signal = eeg_gen.generate_with_artifacts(2.0, 1.0);
        assert_eq!(signal.len(), 1);
        assert_eq!(signal[0].len(), 512);
    }

    #[test]
    fn test_oscillation_generation() {
        let signal = generate_oscillation(1.0, 256.0, 10.0, 1.0);
        assert_eq!(signal.len(), 256);
        // Check approximate frequency content
        assert!(signal.iter().any(|&x| x.abs() > 0.5));
    }

    #[test]
    fn test_pink_noise() {
        let noise = generate_pink_noise(1000);
        assert_eq!(noise.len(), 1000);
        // Pink noise should have non-zero values
        assert!(noise.iter().any(|&x| x.abs() > 0.1));
    }

    /// Power in `[lo, hi)` as a fraction of the total, by DFT.
    fn band_fraction(signal: &[f64], sample_rate: f64, lo: f64, hi: f64) -> f64 {
        let n = signal.len();
        let mean = signal.iter().sum::<f64>() / n as f64;
        let mut total = 0.0;
        let mut in_band = 0.0;
        // Real signal: bins above Nyquist are mirrors, so only half are needed.
        for k in 1..n / 2 {
            let freq = k as f64 * sample_rate / n as f64;
            let (mut re, mut im) = (0.0, 0.0);
            for (i, v) in signal.iter().enumerate() {
                let angle = -2.0 * PI * k as f64 * i as f64 / n as f64;
                re += (v - mean) * angle.cos();
                im += (v - mean) * angle.sin();
            }
            let power = re * re + im * im;
            total += power;
            if freq >= lo && freq < hi {
                in_band += power;
            }
        }
        if total > 0.0 { in_band / total } else { 0.0 }
    }

    /// Band-limited noise must keep its energy inside its band.
    ///
    /// The previous implementation multiplied one white-noise sample by five
    /// cosines and summed the products, which modulates rather than filters:
    /// the result was broadband, so none of the bands were band-limited and
    /// `BandPowerConfig` could not control anything.
    #[test]
    fn band_noise_stays_within_its_band() {
        let sample_rate = 256.0;
        // 16 seconds. A finite window smears each component across neighbouring
        // bins, and the narrower the band the larger that leakage is relative
        // to the band itself: measured over 25 draws, 4-8 Hz holds as little as
        // 74% of its energy in a 4-second window but never less than 95% in a
        // 16-second one. The threshold below sits under that floor with margin,
        // because the frequencies are drawn at random and the fraction varies
        // from draw to draw.
        let n = 4096;
        for (lo, hi) in [(4.0, 8.0), (13.0, 30.0), (30.0, 60.0)] {
            let signal = generate_band_noise(n, sample_rate, lo, hi);
            let inside = band_fraction(&signal, sample_rate, lo, hi);
            assert!(
                inside > 0.9,
                "band {lo}-{hi} Hz holds only {:.0}% of its own energy",
                inside * 100.0
            );
        }
    }

    /// Requested band powers must decide which band dominates.
    ///
    /// Each band used to carry its own amplitude constant on top of the
    /// requested power -- 50 for delta against 10 for gamma -- so a config
    /// asking for dominant gamma still produced a delta-dominant signal.
    #[test]
    fn requested_band_dominates_the_output() {
        let generator = EegGenerator::new(256.0, 1);
        let cases: [(&str, BandPowerConfig, f64, f64); 3] = [
            (
                "delta",
                BandPowerConfig {
                    delta: 0.6,
                    theta: 0.1,
                    alpha: 0.0,
                    beta: 0.1,
                    gamma: 0.1,
                },
                0.5,
                4.0,
            ),
            (
                "beta",
                BandPowerConfig {
                    delta: 0.1,
                    theta: 0.1,
                    alpha: 0.0,
                    beta: 0.6,
                    gamma: 0.1,
                },
                13.0,
                30.0,
            ),
            (
                "gamma",
                BandPowerConfig {
                    delta: 0.1,
                    theta: 0.1,
                    alpha: 0.0,
                    beta: 0.1,
                    gamma: 0.6,
                },
                30.0,
                100.0,
            ),
        ];

        for (name, config, lo, hi) in cases {
            let signal = generator.generate_channel(4.0, &config);
            let fraction = band_fraction(&signal, 256.0, lo, hi);
            assert!(
                fraction > 0.4,
                "{name} was asked for 60% of the power and holds {:.0}%",
                fraction * 100.0
            );
        }
    }
}
