//! Hilbert transform and analytic signal computation.

use num_complex::Complex64;
use rustfft::FftPlanner;

/// Computes the Hilbert transform of a real signal using FFT.
///
/// The Hilbert transform produces a 90-degree phase shift of all frequency components.
/// Returns the analytic signal as a complex vector where:
/// - Real part is the original signal
/// - Imaginary part is the Hilbert transform
///
/// # Arguments
/// * `signal` - Input real signal
///
/// # Returns
/// Complex analytic signal
pub fn hilbert_transform(signal: &[f64]) -> Vec<Complex64> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    // Forward FFT
    let mut buffer: Vec<Complex64> = signal.iter().map(|&x| Complex64::new(x, 0.0)).collect();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut buffer);

    // Apply Hilbert transform in frequency domain
    // Multiply positive frequencies by 2, set negative frequencies to 0
    // Keep DC and Nyquist (if n is even) unchanged
    buffer[0] *= 1.0; // DC component unchanged

    let half = n.div_ceil(2);
    for v in buffer[1..half].iter_mut() {
        *v *= 2.0;
    }

    if n.is_multiple_of(2) {
        buffer[n / 2] *= 1.0; // Nyquist frequency unchanged
    }

    buffer[half..n].fill(Complex64::new(0.0, 0.0));

    // Inverse FFT
    let ifft = planner.plan_fft_inverse(n);
    ifft.process(&mut buffer);

    // Normalize
    let norm = 1.0 / n as f64;
    buffer.iter().map(|c| c * norm).collect()
}

/// Analytic signal representation containing amplitude, phase, and frequency information.
#[derive(Debug, Clone)]
pub struct AnalyticSignal {
    /// Amplitude envelope of the signal
    pub amplitude_envelope: Vec<f64>,
    /// Instantaneous phase in radians
    pub instantaneous_phase: Vec<f64>,
    /// Instantaneous frequency in Hz
    pub instantaneous_frequency: Vec<f64>,
}

impl AnalyticSignal {
    /// Creates a new analytic signal.
    pub fn new(
        amplitude_envelope: Vec<f64>,
        instantaneous_phase: Vec<f64>,
        instantaneous_frequency: Vec<f64>,
    ) -> Self {
        Self {
            amplitude_envelope,
            instantaneous_phase,
            instantaneous_frequency,
        }
    }

    /// Returns the length of the signal.
    pub fn len(&self) -> usize {
        self.amplitude_envelope.len()
    }

    /// Returns true if the signal is empty.
    pub fn is_empty(&self) -> bool {
        self.amplitude_envelope.is_empty()
    }
}

/// Computes the analytic signal representation of a real signal.
///
/// # Arguments
/// * `signal` - Input real signal
/// * `sample_rate` - Sampling rate in Hz
///
/// # Returns
/// AnalyticSignal containing amplitude envelope, instantaneous phase, and frequency
pub fn analytic_signal(signal: &[f64], sample_rate: f64) -> AnalyticSignal {
    let n = signal.len();
    if n == 0 {
        return AnalyticSignal::new(Vec::new(), Vec::new(), Vec::new());
    }

    // Compute Hilbert transform
    let analytic = hilbert_transform(signal);

    // Extract amplitude envelope
    let amplitude_envelope: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();

    // Extract instantaneous phase
    let instantaneous_phase: Vec<f64> = analytic.iter().map(|c| c.arg()).collect();

    // Compute instantaneous frequency from phase derivative
    let mut instantaneous_frequency = Vec::with_capacity(n);

    if n > 0 {
        instantaneous_frequency.push(0.0);
    }

    for i in 1..n {
        // Unwrap phase to handle 2π discontinuities
        let mut phase_diff = instantaneous_phase[i] - instantaneous_phase[i - 1];

        // Unwrap phase
        if phase_diff > std::f64::consts::PI {
            phase_diff -= 2.0 * std::f64::consts::PI;
        } else if phase_diff < -std::f64::consts::PI {
            phase_diff += 2.0 * std::f64::consts::PI;
        }

        // Convert phase derivative to frequency (Hz)
        let freq = phase_diff * sample_rate / (2.0 * std::f64::consts::PI);
        instantaneous_frequency.push(freq);
    }

    AnalyticSignal::new(amplitude_envelope, instantaneous_phase, instantaneous_frequency)
}

/// Convenience function to extract just the amplitude envelope.
///
/// # Arguments
/// * `signal` - Input real signal
///
/// # Returns
/// Amplitude envelope as a vector
pub fn envelope(signal: &[f64]) -> Vec<f64> {
    let analytic = hilbert_transform(signal);
    analytic.iter().map(|c| c.norm()).collect()
}

/// Computes the instantaneous phase of a signal.
///
/// # Arguments
/// * `signal` - Input real signal
///
/// # Returns
/// Instantaneous phase in radians
pub fn instantaneous_phase(signal: &[f64]) -> Vec<f64> {
    let analytic = hilbert_transform(signal);
    analytic.iter().map(|c| c.arg()).collect()
}

/// Computes the instantaneous frequency of a signal.
///
/// # Arguments
/// * `signal` - Input real signal
/// * `sample_rate` - Sampling rate in Hz
///
/// # Returns
/// Instantaneous frequency in Hz
pub fn instantaneous_frequency(signal: &[f64], sample_rate: f64) -> Vec<f64> {
    analytic_signal(signal, sample_rate).instantaneous_frequency
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    #[test]
    fn test_hilbert_transform_sine() {
        // For a sine wave, Hilbert transform should produce a cosine wave
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();

        let analytic = hilbert_transform(&signal);

        // Check that real part matches original signal
        for (i, &original) in signal.iter().enumerate() {
            assert_relative_eq!(analytic[i].re, original, epsilon = 1e-10);
        }

        // Imaginary part should be approximately -cos (90 degree phase shift)
        #[allow(clippy::needless_range_loop)] // i also drives the expected value
        for i in 10..n - 10 {
            // Skip edges due to boundary effects
            let expected = -(2.0 * PI * 5.0 * i as f64 / n as f64).cos();
            assert_relative_eq!(analytic[i].im, expected, epsilon = 0.1);
        }
    }

    #[test]
    fn test_hilbert_transform_cosine() {
        // For a cosine wave, Hilbert transform should produce a sine wave
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).cos())
            .collect();

        let analytic = hilbert_transform(&signal);

        // Imaginary part should be approximately sin (90 degree phase shift)
        #[allow(clippy::needless_range_loop)] // i also drives the expected value
        for i in 10..n - 10 {
            let expected = (2.0 * PI * 5.0 * i as f64 / n as f64).sin();
            assert_relative_eq!(analytic[i].im, expected, epsilon = 0.1);
        }
    }

    #[test]
    fn test_envelope_amplitude_modulated() {
        // Create an amplitude-modulated signal
        let n = 256;
        let carrier_freq = 20.0;
        let modulation_freq = 2.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / n as f64;
                let modulation = 1.0 + 0.5 * (2.0 * PI * modulation_freq * t).cos();
                modulation * (2.0 * PI * carrier_freq * t).sin()
            })
            .collect();

        let env = envelope(&signal);

        // Envelope should follow the modulation
        assert_eq!(env.len(), n);

        // The envelope must track `1 + 0.5*cos(2*pi*f_m*t)` sample for sample.
        //
        // The previous assertions expected 1.0 at t = 0.25 and called it a
        // trough. With f_m = 2 Hz over a 1 s record, t = 0.25 is half a
        // modulation period in: cos(pi) = -1, so the modulation there is 0.5 --
        // the actual trough. The envelope was returning 0.5 correctly and being
        // marked wrong for it. The modulation passes through 1.0 at t = 0.125,
        // a quarter period in.
        for (i, e) in env.iter().enumerate() {
            let t = i as f64 / n as f64;
            let expected = 1.0 + 0.5 * (2.0 * PI * modulation_freq * t).cos();
            assert!(
                (e - expected).abs() < 0.05,
                "sample {i} (t={t:.4}): envelope {e} vs modulation {expected}"
            );
        }

        // Spot-check the three landmarks explicitly.
        assert!(env[0] > 1.45, "peak: {}", env[0]);              // t=0     -> 1.5
        assert!((env[n / 8] - 1.0).abs() < 0.05, "{}", env[n / 8]); // t=0.125 -> 1.0
        assert!(env[n / 4] < 0.55, "trough: {}", env[n / 4]);    // t=0.25  -> 0.5
    }

    #[test]
    fn test_analytic_signal_sine() {
        let n = 256;
        let freq = 10.0;
        let sample_rate = 256.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect();

        let analytic_sig = analytic_signal(&signal, sample_rate);

        assert_eq!(analytic_sig.len(), n);

        // For a pure sine wave, amplitude should be relatively constant (near 1.0)
        #[allow(clippy::needless_range_loop)] // i also drives the expected value
        for i in 10..n - 10 {
            assert!(analytic_sig.amplitude_envelope[i] > 0.8);
            assert!(analytic_sig.amplitude_envelope[i] < 1.2);
        }

        // Instantaneous frequency should be near the carrier frequency
        for i in 20..n - 20 {
            assert!(
                analytic_sig.instantaneous_frequency[i] > freq - 2.0
                    && analytic_sig.instantaneous_frequency[i] < freq + 2.0
            );
        }
    }

    #[test]
    fn test_instantaneous_phase() {
        let n = 128;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * 5.0 * i as f64 / n as f64).sin())
            .collect();

        let phase = instantaneous_phase(&signal);

        assert_eq!(phase.len(), n);

        // Phase should be monotonically increasing (with wrapping)
        // For a sine wave starting at 0, phase should start near -π/2
        assert!(phase[0] > -PI && phase[0] < 0.0);
    }

    #[test]
    fn test_instantaneous_frequency() {
        let n = 512;
        let freq = 10.0;
        let sample_rate = 512.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
            .collect();

        let inst_freq = instantaneous_frequency(&signal, sample_rate);

        assert_eq!(inst_freq.len(), n);

        // Average instantaneous frequency should be close to the true frequency
        let avg_freq: f64 = inst_freq[50..n - 50].iter().sum::<f64>() / (n - 100) as f64;
        assert_relative_eq!(avg_freq, freq, epsilon = 1.0);
    }

    #[test]
    fn test_hilbert_transform_empty() {
        let signal: Vec<f64> = Vec::new();
        let result = hilbert_transform(&signal);
        assert!(result.is_empty());
    }

    #[test]
    fn test_envelope_empty() {
        let signal: Vec<f64> = Vec::new();
        let result = envelope(&signal);
        assert!(result.is_empty());
    }

    #[test]
    fn test_analytic_signal_empty() {
        let signal: Vec<f64> = Vec::new();
        let result = analytic_signal(&signal, 100.0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_analytic_signal_chirp() {
        // Test with a frequency-modulated signal (chirp)
        let n = 512;
        let sample_rate = 512.0;
        let signal: Vec<f64> = (0..n)
            .map(|i| {
                let t = i as f64 / sample_rate;
                // Linear chirp from 5 Hz to 20 Hz
                let phase = 2.0 * PI * (5.0 * t + 7.5 * t * t);
                phase.sin()
            })
            .collect();

        let analytic_sig = analytic_signal(&signal, sample_rate);

        // Instantaneous frequency should increase over time
        let early_freq = analytic_sig.instantaneous_frequency[100..150]
            .iter()
            .sum::<f64>()
            / 50.0;
        let late_freq = analytic_sig.instantaneous_frequency[400..450]
            .iter()
            .sum::<f64>()
            / 50.0;

        assert!(late_freq > early_freq);
    }

}
