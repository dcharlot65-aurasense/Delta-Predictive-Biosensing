//! FFT operations using rustfft.

use crate::error::{DpbError, Result};
use ndarray::{Array1, Array2, ArrayView1};
use num_complex::Complex;
use rustfft::FftPlanner;

/// FFT processor for efficient frequency domain operations.
pub struct FftProcessor {
    forward_planner: FftPlanner<f64>,
    inverse_planner: FftPlanner<f64>,
}

impl FftProcessor {
    /// Creates a new FFT processor.
    pub fn new() -> Self {
        Self {
            forward_planner: FftPlanner::new(),
            inverse_planner: FftPlanner::new(),
        }
    }

    /// Computes the forward FFT.
    pub fn fft(&mut self, signal: ArrayView1<f64>) -> Result<Array1<Complex<f64>>> {
        let n = signal.len();
        if n == 0 {
            return Err(DpbError::InvalidDimensions(
                "Signal length must be positive".to_string(),
            ));
        }

        let mut buffer: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
        let fft = self.forward_planner.plan_fft_forward(n);
        fft.process(&mut buffer);

        Ok(Array1::from_vec(buffer))
    }

    /// Computes the inverse FFT.
    pub fn ifft(&mut self, spectrum: ArrayView1<Complex<f64>>) -> Result<Array1<f64>> {
        let n = spectrum.len();
        if n == 0 {
            return Err(DpbError::InvalidDimensions(
                "Spectrum length must be positive".to_string(),
            ));
        }

        let mut buffer: Vec<Complex<f64>> = spectrum.to_vec();
        let fft = self.inverse_planner.plan_fft_inverse(n);
        fft.process(&mut buffer);

        // Normalize
        let norm = 1.0 / n as f64;
        let result = buffer.iter().map(|c| c.re * norm).collect();

        Ok(Array1::from_vec(result))
    }

    /// Computes the power spectral density.
    pub fn psd(&mut self, signal: ArrayView1<f64>) -> Result<Array1<f64>> {
        let spectrum = self.fft(signal)?;
        let psd = spectrum.iter().map(|c| c.norm_sqr()).collect();
        Ok(Array1::from_vec(psd))
    }

    /// Computes the magnitude spectrum.
    pub fn magnitude_spectrum(&mut self, signal: ArrayView1<f64>) -> Result<Array1<f64>> {
        let spectrum = self.fft(signal)?;
        let magnitude = spectrum.iter().map(|c| c.norm()).collect();
        Ok(Array1::from_vec(magnitude))
    }

    /// Computes the phase spectrum.
    pub fn phase_spectrum(&mut self, signal: ArrayView1<f64>) -> Result<Array1<f64>> {
        let spectrum = self.fft(signal)?;
        let phase = spectrum.iter().map(|c| c.arg()).collect();
        Ok(Array1::from_vec(phase))
    }
}

impl Default for FftProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Computes the Short-Time Fourier Transform (STFT).
pub struct Stft {
    fft_size: usize,
    hop_size: usize,
    window: Array1<f64>,
    processor: FftProcessor,
}

impl Stft {
    /// Creates a new STFT processor.
    pub fn new(fft_size: usize, hop_size: usize, window_type: WindowType) -> Result<Self> {
        if fft_size == 0 || hop_size == 0 {
            return Err(DpbError::InvalidParameter(
                "FFT size and hop size must be positive".to_string(),
            ));
        }

        let window = create_window(fft_size, window_type)?;

        Ok(Self {
            fft_size,
            hop_size,
            window,
            processor: FftProcessor::new(),
        })
    }

    /// Computes the STFT of a signal.
    pub fn stft(&mut self, signal: ArrayView1<f64>) -> Result<Array2<Complex<f64>>> {
        let signal_len = signal.len();
        if signal_len < self.fft_size {
            return Err(DpbError::InvalidDimensions(
                "Signal length must be at least FFT size".to_string(),
            ));
        }

        let num_frames = (signal_len - self.fft_size) / self.hop_size + 1;
        let mut spectrogram = Array2::zeros((num_frames, self.fft_size / 2 + 1));

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_size;
            let end = start + self.fft_size;

            if end > signal_len {
                break;
            }

            let frame = signal.slice(ndarray::s![start..end]);
            let windowed: Array1<f64> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(s, w)| s * w)
                .collect();

            let spectrum = self.processor.fft(windowed.view())?;

            // Store only positive frequencies
            for (i, &value) in spectrum.iter().take(self.fft_size / 2 + 1).enumerate() {
                spectrogram[[frame_idx, i]] = value;
            }
        }

        Ok(spectrogram)
    }

    /// Computes the magnitude spectrogram.
    pub fn magnitude_spectrogram(&mut self, signal: ArrayView1<f64>) -> Result<Array2<f64>> {
        let stft = self.stft(signal)?;
        let magnitude = stft.mapv(|c| c.norm());
        Ok(magnitude)
    }

    /// Computes the power spectrogram.
    pub fn power_spectrogram(&mut self, signal: ArrayView1<f64>) -> Result<Array2<f64>> {
        let stft = self.stft(signal)?;
        let power = stft.mapv(|c| c.norm_sqr());
        Ok(power)
    }
}

/// Window function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// No taper. Best frequency resolution, worst spectral leakage.
    Rectangular,
    /// Raised cosine. The usual default -- good leakage suppression for a
    /// modest widening of the main lobe.
    Hann,
    /// Raised cosine with a non-zero pedestal, cancelling the first side lobe.
    Hamming,
    /// Three-term cosine window; lower side lobes than Hann at the cost of a
    /// wider main lobe.
    Blackman,
    /// Adjustable window trading main-lobe width against side-lobe level.
    Kaiser,
}

/// Creates a window function.
pub fn create_window(size: usize, window_type: WindowType) -> Result<Array1<f64>> {
    if size == 0 {
        return Err(DpbError::InvalidParameter(
            "Window size must be positive".to_string(),
        ));
    }

    let window: Vec<f64> = match window_type {
        WindowType::Rectangular => vec![1.0; size],
        WindowType::Hann => (0..size)
            .map(|i| {
                0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos())
            })
            .collect(),
        WindowType::Hamming => (0..size)
            .map(|i| {
                0.54 - 0.46 * (2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64).cos()
            })
            .collect(),
        WindowType::Blackman => (0..size)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / (size - 1) as f64;
                0.42 - 0.5 * angle.cos() + 0.08 * (2.0 * angle).cos()
            })
            .collect(),
        WindowType::Kaiser => {
            // Simplified Kaiser window (beta = 5)
            create_kaiser_window(size, 5.0)
        }
    };

    Ok(Array1::from_vec(window))
}

fn create_kaiser_window(size: usize, beta: f64) -> Vec<f64> {
    let i0_beta = bessel_i0(beta);
    (0..size)
        .map(|i| {
            let x = 2.0 * i as f64 / (size - 1) as f64 - 1.0;
            let arg = beta * (1.0 - x * x).sqrt();
            bessel_i0(arg) / i0_beta
        })
        .collect()
}

// Modified Bessel function of the first kind (order 0)
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0f64;
    let mut term = 1.0f64;
    let mut k = 1;

    while term.abs() > 1e-12 {
        term *= (x / 2.0) / k as f64;
        term *= (x / 2.0) / k as f64;
        sum += term;
        k += 1;
        if k > 100 {
            break; // Prevent infinite loop
        }
    }

    sum
}

/// Computes frequency bins for FFT output.
pub fn fft_frequencies(n: usize, sample_rate: f64) -> Array1<f64> {
    let freq_resolution = sample_rate / n as f64;
    Array1::from_shape_fn(n / 2 + 1, |i| i as f64 * freq_resolution)
}

/// Computes time bins for STFT output.
pub fn stft_times(
    signal_len: usize,
    fft_size: usize,
    hop_size: usize,
    sample_rate: f64,
) -> Array1<f64> {
    let num_frames = (signal_len - fft_size) / hop_size + 1;
    Array1::from_shape_fn(num_frames, |i| (i * hop_size) as f64 / sample_rate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_fft_basic() {
        let mut processor = FftProcessor::new();
        let signal = Array1::from_vec(vec![1.0, 0.0, -1.0, 0.0]);
        let spectrum = processor.fft(signal.view()).unwrap();

        assert_eq!(spectrum.len(), 4);
    }

    #[test]
    fn test_fft_inverse() {
        let mut processor = FftProcessor::new();
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let spectrum = processor.fft(signal.view()).unwrap();
        let recovered = processor.ifft(spectrum.view()).unwrap();

        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_hann_window() {
        let window = create_window(4, WindowType::Hann).unwrap();
        assert_eq!(window.len(), 4);
        assert_relative_eq!(window[0], 0.0, epsilon = 1e-10);
        assert_relative_eq!(window[3], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stft() {
        let mut stft = Stft::new(4, 2, WindowType::Hann).unwrap();
        let signal = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let spectrogram = stft.stft(signal.view()).unwrap();

        assert!(spectrogram.nrows() > 0);
        assert_eq!(spectrogram.ncols(), 3); // FFT size / 2 + 1
    }

    #[test]
    fn test_fft_frequencies() {
        let freqs = fft_frequencies(8, 100.0);
        assert_eq!(freqs.len(), 5);
        assert_relative_eq!(freqs[0], 0.0);
        assert_relative_eq!(freqs[1], 12.5);
    }
}
