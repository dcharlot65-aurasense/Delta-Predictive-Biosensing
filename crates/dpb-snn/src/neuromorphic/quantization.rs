//! Hardware-specific quantization for neuromorphic hardware
//!
//! Provides quantization methods for weights, thresholds, and time constants
//! to match hardware constraints of different neuromorphic platforms.

use serde::{Deserialize, Serialize};
use super::constraints::WeightBitDepth;

/// Weight quantizer for target hardware
#[derive(Debug, Clone)]
pub struct WeightQuantizer {
    /// Quantization scheme
    scheme: QuantizationScheme,
    /// Bit depth
    bit_depth: WeightBitDepth,
    /// Calibration data (min/max values)
    calibration: Option<CalibrationData>,
}

impl WeightQuantizer {
    /// Create a new weight quantizer
    pub fn new(scheme: QuantizationScheme, bit_depth: WeightBitDepth) -> Self {
        Self {
            scheme,
            bit_depth,
            calibration: None,
        }
    }

    /// Calibrate quantizer using training data
    pub fn calibrate(&mut self, weights: &[f32]) {
        let min = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        self.calibration = Some(CalibrationData {
            min,
            max,
            scale: (max - min) / self.bit_depth.scale_factor(),
            zero_point: 0,
        });
    }

    /// Quantize weights
    pub fn quantize(&self, weights: &[f32]) -> QuantizationResult {
        match self.scheme {
            QuantizationScheme::Symmetric => self.quantize_symmetric(weights),
            QuantizationScheme::Asymmetric => self.quantize_asymmetric(weights),
            QuantizationScheme::PerChannel => self.quantize_per_channel(weights),
            QuantizationScheme::PowerOfTwo => self.quantize_power_of_two(weights),
        }
    }

    /// Symmetric quantization (centered at zero)
    fn quantize_symmetric(&self, weights: &[f32]) -> QuantizationResult {
        let (min_int, max_int) = self.bit_depth.range();

        // Find max absolute value
        let max_abs = weights
            .iter()
            .map(|&w| w.abs())
            .fold(0.0f32, f32::max);

        let scale = if max_abs > 0.0 {
            max_abs / max_int as f32
        } else {
            1.0
        };

        let quantized: Vec<i32> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round() as i32;
                q.clamp(min_int, max_int)
            })
            .collect();

        let dequantized: Vec<f32> = quantized.iter().map(|&q| q as f32 * scale).collect();

        // Calculate quantization error
        let error = self.calculate_error(weights, &dequantized);

        QuantizationResult {
            quantized_weights: quantized,
            scale,
            zero_point: 0,
            error,
            bit_depth: self.bit_depth,
        }
    }

    /// Asymmetric quantization (uses full range)
    fn quantize_asymmetric(&self, weights: &[f32]) -> QuantizationResult {
        let (min_int, max_int) = self.bit_depth.range();
        let range = (max_int - min_int) as f32;

        let min_w = weights.iter().copied().fold(f32::INFINITY, f32::min);
        let max_w = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let scale = if max_w > min_w {
            (max_w - min_w) / range
        } else {
            1.0
        };

        let zero_point = min_int - (min_w / scale).round() as i32;

        let quantized: Vec<i32> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round() as i32 + zero_point;
                q.clamp(min_int, max_int)
            })
            .collect();

        let dequantized: Vec<f32> = quantized
            .iter()
            .map(|&q| (q - zero_point) as f32 * scale)
            .collect();

        let error = self.calculate_error(weights, &dequantized);

        QuantizationResult {
            quantized_weights: quantized,
            scale,
            zero_point,
            error,
            bit_depth: self.bit_depth,
        }
    }

    /// Per-channel quantization (not implemented in basic version)
    fn quantize_per_channel(&self, weights: &[f32]) -> QuantizationResult {
        // Fallback to symmetric for now
        self.quantize_symmetric(weights)
    }

    /// Power-of-two quantization (for efficient hardware shifts)
    fn quantize_power_of_two(&self, weights: &[f32]) -> QuantizationResult {
        let (min_int, max_int) = self.bit_depth.range();

        // Find max absolute value and round to nearest power of 2
        let max_abs = weights
            .iter()
            .map(|&w| w.abs())
            .fold(0.0f32, f32::max);

        // The SCALE is what must be a power of two -- that is the whole point of
        // this scheme, since dequantization then becomes a shift rather than a
        // multiply. Rounding `max_abs` to a power of two and *then* dividing by
        // `max_int` does not achieve that: `max_int` is 127 at 8 bits, so the
        // quotient was never a power of two and the scheme silently degenerated
        // into ordinary symmetric quantization with a worse scale.
        //
        // Pick the exponent on the scale itself. `ceil` guarantees
        // `scale >= max_abs / max_int`, so `max_abs / scale <= max_int` and no
        // weight clips; the cost is at most one extra factor of two of
        // resolution, which is inherent to a power-of-two scale.
        let scale = if max_abs > 0.0 {
            let ideal = max_abs / max_int as f32;
            2.0f32.powf(ideal.log2().ceil())
        } else {
            1.0
        };

        let quantized: Vec<i32> = weights
            .iter()
            .map(|&w| {
                let q = (w / scale).round() as i32;
                q.clamp(min_int, max_int)
            })
            .collect();

        let dequantized: Vec<f32> = quantized.iter().map(|&q| q as f32 * scale).collect();

        let error = self.calculate_error(weights, &dequantized);

        QuantizationResult {
            quantized_weights: quantized,
            scale,
            zero_point: 0,
            error,
            bit_depth: self.bit_depth,
        }
    }

    /// Calculate quantization error metrics
    fn calculate_error(&self, original: &[f32], quantized: &[f32]) -> QuantizationError {
        let n = original.len() as f32;

        // Mean squared error
        let mse = original
            .iter()
            .zip(quantized.iter())
            .map(|(o, q)| (o - q).powi(2))
            .sum::<f32>()
            / n;

        // Signal-to-quantization-noise ratio
        let signal_power = original.iter().map(|o| o.powi(2)).sum::<f32>() / n;
        let sqnr_db = if mse > 0.0 {
            10.0 * (signal_power / mse).log10()
        } else {
            f32::INFINITY
        };

        // Max absolute error
        let max_error = original
            .iter()
            .zip(quantized.iter())
            .map(|(o, q)| (o - q).abs())
            .fold(0.0f32, f32::max);

        QuantizationError {
            mse,
            rmse: mse.sqrt(),
            max_error,
            sqnr_db,
        }
    }

    /// Dequantize weights back to floating point
    pub fn dequantize(&self, quantized: &[i32], scale: f32, zero_point: i32) -> Vec<f32> {
        quantized
            .iter()
            .map(|&q| (q - zero_point) as f32 * scale)
            .collect()
    }
}

/// Threshold quantizer
pub struct ThresholdQuantizer {
    bit_depth: usize,
}

impl ThresholdQuantizer {
    pub fn new(bit_depth: usize) -> Self {
        Self { bit_depth }
    }

    /// Quantize threshold voltage
    pub fn quantize(&self, threshold: f32, v_range: (f32, f32)) -> i32 {
        let (v_min, v_max) = v_range;
        let levels = (1 << self.bit_depth) - 1;

        let normalized = (threshold - v_min) / (v_max - v_min);
        let quantized = (normalized * levels as f32).round() as i32;

        quantized.clamp(0, levels)
    }

    /// Dequantize threshold
    pub fn dequantize(&self, quantized: i32, v_range: (f32, f32)) -> f32 {
        let (v_min, v_max) = v_range;
        let levels = (1 << self.bit_depth) - 1;

        let normalized = quantized as f32 / levels as f32;
        v_min + normalized * (v_max - v_min)
    }
}

/// Time constant quantizer
pub struct TimeConstantQuantizer {
    min_tau: f32,
    max_tau: f32,
    num_levels: usize,
}

impl TimeConstantQuantizer {
    /// Create new time constant quantizer
    pub fn new(min_tau: f32, max_tau: f32, num_levels: usize) -> Self {
        Self {
            min_tau,
            max_tau,
            num_levels,
        }
    }

    /// Quantize time constant (logarithmic scale)
    pub fn quantize(&self, tau: f32) -> usize {
        let tau_clamped = tau.clamp(self.min_tau, self.max_tau);

        // Log scale
        let log_min = self.min_tau.ln();
        let log_max = self.max_tau.ln();
        let log_tau = tau_clamped.ln();

        let normalized = (log_tau - log_min) / (log_max - log_min);
        let level = (normalized * (self.num_levels - 1) as f32).round() as usize;

        level.min(self.num_levels - 1)
    }

    /// Dequantize time constant
    pub fn dequantize(&self, level: usize) -> f32 {
        let level = level.min(self.num_levels - 1);

        let log_min = self.min_tau.ln();
        let log_max = self.max_tau.ln();

        let normalized = level as f32 / (self.num_levels - 1) as f32;
        let log_tau = log_min + normalized * (log_max - log_min);

        log_tau.exp()
    }

    /// Get all available time constants
    pub fn available_values(&self) -> Vec<f32> {
        (0..self.num_levels).map(|i| self.dequantize(i)).collect()
    }
}

/// Quantization scheme
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantizationScheme {
    /// Symmetric quantization (range: [-max, max])
    Symmetric,
    /// Asymmetric quantization (range: [min, max])
    Asymmetric,
    /// Per-channel quantization
    PerChannel,
    /// Power-of-two scale factors
    PowerOfTwo,
}

/// Quantization result
#[derive(Debug, Clone)]
pub struct QuantizationResult {
    /// Quantized weights (integer representation)
    pub quantized_weights: Vec<i32>,
    /// Scale factor
    pub scale: f32,
    /// Zero point (for asymmetric quantization)
    pub zero_point: i32,
    /// Quantization error metrics
    pub error: QuantizationError,
    /// Bit depth used
    pub bit_depth: WeightBitDepth,
}

/// Quantization error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationError {
    /// Mean squared error
    pub mse: f32,
    /// Root mean squared error
    pub rmse: f32,
    /// Maximum absolute error
    pub max_error: f32,
    /// Signal-to-quantization-noise ratio (dB)
    pub sqnr_db: f32,
}

/// Calibration data for quantization
#[derive(Debug, Clone)]
struct CalibrationData {
    min: f32,
    max: f32,
    scale: f32,
    zero_point: i32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization() {
        let quantizer = WeightQuantizer::new(
            QuantizationScheme::Symmetric,
            WeightBitDepth::Bits8,
        );

        let weights = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = quantizer.quantize(&weights);

        assert_eq!(result.zero_point, 0);
        assert!(result.error.sqnr_db > 40.0); // Good quantization quality

        // Check symmetry
        assert_eq!(result.quantized_weights[0], -result.quantized_weights[4]);
        assert_eq!(result.quantized_weights[1], -result.quantized_weights[3]);
    }

    #[test]
    fn test_asymmetric_quantization() {
        let quantizer = WeightQuantizer::new(
            QuantizationScheme::Asymmetric,
            WeightBitDepth::Bits8,
        );

        let weights = vec![0.0, 0.25, 0.5, 0.75, 1.0]; // Positive only
        let result = quantizer.quantize(&weights);

        // Should use full range
        assert_ne!(result.zero_point, 0);

        // First weight (0.0) should map to min value
        let (min_int, _) = WeightBitDepth::Bits8.range();
        assert_eq!(result.quantized_weights[0], min_int);
    }

    #[test]
    fn test_power_of_two_quantization() {
        let quantizer = WeightQuantizer::new(
            QuantizationScheme::PowerOfTwo,
            WeightBitDepth::Bits8,
        );

        let weights = vec![-1.5, -0.75, 0.0, 0.75, 1.5];
        let result = quantizer.quantize(&weights);

        // Scale should be a power of 2
        let log2_scale = result.scale.log2();
        assert!(
            (log2_scale - log2_scale.round()).abs() < 1e-5,
            "scale {} is not a power of two (log2 = {log2_scale})",
            result.scale
        );

        // ...and nothing may clip, or the scale was chosen too small.
        let (min_int, max_int) = WeightBitDepth::Bits8.range();
        for (w, q) in weights.iter().zip(result.quantized_weights.iter()) {
            assert!(
                *q > min_int && *q < max_int,
                "weight {w} quantized to {q}, at the edge of [{min_int}, {max_int}]"
            );
        }
    }

    #[test]
    fn test_quantization_error() {
        let quantizer = WeightQuantizer::new(
            QuantizationScheme::Symmetric,
            WeightBitDepth::Bits4, // Lower precision
        );

        let weights = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = quantizer.quantize(&weights);

        // Lower bit depth should have higher error
        assert!(result.error.rmse > 0.0);
        assert!(result.error.sqnr_db < 40.0);
    }

    #[test]
    fn test_threshold_quantization() {
        let quantizer = ThresholdQuantizer::new(8);
        let v_range = (0.0, 2.0);

        let threshold = 1.0;
        let quantized = quantizer.quantize(threshold, v_range);
        let dequantized = quantizer.dequantize(quantized, v_range);

        assert!((dequantized - threshold).abs() < 0.01);
    }

    #[test]
    fn test_time_constant_quantization() {
        let quantizer = TimeConstantQuantizer::new(1.0, 100.0, 16);

        // Quantize and dequantize
        let tau = 20.0;
        let level = quantizer.quantize(tau);
        let dequantized = quantizer.dequantize(level);

        // Should be close (logarithmic scale)
        assert!((dequantized - tau).abs() / tau < 0.2);
    }

    #[test]
    fn test_time_constant_available_values() {
        let quantizer = TimeConstantQuantizer::new(1.0, 100.0, 8);
        let values = quantizer.available_values();

        assert_eq!(values.len(), 8);
        assert!((values[0] - 1.0).abs() < 0.01);
        assert!((values[7] - 100.0).abs() < 0.01);

        // Should be approximately logarithmic
        for i in 1..values.len() {
            let ratio = values[i] / values[i - 1];
            assert!(ratio > 1.0 && ratio < 3.0);
        }
    }

    #[test]
    fn test_dequantize() {
        let quantizer = WeightQuantizer::new(
            QuantizationScheme::Symmetric,
            WeightBitDepth::Bits8,
        );

        let weights = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let result = quantizer.quantize(&weights);

        let dequantized = quantizer.dequantize(
            &result.quantized_weights,
            result.scale,
            result.zero_point,
        );

        // Dequantized values should be close to originals
        for (orig, deq) in weights.iter().zip(dequantized.iter()) {
            assert!((orig - deq).abs() < 0.05);
        }
    }

    #[test]
    fn test_quantization_preserves_sign() {
        let quantizer = WeightQuantizer::new(
            QuantizationScheme::Symmetric,
            WeightBitDepth::Bits8,
        );

        let weights = vec![-2.0, -1.0, 1.0, 2.0];
        let result = quantizer.quantize(&weights);

        // Negative weights should stay negative
        assert!(result.quantized_weights[0] < 0);
        assert!(result.quantized_weights[1] < 0);
        // Positive weights should stay positive
        assert!(result.quantized_weights[2] > 0);
        assert!(result.quantized_weights[3] > 0);
    }
}
