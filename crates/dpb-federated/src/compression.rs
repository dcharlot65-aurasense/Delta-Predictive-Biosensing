//! Gradient compression for communication efficiency.

use crate::{
    model::{CompressionInfo, ParameterDelta, Tensor},
    FederatedError, Result,
};
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Compression strategy for model updates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompressionStrategy {
    /// No compression.
    None,
    /// Top-K sparsification (keep K largest values).
    TopK,
    /// Random sparsification.
    RandomK,
    /// Quantization to fewer bits.
    Quantize,
    /// Sign-based compression (signSGD).
    SignSGD,
    /// Error feedback compression.
    ErrorFeedback,
}

/// Gradient compressor.
pub struct GradientCompressor {
    /// Compression strategy.
    strategy: CompressionStrategy,
    /// Compression ratio (0.0-1.0, lower = more compression).
    ratio: f32,
    /// Quantization bits (for Quantize strategy).
    num_bits: u8,
    /// Error residual for error feedback.
    error_residual: Option<Vec<f32>>,
}

impl GradientCompressor {
    /// Create a new compressor with the given strategy.
    pub fn new(strategy: CompressionStrategy, ratio: f32) -> Self {
        Self {
            strategy,
            ratio: ratio.clamp(0.01, 1.0),
            num_bits: 8,
            error_residual: None,
        }
    }

    /// Create with quantization.
    pub fn quantize(num_bits: u8) -> Self {
        Self {
            strategy: CompressionStrategy::Quantize,
            ratio: 1.0,
            num_bits: num_bits.clamp(1, 16),
            error_residual: None,
        }
    }

    /// Create TopK compressor.
    pub fn top_k(ratio: f32) -> Self {
        Self::new(CompressionStrategy::TopK, ratio)
    }

    /// Create SignSGD compressor.
    pub fn sign_sgd() -> Self {
        Self {
            strategy: CompressionStrategy::SignSGD,
            ratio: 1.0,
            num_bits: 1,
            error_residual: None,
        }
    }

    /// Compress a parameter delta.
    pub fn compress(&mut self, delta: &mut ParameterDelta) -> Result<()> {
        if self.strategy == CompressionStrategy::None {
            return Ok(());
        }

        let mut total_original = 0usize;
        let mut total_compressed = 0usize;

        for tensor in delta.changes.values_mut() {
            total_original += tensor.data.len() * 4; // f32 = 4 bytes

            match self.strategy {
                CompressionStrategy::None => {}
                CompressionStrategy::TopK => {
                    self.compress_top_k(tensor)?;
                }
                CompressionStrategy::RandomK => {
                    self.compress_random_k(tensor)?;
                }
                CompressionStrategy::Quantize => {
                    self.compress_quantize(tensor)?;
                }
                CompressionStrategy::SignSGD => {
                    self.compress_sign(tensor)?;
                }
                CompressionStrategy::ErrorFeedback => {
                    self.compress_error_feedback(tensor)?;
                }
            }

            // Estimate compressed size
            let non_zero = tensor.data.iter().filter(|v| **v != 0.0).count();
            total_compressed += non_zero * 4; // Sparse representation
        }

        delta.compressed = true;
        delta.compression_info = Some(CompressionInfo {
            original_size: total_original,
            compressed_size: total_compressed,
            algorithm: format!("{:?}", self.strategy),
            sparsity: Some(1.0 - (total_compressed as f32 / total_original as f32)),
        });

        Ok(())
    }

    /// Top-K sparsification: keep only the K largest magnitude values.
    fn compress_top_k(&self, tensor: &mut Tensor) -> Result<()> {
        let k = (tensor.data.len() as f32 * self.ratio).ceil() as usize;
        let k = k.max(1);

        // Find k-th largest magnitude
        let mut magnitudes: Vec<(usize, f32)> = tensor
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.abs()))
            .collect();
        // Sort by magnitude descending (NaN-safe comparison)
        magnitudes.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create mask
        let mut mask = vec![false; tensor.data.len()];
        for (i, _) in magnitudes.iter().take(k) {
            mask[*i] = true;
        }

        // Apply mask
        for (i, v) in tensor.data.iter_mut().enumerate() {
            if !mask[i] {
                *v = 0.0;
            }
        }

        Ok(())
    }

    /// Random sparsification: randomly select K values to keep.
    fn compress_random_k(&self, tensor: &mut Tensor) -> Result<()> {
        let mut rng = rand::thread_rng();

        for v in tensor.data.iter_mut() {
            if rng.gen::<f32>() > self.ratio {
                *v = 0.0;
            } else {
                // Scale up to maintain expected value
                *v /= self.ratio;
            }
        }

        Ok(())
    }

    /// Quantize to fewer bits.
    fn compress_quantize(&self, tensor: &mut Tensor) -> Result<()> {
        let levels = (1 << self.num_bits) as f32;

        // Find min/max for normalization
        let min = tensor.data.iter().cloned().fold(f32::MAX, f32::min);
        let max = tensor.data.iter().cloned().fold(f32::MIN, f32::max);
        let range = max - min;

        if range < 1e-10 {
            return Ok(());
        }

        // Quantize
        for v in tensor.data.iter_mut() {
            let normalized = (*v - min) / range;
            let quantized = (normalized * levels).round() / levels;
            *v = quantized * range + min;
        }

        Ok(())
    }

    /// Sign-based compression (SignSGD).
    fn compress_sign(&self, tensor: &mut Tensor) -> Result<()> {
        for v in tensor.data.iter_mut() {
            *v = v.signum();
        }
        Ok(())
    }

    /// Error feedback compression.
    fn compress_error_feedback(&mut self, tensor: &mut Tensor) -> Result<()> {
        // Initialize error residual if needed
        if self.error_residual.is_none() {
            self.error_residual = Some(vec![0.0; tensor.data.len()]);
        }

        let residual = self.error_residual.as_mut().unwrap();
        if residual.len() != tensor.data.len() {
            residual.resize(tensor.data.len(), 0.0);
        }

        // Add residual to gradient
        for (v, r) in tensor.data.iter_mut().zip(residual.iter()) {
            *v += r;
        }

        // Compress using TopK
        let k = (tensor.data.len() as f32 * self.ratio).ceil() as usize;
        let mut magnitudes: Vec<(usize, f32)> = tensor
            .data
            .iter()
            .enumerate()
            .map(|(i, v)| (i, v.abs()))
            .collect();
        // Sort by magnitude descending (NaN-safe comparison)
        magnitudes.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut mask = vec![false; tensor.data.len()];
        for (i, _) in magnitudes.iter().take(k) {
            mask[*i] = true;
        }

        // Update residual and apply mask
        for (i, (v, r)) in tensor.data.iter_mut().zip(residual.iter_mut()).enumerate() {
            if mask[i] {
                *r = 0.0;
            } else {
                *r = *v;
                *v = 0.0;
            }
        }

        Ok(())
    }

    /// Decompress (reconstruct from sparse representation).
    pub fn decompress(&self, delta: &mut ParameterDelta) -> Result<()> {
        // For most strategies, no explicit decompression needed
        // The compressed values can be used directly
        delta.compressed = false;
        delta.compression_info = None;
        Ok(())
    }

    /// Get current strategy.
    pub fn strategy(&self) -> CompressionStrategy {
        self.strategy
    }

    /// Get compression ratio.
    pub fn ratio(&self) -> f32 {
        self.ratio
    }
}

/// Sparse tensor representation for communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseTensor {
    /// Non-zero indices.
    pub indices: Vec<u32>,
    /// Non-zero values.
    pub values: Vec<f32>,
    /// Original shape.
    pub shape: Vec<usize>,
    /// Original number of elements.
    pub numel: usize,
}

impl SparseTensor {
    /// Create from dense tensor.
    pub fn from_dense(tensor: &Tensor) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, v) in tensor.data.iter().enumerate() {
            if *v != 0.0 {
                indices.push(i as u32);
                values.push(*v);
            }
        }

        Self {
            indices,
            values,
            shape: tensor.shape.clone(),
            numel: tensor.data.len(),
        }
    }

    /// Convert to dense tensor.
    pub fn to_dense(&self) -> Tensor {
        let mut data = vec![0.0; self.numel];
        for (idx, val) in self.indices.iter().zip(&self.values) {
            data[*idx as usize] = *val;
        }
        Tensor::new(data, self.shape.clone())
    }

    /// Sparsity ratio.
    pub fn sparsity(&self) -> f32 {
        1.0 - (self.values.len() as f32 / self.numel as f32)
    }

    /// Number of non-zero elements.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_top_k_compression() {
        let mut compressor = GradientCompressor::top_k(0.5);
        let mut tensor = Tensor::new(vec![1.0, 5.0, 2.0, 3.0], vec![4]);

        compressor.compress_top_k(&mut tensor).unwrap();

        // Should keep top 2: indices 1 (5.0) and 3 (3.0)
        let non_zero: Vec<f32> = tensor.data.iter().filter(|v| **v != 0.0).cloned().collect();
        assert_eq!(non_zero.len(), 2);
        assert!(tensor.data[1] != 0.0); // 5.0
        assert!(tensor.data[3] != 0.0); // 3.0
    }

    #[test]
    fn test_sign_compression() {
        let mut compressor = GradientCompressor::sign_sgd();
        let mut tensor = Tensor::new(vec![-2.5, 0.0, 3.0, -0.1], vec![4]);

        compressor.compress_sign(&mut tensor).unwrap();

        assert_eq!(tensor.data, vec![-1.0, 0.0, 1.0, -1.0]);
    }

    #[test]
    fn test_quantization() {
        let mut compressor = GradientCompressor::quantize(4); // 16 levels
        let mut tensor = Tensor::new(vec![0.0, 0.5, 1.0], vec![3]);

        compressor.compress_quantize(&mut tensor).unwrap();

        // Values should be quantized but still in [0, 1]
        for v in &tensor.data {
            assert!(*v >= 0.0 && *v <= 1.0);
        }
    }

    #[test]
    fn test_sparse_tensor() {
        let tensor = Tensor::new(vec![1.0, 0.0, 0.0, 2.0, 0.0], vec![5]);
        let sparse = SparseTensor::from_dense(&tensor);

        assert_eq!(sparse.nnz(), 2);
        assert_eq!(sparse.indices, vec![0, 3]);
        assert_eq!(sparse.values, vec![1.0, 2.0]);
        assert!((sparse.sparsity() - 0.6).abs() < 1e-6);

        let reconstructed = sparse.to_dense();
        assert_eq!(reconstructed.data, tensor.data);
    }

    #[test]
    fn test_compress_delta() {
        let mut compressor = GradientCompressor::top_k(0.5);
        let mut delta = ParameterDelta::new(
            0,
            [("w".to_string(), Tensor::new(vec![1.0, 5.0, 2.0, 3.0], vec![4]))]
                .into_iter()
                .collect(),
        );

        compressor.compress(&mut delta).unwrap();

        assert!(delta.compressed);
        assert!(delta.compression_info.is_some());
        let info = delta.compression_info.as_ref().unwrap();
        assert!(info.sparsity.unwrap() > 0.0);
    }

    #[test]
    fn test_error_feedback() {
        let mut compressor = GradientCompressor::new(CompressionStrategy::ErrorFeedback, 0.5);

        // First compression
        let mut tensor1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        compressor.compress_error_feedback(&mut tensor1).unwrap();

        let non_zero1: usize = tensor1.data.iter().filter(|v| **v != 0.0).count();
        assert_eq!(non_zero1, 2);

        // Error should be accumulated
        assert!(compressor.error_residual.is_some());
    }

    #[test]
    fn test_compression_strategies() {
        let strategies = [
            CompressionStrategy::None,
            CompressionStrategy::TopK,
            CompressionStrategy::RandomK,
            CompressionStrategy::Quantize,
            CompressionStrategy::SignSGD,
        ];

        for strategy in &strategies {
            let compressor = GradientCompressor::new(*strategy, 0.5);
            assert_eq!(compressor.strategy(), *strategy);
        }
    }
}
