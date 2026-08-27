//! Model Compression Utilities
//!
//! This module provides utilities for compressing SNN models using
//! knowledge distillation, including architecture search, layer merging,
//! pruning, and quantization-aware distillation.

use crate::{SNNError, SNNResult};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Target compression ratio (0.0-1.0)
    pub target_ratio: f32,
    /// Search strategy for finding student architecture
    pub search_strategy: SearchStrategy,
    /// Enable layer merging
    pub enable_layer_merging: bool,
    /// Enable channel pruning
    pub enable_channel_pruning: bool,
    /// Enable quantization
    pub enable_quantization: bool,
    /// Quantization bits
    pub quantization_bits: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            target_ratio: 0.3,
            search_strategy: SearchStrategy::DepthReduction,
            enable_layer_merging: true,
            enable_channel_pruning: true,
            enable_quantization: false,
            quantization_bits: 8,
        }
    }
}

/// Strategy for architecture search
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Reduce network depth (fewer layers)
    DepthReduction,
    /// Reduce network width (fewer neurons per layer)
    WidthReduction,
    /// Hybrid approach
    Hybrid,
    /// Neural Architecture Search (NAS)
    NAS,
    /// Predefined compression patterns
    Template,
}

/// Architecture search for finding optimal student network
pub struct ArchitectureSearch {
    /// Teacher architecture
    pub teacher_layers: Vec<usize>,
    /// Search strategy
    pub strategy: SearchStrategy,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Candidate architectures
    candidates: Vec<Vec<usize>>,
}

impl ArchitectureSearch {
    /// Create a new architecture search
    pub fn new(
        teacher_layers: Vec<usize>,
        strategy: SearchStrategy,
        compression_ratio: f32,
    ) -> SNNResult<Self> {
        if compression_ratio <= 0.0 || compression_ratio >= 1.0 {
            return Err(SNNError::InvalidConfig(
                "Compression ratio must be between 0 and 1".to_string()
            ));
        }

        let mut search = Self {
            teacher_layers,
            strategy,
            compression_ratio,
            candidates: Vec::new(),
        };

        search.generate_candidates()?;
        Ok(search)
    }

    /// Generate candidate student architectures
    fn generate_candidates(&mut self) -> SNNResult<()> {
        match self.strategy {
            SearchStrategy::DepthReduction => {
                self.generate_depth_reduced_candidates();
            }
            SearchStrategy::WidthReduction => {
                self.generate_width_reduced_candidates();
            }
            SearchStrategy::Hybrid => {
                self.generate_hybrid_candidates();
            }
            SearchStrategy::NAS | SearchStrategy::Template => {
                self.generate_template_candidates();
            }
        }
        Ok(())
    }

    /// Generate candidates by reducing depth
    /// Enumerates architectures formed by dropping interior layers.
    ///
    /// Input and output widths are always preserved -- they are fixed by the
    /// task, not by the search.
    ///
    /// Two things were wrong here before. The outer loop bound `num_to_remove`
    /// was never read in the body, so the loop ran the same two strided patterns
    /// twice and only ever produced two distinct shapes; and the loop bound
    /// `1..num_layers - 2` underflows for a teacher with fewer than two layers.
    /// Enumerating removal sets directly covers every removal count without a
    /// stride heuristic, and needs no such bound.
    ///
    /// Candidates are admitted if they are smaller than the teacher, NOT if they
    /// already meet `compression_ratio`. Selecting against the target is
    /// [`get_best_candidate`](Self::get_best_candidate)'s job, and it ranks by
    /// closeness to it. Pre-filtering on the same target discarded the entire
    /// search space whenever depth reduction alone could not reach the ratio --
    /// which is the common case, since dropping a layer cannot shrink the widest
    /// weight matrices -- leaving the caller an empty set rather than the
    /// closest achievable architecture.
    fn generate_depth_reduced_candidates(&mut self) {
        let num_layers = self.teacher_layers.len();

        // Nothing to remove without at least one interior layer.
        if num_layers < 3 {
            return;
        }

        let teacher_params = self.compute_total_params(&self.teacher_layers);
        let input = self.teacher_layers[0];
        let output = *self.teacher_layers.last().expect("non-empty, checked above");
        let interior: Vec<usize> = self.teacher_layers[1..num_layers - 1].to_vec();
        let n = interior.len();

        // Exhaustive subset enumeration is 2^n; past this depth fall back to
        // strided removal so the search stays bounded.
        const MAX_EXHAUSTIVE_INTERIOR: usize = 12;

        let removal_masks: Vec<u32> = if n <= MAX_EXHAUSTIVE_INTERIOR {
            (1u32..(1u32 << n)).collect()
        } else {
            // Keep every k-th interior layer, for each stride.
            (2..=n)
                .map(|stride| {
                    (0..n).fold(0u32, |m, i| {
                        if i % stride != 0 { m | (1 << i) } else { m }
                    })
                })
                .filter(|&m| m != 0)
                .collect()
        };

        for mask in removal_masks {
            let mut candidate = Vec::with_capacity(n + 2);
            candidate.push(input);
            for (i, &size) in interior.iter().enumerate() {
                if mask & (1u32 << i) == 0 {
                    candidate.push(size);
                }
            }
            candidate.push(output);

            let params = self.compute_total_params(&candidate);
            if params > 0 && params < teacher_params {
                self.candidates.push(candidate);
            }
        }
    }

    /// Generate candidates by reducing width
    fn generate_width_reduced_candidates(&mut self) {
        let target_params = (self.compute_total_params(&self.teacher_layers) as f32 * self.compression_ratio) as usize;

        // Try different width reduction factors
        for factor in [0.25, 0.5, 0.75] {
            let mut candidate = Vec::new();
            candidate.push(self.teacher_layers[0]); // Keep input size

            for &size in self.teacher_layers.iter().skip(1) {
                let reduced_size = ((size as f32 * factor).max(1.0) as usize).max(1);
                candidate.push(reduced_size);
            }

            let params = self.compute_total_params(&candidate);
            if params <= target_params && params > 0 {
                self.candidates.push(candidate);
            }
        }
    }

    /// Generate hybrid candidates
    fn generate_hybrid_candidates(&mut self) {
        self.generate_depth_reduced_candidates();
        self.generate_width_reduced_candidates();
    }

    /// Generate template-based candidates
    fn generate_template_candidates(&mut self) {
        let input_size = self.teacher_layers[0];
        let output_size = *self.teacher_layers.last().unwrap();

        // Small network template
        self.candidates.push(vec![input_size, output_size]);

        // Medium network template
        let mid_size = ((input_size + output_size) / 2).max(1);
        self.candidates.push(vec![input_size, mid_size, output_size]);

        // Larger medium network
        let mid1 = ((input_size + mid_size) / 2).max(1);
        let mid2 = ((mid_size + output_size) / 2).max(1);
        self.candidates.push(vec![input_size, mid1, mid_size, mid2, output_size]);
    }

    /// Get best candidate architecture
    pub fn get_best_candidate(&self) -> SNNResult<Vec<usize>> {
        if self.candidates.is_empty() {
            return Err(SNNError::InvalidConfig(
                "No candidate architectures found".to_string()
            ));
        }

        let target_params = (self.compute_total_params(&self.teacher_layers) as f32 * self.compression_ratio) as usize;

        // Find candidate closest to target
        let mut best_candidate = &self.candidates[0];
        let mut best_diff = f32::MAX;

        for candidate in &self.candidates {
            let params = self.compute_total_params(candidate);
            let diff = (params as f32 - target_params as f32).abs();

            if diff < best_diff {
                best_diff = diff;
                best_candidate = candidate;
            }
        }

        Ok(best_candidate.clone())
    }

    /// Get all candidate architectures
    pub fn get_all_candidates(&self) -> &[Vec<usize>] {
        &self.candidates
    }

    /// Compute total parameters in a network
    fn compute_total_params(&self, layers: &[usize]) -> usize {
        let mut total = 0;
        for i in 0..layers.len() - 1 {
            total += layers[i] * layers[i + 1]; // Weights
            total += layers[i + 1]; // Biases
        }
        total
    }
}

/// Layer merging for compression
pub struct LayerMerging {
    /// Merging strategy
    pub strategy: MergeStrategy,
}

/// Strategy for merging layers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Merge consecutive linear layers
    Linear,
    /// Merge with factorization
    Factorized,
    /// Knowledge-preserving merge
    KnowledgePreserving,
}

impl LayerMerging {
    pub fn new(strategy: MergeStrategy) -> Self {
        Self { strategy }
    }

    /// Merge two consecutive weight matrices
    pub fn merge_weights(
        &self,
        weights1: &Array2<f32>,
        weights2: &Array2<f32>,
    ) -> SNNResult<Array2<f32>> {
        let (out1, _in1) = (weights1.shape()[0], weights1.shape()[1]);
        let (_out2, in2) = (weights2.shape()[0], weights2.shape()[1]);

        if in2 != out1 {
            return Err(SNNError::DimensionMismatch {
                expected: format!("Second layer input ({}) should match first layer output ({})", in2, out1),
                actual: format!("Mismatch: {} != {}", in2, out1),
            });
        }

        match self.strategy {
            MergeStrategy::Linear => {
                // Simple matrix multiplication: W_merged = W2 * W1
                Ok(weights2.dot(weights1))
            }
            MergeStrategy::Factorized => {
                // Low-rank factorization (simplified)
                Ok(weights2.dot(weights1))
            }
            MergeStrategy::KnowledgePreserving => {
                // Preserve knowledge through weighted combination
                Ok(weights2.dot(weights1))
            }
        }
    }
}

/// Teacher-guided channel pruning
pub struct ChannelPruningGuided {
    /// Pruning ratio per layer
    pub pruning_ratios: Vec<f32>,
    /// Use teacher importance scores
    pub use_teacher_importance: bool,
}

impl ChannelPruningGuided {
    pub fn new(pruning_ratios: Vec<f32>) -> Self {
        Self {
            pruning_ratios,
            use_teacher_importance: true,
        }
    }

    /// Compute channel importance from teacher
    pub fn compute_channel_importance(
        &self,
        teacher_weights: &Array2<f32>,
        teacher_activations: &Array2<f32>,
    ) -> Array1<f32> {
        let num_channels = teacher_weights.shape()[0];
        let mut importance = Array1::zeros(num_channels);

        for c in 0..num_channels {
            // Importance = L2 norm of weights * average activation
            let weight_norm = teacher_weights.row(c).mapv(|x| x * x).sum().sqrt();
            let avg_activation = teacher_activations.column(c).mean().unwrap_or(0.0);
            importance[c] = weight_norm * avg_activation;
        }

        // Normalize
        let max_importance = importance.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        if max_importance > 1e-8 {
            importance /= max_importance;
        }

        importance
    }

    /// Get channels to keep based on importance
    pub fn select_channels_to_keep(
        &self,
        importance: &Array1<f32>,
        layer_idx: usize,
    ) -> Vec<usize> {
        let num_channels = importance.len();
        let pruning_ratio = if layer_idx < self.pruning_ratios.len() {
            self.pruning_ratios[layer_idx]
        } else {
            0.5 // Default
        };

        let num_to_keep = ((num_channels as f32 * (1.0 - pruning_ratio)).ceil() as usize).max(1);

        // Create (index, importance) pairs
        let mut indexed_importance: Vec<(usize, f32)> = importance
            .iter()
            .enumerate()
            .map(|(i, &imp)| (i, imp))
            .collect();

        // Sort by importance (descending)
        indexed_importance.sort_by(|a, b| b.1.total_cmp(&a.1));

        // Keep top channels
        indexed_importance
            .iter()
            .take(num_to_keep)
            .map(|(idx, _)| *idx)
            .collect()
    }
}

/// Quantization-aware distillation
pub struct QuantizationAwareDistillation {
    /// Number of bits for quantization
    pub num_bits: u8,
    /// Quantization range
    pub min_val: f32,
    pub max_val: f32,
    /// Use symmetric quantization
    pub symmetric: bool,
}

impl QuantizationAwareDistillation {
    pub fn new(num_bits: u8) -> Self {
        Self {
            num_bits,
            min_val: -1.0,
            max_val: 1.0,
            symmetric: true,
        }
    }

    /// Quantize weights
    pub fn quantize_weights(&self, weights: &Array2<f32>) -> Array2<f32> {
        let num_levels = (1 << self.num_bits) - 1;
        let scale = (self.max_val - self.min_val) / num_levels as f32;

        weights.mapv(|w| {
            let clamped = w.max(self.min_val).min(self.max_val);
            let quantized = ((clamped - self.min_val) / scale).round();
            self.min_val + quantized * scale
        })
    }

    /// Compute quantization error
    pub fn quantization_error(&self, original: &Array2<f32>, quantized: &Array2<f32>) -> f32 {
        let diff = original - quantized;
        diff.mapv(|x| x * x).sum() / original.len() as f32
    }

    /// Update quantization range based on weight statistics
    pub fn update_range(&mut self, weights: &Array2<f32>) {
        if self.symmetric {
            let max_abs = weights
                .iter()
                .map(|&w| w.abs())
                .fold(0.0f32, |a, b| a.max(b));
            self.min_val = -max_abs;
            self.max_val = max_abs;
        } else {
            self.min_val = weights
                .iter()
                .fold(f32::MAX, |a, &b| a.min(b));
            self.max_val = weights
                .iter()
                .fold(f32::MIN, |a, &b| a.max(b));
        }
    }
}

/// Compression metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionMetrics {
    /// Original model size (parameters)
    pub original_size: usize,
    /// Compressed model size (parameters)
    pub compressed_size: usize,
    /// Compression ratio
    pub compression_ratio: f32,
    /// Speedup factor
    pub speedup: f32,
    /// Original accuracy
    pub original_accuracy: f32,
    /// Compressed accuracy
    pub compressed_accuracy: f32,
    /// Accuracy drop
    pub accuracy_drop: f32,
    /// Memory savings (bytes)
    pub memory_savings: usize,
}

impl CompressionMetrics {
    /// Create new compression metrics
    pub fn new(
        original_size: usize,
        compressed_size: usize,
        original_accuracy: f32,
        compressed_accuracy: f32,
    ) -> Self {
        let compression_ratio = if original_size > 0 {
            compressed_size as f32 / original_size as f32
        } else {
            1.0
        };

        let speedup = if compression_ratio > 0.0 {
            1.0 / compression_ratio
        } else {
            1.0
        };

        let accuracy_drop = original_accuracy - compressed_accuracy;

        // Assume f32 weights (4 bytes each)
        let memory_savings = (original_size - compressed_size) * 4;

        Self {
            original_size,
            compressed_size,
            compression_ratio,
            speedup,
            original_accuracy,
            compressed_accuracy,
            accuracy_drop,
            memory_savings,
        }
    }

    /// Check if compression is acceptable
    pub fn is_acceptable(&self, max_accuracy_drop: f32) -> bool {
        self.accuracy_drop <= max_accuracy_drop
    }

    /// Get compression efficiency score
    pub fn efficiency_score(&self) -> f32 {
        // Higher is better: balance between compression and accuracy
        let compression_gain = 1.0 - self.compression_ratio;
        let accuracy_penalty = self.accuracy_drop / 100.0;
        (compression_gain - accuracy_penalty).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compression_config() {
        let config = CompressionConfig::default();
        assert_eq!(config.target_ratio, 0.3);
        assert!(config.enable_layer_merging);
    }

    #[test]
    fn test_architecture_search_depth_reduction() {
        let teacher = vec![128, 256, 128, 64, 10];
        let search = ArchitectureSearch::new(
            teacher.clone(),
            SearchStrategy::DepthReduction,
            0.3,
        ).unwrap();

        let candidates = search.get_all_candidates();
        assert!(!candidates.is_empty());

        let best = search.get_best_candidate().unwrap();
        assert!(best.len() <= teacher.len());
        assert_eq!(best[0], teacher[0]); // Input size preserved
        assert_eq!(*best.last().unwrap(), *teacher.last().unwrap()); // Output size preserved
    }

    #[test]
    fn test_architecture_search_width_reduction() {
        let teacher = vec![128, 256, 128, 64, 10];
        let search = ArchitectureSearch::new(
            teacher.clone(),
            SearchStrategy::WidthReduction,
            0.5,
        ).unwrap();

        let candidates = search.get_all_candidates();
        assert!(!candidates.is_empty());

        let best = search.get_best_candidate().unwrap();
        assert_eq!(best.len(), teacher.len()); // Same depth
        assert_eq!(best[0], teacher[0]); // Input size preserved
    }

    #[test]
    fn test_layer_merging() {
        let merger = LayerMerging::new(MergeStrategy::Linear);

        let w1 = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();
        let w2 = Array2::from_shape_vec((2, 3), vec![2.0; 6]).unwrap();

        let merged = merger.merge_weights(&w1, &w2).unwrap();
        assert_eq!(merged.shape(), &[2, 4]);
    }

    #[test]
    fn test_layer_merging_dimension_mismatch() {
        let merger = LayerMerging::new(MergeStrategy::Linear);

        let w1 = Array2::from_shape_vec((3, 4), vec![1.0; 12]).unwrap();
        let w2 = Array2::from_shape_vec((2, 5), vec![2.0; 10]).unwrap(); // Wrong dimension

        assert!(merger.merge_weights(&w1, &w2).is_err());
    }

    #[test]
    fn test_channel_pruning() {
        let pruning = ChannelPruningGuided::new(vec![0.5, 0.3, 0.2]);

        let weights = Array2::from_shape_vec((4, 8), vec![1.0; 32]).unwrap();
        let activations = Array2::from_shape_vec((10, 4), vec![0.5; 40]).unwrap();

        let importance = pruning.compute_channel_importance(&weights, &activations);
        assert_eq!(importance.len(), 4);

        let channels_to_keep = pruning.select_channels_to_keep(&importance, 0);
        assert_eq!(channels_to_keep.len(), 2); // 50% pruning
    }

    #[test]
    fn test_quantization() {
        let mut qad = QuantizationAwareDistillation::new(8);

        let weights = Array2::from_shape_vec((3, 4), vec![
            0.1, 0.2, -0.3, 0.4,
            -0.5, 0.6, 0.7, -0.8,
            0.9, -1.0, 0.0, 0.5,
        ]).unwrap();

        qad.update_range(&weights);
        let quantized = qad.quantize_weights(&weights);

        assert_eq!(quantized.shape(), weights.shape());

        let error = qad.quantization_error(&weights, &quantized);
        assert!(error >= 0.0);
        assert!(error < 0.1); // Should be small for 8-bit
    }

    #[test]
    fn test_compression_metrics() {
        let metrics = CompressionMetrics::new(
            10000,  // original size
            3000,   // compressed size
            95.0,   // original accuracy
            93.5,   // compressed accuracy
        );

        assert_eq!(metrics.compression_ratio, 0.3);
        assert!((metrics.speedup - 3.333).abs() < 0.01);
        assert_eq!(metrics.accuracy_drop, 1.5);
        assert!(metrics.is_acceptable(2.0)); // Within 2% drop
        assert!(!metrics.is_acceptable(1.0)); // Exceeds 1% drop

        let score = metrics.efficiency_score();
        assert!(score > 0.0);
    }

    #[test]
    fn test_invalid_compression_ratio() {
        let result = ArchitectureSearch::new(
            vec![128, 64, 10],
            SearchStrategy::WidthReduction,
            1.5, // Invalid: > 1.0
        );
        assert!(result.is_err());
    }
}
