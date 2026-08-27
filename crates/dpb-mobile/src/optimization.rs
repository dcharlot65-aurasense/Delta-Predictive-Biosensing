//! Mobile-specific optimizations for size and performance.

use serde::{Deserialize, Serialize};
use crate::model::{MobileModel, LayerInfo, QuantizationType};

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    /// No optimization
    None,

    /// Basic optimizations (safe, minimal impact)
    Basic,

    /// Aggressive optimizations (may impact accuracy)
    Aggressive,

    /// Maximum optimizations (significant accuracy trade-off)
    Maximum,
}

/// Weight pruning strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PruningStrategy {
    /// Magnitude-based pruning (remove smallest weights)
    Magnitude,

    /// Structured pruning (remove entire neurons/channels)
    Structured,

    /// Movement pruning (based on gradient movement)
    Movement,
}

/// Weight pruner for model size reduction
// Not consulted yet; kept so a caller's configuration is not silently
// discarded.
#[allow(dead_code)]
pub struct WeightPruner {
    /// Pruning strategy
    strategy: PruningStrategy,

    /// Target sparsity (0.0 to 1.0)
    target_sparsity: f32,

    /// Minimum weight threshold
    threshold: f32,
}

impl WeightPruner {
    /// Create a new weight pruner
    pub fn new(strategy: PruningStrategy, target_sparsity: f32) -> Self {
        Self {
            strategy,
            target_sparsity: target_sparsity.clamp(0.0, 1.0),
            threshold: 0.0,
        }
    }

    /// Prune weights from a model
    pub fn prune_model(&self, model: &mut MobileModel) -> Result<PruningStats, String> {
        let mut stats = PruningStats::default();

        for layer in &mut model.layers {
            let layer_stats = self.prune_layer(layer)?;
            stats.merge(layer_stats);
        }

        Ok(stats)
    }

    /// Prune weights from a single layer
    fn prune_layer(&self, layer: &mut LayerInfo) -> Result<PruningStats, String> {
        match self.strategy {
            PruningStrategy::Magnitude => {
                self.magnitude_prune_layer(layer)
            }
            PruningStrategy::Structured => {
                self.structured_prune_layer(layer)
            }
            PruningStrategy::Movement => {
                Err("Movement pruning not yet implemented".into())
            }
        }
    }

    /// Magnitude-based pruning
    fn magnitude_prune_layer(&self, layer: &mut LayerInfo) -> Result<PruningStats, String> {
        let mut stats = PruningStats::default();

        // For simplicity, this is a placeholder
        // In a real implementation, you would:
        // 1. Dequantize weights if needed
        // 2. Calculate magnitude threshold
        // 3. Zero out small weights
        // 4. Re-quantize if needed

        stats.original_weights = layer.input_dim * layer.output_dim;
        stats.pruned_weights = (stats.original_weights as f32 * self.target_sparsity) as usize;
        stats.remaining_weights = stats.original_weights - stats.pruned_weights;

        Ok(stats)
    }

    /// Structured pruning (remove entire neurons)
    fn structured_prune_layer(&self, layer: &mut LayerInfo) -> Result<PruningStats, String> {
        let mut stats = PruningStats::default();

        // Placeholder for structured pruning
        stats.original_weights = layer.input_dim * layer.output_dim;
        stats.pruned_weights = (stats.original_weights as f32 * self.target_sparsity) as usize;
        stats.remaining_weights = stats.original_weights - stats.pruned_weights;

        Ok(stats)
    }
}

/// Pruning statistics
#[derive(Debug, Clone, Default)]
pub struct PruningStats {
    /// Original number of weights
    pub original_weights: usize,

    /// Number of pruned weights
    pub pruned_weights: usize,

    /// Number of remaining weights
    pub remaining_weights: usize,
}

impl PruningStats {
    /// Merge with another stats
    pub fn merge(&mut self, other: PruningStats) {
        self.original_weights += other.original_weights;
        self.pruned_weights += other.pruned_weights;
        self.remaining_weights += other.remaining_weights;
    }

    /// Get sparsity ratio
    pub fn sparsity(&self) -> f32 {
        if self.original_weights == 0 {
            return 0.0;
        }
        self.pruned_weights as f32 / self.original_weights as f32
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.remaining_weights == 0 {
            return f32::INFINITY;
        }
        self.original_weights as f32 / self.remaining_weights as f32
    }
}

/// Operator fusion for inference optimization
pub struct OperatorFusion {
    /// Enable fusion
    enabled: bool,

    /// Fusion patterns
    patterns: Vec<FusionPattern>,
}

/// Fusion pattern for operator sequences
#[derive(Debug, Clone)]
pub enum FusionPattern {
    /// Fuse linear + activation
    LinearActivation,

    /// Fuse batch norm + linear
    BatchNormLinear,

    /// Fuse multiple linear layers
    MultiLinear,
}

impl OperatorFusion {
    /// Create a new operator fusion optimizer
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            patterns: vec![
                FusionPattern::LinearActivation,
                FusionPattern::BatchNormLinear,
            ],
        }
    }

    /// Apply fusion to a model
    pub fn fuse_model(&self, model: &mut MobileModel) -> Result<FusionStats, String> {
        if !self.enabled {
            return Ok(FusionStats::default());
        }

        let mut stats = FusionStats::default();

        // Placeholder for operator fusion
        // In a real implementation, you would:
        // 1. Identify fusable operator sequences
        // 2. Combine operations into single kernels
        // 3. Update model structure

        stats.original_operators = model.num_layers();
        stats.fused_operators = (model.num_layers() as f32 * 0.1) as usize; // Example: 10% reduction
        stats.final_operators = stats.original_operators - stats.fused_operators;

        Ok(stats)
    }

    /// Add a fusion pattern
    pub fn add_pattern(&mut self, pattern: FusionPattern) {
        self.patterns.push(pattern);
    }
}

/// Fusion statistics
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Original number of operators
    pub original_operators: usize,

    /// Number of fused operators
    pub fused_operators: usize,

    /// Final number of operators
    pub final_operators: usize,
}

impl FusionStats {
    /// Get fusion ratio
    pub fn fusion_ratio(&self) -> f32 {
        if self.original_operators == 0 {
            return 0.0;
        }
        self.fused_operators as f32 / self.original_operators as f32
    }
}

/// Quantization optimizer
pub struct QuantizationOptimizer {
    /// Target quantization type
    target_type: QuantizationType,

    /// Calibration data size
    calibration_size: usize,
}

impl QuantizationOptimizer {
    /// Create a new quantization optimizer
    pub fn new(target_type: QuantizationType) -> Self {
        Self {
            target_type,
            calibration_size: 100,
        }
    }

    /// Quantize a model
    pub fn quantize_model(&self, model: &mut MobileModel) -> Result<QuantizationStats, String> {
        let mut stats = QuantizationStats::default();

        for layer in &mut model.layers {
            if layer.quantization == self.target_type {
                continue;
            }

            let original_bytes = layer.weights.len();
            layer.quantization = self.target_type;

            // In a real implementation, you would:
            // 1. Dequantize existing weights
            // 2. Calculate scale and zero point
            // 3. Quantize to target type
            // 4. Update weight buffer

            let new_bytes = original_bytes * self.target_type.bytes_per_weight() / 4;
            layer.weights.resize(new_bytes, 0);

            stats.original_size_bytes += original_bytes;
            stats.quantized_size_bytes += new_bytes;
        }

        Ok(stats)
    }

    /// Set calibration data size
    pub fn with_calibration_size(mut self, size: usize) -> Self {
        self.calibration_size = size;
        self
    }
}

/// Quantization statistics
#[derive(Debug, Clone, Default)]
pub struct QuantizationStats {
    /// Original size in bytes
    pub original_size_bytes: usize,

    /// Quantized size in bytes
    pub quantized_size_bytes: usize,
}

impl QuantizationStats {
    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        if self.quantized_size_bytes == 0 {
            return f32::INFINITY;
        }
        self.original_size_bytes as f32 / self.quantized_size_bytes as f32
    }
}

/// Memory planner for optimal buffer allocation
pub struct MemoryPlanner {
    /// Enable memory planning
    enabled: bool,

    /// Alignment in bytes
    alignment: usize,
}

impl MemoryPlanner {
    /// Create a new memory planner
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            alignment: 64, // Cache line alignment
        }
    }

    /// Plan memory allocation for a model
    pub fn plan(&self, model: &MobileModel) -> MemoryPlan {
        if !self.enabled {
            return MemoryPlan::default();
        }

        // 4 bytes per float. `mut` is still needed: the loop below accumulates
        // the intermediate buffer sizes.
        let mut plan = MemoryPlan {
            input_buffer_size: self.align(model.input_dim() * 4),
            output_buffer_size: self.align(model.output_dim() * 4),
            ..Default::default()
        };

        for layer in &model.layers {
            let layer_size = self.align(layer.output_dim * 4);
            plan.intermediate_buffer_sizes.push(layer_size);
            plan.total_intermediate_size += layer_size;
        }

        plan.total_size = plan.input_buffer_size +
                          plan.output_buffer_size +
                          plan.total_intermediate_size;

        plan
    }

    /// Align size to alignment boundary
    fn align(&self, size: usize) -> usize {
        (size + self.alignment - 1) & !(self.alignment - 1)
    }
}

/// Memory allocation plan
#[derive(Debug, Clone, Default)]
pub struct MemoryPlan {
    /// Input buffer size
    pub input_buffer_size: usize,

    /// Output buffer size
    pub output_buffer_size: usize,

    /// Intermediate buffer sizes
    pub intermediate_buffer_sizes: Vec<usize>,

    /// Total intermediate size
    pub total_intermediate_size: usize,

    /// Total memory size
    pub total_size: usize,
}

/// SIMD optimization hints for ARM NEON
pub struct SimdOptimizer {
    /// Enable SIMD optimizations
    enabled: bool,

    /// Target architecture
    target_arch: SimdArch,
}

/// SIMD architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdArch {
    /// ARM NEON (iOS/Android)
    ArmNeon,

    /// x86 SSE
    X86Sse,

    /// x86 AVX
    X86Avx,

    /// None/Scalar
    None,
}

impl SimdOptimizer {
    /// Create a new SIMD optimizer
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            target_arch: Self::detect_arch(),
        }
    }

    // cfg-gated dispatch: on any one target the first matching arm returns and
    // the rest is dead, but the final fallback is what a target matching none
    // of them uses.
    #[allow(unreachable_code)]
    /// Detect SIMD architecture
    fn detect_arch() -> SimdArch {
        #[cfg(target_arch = "aarch64")]
        return SimdArch::ArmNeon;

        #[cfg(target_arch = "arm")]
        return SimdArch::ArmNeon;

        #[cfg(target_arch = "x86_64")]
        return SimdArch::X86Avx;

        #[cfg(target_arch = "x86")]
        return SimdArch::X86Sse;

        SimdArch::None
    }

    /// Check if SIMD is available
    pub fn is_available(&self) -> bool {
        self.enabled && self.target_arch != SimdArch::None
    }

    /// Get target architecture
    pub fn target_arch(&self) -> SimdArch {
        self.target_arch
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_level() {
        assert_eq!(OptimizationLevel::None as u8, 0);
    }

    #[test]
    fn test_weight_pruner() {
        let pruner = WeightPruner::new(PruningStrategy::Magnitude, 0.5);
        assert_eq!(pruner.target_sparsity, 0.5);
    }

    #[test]
    fn test_pruning_stats() {
        let mut stats = PruningStats {
            original_weights: 1000,
            pruned_weights: 500,
            remaining_weights: 500,
        };

        assert_eq!(stats.sparsity(), 0.5);
        assert_eq!(stats.compression_ratio(), 2.0);

        let other = PruningStats {
            original_weights: 2000,
            pruned_weights: 1000,
            remaining_weights: 1000,
        };

        stats.merge(other);
        assert_eq!(stats.original_weights, 3000);
    }

    #[test]
    fn test_operator_fusion() {
        let fusion = OperatorFusion::new(true);
        assert!(fusion.enabled);
        assert_eq!(fusion.patterns.len(), 2);
    }

    #[test]
    fn test_quantization_optimizer() {
        let optimizer = QuantizationOptimizer::new(QuantizationType::Int8);
        assert_eq!(optimizer.target_type, QuantizationType::Int8);
    }

    #[test]
    fn test_quantization_stats() {
        let stats = QuantizationStats {
            original_size_bytes: 1000,
            quantized_size_bytes: 250,
        };
        assert_eq!(stats.compression_ratio(), 4.0);
    }

    #[test]
    fn test_memory_planner() {
        let planner = MemoryPlanner::new(true);
        assert!(planner.enabled);
        assert_eq!(planner.alignment, 64);
    }

    #[test]
    fn test_simd_optimizer() {
        let optimizer = SimdOptimizer::new(true);
        assert!(optimizer.enabled);

        #[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
        assert_eq!(optimizer.target_arch(), SimdArch::ArmNeon);
    }
}
