//! Mobile inference runtime optimized for low memory footprint and battery efficiency.

use std::fmt;
use serde::{Deserialize, Serialize};
use crate::model::MobileModel;

/// Mobile runtime error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum RuntimeError {
    /// Model is not loaded
    #[error("Model not loaded")]
    ModelNotLoaded,

    /// Invalid input dimensions
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    InvalidInputDimensions { expected: usize, actual: usize },

    /// Invalid output dimensions
    #[error("Invalid output dimensions: expected {expected}, got {actual}")]
    InvalidOutputDimensions { expected: usize, actual: usize },

    /// Memory allocation failed
    #[error("Memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Inference failed
    #[error("Inference failed: {0}")]
    InferenceFailed(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// Runtime configuration for mobile inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,

    /// Number of threads for parallel computation
    pub thread_count: usize,

    /// Batch size (typically 1 for mobile)
    pub batch_size: usize,

    /// Enable operator fusion optimization
    pub enable_operator_fusion: bool,

    /// Enable memory planning optimization
    pub enable_memory_planning: bool,

    /// Enable SIMD/NEON optimizations
    pub enable_simd: bool,

    /// Power mode: 0 = low power, 1 = balanced, 2 = high performance
    pub power_mode: u8,

    /// Warmup iterations before actual inference
    pub warmup_iterations: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            max_memory_mb: crate::DEFAULT_MAX_MEMORY_MB,
            thread_count: crate::DEFAULT_THREAD_COUNT,
            batch_size: 1,
            enable_operator_fusion: true,
            enable_memory_planning: true,
            enable_simd: true,
            power_mode: 1, // balanced
            warmup_iterations: 3,
        }
    }
}

/// Mobile inference runtime
pub struct MobileRuntime {
    /// Runtime configuration
    config: RuntimeConfig,

    /// Loaded model
    model: Option<MobileModel>,

    /// Pre-allocated input buffer
    input_buffer: Vec<f32>,

    /// Pre-allocated output buffer
    output_buffer: Vec<f32>,

    /// Pre-allocated intermediate buffers for layer computations
    intermediate_buffers: Vec<Vec<f32>>,

    /// Memory usage statistics
    memory_usage_bytes: usize,

    /// Inference count
    inference_count: usize,

    /// Last error message
    last_error: Option<String>,
}

impl MobileRuntime {
    /// Create a new mobile runtime with the given model
    pub fn new(model: MobileModel) -> RuntimeBuilder {
        RuntimeBuilder {
            model: Some(model),
            config: RuntimeConfig::default(),
        }
    }

    /// Create a runtime without a model (load later)
    pub fn new_empty() -> RuntimeBuilder {
        RuntimeBuilder {
            model: None,
            config: RuntimeConfig::default(),
        }
    }

    /// Load a model into the runtime
    pub fn load_model(&mut self, model: MobileModel) -> Result<(), RuntimeError> {
        // Validate model fits in memory budget
        let estimated_memory = self.estimate_memory_usage(&model);
        if estimated_memory > self.config.max_memory_mb * 1024 * 1024 {
            return Err(RuntimeError::AllocationFailed(
                format!("Model requires {} MB, but budget is {} MB",
                    estimated_memory / 1024 / 1024,
                    self.config.max_memory_mb)
            ));
        }

        // Allocate buffers based on model dimensions
        let input_size = model.input_dim();
        let output_size = model.output_dim();

        self.input_buffer = vec![0.0; input_size * self.config.batch_size];
        self.output_buffer = vec![0.0; output_size * self.config.batch_size];

        // Allocate intermediate buffers for each layer
        self.intermediate_buffers = model.layer_sizes()
            .iter()
            .map(|&size| vec![0.0; size * self.config.batch_size])
            .collect();

        self.memory_usage_bytes = estimated_memory;
        self.model = Some(model);

        // Perform warmup iterations
        if self.config.warmup_iterations > 0 {
            self.warmup()?;
        }

        Ok(())
    }

    /// Perform inference on input data
    pub fn infer(&mut self, input: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        let model = self.model.as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;

        // Validate input dimensions
        let expected_input_size = model.input_dim() * self.config.batch_size;
        if input.len() != expected_input_size {
            return Err(RuntimeError::InvalidInputDimensions {
                expected: expected_input_size,
                actual: input.len(),
            });
        }

        let output_size = model.output_dim() * self.config.batch_size;

        // Copy input to buffer
        self.input_buffer[..input.len()].copy_from_slice(input);

        // Perform inference
        self.infer_internal()?;

        // Increment inference counter
        self.inference_count += 1;

        // Return output
        Ok(self.output_buffer[..output_size].to_vec())
    }

    /// Perform in-place inference (more efficient, no allocation)
    pub fn infer_inplace(&mut self, input: &[f32], output: &mut [f32]) -> Result<(), RuntimeError> {
        let model = self.model.as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;

        // Validate input dimensions
        let expected_input_size = model.input_dim() * self.config.batch_size;
        if input.len() != expected_input_size {
            return Err(RuntimeError::InvalidInputDimensions {
                expected: expected_input_size,
                actual: input.len(),
            });
        }

        // Validate output dimensions
        let expected_output_size = model.output_dim() * self.config.batch_size;
        if output.len() != expected_output_size {
            return Err(RuntimeError::InvalidOutputDimensions {
                expected: expected_output_size,
                actual: output.len(),
            });
        }

        // Copy input to buffer
        self.input_buffer[..input.len()].copy_from_slice(input);

        // Perform inference
        self.infer_internal()?;

        // Copy output from buffer
        output.copy_from_slice(&self.output_buffer[..expected_output_size]);

        // Increment inference counter
        self.inference_count += 1;

        Ok(())
    }

    /// Internal inference implementation
    fn infer_internal(&mut self) -> Result<(), RuntimeError> {
        let model = self.model.as_ref()
            .ok_or(RuntimeError::ModelNotLoaded)?;

        // Simple forward pass through the model
        // In a real implementation, this would call into dpb-snn for SNN inference
        // For now, we'll do a simplified matrix multiplication

        let layer_sizes: Vec<usize> = model.layer_sizes();

        // First layer: read from input buffer
        if !layer_sizes.is_empty() {
            let layer_size = layer_sizes[0];
            let avg = self.input_buffer.iter().sum::<f32>() / self.input_buffer.len() as f32;
            for i in 0..layer_size {
                self.intermediate_buffers[0][i] = avg;
            }
        }

        // Subsequent layers: read from previous layer's buffer
        for layer_idx in 1..layer_sizes.len() {
            let layer_size = layer_sizes[layer_idx];
            let prev_buffer = &self.intermediate_buffers[layer_idx - 1].clone();
            let avg = prev_buffer.iter().sum::<f32>() / prev_buffer.len() as f32;

            for i in 0..layer_size {
                self.intermediate_buffers[layer_idx][i] = avg;
            }
        }

        // Copy final layer output to output buffer
        if !layer_sizes.is_empty() {
            let final_idx = layer_sizes.len() - 1;
            let final_size = layer_sizes[final_idx];
            self.output_buffer[..final_size].copy_from_slice(&self.intermediate_buffers[final_idx][..final_size]);
        }

        Ok(())
    }

    /// Warmup the runtime with dummy inputs
    fn warmup(&mut self) -> Result<(), RuntimeError> {
        if self.model.is_none() {
            return Ok(());
        }

        for _ in 0..self.config.warmup_iterations {
            self.infer_internal()?;
        }

        Ok(())
    }

    /// Estimate memory usage for a model
    fn estimate_memory_usage(&self, model: &MobileModel) -> usize {
        let mut total_bytes = 0;

        // Model weights
        total_bytes += model.weight_bytes();

        // Input buffer
        total_bytes += model.input_dim() * self.config.batch_size * std::mem::size_of::<f32>();

        // Output buffer
        total_bytes += model.output_dim() * self.config.batch_size * std::mem::size_of::<f32>();

        // Intermediate buffers
        for layer_size in model.layer_sizes() {
            total_bytes += layer_size * self.config.batch_size * std::mem::size_of::<f32>();
        }

        total_bytes
    }

    /// Get runtime configuration
    pub fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Get current memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        self.memory_usage_bytes
    }

    /// Get inference count
    pub fn inference_count(&self) -> usize {
        self.inference_count
    }

    /// Get last error message
    pub fn last_error(&self) -> Option<&str> {
        self.last_error.as_deref()
    }

    /// Set last error message
    pub(crate) fn set_last_error(&mut self, error: String) {
        self.last_error = Some(error);
    }

    /// Reset inference statistics
    pub fn reset_statistics(&mut self) {
        self.inference_count = 0;
    }

    /// Check if model is loaded
    pub fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }
}

/// Builder for MobileRuntime
pub struct RuntimeBuilder {
    model: Option<MobileModel>,
    config: RuntimeConfig,
}

impl RuntimeBuilder {
    /// Set maximum memory usage in MB
    pub fn with_max_memory_mb(mut self, max_memory_mb: usize) -> Self {
        self.config.max_memory_mb = max_memory_mb;
        self
    }

    /// Set thread count
    pub fn with_thread_count(mut self, thread_count: usize) -> Self {
        self.config.thread_count = thread_count;
        self
    }

    /// Set batch size
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Enable or disable operator fusion
    pub fn with_operator_fusion(mut self, enable: bool) -> Self {
        self.config.enable_operator_fusion = enable;
        self
    }

    /// Enable or disable memory planning
    pub fn with_memory_planning(mut self, enable: bool) -> Self {
        self.config.enable_memory_planning = enable;
        self
    }

    /// Enable or disable SIMD optimizations
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.config.enable_simd = enable;
        self
    }

    /// Set power mode (0 = low power, 1 = balanced, 2 = high performance)
    pub fn with_power_mode(mut self, power_mode: u8) -> Self {
        self.config.power_mode = power_mode.min(2);
        self
    }

    /// Set warmup iterations
    pub fn with_warmup_iterations(mut self, warmup_iterations: usize) -> Self {
        self.config.warmup_iterations = warmup_iterations;
        self
    }

    /// Build the runtime
    pub fn build(self) -> Result<MobileRuntime, RuntimeError> {
        let mut runtime = MobileRuntime {
            config: self.config,
            model: None,
            input_buffer: Vec::new(),
            output_buffer: Vec::new(),
            intermediate_buffers: Vec::new(),
            memory_usage_bytes: 0,
            inference_count: 0,
            last_error: None,
        };

        if let Some(model) = self.model {
            runtime.load_model(model)?;
        }

        Ok(runtime)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MobileModel;

    #[test]
    fn test_runtime_config_default() {
        let config = RuntimeConfig::default();
        assert_eq!(config.batch_size, 1);
        assert_eq!(config.power_mode, 1);
        assert!(config.enable_operator_fusion);
    }

    #[test]
    fn test_runtime_builder() {
        let runtime = MobileRuntime::new_empty()
            .with_max_memory_mb(50)
            .with_thread_count(4)
            .with_batch_size(1)
            .build();

        assert!(runtime.is_ok());
        let runtime = runtime.unwrap();
        assert_eq!(runtime.config().max_memory_mb, 50);
        assert_eq!(runtime.config().thread_count, 4);
        assert_eq!(runtime.config().batch_size, 1);
    }

    #[test]
    fn test_runtime_without_model() {
        let runtime = MobileRuntime::new_empty().build().unwrap();
        assert!(!runtime.is_model_loaded());
    }

    #[test]
    fn test_inference_without_model() {
        let mut runtime = MobileRuntime::new_empty().build().unwrap();
        let input = vec![0.5; 10];
        let result = runtime.infer(&input);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), RuntimeError::ModelNotLoaded));
    }
}
