//! TensorFlow Lite model validation
//!
//! Validates TFLite models for correctness, compatibility, and optimization.

use super::operators::{TFLiteOperator, OperatorType, BuiltinOperator};
use super::tensors::{TFLiteTensor, TensorType};
use super::flatbuffer::Subgraph;
use serde::{Serialize, Deserialize};

/// TFLite model validator
pub struct TFLiteValidator {
    /// Target TFLite runtime version
    target_version: u32,
    /// Strict mode (fail on warnings)
    strict_mode: bool,
}

impl TFLiteValidator {
    /// Create a new validator
    pub fn new() -> Self {
        Self {
            target_version: 3,
            strict_mode: false,
        }
    }

    /// Set target runtime version
    pub fn with_target_version(mut self, version: u32) -> Self {
        self.target_version = version;
        self
    }

    /// Enable strict mode
    pub fn strict(mut self) -> Self {
        self.strict_mode = true;
        self
    }

    /// Validate a subgraph
    pub fn validate_subgraph(&self, subgraph: &Subgraph) -> ValidationResult {
        let mut result = ValidationResult::new();

        // Basic structural validation
        if let Err(e) = subgraph.validate() {
            result.add_error(e);
            return result; // Don't continue if basic validation fails
        }

        // Validate operators
        for (i, op) in subgraph.operators.iter().enumerate() {
            self.validate_operator(op, i, &subgraph.tensors, &mut result);
        }

        // Validate tensors
        for (i, tensor) in subgraph.tensors.iter().enumerate() {
            self.validate_tensor(tensor, i, &mut result);
        }

        // Check for optimization opportunities
        self.check_optimizations(subgraph, &mut result);

        // Check compatibility
        self.check_compatibility(subgraph, &mut result);

        result
    }

    /// Validate an operator
    fn validate_operator(
        &self,
        op: &TFLiteOperator,
        index: usize,
        tensors: &[TFLiteTensor],
        result: &mut ValidationResult,
    ) {
        // Check operator is valid
        if let Err(e) = op.validate() {
            result.add_error(format!("Operator {}: {}", index, e));
            return;
        }

        // Check operator support
        match &op.op_type {
            OperatorType::Builtin(builtin_op) => {
                if !self.is_operator_supported(*builtin_op) {
                    result.add_warning(CompatibilityWarning::UnsupportedOperator {
                        operator: format!("{:?}", builtin_op),
                        index,
                    });
                }
            }
            OperatorType::Custom(custom_op) => {
                result.add_warning(CompatibilityWarning::CustomOperator {
                    operator: custom_op.name().to_string(),
                    index,
                });
            }
        }

        // Validate tensor shapes match
        self.validate_operator_shapes(op, index, tensors, result);

        // Check quantization compatibility
        self.check_operator_quantization(op, index, tensors, result);
    }

    /// Validate tensor shapes for an operator
    fn validate_operator_shapes(
        &self,
        op: &TFLiteOperator,
        index: usize,
        tensors: &[TFLiteTensor],
        result: &mut ValidationResult,
    ) {
        // Get input/output tensors
        let input_tensors: Vec<&TFLiteTensor> = op
            .inputs
            .iter()
            .filter_map(|&i| tensors.get(i))
            .collect();

        let output_tensors: Vec<&TFLiteTensor> = op
            .outputs
            .iter()
            .filter_map(|&i| tensors.get(i))
            .collect();

        // Basic shape validation based on operator type
        match &op.op_type {
            OperatorType::Builtin(BuiltinOperator::Add)
            | OperatorType::Builtin(BuiltinOperator::Mul)
            | OperatorType::Builtin(BuiltinOperator::Sub) => {
                if input_tensors.len() != 2 {
                    result.add_error(format!(
                        "Operator {} (binary op) requires 2 inputs, got {}",
                        index,
                        input_tensors.len()
                    ));
                }

                if output_tensors.len() != 1 {
                    result.add_error(format!(
                        "Operator {} (binary op) requires 1 output, got {}",
                        index,
                        output_tensors.len()
                    ));
                }
            }
            OperatorType::Builtin(BuiltinOperator::FullyConnected) => {
                if input_tensors.is_empty() {
                    result.add_error(format!(
                        "Operator {} (FullyConnected) requires inputs",
                        index
                    ));
                }
            }
            _ => {
                // Generic validation for other operators
            }
        }
    }

    /// Check operator quantization compatibility
    fn check_operator_quantization(
        &self,
        op: &TFLiteOperator,
        index: usize,
        tensors: &[TFLiteTensor],
        result: &mut ValidationResult,
    ) {
        // Check if operator has mixed quantization
        let mut has_quantized = false;
        let mut has_float = false;

        for &input_idx in &op.inputs {
            if let Some(tensor) = tensors.get(input_idx) {
                if tensor.is_quantized() {
                    has_quantized = true;
                } else if tensor.data_type.is_float() {
                    has_float = true;
                }
            }
        }

        if has_quantized && has_float {
            result.add_warning(CompatibilityWarning::MixedPrecision { index });
        }

        // Check if operator supports quantization
        if has_quantized {
            if let OperatorType::Builtin(builtin_op) = &op.op_type {
                if !builtin_op.supports_quantization() {
                    result.add_warning(CompatibilityWarning::QuantizationNotSupported {
                        operator: format!("{:?}", builtin_op),
                        index,
                    });
                }
            }
        }
    }

    /// Validate a tensor
    fn validate_tensor(
        &self,
        tensor: &TFLiteTensor,
        index: usize,
        result: &mut ValidationResult,
    ) {
        // Basic validation
        if let Err(e) = tensor.validate() {
            result.add_error(format!("Tensor {}: {}", index, e));
            return;
        }

        // Check for dynamic shapes
        if tensor.shape.is_dynamic() {
            result.add_info(format!(
                "Tensor {} '{}' has dynamic shape",
                index, tensor.name
            ));
        }

        // Check tensor size
        let size_bytes = tensor.byte_size();
        if size_bytes > 100_000_000 {
            // > 100MB
            result.add_warning(CompatibilityWarning::LargeTensor {
                name: tensor.name.clone(),
                size_mb: size_bytes / 1_000_000,
            });
        }

        // Validate quantization if present
        if let Some(ref quant) = tensor.quantization {
            if let Err(e) = quant.validate() {
                result.add_error(format!("Tensor {} quantization: {}", index, e));
            }
        }
    }

    /// Check for optimization opportunities
    fn check_optimizations(&self, subgraph: &Subgraph, result: &mut ValidationResult) {
        // Check for consecutive operators that could be fused
        for i in 0..subgraph.operators.len().saturating_sub(1) {
            let op1 = &subgraph.operators[i];
            let op2 = &subgraph.operators[i + 1];

            // Check for Conv -> ReLU fusion opportunity
            if matches!(
                (&op1.op_type, &op2.op_type),
                (
                    OperatorType::Builtin(BuiltinOperator::Conv2D),
                    OperatorType::Builtin(BuiltinOperator::Relu)
                )
            ) {
                result.add_hint(format!(
                    "Operators {} and {} could be fused (Conv2D + ReLU)",
                    i,
                    i + 1
                ));
            }

            // Check for FullyConnected -> ReLU fusion
            if matches!(
                (&op1.op_type, &op2.op_type),
                (
                    OperatorType::Builtin(BuiltinOperator::FullyConnected),
                    OperatorType::Builtin(BuiltinOperator::Relu)
                )
            ) {
                result.add_hint(format!(
                    "Operators {} and {} could be fused (FullyConnected + ReLU)",
                    i,
                    i + 1
                ));
            }
        }

        // Check for quantization opportunities
        let has_float_ops = subgraph.operators.iter().any(|op| {
            op.inputs.iter().any(|&idx| {
                subgraph
                    .tensors
                    .get(idx)
                    .map(|t| t.data_type.is_float())
                    .unwrap_or(false)
            })
        });

        if has_float_ops {
            result.add_hint(
                "Model uses floating-point operations. Consider quantization for better performance."
                    .to_string(),
            );
        }
    }

    /// Check compatibility with target runtime
    fn check_compatibility(&self, subgraph: &Subgraph, result: &mut ValidationResult) {
        // Check for custom operators
        let custom_op_count = subgraph
            .operators
            .iter()
            .filter(|op| op.op_type.is_custom())
            .count();

        if custom_op_count > 0 {
            result.add_warning(CompatibilityWarning::RequiresCustomOps {
                count: custom_op_count,
            });
        }

        // Check tensor types
        for tensor in &subgraph.tensors {
            match tensor.data_type {
                TensorType::Float16 => {
                    if self.target_version < 3 {
                        result.add_warning(CompatibilityWarning::UnsupportedDataType {
                            data_type: "Float16".to_string(),
                            min_version: 3,
                        });
                    }
                }
                TensorType::Complex64 => {
                    result.add_warning(CompatibilityWarning::UnsupportedDataType {
                        data_type: "Complex64".to_string(),
                        min_version: 3,
                    });
                }
                _ => {}
            }
        }
    }

    /// Check if operator is supported in target version
    fn is_operator_supported(&self, _op: BuiltinOperator) -> bool {
        // In a real implementation, this would check against a version matrix
        // For now, assume all built-in operators are supported
        true
    }
}

impl Default for TFLiteValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Validation errors (must be fixed)
    pub errors: Vec<String>,
    /// Compatibility warnings
    pub warnings: Vec<CompatibilityWarning>,
    /// Optimization hints
    pub hints: Vec<String>,
    /// Informational messages
    pub info: Vec<String>,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new() -> Self {
        Self {
            errors: Vec::new(),
            warnings: Vec::new(),
            hints: Vec::new(),
            info: Vec::new(),
        }
    }

    /// Add an error
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }

    /// Add a warning
    pub fn add_warning(&mut self, warning: CompatibilityWarning) {
        self.warnings.push(warning);
    }

    /// Add a hint
    pub fn add_hint(&mut self, hint: String) {
        self.hints.push(hint);
    }

    /// Add info
    pub fn add_info(&mut self, info: String) {
        self.info.push(info);
    }

    /// Check if validation passed (no errors)
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    /// Check if there are any warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Get total issue count
    pub fn total_issues(&self) -> usize {
        self.errors.len() + self.warnings.len()
    }

    /// Generate summary report
    pub fn summary(&self) -> String {
        let mut report = String::new();

        if self.is_valid() {
            report.push_str("✓ Validation passed\n");
        } else {
            report.push_str(&format!("✗ Validation failed with {} errors\n", self.errors.len()));
        }

        if !self.errors.is_empty() {
            report.push_str("\nErrors:\n");
            for error in &self.errors {
                report.push_str(&format!("  - {}\n", error));
            }
        }

        if !self.warnings.is_empty() {
            report.push_str(&format!("\nWarnings ({}):\n", self.warnings.len()));
            for warning in &self.warnings {
                report.push_str(&format!("  - {}\n", warning.description()));
            }
        }

        if !self.hints.is_empty() {
            report.push_str(&format!("\nOptimization hints ({}):\n", self.hints.len()));
            for hint in &self.hints {
                report.push_str(&format!("  - {}\n", hint));
            }
        }

        report
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Compatibility warning types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompatibilityWarning {
    /// Unsupported operator
    UnsupportedOperator { operator: String, index: usize },
    /// Custom operator (requires runtime support)
    CustomOperator { operator: String, index: usize },
    /// Mixed precision (quantized + float)
    MixedPrecision { index: usize },
    /// Operator doesn't support quantization
    QuantizationNotSupported { operator: String, index: usize },
    /// Large tensor
    LargeTensor { name: String, size_mb: usize },
    /// Unsupported data type
    UnsupportedDataType { data_type: String, min_version: u32 },
    /// Requires custom operators
    RequiresCustomOps { count: usize },
}

impl CompatibilityWarning {
    /// Get warning description
    pub fn description(&self) -> String {
        match self {
            CompatibilityWarning::UnsupportedOperator { operator, index } => {
                format!("Operator {} at index {} may not be supported", operator, index)
            }
            CompatibilityWarning::CustomOperator { operator, index } => {
                format!(
                    "Custom operator '{}' at index {} requires runtime support",
                    operator, index
                )
            }
            CompatibilityWarning::MixedPrecision { index } => {
                format!("Operator {} has mixed precision inputs (quantized + float)", index)
            }
            CompatibilityWarning::QuantizationNotSupported { operator, index } => {
                format!(
                    "Operator {} at index {} doesn't support quantization",
                    operator, index
                )
            }
            CompatibilityWarning::LargeTensor { name, size_mb } => {
                format!("Tensor '{}' is large ({} MB)", name, size_mb)
            }
            CompatibilityWarning::UnsupportedDataType {
                data_type,
                min_version,
            } => {
                format!(
                    "Data type {} requires TFLite version {} or higher",
                    data_type, min_version
                )
            }
            CompatibilityWarning::RequiresCustomOps { count } => {
                format!(
                    "Model contains {} custom operator(s) requiring runtime support",
                    count
                )
            }
        }
    }

    /// Get severity level
    pub fn severity(&self) -> WarningSeverity {
        match self {
            CompatibilityWarning::UnsupportedOperator { .. } => WarningSeverity::High,
            CompatibilityWarning::CustomOperator { .. } => WarningSeverity::Medium,
            CompatibilityWarning::MixedPrecision { .. } => WarningSeverity::Low,
            CompatibilityWarning::QuantizationNotSupported { .. } => WarningSeverity::Medium,
            CompatibilityWarning::LargeTensor { .. } => WarningSeverity::Low,
            CompatibilityWarning::UnsupportedDataType { .. } => WarningSeverity::High,
            CompatibilityWarning::RequiresCustomOps { .. } => WarningSeverity::Medium,
        }
    }
}

/// Warning severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::tflite::tensors::{TensorShape, QuantizationParams};

    #[test]
    fn test_validator_creation() {
        let validator = TFLiteValidator::new();
        assert_eq!(validator.target_version, 3);
        assert!(!validator.strict_mode);
    }

    #[test]
    fn test_validator_with_options() {
        let validator = TFLiteValidator::new()
            .with_target_version(2)
            .strict();

        assert_eq!(validator.target_version, 2);
        assert!(validator.strict_mode);
    }

    #[test]
    fn test_validation_result_creation() {
        let result = ValidationResult::new();
        assert!(result.is_valid());
        assert!(!result.has_warnings());
        assert_eq!(result.total_issues(), 0);
    }

    #[test]
    fn test_validation_result_with_errors() {
        let mut result = ValidationResult::new();
        result.add_error("Test error".to_string());

        assert!(!result.is_valid());
        assert_eq!(result.total_issues(), 1);
    }

    #[test]
    fn test_validation_result_with_warnings() {
        let mut result = ValidationResult::new();
        result.add_warning(CompatibilityWarning::MixedPrecision { index: 0 });

        assert!(result.is_valid()); // Warnings don't fail validation
        assert!(result.has_warnings());
        assert_eq!(result.total_issues(), 1);
    }

    #[test]
    fn test_validation_result_summary() {
        let mut result = ValidationResult::new();
        result.add_error("Error 1".to_string());
        result.add_warning(CompatibilityWarning::MixedPrecision { index: 0 });
        result.add_hint("Consider optimization".to_string());

        let summary = result.summary();
        assert!(summary.contains("Validation failed"));
        assert!(summary.contains("Error 1"));
        assert!(summary.contains("mixed precision"));
    }

    #[test]
    fn test_compatibility_warning_description() {
        let warning = CompatibilityWarning::UnsupportedOperator {
            operator: "TestOp".to_string(),
            index: 5,
        };

        let desc = warning.description();
        assert!(desc.contains("TestOp"));
        assert!(desc.contains("5"));
    }

    #[test]
    fn test_compatibility_warning_severity() {
        let high = CompatibilityWarning::UnsupportedOperator {
            operator: "Test".to_string(),
            index: 0,
        };
        assert_eq!(high.severity(), WarningSeverity::High);

        let medium = CompatibilityWarning::CustomOperator {
            operator: "Test".to_string(),
            index: 0,
        };
        assert_eq!(medium.severity(), WarningSeverity::Medium);

        let low = CompatibilityWarning::MixedPrecision { index: 0 };
        assert_eq!(low.severity(), WarningSeverity::Low);
    }

    #[test]
    fn test_validate_tensor_valid() {
        let validator = TFLiteValidator::new();
        let mut result = ValidationResult::new();

        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![1, 224, 224, 3]),
            TensorType::Float32,
        );

        validator.validate_tensor(&tensor, 0, &mut result);
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_tensor_dynamic_shape() {
        let validator = TFLiteValidator::new();
        let mut result = ValidationResult::new();

        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![-1, 224, 224, 3]),
            TensorType::Float32,
        );

        validator.validate_tensor(&tensor, 0, &mut result);
        assert!(result.is_valid());
        assert!(!result.info.is_empty()); // Should have info about dynamic shape
    }

    #[test]
    fn test_validate_tensor_large() {
        let validator = TFLiteValidator::new();
        let mut result = ValidationResult::new();

        // Create a very large tensor
        let tensor = TFLiteTensor::new(
            "large".to_string(),
            TensorShape::new(vec![1000, 1000, 100]),
            TensorType::Float32,
        );

        validator.validate_tensor(&tensor, 0, &mut result);
        assert!(result.has_warnings());
    }

    #[test]
    fn test_validate_tensor_invalid_quantization() {
        let validator = TFLiteValidator::new();
        let mut result = ValidationResult::new();

        let mut quant = QuantizationParams::per_tensor(0.5, 0);
        quant.scales[0] = -1.0; // Invalid: negative scale

        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![10]),
            TensorType::Int8,
        )
        .with_quantization(quant);

        validator.validate_tensor(&tensor, 0, &mut result);
        assert!(!result.is_valid());
    }
}
