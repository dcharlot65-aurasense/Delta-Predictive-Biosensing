//! TensorFlow Lite operator definitions
//!
//! Defines built-in and custom operators for TFLite models, including
//! special operators for spiking neural networks.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TensorFlow Lite operator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFLiteOperator {
    /// Operator type (built-in or custom)
    pub op_type: OperatorType,
    /// Input tensor indices
    pub inputs: Vec<usize>,
    /// Output tensor indices
    pub outputs: Vec<usize>,
    /// Operator options (type-specific parameters)
    pub options: OperatorOptions,
    /// Custom code (for custom operators)
    pub custom_code: Option<String>,
}

impl TFLiteOperator {
    /// Create a new operator.
    ///
    /// For a [`OperatorType::Custom`] op the custom code is derived from
    /// [`CustomOperator::name`], because TFLite resolves a custom operator by
    /// exactly that registration string and a model carrying a custom op with no
    /// code is invalid. Deriving it here rather than leaving it to each call site
    /// keeps that invalid state unreachable by default; use
    /// [`with_custom_code`](Self::with_custom_code) to override the name.
    pub fn new(op_type: OperatorType, inputs: Vec<usize>, outputs: Vec<usize>) -> Self {
        let custom_code = match &op_type {
            OperatorType::Custom(op) => Some(op.name().to_string()),
            OperatorType::Builtin(_) => None,
        };
        Self {
            op_type,
            inputs,
            outputs,
            options: OperatorOptions::None,
            custom_code,
        }
    }

    /// Set operator options
    pub fn with_options(mut self, options: OperatorOptions) -> Self {
        self.options = options;
        self
    }

    /// Set custom code (for custom operators)
    pub fn with_custom_code(mut self, code: String) -> Self {
        self.custom_code = Some(code);
        self
    }

    /// Validate operator configuration
    pub fn validate(&self) -> Result<(), String> {
        // Check inputs/outputs not empty
        if self.inputs.is_empty() {
            return Err("Operator must have at least one input".to_string());
        }

        if self.outputs.is_empty() {
            return Err("Operator must have at least one output".to_string());
        }

        // Validate custom operators have custom code
        if matches!(self.op_type, OperatorType::Custom(_)) && self.custom_code.is_none() {
            return Err("Custom operators must have custom code".to_string());
        }

        Ok(())
    }
}

/// Operator type (built-in or custom)
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OperatorType {
    /// Built-in TFLite operator
    Builtin(BuiltinOperator),
    /// Custom operator (for SNN-specific ops)
    Custom(CustomOperator),
}

impl OperatorType {
    /// Get operator version
    pub fn version(&self) -> OperatorVersion {
        match self {
            OperatorType::Builtin(op) => op.version(),
            OperatorType::Custom(op) => op.version(),
        }
    }

    /// Check if operator is custom
    pub fn is_custom(&self) -> bool {
        matches!(self, OperatorType::Custom(_))
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Built-in TFLite operators
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BuiltinOperator {
    // Math operations
    Add,
    Sub,
    Mul,
    Div,
    Exp,
    Log,
    Sqrt,
    Abs,
    Neg,

    // Neural network layers
    FullyConnected,
    Conv2D,
    DepthwiseConv2D,
    TransposeConv,

    // Pooling
    AveragePool2D,
    MaxPool2D,

    // Activations
    Relu,
    Relu6,
    Tanh,
    Sigmoid,
    Softmax,

    // Normalization
    BatchNorm,
    LayerNorm,

    // Recurrent
    LSTM,
    RNN,
    GRU,

    // Reduction
    ReduceSum,
    ReduceMean,
    ReduceMax,
    ReduceMin,

    // Comparison
    Equal,
    NotEqual,
    Greater,
    GreaterEqual,
    Less,
    LessEqual,

    // Shape manipulation
    Reshape,
    Transpose,
    Concat,
    Split,
    Slice,
    Pad,

    // Other
    Cast,
    Quantize,
    Dequantize,
}

impl BuiltinOperator {
    /// Get operator code
    pub fn to_code(self) -> u8 {
        match self {
            BuiltinOperator::Add => 0,
            BuiltinOperator::Sub => 1,
            BuiltinOperator::Mul => 2,
            BuiltinOperator::Div => 3,
            BuiltinOperator::FullyConnected => 9,
            BuiltinOperator::Conv2D => 3,
            BuiltinOperator::DepthwiseConv2D => 4,
            BuiltinOperator::AveragePool2D => 1,
            BuiltinOperator::MaxPool2D => 17,
            BuiltinOperator::Relu => 18,
            BuiltinOperator::Relu6 => 19,
            BuiltinOperator::Reshape => 22,
            BuiltinOperator::Softmax => 25,
            BuiltinOperator::Concat => 2,
            BuiltinOperator::LSTM => 16,
            BuiltinOperator::ReduceSum => 74,
            BuiltinOperator::ReduceMean => 103,
            BuiltinOperator::Cast => 53,
            BuiltinOperator::Quantize => 114,
            BuiltinOperator::Dequantize => 6,
            BuiltinOperator::Sigmoid => 26,
            BuiltinOperator::Tanh => 28,
            BuiltinOperator::Transpose => 39,
            BuiltinOperator::Pad => 34,
            BuiltinOperator::Slice => 45,
            BuiltinOperator::Split => 49,
            _ => 255, // Placeholder for others
        }
    }

    /// Get operator version
    pub fn version(&self) -> OperatorVersion {
        OperatorVersion::V1 // Most operators are version 1
    }

    /// Check if operator supports quantization
    pub fn supports_quantization(&self) -> bool {
        matches!(
            self,
            BuiltinOperator::FullyConnected
                | BuiltinOperator::Conv2D
                | BuiltinOperator::DepthwiseConv2D
                | BuiltinOperator::Add
                | BuiltinOperator::Mul
        )
    }
}

/// Custom operators for SNN-specific operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CustomOperator {
    /// Leaky Integrate-and-Fire neuron
    LIFNeuron,
    /// Adaptive LIF neuron
    AdaptiveLIFNeuron,
    /// Izhikevich neuron model
    IzhikevichNeuron,
    /// Spike encoding (rate, latency, etc.)
    SpikeEncoder,
    /// Spike decoding
    SpikeDecoder,
    /// Spike pooling
    SpikePooling,
    /// Temporal convolution with spikes
    TemporalConv,
}

impl CustomOperator {
    /// Get custom operator name
    pub fn name(&self) -> &'static str {
        match self {
            CustomOperator::LIFNeuron => "LIF_NEURON",
            CustomOperator::AdaptiveLIFNeuron => "ADAPTIVE_LIF_NEURON",
            CustomOperator::IzhikevichNeuron => "IZHIKEVICH_NEURON",
            CustomOperator::SpikeEncoder => "SPIKE_ENCODER",
            CustomOperator::SpikeDecoder => "SPIKE_DECODER",
            CustomOperator::SpikePooling => "SPIKE_POOLING",
            CustomOperator::TemporalConv => "TEMPORAL_CONV",
        }
    }

    /// Get operator version
    pub fn version(&self) -> OperatorVersion {
        OperatorVersion::V1
    }

    /// Get required parameters for custom operator
    pub fn required_params(&self) -> Vec<&'static str> {
        match self {
            CustomOperator::LIFNeuron => {
                vec!["tau_mem", "tau_syn", "v_threshold", "v_reset", "dt"]
            }
            CustomOperator::AdaptiveLIFNeuron => {
                vec![
                    "tau_mem",
                    "tau_syn",
                    "v_threshold",
                    "v_reset",
                    "dt",
                    "tau_adapt",
                ]
            }
            CustomOperator::IzhikevichNeuron => {
                vec!["a", "b", "c", "d", "dt"]
            }
            CustomOperator::SpikeEncoder => {
                vec!["encoding_type", "time_steps"]
            }
            CustomOperator::SpikeDecoder => {
                vec!["decoding_type", "time_steps"]
            }
            CustomOperator::SpikePooling => {
                vec!["pool_type", "kernel_size", "stride"]
            }
            CustomOperator::TemporalConv => {
                vec!["kernel_size", "stride", "time_steps"]
            }
        }
    }
}

/// Operator version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperatorVersion {
    V1 = 1,
    V2 = 2,
    V3 = 3,
    V4 = 4,
}

impl OperatorVersion {
    /// Convert to integer
    pub fn as_i32(&self) -> i32 {
        *self as i32
    }
}

/// Operator options (type-specific parameters)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OperatorOptions {
    None,
    FullyConnected(FullyConnectedOptions),
    Conv2D(Conv2DOptions),
    Pool2D(Pool2DOptions),
    Activation(ActivationOptions),
    Reshape(ReshapeOptions),
    Custom(HashMap<String, OptionValue>),
}

/// Fully connected layer options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullyConnectedOptions {
    pub fused_activation: ActivationType,
    pub keep_num_dims: bool,
}

impl Default for FullyConnectedOptions {
    fn default() -> Self {
        Self {
            fused_activation: ActivationType::None,
            keep_num_dims: false,
        }
    }
}

/// Convolution 2D options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conv2DOptions {
    pub padding: PaddingType,
    pub stride_w: i32,
    pub stride_h: i32,
    pub dilation_w_factor: i32,
    pub dilation_h_factor: i32,
    pub fused_activation: ActivationType,
}

impl Default for Conv2DOptions {
    fn default() -> Self {
        Self {
            padding: PaddingType::Same,
            stride_w: 1,
            stride_h: 1,
            dilation_w_factor: 1,
            dilation_h_factor: 1,
            fused_activation: ActivationType::None,
        }
    }
}

/// Pooling layer options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pool2DOptions {
    pub padding: PaddingType,
    pub stride_w: i32,
    pub stride_h: i32,
    pub filter_width: i32,
    pub filter_height: i32,
    pub fused_activation: ActivationType,
}

impl Default for Pool2DOptions {
    fn default() -> Self {
        Self {
            padding: PaddingType::Same,
            stride_w: 2,
            stride_h: 2,
            filter_width: 2,
            filter_height: 2,
            fused_activation: ActivationType::None,
        }
    }
}

/// Activation function options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationOptions {
    pub activation_type: ActivationType,
}

/// Reshape options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReshapeOptions {
    pub new_shape: Vec<i32>,
}

/// Activation function types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationType {
    None = 0,
    Relu = 1,
    Relu6 = 3,
    Tanh = 4,
    Sigmoid = 5,
}

/// Padding types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaddingType {
    Same,
    Valid,
}

/// Option value (for custom operators)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptionValue {
    Int(i32),
    Float(f32),
    String(String),
    Bool(bool),
    IntArray(Vec<i32>),
    FloatArray(Vec<f32>),
}

impl OptionValue {
    /// Get as integer
    pub fn as_int(&self) -> Option<i32> {
        match self {
            OptionValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as float
    pub fn as_float(&self) -> Option<f32> {
        match self {
            OptionValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string
    pub fn as_string(&self) -> Option<&str> {
        match self {
            OptionValue::String(v) => Some(v),
            _ => None,
        }
    }
}

/// Operator registry for mapping SNN layers to TFLite operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorRegistry {
    mappings: HashMap<String, OperatorType>,
}

impl OperatorRegistry {
    /// Create a new operator registry
    pub fn new() -> Self {
        let mut registry = Self {
            mappings: HashMap::new(),
        };
        registry.register_default_mappings();
        registry
    }

    /// Register default SNN to TFLite operator mappings
    fn register_default_mappings(&mut self) {
        // Standard layers map to built-in operators
        self.register(
            "Linear",
            OperatorType::Builtin(BuiltinOperator::FullyConnected),
        );
        self.register("Conv2d", OperatorType::Builtin(BuiltinOperator::Conv2D));
        self.register(
            "AvgPool2d",
            OperatorType::Builtin(BuiltinOperator::AveragePool2D),
        );
        self.register(
            "MaxPool2d",
            OperatorType::Builtin(BuiltinOperator::MaxPool2D),
        );

        // Spiking layers map to custom operators
        self.register(
            "SpikingLinear",
            OperatorType::Custom(CustomOperator::LIFNeuron),
        );
        self.register("LIFLayer", OperatorType::Custom(CustomOperator::LIFNeuron));
        self.register(
            "AdaptiveLIF",
            OperatorType::Custom(CustomOperator::AdaptiveLIFNeuron),
        );
        self.register(
            "Izhikevich",
            OperatorType::Custom(CustomOperator::IzhikevichNeuron),
        );
    }

    /// Register a custom mapping
    pub fn register(&mut self, layer_type: &str, op_type: OperatorType) {
        self.mappings.insert(layer_type.to_string(), op_type);
    }

    /// Get operator type for layer
    pub fn get_operator(&self, layer_type: &str) -> Option<&OperatorType> {
        self.mappings.get(layer_type)
    }

    /// Check if layer type is registered
    pub fn is_registered(&self, layer_type: &str) -> bool {
        self.mappings.contains_key(layer_type)
    }
}

impl Default for OperatorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_operator_creation() {
        let op = TFLiteOperator::new(
            OperatorType::Builtin(BuiltinOperator::FullyConnected),
            vec![0],
            vec![1],
        );

        assert_eq!(op.inputs, vec![0]);
        assert_eq!(op.outputs, vec![1]);
    }

    #[test]
    fn test_operator_validation() {
        let valid_op = TFLiteOperator::new(
            OperatorType::Builtin(BuiltinOperator::Add),
            vec![0, 1],
            vec![2],
        );
        assert!(valid_op.validate().is_ok());

        let invalid_op =
            TFLiteOperator::new(OperatorType::Builtin(BuiltinOperator::Add), vec![], vec![2]);
        assert!(invalid_op.validate().is_err());
    }

    #[test]
    fn test_builtin_operator_codes() {
        assert_eq!(BuiltinOperator::Add.to_code(), 0);
        assert_eq!(BuiltinOperator::FullyConnected.to_code(), 9);
        assert_eq!(BuiltinOperator::Softmax.to_code(), 25);
    }

    #[test]
    fn test_builtin_operator_quantization_support() {
        assert!(BuiltinOperator::FullyConnected.supports_quantization());
        assert!(BuiltinOperator::Conv2D.supports_quantization());
        assert!(!BuiltinOperator::Softmax.supports_quantization());
    }

    #[test]
    fn test_custom_operator_names() {
        assert_eq!(CustomOperator::LIFNeuron.name(), "LIF_NEURON");
        assert_eq!(CustomOperator::SpikeEncoder.name(), "SPIKE_ENCODER");
    }

    #[test]
    fn test_custom_operator_params() {
        let params = CustomOperator::LIFNeuron.required_params();
        assert!(params.contains(&"tau_mem"));
        assert!(params.contains(&"v_threshold"));
    }

    #[test]
    fn test_operator_type_version() {
        let builtin = OperatorType::Builtin(BuiltinOperator::Add);
        assert_eq!(builtin.version(), OperatorVersion::V1);

        let custom = OperatorType::Custom(CustomOperator::LIFNeuron);
        assert_eq!(custom.version(), OperatorVersion::V1);
    }

    #[test]
    fn test_operator_type_is_custom() {
        let builtin = OperatorType::Builtin(BuiltinOperator::Add);
        assert!(!builtin.is_custom());

        let custom = OperatorType::Custom(CustomOperator::LIFNeuron);
        assert!(custom.is_custom());
    }

    #[test]
    fn test_operator_registry() {
        let registry = OperatorRegistry::new();

        assert!(registry.is_registered("Linear"));
        assert!(registry.is_registered("SpikingLinear"));
        assert!(!registry.is_registered("UnknownLayer"));
    }

    #[test]
    fn test_operator_registry_get() {
        let registry = OperatorRegistry::new();

        let op = registry.get_operator("Linear");
        assert!(op.is_some());
        assert!(matches!(
            op.unwrap(),
            OperatorType::Builtin(BuiltinOperator::FullyConnected)
        ));

        let custom_op = registry.get_operator("SpikingLinear");
        assert!(custom_op.is_some());
        assert!(matches!(
            custom_op.unwrap(),
            OperatorType::Custom(CustomOperator::LIFNeuron)
        ));
    }

    #[test]
    fn test_operator_registry_custom_registration() {
        let mut registry = OperatorRegistry::new();

        registry.register(
            "MyCustomLayer",
            OperatorType::Builtin(BuiltinOperator::Relu),
        );

        assert!(registry.is_registered("MyCustomLayer"));
    }

    #[test]
    fn test_option_value_accessors() {
        let int_val = OptionValue::Int(42);
        assert_eq!(int_val.as_int(), Some(42));
        assert_eq!(int_val.as_float(), None);

        let float_val = OptionValue::Float(2.5);
        assert_eq!(float_val.as_float(), Some(2.5));
        assert_eq!(float_val.as_int(), None);

        let string_val = OptionValue::String("test".to_string());
        assert_eq!(string_val.as_string(), Some("test"));
    }

    #[test]
    fn test_fully_connected_options_default() {
        let opts = FullyConnectedOptions::default();
        assert_eq!(opts.fused_activation, ActivationType::None);
        assert!(!opts.keep_num_dims);
    }

    #[test]
    fn test_conv2d_options_default() {
        let opts = Conv2DOptions::default();
        assert_eq!(opts.padding, PaddingType::Same);
        assert_eq!(opts.stride_w, 1);
        assert_eq!(opts.stride_h, 1);
        assert_eq!(opts.fused_activation, ActivationType::None);
    }

    #[test]
    fn test_pool2d_options_default() {
        let opts = Pool2DOptions::default();
        assert_eq!(opts.padding, PaddingType::Same);
        assert_eq!(opts.stride_w, 2);
        assert_eq!(opts.filter_width, 2);
    }
    /// Every custom operator must validate straight out of `new()`.
    ///
    /// Regression: the exporter built SNN layers as custom ops without setting
    /// `custom_code`, so validation rejected every exported model containing a
    /// spiking layer — the exact case this exporter exists for.
    #[test]
    fn test_custom_operators_carry_registration_name() {
        let all = [
            CustomOperator::LIFNeuron,
            CustomOperator::AdaptiveLIFNeuron,
            CustomOperator::IzhikevichNeuron,
            CustomOperator::SpikeEncoder,
            CustomOperator::SpikeDecoder,
            CustomOperator::SpikePooling,
            CustomOperator::TemporalConv,
        ];
        for op in all {
            let expected = op.name();
            let operator = TFLiteOperator::new(OperatorType::Custom(op), vec![0], vec![1]);
            assert_eq!(
                operator.custom_code.as_deref(),
                Some(expected),
                "{expected} must carry its own registration name"
            );
            assert!(
                operator.validate().is_ok(),
                "{expected} must validate without extra setup"
            );
        }

        // A builtin has no custom code, and must not acquire one.
        let builtin = TFLiteOperator::new(
            OperatorType::Builtin(BuiltinOperator::Add),
            vec![0],
            vec![1],
        );
        assert!(builtin.custom_code.is_none());
        assert!(builtin.validate().is_ok());
    }
}
