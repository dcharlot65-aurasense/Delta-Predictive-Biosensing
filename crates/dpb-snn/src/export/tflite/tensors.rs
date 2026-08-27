//! TensorFlow Lite tensor handling
//!
//! Defines tensor types, shapes, and quantization parameters for TFLite models.

use serde::{Deserialize, Serialize};

/// TensorFlow Lite tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFLiteTensor {
    /// Tensor name
    pub name: String,
    /// Tensor shape (dimensions)
    pub shape: TensorShape,
    /// Data type
    pub data_type: TensorType,
    /// Quantization parameters (if quantized)
    pub quantization: Option<QuantizationParams>,
    /// Buffer index (for weights/constants)
    pub buffer_index: Option<usize>,
    /// Whether this is a variable tensor
    pub is_variable: bool,
}

impl TFLiteTensor {
    /// Create a new tensor
    pub fn new(name: String, shape: TensorShape, data_type: TensorType) -> Self {
        Self {
            name,
            shape,
            data_type,
            quantization: None,
            buffer_index: None,
            is_variable: false,
        }
    }

    /// Set quantization parameters
    pub fn with_quantization(mut self, params: QuantizationParams) -> Self {
        self.quantization = Some(params);
        self
    }

    /// Set buffer index
    pub fn with_buffer(mut self, index: usize) -> Self {
        self.buffer_index = Some(index);
        self
    }

    /// Mark as variable
    /// Marks this tensor as a variable. Consumes and returns `self`, so it
    /// chains like the other builder methods.
    pub fn into_variable(mut self) -> Self {
        self.is_variable = true;
        self
    }

    /// Get total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.num_elements()
    }

    /// Get byte size of tensor data
    pub fn byte_size(&self) -> usize {
        self.num_elements() * self.data_type.size_bytes()
    }

    /// Check if tensor is quantized
    pub fn is_quantized(&self) -> bool {
        self.quantization.is_some()
    }

    /// Validate tensor configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate shape
        if self.shape.is_empty() {
            return Err(format!("Tensor '{}' has empty shape", self.name));
        }

        // Validate quantization matches data type
        if let Some(ref quant) = self.quantization {
            if !self.data_type.supports_quantization() {
                return Err(format!(
                    "Tensor '{}' has quantization params but type {:?} doesn't support quantization",
                    self.name, self.data_type
                ));
            }

            quant.validate()?;
        }

        Ok(())
    }
}

/// Tensor shape
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorShape {
    /// Dimensions (e.g., [batch, height, width, channels])
    pub dims: Vec<i32>,
}

impl TensorShape {
    /// Create new shape
    pub fn new(dims: Vec<i32>) -> Self {
        Self { dims }
    }

    /// Create from usize dimensions
    pub fn from_usize(dims: Vec<usize>) -> Self {
        Self {
            dims: dims.iter().map(|&d| d as i32).collect(),
        }
    }

    /// Get rank (number of dimensions)
    pub fn rank(&self) -> usize {
        self.dims.len()
    }

    /// Check if shape is empty
    pub fn is_empty(&self) -> bool {
        self.dims.is_empty()
    }

    /// Get total number of elements
    pub fn num_elements(&self) -> usize {
        if self.dims.is_empty() {
            return 0;
        }

        self.dims
            .iter()
            .map(|&d| if d < 0 { 1 } else { d as usize })
            .product()
    }

    /// Check if shape contains dynamic dimensions
    pub fn is_dynamic(&self) -> bool {
        self.dims.iter().any(|&d| d < 0)
    }

    /// Get static shape (replace -1 with 1)
    pub fn to_static(&self) -> Self {
        Self {
            dims: self
                .dims
                .iter()
                .map(|&d| if d < 0 { 1 } else { d })
                .collect(),
        }
    }
}

/// TensorFlow Lite data types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TensorType {
    /// 32-bit floating point
    Float32,
    /// 16-bit floating point
    Float16,
    /// 32-bit signed integer
    Int32,
    /// 16-bit signed integer
    Int16,
    /// 8-bit signed integer (quantized)
    Int8,
    /// 8-bit unsigned integer (quantized)
    UInt8,
    /// 64-bit signed integer
    Int64,
    /// String
    String,
    /// Boolean
    Bool,
    /// Complex64
    Complex64,
}

impl TensorType {
    /// Get size in bytes
    pub fn size_bytes(&self) -> usize {
        match self {
            TensorType::Float32 => 4,
            TensorType::Float16 => 2,
            TensorType::Int32 => 4,
            TensorType::Int16 => 2,
            TensorType::Int8 => 1,
            TensorType::UInt8 => 1,
            TensorType::Int64 => 8,
            TensorType::String => 1, // Variable
            TensorType::Bool => 1,
            TensorType::Complex64 => 8,
        }
    }

    /// Check if type supports quantization
    pub fn supports_quantization(&self) -> bool {
        matches!(
            self,
            TensorType::Int8 | TensorType::UInt8 | TensorType::Int16
        )
    }

    /// Check if type is floating point
    pub fn is_float(&self) -> bool {
        matches!(self, TensorType::Float32 | TensorType::Float16)
    }

    /// Check if type is integer
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            TensorType::Int32
                | TensorType::Int16
                | TensorType::Int8
                | TensorType::UInt8
                | TensorType::Int64
        )
    }

    /// Get TFLite type code
    pub fn to_tflite_code(self) -> u8 {
        match self {
            TensorType::Float32 => 0,
            TensorType::Float16 => 1,
            TensorType::Int32 => 2,
            TensorType::UInt8 => 3,
            TensorType::Int64 => 4,
            TensorType::String => 5,
            TensorType::Bool => 6,
            TensorType::Int16 => 7,
            TensorType::Complex64 => 8,
            TensorType::Int8 => 9,
        }
    }

    /// Create from TFLite type code
    pub fn from_tflite_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(TensorType::Float32),
            1 => Some(TensorType::Float16),
            2 => Some(TensorType::Int32),
            3 => Some(TensorType::UInt8),
            4 => Some(TensorType::Int64),
            5 => Some(TensorType::String),
            6 => Some(TensorType::Bool),
            7 => Some(TensorType::Int16),
            8 => Some(TensorType::Complex64),
            9 => Some(TensorType::Int8),
            _ => None,
        }
    }
}

/// Quantization parameters for tensors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationParams {
    /// Scale factors (one per channel for per-channel quantization)
    pub scales: Vec<f32>,
    /// Zero points (one per channel for per-channel quantization)
    pub zero_points: Vec<i32>,
    /// Quantization dimension (for per-channel quantization)
    pub quantized_dimension: Option<usize>,
    /// Min/max values (for debugging)
    pub min: Option<f32>,
    pub max: Option<f32>,
}

impl QuantizationParams {
    /// Create per-tensor quantization parameters
    pub fn per_tensor(scale: f32, zero_point: i32) -> Self {
        Self {
            scales: vec![scale],
            zero_points: vec![zero_point],
            quantized_dimension: None,
            min: None,
            max: None,
        }
    }

    /// Create per-channel quantization parameters
    pub fn per_channel(
        scales: Vec<f32>,
        zero_points: Vec<i32>,
        quantized_dimension: usize,
    ) -> Self {
        Self {
            scales,
            zero_points,
            quantized_dimension: Some(quantized_dimension),
            min: None,
            max: None,
        }
    }

    /// Add min/max values
    pub fn with_min_max(mut self, min: f32, max: f32) -> Self {
        self.min = Some(min);
        self.max = Some(max);
        self
    }

    /// Check if per-channel quantization
    pub fn is_per_channel(&self) -> bool {
        self.quantized_dimension.is_some()
    }

    /// Get number of channels
    pub fn num_channels(&self) -> usize {
        self.scales.len()
    }

    /// Validate quantization parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.scales.is_empty() {
            return Err("Quantization scales cannot be empty".to_string());
        }

        if self.scales.len() != self.zero_points.len() {
            return Err(format!(
                "Scale count ({}) must match zero point count ({})",
                self.scales.len(),
                self.zero_points.len()
            ));
        }

        // Check for zero or negative scales
        if self.scales.iter().any(|&s| s <= 0.0) {
            return Err("Quantization scales must be positive".to_string());
        }

        // Check for NaN or infinite scales
        if self.scales.iter().any(|&s| !s.is_finite()) {
            return Err("Quantization scales must be finite".to_string());
        }

        Ok(())
    }

    /// Quantize a floating point value
    pub fn quantize_value(&self, value: f32, channel: usize) -> i32 {
        let channel_idx = if self.is_per_channel() {
            channel.min(self.scales.len() - 1)
        } else {
            0
        };

        let scale = self.scales[channel_idx];
        let zero_point = self.zero_points[channel_idx];

        ((value / scale).round() as i32 + zero_point).clamp(-128, 127)
    }

    /// Dequantize a quantized value
    pub fn dequantize_value(&self, quantized: i32, channel: usize) -> f32 {
        let channel_idx = if self.is_per_channel() {
            channel.min(self.scales.len() - 1)
        } else {
            0
        };

        let scale = self.scales[channel_idx];
        let zero_point = self.zero_points[channel_idx];

        (quantized - zero_point) as f32 * scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let tensor = TFLiteTensor::new(
            "input".to_string(),
            TensorShape::new(vec![1, 224, 224, 3]),
            TensorType::Float32,
        );

        assert_eq!(tensor.name, "input");
        assert_eq!(tensor.shape.dims, vec![1, 224, 224, 3]);
        assert_eq!(tensor.data_type, TensorType::Float32);
        assert!(!tensor.is_quantized());
    }

    #[test]
    fn test_tensor_with_quantization() {
        let quant = QuantizationParams::per_tensor(0.5, 128);
        let tensor = TFLiteTensor::new(
            "weight".to_string(),
            TensorShape::new(vec![64, 128]),
            TensorType::Int8,
        )
        .with_quantization(quant);

        assert!(tensor.is_quantized());
    }

    #[test]
    fn test_tensor_num_elements() {
        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![2, 3, 4]),
            TensorType::Float32,
        );

        assert_eq!(tensor.num_elements(), 24);
    }

    #[test]
    fn test_tensor_byte_size() {
        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![10]),
            TensorType::Float32,
        );

        assert_eq!(tensor.byte_size(), 40); // 10 * 4 bytes
    }

    #[test]
    fn test_tensor_validation_empty_shape() {
        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![]),
            TensorType::Float32,
        );

        assert!(tensor.validate().is_err());
    }

    #[test]
    fn test_tensor_validation_invalid_quantization() {
        let quant = QuantizationParams::per_tensor(0.5, 128);
        let tensor = TFLiteTensor::new(
            "test".to_string(),
            TensorShape::new(vec![10]),
            TensorType::Float32, // Float doesn't support quantization
        )
        .with_quantization(quant);

        assert!(tensor.validate().is_err());
    }

    #[test]
    fn test_shape_rank() {
        let shape = TensorShape::new(vec![1, 2, 3, 4]);
        assert_eq!(shape.rank(), 4);
    }

    #[test]
    fn test_shape_num_elements() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.num_elements(), 24);
    }

    #[test]
    fn test_shape_dynamic() {
        let static_shape = TensorShape::new(vec![1, 2, 3]);
        assert!(!static_shape.is_dynamic());

        let dynamic_shape = TensorShape::new(vec![-1, 2, 3]);
        assert!(dynamic_shape.is_dynamic());
    }

    #[test]
    fn test_shape_to_static() {
        let dynamic_shape = TensorShape::new(vec![-1, 2, -1, 4]);
        let static_shape = dynamic_shape.to_static();

        assert_eq!(static_shape.dims, vec![1, 2, 1, 4]);
        assert!(!static_shape.is_dynamic());
    }

    #[test]
    fn test_tensor_type_size() {
        assert_eq!(TensorType::Float32.size_bytes(), 4);
        assert_eq!(TensorType::Float16.size_bytes(), 2);
        assert_eq!(TensorType::Int8.size_bytes(), 1);
        assert_eq!(TensorType::Int64.size_bytes(), 8);
    }

    #[test]
    fn test_tensor_type_properties() {
        assert!(TensorType::Float32.is_float());
        assert!(!TensorType::Int32.is_float());

        assert!(TensorType::Int32.is_integer());
        assert!(!TensorType::Float32.is_integer());

        assert!(TensorType::Int8.supports_quantization());
        assert!(!TensorType::Float32.supports_quantization());
    }

    #[test]
    fn test_tensor_type_codes() {
        assert_eq!(TensorType::Float32.to_tflite_code(), 0);
        assert_eq!(TensorType::Int8.to_tflite_code(), 9);

        assert_eq!(TensorType::from_tflite_code(0), Some(TensorType::Float32));
        assert_eq!(TensorType::from_tflite_code(9), Some(TensorType::Int8));
        assert_eq!(TensorType::from_tflite_code(99), None);
    }

    #[test]
    fn test_quantization_per_tensor() {
        let quant = QuantizationParams::per_tensor(0.5, 0);

        assert!(!quant.is_per_channel());
        assert_eq!(quant.num_channels(), 1);
        assert!(quant.validate().is_ok());
    }

    #[test]
    fn test_quantization_per_channel() {
        let quant = QuantizationParams::per_channel(vec![0.1, 0.2, 0.3], vec![0, 0, 0], 0);

        assert!(quant.is_per_channel());
        assert_eq!(quant.num_channels(), 3);
        assert!(quant.validate().is_ok());
    }

    #[test]
    fn test_quantization_validation_empty() {
        let quant = QuantizationParams::per_channel(vec![], vec![], 0);
        assert!(quant.validate().is_err());
    }

    #[test]
    fn test_quantization_validation_mismatch() {
        let quant = QuantizationParams::per_channel(vec![0.1, 0.2], vec![0], 0);
        assert!(quant.validate().is_err());
    }

    #[test]
    fn test_quantization_validation_invalid_scale() {
        let mut quant = QuantizationParams::per_tensor(0.5, 0);
        quant.scales[0] = 0.0;
        assert!(quant.validate().is_err());

        quant.scales[0] = -0.1;
        assert!(quant.validate().is_err());

        quant.scales[0] = f32::NAN;
        assert!(quant.validate().is_err());
    }

    #[test]
    fn test_quantize_dequantize() {
        let quant = QuantizationParams::per_tensor(0.1, 0);

        let value = 1.23;
        let quantized = quant.quantize_value(value, 0);
        let dequantized = quant.dequantize_value(quantized, 0);

        assert!((value - dequantized).abs() < 0.1); // Within quantization error
    }

    #[test]
    fn test_quantize_per_channel() {
        let quant = QuantizationParams::per_channel(vec![0.1, 0.2, 0.3], vec![0, 0, 0], 0);

        let value = 1.0;
        let q0 = quant.quantize_value(value, 0);
        let q1 = quant.quantize_value(value, 1);
        let q2 = quant.quantize_value(value, 2);

        // Different scales should produce different quantized values
        assert_ne!(q0, q1);
        assert_ne!(q1, q2);
    }
}
