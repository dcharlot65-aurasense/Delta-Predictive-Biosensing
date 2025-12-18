# TensorFlow Lite Export Module

This module provides comprehensive TensorFlow Lite export functionality for spiking neural networks (SNNs) in the Delta-Predictive-Biosensing framework.

## Overview

TensorFlow Lite (TFLite) is an optimized framework for deploying machine learning models on mobile and embedded devices. This module enables exporting trained SNNs to TFLite format with support for:

- **Model conversion**: Convert SNN architectures to TFLite-compatible format
- **Quantization**: Post-training quantization (Int8, Float16, dynamic range)
- **Custom operators**: Support for SNN-specific operations (LIF neurons, spike encoding/decoding)
- **Metadata**: Embedded model metadata for deployment
- **Validation**: Comprehensive model validation and compatibility checking

## Features

### Supported Quantization Modes

1. **Float32** (default): Full precision weights and activations
2. **Int8**: 8-bit integer quantization for weights and activations
3. **Float16**: 16-bit floating point quantization
4. **Dynamic Range**: Quantize weights only, keep activations in float

### Custom Operators

The module supports SNN-specific custom operators:

- `LIF_NEURON`: Leaky Integrate-and-Fire neuron dynamics
- `ADAPTIVE_LIF_NEURON`: Adaptive LIF with threshold adaptation
- `IZHIKEVICH_NEURON`: Izhikevich neuron model
- `SPIKE_ENCODER`: Spike encoding (rate, latency, population)
- `SPIKE_DECODER`: Spike decoding to output values
- `SPIKE_POOLING`: Temporal pooling over spike trains
- `TEMPORAL_CONV`: Temporal convolution with spike timing

## Usage

### Basic Export

```rust
use dpb_snn::export::{TFLiteExporter, TFLiteConfig, ModelWeights, LayerConfig};

// Create exporter with configuration
let config = TFLiteConfig::new("my_snn_model".to_string())
    .with_inputs(vec!["input".to_string()])
    .with_outputs(vec!["output".to_string()]);

let exporter = TFLiteExporter::new(config);

// Export model
let result = exporter.export_model(&weights, &layer_configs, &input_shape)?;

// Save to file
result.save_to_file("model.tflite")?;

println!("Model size: {:.2} MB", result.size_mb());
```

### Quantized Export

```rust
use dpb_snn::export::{TFLiteConfig, QuantizationConfig};

// Int8 quantization
let quant_config = QuantizationConfig::int8();
let config = TFLiteConfig::new("quantized_model".to_string())
    .with_quantization(quant_config);

let exporter = TFLiteExporter::new(config);
let result = exporter.export_model(&weights, &layer_configs, &input_shape)?;
```

### Dynamic Range Quantization

```rust
// Quantize weights only, keep activations in float
let quant_config = QuantizationConfig::dynamic();
let config = TFLiteConfig::new("dynamic_model".to_string())
    .with_quantization(quant_config);
```

### Validation

```rust
let config = TFLiteConfig::new("model".to_string());
let exporter = TFLiteExporter::new(config);
let result = exporter.export_model(&weights, &layer_configs, &input_shape)?;

if let Some(validation) = result.validation {
    println!("Validation: {}", if validation.is_valid() { "PASSED" } else { "FAILED" });
    println!("Warnings: {}", validation.warnings.len());
    println!("Hints: {}", validation.hints.len());

    // Print validation summary
    println!("{}", validation.summary());
}
```

### Disable Validation or Metadata

```rust
let config = TFLiteConfig::new("model".to_string())
    .without_validation()
    .without_metadata();
```

## Module Structure

### Core Modules

- **`exporter.rs`**: Main TFLite exporter implementation
- **`operators.rs`**: TFLite operator definitions (built-in and custom)
- **`tensors.rs`**: Tensor types, shapes, and quantization parameters
- **`quantization.rs`**: Quantization strategies and post-training quantization
- **`flatbuffer.rs`**: FlatBuffer serialization for TFLite format
- **`metadata.rs`**: Model metadata and descriptions
- **`validation.rs`**: Model validation and compatibility checking

### Key Types

#### `TFLiteExporter`

Main exporter class that orchestrates the export process.

```rust
pub struct TFLiteExporter {
    config: TFLiteConfig,
}
```

#### `TFLiteConfig`

Configuration for TFLite export.

```rust
pub struct TFLiteConfig {
    pub model_name: String,
    pub quantization: Option<QuantizationConfig>,
    pub input_names: Vec<String>,
    pub output_names: Vec<String>,
    pub include_metadata: bool,
    pub validate: bool,
}
```

#### `TFLiteExportResult`

Result of export operation.

```rust
pub struct TFLiteExportResult {
    pub model_bytes: Vec<u8>,
    pub metadata: ExportMetadata,
    pub validation: Option<ValidationResult>,
    pub warnings: Vec<String>,
}
```

## Quantization Details

### Per-Tensor Quantization

Quantizes entire tensor with single scale and zero-point:

```
quantized_value = round(real_value / scale) + zero_point
```

### Per-Channel Quantization

Quantizes each output channel separately (better accuracy):

```
quantized_value[c] = round(real_value[c] / scale[c]) + zero_point[c]
```

### Calibration

For full integer quantization, calibration data is used to determine activation ranges:

```rust
let mut quantizer = PostTrainingQuantizer::new(config)?;

// Provide representative dataset for calibration
quantizer.calibrate(&calibration_dataset)?;

// Quantize weights and activations
let (quantized, params) = quantizer.quantize_weights(&weights, &shape)?;
```

## Validation

The validator checks for:

- **Structural correctness**: Valid operator and tensor configurations
- **Type compatibility**: Compatible data types between operators
- **Quantization support**: Operators support quantization if used
- **Custom operator requirements**: Custom ops have required parameters
- **Optimization opportunities**: Operator fusion, quantization recommendations

### Validation Warnings

- `UnsupportedOperator`: Operator may not be supported in target runtime
- `CustomOperator`: Custom operator requires runtime implementation
- `MixedPrecision`: Operator has mixed quantized/float inputs
- `LargeTensor`: Tensor size exceeds recommended limits
- `UnsupportedDataType`: Data type requires newer TFLite version

## Limitations

### Temporal Dynamics

SNNs have inherent temporal dynamics that cannot be fully represented in TFLite's static computation graph. The exporter approximates spiking behavior:

1. **Rate coding**: Spike rates are approximated as continuous values
2. **Time steps**: Temporal loops are unrolled or approximated
3. **State**: Neuron states are not preserved between inferences

### Custom Operators

Custom SNN operators require:

1. Implementation in TFLite runtime (C++)
2. Registration with TFLite interpreter
3. Proper parameter serialization

## Best Practices

1. **Use quantization for deployment**: Reduces model size and improves inference speed
2. **Validate models**: Always validate before deployment
3. **Check warnings**: Address compatibility warnings for target platform
4. **Test on device**: Verify model behavior on actual deployment hardware
5. **Calibrate properly**: Use representative data for quantization calibration

## Examples

See `examples/tflite_export.rs` for comprehensive examples:

```bash
cargo run --example tflite_export
```

## Testing

Run tests for the TFLite module:

```bash
cargo test --lib export::tflite
```

Run integration tests:

```bash
cargo test --test tflite_export
```

## References

- [TensorFlow Lite Documentation](https://www.tensorflow.org/lite)
- [TFLite Model Format](https://www.tensorflow.org/lite/guide/ops_compatibility)
- [Post-Training Quantization](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Custom Operators](https://www.tensorflow.org/lite/guide/ops_custom)

## Future Enhancements

- [ ] Support for LSTM/RNN operators
- [ ] Full Float16 quantization implementation
- [ ] Improved calibration with EMA statistics
- [ ] Model optimization (operator fusion, constant folding)
- [ ] Support for TFLite Flex delegates
- [ ] Signature definitions for serving
- [ ] Control flow operators (if/while)
