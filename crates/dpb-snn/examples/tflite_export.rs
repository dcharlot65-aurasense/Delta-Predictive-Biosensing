//! Example: Exporting SNN models to TensorFlow Lite format
//!
//! This example demonstrates how to export a spiking neural network
//! to TensorFlow Lite format with various quantization options.

use dpb_snn::export::{
    TFLiteExporter, TFLiteConfig, QuantizationConfig, QuantizationMode,
    ModelWeights, LayerConfig, LayerType,
    weights::LayerWeights,
};

fn main() -> Result<(), String> {
    println!("=== TensorFlow Lite Export Examples ===\n");

    // Example 1: Basic float32 export
    println!("1. Basic Float32 Export");
    export_basic_model()?;

    // Example 2: Int8 quantized export
    println!("\n2. Int8 Quantized Export");
    export_quantized_int8()?;

    // Example 3: Dynamic range quantization
    println!("\n3. Dynamic Range Quantization");
    export_dynamic_quantization()?;

    // Example 4: Multi-layer network
    println!("\n4. Multi-Layer Network Export");
    export_multi_layer()?;

    println!("\n=== All exports completed successfully! ===");

    Ok(())
}

/// Export a basic SNN model with Float32 weights
fn export_basic_model() -> Result<(), String> {
    // Create model weights
    let mut weights = ModelWeights::new("basic_snn".to_string());

    // Single layer: 128 inputs -> 64 outputs
    let layer = LayerWeights::new(
        "fc1".to_string(),
        "SpikingLinear".to_string(),
        (0..8192).map(|i| ((i as f64) * 0.001).sin()).collect(), // 128 * 64
        vec![128, 64],
    ).with_bias((0..64).map(|i| (i as f64) * 0.01).collect());

    weights.add_layer(layer);
    weights.update_checksum();

    // Layer configuration
    let layers = vec![
        LayerConfig::new(
            "fc1".to_string(),
            LayerType::SpikingLinear,
            128,
            64,
        ),
    ];

    // Create exporter
    let config = TFLiteConfig::new("basic_snn".to_string())
        .with_inputs(vec!["input".to_string()])
        .with_outputs(vec!["output".to_string()]);

    let exporter = TFLiteExporter::new(config);

    // Export model
    let input_shape = vec![128];
    let result = exporter.export_model(&weights, &layers, &input_shape)?;

    println!("  ✓ Model exported");
    println!("  - Size: {:.2} MB", result.size_mb());
    println!("  - Format: {}", result.metadata.export_format);
    println!("  - Quantized: {}", result.metadata.quantized);
    println!("  - Warnings: {}", result.warnings.len());

    // Save to file
    result.save_to_file("basic_snn.tflite")?;
    println!("  - Saved to: basic_snn.tflite");

    Ok(())
}

/// Export with Int8 quantization
fn export_quantized_int8() -> Result<(), String> {
    let mut weights = ModelWeights::new("quantized_snn".to_string());

    // Create larger weights for better quantization demonstration
    let layer = LayerWeights::new(
        "fc1".to_string(),
        "SpikingLinear".to_string(),
        (0..25600).map(|i| ((i as f64) * 0.001).cos()).collect(), // 256 * 100
        vec![256, 100],
    ).with_bias((0..100).map(|i| (i as f64) * 0.001).collect());

    weights.add_layer(layer);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new(
            "fc1".to_string(),
            LayerType::SpikingLinear,
            256,
            100,
        ),
    ];

    // Configure Int8 quantization
    let quant_config = QuantizationConfig::int8();
    let config = TFLiteConfig::new("quantized_snn".to_string())
        .with_quantization(quant_config);

    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![256];
    let result = exporter.export_model(&weights, &layers, &input_shape)?;

    println!("  ✓ Quantized model exported");
    println!("  - Size: {:.2} MB", result.size_mb());
    println!("  - Quantized: {}", result.metadata.quantized);
    println!("  - Total parameters: {}", result.metadata.total_params);

    result.save_to_file("quantized_snn.tflite")?;
    println!("  - Saved to: quantized_snn.tflite");

    Ok(())
}

/// Export with dynamic range quantization
fn export_dynamic_quantization() -> Result<(), String> {
    let mut weights = ModelWeights::new("dynamic_snn".to_string());

    let layer = LayerWeights::new(
        "fc1".to_string(),
        "SpikingLinear".to_string(),
        (0..12800).map(|i| ((i as f64) * 0.002).tanh()).collect(), // 128 * 100
        vec![128, 100],
    ).with_bias((0..100).map(|i| (i as f64) * 0.001).collect());

    weights.add_layer(layer);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new(
            "fc1".to_string(),
            LayerType::SpikingLinear,
            128,
            100,
        ),
    ];

    // Configure dynamic range quantization
    let quant_config = QuantizationConfig::dynamic();
    let config = TFLiteConfig::new("dynamic_snn".to_string())
        .with_quantization(quant_config);

    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![128];
    let result = exporter.export_model(&weights, &layers, &input_shape)?;

    println!("  ✓ Dynamic quantized model exported");
    println!("  - Size: {:.2} MB", result.size_mb());
    println!("  - Warnings: {}", result.warnings.len());

    if let Some(ref validation) = result.validation {
        println!("  - Validation: {}", if validation.is_valid() { "PASSED" } else { "FAILED" });
        if validation.has_warnings() {
            println!("    - Warnings: {}", validation.warnings.len());
        }
    }

    result.save_to_file("dynamic_snn.tflite")?;
    println!("  - Saved to: dynamic_snn.tflite");

    Ok(())
}

/// Export a multi-layer network
fn export_multi_layer() -> Result<(), String> {
    let mut weights = ModelWeights::new("multi_layer_snn".to_string());

    // Layer 1: 784 (28x28 image) -> 256
    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "SpikingLinear".to_string(),
        (0..200704).map(|i| ((i as f64) * 0.0001).sin() * 0.5).collect(),
        vec![784, 256],
    ).with_bias((0..256).map(|i| (i as f64) * 0.001).collect());

    // Layer 2: 256 -> 128
    let layer2 = LayerWeights::new(
        "fc2".to_string(),
        "SpikingLinear".to_string(),
        (0..32768).map(|i| ((i as f64) * 0.001).cos() * 0.3).collect(),
        vec![256, 128],
    ).with_bias((0..128).map(|i| (i as f64) * 0.001).collect());

    // Layer 3: 128 -> 10 (classification)
    let layer3 = LayerWeights::new(
        "fc3".to_string(),
        "SpikingLinear".to_string(),
        (0..1280).map(|i| ((i as f64) * 0.01).tanh() * 0.2).collect(),
        vec![128, 10],
    ).with_bias((0..10).map(|i| (i as f64) * 0.01).collect());

    weights.add_layer(layer1);
    weights.add_layer(layer2);
    weights.add_layer(layer3);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new("fc1".to_string(), LayerType::SpikingLinear, 784, 256),
        LayerConfig::new("fc2".to_string(), LayerType::SpikingLinear, 256, 128),
        LayerConfig::new("fc3".to_string(), LayerType::SpikingLinear, 128, 10),
    ];

    // Export with Int8 quantization for smaller size
    let quant_config = QuantizationConfig::int8();
    let config = TFLiteConfig::new("multi_layer_snn".to_string())
        .with_quantization(quant_config)
        .with_inputs(vec!["image".to_string()])
        .with_outputs(vec!["logits".to_string()]);

    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![784];
    let result = exporter.export_model(&weights, &layers, &input_shape)?;

    println!("  ✓ Multi-layer network exported");
    println!("  - Layers: {}", layers.len());
    println!("  - Total parameters: {}", result.metadata.total_params);
    println!("  - Size: {:.2} MB", result.size_mb());
    println!("  - Input shape: {:?}", result.metadata.input_shape);
    println!("  - Output shape: {:?}", result.metadata.output_shape);

    if let Some(ref validation) = result.validation {
        println!("\n  Validation Results:");
        println!("    - Status: {}", if validation.is_valid() { "✓ PASSED" } else { "✗ FAILED" });
        println!("    - Errors: {}", validation.errors.len());
        println!("    - Warnings: {}", validation.warnings.len());
        println!("    - Hints: {}", validation.hints.len());

        if !validation.warnings.is_empty() {
            println!("\n  Warnings:");
            for warning in validation.warnings.iter().take(3) {
                println!("    - {}", warning.description());
            }
            if validation.warnings.len() > 3 {
                println!("    ... and {} more", validation.warnings.len() - 3);
            }
        }

        if !validation.hints.is_empty() {
            println!("\n  Optimization Hints:");
            for hint in validation.hints.iter().take(3) {
                println!("    - {}", hint);
            }
            if validation.hints.len() > 3 {
                println!("    ... and {} more", validation.hints.len() - 3);
            }
        }
    }

    result.save_to_file("multi_layer_snn.tflite")?;
    println!("\n  - Saved to: multi_layer_snn.tflite");

    Ok(())
}
