//! Integration tests for TensorFlow Lite export functionality

use dpb_snn::export::{
    TFLiteExporter, TFLiteConfig, QuantizationConfig,
    ModelWeights, LayerConfig, LayerType,
    weights::LayerWeights,
};

#[test]
fn test_basic_tflite_export() {
    // Create model weights
    let mut weights = ModelWeights::new("test_snn".to_string());

    // Add a simple linear layer
    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "Linear".to_string(),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        vec![2, 4], // 2x4 weight matrix
    ).with_bias(vec![0.01, 0.02]);

    weights.add_layer(layer1);
    weights.update_checksum();

    // Create layer configurations
    let layers = vec![
        LayerConfig::new(
            "fc1".to_string(),
            LayerType::SpikingLinear,
            4,
            2,
        ),
    ];

    // Create TFLite exporter
    let config = TFLiteConfig::new("test_snn".to_string())
        .with_inputs(vec!["input".to_string()])
        .with_outputs(vec!["output".to_string()]);

    let exporter = TFLiteExporter::new(config);

    // Export model
    let input_shape = vec![4];
    let result = exporter.export_model(&weights, &layers, &input_shape);

    assert!(result.is_ok(), "Export should succeed");

    let export = result.unwrap();
    assert!(!export.model_bytes.is_empty(), "Model bytes should not be empty");
    assert_eq!(export.metadata.export_format, "tflite");
    assert!(!export.metadata.quantized);
}

#[test]
fn test_quantized_tflite_export() {
    // Create model weights
    let mut weights = ModelWeights::new("test_snn_quantized".to_string());

    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "Linear".to_string(),
        vec![1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5],
        vec![2, 4],
    ).with_bias(vec![0.1, -0.1]);

    weights.add_layer(layer1);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new(
            "fc1".to_string(),
            LayerType::SpikingLinear,
            4,
            2,
        ),
    ];

    // Create quantized export configuration
    let quant_config = QuantizationConfig::int8();
    let config = TFLiteConfig::new("test_snn_quantized".to_string())
        .with_quantization(quant_config);

    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![4];
    let result = exporter.export_model(&weights, &layers, &input_shape);

    assert!(result.is_ok(), "Quantized export should succeed");

    let export = result.unwrap();
    assert!(!export.model_bytes.is_empty());
    assert!(export.metadata.quantized, "Model should be marked as quantized");
}

#[test]
fn test_multi_layer_export() {
    // Create model with multiple layers
    let mut weights = ModelWeights::new("multi_layer_snn".to_string());

    // Input layer: 10 -> 8
    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "Linear".to_string(),
        (0..80).map(|i| (i as f64) * 0.01).collect(),
        vec![10, 8],
    ).with_bias((0..8).map(|i| (i as f64) * 0.001).collect());

    // Hidden layer: 8 -> 5
    let layer2 = LayerWeights::new(
        "fc2".to_string(),
        "Linear".to_string(),
        (0..40).map(|i| (i as f64) * 0.01).collect(),
        vec![8, 5],
    ).with_bias((0..5).map(|i| (i as f64) * 0.001).collect());

    // Output layer: 5 -> 3
    let layer3 = LayerWeights::new(
        "fc3".to_string(),
        "Linear".to_string(),
        (0..15).map(|i| (i as f64) * 0.01).collect(),
        vec![5, 3],
    ).with_bias((0..3).map(|i| (i as f64) * 0.001).collect());

    weights.add_layer(layer1);
    weights.add_layer(layer2);
    weights.add_layer(layer3);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new("fc1".to_string(), LayerType::SpikingLinear, 10, 8),
        LayerConfig::new("fc2".to_string(), LayerType::SpikingLinear, 8, 5),
        LayerConfig::new("fc3".to_string(), LayerType::SpikingLinear, 5, 3),
    ];

    let config = TFLiteConfig::new("multi_layer_snn".to_string());
    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![10];
    let result = exporter.export_model(&weights, &layers, &input_shape);

    assert!(result.is_ok(), "Multi-layer export should succeed");

    let export = result.unwrap();
    assert!(!export.model_bytes.is_empty());
    assert_eq!(weights.layers.len(), 3);
}

#[test]
fn test_export_without_validation() {
    let mut weights = ModelWeights::new("test_no_validation".to_string());

    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "Linear".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
        vec![2, 2],
    );

    weights.add_layer(layer1);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new("fc1".to_string(), LayerType::SpikingLinear, 2, 2),
    ];

    let config = TFLiteConfig::new("test_no_validation".to_string())
        .without_validation();

    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![2];
    let result = exporter.export_model(&weights, &layers, &input_shape);

    assert!(result.is_ok());
    let export = result.unwrap();
    assert!(export.validation.is_none(), "Validation should be disabled");
}

#[test]
fn test_export_without_metadata() {
    let mut weights = ModelWeights::new("test_no_metadata".to_string());

    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "Linear".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
        vec![2, 2],
    );

    weights.add_layer(layer1);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new("fc1".to_string(), LayerType::SpikingLinear, 2, 2),
    ];

    let config = TFLiteConfig::new("test_no_metadata".to_string())
        .without_metadata();

    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![2];
    let result = exporter.export_model(&weights, &layers, &input_shape);

    assert!(result.is_ok());
}

#[test]
fn test_export_spiking_layer_warnings() {
    let mut weights = ModelWeights::new("test_spiking_warnings".to_string());

    let layer1 = LayerWeights::new(
        "lif1".to_string(),
        "SpikingLinear".to_string(),
        vec![0.1, 0.2, 0.3, 0.4],
        vec![2, 2],
    );

    weights.add_layer(layer1);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new("lif1".to_string(), LayerType::SpikingLinear, 2, 2),
    ];

    let config = TFLiteConfig::new("test_spiking_warnings".to_string());
    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![2];
    let result = exporter.export_model(&weights, &layers, &input_shape);

    assert!(result.is_ok());
    let export = result.unwrap();

    // Should have warnings about spiking dynamics
    assert!(
        !export.warnings.is_empty(),
        "Should have warnings about spiking layer approximation"
    );
}

#[test]
fn test_config_validation() {
    let config = TFLiteConfig::new("test".to_string());
    let exporter = TFLiteExporter::new(config);

    assert!(exporter.validate_config().is_ok());
}

#[test]
fn test_config_validation_empty_name() {
    let config = TFLiteConfig::new("".to_string());
    let exporter = TFLiteExporter::new(config);

    assert!(exporter.validate_config().is_err());
}

#[test]
fn test_export_result_size_calculation() {
    let mut weights = ModelWeights::new("test_size".to_string());

    let layer1 = LayerWeights::new(
        "fc1".to_string(),
        "Linear".to_string(),
        vec![0.1; 100],
        vec![10, 10],
    );

    weights.add_layer(layer1);
    weights.update_checksum();

    let layers = vec![
        LayerConfig::new("fc1".to_string(), LayerType::SpikingLinear, 10, 10),
    ];

    let config = TFLiteConfig::new("test_size".to_string());
    let exporter = TFLiteExporter::new(config);

    let input_shape = vec![10];
    let result = exporter.export_model(&weights, &layers, &input_shape).unwrap();

    assert!(result.size_bytes() > 0);
    assert!(result.size_mb() > 0.0);
}
