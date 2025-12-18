//! Integration tests for the export module

use dpb_snn::export::{
    OnnxExporter, OnnxConfig, WeightExporter, WeightFormat, ModelWeights,
    ModelConfig, LayerConfig, ExportMetadata,
};
use dpb_snn::export::config::LayerType;
use dpb_snn::export::weights::LayerWeights;

#[test]
fn test_complete_onnx_export_pipeline() {
    // Create model configuration
    let mut config = ModelConfig::new("test_model")
        .with_input_shape(vec![128])
        .with_output_shape(vec![10])
        .with_neuron_model("LIF".to_string())
        .with_time_steps(100);

    // Add layers
    config.add_layer(LayerConfig::new(
        "input_layer".to_string(),
        LayerType::SpikingLinear,
        128,
        64,
    ));

    config.add_layer(LayerConfig::new(
        "hidden_layer".to_string(),
        LayerType::SpikingLinear,
        64,
        32,
    ));

    config.add_layer(LayerConfig::new(
        "output_layer".to_string(),
        LayerType::SpikingLinear,
        32,
        10,
    ));

    // Validate configuration
    assert!(config.validate().is_ok());

    // Create model weights
    let mut weights = ModelWeights::new("test_model".to_string());

    // Layer 1 weights: 128 x 64 = 8192 parameters
    weights.add_layer(
        LayerWeights::new(
            "input_layer".to_string(),
            "SpikingLinear".to_string(),
            vec![0.1; 8192],
            vec![128, 64],
        ).with_bias(vec![0.01; 64]),
    );

    // Layer 2 weights: 64 x 32 = 2048 parameters
    weights.add_layer(
        LayerWeights::new(
            "hidden_layer".to_string(),
            "SpikingLinear".to_string(),
            vec![0.2; 2048],
            vec![64, 32],
        ).with_bias(vec![0.02; 32]),
    );

    // Layer 3 weights: 32 x 10 = 320 parameters
    weights.add_layer(
        LayerWeights::new(
            "output_layer".to_string(),
            "SpikingLinear".to_string(),
            vec![0.3; 320],
            vec![32, 10],
        ).with_bias(vec![0.03; 10]),
    );

    weights.update_checksum();

    // Create ONNX exporter
    let exporter = OnnxExporter::with_default_config();

    // Export model
    let result = exporter.export_snn_model(
        &weights,
        &config.layers,
        &config.input_shape,
    );

    assert!(result.is_ok());
    let export_result = result.unwrap();

    // Verify export result
    assert!(!export_result.model_bytes.is_empty());
    assert_eq!(export_result.metadata.model_name, "test_model");
    assert_eq!(export_result.metadata.export_format, "onnx");

    // Warnings should be present for spiking layers
    assert!(!export_result.warnings.is_empty());

    // Validate exported model
    assert!(exporter.validate(&export_result.model_bytes).is_ok());
}

#[test]
fn test_weight_serialization_formats() {
    // Create test weights
    let mut weights = ModelWeights::new("serialization_test".to_string());

    weights.add_layer(
        LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
        ).with_bias(vec![0.1, 0.2]),
    );

    weights.update_checksum();
    let original_checksum = weights.metadata.checksum.clone();

    // Test Binary format
    let binary_bytes = WeightExporter::export(&weights, WeightFormat::Binary).unwrap();
    assert!(!binary_bytes.is_empty());

    let loaded_binary = WeightExporter::load(&binary_bytes, WeightFormat::Binary).unwrap();
    assert_eq!(loaded_binary.metadata.model_name, weights.metadata.model_name);
    assert_eq!(loaded_binary.layers.len(), weights.layers.len());
    assert_eq!(loaded_binary.layers[0].weights, weights.layers[0].weights);

    // Test JSON format
    let json_bytes = WeightExporter::export(&weights, WeightFormat::Json).unwrap();
    assert!(!json_bytes.is_empty());

    let loaded_json = WeightExporter::load(&json_bytes, WeightFormat::Json).unwrap();
    assert_eq!(loaded_json.metadata.model_name, weights.metadata.model_name);
    assert_eq!(loaded_json.layers[0].weights, weights.layers[0].weights);

    // Verify checksum consistency
    let mut loaded_copy = loaded_json.clone();
    loaded_copy.update_checksum();
    assert_eq!(loaded_copy.metadata.checksum, original_checksum);
}

#[test]
fn test_weight_quantization_and_pruning() {
    // Create test weights with various magnitudes
    let mut weights = ModelWeights::new("optimization_test".to_string());

    weights.add_layer(
        LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![0.001, 0.5, 1.0, 2.0, 3.5, 5.0],
            vec![6],
        ),
    );

    let original_weights = weights.layers[0].weights.clone();

    // Test quantization
    let mut quantized_weights = weights.clone();
    WeightExporter::quantize(&mut quantized_weights, 8).unwrap();

    // Quantized weights should be different but similar
    assert_ne!(quantized_weights.layers[0].weights, original_weights);
    for (q, o) in quantized_weights.layers[0].weights.iter().zip(original_weights.iter()) {
        assert!((q - o).abs() < 0.1); // Within quantization error
    }

    // Test pruning
    let mut pruned_weights = weights.clone();
    let pruned_count = WeightExporter::prune(&mut pruned_weights, 0.1);

    // Should prune the small weight (0.001)
    assert!(pruned_count > 0);
    assert_eq!(pruned_weights.layers[0].weights[0], 0.0);
    assert_ne!(pruned_weights.layers[0].weights[1], 0.0); // Larger weights preserved
}

#[test]
fn test_model_config_validation() {
    // Test valid configuration
    let mut valid_config = ModelConfig::new("valid_model")
        .with_input_shape(vec![128])
        .with_output_shape(vec![10]);

    valid_config.add_layer(LayerConfig::new(
        "layer1".to_string(),
        LayerType::SpikingLinear,
        128,
        64,
    ));

    valid_config.add_layer(LayerConfig::new(
        "layer2".to_string(),
        LayerType::SpikingLinear,
        64,
        10,
    ));

    assert!(valid_config.validate().is_ok());

    // Test dimension mismatch
    let mut invalid_config = ModelConfig::new("invalid_model")
        .with_input_shape(vec![128])
        .with_output_shape(vec![10]);

    invalid_config.add_layer(LayerConfig::new(
        "layer1".to_string(),
        LayerType::SpikingLinear,
        128,
        64,
    ));

    invalid_config.add_layer(LayerConfig::new(
        "layer2".to_string(),
        LayerType::SpikingLinear,
        32, // Mismatch! Should be 64
        10,
    ));

    let result = invalid_config.validate();
    assert!(result.is_err());
    let errors = result.unwrap_err();
    assert!(errors.iter().any(|e| e.contains("dimension mismatch")));

    // Test empty configuration
    let empty_config = ModelConfig::new("empty");
    let result = empty_config.validate();
    assert!(result.is_err());
}

#[test]
fn test_encoder_export() {
    let exporter = OnnxExporter::with_default_config();

    // Test rate encoder
    let params = serde_json::json!({
        "scale": 1.0
    });

    let result = exporter.export_encoder(
        "rate",
        &params,
        &[128],
    );

    assert!(result.is_ok());
    let export_result = result.unwrap();
    assert!(!export_result.model_bytes.is_empty());
    assert!(!export_result.warnings.is_empty()); // Should warn about approximation

    // Test population encoder
    let result = exporter.export_encoder(
        "population",
        &params,
        &[128],
    );

    assert!(result.is_ok());

    // Test unsupported encoder
    let result = exporter.export_encoder(
        "unknown",
        &params,
        &[128],
    );

    assert!(result.is_err());
}

#[test]
fn test_decoder_export() {
    let exporter = OnnxExporter::with_default_config();

    let params = serde_json::json!({});

    // Test spike count decoder
    let result = exporter.export_decoder(
        "spike_count",
        &params,
        &[100, 10],
    );

    assert!(result.is_ok());
    let export_result = result.unwrap();
    assert!(!export_result.model_bytes.is_empty());

    // Test rate decoder
    let result = exporter.export_decoder(
        "rate",
        &params,
        &[100, 10],
    );

    assert!(result.is_ok());

    // Test unsupported decoder
    let result = exporter.export_decoder(
        "unknown",
        &params,
        &[100, 10],
    );

    assert!(result.is_err());
}

#[test]
fn test_export_metadata_creation() {
    let metadata = ExportMetadata::new("test_model", "onnx")
        .with_shapes(vec![128], vec![10])
        .with_total_params(10000)
        .with_quantization(true)
        .with_notes("Test export for integration testing".to_string());

    assert_eq!(metadata.model_name, "test_model");
    assert_eq!(metadata.export_format, "onnx");
    assert_eq!(metadata.input_shape, vec![128]);
    assert_eq!(metadata.output_shape, vec![10]);
    assert_eq!(metadata.total_params, 10000);
    assert!(metadata.quantized);
    assert!(metadata.notes.is_some());

    // Test JSON serialization
    let json = metadata.to_json();
    assert!(!json.is_empty());

    let loaded = ExportMetadata::from_json(&json).unwrap();
    assert_eq!(loaded.model_name, metadata.model_name);
    assert_eq!(loaded.total_params, metadata.total_params);
}

#[test]
fn test_onnx_config_customization() {
    // Test default config
    let default_config = OnnxConfig::default();
    assert_eq!(default_config.opset_version, 13);
    assert!(default_config.optimize);
    assert!(default_config.quantize.is_none());

    // Test custom config
    let custom_config = OnnxConfig {
        opset_version: 14,
        optimize: false,
        quantize: None,
        input_names: vec!["custom_input".to_string()],
        output_names: vec!["custom_output".to_string()],
        dynamic_axes: Some(vec![("input".to_string(), vec![0, 1])]),
    };

    // Create exporter with custom config - config is used internally
    let _exporter = OnnxExporter::new(custom_config.clone());
    // Config was successfully passed to the exporter
}

#[test]
fn test_checksum_consistency() {
    let mut weights1 = ModelWeights::new("model1".to_string());
    weights1.add_layer(
        LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
        ),
    );
    weights1.update_checksum();

    let mut weights2 = ModelWeights::new("model1".to_string());
    weights2.add_layer(
        LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 3.0],
            vec![3],
        ),
    );
    weights2.update_checksum();

    // Same weights should produce same checksum
    assert_eq!(weights1.metadata.checksum, weights2.metadata.checksum);

    // Different weights should produce different checksum
    let mut weights3 = ModelWeights::new("model1".to_string());
    weights3.add_layer(
        LayerWeights::new(
            "layer1".to_string(),
            "Linear".to_string(),
            vec![1.0, 2.0, 4.0], // Different value
            vec![3],
        ),
    );
    weights3.update_checksum();

    assert_ne!(weights1.metadata.checksum, weights3.metadata.checksum);
}

#[test]
fn test_convolutional_layer_export() {
    let mut config = ModelConfig::new("conv_model")
        .with_input_shape(vec![1, 28, 28])
        .with_output_shape(vec![10]);

    config.add_layer(LayerConfig::new(
        "conv1".to_string(),
        LayerType::SpikingConv2d {
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
        },
        784, // 28*28
        6272, // 26*26*9 (approximate)
    ));

    // Validate the layer
    assert!(config.layers[0].validate().is_ok());
}

#[test]
fn test_recurrent_layer_export() {
    let mut config = ModelConfig::new("rnn_model")
        .with_input_shape(vec![100, 128])
        .with_output_shape(vec![10]);

    config.add_layer(LayerConfig::new(
        "rnn1".to_string(),
        LayerType::SpikingRecurrent,
        128,
        64,
    ));

    assert!(config.layers[0].validate().is_ok());
}

#[test]
fn test_dropout_layer_validation() {
    // Valid dropout rate
    let layer1 = LayerConfig::new(
        "dropout1".to_string(),
        LayerType::Dropout { rate: 0.5 },
        128,
        128,
    );
    assert!(layer1.validate().is_ok());

    // Invalid dropout rate (>= 1.0)
    let layer2 = LayerConfig::new(
        "dropout2".to_string(),
        LayerType::Dropout { rate: 1.5 },
        128,
        128,
    );
    assert!(layer2.validate().is_err());

    // Invalid dropout rate (< 0.0)
    let layer3 = LayerConfig::new(
        "dropout3".to_string(),
        LayerType::Dropout { rate: -0.1 },
        128,
        128,
    );
    assert!(layer3.validate().is_err());
}
