//! Example: ANN-to-SNN Conversion
//!
//! This example demonstrates how to convert ANN baseline models to SNNs
//! using various normalization and threshold strategies.
//!
//! Run with: cargo run --example ann_to_snn_conversion

// The crate root aliases these with a `Baseline` prefix to disambiguate from
// the spiking types; inside a glob import of the module they carry their own
// names.
use dpb_snn::baselines::*;

fn main() {
    println!("=== ANN-to-SNN Conversion Example ===\n");

    // Create a sample ANN model (3-layer MLP)
    let mlp = MLP3Layer::new(100, 128, 64, 10, 42);
    println!("Original ANN Model:");
    println!("  {}", mlp.architecture_summary());
    println!("  Parameters: {}", mlp.num_parameters());
    println!("  FLOPs: {}\n", mlp.flops_per_inference());

    // Simulate model weights
    let weights = vec![
        Tensor::randn(vec![100, 128], 42),
        Tensor::randn(vec![128, 64], 43),
        Tensor::randn(vec![64, 10], 44),
    ];

    let biases = vec![
        Tensor::zeros(vec![128]),
        Tensor::zeros(vec![64]),
        Tensor::zeros(vec![10]),
    ];

    // Generate sample calibration data
    let mut sample_data = Vec::new();
    for i in 0..10 {
        sample_data.push(Tensor::randn(vec![100], 1000 + i));
    }

    println!("Calibration Data:");
    println!("  {} samples of shape {:?}\n", sample_data.len(), sample_data[0].shape);

    // Example 1: Default conversion
    println!("1. Default Conversion (Data-based normalization):");
    let (norm_weights1, norm_biases1, converter1) = convert_model_to_snn(
        weights.clone(),
        biases.clone(),
        &sample_data,
        None,
    );

    println!("{}", converter1.conversion_report());

    // Example 2: Model-based normalization
    println!("\n2. Model-based Normalization:");
    let config2 = ConversionConfig {
        weight_norm: WeightNormalizationMethod::ModelBased,
        threshold_strategy: ThresholdBalancingStrategy::Fixed(1.0),
        num_timesteps: 100,
        bias_correction: true,
        fold_batchnorm: true,
        clip_negative_weights: false,
    };

    let (norm_weights2, norm_biases2, converter2) = convert_model_to_snn(
        weights.clone(),
        biases.clone(),
        &sample_data,
        Some(config2),
    );

    println!("{}", converter2.conversion_report());

    // Example 3: Hybrid normalization with percentile threshold
    println!("\n3. Hybrid Normalization with Percentile Threshold:");
    let config3 = ConversionConfig {
        weight_norm: WeightNormalizationMethod::Hybrid,
        threshold_strategy: ThresholdBalancingStrategy::Percentile(0.95),
        num_timesteps: 200,
        bias_correction: true,
        fold_batchnorm: true,
        clip_negative_weights: true,
    };

    let (norm_weights3, norm_biases3, converter3) = convert_model_to_snn(
        weights.clone(),
        biases.clone(),
        &sample_data,
        Some(config3),
    );

    println!("{}", converter3.conversion_report());

    // Example 4: BatchNorm folding demonstration
    println!("\n4. Batch Normalization Folding:");
    let conv_weight = Tensor::randn(vec![64, 32, 3], 100);
    let conv_bias = Tensor::zeros(vec![64]);
    let bn_mean = Tensor::zeros(vec![64]);
    let bn_var = Tensor::ones(vec![64]);
    let bn_gamma = Tensor::ones(vec![64]);
    let bn_beta = Tensor::zeros(vec![64]);

    let converter = ANNToSNNConverter::new();
    let (folded_weight, folded_bias) = converter.fold_batchnorm_into_conv(
        &conv_weight,
        Some(&conv_bias),
        &bn_mean,
        &bn_var,
        &bn_gamma,
        &bn_beta,
    );

    println!("  Original conv weight shape: {:?}", conv_weight.shape);
    println!("  Folded weight shape: {:?}", folded_weight.shape);
    println!("  Folded bias shape: {:?}", folded_bias.shape);

    // Example 5: Activation to spike conversion
    println!("\n5. Activation to Spike Conversion:");
    let activation = Tensor::from_vec(
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 1.5],
        vec![6]
    );

    let converter = ANNToSNNConverter::new();
    let spikes = converter.convert_activation_to_spikes(&activation);

    println!("  Activation values: {:?}", activation.data);
    println!("  Spike rates: {:?}", spikes.data);

    // Example 6: Comparing different timestep counts
    println!("\n6. Impact of Timestep Count:");
    for num_steps in [50, 100, 200, 500] {
        let config = ConversionConfig {
            weight_norm: WeightNormalizationMethod::DataBased,
            threshold_strategy: ThresholdBalancingStrategy::LayerWise,
            num_timesteps: num_steps,
            bias_correction: true,
            fold_batchnorm: true,
            clip_negative_weights: false,
        };

        let (_, _, converter) = convert_model_to_snn(
            weights.clone(),
            biases.clone(),
            &sample_data,
            Some(config),
        );

        let est_loss = converter.estimate_accuracy_loss(weights.len());
        println!("  {} timesteps: estimated accuracy loss = {:.2}%",
                 num_steps, est_loss * 100.0);
    }

    // Summary
    println!("\n=== Conversion Summary ===");
    println!("Available Normalization Methods:");
    println!("  - DataBased: Uses max activation from calibration data");
    println!("  - ModelBased: Uses weight statistics");
    println!("  - Hybrid: Combines both approaches");
    println!("  - None: No normalization");
    println!("\nAvailable Threshold Strategies:");
    println!("  - Fixed(t): Same threshold for all layers");
    println!("  - LayerWise: Adaptive per layer");
    println!("  - Percentile(p): Based on activation percentile");
    println!("  - Learned: Learned during conversion");
    println!("\nRecommendations:");
    println!("  - Use DataBased or Hybrid normalization for best accuracy");
    println!("  - Use LayerWise thresholding for deep networks");
    println!("  - More timesteps = better accuracy but slower inference");
    println!("  - Always calibrate with representative data");
}
