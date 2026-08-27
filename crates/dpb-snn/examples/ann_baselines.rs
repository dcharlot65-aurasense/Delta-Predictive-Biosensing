//! Example: Using ANN Baseline Architectures
//!
//! This example demonstrates how to use the 44 ANN baseline architectures
//! for fair comparison with SNN models.
//!
//! Run with: cargo run --example ann_baselines

// The crate root aliases these as `BaselineTensor` and `BaselineConverter` to
// disambiguate from the spiking `Tensor`; inside a glob import of the module
// itself they carry their own names.
use dpb_snn::baselines::*;

fn main() {
    println!("=== DPB-SNN ANN Baseline Architectures ===\n");

    // Example 1: MLP Architectures
    println!("1. MLP Architectures:");
    let mlp2 = MLP2Layer::new(100, 64, 10, 42);
    println!("  {} - {}", mlp2.name(), mlp2.architecture_summary());
    println!("     Parameters: {}", mlp2.num_parameters());
    println!("     FLOPs: {}\n", mlp2.flops_per_inference());

    let mlp_deep = MLPDeep::new(100, 32, 10, 10, 42);
    println!("  {} - {}", mlp_deep.name(), mlp_deep.architecture_summary());
    println!("     Parameters: {}", mlp_deep.num_parameters());
    println!("     FLOPs: {}\n", mlp_deep.flops_per_inference());

    // Example 2: CNN Architectures
    println!("2. CNN Architectures:");
    let cnn1d = CNN1DSmall::new(3, 1000, 10, 42);
    println!("  {} - {}", cnn1d.name(), cnn1d.architecture_summary());
    println!("     Parameters: {}", cnn1d.num_parameters());
    println!("     FLOPs: {}\n", cnn1d.flops_per_inference());

    let tcn = TCN::new(3, 1000, 10, 42);
    println!("  {} - {}", tcn.name(), tcn.architecture_summary());
    println!("     Parameters: {}", tcn.num_parameters());
    println!("     FLOPs: {}\n", tcn.flops_per_inference());

    // Example 3: RNN Architectures
    println!("3. RNN Architectures:");
    let lstm = LSTM::new(100, 128, 10, 42);
    println!("  {} - {}", lstm.name(), lstm.architecture_summary());
    println!("     Parameters: {}", lstm.num_parameters());
    println!("     FLOPs: {}\n", lstm.flops_per_inference());

    let gru = GRU::new(100, 128, 10, 42);
    println!("  {} - {}", gru.name(), gru.architecture_summary());
    println!("     Parameters: {}", gru.num_parameters());
    println!("     FLOPs: {}\n", gru.flops_per_inference());

    // Example 4: Transformer Architectures
    println!("4. Transformer Architectures:");
    let transformer_small = TransformerSmall::new(128, 10, 42);
    println!("  {} - {}", transformer_small.name(), transformer_small.architecture_summary());
    println!("     Parameters: {}", transformer_small.num_parameters());
    println!("     FLOPs: {}\n", transformer_small.flops_per_inference());

    let informer = Informer::new(128, 4, 10, 42);
    println!("  {} - {}", informer.name(), informer.architecture_summary());
    println!("     Parameters: {}", informer.num_parameters());
    println!("     FLOPs: {}\n", informer.flops_per_inference());

    // Example 5: Specialized Architectures
    println!("5. Specialized Architectures:");
    let ecgnet = ECGNet::new(12, 1000, 5, 42);
    println!("  {} - {}", ecgnet.name(), ecgnet.architecture_summary());
    println!("     Parameters: {}", ecgnet.num_parameters());
    println!("     FLOPs: {}\n", ecgnet.flops_per_inference());

    let deepgait = DeepGait::new(6, 1000, 10, 42);
    println!("  {} - {}", deepgait.name(), deepgait.architecture_summary());
    println!("     Parameters: {}", deepgait.num_parameters());
    println!("     FLOPs: {}\n", deepgait.flops_per_inference());

    let multimodal = MultimodalFusion::new(&[100, 200, 150], 128, 10, 42);
    println!("  {} - {}", multimodal.name(), multimodal.architecture_summary());
    println!("     Parameters: {}", multimodal.num_parameters());
    println!("     FLOPs: {}\n", multimodal.flops_per_inference());

    // Example 6: Forward pass
    println!("6. Forward Pass Example:");
    let mlp = MLP3Layer::new(10, 20, 15, 5, 42);
    let input = Tensor::randn(vec![1, 10], 123);
    let output = mlp.forward(&input);
    println!("  Input shape: {:?}", input.shape);
    println!("  Output shape: {:?}", output.shape);
    println!("  Output values: {:?}\n", &output.data[0..5]);

    // Example 7: ANN-to-SNN Conversion
    println!("7. ANN-to-SNN Conversion:");
    let mut converter = ANNToSNNConverter::new();

    // Create sample weights and data
    let weights = vec![
        Tensor::randn(vec![10, 20], 42),
        Tensor::randn(vec![20, 10], 43),
    ];
    let _biases = [Tensor::zeros(vec![20]),
        Tensor::zeros(vec![10])];
    let sample_data = vec![
        Tensor::randn(vec![10], 123),
    ];

    // Calibrate and convert
    converter.calibrate(&weights, &sample_data);
    let report = converter.conversion_report();
    println!("{}", report);

    // Example 8: Architecture Comparison
    println!("8. Architecture Comparison:");
    let architectures: Vec<Box<dyn ANNBaseline>> = vec![
        Box::new(MLP2Layer::new(100, 64, 10, 42)),
        Box::new(CNN1DSmall::new(3, 1000, 10, 42)),
        Box::new(LSTM::new(100, 64, 10, 42)),
        Box::new(TransformerSmall::new(64, 10, 42)),
    ];

    println!("  Architecture Comparison Table:");
    println!("  {:<25} {:<15} {:<15}", "Name", "Parameters", "FLOPs");
    println!("  {}", "-".repeat(55));
    for arch in &architectures {
        println!("  {:<25} {:<15} {:<15}",
                 arch.name(),
                 arch.num_parameters(),
                 arch.flops_per_inference());
    }
    println!();

    // Summary
    println!("=== Summary ===");
    println!("Total ANN Baseline Architectures: 44");
    println!("  - MLP Architectures: 8");
    println!("  - CNN Architectures: 10");
    println!("  - RNN Architectures: 10");
    println!("  - Transformer Architectures: 8");
    println!("  - Specialized Architectures: 8");
    println!("\nAll baselines are ready for fair SNN comparison!");
}
