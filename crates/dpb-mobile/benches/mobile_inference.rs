use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use dpb_mobile::{
    MobileModel, MobileRuntime,
    model::{LayerInfo, LayerType, QuantizationType},
    optimization::{WeightPruner, PruningStrategy, OperatorFusion, QuantizationOptimizer},
    benchmark::{BenchmarkRunner, BenchmarkConfig},
};

fn create_test_model(input_dim: usize, hidden_dim: usize, output_dim: usize) -> MobileModel {
    let mut model = MobileModel::new("benchmark_model".into(), input_dim, output_dim);

    // Add input layer
    let layer1 = LayerInfo {
        name: "layer1".into(),
        input_dim,
        output_dim: hidden_dim,
        layer_type: LayerType::FullyConnected,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; input_dim * hidden_dim * 4],
        bias: Some(vec![0u8; hidden_dim * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer1);

    // Add hidden layer
    let layer2 = LayerInfo {
        name: "layer2".into(),
        input_dim: hidden_dim,
        output_dim: hidden_dim,
        layer_type: LayerType::Spiking,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; hidden_dim * hidden_dim * 4],
        bias: Some(vec![0u8; hidden_dim * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer2);

    // Add output layer
    let layer3 = LayerInfo {
        name: "layer3".into(),
        input_dim: hidden_dim,
        output_dim,
        layer_type: LayerType::Readout,
        quantization: QuantizationType::Float32,
        weights: vec![0u8; hidden_dim * output_dim * 4],
        bias: Some(vec![0u8; output_dim * 4]),
        scale: None,
        zero_point: None,
    };
    model.add_layer(layer3);

    model
}

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    for size in [32, 64, 128].iter() {
        let model = create_test_model(*size, *size * 2, *size);
        let mut runtime = MobileRuntime::new(model).build().unwrap();
        let input = vec![0.5f32; *size];

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let _ = black_box(runtime.infer(&input).unwrap());
            });
        });
    }

    group.finish();
}

fn bench_model_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_serialization");

    for size in [32, 64, 128].iter() {
        let model = create_test_model(*size, *size * 2, *size);

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let _ = black_box(model.to_bytes().unwrap());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_inference,
    bench_model_serialization,
);

criterion_main!(benches);
