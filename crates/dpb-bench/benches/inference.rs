//! SNN inference benchmarks
//!
//! This benchmark suite measures the performance of full SNN inference
//! pipelines, including feedforward, recurrent, and convolutional architectures.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use dpb_snn::*;
// `forward` is a trait method.
use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::SpikeTensor;
use dpb_neurons::lif::LifConfig;
use ndarray::{Array1, Array2, Array3};

fn benchmark_feedforward_snn(c: &mut Criterion) {
    let mut group = c.benchmark_group("feedforward_snn");

    let config = SNNConfig {
        dt: 1.0,
        num_steps: 50,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    for layer_sizes in [
        vec![10, 20, 10],
        vec![50, 100, 50],
        vec![100, 200, 100],
    ].iter() {
        let mut snn = FeedforwardSNN::new(layer_sizes.clone(), config.clone(), true).expect("SNN");

        // `forward` takes a SpikeTensor of (batch, steps, neurons).
        let input = SpikeTensor::zeros(1, config.num_steps, layer_sizes[0], false);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", layer_sizes)),
            layer_sizes,
            |b, _| {
                b.iter(|| {
                    let output = snn.forward(black_box(&input));
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_snn_timesteps(c: &mut Criterion) {
    let mut group = c.benchmark_group("snn_timesteps");

    let layer_sizes = vec![50, 100, 50];

    for num_steps in [10, 50, 100, 200].iter() {
        let config = SNNConfig {
            dt: 1.0,
            num_steps: *num_steps,
            neuron_model: NeuronModel::LIF,
            neuron_params: NeuronParams::default(),
            ..SNNConfig::default()
        };

        let mut snn = FeedforwardSNN::new(layer_sizes.clone(), config.clone(), true).expect("SNN");
        // `forward` takes a SpikeTensor of (batch, steps, neurons).
        let input = SpikeTensor::zeros(1, config.num_steps, layer_sizes[0], false);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_steps),
            num_steps,
            |b, _| {
                b.iter(|| {
                    let output = snn.forward(black_box(&input));
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_recurrent_snn(c: &mut Criterion) {
    let mut group = c.benchmark_group("recurrent_snn");

    let config = SNNConfig {
        dt: 1.0,
        num_steps: 50,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    for num_neurons in [50, 100, 200].iter() {
        let mut snn = RecurrentSNN::new(*num_neurons, vec![*num_neurons], *num_neurons, config.clone()).expect("SNN");
        let input = SpikeTensor::zeros(1, config.num_steps, *num_neurons, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_neurons),
            num_neurons,
            |b, _| {
                b.iter(|| {
                    let output = snn.forward(black_box(&input));
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_convolutional_snn(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolutional_snn");

    let config = SNNConfig {
        dt: 1.0,
        num_steps: 50,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    for (channels, spatial_size) in [(1, 28), (3, 32)].iter() {
        let mut snn = ConvolutionalSNN::new(*channels, 10, config.clone());

        // Input: (timesteps, channels, height, width)
        let input =
            SpikeTensor::zeros(1, config.num_steps, *channels * spatial_size * spatial_size, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}ch_{}x{}", channels, spatial_size, spatial_size)),
            &(*channels, *spatial_size),
            |b, _| {
                b.iter(|| {
                    let output = snn.forward(black_box(&input));
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_spike_rate_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_rate_decoder");

    for num_neurons in [10, 50, 100].iter() {
        // (batch, steps, neurons), with a spike every fifth step.
        let mut spike_tensor = SpikeTensor::zeros(1, 50, *num_neurons, false);
        for neuron in 0..*num_neurons {
            for step in (0..50).step_by(5) {
                spike_tensor.set_spike(0, step, neuron, 1.0).ok();
            }
        }

        let decoder = SpikeRateDecoder::new(*num_neurons, None, false);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_neurons),
            num_neurons,
            |b, _| {
                b.iter(|| {
                    let output = decoder.decode(black_box(&spike_tensor));
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_population_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("population_decoder");

    let num_neurons = 100;
    let num_classes = 10;

    let mut spike_tensor = SpikeTensor::zeros(1, 50, num_neurons, false);
    for neuron in 0..num_neurons {
        for step in (0..50).step_by(3) {
            spike_tensor.set_spike(0, step, neuron, 1.0).ok();
        }
    }

    // (num_classes, neurons_per_class): the decoder expects their product as
    // its input width.
    let decoder = PopulationDecoder::new(num_classes, num_neurons / num_classes);

    group.bench_function("decode", |b| {
        b.iter(|| {
            let output = decoder.decode(black_box(&spike_tensor));
            black_box(output);
        });
    });

    group.finish();
}

fn benchmark_latency_decoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_decoder");

    let num_neurons = 100;
    let mut spike_tensor = SpikeTensor::zeros(1, 50, num_neurons, false);
    for neuron in 0..num_neurons {
        spike_tensor.set_spike(0, neuron % 50, neuron, 1.0).ok();
    }

    let decoder = LatencyDecoder::new(num_neurons, 50);

    group.bench_function("decode", |b| {
        b.iter(|| {
            let output = decoder.decode(black_box(&spike_tensor));
            black_box(output);
        });
    });

    group.finish();
}

fn benchmark_snn_neuron_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("snn_neuron_types");

    let layer_sizes = vec![50, 100, 50];
    let num_steps = 50;

    for neuron_type in [
        NeuronModel::LIF,
        NeuronModel::AdaptiveLIF,
        NeuronModel::Izhikevich,
    ].iter() {
        let config = SNNConfig {
            dt: 1.0,
            num_steps,
            neuron_model: *neuron_type,
            neuron_params: NeuronParams::default(),
            ..SNNConfig::default()
        };

        let mut snn = FeedforwardSNN::new(layer_sizes.clone(), config.clone(), true).expect("SNN");
        let input = SpikeTensor::zeros(1, num_steps, layer_sizes[0], false);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", neuron_type)),
            neuron_type,
            |b, _| {
                b.iter(|| {
                    let output = snn.forward(black_box(&input));
                    black_box(output);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_end_to_end_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_pipeline");

    // Simulate full pipeline: encoding -> SNN inference -> decoding
    let config = SNNConfig {
        dt: 1.0,
        num_steps: 50,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        ..SNNConfig::default()
    };

    let layer_sizes = vec![50, 100, 10];
    let mut snn = FeedforwardSNN::new(layer_sizes.clone(), config.clone(), true).expect("SNN");
    let decoder = SpikeRateDecoder::new(*layer_sizes.last().unwrap(), None, false);

    // Raw input signal
    let signal = vec![0.5; 50];

    group.bench_function("full_pipeline", |b| {
        b.iter(|| {
            // Simple rate encoding into a (batch, steps, neurons) tensor.
            let mut encoded = SpikeTensor::zeros(1, config.num_steps, layer_sizes[0], false);
            for step in 0..config.num_steps {
                for i in 0..layer_sizes[0].min(signal.len()) {
                    if signal[i] > 0.3 {
                        encoded.set_spike(0, step, i, 1.0).ok();
                    }
                }
            }

            // SNN inference
            let spike_output = snn.forward(black_box(&encoded)).expect("forward");

            // Decode
            let output = decoder.decode(black_box(&spike_output));

            black_box(output);
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_feedforward_snn,
    benchmark_snn_timesteps,
    benchmark_recurrent_snn,
    benchmark_convolutional_snn,
    benchmark_spike_rate_decoder,
    benchmark_population_decoder,
    benchmark_latency_decoder,
    benchmark_snn_neuron_types,
    benchmark_end_to_end_pipeline,
);

criterion_main!(benches);
