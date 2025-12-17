//! Neuron model benchmarks
//!
//! This benchmark suite measures the performance of various neuron models
//! in single neuron and batch processing modes.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use dpb_neurons::prelude::*;
use dpb_neurons::traits::NeuronModel;
use dpb_neurons::batch::BatchLifLayer;
use ndarray::Array1;

fn benchmark_lif_neuron(c: &mut Criterion) {
    let mut group = c.benchmark_group("lif_neuron");

    let config = dpb_neurons::lif::LifConfig::default();
    let mut neuron = LifNeuron::new(config);

    group.bench_function("single_update", |b| {
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    group.bench_function("1000_updates", |b| {
        b.iter(|| {
            for _ in 0..1000 {
                let spiked = neuron.update(black_box(10.0), black_box(1.0));
                black_box(spiked);
            }
        });
    });

    group.finish();
}

fn benchmark_neuron_models(c: &mut Criterion) {
    let mut group = c.benchmark_group("neuron_models");

    // LIF
    group.bench_function("LIF", |b| {
        let config = dpb_neurons::lif::LifConfig::default();
        let mut neuron = LifNeuron::new(config);
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    // ALIF
    group.bench_function("ALIF", |b| {
        let config = dpb_neurons::lif::AlifConfig::default();
        let mut neuron = AlifNeuron::new(config);
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    // Izhikevich
    group.bench_function("Izhikevich", |b| {
        let config = dpb_neurons::izhikevich::IzhikevichConfig::default();
        let mut neuron = IzhikevichNeuron::new(config);
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    // AdEx
    group.bench_function("AdEx", |b| {
        let config = dpb_neurons::adex::AdExConfig::default();
        let mut neuron = AdExNeuron::new(config);
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    group.finish();
}

fn benchmark_batch_neurons(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_neurons");

    for num_neurons in [10, 100, 1000].iter() {
        let config = dpb_neurons::lif::LifConfig::default();
        let mut layer = BatchLifLayer::new(*num_neurons, config);
        let inputs = Array1::from_elem(*num_neurons, 10.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_neurons),
            num_neurons,
            |b, _| {
                b.iter(|| {
                    let spikes = layer.update(black_box(&inputs), black_box(1.0));
                    black_box(spikes);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_batch_timesteps(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_timesteps");

    let num_neurons = 100;
    let config = dpb_neurons::lif::LifConfig::default();

    for num_timesteps in [10, 50, 100].iter() {
        let mut layer = BatchLifLayer::new(num_neurons, config.clone());
        let inputs = Array1::from_elem(num_neurons, 10.0);

        group.bench_with_input(
            BenchmarkId::from_parameter(num_timesteps),
            num_timesteps,
            |b, &timesteps| {
                b.iter(|| {
                    for _ in 0..timesteps {
                        let spikes = layer.update(black_box(&inputs), black_box(1.0));
                        black_box(spikes);
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_hardware_neurons(c: &mut Criterion) {
    let mut group = c.benchmark_group("hardware_neurons");

    // Xylo
    group.bench_function("XyloLIF", |b| {
        let config = dpb_neurons::hardware::XyloLifConfig::default();
        let mut neuron = dpb_neurons::XyloLifNeuron::new(config);
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    // Quantized
    group.bench_function("QuantizedLIF", |b| {
        let config = dpb_neurons::hardware::QuantizedLifConfig::default();
        let mut neuron = dpb_neurons::QuantizedLifNeuron::new(config);
        b.iter(|| {
            let spiked = neuron.update(black_box(10.0), black_box(1.0));
            black_box(spiked);
        });
    });

    group.finish();
}

fn benchmark_spike_response(c: &mut Criterion) {
    let mut group = c.benchmark_group("spike_response");

    let config = dpb_neurons::lif::LifConfig::default();

    // Measure average firing rate vs input current
    for current in [5.0, 10.0, 15.0, 20.0].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}nA", current)),
            current,
            |b, &current| {
                b.iter(|| {
                    let mut neuron = LifNeuron::new(config.clone());
                    let mut spikes = 0;
                    for _ in 0..1000 {
                        if neuron.update(black_box(current), black_box(1.0)) {
                            spikes += 1;
                        }
                    }
                    black_box(spikes);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_refractory_period(c: &mut Criterion) {
    let mut group = c.benchmark_group("refractory_period");

    for refrac_ms in [1.0, 2.0, 5.0].iter() {
        let config = dpb_neurons::lif::LifConfig {
            tau_mem: 20.0,
            tau_syn: 5.0,
            v_threshold: 1.0,
            v_reset: 0.0,
            t_refrac: *refrac_ms,
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}ms", refrac_ms)),
            refrac_ms,
            |b, _| {
                b.iter(|| {
                    let mut neuron = LifNeuron::new(config.clone());
                    for _ in 0..100 {
                        let spiked = neuron.update(black_box(20.0), black_box(1.0));
                        black_box(spiked);
                    }
                });
            },
        );
    }

    group.finish();
}

fn benchmark_surrogate_gradients(c: &mut Criterion) {
    let mut group = c.benchmark_group("surrogate_gradients");

    let voltages = vec![0.0, 0.5, 1.0, 1.5, 2.0];

    group.bench_function("FastSigmoid", |b| {
        let surrogate = FastSigmoid::default();
        b.iter(|| {
            for &v in &voltages {
                let grad = surrogate.gradient(black_box(v));
                black_box(grad);
            }
        });
    });

    group.bench_function("SuperSpike", |b| {
        let surrogate = dpb_neurons::SuperSpike::default();
        b.iter(|| {
            for &v in &voltages {
                let grad = surrogate.gradient(black_box(v));
                black_box(grad);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_lif_neuron,
    benchmark_neuron_models,
    benchmark_batch_neurons,
    benchmark_batch_timesteps,
    benchmark_hardware_neurons,
    benchmark_spike_response,
    benchmark_refractory_period,
    benchmark_surrogate_gradients,
);

criterion_main!(benches);
