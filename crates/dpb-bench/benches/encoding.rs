//! Encoder benchmarks
//!
//! This benchmark suite measures the performance of various encoders
//! on synthetic biosignal data.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use dpb_bench::datasets::{SyntheticECG, SyntheticGait, SyntheticTremor, BenchmarkDataset};
use dpb_encoders::prelude::*;
use dpb_core::{EventEncoder, SignalBuffer};

fn benchmark_level_crossing(c: &mut Criterion) {
    let mut group = c.benchmark_group("level_crossing_encoder");

    for size in [1000, 5000, 10000].iter() {
        // Generate ECG data
        let dataset = SyntheticECG::new(*size, 250.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();

        let encoder = LevelCrossingEncoder::new("benchmark");
        let config = LevelCrossingConfig {
            threshold: 0.3,
            relative: false,
            refractory_period: 0.01,
            ..LevelCrossingConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder.encode(black_box(signal), &config).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_template_deviation(c: &mut Criterion) {
    let mut group = c.benchmark_group("template_deviation_encoder");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticECG::new(*size, 250.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();

        let encoder = TemplateDeviationEncoder::new("benchmark");

        // Create a simple template
        // `template` is the waveform itself, not a wrapped buffer.
        let template = vec![0.0, 0.5, 1.0, 0.5, 0.0];

        let config = TemplateDeviationConfig {
            template,
            threshold: 0.2,
            window_size: 5,
            ..TemplateDeviationConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder.encode(black_box(signal), &config).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ecg_r_peak(c: &mut Criterion) {
    let mut group = c.benchmark_group("ecg_r_peak_encoder");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticECG::new(*size, 250.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();

        let encoder = EcgRPeakEncoder::new();
        let config = EcgRPeakConfig::default();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder.encode(black_box(signal), &config).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_gait_heel_strike(c: &mut Criterion) {
    let mut group = c.benchmark_group("heel_strike_encoder");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticGait::new(*size, 100.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();

        let encoder = HeelStrikeEncoder::new();
        let config = HeelStrikeConfig::default();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder.encode(black_box(signal), &config).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_tremor_frequency(c: &mut Criterion) {
    let mut group = c.benchmark_group("tremor_frequency_encoder");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticTremor::new(*size, 100.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();

        let encoder = TremorFrequencyEncoder::new();
        let config = TremorFrequencyConfig::default();

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder.encode(black_box(signal), &config).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_derivative(c: &mut Criterion) {
    let mut group = c.benchmark_group("derivative_encoder");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticECG::new(*size, 250.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();

        let encoder = DerivativeEncoder::new("benchmark");
        let config = DerivativeConfig {
            threshold: 0.5,
            ..DerivativeConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder.encode(black_box(signal), &config).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder_throughput");
    group.throughput(criterion::Throughput::Elements(1000));

    let dataset = SyntheticECG::new(1000, 250.0, Some(42));
    let (signal, _) = dataset.generate().unwrap();

    let encoders: Vec<(&str, Box<dyn Fn(&SignalBuffer) -> dpb_core::Result<Vec<dpb_core::SpikeEvent>>>)> = vec![
        ("level_crossing", Box::new(|s| {
            let encoder = LevelCrossingEncoder::new("bench");
            let config = LevelCrossingConfig {
                threshold: 0.3,
                relative: false,
                refractory_period: 0.01,
                ..LevelCrossingConfig::default()
            };
            encoder.encode(s, &config)
        })),
        ("derivative", Box::new(|s| {
            let encoder = DerivativeEncoder::new("bench");
            let config = DerivativeConfig {
                threshold: 0.5,
                ..DerivativeConfig::default()
            };
            encoder.encode(s, &config)
        })),
    ];

    for (name, encoder_fn) in encoders {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &signal,
            |b, signal| {
                b.iter(|| {
                    let events = encoder_fn(black_box(signal)).unwrap();
                    black_box(events);
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_level_crossing,
    benchmark_template_deviation,
    benchmark_ecg_r_peak,
    benchmark_gait_heel_strike,
    benchmark_tremor_frequency,
    benchmark_derivative,
    benchmark_throughput,
);

criterion_main!(benches);
