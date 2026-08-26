//! Synthetic data generation benchmarks
//!
//! This benchmark suite measures the performance of synthetic biosignal
//! generators from dpb-synth.

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use std::hint::black_box;
use dpb_bench::datasets::{
    SyntheticECG, SyntheticGait, SyntheticTremor, SyntheticVoice, BenchmarkDataset,
};

fn benchmark_ecg_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ecg_generation");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticECG::new(*size, 250.0, Some(42));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_gait_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gait_generation");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticGait::new(*size, 100.0, Some(42));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_tremor_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tremor_generation");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticTremor::new(*size, 100.0, Some(42));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_voice_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_generation");

    for size in [1000, 5000, 10000].iter() {
        let dataset = SyntheticVoice::new(*size, 16000.0, Some(42));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ecg_heart_rates(c: &mut Criterion) {
    let mut group = c.benchmark_group("ecg_heart_rates");

    let size = 5000;

    for heart_rate in [60.0, 75.0, 100.0, 120.0].iter() {
        let dataset = SyntheticECG::new(size, 250.0, Some(42))
            .with_heart_rate(*heart_rate);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}bpm", heart_rate)),
            heart_rate,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_tremor_frequencies(c: &mut Criterion) {
    let mut group = c.benchmark_group("tremor_frequencies");

    let size = 5000;

    for frequency in [4.0, 5.0, 6.0, 8.0].iter() {
        let dataset = SyntheticTremor::new(size, 100.0, Some(42))
            .with_frequency(*frequency);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}Hz", frequency)),
            frequency,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_voice_f0(c: &mut Criterion) {
    let mut group = c.benchmark_group("voice_f0");

    let size = 5000;

    for f0 in [100.0, 120.0, 150.0, 200.0].iter() {
        let dataset = SyntheticVoice::new(size, 16000.0, Some(42))
            .with_f0(*f0);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{}Hz", f0)),
            f0,
            |b, _| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_dataset_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("dataset_throughput");
    group.throughput(criterion::Throughput::Elements(1000));

    let datasets: Vec<(&str, Box<dyn BenchmarkDataset>)> = vec![
        ("ecg", Box::new(SyntheticECG::new(1000, 250.0, Some(42)))),
        ("gait", Box::new(SyntheticGait::new(1000, 100.0, Some(42)))),
        ("tremor", Box::new(SyntheticTremor::new(1000, 100.0, Some(42)))),
        ("voice", Box::new(SyntheticVoice::new(1000, 16000.0, Some(42)))),
    ];

    for (name, dataset) in datasets {
        group.bench_with_input(
            BenchmarkId::from_parameter(name),
            &dataset,
            |b, dataset| {
                b.iter(|| {
                    let result = dataset.generate().unwrap();
                    black_box(result);
                });
            },
        );
    }

    group.finish();
}

fn benchmark_ground_truth_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ground_truth_generation");

    let dataset = SyntheticECG::new(10000, 250.0, Some(42));

    group.bench_function("with_ground_truth", |b| {
        b.iter(|| {
            let (signal, gt) = dataset.generate().unwrap();
            black_box((signal, gt));
        });
    });

    group.finish();
}

fn benchmark_reproducibility(c: &mut Criterion) {
    let mut group = c.benchmark_group("reproducibility");

    group.bench_function("seeded_generation", |b| {
        b.iter(|| {
            let dataset1 = SyntheticECG::new(1000, 250.0, Some(42));
            let (signal1, _) = dataset1.generate().unwrap();

            let dataset2 = SyntheticECG::new(1000, 250.0, Some(42));
            let (signal2, _) = dataset2.generate().unwrap();

            // Verify they're the same
            assert_eq!(signal1.data, signal2.data);

            black_box((signal1, signal2));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ecg_generation,
    benchmark_gait_generation,
    benchmark_tremor_generation,
    benchmark_voice_generation,
    benchmark_ecg_heart_rates,
    benchmark_tremor_frequencies,
    benchmark_voice_f0,
    benchmark_dataset_throughput,
    benchmark_ground_truth_generation,
    benchmark_reproducibility,
);

criterion_main!(benches);
