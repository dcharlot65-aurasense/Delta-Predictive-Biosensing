//! iai-callgrind encoder benchmarks
//!
//! Deterministic instruction-count benchmarks for spike encoders.
//!
//! Run with: cargo bench --bench iai_encoding
//! Note: Requires valgrind to be installed on the system.

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

use dpb_bench::datasets::{SyntheticECG, SyntheticGait, SyntheticTremor, BenchmarkDataset};
use dpb_core::{EventEncoder, SignalBuffer, SpikeEvent};
use dpb_encoders::prelude::*;

// Setup functions
mod setup {
    use super::*;

    pub fn ecg_signal_1000() -> SignalBuffer {
        let dataset = SyntheticECG::new(1000, 250.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();
        signal
    }

    pub fn ecg_signal_5000() -> SignalBuffer {
        let dataset = SyntheticECG::new(5000, 250.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();
        signal
    }

    pub fn gait_signal_1000() -> SignalBuffer {
        let dataset = SyntheticGait::new(1000, 100.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();
        signal
    }

    pub fn tremor_signal_1000() -> SignalBuffer {
        let dataset = SyntheticTremor::new(1000, 100.0, Some(42));
        let (signal, _) = dataset.generate().unwrap();
        signal
    }

    pub fn level_crossing_encoder() -> (LevelCrossingEncoder, LevelCrossingConfig) {
        let encoder = LevelCrossingEncoder::new("iai_bench");
        let config = LevelCrossingConfig {
            threshold: 0.3,
            relative: false,
            refractory_period: 0.01,
            ..LevelCrossingConfig::default()
        };
        (encoder, config)
    }

    pub fn derivative_encoder() -> (DerivativeEncoder, DerivativeConfig) {
        let encoder = DerivativeEncoder::new("iai_bench");
        // `DerivativeConfig` carries a threshold and a derivative order; it has
        // never had a window or a refractory period.
        let config = DerivativeConfig {
            threshold: 0.5,
            order: 1,
            ..DerivativeConfig::default()
        };
        (encoder, config)
    }

    pub fn ecg_rpeak_encoder() -> (EcgRPeakEncoder, EcgRPeakConfig) {
        let encoder = EcgRPeakEncoder::new();
        let config = EcgRPeakConfig::default();
        (encoder, config)
    }

    pub fn heel_strike_encoder() -> (HeelStrikeEncoder, HeelStrikeConfig) {
        let encoder = HeelStrikeEncoder::new();
        let config = HeelStrikeConfig::default();
        (encoder, config)
    }

    pub fn tremor_frequency_encoder() -> (TremorFrequencyEncoder, TremorFrequencyConfig) {
        let encoder = TremorFrequencyEncoder::new();
        let config = TremorFrequencyConfig::default();
        (encoder, config)
    }
}

// Level Crossing Encoder benchmarks
#[library_benchmark]
#[bench::ecg_1000((setup::level_crossing_encoder(), setup::ecg_signal_1000()))]
fn bench_level_crossing_ecg_1000(
    ((encoder, config), signal): ((LevelCrossingEncoder, LevelCrossingConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

#[library_benchmark]
#[bench::ecg_5000((setup::level_crossing_encoder(), setup::ecg_signal_5000()))]
fn bench_level_crossing_ecg_5000(
    ((encoder, config), signal): ((LevelCrossingEncoder, LevelCrossingConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

// Derivative Encoder benchmarks
#[library_benchmark]
#[bench::ecg_1000((setup::derivative_encoder(), setup::ecg_signal_1000()))]
fn bench_derivative_ecg_1000(
    ((encoder, config), signal): ((DerivativeEncoder, DerivativeConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

#[library_benchmark]
#[bench::ecg_5000((setup::derivative_encoder(), setup::ecg_signal_5000()))]
fn bench_derivative_ecg_5000(
    ((encoder, config), signal): ((DerivativeEncoder, DerivativeConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

// ECG R-Peak Encoder benchmarks
#[library_benchmark]
#[bench::ecg_1000((setup::ecg_rpeak_encoder(), setup::ecg_signal_1000()))]
fn bench_ecg_rpeak_1000(
    ((encoder, config), signal): ((EcgRPeakEncoder, EcgRPeakConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

#[library_benchmark]
#[bench::ecg_5000((setup::ecg_rpeak_encoder(), setup::ecg_signal_5000()))]
fn bench_ecg_rpeak_5000(
    ((encoder, config), signal): ((EcgRPeakEncoder, EcgRPeakConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

// Gait Encoder benchmarks
#[library_benchmark]
#[bench::gait_1000((setup::heel_strike_encoder(), setup::gait_signal_1000()))]
fn bench_heel_strike_1000(
    ((encoder, config), signal): ((HeelStrikeEncoder, HeelStrikeConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

// Tremor Encoder benchmarks
#[library_benchmark]
#[bench::tremor_1000((setup::tremor_frequency_encoder(), setup::tremor_signal_1000()))]
fn bench_tremor_frequency_1000(
    ((encoder, config), signal): ((TremorFrequencyEncoder, TremorFrequencyConfig), SignalBuffer)
) -> Vec<SpikeEvent> {
    black_box(encoder.encode(black_box(&signal), &config).unwrap())
}

library_benchmark_group!(
    name = level_crossing;
    benchmarks =
        bench_level_crossing_ecg_1000,
        bench_level_crossing_ecg_5000
);

library_benchmark_group!(
    name = derivative;
    benchmarks =
        bench_derivative_ecg_1000,
        bench_derivative_ecg_5000
);

library_benchmark_group!(
    name = ecg_rpeak;
    benchmarks =
        bench_ecg_rpeak_1000,
        bench_ecg_rpeak_5000
);

library_benchmark_group!(
    name = domain_specific;
    benchmarks =
        bench_heel_strike_1000,
        bench_tremor_frequency_1000
);

main!(
    library_benchmark_groups =
        level_crossing,
        derivative,
        ecg_rpeak,
        domain_specific
);
