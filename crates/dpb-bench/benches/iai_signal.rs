//! iai-callgrind signal processing benchmarks
//!
//! Deterministic instruction-count benchmarks for signal processing primitives.
//!
//! Run with: cargo bench --bench iai_signal
//! Note: Requires valgrind to be installed on the system.

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

use dpb_core::signal::{
    FftProcessor, FirFilter, IirFilter, FilterType,
    normalize, NormalizationMethod, remove_dc_offset,
    find_peaks, find_valleys, zero_crossings, rms, energy, clip,
    downsample, upsample, resample_linear,
};
use ndarray::Array1;

// Setup functions
mod setup {
    use super::*;

    pub fn signal_256() -> Array1<f64> {
        // Generate a simple test signal (sine wave with noise)
        Array1::from_iter((0..256).map(|i| {
            let t = i as f64 / 256.0;
            (2.0 * std::f64::consts::PI * 10.0 * t).sin() + 0.1 * (i as f64 * 0.1).sin()
        }))
    }

    pub fn signal_1024() -> Array1<f64> {
        Array1::from_iter((0..1024).map(|i| {
            let t = i as f64 / 1024.0;
            (2.0 * std::f64::consts::PI * 10.0 * t).sin() + 0.1 * (i as f64 * 0.1).sin()
        }))
    }

    pub fn signal_4096() -> Array1<f64> {
        Array1::from_iter((0..4096).map(|i| {
            let t = i as f64 / 4096.0;
            (2.0 * std::f64::consts::PI * 10.0 * t).sin() + 0.1 * (i as f64 * 0.1).sin()
        }))
    }

    pub fn fir_filter_ma10() -> FirFilter {
        FirFilter::moving_average(10).unwrap()
    }

    pub fn fir_filter_ma50() -> FirFilter {
        FirFilter::moving_average(50).unwrap()
    }

    pub fn iir_lowpass() -> IirFilter {
        // The constructor names the family; `FilterType` is a plain enum.
        IirFilter::butterworth_lowpass(4, 30.0, 250.0).unwrap()
    }

    pub fn iir_bandpass() -> IirFilter {
        IirFilter::butterworth_bandpass(4, 0.5, 40.0, 250.0).unwrap()
    }

    pub fn fft_processor() -> FftProcessor {
        FftProcessor::new()
    }
}

// FFT benchmarks
#[library_benchmark]
#[bench::fft_256((setup::fft_processor(), setup::signal_256()))]
fn bench_fft_256((mut processor, signal): (FftProcessor, Array1<f64>)) -> dpb_core::Result<ndarray::Array1<num_complex::Complex<f64>>> {
    black_box(processor.fft(black_box(signal.view())))
}

#[library_benchmark]
#[bench::fft_1024((setup::fft_processor(), setup::signal_1024()))]
fn bench_fft_1024((mut processor, signal): (FftProcessor, Array1<f64>)) -> dpb_core::Result<ndarray::Array1<num_complex::Complex<f64>>> {
    black_box(processor.fft(black_box(signal.view())))
}

#[library_benchmark]
#[bench::fft_4096((setup::fft_processor(), setup::signal_4096()))]
fn bench_fft_4096((mut processor, signal): (FftProcessor, Array1<f64>)) -> dpb_core::Result<ndarray::Array1<num_complex::Complex<f64>>> {
    black_box(processor.fft(black_box(signal.view())))
}

// FIR filter benchmarks
#[library_benchmark]
#[bench::fir_ma10_1024((setup::fir_filter_ma10(), setup::signal_1024()))]
fn bench_fir_ma10_1024((mut filter, signal): (FirFilter, Array1<f64>)) -> Array1<f64> {
    filter.reset();
    black_box(filter.filter(black_box(signal.view())))
}

#[library_benchmark]
#[bench::fir_ma50_1024((setup::fir_filter_ma50(), setup::signal_1024()))]
fn bench_fir_ma50_1024((mut filter, signal): (FirFilter, Array1<f64>)) -> Array1<f64> {
    filter.reset();
    black_box(filter.filter(black_box(signal.view())))
}

// IIR filter benchmarks
#[library_benchmark]
#[bench::iir_lowpass_1024((setup::iir_lowpass(), setup::signal_1024()))]
fn bench_iir_lowpass_1024((mut filter, signal): (IirFilter, Array1<f64>)) -> Array1<f64> {
    filter.reset();
    black_box(filter.filter(black_box(signal.view())))
}

#[library_benchmark]
#[bench::iir_bandpass_1024((setup::iir_bandpass(), setup::signal_1024()))]
fn bench_iir_bandpass_1024((mut filter, signal): (IirFilter, Array1<f64>)) -> Array1<f64> {
    filter.reset();
    black_box(filter.filter(black_box(signal.view())))
}

// Normalization benchmarks
#[library_benchmark]
#[bench::zscore_1024(setup::signal_1024())]
fn bench_normalize_zscore(signal: Array1<f64>) -> dpb_core::Result<Array1<f64>> {
    black_box(normalize(black_box(signal.view()), NormalizationMethod::ZScore))
}

#[library_benchmark]
#[bench::minmax_1024(setup::signal_1024())]
fn bench_normalize_minmax(signal: Array1<f64>) -> dpb_core::Result<Array1<f64>> {
    black_box(normalize(black_box(signal.view()), NormalizationMethod::MinMax))
}

#[library_benchmark]
#[bench::robust_1024(setup::signal_1024())]
fn bench_normalize_robust(signal: Array1<f64>) -> dpb_core::Result<Array1<f64>> {
    black_box(normalize(black_box(signal.view()), NormalizationMethod::Robust))
}

// Signal utility benchmarks
#[library_benchmark]
#[bench::dc_offset_1024(setup::signal_1024())]
fn bench_remove_dc_offset(signal: Array1<f64>) -> Array1<f64> {
    black_box(remove_dc_offset(black_box(signal.view())))
}

#[library_benchmark]
#[bench::peaks_1024(setup::signal_1024())]
fn bench_find_peaks(signal: Array1<f64>) -> Vec<usize> {
    black_box(find_peaks(black_box(signal.view()), 0.5))
}

#[library_benchmark]
#[bench::valleys_1024(setup::signal_1024())]
fn bench_find_valleys(signal: Array1<f64>) -> Vec<usize> {
    black_box(find_valleys(black_box(signal.view()), -0.5))
}

#[library_benchmark]
#[bench::crossings_1024(setup::signal_1024())]
fn bench_zero_crossings(signal: Array1<f64>) -> Vec<usize> {
    black_box(zero_crossings(black_box(signal.view())))
}

#[library_benchmark]
#[bench::rms_1024(setup::signal_1024())]
fn bench_rms(signal: Array1<f64>) -> f64 {
    black_box(rms(black_box(signal.view())))
}

#[library_benchmark]
#[bench::energy_1024(setup::signal_1024())]
fn bench_energy(signal: Array1<f64>) -> f64 {
    black_box(energy(black_box(signal.view())))
}

#[library_benchmark]
#[bench::clip_1024(setup::signal_1024())]
fn bench_clip(signal: Array1<f64>) -> Array1<f64> {
    black_box(clip(black_box(signal.view()), -0.5, 0.5))
}

// Resampling benchmarks
#[library_benchmark]
#[bench::downsample_1024(setup::signal_1024())]
fn bench_downsample_1024(signal: Array1<f64>) -> dpb_core::Result<Array1<f64>> {
    black_box(downsample(black_box(signal.view()), 4))
}

#[library_benchmark]
#[bench::upsample_256(setup::signal_256())]
fn bench_upsample_256(signal: Array1<f64>) -> dpb_core::Result<Array1<f64>> {
    black_box(upsample(black_box(signal.view()), 4))
}

#[library_benchmark]
#[bench::resample_linear_1024(setup::signal_1024())]
fn bench_resample_linear(signal: Array1<f64>) -> dpb_core::Result<Array1<f64>> {
    black_box(resample_linear(black_box(signal.view()), 1024.0, 512.0))
}

library_benchmark_group!(
    name = fft;
    benchmarks =
        bench_fft_256,
        bench_fft_1024,
        bench_fft_4096
);

library_benchmark_group!(
    name = fir_filters;
    benchmarks =
        bench_fir_ma10_1024,
        bench_fir_ma50_1024
);

library_benchmark_group!(
    name = iir_filters;
    benchmarks =
        bench_iir_lowpass_1024,
        bench_iir_bandpass_1024
);

library_benchmark_group!(
    name = normalization;
    benchmarks =
        bench_normalize_zscore,
        bench_normalize_minmax,
        bench_normalize_robust
);

library_benchmark_group!(
    name = signal_utilities;
    benchmarks =
        bench_remove_dc_offset,
        bench_find_peaks,
        bench_find_valleys,
        bench_zero_crossings,
        bench_rms,
        bench_energy,
        bench_clip
);

library_benchmark_group!(
    name = resampling;
    benchmarks =
        bench_downsample_1024,
        bench_upsample_256,
        bench_resample_linear
);

main!(
    library_benchmark_groups =
        fft,
        fir_filters,
        iir_filters,
        normalization,
        signal_utilities,
        resampling
);
