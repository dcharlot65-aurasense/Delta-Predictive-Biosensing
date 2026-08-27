use criterion::{Criterion, criterion_group, criterion_main};
use dpb_core::signal::{FftProcessor, FirFilter};
use ndarray::Array1;
use std::hint::black_box;

fn bench_fir_filter(c: &mut Criterion) {
    let mut filter = FirFilter::moving_average(10).unwrap();
    let signal = Array1::from_vec(vec![1.0; 1000]);

    c.bench_function("fir_filter_1000", |b| {
        b.iter(|| {
            filter.reset();
            filter.filter(black_box(signal.view()))
        })
    });
}

fn bench_fft(c: &mut Criterion) {
    let mut processor = FftProcessor::new();
    let signal = Array1::from_vec(vec![1.0; 1024]);

    c.bench_function("fft_1024", |b| {
        b.iter(|| processor.fft(black_box(signal.view())))
    });
}

criterion_group!(benches, bench_fir_filter, bench_fft);
criterion_main!(benches);
