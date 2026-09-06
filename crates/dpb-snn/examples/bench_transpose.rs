//! Compares `W.t().dot(v)` against a row-wise accumulation of the same result.
//!
//! The two are interleaved inside one loop and the median of paired trials is
//! reported: on a loaded machine the absolute times drift, but both variants
//! see the same contention within a pair, so the ratio stays meaningful.

use ndarray::{Array1, Array2};
use std::hint::black_box;
use std::time::Instant;

/// dL/dinput = W^T g, as written today.
fn transposed_dot(w: &Array2<f32>, g: &Array1<f32>) -> Array1<f32> {
    w.t().dot(g)
}

/// The same value, accumulated over rows of W.
///
/// `w.row(i)` is contiguous where `w.t()` is not, so each step is a plain
/// strided axpy rather than a gather.
fn row_accumulate(w: &Array2<f32>, g: &Array1<f32>) -> Array1<f32> {
    let mut out = Array1::zeros(w.shape()[1]);
    for (i, &gi) in g.iter().enumerate() {
        if gi != 0.0 {
            out.scaled_add(gi, &w.row(i));
        }
    }
    out
}

fn main() {
    for (out_size, in_size) in [(128usize, 128usize), (512, 512), (1024, 256)] {
        let w = Array2::from_shape_fn((out_size, in_size), |(i, j)| {
            ((i * 31 + j * 17) % 100) as f32 * 0.01 - 0.5
        });
        let g = Array1::from_shape_fn(out_size, |i| ((i % 7) as f32) * 0.1);

        // Agreement first: a faster wrong answer is worthless.
        let a = transposed_dot(&w, &g);
        let b = row_accumulate(&w, &g);
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "results differ by {worst}");

        const PAIRS: usize = 40;
        const INNER: usize = 60;
        let mut ratios = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            let t0 = Instant::now();
            for _ in 0..INNER {
                black_box(transposed_dot(black_box(&w), black_box(&g)));
            }
            let dot = t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            for _ in 0..INNER {
                black_box(row_accumulate(black_box(&w), black_box(&g)));
            }
            let rows = t1.elapsed().as_secs_f64();

            if rows > 0.0 {
                ratios.push(dot / rows);
            }
        }
        ratios.sort_by(f64::total_cmp);
        let median = ratios[ratios.len() / 2];
        println!(
            "  W({out_size}x{in_size})^T . g   median speedup of row-accumulate: {median:.2}x  \
             (worst |diff| {worst:.2e})"
        );
    }
}
