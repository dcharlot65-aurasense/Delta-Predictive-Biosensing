//! Is the 1-D convolution inner loop paying for arithmetic or for views?
//!
//! Reproduces the exact access pattern of `SpikingConv1d::convolve` two ways --
//! through ndarray slices, and over raw `&[f32]` -- and interleaves them so a
//! busy machine inflates both equally. If the two land in the same place the
//! views are free and there is nothing to win here.

use ndarray::{Array1, Array2, s};
use std::time::{Duration, Instant};

const IN_CH: usize = 8;
const OUT_CH: usize = 16;
const K: usize = 5;
const LEN: usize = 256;

fn ndarray_way(kernel: &ndarray::Array3<f32>, input: &Array1<f32>, out: &mut Array2<f32>) {
    out.fill(0.0);
    for k in 0..K {
        // Padding 2, stride 1: every tap covers the whole output here.
        let (op_lo, op_hi, ip_start) = (0usize, LEN, 0usize);
        let count = op_hi - op_lo;
        for ic in 0..IN_CH {
            let base = ic * LEN + ip_start;
            let end = base + count;
            let source = input.slice(s![base..end; 1]);
            for oc in 0..OUT_CH {
                let weight = kernel[[oc, ic, k]];
                if weight == 0.0 {
                    continue;
                }
                out.slice_mut(s![oc, op_lo..op_hi])
                    .scaled_add(weight, &source);
            }
        }
    }
}

fn slice_way(kernel: &ndarray::Array3<f32>, input: &Array1<f32>, out: &mut Array2<f32>) {
    out.fill(0.0);
    let input = input.as_slice().expect("contiguous");
    let kern = kernel.as_slice().expect("contiguous");
    let out = out.as_slice_mut().expect("contiguous");
    for k in 0..K {
        let (op_lo, op_hi, ip_start) = (0usize, LEN, 0usize);
        let count = op_hi - op_lo;
        for ic in 0..IN_CH {
            let base = ic * LEN + ip_start;
            let source = &input[base..base + count];
            for oc in 0..OUT_CH {
                let weight = kern[(oc * IN_CH + ic) * K + k];
                if weight == 0.0 {
                    continue;
                }
                let row = &mut out[oc * LEN + op_lo..oc * LEN + op_hi];
                for (dst, &src) in row.iter_mut().zip(source) {
                    *dst += weight * src;
                }
            }
        }
    }
}

fn main() {
    let kernel = ndarray::Array3::from_shape_fn((OUT_CH, IN_CH, K), |(o, i, k)| {
        ((o * 31 + i * 7 + k) % 13) as f32 * 0.05 - 0.3
    });
    let input = Array1::from_shape_fn(IN_CH * LEN, |i| (i % 17) as f32 * 0.06);
    let mut out_a = Array2::<f32>::zeros((OUT_CH, LEN));
    let mut out_b = Array2::<f32>::zeros((OUT_CH, LEN));

    // Same answer, or the comparison is meaningless.
    ndarray_way(&kernel, &input, &mut out_a);
    slice_way(&kernel, &input, &mut out_b);
    let worst = out_a
        .iter()
        .zip(out_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 1e-5, "the two paths disagree by {worst}");
    println!("\n  both paths agree (max abs diff {worst:.2e})");

    let (reps, inner) = (5, 512);
    let (mut ta, mut tb) = (Duration::ZERO, Duration::ZERO);
    for _ in 0..reps {
        let start = Instant::now();
        for _ in 0..inner {
            ndarray_way(&kernel, &input, &mut out_a);
        }
        ta += start.elapsed();

        let start = Instant::now();
        for _ in 0..inner {
            slice_way(&kernel, &input, &mut out_b);
        }
        tb += start.elapsed();
    }
    let (a, b) = (
        ta.as_secs_f64() / reps as f64,
        tb.as_secs_f64() / reps as f64,
    );
    println!("  ndarray slices  {:>7.2} ms", a * 1000.0);
    println!("  raw &[f32]      {:>7.2} ms   {:.1}x\n", b * 1000.0, a / b);
}
