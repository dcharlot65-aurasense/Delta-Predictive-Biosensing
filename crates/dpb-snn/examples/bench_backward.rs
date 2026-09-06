//! End-to-end A/B of `SpikingLinear::backward` against the form it replaced.
//!
//! The old inner loop is reimplemented here over the same public fields, so
//! both run in one process and can be interleaved: on a loaded machine the
//! absolute times drift, but a pair sees the same contention.

use dpb_snn::NeuronParams;
use dpb_snn::layers::{SpikingLayer, SpikingLinear};
use ndarray::{Array1, Array2, Array3, s};
use std::hint::black_box;
use std::time::Instant;

/// The previous body, verbatim apart from writing into locals: the input
/// gradient came from a transposed view.
fn backward_via_transpose(
    weights: &Array2<f32>,
    input: &Array3<f32>,
    output_grad: &Array3<f32>,
) -> Array3<f32> {
    let (batch, steps, _) = (input.shape()[0], input.shape()[1], input.shape()[2]);
    let (output_size, input_size) = (weights.shape()[0], weights.shape()[1]);
    let mut weight_grad = Array2::<f32>::zeros((output_size, input_size));
    let mut input_grad = Array3::zeros(input.raw_dim());

    for t in (0..steps).rev() {
        for b in 0..batch {
            let out_grad_t = output_grad.slice(s![b, t, ..]);
            let input_t = input.slice(s![b, t, ..]);
            for i in 0..output_size {
                for j in 0..input_size {
                    weight_grad[[i, j]] += out_grad_t[i] * input_t[j];
                }
            }
            let in_grad = weights.t().dot(&out_grad_t);
            input_grad.slice_mut(s![b, t, ..]).assign(&in_grad);
        }
    }
    black_box(weight_grad);
    input_grad
}

fn main() {
    for (batch, steps, in_size, out_size) in
        [(8usize, 16usize, 256usize, 256usize), (16, 32, 512, 512)]
    {
        let params = NeuronParams::default();
        let mut layer = SpikingLinear::new(in_size, out_size, true, params, 1.0, false);
        let input = Array3::from_shape_fn((batch, steps, in_size), |(b, t, i)| {
            (((b + t + i) % 5) as f32) * 0.2
        });
        let grad = Array3::from_shape_fn((batch, steps, out_size), |(b, t, n)| {
            (((b + t + n) % 3) as f32) * 0.1
        });

        // Agreement first.
        layer.zero_grad();
        let new_grad = layer.backward(&input, &grad).unwrap();
        let old_grad = backward_via_transpose(&layer.weights, &input, &grad);
        let worst = new_grad
            .iter()
            .zip(old_grad.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(worst < 1e-3, "input gradients differ by {worst}");

        const PAIRS: usize = 15;
        let mut ratios = Vec::new();
        for _ in 0..PAIRS {
            let t0 = Instant::now();
            black_box(backward_via_transpose(
                black_box(&layer.weights),
                black_box(&input),
                black_box(&grad),
            ));
            let old = t0.elapsed().as_secs_f64();

            layer.zero_grad();
            let t1 = Instant::now();
            black_box(layer.backward(black_box(&input), black_box(&grad)).unwrap());
            let new = t1.elapsed().as_secs_f64();

            if new > 0.0 {
                ratios.push(old / new);
            }
        }
        ratios.sort_by(f64::total_cmp);
        println!(
            "  backward {batch}x{steps} {in_size}->{out_size}   median speedup {:.2}x   \
             (worst |diff| {worst:.2e})",
            ratios[ratios.len() / 2]
        );
        let _ = Array1::<f32>::zeros(1);
    }
}
