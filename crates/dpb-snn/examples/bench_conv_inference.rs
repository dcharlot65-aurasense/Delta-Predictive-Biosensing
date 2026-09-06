//! Inference forward vs the trace-building forward, interleaved.
//!
//! `forward_recording` is exactly what `SpikingLayer::forward` used to call, so
//! this is a true before/after on one binary: the two run alternately and share
//! whatever load the machine is under.

use dpb_snn::NeuronParams;
use dpb_snn::layers::{SpikingConv1d, SpikingConv2d, SpikingLayer};
use dpb_snn::tensor::SpikeTensor;
use ndarray::Array3;
use std::time::{Duration, Instant};

fn main() {
    let params = NeuronParams::default();
    let reps = 5;

    let (b1, s1) = (16, 32);
    let mut conv1 = SpikingConv1d::new(8, 16, 5, 1, 2, true, params.clone(), 1.0, false);
    let in1 = SpikeTensor::from_dense(Array3::from_elem((b1, s1, 8 * 256), 0.5), false);

    let (b2, s2) = (4, 8);
    let mut conv2 = SpikingConv2d::new(3, 16, (3, 3), (1, 1), (1, 1), true, params, 1.0, false);
    conv2.set_input_shape(32, 32);
    let in2 = SpikeTensor::from_dense(Array3::from_elem((b2, s2, 3 * 32 * 32), 0.5), false);

    // Warm both paths of both layers.
    for _ in 0..2 {
        conv1.reset_state();
        conv1.forward(&in1).unwrap();
        conv1.reset_state();
        conv1.forward_recording(&in1).unwrap();
        conv2.reset_state();
        conv2.forward(&in2).unwrap();
        conv2.reset_state();
        conv2.forward_recording(&in2).unwrap();
    }

    let (mut t1_new, mut t1_old) = (Duration::ZERO, Duration::ZERO);
    let (mut t2_new, mut t2_old) = (Duration::ZERO, Duration::ZERO);

    for _ in 0..reps {
        conv1.reset_state();
        let start = Instant::now();
        std::hint::black_box(conv1.forward(&in1).unwrap());
        t1_new += start.elapsed();

        conv1.reset_state();
        let start = Instant::now();
        std::hint::black_box(conv1.forward_recording(&in1).unwrap());
        t1_old += start.elapsed();

        conv2.reset_state();
        let start = Instant::now();
        std::hint::black_box(conv2.forward(&in2).unwrap());
        t2_new += start.elapsed();

        conv2.reset_state();
        let start = Instant::now();
        std::hint::black_box(conv2.forward_recording(&in2).unwrap());
        t2_old += start.elapsed();
    }

    let ms = |d: Duration| d.as_secs_f64() / reps as f64 * 1000.0;
    println!(
        "\n  {:<30} {:>10} {:>10} {:>8}",
        "layer", "was", "now", "speedup"
    );
    println!(
        "  {:<30} {:>7.2}ms {:>7.2}ms {:>7.1}x",
        format!("SpikingConv1d 8->16 k5 (b{b1} s{s1})"),
        ms(t1_old),
        ms(t1_new),
        ms(t1_old) / ms(t1_new)
    );
    println!(
        "  {:<30} {:>7.2}ms {:>7.2}ms {:>7.1}x\n",
        format!("SpikingConv2d 3->16 32x32 (b{b2} s{s2})"),
        ms(t2_old),
        ms(t2_new),
        ms(t2_old) / ms(t2_new)
    );
}
