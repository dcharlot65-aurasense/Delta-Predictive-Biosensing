//! Wall-clock baseline for the layer forward passes.
//!
//! Not a criterion benchmark: the point is a single reproducible number per
//! layer type to compare against after any optimisation, on shapes close to
//! what the training loop actually runs.

use dpb_snn::NeuronParams;
use dpb_snn::layers::{SpikingConv1d, SpikingConv2d, SpikingLayer, SpikingLinear, SpikingRNN};
use dpb_snn::tensor::SpikeTensor;
use ndarray::Array3;
use std::time::Instant;

fn timed(label: &str, elements: usize, mut run: impl FnMut()) {
    // One warm pass so allocation and first-touch costs are not counted.
    run();
    let reps = 5;
    let start = Instant::now();
    for _ in 0..reps {
        run();
    }
    let per = start.elapsed().as_secs_f64() / reps as f64;
    println!(
        "  {label:<28} {:>8.2} ms   {:>10.1} Kelem/s",
        per * 1000.0,
        elements as f64 / per / 1000.0
    );
}

fn main() {
    let params = NeuronParams::default();
    let (batch, steps) = (16, 32);

    println!("  batch {batch}, {steps} steps\n");

    // Dense layer: 512 -> 512.
    let mut linear = SpikingLinear::new(512, 512, true, params.clone(), 1.0, false);
    let input = SpikeTensor::from_dense(Array3::from_elem((batch, steps, 512), 0.5), false);
    timed("SpikingLinear 512->512", batch * steps * 512, || {
        linear.reset_state();
        linear.forward(&input).unwrap();
    });

    // Recurrent layer: 256 hidden.
    let mut rnn = SpikingRNN::new(256, 256, true, params.clone(), 1.0, false);
    let rnn_in = SpikeTensor::from_dense(Array3::from_elem((batch, steps, 256), 0.5), false);
    timed("SpikingRNN 256->256", batch * steps * 256, || {
        rnn.reset_state();
        rnn.forward(&rnn_in).unwrap();
    });

    // 1-D convolution over a length-256 signal, 8 -> 16 channels.
    let mut conv1 = SpikingConv1d::new(8, 16, 5, 1, 2, true, params.clone(), 1.0, false);
    let c1_in = SpikeTensor::from_dense(Array3::from_elem((batch, steps, 8 * 256), 0.5), false);
    timed("SpikingConv1d 8->16 k5", batch * steps * 16 * 256, || {
        conv1.reset_state();
        conv1.forward(&c1_in).unwrap();
    });

    // 2-D convolution on 32x32, 3 -> 16 channels.
    let mut conv2 = SpikingConv2d::new(
        3,
        16,
        (3, 3),
        (1, 1),
        (1, 1),
        true,
        params.clone(),
        1.0,
        false,
    );
    conv2.set_input_shape(32, 32);
    let c2_in = SpikeTensor::from_dense(Array3::from_elem((4, 8, 3 * 32 * 32), 0.5), false);
    timed("SpikingConv2d 3->16 32x32", 4 * 8 * 16 * 32 * 32, || {
        conv2.reset_state();
        conv2.forward(&c2_in).unwrap();
    });
}
