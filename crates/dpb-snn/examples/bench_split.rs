//! Splits convolution cost from neuron-update cost by varying only the kernel
//! width: the LIF work is identical across these, so the slope is the
//! convolution and the intercept is the neuron update plus overhead.

use dpb_snn::NeuronParams;
use dpb_snn::layers::{SpikingConv1d, SpikingLayer};
use dpb_snn::tensor::SpikeTensor;
use ndarray::Array3;
use std::time::Instant;

fn main() {
    let params = NeuronParams::default();
    let (batch, steps, in_ch, len) = (16, 32, 8, 256);
    let input = SpikeTensor::from_dense(Array3::from_elem((batch, steps, in_ch * len), 0.5), false);

    println!("  kernel   time       delta vs k=1");
    let mut base = 0.0;
    for k in [1usize, 3, 5, 9] {
        let pad = k / 2;
        let mut layer = SpikingConv1d::new(in_ch, 16, k, 1, pad, true, params.clone(), 1.0, false);
        layer.forward(&input).unwrap(); // warm
        let start = Instant::now();
        for _ in 0..3 {
            layer.reset_state();
            layer.forward(&input).unwrap();
        }
        let ms = start.elapsed().as_secs_f64() / 3.0 * 1000.0;
        if k == 1 {
            base = ms;
        }
        println!("  k={k:<6} {ms:>7.2} ms   {:>+8.2} ms", ms - base);
    }
    println!();
    println!("  k=1 is one MAC per (out_ch, pos, in_ch); the LIF update and all");
    println!("  loop overhead are the same at every k, so the k=1 time bounds");
    println!("  how much of the total is NOT the convolution arithmetic.");
}
