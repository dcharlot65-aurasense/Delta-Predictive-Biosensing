//! How much of a forward pass is the LIF update?
//!
//! The layer forwards and the bare LIF loop are timed *interleaved* in one
//! process, so both see the same machine load. Absolute numbers move with the
//! load average; the share does not, and the share is the thing that decides
//! whether the LIF update is worth optimising at all.

use dpb_snn::NeuronParams;
use dpb_snn::layers::{NeuronState, SpikingConv1d, SpikingConv2d, SpikingLayer, SpikingLinear};
use dpb_snn::tensor::SpikeTensor;
use ndarray::{Array1, Array3};
use std::time::{Duration, Instant};

/// A layer to measure: label, neurons its LIF runs over, LIF calls per forward,
/// and the forward pass itself.
type Case = (&'static str, usize, usize, Box<dyn FnMut()>);

/// Input current chosen so a plausible fraction of neurons cross threshold and
/// then sit refractory -- an all-subthreshold input would skip the spike branch
/// entirely and flatter the loop.
fn drive(n: usize) -> Array1<f32> {
    Array1::from_shape_fn(n, |i| if i % 7 == 0 { 1.4 } else { 0.15 })
}

fn lif_only(neurons: usize, calls: usize, params: &NeuronParams) -> Duration {
    let mut state = NeuronState::new(neurons, false);
    let input = drive(neurons);
    let start = Instant::now();
    for _ in 0..calls {
        std::hint::black_box(state.update_lif_recording(&input, params, 1.0));
    }
    start.elapsed()
}

fn main() {
    let params = NeuronParams::default();
    let (batch, steps) = (16, 32);
    let calls = batch * steps;
    let reps = 5;

    let mut linear = SpikingLinear::new(512, 512, true, params.clone(), 1.0, false);
    let lin_in = SpikeTensor::from_dense(Array3::from_elem((batch, steps, 512), 0.5), false);

    let mut conv1 = SpikingConv1d::new(8, 16, 5, 1, 2, true, params.clone(), 1.0, false);
    let c1_in = SpikeTensor::from_dense(Array3::from_elem((batch, steps, 8 * 256), 0.5), false);

    // The 2-D case runs a smaller batch: it is the most expensive layer here,
    // and this matches the shape bench_forward already reports for it.
    let (c2_batch, c2_steps) = (4, 8);
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
    let c2_in = SpikeTensor::from_dense(
        Array3::from_elem((c2_batch, c2_steps, 3 * 32 * 32), 0.5),
        false,
    );

    let mut cases: Vec<Case> = vec![
        (
            "SpikingLinear 512->512",
            512,
            calls,
            Box::new(move || {
                linear.reset_state();
                std::hint::black_box(linear.forward(&lin_in).unwrap());
            }),
        ),
        (
            "SpikingConv1d 8->16 k5 L256",
            16 * 256,
            calls,
            Box::new(move || {
                conv1.reset_state();
                std::hint::black_box(conv1.forward(&c1_in).unwrap());
            }),
        ),
        (
            "SpikingConv2d 3->16 32x32",
            16 * 32 * 32,
            c2_batch * c2_steps,
            Box::new(move || {
                conv2.reset_state();
                std::hint::black_box(conv2.forward(&c2_in).unwrap());
            }),
        ),
    ];

    // Warm every path before any timing.
    for (_, neurons, n_calls, fwd) in cases.iter_mut() {
        fwd();
        lif_only(*neurons, *n_calls, &params);
    }

    let mut fwd_total = vec![Duration::ZERO; cases.len()];
    let mut lif_total = vec![Duration::ZERO; cases.len()];

    for _ in 0..reps {
        for (idx, (_, neurons, n_calls, fwd)) in cases.iter_mut().enumerate() {
            let start = Instant::now();
            fwd();
            fwd_total[idx] += start.elapsed();
            lif_total[idx] += lif_only(*neurons, *n_calls, &params);
        }
    }

    println!(
        "\n  batch {batch}, {steps} steps, {reps} reps, interleaved\n\n  {:<28} {:>9} {:>9} {:>8}",
        "layer", "forward", "lif", "share"
    );
    for (idx, (label, _, _, _)) in cases.iter().enumerate() {
        let fwd = fwd_total[idx].as_secs_f64() / reps as f64;
        let lif = lif_total[idx].as_secs_f64() / reps as f64;
        println!(
            "  {label:<28} {:>7.2}ms {:>7.2}ms {:>7.1}%",
            fwd * 1000.0,
            lif * 1000.0,
            lif / fwd * 100.0
        );
    }
    println!();
}
