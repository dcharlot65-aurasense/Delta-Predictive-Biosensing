//! Every SNN architecture and layer must compute a function of its input.
//!
//! This is the check that found 30 of the 44 ANN baselines returning the same
//! answer for any input, and the spiking side is the part of the library that
//! matters, so it gets the same treatment rather than being assumed sound.
//!
//! Spiking output needs more care than a dense one. Spikes are binary and
//! sparse, so two different inputs can honestly produce the same all-zero
//! train if neither drives the membrane past threshold -- an output that is
//! silent everywhere proves nothing either way. Each model is therefore driven
//! hard enough to fire, and silence is reported as its own failure rather than
//! being mistaken for a difference.
//!
//! State is reset between the two runs. Without that a stateful layer would
//! differ on the second call whatever it was fed, which would make this test
//! pass for the wrong reason.

use dpb_snn::architectures::*;
use dpb_snn::layers::*;
use dpb_snn::tensor::SpikeTensor;
use dpb_snn::{NeuronParams, SNNConfig};
use ndarray::Array3;

/// Two spike trains that are dense enough to drive firing and different enough
/// to tell apart.
///
/// Density matters more than it looks. A sparse drive -- a third of the inputs
/// active over six steps -- produces no output spikes at all with the default
/// neuron: the membrane leaks faster than it fills, so both trains come back
/// silent and compare equal. A check built on that reports every layer as
/// ignoring its input, which is a statement about the stimulus, not the layer.
/// About 90% of inputs active over 32 steps fires every layer here.
fn drives(shape: (usize, usize, usize)) -> (SpikeTensor, SpikeTensor) {
    let a = Array3::from_shape_fn(shape, |(b, t, i)| {
        if (b + t * 3 + i * 7) % 11 == 0 {
            0.0
        } else {
            1.0
        }
    });
    let b = Array3::from_shape_fn(shape, |(b, t, i)| {
        if (b * 5 + t + i * 3) % 9 == 0 {
            0.0
        } else {
            1.0
        }
    });
    (
        SpikeTensor::from_dense(a, false),
        SpikeTensor::from_dense(b, false),
    )
}

/// Replace random initialisation with fixed, mildly positive weights.
///
/// These layers initialise from an unseeded `rand::rng()` and expose no seed,
/// so whether a given neuron fires changes from run to run: an earlier version
/// of this test failed ten times in twenty-five, naming a different model each
/// time. The particular values matter far less than their being identical on
/// every run, and keeping them positive makes firing reliable rather than
/// lucky.
///
/// `parameters_mut` does not reach a convolution's kernel -- it returns
/// `Array2` and a kernel is rank 3 or 4, so both conv layers hand back an
/// empty vector -- hence the separate `kernel` assignments at the call sites.
fn pin(matrices: Vec<&mut ndarray::Array2<f32>>) {
    for (m, matrix) in matrices.into_iter().enumerate() {
        let cols = matrix.ncols().max(1);
        for (idx, w) in matrix.iter_mut().enumerate() {
            let (r, c) = (idx / cols, idx % cols);
            *w = pinned_weight(r, c, m);
        }
    }
}

/// Mixed sign, mean near zero.
///
/// All-positive weights are as useless here as none at all: with a low
/// threshold every neuron fires at every step for any input, the output
/// saturates, and two different drives again produce the same answer. Firing
/// has to be partial for the comparison to mean anything.
fn pinned_weight(row: usize, col: usize, matrix: usize) -> f32 {
    (((row * 7 + col * 13 + matrix * 5) % 11) as f32 / 5.0) - 1.0
}

fn check_architecture(
    name: &str,
    model: &mut dyn SNNArchitecture,
    shape: (usize, usize, usize),
    failures: &mut Vec<String>,
) {
    let (a, b) = drives(shape);

    model.reset();
    let ya = match model.forward(&a) {
        Ok(out) => out.to_dense(),
        Err(e) => {
            failures.push(format!("{name}: forward failed on a valid input: {e}"));
            return;
        }
    };
    model.reset();
    let yb = match model.forward(&b) {
        Ok(out) => out.to_dense(),
        Err(e) => {
            failures.push(format!("{name}: forward failed on a valid input: {e}"));
            return;
        }
    };

    if !ya.iter().all(|v| v.is_finite()) {
        failures.push(format!("{name}: produced a non-finite value"));
        return;
    }
    // Silence is not evidence of computation, and it is not evidence against
    // it either -- but a model that never fires for any drive is not usable,
    // so it is called out separately from the comparison below.
    if ya.sum() == 0.0 && yb.sum() == 0.0 {
        failures.push(format!(
            "{name}: silent for both drives, so nothing can be concluded and nothing spikes"
        ));
        return;
    }
    if ya == yb {
        failures.push(format!("{name}: output does not depend on its input"));
    }
}

fn check_layer(
    name: &str,
    layer: &mut dyn SpikingLayer,
    shape: (usize, usize, usize),
    failures: &mut Vec<String>,
) {
    let (a, b) = drives(shape);

    layer.reset_state();
    let ya = match layer.forward(&a) {
        Ok(out) => out.to_dense(),
        Err(e) => {
            failures.push(format!("{name}: forward failed on a valid input: {e}"));
            return;
        }
    };
    layer.reset_state();
    let yb = match layer.forward(&b) {
        Ok(out) => out.to_dense(),
        Err(e) => {
            failures.push(format!("{name}: forward failed on a valid input: {e}"));
            return;
        }
    };

    if !ya.iter().all(|v| v.is_finite()) {
        failures.push(format!("{name}: produced a non-finite value"));
        return;
    }
    // Same reasoning as above: two silent outputs are equal for a reason that
    // says nothing about whether the layer reads its input.
    if ya.sum() == 0.0 && yb.sum() == 0.0 {
        failures.push(format!(
            "{name}: silent for both drives, so nothing can be concluded and nothing spikes"
        ));
        return;
    }
    if ya == yb {
        failures.push(format!("{name}: output does not depend on its input"));
    }
}

/// Neuron parameters tuned so that firing is partial.
///
/// Whether a neuron crosses the default threshold depends on the weights it
/// was given, and these layers initialise randomly -- `SpikingLSTM` fired on
/// one run of this test and was silent on the next. That makes a firing-rate
/// dependency a source of flakiness, not a property worth asserting here: the
/// question is whether output tracks input, so the threshold is lowered until
/// firing is reliable regardless of the draw.
fn excitable() -> NeuronParams {
    NeuronParams {
        v_threshold: 0.5,
        ..Default::default()
    }
}

/// Architectures are deeper than single layers, and each stage attenuates,
/// so they need a lower threshold than the layer probe to fire at all. The
/// reservoir in `LiquidStateMachine` was intermittently silent at 0.5.
fn config(num_steps: usize) -> SNNConfig {
    SNNConfig {
        num_steps,
        neuron_params: NeuronParams {
            v_threshold: 0.05,
            ..Default::default()
        },
        ..Default::default()
    }
}

#[test]
fn every_architecture_computes_a_function_of_its_input() {
    let mut failures = Vec::new();
    let (batch, steps) = (2usize, 32usize);

    let mut ff = FeedforwardSNN::mlp(16, vec![24, 24], 5, config(steps)).expect("valid MLP");
    pin(ff.parameters_mut());
    check_architecture("FeedforwardSNN", &mut ff, (batch, steps, 16), &mut failures);

    let mut rnn = RecurrentSNN::new(16, vec![24], 5, config(steps)).expect("valid RNN");
    pin(rnn.parameters_mut());
    check_architecture("RecurrentSNN", &mut rnn, (batch, steps, 16), &mut failures);

    let mut lsm = LiquidStateMachine::new(16, 48, 5, 0.9, config(steps));
    pin(lsm.parameters_mut());
    check_architecture(
        "LiquidStateMachine",
        &mut lsm,
        (batch, steps, 16),
        &mut failures,
    );

    let mut transformer = SpikingTransformer::new(16, 16, 2, 1, 5, config(steps));
    pin(transformer.parameters_mut());
    check_architecture(
        "SpikingTransformer",
        &mut transformer,
        (batch, steps, 16),
        &mut failures,
    );

    // Note for this one: `pin` reaches only its two classifier matrices. A
    // convolution's kernel is rank 3 or 4 while the trait yields `Array2`, so
    // both conv layers return nothing from `parameters_mut`, and its kernels
    // stay as the unseeded generator left them. The threshold above is low
    // enough that the stack fires whatever they are.
    let mut conv = ConvolutionalSNN::new(1, 5, config(steps));
    // One warm-up pass first. `ensure_fc_head` sizes the classifier from
    // whatever the convolutional stack produces, so it replaces `fc_layers`
    // with freshly random weights on the first forward -- anything pinned
    // before that is thrown away, which is why this check stayed
    // intermittently silent after the kernels were already fixed.
    {
        let (warm, _) = drives((batch, steps, 16 * 16));
        conv.reset();
        let _ = conv.forward(&warm);
    }

    pin(conv.parameters_mut());
    // Reach the kernels and biases directly, since `parameters_mut` cannot:
    // it yields `Array2`, and a kernel is rank 4 while a bias is rank 1. Both
    // are randomly initialised, and leaving either of them to the draw left
    // this check silent on roughly one run in forty.
    for (n, layer) in conv.conv_layers.iter_mut().enumerate() {
        layer
            .kernel
            .iter_mut()
            .enumerate()
            .for_each(|(i, w)| *w = pinned_weight(i, 0, n + 3));
        if let Some(bias) = layer.bias.as_mut() {
            bias.fill(0.0);
        }
    }
    for layer in conv.fc_layers.iter_mut() {
        if let Some(bias) = layer.bias.as_mut() {
            bias.fill(0.0);
        }
    }
    check_architecture(
        "ConvolutionalSNN",
        &mut conv,
        (batch, steps, 16 * 16),
        &mut failures,
    );

    let adjacency = ndarray::Array2::from_shape_fn((6, 6), |(i, j)| {
        if i == j || i.abs_diff(j) == 1 {
            1.0
        } else {
            0.0
        }
    });
    let mut gcn = SpikingGCN::new(6, 8, vec![12], 5, adjacency, config(steps));
    pin(gcn.parameters_mut());
    check_architecture("SpikingGCN", &mut gcn, (batch, steps, 6 * 8), &mut failures);

    assert!(
        failures.is_empty(),
        "{} architecture(s) do not compute a function of their input:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn every_layer_computes_a_function_of_its_input() {
    let mut failures = Vec::new();
    let (batch, steps) = (2usize, 32usize);
    let params = excitable();

    let mut linear = SpikingLinear::new(16, 8, true, params.clone(), 1.0, false);
    pin(linear.parameters_mut());
    check_layer(
        "SpikingLinear",
        &mut linear,
        (batch, steps, 16),
        &mut failures,
    );

    let mut rnn = SpikingRNN::new(16, 12, true, params.clone(), 1.0, false);
    pin(rnn.parameters_mut());
    check_layer("SpikingRNN", &mut rnn, (batch, steps, 16), &mut failures);

    let mut lstm = SpikingLSTM::new(16, 12, params.clone(), 1.0, false);
    pin(lstm.parameters_mut());
    check_layer("SpikingLSTM", &mut lstm, (batch, steps, 16), &mut failures);

    let mut conv1 = SpikingConv1d::new(2, 4, 3, 1, 1, true, params.clone(), 1.0, false);
    conv1
        .kernel
        .iter_mut()
        .enumerate()
        .for_each(|(i, w)| *w = pinned_weight(i, 0, 1));
    check_layer(
        "SpikingConv1d",
        &mut conv1,
        (batch, steps, 2 * 16),
        &mut failures,
    );

    let mut conv2 = SpikingConv2d::new(
        2,
        4,
        (3, 3),
        (1, 1),
        (1, 1),
        true,
        params.clone(),
        1.0,
        false,
    );
    conv2.set_input_shape(8, 8);
    conv2
        .kernel
        .iter_mut()
        .enumerate()
        .for_each(|(i, w)| *w = pinned_weight(i, 0, 2));
    check_layer(
        "SpikingConv2d",
        &mut conv2,
        (batch, steps, 2 * 8 * 8),
        &mut failures,
    );

    let mut attention = SpikingAttention::new(16, 4, params.clone(), 1.0, false);
    pin(attention.parameters_mut());
    check_layer(
        "SpikingAttention",
        &mut attention,
        (batch, steps, 16),
        &mut failures,
    );

    assert!(
        failures.is_empty(),
        "{} layer(s) do not compute a function of their input:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// A seeded run of a whole architecture must be reproducible.
///
/// This is what `manual_seed` is for. An architecture builds its layers
/// internally, so without a thread-level seed each one would need a seed
/// threaded through its constructor, and until now none of them could be
/// repeated at all -- while every ANN baseline in the crate already took one.
/// Comparing the output of a forward pass, rather than the weights, checks the
/// property a user actually depends on.
#[test]
fn seeded_architectures_reproduce_their_output() {
    let steps = 16;
    let (drive, _) = drives((2, steps, 16));

    let run = |seed: u64| {
        manual_seed(seed);
        let mut net = FeedforwardSNN::mlp(16, vec![24, 24], 5, config(steps)).expect("valid MLP");
        net.reset();
        net.forward(&drive).expect("forward").to_dense()
    };

    let first = run(2024);
    let again = run(2024);
    assert_eq!(
        first, again,
        "the same seed produced a different network, so a run cannot be repeated"
    );

    let other = run(2025);
    assert_ne!(
        first, other,
        "a different seed produced the same network, so the seed is ignored"
    );

    clear_manual_seed();
}
