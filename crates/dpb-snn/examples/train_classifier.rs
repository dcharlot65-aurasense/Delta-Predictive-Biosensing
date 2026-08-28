//! Trains a small spiking classifier and reports whether it actually learns.
//!
//! Two classes, each driving a distinct pair of input channels. Success is not
//! "the loss went down" -- it is that the correct output unit ends up firing
//! more than the incorrect one, which is what a classifier has to do.

use dpb_snn::NeuronParams;
use dpb_snn::layers::SpikingLinear;
use dpb_snn::tensor::SpikeTensor;
use dpb_snn::training::{SGDOptimizer, SpikeCountLoss, SurrogateType, Trainer};
use ndarray::{Array2, Array3};

fn batch(steps: usize) -> (SpikeTensor, Array2<f32>) {
    let (batch, inputs) = (8, 4);
    let mut dense = Array3::zeros((batch, steps, inputs));
    let mut targets = Array2::zeros((batch, 2));

    for b in 0..batch {
        let class = b % 2;
        for t in 0..steps {
            dense[[b, t, class * 2]] = 1.0;
            dense[[b, t, class * 2 + 1]] = 1.0;
        }
        targets[[b, class]] = 0.6;
    }

    (SpikeTensor::from_dense(dense, false), targets)
}

/// Fraction of samples whose correct output unit outspikes the other.
fn accuracy(trainer: &mut Trainer, input: &SpikeTensor) -> f32 {
    let out = trainer.forward(input).unwrap().to_dense();
    let batch = out.shape()[0];
    let mut correct = 0;
    for b in 0..batch {
        let c0: f32 = out.slice(ndarray::s![b, .., 0]).sum();
        let c1: f32 = out.slice(ndarray::s![b, .., 1]).sum();
        let predicted = if c0 >= c1 { 0 } else { 1 };
        if predicted == b % 2 {
            correct += 1;
        }
    }
    correct as f32 / batch as f32
}

fn main() {
    let steps = 20;
    let (input, targets) = batch(steps);
    let params = NeuronParams::default();

    let layers = vec![
        SpikingLinear::new(4, 16, true, params.clone(), 1.0, false),
        SpikingLinear::new(16, 2, true, params, 1.0, false),
    ];

    let mut trainer = Trainer::new(
        layers,
        Box::new(SpikeCountLoss::new(1.0)),
        Box::new(SGDOptimizer::new(0.05, 0.9, 0.0)),
        SurrogateType::FastSigmoid,
    );

    let start_acc = accuracy(&mut trainer, &input);
    let first = trainer.evaluate(&input, &targets).unwrap();

    println!("  epoch   loss      grad norm");
    let mut last = first;
    for epoch in 0..200 {
        let report = trainer.train_step_reporting(&input, &targets).unwrap();
        last = report.loss;
        if epoch % 40 == 0 || epoch == 199 {
            println!(
                "  {epoch:>5}   {:>8.4}  {:>8.4}",
                report.loss, report.grad_norm
            );
        }
    }

    let end_acc = accuracy(&mut trainer, &input);
    println!();
    println!(
        "  loss      {first:.4} -> {last:.4}  ({:.1}% reduction)",
        100.0 * (first - last) / first.max(1e-9)
    );
    println!(
        "  accuracy  {:.0}% -> {:.0}%",
        start_acc * 100.0,
        end_acc * 100.0
    );
}
