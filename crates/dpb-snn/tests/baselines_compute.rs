//! Every ANN baseline must compute a function of its input.
//!
//! Running a model twice on different inputs of the same shape and getting the
//! same answer means the forward pass is not reading its input. That was true
//! of 30 of the 44 models here: some returned `input.clone()`, most built a
//! zero hidden state and projected it, and several panicked on valid input.
//! Weights were allocated, reported in `num_parameters`, and never multiplied
//! by anything -- so an accuracy or energy comparison against them measured
//! nothing. Shape assertions cannot see this, which is why it survived.

use dpb_snn::baselines::*;

fn probe<M: ANNBaseline>(name: &str, build: impl Fn() -> M, shape: Vec<usize>) -> Option<String> {
    let a = Tensor::from_shape_fn(shape.clone(), |i| ((i % 13) as f32 - 6.0) * 0.1);
    let b = Tensor::from_shape_fn(shape.clone(), |i| ((i % 7) as f32 - 3.0) * 0.37 + 0.11);
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = build();
        let ya = model.forward(&a);
        let yb = model.forward(&b);
        (ya.data != yb.data, ya.data.iter().all(|v| v.is_finite()))
    }));
    match run {
        Ok((true, true)) => None,
        Ok((true, false)) => Some(format!("{name}: produced a non-finite output")),
        Ok((false, _)) => Some(format!("{name}: output does not depend on its input")),
        Err(_) => Some(format!(
            "{name}: panicked on a valid input of shape {shape:?}"
        )),
    }
}

#[test]
fn every_baseline_computes_a_function_of_its_input() {
    let mut failures: Vec<String> = Vec::new();

    let (c, l, out, sd) = (3usize, 64usize, 5usize, 7u64);
    let seq = 16usize;
    let d = 32usize;

    macro_rules! check {
        ($name:expr, $model:expr, $shape:expr) => {
            if let Some(failure) = probe($name, || $model, $shape) {
                failures.push(failure);
            }
        };
    }

    // 1-D convolutional: [batch, channels, length]
    check!("CNN1DSmall", CNN1DSmall::new(c, l, out, sd), vec![1, c, l]);
    check!(
        "CNN1DMedium",
        CNN1DMedium::new(c, l, out, sd),
        vec![1, c, l]
    );
    check!("CNN1DLarge", CNN1DLarge::new(c, l, out, sd), vec![1, c, l]);
    check!(
        "CNN1DResidual",
        CNN1DResidual::new(c, l, out, sd),
        vec![1, c, l]
    );
    check!(
        "CNN1DDilated",
        CNN1DDilated::new(c, l, out, sd),
        vec![1, c, l]
    );
    check!("TCN", TCN::new(c, l, out, sd), vec![1, c, l]);

    // 2-D convolutional: [batch, channels, height, width]
    check!(
        "CNN2DLeNet",
        CNN2DLeNet::new(c, (32, 32), out, sd),
        vec![1, c, 32, 32]
    );
    check!(
        "CNN2DVGG",
        CNN2DVGG::new(c, (32, 32), out, sd),
        vec![1, c, 32, 32]
    );
    check!(
        "CNN2DResNet",
        CNN2DResNet::new(c, (32, 32), out, sd),
        vec![1, c, 32, 32]
    );
    check!(
        "CNN2DMobileNet",
        CNN2DMobileNet::new(c, (32, 32), out, sd),
        vec![1, c, 32, 32]
    );

    // Fully connected: [batch, features]
    let f = 32usize;
    check!("MLP2Layer", MLP2Layer::new(f, 64, out, sd), vec![1, f]);
    check!("MLP3Layer", MLP3Layer::new(f, 64, 32, out, sd), vec![1, f]);
    check!(
        "MLP4Layer",
        MLP4Layer::new(f, &[64, 32, 16], out, sd),
        vec![1, f]
    );
    check!(
        "MLPDropout",
        MLPDropout::new(f, &[64, 32], out, 0.5, sd),
        vec![1, f]
    );
    check!(
        "MLPBatchNorm",
        MLPBatchNorm::new(f, &[64, 32], out, sd),
        vec![1, f]
    );
    check!(
        "MLPResidual",
        MLPResidual::new(f, &[64, 64], out, sd),
        vec![1, f]
    );
    check!(
        "MLPWideSingle",
        MLPWideSingle::new(f, 256, out, sd),
        vec![1, f]
    );
    check!("MLPDeep", MLPDeep::new(f, 32, 8, out, sd), vec![1, f]);

    // Recurrent: [batch, sequence, features]
    check!("SimpleRNN", SimpleRNN::new(d, 32, out, sd), vec![1, seq, d]);
    check!("LSTM", LSTM::new(d, 32, out, sd), vec![1, seq, d]);
    check!("BiLSTM", BiLSTM::new(d, 32, out, sd), vec![1, seq, d]);
    check!(
        "StackedLSTM",
        StackedLSTM::new(d, &[32, 32], out, sd),
        vec![1, seq, d]
    );
    check!("GRU", GRU::new(d, 32, out, sd), vec![1, seq, d]);
    check!("BiGRU", BiGRU::new(d, 32, out, sd), vec![1, seq, d]);
    check!(
        "StackedGRU",
        StackedGRU::new(d, &[32, 32], out, sd),
        vec![1, seq, d]
    );
    check!(
        "PeepholeLSTM",
        PeepholeLSTM::new(d, 32, out, sd),
        vec![1, seq, d]
    );
    check!(
        "AttentionLSTM",
        AttentionLSTM::new(d, 32, out, sd),
        vec![1, seq, d]
    );
    check!("IndRNN", IndRNN::new(d, 32, out, sd), vec![1, seq, d]);

    // Transformers: [batch, sequence, d_model]
    check!(
        "TransformerEncoder",
        TransformerEncoder::new(d, 4, 2, out, sd),
        vec![1, seq, d]
    );
    check!(
        "TransformerSmall",
        TransformerSmall::new(d, out, sd),
        vec![1, seq, d]
    );
    check!(
        "TransformerMedium",
        TransformerMedium::new(d, out, sd),
        vec![1, seq, d]
    );
    check!(
        "TransformerLarge",
        TransformerLarge::new(d, out, sd),
        vec![1, seq, d]
    );
    check!(
        "LinearTransformer",
        LinearTransformer::new(d, 2, out, sd),
        vec![1, seq, d]
    );
    check!("Performer", Performer::new(d, 2, out, sd), vec![1, seq, d]);
    check!("Informer", Informer::new(d, 2, out, sd), vec![1, seq, d]);
    check!(
        "Autoformer",
        Autoformer::new(d, 2, out, sd),
        vec![1, seq, d]
    );

    // Specialised
    check!("ECGNet", ECGNet::new(c, l, out, sd), vec![1, c, l]);
    check!("DeepGait", DeepGait::new(c, l, out, sd), vec![1, c, l]);
    check!("TremorNet", TremorNet::new(c, out, sd), vec![1, c, l]);
    check!("VoiceNet", VoiceNet::new(d, out, sd), vec![1, d, l]);
    check!(
        "MultimodalFusion",
        MultimodalFusion::new(&[16, 16], 32, out, sd),
        vec![1, 32]
    );
    check!(
        "AttentionFusion",
        AttentionFusion::new(&[16, 16], 32, out, sd),
        vec![1, 32]
    );
    check!("GraphNN", GraphNN::new(16, 32, 2, out, sd), vec![1, 8, 16]);
    check!(
        "HybridCNNRNN",
        HybridCNNRNN::new(c, l, out, sd),
        vec![1, c, l]
    );
    assert!(
        failures.is_empty(),
        "{} of 44 baselines do not compute a function of their input:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
