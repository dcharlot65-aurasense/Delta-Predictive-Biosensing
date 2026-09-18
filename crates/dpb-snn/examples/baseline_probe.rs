//! Which ANN baselines actually read their input?
//!
//! Runs every baseline twice on different inputs of the same shape. A forward
//! pass whose output does not move is not computing a function of its input.

use dpb_snn::baselines::*;

fn probe<M: ANNBaseline>(name: &str, build: impl Fn() -> M, shape: Vec<usize>) -> bool {
    let a = Tensor::from_shape_fn(shape.clone(), |i| ((i % 13) as f32 - 6.0) * 0.1);
    let b = Tensor::from_shape_fn(shape.clone(), |i| ((i % 7) as f32 - 3.0) * 0.37 + 0.11);
    // Construction is inside the guard too: a constructor assertion would
    // otherwise kill the whole sweep.
    let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let model = build();
        let ya = model.forward(&a);
        let yb = model.forward(&b);
        (ya.data != yb.data, ya.shape.clone())
    }));
    match run {
        Ok((true, _)) => true,
        Ok((false, out)) => {
            println!("  IGNORES INPUT  {name:22} in={shape:?} out={out:?}");
            false
        }
        Err(_) => {
            println!("  PANICS         {name:22} in={shape:?}");
            false
        }
    }
}

fn main() {
    let (c, l, out, sd) = (3usize, 64usize, 5usize, 7u64);
    let seq = 16usize;
    let d = 32usize;
    let mut bad = 0;
    let mut total = 0;

    macro_rules! check {
        ($name:expr, $model:expr, $shape:expr) => {
            total += 1;
            if !probe($name, || $model, $shape) {
                bad += 1;
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

    println!("\n{bad} of {total} baselines ignore their input");
}
