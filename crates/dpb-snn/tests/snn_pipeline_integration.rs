//! Integration tests for SNN training and inference pipelines
//!
//! Tests complete workflows from encoding through training, inference,
//! calibration, and explainability.

use dpb_snn::architectures::SNNArchitecture;
use dpb_snn::tensor::SpikeTensor;
use dpb_snn::*;
use ndarray::Array2;

/// Create a simple synthetic spike tensor for testing
fn create_test_spike_tensor(
    batch_size: usize,
    num_neurons: usize,
    num_timesteps: usize,
) -> SpikeTensor {
    // Create dense tensor with sparse spike patterns
    let mut dense = ndarray::Array3::zeros((batch_size, num_timesteps, num_neurons));

    // Create simple spike patterns
    for b in 0..batch_size {
        for n in 0..num_neurons {
            // Sparse random spikes
            for t in (n * 10..num_timesteps).step_by(20) {
                if t < num_timesteps {
                    dense[[b, t, n]] = 1.0;
                }
            }
        }
    }

    SpikeTensor::from_dense(dense, false)
}

#[test]
fn test_encoder_to_snn_to_decoder() {
    println!("Testing complete pipeline: Encoder → SNN → Decoder");

    // Configuration
    let batch_size = 4;
    let num_input_neurons = 64;
    let num_hidden = 32;
    let num_output = 10;
    let num_timesteps = 100;

    // Step 1: Create input spike tensor (simulating rate encoder output)
    let input_spikes = create_test_spike_tensor(batch_size, num_input_neurons, num_timesteps);

    println!("  Input tensor shape: {:?}", input_spikes.shape());
    // `SpikeTensor` exposes no per-batch spike list; count from the dense form.
    let dense_input = input_spikes.to_dense();
    let input_spike_count: usize = dense_input.iter().filter(|&&v| v > 0.0).count();
    println!("  Total input spikes: {}", input_spike_count);

    // Step 2: Build simple feedforward SNN
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(
        vec![num_input_neurons, num_hidden, num_output],
        snn_config,
        true, // use_bias
    )
    .expect("Failed to create SNN");

    println!(
        "  Created SNN: {} → {} → {}",
        num_input_neurons, num_hidden, num_output
    );

    // Step 3: Forward pass through SNN
    let output_spikes = snn
        .forward(&input_spikes)
        .expect("Failed to run SNN forward pass");

    println!("  Output tensor shape: {:?}", output_spikes.shape());

    // Verify output shape
    // shape() is (batch_size, num_steps, num_neurons) -- a tuple, and the
    // step count precedes the neuron count.
    assert_eq!(
        output_spikes.shape(),
        (batch_size, num_timesteps, num_output)
    );

    // Step 4: Decode using rate decoder
    let rate_decoder = SpikeRateDecoder::new(num_output, None, false);
    let decoded = rate_decoder
        .decode(&output_spikes)
        .expect("Failed to decode spikes");

    println!("  Decoded output shape: {:?}", decoded.shape());
    assert_eq!(decoded.shape(), &[batch_size, num_output]);

    // Step 5: Verify output values are reasonable
    for b in 0..batch_size {
        for o in 0..num_output {
            let rate = decoded[[b, o]];
            assert!(
                (0.0..=1.0).contains(&rate),
                "Decoded rate should be normalized: got {}",
                rate
            );
        }
    }

    println!("  ✓ Pipeline completed successfully");
}

#[test]
fn test_calibrated_snn_predictions() {
    println!("Testing SNN calibration pipeline");

    let batch_size = 100;
    let num_classes = 5;
    let num_timesteps = 50;
    let num_input = 32;

    // Step 1: Create SNN and generate predictions
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(vec![num_input, 20, num_classes], snn_config, true)
        .expect("Failed to create SNN");

    // Generate batch of predictions
    let mut all_logits = Vec::new();
    let mut all_labels = Vec::new();

    for _ in 0..batch_size {
        let input = create_test_spike_tensor(1, num_input, num_timesteps);
        let output = snn.forward(&input).expect("Forward pass failed");

        let decoder = SpikeRateDecoder::new(num_classes, None, false);
        let logits = decoder.decode(&output).expect("Decoding failed");

        all_logits.push(logits.row(0).to_owned());

        // Generate synthetic label (highest spike count)
        let label = logits
            .row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap();
        all_labels.push(label);
    }

    // Convert to arrays
    let logits_array = Array2::from_shape_fn((batch_size, num_classes), |(i, j)| all_logits[i][j]);

    println!("  Generated {} predictions", batch_size);

    // Step 2: Apply temperature scaling calibration
    let mut temp_scaling = TemperatureScaling::new();

    // `TemperatureScaling` takes rows of logits as `Vec<f64>`, not an ndarray.
    let logits_rows: Vec<Vec<f64>> = logits_array
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&v| v as f64).collect())
        .collect();

    // Split into train/val for calibration
    let split = batch_size * 7 / 10;
    let train_logits = logits_rows[..split].to_vec();
    let train_labels = all_labels[..split].to_vec();

    temp_scaling
        .fit(&train_logits, &train_labels)
        .expect("Temperature scaling fit failed");

    // `temperature` is private; read it through the accessor.
    println!("  Fitted temperature: {:.4}", temp_scaling.temperature());

    // Step 3: Apply calibration to validation set
    let val_logits = logits_rows[split..].to_vec();
    let val_labels = &all_labels[split..];

    let calibrated_probs = temp_scaling.calibrate_batch(&val_logits);

    println!("  Calibrated {} validation samples", calibrated_probs.len());

    // Step 4: Verify calibrated probabilities
    for probs in &calibrated_probs {
        let sum: f64 = probs.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "Probabilities should sum to 1, got {}",
            sum
        );

        for &p in probs.iter() {
            assert!((0.0..=1.0).contains(&p), "Probability out of range: {}", p);
        }
    }

    // Step 5: Compute calibration metrics
    //
    // `expected_calibration_error` and `brier_score` are BINARY: they take a
    // confidence per sample and whether that sample was right. The multiclass
    // reduction is the standard one -- the probability assigned to the
    // predicted class, paired with whether that prediction matched the label.
    let (confidences, correct): (Vec<f64>, Vec<bool>) = calibrated_probs
        .iter()
        .zip(val_labels.iter())
        .map(|(probs, &label)| {
            let (predicted, confidence) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map(|(i, &p)| (i, p))
                .expect("non-empty probability vector");
            (confidence, predicted == label)
        })
        .unzip();

    let ece = expected_calibration_error(&confidences, &correct, 10);
    let brier = brier_score(&confidences, &correct);

    println!("  Expected Calibration Error: {:.4}", ece);
    println!("  Brier Score: {:.4}", brier);

    assert!((0.0..=1.0).contains(&ece), "ECE should be in [0, 1]");
    assert!(
        (0.0..=1.0).contains(&brier),
        "Brier score should be in [0, 1]"
    );

    println!("  ✓ Calibration pipeline completed successfully");
}

#[test]
fn test_explainability_pipeline() {
    println!("Testing explainability pipeline");

    let batch_size = 2;
    let num_input = 50;
    let num_hidden = 30;
    let num_output = 5;
    let num_timesteps = 80;

    // Step 1: Create model and generate predictions
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(
        vec![num_input, num_hidden, num_output],
        snn_config,
        true, // use_bias
    )
    .expect("Failed to create SNN");

    let input_spikes = create_test_spike_tensor(batch_size, num_input, num_timesteps);
    let output_spikes = snn.forward(&input_spikes).expect("Forward pass failed");

    println!("  Model: {} → {} → {}", num_input, num_hidden, num_output);

    // Step 2: Compute spike importance
    //
    // `compute_spike_importance` works on per-neuron spike TIMES with the output
    // gradients and layer weights -- not on `SpikeTensor` values -- so derive
    // those from the tensors the network produced.
    let dense_out = output_spikes.to_dense();
    let spike_times: Vec<Vec<f64>> = (0..num_output)
        .map(|n| {
            (0..num_timesteps)
                .filter(|&t| dense_out[[0, t, n]] > 0.0)
                .map(|t| t as f64)
                .collect()
        })
        .collect();

    // Uniform gradients and weights: this test covers the plumbing and the
    // invariants, not the numerical attribution itself.
    let output_gradients = vec![1.0_f64; num_output];
    let layer_weights: Vec<Vec<f64>> = (0..num_output)
        .map(|n| vec![1.0 / (n + 1) as f64; num_output])
        .collect();

    let importance = compute_spike_importance(&spike_times, &output_gradients, &layer_weights);

    println!(
        "  Computed spike importance for {} spikes",
        importance.len()
    );

    // Every record must name a real neuron and a spike time the network emitted.
    for record in &importance {
        assert!(
            record.neuron_id < num_output,
            "neuron {} out of range",
            record.neuron_id
        );
        assert!(
            spike_times[record.neuron_id].contains(&record.spike_time),
            "importance reported for a spike at t={} that neuron {} never emitted",
            record.spike_time,
            record.neuron_id
        );
        assert!(record.importance_score.is_finite());
    }

    // Step 3: Aggregate to neuron-level importance
    let neuron_importance = aggregate_to_neurons(&importance);
    println!("  Aggregated to {} neurons", neuron_importance.len());

    // Aggregation must conserve spikes and must not invent neurons.
    let aggregated_spikes: usize = neuron_importance.iter().map(|n| n.total_spikes).sum();
    assert_eq!(
        aggregated_spikes,
        importance.len(),
        "aggregation lost or invented spikes"
    );
    for n in &neuron_importance {
        assert!(n.max_importance >= n.mean_importance);
    }

    // Step 4: Temporal attention over the same spike train
    let mut temporal_attention = TemporalAttention::new(num_timesteps as f64, 1.0);
    let flat_times: Vec<f64> = spike_times.iter().flatten().copied().collect();
    let flat_weights = vec![1.0_f64; flat_times.len()];
    temporal_attention.update_from_spikes(&flat_times, &flat_weights);

    println!(
        "  Temporal attention over {} bins",
        temporal_attention.attention_weights.len()
    );
    assert_eq!(
        temporal_attention.attention_weights.len(),
        temporal_attention.time_steps.len()
    );
    for &w in &temporal_attention.attention_weights {
        assert!(w >= 0.0 && w.is_finite(), "attention weight {w} invalid");
    }

    // Step 5: Gradient-based attribution
    let inputs: Vec<f64> = (0..num_output)
        .map(|n| spike_times[n].len() as f64)
        .collect();
    let gradient_attr = GradientAttribution::compute(&inputs, &output_gradients);

    assert_eq!(gradient_attr.len(), inputs.len());
    for a in &gradient_attr {
        assert!(a.is_finite(), "attribution {a} not finite");
    }

    // Step 6: Export explanation to JSON
    let explanation_json = export_explanation_json(None, None, Some(&importance));

    println!(
        "  Exported explanation JSON ({} bytes)",
        explanation_json.len()
    );
    assert!(!explanation_json.is_empty());

    // Verify it's valid JSON
    let parsed: serde_json::Value = serde_json::from_str(&explanation_json).expect("Invalid JSON");
    assert!(parsed.is_object());

    println!("  ✓ Explainability pipeline completed successfully");
}

#[test]
fn test_onnx_export_import_consistency() {
    println!("Testing ONNX export consistency");

    use dpb_snn::export::config::LayerType;
    use dpb_snn::export::weights::{LayerWeights, ModelWeights};
    use dpb_snn::export::{LayerConfig, ModelConfig, OnnxExporter};

    // Step 1: Create model configuration
    let mut config = ModelConfig::new("integration_test_model")
        .with_input_shape(vec![128])
        .with_output_shape(vec![10])
        .with_neuron_model("LIF".to_string())
        .with_time_steps(100);

    config.add_layer(LayerConfig::new(
        "layer1".to_string(),
        LayerType::SpikingLinear,
        128,
        64,
    ));

    config.add_layer(LayerConfig::new(
        "layer2".to_string(),
        LayerType::SpikingLinear,
        64,
        10,
    ));

    // Validate configuration
    config.validate().expect("Config validation failed");

    println!("  Created model config: 128 → 64 → 10");

    // Step 2: Create model weights
    let mut weights = ModelWeights::new("integration_test_model".to_string());

    // Layer 1: 128 × 64 = 8192 parameters
    weights.add_layer(
        LayerWeights::new(
            "layer1".to_string(),
            "SpikingLinear".to_string(),
            vec![0.01; 8192],
            vec![128, 64],
        )
        .with_bias(vec![0.0; 64]),
    );

    // Layer 2: 64 × 10 = 640 parameters
    weights.add_layer(
        LayerWeights::new(
            "layer2".to_string(),
            "SpikingLinear".to_string(),
            vec![0.01; 640],
            vec![64, 10],
        )
        .with_bias(vec![0.0; 10]),
    );

    weights.update_checksum();

    println!("  Created model weights: {} layers", weights.layers.len());
    // `ModelWeights` exposes its layers rather than a parameter count.
    let total_parameters: usize = weights
        .layers
        .iter()
        .map(|l| l.weights.len() + l.bias.as_ref().map_or(0, |b| b.len()))
        .sum();
    println!("  Total parameters: {}", total_parameters);

    // Step 3: Export to ONNX
    let exporter = OnnxExporter::with_default_config();
    let export_result = exporter
        .export_snn_model(&weights, &config.layers, &config.input_shape)
        .expect("ONNX export failed");

    println!(
        "  Exported to ONNX ({} bytes)",
        export_result.model_bytes.len()
    );
    assert!(!export_result.model_bytes.is_empty());

    // Step 4: Verify export metadata
    assert_eq!(export_result.metadata.model_name, "integration_test_model");
    assert_eq!(export_result.metadata.export_format, "onnx");
    assert!(!export_result.metadata.created_at.is_empty());

    println!("  Export metadata:");
    println!("    Model: {}", export_result.metadata.model_name);
    println!("    Format: {}", export_result.metadata.export_format);
    println!("    Created at: {}", export_result.metadata.created_at);

    // Step 5: Validate exported model
    let validation_result = exporter.validate(&export_result.model_bytes);
    assert!(
        validation_result.is_ok(),
        "ONNX model validation failed: {:?}",
        validation_result.err()
    );

    println!("  ✓ ONNX export validated successfully");

    // Step 6: Check for warnings (SNN-specific features may not be fully supported)
    if !export_result.warnings.is_empty() {
        println!("  Warnings during export:");
        for warning in &export_result.warnings {
            println!("    - {}", warning);
        }
    }

    println!("  ✓ ONNX export pipeline completed");
}

#[test]
fn test_uncertainty_estimation_pipeline() {
    println!("Testing uncertainty estimation pipeline");

    let batch_size = 10;
    let num_classes = 5;
    let num_timesteps = 50;
    let num_input = 32;

    // Step 1: Create ensemble of SNNs
    let num_models = 5;
    let mut models = Vec::new();

    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    for _ in 0..num_models {
        models.push(
            FeedforwardSNN::new(vec![num_input, 20, num_classes], snn_config.clone(), true)
                .expect("Failed to create SNN"),
        );
    }

    println!("  Created ensemble of {} models", num_models);

    // Step 2: Generate predictions from ensemble
    let input = create_test_spike_tensor(batch_size, num_input, num_timesteps);
    let mut all_predictions = Vec::new();

    for (i, model) in models.iter_mut().enumerate() {
        let output = model.forward(&input).expect("Forward pass failed");
        let decoder = SpikeRateDecoder::new(num_classes, None, false);
        let logits = decoder.decode(&output).expect("Decoding failed");

        if i == 0 {
            println!("  Prediction shape per model: {:?}", logits.shape());
        }

        all_predictions.push(logits);
    }

    // Step 3: Compute ensemble uncertainty
    //
    // `EnsembleUncertainty::new` takes the NUMBER of models; the predictions are
    // passed to `compute_statistics`, which returns per-class mean and variance
    // rather than a probability matrix.
    let uncertainty_estimator = EnsembleUncertainty::new(all_predictions.len());

    // Per sample, gather each model's class scores and reduce across the
    // ensemble.
    let mut means = Vec::with_capacity(batch_size);
    let mut variances = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        let per_model: Vec<Vec<f64>> = all_predictions
            .iter()
            .map(|pred| (0..num_classes).map(|c| pred[[i, c]] as f64).collect())
            .collect();
        let (mean, variance) = uncertainty_estimator.compute_statistics(&per_model);
        means.push(mean);
        variances.push(variance);
    }

    println!(
        "  Computed predictions and uncertainties for {} samples",
        batch_size
    );

    // Every sample must yield one statistic per class.
    for (i, (mean, variance)) in means.iter().zip(variances.iter()).enumerate() {
        assert_eq!(mean.len(), num_classes, "sample {i}");
        assert_eq!(variance.len(), num_classes, "sample {i}");

        // Variance of a real ensemble is non-negative and finite.
        for &v in variance {
            assert!(
                v >= 0.0 && v.is_finite(),
                "variance {v} invalid for sample {i}"
            );
        }
        for &m in mean {
            assert!(m.is_finite(), "mean {m} invalid for sample {i}");
        }
    }

    // Identical models must produce zero ensemble variance -- the property that
    // makes this an uncertainty estimate rather than an arbitrary spread.
    let identical: Vec<Vec<f64>> = vec![vec![0.25, 0.75]; 4];
    let (_, zero_variance) = uncertainty_estimator.compute_statistics(&identical);
    for &v in &zero_variance {
        assert!(v.abs() < 1e-12, "identical predictions gave variance {v}");
    }

    let mean_uncertainty: f64 = variances
        .iter()
        .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        .sum::<f64>()
        / batch_size as f64;
    println!("  Mean uncertainty: {:.4}", mean_uncertainty);

    // Step 4: Compute confidence intervals using bootstrap
    let confidence_level = 0.95;
    for i in 0..batch_size.min(3) {
        // Extract predictions for sample i across all models
        let sample_preds: Vec<f64> = all_predictions
            .iter()
            .map(|pred| pred[[i, 0]] as f64)
            .collect();

        let ci = bootstrap_ci(&sample_preds, 100, confidence_level);

        println!(
            "  Sample {} CI [{:.3}, {:.3}] (width: {:.3})",
            i,
            ci.lower,
            ci.upper,
            ci.upper - ci.lower
        );

        assert!(
            ci.lower <= ci.upper,
            "CI lower bound should be <= upper bound"
        );
    }

    println!("  ✓ Uncertainty estimation pipeline completed");
}

#[test]
fn test_multi_decoder_comparison() {
    println!("Testing multiple decoder strategies");

    let batch_size = 5;
    let num_neurons = 20;
    let num_timesteps = 100;

    let spikes = create_test_spike_tensor(batch_size, num_neurons, num_timesteps);

    // Test different decoders
    let decoders: Vec<(&str, Box<dyn Decoder>)> = vec![
        (
            "Rate",
            Box::new(SpikeRateDecoder::new(num_neurons, None, false)),
        ),
        (
            "First Spike",
            Box::new(FirstSpikeDecoder::new(num_neurons, num_timesteps)),
        ),
        // PopulationDecoder::new takes (num_classes, neurons_per_class) and
        // therefore expects `classes * per_class` input neurons. Passing
        // `num_neurons` as the class count asked for 200 where the tensor has 20.
        (
            "Population",
            Box::new(PopulationDecoder::new(num_neurons / 10, 10)),
        ),
        ("Max Spike", Box::new(MaxSpikeDecoder::new(num_neurons))),
    ];

    println!("  Testing {} decoder types", decoders.len());

    for (name, decoder) in decoders {
        let result = decoder
            .decode(&spikes)
            .unwrap_or_else(|_| panic!("{} decoder failed", name));

        println!("  {} decoder output shape: {:?}", name, result.shape());

        assert_eq!(result.shape()[0], batch_size);

        // Verify no NaN or infinite values
        for &val in result.iter() {
            assert!(
                val.is_finite(),
                "{} decoder produced non-finite value",
                name
            );
        }
    }

    println!("  ✓ All decoders executed successfully");
}

#[test]
fn test_training_with_convergence_analysis() {
    println!("Testing training loop with convergence analysis");

    use dpb_snn::analysis::*;

    let num_epochs = 20;
    let batch_size = 8;
    let num_input = 32;
    let num_output = 5;
    let num_timesteps = 50;

    // Create model
    let snn_config = SNNConfig {
        dt: 1.0,
        num_steps: num_timesteps,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
    };

    let mut snn = FeedforwardSNN::new(vec![num_input, 20, num_output], snn_config, true)
        .expect("Failed to create SNN");

    // Initialize analyzers
    let mut loss_plateau = LossPlateauDetector::new(5, 0.01, 3); // window, threshold, patience
    let mut spike_tracker = SpikeRateTracker::new(0.1); // target spike rate

    println!("  Training for {} epochs", num_epochs);

    // Simulate training loop
    for epoch in 0..num_epochs {
        // Generate batch
        let input = create_test_spike_tensor(batch_size, num_input, num_timesteps);
        let output = snn.forward(&input).expect("Forward pass failed");

        // Compute loss (simplified) - use dense representation
        let output_dense = output.to_dense();
        let target_rate = 0.1;
        let mut loss = 0.0;
        for b in 0..batch_size {
            for t in 0..num_timesteps {
                for n in 0..num_output {
                    let spike_val = output_dense[[b, t, n]] as f64;
                    loss += (spike_val - target_rate).powi(2);
                }
            }
        }
        loss /= (batch_size * num_timesteps * num_output) as f64;

        // Compute spike rate
        let total_spikes: f64 = output_dense.iter().sum::<f32>() as f64;
        let spike_rate = total_spikes / (batch_size * num_output * num_timesteps) as f64;

        // Create TrainingMetrics for analyzers
        let metrics = TrainingMetrics::new(epoch, loss, 0.8)
            .with_learning_rate(0.001)
            .with_gradient_norm(0.5)
            .with_spike_rate(spike_rate)
            .with_weight_norm(1.0);

        // Track metrics
        loss_plateau.update(epoch, &metrics);
        spike_tracker.update(epoch, &metrics);

        if epoch % 5 == 0 {
            println!(
                "  Epoch {}: loss={:.4}, spike_rate={:.4}",
                epoch, loss, spike_rate
            );
        }
    }

    // Check convergence
    let converged = loss_plateau.is_converged();
    println!("  Loss plateau converged: {}", converged);

    let spike_report = spike_tracker.analysis_report();
    println!("  Spike rate analysis:");
    println!("    Convergence status: {}", spike_report.converged);

    println!("  ✓ Training and analysis pipeline completed");
}
