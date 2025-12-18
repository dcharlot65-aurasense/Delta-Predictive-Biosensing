# DPB Framework API Guide

Comprehensive guide to the Delta-Predictive Biosensing (DPB) Framework API.

## Architecture Overview

The Delta-Predictive Biosensing framework consists of these crates:

| Crate | Purpose | Documentation |
|-------|---------|---------------|
| **dpb-core** | Signal processing, pipelines, I/O | Core types and domain-specific analysis |
| **dpb-encoders** | Event-based spike encoders | 77+ encoders for biosignal encoding |
| **dpb-neurons** | Neuron models | 19 neuron models + 6 surrogate gradients |
| **dpb-snn** | SNN architectures and training | Networks, training, calibration, explainability |
| **dpb-synth** | Synthetic data generation | 200+ generators + augmentation |
| **dpb-norms** | Normative databases | Age/sex-stratified reference values |
| **dpb-cognitive** | Cognitive assessment | Multi-modal cognitive analysis |
| **dpb-python** | Python bindings | PyO3-based Python interface |
| **dpb-ffi** | C FFI | C-compatible foreign function interface |

## Common Patterns

### 1. Signal Processing Pipeline

Complete pipeline from raw signal to analyzed results:

```rust
use dpb_core::signal::*;
use dpb_core::pipeline::*;
use ndarray::Array1;

fn analyze_ecg(raw_signal: Vec<f64>, sample_rate: f64) -> dpb_core::Result<()> {
    let signal = Array1::from_vec(raw_signal);

    // 1. Preprocessing: Filter noise
    let mut filter = IirFilter::bandpass(0.5, 40.0, sample_rate, 4)?;
    let filtered = filter.apply(signal.view())?;

    // 2. Normalization
    let normalized = normalize(filtered.view(), NormalizationMethod::ZScore)?;

    // 3. Feature Extraction: Detect R-peaks
    let detector = ecg::PanTompkinsDetector::new(sample_rate);
    let peaks = detector.detect(&normalized)?;

    // 4. Analysis: Compute HRV metrics
    let analyzer = hrv::HrvAnalyzer::new();
    let time_metrics = analyzer.compute_time_domain(&peaks, sample_rate)?;
    let freq_metrics = analyzer.compute_frequency_domain(&peaks, sample_rate)?;

    println!("Heart Rate: {:.1} bpm", time_metrics.mean_hr);
    println!("RMSSD: {:.1} ms", time_metrics.rmssd);
    println!("LF/HF Ratio: {:.2}", freq_metrics.lf_hf_ratio);

    Ok(())
}
```

### 2. Real-Time Streaming Pipeline

Process signals in real-time with latency tracking:

```rust
use dpb_core::pipeline::*;

fn realtime_processing() -> dpb_core::Result<()> {
    // Configure pipeline
    let config = PipelineConfig::new(
        256,     // Window size
        128,     // Hop size (50% overlap)
        1000.0,  // Sample rate
    )
    .with_max_latency(10.0)  // 10ms max latency
    .with_circular_buffer();

    let mut executor = PipelineExecutor::new(config);

    // Process streaming data
    loop {
        let sample = get_next_sample(); // Your acquisition function

        if let Some(result) = executor.process_sample(sample, |window| {
            // Your processing logic on each window
            analyze_window(window)
        }) {
            // Handle result
            handle_result(result);
        }

        // Check latency
        let stats = executor.stats();
        if stats.avg_latency_ms > 10.0 {
            println!("Warning: High latency detected");
        }
    }
}
```

### 3. SNN Training Workflow

Complete SNN training from data to deployment:

```rust
use dpb_snn::*;
use dpb_neurons::prelude::*;
use dpb_encoders::prelude::*;

fn train_snn() -> SNNResult<()> {
    // 1. Build network architecture
    let config = SNNConfig {
        dt: 1.0,
        num_steps: 100,
        neuron_model: NeuronModel::LIF,
        neuron_params: NeuronParams::default(),
        use_gpu: false,
    };

    let mut network = FeedforwardSNN::new(
        vec![128, 64, 10],  // Input -> Hidden -> Output
        config,
    );

    // 2. Configure training
    let surrogate = FastSigmoid::default();
    let mut trainer = BPTT::new(surrogate, 0.001); // learning rate
    let loss_fn = SpikingCrossEntropy::new();

    // 3. Training loop
    for epoch in 0..100 {
        for (input, label) in training_data {
            // Forward pass
            let output = network.forward(&input)?;

            // Compute loss
            let loss = loss_fn.compute(&output, &label)?;

            // Backward pass and update
            trainer.backward(&loss)?;
            trainer.update_weights(&mut network)?;
        }
    }

    // 4. Calibrate confidence scores
    let mut calibrator = TemperatureScaling::new();
    calibrator.fit(&val_logits, &val_labels)?;

    // 5. Export model
    export::export_onnx(&network, "model.onnx")?;

    Ok(())
}
```

### 4. Model Calibration

Ensure reliable confidence scores:

```rust
use dpb_snn::calibration::*;

fn calibrate_model(
    val_logits: &[Vec<f64>],
    val_labels: &[usize],
    test_logits: &[Vec<f64>],
) -> SNNResult<Vec<Vec<f64>>> {
    // 1. Fit calibrator on validation set
    let mut calibrator = TemperatureScaling::new();
    calibrator.fit(val_logits, val_labels)?;

    // 2. Calibrate test predictions
    let calibrated = calibrator.calibrate(test_logits);

    // 3. Evaluate calibration quality
    let ece = expected_calibration_error(&calibrated, val_labels, 10);
    let brier = brier_score(&calibrated, val_labels);

    println!("Expected Calibration Error: {:.4}", ece);
    println!("Brier Score: {:.4}", brier);

    // 4. Generate reliability diagram
    let bins = reliability_diagram(&calibrated, val_labels, 10);
    for bin in bins {
        println!("Conf: {:.2}, Acc: {:.2}, N: {}",
            bin.avg_confidence, bin.avg_accuracy, bin.count);
    }

    Ok(calibrated)
}
```

### 5. Synthetic Data Generation

Generate realistic test data:

```rust
use dpb_synth::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn generate_test_data(seed: u64) -> Result<Vec<f64>> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // 1. Generate base signal
    let generator = contact::ecg::EcgGenerator::new(250.0);
    let (signal, ground_truth) = generator.generate(
        30.0,  // 30 seconds
        75.0,  // 75 bpm
        &mut rng,
    )?;

    // 2. Apply augmentation
    let pipeline = AugmentationPipeline::new()
        .add(GaussianNoise::new(15.0), 0.8)
        .add(BaselineWander::new(0.2), 0.5)
        .add(PowerlineNoise::new(60.0), 0.3);

    let augmented = pipeline.apply(&signal, &mut rng);

    Ok(augmented)
}
```

### 6. Multi-Modal Sensor Fusion

Combine multiple modalities:

```rust
use dpb_snn::fusion::*;

fn multimodal_analysis() -> SNNResult<()> {
    // Configure fusion network
    let config = FusionConfig {
        ecg_channels: 12,
        imu_channels: 6,
        video_channels: 3,
        num_classes: 5,
    };

    // Build cross-modal attention network
    let mut network = CrossModalAttentionSNN::new(config)?;

    // Process multi-modal input
    let ecg_input = /* ECG spike train */;
    let imu_input = /* IMU spike train */;
    let video_input = /* Video features */;

    let output = network.forward_multimodal(
        &ecg_input,
        &imu_input,
        &video_input,
    )?;

    Ok(())
}
```

### 7. Normative Comparison

Compare individual to population norms:

```rust
use dpb_norms::*;

fn compare_to_norms(
    metric: MetricType,
    value: f64,
    age: f64,
    sex: Sex,
) -> Result<NormativeComparison> {
    // Load normative database
    let db = NormativeDatabase::load()?;

    // Create demographics
    let demographics = Demographics {
        age_years: age,
        sex,
        ethnicity: None,
        handedness: None,
        education_level: None,
        height_cm: None,
        weight_kg: None,
    };

    // Get normative stats
    let stats = db.lookup(metric, &demographics)?;

    // Compute comparison
    let comparison = NormativeComparison {
        metric,
        value,
        percentile: stats.percentile(value),
        z_score: stats.z_score(value),
        impairment: stats.impairment_level(value, MetricDirection::HigherIsBetter),
        reference_range: stats.reference_range(),
        mdc95: stats.mdc95,
        demographics,
    };

    println!("Value: {:.1} ({:.0}th percentile, z={:.2})",
        comparison.value,
        comparison.percentile,
        comparison.z_score
    );
    println!("Impairment: {}", comparison.impairment.label());

    Ok(comparison)
}
```

## Error Handling

All operations that can fail return `Result<T, E>`:

```rust
use dpb_core::{Result, DpbError};

fn safe_processing() -> Result<()> {
    // Results can be propagated with ?
    let signal = load_signal()?;
    let filtered = preprocess(signal)?;
    let features = extract_features(filtered)?;

    // Or handled explicitly
    match analyze(features) {
        Ok(result) => println!("Success: {:?}", result),
        Err(DpbError::InvalidDimensions(msg)) => {
            eprintln!("Dimension error: {}", msg);
        }
        Err(e) => {
            eprintln!("Other error: {}", e);
        }
    }

    Ok(())
}
```

## Thread Safety

Most types are `Send + Sync` for concurrent use:

```rust
use std::thread;
use dpb_core::signal::*;

fn parallel_processing(signals: Vec<Vec<f64>>) {
    let handles: Vec<_> = signals
        .into_iter()
        .map(|signal| {
            thread::spawn(move || {
                // Each thread processes independently
                process_signal(signal)
            })
        })
        .collect();

    for handle in handles {
        let result = handle.join().unwrap();
        println!("Result: {:?}", result);
    }
}
```

## GPU Acceleration

Enable GPU compute for performance:

```rust
use dpb_core::gpu::*;
use dpb_neurons::gpu::*;

async fn gpu_processing() -> dpb_core::Result<()> {
    // Initialize GPU context
    let context = GpuContext::new().await?;

    // Create GPU-accelerated neuron simulation
    let params = GpuSimParams {
        num_neurons: 10000,
        num_steps: 1000,
        dt: 1.0,
    };

    let mut gpu_lif = GpuLifNeuron::new(&context, params)?;

    // Run simulation on GPU
    let spikes = gpu_lif.simulate(&input_currents).await?;

    Ok(())
}
```

## Best Practices

### 1. Use Type-Safe Configurations

```rust
// Good: Use builder pattern
let config = PipelineConfig::new(256, 128, 1000.0)
    .with_max_latency(10.0)
    .with_circular_buffer();

// Avoid: Manual struct construction
let config = PipelineConfig {
    window_size: 256,
    hop_size: 128,
    sample_rate: 1000.0,
    max_latency_ms: Some(10.0),
    use_circular_buffer: true,
};
```

### 2. Prefer Prelude Imports

```rust
// Good: Use prelude for common types
use dpb_core::prelude::*;
use dpb_snn::prelude::*;

// Avoid: Individual imports for basics
use dpb_core::types::SpikeEvent;
use dpb_core::types::SpikeTrain;
use dpb_core::error::Result;
// ... etc
```

### 3. Validate Input Data

```rust
use dpb_core::validation::*;

fn process_with_validation(signal: &[f64]) -> Result<()> {
    // Check for NaN/Inf
    validate_finite(signal)?;

    // Check signal quality
    let quality = assess_signal_quality(signal, 250.0)?;
    if quality.overall_quality < 0.7 {
        return Err(DpbError::InvalidSignal(
            "Poor signal quality".to_string()
        ));
    }

    // Continue processing...
    Ok(())
}
```

### 4. Use Seeded RNG for Reproducibility

```rust
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn reproducible_generation(seed: u64) -> Vec<f64> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // All random operations will be deterministic
    let generator = /* ... */;
    generator.generate(&mut rng)
}
```

## Performance Tips

### 1. Use Array Views to Avoid Copies

```rust
use ndarray::{Array1, ArrayView1};

// Good: Use views for read-only operations
fn process_efficient(signal: ArrayView1<f64>) -> f64 {
    signal.iter().sum()
}

// Less efficient: Takes ownership or clones
fn process_inefficient(signal: Array1<f64>) -> f64 {
    signal.iter().sum()
}
```

### 2. Batch Operations

```rust
use dpb_neurons::batch::*;

// Good: Process in batches
let mut layer = BatchLifLayer::new(1000, config);
let batch_output = layer.update_batch(&batch_inputs, dt);

// Avoid: Individual neuron updates
for input in inputs {
    neuron.update(input, dt);
}
```

### 3. Reuse Allocations

```rust
// Good: Reuse buffers
let mut buffer = Vec::with_capacity(1024);
for _ in 0..epochs {
    buffer.clear();
    // Use buffer...
}

// Avoid: Repeated allocations
for _ in 0..epochs {
    let buffer = Vec::new();
    // ...
}
```

## Further Reading

- **API Documentation**: Run `cargo doc --open` to browse full API docs
- **Examples**: See `examples/` directory for complete usage examples
- **System Catalog**: See `docs/SYSTEM_CATALOG.md` for component inventory
- **Benchmarks**: Run `cargo bench` in `dpb-bench` crate

## Getting Help

- GitHub Issues: Report bugs and request features
- Discussions: Ask questions and share ideas
- Documentation: Browse inline docs with `cargo doc`
