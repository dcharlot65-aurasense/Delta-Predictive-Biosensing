# DPB-Bench: Comprehensive Benchmark Suite

A comprehensive benchmarking and validation suite for the Delta-Predictive Biosensing (DPB) Framework.

## Features

- **Standard Datasets**: Synthetic biosignal datasets with ground truth (ECG, Gait, Tremor, Voice)
- **Baseline Comparisons**: Compare SNNs against ANNs and traditional signal processing methods
- **Performance Profiling**: Time, memory, spike activity, and energy consumption profiling
- **Report Generation**: Export results as JSON, CSV, and Markdown
- **Test Scenarios**: Pre-configured benchmarking scenarios for common use cases
- **Criterion Benchmarks**: Detailed micro-benchmarks for encoders, neurons, and inference

## Structure

```
crates/dpb-bench/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Main library interface
│   ├── datasets.rs         # Synthetic dataset generators
│   ├── baselines.rs        # Baseline implementations (ANN, conventional, SNN)
│   ├── profiling.rs        # Performance profiling tools
│   ├── reports.rs          # Report generation and export
│   └── scenarios.rs        # Pre-configured test scenarios
├── benches/
│   ├── encoding.rs         # Encoder benchmarks
│   ├── neurons.rs          # Neuron model benchmarks
│   ├── inference.rs        # SNN inference benchmarks
│   └── synthetic.rs        # Synthetic data generation benchmarks
└── examples/
    └── run_benchmarks.rs   # Complete benchmark suite example
```

## Quick Start

### Running Tests

```bash
# Run all tests
cargo test -p dpb-bench

# Run tests with output
cargo test -p dpb-bench -- --nocapture
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench -p dpb-bench

# Run specific benchmark suite
cargo bench -p dpb-bench --bench encoding
cargo bench -p dpb-bench --bench neurons
cargo bench -p dpb-bench --bench inference
cargo bench -p dpb-bench --bench synthetic
```

### Running the Example

```bash
# Build and run the comprehensive benchmark example
cargo run -p dpb-bench --example run_benchmarks --release
```

This will generate reports in the `target/` directory:
- `benchmark_report.json` - Complete results in JSON format
- `benchmark_report.csv` - Results in CSV format for spreadsheet analysis
- `benchmark_report.md` - Human-readable Markdown report
- `comparison_report.json` - SNN vs ANN comparison results
- `comparison_report.md` - Comparison report in Markdown

## Usage Examples

### Using Standard Datasets

```rust
use dpb_bench::prelude::*;

// Create a synthetic ECG dataset
let dataset = SyntheticECG::new(1000, 250.0, Some(42));
let (signal, ground_truth) = dataset.generate()?;

// Access ground truth annotations
if let Some(temporal) = &ground_truth.temporal {
    println!("Found {} R-peaks", temporal.len());
}
```

### Performance Profiling

```rust
use dpb_bench::prelude::*;

// Time profiling
let mut profiler = TimeProfiler::new("MyOperation");
profiler.start();
// ... perform operation ...
profiler.stop();
println!("Elapsed: {:.2}ms", profiler.elapsed_ms());

// Memory profiling
let mut mem_profiler = MemoryProfiler::new("Memory");
mem_profiler.allocate(1024);
println!("Peak memory: {:.2}MB", mem_profiler.peak_mb());

// Spike profiling
let mut spike_profiler = SpikeProfiler::new("Spikes", 100, 1000);
spike_profiler.record_spike(0, 10);
println!("Sparsity: {:.2}%", spike_profiler.sparsity() * 100.0);

// Energy estimation
let mut energy = EnergyEstimator::neuromorphic("Energy");
energy.record_spikes(1000, 10);
println!("Energy: {:.3}mJ", energy.total_energy_mj());
```

### Baseline Comparisons

```rust
use dpb_bench::prelude::*;

// Create ANN baseline
let ann = ANNBaseline::new(10, vec![20, 10], 3);
let inputs = vec![vec![0.5; 10]; 100];
let ann_result = ann.benchmark(&inputs);

// Create SNN baseline
let snn = SNNBaseline::new(100, 1000, 50);
let snn_result = snn.benchmark(100, 0.1);

// Compare
let metrics = ComparisonMetrics::compute(&snn_result, &ann_result);
println!("Speedup: {:.2}x", metrics.speedup);
println!("Energy efficiency: {:.2}x", metrics.energy_efficiency);
```

### Test Scenarios

```rust
use dpb_bench::prelude::*;

// Encoding scenario
let dataset = SyntheticECG::new(1000, 250.0, Some(42));
let scenario = EncodingScenario::new("ECG").with_trials(10);

let result = scenario.run(&dataset, |signal| {
    // Your encoder function here
    Ok(vec![(0.1, 0), (0.5, 1)])
})?;

println!("Precision: {:.2}%", result.metrics["precision"] * 100.0);
println!("Recall: {:.2}%", result.metrics["recall"] * 100.0);
```

### Report Generation

```rust
use dpb_bench::prelude::*;

let mut report = BenchmarkReport::new("My Benchmarks");

// Add results
report.add_result(BenchmarkResult {
    name: "test".to_string(),
    category: "encoding".to_string(),
    profile: ProfileResult {
        name: "test".to_string(),
        elapsed_ms: 10.0,
        memory_bytes: 1024,
        num_spikes: 100,
        sparsity: 0.05,
        energy_mj: 0.1,
        metadata: HashMap::new(),
    },
    baseline: None,
    metrics: HashMap::new(),
});

// Export in multiple formats
report.export("results.json", ReportFormat::JSON)?;
report.export("results.csv", ReportFormat::CSV)?;
report.export("results.md", ReportFormat::Markdown)?;
```

## Datasets

### SyntheticECG
Generates synthetic ECG signals with known R-peaks.

```rust
let dataset = SyntheticECG::new(1000, 250.0, Some(42))
    .with_heart_rate(75.0)
    .with_noise(0.05);
```

### SyntheticGait
Generates 3-axis accelerometer data with heel strikes and toe-offs.

```rust
let dataset = SyntheticGait::new(1000, 100.0, Some(42));
```

### SyntheticTremor
Generates 3-axis tremor data with known frequency.

```rust
let dataset = SyntheticTremor::new(1000, 100.0, Some(42))
    .with_frequency(5.0);
```

### SyntheticVoice
Generates voice signals with known fundamental frequency.

```rust
let dataset = SyntheticVoice::new(1000, 16000.0, Some(42))
    .with_f0(120.0);
```

## Profiling Tools

### TimeProfiler
Measures execution time with support for multiple measurements and statistics.

Methods:
- `start()` - Start timing
- `stop()` - Stop timing
- `elapsed_ms()` - Total elapsed time
- `average_ms()` - Average time per measurement
- `std_dev_ms()` - Standard deviation

### MemoryProfiler
Tracks memory allocations and peak usage.

Methods:
- `allocate(bytes)` - Record allocation
- `deallocate(bytes)` - Record deallocation
- `peak_bytes()` - Peak memory usage
- `peak_mb()` - Peak memory in MB

### SpikeProfiler
Analyzes spike activity in SNNs.

Methods:
- `record_spike(neuron_id, timestep)` - Record a spike
- `sparsity()` - Compute sparsity (0.0-1.0)
- `average_firing_rate()` - Average spikes per neuron per timestep
- `coefficient_of_variation(neuron_id)` - CV of inter-spike intervals

### EnergyEstimator
Estimates energy consumption for neuromorphic vs. conventional computing.

Energy models:
- **Neuromorphic**: 50 pJ per spike-synaptic operation
- **CMOS**: 4.6 pJ per MAC operation (45nm)

Methods:
- `record_spikes(num_spikes, fanout)` - Record spike operations
- `record_macs(num_macs)` - Record MAC operations
- `total_energy_mj()` - Total energy in millijoules
- `efficiency_vs(baseline)` - Energy efficiency ratio

## Test Scenarios

### EncodingScenario
Tests encoder accuracy vs ground truth.

```rust
let scenario = EncodingScenario::new("test")
    .with_trials(100)
    .with_target_rate(1000.0);
```

### ClassificationScenario
Tests classification accuracy.

```rust
let scenario = ClassificationScenario::new("test", 3)
    .with_samples(50);
```

### RegressionScenario
Tests regression performance (RMSE, MAE).

```rust
let scenario = RegressionScenario::new("test")
    .with_samples(100)
    .with_target_rmse(0.1);
```

### LatencyScenario
Measures end-to-end latency with statistics.

```rust
let scenario = LatencyScenario::new("test")
    .with_iterations(1000)
    .with_target_latency(10.0);
```

## Report Formats

### JSON
Complete structured data for programmatic analysis.

### CSV
Tabular format for spreadsheet analysis and plotting.

### Markdown
Human-readable reports with tables and summaries.

## Development

### Building

```bash
cargo build -p dpb-bench
```

### Testing

```bash
cargo test -p dpb-bench
```

### Documentation

```bash
cargo doc -p dpb-bench --open
```

## Performance Notes

- All datasets support seeded RNG for reproducibility
- Benchmarks use Criterion for statistical analysis
- Memory profiling tracks allocations, not actual system memory
- Energy estimates are based on published neuromorphic hardware specifications
- Sparsity calculations assume uniform distribution across neurons/time

## Future Work

- [ ] GPU-accelerated profiling
- [ ] Real-world dataset loaders (MIT-BIH, etc.)
- [ ] Hardware deployment benchmarks (Loihi, SpiNNaker)
- [ ] Power measurement integration
- [ ] Automated regression testing
- [ ] Benchmark result database
- [ ] Web dashboard for results visualization

## License

MIT OR Apache-2.0
