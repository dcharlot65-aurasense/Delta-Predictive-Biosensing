//! Run comprehensive benchmarks and generate reports
//!
//! This example demonstrates how to run the full benchmark suite and
//! generate reports in various formats.

use dpb_bench::prelude::*;
use dpb_encoders::prelude::*;
use dpb_neurons::prelude::*;
use dpb_core::{EventEncoder, SignalBuffer};
use std::collections::HashMap;

fn main() -> anyhow::Result<()> {
    println!("=== DPB Comprehensive Benchmark Suite ===\n");

    // Create main benchmark report
    let mut report = BenchmarkReport::new("DPB Framework Benchmark");

    // Add metadata
    report.metadata.metadata.insert(
        "description".to_string(),
        "Comprehensive benchmark of DPB framework components".to_string(),
    );

    println!("Running encoder benchmarks...");
    run_encoder_benchmarks(&mut report)?;

    println!("Running neuron benchmarks...");
    run_neuron_benchmarks(&mut report)?;

    println!("Running scenario benchmarks...");
    run_scenario_benchmarks(&mut report)?;

    println!("Running baseline comparisons...");
    let comparison_report = run_baseline_comparisons()?;

    // Print summary
    println!("\n=== Benchmark Summary ===");
    println!("Total benchmarks: {}", report.summary.total_benchmarks);
    println!("Total time: {:.2} ms", report.summary.total_time_ms);
    println!("Average latency: {:.2} ms", report.summary.avg_latency_ms);
    println!("Total energy: {:.3} mJ", report.summary.total_energy_mj);
    println!("Average sparsity: {:.2}%", report.summary.avg_sparsity * 100.0);
    println!("Peak memory: {:.2} MB", report.summary.peak_memory_mb);

    // Export reports
    println!("\nExporting reports...");
    report.export("target/benchmark_report.json", ReportFormat::JSON)?;
    report.export("target/benchmark_report.csv", ReportFormat::CSV)?;
    report.export("target/benchmark_report.md", ReportFormat::Markdown)?;

    comparison_report.export("target/comparison_report.json", ReportFormat::JSON)?;
    comparison_report.export("target/comparison_report.md", ReportFormat::Markdown)?;

    println!("\nReports saved to target/ directory:");
    println!("  - benchmark_report.json");
    println!("  - benchmark_report.csv");
    println!("  - benchmark_report.md");
    println!("  - comparison_report.json");
    println!("  - comparison_report.md");

    Ok(())
}

fn run_encoder_benchmarks(report: &mut BenchmarkReport) -> anyhow::Result<()> {
    // Test Level Crossing Encoder
    {
        let dataset = SyntheticECG::new(5000, 250.0, Some(42));
        let (signal, _) = dataset.generate()?;

        let encoder = LevelCrossingEncoder::new("benchmark");
        let config = LevelCrossingConfig {
            threshold: 0.3,
            relative: false,
            refractory_period: 0.01,
        };

        let mut profiler = TimeProfiler::new("LevelCrossing");
        let mut mem_profiler = MemoryProfiler::new("LevelCrossing");
        let mut energy = EnergyEstimator::neuromorphic("LevelCrossing");

        profiler.start();
        let events = encoder.encode(&signal, &config)?;
        profiler.stop();

        mem_profiler.allocate(events.len() * std::mem::size_of::<dpb_core::SpikeEvent>());
        energy.record_spikes(events.len(), 1);

        let mut metrics = HashMap::new();
        metrics.insert("num_events".to_string(), events.len() as f64);
        metrics.insert("encoding_rate".to_string(), events.len() as f64 / dataset.duration());

        report.add_result(BenchmarkResult {
            name: "LevelCrossing_ECG".to_string(),
            category: "encoding".to_string(),
            profile: ProfileResult {
                name: "LevelCrossing".to_string(),
                elapsed_ms: profiler.elapsed_ms(),
                memory_bytes: mem_profiler.peak_bytes(),
                num_spikes: events.len(),
                sparsity: events.len() as f64 / signal.samples().len() as f64,
                energy_mj: energy.total_energy_mj(),
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics,
        });
    }

    // Test ECG R-Peak Encoder
    {
        let dataset = SyntheticECG::new(5000, 250.0, Some(42));
        let (signal, gt) = dataset.generate()?;

        let encoder = EcgRPeakEncoder::new();
        let config = EcgRPeakConfig::default();

        let mut profiler = TimeProfiler::new("EcgRPeak");
        let mut energy = EnergyEstimator::neuromorphic("EcgRPeak");

        profiler.start();
        let events = encoder.encode(&signal, &config)?;
        profiler.stop();

        energy.record_spikes(events.len(), 1);

        // Calculate accuracy vs ground truth
        let gt_times: Vec<f64> = gt.temporal
            .as_ref()
            .map(|t| t.iter().map(|(time, _, _)| *time).collect())
            .unwrap_or_default();

        let precision = calculate_precision(&events, &gt_times, 0.05);
        let recall = calculate_recall(&events, &gt_times, 0.05);
        let f1 = 2.0 * precision * recall / (precision + recall);

        let mut metrics = HashMap::new();
        metrics.insert("precision".to_string(), precision);
        metrics.insert("recall".to_string(), recall);
        metrics.insert("f1_score".to_string(), f1);
        metrics.insert("num_events".to_string(), events.len() as f64);

        report.add_result(BenchmarkResult {
            name: "EcgRPeak".to_string(),
            category: "encoding".to_string(),
            profile: ProfileResult {
                name: "EcgRPeak".to_string(),
                elapsed_ms: profiler.elapsed_ms(),
                memory_bytes: events.len() * std::mem::size_of::<dpb_core::SpikeEvent>(),
                num_spikes: events.len(),
                sparsity: events.len() as f64 / signal.samples().len() as f64,
                energy_mj: energy.total_energy_mj(),
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics,
        });
    }

    // Test Gait Encoder
    {
        let dataset = SyntheticGait::new(5000, 100.0, Some(42));
        let (signal, _) = dataset.generate()?;

        let encoder = HeelStrikeEncoder::new();
        let config = HeelStrikeConfig::default();

        let mut profiler = TimeProfiler::new("HeelStrike");
        let mut energy = EnergyEstimator::neuromorphic("HeelStrike");

        profiler.start();
        let events = encoder.encode(&signal, &config)?;
        profiler.stop();

        energy.record_spikes(events.len(), 1);

        report.add_result(BenchmarkResult {
            name: "HeelStrike_Gait".to_string(),
            category: "encoding".to_string(),
            profile: ProfileResult {
                name: "HeelStrike".to_string(),
                elapsed_ms: profiler.elapsed_ms(),
                memory_bytes: events.len() * std::mem::size_of::<dpb_core::SpikeEvent>(),
                num_spikes: events.len(),
                sparsity: events.len() as f64 / signal.samples().len() as f64,
                energy_mj: energy.total_energy_mj(),
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics: HashMap::new(),
        });
    }

    Ok(())
}

fn run_neuron_benchmarks(report: &mut BenchmarkReport) -> anyhow::Result<()> {
    // Test LIF neuron
    {
        let config = dpb_neurons::lif::LifConfig::default();
        let mut neuron = LifNeuron::new(config);

        let mut profiler = TimeProfiler::new("LIF");
        let mut spike_profiler = SpikeProfiler::new("LIF", 1, 1000);
        let mut energy = EnergyEstimator::neuromorphic("LIF");

        profiler.start();
        for t in 0..1000 {
            if neuron.update(10.0, 1.0) {
                spike_profiler.record_spike(0, t);
            }
        }
        profiler.stop();

        energy.record_spikes(spike_profiler.total_spikes(), 1);

        report.add_result(BenchmarkResult {
            name: "LIF_1000steps".to_string(),
            category: "neurons".to_string(),
            profile: ProfileResult {
                name: "LIF".to_string(),
                elapsed_ms: profiler.elapsed_ms(),
                memory_bytes: std::mem::size_of::<LifNeuron>(),
                num_spikes: spike_profiler.total_spikes(),
                sparsity: spike_profiler.sparsity(),
                energy_mj: energy.total_energy_mj(),
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics: HashMap::new(),
        });
    }

    // Test batch LIF
    {
        let config = dpb_neurons::lif::LifConfig::default();
        let mut layer = BatchLifLayer::new(100, config);

        let inputs = ndarray::Array1::from_elem(100, 10.0);

        let mut profiler = TimeProfiler::new("BatchLIF");
        let mut spike_profiler = SpikeProfiler::new("BatchLIF", 100, 100);
        let mut energy = EnergyEstimator::neuromorphic("BatchLIF");

        profiler.start();
        for t in 0..100 {
            let spikes = layer.update(&inputs, 1.0);
            for (neuron, &spiked) in spikes.iter().enumerate() {
                if spiked {
                    spike_profiler.record_spike(neuron, t);
                }
            }
        }
        profiler.stop();

        energy.record_spikes(spike_profiler.total_spikes(), 10);

        report.add_result(BenchmarkResult {
            name: "BatchLIF_100x100".to_string(),
            category: "neurons".to_string(),
            profile: ProfileResult {
                name: "BatchLIF".to_string(),
                elapsed_ms: profiler.elapsed_ms(),
                memory_bytes: 100 * std::mem::size_of::<LifNeuron>(),
                num_spikes: spike_profiler.total_spikes(),
                sparsity: spike_profiler.sparsity(),
                energy_mj: energy.total_energy_mj(),
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics: HashMap::new(),
        });
    }

    Ok(())
}

fn run_scenario_benchmarks(report: &mut BenchmarkReport) -> anyhow::Result<()> {
    // Encoding scenario
    {
        let dataset = SyntheticECG::new(1000, 250.0, Some(42));
        let scenario = EncodingScenario::new("ECG_Encoding").with_trials(10);

        let result = scenario.run(&dataset, |signal| {
            let encoder = LevelCrossingEncoder::new("scenario");
            let config = LevelCrossingConfig {
                threshold: 0.3,
                relative: false,
                refractory_period: 0.01,
            };

            let sig_buf = SignalBuffer::single_channel(signal.to_vec(), 250.0);
            let events = encoder.encode(&sig_buf, &config)?;

            Ok(events.into_iter().map(|e| (e.timestamp, e.channel as usize)).collect())
        })?;

        report.add_result(BenchmarkResult {
            name: "EncodingScenario_ECG".to_string(),
            category: "scenarios".to_string(),
            profile: ProfileResult {
                name: "EncodingScenario".to_string(),
                elapsed_ms: result.timing.total_ms,
                memory_bytes: 0,
                num_spikes: 0,
                sparsity: 0.0,
                energy_mj: 0.0,
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics: result.metrics,
        });
    }

    // Classification scenario
    {
        let scenario = ClassificationScenario::new("MockClassification", 3).with_samples(20);

        let result = scenario.run(|sample| {
            // Simple mock classifier
            Ok(if sample[0] < 0.3 { 0 } else if sample[0] < 0.6 { 1 } else { 2 })
        })?;

        report.add_result(BenchmarkResult {
            name: "ClassificationScenario".to_string(),
            category: "scenarios".to_string(),
            profile: ProfileResult {
                name: "ClassificationScenario".to_string(),
                elapsed_ms: result.timing.total_ms,
                memory_bytes: 0,
                num_spikes: 0,
                sparsity: 0.0,
                energy_mj: 0.0,
                metadata: HashMap::new(),
            },
            baseline: None,
            metrics: result.metrics,
        });
    }

    Ok(())
}

fn run_baseline_comparisons() -> anyhow::Result<ComparisonReport> {
    let mut report = ComparisonReport::new("SNN vs ANN Comparison");

    // Simple classification task
    {
        // SNN result
        let snn = SNNBaseline::new(100, 1000, 50);
        let snn_result = snn.benchmark(10, 0.1);

        // ANN result
        let ann = ANNBaseline::new(10, vec![20, 10], 3);
        let inputs = vec![vec![0.5; 10]; 10];
        let ann_result = ann.benchmark(&inputs);

        report.add_comparison("classification", snn_result, Some(ann_result), None);
    }

    Ok(report)
}

fn calculate_precision(events: &[dpb_core::SpikeEvent], ground_truth: &[f64], tolerance: f64) -> f64 {
    if events.is_empty() {
        return 0.0;
    }

    let mut true_positives = 0;
    for event in events {
        if ground_truth.iter().any(|&gt| (event.timestamp - gt).abs() < tolerance) {
            true_positives += 1;
        }
    }

    true_positives as f64 / events.len() as f64
}

fn calculate_recall(events: &[dpb_core::SpikeEvent], ground_truth: &[f64], tolerance: f64) -> f64 {
    if ground_truth.is_empty() {
        return 0.0;
    }

    let mut true_positives = 0;
    for &gt in ground_truth {
        if events.iter().any(|e| (e.timestamp - gt).abs() < tolerance) {
            true_positives += 1;
        }
    }

    true_positives as f64 / ground_truth.len() as f64
}
