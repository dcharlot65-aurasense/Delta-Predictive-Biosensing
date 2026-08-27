//! Example: LSL Encoding Pipeline Configuration
//!
//! This example demonstrates how to configure and set up
//! encoding pipelines for real-time spike encoding.
//!
//! Run with: cargo run --example pipeline_config -p dpb-lsl

use dpb_lsl::pipeline::{
    EncoderParams, EncoderType, PipelineBuilder, PipelineConfig, PipelineStats,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DPB LSL Pipeline Configuration Example ===\n");

    // Method 1: Default configuration
    println!("Method 1: Default configuration");
    println!("-------------------------------");

    let default_config = PipelineConfig::default();
    println!("Default Pipeline Config:");
    println!("  Input type: {}", default_config.input_type);
    println!("  Output name: {}", default_config.output_name);
    println!("  Encoder: {:?}", default_config.encoder_type);
    println!("  Buffer size: {} samples", default_config.buffer_size);
    println!("  Adaptive: {}", default_config.adaptive);
    println!("  Stats interval: {} sec\n", default_config.stats_interval);

    // Method 2: Using PipelineBuilder
    println!("Method 2: Using PipelineBuilder");
    println!("-------------------------------");

    let pipeline = PipelineBuilder::new()
        .input_stream("BioAmp-EEG")
        .input_type("EEG")
        .output_name("DPB_Spikes_EEG")
        .output_source_id("dpb_encoder_001")
        .encoder(EncoderType::LevelCrossing)
        .threshold(0.1)
        .adaptive(true)
        .adaptation_rate(0.01)
        .buffer_size(1024)
        .stats_interval(30.0)
        .build();

    println!("Level Crossing Pipeline:");
    println!("  Input: {}", pipeline.config().input_stream);
    println!("  Output: {}", pipeline.config().output_name);
    println!("  Threshold: {}", pipeline.config().encoder_params.threshold);
    println!("  Running: {}\n", pipeline.is_running());

    // Method 3: Delta encoder configuration
    println!("Method 3: Delta encoder pipeline");
    println!("--------------------------------");

    let delta_pipeline = PipelineBuilder::new()
        .input_stream("HeartMonitor-ECG")
        .input_type("ECG")
        .output_name("DPB_Spikes_ECG")
        .encoder(EncoderType::Delta)
        .threshold(0.05)
        .adaptive(false)
        .buffer_size(512)
        .build();

    println!("Delta Pipeline Config:");
    println!("  Encoder type: {:?}", delta_pipeline.config().encoder_type);
    println!("  Threshold: {}", delta_pipeline.config().encoder_params.threshold);
    println!("  Non-adaptive mode\n");

    // Method 4: Temporal contrast configuration
    println!("Method 4: Temporal contrast pipeline");
    println!("------------------------------------");

    let tc_pipeline = PipelineBuilder::new()
        .input_stream("EMG-Sensor")
        .input_type("EMG")
        .output_name("DPB_Spikes_EMG")
        .encoder(EncoderType::TemporalContrast)
        .threshold(0.15)
        .adaptive(true)
        .adaptation_rate(0.02)
        .buffer_size(256)
        .stats_interval(10.0)
        .build();

    println!("Temporal Contrast Pipeline:");
    println!("  For fast EMG signals");
    println!("  High threshold: {}", tc_pipeline.config().encoder_params.threshold);
    println!("  Faster adaptation: {}\n", tc_pipeline.config().encoder_params.adaptation_rate);

    // Method 5: Send-on-delta with quantization
    println!("Method 5: Send-on-delta pipeline");
    println!("--------------------------------");

    let sod_pipeline = PipelineBuilder::new()
        .input_stream("SlowSensor")
        .input_type("EDA")
        .output_name("DPB_Spikes_EDA")
        .encoder(EncoderType::SendOnDelta)
        .threshold(0.02)
        .num_levels(64)
        .buffer_size(128)
        .build();

    println!("Send-on-Delta Pipeline:");
    println!("  For slow-varying signals (EDA)");
    println!("  Quantization levels: {}", sod_pipeline.config().encoder_params.num_levels);
    println!("  Low threshold: {}\n", sod_pipeline.config().encoder_params.threshold);

    // Method 6: Encoder parameters overview
    println!("Method 6: Encoder Parameters");
    println!("----------------------------");

    let default_params = EncoderParams::default();
    println!("Default EncoderParams:");
    println!("  threshold: {}", default_params.threshold);
    println!("  num_levels: {}", default_params.num_levels);
    println!("  refractory_period: {} sec", default_params.refractory_period);
    println!("  adaptation_rate: {}\n", default_params.adaptation_rate);

    // Method 7: Pipeline statistics structure
    println!("Method 7: Pipeline Statistics");
    println!("-----------------------------");

    let stats = PipelineStats::default();
    println!("PipelineStats fields:");
    println!("  samples_processed: {}", stats.samples_processed);
    println!("  spikes_generated: {}", stats.spikes_generated);
    println!("  spike_rate: {} Hz", stats.spike_rate);
    println!("  samples_dropped: {}", stats.samples_dropped);
    println!("  avg_latency: {} sec", stats.avg_latency);
    println!("  max_latency: {} sec", stats.max_latency);
    println!("  uptime: {} sec\n", stats.uptime);

    // Method 8: Configuration serialization
    println!("Method 8: Configuration serialization");
    println!("-------------------------------------");

    let config = PipelineConfig {
        input_stream: "TestStream".to_string(),
        input_type: "EEG".to_string(),
        output_name: "TestSpikes".to_string(),
        output_source_id: "test_001".to_string(),
        encoder_type: EncoderType::LevelCrossing,
        encoder_params: EncoderParams::default(),
        buffer_size: 1024,
        adaptive: true,
        stats_interval: 60.0,
    };

    let json = serde_json::to_string_pretty(&config)?;
    println!("Serialized configuration:");
    println!("{}\n", json);

    // Deserialize back
    let _parsed: PipelineConfig = serde_json::from_str(&json)?;
    println!("Successfully deserialized config back.\n");

    // Method 9: All encoder types
    println!("Method 9: Available encoder types");
    println!("---------------------------------");

    let encoder_types = [
        (EncoderType::LevelCrossing, "Level Crossing", "Traditional level quantization"),
        (EncoderType::Delta, "Delta", "Change-based encoding"),
        (EncoderType::TemporalContrast, "Temporal Contrast", "Derivative-based encoding"),
        (EncoderType::SendOnDelta, "Send-on-Delta", "Quantized change encoding"),
    ];

    for (etype, name, desc) in &encoder_types {
        println!("  {:?}: {} - {}", etype, name, desc);
    }

    println!("\n=== Example Complete ===");
    Ok(())
}
