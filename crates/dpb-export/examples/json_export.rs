//! Example: JSON Export of Encoder Models
//!
//! This example demonstrates how to export encoder parameters
//! and state to JSON format for interoperability.
//!
//! Run with: cargo run --example json_export -p dpb-export

use dpb_export::{
    ModelExporter,
    encoder_export::{EncoderParams, MockEncoder},
    json::JsonExporter,
    metadata::ModelMetadata,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DPB JSON Export Example ===\n");

    // Create a mock encoder for demonstration
    let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);

    // Method 1: Using the ModelExporter builder
    println!("Method 1: Using ModelExporter builder");
    println!("--------------------------------------");

    let exporter = ModelExporter::new()
        .with_name("EEG Level Crossing Encoder")
        .with_version("1.0.0")
        .with_description("8-channel level crossing encoder for EEG processing")
        .with_metadata("signal_type", "EEG")
        .with_metadata("application", "Research BCI");

    // Export to JSON file
    let output_path = PathBuf::from("/tmp/encoder_model.json");
    exporter.export_json(&output_path, &encoder)?;
    println!("Exported model to: {}\n", output_path.display());

    // Read and display the exported content
    let content = std::fs::read_to_string(&output_path)?;
    println!("Exported JSON (truncated):");
    println!("{}", &content[..content.len().min(500)]);
    println!("...\n");

    // Method 2: Using JsonExporter directly
    println!("Method 2: Using JsonExporter directly");
    println!("--------------------------------------");

    let mut metadata = ModelMetadata::new();
    metadata.set_name("Custom Delta Encoder");
    metadata.set_version("2.0.0");
    metadata.set_description("Delta modulation encoder with adaptive thresholds");
    metadata.add("encoder_type", "delta");
    metadata.add("adaptive", "true");

    let json_exporter = JsonExporter::new(metadata);
    let delta_encoder = MockEncoder::delta(4, 512.0, 0.05, 16);

    // Export to string
    let json_string = json_exporter.export_string(&delta_encoder)?;
    println!("JSON string length: {} bytes\n", json_string.len());

    // Method 3: Creating encoder params manually
    println!("Method 3: Custom EncoderParams");
    println!("-------------------------------");

    let custom_params = EncoderParams::new("temporal_contrast", 16, 1024.0)
        .with_thresholds(vec![0.08; 16])
        .with_adaptive(0.02)
        .with_extra("refractory_ms", serde_json::json!(2.0));

    println!("Encoder type: {}", custom_params.encoder_type);
    println!("Channels: {}", custom_params.num_channels);
    println!("Sample rate: {} Hz", custom_params.sample_rate);
    println!("Adaptive: {}", custom_params.adaptive);
    println!("Thresholds: {:?}", &custom_params.thresholds[..3]);
    println!();

    // Validate the params
    custom_params.validate()?;
    println!("Parameters validated successfully!");

    // Method 4: Importing JSON
    println!("\nMethod 4: Importing from JSON");
    println!("------------------------------");

    let imported = JsonExporter::import(&output_path)?;
    println!("Imported encoder type: {}", imported.encoder_type());
    println!("Imported channels: {}", imported.num_channels());
    println!("Imported sample rate: {} Hz", imported.sample_rate());

    // Cleanup
    std::fs::remove_file(&output_path)?;
    println!("\nCleaned up temporary file.");

    println!("\n=== Example Complete ===");
    Ok(())
}
