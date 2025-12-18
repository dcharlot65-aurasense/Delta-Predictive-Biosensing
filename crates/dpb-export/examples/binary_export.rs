//! Example: Binary Export for Embedded Deployment
//!
//! This example demonstrates how to export encoder models
//! to the compact DPB binary format optimized for embedded systems.
//!
//! Run with: cargo run --example binary_export -p dpb-export

use dpb_export::{
    binary::{BinaryExporter, BinaryImporter},
    encoder_export::MockEncoder,
    metadata::ModelMetadata,
    ModelExporter,
};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DPB Binary Export Example ===\n");

    // Create encoders for demonstration
    let lc_encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
    let delta_encoder = MockEncoder::delta(4, 512.0, 0.05, 16);

    // Method 1: Using ModelExporter builder
    println!("Method 1: Using ModelExporter builder");
    println!("--------------------------------------");

    let exporter = ModelExporter::new()
        .with_name("Embedded EEG Encoder")
        .with_version("1.0.0")
        .with_description("Compact encoder for ARM Cortex-M deployment");

    let output_path = PathBuf::from("/tmp/encoder_model.dpb");
    exporter.export_binary(&output_path, &lc_encoder)?;

    let file_size = std::fs::metadata(&output_path)?.len();
    println!("Exported model to: {}", output_path.display());
    println!("File size: {} bytes\n", file_size);

    // Method 2: Using BinaryExporter with compression
    println!("Method 2: BinaryExporter with compression");
    println!("-----------------------------------------");

    let mut metadata = ModelMetadata::new();
    metadata.set_name("Compressed Delta Encoder");
    metadata.set_version("1.0.0");
    metadata.set_description("Delta encoder with compression for wireless transmission");

    let compressed_exporter = BinaryExporter::new(metadata).with_compression(5);
    let compressed_path = PathBuf::from("/tmp/encoder_compressed.dpb");
    compressed_exporter.export(&compressed_path, &delta_encoder)?;

    let compressed_size = std::fs::metadata(&compressed_path)?.len();
    println!("Exported compressed model: {}", compressed_path.display());
    println!("Compressed file size: {} bytes\n", compressed_size);

    // Method 3: Import and verify
    println!("Method 3: Import and verify");
    println!("---------------------------");

    let imported = BinaryImporter::import(&output_path)?;
    println!("Format version: {}", imported.format_version);
    println!("Model name: {}", imported.metadata.name);
    println!("Encoder type: {}", imported.params.encoder_type);
    println!("Channels: {}", imported.params.num_channels);
    println!("Sample rate: {} Hz", imported.params.sample_rate);
    println!("Thresholds: {:?}", imported.params.thresholds);
    println!("Has state: {}\n", imported.state.is_some());

    // Method 4: Roundtrip verification
    println!("Method 4: Roundtrip verification");
    println!("---------------------------------");

    // Export
    let roundtrip_path = PathBuf::from("/tmp/roundtrip_test.dpb");
    let mut rt_metadata = ModelMetadata::new();
    rt_metadata.set_name("Roundtrip Test");
    rt_metadata.set_version("1.2.3");

    let rt_exporter = BinaryExporter::new(rt_metadata);
    rt_exporter.export(&roundtrip_path, &delta_encoder)?;

    // Import and verify
    let rt_imported = BinaryImporter::import(&roundtrip_path)?;

    assert_eq!(rt_imported.metadata.name, "Roundtrip Test");
    assert_eq!(rt_imported.metadata.version, "1.2.3");
    assert_eq!(rt_imported.params.encoder_type, "delta");
    assert_eq!(rt_imported.params.num_channels, 4);
    assert_eq!(rt_imported.params.sample_rate, 512.0);
    assert_eq!(rt_imported.params.num_levels, Some(16));

    println!("Roundtrip verification: PASSED");
    println!("All fields match expected values.\n");

    // Binary format structure info
    println!("Binary Format Structure");
    println!("-----------------------");
    println!("Magic: DPB\\x00 (4 bytes)");
    println!("Version: u16 (2 bytes)");
    println!("Flags: u16 (2 bytes)");
    println!("Reserved: (8 bytes)");
    println!("META section: Variable JSON");
    println!("PARM section: Fixed encoder params");
    println!("STAT section: Optional state (if present)");
    println!("END marker: (4 bytes)");
    println!("Checksum: u32 (4 bytes)\n");

    // Cleanup
    std::fs::remove_file(&output_path)?;
    std::fs::remove_file(&compressed_path)?;
    std::fs::remove_file(&roundtrip_path)?;
    println!("Cleaned up temporary files.");

    println!("\n=== Example Complete ===");
    Ok(())
}
