//! Example: LSL Stream Information
//!
//! This example demonstrates how to create and configure
//! LSL stream metadata for biosignal streaming.
//!
//! Run with: cargo run --example stream_info -p dpb-lsl

use dpb_lsl::{
    stream_info::{ChannelInfo, StreamInfo, StreamInfoBuilder},
    stream_types, ChannelFormat,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DPB LSL Stream Info Example ===\n");

    // Method 1: Direct StreamInfo creation
    println!("Method 1: Direct StreamInfo creation");
    println!("-------------------------------------");

    let eeg_info = StreamInfo::new(
        "BioAmp-EEG",
        stream_types::EEG,
        8,
        256.0,
        ChannelFormat::Float32,
        "bioamp_001",
    )?;

    println!("Created EEG stream:");
    println!("  Name: {}", eeg_info.name());
    println!("  Type: {}", eeg_info.stream_type());
    println!("  Channels: {}", eeg_info.channel_count());
    println!("  Sample Rate: {} Hz", eeg_info.nominal_srate());
    println!("  Source ID: {}", eeg_info.source_id());
    println!("  Bytes per sample: {}", eeg_info.bytes_per_sample());
    println!("  Irregular rate: {}\n", eeg_info.is_irregular_rate());

    // Method 2: Using StreamInfoBuilder
    println!("Method 2: Using StreamInfoBuilder");
    println!("----------------------------------");

    let ecg_info = StreamInfoBuilder::new()
        .name("HeartMonitor-ECG")
        .stream_type(stream_types::ECG)
        .channel_count(3)
        .nominal_srate(512.0)
        .channel_format(ChannelFormat::Float32)
        .source_id("heart_monitor_001")
        .session_id("session_20250118_001")
        .xml_desc(r#"
            <desc>
                <manufacturer>AuraSense</manufacturer>
                <channels>
                    <channel><label>Lead I</label><unit>mV</unit></channel>
                    <channel><label>Lead II</label><unit>mV</unit></channel>
                    <channel><label>Lead III</label><unit>mV</unit></channel>
                </channels>
            </desc>
        "#)
        .build()?;

    println!("Created ECG stream:");
    println!("  Name: {}", ecg_info.name());
    println!("  Type: {}", ecg_info.stream_type());
    println!("  Channels: {}", ecg_info.channel_count());
    println!("  Sample Rate: {} Hz", ecg_info.nominal_srate());
    println!("  Session ID: {:?}", ecg_info.session_id());
    println!("  Has XML desc: {}\n", ecg_info.xml_desc().is_some());

    // Method 3: Creating marker streams (irregular rate)
    println!("Method 3: Marker stream (irregular rate)");
    println!("----------------------------------------");

    let marker_info = StreamInfo::new(
        "ExperimentMarkers",
        stream_types::MARKERS,
        1,
        0.0, // 0 = irregular rate
        ChannelFormat::String,
        "experiment_markers",
    )?;

    println!("Created Marker stream:");
    println!("  Name: {}", marker_info.name());
    println!("  Type: {}", marker_info.stream_type());
    println!("  Irregular rate: {}\n", marker_info.is_irregular_rate());

    // Method 4: Creating spike output streams
    println!("Method 4: Spike output stream");
    println!("-----------------------------");

    let spike_info = StreamInfo::new(
        "DPB_Spikes",
        stream_types::SPIKES,
        8,
        0.0, // Irregular - spikes occur at variable times
        ChannelFormat::Float32,
        "dpb_encoder_001",
    )?
    .with_session_id("spike_session_001");

    println!("Created Spike stream:");
    println!("  Name: {}", spike_info.name());
    println!("  Type: {}", spike_info.stream_type());
    println!("  Channels: {}", spike_info.channel_count());
    println!("  Session: {:?}\n", spike_info.session_id());

    // Method 5: Channel metadata
    println!("Method 5: Channel metadata");
    println!("--------------------------");

    let channels = [ChannelInfo::eeg("Fp1"),
        ChannelInfo::eeg("Fp2"),
        ChannelInfo::eeg("F3"),
        ChannelInfo::eeg("F4"),
        ChannelInfo::eeg("C3"),
        ChannelInfo::eeg("C4"),
        ChannelInfo::eeg("O1"),
        ChannelInfo::eeg("O2")];

    println!("Standard 10-20 EEG channels:");
    for (i, ch) in channels.iter().enumerate() {
        println!("  Ch{}: {} ({} - {})", i, ch.label, ch.channel_type, ch.unit);
    }
    println!();

    // Different channel types
    let ecg_channels = vec![
        ChannelInfo::ecg("Lead I"),
        ChannelInfo::ecg("Lead II"),
        ChannelInfo::ecg("Lead III"),
    ];

    println!("ECG channels:");
    for ch in &ecg_channels {
        println!("  {}: {} ({})", ch.label, ch.channel_type, ch.unit);
    }
    println!();

    // Method 6: Different channel formats
    println!("Method 6: Channel format sizes");
    println!("------------------------------");

    println!("Format byte sizes:");
    println!("  Float32: {} bytes", ChannelFormat::Float32.bytes_per_sample());
    println!("  Float64: {} bytes", ChannelFormat::Float64.bytes_per_sample());
    println!("  Int32: {} bytes", ChannelFormat::Int32.bytes_per_sample());
    println!("  Int16: {} bytes", ChannelFormat::Int16.bytes_per_sample());
    println!("  Int8: {} bytes", ChannelFormat::Int8.bytes_per_sample());
    println!("  String: {} bytes (variable)", ChannelFormat::String.bytes_per_sample());

    println!("\n=== Example Complete ===");
    Ok(())
}
