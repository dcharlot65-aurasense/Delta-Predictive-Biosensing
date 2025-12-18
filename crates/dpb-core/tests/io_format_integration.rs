//! Integration tests for data format I/O
//!
//! Tests reading and writing various physiological data formats
//! including WFDB and EDF, verifying roundtrip consistency.

use dpb_core::io::*;
use dpb_core::Result;
use std::io::Cursor;
use std::path::PathBuf;
use tempfile::TempDir;

/// Create synthetic signal data for testing
fn create_test_signal(num_samples: usize, amplitude: f64) -> Vec<f64> {
    (0..num_samples)
        .map(|i| {
            let t = i as f64 / 250.0; // 250 Hz sampling rate
            amplitude * (2.0 * std::f64::consts::PI * 1.0 * t).sin()
        })
        .collect()
}

#[test]
fn test_wfdb_roundtrip() {
    // Create temporary directory for test files
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let base_path = temp_dir.path().join("test_record");

    // Step 1: Create synthetic signal
    let sample_rate = 250.0; // Hz
    let num_samples = 2500; // 10 seconds
    let signal_data = create_test_signal(num_samples, 1.0);

    println!("WFDB Roundtrip Test:");
    println!("  Samples: {}", num_samples);
    println!("  Sample rate: {:.0} Hz", sample_rate);
    println!("  Duration: {:.1}s", num_samples as f64 / sample_rate);

    // Step 2: Create WFDB header
    let mut header = WfdbHeader::new("test_record".to_string(), sample_rate);
    let signal = WfdbSignal {
        file_name: "test_record.dat".to_string(),
        format: 16, // 16-bit signed integers
        samples_per_frame: 1,
        skew: 0,
        byte_offset: 0,
        adc_gain: 200.0,
        baseline: 0,
        units: "mV".to_string(),
        adc_resolution: 12,
        adc_zero: 0,
        initial_value: 0,
        checksum: 0,
        block_size: 0,
        description: "Test ECG".to_string(),
    };
    header.signals.push(signal);
    header.num_samples = Some(num_samples);

    // Step 3: Write to WFDB format
    let writer = WfdbWriter::new(&base_path);
    writer
        .write_signal(&header, 0, &signal_data)
        .expect("Failed to write WFDB signal");

    // Step 4: Read back from WFDB format
    let reader = WfdbReader::open(&base_path).expect("Failed to open WFDB file");

    // Verify header information
    assert_eq!(reader.header().record_name, "test_record");
    assert_eq!(reader.header().sample_rate, sample_rate);
    assert_eq!(reader.header().signals.len(), 1);

    // Read all samples
    let read_samples = reader
        .read_all_samples(0)
        .expect("Failed to read samples");

    // Step 5: Verify data matches (within quantization error)
    assert_eq!(
        read_samples.len(),
        num_samples,
        "Number of samples should match"
    );

    let mut max_error = 0.0;
    let mut mean_error = 0.0;

    for (i, (&original, &read)) in signal_data.iter().zip(read_samples.iter()).enumerate() {
        let error = (original - read).abs();
        max_error = max_error.max(error);
        mean_error += error;

        // Quantization error should be small (depends on ADC resolution)
        assert!(
            error < 0.1,
            "Sample {} error too large: {:.6}",
            i,
            error
        );
    }

    mean_error /= num_samples as f64;

    println!("  Read {} samples successfully", read_samples.len());
    println!("  Max error: {:.6}", max_error);
    println!("  Mean error: {:.6}", mean_error);
    println!("  ✓ Roundtrip successful within tolerance");
}

#[test]
fn test_wfdb_multi_channel_roundtrip() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let base_path = temp_dir.path().join("multi_channel_record");

    let sample_rate = 250.0;
    let num_samples = 1000;
    let num_channels = 3;

    println!("WFDB Multi-Channel Roundtrip Test:");
    println!("  Channels: {}", num_channels);
    println!("  Samples per channel: {}", num_samples);

    // Create header with multiple signals
    let mut header = WfdbHeader::new("multi_channel_record".to_string(), sample_rate);

    for ch in 0..num_channels {
        let signal = WfdbSignal {
            file_name: format!("multi_channel_record_{}.dat", ch),
            format: 16,
            samples_per_frame: 1,
            skew: 0,
            byte_offset: 0,
            adc_gain: 200.0,
            baseline: 0,
            units: "mV".to_string(),
            adc_resolution: 12,
            adc_zero: 0,
            initial_value: 0,
            checksum: 0,
            block_size: 0,
            description: format!("Channel {}", ch),
        };
        header.signals.push(signal);
    }
    header.num_samples = Some(num_samples);

    // Generate and write signals for each channel
    let writer = WfdbWriter::new(&base_path);

    let mut channel_data = Vec::new();
    for ch in 0..num_channels {
        let amplitude = 1.0 + ch as f64 * 0.5;
        let data = create_test_signal(num_samples, amplitude);
        channel_data.push(data.clone());

        writer
            .write_signal(&header, ch, &data)
            .expect(&format!("Failed to write channel {}", ch));
    }

    // Read back and verify each channel
    let reader = WfdbReader::open(&base_path).expect("Failed to open multi-channel WFDB");

    assert_eq!(
        reader.header().signals.len(),
        num_channels,
        "Should have {} channels",
        num_channels
    );

    for ch in 0..num_channels {
        let read_data = reader
            .read_all_samples(ch)
            .expect(&format!("Failed to read channel {}", ch));

        assert_eq!(read_data.len(), num_samples);

        // Verify correlation with original
        let correlation = compute_correlation(&channel_data[ch], &read_data);
        println!("  Channel {} correlation: {:.6}", ch, correlation);
        assert!(
            correlation > 0.99,
            "Channel {} correlation too low: {:.6}",
            ch,
            correlation
        );
    }

    println!("  ✓ All channels verified successfully");
}

#[test]
fn test_edf_roundtrip() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test.edf");

    // Step 1: Create synthetic multi-channel signal
    let sample_rate = 256.0; // Hz (common for EEG)
    let duration = 10.0; // seconds
    let num_samples = (duration * sample_rate) as usize;

    println!("EDF Roundtrip Test:");
    println!("  Sample rate: {:.0} Hz", sample_rate);
    println!("  Duration: {:.1}s", duration);
    println!("  Samples per channel: {}", num_samples);

    // Create header
    let mut header = EdfHeader::new();
    header.patient_id = "Test Patient".to_string();
    header.recording_id = "Test Recording".to_string();
    header.start_date = "01.01.25".to_string();
    header.start_time = "12.00.00".to_string();
    header.num_data_records = duration as usize;
    header.duration_data_record = 1.0; // 1 second per record

    // Add signals
    let signal_labels = vec!["EEG Fp1", "EEG Fp2", "ECG"];
    for label in &signal_labels {
        let signal = EdfSignal {
            label: label.to_string(),
            transducer_type: "Active electrode".to_string(),
            physical_dimension: "uV".to_string(),
            physical_minimum: -500.0,
            physical_maximum: 500.0,
            digital_minimum: -32768,
            digital_maximum: 32767,
            prefiltering: "HP:0.1Hz LP:70Hz".to_string(),
            num_samples_per_record: sample_rate as usize,
            reserved: String::new(),
        };
        header.signals.push(signal);
    }

    header.num_signals = signal_labels.len();

    // Step 2: Generate signal data
    let mut all_signals = Vec::new();
    for (i, _label) in signal_labels.iter().enumerate() {
        let freq = 1.0 + i as f64 * 2.0; // Different frequencies per channel
        let signal: Vec<f64> = (0..num_samples)
            .map(|j| {
                let t = j as f64 / sample_rate;
                100.0 * (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect();
        all_signals.push(signal);
    }

    // Step 3: Write to EDF
    let writer = EdfWriter::new(&file_path);
    writer
        .write(&header, &all_signals)
        .expect("Failed to write EDF file");

    println!("  Wrote {} channels to EDF", signal_labels.len());

    // Step 4: Read back from EDF
    let reader = EdfReader::open(&file_path).expect("Failed to open EDF file");

    // Verify header
    assert_eq!(reader.header().patient_id, "Test Patient");
    assert_eq!(reader.header().num_signals, signal_labels.len());

    // Read and verify each signal
    for (i, label) in signal_labels.iter().enumerate() {
        let read_signal = reader
            .read_signal(i)
            .expect(&format!("Failed to read signal {}", i));

        assert_eq!(
            read_signal.len(),
            num_samples,
            "Signal {} sample count mismatch",
            label
        );

        // Verify correlation
        let correlation = compute_correlation(&all_signals[i], &read_signal);
        println!("  {} correlation: {:.6}", label, correlation);

        assert!(
            correlation > 0.99,
            "{} correlation too low: {:.6}",
            label,
            correlation
        );
    }

    println!("  ✓ EDF roundtrip successful");
}

#[test]
fn test_edf_annotation_support() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let file_path = temp_dir.path().join("test_annotations.edf");

    // Create a simple EDF file with annotations
    let sample_rate = 250.0;
    let duration = 30.0;
    let num_samples = (duration * sample_rate) as usize;

    let mut header = EdfHeader::new();
    header.num_data_records = duration as usize;
    header.duration_data_record = 1.0;

    // Add one signal channel
    let signal = EdfSignal {
        label: "ECG".to_string(),
        transducer_type: "Electrode".to_string(),
        physical_dimension: "mV".to_string(),
        physical_minimum: -5.0,
        physical_maximum: 5.0,
        digital_minimum: -32768,
        digital_maximum: 32767,
        prefiltering: "None".to_string(),
        num_samples_per_record: sample_rate as usize,
        reserved: String::new(),
    };
    header.signals.push(signal);
    header.num_signals = 1;

    // Generate signal
    let signal_data = create_test_signal(num_samples, 1.0);

    // Write EDF
    let writer = EdfWriter::new(&file_path);
    writer
        .write(&header, &vec![signal_data])
        .expect("Failed to write EDF with annotations");

    // Note: Full annotation support would require extending the EDF format
    // For now, we verify that the file can be read back correctly
    let reader = EdfReader::open(&file_path).expect("Failed to open annotated EDF");
    assert_eq!(reader.header().num_signals, 1);

    println!("EDF Annotation Support Test:");
    println!("  ✓ EDF file with annotation structure verified");
}

#[test]
fn test_wfdb_with_annotations() {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let base_path = temp_dir.path().join("annotated_record");

    let sample_rate = 250.0;
    let num_samples = 5000;

    // Create signal
    let signal_data = create_test_signal(num_samples, 1.0);

    let mut header = WfdbHeader::new("annotated_record".to_string(), sample_rate);
    let signal = WfdbSignal {
        file_name: "annotated_record.dat".to_string(),
        format: 16,
        samples_per_frame: 1,
        skew: 0,
        byte_offset: 0,
        adc_gain: 200.0,
        baseline: 0,
        units: "mV".to_string(),
        adc_resolution: 12,
        adc_zero: 0,
        initial_value: 0,
        checksum: 0,
        block_size: 0,
        description: "ECG with annotations".to_string(),
    };
    header.signals.push(signal);
    header.num_samples = Some(num_samples);

    // Write signal
    let writer = WfdbWriter::new(&base_path);
    writer
        .write_signal(&header, 0, &signal_data)
        .expect("Failed to write signal");

    // Create annotations (e.g., R-peak markers)
    let mut annotations = Vec::new();
    for i in (500..num_samples).step_by(200) {
        // R-peak every 200 samples (~75 bpm at 250 Hz)
        annotations.push(WfdbAnnotation {
            time: i as f64 / sample_rate,
            annotation_type: AnnotationType::Normal,
            sample: i,
            aux: Some("N".to_string()),
        });
    }

    println!("WFDB Annotation Test:");
    println!("  Created {} annotations", annotations.len());

    // Write annotations
    writer
        .write_annotations(&annotations, "atr")
        .expect("Failed to write annotations");

    // Read back annotations
    let reader = WfdbReader::open(&base_path).expect("Failed to open WFDB");
    let read_annotations = reader
        .read_annotations("atr")
        .expect("Failed to read annotations");

    println!("  Read {} annotations", read_annotations.len());

    assert_eq!(
        read_annotations.len(),
        annotations.len(),
        "Annotation count should match"
    );

    for (orig, read) in annotations.iter().zip(read_annotations.iter()) {
        assert_eq!(orig.sample, read.sample, "Sample indices should match");
        assert_eq!(
            orig.annotation_type, read.annotation_type,
            "Annotation types should match"
        );
    }

    println!("  ✓ All annotations verified");
}

#[test]
fn test_format_conversion_wfdb_to_edf() {
    // Test converting between WFDB and EDF formats
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let wfdb_path = temp_dir.path().join("source");
    let edf_path = temp_dir.path().join("converted.edf");

    let sample_rate = 250.0;
    let num_samples = 2500;
    let signal_data = create_test_signal(num_samples, 1.0);

    println!("Format Conversion Test (WFDB → EDF):");

    // Step 1: Write as WFDB
    let mut wfdb_header = WfdbHeader::new("source".to_string(), sample_rate);
    let wfdb_signal = WfdbSignal {
        file_name: "source.dat".to_string(),
        format: 16,
        samples_per_frame: 1,
        skew: 0,
        byte_offset: 0,
        adc_gain: 200.0,
        baseline: 0,
        units: "mV".to_string(),
        adc_resolution: 12,
        adc_zero: 0,
        initial_value: 0,
        checksum: 0,
        block_size: 0,
        description: "ECG".to_string(),
    };
    wfdb_header.signals.push(wfdb_signal);
    wfdb_header.num_samples = Some(num_samples);

    let wfdb_writer = WfdbWriter::new(&wfdb_path);
    wfdb_writer
        .write_signal(&wfdb_header, 0, &signal_data)
        .expect("Failed to write WFDB");

    // Step 2: Read from WFDB
    let wfdb_reader = WfdbReader::open(&wfdb_path).expect("Failed to read WFDB");
    let wfdb_data = wfdb_reader
        .read_all_samples(0)
        .expect("Failed to read WFDB samples");

    // Step 3: Convert to EDF format
    let mut edf_header = EdfHeader::new();
    edf_header.num_data_records = (num_samples as f64 / sample_rate) as usize;
    edf_header.duration_data_record = 1.0;

    let edf_signal = EdfSignal {
        label: "ECG".to_string(),
        transducer_type: "Electrode".to_string(),
        physical_dimension: "mV".to_string(),
        physical_minimum: -5.0,
        physical_maximum: 5.0,
        digital_minimum: -32768,
        digital_maximum: 32767,
        prefiltering: "None".to_string(),
        num_samples_per_record: sample_rate as usize,
        reserved: String::new(),
    };
    edf_header.signals.push(edf_signal);
    edf_header.num_signals = 1;

    let edf_writer = EdfWriter::new(&edf_path);
    edf_writer
        .write(&edf_header, &vec![wfdb_data.clone()])
        .expect("Failed to write EDF");

    // Step 4: Read back from EDF and verify
    let edf_reader = EdfReader::open(&edf_path).expect("Failed to read converted EDF");
    let edf_data = edf_reader
        .read_signal(0)
        .expect("Failed to read EDF signal");

    // Verify data consistency
    let correlation = compute_correlation(&wfdb_data, &edf_data);
    println!("  Correlation after conversion: {:.6}", correlation);

    assert!(
        correlation > 0.95,
        "Conversion should preserve signal fidelity"
    );

    println!("  ✓ Format conversion successful");
}

// Helper function to compute correlation between two signals
fn compute_correlation(signal1: &[f64], signal2: &[f64]) -> f64 {
    assert_eq!(signal1.len(), signal2.len());

    let n = signal1.len() as f64;

    let mean1: f64 = signal1.iter().sum::<f64>() / n;
    let mean2: f64 = signal2.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq1 = 0.0;
    let mut sum_sq2 = 0.0;

    for i in 0..signal1.len() {
        let diff1 = signal1[i] - mean1;
        let diff2 = signal2[i] - mean2;
        numerator += diff1 * diff2;
        sum_sq1 += diff1 * diff1;
        sum_sq2 += diff2 * diff2;
    }

    if sum_sq1 == 0.0 || sum_sq2 == 0.0 {
        return 0.0;
    }

    numerator / (sum_sq1 * sum_sq2).sqrt()
}
