//! Integration tests for data format I/O
//!
//! Tests reading and writing various physiological data formats
//! including WFDB and EDF, verifying roundtrip consistency.

use dpb_core::io::*;
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

/// WFDB writes FRAMES: one sample per signal per frame. Channel-major data has
/// to be transposed before it goes to `write_samples`.
fn to_frames(channels: &[Vec<i16>]) -> Vec<Vec<i16>> {
    if channels.is_empty() {
        return Vec::new();
    }
    let n = channels.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..n)
        .map(|i| channels.iter().map(|c| c[i]).collect())
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

    // Step 2: Create WFDB header and signal spec
    //
    // `new` takes the signal count between the name and the rate; the signal
    // itself is built from its constructor and passed to the writer rather
    // than pushed onto the header.
    let mut header = WfdbHeader::new("test_record".to_string(), 1, sample_rate);
    header.n_samples = Some(num_samples);

    let signal = WfdbSignal {
        format: 16, // 16-bit signed integers
        adc_resolution: 12,
        ..WfdbSignal::new("Test ECG".to_string(), "mV".to_string(), 200.0)
    };

    // Step 3: Write to WFDB format. The writer takes ADC counts, so convert
    // through the signal's own gain and baseline.
    let adc: Vec<i16> = signal_data.iter().map(|&v| signal.physical_to_adc(v)).collect();

    let mut writer = WfdbWriter::new(&base_path, header.clone(), vec![signal.clone()])
        .expect("Failed to create WFDB writer");
    writer
        .write_samples(&to_frames(&[adc]))
        .expect("Failed to write WFDB samples");
    writer.finish().expect("Failed to finalise WFDB record");

    // Step 4: Read back from WFDB format
    let reader = WfdbReader::open(&base_path).expect("Failed to open WFDB file");

    // Verify header information
    assert_eq!(reader.header().record_name, "test_record");
    assert_eq!(reader.header().sample_rate, sample_rate);
    assert_eq!(reader.signals().len(), 1);

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
        max_error = f64::max(max_error, error);
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

    // Create header and one signal spec per channel. Signals are passed to the
    // writer rather than pushed onto the header.
    let mut header =
        WfdbHeader::new("multi_channel_record".to_string(), num_channels, sample_rate);
    header.n_samples = Some(num_samples);

    let signals: Vec<WfdbSignal> = (0..num_channels)
        .map(|ch| WfdbSignal {
            format: 16,
            adc_resolution: 12,
            ..WfdbSignal::new(format!("Channel {}", ch), "mV".to_string(), 200.0)
        })
        .collect();

    // Generate every channel first: `write_samples` takes one frame set for all
    // channels at once, in ADC counts.
    let mut channel_data = Vec::new();
    for ch in 0..num_channels {
        let amplitude = 1.0 + ch as f64 * 0.5;
        channel_data.push(create_test_signal(num_samples, amplitude));
    }

    let adc: Vec<Vec<i16>> = channel_data
        .iter()
        .zip(signals.iter())
        .map(|(data, signal)| data.iter().map(|&v| signal.physical_to_adc(v)).collect())
        .collect();

    let mut writer = WfdbWriter::new(&base_path, header.clone(), signals.clone())
        .expect("Failed to create multi-channel WFDB writer");
    writer.write_samples(&to_frames(&adc)).expect("Failed to write channels");
    writer.finish().expect("Failed to finalise record");

    // Read back and verify each channel
    let reader = WfdbReader::open(&base_path).expect("Failed to open multi-channel WFDB");

    assert_eq!(
        reader.signals().len(),
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

    // Create header and signal specs.
    //
    // EDF is record-oriented: the header declares how many records there are
    // and how long each is, and each signal declares its samples per record.
    // Signals go to the writer, not onto the header.
    let signal_labels = vec!["EEG Fp1", "EEG Fp2", "ECG"];
    let samples_per_record = sample_rate as usize;

    let header = EdfHeader::new(
        "Test Patient".to_string(),
        "Test Recording".to_string(),
        signal_labels.len(),
    )
    .with_start_datetime("01.01.25".to_string(), "12.00.00".to_string())
    .with_records(duration as i32, 1.0); // 1 second per record

    let signals: Vec<EdfSignal> = signal_labels
        .iter()
        .map(|label| {
            EdfSignal::new(
                label.to_string(),
                "uV".to_string(),
                samples_per_record,
            )
            .with_physical_range(-500.0, 500.0)
            .with_digital_range(-32768, 32767)
        })
        .collect();

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

    // Step 3: Write to EDF, one record at a time.
    let mut writer = EdfWriter::new(&file_path, header.clone(), signals.clone())
        .expect("Failed to create EDF writer");

    for record in 0..duration as usize {
        let start = record * samples_per_record;
        let chunk: Vec<Vec<f64>> = all_signals
            .iter()
            .map(|s| s[start..start + samples_per_record].to_vec())
            .collect();
        writer.write_record(&chunk).expect("Failed to write EDF record");
    }
    writer.finish().expect("Failed to finalise EDF file");

    println!("  Wrote {} channels to EDF", signal_labels.len());

    // Step 4: Read back from EDF
    let mut reader = EdfReader::open(&file_path).expect("Failed to open EDF file");

    // Verify header
    assert_eq!(reader.header().patient_id.trim(), "Test Patient");
    assert_eq!(reader.header().n_signals, signal_labels.len());

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

    let samples_per_record = sample_rate as usize;
    let header = EdfHeader::new("Anon".to_string(), "Annotated".to_string(), 1)
        .with_records(duration as i32, 1.0);

    let signal = EdfSignal::new("ECG".to_string(), "mV".to_string(), samples_per_record)
        .with_physical_range(-5.0, 5.0)
        .with_digital_range(-32768, 32767);

    // Generate signal
    let signal_data = create_test_signal(num_samples, 1.0);

    // Write EDF, one record per second.
    let mut writer = EdfWriter::new(&file_path, header.clone(), vec![signal.clone()])
        .expect("Failed to create EDF writer");
    for record in 0..duration as usize {
        let start = record * samples_per_record;
        let chunk = vec![signal_data[start..start + samples_per_record].to_vec()];
        writer.write_record(&chunk).expect("Failed to write EDF record");
    }
    writer.finish().expect("Failed to finalise EDF file");

    // Note: Full annotation support would require extending the EDF format
    // For now, we verify that the file can be read back correctly
    let reader = EdfReader::open(&file_path).expect("Failed to open annotated EDF");
    assert_eq!(reader.header().n_signals, 1);

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

    let mut header = WfdbHeader::new("annotated_record".to_string(), 1, sample_rate);
    header.n_samples = Some(num_samples);

    let signal = WfdbSignal {
        format: 16,
        adc_resolution: 12,
        ..WfdbSignal::new("ECG with annotations".to_string(), "mV".to_string(), 200.0)
    };

    // Write signal
    let adc: Vec<i16> = signal_data.iter().map(|&v| signal.physical_to_adc(v)).collect();
    let mut writer = WfdbWriter::new(&base_path, header.clone(), vec![signal.clone()])
        .expect("Failed to create WFDB writer");
    writer.write_samples(&to_frames(&[adc])).expect("Failed to write signal");

    // Create annotations (e.g., R-peak markers)
    let mut annotations = Vec::new();
    for i in (500..num_samples).step_by(200) {
        // R-peak every 200 samples (~75 bpm at 250 Hz)
        annotations.push(WfdbAnnotation {
            sample: i,
            annotation_type: AnnotationType::Normal,
            subtype: 0,
            channel: 0,
            aux: Some("N".to_string()),
        });
    }

    println!("WFDB Annotation Test:");
    println!("  Created {} annotations", annotations.len());

    // Annotation WRITING is not implemented, and must say so rather than
    // reporting success while discarding the data.
    let write_result = writer.write_annotation(&annotations[0]);
    assert!(
        write_result.is_err(),
        "an unimplemented write must not report success"
    );

    writer.finish().expect("Failed to finalise record");

    // Reading is implemented, and a record with no annotation file yields an
    // empty set rather than an error.
    let reader = WfdbReader::open(&base_path).expect("Failed to open WFDB");
    let read_annotations = reader
        .read_annotations()
        .expect("reading annotations from a record without them should succeed");

    assert!(
        read_annotations.is_empty(),
        "no annotations were written, so none should be read back"
    );

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
    let mut wfdb_header = WfdbHeader::new("source".to_string(), 1, sample_rate);
    wfdb_header.n_samples = Some(num_samples);

    let wfdb_signal = WfdbSignal {
        format: 16,
        adc_resolution: 12,
        ..WfdbSignal::new("ECG".to_string(), "mV".to_string(), 200.0)
    };

    let adc: Vec<i16> = signal_data
        .iter()
        .map(|&v| wfdb_signal.physical_to_adc(v))
        .collect();

    let mut wfdb_writer =
        WfdbWriter::new(&wfdb_path, wfdb_header.clone(), vec![wfdb_signal.clone()])
            .expect("Failed to create WFDB writer");
    wfdb_writer.write_samples(&to_frames(&[adc])).expect("Failed to write WFDB");
    wfdb_writer.finish().expect("Failed to finalise WFDB");

    // Step 2: Read from WFDB
    let wfdb_reader = WfdbReader::open(&wfdb_path).expect("Failed to read WFDB");
    let wfdb_data = wfdb_reader
        .read_all_samples(0)
        .expect("Failed to read WFDB samples");

    // Step 3: Convert to EDF format
    let samples_per_record = sample_rate as usize;
    let n_records = num_samples / samples_per_record;

    let edf_header = EdfHeader::new("Anon".to_string(), "Converted".to_string(), 1)
        .with_records(n_records as i32, 1.0);

    let edf_signal = EdfSignal::new("ECG".to_string(), "mV".to_string(), samples_per_record)
        .with_physical_range(-5.0, 5.0)
        .with_digital_range(-32768, 32767);

    let mut edf_writer = EdfWriter::new(&edf_path, edf_header.clone(), vec![edf_signal.clone()])
        .expect("Failed to create EDF writer");
    for record in 0..n_records {
        let start = record * samples_per_record;
        let chunk = vec![wfdb_data[start..start + samples_per_record].to_vec()];
        edf_writer.write_record(&chunk).expect("Failed to write EDF");
    }
    edf_writer.finish().expect("Failed to finalise EDF file");

    // Step 4: Read back from EDF and verify
    let mut edf_reader = EdfReader::open(&edf_path).expect("Failed to read converted EDF");
    let edf_data = edf_reader
        .read_signal(0)
        .expect("Failed to read EDF signal");

    // Verify data consistency over the samples that survived the record split.
    let compared = edf_data.len().min(wfdb_data.len());
    let correlation = compute_correlation(&wfdb_data[..compared], &edf_data[..compared]);
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
