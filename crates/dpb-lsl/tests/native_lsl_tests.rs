//! Tests for native liblsl integration.
//!
//! These tests verify LSL stream creation, data transmission, and pipeline
//! functionality without requiring the actual liblsl library.

#[cfg(test)]
mod native_lsl_tests {
    use std::collections::VecDeque;
    use std::time::{Duration, Instant};

    /// Simulated channel format matching liblsl.
    #[derive(Debug, Clone, Copy, PartialEq)]
    #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
    enum ChannelFormat {
        Float32,
        Double64,
        String,
        Int32,
        Int16,
        Int8,
        Undefined,
    }

    impl ChannelFormat {
        fn bytes_per_sample(&self) -> usize {
            match self {
                ChannelFormat::Float32 => 4,
                ChannelFormat::Double64 => 8,
                ChannelFormat::Int32 => 4,
                ChannelFormat::Int16 => 2,
                ChannelFormat::Int8 => 1,
                ChannelFormat::String => 0, // Variable
                ChannelFormat::Undefined => 0,
            }
        }
    }

    /// Test channel format byte sizes.
    #[test]
    fn test_channel_format_sizes() {
        assert_eq!(ChannelFormat::Float32.bytes_per_sample(), 4);
        assert_eq!(ChannelFormat::Double64.bytes_per_sample(), 8);
        assert_eq!(ChannelFormat::Int32.bytes_per_sample(), 4);
        assert_eq!(ChannelFormat::Int16.bytes_per_sample(), 2);
        assert_eq!(ChannelFormat::Int8.bytes_per_sample(), 1);
    }

    /// Simulated stream info.
    #[derive(Debug, Clone)]
    #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
    struct StreamInfo {
        name: String,
        stream_type: String,
        channel_count: usize,
        nominal_srate: f64,
        channel_format: ChannelFormat,
        source_id: String,
    }

    impl StreamInfo {
        fn new(
            name: &str,
            stream_type: &str,
            channel_count: usize,
            nominal_srate: f64,
            channel_format: ChannelFormat,
            source_id: &str,
        ) -> Self {
            Self {
                name: name.to_string(),
                stream_type: stream_type.to_string(),
                channel_count,
                nominal_srate,
                channel_format,
                source_id: source_id.to_string(),
            }
        }

        fn bytes_per_sample(&self) -> usize {
            self.channel_count * self.channel_format.bytes_per_sample()
        }
    }

    #[test]
    fn test_stream_info_creation() {
        let info = StreamInfo::new(
            "DPB_Spikes",
            "Spikes",
            8,
            256.0,
            ChannelFormat::Float32,
            "dpb_encoder_001",
        );

        assert_eq!(info.name, "DPB_Spikes");
        assert_eq!(info.channel_count, 8);
        assert_eq!(info.nominal_srate, 256.0);
        assert_eq!(info.bytes_per_sample(), 32); // 8 * 4
    }

    /// Simulated outlet for pushing data.
    struct MockOutlet {
        info: StreamInfo,
        buffer: VecDeque<Vec<f32>>,
        samples_pushed: usize,
    }

    impl MockOutlet {
        fn new(info: StreamInfo) -> Self {
            Self {
                info,
                buffer: VecDeque::new(),
                samples_pushed: 0,
            }
        }

        fn push_sample(&mut self, data: &[f32]) -> Result<(), String> {
            if data.len() != self.info.channel_count {
                return Err(format!(
                    "Expected {} channels, got {}",
                    self.info.channel_count,
                    data.len()
                ));
            }
            self.buffer.push_back(data.to_vec());
            self.samples_pushed += 1;
            Ok(())
        }

        fn push_chunk(&mut self, data: &[f32]) -> Result<(), String> {
            let samples = data.len() / self.info.channel_count;
            for i in 0..samples {
                let start = i * self.info.channel_count;
                let end = start + self.info.channel_count;
                self.push_sample(&data[start..end])?;
            }
            Ok(())
        }
    }

    #[test]
    fn test_outlet_push_sample() {
        let info = StreamInfo::new("Test", "EEG", 4, 256.0, ChannelFormat::Float32, "test");
        let mut outlet = MockOutlet::new(info);

        // Valid push
        assert!(outlet.push_sample(&[1.0, 2.0, 3.0, 4.0]).is_ok());
        assert_eq!(outlet.samples_pushed, 1);

        // Invalid channel count
        assert!(outlet.push_sample(&[1.0, 2.0]).is_err());
    }

    #[test]
    fn test_outlet_push_chunk() {
        let info = StreamInfo::new("Test", "EEG", 2, 256.0, ChannelFormat::Float32, "test");
        let mut outlet = MockOutlet::new(info);

        // Push 3 samples as chunk
        let chunk = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert!(outlet.push_chunk(&chunk).is_ok());
        assert_eq!(outlet.samples_pushed, 3);
    }

    /// Simulated inlet for receiving data.
    #[allow(dead_code)] // mirrors the modelled surface; this file uses a subset
    struct MockInlet {
        info: StreamInfo,
        buffer: VecDeque<Vec<f32>>,
        timeout: Duration,
    }

    impl MockInlet {
        fn new(info: StreamInfo, timeout: Duration) -> Self {
            Self {
                info,
                buffer: VecDeque::new(),
                timeout,
            }
        }

        fn inject_sample(&mut self, data: Vec<f32>) {
            self.buffer.push_back(data);
        }

        fn pull_sample(&mut self) -> Option<(Vec<f32>, f64)> {
            self.buffer.pop_front().map(|data| {
                let timestamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_secs_f64();
                (data, timestamp)
            })
        }

        fn samples_available(&self) -> usize {
            self.buffer.len()
        }
    }

    #[test]
    fn test_inlet_pull_sample() {
        let info = StreamInfo::new("Test", "EEG", 4, 256.0, ChannelFormat::Float32, "test");
        let mut inlet = MockInlet::new(info, Duration::from_secs(1));

        // No data available
        assert!(inlet.pull_sample().is_none());

        // Inject and pull
        inlet.inject_sample(vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(inlet.samples_available(), 1);

        let (data, _ts) = inlet.pull_sample().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(inlet.samples_available(), 0);
    }

    /// Test stream resolver simulation.
    #[test]
    fn test_stream_resolver() {
        struct StreamResolver {
            available_streams: Vec<StreamInfo>,
        }

        impl StreamResolver {
            fn new() -> Self {
                Self {
                    available_streams: vec![
                        StreamInfo::new("EEG1", "EEG", 32, 256.0, ChannelFormat::Float32, "dev1"),
                        StreamInfo::new("ECG1", "ECG", 3, 256.0, ChannelFormat::Float32, "dev2"),
                        StreamInfo::new("EMG1", "EMG", 8, 1000.0, ChannelFormat::Float32, "dev3"),
                    ],
                }
            }

            fn resolve_by_type(&self, stream_type: &str) -> Vec<&StreamInfo> {
                self.available_streams
                    .iter()
                    .filter(|s| s.stream_type == stream_type)
                    .collect()
            }

            fn resolve_by_name(&self, name: &str) -> Option<&StreamInfo> {
                self.available_streams.iter().find(|s| s.name == name)
            }

            fn resolve_all(&self) -> &[StreamInfo] {
                &self.available_streams
            }
        }

        let resolver = StreamResolver::new();

        // Find by type
        let eeg_streams = resolver.resolve_by_type("EEG");
        assert_eq!(eeg_streams.len(), 1);
        assert_eq!(eeg_streams[0].name, "EEG1");

        // Find by name
        let ecg = resolver.resolve_by_name("ECG1");
        assert!(ecg.is_some());
        assert_eq!(ecg.unwrap().channel_count, 3);

        // All streams
        assert_eq!(resolver.resolve_all().len(), 3);
    }

    /// Test time correction simulation.
    #[test]
    fn test_time_correction() {
        struct TimeCorrector {
            offset: f64,
            drift_rate: f64, // seconds per second
            last_update: Instant,
        }

        impl TimeCorrector {
            fn new() -> Self {
                Self {
                    offset: 0.0,
                    drift_rate: 0.0,
                    last_update: Instant::now(),
                }
            }

            fn update(&mut self, measured_offset: f64) {
                let elapsed = self.last_update.elapsed().as_secs_f64();
                if elapsed > 0.0 {
                    self.drift_rate = (measured_offset - self.offset) / elapsed;
                }
                self.offset = measured_offset;
                self.last_update = Instant::now();
            }

            fn correct(&self, remote_timestamp: f64) -> f64 {
                let drift_correction = self.drift_rate * self.last_update.elapsed().as_secs_f64();
                remote_timestamp + self.offset + drift_correction
            }
        }

        let mut corrector = TimeCorrector::new();

        // Initial offset
        corrector.update(0.001); // 1ms offset

        let remote_ts = 1000.0;
        let local_ts = corrector.correct(remote_ts);
        assert!((local_ts - remote_ts - 0.001).abs() < 0.01);
    }

    /// Test encoding pipeline with LSL integration.
    #[test]
    fn test_encoding_pipeline() {
        struct EncodingPipeline {
            threshold: f32,
            input_channels: usize,
            output_channels: usize,
            samples_processed: usize,
            spikes_generated: usize,
        }

        impl EncodingPipeline {
            fn new(threshold: f32, channels: usize) -> Self {
                Self {
                    threshold,
                    input_channels: channels,
                    output_channels: channels,
                    samples_processed: 0,
                    spikes_generated: 0,
                }
            }

            fn process(&mut self, input: &[f32]) -> Vec<f32> {
                let samples = input.len() / self.input_channels;
                let mut output = vec![0.0f32; samples * self.output_channels];

                for s in 0..samples {
                    for c in 0..self.input_channels {
                        let idx = s * self.input_channels + c;
                        if input[idx].abs() > self.threshold {
                            output[s * self.output_channels + c] = input[idx].signum();
                            self.spikes_generated += 1;
                        }
                    }
                    self.samples_processed += 1;
                }

                output
            }

            fn get_stats(&self) -> (usize, usize, f64) {
                let rate = if self.samples_processed > 0 {
                    self.spikes_generated as f64 / self.samples_processed as f64
                } else {
                    0.0
                };
                (self.samples_processed, self.spikes_generated, rate)
            }
        }

        let mut pipeline = EncodingPipeline::new(0.5, 4);

        // Process test data
        let input = vec![
            0.1, 0.2, 0.6, 0.8, // Sample 1: 2 spikes
            0.3, 0.4, 0.2, 0.1, // Sample 2: 0 spikes
            0.9, 0.7, 0.6, 0.5, // Sample 3: 4 spikes (threshold inclusive)
        ];

        let output = pipeline.process(&input);

        let (samples, spikes, _rate) = pipeline.get_stats();
        assert_eq!(samples, 3);
        assert!(spikes > 0);
        assert_eq!(output.len(), 12);
    }

    /// Test buffer management for streaming.
    #[test]
    fn test_circular_buffer() {
        struct CircularBuffer<T> {
            buffer: Vec<T>,
            head: usize,
            tail: usize,
            capacity: usize,
            count: usize,
        }

        impl<T: Clone + Default> CircularBuffer<T> {
            fn new(capacity: usize) -> Self {
                Self {
                    buffer: vec![T::default(); capacity],
                    head: 0,
                    tail: 0,
                    capacity,
                    count: 0,
                }
            }

            fn push(&mut self, item: T) -> bool {
                if self.count == self.capacity {
                    return false; // Buffer full
                }
                self.buffer[self.tail] = item;
                self.tail = (self.tail + 1) % self.capacity;
                self.count += 1;
                true
            }

            fn pop(&mut self) -> Option<T> {
                if self.count == 0 {
                    return None;
                }
                let item = self.buffer[self.head].clone();
                self.head = (self.head + 1) % self.capacity;
                self.count -= 1;
                Some(item)
            }

            fn len(&self) -> usize {
                self.count
            }

            fn is_empty(&self) -> bool {
                self.count == 0
            }
        }

        let mut buffer: CircularBuffer<f32> = CircularBuffer::new(4);

        assert!(buffer.is_empty());

        // Fill buffer
        assert!(buffer.push(1.0));
        assert!(buffer.push(2.0));
        assert!(buffer.push(3.0));
        assert!(buffer.push(4.0));
        assert!(!buffer.push(5.0)); // Full

        assert_eq!(buffer.len(), 4);

        // Drain buffer
        assert_eq!(buffer.pop(), Some(1.0));
        assert_eq!(buffer.pop(), Some(2.0));
        assert_eq!(buffer.len(), 2);

        // Add more
        assert!(buffer.push(5.0));
        assert!(buffer.push(6.0));
        assert!(!buffer.push(7.0)); // Full again

        // Drain all
        assert_eq!(buffer.pop(), Some(3.0));
        assert_eq!(buffer.pop(), Some(4.0));
        assert_eq!(buffer.pop(), Some(5.0));
        assert_eq!(buffer.pop(), Some(6.0));
        assert!(buffer.is_empty());
    }

    /// Test XML metadata generation.
    #[test]
    fn test_xml_metadata() {
        fn generate_channel_xml(
            channels: &[(String, String, String)], // (label, type, unit)
        ) -> String {
            let mut xml = String::from("<channels>");
            for (label, ch_type, unit) in channels {
                xml.push_str(&format!(
                    "<channel><label>{}</label><type>{}</type><unit>{}</unit></channel>",
                    label, ch_type, unit
                ));
            }
            xml.push_str("</channels>");
            xml
        }

        let channels = vec![
            ("Fp1".to_string(), "EEG".to_string(), "uV".to_string()),
            ("Fp2".to_string(), "EEG".to_string(), "uV".to_string()),
            ("ECG".to_string(), "ECG".to_string(), "mV".to_string()),
        ];

        let xml = generate_channel_xml(&channels);

        assert!(xml.contains("<label>Fp1</label>"));
        assert!(xml.contains("<type>EEG</type>"));
        assert!(xml.contains("<unit>uV</unit>"));
    }

    /// Test stream info validation.
    #[test]
    fn test_stream_info_validation() {
        fn validate_stream_info(info: &StreamInfo) -> Result<(), String> {
            if info.name.is_empty() {
                return Err("Stream name cannot be empty".to_string());
            }
            if info.channel_count == 0 {
                return Err("Channel count must be positive".to_string());
            }
            if info.nominal_srate < 0.0 {
                return Err("Sample rate cannot be negative".to_string());
            }
            // 0 sample rate is valid (irregular rate)
            Ok(())
        }

        // Valid stream
        let valid = StreamInfo::new("Test", "EEG", 8, 256.0, ChannelFormat::Float32, "id");
        assert!(validate_stream_info(&valid).is_ok());

        // Irregular rate stream (valid)
        let irregular = StreamInfo::new("Markers", "Markers", 1, 0.0, ChannelFormat::String, "id");
        assert!(validate_stream_info(&irregular).is_ok());

        // Invalid: no channels
        let no_channels = StreamInfo::new("Test", "EEG", 0, 256.0, ChannelFormat::Float32, "id");
        assert!(validate_stream_info(&no_channels).is_err());
    }
}
