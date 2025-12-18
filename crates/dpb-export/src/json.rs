//! JSON export functionality.

use crate::{
    encoder_export::{EncoderExport, ExportableEncoder},
    error::Result,
    metadata::ModelMetadata,
};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufReader, BufWriter, Write};
use std::path::Path;
use tracing::{debug, info};

/// JSON exporter for encoder parameters and metadata.
pub struct JsonExporter {
    /// Model metadata.
    metadata: ModelMetadata,
    /// Pretty print output.
    pretty: bool,
}

impl JsonExporter {
    /// Create a new JSON exporter.
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            pretty: true,
        }
    }

    /// Set whether to pretty-print output.
    pub fn with_pretty(mut self, pretty: bool) -> Self {
        self.pretty = pretty;
        self
    }

    /// Export an encoder to JSON.
    pub fn export<E: ExportableEncoder>(&self, path: impl AsRef<Path>, encoder: &E) -> Result<()> {
        let path = path.as_ref();
        info!("Exporting encoder to JSON: {}", path.display());

        // Validate encoder
        encoder.validate_for_export()?;

        // Build export structure
        let export = JsonModelExport {
            metadata: self.metadata.clone(),
            encoder: EncoderExport::new(encoder.get_params())
                .with_state(encoder.get_state().unwrap_or_else(|| {
                    crate::encoder_export::EncoderState::new(encoder.num_channels())
                }))
                .with_checksum(),
        };

        // Write to file
        let file = File::create(path)?;
        let writer = BufWriter::new(file);

        if self.pretty {
            serde_json::to_writer_pretty(writer, &export)?;
        } else {
            serde_json::to_writer(writer, &export)?;
        }

        debug!("JSON export complete");
        Ok(())
    }

    /// Export encoder to JSON string.
    pub fn export_string<E: ExportableEncoder>(&self, encoder: &E) -> Result<String> {
        encoder.validate_for_export()?;

        let export = JsonModelExport {
            metadata: self.metadata.clone(),
            encoder: EncoderExport::new(encoder.get_params())
                .with_state(encoder.get_state().unwrap_or_else(|| {
                    crate::encoder_export::EncoderState::new(encoder.num_channels())
                }))
                .with_checksum(),
        };

        let json = if self.pretty {
            serde_json::to_string_pretty(&export)?
        } else {
            serde_json::to_string(&export)?
        };

        Ok(json)
    }

    /// Import encoder parameters from JSON file.
    pub fn import(path: impl AsRef<Path>) -> Result<JsonModelExport> {
        let path = path.as_ref();
        info!("Importing encoder from JSON: {}", path.display());

        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let export: JsonModelExport = serde_json::from_reader(reader)?;

        // Validate
        export.encoder.validate()?;

        debug!("JSON import complete");
        Ok(export)
    }

    /// Import encoder from JSON string.
    pub fn import_string(json: &str) -> Result<JsonModelExport> {
        let export: JsonModelExport = serde_json::from_str(json)?;
        export.encoder.validate()?;
        Ok(export)
    }
}

/// Complete JSON model export structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonModelExport {
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Encoder export data.
    pub encoder: EncoderExport,
}

impl JsonModelExport {
    /// Get the encoder type.
    pub fn encoder_type(&self) -> &str {
        &self.encoder.params.encoder_type
    }

    /// Get the number of channels.
    pub fn num_channels(&self) -> usize {
        self.encoder.params.num_channels
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> f64 {
        self.encoder.params.sample_rate
    }
}

/// Configuration file format for pipeline setup.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Pipeline name.
    pub name: String,
    /// Pipeline version.
    pub version: String,
    /// Input configuration.
    pub input: InputConfig,
    /// Encoder configurations.
    pub encoders: Vec<EncoderConfig>,
    /// Output configuration.
    pub output: OutputConfig,
}

/// Input source configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputConfig {
    /// Source type (file, stream, device).
    pub source_type: String,
    /// Source path or identifier.
    pub source: String,
    /// Number of channels.
    pub channels: usize,
    /// Sample rate.
    pub sample_rate: f64,
    /// Buffer size.
    pub buffer_size: Option<usize>,
}

/// Encoder configuration in pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    /// Encoder name/identifier.
    pub name: String,
    /// Encoder type.
    pub encoder_type: String,
    /// Encoder parameters.
    pub params: serde_json::Value,
    /// Input channels to process.
    pub input_channels: Option<Vec<usize>>,
}

/// Output destination configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output type (file, stream, display).
    pub output_type: String,
    /// Output path or identifier.
    pub destination: String,
    /// Output format.
    pub format: Option<String>,
}

impl PipelineConfig {
    /// Load configuration from file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path.as_ref())?;
        let reader = BufReader::new(file);
        let config: Self = serde_json::from_reader(reader)?;
        Ok(config)
    }

    /// Save configuration to file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let file = File::create(path.as_ref())?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer_pretty(&mut writer, self)?;
        writer.flush()?;
        Ok(())
    }

    /// Create a default configuration.
    pub fn default_eeg() -> Self {
        Self {
            name: "EEG Processing Pipeline".to_string(),
            version: "1.0".to_string(),
            input: InputConfig {
                source_type: "file".to_string(),
                source: "data/eeg.csv".to_string(),
                channels: 8,
                sample_rate: 256.0,
                buffer_size: Some(1024),
            },
            encoders: vec![EncoderConfig {
                name: "main_encoder".to_string(),
                encoder_type: "level_crossing".to_string(),
                params: serde_json::json!({
                    "threshold": 0.1,
                    "adaptive": true
                }),
                input_channels: None,
            }],
            output: OutputConfig {
                output_type: "file".to_string(),
                destination: "output/spikes.json".to_string(),
                format: Some("json".to_string()),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder_export::MockEncoder;
    use crate::metadata::ModelMetadata;
    use tempfile::tempdir;

    #[test]
    fn test_json_exporter_new() {
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata);
        assert!(exporter.pretty);
    }

    #[test]
    fn test_json_exporter_with_pretty() {
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata).with_pretty(false);
        assert!(!exporter.pretty);
    }

    #[test]
    fn test_json_export_to_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_export.json");

        let mut metadata = ModelMetadata::new();
        metadata.set_name("Test Model");

        let exporter = JsonExporter::new(metadata);
        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);

        let result = exporter.export(&path, &encoder);
        assert!(result.is_ok());
        assert!(path.exists());

        // Verify content
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("Test Model"));
        assert!(content.contains("level_crossing"));
    }

    #[test]
    fn test_json_export_string() {
        let mut metadata = ModelMetadata::new();
        metadata.set_name("String Export Test");

        let exporter = JsonExporter::new(metadata);
        let encoder = MockEncoder::delta(4, 512.0, 0.05, 16);

        let result = exporter.export_string(&encoder);
        assert!(result.is_ok());

        let json = result.unwrap();
        assert!(json.contains("String Export Test"));
        assert!(json.contains("delta"));
        assert!(json.contains("512"));
    }

    #[test]
    fn test_json_export_string_compact() {
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata).with_pretty(false);
        let encoder = MockEncoder::level_crossing(2, 256.0, 0.1);

        let result = exporter.export_string(&encoder);
        assert!(result.is_ok());

        let json = result.unwrap();
        // Compact JSON shouldn't have newlines (except possibly in strings)
        assert!(!json.contains("\n  "));
    }

    #[test]
    fn test_json_import_from_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("import_test.json");

        // First export
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata);
        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);
        exporter.export(&path, &encoder).unwrap();

        // Then import
        let result = JsonExporter::import(&path);
        assert!(result.is_ok());

        let imported = result.unwrap();
        assert_eq!(imported.encoder_type(), "level_crossing");
        assert_eq!(imported.num_channels(), 8);
        assert_eq!(imported.sample_rate(), 256.0);
    }

    #[test]
    fn test_json_import_string() {
        // Create export string
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata);
        let encoder = MockEncoder::delta(4, 512.0, 0.05, 16);
        let json = exporter.export_string(&encoder).unwrap();

        // Import from string
        let result = JsonExporter::import_string(&json);
        assert!(result.is_ok());

        let imported = result.unwrap();
        assert_eq!(imported.encoder_type(), "delta");
        assert_eq!(imported.num_channels(), 4);
    }

    #[test]
    fn test_json_model_export_accessors() {
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata);
        let encoder = MockEncoder::level_crossing(16, 1024.0, 0.08);
        let json = exporter.export_string(&encoder).unwrap();

        let imported = JsonExporter::import_string(&json).unwrap();
        assert_eq!(imported.encoder_type(), "level_crossing");
        assert_eq!(imported.num_channels(), 16);
        assert_eq!(imported.sample_rate(), 1024.0);
    }

    #[test]
    fn test_pipeline_config_default_eeg() {
        let config = PipelineConfig::default_eeg();
        assert_eq!(config.name, "EEG Processing Pipeline");
        assert_eq!(config.version, "1.0");
        assert_eq!(config.input.channels, 8);
        assert_eq!(config.input.sample_rate, 256.0);
        assert_eq!(config.encoders.len(), 1);
        assert_eq!(config.encoders[0].encoder_type, "level_crossing");
    }

    #[test]
    fn test_pipeline_config_save_load() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pipeline_config.json");

        let config = PipelineConfig::default_eeg();
        config.save(&path).unwrap();
        assert!(path.exists());

        let loaded = PipelineConfig::load(&path).unwrap();
        assert_eq!(loaded.name, config.name);
        assert_eq!(loaded.input.channels, config.input.channels);
    }

    #[test]
    fn test_pipeline_config_custom() {
        let config = PipelineConfig {
            name: "Custom Pipeline".to_string(),
            version: "2.0".to_string(),
            input: InputConfig {
                source_type: "stream".to_string(),
                source: "lsl://EEG".to_string(),
                channels: 32,
                sample_rate: 512.0,
                buffer_size: Some(2048),
            },
            encoders: vec![
                EncoderConfig {
                    name: "encoder_1".to_string(),
                    encoder_type: "delta".to_string(),
                    params: serde_json::json!({"threshold": 0.05}),
                    input_channels: Some(vec![0, 1, 2, 3]),
                },
                EncoderConfig {
                    name: "encoder_2".to_string(),
                    encoder_type: "temporal_contrast".to_string(),
                    params: serde_json::json!({"threshold": 0.1}),
                    input_channels: Some(vec![4, 5, 6, 7]),
                },
            ],
            output: OutputConfig {
                output_type: "stream".to_string(),
                destination: "lsl://Spikes".to_string(),
                format: Some("binary".to_string()),
            },
        };

        assert_eq!(config.name, "Custom Pipeline");
        assert_eq!(config.encoders.len(), 2);
        assert_eq!(config.encoders[0].input_channels, Some(vec![0, 1, 2, 3]));
    }

    #[test]
    fn test_input_config_serialization() {
        let config = InputConfig {
            source_type: "file".to_string(),
            source: "test.csv".to_string(),
            channels: 8,
            sample_rate: 256.0,
            buffer_size: Some(1024),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: InputConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.source_type, config.source_type);
        assert_eq!(deserialized.channels, config.channels);
        assert_eq!(deserialized.buffer_size, config.buffer_size);
    }

    #[test]
    fn test_encoder_config_serialization() {
        let config = EncoderConfig {
            name: "test_encoder".to_string(),
            encoder_type: "level_crossing".to_string(),
            params: serde_json::json!({"threshold": 0.1, "adaptive": true}),
            input_channels: Some(vec![0, 1, 2]),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: EncoderConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.name, config.name);
        assert_eq!(deserialized.encoder_type, config.encoder_type);
        assert_eq!(deserialized.input_channels, config.input_channels);
    }

    #[test]
    fn test_output_config_serialization() {
        let config = OutputConfig {
            output_type: "file".to_string(),
            destination: "output.json".to_string(),
            format: Some("json".to_string()),
        };

        let json = serde_json::to_string(&config).unwrap();
        let deserialized: OutputConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(deserialized.output_type, config.output_type);
        assert_eq!(deserialized.format, config.format);
    }

    #[test]
    fn test_json_export_invalid_file_path() {
        let metadata = ModelMetadata::new();
        let exporter = JsonExporter::new(metadata);
        let encoder = MockEncoder::level_crossing(8, 256.0, 0.1);

        let result = exporter.export("/nonexistent/path/file.json", &encoder);
        assert!(result.is_err());
    }

    #[test]
    fn test_json_import_invalid_file() {
        let result = JsonExporter::import("/nonexistent/file.json");
        assert!(result.is_err());
    }

    #[test]
    fn test_json_import_invalid_content() {
        let result = JsonExporter::import_string("not valid json");
        assert!(result.is_err());
    }
}
