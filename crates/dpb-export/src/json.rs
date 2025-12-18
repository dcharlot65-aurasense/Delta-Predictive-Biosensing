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
