//! Binary export format for embedded deployment.

use crate::{
    encoder_export::{EncoderExport, EncoderParams, EncoderState, ExportableEncoder},
    error::{ExportError, Result},
    metadata::ModelMetadata,
};
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use tracing::{debug, info};

/// Magic number for DPB binary format.
const MAGIC: &[u8; 4] = b"DPB\x00";

/// Current binary format version.
const FORMAT_VERSION: u16 = 1;

/// Binary exporter for embedded deployment.
///
/// Produces compact binary files optimized for:
/// - Fast loading on resource-constrained devices
/// - Minimal memory footprint
/// - Direct memory mapping
pub struct BinaryExporter {
    /// Model metadata.
    metadata: ModelMetadata,
    /// Compression level (0 = none).
    compression: u8,
}

impl BinaryExporter {
    /// Create a new binary exporter.
    pub fn new(metadata: ModelMetadata) -> Self {
        Self {
            metadata,
            compression: 0,
        }
    }

    /// Enable compression.
    pub fn with_compression(mut self, level: u8) -> Self {
        self.compression = level.min(9);
        self
    }

    /// Export encoder to binary format.
    pub fn export<E: ExportableEncoder>(&self, path: impl AsRef<Path>, encoder: &E) -> Result<()> {
        let path = path.as_ref();
        info!("Exporting encoder to binary: {}", path.display());

        encoder.validate_for_export()?;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        self.write_header(&mut writer)?;

        // Write metadata section
        self.write_metadata_section(&mut writer)?;

        // Write encoder parameters
        self.write_params_section(&mut writer, &encoder.get_params())?;

        // Write state if available
        if let Some(state) = encoder.get_state() {
            self.write_state_section(&mut writer, &state)?;
        }

        // Write footer with checksum
        self.write_footer(&mut writer)?;

        writer.flush()?;
        debug!("Binary export complete");
        Ok(())
    }

    /// Write file header.
    fn write_header<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Magic number
        writer.write_all(MAGIC)?;

        // Format version
        writer.write_all(&FORMAT_VERSION.to_le_bytes())?;

        // Flags (compression, etc.)
        let flags: u16 = if self.compression > 0 { 1 } else { 0 };
        writer.write_all(&flags.to_le_bytes())?;

        // Reserved bytes for future use
        writer.write_all(&[0u8; 8])?;

        Ok(())
    }

    /// Write metadata section.
    fn write_metadata_section<W: Write>(&self, writer: &mut W) -> Result<()> {
        // Section marker
        writer.write_all(b"META")?;

        // Serialize metadata to JSON bytes
        let meta_json = serde_json::to_vec(&self.metadata)?;

        // Write length
        let len = meta_json.len() as u32;
        writer.write_all(&len.to_le_bytes())?;

        // Write data
        writer.write_all(&meta_json)?;

        // Padding to 4-byte boundary
        let padding = (4 - (meta_json.len() % 4)) % 4;
        writer.write_all(&vec![0u8; padding])?;

        Ok(())
    }

    /// Write parameters section.
    fn write_params_section<W: Write>(&self, writer: &mut W, params: &EncoderParams) -> Result<()> {
        // Section marker
        writer.write_all(b"PARM")?;

        // Encoder type (fixed 32 bytes)
        let mut type_bytes = [0u8; 32];
        let type_str = params.encoder_type.as_bytes();
        let len = type_str.len().min(31);
        type_bytes[..len].copy_from_slice(&type_str[..len]);
        writer.write_all(&type_bytes)?;

        // Core parameters
        writer.write_all(&(params.num_channels as u32).to_le_bytes())?;
        writer.write_all(&params.sample_rate.to_le_bytes())?;
        writer.write_all(&(params.num_levels.unwrap_or(0) as u32).to_le_bytes())?;
        writer.write_all(&params.refractory_period.unwrap_or(0.0).to_le_bytes())?;
        writer.write_all(&[if params.adaptive { 1u8 } else { 0u8 }])?;
        writer.write_all(&params.adaptation_rate.unwrap_or(0.0).to_le_bytes())?;

        // Thresholds array
        let num_thresholds = params.thresholds.len() as u32;
        writer.write_all(&num_thresholds.to_le_bytes())?;
        for threshold in &params.thresholds {
            writer.write_all(&threshold.to_le_bytes())?;
        }

        Ok(())
    }

    /// Write state section.
    fn write_state_section<W: Write>(&self, writer: &mut W, state: &EncoderState) -> Result<()> {
        // Section marker
        writer.write_all(b"STAT")?;

        // Number of channels
        let num_channels = state.channel_states.len() as u32;
        writer.write_all(&num_channels.to_le_bytes())?;

        // Per-channel state
        for ch_state in &state.channel_states {
            writer.write_all(&ch_state.last_value.to_le_bytes())?;
            writer.write_all(&ch_state.current_threshold.to_le_bytes())?;
            writer.write_all(&ch_state.current_level.to_le_bytes())?;
            writer.write_all(&ch_state.time_since_spike.to_le_bytes())?;
            writer.write_all(&ch_state.running_mean.to_le_bytes())?;
            writer.write_all(&ch_state.running_var.to_le_bytes())?;
            writer.write_all(&ch_state.spike_count.to_le_bytes())?;
        }

        Ok(())
    }

    /// Write footer with checksum.
    fn write_footer<W: Write>(&self, writer: &mut W) -> Result<()> {
        // End marker
        writer.write_all(b"END\x00")?;

        // Simple checksum (would be CRC32 in production)
        let checksum: u32 = 0; // Placeholder
        writer.write_all(&checksum.to_le_bytes())?;

        Ok(())
    }
}

/// Binary importer for loading exported models.
pub struct BinaryImporter;

impl BinaryImporter {
    /// Import encoder from binary file.
    pub fn import(path: impl AsRef<Path>) -> Result<BinaryModelImport> {
        let path = path.as_ref();
        info!("Importing encoder from binary: {}", path.display());

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        // Read and verify header
        let header = Self::read_header(&mut reader)?;

        // Read metadata
        let metadata = Self::read_metadata_section(&mut reader)?;

        // Read parameters
        let params = Self::read_params_section(&mut reader)?;

        // Try to read state (optional)
        let state = Self::read_state_section(&mut reader).ok();

        debug!("Binary import complete");

        Ok(BinaryModelImport {
            format_version: header.version,
            metadata,
            params,
            state,
        })
    }

    /// Read and verify header.
    fn read_header<R: Read>(reader: &mut R) -> Result<BinaryHeader> {
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if &magic != MAGIC {
            return Err(ExportError::validation("Invalid magic number"));
        }

        let mut version_bytes = [0u8; 2];
        reader.read_exact(&mut version_bytes)?;
        let version = u16::from_le_bytes(version_bytes);

        let mut flags_bytes = [0u8; 2];
        reader.read_exact(&mut flags_bytes)?;
        let flags = u16::from_le_bytes(flags_bytes);

        // Skip reserved bytes
        let mut reserved = [0u8; 8];
        reader.read_exact(&mut reserved)?;

        Ok(BinaryHeader {
            version,
            compressed: (flags & 1) != 0,
        })
    }

    /// Read metadata section.
    fn read_metadata_section<R: Read>(reader: &mut R) -> Result<ModelMetadata> {
        let mut marker = [0u8; 4];
        reader.read_exact(&mut marker)?;

        if &marker != b"META" {
            return Err(ExportError::validation("Expected META section"));
        }

        let mut len_bytes = [0u8; 4];
        reader.read_exact(&mut len_bytes)?;
        let len = u32::from_le_bytes(len_bytes) as usize;

        let mut data = vec![0u8; len];
        reader.read_exact(&mut data)?;

        // Skip padding
        let padding = (4 - (len % 4)) % 4;
        let mut pad = vec![0u8; padding];
        reader.read_exact(&mut pad)?;

        let metadata: ModelMetadata = serde_json::from_slice(&data)?;
        Ok(metadata)
    }

    /// Read parameters section.
    fn read_params_section<R: Read>(reader: &mut R) -> Result<EncoderParams> {
        let mut marker = [0u8; 4];
        reader.read_exact(&mut marker)?;

        if &marker != b"PARM" {
            return Err(ExportError::validation("Expected PARM section"));
        }

        // Encoder type
        let mut type_bytes = [0u8; 32];
        reader.read_exact(&mut type_bytes)?;
        let encoder_type = String::from_utf8_lossy(&type_bytes)
            .trim_end_matches('\0')
            .to_string();

        // Core parameters
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        let mut buf1 = [0u8; 1];

        reader.read_exact(&mut buf4)?;
        let num_channels = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf8)?;
        let sample_rate = f64::from_le_bytes(buf8);

        reader.read_exact(&mut buf4)?;
        let num_levels = u32::from_le_bytes(buf4) as usize;

        reader.read_exact(&mut buf4)?;
        let refractory_period = f32::from_le_bytes(buf4);

        reader.read_exact(&mut buf1)?;
        let adaptive = buf1[0] != 0;

        reader.read_exact(&mut buf4)?;
        let adaptation_rate = f32::from_le_bytes(buf4);

        // Thresholds
        reader.read_exact(&mut buf4)?;
        let num_thresholds = u32::from_le_bytes(buf4) as usize;

        let mut thresholds = Vec::with_capacity(num_thresholds);
        for _ in 0..num_thresholds {
            reader.read_exact(&mut buf4)?;
            thresholds.push(f32::from_le_bytes(buf4));
        }

        Ok(EncoderParams {
            encoder_type,
            num_channels,
            sample_rate,
            thresholds,
            num_levels: if num_levels > 0 {
                Some(num_levels)
            } else {
                None
            },
            refractory_period: if refractory_period > 0.0 {
                Some(refractory_period)
            } else {
                None
            },
            adaptive,
            adaptation_rate: if adaptive {
                Some(adaptation_rate)
            } else {
                None
            },
            extra: std::collections::HashMap::new(),
        })
    }

    /// Read state section.
    fn read_state_section<R: Read>(reader: &mut R) -> Result<EncoderState> {
        let mut marker = [0u8; 4];
        reader.read_exact(&mut marker)?;

        if &marker != b"STAT" {
            return Err(ExportError::validation("Expected STAT section"));
        }

        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];

        reader.read_exact(&mut buf4)?;
        let num_channels = u32::from_le_bytes(buf4) as usize;

        let mut channel_states = Vec::with_capacity(num_channels);
        for _ in 0..num_channels {
            reader.read_exact(&mut buf4)?;
            let last_value = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let current_threshold = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let current_level = i32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let time_since_spike = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let running_mean = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf4)?;
            let running_var = f32::from_le_bytes(buf4);

            reader.read_exact(&mut buf8)?;
            let spike_count = u64::from_le_bytes(buf8);

            channel_states.push(crate::encoder_export::ChannelState {
                last_value,
                current_threshold,
                current_level,
                time_since_spike,
                running_mean,
                running_var,
                spike_count,
            });
        }

        Ok(EncoderState {
            channel_states,
            global: std::collections::HashMap::new(),
        })
    }
}

/// Binary file header.
struct BinaryHeader {
    version: u16,
    compressed: bool,
}

/// Imported binary model.
#[derive(Debug)]
pub struct BinaryModelImport {
    /// Format version.
    pub format_version: u16,
    /// Model metadata.
    pub metadata: ModelMetadata,
    /// Encoder parameters.
    pub params: EncoderParams,
    /// Optional encoder state.
    pub state: Option<EncoderState>,
}
