//! EEG-BIDS specific structures and metadata
//!
//! This module implements the EEG-BIDS extension to the BIDS specification,
//! handling EEG-specific metadata files and data formats.
//!
//! # References
//! - Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain imaging
//!   data structure for electroencephalography. Scientific Data, 6(1), 103.

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

/// EEG-specific metadata from JSON sidecar file
///
/// This metadata accompanies each EEG recording file (*_eeg.json).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct EegMetadata {
    /// Sampling frequency in Hz (required)
    pub sampling_frequency: f64,

    /// Manufacturer of the equipment (recommended)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manufacturer: Option<String>,

    /// Model of the equipment (recommended)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manufacturers_model_name: Option<String>,

    /// Type of cap used (e.g., "actiCAP")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cap_manufacturer: Option<String>,

    /// Model of the cap
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cap_model_name: Option<String>,

    /// Placement scheme of EEG electrodes (e.g., "10-20", "10-10", "10-05")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eeg_placement_scheme: Option<String>,

    /// Reference electrode (e.g., "Cz", "average", "FCz")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eeg_reference: Option<String>,

    /// Ground electrode location
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eeg_ground: Option<String>,

    /// Hardware filters applied (high-pass, low-pass, notch)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware_filters: Option<HashMap<String, FilterDescription>>,

    /// Software filters applied
    #[serde(skip_serializing_if = "Option::is_none")]
    pub software_filters: Option<HashMap<String, FilterDescription>>,

    /// Power line frequency (50 or 60 Hz)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub power_line_frequency: Option<f64>,

    /// Recording duration in seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recording_duration: Option<f64>,

    /// Type of recording (continuous, epoched)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recording_type: Option<String>,

    /// Number of channels
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "EEGChannelCount")]
    pub eeg_channel_count: Option<usize>,

    /// Number of EOG channels
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "EOGChannelCount")]
    pub eog_channel_count: Option<usize>,

    /// Number of ECG channels
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "ECGChannelCount")]
    pub ecg_channel_count: Option<usize>,

    /// Number of EMG channels
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "EMGChannelCount")]
    pub emg_channel_count: Option<usize>,

    /// Number of miscellaneous channels
    #[serde(skip_serializing_if = "Option::is_none")]
    pub misc_channel_count: Option<usize>,

    /// Additional custom fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Filter description for hardware/software filters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterDescription {
    /// Filter type (e.g., "Butterworth", "Chebyshev")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filter_type: Option<String>,

    /// Filter order
    #[serde(skip_serializing_if = "Option::is_none")]
    pub order: Option<u32>,

    /// Cutoff frequency for high-pass/low-pass
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cutoff: Option<f64>,

    /// Lower cutoff for band-pass
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lower_cutoff: Option<f64>,

    /// Upper cutoff for band-pass
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upper_cutoff: Option<f64>,
}

/// Channel information from *_channels.tsv
#[derive(Debug, Clone)]
pub struct ChannelInfo {
    /// Channel name (required)
    pub name: String,

    /// Channel type (required: EEG, EOG, ECG, EMG, MISC, TRIG, etc.)
    pub channel_type: String,

    /// Physical units (required: V, uV, mV, etc.)
    pub units: String,

    /// Low cutoff frequency in Hz (high-pass filter)
    pub low_cutoff: Option<f64>,

    /// High cutoff frequency in Hz (low-pass filter)
    pub high_cutoff: Option<f64>,

    /// Notch filter frequencies
    pub notch: Option<String>,

    /// Sampling frequency if different from main
    pub sampling_frequency: Option<f64>,

    /// Reference channel(s)
    pub reference: Option<String>,

    /// Status (good, bad)
    pub status: Option<String>,

    /// Description
    pub description: Option<String>,

    /// Additional custom columns
    pub extra: HashMap<String, String>,
}

/// Electrode information from *_electrodes.tsv
#[derive(Debug, Clone)]
pub struct ElectrodeInfo {
    /// Electrode name (required, must match channel name)
    pub name: String,

    /// X coordinate (required, in specified units)
    pub x: f64,

    /// Y coordinate (required)
    pub y: f64,

    /// Z coordinate (required)
    pub z: f64,

    /// Electrode material (e.g., "Ag/AgCl")
    pub material: Option<String>,

    /// Electrode impedance in kOhm
    pub impedance: Option<f64>,

    /// Additional custom columns
    pub extra: HashMap<String, String>,
}

/// Coordinate system information from *_coordsystem.json
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct CoordinateSystem {
    /// Coordinate system name (required: "BESA", "Captrak", "CapTrak", "CTF", etc.)
    #[serde(rename = "EEGCoordinateSystem")]
    pub eeg_coordinate_system: String,

    /// Units of the coordinates (required: "m", "mm", "cm")
    #[serde(rename = "EEGCoordinateUnits")]
    pub eeg_coordinate_units: String,

    /// Description of the coordinate system
    #[serde(rename = "EEGCoordinateSystemDescription", skip_serializing_if = "Option::is_none")]
    pub eeg_coordinate_system_description: Option<String>,

    /// Anatomical landmarks used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub anatomical_landmark_coordinates: Option<HashMap<String, [f64; 3]>>,

    /// Fiducials used for co-registration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fiducials_description: Option<String>,
}

/// Event information from *_events.tsv
#[derive(Debug, Clone)]
pub struct EventInfo {
    /// Event onset time in seconds (required)
    pub onset: f64,

    /// Event duration in seconds (required)
    pub duration: f64,

    /// Event type or trial_type (optional but recommended)
    pub trial_type: Option<String>,

    /// Response time
    pub response_time: Option<f64>,

    /// Stimulus file
    pub stim_file: Option<String>,

    /// Event value/code
    pub value: Option<String>,

    /// Sample number
    pub sample: Option<usize>,

    /// Additional custom columns
    pub extra: HashMap<String, String>,
}

/// Task metadata from *_task-<label>_eeg.json
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct TaskMetadata {
    /// Name of the task
    pub task_name: String,

    /// Description of the task
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task_description: Option<String>,

    /// Instructions given to participants
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,

    /// Details about the cognitive atlas task
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cognitive_atlas_task: Option<String>,

    /// Additional custom fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// EEG-BIDS data representation
///
/// This struct provides access to EEG data and all associated BIDS metadata.
pub struct EegBids {
    /// Path to the main EEG data file
    data_file: PathBuf,

    /// EEG metadata from JSON sidecar
    metadata: EegMetadata,

    /// Channel information
    channels: Vec<ChannelInfo>,

    /// Electrode information (optional)
    electrodes: Option<Vec<ElectrodeInfo>>,

    /// Coordinate system (optional)
    coordinate_system: Option<CoordinateSystem>,

    /// Events (optional)
    events: Option<Vec<EventInfo>>,
}

impl EegBids {
    /// Open an EEG-BIDS file set
    ///
    /// # Arguments
    /// * `data_file` - Path to the main EEG data file (e.g., *_eeg.edf)
    ///
    /// This will automatically load all associated metadata files:
    /// - *_eeg.json (required)
    /// - *_channels.tsv (required)
    /// - *_electrodes.tsv (optional)
    /// - *_coordsystem.json (optional)
    /// - *_events.tsv (optional)
    pub fn open<P: AsRef<Path>>(data_file: P) -> Result<Self> {
        let data_file = data_file.as_ref().to_path_buf();

        if !data_file.exists() {
            return Err(DpbError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("EEG data file not found: {}", data_file.display()),
            )));
        }

        // Get base path (remove extension)
        let base_path = data_file.with_extension("");

        // Read required metadata
        let metadata = Self::read_metadata(&base_path)?;
        let channels = Self::read_channels(&base_path)?;

        // Read optional metadata
        let electrodes = Self::read_electrodes(&base_path).ok();
        let coordinate_system = Self::read_coordsystem(&base_path).ok();
        let events = Self::read_events(&base_path).ok();

        Ok(Self {
            data_file,
            metadata,
            channels,
            electrodes,
            coordinate_system,
            events,
        })
    }

    /// Get the data file path
    pub fn data_file(&self) -> &Path {
        &self.data_file
    }

    /// Get the EEG metadata
    pub fn metadata(&self) -> &EegMetadata {
        &self.metadata
    }

    /// Get the channel information
    pub fn channels(&self) -> &[ChannelInfo] {
        &self.channels
    }

    /// Get the electrode information (if available)
    pub fn electrodes(&self) -> Option<&[ElectrodeInfo]> {
        self.electrodes.as_deref()
    }

    /// Get the coordinate system (if available)
    pub fn coordinate_system(&self) -> Option<&CoordinateSystem> {
        self.coordinate_system.as_ref()
    }

    /// Get the events (if available)
    pub fn events(&self) -> Option<&[EventInfo]> {
        self.events.as_deref()
    }

    /// Read EEG metadata from JSON sidecar
    fn read_metadata(base_path: &Path) -> Result<EegMetadata> {
        let json_path = base_path.with_extension("json");
        let content = fs::read_to_string(&json_path).map_err(DpbError::Io)?;
        serde_json::from_str(&content).map_err(|e| DpbError::DataValidation(e.to_string()))
    }

    /// Read channels from TSV file
    fn read_channels(base_path: &Path) -> Result<Vec<ChannelInfo>> {
        let channels_path = PathBuf::from(format!("{}_channels.tsv", base_path.display()));
        if !channels_path.exists() {
            return Err(DpbError::DataValidation("channels.tsv file not found".to_string()));
        }

        let file = File::open(&channels_path).map_err(DpbError::Io)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let header = lines
            .next()
            .ok_or_else(|| DpbError::DataValidation("Empty channels.tsv".to_string()))?
            .map_err(DpbError::Io)?;
        let columns: Vec<&str> = header.split('\t').collect();

        // Find required column indices
        let name_idx = columns
            .iter()
            .position(|&c| c == "name")
            .ok_or_else(|| DpbError::DataValidation("Missing 'name' column in channels.tsv".to_string()))?;
        let type_idx = columns
            .iter()
            .position(|&c| c == "type")
            .ok_or_else(|| DpbError::DataValidation("Missing 'type' column in channels.tsv".to_string()))?;
        let units_idx = columns
            .iter()
            .position(|&c| c == "units")
            .ok_or_else(|| DpbError::DataValidation("Missing 'units' column in channels.tsv".to_string()))?;

        // Find optional column indices
        let low_cutoff_idx = columns.iter().position(|&c| c == "low_cutoff");
        let high_cutoff_idx = columns.iter().position(|&c| c == "high_cutoff");
        let notch_idx = columns.iter().position(|&c| c == "notch");
        let sampling_idx = columns.iter().position(|&c| c == "sampling_frequency");
        let reference_idx = columns.iter().position(|&c| c == "reference");
        let status_idx = columns.iter().position(|&c| c == "status");
        let description_idx = columns.iter().position(|&c| c == "description");

        // Read channels
        let mut channels = Vec::new();
        for line in lines {
            let line = line.map_err(DpbError::Io)?;
            let values: Vec<&str> = line.split('\t').collect();

            let channel = ChannelInfo {
                name: values[name_idx].to_string(),
                channel_type: values[type_idx].to_string(),
                units: values[units_idx].to_string(),
                low_cutoff: low_cutoff_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                high_cutoff: high_cutoff_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                notch: notch_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                sampling_frequency: sampling_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                reference: reference_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                status: status_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                description: description_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                extra: HashMap::new(),
            };

            channels.push(channel);
        }

        Ok(channels)
    }

    /// Read electrodes from TSV file
    fn read_electrodes(base_path: &Path) -> Result<Vec<ElectrodeInfo>> {
        let electrodes_path = PathBuf::from(format!("{}_electrodes.tsv", base_path.display()));
        if !electrodes_path.exists() {
            return Err(DpbError::DataValidation("electrodes.tsv file not found".to_string()));
        }

        let file = File::open(&electrodes_path).map_err(DpbError::Io)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let header = lines
            .next()
            .ok_or_else(|| DpbError::DataValidation("Empty electrodes.tsv".to_string()))?
            .map_err(DpbError::Io)?;
        let columns: Vec<&str> = header.split('\t').collect();

        // Find required column indices
        let name_idx = columns.iter().position(|&c| c == "name")
            .ok_or_else(|| DpbError::DataValidation("Missing 'name' column".to_string()))?;
        let x_idx = columns.iter().position(|&c| c == "x")
            .ok_or_else(|| DpbError::DataValidation("Missing 'x' column".to_string()))?;
        let y_idx = columns.iter().position(|&c| c == "y")
            .ok_or_else(|| DpbError::DataValidation("Missing 'y' column".to_string()))?;
        let z_idx = columns.iter().position(|&c| c == "z")
            .ok_or_else(|| DpbError::DataValidation("Missing 'z' column".to_string()))?;

        // Optional columns
        let material_idx = columns.iter().position(|&c| c == "material");
        let impedance_idx = columns.iter().position(|&c| c == "impedance");

        // Read electrodes
        let mut electrodes = Vec::new();
        for line in lines {
            let line = line.map_err(DpbError::Io)?;
            let values: Vec<&str> = line.split('\t').collect();

            let electrode = ElectrodeInfo {
                name: values[name_idx].to_string(),
                x: values[x_idx].parse().map_err(|e| DpbError::DataValidation(format!("Invalid x: {}", e)))?,
                y: values[y_idx].parse().map_err(|e| DpbError::DataValidation(format!("Invalid y: {}", e)))?,
                z: values[z_idx].parse().map_err(|e| DpbError::DataValidation(format!("Invalid z: {}", e)))?,
                material: material_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                impedance: impedance_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                extra: HashMap::new(),
            };

            electrodes.push(electrode);
        }

        Ok(electrodes)
    }

    /// Read coordinate system from JSON file
    fn read_coordsystem(base_path: &Path) -> Result<CoordinateSystem> {
        let coordsys_path = PathBuf::from(format!("{}_coordsystem.json", base_path.display()));
        let content = fs::read_to_string(&coordsys_path).map_err(DpbError::Io)?;
        serde_json::from_str(&content).map_err(|e| DpbError::DataValidation(e.to_string()))
    }

    /// Read events from TSV file
    fn read_events(base_path: &Path) -> Result<Vec<EventInfo>> {
        let events_path = PathBuf::from(format!("{}_events.tsv", base_path.display()));
        if !events_path.exists() {
            return Err(DpbError::DataValidation("events.tsv file not found".to_string()));
        }

        let file = File::open(&events_path).map_err(DpbError::Io)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let header = lines
            .next()
            .ok_or_else(|| DpbError::DataValidation("Empty events.tsv".to_string()))?
            .map_err(DpbError::Io)?;
        let columns: Vec<&str> = header.split('\t').collect();

        // Find required column indices
        let onset_idx = columns.iter().position(|&c| c == "onset")
            .ok_or_else(|| DpbError::DataValidation("Missing 'onset' column".to_string()))?;
        let duration_idx = columns.iter().position(|&c| c == "duration")
            .ok_or_else(|| DpbError::DataValidation("Missing 'duration' column".to_string()))?;

        // Optional columns
        let trial_type_idx = columns.iter().position(|&c| c == "trial_type");
        let response_time_idx = columns.iter().position(|&c| c == "response_time");
        let stim_file_idx = columns.iter().position(|&c| c == "stim_file");
        let value_idx = columns.iter().position(|&c| c == "value");
        let sample_idx = columns.iter().position(|&c| c == "sample");

        // Read events
        let mut events = Vec::new();
        for line in lines {
            let line = line.map_err(DpbError::Io)?;
            let values: Vec<&str> = line.split('\t').collect();

            let event = EventInfo {
                onset: values[onset_idx].parse().map_err(|e| DpbError::DataValidation(format!("Invalid onset: {}", e)))?,
                duration: values[duration_idx].parse().map_err(|e| DpbError::DataValidation(format!("Invalid duration: {}", e)))?,
                trial_type: trial_type_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                response_time: response_time_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                stim_file: stim_file_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                value: value_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                sample: sample_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                extra: HashMap::new(),
            };

            events.push(event);
        }

        Ok(events)
    }

    /// Write channels TSV file
    pub fn write_channels<P: AsRef<Path>>(
        path: P,
        channels: &[ChannelInfo],
    ) -> Result<()> {
        let mut file = File::create(path.as_ref()).map_err(DpbError::Io)?;

        // Write header
        writeln!(
            file,
            "name\ttype\tunits\tlow_cutoff\thigh_cutoff\tnotch\tsampling_frequency\treference\tstatus\tdescription"
        )
        .map_err(DpbError::Io)?;

        // Write channels
        for channel in channels {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                channel.name,
                channel.channel_type,
                channel.units,
                channel.low_cutoff.map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
                channel.high_cutoff.map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
                channel.notch.as_deref().unwrap_or("n/a"),
                channel.sampling_frequency.map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
                channel.reference.as_deref().unwrap_or("n/a"),
                channel.status.as_deref().unwrap_or("good"),
                channel.description.as_deref().unwrap_or("n/a"),
            )
            .map_err(DpbError::Io)?;
        }

        Ok(())
    }

    /// Write electrodes TSV file
    pub fn write_electrodes<P: AsRef<Path>>(
        path: P,
        electrodes: &[ElectrodeInfo],
    ) -> Result<()> {
        let mut file = File::create(path.as_ref()).map_err(DpbError::Io)?;

        // Write header
        writeln!(file, "name\tx\ty\tz\tmaterial\timpedance").map_err(DpbError::Io)?;

        // Write electrodes
        for electrode in electrodes {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}",
                electrode.name,
                electrode.x,
                electrode.y,
                electrode.z,
                electrode.material.as_deref().unwrap_or("n/a"),
                electrode.impedance.map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
            )
            .map_err(DpbError::Io)?;
        }

        Ok(())
    }

    /// Write events TSV file
    pub fn write_events<P: AsRef<Path>>(
        path: P,
        events: &[EventInfo],
    ) -> Result<()> {
        let mut file = File::create(path.as_ref()).map_err(DpbError::Io)?;

        // Write header
        writeln!(file, "onset\tduration\ttrial_type\tresponse_time\tstim_file\tvalue\tsample")
            .map_err(DpbError::Io)?;

        // Write events
        for event in events {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                event.onset,
                event.duration,
                event.trial_type.as_deref().unwrap_or("n/a"),
                event.response_time.map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
                event.stim_file.as_deref().unwrap_or("n/a"),
                event.value.as_deref().unwrap_or("n/a"),
                event.sample.map(|v| v.to_string()).unwrap_or_else(|| "n/a".to_string()),
            )
            .map_err(DpbError::Io)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_channel_info() {
        let channel = ChannelInfo {
            name: "Fp1".to_string(),
            channel_type: "EEG".to_string(),
            units: "uV".to_string(),
            low_cutoff: Some(0.1),
            high_cutoff: Some(100.0),
            notch: Some("50".to_string()),
            sampling_frequency: None,
            reference: Some("Cz".to_string()),
            status: Some("good".to_string()),
            description: None,
            extra: HashMap::new(),
        };

        assert_eq!(channel.name, "Fp1");
        assert_eq!(channel.channel_type, "EEG");
        assert_eq!(channel.units, "uV");
    }

    #[test]
    fn test_electrode_info() {
        let electrode = ElectrodeInfo {
            name: "Fp1".to_string(),
            x: -0.0295,
            y: 0.0826,
            z: -0.0056,
            material: Some("Ag/AgCl".to_string()),
            impedance: Some(5.2),
            extra: HashMap::new(),
        };

        assert_eq!(electrode.name, "Fp1");
        assert!((electrode.x - (-0.0295)).abs() < 1e-6);
    }

    #[test]
    fn test_write_channels() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let channels_path = temp_dir.path().join("channels.tsv");

        let channels = vec![
            ChannelInfo {
                name: "Fp1".to_string(),
                channel_type: "EEG".to_string(),
                units: "uV".to_string(),
                low_cutoff: Some(0.1),
                high_cutoff: Some(100.0),
                notch: None,
                sampling_frequency: None,
                reference: Some("Cz".to_string()),
                status: Some("good".to_string()),
                description: None,
                extra: HashMap::new(),
            },
            ChannelInfo {
                name: "Fp2".to_string(),
                channel_type: "EEG".to_string(),
                units: "uV".to_string(),
                low_cutoff: Some(0.1),
                high_cutoff: Some(100.0),
                notch: None,
                sampling_frequency: None,
                reference: Some("Cz".to_string()),
                status: Some("good".to_string()),
                description: None,
                extra: HashMap::new(),
            },
        ];

        EegBids::write_channels(&channels_path, &channels)?;
        assert!(channels_path.exists());

        Ok(())
    }

    #[test]
    fn test_write_electrodes() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let electrodes_path = temp_dir.path().join("electrodes.tsv");

        let electrodes = vec![
            ElectrodeInfo {
                name: "Fp1".to_string(),
                x: -0.0295,
                y: 0.0826,
                z: -0.0056,
                material: Some("Ag/AgCl".to_string()),
                impedance: Some(5.2),
                extra: HashMap::new(),
            },
        ];

        EegBids::write_electrodes(&electrodes_path, &electrodes)?;
        assert!(electrodes_path.exists());

        Ok(())
    }

    #[test]
    fn test_write_events() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let events_path = temp_dir.path().join("events.tsv");

        let events = vec![
            EventInfo {
                onset: 0.0,
                duration: 0.5,
                trial_type: Some("visual".to_string()),
                response_time: Some(0.3),
                stim_file: None,
                value: Some("1".to_string()),
                sample: Some(0),
                extra: HashMap::new(),
            },
            EventInfo {
                onset: 2.0,
                duration: 0.5,
                trial_type: Some("auditory".to_string()),
                response_time: Some(0.4),
                stim_file: None,
                value: Some("2".to_string()),
                sample: Some(1000),
                extra: HashMap::new(),
            },
        ];

        EegBids::write_events(&events_path, &events)?;
        assert!(events_path.exists());

        Ok(())
    }
}
