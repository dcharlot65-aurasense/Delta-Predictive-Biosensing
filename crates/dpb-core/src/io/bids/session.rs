//! BIDS session-level organization
//!
//! This module handles session-level directory structures and metadata within a BIDS subject.

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::Modality;

/// Session-level metadata (optional)
///
/// Custom metadata that can be stored in session-level JSON files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionMetadata {
    /// Acquisition date (ISO 8601: YYYY-MM-DD)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acq_time: Option<String>,

    /// Experimental condition or session type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,

    /// Notes about the session
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,

    /// Additional custom fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// BIDS session representation
///
/// A session directory contains modality-specific subdirectories (eeg/, meg/, etc.)
/// with the actual data files.
///
/// # Directory Structure
///
/// ```text
/// ses-<label>/
/// ├── ses-<label>_scans.tsv
/// ├── eeg/
/// │   ├── sub-<label>_ses-<label>_task-<label>_eeg.edf
/// │   ├── sub-<label>_ses-<label>_task-<label>_eeg.json
/// │   ├── sub-<label>_ses-<label>_task-<label>_channels.tsv
/// │   └── ...
/// ├── meg/
/// └── ...
/// ```
#[derive(Debug, Clone)]
pub struct BidsSession {
    /// Dataset root directory
    dataset_root: PathBuf,

    /// Subject ID (without "sub-" prefix)
    subject_id: String,

    /// Session ID (without "ses-" prefix)
    session_id: String,

    /// Session directory path
    path: PathBuf,
}

impl BidsSession {
    /// Create a new session reference
    ///
    /// # Arguments
    /// * `dataset_root` - Root directory of the BIDS dataset
    /// * `subject_id` - Subject identifier (without "sub-" prefix)
    /// * `session_id` - Session identifier (without "ses-" prefix)
    pub fn new<P: AsRef<Path>>(
        dataset_root: P,
        subject_id: impl Into<String>,
        session_id: impl Into<String>,
    ) -> Self {
        let dataset_root = dataset_root.as_ref().to_path_buf();
        let subject_id = subject_id.into();
        let session_id = session_id.into();
        let path = dataset_root
            .join(format!("sub-{}", subject_id))
            .join(format!("ses-{}", session_id));

        Self {
            dataset_root,
            subject_id,
            session_id,
            path,
        }
    }

    /// Get the subject ID (without "sub-" prefix)
    pub fn subject_id(&self) -> &str {
        &self.subject_id
    }

    /// Get the session ID (without "ses-" prefix)
    pub fn id(&self) -> &str {
        &self.session_id
    }

    /// Get the full session label (with "ses-" prefix)
    pub fn label(&self) -> String {
        format!("ses-{}", self.session_id)
    }

    /// Get the session directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if this session exists in the dataset
    pub fn exists(&self) -> bool {
        self.path.exists() && self.path.is_dir()
    }

    /// Create the session directory
    pub fn create(&self) -> Result<()> {
        fs::create_dir_all(&self.path).map_err(DpbError::Io)
    }

    /// Get the path to a modality directory
    ///
    /// # Arguments
    /// * `modality` - The modality (eeg, meg, ieeg, etc.)
    ///
    /// # Returns
    /// Path to the modality directory (may not exist)
    pub fn modality_dir(&self, modality: Modality) -> PathBuf {
        self.path.join(modality.dir_name())
    }

    /// Create a modality directory
    pub fn create_modality_dir(&self, modality: Modality) -> Result<PathBuf> {
        let dir = self.modality_dir(modality);
        fs::create_dir_all(&dir).map_err(DpbError::Io)?;
        Ok(dir)
    }

    /// Get all available modalities in this session
    pub fn modalities(&self) -> Result<Vec<Modality>> {
        if !self.exists() {
            return Ok(Vec::new());
        }

        let mut modalities = Vec::new();

        for entry in fs::read_dir(&self.path).map_err(DpbError::Io)? {
            let entry = entry.map_err(DpbError::Io)?;
            let path = entry.path();

            if path.is_dir()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && let Some(modality) = Modality::from_dir_name(name) {
                        modalities.push(modality);
                    }
        }

        Ok(modalities)
    }

    /// Check if a specific modality exists in this session
    pub fn has_modality(&self, modality: Modality) -> bool {
        self.modality_dir(modality).exists()
    }

    /// List all files for a specific modality
    pub fn files_for_modality(&self, modality: Modality) -> Result<Vec<PathBuf>> {
        let modality_dir = self.modality_dir(modality);
        if !modality_dir.exists() {
            return Ok(Vec::new());
        }

        let mut files = Vec::new();
        for entry in fs::read_dir(&modality_dir).map_err(DpbError::Io)? {
            let entry = entry.map_err(DpbError::Io)?;
            let path = entry.path();
            if path.is_file() {
                files.push(path);
            }
        }

        files.sort();
        Ok(files)
    }

    /// List all data files (excluding JSON sidecars and TSV files) for a specific modality
    pub fn data_files_for_modality(&self, modality: Modality) -> Result<Vec<PathBuf>> {
        let all_files = self.files_for_modality(modality)?;
        let data_files: Vec<PathBuf> = all_files
            .into_iter()
            .filter(|p| {
                if let Some(ext) = p.extension() {
                    let ext_str = ext.to_string_lossy();
                    ext_str != "json" && ext_str != "tsv"
                } else {
                    false
                }
            })
            .collect();

        Ok(data_files)
    }

    /// List all data files in this session (across all modalities)
    pub fn data_files(&self) -> Result<Vec<PathBuf>> {
        let mut all_files = Vec::new();

        for modality in self.modalities()? {
            all_files.extend(self.data_files_for_modality(modality)?);
        }

        all_files.sort();
        Ok(all_files)
    }

    /// Read session-level metadata if it exists
    pub fn read_metadata(&self, suffix: &str) -> Result<SessionMetadata> {
        let filename = format!("sub-{}_{}_{}json", self.subject_id, self.label(), suffix);
        let filename = filename.replace("json", ".json");
        let path = self.path.join(filename);

        if !path.exists() {
            return Ok(SessionMetadata::default());
        }

        let content = fs::read_to_string(&path).map_err(DpbError::Io)?;
        serde_json::from_str(&content).map_err(|e| DpbError::Deserialization(e.to_string()))
    }

    /// Write session-level metadata
    pub fn write_metadata(&self, suffix: &str, metadata: &SessionMetadata) -> Result<()> {
        let filename = format!("sub-{}_{}_{}.json", self.subject_id, self.label(), suffix);
        let path = self.path.join(filename);

        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| DpbError::Serialization(e.to_string()))?;
        fs::write(&path, json).map_err(DpbError::Io)
    }

    /// Get the scans TSV file path
    ///
    /// The scans file lists all acquisitions in this session.
    pub fn scans_file(&self) -> PathBuf {
        self.path
            .join(format!("sub-{}_{}_scans.tsv", self.subject_id, self.label()))
    }

    /// Build a filename following BIDS naming convention
    ///
    /// # Arguments
    /// * `modality` - Data modality
    /// * `task` - Task label (optional)
    /// * `run` - Run number (optional)
    /// * `extension` - File extension (e.g., "edf", "json")
    ///
    /// # Returns
    /// Properly formatted BIDS filename
    ///
    /// # Example
    /// ```
    /// # use dpb_core::io::bids::{BidsSession, Modality};
    /// # let session = BidsSession::new("/data", "01", "01");
    /// let filename = session.build_filename(
    ///     Modality::Eeg,
    ///     Some("rest"),
    ///     Some(1),
    ///     "edf"
    /// );
    /// // Returns: "sub-01_ses-01_task-rest_run-01_eeg.edf"
    /// ```
    pub fn build_filename(
        &self,
        modality: Modality,
        task: Option<&str>,
        run: Option<u32>,
        extension: &str,
    ) -> String {
        let mut parts = vec![
            format!("sub-{}", self.subject_id),
            format!("ses-{}", self.session_id),
        ];

        if let Some(task_name) = task {
            parts.push(format!("task-{}", task_name));
        }

        if let Some(run_num) = run {
            parts.push(format!("run-{:02}", run_num));
        }

        parts.push(modality.suffix().to_string());

        format!("{}.{}", parts.join("_"), extension)
    }

    /// Get the dataset root directory
    pub fn dataset_root(&self) -> &Path {
        &self.dataset_root
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_session_new() {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");

        assert_eq!(session.subject_id(), "01");
        assert_eq!(session.id(), "baseline");
        assert_eq!(session.label(), "ses-baseline");
        assert!(!session.exists());
    }

    #[test]
    fn test_session_create() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");

        assert!(!session.exists());
        session.create()?;
        assert!(session.exists());

        Ok(())
    }

    #[test]
    fn test_modality_dir() {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");

        let eeg_dir = session.modality_dir(Modality::Eeg);
        assert!(eeg_dir.ends_with("sub-01/ses-baseline/eeg"));
    }

    #[test]
    fn test_create_modality_dir() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");
        session.create()?;

        let eeg_dir = session.create_modality_dir(Modality::Eeg)?;
        assert!(eeg_dir.exists());
        assert!(eeg_dir.is_dir());

        Ok(())
    }

    #[test]
    fn test_modalities() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");
        session.create()?;

        // Create some modality directories
        session.create_modality_dir(Modality::Eeg)?;
        session.create_modality_dir(Modality::Meg)?;
        session.create_modality_dir(Modality::Physio)?;

        let modalities = session.modalities()?;
        assert_eq!(modalities.len(), 3);
        assert!(modalities.contains(&Modality::Eeg));
        assert!(modalities.contains(&Modality::Meg));
        assert!(modalities.contains(&Modality::Physio));

        Ok(())
    }

    #[test]
    fn test_has_modality() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");
        session.create()?;

        assert!(!session.has_modality(Modality::Eeg));

        session.create_modality_dir(Modality::Eeg)?;
        assert!(session.has_modality(Modality::Eeg));
        assert!(!session.has_modality(Modality::Meg));

        Ok(())
    }

    #[test]
    fn test_build_filename() {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");

        let filename = session.build_filename(Modality::Eeg, Some("rest"), Some(1), "edf");
        assert_eq!(filename, "sub-01_ses-baseline_task-rest_run-01_eeg.edf");

        let filename = session.build_filename(Modality::Eeg, None, None, "json");
        assert_eq!(filename, "sub-01_ses-baseline_eeg.json");

        let filename = session.build_filename(Modality::Meg, Some("visual"), None, "fif");
        assert_eq!(filename, "sub-01_ses-baseline_task-visual_meg.fif");
    }

    #[test]
    fn test_session_metadata() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");
        session.create()?;

        let metadata = SessionMetadata {
            acq_time: Some("2025-01-15T10:30:00".to_string()),
            condition: Some("resting_state".to_string()),
            ..Default::default()
        };

        session.write_metadata("test", &metadata)?;

        let read_metadata = session.read_metadata("test")?;
        assert_eq!(read_metadata.acq_time, Some("2025-01-15T10:30:00".to_string()));
        assert_eq!(read_metadata.condition, Some("resting_state".to_string()));

        Ok(())
    }

    #[test]
    fn test_scans_file_path() {
        let temp_dir = TempDir::new().unwrap();
        let session = BidsSession::new(temp_dir.path(), "01", "baseline");

        let scans_path = session.scans_file();
        assert!(scans_path.ends_with("sub-01_ses-baseline_scans.tsv"));
    }
}
