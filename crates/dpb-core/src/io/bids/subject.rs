//! BIDS subject-level organization
//!
//! This module handles subject-level directory structures and metadata within a BIDS dataset.

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use super::session::BidsSession;

/// Subject-level metadata (optional)
///
/// Custom metadata that can be stored in subject-level JSON files.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubjectMetadata {
    /// Age of subject at acquisition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<f64>,

    /// Sex of subject (M, F, or O)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sex: Option<String>,

    /// Handedness (left, right, ambidextrous)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub handedness: Option<String>,

    /// Species (homo sapiens, mus musculus, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub species: Option<String>,

    /// Strain (for animal subjects)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strain: Option<String>,

    /// Additional custom fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// BIDS subject representation
///
/// A subject directory contains one or more sessions, or directly contains modality directories
/// if there are no sessions.
///
/// # Directory Structure
///
/// ## Without sessions:
/// ```text
/// sub-<label>/
/// ├── sub-<label>_scans.tsv
/// ├── eeg/
/// ├── meg/
/// └── ...
/// ```
///
/// ## With sessions:
/// ```text
/// sub-<label>/
/// ├── sub-<label>_sessions.tsv
/// ├── ses-<label>/
/// ├── ses-<label>/
/// └── ...
/// ```
#[derive(Debug, Clone)]
pub struct BidsSubject {
    /// Dataset root directory
    dataset_root: PathBuf,

    /// Subject ID (without "sub-" prefix)
    subject_id: String,

    /// Subject directory path
    path: PathBuf,
}

impl BidsSubject {
    /// Create a new subject reference
    ///
    /// # Arguments
    /// * `dataset_root` - Root directory of the BIDS dataset
    /// * `subject_id` - Subject identifier (without "sub-" prefix)
    pub fn new<P: AsRef<Path>>(dataset_root: P, subject_id: impl Into<String>) -> Self {
        let dataset_root = dataset_root.as_ref().to_path_buf();
        let subject_id = subject_id.into();
        let path = dataset_root.join(format!("sub-{}", subject_id));

        Self {
            dataset_root,
            subject_id,
            path,
        }
    }

    /// Get the subject ID (without "sub-" prefix)
    pub fn id(&self) -> &str {
        &self.subject_id
    }

    /// Get the full subject label (with "sub-" prefix)
    pub fn label(&self) -> String {
        format!("sub-{}", self.subject_id)
    }

    /// Get the subject directory path
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if this subject exists in the dataset
    pub fn exists(&self) -> bool {
        self.path.exists() && self.path.is_dir()
    }

    /// Create the subject directory
    pub fn create(&self) -> Result<()> {
        fs::create_dir_all(&self.path).map_err(DpbError::Io)
    }

    /// Check if subject has sessions
    ///
    /// A subject has sessions if it contains subdirectories starting with "ses-"
    pub fn has_sessions(&self) -> Result<bool> {
        if !self.exists() {
            return Ok(false);
        }

        for entry in fs::read_dir(&self.path).map_err(DpbError::Io)? {
            let entry = entry.map_err(DpbError::Io)?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("ses-") {
                        return Ok(true);
                    }
                }
            }
        }

        Ok(false)
    }

    /// Get all sessions for this subject
    ///
    /// Returns an empty vector if the subject has no sessions.
    pub fn sessions(&self) -> Result<Vec<BidsSession>> {
        if !self.exists() {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();

        for entry in fs::read_dir(&self.path).map_err(DpbError::Io)? {
            let entry = entry.map_err(DpbError::Io)?;
            let path = entry.path();

            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if name.starts_with("ses-") {
                        let session_id = name.strip_prefix("ses-").unwrap();
                        sessions.push(BidsSession::new(
                            &self.dataset_root,
                            &self.subject_id,
                            session_id,
                        ));
                    }
                }
            }
        }

        sessions.sort_by(|a, b| a.id().cmp(b.id()));
        Ok(sessions)
    }

    /// Get a specific session by ID
    pub fn session(&self, session_id: &str) -> Result<BidsSession> {
        let session = BidsSession::new(&self.dataset_root, &self.subject_id, session_id);
        if session.path().exists() {
            Ok(session)
        } else {
            Err(DpbError::DataValidation(format!(
                "Session not found: {} for subject {}",
                session_id, self.subject_id
            )))
        }
    }

    /// Get or create a session
    pub fn get_or_create_session(&self, session_id: &str) -> Result<BidsSession> {
        let session = BidsSession::new(&self.dataset_root, &self.subject_id, session_id);
        if !session.exists() {
            session.create()?;
        }
        Ok(session)
    }

    /// Read subject-level metadata if it exists
    ///
    /// Looks for files like `sub-<label>_<suffix>.json` at the subject level.
    pub fn read_metadata(&self, suffix: &str) -> Result<SubjectMetadata> {
        let filename = format!("{}_{}.json", self.label(), suffix);
        let path = self.path.join(filename);

        if !path.exists() {
            return Ok(SubjectMetadata::default());
        }

        let content = fs::read_to_string(&path).map_err(DpbError::Io)?;
        serde_json::from_str(&content).map_err(|e| DpbError::DataValidation(e.to_string()))
    }

    /// Write subject-level metadata
    pub fn write_metadata(&self, suffix: &str, metadata: &SubjectMetadata) -> Result<()> {
        let filename = format!("{}_{}.json", self.label(), suffix);
        let path = self.path.join(filename);

        let json = serde_json::to_string_pretty(metadata)
            .map_err(|e| DpbError::DataValidation(e.to_string()))?;
        fs::write(&path, json).map_err(DpbError::Io)
    }

    /// Get the scans TSV file path
    ///
    /// The scans file lists all imaging acquisitions for this subject.
    pub fn scans_file(&self) -> PathBuf {
        self.path.join(format!("{}_scans.tsv", self.label()))
    }

    /// Get the sessions TSV file path
    ///
    /// The sessions file lists all sessions for this subject with metadata.
    pub fn sessions_file(&self) -> PathBuf {
        self.path.join(format!("{}_sessions.tsv", self.label()))
    }

    /// List all data files for this subject
    ///
    /// Returns paths to all data files (not including JSON sidecars or TSV metadata).
    pub fn data_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        if !self.exists() {
            return Ok(files);
        }

        // If subject has sessions, collect from all sessions
        if self.has_sessions()? {
            for session in self.sessions()? {
                files.extend(session.data_files()?);
            }
        } else {
            // No sessions - look for modality directories directly
            Self::collect_data_files(&self.path, &mut files)?;
        }

        Ok(files)
    }

    /// Helper function to recursively collect data files
    fn collect_data_files(dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(dir).map_err(DpbError::Io)? {
            let entry = entry.map_err(DpbError::Io)?;
            let path = entry.path();

            if path.is_dir() {
                Self::collect_data_files(&path, files)?;
            } else if let Some(ext) = path.extension() {
                // Include data files (not .json or .tsv sidecar files)
                let ext_str = ext.to_string_lossy();
                if ext_str != "json" && ext_str != "tsv" {
                    files.push(path);
                }
            }
        }

        Ok(())
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
    fn test_subject_new() {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");

        assert_eq!(subject.id(), "01");
        assert_eq!(subject.label(), "sub-01");
        assert!(!subject.exists());
    }

    #[test]
    fn test_subject_create() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");

        assert!(!subject.exists());
        subject.create()?;
        assert!(subject.exists());

        Ok(())
    }

    #[test]
    fn test_subject_has_sessions() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");
        subject.create()?;

        assert!(!subject.has_sessions()?);

        // Create a session directory
        let session_dir = subject.path().join("ses-01");
        fs::create_dir(&session_dir).unwrap();

        assert!(subject.has_sessions()?);

        Ok(())
    }

    #[test]
    fn test_subject_sessions() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");
        subject.create()?;

        // Create multiple sessions
        fs::create_dir(subject.path().join("ses-01")).unwrap();
        fs::create_dir(subject.path().join("ses-02")).unwrap();
        fs::create_dir(subject.path().join("ses-03")).unwrap();

        let sessions = subject.sessions()?;
        assert_eq!(sessions.len(), 3);
        assert_eq!(sessions[0].id(), "01");
        assert_eq!(sessions[1].id(), "02");
        assert_eq!(sessions[2].id(), "03");

        Ok(())
    }

    #[test]
    fn test_subject_metadata() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");
        subject.create()?;

        let mut metadata = SubjectMetadata::default();
        metadata.age = Some(25.0);
        metadata.sex = Some("F".to_string());
        metadata.handedness = Some("right".to_string());

        subject.write_metadata("test", &metadata)?;

        let read_metadata = subject.read_metadata("test")?;
        assert_eq!(read_metadata.age, Some(25.0));
        assert_eq!(read_metadata.sex, Some("F".to_string()));

        Ok(())
    }

    #[test]
    fn test_scans_file_path() {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");

        let scans_path = subject.scans_file();
        assert!(scans_path.ends_with("sub-01_scans.tsv"));
    }

    #[test]
    fn test_sessions_file_path() {
        let temp_dir = TempDir::new().unwrap();
        let subject = BidsSubject::new(temp_dir.path(), "01");

        let sessions_path = subject.sessions_file();
        assert!(sessions_path.ends_with("sub-01_sessions.tsv"));
    }
}
