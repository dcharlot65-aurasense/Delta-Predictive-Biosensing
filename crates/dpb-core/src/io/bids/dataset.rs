//! BIDS dataset structure and metadata
//!
//! This module provides the main `BidsDataset` type for working with BIDS-formatted
//! datasets, including reading and writing dataset-level metadata.

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use super::subject::BidsSubject;

/// BIDS dataset description metadata (dataset_description.json)
///
/// This contains the required and recommended fields for describing a BIDS dataset.
///
/// # Required Fields
/// - `name`: Name of the dataset
/// - `bids_version`: Version of BIDS specification used
///
/// # Reference
/// See [BIDS dataset_description.json](https://bids-specification.readthedocs.io/en/stable/03-modality-agnostic-files.html#dataset_descriptionjson)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct DatasetDescription {
    /// Name of the dataset
    pub name: String,

    /// Version of the BIDS specification (e.g., "1.8.0")
    #[serde(rename = "BIDSVersion")]
    pub bids_version: String,

    /// Version of the dataset (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_type: Option<String>,

    /// License for distribution (recommended)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// List of individuals who contributed to the creation/curation of the dataset
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authors: Option<Vec<String>>,

    /// Text acknowledging contributions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acknowledgements: Option<String>,

    /// Instructions how researchers using this dataset should acknowledge the original authors
    #[serde(skip_serializing_if = "Option::is_none")]
    pub how_to_acknowledge: Option<String>,

    /// Sources of funding
    #[serde(skip_serializing_if = "Option::is_none")]
    pub funding: Option<Vec<String>>,

    /// Ethics approval information
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ethics_approvals: Option<Vec<String>>,

    /// List of references to publication that contain information on the dataset
    #[serde(skip_serializing_if = "Option::is_none")]
    pub references_and_links: Option<Vec<String>>,

    /// Data use agreement or license
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data_use_agreement: Option<String>,

    /// DOI of the dataset
    #[serde(rename = "DatasetDOI", skip_serializing_if = "Option::is_none")]
    pub dataset_doi: Option<String>,
}

impl DatasetDescription {
    /// Create a new dataset description with required fields
    pub fn new(name: impl Into<String>, bids_version: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bids_version: bids_version.into(),
            dataset_type: None,
            license: None,
            authors: None,
            acknowledgements: None,
            how_to_acknowledge: None,
            funding: None,
            ethics_approvals: None,
            references_and_links: None,
            data_use_agreement: None,
            dataset_doi: None,
        }
    }

    /// Set the dataset type (e.g., "raw", "derivative")
    pub fn with_dataset_type(mut self, dataset_type: impl Into<String>) -> Self {
        self.dataset_type = Some(dataset_type.into());
        self
    }

    /// Set the license
    pub fn with_license(mut self, license: impl Into<String>) -> Self {
        self.license = Some(license.into());
        self
    }

    /// Set the authors
    pub fn with_authors<I, S>(mut self, authors: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.authors = Some(authors.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Set acknowledgements
    pub fn with_acknowledgements(mut self, acknowledgements: impl Into<String>) -> Self {
        self.acknowledgements = Some(acknowledgements.into());
        self
    }

    /// Set funding sources
    pub fn with_funding<I, S>(mut self, funding: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.funding = Some(funding.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Set references and links
    pub fn with_references<I, S>(mut self, references: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.references_and_links = Some(references.into_iter().map(|s| s.into()).collect());
        self
    }

    /// Set DOI
    pub fn with_doi(mut self, doi: impl Into<String>) -> Self {
        self.dataset_doi = Some(doi.into());
        self
    }
}

/// Participant information from participants.tsv
#[derive(Debug, Clone)]
pub struct Participant {
    /// Participant ID (e.g., "sub-01")
    pub participant_id: String,

    /// Age of participant (optional)
    pub age: Option<f64>,

    /// Sex of participant (optional: M, F, O)
    pub sex: Option<String>,

    /// Group classification (optional)
    pub group: Option<String>,

    /// Additional custom columns
    pub extra: HashMap<String, String>,
}

impl Participant {
    /// Create a new participant with just an ID
    pub fn new(participant_id: impl Into<String>) -> Self {
        Self {
            participant_id: participant_id.into(),
            age: None,
            sex: None,
            group: None,
            extra: HashMap::new(),
        }
    }

    /// Get the subject ID without the "sub-" prefix
    pub fn subject_id(&self) -> &str {
        self.participant_id.strip_prefix("sub-").unwrap_or(&self.participant_id)
    }
}

/// BIDS dataset root structure
///
/// Represents a complete BIDS dataset with all metadata and organizational structure.
///
/// # Directory Structure
///
/// ```text
/// dataset_root/
/// ├── dataset_description.json    (required)
/// ├── participants.tsv            (recommended)
/// ├── participants.json           (optional)
/// ├── README                      (recommended)
/// ├── CHANGES                     (recommended)
/// ├── LICENSE                     (recommended)
/// ├── code/                       (optional)
/// ├── derivatives/                (optional)
/// ├── sourcedata/                 (optional)
/// └── sub-<label>/                (subject directories)
/// ```
pub struct BidsDataset {
    /// Root directory of the dataset
    root: PathBuf,

    /// Dataset description metadata
    description: DatasetDescription,

    /// Participants list
    participants: Vec<Participant>,
}

impl BidsDataset {
    /// Open an existing BIDS dataset
    ///
    /// # Arguments
    /// * `path` - Path to the dataset root directory
    ///
    /// # Returns
    /// * `Ok(BidsDataset)` - Successfully opened dataset
    /// * `Err(_)` - Dataset not found or invalid
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let root = path.as_ref().to_path_buf();

        if !root.exists() {
            return Err(DpbError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Dataset root not found: {}", root.display()),
            )));
        }

        // Read dataset_description.json (required)
        let description = Self::read_dataset_description(&root)?;

        // Read participants.tsv (recommended, but not required)
        let participants = Self::read_participants(&root).unwrap_or_default();

        Ok(Self {
            root,
            description,
            participants,
        })
    }

    /// Create a new BIDS dataset
    ///
    /// # Arguments
    /// * `path` - Path where the dataset should be created
    /// * `description` - Dataset description metadata
    pub fn create<P: AsRef<Path>>(
        path: P,
        description: DatasetDescription,
    ) -> Result<Self> {
        let root = path.as_ref().to_path_buf();

        // Create root directory
        fs::create_dir_all(&root).map_err(DpbError::Io)?;

        // Write dataset_description.json
        let desc_path = root.join("dataset_description.json");
        let desc_json = serde_json::to_string_pretty(&description)
            .map_err(|e| DpbError::DataValidation(e.to_string()))?;
        fs::write(&desc_path, desc_json).map_err(DpbError::Io)?;

        // Create recommended files/directories
        let readme_path = root.join("README");
        if !readme_path.exists() {
            fs::write(
                &readme_path,
                format!("# {}\n\nBIDS dataset created with DPB Framework\n", description.name),
            )
            .map_err(DpbError::Io)?;
        }

        Ok(Self {
            root,
            description,
            participants: Vec::new(),
        })
    }

    /// Get the dataset root directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the dataset name
    pub fn name(&self) -> &str {
        &self.description.name
    }

    /// Get the dataset description
    pub fn description(&self) -> &DatasetDescription {
        &self.description
    }

    /// Get the list of participants
    pub fn participants(&self) -> &[Participant] {
        &self.participants
    }

    /// Get all subjects in the dataset
    pub fn subjects(&self) -> Result<Vec<BidsSubject>> {
        let mut subjects = Vec::new();

        for entry in fs::read_dir(&self.root).map_err(DpbError::Io)? {
            let entry = entry.map_err(DpbError::Io)?;
            let path = entry.path();

            if path.is_dir()
                && let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.starts_with("sub-") {
                        let subject_id = name.strip_prefix("sub-").unwrap();
                        subjects.push(BidsSubject::new(&self.root, subject_id));
                    }
        }

        subjects.sort_by(|a, b| a.id().cmp(b.id()));
        Ok(subjects)
    }

    /// Get a specific subject by ID
    pub fn subject(&self, subject_id: &str) -> Result<BidsSubject> {
        let subject = BidsSubject::new(&self.root, subject_id);
        if subject.path().exists() {
            Ok(subject)
        } else {
            Err(DpbError::DataValidation(format!("Subject not found: {}", subject_id)))
        }
    }

    /// Get the derivatives directory path
    pub fn derivatives_path(&self, pipeline_name: &str) -> PathBuf {
        self.root.join("derivatives").join(pipeline_name)
    }

    /// Add a participant to the dataset
    pub fn add_participant(&mut self, participant: Participant) -> Result<()> {
        self.participants.push(participant);
        self.write_participants()
    }

    /// Write the participants.tsv file
    pub fn write_participants(&self) -> Result<()> {
        if self.participants.is_empty() {
            return Ok(());
        }

        let path = self.root.join("participants.tsv");
        let mut file = File::create(&path).map_err(DpbError::Io)?;

        // Write header
        writeln!(file, "participant_id\tage\tsex\tgroup").map_err(DpbError::Io)?;

        // Write participant rows
        for participant in &self.participants {
            writeln!(
                file,
                "{}\t{}\t{}\t{}",
                participant.participant_id,
                participant.age.map(|a| a.to_string()).unwrap_or_else(|| "n/a".to_string()),
                participant.sex.as_deref().unwrap_or("n/a"),
                participant.group.as_deref().unwrap_or("n/a"),
            )
            .map_err(DpbError::Io)?;
        }

        Ok(())
    }

    /// Read dataset_description.json
    fn read_dataset_description(root: &Path) -> Result<DatasetDescription> {
        let path = root.join("dataset_description.json");
        let content = fs::read_to_string(&path).map_err(DpbError::Io)?;
        serde_json::from_str(&content).map_err(|e| DpbError::DataValidation(e.to_string()))
    }

    /// Read participants.tsv
    fn read_participants(root: &Path) -> Result<Vec<Participant>> {
        let path = root.join("participants.tsv");
        if !path.exists() {
            return Ok(Vec::new());
        }

        let file = File::open(&path).map_err(DpbError::Io)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        // Read header
        let header = lines.next()
            .ok_or_else(|| DpbError::DataValidation("Empty participants.tsv".to_string()))?
            .map_err(DpbError::Io)?;
        let columns: Vec<&str> = header.split('\t').collect();

        // Find standard column indices
        let id_idx = columns.iter().position(|&c| c == "participant_id")
            .ok_or_else(|| DpbError::DataValidation("Missing participant_id column".to_string()))?;
        let age_idx = columns.iter().position(|&c| c == "age");
        let sex_idx = columns.iter().position(|&c| c == "sex");
        let group_idx = columns.iter().position(|&c| c == "group");

        // Read participants
        let mut participants = Vec::new();
        for line in lines {
            let line = line.map_err(DpbError::Io)?;
            let values: Vec<&str> = line.split('\t').collect();

            if values.len() <= id_idx {
                continue;
            }

            let participant = Participant {
                participant_id: values[id_idx].to_string(),
                age: age_idx.and_then(|i| values.get(i).and_then(|v| v.parse().ok())),
                sex: sex_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                group: group_idx.and_then(|i| values.get(i).map(|v| v.to_string())),
                extra: HashMap::new(),
            };

            participants.push(participant);
        }

        Ok(participants)
    }

    /// Read the README file content
    pub fn readme(&self) -> Result<String> {
        let path = self.root.join("README");
        fs::read_to_string(&path).map_err(DpbError::Io)
    }

    /// Read the CHANGES file content
    pub fn changes(&self) -> Result<String> {
        let path = self.root.join("CHANGES");
        fs::read_to_string(&path).map_err(DpbError::Io)
    }

    /// Write README file
    pub fn write_readme(&self, content: &str) -> Result<()> {
        let path = self.root.join("README");
        fs::write(&path, content).map_err(DpbError::Io)
    }

    /// Write CHANGES file
    pub fn write_changes(&self, content: &str) -> Result<()> {
        let path = self.root.join("CHANGES");
        fs::write(&path, content).map_err(DpbError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_dataset_description_new() {
        let desc = DatasetDescription::new("Test Dataset", "1.8.0");
        assert_eq!(desc.name, "Test Dataset");
        assert_eq!(desc.bids_version, "1.8.0");
        assert!(desc.authors.is_none());
    }

    #[test]
    fn test_dataset_description_builder() {
        let desc = DatasetDescription::new("Test Dataset", "1.8.0")
            .with_license("CC0")
            .with_authors(vec!["Alice", "Bob"])
            .with_doi("10.1234/test");

        assert_eq!(desc.license, Some("CC0".to_string()));
        assert_eq!(desc.authors, Some(vec!["Alice".to_string(), "Bob".to_string()]));
        assert_eq!(desc.dataset_doi, Some("10.1234/test".to_string()));
    }

    #[test]
    fn test_participant_new() {
        let participant = Participant::new("sub-01");
        assert_eq!(participant.participant_id, "sub-01");
        assert_eq!(participant.subject_id(), "01");
    }

    #[test]
    fn test_create_dataset() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path().join("test_dataset");

        let description = DatasetDescription::new("Test Dataset", "1.8.0")
            .with_license("CC0");

        let dataset = BidsDataset::create(&dataset_path, description)?;

        assert_eq!(dataset.name(), "Test Dataset");
        assert!(dataset_path.join("dataset_description.json").exists());
        assert!(dataset_path.join("README").exists());

        Ok(())
    }

    #[test]
    fn test_open_dataset() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path().join("test_dataset");

        // Create a dataset first
        let description = DatasetDescription::new("Test Dataset", "1.8.0");
        BidsDataset::create(&dataset_path, description)?;

        // Open it
        let dataset = BidsDataset::open(&dataset_path)?;
        assert_eq!(dataset.name(), "Test Dataset");
        assert_eq!(dataset.description().bids_version, "1.8.0");

        Ok(())
    }

    #[test]
    fn test_add_participant() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let dataset_path = temp_dir.path().join("test_dataset");

        let description = DatasetDescription::new("Test Dataset", "1.8.0");
        let mut dataset = BidsDataset::create(&dataset_path, description)?;

        let mut participant = Participant::new("sub-01");
        participant.age = Some(25.0);
        participant.sex = Some("F".to_string());

        dataset.add_participant(participant)?;

        assert_eq!(dataset.participants().len(), 1);
        assert!(dataset_path.join("participants.tsv").exists());

        Ok(())
    }
}
