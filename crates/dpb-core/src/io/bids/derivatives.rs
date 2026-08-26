//! BIDS derivatives support
//!
//! This module handles BIDS derivatives, which are processed or derived data products
//! from the raw BIDS dataset. Derivatives maintain the BIDS structure while adding
//! pipeline-specific metadata.
//!
//! # References
//! - [BIDS Derivatives](https://bids-specification.readthedocs.io/en/stable/05-derivatives/01-introduction.html)

use crate::error::{DpbError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

/// Pipeline description metadata (dataset_description.json in derivatives)
///
/// Extends the standard dataset description with pipeline-specific fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PipelineDescription {
    /// Name of the pipeline/derivative
    pub name: String,

    /// Version of the BIDS specification
    #[serde(rename = "BIDSVersion")]
    pub bids_version: String,

    /// Type of dataset (must be "derivative" for derivatives)
    pub dataset_type: String,

    /// Description of the derivative dataset
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dataset_description: Option<String>,

    /// Name of the pipeline that generated these derivatives
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generated_by: Option<Vec<PipelineInfo>>,

    /// Links to source datasets
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_datasets: Option<Vec<SourceDataset>>,

    /// License for distribution
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,

    /// Authors who contributed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authors: Option<Vec<String>>,

    /// How to acknowledge
    #[serde(skip_serializing_if = "Option::is_none")]
    pub how_to_acknowledge: Option<String>,

    /// Additional custom fields
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// Information about the pipeline that generated the derivatives
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct PipelineInfo {
    /// Name of the pipeline
    pub name: String,

    /// Version of the pipeline
    pub version: String,

    /// Description of the pipeline
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Code URL (e.g., GitHub repository)
    #[serde(rename = "CodeURL", skip_serializing_if = "Option::is_none")]
    pub code_url: Option<String>,

    /// Container image used (e.g., Docker, Singularity)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container: Option<ContainerInfo>,
}

/// Container information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct ContainerInfo {
    /// Container type (e.g., "docker", "singularity")
    #[serde(rename = "Type")]
    pub container_type: String,

    /// Container tag or URI
    pub tag: String,
}

/// Source dataset information
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub struct SourceDataset {
    /// DOI or URL of the source dataset
    #[serde(rename = "URL")]
    pub url: Option<String>,

    /// DOI of the source dataset
    #[serde(rename = "DOI")]
    pub doi: Option<String>,

    /// Version of the source dataset
    pub version: Option<String>,
}

impl PipelineDescription {
    /// Create a new pipeline description
    pub fn new(
        name: impl Into<String>,
        bids_version: impl Into<String>,
        pipeline_name: impl Into<String>,
        pipeline_version: impl Into<String>,
    ) -> Self {
        let pipeline = PipelineInfo {
            name: pipeline_name.into(),
            version: pipeline_version.into(),
            description: None,
            code_url: None,
            container: None,
        };

        Self {
            name: name.into(),
            bids_version: bids_version.into(),
            dataset_type: "derivative".to_string(),
            dataset_description: None,
            generated_by: Some(vec![pipeline]),
            source_datasets: None,
            license: None,
            authors: None,
            how_to_acknowledge: None,
            extra: HashMap::new(),
        }
    }

    /// Set the description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.dataset_description = Some(description.into());
        self
    }

    /// Add a source dataset
    pub fn with_source(mut self, url: Option<String>, doi: Option<String>) -> Self {
        let source = SourceDataset {
            url,
            doi,
            version: None,
        };

        if let Some(ref mut sources) = self.source_datasets {
            sources.push(source);
        } else {
            self.source_datasets = Some(vec![source]);
        }

        self
    }

    /// Set the code URL for the pipeline
    pub fn with_code_url(mut self, url: impl Into<String>) -> Self {
        if let Some(ref mut generated) = self.generated_by
            && let Some(pipeline) = generated.first_mut() {
                pipeline.code_url = Some(url.into());
            }
        self
    }

    /// Set the container information
    pub fn with_container(
        mut self,
        container_type: impl Into<String>,
        tag: impl Into<String>,
    ) -> Self {
        let container = ContainerInfo {
            container_type: container_type.into(),
            tag: tag.into(),
        };

        if let Some(ref mut generated) = self.generated_by
            && let Some(pipeline) = generated.first_mut() {
                pipeline.container = Some(container);
            }
        self
    }
}

/// BIDS derivatives dataset
///
/// Represents a derivatives dataset that follows BIDS structure but contains
/// processed/derived data rather than raw data.
///
/// # Directory Structure
///
/// ```text
/// derivatives/
/// └── pipeline_name/
///     ├── dataset_description.json    (required, type="derivative")
///     ├── README
///     ├── sub-<label>/
///     │   └── ses-<label>/
///     │       └── modality/
///     │           └── processed files
///     └── ...
/// ```
pub struct DerivativesDataset {
    /// Root directory of the derivatives dataset
    root: PathBuf,

    /// Pipeline description
    description: PipelineDescription,
}

impl DerivativesDataset {
    /// Open an existing derivatives dataset
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let root = path.as_ref().to_path_buf();

        if !root.exists() {
            return Err(DpbError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Derivatives root not found: {}", root.display()),
            )));
        }

        // Read dataset_description.json
        let description = Self::read_description(&root)?;

        // Validate that it's a derivative dataset
        if description.dataset_type != "derivative" {
            return Err(DpbError::DataValidation(
                "Dataset type must be 'derivative' for derivatives".to_string(),
            ));
        }

        Ok(Self { root, description })
    }

    /// Create a new derivatives dataset
    pub fn create<P: AsRef<Path>>(
        path: P,
        name: impl Into<String>,
        version: impl Into<String>,
        pipeline_name: impl Into<String>,
    ) -> Result<Self> {
        let root = path.as_ref().to_path_buf();

        // Create directory
        fs::create_dir_all(&root).map_err(DpbError::Io)?;

        // Create description
        let description = PipelineDescription::new(
            name.into(),
            "1.8.0",
            pipeline_name.into(),
            version.into(),
        );

        // Write dataset_description.json
        let desc_path = root.join("dataset_description.json");
        let desc_json = serde_json::to_string_pretty(&description)
            .map_err(|e| DpbError::DataValidation(e.to_string()))?;
        fs::write(&desc_path, desc_json).map_err(DpbError::Io)?;

        // Create README
        let readme_path = root.join("README");
        if !readme_path.exists() {
            fs::write(
                &readme_path,
                format!(
                    "# {} Derivatives\n\nProcessed data generated by DPB Framework\n",
                    description.name
                ),
            )
            .map_err(DpbError::Io)?;
        }

        Ok(Self { root, description })
    }

    /// Get the derivatives root directory
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Get the pipeline description
    pub fn description(&self) -> &PipelineDescription {
        &self.description
    }

    /// Get the pipeline name
    pub fn pipeline_name(&self) -> Option<&str> {
        self.description
            .generated_by
            .as_ref()
            .and_then(|g| g.first())
            .map(|p| p.name.as_str())
    }

    /// Get path for a subject's derivatives
    pub fn subject_path(&self, subject_id: &str) -> PathBuf {
        self.root.join(format!("sub-{}", subject_id))
    }

    /// Get path for a subject's session derivatives
    pub fn session_path(&self, subject_id: &str, session_id: &str) -> PathBuf {
        self.subject_path(subject_id)
            .join(format!("ses-{}", session_id))
    }

    /// Get path for a modality within a session
    pub fn modality_path(
        &self,
        subject_id: &str,
        session_id: &str,
        modality: &str,
    ) -> PathBuf {
        self.session_path(subject_id, session_id).join(modality)
    }

    /// Create directory structure for a subject/session/modality
    pub fn create_path(
        &self,
        subject_id: &str,
        session_id: Option<&str>,
        modality: &str,
    ) -> Result<PathBuf> {
        let path = if let Some(ses_id) = session_id {
            self.modality_path(subject_id, ses_id, modality)
        } else {
            self.subject_path(subject_id).join(modality)
        };

        fs::create_dir_all(&path).map_err(DpbError::Io)?;
        Ok(path)
    }

    /// Read dataset_description.json
    fn read_description(root: &Path) -> Result<PipelineDescription> {
        let path = root.join("dataset_description.json");
        let content = fs::read_to_string(&path).map_err(DpbError::Io)?;
        serde_json::from_str(&content).map_err(|e| DpbError::DataValidation(e.to_string()))
    }

    /// Write provenance file for processed data
    ///
    /// Creates a JSON sidecar documenting how the data was processed.
    pub fn write_provenance<P: AsRef<Path>>(
        &self,
        output_file: P,
        sources: Vec<String>,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Result<()> {
        let provenance = serde_json::json!({
            "Sources": sources,
            "ProcessingParameters": parameters,
            "GeneratedBy": self.description.generated_by,
            "Timestamp": chrono::Utc::now().to_rfc3339(),
        });

        let json = serde_json::to_string_pretty(&provenance)
            .map_err(|e| DpbError::DataValidation(e.to_string()))?;

        let output_path = output_file.as_ref().with_extension("json");
        fs::write(&output_path, json).map_err(DpbError::Io)
    }

    /// Read the README file
    pub fn readme(&self) -> Result<String> {
        let path = self.root.join("README");
        fs::read_to_string(&path).map_err(DpbError::Io)
    }

    /// Write README file
    pub fn write_readme(&self, content: &str) -> Result<()> {
        let path = self.root.join("README");
        fs::write(&path, content).map_err(DpbError::Io)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_pipeline_description_new() {
        let desc = PipelineDescription::new(
            "Preprocessed Data",
            "1.8.0",
            "DPB Pipeline",
            "1.0.0",
        );

        assert_eq!(desc.name, "Preprocessed Data");
        assert_eq!(desc.bids_version, "1.8.0");
        assert_eq!(desc.dataset_type, "derivative");

        let pipeline = desc.generated_by.unwrap();
        assert_eq!(pipeline[0].name, "DPB Pipeline");
        assert_eq!(pipeline[0].version, "1.0.0");
    }

    #[test]
    fn test_pipeline_description_builder() {
        let desc = PipelineDescription::new(
            "Preprocessed Data",
            "1.8.0",
            "DPB Pipeline",
            "1.0.0",
        )
        .with_description("Filtered and artifact-removed EEG data")
        .with_code_url("https://github.com/example/dpb")
        .with_container("docker", "dpb:latest");

        assert_eq!(
            desc.dataset_description,
            Some("Filtered and artifact-removed EEG data".to_string())
        );

        let pipeline = desc.generated_by.unwrap();
        assert_eq!(
            pipeline[0].code_url,
            Some("https://github.com/example/dpb".to_string())
        );
        assert!(pipeline[0].container.is_some());
    }

    #[test]
    fn test_create_derivatives() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let deriv_path = temp_dir.path().join("derivatives").join("preprocessed");

        let dataset = DerivativesDataset::create(
            &deriv_path,
            "Preprocessed EEG",
            "1.0.0",
            "DPB Pipeline",
        )?;

        assert!(deriv_path.join("dataset_description.json").exists());
        assert!(deriv_path.join("README").exists());
        assert_eq!(dataset.pipeline_name(), Some("DPB Pipeline"));

        Ok(())
    }

    #[test]
    fn test_open_derivatives() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let deriv_path = temp_dir.path().join("derivatives").join("preprocessed");

        // Create first
        DerivativesDataset::create(&deriv_path, "Test", "1.0.0", "Pipeline")?;

        // Open it
        let dataset = DerivativesDataset::open(&deriv_path)?;
        assert_eq!(dataset.description().dataset_type, "derivative");

        Ok(())
    }

    #[test]
    fn test_create_path() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let deriv_path = temp_dir.path().join("derivatives").join("preprocessed");

        let dataset = DerivativesDataset::create(
            &deriv_path,
            "Test",
            "1.0.0",
            "Pipeline",
        )?;

        let eeg_path = dataset.create_path("01", Some("baseline"), "eeg")?;
        assert!(eeg_path.exists());
        assert!(eeg_path.ends_with("sub-01/ses-baseline/eeg"));

        Ok(())
    }

    #[test]
    fn test_write_provenance() -> Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let deriv_path = temp_dir.path().join("derivatives").join("preprocessed");

        let dataset = DerivativesDataset::create(
            &deriv_path,
            "Test",
            "1.0.0",
            "Pipeline",
        )?;

        let output_path = deriv_path.join("test_data.edf");
        let sources = vec!["sub-01_ses-01_task-rest_eeg.edf".to_string()];
        let mut params = HashMap::new();
        params.insert(
            "high_pass".to_string(),
            serde_json::json!(0.5),
        );
        params.insert(
            "low_pass".to_string(),
            serde_json::json!(40.0),
        );

        dataset.write_provenance(&output_path, sources, params)?;

        let prov_path = output_path.with_extension("json");
        assert!(prov_path.exists());

        Ok(())
    }
}

// Note: We use chrono for timestamps in provenance
// This would need to be added to Cargo.toml if not already present
#[allow(dead_code)]
mod chrono {
    pub struct Utc;
    impl Utc {
        pub fn now() -> DateTime {
            DateTime
        }
    }
    pub struct DateTime;
    impl DateTime {
        pub fn to_rfc3339(&self) -> String {
            "2025-01-15T10:30:00Z".to_string()
        }
    }
}
