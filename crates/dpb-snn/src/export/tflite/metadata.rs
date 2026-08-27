//! TensorFlow Lite model metadata
//!
//! Provides metadata support for TFLite models including descriptions,
//! input/output information, and associated files.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// TensorFlow Lite model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TFLiteMetadata {
    /// Model name
    pub name: String,
    /// Model description
    pub description: ModelDescription,
    /// Model version
    pub version: String,
    /// Author information
    pub author: Option<String>,
    /// License
    pub license: Option<String>,
    /// Input tensor metadata
    pub inputs: Vec<TensorMetadata>,
    /// Output tensor metadata
    pub outputs: Vec<TensorMetadata>,
    /// Associated files (labels, config, etc.)
    pub associated_files: Vec<AssociatedFile>,
    /// Custom properties
    pub custom_properties: HashMap<String, String>,
}

impl TFLiteMetadata {
    /// Create new metadata
    pub fn new(name: String, description: String) -> Self {
        Self {
            name,
            description: ModelDescription::new(description),
            version: "1.0.0".to_string(),
            author: None,
            license: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
            associated_files: Vec::new(),
            custom_properties: HashMap::new(),
        }
    }

    /// Set version
    pub fn with_version(mut self, version: String) -> Self {
        self.version = version;
        self
    }

    /// Set author
    pub fn with_author(mut self, author: String) -> Self {
        self.author = Some(author);
        self
    }

    /// Set license
    pub fn with_license(mut self, license: String) -> Self {
        self.license = Some(license);
        self
    }

    /// Add input metadata
    pub fn add_input(&mut self, metadata: TensorMetadata) {
        self.inputs.push(metadata);
    }

    /// Add output metadata
    pub fn add_output(&mut self, metadata: TensorMetadata) {
        self.outputs.push(metadata);
    }

    /// Add associated file
    pub fn add_file(&mut self, file: AssociatedFile) {
        self.associated_files.push(file);
    }

    /// Add custom property
    pub fn add_property(&mut self, key: String, value: String) {
        self.custom_properties.insert(key, value);
    }

    /// Serialize to JSON
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize metadata: {}", e))
    }

    /// Serialize to JSON bytes
    pub fn to_json_bytes(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("Failed to serialize metadata: {}", e))
    }

    /// Deserialize from JSON
    pub fn from_json(json: &str) -> Result<Self, String> {
        serde_json::from_str(json).map_err(|e| format!("Failed to parse metadata: {}", e))
    }

    /// Validate metadata
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        if self.name.is_empty() {
            errors.push("Model name cannot be empty".to_string());
        }

        if self.version.is_empty() {
            errors.push("Model version cannot be empty".to_string());
        }

        // Validate input metadata
        for (i, input) in self.inputs.iter().enumerate() {
            if let Err(e) = input.validate() {
                errors.push(format!("Input {}: {}", i, e));
            }
        }

        // Validate output metadata
        for (i, output) in self.outputs.iter().enumerate() {
            if let Err(e) = output.validate() {
                errors.push(format!("Output {}: {}", i, e));
            }
        }

        // Validate associated files
        for (i, file) in self.associated_files.iter().enumerate() {
            if let Err(e) = file.validate() {
                errors.push(format!("Associated file {}: {}", i, e));
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Model description
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDescription {
    /// Short description
    pub summary: String,
    /// Detailed description
    pub details: Option<String>,
    /// Model type (e.g., "image_classification", "object_detection")
    pub model_type: Option<String>,
    /// Task description
    pub task: Option<String>,
}

impl ModelDescription {
    /// Create new description
    pub fn new(summary: String) -> Self {
        Self {
            summary,
            details: None,
            model_type: None,
            task: None,
        }
    }

    /// Set detailed description
    pub fn with_details(mut self, details: String) -> Self {
        self.details = Some(details);
        self
    }

    /// Set model type
    pub fn with_model_type(mut self, model_type: String) -> Self {
        self.model_type = Some(model_type);
        self
    }

    /// Set task
    pub fn with_task(mut self, task: String) -> Self {
        self.task = Some(task);
        self
    }
}

/// Tensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorMetadata {
    /// Tensor name
    pub name: String,
    /// Description
    pub description: Option<String>,
    /// Content type (e.g., "image", "audio", "feature_vector")
    pub content_type: ContentType,
    /// Normalization parameters
    pub normalization: Option<NormalizationParams>,
    /// Associated files (e.g., label maps)
    pub associated_file_names: Vec<String>,
}

impl TensorMetadata {
    /// Create new tensor metadata
    pub fn new(name: String, content_type: ContentType) -> Self {
        Self {
            name,
            description: None,
            content_type,
            normalization: None,
            associated_file_names: Vec::new(),
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Set normalization
    pub fn with_normalization(mut self, norm: NormalizationParams) -> Self {
        self.normalization = Some(norm);
        self
    }

    /// Add associated file name
    pub fn add_associated_file(mut self, filename: String) -> Self {
        self.associated_file_names.push(filename);
        self
    }

    /// Validate tensor metadata
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("Tensor name cannot be empty".to_string());
        }

        if let Some(ref norm) = self.normalization {
            norm.validate()?;
        }

        Ok(())
    }
}

/// Content type for tensors
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContentType {
    /// Image data (RGB, grayscale, etc.)
    Image { color_space: ColorSpace },
    /// Audio data
    Audio { sample_rate: u32, channels: u8 },
    /// Feature vector
    FeatureVector,
    /// Bounding boxes
    BoundingBoxes,
    /// Text/string
    Text,
    /// Custom content type
    Custom(String),
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
/// Color space for image data
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSpace {
    RGB,
    BGR,
    Grayscale,
    RGBA,
}

/// Normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Mean values (per channel)
    pub mean: Vec<f32>,
    /// Standard deviation values (per channel)
    pub std: Vec<f32>,
}

impl NormalizationParams {
    /// Create new normalization parameters
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        Self { mean, std }
    }

    /// Create from single values (broadcast to all channels)
    pub fn from_single(mean: f32, std: f32) -> Self {
        Self {
            mean: vec![mean],
            std: vec![std],
        }
    }

    /// ImageNet normalization
    pub fn imagenet() -> Self {
        Self {
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    /// Validate normalization parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.mean.is_empty() {
            return Err("Mean values cannot be empty".to_string());
        }

        if self.std.is_empty() {
            return Err("Std values cannot be empty".to_string());
        }

        if self.mean.len() != self.std.len() {
            return Err("Mean and std must have same length".to_string());
        }

        // Check for zero or negative std
        if self.std.iter().any(|&s| s <= 0.0) {
            return Err("Standard deviation must be positive".to_string());
        }

        Ok(())
    }
}

/// Associated file (labels, config, etc.)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssociatedFile {
    /// File name
    pub name: String,
    /// File description
    pub description: Option<String>,
    /// File type
    pub file_type: FileType,
    /// File contents (optional, for embedding)
    pub contents: Option<Vec<u8>>,
}

impl AssociatedFile {
    /// Create new associated file
    pub fn new(name: String, file_type: FileType) -> Self {
        Self {
            name,
            description: None,
            file_type,
            contents: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, desc: String) -> Self {
        self.description = Some(desc);
        self
    }

    /// Set contents
    pub fn with_contents(mut self, contents: Vec<u8>) -> Self {
        self.contents = Some(contents);
        self
    }

    /// Create label file
    pub fn label_file(name: String, labels: Vec<String>) -> Self {
        let contents = labels.join("\n").into_bytes();
        Self {
            name,
            description: Some("Class labels".to_string()),
            file_type: FileType::Labels,
            contents: Some(contents),
        }
    }

    /// Validate associated file
    pub fn validate(&self) -> Result<(), String> {
        if self.name.is_empty() {
            return Err("File name cannot be empty".to_string());
        }

        Ok(())
    }
}

/// File type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileType {
    /// Label file (text file with class names)
    Labels,
    /// Configuration file (JSON, YAML, etc.)
    Config,
    /// Vocabulary file
    Vocabulary,
    /// Custom file type
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_creation() {
        let metadata = TFLiteMetadata::new(
            "test_model".to_string(),
            "Test model description".to_string(),
        );

        assert_eq!(metadata.name, "test_model");
        assert_eq!(metadata.description.summary, "Test model description");
        assert_eq!(metadata.version, "1.0.0");
    }

    #[test]
    fn test_metadata_builder() {
        let metadata = TFLiteMetadata::new("test_model".to_string(), "Description".to_string())
            .with_version("2.0.0".to_string())
            .with_author("Test Author".to_string())
            .with_license("MIT".to_string());

        assert_eq!(metadata.version, "2.0.0");
        assert_eq!(metadata.author, Some("Test Author".to_string()));
        assert_eq!(metadata.license, Some("MIT".to_string()));
    }

    #[test]
    fn test_metadata_add_input_output() {
        let mut metadata = TFLiteMetadata::new("test".to_string(), "Test".to_string());

        let input = TensorMetadata::new(
            "input".to_string(),
            ContentType::Image {
                color_space: ColorSpace::RGB,
            },
        );
        metadata.add_input(input);

        let output = TensorMetadata::new("output".to_string(), ContentType::FeatureVector);
        metadata.add_output(output);

        assert_eq!(metadata.inputs.len(), 1);
        assert_eq!(metadata.outputs.len(), 1);
    }

    #[test]
    fn test_metadata_custom_properties() {
        let mut metadata = TFLiteMetadata::new("test".to_string(), "Test".to_string());

        metadata.add_property("key1".to_string(), "value1".to_string());
        metadata.add_property("key2".to_string(), "value2".to_string());

        assert_eq!(metadata.custom_properties.len(), 2);
        assert_eq!(
            metadata.custom_properties.get("key1"),
            Some(&"value1".to_string())
        );
    }

    #[test]
    fn test_metadata_json_roundtrip() {
        let metadata = TFLiteMetadata::new("test".to_string(), "Test".to_string());

        let json = metadata.to_json().unwrap();
        let loaded = TFLiteMetadata::from_json(&json).unwrap();

        assert_eq!(loaded.name, metadata.name);
        assert_eq!(loaded.version, metadata.version);
    }

    #[test]
    fn test_metadata_validation_empty_name() {
        let metadata = TFLiteMetadata::new("".to_string(), "Test".to_string());

        let result = metadata.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_model_description() {
        let desc = ModelDescription::new("Summary".to_string())
            .with_details("Detailed description".to_string())
            .with_model_type("classification".to_string())
            .with_task("image_classification".to_string());

        assert_eq!(desc.summary, "Summary");
        assert_eq!(desc.details, Some("Detailed description".to_string()));
        assert_eq!(desc.model_type, Some("classification".to_string()));
        assert_eq!(desc.task, Some("image_classification".to_string()));
    }

    #[test]
    fn test_tensor_metadata_creation() {
        let tensor = TensorMetadata::new(
            "input".to_string(),
            ContentType::Image {
                color_space: ColorSpace::RGB,
            },
        );

        assert_eq!(tensor.name, "input");
        assert!(matches!(
            tensor.content_type,
            ContentType::Image {
                color_space: ColorSpace::RGB
            }
        ));
    }

    #[test]
    fn test_tensor_metadata_with_normalization() {
        let norm = NormalizationParams::imagenet();
        let tensor = TensorMetadata::new(
            "input".to_string(),
            ContentType::Image {
                color_space: ColorSpace::RGB,
            },
        )
        .with_normalization(norm);

        assert!(tensor.normalization.is_some());
    }

    #[test]
    fn test_tensor_metadata_validation() {
        let tensor = TensorMetadata::new("input".to_string(), ContentType::FeatureVector);
        assert!(tensor.validate().is_ok());

        let empty_tensor = TensorMetadata::new("".to_string(), ContentType::FeatureVector);
        assert!(empty_tensor.validate().is_err());
    }

    #[test]
    fn test_content_type_image() {
        let content = ContentType::Image {
            color_space: ColorSpace::RGB,
        };

        assert!(matches!(content, ContentType::Image { .. }));
    }

    #[test]
    fn test_content_type_audio() {
        let content = ContentType::Audio {
            sample_rate: 16000,
            channels: 1,
        };

        assert!(matches!(content, ContentType::Audio { .. }));
    }

    #[test]
    fn test_normalization_params() {
        let norm = NormalizationParams::new(vec![0.5, 0.5, 0.5], vec![0.25, 0.25, 0.25]);

        assert_eq!(norm.mean.len(), 3);
        assert_eq!(norm.std.len(), 3);
        assert!(norm.validate().is_ok());
    }

    #[test]
    fn test_normalization_from_single() {
        let norm = NormalizationParams::from_single(0.5, 0.25);

        assert_eq!(norm.mean, vec![0.5]);
        assert_eq!(norm.std, vec![0.25]);
    }

    #[test]
    fn test_normalization_imagenet() {
        let norm = NormalizationParams::imagenet();

        assert_eq!(norm.mean.len(), 3);
        assert_eq!(norm.std.len(), 3);
        assert!(norm.validate().is_ok());
    }

    #[test]
    fn test_normalization_validation() {
        let valid = NormalizationParams::new(vec![0.5], vec![0.25]);
        assert!(valid.validate().is_ok());

        let empty_mean = NormalizationParams::new(vec![], vec![0.25]);
        assert!(empty_mean.validate().is_err());

        let mismatch = NormalizationParams::new(vec![0.5], vec![0.25, 0.25]);
        assert!(mismatch.validate().is_err());

        let zero_std = NormalizationParams::new(vec![0.5], vec![0.0]);
        assert!(zero_std.validate().is_err());
    }

    #[test]
    fn test_associated_file_creation() {
        let file = AssociatedFile::new("labels.txt".to_string(), FileType::Labels);

        assert_eq!(file.name, "labels.txt");
        assert_eq!(file.file_type, FileType::Labels);
    }

    #[test]
    fn test_associated_file_label_file() {
        let labels = vec!["cat".to_string(), "dog".to_string(), "bird".to_string()];

        let file = AssociatedFile::label_file("labels.txt".to_string(), labels);

        assert_eq!(file.name, "labels.txt");
        assert_eq!(file.file_type, FileType::Labels);
        assert!(file.contents.is_some());

        let contents = String::from_utf8(file.contents.unwrap()).unwrap();
        assert!(contents.contains("cat"));
        assert!(contents.contains("dog"));
        assert!(contents.contains("bird"));
    }

    #[test]
    fn test_associated_file_validation() {
        let file = AssociatedFile::new("test.txt".to_string(), FileType::Labels);
        assert!(file.validate().is_ok());

        let empty_file = AssociatedFile::new("".to_string(), FileType::Labels);
        assert!(empty_file.validate().is_err());
    }
}
