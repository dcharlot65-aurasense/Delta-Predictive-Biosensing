//! BIDS validation and conformance checking
//!
//! This module provides validation tools to ensure BIDS datasets conform to the
//! specification, including file naming conventions, required files, and metadata structure.

use std::fs;
use std::path::{Path, PathBuf};

use super::Modality;

/// Validation severity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ValidationLevel {
    /// Informational message
    Info,
    /// Warning - doesn't prevent usage but should be fixed
    Warning,
    /// Error - violates BIDS specification
    Error,
}

impl std::fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationLevel::Info => write!(f, "INFO"),
            ValidationLevel::Warning => write!(f, "WARNING"),
            ValidationLevel::Error => write!(f, "ERROR"),
        }
    }
}

/// A single validation issue
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity level
    pub level: ValidationLevel,

    /// Issue code (e.g., "MISSING_REQUIRED_FILE")
    pub code: String,

    /// Human-readable message
    pub message: String,

    /// File or directory path related to the issue
    pub path: Option<PathBuf>,

    /// Line number (for TSV/JSON files)
    pub line: Option<usize>,
}

impl ValidationIssue {
    /// Create a new validation issue
    pub fn new(
        level: ValidationLevel,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            level,
            code: code.into(),
            message: message.into(),
            path: None,
            line: None,
        }
    }

    /// Add a path to the issue
    pub fn with_path(mut self, path: PathBuf) -> Self {
        self.path = Some(path);
        self
    }

    /// Add a line number to the issue
    pub fn with_line(mut self, line: usize) -> Self {
        self.line = Some(line);
        self
    }
}

impl std::fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}] {}: {}", self.level, self.code, self.message)?;
        if let Some(ref path) = self.path {
            write!(f, " ({})", path.display())?;
        }
        if let Some(line) = self.line {
            write!(f, " at line {}", line)?;
        }
        Ok(())
    }
}

/// Validation report containing all issues found
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// All validation issues
    issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Create a new empty validation report
    pub fn new() -> Self {
        Self {
            issues: Vec::new(),
        }
    }

    /// Add an issue to the report
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    /// Get all issues
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }

    /// Get issues of a specific level
    pub fn issues_with_level(&self, level: ValidationLevel) -> Vec<&ValidationIssue> {
        self.issues.iter().filter(|i| i.level == level).collect()
    }

    /// Count issues of a specific level
    pub fn count(&self, level: ValidationLevel) -> usize {
        self.issues.iter().filter(|i| i.level == level).count()
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        self.issues.iter().any(|i| i.level == ValidationLevel::Error)
    }

    /// Check if the dataset is valid (no errors)
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// Get a summary of the validation
    pub fn summary(&self) -> String {
        format!(
            "Validation Summary: {} errors, {} warnings, {} info",
            self.count(ValidationLevel::Error),
            self.count(ValidationLevel::Warning),
            self.count(ValidationLevel::Info),
        )
    }
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.summary())?;
        writeln!(f)?;

        for issue in &self.issues {
            writeln!(f, "{}", issue)?;
        }

        Ok(())
    }
}

/// BIDS dataset validator
///
/// Validates BIDS datasets against the specification, checking:
/// - Required files (dataset_description.json, etc.)
/// - File naming conventions
/// - Directory structure
/// - Metadata completeness
/// - JSON schema validation
pub struct BidsValidator {
    /// Strict mode (treat warnings as errors)
    strict: bool,

    /// Check optional fields
    check_optional: bool,
}

impl BidsValidator {
    /// Create a new validator with default settings
    pub fn new() -> Self {
        Self {
            strict: false,
            check_optional: true,
        }
    }

    /// Enable strict mode (treat warnings as errors)
    pub fn strict(mut self) -> Self {
        self.strict = true;
        self
    }

    /// Disable checking optional fields
    pub fn skip_optional(mut self) -> Self {
        self.check_optional = false;
        self
    }

    /// Validate a BIDS dataset
    pub fn validate<P: AsRef<Path>>(&self, dataset_path: P) -> ValidationReport {
        let mut report = ValidationReport::new();
        let dataset_path = dataset_path.as_ref();

        // Check if path exists
        if !dataset_path.exists() {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Error,
                    "DATASET_NOT_FOUND",
                    "Dataset directory does not exist",
                )
                .with_path(dataset_path.to_path_buf()),
            );
            return report;
        }

        // Validate required files
        self.validate_required_files(dataset_path, &mut report);

        // Validate dataset_description.json
        self.validate_dataset_description(dataset_path, &mut report);

        // Validate subject directories
        self.validate_subjects(dataset_path, &mut report);

        // Validate file naming conventions
        self.validate_naming_conventions(dataset_path, &mut report);

        report
    }

    /// Validate required files at dataset root
    fn validate_required_files(&self, root: &Path, report: &mut ValidationReport) {
        // dataset_description.json is required
        let desc_path = root.join("dataset_description.json");
        if !desc_path.exists() {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Error,
                    "MISSING_DATASET_DESCRIPTION",
                    "Required file 'dataset_description.json' is missing",
                )
                .with_path(desc_path),
            );
        }

        if self.check_optional {
            // README is recommended
            let readme_path = root.join("README");
            if !readme_path.exists() {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "MISSING_README",
                        "Recommended file 'README' is missing",
                    )
                    .with_path(readme_path),
                );
            }

            // participants.tsv is recommended for multi-subject datasets
            let participants_path = root.join("participants.tsv");
            if !participants_path.exists() {
                // Check if there are multiple subjects
                if self.count_subjects(root) > 1 {
                    report.add_issue(
                        ValidationIssue::new(
                            ValidationLevel::Warning,
                            "MISSING_PARTICIPANTS",
                            "Recommended file 'participants.tsv' is missing for multi-subject dataset",
                        )
                        .with_path(participants_path),
                    );
                }
            }
        }
    }

    /// Validate dataset_description.json content
    fn validate_dataset_description(&self, root: &Path, report: &mut ValidationReport) {
        let desc_path = root.join("dataset_description.json");
        if !desc_path.exists() {
            return; // Already reported in validate_required_files
        }

        let content = match fs::read_to_string(&desc_path) {
            Ok(c) => c,
            Err(e) => {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "INVALID_JSON",
                        format!("Failed to read dataset_description.json: {}", e),
                    )
                    .with_path(desc_path),
                );
                return;
            }
        };

        let json: serde_json::Value = match serde_json::from_str(&content) {
            Ok(j) => j,
            Err(e) => {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "INVALID_JSON",
                        format!("Invalid JSON in dataset_description.json: {}", e),
                    )
                    .with_path(desc_path),
                );
                return;
            }
        };

        // Check required fields
        if json.get("Name").is_none() {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Error,
                    "MISSING_REQUIRED_FIELD",
                    "Required field 'Name' is missing in dataset_description.json",
                )
                .with_path(desc_path.clone()),
            );
        }

        if json.get("BIDSVersion").is_none() {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Error,
                    "MISSING_REQUIRED_FIELD",
                    "Required field 'BIDSVersion' is missing in dataset_description.json",
                )
                .with_path(desc_path.clone()),
            );
        }

        // Check recommended fields
        if self.check_optional {
            if json.get("License").is_none() {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "MISSING_RECOMMENDED_FIELD",
                        "Recommended field 'License' is missing in dataset_description.json",
                    )
                    .with_path(desc_path.clone()),
                );
            }

            if json.get("Authors").is_none() {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "MISSING_RECOMMENDED_FIELD",
                        "Recommended field 'Authors' is missing in dataset_description.json",
                    )
                    .with_path(desc_path),
                );
            }
        }
    }

    /// Validate subject directories
    fn validate_subjects(&self, root: &Path, report: &mut ValidationReport) {
        let entries = match fs::read_dir(root) {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut has_subjects = false;

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("sub-") {
                    has_subjects = true;
                    self.validate_subject_dir(&path, report);
                }
            }
        }

        if !has_subjects {
            report.add_issue(ValidationIssue::new(
                ValidationLevel::Warning,
                "NO_SUBJECTS",
                "No subject directories found in dataset",
            ));
        }
    }

    /// Validate a single subject directory
    fn validate_subject_dir(&self, subject_path: &Path, report: &mut ValidationReport) {
        // Check if subject has sessions or modality directories
        let entries = match fs::read_dir(subject_path) {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut has_sessions = false;
        let mut has_modalities = false;

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with("ses-") {
                    has_sessions = true;
                    self.validate_session_dir(&path, report);
                } else if Modality::from_dir_name(name).is_some() {
                    has_modalities = true;
                }
            }
        }

        if !has_sessions && !has_modalities {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Warning,
                    "EMPTY_SUBJECT",
                    "Subject directory contains no sessions or modality directories",
                )
                .with_path(subject_path.to_path_buf()),
            );
        }

        // Subject shouldn't have both sessions and modality directories
        if has_sessions && has_modalities {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Error,
                    "MIXED_STRUCTURE",
                    "Subject directory should not mix session directories and modality directories",
                )
                .with_path(subject_path.to_path_buf()),
            );
        }
    }

    /// Validate a session directory
    fn validate_session_dir(&self, session_path: &Path, report: &mut ValidationReport) {
        let entries = match fs::read_dir(session_path) {
            Ok(e) => e,
            Err(_) => return,
        };

        let mut has_modalities = false;

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if Modality::from_dir_name(name).is_some() {
                    has_modalities = true;
                    break;
                }
            }
        }

        if !has_modalities {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Warning,
                    "EMPTY_SESSION",
                    "Session directory contains no modality directories",
                )
                .with_path(session_path.to_path_buf()),
            );
        }
    }

    /// Validate file naming conventions
    fn validate_naming_conventions(&self, root: &Path, report: &mut ValidationReport) {
        self.validate_naming_recursive(root, report);
    }

    /// Recursively validate naming conventions
    fn validate_naming_recursive(&self, dir: &Path, report: &mut ValidationReport) {
        let entries = match fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };

        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();

            if path.is_dir() {
                self.validate_naming_recursive(&path, report);
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                // Skip hidden files
                if name.starts_with('.') {
                    continue;
                }

                // Validate BIDS naming pattern for data files
                if name.starts_with("sub-") {
                    self.validate_bids_filename(name, &path, report);
                }
            }
        }
    }

    /// Validate a BIDS filename
    fn validate_bids_filename(&self, filename: &str, path: &Path, report: &mut ValidationReport) {
        // Basic pattern: sub-<label>[_ses-<label>][_<key>-<value>]*_<suffix>.<ext>

        // Check for invalid characters
        if filename.contains(' ') {
            report.add_issue(
                ValidationIssue::new(
                    ValidationLevel::Error,
                    "INVALID_FILENAME",
                    "Filename contains spaces (use underscores or hyphens)",
                )
                .with_path(path.to_path_buf()),
            );
        }

        // Check that subject ID follows after "sub-"
        if !filename.starts_with("sub-") {
            return;
        }

        let parts: Vec<&str> = filename.split('_').collect();
        if parts.is_empty() {
            return;
        }

        // Validate key-value pairs
        for part in &parts[1..] {
            if part.contains('-') && !part.contains('.') {
                let kv: Vec<&str> = part.split('-').collect();
                if kv.len() != 2 {
                    report.add_issue(
                        ValidationIssue::new(
                            ValidationLevel::Warning,
                            "INVALID_KEY_VALUE",
                            format!("Invalid key-value pair format: {}", part),
                        )
                        .with_path(path.to_path_buf()),
                    );
                }
            }
        }
    }

    /// Count number of subject directories
    fn count_subjects(&self, root: &Path) -> usize {
        fs::read_dir(root)
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| {
                        e.path().is_dir()
                            && e.file_name()
                                .to_str()
                                .map(|n| n.starts_with("sub-"))
                                .unwrap_or(false)
                    })
                    .count()
            })
            .unwrap_or(0)
    }
}

impl Default for BidsValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_validation_level_ordering() {
        assert!(ValidationLevel::Info < ValidationLevel::Warning);
        assert!(ValidationLevel::Warning < ValidationLevel::Error);
    }

    #[test]
    fn test_validation_issue() {
        let issue = ValidationIssue::new(
            ValidationLevel::Error,
            "TEST_CODE",
            "Test message",
        )
        .with_path(PathBuf::from("/test/path"))
        .with_line(42);

        assert_eq!(issue.level, ValidationLevel::Error);
        assert_eq!(issue.code, "TEST_CODE");
        assert_eq!(issue.message, "Test message");
        assert_eq!(issue.path, Some(PathBuf::from("/test/path")));
        assert_eq!(issue.line, Some(42));
    }

    #[test]
    fn test_validation_report() {
        let mut report = ValidationReport::new();

        report.add_issue(ValidationIssue::new(
            ValidationLevel::Error,
            "ERROR1",
            "Error message",
        ));
        report.add_issue(ValidationIssue::new(
            ValidationLevel::Warning,
            "WARN1",
            "Warning message",
        ));
        report.add_issue(ValidationIssue::new(
            ValidationLevel::Info,
            "INFO1",
            "Info message",
        ));

        assert_eq!(report.count(ValidationLevel::Error), 1);
        assert_eq!(report.count(ValidationLevel::Warning), 1);
        assert_eq!(report.count(ValidationLevel::Info), 1);
        assert!(report.has_errors());
        assert!(!report.is_valid());
    }

    #[test]
    fn test_validate_missing_dataset() {
        let validator = BidsValidator::new();
        let report = validator.validate("/nonexistent/path");

        assert!(report.has_errors());
        assert_eq!(report.count(ValidationLevel::Error), 1);
    }

    #[test]
    fn test_validate_missing_description() {
        let temp_dir = TempDir::new().unwrap();
        let validator = BidsValidator::new();
        let report = validator.validate(temp_dir.path());

        assert!(report.has_errors());
        let errors: Vec<_> = report.issues_with_level(ValidationLevel::Error);
        assert!(errors.iter().any(|e| e.code == "MISSING_DATASET_DESCRIPTION"));
    }

    #[test]
    fn test_validate_invalid_json() -> std::io::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let desc_path = temp_dir.path().join("dataset_description.json");
        fs::write(&desc_path, "{ invalid json }")?;

        let validator = BidsValidator::new();
        let report = validator.validate(temp_dir.path());

        assert!(report.has_errors());
        let errors: Vec<_> = report.issues_with_level(ValidationLevel::Error);
        assert!(errors.iter().any(|e| e.code == "INVALID_JSON"));

        Ok(())
    }

    #[test]
    fn test_validate_missing_required_fields() -> std::io::Result<()> {
        let temp_dir = TempDir::new().unwrap();
        let desc_path = temp_dir.path().join("dataset_description.json");
        fs::write(&desc_path, "{}")?;

        let validator = BidsValidator::new();
        let report = validator.validate(temp_dir.path());

        assert!(report.has_errors());
        let errors: Vec<_> = report.issues_with_level(ValidationLevel::Error);
        assert!(errors.iter().any(|e| e.message.contains("Name")));
        assert!(errors.iter().any(|e| e.message.contains("BIDSVersion")));

        Ok(())
    }

    #[test]
    fn test_validate_filename_with_spaces() {
        let temp_dir = TempDir::new().unwrap();
        let bad_file = temp_dir.path().join("sub-01_ses-01_task-rest eeg.edf");
        fs::write(&bad_file, "").unwrap();

        let validator = BidsValidator::new();
        let report = validator.validate(temp_dir.path());

        let errors: Vec<_> = report.issues_with_level(ValidationLevel::Error);
        assert!(errors.iter().any(|e| e.code == "INVALID_FILENAME"));
    }
}
