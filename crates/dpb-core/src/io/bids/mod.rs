//! # BIDS (Brain Imaging Data Structure) Format Support
//!
//! This module provides comprehensive support for the Brain Imaging Data Structure (BIDS)
//! specification, enabling standardized organization and sharing of neuroimaging and
//! electrophysiology data.
//!
//! ## BIDS Overview
//!
//! BIDS is a standard for organizing and describing neuroimaging and behavioral data.
//! It specifies:
//! - Directory structure and file naming conventions
//! - Metadata formats (JSON sidecars, TSV files)
//! - Required and optional files
//! - Data organization by subject, session, and modality
//!
//! ## Supported Modalities
//!
//! - **EEG** - Electroencephalography
//! - **MEG** - Magnetoencephalography
//! - **iEEG** - Intracranial electroencephalography
//! - **NIRS** - Near-infrared spectroscopy
//! - **Physio** - Physiological recordings (ECG, EMG, etc.)
//!
//! ## Module Organization
//!
//! - [`dataset`] - BIDS dataset structure and metadata
//! - [`subject`] - Subject-level organization
//! - [`session`] - Session-level organization
//! - [`eeg`] - EEG-specific BIDS structures
//! - [`derivatives`] - BIDS derivatives for processed data
//! - [`validation`] - BIDS specification validation
//!
//! ## Examples
//!
//! ### Reading a BIDS dataset
//!
//! ```rust,no_run
//! use dpb_core::io::bids::BidsDataset;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let dataset = BidsDataset::open(Path::new("/data/bids_dataset"))?;
//! println!("Dataset: {}", dataset.name());
//! // `subjects` walks the filesystem, so it reports failure rather than
//! // returning a bare Vec.
//! let subjects = dataset.subjects()?;
//! println!("Subjects: {}", subjects.len());
//!
//! for subject in subjects {
//!     println!("Subject: {}", subject.id());
//!     for session in subject.sessions()? {
//!         println!("  Session: {}", session.id());
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Reading EEG data
//!
//! ```rust,no_run
//! use dpb_core::io::bids::{BidsDataset, Modality};
//! use dpb_core::io::EdfReader;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let dataset = BidsDataset::open(Path::new("/data/bids_dataset"))?;
//! let subject = dataset.subject("01")?;
//! let session = subject.session("01")?;
//!
//! // `files_for_modality` yields PATHS; open them with the reader for the
//! // format at hand.
//! let eeg_files = session.files_for_modality(Modality::Eeg)?;
//! for eeg_path in eeg_files {
//!     if eeg_path.extension().is_some_and(|e| e == "edf") {
//!         let mut reader = EdfReader::open(&eeg_path)?;
//!         let n_channels = reader.signals().len();
//!         let channel_0 = reader.read_signal(0)?;
//!         println!("EEG data: {} channels, {} samples",
//!                  n_channels, channel_0.len());
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Creating a BIDS dataset
//!
//! ```rust,no_run
//! use dpb_core::io::bids::{BidsDataset, DatasetDescription};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let description = DatasetDescription::new(
//!     "My EEG Study",
//!     "1.8.0"
//! )
//! .with_authors(vec!["Jane Doe", "John Smith"])
//! .with_license("CC0");
//!
//! let dataset = BidsDataset::create(
//!     Path::new("/data/new_bids_dataset"),
//!     description
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ### Writing derivatives
//!
//! ```rust,no_run
//! use dpb_core::io::bids::{BidsDataset, DerivativesDataset};
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let dataset = BidsDataset::open(Path::new("/data/bids_dataset"))?;
//! let derivatives = DerivativesDataset::create(
//!     dataset.derivatives_path("preprocessed"),
//!     "Preprocessed data",
//!     "1.0.0",
//!     "DPB Pipeline"
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## BIDS Specification
//!
//! This implementation follows BIDS v1.8+ specification:
//! - [BIDS Specification](https://bids-specification.readthedocs.io/)
//! - [BIDS Starter Kit](https://bids-standard.github.io/bids-starter-kit/)
//!
//! ## References
//!
//! - Gorgolewski, K. J., et al. (2016). The brain imaging data structure,
//!   a format for organizing and describing outputs of neuroimaging experiments.
//!   Scientific Data, 3, 160044.
//! - Pernet, C. R., et al. (2019). EEG-BIDS, an extension to the brain imaging
//!   data structure for electroencephalography. Scientific Data, 6(1), 103.

pub mod dataset;
pub mod derivatives;
pub mod eeg;
pub mod session;
pub mod subject;
pub mod validation;

// Re-export main types
pub use dataset::{BidsDataset, DatasetDescription, Participant};
pub use derivatives::{DerivativesDataset, PipelineDescription};
pub use eeg::{
    ChannelInfo, CoordinateSystem, EegBids, EegMetadata, ElectrodeInfo, EventInfo,
    TaskMetadata,
};
pub use session::{BidsSession, SessionMetadata};
pub use subject::{BidsSubject, SubjectMetadata};
pub use validation::{BidsValidator, ValidationLevel, ValidationReport};

/// BIDS modalities supported by this implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Electroencephalography
    Eeg,
    /// Magnetoencephalography
    Meg,
    /// Intracranial electroencephalography
    IEeg,
    /// Near-infrared spectroscopy
    Nirs,
    /// Physiological recordings (ECG, EMG, respiration, etc.)
    Physio,
    /// Behavioral data
    Beh,
}

impl Modality {
    /// Get the directory name for this modality
    pub fn dir_name(&self) -> &'static str {
        match self {
            Modality::Eeg => "eeg",
            Modality::Meg => "meg",
            Modality::IEeg => "ieeg",
            Modality::Nirs => "nirs",
            Modality::Physio => "physio",
            Modality::Beh => "beh",
        }
    }

    /// Get the suffix for data files of this modality
    pub fn suffix(&self) -> &'static str {
        match self {
            Modality::Eeg => "eeg",
            Modality::Meg => "meg",
            Modality::IEeg => "ieeg",
            Modality::Nirs => "nirs",
            Modality::Physio => "physio",
            Modality::Beh => "beh",
        }
    }

    /// Parse modality from directory name
    pub fn from_dir_name(name: &str) -> Option<Self> {
        match name {
            "eeg" => Some(Modality::Eeg),
            "meg" => Some(Modality::Meg),
            "ieeg" => Some(Modality::IEeg),
            "nirs" => Some(Modality::Nirs),
            "physio" => Some(Modality::Physio),
            "beh" => Some(Modality::Beh),
            _ => None,
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.dir_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modality_dir_name() {
        assert_eq!(Modality::Eeg.dir_name(), "eeg");
        assert_eq!(Modality::Meg.dir_name(), "meg");
        assert_eq!(Modality::IEeg.dir_name(), "ieeg");
    }

    #[test]
    fn test_modality_suffix() {
        assert_eq!(Modality::Eeg.suffix(), "eeg");
        assert_eq!(Modality::Physio.suffix(), "physio");
    }

    #[test]
    fn test_modality_from_dir_name() {
        assert_eq!(Modality::from_dir_name("eeg"), Some(Modality::Eeg));
        assert_eq!(Modality::from_dir_name("meg"), Some(Modality::Meg));
        assert_eq!(Modality::from_dir_name("invalid"), None);
    }

    #[test]
    fn test_modality_display() {
        assert_eq!(format!("{}", Modality::Eeg), "eeg");
        assert_eq!(format!("{}", Modality::Nirs), "nirs");
    }
}
