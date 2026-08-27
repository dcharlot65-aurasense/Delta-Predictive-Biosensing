//! # HL7 FHIR (Fast Healthcare Interoperability Resources) Support
//!
//! This module provides comprehensive support for HL7 FHIR R4 standard, enabling
//! interoperability with healthcare systems and electronic health records (EHRs).
//!
//! ## Features
//!
//! - **Core Resources**: Patient, Observation, DiagnosticReport, Device, etc.
//! - **Observation Types**: Vital signs, waveforms, multi-component observations
//! - **Bundles**: Document, transaction, batch, and search bundles
//! - **Serialization**: JSON and XML support with FHIR R4 compliance
//! - **Conversion**: DPB types to FHIR resources
//! - **Client**: HTTP FHIR client for server interactions
//!
//! ## Examples
//!
//! ### Creating a Patient Resource
//!
//! ```rust,no_run
//! use dpb_core::io::fhir::{Patient, HumanName, Identifier};
//!
//! let patient = Patient {
//!     id: Some("patient-001".to_string()),
//!     identifier: vec![Identifier {
//!         system: Some("http://example.org/mrn".to_string()),
//!         value: Some("12345".to_string()),
//!         ..Default::default()
//!     }],
//!     name: vec![HumanName {
//!         family: Some("Smith".to_string()),
//!         given: vec!["John".to_string()],
//!         ..Default::default()
//!     }],
//!     gender: Some("male".to_string()),
//!     birth_date: Some("1980-01-01".to_string()),
//!     ..Default::default()
//! };
//! ```
//!
//! ### Creating an Observation
//!
//! ```rust
//! use dpb_core::io::fhir::VitalSignsObservation;
//!
//! // `VitalSignsObservation` carries the LOINC code and units for each vital
//! // sign, so the caller supplies the value and the time.
//! let obs = VitalSignsObservation::heart_rate(
//!     "obs-hr-001".to_string(),
//!     "patient-001".to_string(),
//!     72.0,
//!     "2026-08-01T12:00:00Z".to_string(),
//! );
//! ```

pub mod bundles;
pub mod client;
pub mod conversion;
pub mod observations;
pub mod resources;
pub mod serialization;

// Re-export main types
pub use bundles::{Bundle, BundleEntry, BundleType};
pub use client::{FhirClient, FhirClientConfig};
pub use conversion::{
    AnalysisToReport, MetricsToObservation, PatientDemographics, SignalToObservation,
};
pub use observations::{
    ComponentObservation, ObservationStatus, ReferenceRange, VitalSignsObservation,
    WaveformObservation,
};
pub use resources::{
    Attachment, CodeableConcept, Coding, Condition, Device, DiagnosticReport, Encounter, HumanName,
    Identifier, Observation, Patient, Procedure, Quantity, Reference,
};
pub use serialization::{FhirSerializer, JsonSerializer, XmlSerializer};

use serde::{Deserialize, Serialize};

/// FHIR resource types supported by this implementation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ResourceType {
    /// Patient demographic and administrative information
    Patient,
    /// Measurements and observations
    Observation,
    /// Diagnostic report grouping observations
    DiagnosticReport,
    /// Medical device or sensor
    Device,
    /// Healthcare encounter or session
    Encounter,
    /// Medical procedure or assessment
    Procedure,
    /// Clinical condition or diagnosis
    Condition,
    /// Collection of resources
    Bundle,
}

impl ResourceType {
    /// Returns the FHIR resource type name as a string
    pub fn as_str(&self) -> &'static str {
        match self {
            ResourceType::Patient => "Patient",
            ResourceType::Observation => "Observation",
            ResourceType::DiagnosticReport => "DiagnosticReport",
            ResourceType::Device => "Device",
            ResourceType::Encounter => "Encounter",
            ResourceType::Procedure => "Procedure",
            ResourceType::Condition => "Condition",
            ResourceType::Bundle => "Bundle",
        }
    }

    /// Returns all supported resource types
    pub fn all() -> Vec<ResourceType> {
        vec![
            ResourceType::Patient,
            ResourceType::Observation,
            ResourceType::DiagnosticReport,
            ResourceType::Device,
            ResourceType::Encounter,
            ResourceType::Procedure,
            ResourceType::Condition,
            ResourceType::Bundle,
        ]
    }
}

impl std::fmt::Display for ResourceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for ResourceType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Patient" => Ok(ResourceType::Patient),
            "Observation" => Ok(ResourceType::Observation),
            "DiagnosticReport" => Ok(ResourceType::DiagnosticReport),
            "Device" => Ok(ResourceType::Device),
            "Encounter" => Ok(ResourceType::Encounter),
            "Procedure" => Ok(ResourceType::Procedure),
            "Condition" => Ok(ResourceType::Condition),
            "Bundle" => Ok(ResourceType::Bundle),
            _ => Err(format!("Unknown resource type: {}", s)),
        }
    }
}

/// Generic FHIR resource wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "resourceType")]
pub enum FhirResource {
    /// The subject of care.
    Patient(Patient),
    /// A measurement or finding. Boxed because it is much larger than the
    /// other variants and would otherwise set the size of the whole enum.
    Observation(Box<Observation>),
    /// A grouped set of observations with an interpretation.
    DiagnosticReport(DiagnosticReport),
    /// The instrument a measurement came from.
    Device(Device),
    /// The interaction during which the data was collected.
    Encounter(Encounter),
    /// An action performed on the patient.
    Procedure(Procedure),
    /// A diagnosis or clinical problem.
    Condition(Condition),
    /// A container for a collection of resources.
    Bundle(Bundle),
}

impl FhirResource {
    /// Returns the resource type
    pub fn resource_type(&self) -> ResourceType {
        match self {
            FhirResource::Patient(_) => ResourceType::Patient,
            FhirResource::Observation(_) => ResourceType::Observation,
            FhirResource::DiagnosticReport(_) => ResourceType::DiagnosticReport,
            FhirResource::Device(_) => ResourceType::Device,
            FhirResource::Encounter(_) => ResourceType::Encounter,
            FhirResource::Procedure(_) => ResourceType::Procedure,
            FhirResource::Condition(_) => ResourceType::Condition,
            FhirResource::Bundle(_) => ResourceType::Bundle,
        }
    }

    /// Returns the resource ID if present
    pub fn id(&self) -> Option<&str> {
        match self {
            FhirResource::Patient(r) => r.id.as_deref(),
            FhirResource::Observation(r) => r.id.as_deref(),
            FhirResource::DiagnosticReport(r) => r.id.as_deref(),
            FhirResource::Device(r) => r.id.as_deref(),
            FhirResource::Encounter(r) => r.id.as_deref(),
            FhirResource::Procedure(r) => r.id.as_deref(),
            FhirResource::Condition(r) => r.id.as_deref(),
            FhirResource::Bundle(r) => r.id.as_deref(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_type_str() {
        assert_eq!(ResourceType::Patient.as_str(), "Patient");
        assert_eq!(ResourceType::Observation.as_str(), "Observation");
        assert_eq!(ResourceType::DiagnosticReport.as_str(), "DiagnosticReport");
    }

    #[test]
    fn test_resource_type_from_str() {
        assert_eq!(
            "Patient".parse::<ResourceType>().unwrap(),
            ResourceType::Patient
        );
        assert_eq!(
            "Observation".parse::<ResourceType>().unwrap(),
            ResourceType::Observation
        );
        assert!("Invalid".parse::<ResourceType>().is_err());
    }

    #[test]
    fn test_resource_type_all() {
        let types = ResourceType::all();
        assert_eq!(types.len(), 8);
        assert!(types.contains(&ResourceType::Patient));
        assert!(types.contains(&ResourceType::Observation));
    }
}
