//! Core FHIR Resource Types
//!
//! Implements the primary FHIR R4 resources used in biosensor and medical device data exchange.

use serde::{Deserialize, Serialize};

/// Reference to another FHIR resource
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Reference {
    /// Literal reference (e.g., "Patient/123")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reference: Option<String>,
    /// Type of resource being referenced
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<String>,
    /// Text alternative for the resource
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
    /// Logical reference (UUID)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub identifier: Option<Identifier>,
}

impl Reference {
    /// Creates a new reference to a resource
    pub fn new(resource_type: &str, id: &str) -> Self {
        Self {
            reference: Some(format!("{}/{}", resource_type, id)),
            r#type: Some(resource_type.to_string()),
            display: None,
            identifier: None,
        }
    }

    /// Creates a reference with display text
    pub fn with_display(mut self, display: String) -> Self {
        self.display = Some(display);
        self
    }
}

/// Identifier for a resource (e.g., MRN, SSN)
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Identifier {
    /// Purpose of this identifier (e.g., "official", "temp")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#use: Option<String>,
    /// System that defines the identifier namespace
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The identifier value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Organization that issued the identifier
    #[serde(skip_serializing_if = "Option::is_none")]
    pub assigner: Option<String>,
}

impl Identifier {
    /// Creates a new identifier
    pub fn new(system: String, value: String) -> Self {
        Self {
            r#use: None,
            system: Some(system),
            value: Some(value),
            assigner: None,
        }
    }
}

/// Human name representation
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct HumanName {
    /// How the name should be used (e.g., "official", "nickname")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#use: Option<String>,
    /// Family name (surname)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub family: Option<String>,
    /// Given names (first name, middle names)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub given: Vec<String>,
    /// Parts that come before the name (e.g., "Dr.")
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prefix: Vec<String>,
    /// Parts that come after the name (e.g., "Jr.", "MD")
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub suffix: Vec<String>,
}

/// Contact point (phone, email, etc.)
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ContactPoint {
    /// Type of contact (phone, email, fax, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// The actual contact point value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
    /// Purpose (home, work, temp, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#use: Option<String>,
}

/// Address representation
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Address {
    /// Purpose of this address (home, work, etc.)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#use: Option<String>,
    /// Street address lines
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub line: Vec<String>,
    /// City
    #[serde(skip_serializing_if = "Option::is_none")]
    pub city: Option<String>,
    /// State or province
    #[serde(skip_serializing_if = "Option::is_none")]
    pub state: Option<String>,
    /// Postal code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub postal_code: Option<String>,
    /// Country
    #[serde(skip_serializing_if = "Option::is_none")]
    pub country: Option<String>,
}

/// Patient resource - demographics and administrative information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Patient {
    /// Resource type (always "Patient")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID of the resource
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Patient identifiers (MRN, SSN, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub identifier: Vec<Identifier>,
    /// Whether the patient record is active
    #[serde(skip_serializing_if = "Option::is_none")]
    pub active: Option<bool>,
    /// Patient names
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub name: Vec<HumanName>,
    /// Contact details (phone, email)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub telecom: Vec<ContactPoint>,
    /// Gender: male | female | other | unknown
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
    /// Date of birth (YYYY-MM-DD)
    #[serde(rename = "birthDate", skip_serializing_if = "Option::is_none")]
    pub birth_date: Option<String>,
    /// Addresses
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub address: Vec<Address>,
}


impl Default for Patient {
    fn default() -> Self {
        Self {
            resource_type: "Patient".to_string(),
            id: None,
            identifier: Vec::new(),
            active: Some(true),
            name: Vec::new(),
            telecom: Vec::new(),
            gender: None,
            birth_date: None,
            address: Vec::new(),
        }
    }
}

impl Patient {
    /// Creates a new patient with minimal information
    pub fn new(id: String) -> Self {
        Self {
            id: Some(id),
            ..Default::default()
        }
    }
}

/// Device resource - medical device or sensor metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    /// Resource type (always "Device")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Device identifiers (serial number, etc.)
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub identifier: Vec<Identifier>,
    /// Status: active | inactive | entered-in-error
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Manufacturer name
    #[serde(skip_serializing_if = "Option::is_none")]
    pub manufacturer: Option<String>,
    /// Device model name
    #[serde(rename = "modelName", skip_serializing_if = "Option::is_none")]
    pub model_name: Option<String>,
    /// Device type (e.g., ECG monitor)
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub device_type: Option<CodeableConcept>,
    /// Version information
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub version: Vec<DeviceVersion>,
    /// Patient using this device
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patient: Option<Reference>,
}

impl Default for Device {
    fn default() -> Self {
        Self {
            resource_type: "Device".to_string(),
            id: None,
            identifier: Vec::new(),
            status: Some("active".to_string()),
            manufacturer: None,
            model_name: None,
            device_type: None,
            version: Vec::new(),
            patient: None,
        }
    }
}


/// Device version information
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct DeviceVersion {
    /// Type of version (firmware, software, hardware)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub r#type: Option<CodeableConcept>,
    /// Version value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<String>,
}

/// Codeable concept - code with optional text
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CodeableConcept {
    /// Codes from terminology systems
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub coding: Vec<Coding>,
    /// Plain text representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl CodeableConcept {
    /// Creates a LOINC code
    pub fn loinc(code: &str, display: &str) -> Self {
        Self {
            coding: vec![Coding {
                system: Some("http://loinc.org".to_string()),
                code: Some(code.to_string()),
                display: Some(display.to_string()),
            }],
            text: Some(display.to_string()),
        }
    }

    /// Creates a SNOMED CT code
    pub fn snomed(code: &str, display: &str) -> Self {
        Self {
            coding: vec![Coding {
                system: Some("http://snomed.info/sct".to_string()),
                code: Some(code.to_string()),
                display: Some(display.to_string()),
            }],
            text: Some(display.to_string()),
        }
    }
}

/// Coding - a code from a terminology system
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Coding {
    /// Code system (e.g., LOINC, SNOMED CT)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Code value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
    /// Display text for the code
    #[serde(skip_serializing_if = "Option::is_none")]
    pub display: Option<String>,
}

/// Observation resource - measurements and findings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// Resource type (always "Observation")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Status: registered | preliminary | final | amended
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Classification of observation
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category: Vec<CodeableConcept>,
    /// Type of observation (LOINC code)
    pub code: CodeableConcept,
    /// Patient this observation is about
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<Reference>,
    /// Healthcare encounter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encounter: Option<Reference>,
    /// Clinically relevant time/time-period
    #[serde(rename = "effectiveDateTime", skip_serializing_if = "Option::is_none")]
    pub effective_date_time: Option<String>,
    /// Date/time this was made available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued: Option<String>,
    /// Actual result
    #[serde(rename = "valueQuantity", skip_serializing_if = "Option::is_none")]
    pub value_quantity: Option<Quantity>,
    /// Component results
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub component: Vec<ObservationComponent>,
    /// Device used
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device: Option<Reference>,
}

impl Default for Observation {
    fn default() -> Self {
        Self {
            resource_type: "Observation".to_string(),
            id: None,
            status: Some("final".to_string()),
            category: Vec::new(),
            code: CodeableConcept::default(),
            subject: None,
            encounter: None,
            effective_date_time: None,
            issued: None,
            value_quantity: None,
            component: Vec::new(),
            device: None,
        }
    }
}


/// Observation component (for multi-value observations)
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ObservationComponent {
    /// Type of component
    pub code: CodeableConcept,
    /// Actual component result
    #[serde(rename = "valueQuantity", skip_serializing_if = "Option::is_none")]
    pub value_quantity: Option<Quantity>,
}

/// Quantity with unit
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Quantity {
    /// Numerical value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f64>,
    /// Unit representation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub unit: Option<String>,
    /// Coded form of unit (UCUM)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    /// Coded value
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

impl Quantity {
    /// Creates a new quantity
    pub fn new(value: f64, unit: &str, code: &str) -> Self {
        Self {
            value: Some(value),
            unit: Some(unit.to_string()),
            system: Some("http://unitsofmeasure.org".to_string()),
            code: Some(code.to_string()),
        }
    }
}

/// DiagnosticReport - grouping of observations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticReport {
    /// Resource type (always "DiagnosticReport")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Status: registered | partial | preliminary | final
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Service category
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category: Vec<CodeableConcept>,
    /// Type of diagnostic report
    pub code: CodeableConcept,
    /// Subject of the report
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<Reference>,
    /// Healthcare encounter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encounter: Option<Reference>,
    /// Clinically relevant time
    #[serde(rename = "effectiveDateTime", skip_serializing_if = "Option::is_none")]
    pub effective_date_time: Option<String>,
    /// Date/time report was made available
    #[serde(skip_serializing_if = "Option::is_none")]
    pub issued: Option<String>,
    /// Observations included in report
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub result: Vec<Reference>,
    /// Clinical conclusion
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conclusion: Option<String>,
}

impl Default for DiagnosticReport {
    fn default() -> Self {
        Self {
            resource_type: "DiagnosticReport".to_string(),
            id: None,
            status: Some("final".to_string()),
            category: Vec::new(),
            code: CodeableConcept::default(),
            subject: None,
            encounter: None,
            effective_date_time: None,
            issued: None,
            result: Vec::new(),
            conclusion: None,
        }
    }
}


/// Encounter - healthcare session or visit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Encounter {
    /// Resource type (always "Encounter")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Status: planned | arrived | in-progress | finished
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Classification of encounter
    #[serde(rename = "class", skip_serializing_if = "Option::is_none")]
    pub class_code: Option<Coding>,
    /// Type of encounter
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub r#type: Vec<CodeableConcept>,
    /// The patient present at the encounter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<Reference>,
    /// Time period of encounter
    #[serde(skip_serializing_if = "Option::is_none")]
    pub period: Option<Period>,
}

impl Default for Encounter {
    fn default() -> Self {
        Self {
            resource_type: "Encounter".to_string(),
            id: None,
            status: Some("finished".to_string()),
            class_code: None,
            r#type: Vec::new(),
            subject: None,
            period: None,
        }
    }
}


/// Period of time
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Period {
    /// Start time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<String>,
    /// End time
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<String>,
}

/// Procedure - assessment protocol or procedure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Procedure {
    /// Resource type (always "Procedure")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Status: preparation | in-progress | completed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    /// Type of procedure
    pub code: CodeableConcept,
    /// Who the procedure was performed on
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<Reference>,
    /// Encounter during which procedure occurred
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encounter: Option<Reference>,
    /// When procedure was performed
    #[serde(rename = "performedDateTime", skip_serializing_if = "Option::is_none")]
    pub performed_date_time: Option<String>,
}

impl Default for Procedure {
    fn default() -> Self {
        Self {
            resource_type: "Procedure".to_string(),
            id: None,
            status: Some("completed".to_string()),
            code: CodeableConcept::default(),
            subject: None,
            encounter: None,
            performed_date_time: None,
        }
    }
}


/// Condition - clinical diagnosis or problem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Condition {
    /// Resource type (always "Condition")
    #[serde(rename = "resourceType")]
    pub resource_type: String,
    /// Logical ID
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Clinical status: active | recurrence | relapse | resolved
    #[serde(rename = "clinicalStatus", skip_serializing_if = "Option::is_none")]
    pub clinical_status: Option<CodeableConcept>,
    /// Verification status: unconfirmed | provisional | confirmed
    #[serde(rename = "verificationStatus", skip_serializing_if = "Option::is_none")]
    pub verification_status: Option<CodeableConcept>,
    /// Category of condition
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub category: Vec<CodeableConcept>,
    /// Identification of the condition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub code: Option<CodeableConcept>,
    /// Who has the condition
    #[serde(skip_serializing_if = "Option::is_none")]
    pub subject: Option<Reference>,
    /// Encounter created as part of
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encounter: Option<Reference>,
    /// When condition was first detected
    #[serde(rename = "onsetDateTime", skip_serializing_if = "Option::is_none")]
    pub onset_date_time: Option<String>,
}

impl Default for Condition {
    fn default() -> Self {
        Self {
            resource_type: "Condition".to_string(),
            id: None,
            clinical_status: None,
            verification_status: None,
            category: Vec::new(),
            code: None,
            subject: None,
            encounter: None,
            onset_date_time: None,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reference_creation() {
        let ref_obj = Reference::new("Patient", "123");
        assert_eq!(ref_obj.reference, Some("Patient/123".to_string()));
        assert_eq!(ref_obj.r#type, Some("Patient".to_string()));
    }

    #[test]
    fn test_identifier_creation() {
        let id = Identifier::new("http://example.org/mrn".to_string(), "12345".to_string());
        assert_eq!(id.system, Some("http://example.org/mrn".to_string()));
        assert_eq!(id.value, Some("12345".to_string()));
    }

    #[test]
    fn test_codeable_concept_loinc() {
        let concept = CodeableConcept::loinc("8867-4", "Heart rate");
        assert_eq!(concept.coding.len(), 1);
        assert_eq!(concept.coding[0].system, Some("http://loinc.org".to_string()));
        assert_eq!(concept.coding[0].code, Some("8867-4".to_string()));
    }

    #[test]
    fn test_quantity_creation() {
        let qty = Quantity::new(72.0, "beats/min", "/min");
        assert_eq!(qty.value, Some(72.0));
        assert_eq!(qty.unit, Some("beats/min".to_string()));
    }

    #[test]
    fn test_patient_default() {
        let patient = Patient::default();
        assert_eq!(patient.resource_type, "Patient");
        assert_eq!(patient.active, Some(true));
    }

    #[test]
    fn test_observation_default() {
        let obs = Observation::default();
        assert_eq!(obs.resource_type, "Observation");
        assert_eq!(obs.status, Some("final".to_string()));
    }
}
