//! FHIR Observation Specializations
//!
//! Provides specialized observation types for biosignals and vital signs with
//! standardized LOINC and SNOMED CT codes.

use super::resources::{
    Attachment, CodeableConcept, Coding, Observation, ObservationComponent, Quantity, Reference,
};
use serde::{Deserialize, Serialize};

/// Re-export commonly used types
pub use super::resources::CodeableConcept as Code;

/// Observation status codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ObservationStatus {
    /// The observation is being prepared
    Registered,
    /// Early data, not verified
    Preliminary,
    /// Final verified result
    Final,
    /// Subsequent to final, additional information added
    Amended,
    /// Subsequent to final, errors corrected
    Corrected,
    /// The observation was cancelled
    Cancelled,
    /// The observation was entered in error
    EnteredInError,
    /// Status unknown
    Unknown,
}

impl ObservationStatus {
    /// Convert to FHIR status string
    pub fn as_str(&self) -> &'static str {
        match self {
            ObservationStatus::Registered => "registered",
            ObservationStatus::Preliminary => "preliminary",
            ObservationStatus::Final => "final",
            ObservationStatus::Amended => "amended",
            ObservationStatus::Corrected => "corrected",
            ObservationStatus::Cancelled => "cancelled",
            ObservationStatus::EnteredInError => "entered-in-error",
            ObservationStatus::Unknown => "unknown",
        }
    }
}

/// Reference range for observations
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct ReferenceRange {
    /// Low bound (inclusive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub low: Option<Quantity>,
    /// High bound (inclusive)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub high: Option<Quantity>,
    /// Applicable age range
    #[serde(skip_serializing_if = "Option::is_none")]
    pub age: Option<String>,
    /// Text description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl ReferenceRange {
    /// Creates a new reference range with low and high values
    pub fn new(low: f64, high: f64, unit: &str, code: &str) -> Self {
        Self {
            low: Some(Quantity::new(low, unit, code)),
            high: Some(Quantity::new(high, unit, code)),
            age: None,
            text: None,
        }
    }
}

/// Vital signs observation builder
pub struct VitalSignsObservation {
    observation: Observation,
}

impl VitalSignsObservation {
    /// Creates a new vital signs observation
    pub fn new(id: String, patient_ref: String) -> Self {
        let obs = Observation {
            id: Some(id),
            subject: Some(Reference::new("Patient", &patient_ref)),
            category: vec![CodeableConcept {
                coding: vec![Coding {
                    system: Some(
                        "http://terminology.hl7.org/CodeSystem/observation-category".to_string(),
                    ),
                    code: Some("vital-signs".to_string()),
                    display: Some("Vital Signs".to_string()),
                }],
                text: Some("Vital Signs".to_string()),
            }],
            status: Some(ObservationStatus::Final.as_str().to_string()),
            ..Default::default()
        };

        Self { observation: obs }
    }

    /// Creates a heart rate observation (LOINC: 8867-4)
    /// Time should be in RFC3339 format (e.g., "2025-01-01T12:00:00Z")
    pub fn heart_rate(id: String, patient_ref: String, bpm: f64, time: String) -> Self {
        let mut vs = Self::new(id, patient_ref);
        vs.observation.code = CodeableConcept::loinc("8867-4", "Heart rate");
        vs.observation.value_quantity = Some(Quantity::new(bpm, "beats/min", "/min"));
        vs.observation.effective_date_time = Some(time);
        vs
    }

    /// Creates a blood pressure observation (LOINC: 85354-9)
    /// Time should be in RFC3339 format (e.g., "2025-01-01T12:00:00Z")
    pub fn blood_pressure(
        id: String,
        patient_ref: String,
        systolic: f64,
        diastolic: f64,
        time: String,
    ) -> Self {
        let mut vs = Self::new(id, patient_ref);
        vs.observation.code = CodeableConcept::loinc("85354-9", "Blood pressure panel");
        vs.observation.effective_date_time = Some(time);

        // Add systolic component (LOINC: 8480-6)
        vs.observation.component.push(ObservationComponent {
            code: CodeableConcept::loinc("8480-6", "Systolic blood pressure"),
            value_quantity: Some(Quantity::new(systolic, "mmHg", "mm[Hg]")),
        });

        // Add diastolic component (LOINC: 8462-4)
        vs.observation.component.push(ObservationComponent {
            code: CodeableConcept::loinc("8462-4", "Diastolic blood pressure"),
            value_quantity: Some(Quantity::new(diastolic, "mmHg", "mm[Hg]")),
        });

        vs
    }

    /// Creates a respiratory rate observation (LOINC: 9279-1)
    /// Time should be in RFC3339 format
    pub fn respiratory_rate(id: String, patient_ref: String, breaths_per_min: f64, time: String) -> Self {
        let mut vs = Self::new(id, patient_ref);
        vs.observation.code = CodeableConcept::loinc("9279-1", "Respiratory rate");
        vs.observation.value_quantity = Some(Quantity::new(breaths_per_min, "breaths/min", "/min"));
        vs.observation.effective_date_time = Some(time);
        vs
    }

    /// Creates a body temperature observation (LOINC: 8310-5)
    /// Time should be in RFC3339 format
    pub fn body_temperature(id: String, patient_ref: String, temp_celsius: f64, time: String) -> Self {
        let mut vs = Self::new(id, patient_ref);
        vs.observation.code = CodeableConcept::loinc("8310-5", "Body temperature");
        vs.observation.value_quantity = Some(Quantity::new(temp_celsius, "°C", "Cel"));
        vs.observation.effective_date_time = Some(time);
        vs
    }

    /// Creates an oxygen saturation observation (LOINC: 59408-5)
    /// Time should be in RFC3339 format
    pub fn oxygen_saturation(id: String, patient_ref: String, spo2_percent: f64, time: String) -> Self {
        let mut vs = Self::new(id, patient_ref);
        vs.observation.code = CodeableConcept::loinc("59408-5", "Oxygen saturation");
        vs.observation.value_quantity = Some(Quantity::new(spo2_percent, "%", "%"));
        vs.observation.effective_date_time = Some(time);
        vs
    }

    /// Sets the device used for measurement
    pub fn with_device(mut self, device_ref: String) -> Self {
        self.observation.device = Some(Reference::new("Device", &device_ref));
        self
    }

    /// Sets the encounter context
    pub fn with_encounter(mut self, encounter_ref: String) -> Self {
        self.observation.encounter = Some(Reference::new("Encounter", &encounter_ref));
        self
    }

    /// Sets the status
    pub fn with_status(mut self, status: ObservationStatus) -> Self {
        self.observation.status = Some(status.as_str().to_string());
        self
    }

    /// Builds the final observation
    pub fn build(self) -> Observation {
        self.observation
    }
}

/// Waveform observation for ECG, EEG, etc.
pub struct WaveformObservation {
    observation: Observation,
}

impl WaveformObservation {
    /// Creates a new waveform observation
    pub fn new(id: String, patient_ref: String) -> Self {
        let obs = Observation {
            id: Some(id),
            subject: Some(Reference::new("Patient", &patient_ref)),
            status: Some(ObservationStatus::Final.as_str().to_string()),
            ..Default::default()
        };

        Self { observation: obs }
    }

    /// Creates an ECG waveform observation (LOINC: 131329)
    pub fn ecg(id: String, patient_ref: String, time: String) -> Self {
        let mut wf = Self::new(id, patient_ref);
        wf.observation.code = CodeableConcept::loinc("131329", "ECG 12 channel panel");
        wf.observation.effective_date_time = Some(time);
        wf
    }

    /// Creates an EEG waveform observation (LOINC: 54550-8)
    pub fn eeg(id: String, patient_ref: String, time: String) -> Self {
        let mut wf = Self::new(id, patient_ref);
        wf.observation.code = CodeableConcept::loinc("54550-8", "EEG study");
        wf.observation.effective_date_time = Some(time);
        wf
    }

    /// Creates an EMG waveform observation (LOINC: 68524-8)
    pub fn emg(id: String, patient_ref: String, time: String) -> Self {
        let mut wf = Self::new(id, patient_ref);
        wf.observation.code = CodeableConcept::loinc("68524-8", "Electromyogram study");
        wf.observation.effective_date_time = Some(time);
        wf
    }

    /// Sets the device used for recording
    pub fn with_device(mut self, device_ref: String) -> Self {
        self.observation.device = Some(Reference::new("Device", &device_ref));
        self
    }

    /// Attaches waveform data by reference, as `Observation.valueAttachment`.
    ///
    /// FHIR allows exactly one `value[x]`, so this clears any `valueQuantity`
    /// already set on the builder rather than emitting a resource with both.
    pub fn with_data_attachment(mut self, url: String, content_type: &str) -> Self {
        self.observation.value_quantity = None;
        self.observation.value_attachment = Some(Attachment::new(url, content_type));
        self
    }

    /// Builds the final observation
    pub fn build(self) -> Observation {
        self.observation
    }
}

/// Multi-component observation builder
pub struct ComponentObservation {
    observation: Observation,
}

impl ComponentObservation {
    /// Creates a new component observation
    pub fn new(id: String, patient_ref: String, code: CodeableConcept) -> Self {
        let obs = Observation {
            id: Some(id),
            subject: Some(Reference::new("Patient", &patient_ref)),
            code,
            status: Some(ObservationStatus::Final.as_str().to_string()),
            ..Default::default()
        };

        Self { observation: obs }
    }

    /// Adds a component to the observation
    pub fn add_component(mut self, code: CodeableConcept, value: Quantity) -> Self {
        self.observation.component.push(ObservationComponent {
            code,
            value_quantity: Some(value),
        });
        self
    }

    /// Sets the effective time
    pub fn with_time(mut self, time: String) -> Self {
        self.observation.effective_date_time = Some(time);
        self
    }

    /// Sets the device
    pub fn with_device(mut self, device_ref: String) -> Self {
        self.observation.device = Some(Reference::new("Device", &device_ref));
        self
    }

    /// Builds the final observation
    pub fn build(self) -> Observation {
        self.observation
    }
}

/// Common LOINC codes for biosignals
pub mod loinc_codes {
    /// Heart rate (beats per minute)
    pub const HEART_RATE: &str = "8867-4";
    /// Systolic blood pressure
    pub const SYSTOLIC_BP: &str = "8480-6";
    /// Diastolic blood pressure
    pub const DIASTOLIC_BP: &str = "8462-4";
    /// Respiratory rate
    pub const RESPIRATORY_RATE: &str = "9279-1";
    /// Oxygen saturation
    pub const OXYGEN_SATURATION: &str = "59408-5";
    /// Body temperature
    pub const BODY_TEMPERATURE: &str = "8310-5";
    /// ECG 12-channel panel
    pub const ECG_12_CHANNEL: &str = "131329";
    /// EEG study
    pub const EEG_STUDY: &str = "54550-8";
    /// EMG study
    pub const EMG_STUDY: &str = "68524-8";
    /// Heart rate variability SDNN
    pub const HRV_SDNN: &str = "80404-7";
    /// Heart rate variability RMSSD
    pub const HRV_RMSSD: &str = "80405-4";
}

/// Common SNOMED CT codes for conditions and findings
pub mod snomed_codes {
    /// Atrial fibrillation
    pub const ATRIAL_FIBRILLATION: &str = "49436004";
    /// Ventricular tachycardia
    pub const VENTRICULAR_TACHYCARDIA: &str = "25569003";
    /// Bradycardia
    pub const BRADYCARDIA: &str = "48867003";
    /// Tachycardia
    pub const TACHYCARDIA: &str = "3424008";
    /// Hypertension
    pub const HYPERTENSION: &str = "38341003";
    /// Hypotension
    pub const HYPOTENSION: &str = "45007003";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_status() {
        assert_eq!(ObservationStatus::Final.as_str(), "final");
        assert_eq!(ObservationStatus::Preliminary.as_str(), "preliminary");
    }

    #[test]
    fn test_reference_range() {
        let range = ReferenceRange::new(60.0, 100.0, "beats/min", "/min");
        assert!(range.low.is_some());
        assert!(range.high.is_some());
        assert_eq!(range.low.unwrap().value, Some(60.0));
    }

    #[test]
    fn test_heart_rate_observation() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let obs = VitalSignsObservation::heart_rate(
            "obs-hr-001".to_string(),
            "patient-123".to_string(),
            72.0,
            time,
        )
        .build();

        assert_eq!(obs.resource_type, "Observation");
        assert_eq!(obs.id, Some("obs-hr-001".to_string()));
        assert!(obs.value_quantity.is_some());
        assert_eq!(obs.value_quantity.unwrap().value, Some(72.0));
    }

    #[test]
    fn test_blood_pressure_observation() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let obs = VitalSignsObservation::blood_pressure(
            "obs-bp-001".to_string(),
            "patient-123".to_string(),
            120.0,
            80.0,
            time,
        )
        .build();

        assert_eq!(obs.component.len(), 2);
        assert_eq!(obs.component[0].value_quantity.as_ref().unwrap().value, Some(120.0));
        assert_eq!(obs.component[1].value_quantity.as_ref().unwrap().value, Some(80.0));
    }

    #[test]
    fn test_ecg_waveform() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let obs = WaveformObservation::ecg(
            "obs-ecg-001".to_string(),
            "patient-123".to_string(),
            time,
        )
        .with_device("device-001".to_string())
        .build();

        assert_eq!(obs.resource_type, "Observation");
        assert!(obs.device.is_some());
    }

    #[test]
    fn test_component_observation() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let obs = ComponentObservation::new(
            "obs-comp-001".to_string(),
            "patient-123".to_string(),
            CodeableConcept::loinc("85354-9", "Blood pressure panel"),
        )
        .add_component(
            CodeableConcept::loinc("8480-6", "Systolic BP"),
            Quantity::new(120.0, "mmHg", "mm[Hg]"),
        )
        .add_component(
            CodeableConcept::loinc("8462-4", "Diastolic BP"),
            Quantity::new(80.0, "mmHg", "mm[Hg]"),
        )
        .with_time(time)
        .build();

        assert_eq!(obs.component.len(), 2);
    }

    #[test]
    fn data_attachment_lands_in_value_attachment() {
        let obs = WaveformObservation::new("dpb-1".to_string(), "pat-1".to_string())
            .with_data_attachment("https://example.org/wave.edf".to_string(), "application/EDF")
            .build();

        let att = obs
            .value_attachment
            .as_ref()
            .expect("with_data_attachment must populate valueAttachment");
        assert_eq!(att.url.as_deref(), Some("https://example.org/wave.edf"));
        assert_eq!(att.content_type.as_deref(), Some("application/EDF"));

        // FHIR permits one value[x], so the quantity must not survive alongside it.
        assert!(obs.value_quantity.is_none());

        let json = serde_json::to_string(&obs).unwrap();
        assert!(json.contains("valueAttachment"), "attachment missing from JSON: {json}");
    }
}
