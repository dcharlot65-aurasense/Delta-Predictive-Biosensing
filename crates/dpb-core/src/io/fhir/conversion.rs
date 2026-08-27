//! DPB to FHIR Conversion Utilities
//!
//! Provides converters to transform DPB biosensor data structures into FHIR resources.

use super::observations::{VitalSignsObservation, WaveformObservation, ComponentObservation};
use super::resources::{
    CodeableConcept, Device, DiagnosticReport, Observation, Patient, Quantity, Reference,
};
use crate::types::{Context, SignalBuffer};


/// Converts DPB Context to FHIR Patient demographics
pub struct PatientDemographics;

impl PatientDemographics {
    /// Converts a DPB Context to a FHIR Patient resource
    ///
    /// # Arguments
    /// * `context` - DPB context containing patient information
    /// * `patient_id` - FHIR patient ID to use
    ///
    /// # Returns
    /// A FHIR Patient resource populated with available demographics
    pub fn from_context(context: &Context, patient_id: String) -> Patient {
        let mut patient = Patient::new(patient_id);

        // Set gender
        if let Some(sex) = &context.sex {
            patient.gender = Some(match sex.to_lowercase().as_str() {
                "m" | "male" => "male".to_string(),
                "f" | "female" => "female".to_string(),
                "o" | "other" => "other".to_string(),
                _ => "unknown".to_string(),
            });
        }

        // Note: Age would require calculating birth date, which we can't do without current date
        // Height and weight could be added as Observations instead of Patient fields

        patient
    }

    /// Creates a Patient resource from explicit demographics.
    ///
    /// Named for what it returns rather than `new`, which would suggest it
    /// constructs a `PatientDemographics`; compare `from_context` above.
    pub fn from_demographics(
        patient_id: String,
        family_name: Option<String>,
        given_names: Vec<String>,
        gender: Option<String>,
        birth_date: Option<String>,
    ) -> Patient {
        let mut patient = Patient::new(patient_id);

        if family_name.is_some() || !given_names.is_empty() {
            patient.name.push(super::resources::HumanName {
                family: family_name,
                given: given_names,
                ..Default::default()
            });
        }

        patient.gender = gender;
        patient.birth_date = birth_date;
        patient
    }
}

/// Converts DPB signal data to FHIR Observations
pub struct SignalToObservation;

impl SignalToObservation {
    /// Converts a heart rate value to a FHIR Observation
    pub fn heart_rate(
        obs_id: String,
        patient_id: String,
        bpm: f64,
        time: String,
        device_id: Option<String>,
    ) -> Observation {
        let mut obs = VitalSignsObservation::heart_rate(obs_id, patient_id, bpm, time);

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Converts blood pressure readings to a FHIR Observation
    pub fn blood_pressure(
        obs_id: String,
        patient_id: String,
        systolic: f64,
        diastolic: f64,
        time: String,
        device_id: Option<String>,
    ) -> Observation {
        let mut obs = VitalSignsObservation::blood_pressure(
            obs_id, patient_id, systolic, diastolic, time,
        );

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Converts respiratory rate to a FHIR Observation
    pub fn respiratory_rate(
        obs_id: String,
        patient_id: String,
        breaths_per_min: f64,
        time: String,
        device_id: Option<String>,
    ) -> Observation {
        let mut obs = VitalSignsObservation::respiratory_rate(
            obs_id, patient_id, breaths_per_min, time,
        );

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Converts oxygen saturation to a FHIR Observation
    pub fn oxygen_saturation(
        obs_id: String,
        patient_id: String,
        spo2_percent: f64,
        time: String,
        device_id: Option<String>,
    ) -> Observation {
        let mut obs = VitalSignsObservation::oxygen_saturation(
            obs_id, patient_id, spo2_percent, time,
        );

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Converts body temperature to a FHIR Observation
    pub fn body_temperature(
        obs_id: String,
        patient_id: String,
        temp_celsius: f64,
        time: String,
        device_id: Option<String>,
    ) -> Observation {
        let mut obs = VitalSignsObservation::body_temperature(
            obs_id, patient_id, temp_celsius, time,
        );

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Creates an ECG waveform observation reference
    ///
    /// Note: This creates a reference to waveform data, not the actual waveform.
    /// In practice, waveform data would be stored separately (e.g., as Binary resource)
    pub fn ecg_waveform(
        obs_id: String,
        patient_id: String,
        time: String,
        device_id: Option<String>,
        _signal: &SignalBuffer,
    ) -> Observation {
        let mut obs = WaveformObservation::ecg(obs_id, patient_id, time);

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Creates an EEG waveform observation reference
    pub fn eeg_waveform(
        obs_id: String,
        patient_id: String,
        time: String,
        device_id: Option<String>,
        _signal: &SignalBuffer,
    ) -> Observation {
        let mut obs = WaveformObservation::eeg(obs_id, patient_id, time);

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }

    /// Creates an EMG waveform observation reference
    pub fn emg_waveform(
        obs_id: String,
        patient_id: String,
        time: String,
        device_id: Option<String>,
        _signal: &SignalBuffer,
    ) -> Observation {
        let mut obs = WaveformObservation::emg(obs_id, patient_id, time);

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        obs.build()
    }
}

/// Converts DPB metrics to FHIR Observations
pub struct MetricsToObservation;

impl MetricsToObservation {
    /// Converts HRV time-domain metrics to FHIR Observations
    ///
    /// Creates multiple observations for SDNN, RMSSD, pNN50, etc.
    pub fn hrv_time_domain(
        patient_id: String,
        time: String,
        sdnn_ms: f64,
        rmssd_ms: f64,
        pnn50_percent: f64,
        device_id: Option<String>,
    ) -> Vec<Observation> {
        let mut observations = Vec::new();

        // SDNN observation (LOINC: 80404-7)
        let mut sdnn_obs = ComponentObservation::new(
            format!("obs-hrv-sdnn-{}", time.replace(":", "-")),
            patient_id.clone(),
            CodeableConcept::loinc("80404-7", "Heart rate variability SDNN"),
        )
        .with_time(time.clone());

        if let Some(ref device) = device_id {
            sdnn_obs = sdnn_obs.with_device(device.clone());
        }

        let mut sdnn_final = sdnn_obs.build();
        sdnn_final.value_quantity = Some(Quantity::new(sdnn_ms, "ms", "ms"));
        observations.push(sdnn_final);

        // RMSSD observation (LOINC: 80405-4)
        let mut rmssd_obs = ComponentObservation::new(
            format!("obs-hrv-rmssd-{}", time.replace(":", "-")),
            patient_id.clone(),
            CodeableConcept::loinc("80405-4", "Heart rate variability RMSSD"),
        )
        .with_time(time.clone());

        if let Some(ref device) = device_id {
            rmssd_obs = rmssd_obs.with_device(device.clone());
        }

        let mut rmssd_final = rmssd_obs.build();
        rmssd_final.value_quantity = Some(Quantity::new(rmssd_ms, "ms", "ms"));
        observations.push(rmssd_final);

        // pNN50 observation (no standard LOINC, use custom code)
        let mut pnn50_obs = ComponentObservation::new(
            format!("obs-hrv-pnn50-{}", time.replace(":", "-")),
            patient_id,
            CodeableConcept {
                coding: vec![super::resources::Coding {
                    system: Some("http://loinc.org".to_string()),
                    code: Some("80406-2".to_string()),
                    display: Some("pNN50".to_string()),
                }],
                text: Some("Percentage of successive NN intervals > 50ms".to_string()),
            },
        )
        .with_time(time.clone());

        if let Some(device) = device_id {
            pnn50_obs = pnn50_obs.with_device(device);
        }

        let mut pnn50_final = pnn50_obs.build();
        pnn50_final.value_quantity = Some(Quantity::new(pnn50_percent, "%", "%"));
        observations.push(pnn50_final);

        observations
    }

    /// Converts HRV frequency-domain metrics to FHIR Observations
    pub fn hrv_frequency_domain(
        patient_id: String,
        time: String,
        lf_power_ms2: f64,
        hf_power_ms2: f64,
        lf_hf_ratio: f64,
        device_id: Option<String>,
    ) -> Vec<Observation> {
        let mut observations = Vec::new();

        // LF Power observation
        let mut lf_obs = ComponentObservation::new(
            format!("obs-hrv-lf-{}", time.replace(":", "-")),
            patient_id.clone(),
            CodeableConcept {
                coding: vec![],
                text: Some("HRV Low Frequency Power".to_string()),
            },
        )
        .with_time(time.clone());

        if let Some(ref device) = device_id {
            lf_obs = lf_obs.with_device(device.clone());
        }

        let mut lf_final = lf_obs.build();
        lf_final.value_quantity = Some(Quantity::new(lf_power_ms2, "ms^2", "ms2"));
        observations.push(lf_final);

        // HF Power observation
        let mut hf_obs = ComponentObservation::new(
            format!("obs-hrv-hf-{}", time.replace(":", "-")),
            patient_id.clone(),
            CodeableConcept {
                coding: vec![],
                text: Some("HRV High Frequency Power".to_string()),
            },
        )
        .with_time(time.clone());

        if let Some(ref device) = device_id {
            hf_obs = hf_obs.with_device(device.clone());
        }

        let mut hf_final = hf_obs.build();
        hf_final.value_quantity = Some(Quantity::new(hf_power_ms2, "ms^2", "ms2"));
        observations.push(hf_final);

        // LF/HF Ratio observation
        let mut ratio_obs = ComponentObservation::new(
            format!("obs-hrv-lfhf-{}", time.replace(":", "-")),
            patient_id,
            CodeableConcept {
                coding: vec![],
                text: Some("HRV LF/HF Ratio".to_string()),
            },
        )
        .with_time(time.clone());

        if let Some(device) = device_id {
            ratio_obs = ratio_obs.with_device(device);
        }

        let mut ratio_final = ratio_obs.build();
        ratio_final.value_quantity = Some(Quantity::new(lf_hf_ratio, "ratio", "1"));
        observations.push(ratio_final);

        observations
    }

    /// Creates a generic metric observation
    pub fn generic_metric(
        obs_id: String,
        patient_id: String,
        metric_name: &str,
        value: f64,
        unit: &str,
        time: String,
        device_id: Option<String>,
    ) -> Observation {
        let mut obs = ComponentObservation::new(
            obs_id,
            patient_id,
            CodeableConcept {
                coding: vec![],
                text: Some(metric_name.to_string()),
            },
        )
        .with_time(time.clone());

        if let Some(device) = device_id {
            obs = obs.with_device(device);
        }

        let mut final_obs = obs.build();
        final_obs.value_quantity = Some(Quantity::new(value, unit, unit));
        final_obs
    }
}

/// Converts analysis results to FHIR DiagnosticReport
pub struct AnalysisToReport;

impl AnalysisToReport {
    /// Creates a diagnostic report from a collection of observations
    ///
    /// # Arguments
    /// * `report_id` - FHIR ID for the report
    /// * `patient_id` - Patient reference
    /// * `report_code` - Code describing the type of report
    /// * `observations` - List of observation IDs to include
    /// * `conclusion` - Clinical conclusion text
    /// * `time` - Time of the report
    pub fn create_report(
        report_id: String,
        patient_id: String,
        report_code: CodeableConcept,
        observation_ids: Vec<String>,
        conclusion: Option<String>,
        time: String,
    ) -> DiagnosticReport {
        let mut report = DiagnosticReport {
            id: Some(report_id),
            code: report_code,
            subject: Some(Reference::new("Patient", &patient_id)),
            effective_date_time: Some(time.clone()),
            issued: Some(time),
            conclusion,
            ..Default::default()
        };

        // Add observation references
        for obs_id in observation_ids {
            report.result.push(Reference::new("Observation", &obs_id));
        }

        report
    }

    /// Creates a cardiac assessment report
    pub fn cardiac_assessment(
        report_id: String,
        patient_id: String,
        observation_ids: Vec<String>,
        conclusion: String,
        time: String,
    ) -> DiagnosticReport {
        Self::create_report(
            report_id,
            patient_id,
            CodeableConcept::loinc("34752-6", "Cardiac assessment panel"),
            observation_ids,
            Some(conclusion),
            time,
        )
    }

    /// Creates an EEG study report
    pub fn eeg_study(
        report_id: String,
        patient_id: String,
        observation_ids: Vec<String>,
        conclusion: String,
        time: String,
    ) -> DiagnosticReport {
        Self::create_report(
            report_id,
            patient_id,
            CodeableConcept::loinc("54550-8", "EEG study"),
            observation_ids,
            Some(conclusion),
            time,
        )
    }

    /// Creates a sleep study report
    pub fn sleep_study(
        report_id: String,
        patient_id: String,
        observation_ids: Vec<String>,
        conclusion: String,
        time: String,
    ) -> DiagnosticReport {
        Self::create_report(
            report_id,
            patient_id,
            CodeableConcept::loinc("11520-2", "Polysomnography study"),
            observation_ids,
            Some(conclusion),
            time,
        )
    }
}

/// Device metadata converter
pub struct DeviceConverter;

impl DeviceConverter {
    /// Creates a Device resource from metadata
    pub fn create_device(
        device_id: String,
        manufacturer: Option<String>,
        model_name: Option<String>,
        serial_number: Option<String>,
        device_type_code: CodeableConcept,
    ) -> Device {
        let mut device = Device {
            id: Some(device_id),
            manufacturer,
            model_name,
            device_type: Some(device_type_code),
            ..Default::default()
        };

        if let Some(serial) = serial_number {
            device.identifier.push(super::resources::Identifier {
                system: Some("urn:ietf:rfc:3986".to_string()),
                value: Some(serial),
                r#use: Some("official".to_string()),
                assigner: None,
            });
        }

        device
    }

    /// Creates an ECG device
    pub fn ecg_device(
        device_id: String,
        manufacturer: Option<String>,
        model: Option<String>,
    ) -> Device {
        Self::create_device(
            device_id,
            manufacturer,
            model,
            None,
            CodeableConcept::snomed("706203009", "Electrocardiograph"),
        )
    }

    /// Creates an EEG device
    pub fn eeg_device(
        device_id: String,
        manufacturer: Option<String>,
        model: Option<String>,
    ) -> Device {
        Self::create_device(
            device_id,
            manufacturer,
            model,
            None,
            CodeableConcept::snomed("706211002", "Electroencephalograph"),
        )
    }

    /// Creates a pulse oximeter device
    pub fn pulse_oximeter(
        device_id: String,
        manufacturer: Option<String>,
        model: Option<String>,
    ) -> Device {
        Self::create_device(
            device_id,
            manufacturer,
            model,
            None,
            CodeableConcept::snomed("706172005", "Pulse oximeter"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_patient_from_context() {
        let mut context = Context::new();
        context.sex = Some("M".to_string());

        let patient = PatientDemographics::from_context(&context, "patient-001".to_string());
        assert_eq!(patient.id, Some("patient-001".to_string()));
        assert_eq!(patient.gender, Some("male".to_string()));
    }

    #[test]
    fn test_heart_rate_conversion() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let obs = SignalToObservation::heart_rate(
            "obs-hr-001".to_string(),
            "patient-001".to_string(),
            72.0,
            time,
            Some("device-001".to_string()),
        );

        assert_eq!(obs.id, Some("obs-hr-001".to_string()));
        assert!(obs.device.is_some());
        assert_eq!(obs.value_quantity.unwrap().value, Some(72.0));
    }

    #[test]
    fn test_hrv_metrics_conversion() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let observations = MetricsToObservation::hrv_time_domain(
            "patient-001".to_string(),
            time,
            50.0,  // SDNN
            30.0,  // RMSSD
            20.0,  // pNN50
            Some("device-001".to_string()),
        );

        assert_eq!(observations.len(), 3);
        assert!(observations[0].device.is_some());
    }

    #[test]
    fn test_diagnostic_report_creation() {
        let time = "2025-01-01T00:00:00Z".to_string();
        let report = AnalysisToReport::cardiac_assessment(
            "report-001".to_string(),
            "patient-001".to_string(),
            vec!["obs-1".to_string(), "obs-2".to_string()],
            "Normal sinus rhythm".to_string(),
            time,
        );

        assert_eq!(report.id, Some("report-001".to_string()));
        assert_eq!(report.result.len(), 2);
        assert!(report.conclusion.is_some());
    }

    #[test]
    fn test_device_creation() {
        let device = DeviceConverter::ecg_device(
            "device-001".to_string(),
            Some("Acme Medical".to_string()),
            Some("ECG-3000".to_string()),
        );

        assert_eq!(device.id, Some("device-001".to_string()));
        assert_eq!(device.manufacturer, Some("Acme Medical".to_string()));
        assert!(device.device_type.is_some());
    }
}
