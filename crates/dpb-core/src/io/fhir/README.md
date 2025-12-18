# HL7 FHIR Support for Delta-Predictive-Biosensing

This module provides comprehensive support for HL7 FHIR R4 (Fast Healthcare Interoperability Resources) standard, enabling seamless integration with electronic health record (EHR) systems and clinical data exchange.

## Features

- **Core FHIR Resources**: Patient, Observation, DiagnosticReport, Device, Encounter, Procedure, Condition
- **Vital Signs**: Heart rate, blood pressure, respiratory rate, temperature, SpO2
- **Waveform References**: ECG, EEG, EMG observations
- **HRV Metrics**: Time and frequency domain heart rate variability measures
- **FHIR Bundles**: Document, transaction, batch, and search result bundles
- **Serialization**: JSON (FHIR R4 compliant) and basic XML support
- **DPB Conversion**: Convert biosensor data to FHIR resources

## Quick Start

### Creating a Patient Resource

```rust
use dpb_core::io::fhir::{Patient, HumanName, Identifier};

let mut patient = Patient::new("patient-001".to_string());
patient.name.push(HumanName {
    family: Some("Smith".to_string()),
    given: vec!["John".to_string()],
    ..Default::default()
});
patient.gender = Some("male".to_string());
patient.birth_date = Some("1980-01-01".to_string());
```

### Creating Vital Sign Observations

```rust
use dpb_core::io::fhir::observations::VitalSignsObservation;

// Heart rate
let hr_obs = VitalSignsObservation::heart_rate(
    "obs-hr-001".to_string(),
    "patient-001".to_string(),
    72.0,  // BPM
    "2025-01-01T12:00:00Z".to_string()
)
.with_device("device-001".to_string())
.build();

// Blood pressure
let bp_obs = VitalSignsObservation::blood_pressure(
    "obs-bp-001".to_string(),
    "patient-001".to_string(),
    120.0,  // Systolic
    80.0,   // Diastolic
    "2025-01-01T12:00:00Z".to_string()
)
.build();
```

### Converting DPB Data to FHIR

```rust
use dpb_core::io::fhir::conversion::{SignalToObservation, MetricsToObservation};

// Convert heart rate from biosensor
let hr_observation = SignalToObservation::heart_rate(
    "obs-hr-123".to_string(),
    "patient-001".to_string(),
    75.0,
    "2025-01-01T12:00:00Z".to_string(),
    Some("device-ecg-001".to_string())
);

// Convert HRV metrics
let hrv_observations = MetricsToObservation::hrv_time_domain(
    "patient-001".to_string(),
    "2025-01-01T12:00:00Z".to_string(),
    50.0,  // SDNN (ms)
    30.0,  // RMSSD (ms)
    20.0,  // pNN50 (%)
    Some("device-ecg-001".to_string())
);
```

### Creating a DiagnosticReport

```rust
use dpb_core::io::fhir::conversion::AnalysisToReport;
use dpb_core::io::fhir::CodeableConcept;

let report = AnalysisToReport::cardiac_assessment(
    "report-001".to_string(),
    "patient-001".to_string(),
    vec!["obs-hr-001".to_string(), "obs-bp-001".to_string()],
    "Normal sinus rhythm with appropriate heart rate variability.".to_string(),
    "2025-01-01T12:30:00Z".to_string()
);
```

### Creating FHIR Bundles

```rust
use dpb_core::io::fhir::{Bundle, FhirResource};

let mut bundle = Bundle::collection()
    .with_id("bundle-001".to_string())
    .add_resource(FhirResource::Patient(patient))
    .add_resource(FhirResource::Observation(hr_obs))
    .add_resource(FhirResource::Observation(bp_obs));
```

### JSON Serialization

```rust
use dpb_core::io::fhir::serialization::{JsonSerializer, FhirSerializer};

let serializer = JsonSerializer::pretty();
let json = serializer.serialize(&bundle)?;
println!("{}", json);
```

### Device Metadata

```rust
use dpb_core::io::fhir::conversion::DeviceConverter;

let ecg_device = DeviceConverter::ecg_device(
    "device-ecg-001".to_string(),
    Some("Acme Medical".to_string()),
    Some("ECG-3000".to_string())
);

let pulse_ox = DeviceConverter::pulse_oximeter(
    "device-spo2-001".to_string(),
    Some("MedTech Inc".to_string()),
    Some("PulseOx-Pro".to_string())
);
```

## Standard Codes

### LOINC Codes for Vital Signs

| Code | Description |
|------|-------------|
| 8867-4 | Heart rate |
| 8480-6 | Systolic blood pressure |
| 8462-4 | Diastolic blood pressure |
| 9279-1 | Respiratory rate |
| 8310-5 | Body temperature |
| 59408-5 | Oxygen saturation |
| 80404-7 | HRV SDNN |
| 80405-4 | HRV RMSSD |

### SNOMED CT Codes for Devices

| Code | Description |
|------|-------------|
| 706203009 | Electrocardiograph |
| 706211002 | Electroencephalograph |
| 706172005 | Pulse oximeter |

## FHIR Client (Placeholder)

The `FhirClient` provides a structure for interacting with FHIR servers. Note: Full HTTP implementation requires integrating an HTTP client library like `reqwest`.

```rust
use dpb_core::io::fhir::client::{FhirClientConfig, FhirClient};

let config = FhirClientConfig::new("https://example.com/fhir".to_string())
    .with_auth_token("your-token".to_string());

let client = FhirClient::new(config);
// Note: Actual HTTP operations not implemented in base library
```

## Testing

All FHIR resources include comprehensive tests:

```bash
cargo test --package dpb-core --lib io::fhir
```

## Compliance

This implementation follows HL7 FHIR R4 specification:
- Resource structures match FHIR R4 profiles
- JSON serialization follows FHIR JSON format
- LOINC codes for observations
- SNOMED CT codes for clinical concepts
- Standard code systems (UCUM for units)

## Integration with DPB

The FHIR module integrates seamlessly with other DPB components:

```rust
use dpb_core::types::Context;
use dpb_core::io::fhir::conversion::PatientDemographics;

// Convert DPB Context to FHIR Patient
let context = Context::new()
    .with_sex("M".to_string());

let patient = PatientDemographics::from_context(
    &context,
    "patient-001".to_string()
);
```

## Future Enhancements

- Full HTTP client implementation with `reqwest`
- Complete XML serialization/deserialization
- FHIR Search parameter builders
- Batch/transaction execution
- Subscription support
- Terminology service integration
- SMART on FHIR authentication

## References

- [HL7 FHIR R4 Specification](http://hl7.org/fhir/R4/)
- [LOINC Database](https://loinc.org/)
- [SNOMED CT](https://www.snomed.org/)
- [UCUM Units of Measure](https://ucum.org/)
