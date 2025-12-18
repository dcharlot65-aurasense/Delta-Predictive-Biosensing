//! Integration tests for dpb-clinical
//!
//! Tests clinical utilities end-to-end.

use dpb_clinical::*;
use dpb_clinical::phi::*;

// ============================================================================
// Demographics Tests
// ============================================================================

#[test]
fn test_demographics_creation() {
    let demo = Demographics::new()
        .with_age(45)
        .with_sex(Sex::Female)
        .with_ethnicity(Ethnicity::EastAsian);

    assert_eq!(demo.age, Some(45));
    assert_eq!(demo.sex, Some(Sex::Female));
    assert_eq!(demo.ethnicity, Some(Ethnicity::EastAsian));
}

#[test]
fn test_age_group_classification() {
    assert_eq!(AgeGroup::from_age(5), AgeGroup::Pediatric);
    assert_eq!(AgeGroup::from_age(17), AgeGroup::Pediatric);
    assert_eq!(AgeGroup::from_age(18), AgeGroup::Adult);
    assert_eq!(AgeGroup::from_age(64), AgeGroup::Adult);
    assert_eq!(AgeGroup::from_age(65), AgeGroup::Geriatric);
    assert_eq!(AgeGroup::from_age(90), AgeGroup::Geriatric);
}

#[test]
fn test_ethnicity_variants() {
    let ethnicities = vec![
        Ethnicity::White,
        Ethnicity::Black,
        Ethnicity::Hispanic,
        Ethnicity::EastAsian,
        Ethnicity::SouthAsian,
        Ethnicity::MiddleEastern,
        Ethnicity::NativeAmerican,
        Ethnicity::PacificIslander,
        Ethnicity::Mixed,
        Ethnicity::Other,
    ];

    assert_eq!(ethnicities.len(), 10);
}

// ============================================================================
// Normative Database Tests
// ============================================================================

#[test]
fn test_normative_database() {
    let mut db = NormativeDatabase::new("test_norms");

    // Add a reference
    let reference = NormativeReference {
        metric_name: "alpha_power".to_string(),
        description: "Alpha band power".to_string(),
        unit: "µV²".to_string(),
        population_norms: vec![
            PopulationNorms {
                age_group: AgeGroup::Adult,
                sex: Some(Sex::Male),
                ethnicity: None,
                mean: 10.0,
                std: 2.0,
                percentiles: Some(vec![
                    (5, 6.7),
                    (25, 8.6),
                    (50, 10.0),
                    (75, 11.4),
                    (95, 13.3),
                ]),
            },
        ],
    };

    db.add_reference(reference);

    // Test lookup
    let demo = Demographics::new()
        .with_age(35)
        .with_sex(Sex::Male);

    let norms = db.get_reference(&demo, "alpha_power");
    assert!(norms.is_some());
}

#[test]
fn test_z_score_calculation() {
    let norms = PopulationNorms {
        age_group: AgeGroup::Adult,
        sex: None,
        ethnicity: None,
        mean: 100.0,
        std: 15.0,
        percentiles: None,
    };

    // Value at mean
    let z = norms.z_score(100.0);
    assert!((z - 0.0).abs() < 0.001);

    // Value 1 SD above
    let z = norms.z_score(115.0);
    assert!((z - 1.0).abs() < 0.001);

    // Value 2 SD below
    let z = norms.z_score(70.0);
    assert!((z - (-2.0)).abs() < 0.001);
}

#[test]
fn test_percentile_rank() {
    let norms = PopulationNorms {
        age_group: AgeGroup::Adult,
        sex: None,
        ethnicity: None,
        mean: 100.0,
        std: 15.0,
        percentiles: None,
    };

    // Approximate percentile from z-score
    let percentile = norms.percentile_rank(100.0);
    assert!((percentile - 50.0).abs() < 1.0); // ~50th percentile
}

// ============================================================================
// Treatment Response Tests
// ============================================================================

#[test]
fn test_effect_size_cohens_d() {
    let response = TreatmentResponse::new(
        vec![10.0, 12.0, 11.0, 13.0, 10.0], // Pre
        vec![15.0, 17.0, 16.0, 18.0, 15.0], // Post
    );

    let effect = response.cohens_d();

    // Large effect (d > 0.8)
    assert!(effect.value > 0.8);
    assert_eq!(effect.interpretation(), "Large");
}

#[test]
fn test_effect_size_hedges_g() {
    let response = TreatmentResponse::new(
        vec![10.0, 12.0, 11.0],
        vec![15.0, 17.0, 16.0],
    );

    let hedges = response.hedges_g();

    // Hedges' g should be slightly smaller than Cohen's d for small samples
    let cohens = response.cohens_d();
    assert!(hedges.value <= cohens.value);
}

#[test]
fn test_clinical_significance() {
    let response = TreatmentResponse::new(
        vec![50.0, 55.0, 52.0, 48.0, 51.0],
        vec![35.0, 40.0, 38.0, 32.0, 36.0],
    );

    let is_significant = response.is_clinically_significant(1.96);
    assert!(is_significant);
}

#[test]
fn test_intervention_model() {
    let model = InterventionModel::new()
        .add_timepoint("baseline", vec![50.0, 52.0, 48.0])
        .add_timepoint("week4", vec![45.0, 47.0, 43.0])
        .add_timepoint("week8", vec![40.0, 42.0, 38.0]);

    assert_eq!(model.timepoints().len(), 3);

    // Check trend
    let trend = model.compute_trend();
    assert!(trend < 0.0); // Decreasing trend
}

// ============================================================================
// Comorbidity Tests
// ============================================================================

#[test]
fn test_comorbidity_model() {
    let mut model = ComorbidityModel::new();

    let diabetes = Condition::new("Diabetes Type 2")
        .with_severity(0.6)
        .with_duration_years(5.0);

    let hypertension = Condition::new("Hypertension")
        .with_severity(0.4)
        .with_duration_years(8.0);

    model.add_condition(diabetes);
    model.add_condition(hypertension);

    // Add interaction
    model.add_interaction(Interaction {
        condition_a: "Diabetes Type 2".to_string(),
        condition_b: "Hypertension".to_string(),
        interaction_type: InteractionType::Synergistic,
        modifier: 1.3, // 30% increase in combined effect
    });

    let profile = model.compute_profile();
    assert!(profile.total_burden > 0.0);
}

#[test]
fn test_condition_interactions() {
    let synergistic = InteractionType::Synergistic;
    let antagonistic = InteractionType::Antagonistic;
    let independent = InteractionType::Independent;

    assert_ne!(format!("{:?}", synergistic), format!("{:?}", antagonistic));
    assert_ne!(format!("{:?}", antagonistic), format!("{:?}", independent));
}

// ============================================================================
// Practice Effects Tests
// ============================================================================

#[test]
fn test_practice_effect_corrector() {
    let corrector = PracticeEffectCorrector::new()
        .with_expected_gain(3.0)  // Expected 3-point gain from practice
        .with_reliability(0.85);

    let baseline = 100.0;
    let retest = 108.0;

    let corrected = corrector.correct(baseline, retest);

    // Corrected score should be lower than raw retest
    assert!(corrected < retest);
    // But still higher than baseline (real improvement)
    assert!(corrected > baseline);
}

#[test]
fn test_srb_calculator() {
    let calc = SRBCalculator::new(0.85, 15.0); // reliability, SD

    let baseline = 100.0;
    let retest = 110.0;

    let srb = calc.compute_srb(baseline, retest);

    // SRB is standardized
    assert!(srb.abs() < 5.0); // Reasonable range
}

#[test]
fn test_serial_assessment() {
    let mut assessment = SerialAssessment::new("memory_test");

    assessment.add_score(0, 100.0);  // Baseline
    assessment.add_score(6, 105.0);  // 6 months
    assessment.add_score(12, 108.0); // 12 months

    assert_eq!(assessment.num_timepoints(), 3);

    let change = assessment.total_change();
    assert!((change - 8.0).abs() < 0.001);
}

// ============================================================================
// PHI/HIPAA Tests
// ============================================================================

#[test]
fn test_safe_harbor_deidentification() {
    let deidentifier = DeIdentifier::safe_harbor();

    let record = PatientRecord::new()
        .with_name("John Doe")
        .with_ssn("123-45-6789")
        .with_dob(1980, 5, 15)
        .with_phone("555-123-4567")
        .with_email("john.doe@example.com")
        .with_clinical_value("heart_rate", 72.0);

    let result = deidentifier.deidentify(&record).unwrap();

    // PHI should be removed
    assert!(result.pseudonym_id.is_none()); // No pseudonymization in safe harbor

    // Year retained (allowed under Safe Harbor)
    assert_eq!(result.birth_year, Some(1980));

    // Clinical data preserved
    assert_eq!(result.clinical_data.get("heart_rate"), Some(&72.0));

    // Audit log should have entries
    assert!(!result.audit_log.is_empty());
}

#[test]
fn test_limited_data_set() {
    let deidentifier = DeIdentifier::limited_data_set();

    let record = PatientRecord::new()
        .with_patient_id("PAT001")
        .with_name("Jane Smith")
        .with_dob(1975, 8, 20)
        .with_location("123 Main St", "Boston", "MA", "02101");

    let result = deidentifier.deidentify(&record).unwrap();

    // Should have pseudonym for patient ID
    assert!(result.pseudonym_id.is_some());

    // State retained in limited data set
    assert_eq!(result.state, Some("MA".to_string()));
}

#[test]
fn test_research_pseudonymization() {
    let config = DeIdentificationConfig::research_pseudonymization();
    let deidentifier = DeIdentifier::new(config);

    let record1 = PatientRecord::new()
        .with_patient_id("PAT001")
        .with_name("Alice");

    let record2 = PatientRecord::new()
        .with_patient_id("PAT001")
        .with_name("Alice");

    let result1 = deidentifier.deidentify(&record1).unwrap();
    let result2 = deidentifier.deidentify(&record2).unwrap();

    // Same ID should produce same pseudonym
    assert_eq!(result1.pseudonym_id, result2.pseudonym_id);
}

#[test]
fn test_phi_identifiers_all_18() {
    let all = PhiIdentifier::all();
    assert_eq!(all.len(), 18);
}

#[test]
fn test_k_anonymity() {
    let checker = KAnonymityChecker::new(2);

    // Two records in same equivalence class
    let records = vec![
        DeIdentifiedRecord {
            pseudonym_id: None,
            birth_year: Some(1985),
            age_group: Some("35-44".to_string()),
            state: Some("CA".to_string()),
            device_id_masked: None,
            clinical_data: std::collections::HashMap::new(),
            audit_log: vec![],
        },
        DeIdentifiedRecord {
            pseudonym_id: None,
            birth_year: Some(1985),
            age_group: Some("35-44".to_string()),
            state: Some("CA".to_string()),
            device_id_masked: None,
            clinical_data: std::collections::HashMap::new(),
            audit_log: vec![],
        },
    ];

    let result = checker.check(&records);
    assert!(result.satisfies_k);
    assert_eq!(result.min_group_size, 2);
}

#[test]
fn test_k_anonymity_violation() {
    let checker = KAnonymityChecker::new(3); // Require 3

    // Only 2 records - violates k=3
    let records = vec![
        DeIdentifiedRecord {
            pseudonym_id: None,
            birth_year: Some(1990),
            age_group: Some("30-39".to_string()),
            state: Some("NY".to_string()),
            device_id_masked: None,
            clinical_data: std::collections::HashMap::new(),
            audit_log: vec![],
        },
        DeIdentifiedRecord {
            pseudonym_id: None,
            birth_year: Some(1990),
            age_group: Some("30-39".to_string()),
            state: Some("NY".to_string()),
            device_id_masked: None,
            clinical_data: std::collections::HashMap::new(),
            audit_log: vec![],
        },
    ];

    let result = checker.check(&records);
    assert!(!result.satisfies_k);
    assert!(!result.violations.is_empty());
}

#[test]
fn test_l_diversity() {
    let checker = LDiversityChecker::new(2, "diagnosis_code");

    let mut clinical1 = std::collections::HashMap::new();
    clinical1.insert("diagnosis_code".to_string(), 1.0);

    let mut clinical2 = std::collections::HashMap::new();
    clinical2.insert("diagnosis_code".to_string(), 2.0);

    let records = vec![
        DeIdentifiedRecord {
            pseudonym_id: None,
            birth_year: None,
            age_group: Some("40-49".to_string()),
            state: Some("TX".to_string()),
            device_id_masked: None,
            clinical_data: clinical1,
            audit_log: vec![],
        },
        DeIdentifiedRecord {
            pseudonym_id: None,
            birth_year: None,
            age_group: Some("40-49".to_string()),
            state: Some("TX".to_string()),
            device_id_masked: None,
            clinical_data: clinical2,
            audit_log: vec![],
        },
    ];

    let result = checker.check(&records);
    assert!(result.satisfies_l);
}

#[test]
fn test_batch_deidentification() {
    let deidentifier = DeIdentifier::safe_harbor();

    let records = vec![
        PatientRecord::new()
            .with_name("Patient 1")
            .with_dob(1980, 1, 1),
        PatientRecord::new()
            .with_name("Patient 2")
            .with_dob(1985, 6, 15),
        PatientRecord::new()
            .with_name("Patient 3")
            .with_dob(1990, 12, 31),
    ];

    let results = deidentifier.deidentify_batch(&records).unwrap();
    assert_eq!(results.len(), 3);
}

#[test]
fn test_anonymization_stats() {
    let mut stats = AnonymizationStats::default();

    let record = DeIdentifiedRecord {
        pseudonym_id: Some("PSN-001".to_string()),
        birth_year: Some(1985),
        age_group: None,
        state: Some("CA".to_string()),
        device_id_masked: None,
        clinical_data: std::collections::HashMap::new(),
        audit_log: vec![
            AuditEntry {
                identifier_type: PhiIdentifier::Name,
                method_applied: DeIdentificationMethod::Remove,
                timestamp: 0,
            },
            AuditEntry {
                identifier_type: PhiIdentifier::SocialSecurityNumber,
                method_applied: DeIdentificationMethod::Remove,
                timestamp: 0,
            },
        ],
    };

    stats.update(&record);

    assert_eq!(stats.records_processed, 1);
    assert_eq!(stats.identifiers_removed, 2);
}

#[test]
fn test_age_over_89_generalization() {
    let deidentifier = DeIdentifier::safe_harbor();

    // 95-year-old patient
    let record = PatientRecord::new()
        .with_dob(1930, 1, 1);

    let result = deidentifier.deidentify(&record).unwrap();

    // Should be generalized to "90+"
    assert_eq!(result.age_group, Some("90+".to_string()));
}

#[test]
fn test_deidentification_config_builder() {
    let config = DeIdentificationConfig::new()
        .with_hash_salt("my-secret-salt")
        .with_pseudonym_key("my-pseudonym-key")
        .with_method(PhiIdentifier::Name, DeIdentificationMethod::Hash);

    assert!(config.hash_salt.is_some());
    assert_eq!(
        config.methods.get(&PhiIdentifier::Name),
        Some(&DeIdentificationMethod::Hash)
    );
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_clinical_error_variants() {
    let err1 = ClinicalError::MissingNormativeData("alpha_power".to_string());
    let err2 = ClinicalError::InvalidConfiguration("bad config".to_string());
    let err3 = ClinicalError::InsufficientData(5, 10);

    assert!(format!("{:?}", err1).contains("alpha_power"));
    assert!(format!("{:?}", err2).contains("bad config"));
    assert!(format!("{:?}", err3).contains("5"));
}
