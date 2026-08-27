//! Integration tests for dpb-clinical
//!
//! Tests clinical utilities end-to-end.
//!
//! The first half of this file was rewritten against the crate's actual API.
//! It had never compiled: it referenced `SRBCalculator`, `InteractionType`,
//! `TreatmentResponse::cohens_d`, `PopulationNorms::percentile_rank`,
//! `SerialAssessment::add_score` and others that do not exist, so it
//! contributed no coverage while appearing to, and blocked
//! `cargo test --workspace` outright. The PHI half below was already sound.

use dpb_clinical::phi::*;
use dpb_clinical::*;

// ============================================================================
// Demographics
// ============================================================================

#[test]
fn test_demographics_are_built_incrementally() {
    // Every field is optional: nothing is assumed about a subject who has not
    // reported it.
    let empty = Demographics::new();
    assert!(empty.age.is_none());
    assert!(empty.sex.is_none());

    let demo = Demographics::new()
        .with_age(45)
        .with_sex(Sex::Female)
        .with_ethnicity(Ethnicity::European)
        .with_education(16);

    assert_eq!(demo.age, Some(45));
    assert_eq!(demo.sex, Some(Sex::Female));
    assert_eq!(demo.ethnicity, Some(Ethnicity::European));
    assert_eq!(demo.education_years, Some(16));
}

#[test]
fn test_age_group_classification_is_monotone_and_contiguous() {
    // A child and an older adult must not land in the same group.
    assert_ne!(AgeGroup::from_age(8), AgeGroup::from_age(78));

    // Classification never revisits a group it has already left, which is what
    // makes the groups contiguous bands rather than an arbitrary mapping.
    let mut seen: Vec<AgeGroup> = Vec::new();
    for age in 0u8..=100 {
        let g = AgeGroup::from_age(age);
        if seen.last() != Some(&g) {
            assert!(!seen.contains(&g), "age group {g:?} recurs at age {age}");
            seen.push(g);
        }
    }
    assert!(seen.len() > 1, "every age fell into a single group");

    // Each group's declared range must contain the ages mapped to it.
    for age in 0u8..=100 {
        let (min, max) = AgeGroup::from_age(age).age_range();
        assert!(
            (min..=max).contains(&age),
            "age {age} maps to a group spanning [{min}, {max}]"
        );
    }
}

#[test]
fn test_ethnicity_variants_are_distinct() {
    let variants = [
        Ethnicity::European,
        Ethnicity::EastAsian,
        Ethnicity::SouthAsian,
        Ethnicity::African,
        Ethnicity::Hispanic,
        Ethnicity::Other,
        Ethnicity::Unknown,
    ];
    for (i, a) in variants.iter().enumerate() {
        for b in variants.iter().skip(i + 1) {
            assert_ne!(a, b);
        }
    }
}

// ============================================================================
// Normative comparison
// ============================================================================

fn norms_with(measure: &str, mean: f64, sd: f64) -> PopulationNorms {
    let mut norms = PopulationNorms::new("test population");
    norms.add_reference(measure, NormativeReference::new(mean, sd));
    norms
}

#[test]
fn test_normative_database_round_trip() {
    let mut db = NormativeDatabase::new("Test Norms", "1.0");
    db.add_norms("adult_female", norms_with("alpha_power", 10.0, 2.0), 250);

    assert_eq!(db.sample_size("adult_female"), Some(250));
    assert!(db.get_norms("adult_female").is_some());
    assert!(db.get_norms("missing_population").is_none());
    assert!(db.population_keys().iter().any(|k| *k == "adult_female"));
}

/// A database holding one measure's norms, keyed off the demographics that
/// will be used to look them up.
fn db_for(demo: &Demographics, measure: &str, mean: f64, sd: f64) -> NormativeDatabase {
    let mut db = NormativeDatabase::new("Test Norms", "1.0");
    db.add_norms(&demo.population_key(), norms_with(measure, mean, sd), 100);
    db
}

#[test]
fn test_z_score_is_signed_and_scaled() {
    let demo = Demographics::new().with_age(45).with_sex(Sex::Female);
    let db = db_for(&demo, "alpha_power", 10.0, 2.0);

    // Exactly one SD above the mean.
    let z = db.z_score("alpha_power", 12.0, &demo).expect("z-score");
    assert!((z - 1.0).abs() < 1e-9, "expected z = 1.0, got {z}");

    // ...and one below.
    let z = db.z_score("alpha_power", 8.0, &demo).expect("z-score");
    assert!((z + 1.0).abs() < 1e-9, "expected z = -1.0, got {z}");

    // At the mean.
    let z = db.z_score("alpha_power", 10.0, &demo).expect("z-score");
    assert!(z.abs() < 1e-9);

    // An unknown measure has no reference to compare against.
    assert!(db.z_score("not_a_measure", 1.0, &demo).is_err());
}

#[test]
fn test_percentile_tracks_the_normal_curve() {
    let demo = Demographics::new().with_age(45).with_sex(Sex::Female);
    let db = db_for(&demo, "alpha_power", 10.0, 2.0);

    // The mean sits at the 50th percentile by construction.
    let p = db
        .percentile("alpha_power", 10.0, &demo)
        .expect("percentile");
    assert!(
        (p - 50.0).abs() < 1.0,
        "mean should be ~50th percentile, got {p}"
    );

    // +1 SD is ~84th, -1 SD ~16th.
    let high = db
        .percentile("alpha_power", 12.0, &demo)
        .expect("percentile");
    let low = db
        .percentile("alpha_power", 8.0, &demo)
        .expect("percentile");
    assert!((high - 84.13).abs() < 1.0, "got {high}");
    assert!((low - 15.87).abs() < 1.0, "got {low}");

    // Percentile must be monotone in the score.
    assert!(low < p && p < high);
}

// ============================================================================
// Treatment response and effect size
// ============================================================================

#[test]
fn test_cohens_d_matches_its_definition() {
    // Means one pooled SD apart => d = 1.0.
    let d = EffectSize::cohens_d(12.0, 10.0, 2.0, 2.0, 30, 30).expect("effect size");
    assert!(
        (d.value - 1.0).abs() < 1e-6,
        "expected d = 1.0, got {}",
        d.value
    );

    // A confidence interval must bracket the estimate.
    if let (Some(lo), Some(hi)) = (d.ci_lower, d.ci_upper) {
        assert!(
            lo < d.value && d.value < hi,
            "CI [{lo}, {hi}] excludes {}",
            d.value
        );
    }

    // No difference => no effect.
    let none = EffectSize::cohens_d(10.0, 10.0, 2.0, 2.0, 30, 30).expect("effect size");
    assert!(none.value.abs() < 1e-9);

    // The sign follows the direction of the difference.
    let negative = EffectSize::cohens_d(8.0, 10.0, 2.0, 2.0, 30, 30).expect("effect size");
    assert!(
        negative.value < 0.0,
        "expected a negative effect, got {}",
        negative.value
    );

    // A zero pooled SD leaves the effect undefined rather than infinite.
    assert!(EffectSize::cohens_d(12.0, 10.0, 0.0, 0.0, 30, 30).is_err());
}

#[test]
fn test_hedges_g_is_cohens_d_corrected_downward() {
    let d = EffectSize::cohens_d(12.0, 10.0, 2.0, 2.0, 10, 10).expect("d");
    let g = EffectSize::hedges_g(12.0, 10.0, 2.0, 2.0, 10, 10).expect("g");

    // Hedges' g applies a small-sample correction, so it is strictly smaller in
    // magnitude and approaches d as n grows.
    assert!(
        g.value.abs() < d.value.abs(),
        "g {} should be below d {}",
        g.value,
        d.value
    );

    let d_large = EffectSize::cohens_d(12.0, 10.0, 2.0, 2.0, 500, 500).expect("d");
    let g_large = EffectSize::hedges_g(12.0, 10.0, 2.0, 2.0, 500, 500).expect("g");
    assert!(
        (g_large.value - d_large.value).abs() < (g.value - d.value).abs(),
        "the correction should shrink with sample size"
    );
}

#[test]
fn test_treatment_response_tracks_change() {
    let mut baseline = Assessment::new(0.0, "baseline");
    baseline.add_value("updrs", 40.0);

    let mut response = TreatmentResponse::new("P001", "levodopa", baseline);

    let mut followup = Assessment::new(12.0, "week-12");
    followup.add_value("updrs", 30.0);
    response.add_followup(followup);

    assert!(response.latest_followup().is_some());

    // A 10-point drop from 40 is -10 absolute and -25%.
    let change = response.change_from_baseline("updrs").expect("change");
    assert!((change + 10.0).abs() < 1e-9, "got {change}");

    let percent = response.percent_change("updrs").expect("percent change");
    assert!((percent + 25.0).abs() < 1e-6, "got {percent}");

    // An unmeasured variable has no change to report.
    assert!(response.change_from_baseline("not_measured").is_none());

    // The trajectory carries both timepoints.
    assert_eq!(response.trajectory("updrs").len(), 2);
}

// ============================================================================
// Comorbidity
// ============================================================================

#[test]
fn test_comorbidity_index_grows_with_burden() {
    let mut model = ComorbidityModel::new("test");
    model.add_condition(
        Condition::new("E11", "Type 2 diabetes", ConditionCategory::Metabolic).with_severity(0.5),
    );
    model.add_condition(
        Condition::new("I10", "Hypertension", ConditionCategory::Cardiovascular).with_severity(0.3),
    );

    assert_eq!(model.conditions().len(), 2);
    assert!(model.get_condition("E11").is_some());
    assert!(model.get_condition("nonexistent").is_none());

    let one = model.comorbidity_index(&["E11"]);
    let both = model.comorbidity_index(&["E11", "I10"]);
    assert!(
        both > one,
        "two conditions should not score below one: {both} vs {one}"
    );
    assert!(model.comorbidity_index(&[]) <= one);
}

#[test]
fn test_condition_interactions_are_symmetric_lookups() {
    let mut model = ComorbidityModel::new("test");
    model.add_condition(Condition::new(
        "E11",
        "Diabetes",
        ConditionCategory::Metabolic,
    ));
    model.add_condition(Condition::new(
        "I10",
        "Hypertension",
        ConditionCategory::Cardiovascular,
    ));
    model.add_interaction(
        "E11",
        "I10",
        Interaction::new("E11", "I10", InteractionType::Synergistic),
    );

    assert!(model.get_interaction("E11", "I10").is_some());
    // Order must not matter: the pair is unordered.
    assert!(model.get_interaction("I10", "E11").is_some());
    assert!(model.get_interaction("E11", "Z99").is_none());
}

// ============================================================================
// Practice effects
// ============================================================================

#[test]
fn test_practice_effect_correction_reduces_a_repeat_score() {
    let mut corrector = PracticeEffectCorrector::new("test", CorrectionMethod::SimpleSubtraction);
    corrector.add_effect("trails_b", PracticeEffect::new("trails_b", 5.0));

    assert!(corrector.get_effect("trails_b").is_some());

    // The first assessment carries no practice effect to remove.
    let first = corrector
        .correct_score("trails_b", 50.0, 1, None)
        .expect("correction");
    assert!(
        (first.corrected_score - 50.0).abs() < 1e-9,
        "first session adjusted to {}",
        first.corrected_score
    );
    assert!(first.correction_applied.abs() < 1e-9);

    // A later one does.
    let second = corrector
        .correct_score("trails_b", 50.0, 2, None)
        .expect("correction");
    assert!(
        second.corrected_score < 50.0,
        "a repeat score should be corrected downward, got {}",
        second.corrected_score
    );
    assert_eq!(second.raw_score, 50.0);
    assert_eq!(second.assessment_number, 2);

    // An unknown measure has no effect on record to correct for.
    assert!(corrector.correct_score("unknown", 50.0, 2, None).is_err());
}

#[test]
fn test_practice_effect_expected_score_rises_then_settles() {
    let effect = PracticeEffect::new("trails_b", 5.0)
        .with_second_gain(2.0)
        .with_asymptote(8.0);

    let first = effect.expected_score(50.0, 1);
    let second = effect.expected_score(50.0, 2);
    let tenth = effect.expected_score(50.0, 10);

    assert!(
        second > first,
        "the second exposure should gain: {first} -> {second}"
    );
    assert!(
        tenth <= 50.0 + 8.0 + 1e-6,
        "gains must respect the asymptote, got {tenth}"
    );
}

#[test]
fn test_serial_assessment_accumulates_sessions() {
    let mut serial = SerialAssessment::new("P001");
    assert_eq!(serial.session_count(), 0);

    for n in 1..=3 {
        let mut session = AssessmentSession::new(n, &format!("visit-{n}"));
        session.add_score("trails_b", 50.0 - n as f64);
        serial.add_session(session);
    }

    assert_eq!(serial.session_count(), 3);
    assert_eq!(serial.sessions().len(), 3);
    assert!(serial.get_session(0).is_some());
    assert!(serial.get_session(99).is_none());
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
    let record = PatientRecord::new().with_dob(1930, 1, 1);

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
    let err3 = ClinicalError::InsufficientData {
        required: 10,
        available: 5,
    };

    assert!(format!("{:?}", err1).contains("alpha_power"));
    assert!(format!("{:?}", err2).contains("bad config"));
    assert!(format!("{:?}", err3).contains("5"));
}
