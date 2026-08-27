//! # Protected Health Information (PHI) Utilities
//!
//! De-identification helpers implementing the HIPAA Safe Harbor method
//! (45 CFR 164.514(b)(2)).
//!
//! Using these helpers does not by itself establish HIPAA compliance — that is a
//! property of a covered entity's policies, practices and agreements, not of a
//! software library. Verify any de-identification against your own compliance
//! obligations before relying on it.
//!
//! ## HIPAA Safe Harbor Method
//!
//! This module implements the Safe Harbor method for de-identification,
//! which requires removal or generalization of 18 identifier types:
//!
//! 1. Names
//! 2. Geographic data (smaller than state)
//! 3. Dates (except year) related to an individual
//! 4. Phone numbers
//! 5. Fax numbers
//! 6. Email addresses
//! 7. Social Security numbers
//! 8. Medical record numbers
//! 9. Health plan beneficiary numbers
//! 10. Account numbers
//! 11. Certificate/license numbers
//! 12. Vehicle identifiers and serial numbers
//! 13. Device identifiers and serial numbers
//! 14. Web URLs
//! 15. IP addresses
//! 16. Biometric identifiers
//! 17. Full-face photographs
//! 18. Any other unique identifying number or code
//!
//! ## Example
//!
//! ```rust
//! use dpb_clinical::phi::{DeIdentificationConfig, DeIdentifier, PatientRecord};
//!
//! # fn example() -> dpb_clinical::Result<()> {
//! let config = DeIdentificationConfig::safe_harbor();
//! let deidentifier = DeIdentifier::new(config);
//!
//! let record = PatientRecord::new()
//!     .with_name("John Doe")
//!     .with_dob(1985, 3, 15)
//!     .with_ssn("123-45-6789")
//!     .with_clinical_value("alpha_power", 12.5);
//!
//! let deidentified = deidentifier.deidentify(&record)?;
//!
//! // The de-identified record carries no name or SSN field at all: direct
//! // identifiers are dropped rather than blanked.
//! assert_eq!(deidentified.birth_year, Some(1985)); // Year retained
//! assert!(deidentified.clinical_data.contains_key("alpha_power"));
//!
//! // Every transformation is recorded, so the de-identification is auditable.
//! assert!(!deidentified.audit_log.is_empty());
//! # Ok(())
//! # }
//! ```

use chrono::Datelike;
use crate::Result;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;

/// PHI identifier types per HIPAA Safe Harbor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhiIdentifier {
    /// Patient names
    Name,
    /// Geographic data smaller than state
    GeographicData,
    /// Dates except year (birth, admission, discharge, death)
    Dates,
    /// Telephone numbers
    PhoneNumber,
    /// Fax numbers
    FaxNumber,
    /// Email addresses
    EmailAddress,
    /// Social Security numbers
    SocialSecurityNumber,
    /// Medical record numbers
    MedicalRecordNumber,
    /// Health plan beneficiary numbers
    HealthPlanNumber,
    /// Account numbers
    AccountNumber,
    /// Certificate/license numbers
    CertificateNumber,
    /// Vehicle identifiers
    VehicleIdentifier,
    /// Device identifiers
    DeviceIdentifier,
    /// Web URLs
    WebUrl,
    /// IP addresses
    IpAddress,
    /// Biometric identifiers (fingerprints, voiceprints)
    BiometricIdentifier,
    /// Full-face photographs
    Photograph,
    /// Any other unique identifier
    OtherUniqueId,
}

impl PhiIdentifier {
    /// Returns all 18 HIPAA identifier types.
    pub fn all() -> Vec<PhiIdentifier> {
        vec![
            PhiIdentifier::Name,
            PhiIdentifier::GeographicData,
            PhiIdentifier::Dates,
            PhiIdentifier::PhoneNumber,
            PhiIdentifier::FaxNumber,
            PhiIdentifier::EmailAddress,
            PhiIdentifier::SocialSecurityNumber,
            PhiIdentifier::MedicalRecordNumber,
            PhiIdentifier::HealthPlanNumber,
            PhiIdentifier::AccountNumber,
            PhiIdentifier::CertificateNumber,
            PhiIdentifier::VehicleIdentifier,
            PhiIdentifier::DeviceIdentifier,
            PhiIdentifier::WebUrl,
            PhiIdentifier::IpAddress,
            PhiIdentifier::BiometricIdentifier,
            PhiIdentifier::Photograph,
            PhiIdentifier::OtherUniqueId,
        ]
    }
}

/// De-identification method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeIdentificationMethod {
    /// Complete removal of identifier
    Remove,
    /// Replace with pseudonymous identifier (reversible with key)
    Pseudonymize,
    /// Generalize to broader category
    Generalize,
    /// Replace with random value
    Randomize,
    /// Mask with fixed pattern (e.g., "***-**-1234")
    Mask,
    /// Hash with salt (one-way)
    Hash,
}

/// Configuration for de-identification process
#[derive(Debug, Clone)]
pub struct DeIdentificationConfig {
    /// Methods for each identifier type
    pub methods: HashMap<PhiIdentifier, DeIdentificationMethod>,
    /// Salt for hashing operations
    pub hash_salt: Option<String>,
    /// Whether to retain year in dates
    pub retain_date_year: bool,
    /// Whether to retain state in geographic data
    pub retain_state: bool,
    /// Age threshold for date generalization (ages over this are grouped)
    pub age_over_89_threshold: bool,
    /// Pseudonymization key (for reversible de-identification)
    pseudonym_key: Option<String>,
}

impl DeIdentificationConfig {
    /// Creates a new configuration with default settings.
    pub fn new() -> Self {
        Self {
            methods: HashMap::new(),
            hash_salt: None,
            retain_date_year: true,
            retain_state: true,
            age_over_89_threshold: true,
            pseudonym_key: None,
        }
    }

    /// Creates HIPAA Safe Harbor compliant configuration.
    ///
    /// This configuration removes all 18 identifier types per Safe Harbor guidelines.
    pub fn safe_harbor() -> Self {
        let mut methods = HashMap::new();
        for id in PhiIdentifier::all() {
            methods.insert(id, DeIdentificationMethod::Remove);
        }
        // Dates are the one Safe Harbor category that is generalized rather than
        // removed outright: 45 CFR 164.514(b)(2) strips date elements more
        // specific than the year but permits the year, and requires ages over 89
        // to be aggregated into a single "90+" category. Leaving this as Remove
        // (as the blanket loop above does) makes retain_date_year and
        // age_over_89_threshold below unreachable, because process_date only
        // consults them in the Generalize branch.
        methods.insert(PhiIdentifier::Dates, DeIdentificationMethod::Generalize);

        Self {
            methods,
            hash_salt: None,
            retain_date_year: true, // Year is allowed
            retain_state: true,     // State is allowed
            age_over_89_threshold: true, // Ages 90+ become "90+"
            pseudonym_key: None,
        }
    }

    /// Creates Limited Data Set configuration.
    ///
    /// Retains dates and geographic data at city/state level.
    pub fn limited_data_set() -> Self {
        let mut methods = HashMap::new();

        // Remove direct identifiers
        methods.insert(PhiIdentifier::Name, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::PhoneNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::FaxNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::EmailAddress, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::SocialSecurityNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::MedicalRecordNumber, DeIdentificationMethod::Pseudonymize);
        methods.insert(PhiIdentifier::HealthPlanNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::AccountNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::CertificateNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::VehicleIdentifier, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::DeviceIdentifier, DeIdentificationMethod::Pseudonymize);
        methods.insert(PhiIdentifier::WebUrl, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::IpAddress, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::BiometricIdentifier, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::Photograph, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::OtherUniqueId, DeIdentificationMethod::Hash);

        // Limited data set allows dates and geographic data
        // (handled separately, not in methods map)

        Self {
            methods,
            hash_salt: Some(generate_random_salt()),
            retain_date_year: true,
            retain_state: true,
            age_over_89_threshold: true,
            pseudonym_key: Some(generate_random_salt()),
        }
    }

    /// Creates research-grade pseudonymization configuration.
    ///
    /// Uses consistent pseudonyms for longitudinal research.
    pub fn research_pseudonymization() -> Self {
        let mut methods = HashMap::new();

        // Pseudonymize identifiers for linkage
        methods.insert(PhiIdentifier::Name, DeIdentificationMethod::Pseudonymize);
        methods.insert(PhiIdentifier::MedicalRecordNumber, DeIdentificationMethod::Pseudonymize);
        methods.insert(PhiIdentifier::DeviceIdentifier, DeIdentificationMethod::Pseudonymize);

        // Remove other direct identifiers
        methods.insert(PhiIdentifier::PhoneNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::FaxNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::EmailAddress, DeIdentificationMethod::Hash);
        methods.insert(PhiIdentifier::SocialSecurityNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::HealthPlanNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::AccountNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::CertificateNumber, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::VehicleIdentifier, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::WebUrl, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::IpAddress, DeIdentificationMethod::Mask);
        methods.insert(PhiIdentifier::BiometricIdentifier, DeIdentificationMethod::Hash);
        methods.insert(PhiIdentifier::Photograph, DeIdentificationMethod::Remove);
        methods.insert(PhiIdentifier::OtherUniqueId, DeIdentificationMethod::Hash);

        // Generalize geographic data
        methods.insert(PhiIdentifier::GeographicData, DeIdentificationMethod::Generalize);
        methods.insert(PhiIdentifier::Dates, DeIdentificationMethod::Generalize);

        Self {
            methods,
            hash_salt: Some(generate_random_salt()),
            retain_date_year: true,
            retain_state: true,
            age_over_89_threshold: true,
            pseudonym_key: Some(generate_random_salt()),
        }
    }

    /// Sets the hash salt for one-way hashing.
    pub fn with_hash_salt(mut self, salt: &str) -> Self {
        self.hash_salt = Some(salt.to_string());
        self
    }

    /// Sets the pseudonymization key.
    pub fn with_pseudonym_key(mut self, key: &str) -> Self {
        self.pseudonym_key = Some(key.to_string());
        self
    }

    /// Sets method for a specific identifier type.
    pub fn with_method(mut self, id: PhiIdentifier, method: DeIdentificationMethod) -> Self {
        self.methods.insert(id, method);
        self
    }
}

impl Default for DeIdentificationConfig {
    fn default() -> Self {
        Self::safe_harbor()
    }
}

/// Patient record with PHI fields
#[derive(Debug, Clone)]
pub struct PatientRecord {
    /// Patient identifier (will be de-identified)
    pub patient_id: Option<String>,
    /// Patient name
    pub name: Option<String>,
    /// Date of birth
    pub date_of_birth: Option<DateInfo>,
    /// Social Security Number
    pub ssn: Option<String>,
    /// Medical record number
    pub mrn: Option<String>,
    /// Phone number
    pub phone: Option<String>,
    /// Email address
    pub email: Option<String>,
    /// Geographic location
    pub location: Option<GeographicInfo>,
    /// Device identifier (e.g., biosensor serial)
    pub device_id: Option<String>,
    /// IP address
    pub ip_address: Option<String>,
    /// Additional custom identifiers
    pub custom_identifiers: HashMap<String, String>,
    /// Clinical data (non-PHI)
    pub clinical_data: HashMap<String, f64>,
}

impl PatientRecord {
    /// Creates a new empty patient record.
    pub fn new() -> Self {
        Self {
            patient_id: None,
            name: None,
            date_of_birth: None,
            ssn: None,
            mrn: None,
            phone: None,
            email: None,
            location: None,
            device_id: None,
            ip_address: None,
            custom_identifiers: HashMap::new(),
            clinical_data: HashMap::new(),
        }
    }

    /// Sets the patient name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    /// Sets the patient ID.
    pub fn with_patient_id(mut self, id: &str) -> Self {
        self.patient_id = Some(id.to_string());
        self
    }

    /// Sets the date of birth.
    pub fn with_dob(mut self, year: i32, month: u32, day: u32) -> Self {
        self.date_of_birth = Some(DateInfo { year, month, day });
        self
    }

    /// Sets the SSN.
    pub fn with_ssn(mut self, ssn: &str) -> Self {
        self.ssn = Some(ssn.to_string());
        self
    }

    /// Sets the medical record number.
    pub fn with_mrn(mut self, mrn: &str) -> Self {
        self.mrn = Some(mrn.to_string());
        self
    }

    /// Sets the phone number.
    pub fn with_phone(mut self, phone: &str) -> Self {
        self.phone = Some(phone.to_string());
        self
    }

    /// Sets the email address.
    pub fn with_email(mut self, email: &str) -> Self {
        self.email = Some(email.to_string());
        self
    }

    /// Sets the geographic location.
    pub fn with_location(mut self, street: &str, city: &str, state: &str, zip: &str) -> Self {
        self.location = Some(GeographicInfo {
            street_address: Some(street.to_string()),
            city: Some(city.to_string()),
            state: Some(state.to_string()),
            zip_code: Some(zip.to_string()),
            country: None,
        });
        self
    }

    /// Sets the device ID.
    pub fn with_device_id(mut self, device_id: &str) -> Self {
        self.device_id = Some(device_id.to_string());
        self
    }

    /// Adds clinical data.
    pub fn with_clinical_value(mut self, key: &str, value: f64) -> Self {
        self.clinical_data.insert(key.to_string(), value);
        self
    }
}

impl Default for PatientRecord {
    fn default() -> Self {
        Self::new()
    }
}

/// Date information
#[derive(Debug, Clone)]
pub struct DateInfo {
    pub year: i32,
    pub month: u32,
    pub day: u32,
}

/// Geographic information
#[derive(Debug, Clone)]
pub struct GeographicInfo {
    pub street_address: Option<String>,
    pub city: Option<String>,
    pub state: Option<String>,
    pub zip_code: Option<String>,
    pub country: Option<String>,
}

/// De-identified patient record
#[derive(Debug, Clone)]
pub struct DeIdentifiedRecord {
    /// Pseudonymous patient identifier (if pseudonymization used)
    pub pseudonym_id: Option<String>,
    /// Birth year only (if retained)
    pub birth_year: Option<i32>,
    /// Age group (if generalized)
    pub age_group: Option<String>,
    /// State only (if retained)
    pub state: Option<String>,
    /// Masked device ID (if masked)
    pub device_id_masked: Option<String>,
    /// Clinical data (non-PHI, preserved)
    pub clinical_data: HashMap<String, f64>,
    /// De-identification audit log
    pub audit_log: Vec<AuditEntry>,
}

/// Audit log entry for de-identification
#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub identifier_type: PhiIdentifier,
    pub method_applied: DeIdentificationMethod,
    pub timestamp: u64,
}

/// De-identification engine
#[derive(Debug, Clone)]
pub struct DeIdentifier {
    config: DeIdentificationConfig,
}

impl DeIdentifier {
    /// Creates a new de-identifier with the given configuration.
    pub fn new(config: DeIdentificationConfig) -> Self {
        Self { config }
    }

    /// Creates a de-identifier with Safe Harbor configuration.
    pub fn safe_harbor() -> Self {
        Self::new(DeIdentificationConfig::safe_harbor())
    }

    /// Creates a de-identifier with Limited Data Set configuration.
    pub fn limited_data_set() -> Self {
        Self::new(DeIdentificationConfig::limited_data_set())
    }

    /// De-identifies a patient record.
    pub fn deidentify(&self, record: &PatientRecord) -> Result<DeIdentifiedRecord> {
        let mut audit_log = Vec::new();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        // Process name
        if record.name.is_some() {
            let method = self.config.methods.get(&PhiIdentifier::Name)
                .copied()
                .unwrap_or(DeIdentificationMethod::Remove);
            audit_log.push(AuditEntry {
                identifier_type: PhiIdentifier::Name,
                method_applied: method,
                timestamp,
            });
        }

        // Process SSN
        if record.ssn.is_some() {
            let method = self.config.methods.get(&PhiIdentifier::SocialSecurityNumber)
                .copied()
                .unwrap_or(DeIdentificationMethod::Remove);
            audit_log.push(AuditEntry {
                identifier_type: PhiIdentifier::SocialSecurityNumber,
                method_applied: method,
                timestamp,
            });
        }

        // Generate pseudonym if configured
        let pseudonym_id = if let Some(ref key) = self.config.pseudonym_key {
            record.patient_id.as_ref().map(|id| {
                self.generate_pseudonym(id, key)
            })
        } else {
            None
        };

        // Process date of birth
        let (birth_year, age_group) = if let Some(ref dob) = record.date_of_birth {
            self.process_date(dob, &mut audit_log, timestamp)
        } else {
            (None, None)
        };

        // Process location
        let state = if let Some(ref loc) = record.location {
            self.process_location(loc, &mut audit_log, timestamp)
        } else {
            None
        };

        // Process device ID
        let device_id_masked = if let Some(ref device) = record.device_id {
            let method = self.config.methods.get(&PhiIdentifier::DeviceIdentifier)
                .copied()
                .unwrap_or(DeIdentificationMethod::Remove);
            audit_log.push(AuditEntry {
                identifier_type: PhiIdentifier::DeviceIdentifier,
                method_applied: method,
                timestamp,
            });
            match method {
                DeIdentificationMethod::Mask => Some(self.mask_identifier(device)),
                DeIdentificationMethod::Pseudonymize => {
                    self.config.pseudonym_key.as_ref()
                        .map(|key| self.generate_pseudonym(device, key))
                }
                DeIdentificationMethod::Hash => {
                    Some(self.hash_identifier(device))
                }
                _ => None,
            }
        } else {
            None
        };

        Ok(DeIdentifiedRecord {
            pseudonym_id,
            birth_year,
            age_group,
            state,
            device_id_masked,
            clinical_data: record.clinical_data.clone(),
            audit_log,
        })
    }

    /// Batch de-identifies multiple records.
    pub fn deidentify_batch(&self, records: &[PatientRecord]) -> Result<Vec<DeIdentifiedRecord>> {
        records.iter()
            .map(|r| self.deidentify(r))
            .collect()
    }

    /// Generates a consistent pseudonym from an identifier.
    fn generate_pseudonym(&self, id: &str, key: &str) -> String {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        id.hash(&mut hasher);
        format!("PSN-{:016X}", hasher.finish())
    }

    /// Hashes an identifier with salt.
    fn hash_identifier(&self, id: &str) -> String {
        let salt = self.config.hash_salt.as_deref().unwrap_or("dpb-default-salt");
        let mut hasher = DefaultHasher::new();
        salt.hash(&mut hasher);
        id.hash(&mut hasher);
        format!("{:016X}", hasher.finish())
    }

    /// Masks an identifier.
    fn mask_identifier(&self, id: &str) -> String {
        if id.len() <= 4 {
            "*".repeat(id.len())
        } else {
            format!("{}...{}", &id[..2], &id[id.len()-2..])
        }
    }

    /// Processes date of birth.
    fn process_date(
        &self,
        dob: &DateInfo,
        audit_log: &mut Vec<AuditEntry>,
        timestamp: u64,
    ) -> (Option<i32>, Option<String>) {
        let method = self.config.methods.get(&PhiIdentifier::Dates)
            .copied()
            .unwrap_or(DeIdentificationMethod::Generalize);

        audit_log.push(AuditEntry {
            identifier_type: PhiIdentifier::Dates,
            method_applied: method,
            timestamp,
        });

        // A hardcoded year silently mis-ages every record, and the error grows
        // every January. chrono is already a workspace dependency.
        let current_year = chrono::Utc::now().year();
        let age = current_year - dob.year;

        match method {
            DeIdentificationMethod::Remove => (None, None),
            DeIdentificationMethod::Generalize => {
                // Generalize to age group
                let age_group = if self.config.age_over_89_threshold && age > 89 {
                    "90+".to_string()
                } else if age < 18 {
                    format!("{}-{}", (age / 5) * 5, (age / 5) * 5 + 4)
                } else {
                    format!("{}-{}", (age / 10) * 10, (age / 10) * 10 + 9)
                };

                let year = if self.config.retain_date_year {
                    Some(dob.year)
                } else {
                    None
                };

                (year, Some(age_group))
            }
            _ => {
                if self.config.retain_date_year {
                    (Some(dob.year), None)
                } else {
                    (None, None)
                }
            }
        }
    }

    /// Processes location.
    fn process_location(
        &self,
        loc: &GeographicInfo,
        audit_log: &mut Vec<AuditEntry>,
        timestamp: u64,
    ) -> Option<String> {
        let method = self.config.methods.get(&PhiIdentifier::GeographicData)
            .copied()
            .unwrap_or(DeIdentificationMethod::Generalize);

        audit_log.push(AuditEntry {
            identifier_type: PhiIdentifier::GeographicData,
            method_applied: method,
            timestamp,
        });

        match method {
            DeIdentificationMethod::Remove => None,
            DeIdentificationMethod::Generalize if self.config.retain_state => {
                loc.state.clone()
            }
            _ => None,
        }
    }
}

/// Data anonymization statistics
#[derive(Debug, Clone, Default)]
pub struct AnonymizationStats {
    pub records_processed: usize,
    pub identifiers_removed: usize,
    pub identifiers_pseudonymized: usize,
    pub identifiers_generalized: usize,
    pub identifiers_hashed: usize,
    pub identifiers_masked: usize,
}

impl AnonymizationStats {
    /// Updates stats from a de-identified record.
    pub fn update(&mut self, record: &DeIdentifiedRecord) {
        self.records_processed += 1;
        for entry in &record.audit_log {
            match entry.method_applied {
                DeIdentificationMethod::Remove => self.identifiers_removed += 1,
                DeIdentificationMethod::Pseudonymize => self.identifiers_pseudonymized += 1,
                DeIdentificationMethod::Generalize => self.identifiers_generalized += 1,
                DeIdentificationMethod::Hash => self.identifiers_hashed += 1,
                DeIdentificationMethod::Mask => self.identifiers_masked += 1,
                DeIdentificationMethod::Randomize => self.identifiers_removed += 1,
            }
        }
    }
}

/// K-Anonymity checker
#[derive(Debug, Clone)]
pub struct KAnonymityChecker {
    /// Minimum group size for k-anonymity
    k: usize,
    /// Quasi-identifiers to check
    quasi_identifiers: Vec<String>,
}

impl KAnonymityChecker {
    /// Creates a new k-anonymity checker.
    pub fn new(k: usize) -> Self {
        Self {
            k,
            quasi_identifiers: vec![
                "age_group".to_string(),
                "state".to_string(),
                "birth_year".to_string(),
            ],
        }
    }

    /// Checks if a dataset satisfies k-anonymity.
    pub fn check(&self, records: &[DeIdentifiedRecord]) -> KAnonymityResult {
        let mut equivalence_classes: HashMap<String, usize> = HashMap::new();

        for record in records {
            let key = self.create_equivalence_key(record);
            *equivalence_classes.entry(key).or_insert(0) += 1;
        }

        let violations: Vec<(String, usize)> = equivalence_classes
            .iter()
            .filter(|&(_, &count)| count < self.k)
            .map(|(key, &count)| (key.clone(), count))
            .collect();

        let min_group_size = equivalence_classes.values().copied().min().unwrap_or(0);
        let satisfies_k = violations.is_empty();

        KAnonymityResult {
            k: self.k,
            satisfies_k,
            min_group_size,
            num_equivalence_classes: equivalence_classes.len(),
            violations,
        }
    }

    /// Creates an equivalence class key from quasi-identifiers.
    fn create_equivalence_key(&self, record: &DeIdentifiedRecord) -> String {
        let mut parts = Vec::new();

        if let Some(ref age_group) = record.age_group {
            parts.push(age_group.clone());
        }
        if let Some(ref state) = record.state {
            parts.push(state.clone());
        }
        if let Some(year) = record.birth_year {
            parts.push(year.to_string());
        }

        parts.join("|")
    }

    /// Sets custom quasi-identifiers.
    pub fn with_quasi_identifiers(mut self, ids: Vec<String>) -> Self {
        self.quasi_identifiers = ids;
        self
    }
}

/// Result of k-anonymity check
#[derive(Debug, Clone)]
pub struct KAnonymityResult {
    /// The k value checked
    pub k: usize,
    /// Whether dataset satisfies k-anonymity
    pub satisfies_k: bool,
    /// Minimum group size found
    pub min_group_size: usize,
    /// Number of equivalence classes
    pub num_equivalence_classes: usize,
    /// Violations (groups with fewer than k records)
    pub violations: Vec<(String, usize)>,
}

/// L-Diversity checker for sensitive attributes
#[derive(Debug, Clone)]
pub struct LDiversityChecker {
    /// Minimum diversity for l-diversity
    l: usize,
    /// Sensitive attribute to check
    sensitive_attribute: String,
}

impl LDiversityChecker {
    /// Creates a new l-diversity checker.
    pub fn new(l: usize, sensitive_attribute: &str) -> Self {
        Self {
            l,
            sensitive_attribute: sensitive_attribute.to_string(),
        }
    }

    /// Checks if a dataset satisfies l-diversity for the sensitive attribute.
    pub fn check(&self, records: &[DeIdentifiedRecord]) -> LDiversityResult {
        let mut equivalence_classes: HashMap<String, Vec<f64>> = HashMap::new();

        for record in records {
            let key = format!(
                "{}|{}",
                record.age_group.as_deref().unwrap_or(""),
                record.state.as_deref().unwrap_or("")
            );

            if let Some(&value) = record.clinical_data.get(&self.sensitive_attribute) {
                equivalence_classes
                    .entry(key)
                    .or_default()
                    .push(value);
            }
        }

        let mut min_diversity = usize::MAX;
        let mut violations = Vec::new();

        for (key, values) in &equivalence_classes {
            let unique_count = values.iter()
                .map(|v| (*v * 1000.0) as i64)
                .collect::<std::collections::HashSet<_>>()
                .len();

            if unique_count < self.l {
                violations.push((key.clone(), unique_count));
            }
            min_diversity = min_diversity.min(unique_count);
        }

        LDiversityResult {
            l: self.l,
            satisfies_l: violations.is_empty(),
            min_diversity,
            violations,
        }
    }
}

/// Result of l-diversity check
#[derive(Debug, Clone)]
pub struct LDiversityResult {
    pub l: usize,
    pub satisfies_l: bool,
    pub min_diversity: usize,
    pub violations: Vec<(String, usize)>,
}

/// Generate a random salt for hashing
fn generate_random_salt() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:032X}", nanos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_harbor_config() {
        let config = DeIdentificationConfig::safe_harbor();
        assert_eq!(config.methods.len(), 18);
        assert!(config.retain_date_year);
        assert!(config.retain_state);
    }

    #[test]
    fn test_deidentify_record() {
        let deidentifier = DeIdentifier::safe_harbor();

        let record = PatientRecord::new()
            .with_name("John Doe")
            .with_ssn("123-45-6789")
            .with_dob(1985, 3, 15)
            .with_clinical_value("heart_rate", 72.0);

        let result = deidentifier.deidentify(&record).unwrap();

        // Name and SSN should be removed
        assert!(result.pseudonym_id.is_none());

        // Year should be retained
        assert_eq!(result.birth_year, Some(1985));

        // Clinical data preserved
        assert_eq!(result.clinical_data.get("heart_rate"), Some(&72.0));
    }

    #[test]
    fn test_pseudonymization() {
        let config = DeIdentificationConfig::research_pseudonymization();
        let deidentifier = DeIdentifier::new(config);

        let record1 = PatientRecord::new()
            .with_patient_id("PAT001")
            .with_name("Jane Smith");

        let record2 = PatientRecord::new()
            .with_patient_id("PAT001") // Same ID
            .with_name("Jane Smith");

        let result1 = deidentifier.deidentify(&record1).unwrap();
        let result2 = deidentifier.deidentify(&record2).unwrap();

        // Same patient ID should generate same pseudonym
        assert_eq!(result1.pseudonym_id, result2.pseudonym_id);
    }

    #[test]
    fn test_k_anonymity_check() {
        let checker = KAnonymityChecker::new(2);

        let records = vec![
            DeIdentifiedRecord {
                pseudonym_id: None,
                birth_year: Some(1985),
                age_group: Some("40-49".to_string()),
                state: Some("CA".to_string()),
                device_id_masked: None,
                clinical_data: HashMap::new(),
                audit_log: vec![],
            },
            DeIdentifiedRecord {
                pseudonym_id: None,
                birth_year: Some(1985),
                age_group: Some("40-49".to_string()),
                state: Some("CA".to_string()),
                device_id_masked: None,
                clinical_data: HashMap::new(),
                audit_log: vec![],
            },
        ];

        let result = checker.check(&records);
        assert!(result.satisfies_k);
        assert_eq!(result.min_group_size, 2);
    }

    #[test]
    fn test_phi_identifiers_count() {
        let all = PhiIdentifier::all();
        assert_eq!(all.len(), 18); // HIPAA requires 18 identifiers
    }

    #[test]
    fn test_mask_identifier() {
        let deidentifier = DeIdentifier::safe_harbor();
        assert_eq!(deidentifier.mask_identifier("12345678"), "12...78");
        assert_eq!(deidentifier.mask_identifier("AB"), "**");
    }

    #[test]
    fn test_age_over_89_grouping() {
        let config = DeIdentificationConfig::safe_harbor();
        let deidentifier = DeIdentifier::new(config);

        // 95-year-old should be grouped as "90+"
        let record = PatientRecord::new()
            .with_dob(1930, 1, 1); // ~95 years old in 2025

        let result = deidentifier.deidentify(&record).unwrap();
        assert_eq!(result.age_group, Some("90+".to_string()));
    }
}
