//! Virtual Patient Cohort Generation
//!
//! Provides tools for generating realistic virtual patient cohorts with
//! demographic characteristics, baseline physiological parameters, and
//! clinical conditions for synthetic data studies.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Bernoulli};
use std::collections::HashMap;

/// Biological sex
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sex {
    Male,
    Female,
    Other,
}

impl Sex {
    /// Convert to string representation
    pub fn as_str(&self) -> &str {
        match self {
            Sex::Male => "Male",
            Sex::Female => "Female",
            Sex::Other => "Other",
        }
    }
}

/// Demographic information for a virtual patient
#[derive(Debug, Clone)]
pub struct Demographics {
    /// Age in years
    pub age_years: f64,
    /// Biological sex
    pub sex: Sex,
    /// Body mass index (kg/m²)
    pub bmi: f64,
    /// Ethnicity (optional)
    pub ethnicity: Option<String>,
}

impl Demographics {
    /// Create new demographics with validation
    pub fn new(age_years: f64, sex: Sex, bmi: f64, ethnicity: Option<String>) -> Self {
        assert!((0.0..=120.0).contains(&age_years), "Age must be in [0, 120]");
        assert!(bmi > 0.0 && bmi < 100.0, "BMI must be in (0, 100)");

        Self {
            age_years,
            sex,
            bmi,
            ethnicity,
        }
    }
}

/// A virtual patient with demographic and physiological characteristics
#[derive(Debug, Clone)]
pub struct VirtualPatient {
    /// Unique patient identifier
    pub id: String,
    /// Demographic information
    pub demographics: Demographics,
    /// Baseline heart rate (bpm)
    pub baseline_hr: f64,
    /// Baseline heart rate variability (RMSSD, ms)
    pub baseline_hrv: f64,
    /// List of medical conditions
    pub conditions: Vec<String>,
    /// List of medications
    pub medications: Vec<String>,
}

impl VirtualPatient {
    /// Create a new virtual patient
    pub fn new(
        id: String,
        demographics: Demographics,
        baseline_hr: f64,
        baseline_hrv: f64,
        conditions: Vec<String>,
        medications: Vec<String>,
    ) -> Self {
        assert!(baseline_hr > 0.0 && baseline_hr < 300.0, "HR must be in (0, 300)");
        assert!(baseline_hrv >= 0.0, "HRV must be non-negative");

        Self {
            id,
            demographics,
            baseline_hr,
            baseline_hrv,
            conditions,
            medications,
        }
    }

    /// Check if patient has a specific condition
    pub fn has_condition(&self, condition: &str) -> bool {
        self.conditions.iter().any(|c| c == condition)
    }

    /// Check if patient takes a specific medication
    pub fn takes_medication(&self, medication: &str) -> bool {
        self.medications.iter().any(|m| m == medication)
    }

    /// Get patient age category
    pub fn age_category(&self) -> &str {
        let age = self.demographics.age_years;
        if age < 18.0 {
            "Child"
        } else if age < 40.0 {
            "Young Adult"
        } else if age < 65.0 {
            "Middle Aged"
        } else {
            "Senior"
        }
    }

    /// Get BMI category
    pub fn bmi_category(&self) -> &str {
        let bmi = self.demographics.bmi;
        if bmi < 18.5 {
            "Underweight"
        } else if bmi < 25.0 {
            "Normal"
        } else if bmi < 30.0 {
            "Overweight"
        } else {
            "Obese"
        }
    }
}

/// Generator for creating virtual patient cohorts
pub struct CohortGenerator {
    /// Number of subjects to generate
    n_subjects: usize,
    /// Age distribution (mean, std dev) in years
    age_distribution: (f64, f64),
    /// Proportion of males (0.0 to 1.0)
    sex_ratio: f64,
    /// Disease prevalence rates (condition name -> prevalence)
    disease_prevalence: HashMap<String, f64>,
}

impl CohortGenerator {
    /// Create a new cohort generator with default parameters
    pub fn new(n_subjects: usize) -> Self {
        Self {
            n_subjects,
            age_distribution: (45.0, 18.0),  // Mean age 45, std 18
            sex_ratio: 0.5,  // 50% male
            disease_prevalence: HashMap::new(),
        }
    }

    /// Set the age distribution
    pub fn with_age_distribution(mut self, mean: f64, std: f64) -> Self {
        assert!(mean > 0.0 && std > 0.0);
        self.age_distribution = (mean, std);
        self
    }

    /// Set the sex ratio (proportion male)
    pub fn with_sex_ratio(mut self, ratio: f64) -> Self {
        assert!((0.0..=1.0).contains(&ratio));
        self.sex_ratio = ratio;
        self
    }

    /// Add a disease with a given prevalence rate
    pub fn with_disease(mut self, disease: &str, prevalence: f64) -> Self {
        assert!((0.0..=1.0).contains(&prevalence));
        self.disease_prevalence.insert(disease.to_string(), prevalence);
        self
    }

    /// Generate a cohort of virtual patients
    pub fn generate(&self, seed: u64) -> Vec<VirtualPatient> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut patients = Vec::with_capacity(self.n_subjects);

        for i in 0..self.n_subjects {
            patients.push(self.generate_patient(&mut rng, i));
        }

        patients
    }

    fn generate_patient(&self, rng: &mut impl Rng, id: usize) -> VirtualPatient {
        // Generate demographics
        let age = self.sample_age(rng);
        let sex = self.sample_sex(rng);
        let bmi = self.sample_bmi(rng, age, sex);
        let ethnicity = self.sample_ethnicity(rng);

        let demographics = Demographics::new(age, sex, bmi, ethnicity);

        // Generate baseline physiological parameters
        let baseline_hr = self.sample_baseline_hr(rng, age, sex);
        let baseline_hrv = self.sample_baseline_hrv(rng, age, baseline_hr);

        // Generate conditions and medications
        let conditions = self.sample_conditions(rng, age, bmi);
        let medications = self.sample_medications(rng, &conditions);

        VirtualPatient::new(
            format!("VP{:06}", id),
            demographics,
            baseline_hr,
            baseline_hrv,
            conditions,
            medications,
        )
    }

    fn sample_age(&self, rng: &mut impl Rng) -> f64 {
        let normal = Normal::new(self.age_distribution.0, self.age_distribution.1).unwrap();
        normal.sample(rng).max(18.0).min(100.0)  // Clamp to adult range
    }

    fn sample_sex(&self, rng: &mut impl Rng) -> Sex {
        if ((rng.next_u64() as f64) / (u64::MAX as f64)) < self.sex_ratio {
            Sex::Male
        } else {
            Sex::Female
        }
    }

    fn sample_bmi(&self, rng: &mut impl Rng, _age: f64, _sex: Sex) -> f64 {
        // Simplified BMI distribution (mean ~25, std ~5)
        let normal = Normal::new(25.0, 5.0).unwrap();
        let bmi: f64 = normal.sample(rng);
        bmi.max(15.0).min(50.0)
    }

    fn sample_ethnicity(&self, rng: &mut impl Rng) -> Option<String> {
        // Simplified ethnicity distribution
        let ethnicities = ["Caucasian",
            "African American",
            "Hispanic",
            "Asian",
            "Other"];
        let weights = [0.6, 0.13, 0.18, 0.06, 0.03];

        let r = (rng.next_u64() as f64) / (u64::MAX as f64);
        let mut cumulative = 0.0;

        for (eth, weight) in ethnicities.iter().zip(weights.iter()) {
            cumulative += weight;
            if r < cumulative {
                return Some(eth.to_string());
            }
        }

        Some("Other".to_string())
    }

    fn sample_baseline_hr(&self, rng: &mut impl Rng, age: f64, sex: Sex) -> f64 {
        // Baseline HR decreases slightly with age
        // Females typically have slightly higher HR
        let age_effect = -0.1 * (age - 40.0);
        let sex_offset = if matches!(sex, Sex::Female) { 3.0 } else { 0.0 };

        let mean_hr = 70.0 + age_effect + sex_offset;
        let normal = Normal::new(mean_hr, 10.0).unwrap();

        normal.sample(rng).max(50.0).min(100.0)
    }

    fn sample_baseline_hrv(&self, rng: &mut impl Rng, age: f64, baseline_hr: f64) -> f64 {
        // HRV decreases with age and is inversely related to HR
        let age_factor = (-0.01 * age).exp();
        let hr_factor = 1.0 / (baseline_hr / 70.0);

        let mean_hrv = 40.0 * age_factor * hr_factor;
        let normal = Normal::new(mean_hrv, 15.0).unwrap();

        normal.sample(rng).max(10.0).min(100.0)
    }

    fn sample_conditions(&self, rng: &mut impl Rng, age: f64, bmi: f64) -> Vec<String> {
        let mut conditions = Vec::new();

        // Sample from configured disease prevalence
        for (disease, prevalence) in &self.disease_prevalence {
            // Adjust prevalence based on age and BMI for some conditions
            let adjusted_prevalence = if disease == "Hypertension" {
                prevalence * (1.0 + (age - 40.0) / 100.0).max(0.5).min(2.0)
            } else if disease == "Diabetes" {
                prevalence * (1.0 + (bmi - 25.0) / 50.0).max(0.5).min(3.0)
            } else {
                *prevalence
            };

            let bernoulli = Bernoulli::new(adjusted_prevalence.min(1.0)).unwrap();
            if bernoulli.sample(rng) {
                conditions.push(disease.clone());
            }
        }

        conditions
    }

    fn sample_medications(&self, rng: &mut impl Rng, conditions: &[String]) -> Vec<String> {
        let mut medications = Vec::new();

        // Map conditions to potential medications
        for condition in conditions {
            let meds = match condition.as_str() {
                "Hypertension" => vec!["ACE Inhibitor", "Beta Blocker", "Diuretic"],
                "Diabetes" => vec!["Metformin", "Insulin"],
                "Depression" => vec!["SSRI", "SNRI"],
                "Anxiety" => vec!["Benzodiazepine", "SSRI"],
                _ => vec![],
            };

            // Sample 1-2 medications per condition
            for med in meds {
                if ((rng.next_u64() as f64) / (u64::MAX as f64)) < 0.6 {  // 60% chance of taking each medication
                    if !medications.contains(&med.to_string()) {
                        medications.push(med.to_string());
                    }
                }
            }
        }

        medications
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demographics_creation() {
        let demo = Demographics::new(45.0, Sex::Male, 25.0, Some("Caucasian".to_string()));
        assert_eq!(demo.age_years, 45.0);
        assert_eq!(demo.sex, Sex::Male);
        assert_eq!(demo.bmi, 25.0);
        assert_eq!(demo.ethnicity, Some("Caucasian".to_string()));
    }

    #[test]
    #[should_panic]
    fn test_invalid_age() {
        Demographics::new(150.0, Sex::Male, 25.0, None);
    }

    #[test]
    #[should_panic]
    fn test_invalid_bmi() {
        Demographics::new(45.0, Sex::Male, -5.0, None);
    }

    #[test]
    fn test_virtual_patient_creation() {
        let demo = Demographics::new(45.0, Sex::Male, 25.0, None);
        let patient = VirtualPatient::new(
            "VP000001".to_string(),
            demo,
            70.0,
            40.0,
            vec!["Hypertension".to_string()],
            vec!["ACE Inhibitor".to_string()],
        );

        assert_eq!(patient.id, "VP000001");
        assert_eq!(patient.baseline_hr, 70.0);
        assert!(patient.has_condition("Hypertension"));
        assert!(patient.takes_medication("ACE Inhibitor"));
    }

    #[test]
    fn test_age_categories() {
        let demo1 = Demographics::new(15.0, Sex::Male, 20.0, None);
        let patient1 = VirtualPatient::new("VP1".to_string(), demo1, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient1.age_category(), "Child");

        let demo2 = Demographics::new(25.0, Sex::Male, 23.0, None);
        let patient2 = VirtualPatient::new("VP2".to_string(), demo2, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient2.age_category(), "Young Adult");

        let demo3 = Demographics::new(50.0, Sex::Male, 27.0, None);
        let patient3 = VirtualPatient::new("VP3".to_string(), demo3, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient3.age_category(), "Middle Aged");

        let demo4 = Demographics::new(70.0, Sex::Male, 26.0, None);
        let patient4 = VirtualPatient::new("VP4".to_string(), demo4, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient4.age_category(), "Senior");
    }

    #[test]
    fn test_bmi_categories() {
        let demo1 = Demographics::new(30.0, Sex::Male, 17.0, None);
        let patient1 = VirtualPatient::new("VP1".to_string(), demo1, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient1.bmi_category(), "Underweight");

        let demo2 = Demographics::new(30.0, Sex::Male, 22.0, None);
        let patient2 = VirtualPatient::new("VP2".to_string(), demo2, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient2.bmi_category(), "Normal");

        let demo3 = Demographics::new(30.0, Sex::Male, 27.0, None);
        let patient3 = VirtualPatient::new("VP3".to_string(), demo3, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient3.bmi_category(), "Overweight");

        let demo4 = Demographics::new(30.0, Sex::Male, 32.0, None);
        let patient4 = VirtualPatient::new("VP4".to_string(), demo4, 70.0, 40.0, vec![], vec![]);
        assert_eq!(patient4.bmi_category(), "Obese");
    }

    #[test]
    fn test_cohort_generator_basic() {
        let generator = CohortGenerator::new(10);
        let cohort = generator.generate(42);

        assert_eq!(cohort.len(), 10);

        // Check that all patients have valid IDs
        for (i, patient) in cohort.iter().enumerate() {
            assert_eq!(patient.id, format!("VP{:06}", i));
        }
    }

    #[test]
    fn test_cohort_generator_with_disease() {
        let generator = CohortGenerator::new(100)
            .with_disease("Hypertension", 0.3)
            .with_disease("Diabetes", 0.1);

        let cohort = generator.generate(42);

        // Count patients with conditions
        let hypertension_count = cohort.iter()
            .filter(|p| p.has_condition("Hypertension"))
            .count();

        let diabetes_count = cohort.iter()
            .filter(|p| p.has_condition("Diabetes"))
            .count();

        // Should be approximately 30% and 10% (allow for randomness)
        assert!(hypertension_count > 10 && hypertension_count < 50);
        assert!(diabetes_count < 30);
    }

    #[test]
    fn test_cohort_generator_sex_ratio() {
        let generator = CohortGenerator::new(100)
            .with_sex_ratio(0.7);  // 70% male

        let cohort = generator.generate(42);

        let male_count = cohort.iter()
            .filter(|p| p.demographics.sex == Sex::Male)
            .count();

        // Should be approximately 70% (allow for randomness)
        assert!(male_count > 50 && male_count < 90);
    }

    #[test]
    fn test_cohort_generator_age_distribution() {
        let generator = CohortGenerator::new(100)
            .with_age_distribution(60.0, 10.0);

        let cohort = generator.generate(42);

        let mean_age = cohort.iter()
            .map(|p| p.demographics.age_years)
            .sum::<f64>() / cohort.len() as f64;

        // Mean should be close to 60
        assert!((mean_age - 60.0).abs() < 10.0);
    }

    #[test]
    fn test_reproducibility() {
        let generator = CohortGenerator::new(10);

        let cohort1 = generator.generate(42);
        let cohort2 = generator.generate(42);

        // Same seed should produce same cohort
        for (p1, p2) in cohort1.iter().zip(cohort2.iter()) {
            assert_eq!(p1.id, p2.id);
            assert!((p1.demographics.age_years - p2.demographics.age_years).abs() < 1e-10);
            assert!((p1.baseline_hr - p2.baseline_hr).abs() < 1e-10);
        }
    }

    #[test]
    fn test_sex_as_str() {
        assert_eq!(Sex::Male.as_str(), "Male");
        assert_eq!(Sex::Female.as_str(), "Female");
        assert_eq!(Sex::Other.as_str(), "Other");
    }
}
