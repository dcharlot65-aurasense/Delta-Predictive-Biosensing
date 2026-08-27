//! Demographics data structures for normative stratification
//!
//! Provides comprehensive demographic variables for selecting appropriate
//! normative comparison groups.

use serde::{Deserialize, Serialize};

/// Biological sex for normative stratification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
}

impl Sex {
    /// Parse from string (case-insensitive)
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "male" | "m" => Some(Sex::Male),
            "female" | "f" => Some(Sex::Female),
            _ => None,
        }
    }
}

/// Ethnicity for normative stratification
/// Based on NIH categories
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ethnicity {
    /// American Indian or Alaska Native
    AmericanIndian,
    /// Asian
    Asian,
    /// Black or African American
    Black,
    /// Hispanic or Latino
    Hispanic,
    /// Native Hawaiian or Other Pacific Islander
    PacificIslander,
    /// White
    White,
    /// Two or more races
    Multiracial,
    /// Other or not specified
    Other,
}

impl Ethnicity {
    /// Get all ethnicity variants
    pub fn all() -> Vec<Self> {
        vec![
            Ethnicity::AmericanIndian,
            Ethnicity::Asian,
            Ethnicity::Black,
            Ethnicity::Hispanic,
            Ethnicity::PacificIslander,
            Ethnicity::White,
            Ethnicity::Multiracial,
            Ethnicity::Other,
        ]
    }
}

/// Handedness for motor assessments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Handedness {
    Right,
    Left,
    Ambidextrous,
}

impl Handedness {
    /// Parse from string
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "right" | "r" => Some(Handedness::Right),
            "left" | "l" => Some(Handedness::Left),
            "ambidextrous" | "both" | "a" => Some(Handedness::Ambidextrous),
            _ => None,
        }
    }

    /// Get dominant side
    pub fn dominant_side(&self) -> Option<Side> {
        match self {
            Handedness::Right => Some(Side::Right),
            Handedness::Left => Some(Side::Left),
            Handedness::Ambidextrous => None,
        }
    }
}

/// Body side (for lateralized assessments)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    Left,
    Right,
}

/// Education level for cognitive norms adjustment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EducationLevel {
    /// Less than high school
    LessThanHighSchool,
    /// High school diploma or GED
    HighSchool,
    /// Some college, no degree
    SomeCollege,
    /// Associate's degree
    Associates,
    /// Bachelor's degree
    Bachelors,
    /// Master's degree
    Masters,
    /// Doctoral or professional degree
    Doctorate,
}

impl EducationLevel {
    /// Convert to approximate years of education
    pub fn to_years(self) -> u8 {
        match self {
            EducationLevel::LessThanHighSchool => 10,
            EducationLevel::HighSchool => 12,
            EducationLevel::SomeCollege => 14,
            EducationLevel::Associates => 14,
            EducationLevel::Bachelors => 16,
            EducationLevel::Masters => 18,
            EducationLevel::Doctorate => 20,
        }
    }

    /// Create from years of education
    pub fn from_years(years: u8) -> Self {
        match years {
            0..=11 => EducationLevel::LessThanHighSchool,
            12 => EducationLevel::HighSchool,
            13..=14 => EducationLevel::SomeCollege,
            15..=16 => EducationLevel::Bachelors,
            17..=18 => EducationLevel::Masters,
            _ => EducationLevel::Doctorate,
        }
    }
}

/// Complete demographics profile for normative comparison
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographics {
    /// Age in years
    pub age: u8,
    /// Biological sex
    pub sex: Sex,
    /// Years of education (if known)
    pub education_years: Option<u8>,
    /// Education level (alternative to years)
    pub education_level: Option<EducationLevel>,
    /// Ethnicity (if relevant to norms)
    pub ethnicity: Option<Ethnicity>,
    /// Handedness (for motor assessments)
    pub handedness: Option<Handedness>,
    /// Height in cm (for some physical norms)
    pub height_cm: Option<f64>,
    /// Weight in kg (for some physical norms)
    pub weight_kg: Option<f64>,
}

impl Demographics {
    /// Create demographics with minimal required data
    pub fn new(age: u8, sex: Sex) -> Self {
        Self {
            age,
            sex,
            education_years: None,
            education_level: None,
            ethnicity: None,
            handedness: None,
            height_cm: None,
            weight_kg: None,
        }
    }

    /// Builder pattern: set education years
    pub fn with_education_years(mut self, years: u8) -> Self {
        self.education_years = Some(years);
        self.education_level = Some(EducationLevel::from_years(years));
        self
    }

    /// Builder pattern: set education level
    pub fn with_education_level(mut self, level: EducationLevel) -> Self {
        self.education_level = Some(level);
        self.education_years = Some(level.to_years());
        self
    }

    /// Builder pattern: set ethnicity
    pub fn with_ethnicity(mut self, ethnicity: Ethnicity) -> Self {
        self.ethnicity = Some(ethnicity);
        self
    }

    /// Builder pattern: set handedness
    pub fn with_handedness(mut self, handedness: Handedness) -> Self {
        self.handedness = Some(handedness);
        self
    }

    /// Builder pattern: set height
    pub fn with_height(mut self, height_cm: f64) -> Self {
        self.height_cm = Some(height_cm);
        self
    }

    /// Builder pattern: set weight
    pub fn with_weight(mut self, weight_kg: f64) -> Self {
        self.weight_kg = Some(weight_kg);
        self
    }

    /// Get BMI if height and weight are available
    pub fn bmi(&self) -> Option<f64> {
        match (self.height_cm, self.weight_kg) {
            (Some(h), Some(w)) => {
                let h_m = h / 100.0;
                Some(w / (h_m * h_m))
            }
            _ => None,
        }
    }

    /// Get age decade (for stratified norms)
    pub fn age_decade(&self) -> u8 {
        (self.age / 10) * 10
    }

    /// Get age group for normative lookup
    pub fn age_group(&self) -> AgeGroup {
        match self.age {
            0..=5 => AgeGroup::EarlyChildhood,
            6..=12 => AgeGroup::Childhood,
            13..=17 => AgeGroup::Adolescence,
            18..=29 => AgeGroup::YoungAdult,
            30..=44 => AgeGroup::MiddleAdult,
            45..=59 => AgeGroup::LateAdult,
            60..=74 => AgeGroup::YoungOld,
            75..=84 => AgeGroup::MiddleOld,
            _ => AgeGroup::OldestOld,
        }
    }

    /// Check if demographics match a given filter
    pub fn matches(&self, filter: &DemographicsFilter) -> bool {
        // Age range
        if let Some((min, max)) = filter.age_range
            && (self.age < min || self.age > max)
        {
            return false;
        }

        // Sex
        if let Some(sex) = filter.sex
            && self.sex != sex
        {
            return false;
        }

        // Education years
        if let Some((min, max)) = filter.education_range
            && let Some(years) = self.education_years
            && (years < min || years > max)
        {
            return false;
        }

        true
    }
}

impl Default for Demographics {
    fn default() -> Self {
        Self {
            age: 30,
            sex: Sex::Male,
            education_years: Some(12),
            education_level: Some(EducationLevel::HighSchool),
            ethnicity: None,
            handedness: Some(Handedness::Right),
            height_cm: None,
            weight_kg: None,
        }
    }
}

/// Age groups for normative stratification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeGroup {
    /// 0-5 years
    EarlyChildhood,
    /// 6-12 years
    Childhood,
    /// 13-17 years
    Adolescence,
    /// 18-29 years
    YoungAdult,
    /// 30-44 years
    MiddleAdult,
    /// 45-59 years
    LateAdult,
    /// 60-74 years
    YoungOld,
    /// 75-84 years
    MiddleOld,
    /// 85+ years
    OldestOld,
}

impl AgeGroup {
    /// Get age range for this group
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            AgeGroup::EarlyChildhood => (0, 5),
            AgeGroup::Childhood => (6, 12),
            AgeGroup::Adolescence => (13, 17),
            AgeGroup::YoungAdult => (18, 29),
            AgeGroup::MiddleAdult => (30, 44),
            AgeGroup::LateAdult => (45, 59),
            AgeGroup::YoungOld => (60, 74),
            AgeGroup::MiddleOld => (75, 84),
            AgeGroup::OldestOld => (85, 120),
        }
    }

    /// Get midpoint age for this group
    pub fn midpoint_age(&self) -> u8 {
        let (min, max) = self.age_range();
        (min + max) / 2
    }
}

/// Filter for selecting normative comparison group
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DemographicsFilter {
    /// Age range (inclusive)
    pub age_range: Option<(u8, u8)>,
    /// Specific sex
    pub sex: Option<Sex>,
    /// Education range (years)
    pub education_range: Option<(u8, u8)>,
    /// Specific ethnicity
    pub ethnicity: Option<Ethnicity>,
    /// Specific handedness
    pub handedness: Option<Handedness>,
}

impl DemographicsFilter {
    /// Create filter for specific age and sex
    pub fn age_sex(min_age: u8, max_age: u8, sex: Sex) -> Self {
        Self {
            age_range: Some((min_age, max_age)),
            sex: Some(sex),
            ..Default::default()
        }
    }

    /// Create filter matching a demographics profile
    pub fn from_demographics(demo: &Demographics, age_tolerance: u8) -> Self {
        let age_min = demo.age.saturating_sub(age_tolerance);
        let age_max = demo.age.saturating_add(age_tolerance);

        Self {
            age_range: Some((age_min, age_max)),
            sex: Some(demo.sex),
            education_range: demo
                .education_years
                .map(|y| (y.saturating_sub(2), y.saturating_add(2))),
            ethnicity: demo.ethnicity,
            handedness: demo.handedness,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demographics_creation() {
        let demo = Demographics::new(45, Sex::Female)
            .with_education_years(16)
            .with_handedness(Handedness::Right);

        assert_eq!(demo.age, 45);
        assert_eq!(demo.sex, Sex::Female);
        assert_eq!(demo.education_years, Some(16));
        assert_eq!(demo.education_level, Some(EducationLevel::Bachelors));
    }

    #[test]
    fn test_age_groups() {
        assert_eq!(
            Demographics::new(25, Sex::Male).age_group(),
            AgeGroup::YoungAdult
        );
        assert_eq!(
            Demographics::new(55, Sex::Female).age_group(),
            AgeGroup::LateAdult
        );
        assert_eq!(
            Demographics::new(70, Sex::Male).age_group(),
            AgeGroup::YoungOld
        );
    }

    #[test]
    fn test_bmi_calculation() {
        let demo = Demographics::new(30, Sex::Male)
            .with_height(180.0)
            .with_weight(80.0);

        let bmi = demo.bmi().unwrap();
        assert!((bmi - 24.69).abs() < 0.1);
    }

    #[test]
    fn test_demographics_filter() {
        let demo = Demographics::new(45, Sex::Female).with_education_years(16);

        let filter = DemographicsFilter::age_sex(40, 50, Sex::Female);
        assert!(demo.matches(&filter));

        let wrong_filter = DemographicsFilter::age_sex(20, 30, Sex::Female);
        assert!(!demo.matches(&wrong_filter));
    }

    #[test]
    fn test_education_conversion() {
        assert_eq!(EducationLevel::HighSchool.to_years(), 12);
        assert_eq!(EducationLevel::Bachelors.to_years(), 16);
        assert_eq!(EducationLevel::from_years(14), EducationLevel::SomeCollege);
    }
}
