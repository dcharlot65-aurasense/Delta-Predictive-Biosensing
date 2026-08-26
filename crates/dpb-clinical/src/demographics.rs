//! Patient demographics and population stratification.

use serde::{Deserialize, Serialize};
use crate::{ClinicalError, Result};

/// Patient demographics for normative comparisons.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Demographics {
    /// Age in years.
    pub age: Option<u8>,
    /// Biological sex.
    pub sex: Option<Sex>,
    /// Self-reported ethnicity.
    pub ethnicity: Option<Ethnicity>,
    /// Education level in years.
    pub education_years: Option<u8>,
    /// Handedness.
    pub handedness: Option<Handedness>,
    /// Additional demographic factors.
    pub factors: Vec<DemographicFactor>,
}

impl Demographics {
    /// Create new empty demographics.
    pub fn new() -> Self {
        Self {
            age: None,
            sex: None,
            ethnicity: None,
            education_years: None,
            handedness: None,
            factors: Vec::new(),
        }
    }

    /// Set age.
    pub fn with_age(mut self, age: u8) -> Self {
        self.age = Some(age);
        self
    }

    /// Set sex.
    pub fn with_sex(mut self, sex: Sex) -> Self {
        self.sex = Some(sex);
        self
    }

    /// Set ethnicity.
    pub fn with_ethnicity(mut self, ethnicity: Ethnicity) -> Self {
        self.ethnicity = Some(ethnicity);
        self
    }

    /// Set education years.
    pub fn with_education(mut self, years: u8) -> Self {
        self.education_years = Some(years);
        self
    }

    /// Set handedness.
    pub fn with_handedness(mut self, handedness: Handedness) -> Self {
        self.handedness = Some(handedness);
        self
    }

    /// Add a demographic factor.
    pub fn with_factor(mut self, factor: DemographicFactor) -> Self {
        self.factors.push(factor);
        self
    }

    /// Get age group.
    pub fn age_group(&self) -> Option<AgeGroup> {
        self.age.map(AgeGroup::from_age)
    }

    /// Validate demographics for clinical use.
    pub fn validate(&self) -> Result<()> {
        if let Some(age) = self.age
            && age > 120 {
                return Err(ClinicalError::InvalidDemographics(format!(
                    "Age {} is unrealistic",
                    age
                )));
            }

        if let Some(edu) = self.education_years
            && edu > 30 {
                return Err(ClinicalError::InvalidDemographics(format!(
                    "Education {} years is unrealistic",
                    edu
                )));
            }

        Ok(())
    }

    /// Get population key for normative lookup.
    pub fn population_key(&self) -> String {
        let age_str = self
            .age_group()
            .map(|g| format!("{:?}", g))
            .unwrap_or_else(|| "Unknown".to_string());

        let sex_str = self
            .sex
            .map(|s| format!("{:?}", s))
            .unwrap_or_else(|| "Unknown".to_string());

        let eth_str = self
            .ethnicity
            .map(|e| format!("{:?}", e))
            .unwrap_or_else(|| "Unknown".to_string());

        format!("{}_{}_{}", age_str, sex_str, eth_str)
    }
}

impl Default for Demographics {
    fn default() -> Self {
        Self::new()
    }
}

/// Biological sex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Sex {
    Male,
    Female,
    Intersex,
    Unknown,
}

/// Self-reported ethnicity categories.
///
/// Based on commonly used research categories. These are social constructs
/// used for population stratification, not biological categories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Ethnicity {
    /// European ancestry.
    European,
    /// East Asian ancestry.
    EastAsian,
    /// South Asian ancestry.
    SouthAsian,
    /// African ancestry.
    African,
    /// Hispanic/Latino ancestry.
    Hispanic,
    /// Middle Eastern/North African ancestry.
    MiddleEastern,
    /// Native American/Indigenous ancestry.
    Indigenous,
    /// Pacific Islander ancestry.
    PacificIslander,
    /// Mixed/Multiple ancestries.
    Mixed,
    /// Other/Not listed.
    Other,
    /// Unknown/Not reported.
    Unknown,
}

impl Ethnicity {
    /// Get broader category for statistical grouping.
    pub fn broad_category(&self) -> BroadEthnicCategory {
        match self {
            Ethnicity::European => BroadEthnicCategory::European,
            Ethnicity::EastAsian | Ethnicity::SouthAsian => BroadEthnicCategory::Asian,
            Ethnicity::African => BroadEthnicCategory::African,
            Ethnicity::Hispanic => BroadEthnicCategory::Hispanic,
            Ethnicity::MiddleEastern => BroadEthnicCategory::MiddleEastern,
            Ethnicity::Indigenous | Ethnicity::PacificIslander => BroadEthnicCategory::Indigenous,
            Ethnicity::Mixed | Ethnicity::Other | Ethnicity::Unknown => BroadEthnicCategory::Other,
        }
    }
}

/// Broad ethnic categories for statistical analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BroadEthnicCategory {
    European,
    Asian,
    African,
    Hispanic,
    MiddleEastern,
    Indigenous,
    Other,
}

/// Age groups for normative stratification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AgeGroup {
    /// Pediatric (0-11)
    Pediatric,
    /// Adolescent (12-17)
    Adolescent,
    /// Young Adult (18-29)
    YoungAdult,
    /// Adult (30-49)
    Adult,
    /// Middle Age (50-64)
    MiddleAge,
    /// Senior (65-79)
    Senior,
    /// Elderly (80+)
    Elderly,
}

impl AgeGroup {
    /// Get age group from numeric age.
    pub fn from_age(age: u8) -> Self {
        match age {
            0..=11 => AgeGroup::Pediatric,
            12..=17 => AgeGroup::Adolescent,
            18..=29 => AgeGroup::YoungAdult,
            30..=49 => AgeGroup::Adult,
            50..=64 => AgeGroup::MiddleAge,
            65..=79 => AgeGroup::Senior,
            _ => AgeGroup::Elderly,
        }
    }

    /// Get age range for this group.
    pub fn age_range(&self) -> (u8, u8) {
        match self {
            AgeGroup::Pediatric => (0, 11),
            AgeGroup::Adolescent => (12, 17),
            AgeGroup::YoungAdult => (18, 29),
            AgeGroup::Adult => (30, 49),
            AgeGroup::MiddleAge => (50, 64),
            AgeGroup::Senior => (65, 79),
            AgeGroup::Elderly => (80, 120),
        }
    }
}

/// Handedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Handedness {
    Right,
    Left,
    Ambidextrous,
    Unknown,
}

/// Additional demographic factors.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemographicFactor {
    /// Factor name.
    pub name: String,
    /// Factor value.
    pub value: String,
    /// Factor category.
    pub category: FactorCategory,
}

impl DemographicFactor {
    /// Create a new factor.
    pub fn new(name: &str, value: &str, category: FactorCategory) -> Self {
        Self {
            name: name.to_string(),
            value: value.to_string(),
            category,
        }
    }

    /// Create a socioeconomic factor.
    pub fn socioeconomic(name: &str, value: &str) -> Self {
        Self::new(name, value, FactorCategory::Socioeconomic)
    }

    /// Create a medical factor.
    pub fn medical(name: &str, value: &str) -> Self {
        Self::new(name, value, FactorCategory::Medical)
    }
}

/// Category of demographic factors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FactorCategory {
    /// Socioeconomic status.
    Socioeconomic,
    /// Geographic location.
    Geographic,
    /// Medical/Health factors.
    Medical,
    /// Lifestyle factors.
    Lifestyle,
    /// Environmental factors.
    Environmental,
    /// Other factors.
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_demographics_builder() {
        let demo = Demographics::new()
            .with_age(35)
            .with_sex(Sex::Female)
            .with_ethnicity(Ethnicity::EastAsian)
            .with_education(16);

        assert_eq!(demo.age, Some(35));
        assert_eq!(demo.sex, Some(Sex::Female));
        assert_eq!(demo.ethnicity, Some(Ethnicity::EastAsian));
        assert_eq!(demo.education_years, Some(16));
    }

    #[test]
    fn test_age_group() {
        assert_eq!(AgeGroup::from_age(5), AgeGroup::Pediatric);
        assert_eq!(AgeGroup::from_age(15), AgeGroup::Adolescent);
        assert_eq!(AgeGroup::from_age(25), AgeGroup::YoungAdult);
        assert_eq!(AgeGroup::from_age(40), AgeGroup::Adult);
        assert_eq!(AgeGroup::from_age(55), AgeGroup::MiddleAge);
        assert_eq!(AgeGroup::from_age(70), AgeGroup::Senior);
        assert_eq!(AgeGroup::from_age(85), AgeGroup::Elderly);
    }

    #[test]
    fn test_demographics_validate() {
        let valid = Demographics::new().with_age(50).with_education(16);
        assert!(valid.validate().is_ok());

        let invalid_age = Demographics::new().with_age(150);
        assert!(invalid_age.validate().is_err());
    }

    #[test]
    fn test_ethnicity_broad_category() {
        assert_eq!(
            Ethnicity::European.broad_category(),
            BroadEthnicCategory::European
        );
        assert_eq!(
            Ethnicity::EastAsian.broad_category(),
            BroadEthnicCategory::Asian
        );
        assert_eq!(
            Ethnicity::SouthAsian.broad_category(),
            BroadEthnicCategory::Asian
        );
        assert_eq!(
            Ethnicity::African.broad_category(),
            BroadEthnicCategory::African
        );
    }

    #[test]
    fn test_population_key() {
        let demo = Demographics::new()
            .with_age(35)
            .with_sex(Sex::Male)
            .with_ethnicity(Ethnicity::European);

        let key = demo.population_key();
        assert!(key.contains("Adult"));
        assert!(key.contains("Male"));
        assert!(key.contains("European"));
    }

    #[test]
    fn test_demographic_factor() {
        let factor = DemographicFactor::socioeconomic("income_level", "middle");
        assert_eq!(factor.name, "income_level");
        assert_eq!(factor.category, FactorCategory::Socioeconomic);
    }
}
