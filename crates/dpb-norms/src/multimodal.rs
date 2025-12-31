//! Multi-modal normative comparisons
//!
//! Provides cross-domain comparison capabilities for integrated assessments
//! across cognitive, motor, physiological, and other biosensing domains.

use crate::{
    Demographics, ImpairmentLevel, MetricDirection, MetricDomain, MetricType,
    NormativeComparison, NormativeDatabase, NormsError, Result,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Multi-modal assessment result combining multiple domains
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalAssessment {
    /// Individual metric comparisons
    pub comparisons: Vec<NormativeComparison>,
    /// Domain-level summaries
    pub domain_summaries: HashMap<MetricDomain, DomainSummary>,
    /// Overall profile classification
    pub profile: MultiModalProfile,
    /// Demographics used for comparisons
    pub demographics: Demographics,
}

/// Summary of normative comparisons within a single domain
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainSummary {
    /// Domain
    pub domain: MetricDomain,
    /// Number of metrics in domain
    pub metric_count: usize,
    /// Mean z-score across domain
    pub mean_z_score: f64,
    /// Standard deviation of z-scores
    pub z_score_sd: f64,
    /// Mean percentile
    pub mean_percentile: f64,
    /// Worst impairment level in domain
    pub worst_impairment: ImpairmentLevel,
    /// Count of impaired metrics (z < -1.0)
    pub impaired_count: usize,
    /// Domain-level classification
    pub classification: DomainClassification,
}

/// Classification of performance within a domain
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainClassification {
    /// All metrics within normal limits
    WellPreserved,
    /// Mostly normal with minor weaknesses
    Adequate,
    /// Mixed performance with some impairments
    Heterogeneous,
    /// Consistent mild-moderate impairment
    Impaired,
    /// Severe impairment across domain
    SeverelyImpaired,
}

impl DomainClassification {
    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            DomainClassification::WellPreserved => "Well Preserved",
            DomainClassification::Adequate => "Adequate",
            DomainClassification::Heterogeneous => "Heterogeneous",
            DomainClassification::Impaired => "Impaired",
            DomainClassification::SeverelyImpaired => "Severely Impaired",
        }
    }

    /// Get severity score (0-4)
    pub fn severity(&self) -> u8 {
        match self {
            DomainClassification::WellPreserved => 0,
            DomainClassification::Adequate => 1,
            DomainClassification::Heterogeneous => 2,
            DomainClassification::Impaired => 3,
            DomainClassification::SeverelyImpaired => 4,
        }
    }
}

/// Multi-modal profile classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalProfile {
    /// Primary classification
    pub classification: ProfileClassification,
    /// Domains ordered by impairment (worst first)
    pub domain_ranking: Vec<(MetricDomain, f64)>,
    /// Dissociation patterns detected
    pub dissociations: Vec<Dissociation>,
    /// Global impairment index (0-1)
    pub global_impairment_index: f64,
    /// Discrepancy between domains
    pub domain_discrepancy: f64,
}

/// Overall profile classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileClassification {
    /// All domains normal
    Normal,
    /// Single domain impaired
    FocalImpairment,
    /// Multiple domains impaired
    MultidomainImpairment,
    /// All domains impaired
    GlobalImpairment,
    /// Significant variability between domains
    DissociatedProfile,
}

impl ProfileClassification {
    /// Get descriptive label
    pub fn label(&self) -> &'static str {
        match self {
            ProfileClassification::Normal => "Normal Function",
            ProfileClassification::FocalImpairment => "Focal Impairment",
            ProfileClassification::MultidomainImpairment => "Multi-domain Impairment",
            ProfileClassification::GlobalImpairment => "Global Impairment",
            ProfileClassification::DissociatedProfile => "Dissociated Profile",
        }
    }
}

/// Dissociation between domains or metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dissociation {
    /// Type of dissociation
    pub dissociation_type: DissociationType,
    /// First domain/metric
    pub first: String,
    /// Second domain/metric
    pub second: String,
    /// Z-score difference
    pub z_difference: f64,
    /// Statistical significance (based on expected correlation)
    pub significant: bool,
}

/// Types of dissociations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DissociationType {
    /// Cognitive vs motor dissociation
    CognitiveMotor,
    /// Cognitive vs physiological dissociation
    CognitivePhysiological,
    /// Motor vs physiological dissociation
    MotorPhysiological,
    /// Within-domain metric dissociation
    WithinDomain,
    /// Custom dissociation
    Custom,
}

/// Multi-modal assessment builder
pub struct MultiModalAssessor<'a> {
    database: &'a NormativeDatabase,
    demographics: Demographics,
    measurements: Vec<(MetricType, f64)>,
}

impl<'a> MultiModalAssessor<'a> {
    /// Create a new multi-modal assessor
    pub fn new(database: &'a NormativeDatabase, demographics: Demographics) -> Self {
        Self {
            database,
            demographics,
            measurements: Vec::new(),
        }
    }

    /// Add a measurement
    pub fn add_measurement(&mut self, metric: MetricType, value: f64) -> &mut Self {
        self.measurements.push((metric, value));
        self
    }

    /// Add multiple measurements
    pub fn add_measurements(&mut self, measurements: &[(MetricType, f64)]) -> &mut Self {
        self.measurements.extend_from_slice(measurements);
        self
    }

    /// Perform the multi-modal assessment
    pub fn assess(&self) -> Result<MultiModalAssessment> {
        if self.measurements.is_empty() {
            return Err(NormsError::InsufficientData);
        }

        // Generate individual comparisons
        let comparisons: Vec<NormativeComparison> = self
            .measurements
            .iter()
            .filter_map(|&(metric, value)| {
                self.database
                    .compare(metric, value, &self.demographics)
                    .ok()
            })
            .collect();

        if comparisons.is_empty() {
            return Err(NormsError::InsufficientData);
        }

        // Generate domain summaries
        let domain_summaries = self.calculate_domain_summaries(&comparisons);

        // Generate profile
        let profile = self.generate_profile(&comparisons, &domain_summaries);

        Ok(MultiModalAssessment {
            comparisons,
            domain_summaries,
            profile,
            demographics: self.demographics.clone(),
        })
    }

    /// Calculate domain-level summaries
    fn calculate_domain_summaries(
        &self,
        comparisons: &[NormativeComparison],
    ) -> HashMap<MetricDomain, DomainSummary> {
        let mut domain_map: HashMap<MetricDomain, Vec<&NormativeComparison>> = HashMap::new();

        for comp in comparisons {
            let domain = comp.metric.domain();
            domain_map.entry(domain).or_default().push(comp);
        }

        domain_map
            .into_iter()
            .map(|(domain, comps)| {
                let summary = self.summarize_domain(domain, &comps);
                (domain, summary)
            })
            .collect()
    }

    /// Summarize a single domain
    fn summarize_domain(&self, domain: MetricDomain, comparisons: &[&NormativeComparison]) -> DomainSummary {
        let metric_count = comparisons.len();

        // Calculate z-score statistics
        let z_scores: Vec<f64> = comparisons.iter().map(|c| c.z_score).collect();
        let mean_z_score = z_scores.iter().sum::<f64>() / metric_count as f64;
        let z_score_variance: f64 = z_scores
            .iter()
            .map(|z| (z - mean_z_score).powi(2))
            .sum::<f64>()
            / metric_count as f64;
        let z_score_sd = z_score_variance.sqrt();

        // Calculate percentile statistics
        let percentiles: Vec<f64> = comparisons.iter().map(|c| c.percentile).collect();
        let mean_percentile = percentiles.iter().sum::<f64>() / metric_count as f64;

        // Find worst impairment
        let worst_impairment = comparisons
            .iter()
            .map(|c| c.impairment)
            .max_by_key(|i| i.severity())
            .unwrap_or(ImpairmentLevel::Normal);

        // Count impaired metrics
        let impaired_count = comparisons.iter().filter(|c| c.z_score < -1.0).count();

        // Classify domain
        let classification = self.classify_domain(mean_z_score, z_score_sd, impaired_count, metric_count);

        DomainSummary {
            domain,
            metric_count,
            mean_z_score,
            z_score_sd,
            mean_percentile,
            worst_impairment,
            impaired_count,
            classification,
        }
    }

    /// Classify domain performance
    fn classify_domain(
        &self,
        mean_z: f64,
        z_sd: f64,
        impaired_count: usize,
        total_count: usize,
    ) -> DomainClassification {
        let impaired_ratio = impaired_count as f64 / total_count as f64;

        if mean_z >= -0.5 && impaired_ratio < 0.1 {
            DomainClassification::WellPreserved
        } else if mean_z >= -1.0 && impaired_ratio < 0.3 {
            DomainClassification::Adequate
        } else if z_sd > 1.0 && impaired_ratio > 0.2 && impaired_ratio < 0.7 {
            DomainClassification::Heterogeneous
        } else if mean_z >= -2.0 || impaired_ratio < 0.8 {
            DomainClassification::Impaired
        } else {
            DomainClassification::SeverelyImpaired
        }
    }

    /// Generate multi-modal profile
    fn generate_profile(
        &self,
        comparisons: &[NormativeComparison],
        domain_summaries: &HashMap<MetricDomain, DomainSummary>,
    ) -> MultiModalProfile {
        // Rank domains by impairment
        let mut domain_ranking: Vec<(MetricDomain, f64)> = domain_summaries
            .iter()
            .map(|(d, s)| (*d, s.mean_z_score))
            .collect();
        domain_ranking.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Calculate global impairment index
        let global_z: f64 = comparisons.iter().map(|c| c.z_score).sum::<f64>()
            / comparisons.len() as f64;
        let global_impairment_index = ((-global_z).max(0.0) / 3.0).min(1.0);

        // Calculate domain discrepancy
        let domain_z_scores: Vec<f64> = domain_summaries.values().map(|s| s.mean_z_score).collect();
        let domain_discrepancy = if domain_z_scores.len() >= 2 {
            let max_z = domain_z_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_z = domain_z_scores.iter().cloned().fold(f64::INFINITY, f64::min);
            max_z - min_z
        } else {
            0.0
        };

        // Detect dissociations
        let dissociations = self.detect_dissociations(domain_summaries);

        // Classify profile
        let impaired_domains = domain_summaries
            .values()
            .filter(|s| s.mean_z_score < -1.0)
            .count();
        let total_domains = domain_summaries.len();

        let classification = if global_z >= -0.5 && impaired_domains == 0 {
            ProfileClassification::Normal
        } else if domain_discrepancy > 1.5 && !dissociations.is_empty() {
            ProfileClassification::DissociatedProfile
        } else if impaired_domains == 1 {
            ProfileClassification::FocalImpairment
        } else if impaired_domains < total_domains {
            ProfileClassification::MultidomainImpairment
        } else {
            ProfileClassification::GlobalImpairment
        };

        MultiModalProfile {
            classification,
            domain_ranking,
            dissociations,
            global_impairment_index,
            domain_discrepancy,
        }
    }

    /// Detect dissociations between domains
    fn detect_dissociations(
        &self,
        domain_summaries: &HashMap<MetricDomain, DomainSummary>,
    ) -> Vec<Dissociation> {
        let mut dissociations = Vec::new();

        // Check cognitive-motor dissociation
        if let (Some(cog), Some(mot)) = (
            domain_summaries.get(&MetricDomain::Cognitive),
            domain_summaries.get(&MetricDomain::Motor),
        ) {
            let diff = (cog.mean_z_score - mot.mean_z_score).abs();
            if diff > 1.5 {
                dissociations.push(Dissociation {
                    dissociation_type: DissociationType::CognitiveMotor,
                    first: "Cognitive".to_string(),
                    second: "Motor".to_string(),
                    z_difference: diff,
                    significant: diff > 2.0,
                });
            }
        }

        // Check cognitive-physiological dissociation
        if let (Some(cog), Some(phys)) = (
            domain_summaries.get(&MetricDomain::Cognitive),
            domain_summaries.get(&MetricDomain::Physiological),
        ) {
            let diff = (cog.mean_z_score - phys.mean_z_score).abs();
            if diff > 1.5 {
                dissociations.push(Dissociation {
                    dissociation_type: DissociationType::CognitivePhysiological,
                    first: "Cognitive".to_string(),
                    second: "Physiological".to_string(),
                    z_difference: diff,
                    significant: diff > 2.0,
                });
            }
        }

        // Check motor-physiological dissociation
        if let (Some(mot), Some(phys)) = (
            domain_summaries.get(&MetricDomain::Motor),
            domain_summaries.get(&MetricDomain::Physiological),
        ) {
            let diff = (mot.mean_z_score - phys.mean_z_score).abs();
            if diff > 1.5 {
                dissociations.push(Dissociation {
                    dissociation_type: DissociationType::MotorPhysiological,
                    first: "Motor".to_string(),
                    second: "Physiological".to_string(),
                    z_difference: diff,
                    significant: diff > 2.0,
                });
            }
        }

        dissociations
    }
}

/// Generate a comprehensive multi-modal report
pub fn generate_report(assessment: &MultiModalAssessment) -> String {
    let mut report = String::new();

    report.push_str("=== Multi-Modal Normative Assessment Report ===\n\n");

    // Demographics
    report.push_str(&format!(
        "Subject: {} year old {:?}\n\n",
        assessment.demographics.age, assessment.demographics.sex
    ));

    // Global summary
    report.push_str("== Global Summary ==\n");
    report.push_str(&format!(
        "Profile Classification: {}\n",
        assessment.profile.classification.label()
    ));
    report.push_str(&format!(
        "Global Impairment Index: {:.2} (0=normal, 1=severe)\n",
        assessment.profile.global_impairment_index
    ));
    report.push_str(&format!(
        "Domain Discrepancy: {:.2} SD\n\n",
        assessment.profile.domain_discrepancy
    ));

    // Domain summaries
    report.push_str("== Domain Summaries ==\n");
    for (domain, summary) in &assessment.domain_summaries {
        report.push_str(&format!("\n{:?}:\n", domain));
        report.push_str(&format!(
            "  Metrics: {} | Mean Z: {:.2} | Classification: {}\n",
            summary.metric_count,
            summary.mean_z_score,
            summary.classification.label()
        ));
        report.push_str(&format!(
            "  Impaired: {}/{} | Worst: {}\n",
            summary.impaired_count,
            summary.metric_count,
            summary.worst_impairment.label()
        ));
    }

    // Dissociations
    if !assessment.profile.dissociations.is_empty() {
        report.push_str("\n== Dissociations Detected ==\n");
        for diss in &assessment.profile.dissociations {
            report.push_str(&format!(
                "  {} vs {}: {:.2} SD difference {}\n",
                diss.first,
                diss.second,
                diss.z_difference,
                if diss.significant {
                    "(SIGNIFICANT)"
                } else {
                    ""
                }
            ));
        }
    }

    // Domain ranking
    report.push_str("\n== Domain Ranking (worst to best) ==\n");
    for (i, (domain, z)) in assessment.profile.domain_ranking.iter().enumerate() {
        report.push_str(&format!("  {}. {:?} (z={:.2})\n", i + 1, domain, z));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::demographics::Sex;

    #[test]
    fn test_multimodal_assessor() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(50, Sex::Male);

        let mut assessor = MultiModalAssessor::new(&db, demo);
        assessor
            .add_measurement(MetricType::SimpleReactionTime, 290.0) // Normal
            .add_measurement(MetricType::GaitVelocity, 1.0) // Slightly below avg
            .add_measurement(MetricType::GripStrength, 30.0) // Below avg
            .add_measurement(MetricType::HrvSdnn, 80.0); // Below avg

        let assessment = assessor.assess().unwrap();

        assert!(!assessment.comparisons.is_empty());
        assert!(!assessment.domain_summaries.is_empty());
    }

    #[test]
    fn test_domain_classification() {
        // Test well-preserved domain
        let assessor = MultiModalAssessor::new(
            &NormativeDatabase::with_defaults(),
            Demographics::new(30, Sex::Male),
        );
        let classification = assessor.classify_domain(-0.3, 0.5, 0, 5);
        assert_eq!(classification, DomainClassification::WellPreserved);

        // Test impaired domain
        let classification = assessor.classify_domain(-1.8, 0.3, 4, 5);
        assert_eq!(classification, DomainClassification::Impaired);
    }

    #[test]
    fn test_profile_classification() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(70, Sex::Female);

        // Test with normal values
        let mut assessor = MultiModalAssessor::new(&db, demo.clone());
        assessor
            .add_measurement(MetricType::SimpleReactionTime, 340.0)
            .add_measurement(MetricType::GaitVelocity, 0.98);

        let assessment = assessor.assess().unwrap();
        // Should be close to normal for 70 year old
        assert!(assessment.profile.global_impairment_index < 0.5);
    }

    #[test]
    fn test_report_generation() {
        let db = NormativeDatabase::with_defaults();
        let demo = Demographics::new(45, Sex::Male);

        let mut assessor = MultiModalAssessor::new(&db, demo);
        assessor
            .add_measurement(MetricType::SimpleReactionTime, 280.0)
            .add_measurement(MetricType::GaitVelocity, 1.2)
            .add_measurement(MetricType::HrvSdnn, 100.0);

        let assessment = assessor.assess().unwrap();
        let report = generate_report(&assessment);

        assert!(report.contains("Multi-Modal Normative Assessment Report"));
        assert!(report.contains("Global Summary"));
        assert!(report.contains("Domain Summaries"));
    }
}
