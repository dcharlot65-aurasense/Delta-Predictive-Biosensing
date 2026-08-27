//! Clinical validation metrics
//!
//! Provides tools for validating clinical algorithms:
//! - Test-retest reliability (ICC, SEM, MDC)
//! - Inter-rater reliability
//! - Criterion validity
//! - Sensitivity to change
//! - ROC analysis

pub mod reliability;
pub mod roc;
pub mod validity;

pub use reliability::{IccModel, IccType, ReliabilityMetrics, TestRetestReliability};
pub use roc::{OptimalCutoff, RocAnalyzer, RocCurve, RocPoint};
pub use validity::{BlandAltmanResult, ChangeMetrics, CriterionValidity, ValidityMetrics};
