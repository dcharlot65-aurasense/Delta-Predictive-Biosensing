//! Clinical validation metrics
//!
//! Provides tools for validating clinical algorithms:
//! - Test-retest reliability (ICC, SEM, MDC)
//! - Inter-rater reliability
//! - Criterion validity
//! - Sensitivity to change
//! - ROC analysis

pub mod reliability;
pub mod validity;
pub mod roc;

pub use reliability::{
    IccModel, IccType, ReliabilityMetrics, TestRetestReliability,
};
pub use validity::{
    BlandAltmanResult, ChangeMetrics, CriterionValidity, ValidityMetrics,
};
pub use roc::{
    RocAnalyzer, RocCurve, RocPoint, OptimalCutoff,
};
