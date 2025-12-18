//! Pain assessment algorithms
//!
//! Provides tools for:
//! - Quantitative Sensory Testing (QST)
//! - Pain threshold and tolerance measurement
//! - Clinical pain scales (VAS, NRS, McGill)
//! - Temporal summation and wind-up

pub mod qst;
pub mod scales;

pub use qst::{
    ModalityType, QstAnalyzer, QstMetrics, QstResult, SensoryPhenotype, TemporalSummation,
};
pub use scales::{
    BriefPainInventory, McGillPainQuestionnaire, NeuropathicPainScale, PainCatastrophizing,
    PainScale, VasScore,
};
