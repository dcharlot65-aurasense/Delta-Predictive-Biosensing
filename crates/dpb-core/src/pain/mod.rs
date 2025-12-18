//! Pain assessment algorithms
//!
//! Provides tools for:
//! - Quantitative Sensory Testing (QST)
//! - Pressure pain threshold and tolerance
//! - Conditioned Pain Modulation (CPM)
//! - Clinical pain scales (VAS, NRS, McGill)
//! - Temporal summation and wind-up
//! - Pain-related autonomic responses

pub mod autonomic;
pub mod qst;
pub mod scales;
pub mod threshold;

pub use autonomic::{
    AutonomicPainProfile, HrResponse, PainAutonomicAnalyzer, PupilResponse, RespiratoryResponse,
    ScrMetrics,
};
pub use qst::{
    ModalityType, QstAnalyzer, QstMetrics, QstResult, SensoryPhenotype, TemporalSummation,
};
pub use scales::{
    BriefPainInventory, McGillPainQuestionnaire, NeuropathicPainScale, PainCatastrophizing,
    PainScale, VasScore,
};
pub use threshold::{
    ConditionedPainModulation, CpmMetrics, CpmResponse, PptMetrics, PptSite, PptTrial,
    PressurePainThreshold, Sex, Side, SideDifference, StimulusType,
};
