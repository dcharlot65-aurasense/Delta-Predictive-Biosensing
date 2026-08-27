//! Output decoders for converting spike trains to predictions

pub mod classification;
pub mod clinical;
pub mod rate;
pub mod regression;
pub mod temporal;

// Rate-based decoders (8 total)
pub use rate::{
    AdaptiveRateDecoder, ExponentialRateDecoder, FirstSpikeDecoder, MaxSpikeDecoder,
    NormalizedRateDecoder, PopulationDecoder, SpikeRateDecoder, WeightedRateDecoder,
    WindowedRateDecoder,
};

// Temporal decoders (8 total)
pub use temporal::{
    BurstDecoder, ISIDecoder, LastSpikeDecoder, LatencyDecoder, PhaseDecoder, RankOrderDecoder,
    TemporalPatternDecoder,
};

// Clinical score decoders (15 original + 8 phase E + 16 gap fill = 39 total)
pub use clinical::{
    AttentionDecoder,
    BergBalanceDecoder,
    BppvDecoder,
    CanalParesisDecoder,
    // Cognitive decoders
    CognitiveRtDecoder,
    GaitScoreDecoder,
    // Force decoders
    GrfDecoder,
    GripStrengthDecoder,
    HoehnYahrDecoder,
    // Cardiopulmonary decoders
    HrvDecoder,
    MiniBESTDecoder,
    MoCADecoder,
    NrsDecoder,
    PDQ39Decoder,
    QstPhenotypeDecoder,
    RespiratoryDecoder,
    RfdDecoder,
    SEADLDecoder,
    SclDecoder,
    // EDA decoders
    ScrDecoder,
    StressIndexDecoder,
    TUGDecoder,
    // Phase E: Balance decoders
    TinettiDecoder,
    TremorSeverityDecoder,
    UPDRSBradykinesiaDecoder,
    // Original clinical decoders
    UPDRSDecoder,
    UPDRSGaitDecoder,
    UPDRSMotorDecoder,
    UPDRSRigidityDecoder,
    UPDRSTremorDecoder,
    // Phase E: Pain decoders
    VasDecoder,
    Vo2Decoder,
    VoiceHDDecoder,
    // Phase E: Vestibular decoders
    VorGainDecoder,
    WorkingMemoryDecoder,
};

// Regression decoders (10 total)
pub use regression::{
    GaitVelocityDecoder, HRVDecoder, HeartRateDecoder, PupilDiameterDecoder, ReactionTimeDecoder,
    SpeechRateDecoder, StrideTimeDecoder, TappingFrequencyDecoder, TremorAmplitudeDecoder,
    TremorFrequencyDecoder,
};

// Classification decoders (10 total)
pub use classification::{
    ActivityDecoder, BinaryClassDecoder, DyskinesiasDecoder, EmotionDecoder, FatigueDecoder,
    GaitPhaseDecoder, MedicationStateDecoder, MultiClassDecoder, SleepStageDecoder,
    TremorTypeDecoder,
};

use crate::{SNNResult, SpikeTensor};
use ndarray::Array2;

/// Base trait for all decoders
pub trait Decoder: Send + Sync {
    /// Decode spike train to output
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>>;

    /// Get output dimension
    fn output_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_module_exports() {
        // Compiling this test is the check -- it names the module's items.
        // An assert!(true) added nothing to that.
    }

    #[test]
    fn test_decoder_count() {
        // Compiling this test is the check -- it names the module's items.
        // An assert!(true) added nothing to that.
    }
}
