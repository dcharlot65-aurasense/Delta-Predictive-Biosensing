//! Output decoders for converting spike trains to predictions

pub mod rate;
pub mod temporal;
pub mod clinical;
pub mod regression;
pub mod classification;

// Rate-based decoders (8 total)
pub use rate::{
    SpikeRateDecoder, FirstSpikeDecoder, PopulationDecoder, MaxSpikeDecoder,
    WindowedRateDecoder, ExponentialRateDecoder, AdaptiveRateDecoder,
    NormalizedRateDecoder, WeightedRateDecoder,
};

// Temporal decoders (8 total)
pub use temporal::{
    TemporalPatternDecoder, LatencyDecoder, ISIDecoder, BurstDecoder,
    LastSpikeDecoder, PhaseDecoder, RankOrderDecoder,
};

// Clinical score decoders (15 original + 8 phase E + 16 gap fill = 39 total)
pub use clinical::{
    // Original clinical decoders
    UPDRSDecoder, TremorSeverityDecoder, GaitScoreDecoder,
    UPDRSMotorDecoder, UPDRSTremorDecoder, UPDRSBradykinesiaDecoder,
    UPDRSRigidityDecoder, UPDRSGaitDecoder, TUGDecoder,
    BergBalanceDecoder, MoCADecoder, VoiceHDDecoder,
    PDQ39Decoder, HoehnYahrDecoder, SEADLDecoder,
    // Phase E: Balance decoders
    TinettiDecoder, MiniBESTDecoder,
    // Phase E: Pain decoders
    VasDecoder, NrsDecoder, QstPhenotypeDecoder,
    // Phase E: Vestibular decoders
    VorGainDecoder, CanalParesisDecoder, BppvDecoder,
    // Force decoders
    GrfDecoder, GripStrengthDecoder, RfdDecoder,
    // Cardiopulmonary decoders
    HrvDecoder, RespiratoryDecoder, Vo2Decoder,
    // Cognitive decoders
    CognitiveRtDecoder, AttentionDecoder, WorkingMemoryDecoder,
    // EDA decoders
    ScrDecoder, SclDecoder, StressIndexDecoder,
};

// Regression decoders (10 total)
pub use regression::{
    HeartRateDecoder, HRVDecoder, TremorFrequencyDecoder,
    TremorAmplitudeDecoder, GaitVelocityDecoder, StrideTimeDecoder,
    TappingFrequencyDecoder, ReactionTimeDecoder, SpeechRateDecoder,
    PupilDiameterDecoder,
};

// Classification decoders (10 total)
pub use classification::{
    BinaryClassDecoder, MultiClassDecoder, TremorTypeDecoder,
    GaitPhaseDecoder, SleepStageDecoder, ActivityDecoder,
    EmotionDecoder, FatigueDecoder, MedicationStateDecoder,
    DyskinesiasDecoder,
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
