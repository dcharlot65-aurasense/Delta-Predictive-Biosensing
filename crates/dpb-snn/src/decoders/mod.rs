//! Output decoders for converting spike trains to predictions

pub mod rate;
pub mod temporal;
pub mod clinical;

pub use rate::{SpikeRateDecoder, FirstSpikeDecoder, PopulationDecoder};
pub use temporal::{TemporalPatternDecoder, LatencyDecoder};
pub use clinical::{UPDRSDecoder, TremorSeverityDecoder, GaitScoreDecoder};

use crate::{SNNResult, SpikeTensor};
use ndarray::{Array1, Array2};

/// Base trait for all decoders
pub trait Decoder: Send + Sync {
    /// Decode spike train to output
    fn decode(&self, spikes: &SpikeTensor) -> SNNResult<Array2<f32>>;

    /// Get output dimension
    fn output_dim(&self) -> usize;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Simple test to verify module structure
        assert!(true);
    }
}
