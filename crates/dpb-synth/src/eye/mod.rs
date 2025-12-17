//! Eye movement generators

pub mod saccade;
pub mod fixation;
pub mod pupil;
pub mod pursuit;
pub mod noise;

pub use saccade::*;
pub use fixation::*;
pub use pupil::*;
pub use pursuit::*;
pub use noise::*;
