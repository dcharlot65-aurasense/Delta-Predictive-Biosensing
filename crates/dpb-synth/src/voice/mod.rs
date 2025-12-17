//! Voice signal generators

pub mod phonation;
pub mod prosody;
pub mod articulation;
pub mod pathological;
pub mod noise;

pub use phonation::*;
pub use prosody::*;
pub use articulation::*;
pub use pathological::*;
pub use noise::*;
