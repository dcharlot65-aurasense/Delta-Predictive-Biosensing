//! Contact biosignal generators
//!
//! Generators for ECG, PPG, EDA, tremor, EMG, and respiratory signals

pub mod ecg;
pub mod ppg;
pub mod eda;
pub mod tremor;
pub mod emg;
pub mod respiratory;
pub mod noise;

pub use ecg::*;
pub use ppg::*;
pub use eda::*;
pub use tremor::*;
pub use emg::*;
pub use respiratory::*;
pub use noise::*;
