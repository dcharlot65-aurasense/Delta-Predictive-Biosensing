//! Contact biosignal generators
//!
//! Generators for ECG, PPG, EDA, tremor, EMG, respiratory, thermal, and synchronized signals

pub mod ecg;
pub mod ppg;
pub mod eda;
pub mod tremor;
pub mod emg;
pub mod respiratory;
pub mod noise;
pub mod thermal;
pub mod sync;

pub use ecg::*;
pub use ppg::*;
pub use eda::*;
pub use tremor::*;
pub use emg::*;
pub use respiratory::*;
pub use noise::*;
pub use thermal::*;
pub use sync::*;
