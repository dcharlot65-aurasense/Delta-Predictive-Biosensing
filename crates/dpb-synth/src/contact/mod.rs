//! Contact biosignal generators
//!
//! Generators for ECG, PPG, EDA, tremor, EMG, respiratory, thermal, and synchronized signals

pub mod ecg;
pub mod eda;
pub mod emg;
pub mod noise;
pub mod ppg;
pub mod respiratory;
pub mod sync;
pub mod thermal;
pub mod tremor;

pub use ecg::*;
pub use eda::*;
pub use emg::*;
pub use noise::*;
pub use ppg::*;
pub use respiratory::*;
pub use sync::*;
pub use thermal::*;
pub use tremor::*;
