//! Cardiopulmonary assessment algorithms
//!
//! Provides tools for:
//! - VO2 estimation from submaximal tests
//! - Ventilatory threshold detection
//! - Respiratory gas exchange analysis
//! - Exercise capacity assessment

pub mod gas_exchange;
pub mod ventilatory;
pub mod vo2;

pub use gas_exchange::{GasExchange, GasExchangeMetrics, RespiratoryQuotient};
pub use ventilatory::{VentilatoryThreshold, VtMethod, VtResult};
pub use vo2::{ExerciseProtocol, Vo2Estimator, Vo2Metrics, Vo2Prediction};
