//! Balance and postural control analysis
//!
//! This module will provide tools for analyzing balance and postural control
//! from force plate and motion capture data.
//!
//! Currently implemented:
//! - Center of pressure (CoP) analysis, in [`cop`]
//!
//! Planned:
//! - Postural sway metrics
//! - Stability limits
//! - Dynamic balance assessment

// cop.rs sat beside this file without ever being declared, so its ~380 lines
// were never compiled and CopAnalyzer was unreachable from anywhere.
pub mod cop;

pub use cop::{CopAnalyzer, CopMetrics};
