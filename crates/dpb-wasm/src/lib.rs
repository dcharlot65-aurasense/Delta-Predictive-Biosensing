//! # dpb-wasm
//!
//! WebAssembly bindings for the Delta-Predictive Biosensing (DPB) Framework.
//!
//! This crate provides browser-compatible biosignal processing, enabling:
//! - Real-time spike encoding in web applications
//! - Telehealth dashboards with client-side processing
//! - Interactive biosignal visualization
//! - Zero server round-trip signal analysis
//!
//! ## Usage from JavaScript
//!
//! ```javascript
//! import init, { WasmTimeSeries, WasmLevelCrossingEncoder, version } from './dpb_wasm.js';
//!
//! async function main() {
//!     await init();
//!     console.log(version());
//!
//!     // Create signal data
//!     const data = new Float32Array(1000);
//!     for (let i = 0; i < 1000; i++) {
//!         data[i] = Math.sin(i * 0.01 * 2 * Math.PI);
//!     }
//!
//!     // Create TimeSeries
//!     const ts = new WasmTimeSeries(data, 1000, 1, 1000.0);
//!
//!     // Encode to spikes
//!     const encoder = new WasmLevelCrossingEncoder(0.1);
//!     const spikes = encoder.encode(ts);
//!
//!     console.log(`Generated ${spikes.length} spikes`);
//! }
//! ```

use wasm_bindgen::prelude::*;

pub mod timeseries;
pub mod spiketrain;
pub mod encoders;
pub mod webgpu;
pub mod webnn;
mod utils;

pub use timeseries::WasmTimeSeries;
pub use spiketrain::WasmSpikeTrain;
pub use encoders::{WasmLevelCrossingEncoder, WasmDeltaEncoder};
pub use webgpu::{GpuEncoder, GpuEncoderConfig};
pub use webnn::{WebNNEncoder, WebNNConfig, WebNNDeviceType, WebNNFeatures};

/// Initialize the WASM module.
/// Called automatically when the module loads.
#[wasm_bindgen(start)]
pub fn init() {
    // Was an inline copy of utils::set_panic_hook's body, which left the
    // helper itself unreferenced.
    utils::set_panic_hook();
}

/// Get DPB WASM version string.
#[wasm_bindgen]
pub fn version() -> String {
    format!("DPB WASM v{}", env!("CARGO_PKG_VERSION"))
}

/// Log a message to the browser console.
#[wasm_bindgen]
pub fn log(message: &str) {
    web_sys::console::log_1(&message.into());
}

/// Performance timer for benchmarking.
#[wasm_bindgen]
#[derive(Debug)]
pub struct PerformanceTimer {
    start: f64,
}

#[wasm_bindgen]
impl PerformanceTimer {
    /// Create a new timer starting now.
    ///
    /// # Errors
    /// Returns an error if the browser window or performance API is not available.
    #[wasm_bindgen(constructor)]
    #[must_use = "this Result may contain an error that should be handled"]
    pub fn new() -> Result<PerformanceTimer, JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let performance = window.performance().ok_or("no performance API")?;
        Ok(Self {
            start: performance.now(),
        })
    }

    /// Get elapsed time in milliseconds.
    ///
    /// # Errors
    /// Returns an error if the browser window or performance API is not available.
    #[wasm_bindgen]
    #[must_use = "this Result may contain an error that should be handled"]
    pub fn elapsed_ms(&self) -> Result<f64, JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let performance = window.performance().ok_or("no performance API")?;
        Ok(performance.now() - self.start)
    }

    /// Reset the timer.
    ///
    /// # Errors
    /// Returns an error if the browser window or performance API is not available.
    #[wasm_bindgen]
    #[must_use = "this Result may contain an error that should be handled"]
    pub fn reset(&mut self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or("no window")?;
        let performance = window.performance().ok_or("no performance API")?;
        self.start = performance.now();
        Ok(())
    }
}

impl Default for PerformanceTimer {
    fn default() -> Self {
        Self::new().unwrap_or(Self { start: 0.0 })
    }
}
