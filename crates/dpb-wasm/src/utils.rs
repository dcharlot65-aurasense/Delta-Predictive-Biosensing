//! Utility functions for WASM module

use wasm_bindgen::prelude::*;

/// Set panic hook for better error messages in console.
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

/// Log an error to the console.
#[wasm_bindgen(js_name = "logError")]
pub fn log_error(message: &str) {
    web_sys::console::error_1(&message.into());
}

/// Log a warning to the console.
#[wasm_bindgen(js_name = "logWarning")]
pub fn log_warning(message: &str) {
    web_sys::console::warn_1(&message.into());
}

/// Check if running in a browser environment.
#[wasm_bindgen(js_name = "isBrowser")]
pub fn is_browser() -> bool {
    web_sys::window().is_some()
}

/// Get the user agent string (if available).
#[wasm_bindgen(js_name = "getUserAgent")]
pub fn get_user_agent() -> Option<String> {
    web_sys::window()
        .and_then(|w| w.navigator().user_agent().ok())
}
