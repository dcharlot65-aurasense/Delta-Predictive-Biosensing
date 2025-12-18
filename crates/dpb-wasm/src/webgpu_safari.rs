//! Safari-specific WebGPU compatibility layer.
//!
//! Safari's WebGPU implementation has some differences from Chrome/Firefox.
//! This module provides workarounds and compatibility shims.
//!
//! ## Safari WebGPU Status
//!
//! - Safari 17+: WebGPU support (experimental, behind flag)
//! - Safari 18+: WebGPU support (stable, enabled by default)
//!
//! ## Known Differences
//!
//! 1. **Shader compilation**: Safari may reject some WGSL that works in Chrome
//! 2. **Texture formats**: Some formats not supported
//! 3. **Limits**: Different device limits
//! 4. **Error handling**: Different error message formats
//!
//! ## Usage
//!
//! ```javascript
//! import { detectBrowser, getSafariConfig } from 'dpb-wasm';
//!
//! const browser = detectBrowser();
//! if (browser.isSafari) {
//!     const config = getSafariConfig();
//!     const encoder = await GpuEncoder.newWithConfig(config);
//! }
//! ```

use wasm_bindgen::prelude::*;

/// Browser detection result.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct BrowserInfo {
    /// Browser name (Chrome, Firefox, Safari, Edge, etc.)
    pub name: String,
    /// Major version number
    pub version: u32,
    /// Whether this is Safari
    pub is_safari: bool,
    /// Whether this is iOS (Safari or Chrome on iOS uses WebKit)
    pub is_ios: bool,
    /// Whether WebGPU is available
    pub has_webgpu: bool,
    /// Whether this is a mobile browser
    pub is_mobile: bool,
}

#[wasm_bindgen]
impl BrowserInfo {
    /// Get browser name.
    #[wasm_bindgen(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Get browser version.
    #[wasm_bindgen(getter)]
    pub fn version(&self) -> u32 {
        self.version
    }
}

/// Detect the current browser.
#[wasm_bindgen]
pub fn detect_browser() -> BrowserInfo {
    // In actual implementation, this would use web_sys to access navigator.userAgent
    // For now, return a placeholder that would be filled in by JavaScript

    BrowserInfo {
        name: "Unknown".to_string(),
        version: 0,
        is_safari: false,
        is_ios: false,
        has_webgpu: false,
        is_mobile: false,
    }
}

/// Safari-specific WebGPU configuration.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct SafariWebGPUConfig {
    /// Use compatibility shader syntax
    use_compat_shaders: bool,
    /// Maximum workgroup size (Safari may have lower limits)
    max_workgroup_size: u32,
    /// Preferred texture format
    preferred_texture_format: String,
    /// Enable fallback to WASM if WebGPU fails
    enable_wasm_fallback: bool,
    /// Shader compilation timeout in ms
    shader_compile_timeout_ms: u32,
}

#[wasm_bindgen]
impl SafariWebGPUConfig {
    /// Create default Safari configuration.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            use_compat_shaders: true,
            max_workgroup_size: 256, // Conservative default
            preferred_texture_format: "bgra8unorm".to_string(),
            enable_wasm_fallback: true,
            shader_compile_timeout_ms: 5000,
        }
    }

    /// Create configuration for Safari 17.
    pub fn for_safari_17() -> Self {
        Self {
            use_compat_shaders: true,
            max_workgroup_size: 256,
            preferred_texture_format: "bgra8unorm".to_string(),
            enable_wasm_fallback: true,
            shader_compile_timeout_ms: 10000, // Longer timeout for experimental
        }
    }

    /// Create configuration for Safari 18+.
    pub fn for_safari_18() -> Self {
        Self {
            use_compat_shaders: false,
            max_workgroup_size: 512,
            preferred_texture_format: "bgra8unorm".to_string(),
            enable_wasm_fallback: true,
            shader_compile_timeout_ms: 5000,
        }
    }

    /// Get maximum workgroup size.
    #[wasm_bindgen(getter)]
    pub fn max_workgroup_size(&self) -> u32 {
        self.max_workgroup_size
    }

    /// Whether to use compatibility shaders.
    #[wasm_bindgen(getter)]
    pub fn use_compat_shaders(&self) -> bool {
        self.use_compat_shaders
    }
}

impl Default for SafariWebGPUConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Safari-compatible WGSL shader variants.
pub mod compat_shaders {
    /// Level crossing shader optimized for Safari.
    ///
    /// Changes from standard shader:
    /// - Uses explicit array sizes
    /// - Avoids certain built-in functions
    /// - Simpler control flow
    pub const LEVEL_CROSSING_SAFARI: &str = r#"
// Safari-compatible level crossing shader
// Tested on Safari 17.4+

struct Params {
    threshold: f32,
    num_samples: u32,
    num_channels: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> spike_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: Params;

// Safari prefers explicit workgroup size
@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    // Bounds check (Safari is strict about this)
    if (idx >= params.num_samples || idx == 0u) {
        return;
    }

    let channel = idx % params.num_channels;
    let sample = idx / params.num_channels;

    // Simple indexing (avoid complex expressions)
    let curr_idx = sample * params.num_channels + channel;
    let prev_idx = (sample - 1u) * params.num_channels + channel;

    let curr = input[curr_idx];
    let prev = input[prev_idx];
    let thresh = params.threshold;

    // Explicit conditions (Safari optimizer works better with this)
    var spike: i32 = 0;
    if (prev < thresh && curr >= thresh) {
        spike = 1;
    } else {
        if (prev >= thresh && curr < thresh) {
            spike = -1;
        }
    }

    // Only write if there's a spike
    if (spike != 0) {
        output[idx] = spike;
        atomicAdd(&spike_count, 1u);
    }
}
"#;

    /// Delta modulation shader for Safari.
    pub const DELTA_MODULATION_SAFARI: &str = r#"
// Safari-compatible delta modulation shader

struct Params {
    delta_threshold: f32,
    num_samples: u32,
    num_channels: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;

    if (idx >= params.num_samples || idx == 0u) {
        return;
    }

    let curr = input[idx];
    let prev = input[idx - 1u];
    let delta = curr - prev;
    let thresh = params.delta_threshold;

    var spike: i32 = 0;
    if (delta > thresh) {
        spike = 1;
    } else {
        if (delta < -thresh) {
            spike = -1;
        }
    }

    output[idx] = spike;
}
"#;

    /// Get shader for current browser.
    pub fn get_level_crossing_shader(is_safari: bool) -> &'static str {
        if is_safari {
            LEVEL_CROSSING_SAFARI
        } else {
            super::STANDARD_LEVEL_CROSSING
        }
    }
}

/// Standard (non-Safari) shaders for comparison.
const STANDARD_LEVEL_CROSSING: &str = r#"
struct Params {
    threshold: f32,
    num_samples: u32,
    num_channels: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@group(0) @binding(2) var<storage, read_write> spike_count: atomic<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_samples || idx == 0u { return; }

    let curr = input[idx];
    let prev = input[idx - 1u];
    let spike = select(
        select(0, -1, prev >= params.threshold && curr < params.threshold),
        1,
        prev < params.threshold && curr >= params.threshold
    );

    if spike != 0 {
        output[idx] = spike;
        atomicAdd(&spike_count, 1u);
    }
}
"#;

/// Safari feature detection utilities.
#[wasm_bindgen]
pub struct SafariFeatureDetector {
    /// Detected features
    features: SafariFeatures,
}

/// Safari-specific feature flags.
#[derive(Debug, Clone, Default)]
struct SafariFeatures {
    /// Has compute shaders
    has_compute: bool,
    /// Has storage buffers
    has_storage_buffers: bool,
    /// Has atomic operations
    has_atomics: bool,
    /// Has f16 support
    has_f16: bool,
    /// Maximum buffer size
    max_buffer_size: usize,
    /// Maximum compute invocations
    max_compute_invocations: u32,
}

#[wasm_bindgen]
impl SafariFeatureDetector {
    /// Create a new feature detector.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            features: SafariFeatures::default(),
        }
    }

    /// Detect features (call after WebGPU initialization).
    pub async fn detect(&mut self) -> Result<(), JsValue> {
        // In actual implementation, would query adapter.features and adapter.limits
        self.features = SafariFeatures {
            has_compute: true,
            has_storage_buffers: true,
            has_atomics: true,
            has_f16: false, // Safari 17 doesn't support f16
            max_buffer_size: 256 * 1024 * 1024, // 256 MB
            max_compute_invocations: 65535,
        };
        Ok(())
    }

    /// Check if compute shaders are available.
    pub fn has_compute(&self) -> bool {
        self.features.has_compute
    }

    /// Check if f16 is available.
    pub fn has_f16(&self) -> bool {
        self.features.has_f16
    }

    /// Get maximum buffer size.
    pub fn max_buffer_size(&self) -> usize {
        self.features.max_buffer_size
    }
}

impl Default for SafariFeatureDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Safari WebGPU error handling.
#[wasm_bindgen]
pub struct SafariErrorHandler {
    /// Last error message
    last_error: Option<String>,
    /// Error count
    error_count: u32,
}

#[wasm_bindgen]
impl SafariErrorHandler {
    /// Create new error handler.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            last_error: None,
            error_count: 0,
        }
    }

    /// Handle a WebGPU error.
    pub fn handle_error(&mut self, error: &str) -> String {
        self.error_count += 1;
        self.last_error = Some(error.to_string());

        // Parse Safari-specific error messages and provide guidance
        if error.contains("Shader compilation failed") {
            return "Safari shader compilation error. Try using compatibility shaders.".to_string();
        }
        if error.contains("Buffer size exceeds") {
            return "Buffer too large for Safari. Reduce batch size.".to_string();
        }
        if error.contains("Workgroup size") {
            return "Workgroup size too large for Safari. Use max 256.".to_string();
        }

        format!("Safari WebGPU error: {}", error)
    }

    /// Get last error.
    pub fn last_error(&self) -> Option<String> {
        self.last_error.clone()
    }

    /// Get error count.
    pub fn error_count(&self) -> u32 {
        self.error_count
    }

    /// Check if fallback to WASM is recommended.
    pub fn should_fallback(&self) -> bool {
        self.error_count >= 3
    }
}

impl Default for SafariErrorHandler {
    fn default() -> Self {
        Self::new()
    }
}

/// Safari performance hints.
#[wasm_bindgen]
pub struct SafariPerformanceHints {
    /// Recommended batch size
    recommended_batch_size: usize,
    /// Recommended workgroup size
    recommended_workgroup_size: u32,
    /// Use explicit synchronization
    use_explicit_sync: bool,
}

#[wasm_bindgen]
impl SafariPerformanceHints {
    /// Get performance hints for Safari.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            recommended_batch_size: 1024,
            recommended_workgroup_size: 64,
            use_explicit_sync: true,
        }
    }

    /// Get hints for Safari on Apple Silicon.
    pub fn for_apple_silicon() -> Self {
        Self {
            recommended_batch_size: 4096,
            recommended_workgroup_size: 256,
            use_explicit_sync: false,
        }
    }

    /// Get hints for Safari on Intel Mac.
    pub fn for_intel_mac() -> Self {
        Self {
            recommended_batch_size: 2048,
            recommended_workgroup_size: 128,
            use_explicit_sync: true,
        }
    }

    /// Get hints for iOS Safari.
    pub fn for_ios() -> Self {
        Self {
            recommended_batch_size: 512,
            recommended_workgroup_size: 64,
            use_explicit_sync: true,
        }
    }

    /// Get recommended batch size.
    #[wasm_bindgen(getter)]
    pub fn recommended_batch_size(&self) -> usize {
        self.recommended_batch_size
    }

    /// Get recommended workgroup size.
    #[wasm_bindgen(getter)]
    pub fn recommended_workgroup_size(&self) -> u32 {
        self.recommended_workgroup_size
    }
}

impl Default for SafariPerformanceHints {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safari_config() {
        let config = SafariWebGPUConfig::new();
        assert!(config.use_compat_shaders);
        assert_eq!(config.max_workgroup_size, 256);
    }

    #[test]
    fn test_safari_17_config() {
        let config = SafariWebGPUConfig::for_safari_17();
        assert!(config.use_compat_shaders);
        assert_eq!(config.shader_compile_timeout_ms, 10000);
    }

    #[test]
    fn test_safari_18_config() {
        let config = SafariWebGPUConfig::for_safari_18();
        assert!(!config.use_compat_shaders);
        assert_eq!(config.max_workgroup_size, 512);
    }

    #[test]
    fn test_error_handler() {
        let mut handler = SafariErrorHandler::new();

        let result = handler.handle_error("Shader compilation failed");
        assert!(result.contains("compatibility shaders"));

        let result = handler.handle_error("Buffer size exceeds limit");
        assert!(result.contains("Reduce batch size"));
    }

    #[test]
    fn test_fallback_recommendation() {
        let mut handler = SafariErrorHandler::new();
        assert!(!handler.should_fallback());

        handler.handle_error("Error 1");
        handler.handle_error("Error 2");
        assert!(!handler.should_fallback());

        handler.handle_error("Error 3");
        assert!(handler.should_fallback());
    }

    #[test]
    fn test_performance_hints() {
        let hints = SafariPerformanceHints::new();
        assert_eq!(hints.recommended_workgroup_size, 64);

        let silicon = SafariPerformanceHints::for_apple_silicon();
        assert_eq!(silicon.recommended_workgroup_size, 256);

        let ios = SafariPerformanceHints::for_ios();
        assert_eq!(ios.recommended_batch_size, 512);
    }

    #[test]
    fn test_compat_shaders() {
        let safari_shader = compat_shaders::LEVEL_CROSSING_SAFARI;
        assert!(safari_shader.contains("workgroup_size(64"));
        assert!(safari_shader.contains("_padding"));

        let standard_shader = STANDARD_LEVEL_CROSSING;
        assert!(standard_shader.contains("workgroup_size(256"));
    }
}
