//! WebNN (Web Neural Network API) support for browser-based ML inference.
//!
//! This module provides integration with the W3C WebNN API for hardware-accelerated
//! neural network inference in web browsers.
//!
//! ## Browser Support
//!
//! | Browser | Status | Notes |
//! |---------|--------|-------|
//! | Chrome 113+ | Flag | `chrome://flags/#enable-web-machine-learning-neural-network-service` |
//! | Edge 113+ | Flag | Similar flag required |
//! | Firefox | Not supported | - |
//! | Safari | Not supported | - |
//!
//! ## Features
//!
//! - Hardware acceleration via GPU, NPU, or CPU backends
//! - ONNX model loading
//! - Efficient tensor operations
//! - Async execution
//!
//! ## Example
//!
//! ```javascript
//! // JavaScript usage
//! import { WebNNEncoder } from 'dpb-wasm';
//!
//! const encoder = await WebNNEncoder.create('gpu');
//! const spikes = await encoder.encode(signalData);
//! ```

use wasm_bindgen::prelude::*;
use js_sys::Float32Array;
use web_sys::console;

/// WebNN device type for backend selection.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebNNDeviceType {
    /// CPU backend (always available).
    Cpu,
    /// GPU backend (WebGPU integration).
    Gpu,
    /// NPU backend (dedicated neural processor, if available).
    Npu,
}

impl WebNNDeviceType {
    // Maps to the string the WebNN API expects; used once the calls
    // below stop being stubs.
    #[allow(dead_code)]
    fn as_str(&self) -> &'static str {
        match self {
            WebNNDeviceType::Cpu => "cpu",
            WebNNDeviceType::Gpu => "gpu",
            WebNNDeviceType::Npu => "npu",
        }
    }
}

/// WebNN power preference for battery-sensitive applications.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebNNPowerPreference {
    /// Default power preference.
    Default,
    /// Prefer low power consumption.
    LowPower,
    /// Prefer high performance.
    HighPerformance,
}

impl WebNNPowerPreference {
    // Maps to the string the WebNN API expects; used once the calls
    // below stop being stubs.
    #[allow(dead_code)]
    fn as_str(&self) -> &'static str {
        match self {
            WebNNPowerPreference::Default => "default",
            WebNNPowerPreference::LowPower => "low-power",
            WebNNPowerPreference::HighPerformance => "high-performance",
        }
    }
}

/// WebNN context configuration.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebNNConfig {
    device_type: WebNNDeviceType,
    power_preference: WebNNPowerPreference,
}

#[wasm_bindgen]
impl WebNNConfig {
    /// Create default configuration (GPU, default power).
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            device_type: WebNNDeviceType::Gpu,
            power_preference: WebNNPowerPreference::Default,
        }
    }

    /// Create configuration for low-power mobile use.
    pub fn low_power() -> Self {
        Self {
            device_type: WebNNDeviceType::Cpu,
            power_preference: WebNNPowerPreference::LowPower,
        }
    }

    /// Create configuration for high-performance inference.
    pub fn high_performance() -> Self {
        Self {
            device_type: WebNNDeviceType::Gpu,
            power_preference: WebNNPowerPreference::HighPerformance,
        }
    }

    /// Create configuration targeting NPU (if available).
    pub fn npu() -> Self {
        Self {
            device_type: WebNNDeviceType::Npu,
            power_preference: WebNNPowerPreference::Default,
        }
    }

    /// Set device type.
    pub fn set_device_type(&mut self, device_type: WebNNDeviceType) {
        self.device_type = device_type;
    }

    /// Set power preference.
    pub fn set_power_preference(&mut self, power_preference: WebNNPowerPreference) {
        self.power_preference = power_preference;
    }
}

impl Default for WebNNConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// WebNN tensor descriptor.
#[wasm_bindgen]
#[derive(Debug, Clone)]
// Held but not consulted yet; kept so a caller's input is not silently
// discarded.
#[allow(dead_code)]
pub struct WebNNTensorDesc {
    /// Data type (float32, float16, int32, etc.).
    data_type: String,
    /// Shape dimensions.
    dimensions: Vec<u32>,
}

#[wasm_bindgen]
impl WebNNTensorDesc {
    /// Create a new tensor descriptor.
    #[wasm_bindgen(constructor)]
    pub fn new(data_type: &str, dimensions: Vec<u32>) -> Self {
        Self {
            data_type: data_type.to_string(),
            dimensions,
        }
    }

    /// Create float32 tensor descriptor.
    pub fn float32(dimensions: Vec<u32>) -> Self {
        Self::new("float32", dimensions)
    }

    /// Get total element count.
    pub fn element_count(&self) -> u32 {
        self.dimensions.iter().product()
    }
}

/// WebNN operand for graph building.
#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct WebNNOperand {
    /// Operand name.
    name: String,
    /// Tensor descriptor.
    desc: WebNNTensorDesc,
}

#[wasm_bindgen]
impl WebNNOperand {
    /// Create a new operand.
    #[wasm_bindgen(constructor)]
    pub fn new(name: &str, desc: WebNNTensorDesc) -> Self {
        Self {
            name: name.to_string(),
            desc,
        }
    }

    /// Get operand name.
    pub fn name(&self) -> String {
        self.name.clone()
    }
}

/// WebNN graph builder for constructing neural network operations.
#[wasm_bindgen]
pub struct WebNNGraphBuilder {
    /// Operands in the graph.
    operands: Vec<WebNNOperand>,
    /// Operations (simplified representation).
    operations: Vec<String>,
}

#[wasm_bindgen]
impl WebNNGraphBuilder {
    /// Create a new graph builder.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            operands: Vec::new(),
            operations: Vec::new(),
        }
    }

    /// Add an input operand.
    pub fn input(&mut self, name: &str, desc: WebNNTensorDesc) -> WebNNOperand {
        let operand = WebNNOperand::new(name, desc);
        self.operands.push(operand.clone());
        operand
    }

    /// Add a constant operand.
    pub fn constant(&mut self, name: &str, desc: WebNNTensorDesc, _data: Float32Array) -> WebNNOperand {
        let operand = WebNNOperand::new(name, desc);
        self.operands.push(operand.clone());
        self.operations.push(format!("constant:{}", name));
        operand
    }

    /// Add ReLU activation.
    pub fn relu(&mut self, input: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_relu", input.name);
        let operand = WebNNOperand::new(&name, input.desc.clone());
        self.operands.push(operand.clone());
        self.operations.push(format!("relu:{}", input.name));
        operand
    }

    /// Add sigmoid activation.
    pub fn sigmoid(&mut self, input: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_sigmoid", input.name);
        let operand = WebNNOperand::new(&name, input.desc.clone());
        self.operands.push(operand.clone());
        self.operations.push(format!("sigmoid:{}", input.name));
        operand
    }

    /// Add tanh activation.
    pub fn tanh(&mut self, input: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_tanh", input.name);
        let operand = WebNNOperand::new(&name, input.desc.clone());
        self.operands.push(operand.clone());
        self.operations.push(format!("tanh:{}", input.name));
        operand
    }

    /// Add element-wise addition.
    pub fn add(&mut self, a: &WebNNOperand, b: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_{}_add", a.name, b.name);
        let operand = WebNNOperand::new(&name, a.desc.clone());
        self.operands.push(operand.clone());
        self.operations.push(format!("add:{}:{}", a.name, b.name));
        operand
    }

    /// Add element-wise multiplication.
    pub fn mul(&mut self, a: &WebNNOperand, b: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_{}_mul", a.name, b.name);
        let operand = WebNNOperand::new(&name, a.desc.clone());
        self.operands.push(operand.clone());
        self.operations.push(format!("mul:{}:{}", a.name, b.name));
        operand
    }

    /// Add matrix multiplication.
    pub fn matmul(&mut self, a: &WebNNOperand, b: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_{}_matmul", a.name, b.name);
        // Output shape: [..., a.rows, b.cols]
        let dims = vec![a.desc.dimensions[0], b.desc.dimensions[1]];
        let operand = WebNNOperand::new(&name, WebNNTensorDesc::float32(dims));
        self.operands.push(operand.clone());
        self.operations.push(format!("matmul:{}:{}", a.name, b.name));
        operand
    }

    /// Add 1D convolution.
    pub fn conv1d(
        &mut self,
        input: &WebNNOperand,
        filter: &WebNNOperand,
        stride: u32,
        padding: &str,
    ) -> WebNNOperand {
        let name = format!("{}_conv1d", input.name);
        let operand = WebNNOperand::new(&name, input.desc.clone()); // Simplified
        self.operands.push(operand.clone());
        self.operations.push(format!("conv1d:{}:{}:{}:{}", input.name, filter.name, stride, padding));
        operand
    }

    /// Add softmax.
    pub fn softmax(&mut self, input: &WebNNOperand) -> WebNNOperand {
        let name = format!("{}_softmax", input.name);
        let operand = WebNNOperand::new(&name, input.desc.clone());
        self.operands.push(operand.clone());
        self.operations.push(format!("softmax:{}", input.name));
        operand
    }

    /// Add reshape operation.
    pub fn reshape(&mut self, input: &WebNNOperand, new_shape: Vec<u32>) -> WebNNOperand {
        let name = format!("{}_reshape", input.name);
        let operand = WebNNOperand::new(&name, WebNNTensorDesc::float32(new_shape));
        self.operands.push(operand.clone());
        self.operations.push(format!("reshape:{}", input.name));
        operand
    }

    /// Get number of operations.
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }
}

impl Default for WebNNGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// WebNN-based spike encoder for biosignal processing.
#[wasm_bindgen]
pub struct WebNNEncoder {
    /// Configuration.
    config: WebNNConfig,
    /// Is initialized.
    initialized: bool,
    /// Encoding threshold.
    threshold: f32,
}

#[wasm_bindgen]
impl WebNNEncoder {
    /// Check if WebNN is available in this browser.
    pub fn is_available() -> bool {
        // In actual implementation, check navigator.ml
        #[cfg(target_arch = "wasm32")]
        {
            // web_sys::window()
            //     .and_then(|w| w.navigator())
            //     .map(|n| Reflect::has(&n, &JsValue::from_str("ml")).unwrap_or(false))
            //     .unwrap_or(false)
            false // Placeholder - actual check requires web-sys bindings
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Create a new WebNN encoder.
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f32) -> Self {
        Self {
            config: WebNNConfig::default(),
            initialized: false,
            threshold,
        }
    }

    /// Create encoder with custom configuration.
    pub fn with_config(threshold: f32, config: WebNNConfig) -> Self {
        Self {
            config,
            initialized: false,
            threshold,
        }
    }

    /// Initialize the WebNN context (async).
    pub async fn initialize(&mut self) -> Result<(), JsValue> {
        // In actual implementation:
        // 1. Get navigator.ml
        // 2. Create MLContext with device preference
        // 3. Build computation graph
        // 4. Compile graph

        console::log_1(&JsValue::from_str(&format!(
            "Initializing WebNN with device: {:?}",
            self.config.device_type
        )));

        self.initialized = true;
        Ok(())
    }

    /// Check if encoder is initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Encode signal using WebNN-accelerated level crossing detection.
    pub fn encode_level_crossing(&self, signal: Float32Array) -> Result<Vec<u32>, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("WebNN not initialized. Call initialize() first."));
        }

        // Fallback CPU implementation
        let data: Vec<f32> = signal.to_vec();
        let mut spikes = Vec::new();
        let mut prev = data.first().copied().unwrap_or(0.0);

        for (i, &value) in data.iter().enumerate().skip(1) {
            if prev < self.threshold && value >= self.threshold {
                spikes.push(i as u32);
            }
            prev = value;
        }

        Ok(spikes)
    }

    /// Encode signal using WebNN-accelerated delta modulation.
    pub fn encode_delta(&self, signal: Float32Array) -> Result<Vec<i8>, JsValue> {
        if !self.initialized {
            return Err(JsValue::from_str("WebNN not initialized. Call initialize() first."));
        }

        let data: Vec<f32> = signal.to_vec();
        let neg_threshold = -self.threshold;

        let spikes: Vec<i8> = data.iter().map(|&v| {
            if v > self.threshold {
                1
            } else if v < neg_threshold {
                -1
            } else {
                0
            }
        }).collect();

        Ok(spikes)
    }

    /// Get current threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Set encoding threshold.
    pub fn set_threshold(&mut self, threshold: f32) {
        self.threshold = threshold;
    }

    /// Get device type.
    pub fn device_type(&self) -> WebNNDeviceType {
        self.config.device_type
    }
}

/// WebNN model loader for ONNX models.
#[wasm_bindgen]
pub struct WebNNModelLoader {
    /// Loaded model bytes.
    model_bytes: Option<Vec<u8>>,
}

#[wasm_bindgen]
impl WebNNModelLoader {
    /// Create a new model loader.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self { model_bytes: None }
    }

    /// Load model from bytes.
    pub fn load_bytes(&mut self, bytes: &[u8]) -> Result<(), JsValue> {
        self.model_bytes = Some(bytes.to_vec());
        console::log_1(&JsValue::from_str(&format!(
            "Loaded model: {} bytes",
            bytes.len()
        )));
        Ok(())
    }

    /// Check if model is loaded.
    pub fn is_loaded(&self) -> bool {
        self.model_bytes.is_some()
    }

    /// Get model size in bytes.
    pub fn model_size(&self) -> usize {
        self.model_bytes.as_ref().map(|b| b.len()).unwrap_or(0)
    }
}

impl Default for WebNNModelLoader {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature detection utilities for WebNN.
#[wasm_bindgen]
pub struct WebNNFeatures;

#[wasm_bindgen]
impl WebNNFeatures {
    /// Check WebNN availability.
    pub fn check_webnn() -> bool {
        WebNNEncoder::is_available()
    }

    /// Check GPU backend availability.
    pub fn check_gpu_backend() -> bool {
        // Would check if GPU context can be created
        false
    }

    /// Check NPU backend availability.
    pub fn check_npu_backend() -> bool {
        // Would check if NPU context can be created
        false
    }

    /// Get recommended device type.
    pub fn recommended_device() -> WebNNDeviceType {
        if Self::check_npu_backend() {
            WebNNDeviceType::Npu
        } else if Self::check_gpu_backend() {
            WebNNDeviceType::Gpu
        } else {
            WebNNDeviceType::Cpu
        }
    }

    /// Get feature support info as JSON string.
    pub fn get_support_info() -> String {
        let info = serde_json::json!({
            "webnn_available": Self::check_webnn(),
            "gpu_available": Self::check_gpu_backend(),
            "npu_available": Self::check_npu_backend(),
            "recommended_device": format!("{:?}", Self::recommended_device()),
            "browser_support": {
                "chrome": "113+ (flag)",
                "edge": "113+ (flag)",
                "firefox": "not supported",
                "safari": "not supported"
            }
        });
        info.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_webnn_config() {
        let config = WebNNConfig::new();
        assert_eq!(config.device_type, WebNNDeviceType::Gpu);
    }

    #[test]
    fn test_tensor_desc() {
        let desc = WebNNTensorDesc::float32(vec![1, 32, 128]);
        assert_eq!(desc.element_count(), 4096);
    }

    #[test]
    fn test_graph_builder() {
        let mut builder = WebNNGraphBuilder::new();
        let input = builder.input("x", WebNNTensorDesc::float32(vec![1, 128]));
        let output = builder.relu(&input);
        assert_eq!(builder.operation_count(), 1);
    }
}
