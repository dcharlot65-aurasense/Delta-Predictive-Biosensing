//! TimeSeries for WebAssembly

use js_sys::Float32Array;
use wasm_bindgen::prelude::*;

/// Multi-channel time series data for browser-based processing.
///
/// Stores continuous biosignal data with a specified sample rate.
/// Data is stored internally as f32 for efficient WASM memory usage.
#[wasm_bindgen]
pub struct WasmTimeSeries {
    data: Vec<f32>,
    num_samples: usize,
    num_channels: usize,
    sample_rate: f64,
}

#[wasm_bindgen]
impl WasmTimeSeries {
    /// Create a new TimeSeries from a Float32Array.
    ///
    /// # Arguments
    /// * `data` - Interleaved signal data [s0_c0, s0_c1, ..., s1_c0, s1_c1, ...]
    /// * `num_samples` - Number of samples per channel
    /// * `num_channels` - Number of channels
    /// * `sample_rate` - Sample rate in Hz
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const data = new Float32Array([1.0, 2.0, 3.0, 4.0]); // 2 samples, 2 channels
    /// const ts = new WasmTimeSeries(data, 2, 2, 1000.0);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(
        data: Float32Array,
        num_samples: usize,
        num_channels: usize,
        sample_rate: f64,
    ) -> Result<WasmTimeSeries, JsValue> {
        let expected_len = num_samples * num_channels;
        if data.length() as usize != expected_len {
            return Err(JsValue::from_str(&format!(
                "Data length {} doesn't match expected {} (samples={} x channels={})",
                data.length(),
                expected_len,
                num_samples,
                num_channels
            )));
        }

        if sample_rate <= 0.0 {
            return Err(JsValue::from_str("Sample rate must be positive"));
        }

        Ok(Self {
            data: data.to_vec(),
            num_samples,
            num_channels,
            sample_rate,
        })
    }

    /// Create a TimeSeries from separate channel arrays.
    ///
    /// # Arguments
    /// * `channels` - Array of Float32Array, one per channel
    /// * `sample_rate` - Sample rate in Hz
    #[wasm_bindgen(js_name = "fromChannels")]
    pub fn from_channels(
        channels: Vec<Float32Array>,
        sample_rate: f64,
    ) -> Result<WasmTimeSeries, JsValue> {
        if channels.is_empty() {
            return Err(JsValue::from_str("No channels provided"));
        }

        let num_channels = channels.len();
        let num_samples = channels[0].length() as usize;

        // Verify all channels have same length
        for (i, ch) in channels.iter().enumerate() {
            if ch.length() as usize != num_samples {
                return Err(JsValue::from_str(&format!(
                    "Channel {} has different length ({}) than channel 0 ({})",
                    i,
                    ch.length(),
                    num_samples
                )));
            }
        }

        // Interleave channel data
        let mut data = vec![0.0f32; num_samples * num_channels];
        for (ch_idx, channel) in channels.iter().enumerate() {
            let ch_data = channel.to_vec();
            for (s_idx, &value) in ch_data.iter().enumerate() {
                data[s_idx * num_channels + ch_idx] = value;
            }
        }

        if sample_rate <= 0.0 {
            return Err(JsValue::from_str("Sample rate must be positive"));
        }

        Ok(Self {
            data,
            num_samples,
            num_channels,
            sample_rate,
        })
    }

    /// Create from a single-channel array.
    #[wasm_bindgen(js_name = "fromSingleChannel")]
    pub fn from_single_channel(
        data: Float32Array,
        sample_rate: f64,
    ) -> Result<WasmTimeSeries, JsValue> {
        let num_samples = data.length() as usize;
        Self::new(data, num_samples, 1, sample_rate)
    }

    /// Duration of the time series in seconds.
    #[wasm_bindgen(getter)]
    pub fn duration(&self) -> f64 {
        self.num_samples as f64 / self.sample_rate
    }

    /// Number of samples.
    #[wasm_bindgen(getter, js_name = "numSamples")]
    pub fn num_samples(&self) -> usize {
        self.num_samples
    }

    /// Number of channels.
    #[wasm_bindgen(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Sample rate in Hz.
    #[wasm_bindgen(getter, js_name = "sampleRate")]
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    /// Get all data as Float32Array (interleaved).
    #[wasm_bindgen(js_name = "getData")]
    pub fn get_data(&self) -> Float32Array {
        Float32Array::from(&self.data[..])
    }

    /// Get data for a single channel.
    #[wasm_bindgen(js_name = "getChannel")]
    pub fn get_channel(&self, channel: usize) -> Result<Float32Array, JsValue> {
        if channel >= self.num_channels {
            return Err(JsValue::from_str(&format!(
                "Channel {} out of bounds (max: {})",
                channel,
                self.num_channels - 1
            )));
        }

        let mut channel_data = Vec::with_capacity(self.num_samples);
        for s in 0..self.num_samples {
            channel_data.push(self.data[s * self.num_channels + channel]);
        }

        Ok(Float32Array::from(&channel_data[..]))
    }

    /// Get a sample at a specific index.
    #[wasm_bindgen(js_name = "getSample")]
    pub fn get_sample(&self, index: usize) -> Result<Float32Array, JsValue> {
        if index >= self.num_samples {
            return Err(JsValue::from_str(&format!(
                "Sample index {} out of bounds (max: {})",
                index,
                self.num_samples - 1
            )));
        }

        let start = index * self.num_channels;
        let end = start + self.num_channels;
        Ok(Float32Array::from(&self.data[start..end]))
    }

    /// Get a slice of the time series.
    #[wasm_bindgen(js_name = "slice")]
    pub fn slice(&self, start_sample: usize, end_sample: usize) -> Result<WasmTimeSeries, JsValue> {
        if start_sample >= end_sample {
            return Err(JsValue::from_str("start must be less than end"));
        }
        if end_sample > self.num_samples {
            return Err(JsValue::from_str("end exceeds number of samples"));
        }

        let num_samples = end_sample - start_sample;
        let start_idx = start_sample * self.num_channels;
        let end_idx = end_sample * self.num_channels;

        Ok(Self {
            data: self.data[start_idx..end_idx].to_vec(),
            num_samples,
            num_channels: self.num_channels,
            sample_rate: self.sample_rate,
        })
    }

    /// Calculate basic statistics for a channel.
    #[wasm_bindgen(js_name = "channelStats")]
    pub fn channel_stats(&self, channel: usize) -> Result<JsValue, JsValue> {
        if channel >= self.num_channels {
            return Err(JsValue::from_str("Channel out of bounds"));
        }

        let mut sum = 0.0f64;
        let mut min = f64::MAX;
        let mut max = f64::MIN;

        for s in 0..self.num_samples {
            let val = self.data[s * self.num_channels + channel] as f64;
            sum += val;
            min = min.min(val);
            max = max.max(val);
        }

        let mean = sum / self.num_samples as f64;

        let mut variance_sum = 0.0f64;
        for s in 0..self.num_samples {
            let val = self.data[s * self.num_channels + channel] as f64;
            variance_sum += (val - mean).powi(2);
        }
        let std = (variance_sum / self.num_samples as f64).sqrt();

        // Return as JavaScript object
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"mean".into(), &mean.into())?;
        js_sys::Reflect::set(&obj, &"std".into(), &std.into())?;
        js_sys::Reflect::set(&obj, &"min".into(), &min.into())?;
        js_sys::Reflect::set(&obj, &"max".into(), &max.into())?;

        Ok(obj.into())
    }
}

// Internal methods for use by encoders
impl WasmTimeSeries {
    pub(crate) fn data_ref(&self) -> &[f32] {
        &self.data
    }
}
