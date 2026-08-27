//! Spike encoders for WebAssembly

use crate::spiketrain::{SpikeEvent, WasmSpikeTrain};
use crate::timeseries::WasmTimeSeries;
use wasm_bindgen::prelude::*;

/// Level Crossing Encoder for browser use.
///
/// Generates spikes when the signal crosses quantization levels.
#[wasm_bindgen]
pub struct WasmLevelCrossingEncoder {
    threshold: f64,
}

#[wasm_bindgen]
impl WasmLevelCrossingEncoder {
    /// Create a new Level Crossing Encoder.
    ///
    /// # Arguments
    /// * `threshold` - Quantization threshold (must be positive)
    ///
    /// # Example (JavaScript)
    /// ```javascript
    /// const encoder = new WasmLevelCrossingEncoder(0.1);
    /// const spikes = encoder.encode(timeSeries);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f64) -> Result<WasmLevelCrossingEncoder, JsValue> {
        if threshold <= 0.0 {
            return Err(JsValue::from_str("Threshold must be positive"));
        }
        Ok(Self { threshold })
    }

    /// Get the threshold value.
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Encode a TimeSeries into a SpikeTrain.
    ///
    /// # Arguments
    /// * `ts` - TimeSeries to encode
    ///
    /// # Returns
    /// SpikeTrain containing the encoded spikes
    #[wasm_bindgen]
    pub fn encode(&self, ts: &WasmTimeSeries) -> Result<WasmSpikeTrain, JsValue> {
        let data = ts.data_ref();
        let num_samples = ts.num_samples();
        let num_channels = ts.num_channels();
        let sample_rate = ts.sample_rate();

        let mut spike_train = WasmSpikeTrain::new(num_channels as u32)?;

        // Track quantization level per channel
        let mut last_levels: Vec<i32> = vec![0; num_channels];

        // Initialize levels from first sample
        for ch in 0..num_channels {
            let value = data[ch] as f64;
            last_levels[ch] = (value / self.threshold).floor() as i32;
        }

        // Process all samples
        for s in 1..num_samples {
            let time = s as f64 / sample_rate;

            for ch in 0..num_channels {
                let value = data[s * num_channels + ch] as f64;
                let level = (value / self.threshold).floor() as i32;

                // Check for level crossing
                if level != last_levels[ch] {
                    // Generate spikes for each level crossed
                    let diff = level - last_levels[ch];
                    let polarity: i8 = if diff > 0 { 1 } else { -1 };
                    let num_spikes = diff.abs();

                    for _ in 0..num_spikes {
                        spike_train.push_event(SpikeEvent {
                            timestamp: time,
                            channel: ch as u32,
                            polarity,
                            magnitude: 1.0,
                        });
                    }

                    last_levels[ch] = level;
                }
            }
        }

        Ok(spike_train)
    }

    /// Encode with adaptive threshold based on signal statistics.
    #[wasm_bindgen(js_name = "encodeAdaptive")]
    pub fn encode_adaptive(
        &self,
        ts: &WasmTimeSeries,
        adaptation_factor: f64,
    ) -> Result<WasmSpikeTrain, JsValue> {
        let data = ts.data_ref();
        let num_samples = ts.num_samples();
        let num_channels = ts.num_channels();
        let sample_rate = ts.sample_rate();

        let mut spike_train = WasmSpikeTrain::new(num_channels as u32)?;

        // Calculate per-channel adaptive thresholds
        let mut thresholds = vec![self.threshold; num_channels];

        for ch in 0..num_channels {
            // Calculate channel standard deviation
            let mut sum = 0.0f64;
            let mut sum_sq = 0.0f64;

            for s in 0..num_samples {
                let val = data[s * num_channels + ch] as f64;
                sum += val;
                sum_sq += val * val;
            }

            let mean = sum / num_samples as f64;
            let variance = (sum_sq / num_samples as f64) - (mean * mean);
            let std = variance.sqrt();

            // Adapt threshold based on signal variance
            thresholds[ch] = self.threshold * (1.0 + adaptation_factor * std);
        }

        // Encode with adaptive thresholds
        let mut last_levels: Vec<i32> = vec![0; num_channels];

        for ch in 0..num_channels {
            let value = data[ch] as f64;
            last_levels[ch] = (value / thresholds[ch]).floor() as i32;
        }

        for s in 1..num_samples {
            let time = s as f64 / sample_rate;

            for ch in 0..num_channels {
                let value = data[s * num_channels + ch] as f64;
                let level = (value / thresholds[ch]).floor() as i32;

                if level != last_levels[ch] {
                    let polarity: i8 = if level > last_levels[ch] { 1 } else { -1 };
                    spike_train.push_event(SpikeEvent {
                        timestamp: time,
                        channel: ch as u32,
                        polarity,
                        magnitude: 1.0,
                    });
                    last_levels[ch] = level;
                }
            }
        }

        Ok(spike_train)
    }
}

/// Delta Encoder for browser use.
///
/// Generates spikes when the change from a reference exceeds the threshold.
#[wasm_bindgen]
pub struct WasmDeltaEncoder {
    threshold: f64,
}

#[wasm_bindgen]
impl WasmDeltaEncoder {
    /// Create a new Delta Encoder.
    ///
    /// # Arguments
    /// * `threshold` - Delta threshold (must be positive)
    #[wasm_bindgen(constructor)]
    pub fn new(threshold: f64) -> Result<WasmDeltaEncoder, JsValue> {
        if threshold <= 0.0 {
            return Err(JsValue::from_str("Threshold must be positive"));
        }
        Ok(Self { threshold })
    }

    /// Get the threshold value.
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f64 {
        self.threshold
    }

    /// Encode a TimeSeries into a SpikeTrain.
    #[wasm_bindgen]
    pub fn encode(&self, ts: &WasmTimeSeries) -> Result<WasmSpikeTrain, JsValue> {
        let data = ts.data_ref();
        let num_samples = ts.num_samples();
        let num_channels = ts.num_channels();
        let sample_rate = ts.sample_rate();

        let mut spike_train = WasmSpikeTrain::new(num_channels as u32)?;

        // Reference values per channel (start at first sample)
        let mut references: Vec<f32> = (0..num_channels).map(|ch| data[ch]).collect();

        for s in 1..num_samples {
            let time = s as f64 / sample_rate;

            for ch in 0..num_channels {
                let value = data[s * num_channels + ch];
                let delta = value - references[ch];

                if (delta as f64).abs() >= self.threshold {
                    let polarity: i8 = if delta > 0.0 { 1 } else { -1 };
                    spike_train.push_event(SpikeEvent {
                        timestamp: time,
                        channel: ch as u32,
                        polarity,
                        magnitude: delta.abs(),
                    });
                    references[ch] = value;
                }
            }
        }

        Ok(spike_train)
    }
}

/// Temporal Contrast Encoder.
///
/// Generates spikes based on temporal derivatives of the signal.
#[wasm_bindgen]
pub struct WasmTemporalContrastEncoder {
    threshold: f64,
    refractory_samples: usize,
}

#[wasm_bindgen]
impl WasmTemporalContrastEncoder {
    /// Create a new Temporal Contrast Encoder.
    ///
    /// # Arguments
    /// * `threshold` - Derivative threshold
    /// * `refractory_ms` - Refractory period in milliseconds
    /// * `sample_rate` - Sample rate for refractory calculation
    #[wasm_bindgen(constructor)]
    pub fn new(
        threshold: f64,
        refractory_ms: f64,
        sample_rate: f64,
    ) -> Result<WasmTemporalContrastEncoder, JsValue> {
        if threshold <= 0.0 {
            return Err(JsValue::from_str("Threshold must be positive"));
        }

        let refractory_samples = (refractory_ms / 1000.0 * sample_rate).ceil() as usize;

        Ok(Self {
            threshold,
            refractory_samples,
        })
    }

    /// Encode a TimeSeries into a SpikeTrain.
    #[wasm_bindgen]
    pub fn encode(&self, ts: &WasmTimeSeries) -> Result<WasmSpikeTrain, JsValue> {
        let data = ts.data_ref();
        let num_samples = ts.num_samples();
        let num_channels = ts.num_channels();
        let sample_rate = ts.sample_rate();

        let mut spike_train = WasmSpikeTrain::new(num_channels as u32)?;

        // Track refractory periods per channel
        let mut refractory_counters: Vec<usize> = vec![0; num_channels];

        for s in 1..num_samples {
            let time = s as f64 / sample_rate;

            for ch in 0..num_channels {
                // Decrement refractory counter
                if refractory_counters[ch] > 0 {
                    refractory_counters[ch] -= 1;
                    continue;
                }

                // Calculate temporal derivative
                let prev = data[(s - 1) * num_channels + ch] as f64;
                let curr = data[s * num_channels + ch] as f64;
                let derivative = (curr - prev) * sample_rate; // Scale by sample rate

                if derivative.abs() >= self.threshold {
                    let polarity: i8 = if derivative > 0.0 { 1 } else { -1 };
                    spike_train.push_event(SpikeEvent {
                        timestamp: time,
                        channel: ch as u32,
                        polarity,
                        magnitude: derivative.abs() as f32,
                    });
                    refractory_counters[ch] = self.refractory_samples;
                }
            }
        }

        Ok(spike_train)
    }
}
