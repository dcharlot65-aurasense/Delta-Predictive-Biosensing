//! SpikeTrain for WebAssembly

use wasm_bindgen::prelude::*;
use serde::{Deserialize, Serialize};

/// A single spike event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SpikeEvent {
    pub timestamp: f64,
    pub channel: u32,
    pub polarity: i8,
    pub magnitude: f32,
}

/// Collection of spike events for browser-based processing.
#[wasm_bindgen]
pub struct WasmSpikeTrain {
    events: Vec<SpikeEvent>,
    num_channels: u32,
}

#[wasm_bindgen]
impl WasmSpikeTrain {
    /// Create a new empty SpikeTrain.
    ///
    /// # Arguments
    /// * `num_channels` - Number of channels/neurons
    #[wasm_bindgen(constructor)]
    pub fn new(num_channels: u32) -> Result<WasmSpikeTrain, JsValue> {
        if num_channels == 0 {
            return Err(JsValue::from_str("num_channels must be positive"));
        }

        Ok(Self {
            events: Vec::new(),
            num_channels,
        })
    }

    /// Number of spike events.
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.events.len()
    }

    /// Number of channels.
    #[wasm_bindgen(getter, js_name = "numChannels")]
    pub fn num_channels(&self) -> u32 {
        self.num_channels
    }

    /// Check if the spike train is empty.
    #[wasm_bindgen(getter, js_name = "isEmpty")]
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Add a spike event.
    ///
    /// # Arguments
    /// * `timestamp` - Time of spike in seconds
    /// * `channel` - Channel index (0-based)
    /// * `polarity` - +1 (upward) or -1 (downward)
    #[wasm_bindgen(js_name = "addEvent")]
    pub fn add_event(
        &mut self,
        timestamp: f64,
        channel: u32,
        polarity: i8,
    ) -> Result<(), JsValue> {
        if channel >= self.num_channels {
            return Err(JsValue::from_str(&format!(
                "Channel {} out of bounds (max: {})",
                channel,
                self.num_channels - 1
            )));
        }

        if polarity != 1 && polarity != -1 {
            return Err(JsValue::from_str("Polarity must be +1 or -1"));
        }

        self.events.push(SpikeEvent {
            timestamp,
            channel,
            polarity,
            magnitude: 1.0,
        });

        Ok(())
    }

    /// Add a spike event with magnitude.
    #[wasm_bindgen(js_name = "addEventWithMagnitude")]
    pub fn add_event_with_magnitude(
        &mut self,
        timestamp: f64,
        channel: u32,
        polarity: i8,
        magnitude: f32,
    ) -> Result<(), JsValue> {
        if channel >= self.num_channels {
            return Err(JsValue::from_str("Channel out of bounds"));
        }

        self.events.push(SpikeEvent {
            timestamp,
            channel,
            polarity,
            magnitude,
        });

        Ok(())
    }

    /// Get all events as a JavaScript array of objects.
    #[wasm_bindgen(js_name = "getEvents")]
    pub fn get_events(&self) -> Result<JsValue, JsValue> {
        serde_wasm_bindgen::to_value(&self.events).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get timestamps as Float64Array.
    #[wasm_bindgen(js_name = "getTimestamps")]
    pub fn get_timestamps(&self) -> js_sys::Float64Array {
        let timestamps: Vec<f64> = self.events.iter().map(|e| e.timestamp).collect();
        js_sys::Float64Array::from(&timestamps[..])
    }

    /// Get channels as Uint32Array.
    #[wasm_bindgen(js_name = "getChannels")]
    pub fn get_channels(&self) -> js_sys::Uint32Array {
        let channels: Vec<u32> = self.events.iter().map(|e| e.channel).collect();
        js_sys::Uint32Array::from(&channels[..])
    }

    /// Get polarities as Int8Array.
    #[wasm_bindgen(js_name = "getPolarities")]
    pub fn get_polarities(&self) -> js_sys::Int8Array {
        let polarities: Vec<i8> = self.events.iter().map(|e| e.polarity).collect();
        js_sys::Int8Array::from(&polarities[..])
    }

    /// Get spikes for a specific channel.
    #[wasm_bindgen(js_name = "getChannelSpikes")]
    pub fn get_channel_spikes(&self, channel: u32) -> Result<JsValue, JsValue> {
        let channel_events: Vec<&SpikeEvent> = self
            .events
            .iter()
            .filter(|e| e.channel == channel)
            .collect();

        serde_wasm_bindgen::to_value(&channel_events)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Get spikes in a time range.
    #[wasm_bindgen(js_name = "getSpikesInRange")]
    pub fn get_spikes_in_range(&self, start: f64, end: f64) -> Result<JsValue, JsValue> {
        let range_events: Vec<&SpikeEvent> = self
            .events
            .iter()
            .filter(|e| e.timestamp >= start && e.timestamp < end)
            .collect();

        serde_wasm_bindgen::to_value(&range_events)
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Calculate spike rate (spikes per second).
    #[wasm_bindgen(js_name = "spikeRate")]
    pub fn spike_rate(&self) -> f64 {
        if self.events.is_empty() {
            return 0.0;
        }

        let min_t = self.events.iter().map(|e| e.timestamp).fold(f64::MAX, f64::min);
        let max_t = self.events.iter().map(|e| e.timestamp).fold(f64::MIN, f64::max);

        let duration = max_t - min_t;
        if duration <= 0.0 {
            return 0.0;
        }

        self.events.len() as f64 / duration
    }

    /// Count spikes per channel.
    #[wasm_bindgen(js_name = "spikesPerChannel")]
    pub fn spikes_per_channel(&self) -> js_sys::Uint32Array {
        let mut counts = vec![0u32; self.num_channels as usize];
        for event in &self.events {
            if (event.channel as usize) < counts.len() {
                counts[event.channel as usize] += 1;
            }
        }
        js_sys::Uint32Array::from(&counts[..])
    }

    /// Sort events by timestamp.
    #[wasm_bindgen(js_name = "sortByTime")]
    pub fn sort_by_time(&mut self) {
        self.events.sort_by(|a, b| a.timestamp.total_cmp(&b.timestamp));
    }

    /// Clear all events.
    #[wasm_bindgen]
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Get time range [start, end].
    #[wasm_bindgen(js_name = "timeRange")]
    pub fn time_range(&self) -> Result<JsValue, JsValue> {
        if self.events.is_empty() {
            let arr = js_sys::Array::new();
            arr.push(&0.0.into());
            arr.push(&0.0.into());
            return Ok(arr.into());
        }

        let min_t = self.events.iter().map(|e| e.timestamp).fold(f64::MAX, f64::min);
        let max_t = self.events.iter().map(|e| e.timestamp).fold(f64::MIN, f64::max);

        let arr = js_sys::Array::new();
        arr.push(&min_t.into());
        arr.push(&max_t.into());
        Ok(arr.into())
    }
}

// Internal methods
impl WasmSpikeTrain {
    pub(crate) fn push_event(&mut self, event: SpikeEvent) {
        self.events.push(event);
    }
}
