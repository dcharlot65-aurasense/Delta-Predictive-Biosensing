# DPB LabVIEW Examples

This directory contains example VIs and code snippets demonstrating DPB integration with LabVIEW.

## Example Overview

| Example | Description | Difficulty |
|---------|-------------|------------|
| [Basic Encoding](#1-basic-encoding) | Single-channel level crossing | Beginner |
| [Multi-Channel](#2-multi-channel-encoding) | 8-channel parallel encoding | Beginner |
| [Real-Time DAQ](#3-real-time-daq-integration) | NI-DAQmx continuous acquisition | Intermediate |
| [Spike Visualization](#4-spike-raster-display) | Real-time raster plot | Intermediate |
| [RT System](#5-compactrio-rt) | Deterministic encoding on RT | Advanced |
| [LSL Stream](#6-lsl-integration) | Lab Streaming Layer output | Advanced |

---

## 1. Basic Encoding

**File:** `BasicEncoding.vi`

### Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  [Waveform]     [DBL 2D]     [TimeSeries]     [SpikeTrain]     │
│  Simulate    →  Reshape   →  Create       →   Encode           │
│                                                                 │
│       │              │             │               │            │
│       ▼              ▼             ▼               ▼            │
│  ┌────────┐    ┌─────────┐   ┌─────────┐    ┌──────────┐       │
│  │ 1000   │    │[1000×1] │   │ Handle  │    │ Handle   │       │
│  │samples │    │  Array  │   │ (U64)   │    │ (U64)    │       │
│  │ 256 Hz │    └─────────┘   └─────────┘    └──────────┘       │
│  └────────┘                                                     │
│                                                                 │
│            ┌────────────────────────────────────┐               │
│            │         GetEvents                  │               │
│            ├────────────────────────────────────┤               │
│            │ timestamps: [0.1, 0.3, 0.5, ...]  │               │
│            │ channels:   [0, 0, 0, ...]        │               │
│            │ polarities: [1, -1, 1, ...]       │               │
│            │ count:      42                    │               │
│            └────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### G Code (Pseudo-VI)

```
// Front Panel Controls
Threshold: DBL = 0.1
Sample Rate (Hz): DBL = 256
Num Samples: I32 = 1000

// Generate test signal (sine wave)
signal = Sine Waveform (Num Samples, Sample Rate, freq=10)
data_2d = Reshape Array (signal, [Num Samples, 1])

// Create TimeSeries
ts_handle = Call Library Function (
    "dpb_ffi.dll",
    "dpb_timeseries_new",
    data_2d.pointer, Num Samples, 1, Sample Rate
)

// Create Encoder
encoder_handle = Call Library Function (
    "dpb_ffi.dll",
    "dpb_level_crossing_encoder_new",
    Threshold
)

// Encode
spikes_handle = Call Library Function (
    "dpb_ffi.dll",
    "dpb_encoder_encode",
    encoder_handle, ts_handle
)

// Get Events
timestamps = Array (DBL, max_spikes)
channels = Array (I32, max_spikes)
polarities = Array (I8, max_spikes)
count = Call Library Function (
    "dpb_ffi.dll",
    "dpb_spiketrain_get_events",
    spikes_handle, timestamps.pointer, channels.pointer,
    polarities.pointer, max_spikes
)

// Cleanup
Call Library Function ("dpb_ffi.dll", "dpb_spiketrain_free", spikes_handle)
Call Library Function ("dpb_ffi.dll", "dpb_timeseries_free", ts_handle)
Call Library Function ("dpb_ffi.dll", "dpb_encoder_free", encoder_handle)

// Display
Timestamps Indicator = timestamps[0..count-1]
Spike Count Indicator = count
```

---

## 2. Multi-Channel Encoding

**File:** `MultiChannelEncoding.vi`

### Block Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                     Multi-Channel Encoding                          │
├────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐                                                       │
│  │ 8-Ch DAQ │     ┌─────────────────────────────────┐              │
│  │ Simulate │────▶│ TimeSeries Create               │              │
│  │[1000×8]  │     │ sample_rate: 1000 Hz           │              │
│  └──────────┘     └─────────────────────────────────┘              │
│                              │                                      │
│                              ▼                                      │
│  ┌──────────┐     ┌─────────────────────────────────┐              │
│  │Threshold │────▶│ Level Crossing Encoder          │              │
│  │ Array    │     │ per_channel: True               │              │
│  │[8]       │     └─────────────────────────────────┘              │
│  └──────────┘                │                                      │
│                              ▼                                      │
│                   ┌─────────────────────────────────┐              │
│                   │ For Each Channel                │              │
│                   ├─────────────────────────────────┤              │
│                   │ Ch 0: 42 spikes                 │              │
│                   │ Ch 1: 38 spikes                 │              │
│                   │ Ch 2: 45 spikes                 │              │
│                   │ ...                             │              │
│                   │ Ch 7: 41 spikes                 │              │
│                   └─────────────────────────────────┘              │
│                                                                     │
└────────────────────────────────────────────────────────────────────┘
```

### Per-Channel Threshold Configuration

```
// Per-channel thresholds (adaptive to signal amplitude)
Channel | Signal Type | Threshold
--------|-------------|----------
0       | EEG Fp1     | 0.05 mV
1       | EEG Fp2     | 0.05 mV
2       | EEG C3      | 0.06 mV
3       | EEG C4      | 0.06 mV
4       | ECG Lead I  | 0.50 mV
5       | ECG Lead II | 0.55 mV
6       | EMG         | 0.10 mV
7       | EOG         | 0.08 mV
```

---

## 3. Real-Time DAQ Integration

**File:** `RealtimeDAQ.vi`

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Real-Time DAQ Encoding                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Producer Loop                            │    │
│  │  ┌─────────┐     ┌──────────┐     ┌──────────────┐          │    │
│  │  │ DAQmx   │────▶│ Circular │────▶│ Queue Write  │          │    │
│  │  │ Read    │     │ Buffer   │     │ (Lossless)   │          │    │
│  │  │ 1000 Hz │     │ 10000    │     └──────────────┘          │    │
│  │  └─────────┘     │ samples  │                               │    │
│  │                  └──────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                          │
│                           ▼ Queue                                    │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                     Consumer Loop                            │    │
│  │  ┌──────────────┐   ┌────────────┐   ┌─────────────────┐    │    │
│  │  │ Queue Read   │──▶│ TimeSeries │──▶│ Level Crossing  │    │    │
│  │  │ (100 ms TO)  │   │ Create     │   │ Encode          │    │    │
│  │  └──────────────┘   └────────────┘   └─────────────────┘    │    │
│  │                                              │               │    │
│  │                                              ▼               │    │
│  │                                      ┌───────────────┐      │    │
│  │                                      │ Spike Output  │      │    │
│  │                                      │ - Raster Plot │      │    │
│  │                                      │ - File Log    │      │    │
│  │                                      │ - LSL Stream  │      │    │
│  │                                      └───────────────┘      │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### DAQmx Configuration

```
Task: "Biosignal Acquisition"
Physical Channels: Dev1/ai0:7
Terminal Config: Differential
Sample Rate: 1000 Hz
Samples Per Channel: 100 (100 ms buffers)
Input Range: ±10 V
```

---

## 4. Spike Raster Display

**File:** `SpikeRaster.vi`

### Front Panel

```
┌──────────────────────────────────────────────────────────────────┐
│                      Spike Raster Display                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Time Window: [1.0 ▼] seconds     Channels: 8                    │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Ch 7 ┃  │  │    │ │      │  │    │   │  │   │     │  │    │  │
│  │ Ch 6 ┃   │    │  │ │   │     │  │   │  │      │   │       │  │
│  │ Ch 5 ┃ │   │  │     │  │  │    │  │ │    │  │     │  │    │  │
│  │ Ch 4 ┃    │ │    │    │   │ │      │    │  │   │     │    │  │
│  │ Ch 3 ┃  │    │  │  │    │     │ │    │    │   │  │    │   │  │
│  │ Ch 2 ┃ │  │     │   │  │   │    │  │   │    │      │  │   │  │
│  │ Ch 1 ┃   │   │    │    │ │     │   │  │    │   │    │     │  │
│  │ Ch 0 ┃│    │   │    │     │  │    │   │  │    │  │    │   │  │
│  │      ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━│  │
│  │       0.0                    0.5                      1.0    │  │
│  │                           Time (s)                           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐   │
│  │ Rate: 42.5 Hz   │  │ Total: 340      │  │ ▶ Recording    │   │
│  └─────────────────┘  └─────────────────┘  └────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

### Rendering Logic

```
// XY Graph rendering for raster plot
for each spike in recent_spikes:
    x = spike.timestamp - window_start
    y = spike.channel + 0.5  // Center in channel row

    if spike.polarity > 0:
        color = Green  // Up-crossing
    else:
        color = Red    // Down-crossing

    Plot Point (x, y, color)
```

---

## 5. CompactRIO RT

**File:** `RT_DeterministicEncoding.vi`

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CompactRIO System                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    FPGA (cRIO-9068)                          │   │
│  │  ┌────────────┐   ┌────────────┐   ┌─────────────────────┐  │   │
│  │  │ NI 9205    │──▶│ DMA FIFO   │──▶│ Host Transfer       │  │   │
│  │  │ AI Module  │   │ (RT FIFO)  │   │ (1 kHz trigger)     │  │   │
│  │  └────────────┘   └────────────┘   └─────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                               ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    RT Linux (ARM)                            │   │
│  │  ┌─────────────────────────────────────────────────────────┐│   │
│  │  │              Timed Loop (Priority 50)                   ││   │
│  │  │  ┌────────────┐  ┌────────────┐  ┌─────────────────┐   ││   │
│  │  │  │ RT FIFO    │─▶│ DPB Encode │─▶│ Output FIFO     │   ││   │
│  │  │  │ Read       │  │ (libdpb)   │  │ Write           │   ││   │
│  │  │  └────────────┘  └────────────┘  └─────────────────┘   ││   │
│  │  │                                                         ││   │
│  │  │  Deadline: 1 ms                                        ││   │
│  │  │  Measured: 0.4 ms avg, 0.8 ms max                      ││   │
│  │  └─────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                               │                                      │
│                               ▼ TCP/IP                               │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Host PC (Windows)                         │   │
│  │  ┌─────────────────────────────────────────────────────────┐│   │
│  │  │ Network Stream Read │ Visualization │ File Logging     ││   │
│  │  └─────────────────────────────────────────────────────────┘│   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### RT Optimization Tips

1. **Pre-allocate all arrays** before the timed loop
2. **Use RT FIFOs** instead of queues for determinism
3. **Avoid string operations** in time-critical code
4. **Compile libdpb_ffi.so** for ARM target:
   ```bash
   cross build --release -p dpb-ffi --target aarch64-unknown-linux-gnu
   ```
5. **Set CPU affinity** to isolate encoding from other tasks

---

## 6. LSL Integration

**File:** `LSL_SpikeStream.vi`

### Block Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LSL Spike Streaming                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Initialization                            │    │
│  │  ┌────────────────┐   ┌─────────────────────────────────┐   │    │
│  │  │ LSL Stream     │   │ Stream Info                     │   │    │
│  │  │ Outlet Create  │   │ Name: "DPB_Spikes"              │   │    │
│  │  └────────────────┘   │ Type: "Spikes"                  │   │    │
│  │                       │ Channels: 3 (time, channel, pol)│   │    │
│  │                       │ Format: float32                  │   │    │
│  │                       │ Rate: Irregular                  │   │    │
│  │                       └─────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Main Loop                                 │    │
│  │  ┌──────────────┐  ┌────────────┐  ┌────────────────────┐   │    │
│  │  │ DPB Encode   │─▶│ GetEvents  │─▶│ For Each Spike     │   │    │
│  │  │              │  │            │  │  ┌───────────────┐ │   │    │
│  │  │              │  │            │  │  │ LSL Push      │ │   │    │
│  │  │              │  │            │  │  │ [t, ch, pol]  │ │   │    │
│  │  └──────────────┘  └────────────┘  │  └───────────────┘ │   │    │
│  │                                    └────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### LSL Stream XML Metadata

```xml
<?xml version="1.0"?>
<info>
    <name>DPB_Spikes</name>
    <type>Spikes</type>
    <channel_count>3</channel_count>
    <nominal_srate>0</nominal_srate>
    <channel_format>float32</channel_format>
    <source_id>DPB_LabVIEW_001</source_id>
    <desc>
        <channels>
            <channel>
                <label>timestamp</label>
                <type>Time</type>
                <unit>seconds</unit>
            </channel>
            <channel>
                <label>channel</label>
                <type>Index</type>
                <unit>index</unit>
            </channel>
            <channel>
                <label>polarity</label>
                <type>Polarity</type>
                <unit>sign</unit>
            </channel>
        </channels>
        <encoder>
            <type>LevelCrossing</type>
            <threshold>0.1</threshold>
        </encoder>
    </desc>
</info>
```

---

## Building the Examples

### Prerequisites

1. Install LabVIEW 2020 or later
2. Build DPB FFI library:
   ```bash
   cargo build --release -p dpb-ffi
   ```
3. Copy library to LabVIEW project

### Quick Start

1. Open LabVIEW
2. File → Open Project → `DPB_Examples.lvproj`
3. Select an example VI
4. Run (Ctrl+R)

---

## Performance Benchmarks

| Example | Input Size | Encoding Time | Throughput |
|---------|------------|---------------|------------|
| Basic | 1000 × 1 | 0.5 ms | 2M samples/s |
| Multi-Channel | 1000 × 8 | 0.8 ms | 10M samples/s |
| Real-Time DAQ | 100 × 8 | 0.2 ms | 4M samples/s |
| RT System | 100 × 8 | 0.4 ms | 2M samples/s |

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| DLL not found | Add `libs/` to system PATH or use full path |
| Memory leak | Always call `_free` functions in cleanup |
| Slow encoding | Use 64-bit LabVIEW, enable Release build |
| RT jitter | Increase priority, use RT FIFOs |

---

## License

MIT License - see main project LICENSE file.
