# DPB LabVIEW Integration

LabVIEW interface to the Delta-Predictive Biosensing (DPB) Framework for real-time neuromorphic signal processing and hardware integration.

## Overview

This package provides LabVIEW VIs for:
- Real-time biosignal encoding to spike trains
- Integration with NI DAQ hardware
- Multi-channel signal processing
- Spike visualization and analysis

## Requirements

- **LabVIEW 2020** or later (64-bit recommended)
- **NI-DAQmx** (for hardware integration)
- **DPB FFI Library** (`dpb_ffi.dll` / `libdpb_ffi.so`)

## Installation

### 1. Build the DPB Library

```bash
cd /path/to/Delta-Predictive-Biosensing
cargo build --release -p dpb-ffi
```

### 2. Copy Libraries

Copy the built library to the `libs/` folder:

| Platform | Source | Destination |
|----------|--------|-------------|
| Windows | `target/release/dpb_ffi.dll` | `libs/win64/dpb_ffi.dll` |
| Linux | `target/release/libdpb_ffi.so` | `libs/linux64/libdpb_ffi.so` |
| macOS | `target/release/libdpb_ffi.dylib` | `libs/mac64/libdpb_ffi.dylib` |

### 3. Add to LabVIEW Project

1. Open LabVIEW
2. File → Open → Select `DPB/DPB.lvlib`
3. Add to your project

## VI Reference

### Core VIs

#### DPB_Initialize.vi
Initializes the DPB library and returns version info.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| version | Out | String | Library version |
| error out | Out | Error Cluster | Error information |

#### DPB_Cleanup.vi
Releases all DPB resources.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| error in | In | Error Cluster | Error propagation |
| error out | Out | Error Cluster | Error information |

### TimeSeries VIs

#### DPB_TimeSeries_Create.vi
Creates a TimeSeries object from signal data.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| data | In | DBL 2D Array | Signal data [samples × channels] |
| sample rate (Hz) | In | DBL | Sampling rate |
| TimeSeries Handle | Out | U64 | Opaque handle |
| error out | Out | Error Cluster | Error information |

#### DPB_TimeSeries_Properties.vi
Gets properties of a TimeSeries.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| TimeSeries Handle | In | U64 | Handle from Create |
| duration (s) | Out | DBL | Duration in seconds |
| num samples | Out | I32 | Number of samples |
| num channels | Out | I32 | Number of channels |
| sample rate | Out | DBL | Sample rate in Hz |

#### DPB_TimeSeries_Destroy.vi
Releases TimeSeries resources.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| TimeSeries Handle | In | U64 | Handle to destroy |
| error in/out | In/Out | Error Cluster | Error propagation |

### Encoder VIs

#### DPB_Encoder_LevelCrossing.vi
Creates a level crossing encoder.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| threshold | In | DBL | Level crossing threshold |
| Encoder Handle | Out | U64 | Encoder handle |
| error out | Out | Error Cluster | Error information |

#### DPB_Encoder_Encode.vi
Encodes a TimeSeries to a SpikeTrain.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| Encoder Handle | In | U64 | Encoder handle |
| TimeSeries Handle | In | U64 | Input signal |
| SpikeTrain Handle | Out | U64 | Output spikes |
| error in/out | In/Out | Error Cluster | Error propagation |

### SpikeTrain VIs

#### DPB_SpikeTrain_GetEvents.vi
Extracts spike events as arrays.

| Terminal | Direction | Type | Description |
|----------|-----------|------|-------------|
| SpikeTrain Handle | In | U64 | SpikeTrain handle |
| timestamps | Out | DBL Array | Spike times (seconds) |
| channels | Out | I32 Array | Channel indices |
| polarities | Out | I8 Array | +1 (up) or -1 (down) |
| count | Out | I32 | Number of spikes |

## Example: Real-Time ECG Encoding

```
┌──────────────────────────────────────────────────────────────────┐
│                    Real-Time ECG Encoding                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────┐    ┌───────────────┐    ┌────────────────┐         │
│  │ DAQmx   │───▶│ TimeSeries    │───▶│ Level Crossing │         │
│  │ Read    │    │ Create        │    │ Encode         │         │
│  └─────────┘    └───────────────┘    └────────────────┘         │
│       │                                      │                   │
│       │         ┌────────────────────────────┘                   │
│       │         ▼                                                │
│       │    ┌──────────────┐    ┌─────────────────┐              │
│       │    │ SpikeTrain   │───▶│ Spike Raster    │              │
│       │    │ GetEvents    │    │ Display         │              │
│       │    └──────────────┘    └─────────────────┘              │
│       │                                                          │
│  ┌────┴────────────────────────────────────────────────┐        │
│  │ While Loop (100 ms iteration)                       │        │
│  └─────────────────────────────────────────────────────┘        │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Call Library Function Configuration

For advanced users creating custom VIs:

### dpb_timeseries_new
```
Library Path: <platform>/dpb_ffi.[dll|so|dylib]
Function Name: dpb_timeseries_new
Calling Convention: C
Thread: Reentrant

Return Type: Pointer-sized Unsigned Integer (U64)

Parameters:
1. data: Pointer to Array of SGL (input, pass pointer)
2. num_samples: Pointer-sized Unsigned Integer (value)
3. num_channels: Pointer-sized Unsigned Integer (value)
4. sample_rate: 8-byte Double (value)
```

### dpb_encoder_encode
```
Library Path: <platform>/dpb_ffi.[dll|so|dylib]
Function Name: dpb_encoder_encode
Calling Convention: C

Return Type: Pointer-sized Unsigned Integer (U64)

Parameters:
1. encoder: Pointer-sized Unsigned Integer (value)
2. timeseries: Pointer-sized Unsigned Integer (value)
```

## Real-Time Considerations

### For NI cRIO / CompactRIO

1. Cross-compile the Rust library:
   ```bash
   rustup target add aarch64-unknown-linux-gnu
   cargo build --release -p dpb-ffi --target aarch64-unknown-linux-gnu
   ```

2. Deploy to `/home/lvuser/natinst/lib/`

3. Configure VI for deterministic execution:
   - Use RT FIFO for data transfer
   - Avoid memory allocation in time-critical loops
   - Pre-allocate arrays

### Timing

| Operation | Typical Latency |
|-----------|-----------------|
| TimeSeries Create (1000 samples) | < 1 ms |
| Level Crossing Encode (1000 samples) | < 2 ms |
| SpikeTrain GetEvents (100 spikes) | < 0.5 ms |

## Troubleshooting

### Library Not Found

1. Verify library path in Call Library Function Node
2. Check that library is in system PATH or VI's directory
3. On Windows, ensure Visual C++ Redistributable is installed

### Error Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Null pointer |
| 2 | Invalid parameter |
| 3 | Memory allocation failed |
| 4 | Invalid dimensions |
| 5 | Encoding failed |
| 99 | Unknown error |

### Debug Mode

Enable debug output:
1. Call `DPB_LastError.vi` after any operation
2. Check returned error string for details

## License

MIT License - see LICENSE file.

## See Also

- [DPB Framework Documentation](../../docs/)
- [Cross-Platform Integration Plan](../../docs/CROSS_PLATFORM_INTEGRATION_PLAN.md)
