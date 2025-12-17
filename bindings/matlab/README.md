# DPB MATLAB Bindings

MATLAB bindings for the Delta-Predictive Biosensing (DPB) Framework using MEX interface.

## Overview

These bindings provide a MATLAB interface to the DPB framework's core functionality through the C FFI library. The implementation uses MATLAB's MEX (MATLAB Executable) interface to call native C functions.

## Directory Structure

```
bindings/matlab/
├── +dpb/                      # MATLAB package namespace
│   ├── Contents.m             # Package documentation
│   ├── version.m              # Version query function
│   ├── TimeSeries.m           # Time series class
│   ├── SpikeTrain.m           # Spike train class
│   └── LevelCrossingEncoder.m # Level crossing encoder class
├── private/
│   └── dpb_mex.c              # MEX gateway function (C)
├── build_mex.m                # Build script
├── test_dpb.m                 # Test script
└── README.md                  # This file
```

## Prerequisites

1. **MATLAB** - Tested with R2020b and later (requires `arguments` block support)
2. **C Compiler** - Compatible with MATLAB's `mex` command
   - Windows: Microsoft Visual Studio or MinGW-w64
   - Linux: GCC
   - macOS: Xcode Command Line Tools
3. **DPB C FFI Library** - Must be built first

## Building

### Step 1: Build the C FFI Library

From the repository root:

```bash
cargo build --release -p dpb-ffi
```

This creates the shared library at:
- Linux: `target/release/libdpb_ffi.so`
- macOS: `target/release/libdpb_ffi.dylib`
- Windows: `target/release/dpb_ffi.dll`

### Step 2: Build the MEX Interface

In MATLAB, navigate to the `bindings/matlab` directory:

```matlab
cd /path/to/Delta-Predictive-Biosensing/bindings/matlab
build_mex()
```

This compiles `dpb_mex.c` and links it with the DPB FFI library.

## Usage

### Basic Example

```matlab
% Add the bindings to MATLAB path (if not already done)
addpath('/path/to/Delta-Predictive-Biosensing/bindings/matlab');

% Create a synthetic signal
sample_rate = 250.0;  % Hz
t = linspace(0, 2, 500)';  % 2 seconds
data = single(sin(2*pi*1.5*t));  % 1.5 Hz sine wave

% Create TimeSeries object
ts = dpb.TimeSeries(data, sample_rate);

% Display properties
fprintf('Duration: %.2f s\n', ts.Duration);
fprintf('Samples: %d\n', ts.NumSamples);
fprintf('Channels: %d\n', ts.NumChannels);

% Create encoder
encoder = dpb.LevelCrossingEncoder('Threshold', 0.1);

% Encode signal to spike train
events = encoder.encode(ts);
fprintf('Generated %d events\n', events.NumEvents);
```

### Multi-Channel Signals

```matlab
% Create 3-channel signal
num_samples = 500;
t = linspace(0, 2, num_samples)';
data_multi = single([
    sin(2*pi*1.0*t), ...
    sin(2*pi*2.0*t), ...
    sin(2*pi*3.0*t)
]);

ts_multi = dpb.TimeSeries(data_multi, 250.0);
fprintf('Channels: %d\n', ts_multi.NumChannels);  % Output: 3
```

### Spike Train Manipulation

```matlab
% Create empty spike train
st = dpb.SpikeTrain();

% Add events manually
st.addEvent(0.1, uint32(0), int8(1));   % timestamp, channel, polarity
st.addEvent(0.2, uint32(0), int8(-1));
st.addEvent(0.3, uint32(1), int8(1));

fprintf('Events: %d\n', st.NumEvents);  % Output: 3
```

## API Reference

### dpb.version()

Get the DPB library version string.

```matlab
v = dpb.version();
```

### dpb.TimeSeries

Time series signal data container.

**Constructor:**
```matlab
ts = dpb.TimeSeries(data, sampleRate)
```
- `data`: `single` matrix (samples × channels)
- `sampleRate`: Sampling rate in Hz (positive scalar)

**Properties:**
- `SampleRate`: Sampling rate in Hz (read-only)
- `Duration`: Duration in seconds (dependent)
- `NumSamples`: Number of samples (dependent)
- `NumChannels`: Number of channels (dependent)

**Methods:**
- `getData()`: Get the signal data

### dpb.SpikeTrain

Sequence of spike events.

**Constructor:**
```matlab
st = dpb.SpikeTrain()       % Empty spike train
st = dpb.SpikeTrain(handle) % From internal handle
```

**Properties:**
- `NumEvents`: Number of events (dependent)

**Methods:**
- `addEvent(timestamp, channel, polarity)`: Add a spike event
  - `timestamp`: Event time (double)
  - `channel`: Channel index (uint32)
  - `polarity`: Spike polarity (int8, default=1)
- `getEvents()`: Get all events

### dpb.LevelCrossingEncoder

Level crossing event encoder.

**Constructor:**
```matlab
enc = dpb.LevelCrossingEncoder('Threshold', threshold)
```
- `threshold`: Level crossing threshold (default=0.5)

**Properties:**
- `Threshold`: Threshold value (read-only)

**Methods:**
- `encode(timeSeries)`: Encode a TimeSeries to SpikeTrain
  - Returns: `dpb.SpikeTrain` object

## Testing

Run the test suite:

```matlab
test_dpb()
```

The test script validates:
1. Library version retrieval
2. Single-channel TimeSeries creation
3. Multi-channel TimeSeries creation
4. SpikeTrain creation and event addition
5. Level crossing encoding
6. Different threshold values
7. Memory cleanup (automatic via destructors)

## Implementation Details

### Memory Management

All DPB objects (TimeSeries, SpikeTrain, Encoder) are implemented as MATLAB handle classes with automatic cleanup:

- Objects store a handle (pointer) to the underlying C object
- Destructors (`delete` method) automatically free C resources
- MATLAB's garbage collector handles cleanup when objects go out of scope

### Data Types

- **Handles**: 64-bit unsigned integers (`uint64`)
- **Signal data**: Single-precision float (`single`)
- **Timestamps**: Double-precision float (`double`)
- **Channels**: 32-bit unsigned integer (`uint32`)
- **Polarity**: 8-bit signed integer (`int8`)

### MEX Gateway

The `dpb_mex.c` file implements a command-based gateway:

```matlab
result = dpb_mex('command', arg1, arg2, ...)
```

Commands are dispatched to the appropriate FFI function.

## Platform Notes

### Linux
- Requires GCC and compatible standard libraries
- Library path may need to be added to `LD_LIBRARY_PATH`:
  ```bash
  export LD_LIBRARY_PATH=/path/to/target/release:$LD_LIBRARY_PATH
  ```

### macOS
- Requires Xcode Command Line Tools
- May need to add library to `DYLD_LIBRARY_PATH`:
  ```bash
  export DYLD_LIBRARY_PATH=/path/to/target/release:$DYLD_LIBRARY_PATH
  ```

### Windows
- Requires Visual Studio or MinGW-w64
- DLL must be in PATH or same directory as MEX file
- May need to configure MEX compiler:
  ```matlab
  mex -setup C
  ```

## Troubleshooting

### MEX Build Fails

1. Check that a C compiler is configured:
   ```matlab
   mex -setup C
   ```

2. Verify the DPB FFI library exists:
   ```bash
   ls target/release/libdpb_ffi.*
   ```

### Runtime Errors

1. "Library not found" - Add library directory to system path
2. "Invalid MEX file" - Rebuild MEX with matching MATLAB version
3. "Unknown command" - Update MEX file to match API changes

## License

Same as the DPB framework (see repository root).

## See Also

- [DPB Framework Documentation](../../README.md)
- [C FFI Bindings](../c/README.md)
- [Julia Bindings](../julia/README.md)
- [MATLAB MEX Documentation](https://www.mathworks.com/help/matlab/matlab_external/introducing-mex-files.html)
