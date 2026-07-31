# MATLAB Bindings Quick Start

## Installation

### 1. Build the DPB FFI Library

```bash
cd 
cargo build --release -p dpb-ffi
```

### 2. Build the MEX Interface

In MATLAB:

```matlab
cd bindings/matlab
build_mex()
```

### 3. Add to MATLAB Path

```matlab
addpath('bindings/matlab');
```

Or permanently:

```matlab
addpath('bindings/matlab');
savepath
```

## Quick Example

```matlab
% Check version
fprintf('DPB Version: %s\n', dpb.version());

% Create a signal (2 seconds at 250 Hz)
t = linspace(0, 2, 500)';
data = single(sin(2*pi*1.5*t));  % 1.5 Hz sine wave

% Create TimeSeries
ts = dpb.TimeSeries(data, 250.0);
fprintf('Duration: %.2f s, Samples: %d\n', ts.Duration, ts.NumSamples);

% Encode to events
encoder = dpb.LevelCrossingEncoder('Threshold', 0.1);
events = encoder.encode(ts);
fprintf('Generated %d events\n', events.NumEvents);
```

## Run Tests

```matlab
test_dpb()
```

## Common Issues

### "Library not found" Error

**Linux:**
```bash
export LD_LIBRARY_PATH=target/release:$LD_LIBRARY_PATH
```

**macOS:**
```bash
export DYLD_LIBRARY_PATH=target/release:$DYLD_LIBRARY_PATH
```

**Windows:**
Add `C:\path\to\Delta-Predictive-Biosensing\target\release` to your PATH.

### MEX Compiler Not Found

Configure MATLAB's MEX compiler:

```matlab
mex -setup C
```

## Next Steps

- Read the full [README.md](README.md) for detailed API documentation
- Explore examples in `test_dpb.m`
- Check the [DPB Framework Documentation](../../README.md)
