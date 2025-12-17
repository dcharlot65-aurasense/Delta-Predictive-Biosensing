# DPB FFI - C-Compatible Foreign Function Interface

C-compatible FFI bindings for the Delta-Predictive Biosensing (DPB) Framework. This crate provides the foundation for Julia, MATLAB, Python (ctypes), and other language bindings.

## Features

- **C ABI Compatibility**: All functions use `extern "C"` calling convention
- **Memory Safety**: Strict ownership rules with clear allocation/deallocation patterns
- **Thread-Safe Error Handling**: Per-thread error state using thread-local storage
- **Panic Safety**: All panics are caught and converted to error codes
- **Static & Dynamic Libraries**: Builds both `.a` (static) and `.so`/`.dylib`/`.dll` (dynamic)

## Building

```bash
# Build the FFI library
cargo build -p dpb-ffi --release

# Build artifacts will be in target/release/:
# - libdpb_ffi.a (static library)
# - libdpb_ffi.so (Linux shared library)
# - libdpb_ffi.dylib (macOS shared library)
# - dpb_ffi.dll (Windows DLL)
```

## API Overview

### Version Information

```c
const char* dpb_version(void);
```

### Error Handling

```c
const char* dpb_last_error(void);
void dpb_clear_error(void);
```

### Time Series

```c
DpbTimeSeries* dpb_timeseries_new(const float* data, size_t num_samples,
                                   size_t num_channels, double sample_rate);
void dpb_timeseries_free(DpbTimeSeries* ts);
double dpb_timeseries_duration(const DpbTimeSeries* ts);
size_t dpb_timeseries_num_samples(const DpbTimeSeries* ts);
size_t dpb_timeseries_num_channels(const DpbTimeSeries* ts);
double dpb_timeseries_sample_rate(const DpbTimeSeries* ts);
const float* dpb_timeseries_get_data(const DpbTimeSeries* ts);
```

### Spike Trains

```c
DpbSpikeTrain* dpb_spike_train_new(uint32_t num_channels);
void dpb_spike_train_free(DpbSpikeTrain* st);
int dpb_spike_train_add_event(DpbSpikeTrain* st, double timestamp,
                               uint32_t channel, int8_t polarity);
size_t dpb_spike_train_len(const DpbSpikeTrain* st);
uint32_t dpb_spike_train_num_channels(const DpbSpikeTrain* st);
int dpb_spike_train_get_event(const DpbSpikeTrain* st, size_t index,
                               double* timestamp, uint32_t* channel,
                               int8_t* polarity, float* magnitude);
```

### Encoders

```c
DpbEncoder* dpb_encoder_level_crossing_new(double threshold);
void dpb_encoder_free(DpbEncoder* enc);
DpbSpikeTrain* dpb_encoder_encode(const DpbEncoder* enc, const DpbTimeSeries* ts);
```

## Usage Examples

### C Example

```c
#include "dpb.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // Print version
    printf("DPB Version: %s\n", dpb_version());

    // Create a simple sine wave signal
    size_t num_samples = 1000;
    float* data = malloc(num_samples * sizeof(float));
    for (size_t i = 0; i < num_samples; i++) {
        data[i] = sin(2.0 * 3.14159 * 10.0 * i / 1000.0);
    }

    // Create time series
    DpbTimeSeries* ts = dpb_timeseries_new(data, num_samples, 1, 1000.0);
    free(data);

    if (!ts) {
        fprintf(stderr, "Error creating time series: %s\n", dpb_last_error());
        return 1;
    }

    printf("Duration: %f seconds\n", dpb_timeseries_duration(ts));
    printf("Samples: %zu\n", dpb_timeseries_num_samples(ts));

    // Create encoder
    DpbEncoder* encoder = dpb_encoder_level_crossing_new(0.5);
    if (!encoder) {
        fprintf(stderr, "Error creating encoder: %s\n", dpb_last_error());
        dpb_timeseries_free(ts);
        return 1;
    }

    // Encode signal
    DpbSpikeTrain* spikes = dpb_encoder_encode(encoder, ts);
    if (!spikes) {
        fprintf(stderr, "Error encoding: %s\n", dpb_last_error());
        dpb_encoder_free(encoder);
        dpb_timeseries_free(ts);
        return 1;
    }

    printf("Generated %zu spike events\n", dpb_spike_train_len(spikes));

    // Print first few spikes
    for (size_t i = 0; i < 5 && i < dpb_spike_train_len(spikes); i++) {
        double timestamp;
        uint32_t channel;
        int8_t polarity;

        dpb_spike_train_get_event(spikes, i, &timestamp, &channel, &polarity, NULL);
        printf("  Spike %zu: t=%f, ch=%u, pol=%d\n", i, timestamp, channel, polarity);
    }

    // Clean up
    dpb_spike_train_free(spikes);
    dpb_encoder_free(encoder);
    dpb_timeseries_free(ts);

    return 0;
}
```

Compile with:
```bash
gcc -o example example.c -L/path/to/target/release -ldpb_ffi -lm
```

### Julia Example

```julia
using Libdl

# Load the library
const libdpb = "/path/to/target/release/libdpb_ffi.so"

# Version info
version = unsafe_string(ccall((:dpb_version, libdpb), Ptr{UInt8}, ()))
println("DPB Version: ", version)

# Create time series
data = Float32.(sin.(2π .* 10 .* (0:999) ./ 1000))
ts = ccall((:dpb_timeseries_new, libdpb), Ptr{Cvoid},
           (Ptr{Float32}, Csize_t, Csize_t, Cdouble),
           data, length(data), 1, 1000.0)

if ts == C_NULL
    error_msg = unsafe_string(ccall((:dpb_last_error, libdpb), Ptr{UInt8}, ()))
    error("Failed to create time series: ", error_msg)
end

# Get properties
duration = ccall((:dpb_timeseries_duration, libdpb), Cdouble, (Ptr{Cvoid},), ts)
num_samples = ccall((:dpb_timeseries_num_samples, libdpb), Csize_t, (Ptr{Cvoid},), ts)

println("Duration: ", duration, " seconds")
println("Samples: ", num_samples)

# Create encoder
encoder = ccall((:dpb_encoder_level_crossing_new, libdpb), Ptr{Cvoid}, (Cdouble,), 0.5)

# Encode
spikes = ccall((:dpb_encoder_encode, libdpb), Ptr{Cvoid},
               (Ptr{Cvoid}, Ptr{Cvoid}), encoder, ts)

num_spikes = ccall((:dpb_spike_train_len, libdpb), Csize_t, (Ptr{Cvoid},), spikes)
println("Generated ", num_spikes, " spikes")

# Clean up
ccall((:dpb_spike_train_free, libdpb), Cvoid, (Ptr{Cvoid},), spikes)
ccall((:dpb_encoder_free, libdpb), Cvoid, (Ptr{Cvoid},), encoder)
ccall((:dpb_timeseries_free, libdpb), Cvoid, (Ptr{Cvoid},), ts)
```

### MATLAB Example

```matlab
% Load the library
if ~libisloaded('dpb_ffi')
    loadlibrary('/path/to/libdpb_ffi.so', '/path/to/dpb.h');
end

% Version info
version_ptr = calllib('dpb_ffi', 'dpb_version');
version = char(version_ptr);
fprintf('DPB Version: %s\n', version);

% Create time series
data = single(sin(2*pi*10*(0:999)/1000));
ts = calllib('dpb_ffi', 'dpb_timeseries_new', data, length(data), 1, 1000.0);

if isNull(ts)
    error_ptr = calllib('dpb_ffi', 'dpb_last_error');
    error('Failed to create time series: %s', char(error_ptr));
end

% Get properties
duration = calllib('dpb_ffi', 'dpb_timeseries_duration', ts);
num_samples = calllib('dpb_ffi', 'dpb_timeseries_num_samples', ts);

fprintf('Duration: %f seconds\n', duration);
fprintf('Samples: %d\n', num_samples);

% Create encoder and encode
encoder = calllib('dpb_ffi', 'dpb_encoder_level_crossing_new', 0.5);
spikes = calllib('dpb_ffi', 'dpb_encoder_encode', encoder, ts);

num_spikes = calllib('dpb_ffi', 'dpb_spike_train_len', spikes);
fprintf('Generated %d spikes\n', num_spikes);

% Clean up
calllib('dpb_ffi', 'dpb_spike_train_free', spikes);
calllib('dpb_ffi', 'dpb_encoder_free', encoder);
calllib('dpb_ffi', 'dpb_timeseries_free', ts);
```

## Memory Management

### Allocation Rules

1. Objects created by `*_new()` functions must be freed with corresponding `*_free()` functions
2. Do NOT free pointers returned by:
   - `dpb_version()` (static string)
   - `dpb_last_error()` (thread-local storage)
   - `dpb_timeseries_get_data()` (owned by TimeSeries)
3. Data passed to `*_new()` functions is copied, so you can free the original

### Thread Safety

- Error state is thread-local (each thread has its own error)
- Objects (TimeSeries, SpikeTrain, Encoder) are NOT thread-safe
- Do not share objects between threads without external synchronization

## Error Handling

All functions that can fail either:
1. Return `NULL` pointer (for functions returning pointers)
2. Return non-zero error code (for functions returning `int`)

Always check return values and call `dpb_last_error()` to get details:

```c
DpbTimeSeries* ts = dpb_timeseries_new(data, count, 1, 1000.0);
if (!ts) {
    fprintf(stderr, "Error: %s\n", dpb_last_error());
    // Handle error...
}
```

## Header File

The C header file is located at `include/dpb.h` and contains:
- All function declarations
- Opaque type definitions
- Error code enums
- Documentation comments

## Building Language Bindings

### Julia

Use `ccall` directly (as shown in example above) or create a Julia package using [Clang.jl](https://github.com/JuliaInterop/Clang.jl) to auto-generate bindings.

### MATLAB

Use `loadlibrary` with the header file:
```matlab
loadlibrary('libdpb_ffi.so', 'dpb.h');
```

### Python

Use `ctypes`:
```python
from ctypes import *

libdpb = CDLL('/path/to/libdpb_ffi.so')

# Define return types
libdpb.dpb_version.restype = c_char_p
libdpb.dpb_timeseries_new.restype = c_void_p
# ... etc
```

Or use [CFFI](https://cffi.readthedocs.io/) for better type safety.

## Platform-Specific Notes

### Linux
- Shared library: `libdpb_ffi.so`
- Static library: `libdpb_ffi.a`

### macOS
- Shared library: `libdpb_ffi.dylib`
- Static library: `libdpb_ffi.a`

### Windows
- DLL: `dpb_ffi.dll`
- Import library: `dpb_ffi.dll.lib`
- Static library: `dpb_ffi.lib`

## Testing

Run the built-in tests:
```bash
cargo test -p dpb-ffi
```

## Future Enhancements

- [ ] Add more encoder types (ECG R-peak, PPG pulse, etc.)
- [ ] Expose SNN network inference
- [ ] Add batch encoding functions
- [ ] GPU acceleration support
- [ ] Streaming/online processing API
- [ ] Signal preprocessing functions
- [ ] Auto-generate language bindings using bindgen-style tools

## License

Licensed under MIT OR Apache-2.0 (same as the DPB framework).
