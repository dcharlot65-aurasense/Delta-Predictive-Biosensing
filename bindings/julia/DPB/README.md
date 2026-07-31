# DPB.jl - Julia Bindings for Delta-Predictive Biosensing

Julia bindings for the DPB (Delta-Predictive Biosensing) framework, providing high-performance neuromorphic signal processing capabilities.

## Overview

DPB.jl provides Julia wrappers around the DPB C FFI library, enabling:
- Time-series data processing
- Spike encoding algorithms (Level Crossing, etc.)
- Synthetic biosignal generation (ECG, EEG, EMG)
- Efficient memory management with automatic cleanup

## Installation

### Prerequisites

1. Build the DPB FFI library:
```bash
cd 
cargo build --release -p dpb-ffi
```

2. Add the DPB.jl package to your Julia environment:
```julia
using Pkg
Pkg.develop(path="bindings/julia/DPB")
```

## Quick Start

```julia
using DPB

# Check library version
println("DPB version: ", version())

# Generate synthetic ECG signal
ecg = DPB.Synth.ecg_signal(
    heart_rate=72.0,
    duration=10.0,
    sample_rate=1000.0
)

println("Generated $(num_samples(ecg)) samples in $(num_channels(ecg)) channels")
println("Duration: $(duration(ecg)) seconds")

# Encode to spikes using level-crossing
encoder = LevelCrossingEncoder(threshold=0.1)
spikes = encode(encoder, ecg)

println("Encoded to $(length(spikes)) spikes")

# Iterate over spikes
for spike in spikes
    println("Spike at t=$(spike.timestamp)s, channel=$(spike.channel)")
end
```

## API Reference

### Core Types

#### `TimeSeries`
Represents multi-channel time-series data.

```julia
# Create from matrix (rows=samples, cols=channels)
data = rand(Float32, 1000, 2)  # 1000 samples, 2 channels
ts = TimeSeries(data, 1000.0)  # 1000 Hz sample rate

# Query properties
num_samples(ts)   # Returns number of samples
num_channels(ts)  # Returns number of channels
duration(ts)      # Returns duration in seconds
sample_rate(ts)   # Returns sample rate in Hz
```

#### `SpikeTrain`
Represents a sequence of spike events.

```julia
# Created by encoders or manually
st = SpikeTrain()

# Query and iterate
length(st)        # Number of spikes
isempty(st)       # Check if empty
st[i]             # Get i-th spike (1-based indexing)

for spike in st
    # Process each spike
end

# Collect to vector
spikes = collect_spikes(st)
```

#### `SpikeEvent`
Individual spike event.

```julia
struct SpikeEvent
    timestamp::Float64   # Time in seconds
    channel::UInt32      # Channel index
    polarity::Int8       # -1, 0, or +1
    magnitude::Float32   # Spike magnitude
end
```

### Encoders

#### `LevelCrossingEncoder`
Generates spikes when signal crosses threshold levels.

```julia
encoder = LevelCrossingEncoder(threshold=0.5)
spikes = encode(encoder, timeseries)
```

### Synthetic Data Generators

All generators are in the `DPB.Synth` submodule:

#### Basic Waveforms

```julia
# Sine wave
ts = DPB.Synth.sine_wave(
    frequency=10.0,      # Hz
    amplitude=1.0,
    duration=1.0,        # seconds
    sample_rate=1000.0   # Hz
)

# White noise
ts = DPB.Synth.white_noise(
    amplitude=0.5,
    duration=5.0,
    sample_rate=1000.0,
    seed=42              # Optional random seed
)

# Brownian motion
ts = DPB.Synth.brownian_motion(
    diffusion=0.1,
    duration=10.0,
    sample_rate=1000.0
)
```

#### Biosignals

```julia
# ECG (electrocardiogram)
ecg = DPB.Synth.ecg_signal(
    heart_rate=72.0,     # BPM
    duration=30.0,
    sample_rate=1000.0,
    noise_level=0.05
)

# EEG (electroencephalogram) with alpha rhythm
eeg = DPB.Synth.eeg_alpha(
    frequency=10.0,      # Alpha frequency
    duration=60.0,
    sample_rate=1000.0,
    channels=8           # Number of channels
)

# EMG (electromyography) with bursts
emg = DPB.Synth.emg_burst(
    burst_duration=0.5,  # seconds
    burst_interval=2.0,  # seconds
    duration=20.0,
    sample_rate=1000.0
)
```

## Examples

### Full Processing Pipeline

```julia
using DPB

# 1. Generate synthetic data
signal = DPB.Synth.sine_wave(
    frequency=5.0,
    amplitude=2.0,
    duration=2.0,
    sample_rate=1000.0
)

# 2. Create encoder
encoder = LevelCrossingEncoder(threshold=0.3)

# 3. Encode to spikes
spikes = encode(encoder, signal)

# 4. Analyze results
println("Input: $(num_samples(signal)) samples")
println("Output: $(length(spikes)) spikes")
println("Compression: $(num_samples(signal) / length(spikes))x")

# 5. Process each spike
for (i, spike) in enumerate(spikes)
    if i <= 5  # Show first 5
        println("Spike $i: t=$(spike.timestamp), pol=$(spike.polarity)")
    end
end
```

### Multi-Channel Processing

```julia
# Generate multi-channel data
data = randn(Float32, 1000, 4)  # 4 channels
ts = TimeSeries(data, 1000.0)

println("Processing $(num_channels(ts)) channels")

# Encode all channels
encoder = LevelCrossingEncoder(threshold=0.2)
spikes = encode(encoder, ts)

# Count spikes per channel
channel_counts = zeros(Int, num_channels(ts))
for spike in spikes
    channel_counts[spike.channel + 1] += 1  # 0-indexed to 1-indexed
end

println("Spikes per channel: ", channel_counts)
```

### Biosignal Analysis

```julia
# Generate realistic ECG
ecg = DPB.Synth.ecg_signal(
    heart_rate=75.0,
    duration=60.0,       # 1 minute
    sample_rate=1000.0,
    noise_level=0.03
)

# Encode with fine threshold for detail
encoder = LevelCrossingEncoder(threshold=0.05)
spikes = encode(encoder, ecg)

# Calculate spike rate
spike_rate = length(spikes) / duration(ecg)
println("Spike rate: $(spike_rate) spikes/second")
println("Data reduction: $(100 * (1 - length(spikes) / num_samples(ecg)))%")
```

## Running Tests

```julia
using Pkg
Pkg.test("DPB")
```

Or from the command line:
```bash
cd bindings/julia/DPB
julia --project -e 'using Pkg; Pkg.test()'
```

## Architecture

DPB.jl uses Julia's `ccall` mechanism to interface with the DPB FFI library:

- **Zero dependencies**: Uses only Julia's built-in `ccall`
- **Automatic memory management**: Finalizers ensure cleanup
- **Type-safe wrappers**: Julia types wrap C pointers
- **Idiomatic interface**: Follows Julia conventions

## Memory Management

All DPB objects (TimeSeries, SpikeTrain, Encoders) are automatically cleaned up when garbage collected. You don't need to manually free resources:

```julia
function process_data()
    ts = TimeSeries(data, 1000.0)
    spikes = encode(encoder, ts)
    # ts and spikes automatically freed when function returns
end
```

## Performance Notes

- Use `Float32` for TimeSeries data (native DPB format)
- Matrix layout: rows=samples, columns=channels
- Spike iteration is lazy (no allocation until needed)
- Use `collect_spikes()` only when you need all spikes at once

## Troubleshooting

### Library not found

If you see a warning about the library not loading:
```
Build the FFI library with:
cd 
cargo build --release -p dpb-ffi
```

### Tests failing

Ensure the FFI library is built in release mode and the path is correct.

## License

Part of the Delta-Predictive Biosensing framework.

## Contributing

See the main DPB repository for contribution guidelines.
