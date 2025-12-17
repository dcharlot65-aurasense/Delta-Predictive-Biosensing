# DPB.jl Quick Reference

## Installation

```julia
using Pkg
Pkg.develop(path="/home/user/Delta-Predictive-Biosensing/bindings/julia/DPB")
```

## Basic Workflow

```julia
using DPB

# 1. Create/load data
data = rand(Float32, 1000, 2)
ts = TimeSeries(data, 1000.0)

# 2. Create encoder
encoder = LevelCrossingEncoder(threshold=0.1)

# 3. Encode
spikes = encode(encoder, ts)

# 4. Process spikes
for spike in spikes
    println(spike.timestamp, " ", spike.channel)
end
```

## Common Operations

### TimeSeries

```julia
# Create
ts = TimeSeries(data::Matrix{Float32}, sample_rate::Float64)

# Query
num_samples(ts)   # Int
num_channels(ts)  # Int
duration(ts)      # Float64 (seconds)
sample_rate(ts)   # Float64 (Hz)
```

### SpikeTrain

```julia
# Access
length(spikes)     # Number of spikes
isempty(spikes)    # Bool
spikes[i]          # Get i-th spike (1-based)

# Iterate
for spike in spikes
    # ...
end

# Collect
all_spikes = collect_spikes(spikes)  # Vector{SpikeEvent}
```

### Encoders

```julia
# Level Crossing
encoder = LevelCrossingEncoder(threshold=0.5)
spikes = encode(encoder, timeseries)
```

## Synthetic Data

```julia
using DPB.Synth

# Basic waveforms
sine_wave(frequency=10.0, duration=1.0, sample_rate=1000.0)
white_noise(amplitude=0.5, duration=2.0, sample_rate=1000.0)
brownian_motion(diffusion=0.1, duration=5.0, sample_rate=1000.0)

# Biosignals
ecg_signal(heart_rate=72.0, duration=10.0, sample_rate=1000.0)
eeg_alpha(frequency=10.0, duration=30.0, sample_rate=1000.0, channels=8)
emg_burst(burst_duration=0.5, burst_interval=2.0, duration=10.0)
```

## Code Snippets

### Generate and encode ECG

```julia
ecg = DPB.Synth.ecg_signal(heart_rate=75.0, duration=30.0, sample_rate=1000.0)
encoder = LevelCrossingEncoder(threshold=0.05)
spikes = encode(encoder, ecg)
println("Encoded $(num_samples(ecg)) samples to $(length(spikes)) spikes")
```

### Multi-channel processing

```julia
data = randn(Float32, 1000, 4)  # 4 channels
ts = TimeSeries(data, 1000.0)
spikes = encode(encoder, ts)

# Count per channel
counts = zeros(Int, 4)
for spike in spikes
    counts[spike.channel + 1] += 1
end
```

### Performance timing

```julia
t_start = time()
spikes = encode(encoder, timeseries)
t_elapsed = time() - t_start
rate = num_samples(timeseries) / t_elapsed
println("Encoded at $(rate / 1e6) MSamples/s")
```

### Spike analysis

```julia
timestamps = [s.timestamp for s in spikes]
polarities = [s.polarity for s in spikes]
magnitudes = [s.magnitude for s in spikes]

# Inter-spike intervals
isis = diff(timestamps)
mean_isi = mean(isis)
```

## Error Handling

```julia
try
    encoder = LevelCrossingEncoder(threshold=-1.0)  # Invalid
catch e
    println("Error: ", e)
end

# Check last error
err = DPB.last_error()
if !isempty(err)
    println("Last error: ", err)
end
```

## Tips

1. **Use Float32** for data matrices (native format)
2. **Matrix layout**: rows=samples, cols=channels
3. **Indexing**: Channels are 0-indexed in C but 1-indexed when converting to Julia
4. **Memory**: Objects auto-cleanup via finalizers
5. **Performance**: Avoid creating many small TimeSeries; batch when possible

## Common Patterns

### Batch processing

```julia
encoder = LevelCrossingEncoder(threshold=0.1)
results = []

for file in data_files
    data = load_data(file)
    ts = TimeSeries(data, 1000.0)
    spikes = encode(encoder, ts)
    push!(results, spikes)
end
```

### Parameter sweep

```julia
thresholds = [0.01, 0.05, 0.1, 0.2, 0.5]
spike_counts = []

for thresh in thresholds
    enc = LevelCrossingEncoder(threshold=thresh)
    spikes = encode(enc, signal)
    push!(spike_counts, length(spikes))
end
```

### Custom analysis function

```julia
function analyze_encoding(signal, threshold)
    encoder = LevelCrossingEncoder(threshold=threshold)
    spikes = encode(encoder, signal)

    return (
        n_spikes = length(spikes),
        compression = num_samples(signal) / max(length(spikes), 1),
        spike_rate = length(spikes) / duration(signal)
    )
end
```
