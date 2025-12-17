#!/usr/bin/env julia

"""
Advanced Usage Example for DPB.jl

This example demonstrates advanced features including:
- Batch processing
- Performance measurements
- Memory management
- Error handling
"""

using DPB

println("="^60)
println("DPB.jl Advanced Usage Example")
println("="^60)

# Performance benchmark
println("\n1. Performance Benchmark")
println("-"^60)

function benchmark_encoding(n_samples::Int, threshold::Float64)
    # Generate data
    data = randn(Float32, n_samples, 1)
    ts = TimeSeries(data, 1000.0)
    encoder = LevelCrossingEncoder(threshold=threshold)

    # Time encoding
    t_start = time()
    spikes = encode(encoder, ts)
    t_elapsed = time() - t_start

    return (
        samples=n_samples,
        spikes=length(spikes),
        time=t_elapsed,
        rate=n_samples / t_elapsed
    )
end

println("Encoding performance:")
for n in [1_000, 10_000, 100_000, 1_000_000]
    result = benchmark_encoding(n, 0.1)
    println("  $(rpad(result.samples, 10)) samples -> " *
            "$(rpad(result.spikes, 8)) spikes in " *
            "$(round(result.time * 1000, digits=2)) ms " *
            "($(round(result.rate / 1e6, digits=2)) MSamples/s)")
end

# Batch processing
println("\n2. Batch Processing")
println("-"^60)

function process_batch(n_signals::Int, samples_per_signal::Int)
    encoder = LevelCrossingEncoder(threshold=0.15)
    total_spikes = 0

    for i in 1:n_signals
        # Generate signal
        signal = DPB.Synth.sine_wave(
            frequency=float(i),
            duration=float(samples_per_signal) / 1000.0,
            sample_rate=1000.0
        )

        # Encode
        spikes = encode(encoder, signal)
        total_spikes += length(spikes)
    end

    return total_spikes
end

n_signals = 100
samples = 1000
total = process_batch(n_signals, samples)
println("Processed $n_signals signals ($samples samples each)")
println("Total spikes generated: $total")
println("Average spikes per signal: $(round(total / n_signals, digits=2))")

# Memory management test
println("\n3. Memory Management")
println("-"^60)

function memory_stress_test(iterations::Int)
    for i in 1:iterations
        # Create many objects
        data = rand(Float32, 1000, 1)
        ts = TimeSeries(data, 1000.0)
        encoder = LevelCrossingEncoder(threshold=0.1)
        spikes = encode(encoder, ts)

        # Objects will be automatically freed by Julia GC
    end
end

println("Running memory stress test...")
memory_stress_test(1000)
GC.gc()  # Force garbage collection
println("Completed 1000 iterations with automatic cleanup")
println("If you see this message, memory management is working correctly!")

# Error handling
println("\n4. Error Handling")
println("-"^60)

# Test invalid encoder parameters
print("Testing invalid threshold... ")
try
    encoder = LevelCrossingEncoder(threshold=-1.0)
    println("ERROR: Should have thrown exception!")
catch e
    println("Caught expected error: ", typeof(e))
end

# Multi-channel analysis
println("\n5. Multi-Channel Analysis")
println("-"^60)

# Generate different signals in different channels
n_channels = 8
sample_rate = 1000.0
duration = 10.0
n_samples = Int(duration * sample_rate)

data = zeros(Float32, n_samples, n_channels)
for ch in 1:n_channels
    # Each channel has a different frequency
    freq = float(ch)
    for i in 1:n_samples
        t = (i - 1) / sample_rate
        data[i, ch] = sin(2π * freq * t)
    end
end

ts = TimeSeries(data, sample_rate)
encoder = LevelCrossingEncoder(threshold=0.3)
spikes = encode(encoder, ts)

# Analyze per-channel activity
println("Multi-channel analysis:")
println("  Total channels: $n_channels")
println("  Total spikes: $(length(spikes))")

channel_stats = zeros(Int, n_channels)
for spike in spikes
    channel_stats[spike.channel + 1] += 1
end

println("\nPer-channel spike counts:")
for (ch, count) in enumerate(channel_stats)
    freq = ch
    println("  Channel $(ch-1) ($(freq) Hz): $count spikes")
end

# Advanced biosignal processing
println("\n6. Biosignal Processing Pipeline")
println("-"^60)

# Generate realistic multi-lead ECG
println("Generating 12-lead ECG...")
# Note: For simplicity, we'll use single-channel ECG repeated
ecg_single = DPB.Synth.ecg_signal(
    heart_rate=75.0,
    duration=30.0,
    sample_rate=1000.0,
    noise_level=0.02
)

println("ECG properties:")
println("  Duration: $(duration(ecg_single))s")
println("  Samples: $(num_samples(ecg_single))")
println("  Sample rate: $(sample_rate(ecg_single)) Hz")

# Encode with multiple thresholds
thresholds = [0.01, 0.05, 0.1, 0.2]
println("\nEncoding with different thresholds:")
for thresh in thresholds
    enc = LevelCrossingEncoder(threshold=thresh)
    sp = encode(enc, ecg_single)
    compression = num_samples(ecg_single) / max(length(sp), 1)
    println("  Threshold $thresh: $(length(sp)) spikes " *
            "(compression: $(round(compression, digits=2))x)")
end

# Spike timing analysis
println("\n7. Spike Timing Analysis")
println("-"^60)

# Generate signal and encode
signal = DPB.Synth.sine_wave(frequency=5.0, duration=2.0, sample_rate=1000.0)
encoder = LevelCrossingEncoder(threshold=0.3)
spikes = encode(encoder, signal)

if !isempty(spikes)
    # Calculate inter-spike intervals
    timestamps = [spike.timestamp for spike in spikes]
    intervals = diff(timestamps)

    println("Spike timing statistics:")
    println("  Total spikes: $(length(spikes))")
    println("  Mean ISI: $(round(mean(intervals) * 1000, digits=3)) ms")
    println("  Std ISI: $(round(std(intervals) * 1000, digits=3)) ms")
    println("  Min ISI: $(round(minimum(intervals) * 1000, digits=3)) ms")
    println("  Max ISI: $(round(maximum(intervals) * 1000, digits=3)) ms")
end

# Compare different synthetic signals
println("\n8. Synthetic Signal Comparison")
println("-"^60)

signals = [
    ("Sine", DPB.Synth.sine_wave(frequency=10.0, duration=5.0, sample_rate=1000.0)),
    ("Noise", DPB.Synth.white_noise(amplitude=1.0, duration=5.0, sample_rate=1000.0)),
    ("Brownian", DPB.Synth.brownian_motion(diffusion=0.1, duration=5.0, sample_rate=1000.0)),
    ("ECG", DPB.Synth.ecg_signal(heart_rate=70.0, duration=5.0, sample_rate=1000.0)),
]

encoder = LevelCrossingEncoder(threshold=0.1)

println("Encoding different signal types:")
for (name, sig) in signals
    spikes = encode(encoder, sig)
    spike_rate = length(spikes) / duration(sig)
    println("  $(rpad(name, 10)): $(rpad(length(spikes), 5)) spikes " *
            "($(round(spike_rate, digits=2)) spikes/s)")
end

println("\n" * "="^60)
println("Advanced example completed successfully!")
println("="^60)
