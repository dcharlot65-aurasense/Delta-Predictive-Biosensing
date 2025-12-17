#!/usr/bin/env julia

"""
Basic Usage Example for DPB.jl

This example demonstrates the core functionality of the DPB Julia bindings.
"""

using DPB

println("="^60)
println("DPB.jl Basic Usage Example")
println("="^60)

# Check library version
println("\n1. Library Information")
println("-"^60)
println("DPB version: ", version())

# Create a simple time series
println("\n2. Creating TimeSeries")
println("-"^60)
data = rand(Float32, 1000, 2)  # 1000 samples, 2 channels
ts = TimeSeries(data, 1000.0)

println("Created TimeSeries:")
println("  Samples: ", num_samples(ts))
println("  Channels: ", num_channels(ts))
println("  Duration: ", duration(ts), " seconds")
println("  Sample rate: ", sample_rate(ts), " Hz")

# Create and use an encoder
println("\n3. Encoding with LevelCrossingEncoder")
println("-"^60)
encoder = LevelCrossingEncoder(threshold=0.1)
println("Created encoder: ", encoder)

spikes = encode(encoder, ts)
println("Encoded to ", length(spikes), " spikes")

if !isempty(spikes)
    println("\nFirst 5 spikes:")
    for i in 1:min(5, length(spikes))
        spike = spikes[i]
        println("  Spike $i: t=$(round(spike.timestamp, digits=6))s, " *
                "ch=$(spike.channel), pol=$(spike.polarity), " *
                "mag=$(round(spike.magnitude, digits=3))")
    end
end

# Generate synthetic data
println("\n4. Synthetic Data Generation")
println("-"^60)

# Sine wave
println("Generating sine wave...")
sine = DPB.Synth.sine_wave(
    frequency=10.0,
    amplitude=1.0,
    duration=1.0,
    sample_rate=1000.0
)
println("  Sine wave: $(num_samples(sine)) samples, $(duration(sine))s")

# White noise
println("Generating white noise...")
noise = DPB.Synth.white_noise(
    amplitude=0.5,
    duration=2.0,
    sample_rate=1000.0
)
println("  White noise: $(num_samples(noise)) samples, $(duration(noise))s")

# ECG signal
println("Generating ECG signal...")
ecg = DPB.Synth.ecg_signal(
    heart_rate=72.0,
    duration=5.0,
    sample_rate=1000.0
)
println("  ECG: $(num_samples(ecg)) samples, $(duration(ecg))s")

# Full pipeline: ECG -> Spikes
println("\n5. Full Pipeline: ECG to Spikes")
println("-"^60)
ecg_encoder = LevelCrossingEncoder(threshold=0.05)
ecg_spikes = encode(ecg_encoder, ecg)

println("ECG encoding results:")
println("  Input samples: ", num_samples(ecg))
println("  Output spikes: ", length(ecg_spikes))
println("  Compression ratio: ", round(num_samples(ecg) / max(length(ecg_spikes), 1), digits=2), "x")
println("  Spike rate: ", round(length(ecg_spikes) / duration(ecg), digits=2), " spikes/second")

# Spike statistics
if !isempty(ecg_spikes)
    timestamps = [spike.timestamp for spike in ecg_spikes]
    polarities = [spike.polarity for spike in ecg_spikes]

    println("\nSpike statistics:")
    println("  First spike: ", round(minimum(timestamps), digits=6), "s")
    println("  Last spike: ", round(maximum(timestamps), digits=6), "s")
    println("  Positive spikes: ", count(==(1), polarities))
    println("  Negative spikes: ", count(==(-1), polarities))
end

# Multi-channel example
println("\n6. Multi-Channel Processing")
println("-"^60)
multi_data = randn(Float32, 500, 4)  # 4 channels
multi_ts = TimeSeries(multi_data, 1000.0)

println("Processing $(num_channels(multi_ts)) channels...")
multi_spikes = encode(encoder, multi_ts)

# Count spikes per channel
channel_counts = zeros(Int, num_channels(multi_ts))
for spike in multi_spikes
    channel_counts[spike.channel + 1] += 1  # 0-indexed -> 1-indexed
end

println("Spikes per channel:")
for (ch, count) in enumerate(channel_counts)
    println("  Channel $(ch-1): $count spikes")
end

# Biosignal examples
println("\n7. Biosignal Generators")
println("-"^60)

# EEG
eeg = DPB.Synth.eeg_alpha(
    frequency=10.0,
    duration=5.0,
    sample_rate=1000.0,
    channels=4
)
println("EEG: $(num_samples(eeg)) samples, $(num_channels(eeg)) channels")

# EMG
emg = DPB.Synth.emg_burst(
    burst_duration=0.3,
    burst_interval=1.0,
    duration=5.0,
    sample_rate=1000.0
)
println("EMG: $(num_samples(emg)) samples, $(duration(emg))s")

# Brownian motion
brownian = DPB.Synth.brownian_motion(
    diffusion=0.1,
    duration=3.0,
    sample_rate=1000.0
)
println("Brownian motion: $(num_samples(brownian)) samples")

println("\n" * "="^60)
println("Example completed successfully!")
println("="^60)
