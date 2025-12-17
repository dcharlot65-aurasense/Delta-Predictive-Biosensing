# Synthetic data generators

"""
    Synth

Module containing synthetic data generation utilities for DPB.

This module provides wrappers around the DPB synthetic data generators,
allowing creation of various synthetic signals for testing and development.
"""
module Synth

using ..DPB: libdpb, TimeSeries, last_error

export sine_wave, white_noise, brownian_motion
export ecg_signal, eeg_alpha, emg_burst

# ============================================================================
# Basic waveforms
# ============================================================================

"""
    sine_wave(;
        frequency::Float64=1.0,
        amplitude::Float64=1.0,
        duration::Float64=1.0,
        sample_rate::Float64=1000.0,
        phase::Float64=0.0,
        channels::Int=1
    ) -> TimeSeries

Generate a synthetic sine wave.

# Arguments
- `frequency`: Frequency in Hz (default: 1.0)
- `amplitude`: Peak amplitude (default: 1.0)
- `duration`: Duration in seconds (default: 1.0)
- `sample_rate`: Sample rate in Hz (default: 1000.0)
- `phase`: Initial phase in radians (default: 0.0)
- `channels`: Number of channels (default: 1)

# Example
```julia
# Generate a 10 Hz sine wave for 2 seconds
ts = Synth.sine_wave(frequency=10.0, duration=2.0, sample_rate=1000.0)
```
"""
function sine_wave(;
    frequency::Float64=1.0,
    amplitude::Float64=1.0,
    duration::Float64=1.0,
    sample_rate::Float64=1000.0,
    phase::Float64=0.0,
    channels::Int=1
)
    ptr = ccall((:dpb_synth_sine_wave, libdpb), Ptr{Cvoid},
        (Cdouble, Cdouble, Cdouble, Cdouble, Cdouble, Csize_t),
        frequency, amplitude, duration, sample_rate, phase, channels)

    if ptr == C_NULL
        error("Failed to generate sine wave: $(last_error())")
    end

    return TimeSeries(ptr)
end

"""
    white_noise(;
        amplitude::Float64=1.0,
        duration::Float64=1.0,
        sample_rate::Float64=1000.0,
        channels::Int=1,
        seed::UInt64=0
    ) -> TimeSeries

Generate white noise.

# Arguments
- `amplitude`: RMS amplitude (default: 1.0)
- `duration`: Duration in seconds (default: 1.0)
- `sample_rate`: Sample rate in Hz (default: 1000.0)
- `channels`: Number of channels (default: 1)
- `seed`: Random seed (0 = random, default: 0)

# Example
```julia
# Generate 5 seconds of white noise
ts = Synth.white_noise(duration=5.0, amplitude=0.5)
```
"""
function white_noise(;
    amplitude::Float64=1.0,
    duration::Float64=1.0,
    sample_rate::Float64=1000.0,
    channels::Int=1,
    seed::UInt64=0
)
    ptr = ccall((:dpb_synth_white_noise, libdpb), Ptr{Cvoid},
        (Cdouble, Cdouble, Cdouble, Csize_t, UInt64),
        amplitude, duration, sample_rate, channels, seed)

    if ptr == C_NULL
        error("Failed to generate white noise: $(last_error())")
    end

    return TimeSeries(ptr)
end

"""
    brownian_motion(;
        diffusion::Float64=1.0,
        duration::Float64=1.0,
        sample_rate::Float64=1000.0,
        channels::Int=1,
        seed::UInt64=0
    ) -> TimeSeries

Generate Brownian motion (random walk).

# Arguments
- `diffusion`: Diffusion coefficient (default: 1.0)
- `duration`: Duration in seconds (default: 1.0)
- `sample_rate`: Sample rate in Hz (default: 1000.0)
- `channels`: Number of channels (default: 1)
- `seed`: Random seed (0 = random, default: 0)

# Example
```julia
# Generate Brownian motion
ts = Synth.brownian_motion(duration=10.0, diffusion=0.1)
```
"""
function brownian_motion(;
    diffusion::Float64=1.0,
    duration::Float64=1.0,
    sample_rate::Float64=1000.0,
    channels::Int=1,
    seed::UInt64=0
)
    ptr = ccall((:dpb_synth_brownian_motion, libdpb), Ptr{Cvoid},
        (Cdouble, Cdouble, Cdouble, Csize_t, UInt64),
        diffusion, duration, sample_rate, channels, seed)

    if ptr == C_NULL
        error("Failed to generate Brownian motion: $(last_error())")
    end

    return TimeSeries(ptr)
end

# ============================================================================
# Biosignal generators
# ============================================================================

"""
    ecg_signal(;
        heart_rate::Float64=60.0,
        duration::Float64=10.0,
        sample_rate::Float64=1000.0,
        noise_level::Float64=0.05
    ) -> TimeSeries

Generate a synthetic ECG (electrocardiogram) signal.

# Arguments
- `heart_rate`: Heart rate in BPM (default: 60.0)
- `duration`: Duration in seconds (default: 10.0)
- `sample_rate`: Sample rate in Hz (default: 1000.0)
- `noise_level`: Noise level relative to signal (default: 0.05)

# Example
```julia
# Generate 30 seconds of ECG at 72 BPM
ecg = Synth.ecg_signal(heart_rate=72.0, duration=30.0)
```
"""
function ecg_signal(;
    heart_rate::Float64=60.0,
    duration::Float64=10.0,
    sample_rate::Float64=1000.0,
    noise_level::Float64=0.05
)
    ptr = ccall((:dpb_synth_ecg, libdpb), Ptr{Cvoid},
        (Cdouble, Cdouble, Cdouble, Cdouble),
        heart_rate, duration, sample_rate, noise_level)

    if ptr == C_NULL
        error("Failed to generate ECG signal: $(last_error())")
    end

    return TimeSeries(ptr)
end

"""
    eeg_alpha(;
        frequency::Float64=10.0,
        duration::Float64=10.0,
        sample_rate::Float64=1000.0,
        channels::Int=8
    ) -> TimeSeries

Generate synthetic EEG with alpha rhythm.

# Arguments
- `frequency`: Alpha frequency in Hz (default: 10.0)
- `duration`: Duration in seconds (default: 10.0)
- `sample_rate`: Sample rate in Hz (default: 1000.0)
- `channels`: Number of EEG channels (default: 8)

# Example
```julia
# Generate 8-channel EEG with 10 Hz alpha
eeg = Synth.eeg_alpha(duration=60.0, channels=8)
```
"""
function eeg_alpha(;
    frequency::Float64=10.0,
    duration::Float64=10.0,
    sample_rate::Float64=1000.0,
    channels::Int=8
)
    ptr = ccall((:dpb_synth_eeg_alpha, libdpb), Ptr{Cvoid},
        (Cdouble, Cdouble, Cdouble, Csize_t),
        frequency, duration, sample_rate, channels)

    if ptr == C_NULL
        error("Failed to generate EEG signal: $(last_error())")
    end

    return TimeSeries(ptr)
end

"""
    emg_burst(;
        burst_duration::Float64=0.5,
        burst_interval::Float64=2.0,
        duration::Float64=10.0,
        sample_rate::Float64=1000.0,
        amplitude::Float64=1.0
    ) -> TimeSeries

Generate synthetic EMG (electromyography) with burst patterns.

# Arguments
- `burst_duration`: Duration of each burst in seconds (default: 0.5)
- `burst_interval`: Interval between bursts in seconds (default: 2.0)
- `duration`: Total duration in seconds (default: 10.0)
- `sample_rate`: Sample rate in Hz (default: 1000.0)
- `amplitude`: Burst amplitude (default: 1.0)

# Example
```julia
# Generate EMG with 0.5s bursts every 2 seconds
emg = Synth.emg_burst(duration=20.0)
```
"""
function emg_burst(;
    burst_duration::Float64=0.5,
    burst_interval::Float64=2.0,
    duration::Float64=10.0,
    sample_rate::Float64=1000.0,
    amplitude::Float64=1.0
)
    ptr = ccall((:dpb_synth_emg_burst, libdpb), Ptr{Cvoid},
        (Cdouble, Cdouble, Cdouble, Cdouble, Cdouble),
        burst_duration, burst_interval, duration, sample_rate, amplitude)

    if ptr == C_NULL
        error("Failed to generate EMG signal: $(last_error())")
    end

    return TimeSeries(ptr)
end

end # module Synth
