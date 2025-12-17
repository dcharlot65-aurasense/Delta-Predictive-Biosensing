# Core DPB types

"""
    TimeSeries

A wrapper around DPB's TimeSeries type for multi-channel time-series data.

# Constructor
    TimeSeries(data::Matrix{Float32}, sample_rate::Float64)

Create a new TimeSeries from a matrix where rows are samples and columns are channels.

# Example
```julia
data = rand(Float32, 1000, 2)  # 1000 samples, 2 channels
ts = TimeSeries(data, 1000.0)  # 1000 Hz sample rate
```
"""
mutable struct TimeSeries
    ptr::Ptr{Cvoid}

    function TimeSeries(data::Matrix{Float32}, sample_rate::Float64)
        samples, channels = size(data)
        ptr = ccall((:dpb_timeseries_new, libdpb), Ptr{Cvoid},
            (Ptr{Float32}, Csize_t, Csize_t, Cdouble),
            data, samples, channels, sample_rate)
        if ptr == C_NULL
            error("Failed to create TimeSeries: $(last_error())")
        end
        ts = new(ptr)
        finalizer(free!, ts)
        return ts
    end
end

"""
    free!(ts::TimeSeries)

Free the memory associated with a TimeSeries object.
This is called automatically by the garbage collector.
"""
function free!(ts::TimeSeries)
    if ts.ptr != C_NULL
        ccall((:dpb_timeseries_free, libdpb), Cvoid, (Ptr{Cvoid},), ts.ptr)
        ts.ptr = C_NULL
    end
end

"""
    duration(ts::TimeSeries) -> Float64

Get the duration of the time series in seconds.
"""
duration(ts::TimeSeries) = ccall((:dpb_timeseries_duration, libdpb), Cdouble, (Ptr{Cvoid},), ts.ptr)

"""
    num_samples(ts::TimeSeries) -> Int

Get the number of samples in the time series.
"""
num_samples(ts::TimeSeries) = Int(ccall((:dpb_timeseries_num_samples, libdpb), Csize_t, (Ptr{Cvoid},), ts.ptr))

"""
    num_channels(ts::TimeSeries) -> Int

Get the number of channels in the time series.
"""
num_channels(ts::TimeSeries) = Int(ccall((:dpb_timeseries_num_channels, libdpb), Csize_t, (Ptr{Cvoid},), ts.ptr))

"""
    sample_rate(ts::TimeSeries) -> Float64

Get the sample rate of the time series in Hz.
"""
sample_rate(ts::TimeSeries) = ccall((:dpb_timeseries_sample_rate, libdpb), Cdouble, (Ptr{Cvoid},), ts.ptr)

# ============================================================================
# SpikeTrain
# ============================================================================

"""
    SpikeTrain

A wrapper around DPB's SpikeTrain type representing a sequence of spike events.

# Constructors
    SpikeTrain()                  # Create empty spike train
    SpikeTrain(ptr::Ptr{Cvoid})   # Wrap existing pointer (internal use)
"""
mutable struct SpikeTrain
    ptr::Ptr{Cvoid}

    function SpikeTrain(ptr::Ptr{Cvoid})
        if ptr == C_NULL
            error("Cannot create SpikeTrain from null pointer")
        end
        st = new(ptr)
        finalizer(free!, st)
        return st
    end

    function SpikeTrain()
        ptr = ccall((:dpb_spike_train_new, libdpb), Ptr{Cvoid}, ())
        SpikeTrain(ptr)
    end
end

"""
    free!(st::SpikeTrain)

Free the memory associated with a SpikeTrain object.
This is called automatically by the garbage collector.
"""
function free!(st::SpikeTrain)
    if st.ptr != C_NULL
        ccall((:dpb_spike_train_free, libdpb), Cvoid, (Ptr{Cvoid},), st.ptr)
        st.ptr = C_NULL
    end
end

"""
    length(st::SpikeTrain) -> Int

Get the number of spikes in the spike train.
"""
Base.length(st::SpikeTrain) = Int(ccall((:dpb_spike_train_len, libdpb), Csize_t, (Ptr{Cvoid},), st.ptr))

"""
    isempty(st::SpikeTrain) -> Bool

Check if the spike train is empty.
"""
Base.isempty(st::SpikeTrain) = length(st) == 0

# ============================================================================
# SpikeEvent
# ============================================================================

"""
    SpikeEvent

Represents a single spike event in a spike train.

# Fields
- `timestamp::Float64`: Time of the spike in seconds
- `channel::UInt32`: Channel index
- `polarity::Int8`: Spike polarity (-1, 0, or +1)
- `magnitude::Float32`: Magnitude of the spike
"""
struct SpikeEvent
    timestamp::Float64
    channel::UInt32
    polarity::Int8
    magnitude::Float32
end

Base.show(io::IO, event::SpikeEvent) = print(io,
    "SpikeEvent(t=$(event.timestamp), ch=$(event.channel), " *
    "pol=$(event.polarity), mag=$(event.magnitude))")

"""
    get_spike(st::SpikeTrain, index::Int) -> SpikeEvent

Get the spike at the given index (1-based indexing).
"""
function get_spike(st::SpikeTrain, index::Int)
    if index < 1 || index > length(st)
        throw(BoundsError(st, index))
    end

    event = Ref{SpikeEvent}()
    success = ccall((:dpb_spike_train_get, libdpb), Bool,
        (Ptr{Cvoid}, Csize_t, Ptr{SpikeEvent}),
        st.ptr, index - 1, event)  # Convert to 0-based indexing

    if !success
        error("Failed to get spike at index $index")
    end

    return event[]
end

"""
    getindex(st::SpikeTrain, index::Int) -> SpikeEvent

Get the spike at the given index using array indexing syntax: `st[i]`.
"""
Base.getindex(st::SpikeTrain, index::Int) = get_spike(st, index)

"""
    iterate(st::SpikeTrain, state=1)

Iterate over all spikes in the spike train.

# Example
```julia
for spike in spike_train
    println("Spike at t=", spike.timestamp)
end
```
"""
function Base.iterate(st::SpikeTrain, state=1)
    if state > length(st)
        return nothing
    end
    return (st[state], state + 1)
end

"""
    collect_spikes(st::SpikeTrain) -> Vector{SpikeEvent}

Collect all spikes into a Vector.
"""
function collect_spikes(st::SpikeTrain)
    n = length(st)
    spikes = Vector{SpikeEvent}(undef, n)
    for i in 1:n
        spikes[i] = st[i]
    end
    return spikes
end
