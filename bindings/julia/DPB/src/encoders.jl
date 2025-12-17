# DPB Encoders

"""
    Encoder

Abstract base type for all DPB encoders.

Encoders convert time-series data into spike trains using various algorithms.
"""
abstract type Encoder end

# ============================================================================
# LevelCrossingEncoder
# ============================================================================

"""
    LevelCrossingEncoder <: Encoder

Level-crossing encoder that generates spikes when the signal crosses threshold levels.

# Constructor
    LevelCrossingEncoder(; threshold::Float64=0.5)

Create a level-crossing encoder with the specified threshold.

# Example
```julia
encoder = LevelCrossingEncoder(threshold=0.1)
data = rand(Float32, 1000, 1)
ts = TimeSeries(data, 1000.0)
spikes = encode(encoder, ts)
```
"""
mutable struct LevelCrossingEncoder <: Encoder
    ptr::Ptr{Cvoid}
    threshold::Float64

    function LevelCrossingEncoder(; threshold::Float64=0.5)
        if threshold <= 0.0
            error("Threshold must be positive, got: $threshold")
        end
        ptr = ccall((:dpb_encoder_level_crossing_new, libdpb), Ptr{Cvoid}, (Cdouble,), threshold)
        if ptr == C_NULL
            error("Failed to create LevelCrossingEncoder: $(last_error())")
        end
        enc = new(ptr, threshold)
        finalizer(free!, enc)
        return enc
    end
end

"""
    free!(enc::LevelCrossingEncoder)

Free the memory associated with an encoder.
This is called automatically by the garbage collector.
"""
function free!(enc::LevelCrossingEncoder)
    if enc.ptr != C_NULL
        ccall((:dpb_encoder_free, libdpb), Cvoid, (Ptr{Cvoid},), enc.ptr)
        enc.ptr = C_NULL
    end
end

Base.show(io::IO, enc::LevelCrossingEncoder) =
    print(io, "LevelCrossingEncoder(threshold=$(enc.threshold))")

# ============================================================================
# Encoding
# ============================================================================

"""
    encode(enc::Encoder, ts::TimeSeries) -> SpikeTrain

Encode a time series into a spike train using the specified encoder.

# Arguments
- `enc::Encoder`: The encoder to use
- `ts::TimeSeries`: The time series to encode

# Returns
- `SpikeTrain`: The resulting spike train

# Example
```julia
encoder = LevelCrossingEncoder(threshold=0.1)
data = randn(Float32, 1000, 2)
ts = TimeSeries(data, 1000.0)
spikes = encode(encoder, ts)
println("Generated ", length(spikes), " spikes")
```
"""
function encode(enc::Encoder, ts::TimeSeries)::SpikeTrain
    ptr = ccall((:dpb_encoder_encode, libdpb), Ptr{Cvoid},
        (Ptr{Cvoid}, Ptr{Cvoid}), enc.ptr, ts.ptr)
    if ptr == C_NULL
        error("Encoding failed: $(last_error())")
    end
    return SpikeTrain(ptr)
end

# ============================================================================
# Error handling
# ============================================================================

"""
    last_error() -> String

Get the last error message from the DPB library.
Returns an empty string if no error occurred.
"""
function last_error()
    ptr = ccall((:dpb_last_error, libdpb), Cstring, ())
    ptr == C_NULL ? "" : unsafe_string(ptr)
end

"""
    clear_error()

Clear the last error in the DPB library.
"""
function clear_error()
    ccall((:dpb_clear_error, libdpb), Cvoid, ())
end
