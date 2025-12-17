module DPB

export TimeSeries, SpikeTrain, SpikeEvent
export LevelCrossingEncoder, encode
export version, duration, num_samples, num_channels

# Load the shared library
const libdpb = joinpath(@__DIR__, "..", "..", "..", "target", "release", "libdpb_ffi")

# Include submodules
include("types.jl")
include("encoders.jl")
include("synth.jl")

"""
    version() -> String

Get the version of the DPB library.
"""
function version()
    ccall((:dpb_version, libdpb), Cstring, ()) |> unsafe_string
end

function __init__()
    # Verify library loads
    try
        version()
        @info "DPB.jl initialized successfully (version: $(version()))"
    catch e
        @warn """
        DPB library not found or failed to load.
        Build the library with: cargo build --release -p dpb-ffi
        Error: $e
        """
    end
end

end # module
