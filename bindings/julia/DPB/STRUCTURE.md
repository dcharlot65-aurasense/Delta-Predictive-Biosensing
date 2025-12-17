# DPB.jl Package Structure

Complete Julia bindings for the Delta-Predictive Biosensing framework.

## Directory Structure

```
bindings/julia/DPB/
├── Project.toml              # Package metadata (UUID: 4d8f3a2c-5e1b-4f9a-8c3d-7a2b1e4f6c8d)
├── README.md                 # Full documentation (400+ lines)
├── QUICK_REFERENCE.md        # Quick reference guide
├── CHANGELOG.md              # Version history
├── Makefile                  # Build and test automation
├── .gitignore                # Julia-specific ignores
│
├── src/                      # Source code (658 lines total)
│   ├── DPB.jl               # Main module (38 lines)
│   ├── types.jl             # Core types: TimeSeries, SpikeTrain, SpikeEvent (221 lines)
│   ├── encoders.jl          # Encoder implementations (124 lines)
│   └── synth.jl             # Synthetic data generators (275 lines)
│
├── test/                     # Test suite (203 lines)
│   └── runtests.jl          # Comprehensive unit tests
│
└── examples/                 # Example scripts (391 lines total)
    ├── basic_usage.jl       # Basic usage examples (159 lines)
    └── advanced_usage.jl    # Advanced patterns (232 lines)
```

## File Details

### Core Package Files

#### `Project.toml` (10 lines)
- Package name: DPB
- Version: 0.1.0
- Julia compatibility: 1.6+
- Zero external dependencies (uses ccall)

#### `src/DPB.jl` (38 lines)
Main module that:
- Exports all public API
- Loads the FFI library from `target/release/libdpb_ffi`
- Includes all submodules
- Provides `version()` function
- Initializes library on package load with `__init__()`

#### `src/types.jl` (221 lines)
Core DPB types with full Julia integration:

**TimeSeries**:
- Constructor: `TimeSeries(data::Matrix{Float32}, sample_rate::Float64)`
- Methods: `duration()`, `num_samples()`, `num_channels()`, `sample_rate()`
- Automatic memory management via finalizers

**SpikeTrain**:
- Constructors: empty or from pointer
- Methods: `length()`, `isempty()`, `getindex()`, `iterate()`
- Full iterator protocol support
- `collect_spikes()` to get Vector{SpikeEvent}

**SpikeEvent**:
- Immutable struct with timestamp, channel, polarity, magnitude
- Custom `show()` method for pretty printing

#### `src/encoders.jl` (124 lines)
Encoder implementations:

**Abstract Encoder** type for extensibility

**LevelCrossingEncoder**:
- Constructor with threshold validation
- `encode(encoder, timeseries)` -> SpikeTrain
- Automatic cleanup via finalizers
- Custom `show()` method

**Error handling**:
- `last_error()` - Get last FFI error message
- `clear_error()` - Clear error state

#### `src/synth.jl` (275 lines)
Comprehensive synthetic data generation in `DPB.Synth` submodule:

**Basic Waveforms**:
- `sine_wave()` - Configurable sine waves
- `white_noise()` - Gaussian white noise
- `brownian_motion()` - Random walk

**Biosignals**:
- `ecg_signal()` - Synthetic ECG with configurable heart rate
- `eeg_alpha()` - Multi-channel EEG with alpha rhythm
- `emg_burst()` - EMG with burst patterns

All generators return `TimeSeries` objects with full parameter control.

### Testing

#### `test/runtests.jl` (203 lines)
Comprehensive test suite covering:
- Library loading and version check
- TimeSeries construction and properties
- SpikeTrain creation and manipulation
- Encoder functionality and error handling
- Spike iteration and indexing
- All synthetic data generators
- Full encoding pipeline
- Memory management stress test

Test organization:
- 15 test sets
- Unit tests for all major components
- Integration tests for full workflows
- Performance and memory tests

### Examples

#### `examples/basic_usage.jl` (159 lines)
Demonstrates:
1. Library information and version
2. Creating TimeSeries
3. Encoding with LevelCrossingEncoder
4. Generating all types of synthetic data
5. Full pipeline: ECG -> Spikes
6. Multi-channel processing
7. Biosignal generators

Fully commented and ready to run.

#### `examples/advanced_usage.jl` (232 lines)
Advanced patterns:
1. Performance benchmarking
2. Batch processing
3. Memory management testing
4. Error handling patterns
5. Multi-channel analysis
6. Biosignal processing pipelines
7. Spike timing analysis
8. Signal comparison

Includes timing measurements and statistical analysis.

### Documentation

#### `README.md`
Complete documentation including:
- Installation instructions
- Quick start guide
- Full API reference
- Multiple usage examples
- Architecture overview
- Performance notes
- Troubleshooting guide

#### `QUICK_REFERENCE.md`
Condensed reference with:
- Common operations
- Code snippets
- Usage patterns
- Tips and best practices

#### `CHANGELOG.md`
Version history following Keep a Changelog format.

### Build Tools

#### `Makefile`
Automation targets:
- `make build-ffi` - Build Rust FFI library
- `make test` - Run Julia tests
- `make example` - Run basic example
- `make install` - Install package
- `make clean` - Clean artifacts

#### `.gitignore`
Ignores Julia-specific files:
- Manifest.toml (lock file)
- Coverage files (*.jl.cov)
- Build artifacts (*.so, *.dylib, *.dll)
- IDE files

## Code Statistics

| Category | Files | Lines | Description |
|----------|-------|-------|-------------|
| Core Source | 4 | 658 | Main package code |
| Tests | 1 | 203 | Comprehensive test suite |
| Examples | 2 | 391 | Usage demonstrations |
| Documentation | 3 | ~600 | README, Quick Ref, Changelog |
| **Total** | **10** | **1,852** | **Complete package** |

## API Summary

### Exported Types
- `TimeSeries` - Multi-channel time-series data
- `SpikeTrain` - Sequence of spike events
- `SpikeEvent` - Individual spike
- `LevelCrossingEncoder` - Threshold-based encoder

### Exported Functions
- `version()` - Get library version
- `encode()` - Encode time-series to spikes
- `duration()`, `num_samples()`, `num_channels()`, `sample_rate()` - TimeSeries queries
- `DPB.Synth.*` - Synthetic data generators (9 functions)

### Features
✓ Zero external dependencies
✓ Automatic memory management
✓ Type-safe wrappers
✓ Idiomatic Julia interface
✓ Full iterator protocol support
✓ Comprehensive error handling
✓ Multi-channel support
✓ Performance optimized
✓ Well documented
✓ Extensively tested

## Usage Example

```julia
using DPB

# Generate ECG
ecg = DPB.Synth.ecg_signal(heart_rate=72.0, duration=10.0, sample_rate=1000.0)

# Encode to spikes
encoder = LevelCrossingEncoder(threshold=0.05)
spikes = encode(encoder, ecg)

# Analyze
println("$(num_samples(ecg)) samples -> $(length(spikes)) spikes")
for spike in spikes
    println("t=$(spike.timestamp), ch=$(spike.channel)")
end
```

## Integration with DPB Framework

The Julia bindings interface with the DPB C FFI library via:
- Library path: `../../../target/release/libdpb_ffi`
- Using Julia's `ccall` for all FFI interactions
- No intermediate bindings or generators needed
- Direct memory-efficient communication

## Next Steps

To use the package:
1. Build the FFI library: `make build-ffi`
2. Run tests: `make test`
3. Try examples: `make example`
4. Install in Julia: `make install`

See README.md for detailed usage instructions.
