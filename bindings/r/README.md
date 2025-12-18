# DPB R Package

R interface to the Delta-Predictive Biosensing (DPB) Framework for neuromorphic signal processing and spike encoding.

## Installation

### Prerequisites

1. **Rust toolchain** (>= 1.70): Install from [rustup.rs](https://rustup.rs/)
2. **R** (>= 4.0.0)
3. **R development tools**: `Rtools` (Windows), `r-base-dev` (Debian/Ubuntu), Xcode CLI (macOS)

### Build from Source

```bash
# 1. Build the DPB FFI library
cd /path/to/Delta-Predictive-Biosensing
cargo build --release -p dpb-ffi

# 2. Copy library to R package
cp target/release/libdpb_ffi.so bindings/r/inst/libs/       # Linux
cp target/release/libdpb_ffi.dylib bindings/r/inst/libs/   # macOS
cp target/release/dpb_ffi.dll bindings/r/inst/libs/        # Windows

# 3. Copy header
cp crates/dpb-ffi/include/dpb.h bindings/r/inst/include/

# 4. Install R package
R CMD INSTALL bindings/r
```

### Quick Install (if library is pre-built)

```r
# Install dependencies
install.packages(c("R6", "testthat"))

# Install from local source
install.packages("path/to/bindings/r", repos = NULL, type = "source")
```

## Usage

### Basic Example

```r
library(dpb)

# Check version
print(dpb_version())

# Create a synthetic signal
t <- seq(0, 5, length.out = 5000)
ecg <- sin(2 * pi * 1.2 * t) + 0.3 * sin(2 * pi * 2.4 * t)

# Create TimeSeries object
ts <- TimeSeries$new(ecg, sample_rate = 1000)
print(ts)

# Create encoder
encoder <- LevelCrossingEncoder$new(threshold = 0.1)

# Encode to spikes
spikes <- encoder$encode(ts)
print(spikes)

# Get spike events as data frame
events <- spikes$get_events()
head(events)
```

### Multi-channel Data

```r
# EEG-like multi-channel data
n_samples <- 10000
n_channels <- 8
sample_rate <- 256

# Generate synthetic EEG
data <- matrix(rnorm(n_samples * n_channels), ncol = n_channels)
ts <- TimeSeries$new(data, sample_rate = sample_rate)

print(paste("Duration:", ts$duration, "seconds"))
print(paste("Channels:", ts$num_channels))

# Encode all channels
encoder <- LevelCrossingEncoder$new(threshold = 0.5)
spikes <- encoder$encode(ts)
print(paste("Total spikes:", spikes$length))
```

### Loading from File

```r
# Load ECG data from CSV
ts <- timeseries_from_file("ecg_data.csv", sample_rate = 360)

# Encode
encoder <- LevelCrossingEncoder$new(threshold = 0.05)
spikes <- encoder$encode(ts)
```

## API Reference

### TimeSeries

| Method/Property | Description |
|-----------------|-------------|
| `TimeSeries$new(data, sample_rate)` | Create new TimeSeries |
| `$duration` | Duration in seconds |
| `$num_samples` | Number of samples |
| `$num_channels` | Number of channels |
| `$sample_rate` | Sample rate in Hz |
| `$data` | Raw data as matrix |

### SpikeTrain

| Method/Property | Description |
|-----------------|-------------|
| `SpikeTrain$new(num_channels)` | Create empty SpikeTrain |
| `$add_event(timestamp, channel, polarity)` | Add spike event |
| `$get_events()` | Get events as data.frame |
| `$length` | Number of spikes |
| `$num_channels` | Number of channels |

### Encoders

| Encoder | Description |
|---------|-------------|
| `LevelCrossingEncoder$new(threshold)` | Level crossing encoder |
| `DeltaEncoder$new(threshold)` | Delta modulation encoder |
| `$encode(timeseries)` | Encode TimeSeries to SpikeTrain |

## Integration with Other Packages

### With ggplot2

```r
library(ggplot2)

events <- spikes$get_events()

ggplot(events, aes(x = timestamp, y = channel, color = factor(polarity))) +
  geom_point(alpha = 0.5) +
  scale_color_manual(values = c("-1" = "red", "1" = "blue")) +
  labs(title = "Spike Raster Plot", x = "Time (s)", y = "Channel") +
  theme_minimal()
```

### With signal package

```r
library(signal)

# Filter before encoding
bp <- butter(4, c(0.5, 40) / (sample_rate / 2), type = "pass")
filtered <- filtfilt(bp, raw_signal)

ts <- TimeSeries$new(filtered, sample_rate = sample_rate)
spikes <- encoder$encode(ts)
```

## Performance

The R package uses the same optimized Rust core as other DPB bindings:

| Operation | Performance |
|-----------|-------------|
| TimeSeries creation | ~1M samples/sec |
| Level crossing encoding | ~500K samples/sec |
| Spike event access | O(1) per event |

## Troubleshooting

### Library not found

```r
# Check if library is installed
system.file("libs", package = "dpb")

# Verify library exists
list.files(system.file("libs", package = "dpb"))
```

### Build errors

Ensure Rust is installed and in PATH:
```bash
rustc --version
cargo --version
```

## License

MIT License - see LICENSE file.

## See Also

- [DPB Framework Documentation](../../docs/)
- [Python bindings](../../crates/dpb-python/)
- [Julia bindings](../julia/)
- [MATLAB bindings](../matlab/)
