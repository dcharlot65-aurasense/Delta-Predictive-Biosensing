# Changelog

All notable changes to DPB.jl will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-17

### Added
- Initial release of DPB.jl Julia bindings
- Core types: `TimeSeries`, `SpikeTrain`, `SpikeEvent`
- Level-crossing encoder implementation
- Synthetic data generators:
  - Basic waveforms: sine wave, white noise, Brownian motion
  - Biosignals: ECG, EEG (alpha), EMG (burst)
- Comprehensive test suite
- Example scripts (basic and advanced usage)
- Full documentation and quick reference guide
- Automatic memory management with finalizers
- Zero-dependency implementation using ccall

### Features
- Multi-channel time-series support
- Efficient spike train iteration
- Type-safe wrappers around C FFI
- Idiomatic Julia interface
- Performance benchmarking examples
- Error handling and validation

[Unreleased]: https://github.com/dcharlot65-aurasense/Delta-Predictive-Biosensing/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/dcharlot65-aurasense/Delta-Predictive-Biosensing/releases/tag/v0.1.0
