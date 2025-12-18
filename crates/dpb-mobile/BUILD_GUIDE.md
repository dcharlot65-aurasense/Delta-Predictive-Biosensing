# Build Guide for dpb-mobile

This guide provides comprehensive instructions for building the dpb-mobile runtime for iOS and Android platforms.

## Table of Contents

- [Prerequisites](#prerequisites)
- [iOS Build Instructions](#ios-build-instructions)
- [Android Build Instructions](#android-build-instructions)
- [Cross-Compilation Setup](#cross-compilation-setup)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

1. **Rust 1.92.0 or later**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   rustup update
   ```

2. **Cargo tools**
   ```bash
   cargo install cargo-ndk
   cargo install cbindgen
   ```

### Platform-Specific Requirements

#### macOS (for iOS builds)

- Xcode 14.0 or later
- Xcode Command Line Tools
- iOS SDK

```bash
xcode-select --install
```

#### Linux/macOS/Windows (for Android builds)

- Android NDK r25 or later
- Android SDK

**Install Android NDK:**

```bash
# Using Android Studio:
# 1. Open Android Studio
# 2. Go to SDK Manager
# 3. SDK Tools tab
# 4. Check "NDK (Side by side)"
# 5. Click Apply

# Or download directly:
# https://developer.android.com/ndk/downloads

# Set environment variable:
export ANDROID_NDK_HOME=/path/to/ndk
```

## iOS Build Instructions

### 1. Install Rust Target Toolchains

```bash
# iOS ARM64 (physical devices - iPhone 5s and later)
rustup target add aarch64-apple-ios

# iOS ARM64 Simulator (M1/M2 Macs)
rustup target add aarch64-apple-ios-sim

# iOS x86_64 Simulator (Intel Macs)
rustup target add x86_64-apple-ios

# iOS x86_64 Mac Catalyst (optional)
rustup target add x86_64-apple-darwin
```

### 2. Build for iOS Devices

**Standard build:**
```bash
cargo build --target aarch64-apple-ios --release --features ios
```

**With Metal GPU support (planned):**
```bash
cargo build --target aarch64-apple-ios --release --features "ios,metal"
```

**With CoreML support (planned):**
```bash
cargo build --target aarch64-apple-ios --release --features "ios,coreml"
```

**Output location:**
```
target/aarch64-apple-ios/release/libdpb_mobile.a
```

### 3. Build for iOS Simulator

**M1/M2 Macs (ARM64):**
```bash
cargo build --target aarch64-apple-ios-sim --release --features ios
```

**Intel Macs (x86_64):**
```bash
cargo build --target x86_64-apple-ios --release --features ios
```

### 4. Create Universal iOS Library

Create a fat binary that works on both device and simulator:

```bash
# Build for all architectures
cargo build --target aarch64-apple-ios --release --features ios
cargo build --target aarch64-apple-ios-sim --release --features ios
cargo build --target x86_64-apple-ios --release --features ios

# Create universal binary using lipo
lipo -create \
    target/aarch64-apple-ios/release/libdpb_mobile.a \
    target/aarch64-apple-ios-sim/release/libdpb_mobile.a \
    target/x86_64-apple-ios/release/libdpb_mobile.a \
    -output libdpb_mobile_universal.a

# Verify the architectures
lipo -info libdpb_mobile_universal.a
```

### 5. Generate C Headers for iOS

```bash
cbindgen --config cbindgen.toml \
    --crate dpb-mobile \
    --output include/dpb_mobile.h \
    --lang c
```

### 6. Create XCFramework (Recommended)

```bash
# Create XCFramework for distribution
xcodebuild -create-xcframework \
    -library target/aarch64-apple-ios/release/libdpb_mobile.a \
    -headers include/ \
    -library target/aarch64-apple-ios-sim/release/libdpb_mobile.a \
    -headers include/ \
    -output DpbMobile.xcframework
```

## Android Build Instructions

### 1. Install Rust Target Toolchains

```bash
# Android ARM64 (64-bit ARM devices - most modern phones)
rustup target add aarch64-linux-android

# Android ARMv7 (32-bit ARM devices - older phones)
rustup target add armv7-linux-androideabi

# Android x86_64 (64-bit emulator)
rustup target add x86_64-linux-android

# Android i686 (32-bit emulator)
rustup target add i686-linux-android
```

### 2. Configure Android NDK

Set the NDK path:

```bash
# Linux/macOS
export ANDROID_NDK_HOME=$HOME/Android/Sdk/ndk/25.2.9519653

# Windows (PowerShell)
$env:ANDROID_NDK_HOME = "$env:LOCALAPPDATA\Android\Sdk\ndk\25.2.9519653"
```

### 3. Build for Android ARM64 (Primary Target)

**Standard build:**
```bash
cargo ndk --target aarch64-linux-android --platform 21 -- build --release --features android
```

**With NNAPI support (planned):**
```bash
cargo ndk --target aarch64-linux-android --platform 27 -- build --release --features "android,nnapi"
```

**With Vulkan support (planned):**
```bash
cargo ndk --target aarch64-linux-android --platform 24 -- build --release --features "android,vulkan"
```

**Output location:**
```
target/aarch64-linux-android/release/libdpb_mobile.so
```

### 4. Build for Android ARMv7 (Legacy Devices)

```bash
cargo ndk --target armv7-linux-androideabi --platform 21 -- build --release --features android
```

### 5. Build for All Android Architectures

Build script for all architectures:

```bash
#!/bin/bash

TARGETS=(
    "aarch64-linux-android"
    "armv7-linux-androideabi"
    "x86_64-linux-android"
    "i686-linux-android"
)

for target in "${TARGETS[@]}"; do
    echo "Building for $target..."
    cargo ndk --target $target --platform 21 -- build --release --features android
done

echo "All builds completed!"
```

### 6. Create Android AAR Package

Directory structure for AAR:

```
dpb-mobile-android/
├── jni/
│   ├── arm64-v8a/
│   │   └── libdpb_mobile.so
│   ├── armeabi-v7a/
│   │   └── libdpb_mobile.so
│   ├── x86_64/
│   │   └── libdpb_mobile.so
│   └── x86/
│       └── libdpb_mobile.so
├── classes.jar
└── AndroidManifest.xml
```

Build script:

```bash
#!/bin/bash

# Build for all architectures
cargo ndk --target aarch64-linux-android --platform 21 -- build --release --features android
cargo ndk --target armv7-linux-androideabi --platform 21 -- build --release --features android
cargo ndk --target x86_64-linux-android --platform 21 -- build --release --features android
cargo ndk --target i686-linux-android --platform 21 -- build --release --features android

# Create AAR structure
mkdir -p dpb-mobile-aar/jni/{arm64-v8a,armeabi-v7a,x86_64,x86}

# Copy libraries
cp target/aarch64-linux-android/release/libdpb_mobile.so dpb-mobile-aar/jni/arm64-v8a/
cp target/armv7-linux-androideabi/release/libdpb_mobile.so dpb-mobile-aar/jni/armeabi-v7a/
cp target/x86_64-linux-android/release/libdpb_mobile.so dpb-mobile-aar/jni/x86_64/
cp target/i686-linux-android/release/libdpb_mobile.so dpb-mobile-aar/jni/x86/

# Create AAR (requires Android SDK build tools)
cd dpb-mobile-aar
zip -r ../dpb-mobile.aar .
```

## Cross-Compilation Setup

### Linux to iOS (Not Recommended)

While technically possible, cross-compiling for iOS on Linux is not officially supported by Apple. Use macOS for iOS builds.

### Windows to Android

1. Install NDK via Android Studio or standalone
2. Add NDK to PATH
3. Use the same cargo commands as Linux

```powershell
# PowerShell
$env:ANDROID_NDK_HOME = "C:\Users\YourName\AppData\Local\Android\Sdk\ndk\25.2.9519653"
cargo ndk --target aarch64-linux-android --platform 21 -- build --release --features android
```

## Testing

### Unit Tests (Host Platform)

```bash
cargo test -p dpb-mobile
```

### Integration Tests

```bash
cargo test -p dpb-mobile --test integration_test
```

### Benchmarks

```bash
cargo bench -p dpb-mobile
```

### Example Programs

```bash
# Run basic usage example
cargo run --example basic_usage --release -p dpb-mobile

# Run with specific features
cargo run --example basic_usage --release --features "ios" -p dpb-mobile
```

### Testing on iOS Device

1. Build the library
2. Create an Xcode project
3. Link libdpb_mobile.a
4. Include dpb_mobile.h
5. Run on device or simulator

### Testing on Android Device

1. Build the library
2. Create an Android Studio project
3. Add .so files to jniLibs
4. Load library in Java/Kotlin:
   ```kotlin
   companion object {
       init {
           System.loadLibrary("dpb_mobile")
       }
   }
   ```
5. Run on device or emulator

## Optimization Flags

### Size Optimization (Embedded/Mobile)

Add to `Cargo.toml`:

```toml
[profile.release]
opt-level = "z"      # Optimize for size
lto = true           # Enable Link Time Optimization
codegen-units = 1    # Better optimization
strip = true         # Strip symbols
panic = "abort"      # Smaller panic handler
```

### Performance Optimization

```toml
[profile.release]
opt-level = 3        # Maximum speed optimization
lto = "fat"          # Full LTO
codegen-units = 1    # Better optimization
```

## Troubleshooting

### iOS Build Issues

**Problem: "xcrun: error: SDK 'iphoneos' cannot be located"**

Solution:
```bash
sudo xcode-select --switch /Applications/Xcode.app/Contents/Developer
xcodebuild -showsdks
```

**Problem: "library not found for -lSystem"**

Solution:
```bash
rustup update
rustup target add aarch64-apple-ios --force
```

### Android Build Issues

**Problem: "NDK not found"**

Solution:
```bash
# Verify NDK installation
ls $ANDROID_NDK_HOME

# Set correct path
export ANDROID_NDK_HOME=/path/to/ndk
```

**Problem: "error: linker 'aarch64-linux-android21-clang' not found"**

Solution:
```bash
# Install cargo-ndk
cargo install cargo-ndk

# Use cargo-ndk instead of cargo directly
cargo ndk --target aarch64-linux-android --platform 21 -- build --release
```

**Problem: "undefined reference to `__android_log_print`"**

Solution: Link against log library in your Android project.

### Common Build Errors

**Problem: Edition 2024 errors**

Solution: Ensure you're using Rust 1.92.0 or later:
```bash
rustup update
rustc --version  # Should be 1.92.0 or newer
```

**Problem: Feature conflicts**

Solution: Don't mix iOS and Android features:
```bash
# Correct
cargo build --features ios --target aarch64-apple-ios

# Incorrect
cargo build --features "ios,android" --target aarch64-apple-ios
```

## Performance Benchmarks

Expected inference latency on different platforms:

| Platform | Device | Model Size | Latency | Throughput |
|----------|--------|------------|---------|------------|
| iOS | iPhone 13 | 5MB | 1-2ms | 500-1000 inf/sec |
| iOS | iPhone 11 | 5MB | 2-3ms | 300-500 inf/sec |
| Android | Pixel 7 | 5MB | 2-3ms | 300-500 inf/sec |
| Android | Galaxy S21 | 5MB | 1-2ms | 500-1000 inf/sec |

*Benchmarks are approximate and depend on model complexity and device state.

## Binary Size

Expected binary sizes:

| Platform | Configuration | Size |
|----------|--------------|------|
| iOS (arm64) | Release | ~2-5 MB |
| iOS (arm64) | Release + size-opt | ~1-2 MB |
| Android (arm64) | Release | ~3-6 MB |
| Android (arm64) | Release + size-opt | ~1-3 MB |

## Additional Resources

- [Rust Cross-Compilation Guide](https://rust-lang.github.io/rustup/cross-compilation.html)
- [Android NDK Guide](https://developer.android.com/ndk/guides)
- [iOS C Interop Guide](https://developer.apple.com/documentation/swift/imported_c_and_objective-c_apis)
- [cargo-ndk Documentation](https://github.com/bbqsrc/cargo-ndk)

## Support

For issues specific to dpb-mobile:
- GitHub Issues: https://github.com/AuraSenseTech/dpb-framework/issues
- Documentation: https://docs.dpb-framework.dev

For platform-specific issues:
- iOS: Apple Developer Forums
- Android: Stack Overflow with `android-ndk` tag
