//! FPGA High-Level Synthesis (HLS) export for DPB encoders.
//!
//! This module generates synthesizable C/C++ code targeting FPGA toolchains:
//! - **Xilinx Vitis HLS** (AMD/Xilinx FPGAs)
//! - **Intel HLS Compiler** (Intel/Altera FPGAs)
//! - **Generic HLS** (Catapult, LegUp, etc.)
//!
//! ## Supported Encoders
//!
//! - Level Crossing Detection
//! - Delta Modulation
//! - Temporal Contrast
//! - Moving Window (configurable)
//!
//! ## Output Formats
//!
//! - Vitis HLS C++ with pragmas
//! - Intel HLS C++ with pragmas
//! - Generic synthesizable C
//!
//! ## Example
//!
//! ```rust,ignore
//! use dpb_export::fpga::{FpgaExporter, FpgaTarget, EncoderConfig};
//!
//! let config = EncoderConfig::level_crossing(0.1, 32);
//! let exporter = FpgaExporter::new(FpgaTarget::XilinxVitis);
//!
//! let hls_code = exporter.export(&config)?;
//! std::fs::write("level_crossing.cpp", hls_code)?;
//! ```

use std::fmt::Write;

/// Target FPGA toolchain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpgaTarget {
    /// Xilinx Vitis HLS (AMD/Xilinx FPGAs).
    XilinxVitis,
    /// Intel HLS Compiler (Intel/Altera FPGAs).
    IntelHls,
    /// Generic synthesizable C (portable).
    GenericHls,
}

impl FpgaTarget {
    /// Get target name.
    pub fn name(&self) -> &'static str {
        match self {
            FpgaTarget::XilinxVitis => "Xilinx Vitis HLS",
            FpgaTarget::IntelHls => "Intel HLS Compiler",
            FpgaTarget::GenericHls => "Generic HLS",
        }
    }
}

/// Data type for FPGA implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FpgaDataType {
    /// 32-bit floating point.
    Float32,
    /// 16-bit floating point (half precision).
    Float16,
    /// Fixed-point Q16.16.
    FixedQ16,
    /// Fixed-point Q8.8.
    FixedQ8,
    /// Custom fixed-point.
    FixedCustom { int_bits: u8, frac_bits: u8 },
}

impl FpgaDataType {
    /// Get HLS type string for Xilinx.
    fn xilinx_type(&self) -> String {
        match self {
            FpgaDataType::Float32 => "float".to_string(),
            FpgaDataType::Float16 => "half".to_string(),
            FpgaDataType::FixedQ16 => "ap_fixed<32, 16>".to_string(),
            FpgaDataType::FixedQ8 => "ap_fixed<16, 8>".to_string(),
            FpgaDataType::FixedCustom {
                int_bits,
                frac_bits,
            } => {
                format!("ap_fixed<{}, {}>", int_bits + frac_bits, *int_bits)
            }
        }
    }

    /// Get HLS type string for Intel.
    fn intel_type(&self) -> String {
        match self {
            FpgaDataType::Float32 => "float".to_string(),
            FpgaDataType::Float16 => "ihc::hls_float<5, 10>".to_string(),
            FpgaDataType::FixedQ16 => "ac_fixed<32, 16, true>".to_string(),
            FpgaDataType::FixedQ8 => "ac_fixed<16, 8, true>".to_string(),
            FpgaDataType::FixedCustom {
                int_bits,
                frac_bits,
            } => {
                format!("ac_fixed<{}, {}, true>", int_bits + frac_bits, *int_bits)
            }
        }
    }
}

/// Encoder type for FPGA export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderType {
    /// Level crossing detection.
    LevelCrossing,
    /// Delta modulation.
    DeltaModulation,
    /// Temporal contrast.
    TemporalContrast,
    /// Moving window encoder.
    MovingWindow,
}

/// Configuration for FPGA encoder export.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Encoder type.
    pub encoder_type: EncoderType,
    /// Data type.
    pub data_type: FpgaDataType,
    /// Number of channels.
    pub num_channels: usize,
    /// Encoding threshold.
    pub threshold: f64,
    /// Window size (for moving window encoder).
    pub window_size: usize,
    /// Pipeline initiation interval (clock cycles between inputs).
    pub pipeline_ii: u32,
    /// Enable loop unrolling.
    pub unroll_factor: u32,
    /// Array partitioning factor.
    pub partition_factor: u32,
}

impl EncoderConfig {
    /// Create level crossing encoder config.
    pub fn level_crossing(threshold: f64, num_channels: usize) -> Self {
        Self {
            encoder_type: EncoderType::LevelCrossing,
            data_type: FpgaDataType::FixedQ16,
            num_channels,
            threshold,
            window_size: 1,
            pipeline_ii: 1,
            unroll_factor: 1,
            partition_factor: num_channels as u32,
        }
    }

    /// Create delta modulation encoder config.
    pub fn delta_modulation(threshold: f64, num_channels: usize) -> Self {
        Self {
            encoder_type: EncoderType::DeltaModulation,
            data_type: FpgaDataType::FixedQ16,
            num_channels,
            threshold,
            window_size: 1,
            pipeline_ii: 1,
            unroll_factor: 1,
            partition_factor: num_channels as u32,
        }
    }

    /// Create temporal contrast encoder config.
    pub fn temporal_contrast(threshold: f64, num_channels: usize) -> Self {
        Self {
            encoder_type: EncoderType::TemporalContrast,
            data_type: FpgaDataType::FixedQ16,
            num_channels,
            threshold,
            window_size: 1,
            pipeline_ii: 1,
            unroll_factor: 1,
            partition_factor: num_channels as u32,
        }
    }

    /// Create moving window encoder config.
    pub fn moving_window(threshold: f64, num_channels: usize, window_size: usize) -> Self {
        Self {
            encoder_type: EncoderType::MovingWindow,
            data_type: FpgaDataType::FixedQ16,
            num_channels,
            threshold,
            window_size,
            pipeline_ii: 1,
            unroll_factor: 1,
            partition_factor: num_channels as u32,
        }
    }

    /// Set data type.
    pub fn with_data_type(mut self, data_type: FpgaDataType) -> Self {
        self.data_type = data_type;
        self
    }

    /// Set pipeline initiation interval.
    pub fn with_pipeline_ii(mut self, ii: u32) -> Self {
        self.pipeline_ii = ii;
        self
    }

    /// Set loop unroll factor.
    pub fn with_unroll(mut self, factor: u32) -> Self {
        self.unroll_factor = factor;
        self
    }
}

/// FPGA HLS code exporter.
pub struct FpgaExporter {
    /// Target toolchain.
    target: FpgaTarget,
    /// Include test bench.
    include_testbench: bool,
    /// Include AXI stream interface.
    include_axi_stream: bool,
}

impl FpgaExporter {
    /// Create a new exporter for the specified target.
    pub fn new(target: FpgaTarget) -> Self {
        Self {
            target,
            include_testbench: true,
            include_axi_stream: true,
        }
    }

    /// Set whether to include test bench.
    pub fn with_testbench(mut self, include: bool) -> Self {
        self.include_testbench = include;
        self
    }

    /// Set whether to include AXI stream interface.
    pub fn with_axi_stream(mut self, include: bool) -> Self {
        self.include_axi_stream = include;
        self
    }

    /// Export encoder configuration to HLS code.
    pub fn export(&self, config: &EncoderConfig) -> Result<String, FpgaExportError> {
        match self.target {
            FpgaTarget::XilinxVitis => self.export_xilinx(config),
            FpgaTarget::IntelHls => self.export_intel(config),
            FpgaTarget::GenericHls => self.export_generic(config),
        }
    }

    /// Export header file.
    pub fn export_header(&self, config: &EncoderConfig) -> Result<String, FpgaExportError> {
        let mut code = String::new();

        writeln!(code, "// DPB FPGA Encoder Header")?;
        writeln!(code, "// Target: {}", self.target.name())?;
        writeln!(code, "// Generated by dpb-export")?;
        writeln!(code)?;
        writeln!(code, "#ifndef DPB_ENCODER_H")?;
        writeln!(code, "#define DPB_ENCODER_H")?;
        writeln!(code)?;

        match self.target {
            FpgaTarget::XilinxVitis => {
                writeln!(code, "#include <ap_fixed.h>")?;
                writeln!(code, "#include <hls_stream.h>")?;
            }
            FpgaTarget::IntelHls => {
                writeln!(code, "#include <HLS/hls.h>")?;
                writeln!(code, "#include <HLS/ac_fixed.h>")?;
            }
            FpgaTarget::GenericHls => {
                writeln!(code, "#include <stdint.h>")?;
            }
        }

        writeln!(code)?;
        writeln!(code, "// Configuration")?;
        writeln!(code, "#define NUM_CHANNELS {}", config.num_channels)?;
        writeln!(code, "#define THRESHOLD {:.6}", config.threshold)?;
        writeln!(code, "#define WINDOW_SIZE {}", config.window_size)?;
        writeln!(code)?;

        let data_type = match self.target {
            FpgaTarget::XilinxVitis => config.data_type.xilinx_type(),
            FpgaTarget::IntelHls => config.data_type.intel_type(),
            FpgaTarget::GenericHls => "int32_t".to_string(),
        };

        writeln!(code, "typedef {} data_t;", data_type)?;
        writeln!(code, "typedef int8_t spike_t;")?;
        writeln!(code)?;

        // Function declarations
        let func_name = match config.encoder_type {
            EncoderType::LevelCrossing => "level_crossing_encode",
            EncoderType::DeltaModulation => "delta_modulation_encode",
            EncoderType::TemporalContrast => "temporal_contrast_encode",
            EncoderType::MovingWindow => "moving_window_encode",
        };

        writeln!(code, "void {}(", func_name)?;
        writeln!(code, "    data_t input[NUM_CHANNELS],")?;
        writeln!(code, "    spike_t output[NUM_CHANNELS],")?;
        writeln!(code, "    data_t threshold")?;
        writeln!(code, ");")?;
        writeln!(code)?;
        writeln!(code, "#endif // DPB_ENCODER_H")?;

        Ok(code)
    }

    /// Export for Xilinx Vitis HLS.
    fn export_xilinx(&self, config: &EncoderConfig) -> Result<String, FpgaExportError> {
        let mut code = String::new();

        // Header
        writeln!(code, "// DPB FPGA Encoder - Xilinx Vitis HLS")?;
        writeln!(code, "// Generated by dpb-export")?;
        writeln!(code, "//")?;
        writeln!(code, "// Encoder: {:?}", config.encoder_type)?;
        writeln!(code, "// Channels: {}", config.num_channels)?;
        writeln!(code, "// Threshold: {:.6}", config.threshold)?;
        writeln!(code)?;
        writeln!(code, "#include <ap_fixed.h>")?;
        writeln!(code, "#include <hls_stream.h>")?;
        writeln!(code)?;

        // Type definitions
        let data_type = config.data_type.xilinx_type();
        writeln!(code, "typedef {} data_t;", data_type)?;
        writeln!(code, "typedef ap_int<2> spike_t; // -1, 0, +1")?;
        writeln!(code)?;
        writeln!(code, "#define NUM_CHANNELS {}", config.num_channels)?;
        writeln!(code)?;

        // Generate encoder function
        match config.encoder_type {
            EncoderType::LevelCrossing => {
                self.generate_level_crossing_xilinx(&mut code, config)?;
            }
            EncoderType::DeltaModulation => {
                self.generate_delta_modulation_xilinx(&mut code, config)?;
            }
            EncoderType::TemporalContrast => {
                self.generate_temporal_contrast_xilinx(&mut code, config)?;
            }
            EncoderType::MovingWindow => {
                self.generate_moving_window_xilinx(&mut code, config)?;
            }
        }

        // Add AXI stream wrapper if requested
        if self.include_axi_stream {
            writeln!(code)?;
            self.generate_axi_wrapper_xilinx(&mut code, config)?;
        }

        Ok(code)
    }

    fn generate_level_crossing_xilinx(
        &self,
        code: &mut String,
        config: &EncoderConfig,
    ) -> Result<(), FpgaExportError> {
        writeln!(code, "void level_crossing_encode(")?;
        writeln!(code, "    data_t input[NUM_CHANNELS],")?;
        writeln!(code, "    spike_t output[NUM_CHANNELS],")?;
        writeln!(code, "    data_t threshold")?;
        writeln!(code, ") {{")?;
        writeln!(code, "#pragma HLS PIPELINE II={}", config.pipeline_ii)?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=input complete")?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=output complete")?;
        writeln!(code)?;
        writeln!(code, "    static data_t prev[NUM_CHANNELS];")?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=prev complete")?;
        writeln!(code)?;
        writeln!(code, "    CHANNEL_LOOP:")?;
        writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
        writeln!(code, "#pragma HLS UNROLL factor={}", config.unroll_factor)?;
        writeln!(code, "        data_t curr = input[ch];")?;
        writeln!(code, "        data_t p = prev[ch];")?;
        writeln!(code)?;
        writeln!(
            code,
            "        // Positive crossing: prev < threshold AND curr >= threshold"
        )?;
        writeln!(code, "        if (p < threshold && curr >= threshold) {{")?;
        writeln!(code, "            output[ch] = 1;")?;
        writeln!(code, "        }}")?;
        writeln!(
            code,
            "        // Negative crossing: prev >= threshold AND curr < threshold"
        )?;
        writeln!(
            code,
            "        else if (p >= threshold && curr < threshold) {{"
        )?;
        writeln!(code, "            output[ch] = -1;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else {{")?;
        writeln!(code, "            output[ch] = 0;")?;
        writeln!(code, "        }}")?;
        writeln!(code)?;
        writeln!(code, "        prev[ch] = curr;")?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;

        Ok(())
    }

    fn generate_delta_modulation_xilinx(
        &self,
        code: &mut String,
        config: &EncoderConfig,
    ) -> Result<(), FpgaExportError> {
        writeln!(code, "void delta_modulation_encode(")?;
        writeln!(code, "    data_t input[NUM_CHANNELS],")?;
        writeln!(code, "    spike_t output[NUM_CHANNELS],")?;
        writeln!(code, "    data_t threshold")?;
        writeln!(code, ") {{")?;
        writeln!(code, "#pragma HLS PIPELINE II={}", config.pipeline_ii)?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=input complete")?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=output complete")?;
        writeln!(code)?;
        writeln!(code, "    static data_t reference[NUM_CHANNELS];")?;
        writeln!(
            code,
            "#pragma HLS ARRAY_PARTITION variable=reference complete"
        )?;
        writeln!(code)?;
        writeln!(code, "    CHANNEL_LOOP:")?;
        writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
        writeln!(code, "#pragma HLS UNROLL factor={}", config.unroll_factor)?;
        writeln!(code, "        data_t diff = input[ch] - reference[ch];")?;
        writeln!(code)?;
        writeln!(code, "        if (diff > threshold) {{")?;
        writeln!(code, "            output[ch] = 1;")?;
        writeln!(
            code,
            "            reference[ch] = reference[ch] + threshold;"
        )?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else if (diff < -threshold) {{")?;
        writeln!(code, "            output[ch] = -1;")?;
        writeln!(
            code,
            "            reference[ch] = reference[ch] - threshold;"
        )?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else {{")?;
        writeln!(code, "            output[ch] = 0;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;

        Ok(())
    }

    fn generate_temporal_contrast_xilinx(
        &self,
        code: &mut String,
        config: &EncoderConfig,
    ) -> Result<(), FpgaExportError> {
        writeln!(code, "void temporal_contrast_encode(")?;
        writeln!(code, "    data_t input[NUM_CHANNELS],")?;
        writeln!(code, "    spike_t output[NUM_CHANNELS],")?;
        writeln!(code, "    data_t threshold")?;
        writeln!(code, ") {{")?;
        writeln!(code, "#pragma HLS PIPELINE II={}", config.pipeline_ii)?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=input complete")?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=output complete")?;
        writeln!(code)?;
        writeln!(code, "    static data_t prev[NUM_CHANNELS];")?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=prev complete")?;
        writeln!(code)?;
        writeln!(code, "    CHANNEL_LOOP:")?;
        writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
        writeln!(code, "#pragma HLS UNROLL factor={}", config.unroll_factor)?;
        writeln!(code, "        data_t derivative = input[ch] - prev[ch];")?;
        writeln!(code)?;
        writeln!(code, "        if (derivative > threshold) {{")?;
        writeln!(code, "            output[ch] = 1;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else if (derivative < -threshold) {{")?;
        writeln!(code, "            output[ch] = -1;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else {{")?;
        writeln!(code, "            output[ch] = 0;")?;
        writeln!(code, "        }}")?;
        writeln!(code)?;
        writeln!(code, "        prev[ch] = input[ch];")?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;

        Ok(())
    }

    fn generate_moving_window_xilinx(
        &self,
        code: &mut String,
        config: &EncoderConfig,
    ) -> Result<(), FpgaExportError> {
        writeln!(code, "#define WINDOW_SIZE {}", config.window_size)?;
        writeln!(code)?;
        writeln!(code, "void moving_window_encode(")?;
        writeln!(code, "    data_t input[NUM_CHANNELS],")?;
        writeln!(code, "    spike_t output[NUM_CHANNELS],")?;
        writeln!(code, "    data_t threshold")?;
        writeln!(code, ") {{")?;
        writeln!(code, "#pragma HLS PIPELINE II={}", config.pipeline_ii)?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=input complete")?;
        writeln!(code, "#pragma HLS ARRAY_PARTITION variable=output complete")?;
        writeln!(code)?;
        writeln!(code, "    static data_t window[NUM_CHANNELS][WINDOW_SIZE];")?;
        writeln!(
            code,
            "#pragma HLS ARRAY_PARTITION variable=window complete dim=1"
        )?;
        writeln!(code, "    static int window_idx = 0;")?;
        writeln!(code)?;
        writeln!(code, "    CHANNEL_LOOP:")?;
        writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
        writeln!(code, "#pragma HLS UNROLL factor={}", config.unroll_factor)?;
        writeln!(code, "        // Store new value")?;
        writeln!(code, "        window[ch][window_idx] = input[ch];")?;
        writeln!(code)?;
        writeln!(code, "        // Compute window mean")?;
        writeln!(code, "        data_t sum = 0;")?;
        writeln!(code, "        for (int i = 0; i < WINDOW_SIZE; i++) {{")?;
        writeln!(code, "#pragma HLS UNROLL")?;
        writeln!(code, "            sum += window[ch][i];")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        data_t mean = sum / WINDOW_SIZE;")?;
        writeln!(code)?;
        writeln!(code, "        // Detect spike based on deviation from mean")?;
        writeln!(code, "        data_t deviation = input[ch] - mean;")?;
        writeln!(code, "        if (deviation > threshold) {{")?;
        writeln!(code, "            output[ch] = 1;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else if (deviation < -threshold) {{")?;
        writeln!(code, "            output[ch] = -1;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "        else {{")?;
        writeln!(code, "            output[ch] = 0;")?;
        writeln!(code, "        }}")?;
        writeln!(code, "    }}")?;
        writeln!(code)?;
        writeln!(code, "    window_idx = (window_idx + 1) % WINDOW_SIZE;")?;
        writeln!(code, "}}")?;

        Ok(())
    }

    fn generate_axi_wrapper_xilinx(
        &self,
        code: &mut String,
        config: &EncoderConfig,
    ) -> Result<(), FpgaExportError> {
        writeln!(code, "// AXI Stream Wrapper")?;
        writeln!(code, "typedef struct {{")?;
        writeln!(code, "    data_t data[NUM_CHANNELS];")?;
        writeln!(code, "    ap_uint<1> last;")?;
        writeln!(code, "}} axis_data_t;")?;
        writeln!(code)?;
        writeln!(code, "typedef struct {{")?;
        writeln!(code, "    spike_t data[NUM_CHANNELS];")?;
        writeln!(code, "    ap_uint<1> last;")?;
        writeln!(code, "}} axis_spike_t;")?;
        writeln!(code)?;

        let func_name = match config.encoder_type {
            EncoderType::LevelCrossing => "level_crossing",
            EncoderType::DeltaModulation => "delta_modulation",
            EncoderType::TemporalContrast => "temporal_contrast",
            EncoderType::MovingWindow => "moving_window",
        };

        writeln!(code, "void {}_axi(", func_name)?;
        writeln!(code, "    hls::stream<axis_data_t>& input_stream,")?;
        writeln!(code, "    hls::stream<axis_spike_t>& output_stream,")?;
        writeln!(code, "    data_t threshold,")?;
        writeln!(code, "    int num_samples")?;
        writeln!(code, ") {{")?;
        writeln!(code, "#pragma HLS INTERFACE axis port=input_stream")?;
        writeln!(code, "#pragma HLS INTERFACE axis port=output_stream")?;
        writeln!(code, "#pragma HLS INTERFACE s_axilite port=threshold")?;
        writeln!(code, "#pragma HLS INTERFACE s_axilite port=num_samples")?;
        writeln!(code, "#pragma HLS INTERFACE s_axilite port=return")?;
        writeln!(code)?;
        writeln!(code, "    for (int i = 0; i < num_samples; i++) {{")?;
        writeln!(code, "#pragma HLS PIPELINE II=1")?;
        writeln!(code, "        axis_data_t in_pkt = input_stream.read();")?;
        writeln!(code, "        axis_spike_t out_pkt;")?;
        writeln!(code)?;
        writeln!(
            code,
            "        {}_encode(in_pkt.data, out_pkt.data, threshold);",
            func_name
        )?;
        writeln!(code)?;
        writeln!(code, "        out_pkt.last = in_pkt.last;")?;
        writeln!(code, "        output_stream.write(out_pkt);")?;
        writeln!(code, "    }}")?;
        writeln!(code, "}}")?;

        Ok(())
    }

    /// Export for Intel HLS.
    fn export_intel(&self, config: &EncoderConfig) -> Result<String, FpgaExportError> {
        let mut code = String::new();

        writeln!(code, "// DPB FPGA Encoder - Intel HLS Compiler")?;
        writeln!(code, "// Generated by dpb-export")?;
        writeln!(code)?;
        writeln!(code, "#include <HLS/hls.h>")?;
        writeln!(code, "#include <HLS/ac_fixed.h>")?;
        writeln!(code)?;

        let data_type = config.data_type.intel_type();
        writeln!(code, "typedef {} data_t;", data_type)?;
        writeln!(code, "typedef ac_int<2, true> spike_t;")?;
        writeln!(code)?;
        writeln!(
            code,
            "constexpr int NUM_CHANNELS = {};",
            config.num_channels
        )?;
        writeln!(code)?;

        // Generate encoder based on type
        match config.encoder_type {
            EncoderType::LevelCrossing => {
                writeln!(code, "component void level_crossing_encode(")?;
                writeln!(code, "    ihc::stream_in<data_t>& input,")?;
                writeln!(code, "    ihc::stream_out<spike_t>& output,")?;
                writeln!(code, "    data_t threshold")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t prev[NUM_CHANNELS];")?;
                writeln!(code)?;
                writeln!(code, "    #pragma unroll")?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t curr = input.read();")?;
                writeln!(code, "        spike_t spike = 0;")?;
                writeln!(code)?;
                writeln!(
                    code,
                    "        if (prev[ch] < threshold && curr >= threshold) {{"
                )?;
                writeln!(code, "            spike = 1;")?;
                writeln!(
                    code,
                    "        }} else if (prev[ch] >= threshold && curr < threshold) {{"
                )?;
                writeln!(code, "            spike = -1;")?;
                writeln!(code, "        }}")?;
                writeln!(code)?;
                writeln!(code, "        output.write(spike);")?;
                writeln!(code, "        prev[ch] = curr;")?;
                writeln!(code, "    }}")?;
                writeln!(code, "}}")?;
            }
            EncoderType::DeltaModulation => {
                writeln!(code, "component void delta_modulation_encode(")?;
                writeln!(code, "    ihc::stream_in<data_t>& input,")?;
                writeln!(code, "    ihc::stream_out<spike_t>& output,")?;
                writeln!(code, "    data_t threshold")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t reference[NUM_CHANNELS];")?;
                writeln!(code)?;
                writeln!(code, "    #pragma unroll")?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t curr = input.read();")?;
                writeln!(code, "        data_t diff = curr - reference[ch];")?;
                writeln!(code, "        spike_t spike = 0;")?;
                writeln!(code)?;
                writeln!(code, "        if (diff > threshold) {{")?;
                writeln!(code, "            spike = 1;")?;
                writeln!(
                    code,
                    "            reference[ch] = reference[ch] + threshold;"
                )?;
                writeln!(code, "        }} else if (diff < -threshold) {{")?;
                writeln!(code, "            spike = -1;")?;
                writeln!(
                    code,
                    "            reference[ch] = reference[ch] - threshold;"
                )?;
                writeln!(code, "        }}")?;
                writeln!(code)?;
                writeln!(code, "        output.write(spike);")?;
                writeln!(code, "    }}")?;
                writeln!(code, "}}")?;
            }
            EncoderType::TemporalContrast => {
                writeln!(code, "component void temporal_contrast_encode(")?;
                writeln!(code, "    ihc::stream_in<data_t>& input,")?;
                writeln!(code, "    ihc::stream_out<spike_t>& output,")?;
                writeln!(code, "    data_t threshold")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t prev[NUM_CHANNELS];")?;
                writeln!(code)?;
                writeln!(code, "    #pragma unroll")?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t curr = input.read();")?;
                writeln!(code, "        data_t derivative = curr - prev[ch];")?;
                writeln!(code, "        spike_t spike = 0;")?;
                writeln!(code)?;
                writeln!(code, "        if (derivative > threshold) {{")?;
                writeln!(code, "            spike = 1;")?;
                writeln!(code, "        }} else if (derivative < -threshold) {{")?;
                writeln!(code, "            spike = -1;")?;
                writeln!(code, "        }}")?;
                writeln!(code)?;
                writeln!(code, "        output.write(spike);")?;
                writeln!(code, "        prev[ch] = curr;")?;
                writeln!(code, "    }}")?;
                writeln!(code, "}}")?;
            }
            EncoderType::MovingWindow => {
                writeln!(code, "constexpr int WINDOW_SIZE = {};", config.window_size)?;
                writeln!(code)?;
                writeln!(code, "component void moving_window_encode(")?;
                writeln!(code, "    ihc::stream_in<data_t>& input,")?;
                writeln!(code, "    ihc::stream_out<spike_t>& output,")?;
                writeln!(code, "    data_t threshold")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t window[NUM_CHANNELS][WINDOW_SIZE];")?;
                writeln!(code, "    static int window_idx = 0;")?;
                writeln!(code)?;
                writeln!(code, "    #pragma unroll")?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t curr = input.read();")?;
                writeln!(code, "        window[ch][window_idx] = curr;")?;
                writeln!(code)?;
                writeln!(code, "        // Compute window mean")?;
                writeln!(code, "        data_t sum = 0;")?;
                writeln!(code, "        #pragma unroll")?;
                writeln!(code, "        for (int i = 0; i < WINDOW_SIZE; i++) {{")?;
                writeln!(code, "            sum += window[ch][i];")?;
                writeln!(code, "        }}")?;
                writeln!(code, "        data_t mean = sum / WINDOW_SIZE;")?;
                writeln!(code)?;
                writeln!(code, "        data_t deviation = curr - mean;")?;
                writeln!(code, "        spike_t spike = 0;")?;
                writeln!(code)?;
                writeln!(code, "        if (deviation > threshold) {{")?;
                writeln!(code, "            spike = 1;")?;
                writeln!(code, "        }} else if (deviation < -threshold) {{")?;
                writeln!(code, "            spike = -1;")?;
                writeln!(code, "        }}")?;
                writeln!(code)?;
                writeln!(code, "        output.write(spike);")?;
                writeln!(code, "    }}")?;
                writeln!(code)?;
                writeln!(code, "    window_idx = (window_idx + 1) % WINDOW_SIZE;")?;
                writeln!(code, "}}")?;
            }
        }

        Ok(code)
    }

    /// Export generic synthesizable C.
    fn export_generic(&self, config: &EncoderConfig) -> Result<String, FpgaExportError> {
        let mut code = String::new();

        writeln!(code, "// DPB FPGA Encoder - Generic Synthesizable C")?;
        writeln!(code, "// Generated by dpb-export")?;
        writeln!(code, "// Compatible with: Catapult HLS, LegUp, Bambu")?;
        writeln!(code)?;
        writeln!(code, "#include <stdint.h>")?;
        writeln!(code)?;
        writeln!(code, "#define NUM_CHANNELS {}", config.num_channels)?;
        writeln!(code)?;
        writeln!(code, "typedef int32_t data_t;  // Q16.16 fixed-point")?;
        writeln!(code, "typedef int8_t spike_t;")?;
        writeln!(code)?;
        writeln!(code, "// Fixed-point threshold (Q16.16)")?;
        writeln!(
            code,
            "#define THRESHOLD {}",
            (config.threshold * 65536.0) as i32
        )?;
        writeln!(code)?;

        match config.encoder_type {
            EncoderType::LevelCrossing => {
                writeln!(code, "void level_crossing_encode(")?;
                writeln!(code, "    data_t input[NUM_CHANNELS],")?;
                writeln!(code, "    spike_t output[NUM_CHANNELS]")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t prev[NUM_CHANNELS];")?;
                writeln!(code)?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t curr = input[ch];")?;
                writeln!(code)?;
                writeln!(
                    code,
                    "        if (prev[ch] < THRESHOLD && curr >= THRESHOLD) {{"
                )?;
                writeln!(code, "            output[ch] = 1;")?;
                writeln!(
                    code,
                    "        }} else if (prev[ch] >= THRESHOLD && curr < THRESHOLD) {{"
                )?;
                writeln!(code, "            output[ch] = -1;")?;
                writeln!(code, "        }} else {{")?;
                writeln!(code, "            output[ch] = 0;")?;
                writeln!(code, "        }}")?;
                writeln!(code)?;
                writeln!(code, "        prev[ch] = curr;")?;
                writeln!(code, "    }}")?;
                writeln!(code, "}}")?;
            }
            EncoderType::DeltaModulation => {
                writeln!(code, "void delta_modulation_encode(")?;
                writeln!(code, "    data_t input[NUM_CHANNELS],")?;
                writeln!(code, "    spike_t output[NUM_CHANNELS]")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t reference[NUM_CHANNELS];")?;
                writeln!(code)?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t diff = input[ch] - reference[ch];")?;
                writeln!(code)?;
                writeln!(code, "        if (diff > THRESHOLD) {{")?;
                writeln!(code, "            output[ch] = 1;")?;
                writeln!(
                    code,
                    "            reference[ch] = reference[ch] + THRESHOLD;"
                )?;
                writeln!(code, "        }} else if (diff < -THRESHOLD) {{")?;
                writeln!(code, "            output[ch] = -1;")?;
                writeln!(
                    code,
                    "            reference[ch] = reference[ch] - THRESHOLD;"
                )?;
                writeln!(code, "        }} else {{")?;
                writeln!(code, "            output[ch] = 0;")?;
                writeln!(code, "        }}")?;
                writeln!(code, "    }}")?;
                writeln!(code, "}}")?;
            }
            EncoderType::TemporalContrast => {
                writeln!(code, "void temporal_contrast_encode(")?;
                writeln!(code, "    data_t input[NUM_CHANNELS],")?;
                writeln!(code, "    spike_t output[NUM_CHANNELS]")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t prev[NUM_CHANNELS];")?;
                writeln!(code)?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        data_t derivative = input[ch] - prev[ch];")?;
                writeln!(code)?;
                writeln!(code, "        if (derivative > THRESHOLD) {{")?;
                writeln!(code, "            output[ch] = 1;")?;
                writeln!(code, "        }} else if (derivative < -THRESHOLD) {{")?;
                writeln!(code, "            output[ch] = -1;")?;
                writeln!(code, "        }} else {{")?;
                writeln!(code, "            output[ch] = 0;")?;
                writeln!(code, "        }}")?;
                writeln!(code)?;
                writeln!(code, "        prev[ch] = input[ch];")?;
                writeln!(code, "    }}")?;
                writeln!(code, "}}")?;
            }
            EncoderType::MovingWindow => {
                writeln!(code, "#define WINDOW_SIZE {}", config.window_size)?;
                writeln!(code)?;
                writeln!(code, "void moving_window_encode(")?;
                writeln!(code, "    data_t input[NUM_CHANNELS],")?;
                writeln!(code, "    spike_t output[NUM_CHANNELS]")?;
                writeln!(code, ") {{")?;
                writeln!(code, "    static data_t window[NUM_CHANNELS][WINDOW_SIZE];")?;
                writeln!(code, "    static int window_idx = 0;")?;
                writeln!(code)?;
                writeln!(code, "    for (int ch = 0; ch < NUM_CHANNELS; ch++) {{")?;
                writeln!(code, "        // Store new value")?;
                writeln!(code, "        window[ch][window_idx] = input[ch];")?;
                writeln!(code)?;
                writeln!(code, "        // Compute window mean")?;
                writeln!(code, "        data_t sum = 0;")?;
                writeln!(code, "        for (int i = 0; i < WINDOW_SIZE; i++) {{")?;
                writeln!(code, "            sum += window[ch][i];")?;
                writeln!(code, "        }}")?;
                writeln!(code, "        data_t mean = sum / WINDOW_SIZE;")?;
                writeln!(code)?;
                writeln!(code, "        // Detect spike based on deviation from mean")?;
                writeln!(code, "        data_t deviation = input[ch] - mean;")?;
                writeln!(code, "        if (deviation > THRESHOLD) {{")?;
                writeln!(code, "            output[ch] = 1;")?;
                writeln!(code, "        }} else if (deviation < -THRESHOLD) {{")?;
                writeln!(code, "            output[ch] = -1;")?;
                writeln!(code, "        }} else {{")?;
                writeln!(code, "            output[ch] = 0;")?;
                writeln!(code, "        }}")?;
                writeln!(code, "    }}")?;
                writeln!(code)?;
                writeln!(code, "    window_idx = (window_idx + 1) % WINDOW_SIZE;")?;
                writeln!(code, "}}")?;
            }
        }

        Ok(code)
    }
}

/// Error type for FPGA export operations.
#[derive(Debug)]
pub enum FpgaExportError {
    /// Write error.
    WriteError(std::fmt::Error),
    /// Unsupported configuration.
    UnsupportedConfig(String),
    /// IO error.
    IoError(std::io::Error),
}

impl From<std::fmt::Error> for FpgaExportError {
    fn from(err: std::fmt::Error) -> Self {
        FpgaExportError::WriteError(err)
    }
}

impl From<std::io::Error> for FpgaExportError {
    fn from(err: std::io::Error) -> Self {
        FpgaExportError::IoError(err)
    }
}

impl std::fmt::Display for FpgaExportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FpgaExportError::WriteError(e) => write!(f, "Write error: {}", e),
            FpgaExportError::UnsupportedConfig(msg) => write!(f, "Unsupported config: {}", msg),
            FpgaExportError::IoError(e) => write!(f, "IO error: {}", e),
        }
    }
}

impl std::error::Error for FpgaExportError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_level_crossing_export_xilinx() {
        let config = EncoderConfig::level_crossing(0.1, 32);
        let exporter = FpgaExporter::new(FpgaTarget::XilinxVitis);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("level_crossing_encode"));
        assert!(code.contains("#pragma HLS PIPELINE"));
    }

    #[test]
    fn test_delta_modulation_export() {
        let config = EncoderConfig::delta_modulation(0.05, 16);
        let exporter = FpgaExporter::new(FpgaTarget::XilinxVitis);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("delta_modulation_encode"));
    }

    #[test]
    fn test_intel_export() {
        let config = EncoderConfig::level_crossing(0.1, 8);
        let exporter = FpgaExporter::new(FpgaTarget::IntelHls);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("component void"));
        assert!(code.contains("ihc::stream"));
    }

    #[test]
    fn test_generic_export() {
        let config = EncoderConfig::level_crossing(0.1, 4);
        let exporter = FpgaExporter::new(FpgaTarget::GenericHls);

        let code = exporter.export(&config).unwrap();
        assert!(code.contains("#define THRESHOLD"));
    }

    #[test]
    fn test_header_export() {
        let config = EncoderConfig::level_crossing(0.1, 32);
        let exporter = FpgaExporter::new(FpgaTarget::XilinxVitis);

        let header = exporter.export_header(&config).unwrap();
        assert!(header.contains("#ifndef DPB_ENCODER_H"));
        assert!(header.contains("ap_fixed.h"));
    }
}
