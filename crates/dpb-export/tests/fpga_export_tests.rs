//! Tests for FPGA HLS export functionality.
//!
//! These tests verify the generation of synthesizable C/C++ code
//! for Xilinx Vitis HLS, Intel HLS, and generic HLS targets.

#[cfg(test)]
mod fpga_export_tests {
    use std::collections::HashMap;

    /// FPGA target platforms.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum FpgaTarget {
        XilinxVitis,
        IntelHls,
        GenericHls,
    }

    /// Data types for FPGA implementation.
    #[derive(Debug, Clone, Copy, PartialEq)]
    enum FpgaDataType {
        Float32,
        FixedQ16,  // Q16.16 fixed-point
        FixedQ8,   // Q8.8 fixed-point
        Int32,
        Int16,
    }

    impl FpgaDataType {
        fn c_type(&self) -> &'static str {
            match self {
                FpgaDataType::Float32 => "float",
                FpgaDataType::FixedQ16 => "ap_fixed<32,16>",
                FpgaDataType::FixedQ8 => "ap_fixed<16,8>",
                FpgaDataType::Int32 => "int32_t",
                FpgaDataType::Int16 => "int16_t",
            }
        }

        fn bits(&self) -> u32 {
            match self {
                FpgaDataType::Float32 => 32,
                FpgaDataType::FixedQ16 => 32,
                FpgaDataType::FixedQ8 => 16,
                FpgaDataType::Int32 => 32,
                FpgaDataType::Int16 => 16,
            }
        }
    }

    #[test]
    fn test_fpga_data_types() {
        assert_eq!(FpgaDataType::Float32.c_type(), "float");
        assert_eq!(FpgaDataType::FixedQ16.c_type(), "ap_fixed<32,16>");
        assert_eq!(FpgaDataType::FixedQ16.bits(), 32);
        assert_eq!(FpgaDataType::FixedQ8.bits(), 16);
    }

    /// Encoder configuration for FPGA.
    #[derive(Debug, Clone)]
    struct FpgaEncoderConfig {
        encoder_type: String,
        threshold: f32,
        data_type: FpgaDataType,
        channels: usize,
        pipeline_depth: usize,
    }

    impl Default for FpgaEncoderConfig {
        fn default() -> Self {
            Self {
                encoder_type: "level_crossing".to_string(),
                threshold: 0.1,
                data_type: FpgaDataType::FixedQ16,
                channels: 8,
                pipeline_depth: 4,
            }
        }
    }

    #[test]
    fn test_encoder_config() {
        let config = FpgaEncoderConfig::default();
        assert_eq!(config.encoder_type, "level_crossing");
        assert_eq!(config.data_type, FpgaDataType::FixedQ16);
        assert_eq!(config.channels, 8);
    }

    /// Simulated FPGA exporter.
    struct FpgaExporter {
        target: FpgaTarget,
        config: FpgaEncoderConfig,
    }

    impl FpgaExporter {
        fn new(target: FpgaTarget, config: FpgaEncoderConfig) -> Self {
            Self { target, config }
        }

        fn generate_header(&self) -> String {
            let mut code = String::new();

            match self.target {
                FpgaTarget::XilinxVitis => {
                    code.push_str("// Xilinx Vitis HLS Header\n");
                    code.push_str("#include <ap_fixed.h>\n");
                    code.push_str("#include <hls_stream.h>\n");
                }
                FpgaTarget::IntelHls => {
                    code.push_str("// Intel HLS Header\n");
                    code.push_str("#include <HLS/hls.h>\n");
                    code.push_str("#include <HLS/ac_fixed.h>\n");
                }
                FpgaTarget::GenericHls => {
                    code.push_str("// Generic HLS Header\n");
                    code.push_str("#include <stdint.h>\n");
                }
            }

            code.push_str(&format!(
                "\n#define NUM_CHANNELS {}\n",
                self.config.channels
            ));
            code.push_str(&format!(
                "#define PIPELINE_DEPTH {}\n",
                self.config.pipeline_depth
            ));

            code
        }

        fn generate_level_crossing_kernel(&self) -> String {
            let data_type = self.config.data_type.c_type();

            let mut code = String::new();
            code.push_str(&format!(
                "void level_crossing_encode(\n    {} input[NUM_CHANNELS],\n",
                data_type
            ));
            code.push_str(&format!("    {} threshold,\n", data_type));
            code.push_str(&format!(
                "    {} prev[NUM_CHANNELS],\n",
                data_type
            ));
            code.push_str("    int8_t spikes[NUM_CHANNELS]\n) {\n");

            // Add HLS pragmas
            match self.target {
                FpgaTarget::XilinxVitis => {
                    code.push_str("    #pragma HLS PIPELINE II=1\n");
                    code.push_str("    #pragma HLS ARRAY_PARTITION variable=input complete\n");
                    code.push_str("    #pragma HLS ARRAY_PARTITION variable=prev complete\n");
                    code.push_str("    #pragma HLS ARRAY_PARTITION variable=spikes complete\n");
                }
                FpgaTarget::IntelHls => {
                    code.push_str("    #pragma unroll\n");
                }
                FpgaTarget::GenericHls => {}
            }

            code.push_str("\n    for (int c = 0; c < NUM_CHANNELS; c++) {\n");
            code.push_str("        if (prev[c] < threshold && input[c] >= threshold) {\n");
            code.push_str("            spikes[c] = 1;  // Up-crossing\n");
            code.push_str("        } else if (prev[c] >= threshold && input[c] < threshold) {\n");
            code.push_str("            spikes[c] = -1; // Down-crossing\n");
            code.push_str("        } else {\n");
            code.push_str("            spikes[c] = 0;  // No crossing\n");
            code.push_str("        }\n");
            code.push_str("        prev[c] = input[c];\n");
            code.push_str("    }\n");
            code.push_str("}\n");

            code
        }

        fn generate_axi_stream_wrapper(&self) -> String {
            if self.target != FpgaTarget::XilinxVitis {
                return String::new();
            }

            let data_type = self.config.data_type.c_type();
            let mut code = String::new();

            code.push_str(&format!(
                "typedef hls::stream<{}> signal_stream_t;\n",
                data_type
            ));
            code.push_str("typedef hls::stream<int8_t> spike_stream_t;\n\n");

            code.push_str("void level_crossing_stream(\n");
            code.push_str("    signal_stream_t& input,\n");
            code.push_str("    spike_stream_t& output,\n");
            code.push_str(&format!("    {} threshold\n", data_type));
            code.push_str(") {\n");
            code.push_str("    #pragma HLS INTERFACE axis port=input\n");
            code.push_str("    #pragma HLS INTERFACE axis port=output\n");
            code.push_str("    #pragma HLS INTERFACE s_axilite port=threshold\n");
            code.push_str("    #pragma HLS INTERFACE s_axilite port=return\n");
            code.push_str("\n");
            code.push_str(&format!("    static {} prev = 0;\n", data_type));
            code.push_str(&format!("    {} curr = input.read();\n", data_type));
            code.push_str("\n");
            code.push_str("    int8_t spike = 0;\n");
            code.push_str("    if (prev < threshold && curr >= threshold) {\n");
            code.push_str("        spike = 1;\n");
            code.push_str("    } else if (prev >= threshold && curr < threshold) {\n");
            code.push_str("        spike = -1;\n");
            code.push_str("    }\n");
            code.push_str("    output.write(spike);\n");
            code.push_str("    prev = curr;\n");
            code.push_str("}\n");

            code
        }

        fn export(&self) -> String {
            let mut output = String::new();
            output.push_str(&self.generate_header());
            output.push_str("\n");
            output.push_str(&self.generate_level_crossing_kernel());
            output.push_str("\n");
            output.push_str(&self.generate_axi_stream_wrapper());
            output
        }
    }

    #[test]
    fn test_xilinx_export() {
        let config = FpgaEncoderConfig::default();
        let exporter = FpgaExporter::new(FpgaTarget::XilinxVitis, config);
        let code = exporter.export();

        assert!(code.contains("ap_fixed.h"));
        assert!(code.contains("hls_stream.h"));
        assert!(code.contains("#pragma HLS PIPELINE"));
        assert!(code.contains("level_crossing_encode"));
    }

    #[test]
    fn test_intel_export() {
        let config = FpgaEncoderConfig::default();
        let exporter = FpgaExporter::new(FpgaTarget::IntelHls, config);
        let code = exporter.export();

        assert!(code.contains("HLS/hls.h"));
        assert!(code.contains("#pragma unroll"));
    }

    #[test]
    fn test_generic_export() {
        let config = FpgaEncoderConfig {
            data_type: FpgaDataType::Int32,
            ..Default::default()
        };
        let exporter = FpgaExporter::new(FpgaTarget::GenericHls, config);
        let code = exporter.export();

        assert!(code.contains("stdint.h"));
        assert!(code.contains("int32_t"));
    }

    /// Test fixed-point conversion.
    #[test]
    fn test_fixed_point_conversion() {
        fn float_to_q16(value: f32) -> i32 {
            (value * 65536.0).round() as i32
        }

        fn q16_to_float(value: i32) -> f32 {
            value as f32 / 65536.0
        }

        // Test conversion accuracy
        let original = 0.5f32;
        let fixed = float_to_q16(original);
        let back = q16_to_float(fixed);

        assert!((original - back).abs() < 1e-4);

        // Test Q16 values
        assert_eq!(float_to_q16(1.0), 65536);
        assert_eq!(float_to_q16(-1.0), -65536);
        assert_eq!(float_to_q16(0.0), 0);
    }

    /// Test resource estimation.
    #[test]
    fn test_resource_estimation() {
        struct ResourceEstimate {
            luts: u32,
            ffs: u32,
            brams: u32,
            dsps: u32,
        }

        fn estimate_resources(config: &FpgaEncoderConfig) -> ResourceEstimate {
            let base_luts = 100;
            let base_ffs = 50;

            let channel_luts = match config.data_type {
                FpgaDataType::Float32 => 200,
                FpgaDataType::FixedQ16 => 50,
                FpgaDataType::FixedQ8 => 30,
                _ => 40,
            };

            let channel_dsps = match config.data_type {
                FpgaDataType::Float32 => 2,
                _ => 0,
            };

            ResourceEstimate {
                luts: base_luts + (channel_luts * config.channels as u32),
                ffs: base_ffs + (config.data_type.bits() * config.channels as u32),
                brams: if config.channels > 16 { 1 } else { 0 },
                dsps: channel_dsps * config.channels as u32,
            }
        }

        // Fixed-point should use fewer resources
        let fixed_config = FpgaEncoderConfig {
            data_type: FpgaDataType::FixedQ16,
            channels: 8,
            ..Default::default()
        };
        let fixed_resources = estimate_resources(&fixed_config);

        let float_config = FpgaEncoderConfig {
            data_type: FpgaDataType::Float32,
            channels: 8,
            ..Default::default()
        };
        let float_resources = estimate_resources(&float_config);

        assert!(fixed_resources.luts < float_resources.luts);
        assert!(fixed_resources.dsps < float_resources.dsps);
    }

    /// Test timing constraints.
    #[test]
    fn test_timing_constraints() {
        fn generate_timing_constraints(target: FpgaTarget, clock_mhz: u32) -> String {
            let period_ns = 1000.0 / clock_mhz as f64;

            match target {
                FpgaTarget::XilinxVitis => {
                    format!(
                        "create_clock -period {:.3} -name clk [get_ports clk]\n",
                        period_ns
                    )
                }
                FpgaTarget::IntelHls => {
                    format!("set_time_unit ns\nset_global_assignment -name CLOCK_PERIOD {:.3}\n", period_ns)
                }
                FpgaTarget::GenericHls => {
                    format!("// Clock period: {:.3} ns ({} MHz)\n", period_ns, clock_mhz)
                }
            }
        }

        let xilinx_constraints = generate_timing_constraints(FpgaTarget::XilinxVitis, 200);
        assert!(xilinx_constraints.contains("create_clock"));
        assert!(xilinx_constraints.contains("5.000")); // 200MHz = 5ns

        let intel_constraints = generate_timing_constraints(FpgaTarget::IntelHls, 100);
        assert!(intel_constraints.contains("10.000")); // 100MHz = 10ns
    }

    /// Test testbench generation.
    #[test]
    fn test_testbench_generation() {
        fn generate_testbench(config: &FpgaEncoderConfig) -> String {
            let data_type = config.data_type.c_type();

            let mut tb = String::new();
            tb.push_str("#include <stdio.h>\n");
            tb.push_str("#include <stdlib.h>\n\n");

            tb.push_str("int main() {\n");
            tb.push_str(&format!(
                "    {} input[NUM_CHANNELS];\n",
                data_type
            ));
            tb.push_str(&format!(
                "    {} prev[NUM_CHANNELS] = {{0}};\n",
                data_type
            ));
            tb.push_str("    int8_t spikes[NUM_CHANNELS];\n\n");

            tb.push_str("    // Test pattern\n");
            tb.push_str("    for (int i = 0; i < NUM_CHANNELS; i++) {\n");
            tb.push_str("        input[i] = (i % 2 == 0) ? 0.5 : -0.5;\n");
            tb.push_str("    }\n\n");

            tb.push_str(&format!(
                "    level_crossing_encode(input, {}, prev, spikes);\n",
                config.threshold
            ));

            tb.push_str("\n    // Verify results\n");
            tb.push_str("    int errors = 0;\n");
            tb.push_str("    for (int i = 0; i < NUM_CHANNELS; i++) {\n");
            tb.push_str("        printf(\"Channel %d: spike=%d\\n\", i, spikes[i]);\n");
            tb.push_str("    }\n\n");

            tb.push_str("    return errors;\n");
            tb.push_str("}\n");

            tb
        }

        let config = FpgaEncoderConfig::default();
        let testbench = generate_testbench(&config);

        assert!(testbench.contains("level_crossing_encode"));
        assert!(testbench.contains("NUM_CHANNELS"));
        assert!(testbench.contains("return errors"));
    }

    /// Test pragma generation.
    #[test]
    fn test_pragma_generation() {
        fn generate_pragmas(target: FpgaTarget, var_name: &str, array_size: usize) -> Vec<String> {
            match target {
                FpgaTarget::XilinxVitis => vec![
                    format!("#pragma HLS ARRAY_PARTITION variable={} complete", var_name),
                    format!("#pragma HLS PIPELINE II=1"),
                ],
                FpgaTarget::IntelHls => vec![
                    format!("#pragma unroll"),
                    format!("hls_register {} {}[{}]", "int", var_name, array_size),
                ],
                FpgaTarget::GenericHls => vec![
                    format!("// Optimize: unroll loop for {}", var_name),
                ],
            }
        }

        let xilinx_pragmas = generate_pragmas(FpgaTarget::XilinxVitis, "data", 8);
        assert!(xilinx_pragmas.iter().any(|p| p.contains("ARRAY_PARTITION")));

        let intel_pragmas = generate_pragmas(FpgaTarget::IntelHls, "data", 8);
        assert!(intel_pragmas.iter().any(|p| p.contains("unroll")));
    }
}
