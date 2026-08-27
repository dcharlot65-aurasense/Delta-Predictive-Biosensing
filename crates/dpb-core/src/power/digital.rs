//! Digital/CMOS power models for conventional computing platforms.
//!
//! This module provides power estimation for traditional computing hardware
//! including GPUs, CPUs, FPGAs, and custom ASICs.

use super::{ModelStats, PowerEstimator, PowerMetrics};

/// GPU inference power estimator.
///
/// Models power consumption for deep learning inference on GPUs.
/// Reference: Typical values from NVIDIA Jetson (edge) to A100 (datacenter)
#[derive(Debug, Clone)]
pub struct GpuEstimator {
    /// GPU thermal design power in watts
    pub tdp_w: f64,
    /// Typical utilization during inference (0.0-1.0)
    pub utilization: f64,
    /// FLOPS capacity in TFLOPS
    pub tflops: f64,
}

impl Default for GpuEstimator {
    fn default() -> Self {
        // Models an NVIDIA Jetson-class edge GPU
        Self {
            tdp_w: 15.0,      // 15W TDP
            utilization: 0.7, // 70% utilization
            tflops: 0.5,      // 500 GFLOPS
        }
    }
}

impl GpuEstimator {
    /// Creates a datacenter GPU estimator (e.g., A100)
    pub fn datacenter() -> Self {
        Self {
            tdp_w: 250.0,
            utilization: 0.8,
            tflops: 312.0,
        }
    }

    /// Creates an edge GPU estimator (e.g., Jetson Nano)
    pub fn edge() -> Self {
        Self {
            tdp_w: 5.0,
            utilization: 0.6,
            tflops: 0.472,
        }
    }
}

impl PowerEstimator for GpuEstimator {
    fn name(&self) -> &str {
        "GPU"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        // For ANN inference, estimate number of MACs (multiply-accumulate operations)
        // For SNN, convert synaptic operations to equivalent MACs
        let macs = (model.num_synapses as f64) * (model.timesteps as f64) * model.spike_rate;

        // GPU power during inference
        let dynamic_power_mw = self.tdp_w * 1000.0 * self.utilization;
        let static_power_mw = self.tdp_w * 1000.0 * 0.1; // ~10% idle power

        let total_power_mw = dynamic_power_mw + static_power_mw;

        // Estimate inference time based on FLOPS
        let ops_per_second = self.tflops * 1e12;
        let inference_time_s = (2.0 * macs) / ops_per_second; // 2 ops per MAC

        let energy_per_inference_uj = total_power_mw * inference_time_s * 1000.0;

        let efficiency_tops_per_w = self.tflops / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: macs as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // GPUs don't have native spike operations, use MAC equivalent
        self.energy_per_synop()
    }

    fn energy_per_synop(&self) -> f64 {
        // Energy per MAC operation
        // TDP / (FLOPS * duty_cycle) = energy per op
        let ops_per_second = self.tflops * 1e12;
        let power_w = self.tdp_w * self.utilization;
        power_w / ops_per_second // Joules per operation
    }
}

/// CPU inference power estimator.
///
/// Models power consumption for inference on general-purpose CPUs.
/// Reference: Intel and AMD processor specifications
#[derive(Debug, Clone)]
pub struct CpuEstimator {
    /// CPU thermal design power in watts
    pub tdp_w: f64,
    /// Number of cores
    pub num_cores: usize,
    /// Active cores during inference
    pub active_cores: usize,
    /// Clock frequency in GHz
    pub frequency_ghz: f64,
}

impl Default for CpuEstimator {
    fn default() -> Self {
        // Models a typical embedded/mobile CPU
        Self {
            tdp_w: 10.0,
            num_cores: 4,
            active_cores: 4,
            frequency_ghz: 2.0,
        }
    }
}

impl CpuEstimator {
    /// Creates a server CPU estimator
    pub fn server() -> Self {
        Self {
            tdp_w: 150.0,
            num_cores: 32,
            active_cores: 32,
            frequency_ghz: 3.5,
        }
    }

    /// Creates an embedded CPU estimator
    pub fn embedded() -> Self {
        Self {
            tdp_w: 3.0,
            num_cores: 2,
            active_cores: 2,
            frequency_ghz: 1.2,
        }
    }
}

impl PowerEstimator for CpuEstimator {
    fn name(&self) -> &str {
        "CPU"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        let macs = (model.num_synapses as f64) * (model.timesteps as f64) * model.spike_rate;

        // CPU power scales with active cores
        let core_utilization = (self.active_cores as f64) / (self.num_cores as f64);
        let dynamic_power_mw = self.tdp_w * 1000.0 * core_utilization * 0.8; // 80% efficiency
        let static_power_mw = self.tdp_w * 1000.0 * 0.15; // 15% base power

        let total_power_mw = dynamic_power_mw + static_power_mw;

        // CPU is less efficient than GPU for parallel workloads
        // Estimate ~10 GFLOPS per core at 2 GHz
        let gflops_per_core = 10.0 * (self.frequency_ghz / 2.0);
        let total_gflops = gflops_per_core * (self.active_cores as f64);
        let ops_per_second = total_gflops * 1e9;

        let inference_time_s = (2.0 * macs) / ops_per_second;
        let energy_per_inference_uj = total_power_mw * inference_time_s * 1000.0;

        let efficiency_tops_per_w = (total_gflops / 1000.0) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: macs as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        self.energy_per_synop()
    }

    fn energy_per_synop(&self) -> f64 {
        // Estimate energy per operation
        let gflops_per_core = 10.0 * (self.frequency_ghz / 2.0);
        let total_gflops = gflops_per_core * (self.active_cores as f64);
        let ops_per_second = total_gflops * 1e9;

        let core_utilization = (self.active_cores as f64) / (self.num_cores as f64);
        let power_w = self.tdp_w * core_utilization * 0.8;

        power_w / ops_per_second
    }
}

/// FPGA power estimator.
///
/// Models power consumption for inference on FPGAs.
/// Reference: Xilinx and Intel FPGA datasheets
#[derive(Debug, Clone)]
pub struct FpgaEstimator {
    /// Static power in watts (leakage)
    pub static_power_w: f64,
    /// Dynamic power per operation in picojoules
    pub dynamic_power_per_op_pj: f64,
    /// DSP blocks available
    pub dsp_blocks: usize,
    /// Clock frequency in MHz
    pub frequency_mhz: f64,
}

impl Default for FpgaEstimator {
    fn default() -> Self {
        // Models a mid-range FPGA (e.g., Xilinx Zynq)
        Self {
            static_power_w: 2.0,
            dynamic_power_per_op_pj: 100.0,
            dsp_blocks: 220,
            frequency_mhz: 200.0,
        }
    }
}

impl FpgaEstimator {
    /// Creates a high-end FPGA estimator (e.g., Virtex UltraScale)
    pub fn high_end() -> Self {
        Self {
            static_power_w: 10.0,
            dynamic_power_per_op_pj: 80.0,
            dsp_blocks: 2500,
            frequency_mhz: 500.0,
        }
    }

    /// Creates a low-power FPGA estimator (e.g., for edge applications)
    pub fn low_power() -> Self {
        Self {
            static_power_w: 0.5,
            dynamic_power_per_op_pj: 150.0,
            dsp_blocks: 80,
            frequency_mhz: 100.0,
        }
    }
}

impl PowerEstimator for FpgaEstimator {
    fn name(&self) -> &str {
        "FPGA"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        let macs = (model.num_synapses as f64) * (model.timesteps as f64) * model.spike_rate;

        // Dynamic power based on operations
        let energy_per_mac_j = self.dynamic_power_per_op_pj * 1e-12;
        let total_energy_j = macs * energy_per_mac_j;

        // Estimate inference time based on DSP throughput
        let ops_per_second = (self.dsp_blocks as f64) * (self.frequency_mhz * 1e6);
        let inference_time_s = macs / ops_per_second;

        let dynamic_power_mw = (total_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = self.static_power_w * 1000.0;
        let total_power_mw = dynamic_power_mw + static_power_mw;

        let energy_per_inference_uj =
            (total_energy_j + self.static_power_w * inference_time_s) * 1e6;

        let efficiency_tops_per_w = (ops_per_second / 1e12) / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: macs as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        self.dynamic_power_per_op_pj * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        self.dynamic_power_per_op_pj * 1e-12
    }
}

/// Custom ASIC power estimator.
///
/// Models power consumption for custom neuromorphic or AI ASICs.
/// Reference: Generic ASIC design parameters
#[derive(Debug, Clone)]
pub struct AsicEstimator {
    /// Technology node in nanometers
    pub technology_nm: u32,
    /// Static power in milliwatts
    pub static_power_mw: f64,
    /// Energy per MAC in picojoules
    pub energy_per_mac_pj: f64,
    /// Peak throughput in TOPS
    pub peak_tops: f64,
}

impl Default for AsicEstimator {
    fn default() -> Self {
        // Models a 28nm neuromorphic ASIC
        Self {
            technology_nm: 28,
            static_power_mw: 5.0,
            energy_per_mac_pj: 50.0,
            peak_tops: 1.0,
        }
    }
}

impl AsicEstimator {
    /// Creates an advanced node ASIC (7nm)
    pub fn advanced_node() -> Self {
        Self {
            technology_nm: 7,
            static_power_mw: 10.0,
            energy_per_mac_pj: 10.0,
            peak_tops: 10.0,
        }
    }

    /// Creates an ultra-low-power ASIC (40nm)
    pub fn ultra_low_power() -> Self {
        Self {
            technology_nm: 40,
            static_power_mw: 1.0,
            energy_per_mac_pj: 100.0,
            peak_tops: 0.1,
        }
    }
}

impl PowerEstimator for AsicEstimator {
    fn name(&self) -> &str {
        "ASIC"
    }

    fn estimate_inference(&self, model: &ModelStats) -> PowerMetrics {
        let macs = (model.num_synapses as f64) * (model.timesteps as f64) * model.spike_rate;

        // Dynamic power from operations
        let energy_per_mac_j = self.energy_per_mac_pj * 1e-12;
        let total_dynamic_energy_j = macs * energy_per_mac_j;

        // Inference time based on throughput
        let ops_per_second = self.peak_tops * 1e12;
        let inference_time_s = macs / ops_per_second;

        let dynamic_power_mw = (total_dynamic_energy_j / inference_time_s) * 1000.0;
        let static_power_mw = self.static_power_mw;
        let total_power_mw = dynamic_power_mw + static_power_mw;

        let energy_per_inference_uj =
            (total_dynamic_energy_j + (static_power_mw / 1000.0) * inference_time_s) * 1e6;

        let efficiency_tops_per_w = self.peak_tops / (total_power_mw / 1000.0);

        PowerMetrics {
            dynamic_power_mw,
            static_power_mw,
            total_power_mw,
            energy_per_inference_uj,
            synops_per_inference: macs as u64,
            efficiency_tops_per_w,
        }
    }

    fn energy_per_spike(&self) -> f64 {
        // For neuromorphic ASIC, spike energy is lower than MAC
        self.energy_per_mac_pj * 0.1 * 1e-12
    }

    fn energy_per_synop(&self) -> f64 {
        self.energy_per_mac_pj * 1e-12
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_estimator() {
        let estimator = GpuEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 1000.0); // Should be in watts range
        assert!(metrics.energy_per_inference_uj > 0.0);
        assert_eq!(estimator.name(), "GPU");
    }

    #[test]
    fn test_gpu_variants() {
        let edge = GpuEstimator::edge();
        let datacenter = GpuEstimator::datacenter();
        let stats = ModelStats::default();

        let edge_metrics = edge.estimate_inference(&stats);
        let dc_metrics = datacenter.estimate_inference(&stats);

        // Datacenter GPU should use more power
        assert!(dc_metrics.total_power_mw > edge_metrics.total_power_mw);
        // But datacenter should be more efficient (TOPS/W)
        assert!(dc_metrics.efficiency_tops_per_w > edge_metrics.efficiency_tops_per_w);
    }

    #[test]
    fn test_cpu_estimator() {
        let estimator = CpuEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert_eq!(estimator.name(), "CPU");
    }

    #[test]
    fn test_cpu_variants() {
        let embedded = CpuEstimator::embedded();
        let server = CpuEstimator::server();
        let stats = ModelStats::default();

        let embedded_metrics = embedded.estimate_inference(&stats);
        let server_metrics = server.estimate_inference(&stats);

        assert!(server_metrics.total_power_mw > embedded_metrics.total_power_mw);
    }

    #[test]
    fn test_fpga_estimator() {
        let estimator = FpgaEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert!(metrics.static_power_mw > 0.0); // FPGAs have significant static power
        assert_eq!(estimator.name(), "FPGA");
    }

    #[test]
    fn test_fpga_variants() {
        let low_power = FpgaEstimator::low_power();
        let high_end = FpgaEstimator::high_end();
        let stats = ModelStats::default();

        let lp_metrics = low_power.estimate_inference(&stats);
        let he_metrics = high_end.estimate_inference(&stats);

        assert!(he_metrics.total_power_mw > lp_metrics.total_power_mw);
    }

    #[test]
    fn test_asic_estimator() {
        let estimator = AsicEstimator::default();
        let stats = ModelStats::default();
        let metrics = estimator.estimate_inference(&stats);

        assert!(metrics.total_power_mw > 0.0);
        assert_eq!(estimator.name(), "ASIC");
    }

    #[test]
    fn test_asic_technology_scaling() {
        let advanced = AsicEstimator::advanced_node();
        let default = AsicEstimator::default();

        // Advanced node should have better energy efficiency
        assert!(advanced.energy_per_mac_pj < default.energy_per_mac_pj);
    }

    #[test]
    fn test_power_consumption_ordering() {
        let stats = ModelStats {
            num_neurons: 10_000,
            num_synapses: 1_000_000,
            timesteps: 100,
            spike_rate: 0.1,
            ..Default::default()
        };

        let gpu = GpuEstimator::default().estimate_inference(&stats);
        let _cpu = CpuEstimator::default().estimate_inference(&stats);
        let fpga = FpgaEstimator::default().estimate_inference(&stats);

        // GPU typically uses more power than FPGA and CPU for edge devices
        assert!(gpu.total_power_mw > fpga.total_power_mw);
    }
}
