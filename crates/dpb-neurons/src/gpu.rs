//! GPU-accelerated neuron kernels using WGSL shaders.
//!
//! This module provides GPU compute shaders for massively parallel
//! neuron simulations.

use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

/// GPU buffer layout for LIF neurons.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct GpuLifNeuron {
    /// Membrane potential in mV
    pub v: f32,
    /// Refractory timer in ms
    pub refrac: f32,
    /// Threshold in mV
    pub v_thresh: f32,
    /// Reset potential in mV
    pub v_reset: f32,
    /// Resting potential in mV
    pub v_rest: f32,
    /// Membrane time constant in ms
    pub tau_mem: f32,
    /// Membrane resistance in MΩ
    pub r_m: f32,
    /// Refractory period in ms
    pub tau_refrac: f32,
}

/// GPU buffer layout for ALIF neurons.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct GpuAlifNeuron {
    pub v: f32,
    pub adapt: f32,
    pub refrac: f32,
    pub v_thresh: f32,
    pub v_reset: f32,
    pub v_rest: f32,
    pub tau_mem: f32,
    pub tau_adapt: f32,
    pub r_m: f32,
    pub tau_refrac: f32,
    pub adapt_increment: f32,
    pub _padding: f32,
}

/// GPU buffer layout for Izhikevich neurons.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct GpuIzhikevichNeuron {
    pub v: f32,
    pub u: f32,
    pub a: f32,
    pub b: f32,
    pub c: f32,
    pub d: f32,
    pub v_thresh: f32,
    pub _padding: f32,
}

/// WGSL shader for LIF neuron update.
pub const LIF_SHADER: &str = r#"
struct LifNeuron {
    v: f32,
    refrac: f32,
    v_thresh: f32,
    v_reset: f32,
    v_rest: f32,
    tau_mem: f32,
    r_m: f32,
    tau_refrac: f32,
}

struct SimParams {
    dt: f32,
    n_neurons: u32,
}

@group(0) @binding(0) var<storage, read_write> neurons: array<LifNeuron>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_neurons) {
        return;
    }

    var neuron = neurons[idx];
    let input = inputs[idx];
    var spiked: u32 = 0u;

    // Update refractory period
    if (neuron.refrac > 0.0) {
        neuron.refrac = max(neuron.refrac - params.dt, 0.0);
        neurons[idx] = neuron;
        spikes[idx] = 0u;
        return;
    }

    // LIF dynamics: dv/dt = -(v - v_rest)/tau_mem + R_m * I / tau_mem
    let dv = (-(neuron.v - neuron.v_rest) + neuron.r_m * input) / neuron.tau_mem;
    neuron.v = neuron.v + dv * params.dt;

    // Check for spike
    if (neuron.v >= neuron.v_thresh) {
        neuron.v = neuron.v_reset;
        neuron.refrac = neuron.tau_refrac;
        spiked = 1u;
    }

    neurons[idx] = neuron;
    spikes[idx] = spiked;
}
"#;

/// WGSL shader for ALIF neuron update.
pub const ALIF_SHADER: &str = r#"
struct AlifNeuron {
    v: f32,
    adapt: f32,
    refrac: f32,
    v_thresh: f32,
    v_reset: f32,
    v_rest: f32,
    tau_mem: f32,
    tau_adapt: f32,
    r_m: f32,
    tau_refrac: f32,
    adapt_increment: f32,
    padding: f32,
}

struct SimParams {
    dt: f32,
    n_neurons: u32,
}

@group(0) @binding(0) var<storage, read_write> neurons: array<AlifNeuron>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_neurons) {
        return;
    }

    var neuron = neurons[idx];
    let input = inputs[idx];
    var spiked: u32 = 0u;

    // Update refractory period
    if (neuron.refrac > 0.0) {
        neuron.refrac = max(neuron.refrac - params.dt, 0.0);
        neurons[idx] = neuron;
        spikes[idx] = 0u;
        return;
    }

    // Update adaptation
    let da = -neuron.adapt / neuron.tau_adapt;
    neuron.adapt = neuron.adapt + da * params.dt;

    // LIF dynamics with adaptation
    let dv = (-(neuron.v - neuron.v_rest) + neuron.r_m * input) / neuron.tau_mem;
    neuron.v = neuron.v + dv * params.dt;

    // Check for spike with adaptive threshold
    let effective_thresh = neuron.v_thresh + neuron.adapt;
    if (neuron.v >= effective_thresh) {
        neuron.v = neuron.v_reset;
        neuron.adapt = neuron.adapt + neuron.adapt_increment;
        neuron.refrac = neuron.tau_refrac;
        spiked = 1u;
    }

    neurons[idx] = neuron;
    spikes[idx] = spiked;
}
"#;

/// WGSL shader for Izhikevich neuron update.
pub const IZHIKEVICH_SHADER: &str = r#"
struct IzhikevichNeuron {
    v: f32,
    u: f32,
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    v_thresh: f32,
    padding: f32,
}

struct SimParams {
    dt: f32,
    n_neurons: u32,
}

@group(0) @binding(0) var<storage, read_write> neurons: array<IzhikevichNeuron>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_neurons) {
        return;
    }

    var neuron = neurons[idx];
    let input = inputs[idx];
    var spiked: u32 = 0u;

    // Izhikevich dynamics
    // dv/dt = 0.04*v^2 + 5*v + 140 - u + I
    // du/dt = a(b*v - u)
    let dv = 0.04 * neuron.v * neuron.v + 5.0 * neuron.v + 140.0 - neuron.u + input;
    let du = neuron.a * (neuron.b * neuron.v - neuron.u);

    neuron.v = neuron.v + dv * params.dt;
    neuron.u = neuron.u + du * params.dt;

    // Check for spike
    if (neuron.v >= neuron.v_thresh) {
        neuron.v = neuron.c;
        neuron.u = neuron.u + neuron.d;
        spiked = 1u;
    }

    neurons[idx] = neuron;
    spikes[idx] = spiked;
}
"#;

/// WGSL shader for exponential LIF neuron.
pub const ELIF_SHADER: &str = r#"
struct ElifNeuron {
    v: f32,
    refrac: f32,
    v_thresh: f32,
    v_reset: f32,
    v_rest: f32,
    v_spike: f32,
    tau_mem: f32,
    delta_t: f32,
    r_m: f32,
    tau_refrac: f32,
}

struct SimParams {
    dt: f32,
    n_neurons: u32,
}

@group(0) @binding(0) var<storage, read_write> neurons: array<ElifNeuron>;
@group(0) @binding(1) var<storage, read> inputs: array<f32>;
@group(0) @binding(2) var<storage, read_write> spikes: array<u32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_neurons) {
        return;
    }

    var neuron = neurons[idx];
    let input = inputs[idx];
    var spiked: u32 = 0u;

    // Update refractory period
    if (neuron.refrac > 0.0) {
        neuron.refrac = max(neuron.refrac - params.dt, 0.0);
        neurons[idx] = neuron;
        spikes[idx] = 0u;
        return;
    }

    // Exponential spike mechanism
    let exp_arg = (neuron.v - neuron.v_thresh) / neuron.delta_t;
    var exp_term: f32;
    if (exp_arg > 10.0) {
        exp_term = neuron.delta_t * 22026.0; // exp(10)
    } else {
        exp_term = neuron.delta_t * exp(exp_arg);
    }

    let dv = (-(neuron.v - neuron.v_rest) + exp_term + neuron.r_m * input) / neuron.tau_mem;
    neuron.v = neuron.v + dv * params.dt;

    // Check for spike
    if (neuron.v >= neuron.v_spike) {
        neuron.v = neuron.v_reset;
        neuron.refrac = neuron.tau_refrac;
        spiked = 1u;
    }

    neurons[idx] = neuron;
    spikes[idx] = spiked;
}
"#;

/// GPU simulation parameters.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Pod, Zeroable)]
pub struct GpuSimParams {
    /// Time step in ms
    pub dt: f32,
    /// Number of neurons
    pub n_neurons: u32,
    pub _padding: [u32; 2],
}

impl Default for GpuSimParams {
    fn default() -> Self {
        Self {
            dt: 1.0,
            n_neurons: 0,
            _padding: [0; 2],
        }
    }
}

/// GPU neuron kernel types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuNeuronKernel {
    Lif,
    Alif,
    Izhikevich,
    Elif,
}

impl GpuNeuronKernel {
    /// Get the WGSL shader source for this kernel.
    pub fn shader_source(&self) -> &'static str {
        match self {
            GpuNeuronKernel::Lif => LIF_SHADER,
            GpuNeuronKernel::Alif => ALIF_SHADER,
            GpuNeuronKernel::Izhikevich => IZHIKEVICH_SHADER,
            GpuNeuronKernel::Elif => ELIF_SHADER,
        }
    }

    /// Get the workgroup size.
    pub fn workgroup_size(&self) -> u32 {
        256
    }

    /// Get the entry point name.
    pub fn entry_point(&self) -> &'static str {
        "main"
    }
}

/// GPU buffer descriptor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuBufferDescriptor {
    /// Buffer size in bytes
    pub size: usize,
    /// Buffer usage flags (storage, uniform, etc.)
    pub usage: String,
    /// Binding index
    pub binding: u32,
}

impl GpuBufferDescriptor {
    /// Create descriptor for neuron state buffer.
    pub fn neuron_buffer(n_neurons: usize, neuron_size: usize) -> Self {
        Self {
            size: n_neurons * neuron_size,
            usage: "STORAGE | COPY_SRC | COPY_DST".to_string(),
            binding: 0,
        }
    }

    /// Create descriptor for input buffer.
    pub fn input_buffer(n_neurons: usize) -> Self {
        Self {
            size: n_neurons * std::mem::size_of::<f32>(),
            usage: "STORAGE | COPY_DST".to_string(),
            binding: 1,
        }
    }

    /// Create descriptor for spike output buffer.
    pub fn spike_buffer(n_neurons: usize) -> Self {
        Self {
            size: n_neurons * std::mem::size_of::<u32>(),
            usage: "STORAGE | COPY_SRC".to_string(),
            binding: 2,
        }
    }

    /// Create descriptor for simulation parameters.
    pub fn params_buffer() -> Self {
        Self {
            size: std::mem::size_of::<GpuSimParams>(),
            usage: "UNIFORM | COPY_DST".to_string(),
            binding: 3,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_lif_layout() {
        let neuron = GpuLifNeuron {
            v: -65.0,
            refrac: 0.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            tau_mem: 20.0,
            r_m: 10.0,
            tau_refrac: 2.0,
        };

        assert_eq!(std::mem::size_of::<GpuLifNeuron>(), 32);
        assert_eq!(neuron.v, -65.0);
    }

    #[test]
    fn test_gpu_alif_layout() {
        let neuron = GpuAlifNeuron {
            v: -65.0,
            adapt: 0.0,
            refrac: 0.0,
            v_thresh: -50.0,
            v_reset: -65.0,
            v_rest: -65.0,
            tau_mem: 20.0,
            tau_adapt: 100.0,
            r_m: 10.0,
            tau_refrac: 2.0,
            adapt_increment: 1.0,
            _padding: 0.0,
        };

        assert_eq!(std::mem::size_of::<GpuAlifNeuron>(), 48);
        assert_eq!(neuron.adapt, 0.0);
    }

    #[test]
    fn test_gpu_izhikevich_layout() {
        let neuron = GpuIzhikevichNeuron {
            v: -65.0,
            u: -13.0,
            a: 0.02,
            b: 0.2,
            c: -65.0,
            d: 8.0,
            v_thresh: 30.0,
            _padding: 0.0,
        };

        assert_eq!(std::mem::size_of::<GpuIzhikevichNeuron>(), 32);
        assert_eq!(neuron.a, 0.02);
    }

    #[test]
    fn test_shader_sources() {
        assert!(LIF_SHADER.contains("LifNeuron"));
        assert!(ALIF_SHADER.contains("AlifNeuron"));
        assert!(IZHIKEVICH_SHADER.contains("IzhikevichNeuron"));
        assert!(ELIF_SHADER.contains("ElifNeuron"));
    }

    #[test]
    fn test_gpu_kernel_enum() {
        let lif = GpuNeuronKernel::Lif;
        assert_eq!(lif.entry_point(), "main");
        assert_eq!(lif.workgroup_size(), 256);
        assert!(lif.shader_source().len() > 0);
    }

    #[test]
    fn test_buffer_descriptors() {
        let neuron_buf = GpuBufferDescriptor::neuron_buffer(1000, 32);
        assert_eq!(neuron_buf.size, 32000);
        assert_eq!(neuron_buf.binding, 0);

        let input_buf = GpuBufferDescriptor::input_buffer(1000);
        assert_eq!(input_buf.size, 4000);
        assert_eq!(input_buf.binding, 1);

        let spike_buf = GpuBufferDescriptor::spike_buffer(1000);
        assert_eq!(spike_buf.size, 4000);
        assert_eq!(spike_buf.binding, 2);

        let params_buf = GpuBufferDescriptor::params_buffer();
        assert_eq!(params_buf.binding, 3);
    }

    #[test]
    fn test_sim_params() {
        let params = GpuSimParams {
            dt: 0.1,
            n_neurons: 10000,
            _padding: [0; 2],
        };

        assert_eq!(params.dt, 0.1);
        assert_eq!(params.n_neurons, 10000);
    }

    #[test]
    fn test_gpu_types_are_pod() {
        // These should compile if types are Pod
        let _lif_bytes = bytemuck::bytes_of(&GpuLifNeuron {
            v: 0.0,
            refrac: 0.0,
            v_thresh: 0.0,
            v_reset: 0.0,
            v_rest: 0.0,
            tau_mem: 0.0,
            r_m: 0.0,
            tau_refrac: 0.0,
        });

        let _params_bytes = bytemuck::bytes_of(&GpuSimParams::default());
    }
}
