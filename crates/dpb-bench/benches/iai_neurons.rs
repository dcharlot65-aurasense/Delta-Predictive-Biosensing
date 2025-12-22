//! iai-callgrind neuron benchmarks
//!
//! These benchmarks use instruction counting via Valgrind/Callgrind to provide
//! deterministic, machine-independent performance measurements.
//!
//! Run with: cargo bench --bench iai_neurons
//! Note: Requires valgrind to be installed on the system.

use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

// Setup functions that create test data
mod setup {
    use dpb_neurons::prelude::*;
    use dpb_neurons::lif::{LifConfig, AlifConfig};
    use dpb_neurons::izhikevich::IzhikevichConfig;
    use dpb_neurons::adex::AdExConfig;
    use dpb_neurons::batch::BatchLifLayer;
    use ndarray::Array1;

    pub fn lif_neuron() -> LifNeuron {
        LifNeuron::new(LifConfig::default())
    }

    pub fn alif_neuron() -> AlifNeuron {
        AlifNeuron::new(AlifConfig::default())
    }

    pub fn izhikevich_neuron() -> IzhikevichNeuron {
        IzhikevichNeuron::new(IzhikevichConfig::default())
    }

    pub fn adex_neuron() -> AdExNeuron {
        AdExNeuron::new(AdExConfig::default())
    }

    pub fn batch_lif_10() -> (BatchLifLayer, Array1<f64>) {
        let layer = BatchLifLayer::new(10, LifConfig::default());
        let inputs = Array1::from_elem(10, 10.0);
        (layer, inputs)
    }

    pub fn batch_lif_100() -> (BatchLifLayer, Array1<f64>) {
        let layer = BatchLifLayer::new(100, LifConfig::default());
        let inputs = Array1::from_elem(100, 10.0);
        (layer, inputs)
    }

    pub fn batch_lif_1000() -> (BatchLifLayer, Array1<f64>) {
        let layer = BatchLifLayer::new(1000, LifConfig::default());
        let inputs = Array1::from_elem(1000, 10.0);
        (layer, inputs)
    }

    pub fn xylo_lif_neuron() -> dpb_neurons::XyloLifNeuron {
        dpb_neurons::XyloLifNeuron::new(dpb_neurons::hardware::XyloLifConfig::default())
    }

    pub fn quantized_lif_neuron() -> dpb_neurons::QuantizedLifNeuron {
        dpb_neurons::QuantizedLifNeuron::new(dpb_neurons::hardware::QuantizedLifConfig::default())
    }
}

// Single neuron update benchmarks
#[library_benchmark]
#[bench::lif(setup::lif_neuron())]
fn bench_lif_single_update(mut neuron: dpb_neurons::prelude::LifNeuron) -> bool {
    use dpb_neurons::traits::NeuronModel;
    black_box(neuron.update(black_box(10.0), black_box(1.0)))
}

#[library_benchmark]
#[bench::alif(setup::alif_neuron())]
fn bench_alif_single_update(mut neuron: dpb_neurons::prelude::AlifNeuron) -> bool {
    use dpb_neurons::traits::NeuronModel;
    black_box(neuron.update(black_box(10.0), black_box(1.0)))
}

#[library_benchmark]
#[bench::izhikevich(setup::izhikevich_neuron())]
fn bench_izhikevich_single_update(mut neuron: dpb_neurons::prelude::IzhikevichNeuron) -> bool {
    use dpb_neurons::traits::NeuronModel;
    black_box(neuron.update(black_box(10.0), black_box(1.0)))
}

#[library_benchmark]
#[bench::adex(setup::adex_neuron())]
fn bench_adex_single_update(mut neuron: dpb_neurons::prelude::AdExNeuron) -> bool {
    use dpb_neurons::traits::NeuronModel;
    black_box(neuron.update(black_box(10.0), black_box(1.0)))
}

// Hardware neuron benchmarks
#[library_benchmark]
#[bench::xylo(setup::xylo_lif_neuron())]
fn bench_xylo_single_update(mut neuron: dpb_neurons::XyloLifNeuron) -> bool {
    use dpb_neurons::traits::NeuronModel;
    black_box(neuron.update(black_box(10.0), black_box(1.0)))
}

#[library_benchmark]
#[bench::quantized(setup::quantized_lif_neuron())]
fn bench_quantized_single_update(mut neuron: dpb_neurons::QuantizedLifNeuron) -> bool {
    use dpb_neurons::traits::NeuronModel;
    black_box(neuron.update(black_box(10.0), black_box(1.0)))
}

// Batch neuron update benchmarks
#[library_benchmark]
#[bench::batch_10(setup::batch_lif_10())]
fn bench_batch_lif_10((mut layer, inputs): (dpb_neurons::batch::BatchLifLayer, ndarray::Array1<f64>)) -> ndarray::Array1<bool> {
    black_box(layer.update(black_box(&inputs), black_box(1.0)))
}

#[library_benchmark]
#[bench::batch_100(setup::batch_lif_100())]
fn bench_batch_lif_100((mut layer, inputs): (dpb_neurons::batch::BatchLifLayer, ndarray::Array1<f64>)) -> ndarray::Array1<bool> {
    black_box(layer.update(black_box(&inputs), black_box(1.0)))
}

#[library_benchmark]
#[bench::batch_1000(setup::batch_lif_1000())]
fn bench_batch_lif_1000((mut layer, inputs): (dpb_neurons::batch::BatchLifLayer, ndarray::Array1<f64>)) -> ndarray::Array1<bool> {
    black_box(layer.update(black_box(&inputs), black_box(1.0)))
}

// Multi-timestep benchmarks
#[library_benchmark]
#[bench::lif_100_steps(setup::lif_neuron())]
fn bench_lif_100_timesteps(mut neuron: dpb_neurons::prelude::LifNeuron) -> i32 {
    use dpb_neurons::traits::NeuronModel;
    let mut spike_count = 0;
    for _ in 0..100 {
        if neuron.update(black_box(10.0), black_box(1.0)) {
            spike_count += 1;
        }
    }
    black_box(spike_count)
}

#[library_benchmark]
#[bench::batch_100_steps(setup::batch_lif_100())]
fn bench_batch_100_timesteps((mut layer, inputs): (dpb_neurons::batch::BatchLifLayer, ndarray::Array1<f64>)) -> i32 {
    let mut spike_count = 0;
    for _ in 0..100 {
        let spikes = layer.update(black_box(&inputs), black_box(1.0));
        spike_count += spikes.iter().filter(|&&s| s).count() as i32;
    }
    black_box(spike_count)
}

// Surrogate gradient benchmarks
#[library_benchmark]
fn bench_fast_sigmoid_gradient() -> f64 {
    use dpb_neurons::prelude::FastSigmoid;
    use dpb_neurons::SurrogateGradient;
    let surrogate = FastSigmoid::default();
    let mut total = 0.0;
    for v in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
        total += surrogate.gradient(black_box(v));
    }
    black_box(total)
}

#[library_benchmark]
fn bench_superspike_gradient() -> f64 {
    use dpb_neurons::prelude::SuperSpike;
    use dpb_neurons::SurrogateGradient;
    let surrogate = SuperSpike::default();
    let mut total = 0.0;
    for v in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0] {
        total += surrogate.gradient(black_box(v));
    }
    black_box(total)
}

library_benchmark_group!(
    name = single_neuron_updates;
    benchmarks =
        bench_lif_single_update,
        bench_alif_single_update,
        bench_izhikevich_single_update,
        bench_adex_single_update
);

library_benchmark_group!(
    name = hardware_neurons;
    benchmarks =
        bench_xylo_single_update,
        bench_quantized_single_update
);

library_benchmark_group!(
    name = batch_neurons;
    benchmarks =
        bench_batch_lif_10,
        bench_batch_lif_100,
        bench_batch_lif_1000
);

library_benchmark_group!(
    name = timestep_simulations;
    benchmarks =
        bench_lif_100_timesteps,
        bench_batch_100_timesteps
);

library_benchmark_group!(
    name = surrogate_gradients;
    benchmarks =
        bench_fast_sigmoid_gradient,
        bench_superspike_gradient
);

main!(
    library_benchmark_groups =
        single_neuron_updates,
        hardware_neurons,
        batch_neurons,
        timestep_simulations,
        surrogate_gradients
);
