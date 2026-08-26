# DPB Python Bindings

Python bindings for the Delta-Predictive Biosensing Framework using PyO3.

## ⚠️ These bindings do not yet call the library

**Every algorithm here is a standalone placeholder, not a binding.** The crate
declares `dpb-core`, `dpb-encoders`, `dpb-neurons`, `dpb-snn` and `dpb-synth` as
dependencies and imports none of them — searching `src/` for any of those names
returns nothing. Several functions say as much in a comment:
`// Placeholder implementation - would call Rust encoder`.

So the Python API has the *shape* of the Rust library but its own behaviour, and
fixes made to the Rust crates do not reach Python users. Do not attribute
results obtained through this module to DPB, and do not publish it to PyPI in
this state.

The surface below is worth keeping as a specification of what the bindings
should expose. It is not usable as bindings until each function is wired to the
crate it names.

## Overview

This crate defines the intended Python surface for the DPB framework: encoders,
neuron models, SNN layers, training infrastructure, synthetic data generators,
and evaluation metrics.

## Installation

### From Source

```bash
# Install maturin
pip install maturin

# Build and install the package
cd crates/dpb-python
maturin develop --release

# Or build a wheel
maturin build --release
```

### Using pip (when published)

```bash
pip install dpb
```

## Quick Start

```python
import numpy as np
import dpb

# Generate synthetic ECG data
gen = dpb.synth.EcgGenerator(heart_rate=70, hrv_sdnn=50)
signal, ground_truth = gen.generate(duration=60.0, sample_rate=250.0)

# Encode signal to spike events
encoder = dpb.encoders.EcgRPeakEncoder(threshold=0.5, refractory_ms=200)
events = encoder.encode(signal)

# Create SNN model
model = dpb.snn.Sequential([
    dpb.snn.SpikingLinear(input_size=100, output_size=64, neuron="lif"),
    dpb.snn.SpikingLinear(input_size=64, output_size=10, neuron="lif"),
])

# Train model
trainer = dpb.training.Trainer(model, loss="spike_count", optimizer="adam")
history = trainer.fit(train_loader, epochs=10)
```

## Architecture

### Module Structure

```
dpb/
├── lib.rs              # Main PyO3 module registration
├── types.rs            # Core type wrappers (SpikeEvent, SpikeTrain, TimeSeries, etc.)
├── numpy_utils.rs      # NumPy integration utilities
├── encoders.rs         # Event-based encoder bindings
├── neurons.rs          # Neuron model bindings
├── snn.rs              # SNN layer and network bindings
├── training.rs         # Training infrastructure
├── synth.rs            # Synthetic data generators
├── metrics.rs          # Evaluation metrics
└── gpu.rs              # GPU context management
```

### Core Types

#### SpikeEvent

Represents a single spike event with timestamp, channel, polarity, and magnitude.

```python
event = dpb.SpikeEvent(timestamp=0.1, channel=0, polarity=1, magnitude=1.0)
```

#### SpikeTrain

Collection of spike events representing encoder output or layer activations.

```python
train = dpb.SpikeTrain(events=[event1, event2], duration=1.0, num_channels=10)
spike_rate = train.spike_rate()  # Get overall spike rate
per_channel = train.spike_rate_per_channel()  # Get per-channel rates
```

#### TimeSeries

Time series signal data with NumPy array backend.

```python
signal = dpb.TimeSeries(data=numpy_array, sample_rate=250.0, start_time=0.0)
duration = signal.duration()
channel_data = signal.get_channel(0)
```

#### GroundTruth

Ground truth annotations for synthetic or labeled data.

```python
gt = dpb.GroundTruth()
gt.add_label("modality", "ecg")
gt.add_timestamp(0.5)  # R-peak at 0.5s
gt.add_region(1.0, 2.0)  # Annotated region
```

## Encoders

### Available Encoders

- **LevelCrossingEncoder**: Threshold-based spike generation
- **TemplateDeviationEncoder**: Population template deviation
- **DerivativeEncoder**: Derivative-based encoding
- **EcgRPeakEncoder**: ECG R-peak detection
- **PpgPeakEncoder**: PPG peak detection

### Example

```python
# Level crossing encoder
encoder = dpb.encoders.LevelCrossingEncoder(
    threshold=0.5,
    positive_polarity=True,
    negative_polarity=True
)
spike_train = encoder.encode(signal)

# Factory method
encoder = dpb.encoders.create_encoder('level_crossing', {'threshold': 0.5})
```

## Neuron Models

### Available Models

- **LifNeuron**: Leaky Integrate-and-Fire
- **AlifNeuron**: Adaptive LIF
- **IzhikevichNeuron**: Izhikevich model
- **HodgkinHuxleyNeuron**: Hodgkin-Huxley model

### Example

```python
# Create LIF neuron
neuron = dpb.neurons.LifNeuron(tau=0.02, threshold=1.0, reset=0.0)

# Simulate
dt = 0.001  # 1ms timestep
input_current = 1.5
spiked = neuron.step(input_current, dt)

# Factory method
neuron = dpb.neurons.create_neuron('lif', {'tau': 0.02, 'threshold': 1.0})
```

## SNN Layers

### Available Layers

- **SpikingLinear**: Fully connected layer
- **SpikingConv2d**: 2D convolutional layer
- **SpikingRecurrent**: Recurrent layer
- **SpikingPooling**: Pooling layer

### Example

```python
# Build network with builder
builder = dpb.snn.SNNBuilder()
builder.add_linear(100, 64, neuron='lif')
builder.add_linear(64, 10, neuron='lif')
model = builder.build()

# Or create sequential model
model = dpb.snn.Sequential([
    dpb.snn.SpikingLinear(100, 64, neuron='lif'),
    dpb.snn.SpikingLinear(64, 10, neuron='lif'),
])

# Forward pass
output = model.forward(input_spikes, dt=0.001)
```

## Training

### Loss Functions

- **SpikeCountLoss**: Penalize spike count differences
- **SpikeTimeLoss**: Penalize spike timing differences
- **CrossEntropyLoss**: Standard cross-entropy

### Optimizers

- **Adam**: Adaptive moment estimation
- **SGD**: Stochastic gradient descent with momentum

### Example

```python
# Create trainer
trainer = dpb.training.Trainer(
    model,
    loss='spike_count',
    optimizer='adam',
    learning_rate=0.001
)

# Add callbacks
callback = dpb.training.Callback('my_callback')
trainer.add_callback(callback)

# Train
history = trainer.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    verbose=True
)

# Learning rate scheduling
scheduler = dpb.training.LRScheduler(
    optimizer,
    schedule_type='step',
    step_size=10,
    gamma=0.1
)
```

## Synthetic Data Generators

### Available Generators

- **EcgGenerator**: ECG signals
- **PpgGenerator**: PPG signals
- **AccelerometerGenerator**: 3-axis accelerometer data
- **EmgGenerator**: EMG signals
- **EegGenerator**: Multi-channel EEG signals

### Example

```python
# ECG generator
gen = dpb.synth.EcgGenerator(
    heart_rate=70,
    hrv_sdnn=50,
    noise_level=0.05,
    artifacts=False
)
signal, ground_truth = gen.generate(duration=60.0, sample_rate=250.0, seed=42)

# Batch generation
batch_gen = dpb.synth.BatchGenerator(gen, batch_size=32)
signals, ground_truths = batch_gen.generate_batch(duration=10.0, sample_rate=250.0)

# Factory method
gen = dpb.synth.create_generator('ppg', {'heart_rate': 75})
```

## Metrics

### Available Metrics

- **Accuracy**: Classification accuracy
- **Precision**: Precision score
- **Recall**: Recall/sensitivity
- **F1Score**: F1 score
- **MSE**: Mean squared error
- **RMSE**: Root mean squared error
- **MAE**: Mean absolute error
- **ConfusionMatrix**: Confusion matrix
- **ROCCurve**: ROC curve
- **AUC**: Area under curve
- **SpikeDistance**: Spike train distance

### Example

```python
# Single metric
accuracy = dpb.metrics.Accuracy()
score = accuracy.compute(predictions, targets)

# Multiple metrics
collection = dpb.metrics.MetricCollection([
    dpb.metrics.Accuracy(),
    dpb.metrics.Precision(),
    dpb.metrics.Recall(),
    dpb.metrics.F1Score(),
])
results = collection.compute(predictions, targets)

# Confusion matrix
cm = dpb.metrics.ConfusionMatrix(num_classes=3)
matrix = cm.compute(predictions, targets)
```

## GPU Support

### GPU Context Management

```python
# Check availability
if dpb.gpu.is_available():
    # List devices
    devices = dpb.gpu.GpuContext.list_devices()

    # Create context
    ctx = dpb.gpu.GpuContext(device_id=0)
    ctx.initialize()

    # Get device info
    info = ctx.device_info()
    print(f"Using {info.name}")

    # Memory stats
    stats = ctx.memory_stats()
    print(f"GPU memory: {stats['used']} / {stats['total']}")

    ctx.release()

# Or use context manager
with dpb.gpu.GpuContext() as ctx:
    # GPU operations here
    pass
```

### GPU Buffers and Shaders

```python
# Create buffer
buffer = dpb.gpu.GpuBuffer(size=1024, usage='storage')
buffer.allocate(ctx)
buffer.write(data)
result = buffer.read()

# Create shader
shader = dpb.gpu.GpuShader(source=shader_code, entry_point='main')
shader.compile(ctx)
shader.dispatch(workgroups=(64, 1, 1))

# Profiling
profiler = dpb.gpu.GpuProfiler()
profiler.start('operation')
# ... GPU operations ...
profiler.stop('operation')
stats = profiler.get_stats()
```

## NumPy Integration

The bindings provide zero-copy NumPy integration where possible:

```python
# TimeSeries uses NumPy arrays directly
data = np.random.randn(1000, 3)  # 1000 samples, 3 channels
signal = dpb.TimeSeries(data=data, sample_rate=100.0)

# Get channel as NumPy array
channel_data = signal.get_channel(0)  # Returns NumPy array

# Time axis
time = signal.time_axis()  # Returns NumPy array
```

## Development

### Building

```bash
# Check for errors
cargo check

# Run tests
cargo test

# Build Python package
maturin develop

# Build release wheel
maturin build --release
```

### Testing

```bash
# Run Python tests
pytest tests/

# With coverage
pytest tests/ --cov=dpb --cov-report=html
```

## Performance Notes

- NumPy arrays are used for zero-copy data transfer where possible
- GPU operations are asynchronous when supported
- Batch processing is optimized for throughput
- Type conversions are minimized

## License

MIT OR Apache-2.0

## Credits

Developed by AuraSense Tech Corporation as part of the Delta-Predictive Biosensing Framework.
