"""
Example usage of the DPB Python bindings

This demonstrates the main features of the dpb-python package.
"""

import numpy as np
import dpb

def example_encoder():
    """Demonstrate event-based encoding"""
    print("=== Event-Based Encoding ===")

    # Create synthetic ECG signal
    gen = dpb.synth.EcgGenerator(heart_rate=70, hrv_sdnn=50)
    signal, ground_truth = gen.generate(duration=10.0, sample_rate=250.0, seed=42)

    print(f"Generated signal: {signal}")
    print(f"Ground truth: {ground_truth}")

    # Create encoder
    encoder = dpb.encoders.LevelCrossingEncoder(threshold=0.5)

    # Encode signal to spike events
    spike_train = encoder.encode(signal)
    print(f"Encoded spike train: {spike_train}")
    print(f"Spike rate: {spike_train.spike_rate():.2f} Hz")
    print()


def example_snn():
    """Demonstrate SNN model creation"""
    print("=== Spiking Neural Network ===")

    # Build SNN model
    builder = dpb.snn.SNNBuilder()
    builder.add_linear(100, 64, neuron='lif')
    builder.add_linear(64, 10, neuron='lif')
    model = builder.build()

    print(f"Created SNN model: {model}")
    print(f"Number of layers: {len(model)}")
    print()


def example_neuron():
    """Demonstrate neuron models"""
    print("=== Neuron Models ===")

    # Create different neuron types
    lif = dpb.neurons.LifNeuron(tau=0.02, threshold=1.0)
    alif = dpb.neurons.AlifNeuron(tau=0.02, threshold=1.0)
    izh = dpb.neurons.IzhikevichNeuron(a=0.02, b=0.2, c=-65.0, d=8.0)

    # Simulate neuron
    dt = 0.001  # 1ms timestep
    input_current = 1.5

    spiked = lif.step(input_current, dt)
    print(f"LIF neuron spiked: {spiked}, voltage: {lif.voltage:.3f}")

    spiked = izh.step(input_current * 10, dt)
    print(f"Izhikevich neuron spiked: {spiked}, voltage: {izh.voltage:.3f}")
    print()


def example_training():
    """Demonstrate training infrastructure"""
    print("=== Training Infrastructure ===")

    # Create loss function and optimizer
    loss = dpb.training.SpikeCountLoss(weight=1.0)
    optimizer = dpb.training.Adam(learning_rate=0.001)

    print(f"Loss function: {loss}")
    print(f"Optimizer: {optimizer}")

    # Create trainer (with dummy model)
    builder = dpb.snn.SNNBuilder()
    builder.add_linear(10, 5, neuron='lif')
    model = builder.build()

    trainer = dpb.training.Trainer(
        model,
        loss='spike_count',
        optimizer='adam',
        learning_rate=0.001
    )
    print(f"Created trainer")
    print()


def example_metrics():
    """Demonstrate evaluation metrics"""
    print("=== Evaluation Metrics ===")

    # Create sample predictions and targets
    predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    targets = np.array([1, 0, 1, 0, 0, 1, 0, 1])

    # Compute metrics
    accuracy = dpb.metrics.Accuracy()
    precision = dpb.metrics.Precision()
    recall = dpb.metrics.Recall()
    f1 = dpb.metrics.F1Score()

    print(f"Accuracy: {accuracy.compute(predictions, targets):.3f}")
    print(f"Precision: {precision.compute(predictions, targets):.3f}")
    print(f"Recall: {recall.compute(predictions, targets):.3f}")
    print(f"F1 Score: {f1.compute(predictions, targets):.3f}")

    # Confusion matrix
    cm = dpb.metrics.ConfusionMatrix(num_classes=2)
    matrix = cm.compute(predictions, targets)
    print(f"Confusion Matrix:\n{matrix}")
    print()


def example_gpu():
    """Demonstrate GPU context"""
    print("=== GPU Context ===")

    # Check GPU availability
    if dpb.gpu.is_available():
        print("GPU is available")

        # List available devices
        devices = dpb.gpu.GpuContext.list_devices()
        print(f"Available devices: {len(devices)}")
        for i, dev in enumerate(devices):
            print(f"  Device {i}: {dev}")

        # Create GPU context
        ctx = dpb.gpu.GpuContext(device_id=0)
        ctx.initialize()
        print(f"Initialized GPU context: {ctx}")

        info = ctx.device_info()
        print(f"Device info: {info}")

        ctx.release()
    else:
        print("GPU is not available")
    print()


def example_synthetic_generators():
    """Demonstrate synthetic data generators"""
    print("=== Synthetic Data Generators ===")

    # ECG generator
    ecg_gen = dpb.synth.EcgGenerator(heart_rate=70, hrv_sdnn=50)
    ecg_signal, ecg_gt = ecg_gen.generate(duration=10.0, sample_rate=250.0)
    print(f"ECG signal: {ecg_signal}")

    # PPG generator
    ppg_gen = dpb.synth.PpgGenerator(heart_rate=75, hrv_sdnn=40)
    ppg_signal, ppg_gt = ppg_gen.generate(duration=10.0, sample_rate=100.0)
    print(f"PPG signal: {ppg_signal}")

    # Accelerometer generator
    accel_gen = dpb.synth.AccelerometerGenerator(activity='walk')
    accel_signal, accel_gt = accel_gen.generate(duration=5.0, sample_rate=50.0)
    print(f"Accelerometer signal: {accel_signal}")

    # Factory method
    gen = dpb.synth.create_generator('ecg', {'heart_rate': 80})
    signal, gt = gen.generate(duration=5.0, sample_rate=250.0)
    print(f"Factory-created signal: {signal}")
    print()


def main():
    """Run all examples"""
    print("DPB Framework Python Bindings - Example Usage")
    print("=" * 60)
    print()

    try:
        example_synthetic_generators()
        example_encoder()
        example_neuron()
        example_snn()
        example_training()
        example_metrics()
        example_gpu()

        print("=" * 60)
        print("All examples completed successfully!")

    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
