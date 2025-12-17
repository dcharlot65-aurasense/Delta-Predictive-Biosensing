# ANN Baseline Architectures for DPB-SNN

This document describes the 44 ANN baseline architectures implemented for fair comparison with Spiking Neural Networks (SNNs) in the Delta-Predictive Biosensing framework.

## Overview

The baselines module provides reference implementations of conventional Artificial Neural Networks (ANNs) to benchmark SNN performance across different architectural paradigms. All architectures are implemented with:

- **Consistent interface** via the `ANNBaseline` trait
- **Parameter counting** for fair model size comparisons
- **FLOPs estimation** for computational complexity analysis
- **Architecture summaries** for documentation
- **Comprehensive tests** (55 test functions)

## Module Structure

```
crates/dpb-snn/src/baselines/
├── mod.rs              # Core Tensor type and ANNBaseline trait (456 lines)
├── mlp.rs              # MLP architectures (609 lines)
├── cnn.rs              # CNN architectures (662 lines)
├── rnn.rs              # RNN/LSTM/GRU architectures (620 lines)
├── transformer.rs      # Transformer architectures (533 lines)
├── specialized.rs      # Domain-specific architectures (768 lines)
└── conversion.rs       # ANN-to-SNN conversion utilities (450 lines)

Total: 4,098 lines of code
```

## Implemented Architectures

### MLP Architectures (8)

1. **MLP2Layer** - Simple 2-layer MLP
   - Use case: Baseline for simple classification tasks
   - Parameters: ~10K-100K depending on size

2. **MLP3Layer** - 3-layer MLP
   - Use case: Medium complexity tasks
   - Parameters: ~50K-500K

3. **MLP4Layer** - 4-layer MLP
   - Use case: Deeper representations
   - Parameters: ~100K-1M

4. **MLPDropout** - MLP with dropout regularization
   - Use case: Preventing overfitting
   - Parameters: Similar to base MLP

5. **MLPBatchNorm** - MLP with batch normalization
   - Use case: Faster training convergence
   - Parameters: +2x feature dimensions for BN params

6. **MLPResidual** - MLP with residual connections
   - Use case: Very deep networks
   - Parameters: +projection layers if needed

7. **MLPWideSingle** - Wide single hidden layer
   - Use case: Universal approximation with shallow network
   - Parameters: Can exceed 1M with wide layer

8. **MLPDeep** - Deep narrow MLP (8+ layers)
   - Use case: Deep feature hierarchies
   - Parameters: ~100K-500K

### CNN Architectures (10)

9. **CNN1DSmall** - Small 1D CNN for signals
   - Use case: Basic time series classification
   - Layers: 2 conv + 2 FC
   - Parameters: ~100K

10. **CNN1DMedium** - Medium 1D CNN
    - Use case: Complex time series patterns
    - Layers: 3 conv + 2 FC
    - Parameters: ~500K

11. **CNN1DLarge** - Large 1D CNN
    - Use case: Deep feature extraction
    - Layers: 6 conv (in blocks) + 3 FC
    - Parameters: ~2M

12. **CNN1DResidual** - ResNet-style 1D CNN
    - Use case: Very deep 1D convolutions
    - Features: Skip connections
    - Parameters: ~1M

13. **CNN1DDilated** - Dilated convolutions (WaveNet-style)
    - Use case: Large receptive fields
    - Features: Exponentially growing dilation
    - Parameters: ~800K

14. **CNN2DLeNet** - LeNet-style for spectrograms
    - Use case: Classic 2D pattern recognition
    - Parameters: ~60K

15. **CNN2DVGG** - VGG-style deeper CNN
    - Use case: Deep 2D feature extraction
    - Parameters: ~15M

16. **CNN2DResNet** - ResNet blocks for 2D
    - Use case: Deep 2D networks with skip connections
    - Parameters: ~11M

17. **CNN2DMobileNet** - Efficient depthwise separable convolutions
    - Use case: Low-resource 2D classification
    - Parameters: ~3M

18. **TCN** - Temporal Convolutional Network
    - Use case: Sequence modeling with convolutions
    - Features: Causal convolutions, residual blocks
    - Parameters: ~1.2M

### RNN Architectures (10)

19. **SimpleRNN** - Basic recurrent neural network
    - Use case: Simple sequence modeling baseline
    - Parameters: ~50K-200K

20. **LSTM** - Standard Long Short-Term Memory
    - Use case: Long-term dependencies
    - Parameters: 4x larger than SimpleRNN (gates)

21. **BiLSTM** - Bidirectional LSTM
    - Use case: Sequence labeling, context from both directions
    - Parameters: 2x LSTM

22. **StackedLSTM** - Multi-layer LSTM (2-3 layers)
    - Use case: Hierarchical sequence features
    - Parameters: ~500K-2M

23. **GRU** - Gated Recurrent Unit
    - Use case: LSTM alternative with fewer parameters
    - Parameters: 3x larger than SimpleRNN

24. **BiGRU** - Bidirectional GRU
    - Use case: Bidirectional sequence modeling
    - Parameters: 2x GRU

25. **StackedGRU** - Multi-layer GRU
    - Use case: Deep GRU hierarchies
    - Parameters: ~400K-1.5M

26. **PeepholeLSTM** - LSTM with peephole connections
    - Use case: Cell state visibility for gates
    - Parameters: ~2x LSTM

27. **AttentionLSTM** - LSTM with attention mechanism
    - Use case: Selective sequence focusing
    - Parameters: LSTM + attention weights

28. **IndRNN** - Independently Recurrent Neural Network
    - Use case: Efficient recurrence with diagonal weights
    - Parameters: Much smaller than LSTM

### Transformer Architectures (8)

29. **TransformerEncoder** - Standard encoder-only transformer
    - Use case: General sequence modeling
    - Parameters: ~100K-10M depending on size
    - Complexity: O(n²) in sequence length

30. **TransformerSmall** - Small transformer (2 layers, 4 heads)
    - Use case: Low-resource tasks
    - Parameters: ~500K

31. **TransformerMedium** - Medium transformer (4 layers, 8 heads)
    - Use case: Standard NLP/time series tasks
    - Parameters: ~5M

32. **TransformerLarge** - Large transformer (6 layers, 12 heads)
    - Use case: Complex sequence tasks
    - Parameters: ~20M+

33. **LinearTransformer** - Linear attention transformer
    - Use case: Efficient long sequences
    - Complexity: O(n) in sequence length
    - Parameters: Similar to standard transformer

34. **Performer** - FAVOR+ fast attention
    - Use case: Very long sequences
    - Features: Random feature approximation
    - Complexity: O(n) with random features

35. **Informer** - ProbSparse self-attention for time series
    - Use case: Long time series forecasting
    - Complexity: O(n log n)
    - Parameters: Similar to transformer + distilling

36. **Autoformer** - Auto-correlation mechanism
    - Use case: Time series with trend/seasonality
    - Features: Series decomposition
    - Complexity: O(n log n)

### Specialized Architectures (8)

37. **ECGNet** - ECG-specific architecture
    - Use case: Cardiac signal analysis
    - Features: Multi-scale conv + residual + LSTM
    - Parameters: ~2M

38. **DeepGait** - Gait analysis network
    - Use case: Movement disorder detection
    - Features: Spatial conv + BiLSTM + Attention
    - Parameters: ~1.5M

39. **TremorNet** - Tremor classification
    - Use case: Parkinson's tremor analysis
    - Features: Multi-frequency parallel branches
    - Parameters: ~800K

40. **VoiceNet** - Voice assessment
    - Use case: Speech-based neurological assessment
    - Features: Spectrogram CNN + Prosody LSTM + Fusion
    - Parameters: ~1.2M

41. **MultimodalFusion** - Late fusion network
    - Use case: Combining multiple sensor modalities
    - Features: Per-modality encoders + fusion layer
    - Parameters: ~500K per modality

42. **AttentionFusion** - Attention-based multimodal fusion
    - Use case: Learned modality weighting
    - Features: Cross-modal attention mechanism
    - Parameters: ~600K

43. **GraphNN** - Graph neural network for skeleton data
    - Use case: Pose-based movement analysis
    - Features: Graph convolutions + node pooling
    - Parameters: ~400K

44. **HybridCNNRNN** - Combined CNN+RNN
    - Use case: Spatial-temporal feature learning
    - Features: CNN for spatial + LSTM for temporal
    - Parameters: ~1.8M

## Core Components

### Tensor Type

```rust
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}
```

Supports:
- Basic operations: `zeros()`, `ones()`, `randn()`, `from_vec()`
- Matrix operations: `matmul()`, `add()`, `mul()`, `scale()`
- Activations: `relu()`, `sigmoid()`, `tanh()`, `softmax()`
- Convolutions: `conv1d()`, `max_pool1d()`
- Normalization: `batch_norm()`, `layer_norm()`

### ANNBaseline Trait

```rust
pub trait ANNBaseline: Send + Sync {
    fn name(&self) -> &str;
    fn forward(&self, input: &Tensor) -> Tensor;
    fn num_parameters(&self) -> usize;
    fn flops_per_inference(&self) -> u64;
    fn architecture_summary(&self) -> String;
}
```

## ANN-to-SNN Conversion

### Normalization Methods

1. **DataBased** - Uses max activation from calibration data
2. **ModelBased** - Based on weight statistics
3. **Hybrid** - Combines both approaches
4. **None** - No normalization

### Threshold Strategies

1. **Fixed(t)** - Same threshold for all layers
2. **LayerWise** - Adaptive threshold per layer
3. **Percentile(p)** - Based on activation percentile
4. **Learned** - Learned during conversion

### Conversion Features

- **Weight normalization** for rate-based SNN conversion
- **Threshold balancing** to prevent activation imbalance
- **Batch normalization folding** into preceding layers
- **Bias correction** for accurate conversion
- **Calibration** using sample data for optimal thresholds

### Example Usage

```rust
use dpb_snn::baselines::*;

// Create baseline model
let mlp = MLP3Layer::new(100, 128, 64, 10, 42);

// Prepare for conversion
let weights = vec![/* ... */];
let biases = vec![/* ... */];
let calibration_data = vec![/* ... */];

// Configure conversion
let config = BaselineConversionConfig {
    weight_norm: WeightNormalizationMethod::Hybrid,
    threshold_strategy: ThresholdBalancingStrategy::LayerWise,
    num_timesteps: 200,
    bias_correction: true,
    fold_batchnorm: true,
    clip_negative_weights: false,
};

// Convert to SNN
let (snn_weights, snn_biases, converter) = convert_model_to_snn(
    weights, biases, &calibration_data, Some(config)
);

// Get conversion report
println!("{}", converter.conversion_report());
```

## Usage Examples

### Basic Forward Pass

```rust
use dpb_snn::baselines::*;

let mlp = MLP2Layer::new(100, 64, 10, 42);
let input = BaselineTensor::randn(vec![1, 100], 123);
let output = mlp.forward(&input);

println!("Architecture: {}", mlp.architecture_summary());
println!("Parameters: {}", mlp.num_parameters());
println!("FLOPs: {}", mlp.flops_per_inference());
```

### Architecture Comparison

```rust
let models: Vec<Box<dyn ANNBaseline>> = vec![
    Box::new(MLP2Layer::new(100, 64, 10, 42)),
    Box::new(CNN1DSmall::new(3, 1000, 10, 42)),
    Box::new(LSTM::new(100, 64, 10, 42)),
    Box::new(TransformerSmall::new(64, 10, 42)),
];

for model in &models {
    println!("{}: {} params, {} FLOPs",
             model.name(),
             model.num_parameters(),
             model.flops_per_inference());
}
```

## Testing

The module includes 55 comprehensive tests covering:

- Architecture instantiation
- Forward pass correctness
- Parameter counting
- Shape validation
- Conversion utilities
- Tensor operations

Run tests with:
```bash
cargo test -p dpb-snn baselines
```

## Performance Characteristics

| Architecture Type | Typical Params | Typical FLOPs | Best For |
|------------------|----------------|---------------|----------|
| MLP              | 10K-1M         | 100K-10M      | Tabular data, simple tasks |
| CNN              | 100K-15M       | 500K-20M      | Spatial/temporal patterns |
| RNN/LSTM         | 50K-5M         | 500K-50M      | Sequential data |
| Transformer      | 500K-50M       | 1M-500M       | Long sequences, NLP |
| Specialized      | 400K-2M        | 800K-2M       | Domain-specific tasks |

## References

- **MLPs**: Universal approximation, deep learning basics
- **CNNs**: LeNet, VGG, ResNet, MobileNet architectures
- **RNNs**: LSTM (Hochreiter & Schmidhuber), GRU, IndRNN
- **Transformers**: "Attention is All You Need", Informer, Autoformer
- **TCN**: Temporal Convolutional Networks (Bai et al.)
- **ANN-to-SNN**: Rate coding, threshold balancing, normalization methods

## Future Extensions

Potential additions:
- More specialized architectures (EEG, EMG, etc.)
- Advanced conversion methods (layer-wise fine-tuning)
- Hybrid SNN-ANN architectures
- Dynamic architecture search
- Quantization-aware training for SNNs

## Citation

If you use these baselines in your research, please cite:

```bibtex
@software{dpb_snn_baselines,
  title = {ANN Baselines for Delta-Predictive Biosensing SNNs},
  author = {DPB Framework Contributors},
  year = {2025},
  url = {https://github.com/yourusername/Delta-Predictive-Biosensing}
}
```

## License

Part of the Delta-Predictive Biosensing framework.
