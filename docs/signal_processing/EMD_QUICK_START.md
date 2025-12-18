# EMD Quick Start Guide

## Quick Import

```rust
use dpb_core::signal::emd::*;
use ndarray::Array1;
```

## 30-Second Examples

### Basic EMD
```rust
let mut emd = Emd::new(EmdConfig::default());
let imf_set = emd.decompose(&signal)?;
println!("Extracted {} IMFs", imf_set.num_imfs());
```

### EEMD (Recommended for most use cases)
```rust
let mut eemd = Eemd::new(EemdConfig {
    num_ensembles: 100,
    noise_amplitude: 0.2,
    parallel: true,
    ..Default::default()
});
let imf_set = eemd.decompose(&signal)?;
```

### Hilbert-Huang Transform
```rust
let mut hht = HilbertHuangTransform::new(EmdConfig::default());
let spectrum = hht.compute(&signal, sample_rate)?;

// Access instantaneous frequency
for (i, freq) in spectrum.instantaneous_frequencies.iter().enumerate() {
    println!("IMF {} mean freq: {:.2} Hz", i, freq.iter().sum::<f64>() / freq.len() as f64);
}
```

### Selective Reconstruction (Filtering)
```rust
let imf_set = emd.decompose(&signal)?;

// Use only first 3 IMFs (high-frequency components)
let high_freq = imf_set.reconstruct_from_indices(&[0, 1, 2])?;

// Use last 3 IMFs (low-frequency components)
let n = imf_set.num_imfs();
let low_freq = imf_set.reconstruct_from_indices(&[n-3, n-2, n-1])?;
```

## Common Biosignal Applications

### ECG Baseline Wander Removal
```rust
let mut emd = Emd::new(EmdConfig::default());
let imf_set = emd.decompose(&ecg_signal)?;

// Remove last IMF (baseline wander) and residue
let n = imf_set.num_imfs();
let clean_ecg = imf_set.reconstruct_from_indices(&(0..n-1).collect::<Vec<_>>())?;
```

### Extract Frequency Band
```rust
let mut eemd = Eemd::new(EemdConfig::default());
let mut imf_set = eemd.decompose(&signal)?;

// Compute mean frequencies
imf_set.compute_mean_frequencies(sample_rate);

// Select IMFs in desired frequency range (e.g., 8-13 Hz for alpha band)
let alpha_imfs: Vec<usize> = imf_set.imfs
    .iter()
    .enumerate()
    .filter(|(_, imf)| {
        if let Some(f) = imf.mean_frequency {
            f >= 8.0 && f <= 13.0
        } else {
            false
        }
    })
    .map(|(i, _)| i)
    .collect();

let alpha_band = imf_set.reconstruct_from_indices(&alpha_imfs)?;
```

## Algorithm Selection Guide

| Use Case | Algorithm | Config |
|----------|-----------|--------|
| Fast exploratory analysis | EMD | `max_imfs: 10` |
| Production biosignal processing | EEMD | `num_ensembles: 100` |
| High-quality research | CEEMDAN | `num_ensembles: 200` |
| Time-frequency analysis | HHT | Default |
| Known mode count | VMD | `num_modes: 5` |

## Performance Tips

1. **Use EEMD with parallel processing** for production:
   ```rust
   EemdConfig { parallel: true, ..Default::default() }
   ```

2. **Adjust ensemble size** based on quality needs:
   - Fast: 50 ensembles
   - Standard: 100 ensembles
   - High quality: 200 ensembles

3. **Set max_imfs** to avoid over-decomposition:
   ```rust
   EmdConfig { max_imfs: 8, ..Default::default() }
   ```

## Testing Your Configuration

```rust
// Test decomposition quality
let imf_set = emd.decompose(&signal)?;

// Check reconstruction error
let reconstructed = imf_set.reconstruct();
let error: f64 = signal.iter()
    .zip(reconstructed.iter())
    .map(|(a, b)| (a - b).powi(2))
    .sum::<f64>()
    .sqrt() / signal.len() as f64;

println!("Reconstruction RMSE: {:.6}", error);

// Check orthogonality (lower is better)
let oi = imf_set.orthogonality_index();
println!("Orthogonality index: {:.6}", oi);

// Check energy distribution
let energy = imf_set.energy_distribution();
for (i, e) in energy.iter().enumerate() {
    println!("IMF {} energy: {:.1}%", i, e * 100.0);
}
```

## Full Example: Multi-Step Pipeline

```rust
use dpb_core::signal::emd::*;
use dpb_core::signal::{remove_dc_offset, normalize, NormalizationMethod};
use ndarray::Array1;

fn process_biosignal(raw_signal: &Array1<f64>, sample_rate: f64) -> dpb_core::Result<()> {
    // 1. Preprocess
    let signal = remove_dc_offset(raw_signal.view());
    let signal = normalize(signal.view(), NormalizationMethod::ZScore)?;

    // 2. Decompose
    let config = EemdConfig {
        num_ensembles: 100,
        noise_amplitude: 0.2,
        parallel: true,
        seed: Some(42),
        ..Default::default()
    };

    let mut eemd = Eemd::new(config);
    let mut imf_set = eemd.decompose(&signal)?;

    // 3. Analyze
    imf_set.compute_mean_frequencies(sample_rate);
    let energy_dist = imf_set.energy_distribution();
    let oi = imf_set.orthogonality_index();

    println!("Decomposition quality:");
    println!("  Number of IMFs: {}", imf_set.num_imfs());
    println!("  Orthogonality index: {:.4}", oi);

    // 4. Extract features
    for (i, imf) in imf_set.imfs.iter().enumerate() {
        if let Some(freq) = imf.mean_frequency {
            println!("  IMF {}: {:.2} Hz ({:.1}% energy)",
                     i, freq, energy_dist[i] * 100.0);
        }
    }

    // 5. Selective reconstruction
    let relevant_imfs: Vec<usize> = (0..imf_set.num_imfs())
        .filter(|&i| energy_dist[i] > 0.05) // Keep IMFs with >5% energy
        .collect();

    let filtered_signal = imf_set.reconstruct_from_indices(&relevant_imfs)?;

    Ok(())
}
```

## Troubleshooting

### "Too few IMFs extracted"
- Increase `max_imfs`
- Decrease `min_extrema`
- Check if signal has enough oscillations

### "Too many IMFs extracted"
- Decrease `max_imfs`
- Tighten stopping criteria

### "Mode mixing observed"
- Use EEMD instead of EMD
- Increase `num_ensembles`
- Increase `noise_amplitude`

### "High reconstruction error"
- Check for signal discontinuities
- Increase `max_sift_iterations`
- Use CEEMDAN for better accuracy

### "Slow performance"
- Enable parallel processing: `parallel: true`
- Reduce `num_ensembles` for EEMD/CEEMDAN
- Reduce `max_sift_iterations`
- Use EMD instead of ensemble methods for quick analysis

## References

- Full Documentation: `docs/signal_processing/EMD_IMPLEMENTATION.md`
- Example Code: `examples/emd_demo.rs`
- API Reference: Run `cargo doc --open`
