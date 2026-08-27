//! # Empirical Mode Decomposition (EMD) Demo
//!
//! This example demonstrates the use of EMD, EEMD, CEEMDAN, and Hilbert-Huang Transform
//! for analyzing biosignals.

use dpb_core::Result;
use dpb_core::signal::emd::{
    Ceemdan, CeemdanConfig, Eemd, EemdConfig, Emd, EmdConfig, HilbertHuangTransform,
    StoppingCriterion,
};
use ndarray::Array1;
use std::f64::consts::PI;

fn main() -> Result<()> {
    println!("=== Empirical Mode Decomposition Demo ===\n");

    // Create a synthetic biosignal: mix of multiple frequency components
    let sample_rate = 100.0; // Hz
    let duration = 10.0; // seconds
    let n = (sample_rate * duration) as usize;

    println!("Generating synthetic biosignal...");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Duration: {} seconds", duration);
    println!("  Samples: {}\n", n);

    let signal = generate_test_signal(n, sample_rate);

    // 1. Basic EMD
    demo_basic_emd(&signal)?;

    // 2. Ensemble EMD
    demo_eemd(&signal)?;

    // 3. Complete EEMD with Adaptive Noise
    demo_ceemdan(&signal)?;

    // 4. Hilbert-Huang Transform
    demo_hilbert_huang(&signal, sample_rate)?;

    // 5. IMF Analysis and Reconstruction
    demo_imf_analysis(&signal)?;

    println!("\n=== Demo Complete ===");

    Ok(())
}

/// Generates a test signal with multiple frequency components
fn generate_test_signal(n: usize, sample_rate: f64) -> Array1<f64> {
    let mut signal = Array1::zeros(n);

    for i in 0..n {
        let t = i as f64 / sample_rate;

        // Component 1: Low frequency trend (0.5 Hz)
        let trend = 2.0 * (2.0 * PI * 0.5 * t).sin();

        // Component 2: Medium frequency oscillation (5 Hz)
        let medium = 1.5 * (2.0 * PI * 5.0 * t).sin();

        // Component 3: High frequency oscillation (20 Hz)
        let high = 0.5 * (2.0 * PI * 20.0 * t).sin();

        // Component 4: Amplitude modulated signal
        let am = (1.0 + 0.5 * (2.0 * PI * 1.0 * t).cos()) * (2.0 * PI * 10.0 * t).sin();

        // Add some noise
        let noise = 0.1 * ((t * 1000.0).sin() + (t * 2000.0).cos());

        signal[i] = trend + medium + high + am + noise;
    }

    signal
}

fn demo_basic_emd(signal: &Array1<f64>) -> Result<()> {
    println!("--- 1. Basic EMD ---");

    let config = EmdConfig {
        max_imfs: 8,
        max_sift_iterations: 100,
        stopping_criterion: StoppingCriterion::Combined {
            sd_threshold: 0.2,
            s_number: 5,
        },
        min_extrema: 3,
    };

    let mut emd = Emd::new(config);
    let imf_set = emd.decompose(signal)?;

    println!("Extracted {} IMFs", imf_set.num_imfs());

    // Compute energy distribution
    let energy_dist = imf_set.energy_distribution();
    println!("\nEnergy distribution:");
    for (i, energy) in energy_dist.iter().enumerate() {
        println!("  IMF {}: {:.2}%", i + 1, energy * 100.0);
    }

    // Check reconstruction error
    let reconstructed = imf_set.reconstruct();
    let error: f64 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(orig, recon)| (orig - recon).powi(2))
        .sum::<f64>()
        .sqrt()
        / signal.len() as f64;

    println!("\nReconstruction RMSE: {:.6}", error);

    // Check orthogonality
    let oi = imf_set.orthogonality_index();
    println!("Orthogonality index: {:.6} (lower is better)", oi);

    println!();
    Ok(())
}

fn demo_eemd(signal: &Array1<f64>) -> Result<()> {
    println!("--- 2. Ensemble EMD (EEMD) ---");

    let config = EemdConfig {
        num_ensembles: 50,
        noise_amplitude: 0.2,
        emd_config: EmdConfig {
            max_imfs: 8,
            max_sift_iterations: 100,
            ..Default::default()
        },
        parallel: true,
        seed: Some(42),
    };

    let mut eemd = Eemd::new(config);
    println!("Running EEMD with {} ensemble members...", 50);

    let imf_set = eemd.decompose(signal)?;

    println!("Extracted {} IMFs", imf_set.num_imfs());

    let oi = imf_set.orthogonality_index();
    println!("Orthogonality index: {:.6}", oi);

    println!();
    Ok(())
}

fn demo_ceemdan(signal: &Array1<f64>) -> Result<()> {
    println!("--- 3. Complete EEMD with Adaptive Noise (CEEMDAN) ---");

    let config = CeemdanConfig {
        num_ensembles: 50,
        noise_amplitude: 0.2,
        emd_config: EmdConfig {
            max_imfs: 8,
            max_sift_iterations: 100,
            ..Default::default()
        },
        seed: Some(42),
    };

    let mut ceemdan = Ceemdan::new(config);
    println!("Running CEEMDAN with {} ensemble members...", 50);

    let imf_set = ceemdan.decompose(signal)?;

    println!("Extracted {} IMFs", imf_set.num_imfs());

    let oi = imf_set.orthogonality_index();
    println!("Orthogonality index: {:.6}", oi);

    println!();
    Ok(())
}

fn demo_hilbert_huang(signal: &Array1<f64>, sample_rate: f64) -> Result<()> {
    println!("--- 4. Hilbert-Huang Transform ---");

    let config = EmdConfig {
        max_imfs: 6,
        max_sift_iterations: 100,
        ..Default::default()
    };

    let mut hht = HilbertHuangTransform::new(config);
    println!("Computing Hilbert-Huang spectrum...");

    let spectrum = hht.compute(signal, sample_rate)?;

    println!("Computed spectrum with {} IMFs", spectrum.num_imfs);
    println!("Time resolution: {} points", spectrum.time.len());

    // Analyze marginal spectrum
    if let Some((freqs, marginal)) = &spectrum.marginal_spectrum {
        println!("\nMarginal Spectrum:");
        let peak_idx = marginal
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0);

        println!("  Peak frequency: {:.2} Hz", freqs[peak_idx]);
        println!("  Peak amplitude: {:.4}", marginal[peak_idx]);
    }

    // Analyze instantaneous frequencies for each IMF
    println!("\nMean instantaneous frequencies:");
    for (i, inst_freq) in spectrum.instantaneous_frequencies.iter().enumerate() {
        let mean_freq: f64 = inst_freq[10..inst_freq.len() - 10]
            .iter()
            .map(|f| f.abs())
            .sum::<f64>()
            / (inst_freq.len() - 20) as f64;
        println!("  IMF {}: {:.2} Hz", i + 1, mean_freq);
    }

    println!();
    Ok(())
}

fn demo_imf_analysis(signal: &Array1<f64>) -> Result<()> {
    println!("--- 5. IMF Analysis and Selective Reconstruction ---");

    let config = EmdConfig::default();
    let mut emd = Emd::new(config);
    let mut imf_set = emd.decompose(signal)?;

    // Compute mean frequencies
    imf_set.compute_mean_frequencies(100.0);

    println!("IMF characteristics:");
    for (i, imf) in imf_set.imfs.iter().enumerate() {
        println!("\nIMF {}:", i + 1);
        println!("  Length: {}", imf.len());
        println!("  Energy: {:.4}", imf.energy);

        if let Some(freq) = imf.mean_frequency {
            println!("  Mean frequency: {:.2} Hz", freq);
        }
    }

    // Selective reconstruction: use only low-frequency IMFs
    println!("\n\nSelective reconstruction (using only last 3 IMFs):");
    let num_imfs = imf_set.num_imfs();
    if num_imfs >= 3 {
        let indices: Vec<usize> = (num_imfs - 3..num_imfs).collect();
        let reconstructed = imf_set.reconstruct_from_indices(&indices)?;

        println!("Reconstructed signal length: {}", reconstructed.len());
        println!("This represents the low-frequency components of the signal");
    }

    // High-frequency reconstruction
    println!("\nHigh-frequency reconstruction (using first 2 IMFs):");
    if num_imfs >= 2 {
        let reconstructed = imf_set.reconstruct_from_indices(&[0, 1])?;
        println!("Reconstructed signal length: {}", reconstructed.len());
        println!("This represents the high-frequency components of the signal");
    }

    println!();
    Ok(())
}
