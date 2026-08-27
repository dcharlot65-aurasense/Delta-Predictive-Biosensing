//! Demonstration of signal augmentation and virtual patient cohort generation
//!
//! This example shows how to:
//! 1. Generate synthetic biosignals
//! 2. Apply various augmentation techniques (noise, temporal, spectral)
//! 3. Generate virtual patient cohorts with realistic demographics
//! 4. Create augmented datasets for ML training

use dpb_synth::{
    // Augmentation imports
    AugmentationPipeline, GaussianNoise, BaselineWander,
    PowerlineNoise, MotionArtifact, TimeWarp, MagnitudeScale,
    // Cohort imports
    CohortGenerator, Sex,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::f64::consts::PI;

fn main() {
    println!("=== Signal Augmentation and Cohort Generation Demo ===\n");

    // Example 1: Signal Augmentation Pipeline
    demonstrate_augmentation();

    // Example 2: Virtual Patient Cohort Generation
    demonstrate_cohort_generation();

    // Example 3: Combined usage for ML training data
    demonstrate_ml_dataset_generation();
}

fn demonstrate_augmentation() {
    println!("--- Example 1: Signal Augmentation Pipeline ---");

    // Create a simple synthetic ECG-like signal
    let signal: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / 250.0;  // 250 Hz sampling rate
            // Simplified ECG: sum of sinusoids
            (2.0 * PI * 1.2 * t).sin() +  // Heart rate ~72 bpm
            0.3 * (2.0 * PI * 10.0 * t).sin()  // High frequency component
        })
        .collect();

    println!("Original signal: {} samples", signal.len());

    // Create augmentation pipeline with multiple techniques
    let pipeline = AugmentationPipeline::new()
        // Add realistic ECG noise
        .add(GaussianNoise::new(20.0), 0.8)  // 80% chance of white noise
        .add(PowerlineNoise::new(60.0, 0.05), 0.5)  // 50% chance of 60 Hz interference
        .add(BaselineWander::new(0.3, 0.1), 0.6)  // 60% chance of baseline drift
        // Add temporal variations
        .add(TimeWarp::new(0.1, 4), 0.3)  // 30% chance of time warping
        // Add physiological variations
        .add(MagnitudeScale::new((0.85, 1.15)), 0.9);  // 90% chance of amplitude variation

    // Apply augmentation
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let augmented = pipeline.apply(&signal, &mut rng);

    println!("Augmented signal: {} samples", augmented.len());
    println!("Pipeline contains {} augmentations", pipeline.len());
    println!();
}

fn demonstrate_cohort_generation() {
    println!("--- Example 2: Virtual Patient Cohort Generation ---");

    // Create a cohort generator for a cardiovascular study
    let generator = CohortGenerator::new(50)
        .with_age_distribution(55.0, 12.0)  // Mean age 55, std 12 years
        .with_sex_ratio(0.6)  // 60% male
        .with_disease("Hypertension", 0.35)  // 35% prevalence
        .with_disease("Diabetes", 0.15)  // 15% prevalence
        .with_disease("CAD", 0.10);  // 10% prevalence of coronary artery disease

    // Generate cohort
    let cohort = generator.generate(42);

    println!("Generated cohort of {} patients", cohort.len());

    // Analyze cohort statistics
    let avg_age = cohort.iter().map(|p| p.demographics.age_years).sum::<f64>() / cohort.len() as f64;
    let male_count = cohort.iter().filter(|p| p.demographics.sex == Sex::Male).count();
    let htn_count = cohort.iter().filter(|p| p.has_condition("Hypertension")).count();
    let diabetes_count = cohort.iter().filter(|p| p.has_condition("Diabetes")).count();

    println!("  Average age: {:.1} years", avg_age);
    println!("  Male: {}%, Female: {}%",
             male_count * 100 / cohort.len(),
             (cohort.len() - male_count) * 100 / cohort.len());
    println!("  Hypertension: {}%", htn_count * 100 / cohort.len());
    println!("  Diabetes: {}%", diabetes_count * 100 / cohort.len());

    // Show examples of individual patients
    println!("\nSample patients:");
    for (i, patient) in cohort.iter().take(3).enumerate() {
        println!("  Patient {}: {} year old {}, BMI {:.1}, HR {:.0} bpm, HRV {:.1} ms",
                 i + 1,
                 patient.demographics.age_years.round(),
                 patient.demographics.sex.as_str(),
                 patient.demographics.bmi,
                 patient.baseline_hr,
                 patient.baseline_hrv);
        if !patient.conditions.is_empty() {
            println!("    Conditions: {}", patient.conditions.join(", "));
        }
        if !patient.medications.is_empty() {
            println!("    Medications: {}", patient.medications.join(", "));
        }
    }
    println!();
}

fn demonstrate_ml_dataset_generation() {
    println!("--- Example 3: ML Training Dataset Generation ---");

    // Generate a small cohort
    let cohort = CohortGenerator::new(10)
        .with_age_distribution(50.0, 15.0)
        .with_disease("Atrial Fibrillation", 0.3)
        .generate(123);

    // Create different augmentation strategies for each patient
    let mut rng = ChaCha8Rng::seed_from_u64(456);

    println!("Generating augmented dataset for {} patients", cohort.len());

    for patient in cohort.iter().take(3) {
        // Generate base signal tailored to patient
        let hr_hz = patient.baseline_hr / 60.0;
        let signal: Vec<f64> = (0..1000)
            .map(|i| {
                let t = i as f64 / 250.0;
                (2.0 * PI * hr_hz * t).sin()
            })
            .collect();

        // Create patient-specific augmentation pipeline
        let mut pipeline = AugmentationPipeline::new();

        // Older patients have more baseline wander
        if patient.demographics.age_years > 60.0 {
            pipeline = pipeline.add(BaselineWander::new(0.5, 0.15), 0.8);
        }

        // Patients with AFib get irregular timing
        if patient.has_condition("Atrial Fibrillation") {
            pipeline = pipeline
                .add(TimeWarp::new(0.3, 6), 0.9)
                .add(MotionArtifact::new(0.2, (0.05, 0.15)), 0.4);
        }

        // Always add some noise
        pipeline = pipeline.add(GaussianNoise::new(15.0), 1.0);

        // Generate multiple augmented versions
        let num_augmentations = 5;
        println!("  {}: Generating {} augmented samples", patient.id, num_augmentations);

        for aug_id in 0..num_augmentations {
            let augmented = pipeline.apply(&signal, &mut rng);
            println!("    Sample {}: {} samples", aug_id + 1, augmented.len());
            // In a real application, you would save these to disk or a database
        }
    }

    println!("\nDataset generation complete!");
    println!("In a real application, these would be saved for ML training.");
}
