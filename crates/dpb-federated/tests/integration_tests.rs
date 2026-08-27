//! Integration tests for dpb-federated
//!
//! Tests federated learning workflows end-to-end.
//!
//! Rewritten against the crate's actual API. The previous version of this file
//! had never compiled: it referenced `ClientUpdate`, `FedAvgAggregator`,
//! `FedConfigBuilder`, `NoiseType`, `PrivacyAccountant::compose` and a dozen
//! other names that do not exist here, so it contributed no coverage while
//! appearing to, and blocked `cargo test --workspace` outright.

use dpb_federated::*;
use std::collections::HashMap;

/// Build a weight set with a single named tensor.
fn weights(name: &str, data: Vec<f32>) -> ModelWeights {
    let shape = vec![data.len()];
    let mut params = HashMap::new();
    params.insert(name.to_string(), Tensor::new(data, shape));
    ModelWeights::new(params)
}

/// Build a client update carrying one tensor of deltas.
fn update(client: &str, round: u64, data: Vec<f32>, num_samples: usize) -> ModelUpdate {
    let shape = vec![data.len()];
    let mut changes = HashMap::new();
    changes.insert("layer0".to_string(), Tensor::new(data, shape));
    ModelUpdate::new(client, round, ParameterDelta::new(0, changes), num_samples)
}

fn delta_values(delta: &ParameterDelta) -> Vec<f32> {
    delta
        .changes
        .get("layer0")
        .expect("layer0 present")
        .data
        .clone()
}

// ── Configuration ────────────────────────────────────────────────────────

#[test]
fn test_fed_config_defaults_are_coherent() {
    let config = FedConfig::default();

    assert!(config.num_rounds > 0);
    assert!(config.min_clients > 0);
    assert!(
        (0.0..=1.0).contains(&config.client_fraction),
        "client_fraction {} outside [0, 1]",
        config.client_fraction
    );
    assert!(config.local_epochs > 0);
    assert!(config.learning_rate > 0.0);

    if let Some(max) = config.max_clients {
        assert!(max >= config.min_clients, "max_clients below min_clients");
    }
}

#[test]
fn test_fed_config_is_customisable() {
    let config = FedConfig {
        num_rounds: 100,
        min_clients: 10,
        local_epochs: 5,
        aggregation: AggregationStrategy::FedAvg,
        ..FedConfig::default()
    };

    assert_eq!(config.num_rounds, 100);
    assert_eq!(config.min_clients, 10);
    assert_eq!(config.local_epochs, 5);
    assert_eq!(config.aggregation, AggregationStrategy::FedAvg);
}

// ── Model weights and tensors ────────────────────────────────────────────

#[test]
fn test_model_weights_arithmetic() {
    let mut a = weights("layer0", vec![1.0, 2.0, 3.0]);
    let b = weights("layer0", vec![0.5, 0.5, 0.5]);

    a.add(&b).expect("shapes match");
    assert_eq!(a.get("layer0").unwrap().data, vec![1.5, 2.5, 3.5]);

    a.scale(2.0);
    assert_eq!(a.get("layer0").unwrap().data, vec![3.0, 5.0, 7.0]);
}

#[test]
fn test_model_weights_flatten_round_trip() {
    let original = weights("layer0", vec![1.0, -2.0, 3.5, 0.0]);
    let flat = original.flatten();
    assert_eq!(flat.len(), 4);

    let mut restored = ModelWeights::zeros(&[("layer0", vec![4])]);
    restored.unflatten(&flat).expect("same total size");
    assert_eq!(
        restored.get("layer0").unwrap().data,
        original.get("layer0").unwrap().data
    );
}

#[test]
fn test_model_weights_clip_norm_bounds_magnitude() {
    // L2 norm of [3, 4] is exactly 5.
    let mut w = weights("layer0", vec![3.0, 4.0]);
    assert!((w.l2_norm() - 5.0).abs() < 1e-6);

    w.clip_norm(1.0);
    assert!(
        w.l2_norm() <= 1.0 + 1e-6,
        "clipped norm {} exceeds the limit",
        w.l2_norm()
    );

    // Direction must be preserved: the ratio between components is unchanged.
    let d = &w.get("layer0").unwrap().data;
    assert!((d[1] / d[0] - 4.0 / 3.0).abs() < 1e-5);

    // A vector already inside the bound must not be touched.
    let mut small = weights("layer0", vec![0.1, 0.1]);
    let before = small.get("layer0").unwrap().data.clone();
    small.clip_norm(10.0);
    assert_eq!(small.get("layer0").unwrap().data, before);
}

#[test]
fn test_tensor_shape_and_size() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.shape, vec![2, 3]);

    assert_eq!(Tensor::zeros(vec![2, 2]).data, vec![0.0; 4]);
    assert_eq!(Tensor::ones(vec![3]).data, vec![1.0; 3]);
}

// ── Aggregation ──────────────────────────────────────────────────────────

#[test]
fn test_fedavg_averages_updates() {
    let updates = vec![
        update("c0", 1, vec![1.0, 1.0], 10),
        update("c1", 1, vec![3.0, 3.0], 10),
    ];

    let aggregated = FedAvg::new().aggregate(&updates).expect("aggregation");
    let values = delta_values(&aggregated);

    for v in values {
        assert!((v - 2.0).abs() < 1e-6, "expected the mean 2.0, got {v}");
    }
}

#[test]
fn test_weighted_avg_follows_sample_counts() {
    // 90 samples at 1.0 against 10 at 11.0 => weighted mean 2.0.
    let updates = vec![
        update("c0", 1, vec![1.0], 90),
        update("c1", 1, vec![11.0], 10),
    ];

    let aggregated = WeightedAvg::new().aggregate(&updates).expect("aggregation");
    let value = delta_values(&aggregated)[0];
    assert!(
        (value - 2.0).abs() < 1e-5,
        "weighted mean should be 2.0, got {value}"
    );

    // The unweighted mean would be 6.0; make sure the weighting really applied.
    assert!((value - 6.0).abs() > 1.0);
}

#[test]
fn test_median_resists_an_outlier() {
    let updates = vec![
        update("c0", 1, vec![1.0], 10),
        update("c1", 1, vec![2.0], 10),
        update("c2", 1, vec![1000.0], 10), // adversarial
    ];

    let median = MedianAggregator::new()
        .aggregate(&updates)
        .expect("aggregation");
    let value = delta_values(&median)[0];
    assert!(
        (value - 2.0).abs() < 1e-6,
        "median should ignore the outlier, got {value}"
    );

    // FedAvg, by contrast, is dragged away by it. That contrast is the reason
    // a robust aggregator exists.
    let mean = FedAvg::new().aggregate(&updates).expect("aggregation");
    assert!(delta_values(&mean)[0] > 100.0);
}

#[test]
fn test_trimmed_mean_discards_extremes() {
    let updates = vec![
        update("c0", 1, vec![-1000.0], 10),
        update("c1", 1, vec![1.0], 10),
        update("c2", 1, vec![2.0], 10),
        update("c3", 1, vec![3.0], 10),
        update("c4", 1, vec![1000.0], 10),
    ];

    let trimmed = TrimmedMeanAggregator::new(0.2)
        .aggregate(&updates)
        .expect("aggregation");
    let value = delta_values(&trimmed)[0];

    assert!(
        value.abs() < 10.0,
        "trimming should remove both extremes, got {value}"
    );
}

#[test]
fn test_aggregating_nothing_is_an_error() {
    assert!(FedAvg::new().aggregate(&[]).is_err());
    assert!(MedianAggregator::new().aggregate(&[]).is_err());
}

#[test]
fn test_single_client_aggregation_is_the_identity() {
    let updates = vec![update("solo", 1, vec![1.5, -2.5], 10)];
    let aggregated = FedAvg::new().aggregate(&updates).expect("aggregation");
    assert_eq!(delta_values(&aggregated), vec![1.5, -2.5]);
}

#[test]
fn test_aggregators_report_their_strategy() {
    assert_eq!(FedAvg::new().strategy(), AggregationStrategy::FedAvg);
    assert_eq!(
        WeightedAvg::new().strategy(),
        AggregationStrategy::WeightedAvg
    );
    assert_eq!(
        MedianAggregator::new().strategy(),
        AggregationStrategy::Median
    );
    assert_eq!(
        TrimmedMeanAggregator::new(0.1).strategy(),
        AggregationStrategy::TrimmedMean
    );
}

// ── Privacy ──────────────────────────────────────────────────────────────

#[test]
fn test_differential_privacy_perturbs_and_spends_budget() {
    let config = PrivacyConfig::new(1.0, 1e-5);
    let mut dp = DifferentialPrivacy::new(config);

    let (eps_before, _) = dp.remaining_budget();
    assert!(!dp.is_budget_exhausted());

    let mut w = weights("layer0", vec![0.0; 64]);
    dp.privatize(&mut w).expect("privatize");

    // Noise must actually have been added.
    let perturbed = &w.get("layer0").unwrap().data;
    assert!(
        perturbed.iter().any(|&v| v != 0.0),
        "privatize left every value untouched"
    );
    assert!(perturbed.iter().all(|v| v.is_finite()));

    // ...and it must cost budget.
    let (eps_after, _) = dp.remaining_budget();
    assert!(
        eps_after < eps_before,
        "budget did not decrease: {eps_before} -> {eps_after}"
    );
    assert!(dp.num_compositions() > 0);
}

#[test]
fn test_privacy_accountant_tracks_composition() {
    let mut accountant = PrivacyAccountant::new(1.0, 1e-5);
    assert_eq!(accountant.num_steps(), 0);

    let (eps_start, _) = accountant.compute_epsilon_basic();

    for _ in 0..10 {
        accountant.step(1.0, 0.01);
    }

    assert_eq!(accountant.num_steps(), 10);
    let (eps_after, _) = accountant.compute_epsilon_basic();
    assert!(
        eps_after > eps_start,
        "epsilon must grow with composition: {eps_start} -> {eps_after}"
    );
}

#[test]
fn test_local_dp_randomised_response_is_biased_toward_truth() {
    // At a high epsilon the mechanism should usually tell the truth.
    let dp = LocalDP::new(5.0);
    let truthful = (0..1000).filter(|_| dp.randomized_response(true)).count();
    assert!(
        truthful > 600,
        "high epsilon should mostly preserve the value, got {truthful}/1000"
    );

    // Noise must be finite and scale with sensitivity.
    let noised = dp.add_noise(1.0, 1.0);
    assert!(noised.is_finite());
}

// ── Compression ──────────────────────────────────────────────────────────

fn one_delta(data: Vec<f32>) -> ParameterDelta {
    let shape = vec![data.len()];
    let mut changes = HashMap::new();
    changes.insert("layer0".to_string(), Tensor::new(data, shape));
    ParameterDelta::new(0, changes)
}

#[test]
fn test_top_k_compression_keeps_the_largest() {
    let mut delta = one_delta(vec![0.01, 5.0, 0.02, -6.0, 0.03]);
    let mut compressor = GradientCompressor::top_k(0.4); // keep 2 of 5

    compressor.compress(&mut delta).expect("compress");
    let kept = delta_values(&delta);

    // The two large-magnitude entries survive; the rest are zeroed.
    assert_eq!(kept[1], 5.0);
    assert_eq!(kept[3], -6.0);
    assert_eq!(kept[0], 0.0);
    assert_eq!(kept[2], 0.0);
    assert_eq!(kept[4], 0.0);
}

#[test]
fn test_sign_sgd_maps_to_three_values() {
    let mut delta = one_delta(vec![2.5, -0.001, 0.0, -7.0]);
    let mut compressor = GradientCompressor::sign_sgd();

    compressor.compress(&mut delta).expect("compress");

    for v in delta_values(&delta) {
        assert!(
            v == -1.0 || v == 0.0 || v == 1.0,
            "sign compression produced {v}"
        );
    }

    let signs = delta_values(&delta);
    assert_eq!(signs[0], 1.0);
    assert_eq!(signs[1], -1.0);
    // Zero carries no direction and must stay zero.
    assert_eq!(signs[2], 0.0);
    assert_eq!(signs[3], -1.0);
}

#[test]
fn test_quantization_preserves_magnitude_roughly() {
    let original = vec![1.0, -2.0, 3.0, -4.0];
    let mut delta = one_delta(original.clone());
    let mut compressor = GradientCompressor::quantize(8);

    compressor.compress(&mut delta).expect("compress");
    compressor.decompress(&mut delta).expect("decompress");

    for (before, after) in original.iter().zip(delta_values(&delta)) {
        assert!(
            (before - after).abs() < 0.1,
            "8-bit quantization moved {before} to {after}"
        );
    }
}

#[test]
fn test_sparse_tensor_round_trip() {
    let dense = Tensor::new(vec![0.0, 3.0, 0.0, 0.0, -1.5], vec![5]);
    let sparse = SparseTensor::from_dense(&dense);
    let restored = sparse.to_dense();

    assert_eq!(restored.shape, dense.shape);
    assert_eq!(restored.data, dense.data);
}

// ── Client and server ────────────────────────────────────────────────────

#[test]
fn test_client_construction() {
    let client = FederatedClient::new("client-01", weights("layer0", vec![0.0; 8]));
    assert_eq!(client.id(), "client-01");
    assert_eq!(client.current_round(), 0);
    assert_eq!(client.model().get("layer0").unwrap().numel(), 8);
}

#[test]
fn test_server_starts_idle_and_tracks_clients() {
    let mut server = FederatedServer::new(FedConfig::default());
    assert_eq!(server.state(), &ServerState::Idle);
    assert_eq!(server.current_round(), 0);
    assert_eq!(server.num_clients(), 0);

    server.register_client("client-01").expect("register");
    assert_eq!(server.num_clients(), 1);

    server.unregister_client("client-01").expect("unregister");
    assert_eq!(server.num_clients(), 0);
}

#[test]
fn test_model_update_carries_its_provenance() {
    let u = update("client-7", 3, vec![1.0, 2.0], 256).with_loss(0.42);

    assert_eq!(u.client_id, "client-7");
    assert_eq!(u.round, 3);
    assert_eq!(u.num_samples, 256);
    assert_eq!(u.loss, Some(0.42));
    assert!(!u.delta.compressed);
}
