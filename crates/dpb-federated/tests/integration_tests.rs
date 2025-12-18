//! Integration tests for dpb-federated
//!
//! Tests federated learning workflows end-to-end.

use dpb_federated::*;

#[test]
fn test_federated_config_builder() {
    let config = FedConfigBuilder::new()
        .num_clients(10)
        .rounds(100)
        .local_epochs(5)
        .aggregation_strategy(AggregationStrategy::FedAvg)
        .build();

    assert_eq!(config.num_clients, 10);
    assert_eq!(config.rounds, 100);
    assert_eq!(config.local_epochs, 5);
    assert_eq!(config.aggregation_strategy, AggregationStrategy::FedAvg);
}

#[test]
fn test_model_weights_operations() {
    // Create initial weights
    let weights1 = ModelWeights::new(vec![
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        Tensor::new(vec![4.0, 5.0], vec![2]),
    ]);

    let weights2 = ModelWeights::new(vec![
        Tensor::new(vec![3.0, 4.0, 5.0], vec![3]),
        Tensor::new(vec![6.0, 7.0], vec![2]),
    ]);

    // Test averaging
    let averaged = ModelWeights::average(&[weights1.clone(), weights2.clone()]);
    let expected_first = vec![2.0, 3.0, 4.0];
    assert_eq!(averaged.tensors[0].data, expected_first);
}

#[test]
fn test_tensor_operations() {
    let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

    assert_eq!(tensor.shape, vec![2, 2]);
    assert_eq!(tensor.numel(), 4);

    // Test scalar operations
    let scaled = tensor.scale(2.0);
    assert_eq!(scaled.data, vec![2.0, 4.0, 6.0, 8.0]);

    // Test element-wise add
    let other = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
    let added = tensor.add(&other);
    assert_eq!(added.data, vec![2.0, 3.0, 4.0, 5.0]);
}

#[test]
fn test_fedavg_aggregation() {
    let aggregator = FedAvgAggregator::new();

    let updates = vec![
        ClientUpdate {
            client_id: "client1".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![1.0, 2.0], vec![2]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client2".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![3.0, 4.0], vec![2]),
            ]),
            num_samples: 100,
            round: 0,
        },
    ];

    let result = aggregator.aggregate(&updates).unwrap();

    // FedAvg should produce simple average when samples are equal
    assert_eq!(result.tensors[0].data, vec![2.0, 3.0]);
}

#[test]
fn test_weighted_average_aggregation() {
    let aggregator = WeightedAverageAggregator::new();

    let updates = vec![
        ClientUpdate {
            client_id: "client1".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![1.0, 2.0], vec![2]),
            ]),
            num_samples: 100, // Weight 1
            round: 0,
        },
        ClientUpdate {
            client_id: "client2".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![5.0, 6.0], vec![2]),
            ]),
            num_samples: 300, // Weight 3
            round: 0,
        },
    ];

    let result = aggregator.aggregate(&updates).unwrap();

    // Weighted: (1*100 + 5*300) / 400 = 4.0, (2*100 + 6*300) / 400 = 5.0
    assert_eq!(result.tensors[0].data, vec![4.0, 5.0]);
}

#[test]
fn test_median_aggregation() {
    let aggregator = MedianAggregator::new();

    let updates = vec![
        ClientUpdate {
            client_id: "client1".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![1.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client2".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![3.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client3".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![100.0], vec![1]), // Byzantine client
            ]),
            num_samples: 100,
            round: 0,
        },
    ];

    let result = aggregator.aggregate(&updates).unwrap();

    // Median should be 3.0 (ignoring Byzantine outlier)
    assert_eq!(result.tensors[0].data, vec![3.0]);
}

#[test]
fn test_trimmed_mean_aggregation() {
    let aggregator = TrimmedMeanAggregator::new(0.2); // 20% trim

    let updates = vec![
        ClientUpdate {
            client_id: "client1".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![1.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client2".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![2.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client3".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![3.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client4".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![4.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
        ClientUpdate {
            client_id: "client5".to_string(),
            weights: ModelWeights::new(vec![
                Tensor::new(vec![5.0], vec![1]),
            ]),
            num_samples: 100,
            round: 0,
        },
    ];

    let result = aggregator.aggregate(&updates).unwrap();

    // 20% trim of 5 values = 1 from each end
    // Remaining: [2.0, 3.0, 4.0], mean = 3.0
    assert!((result.tensors[0].data[0] - 3.0).abs() < 0.01);
}

#[test]
fn test_privacy_accountant() {
    let mut accountant = PrivacyAccountant::new(1.0, 1e-5);

    // Initial budget
    assert!(accountant.remaining_epsilon() > 0.0);

    // Compose privacy after some operations
    accountant.compose(0.1, 1e-6);
    assert!(accountant.remaining_epsilon() < 1.0);

    // Check if budget is exceeded
    assert!(!accountant.budget_exceeded());
}

#[test]
fn test_differential_privacy_noise() {
    let dp = DifferentialPrivacy::new(NoiseType::Gaussian, 1.0, 1e-5, 1.0);

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let noised = dp.add_noise(&data);

    // Data should be modified
    assert_ne!(data, noised);

    // Same length
    assert_eq!(data.len(), noised.len());
}

#[test]
fn test_gradient_compression_topk() {
    let compressor = GradientCompressor::top_k(2);

    let gradient = Tensor::new(vec![0.1, 0.5, 0.2, 0.8, 0.3], vec![5]);
    let compressed = compressor.compress(&gradient);

    // Only top 2 values should be non-zero
    let non_zero_count = compressed.data.iter().filter(|&&x| x.abs() > 1e-10).count();
    assert_eq!(non_zero_count, 2);
}

#[test]
fn test_gradient_compression_random_k() {
    let compressor = GradientCompressor::random_k(3);

    let gradient = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
    let compressed = compressor.compress(&gradient);

    // Only 3 values should be retained
    let non_zero_count = compressed.data.iter().filter(|&&x| x.abs() > 1e-10).count();
    assert_eq!(non_zero_count, 3);
}

#[test]
fn test_gradient_compression_signsgd() {
    let compressor = GradientCompressor::sign_sgd();

    let gradient = Tensor::new(vec![-1.5, 0.0, 2.3, -0.1, 0.0], vec![5]);
    let compressed = compressor.compress(&gradient);

    // SignSGD should produce -1, 0, or 1
    for &val in &compressed.data {
        assert!(val == -1.0 || val == 0.0 || val == 1.0);
    }
}

#[test]
fn test_sparse_tensor() {
    let dense = Tensor::new(vec![0.0, 1.0, 0.0, 2.0, 0.0], vec![5]);
    let sparse = SparseTensor::from_dense(&dense, 1e-10);

    // Should have 2 non-zero entries
    assert_eq!(sparse.nnz(), 2);

    // Convert back to dense
    let reconstructed = sparse.to_dense();
    assert_eq!(reconstructed.data, dense.data);
}

#[test]
fn test_client_update_creation() {
    let weights = ModelWeights::new(vec![
        Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
    ]);

    let update = ClientUpdate {
        client_id: "test_client".to_string(),
        weights,
        num_samples: 500,
        round: 5,
    };

    assert_eq!(update.client_id, "test_client");
    assert_eq!(update.num_samples, 500);
    assert_eq!(update.round, 5);
}

#[test]
fn test_federated_client() {
    let config = FedConfigBuilder::new()
        .local_epochs(3)
        .learning_rate(0.01)
        .build();

    let client = FederatedClient::new("client_1".to_string(), config);

    assert_eq!(client.id(), "client_1");
    assert_eq!(client.state(), ClientState::Idle);
}

#[test]
fn test_federated_server() {
    let config = FedConfigBuilder::new()
        .num_clients(5)
        .rounds(10)
        .min_clients_per_round(3)
        .aggregation_strategy(AggregationStrategy::FedAvg)
        .build();

    let server = FederatedServer::new(config);

    assert_eq!(server.current_round(), 0);
    assert_eq!(server.state(), ServerState::WaitingForClients);
}

#[test]
fn test_parameter_delta() {
    let before = ModelWeights::new(vec![
        Tensor::new(vec![1.0, 2.0], vec![2]),
    ]);

    let after = ModelWeights::new(vec![
        Tensor::new(vec![1.5, 2.5], vec![2]),
    ]);

    let delta = ParameterDelta::compute(&before, &after);

    assert_eq!(delta.delta.tensors[0].data, vec![0.5, 0.5]);
}

#[test]
fn test_local_differential_privacy() {
    let ldp = LocalDP::new(1.0);

    let value = 5.0;
    let noised = ldp.randomize(value);

    // Should be different (with high probability)
    // The difference should be bounded by sensitivity/epsilon
    assert!((noised - value).abs() < 10.0); // Reasonable bound
}

#[test]
fn test_aggregation_strategy_variants() {
    assert_eq!(
        AggregationStrategy::FedAvg,
        AggregationStrategy::FedAvg
    );
    assert_ne!(
        AggregationStrategy::FedAvg,
        AggregationStrategy::WeightedAverage
    );
    assert_ne!(
        AggregationStrategy::Median,
        AggregationStrategy::TrimmedMean { trim_ratio: 0.1 }
    );
}

#[test]
fn test_noise_types() {
    let gaussian = NoiseType::Gaussian;
    let laplace = NoiseType::Laplace;

    assert_ne!(format!("{:?}", gaussian), format!("{:?}", laplace));
}

#[test]
fn test_privacy_budget_tracking() {
    let mut accountant = PrivacyAccountant::new(1.0, 1e-5);

    // Start with full budget
    let initial = accountant.remaining_epsilon();

    // Use some budget
    accountant.compose(0.3, 1e-6);
    let after_first = accountant.remaining_epsilon();

    // Use more budget
    accountant.compose(0.3, 1e-6);
    let after_second = accountant.remaining_epsilon();

    // Budget should decrease
    assert!(after_first < initial);
    assert!(after_second < after_first);
}

#[test]
fn test_empty_aggregation_fails() {
    let aggregator = FedAvgAggregator::new();
    let updates: Vec<ClientUpdate> = vec![];

    let result = aggregator.aggregate(&updates);
    assert!(result.is_err());
}

#[test]
fn test_single_client_aggregation() {
    let aggregator = FedAvgAggregator::new();

    let updates = vec![ClientUpdate {
        client_id: "solo".to_string(),
        weights: ModelWeights::new(vec![
            Tensor::new(vec![1.0, 2.0, 3.0], vec![3]),
        ]),
        num_samples: 100,
        round: 0,
    }];

    let result = aggregator.aggregate(&updates).unwrap();

    // Single client: weights unchanged
    assert_eq!(result.tensors[0].data, vec![1.0, 2.0, 3.0]);
}
