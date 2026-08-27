fn main() {
    use dpb_export::neuromorphic::*;
    let cfg = NetworkConfig {
        name: "dpb_demo".into(),
        layers: vec![
            LayerConfig {
                name: "l0".into(),
                size: 4,
                neuron_model: NeuronModel::LIF,
                params: NeuronParams::default(),
            },
            LayerConfig {
                name: "l1".into(),
                size: 3,
                neuron_model: NeuronModel::LIF,
                params: NeuronParams::default(),
            },
        ],
        connections: vec![ConnectionConfig {
            source: "l0".into(),
            target: "l1".into(),
            conn_type: ConnectionType::FromMatrix,
            weights: Some(vec![vec![0.25, -0.5, 0.75, 1.0]; 3]),
            synapse_model: SynapseModel::Static,
            delay: 1.0,
        }],
        duration_ms: 100.0,
        dt: 1.0,
    };
    print!(
        "{}",
        NeuromorphicExporter::new(NeuromorphicTarget::Nir)
            .export(&cfg)
            .unwrap()
    );
}
