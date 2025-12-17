//! Multi-Layer Perceptron (MLP) baseline architectures

use super::{ANNBaseline, Tensor, count_params, xavier_init};

/// 1. MLP with 2 layers
pub struct MLP2Layer {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    hidden_size: usize,
}

impl MLP2Layer {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w1: xavier_init(vec![input_size, hidden_size], seed),
            b1: Tensor::zeros(vec![hidden_size]),
            w2: xavier_init(vec![hidden_size, output_size], seed + 1),
            b2: Tensor::zeros(vec![output_size]),
            hidden_size,
        }
    }
}

impl ANNBaseline for MLP2Layer {
    fn name(&self) -> &str {
        "MLP2Layer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let h = input.matmul(&self.w1).add(&self.b1).relu();
        h.matmul(&self.w2).add(&self.b2)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w1.shape) + count_params(&self.b1.shape) +
        count_params(&self.w2.shape) + count_params(&self.b2.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let w1_ops = (self.w1.shape[0] * self.w1.shape[1] * 2) as u64;
        let w2_ops = (self.w2.shape[0] * self.w2.shape[1] * 2) as u64;
        w1_ops + w2_ops
    }

    fn architecture_summary(&self) -> String {
        format!("MLP2Layer: {} -> {} -> {}",
                self.w1.shape[0], self.hidden_size, self.w2.shape[1])
    }
}

/// 2. MLP with 3 layers
pub struct MLP3Layer {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    w3: Tensor,
    b3: Tensor,
}

impl MLP3Layer {
    pub fn new(input_size: usize, hidden1: usize, hidden2: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w1: xavier_init(vec![input_size, hidden1], seed),
            b1: Tensor::zeros(vec![hidden1]),
            w2: xavier_init(vec![hidden1, hidden2], seed + 1),
            b2: Tensor::zeros(vec![hidden2]),
            w3: xavier_init(vec![hidden2, output_size], seed + 2),
            b3: Tensor::zeros(vec![output_size]),
        }
    }
}

impl ANNBaseline for MLP3Layer {
    fn name(&self) -> &str {
        "MLP3Layer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let h1 = input.matmul(&self.w1).add(&self.b1).relu();
        let h2 = h1.matmul(&self.w2).add(&self.b2).relu();
        h2.matmul(&self.w3).add(&self.b3)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w1.shape) + count_params(&self.b1.shape) +
        count_params(&self.w2.shape) + count_params(&self.b2.shape) +
        count_params(&self.w3.shape) + count_params(&self.b3.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let w1_ops = (self.w1.shape[0] * self.w1.shape[1] * 2) as u64;
        let w2_ops = (self.w2.shape[0] * self.w2.shape[1] * 2) as u64;
        let w3_ops = (self.w3.shape[0] * self.w3.shape[1] * 2) as u64;
        w1_ops + w2_ops + w3_ops
    }

    fn architecture_summary(&self) -> String {
        format!("MLP3Layer: {} -> {} -> {} -> {}",
                self.w1.shape[0], self.w2.shape[0], self.w3.shape[0], self.w3.shape[1])
    }
}

/// 3. MLP with 4 layers
pub struct MLP4Layer {
    layers: Vec<(Tensor, Tensor)>, // (weight, bias) pairs
}

impl MLP4Layer {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, seed: u64) -> Self {
        assert_eq!(hidden_sizes.len(), 3, "MLP4Layer requires 3 hidden layers");

        let mut layers = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            layers.push((
                xavier_init(vec![in_size, h_size], seed + i as u64),
                Tensor::zeros(vec![h_size]),
            ));
            in_size = h_size;
        }

        layers.push((
            xavier_init(vec![in_size, output_size], seed + 3),
            Tensor::zeros(vec![output_size]),
        ));

        Self { layers }
    }
}

impl ANNBaseline for MLP4Layer {
    fn name(&self) -> &str {
        "MLP4Layer"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum()
    }

    fn flops_per_inference(&self) -> u64 {
        self.layers.iter()
            .map(|(w, _)| (w.shape[0] * w.shape[1] * 2) as u64)
            .sum()
    }

    fn architecture_summary(&self) -> String {
        let sizes: Vec<String> = self.layers.iter()
            .map(|(w, _)| w.shape[1].to_string())
            .collect();
        format!("MLP4Layer: {} -> {}", self.layers[0].0.shape[0], sizes.join(" -> "))
    }
}

/// 4. MLP with dropout
pub struct MLPDropout {
    layers: Vec<(Tensor, Tensor)>,
    dropout_rate: f32,
}

impl MLPDropout {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, dropout_rate: f32, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            layers.push((
                xavier_init(vec![in_size, h_size], seed + i as u64),
                Tensor::zeros(vec![h_size]),
            ));
            in_size = h_size;
        }

        layers.push((
            xavier_init(vec![in_size, output_size], seed + hidden_sizes.len() as u64),
            Tensor::zeros(vec![output_size]),
        ));

        Self { layers, dropout_rate }
    }
}

impl ANNBaseline for MLPDropout {
    fn name(&self) -> &str {
        "MLPDropout"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.layers.len() - 1 {
                x = x.relu().dropout(self.dropout_rate);
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum()
    }

    fn flops_per_inference(&self) -> u64 {
        self.layers.iter()
            .map(|(w, _)| (w.shape[0] * w.shape[1] * 2) as u64)
            .sum()
    }

    fn architecture_summary(&self) -> String {
        format!("MLPDropout (p={}): {} layers", self.dropout_rate, self.layers.len())
    }
}

/// 5. MLP with batch normalization
pub struct MLPBatchNorm {
    layers: Vec<(Tensor, Tensor)>,
    bn_params: Vec<(Tensor, Tensor, Tensor, Tensor)>, // (mean, var, gamma, beta)
}

impl MLPBatchNorm {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut bn_params = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            layers.push((
                xavier_init(vec![in_size, h_size], seed + i as u64 * 2),
                Tensor::zeros(vec![h_size]),
            ));
            bn_params.push((
                Tensor::zeros(vec![h_size]),     // mean
                Tensor::ones(vec![h_size]),      // var
                Tensor::ones(vec![h_size]),      // gamma
                Tensor::zeros(vec![h_size]),     // beta
            ));
            in_size = h_size;
        }

        layers.push((
            xavier_init(vec![in_size, output_size], seed + hidden_sizes.len() as u64 * 2),
            Tensor::zeros(vec![output_size]),
        ));

        Self { layers, bn_params }
    }
}

impl ANNBaseline for MLPBatchNorm {
    fn name(&self) -> &str {
        "MLPBatchNorm"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = x.matmul(w).add(b);

            if i < self.bn_params.len() {
                let (mean, var, gamma, beta) = &self.bn_params[i];
                x = x.batch_norm(mean, var, gamma, beta).relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let bn_params: usize = self.bn_params.iter()
            .map(|(mean, var, gamma, beta)| {
                count_params(&mean.shape) + count_params(&var.shape) +
                count_params(&gamma.shape) + count_params(&beta.shape)
            })
            .sum();

        layer_params + bn_params
    }

    fn flops_per_inference(&self) -> u64 {
        let matmul_ops: u64 = self.layers.iter()
            .map(|(w, _)| (w.shape[0] * w.shape[1] * 2) as u64)
            .sum();

        let bn_ops: u64 = self.bn_params.iter()
            .map(|(mean, _, _, _)| (mean.shape[0] * 10) as u64) // Rough estimate
            .sum();

        matmul_ops + bn_ops
    }

    fn architecture_summary(&self) -> String {
        format!("MLPBatchNorm: {} layers with BN", self.layers.len())
    }
}

/// 6. MLP with residual connections
pub struct MLPResidual {
    layers: Vec<(Tensor, Tensor)>,
    projections: Vec<Option<Tensor>>, // Projection layers for dimension mismatch
}

impl MLPResidual {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut projections = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            layers.push((
                xavier_init(vec![in_size, h_size], seed + i as u64 * 2),
                Tensor::zeros(vec![h_size]),
            ));

            // Add projection if dimensions don't match
            if in_size != h_size {
                projections.push(Some(xavier_init(vec![in_size, h_size], seed + i as u64 * 2 + 1)));
            } else {
                projections.push(None);
            }

            in_size = h_size;
        }

        layers.push((
            xavier_init(vec![in_size, output_size], seed + hidden_sizes.len() as u64 * 2),
            Tensor::zeros(vec![output_size]),
        ));
        projections.push(None);

        Self { layers, projections }
    }
}

impl ANNBaseline for MLPResidual {
    fn name(&self) -> &str {
        "MLPResidual"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for (i, (w, b)) in self.layers.iter().enumerate() {
            let residual = if let Some(proj) = &self.projections[i] {
                x.matmul(proj)
            } else {
                x.clone()
            };

            x = x.matmul(w).add(b);

            if i < self.layers.len() - 1 {
                x = x.add(&residual).relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        let proj_params: usize = self.projections.iter()
            .filter_map(|p| p.as_ref())
            .map(|w| count_params(&w.shape))
            .sum();

        layer_params + proj_params
    }

    fn flops_per_inference(&self) -> u64 {
        let layer_ops: u64 = self.layers.iter()
            .map(|(w, _)| (w.shape[0] * w.shape[1] * 2) as u64)
            .sum();

        let proj_ops: u64 = self.projections.iter()
            .filter_map(|p| p.as_ref())
            .map(|w| (w.shape[0] * w.shape[1] * 2) as u64)
            .sum();

        layer_ops + proj_ops
    }

    fn architecture_summary(&self) -> String {
        format!("MLPResidual: {} layers with skip connections", self.layers.len())
    }
}

/// 7. Wide single hidden layer MLP
pub struct MLPWideSingle {
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
}

impl MLPWideSingle {
    pub fn new(input_size: usize, wide_hidden: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w1: xavier_init(vec![input_size, wide_hidden], seed),
            b1: Tensor::zeros(vec![wide_hidden]),
            w2: xavier_init(vec![wide_hidden, output_size], seed + 1),
            b2: Tensor::zeros(vec![output_size]),
        }
    }
}

impl ANNBaseline for MLPWideSingle {
    fn name(&self) -> &str {
        "MLPWideSingle"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let h = input.matmul(&self.w1).add(&self.b1).relu();
        h.matmul(&self.w2).add(&self.b2)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w1.shape) + count_params(&self.b1.shape) +
        count_params(&self.w2.shape) + count_params(&self.b2.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let w1_ops = (self.w1.shape[0] * self.w1.shape[1] * 2) as u64;
        let w2_ops = (self.w2.shape[0] * self.w2.shape[1] * 2) as u64;
        w1_ops + w2_ops
    }

    fn architecture_summary(&self) -> String {
        format!("MLPWideSingle: {} -> {} -> {} (wide shallow)",
                self.w1.shape[0], self.w1.shape[1], self.w2.shape[1])
    }
}

/// 8. Deep narrow MLP (8+ layers)
pub struct MLPDeep {
    layers: Vec<(Tensor, Tensor)>,
}

impl MLPDeep {
    pub fn new(input_size: usize, narrow_hidden: usize, num_layers: usize, output_size: usize, seed: u64) -> Self {
        assert!(num_layers >= 8, "MLPDeep requires at least 8 layers");

        let mut layers = Vec::new();

        // First layer
        layers.push((
            xavier_init(vec![input_size, narrow_hidden], seed),
            Tensor::zeros(vec![narrow_hidden]),
        ));

        // Hidden layers
        for i in 1..num_layers - 1 {
            layers.push((
                xavier_init(vec![narrow_hidden, narrow_hidden], seed + i as u64),
                Tensor::zeros(vec![narrow_hidden]),
            ));
        }

        // Output layer
        layers.push((
            xavier_init(vec![narrow_hidden, output_size], seed + (num_layers - 1) as u64),
            Tensor::zeros(vec![output_size]),
        ));

        Self { layers }
    }
}

impl ANNBaseline for MLPDeep {
    fn name(&self) -> &str {
        "MLPDeep"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for (i, (w, b)) in self.layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        self.layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum()
    }

    fn flops_per_inference(&self) -> u64 {
        self.layers.iter()
            .map(|(w, _)| (w.shape[0] * w.shape[1] * 2) as u64)
            .sum()
    }

    fn architecture_summary(&self) -> String {
        format!("MLPDeep: {} layers (deep narrow architecture)", self.layers.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp2layer() {
        let mlp = MLP2Layer::new(10, 20, 5, 42);
        assert_eq!(mlp.name(), "MLP2Layer");

        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);

        let params = mlp.num_parameters();
        assert_eq!(params, 10*20 + 20 + 20*5 + 5);
    }

    #[test]
    fn test_mlp3layer() {
        let mlp = MLP3Layer::new(10, 20, 15, 5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);
    }

    #[test]
    fn test_mlp4layer() {
        let mlp = MLP4Layer::new(10, &[20, 15, 10], 5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);
    }

    #[test]
    fn test_mlp_dropout() {
        let mlp = MLPDropout::new(10, &[20, 15], 5, 0.5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);
    }

    #[test]
    fn test_mlp_batchnorm() {
        let mlp = MLPBatchNorm::new(10, &[20, 15], 5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);
    }

    #[test]
    fn test_mlp_residual() {
        let mlp = MLPResidual::new(10, &[20, 20, 20], 5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);
    }

    #[test]
    fn test_mlp_wide_single() {
        let mlp = MLPWideSingle::new(10, 1000, 5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);

        assert!(mlp.num_parameters() > 10000); // Wide layer
    }

    #[test]
    fn test_mlp_deep() {
        let mlp = MLPDeep::new(10, 16, 10, 5, 42);
        let input = Tensor::randn(vec![1, 10], 123);
        let output = mlp.forward(&input);
        assert_eq!(output.shape, vec![1, 5]);

        assert_eq!(mlp.layers.len(), 10);
    }
}
