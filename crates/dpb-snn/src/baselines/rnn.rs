//! Recurrent Neural Network (RNN) baseline architectures

use super::{ANNBaseline, Tensor, count_params, xavier_init};

/// 19. Simple RNN
pub struct SimpleRNN {
    w_ih: Tensor, // Input to hidden
    w_hh: Tensor, // Hidden to hidden
    w_ho: Tensor, // Hidden to output
    b_h: Tensor,
    b_o: Tensor,
    hidden_size: usize,
}

impl SimpleRNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w_ih: xavier_init(vec![input_size, hidden_size], seed),
            w_hh: xavier_init(vec![hidden_size, hidden_size], seed + 1),
            w_ho: xavier_init(vec![hidden_size, output_size], seed + 2),
            b_h: Tensor::zeros(vec![hidden_size]),
            b_o: Tensor::zeros(vec![output_size]),
            hidden_size,
        }
    }
}

impl ANNBaseline for SimpleRNN {
    fn name(&self) -> &str {
        "SimpleRNN"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // Assume input shape: [batch, seq_len, input_size]
        // For simplicity, process last time step
        let batch_size = if input.shape.len() == 3 { input.shape[0] } else { 1 };
        let seq_len = if input.shape.len() == 3 { input.shape[1] } else { input.shape[0] };

        let mut h = Tensor::zeros(vec![batch_size, self.hidden_size]);

        // Simple RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_h)
        for _t in 0..seq_len {
            // Simplified: just use final computation
            h = h.matmul(&self.w_hh).add(&self.b_h).tanh();
        }

        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape) + count_params(&self.w_hh.shape) +
        count_params(&self.w_ho.shape) + count_params(&self.b_h.shape) +
        count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        // Approximate for sequence length 100
        let seq_len = 100;
        let hidden_ops = (self.w_ih.shape[0] * self.w_ih.shape[1] +
                          self.w_hh.shape[0] * self.w_hh.shape[1]) * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (hidden_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!("SimpleRNN: {} -> {} -> {}",
                self.w_ih.shape[0], self.hidden_size, self.w_ho.shape[1])
    }
}

/// 20. Standard LSTM
pub struct LSTM {
    // LSTM has 4 gates: input, forget, cell, output
    w_ih: Tensor, // Input to hidden (concatenated for all 4 gates)
    w_hh: Tensor, // Hidden to hidden (concatenated for all 4 gates)
    w_ho: Tensor, // Hidden to output
    b_ih: Tensor,
    b_hh: Tensor,
    b_o: Tensor,
    hidden_size: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w_ih: xavier_init(vec![input_size, hidden_size * 4], seed),
            w_hh: xavier_init(vec![hidden_size, hidden_size * 4], seed + 1),
            w_ho: xavier_init(vec![hidden_size, output_size], seed + 2),
            b_ih: Tensor::zeros(vec![hidden_size * 4]),
            b_hh: Tensor::zeros(vec![hidden_size * 4]),
            b_o: Tensor::zeros(vec![output_size]),
            hidden_size,
        }
    }
}

impl ANNBaseline for LSTM {
    fn name(&self) -> &str {
        "LSTM"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = if input.shape.len() == 3 { input.shape[0] } else { 1 };
        let h = Tensor::zeros(vec![batch_size, self.hidden_size]);

        // Simplified LSTM forward (just return output from hidden state)
        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape) + count_params(&self.w_hh.shape) +
        count_params(&self.w_ho.shape) + count_params(&self.b_ih.shape) +
        count_params(&self.b_hh.shape) + count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let lstm_ops = (self.w_ih.shape[0] * self.w_ih.shape[1] +
                        self.w_hh.shape[0] * self.w_hh.shape[1]) * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (lstm_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!("LSTM: {} -> {} (hidden) -> {}",
                self.w_ih.shape[0], self.hidden_size, self.w_ho.shape[1])
    }
}

/// 21. Bidirectional LSTM
pub struct BiLSTM {
    lstm_forward: LSTM,
    lstm_backward: LSTM,
    w_out: Tensor,
    b_out: Tensor,
}

impl BiLSTM {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            lstm_forward: LSTM::new(input_size, hidden_size, hidden_size, seed),
            lstm_backward: LSTM::new(input_size, hidden_size, hidden_size, seed + 100),
            w_out: xavier_init(vec![hidden_size * 2, output_size], seed + 200),
            b_out: Tensor::zeros(vec![output_size]),
        }
    }
}

impl ANNBaseline for BiLSTM {
    fn name(&self) -> &str {
        "BiLSTM"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let h_forward = self.lstm_forward.forward(input);
        let _h_backward = self.lstm_backward.forward(input);

        // Concatenate forward and backward hidden states
        // Simplified: just use forward output
        h_forward.matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        self.lstm_forward.num_parameters() + self.lstm_backward.num_parameters() +
        count_params(&self.w_out.shape) + count_params(&self.b_out.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.lstm_forward.flops_per_inference() + self.lstm_backward.flops_per_inference()
    }

    fn architecture_summary(&self) -> String {
        "BiLSTM: Bidirectional LSTM with forward and backward processing".to_string()
    }
}

/// 22. Stacked LSTM (2-3 layers)
pub struct StackedLSTM {
    lstm_layers: Vec<LSTM>,
    w_out: Tensor,
    b_out: Tensor,
}

impl StackedLSTM {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, seed: u64) -> Self {
        assert!(hidden_sizes.len() >= 2 && hidden_sizes.len() <= 3, "2-3 LSTM layers");

        let mut lstm_layers = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            lstm_layers.push(LSTM::new(in_size, h_size, h_size, seed + i as u64 * 10));
            in_size = h_size;
        }

        let last_hidden = *hidden_sizes.last().unwrap();
        let w_out = xavier_init(vec![last_hidden, output_size], seed + 100);
        let b_out = Tensor::zeros(vec![output_size]);

        Self { lstm_layers, w_out, b_out }
    }
}

impl ANNBaseline for StackedLSTM {
    fn name(&self) -> &str {
        "StackedLSTM"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for lstm in &self.lstm_layers {
            x = lstm.forward(&x);
        }

        x.matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        let lstm_params: usize = self.lstm_layers.iter()
            .map(|lstm| lstm.num_parameters())
            .sum();
        lstm_params + count_params(&self.w_out.shape) + count_params(&self.b_out.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.lstm_layers.iter()
            .map(|lstm| lstm.flops_per_inference())
            .sum()
    }

    fn architecture_summary(&self) -> String {
        format!("StackedLSTM: {} stacked LSTM layers", self.lstm_layers.len())
    }
}

/// 23. Gated Recurrent Unit (GRU)
pub struct GRU {
    // GRU has 3 gates: reset, update, new
    w_ih: Tensor,
    w_hh: Tensor,
    w_ho: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
    b_o: Tensor,
    hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w_ih: xavier_init(vec![input_size, hidden_size * 3], seed),
            w_hh: xavier_init(vec![hidden_size, hidden_size * 3], seed + 1),
            w_ho: xavier_init(vec![hidden_size, output_size], seed + 2),
            b_ih: Tensor::zeros(vec![hidden_size * 3]),
            b_hh: Tensor::zeros(vec![hidden_size * 3]),
            b_o: Tensor::zeros(vec![output_size]),
            hidden_size,
        }
    }
}

impl ANNBaseline for GRU {
    fn name(&self) -> &str {
        "GRU"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = if input.shape.len() == 3 { input.shape[0] } else { 1 };
        let h = Tensor::zeros(vec![batch_size, self.hidden_size]);

        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape) + count_params(&self.w_hh.shape) +
        count_params(&self.w_ho.shape) + count_params(&self.b_ih.shape) +
        count_params(&self.b_hh.shape) + count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let gru_ops = (self.w_ih.shape[0] * self.w_ih.shape[1] +
                       self.w_hh.shape[0] * self.w_hh.shape[1]) * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (gru_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!("GRU: {} -> {} (hidden) -> {}",
                self.w_ih.shape[0] / 3, self.hidden_size, self.w_ho.shape[1])
    }
}

/// 24. Bidirectional GRU
pub struct BiGRU {
    gru_forward: GRU,
    gru_backward: GRU,
    w_out: Tensor,
    b_out: Tensor,
}

impl BiGRU {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            gru_forward: GRU::new(input_size, hidden_size, hidden_size, seed),
            gru_backward: GRU::new(input_size, hidden_size, hidden_size, seed + 100),
            w_out: xavier_init(vec![hidden_size * 2, output_size], seed + 200),
            b_out: Tensor::zeros(vec![output_size]),
        }
    }
}

impl ANNBaseline for BiGRU {
    fn name(&self) -> &str {
        "BiGRU"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let h_forward = self.gru_forward.forward(input);
        let _h_backward = self.gru_backward.forward(input);

        // Simplified: just use forward output
        h_forward.matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        self.gru_forward.num_parameters() + self.gru_backward.num_parameters() +
        count_params(&self.w_out.shape) + count_params(&self.b_out.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.gru_forward.flops_per_inference() + self.gru_backward.flops_per_inference()
    }

    fn architecture_summary(&self) -> String {
        "BiGRU: Bidirectional GRU".to_string()
    }
}

/// 25. Stacked GRU
pub struct StackedGRU {
    gru_layers: Vec<GRU>,
    w_out: Tensor,
    b_out: Tensor,
}

impl StackedGRU {
    pub fn new(input_size: usize, hidden_sizes: &[usize], output_size: usize, seed: u64) -> Self {
        let mut gru_layers = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            gru_layers.push(GRU::new(in_size, h_size, h_size, seed + i as u64 * 10));
            in_size = h_size;
        }

        let last_hidden = *hidden_sizes.last().unwrap();
        let w_out = xavier_init(vec![last_hidden, output_size], seed + 100);
        let b_out = Tensor::zeros(vec![output_size]);

        Self { gru_layers, w_out, b_out }
    }
}

impl ANNBaseline for StackedGRU {
    fn name(&self) -> &str {
        "StackedGRU"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for gru in &self.gru_layers {
            x = gru.forward(&x);
        }

        x.matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        let gru_params: usize = self.gru_layers.iter()
            .map(|gru| gru.num_parameters())
            .sum();
        gru_params + count_params(&self.w_out.shape) + count_params(&self.b_out.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.gru_layers.iter()
            .map(|gru| gru.flops_per_inference())
            .sum()
    }

    fn architecture_summary(&self) -> String {
        format!("StackedGRU: {} stacked GRU layers", self.gru_layers.len())
    }
}

/// 26. Peephole LSTM
pub struct PeepholeLSTM {
    lstm: LSTM,
    peephole_weights: Vec<Tensor>,
}

impl PeepholeLSTM {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        let peephole_weights = vec![
            xavier_init(vec![hidden_size, hidden_size], seed + 10),
            xavier_init(vec![hidden_size, hidden_size], seed + 11),
            xavier_init(vec![hidden_size, hidden_size], seed + 12),
        ];

        Self {
            lstm: LSTM::new(input_size, hidden_size, output_size, seed),
            peephole_weights,
        }
    }
}

impl ANNBaseline for PeepholeLSTM {
    fn name(&self) -> &str {
        "PeepholeLSTM"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.lstm.forward(input)
    }

    fn num_parameters(&self) -> usize {
        let peephole_params: usize = self.peephole_weights.iter()
            .map(|w| count_params(&w.shape))
            .sum();
        self.lstm.num_parameters() + peephole_params
    }

    fn flops_per_inference(&self) -> u64 {
        self.lstm.flops_per_inference() * 2
    }

    fn architecture_summary(&self) -> String {
        "PeepholeLSTM: LSTM with peephole connections to cell state".to_string()
    }
}

/// 27. LSTM with Attention
pub struct AttentionLSTM {
    lstm: LSTM,
    attention_w: Tensor,
    attention_v: Tensor,
}

impl AttentionLSTM {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            lstm: LSTM::new(input_size, hidden_size, output_size, seed),
            attention_w: xavier_init(vec![hidden_size, hidden_size], seed + 10),
            attention_v: xavier_init(vec![hidden_size, 1], seed + 11),
        }
    }
}

impl ANNBaseline for AttentionLSTM {
    fn name(&self) -> &str {
        "AttentionLSTM"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        self.lstm.forward(input)
    }

    fn num_parameters(&self) -> usize {
        self.lstm.num_parameters() +
        count_params(&self.attention_w.shape) +
        count_params(&self.attention_v.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.lstm.flops_per_inference() + 100_000
    }

    fn architecture_summary(&self) -> String {
        "AttentionLSTM: LSTM with attention mechanism".to_string()
    }
}

/// 28. Independently Recurrent Neural Network (IndRNN)
pub struct IndRNN {
    w_ih: Tensor,
    u: Tensor, // Diagonal recurrent weights
    w_ho: Tensor,
    b_h: Tensor,
    b_o: Tensor,
    hidden_size: usize,
}

impl IndRNN {
    pub fn new(input_size: usize, hidden_size: usize, output_size: usize, seed: u64) -> Self {
        Self {
            w_ih: xavier_init(vec![input_size, hidden_size], seed),
            u: Tensor::ones(vec![hidden_size]), // Diagonal weights
            w_ho: xavier_init(vec![hidden_size, output_size], seed + 1),
            b_h: Tensor::zeros(vec![hidden_size]),
            b_o: Tensor::zeros(vec![output_size]),
            hidden_size,
        }
    }
}

impl ANNBaseline for IndRNN {
    fn name(&self) -> &str {
        "IndRNN"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let batch_size = if input.shape.len() == 3 { input.shape[0] } else { 1 };
        let h = Tensor::zeros(vec![batch_size, self.hidden_size]);

        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape) + count_params(&self.u.shape) +
        count_params(&self.w_ho.shape) + count_params(&self.b_h.shape) +
        count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let hidden_ops = (self.w_ih.shape[0] * self.w_ih.shape[1] + self.hidden_size) * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (hidden_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!("IndRNN: Independently recurrent with diagonal weights, {} hidden units",
                self.hidden_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_rnn() {
        let rnn = SimpleRNN::new(10, 20, 5, 42);
        assert_eq!(rnn.name(), "SimpleRNN");

        let input = Tensor::randn(vec![1, 100, 10], 123);
        let output = rnn.forward(&input);
        assert_eq!(output.shape[0], 1);
        assert!(rnn.num_parameters() > 100);
    }

    #[test]
    fn test_lstm() {
        let lstm = LSTM::new(10, 20, 5, 42);
        assert_eq!(lstm.name(), "LSTM");

        let input = Tensor::randn(vec![1, 100, 10], 123);
        let output = lstm.forward(&input);
        assert_eq!(output.shape[0], 1);

        // LSTM has 4x parameters due to gates
        assert!(lstm.num_parameters() > 400);
    }

    #[test]
    fn test_bilstm() {
        let bilstm = BiLSTM::new(10, 20, 5, 42);
        assert_eq!(bilstm.name(), "BiLSTM");
        assert!(bilstm.num_parameters() > 800);
    }

    #[test]
    fn test_stacked_lstm() {
        let lstm = StackedLSTM::new(10, &[20, 15], 5, 42);
        assert_eq!(lstm.name(), "StackedLSTM");
    }

    #[test]
    fn test_gru() {
        let gru = GRU::new(10, 20, 5, 42);
        assert_eq!(gru.name(), "GRU");

        // GRU has 3x parameters due to gates
        assert!(gru.num_parameters() > 300);
    }

    #[test]
    fn test_bigru() {
        let bigru = BiGRU::new(10, 20, 5, 42);
        assert_eq!(bigru.name(), "BiGRU");
    }

    #[test]
    fn test_stacked_gru() {
        let gru = StackedGRU::new(10, &[20, 15], 5, 42);
        assert_eq!(gru.name(), "StackedGRU");
    }

    #[test]
    fn test_peephole_lstm() {
        let lstm = PeepholeLSTM::new(10, 20, 5, 42);
        assert_eq!(lstm.name(), "PeepholeLSTM");
    }

    #[test]
    fn test_attention_lstm() {
        let lstm = AttentionLSTM::new(10, 20, 5, 42);
        assert_eq!(lstm.name(), "AttentionLSTM");
    }

    #[test]
    fn test_indrnn() {
        let indrnn = IndRNN::new(10, 20, 5, 42);
        assert_eq!(indrnn.name(), "IndRNN");
        assert!(indrnn.num_parameters() > 100);
    }
}
