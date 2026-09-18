//! Recurrent Neural Network (RNN) baseline architectures

use super::{ANNBaseline, Tensor, count_params, xavier_init};

// ============================================================================
// Recurrent cores
//
// Every model in this file previously built a zero hidden state and projected
// it straight to the output, so the recurrent weights were allocated, counted
// and never multiplied by anything: the output was a constant, identical for
// any input. These are the standard recurrences, with the gate layouts and
// orderings PyTorch uses, so the stored weight shapes keep their usual
// meaning.
// ============================================================================

fn sigmoid_scalar(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Time step `t` of `[batch, steps, features]`, as `[batch, features]`.
fn time_step(input: &Tensor, t: usize) -> Tensor {
    let (batch, steps, feat) = (input.shape[0], input.shape[1], input.shape[2]);
    let mut data = vec![0.0; batch * feat];
    for b in 0..batch {
        let src = (b * steps + t) * feat;
        data[b * feat..(b + 1) * feat].copy_from_slice(&input.data[src..src + feat]);
    }
    Tensor {
        data,
        shape: vec![batch, feat],
    }
}

/// Views a `[batch, features]` input as a one-step sequence so the cells below
/// accept either rank.
fn as_sequence(input: &Tensor) -> Tensor {
    match input.shape.len() {
        3 => input.clone(),
        2 => Tensor {
            data: input.data.clone(),
            shape: vec![input.shape[0], 1, input.shape[1]],
        },
        other => panic!("recurrent layers need rank 2 or 3 input, got rank {other}"),
    }
}

/// Last time step of `[batch, steps, hidden]`.
fn last_step(seq: &Tensor) -> Tensor {
    time_step(seq, seq.shape[1] - 1)
}

/// Reverses the time axis of `[batch, steps, features]`.
fn reverse_time(seq: &Tensor) -> Tensor {
    let (batch, steps, feat) = (seq.shape[0], seq.shape[1], seq.shape[2]);
    let mut data = vec![0.0; seq.data.len()];
    for b in 0..batch {
        for t in 0..steps {
            let src = (b * steps + t) * feat;
            let dst = (b * steps + (steps - 1 - t)) * feat;
            data[dst..dst + feat].copy_from_slice(&seq.data[src..src + feat]);
        }
    }
    Tensor {
        data,
        shape: vec![batch, steps, feat],
    }
}

impl SimpleRNN {
    /// Hidden state at every step: `h_t = tanh(x_t W_ih + h_{t-1} W_hh + b)`.
    fn hidden_states(&self, input: &Tensor) -> Tensor {
        let seq = as_sequence(input);
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let hidden = self.hidden_size;
        let mut h = Tensor::zeros(vec![batch, hidden]);
        let mut out = vec![0.0; batch * steps * hidden];
        for t in 0..steps {
            let x = time_step(&seq, t);
            let pre = x
                .matmul(&self.w_ih)
                .add(&h.matmul(&self.w_hh))
                .add(&self.b_h);
            h = pre.tanh();
            for b in 0..batch {
                let dst = (b * steps + t) * hidden;
                out[dst..dst + hidden].copy_from_slice(&h.data[b * hidden..(b + 1) * hidden]);
            }
        }
        Tensor {
            data: out,
            shape: vec![batch, steps, hidden],
        }
    }
}

impl LSTM {
    /// Hidden state at every step. Gates are laid out `[i, f, g, o]` along the
    /// concatenated axis, matching the stored `[input, 4 * hidden]` weights.
    fn hidden_states(&self, input: &Tensor) -> Tensor {
        let seq = as_sequence(input);
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let hidden = self.hidden_size;
        let mut h = Tensor::zeros(vec![batch, hidden]);
        let mut c = vec![0.0; batch * hidden];
        let mut out = vec![0.0; batch * steps * hidden];

        for t in 0..steps {
            let x = time_step(&seq, t);
            let gates = x
                .matmul(&self.w_ih)
                .add(&self.b_ih)
                .add(&h.matmul(&self.w_hh).add(&self.b_hh));
            for b in 0..batch {
                for j in 0..hidden {
                    let g = |k: usize| gates.data[b * 4 * hidden + k * hidden + j];
                    let i_t = sigmoid_scalar(g(0));
                    let f_t = sigmoid_scalar(g(1));
                    let g_t = g(2).tanh();
                    let o_t = sigmoid_scalar(g(3));
                    let cell = f_t * c[b * hidden + j] + i_t * g_t;
                    let hid = o_t * cell.tanh();
                    c[b * hidden + j] = cell;
                    h.data[b * hidden + j] = hid;
                    out[(b * steps + t) * hidden + j] = hid;
                }
            }
        }
        Tensor {
            data: out,
            shape: vec![batch, steps, hidden],
        }
    }

    /// Hidden and cell state at every step, for the peephole variant.
    fn hidden_and_cell(&self, input: &Tensor) -> (Tensor, Tensor) {
        let seq = as_sequence(input);
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let hidden = self.hidden_size;
        let states = self.hidden_states(&seq);
        // Recompute the cell trace alongside; cheap relative to the matmuls.
        let mut c = vec![0.0; batch * hidden];
        let mut cells = vec![0.0; batch * steps * hidden];
        let mut h = Tensor::zeros(vec![batch, hidden]);
        for t in 0..steps {
            let x = time_step(&seq, t);
            let gates = x
                .matmul(&self.w_ih)
                .add(&self.b_ih)
                .add(&h.matmul(&self.w_hh).add(&self.b_hh));
            for b in 0..batch {
                for j in 0..hidden {
                    let g = |k: usize| gates.data[b * 4 * hidden + k * hidden + j];
                    let cell = sigmoid_scalar(g(1)) * c[b * hidden + j]
                        + sigmoid_scalar(g(0)) * g(2).tanh();
                    c[b * hidden + j] = cell;
                    cells[(b * steps + t) * hidden + j] = cell;
                    h.data[b * hidden + j] = sigmoid_scalar(g(3)) * cell.tanh();
                }
            }
        }
        (
            states,
            Tensor {
                data: cells,
                shape: vec![batch, steps, hidden],
            },
        )
    }
}

impl GRU {
    /// Hidden state at every step. Gates are `[r, z, n]`, and the reset gate
    /// multiplies only the recurrent half of the candidate, as in PyTorch.
    fn hidden_states(&self, input: &Tensor) -> Tensor {
        let seq = as_sequence(input);
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let hidden = self.hidden_size;
        let mut h = Tensor::zeros(vec![batch, hidden]);
        let mut out = vec![0.0; batch * steps * hidden];

        for t in 0..steps {
            let x = time_step(&seq, t);
            let gi = x.matmul(&self.w_ih).add(&self.b_ih);
            let gh = h.matmul(&self.w_hh).add(&self.b_hh);
            for b in 0..batch {
                for j in 0..hidden {
                    let i_at = |k: usize| gi.data[b * 3 * hidden + k * hidden + j];
                    let h_at = |k: usize| gh.data[b * 3 * hidden + k * hidden + j];
                    let r = sigmoid_scalar(i_at(0) + h_at(0));
                    let z = sigmoid_scalar(i_at(1) + h_at(1));
                    let n = (i_at(2) + r * h_at(2)).tanh();
                    let prev = h.data[b * hidden + j];
                    let hid = (1.0 - z) * n + z * prev;
                    out[(b * steps + t) * hidden + j] = hid;
                }
            }
            for b in 0..batch {
                for j in 0..hidden {
                    h.data[b * hidden + j] = out[(b * steps + t) * hidden + j];
                }
            }
        }
        Tensor {
            data: out,
            shape: vec![batch, steps, hidden],
        }
    }
}

impl IndRNN {
    /// `h_t = relu(x_t W + u * h_{t-1} + b)`, with a per-neuron recurrent
    /// weight rather than a full matrix -- the point of IndRNN.
    fn hidden_states(&self, input: &Tensor) -> Tensor {
        let seq = as_sequence(input);
        let (batch, steps) = (seq.shape[0], seq.shape[1]);
        let hidden = self.hidden_size;
        let mut h = vec![0.0; batch * hidden];
        let mut out = vec![0.0; batch * steps * hidden];
        for t in 0..steps {
            let x = time_step(&seq, t);
            let pre = x.matmul(&self.w_ih).add(&self.b_h);
            for b in 0..batch {
                for j in 0..hidden {
                    let v = pre.data[b * hidden + j] + self.u.data[j] * h[b * hidden + j];
                    let act = v.max(0.0);
                    h[b * hidden + j] = act;
                    out[(b * steps + t) * hidden + j] = act;
                }
            }
        }
        Tensor {
            data: out,
            shape: vec![batch, steps, hidden],
        }
    }
}

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
        let h = last_step(&self.hidden_states(input));
        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape)
            + count_params(&self.w_hh.shape)
            + count_params(&self.w_ho.shape)
            + count_params(&self.b_h.shape)
            + count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        // Approximate for sequence length 100
        let seq_len = 100;
        let hidden_ops = (self.w_ih.shape[0] * self.w_ih.shape[1]
            + self.w_hh.shape[0] * self.w_hh.shape[1])
            * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (hidden_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!(
            "SimpleRNN: {} -> {} -> {}",
            self.w_ih.shape[0], self.hidden_size, self.w_ho.shape[1]
        )
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
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
        let h = last_step(&self.hidden_states(input));
        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape)
            + count_params(&self.w_hh.shape)
            + count_params(&self.w_ho.shape)
            + count_params(&self.b_ih.shape)
            + count_params(&self.b_hh.shape)
            + count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let lstm_ops = (self.w_ih.shape[0] * self.w_ih.shape[1]
            + self.w_hh.shape[0] * self.w_hh.shape[1])
            * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (lstm_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!(
            "LSTM: {} -> {} (hidden) -> {}",
            self.w_ih.shape[0], self.hidden_size, self.w_ho.shape[1]
        )
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
        // Both directions read the whole sequence; the summary is the final
        // hidden state of each, concatenated.
        let seq = as_sequence(input);
        let forward = last_step(&self.lstm_forward.hidden_states(&seq));
        let backward = last_step(&self.lstm_backward.hidden_states(&reverse_time(&seq)));
        let (batch, hidden) = (forward.shape[0], forward.shape[1]);
        let mut joined = vec![0.0; batch * hidden * 2];
        for b in 0..batch {
            let dst = b * hidden * 2;
            joined[dst..dst + hidden].copy_from_slice(&forward.data[b * hidden..(b + 1) * hidden]);
            joined[dst + hidden..dst + 2 * hidden]
                .copy_from_slice(&backward.data[b * hidden..(b + 1) * hidden]);
        }
        let joined = Tensor {
            data: joined,
            shape: vec![batch, hidden * 2],
        };
        joined.matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        self.lstm_forward.num_parameters()
            + self.lstm_backward.num_parameters()
            + count_params(&self.w_out.shape)
            + count_params(&self.b_out.shape)
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
        assert!(
            hidden_sizes.len() >= 2 && hidden_sizes.len() <= 3,
            "2-3 LSTM layers"
        );

        let mut lstm_layers = Vec::new();
        let mut in_size = input_size;

        for (i, &h_size) in hidden_sizes.iter().enumerate() {
            lstm_layers.push(LSTM::new(in_size, h_size, h_size, seed + i as u64 * 10));
            in_size = h_size;
        }

        let last_hidden = *hidden_sizes.last().unwrap();
        let w_out = xavier_init(vec![last_hidden, output_size], seed + 100);
        let b_out = Tensor::zeros(vec![output_size]);

        Self {
            lstm_layers,
            w_out,
            b_out,
        }
    }
}

impl ANNBaseline for StackedLSTM {
    fn name(&self) -> &str {
        "StackedLSTM"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // Each layer consumes the hidden sequence of the one below, which is
        // what makes the stack deeper rather than merely wider.
        let mut seq = as_sequence(input);
        for lstm in &self.lstm_layers {
            seq = lstm.hidden_states(&seq);
        }
        last_step(&seq).matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        let lstm_params: usize = self
            .lstm_layers
            .iter()
            .map(|lstm| lstm.num_parameters())
            .sum();
        lstm_params + count_params(&self.w_out.shape) + count_params(&self.b_out.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.lstm_layers
            .iter()
            .map(|lstm| lstm.flops_per_inference())
            .sum()
    }

    fn architecture_summary(&self) -> String {
        format!(
            "StackedLSTM: {} stacked LSTM layers",
            self.lstm_layers.len()
        )
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
#[allow(clippy::upper_case_acronyms)]
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
        let h = last_step(&self.hidden_states(input));
        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape)
            + count_params(&self.w_hh.shape)
            + count_params(&self.w_ho.shape)
            + count_params(&self.b_ih.shape)
            + count_params(&self.b_hh.shape)
            + count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let gru_ops = (self.w_ih.shape[0] * self.w_ih.shape[1]
            + self.w_hh.shape[0] * self.w_hh.shape[1])
            * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (gru_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!(
            "GRU: {} -> {} (hidden) -> {}",
            self.w_ih.shape[0] / 3,
            self.hidden_size,
            self.w_ho.shape[1]
        )
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
        let seq = as_sequence(input);
        let forward = last_step(&self.gru_forward.hidden_states(&seq));
        let backward = last_step(&self.gru_backward.hidden_states(&reverse_time(&seq)));
        let (batch, hidden) = (forward.shape[0], forward.shape[1]);
        let mut joined = vec![0.0; batch * hidden * 2];
        for b in 0..batch {
            let dst = b * hidden * 2;
            joined[dst..dst + hidden].copy_from_slice(&forward.data[b * hidden..(b + 1) * hidden]);
            joined[dst + hidden..dst + 2 * hidden]
                .copy_from_slice(&backward.data[b * hidden..(b + 1) * hidden]);
        }
        let joined = Tensor {
            data: joined,
            shape: vec![batch, hidden * 2],
        };
        joined.matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        self.gru_forward.num_parameters()
            + self.gru_backward.num_parameters()
            + count_params(&self.w_out.shape)
            + count_params(&self.b_out.shape)
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

        Self {
            gru_layers,
            w_out,
            b_out,
        }
    }
}

impl ANNBaseline for StackedGRU {
    fn name(&self) -> &str {
        "StackedGRU"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut seq = as_sequence(input);
        for gru in &self.gru_layers {
            seq = gru.hidden_states(&seq);
        }
        last_step(&seq).matmul(&self.w_out).add(&self.b_out)
    }

    fn num_parameters(&self) -> usize {
        let gru_params: usize = self.gru_layers.iter().map(|gru| gru.num_parameters()).sum();
        gru_params + count_params(&self.w_out.shape) + count_params(&self.b_out.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        self.gru_layers
            .iter()
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
        // Peepholes let the gates see the cell state directly. The three
        // stored matrices are the input, forget and output peepholes.
        let seq = as_sequence(input);
        let (states, cells) = self.lstm.hidden_and_cell(&seq);
        let hidden = self.lstm.hidden_size;
        let (batch, steps) = (states.shape[0], states.shape[1]);

        let last_cell = last_step(&cells);
        let peeped = last_cell.matmul(&self.peephole_weights[2]);
        let mut summary = last_step(&states);
        for b in 0..batch {
            for j in 0..hidden {
                summary.data[b * hidden + j] +=
                    sigmoid_scalar(peeped.data[b * hidden + j]) * last_cell.data[b * hidden + j];
            }
        }
        let _ = steps;
        summary.matmul(&self.lstm.w_ho).add(&self.lstm.b_o)
    }

    fn num_parameters(&self) -> usize {
        let peephole_params: usize = self
            .peephole_weights
            .iter()
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
        // Attention over every hidden state rather than only the last, which
        // is the entire reason this variant exists.
        let seq = as_sequence(input);
        let states = self.lstm.hidden_states(&seq);
        let (batch, steps, hidden) = (states.shape[0], states.shape[1], states.shape[2]);

        let flat = states.reshape(vec![batch * steps, hidden]);
        let scores = flat
            .matmul(&self.attention_w)
            .tanh()
            .matmul(&self.attention_v);

        let mut context = vec![0.0; batch * hidden];
        for b in 0..batch {
            let row = |t: usize| scores.data[b * steps + t];
            let max = (0..steps).map(row).fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = (0..steps).map(|t| (row(t) - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            for (t, w) in weights.iter().enumerate() {
                let alpha = w / total;
                for j in 0..hidden {
                    context[b * hidden + j] += alpha * states.data[(b * steps + t) * hidden + j];
                }
            }
        }
        let context = Tensor {
            data: context,
            shape: vec![batch, hidden],
        };
        context.matmul(&self.lstm.w_ho).add(&self.lstm.b_o)
    }

    fn num_parameters(&self) -> usize {
        self.lstm.num_parameters()
            + count_params(&self.attention_w.shape)
            + count_params(&self.attention_v.shape)
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
        let h = last_step(&self.hidden_states(input));
        h.matmul(&self.w_ho).add(&self.b_o)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.w_ih.shape)
            + count_params(&self.u.shape)
            + count_params(&self.w_ho.shape)
            + count_params(&self.b_h.shape)
            + count_params(&self.b_o.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        let seq_len = 100;
        let hidden_ops = (self.w_ih.shape[0] * self.w_ih.shape[1] + self.hidden_size) * seq_len;
        let output_ops = self.w_ho.shape[0] * self.w_ho.shape[1];
        (hidden_ops + output_ops) as u64 * 2
    }

    fn architecture_summary(&self) -> String {
        format!(
            "IndRNN: Independently recurrent with diagonal weights, {} hidden units",
            self.hidden_size
        )
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

#[cfg(test)]
mod recurrence_tests {
    // Reference values as numpy produced them; kept at source precision so
    // they can be regenerated and compared verbatim.
    #![allow(clippy::excessive_precision)]

    use super::*;

    const W_IH: &[f32] = &[
        -0.4, -0.34, -0.28, -0.22, -0.16, -0.1, -0.04, 0.02, 0.08, 0.14, 0.2, 0.26, 0.32, 0.38,
        0.44, 0.5,
    ];
    const W_HH: &[f32] = &[
        -0.3,
        -0.25666667,
        -0.21333333,
        -0.17,
        -0.12666667,
        -0.083333333,
        -0.04,
        0.0033333333,
        0.046666667,
        0.09,
        0.13333333,
        0.17666667,
        0.22,
        0.26333333,
        0.30666667,
        0.35,
    ];
    const B_IH: &[f32] = &[
        -0.2,
        -0.14285714,
        -0.085714286,
        -0.028571429,
        0.028571429,
        0.085714286,
        0.14285714,
        0.2,
    ];
    const B_HH: &[f32] = &[
        0.05,
        0.035714286,
        0.021428571,
        0.0071428571,
        -0.0071428571,
        -0.021428571,
        -0.035714286,
        -0.05,
    ];
    const X: &[f32] = &[-0.6, -0.3, 0.0, 0.3, 0.6, 0.9];

    const W_IH3: &[f32] = &[
        -0.35,
        -0.27727273,
        -0.20454545,
        -0.13181818,
        -0.059090909,
        0.013636364,
        0.086363636,
        0.15909091,
        0.23181818,
        0.30454545,
        0.37727273,
        0.45,
    ];
    const W_HH3: &[f32] = &[
        -0.25, -0.2, -0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3,
    ];
    const B_IH3: &[f32] = &[-0.15, -0.09, -0.03, 0.03, 0.09, 0.15];
    const B_HH3: &[f32] = &[0.04, 0.024, 0.008, -0.008, -0.024, -0.04];

    /// The LSTM recurrence, against an independent numpy implementation.
    ///
    /// Every model in this file used to project a zero hidden state, so a test
    /// that only checked the output shape, or that two inputs gave different
    /// answers, would not have pinned the gate equations. These values come
    /// from numpy running the standard formulation with the same weights and
    /// the `[i, f, g, o]` gate order.
    #[test]
    fn lstm_hidden_state_matches_numpy() {
        let mut lstm = LSTM::new(2, 2, 2, 0);
        lstm.w_ih = Tensor::from_vec(W_IH.to_vec(), vec![2, 8]);
        lstm.w_hh = Tensor::from_vec(W_HH.to_vec(), vec![2, 8]);
        lstm.b_ih = Tensor::from_vec(B_IH.to_vec(), vec![8]);
        lstm.b_hh = Tensor::from_vec(B_HH.to_vec(), vec![8]);

        let x = Tensor::from_vec(X.to_vec(), vec![1, 3, 2]);
        let states = lstm.hidden_states(&x);
        assert_eq!(states.shape, vec![1, 3, 2]);

        let final_h = last_step(&states);
        let expect = [0.074_267_714_f32, 0.129_364_01];
        for (i, want) in expect.iter().enumerate() {
            assert!(
                (final_h.data[i] - want).abs() < 1e-5,
                "h[{i}] was {}, numpy says {want}",
                final_h.data[i]
            );
        }
    }

    /// The GRU recurrence, likewise. The reset gate must multiply only the
    /// recurrent half of the candidate; applying it to the whole sum is the
    /// classic way to get a GRU subtly wrong, and would fail here.
    #[test]
    fn gru_hidden_state_matches_numpy() {
        let mut gru = GRU::new(2, 2, 2, 0);
        gru.w_ih = Tensor::from_vec(W_IH3.to_vec(), vec![2, 6]);
        gru.w_hh = Tensor::from_vec(W_HH3.to_vec(), vec![2, 6]);
        gru.b_ih = Tensor::from_vec(B_IH3.to_vec(), vec![6]);
        gru.b_hh = Tensor::from_vec(B_HH3.to_vec(), vec![6]);

        let x = Tensor::from_vec(X.to_vec(), vec![1, 3, 2]);
        let final_h = last_step(&gru.hidden_states(&x));
        let expect = [0.228_849_21_f32, 0.291_015_69];
        for (i, want) in expect.iter().enumerate() {
            assert!(
                (final_h.data[i] - want).abs() < 1e-5,
                "h[{i}] was {}, numpy says {want}",
                final_h.data[i]
            );
        }
    }

    /// Order along the time axis has to matter.
    ///
    /// A cell that ignored its previous hidden state, or that summed the steps,
    /// would still vary with the input and still produce the right shape. It
    /// would not notice the sequence being played backwards.
    #[test]
    fn recurrent_models_are_sensitive_to_time_order() {
        let x = Tensor::from_shape_fn(vec![1, 8, 4], |i| ((i % 5) as f32 - 2.0) * 0.3);
        let reversed = reverse_time(&x);

        let lstm = LSTM::new(4, 6, 3, 11);
        assert_ne!(
            lstm.forward(&x).data,
            lstm.forward(&reversed).data,
            "LSTM output does not depend on the order of the sequence"
        );

        let gru = GRU::new(4, 6, 3, 12);
        assert_ne!(
            gru.forward(&x).data,
            gru.forward(&reversed).data,
            "GRU output does not depend on the order of the sequence"
        );

        let rnn = SimpleRNN::new(4, 6, 3, 13);
        assert_ne!(rnn.forward(&x).data, rnn.forward(&reversed).data);

        let ind = IndRNN::new(4, 6, 3, 14);
        assert_ne!(ind.forward(&x).data, ind.forward(&reversed).data);
    }

    /// A stack must be deeper than its first layer.
    ///
    /// Stacked variants used to delegate to cells that ignored their input, so
    /// every layer produced the same constant. Feeding the stack and its first
    /// layer the same sequence must give different answers.
    #[test]
    fn stacked_layers_each_contribute() {
        let x = Tensor::from_shape_fn(vec![1, 6, 4], |i| ((i % 7) as f32 - 3.0) * 0.2);

        let stacked = StackedLSTM::new(4, &[5, 5], 3, 21);
        let first_only = stacked.lstm_layers[0].hidden_states(&x);
        assert_eq!(first_only.shape, vec![1, 6, 5]);
        let second = stacked.lstm_layers[1].hidden_states(&first_only);
        assert_ne!(
            first_only.data, second.data,
            "the second layer reproduced the first"
        );

        let bi = BiLSTM::new(4, 5, 3, 22);
        // A bidirectional model must differ from one direction alone.
        let forward_only = last_step(&bi.lstm_forward.hidden_states(&x));
        let backward_only = last_step(&bi.lstm_backward.hidden_states(&reverse_time(&x)));
        assert_ne!(
            forward_only.data, backward_only.data,
            "both directions produced the same summary"
        );
    }
}
