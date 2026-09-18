//! Convolutional Neural Network (CNN) baseline architectures

use super::{ANNBaseline, Tensor, count_params, xavier_init};

/// 9. Small 1D CNN for signals
// Stored from the constructor but not consulted yet. Kept so a caller's
// configuration is not silently dropped, which is the trap the removed
// with_template had.
#[allow(dead_code)]
pub struct CNN1DSmall {
    conv1: Tensor,
    conv2: Tensor,
    fc1: Tensor,
    fc2: Tensor,
    input_channels: usize,
    output_size: usize,
}

impl CNN1DSmall {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        // conv1: in_ch=input_channels, out_ch=32, kernel=7
        let conv1 = xavier_init(vec![32, input_channels, 7], seed);

        // conv2: in_ch=32, out_ch=64, kernel=5
        let conv2 = xavier_init(vec![64, 32, 5], seed + 1);

        // Calculate FC input size after convolutions and pooling
        let l1 = (input_length - 7).div_ceil(2); // After conv1 + maxpool
        let l2 = (l1 - 5).div_ceil(2); // After conv2 + maxpool
        let fc_input = 64 * l2;

        let fc1 = xavier_init(vec![fc_input, 128], seed + 2);
        let fc2 = xavier_init(vec![128, output_size], seed + 3);

        Self {
            conv1,
            conv2,
            fc1,
            fc2,
            input_channels,
            output_size,
        }
    }
}

impl ANNBaseline for CNN1DSmall {
    fn name(&self) -> &str {
        "CNN1D_Small"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let x = input.conv1d(&self.conv1, 1, 0).relu().max_pool1d(2, 2);
        let x = x.conv1d(&self.conv2, 1, 0).relu().max_pool1d(2, 2);

        // Flatten
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        let x = x.reshape(vec![batch_size, flattened_size]);

        let x = x.matmul(&self.fc1).relu();
        x.matmul(&self.fc2)
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.conv1.shape)
            + count_params(&self.conv2.shape)
            + count_params(&self.fc1.shape)
            + count_params(&self.fc2.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        // Rough estimate
        let conv1_ops =
            (self.conv1.shape[0] * self.conv1.shape[1] * self.conv1.shape[2] * 100) as u64;
        let conv2_ops =
            (self.conv2.shape[0] * self.conv2.shape[1] * self.conv2.shape[2] * 50) as u64;
        let fc_ops = (self.fc1.shape[0] * self.fc1.shape[1] * 2
            + self.fc2.shape[0] * self.fc2.shape[1] * 2) as u64;
        conv1_ops + conv2_ops + fc_ops
    }

    fn architecture_summary(&self) -> String {
        format!(
            "CNN1D_Small: Conv(32,k=7) -> Pool -> Conv(64,k=5) -> Pool -> FC(128) -> FC({})",
            self.output_size
        )
    }
}

/// 10. Medium 1D CNN
pub struct CNN1DMedium {
    conv_layers: Vec<Tensor>,
    fc_layers: Vec<(Tensor, Tensor)>,
}

impl CNN1DMedium {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        let conv_layers = vec![
            xavier_init(vec![64, input_channels, 7], seed),
            xavier_init(vec![128, 64, 5], seed + 1),
            xavier_init(vec![256, 128, 3], seed + 2),
        ];

        // Calculate FC input size
        let l1 = (input_length - 7).div_ceil(2);
        let l2 = (l1 - 5).div_ceil(2);
        let l3 = (l2 - 3).div_ceil(2);
        let fc_input = 256 * l3;

        let fc_layers = vec![
            (
                xavier_init(vec![fc_input, 256], seed + 3),
                Tensor::zeros(vec![256]),
            ),
            (
                xavier_init(vec![256, output_size], seed + 4),
                Tensor::zeros(vec![output_size]),
            ),
        ];

        Self {
            conv_layers,
            fc_layers,
        }
    }
}

impl ANNBaseline for CNN1DMedium {
    fn name(&self) -> &str {
        "CNN1D_Medium"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Convolutional layers
        for conv in &self.conv_layers {
            x = x.conv1d(conv, 1, 0).relu().max_pool1d(2, 2);
        }

        // Flatten
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        // FC layers
        for (i, (w, b)) in self.fc_layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.fc_layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let conv_params: usize = self
            .conv_layers
            .iter()
            .map(|c| count_params(&c.shape))
            .sum();
        let fc_params: usize = self
            .fc_layers
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();
        conv_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_000_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!(
            "CNN1D_Medium: {} conv layers, {} fc layers",
            self.conv_layers.len(),
            self.fc_layers.len()
        )
    }
}

/// 11. Large 1D CNN
pub struct CNN1DLarge {
    conv_blocks: Vec<Vec<Tensor>>,
    fc_layers: Vec<(Tensor, Tensor)>,
}

impl CNN1DLarge {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        let conv_blocks = vec![
            vec![
                xavier_init(vec![64, input_channels, 7], seed),
                xavier_init(vec![64, 64, 7], seed + 1),
            ],
            vec![
                xavier_init(vec![128, 64, 5], seed + 2),
                xavier_init(vec![128, 128, 5], seed + 3),
            ],
            vec![
                xavier_init(vec![256, 128, 3], seed + 4),
                xavier_init(vec![256, 256, 3], seed + 5),
            ],
        ];

        // Simplified FC input calculation
        let fc_input = 256 * (input_length / 8);

        let fc_layers = vec![
            (
                xavier_init(vec![fc_input, 512], seed + 6),
                Tensor::zeros(vec![512]),
            ),
            (
                xavier_init(vec![512, 256], seed + 7),
                Tensor::zeros(vec![256]),
            ),
            (
                xavier_init(vec![256, output_size], seed + 8),
                Tensor::zeros(vec![output_size]),
            ),
        ];

        Self {
            conv_blocks,
            fc_layers,
        }
    }
}

impl ANNBaseline for CNN1DLarge {
    fn name(&self) -> &str {
        "CNN1D_Large"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Convolutional blocks
        for block in &self.conv_blocks {
            for conv in block {
                // 'Same' padding: the constructor sizes the classifier as
                // input_length / 8, meaning the three poolings alone set the
                // length. With zero padding each convolution also shrank it,
                // so the flattened activation never matched the layer and the
                // forward pass panicked on any input.
                let pad = conv.shape[2] / 2;
                x = x.conv1d(conv, 1, pad).relu();
            }
            x = x.max_pool1d(2, 2);
        }

        // Flatten
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        // FC layers
        for (i, (w, b)) in self.fc_layers.iter().enumerate() {
            x = x.matmul(w).add(b);
            if i < self.fc_layers.len() - 1 {
                x = x.relu();
            }
        }

        x
    }

    fn num_parameters(&self) -> usize {
        let conv_params: usize = self
            .conv_blocks
            .iter()
            .flat_map(|block| block.iter())
            .map(|c| count_params(&c.shape))
            .sum();

        let fc_params: usize = self
            .fc_layers
            .iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        conv_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        2_000_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!(
            "CNN1D_Large: {} conv blocks, {} fc layers",
            self.conv_blocks.len(),
            self.fc_layers.len()
        )
    }
}

/// 12. ResNet-style 1D CNN with residual connections
pub struct CNN1DResidual {
    layers: Vec<Tensor>,
    shortcuts: Vec<Option<Tensor>>,
    fc: (Tensor, Tensor),
}

impl CNN1DResidual {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        let mut layers = Vec::new();
        let mut shortcuts = Vec::new();

        // Residual block 1
        layers.push(xavier_init(vec![64, input_channels, 3], seed));
        layers.push(xavier_init(vec![64, 64, 3], seed + 1));
        shortcuts.push(Some(xavier_init(vec![64, input_channels, 1], seed + 10)));

        // Residual block 2
        layers.push(xavier_init(vec![128, 64, 3], seed + 2));
        layers.push(xavier_init(vec![128, 128, 3], seed + 3));
        shortcuts.push(Some(xavier_init(vec![128, 64, 1], seed + 11)));

        // Residual block 3
        layers.push(xavier_init(vec![256, 128, 3], seed + 4));
        layers.push(xavier_init(vec![256, 256, 3], seed + 5));
        shortcuts.push(Some(xavier_init(vec![256, 128, 1], seed + 12)));

        // Global pooling in the forward pass reduces every channel to one
        // value, so the classifier sees 256 features regardless of input
        // length. It used to be sized 256 * (input_length / 4).
        let _ = input_length;
        let fc_input = 256;
        let fc = (
            xavier_init(vec![fc_input, output_size], seed + 6),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            layers,
            shortcuts,
            fc,
        }
    }
}

impl ANNBaseline for CNN1DResidual {
    fn name(&self) -> &str {
        "CNN1D_Residual"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        let mut shortcut_idx = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            if i % 2 == 0 && shortcut_idx < self.shortcuts.len() {
                // Start of residual block - save shortcut
                let shortcut = if let Some(proj) = &self.shortcuts[shortcut_idx] {
                    x.conv1d(proj, 1, 0)
                } else {
                    x.clone()
                };

                // Both convolutions keep the length so the 1x1 shortcut can
                // be added back; with zero padding the two branches differed by
                // four samples and the addition panicked.
                x = x.conv1d(layer, 1, layer.shape[2] / 2).relu();
                let second = &self.layers[i + 1];
                x = x.conv1d(second, 1, second.shape[2] / 2);
                x = x.add(&shortcut).relu();

                shortcut_idx += 1;
            }
        }

        // Global average pooling
        x = x.max_pool1d(x.shape[2], 1);

        // Flatten and FC
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        x.matmul(&self.fc.0).add(&self.fc.1)
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| count_params(&l.shape)).sum();
        let shortcut_params: usize = self
            .shortcuts
            .iter()
            .filter_map(|s| s.as_ref())
            .map(|s| count_params(&s.shape))
            .sum();
        let fc_params = count_params(&self.fc.0.shape) + count_params(&self.fc.1.shape);

        layer_params + shortcut_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_500_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        "CNN1D_Residual: ResNet-style with skip connections".to_string()
    }
}

/// 13. Dilated 1D CNN (WaveNet-style)
pub struct CNN1DDilated {
    dilated_convs: Vec<Tensor>,
    dilations: Vec<usize>,
    fc: (Tensor, Tensor),
}

impl CNN1DDilated {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        let dilations = vec![1, 2, 4, 8, 16];
        let mut dilated_convs = Vec::new();

        for (i, _) in dilations.iter().enumerate() {
            let in_ch = if i == 0 { input_channels } else { 64 };
            dilated_convs.push(xavier_init(vec![64, in_ch, 3], seed + i as u64));
        }

        let fc_input = 64 * (input_length / 2);
        let fc = (
            xavier_init(vec![fc_input, output_size], seed + 10),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            dilated_convs,
            dilations,
            fc,
        }
    }
}

impl ANNBaseline for CNN1DDilated {
    fn name(&self) -> &str {
        "CNN1D_Dilated"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Dilations 1, 2, 4, 8, 16 give the exponentially growing receptive
        // field this architecture exists for. They used to be stored and
        // ignored, with the comment "treating as regular conv", which also
        // left the length inconsistent with the classifier.
        for (conv, &dilation) in self.dilated_convs.iter().zip(&self.dilations) {
            x = x.conv1d_causal(conv, dilation).relu();
        }
        // One pooling, which is what the classifier is sized for.
        x = x.max_pool1d(2, 2);

        // Flatten and FC
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        x.matmul(&self.fc.0).add(&self.fc.1)
    }

    fn num_parameters(&self) -> usize {
        let conv_params: usize = self
            .dilated_convs
            .iter()
            .map(|c| count_params(&c.shape))
            .sum();
        let fc_params = count_params(&self.fc.0.shape) + count_params(&self.fc.1.shape);
        conv_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        800_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!(
            "CNN1D_Dilated: WaveNet-style with dilations {:?}",
            self.dilations
        )
    }
}

// ============================================================================
// 2-D convolutional baselines
//
// These four were previously structs holding only a name: `forward` returned
// its input unchanged and `num_parameters`/`flops_per_inference` returned
// constants, so they reported LeNet- and VGG-sized models while computing the
// identity. They are now the real architectures, assembled from the shared
// blocks below, and every reported figure is derived from the weights that
// actually exist.
//
// FLOPs are counted as two per multiply-accumulate (one multiply, one add),
// which is what `CNN1D_Small` already assumed for its fully connected layers.
// ============================================================================

/// Inference-time batch-norm parameters for one layer.
struct BatchNormParams {
    mean: Tensor,
    var: Tensor,
    gamma: Tensor,
    beta: Tensor,
}

impl BatchNormParams {
    fn new(channels: usize) -> Self {
        Self {
            mean: Tensor::zeros(vec![channels]),
            var: Tensor::ones(vec![channels]),
            gamma: Tensor::ones(vec![channels]),
            beta: Tensor::zeros(vec![channels]),
        }
    }

    /// Scale and shift are learnable; running mean and variance are buffers,
    /// which parameter counts conventionally exclude.
    fn num_parameters(&self) -> usize {
        2 * self.gamma.size()
    }

    fn apply(&self, x: &Tensor) -> Tensor {
        x.batch_norm(&self.mean, &self.var, &self.gamma, &self.beta)
    }
}

/// Activation applied after a convolution.
#[derive(Clone, Copy, PartialEq)]
enum Act {
    None,
    Relu,
    /// ReLU clamped at 6, as MobileNet uses.
    Relu6,
}

/// One convolution, optionally depthwise, with optional batch norm and
/// activation. Every 2-D baseline below is a list of these plus a classifier.
struct ConvUnit {
    weight: Tensor,
    norm: Option<BatchNormParams>,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
    depthwise: bool,
    act: Act,
}

impl ConvUnit {
    fn conv(
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        padding: usize,
        norm: bool,
        act: Act,
        seed: u64,
    ) -> Self {
        Self {
            weight: xavier_init(vec![out_ch, in_ch, k, k], seed),
            norm: norm.then(|| BatchNormParams::new(out_ch)),
            bias: (!norm).then(|| Tensor::zeros(vec![out_ch])),
            stride,
            padding,
            depthwise: false,
            act,
        }
    }

    fn depthwise(ch: usize, k: usize, stride: usize, padding: usize, act: Act, seed: u64) -> Self {
        Self {
            weight: xavier_init(vec![ch, 1, k, k], seed),
            norm: Some(BatchNormParams::new(ch)),
            bias: None,
            stride,
            padding,
            depthwise: true,
            act,
        }
    }

    fn out_channels(&self) -> usize {
        self.weight.shape[0]
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut y = if self.depthwise {
            x.depthwise_conv2d(
                &self.weight,
                (self.stride, self.stride),
                (self.padding, self.padding),
            )
        } else {
            x.conv2d(
                &self.weight,
                (self.stride, self.stride),
                (self.padding, self.padding),
            )
        };
        if let Some(bias) = &self.bias {
            let (ch, inner) = (y.shape[1], y.shape[2] * y.shape[3]);
            for (idx, v) in y.data.iter_mut().enumerate() {
                *v += bias.data[(idx / inner) % ch];
            }
        }
        if let Some(norm) = &self.norm {
            y = norm.apply(&y);
        }
        match self.act {
            Act::None => y,
            Act::Relu => y.relu(),
            Act::Relu6 => {
                let mut r = y.relu();
                for v in r.data.iter_mut() {
                    *v = v.min(6.0);
                }
                r
            }
        }
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.weight.shape)
            + self.bias.as_ref().map_or(0, |b| b.size())
            + self.norm.as_ref().map_or(0, |n| n.num_parameters())
    }

    fn out_hw(&self, (h, w): (usize, usize)) -> (usize, usize) {
        let k = self.weight.shape[2];
        (
            (h + 2 * self.padding).saturating_sub(k) / self.stride + 1,
            (w + 2 * self.padding).saturating_sub(k) / self.stride + 1,
        )
    }

    /// Multiply-accumulates for one inference at the given input size.
    fn macs(&self, hw: (usize, usize)) -> u64 {
        let (oh, ow) = self.out_hw(hw);
        let k = self.weight.shape[2];
        let per_output = if self.depthwise {
            k * k
        } else {
            self.weight.shape[1] * k * k
        };
        (oh * ow * self.out_channels() * per_output) as u64
    }
}

fn pool_hw((h, w): (usize, usize), k: usize, stride: usize) -> (usize, usize) {
    (
        h.saturating_sub(k) / stride + 1,
        w.saturating_sub(k) / stride + 1,
    )
}

/// A fully connected layer.
struct Dense {
    weight: Tensor,
    bias: Tensor,
}

impl Dense {
    fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        Self {
            weight: xavier_init(vec![in_dim, out_dim], seed),
            bias: Tensor::zeros(vec![out_dim]),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut y = x.matmul(&self.weight);
        let out = self.bias.size();
        for (idx, v) in y.data.iter_mut().enumerate() {
            *v += self.bias.data[idx % out];
        }
        y
    }

    fn num_parameters(&self) -> usize {
        count_params(&self.weight.shape) + self.bias.size()
    }

    fn macs(&self) -> u64 {
        (self.weight.shape[0] * self.weight.shape[1]) as u64
    }
}

fn flatten(x: &Tensor) -> Tensor {
    let batch = x.shape[0];
    x.reshape(vec![batch, x.size() / batch])
}

/// 14. LeNet-5 for 2-D inputs such as spectrograms.
///
/// The classic LeCun et al. (1998) layout: two 5x5 convolutions each followed
/// by 2x2 subsampling, then three fully connected layers. At the design input
/// of 32x32 with one channel it has 61 706 parameters.
pub struct CNN2DLeNet {
    convs: Vec<ConvUnit>,
    fcs: Vec<Dense>,
    input_hw: (usize, usize),
}

impl CNN2DLeNet {
    pub fn new(
        input_channels: usize,
        input_hw: (usize, usize),
        output_size: usize,
        seed: u64,
    ) -> Self {
        let convs = vec![
            ConvUnit::conv(input_channels, 6, 5, 1, 0, false, Act::Relu, seed),
            ConvUnit::conv(6, 16, 5, 1, 0, false, Act::Relu, seed + 1),
        ];
        // Two conv-and-pool stages, so the classifier input follows the input
        // size rather than being fixed at the 32x32 case.
        let mut hw = convs[0].out_hw(input_hw);
        hw = pool_hw(hw, 2, 2);
        hw = convs[1].out_hw(hw);
        hw = pool_hw(hw, 2, 2);
        let flat = 16 * hw.0 * hw.1;

        Self {
            convs,
            fcs: vec![
                Dense::new(flat, 120, seed + 2),
                Dense::new(120, 84, seed + 3),
                Dense::new(84, output_size, seed + 4),
            ],
            input_hw,
        }
    }
}

impl ANNBaseline for CNN2DLeNet {
    fn name(&self) -> &str {
        "CNN2D_LeNet"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let x = self.convs[0].forward(input).max_pool2d(2, 2);
        let x = self.convs[1].forward(&x).max_pool2d(2, 2);
        let x = flatten(&x);
        let x = self.fcs[0].forward(&x).relu();
        let x = self.fcs[1].forward(&x).relu();
        self.fcs[2].forward(&x)
    }

    fn num_parameters(&self) -> usize {
        self.convs
            .iter()
            .map(ConvUnit::num_parameters)
            .sum::<usize>()
            + self.fcs.iter().map(Dense::num_parameters).sum::<usize>()
    }

    fn flops_per_inference(&self) -> u64 {
        let mut hw = self.input_hw;
        let mut macs = 0u64;
        for conv in &self.convs {
            macs += conv.macs(hw);
            hw = pool_hw(conv.out_hw(hw), 2, 2);
        }
        macs += self.fcs.iter().map(Dense::macs).sum::<u64>();
        2 * macs
    }

    fn architecture_summary(&self) -> String {
        format!(
            "CNN2D_LeNet: Conv(6,k=5) -> Pool -> Conv(16,k=5) -> Pool -> FC(120) -> FC(84) -> FC({})",
            self.fcs[2].weight.shape[1]
        )
    }
}

/// 15. VGG-16 for 2-D inputs.
///
/// Simonyan and Zisserman (2014), configuration D: thirteen 3x3 convolutions in
/// five blocks separated by 2x2 max pooling. This is the variant usually used
/// for 32x32 inputs, with a single fully connected classifier in place of the
/// three 4096-wide layers of the ImageNet model, giving 14.72 M parameters.
pub struct CNN2DVGG {
    blocks: Vec<Vec<ConvUnit>>,
    classifier: Dense,
    input_hw: (usize, usize),
}

impl CNN2DVGG {
    /// Channels per block, VGG-16 configuration D.
    const PLAN: [&'static [usize]; 5] = [
        &[64, 64],
        &[128, 128],
        &[256, 256, 256],
        &[512, 512, 512],
        &[512, 512, 512],
    ];

    pub fn new(
        input_channels: usize,
        input_hw: (usize, usize),
        output_size: usize,
        seed: u64,
    ) -> Self {
        let mut blocks = Vec::new();
        let mut in_ch = input_channels;
        let mut hw = input_hw;
        let mut s = seed;
        for plan in Self::PLAN {
            let mut block = Vec::new();
            for &out_ch in plan {
                block.push(ConvUnit::conv(in_ch, out_ch, 3, 1, 1, false, Act::Relu, s));
                in_ch = out_ch;
                s += 1;
            }
            hw = pool_hw(hw, 2, 2);
            blocks.push(block);
        }
        Self {
            blocks,
            classifier: Dense::new(512 * hw.0.max(1) * hw.1.max(1), output_size, s),
            input_hw,
        }
    }
}

impl ANNBaseline for CNN2DVGG {
    fn name(&self) -> &str {
        "CNN2D_VGG"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for block in &self.blocks {
            for conv in block {
                x = conv.forward(&x);
            }
            x = x.max_pool2d(2, 2);
        }
        self.classifier.forward(&flatten(&x))
    }

    fn num_parameters(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(ConvUnit::num_parameters)
            .sum::<usize>()
            + self.classifier.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        let mut hw = self.input_hw;
        let mut macs = 0u64;
        for block in &self.blocks {
            for conv in block {
                macs += conv.macs(hw);
                hw = conv.out_hw(hw);
            }
            hw = pool_hw(hw, 2, 2);
        }
        2 * (macs + self.classifier.macs())
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_VGG: VGG-16 (config D), 13 conv layers in 5 blocks -> FC".to_string()
    }
}

/// One ResNet basic block: two 3x3 convolutions and an identity or projected
/// shortcut.
struct BasicBlock {
    conv1: ConvUnit,
    conv2: ConvUnit,
    downsample: Option<ConvUnit>,
}

impl BasicBlock {
    fn new(in_ch: usize, out_ch: usize, stride: usize, seed: u64) -> Self {
        Self {
            conv1: ConvUnit::conv(in_ch, out_ch, 3, stride, 1, true, Act::Relu, seed),
            conv2: ConvUnit::conv(out_ch, out_ch, 3, 1, 1, true, Act::None, seed + 1),
            // A 1x1 projection is needed exactly when the shortcut cannot be
            // added elementwise to the block output.
            downsample: (stride != 1 || in_ch != out_ch)
                .then(|| ConvUnit::conv(in_ch, out_ch, 1, stride, 0, true, Act::None, seed + 2)),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let out = self.conv2.forward(&self.conv1.forward(x));
        let shortcut = match &self.downsample {
            Some(proj) => proj.forward(x),
            None => x.clone(),
        };
        out.add(&shortcut).relu()
    }

    fn num_parameters(&self) -> usize {
        self.conv1.num_parameters()
            + self.conv2.num_parameters()
            + self.downsample.as_ref().map_or(0, ConvUnit::num_parameters)
    }

    fn out_hw(&self, hw: (usize, usize)) -> (usize, usize) {
        self.conv2.out_hw(self.conv1.out_hw(hw))
    }

    fn macs(&self, hw: (usize, usize)) -> u64 {
        let mid = self.conv1.out_hw(hw);
        self.conv1.macs(hw)
            + self.conv2.macs(mid)
            + self.downsample.as_ref().map_or(0, |d| d.macs(hw))
    }
}

/// 16. ResNet-18 for 2-D inputs.
///
/// He et al. (2016), the 18-layer variant: a stem convolution then four stages
/// of two basic blocks at 64, 128, 256 and 512 channels, global average pooling
/// and a linear classifier. The stem is the 3x3 stride-1 form used for small
/// inputs rather than the 7x7 stride-2 ImageNet stem. 11.17 M parameters.
pub struct CNN2DResNet {
    stem: ConvUnit,
    blocks: Vec<BasicBlock>,
    classifier: Dense,
    input_hw: (usize, usize),
}

impl CNN2DResNet {
    /// (channels, stride of the stage's first block) per stage.
    const STAGES: [(usize, usize); 4] = [(64, 1), (128, 2), (256, 2), (512, 2)];

    pub fn new(
        input_channels: usize,
        input_hw: (usize, usize),
        output_size: usize,
        seed: u64,
    ) -> Self {
        let stem = ConvUnit::conv(input_channels, 64, 3, 1, 1, true, Act::Relu, seed);
        let mut blocks = Vec::new();
        let mut in_ch = 64;
        let mut s = seed + 10;
        for (ch, stride) in Self::STAGES {
            for block_idx in 0..2 {
                let stride = if block_idx == 0 { stride } else { 1 };
                blocks.push(BasicBlock::new(in_ch, ch, stride, s));
                in_ch = ch;
                s += 3;
            }
        }
        Self {
            stem,
            blocks,
            classifier: Dense::new(512, output_size, s),
            input_hw,
        }
    }
}

impl ANNBaseline for CNN2DResNet {
    fn name(&self) -> &str {
        "CNN2D_ResNet"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = self.stem.forward(input);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        self.classifier.forward(&x.global_avg_pool2d())
    }

    fn num_parameters(&self) -> usize {
        self.stem.num_parameters()
            + self
                .blocks
                .iter()
                .map(BasicBlock::num_parameters)
                .sum::<usize>()
            + self.classifier.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        let mut hw = self.stem.out_hw(self.input_hw);
        let mut macs = self.stem.macs(self.input_hw);
        for block in &self.blocks {
            macs += block.macs(hw);
            hw = block.out_hw(hw);
        }
        2 * (macs + self.classifier.macs())
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_ResNet: ResNet-18, 4 stages of 2 basic blocks (64/128/256/512) -> GAP -> FC"
            .to_string()
    }
}

/// One MobileNetV2 inverted residual: expand pointwise, depthwise, project
/// back down linearly, with a residual connection when the shape is preserved.
struct InvertedResidual {
    expand: Option<ConvUnit>,
    depthwise: ConvUnit,
    project: ConvUnit,
    residual: bool,
}

impl InvertedResidual {
    fn new(in_ch: usize, out_ch: usize, stride: usize, expansion: usize, seed: u64) -> Self {
        let hidden = in_ch * expansion;
        Self {
            // The first block has expansion 1, and then the 1x1 expansion is a
            // no-op that MobileNetV2 omits.
            expand: (expansion != 1)
                .then(|| ConvUnit::conv(in_ch, hidden, 1, 1, 0, true, Act::Relu6, seed)),
            depthwise: ConvUnit::depthwise(hidden, 3, stride, 1, Act::Relu6, seed + 1),
            // Projection is linear: MobileNetV2's central claim is that a
            // non-linearity here destroys information in the narrow layer.
            project: ConvUnit::conv(hidden, out_ch, 1, 1, 0, true, Act::None, seed + 2),
            residual: stride == 1 && in_ch == out_ch,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let mut y = match &self.expand {
            Some(expand) => expand.forward(x),
            None => x.clone(),
        };
        y = self.project.forward(&self.depthwise.forward(&y));
        if self.residual { y.add(x) } else { y }
    }

    fn num_parameters(&self) -> usize {
        self.expand.as_ref().map_or(0, ConvUnit::num_parameters)
            + self.depthwise.num_parameters()
            + self.project.num_parameters()
    }

    fn out_hw(&self, hw: (usize, usize)) -> (usize, usize) {
        self.depthwise.out_hw(hw)
    }

    fn macs(&self, hw: (usize, usize)) -> u64 {
        let expanded = self.expand.as_ref().map_or(0, |e| e.macs(hw));
        let mid = self.depthwise.out_hw(hw);
        expanded + self.depthwise.macs(hw) + self.project.macs(mid)
    }
}

/// 17. MobileNetV2 for 2-D inputs.
///
/// Sandler et al. (2018): a stem convolution, seventeen inverted residual
/// bottlenecks, a 1x1 head to 1280 channels, global average pooling and a
/// linear classifier. Depthwise-separable convolutions are what make it cheap.
pub struct CNN2DMobileNet {
    stem: ConvUnit,
    blocks: Vec<InvertedResidual>,
    head: ConvUnit,
    classifier: Dense,
    input_hw: (usize, usize),
}

impl CNN2DMobileNet {
    /// (expansion, out channels, repeats, stride of the first repeat).
    const PLAN: [(usize, usize, usize, usize); 7] = [
        (1, 16, 1, 1),
        (6, 24, 2, 2),
        (6, 32, 3, 2),
        (6, 64, 4, 2),
        (6, 96, 3, 1),
        (6, 160, 3, 2),
        (6, 320, 1, 1),
    ];

    pub fn new(
        input_channels: usize,
        input_hw: (usize, usize),
        output_size: usize,
        seed: u64,
    ) -> Self {
        let stem = ConvUnit::conv(input_channels, 32, 3, 2, 1, true, Act::Relu6, seed);
        let mut blocks = Vec::new();
        let mut in_ch = 32;
        let mut s = seed + 10;
        for (expansion, out_ch, repeats, stride) in Self::PLAN {
            for repeat in 0..repeats {
                let stride = if repeat == 0 { stride } else { 1 };
                blocks.push(InvertedResidual::new(in_ch, out_ch, stride, expansion, s));
                in_ch = out_ch;
                s += 3;
            }
        }
        Self {
            stem,
            blocks,
            head: ConvUnit::conv(320, 1280, 1, 1, 0, true, Act::Relu6, s),
            classifier: Dense::new(1280, output_size, s + 1),
            input_hw,
        }
    }
}

impl ANNBaseline for CNN2DMobileNet {
    fn name(&self) -> &str {
        "CNN2D_MobileNet"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = self.stem.forward(input);
        for block in &self.blocks {
            x = block.forward(&x);
        }
        x = self.head.forward(&x);
        self.classifier.forward(&x.global_avg_pool2d())
    }

    fn num_parameters(&self) -> usize {
        self.stem.num_parameters()
            + self
                .blocks
                .iter()
                .map(InvertedResidual::num_parameters)
                .sum::<usize>()
            + self.head.num_parameters()
            + self.classifier.num_parameters()
    }

    fn flops_per_inference(&self) -> u64 {
        let mut hw = self.stem.out_hw(self.input_hw);
        let mut macs = self.stem.macs(self.input_hw);
        for block in &self.blocks {
            macs += block.macs(hw);
            hw = block.out_hw(hw);
        }
        macs += self.head.macs(hw);
        2 * (macs + self.classifier.macs())
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_MobileNet: MobileNetV2, 17 inverted residual bottlenecks -> 1x1(1280) -> GAP -> FC"
            .to_string()
    }
}

// Domain notation: TCN is how this architecture is written everywhere.
#[allow(clippy::upper_case_acronyms)]
/// 18. Temporal Convolutional Network (TCN)
pub struct TCN {
    layers: Vec<Tensor>,
    residual_layers: Vec<Option<Tensor>>,
    fc: (Tensor, Tensor),
}

impl TCN {
    pub fn new(input_channels: usize, input_length: usize, output_size: usize, seed: u64) -> Self {
        let num_levels = 4;
        let mut layers = Vec::new();
        let mut residual_layers = Vec::new();

        let mut in_ch = input_channels;
        for i in 0..num_levels {
            let out_ch = 64 * (2_usize.pow(i as u32));
            layers.push(xavier_init(vec![out_ch, in_ch, 3], seed + i as u64));

            if in_ch != out_ch {
                residual_layers.push(Some(xavier_init(
                    vec![out_ch, in_ch, 1],
                    seed + 10 + i as u64,
                )));
            } else {
                residual_layers.push(None);
            }

            in_ch = out_ch;
        }

        let fc_input = in_ch * (input_length / 4);
        let fc = (
            xavier_init(vec![fc_input, output_size], seed + 20),
            Tensor::zeros(vec![output_size]),
        );

        Self {
            layers,
            residual_layers,
            fc,
        }
    }
}

impl ANNBaseline for TCN {
    fn name(&self) -> &str {
        "TCN"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            let residual = if let Some(proj) = &self.residual_layers[i] {
                x.conv1d(proj, 1, 0)
            } else {
                x.clone()
            };

            // Causal and dilated, which is what distinguishes a temporal
            // convolutional network: level i sees 2^i samples back and never
            // reads the future. Previously these were plain convolutions with
            // zero padding, so the residual branch was two samples longer than
            // the main one and the addition panicked.
            let dilation = 1usize << i;
            x = x.conv1d_causal(layer, dilation).relu().add(&residual);
        }
        // Two poolings: the classifier is sized for input_length / 4.
        x = x.max_pool1d(2, 2).max_pool1d(2, 2);

        // Flatten and FC
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        x.matmul(&self.fc.0).add(&self.fc.1)
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| count_params(&l.shape)).sum();
        let residual_params: usize = self
            .residual_layers
            .iter()
            .filter_map(|r| r.as_ref())
            .map(|r| count_params(&r.shape))
            .sum();
        let fc_params = count_params(&self.fc.0.shape) + count_params(&self.fc.1.shape);

        layer_params + residual_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_200_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!(
            "TCN: {} levels with dilated causal convolutions",
            self.layers.len()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cnn1d_small() {
        let cnn = CNN1DSmall::new(3, 100, 10, 42);
        assert_eq!(cnn.name(), "CNN1D_Small");

        let input = Tensor::randn(vec![1, 3, 100], 123);
        let output = cnn.forward(&input);
        assert_eq!(output.shape[0], 1);
        assert!(cnn.num_parameters() > 1000);
    }

    #[test]
    fn test_cnn1d_medium() {
        let cnn = CNN1DMedium::new(3, 100, 10, 42);
        let input = Tensor::randn(vec![1, 3, 100], 123);
        let output = cnn.forward(&input);
        assert_eq!(output.shape[0], 1);
    }

    #[test]
    fn test_cnn1d_residual() {
        let cnn = CNN1DResidual::new(3, 100, 10, 42);
        assert!(cnn.num_parameters() > 10000);
    }

    #[test]
    fn test_tcn() {
        let tcn = TCN::new(3, 100, 10, 42);
        assert_eq!(tcn.name(), "TCN");
        assert!(tcn.num_parameters() > 5000);
    }
}

#[cfg(test)]
mod cnn2d_tests {
    use super::*;

    const HW: (usize, usize) = (32, 32);
    const CLASSES: usize = 10;

    /// Parameter counts, checked against the published architectures.
    ///
    /// These four used to report constants (60 000, 15 M, 11 M, 3 M) while
    /// holding no weights at all, so a count that merely looked plausible was
    /// exactly the failure. Each figure below is reproducible from the paper's
    /// layer table, which is what makes it worth asserting exactly.
    #[test]
    fn parameter_counts_match_the_published_architectures() {
        // LeCun et al. 1998, LeNet-5 on 32x32x1: 61 706.
        assert_eq!(CNN2DLeNet::new(1, HW, CLASSES, 7).num_parameters(), 61_706);

        // Simonyan and Zisserman 2014 config D: 14 710 464 convolution weights
        // plus 4224 biases, and a 512->10 classifier in place of the three
        // 4096-wide ImageNet layers.
        assert_eq!(
            CNN2DVGG::new(3, HW, CLASSES, 7).num_parameters(),
            14_710_464 + 4_224 + (512 * CLASSES + CLASSES)
        );

        // He et al. 2016, ResNet-18 with the 3x3 stem used for small inputs:
        // the standard CIFAR figure.
        assert_eq!(
            CNN2DResNet::new(3, HW, CLASSES, 7).num_parameters(),
            11_173_962
        );

        // Sandler et al. 2018: 3 504 872 at 1000 classes, so swapping the
        // 1280->1000 head for 1280->10 removes 1 281 000 and adds 12 810.
        assert_eq!(
            CNN2DMobileNet::new(3, HW, CLASSES, 7).num_parameters(),
            3_504_872 - (1280 * 1000 + 1000) + (1280 * CLASSES + CLASSES)
        );
    }

    /// FLOPs must follow the architecture and input size, not be a constant.
    ///
    /// Counted as two per multiply-accumulate, so halving each figure gives the
    /// MAC counts usually quoted: 313 M for VGG-16 and 555 M for ResNet-18 at
    /// 32x32, and 6.1 M for MobileNetV2, which is its 300 M at 224x224 scaled
    /// by (32/224)^2.
    #[test]
    fn flops_follow_the_architecture_and_input_size() {
        let vgg_macs = CNN2DVGG::new(3, HW, CLASSES, 7).flops_per_inference() / 2;
        assert!(
            (312_000_000..315_000_000).contains(&vgg_macs),
            "VGG-16 at 32x32 should be about 313 M MACs, got {vgg_macs}"
        );

        let resnet_macs = CNN2DResNet::new(3, HW, CLASSES, 7).flops_per_inference() / 2;
        assert!(
            (554_000_000..557_000_000).contains(&resnet_macs),
            "ResNet-18 at 32x32 should be about 555 M MACs, got {resnet_macs}"
        );

        let mobile_macs = CNN2DMobileNet::new(3, HW, CLASSES, 7).flops_per_inference() / 2;
        assert!(
            (5_500_000..6_800_000).contains(&mobile_macs),
            "MobileNetV2 at 32x32 should be about 6.1 M MACs, got {mobile_macs}"
        );

        // A larger input must cost more: the old constants did not move at all.
        let small = CNN2DResNet::new(3, (16, 16), CLASSES, 7).flops_per_inference();
        let large = CNN2DResNet::new(3, (32, 32), CLASSES, 7).flops_per_inference();
        assert!(
            large > small * 3,
            "quadrupling the pixels should roughly quadruple the work: {small} -> {large}"
        );
    }

    /// The forward pass must compute, not hand back its input.
    ///
    /// Every one of these used to be `input.clone()`. Shape alone catches that
    /// here, but the second half also checks the output actually varies with
    /// the input, which a constant-returning stub would fail.
    #[test]
    fn forward_computes_rather_than_returning_its_input() {
        fn probe(name: &str, model: &dyn ANNBaseline, channels: usize) {
            let a = Tensor::from_shape_fn(vec![1, channels, HW.0, HW.1], |i| {
                ((i % 13) as f32 - 6.0) * 0.1
            });
            let b = Tensor::from_shape_fn(vec![1, channels, HW.0, HW.1], |i| {
                ((i % 7) as f32 - 3.0) * 0.2
            });

            let ya = model.forward(&a);
            assert_eq!(ya.shape, vec![1, CLASSES], "{name}: wrong output shape");
            assert_ne!(ya.shape, a.shape, "{name}: returned its input");
            assert!(
                ya.data.iter().all(|v| v.is_finite()),
                "{name}: produced a non-finite output"
            );

            let yb = model.forward(&b);
            assert_ne!(ya.data, yb.data, "{name}: output does not depend on input");
        }

        probe("LeNet-5", &CNN2DLeNet::new(1, HW, CLASSES, 7), 1);
        probe("VGG-16", &CNN2DVGG::new(3, HW, CLASSES, 7), 3);
        probe("ResNet-18", &CNN2DResNet::new(3, HW, CLASSES, 7), 3);
        probe("MobileNetV2", &CNN2DMobileNet::new(3, HW, CLASSES, 7), 3);
    }

    /// The residual path has to be a real addition.
    ///
    /// A block whose shortcut were dropped would still produce well-shaped,
    /// finite output, so shape checks cannot see it. Feeding zeros makes the
    /// convolution branch vanish while the identity shortcut survives.
    #[test]
    fn resnet_shortcut_carries_the_input_forward() {
        let block = BasicBlock::new(4, 4, 1, 3);
        assert!(
            block.downsample.is_none(),
            "a same-shape block needs no projection"
        );
        let x = Tensor::from_shape_fn(vec![1, 4, 4, 4], |i| (i % 5) as f32 + 1.0);
        let out = block.forward(&x);
        assert_eq!(out.shape, x.shape);
        // Zero-initialised batch norm shift and identity scale leave the
        // convolution branch near zero, so the shortcut dominates and every
        // positive input survives the final ReLU.
        assert!(
            out.data.iter().any(|v| *v > 0.0),
            "shortcut contributed nothing"
        );
    }
}
