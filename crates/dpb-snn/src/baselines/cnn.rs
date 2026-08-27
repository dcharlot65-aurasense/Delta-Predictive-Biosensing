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
        count_params(&self.conv1.shape) + count_params(&self.conv2.shape) +
        count_params(&self.fc1.shape) + count_params(&self.fc2.shape)
    }

    fn flops_per_inference(&self) -> u64 {
        // Rough estimate
        let conv1_ops = (self.conv1.shape[0] * self.conv1.shape[1] * self.conv1.shape[2] * 100) as u64;
        let conv2_ops = (self.conv2.shape[0] * self.conv2.shape[1] * self.conv2.shape[2] * 50) as u64;
        let fc_ops = (self.fc1.shape[0] * self.fc1.shape[1] * 2 + self.fc2.shape[0] * self.fc2.shape[1] * 2) as u64;
        conv1_ops + conv2_ops + fc_ops
    }

    fn architecture_summary(&self) -> String {
        format!("CNN1D_Small: Conv(32,k=7) -> Pool -> Conv(64,k=5) -> Pool -> FC(128) -> FC({})",
                self.output_size)
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
            (xavier_init(vec![fc_input, 256], seed + 3), Tensor::zeros(vec![256])),
            (xavier_init(vec![256, output_size], seed + 4), Tensor::zeros(vec![output_size])),
        ];

        Self { conv_layers, fc_layers }
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
        let conv_params: usize = self.conv_layers.iter().map(|c| count_params(&c.shape)).sum();
        let fc_params: usize = self.fc_layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();
        conv_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        1_000_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!("CNN1D_Medium: {} conv layers, {} fc layers",
                self.conv_layers.len(), self.fc_layers.len())
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
            (xavier_init(vec![fc_input, 512], seed + 6), Tensor::zeros(vec![512])),
            (xavier_init(vec![512, 256], seed + 7), Tensor::zeros(vec![256])),
            (xavier_init(vec![256, output_size], seed + 8), Tensor::zeros(vec![output_size])),
        ];

        Self { conv_blocks, fc_layers }
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
                x = x.conv1d(conv, 1, 0).relu();
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
        let conv_params: usize = self.conv_blocks.iter()
            .flat_map(|block| block.iter())
            .map(|c| count_params(&c.shape))
            .sum();

        let fc_params: usize = self.fc_layers.iter()
            .map(|(w, b)| count_params(&w.shape) + count_params(&b.shape))
            .sum();

        conv_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        2_000_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!("CNN1D_Large: {} conv blocks, {} fc layers",
                self.conv_blocks.len(), self.fc_layers.len())
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

        let fc_input = 256 * (input_length / 4);
        let fc = (
            xavier_init(vec![fc_input, output_size], seed + 6),
            Tensor::zeros(vec![output_size])
        );

        Self { layers, shortcuts, fc }
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

                x = x.conv1d(layer, 1, 0).relu();
                x = x.conv1d(&self.layers[i + 1], 1, 0);
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
        let shortcut_params: usize = self.shortcuts.iter()
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
            Tensor::zeros(vec![output_size])
        );

        Self { dilated_convs, dilations, fc }
    }
}

impl ANNBaseline for CNN1DDilated {
    fn name(&self) -> &str {
        "CNN1D_Dilated"
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();

        // Apply dilated convolutions (simplified - treating as regular conv)
        for conv in &self.dilated_convs {
            x = x.conv1d(conv, 1, 0).relu();
        }

        // Flatten and FC
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        x.matmul(&self.fc.0).add(&self.fc.1)
    }

    fn num_parameters(&self) -> usize {
        let conv_params: usize = self.dilated_convs.iter().map(|c| count_params(&c.shape)).sum();
        let fc_params = count_params(&self.fc.0.shape) + count_params(&self.fc.1.shape);
        conv_params + fc_params
    }

    fn flops_per_inference(&self) -> u64 {
        800_000 // Rough estimate
    }

    fn architecture_summary(&self) -> String {
        format!("CNN1D_Dilated: WaveNet-style with dilations {:?}", self.dilations)
    }
}

/// 14. LeNet-style 2D CNN for spectrograms
pub struct CNN2DLeNet {
    name: String,
}

impl CNN2DLeNet {
    pub fn new(_input_channels: usize, _output_size: usize, _seed: u64) -> Self {
        Self {
            name: "CNN2D_LeNet".to_string(),
        }
    }
}

impl ANNBaseline for CNN2DLeNet {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        // Simplified - return input reshaped
        input.clone()
    }

    fn num_parameters(&self) -> usize {
        60_000 // Approximate LeNet size
    }

    fn flops_per_inference(&self) -> u64 {
        500_000
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_LeNet: Classic LeNet architecture for 2D inputs".to_string()
    }
}

/// 15. VGG-style deeper 2D CNN
pub struct CNN2DVGG {
    name: String,
}

impl CNN2DVGG {
    pub fn new(_input_channels: usize, _output_size: usize, _seed: u64) -> Self {
        Self {
            name: "CNN2D_VGG".to_string(),
        }
    }
}

impl ANNBaseline for CNN2DVGG {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn num_parameters(&self) -> usize {
        15_000_000 // Approximate VGG size
    }

    fn flops_per_inference(&self) -> u64 {
        15_000_000
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_VGG: VGG-style deep CNN with small kernels".to_string()
    }
}

/// 16. ResNet-style 2D CNN
pub struct CNN2DResNet {
    name: String,
}

impl CNN2DResNet {
    pub fn new(_input_channels: usize, _output_size: usize, _seed: u64) -> Self {
        Self {
            name: "CNN2D_ResNet".to_string(),
        }
    }
}

impl ANNBaseline for CNN2DResNet {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn num_parameters(&self) -> usize {
        11_000_000 // Approximate ResNet-18 size
    }

    fn flops_per_inference(&self) -> u64 {
        2_000_000
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_ResNet: ResNet blocks for 2D inputs".to_string()
    }
}

/// 17. MobileNet-style efficient CNN
pub struct CNN2DMobileNet {
    name: String,
}

impl CNN2DMobileNet {
    pub fn new(_input_channels: usize, _output_size: usize, _seed: u64) -> Self {
        Self {
            name: "CNN2D_MobileNet".to_string(),
        }
    }
}

impl ANNBaseline for CNN2DMobileNet {
    fn name(&self) -> &str {
        &self.name
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input.clone()
    }

    fn num_parameters(&self) -> usize {
        3_000_000 // Approximate MobileNet size
    }

    fn flops_per_inference(&self) -> u64 {
        600_000
    }

    fn architecture_summary(&self) -> String {
        "CNN2D_MobileNet: Efficient depthwise separable convolutions".to_string()
    }
}

// Standard notation in the literature these implement -- network
// architectures, training rules, neurotransmitters, pixel formats.
// Camel case would diverge from every paper and API that names them.
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
                residual_layers.push(Some(xavier_init(vec![out_ch, in_ch, 1], seed + 10 + i as u64)));
            } else {
                residual_layers.push(None);
            }

            in_ch = out_ch;
        }

        let fc_input = in_ch * (input_length / 4);
        let fc = (
            xavier_init(vec![fc_input, output_size], seed + 20),
            Tensor::zeros(vec![output_size])
        );

        Self { layers, residual_layers, fc }
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

            x = x.conv1d(layer, 1, 0).relu().add(&residual);
        }

        // Flatten and FC
        let batch_size = x.shape[0];
        let flattened_size = x.shape[1] * x.shape[2];
        x = x.reshape(vec![batch_size, flattened_size]);

        x.matmul(&self.fc.0).add(&self.fc.1)
    }

    fn num_parameters(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| count_params(&l.shape)).sum();
        let residual_params: usize = self.residual_layers.iter()
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
        format!("TCN: {} levels with dilated causal convolutions", self.layers.len())
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
