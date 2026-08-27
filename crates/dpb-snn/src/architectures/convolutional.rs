//! Convolutional SNN architectures for image and spectrogram processing

use super::SNNArchitecture;
use crate::{
    SNNConfig, SNNResult, SpikeTensor,
    layers::{SpikingConv2d, SpikingLayer, SpikingLinear, SpikingSumPool2d},
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Convolutional Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvolutionalSNN {
    /// Convolutional layers
    #[serde(skip)]
    pub conv_layers: Vec<SpikingConv2d>,
    /// Pooling layers
    #[serde(skip)]
    pub pool_layers: Vec<SpikingSumPool2d>,
    /// Fully connected layers
    #[serde(skip)]
    pub fc_layers: Vec<SpikingLinear>,
    /// Network configuration
    pub config: SNNConfig,
    /// Input channels
    pub input_channels: usize,
    /// Number of classes
    pub num_classes: usize,
}

impl ConvolutionalSNN {
    /// Create a simple convolutional SNN
    /// Build a convolutional SNN.
    ///
    /// The fully-connected head is sized on the first forward pass, once the
    /// width of the flattened convolutional output is actually known. This
    /// constructor receives only a channel count, so it cannot compute that
    /// width itself -- it depends on the input's spatial extent after two
    /// conv/pool stages. The placeholder below is replaced by
    /// [`ensure_fc_head`](Self::ensure_fc_head) before it is ever used.
    pub fn new(input_channels: usize, num_classes: usize, config: SNNConfig) -> Self {
        // Simple architecture: Conv -> Pool -> Conv -> Pool -> FC -> FC
        let mut conv_layers = Vec::new();
        let mut pool_layers = Vec::new();
        let mut fc_layers = Vec::new();

        // First conv layer: input_channels -> 16
        conv_layers.push(SpikingConv2d::new(
            input_channels,
            16,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        ));
        pool_layers.push(SpikingSumPool2d::new((2, 2), (2, 2)));

        // Second conv layer.
        //
        // Its input width is the FIRST conv's 16 outputs after the pooling
        // stage, which divides the neuron axis by the pool factor -- so 4, not
        // 16. `SpikingConv2d` treats that axis as channels and
        // `SpikingSumPool2d` shrinks it, and the two were declared as though
        // the pool were not there: conv2 asked for 16 inputs and received 4,
        // so `forward` failed on a shape mismatch for EVERY input, whatever
        // its size.
        const POOL_FACTOR: usize = 2 * 2;
        conv_layers.push(SpikingConv2d::new(
            16 / POOL_FACTOR,
            32,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        ));
        pool_layers.push(SpikingSumPool2d::new((2, 2), (2, 2)));

        // Placeholder head. The real input width is not knowable here -- it
        // depends on the input's spatial extent after the conv/pool stages --
        // so `ensure_fc_head` rebuilds this on the first forward pass. It was
        // previously left at this fixed 128, which made `forward` fail with a
        // matrix-shape mismatch for any input whose flattened width differed.
        fc_layers.push(SpikingLinear::new(
            128, // placeholder; resized on first forward
            64,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        ));

        fc_layers.push(SpikingLinear::new(
            64,
            num_classes,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        ));

        Self {
            conv_layers,
            pool_layers,
            fc_layers,
            config,
            input_channels,
            num_classes,
        }
    }

    /// Rebuild the fully-connected head for a given flattened input width.
    ///
    /// Called from `forward` once the convolutional stack has run and the width
    /// is known. A no-op when the head is already the right size, so it costs
    /// one comparison per pass after the first.
    fn ensure_fc_head(&mut self, flattened: usize) {
        let current_input = self.fc_layers.first().map(|l| l.weights.ncols());
        if current_input == Some(flattened) {
            return;
        }

        const HIDDEN: usize = 64;
        self.fc_layers = vec![
            SpikingLinear::new(
                flattened,
                HIDDEN,
                true,
                self.config.neuron_params.clone(),
                self.config.dt,
                false,
            ),
            SpikingLinear::new(
                HIDDEN,
                self.num_classes,
                true,
                self.config.neuron_params.clone(),
                self.config.dt,
                false,
            ),
        ];
    }
}

impl SNNArchitecture for ConvolutionalSNN {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let mut current = input.clone();

        // Forward through conv and pool layers
        for i in 0..self.conv_layers.len() {
            current = self.conv_layers[i].forward(&current)?;
            if i < self.pool_layers.len() {
                current = self.pool_layers[i].forward(&current)?;
            }
        }

        // Size the head to whatever the convolutional stack actually produced.
        self.ensure_fc_head(current.num_neurons());

        // Forward through FC layers
        for layer in &mut self.fc_layers {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    fn reset(&mut self) {
        for layer in &mut self.conv_layers {
            layer.reset_state();
        }
        for layer in &mut self.fc_layers {
            layer.reset_state();
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        self.fc_layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        self.fc_layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn zero_grad(&mut self) {
        for layer in &mut self.conv_layers {
            layer.zero_grad();
        }
        for layer in &mut self.fc_layers {
            layer.zero_grad();
        }
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// VGG-like Spiking CNN
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingVGG {
    /// Network blocks
    #[serde(skip)]
    pub conv_blocks: Vec<Vec<SpikingConv2d>>,
    /// Pooling layers
    #[serde(skip)]
    pub pool_layers: Vec<SpikingSumPool2d>,
    /// Classifier
    #[serde(skip)]
    pub classifier: Vec<SpikingLinear>,
    /// Config
    pub config: SNNConfig,
    /// Number of classes
    pub num_classes: usize,
}

impl SpikingVGG {
    /// Create VGG11-like architecture
    pub fn vgg11(num_classes: usize, config: SNNConfig) -> Self {
        let conv_configs = vec![
            (3, 64, 1),    // Block 1: 3->64, 1 layer
            (64, 128, 1),  // Block 2: 64->128, 1 layer
            (128, 256, 2), // Block 3: 128->256, 2 layers
            (256, 512, 2), // Block 4: 256->512, 2 layers
            (512, 512, 2), // Block 5: 512->512, 2 layers
        ];

        let mut conv_blocks = Vec::new();
        let mut pool_layers = Vec::new();

        for (in_ch, out_ch, num_layers) in conv_configs {
            let mut block = Vec::new();
            let mut ch_in = in_ch;

            for _ in 0..num_layers {
                block.push(SpikingConv2d::new(
                    ch_in,
                    out_ch,
                    (3, 3),
                    (1, 1),
                    (1, 1),
                    true,
                    config.neuron_params.clone(),
                    config.dt,
                    false,
                ));
                ch_in = out_ch;
            }

            conv_blocks.push(block);
            pool_layers.push(SpikingSumPool2d::new((2, 2), (2, 2)));
        }

        // Classifier
        let classifier = vec![
            SpikingLinear::new(
                512, // Flattened feature size
                256,
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ),
            SpikingLinear::new(
                256,
                num_classes,
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ),
        ];

        Self {
            conv_blocks,
            pool_layers,
            classifier,
            config,
            num_classes,
        }
    }
}

impl SNNArchitecture for SpikingVGG {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let mut current = input.clone();

        // Forward through conv blocks with pooling
        for (i, block) in self.conv_blocks.iter_mut().enumerate() {
            for conv in block {
                current = conv.forward(&current)?;
            }
            if i < self.pool_layers.len() {
                current = self.pool_layers[i].forward(&current)?;
            }
        }

        // Forward through classifier
        for layer in &mut self.classifier {
            current = layer.forward(&current)?;
        }

        Ok(current)
    }

    fn reset(&mut self) {
        for block in &mut self.conv_blocks {
            for conv in block {
                conv.reset_state();
            }
        }
        for layer in &mut self.classifier {
            layer.reset_state();
        }
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        self.classifier
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        self.classifier
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }

    fn zero_grad(&mut self) {
        for block in &mut self.conv_blocks {
            for conv in block {
                conv.zero_grad();
            }
        }
        for layer in &mut self.classifier {
            layer.zero_grad();
        }
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

/// ResNet-like Spiking CNN with skip connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingResNet {
    /// Initial conv layer
    #[serde(skip)]
    pub initial_conv: SpikingConv2d,
    /// Residual blocks
    #[serde(skip)]
    pub res_blocks: Vec<ResidualBlock>,
    /// Final classifier
    #[serde(skip)]
    pub classifier: SpikingLinear,
    /// Config
    pub config: SNNConfig,
    /// Number of classes
    pub num_classes: usize,
}

/// Residual block for Spiking ResNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlock {
    #[serde(skip)]
    pub conv1: SpikingConv2d,
    #[serde(skip)]
    pub conv2: SpikingConv2d,
    #[serde(skip)]
    pub shortcut: Option<SpikingConv2d>,
}

impl ResidualBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        stride: (usize, usize),
        config: &SNNConfig,
    ) -> Self {
        let conv1 = SpikingConv2d::new(
            in_channels,
            out_channels,
            (3, 3),
            stride,
            (1, 1),
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let conv2 = SpikingConv2d::new(
            out_channels,
            out_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let shortcut = if in_channels != out_channels || stride != (1, 1) {
            Some(SpikingConv2d::new(
                in_channels,
                out_channels,
                (1, 1),
                stride,
                (0, 0),
                true,
                config.neuron_params.clone(),
                config.dt,
                false,
            ))
        } else {
            None
        };

        Self {
            conv1,
            conv2,
            shortcut,
        }
    }

    pub fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let identity = input.clone();

        let mut out = self.conv1.forward(input)?;
        out = self.conv2.forward(&out)?;

        // Apply shortcut if needed
        let shortcut_out = if let Some(ref mut shortcut) = self.shortcut {
            shortcut.forward(&identity)?
        } else {
            identity
        };

        // Add skip connection (element-wise addition of spike trains)
        let out_dense = out.to_dense();
        let shortcut_dense = shortcut_out.to_dense();
        let combined = out_dense + shortcut_dense;

        Ok(SpikeTensor::from_dense(combined, input.requires_grad))
    }

    pub fn reset_state(&mut self) {
        self.conv1.reset_state();
        self.conv2.reset_state();
        if let Some(ref mut shortcut) = self.shortcut {
            shortcut.reset_state();
        }
    }
}

impl SpikingResNet {
    /// Create ResNet18-like architecture
    pub fn resnet18(num_classes: usize, config: SNNConfig) -> Self {
        let initial_conv = SpikingConv2d::new(
            3,
            64,
            (7, 7),
            (2, 2),
            (3, 3),
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        let mut res_blocks = Vec::new();
        let block_configs = vec![
            (64, 64, (1, 1), 2),
            (64, 128, (2, 2), 2),
            (128, 256, (2, 2), 2),
            (256, 512, (2, 2), 2),
        ];

        for (in_ch, out_ch, stride, num_blocks) in block_configs {
            for i in 0..num_blocks {
                let s = if i == 0 { stride } else { (1, 1) };
                let in_c = if i == 0 { in_ch } else { out_ch };
                res_blocks.push(ResidualBlock::new(in_c, out_ch, s, &config));
            }
        }

        let classifier = SpikingLinear::new(
            512,
            num_classes,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        );

        Self {
            initial_conv,
            res_blocks,
            classifier,
            config,
            num_classes,
        }
    }
}

impl SNNArchitecture for SpikingResNet {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let mut current = self.initial_conv.forward(input)?;

        for block in &mut self.res_blocks {
            current = block.forward(&current)?;
        }

        current = self.classifier.forward(&current)?;

        Ok(current)
    }

    fn reset(&mut self) {
        self.initial_conv.reset_state();
        for block in &mut self.res_blocks {
            block.reset_state();
        }
        self.classifier.reset_state();
    }

    fn parameters(&self) -> Vec<&Array2<f32>> {
        self.classifier.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Array2<f32>> {
        self.classifier.parameters_mut()
    }

    fn zero_grad(&mut self) {
        self.initial_conv.zero_grad();
        for block in &mut self.res_blocks {
            block.conv1.zero_grad();
            block.conv2.zero_grad();
            if let Some(ref mut shortcut) = block.shortcut {
                shortcut.zero_grad();
            }
        }
        self.classifier.zero_grad();
    }

    fn config(&self) -> &SNNConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convolutional_snn_creation() {
        let snn = ConvolutionalSNN::new(3, 10, SNNConfig::default());
        assert_eq!(snn.input_channels, 3);
        assert_eq!(snn.num_classes, 10);
    }

    #[test]
    fn test_spiking_vgg_creation() {
        let vgg = SpikingVGG::vgg11(10, SNNConfig::default());
        assert_eq!(vgg.num_classes, 10);
        assert_eq!(vgg.conv_blocks.len(), 5);
    }

    #[test]
    fn test_spiking_resnet_creation() {
        let resnet = SpikingResNet::resnet18(10, SNNConfig::default());
        assert_eq!(resnet.num_classes, 10);
        assert_eq!(resnet.res_blocks.len(), 8); // 2+2+2+2 blocks
    }
}
