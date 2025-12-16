//! Convolutional SNN architectures for image and spectrogram processing

use super::SNNArchitecture;
use crate::{
    layers::{SpikingConv2d, SpikingLayer, SpikingLinear, SpikingSumPool2d},
    SNNConfig, SNNError, SNNResult, SpikeTensor,
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
    pub fn new(
        input_channels: usize,
        num_classes: usize,
        config: SNNConfig,
    ) -> Self {
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

        // Second conv layer: 16 -> 32
        conv_layers.push(SpikingConv2d::new(
            16,
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

        // FC layers (sizes are placeholders, would need proper calculation)
        fc_layers.push(SpikingLinear::new(
            128, // Flattened conv output size (approximate)
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
            (3, 64, 1),   // Block 1: 3->64, 1 layer
            (64, 128, 1), // Block 2: 64->128, 1 layer
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
        let mut classifier = Vec::new();
        classifier.push(SpikingLinear::new(
            512, // Flattened feature size
            256,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        ));
        classifier.push(SpikingLinear::new(
            256,
            num_classes,
            true,
            config.neuron_params.clone(),
            config.dt,
            false,
        ));

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
