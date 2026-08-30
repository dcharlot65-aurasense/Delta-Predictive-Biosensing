//! Convolutional SNN architectures for image and spectrogram processing

use super::SNNArchitecture;
use crate::{
    SNNConfig, SNNError, SNNResult, SpikeTensor,
    layers::{SpikingConv2d, SpikingLayer, SpikingLinear, SpikingSumPool2d},
};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// Convolutional Spiking Neural Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvolutionalSNN {
    /// Convolutional layers
    pub conv_layers: Vec<SpikingConv2d>,
    /// Pooling layers
    pub pool_layers: Vec<SpikingSumPool2d>,
    /// Fully connected layers
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

        // Second conv layer, taking the first conv's 16 channels.
        //
        // Pooling now shrinks height and width and leaves the channel count
        // alone, as 2-D pooling does. The previous `16 / POOL_FACTOR` here was
        // a workaround for pooling that divided the flat neuron axis instead,
        // which is no longer what it does.
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
    /// Thread the input's spatial extent through the conv/pool stack.
    ///
    /// A `SpikeTensor` is `(batch, steps, flat)` and carries no spatial shape,
    /// so each convolution and pool has to be told the `(channels, height,
    /// width)` it receives. Only the network knows how they chain, and only the
    /// first forward pass knows the input size, so it is resolved here -- the
    /// same reason `ensure_fc_head` sizes the classifier lazily.
    ///
    /// A square input is assumed, matching what the layers infer on their own;
    /// a non-square input is reported rather than silently reshaped.
    fn configure_shapes(&mut self, flat: usize) -> SNNResult<()> {
        let plane = flat / self.input_channels.max(1);
        let side = (plane as f64).sqrt().round() as usize;
        if self.input_channels * side * side != flat {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{} channels of a square image", self.input_channels),
                actual: format!("{flat} values per time step"),
            });
        }

        let (mut h, mut w) = (side, side);
        for i in 0..self.conv_layers.len() {
            self.conv_layers[i].set_input_shape(h, w);
            let (oh, ow) = self.conv_layers[i].output_size(h, w);
            let c = self.conv_layers[i].kernel.shape()[0];
            h = oh;
            w = ow;

            if i < self.pool_layers.len() {
                self.pool_layers[i].set_input_shape(c, h, w);
                let (ph, pw) = self.pool_layers[i].output_size(h, w);
                h = ph;
                w = pw;
            }
        }
        Ok(())
    }

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

        // Tell every spatial layer what it is receiving, which only the input
        // reveals.
        self.configure_shapes(current.num_neurons())?;

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
    pub conv_blocks: Vec<Vec<SpikingConv2d>>,
    /// Pooling layers
    pub pool_layers: Vec<SpikingSumPool2d>,
    /// Classifier
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

impl SpikingVGG {
    /// Thread the input's spatial extent through the conv/pool stack.
    ///
    /// See [`ConvolutionalSNN::configure_shapes`] for why this cannot be done
    /// in the constructor: a `SpikeTensor` carries no spatial shape, so the
    /// input size is not known until the first forward pass.
    fn configure_shapes(&mut self, flat: usize) -> SNNResult<usize> {
        let in_channels = self
            .conv_blocks
            .first()
            .and_then(|b| b.first())
            .map(|c| c.kernel.shape()[1])
            .unwrap_or(1);

        let plane = flat / in_channels.max(1);
        let side = (plane as f64).sqrt().round() as usize;
        if in_channels * side * side != flat {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{in_channels} channels of a square image"),
                actual: format!("{flat} values per time step"),
            });
        }

        let (mut c, mut h, mut w) = (in_channels, side, side);
        for i in 0..self.conv_blocks.len() {
            for j in 0..self.conv_blocks[i].len() {
                self.conv_blocks[i][j].set_input_shape(h, w);
                let (oh, ow) = self.conv_blocks[i][j].output_size(h, w);
                c = self.conv_blocks[i][j].kernel.shape()[0];
                h = oh;
                w = ow;
            }
            if i < self.pool_layers.len() {
                self.pool_layers[i].set_input_shape(c, h, w);
                let (ph, pw) = self.pool_layers[i].output_size(h, w);
                h = ph;
                w = pw;
            }
        }
        Ok(c * h * w)
    }

    /// Size the classifier to whatever the convolutional stack produces.
    ///
    /// The constructor hardcoded 512, which is only right for one input size.
    fn ensure_classifier(&mut self, flattened: usize) {
        if self.classifier.first().map(|l| l.weights.ncols()) == Some(flattened) {
            return;
        }
        const HIDDEN: usize = 256;
        self.classifier = vec![
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

impl SNNArchitecture for SpikingVGG {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        let mut current = input.clone();

        self.configure_shapes(current.num_neurons())?;

        // Forward through conv blocks with pooling
        for (i, block) in self.conv_blocks.iter_mut().enumerate() {
            for conv in block {
                current = conv.forward(&current)?;
            }
            if i < self.pool_layers.len() {
                current = self.pool_layers[i].forward(&current)?;
            }
        }

        self.ensure_classifier(current.num_neurons());

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
    pub initial_conv: SpikingConv2d,
    /// Residual blocks
    pub res_blocks: Vec<ResidualBlock>,
    /// Final classifier
    pub classifier: SpikingLinear,
    /// Config
    pub config: SNNConfig,
    /// Number of classes
    pub num_classes: usize,
}

/// Residual block for Spiking ResNet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResidualBlock {
    pub conv1: SpikingConv2d,
    pub conv2: SpikingConv2d,
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

    /// Thread `(channels, height, width)` through the block, returning what it
    /// produces.
    ///
    /// The shortcut is configured from the block's *input* shape, since that is
    /// what it sees, and its output must match the residual branch for the skip
    /// connection to add.
    fn configure_shapes(
        &mut self,
        c: usize,
        h: usize,
        w: usize,
    ) -> SNNResult<(usize, usize, usize)> {
        self.conv1.set_input_shape(h, w);
        let (h1, w1) = self.conv1.output_size(h, w);

        self.conv2.set_input_shape(h1, w1);
        let (h2, w2) = self.conv2.output_size(h1, w1);
        let c2 = self.conv2.kernel.shape()[0];

        if let Some(ref mut shortcut) = self.shortcut {
            shortcut.set_input_shape(h, w);
            let (sh, sw) = shortcut.output_size(h, w);
            let sc = shortcut.kernel.shape()[0];
            if (sc, sh, sw) != (c2, h2, w2) {
                return Err(SNNError::DimensionMismatch {
                    expected: format!("shortcut output {c2}x{h2}x{w2}"),
                    actual: format!("{sc}x{sh}x{sw}"),
                });
            }
        } else if (c, h, w) != (c2, h2, w2) {
            // Without a projection the identity has to already match, which is
            // the whole reason a block declares a shortcut.
            return Err(SNNError::DimensionMismatch {
                expected: format!("identity {c2}x{h2}x{w2} for a block with no shortcut"),
                actual: format!("{c}x{h}x{w}"),
            });
        }

        Ok((c2, h2, w2))
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
        if out_dense.shape() != shortcut_dense.shape() {
            // ndarray would panic on the add; a shape report is more useful
            // than a backtrace out of an arithmetic operator.
            return Err(SNNError::DimensionMismatch {
                expected: format!("shortcut shaped {:?}", out_dense.shape()),
                actual: format!("{:?}", shortcut_dense.shape()),
            });
        }
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

impl SpikingResNet {
    /// Thread the input's spatial extent through the initial conv and every
    /// residual block.
    fn configure_shapes(&mut self, flat: usize) -> SNNResult<()> {
        let in_channels = self.initial_conv.kernel.shape()[1];
        let plane = flat / in_channels.max(1);
        let side = (plane as f64).sqrt().round() as usize;
        if in_channels * side * side != flat {
            return Err(SNNError::DimensionMismatch {
                expected: format!("{in_channels} channels of a square image"),
                actual: format!("{flat} values per time step"),
            });
        }

        self.initial_conv.set_input_shape(side, side);
        let (mut h, mut w) = self.initial_conv.output_size(side, side);
        let mut c = self.initial_conv.kernel.shape()[0];

        for i in 0..self.res_blocks.len() {
            let (nc, nh, nw) = self.res_blocks[i].configure_shapes(c, h, w)?;
            c = nc;
            h = nh;
            w = nw;
        }
        Ok(())
    }

    /// Size the classifier to whatever the residual stack produces.
    fn ensure_classifier(&mut self, flattened: usize) {
        if self.classifier.weights.ncols() == flattened {
            return;
        }
        self.classifier = SpikingLinear::new(
            flattened,
            self.num_classes,
            true,
            self.config.neuron_params.clone(),
            self.config.dt,
            false,
        );
    }
}

impl SNNArchitecture for SpikingResNet {
    fn forward(&mut self, input: &SpikeTensor) -> SNNResult<SpikeTensor> {
        self.configure_shapes(input.num_neurons())?;

        let mut current = self.initial_conv.forward(input)?;

        for block in &mut self.res_blocks {
            current = block.forward(&current)?;
        }

        self.ensure_classifier(current.num_neurons());
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

    /// A forward pass must actually run end to end. The other tests here only
    /// build the network, so a broken forward would not have shown up.
    #[test]
    fn convolutional_snn_runs_a_forward_pass() {
        let mut snn = ConvolutionalSNN::new(1, 10, SNNConfig::default());
        // One channel, 16x16.
        // One channel of a 16x16 image.
        let input = SpikeTensor::zeros(2, 4, 16 * 16, false);
        let out = snn.forward(&input).expect("forward failed");
        assert_eq!(out.shape().0, 2);
        assert_eq!(out.shape().2, 10, "output width should be the class count");
    }

    /// A non-square input is reported rather than silently reshaped.
    #[test]
    fn convolutional_snn_rejects_a_non_square_input() {
        let mut snn = ConvolutionalSNN::new(1, 10, SNNConfig::default());
        assert!(snn.forward(&SpikeTensor::zeros(1, 2, 30, false)).is_err());
    }

    /// VGG must run end to end too, and size its classifier to the real
    /// flattened width rather than the hardcoded 512 it was built with.
    #[test]
    fn spiking_vgg_runs_a_forward_pass() {
        let mut vgg = SpikingVGG::vgg11(10, SNNConfig::default());
        // 3 channels, 32x32 -- five pooling stages take it to 1x1.
        let input = SpikeTensor::zeros(1, 2, 3 * 32 * 32, false);
        let out = vgg.forward(&input).expect("vgg forward failed");
        assert_eq!(out.shape().2, 10);
    }

    #[test]
    fn spiking_resnet_runs_a_forward_pass() {
        let mut resnet = SpikingResNet::resnet18(10, SNNConfig::default());
        let input = SpikeTensor::zeros(1, 2, 3 * 32 * 32, false);
        let out = resnet.forward(&input).expect("resnet forward failed");
        assert_eq!(out.shape().2, 10);
    }
}
