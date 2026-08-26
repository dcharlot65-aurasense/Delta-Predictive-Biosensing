//! Neural Style Transfer for Skeleton Visualization
//!
//! This module applies neural style transfer to skeleton renderings,
//! transforming simple stick figures into artistic or realistic visualizations.
//!
//! # Overview
//!
//! Style transfer uses deep neural networks to:
//! - Extract content features from the skeleton rendering
//! - Extract style features from a reference image
//! - Combine them to create a stylized output
//!
//! # Available Styles
//!
//! - **Realistic**: Transfer from real human motion capture videos
//! - **Artistic**: Van Gogh, Picasso, and other art styles
//! - **Medical**: X-ray, MRI-like visualizations
//! - **Custom**: User-provided style reference images
//!
//! # Requirements
//!
//! - Python 3.8+
//! - PyTorch 1.10+
//! - torchvision
//! - Pre-trained VGG model (downloaded automatically)
//!
//! # Usage
//!
//! ```rust,ignore
//! use dpb_synth::level3::style_transfer::{StyleTransferRenderer, StyleTransferParams, StyleType};
//! use std::path::Path;
//!
//! let renderer = StyleTransferRenderer::new(StyleTransferParams::default())?;
//!
//! // Apply realistic human style to skeleton
//! let output = renderer.transfer_style(
//!     Path::new("skeleton_frame.png"),
//!     Path::new("styled_frame.png"),
//! )?;
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Result type for style transfer
pub type StyleResult<T> = Result<T, StyleTransferError>;

/// Errors that can occur during style transfer
#[derive(Debug, thiserror::Error)]
pub enum StyleTransferError {
    #[error("Python not found: {0}")]
    PythonNotFound(String),

    #[error("PyTorch not installed: {0}")]
    PyTorchNotFound(String),

    #[error("Style image not found: {0}")]
    StyleImageNotFound(PathBuf),

    #[error("Content image not found: {0}")]
    ContentImageNotFound(PathBuf),

    #[error("Script not found: {0}")]
    ScriptNotFound(PathBuf),

    #[error("Style transfer failed: {0}")]
    TransferError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Image error: {0}")]
    ImageError(String),
}

/// Pre-defined style types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum StyleType {
    /// Realistic human appearance from motion capture
    #[default]
    RealisticHuman,
    /// Van Gogh's Starry Night style
    VanGogh,
    /// Picasso's cubist style
    Picasso,
    /// Japanese ukiyo-e woodblock print style
    Ukiyoe,
    /// X-ray / radiograph style
    XRay,
    /// MRI scan style
    Mri,
    /// Thermal imaging style
    Thermal,
    /// Line drawing / sketch style
    Sketch,
    /// Watercolor painting style
    Watercolor,
    /// Neon glow effect
    NeonGlow,
    /// Silhouette style
    Silhouette,
    /// Anatomical illustration style
    Anatomical,
    /// Custom style from user-provided image
    Custom,
}


impl StyleType {
    /// Get the built-in style image filename
    pub fn style_image_name(&self) -> Option<&'static str> {
        match self {
            StyleType::RealisticHuman => Some("style_realistic_human.jpg"),
            StyleType::VanGogh => Some("style_vangogh.jpg"),
            StyleType::Picasso => Some("style_picasso.jpg"),
            StyleType::Ukiyoe => Some("style_ukiyoe.jpg"),
            StyleType::XRay => Some("style_xray.jpg"),
            StyleType::Mri => Some("style_mri.jpg"),
            StyleType::Thermal => Some("style_thermal.jpg"),
            StyleType::Sketch => Some("style_sketch.jpg"),
            StyleType::Watercolor => Some("style_watercolor.jpg"),
            StyleType::NeonGlow => Some("style_neon.jpg"),
            StyleType::Silhouette => Some("style_silhouette.jpg"),
            StyleType::Anatomical => Some("style_anatomical.jpg"),
            StyleType::Custom => None,
        }
    }

    /// Get recommended iteration count for this style
    pub fn recommended_iterations(&self) -> u32 {
        match self {
            StyleType::RealisticHuman => 500,
            StyleType::VanGogh => 300,
            StyleType::Picasso => 300,
            StyleType::Ukiyoe => 300,
            StyleType::XRay => 200,
            StyleType::Mri => 200,
            StyleType::Thermal => 150,
            StyleType::Sketch => 200,
            StyleType::Watercolor => 300,
            StyleType::NeonGlow => 200,
            StyleType::Silhouette => 100,
            StyleType::Anatomical => 400,
            StyleType::Custom => 300,
        }
    }

    /// Get recommended content weight for this style
    pub fn content_weight(&self) -> f64 {
        match self {
            StyleType::RealisticHuman => 1.0,
            StyleType::Sketch | StyleType::Silhouette => 2.0,
            StyleType::XRay | StyleType::Mri => 1.5,
            _ => 1.0,
        }
    }

    /// Get recommended style weight for this style
    pub fn style_weight(&self) -> f64 {
        match self {
            StyleType::VanGogh | StyleType::Picasso | StyleType::Ukiyoe => 1e6,
            StyleType::RealisticHuman => 1e5,
            StyleType::Sketch | StyleType::Silhouette => 5e4,
            StyleType::XRay | StyleType::Mri | StyleType::Thermal => 1e5,
            _ => 1e5,
        }
    }
}

/// Neural network architecture for style transfer
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum StyleNetwork {
    /// VGG19 (Gatys et al. original)
    #[default]
    Vgg19,
    /// VGG16 (lighter weight)
    Vgg16,
    /// Fast style transfer (Johnson et al.)
    FastStyleTransfer,
    /// AdaIN (Adaptive Instance Normalization)
    AdaIn,
}


/// Parameters for style transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTransferParams {
    /// Style type to apply
    pub style_type: StyleType,
    /// Custom style image path (for StyleType::Custom)
    pub custom_style_path: Option<PathBuf>,
    /// Neural network to use
    pub network: StyleNetwork,
    /// Number of optimization iterations
    pub iterations: u32,
    /// Content loss weight
    pub content_weight: f64,
    /// Style loss weight
    pub style_weight: f64,
    /// Total variation loss weight (smoothing)
    pub tv_weight: f64,
    /// Learning rate for optimization
    pub learning_rate: f64,
    /// Output resolution (width, height)
    pub output_resolution: Option<(u32, u32)>,
    /// Whether to preserve colors from content image
    pub preserve_color: bool,
    /// Content layers in VGG
    pub content_layers: Vec<String>,
    /// Style layers in VGG
    pub style_layers: Vec<String>,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Python executable path
    pub python_path: PathBuf,
    /// Script directory
    pub script_dir: PathBuf,
    /// Style images directory
    pub style_images_dir: PathBuf,
}

impl Default for StyleTransferParams {
    fn default() -> Self {
        Self {
            style_type: StyleType::RealisticHuman,
            custom_style_path: None,
            network: StyleNetwork::Vgg19,
            iterations: 300,
            content_weight: 1.0,
            style_weight: 1e5,
            tv_weight: 1e-6,
            learning_rate: 0.01,
            output_resolution: None,
            preserve_color: false,
            content_layers: vec!["conv4_2".to_string()],
            style_layers: vec![
                "conv1_1".to_string(),
                "conv2_1".to_string(),
                "conv3_1".to_string(),
                "conv4_1".to_string(),
                "conv5_1".to_string(),
            ],
            seed: None,
            python_path: PathBuf::from("python3"),
            script_dir: PathBuf::from("tools/level3_video"),
            style_images_dir: PathBuf::from("assets/style_images"),
        }
    }
}

impl StyleTransferParams {
    /// Create parameters for realistic human style transfer
    pub fn realistic() -> Self {
        Self {
            style_type: StyleType::RealisticHuman,
            iterations: 500,
            content_weight: 1.0,
            style_weight: 1e5,
            ..Default::default()
        }
    }

    /// Create parameters for artistic style transfer
    pub fn artistic(style: StyleType) -> Self {
        Self {
            style_type: style,
            iterations: style.recommended_iterations(),
            content_weight: style.content_weight(),
            style_weight: style.style_weight(),
            ..Default::default()
        }
    }

    /// Create parameters for medical visualization
    pub fn medical(style: StyleType) -> Self {
        Self {
            style_type: style,
            iterations: 200,
            content_weight: 1.5,
            style_weight: 1e5,
            preserve_color: false,
            ..Default::default()
        }
    }

    /// Create parameters for fast preview
    pub fn preview() -> Self {
        Self {
            iterations: 50,
            learning_rate: 0.1,
            ..Default::default()
        }
    }

    /// Create parameters with custom style image
    pub fn custom(style_image_path: PathBuf) -> Self {
        Self {
            style_type: StyleType::Custom,
            custom_style_path: Some(style_image_path),
            ..Default::default()
        }
    }
}

/// Output from style transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StyleTransferOutput {
    /// Output image path
    pub output_path: PathBuf,
    /// Final content loss
    pub content_loss: f64,
    /// Final style loss
    pub style_loss: f64,
    /// Final total loss
    pub total_loss: f64,
    /// Number of iterations run
    pub iterations: u32,
    /// Processing time in seconds
    pub processing_time: f64,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Style transfer progress callback
pub type ProgressCallback = Box<dyn Fn(u32, f64) + Send + Sync>;

/// Main style transfer renderer
pub struct StyleTransferRenderer {
    params: StyleTransferParams,
}

impl StyleTransferRenderer {
    /// Create a new style transfer renderer
    pub fn new(params: StyleTransferParams) -> StyleResult<Self> {
        Ok(Self { params })
    }

    /// Check if Python and required packages are available
    pub fn check_requirements(&self) -> StyleResult<bool> {
        let output = Command::new(&self.params.python_path)
            .args(["-c", "import torch; import torchvision; print('OK')"])
            .output()
            .map_err(|e| StyleTransferError::PythonNotFound(e.to_string()))?;

        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout.contains("OK"))
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Err(StyleTransferError::PyTorchNotFound(stderr.to_string()))
        }
    }

    /// Get PyTorch version
    pub fn pytorch_version(&self) -> StyleResult<String> {
        let output = Command::new(&self.params.python_path)
            .args(["-c", "import torch; print(torch.__version__)"])
            .output()
            .map_err(|e| StyleTransferError::PythonNotFound(e.to_string()))?;

        Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
    }

    /// Update rendering parameters
    pub fn set_params(&mut self, params: StyleTransferParams) {
        self.params = params;
    }

    /// Get the style image path for the current style type
    fn get_style_image_path(&self) -> StyleResult<PathBuf> {
        match self.params.style_type {
            StyleType::Custom => {
                self.params.custom_style_path.clone()
                    .ok_or_else(|| StyleTransferError::StyleImageNotFound(
                        PathBuf::from("No custom style path provided")
                    ))
            }
            _ => {
                let filename = self.params.style_type.style_image_name()
                    .ok_or_else(|| StyleTransferError::StyleImageNotFound(
                        PathBuf::from("Unknown style type")
                    ))?;
                Ok(self.params.style_images_dir.join(filename))
            }
        }
    }

    /// Apply style transfer to a single image
    pub fn transfer_style(
        &self,
        content_path: &Path,
        output_path: &Path,
    ) -> StyleResult<StyleTransferOutput> {
        let script_path = self.params.script_dir.join("style_transfer.py");
        if !script_path.exists() {
            return Err(StyleTransferError::ScriptNotFound(script_path));
        }

        if !content_path.exists() {
            return Err(StyleTransferError::ContentImageNotFound(content_path.to_path_buf()));
        }

        let style_path = self.get_style_image_path()?;

        // Serialize parameters
        let transfer_params = serde_json::json!({
            "content_path": content_path,
            "style_path": style_path,
            "output_path": output_path,
            "network": format!("{:?}", self.params.network).to_lowercase(),
            "iterations": self.params.iterations,
            "content_weight": self.params.content_weight,
            "style_weight": self.params.style_weight,
            "tv_weight": self.params.tv_weight,
            "learning_rate": self.params.learning_rate,
            "output_resolution": self.params.output_resolution,
            "preserve_color": self.params.preserve_color,
            "content_layers": self.params.content_layers,
            "style_layers": self.params.style_layers,
            "seed": self.params.seed,
        });

        let params_json = serde_json::to_string(&transfer_params)?;

        let output = Command::new(&self.params.python_path)
            .args([
                script_path.to_str().unwrap(),
                "--params",
                &params_json,
            ])
            .output()
            .map_err(|e| StyleTransferError::TransferError(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(StyleTransferError::TransferError(stderr.to_string()));
        }

        // Parse output
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: StyleTransferOutput = serde_json::from_str(&stdout)
            .unwrap_or_else(|_| StyleTransferOutput {
                output_path: output_path.to_path_buf(),
                content_loss: 0.0,
                style_loss: 0.0,
                total_loss: 0.0,
                iterations: self.params.iterations,
                processing_time: 0.0,
                metadata: HashMap::new(),
            });

        Ok(result)
    }

    /// Apply style transfer to a sequence of images
    pub fn transfer_sequence(
        &self,
        content_paths: &[PathBuf],
        output_dir: &Path,
        prefix: &str,
    ) -> StyleResult<Vec<StyleTransferOutput>> {
        std::fs::create_dir_all(output_dir)?;

        let mut outputs = Vec::with_capacity(content_paths.len());

        for (i, content_path) in content_paths.iter().enumerate() {
            let filename = format!("{}_{:06}.png", prefix, i);
            let output_path = output_dir.join(&filename);

            let output = self.transfer_style(content_path, &output_path)?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Apply fast style transfer using pre-trained model
    pub fn fast_transfer(
        &self,
        content_path: &Path,
        output_path: &Path,
    ) -> StyleResult<StyleTransferOutput> {
        let script_path = self.params.script_dir.join("fast_style_transfer.py");
        if !script_path.exists() {
            return Err(StyleTransferError::ScriptNotFound(script_path));
        }

        let style_model = match self.params.style_type {
            StyleType::VanGogh => "vangogh",
            StyleType::Picasso => "picasso",
            StyleType::Ukiyoe => "ukiyoe",
            StyleType::Sketch => "sketch",
            _ => "default",
        };

        let transfer_params = serde_json::json!({
            "content_path": content_path,
            "output_path": output_path,
            "style_model": style_model,
            "output_resolution": self.params.output_resolution,
        });

        let params_json = serde_json::to_string(&transfer_params)?;

        let output = Command::new(&self.params.python_path)
            .args([
                script_path.to_str().unwrap(),
                "--params",
                &params_json,
            ])
            .output()
            .map_err(|e| StyleTransferError::TransferError(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(StyleTransferError::TransferError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: StyleTransferOutput = serde_json::from_str(&stdout)
            .unwrap_or_else(|_| StyleTransferOutput {
                output_path: output_path.to_path_buf(),
                content_loss: 0.0,
                style_loss: 0.0,
                total_loss: 0.0,
                iterations: 1,  // Fast transfer is single-pass
                processing_time: 0.0,
                metadata: HashMap::new(),
            });

        Ok(result)
    }

    /// Create a blended style from multiple style images
    pub fn blend_styles(
        &self,
        content_path: &Path,
        style_paths: &[PathBuf],
        style_weights: &[f64],
        output_path: &Path,
    ) -> StyleResult<StyleTransferOutput> {
        if style_paths.len() != style_weights.len() {
            return Err(StyleTransferError::TransferError(
                "Style paths and weights must have same length".to_string()
            ));
        }

        let script_path = self.params.script_dir.join("style_transfer_blend.py");
        if !script_path.exists() {
            return Err(StyleTransferError::ScriptNotFound(script_path));
        }

        let transfer_params = serde_json::json!({
            "content_path": content_path,
            "style_paths": style_paths,
            "style_weights": style_weights,
            "output_path": output_path,
            "iterations": self.params.iterations,
            "content_weight": self.params.content_weight,
            "learning_rate": self.params.learning_rate,
        });

        let params_json = serde_json::to_string(&transfer_params)?;

        let output = Command::new(&self.params.python_path)
            .args([
                script_path.to_str().unwrap(),
                "--params",
                &params_json,
            ])
            .output()
            .map_err(|e| StyleTransferError::TransferError(e.to_string()))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(StyleTransferError::TransferError(stderr.to_string()));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: StyleTransferOutput = serde_json::from_str(&stdout)
            .unwrap_or_else(|_| StyleTransferOutput {
                output_path: output_path.to_path_buf(),
                content_loss: 0.0,
                style_loss: 0.0,
                total_loss: 0.0,
                iterations: self.params.iterations,
                processing_time: 0.0,
                metadata: HashMap::new(),
            });

        Ok(result)
    }

    /// Get the style transfer script content
    pub fn get_script_template() -> String {
        generate_style_transfer_script()
    }
}

/// Generate the Python style transfer script content
pub fn generate_style_transfer_script() -> String {
    r#"#!/usr/bin/env python3
"""
Neural Style Transfer for DPB Framework

This script applies neural style transfer using VGG19 and PyTorch.
Based on Gatys et al. "A Neural Algorithm of Artistic Style"

Requirements:
    pip install torch torchvision pillow

Usage:
    python style_transfer.py --params '{"content_path": "...", ...}'
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision.transforms as transforms
    import torchvision.models as models
    from PIL import Image
except ImportError as e:
    print(json.dumps({"error": f"Missing dependency: {e}"}))
    sys.exit(1)


class ContentLoss(nn.Module):
    """Content loss layer."""

    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = 0

    def forward(self, input):
        self.loss = nn.functional.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    """Compute Gram matrix for style representation."""
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)


class StyleLoss(nn.Module):
    """Style loss layer."""

    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = nn.functional.mse_loss(G, self.target)
        return input


class Normalization(nn.Module):
    """Normalize input image."""

    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean.to(img.device)) / self.std.to(img.device)


def load_image(path, size=None):
    """Load and preprocess image."""
    image = Image.open(path).convert("RGB")
    if size:
        image = image.resize(size, Image.LANCZOS)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image


def save_image(tensor, path):
    """Save tensor as image."""
    image = tensor.cpu().clone().squeeze(0)
    image = image.clamp(0, 1)
    image = transforms.ToPILImage()(image)
    image.save(path)


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, content_layers, style_layers,
                               device):
    """Build style transfer model with content and style losses."""
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Trim layers after last content/style loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(params):
    """Run neural style transfer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    start_time = time.time()

    # Load images
    content_img = load_image(params["content_path"]).to(device)
    size = (content_img.shape[3], content_img.shape[2])
    if params.get("output_resolution"):
        size = tuple(params["output_resolution"])

    style_img = load_image(params["style_path"], size).to(device)
    content_img = load_image(params["content_path"], size).to(device)

    # Initialize output
    if params.get("seed"):
        torch.manual_seed(params["seed"])
    input_img = content_img.clone()

    # Load VGG
    cnn = models.vgg19(pretrained=True).features.to(device).eval()
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # Map layer names
    content_layers_map = {
        "conv4_2": "conv_8",
        "conv5_2": "conv_13",
    }
    style_layers_map = {
        "conv1_1": "conv_1",
        "conv2_1": "conv_3",
        "conv3_1": "conv_5",
        "conv4_1": "conv_7",
        "conv5_1": "conv_12",
    }

    content_layers = [content_layers_map.get(l, l) for l in params["content_layers"]]
    style_layers = [style_layers_map.get(l, l) for l in params["style_layers"]]

    # Build model
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, cnn_normalization_mean, cnn_normalization_std,
        style_img, content_img, content_layers, style_layers, device
    )

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img], lr=params["learning_rate"])

    iterations = params["iterations"]
    content_weight = params["content_weight"]
    style_weight = params["style_weight"]
    tv_weight = params.get("tv_weight", 0)

    run = [0]
    final_content_loss = 0
    final_style_loss = 0

    while run[0] < iterations:
        def closure():
            nonlocal final_content_loss, final_style_loss

            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            content_score = sum(cl.loss for cl in content_losses)
            style_score = sum(sl.loss for sl in style_losses)

            content_score *= content_weight
            style_score *= style_weight

            loss = content_score + style_score

            # Total variation loss
            if tv_weight > 0:
                tv_loss = (
                    torch.sum(torch.abs(input_img[:, :, :, :-1] - input_img[:, :, :, 1:])) +
                    torch.sum(torch.abs(input_img[:, :, :-1, :] - input_img[:, :, 1:, :]))
                )
                loss += tv_weight * tv_loss

            loss.backward()

            run[0] += 1
            final_content_loss = content_score.item()
            final_style_loss = style_score.item()

            return loss

        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    # Save output
    save_image(input_img, params["output_path"])

    processing_time = time.time() - start_time

    return {
        "output_path": str(params["output_path"]),
        "content_loss": final_content_loss,
        "style_loss": final_style_loss,
        "total_loss": final_content_loss + final_style_loss,
        "iterations": iterations,
        "processing_time": processing_time,
        "metadata": {
            "device": str(device),
            "content_weight": str(content_weight),
            "style_weight": str(style_weight),
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Neural Style Transfer")
    parser.add_argument("--params", type=str, required=True, help="JSON parameters")
    args = parser.parse_args()

    try:
        params = json.loads(args.params)
        result = run_style_transfer(params)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()
"#.to_string()
}

/// Generate simple image filters as an alternative to neural style transfer
/// These can be applied purely in Rust without Python dependencies
pub mod filters {
    use image::{ImageBuffer, Rgb, RgbImage};

    /// Apply an X-ray style filter (inverted, high contrast)
    pub fn xray_filter(img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in img.enumerate_pixels() {
            // Convert to grayscale
            let gray = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;
            // Invert
            let inverted = 255 - gray;
            output.put_pixel(x, y, Rgb([inverted, inverted, inverted]));
        }

        output
    }

    /// Apply a thermal imaging style filter
    pub fn thermal_filter(img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in img.enumerate_pixels() {
            // Convert to grayscale intensity
            let intensity = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) / 255.0;

            // Map to thermal colormap (blue -> cyan -> green -> yellow -> red -> white)
            let (r, g, b) = if intensity < 0.2 {
                let t = intensity / 0.2;
                (0.0, 0.0, t)
            } else if intensity < 0.4 {
                let t = (intensity - 0.2) / 0.2;
                (0.0, t, 1.0)
            } else if intensity < 0.6 {
                let t = (intensity - 0.4) / 0.2;
                (t, 1.0, 1.0 - t)
            } else if intensity < 0.8 {
                let t = (intensity - 0.6) / 0.2;
                (1.0, 1.0 - t, 0.0)
            } else {
                let t = (intensity - 0.8) / 0.2;
                (1.0, t, t)
            };

            output.put_pixel(x, y, Rgb([
                (r * 255.0) as u8,
                (g * 255.0) as u8,
                (b * 255.0) as u8,
            ]));
        }

        output
    }

    /// Apply a sketch/edge detection filter
    pub fn sketch_filter(img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::new(width, height);

        // Sobel operators
        let sobel_x: [[i32; 3]; 3] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]];
        let sobel_y: [[i32; 3]; 3] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]];

        for y in 1..(height - 1) {
            for x in 1..(width - 1) {
                let mut gx = 0i32;
                let mut gy = 0i32;

                for ky in 0..3 {
                    for kx in 0..3 {
                        let px = img.get_pixel(x + kx - 1, y + ky - 1);
                        let gray = (0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32) as i32;

                        gx += gray * sobel_x[ky as usize][kx as usize];
                        gy += gray * sobel_y[ky as usize][kx as usize];
                    }
                }

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt().min(255.0) as u8;
                let edge = 255 - magnitude; // Invert for sketch effect

                output.put_pixel(x, y, Rgb([edge, edge, edge]));
            }
        }

        output
    }

    /// Apply a neon glow effect
    pub fn neon_glow_filter(img: &RgbImage, glow_color: [u8; 3]) -> RgbImage {
        // First get edges
        let edges = sketch_filter(img);
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::from_pixel(width, height, Rgb([10, 10, 30])); // Dark background

        // Apply glow to edges
        for (x, y, pixel) in edges.enumerate_pixels() {
            let edge_intensity = 1.0 - (pixel[0] as f32 / 255.0);

            if edge_intensity > 0.1 {
                let r = (glow_color[0] as f32 * edge_intensity) as u8;
                let g = (glow_color[1] as f32 * edge_intensity) as u8;
                let b = (glow_color[2] as f32 * edge_intensity) as u8;
                output.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        output
    }

    /// Apply a silhouette filter
    pub fn silhouette_filter(img: &RgbImage, threshold: u8) -> RgbImage {
        let (width, height) = img.dimensions();
        let mut output = ImageBuffer::new(width, height);

        for (x, y, pixel) in img.enumerate_pixels() {
            let gray = (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32) as u8;

            let value = if gray > threshold { 255 } else { 0 };
            output.put_pixel(x, y, Rgb([value, value, value]));
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_style_type_defaults() {
        let params = StyleTransferParams::default();
        assert_eq!(params.style_type, StyleType::RealisticHuman);
        assert_eq!(params.iterations, 300);
    }

    #[test]
    fn test_style_type_recommendations() {
        assert_eq!(StyleType::RealisticHuman.recommended_iterations(), 500);
        assert_eq!(StyleType::VanGogh.recommended_iterations(), 300);
        assert_eq!(StyleType::Silhouette.recommended_iterations(), 100);
    }

    #[test]
    fn test_style_image_names() {
        assert!(StyleType::VanGogh.style_image_name().is_some());
        assert!(StyleType::Custom.style_image_name().is_none());
    }

    #[test]
    fn test_artistic_params() {
        let params = StyleTransferParams::artistic(StyleType::VanGogh);
        assert_eq!(params.style_type, StyleType::VanGogh);
        assert_eq!(params.iterations, 300);
    }

    #[test]
    fn test_medical_params() {
        let params = StyleTransferParams::medical(StyleType::XRay);
        assert_eq!(params.style_type, StyleType::XRay);
        assert_eq!(params.iterations, 200);
    }

    #[test]
    fn test_preview_params() {
        let params = StyleTransferParams::preview();
        assert_eq!(params.iterations, 50);
        assert_eq!(params.learning_rate, 0.1);
    }

    #[test]
    fn test_custom_params() {
        let params = StyleTransferParams::custom(PathBuf::from("/tmp/style.jpg"));
        assert_eq!(params.style_type, StyleType::Custom);
        assert!(params.custom_style_path.is_some());
    }

    #[test]
    fn test_generate_script() {
        let script = generate_style_transfer_script();
        assert!(script.contains("VGG"));
        assert!(script.contains("gram_matrix"));
        assert!(script.contains("def run_style_transfer"));
    }

    #[test]
    fn test_xray_filter() {
        use image::{ImageBuffer, Rgb, RgbImage};
        let img: RgbImage = ImageBuffer::from_pixel(10, 10, Rgb([100, 100, 100]));
        let result = filters::xray_filter(&img);
        assert_eq!(result.dimensions(), (10, 10));
        // Should be inverted grayscale
        assert!(result.get_pixel(5, 5)[0] > 100);
    }

    #[test]
    fn test_thermal_filter() {
        use image::{ImageBuffer, Rgb, RgbImage};
        let img: RgbImage = ImageBuffer::from_pixel(10, 10, Rgb([128, 128, 128]));
        let result = filters::thermal_filter(&img);
        assert_eq!(result.dimensions(), (10, 10));
    }

    #[test]
    fn test_sketch_filter() {
        use image::{ImageBuffer, Rgb, RgbImage};
        let mut img: RgbImage = ImageBuffer::from_pixel(20, 20, Rgb([255, 255, 255]));
        // Add a dark line
        for x in 5..15 {
            img.put_pixel(x, 10, Rgb([0, 0, 0]));
        }
        let result = filters::sketch_filter(&img);
        assert_eq!(result.dimensions(), (20, 20));
    }

    #[test]
    fn test_neon_glow_filter() {
        use image::{ImageBuffer, Rgb, RgbImage};
        let mut img: RgbImage = ImageBuffer::from_pixel(20, 20, Rgb([255, 255, 255]));
        for x in 5..15 {
            img.put_pixel(x, 10, Rgb([0, 0, 0]));
        }
        let result = filters::neon_glow_filter(&img, [0, 255, 200]);
        assert_eq!(result.dimensions(), (20, 20));
    }

    #[test]
    fn test_silhouette_filter() {
        use image::{ImageBuffer, Rgb, RgbImage};
        let img: RgbImage = ImageBuffer::from_pixel(10, 10, Rgb([100, 100, 100]));
        let result = filters::silhouette_filter(&img, 128);
        assert_eq!(result.dimensions(), (10, 10));
        // Should be black (below threshold)
        assert_eq!(result.get_pixel(5, 5)[0], 0);
    }
}
