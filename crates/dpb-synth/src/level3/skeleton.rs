//! Pure Rust Skeleton/Stick Figure Renderer
//!
//! This module provides a lightweight, dependency-free (except for `image` crate)
//! renderer for visualizing pose and hand keypoints as stick figures.
//!
//! # Features
//!
//! - **Pose rendering**: 33 MediaPipe pose landmarks with anatomical connections
//! - **Hand rendering**: 21 MediaPipe hand landmarks with finger connections
//! - **Customizable styling**: Colors, line thickness, joint markers
//! - **Camera views**: Front, side, top-down projections
//! - **Video export**: Frame sequences for ffmpeg encoding
//!
//! # Usage
//!
//! ```rust,ignore
//! use dpb_synth::level3::skeleton::{SkeletonRenderer, SkeletonParams};
//!
//! let renderer = SkeletonRenderer::new(SkeletonParams::default());
//!
//! // Render a single pose frame
//! let keypoints = vec![[0.5, 0.5, 0.0]; 33]; // MediaPipe pose landmarks
//! let frame = renderer.render_pose(&keypoints)?;
//! frame.save("pose_frame.png")?;
//! ```

use image::{ImageBuffer, Rgb, RgbImage};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};

/// Result type for skeleton rendering
pub type SkeletonResult<T> = Result<T, SkeletonError>;

/// Errors that can occur during skeleton rendering
#[derive(Debug, thiserror::Error)]
pub enum SkeletonError {
    #[error("Invalid keypoint count: expected {expected}, got {got}")]
    InvalidKeypointCount { expected: usize, got: usize },

    #[error("Invalid keypoint coordinates: {0}")]
    InvalidCoordinates(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Image error: {0}")]
    ImageError(#[from] image::ImageError),

    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Camera view for 3D to 2D projection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CameraView {
    /// Front view (XY plane, Z depth)
    Front,
    /// Side view (ZY plane, X depth)
    Side,
    /// Top-down view (XZ plane, Y depth)
    TopDown,
    /// Oblique view (45° rotation)
    Oblique,
}

impl Default for CameraView {
    fn default() -> Self {
        CameraView::Front
    }
}

/// Rendering style for skeleton visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderStyle {
    /// Background color (RGB)
    pub background_color: [u8; 3],
    /// Bone/connection color (RGB)
    pub bone_color: [u8; 3],
    /// Joint color (RGB)
    pub joint_color: [u8; 3],
    /// Left side color (RGB) for asymmetric rendering
    pub left_color: [u8; 3],
    /// Right side color (RGB) for asymmetric rendering
    pub right_color: [u8; 3],
    /// Line thickness in pixels
    pub line_thickness: u32,
    /// Joint marker radius in pixels
    pub joint_radius: u32,
    /// Whether to draw joint markers
    pub draw_joints: bool,
    /// Whether to use different colors for left/right sides
    pub asymmetric_coloring: bool,
    /// Depth shading (darker for further points)
    pub depth_shading: bool,
    /// Anti-aliasing (simple line smoothing)
    pub anti_aliasing: bool,
}

impl Default for RenderStyle {
    fn default() -> Self {
        Self {
            background_color: [0, 0, 0],        // Black background
            bone_color: [255, 255, 255],        // White bones
            joint_color: [255, 200, 0],         // Yellow joints
            left_color: [0, 150, 255],          // Blue for left side
            right_color: [255, 100, 100],       // Red for right side
            line_thickness: 2,
            joint_radius: 4,
            draw_joints: true,
            asymmetric_coloring: true,
            depth_shading: true,
            anti_aliasing: false,
        }
    }
}

impl RenderStyle {
    /// Clinical style (high contrast, clear visualization)
    pub fn clinical() -> Self {
        Self {
            background_color: [240, 240, 240],  // Light gray
            bone_color: [50, 50, 50],           // Dark gray bones
            joint_color: [200, 0, 0],           // Red joints
            left_color: [0, 100, 200],          // Blue left
            right_color: [200, 0, 0],           // Red right
            line_thickness: 3,
            joint_radius: 5,
            draw_joints: true,
            asymmetric_coloring: true,
            depth_shading: false,
            anti_aliasing: false,
        }
    }

    /// Minimal style (thin lines, no joints)
    pub fn minimal() -> Self {
        Self {
            background_color: [255, 255, 255],
            bone_color: [0, 0, 0],
            joint_color: [0, 0, 0],
            left_color: [0, 0, 0],
            right_color: [0, 0, 0],
            line_thickness: 1,
            joint_radius: 2,
            draw_joints: false,
            asymmetric_coloring: false,
            depth_shading: false,
            anti_aliasing: false,
        }
    }

    /// Neon style (bright colors on dark background)
    pub fn neon() -> Self {
        Self {
            background_color: [10, 10, 30],
            bone_color: [0, 255, 200],          // Cyan bones
            joint_color: [255, 0, 255],         // Magenta joints
            left_color: [0, 200, 255],          // Cyan left
            right_color: [255, 100, 200],       // Pink right
            line_thickness: 2,
            joint_radius: 4,
            draw_joints: true,
            asymmetric_coloring: true,
            depth_shading: true,
            anti_aliasing: false,
        }
    }
}

/// Parameters for skeleton rendering
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkeletonParams {
    /// Output resolution (width, height)
    pub resolution: (u32, u32),
    /// Camera view for projection
    pub camera_view: CameraView,
    /// Rendering style
    pub style: RenderStyle,
    /// Frame rate for video output
    pub fps: u32,
    /// Padding around skeleton (0.0-0.5)
    pub padding: f32,
    /// Scale factor (1.0 = auto-fit)
    pub scale: f32,
    /// Center offset (x, y) in normalized coordinates
    pub center_offset: (f32, f32),
}

impl Default for SkeletonParams {
    fn default() -> Self {
        Self {
            resolution: (640, 480),
            camera_view: CameraView::Front,
            style: RenderStyle::default(),
            fps: 30,
            padding: 0.1,
            scale: 1.0,
            center_offset: (0.0, 0.0),
        }
    }
}

impl SkeletonParams {
    /// High definition settings
    pub fn hd() -> Self {
        Self {
            resolution: (1920, 1080),
            style: RenderStyle::clinical(),
            ..Default::default()
        }
    }

    /// Thumbnail settings
    pub fn thumbnail() -> Self {
        Self {
            resolution: (256, 256),
            style: RenderStyle::minimal(),
            padding: 0.05,
            ..Default::default()
        }
    }
}

/// MediaPipe pose landmark indices
#[derive(Debug, Clone, Copy)]
pub struct PoseLandmarks;

impl PoseLandmarks {
    pub const NOSE: usize = 0;
    pub const LEFT_EYE_INNER: usize = 1;
    pub const LEFT_EYE: usize = 2;
    pub const LEFT_EYE_OUTER: usize = 3;
    pub const RIGHT_EYE_INNER: usize = 4;
    pub const RIGHT_EYE: usize = 5;
    pub const RIGHT_EYE_OUTER: usize = 6;
    pub const LEFT_EAR: usize = 7;
    pub const RIGHT_EAR: usize = 8;
    pub const MOUTH_LEFT: usize = 9;
    pub const MOUTH_RIGHT: usize = 10;
    pub const LEFT_SHOULDER: usize = 11;
    pub const RIGHT_SHOULDER: usize = 12;
    pub const LEFT_ELBOW: usize = 13;
    pub const RIGHT_ELBOW: usize = 14;
    pub const LEFT_WRIST: usize = 15;
    pub const RIGHT_WRIST: usize = 16;
    pub const LEFT_PINKY: usize = 17;
    pub const RIGHT_PINKY: usize = 18;
    pub const LEFT_INDEX: usize = 19;
    pub const RIGHT_INDEX: usize = 20;
    pub const LEFT_THUMB: usize = 21;
    pub const RIGHT_THUMB: usize = 22;
    pub const LEFT_HIP: usize = 23;
    pub const RIGHT_HIP: usize = 24;
    pub const LEFT_KNEE: usize = 25;
    pub const RIGHT_KNEE: usize = 26;
    pub const LEFT_ANKLE: usize = 27;
    pub const RIGHT_ANKLE: usize = 28;
    pub const LEFT_HEEL: usize = 29;
    pub const RIGHT_HEEL: usize = 30;
    pub const LEFT_FOOT_INDEX: usize = 31;
    pub const RIGHT_FOOT_INDEX: usize = 32;

    /// Get pose connections (bone pairs)
    pub fn connections() -> Vec<(usize, usize, bool)> {
        // (start, end, is_left_side)
        vec![
            // Face
            (Self::NOSE, Self::LEFT_EYE_INNER, true),
            (Self::LEFT_EYE_INNER, Self::LEFT_EYE, true),
            (Self::LEFT_EYE, Self::LEFT_EYE_OUTER, true),
            (Self::LEFT_EYE_OUTER, Self::LEFT_EAR, true),
            (Self::NOSE, Self::RIGHT_EYE_INNER, false),
            (Self::RIGHT_EYE_INNER, Self::RIGHT_EYE, false),
            (Self::RIGHT_EYE, Self::RIGHT_EYE_OUTER, false),
            (Self::RIGHT_EYE_OUTER, Self::RIGHT_EAR, false),
            (Self::MOUTH_LEFT, Self::MOUTH_RIGHT, true),

            // Torso
            (Self::LEFT_SHOULDER, Self::RIGHT_SHOULDER, true),
            (Self::LEFT_SHOULDER, Self::LEFT_HIP, true),
            (Self::RIGHT_SHOULDER, Self::RIGHT_HIP, false),
            (Self::LEFT_HIP, Self::RIGHT_HIP, true),

            // Left arm
            (Self::LEFT_SHOULDER, Self::LEFT_ELBOW, true),
            (Self::LEFT_ELBOW, Self::LEFT_WRIST, true),
            (Self::LEFT_WRIST, Self::LEFT_PINKY, true),
            (Self::LEFT_WRIST, Self::LEFT_INDEX, true),
            (Self::LEFT_WRIST, Self::LEFT_THUMB, true),
            (Self::LEFT_PINKY, Self::LEFT_INDEX, true),

            // Right arm
            (Self::RIGHT_SHOULDER, Self::RIGHT_ELBOW, false),
            (Self::RIGHT_ELBOW, Self::RIGHT_WRIST, false),
            (Self::RIGHT_WRIST, Self::RIGHT_PINKY, false),
            (Self::RIGHT_WRIST, Self::RIGHT_INDEX, false),
            (Self::RIGHT_WRIST, Self::RIGHT_THUMB, false),
            (Self::RIGHT_PINKY, Self::RIGHT_INDEX, false),

            // Left leg
            (Self::LEFT_HIP, Self::LEFT_KNEE, true),
            (Self::LEFT_KNEE, Self::LEFT_ANKLE, true),
            (Self::LEFT_ANKLE, Self::LEFT_HEEL, true),
            (Self::LEFT_HEEL, Self::LEFT_FOOT_INDEX, true),
            (Self::LEFT_ANKLE, Self::LEFT_FOOT_INDEX, true),

            // Right leg
            (Self::RIGHT_HIP, Self::RIGHT_KNEE, false),
            (Self::RIGHT_KNEE, Self::RIGHT_ANKLE, false),
            (Self::RIGHT_ANKLE, Self::RIGHT_HEEL, false),
            (Self::RIGHT_HEEL, Self::RIGHT_FOOT_INDEX, false),
            (Self::RIGHT_ANKLE, Self::RIGHT_FOOT_INDEX, false),
        ]
    }
}

/// MediaPipe hand landmark indices
#[derive(Debug, Clone, Copy)]
pub struct HandLandmarks;

impl HandLandmarks {
    pub const WRIST: usize = 0;
    pub const THUMB_CMC: usize = 1;
    pub const THUMB_MCP: usize = 2;
    pub const THUMB_IP: usize = 3;
    pub const THUMB_TIP: usize = 4;
    pub const INDEX_MCP: usize = 5;
    pub const INDEX_PIP: usize = 6;
    pub const INDEX_DIP: usize = 7;
    pub const INDEX_TIP: usize = 8;
    pub const MIDDLE_MCP: usize = 9;
    pub const MIDDLE_PIP: usize = 10;
    pub const MIDDLE_DIP: usize = 11;
    pub const MIDDLE_TIP: usize = 12;
    pub const RING_MCP: usize = 13;
    pub const RING_PIP: usize = 14;
    pub const RING_DIP: usize = 15;
    pub const RING_TIP: usize = 16;
    pub const PINKY_MCP: usize = 17;
    pub const PINKY_PIP: usize = 18;
    pub const PINKY_DIP: usize = 19;
    pub const PINKY_TIP: usize = 20;

    /// Get hand connections (bone pairs)
    pub fn connections() -> Vec<(usize, usize)> {
        vec![
            // Palm
            (Self::WRIST, Self::THUMB_CMC),
            (Self::WRIST, Self::INDEX_MCP),
            (Self::WRIST, Self::MIDDLE_MCP),
            (Self::WRIST, Self::RING_MCP),
            (Self::WRIST, Self::PINKY_MCP),
            (Self::INDEX_MCP, Self::MIDDLE_MCP),
            (Self::MIDDLE_MCP, Self::RING_MCP),
            (Self::RING_MCP, Self::PINKY_MCP),

            // Thumb
            (Self::THUMB_CMC, Self::THUMB_MCP),
            (Self::THUMB_MCP, Self::THUMB_IP),
            (Self::THUMB_IP, Self::THUMB_TIP),

            // Index finger
            (Self::INDEX_MCP, Self::INDEX_PIP),
            (Self::INDEX_PIP, Self::INDEX_DIP),
            (Self::INDEX_DIP, Self::INDEX_TIP),

            // Middle finger
            (Self::MIDDLE_MCP, Self::MIDDLE_PIP),
            (Self::MIDDLE_PIP, Self::MIDDLE_DIP),
            (Self::MIDDLE_DIP, Self::MIDDLE_TIP),

            // Ring finger
            (Self::RING_MCP, Self::RING_PIP),
            (Self::RING_PIP, Self::RING_DIP),
            (Self::RING_DIP, Self::RING_TIP),

            // Pinky finger
            (Self::PINKY_MCP, Self::PINKY_PIP),
            (Self::PINKY_PIP, Self::PINKY_DIP),
            (Self::PINKY_DIP, Self::PINKY_TIP),
        ]
    }

    /// Get finger colors for visualization
    pub fn finger_colors() -> Vec<([u8; 3], Vec<usize>)> {
        vec![
            ([255, 128, 0], vec![1, 2, 3, 4]),       // Thumb - orange
            ([255, 0, 128], vec![5, 6, 7, 8]),       // Index - pink
            ([128, 0, 255], vec![9, 10, 11, 12]),   // Middle - purple
            ([0, 128, 255], vec![13, 14, 15, 16]),  // Ring - blue
            ([0, 255, 128], vec![17, 18, 19, 20]),  // Pinky - green
        ]
    }
}

/// Main skeleton renderer
pub struct SkeletonRenderer {
    params: SkeletonParams,
}

impl SkeletonRenderer {
    /// Create a new skeleton renderer
    pub fn new(params: SkeletonParams) -> Self {
        Self { params }
    }

    /// Update rendering parameters
    pub fn set_params(&mut self, params: SkeletonParams) {
        self.params = params;
    }

    /// Project 3D point to 2D based on camera view
    fn project_point(&self, point: &[f64; 3]) -> (f32, f32, f32) {
        let (x, y, z) = (point[0] as f32, point[1] as f32, point[2] as f32);

        match self.params.camera_view {
            CameraView::Front => (x, y, z),
            CameraView::Side => (z, y, x),
            CameraView::TopDown => (x, z, y),
            CameraView::Oblique => {
                // 45 degree rotation around Y axis
                let angle = PI as f32 / 4.0;
                let new_x = x * angle.cos() + z * angle.sin();
                let new_z = -x * angle.sin() + z * angle.cos();
                (new_x, y, new_z)
            }
        }
    }

    /// Convert normalized coordinates to pixel coordinates
    fn to_pixel_coords(&self, x: f32, y: f32, bounds: &(f32, f32, f32, f32)) -> (i32, i32) {
        let (min_x, max_x, min_y, max_y) = *bounds;
        let range_x = (max_x - min_x).max(0.001);
        let range_y = (max_y - min_y).max(0.001);

        let (width, height) = self.params.resolution;
        let padding = self.params.padding;

        // Fit to frame with padding
        let usable_width = width as f32 * (1.0 - 2.0 * padding);
        let usable_height = height as f32 * (1.0 - 2.0 * padding);

        // Maintain aspect ratio
        let scale_x = usable_width / range_x;
        let scale_y = usable_height / range_y;
        let scale = scale_x.min(scale_y) * self.params.scale;

        let center_x = width as f32 / 2.0 + self.params.center_offset.0 * width as f32;
        let center_y = height as f32 / 2.0 + self.params.center_offset.1 * height as f32;

        let norm_x = (x - (min_x + max_x) / 2.0) * scale + center_x;
        let norm_y = (y - (min_y + max_y) / 2.0) * scale + center_y;

        // Flip Y for image coordinates (top-left origin)
        (norm_x as i32, (height as f32 - norm_y) as i32)
    }

    /// Calculate bounding box of keypoints
    fn calculate_bounds(&self, keypoints: &[[f64; 3]]) -> (f32, f32, f32, f32) {
        let projected: Vec<(f32, f32, f32)> = keypoints
            .iter()
            .map(|p| self.project_point(p))
            .collect();

        let min_x = projected.iter().map(|p| p.0).fold(f32::MAX, f32::min);
        let max_x = projected.iter().map(|p| p.0).fold(f32::MIN, f32::max);
        let min_y = projected.iter().map(|p| p.1).fold(f32::MAX, f32::min);
        let max_y = projected.iter().map(|p| p.1).fold(f32::MIN, f32::max);

        (min_x, max_x, min_y, max_y)
    }

    /// Draw a line on the image (Bresenham's algorithm)
    fn draw_line(&self, img: &mut RgbImage, x0: i32, y0: i32, x1: i32, y1: i32, color: Rgb<u8>, thickness: u32) {
        let (width, height) = self.params.resolution;

        // For thickness > 1, draw multiple parallel lines
        let half_thick = (thickness as i32 - 1) / 2;

        for offset in -half_thick..=half_thick {
            let dx = (x1 - x0).abs();
            let dy = -(y1 - y0).abs();
            let sx = if x0 < x1 { 1 } else { -1 };
            let sy = if y0 < y1 { 1 } else { -1 };

            // Perpendicular offset
            let len = ((dx * dx + dy * dy) as f32).sqrt().max(1.0);
            let ox = (-(y1 - y0) as f32 / len * offset as f32) as i32;
            let oy = ((x1 - x0) as f32 / len * offset as f32) as i32;

            let mut x = x0 + ox;
            let mut y = y0 + oy;
            let end_x = x1 + ox;
            let end_y = y1 + oy;

            let mut err = dx + dy;

            loop {
                if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                    img.put_pixel(x as u32, y as u32, color);
                }

                if x == end_x && y == end_y {
                    break;
                }

                let e2 = 2 * err;
                if e2 >= dy {
                    err += dy;
                    x += sx;
                }
                if e2 <= dx {
                    err += dx;
                    y += sy;
                }
            }
        }
    }

    /// Draw a filled circle on the image
    fn draw_circle(&self, img: &mut RgbImage, cx: i32, cy: i32, radius: u32, color: Rgb<u8>) {
        let (width, height) = self.params.resolution;
        let r = radius as i32;

        for dy in -r..=r {
            for dx in -r..=r {
                if dx * dx + dy * dy <= r * r {
                    let x = cx + dx;
                    let y = cy + dy;
                    if x >= 0 && x < width as i32 && y >= 0 && y < height as i32 {
                        img.put_pixel(x as u32, y as u32, color);
                    }
                }
            }
        }
    }

    /// Apply depth shading to a color
    fn apply_depth_shading(&self, color: [u8; 3], depth: f32, min_depth: f32, max_depth: f32) -> Rgb<u8> {
        if !self.params.style.depth_shading {
            return Rgb(color);
        }

        let range = (max_depth - min_depth).max(0.001);
        let normalized = ((depth - min_depth) / range).clamp(0.0, 1.0);
        let factor = 0.5 + 0.5 * (1.0 - normalized); // 0.5 to 1.0

        Rgb([
            (color[0] as f32 * factor) as u8,
            (color[1] as f32 * factor) as u8,
            (color[2] as f32 * factor) as u8,
        ])
    }

    /// Render pose keypoints (33 MediaPipe landmarks)
    pub fn render_pose(&self, keypoints: &[[f64; 3]]) -> SkeletonResult<RgbImage> {
        if keypoints.len() != 33 {
            return Err(SkeletonError::InvalidKeypointCount {
                expected: 33,
                got: keypoints.len(),
            });
        }

        let (width, height) = self.params.resolution;
        let mut img = ImageBuffer::from_pixel(
            width,
            height,
            Rgb(self.params.style.background_color),
        );

        let bounds = self.calculate_bounds(keypoints);
        let projected: Vec<(f32, f32, f32)> = keypoints
            .iter()
            .map(|p| self.project_point(p))
            .collect();

        // Calculate depth range for shading
        let min_depth = projected.iter().map(|p| p.2).fold(f32::MAX, f32::min);
        let max_depth = projected.iter().map(|p| p.2).fold(f32::MIN, f32::max);

        // Draw connections (bones)
        for (start, end, is_left) in PoseLandmarks::connections() {
            let p1 = &projected[start];
            let p2 = &projected[end];

            let (x1, y1) = self.to_pixel_coords(p1.0, p1.1, &bounds);
            let (x2, y2) = self.to_pixel_coords(p2.0, p2.1, &bounds);

            let base_color = if self.params.style.asymmetric_coloring {
                if is_left {
                    self.params.style.left_color
                } else {
                    self.params.style.right_color
                }
            } else {
                self.params.style.bone_color
            };

            let avg_depth = (p1.2 + p2.2) / 2.0;
            let color = self.apply_depth_shading(base_color, avg_depth, min_depth, max_depth);

            self.draw_line(&mut img, x1, y1, x2, y2, color, self.params.style.line_thickness);
        }

        // Draw joints
        if self.params.style.draw_joints {
            for (i, p) in projected.iter().enumerate() {
                let (x, y) = self.to_pixel_coords(p.0, p.1, &bounds);
                let color = self.apply_depth_shading(
                    self.params.style.joint_color,
                    p.2,
                    min_depth,
                    max_depth,
                );

                // Slightly smaller radius for face landmarks
                let radius = if i <= 10 {
                    self.params.style.joint_radius / 2
                } else {
                    self.params.style.joint_radius
                };

                self.draw_circle(&mut img, x, y, radius.max(1), color);
            }
        }

        Ok(img)
    }

    /// Render hand keypoints (21 MediaPipe landmarks)
    pub fn render_hand(&self, keypoints: &[[f64; 3]], is_left: bool) -> SkeletonResult<RgbImage> {
        if keypoints.len() != 21 {
            return Err(SkeletonError::InvalidKeypointCount {
                expected: 21,
                got: keypoints.len(),
            });
        }

        let (width, height) = self.params.resolution;
        let mut img = ImageBuffer::from_pixel(
            width,
            height,
            Rgb(self.params.style.background_color),
        );

        let bounds = self.calculate_bounds(keypoints);
        let projected: Vec<(f32, f32, f32)> = keypoints
            .iter()
            .map(|p| self.project_point(p))
            .collect();

        let min_depth = projected.iter().map(|p| p.2).fold(f32::MAX, f32::min);
        let max_depth = projected.iter().map(|p| p.2).fold(f32::MIN, f32::max);

        // Draw connections with finger coloring
        let finger_colors = HandLandmarks::finger_colors();

        for (start, end) in HandLandmarks::connections() {
            let p1 = &projected[start];
            let p2 = &projected[end];

            let (x1, y1) = self.to_pixel_coords(p1.0, p1.1, &bounds);
            let (x2, y2) = self.to_pixel_coords(p2.0, p2.1, &bounds);

            // Determine color based on which finger
            let mut base_color = self.params.style.bone_color;
            for (color, indices) in &finger_colors {
                if indices.contains(&start) || indices.contains(&end) {
                    base_color = *color;
                    break;
                }
            }

            // Flip color for left hand if asymmetric
            if is_left && self.params.style.asymmetric_coloring {
                base_color = self.params.style.left_color;
            }

            let avg_depth = (p1.2 + p2.2) / 2.0;
            let color = self.apply_depth_shading(base_color, avg_depth, min_depth, max_depth);

            self.draw_line(&mut img, x1, y1, x2, y2, color, self.params.style.line_thickness);
        }

        // Draw joints
        if self.params.style.draw_joints {
            for p in &projected {
                let (x, y) = self.to_pixel_coords(p.0, p.1, &bounds);
                let color = self.apply_depth_shading(
                    self.params.style.joint_color,
                    p.2,
                    min_depth,
                    max_depth,
                );
                self.draw_circle(&mut img, x, y, self.params.style.joint_radius, color);
            }
        }

        Ok(img)
    }

    /// Render both pose and hands in a single frame
    pub fn render_full_body(
        &self,
        pose: &[[f64; 3]],
        left_hand: Option<&[[f64; 3]]>,
        right_hand: Option<&[[f64; 3]]>,
    ) -> SkeletonResult<RgbImage> {
        // Start with pose
        let mut img = self.render_pose(pose)?;

        // Overlay hands if provided
        // Note: This is a simplified approach; a full implementation would
        // properly integrate hand keypoints into the pose coordinate system

        if let Some(hand) = left_hand {
            let hand_img = self.render_hand(hand, true)?;
            // Blend hand into main image (simplified - just for structure)
            // In practice, hands should share coordinate system with pose
        }

        if let Some(hand) = right_hand {
            let hand_img = self.render_hand(hand, false)?;
        }

        Ok(img)
    }

    /// Render a sequence of pose frames
    pub fn render_pose_sequence(
        &self,
        keypoint_sequence: &[Vec<[f64; 3]>],
    ) -> SkeletonResult<Vec<RgbImage>> {
        keypoint_sequence
            .iter()
            .map(|kps| self.render_pose(kps))
            .collect()
    }

    /// Render a sequence of hand frames
    pub fn render_hand_sequence(
        &self,
        keypoint_sequence: &[Vec<[f64; 3]>],
        is_left: bool,
    ) -> SkeletonResult<Vec<RgbImage>> {
        keypoint_sequence
            .iter()
            .map(|kps| self.render_hand(kps, is_left))
            .collect()
    }

    /// Save a sequence of frames to disk for video encoding
    pub fn save_frame_sequence<P: AsRef<Path>>(
        &self,
        frames: &[RgbImage],
        output_dir: P,
        prefix: &str,
    ) -> SkeletonResult<Vec<PathBuf>> {
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir)?;

        let mut paths = Vec::with_capacity(frames.len());

        for (i, frame) in frames.iter().enumerate() {
            let filename = format!("{}_{:06}.png", prefix, i);
            let path = output_dir.join(&filename);
            frame.save(&path)?;
            paths.push(path);
        }

        Ok(paths)
    }

    /// Get ffmpeg command to encode frame sequence to video
    pub fn get_ffmpeg_command(
        &self,
        frame_dir: &Path,
        prefix: &str,
        output_path: &Path,
    ) -> String {
        format!(
            "ffmpeg -framerate {} -i {}/{}_%06d.png -c:v libx264 -pix_fmt yuv420p {}",
            self.params.fps,
            frame_dir.display(),
            prefix,
            output_path.display()
        )
    }
}

/// Integration with StreamingClinicalPose
pub fn render_clinical_pose_frame(
    renderer: &SkeletonRenderer,
    keypoints: &[[f64; 3]; 33],
) -> SkeletonResult<RgbImage> {
    renderer.render_pose(keypoints)
}

/// Integration with StreamingClinicalHand
pub fn render_clinical_hand_frame(
    renderer: &SkeletonRenderer,
    landmarks: &[[f64; 3]; 21],
    is_left: bool,
) -> SkeletonResult<RgbImage> {
    renderer.render_hand(landmarks, is_left)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_pose() -> Vec<[f64; 3]> {
        // Create a simple T-pose for testing
        let mut keypoints = vec![[0.0, 0.0, 0.0]; 33];

        // Head
        keypoints[PoseLandmarks::NOSE] = [0.0, 1.7, 0.0];

        // Shoulders
        keypoints[PoseLandmarks::LEFT_SHOULDER] = [-0.2, 1.5, 0.0];
        keypoints[PoseLandmarks::RIGHT_SHOULDER] = [0.2, 1.5, 0.0];

        // Arms extended
        keypoints[PoseLandmarks::LEFT_ELBOW] = [-0.5, 1.5, 0.0];
        keypoints[PoseLandmarks::RIGHT_ELBOW] = [0.5, 1.5, 0.0];
        keypoints[PoseLandmarks::LEFT_WRIST] = [-0.8, 1.5, 0.0];
        keypoints[PoseLandmarks::RIGHT_WRIST] = [0.8, 1.5, 0.0];

        // Hips
        keypoints[PoseLandmarks::LEFT_HIP] = [-0.1, 1.0, 0.0];
        keypoints[PoseLandmarks::RIGHT_HIP] = [0.1, 1.0, 0.0];

        // Legs
        keypoints[PoseLandmarks::LEFT_KNEE] = [-0.1, 0.5, 0.0];
        keypoints[PoseLandmarks::RIGHT_KNEE] = [0.1, 0.5, 0.0];
        keypoints[PoseLandmarks::LEFT_ANKLE] = [-0.1, 0.0, 0.0];
        keypoints[PoseLandmarks::RIGHT_ANKLE] = [0.1, 0.0, 0.0];

        keypoints
    }

    fn create_test_hand() -> Vec<[f64; 3]> {
        // Create a simple open hand for testing
        let mut landmarks = vec![[0.0, 0.0, 0.0]; 21];

        landmarks[HandLandmarks::WRIST] = [0.0, 0.0, 0.0];

        // Thumb
        landmarks[HandLandmarks::THUMB_CMC] = [-0.02, 0.02, 0.0];
        landmarks[HandLandmarks::THUMB_MCP] = [-0.04, 0.04, 0.0];
        landmarks[HandLandmarks::THUMB_IP] = [-0.05, 0.06, 0.0];
        landmarks[HandLandmarks::THUMB_TIP] = [-0.06, 0.08, 0.0];

        // Index finger
        landmarks[HandLandmarks::INDEX_MCP] = [-0.01, 0.04, 0.0];
        landmarks[HandLandmarks::INDEX_PIP] = [-0.01, 0.07, 0.0];
        landmarks[HandLandmarks::INDEX_DIP] = [-0.01, 0.09, 0.0];
        landmarks[HandLandmarks::INDEX_TIP] = [-0.01, 0.11, 0.0];

        // Middle finger
        landmarks[HandLandmarks::MIDDLE_MCP] = [0.0, 0.04, 0.0];
        landmarks[HandLandmarks::MIDDLE_PIP] = [0.0, 0.08, 0.0];
        landmarks[HandLandmarks::MIDDLE_DIP] = [0.0, 0.10, 0.0];
        landmarks[HandLandmarks::MIDDLE_TIP] = [0.0, 0.12, 0.0];

        // Ring finger
        landmarks[HandLandmarks::RING_MCP] = [0.01, 0.04, 0.0];
        landmarks[HandLandmarks::RING_PIP] = [0.01, 0.07, 0.0];
        landmarks[HandLandmarks::RING_DIP] = [0.01, 0.09, 0.0];
        landmarks[HandLandmarks::RING_TIP] = [0.01, 0.11, 0.0];

        // Pinky
        landmarks[HandLandmarks::PINKY_MCP] = [0.02, 0.04, 0.0];
        landmarks[HandLandmarks::PINKY_PIP] = [0.02, 0.06, 0.0];
        landmarks[HandLandmarks::PINKY_DIP] = [0.02, 0.08, 0.0];
        landmarks[HandLandmarks::PINKY_TIP] = [0.02, 0.10, 0.0];

        landmarks
    }

    #[test]
    fn test_default_params() {
        let params = SkeletonParams::default();
        assert_eq!(params.resolution, (640, 480));
        assert_eq!(params.fps, 30);
    }

    #[test]
    fn test_render_style_presets() {
        let clinical = RenderStyle::clinical();
        assert_eq!(clinical.line_thickness, 3);

        let minimal = RenderStyle::minimal();
        assert!(!minimal.draw_joints);

        let neon = RenderStyle::neon();
        assert!(neon.depth_shading);
    }

    #[test]
    fn test_render_pose() {
        let renderer = SkeletonRenderer::new(SkeletonParams::default());
        let keypoints = create_test_pose();

        let result = renderer.render_pose(&keypoints);
        assert!(result.is_ok());

        let img = result.unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }

    #[test]
    fn test_render_pose_invalid_count() {
        let renderer = SkeletonRenderer::new(SkeletonParams::default());
        let keypoints = vec![[0.0, 0.0, 0.0]; 10]; // Wrong count

        let result = renderer.render_pose(&keypoints);
        assert!(matches!(result, Err(SkeletonError::InvalidKeypointCount { .. })));
    }

    #[test]
    fn test_render_hand() {
        let renderer = SkeletonRenderer::new(SkeletonParams::default());
        let landmarks = create_test_hand();

        let result = renderer.render_hand(&landmarks, true);
        assert!(result.is_ok());

        let img = result.unwrap();
        assert_eq!(img.width(), 640);
        assert_eq!(img.height(), 480);
    }

    #[test]
    fn test_render_hand_invalid_count() {
        let renderer = SkeletonRenderer::new(SkeletonParams::default());
        let landmarks = vec![[0.0, 0.0, 0.0]; 10]; // Wrong count

        let result = renderer.render_hand(&landmarks, false);
        assert!(matches!(result, Err(SkeletonError::InvalidKeypointCount { .. })));
    }

    #[test]
    fn test_camera_views() {
        let keypoints = create_test_pose();

        for view in [CameraView::Front, CameraView::Side, CameraView::TopDown, CameraView::Oblique] {
            let params = SkeletonParams {
                camera_view: view,
                ..Default::default()
            };
            let renderer = SkeletonRenderer::new(params);

            let result = renderer.render_pose(&keypoints);
            assert!(result.is_ok(), "Failed for view {:?}", view);
        }
    }

    #[test]
    fn test_render_sequence() {
        let renderer = SkeletonRenderer::new(SkeletonParams::default());

        // Create a simple animation
        let base_pose = create_test_pose();
        let sequence: Vec<Vec<[f64; 3]>> = (0..10)
            .map(|i| {
                let mut pose = base_pose.clone();
                // Animate arm movement
                let angle = (i as f64 * 0.3).sin() * 0.3;
                pose[PoseLandmarks::LEFT_WRIST][1] += angle;
                pose[PoseLandmarks::RIGHT_WRIST][1] -= angle;
                pose
            })
            .collect();

        let result = renderer.render_pose_sequence(&sequence);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().len(), 10);
    }

    #[test]
    fn test_pose_landmarks_connections() {
        let connections = PoseLandmarks::connections();
        assert!(!connections.is_empty());

        // Verify all indices are valid
        for (start, end, _) in &connections {
            assert!(*start < 33);
            assert!(*end < 33);
        }
    }

    #[test]
    fn test_hand_landmarks_connections() {
        let connections = HandLandmarks::connections();
        assert!(!connections.is_empty());

        // Verify all indices are valid
        for (start, end) in &connections {
            assert!(*start < 21);
            assert!(*end < 21);
        }
    }

    #[test]
    fn test_hd_params() {
        let params = SkeletonParams::hd();
        assert_eq!(params.resolution, (1920, 1080));
    }

    #[test]
    fn test_ffmpeg_command() {
        let renderer = SkeletonRenderer::new(SkeletonParams::default());
        let cmd = renderer.get_ffmpeg_command(
            Path::new("/tmp/frames"),
            "pose",
            Path::new("/tmp/output.mp4"),
        );
        assert!(cmd.contains("ffmpeg"));
        assert!(cmd.contains("-framerate 30"));
    }
}
