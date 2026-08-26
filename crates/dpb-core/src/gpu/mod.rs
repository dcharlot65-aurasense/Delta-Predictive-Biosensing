//! GPU infrastructure for accelerated neuromorphic processing.
//!
//! This module provides both high-level and low-level GPU abstractions:
//!
//! - [`backend`]: Cross-platform compute backend abstraction (WebGPU, CUDA, Metal)
//! - [`context`]: WebGPU-specific context management
//! - [`buffer`]: GPU buffer utilities
//!
//! # Backend Selection
//!
//! Use the [`backend`] module for cross-platform code:
//!
//! ```rust,ignore
//! use dpb_core::gpu::backend::*;
//!
//! // Auto-select best backend (CUDA > Metal > WebGPU)
//! let backend = create_default_backend().await?;
//! ```
//!
//! Or use [`context::GpuContext`] for direct WebGPU access:
//!
//! ```rust,ignore
//! use dpb_core::gpu::GpuContext;
//!
//! let ctx = GpuContext::new_default().await?;
//! ```

pub mod backend;
pub mod buffer;
pub mod context;

// Re-export backend types
pub use backend::{
    available_backends, create_backend, create_default_backend, BackendType, BufferHandle,
    ComputeBackend, DeviceProperties, KernelHandle, WebGPUBackend,
    // Helper functions for typed buffer operations
    create_buffer_from_slice, upload_buffer, download_buffer,
};

#[cfg(feature = "cuda")]
pub use backend::cuda::CUDABackend;

// Re-export existing types
pub use buffer::{create_storage_buffer, create_uniform_buffer, BufferPool, GpuBuffer, GpuVec4};
pub use context::GpuContext;

use crate::error::{DpbError, Result};
use wgpu;

/// Shader compilation utilities.
pub struct ShaderCompiler;

impl ShaderCompiler {
    /// Compiles WGSL shader source.
    pub fn compile_wgsl(
        device: &wgpu::Device,
        source: &str,
        label: Option<&str>,
    ) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label,
            source: wgpu::ShaderSource::Wgsl(source.into()),
        })
    }

    /// Validates shader compilation.
    pub fn validate_shader(source: &str) -> Result<()> {
        // Basic validation - check for common syntax errors
        if source.is_empty() {
            return Err(DpbError::Gpu("Shader source is empty".to_string()));
        }

        // Check for required entry point patterns
        if !source.contains("@compute") && !source.contains("@vertex") && !source.contains("@fragment") {
            return Err(DpbError::Gpu(
                "Shader must contain at least one entry point".to_string(),
            ));
        }

        Ok(())
    }
}

/// Compute pipeline builder for easier pipeline creation.
pub struct ComputePipelineBuilder {
    shader_source: String,
    entry_point: String,
    bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl ComputePipelineBuilder {
    /// Creates a new pipeline builder.
    pub fn new() -> Self {
        Self {
            shader_source: String::new(),
            entry_point: "main".to_string(),
            bind_group_layouts: Vec::new(),
        }
    }

    /// Sets the shader source.
    pub fn shader(mut self, source: impl Into<String>) -> Self {
        self.shader_source = source.into();
        self
    }

    /// Sets the entry point.
    pub fn entry_point(mut self, entry_point: impl Into<String>) -> Self {
        self.entry_point = entry_point.into();
        self
    }

    /// Adds a bind group layout.
    pub fn bind_group_layout(mut self, layout: wgpu::BindGroupLayout) -> Self {
        self.bind_group_layouts.push(layout);
        self
    }

    /// Builds the compute pipeline.
    pub fn build(self, device: &wgpu::Device) -> Result<wgpu::ComputePipeline> {
        ShaderCompiler::validate_shader(&self.shader_source)?;

        let shader = ShaderCompiler::compile_wgsl(device, &self.shader_source, Some("Compute Shader"));

        let layout_refs: Vec<_> = self.bind_group_layouts.iter().map(Some).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &layout_refs,
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(&self.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok(pipeline)
    }
}

impl Default for ComputePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper for creating common bind group layouts.
pub struct BindGroupLayoutBuilder {
    entries: Vec<wgpu::BindGroupLayoutEntry>,
}

impl BindGroupLayoutBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a storage buffer binding.
    pub fn storage_buffer(mut self, binding: u32, read_only: bool) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self
    }

    /// Adds a uniform buffer binding.
    pub fn uniform_buffer(mut self, binding: u32) -> Self {
        self.entries.push(wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        self
    }

    /// Builds the bind group layout.
    pub fn build(self, device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &self.entries,
        })
    }
}

impl Default for BindGroupLayoutBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility for workgroup size calculations.
pub fn calculate_workgroups(total_elements: u32, workgroup_size: u32) -> u32 {
    (total_elements + workgroup_size - 1) / workgroup_size
}

/// Calculates 2D workgroup dimensions.
pub fn calculate_workgroups_2d(
    width: u32,
    height: u32,
    workgroup_x: u32,
    workgroup_y: u32,
) -> (u32, u32) {
    (
        calculate_workgroups(width, workgroup_x),
        calculate_workgroups(height, workgroup_y),
    )
}

/// Calculates 3D workgroup dimensions.
pub fn calculate_workgroups_3d(
    width: u32,
    height: u32,
    depth: u32,
    workgroup_x: u32,
    workgroup_y: u32,
    workgroup_z: u32,
) -> (u32, u32, u32) {
    (
        calculate_workgroups(width, workgroup_x),
        calculate_workgroups(height, workgroup_y),
        calculate_workgroups(depth, workgroup_z),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_workgroups() {
        assert_eq!(calculate_workgroups(100, 64), 2);
        assert_eq!(calculate_workgroups(64, 64), 1);
        assert_eq!(calculate_workgroups(65, 64), 2);
        assert_eq!(calculate_workgroups(128, 64), 2);
    }

    #[test]
    fn test_calculate_workgroups_2d() {
        let (x, y) = calculate_workgroups_2d(100, 100, 8, 8);
        assert_eq!(x, 13);
        assert_eq!(y, 13);
    }

    #[test]
    fn test_shader_validation() {
        let valid_shader = "@compute @workgroup_size(64) fn main() {}";
        assert!(ShaderCompiler::validate_shader(valid_shader).is_ok());

        let invalid_shader = "";
        assert!(ShaderCompiler::validate_shader(invalid_shader).is_err());
    }

    #[test]
    fn test_bind_group_layout_builder() {
        // This test just ensures the API compiles correctly
        let _builder = BindGroupLayoutBuilder::new()
            .storage_buffer(0, true)
            .uniform_buffer(1);
    }

    #[test]
    fn test_compute_pipeline_builder() {
        let _builder = ComputePipelineBuilder::new()
            .shader("@compute @workgroup_size(1) fn main() {}")
            .entry_point("main");
    }
}
