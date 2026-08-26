//! GPU buffer management utilities.

use crate::error::{DpbError, Result};
use bytemuck::{Pod, Zeroable};
use std::marker::PhantomData;
use wgpu;
use wgpu::util::DeviceExt;

/// A typed GPU buffer wrapper.
pub struct GpuBuffer<T: Pod> {
    /// Underlying wgpu buffer
    buffer: wgpu::Buffer,
    /// Number of elements
    len: usize,
    /// Phantom data for type safety
    _phantom: PhantomData<T>,
}

impl<T: Pod> GpuBuffer<T> {
    /// Creates a new GPU buffer from data.
    pub fn from_slice(device: &wgpu::Device, data: &[T], usage: wgpu::BufferUsages) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GPU Buffer"),
            contents: bytemuck::cast_slice(data),
            usage,
        });

        Self {
            buffer,
            len: data.len(),
            _phantom: PhantomData,
        }
    }

    /// Creates a new uninitialized GPU buffer.
    pub fn new(device: &wgpu::Device, len: usize, usage: wgpu::BufferUsages) -> Self {
        let size = (len * std::mem::size_of::<T>()) as u64;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GPU Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            len,
            _phantom: PhantomData,
        }
    }

    /// Creates a staging buffer for reading back data.
    pub fn staging(device: &wgpu::Device, len: usize) -> Self {
        Self::new(
            device,
            len,
            wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        )
    }

    /// Returns the number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the buffer size in bytes.
    pub fn size(&self) -> u64 {
        (self.len * std::mem::size_of::<T>()) as u64
    }

    /// Returns a reference to the underlying buffer.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Writes data to the buffer.
    pub fn write(&self, queue: &wgpu::Queue, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(DpbError::InvalidDimensions(
                format!("Data length {} does not match buffer length {}", data.len(), self.len),
            ));
        }
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
        Ok(())
    }

    /// Reads data from the buffer (requires MAP_READ usage).
    pub async fn read(&self, device: &wgpu::Device) -> Result<Vec<T>> {
        let buffer_slice = self.buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .map_err(|e| DpbError::Gpu(format!("Device poll failed: {e}")))?;

        receiver
            .await
            .map_err(|_| DpbError::Gpu("Failed to receive map result".to_string()))?
            .map_err(|e| DpbError::Gpu(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice
            .get_mapped_range()
            .map_err(|e| DpbError::Gpu(format!("Failed to read mapped range: {e}")))?;
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        self.buffer.unmap();

        Ok(result)
    }

    /// Copies this buffer to another buffer.
    pub fn copy_to(&self, encoder: &mut wgpu::CommandEncoder, dst: &GpuBuffer<T>) -> Result<()> {
        if self.len != dst.len {
            return Err(DpbError::InvalidDimensions(
                "Source and destination buffers must have the same length".to_string(),
            ));
        }

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &dst.buffer, 0, self.size());
        Ok(())
    }
}

/// Creates a uniform buffer from data.
pub fn create_uniform_buffer<T: Pod>(device: &wgpu::Device, data: &T) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Uniform Buffer"),
        contents: bytemuck::bytes_of(data),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    })
}

/// Creates a storage buffer with data.
pub fn create_storage_buffer<T: Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC,
    })
}

/// Utility for managing multiple buffers.
pub struct BufferPool {
    device: wgpu::Device,
    buffers: Vec<wgpu::Buffer>,
}

impl BufferPool {
    /// Creates a new buffer pool.
    pub fn new(device: wgpu::Device) -> Self {
        Self {
            device,
            buffers: Vec::new(),
        }
    }

    /// Allocates a new buffer.
    pub fn allocate(&mut self, size: u64, usage: wgpu::BufferUsages) -> usize {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Pooled Buffer"),
            size,
            usage,
            mapped_at_creation: false,
        });

        self.buffers.push(buffer);
        self.buffers.len() - 1
    }

    /// Gets a buffer by index.
    pub fn get(&self, index: usize) -> Option<&wgpu::Buffer> {
        self.buffers.get(index)
    }

    /// Clears all buffers.
    pub fn clear(&mut self) {
        self.buffers.clear();
    }

    /// Returns the number of buffers.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// Returns true if the pool is empty.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }
}

/// Represents GPU-accessible data aligned for shader use.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GpuVec4 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl GpuVec4 {
    pub fn new(x: f32, y: f32, z: f32, w: f32) -> Self {
        Self { x, y, z, w }
    }

    pub fn splat(value: f32) -> Self {
        Self::new(value, value, value, value)
    }
}

impl From<[f32; 4]> for GpuVec4 {
    fn from(arr: [f32; 4]) -> Self {
        Self::new(arr[0], arr[1], arr[2], arr[3])
    }
}

impl From<glam::Vec4> for GpuVec4 {
    fn from(v: glam::Vec4) -> Self {
        Self::new(v.x, v.y, v.z, v.w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C)]
    #[derive(Copy, Clone, Pod, Zeroable)]
    struct TestData {
        value: f32,
    }

    // Note: These tests require a GPU context, which may not be available in all environments
    // They serve as documentation of the API

    #[test]
    fn test_gpu_vec4() {
        let v = GpuVec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(v.x, 1.0);
        assert_eq!(v.y, 2.0);

        let v2 = GpuVec4::splat(5.0);
        assert_eq!(v2.x, 5.0);
        assert_eq!(v2.w, 5.0);
    }

    #[test]
    fn test_gpu_vec4_from_array() {
        let v: GpuVec4 = [1.0, 2.0, 3.0, 4.0].into();
        assert_eq!(v.x, 1.0);
        assert_eq!(v.w, 4.0);
    }
}
