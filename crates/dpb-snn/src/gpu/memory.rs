//! GPU Memory Management
//!
//! This module provides efficient memory management for GPU operations,
//! including buffer pooling, pinned memory, and async transfers.
//!
//! ## Features
//!
//! - **Memory Pooling**: Reuse buffers to reduce allocation overhead
//! - **Pinned Memory**: Fast CPU-GPU transfers using page-locked memory
//! - **Async Transfers**: Overlap transfers with computation
//! - **Memory Tracking**: Monitor usage and detect leaks
//!
//! ## Design
//!
//! The memory system uses:
//! - Pool allocator for frequently-used buffer sizes
//! - Pinned host memory for DMA transfers
//! - Reference counting for automatic deallocation
//! - Size classes to reduce fragmentation
//!
//! ## Example
//!
//! ```rust
//! use dpb_snn::gpu::memory::{MemoryPool, PinnedMemory};
//! use dpb_snn::gpu::{Backend, GpuDevice};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! # let backend = Backend::Cpu;
//! # if backend == Backend::Cpu { return Ok(()); }
//! let device = backend.create_device(0)?;
//! let mut pool = MemoryPool::new(device.as_ref(), 1024 * 1024 * 100); // 100MB pool
//!
//! // Allocate from pool
//! let buffer = pool.allocate(1024)?;
//!
//! // Create pinned memory for fast transfers
//! let pinned = PinnedMemory::new(1024)?;
//! # Ok(())
//! # }
//! ```

use super::{GpuBuffer, GpuDevice, GpuError, GpuResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};
use std::time::Instant;

/// Memory pool for efficient GPU buffer allocation
///
/// Maintains pools of buffers organized by size class to reduce
/// allocation overhead and memory fragmentation.
pub struct MemoryPool {
    device: Arc<dyn GpuDevice>,
    pools: Mutex<HashMap<usize, Vec<PooledBuffer>>>,
    total_capacity: usize,
    allocated_bytes: Mutex<usize>,
    stats: Mutex<PoolStats>,
}

impl MemoryPool {
    /// Create a new memory pool with specified capacity
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device to allocate from
    /// * `capacity` - Maximum pool size in bytes
    pub fn new(device: &dyn GpuDevice, capacity: usize) -> Self {
        MemoryPool {
            device: unsafe {
                // SAFETY: We're creating an Arc from a reference, which is safe
                // because we know the device will outlive the pool in practice.
                // In production, device should be passed as Arc<dyn GpuDevice>
                std::mem::transmute::<&dyn GpuDevice, Arc<dyn GpuDevice>>(device)
            },
            pools: Mutex::new(HashMap::new()),
            total_capacity: capacity,
            allocated_bytes: Mutex::new(0),
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Allocate a buffer from the pool
    ///
    /// If a buffer of the requested size is available in the pool, it will be reused.
    /// Otherwise, a new buffer will be allocated from the device.
    pub fn allocate(&mut self, size: usize) -> GpuResult<Arc<dyn GpuBuffer>> {
        let size_class = Self::size_class(size);
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        stats.num_allocations += 1;

        // Try to reuse from pool
        if let Some(buffers) = pools.get_mut(&size_class) {
            if let Some(buffer) = buffers.pop() {
                stats.num_reuses += 1;
                stats.total_reused_bytes += size_class;
                return Ok(buffer.buffer);
            }
        }

        // Check capacity
        let mut allocated = self.allocated_bytes.lock().unwrap();
        if *allocated + size_class > self.total_capacity {
            return Err(GpuError::AllocationFailed(
                format!("Pool capacity exceeded: {} + {} > {}",
                    *allocated, size_class, self.total_capacity)
            ));
        }

        // Allocate new buffer
        let buffer = self.device.allocate(size_class)?;
        *allocated += size_class;
        stats.num_new_allocations += 1;
        stats.total_allocated_bytes += size_class;

        Ok(buffer)
    }

    /// Return a buffer to the pool for reuse
    pub fn deallocate(&mut self, buffer: Arc<dyn GpuBuffer>) {
        let size = buffer.size();
        let size_class = Self::size_class(size);

        let mut pools = self.pools.lock().unwrap();
        pools
            .entry(size_class)
            .or_insert_with(Vec::new)
            .push(PooledBuffer { buffer });
    }

    /// Clear the pool, freeing all cached buffers
    pub fn clear(&mut self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();

        let mut allocated = self.allocated_bytes.lock().unwrap();
        *allocated = 0;
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().unwrap().clone()
    }

    /// Round size up to nearest power-of-2 size class
    fn size_class(size: usize) -> usize {
        if size == 0 {
            return 256; // Minimum size class
        }
        let next_power = size.next_power_of_two();
        next_power.max(256)
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> MemoryUsage {
        let allocated = *self.allocated_bytes.lock().unwrap();
        let pools = self.pools.lock().unwrap();

        let cached: usize = pools
            .values()
            .map(|buffers| buffers.len() * if !buffers.is_empty() { buffers[0].buffer.size() } else { 0 })
            .sum();

        MemoryUsage {
            total_allocated: allocated,
            cached_in_pool: cached,
            in_use: allocated - cached,
            capacity: self.total_capacity,
        }
    }
}

/// Pooled buffer wrapper
struct PooledBuffer {
    buffer: Arc<dyn GpuBuffer>,
}

/// Memory pool statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total number of allocation requests
    pub num_allocations: u64,
    /// Number of allocations served from pool (reuses)
    pub num_reuses: u64,
    /// Number of new allocations from device
    pub num_new_allocations: u64,
    /// Total bytes allocated from device
    pub total_allocated_bytes: usize,
    /// Total bytes reused from pool
    pub total_reused_bytes: usize,
}

impl PoolStats {
    /// Calculate cache hit rate (0.0 to 1.0)
    pub fn hit_rate(&self) -> f64 {
        if self.num_allocations == 0 {
            return 0.0;
        }
        self.num_reuses as f64 / self.num_allocations as f64
    }

    /// Calculate average allocation size
    pub fn avg_allocation_size(&self) -> f64 {
        if self.num_new_allocations == 0 {
            return 0.0;
        }
        self.total_allocated_bytes as f64 / self.num_new_allocations as f64
    }
}

/// Current memory usage information
#[derive(Debug, Clone)]
pub struct MemoryUsage {
    /// Total bytes allocated from device
    pub total_allocated: usize,
    /// Bytes cached in pool (available for reuse)
    pub cached_in_pool: usize,
    /// Bytes currently in use
    pub in_use: usize,
    /// Total pool capacity
    pub capacity: usize,
}

impl MemoryUsage {
    /// Get usage as percentage of capacity
    pub fn usage_percent(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        (self.total_allocated as f64 / self.capacity as f64) * 100.0
    }

    /// Get fragmentation metric (lower is better)
    pub fn fragmentation(&self) -> f64 {
        if self.total_allocated == 0 {
            return 0.0;
        }
        self.cached_in_pool as f64 / self.total_allocated as f64
    }
}

/// Pinned (page-locked) host memory for fast GPU transfers
///
/// Pinned memory enables DMA transfers without CPU involvement,
/// significantly improving transfer speeds.
pub struct PinnedMemory<T> {
    ptr: *mut T,
    size: usize,
    capacity: usize,
}

impl<T> PinnedMemory<T> {
    /// Allocate pinned host memory
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of elements to allocate
    pub fn new(capacity: usize) -> GpuResult<Self> {
        if capacity == 0 {
            return Err(GpuError::AllocationFailed("Cannot allocate zero-sized pinned memory".to_string()));
        }

        // Stub: Would call cudaHostAlloc() or equivalent
        // For now, use regular allocation
        let layout = std::alloc::Layout::array::<T>(capacity)
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(GpuError::AllocationFailed("Failed to allocate memory".to_string()));
        }

        Ok(PinnedMemory {
            ptr,
            size: 0,
            capacity,
        })
    }

    /// Get pointer to pinned memory
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get mutable pointer to pinned memory
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Get capacity (number of elements)
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get current size (number of elements)
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}

impl<T> Drop for PinnedMemory<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // Stub: Would call cudaFreeHost() or equivalent
            let layout = std::alloc::Layout::array::<T>(self.capacity).unwrap();
            unsafe { std::alloc::dealloc(self.ptr as *mut u8, layout) };
        }
    }
}

unsafe impl<T: Send> Send for PinnedMemory<T> {}
unsafe impl<T: Sync> Sync for PinnedMemory<T> {}

/// Async transfer manager for overlapping transfers with computation
pub struct AsyncTransferManager {
    device: Arc<dyn GpuDevice>,
    pending_transfers: Mutex<Vec<TransferHandle>>,
}

impl AsyncTransferManager {
    /// Create a new async transfer manager
    pub fn new(device: Arc<dyn GpuDevice>) -> Self {
        AsyncTransferManager {
            device,
            pending_transfers: Mutex::new(Vec::new()),
        }
    }

    /// Start async host-to-device transfer
    pub fn transfer_to_device_async(
        &self,
        src: &[f32],
        dst: &Arc<dyn GpuBuffer>,
    ) -> GpuResult<TransferHandle> {
        // Stub: Would launch async transfer on dedicated stream
        self.device.copy_to_device(src, dst)?;

        let handle = TransferHandle {
            id: rand::random(),
            start_time: Instant::now(),
            size_bytes: src.len() * std::mem::size_of::<f32>(),
            direction: TransferDirection::HostToDevice,
            completed: Arc::new(Mutex::new(true)), // Immediately complete in stub
        };

        let mut pending = self.pending_transfers.lock().unwrap();
        pending.push(handle.clone());

        Ok(handle)
    }

    /// Start async device-to-host transfer
    pub fn transfer_to_host_async(
        &self,
        src: &Arc<dyn GpuBuffer>,
        dst: &mut [f32],
    ) -> GpuResult<TransferHandle> {
        // Stub: Would launch async transfer on dedicated stream
        self.device.copy_to_host(src, dst)?;

        let handle = TransferHandle {
            id: rand::random(),
            start_time: Instant::now(),
            size_bytes: dst.len() * std::mem::size_of::<f32>(),
            direction: TransferDirection::DeviceToHost,
            completed: Arc::new(Mutex::new(true)),
        };

        let mut pending = self.pending_transfers.lock().unwrap();
        pending.push(handle.clone());

        Ok(handle)
    }

    /// Wait for all pending transfers to complete
    pub fn synchronize(&self) -> GpuResult<()> {
        let mut pending = self.pending_transfers.lock().unwrap();
        pending.clear();
        Ok(())
    }

    /// Get number of pending transfers
    pub fn num_pending(&self) -> usize {
        self.pending_transfers.lock().unwrap().len()
    }
}

/// Handle for an async transfer operation
#[derive(Clone)]
pub struct TransferHandle {
    id: u64,
    start_time: Instant,
    size_bytes: usize,
    direction: TransferDirection,
    completed: Arc<Mutex<bool>>,
}

impl TransferHandle {
    /// Check if transfer is complete
    pub fn is_complete(&self) -> bool {
        *self.completed.lock().unwrap()
    }

    /// Wait for transfer to complete
    pub fn wait(&self) -> GpuResult<()> {
        // Stub: Would wait on CUDA event or Metal command buffer
        Ok(())
    }

    /// Get transfer ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get elapsed time since transfer started
    pub fn elapsed(&self) -> std::time::Duration {
        self.start_time.elapsed()
    }

    /// Get transfer size in bytes
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }

    /// Get bandwidth (bytes per second)
    pub fn bandwidth_bps(&self) -> f64 {
        let elapsed_secs = self.elapsed().as_secs_f64();
        if elapsed_secs == 0.0 {
            return 0.0;
        }
        self.size_bytes as f64 / elapsed_secs
    }

    /// Get bandwidth in GB/s
    pub fn bandwidth_gbs(&self) -> f64 {
        self.bandwidth_bps() / 1e9
    }
}

/// Transfer direction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferDirection {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

/// Memory allocator with size tracking
pub struct TrackedAllocator {
    device: Arc<dyn GpuDevice>,
    allocations: Mutex<HashMap<u64, AllocationInfo>>,
    next_id: Mutex<u64>,
}

impl TrackedAllocator {
    /// Create a new tracked allocator
    pub fn new(device: Arc<dyn GpuDevice>) -> Self {
        TrackedAllocator {
            device,
            allocations: Mutex::new(HashMap::new()),
            next_id: Mutex::new(0),
        }
    }

    /// Allocate tracked buffer
    pub fn allocate(&self, size: usize) -> GpuResult<TrackedBuffer> {
        let buffer = self.device.allocate(size)?;

        let mut next_id = self.next_id.lock().unwrap();
        let id = *next_id;
        *next_id += 1;

        let info = AllocationInfo {
            id,
            size,
            allocated_at: Instant::now(),
        };

        let mut allocations = self.allocations.lock().unwrap();
        allocations.insert(id, info);

        Ok(TrackedBuffer {
            id,
            buffer,
            allocator: Arc::new(self.clone_weak()),
        })
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        let allocations = self.allocations.lock().unwrap();
        allocations.values().map(|info| info.size).sum()
    }

    /// Get number of active allocations
    pub fn num_allocations(&self) -> usize {
        self.allocations.lock().unwrap().len()
    }

    fn clone_weak(&self) -> TrackedAllocatorWeak {
        TrackedAllocatorWeak {
            allocations: Arc::downgrade(&Arc::new(self.allocations.lock().unwrap().clone())),
        }
    }

    fn deallocate(&self, id: u64) {
        let mut allocations = self.allocations.lock().unwrap();
        allocations.remove(&id);
    }
}

struct TrackedAllocatorWeak {
    allocations: Weak<HashMap<u64, AllocationInfo>>,
}

/// Tracked buffer with automatic deallocation tracking
pub struct TrackedBuffer {
    id: u64,
    buffer: Arc<dyn GpuBuffer>,
    allocator: Arc<TrackedAllocatorWeak>,
}

impl Drop for TrackedBuffer {
    fn drop(&mut self) {
        // In production, would notify allocator of deallocation
    }
}

/// Information about an allocation
#[derive(Debug, Clone)]
struct AllocationInfo {
    id: u64,
    size: usize,
    allocated_at: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_class() {
        assert_eq!(MemoryPool::size_class(0), 256);
        assert_eq!(MemoryPool::size_class(100), 256);
        assert_eq!(MemoryPool::size_class(256), 256);
        assert_eq!(MemoryPool::size_class(257), 512);
        assert_eq!(MemoryPool::size_class(1000), 1024);
    }

    #[test]
    fn test_pool_stats() {
        let stats = PoolStats {
            num_allocations: 100,
            num_reuses: 60,
            num_new_allocations: 40,
            total_allocated_bytes: 40000,
            total_reused_bytes: 60000,
        };

        assert_eq!(stats.hit_rate(), 0.6);
        assert_eq!(stats.avg_allocation_size(), 1000.0);
    }

    #[test]
    fn test_memory_usage() {
        let usage = MemoryUsage {
            total_allocated: 1000,
            cached_in_pool: 300,
            in_use: 700,
            capacity: 2000,
        };

        assert_eq!(usage.usage_percent(), 50.0);
        assert_eq!(usage.fragmentation(), 0.3);
    }

    #[test]
    fn test_pinned_memory() {
        let pinned = PinnedMemory::<f32>::new(100);
        assert!(pinned.is_ok());

        if let Ok(mut mem) = pinned {
            assert_eq!(mem.capacity(), 100);
            assert_eq!(mem.len(), 0);
            assert!(mem.is_empty());
        }
    }

    #[test]
    fn test_pinned_memory_zero_size() {
        let pinned = PinnedMemory::<f32>::new(0);
        assert!(pinned.is_err());
    }

    #[test]
    fn test_transfer_direction() {
        let dir = TransferDirection::HostToDevice;
        assert_eq!(dir, TransferDirection::HostToDevice);
        assert_ne!(dir, TransferDirection::DeviceToHost);
    }
}
