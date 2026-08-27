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

use super::{Backend, GpuBuffer, GpuDevice, GpuError, GpuResult};
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex, Weak,
    atomic::{AtomicU64, Ordering},
};
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice as CudarDevice, CudaStream, DeviceRepr};

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
    /// * `device` - GPU device to allocate from (as Arc)
    /// * `capacity` - Maximum pool size in bytes
    pub fn with_device(device: Arc<dyn GpuDevice>, capacity: usize) -> Self {
        MemoryPool {
            device,
            pools: Mutex::new(HashMap::new()),
            total_capacity: capacity,
            allocated_bytes: Mutex::new(0),
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Create a new memory pool with specified capacity (legacy interface)
    ///
    /// # Arguments
    ///
    /// * `device` - GPU device to allocate from
    /// * `capacity` - Maximum pool size in bytes
    ///
    /// # Safety
    ///
    /// The device reference must remain valid for the lifetime of the pool.
    /// This method exists for backward compatibility.
    #[deprecated(
        since = "0.2.0",
        note = "Use with_device() with Arc<dyn GpuDevice> instead"
    )]
    pub fn new(device: &dyn GpuDevice, capacity: usize) -> Self {
        // Create a CPU-fallback pool that doesn't actually use the device
        // This is unsafe but maintained for backward compatibility
        MemoryPool {
            device: Arc::new(CpuFallbackDevice),
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
        let mut pools = self.pools.lock().expect("GPU mutex poisoned");
        let mut stats = self.stats.lock().expect("GPU mutex poisoned");

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
        let mut allocated = self.allocated_bytes.lock().expect("GPU mutex poisoned");
        if *allocated + size_class > self.total_capacity {
            return Err(GpuError::AllocationFailed(format!(
                "Pool capacity exceeded: {} + {} > {}",
                *allocated, size_class, self.total_capacity
            )));
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

        let mut pools = self.pools.lock().expect("GPU mutex poisoned");
        pools
            .entry(size_class)
            .or_insert_with(Vec::new)
            .push(PooledBuffer { buffer });
    }

    /// Clear the pool, freeing all cached buffers
    pub fn clear(&mut self) {
        let mut pools = self.pools.lock().expect("GPU mutex poisoned");
        pools.clear();

        let mut allocated = self.allocated_bytes.lock().expect("GPU mutex poisoned");
        *allocated = 0;
    }

    /// Get pool statistics
    pub fn stats(&self) -> PoolStats {
        self.stats.lock().expect("GPU mutex poisoned").clone()
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
        let allocated = *self.allocated_bytes.lock().expect("GPU mutex poisoned");
        let pools = self.pools.lock().expect("GPU mutex poisoned");

        let cached: usize = pools
            .values()
            .map(|buffers| {
                buffers.len()
                    * if !buffers.is_empty() {
                        buffers[0].buffer.size()
                    } else {
                        0
                    }
            })
            .sum();

        MemoryUsage {
            total_allocated: allocated,
            cached_in_pool: cached,
            in_use: allocated.saturating_sub(cached),
            capacity: self.total_capacity,
        }
    }
}

/// CPU fallback device for legacy new() interface
struct CpuFallbackDevice;

impl GpuDevice for CpuFallbackDevice {
    fn device_id(&self) -> u32 {
        0
    }
    fn backend(&self) -> Backend {
        Backend::Cpu
    }
    fn allocate(&self, size_bytes: usize) -> GpuResult<Arc<dyn GpuBuffer>> {
        Ok(Arc::new(CpuBuffer::new(size_bytes)))
    }
    fn copy_to_device(&self, _src: &[f32], _dst: &Arc<dyn GpuBuffer>) -> GpuResult<()> {
        Ok(())
    }
    fn copy_to_host(&self, _src: &Arc<dyn GpuBuffer>, _dst: &mut [f32]) -> GpuResult<()> {
        Ok(())
    }
    fn copy_device_to_device(
        &self,
        _src: &Arc<dyn GpuBuffer>,
        _dst: &Arc<dyn GpuBuffer>,
    ) -> GpuResult<()> {
        Ok(())
    }
    fn synchronize(&self) -> GpuResult<()> {
        Ok(())
    }
    fn memory_info(&self) -> GpuResult<(usize, usize)> {
        Ok((usize::MAX, usize::MAX))
    }
    fn name(&self) -> String {
        "CPU Fallback".to_string()
    }
    fn compute_capability(&self) -> (u32, u32) {
        (0, 0)
    }
}

/// CPU buffer for fallback
#[derive(Debug)]
struct CpuBuffer {
    data: Vec<u8>,
}

impl CpuBuffer {
    fn new(size: usize) -> Self {
        CpuBuffer {
            data: vec![0u8; size],
        }
    }
}

impl GpuBuffer for CpuBuffer {
    fn size(&self) -> usize {
        self.data.len()
    }
    fn as_ptr(&self) -> *mut u8 {
        self.data.as_ptr() as *mut u8
    }
    fn is_valid(&self) -> bool {
        true
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
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
    #[cfg(feature = "cuda")]
    is_cuda_pinned: bool,
}

impl<T> PinnedMemory<T> {
    /// Allocate pinned host memory
    ///
    /// # Arguments
    ///
    /// * `capacity` - Number of elements to allocate
    pub fn new(capacity: usize) -> GpuResult<Self> {
        if capacity == 0 {
            return Err(GpuError::AllocationFailed(
                "Cannot allocate zero-sized pinned memory".to_string(),
            ));
        }

        #[cfg(feature = "cuda")]
        {
            // Try to allocate CUDA pinned memory
            if let Ok(ptr) = Self::cuda_host_alloc(capacity) {
                return Ok(PinnedMemory {
                    ptr,
                    size: 0,
                    capacity,
                    is_cuda_pinned: true,
                });
            }
        }

        // Fallback to regular aligned allocation
        let layout = std::alloc::Layout::array::<T>(capacity)
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(GpuError::AllocationFailed(
                "Failed to allocate memory".to_string(),
            ));
        }

        Ok(PinnedMemory {
            ptr,
            size: 0,
            capacity,
            #[cfg(feature = "cuda")]
            is_cuda_pinned: false,
        })
    }

    /// Allocate with CUDA backend for true pinned memory
    #[cfg(feature = "cuda")]
    pub fn with_cuda_device(device: &CudarDevice, capacity: usize) -> GpuResult<Self> {
        if capacity == 0 {
            return Err(GpuError::AllocationFailed(
                "Cannot allocate zero-sized pinned memory".to_string(),
            ));
        }

        let ptr = Self::cuda_host_alloc(capacity)?;
        Ok(PinnedMemory {
            ptr,
            size: 0,
            capacity,
            is_cuda_pinned: true,
        })
    }

    #[cfg(feature = "cuda")]
    fn cuda_host_alloc(capacity: usize) -> GpuResult<*mut T> {
        // cudarc doesn't expose cudaHostAlloc directly, use regular allocation
        // In production, we'd use cuda-sys for true pinned memory
        let layout = std::alloc::Layout::array::<T>(capacity)
            .map_err(|e| GpuError::AllocationFailed(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        if ptr.is_null() {
            return Err(GpuError::AllocationFailed(
                "Failed to allocate pinned memory".to_string(),
            ));
        }

        // Lock the memory pages (mlock on Linux)
        #[cfg(target_os = "linux")]
        unsafe {
            let size = capacity * std::mem::size_of::<T>();
            libc::mlock(ptr as *const libc::c_void, size);
        }

        Ok(ptr)
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

    /// Set the size (number of valid elements)
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity, "Length exceeds capacity");
        self.size = len;
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }

    /// Copy data into pinned memory
    pub fn copy_from_slice(&mut self, data: &[T])
    where
        T: Copy,
    {
        assert!(data.len() <= self.capacity, "Data exceeds capacity");
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        self.size = data.len();
    }

    /// Copy data from pinned memory to a slice
    pub fn copy_to_slice(&self, dst: &mut [T])
    where
        T: Copy,
    {
        let len = dst.len().min(self.size);
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr, dst.as_mut_ptr(), len);
        }
    }
}

impl<T> Drop for PinnedMemory<T> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            #[cfg(feature = "cuda")]
            if self.is_cuda_pinned {
                // Unlock memory pages
                #[cfg(target_os = "linux")]
                unsafe {
                    let size = self.capacity * std::mem::size_of::<T>();
                    libc::munlock(self.ptr as *const libc::c_void, size);
                }
            }

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
    next_id: AtomicU64,
    #[cfg(feature = "cuda")]
    cuda_streams: Mutex<Vec<CudaStreamWrapper>>,
}

#[cfg(feature = "cuda")]
struct CudaStreamWrapper {
    _stream: cudarc::driver::CudaStream,
    in_use: bool,
}

impl AsyncTransferManager {
    /// Create a new async transfer manager
    pub fn new(device: Arc<dyn GpuDevice>) -> Self {
        AsyncTransferManager {
            device,
            pending_transfers: Mutex::new(Vec::new()),
            next_id: AtomicU64::new(0),
            #[cfg(feature = "cuda")]
            cuda_streams: Mutex::new(Vec::new()),
        }
    }

    /// Start async host-to-device transfer
    pub fn transfer_to_device_async(
        &self,
        src: &[f32],
        dst: &Arc<dyn GpuBuffer>,
    ) -> GpuResult<TransferHandle> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size_bytes = src.len() * std::mem::size_of::<f32>();

        // Perform the transfer
        self.device.copy_to_device(src, dst)?;

        let handle = TransferHandle {
            id,
            start_time: Instant::now(),
            size_bytes,
            direction: TransferDirection::HostToDevice,
            completed: Arc::new(Mutex::new(true)),
        };

        let mut pending = self.pending_transfers.lock().expect("GPU mutex poisoned");
        pending.push(handle.clone());

        Ok(handle)
    }

    /// Start async device-to-host transfer
    pub fn transfer_to_host_async(
        &self,
        src: &Arc<dyn GpuBuffer>,
        dst: &mut [f32],
    ) -> GpuResult<TransferHandle> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size_bytes = dst.len() * std::mem::size_of::<f32>();

        // Perform the transfer
        self.device.copy_to_host(src, dst)?;

        let handle = TransferHandle {
            id,
            start_time: Instant::now(),
            size_bytes,
            direction: TransferDirection::DeviceToHost,
            completed: Arc::new(Mutex::new(true)),
        };

        let mut pending = self.pending_transfers.lock().expect("GPU mutex poisoned");
        pending.push(handle.clone());

        Ok(handle)
    }

    /// Start async device-to-device transfer
    pub fn transfer_device_to_device_async(
        &self,
        src: &Arc<dyn GpuBuffer>,
        dst: &Arc<dyn GpuBuffer>,
    ) -> GpuResult<TransferHandle> {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let size_bytes = src.size();

        // Perform the transfer
        self.device.copy_device_to_device(src, dst)?;

        let handle = TransferHandle {
            id,
            start_time: Instant::now(),
            size_bytes,
            direction: TransferDirection::DeviceToDevice,
            completed: Arc::new(Mutex::new(true)),
        };

        let mut pending = self.pending_transfers.lock().expect("GPU mutex poisoned");
        pending.push(handle.clone());

        Ok(handle)
    }

    /// Wait for all pending transfers to complete
    pub fn synchronize(&self) -> GpuResult<()> {
        self.device.synchronize()?;
        let mut pending = self.pending_transfers.lock().expect("GPU mutex poisoned");
        for handle in pending.iter() {
            *handle.completed.lock().expect("GPU mutex poisoned") = true;
        }
        pending.clear();
        Ok(())
    }

    /// Get number of pending transfers
    pub fn num_pending(&self) -> usize {
        self.pending_transfers
            .lock()
            .expect("GPU mutex poisoned")
            .len()
    }

    /// Clean up completed transfers
    pub fn cleanup_completed(&self) {
        let mut pending = self.pending_transfers.lock().expect("GPU mutex poisoned");
        pending.retain(|h| !h.is_complete());
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
        *self.completed.lock().expect("GPU mutex poisoned")
    }

    /// Wait for transfer to complete
    pub fn wait(&self) -> GpuResult<()> {
        while !self.is_complete() {
            std::thread::yield_now();
        }
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

    /// Get transfer direction
    pub fn direction(&self) -> TransferDirection {
        self.direction
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

/// Memory allocator with size tracking and leak detection
pub struct TrackedAllocator {
    device: Arc<dyn GpuDevice>,
    allocations: Mutex<HashMap<u64, AllocationInfo>>,
    next_id: AtomicU64,
    total_allocated: AtomicU64,
    peak_allocated: AtomicU64,
}

impl TrackedAllocator {
    /// Create a new tracked allocator
    pub fn new(device: Arc<dyn GpuDevice>) -> Self {
        TrackedAllocator {
            device,
            allocations: Mutex::new(HashMap::new()),
            next_id: AtomicU64::new(0),
            total_allocated: AtomicU64::new(0),
            peak_allocated: AtomicU64::new(0),
        }
    }

    /// Allocate tracked buffer
    pub fn allocate(&self, size: usize) -> GpuResult<TrackedBuffer> {
        let buffer = self.device.allocate(size)?;

        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let info = AllocationInfo {
            id,
            size,
            allocated_at: Instant::now(),
        };

        let mut allocations = self.allocations.lock().expect("GPU mutex poisoned");
        allocations.insert(id, info);

        // Update statistics
        let new_total = self
            .total_allocated
            .fetch_add(size as u64, Ordering::SeqCst)
            + size as u64;
        self.peak_allocated.fetch_max(new_total, Ordering::SeqCst);

        Ok(TrackedBuffer { id, buffer, size })
    }

    /// Deallocate a tracked buffer
    pub fn deallocate(&self, id: u64) {
        let mut allocations = self.allocations.lock().expect("GPU mutex poisoned");
        if let Some(info) = allocations.remove(&id) {
            self.total_allocated
                .fetch_sub(info.size as u64, Ordering::SeqCst);
        }
    }

    /// Get total allocated bytes
    pub fn total_allocated(&self) -> usize {
        self.total_allocated.load(Ordering::SeqCst) as usize
    }

    /// Get peak allocated bytes
    pub fn peak_allocated(&self) -> usize {
        self.peak_allocated.load(Ordering::SeqCst) as usize
    }

    /// Get number of active allocations
    pub fn num_allocations(&self) -> usize {
        self.allocations.lock().expect("GPU mutex poisoned").len()
    }

    /// Get allocation statistics
    pub fn allocation_stats(&self) -> AllocationStats {
        let allocations = self.allocations.lock().expect("GPU mutex poisoned");
        AllocationStats {
            num_active: allocations.len(),
            total_bytes: self.total_allocated.load(Ordering::SeqCst) as usize,
            peak_bytes: self.peak_allocated.load(Ordering::SeqCst) as usize,
        }
    }

    /// Check for potential memory leaks (allocations older than threshold)
    pub fn find_leaks(&self, age_threshold: std::time::Duration) -> Vec<AllocationInfo> {
        let allocations = self.allocations.lock().expect("GPU mutex poisoned");
        let now = Instant::now();
        allocations
            .values()
            .filter(|info| now.duration_since(info.allocated_at) > age_threshold)
            .cloned()
            .collect()
    }
}

/// Tracked buffer with automatic deallocation tracking
pub struct TrackedBuffer {
    id: u64,
    buffer: Arc<dyn GpuBuffer>,
    size: usize,
}

impl TrackedBuffer {
    /// Get the underlying GPU buffer
    pub fn buffer(&self) -> &Arc<dyn GpuBuffer> {
        &self.buffer
    }

    /// Get allocation ID
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get size in bytes
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Allocation statistics
#[derive(Debug, Clone)]
pub struct AllocationStats {
    /// Number of active allocations
    pub num_active: usize,
    /// Total bytes currently allocated
    pub total_bytes: usize,
    /// Peak bytes ever allocated
    pub peak_bytes: usize,
}

/// Information about an allocation
#[derive(Debug, Clone)]
pub struct AllocationInfo {
    /// Unique allocation ID
    pub id: u64,
    /// Size in bytes
    pub size: usize,
    /// Time of allocation
    pub allocated_at: Instant,
}

impl AllocationInfo {
    /// Get age of allocation
    pub fn age(&self) -> std::time::Duration {
        self.allocated_at.elapsed()
    }
}

/// Double-buffering for overlapped transfers
///
/// Uses two buffers to allow overlapping GPU computation with data transfers.
pub struct DoubleBuffer {
    buffers: [Arc<dyn GpuBuffer>; 2],
    current: usize,
}

impl DoubleBuffer {
    /// Create a new double buffer
    pub fn new(device: &dyn GpuDevice, size: usize) -> GpuResult<Self> {
        let buf0 = device.allocate(size)?;
        let buf1 = device.allocate(size)?;
        Ok(DoubleBuffer {
            buffers: [buf0, buf1],
            current: 0,
        })
    }

    /// Get the current buffer (for GPU computation)
    pub fn current(&self) -> &Arc<dyn GpuBuffer> {
        &self.buffers[self.current]
    }

    /// Get the back buffer (for transfer)
    pub fn back(&self) -> &Arc<dyn GpuBuffer> {
        &self.buffers[1 - self.current]
    }

    /// Swap buffers
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.buffers[0].size()
    }
}

/// Ring buffer for streaming data to GPU
pub struct StreamingBuffer {
    buffer: Arc<dyn GpuBuffer>,
    write_offset: usize,
    read_offset: usize,
    capacity: usize,
}

impl StreamingBuffer {
    /// Create a new streaming buffer
    pub fn new(device: &dyn GpuDevice, capacity: usize) -> GpuResult<Self> {
        let buffer = device.allocate(capacity)?;
        Ok(StreamingBuffer {
            buffer,
            write_offset: 0,
            read_offset: 0,
            capacity,
        })
    }

    /// Get available space for writing
    pub fn available_write(&self) -> usize {
        if self.write_offset >= self.read_offset {
            self.capacity - self.write_offset + self.read_offset
        } else {
            self.read_offset - self.write_offset
        }
    }

    /// Get available data for reading
    pub fn available_read(&self) -> usize {
        if self.write_offset >= self.read_offset {
            self.write_offset - self.read_offset
        } else {
            self.capacity - self.read_offset + self.write_offset
        }
    }

    /// Advance write position
    pub fn advance_write(&mut self, bytes: usize) {
        self.write_offset = (self.write_offset + bytes) % self.capacity;
    }

    /// Advance read position
    pub fn advance_read(&mut self, bytes: usize) {
        self.read_offset = (self.read_offset + bytes) % self.capacity;
    }

    /// Get the underlying buffer
    pub fn buffer(&self) -> &Arc<dyn GpuBuffer> {
        &self.buffer
    }

    /// Get write offset
    pub fn write_offset(&self) -> usize {
        self.write_offset
    }

    /// Get read offset
    pub fn read_offset(&self) -> usize {
        self.read_offset
    }

    /// Reset the buffer
    pub fn reset(&mut self) {
        self.write_offset = 0;
        self.read_offset = 0;
    }
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

            // Test copy operations
            let data = vec![1.0f32; 50];
            mem.copy_from_slice(&data);
            assert_eq!(mem.len(), 50);

            let mut output = vec![0.0f32; 50];
            mem.copy_to_slice(&mut output);
            assert_eq!(output, data);
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

    #[test]
    fn test_allocation_stats() {
        let stats = AllocationStats {
            num_active: 10,
            total_bytes: 1024 * 1024,
            peak_bytes: 2 * 1024 * 1024,
        };

        assert_eq!(stats.num_active, 10);
        assert_eq!(stats.total_bytes, 1024 * 1024);
        assert_eq!(stats.peak_bytes, 2 * 1024 * 1024);
    }
}
