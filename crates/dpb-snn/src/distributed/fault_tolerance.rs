//! # Fault Tolerance
//!
//! Fault detection, recovery, and elastic training support.

use super::{DistributedError, DistributedResult, DistributedRuntime};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// Heartbeat monitor for worker health
pub struct HeartbeatMonitor {
    runtime: Arc<DistributedRuntime>,
    state: Arc<RwLock<HeartbeatState>>,
    config: HeartbeatConfig,
}

/// Heartbeat configuration
#[derive(Debug, Clone)]
pub struct HeartbeatConfig {
    /// Heartbeat interval (seconds)
    pub interval: Duration,
    /// Timeout threshold (seconds)
    pub timeout: Duration,
    /// Maximum missed heartbeats before failure
    pub max_missed: usize,
}

impl Default for HeartbeatConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            timeout: Duration::from_secs(30),
            max_missed: 3,
        }
    }
}

/// Heartbeat state
#[derive(Debug)]
struct HeartbeatState {
    /// Last heartbeat time for each worker
    last_heartbeat: HashMap<usize, Instant>,
    /// Consecutive missed heartbeats
    missed_count: HashMap<usize, usize>,
    /// Failed workers
    failed_workers: HashSet<usize>,
    /// Active monitoring
    monitoring: bool,
}

impl HeartbeatMonitor {
    /// Create new heartbeat monitor
    pub fn new(runtime: Arc<DistributedRuntime>) -> Self {
        Self::with_config(runtime, HeartbeatConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(runtime: Arc<DistributedRuntime>, config: HeartbeatConfig) -> Self {
        let world_size = runtime.world_size();
        let mut last_heartbeat = HashMap::new();
        let now = Instant::now();

        for rank in 0..world_size {
            last_heartbeat.insert(rank, now);
        }

        let state = HeartbeatState {
            last_heartbeat,
            missed_count: HashMap::new(),
            failed_workers: HashSet::new(),
            monitoring: false,
        };

        Self {
            runtime,
            state: Arc::new(RwLock::new(state)),
            config,
        }
    }

    /// Start monitoring
    pub fn start(&self) {
        let mut state = self.state.write().unwrap();
        state.monitoring = true;
    }

    /// Stop monitoring
    pub fn stop(&self) {
        let mut state = self.state.write().unwrap();
        state.monitoring = false;
    }

    /// Record heartbeat from worker
    pub fn record_heartbeat(&self, rank: usize) {
        let mut state = self.state.write().unwrap();
        state.last_heartbeat.insert(rank, Instant::now());
        state.missed_count.insert(rank, 0);
    }

    /// Check for failed workers
    pub fn check_health(&self) -> Vec<usize> {
        let mut state = self.state.write().unwrap();

        if !state.monitoring {
            return Vec::new();
        }

        let now = Instant::now();
        let mut newly_failed = Vec::new();

        // Collect ranks to check (to avoid borrowing issues)
        let ranks_to_check: Vec<(usize, Instant)> = state
            .last_heartbeat
            .iter()
            .filter(|(rank, _)| !state.failed_workers.contains(*rank))
            .map(|(rank, last_beat)| (*rank, *last_beat))
            .collect();

        // Check each rank
        for (rank, last_beat) in ranks_to_check {
            let elapsed = now.duration_since(last_beat);

            if elapsed > self.config.timeout {
                let missed = state.missed_count.entry(rank).or_insert(0);
                *missed += 1;

                if *missed >= self.config.max_missed {
                    state.failed_workers.insert(rank);
                    newly_failed.push(rank);
                }
            }
        }

        newly_failed
    }

    /// Get failed workers
    pub fn failed_workers(&self) -> Vec<usize> {
        let state = self.state.read().unwrap();
        state.failed_workers.iter().copied().collect()
    }

    /// Check if worker is healthy
    pub fn is_healthy(&self, rank: usize) -> bool {
        let state = self.state.read().unwrap();
        !state.failed_workers.contains(&rank)
    }

    /// Get number of healthy workers
    pub fn healthy_count(&self) -> usize {
        let state = self.state.read().unwrap();
        self.runtime.world_size() - state.failed_workers.len()
    }

    /// Reset worker health status
    pub fn reset_worker(&self, rank: usize) {
        let mut state = self.state.write().unwrap();
        state.failed_workers.remove(&rank);
        state.missed_count.insert(rank, 0);
        state.last_heartbeat.insert(rank, Instant::now());
    }
}

/// Checkpoint manager
pub struct CheckpointManager {
    checkpoint_dir: PathBuf,
    config: CheckpointConfig,
    state: Arc<RwLock<CheckpointState>>,
}

/// Checkpoint configuration
#[derive(Debug, Clone)]
pub struct CheckpointConfig {
    /// Maximum checkpoints to keep
    pub max_checkpoints: usize,
    /// Checkpoint frequency (epochs)
    pub frequency: usize,
    /// Enable incremental checkpointing
    pub incremental: bool,
    /// Compression enabled
    pub compress: bool,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            max_checkpoints: 5,
            frequency: 1,
            incremental: false,
            compress: true,
        }
    }
}

/// Checkpoint state
#[derive(Debug)]
struct CheckpointState {
    /// Available checkpoints
    checkpoints: Vec<CheckpointMetadata>,
    /// Last checkpoint time
    last_checkpoint: Option<Instant>,
}

/// Checkpoint metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    /// Checkpoint ID
    pub id: String,
    /// Epoch number
    pub epoch: usize,
    /// Step number
    pub step: usize,
    /// File path
    pub path: PathBuf,
    /// File size (bytes)
    pub size: usize,
    /// Timestamp
    pub timestamp: u64,
    /// Checksum (for validation)
    pub checksum: String,
}

impl CheckpointManager {
    /// Create new checkpoint manager
    pub fn new<P: AsRef<Path>>(checkpoint_dir: P) -> DistributedResult<Self> {
        Self::with_config(checkpoint_dir, CheckpointConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config<P: AsRef<Path>>(
        checkpoint_dir: P,
        config: CheckpointConfig,
    ) -> DistributedResult<Self> {
        let checkpoint_dir = checkpoint_dir.as_ref().to_path_buf();

        // Create directory if it doesn't exist
        std::fs::create_dir_all(&checkpoint_dir).map_err(|e| {
            DistributedError::Checkpoint(format!("Failed to create checkpoint directory: {}", e))
        })?;

        let state = CheckpointState {
            checkpoints: Vec::new(),
            last_checkpoint: None,
        };

        Ok(Self {
            checkpoint_dir,
            config,
            state: Arc::new(RwLock::new(state)),
        })
    }

    /// Save checkpoint
    pub fn save(
        &self,
        epoch: usize,
        step: usize,
        data: &[u8],
    ) -> DistributedResult<CheckpointMetadata> {
        let id = format!("checkpoint_epoch{}_step{}", epoch, step);
        let filename = format!("{}.ckpt", id);
        let path = self.checkpoint_dir.join(&filename);

        // Write checkpoint file
        std::fs::write(&path, data).map_err(|e| {
            DistributedError::Checkpoint(format!("Failed to write checkpoint: {}", e))
        })?;

        // Compute checksum
        let checksum = compute_checksum(data);

        let metadata = CheckpointMetadata {
            id,
            epoch,
            step,
            path,
            size: data.len(),
            timestamp: current_timestamp(),
            checksum,
        };

        // Update state
        let mut state = self.state.write().unwrap();
        state.checkpoints.push(metadata.clone());
        state.last_checkpoint = Some(Instant::now());

        // Cleanup old checkpoints
        self.cleanup_checkpoints(&mut state)?;

        Ok(metadata)
    }

    /// Load checkpoint
    pub fn load(&self, checkpoint_id: &str) -> DistributedResult<Vec<u8>> {
        let state = self.state.read().unwrap();

        let metadata = state
            .checkpoints
            .iter()
            .find(|c| c.id == checkpoint_id)
            .ok_or_else(|| {
                DistributedError::Checkpoint(format!("Checkpoint not found: {}", checkpoint_id))
            })?;

        let data = std::fs::read(&metadata.path).map_err(|e| {
            DistributedError::Checkpoint(format!("Failed to read checkpoint: {}", e))
        })?;

        // Validate checksum
        let checksum = compute_checksum(&data);
        if checksum != metadata.checksum {
            return Err(DistributedError::Checkpoint(
                "Checkpoint checksum mismatch".to_string(),
            ));
        }

        Ok(data)
    }

    /// Load latest checkpoint
    pub fn load_latest(&self) -> DistributedResult<Option<Vec<u8>>> {
        let state = self.state.read().unwrap();

        if let Some(latest) = state.checkpoints.last() {
            Ok(Some(self.load(&latest.id)?))
        } else {
            Ok(None)
        }
    }

    /// Get checkpoint metadata
    pub fn get_metadata(&self, checkpoint_id: &str) -> Option<CheckpointMetadata> {
        let state = self.state.read().unwrap();
        state
            .checkpoints
            .iter()
            .find(|c| c.id == checkpoint_id)
            .cloned()
    }

    /// List all checkpoints
    pub fn list_checkpoints(&self) -> Vec<CheckpointMetadata> {
        let state = self.state.read().unwrap();
        state.checkpoints.clone()
    }

    /// Get latest checkpoint metadata
    pub fn latest_checkpoint(&self) -> Option<CheckpointMetadata> {
        let state = self.state.read().unwrap();
        state.checkpoints.last().cloned()
    }

    /// Delete checkpoint
    pub fn delete(&self, checkpoint_id: &str) -> DistributedResult<()> {
        let mut state = self.state.write().unwrap();

        if let Some(pos) = state.checkpoints.iter().position(|c| c.id == checkpoint_id) {
            let metadata = state.checkpoints.remove(pos);
            std::fs::remove_file(&metadata.path).map_err(|e| {
                DistributedError::Checkpoint(format!("Failed to delete checkpoint: {}", e))
            })?;
        }

        Ok(())
    }

    /// Cleanup old checkpoints
    fn cleanup_checkpoints(&self, state: &mut CheckpointState) -> DistributedResult<()> {
        while state.checkpoints.len() > self.config.max_checkpoints {
            let oldest = state.checkpoints.remove(0);
            std::fs::remove_file(&oldest.path).ok(); // Ignore errors
        }
        Ok(())
    }
}

/// Elastic training manager
pub struct ElasticTrainingManager {
    runtime: Arc<DistributedRuntime>,
    min_workers: usize,
    max_workers: usize,
    state: Arc<RwLock<ElasticState>>,
}

/// Elastic state
#[derive(Debug)]
struct ElasticState {
    /// Current active workers
    active_workers: HashSet<usize>,
    /// Pending workers (joining)
    pending_workers: HashSet<usize>,
    /// Generation (incremented on topology change)
    generation: usize,
}

impl ElasticTrainingManager {
    /// Create new elastic training manager
    pub fn new(runtime: Arc<DistributedRuntime>, min_workers: usize, max_workers: usize) -> Self {
        let world_size = runtime.world_size();
        let active_workers: HashSet<usize> = (0..world_size).collect();

        let state = ElasticState {
            active_workers,
            pending_workers: HashSet::new(),
            generation: 0,
        };

        Self {
            runtime,
            min_workers,
            max_workers,
            state: Arc::new(RwLock::new(state)),
        }
    }

    /// Add worker to training
    pub fn add_worker(&self, rank: usize) -> DistributedResult<()> {
        let mut state = self.state.write().unwrap();

        if state.active_workers.len() >= self.max_workers {
            return Err(DistributedError::InvalidConfig(
                "Maximum workers reached".to_string(),
            ));
        }

        state.pending_workers.insert(rank);
        Ok(())
    }

    /// Remove worker from training
    pub fn remove_worker(&self, rank: usize) -> DistributedResult<()> {
        let mut state = self.state.write().unwrap();

        if state.active_workers.len() <= self.min_workers {
            return Err(DistributedError::InvalidConfig(
                "Minimum workers required".to_string(),
            ));
        }

        state.active_workers.remove(&rank);
        state.generation += 1;

        Ok(())
    }

    /// Commit pending workers
    pub fn commit_workers(&self) -> DistributedResult<()> {
        let mut state = self.state.write().unwrap();

        let pending: Vec<usize> = state.pending_workers.drain().collect();
        for rank in pending {
            state.active_workers.insert(rank);
        }

        state.generation += 1;
        Ok(())
    }

    /// Get active workers
    /// The runtime this manager is elastic over.
    pub fn runtime(&self) -> &Arc<DistributedRuntime> {
        &self.runtime
    }

    pub fn active_workers(&self) -> Vec<usize> {
        let state = self.state.read().unwrap();
        state.active_workers.iter().copied().collect()
    }

    /// Get generation
    pub fn generation(&self) -> usize {
        let state = self.state.read().unwrap();
        state.generation
    }

    /// Check if topology is stable
    pub fn is_stable(&self) -> bool {
        let state = self.state.read().unwrap();
        state.pending_workers.is_empty()
    }

    /// Can continue training
    pub fn can_continue(&self) -> bool {
        let state = self.state.read().unwrap();
        state.active_workers.len() >= self.min_workers
    }
}

/// Compute checksum for data
fn compute_checksum(data: &[u8]) -> String {
    // Simple checksum (in production, use proper hash like SHA256)
    let sum: u64 = data.iter().map(|&b| b as u64).sum();
    format!("{:x}", sum)
}

/// Get current timestamp
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::{DistributedBackend, DistributedConfig};
    use std::thread;

    #[test]
    fn test_heartbeat_monitor() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let monitor = HeartbeatMonitor::new(runtime);

        monitor.start();

        assert_eq!(monitor.healthy_count(), 4);
        assert!(monitor.is_healthy(0));
        assert!(monitor.is_healthy(1));

        monitor.record_heartbeat(0);
        assert!(monitor.is_healthy(0));
    }

    #[test]
    fn test_heartbeat_failure_detection() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());

        let hb_config = HeartbeatConfig {
            interval: Duration::from_millis(10),
            timeout: Duration::from_millis(50),
            max_missed: 1,
        };

        let monitor = HeartbeatMonitor::with_config(runtime, hb_config);
        monitor.start();

        // Wait for timeout
        thread::sleep(Duration::from_millis(100));

        let failed = monitor.check_health();
        assert!(!failed.is_empty());
    }

    #[test]
    fn test_checkpoint_manager() {
        let temp_dir = std::env::temp_dir().join("dpb_test_checkpoints");
        let manager = CheckpointManager::new(&temp_dir).unwrap();

        let data = vec![1, 2, 3, 4, 5];
        let metadata = manager.save(1, 100, &data).unwrap();

        assert_eq!(metadata.epoch, 1);
        assert_eq!(metadata.step, 100);

        let loaded = manager.load(&metadata.id).unwrap();
        assert_eq!(loaded, data);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_checkpoint_latest() {
        let temp_dir = std::env::temp_dir().join("dpb_test_checkpoints_latest");
        let manager = CheckpointManager::new(&temp_dir).unwrap();

        manager.save(1, 100, &[1, 2, 3]).unwrap();
        manager.save(2, 200, &[4, 5, 6]).unwrap();

        let latest = manager.load_latest().unwrap().unwrap();
        assert_eq!(latest, vec![4, 5, 6]);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_checkpoint_cleanup() {
        let temp_dir = std::env::temp_dir().join("dpb_test_checkpoints_cleanup");
        let config = CheckpointConfig {
            max_checkpoints: 2,
            ..Default::default()
        };
        let manager = CheckpointManager::with_config(&temp_dir, config).unwrap();

        manager.save(1, 100, &[1]).unwrap();
        manager.save(2, 200, &[2]).unwrap();
        manager.save(3, 300, &[3]).unwrap();

        let checkpoints = manager.list_checkpoints();
        assert_eq!(checkpoints.len(), 2);
        assert_eq!(checkpoints[0].epoch, 2);
        assert_eq!(checkpoints[1].epoch, 3);

        // Cleanup
        std::fs::remove_dir_all(temp_dir).ok();
    }

    #[test]
    fn test_elastic_training() {
        let config = DistributedConfig::new(4, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let manager = ElasticTrainingManager::new(runtime, 2, 8);

        assert_eq!(manager.active_workers().len(), 4);
        assert!(manager.can_continue());

        manager.add_worker(5).unwrap();
        assert!(!manager.is_stable());

        manager.commit_workers().unwrap();
        assert!(manager.is_stable());
        assert_eq!(manager.active_workers().len(), 5);
    }

    #[test]
    fn test_elastic_training_limits() {
        let config = DistributedConfig::new(2, 0, DistributedBackend::Mock);
        let runtime = Arc::new(DistributedRuntime::init(config).unwrap());
        let manager = ElasticTrainingManager::new(runtime, 2, 4);

        // Try to remove below minimum
        assert!(manager.remove_worker(0).is_err());

        // Add workers up to maximum
        manager.add_worker(2).unwrap();
        manager.add_worker(3).unwrap();
        manager.commit_workers().unwrap();

        // Try to add beyond maximum
        assert!(manager.add_worker(4).is_err());
    }
}
