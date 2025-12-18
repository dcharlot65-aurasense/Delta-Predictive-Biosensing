//! # Inter-Node Communication
//!
//! Low-level communication primitives for distributed training.

use super::{CommunicationBackend, DistributedError, DistributedResult, ReduceOp};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Message types for distributed communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageType {
    /// Gradient update
    Gradients {
        param_name: String,
        data: Vec<f32>,
        step: usize,
    },
    /// Weight/parameter update
    Weights {
        param_name: String,
        data: Vec<f32>,
        version: usize,
    },
    /// Spike train data
    Spikes {
        layer_id: usize,
        data: Vec<bool>,
        timestep: usize,
    },
    /// Control message
    Control(ControlMessage),
}

/// Control messages
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlMessage {
    /// Start training
    StartTraining,
    /// Stop training
    StopTraining,
    /// Checkpoint request
    Checkpoint,
    /// Worker ready
    Ready,
    /// Worker failed
    Failed,
    /// Barrier sync
    Barrier,
}

/// Message envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    /// Source rank
    pub source: usize,
    /// Destination rank (or broadcast if None)
    pub destination: Option<usize>,
    /// Message tag
    pub tag: i32,
    /// Message payload
    pub payload: MessageType,
    /// Timestamp
    pub timestamp: u64,
}

impl Message {
    /// Create new message
    pub fn new(source: usize, destination: Option<usize>, tag: i32, payload: MessageType) -> Self {
        Self {
            source,
            destination,
            tag,
            payload,
            timestamp: current_timestamp(),
        }
    }

    /// Create gradient message
    pub fn gradients(
        source: usize,
        destination: usize,
        param_name: String,
        data: Vec<f32>,
        step: usize,
    ) -> Self {
        Self::new(
            source,
            Some(destination),
            0,
            MessageType::Gradients {
                param_name,
                data,
                step,
            },
        )
    }

    /// Create weight message
    pub fn weights(
        source: usize,
        destination: usize,
        param_name: String,
        data: Vec<f32>,
        version: usize,
    ) -> Self {
        Self::new(
            source,
            Some(destination),
            0,
            MessageType::Weights {
                param_name,
                data,
                version,
            },
        )
    }

    /// Create control message
    pub fn control(source: usize, control: ControlMessage) -> Self {
        Self::new(source, None, 0, MessageType::Control(control))
    }

    /// Check if message is broadcast
    pub fn is_broadcast(&self) -> bool {
        self.destination.is_none()
    }
}

/// Communication operations
pub struct CommunicationOps {
    backend: Arc<dyn CommunicationBackend>,
    rank: usize,
    world_size: usize,
}

impl CommunicationOps {
    /// Create new communication operations
    pub fn new(backend: Arc<dyn CommunicationBackend>, rank: usize, world_size: usize) -> Self {
        Self {
            backend,
            rank,
            world_size,
        }
    }

    /// Send data to destination
    pub fn send(&self, data: &[f32], dst: usize, tag: i32) -> DistributedResult<()> {
        self.backend.send(data, dst, tag)
    }

    /// Receive data from source
    pub fn recv(&self, data: &mut [f32], src: usize, tag: i32) -> DistributedResult<()> {
        self.backend.recv(data, src, tag)
    }

    /// Broadcast from root to all workers
    pub fn broadcast(&self, data: &mut [f32], root: usize) -> DistributedResult<()> {
        self.backend.broadcast(data, root)
    }

    /// All-reduce across all workers
    pub fn all_reduce(&self, data: &mut [f32], op: ReduceOp) -> DistributedResult<()> {
        self.backend.all_reduce(data, op)
    }

    /// Reduce to root
    pub fn reduce(&self, data: &mut [f32], root: usize, op: ReduceOp) -> DistributedResult<()> {
        self.backend.reduce(data, root, op)
    }

    /// Scatter data from root
    pub fn scatter(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        root: usize,
    ) -> DistributedResult<()> {
        self.backend.scatter(send_data, recv_data, root)
    }

    /// Gather data to root
    pub fn gather(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        root: usize,
    ) -> DistributedResult<()> {
        self.backend.gather(send_data, recv_data, root)
    }

    /// All-gather across all workers
    pub fn all_gather(&self, send_data: &[f32], recv_data: &mut [f32]) -> DistributedResult<()> {
        self.backend.all_gather(send_data, recv_data)
    }

    /// Ring all-reduce (bandwidth-optimal)
    pub fn ring_all_reduce(&self, data: &mut [f32], op: ReduceOp) -> DistributedResult<()> {
        if self.world_size == 1 {
            return Ok(());
        }

        let chunk_size = (data.len() + self.world_size - 1) / self.world_size;
        let mut chunks: Vec<Vec<f32>> = Vec::new();

        // Split data into chunks
        for i in 0..self.world_size {
            let start = i * chunk_size;
            let end = (start + chunk_size).min(data.len());
            if start < data.len() {
                chunks.push(data[start..end].to_vec());
            } else {
                chunks.push(Vec::new());
            }
        }

        // Reduce-scatter phase
        for step in 0..self.world_size - 1 {
            let send_idx = (self.rank + self.world_size - step) % self.world_size;
            let recv_idx = (self.rank + self.world_size - step - 1) % self.world_size;

            let next_rank = (self.rank + 1) % self.world_size;
            let prev_rank = (self.rank + self.world_size - 1) % self.world_size;

            // Send chunk to next rank
            if !chunks[send_idx].is_empty() {
                self.send(&chunks[send_idx], next_rank, step as i32)?;
            }

            // Receive chunk from previous rank
            if !chunks[recv_idx].is_empty() {
                let mut recv_chunk = vec![0.0; chunks[recv_idx].len()];
                self.recv(&mut recv_chunk, prev_rank, step as i32)?;

                // Reduce received chunk
                for (i, val) in recv_chunk.iter().enumerate() {
                    chunks[recv_idx][i] = match op {
                        ReduceOp::Sum | ReduceOp::Mean => chunks[recv_idx][i] + val,
                        ReduceOp::Min => chunks[recv_idx][i].min(*val),
                        ReduceOp::Max => chunks[recv_idx][i].max(*val),
                        ReduceOp::Product => chunks[recv_idx][i] * val,
                    };
                }
            }
        }

        // All-gather phase
        for step in 0..self.world_size - 1 {
            let send_idx = (self.rank + 1 - step + self.world_size) % self.world_size;
            let recv_idx = (self.rank - step + self.world_size) % self.world_size;

            let next_rank = (self.rank + 1) % self.world_size;
            let prev_rank = (self.rank + self.world_size - 1) % self.world_size;

            // Send complete chunk to next rank
            if !chunks[send_idx].is_empty() {
                self.send(
                    &chunks[send_idx],
                    next_rank,
                    (self.world_size + step) as i32,
                )?;
            }

            // Receive complete chunk from previous rank
            if !chunks[recv_idx].is_empty() {
                self.recv(
                    &mut chunks[recv_idx],
                    prev_rank,
                    (self.world_size + step) as i32,
                )?;
            }
        }

        // Reassemble data
        let mut offset = 0;
        for chunk in chunks.iter() {
            let len = chunk.len().min(data.len() - offset);
            if len > 0 {
                data[offset..offset + len].copy_from_slice(&chunk[..len]);
                offset += len;
            }
        }

        // Apply averaging for Mean operation
        if matches!(op, ReduceOp::Mean) {
            let world_size = self.world_size as f32;
            for val in data.iter_mut() {
                *val /= world_size;
            }
        }

        Ok(())
    }

    /// Reduce-scatter operation
    pub fn reduce_scatter(
        &self,
        send_data: &[f32],
        recv_data: &mut [f32],
        op: ReduceOp,
    ) -> DistributedResult<()> {
        // Simplified implementation - actual would use efficient algorithm
        let chunk_size = send_data.len() / self.world_size;

        // All-to-all reduction
        let mut all_data = vec![0.0; send_data.len()];
        all_data.copy_from_slice(send_data);
        self.all_reduce(&mut all_data, op)?;

        // Extract this rank's chunk
        let start = self.rank * chunk_size;
        let end = start + recv_data.len();
        recv_data.copy_from_slice(&all_data[start..end]);

        Ok(())
    }

    /// All-to-all communication
    pub fn all_to_all(&self, send_data: &[f32], recv_data: &mut [f32]) -> DistributedResult<()> {
        let chunk_size = send_data.len() / self.world_size;

        for rank in 0..self.world_size {
            let send_start = rank * chunk_size;
            let send_end = send_start + chunk_size;
            let recv_start = rank * chunk_size;
            let recv_end = recv_start + chunk_size;

            if rank == self.rank {
                // Local copy
                recv_data[recv_start..recv_end].copy_from_slice(&send_data[send_start..send_end]);
            } else {
                // Exchange with peer
                let send_chunk = &send_data[send_start..send_end];
                let recv_chunk = &mut recv_data[recv_start..recv_end];

                // Send to rank
                self.send(send_chunk, rank, rank as i32)?;
                // Receive from rank
                self.recv(recv_chunk, rank, rank as i32)?;
            }
        }

        Ok(())
    }

    /// Barrier synchronization
    pub fn barrier(&self) -> DistributedResult<()> {
        self.backend.barrier()
    }
}

/// Message queue for asynchronous communication
pub struct MessageQueue {
    queue: Arc<Mutex<VecDeque<Message>>>,
    max_size: usize,
}

impl MessageQueue {
    /// Create new message queue
    pub fn new(max_size: usize) -> Self {
        Self {
            queue: Arc::new(Mutex::new(VecDeque::new())),
            max_size,
        }
    }

    /// Push message to queue
    pub fn push(&self, message: Message) -> DistributedResult<()> {
        let mut queue = self.queue.lock().unwrap();

        if queue.len() >= self.max_size {
            return Err(DistributedError::Communication(
                "Message queue full".to_string(),
            ));
        }

        queue.push_back(message);
        Ok(())
    }

    /// Pop message from queue
    pub fn pop(&self) -> Option<Message> {
        let mut queue = self.queue.lock().unwrap();
        queue.pop_front()
    }

    /// Peek at next message without removing
    pub fn peek(&self) -> Option<Message> {
        let queue = self.queue.lock().unwrap();
        queue.front().cloned()
    }

    /// Get queue size
    pub fn len(&self) -> usize {
        let queue = self.queue.lock().unwrap();
        queue.len()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear queue
    pub fn clear(&self) {
        let mut queue = self.queue.lock().unwrap();
        queue.clear();
    }
}

/// Get current timestamp (milliseconds since epoch)
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Communication statistics
#[derive(Debug, Clone, Default)]
pub struct CommStats {
    /// Total bytes sent
    pub bytes_sent: usize,
    /// Total bytes received
    pub bytes_received: usize,
    /// Number of send operations
    pub sends: usize,
    /// Number of receive operations
    pub receives: usize,
    /// Number of collective operations
    pub collectives: usize,
    /// Total communication time (ms)
    pub total_time_ms: u64,
}

impl CommStats {
    /// Create new stats
    pub fn new() -> Self {
        Self::default()
    }

    /// Record send operation
    pub fn record_send(&mut self, bytes: usize, time_ms: u64) {
        self.bytes_sent += bytes;
        self.sends += 1;
        self.total_time_ms += time_ms;
    }

    /// Record receive operation
    pub fn record_recv(&mut self, bytes: usize, time_ms: u64) {
        self.bytes_received += bytes;
        self.receives += 1;
        self.total_time_ms += time_ms;
    }

    /// Record collective operation
    pub fn record_collective(&mut self, bytes: usize, time_ms: u64) {
        self.collectives += 1;
        self.total_time_ms += time_ms;
        // Collectives involve both send and receive
        self.bytes_sent += bytes;
        self.bytes_received += bytes;
    }

    /// Get total bytes transferred
    pub fn total_bytes(&self) -> usize {
        self.bytes_sent + self.bytes_received
    }

    /// Get average latency (ms)
    pub fn avg_latency_ms(&self) -> f64 {
        let total_ops = self.sends + self.receives + self.collectives;
        if total_ops > 0 {
            self.total_time_ms as f64 / total_ops as f64
        } else {
            0.0
        }
    }

    /// Get bandwidth (MB/s)
    pub fn bandwidth_mbps(&self) -> f64 {
        if self.total_time_ms > 0 {
            let mb = self.total_bytes() as f64 / (1024.0 * 1024.0);
            let seconds = self.total_time_ms as f64 / 1000.0;
            mb / seconds
        } else {
            0.0
        }
    }

    /// Merge with other stats
    pub fn merge(&mut self, other: &CommStats) {
        self.bytes_sent += other.bytes_sent;
        self.bytes_received += other.bytes_received;
        self.sends += other.sends;
        self.receives += other.receives;
        self.collectives += other.collectives;
        self.total_time_ms += other.total_time_ms;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::gradients(0, 1, "layer1.weight".to_string(), vec![1.0, 2.0], 10);

        assert_eq!(msg.source, 0);
        assert_eq!(msg.destination, Some(1));
        assert!(!msg.is_broadcast());

        match msg.payload {
            MessageType::Gradients { step, .. } => assert_eq!(step, 10),
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_control_message() {
        let msg = Message::control(0, ControlMessage::StartTraining);
        assert!(msg.is_broadcast());

        match msg.payload {
            MessageType::Control(ControlMessage::StartTraining) => {}
            _ => panic!("Wrong message type"),
        }
    }

    #[test]
    fn test_message_queue() {
        let queue = MessageQueue::new(10);

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        let msg = Message::control(0, ControlMessage::Ready);
        queue.push(msg.clone()).unwrap();

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let popped = queue.pop().unwrap();
        assert_eq!(popped.source, msg.source);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_message_queue_overflow() {
        let queue = MessageQueue::new(2);

        let msg1 = Message::control(0, ControlMessage::Ready);
        let msg2 = Message::control(1, ControlMessage::Ready);
        let msg3 = Message::control(2, ControlMessage::Ready);

        assert!(queue.push(msg1).is_ok());
        assert!(queue.push(msg2).is_ok());
        assert!(queue.push(msg3).is_err()); // Should fail
    }

    #[test]
    fn test_comm_stats() {
        let mut stats = CommStats::new();

        stats.record_send(1024, 10);
        stats.record_recv(2048, 20);
        stats.record_collective(4096, 30);

        assert_eq!(stats.sends, 1);
        assert_eq!(stats.receives, 1);
        assert_eq!(stats.collectives, 1);
        assert_eq!(stats.total_bytes(), 1024 + 2048 + 4096 * 2);
        assert_eq!(stats.total_time_ms, 60);
        assert_eq!(stats.avg_latency_ms(), 20.0);
    }

    #[test]
    fn test_comm_stats_merge() {
        let mut stats1 = CommStats::new();
        stats1.record_send(1024, 10);

        let mut stats2 = CommStats::new();
        stats2.record_recv(2048, 20);

        stats1.merge(&stats2);

        assert_eq!(stats1.bytes_sent, 1024);
        assert_eq!(stats1.bytes_received, 2048);
        assert_eq!(stats1.total_time_ms, 30);
    }
}
