use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{unbounded, Receiver, Sender};
use rosc::{OscMessage, OscPacket, OscType};
use tokio::net::UdpSocket;

/// Connection status for the OSC receiver
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectionStatus {
    /// Not yet started
    Stopped,
    /// Socket bound, waiting for data
    Listening,
    /// Actively receiving data
    Receiving,
    /// Error state
    Error,
}

/// Events sent from the OSC receiver to the GUI
#[derive(Debug, Clone)]
pub enum ReceiverEvent {
    /// A frame of data was received (coordinates only, no visibility)
    Frame(Vec<f32>),
    /// A frame with per-joint visibility scores (coordinates, visibility)
    FrameWithVisibility(Vec<f32>, Vec<f32>),
    /// Connection status changed
    StatusChanged(ConnectionStatus),
    /// An error occurred
    Error(String),
}

/// A received frame with optional per-joint visibility scores.
#[derive(Debug, Clone)]
pub struct ReceivedFrame {
    /// Coordinate data (e.g., 66 floats for 33 joints × XY)
    pub coords: Vec<f32>,
    /// Per-joint visibility scores (e.g., 33 floats, one per joint).
    /// None for legacy 66-float frames (treated as all visible).
    pub visibility: Option<Vec<f32>>,
}

/// Shared state between the receiver task and the GUI
pub struct OscReceiverState {
    pub status: ConnectionStatus,
    pub frame_count: u64,
    pub last_frame_time: Option<Instant>,
    pub error_message: Option<String>,
}

impl Default for OscReceiverState {
    fn default() -> Self {
        Self {
            status: ConnectionStatus::Stopped,
            frame_count: 0,
            last_frame_time: None,
            error_message: None,
        }
    }
}

/// Handle to control the OSC receiver from the GUI
pub struct OscReceiverHandle {
    /// Channel to receive events from the receiver
    pub event_rx: Receiver<ReceiverEvent>,
    /// Flag to signal shutdown
    #[allow(dead_code)]
    shutdown: Arc<AtomicBool>,
    /// Current state (updated by polling events)
    pub state: OscReceiverState,
}

impl OscReceiverHandle {
    /// Poll for new events and update state.
    /// Returns any frames received since last poll.
    /// Call this every frame in the GUI update loop.
    pub fn poll(&mut self) -> Vec<ReceivedFrame> {
        let mut frames = Vec::new();

        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                ReceiverEvent::Frame(data) => {
                    self.state.frame_count += 1;
                    self.state.last_frame_time = Some(Instant::now());
                    self.state.status = ConnectionStatus::Receiving;
                    frames.push(ReceivedFrame {
                        coords: data,
                        visibility: None,
                    });
                }
                ReceiverEvent::FrameWithVisibility(coords, visibility) => {
                    self.state.frame_count += 1;
                    self.state.last_frame_time = Some(Instant::now());
                    self.state.status = ConnectionStatus::Receiving;
                    frames.push(ReceivedFrame {
                        coords,
                        visibility: Some(visibility),
                    });
                }
                ReceiverEvent::StatusChanged(status) => {
                    self.state.status = status;
                }
                ReceiverEvent::Error(msg) => {
                    self.state.status = ConnectionStatus::Error;
                    self.state.error_message = Some(msg);
                }
            }
        }

        frames
    }

    /// Get time since last frame in milliseconds
    pub fn ms_since_last_frame(&self) -> Option<u64> {
        self.state
            .last_frame_time
            .map(|t| t.elapsed().as_millis() as u64)
    }

    /// Signal the receiver to shut down
    #[allow(dead_code)]
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

/// MediaPipe Pose landmark constants
const JOINTS: usize = 33;
const COORDS_PER_JOINT: usize = 2;
const LEGACY_FRAME_LEN: usize = JOINTS * COORDS_PER_JOINT; // 66
const VISIBILITY_FRAME_LEN: usize = JOINTS * (COORDS_PER_JOINT + 1); // 99

/// OSC Receiver that listens for skeleton data
pub struct OscReceiver {
    port: u16,
    address_filter: String,
    event_tx: Sender<ReceiverEvent>,
    shutdown: Arc<AtomicBool>,
}

impl OscReceiver {
    /// Create a new OSC receiver and return the handle for the GUI
    pub fn new(port: u16, address_filter: &str) -> (Self, OscReceiverHandle) {
        let (event_tx, event_rx) = unbounded();
        let shutdown = Arc::new(AtomicBool::new(false));

        let receiver = Self {
            port,
            address_filter: address_filter.to_string(),
            event_tx,
            shutdown: shutdown.clone(),
        };

        let handle = OscReceiverHandle {
            event_rx,
            shutdown,
            state: OscReceiverState::default(),
        };

        (receiver, handle)
    }

    /// Start the receiver (runs until shutdown is signaled)
    pub async fn run(self) {
        // Try to bind the socket
        let bind_addr = format!("0.0.0.0:{}", self.port);
        let socket = match UdpSocket::bind(&bind_addr).await {
            Ok(s) => {
                let _ = self
                    .event_tx
                    .send(ReceiverEvent::StatusChanged(ConnectionStatus::Listening));
                s
            }
            Err(e) => {
                let _ = self.event_tx.send(ReceiverEvent::Error(format!(
                    "Failed to bind to {}: {}",
                    bind_addr, e
                )));
                return;
            }
        };

        let mut buf = [0u8; 65535]; // Max UDP packet size

        loop {
            // Check for shutdown
            if self.shutdown.load(Ordering::SeqCst) {
                break;
            }

            // Receive with timeout so we can check shutdown periodically
            let recv_result = tokio::time::timeout(
                tokio::time::Duration::from_millis(10),
                socket.recv_from(&mut buf),
            )
            .await;

            match recv_result {
                Ok(Ok((size, _addr))) => {
                    // Parse the OSC packet
                    if let Ok(packet) = rosc::decoder::decode_udp(&buf[..size]) {
                        self.handle_packet(&packet.1);
                    }
                }
                Ok(Err(e)) => {
                    let _ = self
                        .event_tx
                        .send(ReceiverEvent::Error(format!("Receive error: {}", e)));
                }
                Err(_) => {
                    // Timeout, just continue to check shutdown flag
                }
            }
        }

        let _ = self
            .event_tx
            .send(ReceiverEvent::StatusChanged(ConnectionStatus::Stopped));
    }

    fn handle_packet(&self, packet: &OscPacket) {
        match packet {
            OscPacket::Message(msg) => {
                self.handle_message(msg);
            }
            OscPacket::Bundle(bundle) => {
                for packet in &bundle.content {
                    self.handle_packet(packet);
                }
            }
        }
    }

    fn handle_message(&self, msg: &OscMessage) {
        // Accept both legacy Wekinator address (configured filter, typically "/wek/inputs")
        // and native RALF address ("/ralf/pose") for forward compatibility.
        if msg.addr != self.address_filter && msg.addr != "/ralf/pose" {
            return;
        }

        // Extract float values from the message
        let mut floats = Vec::new();
        for arg in &msg.args {
            match arg {
                OscType::Float(f) => floats.push(*f),
                OscType::Double(d) => floats.push(*d as f32),
                OscType::Int(i) => floats.push(*i as f32),
                _ => {} // Ignore other types
            }
        }

        if !floats.is_empty() {
            // Auto-detect visibility data based on frame length:
            // 99 floats = 33 joints × 3 (x, y, visibility)
            // 66 floats = 33 joints × 2 (x, y) — legacy format
            if floats.len() == VISIBILITY_FRAME_LEN {
                let mut coords = Vec::with_capacity(LEGACY_FRAME_LEN);
                let mut visibility = Vec::with_capacity(JOINTS);
                for chunk in floats.chunks(COORDS_PER_JOINT + 1) {
                    coords.push(chunk[0]); // x
                    coords.push(chunk[1]); // y
                    visibility.push(chunk[2]); // visibility
                }
                let _ = self
                    .event_tx
                    .send(ReceiverEvent::FrameWithVisibility(coords, visibility));
            } else {
                let _ = self.event_tx.send(ReceiverEvent::Frame(floats));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_receiver_creation() {
        let (receiver, handle) = OscReceiver::new(6448, "/wek/inputs");
        assert_eq!(receiver.port, 6448);
        assert_eq!(receiver.address_filter, "/wek/inputs");
        assert_eq!(handle.state.status, ConnectionStatus::Stopped);
        assert_eq!(handle.state.frame_count, 0);
    }

    #[test]
    fn test_handle_polls_events() {
        let (receiver, mut handle) = OscReceiver::new(6448, "/wek/inputs");

        // Simulate sending an event
        receiver
            .event_tx
            .send(ReceiverEvent::StatusChanged(ConnectionStatus::Listening))
            .unwrap();

        // Poll should update state
        handle.poll();
        assert_eq!(handle.state.status, ConnectionStatus::Listening);
    }

    #[test]
    fn test_frame_count_increments() {
        let (receiver, mut handle) = OscReceiver::new(6448, "/wek/inputs");

        // Simulate receiving frames
        receiver
            .event_tx
            .send(ReceiverEvent::Frame(vec![1.0, 2.0, 3.0]))
            .unwrap();
        receiver
            .event_tx
            .send(ReceiverEvent::Frame(vec![4.0, 5.0, 6.0]))
            .unwrap();

        handle.poll();
        assert_eq!(handle.state.frame_count, 2);
        assert_eq!(handle.state.status, ConnectionStatus::Receiving);
    }

    #[test]
    fn test_legacy_66_float_frame_has_no_visibility() {
        let (receiver, mut handle) = OscReceiver::new(6448, "/wek/inputs");

        // 66 floats = legacy frame, no visibility
        let frame_66: Vec<f32> = (0..66).map(|i| i as f32).collect();
        receiver
            .event_tx
            .send(ReceiverEvent::Frame(frame_66.clone()))
            .unwrap();

        let frames = handle.poll();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].coords, frame_66);
        assert!(frames[0].visibility.is_none());
    }

    #[test]
    fn test_99_float_frame_extracts_visibility() {
        let (receiver, mut handle) = OscReceiver::new(6448, "/wek/inputs");

        // 99 floats = 33 joints × (x, y, visibility)
        // Build: joint 0 = (0.1, 0.2, 0.9), joint 1 = (0.3, 0.4, 0.8), ...
        let mut frame_99 = Vec::with_capacity(99);
        for i in 0..33 {
            frame_99.push(i as f32 * 0.1); // x
            frame_99.push(i as f32 * 0.1 + 0.01); // y
            frame_99.push(1.0 - i as f32 * 0.02); // visibility
        }

        receiver
            .event_tx
            .send(ReceiverEvent::FrameWithVisibility(
                // coords
                {
                    let mut c = Vec::with_capacity(66);
                    for chunk in frame_99.chunks(3) {
                        c.push(chunk[0]);
                        c.push(chunk[1]);
                    }
                    c
                },
                // visibility
                frame_99.chunks(3).map(|c| c[2]).collect(),
            ))
            .unwrap();

        let frames = handle.poll();
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].coords.len(), 66);
        let vis = frames[0].visibility.as_ref().unwrap();
        assert_eq!(vis.len(), 33);
        // First joint should have high visibility
        assert!((vis[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_with_visibility_increments_count() {
        let (receiver, mut handle) = OscReceiver::new(6448, "/wek/inputs");

        receiver
            .event_tx
            .send(ReceiverEvent::FrameWithVisibility(
                vec![0.0; 66],
                vec![1.0; 33],
            ))
            .unwrap();

        handle.poll();
        assert_eq!(handle.state.frame_count, 1);
        assert_eq!(handle.state.status, ConnectionStatus::Receiving);
    }
}
