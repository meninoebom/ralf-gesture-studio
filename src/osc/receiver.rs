use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, unbounded};
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
    /// A frame of data was received
    Frame(Vec<f32>),
    /// Connection status changed
    StatusChanged(ConnectionStatus),
    /// An error occurred
    Error(String),
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
    pub fn poll(&mut self) -> Vec<Vec<f32>> {
        let mut frames = Vec::new();

        while let Ok(event) = self.event_rx.try_recv() {
            match event {
                ReceiverEvent::Frame(data) => {
                    self.state.frame_count += 1;
                    self.state.last_frame_time = Some(Instant::now());
                    self.state.status = ConnectionStatus::Receiving;
                    frames.push(data);
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
        self.state.last_frame_time.map(|t| t.elapsed().as_millis() as u64)
    }

    /// Signal the receiver to shut down
    #[allow(dead_code)]
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::SeqCst);
    }
}

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
                let _ = self.event_tx.send(ReceiverEvent::StatusChanged(ConnectionStatus::Listening));
                s
            }
            Err(e) => {
                let _ = self.event_tx.send(ReceiverEvent::Error(format!("Failed to bind to {}: {}", bind_addr, e)));
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
                tokio::time::Duration::from_millis(100),
                socket.recv_from(&mut buf)
            ).await;

            match recv_result {
                Ok(Ok((size, _addr))) => {
                    // Parse the OSC packet
                    if let Ok(packet) = rosc::decoder::decode_udp(&buf[..size]) {
                        self.handle_packet(&packet.1);
                    }
                }
                Ok(Err(e)) => {
                    let _ = self.event_tx.send(ReceiverEvent::Error(format!("Receive error: {}", e)));
                }
                Err(_) => {
                    // Timeout, just continue to check shutdown flag
                }
            }
        }

        let _ = self.event_tx.send(ReceiverEvent::StatusChanged(ConnectionStatus::Stopped));
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
        // Check if this message matches our address filter
        if msg.addr != self.address_filter {
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
            let _ = self.event_tx.send(ReceiverEvent::Frame(floats));
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
        receiver.event_tx.send(ReceiverEvent::StatusChanged(ConnectionStatus::Listening)).unwrap();

        // Poll should update state
        handle.poll();
        assert_eq!(handle.state.status, ConnectionStatus::Listening);
    }

    #[test]
    fn test_frame_count_increments() {
        let (receiver, mut handle) = OscReceiver::new(6448, "/wek/inputs");

        // Simulate receiving frames
        receiver.event_tx.send(ReceiverEvent::Frame(vec![1.0, 2.0, 3.0])).unwrap();
        receiver.event_tx.send(ReceiverEvent::Frame(vec![4.0, 5.0, 6.0])).unwrap();

        handle.poll();
        assert_eq!(handle.state.frame_count, 2);
        assert_eq!(handle.state.status, ConnectionStatus::Receiving);
    }
}
