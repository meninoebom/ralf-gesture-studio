use std::net::UdpSocket;
use std::time::Instant;

use rosc::{OscMessage, OscPacket, OscType, encoder};

/// Status of the OSC sender
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SenderStatus {
    /// Ready to send
    Ready,
    /// Recently sent a message
    Sent,
    /// Error state
    Error,
}

/// State for the OSC sender (shared with GUI)
pub struct OscSenderState {
    pub status: SenderStatus,
    pub last_send_time: Option<Instant>,
    pub send_count: u64,
    pub error_message: Option<String>,
}

impl Default for OscSenderState {
    fn default() -> Self {
        Self {
            status: SenderStatus::Ready,
            last_send_time: None,
            send_count: 0,
            error_message: None,
        }
    }
}

/// OSC Sender for sending hit messages
pub struct OscSender {
    host: String,
    port: u16,
    socket: Option<UdpSocket>,
    pub state: OscSenderState,
}

impl OscSender {
    /// Create a new OSC sender
    pub fn new(host: &str, port: u16) -> Self {
        Self {
            host: host.to_string(),
            port,
            socket: None,
            state: OscSenderState::default(),
        }
    }

    /// Initialize the socket (lazy initialization)
    fn ensure_socket(&mut self) -> Result<(), String> {
        if self.socket.is_none() {
            match UdpSocket::bind("0.0.0.0:0") {
                Ok(socket) => {
                    self.socket = Some(socket);
                    self.state.status = SenderStatus::Ready;
                    self.state.error_message = None;
                }
                Err(e) => {
                    self.state.status = SenderStatus::Error;
                    self.state.error_message = Some(format!("Failed to create socket: {}", e));
                    return Err(self.state.error_message.clone().unwrap());
                }
            }
        }
        Ok(())
    }

    /// Send an OSC message to the configured host/port
    pub fn send_message(&mut self, address: &str, args: Vec<OscType>) -> Result<(), String> {
        self.ensure_socket()?;

        let msg = OscMessage {
            addr: address.to_string(),
            args,
        };

        let packet = OscPacket::Message(msg);
        let encoded = encoder::encode(&packet)
            .map_err(|e| format!("Failed to encode OSC message: {:?}", e))?;

        let target = format!("{}:{}", self.host, self.port);

        if let Some(ref socket) = self.socket {
            match socket.send_to(&encoded, &target) {
                Ok(_) => {
                    self.state.status = SenderStatus::Sent;
                    self.state.last_send_time = Some(Instant::now());
                    self.state.send_count += 1;
                    self.state.error_message = None;
                    Ok(())
                }
                Err(e) => {
                    self.state.status = SenderStatus::Error;
                    self.state.error_message = Some(format!("Send failed: {}", e));
                    Err(self.state.error_message.clone().unwrap())
                }
            }
        } else {
            Err("Socket not initialized".to_string())
        }
    }

    /// Send a hit message for a gesture (sends a bang/trigger)
    pub fn send_hit(&mut self, address: &str) -> Result<(), String> {
        // Send a "bang" - just the address with a 1.0 float to indicate trigger
        self.send_message(address, vec![OscType::Float(1.0)])
    }

    /// Get time since last send in milliseconds
    pub fn ms_since_last_send(&self) -> Option<u64> {
        self.state.last_send_time.map(|t| t.elapsed().as_millis() as u64)
    }

    /// Update the target host
    #[allow(dead_code)]
    pub fn set_host(&mut self, host: &str) {
        self.host = host.to_string();
    }

    /// Update the target port
    #[allow(dead_code)]
    pub fn set_port(&mut self, port: u16) {
        self.port = port;
    }

    /// Get current host
    #[allow(dead_code)]
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Get current port
    #[allow(dead_code)]
    pub fn port(&self) -> u16 {
        self.port
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sender_creation() {
        let sender = OscSender::new("localhost", 12000);
        assert_eq!(sender.host(), "localhost");
        assert_eq!(sender.port(), 12000);
        assert_eq!(sender.state.status, SenderStatus::Ready);
        assert_eq!(sender.state.send_count, 0);
    }

    #[test]
    fn test_sender_config_update() {
        let mut sender = OscSender::new("localhost", 12000);
        sender.set_host("192.168.1.100");
        sender.set_port(9000);
        assert_eq!(sender.host(), "192.168.1.100");
        assert_eq!(sender.port(), 9000);
    }

    #[test]
    fn test_send_increments_count() {
        let mut sender = OscSender::new("127.0.0.1", 12000);
        // This will actually send UDP packets, but to localhost it's safe
        let _ = sender.send_hit("/test/hit");
        assert_eq!(sender.state.send_count, 1);
        assert!(sender.state.last_send_time.is_some());
    }

    #[test]
    fn test_ms_since_last_send() {
        let mut sender = OscSender::new("127.0.0.1", 12000);
        assert!(sender.ms_since_last_send().is_none());

        let _ = sender.send_hit("/test");
        // Should have a value now
        let ms = sender.ms_since_last_send();
        assert!(ms.is_some());
        assert!(ms.unwrap() < 1000); // Should be very recent
    }
}
