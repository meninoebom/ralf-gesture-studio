mod receiver;
mod sender;

pub use receiver::{ConnectionStatus, OscReceiver, OscReceiverHandle};
pub use sender::{OscSender, SenderStatus};

// Re-export for potential external consumers
#[allow(unused_imports)]
pub use receiver::ReceiverEvent;
