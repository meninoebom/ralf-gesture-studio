//! Recognition engine for gesture matching.
//!
//! This module contains the core algorithms for gesture recognition,
//! including Dynamic Time Warping (DTW) for comparing gesture sequences.

pub mod dtw;
pub mod buffer;
pub mod recognizer;
pub mod training;

// Core types used by the app
pub use recognizer::{Recognizer, HitLog, RecognitionConfig};
pub use training::{TrainingSession, TrainingConfig, SessionState};

// Re-export for future use
#[allow(unused_imports)]
pub use buffer::RecordingSession;

// Re-export for external consumers and future use
#[allow(unused_imports)]
pub use dtw::{
    Frame,
    Sequence,
    euclidean_distance,
    dtw_distance,
    dtw_distance_normalized,
    find_best_match,
};

#[allow(unused_imports)]
pub use buffer::{FrameBuffer, TimestampedFrame};

#[allow(unused_imports)]
pub use recognizer::{GestureState, RecognitionResult, HitLogEntry};
