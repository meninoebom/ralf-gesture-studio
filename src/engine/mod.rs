//! Recognition engine for gesture matching.
//!
//! This module contains the core algorithms for gesture recognition,
//! including Dynamic Time Warping (DTW) for comparing gesture sequences.

pub mod dtw;
pub mod buffer;
pub mod recognizer;

// Core types used by the app
pub use buffer::RecordingSession;
pub use recognizer::{Recognizer, HitLog};

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
