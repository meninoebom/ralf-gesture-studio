//! Recognition engine for gesture matching.
//!
//! This module contains the core algorithms for gesture recognition,
//! using a Wekinator-style DTW approach.
//!
//! ## Algorithm (Wekinator-Style)
//!
//! The recognizer is modeled after Wekinator's DtwModel.java:
//! - Compare against all training examples (not prototypes)
//! - Try multiple candidate window sizes based on example lengths
//! - Simple threshold check: distance < threshold = match
//! - Explicit "no match" state when nothing is close enough
//!
//! Reference: `fiebrink1/wekinator` - `src/wekimini/learning/dtw/DtwModel.java`

pub mod dtw;
pub mod buffer;
pub mod recognizer;
pub mod training;
pub mod statistics;
pub mod diagnostics;

// Core types used by the app
pub use recognizer::{Recognizer, HitLog, RecognitionConfig};
pub use training::{TrainingSession, TrainingConfig, SessionState};
pub use statistics::compute_threshold_stats;
pub use diagnostics::{DiagnosticLogger, DiagnosticEvent, GestureDiag};

// Additional exports available for future use
#[allow(unused_imports)]
pub use statistics::{compute_threshold_stats_downsampled, ThresholdStats};

// Re-export for future use
#[allow(unused_imports)]
pub use buffer::RecordingSession;

// Re-export DTW primitives for external consumers
#[allow(unused_imports)]
pub use dtw::{
    Frame,
    Sequence,
    euclidean_distance,
    dtw_distance,
    dtw_distance_normalized,
    // Constrained variants (available for future optimization)
    dtw_distance_constrained,
    dtw_distance_constrained_normalized,
    find_best_match,
    // Motion analysis utilities
    motion_energy,
    average_motion_energy,
    is_active,
    compute_prototype,
};

#[allow(unused_imports)]
pub use buffer::{FrameBuffer, TimestampedFrame};

#[allow(unused_imports)]
pub use recognizer::{GestureState, RecognitionResult, HitLogEntry, RecognitionState, StateTransition, GestureStateTransition};
