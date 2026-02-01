//! Recognition engine for gesture matching.
//!
//! This module contains the core algorithms for gesture recognition,
//! using a Wekinator-style DTW approach with VAD-style state machine
//! and two-layer echo defense (safety valve + global cooldown).
//!
//! ## Architecture
//!
//! - `dtw` - Dynamic Time Warping with Sakoe-Chiba band, early abandoning, LB_Keogh
//! - `preprocess` - Frame preprocessing pipeline (hip centering, scale normalization, velocity)
//! - `recognizer` - Real-time recognition state machine
//! - `training` - Training session with audio cues
//! - `statistics` - Statistical threshold computation (μ+σ)
//! - `diagnostics` - Diagnostic logging for analysis

pub mod diagnostics;
pub mod dtw;
pub mod preprocess;
pub mod recognizer;
pub mod statistics;
pub mod training;

// Core types used by the app
pub use diagnostics::{DiagnosticEvent, DiagnosticLogger, GestureDiag};
pub use preprocess::Preprocessor;
pub use recognizer::{HitLog, RecognitionConfig, Recognizer};
pub use statistics::compute_threshold_stats;
pub use training::{SessionState, TrainingConfig, TrainingSession};
