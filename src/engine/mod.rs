//! Recognition engine for gesture matching.
//!
//! This module contains the core algorithms for gesture recognition,
//! using a Wekinator-style DTW approach with VAD-style state machine
//! and three-layer echo defense.
//!
//! ## Architecture
//!
//! - `dtw` - Dynamic Time Warping with Sakoe-Chiba band, early abandoning, LB_Keogh
//! - `recognizer` - Real-time recognition state machine
//! - `training` - Training session with audio cues
//! - `statistics` - Statistical threshold computation (μ+σ)
//! - `diagnostics` - Diagnostic logging for analysis

pub mod dtw;
pub mod recognizer;
pub mod training;
pub mod statistics;
pub mod diagnostics;

// Core types used by the app
pub use recognizer::{Recognizer, HitLog, RecognitionConfig};
pub use training::{TrainingSession, TrainingConfig, SessionState};
pub use statistics::compute_threshold_stats;
pub use diagnostics::{DiagnosticLogger, DiagnosticEvent, GestureDiag};
