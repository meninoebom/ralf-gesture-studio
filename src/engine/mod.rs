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
//! - `augmentation` - Data augmentation (temporal stretch, spatial jitter, horizontal mirror)
//! - `quality` - Example quality assessment (TooShort, TooStill, Outlier)
//! - `recognizer` - Real-time recognition state machine
//! - `training` - Training session with audio cues
//! - `statistics` - Statistical threshold computation (μ+σ)
//! - `diagnostics` - Diagnostic logging for analysis
//! - `weighting` - Variance-based joint weighting for DTW

pub mod augmentation;
pub mod diagnostics;
pub mod dtw;
pub mod preprocess;
pub mod quality;
pub mod recognizer;
pub mod statistics;
pub mod training;
pub mod weighting;

// Core types used by the app
pub use augmentation::generate_augmented;
pub use diagnostics::{DiagnosticEvent, DiagnosticLogger, GestureDiag};
pub use preprocess::Preprocessor;
pub use quality::assess_example;
pub use recognizer::{HitLog, RecognitionConfig, Recognizer};
pub use statistics::compute_threshold_stats;
pub use training::{SessionState, TrainingConfig, TrainingSession};
pub use weighting::compute_joint_weights;
