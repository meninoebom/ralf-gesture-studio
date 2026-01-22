//! Recognition engine for gesture matching.
//!
//! This module contains the core algorithms for gesture recognition,
//! including Dynamic Time Warping (DTW) for comparing gesture sequences.

pub mod dtw;

pub use dtw::{
    Frame,
    Sequence,
    euclidean_distance,
    dtw_distance,
    dtw_distance_normalized,
    find_best_match,
};
