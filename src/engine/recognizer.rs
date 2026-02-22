//! Real-time gesture recognizer with VAD-style state machine.
//!
//! Based on research from GRT, Wekinator, and speech recognition (VAD):
//!
//! **Key patterns from speech recognition**:
//! - Frame accumulation: Require N consecutive frames (prevents noise spikes)
//! - Hangover: Stay in recovery state for M ms (prevents echo)
//!
//! **State Machine**:
//! ```text
//! IDLE → BUILDING → PEAK (fire!) → RECOVERY → IDLE
//!   ↑       ↓         ↓              ↓
//!   └───────┴─────────┴──────────────┘ (exit conditions)
//! ```
//!
//! References:
//! - GRT: nickgillian/grt - DTW.cpp
//! - Wekinator: fiebrink1/wekinator - DtwModel.java
//! - CMU Sphinx VAD: Frame accumulation + hangover

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::dtw::{
    compute_lb_envelope, dtw_distance, dtw_distance_visibility_weighted, dtw_distance_with_abandon,
    lb_keogh, visibility_to_dimension_weights, Frame, LBEnvelope, Sequence,
};
use super::weighting::apply_weights_to_sequence;

/// Number of distance samples to keep for slope checking
const DISTANCE_HISTORY_SIZE: usize = 3;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for gesture recognition
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    /// Cooldown between hits for same gesture (ms) - backup protection
    pub cooldown_ms: u64,

    // --- VAD-style parameters ---
    /// Hysteresis factor for entry (1.0 = trained threshold)
    /// Distance must be below threshold × threshold_high_factor to enter Building
    pub threshold_high_factor: f32,

    /// Number of consecutive frames below threshold required to fire
    /// (at ~15Hz DTW rate, 3 frames = ~200ms of confirmation)
    pub frames_to_fire: usize,

    /// Maximum time in Recovery before forcing re-arm (safety valve, ms).
    /// Prevents permanent stuck state when resting distance < threshold (e.g., jump).
    pub max_recovery_ms: u64,

    // --- Global echo suppression (NMS) ---
    /// Global cooldown after ANY gesture fires (ms).
    /// Blocks all gestures from entering Building during this period.
    /// Prevents cross-gesture round-robin echo chains.
    /// (Temporal equivalent of Non-Maximum Suppression in object detection)
    pub global_cooldown_ms: u64,

    // --- DTW optimization ---
    /// Sakoe-Chiba band as fraction of sequence length (e.g., 0.15 = 15%)
    /// Limits warping path to diagonal band, reducing O(N²) to O(N×B)
    /// Also prevents pathological warping (unrealistic time stretching)
    /// Set to 0.0 to disable (use unconstrained DTW)
    pub sakoe_chiba_band: f32,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            cooldown_ms: 500,
            // Entry at 100% of threshold
            threshold_high_factor: 1.0,
            // Frame accumulation: require 2 consecutive frames (~133ms at 15Hz DTW)
            frames_to_fire: 2,
            // Safety valve: force re-arm after 5s (prevents stuck state for jump)
            max_recovery_ms: 5000,
            // Global cooldown: 1500ms after any gesture fires, block all from Building
            global_cooldown_ms: 1000,
            // Sakoe-Chiba band: 15% of sequence length (recommended for gesture recognition)
            sakoe_chiba_band: 0.15,
        }
    }
}

// ============================================================================
// Recognition State Machine
// ============================================================================

/// State of the recognition state machine (per-gesture)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionState {
    /// Idle - waiting for gesture, ready to detect
    Idle,
    /// Building - distance below entry threshold, accumulating frames
    Building,
    /// Peak - gesture detected, fire OSC
    Peak,
    /// Recovery - hangover period after gesture, blocking new detections
    Recovery,
}

/// Information about a state transition (for diagnostic logging)
#[derive(Debug, Clone)]
pub struct StateTransition {
    pub from_state: RecognitionState,
    pub to_state: RecognitionState,
    pub frames_in_prev_state: usize,
    pub reason: &'static str,
}

/// Result of processing a frame through the state machine
#[derive(Debug, Clone)]
pub struct StateMachineResult {
    /// Whether to fire a hit
    pub should_fire: bool,
    /// State transition that occurred (if any)
    pub transition: Option<StateTransition>,
}

impl std::fmt::Display for RecognitionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RecognitionState::Idle => write!(f, "Idle"),
            RecognitionState::Building => write!(f, "Building"),
            RecognitionState::Peak => write!(f, "Peak"),
            RecognitionState::Recovery => write!(f, "Recovery"),
        }
    }
}

// ============================================================================
// Recognition Result
// ============================================================================

/// Result of processing a frame
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    pub gesture_id: Option<u32>,
    pub gesture_name: Option<String>,
    pub distance: f32,
}

// ============================================================================
// Gesture State
// ============================================================================

/// State for one gesture
#[derive(Debug)]
pub struct GestureState {
    pub id: u32,
    pub name: String,
    pub osc_address: String,
    pub threshold: f32,
    examples: Vec<Sequence>,
    /// Precomputed LB_Keogh envelopes for each example (computed on start)
    lb_envelopes: Vec<LBEnvelope>,
    pub current_distance: Option<f32>,
    pub last_hit_time: Option<Instant>,

    // --- VAD state machine ---
    /// Current state in the recognition state machine
    state: RecognitionState,
    /// Count of consecutive frames below entry threshold (used in Building state)
    frames_below_threshold: usize,
    /// When we entered Recovery state (for hangover timing)
    recovery_start: Option<Instant>,
    /// History of recent distance values (for slope detection)
    distance_history: VecDeque<f32>,
    /// Per-dimension weights from joint variance (None = no weighting)
    weights: Option<Vec<f32>>,
    /// Whether consensus gate is active for this gesture
    consensus_enabled: bool,
    /// Minimum fraction of examples that must agree (default 0.5)
    consensus_threshold: f32,
    /// Count of consecutive frames above threshold in Recovery state (for distance-based exit)
    recovery_frames_above: usize,
}

impl GestureState {
    pub fn new(id: u32, name: &str, osc_address: &str, threshold: f32) -> Self {
        Self {
            id,
            name: name.to_string(),
            osc_address: osc_address.to_string(),
            threshold,
            examples: Vec::new(),
            lb_envelopes: Vec::new(),
            current_distance: None,
            last_hit_time: None,
            // VAD state machine
            state: RecognitionState::Idle,
            frames_below_threshold: 0,
            recovery_start: None,
            distance_history: VecDeque::with_capacity(DISTANCE_HISTORY_SIZE),
            // Joint weighting + consensus (Phase 3)
            weights: None,
            consensus_enabled: false,
            consensus_threshold: 0.5,
            recovery_frames_above: 0,
        }
    }

    /// Set per-dimension weights for this gesture (from joint variance computation)
    pub fn set_weights(&mut self, weights: Option<Vec<f32>>) {
        self.weights = weights;
    }

    /// Get the per-dimension weights (if any)
    #[allow(dead_code)]
    pub fn weights(&self) -> Option<&[f32]> {
        self.weights.as_deref()
    }

    /// Set consensus scoring configuration
    pub fn set_consensus(&mut self, enabled: bool, threshold: f32) {
        self.consensus_enabled = enabled;
        self.consensus_threshold = threshold;
    }

    pub fn add_example(&mut self, example: Sequence) {
        self.examples.push(example);
        // Clear envelopes - they'll be recomputed on start()
        self.lb_envelopes.clear();
    }

    /// Get the LB_Keogh envelopes (precomputed for recognition)
    pub fn lb_envelopes(&self) -> &[LBEnvelope] {
        &self.lb_envelopes
    }

    /// Compute LB_Keogh envelopes for all examples.
    /// Called at recognition start to enable fast pruning.
    pub fn compute_envelopes(&mut self, band_width: usize, downsample: usize) {
        self.lb_envelopes.clear();
        for example in &self.examples {
            // Downsample example before computing envelope
            let example_ds: Sequence = example.iter().step_by(downsample).cloned().collect();
            let envelope = compute_lb_envelope(&example_ds, band_width);
            self.lb_envelopes.push(envelope);
        }
    }

    #[cfg(test)]
    pub fn has_examples(&self) -> bool {
        !self.examples.is_empty()
    }

    pub fn examples(&self) -> &[Sequence] {
        &self.examples
    }

    pub fn example_count(&self) -> usize {
        self.examples.len()
    }

    pub fn in_cooldown(&self, cooldown: Duration) -> bool {
        self.last_hit_time
            .map(|t| t.elapsed() < cooldown)
            .unwrap_or(false)
    }

    pub fn record_hit(&mut self) {
        self.last_hit_time = Some(Instant::now());
    }

    /// Get current recognition state
    #[cfg(test)]
    pub fn recognition_state(&self) -> RecognitionState {
        self.state
    }

    /// Reset state machine to Idle
    fn reset_to_idle(&mut self) {
        self.state = RecognitionState::Idle;
        self.frames_below_threshold = 0;
        self.recovery_start = None;
        self.recovery_frames_above = 0;
        // Note: We don't clear distance_history here - it's useful for the next detection
    }

    /// Check if distance is actively falling (negative slope).
    /// Returns true only when distance is decreasing — indicating the user
    /// is approaching a gesture. Flat or rising distances are rejected to
    /// prevent false triggers from resting poses that happen to be below threshold.
    ///
    /// Note: record_distance() is called before this, so the current distance
    /// is already at the back of the history. We compare against the second-to-last.
    fn is_distance_falling(&self, current: f32) -> bool {
        if self.distance_history.len() < 2 {
            return false; // Not enough history — wait for data rather than guess
        }

        // Get the previous distance (second-to-last, since current is already at back)
        let prev = self.distance_history.iter().rev().nth(1).copied().unwrap_or(current);

        // Require strictly negative slope (distance is decreasing)
        current < prev
    }

    /// Record a distance value in history (for slope detection)
    fn record_distance(&mut self, distance: f32) {
        if self.distance_history.len() >= DISTANCE_HISTORY_SIZE {
            self.distance_history.pop_front();
        }
        self.distance_history.push_back(distance);
    }

    /// Process a distance value through the state machine.
    /// Returns result with fire signal and any state transition that occurred.
    ///
    /// `in_global_cooldown`: If true, blocks Idle→Building entry (NMS suppression).
    fn process_state_machine(
        &mut self,
        distance: f32,
        config: &RecognitionConfig,
        in_global_cooldown: bool,
    ) -> StateMachineResult {
        // Record distance for slope detection (before state processing)
        self.record_distance(distance);

        let entry_threshold = self.threshold * config.threshold_high_factor;
        let prev_state = self.state;
        let prev_frames = self.frames_below_threshold;

        let (should_fire, new_state, reason) = match self.state {
            RecognitionState::Idle => {
                if in_global_cooldown {
                    // Global cooldown active - suppress all new detections (NMS)
                    (false, None, "")
                } else if distance < entry_threshold && self.is_distance_falling(distance) {
                    // Start building
                    self.state = RecognitionState::Building;
                    self.frames_below_threshold = 1;

                    // Check if we already have enough frames (handles frames_to_fire = 1)
                    if self.frames_below_threshold >= config.frames_to_fire {
                        self.state = RecognitionState::Peak;
                        self.record_hit();
                        (
                            true,
                            Some(RecognitionState::Peak),
                            "below_threshold_instant_fire",
                        )
                    } else {
                        (
                            false,
                            Some(RecognitionState::Building),
                            "below_threshold_falling",
                        )
                    }
                } else if distance < entry_threshold {
                    // Below threshold but not falling - likely noise/echo
                    (false, None, "") // Stay in Idle
                } else {
                    (false, None, "")
                }
            }

            RecognitionState::Building => {
                if distance < entry_threshold {
                    // Still below threshold, accumulate
                    self.frames_below_threshold += 1;

                    // Check if we've accumulated enough frames
                    if self.frames_below_threshold >= config.frames_to_fire {
                        // Transition to Peak - FIRE!
                        self.state = RecognitionState::Peak;
                        self.record_hit();
                        (true, Some(RecognitionState::Peak), "accumulated_frames")
                    } else {
                        (false, None, "") // Still building, no transition
                    }
                } else {
                    // Distance rose above threshold, reset to Idle
                    self.reset_to_idle();
                    (false, Some(RecognitionState::Idle), "above_threshold")
                }
            }

            RecognitionState::Peak => {
                // Immediately transition to Recovery after firing
                self.state = RecognitionState::Recovery;
                self.recovery_start = Some(Instant::now());
                (false, Some(RecognitionState::Recovery), "post_fire")
            }

            RecognitionState::Recovery => {
                // Time-based recovery: re-arm after cooldown expires.
                // Distance-based recovery was removed because resting distances
                // often stay below threshold (standing still matches still portions
                // of training examples), creating a death spiral where recovery
                // never completes and the safety valve fires false positives.
                let elapsed = self
                    .recovery_start
                    .map(|t| t.elapsed())
                    .unwrap_or(Duration::ZERO);
                let recovery_duration = Duration::from_millis(config.cooldown_ms);

                if elapsed >= recovery_duration {
                    self.reset_to_idle();
                    (false, Some(RecognitionState::Idle), "cooldown_complete")
                } else {
                    (false, None, "")
                }
            }
        };

        // Build transition info if state changed
        let transition = new_state.map(|to_state| StateTransition {
            from_state: prev_state,
            to_state,
            frames_in_prev_state: prev_frames,
            reason,
        });

        StateMachineResult {
            should_fire,
            transition,
        }
    }
}

// ============================================================================
// Frame Buffer
// ============================================================================

/// Simple sliding window buffer
#[derive(Debug)]
pub struct FrameBuffer {
    frames: VecDeque<Frame>,
    max_size: usize,
}

impl FrameBuffer {
    pub fn new(max_size: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    pub fn push(&mut self, frame: Frame) {
        if self.frames.len() >= self.max_size {
            self.frames.pop_front();
        }
        self.frames.push_back(frame);
    }

    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns the dimension count of frames in the buffer, or None if empty.
    pub fn frame_dimensions(&self) -> Option<usize> {
        self.frames.front().map(|f| f.len())
    }

    /// Get the most recent N frames as a sequence
    pub fn recent(&self, n: usize) -> Sequence {
        let start = self.frames.len().saturating_sub(n);
        self.frames.iter().skip(start).cloned().collect()
    }

    /// Clear all frames from the buffer.
    /// Used after a hit fires to prevent re-triggering from stale gesture data.
    pub fn clear(&mut self) {
        self.frames.clear();
    }
}

// ============================================================================
// Recognizer
// ============================================================================

/// A state transition with gesture context (for logging)
#[derive(Debug, Clone)]
pub struct GestureStateTransition {
    pub gesture_name: String,
    pub transition: StateTransition,
    pub distance: f32,
    pub threshold: f32,
}

/// The recognizer with VAD-style state machine
#[derive(Debug)]
pub struct Recognizer {
    pub buffer: FrameBuffer,
    gestures: Vec<GestureState>,
    config: RecognitionConfig,
    active: bool,
    /// Window size for matching (derived from first example)
    window_size: usize,
    /// Frame counter for skipping (DTW is expensive)
    frame_count: usize,
    /// How often to run DTW (every Nth frame)
    dtw_skip: usize,
    /// Downsample factor for DTW (compare at 15fps instead of 60fps)
    downsample: usize,
    /// Pending state transitions to be logged (cleared after retrieval)
    pending_transitions: Vec<GestureStateTransition>,
    /// When any gesture last fired (for global cooldown / NMS)
    last_any_hit_time: Option<Instant>,
    /// Current per-dimension visibility weights (from most recent frame).
    /// None = no visibility data (all joints equally weighted).
    current_visibility_weights: Option<Vec<f32>>,
}

impl Recognizer {
    #[cfg(test)]
    pub fn new(buffer_size: usize, window_size: usize) -> Self {
        Self::with_config(buffer_size, window_size, RecognitionConfig::default())
    }

    pub fn with_config(buffer_size: usize, window_size: usize, config: RecognitionConfig) -> Self {
        Self {
            buffer: FrameBuffer::new(buffer_size),
            gestures: Vec::new(),
            config,
            active: false,
            window_size,
            frame_count: 0,
            dtw_skip: 4,   // Compute DTW every 4th frame (15Hz @ 60fps input)
            downsample: 4, // Compare at 15fps
            pending_transitions: Vec::new(),
            last_any_hit_time: None,
            current_visibility_weights: None,
        }
    }

    /// Take pending state transitions (for diagnostic logging).
    /// Returns transitions and clears the internal list.
    pub fn take_transitions(&mut self) -> Vec<GestureStateTransition> {
        std::mem::take(&mut self.pending_transitions)
    }

    /// Downsample a sequence by taking every Nth frame
    fn downsample_seq(seq: &Sequence, factor: usize) -> Sequence {
        if factor <= 1 {
            return seq.clone();
        }
        seq.iter().step_by(factor).cloned().collect()
    }

    /// Find the best (minimum) distance to any example in the list
    /// Uses LB_Keogh pruning and early abandoning to skip computations.
    ///
    /// # Arguments
    /// * `window` - Input sequence (already downsampled)
    /// * `examples` - List of training examples
    /// * `envelopes` - Precomputed LB_Keogh envelopes (same order as examples)
    /// * `downsample` - Downsample factor for examples
    /// * `sakoe_chiba_band` - Band fraction (0.0 = unconstrained, 0.15 = 15% band)
    /// * `best_so_far` - Current best distance (for early abandoning across gestures)
    /// * `visibility_weights` - Optional per-dimension visibility weights
    ///
    /// # Returns
    /// The best distance found, which may be >= best_so_far if no better match exists
    fn find_best_distance(
        window: &Sequence,
        examples: &[Sequence],
        envelopes: &[LBEnvelope],
        downsample: usize,
        sakoe_chiba_band: f32,
        best_so_far: f32,
        visibility_weights: Option<&[f32]>,
    ) -> f32 {
        let mut best = best_so_far;
        for (i, example) in examples.iter().enumerate() {
            // Layer 1: LB_Keogh pruning (O(n) lower bound check)
            // LB_Keogh envelopes were computed without visibility weights,
            // so the lower bound is invalid when weights change per-frame.
            if visibility_weights.is_none() {
                if let Some(envelope) = envelopes.get(i) {
                    let lb = lb_keogh(window, envelope);
                    if lb >= best {
                        continue; // Lower bound already exceeds best, skip DTW
                    }
                }
            }

            // Layer 2: Full DTW with early abandoning
            let example_ds = Self::downsample_seq(example, downsample);
            let dist = if sakoe_chiba_band > 0.0 {
                let max_len = window.len().max(example_ds.len());
                let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;

                if let Some(weights) = visibility_weights {
                    // Use visibility-weighted DTW
                    match dtw_distance_visibility_weighted(
                        window,
                        &example_ds,
                        band_width,
                        best,
                        weights,
                    ) {
                        Some(d) => d,
                        None => continue,
                    }
                } else {
                    // Use standard DTW with early abandoning
                    match dtw_distance_with_abandon(window, &example_ds, band_width, best) {
                        Some(d) => d,
                        None => continue,
                    }
                }
            } else {
                // Use unconstrained DTW (no early abandoning available)
                dtw_distance(window, &example_ds)
            };
            if dist < best {
                best = dist;
            }
        }
        best
    }

    pub fn set_cooldown_ms(&mut self, ms: u64) {
        self.config.cooldown_ms = ms;
    }

    pub fn add_gesture(&mut self, id: u32, name: &str, osc_address: &str, threshold: f32) {
        self.gestures
            .push(GestureState::new(id, name, osc_address, threshold));
    }

    pub fn get_gesture_mut(&mut self, id: u32) -> Option<&mut GestureState> {
        self.gestures.iter_mut().find(|g| g.id == id)
    }

    pub fn get_gesture(&self, id: u32) -> Option<&GestureState> {
        self.gestures.iter().find(|g| g.id == id)
    }

    #[cfg(test)]
    pub fn gestures(&self) -> &[GestureState] {
        &self.gestures
    }

    /// Add a training example. Also updates window_size based on example length.
    pub fn add_example(&mut self, gesture_id: u32, example: Sequence) -> bool {
        // Update window size to match example length (use first example's length)
        if self.window_size == 0 || self.all_examples_count() == 0 {
            self.window_size = example.len();
        }

        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.add_example(example);
            true
        } else {
            false
        }
    }

    fn all_examples_count(&self) -> usize {
        self.gestures.iter().map(|g| g.example_count()).sum()
    }

    pub fn start(&mut self) {
        self.active = true;
        self.last_any_hit_time = None;

        // Compute LB_Keogh envelopes for all gestures
        // Band width is computed from config (default 15%)
        let max_len = self.window_size / self.downsample;
        let band_width = ((max_len as f32) * self.config.sakoe_chiba_band).ceil() as usize;

        for gesture in &mut self.gestures {
            gesture.current_distance = None;
            gesture.reset_to_idle();
            // Precompute envelopes for fast LB_Keogh pruning
            gesture.compute_envelopes(band_width, self.downsample);
        }
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Compute DTW distances for all gestures against the current window.
    ///
    /// Uses LB_Keogh pruning and early abandoning across gestures for efficiency.
    /// Applies per-gesture joint weights to the window before DTW comparison.
    /// Returns a Vec of (gesture_index, best_distance, consensus) for gestures with examples.
    /// Consensus is 1.0 when consensus scoring is disabled for a gesture.
    fn compute_distances(&self, window: &Sequence) -> Vec<(usize, f32, f32)> {
        let mut distances = Vec::new();
        let mut best_so_far = f32::MAX;
        let sakoe_chiba_band = self.config.sakoe_chiba_band;
        let vis_weights = self.current_visibility_weights.as_deref();

        // When visibility weights are active, distances are systematically lower
        // because each dimension is scaled by weight ∈ [0,1]. To keep distances
        // comparable to unweighted training thresholds, we normalize by the mean
        // visibility weight (dividing restores the original distance scale).
        let visibility_normalizer = vis_weights
            .filter(|w| !w.is_empty())
            .map(|w| {
                let mean = w.iter().sum::<f32>() / w.len() as f32;
                if mean > 1e-6 { mean } else { 1.0 }
            })
            .unwrap_or(1.0);

        for (idx, gesture) in self.gestures.iter().enumerate() {
            if gesture.examples().is_empty() {
                continue;
            }

            // Apply per-gesture joint weights to the window
            let weighted_window;
            let window_ref = match &gesture.weights {
                Some(w) => {
                    weighted_window = apply_weights_to_sequence(window, w);
                    &weighted_window
                }
                None => window,
            };

            let (examples, envelopes) = (gesture.examples(), gesture.lb_envelopes());

            // Early abandoning needs `best_so_far` in the same domain as raw
            // DTW distances. Visibility weighting scales each dimension by w_i,
            // so raw distances are smaller by ~mean(w). Convert best_so_far from
            // normalized space → weighted space, then convert result back.
            let raw_dist = Self::find_best_distance(
                window_ref,
                examples,
                envelopes,
                self.downsample,
                sakoe_chiba_band,
                best_so_far * visibility_normalizer,
                vis_weights,
            );
            // Normalize visibility-weighted distance to unweighted scale
            let dist = raw_dist / visibility_normalizer;
            if dist < best_so_far {
                best_so_far = dist;
            }

            // Compute consensus if enabled and distance is below threshold
            let consensus = if gesture.consensus_enabled && dist < gesture.threshold {
                Self::compute_consensus(
                    window_ref,
                    examples,
                    envelopes,
                    self.downsample,
                    sakoe_chiba_band,
                    gesture.threshold,
                )
            } else if gesture.consensus_enabled {
                0.0 // Above threshold, consensus is 0
            } else {
                1.0 // Consensus disabled, always passes
            };

            distances.push((idx, dist, consensus));
        }

        distances
    }

    /// Compute the fraction of examples with distance below threshold.
    ///
    /// Uses LB_Keogh to quickly reject examples that are clearly above threshold,
    /// and early abandoning with threshold as cutoff.
    fn compute_consensus(
        window: &Sequence,
        examples: &[Sequence],
        envelopes: &[LBEnvelope],
        downsample: usize,
        sakoe_chiba_band: f32,
        threshold: f32,
    ) -> f32 {
        if examples.is_empty() {
            return 0.0;
        }

        let mut below_count = 0usize;

        for (i, example) in examples.iter().enumerate() {
            // LB_Keogh pruning: if lower bound >= threshold, definitely above
            if let Some(envelope) = envelopes.get(i) {
                let lb = lb_keogh(window, envelope);
                if lb >= threshold {
                    continue;
                }
            }

            // Full DTW with threshold as abandon cutoff
            let example_ds = Self::downsample_seq(example, downsample);
            let below = if sakoe_chiba_band > 0.0 {
                let max_len = window.len().max(example_ds.len());
                let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
                dtw_distance_with_abandon(window, &example_ds, band_width, threshold).is_some()
            } else {
                dtw_distance(window, &example_ds) < threshold
            };

            if below {
                below_count += 1;
            }
        }

        below_count as f32 / examples.len() as f32
    }

    /// Run state machines for all gestures and detect hits.
    ///
    /// Updates gesture distances (for UI), processes the best-matching gesture's
    /// state machine, resets non-best gestures, and returns a hit if one fired.
    /// Applies consensus gate: if consensus scoring is enabled for the best gesture,
    /// the gesture must meet its consensus threshold to enter Building state.
    fn run_state_machines(&mut self, distances: &[(usize, f32, f32)]) -> Option<RecognitionResult> {
        // Check global cooldown (NMS: suppress all detections after any hit)
        let in_global_cooldown = self
            .last_any_hit_time
            .map(|t| t.elapsed() < Duration::from_millis(self.config.global_cooldown_ms))
            .unwrap_or(false);

        // Find the best-matching gesture index
        let best_idx = distances
            .iter()
            .min_by(|(_, a, _), (_, b, _)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _, _)| *i);

        // Check if the best gesture fails the consensus gate
        let consensus_blocked = distances
            .iter()
            .find(|(i, _, _)| Some(*i) == best_idx)
            .map(|(_, _, consensus)| {
                let gesture = &self.gestures[best_idx.unwrap()];
                gesture.consensus_enabled && *consensus < gesture.consensus_threshold
            })
            .unwrap_or(false);

        let mut hit_result: Option<(u32, String, f32)> = None;

        for (idx, gesture) in self.gestures.iter_mut().enumerate() {
            // Find this gesture's distance
            let distance = distances
                .iter()
                .find(|(i, _, _)| *i == idx)
                .map(|(_, d, _)| *d);

            gesture.current_distance = distance;

            if let Some(dist) = distance {
                if Some(idx) == best_idx {
                    // Best-matching gesture: run its state machine
                    // Consensus gate: treat as global cooldown (suppress entry) if consensus fails
                    let suppressed = in_global_cooldown || consensus_blocked;
                    let result = gesture.process_state_machine(dist, &self.config, suppressed);

                    if let Some(transition) = result.transition {
                        self.pending_transitions.push(GestureStateTransition {
                            gesture_name: gesture.name.clone(),
                            transition,
                            distance: dist,
                            threshold: gesture.threshold,
                        });
                    }

                    if result.should_fire {
                        hit_result = Some((gesture.id, gesture.name.clone(), dist));
                    }
                } else if gesture.state == RecognitionState::Building {
                    // Non-best gestures: reset Building to prevent multiple simultaneous builds
                    gesture.reset_to_idle();
                }
            }
        }

        if let Some((id, name, distance)) = hit_result {
            // Set global cooldown timestamp (NMS: suppress all gestures)
            self.last_any_hit_time = Some(Instant::now());

            // Clear the buffer to prevent re-triggering from stale gesture data.
            // The buffer must refill (~1-1.5s) before the next detection can occur.
            // Do NOT reset gesture state machines — Recovery state must persist
            // so the user must complete a full gesture cycle to retrigger.
            self.buffer.clear();
            for gesture in &mut self.gestures {
                gesture.current_distance = None;
            }

            Some(RecognitionResult {
                gesture_id: Some(id),
                gesture_name: Some(name),
                distance,
            })
        } else {
            let best_distance = distances
                .iter()
                .map(|(_, d, _)| *d)
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f32::MAX);

            Some(RecognitionResult {
                gesture_id: None,
                gesture_name: None,
                distance: best_distance,
            })
        }
    }

    /// Process a frame through the VAD-style state machine.
    ///
    /// **Algorithm**:
    /// 1. Buffer frame, skip if not a DTW frame
    /// 2. Compute DTW distance to all gestures
    /// 3. Run state machines (only best-matching gesture advances)
    /// 4. Fire if state machine transitions to Peak
    pub fn process_frame(&mut self, frame: Frame) -> Option<RecognitionResult> {
        // Defense-in-depth: skip frames with mismatched dimensions.
        // Primary validation is in AppState::process_frames().
        if let Some(expected) = self.buffer.frame_dimensions() {
            if frame.len() != expected {
                return None;
            }
        }

        self.buffer.push(frame);
        self.frame_count += 1;

        if !self.active {
            return None;
        }

        // Need window_size frames for DTW
        if self.window_size == 0 || self.buffer.len() < self.window_size {
            return None;
        }

        // Skip frames to reduce CPU load (DTW is expensive)
        if !self.frame_count.is_multiple_of(self.dtw_skip) {
            return Some(RecognitionResult {
                gesture_id: None,
                gesture_name: None,
                distance: f32::MAX,
            });
        }

        // Get current window and downsample for efficient DTW
        let window_full = self.buffer.recent(self.window_size);
        let window = Self::downsample_seq(&window_full, self.downsample);

        let distances = self.compute_distances(&window);
        self.run_state_machines(&distances)
    }

    /// Process a frame with associated per-joint visibility scores.
    ///
    /// Converts visibility to per-dimension weights and uses them during DTW.
    pub fn process_frame_with_visibility(
        &mut self,
        frame: Frame,
        visibility: &[f32],
    ) -> Option<RecognitionResult> {
        // Convert per-joint visibility to per-dimension weights (2 coords per joint for XY)
        self.current_visibility_weights =
            Some(visibility_to_dimension_weights(visibility, 2));
        self.process_frame(frame)
    }

    pub fn current_distances(&self) -> Vec<(u32, String, Option<f32>, f32)> {
        self.gestures
            .iter()
            .map(|g| (g.id, g.name.clone(), g.current_distance, g.threshold))
            .collect()
    }

    pub fn set_threshold(&mut self, gesture_id: u32, threshold: f32) {
        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.threshold = threshold;
        }
    }

    /// Set per-dimension weights for a gesture (from joint variance computation)
    pub fn set_weights(&mut self, gesture_id: u32, weights: Option<Vec<f32>>) {
        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.set_weights(weights);
        }
    }

    /// Set consensus scoring configuration for a gesture
    pub fn set_consensus(&mut self, gesture_id: u32, enabled: bool, threshold: f32) {
        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.set_consensus(enabled, threshold);
        }
    }

    pub fn example_count(&self, gesture_id: u32) -> usize {
        self.get_gesture(gesture_id)
            .map(|g| g.example_count())
            .unwrap_or(0)
    }

    /// Get the current window size (for debugging display)
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Get total example count across all gestures
    pub fn total_example_count(&self) -> usize {
        self.all_examples_count()
    }
}

// ============================================================================
// Hit Log
// ============================================================================

/// Hit log entry
#[derive(Debug, Clone)]
pub struct HitLogEntry {
    pub timestamp: Instant,
    #[allow(dead_code)]
    pub gesture_id: u32,
    pub gesture_name: String,
    #[allow(dead_code)]
    pub distance: f32,
    #[allow(dead_code)]
    pub osc_address: String,
}

/// Rolling log of recent hits
#[derive(Debug)]
pub struct HitLog {
    entries: VecDeque<HitLogEntry>,
    max_entries: usize,
}

impl HitLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(max_entries),
            max_entries,
        }
    }

    pub fn record(
        &mut self,
        gesture_id: u32,
        gesture_name: &str,
        distance: f32,
        osc_address: &str,
    ) {
        if self.entries.len() >= self.max_entries {
            self.entries.pop_front();
        }
        self.entries.push_back(HitLogEntry {
            timestamp: Instant::now(),
            gesture_id,
            gesture_name: gesture_name.to_string(),
            distance,
            osc_address: osc_address.to_string(),
        });
    }

    pub fn recent(&self, count: usize) -> Vec<&HitLogEntry> {
        self.entries.iter().rev().take(count).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gesture_state_creation() {
        let gesture = GestureState::new(1, "wave", "/gesture/1", 15.0);
        assert_eq!(gesture.id, 1);
        assert_eq!(gesture.name, "wave");
        assert_eq!(gesture.threshold, 15.0);
        assert!(!gesture.has_examples());
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
    }

    #[test]
    fn test_gesture_add_example() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 15.0);
        gesture.add_example(vec![vec![1.0], vec![2.0], vec![3.0]]);
        assert!(gesture.has_examples());
        assert_eq!(gesture.example_count(), 1);
    }

    #[test]
    fn test_gesture_cooldown() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 15.0);
        assert!(!gesture.in_cooldown(Duration::from_millis(500)));
        gesture.record_hit();
        assert!(gesture.in_cooldown(Duration::from_millis(500)));
        std::thread::sleep(Duration::from_millis(100));
        assert!(!gesture.in_cooldown(Duration::from_millis(50)));
    }

    /// Helper: prime distance history with a high value so the next low value
    /// registers as "falling" and can enter Building state.
    fn prime_history(gesture: &mut GestureState, config: &RecognitionConfig) {
        gesture.process_state_machine(200.0, &config, false);
        gesture.process_state_machine(200.0, &config, false);
    }

    #[test]
    fn test_state_machine_idle_to_building() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        // Prime with high distance so the drop to 50 is "falling"
        prime_history(&mut gesture, &config);

        // Distance below threshold and falling should transition to Building
        let result = gesture.process_state_machine(50.0, &config, false);
        assert!(!result.should_fire);
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);
        assert_eq!(gesture.frames_below_threshold, 1);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.from_state, RecognitionState::Idle);
        assert_eq!(t.to_state, RecognitionState::Building);
    }

    #[test]
    fn test_state_machine_building_to_peak() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig {
            frames_to_fire: 3,
            ..Default::default()
        };

        prime_history(&mut gesture, &config);

        // Accumulate frames (falling sequence)
        gesture.process_state_machine(50.0, &config, false); // Building, count=1
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);

        gesture.process_state_machine(45.0, &config, false); // Building, count=2
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);

        let result = gesture.process_state_machine(40.0, &config, false); // Peak!
        assert!(result.should_fire);
        assert_eq!(gesture.recognition_state(), RecognitionState::Peak);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Peak);
        assert_eq!(t.reason, "accumulated_frames");
    }

    #[test]
    fn test_state_machine_building_reset_on_rise() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        prime_history(&mut gesture, &config);
        gesture.process_state_machine(50.0, &config, false); // Building (falling from 200)
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);

        // Distance rises above threshold - should reset to Idle
        let result = gesture.process_state_machine(150.0, &config, false);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert_eq!(gesture.frames_below_threshold, 0);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Idle);
        assert_eq!(t.reason, "above_threshold");
    }

    #[test]
    fn test_state_machine_peak_to_recovery() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig {
            frames_to_fire: 1, // Fire on first frame in Building
            ..Default::default()
        };

        prime_history(&mut gesture, &config);

        // Falling below threshold: Idle → Building → Peak (fires)
        let result = gesture.process_state_machine(50.0, &config, false);
        assert!(result.should_fire, "Should fire when frames_to_fire=1");
        assert_eq!(gesture.recognition_state(), RecognitionState::Peak);

        // Next frame should transition Peak → Recovery
        let result = gesture.process_state_machine(50.0, &config, false);
        assert_eq!(gesture.recognition_state(), RecognitionState::Recovery);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.from_state, RecognitionState::Peak);
        assert_eq!(t.to_state, RecognitionState::Recovery);
    }

    #[test]
    fn test_state_machine_recovery_cooldown() {
        // Test time-based recovery: re-arm after cooldown_ms
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig {
            frames_to_fire: 1,
            cooldown_ms: 100, // Short cooldown for test
            ..Default::default()
        };

        prime_history(&mut gesture, &config);

        // Go to Peak and then Recovery (falling entry)
        gesture.process_state_machine(50.0, &config, false); // Peak
        gesture.process_state_machine(50.0, &config, false); // Recovery

        // Distance stays low — doesn't matter, recovery is time-based
        gesture.process_state_machine(50.0, &config, false);

        // Wait less than cooldown
        std::thread::sleep(Duration::from_millis(60));
        gesture.process_state_machine(50.0, &config, false);
        assert_eq!(gesture.recognition_state(), RecognitionState::Recovery);

        // Wait for cooldown to expire
        std::thread::sleep(Duration::from_millis(50)); // Total ~110ms > 100ms

        let result = gesture.process_state_machine(50.0, &config, false);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Idle);
        assert_eq!(t.reason, "cooldown_complete");
    }

    #[test]
    fn test_global_cooldown_blocks_building_entry() {
        // Test that global cooldown (NMS) blocks Idle→Building entry
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        prime_history(&mut gesture, &config);

        // Distance below threshold and falling, but global cooldown active → stay in Idle
        let result = gesture.process_state_machine(50.0, &config, true); // in_global_cooldown=true
        assert!(!result.should_fire);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert!(result.transition.is_none()); // No transition — suppressed by NMS

        // Same distance without global cooldown → enters Building (still falling from 50 to 40)
        let result = gesture.process_state_machine(40.0, &config, false);
        assert!(!result.should_fire);
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);
    }

    #[test]
    fn test_state_machine_slope_check_blocks_flat_entry() {
        // Flat distances below threshold should NOT enter Building
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        // Build history with flat low values
        gesture.process_state_machine(50.0, &config, false);
        gesture.process_state_machine(50.0, &config, false);
        gesture.process_state_machine(50.0, &config, false);

        // Still in Idle — distance is below threshold but not falling
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);

        // Rising distances below threshold should also NOT enter Building
        let mut gesture2 = GestureState::new(2, "wave2", "/gesture/2", 100.0);
        gesture2.process_state_machine(40.0, &config, false);
        gesture2.process_state_machine(50.0, &config, false); // rising
        gesture2.process_state_machine(60.0, &config, false); // still rising
        assert_eq!(gesture2.recognition_state(), RecognitionState::Idle);
    }

    #[test]
    fn test_state_machine_slope_check_allows_falling_entry() {
        // Test that slope check allows entry when distance is falling
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        // Start with high distance, build some history
        gesture.process_state_machine(150.0, &config, false);
        gesture.process_state_machine(120.0, &config, false); // Falling
        gesture.process_state_machine(90.0, &config, false); // Falling, below threshold

        // Should have entered Building when distance dropped below threshold while falling
        // Check: 90 < 100 (threshold) and 90 < 120 (prev) so slope is negative
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);
    }

    #[test]
    fn test_recognizer_creation() {
        let recognizer = Recognizer::new(1000, 100);
        assert!(!recognizer.is_active());
        assert_eq!(recognizer.gestures().len(), 0);
    }

    #[test]
    fn test_recognizer_add_gesture() {
        let mut recognizer = Recognizer::new(1000, 100);
        recognizer.add_gesture(1, "wave", "/gesture/1", 15.0);
        recognizer.add_gesture(2, "jump", "/gesture/2", 20.0);
        assert_eq!(recognizer.gestures().len(), 2);
    }

    #[test]
    fn test_recognizer_add_example() {
        let mut recognizer = Recognizer::new(1000, 100);
        recognizer.add_gesture(1, "wave", "/gesture/1", 15.0);
        let example = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(recognizer.add_example(1, example));
        assert_eq!(recognizer.example_count(1), 1);
    }

    #[test]
    fn test_recognizer_process_frame_inactive() {
        let mut recognizer = Recognizer::new(1000, 10);
        recognizer.add_gesture(1, "wave", "/gesture/1", 15.0);
        let example = vec![vec![1.0]; 10];
        recognizer.add_example(1, example);
        let result = recognizer.process_frame(vec![1.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_recognizer_matches_similar_gesture() {
        // Test that recognizer fires when distance < threshold
        let mut recognizer = Recognizer::with_config(
            100,
            5,
            RecognitionConfig {
                frames_to_fire: 1, // Fire on first frame below threshold
                ..Default::default()
            },
        );

        // Set high threshold so matching frames will be below it
        recognizer.add_gesture(1, "wave", "/gesture/1", 50.0);

        // Add example: frames 1,2,3,4,5
        let example = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Fill buffer with frames close to the example
        let mut hit = None;
        for i in 0..50 {
            // Frames similar to example (should match)
            let frame = vec![(i % 5 + 1) as f32];
            if let Some(result) = recognizer.process_frame(frame) {
                if result.gesture_id.is_some() {
                    hit = Some(result);
                    break;
                }
            }
        }

        assert!(hit.is_some(), "Should detect matching gesture");
        assert_eq!(hit.unwrap().gesture_id, Some(1));
    }

    #[test]
    fn test_hit_log() {
        let mut log = HitLog::new(10);
        log.record(1, "wave", 42.0, "/gesture/1");
        log.record(2, "jump", 55.0, "/gesture/2");
        assert_eq!(log.len(), 2);
        let recent = log.recent(5);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].gesture_name, "jump");
    }

    #[test]
    fn test_hit_log_max_entries() {
        let mut log = HitLog::new(3);
        for i in 0..5 {
            log.record(i, &format!("gesture{}", i), i as f32, "/test");
        }
        assert_eq!(log.len(), 3);
        let recent = log.recent(5);
        assert_eq!(recent[0].gesture_name, "gesture4");
    }
}
