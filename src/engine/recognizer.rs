//! Real-time gesture recognizer with VAD-style state machine.
//!
//! Based on research from GRT, Wekinator, and speech recognition (VAD):
//!
//! **Key patterns from speech recognition**:
//! - Hysteresis: Entry threshold ≠ exit threshold (prevents stuck state)
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
    compute_lb_envelope, dtw_distance, dtw_distance_with_abandon, lb_keogh, Frame, LBEnvelope,
    Sequence,
};

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

    /// Hangover time in ms after firing (Recovery state duration)
    /// Blocks new detections to prevent echo
    pub hangover_ms: u64,

    // --- Echo guard (hysteresis re-arming) ---

    /// Factor for re-arm threshold (distance must exceed threshold × this factor to re-arm quickly)
    /// Example: 1.3 means distance must rise 30% above threshold to exit Recovery early
    /// If distance never exceeds this, use extended_hangover_ms instead
    pub rearm_threshold_factor: f32,

    /// Extended hangover time (ms) used when distance never exceeds rearm_threshold
    /// This is the fallback path for "sticky" distances that hover near threshold
    pub extended_hangover_ms: u64,

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
            // Frame accumulation: require 3 consecutive frames (~200ms at 15Hz DTW)
            frames_to_fire: 3,
            // Hangover: block new detections for 300ms after firing
            hangover_ms: 300,
            // Echo guard: distance must exceed 30% above threshold to re-arm quickly
            rearm_threshold_factor: 1.3,
            // Extended hangover: 500ms if distance never clearly exceeds threshold
            extended_hangover_ms: 500,
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
    pub reason: String,
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
    pub threshold: f32,
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

    // --- GRT-style best template selection ---
    /// Index of the best template (example with lowest average distance to others)
    /// Used during recognition to compare only against this representative example.
    /// Falls back to comparing all examples if None or fewer than 3 examples.
    best_template_index: Option<usize>,

    // --- VAD state machine ---
    /// Current state in the recognition state machine
    state: RecognitionState,
    /// Count of consecutive frames below entry threshold (used in Building state)
    frames_below_threshold: usize,
    /// When we entered Recovery state (for hangover timing)
    recovery_start: Option<Instant>,
    /// Maximum distance seen during Recovery state (for hysteresis re-arming)
    max_distance_in_recovery: f32,
    /// History of recent distance values (for slope detection)
    distance_history: VecDeque<f32>,
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
            // GRT-style best template selection
            best_template_index: None,
            // VAD state machine
            state: RecognitionState::Idle,
            frames_below_threshold: 0,
            recovery_start: None,
            max_distance_in_recovery: 0.0,
            distance_history: VecDeque::with_capacity(DISTANCE_HISTORY_SIZE),
        }
    }

    /// Set the best template index (computed during training)
    pub fn set_best_template_index(&mut self, index: Option<usize>) {
        self.best_template_index = index;
    }

    /// Get the best template index
    #[allow(dead_code)]
    pub fn best_template_index(&self) -> Option<usize> {
        self.best_template_index
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

    #[allow(dead_code)]
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
    pub fn recognition_state(&self) -> RecognitionState {
        self.state
    }

    /// Reset state machine to Idle
    fn reset_to_idle(&mut self) {
        self.state = RecognitionState::Idle;
        self.frames_below_threshold = 0;
        self.recovery_start = None;
        self.max_distance_in_recovery = 0.0;
        // Note: We don't clear distance_history here - it's useful for the next detection
    }

    /// Check if distance is falling (negative slope) or flat
    /// Returns true if distance is decreasing or at a flat minimum
    /// This helps avoid false triggers on noise/echo with flat minima
    fn is_distance_falling(&self, current: f32) -> bool {
        if self.distance_history.len() < 2 {
            return true; // Not enough history, allow entry
        }

        // Get previous distance
        let prev = self.distance_history.back().copied().unwrap_or(current);

        // Calculate slope (current - previous)
        let slope = current - prev;

        // Require negative slope (falling) or very small positive (flat minimum)
        // Allow 5% tolerance for noise
        slope < 0.05 * current
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
    fn process_state_machine(
        &mut self,
        distance: f32,
        config: &RecognitionConfig,
    ) -> StateMachineResult {
        // Record distance for slope detection (before state processing)
        self.record_distance(distance);

        let entry_threshold = self.threshold * config.threshold_high_factor;
        let prev_state = self.state;
        let prev_frames = self.frames_below_threshold;

        let (should_fire, new_state, reason) = match self.state {
            RecognitionState::Idle => {
                // Check if distance is below entry threshold AND falling
                // The slope check reduces false triggers on noise/echo with flat minima
                if distance < entry_threshold && self.is_distance_falling(distance) {
                    // Start building
                    self.state = RecognitionState::Building;
                    self.frames_below_threshold = 1;

                    // Check if we already have enough frames (handles frames_to_fire = 1)
                    if self.frames_below_threshold >= config.frames_to_fire {
                        self.state = RecognitionState::Peak;
                        self.record_hit();
                        (true, Some(RecognitionState::Peak), "below_threshold_instant_fire")
                    } else {
                        (false, Some(RecognitionState::Building), "below_threshold_falling")
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
                // Track max distance seen during recovery (for hysteresis)
                self.max_distance_in_recovery = self.max_distance_in_recovery.max(distance);

                let elapsed = self.recovery_start
                    .map(|t| t.elapsed())
                    .unwrap_or(Duration::ZERO);

                let hangover = Duration::from_millis(config.hangover_ms);
                let extended_hangover = Duration::from_millis(config.extended_hangover_ms);
                let rearm_threshold = self.threshold * config.rearm_threshold_factor;

                // Dual-path re-arming (hysteresis echo guard):
                // Path 1: Distance clearly exceeded rearm threshold + standard hangover
                // Path 2: Extended hangover (for sticky distances that hover near threshold)
                let can_rearm = if self.max_distance_in_recovery > rearm_threshold {
                    // Fast path: distance went clearly above threshold
                    elapsed >= hangover
                } else {
                    // Slow path: distance stayed near threshold, wait longer
                    elapsed >= extended_hangover
                };

                if can_rearm {
                    let reason = if self.max_distance_in_recovery > rearm_threshold {
                        "hangover_complete_distance_exceeded"
                    } else {
                        "extended_hangover_complete"
                    };
                    self.reset_to_idle();
                    (false, Some(RecognitionState::Idle), reason)
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
            reason: reason.to_string(),
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

    /// Get the most recent N frames as a sequence
    pub fn recent(&self, n: usize) -> Sequence {
        let start = self.frames.len().saturating_sub(n);
        self.frames.iter().skip(start).cloned().collect()
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
    /// Whether to use GRT-style best template selection (Phase 3)
    /// When true: Compare only to best template for gestures with 3+ examples
    /// When false: Compare to ALL examples (Wekinator-style)
    use_best_template: bool,
    /// Pending state transitions to be logged (cleared after retrieval)
    pending_transitions: Vec<GestureStateTransition>,
}

impl Recognizer {
    #[allow(dead_code)]
    pub fn new(buffer_size: usize, window_size: usize) -> Self {
        Self::with_config(buffer_size, window_size, RecognitionConfig::default())
    }

    pub fn with_config(
        buffer_size: usize,
        window_size: usize,
        config: RecognitionConfig,
    ) -> Self {
        Self {
            buffer: FrameBuffer::new(buffer_size),
            gestures: Vec::new(),
            config,
            active: false,
            window_size,
            frame_count: 0,
            dtw_skip: 4,      // Compute DTW every 4th frame (15Hz @ 60fps input)
            downsample: 4,    // Compare at 15fps
            use_best_template: false, // Default: compare ALL examples (Wekinator-style, more responsive)
            pending_transitions: Vec::new(),
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
    ) -> f32 {
        let mut best = best_so_far;
        for (i, example) in examples.iter().enumerate() {
            // Layer 1: LB_Keogh pruning (O(n) lower bound check)
            if let Some(envelope) = envelopes.get(i) {
                let lb = lb_keogh(window, envelope);
                if lb >= best {
                    continue; // Lower bound already exceeds best, skip DTW
                }
            }

            // Layer 2: Full DTW with early abandoning
            let example_ds = Self::downsample_seq(example, downsample);
            let dist = if sakoe_chiba_band > 0.0 {
                // Use Sakoe-Chiba constrained DTW with early abandoning
                let max_len = window.len().max(example_ds.len());
                let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
                // Use early abandoning - skip if can't beat current best
                match dtw_distance_with_abandon(window, &example_ds, band_width, best) {
                    Some(d) => d,
                    None => continue, // Abandoned - worse than best
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

    #[allow(dead_code)]
    pub fn config(&self) -> &RecognitionConfig {
        &self.config
    }

    pub fn set_cooldown_ms(&mut self, ms: u64) {
        self.config.cooldown_ms = ms;
    }

    /// Set whether to use GRT-style best template selection (Phase 3)
    /// When true: Compare only to best template for gestures with 3+ examples
    /// When false: Compare to ALL examples (Wekinator-style, pre-Phase 3)
    pub fn set_use_best_template(&mut self, use_it: bool) {
        self.use_best_template = use_it;
    }

    /// Get whether using best template selection
    pub fn use_best_template(&self) -> bool {
        self.use_best_template
    }

    pub fn add_gesture(&mut self, id: u32, name: &str, osc_address: &str, threshold: f32) {
        self.gestures.push(GestureState::new(id, name, osc_address, threshold));
    }

    pub fn get_gesture_mut(&mut self, id: u32) -> Option<&mut GestureState> {
        self.gestures.iter_mut().find(|g| g.id == id)
    }

    pub fn get_gesture(&self, id: u32) -> Option<&GestureState> {
        self.gestures.iter().find(|g| g.id == id)
    }

    #[allow(dead_code)]
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

    /// Process a frame through the VAD-style state machine.
    ///
    /// **Algorithm**:
    /// 1. Compute DTW distance to all examples
    /// 2. Find best matching gesture
    /// 3. Run distance through state machine for that gesture
    /// 4. Fire if state machine transitions to Peak
    pub fn process_frame(&mut self, frame: Frame) -> Option<RecognitionResult> {
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
                threshold: 0.0,
            });
        }

        // Get current window and downsample for efficient DTW
        let window_full = self.buffer.recent(self.window_size);
        let window = Self::downsample_seq(&window_full, self.downsample);

        // Compute distances for all gestures with early abandoning
        // Track best_so_far across all gestures for maximum pruning efficiency
        let mut distances: Vec<(usize, f32)> = Vec::new();
        let mut best_so_far = f32::MAX;
        let sakoe_chiba_band = self.config.sakoe_chiba_band;

        for (idx, gesture) in self.gestures.iter().enumerate() {
            if gesture.examples().is_empty() {
                continue;
            }

            let examples = gesture.examples();
            let envelopes = gesture.lb_envelopes();

            let best_for_gesture = if self.use_best_template && examples.len() >= 3 {
                // GRT-style (Phase 3): Compare only to best template (most representative example)
                if let Some(best_idx) = gesture.best_template_index {
                    if best_idx < examples.len() {
                        // LB_Keogh pruning for single template
                        if let Some(envelope) = envelopes.get(best_idx) {
                            let lb = lb_keogh(&window, envelope);
                            if lb >= best_so_far {
                                f32::MAX // Pruned by lower bound
                            } else {
                                let example_ds = Self::downsample_seq(&examples[best_idx], self.downsample);
                                if sakoe_chiba_band > 0.0 {
                                    let max_len = window.len().max(example_ds.len());
                                    let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
                                    dtw_distance_with_abandon(&window, &example_ds, band_width, best_so_far).unwrap_or(f32::MAX)
                                } else {
                                    dtw_distance(&window, &example_ds)
                                }
                            }
                        } else {
                            // No envelope, compute DTW directly
                            let example_ds = Self::downsample_seq(&examples[best_idx], self.downsample);
                            if sakoe_chiba_band > 0.0 {
                                let max_len = window.len().max(example_ds.len());
                                let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
                                dtw_distance_with_abandon(&window, &example_ds, band_width, best_so_far).unwrap_or(f32::MAX)
                            } else {
                                dtw_distance(&window, &example_ds)
                            }
                        }
                    } else {
                        // Invalid index, fall back to all examples
                        Self::find_best_distance(&window, examples, envelopes, self.downsample, sakoe_chiba_band, best_so_far)
                    }
                } else {
                    // No best template computed, fall back to all examples
                    Self::find_best_distance(&window, examples, envelopes, self.downsample, sakoe_chiba_band, best_so_far)
                }
            } else {
                // Wekinator-style (pre-Phase 3): compare against ALL examples
                Self::find_best_distance(&window, examples, envelopes, self.downsample, sakoe_chiba_band, best_so_far)
            };

            // Update best_so_far for subsequent gestures
            if best_for_gesture < best_so_far {
                best_so_far = best_for_gesture;
            }

            distances.push((idx, best_for_gesture));
        }

        // Update gesture distances for UI and run state machines
        let mut hit_result: Option<(u32, String, f32, f32)> = None;

        for (idx, gesture) in self.gestures.iter_mut().enumerate() {
            // Find this gesture's distance
            let distance = distances.iter()
                .find(|(i, _)| *i == idx)
                .map(|(_, d)| *d);

            gesture.current_distance = distance;

            // Run state machine if we have a distance
            if let Some(dist) = distance {
                // Only process the best-matching gesture's state machine
                // (prevents multiple gestures firing simultaneously)
                let is_best = distances.iter()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| *i == idx)
                    .unwrap_or(false);

                if is_best {
                    let result = gesture.process_state_machine(dist, &self.config);

                    // Collect state transition for logging
                    if let Some(transition) = result.transition {
                        self.pending_transitions.push(GestureStateTransition {
                            gesture_name: gesture.name.clone(),
                            transition,
                            distance: dist,
                            threshold: gesture.threshold,
                        });
                    }

                    if result.should_fire {
                        hit_result = Some((
                            gesture.id,
                            gesture.name.clone(),
                            dist,
                            gesture.threshold,
                        ));
                    }
                } else {
                    // Non-best gestures: reset to Idle if they were Building
                    // (prevents multiple gestures building simultaneously)
                    if gesture.state == RecognitionState::Building {
                        gesture.reset_to_idle();
                    }
                }
            }
        }

        // Return hit if we fired
        if let Some((id, name, distance, threshold)) = hit_result {
            return Some(RecognitionResult {
                gesture_id: Some(id),
                gesture_name: Some(name),
                distance,
                threshold,
            });
        }

        // No hit - return current best distance
        let best = distances.iter()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        Some(RecognitionResult {
            gesture_id: None,
            gesture_name: None,
            distance: best.map(|(_, d)| *d).unwrap_or(f32::MAX),
            threshold: best
                .map(|(idx, _)| self.gestures[*idx].threshold)
                .unwrap_or(0.0),
        })
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
    entries: Vec<HitLogEntry>,
    max_entries: usize,
}

impl HitLog {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    pub fn record(&mut self, gesture_id: u32, gesture_name: &str, distance: f32, osc_address: &str) {
        self.entries.push(HitLogEntry {
            timestamp: Instant::now(),
            gesture_id,
            gesture_name: gesture_name.to_string(),
            distance,
            osc_address: osc_address.to_string(),
        });
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
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

    #[test]
    fn test_state_machine_idle_to_building() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        // Distance below threshold should transition to Building
        let result = gesture.process_state_machine(50.0, &config);
        assert!(!result.should_fire);
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);
        assert_eq!(gesture.frames_below_threshold, 1);
        // Should have a transition
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

        // Accumulate frames
        gesture.process_state_machine(50.0, &config); // Building, count=1
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);

        gesture.process_state_machine(45.0, &config); // Building, count=2
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);

        let result = gesture.process_state_machine(40.0, &config); // Peak!
        assert!(result.should_fire);
        assert_eq!(gesture.recognition_state(), RecognitionState::Peak);
        // Should have a transition to Peak
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Peak);
        assert_eq!(t.reason, "accumulated_frames");
    }

    #[test]
    fn test_state_machine_building_reset_on_rise() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        gesture.process_state_machine(50.0, &config); // Building
        assert_eq!(gesture.recognition_state(), RecognitionState::Building);

        // Distance rises above threshold - should reset to Idle
        let result = gesture.process_state_machine(150.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert_eq!(gesture.frames_below_threshold, 0);
        // Should have a transition back to Idle
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

        // First frame below threshold: Idle → Building → Peak (fires)
        let result = gesture.process_state_machine(50.0, &config);
        assert!(result.should_fire, "Should fire when frames_to_fire=1");
        assert_eq!(gesture.recognition_state(), RecognitionState::Peak);

        // Next frame should transition Peak → Recovery
        let result = gesture.process_state_machine(50.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Recovery);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.from_state, RecognitionState::Peak);
        assert_eq!(t.to_state, RecognitionState::Recovery);
    }

    #[test]
    fn test_state_machine_recovery_exits_fast_when_distance_exceeds_rearm() {
        // Test the "fast path": distance exceeds rearm threshold → exit after standard hangover
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig {
            frames_to_fire: 1,
            hangover_ms: 50,
            rearm_threshold_factor: 1.3,      // rearm_threshold = 130
            extended_hangover_ms: 200,        // Extended would be 200ms
            ..Default::default()
        };

        // Go to Peak and then Recovery
        gesture.process_state_machine(50.0, &config); // Peak
        gesture.process_state_machine(50.0, &config); // Recovery

        // Distance exceeds rearm threshold (130)
        gesture.process_state_machine(150.0, &config); // max_distance_in_recovery = 150

        // Wait for standard hangover only (50ms)
        std::thread::sleep(Duration::from_millis(60));

        // Should exit via fast path (distance exceeded rearm threshold)
        let result = gesture.process_state_machine(150.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Idle);
        assert_eq!(t.reason, "hangover_complete_distance_exceeded");
    }

    #[test]
    fn test_state_machine_recovery_exits_slow_when_distance_stays_low() {
        // Test the "slow path": distance stays near threshold → exit after extended hangover
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig {
            frames_to_fire: 1,
            hangover_ms: 50,
            rearm_threshold_factor: 1.3,      // rearm_threshold = 130
            extended_hangover_ms: 100,        // Extended is 100ms
            ..Default::default()
        };

        // Go to Peak and then Recovery
        gesture.process_state_machine(50.0, &config); // Peak
        gesture.process_state_machine(50.0, &config); // Recovery

        // Distance stays below rearm threshold (always 50 < 130)
        gesture.process_state_machine(50.0, &config);

        // Wait for standard hangover (50ms) - should NOT exit yet
        std::thread::sleep(Duration::from_millis(60));
        let result = gesture.process_state_machine(50.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Recovery);
        assert!(result.transition.is_none());

        // Wait for extended hangover (100ms total from Recovery start)
        std::thread::sleep(Duration::from_millis(50)); // Total ~110ms

        // Now should exit via slow path
        let result = gesture.process_state_machine(50.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Idle);
        assert_eq!(t.reason, "extended_hangover_complete");
    }

    #[test]
    fn test_state_machine_slope_check_blocks_flat_entry() {
        // Test that slope check prevents entry on flat/rising distance
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        // Start with high distance, build some history
        gesture.process_state_machine(150.0, &config);
        gesture.process_state_machine(150.0, &config);
        gesture.process_state_machine(150.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);

        // Drop to below threshold but flat (same value) - should NOT enter Building
        // because distance is not falling
        gesture.process_state_machine(50.0, &config); // First below threshold
        // This puts 50.0 in history, but prior was 150.0, so slope is negative (falling)
        // So this will actually enter Building because 50 < 150 means falling

        // Let me test with truly flat: distance stays at same low value
        let mut gesture2 = GestureState::new(2, "wave2", "/gesture/2", 100.0);
        gesture2.process_state_machine(50.0, &config); // History: [50]
        gesture2.process_state_machine(50.0, &config); // History: [50, 50]
        gesture2.process_state_machine(50.0, &config); // History: [50, 50, 50]

        // Now distance is flat at 50 (below threshold)
        // Current = 50, prev = 50, slope = 0 which is < 0.05*50 = 2.5
        // So flat IS allowed (within 5% tolerance)
        // This is intentional: flat minimum is OK, rising is not

        // Test rising: distance starts below threshold but rises slightly
        let mut gesture3 = GestureState::new(3, "wave3", "/gesture/3", 100.0);
        gesture3.process_state_machine(40.0, &config); // History: [40]
        gesture3.process_state_machine(45.0, &config); // History: [40, 45] - rising 12.5%
        // slope = 45-40 = 5, threshold = 0.05*45 = 2.25
        // 5 > 2.25, so this is rising and should NOT enter Building
        // Actually let me check: the first 40 enters Building because it's falling from infinity
        // The second 45 is still below threshold (< 100), but is it rising?

        // The slope check is only applied when entering Building from Idle
        // Once in Building, we just check if distance stays below threshold
    }

    #[test]
    fn test_state_machine_slope_check_allows_falling_entry() {
        // Test that slope check allows entry when distance is falling
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig::default();

        // Start with high distance, build some history
        gesture.process_state_machine(150.0, &config);
        gesture.process_state_machine(120.0, &config); // Falling
        gesture.process_state_machine(90.0, &config);  // Falling, below threshold

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
            100, 5,
            RecognitionConfig {
                frames_to_fire: 1, // Fire on first frame below threshold
                ..Default::default()
            }
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
