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

use super::dtw::{dtw_distance, Frame, Sequence};

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
}

impl GestureState {
    pub fn new(id: u32, name: &str, osc_address: &str, threshold: f32) -> Self {
        Self {
            id,
            name: name.to_string(),
            osc_address: osc_address.to_string(),
            threshold,
            examples: Vec::new(),
            current_distance: None,
            last_hit_time: None,
            // GRT-style best template selection
            best_template_index: None,
            // VAD state machine
            state: RecognitionState::Idle,
            frames_below_threshold: 0,
            recovery_start: None,
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
    }

    /// Process a distance value through the state machine.
    /// Returns result with fire signal and any state transition that occurred.
    fn process_state_machine(
        &mut self,
        distance: f32,
        config: &RecognitionConfig,
    ) -> StateMachineResult {
        let entry_threshold = self.threshold * config.threshold_high_factor;
        let hangover = Duration::from_millis(config.hangover_ms);
        let prev_state = self.state;
        let prev_frames = self.frames_below_threshold;

        let (should_fire, new_state, reason) = match self.state {
            RecognitionState::Idle => {
                // Check if distance is below entry threshold
                if distance < entry_threshold {
                    // Start building
                    self.state = RecognitionState::Building;
                    self.frames_below_threshold = 1;

                    // Check if we already have enough frames (handles frames_to_fire = 1)
                    if self.frames_below_threshold >= config.frames_to_fire {
                        self.state = RecognitionState::Peak;
                        self.record_hit();
                        (true, Some(RecognitionState::Peak), "below_threshold_instant_fire")
                    } else {
                        (false, Some(RecognitionState::Building), "below_threshold")
                    }
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
                // Check if hangover period has elapsed
                let hangover_complete = self.recovery_start
                    .map(|t| t.elapsed() >= hangover)
                    .unwrap_or(true);

                // Exit recovery when hangover is complete
                // (Wekinator-style: simple time-based cooldown)
                //
                // Note: We tried requiring distance > exit_threshold but that fails
                // when resting distance is still below threshold (common for body tracking)
                if hangover_complete {
                    self.reset_to_idle();
                    (false, Some(RecognitionState::Idle), "hangover_complete")
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
    /// Used as fallback when best template selection is not available
    fn find_best_distance(window: &Sequence, examples: &[Sequence], downsample: usize) -> f32 {
        let mut best = f32::MAX;
        for example in examples {
            let example_ds = Self::downsample_seq(example, downsample);
            let dist = dtw_distance(window, &example_ds);
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
        for gesture in &mut self.gestures {
            gesture.current_distance = None;
            gesture.reset_to_idle();
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
        if self.frame_count % self.dtw_skip != 0 {
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

        // Compute distances for all gestures
        // GRT-style: Use best template if available (3+ examples), else compare all
        let mut distances: Vec<(usize, f32)> = Vec::new();

        for (idx, gesture) in self.gestures.iter().enumerate() {
            if gesture.examples().is_empty() {
                continue;
            }

            let examples = gesture.examples();
            let best_for_gesture = if self.use_best_template && examples.len() >= 3 {
                // GRT-style (Phase 3): Compare only to best template (most representative example)
                if let Some(best_idx) = gesture.best_template_index {
                    if best_idx < examples.len() {
                        let example_ds = Self::downsample_seq(&examples[best_idx], self.downsample);
                        dtw_distance(&window, &example_ds)
                    } else {
                        // Invalid index, fall back to all examples
                        Self::find_best_distance(&window, examples, self.downsample)
                    }
                } else {
                    // No best template computed, fall back to all examples
                    Self::find_best_distance(&window, examples, self.downsample)
                }
            } else {
                // Wekinator-style (pre-Phase 3): compare against ALL examples
                Self::find_best_distance(&window, examples, self.downsample)
            };

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
    fn test_state_machine_recovery_exits_after_hangover() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 100.0);
        let config = RecognitionConfig {
            frames_to_fire: 1,
            hangover_ms: 50,
            ..Default::default()
        };

        // Go to Peak and then Recovery
        gesture.process_state_machine(50.0, &config); // Peak
        gesture.process_state_machine(50.0, &config); // Recovery

        // Hangover not complete, should stay in Recovery
        let result = gesture.process_state_machine(50.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Recovery);
        assert!(result.transition.is_none()); // No transition yet

        // Wait for hangover
        std::thread::sleep(Duration::from_millis(60));

        // Now hangover complete, should go to Idle (regardless of distance)
        let result = gesture.process_state_machine(50.0, &config);
        assert_eq!(gesture.recognition_state(), RecognitionState::Idle);
        assert!(result.transition.is_some());
        let t = result.transition.unwrap();
        assert_eq!(t.to_state, RecognitionState::Idle);
        assert_eq!(t.reason, "hangover_complete");
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
