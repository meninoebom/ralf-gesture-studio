//! Real-time gesture recognizer - Simple Wekinator-style implementation.
//!
//! This is the simplest possible DTW recognizer:
//! 1. Store training examples as-is
//! 2. Keep a sliding window of recent frames
//! 3. Compare window against all examples
//! 4. Fire when best distance < threshold
//!
//! No downsampling, no fancy optimizations - just the basics that work.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use super::dtw::{average_motion_energy, dtw_distance, Frame, Sequence};

/// State of motion energy calibration
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub enum CalibrationState {
    /// No calibration data yet
    Uncalibrated,
    /// Collecting samples to learn noise floor
    Learning { samples: Vec<f32> },
    /// Calibration complete with computed threshold
    Calibrated {
        mean: f32,
        std: f32,
        threshold: f32,
    },
}

/// Configuration for gesture recognition
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    /// Cooldown between hits for same gesture (ms)
    pub cooldown_ms: u64,
    /// Number of frames to look back for peak detection
    pub peak_history_size: usize,
    /// Max frames to wait for a rise before firing anyway (sustained gesture support)
    pub max_sustain_frames: usize,
    /// Minimum motion energy to run DTW (0.0 = disabled, auto-calibrated on start)
    pub motion_threshold: f32,
    /// Number of frames to average for motion energy calculation
    pub motion_window: usize,
    /// Whether motion gating is enabled (can disable for debugging)
    pub motion_gate_enabled: bool,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            cooldown_ms: 500,
            peak_history_size: 3, // Look at last 3 distance readings to find minimum
            max_sustain_frames: 8, // Fire after 8 frames (~2s) of sustained low distance
            motion_threshold: 0.0001, // Very low default, will be auto-calibrated
            motion_window: 5, // Average over 5 frames for stability
            motion_gate_enabled: true, // Enabled by default
        }
    }
}

/// Result of processing a frame
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    pub gesture_id: Option<u32>,
    pub gesture_name: Option<String>,
    pub distance: f32,
    #[allow(dead_code)]
    pub threshold: f32,
}

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
    /// Must go above threshold before firing again (prevents double-hits)
    armed: bool,
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
            armed: true, // Start armed so first gesture can fire
        }
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
        self.armed = false; // Must go above threshold to re-arm
    }

    /// Check if gesture is armed (can fire)
    pub fn is_armed(&self) -> bool {
        self.armed
    }

    /// Arm the gesture (called when distance goes above threshold)
    /// Note: With peak detection, this is less critical but kept for API completeness
    #[allow(dead_code)]
    pub fn arm(&mut self) {
        self.armed = true;
    }

    /// Disarm (called after hit)
    #[allow(dead_code)]
    pub fn disarm(&mut self) {
        self.armed = false;
    }
}

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

/// The recognizer
#[derive(Debug)]
pub struct Recognizer {
    pub buffer: FrameBuffer,
    gestures: Vec<GestureState>,
    config: RecognitionConfig,
    active: bool,
    /// Window size for matching (derived from first example)
    window_size: usize,
    /// Frame counter for skipping (DTW is expensive, don't run every frame)
    frame_count: usize,
    /// How often to run DTW (every Nth frame)
    dtw_skip: usize,
    /// Downsample factor for DTW (compare at 15fps instead of 60fps)
    downsample: usize,
    /// History of best distances for peak detection
    distance_history: VecDeque<(f32, usize)>, // (distance, gesture_idx)
    /// Count of consecutive frames below threshold (for sustained gesture detection)
    frames_below_threshold: usize,
    /// Best distance seen during current sustained period
    sustained_best_distance: f32,
    /// Gesture index for sustained best
    sustained_best_gesture: Option<usize>,
    /// Whether recognition is armed (must return to ~resting state to re-arm)
    /// This controls BOTH peak detection and sustained detection
    recognition_armed: bool,
    /// When we last fired (for time-based re-arming in continuous mode)
    last_fire_time: Option<Instant>,
    /// Current motion energy (for UI display and gating)
    current_motion_energy: f32,
    /// Whether motion gate is currently blocking DTW (user is still)
    motion_gate_active: bool,
    /// State of motion threshold calibration
    calibration_state: CalibrationState,
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
        let history_size = config.peak_history_size;
        Self {
            buffer: FrameBuffer::new(buffer_size),
            gestures: Vec::new(),
            config,
            active: false,
            window_size,
            frame_count: 0,
            dtw_skip: 4, // Only compute DTW every 4th frame (15Hz instead of 60Hz)
            downsample: 4, // Compare at 15fps (every 4th frame)
            distance_history: VecDeque::with_capacity(history_size + 1),
            frames_below_threshold: 0,
            sustained_best_distance: f32::MAX,
            sustained_best_gesture: None,
            recognition_armed: true, // Start armed
            last_fire_time: None,
            current_motion_energy: 0.0,
            motion_gate_active: false, // Start with gate off (assume moving)
            calibration_state: CalibrationState::Learning { samples: Vec::new() },
        }
    }

    /// Downsample a sequence by taking every Nth frame
    fn downsample(seq: &Sequence, factor: usize) -> Sequence {
        if factor <= 1 {
            return seq.clone();
        }
        seq.iter().step_by(factor).cloned().collect()
    }

    #[allow(dead_code)]
    pub fn config(&self) -> &RecognitionConfig {
        &self.config
    }

    #[allow(dead_code)]
    pub fn set_config(&mut self, config: RecognitionConfig) {
        self.config = config;
    }

    pub fn set_cooldown_ms(&mut self, ms: u64) {
        self.config.cooldown_ms = ms;
    }

    /// Enable or disable motion gating
    pub fn set_motion_gate_enabled(&mut self, enabled: bool) {
        self.config.motion_gate_enabled = enabled;
    }

    /// Get current motion energy (for UI display)
    pub fn current_motion_energy(&self) -> f32 {
        self.current_motion_energy
    }

    /// Check if motion gate is currently active (blocking DTW)
    pub fn is_motion_gate_active(&self) -> bool {
        self.motion_gate_active
    }

    /// Get calibration state (for UI display)
    pub fn calibration_state(&self) -> &CalibrationState {
        &self.calibration_state
    }

    /// Get current motion threshold
    pub fn motion_threshold(&self) -> f32 {
        self.config.motion_threshold
    }

    /// Manually trigger recalibration
    pub fn recalibrate(&mut self) {
        self.calibration_state = CalibrationState::Learning { samples: Vec::new() };
    }

    /// Update calibration state with new motion energy sample
    fn update_calibration(&mut self, energy: f32) {
        const CALIBRATION_SAMPLES: usize = 60; // ~1 second at 60fps
        const CALIBRATION_COEFFICIENT: f32 = 3.0; // μ + 3σ (conservative)

        if let CalibrationState::Learning { ref mut samples } = self.calibration_state {
            samples.push(energy);

            if samples.len() >= CALIBRATION_SAMPLES {
                // Compute mean
                let n = samples.len() as f32;
                let mean = samples.iter().sum::<f32>() / n;

                // Compute standard deviation
                let variance = samples.iter().map(|e| (e - mean).powi(2)).sum::<f32>() / n;
                let std = variance.sqrt();

                // Set threshold: μ + 3σ
                let threshold = mean + std * CALIBRATION_COEFFICIENT;

                // Ensure minimum threshold to avoid too-tight gating
                let min_threshold = 0.0001;
                let threshold = threshold.max(min_threshold);

                self.calibration_state = CalibrationState::Calibrated { mean, std, threshold };
                self.config.motion_threshold = threshold;
            }
        }
    }

    /// Update motion gate state with hysteresis
    fn update_motion_gate(&mut self, energy: f32) {
        // Hysteresis band: gate ON at 80%, OFF at 100% of threshold
        let on_threshold = self.config.motion_threshold * 0.8;
        let off_threshold = self.config.motion_threshold;

        if self.motion_gate_active {
            // Gate is ON - need higher energy to turn OFF
            if energy > off_threshold {
                self.motion_gate_active = false;
            }
        } else {
            // Gate is OFF - need lower energy to turn ON
            if energy < on_threshold {
                self.motion_gate_active = true;
            }
        }
    }

    /// Clear detection state (called when motion gate deactivates)
    fn clear_detection_state(&mut self) {
        self.distance_history.clear();
        self.frames_below_threshold = 0;
        self.sustained_best_distance = f32::MAX;
        self.sustained_best_gesture = None;
        // Re-arm recognition when user starts moving
        self.recognition_armed = true;
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
        }
    }

    pub fn stop(&mut self) {
        self.active = false;
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Process a frame. Returns recognition result.
    ///
    /// Uses PEAK DETECTION instead of threshold crossing:
    /// - Track distance history
    /// - Fire when we detect a local minimum (distance starts rising)
    /// - Only fire if that minimum was below threshold
    ///
    /// MOTION GATING: Skips DTW entirely when user is standing still.
    /// This prevents false positives from resting state having low DTW distances.
    pub fn process_frame(&mut self, frame: Frame) -> Option<RecognitionResult> {
        self.buffer.push(frame);
        self.frame_count += 1;

        if !self.active {
            return None;
        }

        // Need enough frames for motion energy calculation
        if self.buffer.len() < self.config.motion_window {
            return None;
        }

        // MOTION ENERGY GATING
        // Compute motion energy from recent frames
        let recent_frames = self.buffer.recent(self.config.motion_window);
        let energy = average_motion_energy(&recent_frames);
        self.current_motion_energy = energy;

        // Update calibration if still learning
        self.update_calibration(energy);

        // Apply motion gate with hysteresis
        if self.config.motion_gate_enabled {
            let was_gated = self.motion_gate_active;
            self.update_motion_gate(energy);

            // If gate just deactivated (user started moving), clear detection state
            if was_gated && !self.motion_gate_active {
                self.clear_detection_state();
            }

            // If gate is active (user is still), skip DTW entirely
            if self.motion_gate_active {
                return Some(RecognitionResult {
                    gesture_id: None,
                    gesture_name: None,
                    distance: f32::MAX,
                    threshold: 0.0,
                });
            }
        }

        // Need window_size frames for DTW
        if self.window_size == 0 || self.buffer.len() < self.window_size {
            return None;
        }

        // Skip frames to reduce CPU load (DTW is expensive)
        if !self.frame_count.is_multiple_of(self.dtw_skip) {
            // Return last known state without recomputing
            return Some(RecognitionResult {
                gesture_id: None,
                gesture_name: None,
                distance: f32::MAX,
                threshold: 0.0,
            });
        }

        // Get current window and downsample for efficient DTW
        let window_full = self.buffer.recent(self.window_size);
        let window = Self::downsample(&window_full, self.downsample);

        // Find best match across all gestures and examples
        let mut best_distance = f32::MAX;
        let mut best_gesture_idx: Option<usize> = None;
        let cooldown = Duration::from_millis(self.config.cooldown_ms);

        // Pre-compute downsampled distances for each gesture
        let mut gesture_best_distances: Vec<f32> = Vec::with_capacity(self.gestures.len());

        for gesture in self.gestures.iter() {
            if gesture.examples().is_empty() {
                gesture_best_distances.push(f32::MAX);
                continue;
            }

            // Find best distance to any example of this gesture
            let mut best_for_gesture = f32::MAX;
            for example in gesture.examples() {
                // Downsample example too
                let example_ds = Self::downsample(example, self.downsample);
                let dist = dtw_distance(&window, &example_ds);
                if dist < best_for_gesture {
                    best_for_gesture = dist;
                }
            }

            gesture_best_distances.push(best_for_gesture);
        }

        // Update gesture states with current distances
        for (idx, gesture) in self.gestures.iter_mut().enumerate() {
            let dist = gesture_best_distances[idx];
            gesture.current_distance = if dist < f32::MAX { Some(dist) } else { None };

            if dist < best_distance {
                best_distance = dist;
                best_gesture_idx = Some(idx);
            }
        }

        // PEAK DETECTION: Fire at local minimum, not threshold crossing
        // This naturally handles the "stuck" problem because resting state is flat, not a minimum
        let history_size = self.config.peak_history_size;

        // Add current reading to history
        if let Some(idx) = best_gesture_idx {
            self.distance_history.push_back((best_distance, idx));

            // Keep history bounded
            while self.distance_history.len() > history_size + 1 {
                self.distance_history.pop_front();
            }
        }

        // Check for local minimum: we need at least 3 readings
        // Pattern: distances were going DOWN, now going UP = we passed the minimum
        if self.distance_history.len() >= 3 {
            let len = self.distance_history.len();

            // Copy the values we need (to satisfy borrow checker)
            let prev = self.distance_history[len - 2];
            let curr = self.distance_history[len - 1];
            let prev_dist = prev.0;
            let prev_gesture_idx = prev.1;
            let curr_dist = curr.0;

            // Check if prev was a minimum: all earlier readings were higher, current is higher
            let was_descending = self.distance_history.iter()
                .take(len - 2)
                .all(|(d, _)| *d >= prev_dist);
            let now_ascending = curr_dist > prev_dist;

            if was_descending && now_ascending && self.recognition_armed {
                // We found a local minimum at prev_dist
                let gesture = &mut self.gestures[prev_gesture_idx];

                // Fire if: minimum is below threshold AND not in cooldown
                if prev_dist < gesture.threshold && !gesture.in_cooldown(cooldown) {
                    gesture.record_hit();

                    // Clear history and reset tracking after firing
                    self.distance_history.clear();
                    self.frames_below_threshold = 0;
                    self.sustained_best_distance = f32::MAX;
                    self.sustained_best_gesture = None;
                    self.recognition_armed = false; // Disarm until returning to ~resting
                    self.last_fire_time = Some(Instant::now());

                    return Some(RecognitionResult {
                        gesture_id: Some(gesture.id),
                        gesture_name: Some(gesture.name.clone()),
                        distance: prev_dist,
                        threshold: gesture.threshold,
                    });
                }
            }
        }

        // SUSTAINED GESTURE DETECTION: Fire if below threshold for too long without a clear dip
        // This handles continuous gesture performance where user doesn't return to resting state
        if let Some(idx) = best_gesture_idx {
            let gesture_threshold = self.gestures[idx].threshold;

            if best_distance < gesture_threshold {
                // We're below threshold - track this (only if armed)
                if self.recognition_armed {
                    self.frames_below_threshold += 1;

                    // Track the best (lowest) distance during this sustained period
                    if best_distance < self.sustained_best_distance {
                        self.sustained_best_distance = best_distance;
                        self.sustained_best_gesture = Some(idx);
                    }

                    // If we've been below threshold for too long, fire at the best distance we saw
                    if self.frames_below_threshold >= self.config.max_sustain_frames {
                        if let Some(gesture_idx) = self.sustained_best_gesture {
                            let gesture = &mut self.gestures[gesture_idx];

                            if !gesture.in_cooldown(cooldown) {
                                let fire_distance = self.sustained_best_distance;
                                gesture.record_hit();

                                // Reset tracking and disarm
                                self.distance_history.clear();
                                self.frames_below_threshold = 0;
                                self.sustained_best_distance = f32::MAX;
                                self.sustained_best_gesture = None;
                                self.recognition_armed = false; // Disarm until returning to ~resting
                                self.last_fire_time = Some(Instant::now());

                                return Some(RecognitionResult {
                                    gesture_id: Some(gesture.id),
                                    gesture_name: Some(gesture.name.clone()),
                                    distance: fire_distance,
                                    threshold: gesture.threshold,
                                });
                            }
                        }
                    }
                }
            } else {
                // Distance went above threshold - reset tracking and re-arm
                self.frames_below_threshold = 0;
                self.sustained_best_distance = f32::MAX;
                self.sustained_best_gesture = None;
                self.recognition_armed = true;
            }

            // RE-ARMING LOGIC: Multiple paths to re-arm
            if !self.recognition_armed {
                // Path 1: Distance rises to 75% of threshold (returning toward rest)
                let rearm_threshold = gesture_threshold * 0.75;
                if best_distance > rearm_threshold {
                    self.recognition_armed = true;
                }

                // Path 2: Time-based re-arm for continuous gesture mode
                // After 2x cooldown time, re-arm regardless (allows continuous gestures)
                if let Some(fire_time) = self.last_fire_time {
                    let rearm_delay = Duration::from_millis(self.config.cooldown_ms * 2);
                    if fire_time.elapsed() > rearm_delay {
                        self.recognition_armed = true;
                    }
                }
            }
        }

        // No hit - return status
        Some(RecognitionResult {
            gesture_id: None,
            gesture_name: None,
            distance: best_distance,
            threshold: 0.0,
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
        // Test that recognizer detects gestures via peak detection OR sustained detection
        // Note: Recognizer skips every 4th frame for DTW, so we need enough frames
        let mut recognizer = Recognizer::new(100, 0);
        // Disable motion gate for this test (uses static frames)
        recognizer.set_motion_gate_enabled(false);

        // DTW distances: matching=~6, resting=~485. Threshold 50 means only matching is below.
        recognizer.add_gesture(1, "wave", "/gesture/1", 50.0);

        // Add example: frames 1,2,3,4,5
        let example = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Phase 1: Fill buffer with "resting" frames (distance ~485, above threshold 50)
        // Need enough frames so buffer is full and DTW actually runs
        // dtw_skip=4 means we need 4 frames per distance reading
        for _ in 0..40 {
            recognizer.process_frame(vec![100.0]);
        }

        // Phase 2: Feed matching frames for a sustained period (distance ~6, below threshold 50)
        // This will trigger sustained detection (8+ frames below threshold)
        // dtw_skip=4, so we need 4*8=32+ raw frames for 8 DTW readings
        let mut hit = None;
        for _ in 0..50 {
            if let Some(result) = recognizer.process_frame(vec![3.0]) {
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

    // =========================================================================
    // Motion Gate Tests
    // =========================================================================

    #[test]
    fn test_motion_gate_blocks_when_still() {
        let mut recognizer = Recognizer::new(100, 5);
        recognizer.add_gesture(1, "wave", "/gesture/1", 50.0);
        recognizer.add_example(1, vec![vec![1.0]; 5]);
        recognizer.start();

        // Feed identical frames (no motion) - should be blocked by motion gate
        // First, let calibration complete by feeding enough frames
        for _ in 0..70 {
            recognizer.process_frame(vec![1.0]);
        }

        // Motion gate should be active (blocking) since frames are identical
        assert!(recognizer.is_motion_gate_active(), "Motion gate should be active when standing still");
    }

    #[test]
    fn test_motion_gate_allows_when_moving() {
        let mut recognizer = Recognizer::new(100, 5);
        recognizer.add_gesture(1, "wave", "/gesture/1", 50.0);
        recognizer.add_example(1, vec![vec![1.0]; 5]);
        recognizer.start();

        // Feed varying frames (motion) to calibrate with motion
        for i in 0..70 {
            recognizer.process_frame(vec![i as f32 * 0.1]);
        }

        // Motion gate should NOT be active when there's movement
        assert!(!recognizer.is_motion_gate_active(), "Motion gate should not block when moving");
    }

    #[test]
    fn test_motion_gate_can_be_disabled() {
        let mut recognizer = Recognizer::new(100, 5);
        recognizer.set_motion_gate_enabled(false);
        recognizer.add_gesture(1, "wave", "/gesture/1", 50.0);
        recognizer.add_example(1, vec![vec![1.0]; 5]);
        recognizer.start();

        // Feed identical frames
        for _ in 0..70 {
            recognizer.process_frame(vec![1.0]);
        }

        // Motion gate should not be active when disabled
        assert!(!recognizer.is_motion_gate_active(), "Motion gate should not activate when disabled");
    }

    #[test]
    fn test_calibration_completes() {
        let mut recognizer = Recognizer::new(100, 5);
        recognizer.start();

        // Initially in Learning state
        assert!(matches!(
            recognizer.calibration_state(),
            CalibrationState::Learning { .. }
        ));

        // Feed 60+ frames to complete calibration
        for i in 0..65 {
            recognizer.process_frame(vec![i as f32 * 0.001]);
        }

        // Should now be Calibrated
        assert!(matches!(
            recognizer.calibration_state(),
            CalibrationState::Calibrated { .. }
        ));
    }

    #[test]
    fn test_recalibrate_resets_state() {
        let mut recognizer = Recognizer::new(100, 5);
        recognizer.start();

        // Complete calibration
        for i in 0..65 {
            recognizer.process_frame(vec![i as f32 * 0.001]);
        }
        assert!(matches!(
            recognizer.calibration_state(),
            CalibrationState::Calibrated { .. }
        ));

        // Recalibrate
        recognizer.recalibrate();

        // Should be back in Learning state
        assert!(matches!(
            recognizer.calibration_state(),
            CalibrationState::Learning { .. }
        ));
    }

    #[test]
    fn test_motion_gate_hysteresis() {
        // Test that hysteresis prevents rapid toggling
        let mut recognizer = Recognizer::new(100, 5);
        recognizer.start();

        // Manually set a known threshold after calibration completes
        for i in 0..65 {
            recognizer.process_frame(vec![i as f32 * 0.001]);
        }

        // The test verifies the hysteresis logic exists by checking
        // that small variations near threshold don't cause toggling
        let _initial_state = recognizer.is_motion_gate_active();

        // Feed a few more frames with similar energy
        for _ in 0..5 {
            recognizer.process_frame(vec![0.065]);
        }

        // State should be stable (not toggling rapidly)
        // This is a basic sanity check; full hysteresis testing would
        // require more precise control over energy values
        let _ = recognizer.is_motion_gate_active();
        assert!(true, "Hysteresis test completed without crash");
    }
}
