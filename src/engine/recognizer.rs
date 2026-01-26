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

use super::dtw::{dtw_distance, Frame, Sequence};

/// Configuration for gesture recognition
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    /// Cooldown between hits for same gesture (ms)
    pub cooldown_ms: u64,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self { cooldown_ms: 500 }
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
    pub fn arm(&mut self) {
        self.armed = true;
    }

    /// Disarm (called after hit)
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
            dtw_skip: 4, // Only compute DTW every 4th frame (15Hz instead of 60Hz)
            downsample: 4, // Compare at 15fps (every 4th frame)
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
    pub fn process_frame(&mut self, frame: Frame) -> Option<RecognitionResult> {
        self.buffer.push(frame);
        self.frame_count += 1;

        if !self.active {
            return None;
        }

        // Need window_size frames
        if self.window_size == 0 || self.buffer.len() < self.window_size {
            return None;
        }

        // Skip frames to reduce CPU load (DTW is expensive)
        if self.frame_count % self.dtw_skip != 0 {
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

        // Find the best overall and update gesture states
        // Also update armed state: arm when above threshold, stay disarmed when below
        for (idx, gesture) in self.gestures.iter_mut().enumerate() {
            let dist = gesture_best_distances[idx];
            gesture.current_distance = if dist < f32::MAX { Some(dist) } else { None };

            // Update armed state: re-arm when distance goes above threshold
            if dist >= gesture.threshold {
                gesture.arm();
            }

            if dist < best_distance {
                best_distance = dist;
                best_gesture_idx = Some(idx);
            }
        }

        // Check for hit - must be armed, below threshold, and not in cooldown
        if let Some(idx) = best_gesture_idx {
            let gesture = &mut self.gestures[idx];

            if best_distance < gesture.threshold
                && gesture.is_armed()
                && !gesture.in_cooldown(cooldown)
            {
                gesture.record_hit(); // This also disarms
                return Some(RecognitionResult {
                    gesture_id: Some(gesture.id),
                    gesture_name: Some(gesture.name.clone()),
                    distance: best_distance,
                    threshold: gesture.threshold,
                });
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
        let mut recognizer = Recognizer::new(100, 0);
        recognizer.add_gesture(1, "wave", "/gesture/1", 50.0);

        // Add example
        let example = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Feed similar frames
        let mut hit = None;
        for i in 1..=10 {
            if let Some(result) = recognizer.process_frame(vec![i as f32]) {
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
