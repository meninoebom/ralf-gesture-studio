//! Real-time gesture recognizer.
//!
//! Performs continuous DTW matching against stored examples and fires hits
//! when gestures are recognized.
//!
//! ## Hit Detection Logic
//!
//! A hit fires when ALL conditions are met:
//! 1. Distance < Threshold (gesture matches)
//! 2. Distance has been below threshold for `confirm_ms` (debounce - filters noise)
//! 3. Not in cooldown from previous hit (rate limiting)
//!
//! ## Configuration
//!
//! - `confirm_ms`: How long distance must stay below threshold before firing (debounce)
//! - `refractory_ms`: Minimum time between hits for same gesture (cooldown/rate limit)

use std::time::{Duration, Instant};

use super::buffer::FrameBuffer;
use super::dtw::{dtw_distance_normalized, Frame, Sequence};

/// Configuration for gesture recognition behavior
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    /// Debounce: distance must stay below threshold for this long before hit fires (ms)
    /// Higher = more stable, filters oscillation. Lower = more responsive.
    /// Recommended: 50-150ms
    pub confirm_ms: u64,

    /// Cooldown: minimum time between hits for same gesture (ms)
    /// Prevents rapid-fire hits. Lower = faster repetition allowed.
    /// Recommended: 300-800ms
    pub refractory_ms: u64,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            confirm_ms: 80,      // 80ms debounce - filters brief noise
            refractory_ms: 500,  // 500ms cooldown - allows 2 hits/sec
        }
    }
}

/// Result of a recognition check
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// ID of the gesture that was recognized
    pub gesture_id: Option<u32>,
    /// Name of the gesture
    pub gesture_name: Option<String>,
    /// Distance to the best matching example
    pub distance: f32,
    /// The threshold that was used (for debugging/display)
    #[allow(dead_code)]
    pub threshold: f32,
}

/// State for a single gesture being tracked
#[derive(Debug)]
pub struct GestureState {
    /// Gesture ID
    pub id: u32,
    /// Gesture name
    pub name: String,
    /// OSC address to send on hit
    pub osc_address: String,
    /// Recognition threshold (distance must be below this to match)
    pub threshold: f32,
    /// Training examples for this gesture
    pub examples: Vec<Sequence>,
    /// Current distance to best match (for display)
    pub current_distance: Option<f32>,
    /// Last time this gesture fired a hit (for cooldown)
    pub last_hit_time: Option<Instant>,
    /// When distance first dropped below threshold (for debounce)
    /// None = currently above threshold
    below_threshold_since: Option<Instant>,
}

impl GestureState {
    /// Create a new gesture state
    pub fn new(id: u32, name: &str, osc_address: &str, threshold: f32) -> Self {
        Self {
            id,
            name: name.to_string(),
            osc_address: osc_address.to_string(),
            threshold,
            examples: Vec::new(),
            current_distance: None,
            last_hit_time: None,
            below_threshold_since: None,
        }
    }

    /// Check if distance has been below threshold long enough (debounce check)
    pub fn is_confirmed(&self, confirm_duration: Duration) -> bool {
        match self.below_threshold_since {
            Some(since) => since.elapsed() >= confirm_duration,
            None => false,
        }
    }

    /// Add a training example
    pub fn add_example(&mut self, example: Sequence) {
        self.examples.push(example);
    }

    /// Check if this gesture has training examples
    pub fn has_examples(&self) -> bool {
        !self.examples.is_empty()
    }

    /// Check if this gesture is in refractory period
    pub fn in_refractory(&self, refractory_duration: Duration) -> bool {
        match self.last_hit_time {
            Some(time) => time.elapsed() < refractory_duration,
            None => false,
        }
    }

    /// Record that this gesture was hit
    pub fn record_hit(&mut self) {
        self.last_hit_time = Some(Instant::now());
    }

    /// Get time since last hit in milliseconds
    #[allow(dead_code)]
    pub fn ms_since_last_hit(&self) -> Option<u64> {
        self.last_hit_time.map(|t| t.elapsed().as_millis() as u64)
    }
}

/// Real-time gesture recognizer
#[derive(Debug)]
pub struct Recognizer {
    /// Frame buffer for incoming data
    pub buffer: FrameBuffer,
    /// Gestures being tracked
    gestures: Vec<GestureState>,
    /// Recognition configuration (debounce, cooldown)
    config: RecognitionConfig,
    /// Whether recognition is active
    active: bool,
    /// Window size for matching (in frames)
    window_size: usize,
}

impl Recognizer {
    /// Create a new recognizer with default config
    ///
    /// # Arguments
    /// * `buffer_size` - Maximum frames to keep in buffer
    /// * `window_size` - Number of frames to use for matching
    #[allow(dead_code)]
    pub fn new(buffer_size: usize, window_size: usize) -> Self {
        Self::with_config(buffer_size, window_size, RecognitionConfig::default())
    }

    /// Create a new recognizer with custom config
    pub fn with_config(buffer_size: usize, window_size: usize, config: RecognitionConfig) -> Self {
        Self {
            buffer: FrameBuffer::new(buffer_size),
            gestures: Vec::new(),
            config,
            active: false,
            window_size,
        }
    }

    /// Get the current recognition config
    #[allow(dead_code)]
    pub fn config(&self) -> &RecognitionConfig {
        &self.config
    }

    /// Update recognition config
    #[allow(dead_code)]
    pub fn set_config(&mut self, config: RecognitionConfig) {
        self.config = config;
    }

    /// Set debounce time (confirm_ms)
    pub fn set_confirm_ms(&mut self, ms: u64) {
        self.config.confirm_ms = ms;
    }

    /// Set cooldown time (refractory_ms)
    pub fn set_refractory_ms(&mut self, ms: u64) {
        self.config.refractory_ms = ms;
    }

    /// Add a gesture to track
    pub fn add_gesture(&mut self, id: u32, name: &str, osc_address: &str, threshold: f32) {
        self.gestures.push(GestureState::new(id, name, osc_address, threshold));
    }

    /// Get a mutable reference to a gesture by ID
    pub fn get_gesture_mut(&mut self, id: u32) -> Option<&mut GestureState> {
        self.gestures.iter_mut().find(|g| g.id == id)
    }

    /// Get a reference to a gesture by ID
    pub fn get_gesture(&self, id: u32) -> Option<&GestureState> {
        self.gestures.iter().find(|g| g.id == id)
    }

    /// Get all gestures
    #[allow(dead_code)]
    pub fn gestures(&self) -> &[GestureState] {
        &self.gestures
    }

    /// Add a training example to a gesture
    pub fn add_example(&mut self, gesture_id: u32, example: Sequence) -> bool {
        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.add_example(example);
            true
        } else {
            false
        }
    }

    /// Start recognition
    pub fn start(&mut self) {
        self.active = true;
    }

    /// Stop recognition
    pub fn stop(&mut self) {
        self.active = false;
    }

    /// Check if recognition is active
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Process a new incoming frame
    ///
    /// Adds the frame to the buffer and, if active, checks for gesture matches.
    /// Returns a hit result if a gesture was recognized.
    pub fn process_frame(&mut self, frame: Frame) -> Option<RecognitionResult> {
        self.buffer.push(frame);

        if !self.active {
            return None;
        }

        // Need enough frames for matching
        if self.buffer.len() < self.window_size / 2 {
            return None;
        }

        // Get the current window of frames
        let window = self.buffer.recent_frames(self.window_size);

        // Check each gesture
        let mut best_hit: Option<RecognitionResult> = None;

        for gesture in &mut self.gestures {
            if !gesture.has_examples() {
                gesture.current_distance = None;
                continue;
            }

            // Find best match among all examples
            let mut best_distance = f32::INFINITY;
            for example in &gesture.examples {
                let distance = dtw_distance_normalized(&window, example);
                if distance < best_distance {
                    best_distance = distance;
                }
            }

            gesture.current_distance = Some(best_distance);

            // Track debounce timing: when did we first go below threshold?
            let below_threshold = best_distance < gesture.threshold;

            if below_threshold {
                // Start tracking if we just dropped below threshold
                if gesture.below_threshold_since.is_none() {
                    gesture.below_threshold_since = Some(Instant::now());
                }
            } else {
                // Reset tracking when we go above threshold
                gesture.below_threshold_since = None;
            }

            // Hit fires when:
            // 1. Below threshold
            // 2. Has been below threshold for confirm_ms (debounce)
            // 3. Not in cooldown (refractory period)
            let confirm_duration = Duration::from_millis(self.config.confirm_ms);
            let refractory_duration = Duration::from_millis(self.config.refractory_ms);

            let is_confirmed = gesture.is_confirmed(confirm_duration);
            let not_in_cooldown = !gesture.in_refractory(refractory_duration);
            let is_hit = below_threshold && is_confirmed && not_in_cooldown;

            if is_hit {
                gesture.record_hit();
                // Reset debounce tracking after hit (so we need to re-confirm for next hit)
                gesture.below_threshold_since = None;

                let result = RecognitionResult {
                    gesture_id: Some(gesture.id),
                    gesture_name: Some(gesture.name.clone()),
                    distance: best_distance,
                    threshold: gesture.threshold,
                };

                // Keep track of best hit (lowest distance)
                if best_hit.is_none() || best_distance < best_hit.as_ref().unwrap().distance {
                    best_hit = Some(result);
                }
            }
        }

        best_hit
    }

    /// Get current distances for all gestures (for display)
    /// Returns: (id, name, current_distance, threshold)
    pub fn current_distances(&self) -> Vec<(u32, String, Option<f32>, f32)> {
        self.gestures
            .iter()
            .map(|g| (g.id, g.name.clone(), g.current_distance, g.threshold))
            .collect()
    }

    /// Update threshold for a gesture
    pub fn set_threshold(&mut self, gesture_id: u32, threshold: f32) {
        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.threshold = threshold;
        }
    }

    /// Clear all gesture examples
    #[allow(dead_code)]
    pub fn clear_examples(&mut self, gesture_id: u32) {
        if let Some(gesture) = self.get_gesture_mut(gesture_id) {
            gesture.examples.clear();
        }
    }

    /// Get the number of examples for a gesture
    pub fn example_count(&self, gesture_id: u32) -> usize {
        self.get_gesture(gesture_id)
            .map(|g| g.examples.len())
            .unwrap_or(0)
    }

    /// Get the current refractory period in milliseconds
    #[allow(dead_code)]
    pub fn refractory_ms(&self) -> u64 {
        self.config.refractory_ms
    }

    /// Get the current confirm (debounce) period in milliseconds
    #[allow(dead_code)]
    pub fn confirm_ms(&self) -> u64 {
        self.config.confirm_ms
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
    /// Create a new hit log
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_entries,
        }
    }

    /// Record a hit
    pub fn record(&mut self, gesture_id: u32, gesture_name: &str, distance: f32, osc_address: &str) {
        let entry = HitLogEntry {
            timestamp: Instant::now(),
            gesture_id,
            gesture_name: gesture_name.to_string(),
            distance,
            osc_address: osc_address.to_string(),
        };

        self.entries.push(entry);

        // Trim if too many entries
        if self.entries.len() > self.max_entries {
            self.entries.remove(0);
        }
    }

    /// Get recent entries (newest first)
    pub fn recent(&self, count: usize) -> Vec<&HitLogEntry> {
        self.entries
            .iter()
            .rev()
            .take(count)
            .collect()
    }

    /// Get total number of hits
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear the log
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
        gesture.add_example(vec![vec![1.0], vec![2.0]]);

        assert!(gesture.has_examples());
        assert_eq!(gesture.examples.len(), 1);
    }

    #[test]
    fn test_gesture_refractory() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 15.0);

        // Not in refractory initially
        assert!(!gesture.in_refractory(Duration::from_millis(500)));

        // Record a hit
        gesture.record_hit();

        // Now in refractory
        assert!(gesture.in_refractory(Duration::from_millis(500)));

        // After waiting, no longer in refractory
        std::thread::sleep(Duration::from_millis(100));
        assert!(!gesture.in_refractory(Duration::from_millis(50)));
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
        assert!(recognizer.get_gesture(1).is_some());
        assert!(recognizer.get_gesture(2).is_some());
        assert!(recognizer.get_gesture(3).is_none());
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

        // Add example
        let example = vec![vec![1.0]; 10];
        recognizer.add_example(1, example);

        // Process frame while inactive - should return None
        let result = recognizer.process_frame(vec![1.0]);
        assert!(result.is_none());
    }

    #[test]
    fn test_recognizer_detects_match() {
        // Use zero confirm_ms for instant hit detection in tests
        let config = RecognitionConfig {
            confirm_ms: 0,
            refractory_ms: 500,
        };
        let mut recognizer = Recognizer::with_config(1000, 5, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0); // Low threshold (normalized scale)

        // Add a simple example
        let example = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Feed matching frames
        for i in 1..=5 {
            recognizer.process_frame(vec![i as f32]);
        }

        // Check the distance was computed
        let gesture = recognizer.get_gesture(1).unwrap();
        assert!(gesture.current_distance.is_some());
        assert_eq!(gesture.current_distance.unwrap(), 0.0); // Exact match
    }

    #[test]
    fn test_recognizer_refractory_prevents_double_trigger() {
        // Use zero confirm_ms for instant hit, 1 second cooldown
        let config = RecognitionConfig {
            confirm_ms: 0,
            refractory_ms: 1000,
        };
        let mut recognizer = Recognizer::with_config(1000, 3, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0); // Normalized scale

        let example = vec![vec![1.0], vec![1.0], vec![1.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // First match should hit
        for _ in 0..3 {
            recognizer.process_frame(vec![1.0]);
        }
        let gesture = recognizer.get_gesture(1).unwrap();
        assert!(gesture.last_hit_time.is_some());

        // Immediately after, should be in refractory
        assert!(gesture.in_refractory(Duration::from_millis(1000)));
    }

    #[test]
    fn test_debounce_prevents_instant_hit() {
        // Use 100ms debounce
        let config = RecognitionConfig {
            confirm_ms: 100,
            refractory_ms: 500,
        };
        let mut recognizer = Recognizer::with_config(1000, 3, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0);

        let example = vec![vec![1.0], vec![1.0], vec![1.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // First frames should NOT trigger hit immediately due to debounce
        let result = recognizer.process_frame(vec![1.0]);
        assert!(result.is_none()); // No hit yet - need to wait for confirm_ms

        recognizer.process_frame(vec![1.0]);
        let result = recognizer.process_frame(vec![1.0]);
        assert!(result.is_none()); // Still no hit - confirm_ms not elapsed

        // After waiting, the hit should fire
        std::thread::sleep(Duration::from_millis(110));
        let result = recognizer.process_frame(vec![1.0]);
        assert!(result.is_some()); // Now we get the hit
    }

    #[test]
    fn test_hit_log() {
        let mut log = HitLog::new(10);

        log.record(1, "wave", 42.0, "/gesture/1");
        log.record(2, "jump", 55.0, "/gesture/2");

        assert_eq!(log.len(), 2);

        let recent = log.recent(5);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].gesture_name, "jump"); // Newest first
        assert_eq!(recent[1].gesture_name, "wave");
    }

    #[test]
    fn test_hit_log_max_entries() {
        let mut log = HitLog::new(3);

        for i in 0..5 {
            log.record(i, &format!("gesture{}", i), i as f32, "/test");
        }

        assert_eq!(log.len(), 3);

        // Should have only the last 3
        let recent = log.recent(5);
        assert_eq!(recent[0].gesture_name, "gesture4");
    }
}
