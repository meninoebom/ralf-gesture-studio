//! Real-time gesture recognizer.
//!
//! Performs continuous DTW matching against stored examples and fires hits
//! when gestures are recognized.
//!
//! ## Hit Detection Logic (Edge Detection)
//!
//! A hit fires when BOTH conditions are met:
//! 1. Distance crosses below threshold (edge detection - the moment of recognition)
//! 2. Not in cooldown from previous hit (rate limiting)
//!
//! This is "edge detection" - we fire at the moment distance dips below threshold,
//! not when it stays below. This catches quick gestures that only briefly match.
//!
//! ## Configuration
//!
//! - `cooldown_ms`: Minimum time between hits for same gesture (prevents rapid-fire)

use std::time::{Duration, Instant};

use super::buffer::FrameBuffer;
use super::dtw::{dtw_distance_normalized, Frame, Sequence};

/// Configuration for gesture recognition behavior
#[derive(Debug, Clone)]
pub struct RecognitionConfig {
    /// Cooldown: minimum time between hits for same gesture (ms)
    /// After a hit fires, ignore this gesture until cooldown expires.
    /// Prevents rapid-fire hits. Lower = faster repetition allowed.
    /// Recommended: 300-500ms for dance movements
    pub cooldown_ms: u64,
}

impl Default for RecognitionConfig {
    fn default() -> Self {
        Self {
            cooldown_ms: 400,  // 400ms cooldown - allows ~2.5 hits/sec
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
    /// Whether distance was below threshold on previous frame (for edge detection)
    /// We only fire on the transition from above to below
    was_below_threshold: bool,
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
            was_below_threshold: false,
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

    /// Check if this gesture is in cooldown period
    pub fn in_cooldown(&self, cooldown_duration: Duration) -> bool {
        match self.last_hit_time {
            Some(time) => time.elapsed() < cooldown_duration,
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
    /// Recognition configuration (cooldown timing)
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

    /// Set cooldown time
    pub fn set_cooldown_ms(&mut self, ms: u64) {
        self.config.cooldown_ms = ms;
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

            // Edge detection: fire when distance CROSSES below threshold
            let below_threshold = best_distance < gesture.threshold;
            let cooldown_duration = Duration::from_millis(self.config.cooldown_ms);

            // Hit fires when:
            // 1. Distance is below threshold NOW
            // 2. Distance was NOT below threshold on previous frame (edge detection)
            // 3. Not in cooldown from previous hit
            let is_crossing_down = below_threshold && !gesture.was_below_threshold;
            let not_in_cooldown = !gesture.in_cooldown(cooldown_duration);
            let is_hit = is_crossing_down && not_in_cooldown;

            // Update state for next frame's edge detection
            gesture.was_below_threshold = below_threshold;

            if is_hit {
                gesture.record_hit();

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

        // "Bounce back" pattern: After a hit, clear the buffer to reset to neutral state
        // This prevents the gesture from lingering in the sliding window
        // The system returns to "not enough data" state and must collect fresh frames
        //
        // IMPORTANT: We do NOT reset was_below_threshold here. Keeping it true means
        // the user must move away (distance goes above threshold) before another hit
        // can fire. This prevents infinite hit loops when the user stays in position.
        if best_hit.is_some() {
            self.buffer.clear();
            // Clear display distances while buffer refills
            for gesture in &mut self.gestures {
                gesture.current_distance = None;
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

    /// Get the current cooldown period in milliseconds
    #[allow(dead_code)]
    pub fn cooldown_ms(&self) -> u64 {
        self.config.cooldown_ms
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
    fn test_gesture_cooldown() {
        let mut gesture = GestureState::new(1, "wave", "/gesture/1", 15.0);

        // Not in cooldown initially
        assert!(!gesture.in_cooldown(Duration::from_millis(500)));

        // Record a hit
        gesture.record_hit();

        // Now in cooldown
        assert!(gesture.in_cooldown(Duration::from_millis(500)));

        // After waiting, no longer in cooldown
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
        let config = RecognitionConfig { cooldown_ms: 500 };
        let mut recognizer = Recognizer::with_config(1000, 5, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0); // Low threshold (normalized scale)

        // Add a simple example
        let example = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Feed matching frames and track if we got a hit
        let mut hit_result = None;
        for i in 1..=5 {
            if let Some(result) = recognizer.process_frame(vec![i as f32]) {
                hit_result = Some(result);
            }
        }

        // Should have detected a match (hit fired)
        // Note: After a hit, buffer clears and current_distance becomes None
        // So we verify the hit occurred, not the final distance state
        assert!(hit_result.is_some(), "Should have detected a matching gesture");
        let result = hit_result.unwrap();
        assert_eq!(result.gesture_id, Some(1));
        assert_eq!(result.gesture_name, Some("wave".to_string()));
    }

    #[test]
    fn test_recognizer_cooldown_prevents_double_trigger() {
        let config = RecognitionConfig { cooldown_ms: 1000 };
        let mut recognizer = Recognizer::with_config(1000, 3, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0); // Normalized scale

        let example = vec![vec![1.0], vec![1.0], vec![1.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // First match should hit (edge detection fires on first crossing)
        for _ in 0..3 {
            recognizer.process_frame(vec![1.0]);
        }
        let gesture = recognizer.get_gesture(1).unwrap();
        assert!(gesture.last_hit_time.is_some());

        // Immediately after, should be in cooldown
        assert!(gesture.in_cooldown(Duration::from_millis(1000)));
    }

    #[test]
    fn test_edge_detection_fires_on_crossing() {
        let config = RecognitionConfig { cooldown_ms: 500 };
        let mut recognizer = Recognizer::with_config(1000, 3, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0);

        let example = vec![vec![1.0], vec![1.0], vec![1.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Fill buffer with non-matching frames first (high distance)
        for _ in 0..3 {
            recognizer.process_frame(vec![100.0]); // Far from example
        }

        // First matching frame should trigger hit immediately (edge detection)
        recognizer.process_frame(vec![1.0]);
        // Note: May need 3 frames to get proper window for matching
        recognizer.process_frame(vec![1.0]);
        recognizer.process_frame(vec![1.0]);

        // Should have fired by now since we crossed below threshold
        let gesture = recognizer.get_gesture(1).unwrap();
        assert!(gesture.last_hit_time.is_some(), "Edge detection should fire when crossing threshold");
    }

    #[test]
    fn test_no_hit_while_staying_below_threshold() {
        let config = RecognitionConfig { cooldown_ms: 100 };
        let mut recognizer = Recognizer::with_config(1000, 3, config);
        recognizer.add_gesture(1, "wave", "/gesture/1", 5.0);

        let example = vec![vec![1.0], vec![1.0], vec![1.0]];
        recognizer.add_example(1, example);

        recognizer.start();

        // Initial frames to get below threshold
        for _ in 0..3 {
            recognizer.process_frame(vec![1.0]);
        }

        // Wait for cooldown to expire
        std::thread::sleep(Duration::from_millis(150));

        // Continue with matching frames - should NOT fire again
        // because we never went above threshold (no edge crossing)
        let result = recognizer.process_frame(vec![1.0]);
        assert!(result.is_none(), "Should not fire without crossing above threshold first");
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
