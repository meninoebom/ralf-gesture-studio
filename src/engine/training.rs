//! Training session management with audio cues.
//!
//! Implements a state machine for structured gesture training:
//! IDLE → COUNTDOWN → CAPTURING → RESTING → (repeat) → COMPLETE

use std::time::{Duration, Instant};

use rodio::source::{SineWave, Source};
use rodio::{OutputStream, Sink};

/// Training session states
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Waiting to start
    Idle,
    /// Counting down before capture (3, 2, 1...)
    Countdown,
    /// Actively recording gesture
    Capturing,
    /// Resting between repetitions
    Resting,
    /// All repetitions complete
    Complete,
}

/// Configuration for a gesture training session
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of repetitions to record
    pub reps: u32,
    /// Duration of each capture in seconds
    pub duration_secs: f32,
    /// Rest time between captures in seconds
    pub rest_secs: f32,
    /// Countdown time in seconds
    pub countdown_secs: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            reps: 5,
            duration_secs: 2.0, // Reduced from 3.0 for quicker training
            rest_secs: 2.0,
            countdown_secs: 2.0, // Reduced from 3.0 for quicker count-in
        }
    }
}

/// Play a one-shot audio tone (doesn't require storing audio state)
/// This is Send-safe because it creates and drops the audio stream immediately.
fn play_tone_oneshot(freq: f32, duration_secs: f32) {
    std::thread::spawn(move || {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                let source = SineWave::new(freq)
                    .take_duration(Duration::from_secs_f32(duration_secs))
                    .amplify(0.3);
                sink.append(source);
                sink.sleep_until_end();
            }
        }
    });
}

/// Play a countdown tick (short, low pitch)
pub fn play_tick() {
    play_tone_oneshot(300.0, 0.08);
}

/// Play capture start beep (long, high pitch)
pub fn play_capture_start() {
    play_tone_oneshot(800.0, 0.3);
}

/// Play capture end beep (long, medium pitch)
pub fn play_capture_end() {
    play_tone_oneshot(600.0, 0.3);
}

/// Play session complete (double ding)
pub fn play_complete() {
    std::thread::spawn(|| {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                let source = SineWave::new(1000.0)
                    .take_duration(Duration::from_secs_f32(0.15))
                    .amplify(0.3);
                sink.append(source);
                sink.sleep_until_end();
            }
            std::thread::sleep(Duration::from_millis(200));
            if let Ok(sink) = Sink::try_new(&handle) {
                let source = SineWave::new(1000.0)
                    .take_duration(Duration::from_secs_f32(0.15))
                    .amplify(0.3);
                sink.append(source);
                sink.sleep_until_end();
            }
        }
    });
}

/// Training session state machine
pub struct TrainingSession {
    /// Current state
    pub state: SessionState,
    /// Configuration
    pub config: TrainingConfig,
    /// Gesture ID being trained
    pub gesture_id: u32,
    /// Gesture name (for display)
    pub gesture_name: String,
    /// Number of completed repetitions
    pub completed_reps: u32,
    /// When the current state started
    state_start: Option<Instant>,
    /// Frames captured in current rep
    current_frames: Vec<Vec<f32>>,
    /// All completed examples (frames for each rep)
    pub completed_examples: Vec<Vec<Vec<f32>>>,
    /// Last countdown tick played (for tracking)
    last_tick: Option<u32>,
    /// Whether to play audio cues
    audio_enabled: bool,
}

impl TrainingSession {
    /// Create a new training session with audio enabled
    pub fn new() -> Self {
        Self::with_audio(true)
    }

    /// Create a new training session, optionally enabling audio
    pub fn with_audio(enable_audio: bool) -> Self {
        Self {
            state: SessionState::Idle,
            config: TrainingConfig::default(),
            gesture_id: 0,
            gesture_name: String::new(),
            completed_reps: 0,
            state_start: None,
            current_frames: Vec::new(),
            completed_examples: Vec::new(),
            last_tick: None,
            audio_enabled: enable_audio,
        }
    }

    /// Start a training session for a gesture
    pub fn start(&mut self, gesture_id: u32, gesture_name: &str, config: TrainingConfig) {
        self.gesture_id = gesture_id;
        self.gesture_name = gesture_name.to_string();
        self.config = config;
        self.completed_reps = 0;
        self.current_frames.clear();
        self.completed_examples.clear();
        self.last_tick = None;
        self.transition_to(SessionState::Countdown);
    }

    /// Cancel the current session, discarding all captured data
    pub fn cancel(&mut self) {
        self.transition_to(SessionState::Idle);
        self.current_frames.clear();
        self.completed_examples.clear();
    }

    /// Get elapsed time in current state
    pub fn elapsed(&self) -> Duration {
        self.state_start.map(|t| t.elapsed()).unwrap_or_default()
    }

    /// Get remaining time in current state (for countdown/capture/rest)
    pub fn remaining_secs(&self) -> f32 {
        let target = match self.state {
            SessionState::Countdown => self.config.countdown_secs,
            SessionState::Capturing => self.config.duration_secs,
            SessionState::Resting => self.config.rest_secs,
            _ => 0.0,
        };
        (target - self.elapsed().as_secs_f32()).max(0.0)
    }

    /// Get progress in current state (0.0 to 1.0)
    pub fn progress(&self) -> f32 {
        let target = match self.state {
            SessionState::Countdown => self.config.countdown_secs,
            SessionState::Capturing => self.config.duration_secs,
            SessionState::Resting => self.config.rest_secs,
            _ => return 0.0,
        };
        (self.elapsed().as_secs_f32() / target).min(1.0)
    }

    /// Get countdown value (3, 2, 1)
    pub fn countdown_value(&self) -> u32 {
        if self.state != SessionState::Countdown {
            return 0;
        }
        let remaining = self.remaining_secs();
        remaining.ceil() as u32
    }

    /// Update the session (call every frame)
    /// Returns true if state changed
    pub fn update(&mut self) -> bool {
        match self.state {
            SessionState::Countdown => {
                // Play tick sounds
                let current_tick = self.countdown_value();
                if self.last_tick != Some(current_tick) && current_tick > 0 {
                    if self.audio_enabled {
                        play_tick();
                    }
                    self.last_tick = Some(current_tick);
                }

                // Check if countdown complete
                if self.elapsed().as_secs_f32() >= self.config.countdown_secs {
                    self.transition_to(SessionState::Capturing);
                    if self.audio_enabled {
                        play_capture_start();
                    }
                    return true;
                }
            }
            SessionState::Capturing => {
                // Check if capture complete
                if self.elapsed().as_secs_f32() >= self.config.duration_secs {
                    self.on_capture_complete();
                    return true;
                }
            }
            SessionState::Resting => {
                // Check if rest complete
                if self.elapsed().as_secs_f32() >= self.config.rest_secs {
                    // Start next rep
                    self.transition_to(SessionState::Countdown);
                    return true;
                }
            }
            _ => {}
        }
        false
    }

    /// Add a frame during capture
    pub fn add_frame(&mut self, frame: Vec<f32>) {
        if self.state == SessionState::Capturing {
            self.current_frames.push(frame);
        }
    }

    /// Get the number of frames captured in current rep
    pub fn current_frame_count(&self) -> usize {
        self.current_frames.len()
    }

    /// Called when capture is complete
    fn on_capture_complete(&mut self) {
        // Play end sound
        if self.audio_enabled {
            play_capture_end();
        }

        // Save the captured frames
        if !self.current_frames.is_empty() {
            self.completed_examples
                .push(std::mem::take(&mut self.current_frames));
        }
        self.completed_reps += 1;

        // Check if all reps complete
        if self.completed_reps >= self.config.reps {
            self.transition_to(SessionState::Complete);
            // Play completion sound (slight delay built into the double-ding)
            if self.audio_enabled {
                play_complete();
            }
        } else {
            self.transition_to(SessionState::Resting);
        }
    }

    /// Transition to a new state
    fn transition_to(&mut self, new_state: SessionState) {
        self.state = new_state;
        self.state_start = Some(Instant::now());
        self.last_tick = None;
    }

    /// Check if session is active (not idle or complete)
    #[allow(dead_code)]
    pub fn is_active(&self) -> bool {
        matches!(
            self.state,
            SessionState::Countdown | SessionState::Capturing | SessionState::Resting
        )
    }

    /// Take the completed examples (consumes them)
    pub fn take_examples(&mut self) -> Vec<Vec<Vec<f32>>> {
        std::mem::take(&mut self.completed_examples)
    }
}

impl Default for TrainingSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread::sleep;

    // Helper to create a session without audio for faster tests
    fn test_session() -> TrainingSession {
        TrainingSession::with_audio(false)
    }

    #[test]
    fn test_session_creation() {
        let session = test_session();
        assert_eq!(session.state, SessionState::Idle);
        assert_eq!(session.completed_reps, 0);
    }

    #[test]
    fn test_session_start() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 3,
            duration_secs: 2.0,
            rest_secs: 1.0,
            countdown_secs: 1.0,
        };

        session.start(1, "wave", config);

        assert_eq!(session.state, SessionState::Countdown);
        assert_eq!(session.gesture_id, 1);
        assert_eq!(session.gesture_name, "wave");
        assert_eq!(session.completed_reps, 0);
    }

    #[test]
    fn test_session_cancel() {
        let mut session = test_session();
        session.start(1, "wave", TrainingConfig::default());

        session.cancel();

        assert_eq!(session.state, SessionState::Idle);
    }

    #[test]
    fn test_countdown_to_capture() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 1,
            duration_secs: 0.1,
            rest_secs: 0.1,
            countdown_secs: 0.05, // Very short for testing
        };

        session.start(1, "wave", config);
        assert_eq!(session.state, SessionState::Countdown);

        // Wait for countdown
        sleep(Duration::from_millis(100));
        session.update();

        assert_eq!(session.state, SessionState::Capturing);
    }

    #[test]
    fn test_capture_to_complete() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 1,
            duration_secs: 0.05,
            rest_secs: 0.1,
            countdown_secs: 0.05,
        };

        session.start(1, "wave", config);

        // Wait through countdown
        sleep(Duration::from_millis(100));
        session.update();
        assert_eq!(session.state, SessionState::Capturing);

        // Add some frames
        session.add_frame(vec![1.0, 2.0]);
        session.add_frame(vec![3.0, 4.0]);

        // Wait for capture to complete
        sleep(Duration::from_millis(100));
        session.update();

        // With only 1 rep, should go to Complete
        assert_eq!(session.state, SessionState::Complete);
        assert_eq!(session.completed_reps, 1);
        assert_eq!(session.completed_examples.len(), 1);
    }

    #[test]
    fn test_multiple_reps() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 2,
            duration_secs: 0.05,
            rest_secs: 0.05,
            countdown_secs: 0.05,
        };

        session.start(1, "wave", config);

        // First rep: countdown
        sleep(Duration::from_millis(100));
        session.update();
        assert_eq!(session.state, SessionState::Capturing);

        // First rep: capture
        session.add_frame(vec![1.0]);
        sleep(Duration::from_millis(100));
        session.update();
        assert_eq!(session.state, SessionState::Resting);
        assert_eq!(session.completed_reps, 1);

        // Rest period
        sleep(Duration::from_millis(100));
        session.update();
        assert_eq!(session.state, SessionState::Countdown);

        // Second rep: countdown
        sleep(Duration::from_millis(100));
        session.update();
        assert_eq!(session.state, SessionState::Capturing);

        // Second rep: capture
        session.add_frame(vec![2.0]);
        sleep(Duration::from_millis(100));
        session.update();

        assert_eq!(session.state, SessionState::Complete);
        assert_eq!(session.completed_reps, 2);
    }

    #[test]
    fn test_progress() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 1,
            duration_secs: 1.0,
            rest_secs: 1.0,
            countdown_secs: 1.0,
        };

        session.start(1, "wave", config);

        // Progress should start near 0
        assert!(session.progress() < 0.2);

        // After some time, progress should increase
        sleep(Duration::from_millis(300));
        let progress = session.progress();
        assert!(progress > 0.2 && progress < 0.5);
    }

    #[test]
    fn test_remaining_secs() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 1,
            duration_secs: 2.0,
            rest_secs: 1.0,
            countdown_secs: 3.0,
        };

        session.start(1, "wave", config);

        // Should have close to 3 seconds remaining
        let remaining = session.remaining_secs();
        assert!(remaining > 2.5 && remaining <= 3.0);
    }

    #[test]
    fn test_is_active() {
        let mut session = test_session();

        assert!(!session.is_active());

        session.start(1, "wave", TrainingConfig::default());
        assert!(session.is_active());

        session.cancel();
        assert!(!session.is_active());
    }

    #[test]
    fn test_take_examples() {
        let mut session = test_session();
        let config = TrainingConfig {
            reps: 1,
            duration_secs: 0.05,
            rest_secs: 0.1,
            countdown_secs: 0.05,
        };

        session.start(1, "wave", config);

        // Go through full session
        sleep(Duration::from_millis(100));
        session.update();
        session.add_frame(vec![1.0, 2.0]);
        session.add_frame(vec![3.0, 4.0]);
        sleep(Duration::from_millis(100));
        session.update();

        assert_eq!(session.state, SessionState::Complete);

        let examples = session.take_examples();
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].len(), 2); // 2 frames

        // After taking, should be empty
        assert!(session.completed_examples.is_empty());
    }
}
