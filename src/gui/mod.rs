use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tauri::State;

use crate::model::{Vocabulary, Example};
use crate::model::{save_vocabulary as model_save_vocabulary, load_vocabulary as model_load_vocabulary, default_vocabulary_dir};
use crate::osc::{OscReceiverHandle, ConnectionStatus, OscSender, SenderStatus};
use crate::engine::{Recognizer, HitLog, TrainingSession, TrainingConfig, SessionState, RecognitionConfig, compute_threshold_stats};

/// The two modes of the application
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AppMode {
    Training,
    Performance,
}

/// Application state shared between Tauri commands
pub struct AppState {
    /// The currently loaded vocabulary
    vocabulary: Vocabulary,
    /// Current file path (None if unsaved)
    file_path: Option<PathBuf>,
    /// Whether there are unsaved changes
    dirty: bool,
    /// Current application mode
    mode: AppMode,
    /// Handle to the OSC receiver
    osc_receiver: OscReceiverHandle,
    /// OSC sender for hit messages
    osc_sender: OscSender,
    /// Gesture recognizer
    recognizer: Recognizer,
    /// Hit log for performance mode
    hit_log: HitLog,
    /// Currently selected gesture ID for training
    selected_gesture_id: Option<u32>,
    /// Training session
    training_session: TrainingSession,
    /// Training configuration
    training_config: TrainingConfig,
    /// Recognition config (cooldown timing)
    recognition_config: RecognitionConfig,
}

impl AppState {
    pub fn new(osc_receiver: OscReceiverHandle) -> Self {
        // Create a demo vocabulary
        let mut vocabulary = Vocabulary::new("Demo Vocabulary");
        vocabulary.add_gesture("wave");
        vocabulary.add_gesture("jump");
        vocabulary.add_gesture("spin");

        // Create OSC sender
        let osc_sender = OscSender::new(&vocabulary.output.host, vocabulary.output.port);

        // Create recognizer
        let recognition_config = RecognitionConfig::default();
        let mut recognizer = Recognizer::with_config(600, 180, recognition_config.clone());

        // Add gestures to recognizer
        for gesture in &vocabulary.gestures {
            recognizer.add_gesture(
                gesture.id,
                &gesture.name,
                &gesture.osc_address,
                gesture.threshold,
            );
        }

        let selected_gesture_id = vocabulary.gestures.first().map(|g| g.id);

        Self {
            vocabulary,
            file_path: None,
            dirty: false,
            mode: AppMode::Training,
            osc_receiver,
            osc_sender,
            recognizer,
            hit_log: HitLog::new(100),
            selected_gesture_id,
            // Use with_audio(false) to avoid Send/Sync issues with Tauri state
            training_session: TrainingSession::with_audio(false),
            training_config: TrainingConfig::default(),
            recognition_config,
        }
    }

    /// Sync recognizer with vocabulary
    fn sync_recognizer(&mut self) {
        let was_active = self.recognizer.is_active();

        self.recognizer = Recognizer::with_config(600, 180, self.recognition_config.clone());

        for gesture in &self.vocabulary.gestures {
            self.recognizer.add_gesture(
                gesture.id,
                &gesture.name,
                &gesture.osc_address,
                gesture.threshold,
            );

            for example in &gesture.examples {
                self.recognizer.add_example(gesture.id, example.frames.clone());
            }
        }

        if was_active {
            self.recognizer.start();
        }
    }

    /// Save completed training examples
    fn save_training_examples(&mut self) {
        let gesture_id = self.training_session.gesture_id;
        let examples = self.training_session.take_examples();

        for frames in examples {
            if frames.is_empty() {
                continue;
            }

            let frame_count = frames.len();
            let example = Example::new(frames.clone(), frame_count as u64);

            if let Some(gesture) = self.vocabulary.get_gesture_mut(gesture_id) {
                gesture.add_example(example);
            }

            self.recognizer.add_example(gesture_id, frames);
        }

        // Compute statistical threshold
        self.compute_gesture_statistics(gesture_id);

        self.dirty = true;
        self.auto_save();
    }

    /// Compute threshold statistics for a gesture
    fn compute_gesture_statistics(&mut self, gesture_id: u32) {
        let examples: Vec<Vec<Vec<f32>>> = self.vocabulary
            .get_gesture(gesture_id)
            .map(|g| g.examples.iter().map(|e| e.frames.clone()).collect())
            .unwrap_or_default();

        if examples.len() < 2 {
            return;
        }

        let coefficient = self.vocabulary
            .get_gesture(gesture_id)
            .map(|g| g.threshold_coefficient)
            .unwrap_or(2.0);

        if let Some(stats) = compute_threshold_stats(&examples, coefficient) {
            if let Some(gesture) = self.vocabulary.get_gesture_mut(gesture_id) {
                gesture.update_statistics(stats.mean, stats.std);
                self.recognizer.set_threshold(gesture_id, gesture.threshold);
            }
        }
    }

    /// Auto-save if we have a file path
    fn auto_save(&mut self) {
        if let Some(ref path) = self.file_path {
            if let Err(e) = model_save_vocabulary(&self.vocabulary, path) {
                eprintln!("Auto-save failed: {}", e);
            } else {
                self.dirty = false;
            }
        }
    }

    /// Mark as dirty and auto-save
    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.auto_save();
    }

    /// Process incoming OSC frames and update state
    fn process_frames(&mut self) {
        let mut frames = self.osc_receiver.poll();

        // In Performance mode, only process the most recent frame
        if self.mode == AppMode::Performance && frames.len() > 1 {
            let last_frame = frames.pop().unwrap();
            frames.clear();
            frames.push(last_frame);
        }

        for frame in frames {
            // If training, add frames to session
            if self.training_session.state == SessionState::Capturing {
                self.training_session.add_frame(frame.clone());
            }

            // Feed to recognizer
            if let Some(result) = self.recognizer.process_frame(frame.clone()) {
                if let (Some(id), Some(name)) = (result.gesture_id, result.gesture_name.clone()) {
                    let osc_address = self.recognizer
                        .get_gesture(id)
                        .map(|g| g.osc_address.clone())
                        .unwrap_or_else(|| format!("/gesture/{}", id));

                    self.hit_log.record(id, &name, result.distance, &osc_address);
                    let _ = self.osc_sender.send_hit(&osc_address);
                }
            }
        }

        // Update training session
        self.training_session.update();

        // Check if training completed
        if self.training_session.state == SessionState::Complete {
            self.save_training_examples();
            self.training_session = TrainingSession::new();
        }
    }
}

// =============================================================================
// State DTOs for frontend
// =============================================================================

#[derive(Serialize)]
pub struct StateResponse {
    vocabulary: VocabularyDto,
    file_path: Option<String>,
    dirty: bool,
    mode: AppMode,
    selected_gesture_id: Option<u32>,
    osc_status: OscStatusDto,
    training: TrainingDto,
    monitor: MonitorDto,
    hit_log: HitLogDto,
}

#[derive(Serialize)]
pub struct VocabularyDto {
    name: String,
    gestures: Vec<GestureDto>,
}

#[derive(Serialize)]
pub struct GestureDto {
    id: u32,
    name: String,
    osc_address: String,
    threshold: f32,
    examples: Vec<ExampleDto>,
}

#[derive(Serialize)]
pub struct ExampleDto {
    frame_count: usize,
}

#[derive(Serialize)]
pub struct OscStatusDto {
    input_status: String,
    ms_since_last_frame: u64,
    input_error: Option<String>,
    frame_count: u64,
    output_status: String,
    ms_since_last_send: u64,
    output_error: Option<String>,
    send_count: u64,
}

#[derive(Serialize)]
pub struct TrainingDto {
    state: String,
    countdown: u32,
    current_rep: u32,
    total_reps: u32,
    completed_reps: u32,
    remaining: f32,
    progress: f32,
    frame_count: usize,
}

#[derive(Serialize)]
pub struct MonitorDto {
    active: bool,
    buffer_len: usize,
    window_size: usize,
    total_examples: usize,
    gestures: Vec<GestureMonitorDto>,
    recent_hit: Option<String>,
}

#[derive(Serialize)]
pub struct GestureMonitorDto {
    id: u32,
    name: String,
    example_count: usize,
    distance: Option<f32>,
    threshold: f32,
    auto_mode: bool,
    recent_hit: bool,
}

#[derive(Serialize)]
pub struct HitLogDto {
    total: usize,
    entries: Vec<HitEntryDto>,
}

#[derive(Serialize)]
pub struct HitEntryDto {
    name: String,
    ms_ago: u64,
    recent: bool,
}

// =============================================================================
// Tauri Commands
// =============================================================================

#[tauri::command]
pub fn get_state(state: State<Arc<Mutex<AppState>>>) -> Result<StateResponse, String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    // Process any pending OSC frames
    app.process_frames();

    // Build response
    let vocabulary = VocabularyDto {
        name: app.vocabulary.name.clone(),
        gestures: app.vocabulary.gestures.iter().map(|g| GestureDto {
            id: g.id,
            name: g.name.clone(),
            osc_address: g.osc_address.clone(),
            threshold: g.threshold,
            examples: g.examples.iter().map(|e| ExampleDto {
                frame_count: e.frames.len(),
            }).collect(),
        }).collect(),
    };

    let input_status = match app.osc_receiver.state.status {
        ConnectionStatus::Stopped => "Stopped".to_string(),
        ConnectionStatus::Listening => "Listening".to_string(),
        ConnectionStatus::Receiving => "Receiving".to_string(),
        ConnectionStatus::Error => "Error".to_string(),
    };

    let output_status = match app.osc_sender.state.status {
        SenderStatus::Ready => "Ready".to_string(),
        SenderStatus::Sent => "Sent".to_string(),
        SenderStatus::Error => "Error".to_string(),
    };

    let osc_status = OscStatusDto {
        input_status,
        ms_since_last_frame: app.osc_receiver.ms_since_last_frame().unwrap_or(0) as u64,
        input_error: app.osc_receiver.state.error_message.clone(),
        frame_count: app.osc_receiver.state.frame_count,
        output_status,
        ms_since_last_send: app.osc_sender.ms_since_last_send().unwrap_or(0) as u64,
        output_error: app.osc_sender.state.error_message.clone(),
        send_count: app.osc_sender.state.send_count,
    };

    let training_state = match app.training_session.state {
        SessionState::Idle => "idle",
        SessionState::Countdown => "countdown",
        SessionState::Capturing => "capturing",
        SessionState::Resting => "resting",
        SessionState::Complete => "complete",
    };

    let training = TrainingDto {
        state: training_state.to_string(),
        countdown: app.training_session.countdown_value(),
        current_rep: app.training_session.completed_reps + 1,
        total_reps: app.training_session.config.reps,
        completed_reps: app.training_session.completed_reps,
        remaining: app.training_session.remaining_secs(),
        progress: app.training_session.progress(),
        frame_count: app.training_session.current_frame_count(),
    };

    let distances = app.recognizer.current_distances();
    let recent_hits = app.hit_log.recent(1);
    let recent_hit_name = recent_hits.first()
        .filter(|h| h.timestamp.elapsed().as_millis() < 300)
        .map(|h| h.gesture_name.clone());

    let monitor = MonitorDto {
        active: app.recognizer.is_active(),
        buffer_len: app.recognizer.buffer.len(),
        window_size: app.recognizer.window_size(),
        total_examples: app.recognizer.total_example_count(),
        gestures: distances.iter().map(|(id, name, dist, thresh)| {
            let gesture = app.vocabulary.get_gesture(*id);
            let auto_mode = gesture.map(|g| !g.threshold_manual_override).unwrap_or(false);
            let recent_hit = recent_hit_name.as_ref().map(|n| n == name).unwrap_or(false);

            GestureMonitorDto {
                id: *id,
                name: name.clone(),
                example_count: app.recognizer.example_count(*id),
                distance: *dist,
                threshold: *thresh,
                auto_mode,
                recent_hit,
            }
        }).collect(),
        recent_hit: recent_hit_name,
    };

    let hit_entries: Vec<HitEntryDto> = app.hit_log.recent(10).iter().map(|e| {
        let ms_ago = e.timestamp.elapsed().as_millis() as u64;
        HitEntryDto {
            name: e.gesture_name.clone(),
            ms_ago,
            recent: ms_ago < 2000,
        }
    }).collect();

    let hit_log = HitLogDto {
        total: app.hit_log.len(),
        entries: hit_entries,
    };

    Ok(StateResponse {
        vocabulary,
        file_path: app.file_path.as_ref().map(|p| p.display().to_string()),
        dirty: app.dirty,
        mode: app.mode,
        selected_gesture_id: app.selected_gesture_id,
        osc_status,
        training,
        monitor,
        hit_log,
    })
}

#[tauri::command]
pub fn set_mode(state: State<Arc<Mutex<AppState>>>, mode: AppMode) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    if app.mode != mode {
        app.mode = mode;
        if mode == AppMode::Performance {
            app.recognizer.start();
        } else {
            app.recognizer.stop();
        }
    }

    Ok(())
}

#[tauri::command]
pub fn new_vocabulary(state: State<Arc<Mutex<AppState>>>) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    app.vocabulary = Vocabulary::new("New Vocabulary");
    app.file_path = None;
    app.dirty = false;
    app.selected_gesture_id = None;
    app.sync_recognizer();

    Ok(())
}

#[tauri::command]
pub fn open_vocabulary(state: State<Arc<Mutex<AppState>>>) -> Result<(), String> {
    let default_dir = default_vocabulary_dir().ok();

    let mut dialog = rfd::FileDialog::new()
        .add_filter("RALF Vocabulary", &["ralf"])
        .set_title("Open Vocabulary");

    if let Some(dir) = default_dir {
        dialog = dialog.set_directory(dir);
    }

    if let Some(path) = dialog.pick_file() {
        let mut app = state.lock().map_err(|e| e.to_string())?;

        match model_load_vocabulary(&path) {
            Ok(vocab) => {
                app.vocabulary = vocab;
                app.file_path = Some(path);
                app.dirty = false;
                app.selected_gesture_id = app.vocabulary.gestures.first().map(|g| g.id);
                app.sync_recognizer();

                app.osc_sender = OscSender::new(
                    &app.vocabulary.output.host,
                    app.vocabulary.output.port,
                );
            }
            Err(e) => {
                return Err(format!("Failed to open vocabulary: {}", e));
            }
        }
    }

    Ok(())
}

#[tauri::command]
pub fn save_vocabulary(state: State<Arc<Mutex<AppState>>>) -> Result<(), String> {
    let default_dir = default_vocabulary_dir().ok();

    let vocab_name = {
        let app = state.lock().map_err(|e| e.to_string())?;
        app.vocabulary.name.clone()
    };

    let mut dialog = rfd::FileDialog::new()
        .add_filter("RALF Vocabulary", &["ralf"])
        .set_title("Save Vocabulary")
        .set_file_name(&format!("{}.ralf", vocab_name));

    if let Some(dir) = default_dir {
        dialog = dialog.set_directory(dir);
    }

    if let Some(path) = dialog.save_file() {
        let mut app = state.lock().map_err(|e| e.to_string())?;

        match model_save_vocabulary(&app.vocabulary, &path) {
            Ok(()) => {
                app.file_path = Some(path);
                app.dirty = false;
            }
            Err(e) => {
                return Err(format!("Failed to save vocabulary: {}", e));
            }
        }
    }

    Ok(())
}

#[tauri::command]
pub fn send_test_hit(state: State<Arc<Mutex<AppState>>>) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.osc_sender.send_hit("/test/hit").map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn add_gesture(state: State<Arc<Mutex<AppState>>>) -> Result<u32, String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    let name = format!("gesture{}", app.vocabulary.gestures.len() + 1);
    let new_id = app.vocabulary.add_gesture(&name);

    // Get gesture data before mutable borrow of recognizer
    let gesture_data = app.vocabulary.get_gesture(new_id).map(|g| {
        (g.id, g.name.clone(), g.osc_address.clone(), g.threshold)
    });

    if let Some((id, name, osc_address, threshold)) = gesture_data {
        app.recognizer.add_gesture(id, &name, &osc_address, threshold);
    }

    app.selected_gesture_id = Some(new_id);
    app.mark_dirty();

    Ok(new_id)
}

#[tauri::command]
pub fn select_gesture(state: State<Arc<Mutex<AppState>>>, gesture_id: u32) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.selected_gesture_id = Some(gesture_id);
    Ok(())
}

#[tauri::command]
pub fn rename_gesture(state: State<Arc<Mutex<AppState>>>, gesture_id: u32, name: String) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    if let Some(gesture) = app.vocabulary.get_gesture_mut(gesture_id) {
        gesture.name = name;
        app.mark_dirty();
    }

    Ok(())
}

#[tauri::command]
pub fn delete_gesture(state: State<Arc<Mutex<AppState>>>, gesture_id: u32) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    app.vocabulary.remove_gesture(gesture_id);

    if app.selected_gesture_id == Some(gesture_id) {
        app.selected_gesture_id = app.vocabulary.gestures.first().map(|g| g.id);
    }

    app.sync_recognizer();
    app.mark_dirty();

    Ok(())
}

#[tauri::command]
pub fn start_training(
    state: State<Arc<Mutex<AppState>>>,
    gesture_id: u32,
    reps: u32,
    countdown_secs: u32,
    duration_secs: u32,
    rest_secs: u32,
) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    // Get gesture name before mutable operations
    let gesture_name = app.vocabulary.get_gesture(gesture_id).map(|g| g.name.clone());

    if let Some(name) = gesture_name {
        let config = TrainingConfig {
            reps,
            countdown_secs: countdown_secs as f32,
            duration_secs: duration_secs as f32,
            rest_secs: rest_secs as f32,
        };

        app.training_config = config.clone();
        app.training_session.start(gesture_id, &name, config);
    }

    Ok(())
}

#[tauri::command]
pub fn cancel_training(state: State<Arc<Mutex<AppState>>>) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.training_session.cancel();
    Ok(())
}

#[tauri::command]
pub fn set_threshold(
    state: State<Arc<Mutex<AppState>>>,
    gesture_id: u32,
    threshold: f32,
    manual: bool,
) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    app.recognizer.set_threshold(gesture_id, threshold);

    if let Some(gesture) = app.vocabulary.get_gesture_mut(gesture_id) {
        gesture.threshold = threshold;
        gesture.threshold_manual_override = manual;
    }

    app.mark_dirty();

    Ok(())
}

#[tauri::command]
pub fn toggle_threshold_mode(state: State<Arc<Mutex<AppState>>>, gesture_id: u32) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    // Calculate new threshold if switching to auto mode
    let new_threshold = if let Some(gesture) = app.vocabulary.get_gesture(gesture_id) {
        if gesture.threshold_manual_override {
            // Switching to auto mode - calculate threshold
            if let (Some(mean), Some(std)) = (gesture.distance_mean, gesture.distance_std) {
                Some(mean + std * gesture.threshold_coefficient)
            } else {
                None
            }
        } else {
            None // Just switching to manual, no threshold change
        }
    } else {
        None
    };

    // Now do the mutable operations
    if let Some(gesture) = app.vocabulary.get_gesture_mut(gesture_id) {
        if gesture.threshold_manual_override {
            // Switch to auto mode
            if let Some(thresh) = new_threshold {
                gesture.threshold = thresh;
                gesture.threshold_manual_override = false;
            }
        } else {
            // Switch to manual mode - keep current threshold
            gesture.threshold_manual_override = true;
        }
    }

    // Update recognizer with new threshold
    if let Some(thresh) = new_threshold {
        app.recognizer.set_threshold(gesture_id, thresh);
    }

    app.mark_dirty();

    Ok(())
}

#[tauri::command]
pub fn set_cooldown(state: State<Arc<Mutex<AppState>>>, ms: u64) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.recognition_config.cooldown_ms = ms;
    app.recognizer.set_cooldown_ms(ms);
    Ok(())
}
