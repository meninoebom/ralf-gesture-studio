use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::State;

use crate::engine::{
    assess_example, compute_joint_weights, compute_threshold_stats, generate_augmented,
    DiagnosticEvent, DiagnosticLogger, GestureDiag, HitLog, Preprocessor, RecognitionConfig,
    Recognizer, SessionState, TrainingConfig, TrainingSession,
};
use crate::model::{
    default_vocabulary_dir, load_vocabulary as model_load_vocabulary,
    save_vocabulary as model_save_vocabulary,
};
use crate::model::{Example, Vocabulary};
use crate::osc::{ConnectionStatus, OscReceiverHandle, OscSender, SenderStatus};

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
    /// Diagnostic logger for recognition analysis
    diagnostic_logger: DiagnosticLogger,
    /// Tracks frame dimension mismatch for UI feedback (None = OK)
    dimension_mismatch: Option<(usize, usize)>,
    /// Frame preprocessor (hip centering, scale normalization, velocity features)
    preprocessor: Preprocessor,
    /// Quality feedback from most recent training session (cleared on next training start)
    quality_feedback: Vec<QualityFeedbackEntry>,
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

        let preprocessor = Preprocessor::new(
            vocabulary.preprocessing.clone(),
            &vocabulary.tracking_system,
        );

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
            training_session: TrainingSession::new(),
            training_config: TrainingConfig::default(),
            recognition_config,
            diagnostic_logger: DiagnosticLogger::new(),
            dimension_mismatch: None,
            preprocessor,
            quality_feedback: Vec::new(),
        }
    }

    /// Sync recognizer with vocabulary.
    ///
    /// Rebuilds the recognizer from scratch: preprocess → weight → augment → add.
    /// This is the canonical rebuild path — called after loading, training, or toggling features.
    fn sync_recognizer(&mut self) {
        use crate::engine::weighting::apply_weights_to_sequence;

        let was_active = self.recognizer.is_active();

        self.recognizer = Recognizer::with_config(600, 180, self.recognition_config.clone());
        self.preprocessor.reset();

        for gesture in &self.vocabulary.gestures {
            self.recognizer.add_gesture(
                gesture.id,
                &gesture.name,
                &gesture.osc_address,
                gesture.threshold,
            );

            // Set consensus config from vocabulary
            self.recognizer.set_consensus(
                gesture.id,
                gesture.consensus_enabled,
                gesture.consensus_threshold,
            );

            // Preprocess all examples
            let processed: Vec<Vec<Vec<f32>>> = gesture
                .examples
                .iter()
                .map(|e| self.preprocessor.process_sequence(&e.frames))
                .collect();

            // Compute joint weights if enabled and enough examples
            let weights = if self.vocabulary.joint_weighting && processed.len() >= 2 {
                compute_joint_weights(&processed)
            } else {
                None
            };

            // Store weights in recognizer for runtime window scaling
            self.recognizer.set_weights(gesture.id, weights.clone());

            for (ex_idx, example) in processed.iter().enumerate() {
                // Apply joint weights (if any)
                let scaled = match &weights {
                    Some(w) => apply_weights_to_sequence(example, w),
                    None => example.clone(),
                };

                self.recognizer.add_example(gesture.id, scaled.clone());

                // Add ephemeral augmented copies (of the scaled data)
                for aug in generate_augmented(
                    &scaled,
                    &self.vocabulary.augmentation,
                    gesture.id,
                    ex_idx,
                ) {
                    self.recognizer.add_example(gesture.id, aug);
                }
            }
        }

        if was_active {
            self.recognizer.start();
        }
    }

    /// Save completed training examples.
    ///
    /// Saves raw examples to vocabulary, runs quality checks, computes statistics,
    /// then rebuilds the recognizer via sync_recognizer() (which handles preprocessing,
    /// joint weighting, and augmentation).
    fn save_training_examples(&mut self) {
        let gesture_id = self.training_session.gesture_id;
        let new_frame_sets = self.training_session.take_examples();

        let gesture_name = self
            .vocabulary
            .get_gesture(gesture_id)
            .map(|g| g.name.clone())
            .unwrap_or_default();

        // Snapshot existing processed examples for quality assessment (before adding new ones)
        let existing_processed: Vec<Vec<Vec<f32>>> = self
            .vocabulary
            .get_gesture(gesture_id)
            .map(|g| {
                g.examples
                    .iter()
                    .map(|e| self.preprocessor.process_sequence(&e.frames))
                    .collect()
            })
            .unwrap_or_default();

        let existing_count = existing_processed.len();

        // Store raw examples in vocabulary and run quality checks
        for (new_idx, frames) in new_frame_sets.iter().enumerate() {
            if frames.is_empty() {
                continue;
            }

            // Store RAW frames in the vocabulary (preprocessing applied during sync)
            let example = Example::new(frames.clone(), frames.len() as u64);

            if let Some(gesture) = self.vocabulary.get_gesture_mut(gesture_id) {
                gesture.add_example(example);
            }

            // Quality assessment: check new example against pre-training examples
            if !existing_processed.is_empty() {
                let processed = self.preprocessor.process_sequence(frames);
                let example_index = existing_count + new_idx;
                if let Some(issue) = assess_example(&processed, &existing_processed) {
                    self.quality_feedback.push(QualityFeedbackEntry {
                        gesture_name: gesture_name.clone(),
                        example_index: example_index + 1, // 1-indexed for display
                        label: issue.label().to_string(),
                        message: issue.message(),
                    });
                }
            }
        }

        // Compute statistical threshold (uses real unweighted examples)
        self.compute_gesture_statistics(gesture_id);

        // Rebuild recognizer with all examples (preprocess → weight → augment)
        self.sync_recognizer();

        self.dirty = true;
        self.auto_save();
    }

    /// Compute threshold statistics for a gesture
    fn compute_gesture_statistics(&mut self, gesture_id: u32) {
        // Preprocess raw stored examples before computing pairwise DTW distances
        let examples: Vec<Vec<Vec<f32>>> = self
            .vocabulary
            .get_gesture(gesture_id)
            .map(|g| {
                g.examples
                    .iter()
                    .map(|e| self.preprocessor.process_sequence(&e.frames))
                    .collect()
            })
            .unwrap_or_default();

        if examples.len() < 2 {
            return;
        }

        let coefficient = self
            .vocabulary
            .get_gesture(gesture_id)
            .map(|g| g.threshold_coefficient)
            .unwrap_or(2.0);

        if let Some(stats) = compute_threshold_stats(&examples, coefficient) {
            if let Some(gesture) = self.vocabulary.get_gesture_mut(gesture_id) {
                gesture.update_statistics(stats.mean, stats.std);
                self.recognizer.set_threshold(gesture_id, gesture.threshold);

                // Log training completion
                if self.diagnostic_logger.is_enabled() {
                    self.diagnostic_logger
                        .log(DiagnosticEvent::TrainingComplete {
                            gesture_name: gesture.name.clone(),
                            example_count: gesture.examples.len(),
                            mean: gesture.distance_mean,
                            std: gesture.distance_std,
                            threshold: gesture.threshold,
                        });
                }
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

        let cooldown_ms = self.recognition_config.cooldown_ms;
        let expected_dims = self.vocabulary.input.dimensions;

        for frame in frames {
            // Validate frame dimensions against vocabulary config
            if frame.len() != expected_dims {
                if self.dimension_mismatch.map(|(_, a)| a) != Some(frame.len()) {
                    eprintln!(
                        "Frame dimension mismatch: expected {} dimensions, got {}. Frame dropped.",
                        expected_dims,
                        frame.len()
                    );
                }
                self.dimension_mismatch = Some((expected_dims, frame.len()));
                continue;
            }

            // Clear mismatch on successful frame
            if self.dimension_mismatch.is_some() {
                eprintln!(
                    "Frame dimensions now match expected {}. Resuming normal operation.",
                    expected_dims
                );
                self.dimension_mismatch = None;
            }

            // If training, add RAW frames to session (stored unprocessed in .ralf files)
            if self.training_session.state == SessionState::Capturing {
                self.training_session.add_frame(frame.clone());
            }

            // Preprocess frame before recognition
            let processed = self.preprocessor.process_frame(&frame);

            // Feed preprocessed frame to recognizer
            if let Some(result) = self.recognizer.process_frame(processed) {
                let frame_num = self.osc_receiver.state.frame_count as usize;

                // Log state transitions if diagnostics enabled
                if self.diagnostic_logger.is_enabled() && self.mode == AppMode::Performance {
                    for transition in self.recognizer.take_transitions() {
                        let margin_pct = ((transition.threshold - transition.distance)
                            / transition.threshold)
                            * 100.0;
                        self.diagnostic_logger.log(DiagnosticEvent::StateChange {
                            gesture_name: transition.gesture_name,
                            from_state: format!("{}", transition.transition.from_state),
                            to_state: format!("{}", transition.transition.to_state),
                            distance: transition.distance,
                            threshold: transition.threshold,
                            margin_pct,
                            frames_in_state: transition.transition.frames_in_prev_state,
                            reason: transition.transition.reason,
                        });
                    }
                }

                // Log diagnostic data if enabled
                if self.diagnostic_logger.is_enabled() && self.mode == AppMode::Performance {
                    self.log_recognition_diagnostics(frame_num, cooldown_ms);
                }

                if let (Some(id), Some(name)) = (result.gesture_id, result.gesture_name.clone()) {
                    let osc_address = self
                        .recognizer
                        .get_gesture(id)
                        .map(|g| g.osc_address.clone())
                        .unwrap_or_else(|| format!("/gesture/{}", id));

                    // Log hit
                    if self.diagnostic_logger.is_enabled() {
                        let threshold = self
                            .recognizer
                            .get_gesture(id)
                            .map(|g| g.threshold)
                            .unwrap_or(0.0);
                        let margin_pct = ((threshold - result.distance) / threshold) * 100.0;
                        self.diagnostic_logger.log(DiagnosticEvent::Hit {
                            frame_num,
                            gesture_name: name.clone(),
                            distance: result.distance,
                            threshold,
                            margin_pct,
                        });
                    }

                    self.hit_log
                        .record(id, &name, result.distance, &osc_address);
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

    /// Log recognition diagnostics for all gestures
    fn log_recognition_diagnostics(&mut self, frame_num: usize, cooldown_ms: u64) {
        use std::time::Duration;

        let buffer_len = self.recognizer.buffer.len();
        let window_size = self.recognizer.window_size();
        let near_miss_pct = self.diagnostic_logger.near_miss_pct();
        let cooldown = Duration::from_millis(cooldown_ms);

        // Collect gesture diagnostics
        let mut gestures: Vec<GestureDiag> = Vec::new();
        let mut near_misses: Vec<(String, f32, f32, &'static str)> = Vec::new();

        for (id, name, distance, threshold) in self.recognizer.current_distances() {
            let gesture = self.recognizer.get_gesture(id);
            let in_cooldown = gesture.map(|g| g.in_cooldown(cooldown)).unwrap_or(false);
            let example_count = gesture.map(|g| g.example_count()).unwrap_or(0);

            gestures.push(GestureDiag {
                name: name.clone(),
                distance,
                threshold,
                armed: true, // Simple mode: always armed (cooldown handles repetition)
                in_cooldown,
                example_count,
            });

            // Check for near-misses
            if let Some(dist) = distance {
                // Near miss: within threshold + margin%, but didn't fire
                if dist < threshold * (1.0 + near_miss_pct / 100.0) && dist >= threshold {
                    near_misses.push((name.clone(), dist, threshold, "above_threshold"));
                } else if dist < threshold && in_cooldown {
                    near_misses.push((name.clone(), dist, threshold, "in_cooldown"));
                }
            }
        }

        // Log recognition cycle
        self.diagnostic_logger.log(DiagnosticEvent::Recognition {
            frame_num,
            buffer_len,
            window_size,
            gestures,
        });

        // Log near misses
        for (name, dist, threshold, reason) in near_misses {
            let margin_pct = ((threshold - dist) / threshold) * 100.0;
            self.diagnostic_logger.log(DiagnosticEvent::NearMiss {
                frame_num,
                gesture_name: name,
                distance: dist,
                threshold,
                margin_pct,
                reason,
            });
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
    augmentation: AugmentationConfigDto,
    joint_weighting: bool,
}

#[derive(Serialize)]
pub struct AugmentationConfigDto {
    enabled: bool,
    multiplier: u32,
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
    dimension_mismatch_expected: Option<usize>,
    dimension_mismatch_actual: Option<usize>,
}

#[derive(Serialize)]
pub struct TrainingDto {
    state: String,
    gesture_id: Option<u32>,
    countdown: u32,
    current_rep: u32,
    total_reps: u32,
    completed_reps: u32,
    remaining: f32,
    progress: f32,
    frame_count: usize,
    quality_issues: Vec<QualityIssueDto>,
}

#[derive(Debug, Clone, Serialize)]
struct QualityFeedbackEntry {
    gesture_name: String,
    example_index: usize,
    label: String,
    message: String,
}

#[derive(Serialize)]
struct QualityIssueDto {
    example_index: usize,
    label: String,
    message: String,
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
    // Statistics for auto-calibration display
    distance_mean: Option<f32>,
    distance_std: Option<f32>,
    threshold_coefficient: f32,
    consensus_enabled: bool,
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
        gestures: app
            .vocabulary
            .gestures
            .iter()
            .map(|g| GestureDto {
                id: g.id,
                name: g.name.clone(),
                osc_address: g.osc_address.clone(),
                threshold: g.threshold,
                examples: g
                    .examples
                    .iter()
                    .map(|e| ExampleDto {
                        frame_count: e.frames.len(),
                    })
                    .collect(),
            })
            .collect(),
        augmentation: AugmentationConfigDto {
            enabled: app.vocabulary.augmentation.enabled,
            multiplier: app.vocabulary.augmentation.multiplier,
        },
        joint_weighting: app.vocabulary.joint_weighting,
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
        ms_since_last_frame: app.osc_receiver.ms_since_last_frame().unwrap_or(0),
        input_error: app.osc_receiver.state.error_message.clone(),
        frame_count: app.osc_receiver.state.frame_count,
        output_status,
        ms_since_last_send: app.osc_sender.ms_since_last_send().unwrap_or(0),
        output_error: app.osc_sender.state.error_message.clone(),
        send_count: app.osc_sender.state.send_count,
        dimension_mismatch_expected: app.dimension_mismatch.map(|(e, _)| e),
        dimension_mismatch_actual: app.dimension_mismatch.map(|(_, a)| a),
    };

    let training_state = match app.training_session.state {
        SessionState::Idle => "idle",
        SessionState::Countdown => "countdown",
        SessionState::Capturing => "capturing",
        SessionState::Resting => "resting",
        SessionState::Complete => "complete",
    };

    let quality_issues: Vec<QualityIssueDto> = app
        .quality_feedback
        .iter()
        .map(|entry| QualityIssueDto {
            example_index: entry.example_index,
            label: entry.label.clone(),
            message: entry.message.clone(),
        })
        .collect();

    let training = TrainingDto {
        state: training_state.to_string(),
        gesture_id: if app.training_session.state != SessionState::Idle {
            Some(app.training_session.gesture_id)
        } else {
            None
        },
        countdown: app.training_session.countdown_value(),
        current_rep: app.training_session.completed_reps + 1,
        total_reps: app.training_session.config.reps,
        completed_reps: app.training_session.completed_reps,
        remaining: app.training_session.remaining_secs(),
        progress: app.training_session.progress(),
        frame_count: app.training_session.current_frame_count(),
        quality_issues,
    };

    let distances = app.recognizer.current_distances();
    let recent_hits = app.hit_log.recent(1);
    let recent_hit_name = recent_hits
        .first()
        .filter(|h| h.timestamp.elapsed().as_millis() < 300)
        .map(|h| h.gesture_name.clone());

    let monitor = MonitorDto {
        active: app.recognizer.is_active(),
        buffer_len: app.recognizer.buffer.len(),
        window_size: app.recognizer.window_size(),
        total_examples: app.recognizer.total_example_count(),
        gestures: distances
            .iter()
            .map(|(id, name, dist, thresh)| {
                let gesture = app.vocabulary.get_gesture(*id);
                let auto_mode = gesture
                    .map(|g| !g.threshold_manual_override)
                    .unwrap_or(false);
                let recent_hit = recent_hit_name.as_ref().map(|n| n == name).unwrap_or(false);
                let (distance_mean, distance_std, threshold_coefficient, consensus_enabled) =
                    gesture
                        .map(|g| {
                            (
                                g.distance_mean,
                                g.distance_std,
                                g.threshold_coefficient,
                                g.consensus_enabled,
                            )
                        })
                        .unwrap_or((None, None, 2.0, false));

                GestureMonitorDto {
                    id: *id,
                    name: name.clone(),
                    example_count: app.recognizer.example_count(*id),
                    distance: *dist,
                    threshold: *thresh,
                    auto_mode,
                    recent_hit,
                    distance_mean,
                    distance_std,
                    threshold_coefficient,
                    consensus_enabled,
                }
            })
            .collect(),
        recent_hit: recent_hit_name,
    };

    let hit_entries: Vec<HitEntryDto> = app
        .hit_log
        .recent(10)
        .iter()
        .map(|e| {
            let ms_ago = e.timestamp.elapsed().as_millis() as u64;
            HitEntryDto {
                name: e.gesture_name.clone(),
                ms_ago,
                recent: ms_ago < 2000,
            }
        })
        .collect();

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
    app.dimension_mismatch = None;
    app.preprocessor = Preprocessor::new(
        app.vocabulary.preprocessing.clone(),
        &app.vocabulary.tracking_system,
    );
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
                app.dimension_mismatch = None;
                app.preprocessor = Preprocessor::new(
                    app.vocabulary.preprocessing.clone(),
                    &app.vocabulary.tracking_system,
                );
                app.sync_recognizer();

                app.osc_sender =
                    OscSender::new(&app.vocabulary.output.host, app.vocabulary.output.port);
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
        .set_file_name(format!("{}.ralf", vocab_name));

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
    app.osc_sender
        .send_hit("/test/hit")
        .map_err(|e| e.to_string())?;
    Ok(())
}

#[tauri::command]
pub fn add_gesture(state: State<Arc<Mutex<AppState>>>) -> Result<u32, String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    let name = format!("gesture{}", app.vocabulary.gestures.len() + 1);
    let new_id = app.vocabulary.add_gesture(&name);

    // Get gesture data before mutable borrow of recognizer
    let gesture_data = app
        .vocabulary
        .get_gesture(new_id)
        .map(|g| (g.id, g.name.clone(), g.osc_address.clone(), g.threshold));

    if let Some((id, name, osc_address, threshold)) = gesture_data {
        app.recognizer
            .add_gesture(id, &name, &osc_address, threshold);
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
pub fn rename_gesture(
    state: State<Arc<Mutex<AppState>>>,
    gesture_id: u32,
    name: String,
) -> Result<(), String> {
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
    let gesture_name = app
        .vocabulary
        .get_gesture(gesture_id)
        .map(|g| g.name.clone());

    if let Some(name) = gesture_name {
        // Clear stale quality feedback from previous training session
        app.quality_feedback.clear();

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
pub fn toggle_threshold_mode(
    state: State<Arc<Mutex<AppState>>>,
    gesture_id: u32,
) -> Result<(), String> {
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

// Motion gate commands removed in Phase 1 simplification
// Best template commands removed — A/B test showed All Examples wins by 30%
// The simple Wekinator-style approach doesn't need motion gating

#[tauri::command]
pub fn enable_diagnostics(state: State<Arc<Mutex<AppState>>>) -> Result<String, String> {
    use chrono::Local;

    let mut app = state.lock().map_err(|e| e.to_string())?;

    // Generate path: ~/Documents/RALF/ralf-diagnostics-<timestamp>.log
    let ralf_dir = default_vocabulary_dir().map_err(|e| e.to_string())?;
    std::fs::create_dir_all(&ralf_dir).map_err(|e| format!("Failed to create directory: {}", e))?;

    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
    let filename = format!("ralf-diagnostics-{}.log", timestamp);
    let path = ralf_dir.join(filename);

    app.diagnostic_logger
        .enable(path.clone())
        .map_err(|e| format!("Failed to enable diagnostics: {}", e))?;

    Ok(path.display().to_string())
}

#[tauri::command]
pub fn disable_diagnostics(state: State<Arc<Mutex<AppState>>>) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.diagnostic_logger.disable();
    Ok(())
}

#[tauri::command]
pub fn is_diagnostics_enabled(state: State<Arc<Mutex<AppState>>>) -> Result<bool, String> {
    let app = state.lock().map_err(|e| e.to_string())?;
    Ok(app.diagnostic_logger.is_enabled())
}

#[tauri::command]
pub fn set_augmentation_enabled(
    state: State<Arc<Mutex<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.vocabulary.augmentation.enabled = enabled;
    app.sync_recognizer();
    app.mark_dirty();
    Ok(())
}

#[tauri::command]
pub fn set_joint_weighting(
    state: State<Arc<Mutex<AppState>>>,
    enabled: bool,
) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    app.vocabulary.joint_weighting = enabled;
    app.sync_recognizer();
    app.mark_dirty();
    Ok(())
}

#[tauri::command]
pub fn set_consensus(
    state: State<Arc<Mutex<AppState>>>,
    gesture_id: u32,
    enabled: bool,
) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;

    if let Some(gesture) = app.vocabulary.get_gesture_mut(gesture_id) {
        gesture.consensus_enabled = enabled;
    }
    app.recognizer
        .set_consensus(gesture_id, enabled, 0.5);
    app.mark_dirty();

    Ok(())
}
