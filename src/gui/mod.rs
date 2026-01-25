use std::path::PathBuf;

use eframe::egui;

use crate::model::{Vocabulary, Example, save_vocabulary, load_vocabulary, default_vocabulary_dir};
use crate::osc::{OscReceiverHandle, ConnectionStatus, OscSender, SenderStatus};
use crate::engine::{Recognizer, HitLog, TrainingSession, TrainingConfig, BaselineConfig, SessionState, RecognitionConfig};

// Custom colors - gold instead of yellow for better readability
const GOLD: egui::Color32 = egui::Color32::from_rgb(255, 185, 50);
const BRIGHT_GREEN: egui::Color32 = egui::Color32::from_rgb(100, 220, 100);
const BRIGHT_RED: egui::Color32 = egui::Color32::from_rgb(255, 100, 100);
const BRIGHT_BLUE: egui::Color32 = egui::Color32::from_rgb(100, 150, 255);
const BRIGHT_ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 165, 50);

/// The two modes of the application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    Training,
    Performance,
}

impl AppMode {
    fn as_str(&self) -> &'static str {
        match self {
            AppMode::Training => "Training",
            AppMode::Performance => "Performance",
        }
    }
}

/// Simple baseline recording state
#[derive(Debug, Clone)]
pub enum BaselineState {
    Idle,
    Countdown { start_time: std::time::Instant },
    Recording { frames: Vec<Vec<f32>>, start_time: std::time::Instant },
}

impl PartialEq for BaselineState {
    fn eq(&self, other: &Self) -> bool {
        matches!((self, other),
            (BaselineState::Idle, BaselineState::Idle) |
            (BaselineState::Countdown { .. }, BaselineState::Countdown { .. }) |
            (BaselineState::Recording { .. }, BaselineState::Recording { .. }))
    }
}

/// Main application state
pub struct GestureStudioApp {
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
    /// Training session (replaces simple recording)
    training_session: TrainingSession,
    /// Training configuration
    training_config: TrainingConfig,
    /// Last detected gesture name (for display)
    last_detected: Option<String>,
    /// Gesture being renamed (ID, current text)
    renaming_gesture: Option<(u32, String)>,
    /// Gesture ID to delete (confirmation pending)
    delete_gesture_id: Option<u32>,
    /// Vocabulary name being edited
    renaming_vocabulary: Option<String>,
    /// Baseline recording state
    baseline_state: BaselineState,
    /// Baseline recording configuration
    baseline_config: BaselineConfig,
    /// Recognition config (debounce + cooldown)
    recognition_config: RecognitionConfig,
}

impl GestureStudioApp {
    /// Create a new application with a demo vocabulary
    pub fn new(cc: &eframe::CreationContext<'_>, osc_receiver: OscReceiverHandle) -> Self {
        // Configure larger fonts for better readability
        let mut style = (*cc.egui_ctx.style()).clone();

        // Increase all font sizes
        for (_text_style, font_id) in style.text_styles.iter_mut() {
            font_id.size *= 1.3; // 30% larger
        }

        // Increase spacing
        style.spacing.item_spacing = egui::vec2(10.0, 8.0);
        style.spacing.button_padding = egui::vec2(12.0, 6.0);

        cc.egui_ctx.set_style(style);

        // Create a demo vocabulary to show the UI working
        let mut vocabulary = Vocabulary::new("Demo Vocabulary");
        vocabulary.add_gesture("wave");
        vocabulary.add_gesture("jump");
        vocabulary.add_gesture("spin");

        // Create OSC sender with default output settings
        let osc_sender = OscSender::new(&vocabulary.output.host, vocabulary.output.port);

        // Create recognizer - buffer 10 seconds at 60fps, match window of 3 seconds
        // Default recognition config: 80ms debounce, 500ms cooldown
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
            training_session: TrainingSession::new(),
            training_config: TrainingConfig::default(),
            last_detected: None,
            renaming_gesture: None,
            delete_gesture_id: None,
            renaming_vocabulary: None,
            baseline_state: BaselineState::Idle,
            baseline_config: BaselineConfig::default(),
            recognition_config,
        }
    }

    /// Sync recognizer with vocabulary
    fn sync_recognizer(&mut self) {
        // Remember if recognizer was active (for Performance mode)
        let was_active = self.recognizer.is_active();

        // Rebuild recognizer with current vocabulary
        self.recognizer = Recognizer::with_config(600, 180, self.recognition_config.clone());

        for gesture in &self.vocabulary.gestures {
            self.recognizer.add_gesture(
                gesture.id,
                &gesture.name,
                &gesture.osc_address,
                gesture.threshold,
            );

            // Add examples
            for example in &gesture.examples {
                self.recognizer.add_example(gesture.id, example.frames.clone());
            }
        }

        // Restore active state if we were in Performance mode
        if was_active {
            self.recognizer.start();
        }
    }

    /// Start a training session for the selected gesture
    fn start_training(&mut self) {
        if let Some(gesture_id) = self.selected_gesture_id {
            if let Some(gesture) = self.vocabulary.get_gesture(gesture_id) {
                self.training_session.start(
                    gesture_id,
                    &gesture.name,
                    self.training_config.clone(),
                );
            }
        }
    }

    /// Save completed training examples to vocabulary and recognizer
    fn save_training_examples(&mut self) {
        let gesture_id = self.training_session.gesture_id;
        let examples = self.training_session.take_examples();

        for frames in examples {
            if frames.is_empty() {
                continue;
            }

            let frame_count = frames.len();
            let example = Example::new(frames.clone(), frame_count as u64);

            // Add to vocabulary
            if let Some(gesture) = self.vocabulary.get_gesture_mut(gesture_id) {
                gesture.add_example(example);
            }

            // Add to recognizer
            self.recognizer.add_example(gesture_id, frames);
        }

        self.dirty = true;
        self.auto_save();
    }

    /// Auto-calibrate gesture thresholds based on baseline
    /// Sets each threshold to 80% of the distance between baseline and gesture examples
    fn auto_calibrate_thresholds(&mut self) {
        use crate::engine::dtw::dtw_distance_normalized;

        let baseline = match &self.vocabulary.baseline {
            Some(b) if !b.is_empty() => b.clone(),
            _ => return, // No baseline, can't calibrate
        };

        for gesture in &mut self.vocabulary.gestures {
            if gesture.examples.is_empty() {
                continue;
            }

            // Compute average distance from baseline to each gesture example
            let mut total_dist = 0.0;
            let mut count = 0;
            for example in &gesture.examples {
                let dist = dtw_distance_normalized(&baseline, &example.frames);
                if dist.is_finite() {
                    total_dist += dist;
                    count += 1;
                }
            }

            if count > 0 {
                let avg_dist = total_dist / count as f32;
                // Set threshold to 80% of average distance (gesture should be closer than baseline)
                gesture.threshold = (avg_dist * 0.8).max(100.0);
            }
        }

        // Sync to recognizer
        self.sync_recognizer();
        self.mark_dirty();
    }

    /// Create a new empty vocabulary
    fn new_vocabulary(&mut self) {
        self.vocabulary = Vocabulary::new("New Vocabulary");
        self.file_path = None;
        self.dirty = false;
        self.selected_gesture_id = None;
        self.sync_recognizer();
    }

    /// Open a vocabulary file
    fn open_vocabulary(&mut self) {
        let default_dir = default_vocabulary_dir().ok();

        let mut dialog = rfd::FileDialog::new()
            .add_filter("RALF Vocabulary", &["ralf"])
            .set_title("Open Vocabulary");

        if let Some(dir) = default_dir {
            dialog = dialog.set_directory(dir);
        }

        if let Some(path) = dialog.pick_file() {
            match load_vocabulary(&path) {
                Ok(vocab) => {
                    self.vocabulary = vocab;
                    self.file_path = Some(path);
                    self.dirty = false;
                    self.selected_gesture_id = self.vocabulary.gestures.first().map(|g| g.id);
                    self.sync_recognizer();

                    // Update OSC sender with loaded config
                    self.osc_sender = OscSender::new(
                        &self.vocabulary.output.host,
                        self.vocabulary.output.port,
                    );
                }
                Err(e) => {
                    eprintln!("Failed to open vocabulary: {}", e);
                }
            }
        }
    }

    /// Save vocabulary to current file or prompt for new file
    fn save_vocabulary_as(&mut self) {
        let default_dir = default_vocabulary_dir().ok();

        let mut dialog = rfd::FileDialog::new()
            .add_filter("RALF Vocabulary", &["ralf"])
            .set_title("Save Vocabulary")
            .set_file_name(&format!("{}.ralf", self.vocabulary.name));

        if let Some(dir) = default_dir {
            dialog = dialog.set_directory(dir);
        }

        if let Some(path) = dialog.save_file() {
            match save_vocabulary(&self.vocabulary, &path) {
                Ok(()) => {
                    self.file_path = Some(path);
                    self.dirty = false;
                }
                Err(e) => {
                    eprintln!("Failed to save vocabulary: {}", e);
                }
            }
        }
    }

    /// Auto-save if we have a file path
    fn auto_save(&mut self) {
        if let Some(ref path) = self.file_path {
            if let Err(e) = save_vocabulary(&self.vocabulary, path) {
                eprintln!("Auto-save failed: {}", e);
            } else {
                self.dirty = false;
            }
        }
    }

    /// Mark vocabulary as dirty (has unsaved changes)
    fn mark_dirty(&mut self) {
        self.dirty = true;
        self.auto_save();
    }
}

impl eframe::App for GestureStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle keyboard input
        ctx.input(|i| {
            // Spacebar to start training (when idle and can start)
            if i.key_pressed(egui::Key::Space) && !self.training_session.is_active() {
                if self.selected_gesture_id.is_some()
                    && self.osc_receiver.state.status == ConnectionStatus::Receiving
                    && self.mode == AppMode::Training
                    && self.renaming_gesture.is_none()
                {
                    self.start_training();
                }
            }

            // Escape to cancel training or editing
            if i.key_pressed(egui::Key::Escape) {
                if self.training_session.is_active() {
                    self.training_session.cancel();
                }
                self.renaming_gesture = None;
                self.renaming_vocabulary = None;
                self.delete_gesture_id = None;
            }
        });

        // Poll the OSC receiver for new frames
        let mut frames = self.osc_receiver.poll();

        // In Performance mode, only process the most recent frame to avoid backlog
        // (DTW is expensive, we can't keep up with 60fps of full processing)
        if self.mode == AppMode::Performance && frames.len() > 1 {
            // Keep only the last frame, discard older ones
            let last_frame = frames.pop().unwrap();
            frames.clear();
            frames.push(last_frame);
        }

        // Process frames
        for frame in frames {
            // Handle baseline countdown -> recording transition
            if let BaselineState::Countdown { start_time } = self.baseline_state {
                if start_time.elapsed().as_secs_f32() >= self.baseline_config.countdown_secs {
                    self.baseline_state = BaselineState::Recording {
                        frames: Vec::new(),
                        start_time: std::time::Instant::now(),
                    };
                }
            }

            // If recording baseline, add frames
            if let BaselineState::Recording { ref mut frames, start_time } = self.baseline_state {
                frames.push(frame.clone());
                // Auto-complete after baseline_duration seconds
                if start_time.elapsed().as_secs_f32() >= self.baseline_config.duration_secs {
                    let baseline_frames = std::mem::take(frames);
                    self.vocabulary.baseline = Some(baseline_frames);
                    self.baseline_state = BaselineState::Idle;
                    self.mark_dirty();
                }
            }

            // If training gesture, add frames to training session
            if self.training_session.state == SessionState::Capturing {
                self.training_session.add_frame(frame.clone());
            }

            // Always feed to recognizer for buffer/matching
            if let Some(result) = self.recognizer.process_frame(frame.clone()) {
                // Hit detected!
                if let (Some(id), Some(name)) = (result.gesture_id, result.gesture_name.clone()) {
                    self.last_detected = Some(name.clone());

                    // Get OSC address
                    let osc_address = self.recognizer
                        .get_gesture(id)
                        .map(|g| g.osc_address.clone())
                        .unwrap_or_else(|| format!("/gesture/{}", id));

                    // Log the hit
                    self.hit_log.record(id, &name, result.distance, &osc_address);

                    // Send OSC hit
                    let _ = self.osc_sender.send_hit(&osc_address);
                }
            }
        }

        // Update training session
        self.training_session.update();

        // Check if training just completed
        if self.training_session.state == SessionState::Complete {
            self.save_training_examples();
            // Reset to idle after saving
            self.training_session = TrainingSession::new();
        }

        // Request continuous repaints to keep the UI responsive
        ctx.request_repaint_after(std::time::Duration::from_millis(16)); // ~60fps for smooth updates

        // Top panel with title and mode selector
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("RALF Gesture Studio");

                // Show dirty indicator
                if self.dirty {
                    ui.colored_label(GOLD, "●");
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Mode selector dropdown (disabled during training)
                    let old_mode = self.mode;
                    ui.add_enabled_ui(!self.training_session.is_active(), |ui| {
                        egui::ComboBox::from_id_salt("mode_selector")
                            .selected_text(self.mode.as_str())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(&mut self.mode, AppMode::Training, "Training");
                                ui.selectable_value(&mut self.mode, AppMode::Performance, "Performance");
                            });
                    });

                    // Start/stop recognizer when switching modes
                    if old_mode != self.mode {
                        if self.mode == AppMode::Performance {
                            self.recognizer.start();
                            // Auto-calibrate thresholds based on baseline
                            self.auto_calibrate_thresholds();
                        } else {
                            self.recognizer.stop();
                        }
                    }
                });
            });
        });

        // Main content area
        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                // Vocabulary panel
                self.show_vocabulary_panel(ui);
                ui.add_space(8.0);

                // Connection panel
                self.show_connection_panel(ui);
                ui.add_space(8.0);

                // Gestures panel
                self.show_gestures_panel(ui);
                ui.add_space(8.0);

                // Baseline and Train panels (only in Training mode)
                if self.mode == AppMode::Training {
                    self.show_baseline_panel(ui);
                    ui.add_space(8.0);
                    self.show_train_panel(ui);
                }

                // Performance mode panels
                if self.mode == AppMode::Performance {
                    self.show_monitor_panel(ui);
                    ui.add_space(8.0);
                    self.show_hit_log_panel(ui);
                }
            });
        });
    }
}

impl GestureStudioApp {
    /// Render the vocabulary panel
    fn show_vocabulary_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("VOCABULARY");
            });
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Name:");

                // Check if we're renaming the vocabulary
                if let Some(ref mut new_name) = self.renaming_vocabulary {
                    let response = ui.add(
                        egui::TextEdit::singleline(new_name)
                            .desired_width(150.0)
                    );

                    // Auto-focus
                    response.request_focus();

                    // Commit on focus loss (Enter or click elsewhere)
                    if response.lost_focus() {
                        self.vocabulary.name = new_name.clone();
                        self.mark_dirty();
                        self.renaming_vocabulary = None;
                    }
                } else {
                    // Show name - clickable to edit
                    let name_response = ui.add(
                        egui::Label::new(egui::RichText::new(&self.vocabulary.name).strong())
                            .selectable(false)
                            .sense(egui::Sense::click())
                    );

                    // Show underline on hover to indicate editable
                    if name_response.hovered() {
                        let rect = name_response.rect;
                        let underline_y = rect.bottom() - 1.0;
                        ui.painter().line_segment(
                            [egui::pos2(rect.left(), underline_y), egui::pos2(rect.right(), underline_y)],
                            egui::Stroke::new(1.0, egui::Color32::GRAY),
                        );
                    }

                    if name_response.clicked() {
                        self.renaming_vocabulary = Some(self.vocabulary.name.clone());
                    }
                }

                // Show file path if saved
                if let Some(ref path) = self.file_path {
                    ui.colored_label(egui::Color32::GRAY, format!("({})", path.display()));
                }

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new("Save As").min_size(egui::vec2(80.0, 28.0))).clicked() {
                        self.save_vocabulary_as();
                    }
                    if ui.add(egui::Button::new("Open").min_size(egui::vec2(70.0, 28.0))).clicked() {
                        self.open_vocabulary();
                    }
                    if ui.add(egui::Button::new("New").min_size(egui::vec2(60.0, 28.0))).clicked() {
                        self.new_vocabulary();
                    }
                });
            });

            ui.horizontal(|ui| {
                ui.label(format!("Gestures: {}", self.vocabulary.gestures.len()));
            });
        });
    }

    /// Render the connection panel
    fn show_connection_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("CONNECTION");
            });
            ui.separator();

            ui.horizontal(|ui| {
                // Input section
                ui.vertical(|ui| {
                    ui.label("INPUT");
                    ui.horizontal(|ui| {
                        ui.label("Port:");
                        ui.label(format!("{}", self.vocabulary.input.port));
                        ui.label("Address:");
                        ui.label(&self.vocabulary.input.address);
                    });

                    // Show live connection status
                    ui.horizontal(|ui| {
                        let (color, status_text, detail) = match self.osc_receiver.state.status {
                            ConnectionStatus::Stopped => {
                                (egui::Color32::GRAY, "STOPPED", String::new())
                            }
                            ConnectionStatus::Listening => {
                                (GOLD, "LISTENING", "(waiting for data)".to_string())
                            }
                            ConnectionStatus::Receiving => {
                                let ms = self.osc_receiver.ms_since_last_frame().unwrap_or(0);
                                let secs = ms as f32 / 1000.0;
                                (BRIGHT_GREEN, "RECEIVING", format!("({:.1}s ago)", secs))
                            }
                            ConnectionStatus::Error => {
                                let msg = self.osc_receiver.state.error_message
                                    .as_deref()
                                    .unwrap_or("Unknown error");
                                (BRIGHT_RED, "ERROR", format!("({})", msg))
                            }
                        };
                        ui.colored_label(color, "●");
                        ui.colored_label(color, status_text);
                        if !detail.is_empty() {
                            ui.label(detail);
                        }
                    });

                    // Show frame count
                    ui.horizontal(|ui| {
                        ui.label(format!("Frames: {}", self.osc_receiver.state.frame_count));
                    });
                });

                ui.add_space(40.0);

                // Output section
                ui.vertical(|ui| {
                    ui.label("OUTPUT");
                    ui.horizontal(|ui| {
                        ui.label("Host:");
                        ui.label(&self.vocabulary.output.host);
                        ui.label("Port:");
                        ui.label(format!("{}", self.vocabulary.output.port));
                    });

                    // Show sender status
                    ui.horizontal(|ui| {
                        let (color, status_text, detail) = match self.osc_sender.state.status {
                            SenderStatus::Ready => {
                                (BRIGHT_GREEN, "READY", String::new())
                            }
                            SenderStatus::Sent => {
                                let ms = self.osc_sender.ms_since_last_send().unwrap_or(0);
                                let secs = ms as f32 / 1000.0;
                                (BRIGHT_GREEN, "SENT", format!("({:.1}s ago)", secs))
                            }
                            SenderStatus::Error => {
                                let msg = self.osc_sender.state.error_message
                                    .as_deref()
                                    .unwrap_or("Unknown error");
                                (BRIGHT_RED, "ERROR", format!("({})", msg))
                            }
                        };
                        ui.colored_label(color, "●");
                        ui.colored_label(color, status_text);
                        if !detail.is_empty() {
                            ui.label(detail);
                        }
                    });

                    // Show send count
                    ui.horizontal(|ui| {
                        ui.label(format!("Sent: {}", self.osc_sender.state.send_count));
                    });

                    // Send Test Hit button
                    ui.add_space(4.0);
                    if ui.add(egui::Button::new("Send Test Hit").min_size(egui::vec2(120.0, 28.0))).clicked() {
                        let _ = self.osc_sender.send_hit("/test/hit");
                    }
                });
            });
        });
    }

    /// Render the gestures panel
    fn show_gestures_panel(&mut self, ui: &mut egui::Ui) {
        // Disable during active training
        let enabled = !self.training_session.is_active();

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("GESTURES");
            });
            ui.separator();

            // Header row
            ui.horizontal(|ui| {
                ui.label("     "); // Status indicator space
                ui.label("Name");
                ui.add_space(80.0);
                ui.label("Examples");
                ui.add_space(20.0);
                ui.label("Output Address");
            });
            ui.separator();

            // Gesture rows - collect data first to avoid borrow issues
            let gesture_data: Vec<_> = self.vocabulary.gestures.iter()
                .map(|g| (g.id, g.name.clone(), g.example_count(), g.osc_address.clone(), g.has_examples()))
                .collect();

            let mut gesture_to_select = None;
            let mut gesture_to_delete = None;
            let mut start_rename = None;

            for (id, name, example_count, osc_address, has_examples) in gesture_data {
                ui.horizontal(|ui| {
                    // Status indicator (filled if has examples, empty if not)
                    if has_examples {
                        ui.colored_label(BRIGHT_GREEN, "●");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "○");
                    }

                    // Check if this gesture is being renamed
                    let is_renaming = self.renaming_gesture.as_ref().map(|(rid, _)| *rid == id).unwrap_or(false);
                    let is_selected = self.selected_gesture_id == Some(id);

                    if is_renaming {
                        // Show text edit for renaming
                        if let Some((_, ref mut new_name)) = self.renaming_gesture {
                            let response = ui.add(
                                egui::TextEdit::singleline(new_name)
                                    .desired_width(100.0)
                            );

                            // Auto-focus
                            response.request_focus();

                            // Commit on focus loss (Enter or click elsewhere)
                            // Escape is handled separately and clears renaming_gesture
                            if response.lost_focus() {
                                if let Some(gesture) = self.vocabulary.get_gesture_mut(id) {
                                    gesture.name = new_name.clone();
                                    self.mark_dirty();
                                }
                                self.renaming_gesture = None;
                            }
                        }
                    } else {
                        // Show name - clickable to rename (with visual hint)
                        let name_text = egui::RichText::new(&name);
                        let name_text = if is_selected {
                            name_text.color(BRIGHT_BLUE)
                        } else {
                            name_text
                        };

                        let name_response = ui.add(
                            egui::Label::new(name_text)
                                .selectable(false)
                                .sense(egui::Sense::click())
                        );

                        // Show underline on hover to indicate editable
                        if name_response.hovered() && enabled {
                            let rect = name_response.rect;
                            let underline_y = rect.bottom() - 1.0;
                            ui.painter().line_segment(
                                [egui::pos2(rect.left(), underline_y), egui::pos2(rect.right(), underline_y)],
                                egui::Stroke::new(1.0, egui::Color32::GRAY),
                            );
                        }

                        if name_response.clicked() && enabled {
                            start_rename = Some((id, name.clone()));
                        }
                    }

                    ui.add_space((80.0 - name.len() as f32 * 7.0).max(10.0));
                    ui.label(format!("{}", example_count));
                    ui.add_space(45.0);
                    ui.label(&osc_address);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.add_enabled_ui(enabled, |ui| {
                            // Delete button
                            if ui.add(egui::Button::new("×").min_size(egui::vec2(32.0, 28.0))).clicked() {
                                gesture_to_delete = Some(id);
                            }

                            // Select button
                            let select_text = if is_selected { "Selected" } else { "Select" };
                            if ui.add(egui::Button::new(select_text).min_size(egui::vec2(70.0, 28.0))).clicked() {
                                gesture_to_select = Some(id);
                            }
                        });
                    });
                });
            }

            // Apply actions after iteration
            if let Some(id) = gesture_to_select {
                self.selected_gesture_id = Some(id);
            }

            if let Some((id, name)) = start_rename {
                self.renaming_gesture = Some((id, name));
            }

            if let Some(id) = gesture_to_delete {
                self.delete_gesture_id = Some(id);
            }

            // Delete confirmation dialog
            if let Some(delete_id) = self.delete_gesture_id {
                let gesture_name = self.vocabulary.get_gesture(delete_id)
                    .map(|g| g.name.clone())
                    .unwrap_or_default();

                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.colored_label(BRIGHT_RED, format!("Delete \"{}\"?", gesture_name));
                    if ui.button("Yes").clicked() {
                        self.vocabulary.remove_gesture(delete_id);
                        if self.selected_gesture_id == Some(delete_id) {
                            self.selected_gesture_id = self.vocabulary.gestures.first().map(|g| g.id);
                        }
                        self.sync_recognizer();
                        self.mark_dirty();
                        self.delete_gesture_id = None;
                    }
                    if ui.button("No").clicked() {
                        self.delete_gesture_id = None;
                    }
                });
            }

            ui.add_space(12.0);
            ui.add_enabled_ui(enabled, |ui| {
                if ui.add(egui::Button::new("+ Add Gesture").min_size(egui::vec2(140.0, 32.0))).clicked() {
                    // Add a new gesture
                    let new_id = self.vocabulary.add_gesture(&format!("gesture{}", self.vocabulary.gestures.len() + 1));
                    if let Some(gesture) = self.vocabulary.get_gesture(new_id) {
                        self.recognizer.add_gesture(
                            gesture.id,
                            &gesture.name,
                            &gesture.osc_address,
                            gesture.threshold,
                        );
                    }
                    self.selected_gesture_id = Some(new_id);
                    self.mark_dirty();
                }
            });
        });
    }

    /// Render the baseline training panel
    fn show_baseline_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("1. BASELINE");
                ui.colored_label(egui::Color32::GRAY, "(your rest position)");
            });
            ui.separator();

            let has_baseline = self.vocabulary.baseline.is_some();
            let is_receiving = self.osc_receiver.state.status == ConnectionStatus::Receiving;

            match &self.baseline_state {
                BaselineState::Idle => {
                    ui.horizontal(|ui| {
                        if has_baseline {
                            ui.colored_label(BRIGHT_GREEN, "✓ Baseline recorded");
                            if let Some(ref baseline) = self.vocabulary.baseline {
                                ui.colored_label(egui::Color32::GRAY, format!("({} frames)", baseline.len()));
                            }
                        } else {
                            ui.colored_label(GOLD, "⚠ No baseline - record your rest position first");
                        }
                    });

                    ui.add_space(8.0);

                    ui.horizontal(|ui| {
                        let button_text = if has_baseline { "Re-record Baseline" } else { "Record Baseline" };
                        let can_record = is_receiving && !self.training_session.is_active();

                        if ui.add_enabled(can_record, egui::Button::new(button_text)).clicked() {
                            self.baseline_state = BaselineState::Countdown {
                                start_time: std::time::Instant::now(),
                            };
                        }

                        ui.add_space(15.0);
                        ui.label("Count-in:");
                        ui.add(egui::DragValue::new(&mut self.baseline_config.countdown_secs)
                            .range(1.0..=10.0)
                            .speed(0.1)
                            .suffix("s"));

                        ui.add_space(10.0);
                        ui.label("Duration:");
                        ui.add(egui::DragValue::new(&mut self.baseline_config.duration_secs)
                            .range(1.0..=10.0)
                            .speed(0.1)
                            .suffix("s"));

                        if !is_receiving {
                            ui.add_space(10.0);
                            ui.colored_label(egui::Color32::GRAY, "(waiting for OSC data)");
                        }
                    });
                }
                BaselineState::Countdown { start_time } => {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let remaining = (self.baseline_config.countdown_secs - elapsed).max(0.0);
                    let countdown_value = remaining.ceil() as u32;

                    ui.vertical_centered(|ui| {
                        ui.colored_label(
                            GOLD,
                            egui::RichText::new("GET READY").size(20.0).strong()
                        );
                        ui.add_space(8.0);
                        ui.colored_label(
                            BRIGHT_BLUE,
                            egui::RichText::new(format!("{}", countdown_value.max(1))).size(48.0).strong()
                        );
                        ui.add_space(4.0);
                        ui.colored_label(egui::Color32::GRAY, "Stand in your rest position...");
                    });
                }
                BaselineState::Recording { frames, start_time } => {
                    let elapsed = start_time.elapsed().as_secs_f32();
                    let remaining = (self.baseline_config.duration_secs - elapsed).max(0.0);
                    let progress = (elapsed / self.baseline_config.duration_secs).min(1.0);

                    ui.vertical_centered(|ui| {
                        ui.colored_label(
                            BRIGHT_BLUE,
                            egui::RichText::new("STAND STILL").size(24.0).strong()
                        );
                        ui.add_space(8.0);
                        ui.label(format!("{:.1}s remaining", remaining));
                        ui.add(egui::ProgressBar::new(progress).desired_width(200.0));
                        ui.add_space(4.0);
                        ui.colored_label(egui::Color32::GRAY, format!("{} frames captured", frames.len()));
                    });
                }
            }
        });
    }

    /// Render the training panel
    fn show_train_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("2. TRAIN GESTURES");
            });
            ui.separator();

            // Get selected gesture name
            let selected_name = self.selected_gesture_id
                .and_then(|id| self.vocabulary.get_gesture(id))
                .map(|g| g.name.clone())
                .unwrap_or_else(|| "(none)".to_string());

            // Training parameters (only editable when idle)
            let is_idle = self.training_session.state == SessionState::Idle;

            ui.horizontal(|ui| {
                ui.label("Gesture:");
                ui.colored_label(BRIGHT_BLUE, &selected_name);

                ui.add_space(20.0);

                ui.add_enabled_ui(is_idle, |ui| {
                    ui.label("Reps:");
                    let mut reps = self.training_config.reps as i32;
                    if ui.add(egui::DragValue::new(&mut reps).range(1..=20)).changed() {
                        self.training_config.reps = reps as u32;
                    }

                    ui.add_space(10.0);
                    ui.label("Count-in:");
                    ui.add(egui::DragValue::new(&mut self.training_config.countdown_secs)
                        .range(1.0..=10.0)
                        .speed(0.1)
                        .suffix("s"));

                    ui.add_space(10.0);
                    ui.label("Capture:");
                    ui.add(egui::DragValue::new(&mut self.training_config.duration_secs)
                        .range(0.5..=10.0)
                        .speed(0.1)
                        .suffix("s"));

                    ui.add_space(10.0);
                    ui.label("Rest:");
                    ui.add(egui::DragValue::new(&mut self.training_config.rest_secs)
                        .range(0.5..=10.0)
                        .speed(0.1)
                        .suffix("s"));
                });
            });

            ui.add_space(16.0);

            // Training state display
            match self.training_session.state {
                SessionState::Idle => {
                    self.show_train_idle(ui, &selected_name);
                }
                SessionState::Countdown => {
                    self.show_train_countdown(ui);
                }
                SessionState::Capturing => {
                    self.show_train_capturing(ui);
                }
                SessionState::Resting => {
                    self.show_train_resting(ui);
                }
                SessionState::Complete => {
                    // This should be brief as we immediately save and reset
                    ui.vertical_centered(|ui| {
                        ui.colored_label(BRIGHT_GREEN, egui::RichText::new("COMPLETE!").size(28.0).strong());
                    });
                }
            }
        });
    }

    /// Show idle training state
    fn show_train_idle(&mut self, ui: &mut egui::Ui, selected_name: &str) {
        ui.vertical_centered(|ui| {
            let can_start = self.selected_gesture_id.is_some()
                && self.osc_receiver.state.status == ConnectionStatus::Receiving;

            let button = egui::Button::new(
                egui::RichText::new("▶ START TRAINING")
                    .size(24.0)
                    .strong()
            )
            .min_size(egui::vec2(250.0, 60.0));

            let response = ui.add_enabled(can_start, button);
            if response.clicked() {
                self.start_training();
            }

            ui.add_space(8.0);

            if !can_start {
                if self.selected_gesture_id.is_none() {
                    ui.label("Select a gesture above to train");
                } else {
                    ui.label("Waiting for OSC data...");
                }
            } else {
                ui.label(format!("Press [Space] or click to train \"{}\"", selected_name));
                ui.label(format!(
                    "Will record {} reps × {:.1}s with {:.1}s rest",
                    self.training_config.reps,
                    self.training_config.duration_secs,
                    self.training_config.rest_secs
                ));
            }
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Status:");
            ui.colored_label(egui::Color32::GRAY, "IDLE");
        });
    }

    /// Show countdown state
    fn show_train_countdown(&self, ui: &mut egui::Ui) {
        let countdown = self.training_session.countdown_value();
        let rep_num = self.training_session.completed_reps + 1;
        let total_reps = self.training_session.config.reps;

        ui.vertical_centered(|ui| {
            // Large countdown number
            ui.colored_label(
                GOLD,
                egui::RichText::new(format!("{}", countdown))
                    .size(72.0)
                    .strong()
            );

            ui.add_space(8.0);
            ui.label(format!(
                "Get ready for rep {} of {} for \"{}\"",
                rep_num,
                total_reps,
                self.training_session.gesture_name
            ));

            ui.add_space(16.0);
            ui.colored_label(egui::Color32::GRAY, "Press [Esc] to cancel");
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Status:");
            ui.colored_label(GOLD, "COUNTDOWN");
        });
    }

    /// Show capturing state
    fn show_train_capturing(&self, ui: &mut egui::Ui) {
        let progress = self.training_session.progress();
        let remaining = self.training_session.remaining_secs();
        let frame_count = self.training_session.current_frame_count();
        let rep_num = self.training_session.completed_reps + 1;
        let total_reps = self.training_session.config.reps;

        ui.vertical_centered(|ui| {
            // CAPTURING header with pulsing effect
            ui.colored_label(
                BRIGHT_RED,
                egui::RichText::new("███ CAPTURING ███")
                    .size(28.0)
                    .strong()
            );

            ui.add_space(16.0);

            // Time remaining
            ui.colored_label(
                BRIGHT_RED,
                egui::RichText::new(format!("{:.1}s", remaining))
                    .size(48.0)
                    .strong()
            );

            ui.add_space(8.0);

            // Progress bar
            let progress_bar = egui::ProgressBar::new(progress)
                .animate(true);
            ui.add(progress_bar);

            ui.add_space(8.0);
            ui.label(format!("{} frames captured", frame_count));

            ui.add_space(8.0);
            ui.label(format!(
                "Recording example {} of {} for \"{}\"",
                rep_num,
                total_reps,
                self.training_session.gesture_name
            ));

            ui.add_space(8.0);
            ui.colored_label(egui::Color32::GRAY, "Press [Esc] to cancel");
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Status:");
            ui.colored_label(BRIGHT_RED, "CAPTURING");
        });
    }

    /// Show resting state
    fn show_train_resting(&self, ui: &mut egui::Ui) {
        let remaining = self.training_session.remaining_secs();
        let completed = self.training_session.completed_reps;
        let total_reps = self.training_session.config.reps;

        ui.vertical_centered(|ui| {
            ui.colored_label(
                BRIGHT_ORANGE,
                egui::RichText::new("REST")
                    .size(36.0)
                    .strong()
            );

            ui.add_space(16.0);

            // Time remaining
            ui.label(egui::RichText::new(format!("{:.1}s", remaining)).size(32.0));

            ui.add_space(16.0);

            // Progress summary
            ui.colored_label(
                BRIGHT_GREEN,
                format!("Completed {} of {} reps", completed, total_reps)
            );

            ui.add_space(8.0);
            ui.label("Relax... next rep starting soon");

            ui.add_space(8.0);
            ui.colored_label(egui::Color32::GRAY, "Press [Esc] to cancel");
        });

        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label("Status:");
            ui.colored_label(BRIGHT_ORANGE, "RESTING");
        });
    }

    /// Render the gesture monitor panel (Performance mode)
    fn show_monitor_panel(&mut self, ui: &mut egui::Ui) {
        // Check for recent hits (for flash indicators)
        let recent_hits = self.hit_log.recent(1);
        let recent_hit_info: Option<(String, u128)> = recent_hits.first().map(|h| {
            (h.gesture_name.clone(), h.timestamp.elapsed().as_millis())
        });

        // LARGE HIT INDICATOR - visible from far away
        egui::Frame::group(ui.style())
            .fill(egui::Color32::from_rgb(20, 20, 20))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.set_height(80.0);

                ui.centered_and_justified(|ui| {
                    if let Some((hit_name, ms)) = &recent_hit_info {
                        if *ms < 800 {
                            ui.colored_label(
                                BRIGHT_GREEN,
                                egui::RichText::new(format!("● {}", hit_name))
                                    .size(48.0)
                                    .strong()
                            );
                        } else {
                            ui.colored_label(
                                egui::Color32::DARK_GRAY,
                                egui::RichText::new("—").size(48.0)
                            );
                        }
                    } else {
                        ui.colored_label(
                            egui::Color32::DARK_GRAY,
                            egui::RichText::new("—").size(48.0)
                        );
                    }
                });
            });

        ui.add_space(8.0);

        // Baseline info panel
        if let Some(ref baseline) = self.vocabulary.baseline {
            egui::Frame::group(ui.style()).show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.colored_label(egui::Color32::GRAY,
                        egui::RichText::new(format!("Baseline: {} frames | Thresholds auto-calibrated", baseline.len())).size(14.0));
                });
            });
            ui.add_space(4.0);
        }

        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("GESTURE MONITOR").strong().size(16.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Show buffer status
                    let buffer_len = self.recognizer.buffer.len();
                    let buffer_needed = 90; // window_size / 2
                    if buffer_len < buffer_needed {
                        ui.colored_label(GOLD, egui::RichText::new(format!("Buffer: {}/{}", buffer_len, buffer_needed)).size(14.0));
                    } else {
                        ui.colored_label(BRIGHT_GREEN, egui::RichText::new(format!("Buffer: {}", buffer_len)).size(14.0));
                    }

                    ui.add_space(20.0);

                    // Show recognizer status
                    let status_text = if self.recognizer.is_active() { "ACTIVE" } else { "STOPPED" };
                    let status_color = if self.recognizer.is_active() { BRIGHT_GREEN } else { BRIGHT_RED };
                    ui.colored_label(status_color, egui::RichText::new(status_text).size(14.0));
                });
            });
            ui.separator();

            // Get current distances from recognizer
            let distances = self.recognizer.current_distances();

            // Collect threshold changes to apply after
            let mut threshold_changes: Vec<(u32, f32)> = Vec::new();

            // Use Grid for clean layout - LARGER FONTS
            egui::Grid::new("gesture_monitor_grid")
                .num_columns(4)
                .spacing([16.0, 12.0])
                .striped(true)
                .show(ui, |ui| {
                    // Header row
                    ui.label(egui::RichText::new("Gesture").strong().size(16.0));
                    ui.label(egui::RichText::new("Distance").strong().size(16.0));
                    ui.label(egui::RichText::new("< Threshold").strong().size(16.0));
                    ui.label(""); // Hit column
                    ui.end_row();

                    // Gesture rows
                    for (id, name, current_dist, threshold) in &distances {
                        let example_count = self.recognizer.example_count(*id);
                        let has_examples = example_count > 0;

                        // Check if THIS gesture had a recent hit (within 600ms)
                        let had_recent_hit = recent_hit_info.as_ref()
                            .map(|(hit_name, ms)| hit_name == name && *ms < 600)
                            .unwrap_or(false);

                        // Gesture name with example count (fixed width)
                        ui.allocate_ui_with_layout(
                            egui::vec2(140.0, 28.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                if has_examples {
                                    ui.label(egui::RichText::new(format!("{} ({})", name, example_count)).size(18.0));
                                } else {
                                    ui.colored_label(GOLD, egui::RichText::new(format!("{} (train)", name)).size(18.0));
                                }
                            }
                        );

                        // Current distance (fixed width) - LARGE, this is key info
                        ui.allocate_ui_with_layout(
                            egui::vec2(90.0, 28.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                if !has_examples {
                                    ui.label(egui::RichText::new("--").size(22.0));
                                } else {
                                    match current_dist {
                                        Some(dist) => {
                                            // Color based on whether below threshold
                                            let color = if *dist < *threshold {
                                                BRIGHT_GREEN
                                            } else {
                                                egui::Color32::LIGHT_GRAY
                                            };
                                            ui.colored_label(color, egui::RichText::new(format!("{:.0}", dist)).size(22.0).strong());
                                        }
                                        None => {
                                            ui.colored_label(egui::Color32::GRAY, egui::RichText::new("...").size(22.0));
                                        }
                                    }
                                }
                            }
                        );

                        // Threshold slider (higher = easier to hit)
                        let mut thresh = *threshold;
                        ui.horizontal(|ui| {
                            let slider = egui::Slider::new(&mut thresh, 100.0..=10000.0)
                                .logarithmic(true)
                                .show_value(false)
                                .clamping(egui::SliderClamping::Always);

                            if ui.add(slider).changed() {
                                threshold_changes.push((*id, thresh));
                            }

                            // Show numeric value - larger
                            ui.colored_label(egui::Color32::GRAY, egui::RichText::new(format!("{:.0}", thresh)).size(16.0));
                        });

                        // Hit indicator (fixed width)
                        ui.allocate_ui_with_layout(
                            egui::vec2(70.0, 28.0),
                            egui::Layout::left_to_right(egui::Align::Center),
                            |ui| {
                                if had_recent_hit {
                                    ui.colored_label(BRIGHT_GREEN, egui::RichText::new("● HIT").size(18.0).strong());
                                }
                            }
                        );

                        ui.end_row();
                    }
                });

            ui.add_space(8.0);

            // Recognition timing controls
            ui.horizontal(|ui| {
                ui.colored_label(egui::Color32::GRAY, egui::RichText::new("HIT fires when Distance stays below Threshold").size(13.0));
            });

            ui.horizontal(|ui| {
                // Debounce control
                ui.label("Debounce:");
                let mut confirm = self.recognition_config.confirm_ms as i32;
                if ui.add(egui::DragValue::new(&mut confirm)
                    .range(0..=500)
                    .speed(5)
                    .suffix("ms")).changed()
                {
                    self.recognition_config.confirm_ms = confirm as u64;
                    self.recognizer.set_confirm_ms(self.recognition_config.confirm_ms);
                }
                ui.colored_label(egui::Color32::GRAY, "(filters noise)");

                ui.add_space(20.0);

                // Cooldown control
                ui.label("Cooldown:");
                let mut refractory = self.recognition_config.refractory_ms as i32;
                if ui.add(egui::DragValue::new(&mut refractory)
                    .range(100..=2000)
                    .speed(10)
                    .suffix("ms")).changed()
                {
                    self.recognition_config.refractory_ms = refractory as u64;
                    self.recognizer.set_refractory_ms(self.recognition_config.refractory_ms);
                }
                ui.colored_label(egui::Color32::GRAY, "(min time between hits)");
            });

            // Apply threshold changes
            for (id, new_threshold) in threshold_changes {
                self.recognizer.set_threshold(id, new_threshold);
                if let Some(gesture) = self.vocabulary.get_gesture_mut(id) {
                    gesture.threshold = new_threshold;
                }
                self.mark_dirty();
            }

        });
    }

    /// Render the hit log panel (Performance mode)
    fn show_hit_log_panel(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("HIT LOG");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.colored_label(egui::Color32::GRAY, format!("{} total", self.hit_log.len()));
                });
            });
            ui.separator();

            let recent = self.hit_log.recent(8);
            if recent.is_empty() {
                ui.colored_label(egui::Color32::GRAY, "No hits yet - perform a trained gesture");
            } else {
                for (i, entry) in recent.iter().enumerate() {
                    let ms_ago = entry.timestamp.elapsed().as_millis();
                    let is_recent = ms_ago < 1000;

                    ui.horizontal(|ui| {
                        // Time indicator
                        let time_text = if ms_ago < 1000 {
                            "just now".to_string()
                        } else {
                            format!("{:.0}s ago", ms_ago as f32 / 1000.0)
                        };
                        ui.colored_label(egui::Color32::GRAY, format!("{:>8}", time_text));

                        // Gesture name - brighter if recent
                        let name_color = if is_recent && i == 0 { BRIGHT_GREEN } else { egui::Color32::LIGHT_GRAY };
                        ui.colored_label(name_color, &entry.gesture_name);

                        // OSC address sent
                        ui.colored_label(egui::Color32::DARK_GRAY, format!("→ {}", entry.osc_address));
                    });
                }
            }
        });
    }
}
