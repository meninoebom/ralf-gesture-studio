use std::path::PathBuf;

use eframe::egui;

use crate::model::{Vocabulary, Example, save_vocabulary, load_vocabulary, default_vocabulary_dir};
use crate::osc::{OscReceiverHandle, ConnectionStatus, OscSender, SenderStatus};
use crate::engine::{Recognizer, HitLog, TrainingSession, TrainingConfig, SessionState};

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
        let mut recognizer = Recognizer::new(600, 180, 1000);

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
        }
    }

    /// Sync recognizer with vocabulary
    fn sync_recognizer(&mut self) {
        // Rebuild recognizer with current vocabulary
        self.recognizer = Recognizer::new(600, 180, 1000);

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
                self.delete_gesture_id = None;
            }
        });

        // Poll the OSC receiver for new frames
        let frames = self.osc_receiver.poll();

        // Process frames
        for frame in frames {
            // If training, add frames to training session
            if self.training_session.state == SessionState::Capturing {
                self.training_session.add_frame(frame.clone());
            }

            // Always feed to recognizer for buffer/matching
            if let Some(result) = self.recognizer.process_frame(frame) {
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

                // Train panel (only in Training mode)
                if self.mode == AppMode::Training {
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
                ui.label(&self.vocabulary.name);

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

                            // Finish on Enter
                            if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                if let Some(gesture) = self.vocabulary.get_gesture_mut(id) {
                                    gesture.name = new_name.clone();
                                    self.mark_dirty();
                                }
                                self.renaming_gesture = None;
                            }
                        }
                    } else {
                        // Show name (double-click to rename)
                        let name_response = if is_selected {
                            ui.colored_label(BRIGHT_BLUE, &name)
                        } else {
                            ui.label(&name)
                        };

                        if name_response.double_clicked() && enabled {
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

    /// Render the training panel
    fn show_train_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("TRAIN");
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
                    ui.label("Duration:");
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
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("GESTURE MONITOR");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let status_text = if self.recognizer.is_active() { "ACTIVE" } else { "STOPPED" };
                    let status_color = if self.recognizer.is_active() { BRIGHT_GREEN } else { egui::Color32::GRAY };
                    ui.colored_label(status_color, status_text);
                });
            });
            ui.separator();

            // Header
            ui.horizontal(|ui| {
                ui.label("Gesture");
                ui.add_space(50.0);
                ui.label("Examples");
                ui.add_space(10.0);
                ui.label("Distance");
                ui.add_space(30.0);
                ui.label("Threshold");
            });
            ui.separator();

            // Get current distances from recognizer
            let distances = self.recognizer.current_distances();

            // Collect threshold changes to apply after
            let mut threshold_changes: Vec<(u32, f32)> = Vec::new();

            // Gesture rows with threshold sliders
            for (id, name, current_dist, threshold) in distances {
                let example_count = self.recognizer.example_count(id);

                ui.horizontal(|ui| {
                    ui.label(&name);
                    ui.add_space((50.0 - name.len() as f32 * 7.0).max(5.0));

                    ui.label(format!("{}", example_count));
                    ui.add_space(35.0);

                    // Distance with color indicator
                    match current_dist {
                        Some(dist) => {
                            let color = if dist < threshold { BRIGHT_GREEN } else { egui::Color32::GRAY };
                            ui.colored_label(color, format!("{:>5.0}", dist));
                        }
                        None => {
                            ui.colored_label(egui::Color32::GRAY, "   --");
                        }
                    };
                    ui.add_space(20.0);

                    // Threshold slider
                    let mut thresh = threshold;
                    let slider = egui::Slider::new(&mut thresh, 10.0..=500.0)
                        .show_value(true)
                        .clamping(egui::SliderClamping::Always);

                    if ui.add(slider).changed() {
                        threshold_changes.push((id, thresh));
                    }
                });
            }

            // Apply threshold changes
            for (id, new_threshold) in threshold_changes {
                self.recognizer.set_threshold(id, new_threshold);
                if let Some(gesture) = self.vocabulary.get_gesture_mut(id) {
                    gesture.threshold = new_threshold;
                }
                self.mark_dirty();
            }

            ui.add_space(16.0);
            ui.separator();

            // Last detected gesture
            ui.vertical_centered(|ui| {
                match &self.last_detected {
                    Some(name) => {
                        ui.colored_label(BRIGHT_GREEN, format!("★ {} DETECTED ★", name));
                    }
                    None => {
                        ui.colored_label(egui::Color32::GRAY, "(no gesture detected)");
                    }
                }
            });
        });
    }

    /// Render the hit log panel (Performance mode)
    fn show_hit_log_panel(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("HIT LOG");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("{} hits", self.hit_log.len()));
                });
            });
            ui.separator();

            let recent = self.hit_log.recent(10);
            if recent.is_empty() {
                ui.colored_label(egui::Color32::GRAY, "(no hits recorded)");
            } else {
                for entry in recent {
                    ui.horizontal(|ui| {
                        let ms_ago = entry.timestamp.elapsed().as_millis();
                        ui.colored_label(egui::Color32::GRAY, format!("{:.1}s ago", ms_ago as f32 / 1000.0));
                        ui.colored_label(BRIGHT_GREEN, &entry.gesture_name);
                        ui.label(format!("dist: {:.0}", entry.distance));
                        ui.label(format!("→ {}", entry.osc_address));
                    });
                }
            }
        });
    }
}
