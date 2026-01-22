use eframe::egui;

use crate::model::{Vocabulary, Example};
use crate::osc::{OscReceiverHandle, ConnectionStatus, OscSender, SenderStatus};
use crate::engine::{Recognizer, RecordingSession, HitLog};

// Custom colors - gold instead of yellow for better readability
const GOLD: egui::Color32 = egui::Color32::from_rgb(255, 185, 50);
const BRIGHT_GREEN: egui::Color32 = egui::Color32::from_rgb(100, 220, 100);
const BRIGHT_RED: egui::Color32 = egui::Color32::from_rgb(255, 100, 100);
const BRIGHT_BLUE: egui::Color32 = egui::Color32::from_rgb(100, 150, 255);

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

/// Training state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingState {
    Idle,
    Recording,
}

/// Main application state
pub struct GestureStudioApp {
    /// The currently loaded vocabulary
    vocabulary: Vocabulary,
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
    /// Current training state
    training_state: TrainingState,
    /// Active recording session
    recording_session: Option<RecordingSession>,
    /// Recording duration in seconds
    record_duration: f32,
    /// Last detected gesture name (for display)
    last_detected: Option<String>,
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
            mode: AppMode::Training,
            osc_receiver,
            osc_sender,
            recognizer,
            hit_log: HitLog::new(100),
            selected_gesture_id,
            training_state: TrainingState::Idle,
            recording_session: None,
            record_duration: 3.0,
            last_detected: None,
        }
    }

    /// Sync recognizer with vocabulary examples
    #[allow(dead_code)]
    fn sync_recognizer_examples(&mut self) {
        for gesture in &self.vocabulary.gestures {
            // Clear existing examples in recognizer
            self.recognizer.clear_examples(gesture.id);

            // Add examples from vocabulary
            for example in &gesture.examples {
                self.recognizer.add_example(gesture.id, example.frames.clone());
            }
        }
    }
}

impl eframe::App for GestureStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the OSC receiver for new frames
        let frames = self.osc_receiver.poll();

        // Process frames
        for frame in frames {
            // If recording, add to recording session
            let mut recording_complete = false;
            if let Some(ref mut session) = self.recording_session {
                recording_complete = session.add_frame(frame.clone());
            }

            if recording_complete {
                // Recording finished - save the example
                if let Some(gesture_id) = self.selected_gesture_id {
                    if let Some(session) = self.recording_session.take() {
                        let frame_count = session.frame_count();
                        let frames = session.into_frames();
                        let example = Example::new(frames.clone(), frame_count as u64);

                        // Add to vocabulary
                        if let Some(gesture) = self.vocabulary.get_gesture_mut(gesture_id) {
                            gesture.add_example(example);
                        }

                        // Add to recognizer
                        self.recognizer.add_example(gesture_id, frames);
                    }
                }
                self.training_state = TrainingState::Idle;
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

        // Request continuous repaints to keep the UI responsive
        ctx.request_repaint_after(std::time::Duration::from_millis(16)); // ~60fps for smooth updates

        // Top panel with title and mode selector
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("RALF Gesture Studio");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Mode selector dropdown
                    let old_mode = self.mode;
                    egui::ComboBox::from_id_salt("mode_selector")
                        .selected_text(self.mode.as_str())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.mode, AppMode::Training, "Training");
                            ui.selectable_value(&mut self.mode, AppMode::Performance, "Performance");
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
    fn show_vocabulary_panel(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("VOCABULARY");
            });
            ui.separator();

            ui.horizontal(|ui| {
                ui.label("Name:");
                ui.label(&self.vocabulary.name);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new("Save As").min_size(egui::vec2(80.0, 28.0))).clicked() {
                        // TODO: Implement save as
                    }
                    if ui.add(egui::Button::new("Open").min_size(egui::vec2(70.0, 28.0))).clicked() {
                        // TODO: Implement open
                    }
                    if ui.add(egui::Button::new("New").min_size(egui::vec2(60.0, 28.0))).clicked() {
                        // TODO: Implement new
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

            // Gesture rows - collect IDs first to avoid borrow issues
            let gesture_data: Vec<_> = self.vocabulary.gestures.iter()
                .map(|g| (g.id, g.name.clone(), g.example_count(), g.osc_address.clone(), g.has_examples()))
                .collect();

            let mut gesture_to_select = None;

            for (id, name, example_count, osc_address, has_examples) in gesture_data {
                ui.horizontal(|ui| {
                    // Status indicator (filled if has examples, empty if not)
                    if has_examples {
                        ui.colored_label(BRIGHT_GREEN, "●");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "○");
                    }

                    // Highlight if selected
                    let is_selected = self.selected_gesture_id == Some(id);
                    if is_selected {
                        ui.colored_label(BRIGHT_BLUE, &name);
                    } else {
                        ui.label(&name);
                    }
                    ui.add_space(80.0 - name.len() as f32 * 7.0);
                    ui.label(format!("{}", example_count));
                    ui.add_space(45.0);
                    ui.label(&osc_address);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new("×").min_size(egui::vec2(32.0, 28.0))).clicked() {
                            // TODO: Delete gesture
                        }
                        // Select button to choose for training
                        let select_text = if is_selected { "Selected" } else { "Select" };
                        if ui.add(egui::Button::new(select_text).min_size(egui::vec2(70.0, 28.0))).clicked() {
                            gesture_to_select = Some(id);
                        }
                    });
                });
            }

            // Apply selection after iteration
            if let Some(id) = gesture_to_select {
                self.selected_gesture_id = Some(id);
            }

            ui.add_space(12.0);
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
            }
        });
    }

    /// Render the training panel
    fn show_train_panel(&mut self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("RECORD");
            });
            ui.separator();

            // Get selected gesture name
            let selected_name = self.selected_gesture_id
                .and_then(|id| self.vocabulary.get_gesture(id))
                .map(|g| g.name.clone())
                .unwrap_or_else(|| "(none)".to_string());

            // Training parameters
            ui.horizontal(|ui| {
                ui.label("Gesture:");
                ui.colored_label(BRIGHT_BLUE, &selected_name);

                ui.add_space(20.0);
                ui.label("Duration:");
                ui.add(egui::DragValue::new(&mut self.record_duration)
                    .range(0.5..=10.0)
                    .speed(0.1)
                    .suffix("s"));
            });

            ui.add_space(16.0);

            // Recording state display
            match self.training_state {
                TrainingState::Idle => {
                    // Start button area
                    ui.vertical_centered(|ui| {
                        let can_record = self.selected_gesture_id.is_some()
                            && self.osc_receiver.state.status == ConnectionStatus::Receiving;

                        let button = egui::Button::new(
                            egui::RichText::new("● RECORD")
                                .size(24.0)
                                .strong()
                        )
                        .min_size(egui::vec2(200.0, 60.0));

                        let response = ui.add_enabled(can_record, button);
                        if response.clicked() {
                            // Start recording
                            self.training_state = TrainingState::Recording;
                            self.recording_session = Some(RecordingSession::new(self.record_duration));
                        }

                        if !can_record {
                            if self.selected_gesture_id.is_none() {
                                ui.label("Select a gesture above to record");
                            } else {
                                ui.label("Waiting for OSC data...");
                            }
                        }
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("Status:");
                        ui.colored_label(egui::Color32::GRAY, "IDLE");
                    });
                }

                TrainingState::Recording => {
                    // Show recording progress
                    let progress = self.recording_session
                        .as_ref()
                        .map(|s| s.progress())
                        .unwrap_or(0.0);

                    let frame_count = self.recording_session
                        .as_ref()
                        .map(|s| s.frame_count())
                        .unwrap_or(0);

                    ui.vertical_centered(|ui| {
                        ui.colored_label(BRIGHT_RED, egui::RichText::new("● RECORDING").size(24.0).strong());

                        ui.add_space(8.0);

                        // Progress bar
                        let progress_bar = egui::ProgressBar::new(progress)
                            .show_percentage()
                            .animate(true);
                        ui.add(progress_bar);

                        ui.add_space(4.0);
                        ui.label(format!("{} frames captured", frame_count));
                    });

                    ui.add_space(8.0);
                    ui.horizontal(|ui| {
                        ui.label("Status:");
                        ui.colored_label(BRIGHT_RED, "RECORDING");
                    });
                }
            }
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
                ui.add_space(60.0);
                ui.label("Examples");
                ui.add_space(20.0);
                ui.label("Distance");
                ui.add_space(40.0);
                ui.label("Threshold");
            });
            ui.separator();

            // Get current distances from recognizer
            let distances = self.recognizer.current_distances();

            // Gesture rows
            for (id, name, current_dist, threshold) in distances {
                let example_count = self.recognizer.example_count(id);

                ui.horizontal(|ui| {
                    ui.label(&name);
                    ui.add_space(60.0 - name.len() as f32 * 7.0);

                    ui.label(format!("{}", example_count));
                    ui.add_space(45.0);

                    // Distance
                    match current_dist {
                        Some(dist) => {
                            let color = if dist < threshold { BRIGHT_GREEN } else { egui::Color32::GRAY };
                            ui.colored_label(color, format!("{:.0}", dist));
                        }
                        None => {
                            ui.colored_label(egui::Color32::GRAY, "--");
                        }
                    };
                    ui.add_space(60.0);

                    // Threshold
                    ui.label(format!("{:.0}", threshold));
                });
            }

            ui.add_space(16.0);
            ui.separator();

            // Last detected gesture
            ui.vertical_centered(|ui| {
                match &self.last_detected {
                    Some(name) => {
                        ui.colored_label(BRIGHT_GREEN, format!("Last: {}", name));
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
