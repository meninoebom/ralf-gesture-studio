use eframe::egui;

use crate::model::Vocabulary;
use crate::osc::{OscReceiverHandle, ConnectionStatus};

// Custom colors - gold instead of yellow for better readability
const GOLD: egui::Color32 = egui::Color32::from_rgb(255, 185, 50);
const BRIGHT_GREEN: egui::Color32 = egui::Color32::from_rgb(100, 220, 100);
const BRIGHT_RED: egui::Color32 = egui::Color32::from_rgb(255, 100, 100);

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
    /// Current application mode
    mode: AppMode,
    /// Handle to the OSC receiver
    osc_receiver: OscReceiverHandle,
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

        Self {
            vocabulary,
            mode: AppMode::Training,
            osc_receiver,
        }
    }
}

impl eframe::App for GestureStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll the OSC receiver for new events
        self.osc_receiver.poll();

        // Request continuous repaints to keep the UI responsive
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        // Top panel with title and mode selector
        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("RALF Gesture Studio");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    // Mode selector dropdown
                    egui::ComboBox::from_id_salt("mode_selector")
                        .selected_text(self.mode.as_str())
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut self.mode, AppMode::Training, "Training");
                            ui.selectable_value(&mut self.mode, AppMode::Performance, "Performance");
                        });
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
    fn show_connection_panel(&self, ui: &mut egui::Ui) {
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
                                (BRIGHT_GREEN, "RECEIVING", format!("({}ms ago)", ms))
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
                    ui.horizontal(|ui| {
                        ui.colored_label(BRIGHT_GREEN, "●");
                        ui.colored_label(BRIGHT_GREEN, "READY");
                    });
                });
            });
        });
    }

    /// Render the gestures panel
    fn show_gestures_panel(&self, ui: &mut egui::Ui) {
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

            // Gesture rows
            for gesture in &self.vocabulary.gestures {
                ui.horizontal(|ui| {
                    // Status indicator (filled if has examples, empty if not)
                    if gesture.has_examples() {
                        ui.colored_label(BRIGHT_GREEN, "●");
                    } else {
                        ui.colored_label(egui::Color32::GRAY, "○");
                    }

                    ui.label(&gesture.name);
                    ui.add_space(80.0 - gesture.name.len() as f32 * 7.0);
                    ui.label(format!("{}", gesture.example_count()));
                    ui.add_space(45.0);
                    ui.label(&gesture.osc_address);

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.add(egui::Button::new("×").min_size(egui::vec2(32.0, 28.0))).clicked() {
                            // TODO: Delete gesture
                        }
                        if ui.add(egui::Button::new("Train").min_size(egui::vec2(70.0, 28.0))).clicked() {
                            // TODO: Start training
                        }
                    });
                });
            }

            ui.add_space(12.0);
            if ui.add(egui::Button::new("+ Add Gesture").min_size(egui::vec2(140.0, 32.0))).clicked() {
                // TODO: Add gesture
            }
        });
    }

    /// Render the training panel
    fn show_train_panel(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("TRAIN");
            });
            ui.separator();

            // Training parameters
            ui.horizontal(|ui| {
                ui.label("Gesture:");
                egui::ComboBox::from_id_salt("gesture_selector")
                    .selected_text(
                        self.vocabulary
                            .gestures
                            .first()
                            .map(|g| g.name.as_str())
                            .unwrap_or("(none)")
                    )
                    .show_ui(ui, |ui| {
                        for gesture in &self.vocabulary.gestures {
                            ui.selectable_label(false, &gesture.name);
                        }
                    });

                ui.add_space(20.0);
                ui.label("Reps:");
                ui.label("5");

                ui.add_space(20.0);
                ui.label("Duration:");
                ui.label("3.0s");

                ui.add_space(20.0);
                ui.label("Rest:");
                ui.label("2.0s");
            });

            ui.add_space(16.0);

            // Start button area
            ui.add_space(8.0);
            ui.vertical_centered(|ui| {
                let button = egui::Button::new(
                    egui::RichText::new("START\n(spacebar)")
                        .size(24.0)
                        .strong()
                )
                .min_size(egui::vec2(240.0, 80.0));

                if ui.add(button).clicked() {
                    // TODO: Start training session
                }
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                ui.label("Status:");
                ui.colored_label(egui::Color32::GRAY, "IDLE");
            });
        });
    }

    /// Render the gesture monitor panel (Performance mode)
    fn show_monitor_panel(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("GESTURE MONITOR");
            });
            ui.separator();

            // Header
            ui.horizontal(|ui| {
                ui.label("Gesture");
                ui.add_space(40.0);
                ui.label("Threshold");
                ui.add_space(60.0);
                ui.label("Distance");
                ui.add_space(40.0);
                ui.label("Output");
                ui.add_space(40.0);
                ui.label("Status");
            });
            ui.separator();

            // Gesture rows
            for gesture in &self.vocabulary.gestures {
                ui.horizontal(|ui| {
                    ui.label(&gesture.name);
                    ui.add_space(40.0 - gesture.name.len() as f32 * 7.0);

                    // Threshold slider placeholder
                    ui.label(format!("{:.0}", gesture.threshold));
                    ui.add_space(80.0);

                    // Distance placeholder
                    ui.label("--");
                    ui.add_space(60.0);

                    // Output address
                    ui.label(&gesture.osc_address);
                    ui.add_space(40.0);

                    // Status indicator
                    ui.colored_label(egui::Color32::GRAY, "●");
                });
            }

            ui.add_space(16.0);
            ui.separator();
            ui.vertical_centered(|ui| {
                ui.colored_label(egui::Color32::GRAY, "(no gesture detected)");
            });
        });
    }

    /// Render the hit log panel (Performance mode)
    fn show_hit_log_panel(&self, ui: &mut egui::Ui) {
        egui::Frame::group(ui.style()).show(ui, |ui| {
            ui.set_width(ui.available_width());

            ui.horizontal(|ui| {
                ui.strong("HIT LOG");
            });
            ui.separator();

            ui.colored_label(egui::Color32::GRAY, "(no hits recorded)");
        });
    }
}
