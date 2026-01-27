---
title: UX Improvements - Threshold, Training Config, and Visual Clarity
type: feat
date: 2026-01-26
---

# UX Improvements - Threshold, Training Config, and Visual Clarity

## Overview

Three UX improvements to make the gesture training and recognition workflow smoother:

1. **Threshold calibration** - Address the issue where users need to push threshold toward the top of the range for smooth recognition
2. **Training configuration controls** - Make count-in, duration, and reps use integer values with better defaults and mouse-draggable sliders
3. **Mode toggle visibility** - Make Training/Performance toggle more prominent and improve overall readability

## Problem Statement

1. **Threshold Issue**: Users consistently need to set threshold near the top of the range (e.g., ~8000-10000) to get reliable gesture recognition. The current range of 100-10000 with logarithmic scaling doesn't match user expectations.

2. **Training Config UX**: The DragValue controls for training parameters show floats (e.g., "2.0s") when integers would be clearer since they represent whole seconds. No mouse wheel support for quick adjustments. Defaults aren't optimal (3s duration is too long, 3s countdown is excessive).

3. **Visual Clarity**: The Training/Performance mode toggle is a small ComboBox easily missed. The overall gray-on-gray palette makes it hard to scan quickly, especially from a distance during performance.

## Proposed Solution

### 1. Threshold Calibration

**Root Cause Analysis**: The threshold slider range (100-10000) with logarithmic scaling means:
- Low end (100-1000): Very sensitive, fires too easily
- High end (5000-10000): Sweet spot for most users
- Current default positioning doesn't guide users to the optimal range

**Solution**: Invert the threshold semantics OR adjust the default/range:

Option A (Recommended): **Keep semantics, adjust defaults and hints**
- Change default threshold from current value to ~5000 (mid-high range)
- Add visual hint showing "← Stricter | Looser →" labels on slider
- Keep range 100-10000 logarithmic

Option B: **Invert to "Sensitivity" semantics**
- Rename to "Sensitivity" where higher = more sensitive (fires easier)
- Internally convert: sensitivity = 10100 - threshold
- More intuitive: "turn up sensitivity to fire more often"

**Decision**: Go with Option A - less risky, preserves existing behavior, just improves defaults and guidance.

### 2. Training Configuration Controls

**Changes to `gui/mod.rs`** in `show_train_panel`:

```rust
// BEFORE (current)
ui.label("Count-in:");
ui.add(egui::DragValue::new(&mut self.training_config.countdown_secs)
    .range(1.0..=10.0)
    .speed(0.1)
    .suffix("s"));

// AFTER (improved)
ui.label("Count-in:");
ui.add(egui::DragValue::new(&mut countdown_int)
    .range(1..=10)
    .speed(0.5)  // Slower speed for easier mouse control
    .suffix("s"));
self.training_config.countdown_secs = countdown_int as f32;
```

**New Defaults** (in `engine/training.rs`):

```rust
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            reps: 5,              // Keep at 5 (good)
            duration_secs: 2.0,   // Was 3.0, reduce to 2.0
            rest_secs: 2.0,       // Keep at 2.0 (good)
            countdown_secs: 2.0,  // Was 3.0, reduce to 2.0
        }
    }
}
```

**Implementation Notes**:
- Use temporary integer variables in the UI, cast to f32 for storage
- Set DragValue speed to 0.5 for comfortable mouse dragging
- Range limits: count-in (1-10), duration (1-10), rest (1-10), reps (1-20)

### 3. Mode Toggle and Visual Clarity

**Replace ComboBox with prominent toggle buttons:**

```rust
// BEFORE: Small ComboBox
egui::ComboBox::from_id_salt("mode_selector")
    .selected_text(self.mode.as_str())
    .show_ui(ui, |ui| { ... });

// AFTER: Large toggle buttons
ui.horizontal(|ui| {
    let training_btn = egui::Button::new(
        egui::RichText::new("TRAINING")
            .size(18.0)
            .strong()
    )
    .fill(if self.mode == AppMode::Training { BRIGHT_BLUE } else { egui::Color32::DARK_GRAY })
    .min_size(egui::vec2(120.0, 36.0));

    if ui.add(training_btn).clicked() {
        self.mode = AppMode::Training;
    }

    let perf_btn = egui::Button::new(
        egui::RichText::new("PERFORMANCE")
            .size(18.0)
            .strong()
    )
    .fill(if self.mode == AppMode::Performance { BRIGHT_GREEN } else { egui::Color32::DARK_GRAY })
    .min_size(egui::vec2(140.0, 36.0));

    if ui.add(perf_btn).clicked() {
        self.mode = AppMode::Performance;
    }
});
```

**Improve visual contrast:**

1. **Panel headers** - Add subtle background color differentiation
2. **Section separators** - Make separators slightly more visible
3. **Active states** - Use brighter colors for active/selected states
4. **Labels** - Increase contrast between labels and values

```rust
// Add panel-specific styling
const PANEL_BG_DARK: egui::Color32 = egui::Color32::from_rgb(30, 30, 35);
const PANEL_BG_LIGHT: egui::Color32 = egui::Color32::from_rgb(40, 40, 45);

// Use alternating backgrounds for panels
egui::Frame::group(ui.style())
    .fill(PANEL_BG_DARK)
    .show(ui, |ui| { ... });
```

## Acceptance Criteria

- [x] Threshold slider shows guidance labels ("← Stricter | Easier →")
- [x] Default gesture threshold is 5000 (not 1000 or lower)
- [x] Training config controls increment by whole integers only
- [x] Training config defaults: 5 reps, 2s count-in, 2s duration, 2s rest
- [x] Mode toggle is prominent buttons, not a dropdown
- [x] Active mode button is visually distinct (colored fill)
- [x] Panel sections have improved contrast for readability
- [x] All changes work with existing vocabulary files (backwards compatible)

## Technical Considerations

### Backwards Compatibility
- Vocabulary files store per-gesture thresholds as floats - no migration needed
- TrainingConfig is not persisted - defaults change is safe
- AppMode enum unchanged - UI-only change

### Testing
- Manual testing with existing .ralf files to ensure thresholds still work
- Verify training flow still captures correct number of frames
- Test mode switching during idle and verify recognizer starts/stops correctly

## Files to Modify

| File | Changes |
|------|---------|
| `src/gui/mod.rs` | Mode toggle buttons, panel styling, DragValue integer conversion, threshold slider hints |
| `src/engine/training.rs` | Update TrainingConfig defaults |
| `src/model/vocabulary.rs` | Update default gesture threshold (if hardcoded) |

## MVP Implementation

### src/engine/training.rs

```rust
impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            reps: 5,
            duration_secs: 2.0,   // Changed from 3.0
            rest_secs: 2.0,
            countdown_secs: 2.0,  // Changed from 3.0
        }
    }
}
```

### src/gui/mod.rs - Mode Toggle

```rust
// In top_panel, replace ComboBox with:
ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
    ui.add_enabled_ui(!self.training_session.is_active(), |ui| {
        let old_mode = self.mode;

        // Performance button first (right side)
        let perf_selected = self.mode == AppMode::Performance;
        let perf_btn = egui::Button::new(
            egui::RichText::new("PERFORMANCE")
                .size(16.0)
                .color(if perf_selected { egui::Color32::WHITE } else { egui::Color32::GRAY })
        )
        .fill(if perf_selected { BRIGHT_GREEN } else { egui::Color32::from_rgb(50, 50, 50) })
        .min_size(egui::vec2(130.0, 32.0));

        if ui.add(perf_btn).clicked() && !perf_selected {
            self.mode = AppMode::Performance;
        }

        // Training button
        let train_selected = self.mode == AppMode::Training;
        let train_btn = egui::Button::new(
            egui::RichText::new("TRAINING")
                .size(16.0)
                .color(if train_selected { egui::Color32::WHITE } else { egui::Color32::GRAY })
        )
        .fill(if train_selected { BRIGHT_BLUE } else { egui::Color32::from_rgb(50, 50, 50) })
        .min_size(egui::vec2(110.0, 32.0));

        if ui.add(train_btn).clicked() && !train_selected {
            self.mode = AppMode::Training;
        }

        // Handle mode change
        if old_mode != self.mode {
            if self.mode == AppMode::Performance {
                self.recognizer.start();
            } else {
                self.recognizer.stop();
            }
        }
    });
});
```

### src/gui/mod.rs - Training Config Integers

```rust
// In show_train_panel, replace DragValue controls:
ui.add_enabled_ui(is_idle, |ui| {
    ui.label("Reps:");
    let mut reps = self.training_config.reps as i32;
    if ui.add(egui::DragValue::new(&mut reps).range(1..=20).speed(0.5)).changed() {
        self.training_config.reps = reps as u32;
    }

    ui.add_space(10.0);
    ui.label("Count-in:");
    let mut countdown = self.training_config.countdown_secs.round() as i32;
    if ui.add(egui::DragValue::new(&mut countdown).range(1..=10).speed(0.5).suffix("s")).changed() {
        self.training_config.countdown_secs = countdown as f32;
    }

    ui.add_space(10.0);
    ui.label("Capture:");
    let mut duration = self.training_config.duration_secs.round() as i32;
    if ui.add(egui::DragValue::new(&mut duration).range(1..=10).speed(0.5).suffix("s")).changed() {
        self.training_config.duration_secs = duration as f32;
    }

    ui.add_space(10.0);
    ui.label("Rest:");
    let mut rest = self.training_config.rest_secs.round() as i32;
    if ui.add(egui::DragValue::new(&mut rest).range(1..=10).speed(0.5).suffix("s")).changed() {
        self.training_config.rest_secs = rest as f32;
    }
});
```

### src/gui/mod.rs - Threshold Slider with Hints

```rust
// In show_monitor_panel, after the threshold slider:
ui.horizontal(|ui| {
    let slider = egui::Slider::new(&mut thresh, 100.0..=10000.0)
        .logarithmic(true)
        .show_value(false)
        .clamping(egui::SliderClamping::Always);

    if ui.add(slider).changed() {
        threshold_changes.push((*id, thresh));
    }

    // Show numeric value
    ui.colored_label(egui::Color32::GRAY, egui::RichText::new(format!("{:.0}", thresh)).size(16.0));
});

// Add hint row below the grid
ui.add_space(4.0);
ui.horizontal(|ui| {
    ui.colored_label(egui::Color32::DARK_GRAY, egui::RichText::new("← Stricter").size(12.0));
    ui.add_space(ui.available_width() - 100.0);
    ui.colored_label(egui::Color32::DARK_GRAY, egui::RichText::new("Easier →").size(12.0));
});
```

### src/model/vocabulary.rs - Default Threshold

```rust
// In Gesture::new or wherever default threshold is set:
pub fn new(id: u32, name: &str) -> Self {
    Self {
        id,
        name: name.to_string(),
        osc_address: format!("/gesture/{}", id),
        threshold: 5000.0,  // Changed from lower default
        examples: Vec::new(),
    }
}
```

## References

- egui DragValue docs: https://docs.rs/egui/latest/egui/widgets/struct.DragValue.html
- Current implementation: `src/gui/mod.rs:808-886` (training panel)
- Mode selector: `src/gui/mod.rs:376-394`
- Threshold slider: `src/gui/mod.rs:1209-1220`
