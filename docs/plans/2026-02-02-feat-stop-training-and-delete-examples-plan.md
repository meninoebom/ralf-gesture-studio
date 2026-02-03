---
title: "feat: Stop Training Button & Delete Individual Examples"
type: feat
date: 2026-02-02
---

# Stop Training Button & Delete Individual Examples

## Overview

Two focused UI features to improve the training workflow:

1. **Visible Stop button** during active training sessions (replacing the Esc-only hint)
2. **Delete individual examples** from a gesture (currently impossible — only the count is visible)

## Problem Statement

**Stop Training**: The only way to cancel a training session is pressing Escape. There's no visible button, which is easy to forget during flow state. Dancers need a large, obvious click target.

**Delete Examples**: Once recorded, examples cannot be individually removed. If a bad take is captured, the only option is to delete the entire gesture and re-record everything. This wastes significant training effort.

## Proposed Solution

### Feature 1: Visible Stop Button

Replace the `Press [Esc] to cancel` hint text in all active training states (countdown, capturing, resting) with a clickable Stop button that also mentions the Esc shortcut.

**Backend**: The `cancel_training` Tauri command and `TrainingSession::cancel()` method already exist. One fix needed: `cancel()` should also clear `completed_examples` to match the "discard everything" intent.

**Frontend**: Replace the `<p class="dim">Press [Esc] to cancel</p>` in each active state render with a styled button.

### Feature 2: Delete Individual Examples

Add a new `delete_example` Tauri command and expand the UI to show per-example details with delete buttons.

**Layers to modify**:
- Model: add `Gesture::remove_example(index)` method
- GUI: add `delete_example` command, expand `ExampleDto` with timestamp and duration
- Frontend: make example count expandable to show example list with delete buttons

## Acceptance Criteria

### Stop Training Button
- [x] A visible "Stop Training" button appears during countdown, capturing, and resting states
- [x] Button calls existing `cancel_training` command (same as Escape key)
- [x] `cancel()` explicitly clears `completed_examples` in addition to `current_frames`
- [x] No confirmation dialog (matches Escape key behavior, preserves flow state)
- [x] Button shows Esc shortcut hint: `Stop (Esc)`

### Delete Individual Examples
- [x] Example count in gesture list is clickable and expands to show individual examples
- [x] Each example row shows: recorded time, duration, frame count
- [x] Each example row has a delete (x) button
- [ ] Deletion shows `confirm()` dialog (consistent with `delete_gesture`) — deferred; removed during debugging, add back later
- [x] Backend validates gesture exists and index is in bounds
- [x] After deletion: statistics are cleared if < 2 examples remain
- [x] After deletion: `sync_recognizer()` rebuilds recognizer
- [x] After deletion: `mark_dirty()` triggers auto-save
- [ ] Delete buttons disabled during active training sessions — deferred; `trainingActive` removed from hash to fix handler stability
- [x] Example list only rendered in Training mode (not Performance mode)

## Technical Approach

### Files to Modify

| File | Changes |
|------|---------|
| `src/engine/training.rs` | Add `self.completed_examples.clear()` to `cancel()` |
| `src/model/vocabulary.rs` | Add `Gesture::remove_example(index) -> Result<Example, String>` |
| `src/gui/mod.rs` | Add `delete_example` command; expand `ExampleDto` with `recorded_at` and `duration_ms` |
| `src/main.rs` | Register `gui::delete_example` in `invoke_handler` |
| `ui/main.js` | Stop button in training states; expandable example list with delete |
| `ui/styles.css` | Stop button styling; example list styling |

### Backend Changes

#### 1. Fix `cancel()` — `src/engine/training.rs:163-166`

```rust
pub fn cancel(&mut self) {
    self.transition_to(SessionState::Idle);
    self.current_frames.clear();
    self.completed_examples.clear();  // NEW: explicitly discard completed reps
}
```

#### 2. Add `remove_example()` — `src/model/vocabulary.rs`

```rust
impl Gesture {
    pub fn remove_example(&mut self, index: usize) -> Result<Example, String> {
        if index >= self.examples.len() {
            return Err(format!(
                "Example index {} out of bounds (gesture has {} examples)",
                index, self.examples.len()
            ));
        }
        let removed = self.examples.remove(index);
        if self.examples.len() < 2 {
            self.clear_statistics();
        }
        Ok(removed)
    }
}
```

#### 3. Expand `ExampleDto` — `src/gui/mod.rs`

```rust
#[derive(Serialize, Clone)]
pub struct ExampleDto {
    pub frame_count: usize,
    pub duration_ms: u64,          // NEW
    pub recorded_at: String,       // NEW: ISO 8601 string
}
```

Update `to_dto()` or wherever `ExampleDto` is constructed to include the new fields.

#### 4. Add `delete_example` command — `src/gui/mod.rs`

Follow the `delete_gesture` pattern at `src/gui/mod.rs:966-979`:

```rust
#[tauri::command]
pub fn delete_example(
    state: State<Arc<Mutex<AppState>>>,
    gesture_id: u32,
    example_index: usize,
) -> Result<(), String> {
    let mut app = state.lock().map_err(|e| e.to_string())?;
    let gesture = app.vocabulary.get_gesture_mut(gesture_id)
        .ok_or_else(|| format!("Gesture {} not found", gesture_id))?;
    gesture.remove_example(example_index)?;
    app.sync_recognizer();
    app.mark_dirty();
    Ok(())
}
```

#### 5. Register command — `src/main.rs`

Add `gui::delete_example` to the `invoke_handler` list.

### Frontend Changes

#### 6. Stop Button — `ui/main.js` in `updateTrainingState()`

In each of the three active state branches (countdown, capturing, resting), replace:
```html
<p class="dim">Press [Esc] to cancel</p>
```
with:
```html
<button class="stop-training-btn" onclick="cancelTraining()">Stop (Esc)</button>
```

#### 7. Expandable Example List — `ui/main.js` in `renderGestures()`

Make the example count clickable. When clicked, toggle an inline sub-list:

```
[gesture row]
  ├── Example 1: 10:32am · 2.8s · 171 frames  [x]
  ├── Example 2: 10:35am · 3.1s · 189 frames  [x]
  └── Example 3: 10:37am · 2.6s · 158 frames  [x]
```

Add a `deleteExample(gestureId, index)` function:

```javascript
async function deleteExample(gestureId, index) {
    if (!confirm('Delete this example?')) return;
    await invoke('delete_example', {
        gestureId: gestureId,
        exampleIndex: index,
    });
}
```

#### 8. Styles — `ui/styles.css`

- `.stop-training-btn`: Red/destructive styling, full-width, large click target
- `.example-list`: Compact list under gesture row
- `.example-item`: Row with details and delete button
- `.example-delete-btn`: Small (x) button matching existing gesture delete style

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Stop while countdown | Immediate cancel, no examples captured → back to idle |
| Stop while capturing (rep 3 of 5) | All 2 completed reps + in-progress frames discarded |
| Stop during rest between reps | All completed reps discarded |
| Training completes during Stop click (~50ms race) | Mutex serializes: either cancel wins (discard) or complete wins (save). Accept this. |
| Delete last example (1→0) | Statistics cleared, gesture shows as untrained (gray dot) |
| Delete to below 2 examples (2→1) | Statistics cleared, auto-threshold has no data |
| Rapid successive deletes | Mutex serializes. UI refreshes at 50ms. Indices shift but each delete validates bounds. |
| Delete during active training | Delete buttons disabled while `trainingState !== 'idle'` |
| Delete in Performance mode | Example list not rendered in Performance mode |

## Testing

- [x] Unit test: `TrainingSession::cancel()` clears both `current_frames` and `completed_examples`
- [x] Unit test: `Gesture::remove_example()` removes correct example, returns error on out-of-bounds
- [x] Unit test: `Gesture::remove_example()` clears statistics when < 2 examples remain
- [ ] Integration: `delete_example` command removes example, triggers sync and dirty flag
- [x] Manual: Stop button visible and functional in all three active training states
- [x] Manual: Example list expands/collapses, delete works, list stays expanded after delete

## References

- Training state machine: `src/engine/training.rs:105-166`
- Existing cancel command: `src/gui/mod.rs:1017-1021`
- Delete gesture pattern: `src/gui/mod.rs:966-979`
- ExampleDto: `src/gui/mod.rs:527-530`
- Frontend training UI: `ui/main.js:354-453`
- Frontend gesture list: `ui/main.js:288-352`
- Frontend cancel function: `ui/main.js:688-692`
