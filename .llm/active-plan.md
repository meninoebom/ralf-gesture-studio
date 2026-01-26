# RALF Gesture Studio - Implementation Plan

**Current Version**: v0.2.0 (Working Baseline)
**Status**: Gesture recognition working reliably

---

## Working Baseline (2026-01-26)

**We have achieved reliable gesture recognition.** This is the foundation to build from.

### What Works
- Train a gesture with 5 repetitions
- Switch to Performance mode
- Perform gesture → Single clean hit detected
- Threshold ~8000 for skeleton data (68 dimensions)

### Reference Implementation
- **Source**: [fiebrink1/wekinator](https://github.com/fiebrink1/wekinator)
- **File**: `src/wekimini/learning/dtw/DtwModel.java`
- **Method**: `classifyContinuous()` (lines 920-970)

---

## The Algorithm

### Core Logic (from recognizer.rs)

```rust
pub fn process_frame(&mut self, frame: Frame) -> Option<RecognitionResult> {
    self.buffer.push(frame);
    self.frame_count += 1;

    // Skip frames for performance (DTW every 4th frame)
    if self.frame_count % self.dtw_skip != 0 {
        return None;
    }

    // Get window and downsample
    let window = Self::downsample(&self.buffer.recent(self.window_size), 4);

    // Compare against all examples for all gestures
    for gesture in &self.gestures {
        for example in gesture.examples() {
            let example_ds = Self::downsample(example, 4);
            let dist = dtw_distance(&window, &example_ds);
            // Track best distance...
        }
    }

    // Update armed state (re-arm when above threshold)
    if distance >= threshold {
        gesture.arm();
    }

    // Fire hit if: below threshold AND armed AND not in cooldown
    if distance < threshold && gesture.is_armed() && !gesture.in_cooldown() {
        gesture.record_hit();  // Disarms
        return Some(hit);
    }
}
```

### Key Components

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Window Size | Fixed (first example length) | Predictable matching |
| Comparison | All examples (not prototype) | Captures variance |
| Frame Skip | Every 4th frame (15Hz) | CPU performance |
| Downsample | 4x (45 frames vs 180) | DTW speed |
| Armed State | Must exceed threshold to re-arm | Prevents double-hits |
| Cooldown | 500ms between same gesture | Prevents rapid-fire |

---

## v0.1.0 Milestones (COMPLETE)

All 8 milestones from initial build:

1. ✅ Data Model - Vocabulary/Gesture/Example structs, JSON persistence
2. ✅ GUI Shell - eframe/egui window with panel layout
3. ✅ OSC Receiver - Async UDP receiver with status tracking
4. ✅ OSC Sender - Hit message output with test button
5. ✅ DTW Algorithm - Dynamic Time Warping for gesture matching
6. ✅ Recording + Matching - Real-time recognition
7. ✅ Training Session - State machine with audio cues
8. ✅ Polish + Performance Mode - File dialogs, threshold sliders, auto-save

---

## v0.2.0: Working Recognition (COMPLETE)

### What We Built

1. **Simple Wekinator-style recognizer**
   - Fixed window size from first example
   - Compare against all training examples
   - Simple threshold check

2. **Performance optimizations**
   - Frame skipping (4x)
   - Downsampling (4x)
   - Combined: ~64x faster

3. **Double-hit prevention**
   - Armed/disarmed state
   - Must go above threshold to re-arm

4. **Improved UI**
   - Larger window (1400x900)
   - Side-by-side layout in Performance mode
   - Large hit log on right
   - Quick hit indicator (300ms)

### Key Learnings

See `.llm/gesture-recognition-learnings.md` for detailed documentation of:
- What failed and why
- Performance optimization details
- Double-hit prevention mechanism
- Threshold calibration process
- Code architecture

---

## Future Phases

### Phase 2: Feature Engineering
- [ ] Hip-centered normalization (position invariance)
- [ ] Scale normalization (size invariance)
- [ ] Velocity features

### Phase 3: Auto-Calibration
- [ ] Baseline recording (neutral stance)
- [ ] Automatic threshold computation
- [ ] Inter-gesture separation warnings

### Phase 4: Advanced (Stretch)
- [ ] LB_Keogh pruning
- [ ] Subsequence DTW
- [ ] Variable-length gestures

---

## Quick Reference

### Running
```bash
cargo run --release
```

### Testing
```bash
cargo test
```

### Key Files
- `src/engine/recognizer.rs` - The working recognizer
- `src/engine/dtw.rs` - Core DTW algorithm
- `src/gui/mod.rs` - UI with Performance mode
- `.llm/gesture-recognition-learnings.md` - Detailed learnings

### Default Settings
- Window size: From first training example (~180 frames at 60fps)
- Frame skip: 4 (15Hz DTW rate)
- Downsample: 4 (compare ~45 frame sequences)
- Cooldown: 500ms
- Threshold: User-adjustable (start ~8000 for skeleton data)

---

*Last updated: 2026-01-26*
*Status: Working baseline established*
