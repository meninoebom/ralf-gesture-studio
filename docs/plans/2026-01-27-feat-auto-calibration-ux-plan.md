---
title: "feat: Auto-Calibration UX - Make Thresholds Just Work"
type: feat
date: 2026-01-27
---

# Auto-Calibration UX - Make Thresholds Just Work

## Overview

The statistical threshold (μ+σ) approach is already implemented in the backend, but the UI doesn't communicate it well. Users struggle with unusable sliders and don't realize the system can auto-calibrate. The goal: **train gestures → switch to performance → gestures just fire**.

## Problem Statement

1. **Sliders are unusable**: Range `10-500` but real thresholds are often `1000-20000+`
2. **User doesn't trust auto mode**: No visual feedback that calibration happened or what values were computed
3. **Manual tuning is the default UX**: Instead of auto-calibration being the happy path
4. **No coefficient adjustment**: Can't tune sensitivity globally

## Proposed Solution

### Design Principle: Auto-First, Manual-Optional

The UI should:
1. **Celebrate successful calibration** after training ("✓ Calibrated! threshold=8234")
2. **Show computed stats** so user understands the math (μ=6500, σ=867)
3. **Make AUTO the obvious default** with manual as an escape hatch
4. **Fix the slider** to have a sensible range based on computed values

## Technical Approach

### Phase 1: Fix the Slider (Quick Win)

**Problem**: `min="10" max="500"` is wrong for DTW distances.

**Solution**: Dynamic range based on computed threshold or sensible defaults.

#### ui/main.js changes

```javascript
// Current (broken):
<input type="range" min="10" max="500" value="${g.threshold}">

// Fixed: Dynamic range based on computed threshold
const sliderMin = Math.max(100, Math.floor(g.threshold * 0.1));
const sliderMax = Math.max(1000, Math.ceil(g.threshold * 3));

<input type="range"
       min="${sliderMin}"
       max="${sliderMax}"
       value="${Math.round(g.threshold)}"
       step="${Math.max(1, Math.floor((sliderMax - sliderMin) / 100))}">
```

**Acceptance Criteria**:
- [ ] Slider range adapts to current threshold value
- [ ] Slider is easy to move (not stuck at edges)
- [ ] Step size makes fine-tuning possible

---

### Phase 2: Expose Stats in UI (Core Feature)

**Problem**: User can't see μ and σ, so they don't understand the calibration.

**Solution**: Add stats to the DTO and display them.

#### src/gui/mod.rs changes

Add to `GestureMonitorDto`:
```rust
#[derive(Serialize)]
pub struct GestureMonitorDto {
    id: u32,
    name: String,
    example_count: usize,
    distance: Option<f32>,
    threshold: f32,
    auto_mode: bool,
    recent_hit: bool,
    // NEW: Statistics for display
    distance_mean: Option<f32>,
    distance_std: Option<f32>,
    threshold_coefficient: f32,
}
```

Update state building to include stats:
```rust
let (distance_mean, distance_std, threshold_coefficient) = gesture
    .map(|g| (g.distance_mean, g.distance_std, g.threshold_coefficient))
    .unwrap_or((None, None, 2.0));

GestureMonitorDto {
    // ... existing fields ...
    distance_mean,
    distance_std,
    threshold_coefficient,
}
```

#### ui/main.js changes

Display stats when in AUTO mode:
```javascript
// Show μ±σ when auto mode and stats available
const statsDisplay = g.auto_mode && g.distance_mean != null
    ? `<span class="stats-hint dim">μ=${Math.round(g.distance_mean)} σ=${Math.round(g.distance_std)}</span>`
    : '';

row.innerHTML = `
    <span class="gesture-name col-gesture">${g.name} (${g.example_count})</span>
    <span class="distance col-distance ${distanceColor}">${distanceText}</span>
    <span class="threshold-control col-threshold">
        <input type="range" min="${sliderMin}" max="${sliderMax}"
               value="${Math.round(g.threshold)}" class="threshold-slider">
        <span class="threshold-value">${Math.round(g.threshold)}</span>
        ${statsDisplay}
    </span>
    <span class="mode-toggle col-mode ${g.auto_mode ? 'green' : 'dim'}">
        ${g.auto_mode ? 'AUTO' : 'MAN'}
    </span>
    ...
`;
```

**Acceptance Criteria**:
- [ ] μ and σ values display next to threshold when in AUTO mode
- [ ] Stats are hidden when in MAN mode (user overrode, so stats are stale)
- [ ] Stats update after training completes

---

### Phase 3: Training Completion Feedback

**Problem**: After training, user doesn't know calibration happened.

**Solution**: Show calibration success in training completion state.

#### ui/main.js changes

Update the `complete` case in `updateTrainingState`:
```javascript
case 'complete':
    // Get the gesture that was just trained
    const trainedGesture = state.vocabulary?.gestures.find(
        g => g.id === training.gesture_id
    );
    const calibrationInfo = trainedGesture && trainedGesture.distance_mean
        ? `Threshold auto-set to ${Math.round(trainedGesture.threshold)}`
        : 'Calibration pending...';

    elements.trainingDisplay.innerHTML = `
        <span style="font-size: 28px; font-weight: 700; color: var(--green);">✓ COMPLETE!</span>
        <p class="dim">${calibrationInfo}</p>
    `;
    elements.trainingStatus.textContent = 'COMPLETE';
    elements.trainingStatus.className = 'green';
    break;
```

#### src/gui/mod.rs changes

Add `gesture_id` to `TrainingDto`:
```rust
#[derive(Serialize)]
pub struct TrainingDto {
    state: String,
    gesture_id: Option<u32>,  // NEW: Which gesture is being trained
    countdown: u32,
    // ... rest unchanged
}
```

**Acceptance Criteria**:
- [ ] Training completion shows the auto-calibrated threshold
- [ ] User sees confirmation that calibration happened
- [ ] Threshold value is displayed (not just "complete")

---

### Phase 4: Global Coefficient Control (Optional Enhancement)

**Problem**: Default coefficient (2.0) may be too tight or too loose.

**Solution**: Add a single global control to adjust sensitivity.

#### UI Addition

Add to performance panel (below cooldown control):
```html
<div class="row">
    <span class="label">Sensitivity:</span>
    <input type="range" id="threshold-coefficient"
           min="1.0" max="4.0" step="0.5" value="2.0">
    <span id="coefficient-value">2.0</span>
    <span class="dim">(lower = stricter, higher = easier)</span>
</div>
```

**This is optional** - only implement if the μ+σ default doesn't work well in practice.

---

## Implementation Order

1. **Phase 1: Fix Slider** (~15 min) - Immediate usability win
2. **Phase 2: Show Stats** (~30 min) - Core understanding feature
3. **Phase 3: Training Feedback** (~20 min) - Confidence builder
4. **Phase 4: Coefficient** (optional) - Only if needed

Total: ~1 hour for phases 1-3

## Success Metrics

After implementation, the user should be able to:
1. Train 3-5 examples of a gesture
2. Switch to Performance mode
3. Perform the gesture and see it fire **without touching the threshold slider**

The "just works" experience is the goal.

## Files to Modify

| File | Changes |
|------|---------|
| `ui/main.js` | Fix slider range, add stats display, update training complete |
| `src/gui/mod.rs` | Add stats to GestureMonitorDto, add gesture_id to TrainingDto |
| `ui/styles.css` | Add `.stats-hint` style (optional) |

## References

- Statistical threshold (μ+σ) documented in `CLAUDE.md` under "Statistical Threshold"
- Implementation in `src/engine/statistics.rs:62-99`
- GRT reference: https://github.com/nickgillian/grt
- RAW_LEARNINGS.md lines 58-103 for algorithm details
