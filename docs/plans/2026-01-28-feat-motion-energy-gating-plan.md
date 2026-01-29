---
title: "feat: Motion Energy Gating for False Positive Prevention"
type: feat
date: 2026-01-28
---

# Motion Energy Gating for False Positive Prevention

## Overview

Add a motion energy gate that prevents DTW gesture recognition from running when the user is standing still. This eliminates false positives caused by resting state producing low DTW distances that fall below gesture thresholds.

## Problem Statement

**Observed Behavior**: During performance mode testing, the recognizer fires gestures constantly even when the user is standing still:
- 29 "jump" hits in 91 seconds while standing still
- Wave detected only 4 times (should be more)
- Spin detected 6 times spuriously

**Root Cause Analysis** (from diagnostic logs):

| State | Distance | Threshold | Result |
|-------|----------|-----------|--------|
| Resting (still) | 17-23 | jump: 37 | **Below threshold** - fires |
| Resting (still) | 17-23 | wave: 98 | Below threshold but not armed |
| Resting (still) | 23-28 | spin: 47 | **Below threshold** - fires |

The statistical threshold (μ+σ) approach assumes **gestures produce lower distances than resting**. But with the current data:
- Resting produces distances **below** all thresholds
- The system can't distinguish "standing still" from "doing a gesture"
- Re-arming never occurs because distance never rises above threshold

**Why This Happens**: The trained gesture examples likely contain frames where the user is in similar positions to the resting state (start/end of gesture, or gesture uses similar joint configurations).

## Proposed Solution

**Motion Energy Gating**: Only run DTW recognition when the user is actually moving.

```
Current:  Frame → DTW → Threshold → Fire/Don't
Proposed: Frame → Motion Check → [Skip if still] → DTW → Fire
```

This is the proven approach used by:
- **GRT (Gesture Recognition Toolkit)** - SwipeDetector uses `movementVelocity` gating
- **Voice Activity Detection (VAD)** - Equivalent pattern in speech recognition
- **Academic literature** - Standard practice for continuous gesture recognition

**Key Insight**: Standing still = no motion energy = skip DTW entirely. The resting state never gets compared because the gate prevents it.

## Technical Approach

### Core Algorithm

```rust
// Motion energy = sum of squared frame-to-frame differences
fn motion_energy(prev: &Frame, curr: &Frame) -> f32 {
    prev.iter().zip(curr.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum()
}

// In process_frame(), early return if not moving:
let energy = average_motion_energy(&self.recent_frames(5));
if energy < self.config.motion_threshold {
    return RecognitionResult::Idle;  // Skip DTW
}
// Continue with normal DTW recognition...
```

### Architecture Changes

**New State in Recognizer** (`src/engine/recognizer.rs`):

```rust
pub struct RecognitionConfig {
    // Existing fields...
    pub motion_threshold: f32,      // Min energy to run DTW (default: auto-calibrated)
    pub motion_window: usize,       // Frames for energy average (default: 5)
    pub motion_gate_enabled: bool,  // Allow disable for debugging
}

pub struct Recognizer {
    // Existing fields...
    current_motion_energy: f32,     // For UI display
    motion_gate_active: bool,       // Is gate currently blocking DTW?
    calibration_state: CalibrationState,
}

enum CalibrationState {
    Uncalibrated,
    Learning { samples: Vec<f32> },
    Calibrated { mean: f32, std: f32, threshold: f32 },
}
```

**Insertion Point**: After `self.buffer.push(frame)` (line ~297), before DTW computation (line ~320).

### Auto-Calibration

On app start, learn the "noise floor" from initial stillness:

1. Collect motion energy for 60 frames (~1 second)
2. Compute mean (μ) and standard deviation (σ)
3. Set threshold = μ + 3σ (conservative to avoid blocking real gestures)
4. Calibration runs in background (non-blocking)

```rust
impl Recognizer {
    fn update_calibration(&mut self, energy: f32) {
        match &mut self.calibration_state {
            CalibrationState::Learning { samples } => {
                samples.push(energy);
                if samples.len() >= 60 {
                    let mean = samples.iter().sum::<f32>() / samples.len() as f32;
                    let variance = samples.iter()
                        .map(|e| (e - mean).powi(2))
                        .sum::<f32>() / samples.len() as f32;
                    let std = variance.sqrt();
                    let threshold = mean + std * 3.0;

                    self.calibration_state = CalibrationState::Calibrated {
                        mean, std, threshold
                    };
                    self.config.motion_threshold = threshold;
                }
            }
            _ => {}
        }
    }
}
```

### Hysteresis (Prevents Toggling)

Use 20% hysteresis to prevent rapid gate switching:

```rust
// Gate turns ON when energy < threshold * 0.8
// Gate turns OFF when energy > threshold * 1.0
let on_threshold = self.config.motion_threshold * 0.8;
let off_threshold = self.config.motion_threshold;

if self.motion_gate_active {
    if energy > off_threshold {
        self.motion_gate_active = false;
        self.clear_peak_detection_state();  // Fresh start
    }
} else {
    if energy < on_threshold {
        self.motion_gate_active = true;
    }
}
```

### Training Mode Integration

Motion gate is **completely disabled** during training to ensure all frames are captured:

```rust
// In process_frame()
if self.training_mode_active {
    // Skip motion gate entirely during training
} else {
    // Normal motion gate logic
}
```

## Implementation Phases

### Phase 1: Core Motion Gate

**Files to modify:**

| File | Changes |
|------|---------|
| `src/engine/recognizer.rs` | Add motion gate fields, implement gate logic |
| `src/engine/dtw.rs` | Already has `motion_energy()` - just use it |
| `src/engine/mod.rs` | Export new types if needed |

**Tasks:**

- [x] Add `motion_threshold`, `motion_window`, `motion_gate_enabled` to `RecognitionConfig`
- [x] Add `current_motion_energy`, `motion_gate_active` to `Recognizer`
- [x] Implement motion gate check in `process_frame()` (early return if gated)
- [x] Add hysteresis logic (20% band)
- [x] Clear peak detection state when gate transitions off
- [x] Disable motion gate during training sessions
- [x] Add unit tests for motion gate logic

**Acceptance Criteria:**
- [x] Standing still produces no gesture hits
- [x] Moving triggers DTW recognition
- [x] Gate transitions smoothly (no rapid toggling)
- [x] Training mode captures all frames regardless of motion

### Phase 2: Auto-Calibration

**Files to modify:**

| File | Changes |
|------|---------|
| `src/engine/recognizer.rs` | Add `CalibrationState`, implement learning |

**Tasks:**

- [x] Add `CalibrationState` enum (Uncalibrated, Learning, Calibrated)
- [x] Collect motion energy samples during first 60 frames
- [x] Compute mean, std, threshold (μ + 3σ)
- [x] Fall back to reasonable default if calibration fails
- [x] Add `recalibrate()` method for manual trigger
- [x] Add tests for calibration state machine

**Acceptance Criteria:**
- [x] App starts with calibration in "Learning" state
- [x] After ~1 second, threshold is auto-set
- [x] Manual recalibration works
- [x] Reasonable default if no calibration data

### Phase 3: UI Integration

**Files to modify:**

| File | Changes |
|------|---------|
| `src/gui/mod.rs` | Add motion energy to MonitorDto, add controls |

**Tasks:**

- [x] Add `motion_energy` and `motion_gate_active` to `MonitorDto`
- [ ] Display motion energy bar in Performance mode (visual feedback)
- [ ] Show "STILL" indicator when gate is active
- [x] Add toggle to enable/disable motion gate (for debugging)
- [x] Add "Recalibrate" button to settings
- [x] Show calibration state (Learning/Ready)

**Acceptance Criteria:**
- [x] Users can see current motion level
- [x] Users know when motion gate is blocking recognition
- [x] Users can disable motion gate for debugging
- [x] Users can trigger recalibration

### Phase 4: Diagnostic Logging

**Files to modify:**

| File | Changes |
|------|---------|
| `src/gui/mod.rs` | Add motion gate events to diagnostic log |

**Tasks:**

- [ ] Log motion gate state changes (activated/deactivated)
- [ ] Log current motion energy with each REC line
- [ ] Log calibration completion with threshold value
- [ ] Add "GATE" event type to diagnostic format

**Log format addition:**

```
# GATE: Motion gate state change
timestamp,GATE,frame,state,energy,threshold
1234,GATE,8500,activated,0.0012,0.002
1235,GATE,8520,deactivated,0.0025,0.002
```

## Design Decisions

### Why Motion Energy, Not Activity Detection?

The previous v0.1 attempt at "activity gating" failed because:
- It added a separate threshold to tune
- The main DTW threshold was handling idle state naturally

**Current situation is different:**
- Resting distances are BELOW thresholds (not above)
- Main threshold can't distinguish rest from gesture
- Motion energy is orthogonal to DTW distance

### Why 5-Frame Window?

- Single frame: Too noisy (tracking jitter)
- 3 frames: Still somewhat noisy
- **5 frames: Good balance** (matches `ACTIVITY_FRAMES` in design doc)
- 10+ frames: Adds latency, slower response

### Why 3σ for Calibration Coefficient?

- 2σ (as used in DTW threshold): Might be too tight, could gate real gestures
- **3σ: Conservative** - allows more recognition, errs on side of sensitivity
- 4σ+: Too loose, wouldn't filter much

### Why Hysteresis?

Without hysteresis, motion near threshold causes rapid toggling:
- Frame N: energy = 0.0019, threshold = 0.002 → gate ON
- Frame N+1: energy = 0.0021 → gate OFF
- Frame N+2: energy = 0.0019 → gate ON

20% hysteresis band prevents this:
- Gate turns ON below 80% of threshold
- Gate turns OFF above 100% of threshold
- Stable in the 80-100% range

### Out of Scope (v1)

These are explicitly NOT addressed in this implementation:

| Feature | Reason |
|---------|--------|
| **Pose recognition** | Requires motion during performance; static poses need different approach |
| **Per-vocabulary thresholds** | Start simple with global; add later if needed |
| **Joint weighting** | Use all joints equally; optimize later if needed |
| **Slow gesture support** | Lower threshold should handle most cases; tune coefficient if issues |

## Success Metrics

| Metric | Before | Target |
|--------|--------|--------|
| False positives while standing | 29 in 91s | <2 in 91s |
| Gesture recognition when moving | ~30% | >90% |
| User-perceived latency | N/A | <100ms added |
| Configuration required | Manual threshold per gesture | Auto-calibrated |

## Dependencies & Risks

**Dependencies:**
- Existing `motion_energy()` function in `dtw.rs` (already implemented, unused)
- `RecognitionConfig` pattern for adding parameters
- `MonitorDto` pattern for UI feedback

**Risks:**

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Gate too aggressive (blocks real gestures) | Medium | Use conservative 3σ coefficient; add disable toggle |
| Gate too loose (false positives persist) | Low | Calibration learns actual noise floor |
| Adds latency | Low | Motion energy is O(n) per frame, very cheap |
| Calibration requires standing still | Medium | Clear UI prompt; reasonable default fallback |

## References & Research

### Internal References
- Existing motion energy functions: `src/engine/dtw.rs:326-377`
- Recognition config pattern: `src/engine/recognizer.rs:16-35`
- Diagnostic logging: `src/gui/mod.rs:260-321`
- Design doc (activity gating): `.llm/gesture-recognition-design.md:226-248`
- Previous attempt learnings: `.llm/gesture-recognition-learnings.md:73-76`

### External References
- GRT SwipeDetector (motion gating): `github.com/nickgillian/grt/GRT/ClassificationModules/SwipeDetector/`
- Wekinator DTW: `github.com/fiebrink1/wekinator/src/wekimini/learning/dtw/DtwModel.java`
- Voice Activity Detection pattern: Standard speech recognition technique

### Diagnostic Log Analyzed
- File: `/Users/brandon/Documents/RALF/ralf-diagnostics-2026-01-28_12-01-14.log`
- Duration: 91 seconds
- Key finding: Resting distances (17-23) below all gesture thresholds (37-98)
