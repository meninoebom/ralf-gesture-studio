---
title: "Fix Recognition: Accuracy, Timing, and Echo"
type: fix
date: 2026-01-29
---

# Fix Recognition: Accuracy, Timing, and Echo

## Overview

The gesture recognizer has three interrelated problems:
1. **Inaccurate** - fires on wrong gestures or misses correct ones
2. **Laggy/inconsistent timing** - doesn't fire at the "right moment"
3. **Echoey** - fires multiple times for one gesture

These are **solved problems** in speech recognition (VAD, keyword spotting) and gesture toolkits (GRT, Wekinator). This plan applies their proven techniques.

## Research Findings

### What GRT Does (Nick Gillian's Gesture Recognition Toolkit)

1. **Single best template** - Doesn't average examples. Selects the ONE example with lowest average distance to all others.
2. **Statistical threshold** - `threshold = μ + σ × coefficient` where μ/σ are computed from training distances
3. **Stateless prediction** - Each frame is independent. No armed/disarmed complexity.
4. **Null rejection** - If distance > threshold, return "no match" (not the closest gesture)

### What Wekinator Does

1. **Compare to ALL examples** - Returns minimum distance across all training examples
2. **Simple threshold** - User-adjustable, fires when `distance < threshold`
3. **Sliding window** - Maintains circular buffer, generates candidates at different lengths
4. **No echo prevention** - Relies on DTW distance dynamics (sharp valleys are naturally sparse)

### What Speech Recognition Does (CMU Sphinx, Kaldi, WebRTC VAD)

1. **State machine** - Not threshold crossing, but state transitions: Idle → Active → Recovery → Idle
2. **Hysteresis** - Entry threshold (strict) ≠ exit threshold (lenient)
3. **Hangover frames** - After exit, stay in recovery for 50-100ms
4. **Frame accumulation** - Require 3-5 consecutive frames below threshold before firing
5. **Noise floor tracking** - Adapt baseline from resting state, not fixed threshold

## Root Cause Analysis

| Problem | Current Code | Why It Fails |
|---------|-------------|--------------|
| **Inaccuracy** | Adaptive threshold from percentiles | Threshold computed from wrong data (all distances, not training) |
| **Latency** | Fire on first frame below threshold | Noise spikes cause premature firing |
| **Echo** | Cooldown timer only | Cooldown expires while still in gesture zone |

## Solution: State Machine with Hysteresis

Replace the current threshold-crossing logic with a VAD-inspired state machine:

```
                    ┌─────────────┐
                    │    IDLE     │◄──────────────────┐
                    │  (armed)    │                   │
                    └──────┬──────┘                   │
                           │                          │
          distance < threshold_high                   │
          for N consecutive frames                    │
                           │                          │
                           ▼                          │
                    ┌─────────────┐                   │
                    │  BUILDING   │                   │
                    │ (accumulate)│                   │
                    └──────┬──────┘                   │
                           │                          │
          accumulated >= required_frames              │
                           │                          │
                           ▼                          │
              ┌────────────────────────┐              │
              │         PEAK           │              │
              │  *** FIRE GESTURE ***  │              │
              └───────────┬────────────┘              │
                          │                           │
          distance > threshold_low                    │
          (hysteresis exit)                           │
                          │                           │
                          ▼                           │
                   ┌─────────────┐                    │
                   │  RECOVERY   │                    │
                   │ (hangover)  │────────────────────┘
                   └─────────────┘
                   after hangover_ms
```

## Implementation Plan

### Phase 1: Simplify Recognition (Remove Complexity) ✅ COMPLETE

**Goal**: Strip back to Wekinator's proven simple approach before adding improvements.

- [x] Remove `AdaptiveThreshold` struct (was my invention, not standard)
- [x] Remove motion gate (was my invention, causes blocking)
- [x] Remove `armed` state tracking (vestigial, not used)
- [x] Use gesture's trained threshold directly (`μ + σ × coefficient`)
- [x] Keep only: DTW distance + threshold check + cooldown

**Files**: `src/engine/recognizer.rs`, `src/gui/mod.rs`, `src/main.rs`

**Result**: Recognizer simplified from ~500 lines to ~350 lines. Motion gate removed. Tests pass (106/106).

### Phase 2: Add State Machine (Solve Echo + Timing) ✅ COMPLETE

**Goal**: Implement VAD-style state machine with hysteresis.

- [x] Create `RecognitionState` enum: `Idle`, `Building`, `Peak`, `Recovery`
- [x] Add hysteresis thresholds: `threshold_high` (entry), `threshold_low` (exit)
- [x] Implement frame accumulation: require N frames below threshold to enter Peak
- [x] Implement hangover: stay in Recovery for M ms after exit
- [x] Fire gesture ONCE when entering Peak state

**Configuration** (defaults tuned for real-world use):
```rust
pub struct RecognitionConfig {
    pub cooldown_ms: 500,              // Backup protection
    pub threshold_high_factor: 1.0,    // Entry at 100% of threshold
    pub threshold_low_factor: 1.5,     // Exit at 150% (must return to rest)
    pub frames_to_fire: 3,             // ~200ms of confirmation at 15Hz DTW
    pub hangover_ms: 300,              // 300ms recovery period
}
```

**Files**: `src/engine/recognizer.rs`

**Result**: VAD-style state machine implemented. Tests pass (111/111).

### Phase 3: Improve Accuracy (GRT-Style Template Selection)

**Goal**: Use best template instead of comparing to all examples.

- [ ] During training, compute distances between all pairs of examples
- [ ] Select "best template" = example with lowest average distance to others
- [ ] Store `best_template_index` in `Gesture` struct
- [ ] During recognition, compare only to best template (not all examples)
- [ ] Fall back to all-example comparison if only 1-2 examples

**Files**: `src/engine/statistics.rs`, `src/model/vocabulary.rs`, `src/engine/recognizer.rs`

**Test**: More consistent recognition, especially with varied training data

### Phase 4: Diagnostic Enhancement

**Goal**: Log state transitions for debugging.

- [ ] Log state transitions: `IDLE→BUILDING`, `BUILDING→PEAK`, etc.
- [ ] Log reason for each transition
- [ ] Include margin% (how far below/above threshold)
- [ ] Include frame count in each state

**Log format**:
```
timestamp,STATE_CHANGE,from_state,to_state,gesture,distance,threshold,margin%,reason
1234,STATE_CHANGE,Idle,Building,wave,6500,8000,18%,below_high_threshold
1234,STATE_CHANGE,Building,Peak,wave,6200,8000,22%,accumulated_5_frames
1234,HIT,wave,6200,8000,22%
1234,STATE_CHANGE,Peak,Recovery,wave,8500,6400,-32%,above_low_threshold
1234,STATE_CHANGE,Recovery,Idle,wave,9000,8000,-12%,hangover_complete
```

**Files**: `src/engine/diagnostics.rs`, `src/gui/mod.rs`

## Acceptance Criteria

### Accuracy
- [ ] Correct gesture fires 90%+ of attempts
- [ ] Wrong gesture fires <5% of attempts
- [ ] No false positives when standing still

### Timing
- [ ] Hit fires within 100ms of gesture completion
- [ ] Consistent timing across repeated performances
- [ ] No premature firing during gesture buildup

### Echo Prevention
- [ ] Exactly ONE hit per gesture performance
- [ ] No echo even when holding gesture pose
- [ ] Clean re-arm for next gesture

## References

### Code References
- GRT DTW: `nickgillian/grt/blob/master/GRT/ClassificationModules/DTW/DTW.cpp`
- Wekinator DTW: `fiebrink1/wekinator/blob/master/src/wekimini/learning/dtw/DtwModel.java`

### Research
- CMU Sphinx VAD: https://cmusphinx.github.io/wiki/asr/vad/
- Kaldi Online Decoders: https://kaldi-asr.org/doc/online_programs.html
- WebRTC VAD: Frame accumulation + hangover pattern

### Key Algorithms
- **Hysteresis**: Entry threshold ≠ exit threshold (prevents stuck state)
- **Hangover**: Stay silent for N frames after exit (prevents echo)
- **Frame accumulation**: Require M consecutive frames (prevents noise spikes)
- **Best template selection**: Choose most representative example (improves accuracy)
