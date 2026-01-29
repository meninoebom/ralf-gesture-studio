---
title: "Recognition Optimization v2 (Reviewed)"
type: feat
date: 2026-01-30
---

# Recognition Optimization v2 (Expert-Reviewed)

Updated plan incorporating feedback from external review. Original research validated; critical risks identified and mitigated.

## Review Summary

**Verdict**: "You're not flailing — you're converging. This is approaching professional-grade gesture recognition."

### Validations ✅
- VAD-style state machine is correct
- Sakoe-Chiba is "mandatory, ship first"
- Dual-threshold hysteresis is "exactly how it's done in professional systems"
- LB_Keogh is valuable (but see caveats)

### Critical Risks Identified ⚠️

| Risk | Severity | Mitigation |
|------|----------|------------|
| **Spatial resampling destroys velocity** | 🔴 Critical | Use temporal downsampling only (already doing this!) |
| LB_Keogh less effective in high-D | 🟡 Medium | Expect 50-70% pruning, not 90% |
| Resampling + Sakoe-Chiba interaction | 🟡 Medium | Tune together, not independently |
| μ+σ assumes unimodal distribution | 🟢 Low | Monitor as dancer diversity increases |

### Missing Optimizations Added
1. **Early abandoning inside DTW** - "2 lines of code, 50-70% speedup"
2. **Distance slope check** - Verify distance is *falling*, not just below threshold
3. **Velocity features** - Future accuracy lever (not in this phase)

---

## Critical Finding: Do NOT Use Spatial Resampling

### The Problem

The original plan proposed **spatial resampling** from the $1 Recognizer:
- Interpolate frames based on *distance traveled*
- Make points equidistant in space

### Why This Breaks Dance

> "In the $1 Recognizer (pen strokes), drawing a 'V' slowly or quickly should trigger the same result. In Dance, a **Slow Float** and a **Fast Strike** might be the same spatial shape but are completely different semantic gestures."

**Spatial resampling destroys velocity information.**

| Input | Spatial Resampling Output |
|-------|---------------------------|
| Slow arm raise (2 seconds) | 45 equidistant points |
| Fast arm raise (0.5 seconds) | 45 equidistant points |
| **Result** | IDENTICAL sequences! |

For dancers who differentiate dynamics and intensity, this erases critical information.

### The Solution: Temporal Downsampling (Already Implemented!)

We're already doing the right thing:
```rust
// Current approach: fixed-rate temporal downsampling
downsample_factor: 4,  // 60fps → 15fps
```

This **preserves velocity** because fast movements cover more space between frames.

**Action**: Remove spatial resampling from the plan. Keep temporal downsampling.

---

## Revised Priority Order

Based on expert review, re-ordered for maximum impact with minimum risk:

| Phase | Optimization | Impact | Risk | Effort |
|-------|--------------|--------|------|--------|
| **1a** | Dual-threshold echo guard | High | None | Low |
| **1b** | Sakoe-Chiba band | High | None | Low |
| **1c** | Early abandoning (inside DTW) | Medium | None | Trivial |
| **2** | Distance slope check | Medium | None | Low |
| **3** | LB_Keogh pruning | Medium | Low | Medium |
| ~~4~~ | ~~Spatial resampling~~ | ~~-~~ | ~~High~~ | ~~Removed~~ |
| **4** | Latency tuning (optional) | Low | Medium | Low |

---

## Phase 1: No-Brainers (Ship Immediately)

### 1a. Dual-Threshold Echo Guard

**Goal**: Eliminate the 5 echoes from stress testing.

**Diagnosis**: Distance hovers near threshold → repeated re-entry (Schmitt trigger problem)

**Solution**: Hysteresis with re-arm threshold

```rust
pub struct RecognitionConfig {
    // ... existing ...
    pub rearm_threshold_factor: f32,  // 1.2-1.5 (must exceed to re-arm)
    pub extended_hangover_ms: u64,    // Fallback if distance never exceeds rearm
}

// State machine update
RecognitionState::Recovery { start_time, max_distance_seen } => {
    let elapsed = start_time.elapsed().as_millis() as u64;
    let rearm_threshold = gesture.threshold * config.rearm_threshold_factor;

    // Track max distance seen during recovery
    let new_max = max_distance_seen.max(current_distance);

    // Re-arm conditions:
    // 1. Distance clearly above threshold + standard hangover
    // 2. Extended hangover (regardless of distance)
    let can_rearm = if new_max > rearm_threshold {
        elapsed >= config.hangover_ms
    } else {
        elapsed >= config.extended_hangover_ms
    };

    if can_rearm {
        RecognitionState::Idle
    } else {
        RecognitionState::Recovery { start_time, max_distance_seen: new_max }
    }
}
```

**Recommended values**:
- `rearm_threshold_factor`: 1.3 (30% above threshold)
- `extended_hangover_ms`: 500ms

**Reviewer note**: "You might need it as high as 1.5 or 2.0 depending on resting similarity."

**Files**: `src/engine/recognizer.rs`

### 1b. Sakoe-Chiba Band Constraint

**Goal**: Faster DTW + prevent pathological warping

**Why it's safe**: "One of those rare 'faster and better' constraints"

```rust
// src/engine/dtw.rs
pub fn dtw_distance_sakoe_chiba(
    seq1: &Sequence,
    seq2: &Sequence,
    band_fraction: f32
) -> f32 {
    let m = seq1.len();
    let n = seq2.len();
    let band_width = ((m.max(n) as f32) * band_fraction).ceil() as usize;

    let mut dtw = vec![vec![f32::INFINITY; n + 1]; m + 1];
    dtw[0][0] = 0.0;

    for i in 1..=m {
        let j_start = (i as isize - band_width as isize).max(1) as usize;
        let j_end = (i + band_width).min(n);

        for j in j_start..=j_end {
            let cost = euclidean_distance(&seq1[i-1], &seq2[j-1]);
            dtw[i][j] = cost + dtw[i-1][j]
                .min(dtw[i][j-1])
                .min(dtw[i-1][j-1]);
        }
    }
    dtw[m][n]
}
```

**Recommended band**: 15% (`band_fraction = 0.15`)

**Reviewer note**: "Matches how dancers actually move (tempo varies, structure doesn't)"

**Files**: `src/engine/dtw.rs`, `src/engine/recognizer.rs`

### 1c. Early Abandoning Inside DTW

**Goal**: Stop unpromising DTW computations mid-calculation

**Why**: "2 lines of code, 50-70% speedup" - stacks with LB_Keogh

```rust
pub fn dtw_distance_with_abandon(
    seq1: &Sequence,
    seq2: &Sequence,
    band_fraction: f32,
    best_so_far: f32,  // Current best distance found
) -> Option<f32> {
    let m = seq1.len();
    let n = seq2.len();
    let band_width = ((m.max(n) as f32) * band_fraction).ceil() as usize;

    let mut prev_row = vec![f32::INFINITY; n + 1];
    let mut curr_row = vec![f32::INFINITY; n + 1];
    prev_row[0] = 0.0;

    for i in 1..=m {
        curr_row.fill(f32::INFINITY);
        let j_start = (i as isize - band_width as isize).max(1) as usize;
        let j_end = (i + band_width).min(n);

        let mut row_min = f32::INFINITY;  // Track minimum in this row

        for j in j_start..=j_end {
            let cost = euclidean_distance(&seq1[i-1], &seq2[j-1]);
            curr_row[j] = cost + prev_row[j]
                .min(curr_row[j-1])
                .min(prev_row[j-1]);
            row_min = row_min.min(curr_row[j]);
        }

        // EARLY ABANDON: if minimum possible path exceeds best, stop
        if row_min > best_so_far {
            return None;
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    Some(prev_row[n])
}
```

**Usage**:
```rust
let mut best_dist = f32::INFINITY;
for example in gesture.examples() {
    if let Some(dist) = dtw_distance_with_abandon(&window, example, 0.15, best_dist) {
        if dist < best_dist {
            best_dist = dist;
        }
    }
    // If None returned, this example was abandoned (worse than current best)
}
```

**Files**: `src/engine/dtw.rs`, `src/engine/recognizer.rs`

---

## Phase 2: Distance Slope Check

**Goal**: Reduce false triggers by verifying distance is *falling*, not just below threshold

**Insight**: "Echoes often occur at flat minima. Real gestures show steep descent. Noise does not."

```rust
pub struct GestureStateData {
    // ... existing ...
    pub distance_history: VecDeque<f32>,  // Last 3-4 distances
}

impl GestureStateData {
    /// Check if distance is falling (negative slope)
    fn is_distance_falling(&self, current: f32) -> bool {
        if self.distance_history.len() < 2 {
            return true;  // Not enough history, allow
        }

        let prev = self.distance_history.back().unwrap();
        let slope = current - prev;

        // Require negative slope (falling) or very small positive (flat minimum)
        slope < 0.05 * current  // Allow 5% tolerance
    }
}

// In state machine, Building state entry:
if distance < threshold && self.is_distance_falling(distance) {
    // Enter Building
} else {
    // Stay in Idle - distance below threshold but not falling (noise/echo)
}
```

**Why this helps**: "Peak detectors avoid chatter this way. Very cheap. Very effective."

**Files**: `src/engine/recognizer.rs`

---

## Phase 3: LB_Keogh Pruning (With Caveats)

**Goal**: Prune DTW computations before they start

**Caveat from review**:
> "33 joints × 3 dimensions = 99 dimensions. Envelope looseness increases with dimensionality. You may not get 90% pruning — might get 50-70%. Still very good."

**Implementation** (same as original plan):

```rust
pub struct LBEnvelope {
    pub upper: Sequence,
    pub lower: Sequence,
}

pub fn compute_lb_envelope(sequence: &Sequence, band_width: usize) -> LBEnvelope {
    // ... implementation from original plan ...
}

pub fn lb_keogh(candidate: &Sequence, envelope: &LBEnvelope) -> f32 {
    // ... implementation from original plan ...
}
```

**Usage with early abandon**:
```rust
fn find_best_match(&self, window: &Sequence) -> Option<(GestureId, f32)> {
    let mut best_dist = f32::INFINITY;
    let mut best_gesture = None;

    for gesture in &self.gestures {
        for example in gesture.examples() {
            // Layer 1: LB_Keogh prune
            if let Some(ref envelope) = example.lb_envelope {
                let lb = lb_keogh(window, envelope);
                if lb >= best_dist {
                    continue;  // Prune!
                }
            }

            // Layer 2: DTW with early abandon
            if let Some(dist) = dtw_distance_with_abandon(
                window, &example.frames, 0.15, best_dist
            ) {
                if dist < best_dist {
                    best_dist = dist;
                    best_gesture = Some(gesture.id);
                }
            }
        }
    }

    best_gesture.map(|id| (id, best_dist))
}
```

**Future optimization** (noted for later):
> "Apply LB_Keogh on reduced features, not raw joints: joint velocities, torso-relative positions, or PCA-reduced pose vectors."

**Files**: `src/engine/dtw.rs`, `src/model/vocabulary.rs`, `src/engine/recognizer.rs`

---

## Phase 4: Latency Tuning (Optional, Careful)

**Original proposal**: Reduce `frames_to_fire` from 3 to 2

**Reviewer feedback**: Mixed
- "Reasonable once pruning is active"
- "Skeleton tracking is jittery. Stick to 3 frames."

**Consensus**: Gain latency budget through faster processing (Sakoe-Chiba, early abandon), NOT fewer confirmation frames.

**Alternative approach**:
```rust
// Instead of reducing frames_to_fire, increase DTW rate
RecognitionConfig {
    dtw_skip: 2,          // 30Hz instead of 15Hz (was 4)
    frames_to_fire: 3,    // Keep at 3 for jitter protection
    // ...
}
```

This gives ~100ms latency (3 frames × 33ms) while maintaining jitter protection.

**Recommendation**: Defer until after Phase 1-3 are validated. May not be needed.

---

## Removed: Spatial Resampling

**Original proposal**: Resample sequences to fixed length using path-distance interpolation ($1 Recognizer style)

**Why removed**:
> "Spatial resampling destroys velocity information. A slow wave and fast wave will generate identical sequences. If your dancers need to differentiate dynamics or intensity, this optimization will erase that data."

**What we keep**: Temporal downsampling (60fps → 15fps), which preserves velocity.

---

## Future Optimizations (Not This Phase)

### Velocity Features
> "Dance is defined by movement, not position. Even concatenating [x, y, z, dx, dy, dz] per joint improves separability. This is probably your biggest future accuracy lever."

### Reduced Feature Space for LB_Keogh
> "Apply LB_Keogh on reduced features: joint velocities, torso-relative positions, or PCA-reduced pose vectors. This dramatically improves bound tightness."

### Per-Gesture Threshold Coefficients
> "Different gestures may need different coefficients. Simple consistent gestures: 2.0. Complex variable gestures: 3.0-4.0."

---

## Implementation Checklist

### Phase 1a: Echo Guard
- [ ] Add `rearm_threshold_factor` to `RecognitionConfig` (default: 1.3)
- [ ] Add `extended_hangover_ms` to `RecognitionConfig` (default: 500)
- [ ] Track `max_distance_seen` in Recovery state
- [ ] Implement dual-path re-arming logic
- [ ] Update diagnostic logging for new re-arm conditions

### Phase 1b: Sakoe-Chiba
- [ ] Add `dtw_distance_sakoe_chiba()` function
- [ ] Add `sakoe_chiba_band` to config (default: 0.15)
- [ ] Update recognizer to use constrained DTW
- [ ] Verify tests pass with new DTW

### Phase 1c: Early Abandoning
- [ ] Add `dtw_distance_with_abandon()` function
- [ ] Update recognizer to pass `best_so_far` to DTW
- [ ] Verify performance improvement

### Phase 2: Distance Slope
- [ ] Add `distance_history` to `GestureStateData`
- [ ] Add `is_distance_falling()` method
- [ ] Gate Building state entry on falling distance
- [ ] Add diagnostic logging for slope rejections

### Phase 3: LB_Keogh
- [ ] Add `LBEnvelope` struct
- [ ] Add `compute_lb_envelope()` function
- [ ] Add `lb_keogh()` function
- [ ] Store envelopes in `Example` struct
- [ ] Precompute envelopes after training
- [ ] Add pruning to recognition loop
- [ ] Measure actual pruning percentage

### Testing
- [ ] A/B test: baseline vs Phase 1 optimizations
- [ ] Stress test: count echoes (target: 0)
- [ ] Latency measurement
- [ ] CPU usage measurement

---

## Expected Outcomes

| Metric | Current | After Phase 1 | After All |
|--------|---------|---------------|-----------|
| Echo rate | 5 in stress test | 0-1 | 0 |
| DTW speed | O(N²) | O(N×B) | O(N×B) with 50-70% pruning |
| Latency | ~200ms | ~200ms | ~150ms (optional) |
| Accuracy | 100% | 100% | 100% |

---

## References

### Review Sources
- Expert review from Claude (Anthropic) - signal processing perspective
- Expert review from GPT-4 - implementation risk assessment

### Research Sources
- [UCR Suite](https://www.cs.ucr.edu/~eamonn/UCRsuite.html) - DTW optimization
- [FastDTW Critique (Wu & Keogh, 2020)](https://arxiv.org/abs/2003.11246)
- [Sakoe-Chiba Band Impact](https://www.researchgate.net/publication/301952855)
- [$1 Recognizer](https://depts.washington.edu/acelab/proj/dollar/index.html)
- WebRTC VAD - hangover/hysteresis patterns
