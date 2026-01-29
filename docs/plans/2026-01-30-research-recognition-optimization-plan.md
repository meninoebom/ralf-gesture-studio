---
title: "Research: Gesture Recognition Optimization"
type: research
date: 2026-01-30
---

# Research: Gesture Recognition Optimization

Deep research on optimizing DTW-based gesture recognition for **accuracy**, **timeliness**, and **echo mitigation**.

## Executive Summary

The current VAD-style state machine is well-aligned with proven patterns. Research identifies **three high-impact optimizations** and several refinements that can further improve the three goals.

### Current Status (Baseline)

| Goal | Status | Metric |
|------|--------|--------|
| **Accuracy** | ✅ Strong | 100% (7/7 correct, 0 false positives) |
| **Timeliness** | ⚠️ Good | ~200ms latency |
| **Echo Prevention** | ✅ Strong | 1 hit per gesture (5 echoes in stress test) |

### Recommended Optimizations

| Priority | Optimization | Impact | Effort |
|----------|--------------|--------|--------|
| **1** | Sakoe-Chiba band constraint | Faster DTW + better accuracy | Low |
| **2** | LB_Keogh lower bound | 90% DTW computation pruning | Medium |
| **3** | Sequence resampling | More consistent matching | Low |
| **4** | Dual threshold echo guard | Eliminate remaining echoes | Low |
| **5** | Reduce latency parameters | <150ms response | Low |

---

## Part 1: What We Learned From Research

### 1.1 Our Current Approach is Validated

The research confirms our VAD-style state machine matches proven patterns from:
- **WebRTC VAD**: hangover frames + minimum duration before firing
- **GRT (Gesture Recognition Toolkit)**: μ+σ threshold, best template selection
- **Wekinator**: Compare to all examples, sliding window

**Key validation**: Our "All Examples" approach (Wekinator-style) beating "Best Template" (GRT-style) is explained by body tracking having more variation than audio gestures.

### 1.2 FastDTW is NOT Recommended

**Surprising finding** from 2020 research: FastDTW is actually **slower** than constrained DTW for sequences under 900 points.

> "The vast majority of researchers that used FastDTW would have been better off simply using constrained DTW."
> — [Wu & Keogh, 2020](https://arxiv.org/abs/2003.11246)

**Recommendation**: Use Sakoe-Chiba constrained DTW instead.

### 1.3 What the Best Systems Do

| System | Threshold | Echo Prevention | Key Technique |
|--------|-----------|-----------------|---------------|
| **GRT** | μ + 3σ (default) | Class label filter | Best template selection |
| **Wekinator** | User threshold | Cooldown | All examples, min distance |
| **WebRTC VAD** | Adaptive | Hangover frames (8) | Min duration (3 frames) |
| **$1 Recognizer** | Fixed | N/A (discrete) | Preprocessing pipeline |

### 1.4 Critical Insight: Preprocessing Matters

The $1 Recognizer achieves 97%+ accuracy with just 1 template through **preprocessing**:
1. Resample to fixed N points (64)
2. Center on first point (position invariance)
3. Scale to unit size (optional)

**For body tracking**: Resampling to fixed length before DTW improves consistency.

---

## Part 2: Optimization Recommendations

### 2.1 Sakoe-Chiba Band Constraint (HIGH PRIORITY)

**What**: Constrain DTW warping path to stay within a band around the diagonal.

**Why**:
- Prevents "pathological warping" (unrealistic time stretching)
- **Often improves accuracy** by preventing overfitting
- Reduces computation from O(N²) to O(N × B) where B = band width

**Implementation**:

```rust
// src/engine/dtw.rs - Add constrained version
pub fn dtw_distance_sakoe_chiba(seq1: &Sequence, seq2: &Sequence, band_percent: f32) -> f32 {
    let m = seq1.len();
    let n = seq2.len();
    let band_width = ((m.max(n) as f32) * band_percent) as usize;

    let mut dtw = vec![vec![f32::INFINITY; n + 1]; m + 1];
    dtw[0][0] = 0.0;

    for i in 1..=m {
        let j_start = (i as isize - band_width as isize).max(1) as usize;
        let j_end = (i + band_width).min(n);

        for j in j_start..=j_end {
            let cost = euclidean_distance(&seq1[i-1], &seq2[j-1]);
            dtw[i][j] = cost + dtw[i-1][j].min(dtw[i][j-1]).min(dtw[i-1][j-1]);
        }
    }
    dtw[m][n]
}
```

**Recommended band**: 10-20% of sequence length (e.g., `band_percent = 0.15`)

**Configuration**:
```rust
pub struct RecognitionConfig {
    // ... existing fields ...
    pub sakoe_chiba_band: f32,  // 0.15 = 15% of sequence length
}
```

### 2.2 LB_Keogh Lower Bound (HIGH PRIORITY)

**What**: O(n) lower bound that prunes candidates before expensive DTW.

**Why**: UCR Suite research shows **90%+ DTW computations can be pruned**.

**How it works**:
1. Create envelope {Upper, Lower} around each training example
2. Compute cheap Euclidean distance from candidate to envelope
3. If LB > best_so_far, skip DTW entirely

**Implementation**:

```rust
// src/engine/dtw.rs - Add lower bound computation
pub struct LBEnvelope {
    pub upper: Sequence,
    pub lower: Sequence,
}

pub fn compute_lb_envelope(sequence: &Sequence, band_width: usize) -> LBEnvelope {
    let mut upper = Vec::with_capacity(sequence.len());
    let mut lower = Vec::with_capacity(sequence.len());

    for i in 0..sequence.len() {
        let start = i.saturating_sub(band_width);
        let end = (i + band_width + 1).min(sequence.len());

        let mut u = sequence[start].clone();
        let mut l = sequence[start].clone();

        for j in start..end {
            for k in 0..sequence[j].len() {
                u[k] = u[k].max(sequence[j][k]);
                l[k] = l[k].min(sequence[j][k]);
            }
        }
        upper.push(u);
        lower.push(l);
    }
    LBEnvelope { upper, lower }
}

pub fn lb_keogh(candidate: &Sequence, envelope: &LBEnvelope) -> f32 {
    let mut lb = 0.0;
    let len = candidate.len().min(envelope.upper.len());

    for i in 0..len {
        for k in 0..candidate[i].len() {
            if candidate[i][k] > envelope.upper[i][k] {
                lb += (candidate[i][k] - envelope.upper[i][k]).powi(2);
            } else if candidate[i][k] < envelope.lower[i][k] {
                lb += (envelope.lower[i][k] - candidate[i][k]).powi(2);
            }
        }
    }
    lb.sqrt()
}
```

**Usage pattern**:
```rust
// In recognizer, precompute envelopes for training examples
fn precompute_envelopes(&mut self) {
    for gesture in &mut self.gestures {
        for example in gesture.examples_mut() {
            example.lb_envelope = Some(compute_lb_envelope(&example.frames, band_width));
        }
    }
}

// During recognition, prune before DTW
fn find_best_match(&self, window: &Sequence) -> Option<(GestureId, f32)> {
    let mut best_dist = f32::INFINITY;
    let mut best_gesture = None;

    for gesture in &self.gestures {
        for example in gesture.examples() {
            // Early prune with LB_Keogh
            if let Some(ref envelope) = example.lb_envelope {
                let lb = lb_keogh(window, envelope);
                if lb >= best_dist {
                    continue;  // Skip DTW!
                }
            }

            let dist = dtw_distance_sakoe_chiba(window, &example.frames, 0.15);
            if dist < best_dist {
                best_dist = dist;
                best_gesture = Some(gesture.id);
            }
        }
    }
    best_gesture.map(|id| (id, best_dist))
}
```

### 2.3 Sequence Resampling (MEDIUM PRIORITY)

**What**: Resample all sequences to a fixed number of frames before comparison.

**Why**:
- More consistent DTW computation
- Removes dependency on performance speed variation
- $1 Recognizer achieves 97% accuracy with this technique

**Implementation**:

```rust
// src/engine/dtw.rs - Add resampling
pub fn resample_sequence(sequence: &Sequence, target_len: usize) -> Sequence {
    if sequence.len() == target_len {
        return sequence.clone();
    }

    let path_length = compute_path_length(sequence);
    let ideal_spacing = path_length / (target_len - 1) as f32;

    let mut result = Vec::with_capacity(target_len);
    result.push(sequence[0].clone());

    let mut d = 0.0;
    let mut i = 1;

    while result.len() < target_len && i < sequence.len() {
        let segment_len = euclidean_distance(&sequence[i-1], &sequence[i]);

        if d + segment_len >= ideal_spacing {
            let t = (ideal_spacing - d) / segment_len;
            let interpolated = interpolate_frames(&sequence[i-1], &sequence[i], t);
            result.push(interpolated);
            d = segment_len - (ideal_spacing - d);
        } else {
            d += segment_len;
            i += 1;
        }
    }

    // Pad with last frame if needed
    while result.len() < target_len {
        result.push(sequence.last().unwrap().clone());
    }

    result
}

fn compute_path_length(sequence: &Sequence) -> f32 {
    sequence.windows(2)
        .map(|w| euclidean_distance(&w[0], &w[1]))
        .sum()
}

fn interpolate_frames(a: &Frame, b: &Frame, t: f32) -> Frame {
    a.iter().zip(b.iter())
        .map(|(va, vb)| va + t * (vb - va))
        .collect()
}
```

**Recommended target length**: 45-60 frames (matches current downsampled size)

### 2.4 Dual Threshold Echo Guard (MEDIUM PRIORITY)

**What**: Add a second check to prevent near-threshold echoes.

**Why**: A/B test showed 5 echoes with margins of 1-2%. These are borderline cases where the gesture distance hovers near threshold.

**Current behavior**:
- Distance drops below threshold → Building
- 3 frames → Fire
- Distance rises → Recovery (300ms)
- Distance drops again quickly → **ECHO** (if 300ms elapsed)

**Enhanced behavior**:
- After firing, require distance to exceed **re-arm threshold** (e.g., 1.2× threshold)
- OR wait for **extended hangover** (e.g., 500ms instead of 300ms)

**Implementation**:

```rust
pub struct RecognitionConfig {
    // ... existing ...
    pub rearm_threshold_factor: f32,  // 1.2 = must go 20% above threshold to re-arm
    pub extended_hangover_ms: u64,    // 500ms if distance never exceeded rearm threshold
}

// In state machine
RecognitionState::Recovery { start_time, max_distance_seen } => {
    let elapsed = start_time.elapsed().as_millis() as u64;
    let rearm_threshold = gesture.threshold * config.rearm_threshold_factor;

    // Track maximum distance seen during recovery
    let new_max = max_distance_seen.max(current_distance);

    // Re-arm conditions:
    // 1. Standard hangover + distance clearly above threshold
    // 2. Extended hangover (regardless of distance)
    let can_rearm = if new_max > rearm_threshold {
        elapsed >= config.hangover_ms  // Standard path
    } else {
        elapsed >= config.extended_hangover_ms  // Extended path for sticky distances
    };

    if can_rearm {
        RecognitionState::Idle
    } else {
        RecognitionState::Recovery { start_time, max_distance_seen: new_max }
    }
}
```

**Recommended values**:
- `rearm_threshold_factor`: 1.2 (20% above threshold)
- `extended_hangover_ms`: 500ms (vs 300ms standard)

### 2.5 Latency Reduction (LOW PRIORITY - Optional)

**Current latency**: ~200ms (3 frames at 15Hz × 66ms)

**To achieve <150ms**:
1. Reduce `frames_to_fire` from 3 to 2 (~133ms)
2. Reduce `dtw_skip` from 4 to 2 (30Hz DTW rate)
3. Use LB_Keogh to offset CPU increase

**Configuration for low latency**:
```rust
RecognitionConfig {
    frames_to_fire: 2,        // ~100ms confirmation
    dtw_skip: 2,              // 30Hz DTW rate
    hangover_ms: 200,         // Faster recovery
    use_lb_keogh: true,       // Prune to offset CPU
    // ... rest unchanged
}
```

**Trade-off**: Slightly higher false positive risk. Monitor in testing.

---

## Part 3: What NOT to Do

### Avoid These Patterns (Documented Failures)

| ❌ Don't | Why |
|---------|-----|
| Distance-based recovery exit | Body tracking resting distance is close to threshold → stuck recognition |
| Best template only (GRT-style) | 30% fewer detections in A/B test; body tracking has more variation |
| Motion gate for recognition | Blocked valid gestures, added tuning complexity |
| Adaptive threshold from running data | Computed from wrong data, over-complicated |
| FastDTW | Actually slower than Sakoe-Chiba for our sequence sizes |
| Peak detection (local minima) | Added latency without improving accuracy |

### Validated Patterns to Keep

| ✅ Keep | Why |
|--------|-----|
| Time-based recovery (300ms) | Simple, reliable, no stuck states |
| Frame accumulation (3 frames) | Prevents noise spikes |
| All examples comparison | More forgiving of gesture variation |
| μ+σ threshold (coefficient 2.0) | Automatic calibration, adapts to complexity |
| Downsampling (15fps) | 64x speedup without accuracy loss |

---

## Part 4: Implementation Plan

### Phase 1: Core DTW Optimizations

- [ ] Add `dtw_distance_sakoe_chiba()` with band constraint
- [ ] Add `LBEnvelope` struct and `compute_lb_envelope()`
- [ ] Add `lb_keogh()` lower bound function
- [ ] Update `Gesture` to store precomputed envelopes
- [ ] Update recognizer to use constrained DTW with LB pruning
- [ ] Add configuration: `sakoe_chiba_band`, `use_lb_keogh`

**Files**: `src/engine/dtw.rs`, `src/engine/recognizer.rs`, `src/model/vocabulary.rs`

### Phase 2: Sequence Resampling

- [ ] Add `resample_sequence()` function
- [ ] Add `compute_path_length()` helper
- [ ] Add `interpolate_frames()` helper
- [ ] Apply resampling during training (normalize example lengths)
- [ ] Apply resampling during recognition (normalize window)
- [ ] Add configuration: `resample_target_length`

**Files**: `src/engine/dtw.rs`, `src/engine/training.rs`, `src/engine/recognizer.rs`

### Phase 3: Echo Guard Enhancement

- [ ] Add `rearm_threshold_factor` to config
- [ ] Add `extended_hangover_ms` to config
- [ ] Track `max_distance_seen` in Recovery state
- [ ] Implement dual-path re-arming logic
- [ ] Update diagnostic logging for new logic

**Files**: `src/engine/recognizer.rs`, `src/engine/diagnostics.rs`

### Phase 4: Testing & Tuning

- [ ] Run A/B test: current vs optimized
- [ ] Measure latency improvement
- [ ] Measure CPU usage change
- [ ] Count echo rate
- [ ] Document optimal parameter values

---

## Part 5: Expected Outcomes

### Performance Improvements

| Metric | Current | Expected | Notes |
|--------|---------|----------|-------|
| DTW computations | 100% | 10-20% | LB_Keogh pruning |
| DTW speed | O(N²) | O(N×B) | Sakoe-Chiba band |
| Match consistency | Good | Better | Resampling |

### Quality Improvements

| Goal | Current | Expected | How |
|------|---------|----------|-----|
| **Accuracy** | 100% | 100% | Constrained DTW prevents overfitting |
| **Latency** | ~200ms | ~150ms | Optional: reduce frames_to_fire |
| **Echo rate** | 5 in stress test | 0-1 | Dual threshold re-arming |

---

## References

### Internal Documentation
- `docs/plans/2026-01-29-fix-recognition-accuracy-timing-echo-plan.md` - VAD state machine learnings
- `.claude/commands/dtw-gesture-recognition.md` - DTW algorithms reference
- `CLAUDE.md` - A/B test results

### External Sources
- [UCR Suite - Searching Trillions of Time Series](https://www.cs.ucr.edu/~eamonn/UCRsuite.html)
- [FastDTW Critique Paper (Wu & Keogh, 2020)](https://arxiv.org/abs/2003.11246)
- [GRT - Gesture Recognition Toolkit](https://github.com/nickgillian/grt)
- [Wekinator DTW Implementation](https://github.com/fiebrink1/wekinator)
- [$1 Recognizer](https://depts.washington.edu/acelab/proj/dollar/index.html)
- [WebRTC VAD](https://github.com/nickcano/webrtc-vad)
- [Sakoe-Chiba Band Impact Research](https://www.researchgate.net/publication/301952855)

### Key Papers
- Sakoe & Chiba (1978): "Dynamic programming algorithm optimization for spoken word recognition"
- Keogh & Ratanamahatana (2005): "Exact indexing of dynamic time warping"
- Wobbrock et al. (2007): "$1 Unistroke Recognizer"
