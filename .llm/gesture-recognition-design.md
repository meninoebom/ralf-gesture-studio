# Gesture Recognition System Design: Deep Research Analysis

> **TL;DR**: Your current DTW approach is fundamentally correct for your use case (few training examples, real-time, flow-state UX). However, key optimizations can make it work more seamlessly. This document synthesizes research across voice recognition, gesture recognition, and time series analysis to provide a definitive design for RALF.

---

## Executive Summary

### What You're Trying to Achieve

You want gesture recognition that works like voice recognition:
- **Train on a handful of examples** (like teaching Siri a custom phrase)
- **Recognize in real-time** from continuous streams
- **Feel seamless** - instant recognition with no false positives

### The Core Insight

Voice recognition systems solved this problem decades ago. The key patterns that transfer:

| Voice Recognition Pattern | RALF Equivalent |
|--------------------------|-----------------|
| Phoneme detection via edge crossing | Fire when DTW distance crosses below threshold |
| Refractory period after keyword detected | Cooldown after hit fires (400-500ms) |
| State reset after detection | Clear buffer after hit |
| Wake word cascading (cheap filter → expensive recognition) | Activity detection → DTW |
| Streaming with sliding windows | Frame buffer with continuous matching |

**Your current implementation already has most of this.** The gaps are in efficiency and robustness.

---

## Part 1: Why DTW Is The Right Choice

### The Few-Shot Problem

Your use case has a fundamental constraint: **dancers train gestures during rehearsal, not weeks before**.

| Approach | Minimum Examples Needed | Training Time | Fits Your UX? |
|----------|------------------------|---------------|---------------|
| DTW | 1-5 | None (template) | **Yes** |
| $P Recognizer | 1 | None | Partially (loses temporal info) |
| Siamese Networks | 5-20 | Minutes-hours | No |
| GCN/Transformer | 100-1000+ | Hours-days | No |
| HMM | 10-50 | Minutes | Marginal |

Research confirms: **DTW is the gold standard for few-shot time series matching**.

From [Wang et al., 2013](https://link.springer.com/chapter/10.1007/978-3-642-33275-3_29):
> "DTW outperforms HMM in speed while maintaining competitive accuracy for gesture recognition with limited training data."

### What DTW Gets Right

1. **Works with 1 example** - True one-shot learning
2. **No training phase** - Instant feedback, stays in flow state
3. **Interpretable** - Threshold is a meaningful distance
4. **Handles temporal warping** - Slow and fast versions of same gesture match
5. **Simple implementation** - ~100 lines of core code

### What DTW Struggles With

1. **O(N×M) complexity** - Slow for long sequences
2. **No generalization** - Each gesture is independent
3. **Sensitive to noise** - Outlier frames affect distance
4. **Fixed vocabulary scaling** - More gestures = more comparisons

**These are solvable without abandoning DTW.**

---

## Part 2: Current Implementation Analysis

Based on code review of your implementation:

### What's Working Well

1. **Edge detection** (`RAW_LEARNINGS.md:101-133`)
   - Fire on threshold crossing, not staying below
   - Matches phoneme detection in ASR systems
   - Quick gestures (50-100ms) are detected

2. **Buffer reset after detection** (`RAW_LEARNINGS.md:26-63`)
   - Clears stale gesture frames
   - Forces natural recovery (buffer refill ~1.5s)
   - Backed by ASR literature on state reset

3. **Preserve edge state after reset** (`RAW_LEARNINGS.md:66-97`)
   - Prevents infinite re-triggering
   - User must complete full cycle: gesture → rest → gesture

4. **Refractory period** (`RAW_LEARNINGS.md:221-238`)
   - 400-500ms cooldown prevents accidental doubles
   - Allows ~2 hits/sec for intentional rapid sequences

5. **Normalized DTW** (`RAW_LEARNINGS.md:292-305`)
   - Divides by average sequence length
   - Makes thresholds comparable across gesture durations

### What Needs Improvement

1. **Performance under load** (`RAW_LEARNINGS.md:136-171`)
   - DTW is O(N×M) per comparison
   - 180 frames × 180 frames × 68 dimensions × 15 examples = millions of ops/frame
   - Currently solved by frame skipping, but this is a band-aid

2. **Fixed-length window assumption**
   - Current: Compare against 180-frame window
   - Problem: Gestures of different lengths match differently
   - Some gestures are 1 second, others are 5 seconds

3. **No activity detection**
   - Running DTW even when dancer is standing still
   - Wasted computation and potential false positives

4. **Raw coordinate features**
   - Using absolute X-Y joint positions
   - Sensitive to where dancer stands in frame
   - Doesn't capture velocity/acceleration (the "how" of movement)

---

## Part 3: Recommended Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RECOGNITION PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  OSC Input (60fps)                                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────┐                                                │
│  │ Frame Preprocess │  - Normalize skeleton (translation/scale)    │
│  │                  │  - Downsample to 15fps                        │
│  │                  │  - Compute velocity features                  │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ Activity Gate   │  - Motion energy threshold                    │
│  │                  │  - Skip DTW if motion < threshold             │
│  │                  │  - Cheap filter (O(1) per frame)              │
│  └────────┬────────┘                                                │
│           │ (only if active)                                        │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ Sliding Window  │  - Circular buffer of processed frames        │
│  │ Buffer          │  - Multiple window sizes per gesture          │
│  │                  │  - Configurable based on gesture duration     │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ DTW Matching    │  - Compare window to gesture prototypes       │
│  │                  │  - Early termination with LB_Keogh           │
│  │                  │  - Sakoe-Chiba band constraint               │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │ Hit Detection   │  - Edge detection (crossing threshold)        │
│  │                  │  - Refractory period per gesture             │
│  │                  │  - Buffer reset on hit                        │
│  └────────┬────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  OSC Output                                                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Stage 1: Frame Preprocessing

**Goal**: Normalize input and reduce dimensionality.

```rust
struct ProcessedFrame {
    // Normalized joint positions (relative to hip/torso)
    positions: [f32; 68],    // 34 joints × XY

    // Velocity features (how fast each joint is moving)
    velocities: [f32; 68],   // First derivative

    // Motion energy (scalar summarizing total movement)
    motion_energy: f32,      // Sum of squared velocities
}

fn preprocess_frame(raw: &[f32], prev: Option<&ProcessedFrame>) -> ProcessedFrame {
    // 1. Translate so hip is at origin
    let hip_x = raw[0];  // Assuming hip is first joint
    let hip_y = raw[1];
    let positions: Vec<f32> = raw.chunks(2)
        .flat_map(|chunk| [chunk[0] - hip_x, chunk[1] - hip_y])
        .collect();

    // 2. Scale normalize (optional, based on shoulder width)

    // 3. Compute velocity if previous frame exists
    let velocities = match prev {
        Some(p) => positions.iter()
            .zip(p.positions.iter())
            .map(|(curr, prev)| curr - prev)
            .collect(),
        None => vec![0.0; 68],
    };

    // 4. Motion energy
    let motion_energy: f32 = velocities.iter().map(|v| v * v).sum();

    ProcessedFrame { positions, velocities, motion_energy }
}
```

**Why velocity features?**
- Captures the "dynamics" of movement (fast vs slow)
- Less sensitive to where dancer stands
- Helps distinguish similar poses with different intentions

**Frame downsampling**:
- MoveNet outputs at 30-60fps
- Gesture recognition only needs ~15fps
- Reduces DTW computation by 4-16x
- Skip every 2nd or 4th frame

### Stage 2: Activity Gate

**Goal**: Only run expensive DTW when meaningful movement is occurring.

```rust
const ACTIVITY_THRESHOLD: f32 = 0.01;  // Calibrate based on noise floor
const ACTIVITY_FRAMES: usize = 5;       // Require sustained movement

fn is_active(recent_frames: &[ProcessedFrame]) -> bool {
    let avg_energy: f32 = recent_frames.iter()
        .map(|f| f.motion_energy)
        .sum::<f32>() / recent_frames.len() as f32;

    avg_energy > ACTIVITY_THRESHOLD
}
```

**Why this helps**:
- Dancers stand still between gestures
- DTW is wasted on static poses
- Reduces false positives from noise
- Voice recognition uses equivalent "Voice Activity Detection" (VAD)

### Stage 3: Improved DTW

#### 3.1: Sakoe-Chiba Band Constraint

Limit how much temporal warping is allowed:

```rust
fn dtw_constrained(seq1: &[Vec<f32>], seq2: &[Vec<f32>], band_width: usize) -> f32 {
    let n = seq1.len();
    let m = seq2.len();

    // Only compute cells within band
    let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];
    cost[0][0] = 0.0;

    for i in 1..=n {
        // Only iterate within band of diagonal
        let j_start = (i as i32 - band_width as i32).max(1) as usize;
        let j_end = (i + band_width).min(m);

        for j in j_start..=j_end {
            let frame_dist = euclidean_distance(&seq1[i-1], &seq2[j-1]);
            let min_prev = cost[i-1][j-1]
                .min(cost[i-1][j])
                .min(cost[i][j-1]);
            cost[i][j] = frame_dist + min_prev;
        }
    }

    cost[n][m]
}
```

**Complexity reduction**: O(N×M) → O(N×W) where W is band width (typically W << M).

**Recommended band_width**: 10-20% of sequence length.

#### 3.2: LB_Keogh Lower Bound for Early Termination

Before computing full DTW, check if it's worth computing:

```rust
fn lb_keogh(query: &[Vec<f32>], template: &[Vec<f32>], envelope_width: usize) -> f32 {
    // Compute upper/lower envelope for template
    let (upper, lower) = compute_envelope(template, envelope_width);

    // Sum of distances where query falls outside envelope
    let mut lb = 0.0;
    for (i, q) in query.iter().enumerate() {
        if i >= upper.len() { break; }

        for (j, &qj) in q.iter().enumerate() {
            if qj > upper[i][j] {
                lb += (qj - upper[i][j]).powi(2);
            } else if qj < lower[i][j] {
                lb += (lower[i][j] - qj).powi(2);
            }
        }
    }

    lb.sqrt()
}

fn match_with_pruning(window: &[Vec<f32>], gestures: &[Gesture]) -> Option<Hit> {
    let mut best_distance = f32::MAX;
    let mut best_gesture = None;

    for gesture in gestures {
        for example in &gesture.examples {
            // Lower bound check - O(N) instead of O(N×M)
            let lb = lb_keogh(window, &example.frames, 5);

            if lb >= gesture.threshold {
                continue;  // Can't possibly match, skip DTW
            }
            if lb >= best_distance {
                continue;  // Already found a better match
            }

            // Only compute full DTW if lower bound is promising
            let dist = dtw_constrained(window, &example.frames, 20);

            if dist < best_distance {
                best_distance = dist;
                best_gesture = Some(gesture.clone());
            }
        }
    }

    best_gesture.filter(|g| best_distance < g.threshold)
        .map(|g| Hit { gesture: g, distance: best_distance })
}
```

**Speedup**: Typically prunes 80-90% of DTW computations.

#### 3.3: Prototype Templates

Instead of comparing against all examples, compute a single prototype per gesture:

```rust
struct GesturePrototype {
    // Average of all aligned examples
    frames: Vec<Vec<f32>>,

    // Variance for adaptive thresholding (future)
    variance: Vec<Vec<f32>>,
}

fn compute_prototype(examples: &[Example]) -> GesturePrototype {
    // 1. Pick reference example (longest or most representative)
    let reference = examples.iter()
        .max_by_key(|e| e.frames.len())
        .unwrap();

    // 2. Align all examples to reference using DTW path
    let aligned: Vec<_> = examples.iter()
        .map(|e| align_to_reference(&e.frames, &reference.frames))
        .collect();

    // 3. Average aligned sequences
    let prototype_frames = average_sequences(&aligned);

    GesturePrototype {
        frames: prototype_frames,
        variance: compute_variance(&aligned),
    }
}
```

**Benefit**: N examples → 1 comparison per gesture instead of N.

### Stage 4: Multi-Resolution Window Matching

**Problem**: Fixed 180-frame window assumes all gestures are ~3 seconds.

**Solution**: Variable windows based on gesture duration.

```rust
struct GestureConfig {
    // Derived from training examples
    min_frames: usize,
    max_frames: usize,
    typical_frames: usize,  // Average of examples
}

fn match_gesture(buffer: &FrameBuffer, gesture: &Gesture, config: &GestureConfig) -> Option<f32> {
    // Try windows at multiple lengths
    let window_sizes = [
        config.min_frames,
        config.typical_frames,
        config.max_frames,
    ];

    let mut best_distance = f32::MAX;

    for size in window_sizes {
        if buffer.len() < size { continue; }

        let window = buffer.recent_frames(size);
        let distance = dtw_normalized(&window, &gesture.prototype.frames);

        best_distance = best_distance.min(distance);
    }

    Some(best_distance)
}
```

**Better approach for continuous spotting**: Subsequence DTW (see next section).

---

## Part 4: Continuous Gesture Spotting (SPRING Algorithm)

Your current approach uses a **fixed-length sliding window**. This has limitations:

- Gestures must align with window boundaries
- Short gestures get padded with irrelevant data
- Long gestures get truncated

**Subsequence DTW** solves this by finding the best-matching subsequence of any length.

### The SPRING Algorithm

From [Sakurai et al., 2007](https://www.cs.cmu.edu/~christos/PUBLICATIONS/ICDE07-spring.pdf):

> "SPRING maintains only O(1) space and O(1) time per tick, independent of the length of the data stream."

```rust
// Simplified SPRING implementation
struct SpringMatcher {
    // DTW matrix column (only need previous column)
    prev_column: Vec<f32>,

    // Start positions of potential matches
    start_positions: Vec<usize>,

    // Template to match against
    template: Vec<Vec<f32>>,
}

impl SpringMatcher {
    fn process_frame(&mut self, frame: &[f32], timestamp: usize) -> Option<Match> {
        let m = self.template.len();
        let mut curr_column = vec![f32::INFINITY; m + 1];
        curr_column[0] = 0.0;  // Can start matching at any time

        for j in 1..=m {
            let frame_dist = euclidean_distance(frame, &self.template[j-1]);

            // Standard DTW recurrence
            let match_cost = self.prev_column[j-1] + frame_dist;  // Diagonal
            let insert_cost = self.prev_column[j] + frame_dist;    // Vertical
            let delete_cost = curr_column[j-1] + frame_dist;       // Horizontal

            curr_column[j] = match_cost.min(insert_cost).min(delete_cost);
        }

        // Check if we've completed a match
        let final_cost = curr_column[m];
        let normalized_cost = final_cost / m as f32;

        self.prev_column = curr_column;

        if normalized_cost < THRESHOLD {
            Some(Match { end: timestamp, distance: normalized_cost })
        } else {
            None
        }
    }
}
```

**Key insight**: We only need the previous column of the DTW matrix, not the full matrix.

**Complexity per frame**: O(M) where M is template length (independent of stream length).

### When to Use Subsequence DTW

| Scenario | Fixed Window | Subsequence DTW |
|----------|--------------|-----------------|
| Gestures of known, consistent length | Good | Overkill |
| Variable-length gestures | Poor | **Recommended** |
| Multiple gestures overlapping | Poor | **Recommended** |
| Real-time continuous stream | Adequate | **Better** |

**Recommendation**: Start with fixed window (simpler), migrate to SPRING if variable-length becomes a problem.

---

## Part 5: Feature Engineering

### Current Features (Position-Only)

```
Frame = [x0, y0, x1, y1, ..., x33, y33]  // 68 floats
```

**Problems**:
1. Sensitive to where dancer stands in frame
2. Doesn't capture velocity (fast wave ≠ slow wave)
3. Doesn't capture relative positions (hands together vs apart)

### Recommended Features

```rust
struct EnhancedFrame {
    // Position features (normalized to hip-centered)
    positions: [f32; 68],

    // Velocity features (first derivative)
    velocities: [f32; 68],

    // Key relative distances (semantically meaningful)
    hand_distance: f32,              // Distance between hands
    hand_to_hip_left: f32,           // Left hand height relative to hip
    hand_to_hip_right: f32,          // Right hand height relative to hip
    feet_distance: f32,              // Stance width

    // Joint angles (rotation-invariant)
    left_elbow_angle: f32,
    right_elbow_angle: f32,
    left_knee_angle: f32,
    right_knee_angle: f32,
    spine_bend: f32,
}
```

### Joint Importance Weighting

Not all joints matter equally for every gesture:

```rust
struct GestureWeights {
    // Per-joint importance weights [0, 1]
    joint_weights: [f32; 34],
}

fn weighted_distance(a: &[f32], b: &[f32], weights: &[f32]) -> f32 {
    a.chunks(2)
        .zip(b.chunks(2))
        .zip(weights.iter())
        .map(|((ac, bc), &w)| {
            let dx = ac[0] - bc[0];
            let dy = ac[1] - bc[1];
            w * (dx * dx + dy * dy).sqrt()
        })
        .sum()
}

// Example: "wave" gesture focuses on hands
let wave_weights = GestureWeights {
    joint_weights: {
        let mut w = [0.1; 34];  // Low weight for most joints
        w[HAND_LEFT] = 1.0;      // High weight for hands
        w[HAND_RIGHT] = 1.0;
        w[WRIST_LEFT] = 0.8;
        w[WRIST_RIGHT] = 0.8;
        w
    }
};
```

**Implementation**: Either:
1. Auto-detect from training examples (joints with high variance are important)
2. Let user specify "key joints" per gesture in UI

---

## Part 6: Threshold Auto-Calibration

### The Problem

From `RAW_LEARNINGS.md:239-259`:
> Users don't know what threshold values to set. Raw DTW distances are meaningless numbers.

### Solution: Baseline-Based Calibration

```rust
struct CalibrationData {
    // Distance between neutral pose and gesture
    baseline_distance: f32,

    // Distance between different examples of same gesture
    intra_class_variance: f32,

    // Distance to nearest other gesture (if multiple gestures)
    inter_class_distance: Option<f32>,
}

fn auto_calibrate_threshold(gesture: &Gesture, baseline: &Example) -> f32 {
    // 1. Compute distance from baseline (rest pose) to gesture examples
    let baseline_distances: Vec<f32> = gesture.examples.iter()
        .map(|e| dtw_normalized(&baseline.frames, &e.frames))
        .collect();

    let avg_baseline_dist = baseline_distances.iter().sum::<f32>()
        / baseline_distances.len() as f32;

    // 2. Compute variance within gesture examples
    let intra_distances: Vec<f32> = gesture.examples.iter()
        .flat_map(|e1| gesture.examples.iter()
            .map(move |e2| dtw_normalized(&e1.frames, &e2.frames)))
        .collect();

    let max_intra = intra_distances.iter().cloned().fold(0.0, f32::max);

    // 3. Threshold = midpoint between intra-class max and baseline
    // This ensures all examples match while baseline doesn't
    let threshold = (max_intra + avg_baseline_dist * 0.8) / 2.0;

    // 4. Add safety margin
    threshold * 1.1
}
```

### UX Flow

1. **First launch**: Prompt user to record 3 seconds of "neutral stance"
2. **After training each gesture**: Auto-compute threshold
3. **Performance mode**: Show warning if gesture distance overlaps with baseline
4. **Manual override**: Always allow threshold slider adjustment

---

## Part 7: Implementation Roadmap

### Phase 1: Quick Wins (Immediate)

**Goal**: Improve current system without architectural changes.

| Change | Effort | Impact |
|--------|--------|--------|
| Frame downsampling (60fps → 15fps) | 1 hour | 4x faster DTW |
| Activity gate (skip DTW when still) | 2 hours | 50-80% computation reduction |
| Sakoe-Chiba band constraint | 2 hours | 3-5x faster DTW |
| Prototype averaging | 4 hours | N examples → 1 comparison |

**Expected outcome**: Real-time performance without frame skipping.

### Phase 2: Feature Engineering (v0.2)

**Goal**: More robust recognition across positions and speeds.

| Change | Effort | Impact |
|--------|--------|--------|
| Hip-centered normalization | 2 hours | Position-independent |
| Velocity features | 4 hours | Speed-dependent matching |
| Per-gesture joint weights | 8 hours | Focused attention |

**Expected outcome**: Gestures work from different positions in room.

### Phase 3: Auto-Calibration (v0.3)

**Goal**: Eliminate manual threshold tuning.

| Change | Effort | Impact |
|--------|--------|--------|
| Baseline recording UI | 4 hours | Reference for thresholds |
| Threshold auto-compute | 4 hours | Users don't guess numbers |
| Inter-gesture separation check | 4 hours | Warn on ambiguous gestures |

**Expected outcome**: Train → Works. No tuning needed.

### Phase 4: Advanced Matching (v1.0+)

**Goal**: Handle complex scenarios.

| Change | Effort | Impact |
|--------|--------|--------|
| Subsequence DTW (SPRING) | 16 hours | Variable-length detection |
| LB_Keogh pruning | 8 hours | Order-of-magnitude speedup |
| Multi-resolution windows | 8 hours | Mix of short and long gestures |

**Expected outcome**: Seamless experience matching voice recognition.

---

## Part 8: What NOT to Do

Based on research, these approaches are **not recommended** for your use case:

### 1. Neural Networks (GCN, LSTM, Transformer)

**Why not**: Require 100+ examples per gesture, training breaks flow state, need GPU.

**Only consider if**: You build a pre-trained model on public datasets (NTU RGB+D) and fine-tune with user's examples. But this adds significant complexity.

### 2. FastDTW

**Why not**: Despite the name, [research shows](https://arxiv.org/pdf/2003.11246) it's often slower than standard DTW in realistic settings.

**Use instead**: Sakoe-Chiba constraint + LB_Keogh pruning.

### 3. $P Recognizer

**Why not**: Treats gesture as unordered point cloud, loses temporal information. "Fast wave" and "slow wave" become identical.

**Use for**: Touch/pen gestures where timing doesn't matter.

### 4. Hysteresis

From `RAW_LEARNINGS.md:206-219`:
> Dancers cannot use this. It forces them to "return to rest position" between every gesture.

**Use instead**: Refractory period (cooldown).

### 5. Debounce for Hit Detection

From `RAW_LEARNINGS.md:101-133`:
> Debounce missed quick gestures and felt unresponsive.

**Use instead**: Edge detection (fire on threshold crossing).

---

## Part 9: Comparison to Wekinator

Your system is essentially a Rust reimplementation of Wekinator's core. Here's how they compare:

| Feature | Wekinator | RALF Gesture Studio |
|---------|-----------|---------------------|
| DTW Implementation | FastDTW (Java) | Custom (Rust) |
| Neural Net Mode | WEKA MultilayerPerceptron | Not implemented |
| Training UX | Real-time, interactive | Structured sessions with audio |
| Threshold Tuning | Manual slider | Manual (auto-calibration planned) |
| OSC Compatibility | Port 6448/12000 | Same defaults |
| Performance | Adequate | Better (Rust) |
| Flexibility | GUI only | Potential for embedding |

**What Wekinator gets right**: The fundamental architecture. It's been battle-tested for 15+ years by creative coders.

**What you can improve**:
1. Rust performance vs Java
2. Structured training sessions with audio cues (dancers love this)
3. Auto-calibration (Wekinator requires manual tuning)
4. Modern optimizations (LB_Keogh, band constraints)

---

## Part 10: Key Resources

### Essential Reading

1. **[FastDTW: Toward Accurate DTW in Linear Time and Space](http://cs.fit.edu/~pkc/papers/tdm04.pdf)**
   - Original FastDTW paper, but note caveats about performance

2. **[SPRING: Subsequence Matching in Streams](https://www.cs.cmu.edu/~christos/PUBLICATIONS/ICDE07-spring.pdf)**
   - Continuous gesture spotting algorithm

3. **[LB_Keogh for DTW Pruning](https://www.cs.ucr.edu/~eamonn/keogh_DMKD2004.pdf)**
   - Lower bound that prunes 80-90% of comparisons

4. **[Wekinator Source Code](https://github.com/fiebrink1/wekinator)**
   - See how Wekinator implements DTW + neural net modes

### Rust Crates

| Crate | Purpose | Notes |
|-------|---------|-------|
| [augurs-dtw](https://crates.io/crates/augurs-dtw) | DTW with optimizations | UCR Suite influences |
| [ndarray](https://crates.io/crates/ndarray) | Efficient array ops | For matrix computations |

### Academic Papers on Gesture Recognition

1. **[Skeleton-based Action Recognition Survey 2023](https://arxiv.org/abs/2302.05537)** - Comprehensive overview (for reference, not implementation)

2. **[Multi-dimensional DTW for Gestures](https://www.researchgate.net/publication/228740947)** - Per-joint weighting theory

3. **[Contrastive Learning for Keyword Spotting](https://arxiv.org/html/2401.06485)** - State reset after detection pattern

---

## Conclusion

Your current approach is fundamentally sound. DTW with edge detection, refractory periods, and buffer reset is exactly what voice recognition systems use. The key improvements are:

1. **Performance**: Sakoe-Chiba constraint + LB_Keogh pruning + frame downsampling
2. **Robustness**: Hip-centered normalization + velocity features
3. **Usability**: Auto-threshold calibration from baseline

You're not over-engineering. You're implementing the right algorithm for your constraints. The optimizations above will make it feel as seamless as voice recognition.

**The voice recognition analogy holds**: Train a few examples, recognize in real-time, immediate feedback. DTW gives you this. Neural networks would break the flow-state UX that makes RALF special.

---

*Document generated from deep research on 2026-01-24*
*Sources: Voice recognition literature, gesture recognition surveys, time series analysis papers, Wekinator architecture*
