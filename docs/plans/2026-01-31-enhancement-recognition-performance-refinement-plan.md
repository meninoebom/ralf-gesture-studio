---
title: "Enhancement: Recognition Performance Refinement"
type: enhancement
date: 2026-01-31
source: gesture_recognition_research_report.md
---

# Enhancement: Recognition Performance Refinement

## Overview

Refine RALF Gesture Studio's gesture recognition across four dimensions: **accuracy**, **responsiveness**, **false positive reduction**, and **echo prevention durability** — based on a comprehensive research report covering DTW alternatives, feature engineering, and training UX improvements.

The current system (v0.7.0) achieves 0% echo rate, 95%+ hit margin, and ~200ms latency with 3 gestures and 38 training examples. These refinements prepare the system for scaling to 10-30 gestures while maintaining or improving those metrics.

## Related Issues & Prior Work

| Issue | Relationship | Status |
|-------|-------------|--------|
| **#13** — Add preprocessing pipeline between receiver and recognizer | **Parent issue.** Phase 1 of this plan implements the first concrete preprocessors under the architecture #13 describes. #13 is broader (quaternion conversion, dimension selection, multi-sensor fusion); this plan is the first deliverable toward that vision. | Open |
| **#11** — Add frame dimension validation | **Prerequisite (done).** Committed in `c1dcbeb`. Validation now runs in `process_frames()` and in the recognizer, exactly where the preprocessing pipeline will be inserted. | Closed |
| **#6** — Tune recognition: reduce latency and eliminate echoes | **Superseded.** The work described in #6 was completed across v0.5.0–v0.7.0 (VAD state machine, safety valve, global NMS → 0% echo rate). This plan's Phase 3 is the next evolution. | Closed |
| **#7** — Research metadata / vocabulary format | **Compatible.** #7 introduced v1.1 format with metadata fields. This plan extends the format to v1.2 by adding a `preprocessing` field. Same files touched (`vocabulary.rs`, `persistence.rs`, `FORMAT.md`) but changes are additive. | Open |
| **#12** — Wire InputConfig to OSC receiver | **Independent.** Multi-source OSC plumbing. No conflict. | Open |

**Prior plan**: `docs/plans/2026-01-30-recognition-optimization-v2-plan.md` explicitly lists velocity features and hip-centered normalization as "Future Optimizations — not in this phase." This plan picks up where that plan left off.

## Problem Statement / Motivation

The recognition pipeline currently operates on **raw, unprocessed skeleton coordinates**. OSC frames (66 floats: 33 joints x XY) pass directly into the DTW comparator with no feature engineering. While this works well for 3 distinct gestures (wave, jump, spin), it has known limitations:

1. **Position sensitivity** — dancer's location in camera frame affects recognition
2. **Scale sensitivity** — different body sizes produce different distances
3. **Velocity is implicit** — DTW captures dynamics indirectly but cannot distinguish "fast wave" from "slow wave" explicitly
4. **Training burden** — 6+ examples per gesture required, no quality feedback, no data augmentation
5. **Scaling risk** — as gestures increase, similar movements may become confusable without better feature discrimination

## Proposed Solution

A three-phase approach that builds the **preprocessing pipeline** first (foundation), then **data quality tools** (leverage), then **recognition refinements** (tuning). Each phase is independently valuable and shippable.

## Architectural Decision: Store Raw, Preprocess at Comparison Time

**Decision**: Store raw frames in `.ralf` files. Apply preprocessing at load time and comparison time.

**Rationale**:
- Preserves backward compatibility — v1.0/v1.1 files load unchanged
- Enables feature toggling without re-recording
- Raw data is the ground truth; preprocessing is derived
- CPU cost of preprocessing is negligible compared to DTW (~0.01ms per frame vs ~1-5ms per DTW comparison)

**Implication**: A `PreprocessingConfig` is stored per-vocabulary. When a vocabulary loads, stored examples are preprocessed once into an in-memory representation. Live frames are preprocessed on arrival. Thresholds are recomputed whenever the pipeline config changes.

## Technical Approach

### Preprocessing Pipeline Architecture

```
Raw OSC Frame (66 floats)
    │
    ▼
┌─────────────────────┐
│  1. Hip Centering    │  Subtract hip-center from all joints
│     (position inv.)  │  MediaPipe: avg of joints 23+24
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  2. Scale Normalize  │  Divide by shoulder width
│     (body-size inv.) │  MediaPipe: dist(joint 11, joint 12)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  3. Velocity Features│  Append dx,dy per joint
│     (dynamics)       │  Frame grows: 66 → 132 floats
└─────────┬───────────┘
          │
          ▼
Preprocessed Frame (132 floats) → Buffer → DTW
```

Each stage is independently toggleable. The pipeline is applied in a new module `src/engine/preprocess.rs`.

### Joint Index Mapping

Hard-code MediaPipe Pose 33 keypoints for `tracking_system: "mediapipe-pose-33-xy"`:

```rust
// MediaPipe Pose landmark indices (0-indexed)
// Frame layout: [x0,y0, x1,y1, ..., x32,y32] = 66 floats
const LEFT_HIP: usize = 23;    // indices [46, 47]
const RIGHT_HIP: usize = 24;   // indices [48, 49]
const LEFT_SHOULDER: usize = 11;  // indices [22, 23]
const RIGHT_SHOULDER: usize = 12; // indices [24, 25]

// Mirror swap pairs (left_index, right_index)
const MIRROR_PAIRS: [(usize, usize); 11] = [
    (11, 12), (13, 14), (15, 16), (17, 18),
    (19, 20), (21, 22), (23, 24), (25, 26),
    (27, 28), (29, 30), (31, 32),
];
```

Disable normalization for unknown tracking systems; log a warning and pass through raw frames.

---

## Implementation Phases

### Phase 1: Preprocessing Pipeline (Foundation)

**Goal**: Position-invariant, scale-invariant, velocity-aware features. This is the highest-impact change identified by the research report.

#### 1a. Create `src/engine/preprocess.rs`

New module following the one-concept-per-file convention:

```rust
pub struct PreprocessingConfig {
    pub hip_normalize: bool,     // Subtract hip center from all joints
    pub scale_normalize: bool,   // Divide by shoulder width
    pub velocity_features: bool, // Append first derivative
}

pub struct Preprocessor {
    config: PreprocessingConfig,
    tracking: TrackingLayout,    // Joint index mapping
    prev_frame: Option<Vec<f32>>, // For velocity computation
}

impl Preprocessor {
    /// Process a single raw frame. Returns preprocessed frame.
    pub fn process_frame(&mut self, raw: &[f32]) -> Vec<f32>;

    /// Process a stored sequence (for training examples at load time).
    /// Handles first-frame velocity boundary condition internally.
    pub fn process_sequence(&self, raw: &[Vec<f32>]) -> Vec<Vec<f32>>;

    /// Reset state (clear prev_frame). Call when starting new recording.
    pub fn reset(&mut self);
}
```

**Edge cases**:
- **First frame velocity**: Use zero velocity for frame 0 (pad with zeros). This adds a slight bias but is the standard approach (sktime, GRT both do this).
- **Shoulder width near zero**: Clamp minimum shoulder width to 0.01 (prevents division by zero). Log a warning when this fires — likely indicates bad tracking data.
- **Unknown tracking system**: Skip normalization, pass raw frame through. Velocity features can still apply (dimension-agnostic).

#### 1b. Integrate into `process_frames()` in `src/gui/mod.rs`

The insertion point sits immediately after the dimension validation added in #11 (`c1dcbeb`). The flow becomes: validate dimensions → preprocess → feed to training/recognition. This ensures both paths get identical preprocessing:

```rust
// Current (after #11): validate dimensions → self.recognizer.process_frame(frame.clone());
// New:                  validate dimensions → preprocess → self.recognizer.process_frame(preprocessed);
// Same for training session path.
```

This implements the first concrete preprocessor in the architecture described by #13.

#### 1c. Preprocess stored examples on vocabulary load

When a vocabulary loads via `sync_recognizer()`, apply the preprocessing pipeline to all stored examples before passing them to the recognizer. This transforms raw stored frames into the same representation as live frames.

#### 1d. Auto-recompute thresholds

After preprocessing, DTW distances change. Call `compute_threshold_stats()` on the preprocessed examples. If a gesture had `threshold_manual_override: false`, its threshold updates automatically. If manual override is on, show a one-time notification that distances have changed.

#### 1e. Add `preprocessing` field to Vocabulary

```rust
pub struct Vocabulary {
    // ... existing fields ...
    pub preprocessing: PreprocessingConfig, // NEW in v1.2
}
```

Bump format version to `"1.2"`. On loading v1.1 files, default to `PreprocessingConfig { hip_normalize: false, scale_normalize: false, velocity_features: false }` — existing behavior preserved.

**Files changed**: `src/engine/preprocess.rs` (new), `src/engine/mod.rs`, `src/gui/mod.rs`, `src/model/vocabulary.rs`, `src/model/persistence.rs`, `FORMAT.md`

### Phase 2: Data Quality (Leverage)

**Goal**: Get more value from fewer recordings. Reduce training burden from 6+ examples to 4-5 real + augmented.

#### 2a. Create `src/engine/augmentation.rs`

Data augmentation that operates on preprocessed sequences:

| Technique | Parameters | Effect |
|-----------|------------|--------|
| Temporal stretch | 0.85x - 1.15x, linear interp, resample to original length | Speed variation |
| Spatial jitter | sigma = 1% of mean joint displacement | Noise robustness |
| Horizontal mirror | MediaPipe joint swap pairs | Left/right variation |

**Storage**: Ephemeral. Only real recorded examples are stored in `.ralf` files. Augmented examples are generated when recognition starts (in `recognizer.start()`), using a deterministic seed stored per-gesture for reproducibility.

**Configuration** (per-gesture):
```rust
pub struct AugmentationConfig {
    pub enabled: bool,
    pub temporal_stretch: bool,
    pub spatial_jitter: bool,
    pub mirror: bool,
    pub seed: u64,
    pub multiplier: u32,  // How many augmented copies per real example
}
```

**Constraint**: Augmented examples are used for DTW comparison but NOT for threshold statistics. Thresholds should reflect real gesture variance, not artificial noise.

**Sequence length invariant**: Temporal stretching resamples to the original frame count via linear interpolation. This preserves the `window_size` assumption in the recognizer.

#### 2b. Create `src/engine/quality.rs`

Example quality assessment, run after each training example is recorded:

```rust
pub enum QualityIssue {
    TooStill { motion_energy: f32, threshold: f32 },
    Outlier { distance_to_others: f32, expected_range: f32 },
    TooShort { frames: usize, expected: usize },
}

/// Assess quality of a new example against existing examples.
/// Returns None if quality is acceptable.
pub fn assess_example(
    new: &Sequence,
    existing: &[Sequence],
    config: &QualityConfig,
) -> Option<QualityIssue>;
```

**"Too still" threshold**: Compute total frame-to-frame displacement summed across all joints. If below 5% of the mean displacement across existing examples, flag it. This is adaptive per-gesture (a jump has higher expected motion than a head nod).

**"Outlier" threshold**: If mean DTW distance to existing examples exceeds 3x the inter-example mean distance, flag it.

**"Too short"**: If frame count is less than 50% of the mean frame count of existing examples.

**UX**: Quality feedback appears **post-session** as a non-blocking summary: "3 of 5 examples look good. 2 flagged: #2 (low motion), #4 (outlier). You can delete them in the gesture panel." This preserves flow state during training.

**Files changed**: `src/engine/augmentation.rs` (new), `src/engine/quality.rs` (new), `src/engine/mod.rs`, `src/engine/recognizer.rs` (augmentation at start), `src/gui/mod.rs` (quality feedback display)

### Phase 3: Recognition Refinements (Tuning)

**Goal**: Fine-tune recognition accuracy and robustness for scaling to more gestures.

#### 3a. Variance-Based Joint Weighting

Automatically down-weight joints that don't move during a gesture:

```rust
/// Compute per-dimension weights from training examples.
/// Dimensions with near-zero variance get near-zero weight.
pub fn compute_joint_weights(examples: &[Sequence]) -> Vec<f32>;
```

Weights are applied by scaling frame data before DTW (multiply each dimension by its weight). This preserves the LB_Keogh lower bound property because scaling is monotonic.

Stored per-gesture in the `Gesture` struct. Recomputed when examples change.

**Modified**: `euclidean_distance()` in `src/engine/dtw.rs` gains an optional weights parameter.

#### 3b. Derivative DTW (DDTW) — Optional Complement

Run DTW on velocity profiles (first derivatives) alongside position DTW:

```rust
/// Compute derivative sequence using central differences.
/// derivative[t] = (seq[t+1] - seq[t-1]) / 2
/// Boundary: forward/backward difference for first/last frames.
pub fn compute_derivative(sequence: &Sequence) -> Sequence;
```

**Combined distance**: `combined = alpha * position_dtw + (1 - alpha) * derivative_dtw` where `alpha` defaults to 0.7 (position-weighted). This is tunable per-vocabulary.

**Implementation note**: If velocity features are already enabled in the preprocessing pipeline, DDTW can operate on the position portion of the preprocessed frame (first 66 dimensions) and the velocity portion (last 66 dimensions) is already available. The derivative DTW is then redundant with velocity features — they capture similar information. **Recommend implementing one or the other, not both.**

**Decision**: If velocity features (Phase 1) are enabled, DDTW is unnecessary. DDTW is the fallback if velocity features prove too noisy or if the user wants to compare velocity profiles of velocity features (second derivative / acceleration).

#### 3c. Consensus Scoring (Optional Enhancement)

Add a secondary gate: require at least 50% of training examples to have distance below threshold before firing:

```rust
// In compute_distances(), alongside min_distance:
let vote_count = distances.iter()
    .filter(|d| **d < threshold)
    .count();
let consensus = vote_count as f32 / distances.len() as f32;

// Fire requires both: min_distance < threshold AND consensus >= 0.5
```

This is a **strictness increase** — it prevents firing when a single outlier example happens to match but most don't. Most useful when scaling to confusable gestures.

Default: OFF. Toggled per-gesture for gestures that have confusion problems.

**Files changed**: `src/engine/dtw.rs` (weighted distance, derivative), `src/engine/recognizer.rs` (consensus scoring, joint weights integration)

---

## What We Are NOT Doing (and Why)

| Technique | Verdict | Reason |
|-----------|---------|--------|
| SPRING / Subsequence DTW | Skip | Fixed-window + VAD state machine already handles continuous recognition |
| FastDTW | Skip | Actually slower than Sakoe-Chiba for our sequence sizes (Keogh 2020) |
| Soft-DTW | Future | Only needed if adding differentiable ML components |
| TCN / Neural networks | Future | Requires thousands of examples; our few-shot constraint rules it out |
| Hierarchical recognition | Future | Useful at 10+ gestures but premature for current 3 |
| VP-tree indexing | Future | Useful at 50+ gestures |
| LMA features | Future | Interesting for movement quality but not core recognition |
| Velocity-based recovery | Defer | Timer-based safety valve works; distance/velocity recovery has failed before (documented in learnings) |
| Motion energy gating | Defer | Was tried and removed; VAD state machine handles idle state |
| Negative examples / garbage model | Defer | Threshold-based approach working; revisit when false positives emerge |
| PCA dimensionality reduction | Skip | 66-132 dimensions is manageable; PCA risks losing gesture-specific info |

---

## Acceptance Criteria

### Phase 1 (Preprocessing Pipeline)

- [ ] `Preprocessor` struct in `src/engine/preprocess.rs` with hip centering, scale normalization, and velocity features
- [ ] Each preprocessing stage independently toggleable via `PreprocessingConfig`
- [ ] MediaPipe Pose joint index mapping with graceful fallback for unknown tracking systems
- [ ] Same preprocessing applied to both training recording path and recognition path
- [ ] Stored examples preprocessed at vocabulary load time
- [ ] Thresholds auto-recompute when pipeline config changes
- [ ] Format version bumped to 1.2 with `preprocessing` field
- [ ] v1.0/v1.1 vocabularies load with preprocessing defaulting to OFF (backward compatible)
- [ ] Test with existing test vocabulary (`test_data/26-01-30-0035_test-vocabulary.ralf`) — loads and recognizes correctly
- [ ] No regression: recognition still works with all preprocessing toggles OFF (matches v0.7.0 behavior exactly)

### Phase 2 (Data Quality)

- [ ] `augment_examples()` in `src/engine/augmentation.rs` producing temporal stretches, spatial jitter, and mirrors
- [ ] Augmented examples are ephemeral (not stored in .ralf files)
- [ ] Deterministic augmentation using stored seed
- [ ] `assess_example()` in `src/engine/quality.rs` detecting too-still, outlier, and too-short recordings
- [ ] Quality feedback displayed post-training-session as non-blocking summary
- [ ] Augmented examples used for DTW comparison but NOT for threshold statistics

### Phase 3 (Recognition Refinements)

- [ ] Variance-based joint weights computed per-gesture from training examples
- [ ] Weighted Euclidean distance option in DTW
- [ ] Joint weights preserved through LB_Keogh pruning (scale data, not envelope)
- [ ] Consensus scoring as optional per-gesture gate (default OFF)
- [ ] UI toggles for Phase 3 features in a settings panel (not visible during training flow)

### Cross-Cutting

- [ ] All features compile with `cargo build` and pass `cargo test`
- [ ] `cargo clippy` clean
- [ ] No regressions in existing recognition tests
- [ ] CPU budget validated: preprocessing + augmented examples at 15Hz DTW rate with 10 gestures x 12 examples each

## Success Metrics

| Metric | Current (v0.7.0) | Target | How to Measure |
|--------|-------------------|--------|----------------|
| Echo rate | 0.0% | 0.0% | Diagnostic logging in performance session |
| Hit margin | 95%+ | 95%+ | Distance / threshold ratio in diagnostic logs |
| Recognition latency | ~200ms (3 frames) | <= 250ms | Frame count from distance drop to HIT log |
| False positive rate | ~0% (3 gestures) | < 5% at 10 gestures | Intentional non-gesture movements during testing |
| Training examples needed | 6+ per gesture | 4-5 per gesture (with augmentation) | User testing |
| Position sensitivity | High (dancer location matters) | Low (hip-centered) | Train in one spot, test in another |
| CPU utilization at 15Hz DTW | ~2% (3 gestures) | < 20% at 10 gestures | System monitor during performance mode |

## Dependencies & Risks

### Dependencies

- **#11 frame dimension validation** — prerequisite, already committed (`c1dcbeb`)
- **#13 preprocessing pipeline architecture** — Phase 1 is the first implementation; align with the `FramePreprocessor` trait design in #13
- Phase 2 depends on Phase 1 (augmentation should operate on preprocessed data)
- Phase 3b (DDTW) is an alternative to Phase 1's velocity features, not a complement — choose one
- Threshold recomputation depends on the preprocessing pipeline being stable

### Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Preprocessing changes distance scale, breaks existing trained gestures | Users must retrain or accept auto-recalibrated thresholds | Auto-recompute thresholds; keep preprocessing OFF by default for loaded v1.1 files |
| Augmentation loosens gesture boundaries, increasing inter-gesture confusion | False positives between similar gestures | Don't use augmented examples for threshold computation; validate inter-gesture separation |
| Velocity features amplify tracking noise | Jittery distances, unstable recognition | Velocity computed from 60fps data (smooth); can be disabled independently |
| CPU budget exceeded at scale | DTW rate drops below 15Hz, causing lag | Benchmark at 10 gestures before committing to augmentation multiplier; LB_Keogh prunes effectively |
| Shoulder width normalization fails with bad tracking | Division by near-zero, NaN propagation | Clamp minimum shoulder width to 0.01; log warning |

## References & Research

### Internal References

- Research report: `/Users/brandon/Desktop/gesture_recognition_research_report.md`
- DTW implementation: `src/engine/dtw.rs`
- Recognizer state machine: `src/engine/recognizer.rs`
- Existing optimization plan: `docs/plans/2026-01-30-recognition-optimization-v2-plan.md`
- Raw learnings: `RAW_LEARNINGS.md`
- Gesture recognition learnings: `.llm/gesture-recognition-learnings.md`
- Test vocabulary: `test_data/26-01-30-0035_test-vocabulary.ralf`

### External References

- Derivative DTW: Keogh & Pazzani (2001), "Derivative Dynamic Time Warping"
- Weighted DTW: Celebi et al. (2013), joint discriminant power weighting
- Skeleton normalization: NTU RGB+D dataset preprocessing standard
- Data augmentation for skeleton: Wang et al. (2024), Taylor-transformed skeletons
- FastDTW debunked: Keogh et al. (2020), "FastDTW is Approximate and Generally Slower than Constrained DTW"
- Sakoe-Chiba: Sakoe & Chiba (1978), dynamic programming optimization for time warping
- GRT statistical threshold: Gesture Recognition Toolkit by Nick Gillian
