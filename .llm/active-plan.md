# RALF Gesture Studio - Implementation Plan

**Current Version**: v0.1.0 (Complete)
**Next Version**: v0.2.0 (Performance & Robustness)

---

## v0.1.0 Milestones (COMPLETE)

All 8 milestones from initial build are complete:

| Milestone | Status | Description |
|-----------|--------|-------------|
| 1 | ✅ | Data Model - Vocabulary/Gesture/Example structs, JSON persistence |
| 2 | ✅ | GUI Shell - eframe/egui window with panel layout |
| 3 | ✅ | OSC Receiver - Async UDP receiver with status tracking |
| 4 | ✅ | OSC Sender - Hit message output with test button |
| 5 | ✅ | DTW Algorithm - Dynamic Time Warping for gesture matching |
| 6 | ✅ | Recording + Matching - Real-time recognition with refractory period |
| 7 | ✅ | Training Session - State machine with audio cues (rodio) |
| 8 | ✅ | Polish + Performance Mode - File dialogs, threshold sliders, auto-save |

See `gesture-recognition-design.md` for deep research on architecture improvements.

---

# v0.2.0 Roadmap: Performance & Robustness

**Goal**: Make gesture recognition as seamless as voice recognition.

Based on deep research across voice recognition, gesture recognition, and time series analysis. See `.llm/gesture-recognition-design.md` for full rationale.

---

## Phase 1: Quick Wins (Performance Optimization)

**Goal**: Real-time performance without frame skipping.

**Estimated Effort**: 1-2 days

### Tasks

- [ ] **1.1 Frame Downsampling**
  - Downsample incoming 60fps to 15fps
  - Only process every 4th frame for DTW
  - Keep full-rate buffer for recording (training needs all frames)

- [ ] **1.2 Activity Gate**
  - Compute motion energy per frame (sum of squared velocities)
  - Skip DTW computation when motion energy < threshold
  - Configurable activity threshold with sensible default

- [ ] **1.3 Sakoe-Chiba Band Constraint**
  - Limit DTW warping path to diagonal band
  - Band width = 20% of sequence length
  - Reduces O(N×M) to O(N×W) where W << M

- [ ] **1.4 Prototype Averaging**
  - Compute single prototype per gesture from all examples
  - Align examples using DTW path, then average
  - Reduces N comparisons to 1 per gesture

### Verification

#### Unit Tests

```rust
// tests/phase1_performance.rs

#[test]
fn test_frame_downsampling() {
    let buffer = FrameBuffer::new(600);
    // Add 60 frames (1 second at 60fps)
    for i in 0..60 {
        buffer.push(vec![i as f32; 68]);
    }

    let downsampled = buffer.downsampled(4); // Every 4th frame
    assert_eq!(downsampled.len(), 15); // 60/4 = 15
}

#[test]
fn test_activity_gate_filters_stillness() {
    let still_frames = vec![
        ProcessedFrame { motion_energy: 0.001, .. },
        ProcessedFrame { motion_energy: 0.002, .. },
    ];
    assert!(!is_active(&still_frames, 0.01)); // Below threshold

    let moving_frames = vec![
        ProcessedFrame { motion_energy: 0.05, .. },
        ProcessedFrame { motion_energy: 0.08, .. },
    ];
    assert!(is_active(&moving_frames, 0.01)); // Above threshold
}

#[test]
fn test_sakoe_chiba_constraint() {
    let seq1 = generate_test_sequence(100);
    let seq2 = generate_test_sequence(100);

    // Constrained should be faster and produce similar result
    let unconstrained = dtw_distance(&seq1, &seq2);
    let constrained = dtw_constrained(&seq1, &seq2, 20);

    // Allow 5% error for approximation
    assert!((constrained - unconstrained).abs() / unconstrained < 0.05);
}

#[test]
fn test_prototype_matches_examples() {
    let examples = vec![
        generate_gesture_example(100),
        generate_gesture_example(105),
        generate_gesture_example(98),
    ];

    let prototype = compute_prototype(&examples);

    // Prototype should be close to all examples
    for example in &examples {
        let dist = dtw_distance(&prototype.frames, &example.frames);
        assert!(dist < 50.0); // Should be very similar
    }
}
```

#### Benchmark Tests

```rust
// benches/dtw_performance.rs

#[bench]
fn bench_dtw_unconstrained(b: &mut Bencher) {
    let seq1 = generate_test_sequence(180);
    let seq2 = generate_test_sequence(180);

    b.iter(|| dtw_distance(&seq1, &seq2));
}

#[bench]
fn bench_dtw_constrained(b: &mut Bencher) {
    let seq1 = generate_test_sequence(180);
    let seq2 = generate_test_sequence(180);

    b.iter(|| dtw_constrained(&seq1, &seq2, 36)); // 20% band
}

// Target: constrained should be 3-5x faster
```

#### Manual Verification

```bash
# 1. Run app with OSC input at 60fps
cargo run --release

# 2. Open Activity Monitor / htop
# - CPU usage should stay under 30% (was spiking to 100%+)
# - No "spinning beach ball" when switching to Performance mode

# 3. Check frame processing
# - In Performance mode, distances should update smoothly
# - No visible lag between gesture and recognition

# 4. Verify activity gate
# - Stand still for 5 seconds
# - Distance displays should show "--" or "inactive"
# - Move, distances should appear
```

#### Acceptance Criteria

| Metric | Before | Target |
|--------|--------|--------|
| CPU usage (Performance mode) | 80-100% | < 30% |
| DTW computation time | ~50ms | < 10ms |
| Frame processing lag | Visible | Imperceptible |
| Activity filtering | None | Skip 50%+ of idle time |

---

## Phase 2: Feature Engineering

**Goal**: More robust recognition across positions and speeds.

**Estimated Effort**: 2-3 days

### Tasks

- [ ] **2.1 Hip-Centered Normalization**
  - Translate all joint positions relative to hip joint
  - Dancer can stand anywhere in frame
  - Store normalized frames, not raw

- [ ] **2.2 Scale Normalization**
  - Normalize skeleton size based on shoulder width or height
  - Works for dancers of different sizes
  - Optional: configurable reference skeleton size

- [ ] **2.3 Velocity Features**
  - Compute first derivative (velocity) for each joint
  - Add velocity dimensions to feature vector
  - Motion energy = sum of squared velocities

- [ ] **2.4 Per-Gesture Joint Weights**
  - Auto-detect important joints from training variance
  - Store weights per gesture
  - Apply weights in DTW distance calculation

### Data Structures

```rust
/// Enhanced frame with derived features
struct ProcessedFrame {
    /// Original timestamp
    timestamp: Instant,

    /// Hip-centered, scale-normalized positions [68 floats]
    positions: Vec<f32>,

    /// Velocity of each joint [68 floats]
    velocities: Vec<f32>,

    /// Total motion energy (scalar)
    motion_energy: f32,
}

/// Per-gesture configuration
struct GestureConfig {
    /// Which joints matter for this gesture [34 weights, 0.0-1.0]
    joint_weights: Vec<f32>,

    /// Derived from training examples
    min_duration_frames: usize,
    max_duration_frames: usize,
    typical_duration_frames: usize,
}
```

### Verification

#### Unit Tests

```rust
// tests/phase2_features.rs

#[test]
fn test_hip_centering() {
    // Frame with hip at (100, 200)
    let raw = vec![100.0, 200.0, 150.0, 180.0, /* ... */];
    let centered = normalize_to_hip(&raw);

    // Hip should now be at origin
    assert_eq!(centered[0], 0.0);
    assert_eq!(centered[1], 0.0);

    // Other joints should be relative
    assert_eq!(centered[2], 50.0);  // Was 150, hip was 100
    assert_eq!(centered[3], -20.0); // Was 180, hip was 200
}

#[test]
fn test_scale_normalization() {
    // Same pose, different sizes
    let small = vec![0.0, 0.0, 10.0, 0.0, /* ... */]; // Shoulder at 10
    let large = vec![0.0, 0.0, 20.0, 0.0, /* ... */]; // Shoulder at 20

    let norm_small = normalize_scale(&small, SHOULDER_LEFT);
    let norm_large = normalize_scale(&large, SHOULDER_LEFT);

    // After normalization, should be identical
    assert!((norm_small[2] - norm_large[2]).abs() < 0.01);
}

#[test]
fn test_velocity_computation() {
    let prev = ProcessedFrame { positions: vec![0.0, 0.0], .. };
    let curr = ProcessedFrame { positions: vec![10.0, 5.0], .. };

    let velocities = compute_velocities(&prev, &curr, dt: 0.016);

    // Velocity = (curr - prev) / dt
    assert!((velocities[0] - 625.0).abs() < 1.0); // 10 / 0.016
}

#[test]
fn test_joint_weight_auto_detection() {
    // Create examples where only hands move
    let examples = vec![
        create_example_with_moving_hands(),
        create_example_with_moving_hands(),
    ];

    let weights = auto_detect_joint_weights(&examples);

    // Hand joints should have high weight
    assert!(weights[HAND_LEFT] > 0.8);
    assert!(weights[HAND_RIGHT] > 0.8);

    // Feet should have low weight (didn't move)
    assert!(weights[FOOT_LEFT] < 0.2);
}

#[test]
fn test_weighted_distance() {
    let a = vec![0.0, 0.0, 10.0, 10.0]; // Joints 0,1 at origin; 2,3 offset
    let b = vec![5.0, 5.0, 10.0, 10.0]; // Joints 0,1 offset; 2,3 same

    // Equal weights: both differences count
    let equal_weights = vec![1.0, 1.0, 1.0, 1.0];
    let dist_equal = weighted_distance(&a, &b, &equal_weights);

    // Zero weight on joints 0,1: only 2,3 matter (same, so dist=0)
    let focused_weights = vec![0.0, 0.0, 1.0, 1.0];
    let dist_focused = weighted_distance(&a, &b, &focused_weights);

    assert!(dist_equal > 0.0);
    assert_eq!(dist_focused, 0.0);
}
```

#### Integration Tests

```rust
#[test]
fn test_position_invariance() {
    // Train gesture at position A
    let examples_a = record_gesture_at_position(100.0, 100.0);

    // Create gesture with these examples
    let mut gesture = Gesture::new("wave");
    gesture.add_examples(examples_a);
    gesture.compute_prototype();

    // Perform same gesture at position B (different location in frame)
    let test_at_b = record_gesture_at_position(300.0, 200.0);

    // Should still match after normalization
    let distance = gesture.match_against(&test_at_b);
    assert!(distance < gesture.threshold);
}

#[test]
fn test_speed_variation_handling() {
    // Train with medium-speed gesture
    let medium_examples = record_gesture_at_speed(1.0);

    let mut gesture = Gesture::new("punch");
    gesture.add_examples(medium_examples);

    // Fast version should still match (DTW handles this)
    let fast = record_gesture_at_speed(1.5);
    let dist_fast = gesture.match_against(&fast);

    // Slow version should still match
    let slow = record_gesture_at_speed(0.7);
    let dist_slow = gesture.match_against(&slow);

    assert!(dist_fast < gesture.threshold);
    assert!(dist_slow < gesture.threshold);
}
```

#### Manual Verification

```bash
# 1. Position invariance test
# - Train a gesture standing on the left side of camera view
# - Move to right side of view
# - Perform gesture, should still be recognized

# 2. Scale invariance test
# - Train gesture standing 6 feet from camera
# - Move to 10 feet from camera (skeleton appears smaller)
# - Gesture should still be recognized

# 3. Speed variation test
# - Train "wave" at normal speed
# - Perform wave faster (1.5x speed)
# - Perform wave slower (0.5x speed)
# - Both should be recognized

# 4. Joint weight verification
# - Train a hand-only gesture (wave)
# - Check gesture config shows high weight for hands, low for feet
# - Moving feet during wave should not affect recognition
```

#### Acceptance Criteria

| Scenario | Before | Target |
|----------|--------|--------|
| Position change (±200px) | May fail | Always matches |
| Scale change (±30%) | May fail | Always matches |
| Speed variation (0.5x - 2x) | May fail | Matches within range |
| Unrelated joint movement | May cause false rejection | Ignored |

---

## Phase 3: Auto-Calibration

**Goal**: Eliminate manual threshold tuning.

**Estimated Effort**: 2-3 days

### Tasks

- [ ] **3.1 Baseline Recording UI**
  - "Record Baseline" button in Training mode
  - Record 3 seconds of neutral stance
  - Store as special baseline example in vocabulary

- [ ] **3.2 Automatic Threshold Computation**
  - After training, compute threshold from:
    - Intra-class variance (how different are examples of same gesture)
    - Baseline distance (how far is neutral from this gesture)
  - Formula: `threshold = (max_intra + 0.8 * baseline_dist) / 2 * 1.1`

- [ ] **3.3 Inter-Gesture Separation Check**
  - When training, check if new gesture is too similar to existing
  - Warn user if gestures may be confused
  - Suggest: different gesture, more examples, or adjust thresholds

- [ ] **3.4 Threshold Recommendation UI**
  - Show computed threshold with "recommended" indicator
  - Allow manual override with slider
  - Show warning if manual threshold is outside safe range

### Data Structures

```rust
/// Calibration metadata per vocabulary
struct CalibrationData {
    /// Neutral stance recording
    baseline: Option<Example>,

    /// When baseline was recorded
    baseline_recorded_at: Option<DateTime<Utc>>,
}

/// Per-gesture calibration info
struct GestureCalibration {
    /// Distance from baseline to this gesture
    baseline_distance: f32,

    /// Max distance between any two examples of this gesture
    intra_class_max: f32,

    /// Distance to nearest other gesture
    nearest_other_gesture: Option<(String, f32)>,

    /// Computed recommended threshold
    recommended_threshold: f32,

    /// Is current threshold in safe range?
    threshold_status: ThresholdStatus,
}

enum ThresholdStatus {
    /// Threshold is in recommended range
    Good,

    /// Threshold too high, may have false positives
    TooLoose { recommended: f32 },

    /// Threshold too low, may miss valid gestures
    TooStrict { recommended: f32 },

    /// Gesture overlaps with another, confusion likely
    Ambiguous { conflicting_gesture: String },
}
```

### Verification

#### Unit Tests

```rust
// tests/phase3_calibration.rs

#[test]
fn test_baseline_distance_computation() {
    let baseline = create_neutral_stance_example();
    let wave_examples = vec![
        create_wave_example(),
        create_wave_example(),
    ];

    let baseline_dist = compute_baseline_distance(&baseline, &wave_examples);

    // Wave should be far from neutral
    assert!(baseline_dist > 100.0);
}

#[test]
fn test_intra_class_variance() {
    let examples = vec![
        create_wave_example(),
        create_wave_example_variant(),
        create_wave_example(),
    ];

    let variance = compute_intra_class_max(&examples);

    // Variants of same gesture should be close
    assert!(variance < 50.0);
}

#[test]
fn test_auto_threshold_computation() {
    let baseline = create_neutral_stance_example();
    let examples = vec![create_wave_example(); 3];

    let threshold = auto_calibrate_threshold(&examples, &baseline);

    // Threshold should be between intra-class max and baseline distance
    let intra_max = compute_intra_class_max(&examples);
    let baseline_dist = compute_baseline_distance(&baseline, &examples);

    assert!(threshold > intra_max);
    assert!(threshold < baseline_dist);
}

#[test]
fn test_gesture_separation_warning() {
    let wave = Gesture::with_examples("wave", create_wave_examples());
    let similar_wave = Gesture::with_examples("wave2", create_similar_wave_examples());

    let separation = check_gesture_separation(&wave, &similar_wave);

    // Similar gestures should trigger warning
    assert!(separation.is_ambiguous());
    assert!(separation.overlap_distance < 50.0);
}

#[test]
fn test_threshold_status_detection() {
    let gesture = create_trained_gesture();

    // Threshold in good range
    gesture.threshold = gesture.recommended_threshold;
    assert_eq!(gesture.threshold_status(), ThresholdStatus::Good);

    // Threshold too high
    gesture.threshold = gesture.recommended_threshold * 2.0;
    assert!(matches!(gesture.threshold_status(), ThresholdStatus::TooLoose { .. }));

    // Threshold too low
    gesture.threshold = gesture.recommended_threshold * 0.5;
    assert!(matches!(gesture.threshold_status(), ThresholdStatus::TooStrict { .. }));
}
```

#### Integration Tests

```rust
#[test]
fn test_auto_calibration_end_to_end() {
    let mut vocab = Vocabulary::new("Test");

    // Record baseline
    vocab.record_baseline(create_neutral_frames());

    // Train gesture
    let mut gesture = vocab.add_gesture("wave");
    for _ in 0..5 {
        gesture.add_example(create_wave_frames());
    }

    // Auto-calibrate
    gesture.auto_calibrate(&vocab.baseline);

    // Test with actual gesture - should match
    let test_wave = create_wave_frames();
    let distance = gesture.match_against(&test_wave);
    assert!(distance < gesture.threshold);

    // Test with baseline - should NOT match
    let test_neutral = create_neutral_frames();
    let distance = gesture.match_against(&test_neutral);
    assert!(distance > gesture.threshold);
}
```

#### Manual Verification

```bash
# 1. Baseline recording flow
# - Open app, create new vocabulary
# - See prompt: "Record baseline (neutral stance)"
# - Click "Record Baseline", stand still for 3 seconds
# - See confirmation: "Baseline recorded"

# 2. Auto-threshold verification
# - Train a gesture (5 reps)
# - See "Recommended threshold: 125" (or similar)
# - Threshold slider should be at recommended value
# - Perform gesture - should match
# - Stand still - should NOT match

# 3. Separation warning
# - Train "wave" gesture
# - Train "wave_fast" with very similar motion
# - See warning: "wave_fast may be confused with wave"
# - Suggestions shown: "Record more varied examples" or "Adjust thresholds"

# 4. No-tuning workflow
# - Create vocabulary, record baseline
# - Train 3 gestures without touching any thresholds
# - Switch to Performance mode
# - All gestures should work correctly first try
```

#### Acceptance Criteria

| Scenario | Before | Target |
|----------|--------|--------|
| New user threshold confusion | Common | Eliminated |
| Gestures work first try | Sometimes | 90%+ of cases |
| False positive rate | Variable | < 5% |
| Ambiguous gesture warning | None | Clear warning + guidance |

---

## Phase 4: Advanced Matching (v1.0+ Stretch Goals)

**Goal**: Handle complex scenarios, variable-length gestures.

**Estimated Effort**: 1-2 weeks

### Tasks

- [ ] **4.1 LB_Keogh Lower Bound Pruning**
  - Compute cheap lower bound before full DTW
  - Skip DTW if lower bound > threshold
  - Target: prune 80% of comparisons

- [ ] **4.2 Subsequence DTW (SPRING Algorithm)**
  - Detect gestures of variable length in continuous stream
  - No fixed window size required
  - O(M) per frame instead of O(N×M)

- [ ] **4.3 Multi-Resolution Windows**
  - Support gestures of very different durations in same vocabulary
  - Short gesture (0.5s) and long gesture (5s) coexist
  - Automatic window selection based on gesture config

- [ ] **4.4 Gesture Chaining Detection**
  - Detect sequences of gestures (A → B → C)
  - Configurable timing constraints
  - Output compound gesture events

### Verification

#### Unit Tests

```rust
// tests/phase4_advanced.rs

#[test]
fn test_lb_keogh_is_lower_bound() {
    let query = generate_test_sequence(100);
    let template = generate_test_sequence(100);

    let lb = lb_keogh(&query, &template, 10);
    let actual = dtw_distance(&query, &template);

    // Lower bound must always be <= actual distance
    assert!(lb <= actual);
}

#[test]
fn test_lb_keogh_prunes_non_matches() {
    let gesture = generate_gesture_sequence();
    let unrelated = generate_completely_different_sequence();

    let lb = lb_keogh(&unrelated, &gesture, 10);
    let threshold = 100.0;

    // Lower bound should exceed threshold, allowing pruning
    assert!(lb > threshold);
}

#[test]
fn test_subsequence_dtw_finds_gesture() {
    // Long stream with gesture embedded in middle
    let mut stream = generate_noise(100);
    stream.extend(generate_gesture());
    stream.extend(generate_noise(100));

    let template = generate_gesture();

    let matches = subsequence_dtw(&stream, &template, threshold: 50.0);

    // Should find exactly one match
    assert_eq!(matches.len(), 1);

    // Match should be around frame 100-150 (where gesture was)
    assert!(matches[0].end_frame > 100);
    assert!(matches[0].end_frame < 200);
}

#[test]
fn test_variable_length_gestures() {
    let mut vocab = Vocabulary::new("Test");

    // Short gesture: 0.5 seconds (30 frames at 60fps)
    vocab.add_gesture_with_examples("snap", create_snap_examples());

    // Long gesture: 3 seconds (180 frames)
    vocab.add_gesture_with_examples("slow_wave", create_slow_wave_examples());

    // Both should be recognized from same stream
    let stream = /* stream containing both */;

    let hits = vocab.recognize(&stream);
    assert!(hits.contains("snap"));
    assert!(hits.contains("slow_wave"));
}

#[test]
fn test_gesture_chain_detection() {
    let vocab = create_vocab_with_gestures(&["punch", "kick", "spin"]);

    // Configure chain: punch -> kick within 1 second = "combo"
    vocab.add_chain("combo", vec!["punch", "kick"], max_gap_ms: 1000);

    // Stream with punch, then kick 500ms later
    let stream = create_stream_with_sequence(&[
        ("punch", 0),
        ("kick", 500),
    ]);

    let hits = vocab.recognize(&stream);

    // Should get individual hits AND combo
    assert!(hits.contains("punch"));
    assert!(hits.contains("kick"));
    assert!(hits.contains("combo"));
}
```

#### Benchmark Tests

```rust
#[bench]
fn bench_lb_keogh_pruning(b: &mut Bencher) {
    let gestures = create_gesture_vocabulary(20); // 20 gestures
    let window = generate_test_window();

    b.iter(|| {
        let mut comparisons = 0;
        for gesture in &gestures {
            let lb = lb_keogh(&window, &gesture.prototype, 10);
            if lb < gesture.threshold {
                // Would compute full DTW here
                comparisons += 1;
            }
        }
        comparisons
    });

    // Target: < 5 comparisons out of 20 (75%+ pruning)
}

#[bench]
fn bench_spring_per_frame(b: &mut Bencher) {
    let matcher = SpringMatcher::new(generate_gesture_template());

    b.iter(|| {
        let frame = generate_random_frame();
        matcher.process_frame(&frame)
    });

    // Target: < 0.5ms per frame for real-time
}
```

#### Manual Verification

```bash
# 1. Variable-length gesture test
# - Train "snap" (very short, < 0.5s)
# - Train "slow_circle" (very long, 5s)
# - Both should be recognized correctly
# - Snap should fire quickly, not wait for slow_circle window

# 2. Gesture chaining test
# - Train "punch" and "kick" separately
# - Configure combo: punch + kick within 1s
# - Perform punch, then kick quickly
# - See "combo" hit fire (in addition to individual hits)

# 3. Performance with many gestures
# - Create vocabulary with 20+ gestures
# - CPU usage should remain under 50%
# - Recognition latency should be imperceptible
```

#### Acceptance Criteria

| Metric | Before | Target |
|--------|--------|--------|
| DTW comparisons pruned | 0% | 75%+ |
| Variable-length support | Fixed 3s window | 0.5s - 10s |
| Per-frame latency | ~10ms | < 2ms |
| Gesture chains | Not supported | Working |

---

## Summary: Verification Commands

| Phase | Test Command | Benchmark | Manual Check |
|-------|--------------|-----------|--------------|
| 1 | `cargo test phase1` | `cargo bench dtw` | CPU < 30%, no lag |
| 2 | `cargo test phase2` | — | Position/speed invariance |
| 3 | `cargo test phase3` | — | No-tuning workflow |
| 4 | `cargo test phase4` | `cargo bench spring` | 20+ gestures smooth |

### Running All Tests

```bash
# Unit tests for specific phase
cargo test phase1_performance
cargo test phase2_features
cargo test phase3_calibration
cargo test phase4_advanced

# All tests
cargo test

# Benchmarks (requires nightly or criterion)
cargo bench

# Integration test with real OSC
./scripts/test_with_osc.sh
```

### CI Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo test --all-features

  bench:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - run: cargo bench --no-run  # Compile benchmarks
```

---

## Implementation Order

1. **Phase 1** first - establishes performance baseline, unblocks other work
2. **Phase 2** second - feature engineering needs Phase 1's frame processing
3. **Phase 3** third - calibration needs Phase 2's normalized features
4. **Phase 4** as stretch goals - independent optimizations

Each phase is independently valuable. Can ship after any phase.

---

## Dependencies

### New Crates for v0.2.0

| Crate | Purpose | Phase |
|-------|---------|-------|
| `criterion` | Benchmarking | 1 |
| `ndarray` | Efficient array operations | 2 |

### Existing Crates (Already in Use)

- `eframe` / `egui` - GUI
- `rosc` - OSC
- `tokio` - Async
- `rodio` - Audio
- `serde` - Serialization

---

*Last updated: 2026-01-24*
*Based on deep research in `.llm/gesture-recognition-design.md`*
