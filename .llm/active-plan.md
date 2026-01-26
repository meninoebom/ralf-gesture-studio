# RALF Gesture Studio - Implementation Plan

**Current Version**: v0.1.0 (Complete)
**Next Version**: v0.2.0 (Wekinator-Style Recognition)

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

---

# v0.2.0: Wekinator-Style DTW Recognition

**Goal**: Match the proven recognition approach from Wekinator for reliable gesture detection.

**Reference Implementation**: [fiebrink1/wekinator](https://github.com/fiebrink1/wekinator) - `src/wekimini/learning/dtw/DtwModel.java`

---

## Wekinator's DTW Model (Reference)

Based on analysis of Wekinator's source code (DtwModel.java), here's how their gesture recognition works:

### Core Algorithm (from DtwModel.java lines 920-970)

```java
// Wekinator's classifyContinuous() method - the heart of their DTW recognition

// 1. Determine candidate window sizes from training examples
int min = settings.getDownsamplePolicy() == NO_DOWNSAMPLING
    ? minSizeInExamples
    : minSizeInDownsampledExamples;
int max = settings.getDownsamplePolicy() == NO_DOWNSAMPLING
    ? maxSizeInExamples
    : maxSizeInDownsampledExamples;

// 2. Generate multiple candidate windows of varying lengths
List<TimeSeries> candidates = data.getCandidateSeriesFromCurrentRun(min, max, hopSize);

// 3. Compare each candidate against ALL training examples (not just prototype)
double closestDist = Double.MAX_VALUE;
int closestClass = -1;

for (int whichClass = 0; whichClass < numGestures; whichClass++) {
    if (isGestureActive[whichClass]) {
        for (TimeSeries candidate : candidates) {
            for (DtwExample ex : data.getExamplesForGesture(whichClass)) {
                TimeSeries ts = settings.getDownsamplePolicy() == NO_DOWNSAMPLING
                    ? ex.getTimeSeries()
                    : ex.getDownsampledTimeSeries();

                double dist = FastDTW.getWarpDistBetween(ts, candidate, matchWidth, distanceFunction);

                if (closestDistances[whichClass] > dist) {
                    closestDistances[whichClass] = dist;
                }
                if (dist < closestDist) {
                    closestDist = dist;
                    closestClass = whichClass;
                }
            }
        }
    }
}

// 4. Simple threshold check - this is the key insight!
if (closestDist < matchThreshold) {
    setCurrentMatch(closestClass);  // Fire the gesture
} else {
    setCurrentMatch(-1);  // No match / idle state
}
```

### Key Design Decisions

| Aspect | Wekinator's Approach | Rationale |
|--------|---------------------|-----------|
| **Threshold Type** | Absolute threshold (fire when below) | Simple, predictable behavior |
| **Window Size** | Multiple candidates based on example lengths | Handles gestures of varying duration |
| **Example Matching** | Compare against ALL examples | Captures full variance of training data |
| **No-Match State** | Explicit `-1` when nothing matches | Clear idle state, no false positives |
| **Default Threshold** | `matchThreshold = 5` | Low default, user can increase |

### Source Code Locations (for future reference)

- **Main model**: `src/wekimini/learning/dtw/DtwModel.java`
- **Threshold declaration**: Line 85 - `private double matchThreshold = 5;`
- **Classification logic**: Lines 920-970 - `classifyContinuous()` method
- **Match state management**: Lines 510-520 - `setCurrentMatch()` method
- **Candidate generation**: Via `data.getCandidateSeriesFromCurrentRun(min, max, hopSize)`

---

## Phase 1: Wekinator-Style Recognition ✅ COMPLETE

**Completed**: 2026-01-26

### Breakthrough: It Works!

After several iterations, the simple approach works. Key learnings:

1. **Simplicity wins** - The complex version (candidate windows, prototypes, activity gates) didn't work. The simple version (fixed window, compare all examples, threshold) does.

2. **Performance is critical** - DTW at 60fps with 180 frames was too slow and caused "stuck" behavior. Solution:
   - Skip frames: DTW every 4th frame (15Hz instead of 60Hz)
   - Downsample: Compare 45 frames vs 45 frames (not 180 vs 180)
   - Combined: ~64x faster

3. **Threshold calibration** - User found success at threshold ~8000 (much higher than default 1500). The distance scale depends on skeleton data magnitude.

4. **The core algorithm is sound** - Wekinator's approach (sliding window + DTW + all examples + threshold) works when implemented simply.

### What We Implemented

Rewrote `src/engine/recognizer.rs` to match Wekinator's approach:

#### 1. Simple Threshold Classification (Not Edge Detection)

```rust
// OLD approach (edge detection) - REMOVED
let is_crossing_down = below_threshold && !gesture.was_below_threshold;
if is_crossing_down && not_in_cooldown { ... }

// NEW approach (Wekinator-style) - IMPLEMENTED
if best_distance < gesture.threshold && !gesture.in_cooldown(cooldown_duration) {
    gesture.record_hit();
    return Some(RecognitionResult { gesture_id: Some(gesture.id), ... });
}
// Return None gesture_id when nothing matches (idle state)
```

#### 2. Multiple Candidate Window Sizes

```rust
/// Generate candidate window sizes between min and max
fn generate_candidate_lengths(&self) -> Vec<usize> {
    // Find global min/max across all gestures from training examples
    let mut global_min = usize::MAX;
    let mut global_max = 0;

    for gesture in &self.gestures {
        let (min, max) = gesture.example_length_range();
        global_min = global_min.min(min);
        global_max = global_max.max(max);
    }

    // Generate evenly spaced candidate lengths (default: 5 candidates)
    // ...
}
```

#### 3. Compare Against All Examples (Not Prototype)

```rust
// For each gesture, compare against ALL training examples
for example in gesture.examples() {
    let distance = dtw_distance(&window, example);

    // Track best distance for this gesture
    if distance < gesture_distances[gesture_idx] {
        gesture_distances[gesture_idx] = distance;
    }

    // Track global best
    if distance < best_distance {
        best_distance = distance;
        best_gesture_idx = Some(gesture_idx);
    }
}
```

#### 4. Per-Gesture Example Length Tracking

```rust
pub struct GestureState {
    // ...
    examples: Vec<Sequence>,
    min_example_len: usize,  // Shortest training example
    max_example_len: usize,  // Longest training example
}

impl GestureState {
    pub fn add_example(&mut self, example: Sequence) {
        let len = example.len();
        self.examples.push(example);

        // Track min/max for candidate generation
        self.min_example_len = self.min_example_len.min(len);
        self.max_example_len = self.max_example_len.max(len);
    }
}
```

### RecognitionConfig

```rust
pub struct RecognitionConfig {
    /// Cooldown: minimum time between hits for same gesture (ms)
    pub cooldown_ms: u64,  // Default: 400

    /// Downsample factor: process every Nth frame for DTW
    pub downsample_factor: usize,  // Default: 4 (60fps → 15fps)

    /// Number of candidate windows to try between min and max lengths
    pub num_candidates: usize,  // Default: 5
}
```

### What We Removed

- Edge detection (`was_below_threshold` tracking)
- Activity gate (motion energy threshold)
- Prototype averaging (now compare against all examples)
- Sakoe-Chiba band constraint (using simpler unbounded DTW)

These can be added back in future phases if needed for performance.

---

## Phase 2: Feature Engineering

**Goal**: More robust recognition across positions and speeds.

**Status**: Not started

### Tasks

- [ ] **2.1 Hip-Centered Normalization**
  - Translate all joint positions relative to hip joint
  - Dancer can stand anywhere in frame

- [ ] **2.2 Scale Normalization**
  - Normalize skeleton size based on shoulder width or height
  - Works for dancers of different sizes

- [ ] **2.3 Velocity Features**
  - Compute first derivative (velocity) for each joint
  - Add velocity dimensions to feature vector

- [ ] **2.4 Per-Gesture Joint Weights**
  - Auto-detect important joints from training variance
  - Apply weights in DTW distance calculation

---

## Phase 3: Auto-Calibration

**Goal**: Eliminate manual threshold tuning.

**Status**: Not started

### Tasks

- [ ] **3.1 Baseline Recording UI**
  - "Record Baseline" button in Training mode
  - Record 3 seconds of neutral stance

- [ ] **3.2 Automatic Threshold Computation**
  - Compute threshold from intra-class variance and baseline distance
  - Formula: `threshold = (max_intra + 0.8 * baseline_dist) / 2 * 1.1`

- [ ] **3.3 Inter-Gesture Separation Check**
  - Warn user if new gesture is too similar to existing ones

- [ ] **3.4 Threshold Recommendation UI**
  - Show computed threshold with "recommended" indicator
  - Allow manual override

---

## Phase 4: Advanced Matching (Stretch Goals)

**Goal**: Handle complex scenarios, variable-length gestures.

**Status**: Not started

### Tasks

- [ ] **4.1 LB_Keogh Lower Bound Pruning**
- [ ] **4.2 Subsequence DTW (SPRING Algorithm)**
- [ ] **4.3 Multi-Resolution Windows**
- [ ] **4.4 Gesture Chaining Detection**

---

## Implementation Notes

### Why Wekinator's Approach Works

1. **Simple mental model**: Distance below threshold = match, above = no match
2. **Handles gesture variability**: Multiple candidate lengths accommodate timing differences
3. **Captures training variance**: Comparing against all examples catches edge cases
4. **Clear idle state**: Returning -1 when nothing matches prevents constant firing

### Differences from Our Previous Approach

| Aspect | Our V1 | Wekinator (V2) |
|--------|--------|----------------|
| Match detection | Edge detection (crossing threshold) | Simple threshold check |
| Window size | Fixed (3 seconds) | Variable (min-max from examples) |
| Example usage | Prototype (averaged) | All examples compared |
| Activity gate | Motion energy filter | None (threshold handles idle) |
| DTW constraint | Sakoe-Chiba band | Unconstrained (FastDTW) |

### Tuning Guidance

- **Threshold too high**: False positives (fires when standing still)
- **Threshold too low**: Misses valid gestures
- **Start high, lower until gestures are detected reliably**
- **Wekinator default is 5** - our default may need to be higher depending on data scale

---

*Last updated: 2026-01-25*
*Based on analysis of Wekinator's DtwModel.java*
