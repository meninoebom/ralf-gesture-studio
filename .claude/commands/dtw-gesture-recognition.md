# DTW Gesture Recognition

Implement real-time gesture recognition using Dynamic Time Warping (DTW). This approach is proven to work for continuous gesture recognition from streaming sensor data (skeleton tracking, motion capture, etc.).

## Reference Implementation

**Source**: [fiebrink1/wekinator](https://github.com/fiebrink1/wekinator)
**Key File**: `src/wekimini/learning/dtw/DtwModel.java`
**Key Method**: `classifyContinuous()` (lines 920-970)

## The Algorithm

### Core Loop (Pseudocode)

```
for each incoming frame:
    1. Add frame to sliding window buffer
    2. Skip frames for performance (e.g., process every 4th frame)
    3. Get window of size N (where N = first training example's length)
    4. Downsample window for faster comparison

    for each gesture:
        best_distance = infinity
        for each training example:
            example_downsampled = downsample(example)
            distance = dtw_distance(window, example_downsampled)
            best_distance = min(best_distance, distance)

        gesture.current_distance = best_distance

        # Re-arm when distance goes above threshold
        if best_distance >= threshold:
            gesture.arm()

        # Fire hit if: below threshold AND armed AND not in cooldown
        if best_distance < threshold AND gesture.is_armed AND not gesture.in_cooldown:
            gesture.record_hit()  # Also disarms
            emit_hit(gesture)
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Window size | Fixed (first example's length) | Simple, predictable |
| Example comparison | All examples, not prototype | Captures full variance |
| Threshold check | Simple `distance < threshold` | Easy to understand, robust |
| Double-hit prevention | Armed/disarmed state | Must "release" before next hit |
| Performance | Frame skip + downsample | ~64x faster than naive |

## Critical Components

### 1. Sliding Window Buffer

Store recent frames in a circular buffer. Window size is determined by the first training example's length.

```
Buffer capacity: window_size (e.g., 180 frames at 60fps = 3 seconds)
On each frame: push new frame, oldest drops off automatically
```

### 2. DTW Distance Function

Standard Dynamic Time Warping with Euclidean distance:

```
function dtw_distance(seq1, seq2):
    m = length(seq1)
    n = length(seq2)

    # Initialize cost matrix with infinity
    dtw = matrix[m+1][n+1] filled with infinity
    dtw[0][0] = 0

    for i = 1 to m:
        for j = 1 to n:
            cost = euclidean_distance(seq1[i-1], seq2[j-1])
            dtw[i][j] = cost + min(
                dtw[i-1][j],     # insertion
                dtw[i][j-1],     # deletion
                dtw[i-1][j-1]    # match
            )

    return dtw[m][n]
```

### 2a. Z-Normalization (Scale/Offset Invariance)

**What:** Normalize each sequence independently to mean=0, std=1 before DTW comparison.

**Formula:** `z = (x - mean) / std`

**Why it matters:**
- Makes DTW **scale-invariant**: 0-1 normalized coords vs 0-640 pixel coords produce same distances
- Makes DTW **offset-invariant**: Baseline shifts (person standing in different position) don't affect matching
- Each sequence uses its OWN mean/std (not a global normalization across all sequences)

**Rust implementation:**
```rust
fn z_normalize(sequence: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let all_values: Vec<f32> = sequence.iter().flatten().copied().collect();
    let n = all_values.len() as f32;
    let mean = all_values.iter().sum::<f32>() / n;
    let variance = all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    // Edge case: constant sequence (avoid division by zero)
    let std = if std < 1e-10 { 1.0 } else { std };

    sequence.iter()
        .map(|frame| frame.iter().map(|x| (x - mean) / std).collect())
        .collect()
}
```

**Edge case:** If std=0 (constant sequence, e.g., standing perfectly still), set std=1 to avoid NaN.

**Reference implementations:** scipy.stats.zscore, tslearn.TimeSeriesScalerMeanVariance, dtaidistance.preprocessing.znormal

### 2b. Path-Length Normalization (Duration Invariance)

**What:** Divide the raw DTW distance by (n + m) where n and m are the sequence lengths.

**Formula:** `normalized_distance = raw_distance / (n + m)`

**Why it matters:**
- Raw DTW distance scales with sequence length (longer gestures = higher distances)
- Normalization makes distances **comparable across different gesture durations**
- A 2-second wave and a 5-second wave can have the same normalized distance if equally well-matched

**Important:** Most DTW libraries do NOT apply this automatically. You must do it yourself.

**Rust implementation:**
```rust
fn dtw_distance_normalized(seq1: &[Vec<f32>], seq2: &[Vec<f32>]) -> f32 {
    let raw_distance = dtw_distance(seq1, seq2);
    let path_length = (seq1.len() + seq2.len()) as f32;
    raw_distance / path_length
}
```

**Note:** For asymmetric step patterns, divide by n only (query length). The (n + m) normalization is recommended by Sakoe-Chiba (1978) for symmetric2 step pattern.

### 2c. Combined Preprocessing Pipeline

For robust gesture matching, apply both normalizations:

```rust
fn compare_gestures(window: &[Vec<f32>], example: &[Vec<f32>]) -> f32 {
    // 1. Z-normalize each sequence independently
    let window_normalized = z_normalize(window);
    let example_normalized = z_normalize(example);

    // 2. Compute DTW distance
    let raw_distance = dtw_distance(&window_normalized, &example_normalized);

    // 3. Path-length normalize
    let path_length = (window.len() + example.len()) as f32;
    raw_distance / path_length
}
```

**Combined effect:** Distances become comparable regardless of:
- Input coordinate scale (pixels vs normalized 0-1)
- Baseline offset (person standing left vs right)
- Gesture duration (fast vs slow performance)

### 3. Performance Optimizations (ESSENTIAL)

**Frame Skipping** - Only compute DTW every Nth frame:
```
if frame_count % skip_factor != 0:
    return  # Skip this frame

# For 60fps input, skip_factor=4 gives 15 DTW/second
```

**Downsampling** - Compare shorter sequences:
```
function downsample(sequence, factor):
    return [sequence[i] for i in range(0, len(sequence), factor)]

# 180 frames -> 45 frames (factor=4)
# DTW matrix: 180x180 = 32,400 cells -> 45x45 = 2,025 cells
```

**Combined Impact**:
- Before: 60 DTW/sec × 32,400 cells = 1.9M operations/sec
- After: 15 DTW/sec × 2,025 cells = 30K operations/sec
- **Speedup: ~64x**

### 4. Double-Hit Prevention (Armed State)

**The Problem**: After a hit fires, the distance might stay below threshold while gesture completes. When cooldown expires, another hit fires even though user did one gesture.

**The Solution**:
```
struct GestureState:
    armed: bool  # Must go above threshold to re-arm
    last_hit_time: timestamp

# In processing loop:
if distance >= threshold:
    gesture.arm()  # Re-arm when distance goes up

if distance < threshold AND gesture.is_armed AND not in_cooldown:
    gesture.record_hit()  # This also disarms
    fire_hit()
```

**State Flow**:
```
1. Idle → distance high → armed
2. Gesture starts → distance drops → still armed
3. Distance < threshold → HIT fires → disarmed
4. Gesture continues → distance still low → can't fire (disarmed)
5. Gesture ends → distance rises above threshold → re-armed
6. Ready for next gesture
```

### 5. Cooldown

Minimum time between hits for the same gesture (e.g., 500ms). Prevents rapid-fire during sustained low distances.

```
function in_cooldown(gesture, cooldown_duration):
    if gesture.last_hit_time is None:
        return false
    return (now - gesture.last_hit_time) < cooldown_duration
```

## What NOT To Do (Failed Approaches)

### Edge Detection (FAILED)
- **Idea**: Fire only when distance *crosses* below threshold
- **Problem**: Noisy distance values cause spurious crossings
- **Lesson**: Simple threshold is more robust

### Prototype Averaging (FAILED)
- **Idea**: Average all examples into one prototype, compare against that
- **Problem**: Loses variance information, doesn't match real gestures
- **Lesson**: Compare against all examples individually

### Multiple Candidate Windows (FAILED)
- **Idea**: Try many window sizes based on min/max example lengths
- **Problem**: Added complexity without clear benefit
- **Lesson**: Fixed window size (from first example) is sufficient

### Activity Gate (FAILED)
- **Idea**: Skip DTW when motion energy is low (standing still)
- **Problem**: Added another threshold to tune, didn't help much
- **Lesson**: The main threshold handles idle state naturally

### No Performance Optimization (FAILED)
- **Idea**: Run DTW at full frame rate on full-length sequences
- **Problem**: CPU overload, distances get "stuck"
- **Lesson**: Frame skipping + downsampling is essential

## Threshold Calibration

Distance scale depends on:
- Number of dimensions (e.g., 68 for skeleton = 34 joints × 2 coordinates)
- Magnitude of values
- Gesture duration (more frames = higher cumulative distance)

**Manual Calibration Process**:
1. Start with threshold at maximum
2. Perform gesture repeatedly
3. Watch the distance values
4. Lower threshold until gestures trigger reliably
5. Raise slightly to reduce false positives

**Typical Values** (skeleton data, 68 dimensions):
- Threshold ~8000 worked well
- Default: user-adjustable slider

### Statistical Threshold: The μ+σ Approach (GRT Method)

**Source**: [Gesture Recognition Toolkit (GRT)](https://github.com/nickgillian/grt) — battle-tested C++ library for real-time gesture recognition.

**The key insight**: Each gesture automatically gets its own threshold based on its natural variability during training. Gestures with more variation get looser thresholds; precise gestures get tighter thresholds.

**How it works**:

1. **During training**, for each gesture class:
   - Compute DTW distances between all training examples
   - Find the "best template" (example with lowest average distance to others)
   - Calculate mean (μ) and standard deviation (σ) of distances to best template

2. **Set per-gesture threshold**:
   ```
   threshold = μ + (σ × null_rejection_coeff)
   ```

3. **Default null_rejection_coeff = 3.0** (allows matches within 3 sigma of training mean)

**Implementation pattern**:

```rust
struct GestureTemplate {
    examples: Vec<Sequence>,
    best_template: Sequence,  // Example with lowest avg distance to others
    training_mu: f32,         // Mean distance during training
    training_sigma: f32,      // Std dev of distances during training
    threshold: f32,           // mu + sigma * coefficient
}

fn compute_threshold(examples: &[Sequence], null_rejection_coeff: f32) -> ThresholdStats {
    // 1. Compute all pairwise DTW distances
    let mut all_distances: Vec<Vec<f32>> = Vec::new();
    for (i, ex_i) in examples.iter().enumerate() {
        let distances: Vec<f32> = examples.iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, ex_j)| dtw_distance(ex_i, ex_j))
            .collect();
        all_distances.push(distances);
    }

    // 2. Find best template (lowest average distance to others)
    let avg_distances: Vec<f32> = all_distances.iter()
        .map(|d| d.iter().sum::<f32>() / d.len() as f32)
        .collect();
    let best_idx = avg_distances.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    // 3. Compute mu and sigma from distances to best template
    let distances_to_best: Vec<f32> = examples.iter()
        .enumerate()
        .filter(|(i, _)| *i != best_idx)
        .map(|(_, ex)| dtw_distance(&examples[best_idx], ex))
        .collect();

    let mu = distances_to_best.iter().sum::<f32>() / distances_to_best.len() as f32;
    let variance = distances_to_best.iter()
        .map(|d| (d - mu).powi(2))
        .sum::<f32>() / distances_to_best.len() as f32;
    let sigma = variance.sqrt();

    // 4. Compute threshold
    let threshold = mu + sigma * null_rejection_coeff;

    ThresholdStats { best_template: examples[best_idx].clone(), mu, sigma, threshold }
}
```

**Parameters**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `null_rejection_coeff` | 3.0 | Higher = more permissive (fewer false negatives), Lower = stricter (fewer false positives) |

**Why this is better than manual thresholds**:
- **No per-gesture tuning**: One global `null_rejection_coeff` works for all gestures
- **Adapts to complexity**: Simple gestures (clap) get tight thresholds; complex gestures (dance phrase) get looser thresholds
- **Battle-tested**: Used in production systems for years
- **Automatic recalibration**: Threshold updates when you add/remove training examples

**Minimum examples required**: At least 3 examples per gesture to compute meaningful statistics.

### Optional Enhancement: Winner-Take-All Hybrid

For multi-gesture systems, add a secondary check using likelihood scores:

```rust
fn classify_with_likelihood(
    window: &Sequence,
    gestures: &[GestureTemplate],
) -> Option<usize> {
    // 1. Compute distances to all gestures
    let distances: Vec<f32> = gestures.iter()
        .map(|g| dtw_distance(window, &g.best_template))
        .collect();

    // 2. Find best match
    let (best_idx, best_distance) = distances.iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap();

    // 3. Check threshold (primary gate)
    if *best_distance > gestures[best_idx].threshold {
        return None;  // No match
    }

    // 4. Compute likelihood scores (1/distance, normalized)
    let inv_distances: Vec<f32> = distances.iter()
        .map(|d| 1.0 / (d + 1e-6))  // Avoid division by zero
        .collect();
    let sum: f32 = inv_distances.iter().sum();
    let likelihoods: Vec<f32> = inv_distances.iter()
        .map(|d| d / sum)
        .collect();

    // 5. Winner must have >50% likelihood
    if likelihoods[best_idx] > 0.5 {
        Some(best_idx)
    } else {
        None  // Ambiguous match
    }
}
```

**Why dual protection helps**:
- Threshold alone might accept a gesture that's close to two different templates
- Likelihood check ensures the best match is "clearly" the best, not just marginally better
- Reduces false positives in multi-gesture vocabularies

## Data Structures

```
Vocabulary
├── name: string
├── gestures: Gesture[]
└── config (input/output settings)

Gesture
├── id: int
├── name: string
├── examples: Example[]
├── threshold: float (user-adjustable)
└── osc_address: string (output)

Example
├── frames: Frame[]  # Full frame rate (60fps)
├── duration_ms: int
└── recorded_at: timestamp

Frame = float[]  # Vector of feature values

GestureState (runtime)
├── gesture reference
├── current_distance: float
├── armed: bool
├── last_hit_time: timestamp
└── downsampled_examples: cached
```

## Default Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Window size | First example's length | ~180 frames at 60fps |
| Frame skip | 4 | 15 DTW computations/sec |
| Downsample | 4 | Compare ~45 frame sequences |
| Cooldown | 500ms | Adjust per use case |
| Threshold | User-adjustable | ~8000 for skeleton data |

## Implementation Checklist

- [ ] Sliding window buffer (circular, fixed size)
- [ ] Standard DTW distance function
- [ ] Frame skipping (every Nth frame)
- [ ] Downsampling for both window and examples
- [ ] Compare against ALL training examples
- [ ] Track best (minimum) distance per gesture
- [ ] Armed/disarmed state for double-hit prevention
- [ ] Cooldown timer per gesture
- [ ] Real-time distance display (debugging)
- [ ] Threshold slider (user-adjustable)

## Testing Strategy

1. **Unit test DTW**: Known sequences with expected distances
2. **Unit test buffer**: Verify circular behavior, correct window extraction
3. **Integration test**: Record example, verify recognition triggers
4. **Manual test**: Train gesture, switch to performance mode, verify single clean hits

---

## VAD-Style State Machine Recognition (v0.5.0) ✅ PRODUCTION READY

**Breakthrough 2026-01-29**: After multiple complex approaches failed, a VAD (Voice Activity Detection) inspired state machine proved to be the winning pattern. This approach is borrowed from speech recognition systems like CMU Sphinx, Kaldi, and WebRTC VAD.

### The Problem with Previous Approaches

| Approach | Problem |
|----------|---------|
| **Simple threshold crossing** | Echo hits when distance stays below threshold |
| **Peak detection** | Latency waiting for rise, misses flat-bottom gestures |
| **Distance-based re-arming** | Fails when resting distance is still close to threshold |
| **Adaptive threshold** | Over-complicated, hard to tune, didn't solve root issues |

### The Solution: VAD-Style State Machine

A state machine with **frame accumulation** for confirmation and **time-based hangover** for echo prevention.

```
                    ┌─────────────┐
                    │    IDLE     │◄──────────────────┐
                    │  (armed)    │                   │
                    └──────┬──────┘                   │
                           │                          │
          distance < threshold                        │
                           │                          │
                           ▼                          │
                    ┌─────────────┐                   │
                    │  BUILDING   │                   │
                    │ (accumulate)│                   │
                    └──────┬──────┘                   │
                           │                          │
          accumulated >= 3 frames (~200ms)            │
                           │                          │
                           ▼                          │
              ┌────────────────────────┐              │
              │         PEAK           │              │
              │  *** FIRE GESTURE ***  │              │
              └───────────┬────────────┘              │
                          │                           │
              immediately transition                  │
                          │                           │
                          ▼                           │
                   ┌─────────────┐                    │
                   │  RECOVERY   │                    │
                   │ (hangover)  │────────────────────┘
                   └─────────────┘
                   after hangover_ms (300ms)
```

### ⚠️ CRITICAL INSIGHT: Recovery Must Be Time-Based Only

**The #1 learning**: Recovery MUST exit based on time alone, NOT distance.

**Why distance-based recovery fails**:
- With body tracking, "resting" distance is often still close to threshold
- Example from real use: threshold=17, resting distance=21-24
- If exit threshold was 1.5× = 25.5, user barely exceeds it
- Earlier bug: exit at 1.5× threshold caused stuck recognition (one hit, never re-arms)

**What works**:
```rust
RecognitionState::Recovery => {
    let hangover_complete = self.recovery_start
        .map(|t| t.elapsed() >= hangover)
        .unwrap_or(true);

    // Exit recovery when hangover is complete (TIME-BASED ONLY)
    // Do NOT check distance here - user may still be close to gesture zone
    if hangover_complete {
        self.reset_to_idle();
    }
    false
}
```

### Why This Works

| Mechanism | Purpose | Why It Works |
|-----------|---------|--------------|
| **Frame accumulation** | Prevents noise spikes from firing | 3 frames = ~200ms of consistent low distance |
| **Time-based hangover** | Prevents echo hits | Blocks new detections regardless of distance |
| **No hysteresis exit** | Prevents stuck recognition | Works even when resting distance ≈ threshold |
| **Simple state machine** | Predictable behavior | Easy to debug, no complex armed/disarmed logic |

### Production Configuration

```rust
pub struct RecognitionConfig {
    pub cooldown_ms: 500,              // Backup protection (rarely used)
    pub threshold_high_factor: 1.0,    // Entry at 100% of threshold
    pub frames_to_fire: 3,             // ~200ms confirmation at 15Hz DTW
    pub hangover_ms: 300,              // 300ms recovery before re-arming
}
```

### Real-World Results (2026-01-29)

Testing with "wings" gesture (lifting both arms):
- **7 HITs, 0 false positives, 0 echo**
- Threshold: 17 (AUTO from μ+σ)
- Resting distance: ~21-24
- Gesture distance: ~14-15 when performing
- Each HIT followed by ~300ms recovery, then re-armed

### Log Pattern for Successful Recognition

```
# Pattern: Resting → Gesture → HIT → Recovery → Resting → Ready
timestamp,REC,frame,buffer,window,wings:68:27:1
timestamp,REC,frame,buffer,window,wings:65:27:1
timestamp,REC,frame,buffer,window,wings:15:27:1  # Distance drops
timestamp,REC,frame,buffer,window,wings:14:27:1
timestamp,REC,frame,buffer,window,wings:14:27:1  # Building (3 frames)
timestamp,HIT,frame,wings,14.2,27.0,47%          # FIRE!
timestamp,REC,frame,buffer,window,wings:15:27:0:in_cooldown  # Recovery
# ... 300ms later ...
timestamp,REC,frame,buffer,window,wings:68:27:1  # Re-armed, ready
```

### Implementation (from recognizer.rs)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionState {
    Idle,      // Waiting for gesture, ready to detect
    Building,  // Distance below threshold, accumulating frames
    Peak,      // Gesture detected, fire OSC
    Recovery,  // Hangover period, blocking new detections
}

fn process_state_machine(&mut self, distance: f32, config: &RecognitionConfig) -> bool {
    let entry_threshold = self.threshold * config.threshold_high_factor;
    let hangover = Duration::from_millis(config.hangover_ms);

    match self.state {
        RecognitionState::Idle => {
            if distance < entry_threshold {
                self.state = RecognitionState::Building;
                self.frames_below_threshold = 1;
                // Check for immediate fire if frames_to_fire = 1
                if self.frames_below_threshold >= config.frames_to_fire {
                    self.state = RecognitionState::Peak;
                    self.record_hit();
                    return true;
                }
            }
            false
        }

        RecognitionState::Building => {
            if distance < entry_threshold {
                self.frames_below_threshold += 1;
                if self.frames_below_threshold >= config.frames_to_fire {
                    self.state = RecognitionState::Peak;
                    self.record_hit();
                    return true;
                }
            } else {
                self.reset_to_idle(); // Distance rose, reset
            }
            false
        }

        RecognitionState::Peak => {
            // Immediately transition to Recovery
            self.state = RecognitionState::Recovery;
            self.recovery_start = Some(Instant::now());
            false
        }

        RecognitionState::Recovery => {
            let hangover_complete = self.recovery_start
                .map(|t| t.elapsed() >= hangover)
                .unwrap_or(true);
            // TIME-BASED ONLY - do not check distance!
            if hangover_complete {
                self.reset_to_idle();
            }
            false
        }
    }
}
```

### What We Removed (Simplification)

| Removed | Why |
|---------|-----|
| `AdaptiveThreshold` | Computed from wrong data, over-complicated |
| Motion gate | Blocked recognition when it shouldn't |
| Calibration step | Not needed with statistical threshold |
| Armed/disarmed tracking | State machine handles this implicitly |
| Peak detection (local minima) | Frame accumulation is simpler and works |
| Distance-based re-arming | Fails with body tracking data |

**Result**: ~500 lines → ~350 lines, better results.

---

## DEPRECATED: Peak Detection + Sustained Detection (v0.4.0)

> **Note**: This approach was superseded by the VAD-style state machine (v0.5.0).
> Kept for historical reference only.

<details>
<summary>Click to expand deprecated approach</summary>

The simple threshold approach works but has issues with **continuous gesture performance** where the user performs gestures without returning to a resting state. This section documents the more sophisticated approach developed for RALF Gesture Studio.

### The Problem with Simple Threshold

When a user performs gestures continuously:
1. Distance drops below threshold → HIT fires
2. Distance stays below threshold during gesture
3. Cooldown expires while still below threshold
4. Another HIT fires (echo)
5. Repeat until user stops

**Result**: 3-5 echo hits following each intentional gesture.

### Solution: Dual Detection Strategy

Instead of firing when `distance < threshold`, use two complementary detection methods:

#### 1. Peak Detection (Primary)

Fire when distance reaches a **local minimum** (the gesture's "best match" point).

```rust
// Track distance history
let distance_history: VecDeque<(f32, usize)> = VecDeque::with_capacity(5);

// Detect descending→ascending pattern (local minimum)
let (prev_dist, _) = distance_history.get(history.len() - 2);
let (curr_dist, _) = distance_history.back();

let was_descending = prev_dist > prev_prev_dist;  // Going down
let now_ascending = curr_dist > prev_dist;         // Now going up

if was_descending && now_ascending && recognition_armed {
    if prev_dist < threshold && !in_cooldown {
        // Fire at the minimum point!
        fire_hit(gesture, prev_dist);
        recognition_armed = false;
    }
}
```

**Why this works**: Peak detection fires at the moment of best match, not just any time below threshold.

#### 2. Sustained Detection (Backup)

For continuous gestures that don't produce clear peaks, fire after N frames below threshold.

```rust
const MAX_SUSTAIN_FRAMES: usize = 8;

if best_distance < threshold {
    frames_below_threshold += 1;
    if frames_below_threshold >= MAX_SUSTAIN_FRAMES && recognition_armed {
        // Held below threshold long enough - fire!
        fire_hit(gesture, best_distance);
        recognition_armed = false;
        frames_below_threshold = 0;
    }
} else {
    frames_below_threshold = 0;  // Reset counter
}
```

**Why this works**: If distance stays low without a clear peak, we eventually fire anyway.

### Unified Armed State

**Critical**: Both detection methods share ONE armed state to prevent echoes.

After ANY hit (peak or sustained), set `recognition_armed = false`. This prevents:
- Peak detection from firing, then sustained detection firing
- Multiple peak detections from tiny fluctuations

### Re-arming Logic (The Hard Part)

Two re-arming paths:

#### Path 1: Distance-Based Re-arming

```rust
// Re-arm when distance goes above 75% of threshold (hysteresis)
let rearm_threshold = gesture_threshold * 0.75;
if best_distance > rearm_threshold {
    recognition_armed = true;
}
```

#### Path 2: Time-Based Re-arming (for continuous gestures)

```rust
// Re-arm after 2× cooldown time regardless of distance
if let Some(fire_time) = last_fire_time {
    let rearm_delay = Duration::from_millis(cooldown_ms * 2);
    if fire_time.elapsed() > rearm_delay {
        recognition_armed = true;
    }
}
```

**Known issue**: Time-based re-arming can cause echo hits at regular intervals.

</details>

---

## Tuning Workflow

When recognition isn't working well:

1. **Enable diagnostic logging** to a file
2. **Perform gestures** while watching the log
3. **Analyze patterns**:
   - Are HITs at correct times? → Check peak detection
   - Echo hits? → Check re-arming logic
   - Missed hits? → Check armed state and threshold
4. **Extract metrics** from logs:
   - Latency: Time between gesture start and HIT
   - Echo rate: % of HITs followed by echo within 2 seconds
   - False positive rate: HITs with no corresponding gesture
5. **Adjust one parameter at a time** and retest

### Key Metrics

| Metric | Target | How to Measure |
|--------|--------|----------------|
| Latency | <200ms | Time from distance dip to HIT |
| Echo rate | <5% | HITs within 2s of previous HIT |
| Accuracy | >90% | Correct HITs / total gestures |
| False positive | <10% | Spurious HITs / total HITs |

---

Use this approach as the baseline for any gesture recognition implementation. The simplicity of the basic approach is intentional - more complex approaches consistently failed while this Wekinator-style approach works reliably. The advanced dual-detection approach builds on this foundation for continuous gesture performance scenarios.
