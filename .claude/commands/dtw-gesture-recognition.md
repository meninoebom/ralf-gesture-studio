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

### 3a. Sakoe-Chiba Band Constraint (ESSENTIAL)

**What:** Limit DTW warping path to a diagonal band. Prevents pathological alignments where one frame maps to dozens in the other sequence.

**Why it's critical for dance:** "Matches how dancers actually move — tempo varies, but structure doesn't." A wave might be faster or slower, but the arm still goes up before coming down. The band prevents matching "arm up" to "arm down."

```
function dtw_distance_sakoe_chiba(seq1, seq2, band_fraction=0.15):
    m = length(seq1), n = length(seq2)
    band_width = ceil(max(m, n) * band_fraction)

    dtw = matrix[m+1][n+1] filled with infinity
    dtw[0][0] = 0

    for i = 1 to m:
        j_start = max(1, i - band_width)
        j_end = min(n, i + band_width)
        for j = j_start to j_end:
            cost = euclidean_distance(seq1[i-1], seq2[j-1])
            dtw[i][j] = cost + min(dtw[i-1][j], dtw[i][j-1], dtw[i-1][j-1])

    return dtw[m][n]
```

**Impact:** O(N²) → O(N×B) where B = band_width. With 15% band, ~85% fewer cells computed. Also produces better-quality distances by preventing degenerate alignments.

**Recommended band:** 15% (`band_fraction = 0.15`). Expert review: "One of those rare 'faster and better' constraints."

### 3b. Early Abandoning Inside DTW

**What:** Track the minimum value in each row of the DTW matrix. If it exceeds the best distance found so far, stop — this candidate cannot be better.

```
function dtw_with_abandon(seq1, seq2, band_fraction, best_so_far):
    // ... same as Sakoe-Chiba above, but add:
    for i = 1 to m:
        row_min = infinity
        for j = j_start to j_end:
            // ... compute dtw[i][j] ...
            row_min = min(row_min, dtw[i][j])

        if row_min > best_so_far:
            return None  // Abandon: can't beat current best

    return Some(dtw[m][n])
```

**Impact:** "2 lines of code, 50-70% speedup." Stacks with Sakoe-Chiba band.

**Usage pattern:** Sort candidates or iterate with running best:
```
best_dist = infinity
for each example:
    result = dtw_with_abandon(window, example, 0.15, best_dist)
    if result is Some(dist) and dist < best_dist:
        best_dist = dist
```

### 3c. LB_Keogh Pruning (Optional, Medium Effort)

**What:** Compute an O(N) lower bound on DTW distance using envelope sequences. If the lower bound exceeds the best distance found so far, skip the full DTW computation entirely.

**How:** For each training example, precompute upper/lower envelopes based on the Sakoe-Chiba band width. Compare the candidate against the envelope — points outside the envelope contribute to the lower bound.

**Caveat:** With high-dimensional data (33 joints × 2-3 coords = 66-99 dimensions), envelope looseness increases. Expect 50-70% pruning, not 90%.

**Stack order:** LB_Keogh prune → DTW with early abandon → Sakoe-Chiba band. Each layer catches what the previous missed.

### 3d. Do NOT Use Spatial Resampling

**Critical warning:** Spatial resampling (equidistant points, $1 Recognizer style) **destroys velocity information**. A slow wave and fast wave become identical sequences. For dance where dynamics and intensity matter, this erases critical information.

**Use temporal downsampling only:** Fixed-rate subsampling (60fps → 15fps) preserves velocity because fast movements cover more space between frames.

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
    training_mu: f32,         // Mean pairwise distance during training
    training_sigma: f32,      // Std dev of pairwise distances during training
    threshold: f32,           // mu + sigma * coefficient
}

fn compute_threshold(examples: &[Sequence], null_rejection_coeff: f32) -> ThresholdStats {
    // 1. Compute all pairwise DTW distances
    let mut all_distances: Vec<f32> = Vec::new();
    for i in 0..examples.len() {
        for j in (i + 1)..examples.len() {
            let dist = dtw_distance(&examples[i], &examples[j]);
            if dist.is_finite() {
                all_distances.push(dist);
            }
        }
    }

    // 2. Compute mu and sigma from all pairwise distances
    let n = all_distances.len() as f32;
    let mu = all_distances.iter().sum::<f32>() / n;
    let variance = all_distances.iter()
        .map(|d| (d - mu).powi(2))
        .sum::<f32>() / n;
    let sigma = variance.sqrt();

    // 3. Compute threshold
    let threshold = mu + sigma * null_rejection_coeff;

    ThresholdStats { mu, sigma, threshold }
}
```

> **Note (v0.7.0)**: An earlier version selected a "best template" (example with lowest average distance to others) and computed statistics relative to it. A/B testing showed this was strictly worse than using all examples — 30% fewer gestures detected. The best template approach was removed in v0.7.0.

**Parameters**:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `null_rejection_coeff` | 3.0 | Higher = more permissive (fewer false negatives), Lower = stricter (fewer false positives) |

**Why this is better than manual thresholds**:
- **No per-gesture tuning**: One global `null_rejection_coeff` works for all gestures
- **Adapts to complexity**: Simple gestures (clap) get tight thresholds; complex gestures (dance phrase) get looser thresholds
- **Battle-tested**: Used in production systems for years
- **Automatic recalibration**: Threshold updates when you add/remove training examples

**Minimum examples required**: At least 3 examples per gesture to compute statistics, but **6+ recommended** for stable thresholds. With n=4, there are only C(4,2)=6 pairwise distances — small sample underestimates true variance. With n=6, there are C(6,2)=15 pairwise distances, producing 2.5× more data points.

**Training data quantity impact (tested 2026-01-29):**
- 4 examples/gesture: thresholds shift significantly when retrained
- 6 examples/gesture: thresholds stabilize, better capture natural gesture variation
- With fewer than 6 examples, consider using coefficient=2.5-3.0 to compensate for undersampled variance

### Optional Enhancement: Winner-Take-All Hybrid

For multi-gesture systems, add a secondary check using likelihood scores:

```rust
fn classify_with_likelihood(
    window: &Sequence,
    gestures: &[GestureTemplate],
) -> Option<usize> {
    // 1. Compute distances to all gestures (minimum across all examples)
    let distances: Vec<f32> = gestures.iter()
        .map(|g| g.examples.iter()
            .map(|ex| dtw_distance(window, ex))
            .fold(f32::INFINITY, f32::min))
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
| Window size | First example's length | ~120 frames at 60fps (~2s gesture) |
| Frame skip | 4 | 15 DTW computations/sec |
| Downsample | 4 | Compare ~30 frame sequences |
| Per-gesture cooldown | 500ms | Minimum between same-gesture hits |
| Global cooldown (NMS) | 1500ms | Block all gestures after any hit |
| Safety valve | 5000ms | Force re-arm timeout |
| Frames to fire | 3 | ~200ms confirmation at 15Hz DTW |
| Sakoe-Chiba band | 0.15 | 15% warping constraint |
| Threshold | AUTO (μ+σ×2.0) | 6+ examples recommended |
| Training examples | 6+ | Minimum for stable statistics |

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

A state machine with **frame accumulation** for confirmation and **time-based recovery** (safety valve + global cooldown) for echo prevention.

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
                   │(safety valve)│───────────────────┘
                   └─────────────┘
                   after max_recovery_ms (5000ms)
```

### ⚠️ CRITICAL INSIGHT: Two-Layer Echo Defense (v0.7.0)

**The #1 learning**: A single mechanism cannot prevent all echo types. Two layers are needed:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| **1. Safety Valve** | Force re-arm after `max_recovery_ms` (5s) | Stuck recovery when resting distance < threshold |
| **2. Global Cooldown (NMS)** | Block ALL gestures for 1500ms after ANY hit | Cross-gesture round-robin echoes |

**Why distance-based recovery alone fails**:
- With body tracking, resting distance is often permanently below threshold
- Example: jump threshold=53, resting distance=15. Distance NEVER clears above threshold.
- The safety valve (time-based) handles this — timer-based recovery is essential for body tracking.

**What works** (safety valve + global NMS):
```rust
RecognitionState::Recovery => {
    let elapsed = self.recovery_start
        .map(|t| t.elapsed())
        .unwrap_or(Duration::ZERO);
    let max_recovery = Duration::from_millis(config.max_recovery_ms);

    if elapsed >= max_recovery {
        self.reset_to_idle();
        (false, Some(RecognitionState::Idle), "safety_valve_timeout")
    } else {
        (false, None, "")
    }
}

// In process_frame, BEFORE the per-gesture loop:
let in_global_cooldown = self.last_any_hit_time
    .map(|t| t.elapsed() < Duration::from_millis(config.global_cooldown_ms))
    .unwrap_or(false);

// In Idle state:
if in_global_cooldown {
    // Block Idle→Building for ALL gestures after ANY hit
    return (false, None);
}
```

**History**: An earlier version (v0.6.0) included a third layer — Schmitt trigger hysteresis — that tracked `min_distance` in Recovery and re-armed when consistently above `threshold × 1.1`. Across 55+ hits in two production sessions, the hysteresis path never fired; all Recovery→Idle transitions used the safety valve timeout. Resting distances run at 4-8% of threshold, permanently below the re-arm point. The Schmitt trigger was removed in v0.7.0 as provably dead code.

### Why This Works

| Mechanism | Purpose | Why It Works |
|-----------|---------|--------------|
| **Frame accumulation** | Prevents noise spikes from firing | 3 frames = ~200ms of consistent low distance |
| **Distance slope check** | Prevents entry during noise/flat sections | Only enter Building when distance is falling |
| **Safety valve timeout** | Prevents stuck recovery | Force re-arm after 5s even if distance stays low |
| **Global cooldown (NMS)** | Prevents cross-gesture echoes | Block ALL gestures for 1.5s after ANY hit |
| **Sakoe-Chiba band** | Prevents pathological DTW warping | Better distances + 85% fewer cells computed |

### Production Configuration (v0.7.0)

```rust
pub struct RecognitionConfig {
    pub cooldown_ms: 500,              // Per-gesture minimum between hits
    pub threshold_high_factor: 1.0,    // Entry at 100% of threshold
    pub frames_to_fire: 3,             // ~200ms confirmation at 15Hz DTW
    pub max_recovery_ms: 5000,         // Safety valve: force re-arm after 5s
    // Global non-maximum suppression
    pub global_cooldown_ms: 1500,      // Block ALL gestures after ANY hit
    // DTW optimization
    pub sakoe_chiba_band: 0.15,        // 15% warping constraint
}
```

### Real-World Results

**Session 1 (2026-01-29, v0.6.0)** — 3 gestures, 6 examples each:
- **43 HITs, 0 echoes (0.0% echo rate)**
- Previous: 60 HITs, 38 echoes (63.3% echo rate)
- All Recovery→Idle via safety valve timeout
- Cross-gesture minimum gap: 2029ms (global cooldown prevents round-robin)
- 45 Building entries, 44 Peak fires, 0 aborted (100% Building→Peak conversion)

**Session 2 (2026-01-30, v0.7.0)** — 3 gestures, 38 examples total (12/17/9):
- **12 HITs, 0 echoes (0.0% echo rate)**
- All Recovery→Idle via safety valve timeout
- Hit margins: 94-97% below threshold (massive headroom)
- More training examples = more consistent detection

**Threshold values (AUTO, coefficient=2.0):**

| Gesture | Examples | μ | σ | Threshold |
|---------|----------|---|---|-----------|
| wave (Session 1) | 6 | 28.7 | 15.6 | 59.8 |
| wave (Session 2) | 12 | ~107 | ~55 | 216.3 |
| jump (Session 2) | 17 | ~148 | ~93 | 334.1 |
| spin (Session 2) | 9 | ~108 | ~64 | 236.9 |

**Key observation:** Resting distances run at 4-8% of threshold for all gestures. The system is fundamentally a rate-limiting system on an always-triggered signal. The safety valve handles this cleanly.

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
# ... 5000ms later (safety valve) ...
timestamp,REC,frame,buffer,window,wings:68:27:1  # Re-armed, ready
```

### Implementation (from recognizer.rs, v0.7.0)

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionState {
    Idle,      // Waiting for gesture, ready to detect
    Building,  // Distance below threshold, accumulating frames
    Peak,      // Gesture detected, fire OSC
    Recovery,  // Safety valve period, blocking new detections
}

fn process_state_machine(
    &mut self,
    distance: f32,
    config: &RecognitionConfig,
    in_global_cooldown: bool,  // NMS: true if ANY gesture fired recently
) -> StateMachineResult {
    let entry_threshold = self.threshold * config.threshold_high_factor;
    let max_recovery = Duration::from_millis(config.max_recovery_ms);

    match self.state {
        RecognitionState::Idle => {
            // Global cooldown blocks ALL gestures from entering Building
            if in_global_cooldown {
                return (false, None);
            }
            // Distance slope check: only enter Building if distance is falling
            if distance < entry_threshold && self.is_distance_falling(distance) {
                self.state = RecognitionState::Building;
                self.frames_below_threshold = 1;
                if self.frames_below_threshold >= config.frames_to_fire {
                    self.state = RecognitionState::Peak;
                    self.record_hit();
                    return (true, Some(RecognitionState::Peak));
                }
            }
            (false, None)
        }

        RecognitionState::Building => {
            if distance < entry_threshold {
                self.frames_below_threshold += 1;
                if self.frames_below_threshold >= config.frames_to_fire {
                    self.state = RecognitionState::Peak;
                    self.record_hit();
                    return (true, Some(RecognitionState::Peak));
                }
            } else {
                self.reset_to_idle(); // Distance rose, reset
            }
            (false, None)
        }

        RecognitionState::Peak => {
            self.state = RecognitionState::Recovery;
            self.recovery_start = Some(Instant::now());
            (false, Some(RecognitionState::Recovery))
        }

        RecognitionState::Recovery => {
            // Safety valve: force re-arm after max_recovery_ms
            let elapsed = self.recovery_start
                .map(|t| t.elapsed())
                .unwrap_or(Duration::ZERO);

            if elapsed >= max_recovery {
                self.reset_to_idle();
                (false, Some(RecognitionState::Idle), "safety_valve_timeout")
            } else {
                (false, None, "")
            }
        }
    }
}
```

**Global cooldown** is computed in `process_frame()` before the per-gesture loop:
```rust
let in_global_cooldown = self.last_any_hit_time
    .map(|t| t.elapsed() < Duration::from_millis(config.global_cooldown_ms))
    .unwrap_or(false);

// When any gesture fires:
self.last_any_hit_time = Some(Instant::now());
```

### What We Removed (Simplification)

| Removed | Version | Why |
|---------|---------|-----|
| `AdaptiveThreshold` | v0.5.0 | Computed from wrong data, over-complicated |
| Motion gate | v0.5.0 | Blocked recognition when it shouldn't |
| Calibration step | v0.5.0 | Not needed with statistical threshold |
| Armed/disarmed tracking | v0.5.0 | State machine handles this implicitly |
| Peak detection (local minima) | v0.5.0 | Frame accumulation is simpler and works |
| Simple distance-based re-arming | v0.5.0 | Fails when resting distance ≈ threshold; replaced with safety valve |
| Spatial resampling | v0.5.0 | Destroys velocity info critical for dance |
| Schmitt trigger hysteresis | v0.7.0 | Never fired in 55+ hits across 2 sessions; resting distance always below re-arm threshold |
| Best template comparison | v0.7.0 | Lost A/B test by 30% detection rate vs all-examples comparison |
| `hangover_ms` | v0.7.0 | Only used by Schmitt trigger path; became dead code when hysteresis removed |

**Result**: ~500 lines → ~350 lines, 0% echo rate (from 63.3%).

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

| Metric | Target | Achieved (v0.7.0) | How to Measure |
|--------|--------|-------------------|----------------|
| Latency | <200ms | ~200ms | Time from distance dip to HIT (3 frames × 67ms) |
| Echo rate | 0% | 0% (0/43) | HITs within 2s of previous same-gesture HIT |
| Accuracy | >90% | 100% | Correct HITs / total gestures performed |
| False positive | <10% | 0% | Spurious HITs / total HITs |
| Building→Peak | >95% | 98% (44/45) | Building entries that reach Peak (not aborted) |

---

Use this approach as the baseline for any gesture recognition implementation. The simplicity of the basic approach is intentional - more complex approaches consistently failed while this Wekinator-style approach works reliably. The advanced dual-detection approach builds on this foundation for continuous gesture performance scenarios.
