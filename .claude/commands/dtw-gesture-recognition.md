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

**Future: Auto-Calibration**:
- Compute intra-class distance (between examples of same gesture)
- Set threshold slightly above max intra-class distance

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

Use this approach as the baseline for any gesture recognition implementation. The simplicity is intentional - more complex approaches consistently failed while this simple Wekinator-style approach works reliably.
