# Gesture Recognition Learnings

**Date**: 2026-01-26
**Status**: Working baseline established

---

## The Breakthrough

After multiple failed attempts with complex approaches, we achieved reliable gesture recognition with a **simple implementation** modeled after Wekinator's DTW approach.

### What Works

1. **Train a gesture** (wave) with 5 repetitions
2. **Switch to Performance mode**
3. **Perform the gesture** → Recognized reliably with single clean hit
4. **Threshold ~8000** works well for skeleton data (34 joints × 2 coordinates = 68 dimensions)

---

## The Algorithm (Wekinator-Style)

### Source Reference
- **Repository**: [fiebrink1/wekinator](https://github.com/fiebrink1/wekinator)
- **Key File**: `src/wekimini/learning/dtw/DtwModel.java`
- **Key Method**: `classifyContinuous()` (lines 920-970)

### Core Logic

```
For each frame:
  1. Add frame to sliding window buffer
  2. Get window of size N (where N = first training example's length)
  3. For each gesture:
     - Compare window against ALL training examples using DTW
     - Track best (lowest) distance
  4. If best_distance < threshold AND armed AND not_in_cooldown:
     - Fire hit
     - Disarm (must go above threshold to re-arm)
  5. If distance >= threshold:
     - Re-arm gesture
```

### Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Window size | Fixed (first example's length) | Simple, predictable |
| Example comparison | All examples, not prototype | Captures full variance |
| Threshold check | Simple `distance < threshold` | Easy to understand |
| Double-hit prevention | Armed/disarmed state | Must "release" before next hit |
| Performance | Frame skip + downsample | ~64x faster than naive |

---

## What Failed (And Why)

### Failed Approach 1: Edge Detection
- **Idea**: Fire only when distance *crosses* below threshold
- **Problem**: Noisy distance values caused spurious crossings
- **Lesson**: Simple threshold is more robust

### Failed Approach 2: Prototype Averaging
- **Idea**: Average all examples into one prototype, compare against that
- **Problem**: Lost variance information, didn't match real gestures well
- **Lesson**: Compare against all examples individually

### Failed Approach 3: Multiple Candidate Windows
- **Idea**: Try many window sizes based on min/max example lengths
- **Problem**: Added complexity without clear benefit, harder to debug
- **Lesson**: Fixed window size (from first example) is sufficient

### Failed Approach 4: Activity Gate
- **Idea**: Skip DTW when motion energy is low (standing still)
- **Problem**: Added another threshold to tune, didn't help much
- **Lesson**: The main threshold handles idle state naturally

### Failed Approach 5: No Performance Optimization
- **Idea**: Run DTW at full 60fps on full-length sequences
- **Problem**: CPU overload, UI became unresponsive, distances got "stuck"
- **Lesson**: Frame skipping + downsampling is essential

---

## Performance Optimizations (Essential)

### 1. Frame Skipping
```rust
// Only compute DTW every 4th frame (15Hz instead of 60Hz)
if self.frame_count % self.dtw_skip != 0 {
    return; // Skip this frame
}
```
**Impact**: 4x reduction in DTW computations

### 2. Downsampling
```rust
// Compare at 15fps equivalent (every 4th frame of sequences)
let window = Self::downsample(&window_full, 4);      // 180 → 45 frames
let example = Self::downsample(&example_full, 4);   // 180 → 45 frames
```
**Impact**: DTW matrix goes from 180×180 to 45×45 = 16x fewer cells

### Combined Impact
- **Before**: 60 DTW/sec × 32,400 cells = 1.9M operations/sec
- **After**: 15 DTW/sec × 2,025 cells = 30K operations/sec
- **Speedup**: ~64x

---

## Double-Hit Prevention (Armed State)

### The Problem
After a hit fires, the distance might stay below threshold for a while (gesture is still in progress). When cooldown expires, another hit fires even though user only did one gesture.

### The Solution
```rust
pub struct GestureState {
    armed: bool,  // Must go above threshold to re-arm
    // ...
}

// In process_frame:
if distance >= threshold {
    gesture.arm();  // Re-arm when distance goes up
}

if distance < threshold && gesture.is_armed() && !gesture.in_cooldown() {
    gesture.record_hit();  // This also disarms
    // Fire hit...
}
```

### How It Works
```
1. Idle → distance high → armed
2. Gesture starts → distance drops → still armed
3. Distance < threshold → HIT fires → disarmed
4. Gesture continues → distance still low → can't fire (disarmed)
5. Gesture ends → distance rises above threshold → re-armed
6. Ready for next gesture
```

---

## Threshold Calibration

### What We Learned
- **Default threshold (1500)** was too low for our skeleton data
- **Threshold ~8000** worked well for the user
- Distance scale depends on:
  - Number of dimensions (68 for skeleton)
  - Magnitude of values (skeleton coordinates)
  - Gesture duration (more frames = higher cumulative distance)

### Calibration Process
1. Start with threshold at maximum (slider right)
2. Perform gesture repeatedly
3. Watch the distance values
4. Lower threshold until gestures trigger reliably
5. Raise slightly to reduce false positives

### Future Improvement
Auto-calibration based on training examples:
- Compute intra-class distance (between examples of same gesture)
- Set threshold slightly above max intra-class distance

---

## UI/UX Learnings

### What Helps Debugging
1. **Large hit indicator** that disappears quickly (300ms)
2. **Hit log on right side** visible from distance
3. **Distance display** showing real-time values
4. **Debug info** (window size, example count, buffer status)

### What Was Confusing
1. Hit indicator staying too long (was 800ms, now 300ms)
2. Can't see UI from across the room (increased window size)
3. Not knowing if examples were loaded (added example count display)

---

## Code Architecture

### Key Files
```
src/engine/
├── recognizer.rs   # The working recognizer (simple Wekinator-style)
├── dtw.rs          # Core DTW algorithm (unchanged, works well)
├── buffer.rs       # Frame buffer (simplified version in recognizer.rs)
└── training.rs     # Training session state machine

src/gui/
└── mod.rs          # UI with Performance mode monitor
```

### Recognizer Structure
```rust
pub struct Recognizer {
    buffer: FrameBuffer,      // Sliding window of recent frames
    gestures: Vec<GestureState>,
    window_size: usize,       // Fixed, from first example
    frame_count: usize,       // For frame skipping
    dtw_skip: usize,          // Skip factor (4 = 15Hz)
    downsample: usize,        // Downsample factor (4 = 15fps equivalent)
}

pub struct GestureState {
    examples: Vec<Sequence>,  // All training examples
    threshold: f32,           // Match threshold
    armed: bool,              // Double-hit prevention
    current_distance: Option<f32>,  // For display
}
```

---

## Next Steps (Future Improvements)

### Phase 2: Feature Engineering
- Hip-centered normalization (position invariance)
- Scale normalization (size invariance)
- Velocity features

### Phase 3: Auto-Calibration
- Baseline recording (neutral stance)
- Automatic threshold computation
- Inter-gesture separation warnings

### Phase 4: Advanced (Stretch)
- LB_Keogh pruning for many gestures
- Subsequence DTW (SPRING)
- Variable-length gesture support

---

## Summary: The Recipe That Works

1. **Store examples as-is** (full 60fps)
2. **Fixed window size** = first example's length
3. **Compare against all examples** using standard DTW
4. **Downsample both** window and examples for comparison (4x)
5. **Skip frames** (compute DTW every 4th frame)
6. **Simple threshold**: distance < threshold = potential hit
7. **Armed state**: must go above threshold before next hit
8. **Cooldown**: minimum time between hits (500ms default)

This is the baseline. Build from here.
