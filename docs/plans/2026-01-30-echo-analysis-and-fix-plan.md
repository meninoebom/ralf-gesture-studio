---
title: "Echo Analysis & Fix Plan"
type: analysis
date: 2026-01-30
status: reviewed
---

# Echo Analysis & Fix Plan

Detailed analysis of echo behavior observed in real-world testing on 2026-01-29, with proposed fixes. Incorporates external expert review.

## Design Insight

> "Continuous gesture recognition is not classification. It is: temporal event detection, with memory, and hysteresis, and refractory dynamics. Your system is not 'DTW with thresholds'. It is **a temporal control system wrapped around DTW**. And your fixes all target the control system — not DTW. That's exactly right."

This framing is essential. The DTW algorithm is working correctly — it produces accurate distance measurements. The problem is entirely in the **control system** that decides when those distances constitute a "gesture event" vs noise, rest, or echo.

## Context

**System**: RALF Gesture Studio — a desktop app for training and recognizing dance gestures using Dynamic Time Warping (DTW). Receives 66-dimensional skeleton data (33 MediaPipe joints × XY) via OSC at 60fps. Compares a sliding window of recent frames against training examples using DTW, and fires OSC "hit" messages when a match is detected.

**Recognition pipeline**:
1. Skeleton frames arrive at 60fps
2. Every 4th frame (~15Hz), extract a sliding window of the last 122 frames (~2 seconds)
3. Downsample the window by 4× (compare at 15fps)
4. Compute DTW distance between the window and each gesture's training examples
5. Find the gesture with the lowest distance
6. Run that gesture through a VAD-style state machine: Idle → Building → Peak (fire!) → Recovery → Idle
7. Building requires 3 consecutive frames below threshold (~200ms confirmation)
8. Recovery uses a dual-path re-arm: fast path if distance clearly exceeds threshold, slow path (extended hangover timer) otherwise

**Test setup**: 3 gestures (wave, jump, spin), each with 4 training examples of ~122 frames (~2 seconds). Statistical threshold using μ + 2σ (mean + 2× standard deviation of inter-example DTW distances).

**Gesture statistics**:

| Gesture | Examples | μ (mean) | σ (std) | σ/μ ratio | Threshold (μ+2σ) |
|---------|----------|----------|---------|-----------|-------------------|
| wave | 4 | 10.0 | 6.6 | 0.66 | 23.2 |
| spin | 4 | 14.5 | 3.4 | 0.23 | 21.3 |
| jump | 4 | 26.5 | 14.8 | 0.56 | 56.1 |

**Statistical risk note**: n=4 examples per gesture is statistically fragile. One "sloppy" training example can inflate σ significantly. For jump, σ/μ = 0.56 is high, suggesting the training examples vary substantially. 8–10 examples per gesture would produce more stable statistics and potentially tighter thresholds.

**User report**: "Accuracy is getting very high... timeliness is good... echoes are pretty bad; they're actually very bad."

---

## Part 1: What the Data Shows

### 1.1 Overall Numbers

From `ralf-diagnostics-2026-01-29_16-43-06.log` (83.4 seconds of recognition):

- **60 total HITs** fired
- **22 estimated real gestures** (first hit in each temporal cluster)
- **38 echoes** (subsequent hits within 2 seconds of the same gesture)
- **63.3% echo rate**

Per-gesture breakdown:

| Gesture | Total Hits | Real | Echoes | Echo Rate | Longest Chain |
|---------|-----------|------|--------|-----------|---------------|
| jump | 27 | 6 | 21 | 78% | 9 consecutive echoes |
| spin | 25 | 10 | 15 | 60% | 5 consecutive echoes |
| wave | 8 | 6 | 2 | 25% | 3 consecutive echoes |

### 1.2 The Echo Cycle Is Extremely Regular

Every echo occurs at almost exactly **~1420ms** after the previous hit. The gaps between consecutive same-gesture hits:

```
jump echo gaps (ms): 1417, 1425, 1416, 1421, 1425, 1417, 1425, 1423, 1423, 1422,
                     1424, 1420, 1421, 1420, 1420, 1623, 1418, 1421, 1421, 1419, 1423

spin echo gaps (ms): 1416, 1419, 1417, 1422, 1423, 1419, 1425, 1419, 1421, 1423,
                     1418, 1417, 1419, 1424, 1418
```

Standard deviation of these gaps is ~5ms. This is a perfectly periodic cycle, not random noise.

### 1.3 The Cycle Anatomy

By tracing the STATE transitions in the log, every echo follows this exact sequence:

```
t=0ms      HIT fires (Building → Peak)
t=~50ms    Peak → Recovery (post_fire)
t=~810ms   Recovery → Idle (extended_hangover_complete, after 500ms + processing)
t=~820ms   Idle → Building (below_threshold_falling — distance is still low)
t=~1420ms  Building → Peak → HIT fires (3 frames accumulated at ~200ms intervals)
```

**Key observation**: 100% of Recovery→Idle transitions use reason `extended_hangover_complete`. Not a single one uses `hangover_complete_distance_exceeded`. The "fast path" (distance exceeding the rearm threshold) never triggers.

### 1.4 Why the Fast Path Never Triggers

The re-arm threshold is `threshold × rearm_threshold_factor` where `rearm_threshold_factor = 1.3`:

| Gesture | Threshold | Rearm Threshold (×1.3) | Max Distance in Recovery | Ever Reached? |
|---------|-----------|----------------------|-------------------------|---------------|
| wave | 23.2 | 30.2 | ~22 | No |
| spin | 21.3 | 27.7 | ~18 | No |
| jump | 56.1 | 72.9 | ~41 | No |

The distances during Recovery are far too low to ever reach the rearm threshold. All re-arming happens via the 500ms extended hangover timer.

### 1.5 True Resting Distances

From the first 5 seconds of the log (before any gesture performance), the distances observed for each gesture:

| Gesture | Resting Distance Range | Resting Average | Threshold | Resting vs Threshold |
|---------|----------------------|-----------------|-----------|---------------------|
| wave | 33–45 | 41.1 | 23.2 | 77% above threshold |
| spin | 30–42 | 38.0 | 21.3 | 78% above threshold |
| jump | 30–42 | 38.0 | 56.1 | **32% below threshold** |

For wave and spin, the resting body distance is **well above** the recognition threshold. This means a correctly-timed system would stop detecting these gestures when the dancer returns to rest.

For jump, the resting body distance (38) is **permanently below** the recognition threshold (56.1). The system can never distinguish "performing a jump" from "standing at rest" using the current threshold.

### 1.6 Distance During an Echo Chain

During the jump echo chain from t=54554ms to t=63087ms (7 HITs in 8.5 seconds), every single REC sample shows jump distance of 9–14:

```
t=54097ms  jump_dist=13
t=54301ms  jump_dist=12
t=54504ms  jump_dist=12
t=54709ms  jump_dist=13
...
t=62834ms  jump_dist=14
t=63036ms  jump_dist=14
t=63239ms  jump_dist=14
```

The distance **never rises above 14** during this entire 9-second window. Compare to the resting distance of 38. Either:
- The user is continuously performing jump-like movements for 9 seconds, OR
- The sliding window retains gesture data for longer than expected

Both are likely contributing. The window is 122 frames (2.03 seconds), so after 2 seconds the window should be fully refreshed. A 9-second chain at distance 13 means the user is likely performing repeated jumps or continuous jump-like motion. But the system fires every 1.4 seconds instead of once, because the hangover (500ms) is much shorter than the gesture duration.

### 1.7 Cross-Gesture Echoes

20 instances of different gestures firing within 3 seconds of each other. The pattern: when gesture A enters Recovery, gesture B becomes the "best match" (lowest distance) and starts its own state machine cycle.

Example from the log:
```
t=8464ms   spin fires  → spin enters Recovery
t=9276ms   wave fires  → (812ms later, while spin is recovering)
t=10297ms  spin fires  → spin recovered and immediately echoed
```

The current code only processes the state machine for the **lowest-distance gesture** each frame. When spin is in Recovery, wave or jump takes over as "best" and may fire. This creates a cascading cross-gesture echo chain.

---

## Part 2: Root Cause Analysis

### Root Cause 1: Window Memory Mismatch ("Ghost Signal")

**The core timing problem**: The DTW sliding window is 122 frames = **2033ms** at 60fps. The extended hangover is only **500ms**. When the hangover expires and the gesture re-arms, 75% of the sliding window still contains the original gesture data:

```
Timeline:
────────────────────────────────────────────────────────→ time
|←───── window (2033ms) ─────→|
[GESTURE DATA GESTURE DATA GEST|ure d|NEW FRAMES]
                                     ↑
                              Re-arm at 500ms
                              75% of window = old gesture data
                              Distance: still low → immediate re-fire
```

After 500ms, only ~30 new frames have entered the window (out of 122). The DTW comparison is still dominated by the old gesture data, so the distance remains well below threshold.

In continuous DTW, a high-scoring pattern remains in the sliding window buffer until it "slides out." Because DTW is elastic, it can warp the remaining 75% of the gesture to match the training example again with a low enough distance to trigger.

**This is the "smoking gun" finding**: the 1.4-second echo period is a direct result of window length (2s) minus hangover (0.5s), plus frame accumulation time (0.4s).

**This explains why spin and wave echo**: Even though their resting distances (38-41) are above their thresholds (21-23), it takes ~2 seconds for the window to fully refresh. The 500ms hangover only covers 25% of that time.

### Root Cause 2: Threshold-Resting Gap ("Always-On Signal")

Jump's threshold is **set above the resting distance**, making it impossible for the system to distinguish between "performing a jump" and "standing still":

```
Jump distances:
  Performing:  9–17  ██████████░░░░░░░░░░░░░░░░░░░░░
  Resting:     30–42 ░░░░░░░░░░░░░░░░██████████░░░░░░
  Threshold:   56.1  ░░░░░░░░░░░░░░░░░░░░░░░░░░░░56.1
                     ▲                              ▲
                     Both regions are below threshold!
```

The μ+2σ formula measures **inter-example variability**, not the **decision boundary between gesture and non-gesture**. Jump has high variability (σ/μ = 0.56), which pushes the threshold to 56.1 — far above the resting distance of 38.

This is a **One-Class Classification** failure. The system is trained only on positive examples (gestures) and must infer the decision boundary from their variation alone. With n=4 examples, a single inconsistent performance inflates σ and makes the threshold uselessly loose.

With any hangover duration, jump will re-echo because the resting distance is always below threshold. This problem **cannot be solved by timers alone** — it requires fixing the threshold.

### Root Cause 3: Cross-Gesture State Machine Isolation ("Round Robin")

Each gesture has an independent state machine, but only the lowest-distance gesture gets its state machine processed each frame. When the winning gesture enters Recovery (blocked from firing), another gesture becomes the "lowest" and can start Building.

This design was intended to prevent multiple gestures from firing simultaneously, but it creates a **round-robin echo** effect where gestures take turns firing. This is the same problem solved by **Non-Maximum Suppression (NMS)** in computer vision object detection — when multiple classifiers compete for the same temporal window, firing one should suppress the others.

---

## Part 3: Proposed Fixes

### Understanding the Fix Categories

The fixes target the **temporal control system**, not DTW. They fall into two categories:

1. **Timer-based** (Fixes 1, 2): Effective band-aids that enforce minimum intervals. Simple but inflexible — they don't adapt to the dancer's actual movement.
2. **Signal-based** (Fixes 3, 4): Address the root cause by making the control system respond to actual distance dynamics. More robust long-term but more complex.

External review recommends shipping timer fixes immediately to stop the bleeding, then implementing signal-based fixes for the real solution.

### Fix 1: Extend Hangover to Match Window Duration (Band-Aid)

**Change**: Increase `extended_hangover_ms` from 500 to 2000 (matching the ~2033ms window duration).

**Why it works**: After 2 seconds, the sliding window is fully refreshed with post-gesture data. For spin and wave, the resting distance (38-41) is above threshold (21-23), so the distance will naturally rise above threshold and recognition stops.

**Impact**: Eliminates spin and wave echoes. Does NOT fix jump (resting distance is below threshold regardless of hangover duration).

**Trade-off**: Enforces a maximum gesture tempo of ~30 BPM (one gesture per 2 seconds). For dance, this is slow but acceptable as a stability fix. Fix 3 provides a better long-term solution that adapts to the dancer's speed.

**Risk**: None for correctness. Reduces maximum fire rate, but performing a distinct gesture in under 2 seconds is already fast.

**Implementation**: Change one constant.

```rust
RecognitionConfig {
    extended_hangover_ms: 2000,  // was 500; matches window duration
    // ...
}
```

### Fix 2: Global Post-Fire Cooldown (Debouncing / NMS)

**Change**: After ANY gesture fires, prevent ALL gestures from entering Building for a configurable period (e.g., 1500ms).

**Why it works**: This is the temporal equivalent of **Non-Maximum Suppression** in object detection. When multiple classifiers compete for the same temporal window (jump vs spin vs wave), firing one should suppress the others to prevent the round-robin echo cascade.

**Impact**: Eliminates cross-gesture echo chains. The 20 instances of different gestures firing within 3 seconds of each other would be prevented.

**Risk**: If the dancer intentionally performs two different gestures in rapid succession (< 1.5 seconds apart), the second would be missed. This seems unlikely in practice — distinct gesture performance typically takes 2+ seconds.

**Implementation note**: The cooldown must block the **start** of the next detection (Idle→Building transition), not just the firing event. This ensures no gesture can begin building while the global cooldown is active.

```rust
// In Recognizer:
last_any_hit_time: Option<Instant>,
global_cooldown_ms: u64,  // e.g., 1500

// In process_frame, before any gesture's state machine:
let in_global_cooldown = self.last_any_hit_time
    .map(|t| t.elapsed() < Duration::from_millis(self.global_cooldown_ms))
    .unwrap_or(false);

// When any gesture fires:
self.last_any_hit_time = Some(Instant::now());

// Block Building entry during global cooldown (in Idle state):
if distance < entry_threshold
    && !in_global_cooldown
    && self.is_distance_falling(distance)
{
    // Enter Building
}
```

### Fix 3: Schmitt Trigger Hysteresis (The Real Fix)

**Change**: After firing, the gesture's distance must **consistently stay above its threshold** before it can re-arm. Track `min_distance` during recovery and only re-arm when `min_distance > threshold × safety_factor`. Add a safety valve timeout (e.g., 5 seconds) to prevent permanent stuck state.

**Why it's the real fix**: Fixed timers (Fix 1) are brittle — they don't adapt to the user's speed. A Schmitt trigger uses two distinct thresholds: a low one to trigger (fire) and a high one to reset (re-arm). This naturally adapts: if a user holds a gesture pose, the hysteresis waits until they actually move away. If they move away quickly, re-arming happens quickly.

**Why `min_distance` not `max_distance`**: The original plan tracked `max_distance_in_recovery` — "did distance ever spike above threshold?" But a brief spike (from noise or a twitch) doesn't mean the user has truly left the gesture zone. Tracking `min_distance` and requiring it to exceed `threshold × 1.1` means the distance was **consistently** above threshold, not just a momentary fluctuation. This prevents "flickering" near the boundary.

**For jump**: The resting distance (38) is below threshold (56), so the min_distance will never exceed the threshold. The safety valve timeout (5s) forces re-arm eventually, but then the cycle restarts. This fix alone doesn't solve jump's problem — jump needs Fix 4.

**Risk**: Could feel "stuck" if the dancer's resting pose is near the threshold boundary. The safety valve prevents permanent stuck state.

**Implementation**: Replace the current dual-path Recovery logic.

```rust
RecognitionState::Recovery => {
    // Track min distance seen during recovery (for Schmitt trigger)
    self.min_distance_in_recovery = self.min_distance_in_recovery.min(distance);

    let elapsed = self.recovery_start
        .map(|t| t.elapsed())
        .unwrap_or(Duration::ZERO);

    let min_hangover = Duration::from_millis(config.hangover_ms);
    let max_recovery = Duration::from_millis(config.max_recovery_ms); // 5000ms
    let rearm_threshold = self.threshold * config.rearm_safety_factor; // 1.1

    // Schmitt trigger: distance must be consistently above threshold
    // We check min_distance (not max) to ensure the distance STAYED above,
    // not just spiked briefly due to noise
    let distance_cleared = self.min_distance_in_recovery > rearm_threshold;

    let can_rearm = if distance_cleared && elapsed >= min_hangover {
        // Normal re-arm: distance consistently above threshold + minimum hangover
        true
    } else if elapsed >= max_recovery {
        // Safety valve: force re-arm after max recovery time
        // (prevents permanent stuck state for gestures like jump
        //  where resting distance is below threshold)
        true
    } else {
        false
    };

    if can_rearm {
        let reason = if distance_cleared {
            "hysteresis_cleared"
        } else {
            "safety_valve_timeout"
        };
        self.reset_to_idle();
        (false, Some(RecognitionState::Idle), reason)
    } else {
        (false, None, "")
    }
}
```

**New config fields**:
```rust
pub struct RecognitionConfig {
    // ... existing ...

    /// Safety factor for Schmitt trigger re-arm.
    /// Distance must consistently exceed threshold × this factor.
    /// Example: 1.1 means distance must be 10% above threshold.
    pub rearm_safety_factor: f32,   // default: 1.1

    /// Maximum time in Recovery before forcing re-arm (safety valve).
    /// Prevents permanent stuck state when resting distance < threshold.
    pub max_recovery_ms: u64,       // default: 5000
}
```

**Replaces**: `rearm_threshold_factor` and the current dual-path logic. The old `rearm_threshold_factor` (1.3) was applied as `threshold × 1.3`, which set the bar unreachably high. The new `rearm_safety_factor` (1.1) is applied the same way (`threshold × 1.1`) but the key behavioral change is tracking `min_distance` instead of `max_distance`, and removing the extended_hangover timer path entirely.

**Long-term**: This should eventually replace Fix 1 as the primary echo prevention mechanism. Fix 1 is the band-aid; Fix 3 is the real solution.

### Fix 4: Fix the Threshold (The "Jump" Problem)

**The math problem**: Using μ+2σ on n=4 examples is statistically risky. If a user performs one "sloppy" jump during training (high DTW distance to the other examples), σ explodes and the threshold becomes uselessly high. Jump's σ/μ ratio of 0.56 is evidence of this.

**Why this can't be fixed by timers**: The resting distance for jump (38) is below its threshold (56.1). No amount of hangover, cooldown, or hysteresis timing will fix a threshold that doesn't separate gesture from non-gesture. Even Fix 3's Schmitt trigger can't help — the distance never clears the threshold during rest.

Three approaches, in increasing order of complexity:

#### Fix 4a: Lower the Global Coefficient

Reduce the default coefficient from 2.0 to a lower value:

| Coefficient | Jump Threshold | vs Resting (38) | Wave Threshold | vs Performing (14-17) |
|-------------|---------------|------------------|----------------|----------------------|
| 2.0 (current) | 56.1 | Below by 32% | 23.2 | 6.2 above highest |
| 1.5 | 48.7 | Below by 22% | 19.9 | 2.9 above highest |
| 1.0 | 41.3 | Below by 8% | 16.6 | Borderline |
| 0.5 | 33.9 | Above by 12% | 13.3 | **Below performing!** |

**Problem**: A single global coefficient can't serve all gestures. At 0.5, jump finally separates but wave's threshold (13.3) drops below its performing distance (14-17), meaning wave detections would be **missed**.

#### Fix 4b: Per-Gesture Coefficients

Allow different coefficients per gesture via the existing `threshold_coefficient` field in the Gesture struct (already present in the data model):

- wave: coefficient=2.0 (threshold 23.2 — works well)
- spin: coefficient=2.0 (threshold 21.3 — works well)
- jump: coefficient=0.5 (threshold 33.9 — separates from resting distance 38)

**Pros**: Simple, uses existing data model field. No code changes to the model, only UI to expose per-gesture tuning.
**Cons**: Requires manual tuning per gesture. User must know something is wrong and adjust.

#### Fix 4c: Resting Baseline Calibration (Background Model)

**Standard practice in one-class classification**: Compute a "background model" by sampling the user's resting pose, then set the threshold between the gesture distance and the resting distance.

**Calibration step**: At startup or on demand, sample ~60 frames (1 second) of the user standing still. Compute the DTW distance from this resting pose to each gesture's training examples.

**Formula**:
```
threshold = (average_performing_distance + average_resting_distance) / 2
```

Applied to our data:

| Gesture | Performing Avg | Resting Avg | Midpoint Threshold | Current Threshold |
|---------|---------------|-------------|-------------------|-------------------|
| wave | 15 | 41 | 28.0 | 23.2 |
| spin | 13 | 38 | 25.5 | 21.3 |
| jump | 13 | 38 | 25.5 | 56.1 |

This gives jump a threshold of 25.5, which cleanly separates performing (13) from resting (38) with equal margin on both sides. The existing μ+σ threshold could serve as a fallback when no baseline is available.

**Pros**: Mathematically optimal decision boundary. Adapts to each user's body and resting posture.
**Cons**: Requires a calibration UX step ("stand still for 5 seconds"). Adds complexity to the training flow.

**External review verdict**: "You cannot fix the jump gesture with timers. You must either manually override the threshold or implement the resting baseline."

---

## Part 4: Impact vs Risk Matrix

| Fix | Category | What It Fixes | Impact | Risk | Effort |
|-----|----------|---------------|--------|------|--------|
| **1. Extend hangover** (→2000ms) | Timer | Spin/wave window-memory echoes | High | None | Config change |
| **2. Global cooldown** (1500ms) | Timer | Cross-gesture round-robin echoes | High | Low | Small code change |
| **3. Schmitt trigger** (min_dist) | Signal | All same-gesture echoes (where resting > threshold) | High | Medium | Medium code change |
| **4a. Lower coefficient** (→?) | Threshold | Jump's loose threshold | Medium | Medium — hurts wave/spin | Config change |
| **4b. Per-gesture coefficients** | Threshold | Jump specifically | Medium | Low | UI change |
| **4c. Resting baseline** | Threshold | All threshold problems | High | Medium — adds UX step | Significant change |

### Recommended Implementation Order

**Immediate (stop the bleeding)**:
- **Fix 1 + Fix 2**: Extend hangover to 2000ms and add global cooldown. These are safe, simple, and eliminate spin/wave echoes plus cross-gesture cycling. No risk to accuracy. This sacrifices rapid-fire capability for reliability, which is the right trade-off right now.

**Next (the real solution)**:
- **Fix 3**: Schmitt trigger with min_distance tracking and safety_factor=1.1. This makes the system feel responsive rather than timer-gated. It adapts to the dancer's speed — fast return to rest means fast re-arm. Long-term, this replaces Fix 1 as the primary echo prevention.

**Critical for jump**:
- **Fix 4b** (per-gesture coefficient) as an immediate workaround. Set jump's coefficient to 0.5, giving threshold=33.9 (below resting distance 38).
- **Fix 4c** (resting baseline) as the proper long-term solution. This eliminates the need for manual coefficient tuning entirely.

### Fallback: Sakoe-Chiba Band Tightening

If echoes persist after Fixes 1-3, try reducing the Sakoe-Chiba band from 0.15 to 0.10 or 0.05. When the band is too wide, DTW has freedom to warp the tail end of a gesture (the "ghost") to look like the start of a new one. A tighter band constrains this.

---

## Part 5: What We Don't Know Yet

1. **Was the user continuously performing during echo chains?** The 9-second jump echo chain shows distance 9-14 throughout. This could mean the user performed repeated jumps for 9 seconds (correct detections, not echoes), or it could mean one jump's data lingered in the window while the user stood still. Only the user can clarify.

2. **Are some detected gestures actually cross-gesture confusion?** Some spin detections (distance 19-20, margin 4-13%) are borderline. With the current data, it's unclear if these are correctly detected spins or false positives from wave-like movements.

3. **What's the ideal fire rate for dance performance?** If the dancer expects one fire per discrete gesture execution, the system should fire once and then require a full return to rest before re-arming. If the dancer expects repeated fires during sustained movement (e.g., continuous spinning), the system should fire periodically. The current behavior fires every ~1.4 seconds, which might be correct for sustained detection or too frequent for discrete detection.

4. **How do more training examples affect the distances?** With only n=4 examples per gesture, the statistical threshold has high variance. A single inconsistent training example can inflate σ dramatically. 8-10 examples per gesture would produce more stable statistics and potentially tighter thresholds. This is especially important for jump (σ/μ = 0.56).

5. **Would window flushing help?** An alternative to Fix 1: fill the buffer with neutral/zero data upon a confirmed hit, rather than waiting for the window to slide out. This prevents the "ghost signal" immediately at the cost of a ramp-up period while the buffer refills. The 2000ms hangover (Fix 1) achieves a similar result with less code complexity, so this is deferred unless needed.

---

## Part 6: Raw Data Reference

### Gesture Parameters

```
wave: threshold=23.2, μ=10.0, σ=6.6, coefficient=2.0, 4 examples × 122-123 frames
jump: threshold=56.1, μ=26.5, σ=14.8, coefficient=2.0, 4 examples × 121-122 frames
spin: threshold=21.3, μ=14.5, σ=3.4, coefficient=2.0, 4 examples × 121-122 frames
```

### Recognition Config (before this fix)

```
cooldown_ms: 500
threshold_high_factor: 1.0
frames_to_fire: 3
hangover_ms: 300
rearm_threshold_factor: 1.3
extended_hangover_ms: 500
sakoe_chiba_band: 0.15
```

### Recognition Config (after echo fixes)

```
cooldown_ms: 500
threshold_high_factor: 1.0
frames_to_fire: 3
hangover_ms: 300
rearm_safety_factor: 1.1       # Schmitt trigger (replaces rearm_threshold_factor)
max_recovery_ms: 5000          # Safety valve (replaces extended_hangover_ms)
global_cooldown_ms: 1500       # NMS: suppress all gestures after any hit
sakoe_chiba_band: 0.15
```

### All 60 HITs (chronological)

```
 t(ms)    gesture  dist  thresh  margin
  1366    jump       39      56   30.1%
  2783    jump       38      56   32.0%
  7048    spin       14      21   33.9%
  8464    spin       12      21   41.4%
  9276    wave       14      23   38.9%
 10297    spin       14      21   34.7%
 11716    spin       13      21   40.6%
 13133    spin        9      21   56.4%
 14960    jump        9      56   83.9%
 16385    jump       12      56   79.2%
 17801    jump       12      56   79.0%
 19222    jump       13      56   76.7%
 20644    wave       17      23   28.5%
 21659    spin       14      21   34.8%
 23081    spin       10      21   51.2%
 24908    jump       12      56   77.8%
 26333    jump       13      56   76.3%
 27750    jump       13      56   76.3%
 29175    jump       14      56   75.5%
 30391    spin       14      21   34.6%
 31814    spin       13      21   39.9%
 33233    spin       15      21   28.0%
 34657    wave       15      23   37.3%
 35671    spin       16      21   24.5%
 37090    spin       11      21   48.4%
 38515    spin       13      21   38.2%
 39737    jump       16      56   71.3%
 40549    spin       15      21   29.1%
 41360    wave       15      23   36.3%
 43597    spin       12      21   45.8%
 45018    spin       13      21   38.1%
 46441    spin       15      21   30.3%
 47859    spin       11      21   47.0%
 49276    spin       15      21   29.3%
 50701    wave       16      23   31.9%
 51715    spin       16      21   26.6%
 53134    spin       12      21   44.3%
 54554    jump       13      56   77.4%
 55977    jump       13      56   77.0%
 57400    jump       11      56   80.4%
 58822    jump       14      56   75.5%
 60246    jump       11      56   79.9%
 61666    jump       12      56   78.7%
 63087    jump       14      56   74.3%
 64710    spin       14      21   33.1%
 66134    spin       12      21   43.3%
 67555    jump       16      56   71.3%
 68975    jump       15      56   72.9%
 70395    jump       17      56   70.3%
 72018    jump       17      56   70.5%
 73436    jump       16      56   71.1%
 74857    jump       17      56   70.3%
 76278    jump       17      56   69.4%
 77697    jump       16      56   71.8%
 79120    jump       15      56   72.5%
 79726    wave       16      23   31.1%
 81152    wave       16      23   29.6%
 82570    wave       17      23   28.0%
 83385    spin       19      21   13.0%
 84803    spin       20      21    4.6%
```

### Echo Cluster Detail

```
Cluster  Time      Gesture  Hits  Echoes  Echo Distances
 1        1366ms   jump     2     1       [38]
 2        7048ms   spin     2     1       [12]
 3        9276ms   wave     1     0       —
 4       10297ms   spin     3     2       [13, 9]
 5       14960ms   jump     4     3       [12, 12, 13]
 6       20644ms   wave     1     0       —
 7       21659ms   spin     2     1       [10]
 8       24908ms   jump     4     3       [13, 13, 14]
 9       30391ms   spin     3     2       [13, 15]
10       34657ms   wave     1     0       —
11       35671ms   spin     3     2       [11, 13]
12       39737ms   jump     1     0       —
13       40549ms   spin     1     0       —
14       41360ms   wave     1     0       —
15       43597ms   spin     5     4       [13, 15, 11, 15]
16       50701ms   wave     1     0       —
17       51715ms   spin     2     1       [12]
18       54554ms   jump     7     6       [13, 11, 14, 11, 12, 14]
19       64710ms   spin     2     1       [12]
20       67555ms   jump     9     8       [15, 17, 17, 16, 17, 17, 16, 15]
21       79726ms   wave     3     2       [16, 17]
22       83385ms   spin     2     1       [20]
```

### State Machine Transition Counts

```
jump:  28× Idle→Building, 28× Building→Peak, 27× Peak→Recovery, 28× Recovery→Idle
spin:  27× Idle→Building, 25× Building→Peak, 24× Peak→Recovery, 24× Recovery→Idle
wave:  10× Idle→Building,  8× Building→Peak,  9× Peak→Recovery,  8× Recovery→Idle
```

All Recovery→Idle transitions use reason: `extended_hangover_complete` (never `hangover_complete_distance_exceeded`).

---

## Implementation Checklist

### Phase 1: Timer Fixes (Ship Immediately)
- [x] ~~Increase `extended_hangover_ms` from 500 to 2000~~ (superseded by Phase 2)
- [x] Add `global_cooldown_ms` field to RecognitionConfig (default: 1500)
- [x] Add `last_any_hit_time: Option<Instant>` to Recognizer
- [x] Block Idle→Building entry during global cooldown
- [x] Set `last_any_hit_time` when any gesture fires
- [x] Global cooldown suppression passed to state machine via `in_global_cooldown` param
- [x] Test: verify spin/wave echoes eliminated — **0 echoes across 43 HITs (2026-01-29 test)**
- [x] Test: verify cross-gesture cycling eliminated — **min gap 2029ms, all beyond 1500ms global cooldown**

### Phase 2: Schmitt Trigger Hysteresis
- [x] Add `min_distance_in_recovery` to GestureState (initialize to f32::MAX)
- [x] Add `rearm_safety_factor` to RecognitionConfig (default: 1.1)
- [x] Add `max_recovery_ms` to RecognitionConfig (default: 5000)
- [x] Replace dual-path Recovery logic with Schmitt trigger
- [x] Remove `rearm_threshold_factor` and `extended_hangover_ms` (superseded)
- [x] Update diagnostic logging: `hysteresis_cleared` vs `safety_valve_timeout`
- [x] Test: verify re-arm only after distance consistently exceeds threshold (unit test)
- [x] Test: verify safety valve fires after max_recovery_ms (unit test)
- [x] Test: verify dip below threshold blocks hysteresis re-arm (unit test)
- [x] Test: verify global cooldown blocks Building entry (unit test)

### Phase 3: Jump Threshold Fix
- [ ] Expose per-gesture coefficient in UI (Fix 4b)
- [ ] Set jump's coefficient to 0.5 as initial workaround
- [ ] Design resting baseline calibration UX (Fix 4c)
- [ ] Implement resting distance measurement
- [ ] Compute threshold as midpoint between performing and resting distances
- [ ] Test with real-world gesture performance

### Optional: Sakoe-Chiba Tuning
- [ ] If echoes persist, try band=0.10 then 0.05
- [ ] Measure accuracy impact of tighter band

---

## Appendix: Source Files

- **Log file**: `/Users/brandon/Documents/RALF/ralf-diagnostics-2026-01-29_16-43-06.log`
- **Vocabulary**: `~/Desktop/26-01-29-1645_test-vocabulary.ralf`
- **Recognizer code**: `src/engine/recognizer.rs`
- **DTW code**: `src/engine/dtw.rs`
- **Optimization plan**: `docs/plans/2026-01-30-recognition-optimization-v2-plan.md`

## Appendix: External Review Sources

- Expert review 1: Framing as temporal control system (signal processing perspective)
- Expert review 2: Detailed assessment referencing Schmitt triggers, NMS, one-class classification, and sliding window best practices
