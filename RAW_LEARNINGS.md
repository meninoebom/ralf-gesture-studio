# RALF Gesture Studio — Raw Learnings

A living document capturing insights, calibrations, bugs, and best practices discovered during development and use of the gesture recognition system.

**How to use this document:**
- Add new entries with date and context
- Be specific about the problem AND the solution
- Include concrete numbers where applicable
- Tag entries with categories for searchability

---

## Categories

- `[CALIBRATION]` — Timing, thresholds, default values
- `[UX]` — User experience discoveries, workflow insights
- `[ALGORITHM]` — DTW behavior, recognition logic
- `[BUG]` — Code bugs and their fixes
- `[PLATFORM]` — OS-specific issues
- `[INTEGRATION]` — OSC, external tools, pipeline issues

---

## Learnings Log

### 2026-01-26 — Statistical Threshold (μ+σ) for Auto-Calibration

`[ALGORITHM]` `[RESEARCH]` `[CALIBRATION]`

**Problem:** Manual threshold calibration is tedious and doesn't scale. Each gesture needs its own threshold, and there's no principled way to set it. Users see meaningless numbers (500? 2000? 10000?) and have to guess.

**Discovery:** The Gesture Recognition Toolkit (GRT) uses a statistical approach that's been battle-tested in production systems.

**Source:** [nickgillian/grt](https://github.com/nickgillian/grt) — C++ library for real-time gesture recognition

**The μ+σ Method:**

1. During training, compute DTW distances between all examples of a gesture
2. Find the "best template" (example with lowest average distance to others)
3. Compute mean (μ) and standard deviation (σ) of distances to best template
4. Set threshold = μ + (σ × null_rejection_coeff)

**Formula:**
```
threshold_gesture = mean_distance + (std_distance × null_rejection_coeff)
```

**Default null_rejection_coeff = 3.0** (allows matches within 3 sigma of training mean)

**Why this works:**
- **Adapts to gesture complexity**: A precise clap has low variance → tight threshold. A flowing dance phrase has high variance → looser threshold.
- **One global parameter**: `null_rejection_coeff` works for all gestures. No per-gesture tuning.
- **Auto-updates**: Threshold recalculates when training examples change.
- **Minimum examples**: Need at least 3 examples for meaningful statistics.

**Implementation pattern:**
```rust
struct GestureTemplate {
    examples: Vec<Sequence>,
    best_template: Sequence,  // Example with lowest avg distance to others
    training_mu: f32,         // Mean distance during training
    training_sigma: f32,      // Std dev of distances during training
    threshold: f32,           // mu + sigma * coefficient
}
```

**Optional enhancement — Winner-Take-All Hybrid:**
Also compute likelihood scores (1/distance normalized across all gestures) and require best gesture to have >50% likelihood. Provides dual protection against ambiguous matches.

**Full documentation:** See `.claude/commands/dtw-gesture-recognition.md` section "Statistical Threshold: The μ+σ Approach (GRT Method)"

---

### 2026-01-26 — Z-Normalization and Path-Length Normalization for DTW

`[ALGORITHM]` `[RESEARCH]`

**Problem:** Raw DTW distances are sensitive to:
1. **Scale**: 0-1 normalized skeleton coords vs 0-640 pixel coords produce wildly different distances
2. **Offset**: Person standing left vs right of frame shifts all coordinates
3. **Duration**: Longer gestures accumulate more distance even if equally well-matched

**Solution 1: Z-Normalization (per-sequence)**

Normalize each sequence independently to mean=0, std=1 before comparison.

Formula: `z = (x - mean) / std`

```rust
fn z_normalize(sequence: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let all_values: Vec<f32> = sequence.iter().flatten().copied().collect();
    let n = all_values.len() as f32;
    let mean = all_values.iter().sum::<f32>() / n;
    let variance = all_values.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    let std = if variance.sqrt() < 1e-10 { 1.0 } else { variance.sqrt() };

    sequence.iter()
        .map(|frame| frame.iter().map(|x| (x - mean) / std).collect())
        .collect()
}
```

**Key insight:** Each sequence uses its OWN mean/std. This is NOT global normalization across all training data.

**Edge case:** If std=0 (constant sequence), set std=1 to avoid division by zero.

**Solution 2: Path-Length Normalization**

Divide raw DTW distance by (n + m) where n and m are sequence lengths.

Formula: `normalized_distance = raw_distance / (n + m)`

**Key insight:** Most DTW libraries do NOT apply this automatically. Recommended by Sakoe-Chiba (1978) for symmetric2 step pattern.

**Combined effect:** Distances become comparable regardless of input scale, baseline offset, or gesture duration.

**Reference implementations:** scipy.stats.zscore, tslearn.TimeSeriesScalerMeanVariance, dtaidistance.preprocessing.znormal

**Full documentation:** See `.claude/commands/dtw-gesture-recognition.md` sections 2a, 2b, 2c.

---

### 2026-01-24 — Buffer Reset After Detection: The "Bounce Back" Pattern

`[ALGORITHM]` `[RESEARCH]`

**Problem:** After detecting a gesture, the distance stays low because the gesture frames remain in the sliding window. It takes ~3 seconds for them to naturally "slide out", during which the system can't reliably detect the next gesture.

**Research finding:** Voice recognition and keyword spotting systems solve this with **buffer reset after detection**.

**Evidence from academic literature:**

1. **"Contrastive Learning with Audio Discrimination for Customizable Keyword Spotting" (2024)** - [arXiv:2401.06485](https://arxiv.org/html/2401.06485)
   > "When a keyword score surpasses the detection threshold, the system **resets the model state** and introduces a **1-second cooldown period** to prevent recurrent wake-up on the same keyword."

2. **"Streaming keyword spotting on mobile devices" (2020)** - [arXiv:2005.06720](https://arxiv.org/pdf/2005.06720)
   > Documents up to **2x accuracy reduction** when RNN states aren't reset between sequences. Shows ring buffer implementation patterns.

3. **"Attentive Decision-making and Dynamic Resetting of SRNNs" (2022)** - [ACM DL](https://dl.acm.org/doi/abs/10.1145/3546790.3546795)
   > "Network dynamics become saturated... requiring network state resetting events."

4. **"Gesture Recognition Using DTW and Kinect"** - [ResearchGate](https://www.researchgate.net/publication/324177668)
   > Addresses buffer issues: "cannot identify when a movement has finished, causing partial fill of the test data buffer"

**Why the 1-second cooldown is universal:**
- Prevents overlapping sliding windows from generating duplicate detections
- Addresses state saturation in continuous streams
- Empirically validated across keyword spotting, wake word detection, and ASR systems

**Solution for gesture recognition:**
```rust
// After hit fires:
1. Clear the frame buffer
2. System enters "collecting data" state (not enough frames to match)
3. Natural gap while buffer refills (~1-1.5 seconds)
4. Ready to detect next gesture
```

**Key insight:** The "bounce back" to neutral is achieved by clearing the buffer, which returns the system to "insufficient data" state. Distance effectively becomes undefined/infinity until enough new frames arrive.

---

### 2026-01-24 — Preserve Edge Detection State After Buffer Clear

`[BUG]` `[ALGORITHM]`

**Problem:** After implementing buffer clear, hits fired in an infinite loop. Once a gesture triggered, it would keep triggering every ~1.5 seconds (the time for the buffer to refill).

**Root cause:** When clearing the buffer after a hit, we were also resetting `was_below_threshold = false`. This made the system "forget" that it was already below threshold.

**The bug cycle:**
1. Distance crosses below threshold → hit fires
2. Buffer clears, `was_below_threshold = false` ← BUG
3. ~1.5s later, buffer refills with 90 frames
4. Distance computed, still below threshold (user hasn't moved)
5. Edge detection sees: `below_threshold=true`, `was_below_threshold=false` → new crossing!
6. Hit fires → buffer clears → repeat infinitely

**Solution:** Do NOT reset `was_below_threshold` when clearing the buffer. Keep it `true` so the user must move away (distance goes above threshold) before another hit can fire.

```rust
// After hit fires:
if best_hit.is_some() {
    self.buffer.clear();
    // Clear display distances while buffer refills
    for gesture in &mut self.gestures {
        gesture.current_distance = None;
        // DO NOT reset was_below_threshold!
        // User must move away before next hit can fire.
    }
}
```

**Key insight:** Buffer clear resets the *data*, but edge detection state must persist. The user must complete a full cycle (below → above → below) to trigger another hit. This prevents "camping" on a gesture position.

---

### 2026-01-24 — Edge Detection is Correct, Debounce is Wrong

`[ALGORITHM]` `[UX]`

**Problem:** Hit detection using "distance must stay below threshold for N ms" (debounce) missed quick gestures and felt unresponsive.

**Research findings:** Surveyed Wekinator, librosa onset detection, signal processing literature. Consensus:
- **Edge detection** (fire when crossing threshold) is correct for gesture recognition
- **Debounce** is for filtering noise, but it blocks fast movements
- **Refractory/cooldown period** is sufficient for preventing double-triggers

**Solution:** Fire immediately when distance crosses below threshold, not when it stays below.

```rust
// Edge detection: fire on TRANSITION from above to below threshold
let below_threshold = distance < gesture.threshold;
let is_crossing_down = below_threshold && !gesture.was_below_threshold;
let not_in_cooldown = !gesture.in_cooldown(cooldown_duration);

if is_crossing_down && not_in_cooldown {
    fire_hit();
}

// Update state for next frame
gesture.was_below_threshold = below_threshold;
```

**Key insight:** Like phoneme detection in speech recognition — it's about detecting the moment you enter a recognized state, not how long you stay there. Quick gestures (claps, stomps) may only dip below threshold for 50-100ms.

**Config simplification:**
- Removed: `confirm_ms` (debounce)
- Kept: `cooldown_ms` (400ms default, allows ~2.5 hits/sec)

---

### 2026-01-24 — DTW Performance: Skip Frames to Prevent GUI Freeze

`[ALGORITHM]` `[BUG]`

**Problem:** Switching to Performance mode caused "spinning beach ball" (GUI freeze). The app became unresponsive.

**Root cause:** DTW is O(n×m) per comparison. With:
- 180-frame window
- 3 gestures × 5 examples = 15 DTW computations per frame
- 60 frames/second arriving via OSC
- Each DTW: 180 × 180 = 32,400 matrix cells
- Each cell: Euclidean distance on 68-dimensional vector

Total: **millions of operations per second**, all on the main thread.

The GUI polls ALL queued frames and processes EACH one. When processing falls behind, the backlog grows infinitely.

**Solution:** In Performance mode, only process the most recent frame when there's a backlog:

```rust
// In Performance mode, only process the most recent frame to avoid backlog
if self.mode == AppMode::Performance && frames.len() > 1 {
    let last_frame = frames.pop().unwrap();
    frames.clear();
    frames.push(last_frame);
}
```

**Key insight:** For real-time gesture recognition, it's better to skip frames than to fall behind. The dancer's current position matters more than processing every historical frame.

**Future optimization options:**
1. FastDTW (linear time approximation)
2. Sakoe-Chiba band constraint (O(n×k) instead of O(n×m))
3. Move DTW to background thread
4. Reduce window size

---

### 2026-01-24 — Debounce Required for Stable Hit Detection

`[ALGORITHM]` `[CALIBRATION]`

**Problem:** When distance oscillates around the threshold (e.g., 1480↔1520 with threshold 1500), hits fire every frame the distance dips below. This creates dozens of spurious hits per second.

**Root cause:** Checking `distance < threshold` alone fires on every qualifying frame.

**Solution:** Debounce — require distance to stay below threshold for N milliseconds before firing.

**Recommended value:** `confirm_ms = 80ms`
- Too low (0-30ms): Still get spurious hits from noise
- Too high (200ms+): Feels laggy, misses quick gestures
- Sweet spot: 50-100ms filters noise while staying responsive

**Code pattern:**
```rust
// Track when distance first dropped below threshold
if distance < threshold {
    if below_threshold_since.is_none() {
        below_threshold_since = Some(Instant::now());
    }
} else {
    below_threshold_since = None;  // Reset when above
}

// Only fire if confirmed
let confirmed = below_threshold_since
    .map(|t| t.elapsed().as_millis() >= confirm_ms)
    .unwrap_or(false);
```

---

### 2026-01-24 — Hysteresis Breaks Dance Flow (Do Not Use)

`[ALGORITHM]` `[UX]`

**Problem:** Initial implementation used hysteresis — requiring distance to go ABOVE threshold×1.2 before another hit could fire. This was intended to prevent double-triggers.

**Discovery:** Dancers cannot use this. It forces them to "return to rest position" between every gesture. Dance is fluid; gestures chain together.

**Solution:** Remove hysteresis entirely. Use only cooldown (refractory period) for rate limiting.

**Key insight:** The mental model "do gesture → hit fires → wait for cooldown → do gesture again" is intuitive. The mental model "do gesture → hit fires → return to rest → do gesture again" is not.

---

### 2026-01-24 — Cooldown (Refractory Period) Calibration

`[CALIBRATION]`

**Problem:** What's the minimum time between hits that prevents accidental double-triggers but allows intentional rapid gestures?

**Testing results:**
- 100ms: Too fast, get accidental doubles
- 300ms: Works but feels limiting for fast sequences
- 500ms: Good balance — allows 2 hits/sec, prevents accidents
- 1000ms: Too slow for dance

**Recommended default:** `refractory_ms = 500ms`

**Make it configurable:** Different performances may need different values. Expose in UI with range 100-2000ms.

---

### 2026-01-24 — Baseline Recording Enables Auto-Calibration

`[CALIBRATION]` `[UX]`

**Problem:** Users don't know what threshold values to set. Raw DTW distances are meaningless numbers (500? 2000? 10000?).

**Solution:** Record a "baseline" of the user standing still (rest position). Then auto-calibrate thresholds:

```
threshold = 0.8 × average_distance(baseline, gesture_examples)
```

**Rationale:** A gesture should match better (lower distance) than the baseline. Setting threshold to 80% of baseline distance gives headroom.

**UX flow:**
1. Record baseline first (before training gestures)
2. Train gestures
3. Switch to Performance mode → thresholds auto-calibrate
4. User can still manually adjust if needed

---

### 2026-01-24 — Use 127.0.0.1 Not localhost on macOS

`[PLATFORM]` `[BUG]`

**Problem:** OSC messages were occasionally delayed or dropped when using `"localhost"` as the output host.

**Root cause:** macOS performs DNS resolution for "localhost" which can introduce latency, especially if mDNS is involved.

**Solution:** Always use `"127.0.0.1"` for local OSC communication.

**Applies to:** Both sender (output host) and any configuration that specifies local addresses.

---

### 2026-01-24 — UI Must Be Readable from 6+ Feet

`[UX]`

**Problem:** Dancers look at the screen while moving across a room. Standard font sizes are unreadable.

**Solution:**
- Increase base font size by 30% (`font_id.size *= 1.3`)
- Make hit indicator HUGE: 80px panel height, 48pt text
- Show current distance in large text (22pt), threshold smaller
- Use high-contrast colors (bright green, not subtle)

**Key insight:** The most important information (HIT / current distance) should be visible in peripheral vision while dancing.

---

### 2026-01-24 — Normalized DTW Required for Comparable Distances

`[ALGORITHM]`

**Problem:** Raw DTW distance depends on sequence length. A 5-second gesture has higher raw distance than a 2-second gesture, even if equally well-matched.

**Solution:** Normalize by dividing by average sequence length:

```rust
let path_length = (seq1.len() + seq2.len()) as f32 / 2.0;
dtw_distance / path_length
```

**Impact:** Makes thresholds transferable across gestures of different durations.

---

### 2026-01-24 — Training Session Audio Cues Are Essential

`[UX]`

**Problem:** Dancers can't watch the screen during training — they need to focus on their movement.

**Solution:** Audio cues for every state transition:
- Countdown ticks (300 Hz, short) — "get ready"
- Capture start (800 Hz, long) — "GO"
- Capture end (600 Hz, long) — "stop"
- Session complete (double ding 1000 Hz) — "done"

**Implementation:** Generate sine waves with `rodio`. No external audio files needed.

**Key insight:** The sounds become Pavlovian — dancers learn the rhythm and can train with eyes closed.

---

### 2026-01-24 — Configuration Structs vs Scattered Primitives

`[BUG]` `[CODE QUALITY]`

**Problem:** App had `refractory_ms: u64` as a standalone field. When adding `confirm_ms`, this created scattered related values.

**Solution:** Group related configuration into structs:

```rust
// BAD
struct App {
    refractory_ms: u64,
    confirm_ms: u64,
    // ... scattered
}

// GOOD
struct RecognitionConfig {
    confirm_ms: u64,
    refractory_ms: u64,
}

struct App {
    recognition_config: RecognitionConfig,
}
```

**Benefits:**
- Clear what settings belong together
- Easy to pass config to functions
- Can derive Default, Clone, Serialize
- Adding new related settings is obvious

---

### 2026-01-24 — Reset Debounce Timer After Hit Fires

`[BUG]`

**Problem:** After a hit fired, `below_threshold_since` wasn't reset. If the user held the gesture pose (staying below threshold), another hit would fire immediately after cooldown ended.

**Solution:** Reset `below_threshold_since = None` after firing a hit. This requires the distance to go above threshold and come back down (natural gesture completion) before another hit can fire.

```rust
if is_hit {
    gesture.record_hit();
    gesture.below_threshold_since = None;  // CRITICAL: Reset debounce
}
```

---

### 2026-01-24 — Window Size vs Buffer Size

`[CALIBRATION]`

**Terms:**
- **Buffer size:** Total frames kept in memory (e.g., 600 = ~10 seconds at 60fps)
- **Window size:** Frames used for matching (e.g., 180 = ~3 seconds)

**Current defaults:**
- Buffer: 600 frames (10 seconds) — enough history for any gesture
- Window: 180 frames (3 seconds) — typical gesture duration
- Minimum frames before matching: window_size / 2 = 90 frames

**Insight:** Window size should roughly match expected gesture duration. Too short = misses slow gestures. Too long = includes irrelevant movement.

---

## Template for New Entries

```markdown
### YYYY-MM-DD — Title

`[CATEGORY]` `[CATEGORY]`

**Problem:** What went wrong or what question arose?

**Discovery/Root cause:** What did we learn?

**Solution:** What fixed it or what's the recommendation?

**Concrete values:** (if applicable)

**Code pattern:** (if applicable)
```

---

## Quick Reference: Recommended Defaults

| Parameter | Value | Notes |
|-----------|-------|-------|
| Debounce (confirm_ms) | 80ms | Filters noise, stays responsive |
| Cooldown (refractory_ms) | 500ms | Allows 2 hits/sec |
| Training reps | 5 | Good balance of data vs time |
| Training duration | 3.0s | Long enough for most gestures |
| Training rest | 2.0s | Enough to reset position |
| Countdown | 3.0s | Standard "3, 2, 1" |
| Baseline duration | 3.0s | Enough frames to average |
| Buffer size | 600 frames | ~10 sec at 60fps |
| Window size | 180 frames | ~3 sec |
| Threshold | Auto-calibrate | 80% of baseline distance |
| OSC host | 127.0.0.1 | NOT "localhost" |

---

*Last updated: 2026-01-26*
