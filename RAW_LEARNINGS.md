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

*Last updated: 2026-01-24*
