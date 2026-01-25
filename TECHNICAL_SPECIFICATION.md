# RALF Gesture Studio: Technical Specification

*Version 2.0 — January 2026*

A complete specification for building a real-time gesture recognition application for dancers and performers. This document captures all requirements, learned lessons, and technology recommendations for building the system from scratch.

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Domain](#2-problem-domain)
3. [Functional Requirements](#3-functional-requirements)
4. [Data Model](#4-data-model)
5. [Recognition Engine](#5-recognition-engine)
6. [OSC Communication](#6-osc-communication)
7. [GUI Specification](#7-gui-specification)
8. [Training Workflow](#8-training-workflow)
9. [Lessons Learned & Critical Issues](#9-lessons-learned--critical-issues)
10. [Technology Recommendations](#10-technology-recommendations)
11. [Implementation Guidance](#11-implementation-guidance)
12. [Testing Strategy](#12-testing-strategy)
13. [Appendices](#13-appendices)

---

## 1. Executive Summary

### What This Application Does

RALF Gesture Studio receives real-time skeleton tracking data (e.g., from MoveNet via OSC), allows users to train gesture recognition models through recorded examples, and emits OSC messages ("hits") when trained gestures are recognized during performance.

### Key Use Case

A dancer trains 3-5 gestures (e.g., "wave", "jump", "spin") by performing each gesture 5 times. During performance, the system continuously monitors their movement and triggers audio/visual events in Ableton Live when gestures are recognized.

### Design Principles

| Principle | Rationale |
|-----------|-----------|
| **GUI-first** | Dancers shouldn't need terminal skills |
| **Flow-state friendly** | Training UX minimizes cognitive load |
| **Portable data** | Vocabularies are self-contained files |
| **Simple integration** | OSC output is trivial to consume |
| **Immediate feedback** | Visual and audio indicators for all states |

---

## 2. Problem Domain

### What is Gesture Recognition?

Comparing a live stream of body positions to previously recorded examples to determine if the performer is executing a known gesture.

### Why Dynamic Time Warping (DTW)?

| Approach | Pros | Cons |
|----------|------|------|
| **DTW** | Handles varying speeds, no training required, works with few examples | O(n×m) per comparison |
| Machine Learning | Can generalize better | Needs many examples, complex setup |
| Template matching | Simple | Speed-sensitive, brittle |

**Decision**: DTW is ideal for this use case because:
- Dancers perform gestures at varying speeds
- Users want to train with 3-10 examples, not hundreds
- No GPU or model training infrastructure needed
- Interpretable (it's just "how similar is this movement?")

### Data Flow

```
┌──────────────┐     OSC      ┌─────────────────────┐     OSC      ┌─────────────────┐
│   MoveNet    │ ──────────►  │  RALF Gesture       │ ──────────►  │   Ableton Live  │
│   Skeleton   │  /wek/inputs │  Studio             │  /gesture/N  │   Max4Live      │
│   Tracker    │   (68 floats)│                     │   (hit msg)  │                 │
└──────────────┘              └─────────────────────┘              └─────────────────┘
```

---

## 3. Functional Requirements

### 3.1 Core Features

| ID | Feature | Priority | Description |
|----|---------|----------|-------------|
| F1 | Vocabulary management | Must | Create, open, save, rename vocabularies |
| F2 | Gesture management | Must | Add, rename, delete gestures |
| F3 | Training sessions | Must | Structured countdown → capture → rest workflow |
| F4 | Real-time recognition | Must | Continuous DTW matching against trained examples |
| F5 | OSC input | Must | Receive skeleton data from external tracker |
| F6 | OSC output | Must | Send hit messages to downstream systems |
| F7 | Threshold adjustment | Must | Real-time tuning of recognition sensitivity |
| F8 | Audio feedback | Should | Beeps/tones for training session states |
| F9 | Baseline recording | Should | Record "rest position" for auto-calibration |
| F10 | Hit debounce | Must | Prevent rapid-fire hits from threshold oscillation |
| F11 | Hit cooldown | Must | Minimum time between hits for same gesture |

### 3.2 Application Modes

**Training Mode**
- Full vocabulary/gesture editing
- Training session workflow
- Baseline recording
- No active recognition

**Performance Mode**
- Recognition active
- Real-time distance monitoring
- Threshold adjustment
- Hit log display
- Read-only vocabulary (no editing)

### 3.3 Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Start training session (Training mode only) |
| Escape | Cancel training session / close dialogs |
| Cmd/Ctrl+N | New vocabulary |
| Cmd/Ctrl+O | Open vocabulary |
| Cmd/Ctrl+S | Save As |

---

## 4. Data Model

### 4.1 Hierarchy

```
Vocabulary
├── metadata (name, version, timestamps)
├── input config (port, address)
├── output config (host, port)
├── baseline (optional rest position recording)
└── gestures[]
    ├── id, name, osc_address, threshold
    └── examples[]
        ├── metadata (timestamp, duration, frame_count)
        └── frames[][] (the actual motion data)
```

### 4.2 Schema

```rust
// Vocabulary - root container, one per file
struct Vocabulary {
    version: String,           // "1.0"
    name: String,              // User-editable
    created_at: DateTime,
    modified_at: DateTime,
    input: InputConfig,
    output: OutputConfig,
    baseline: Option<Vec<Vec<f32>>>,  // Rest position frames
    gestures: Vec<Gesture>,
    next_gesture_id: u32,      // For unique IDs
}

struct InputConfig {
    port: u16,                 // Default: 6448
    address: String,           // Default: "/wek/inputs"
}

struct OutputConfig {
    host: String,              // Default: "127.0.0.1"
    port: u16,                 // Default: 12000
}

struct Gesture {
    id: u32,                   // Unique within vocabulary
    name: String,              // User-editable
    osc_address: String,       // Default: "/gesture/{id}"
    threshold: f32,            // Recognition threshold
    created_at: DateTime,
    examples: Vec<Example>,
}

struct Example {
    recorded_at: DateTime,
    duration_ms: u64,
    frame_count: usize,
    frames: Vec<Vec<f32>>,     // [frame_count][dimensions]
}
```

### 4.3 File Format

**Extension**: `.ralf`

**Format**: JSON (human-readable, debuggable, sufficient performance)

**Example**:
```json
{
  "version": "1.0",
  "name": "House Foundations",
  "created_at": "2025-01-21T10:30:00Z",
  "modified_at": "2025-01-21T14:22:00Z",
  "input": { "port": 6448, "address": "/wek/inputs" },
  "output": { "host": "127.0.0.1", "port": 12000 },
  "baseline": [[0.1, 0.2, ...], [0.1, 0.2, ...], ...],
  "gestures": [
    {
      "id": 1,
      "name": "wave",
      "osc_address": "/gesture/1",
      "threshold": 1500.0,
      "examples": [...]
    }
  ]
}
```

### 4.4 Storage

**Default location**: `~/Documents/RALF/`

**Auto-save**: After every meaningful change (rename, add gesture, complete training, threshold change)

---

## 5. Recognition Engine

### 5.1 Algorithm: Dynamic Time Warping

DTW finds the optimal alignment between two sequences that may vary in speed.

**Core algorithm**:
```
function dtw_distance(seq1, seq2):
    n, m = len(seq1), len(seq2)
    cost = matrix of size (n+1) × (m+1), initialized to infinity
    cost[0][0] = 0

    for i in 1..=n:
        for j in 1..=m:
            dist = euclidean_distance(seq1[i-1], seq2[j-1])
            cost[i][j] = dist + min(
                cost[i-1][j-1],  // match (diagonal)
                cost[i-1][j],    // insertion
                cost[i][j-1]     // deletion
            )

    return cost[n][m]
```

**Normalization**: Divide by average sequence length to make distances comparable:
```
normalized_distance = dtw_distance / ((len(seq1) + len(seq2)) / 2)
```

### 5.2 Continuous Matching

During performance mode:

```
for each incoming_frame:
    buffer.push(incoming_frame)

    if buffer.len() < minimum_frames:
        continue

    window = buffer.recent(window_size)

    for each gesture with examples:
        best_distance = infinity
        for each example in gesture.examples:
            distance = dtw_distance_normalized(window, example)
            best_distance = min(best_distance, distance)

        gesture.current_distance = best_distance

        // Hit detection (see Section 9 for critical details)
        if should_fire_hit(gesture, best_distance):
            emit_hit(gesture)
```

### 5.3 Hit Detection Logic (CRITICAL)

A hit fires when ALL conditions are met:

1. **Distance < Threshold** — gesture matches
2. **Confirmed for debounce period** — distance stayed below threshold for `confirm_ms`
3. **Not in cooldown** — enough time since last hit for this gesture

```rust
struct RecognitionConfig {
    confirm_ms: u64,     // Default: 80ms (debounce)
    refractory_ms: u64,  // Default: 500ms (cooldown)
}

struct GestureState {
    current_distance: Option<f32>,
    threshold: f32,
    below_threshold_since: Option<Instant>,  // For debounce
    last_hit_time: Option<Instant>,          // For cooldown
}

fn should_fire_hit(gesture: &GestureState, distance: f32, config: &RecognitionConfig) -> bool {
    let below_threshold = distance < gesture.threshold;

    // Track debounce timing
    if below_threshold {
        if gesture.below_threshold_since.is_none() {
            gesture.below_threshold_since = Some(now());
        }
    } else {
        gesture.below_threshold_since = None;
        return false;
    }

    // Check if confirmed (below threshold long enough)
    let confirmed = match gesture.below_threshold_since {
        Some(since) => since.elapsed() >= config.confirm_ms,
        None => false,
    };

    // Check cooldown
    let not_in_cooldown = match gesture.last_hit_time {
        Some(time) => time.elapsed() >= config.refractory_ms,
        None => true,
    };

    if confirmed && not_in_cooldown {
        gesture.last_hit_time = Some(now());
        gesture.below_threshold_since = None;  // Reset for next hit
        return true;
    }

    false
}
```

### 5.4 Buffer Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| Buffer size | 600 frames | ~10 seconds at 60fps |
| Window size | 180 frames | ~3 seconds for matching |
| Minimum frames | window_size / 2 | Don't match until enough data |

---

## 6. OSC Communication

### 6.1 Input (Skeleton Data)

| Setting | Default | Description |
|---------|---------|-------------|
| Port | 6448 | UDP port (Wekinator-compatible) |
| Address | /wek/inputs | OSC address filter |

**Message format**: `/wek/inputs f f f f...` (array of floats, typically 68 for 34 joints × XY)

**Important**: Use `127.0.0.1` not `localhost` to avoid DNS resolution delays on macOS.

### 6.2 Output (Hit Messages)

| Setting | Default | Description |
|---------|---------|-------------|
| Host | 127.0.0.1 | Target host |
| Port | 12000 | Target UDP port |

**Message format**: `/gesture/N` (no arguments) — simple, easy to consume in Max4Live

### 6.3 Connection States

**Input states**:
- `Listening` — socket bound, waiting for data
- `Receiving` — data arriving, show time since last packet
- `Error` — cannot bind (port in use)

**Output states**:
- `Ready` — configured, no recent sends
- `Sent` — recently sent, show time since
- `Error` — send failed

---

## 7. GUI Specification

### 7.1 Framework Requirements

- **Immediate mode** preferred (UI rebuilt each frame, simple mental model)
- **Single codebase** (no web technologies, no Electron)
- **Cross-platform** (macOS primary, Linux/Windows secondary)
- **Native file dialogs** for open/save

### 7.2 Panel Layout

```
┌─────────────────────────────────────────────────────────────────────┐
│  RALF Gesture Studio                          [Training ▼]          │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─ VOCABULARY ────────────────────────────────────────────────┐   │
│  │  Name: [editable]                        [New] [Open] [Save] │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌─ CONNECTION ────────────────────────────────────────────────┐   │
│  │  INPUT: Port [6448] ● RECEIVING          OUTPUT: ● READY     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌─ GESTURES ──────────────────────────────────────────────────┐   │
│  │  ● wave (5)    /gesture/1    [Select]                        │   │
│  │  ○ jump (0)    /gesture/2    [Select]                        │   │
│  │  [+ Add Gesture]                                              │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌─ BASELINE ──────────────────────────────────────────────────┐   │
│  │  ✓ Recorded (180 frames)              [Re-record Baseline]   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌─ TRAIN ─────────────────────────────────────────────────────┐   │
│  │  Reps: [5]  Duration: [3.0s]  Rest: [2.0s]                   │   │
│  │                   [ START TRAINING ]                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Performance Mode Monitor

```
┌─ GESTURE MONITOR ──────────────────────────────────────────────────┐
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                        ● wave                                │   │  ← Large hit indicator
│  └─────────────────────────────────────────────────────────────┘   │     (80px, 48pt font)
│                                                                     │
│  Gesture      Distance   < Threshold          Hit                  │
│  wave (5)        847     [====●=====] 1500    ● HIT                │  ← Distance in LARGE font
│  jump (3)       2341     [===●======] 1200                         │
│                                                                     │
│  Debounce: [80ms]  Cooldown: [500ms]                               │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.4 Visual Design Requirements

| Element | Requirement |
|---------|-------------|
| Font size | Large enough to read from 6+ feet away |
| Hit indicator | Huge, unmissable, visible in peripheral vision |
| Colors | High contrast (green=good, red=recording, yellow=waiting) |
| Status indicators | Clear state at a glance |

---

## 8. Training Workflow

### 8.1 Training Session State Machine

```
         START (spacebar/button)
                │
                ▼
┌────────┐    ┌───────────┐    ┌───────────┐    ┌──────────┐
│  IDLE  │───►│ COUNTDOWN │───►│ CAPTURING │───►│ RESTING  │
└────────┘    │ (3s)      │    │ (3s)      │    │ (2s)     │
    ▲         └───────────┘    └───────────┘    └────┬─────┘
    │                                                │
    │  ┌───────────┐                                │ more reps?
    └──│ COMPLETE  │◄───────────────────────────────┘ no
       └───────────┘                          yes ───┘ (→ COUNTDOWN)
```

### 8.2 Audio Cues

| Event | Sound | Frequency | Duration |
|-------|-------|-----------|----------|
| Countdown tick | Low beep | 300 Hz | 80ms |
| Capture start | High beep | 800 Hz | 300ms |
| Capture end | Medium beep | 600 Hz | 300ms |
| Session complete | Double ding | 1000 Hz | 2 × 150ms |

**Implementation**: Generate sine waves programmatically. No external audio files.

### 8.3 Baseline Recording

**Purpose**: Record the user's "rest position" to:
1. Auto-calibrate thresholds (gesture should be closer than rest)
2. Provide reference for "how far from rest is this movement"

**Workflow**:
1. User clicks "Record Baseline"
2. 3-second countdown
3. 3-second recording (user stands still)
4. Save as `vocabulary.baseline`

**Auto-calibration**: Set each gesture's threshold to 80% of the average distance from baseline to its examples.

---

## 9. Lessons Learned & Critical Issues

This section documents problems encountered during development and their solutions. **These are essential for anyone rebuilding the system.**

### 9.1 Issue: Hysteresis Breaks Dance Flow

**Problem**: Initial implementation required the distance to go ABOVE the threshold before another hit could fire (hysteresis). This forced dancers to "return to rest position" between gestures.

**Impact**: Completely unusable for dance. Dancers need to chain movements fluidly.

**Solution**: Remove hysteresis entirely. Use only cooldown (refractory period) for rate limiting.

```rust
// BAD: Hysteresis approach
if distance < threshold && armed {
    fire_hit();
    armed = false;  // Must go above threshold to re-arm
}
if distance > threshold * 1.2 {
    armed = true;
}

// GOOD: Cooldown-only approach
if distance < threshold && !in_cooldown() {
    fire_hit();
    last_hit_time = now();
}
```

### 9.2 Issue: Threshold Oscillation Causes Hit Spam

**Problem**: When distance hovers around the threshold (e.g., oscillating between 1480 and 1520 with threshold 1500), hits fire rapidly every frame the distance dips below.

**Impact**: Dozens of hits per second when "almost" performing a gesture.

**Solution**: Debounce — require distance to stay below threshold for N milliseconds before firing.

```rust
// Track when we first dropped below threshold
if distance < threshold {
    if below_threshold_since.is_none() {
        below_threshold_since = Some(now());
    }
    // Only fire if below threshold for confirm_ms
    if below_threshold_since.elapsed() >= confirm_ms {
        fire_hit();
        below_threshold_since = None;  // Reset for next hit
    }
} else {
    below_threshold_since = None;  // Reset when above threshold
}
```

**Recommended defaults**:
- `confirm_ms`: 80ms (filters noise, still responsive)
- `refractory_ms`: 500ms (allows ~2 hits/second)

### 9.3 Issue: Threshold Scale Confusion

**Problem**: Raw DTW distance values vary wildly based on:
- Number of dimensions (68 floats vs 2 floats)
- Sequence length
- Movement magnitude

Users couldn't intuit what threshold values meant.

**Solution**:
1. Use normalized DTW (divide by average sequence length)
2. Provide visual feedback (show current distance vs threshold)
3. Auto-calibrate from baseline when possible

### 9.4 Issue: localhost vs 127.0.0.1

**Problem**: Using `"localhost"` as OSC host caused DNS resolution delays on macOS, resulting in dropped or delayed messages.

**Solution**: Always use `"127.0.0.1"` for local communication.

### 9.5 Issue: UI Too Small for Dancers

**Problem**: Dancers need to see the UI from across the room while moving. Standard font sizes were unreadable.

**Solution**:
- Increase all fonts by 30%
- Make hit indicator HUGE (80px panel, 48pt text)
- Use high-contrast colors
- Show distance in large text, threshold in smaller text

### 9.6 Issue: "Armed" Indicator Was Confusing

**Problem**: Showing "READY/wait" armed state confused users who didn't understand hysteresis.

**Solution**: Remove hysteresis (and thus the armed indicator). Simpler mental model: "distance below threshold = gesture recognized".

---

## 10. Technology Recommendations

### 10.1 Language: Rust

**Why Rust?**
- Performance for real-time processing
- Memory safety without garbage collection pauses
- Excellent cross-platform support
- Strong ecosystem for GUI and networking

**Alternatives considered**:
| Language | Verdict |
|----------|---------|
| Python | Too slow for real-time DTW, GIL issues |
| JavaScript/Electron | Heavy, slow startup, not suitable |
| Go | Good, but Rust has better GUI ecosystem |
| C++ | Manual memory management, error-prone |
| Swift | macOS only |

### 10.2 Recommended Crates

| Category | Crate | Purpose | Notes |
|----------|-------|---------|-------|
| **GUI** | `eframe`/`egui` | Immediate mode GUI | Simple, fast, cross-platform |
| **OSC** | `rosc` | OSC encoding/decoding | Mature, well-maintained |
| **Async** | `tokio` | Async runtime | For OSC networking |
| **Channels** | `crossbeam-channel` | Thread communication | Fast, ergonomic |
| **Audio** | `rodio` | Audio playback | Simple sine wave generation |
| **Serialization** | `serde` + `serde_json` | JSON files | Industry standard |
| **Time** | `chrono` | Timestamps | ISO 8601 support |
| **Paths** | `directories` | Cross-platform paths | For ~/Documents/RALF |
| **Dialogs** | `rfd` | Native file dialogs | Open/Save As |
| **Errors** | `thiserror` | Error types | Clean error handling |

### 10.3 Why NOT These Alternatives

| Technology | Why Not |
|------------|---------|
| Qt (via rust-qt) | Complex bindings, licensing concerns |
| GTK | Linux-centric, complex on macOS/Windows |
| Tauri/web | Overkill, web complexity unnecessary |
| iced | Less mature than egui, async-heavy |
| External DTW library | Simple enough to implement, full control needed |

---

## 11. Implementation Guidance

### 11.1 Module Structure

```
src/
├── main.rs                 # Entry point
├── model/
│   ├── mod.rs              # Re-exports
│   ├── vocabulary.rs       # Vocabulary, Gesture, Example
│   └── persistence.rs      # JSON save/load
├── engine/
│   ├── mod.rs              # Re-exports
│   ├── dtw.rs              # DTW algorithm
│   ├── buffer.rs           # Frame buffer, recording
│   ├── recognizer.rs       # Recognition logic, hit detection
│   └── training.rs         # Training session state machine
├── osc/
│   ├── mod.rs              # Re-exports
│   ├── receiver.rs         # Async OSC input
│   └── sender.rs           # OSC output
└── gui/
    └── mod.rs              # All GUI code (or split into panels)
```

### 11.2 Threading Model

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Thread                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │                    egui GUI                         │   │
│  │   - Render at ~60fps                                │   │
│  │   - Process user input                              │   │
│  │   - Update state                                    │   │
│  └──────────────────────────┬──────────────────────────┘   │
│                             │ poll channels                 │
│  ┌──────────────────────────┴──────────────────────────┐   │
│  │              Tokio Async Runtime                    │   │
│  │   ┌───────────────┐  ┌───────────────┐              │   │
│  │   │ OSC Receiver  │  │ OSC Sender    │              │   │
│  │   │ (async task)  │  │ (sync, main)  │              │   │
│  │   └───────────────┘  └───────────────┘              │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Key points**:
- GUI runs on main thread
- OSC receiver is async task, sends frames via channel
- OSC sender can be sync (send is fast, non-blocking UDP)
- Poll channels each frame, don't block GUI

### 11.3 Configuration Pattern

Separate configuration structs from runtime state:

```rust
// GOOD: Configuration struct (pure data, serializable)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub reps: u32,
    pub duration_secs: f32,
    pub rest_secs: f32,
    pub countdown_secs: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self { reps: 5, duration_secs: 3.0, rest_secs: 2.0, countdown_secs: 3.0 }
    }
}

// State enum (runtime only, not persisted)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    Idle,
    Countdown,
    Capturing,
    Resting,
    Complete,
}

// App holds both
pub struct App {
    training_config: TrainingConfig,  // User settings
    training_state: SessionState,     // Runtime state
}
```

### 11.4 Implementation Order

**Phase 1: Foundation** (proof of concept)
1. Data model (Vocabulary, Gesture, Example)
2. JSON persistence
3. DTW algorithm
4. OSC receiver (basic)
5. Minimal GUI (one gesture, start/stop)

**Phase 2: Core Features**
1. Multi-gesture support
2. Training session state machine
3. Audio cues
4. OSC sender
5. Recognition with debounce/cooldown

**Phase 3: Full GUI**
1. Mode switching (Training/Performance)
2. Vocabulary management (New/Open/Save)
3. Gesture editing (rename, delete)
4. Threshold sliders
5. Monitor panel
6. Hit log

**Phase 4: Polish**
1. Baseline recording & auto-calibration
2. Large fonts & visibility
3. Error handling
4. Cross-platform testing

---

## 12. Testing Strategy

### 12.1 Unit Tests

| Module | What to Test |
|--------|--------------|
| DTW | Identical sequences = 0, empty sequences = infinity, symmetry |
| Buffer | Circular behavior, recent_frames extraction |
| Recognizer | Hit detection logic, debounce, cooldown |
| Training | State transitions, timer behavior |
| Persistence | Round-trip (save → load → save = same) |

### 12.2 Integration Tests

- OSC receiver → recognizer → OSC sender pipeline
- Training session captures correct frames
- File operations with real filesystem

### 12.3 Manual Testing

- Test with actual skeleton tracking data
- Train a gesture, verify recognition
- Test threshold adjustment in real-time
- Test from dancer's viewing distance

---

## 13. Appendices

### 13.1 OSC Message Reference

**Input**:
```
Address: /wek/inputs
Arguments: f f f f... (68 floats for 34 joints × XY)
```

**Output**:
```
Address: /gesture/N (configurable per gesture)
Arguments: (none)
```

### 13.2 Default Values Reference

| Setting | Default | Range | Notes |
|---------|---------|-------|-------|
| Input port | 6448 | 1024-65535 | Wekinator-compatible |
| Input address | /wek/inputs | — | Wekinator-compatible |
| Output host | 127.0.0.1 | — | Use IP, not "localhost" |
| Output port | 12000 | 1024-65535 | — |
| Training reps | 5 | 1-20 | — |
| Training duration | 3.0s | 0.5-10.0 | — |
| Training rest | 2.0s | 0.5-10.0 | — |
| Training countdown | 3.0s | 1.0-10.0 | — |
| Baseline countdown | 3.0s | 1.0-10.0 | — |
| Baseline duration | 3.0s | 1.0-10.0 | — |
| Debounce (confirm_ms) | 80ms | 0-500 | 0 = instant |
| Cooldown (refractory_ms) | 500ms | 100-2000 | — |
| Threshold | 1500.0 | 100-10000 | Gesture-specific |
| Buffer size | 600 frames | — | ~10s at 60fps |
| Window size | 180 frames | — | ~3s |

### 13.3 File Locations

| Item | Path |
|------|------|
| Default vocabulary dir | `~/Documents/RALF/` |
| Vocabulary files | `*.ralf` |

### 13.4 Wekinator Compatibility

| Wekinator | RALF Gesture Studio |
|-----------|---------------------|
| Project | Vocabulary |
| DTW Output | Gesture |
| Training example | Example |
| /wek/inputs | Same |
| Port 6448 | Same |
| Port 12000 | Same |

---

*Document version: 2.0*
*Last updated: January 2026*
*Covers implementation through v0.1.0 with debounce support*
