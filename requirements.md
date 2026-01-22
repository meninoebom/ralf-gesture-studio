# RALF Gesture Studio: Final Requirements Document

## Project Overview

**RALF Gesture Studio** is a desktop application for training and recognizing movement gestures using Dynamic Time Warping (DTW). It receives skeleton tracking data via OSC, allows users to record gesture examples through a structured training workflow, and emits OSC "hit" signals when gestures are recognized during performance.

This application replaces Wekinator for the RALF (Responsive Audio Locomotion Framework) project, providing a focused, user-friendly tool specifically designed for choreomusical gesture recognition.

### Design Principles

1. **GUI-first**: No CLI required; all interaction through a visual interface
2. **Flow-state friendly**: Training UX designed for dancers who need to stay in their bodies
3. **Portable data**: Vocabularies are self-contained files that can be shared and archived
4. **Simple integration**: OSC output is dead simple to consume in Max4Live or similar tools
5. **Immediate feedback**: Clear visual and audio indicators for all system states

---

## Core Concepts

### Terminology

| Term | Definition |
|------|------------|
| **Vocabulary** | A collection of gestures that work together (e.g., "House Foundations", "Bomba Basics"). Saved as a single portable file. |
| **Gesture** | A single trained movement pattern (e.g., "jack", "wave", "spin"). Has a name, threshold, and output address. |
| **Example** | One recording of a gesture. Multiple examples per gesture improve recognition accuracy. |
| **Training Session** | A structured workflow for recording multiple examples: countdown → capture → rest → repeat. |
| **Hit** | When a gesture is recognized during performance, triggering an OSC output message. |

### Conceptual Hierarchy

```
Vocabulary ("House Foundations")
├── Gesture ("jack")
│   ├── Example 1 (recorded 10:32am, 2.8s, 171 frames)
│   ├── Example 2 (recorded 10:33am, 3.1s, 186 frames)
│   └── Example 3 (recorded 10:34am, 2.9s, 174 frames)
├── Gesture ("wave")
│   ├── Example 1
│   └── Example 2
└── Gesture ("drop")
    └── (no examples yet)
```

---

## Data Model

### Vocabulary

The root container for all gesture data. One vocabulary = one `.ralf` file.

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | File format version (e.g., "1.0") |
| `name` | string | User-editable vocabulary name |
| `created_at` | ISO 8601 timestamp | When vocabulary was created |
| `modified_at` | ISO 8601 timestamp | When vocabulary was last modified |
| `input.dimensions` | integer | Number of floats per input frame (e.g., 68 for 34 joints × XY) |
| `input.port` | integer | UDP port to listen on (default: 6448) |
| `input.address` | string | OSC address to listen for (default: "/wek/inputs") |
| `output.host` | string | Target host for hit messages (default: "localhost") |
| `output.port` | integer | Target UDP port for hit messages (default: 12000) |
| `gestures` | array | List of Gesture objects |

### Gesture

A named movement pattern with recognition settings.

| Field | Type | Description |
|-------|------|-------------|
| `id` | integer | Unique identifier within vocabulary (1, 2, 3...) |
| `name` | string | User-editable gesture name (e.g., "jack") |
| `osc_address` | string | OSC address for hit output (e.g., "/gesture/1") |
| `threshold` | float | DTW distance threshold for recognition (lower = stricter) |
| `created_at` | ISO 8601 timestamp | When gesture was created |
| `examples` | array | List of Example objects |

### Example

One recorded instance of a gesture.

| Field | Type | Description |
|-------|------|-------------|
| `recorded_at` | ISO 8601 timestamp | When example was recorded |
| `duration_ms` | integer | Duration in milliseconds |
| `frame_count` | integer | Number of frames captured |
| `frames` | array of arrays | Motion data: `[[f32; dimensions]; frame_count]` |

### File Format

**Extension**: `.ralf`

**Format**: JSON (human-readable, easy to debug, sufficient performance for this data size)

**Example file**:

```json
{
  "version": "1.0",
  "name": "House Foundations",
  "created_at": "2025-01-21T10:30:00Z",
  "modified_at": "2025-01-21T14:22:00Z",
  "input": {
    "dimensions": 68,
    "port": 6448,
    "address": "/wek/inputs"
  },
  "output": {
    "host": "localhost",
    "port": 12000
  },
  "gestures": [
    {
      "id": 1,
      "name": "jack",
      "osc_address": "/gesture/1",
      "threshold": 150.0,
      "created_at": "2025-01-21T10:31:00Z",
      "examples": [
        {
          "recorded_at": "2025-01-21T10:32:00Z",
          "duration_ms": 2850,
          "frame_count": 171,
          "frames": [
            [0.123, 0.456, 0.789, 0.012],
            [0.124, 0.458, 0.791, 0.014]
          ]
        }
      ]
    }
  ]
}
```

### Storage Location

**Default directory**: `~/Documents/RALF/` (macOS/Linux) or `%USERPROFILE%\Documents\RALF\` (Windows)

**Behavior**:
- On first launch, create default directory if it doesn't exist
- "New" creates file in default directory
- "Open" and "Save As" allow navigation to any location

### Auto-Save Behavior

The vocabulary file is automatically saved after every meaningful change:

| Action | Triggers Save |
|--------|---------------|
| Rename vocabulary | Yes |
| Add gesture | Yes |
| Rename gesture | Yes |
| Change gesture OSC address | Yes |
| Change gesture threshold | Yes |
| Complete training session (examples added) | Yes |
| Delete gesture | Yes |
| Change input/output port settings | Yes |

No "unsaved changes" warnings. The file always reflects current state.

---

## Application Modes

### Training Mode

For building vocabularies: creating gestures, recording examples, organizing.

**Available actions**:
- Create/open/save vocabularies
- Add/rename/delete gestures
- Configure gesture output addresses
- Run training sessions to record examples
- Configure input/output ports

### Performance Mode

For live recognition: running trained gestures, monitoring hits, tuning thresholds.

**Available actions**:
- Load vocabularies (read-only in terms of examples)
- Adjust thresholds in real-time
- Monitor DTW distances for all gestures
- View hit log
- Configure input/output ports

---

## GUI Specification

### Technology

**Framework**: egui (via eframe)
- Immediate mode GUI, well-suited for tool applications
- Single Rust codebase, no web dependencies
- Cross-platform (macOS primary, Linux/Windows secondary)

### Window Layout: Training Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RALF Gesture Studio                              [Training ▼|Performance]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ VOCABULARY ─────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Name: [ House Foundations________ ]           [New] [Open] [Save As] │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ CONNECTION ─────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  INPUT                               OUTPUT                           │  │
│  │  Port: [6448]  Addr: [/wek/inputs]   Host: [localhost] Port: [12000] │  │
│  │  ● RECEIVING (3ms ago)               ● READY                          │  │
│  │  ▁▃▅▇▅▃▁▃▅▇▅▃▁ (activity)                                            │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ GESTURES ───────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  ●  Name              Examples    Output Address         Actions      │  │
│  │  ─────────────────────────────────────────────────────────────────── │  │
│  │  ●  [jack________]    12          [/gesture/1____]       [Train] [×] │  │
│  │  ●  [wave________]    8           [/gesture/2____]       [Train] [×] │  │
│  │  ○  [drop________]    0           [/gesture/3____]       [Train] [×] │  │
│  │                                                                       │  │
│  │  [+ Add Gesture]                                                      │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ TRAIN ──────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Gesture: [wave ▼]   Reps: [5]   Duration: [3.0]s   Rest: [2.0]s     │  │
│  │                                                                       │  │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │  │
│  │  │                                                                 │ │  │
│  │  │                          START                                  │ │  │
│  │  │                       (spacebar)                                │ │  │
│  │  │                                                                 │ │  │
│  │  └─────────────────────────────────────────────────────────────────┘ │  │
│  │                                                                       │  │
│  │  Status: IDLE                                                         │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Window Layout: Performance Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RALF Gesture Studio                              [Training|Performance ▼]  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─ VOCABULARY ─────────────────────────────────────────────────────────┐  │
│  │  House Foundations                                     [Open]         │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ CONNECTION ─────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  INPUT                               OUTPUT                           │  │
│  │  Port: [6448]  Addr: [/wek/inputs]   Host: [localhost] Port: [12000] │  │
│  │  ● RECEIVING (3ms ago)               ● SENT (0.8s ago)                │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ GESTURE MONITOR ────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  Gesture      Threshold       Distance        Output        Status   │  │
│  │  ──────────────────────────────────────────────────────────────────  │  │
│  │  jack         [====●===] 150  ████░░░░ 89    /gesture/1      ●       │  │
│  │  wave         [===●====] 120  ██░░░░░░ 43    /gesture/2     ███      │  │
│  │  drop         [==●=====] 100  ██████░░ 167   /gesture/3      ●       │  │
│  │                                                                       │  │
│  │  ─────────────────────────────────────────────────────────────────── │  │
│  │                                                                       │  │
│  │                        ★ wave DETECTED ★                              │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  ┌─ HIT LOG ────────────────────────────────────────────────────────────┐  │
│  │                                                                       │  │
│  │  14:32:01.234   wave   distance: 43    → /gesture/2                  │  │
│  │  14:31:58.891   jack   distance: 72    → /gesture/1                  │  │
│  │  14:31:55.002   jack   distance: 68    → /gesture/1                  │  │
│  │                                                                       │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Indicator States

#### Input Status

| State | Visual | Meaning |
|-------|--------|---------|
| LISTENING | Yellow dot, "LISTENING" | Socket bound, waiting for data |
| RECEIVING | Green dot (pulsing), "RECEIVING (Nms ago)" | Data arriving, shows time since last packet |
| ERROR | Red dot, "ERROR: [message]" | Cannot bind socket (port in use, etc.) |

#### Output Status

| State | Visual | Meaning |
|-------|--------|---------|
| READY | Green dot, "READY" | Output configured, no recent sends |
| SENT | Green dot (flash), "SENT (Ns ago)" | Recently sent a hit, shows time since |
| ERROR | Red dot, "ERROR: [message]" | Cannot send (network error, etc.) |

#### Gesture Status (in list)

| State | Visual | Meaning |
|-------|--------|---------|
| Has examples | Filled circle (green) | Ready for recognition |
| No examples | Empty circle (gray) | Needs training |

#### Training Session States

| State | Visual | Audio | Meaning |
|-------|--------|-------|---------|
| IDLE | Gray panel, "IDLE" | — | Ready to start |
| COUNTDOWN | Yellow panel, large countdown "3... 2... 1..." | Tick sounds | Get ready |
| CAPTURING | Green panel, "CAPTURING", progress bar | BEEP (start) | Recording motion data |
| RESTING | Yellow panel, "REST", countdown to next | BEEP (end), soft tone | Pause between reps |
| COMPLETE | Green flash, "Complete! N examples recorded" | Double ding | All reps done |

### Training Session Panel During Capture

When actively training, the panel transforms:

```
┌─ TRAIN ──────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ████████████████████████████████████████████████████████████████████    │
│  █                                                                  █    │
│  █                        CAPTURING                                 █    │
│  █                                                                  █    │
│  █                          2.1s                                    █    │
│  █                                                                  █    │
│  █                     [███████████░░░░░░░░░]                       █    │
│  █                                                                  █    │
│  ████████████████████████████████████████████████████████████████████    │
│                                                                          │
│  Recording example 3 of 5 for "wave"                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Space | Start training session (when in Training mode, IDLE state) |
| Escape | Cancel training session (during countdown/capture/rest) |
| Cmd/Ctrl + N | New vocabulary |
| Cmd/Ctrl + O | Open vocabulary |
| Cmd/Ctrl + S | Save As (explicit save to new location) |

---

## Training Session Specification

### Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| Gesture | selection | (first gesture) | — | Which gesture to train |
| Repetitions | integer | 5 | 1-10 | Number of examples to record |
| Duration | float | 3.0 | 1.0-10.0 | Seconds per capture |
| Rest | float | 2.0 | 1.0-10.0 | Seconds between captures |

### State Machine

```
                         START triggered
                         (spacebar/button)
                               │
                               ▼
┌──────────┐           ┌──────────────┐
│   IDLE   │◄──────────│   COMPLETE   │
└──────────┘           └──────────────┘
                               ▲
                               │ all reps done
                               │
┌──────────────┐       ┌──────────────┐
│  COUNTDOWN   │──────▶│  CAPTURING   │
│  (3 seconds) │       │  (N seconds) │
└──────────────┘       └──────┬───────┘
       ▲                      │
       │                      ▼
       │               ┌──────────────┐
       │               │   RESTING    │
       └───────────────│  (N seconds) │
         more reps     └──────────────┘
```

### Behavior Details

1. **IDLE → COUNTDOWN**: User triggers start. Begin 3-second countdown with tick sounds.

2. **COUNTDOWN → CAPTURING**: Countdown ends. Play start beep. Begin buffering incoming OSC data.

3. **CAPTURING → RESTING** (or COMPLETE): Duration ends. Play end beep. Save buffered frames as new Example. If more reps remaining, enter RESTING. Otherwise, COMPLETE.

4. **RESTING → COUNTDOWN**: Rest duration ends. Play soft tone. Return to COUNTDOWN for next rep.

5. **COMPLETE → IDLE**: Display success message ("5 examples recorded!"). Save vocabulary. Return to IDLE.

6. **CANCEL (any state)**: User presses Escape. Discard any in-progress capture. Return to IDLE. Do not save partial data.

### Audio Feedback

| Event | Sound | Frequency/Character |
|-------|-------|---------------------|
| Countdown tick | Short beep | Low pitch (300 Hz), 100ms |
| Capture start | Long beep | High pitch (800 Hz), 300ms |
| Capture end | Long beep | Different high pitch (600 Hz), 300ms |
| Rest period | Soft tone | Medium pitch (500 Hz), gentle, 150ms |
| Session complete | Double ding | High pitch (1000 Hz), two 150ms tones |

**Implementation**: Generate simple sine wave tones using `rodio` crate. No external audio files needed.

---

## OSC Communication

### Input (Receiving Skeleton Data)

| Setting | Default | Description |
|---------|---------|-------------|
| Port | 6448 | UDP port to listen on |
| Address | /wek/inputs | OSC address to accept |
| Dimensions | (configured) | Expected number of floats per message |

**Behavior**:
- Bind UDP socket on startup
- Accept OSC messages matching configured address
- Parse float array from message
- Validate array length matches expected dimensions
- Feed data to sliding window buffer (performance mode) or training capture buffer (training mode)

**Wekinator compatibility**: Uses same defaults as Wekinator for drop-in replacement.

### Output (Sending Hits)

| Setting | Default | Description |
|---------|---------|-------------|
| Host | localhost | Target hostname/IP |
| Port | 12000 | Target UDP port |

**Hit message format**:

```
Address: /gesture/1  (or whatever is configured per gesture)
Arguments: (none)
```

Simple, clean, easy to receive in Max4Live with a single `[udpreceive]` → `[route /gesture/1 /gesture/2 ...]`.

**Optional enhancement** (future): Include distance as float argument for threshold visualization in Max.

### Port Configuration

- Input and output ports are editable in the GUI
- Changes take effect immediately (rebind socket)
- Saved per-vocabulary

---

## Recognition Engine

### Algorithm: Dynamic Time Warping (DTW)

**Variant**: FastDTW or Sakoe-Chiba constrained DTW

**Distance metric**: Euclidean distance between frame vectors

**Multi-dimensional**: Each frame is a vector of N floats (e.g., 68 for 34 joints × XY). DTW computes distance across the full vector at each time step.

### Continuous Matching

In performance mode:

1. Maintain a **sliding window buffer** of recent input frames
2. On each new input frame:
   - Add frame to buffer
   - For each gesture with examples:
     - Compute DTW distance between buffer and each example
     - Take minimum distance across all examples (nearest neighbor)
   - If minimum distance < gesture threshold → **HIT**
3. **Refractory period**: After a hit, ignore that gesture for N frames (default: 30, ~0.5s at 60fps) to prevent rapid re-triggering

### Threshold Behavior

- Each gesture has its own threshold (float, typically 50-500 depending on gesture complexity)
- Lower threshold = stricter matching (fewer false positives, may miss variations)
- Higher threshold = looser matching (catches variations, may have false positives)
- Threshold is adjustable in real-time during performance mode

### Buffer Management

- Buffer size: Dynamically sized based on longest example in vocabulary
- Minimum frames: Don't attempt matching until buffer has at least as many frames as shortest example
- Circular buffer: Old frames drop off as new ones arrive

---

## Technical Architecture

### Crate Dependencies

| Crate | Purpose |
|-------|---------|
| `eframe` / `egui` | GUI framework |
| `rosc` | OSC message encoding/decoding |
| `tokio` | Async runtime for concurrent OSC send/receive |
| `rodio` | Audio playback for training beeps |
| `serde` + `serde_json` | Vocabulary serialization |
| `dtw` or `augurs-dtw` | DTW algorithm (or custom implementation) |
| `directories` | Cross-platform default paths |
| `tracing` | Structured logging |

### Module Structure

```
src/
├── main.rs                     # Entry point, launch GUI
│
├── model/                      # Core data structures
│   ├── mod.rs
│   ├── vocabulary.rs           # Vocabulary, Gesture, Example structs
│   ├── persistence.rs          # Load/save JSON files
│   └── validation.rs           # Data validation
│
├── engine/                     # Recognition engine
│   ├── mod.rs
│   ├── dtw.rs                  # DTW algorithm
│   ├── matcher.rs              # Continuous matching logic
│   └── buffer.rs               # Sliding window buffer
│
├── session/                    # Training session
│   ├── mod.rs
│   ├── state_machine.rs        # Session states and transitions
│   ├── timer.rs                # Async countdown/duration timers
│   └── audio.rs                # Beep generation
│
├── osc/                        # OSC communication
│   ├── mod.rs
│   ├── receiver.rs             # Input listener
│   └── sender.rs               # Output sender
│
├── app/                        # Application state
│   ├── mod.rs
│   ├── state.rs                # Global app state
│   └── actions.rs              # User actions (add gesture, etc.)
│
└── gui/                        # GUI components
    ├── mod.rs
    ├── app_window.rs           # Main window, mode switching
    ├── training_view.rs        # Training mode layout
    ├── performance_view.rs     # Performance mode layout
    ├── vocabulary_panel.rs     # Vocabulary name, file operations
    ├── connection_panel.rs     # Input/output status and config
    ├── gesture_list.rs         # Gesture table with inline editing
    ├── train_panel.rs          # Training session controls
    ├── monitor_panel.rs        # Live gesture distances
    └── hit_log.rs              # Hit history list
```

### Threading Model

```
┌─────────────────────────────────────────────────────────────┐
│                      Main Thread                            │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐   │
│   │                    egui GUI                         │   │
│   │   - Render UI                                       │   │
│   │   - Handle user input                               │   │
│   │   - Update app state                                │   │
│   └─────────────────────────────────────────────────────┘   │
│                            │                                │
│                            │ channels                       │
│                            ▼                                │
│   ┌─────────────────────────────────────────────────────┐   │
│   │              Async Runtime (tokio)                  │   │
│   │                                                     │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌───────────┐   │   │
│   │   │OSC Receiver │  │OSC Sender   │  │  Timers   │   │   │
│   │   │(task)       │  │(task)       │  │  (task)   │   │   │
│   │   └─────────────┘  └─────────────┘  └───────────┘   │   │
│   │                                                     │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

Communication via channels:
- OSC Receiver → App State: incoming frames
- App State → OSC Sender: hit events
- Timers → App State: session state transitions

---

## Implementation Phases

### Phase 1: Foundation (MVP)

**Goal**: Can record one gesture and recognize it.

- [ ] Project setup (Cargo, dependencies)
- [ ] Data model structs (Vocabulary, Gesture, Example)
- [ ] JSON persistence (save/load)
- [ ] OSC receiver (listen for input)
- [ ] Basic DTW implementation
- [ ] OSC sender (emit hits)
- [ ] Minimal GUI (one hardcoded gesture, start/stop recording, run mode)

**Deliverable**: Working prototype that proves the core loop.

### Phase 2: Training UX

**Goal**: Full training session workflow.

- [ ] Training session state machine
- [ ] Countdown/capture/rest timers
- [ ] Audio feedback (beeps)
- [ ] GUI: training panel with session controls
- [ ] GUI: gesture list with add/delete
- [ ] Multiple gestures in vocabulary

**Deliverable**: Can train multiple gestures with the structured session workflow.

### Phase 3: Full GUI

**Goal**: Complete, polished interface.

- [ ] GUI: vocabulary panel (new/open/save as)
- [ ] GUI: connection panel with status indicators
- [ ] GUI: inline editing (gesture names, output addresses)
- [ ] GUI: mode switching (training/performance)
- [ ] GUI: performance view with live distances
- [ ] GUI: threshold sliders
- [ ] GUI: hit log

**Deliverable**: Feature-complete application.

### Phase 4: Polish

**Goal**: Production-ready.

- [ ] Error handling and user feedback
- [ ] Performance optimization (DTW caching, etc.)
- [ ] Cross-platform testing (macOS, Linux, Windows)
- [ ] App icon and packaging
- [ ] Documentation

**Deliverable**: Releasable v1.0.

---

## Future Enhancements (Post-v1.0)

These are explicitly out of scope for v1.0 but documented for future consideration:

1. **MIDI input**: Trigger training session via foot pedal or MIDI controller
2. **Example management**: View, delete, or re-record individual examples
3. **Threshold auto-calibration**: Suggest threshold based on example spread
4. **Gesture visualization**: Simple stick figure playback of recorded examples
5. **Export/import gestures**: Share individual gestures between vocabularies
6. **Multiple vocabulary comparison**: A/B test different training approaches
7. **OSC output with distance**: Include DTW distance in hit messages
8. **Wekinator project import**: Convert .wek files to .ralf format

---

## Appendix A: Wekinator Compatibility

For users migrating from Wekinator, this table shows equivalent concepts:

| Wekinator | RALF Gesture Studio |
|-----------|---------------------|
| Project | Vocabulary |
| DTW Output | Gesture |
| Training example | Example |
| /wek/inputs (default) | Same default |
| Port 6448 (default) | Same default |
| Port 12000 (default output) | Same default |
| /output/N | /gesture/N (configurable) |

The default OSC addresses and ports match Wekinator, allowing the same skeleton tracking inputs to work without reconfiguration.

---

## Appendix B: OSC Message Reference

### Messages Received (Input)

| Address | Arguments | Description |
|---------|-----------|-------------|
| `/wek/inputs` (default) | float[] | Skeleton joint positions |

### Messages Sent (Output)

| Address | Arguments | Description |
|---------|-----------|-------------|
| `/gesture/N` (configurable) | (none) | Gesture N recognized |

---

## Appendix C: File Locations

| Item | Location |
|------|----------|
| Default vocabulary directory | `~/Documents/RALF/` |
| Vocabulary files | `*.ralf` |
| Application logs | `~/.ralf/logs/` (if implemented) |
| Application config | `~/.ralf/config.json` (if needed for app-level prefs) |

---

*Document version: 1.0*
*Last updated: January 21, 2025*
