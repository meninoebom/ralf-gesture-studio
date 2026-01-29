# RALF Gesture Studio

A desktop application for training and recognizing movement gestures using Dynamic Time Warping (DTW). Built for dancers and choreographers working with the RALF (Responsive Audio Locomotion Framework) system.

## Quick Reference

- **Language**: Rust (edition 2021)
- **GUI Framework**: egui via eframe
- **Async Runtime**: tokio
- **Data Format**: JSON (.ralf files)
- **Requirements**: See `requirements.md` for full specification

## Project Context

This application replaces Wekinator for gesture recognition in choreomusical performance. It receives skeleton tracking data via OSC, allows structured training of gesture examples, and emits OSC "hit" signals when gestures are recognized.

**Target users**: Dancers who need to stay in flow state while training gestures. The UX prioritizes minimal cognitive load during training sessions.

## Design Principles

1. **GUI-first**: No CLI required; all interaction through visual interface
2. **Flow-state friendly**: Training UX designed for dancers who need to stay in their bodies
3. **Portable data**: Vocabularies are self-contained .ralf files
4. **Simple integration**: OSC output is dead simple to consume in Max4Live
5. **Immediate feedback**: Clear visual and audio indicators for all system states

## Architecture Overview

```
src/
├── main.rs                     # Entry point, integration tests
├── model/
│   ├── mod.rs                  # Module exports
│   ├── vocabulary.rs           # Vocabulary, Gesture, Example structs
│   └── persistence.rs          # JSON file save/load
├── engine/
│   ├── mod.rs                  # Module exports
│   ├── dtw.rs                  # Dynamic Time Warping algorithm
│   ├── buffer.rs               # Frame buffer and recording session
│   ├── recognizer.rs           # Real-time gesture recognition
│   └── training.rs             # Training session state machine with audio
├── osc/
│   ├── mod.rs                  # Module exports
│   ├── receiver.rs             # Async OSC receiver
│   └── sender.rs               # OSC sender for hit messages
└── gui/
    └── mod.rs                  # egui GUI (Training/Performance modes)
```

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `eframe` / `egui` | Immediate mode GUI |
| `rosc` | OSC message encoding/decoding |
| `tokio` | Async runtime for OSC + timers |
| `rodio` | Audio feedback (training beeps) |
| `rfd` | Native file dialogs (Open, Save As) |
| `serde` + `serde_json` | Vocabulary serialization |
| `uuid` | Unique vocabulary identifiers |
| `directories` | Cross-platform default paths |
| `crossbeam-channel` | Thread-safe communication |

## Data Model

**Hierarchy**: Vocabulary → Gesture → Example

- **Vocabulary**: Collection of gestures, stored as single .ralf file
- **Gesture**: Named movement pattern with threshold and OSC output address
- **Example**: One recorded instance (timestamped frames of skeleton data)

**File location**: `~/Documents/RALF/` by default
**Format specification**: See `FORMAT.md` for complete field reference

### Research-Ready Metadata (v1.1)

Vocabularies include metadata for research portability and future computational musicology work:

```rust
pub struct Vocabulary {
    pub version: String,           // "1.1" (SchemaVer format)
    pub uuid: Uuid,                // Unique identifier across systems
    pub name: String,
    // ... timestamps, input/output config ...

    // Research metadata
    pub tracking_system: String,   // "mediapipe-pose-33-xy"
    pub coordinate_system: String, // "normalized-0-1-xy"
    pub source_fps: Option<f32>,   // 60.0
    pub license: Option<String>,   // "CC-BY-4.0"
    pub creator: Option<String>,   // Attribution
    pub tags: Vec<String>,         // ["house", "dance", "foundations"]
    pub extensions: HashMap<String, Value>, // Future extensibility

    pub gestures: Vec<Gesture>,
}
```

**FAIR Principles**: UUID enables unique identification, tracking_system documents data compatibility, license clarifies usage rights.

**Migration**: v1.0 files are automatically upgraded when loaded (UUID generated, defaults applied).

## Recognition Algorithm (VAD-Style State Machine)

The recognizer uses a DTW approach combined with a VAD (Voice Activity Detection) style state machine, borrowing patterns from speech recognition systems like CMU Sphinx, Kaldi, and WebRTC VAD.

**References**:
- Wekinator: `fiebrink1/wekinator` - `src/wekimini/learning/dtw/DtwModel.java`
- GRT: `nickgillian/grt` - `GRT/ClassificationModules/DTW/DTW.cpp`
- CMU Sphinx VAD, WebRTC VAD

### Breakthrough (2026-01-29)

**VAD-style state machine with time-based hangover** - After multiple complex approaches failed (peak detection, hysteresis, distance-based re-arming), this simple state machine works reliably:

```
IDLE → BUILDING → PEAK (fire!) → RECOVERY → IDLE
```

Key success factors:
- **Frame accumulation**: 3 consecutive frames below threshold (~200ms confirmation)
- **Time-based hangover**: 300ms recovery blocks new detections
- **No distance-based exit from recovery**: Critical for body tracking where resting distance is close to threshold

### How It Works

1. **Fixed Window**: Window size = first training example's length
2. **Compare Against All Examples**: For each gesture, compare against every training example
3. **State Machine**: Not simple threshold crossing - uses VAD-style states
4. **Frame Accumulation**: Require 3 frames below threshold before firing
5. **Time-Based Recovery**: 300ms hangover prevents echo, exits regardless of distance
6. **Performance**: Skip frames + downsample = ~64x faster than naive implementation

### State Machine Flow

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

### ⚠️ Critical Learning: Time-Based Recovery Only

**Recovery MUST exit based on time alone, NOT distance.**

Why distance-based recovery fails:
- With body tracking, "resting" distance is often still close to threshold
- Example from real use: threshold=17, resting distance=21-24
- If exit required distance > 25, user barely exceeds it
- Bug caught: exit at 1.5× threshold caused stuck recognition (one hit, never re-arms)

```rust
// CORRECT - time-based only
RecognitionState::Recovery => {
    if hangover_time_elapsed >= 300ms {
        self.reset_to_idle();  // Ready for next gesture
    }
    false  // Never fire in Recovery
}

// WRONG - fails with body tracking
if distance > threshold * 1.5 {  // User might never exceed this!
    self.reset_to_idle();
}
```

### Configuration (v0.5.0 Production)

```rust
RecognitionConfig {
    cooldown_ms: 500,              // Backup protection (rarely used)
    threshold_high_factor: 1.0,    // Entry at 100% of threshold
    frames_to_fire: 3,             // ~200ms confirmation at 15Hz DTW
    hangover_ms: 300,              // 300ms recovery before re-arming
}
```

### Real-World Results (2026-01-29)

Testing with "wings" gesture (lifting both arms):
- **7 HITs, 0 false positives, 0 echo**
- Threshold: 17 (AUTO from μ+σ)
- Resting distance: ~21-24
- Gesture distance: ~14-15 when performing

### Key Learnings (2026-01-29)

1. **VAD patterns from speech recognition work** - frame accumulation + hangover
2. **Time-based recovery is essential** - distance-based fails with body tracking
3. **Simplification helps** - removed motion gate, adaptive threshold, peak detection
4. **Frame accumulation prevents noise** - 3 frames = ~200ms confirmation
5. **Hangover prevents echo** - 300ms blocks all new detections after fire

### A/B Test: Best Template vs All Examples (2026-01-29)

Tested GRT-style "best template" (single most representative example) vs Wekinator-style "all examples" (minimum distance across all training examples):

| Metric | Best Template | All Examples | Winner |
|--------|---------------|--------------|--------|
| Gestures Detected | 13 | 17 | **All Examples (+30%)** |
| Echoes | 5 | 5 | Tie |
| Detection Rate | 6.8/min | 9.5/min | **All Examples** |

**Decision**: Default to "All Examples" comparison mode.

**Why All Examples wins for body tracking**:
- Takes minimum distance across ALL training examples
- More forgiving of gesture variations
- User's natural performance varies more than GRT's audio gesture assumptions

**Why Best Template failed**:
- Single template can't represent gesture variability
- May pick a template closer to resting pose
- Results in narrower "hit zone" that misses valid gestures

**Note**: Toggle available in Performance mode UI for future A/B testing.

#### Diagnostic Logging

Enable via UI button to write detailed logs:
```
# Format: timestamp,event_type,data...
1234,REC,frame,buffer,window,gesture:dist:thresh:armed,...
1234,HIT,frame,gesture,distance,threshold,margin%
1234,NEAR,frame,gesture,distance,threshold,margin%,reason
```

Reasons for NEAR misses: `in_cooldown`, `above_threshold`

### Statistical Threshold (μ+σ Approach)

**v0.3.0 Feature**: Automatic threshold calibration using the GRT-style statistical approach.

Instead of manually tuning thresholds per gesture, the system computes statistics from training examples:
1. After training, compute DTW distances between all pairs of examples
2. Calculate mean (μ) and standard deviation (σ) of these distances
3. Set threshold = μ + σ × coefficient (default coefficient = 2.0)

**Key Benefits**:
- **No manual tuning**: One global coefficient works for all gestures
- **Adapts to complexity**: Simple gestures get tight thresholds; complex gestures get looser ones
- **Automatic recalibration**: Threshold updates when examples are added

**UI Features**:
- **AUTO/MAN indicator**: Shows whether using computed or manual threshold
- **μ±σ display**: Shows mean and std when in AUTO mode
- **Click to toggle**: Switch between auto and manual modes

**Data Model** (in `Gesture` struct):
```rust
distance_mean: Option<f32>,        // Computed after training
distance_std: Option<f32>,         // Computed after training
threshold_manual_override: bool,   // If true, use manual threshold
threshold_coefficient: f32,        // Default 2.0 (μ + σ×coeff)
```

**Reference**: Gesture Recognition Toolkit (GRT) by Nick Gillian

See `.llm/active-plan.md` for detailed algorithm documentation and Wekinator source references.

## Implementation Status

**v0.5.0 COMPLETE** - VAD-style state machine (2026-01-29):

| Feature | Status | Description |
|---------|--------|-------------|
| State Machine | ✅ | Idle → Building → Peak → Recovery → Idle |
| Frame Accumulation | ✅ | 3 frames below threshold to fire (~200ms) |
| Time-Based Hangover | ✅ | 300ms recovery blocks new detections |
| Simplification | ✅ | Removed motion gate, adaptive threshold, peak detection |

**v0.3.0 COMPLETE** - Statistical threshold (μ+σ approach):

| Feature | Status | Description |
|---------|--------|-------------|
| Statistical Computation | ✅ | Compute μ and σ from training examples |
| Auto Threshold | ✅ | threshold = μ + σ × coefficient |
| UI Integration | ✅ | AUTO/MAN toggle, μ±σ display |
| Persistence | ✅ | Statistics saved/loaded with vocabulary |

**v0.2.0 COMPLETE** - Wekinator-style recognition:

| Feature | Status | Description |
|---------|--------|-------------|
| Simple Threshold | ✅ | Fire when distance < threshold (not edge detection) |
| Multiple Candidates | ✅ | Try different window sizes based on example lengths |
| All Examples | ✅ | Compare against all training examples, not prototype |
| Cooldown | ✅ | Prevent repeated firing (configurable) |

**v0.1.0 COMPLETE** - All 8 milestones implemented:

1. ✅ Data Model - Vocabulary/Gesture/Example structs, JSON persistence
2. ✅ GUI Shell - eframe/egui window with panel layout
3. ✅ OSC Receiver - Async UDP receiver with status tracking
4. ✅ OSC Sender - Hit message output with test button
5. ✅ DTW Algorithm - Dynamic Time Warping for gesture matching
6. ✅ Recording + Matching - Real-time recognition with refractory period
7. ✅ Training Session - State machine with audio cues (rodio)
8. ✅ Polish + Performance Mode - File dialogs, threshold sliders, auto-save

See `.llm/active-plan.md` for detailed milestone documentation.

## Coding Guidelines

### Rust Conventions

- Use `Result<T, E>` for fallible operations; avoid panics in library code
- Prefer `thiserror` for custom error types
- Use `tracing` for structured logging, not `println!`
- Keep modules focused; one concept per file

### egui Patterns

- Immediate mode: UI is rebuilt every frame
- State lives in the app struct, not in UI components
- Use `egui::Context::request_repaint()` when background tasks update state
- Keep UI code in `gui/` module, separate from business logic

### OSC Communication

- Default input port: 6448 (Wekinator compatible)
- Default output port: 12000
- Input address: `/wek/inputs`
- Output address: `/gesture/N` (configurable per gesture)

### Threading Model

- Main thread: egui GUI
- Tokio runtime: OSC receiver, OSC sender, training session timers
- Communication via channels (crossbeam or tokio::sync)

## Testing Strategy

- Unit tests for DTW algorithm and data model
- Integration tests for OSC round-trip
- Manual testing for GUI and training workflow
- Test with real skeleton data from MoveNet pipeline

## Commands

```bash
# Build
cargo build

# Run
cargo run

# Run with release optimizations
cargo run --release

# Run tests
cargo test

# Check formatting
cargo fmt --check

# Lint
cargo clippy
```

## OSC Testing

For development without a real skeleton tracker:

```bash
# Send test OSC messages (requires oscsend or similar)
oscsend localhost 6448 /wek/inputs f f f f ...
```

## Related Files

- `requirements.md` - Full specification document
- `FORMAT.md` - Complete .ralf file format specification
- `ralf-graphviz.dot` - System architecture diagram
- `RALF in context.png` - Context diagram showing RALF in the larger system

## Out of Scope (v1.0)

- MIDI input for training triggers
- Individual example management (view/delete)
- Threshold auto-calibration (see `.claude/commands/dtw-gesture-recognition.md` for μ+σ approach when ready to implement)
- Gesture visualization/playback
- Wekinator project import
