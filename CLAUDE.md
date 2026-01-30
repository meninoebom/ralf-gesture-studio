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

## Recognition Algorithm (VAD-Style State Machine + Three-Layer Echo Defense)

The recognizer uses a DTW approach combined with a VAD (Voice Activity Detection) style state machine, borrowing patterns from speech recognition systems like CMU Sphinx, Kaldi, and WebRTC VAD. Echo prevention uses a three-layer defense: Schmitt trigger hysteresis, safety valve timeout, and global non-maximum suppression.

**References**:
- Wekinator: `fiebrink1/wekinator` - `src/wekimini/learning/dtw/DtwModel.java`
- GRT: `nickgillian/grt` - `GRT/ClassificationModules/DTW/DTW.cpp`
- CMU Sphinx VAD, WebRTC VAD
- Sakoe-Chiba (1978), UCR Suite DTW optimization

### Breakthrough (2026-01-29)

**VAD-style state machine with three-layer echo defense** - After multiple complex approaches failed (peak detection, hysteresis, distance-based re-arming), this state machine with layered defenses works reliably:

```
IDLE → BUILDING → PEAK (fire!) → RECOVERY → IDLE
```

Key success factors:
- **Frame accumulation**: 3 consecutive frames below threshold (~200ms confirmation)
- **Distance slope check**: Only enter Building when distance is falling (not flat/rising)
- **Schmitt trigger recovery**: Track min_distance_in_recovery, re-arm when consistently above threshold×1.1
- **Safety valve timeout**: Force re-arm after 5000ms regardless of distance
- **Global cooldown (NMS)**: Block ALL gestures for 1500ms after ANY gesture fires

### How It Works

1. **Fixed Window**: Window size = first training example's length
2. **Compare Against All Examples**: For each gesture, compare against every training example
3. **State Machine**: VAD-style states with three-layer echo defense
4. **Frame Accumulation**: Require 3 frames below threshold before firing
5. **Distance Slope Check**: Only enter Building when distance is falling
6. **Schmitt Trigger Recovery**: Track min_distance; re-arm when consistently cleared
7. **Safety Valve**: Force re-arm after 5s (handles "always-on" gestures)
8. **Global Cooldown (NMS)**: Block all gestures for 1.5s after any hit
9. **DTW Optimizations**: Sakoe-Chiba band (15%), early abandoning, LB_Keogh pruning

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

### ⚠️ Critical Learning: Three-Layer Echo Defense

**A single mechanism cannot prevent all echo types.** Three layers are needed:

| Layer | Mechanism | What It Prevents |
|-------|-----------|-----------------|
| Schmitt Trigger | Track `min_distance` in Recovery, re-arm when min > threshold×1.1 | Same-gesture noise spikes faking clearance |
| Safety Valve | Force re-arm after 5000ms | Stuck recovery (resting distance < threshold) |
| Global Cooldown | Block ALL gestures for 1500ms after ANY hit | Cross-gesture round-robin echoes |

**Why distance-based recovery alone fails:**
- With body tracking, resting distance is often permanently below threshold
- Example: jump threshold=53, resting distance=15 — distance NEVER clears
- Safety valve handles this; Schmitt trigger handles cases where distance does clear

**Real-world finding:** In testing, `hysteresis_cleared` never triggers — all Recovery→Idle transitions use `safety_valve_timeout`. The global cooldown is the primary echo prevention mechanism.

### Configuration (v0.6.0 Production)

```rust
RecognitionConfig {
    cooldown_ms: 500,              // Per-gesture minimum between hits
    threshold_high_factor: 1.0,    // Entry at 100% of threshold
    frames_to_fire: 3,             // ~200ms confirmation at 15Hz DTW
    hangover_ms: 300,              // Minimum recovery time
    // Schmitt trigger hysteresis
    rearm_safety_factor: 1.1,      // Re-arm when min_distance > threshold × 1.1
    max_recovery_ms: 5000,         // Safety valve: force re-arm after 5s
    // Global non-maximum suppression
    global_cooldown_ms: 1500,      // Block ALL gestures after ANY hit
    // DTW optimization
    sakoe_chiba_band: 0.15,        // 15% warping constraint
}
```

### Real-World Results (2026-01-29, v0.6.0)

Testing with 3 gestures (wave, jump, spin), 6 examples each:
- **43 HITs, 0 echoes (0.0% echo rate)** — down from 63.3%
- All Recovery→Idle via safety valve timeout
- Cross-gesture minimum gap: 2029ms (global cooldown working)
- 45 Building entries, 44 Peak fires, 0 aborted (98% conversion)

### Key Learnings (2026-01-29)

1. **VAD patterns from speech recognition work** - frame accumulation + hangover
2. **Layer echo defenses** - Schmitt trigger + safety valve + global cooldown
3. **Timer-based > distance-based** for body tracking where resting distance < threshold
4. **Global cooldown (NMS) is the primary mechanism** - prevents cross-gesture round-robin
5. **6+ training examples** - stabilize μ+σ thresholds (n=4 is statistically fragile)
6. **Sakoe-Chiba band is "faster AND better"** - prevents pathological DTW warping
7. **Do NOT spatially resample** - destroys velocity info critical for dance

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

**v0.6.0 COMPLETE** - Three-layer echo defense + DTW optimizations (2026-01-29):

| Feature | Status | Description |
|---------|--------|-------------|
| Schmitt Trigger Recovery | ✅ | Track min_distance, re-arm when cleared above threshold×1.1 |
| Safety Valve Timeout | ✅ | Force re-arm after 5000ms |
| Global Cooldown (NMS) | ✅ | Block all gestures for 1500ms after any hit |
| Distance Slope Check | ✅ | Only enter Building when distance is falling |
| Sakoe-Chiba Band | ✅ | 15% warping constraint on DTW |
| Early Abandoning | ✅ | Stop DTW mid-calculation if row min > best_so_far |
| LB_Keogh Pruning | ✅ | O(n) lower bound prunes before DTW starts |
| Echo Rate | ✅ | 0% (down from 63.3%) |

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
