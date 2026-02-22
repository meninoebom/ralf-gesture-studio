# RALF Gesture Studio

A desktop application for training and recognizing movement gestures using Dynamic Time Warping (DTW). Built for dancers and choreographers working with the RALF (Responsive Audio Locomotion Framework) system.

## Quick Reference

- **Language**: Rust (edition 2021)
- **GUI Framework**: Tauri 2.0
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
├── main.rs                     # Entry point, Tauri setup
├── model/
│   ├── mod.rs                  # Module exports
│   ├── vocabulary.rs           # Vocabulary, Gesture, Example structs
│   └── persistence.rs          # JSON file save/load
├── engine/
│   ├── mod.rs                  # Module exports
│   ├── dtw.rs                  # Dynamic Time Warping algorithm
│   ├── recognizer.rs           # Real-time gesture recognition (VAD-style state machine)
│   ├── training.rs             # Training session state machine with audio
│   ├── preprocess.rs           # Frame normalization pipeline
│   ├── statistics.rs           # Statistical threshold calibration (mu+sigma)
│   └── diagnostics.rs          # Diagnostic logging for session analysis
├── osc/
│   ├── mod.rs                  # Module exports
│   ├── receiver.rs             # Async OSC receiver
│   └── sender.rs               # OSC sender for hit messages
└── gui/
    └── mod.rs                  # Tauri GUI (Training/Performance modes)

tools/
└── gesture-viewer.html         # Standalone .ralf file visualizer (MediaPipe skeleton)
```

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `tauri` | Desktop application framework |
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

## Recognition Config (v0.8.0)

```rust
RecognitionConfig {
    cooldown_ms: 500,              // Per-gesture minimum between hits
    threshold_high_factor: 1.0,    // Entry at 100% of threshold
    frames_to_fire: 3,             // ~200ms confirmation at 15Hz DTW
    max_recovery_ms: 5000,         // Safety valve: force re-arm after 5s
    global_cooldown_ms: 1500,      // Block ALL gestures after ANY hit
    sakoe_chiba_band: 0.15,        // 15% warping constraint
    use_subsequence_dtw: true,     // sDTW with wavefront banding (93.3% benchmark)
}
```

For algorithm details, echo defense design, preprocessing pipeline, threshold calibration, and diagnostic log analysis, see the `dtw-gesture-recognition` skill.

## Operational Learnings

- **Warm-up effect in first 3-5 seconds.** The sliding window needs real movement data before distances are meaningful. First 3 hits in a session average ~34% margin vs ~68% for the rest. The very first hit may barely clear threshold (6.6% margin observed). This is normal — not a bug. Account for it when analyzing diagnostic logs.
- **Detection latency ~400ms.** Mean time from Building entry to Peak fire is ~400ms (3 frames of accumulation at ~15Hz DTW rate). This is the baseline responsiveness of the system.
- **Jump-style gestures produce tight margins.** Gestures with low inter-example variance (e.g., jumps) get tight auto-thresholds (low σ), resulting in lower margins (~39%) compared to complex gestures like spins (~75%). This is correct behavior — tight thresholds mean precise detection, not fragile detection.
- **NEVER compute DTW in a Tauri polling command.** `get_state()` is called ~60Hz by the frontend. Any pairwise DTW computation here (even banded) will hang the app once gestures have 8+ examples. Cache expensive results (consistency, thresholds) when data changes, read the cached value in the poll. This caused repeated beach-ball-of-death bugs.
- **Every DTW call in production must be banded.** Use `dtw_distance_with_abandon` (Sakoe-Chiba banded + early abandoning), never bare `dtw_distance` (unbanded O(N×M) with full matrix allocation). The unbanded version exists only for unit tests. Grep for `dtw_distance(` (without `_with_abandon` or `_constrained`) to audit.
- **Threshold calibration must match the distance metric.** If recognition uses sDTW, thresholds must be computed with `compute_threshold_stats_sdtw`. Mixing standard DTW thresholds with sDTW recognition gives worse results than standard DTW alone (76.7% vs 83.3%). The distance scales are different.
- **sDTW needs wavefront banding, not Sakoe-Chiba.** Standard Sakoe-Chiba assumes diagonal alignment. sDTW has free start (template can match anywhere in window), so there's no fixed diagonal. Use a `j_frontier` tracker that limits template column advance to `1 + band_width` per row. Without this, sDTW is unbounded O(N×M) and hangs in real-time.
- **10 examples per gesture is the stress test threshold.** With 10 examples, pairwise operations hit 45 pairs. Any O(n²) computation on raw data becomes noticeable. Always downsample (factor 4) for pairwise statistics.
- **Margin rejection is incompatible with sDTW.** sDTW free-start finds the "best matching subsequence" independently per template. During idle/transition periods, ALL templates find similar costs against the same window, producing tied distances. Margin rejection (requiring X% gap between best and second-best) blocks ALL hits when distances tie. Set `margin_rejection_ratio: 0.0` when sDTW is enabled. The per-gesture threshold already provides sufficient discrimination.
- **Production config must match benchmark config.** When a benchmark test works but production doesn't, diff the configs. Common culprits: margin_rejection_ratio, threshold_coefficient, medoid selection, use_subsequence_dtw. The benchmark is the source of truth for what settings work.
- **Only recompute thresholds on vocabulary load.** When loading a .ralf file, existing thresholds may have been computed with a different distance metric or coefficient. Always call `compute_gesture_statistics` for all gestures after loading. Without this, stale thresholds from standard DTW get used with sDTW recognition.
- **Serde defaults only apply to missing fields.** Changing `default_threshold_coefficient()` from 2.0→3.0 does NOTHING for existing .ralf files that already have `threshold_coefficient` serialized. The serde `#[serde(default = "...")]` attribute only kicks in when the field is absent from JSON. To upgrade existing vocabularies, explicitly overwrite the field on load (see `load_vocabulary` in gui/mod.rs).
- **Winner-take-all is correct; inter-class confusion is a data/threshold problem.** Only the gesture with the lowest raw distance advances its state machine each frame. Attempts to fix this in the state machine (independent advancement, margin-based selection, threshold-aware selection) all degraded benchmark accuracy (93.3% → 37-50%). When one gesture consistently steals another's detections, the fix is better training data or F1-optimized thresholds (negative examples), NOT state machine changes.
- **Skip medoid selection with sDTW.** Medoid-only recognition (1 template per gesture) inflates distances vs. threshold calibration which uses all pairwise examples. With sDTW enabled, use ALL examples as templates (match benchmark behavior). The wavefront-banded sDTW is already O(N×B) per template, so the cost is acceptable.
- **High-variance gestures get dominated in winner-take-all.** Gestures with high inter-example variance (e.g., Reach Up: threshold 102) produce higher raw distances than tight gestures (Cross Arms: threshold 22), even when the high-variance gesture is actually being performed. The tight gesture's lower absolute distance wins the competition. F1-optimized thresholds using negative examples are the right solution.

## Current Status (v0.8.0)

All core features implemented: DTW recognition with VAD-style state machine, **subsequence DTW (sDTW) with wavefront banding** for production recognition (93.3% benchmark accuracy), statistical threshold calibration (mu+sigma) with sDTW-calibrated variant, two-layer echo defense (0% echo rate in production), LB_Keogh + Sakoe-Chiba + early abandoning optimizations, onset trimming for sliding window, preprocessing pipeline (hip centering, scale normalization, velocity features), data augmentation, example quality assessment, joint weighting, consensus scoring, diagnostic logging, and research-ready vocabulary format (v1.2). Stop training button and individual example deletion. Standalone gesture viewer for visual inspection of recorded examples.

## Coding Guidelines

### Rust Conventions

- Use `Result<T, E>` for fallible operations; avoid panics in library code
- Prefer `thiserror` for custom error types
- Use `tracing` for structured logging, not `println!`
- Keep modules focused; one concept per file

### Tauri Patterns

- Commands return `Result<T, String>` for IPC error handling
- State managed via `State<Arc<Mutex<AppState>>>`
- Use `tauri::command` attribute macro for frontend-callable functions

### OSC Communication

- Default input port: 6448 (Wekinator compatible)
- Default output port: 12000
- Input address: `/wek/inputs`
- Output address: `/gesture/N` (configurable per gesture)

### Threading Model

- Main thread: Tauri GUI
- Tokio runtime: OSC receiver, OSC sender, training session timers
- Communication via channels (crossbeam or tokio::sync)

## Testing Strategy

- Unit tests for DTW algorithm and data model
- Integration tests for OSC round-trip
- Manual testing for GUI and training workflow
- Test with real skeleton data from MediaPipe pipeline

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

## Tools

### Gesture Viewer (`tools/gesture-viewer.html`)

Standalone HTML file for visual playback of recorded gesture examples. Open in any browser, drag-drop a `.ralf` file, select a gesture and example, and watch a stick figure animate through the recorded frames.

- **Skeleton rendering**: MediaPipe Pose 33 keypoints with left/right body coloring
- **Playback**: Play/pause, frame scrubber, speed control (0.25x–2x), loop
- **Ghost trails**: Configurable motion history overlay (0–15 previous frames)
- **Keyboard**: Space (play/pause), arrows (step), Home/End (jump)
- **Zero dependencies**: Single HTML file, no build step

Currently renders MediaPipe skeleton topology. For unknown `tracking_system` values, could fall back to point-cloud rendering (not yet implemented).

## Related Files

- `requirements.md` - Full specification document
- `FORMAT.md` - Complete .ralf file format specification
- `tools/gesture-viewer.html` - Standalone .ralf gesture visualizer
- `ralf-graphviz.dot` - System architecture diagram
- `RALF in context.png` - Context diagram showing RALF in the larger system
