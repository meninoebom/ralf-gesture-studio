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
| `directories` | Cross-platform default paths |
| `crossbeam-channel` | Thread-safe communication |

## Data Model

**Hierarchy**: Vocabulary → Gesture → Example

- **Vocabulary**: Collection of gestures, stored as single .ralf file
- **Gesture**: Named movement pattern with threshold and OSC output address
- **Example**: One recorded instance (timestamped frames of skeleton data)

File location: `~/Documents/RALF/` by default

## Recognition Algorithm (Wekinator-Style)

The recognizer uses a DTW approach modeled after Wekinator's proven implementation.

**Reference**: `fiebrink1/wekinator` - `src/wekimini/learning/dtw/DtwModel.java`

### Breakthrough (2026-01-26)

**Simple implementations work best.** After multiple complex attempts failed:
- Fixed window size (matches first training example length)
- Compare against ALL training examples
- Simple threshold check: distance < threshold = hit
- Frame skipping (DTW every 4th frame) for performance
- Downsampling (compare at 15fps, not 60fps) for performance

User successfully recognized gestures at threshold ~8000.

### How It Works

1. **Fixed Window**: Window size = first training example's length
2. **Compare Against All Examples**: For each gesture, compare against every training example (not just a prototype)
3. **Simple Threshold Check**: If best distance < threshold, fire the gesture; otherwise return "no match"
4. **Cooldown Period**: Prevent same gesture from firing repeatedly (default: 500ms)
5. **Performance**: Skip frames + downsample = ~64x faster than naive implementation

### Key Code (from recognizer.rs)

```rust
// Generate candidates between min/max example lengths
let candidate_lengths = self.generate_candidate_lengths();

// Compare against all examples for all gestures
for candidate_len in &candidate_lengths {
    let window = self.buffer.downsampled(*candidate_len, downsample_factor);
    for gesture in &self.gestures {
        for example in gesture.examples() {
            let distance = dtw_distance(&window, example);
            // Track best match...
        }
    }
}

// Simple threshold check (Wekinator-style)
if best_distance < gesture.threshold {
    return Some(hit);  // Matched!
} else {
    return Some(no_match);  // Idle state
}
```

### Configuration

```rust
RecognitionConfig {
    cooldown_ms: 400,       // Min time between same gesture hits
    downsample_factor: 4,   // 60fps → 15fps
    num_candidates: 5,      // Window sizes to try
}
```

See `.llm/active-plan.md` for detailed algorithm documentation and Wekinator source references.

## Implementation Status

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
- `ralf-graphviz.dot` - System architecture diagram
- `RALF in context.png` - Context diagram showing RALF in the larger system

## Out of Scope (v1.0)

- MIDI input for training triggers
- Individual example management (view/delete)
- Threshold auto-calibration
- Gesture visualization/playback
- Wekinator project import
