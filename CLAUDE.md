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
├── main.rs                     # Entry point, launch GUI
├── model/                      # Core data structures (Vocabulary, Gesture, Example)
├── engine/                     # DTW recognition engine
├── session/                    # Training session state machine
├── osc/                        # OSC send/receive
├── app/                        # Application state
└── gui/                        # egui UI components
```

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `eframe` / `egui` | Immediate mode GUI |
| `rosc` | OSC message encoding/decoding |
| `tokio` | Async runtime for OSC + timers |
| `rodio` | Audio feedback (training beeps) |
| `serde` + `serde_json` | Vocabulary serialization |
| `directories` | Cross-platform default paths |

## Data Model

**Hierarchy**: Vocabulary → Gesture → Example

- **Vocabulary**: Collection of gestures, stored as single .ralf file
- **Gesture**: Named movement pattern with threshold and OSC output address
- **Example**: One recorded instance (timestamped frames of skeleton data)

File location: `~/Documents/RALF/` by default

## Implementation Phases

Development follows four phases (see requirements.md for details):

1. **Phase 1 - Foundation**: Basic recording and recognition of one gesture
2. **Phase 2 - Training UX**: Full training session workflow with audio feedback
3. **Phase 3 - Full GUI**: Complete interface with both modes
4. **Phase 4 - Polish**: Error handling, optimization, packaging

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
