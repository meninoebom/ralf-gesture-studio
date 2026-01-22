# RALF Gesture Studio

A desktop application for training and recognizing movement gestures using Dynamic Time Warping (DTW). Built for dancers and choreographers working with the RALF (Responsive Audio Locomotion Framework) system.

RALF Gesture Studio receives skeleton tracking data via OSC, allows users to record gesture examples through a structured training workflow, and emits OSC "hit" signals when gestures are recognized during performance.

## Prerequisites

- **Rust** (1.70 or later) - Install via [rustup](https://rustup.rs/):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

After installation, restart your terminal or run:
```bash
source "$HOME/.cargo/env"
```

## Quick Start

```bash
# Clone or navigate to the project
cd /path/to/ralf

# Build the project
cargo build

# Run the application
cargo run

# Run all tests
cargo test
```

## Project Structure

```
ralf/
├── Cargo.toml                 # Dependencies and project metadata
├── src/
│   ├── main.rs                # Entry point and integration tests
│   └── model/
│       ├── mod.rs             # Module exports
│       ├── vocabulary.rs      # Core data structures
│       └── persistence.rs     # JSON file save/load
├── CLAUDE.md                  # AI development guidelines
├── requirements.md            # Full specification document
└── .llm/
    └── active-plan.md         # Implementation roadmap
```

## Commands

| Command | Description |
|---------|-------------|
| `cargo build` | Compile the project |
| `cargo run` | Build and run the application |
| `cargo test` | Run all tests |
| `cargo test -- --nocapture` | Run tests with println output visible |
| `cargo clippy` | Run the Rust linter |
| `cargo fmt` | Format code |
| `cargo doc --open` | Generate and view documentation |

## Running Tests

```bash
# Run all tests
cargo test

# Run tests for a specific module
cargo test model::persistence

# Run a specific test by name
cargo test test_save_and_load_roundtrip

# Run tests with output visible
cargo test -- --nocapture
```

Expected output:
```
running 14 tests
test model::persistence::tests::test_default_vocabulary_dir ... ok
test model::persistence::tests::test_load_nonexistent_file ... ok
test model::persistence::tests::test_save_and_load_roundtrip ... ok
test tests::test_add_example_to_gesture ... ok
test tests::test_add_gesture ... ok
test tests::test_add_multiple_gestures ... ok
test tests::test_create_empty_vocabulary ... ok
test tests::test_default_input_config ... ok
test tests::test_default_output_config ... ok
test tests::test_gesture_ids_continue_after_load ... ok
test tests::test_get_gesture ... ok
test tests::test_remove_gesture ... ok
test tests::test_save_and_load_roundtrip ... ok
test tests::test_vocabulary_file_is_readable_json ... ok

test result: ok. 14 passed; 0 failed
```

## Data Model

### Vocabulary
A collection of gestures saved as a single `.ralf` file (JSON format).

### Gesture
A named movement pattern (e.g., "wave", "jump") with:
- Unique ID
- Recognition threshold
- OSC output address
- Recorded examples

### Example
One recording of a gesture containing timestamped motion capture frames.

### File Format

Vocabulary files are human-readable JSON with the `.ralf` extension:

```json
{
  "version": "1.0",
  "name": "House Foundations",
  "gestures": [
    {
      "id": 1,
      "name": "jack",
      "osc_address": "/gesture/1",
      "threshold": 150.0,
      "examples": []
    }
  ]
}
```

Default save location: `~/Documents/RALF/`

## Current Status

### Milestone 1: Data Model ✅
- [x] Vocabulary, Gesture, Example structs
- [x] JSON serialization (save/load .ralf files)
- [x] 14 passing tests

### Milestone 2: GUI Shell (next)
- [ ] egui window with panel layout
- [ ] Display vocabulary info

### Future Milestones
- OSC receiver (skeleton input)
- OSC sender (hit output)
- DTW recognition algorithm
- Recording and matching
- Training session workflow
- Performance mode with threshold tuning

See `.llm/active-plan.md` for the full implementation roadmap.

## Configuration Defaults

| Setting | Default | Description |
|---------|---------|-------------|
| Input port | 6448 | UDP port for skeleton data (Wekinator compatible) |
| Input address | /wek/inputs | OSC address to listen for |
| Output host | localhost | Target for hit messages |
| Output port | 12000 | UDP port for hit messages |
| Dimensions | 68 | Floats per frame (34 joints × XY) |

## Troubleshooting

### `cargo: command not found`
Rust isn't in your PATH. Run:
```bash
source "$HOME/.cargo/env"
```

### Tests fail with permission errors
The tests use temporary directories. Ensure `/tmp` is writable.

### Build fails with missing dependencies
Update your Rust toolchain:
```bash
rustup update
```

## Contributing

This project is in active development. See `requirements.md` for the full specification and `.llm/active-plan.md` for the implementation plan.

## License

MIT
