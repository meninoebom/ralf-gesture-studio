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
│   ├── model/
│   │   ├── mod.rs             # Module exports
│   │   ├── vocabulary.rs      # Core data structures
│   │   └── persistence.rs     # JSON file save/load
│   ├── gui/
│   │   └── mod.rs             # egui GUI (Training/Performance modes)
│   └── osc/
│       ├── mod.rs             # Module exports
│       ├── receiver.rs        # Async OSC receiver with status tracking
│       └── sender.rs          # OSC sender for hit messages
│   └── engine/
│       ├── mod.rs             # Module exports
│       ├── dtw.rs             # Dynamic Time Warping algorithm
│       ├── buffer.rs          # Frame buffer and recording session
│       ├── recognizer.rs      # Real-time gesture recognition
│       └── training.rs        # Training session state machine with audio cues
├── test_osc_sender.py         # Python script to test OSC input
├── test_osc_receiver.py       # Python script to test OSC output
├── tools/
│   └── gesture-viewer.html    # Standalone .ralf gesture visualizer
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
running 76 tests
test model::persistence::tests::test_default_vocabulary_dir ... ok
test model::persistence::tests::test_load_nonexistent_file ... ok
test model::persistence::tests::test_save_and_load_roundtrip ... ok
test osc::receiver::tests::test_frame_count_increments ... ok
test osc::receiver::tests::test_handle_polls_events ... ok
test osc::receiver::tests::test_receiver_creation ... ok
test osc::sender::tests::test_ms_since_last_send ... ok
test osc::sender::tests::test_send_increments_count ... ok
test osc::sender::tests::test_sender_config_update ... ok
test osc::sender::tests::test_sender_creation ... ok
test engine::dtw::tests::test_dtw_identical_sequences ... ok
test engine::dtw::tests::test_euclidean_simple ... ok
... (22 DTW tests, 10 training tests)
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

test result: ok. 76 passed; 0 failed
```

## Testing OSC Input

A Python test script is included to simulate skeleton tracking data. Requires [uv](https://github.com/astral-sh/uv) (recommended) or Python with `python-osc`.

### Using uv (recommended)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the test sender (dependencies auto-installed)
uv run test_osc_sender.py
```

### Manual Testing

```bash
# In terminal 1: Start the app
cargo run

# In terminal 2: Send test OSC data
uv run test_osc_sender.py
```

You should see the GUI update:
- **LISTENING** (yellow) → waiting for data
- **RECEIVING** (green) → data arriving, shows ms since last frame
- **Frames: N** → increments as data arrives

### Test Script Options

```bash
uv run test_osc_sender.py --help

Options:
  --host HOST        Target host (default: 127.0.0.1)
  --port PORT        Target port (default: 6448)
  --address ADDRESS  OSC address (default: /wek/inputs)
  --fps FPS          Frames per second (default: 60)
  --dimensions N     Number of floats per frame (default: 4)
```

## Testing OSC Output

A Python test receiver script is included to verify hit messages are being sent.

### Testing the Send Button

```bash
# In terminal 1: Start the receiver
uv run test_osc_receiver.py

# In terminal 2: Start the app
cargo run

# In the app: Click "Send Test Hit" button in the CONNECTION panel
```

You should see output in the receiver terminal:
```
[14:32:01.123] /test/hit → 1.0
```

The GUI will show:
- **SENT** (green) → message sent, shows ms since last send
- **Sent: N** → increments each time you click the button

### Receiver Script Options

```bash
uv run test_osc_receiver.py --help

Options:
  --host HOST   Listen host (default: 127.0.0.1)
  --port PORT   Listen port (default: 12000)
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

### File Format (v1.1)

Vocabulary files are human-readable JSON with the `.ralf` extension.

**Key features:**
- **UUID**: Each vocabulary has a unique identifier for cross-system references
- **Research metadata**: Tracking system, coordinate system, license, tags for future computational musicology work
- **Automatic migration**: v1.0 files are upgraded when loaded

```json
{
  "version": "1.1",
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "name": "House Foundations",
  "tracking_system": "mediapipe-pose-33-xy",
  "coordinate_system": "normalized-0-1-xy",
  "source_fps": 60.0,
  "license": "CC-BY-4.0",
  "tags": ["house", "dance", "foundations"],
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

See `FORMAT.md` for complete field reference.

## Gesture Viewer

A standalone HTML tool for visual playback of recorded gesture examples. No install required — open in any browser.

```bash
open tools/gesture-viewer.html
```

Drag a `.ralf` file onto the page, select a gesture and example, and watch the stick figure animate through the recorded frames. Includes playback controls, speed adjustment, and ghost trails for visualizing motion over time.

## Current Status

### Milestone 1: Data Model ✅
- [x] Vocabulary, Gesture, Example structs
- [x] JSON serialization (save/load .ralf files)

### Milestone 2: GUI Shell ✅
- [x] egui window with panel layout
- [x] Training and Performance mode views
- [x] Vocabulary, Connection, Gestures panels
- [x] Custom gold color for better readability
- [x] 30% larger fonts for accessibility

### Milestone 3: OSC Receiver ✅
- [x] Async UDP receiver with tokio
- [x] Live connection status (Stopped → Listening → Receiving)
- [x] Frame counter and time-since-last-frame display
- [x] Error handling and status indicators

### Milestone 4: OSC Sender ✅
- [x] Send hit messages via UDP
- [x] "Send Test Hit" button in GUI
- [x] Output status indicator (Ready → Sent)
- [x] Time-since-last-send and send count display

### Milestone 5: DTW Algorithm ✅
- [x] Euclidean distance for frame comparison
- [x] DTW distance function with dynamic programming
- [x] Normalized DTW for comparing different-length sequences
- [x] Best match finder for gesture recognition

### Milestone 6: Recording + Matching ✅
- [x] Frame buffer for incoming OSC data
- [x] Recording session with progress bar
- [x] Real-time gesture recognition with DTW
- [x] Hit detection and OSC hit output
- [x] Refractory period (1 second between same-gesture hits)
- [x] Hit log with timestamps

### Milestone 7: Training Session ✅
- [x] Training session state machine (IDLE → COUNTDOWN → CAPTURING → RESTING → COMPLETE)
- [x] Configurable parameters (reps, duration, rest time)
- [x] Countdown timer with visual display
- [x] Audio beeps using rodio (countdown ticks, capture start/end, completion ding)
- [x] Spacebar to start, Escape to cancel
- [x] Progress indicator ("Recording 3 of 5")
- [x] 76 passing tests

### Milestone 8: Polish + Performance Mode ✅
- [x] File operations (New, Open, Save As) with native file dialogs
- [x] Gesture management (add, rename, delete)
- [x] Threshold sliders in Performance mode (real-time adjustment)
- [x] OSC address editing
- [x] Auto-save when file path exists
- [x] Dirty indicator in title bar
- [x] 76 passing tests

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
