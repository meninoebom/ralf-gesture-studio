# RALF Gesture Studio - Implementation Plan

**Approach**: 8 small milestones, each with tests and visible progress
**Target user**: Developer learning Rust while building

---

## Milestone 1: Project Setup + Data Model ✅

**Goal**: Rust project that can create, save, and load vocabulary files.

### What to Build

- [x] Cargo project with dependencies
- [x] `Vocabulary` struct with nested `Gesture` and `Example`
- [x] JSON serialization (save to .ralf file)
- [x] JSON deserialization (load from .ralf file)
- [x] Default file location (`~/Documents/RALF/`)

### Tests to Write

```rust
#[test]
fn test_create_empty_vocabulary() {
    let vocab = Vocabulary::new("Test Vocab");
    assert_eq!(vocab.name, "Test Vocab");
    assert!(vocab.gestures.is_empty());
}

#[test]
fn test_add_gesture() {
    let mut vocab = Vocabulary::new("Test");
    vocab.add_gesture("wave");
    assert_eq!(vocab.gestures.len(), 1);
    assert_eq!(vocab.gestures[0].name, "wave");
}

#[test]
fn test_save_and_load_roundtrip() {
    let mut vocab = Vocabulary::new("Test");
    vocab.add_gesture("wave");

    let path = "/tmp/test_vocab.ralf";
    vocab.save(path).unwrap();

    let loaded = Vocabulary::load(path).unwrap();
    assert_eq!(loaded.name, "Test");
    assert_eq!(loaded.gestures.len(), 1);
}
```

### How to Verify

```bash
cargo test
```

Expected output:
```
running 3 tests
test model::tests::test_create_empty_vocabulary ... ok
test model::tests::test_add_gesture ... ok
test model::tests::test_save_and_load_roundtrip ... ok
```

### Rust Concepts Introduced

- `struct` and nested structs
- `impl` blocks and methods
- `serde` derive macros (`Serialize`, `Deserialize`)
- `Result<T, E>` for error handling
- File I/O with `std::fs`
- `chrono` for timestamps

### Checkpoint

**What works**: You can create a vocabulary in code, add gestures, save it to a .ralf file, and load it back. The JSON file is human-readable.

**What doesn't work yet**: No GUI, no OSC, no recognition.

---

## Milestone 2: Minimal GUI Shell ✅

**Goal**: A window that displays vocabulary info and has the basic panel layout.

### What to Build

- [x] eframe app skeleton
- [x] Main window with title bar
- [x] Mode selector (Training/Performance) - visual only, no logic yet
- [x] Vocabulary panel showing name and gesture count
- [x] Placeholder panels for Connection, Gestures, Train
- [x] Load a hardcoded or test vocabulary on startup

### GUI Appearance

```
┌─────────────────────────────────────────────────┐
│  RALF Gesture Studio          [Training ▼]     │
├─────────────────────────────────────────────────┤
│ ┌─ VOCABULARY ────────────────────────────────┐ │
│ │  Name: Test Vocabulary                      │ │
│ │  Gestures: 2                                │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ CONNECTION ────────────────────────────────┐ │
│ │  (placeholder)                              │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ GESTURES ──────────────────────────────────┐ │
│ │  (placeholder)                              │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### How to Verify

```bash
cargo run
```

- Window opens
- Title shows "RALF Gesture Studio"
- Vocabulary name and gesture count are visible
- Panels are laid out correctly

### Rust Concepts Introduced

- `eframe::App` trait implementation
- `egui::CentralPanel`, `egui::TopBottomPanel`
- `egui::Frame` for styled containers
- Mutable app state (`&mut self`)
- The immediate mode GUI paradigm

### Checkpoint

**What works**: GUI window opens, shows vocabulary data, has the panel structure.

**What doesn't work yet**: No file operations in GUI, no OSC, buttons don't do anything.

---

## Milestone 3: OSC Receiver ✅

**Goal**: Receive skeleton data via OSC and show connection status in GUI.

### What to Build

- [x] UDP socket listener on configurable port (default 6448)
- [x] OSC message parsing with `rosc`
- [x] Filter for configured address (default `/wek/inputs`)
- [x] Channel to send frames from receiver task to GUI
- [x] Connection status indicator (LISTENING → RECEIVING)
- [x] Frame counter in GUI
- [x] "Time since last frame" display

### GUI Appearance

```
┌─ CONNECTION ──────────────────────────────────┐
│  INPUT                                        │
│  Port: 6448  Address: /wek/inputs             │
│  ● RECEIVING (12ms ago)                       │
│  Frames: 1,247                                │
└───────────────────────────────────────────────┘
```

### Tests to Write

```rust
#[test]
fn test_parse_osc_message() {
    // Create a mock OSC message with float array
    let msg = create_test_osc_message("/wek/inputs", vec![0.1, 0.2, 0.3]);
    let frame = parse_osc_frame(&msg).unwrap();
    assert_eq!(frame.len(), 3);
}

#[test]
fn test_reject_wrong_address() {
    let msg = create_test_osc_message("/wrong/address", vec![0.1, 0.2]);
    let result = parse_osc_frame(&msg);
    assert!(result.is_none());
}
```

### How to Verify

1. Run the app: `cargo run`
2. Send test OSC messages (use a tool like `oscsend` or a Python script)
3. Watch the frame counter increment
4. Status should show "RECEIVING" with recent timestamp

Test sender script (Python):
```python
# test_osc_sender.py
from pythonosc import udp_client
import time

client = udp_client.SimpleUDPClient("127.0.0.1", 6448)
while True:
    # Send 4 floats as test data
    client.send_message("/wek/inputs", [0.1, 0.2, 0.3, 0.4])
    time.sleep(0.016)  # ~60fps
```

### Rust Concepts Introduced

- `tokio` async runtime
- `tokio::spawn` for background tasks
- `std::sync::mpsc` or `tokio::sync::mpsc` channels
- `Arc<Mutex<T>>` for shared state (or channel-based approach)
- UDP sockets with `tokio::net::UdpSocket`

### Checkpoint

**What works**: App receives OSC data and shows live connection status. You can see frames arriving.

**What doesn't work yet**: Data isn't stored or used for anything. No sending, no recording.

---

## Milestone 4: OSC Sender

**Goal**: Send OSC hit messages and verify with test button.

### What to Build

- [ ] OSC sender task/function
- [ ] Configurable output host and port (default localhost:12000)
- [ ] "Send Test Hit" button in GUI
- [ ] Output status indicator (READY → SENT)
- [ ] "Time since last send" display

### GUI Appearance

```
┌─ CONNECTION ──────────────────────────────────┐
│  INPUT                      OUTPUT            │
│  Port: 6448                 Host: localhost   │
│  ● RECEIVING (3ms ago)      Port: 12000       │
│                             ● SENT (1.2s ago) │
│                             [Send Test Hit]   │
└───────────────────────────────────────────────┘
```

### Tests to Write

```rust
#[test]
fn test_create_hit_message() {
    let msg = create_hit_message("/gesture/1");
    assert_eq!(msg.addr, "/gesture/1");
}

// Integration test - requires port binding
#[test]
fn test_send_receive_hit() {
    // Bind a receiver on port 12000
    // Send a hit
    // Verify message received
}
```

### How to Verify

1. Run a listener: `oscdump 12000` (or Python script below)
2. Run the app: `cargo run`
3. Click "Send Test Hit"
4. See message arrive in listener

Test receiver script (Python):
```python
# test_osc_receiver.py
from pythonosc import dispatcher, osc_server

def handle_hit(address, *args):
    print(f"Received: {address} {args}")

disp = dispatcher.Dispatcher()
disp.set_default_handler(handle_hit)

server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 12000), disp)
print("Listening on port 12000...")
server.serve_forever()
```

### Rust Concepts Introduced

- Sending UDP packets
- Building OSC messages with `rosc`
- GUI button callbacks
- Updating shared state from async task

### Checkpoint

**What works**: Full OSC round-trip. Can receive skeleton data and send hit messages. Connection status shows both directions.

**What doesn't work yet**: No recognition logic, no recording, hits are manual only.

---

## Milestone 5: DTW Algorithm

**Goal**: Implement Dynamic Time Warping and verify with tests.

### What to Build

- [ ] DTW distance function for two sequences
- [ ] Support for multi-dimensional frames (e.g., 68 floats per frame)
- [ ] Euclidean distance for frame comparison
- [ ] Tests with known sequences and expected distances

### Tests to Write

```rust
#[test]
fn test_dtw_identical_sequences() {
    let seq = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let distance = dtw_distance(&seq, &seq);
    assert_eq!(distance, 0.0);
}

#[test]
fn test_dtw_different_sequences() {
    let seq1 = vec![vec![0.0], vec![1.0], vec![2.0]];
    let seq2 = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let distance = dtw_distance(&seq1, &seq2);
    assert!(distance > 0.0);
    assert!(distance < 2.0); // Should be relatively close
}

#[test]
fn test_dtw_time_warping() {
    // Same pattern, different speeds
    let slow = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0]];
    let fast = vec![vec![0.0], vec![1.0]];
    let distance = dtw_distance(&slow, &fast);
    // Should recognize as similar despite timing difference
    assert!(distance < 1.0);
}

#[test]
fn test_dtw_completely_different() {
    let seq1 = vec![vec![0.0, 0.0]];
    let seq2 = vec![vec![100.0, 100.0]];
    let distance = dtw_distance(&seq1, &seq2);
    assert!(distance > 100.0);
}
```

### How to Verify

```bash
cargo test engine::dtw
```

All DTW tests should pass. This is a pure algorithm milestone - no GUI changes.

### Rust Concepts Introduced

- 2D vectors (`Vec<Vec<f32>>`)
- Implementing algorithms from scratch
- `f32::min()` and floating point operations
- Optional: using a crate like `augurs-dtw` instead of rolling your own

### Checkpoint

**What works**: DTW algorithm correctly computes distance between two sequences. Tests prove it handles identical, similar, and different sequences correctly.

**What doesn't work yet**: Not connected to anything. No recording, no real-time matching.

---

## Milestone 6: Recording + Matching

**Goal**: Record examples and recognize gestures in real-time.

### What to Build

- [ ] Frame buffer that stores incoming OSC data
- [ ] "Record" button that captures N seconds of frames
- [ ] Store recorded frames as Example in current Gesture
- [ ] Sliding window buffer for real-time matching
- [ ] Continuous DTW comparison against stored examples
- [ ] Hit detection when distance < threshold
- [ ] Fire OSC hit when gesture detected
- [ ] Refractory period (don't re-trigger same gesture immediately)

### GUI Appearance

```
┌─ GESTURES ────────────────────────────────────┐
│  ●  wave     3 examples    /gesture/1         │
│  ○  jump     0 examples    /gesture/2         │
│                                               │
│  [+ Add Gesture]                              │
└───────────────────────────────────────────────┘

┌─ RECORD ──────────────────────────────────────┐
│  Gesture: [wave ▼]    Duration: [3.0]s        │
│                                               │
│  [ ● RECORD ]                                 │
│                                               │
│  Status: Recorded! (174 frames)               │
└───────────────────────────────────────────────┘

┌─ RECOGNITION ─────────────────────────────────┐
│  ● ACTIVE                                     │
│                                               │
│  wave: distance 89  (threshold: 150) ✓        │
│  jump: distance --  (no examples)             │
│                                               │
│  Last hit: wave (0.8s ago)                    │
└───────────────────────────────────────────────┘
```

### Tests to Write

```rust
#[test]
fn test_buffer_stores_frames() {
    let mut buffer = FrameBuffer::new(100);
    buffer.push(vec![1.0, 2.0]);
    buffer.push(vec![3.0, 4.0]);
    assert_eq!(buffer.len(), 2);
}

#[test]
fn test_buffer_slides() {
    let mut buffer = FrameBuffer::new(3);
    buffer.push(vec![1.0]);
    buffer.push(vec![2.0]);
    buffer.push(vec![3.0]);
    buffer.push(vec![4.0]); // Should drop first frame
    assert_eq!(buffer.len(), 3);
    assert_eq!(buffer.frames()[0], vec![2.0]);
}

#[test]
fn test_recognition_fires_hit() {
    // Record an example
    // Play back similar data
    // Verify hit is detected
}
```

### How to Verify

1. Run app with OSC sender script providing input
2. Select a gesture and click Record
3. Move/generate distinct pattern during recording
4. Stop recording, see example count increase
5. Repeat the pattern
6. See "Last hit" update and OSC message sent

### Rust Concepts Introduced

- Circular buffers / VecDeque
- Combining multiple async streams
- State management with multiple interacting components
- Threshold comparison logic

### Checkpoint

**What works**: Full recognition loop! Can record examples and recognize them in real-time. OSC hits are sent automatically.

**What doesn't work yet**: No structured training session (countdown/rest), no audio feedback, basic UI for recording.

---

## Milestone 7: Training Session

**Goal**: Full training workflow with countdown, capture, rest, and audio cues.

### What to Build

- [ ] Training session state machine (IDLE → COUNTDOWN → CAPTURING → RESTING → COMPLETE)
- [ ] Configurable parameters (reps, duration, rest time)
- [ ] Countdown timer with visual display
- [ ] Audio beeps using `rodio`:
  - Countdown ticks (300 Hz, short)
  - Capture start (800 Hz, long)
  - Capture end (600 Hz, long)
  - Session complete (1000 Hz, double ding)
- [ ] Spacebar to start, Escape to cancel
- [ ] Progress indicator ("Recording 3 of 5")

### GUI Appearance (During Capture)

```
┌─ TRAIN ───────────────────────────────────────┐
│  Gesture: [wave ▼]  Reps: [5]  Duration: [3]s │
│                                               │
│  ┌─────────────────────────────────────────┐  │
│  │           ███ CAPTURING ███             │  │
│  │                                         │  │
│  │                2.1s                     │  │
│  │         [████████████░░░░░░]            │  │
│  │                                         │  │
│  └─────────────────────────────────────────┘  │
│                                               │
│  Recording example 3 of 5 for "wave"          │
│  Press [Esc] to cancel                        │
└───────────────────────────────────────────────┘
```

### Tests to Write

```rust
#[test]
fn test_session_state_transitions() {
    let mut session = TrainingSession::new();
    assert_eq!(session.state, SessionState::Idle);

    session.start(gesture_id, reps: 3, duration: 2.0, rest: 1.0);
    assert_eq!(session.state, SessionState::Countdown);

    // Simulate countdown complete
    session.on_countdown_complete();
    assert_eq!(session.state, SessionState::Capturing);

    // Simulate capture complete
    session.on_capture_complete();
    assert_eq!(session.state, SessionState::Resting);
    assert_eq!(session.completed_reps, 1);
}

#[test]
fn test_session_cancel() {
    let mut session = TrainingSession::new();
    session.start(...);
    session.cancel();
    assert_eq!(session.state, SessionState::Idle);
}
```

### How to Verify

1. Run app with OSC input active
2. Select gesture, set reps to 3
3. Press spacebar or click Start
4. Hear countdown ticks: tick... tick... tick...
5. Hear capture beep, see "CAPTURING" state
6. Perform gesture
7. Hear end beep, see "RESTING" state
8. Repeat for each rep
9. Hear double-ding on completion
10. Check gesture now has 3 new examples

### Rust Concepts Introduced

- State machines with enums
- `rodio` audio synthesis
- Timer management with `tokio::time`
- Keyboard input handling in egui

### Checkpoint

**What works**: Full training UX! Dancers can train gestures hands-free with audio cues. The flow-state-friendly design goal is achieved.

**What doesn't work yet**: No performance mode view, no threshold adjustment, no hit log.

---

## Milestone 8: Polish + Performance Mode

**Goal**: Complete the app with both modes, threshold tuning, and hit log.

### What to Build

- [ ] Mode switching (Training ↔ Performance)
- [ ] Performance mode layout:
  - Live distance display for each gesture
  - Threshold sliders (adjustable in real-time)
  - Hit detection indicator
  - Hit log with timestamps
- [ ] File operations in GUI:
  - New vocabulary
  - Open vocabulary (file picker)
  - Save As
- [ ] Gesture management:
  - Add gesture
  - Rename gesture (inline edit)
  - Delete gesture
  - Edit OSC address
- [ ] Input/output port configuration
- [ ] Auto-save after changes

### GUI: Performance Mode

```
┌─────────────────────────────────────────────────┐
│  RALF Gesture Studio          [Performance ▼]  │
├─────────────────────────────────────────────────┤
│ ┌─ VOCABULARY ────────────────────────────────┐ │
│ │  House Foundations                  [Open]  │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ CONNECTION ────────────────────────────────┐ │
│ │  INPUT: ● RECEIVING    OUTPUT: ● READY      │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ GESTURE MONITOR ───────────────────────────┐ │
│ │  Gesture   Threshold      Distance   Status │ │
│ │  ──────────────────────────────────────────│ │
│ │  wave      [===●===] 150  ████░░ 89    ●   │ │
│ │  jack      [==●====] 120  ██░░░░ 43   ███  │ │
│ │                                             │ │
│ │            ★ jack DETECTED ★                │ │
│ └─────────────────────────────────────────────┘ │
│ ┌─ HIT LOG ───────────────────────────────────┐ │
│ │  14:32:01  jack  dist: 43  → /gesture/2     │ │
│ │  14:31:58  wave  dist: 72  → /gesture/1     │ │
│ └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

### Tests to Write

```rust
#[test]
fn test_threshold_adjustment_affects_recognition() {
    // With threshold 100, gesture is recognized
    // With threshold 50, same distance is rejected
}

#[test]
fn test_hit_log_records_events() {
    let mut log = HitLog::new(100); // Keep last 100
    log.record(Hit { gesture: "wave", distance: 43.0, ... });
    assert_eq!(log.len(), 1);
}

#[test]
fn test_auto_save_on_change() {
    // Modify vocabulary
    // Verify file was updated
}
```

### How to Verify

1. Create vocabulary with 2-3 trained gestures
2. Switch to Performance mode
3. Generate OSC input with gestures
4. See distances update in real-time
5. Adjust threshold slider, see recognition behavior change
6. Check hit log shows all recognized gestures
7. Switch back to Training mode, add gesture
8. Verify file was saved automatically

### Rust Concepts Introduced

- Complex UI state management
- File dialogs with `rfd` crate
- Real-time slider updates
- Scrolling lists in egui

### Checkpoint

**What works**: Everything! Full application with training and performance modes. Ready for real-world use.

**Future enhancements** (post-v1.0): MIDI input, example management, gesture visualization, threshold auto-calibration.

---

## Summary: Test Commands at Each Milestone

| Milestone | Test Command | What to See |
|-----------|--------------|-------------|
| 1 | `cargo test` | 3 data model tests pass |
| 2 | `cargo run` | GUI window with panels |
| 3 | `cargo run` + OSC sender | "RECEIVING" status, frame count |
| 4 | `cargo run` + click button | OSC hit received externally |
| 5 | `cargo test engine::dtw` | DTW algorithm tests pass |
| 6 | `cargo run` + record + perform | Hit detected and sent |
| 7 | `cargo run` + spacebar | Training session with beeps |
| 8 | `cargo run` | Full app, both modes work |

---

## Getting Started

When ready to begin Milestone 1, we'll:

1. Create the Cargo project with dependencies
2. Set up the module structure
3. Implement `Vocabulary`, `Gesture`, `Example` structs
4. Add serde serialization
5. Write and run the tests

Each milestone builds on the previous. Take them one at a time.
