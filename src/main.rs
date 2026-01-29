mod model;
mod gui;
mod osc;
mod engine;

use std::sync::{Arc, Mutex};
use gui::AppState;
use osc::OscReceiver;

fn main() {
    // Create the OSC receiver and get the handle for the app
    let (receiver, receiver_handle) = OscReceiver::new(6448, "/wek/inputs");

    // Start the tokio runtime in a background thread for async OSC receiving
    std::thread::spawn(move || {
        let rt = tokio::runtime::Runtime::new().expect("Failed to create tokio runtime");
        rt.block_on(receiver.run());
    });

    // Create shared application state
    let app_state = Arc::new(Mutex::new(AppState::new(receiver_handle)));

    // Run the Tauri application
    tauri::Builder::default()
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            gui::get_state,
            gui::set_mode,
            gui::new_vocabulary,
            gui::open_vocabulary,
            gui::save_vocabulary,
            gui::send_test_hit,
            gui::add_gesture,
            gui::select_gesture,
            gui::rename_gesture,
            gui::delete_gesture,
            gui::start_training,
            gui::cancel_training,
            gui::set_threshold,
            gui::toggle_threshold_mode,
            gui::set_cooldown,
            gui::enable_diagnostics,
            gui::disable_diagnostics,
            gui::is_diagnostics_enabled,
        ])
        .run(tauri::generate_context!())
        .expect("Error running Tauri application");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use crate::model::{Vocabulary, Example, load_vocabulary, save_vocabulary};
    use tempfile::tempdir;

    #[test]
    fn test_create_empty_vocabulary() {
        let vocab = Vocabulary::new("Test Vocab");
        assert_eq!(vocab.name, "Test Vocab");
        assert!(vocab.gestures.is_empty());
        assert_eq!(vocab.version, Vocabulary::CURRENT_VERSION);
    }

    #[test]
    fn test_add_gesture() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");

        assert_eq!(vocab.gestures.len(), 1);
        assert_eq!(vocab.gestures[0].name, "wave");
        assert_eq!(vocab.gestures[0].id, id);
        assert_eq!(vocab.gestures[0].osc_address, "/gesture/1");
    }

    #[test]
    fn test_add_multiple_gestures() {
        let mut vocab = Vocabulary::new("Test");
        vocab.add_gesture("wave");
        vocab.add_gesture("jump");
        vocab.add_gesture("spin");

        assert_eq!(vocab.gestures.len(), 3);
        assert_eq!(vocab.gestures[0].id, 1);
        assert_eq!(vocab.gestures[1].id, 2);
        assert_eq!(vocab.gestures[2].id, 3);
    }

    #[test]
    fn test_remove_gesture() {
        let mut vocab = Vocabulary::new("Test");
        vocab.add_gesture("wave");
        let id = vocab.add_gesture("jump");

        assert!(vocab.remove_gesture(id));
        assert_eq!(vocab.gestures.len(), 1);
        assert_eq!(vocab.gestures[0].name, "wave");
    }

    #[test]
    fn test_get_gesture() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");

        let gesture = vocab.get_gesture(id);
        assert!(gesture.is_some());
        assert_eq!(gesture.unwrap().name, "wave");

        assert!(vocab.get_gesture(999).is_none());
    }

    #[test]
    fn test_add_example_to_gesture() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");

        // Create some fake frame data
        let frames = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ];
        let example = Example::new(frames, 500);

        let gesture = vocab.get_gesture_mut(id).unwrap();
        gesture.add_example(example);

        assert_eq!(gesture.example_count(), 1);
        assert!(gesture.has_examples());
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let mut vocab = Vocabulary::new("Test Vocabulary");
        vocab.add_gesture("wave");

        // Add an example with frame data
        let frames = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let example = Example::new(frames, 1000);
        vocab.get_gesture_mut(1).unwrap().add_example(example);

        // Save to temp file
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_vocab.ralf");

        save_vocabulary(&vocab, &path).unwrap();

        // Load it back
        let loaded = load_vocabulary(&path).unwrap();

        assert_eq!(loaded.name, "Test Vocabulary");
        assert_eq!(loaded.gestures.len(), 1);
        assert_eq!(loaded.gestures[0].name, "wave");
        assert_eq!(loaded.gestures[0].example_count(), 1);
        assert_eq!(loaded.gestures[0].examples[0].frame_count, 2);
    }

    #[test]
    fn test_vocabulary_file_is_readable_json() {
        let mut vocab = Vocabulary::new("My Vocabulary");
        vocab.add_gesture("test");

        let dir = tempdir().unwrap();
        let path = dir.path().join("readable.ralf");

        save_vocabulary(&vocab, &path).unwrap();

        // Read the raw file and verify it's valid JSON
        let contents = std::fs::read_to_string(&path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&contents).unwrap();

        assert_eq!(parsed["name"], "My Vocabulary");
        assert_eq!(parsed["version"], Vocabulary::CURRENT_VERSION);
        assert!(parsed["gestures"].is_array());
    }

    #[test]
    fn test_default_input_config() {
        let vocab = Vocabulary::new("Test");
        assert_eq!(vocab.input.port, 6448);
        assert_eq!(vocab.input.address, "/wek/inputs");
        assert_eq!(vocab.input.dimensions, 66);
    }

    #[test]
    fn test_default_output_config() {
        let vocab = Vocabulary::new("Test");
        assert_eq!(vocab.output.host, "127.0.0.1");
        assert_eq!(vocab.output.port, 12000);
    }

    #[test]
    fn test_gesture_ids_continue_after_load() {
        let mut vocab = Vocabulary::new("Test");
        vocab.add_gesture("one");
        vocab.add_gesture("two");

        let dir = tempdir().unwrap();
        let path = dir.path().join("test.ralf");
        save_vocabulary(&vocab, &path).unwrap();

        let mut loaded = load_vocabulary(&path).unwrap();
        let new_id = loaded.add_gesture("three");

        // New gesture should have id 3, not 1
        assert_eq!(new_id, 3);
    }
}
