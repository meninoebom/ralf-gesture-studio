mod gui;
mod osc;

use gui::AppState;
use osc::OscReceiver;
use std::sync::{Arc, Mutex};

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
            gui::delete_example,
            gui::start_training,
            gui::cancel_training,
            gui::set_threshold,
            gui::toggle_threshold_mode,
            gui::set_cooldown,
            gui::enable_diagnostics,
            gui::disable_diagnostics,
            gui::is_diagnostics_enabled,
            gui::set_augmentation_enabled,
            gui::set_joint_weighting,
            gui::set_complexity_correction,
            gui::set_consensus,
        ])
        .run(tauri::generate_context!())
        .expect("Error running Tauri application");
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use ralf_gesture_studio::model::{load_vocabulary, save_vocabulary, Example, Vocabulary};
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
    fn test_preprocessing_config_defaults_on() {
        let vocab = Vocabulary::new("Test");
        assert!(vocab.preprocessing.hip_normalize);
        assert!(vocab.preprocessing.scale_normalize);
        assert!(!vocab.preprocessing.velocity_features);
    }

    #[test]
    fn test_preprocessing_config_roundtrip() {
        use ralf_gesture_studio::engine::preprocess::PreprocessingConfig;

        let mut vocab = Vocabulary::new("Test Preprocessing");
        vocab.preprocessing = PreprocessingConfig {
            hip_normalize: true,
            scale_normalize: true,
            velocity_features: false,
            angle_features: false,
        };
        vocab.add_gesture("wave");

        let dir = tempdir().unwrap();
        let path = dir.path().join("preprocess.ralf");
        save_vocabulary(&vocab, &path).unwrap();

        let loaded = load_vocabulary(&path).unwrap();
        assert!(loaded.preprocessing.hip_normalize);
        assert!(loaded.preprocessing.scale_normalize);
        assert!(!loaded.preprocessing.velocity_features);
        assert_eq!(loaded.version, "1.2");
    }

    #[test]
    fn test_passthrough_preprocessing_matches_raw() {
        use ralf_gesture_studio::engine::preprocess::{PreprocessingConfig, Preprocessor};

        // With all preprocessing OFF, output should equal input
        let config = PreprocessingConfig {
            hip_normalize: false,
            scale_normalize: false,
            velocity_features: false,
            angle_features: false,
        };
        let preprocessor = Preprocessor::new(config, "mediapipe-pose-33-xy");

        let raw_frames: Vec<Vec<f32>> = vec![vec![0.1; 66], vec![0.2; 66], vec![0.3; 66]];

        let processed = preprocessor.process_sequence(&raw_frames);
        assert_eq!(processed.len(), raw_frames.len());
        for (raw, proc) in raw_frames.iter().zip(processed.iter()) {
            assert_eq!(raw.len(), proc.len());
            for (r, p) in raw.iter().zip(proc.iter()) {
                assert!((r - p).abs() < 1e-6);
            }
        }
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

    #[test]
    fn test_augmentation_config_roundtrip() {
        use ralf_gesture_studio::engine::augmentation::AugmentationConfig;

        let mut vocab = Vocabulary::new("Test Augmentation");
        vocab.augmentation = AugmentationConfig {
            enabled: true,
            multiplier: 3,
            temporal_stretch: true,
            spatial_jitter: false,
            horizontal_mirror: true,
        };
        vocab.add_gesture("wave");

        let dir = tempdir().unwrap();
        let path = dir.path().join("augmentation.ralf");
        save_vocabulary(&vocab, &path).unwrap();

        let loaded = load_vocabulary(&path).unwrap();
        assert!(loaded.augmentation.enabled);
        assert_eq!(loaded.augmentation.multiplier, 3);
        assert!(loaded.augmentation.temporal_stretch);
        assert!(!loaded.augmentation.spatial_jitter);
        assert!(loaded.augmentation.horizontal_mirror);
    }

    #[test]
    fn test_augmentation_defaults_off_for_new_vocabulary() {
        let vocab = Vocabulary::new("Test");
        assert!(!vocab.augmentation.enabled);
        assert_eq!(vocab.augmentation.multiplier, 2);
    }

    #[test]
    fn test_old_file_without_augmentation_loads_with_defaults() {
        // Simulate loading an old file without augmentation field
        let json = r#"{
            "version": "1.2",
            "name": "Legacy Vocab",
            "created_at": "2026-01-01T00:00:00Z",
            "modified_at": "2026-01-01T00:00:00Z",
            "input": {"dimensions": 66, "port": 6448, "address": "/wek/inputs"},
            "output": {"host": "127.0.0.1", "port": 12000},
            "gestures": []
        }"#;

        let vocab: Vocabulary = serde_json::from_str(json).unwrap();
        assert!(!vocab.augmentation.enabled);
        assert_eq!(vocab.augmentation.multiplier, 2);
    }

    // --- Example deletion ---

    #[test]
    fn test_remove_example_valid_index() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");
        let gesture = vocab.get_gesture_mut(id).unwrap();

        gesture.add_example(Example::new(vec![vec![1.0]], 100));
        gesture.add_example(Example::new(vec![vec![2.0]], 200));
        gesture.add_example(Example::new(vec![vec![3.0]], 300));
        assert_eq!(gesture.example_count(), 3);

        let removed = gesture.remove_example(1).unwrap();
        assert_eq!(removed.duration_ms, 200);
        assert_eq!(gesture.example_count(), 2);
    }

    #[test]
    fn test_remove_example_out_of_bounds() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");
        let gesture = vocab.get_gesture_mut(id).unwrap();
        gesture.add_example(Example::new(vec![vec![1.0]], 100));

        let result = gesture.remove_example(5);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of bounds"));
    }

    #[test]
    fn test_remove_example_clears_statistics_when_below_two() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");
        let gesture = vocab.get_gesture_mut(id).unwrap();

        gesture.add_example(Example::new(vec![vec![1.0]], 100));
        gesture.add_example(Example::new(vec![vec![2.0]], 200));

        // Set fake statistics
        gesture.update_statistics(50.0, 10.0);
        assert!(gesture.distance_mean.is_some());
        assert!(gesture.distance_std.is_some());

        // Remove one — drops to 1, statistics should be cleared
        gesture.remove_example(0).unwrap();
        assert_eq!(gesture.example_count(), 1);
        assert!(
            gesture.distance_mean.is_none(),
            "statistics should be cleared when < 2 examples"
        );
        assert!(gesture.distance_std.is_none());
    }

    #[test]
    fn test_remove_last_example() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");
        let gesture = vocab.get_gesture_mut(id).unwrap();
        gesture.add_example(Example::new(vec![vec![1.0]], 100));

        gesture.remove_example(0).unwrap();
        assert_eq!(gesture.example_count(), 0);
        assert!(!gesture.has_examples());
    }

    // --- Phase 3: Joint weighting + Consensus ---

    #[test]
    fn test_joint_weighting_defaults_off_for_new_vocabulary() {
        let vocab = Vocabulary::new("Test");
        assert!(!vocab.joint_weighting);
    }

    #[test]
    fn test_consensus_defaults_off_for_new_gesture() {
        let mut vocab = Vocabulary::new("Test");
        let id = vocab.add_gesture("wave");
        let gesture = vocab.get_gesture(id).unwrap();
        assert!(!gesture.consensus_enabled);
        assert!((gesture.consensus_threshold - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_joint_weighting_roundtrip() {
        let mut vocab = Vocabulary::new("Test Weighting");
        vocab.joint_weighting = true;
        vocab.add_gesture("wave");

        let dir = tempdir().unwrap();
        let path = dir.path().join("weighting.ralf");
        save_vocabulary(&vocab, &path).unwrap();

        let loaded = load_vocabulary(&path).unwrap();
        assert!(loaded.joint_weighting);
    }

    #[test]
    fn test_consensus_config_roundtrip() {
        let mut vocab = Vocabulary::new("Test Consensus");
        let id = vocab.add_gesture("wave");
        {
            let gesture = vocab.get_gesture_mut(id).unwrap();
            gesture.consensus_enabled = true;
            gesture.consensus_threshold = 0.75;
        }

        let dir = tempdir().unwrap();
        let path = dir.path().join("consensus.ralf");
        save_vocabulary(&vocab, &path).unwrap();

        let loaded = load_vocabulary(&path).unwrap();
        let gesture = loaded.get_gesture(id).unwrap();
        assert!(gesture.consensus_enabled);
        assert!((gesture.consensus_threshold - 0.75).abs() < f32::EPSILON);
    }

    #[test]
    fn test_old_file_without_phase3_fields_loads_with_defaults() {
        // Simulate loading a file from before Phase 3 (no joint_weighting, no consensus)
        let json = r#"{
            "version": "1.2",
            "name": "Pre-Phase3 Vocab",
            "created_at": "2026-01-01T00:00:00Z",
            "modified_at": "2026-01-01T00:00:00Z",
            "input": {"dimensions": 66, "port": 6448, "address": "/wek/inputs"},
            "output": {"host": "127.0.0.1", "port": 12000},
            "gestures": [{
                "id": 1,
                "name": "wave",
                "osc_address": "/gesture/1",
                "threshold": 100.0,
                "created_at": "2026-01-01T00:00:00Z",
                "examples": []
            }]
        }"#;

        let vocab: Vocabulary = serde_json::from_str(json).unwrap();
        assert!(!vocab.joint_weighting);

        let gesture = vocab.get_gesture(1).unwrap();
        assert!(!gesture.consensus_enabled);
        assert!((gesture.consensus_threshold - 0.5).abs() < f32::EPSILON);
    }
}
