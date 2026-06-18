//! Integration tests for the gesture recognition pipeline.
//!
//! Two test suites:
//! 1. **Synthetic** — Mathematically generated gesture sequences with known geometry.
//!    These are self-contained (no external files) and verify the pipeline is fundamentally sound.
//! 2. **Real data** — Holdout tests using a fixture `.ralf` file with actual MediaPipe recordings.
//!    These verify the pipeline works with real-world noise and human movement variation.

use ralf_gesture_studio::engine::preprocess::{Preprocessor, PreprocessingConfig};
use ralf_gesture_studio::engine::recognizer::{RecognitionConfig, Recognizer};

// ============================================================================
// Synthetic Gesture Generator
// ============================================================================

/// Number of joints in MediaPipe Pose (33 landmarks × 2 coords = 66 floats).
const DIMS: usize = 66;

/// Joint indices (MediaPipe Pose 33).
mod joint {
    pub const LEFT_SHOULDER: usize = 11;
    pub const RIGHT_SHOULDER: usize = 12;
    pub const LEFT_ELBOW: usize = 13;
    pub const RIGHT_ELBOW: usize = 14;
    pub const LEFT_WRIST: usize = 15;
    pub const RIGHT_WRIST: usize = 16;
    pub const LEFT_HIP: usize = 23;
    pub const RIGHT_HIP: usize = 24;
    pub const LEFT_KNEE: usize = 25;
    pub const RIGHT_KNEE: usize = 26;
}

/// A realistic standing pose derived from actual MediaPipe data.
/// Coordinates: index 2*j = vertical (0=top, 1=bottom), index 2*j+1 = horizontal.
fn base_pose() -> Vec<f32> {
    let mut frame = vec![0.0; DIMS];

    // Head & face (cluster near top-center)
    set_joint(&mut frame, 0, 0.20, 0.48); // nose
    set_joint(&mut frame, 1, 0.18, 0.48); // left_eye_inner
    set_joint(&mut frame, 2, 0.18, 0.49); // left_eye
    set_joint(&mut frame, 3, 0.18, 0.49); // left_eye_outer
    set_joint(&mut frame, 4, 0.18, 0.47); // right_eye_inner
    set_joint(&mut frame, 5, 0.18, 0.47); // right_eye
    set_joint(&mut frame, 6, 0.18, 0.46); // right_eye_outer
    set_joint(&mut frame, 7, 0.19, 0.49); // left_ear
    set_joint(&mut frame, 8, 0.19, 0.46); // right_ear
    set_joint(&mut frame, 9, 0.22, 0.49); // mouth_left
    set_joint(&mut frame, 10, 0.22, 0.47); // mouth_right

    // Torso
    set_joint(&mut frame, joint::LEFT_SHOULDER, 0.29, 0.52);
    set_joint(&mut frame, joint::RIGHT_SHOULDER, 0.30, 0.43);

    // Arms at sides
    set_joint(&mut frame, joint::LEFT_ELBOW, 0.43, 0.53);
    set_joint(&mut frame, joint::RIGHT_ELBOW, 0.43, 0.41);
    set_joint(&mut frame, joint::LEFT_WRIST, 0.54, 0.54);
    set_joint(&mut frame, joint::RIGHT_WRIST, 0.55, 0.41);

    // Hands (near wrists)
    set_joint(&mut frame, 17, 0.57, 0.55); // left_pinky
    set_joint(&mut frame, 18, 0.59, 0.41); // right_pinky
    set_joint(&mut frame, 19, 0.57, 0.54); // left_index
    set_joint(&mut frame, 20, 0.58, 0.42); // right_index
    set_joint(&mut frame, 21, 0.56, 0.54); // left_thumb
    set_joint(&mut frame, 22, 0.57, 0.42); // right_thumb

    // Hips
    set_joint(&mut frame, joint::LEFT_HIP, 0.55, 0.50);
    set_joint(&mut frame, joint::RIGHT_HIP, 0.55, 0.45);

    // Legs
    set_joint(&mut frame, joint::LEFT_KNEE, 0.71, 0.51);
    set_joint(&mut frame, joint::RIGHT_KNEE, 0.72, 0.45);
    set_joint(&mut frame, 27, 0.84, 0.51); // left_ankle
    set_joint(&mut frame, 28, 0.85, 0.44); // right_ankle
    set_joint(&mut frame, 29, 0.85, 0.50); // left_heel
    set_joint(&mut frame, 30, 0.86, 0.44); // right_heel
    set_joint(&mut frame, 31, 0.90, 0.52); // left_foot_index
    set_joint(&mut frame, 32, 0.92, 0.43); // right_foot_index

    frame
}

fn set_joint(frame: &mut [f32], joint_idx: usize, x: f32, y: f32) {
    frame[joint_idx * 2] = x;
    frame[joint_idx * 2 + 1] = y;
}

fn get_joint(frame: &[f32], joint_idx: usize) -> (f32, f32) {
    (frame[joint_idx * 2], frame[joint_idx * 2 + 1])
}

/// Linear interpolation between two values.
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Generate a gesture sequence by animating specific joints from start to end positions.
/// The motion follows a smooth ease-in-ease-out curve.
/// `animations`: list of (joint_index, end_x, end_y) — start positions come from base_pose.
fn generate_gesture(num_frames: usize, animations: &[(usize, f32, f32)]) -> Vec<Vec<f32>> {
    let base = base_pose();
    let mut frames = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let mut frame = base.clone();
        // Smooth ease-in-ease-out: t = 0→1→0 (gesture out and back)
        let raw_t = i as f32 / (num_frames - 1) as f32;
        // Bell curve: peak at 0.5
        let t = if raw_t < 0.5 {
            2.0 * raw_t // 0→1 in first half
        } else {
            2.0 * (1.0 - raw_t) // 1→0 in second half
        };

        for &(joint_idx, end_x, end_y) in animations {
            let (start_x, start_y) = get_joint(&base, joint_idx);
            set_joint(&mut frame, joint_idx, lerp(start_x, end_x, t), lerp(start_y, end_y, t));

            // Move connected hand joints with the wrist
            if joint_idx == joint::RIGHT_WRIST {
                let dx = lerp(start_x, end_x, t) - start_x;
                let dy = lerp(start_y, end_y, t) - start_y;
                for hand_joint in [18, 20, 22] {
                    // right_pinky, right_index, right_thumb
                    let (bx, by) = get_joint(&base, hand_joint);
                    set_joint(&mut frame, hand_joint, bx + dx, by + dy);
                }
            }
            if joint_idx == joint::LEFT_WRIST {
                let dx = lerp(start_x, end_x, t) - start_x;
                let dy = lerp(start_y, end_y, t) - start_y;
                for hand_joint in [17, 19, 21] {
                    // left_pinky, left_index, left_thumb
                    let (bx, by) = get_joint(&base, hand_joint);
                    set_joint(&mut frame, hand_joint, bx + dx, by + dy);
                }
            }
        }

        frames.push(frame);
    }
    frames
}

/// Add small random-ish noise to a sequence to simulate natural variation.
/// Uses a simple deterministic pattern (not true random) for reproducibility.
fn add_variation(frames: &[Vec<f32>], seed: u32, magnitude: f32) -> Vec<Vec<f32>> {
    frames
        .iter()
        .enumerate()
        .map(|(i, frame)| {
            frame
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    // Deterministic pseudo-noise based on frame index, dimension, and seed
                    let hash = ((i as u32).wrapping_mul(2654435761))
                        ^ ((j as u32).wrapping_mul(2246822519))
                        ^ seed.wrapping_mul(3266489917);
                    let noise = ((hash % 1000) as f32 / 500.0 - 1.0) * magnitude;
                    v + noise
                })
                .collect()
        })
        .collect()
}

/// Right arm raise: right wrist + elbow move up (x decreases = toward head).
fn right_arm_raise(num_frames: usize) -> Vec<Vec<f32>> {
    generate_gesture(
        num_frames,
        &[
            (joint::RIGHT_WRIST, 0.15, 0.43), // wrist goes up
            (joint::RIGHT_ELBOW, 0.25, 0.42), // elbow goes up
        ],
    )
}

/// Left arm raise: left wrist + elbow move up.
fn left_arm_raise(num_frames: usize) -> Vec<Vec<f32>> {
    generate_gesture(
        num_frames,
        &[
            (joint::LEFT_WRIST, 0.15, 0.53),
            (joint::LEFT_ELBOW, 0.25, 0.53),
        ],
    )
}

/// Both arms raise simultaneously.
#[allow(dead_code)]
fn both_arms_raise(num_frames: usize) -> Vec<Vec<f32>> {
    generate_gesture(
        num_frames,
        &[
            (joint::RIGHT_WRIST, 0.15, 0.43),
            (joint::RIGHT_ELBOW, 0.25, 0.42),
            (joint::LEFT_WRIST, 0.15, 0.53),
            (joint::LEFT_ELBOW, 0.25, 0.53),
        ],
    )
}

/// Standing still (no movement).
fn standing_still(num_frames: usize) -> Vec<Vec<f32>> {
    vec![base_pose(); num_frames]
}

// ============================================================================
// Test Helpers
// ============================================================================

/// Build a recognizer, train it with examples, then replay a sequence and return hits.
/// This is the core test harness — it simulates exactly what happens in the real app.
fn run_recognition(
    training_gestures: &[(&str, &[Vec<Vec<f32>>])],
    live_sequence: &[Vec<f32>],
    config: RecognitionConfig,
    preprocessing: PreprocessingConfig,
) -> Vec<String> {
    let preprocessor = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");
    let mut recognizer = Recognizer::with_config(600, 0, config);

    // Add gestures and preprocessed training examples
    for (i, (name, examples)) in training_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        let osc_address = format!("/gesture/{}", gesture_id);
        recognizer.add_gesture(gesture_id, name, &osc_address, 0.0);

        for example in *examples {
            let processed = preprocessor.process_sequence(example);
            recognizer.add_example(gesture_id, processed);
        }
    }

    // Compute thresholds from training data
    for (i, (_, examples)) in training_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        let processed: Vec<Vec<Vec<f32>>> = examples
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect();

        if let Some(stats) =
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
                &processed, 3.0, // coefficient — generous for testing
                4,    // downsample
                0.15, // sakoe_chiba_band
            )
        {
            recognizer.set_threshold(gesture_id, stats.threshold);
        }
    }

    recognizer.start();

    // Replay live sequence frame by frame through the recognizer
    // Use the SAME preprocessing config for live as for training
    let mut hits = Vec::new();
    let mut preprocessor_live = Preprocessor::new(preprocessing, "mediapipe-pose-33-xy");

    for frame in live_sequence {
        let processed = preprocessor_live.process_frame(frame);
        if let Some(result) = recognizer.process_frame(processed) {
            if let Some(name) = result.gesture_name.clone() {
                hits.push(name);
            }
        }
    }

    hits
}

/// Default config tuned for synthetic test reliability.
fn test_config() -> RecognitionConfig {
    RecognitionConfig {
        cooldown_ms: 200,
        threshold_high_factor: 1.0,
        frames_to_fire: 2,
        max_recovery_ms: 5000,
        global_cooldown_ms: 200,
        sakoe_chiba_band: 0.15,
        margin_rejection_ratio: 0.0, // disabled for tests — single-gesture scenarios
        use_subsequence_dtw: false,
        complexity_correction: false,
        slope_gate_enabled: true,
    }
}

fn no_preprocessing() -> PreprocessingConfig {
    PreprocessingConfig {
        hip_normalize: false,
        scale_normalize: false,
        velocity_features: false,
        angle_features: false,
    }
}

// ============================================================================
// Synthetic Tests
// ============================================================================

#[test]
fn synthetic_right_arm_raise_fires() {
    let training = vec![
        add_variation(&right_arm_raise(120), 1, 0.01),
        add_variation(&right_arm_raise(120), 2, 0.01),
        add_variation(&right_arm_raise(120), 3, 0.01),
        add_variation(&right_arm_raise(120), 4, 0.01),
        add_variation(&right_arm_raise(120), 5, 0.01),
    ];

    // Live performance: same gesture with different variation
    let mut live = standing_still(200); // lead-in
    live.extend(add_variation(&right_arm_raise(120), 99, 0.01));
    live.extend(standing_still(100)); // tail

    let hits = run_recognition(
        &[("right_arm_raise", &training)],
        &live,
        test_config(),
        no_preprocessing(),
    );

    assert!(
        !hits.is_empty(),
        "Right arm raise should fire at least once"
    );
    assert!(
        hits.iter().all(|h| h == "right_arm_raise"),
        "All hits should be right_arm_raise, got: {:?}",
        hits
    );
}

#[test]
fn synthetic_left_arm_raise_fires() {
    let training = vec![
        add_variation(&left_arm_raise(120), 1, 0.01),
        add_variation(&left_arm_raise(120), 2, 0.01),
        add_variation(&left_arm_raise(120), 3, 0.01),
        add_variation(&left_arm_raise(120), 4, 0.01),
        add_variation(&left_arm_raise(120), 5, 0.01),
    ];

    let mut live = standing_still(200);
    live.extend(add_variation(&left_arm_raise(120), 99, 0.01));
    live.extend(standing_still(100));

    let hits = run_recognition(
        &[("left_arm_raise", &training)],
        &live,
        test_config(),
        no_preprocessing(),
    );

    assert!(!hits.is_empty(), "Left arm raise should fire at least once");
}

#[test]
fn synthetic_no_false_positives_when_standing_still() {
    let training_right = vec![
        add_variation(&right_arm_raise(120), 1, 0.01),
        add_variation(&right_arm_raise(120), 2, 0.01),
        add_variation(&right_arm_raise(120), 3, 0.01),
        add_variation(&right_arm_raise(120), 4, 0.01),
        add_variation(&right_arm_raise(120), 5, 0.01),
    ];

    // Live: just standing still for a long time — should produce ZERO hits
    let live = standing_still(600);

    let hits = run_recognition(
        &[("right_arm_raise", &training_right)],
        &live,
        test_config(),
        no_preprocessing(),
    );

    assert!(
        hits.is_empty(),
        "Standing still should not trigger any gesture, got {} hits: {:?}",
        hits.len(),
        hits
    );
}

#[test]
fn synthetic_wrong_gesture_does_not_fire() {
    let training_right = vec![
        add_variation(&right_arm_raise(120), 1, 0.01),
        add_variation(&right_arm_raise(120), 2, 0.01),
        add_variation(&right_arm_raise(120), 3, 0.01),
        add_variation(&right_arm_raise(120), 4, 0.01),
        add_variation(&right_arm_raise(120), 5, 0.01),
    ];
    let training_left = vec![
        add_variation(&left_arm_raise(120), 1, 0.01),
        add_variation(&left_arm_raise(120), 2, 0.01),
        add_variation(&left_arm_raise(120), 3, 0.01),
        add_variation(&left_arm_raise(120), 4, 0.01),
        add_variation(&left_arm_raise(120), 5, 0.01),
    ];

    // Live: perform RIGHT arm raise — should NOT fire left_arm_raise
    let mut live = standing_still(200);
    live.extend(add_variation(&right_arm_raise(120), 99, 0.01));
    live.extend(standing_still(100));

    let hits = run_recognition(
        &[
            ("right_arm_raise", &training_right),
            ("left_arm_raise", &training_left),
        ],
        &live,
        test_config(),
        no_preprocessing(),
    );

    let wrong_hits: Vec<_> = hits.iter().filter(|h| h.as_str() == "left_arm_raise").collect();
    assert!(
        wrong_hits.is_empty(),
        "Left arm raise should NOT fire when performing right arm raise, got {} wrong hits",
        wrong_hits.len()
    );
}

#[test]
fn synthetic_multiple_gestures_in_sequence() {
    let training_right = vec![
        add_variation(&right_arm_raise(120), 1, 0.01),
        add_variation(&right_arm_raise(120), 2, 0.01),
        add_variation(&right_arm_raise(120), 3, 0.01),
        add_variation(&right_arm_raise(120), 4, 0.01),
        add_variation(&right_arm_raise(120), 5, 0.01),
    ];
    let training_left = vec![
        add_variation(&left_arm_raise(120), 1, 0.01),
        add_variation(&left_arm_raise(120), 2, 0.01),
        add_variation(&left_arm_raise(120), 3, 0.01),
        add_variation(&left_arm_raise(120), 4, 0.01),
        add_variation(&left_arm_raise(120), 5, 0.01),
    ];

    // Live: right arm raise → pause → left arm raise
    let mut live = standing_still(200);
    live.extend(add_variation(&right_arm_raise(120), 10, 0.01));
    live.extend(standing_still(200)); // pause between gestures
    live.extend(add_variation(&left_arm_raise(120), 20, 0.01));
    live.extend(standing_still(100));

    let hits = run_recognition(
        &[
            ("right_arm_raise", &training_right),
            ("left_arm_raise", &training_left),
        ],
        &live,
        test_config(),
        no_preprocessing(),
    );

    let right_hits = hits.iter().filter(|h| h.as_str() == "right_arm_raise").count();
    let left_hits = hits.iter().filter(|h| h.as_str() == "left_arm_raise").count();

    assert!(right_hits >= 1, "Should detect right arm raise");
    assert!(left_hits >= 1, "Should detect left arm raise");
}

// ============================================================================
// Real Data Tests (fixture-based)
// ============================================================================

/// Load the test fixture and run holdout recognition.
/// For each gesture, trains on examples 0..N-1 and replays example N.
#[test]
fn real_data_holdout_fires() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/arm_raises.ralf"
    );
    let content = std::fs::read_to_string(fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read fixture {}: {}", fixture_path, e));
    let vocab: serde_json::Value = serde_json::from_str(&content).expect("Invalid JSON");

    let gestures = vocab["gestures"].as_array().expect("gestures array");

    for gesture_val in gestures {
        let name = gesture_val["name"].as_str().unwrap();
        let examples: Vec<Vec<Vec<f32>>> = gesture_val["examples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|ex| {
                ex["frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|frame| {
                        frame
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                            .collect()
                    })
                    .collect()
            })
            .collect();

        assert!(
            examples.len() >= 3,
            "Need at least 3 examples for holdout test, gesture '{}' has {}",
            name,
            examples.len()
        );

        // Train on all but last, replay last
        let (training, holdout) = examples.split_at(examples.len() - 1);
        let holdout_seq = &holdout[0];

        // Build live sequence: standing still lead-in → holdout → standing still tail
        let base = holdout_seq[0].clone(); // use first frame as "standing"
        let mut live: Vec<Vec<f32>> = vec![base; 200]; // lead-in
        live.extend(holdout_seq.iter().cloned());
        live.extend(vec![holdout_seq.last().unwrap().clone(); 100]); // tail

        let training_refs: Vec<Vec<Vec<f32>>> = training.to_vec();
        let training_slices: Vec<&[Vec<f32>]> =
            training_refs.iter().map(|e| e.as_slice()).collect();

        // Build recognizer manually for this single gesture
        let preprocessor = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
        let mut recognizer = Recognizer::with_config(600, 0, test_config());

        recognizer.add_gesture(1, name, "/gesture/1", 0.0);

        let processed_examples: Vec<Vec<Vec<f32>>> = training_slices
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect();

        for example in &processed_examples {
            recognizer.add_example(1, example.clone());
        }

        // Compute threshold
        if let Some(stats) =
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
                &processed_examples,
                3.0,
                4,
                0.15,
            )
        {
            recognizer.set_threshold(1, stats.threshold);
        }

        recognizer.start();

        // Replay
        let mut preprocessor_live = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
        let mut hits = Vec::new();
        for frame in &live {
            let processed = preprocessor_live.process_frame(frame);
            if let Some(result) = recognizer.process_frame(processed) {
                if let Some(name) = result.gesture_name.clone() {
                hits.push(name);
            }
            }
        }

        assert!(
            !hits.is_empty(),
            "Real data holdout test: gesture '{}' should fire when replaying a held-out example. \
             Training examples: {}, holdout frames: {}",
            name,
            training.len(),
            holdout_seq.len()
        );
    }
}

/// Verify that replaying one gesture's example doesn't trigger a different gesture.
#[test]
fn real_data_no_cross_trigger() {
    let fixture_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/arm_raises.ralf"
    );
    let content = std::fs::read_to_string(fixture_path).expect("Failed to read fixture");
    let vocab: serde_json::Value = serde_json::from_str(&content).expect("Invalid JSON");

    let gestures = vocab["gestures"].as_array().expect("gestures array");
    assert!(
        gestures.len() >= 2,
        "Need at least 2 gestures for cross-trigger test"
    );

    // Parse all gestures
    let mut all_gestures: Vec<(String, Vec<Vec<Vec<f32>>>)> = Vec::new();
    for gesture_val in gestures {
        let name = gesture_val["name"].as_str().unwrap().to_string();
        let examples: Vec<Vec<Vec<f32>>> = gesture_val["examples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|ex| {
                ex["frames"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|frame| {
                        frame
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_f64().unwrap() as f32)
                            .collect()
                    })
                    .collect()
            })
            .collect();
        all_gestures.push((name, examples));
    }

    let preprocessor = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
    let mut recognizer = Recognizer::with_config(600, 0, test_config());

    // Train all gestures (using all but last example each)
    for (i, (name, examples)) in all_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        recognizer.add_gesture(gesture_id, name, &format!("/gesture/{}", gesture_id), 0.0);

        let training = &examples[..examples.len() - 1];
        let processed: Vec<Vec<Vec<f32>>> = training
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect();

        for example in &processed {
            recognizer.add_example(gesture_id, example.clone());
        }

        if let Some(stats) =
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
                &processed, 3.0, 4, 0.15,
            )
        {
            recognizer.set_threshold(gesture_id, stats.threshold);
        }
    }

    recognizer.start();

    // Replay each gesture's holdout and check no OTHER gesture fires
    for (_gesture_idx, (name, examples)) in all_gestures.iter().enumerate() {
        let holdout = examples.last().unwrap();
        let base = holdout[0].clone();

        let mut live: Vec<Vec<f32>> = vec![base; 200];
        live.extend(holdout.iter().cloned());
        live.extend(vec![holdout.last().unwrap().clone(); 100]);

        // Fresh recognizer state for each replay
        let mut recognizer_copy = Recognizer::with_config(600, 0, test_config());
        for (i, (gname, examples)) in all_gestures.iter().enumerate() {
            let gid = (i + 1) as u32;
            recognizer_copy.add_gesture(gid, gname, &format!("/gesture/{}", gid), 0.0);
            let training = &examples[..examples.len() - 1];
            let processed: Vec<Vec<Vec<f32>>> = training
                .iter()
                .map(|e| preprocessor.process_sequence(e))
                .collect();
            for ex in &processed {
                recognizer_copy.add_example(gid, ex.clone());
            }
            if let Some(stats) =
                ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
                    &processed, 3.0, 4, 0.15,
                )
            {
                recognizer_copy.set_threshold(gid, stats.threshold);
            }
        }
        recognizer_copy.start();

        let mut preprocessor_live = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
        let mut wrong_hits = Vec::new();
        for frame in &live {
            let processed = preprocessor_live.process_frame(frame);
            if let Some(result) = recognizer_copy.process_frame(processed) {
                if let Some(ref hit_name) = result.gesture_name {
                    if hit_name != name {
                        wrong_hits.push(hit_name.clone());
                    }
                }
            }
        }

        assert!(
            wrong_hits.is_empty(),
            "Replaying '{}' triggered other gestures: {:?}",
            name,
            wrong_hits
        );
    }
}

// ============================================================================
// Benchmark: Performance Criteria
// ============================================================================
//
// Leave-one-out cross-validation on the benchmark vocabulary.
// For each gesture, each example takes a turn as the "live performance"
// while the remaining N-1 examples serve as training data.
//
// Targets:
//   - Hit rate >= 90% (correct gesture fires)
//   - False positive rate <= 5% (wrong gesture fires)
//   - Detection latency < 800ms (~12 frames at 15Hz DTW)

/// Parsed fixture: gestures + preprocessing config from the vocabulary.
struct Fixture {
    gestures: Vec<(String, Vec<Vec<Vec<f32>>>)>,
    preprocessing: PreprocessingConfig,
}

/// Parse a .ralf fixture into gestures and its preprocessing config.
fn load_fixture(filename: &str) -> Fixture {
    let fixture_path = format!("{}/tests/fixtures/{}", env!("CARGO_MANIFEST_DIR"), filename);
    let content = std::fs::read_to_string(&fixture_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", fixture_path, e));
    let vocab: serde_json::Value = serde_json::from_str(&content).expect("Invalid JSON");

    let preprocessing = if let Some(pp) = vocab.get("preprocessing") {
        PreprocessingConfig {
            hip_normalize: pp["hip_normalize"].as_bool().unwrap_or(false),
            scale_normalize: pp["scale_normalize"].as_bool().unwrap_or(false),
            velocity_features: pp["velocity_features"].as_bool().unwrap_or(false),
            angle_features: pp["angle_features"].as_bool().unwrap_or(false),
        }
    } else {
        no_preprocessing()
    };

    let gestures = vocab["gestures"]
        .as_array()
        .expect("gestures array")
        .iter()
        .map(|g| {
            let name = g["name"].as_str().unwrap().to_string();
            let examples: Vec<Vec<Vec<f32>>> = g["examples"]
                .as_array()
                .unwrap()
                .iter()
                .map(|ex| {
                    ex["frames"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|frame| {
                            frame
                                .as_array()
                                .unwrap()
                                .iter()
                                .map(|v| v.as_f64().unwrap() as f32)
                                .collect()
                        })
                        .collect()
                })
                .collect();
            (name, examples)
        })
        .collect();

    Fixture {
        gestures,
        preprocessing,
    }
}

/// Recognition config for benchmark — matches app defaults but with
/// margin rejection disabled. The benchmark measures raw recognition
/// quality; margin rejection is a separate layer tested independently.
fn benchmark_config() -> RecognitionConfig {
    RecognitionConfig {
        cooldown_ms: 200,
        threshold_high_factor: 1.0,
        frames_to_fire: 2,
        max_recovery_ms: 5000,
        global_cooldown_ms: 200,
        sakoe_chiba_band: 0.15,
        margin_rejection_ratio: 0.0,
        use_subsequence_dtw: true, // sDTW with wavefront banding
        complexity_correction: false,
        slope_gate_enabled: true,
    }
}

/// Run leave-one-out recognition for one holdout example.
/// Returns (correct_hit: bool, false_positive: Option<String>).
fn run_holdout_trial(
    all_gestures: &[(String, Vec<Vec<Vec<f32>>>)],
    preprocessing: &PreprocessingConfig,
    target_gesture_idx: usize,
    holdout_idx: usize,
) -> (bool, Option<String>) {
    let preprocessor = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");
    let config = benchmark_config();
    let use_sdtw = config.use_subsequence_dtw;
    let mut recognizer = Recognizer::with_config(600, 0, config);

    // Train all gestures, excluding the holdout example from the target gesture
    for (i, (name, examples)) in all_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        recognizer.add_gesture(gesture_id, name, &format!("/gesture/{}", gesture_id), 0.0);

        let training: Vec<&Vec<Vec<f32>>> = if i == target_gesture_idx {
            examples
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != holdout_idx)
                .map(|(_, e)| e)
                .collect()
        } else {
            examples.iter().collect()
        };

        let processed: Vec<Vec<Vec<f32>>> = training
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect();

        for example in &processed {
            recognizer.add_example(gesture_id, example.clone());
        }

        let stats = if use_sdtw {
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_sdtw(
                &processed, 3.0, 4, 0.15,
            )
        } else {
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
                &processed, 3.0, 4, 0.15,
            )
        };
        if let Some(stats) = stats {
            recognizer.set_threshold(gesture_id, stats.threshold);
        }
    }

    recognizer.start();

    // Build live sequence: standing lead-in → holdout → standing tail
    let holdout = &all_gestures[target_gesture_idx].1[holdout_idx];
    let standing_frame = holdout[0].clone();
    let mut live: Vec<Vec<f32>> = vec![standing_frame.clone(); 200];
    live.extend(holdout.iter().cloned());
    live.extend(vec![standing_frame; 100]);

    // Replay and track per-gesture minimum distances
    let mut preprocessor_live = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");
    let target_name = &all_gestures[target_gesture_idx].0;
    let mut correct_hit = false;
    let mut false_positive: Option<String> = None;

    for frame in &live {
        let processed = preprocessor_live.process_frame(frame);
        if let Some(result) = recognizer.process_frame(processed) {
            if let Some(ref hit_name) = result.gesture_name {
                if hit_name == target_name {
                    correct_hit = true;
                } else {
                    false_positive = Some(hit_name.clone());
                }
            }
        }
    }

    (correct_hit, false_positive)
}

#[test]
fn benchmark_hit_rate() {
    let fixture = load_fixture("benchmark.ralf");
    let mut total_trials = 0;
    let mut correct_hits = 0;
    let mut false_positives = 0;
    let mut misses: Vec<String> = Vec::new();
    let mut fp_details: Vec<String> = Vec::new();

    // Override: test with no preprocessing
    let preprocessing = no_preprocessing();

    for (gesture_idx, (name, examples)) in fixture.gestures.iter().enumerate() {
        for holdout_idx in 0..examples.len() {
            total_trials += 1;
            let (hit, fp) = run_holdout_trial(
                &fixture.gestures,
                &preprocessing,
                gesture_idx,
                holdout_idx,
            );

            if hit {
                correct_hits += 1;
            } else {
                misses.push(format!("{}[{}]", name, holdout_idx));
            }

            if let Some(wrong_name) = fp {
                false_positives += 1;
                fp_details.push(format!(
                    "{}[{}] triggered '{}'",
                    name, holdout_idx, wrong_name
                ));
            }
        }
    }

    let hit_rate = correct_hits as f64 / total_trials as f64;
    let fp_rate = false_positives as f64 / total_trials as f64;

    eprintln!("\n=== Benchmark Results ===");
    eprintln!(
        "Hit rate:  {}/{} ({:.1}%)",
        correct_hits,
        total_trials,
        hit_rate * 100.0
    );
    eprintln!(
        "FP rate:   {}/{} ({:.1}%)",
        false_positives,
        total_trials,
        fp_rate * 100.0
    );
    if !misses.is_empty() {
        eprintln!("Misses:    {:?}", misses);
    }
    if !fp_details.is_empty() {
        eprintln!("FPs:       {:?}", fp_details);
    }
    eprintln!("=========================\n");

    assert!(
        hit_rate >= 0.90,
        "Hit rate {:.1}% is below 90% target. Misses: {:?}",
        hit_rate * 100.0,
        misses
    );
    assert!(
        fp_rate <= 0.05,
        "False positive rate {:.1}% exceeds 5% target. Details: {:?}",
        fp_rate * 100.0,
        fp_details
    );
}

/// Diagnostic: print raw DTW distances for each holdout to understand confusion patterns.
#[test]
#[ignore] // Run with: cargo test diag_distances -- --ignored --nocapture
fn diag_distances() {
    let fixture = load_fixture("benchmark.ralf");
    let preprocessing = no_preprocessing();
    let preprocessor = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");

    let mut gesture_templates: Vec<(String, Vec<Vec<Vec<f32>>>)> = Vec::new();
    for (name, examples) in &fixture.gestures {
        let processed: Vec<Vec<Vec<f32>>> = examples
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect();
        gesture_templates.push((name.clone(), processed));
    }

    for (target_idx, (target_name, target_examples)) in gesture_templates.iter().enumerate() {
        eprintln!("\n--- {} holdout distances (threshold from file) ---", target_name);
        for holdout_idx in 0..target_examples.len() {
            let holdout = &target_examples[holdout_idx];
            eprint!("  [{}]:", holdout_idx);
            for (gi, (gname, gexamples)) in gesture_templates.iter().enumerate() {
                let templates: Vec<&Vec<Vec<f32>>> = if gi == target_idx {
                    gexamples
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != holdout_idx)
                        .map(|(_, e)| e)
                        .collect()
                } else {
                    gexamples.iter().collect()
                };

                let mut min_dist = f32::MAX;
                for tmpl in &templates {
                    let band =
                        ((holdout.len().max(tmpl.len()) as f32) * 0.15).ceil() as usize;
                    if let Some(d) =
                        ralf_gesture_studio::engine::dtw::dtw_distance_with_abandon(
                            holdout, tmpl, band, f32::INFINITY,
                        )
                    {
                        if d < min_dist {
                            min_dist = d;
                        }
                    }
                }
                eprint!("  {}={:.1}", gname, min_dist);
            }
            eprintln!();
        }
    }
}

// ============================================================================
// RC-1: F1-swept threshold reaches recognition (regression guard)
// ============================================================================

/// End-to-end guard for RC-1: the F1 sweep must (a) run when negatives are
/// present, (b) produce a threshold distinct from μ+σ for separable classes,
/// and (c) survive routing through the production selection method. Before the
/// fix, `compute_gesture_statistics` discarded the swept value by recomputing
/// μ+σ, leaving the F1 feature inert even when enabled.
/// See docs/research/2026-06-18-four-way-tension-hardening-analysis.md.
#[test]
fn rc1_f1_swept_threshold_separates_and_reaches_gesture() {
    use ralf_gesture_studio::engine::statistics::compute_threshold_f1;
    use ralf_gesture_studio::model::Gesture;

    // A short sequence shaped like a ramp, offset to place it in feature space.
    let make = |offset: f32, jitter: f32| -> Vec<Vec<f32>> {
        (0..10)
            .map(|t| {
                let base = offset + t as f32 * 0.1;
                vec![base + jitter, base - jitter, base]
            })
            .collect()
    };
    // Tight positive cluster, and a negative cluster shifted far away.
    let positives = vec![
        make(0.0, 0.00),
        make(0.0, 0.01),
        make(0.0, 0.02),
        make(0.0, 0.015),
    ];
    let negatives = vec![
        make(10.0, 0.00),
        make(10.0, 0.01),
        make(10.0, 0.02),
        make(10.0, 0.015),
    ];

    let stats = compute_threshold_f1(&positives, &negatives, 1, 0.15, 2.0)
        .expect("F1 calibration should produce stats for separable classes");

    assert!(
        stats.f1_score.is_some(),
        "F1 sweep must run (f1_score present) when negatives are supplied"
    );
    let mu_sigma = stats.mean + stats.std * 2.0;
    assert!(
        (stats.threshold - mu_sigma).abs() > 1e-3,
        "swept F1 threshold ({}) should differ from μ+σ ({}) for separable classes",
        stats.threshold,
        mu_sigma
    );

    // Route through the production selection path and confirm the swept value lands.
    let mut g = Gesture::new(1, "test");
    g.threshold_coefficient = 2.0;
    g.apply_threshold_stats(stats.mean, stats.std, stats.threshold, stats.f1_score);
    assert_eq!(
        g.threshold, stats.threshold,
        "recognition threshold must equal the swept F1 value, not μ+σ"
    );
}

// ============================================================================
// RC-2: Per-gesture F1-on vs F1-off holdout harness
//
// The aggregate benchmark (benchmark_hit_rate) reports a single 93.3%/3.3%
// number that masks inter-class "stealing". This harness runs leave-one-out
// twice — μ+σ thresholds vs F1-with-negatives thresholds — and reports the
// per-gesture hit rate, false-positive count, chosen threshold, and best F1,
// at BOTH coefficient 2.0 (production) and 3.0 (benchmark default), because
// the F1 threshold-delta direction can flip between them.
//
// It answers, with a measured number, whether the open false-negative corner
// is a THRESHOLD problem (F1 fixes it) or a DATA problem (F1 cannot). A low
// per-gesture best-F1 is the direct "data problem" signal.
//
// CAVEAT (do not skip): a green result here does NOT license enabling F1 for a
// dancer. This runs with no preprocessing on a fixed vocabulary; production
// runs preprocessing on, at coefficient 2.0, on different data. Treat these as
// a LOWER BOUND; production go/no-go is rehearsal-gated.
//
// Marked #[ignore] because it runs leave-one-out four times (~minutes). Run on
// demand:  cargo test --test recognition_integration -- --ignored rc2
// See docs/research/2026-06-18-four-way-tension-hardening-analysis.md (RC-2).
// ============================================================================

/// One leave-one-out trial under a chosen threshold strategy.
/// Returns (correct_hit, false_positive_name, target_threshold, target_best_f1).
fn run_holdout_trial_param(
    all_gestures: &[(String, Vec<Vec<Vec<f32>>>)],
    preprocessing: &PreprocessingConfig,
    target_gesture_idx: usize,
    holdout_idx: usize,
    coefficient: f32,
    use_f1: bool,
) -> (bool, Option<String>, f32, Option<f32>) {
    use ralf_gesture_studio::engine::statistics::{
        compute_threshold_f1, compute_threshold_stats_sdtw,
    };
    let preprocessor = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");
    let mut recognizer = Recognizer::with_config(600, 0, benchmark_config());

    let mut target_threshold = 0.0_f32;
    let mut target_best_f1 = None;

    for (i, (name, examples)) in all_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        recognizer.add_gesture(gesture_id, name, &format!("/gesture/{}", gesture_id), 0.0);

        // Training set: drop the holdout example only from the target gesture.
        let training: Vec<Vec<Vec<f32>>> = examples
            .iter()
            .enumerate()
            .filter(|(j, _)| !(i == target_gesture_idx && *j == holdout_idx))
            .map(|(_, e)| preprocessor.process_sequence(e))
            .collect();

        for example in &training {
            recognizer.add_example(gesture_id, example.clone());
        }

        // Threshold strategy.
        let (threshold, best_f1) = if use_f1 {
            // Negatives = every OTHER gesture's full example set.
            let negatives: Vec<Vec<Vec<f32>>> = all_gestures
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != i)
                .flat_map(|(_, (_, exs))| exs.iter().map(|e| preprocessor.process_sequence(e)))
                .collect();
            match compute_threshold_f1(&training, &negatives, 4, 0.15, coefficient) {
                Some(s) => (s.threshold, s.f1_score),
                None => (0.0, None),
            }
        } else {
            match compute_threshold_stats_sdtw(&training, coefficient, 4, 0.15) {
                Some(s) => (s.threshold, None),
                None => (0.0, None),
            }
        };
        recognizer.set_threshold(gesture_id, threshold);
        if i == target_gesture_idx {
            target_threshold = threshold;
            target_best_f1 = best_f1;
        }
    }

    recognizer.start();

    let holdout = &all_gestures[target_gesture_idx].1[holdout_idx];
    let standing_frame = holdout[0].clone();
    let mut live: Vec<Vec<f32>> = vec![standing_frame.clone(); 200];
    live.extend(holdout.iter().cloned());
    live.extend(vec![standing_frame; 100]);

    let mut preprocessor_live = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");
    let target_name = &all_gestures[target_gesture_idx].0;
    let mut correct_hit = false;
    let mut false_positive = None;
    for frame in &live {
        let processed = preprocessor_live.process_frame(frame);
        if let Some(result) = recognizer.process_frame(processed) {
            if let Some(ref hit_name) = result.gesture_name {
                if hit_name == target_name {
                    correct_hit = true;
                } else {
                    false_positive = Some(hit_name.clone());
                }
            }
        }
    }
    (correct_hit, false_positive, target_threshold, target_best_f1)
}

#[derive(Default, Clone)]
struct GestureScore {
    trials: usize,
    hits: usize,
    false_positives: usize,
    threshold_sum: f32,
    f1_sum: f32,
    f1_count: usize,
}

#[test]
#[ignore = "measurement harness: leave-one-out x4, run with --ignored"]
fn rc2_f1_vs_musigma_holdout() {
    let fixture = load_fixture("jump-wave-spin-test.ralf");
    let preprocessing = no_preprocessing();

    println!("\n=== RC-2: F1-on vs F1-off per-gesture holdout ===");
    println!("fixture: jump-wave-spin-test.ralf, {} gestures\n", fixture.gestures.len());

    let mut any_f1_score = false;

    for &coefficient in &[2.0_f32, 3.0_f32] {
        for &use_f1 in &[false, true] {
            let mut scores: Vec<GestureScore> =
                vec![GestureScore::default(); fixture.gestures.len()];

            for (g_idx, (_, examples)) in fixture.gestures.iter().enumerate() {
                for holdout_idx in 0..examples.len() {
                    let (hit, fp, threshold, f1) = run_holdout_trial_param(
                        &fixture.gestures,
                        &preprocessing,
                        g_idx,
                        holdout_idx,
                        coefficient,
                        use_f1,
                    );
                    let s = &mut scores[g_idx];
                    s.trials += 1;
                    if hit {
                        s.hits += 1;
                    }
                    if fp.is_some() {
                        s.false_positives += 1;
                    }
                    s.threshold_sum += threshold;
                    if let Some(f) = f1 {
                        s.f1_sum += f;
                        s.f1_count += 1;
                        any_f1_score = true;
                    }
                }
            }

            let label = if use_f1 { "F1-with-negatives" } else { "mu+sigma" };
            println!("--- coefficient {:.1}, strategy {} ---", coefficient, label);
            println!(
                "{:<14} {:>8} {:>6} {:>10} {:>9}",
                "gesture", "hit-rate", "FPs", "threshold", "best-F1"
            );
            for (g_idx, (name, _)) in fixture.gestures.iter().enumerate() {
                let s = &scores[g_idx];
                let hit_rate = s.hits as f32 / s.trials.max(1) as f32 * 100.0;
                let avg_thresh = s.threshold_sum / s.trials.max(1) as f32;
                let avg_f1 = if s.f1_count > 0 {
                    format!("{:.3}", s.f1_sum / s.f1_count as f32)
                } else {
                    "-".to_string()
                };
                println!(
                    "{:<14} {:>7.1}% {:>6} {:>10.1} {:>9}",
                    name, hit_rate, s.false_positives, avg_thresh, avg_f1
                );
            }
            println!();
        }
    }

    // Sanity: the F1 path actually produced F1 scores (i.e. it ran the sweep,
    // not the no-negatives fallback). The numbers above are the deliverable.
    assert!(
        any_f1_score,
        "F1 strategy should have produced at least one best-F1 score"
    );
}

// ============================================================================
// RC-4: Executable echo / false-positive regression guards
//
// These lock in three invariants that are otherwise enforced only by
// convention and by their absence in production logs. See
// docs/research/2026-06-18-four-way-tension-hardening-analysis.md (RC-4).
//
// Design note (verified subtlety): this harness replays frames in a tight,
// no-sleep loop, but global cooldown and Recovery dwell are WALL-CLOCK
// (Instant::elapsed). So the buffer-clear-on-hit plus window refill
// (frame-domain) is the only echo gate the tight loop reliably exercises.
// The same-gesture guard is scoped to that gate; the wall-clock cooldown is
// exercised separately with a real sleep.
// ============================================================================

/// Like `run_recognition`, but returns (gesture_name, distance, threshold) for
/// every hit so invariants about the firing distance can be asserted.
fn run_recognition_detailed(
    training_gestures: &[(&str, &[Vec<Vec<f32>>])],
    live_sequence: &[Vec<f32>],
    config: RecognitionConfig,
    preprocessing: PreprocessingConfig,
) -> Vec<(String, f32, f32)> {
    let preprocessor = Preprocessor::new(preprocessing.clone(), "mediapipe-pose-33-xy");
    let mut recognizer = Recognizer::with_config(600, 0, config);

    for (i, (name, examples)) in training_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        let osc_address = format!("/gesture/{}", gesture_id);
        recognizer.add_gesture(gesture_id, name, &osc_address, 0.0);
        for example in *examples {
            let processed = preprocessor.process_sequence(example);
            recognizer.add_example(gesture_id, processed);
        }
    }
    for (i, (_, examples)) in training_gestures.iter().enumerate() {
        let gesture_id = (i + 1) as u32;
        let processed: Vec<Vec<Vec<f32>>> = examples
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect();
        if let Some(stats) =
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
                &processed, 3.0, 4, 0.15,
            )
        {
            recognizer.set_threshold(gesture_id, stats.threshold);
        }
    }

    recognizer.start();

    let mut hits = Vec::new();
    let mut preprocessor_live = Preprocessor::new(preprocessing, "mediapipe-pose-33-xy");
    for frame in live_sequence {
        let processed = preprocessor_live.process_frame(frame);
        if let Some(result) = recognizer.process_frame(processed) {
            if let (Some(id), Some(name)) = (result.gesture_id, result.gesture_name.clone()) {
                let threshold = recognizer.get_gesture(id).map(|g| g.threshold).unwrap_or(0.0);
                hits.push((name, result.distance, threshold));
            }
        }
    }
    hits
}

/// A config with both wall-clock echo gates disabled, so the only thing that
/// can prevent a same-gesture re-fire is the frame-domain buffer clear.
fn echo_guard_config(cooldown_ms: u64, global_cooldown_ms: u64) -> RecognitionConfig {
    RecognitionConfig {
        cooldown_ms,
        threshold_high_factor: 1.0,
        frames_to_fire: 2,
        max_recovery_ms: 5000,
        global_cooldown_ms,
        sakoe_chiba_band: 0.15,
        margin_rejection_ratio: 0.0,
        use_subsequence_dtw: false,
        complexity_correction: false,
        slope_gate_enabled: true,
    }
}

/// RC-4 guard #1: a hit may NEVER fire while distance is above threshold.
/// The margin (threshold - distance) must be >= 0. This is a margin>=0 check,
/// NOT a positive floor: the analysis notes a legitimate +1.8% warm-up hit can
/// sit right at the threshold, so equality is allowed.
#[test]
fn rc4_no_negative_margin_fire() {
    let training: Vec<Vec<Vec<f32>>> = (1..=5)
        .map(|s| add_variation(&right_arm_raise(120), s, 0.01))
        .collect();

    let mut live = standing_still(200);
    live.extend(add_variation(&right_arm_raise(120), 99, 0.01));
    live.extend(standing_still(100));

    let hits = run_recognition_detailed(
        &[("right_arm_raise", &training)],
        &live,
        test_config(),
        no_preprocessing(),
    );

    assert!(!hits.is_empty(), "gesture should fire at least once");
    for (name, distance, threshold) in &hits {
        let margin = threshold - distance;
        assert!(
            margin >= 0.0,
            "{} fired with NEGATIVE margin: distance {:.2} > threshold {:.2} (margin {:.2}). \
             A hit must never fire above threshold.",
            name,
            distance,
            threshold,
            margin
        );
    }
}

/// RC-4 guard #2: a single gesture instance fires exactly once, even with BOTH
/// wall-clock echo gates disabled (`cooldown_ms=0`, `global_cooldown_ms=0`).
///
/// A gesture stays below threshold for many consecutive frames as the sliding
/// window passes over the complete match. With both timer-based cooldowns off,
/// the ONLY thing preventing it from firing on every one of those frames is the
/// frame-domain buffer clear: on the hit, the buffer empties and must refill
/// (window_size frames) before any new detection, by which point the gesture
/// has slid out of view. So if `buffer.clear()` were removed, this single
/// instance would produce many hits. Exactly one hit, plus a direct check that
/// the buffer is empty on the hit frame, proves the clear is the active gate.
/// This is the frame-domain echo defense the tight no-sleep loop can exercise.
#[test]
fn rc4_no_same_gesture_refire_via_buffer_clear() {
    let preproc = no_preprocessing();
    let preprocessor = Preprocessor::new(preproc.clone(), "mediapipe-pose-33-xy");
    let mut recognizer = Recognizer::with_config(600, 0, echo_guard_config(0, 0));
    recognizer.add_gesture(1, "right_arm_raise", "/gesture/1", 0.0);

    let training: Vec<Vec<Vec<f32>>> = (1..=5)
        .map(|s| add_variation(&right_arm_raise(120), s, 0.01))
        .collect();
    for ex in &training {
        recognizer.add_example(1, preprocessor.process_sequence(ex));
    }
    if let Some(stats) = ralf_gesture_studio::engine::statistics::compute_threshold_stats_banded(
        &training
            .iter()
            .map(|e| preprocessor.process_sequence(e))
            .collect::<Vec<_>>(),
        3.0,
        4,
        0.15,
    ) {
        recognizer.set_threshold(1, stats.threshold);
    }
    recognizer.start();

    let mut live = standing_still(200);
    live.extend(add_variation(&right_arm_raise(120), 99, 0.01));
    live.extend(standing_still(150));

    let window = recognizer.window_size();
    let mut hit_count = 0usize;
    let mut buffer_len_at_hit = usize::MAX;
    let mut preprocessor_live = Preprocessor::new(preproc, "mediapipe-pose-33-xy");
    for frame in &live {
        let processed = preprocessor_live.process_frame(frame);
        if let Some(result) = recognizer.process_frame(processed) {
            if result.gesture_id.is_some() {
                hit_count += 1;
                buffer_len_at_hit = recognizer.buffer.len();
            }
        }
    }

    assert_eq!(
        hit_count, 1,
        "expected exactly one hit with all wall-clock cooldowns disabled; got {}. \
         More than one means the buffer-clear echo defense regressed.",
        hit_count
    );
    // Direct evidence the clear ran: the buffer is emptied on the hit frame,
    // far below a full window.
    assert!(
        buffer_len_at_hit < window,
        "buffer should be cleared on the hit (len {} < window {}); the clear is the echo gate",
        buffer_len_at_hit,
        window
    );
}

/// RC-4 guard #3 (wall-clock, labeled): the per-gesture cooldown is real time,
/// not frames. With a real sleep shorter than the cooldown a re-fire is still
/// blocked; after sleeping past it, the gesture re-arms. This exercises the
/// wall-clock gate that the tight no-sleep loop cannot.
#[test]
fn rc4_per_gesture_cooldown_is_wall_clock() {
    use ralf_gesture_studio::engine::recognizer::GestureState;
    use std::time::Duration;

    let cooldown = Duration::from_millis(200);
    let mut g = GestureState::new(1, "wave", "/gesture/1", 100.0);

    // Brand-new gesture is not in cooldown.
    assert!(!g.in_cooldown(cooldown), "fresh gesture should not be in cooldown");

    // After a hit it is in cooldown immediately, and stays so until the
    // wall-clock window elapses — this is real time, not frames.
    g.record_hit();
    assert!(
        g.in_cooldown(cooldown),
        "should be in cooldown immediately after a hit"
    );
    std::thread::sleep(Duration::from_millis(220));
    assert!(
        !g.in_cooldown(cooldown),
        "cooldown must expire after the wall-clock window passes"
    );
}

// ============================================================================
// Issue #25, Step 0: variable-length window probe (measure the penalty first)
//
// window_size is pinned to the FIRST trained example's length. A gesture longer
// than the window is truncated to its last window_size frames before any DTW.
// The analysis asserts this causes a false-negative penalty for a long gesture
// in a short-seeded window, but notes the penalty is "asserted, not yet
// measured" (trim_to_onset + sDTW free-start may absorb much of it). This probe
// MEASURES it: same vocabulary, window seeded short vs long, report the delta.
// Per the gating decision, shipping Option A (monotonic-max window) waits on
// this number. See docs/research/2026-06-18-four-way-tension-hardening-analysis.md.
// ============================================================================

/// Build a recognizer whose window is seeded by the first example added, train
/// both a short and a long gesture, and return it ready to run. `seed_long_first`
/// chooses which gesture's example sets `window_size`.
fn build_varlen_recognizer(
    short_examples: &[Vec<Vec<f32>>],
    long_examples: &[Vec<Vec<f32>>],
    seed_long_first: bool,
) -> Recognizer {
    let prep = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
    let mut recognizer = Recognizer::with_config(2048, 0, benchmark_config());
    recognizer.add_gesture(1, "short", "/gesture/1", 0.0);
    recognizer.add_gesture(2, "long", "/gesture/2", 0.0);

    let short_p: Vec<Vec<Vec<f32>>> = short_examples
        .iter()
        .map(|e| prep.process_sequence(e))
        .collect();
    let long_p: Vec<Vec<Vec<f32>>> = long_examples
        .iter()
        .map(|e| prep.process_sequence(e))
        .collect();

    // The FIRST add_example sets the window. Order controls the seed length.
    if seed_long_first {
        for e in &long_p {
            recognizer.add_example(2, e.clone());
        }
        for e in &short_p {
            recognizer.add_example(1, e.clone());
        }
    } else {
        for e in &short_p {
            recognizer.add_example(1, e.clone());
        }
        for e in &long_p {
            recognizer.add_example(2, e.clone());
        }
    }

    for (gid, ex) in [(1u32, &short_p), (2u32, &long_p)] {
        if let Some(s) =
            ralf_gesture_studio::engine::statistics::compute_threshold_stats_sdtw(ex, 3.0, 4, 0.15)
        {
            recognizer.set_threshold(gid, s.threshold);
        }
    }
    recognizer.start();
    recognizer
}

/// Replay a live sequence and count (short_hits, long_hits).
fn count_hits(recognizer: &mut Recognizer, live: &[Vec<f32>]) -> (usize, usize) {
    let mut prep = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
    let (mut short_hits, mut long_hits) = (0usize, 0usize);
    for frame in live {
        let processed = prep.process_frame(frame);
        if let Some(r) = recognizer.process_frame(processed) {
            match r.gesture_name.as_deref() {
                Some("short") => short_hits += 1,
                Some("long") => long_hits += 1,
                _ => {}
            }
        }
    }
    (short_hits, long_hits)
}

#[test]
fn issue25_varlen_window_truncation_probe() {
    // Short gesture ~40 frames, long gesture ~200 frames, distinct motions.
    let short_train: Vec<Vec<Vec<f32>>> = (1..=5)
        .map(|s| add_variation(&right_arm_raise(40), s, 0.01))
        .collect();
    let long_train: Vec<Vec<Vec<f32>>> = (1..=5)
        .map(|s| add_variation(&left_arm_raise(200), s, 0.01))
        .collect();

    // Live long-gesture performance: idle lead-in + long gesture + idle tail.
    let long_live_seq = {
        let mut v = standing_still(220);
        v.extend(add_variation(&left_arm_raise(200), 99, 0.01));
        v.extend(standing_still(120));
        v
    };
    let idle_seq = standing_still(400);

    // Window seeded SHORT (truncates the long gesture to its last ~40 frames).
    let mut short_seed = build_varlen_recognizer(&short_train, &long_train, false);
    let (_, long_hits_short_window) = count_hits(&mut short_seed, &long_live_seq);

    // Window seeded LONG (the long gesture fits).
    let mut long_seed = build_varlen_recognizer(&short_train, &long_train, true);
    let (_, long_hits_long_window) = count_hits(&mut long_seed, &long_live_seq);

    // Short-gesture pressure in the long-seeded (wide) window.
    let short_live = {
        let mut v = standing_still(220);
        v.extend(add_variation(&right_arm_raise(40), 99, 0.01));
        v.extend(standing_still(120));
        v
    };
    let mut long_seed2 = build_varlen_recognizer(&short_train, &long_train, true);
    let (short_hits_wide, _) = count_hits(&mut long_seed2, &short_live);

    // Idle must not fire in either seeding.
    let mut idle_rec = build_varlen_recognizer(&short_train, &long_train, true);
    let (idle_s, idle_l) = count_hits(&mut idle_rec, &idle_seq);

    println!("\n=== Issue #25 variable-length window probe ===");
    println!("long gesture (200 frames):");
    println!("  hits, SHORT-seeded window (~40): {}", long_hits_short_window);
    println!("  hits, LONG-seeded  window (~200): {}", long_hits_long_window);
    println!(
        "  truncation FN penalty (long-window minus short-window): {}",
        long_hits_long_window as i64 - long_hits_short_window as i64
    );
    println!("short gesture hits in wide window: {}", short_hits_wide);
    println!("idle-segment hits: short={}, long={}", idle_s, idle_l);

    // Truncation can only hurt the long gesture, never help it.
    assert!(
        long_hits_long_window >= long_hits_short_window,
        "long-seeded window should detect the long gesture at least as often as the truncated short-seeded window"
    );
    assert_eq!(idle_s + idle_l, 0, "pure idle must not fire any gesture");
}

// ============================================================================
// RC-3: Confusion detector uses the recognition metric (sDTW), recentred cutoff
// ============================================================================

/// Build detect_confusion_pairs input from a fixture: preprocess each gesture's
/// examples and compute its sDTW threshold (coefficient 3.0, matching the
/// benchmark calibration).
fn confusion_input_from_fixture(fname: &str) -> Vec<(Vec<Vec<Vec<f32>>>, f32)> {
    let fixture = load_fixture(fname);
    let prep = Preprocessor::new(no_preprocessing(), "mediapipe-pose-33-xy");
    fixture
        .gestures
        .iter()
        .map(|(_, exs)| {
            let processed: Vec<Vec<Vec<f32>>> =
                exs.iter().map(|e| prep.process_sequence(e)).collect();
            let threshold =
                ralf_gesture_studio::engine::statistics::compute_threshold_stats_sdtw(
                    &processed, 3.0, 4, 0.15,
                )
                .map(|s| s.threshold)
                .unwrap_or(0.0);
            (processed, threshold)
        })
        .collect()
}

/// The well-separated benchmark trio (Squat / Reach Up / Cross Arms) must NOT be
/// flagged as confused. Before the metric fix, the cross-distances were computed
/// with full-resolution standard DTW against sDTW thresholds (a metric mismatch);
/// after the fix the recentred cutoff leaves the well-separated trio clean.
#[test]
fn rc3_benchmark_trio_not_flagged() {
    use ralf_gesture_studio::engine::statistics::detect_confusion_pairs;
    let input = confusion_input_from_fixture("benchmark.ralf");
    let pairs = detect_confusion_pairs(&input);
    assert!(
        pairs.is_empty(),
        "well-separated benchmark trio must not be flagged as confused, got: {:?}",
        pairs
    );
}

/// A genuinely confusable vocabulary (Arm Sweep / jump / spin — the RC-2 fixture
/// where Arm Sweep and jump steal detections) MUST be flagged, so the detector
/// is not simply silent everywhere. Positive control for the recentred cutoff.
#[test]
#[ignore = "uses 12 MB fixture; positive control, run with --ignored rc3"]
fn rc3_confusable_vocab_is_flagged() {
    use ralf_gesture_studio::engine::statistics::detect_confusion_pairs;
    let input = confusion_input_from_fixture("jump-wave-spin-test.ralf");
    let pairs = detect_confusion_pairs(&input);
    assert!(
        !pairs.is_empty(),
        "confusable Arm Sweep/jump/spin vocabulary should be flagged"
    );
}

// ============================================================================
// RC-5: Measure the slope gate's FN-vs-FP value (measure, do not ship a change)
//
// The slope gate requires distance to be strictly falling to enter Building,
// suppressing resting-pose false positives. One Euro smoothing may already
// remove the jitter it defends against, so its continued value is unmeasured.
// This experiment toggles the gate (config flag, default ON = production) and
// measures, on clean vs jitter-injected data:
//   - FP defense: false fires during a pure noisy-resting segment (no gesture)
//   - FN cost:    whether a real, clean gesture still fires with the gate on
//
// DISCIPLINE: a clean-data green result does NOT license relaxing the gate.
// Final feel is rehearsal-gated (decision #4, signed off). This only measures.
// See docs/research/2026-06-18-four-way-tension-hardening-analysis.md (RC-5).
// ============================================================================

fn slope_gate_config(enabled: bool) -> RecognitionConfig {
    let mut c = test_config();
    c.slope_gate_enabled = enabled;
    c
}

#[test]
fn rc5_slope_gate_fn_vs_fp_measurement() {
    let training: Vec<Vec<Vec<f32>>> = (1..=5)
        .map(|s| add_variation(&right_arm_raise(120), s, 0.01))
        .collect();

    // Clean real gesture (FN probe): does it still fire with the gate on?
    let clean_gesture = {
        let mut v = standing_still(200);
        v.extend(add_variation(&right_arm_raise(120), 99, 0.01));
        v.extend(standing_still(100));
        v
    };

    // Pure noisy resting, NO gesture (FP probe): jitter injected into the
    // standing pose, the exact resting-pose noise the gate defends against.
    let noisy_resting: Vec<Vec<f32>> = add_variation(&standing_still(600), 42, 0.06);

    let run = |seq: &[Vec<f32>], enabled: bool| -> usize {
        run_recognition(
            &[("right_arm_raise", &training)],
            seq,
            slope_gate_config(enabled),
            no_preprocessing(),
        )
        .len()
    };

    let tp_gate_on = run(&clean_gesture, true);
    let tp_gate_off = run(&clean_gesture, false);
    let fp_gate_on = run(&noisy_resting, true);
    let fp_gate_off = run(&noisy_resting, false);

    println!("\n=== RC-5 slope gate FN-vs-FP measurement (clean-data lower bound) ===");
    println!(
        "clean gesture hits (TP):   gate ON = {}, gate OFF = {}",
        tp_gate_on, tp_gate_off
    );
    println!(
        "noisy-resting hits (FP):   gate ON = {}, gate OFF = {}",
        fp_gate_on, fp_gate_off
    );
    println!(
        "FP suppressed by the gate: {} (OFF {} -> ON {})",
        fp_gate_off as i64 - fp_gate_on as i64,
        fp_gate_off,
        fp_gate_on
    );
    println!("DISCIPLINE: clean-data result does NOT license relaxing the gate (rehearsal-gated).");

    // Invariants (not a recommendation): the gate must not INCREASE false
    // positives, and must not cost the clean true positive on this data.
    assert!(
        fp_gate_on <= fp_gate_off,
        "slope gate should never increase resting-pose false positives"
    );
    assert!(
        tp_gate_on >= 1,
        "the clean gesture must still fire with the gate enabled (no FN on clean data)"
    );
}
