//! Data augmentation for gesture recognition.
//!
//! Generates ephemeral augmented copies of training examples for DTW comparison.
//! Augmented examples are NOT stored in .ralf files — they are regenerated
//! deterministically whenever the recognizer is rebuilt.
//!
//! Three techniques are available, each independently toggleable:
//!
//! 1. **Temporal stretch** — resample at 0.85x–1.15x speed (linear interpolation)
//! 2. **Spatial jitter** — Gaussian noise (sigma = 1% of mean inter-frame displacement)
//! 3. **Horizontal mirror** — swap left/right joint pairs + negate X (50% of copies)

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use super::dtw::Sequence;

/// MediaPipe Pose 33-keypoint left/right joint pairs for horizontal mirroring.
/// Each tuple is (left_joint_index, right_joint_index).
const MEDIAPIPE_MIRROR_PAIRS: [(usize, usize); 16] = [
    (1, 4),   // LEFT_EYE_INNER, RIGHT_EYE_INNER
    (2, 5),   // LEFT_EYE, RIGHT_EYE
    (3, 6),   // LEFT_EYE_OUTER, RIGHT_EYE_OUTER
    (7, 8),   // LEFT_EAR, RIGHT_EAR
    (9, 10),  // MOUTH_LEFT, MOUTH_RIGHT
    (11, 12), // LEFT_SHOULDER, RIGHT_SHOULDER
    (13, 14), // LEFT_ELBOW, RIGHT_ELBOW
    (15, 16), // LEFT_WRIST, RIGHT_WRIST
    (17, 18), // LEFT_PINKY, RIGHT_PINKY
    (19, 20), // LEFT_INDEX, RIGHT_INDEX
    (21, 22), // LEFT_THUMB, RIGHT_THUMB
    (23, 24), // LEFT_HIP, RIGHT_HIP
    (25, 26), // LEFT_KNEE, RIGHT_KNEE
    (27, 28), // LEFT_ANKLE, RIGHT_ANKLE
    (29, 30), // LEFT_HEEL, RIGHT_HEEL
    (31, 32), // LEFT_FOOT_INDEX, RIGHT_FOOT_INDEX
];

/// Number of coordinates per joint (XY = 2 for mediapipe-pose-33-xy).
const COORDS_PER_JOINT: usize = 2;

/// Minimum frame count for a 66-dim MediaPipe frame to be eligible for mirroring.
const MIN_MIRROR_DIMS: usize = 66;

/// Vocabulary-level augmentation configuration.
/// Stored in Vocabulary struct with `#[serde(default)]` for backward compatibility.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AugmentationConfig {
    /// Master enable/disable toggle
    #[serde(default)]
    pub enabled: bool,
    /// Number of augmented copies per real example (1–5)
    #[serde(default = "default_multiplier")]
    pub multiplier: u32,
    /// Enable temporal stretch (0.85x–1.15x speed variation)
    #[serde(default = "default_true")]
    pub temporal_stretch: bool,
    /// Enable spatial jitter (Gaussian noise)
    #[serde(default = "default_true")]
    pub spatial_jitter: bool,
    /// Enable horizontal mirror (swap left/right joints + negate X)
    #[serde(default = "default_true")]
    pub horizontal_mirror: bool,
}

fn default_multiplier() -> u32 {
    2
}
fn default_true() -> bool {
    true
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Off by default — user must opt in
            multiplier: 2,
            temporal_stretch: true,
            spatial_jitter: true,
            horizontal_mirror: true,
        }
    }
}

/// Generate augmented copies for a single preprocessed example.
///
/// Deterministic: same inputs always produce the same augmented copies.
/// Returns `multiplier` copies, each with a random combination of enabled techniques.
pub fn generate_augmented(
    original: &Sequence,
    config: &AugmentationConfig,
    gesture_id: u32,
    example_index: usize,
) -> Vec<Sequence> {
    if !config.enabled || original.is_empty() {
        return Vec::new();
    }

    let mut augmented = Vec::with_capacity(config.multiplier as usize);

    for copy_idx in 0..config.multiplier {
        let seed = deterministic_seed(gesture_id, example_index, copy_idx as usize);
        let mut rng = StdRng::seed_from_u64(seed);

        let mut seq = original.clone();

        if config.temporal_stretch {
            let factor = rng.gen_range(0.85_f32..=1.15_f32);
            seq = temporal_stretch(&seq, factor);
        }

        if config.spatial_jitter {
            let sigma = compute_jitter_sigma(&seq);
            apply_spatial_jitter(&mut seq, sigma, &mut rng);
        }

        if config.horizontal_mirror && rng.gen_bool(0.5) {
            apply_horizontal_mirror(&mut seq);
        }

        augmented.push(seq);
    }

    augmented
}

/// Create a deterministic seed from gesture_id, example_index, and copy_index.
fn deterministic_seed(gesture_id: u32, example_index: usize, copy_index: usize) -> u64 {
    let mut h: u64 = 0x517c_c1b7_2722_0a95;
    h = h
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(gesture_id as u64);
    h = h
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(example_index as u64);
    h = h
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(copy_index as u64);
    h
}

/// Resample a sequence by the given stretch factor using linear interpolation.
/// Result always has the same number of frames as the original.
fn temporal_stretch(seq: &Sequence, stretch_factor: f32) -> Sequence {
    let n = seq.len();
    if n <= 1 {
        return seq.clone();
    }

    let mut result = Vec::with_capacity(n);

    for i in 0..n {
        let src_pos = (i as f32) * stretch_factor;
        let src_idx = src_pos.floor() as usize;
        let frac = src_pos - src_pos.floor();

        let idx0 = src_idx.min(n - 1);
        let idx1 = (src_idx + 1).min(n - 1);

        let frame: Vec<f32> = seq[idx0]
            .iter()
            .zip(seq[idx1].iter())
            .map(|(a, b)| a * (1.0 - frac) + b * frac)
            .collect();
        result.push(frame);
    }

    result
}

/// Compute jitter sigma: 1% of mean inter-frame Euclidean displacement.
fn compute_jitter_sigma(seq: &Sequence) -> f32 {
    if seq.len() < 2 {
        return 0.0;
    }

    let mut total_displacement = 0.0_f32;
    for pair in seq.windows(2) {
        let sum_sq: f32 = pair[0]
            .iter()
            .zip(pair[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        total_displacement += sum_sq.sqrt();
    }

    let mean_displacement = total_displacement / (seq.len() - 1) as f32;
    mean_displacement * 0.01
}

/// Add Gaussian noise to each value in the sequence.
/// Uses Box-Muller transform (avoids rand_distr dependency).
fn apply_spatial_jitter(seq: &mut Sequence, sigma: f32, rng: &mut StdRng) {
    if sigma <= 0.0 {
        return;
    }
    for frame in seq.iter_mut() {
        for val in frame.iter_mut() {
            let u1: f32 = rng.gen_range(0.0001_f32..1.0_f32);
            let u2: f32 = rng.gen::<f32>() * std::f32::consts::TAU;
            let z = (-2.0 * u1.ln()).sqrt() * u2.cos();
            *val += z * sigma;
        }
    }
}

/// Swap left/right MediaPipe joint pairs and negate X coordinates.
/// Both operations together produce a true horizontal mirror.
fn apply_horizontal_mirror(seq: &mut Sequence) {
    for frame in seq.iter_mut() {
        if frame.len() < MIN_MIRROR_DIMS {
            continue;
        }
        // Step 1: Swap left/right joint data
        for &(left, right) in &MEDIAPIPE_MIRROR_PAIRS {
            let lo = left * COORDS_PER_JOINT;
            let ro = right * COORDS_PER_JOINT;
            for c in 0..COORDS_PER_JOINT {
                frame.swap(lo + c, ro + c);
            }
        }
        // Step 2: Negate X coordinates (every even index in XY layout)
        let joint_count = frame.len() / COORDS_PER_JOINT;
        for joint in 0..joint_count {
            frame[joint * COORDS_PER_JOINT] = -frame[joint * COORDS_PER_JOINT];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_sequence(n_frames: usize) -> Sequence {
        (0..n_frames)
            .map(|i| (0..66).map(|d| (i * 66 + d) as f32 * 0.01).collect())
            .collect()
    }

    #[test]
    fn test_disabled_returns_empty() {
        let config = AugmentationConfig::default();
        let seq = make_test_sequence(30);
        let result = generate_augmented(&seq, &config, 1, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_empty_sequence_returns_empty() {
        let config = AugmentationConfig {
            enabled: true,
            ..Default::default()
        };
        let empty: Sequence = Vec::new();
        let result = generate_augmented(&empty, &config, 1, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_produces_correct_count() {
        let config = AugmentationConfig {
            enabled: true,
            multiplier: 3,
            ..Default::default()
        };
        let seq = make_test_sequence(30);
        let result = generate_augmented(&seq, &config, 1, 0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_preserves_frame_count() {
        let config = AugmentationConfig {
            enabled: true,
            multiplier: 2,
            temporal_stretch: true,
            spatial_jitter: false,
            horizontal_mirror: false,
        };
        let seq = make_test_sequence(30);
        let result = generate_augmented(&seq, &config, 1, 0);
        for aug in &result {
            assert_eq!(aug.len(), seq.len());
        }
    }

    #[test]
    fn test_preserves_dimensions() {
        let config = AugmentationConfig {
            enabled: true,
            multiplier: 2,
            ..Default::default()
        };
        let seq = make_test_sequence(20);
        let result = generate_augmented(&seq, &config, 1, 0);
        for aug in &result {
            for frame in aug {
                assert_eq!(frame.len(), 66);
            }
        }
    }

    #[test]
    fn test_deterministic() {
        let config = AugmentationConfig {
            enabled: true,
            multiplier: 2,
            ..Default::default()
        };
        let seq = make_test_sequence(30);
        let r1 = generate_augmented(&seq, &config, 1, 0);
        let r2 = generate_augmented(&seq, &config, 1, 0);
        assert_eq!(r1.len(), r2.len());
        for (a, b) in r1.iter().zip(r2.iter()) {
            for (fa, fb) in a.iter().zip(b.iter()) {
                for (va, vb) in fa.iter().zip(fb.iter()) {
                    assert!((va - vb).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_different_seeds_differ() {
        let config = AugmentationConfig {
            enabled: true,
            multiplier: 1,
            ..Default::default()
        };
        let seq = make_test_sequence(30);
        let r1 = generate_augmented(&seq, &config, 1, 0);
        let r2 = generate_augmented(&seq, &config, 1, 1);
        let any_diff = r1[0].iter().zip(r2[0].iter()).any(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .any(|(va, vb)| (va - vb).abs() > 1e-6)
        });
        assert!(any_diff);
    }

    #[test]
    fn test_temporal_stretch_identity() {
        let seq = make_test_sequence(10);
        let stretched = temporal_stretch(&seq, 1.0);
        assert_eq!(stretched.len(), seq.len());
        for (a, b) in seq.iter().zip(stretched.iter()) {
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn test_temporal_stretch_preserves_length() {
        let seq = make_test_sequence(20);
        assert_eq!(temporal_stretch(&seq, 0.85).len(), 20);
        assert_eq!(temporal_stretch(&seq, 1.15).len(), 20);
    }

    #[test]
    fn test_temporal_stretch_single_frame() {
        let seq = vec![vec![1.0, 2.0, 3.0]];
        let stretched = temporal_stretch(&seq, 0.9);
        assert_eq!(stretched.len(), 1);
    }

    #[test]
    fn test_mirror_involution() {
        let seq = make_test_sequence(5);
        let mut mirrored = seq.clone();
        apply_horizontal_mirror(&mut mirrored);
        apply_horizontal_mirror(&mut mirrored);
        for (a, b) in seq.iter().zip(mirrored.iter()) {
            for (va, vb) in a.iter().zip(b.iter()) {
                assert!((va - vb).abs() < 1e-6, "Double mirror should be identity");
            }
        }
    }

    #[test]
    fn test_mirror_changes_values() {
        let seq = make_test_sequence(5);
        let mut mirrored = seq.clone();
        apply_horizontal_mirror(&mut mirrored);
        let any_diff = seq.iter().zip(mirrored.iter()).any(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .any(|(va, vb)| (va - vb).abs() > 1e-6)
        });
        assert!(any_diff);
    }

    #[test]
    fn test_mirror_skips_short_frames() {
        let mut seq = vec![vec![1.0, 2.0, 3.0]]; // too short for 66-dim
        let original = seq.clone();
        apply_horizontal_mirror(&mut seq);
        assert_eq!(seq, original);
    }

    #[test]
    fn test_jitter_modifies_values() {
        let mut seq = make_test_sequence(10);
        let original = seq.clone();
        let mut rng = StdRng::seed_from_u64(42);
        apply_spatial_jitter(&mut seq, 0.1, &mut rng);
        let any_diff = original.iter().zip(seq.iter()).any(|(a, b)| {
            a.iter()
                .zip(b.iter())
                .any(|(va, vb)| (va - vb).abs() > 1e-10)
        });
        assert!(any_diff);
    }

    #[test]
    fn test_jitter_zero_sigma_no_change() {
        let mut seq = make_test_sequence(5);
        let original = seq.clone();
        let mut rng = StdRng::seed_from_u64(42);
        apply_spatial_jitter(&mut seq, 0.0, &mut rng);
        assert_eq!(seq, original);
    }

    #[test]
    fn test_jitter_sigma_computation() {
        // Stationary: all frames identical
        let stationary: Sequence = vec![vec![1.0, 2.0]; 10];
        assert_eq!(compute_jitter_sigma(&stationary), 0.0);

        // Moving: 1 unit per frame along x
        let moving: Sequence = (0..10).map(|i| vec![i as f32, 0.0]).collect();
        let sigma = compute_jitter_sigma(&moving);
        assert!((sigma - 0.01).abs() < 0.001); // mean displacement = 1.0, sigma = 1%
    }

    #[test]
    fn test_jitter_sigma_single_frame() {
        let seq = vec![vec![1.0, 2.0]];
        assert_eq!(compute_jitter_sigma(&seq), 0.0);
    }

    #[test]
    fn test_default_config() {
        let config = AugmentationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.multiplier, 2);
        assert!(config.temporal_stretch);
        assert!(config.spatial_jitter);
        assert!(config.horizontal_mirror);
    }
}
