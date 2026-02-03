//! Variance-based joint weighting for DTW.
//!
//! Computes per-dimension weights from training example variance.
//! High-variance dimensions (joints that move during the gesture)
//! receive higher weight; low-variance dimensions (joints that stay still)
//! are down-weighted. Weights are applied by scaling frame data before DTW,
//! so no changes to the DTW algorithm are needed.

use super::dtw::{Frame, Sequence};

/// Minimum weight to prevent division by zero / numerical instability.
const MIN_WEIGHT: f32 = 0.01;

/// Compute per-dimension weights from variance across training examples.
///
/// Returns `None` if fewer than 2 examples (variance is undefined).
///
/// Algorithm:
/// 1. Compute variance for each dimension across all frames of all examples
/// 2. Raw weight = sqrt(variance) — maps to same units as data
/// 3. Normalize so mean weight = 1.0 — preserves overall distance scale
/// 4. Clamp minimum to 0.01 — numerical stability
pub fn compute_joint_weights(examples: &[Sequence]) -> Option<Vec<f32>> {
    if examples.len() < 2 {
        return None;
    }

    // Determine dimensionality from first frame of first example
    let dims = examples.first()?.first()?.len();
    if dims == 0 {
        return None;
    }

    // Compute mean and variance for each dimension using Welford's online algorithm
    let mut count = 0u64;
    let mut mean = vec![0.0f64; dims];
    let mut m2 = vec![0.0f64; dims];

    for example in examples {
        for frame in example {
            if frame.len() != dims {
                continue; // Skip mismatched frames
            }
            count += 1;
            for d in 0..dims {
                let x = frame[d] as f64;
                let delta = x - mean[d];
                mean[d] += delta / count as f64;
                let delta2 = x - mean[d];
                m2[d] += delta * delta2;
            }
        }
    }

    if count < 2 {
        return None;
    }

    // Compute variance and raw weights
    let mut raw_weights: Vec<f32> = m2
        .iter()
        .map(|m| {
            let variance = (*m / (count - 1) as f64) as f32;
            variance.sqrt().max(MIN_WEIGHT)
        })
        .collect();

    // Normalize so mean weight = 1.0 (preserves distance scale)
    let weight_sum: f32 = raw_weights.iter().sum();
    let weight_mean = weight_sum / raw_weights.len() as f32;

    if weight_mean < f32::EPSILON {
        return None; // All dimensions have zero variance
    }

    for w in &mut raw_weights {
        *w /= weight_mean;
    }

    Some(raw_weights)
}

/// Scale every frame in a sequence by per-dimension weights.
pub fn apply_weights_to_sequence(seq: &Sequence, weights: &[f32]) -> Sequence {
    seq.iter()
        .map(|frame| apply_weights_to_frame(frame, weights))
        .collect()
}

/// Scale a single frame by per-dimension weights.
///
/// Each dimension is multiplied by its corresponding weight.
/// If the frame and weights have different lengths, scales up to the shorter length
/// and copies remaining dimensions unchanged.
pub fn apply_weights_to_frame(frame: &[f32], weights: &[f32]) -> Frame {
    frame
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| v * w)
        .chain(frame.iter().skip(weights.len()).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_weights_returns_none_for_single_example() {
        let examples = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
        assert!(compute_joint_weights(&examples).is_none());
    }

    #[test]
    fn test_compute_weights_returns_none_for_empty() {
        let examples: Vec<Sequence> = Vec::new();
        assert!(compute_joint_weights(&examples).is_none());
    }

    #[test]
    fn test_compute_weights_equal_variance_gives_uniform_weights() {
        // Two examples with equal variance in both dimensions
        let examples = vec![
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
            vec![vec![0.0, 0.0], vec![1.0, 1.0]],
        ];
        let weights = compute_joint_weights(&examples).unwrap();

        assert_eq!(weights.len(), 2);
        // Both dimensions have equal variance → both weights ≈ 1.0
        assert!((weights[0] - 1.0).abs() < 0.01);
        assert!((weights[1] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_weights_high_variance_gets_higher_weight() {
        // Dimension 0: high variance (0 to 10)
        // Dimension 1: low variance (5.0 to 5.1)
        let examples = vec![
            vec![vec![0.0, 5.0], vec![10.0, 5.1]],
            vec![vec![0.0, 5.0], vec![10.0, 5.1]],
        ];
        let weights = compute_joint_weights(&examples).unwrap();

        assert_eq!(weights.len(), 2);
        // Dimension 0 should have much higher weight
        assert!(
            weights[0] > weights[1] * 5.0,
            "High-variance dim should have much higher weight: {:?}",
            weights
        );
    }

    #[test]
    fn test_compute_weights_zero_variance_gets_min_weight() {
        // Dimension 0: varies, Dimension 1: constant
        let examples = vec![
            vec![vec![0.0, 5.0], vec![10.0, 5.0]],
            vec![vec![0.0, 5.0], vec![10.0, 5.0]],
        ];
        let weights = compute_joint_weights(&examples).unwrap();

        // Dimension 1 (constant) should get minimum weight (clamped)
        assert!(
            weights[1] < 0.1,
            "Zero-variance dim should be near-zero: {}",
            weights[1]
        );
    }

    #[test]
    fn test_compute_weights_mean_is_one() {
        // Verify normalization: mean weight should be ~1.0
        let examples = vec![
            vec![vec![0.0, 5.0, 1.0], vec![10.0, 5.1, 3.0]],
            vec![vec![2.0, 5.0, 0.5], vec![8.0, 5.1, 2.5]],
            vec![vec![1.0, 5.0, 0.0], vec![9.0, 5.1, 4.0]],
        ];
        let weights = compute_joint_weights(&examples).unwrap();

        let mean_weight: f32 = weights.iter().sum::<f32>() / weights.len() as f32;
        assert!(
            (mean_weight - 1.0).abs() < 0.05,
            "Mean weight should be ~1.0, got {}",
            mean_weight
        );
    }

    #[test]
    fn test_apply_weights_to_frame() {
        let frame = vec![2.0, 3.0, 4.0];
        let weights = vec![1.0, 0.5, 2.0];
        let scaled = apply_weights_to_frame(&frame, &weights);

        assert_eq!(scaled, vec![2.0, 1.5, 8.0]);
    }

    #[test]
    fn test_apply_weights_to_frame_identity() {
        let frame = vec![2.0, 3.0, 4.0];
        let weights = vec![1.0, 1.0, 1.0];
        let scaled = apply_weights_to_frame(&frame, &weights);

        assert_eq!(scaled, frame);
    }

    #[test]
    fn test_apply_weights_to_sequence() {
        let seq = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let weights = vec![2.0, 0.5];
        let scaled = apply_weights_to_sequence(&seq, &weights);

        assert_eq!(scaled, vec![vec![2.0, 1.0], vec![6.0, 2.0]]);
    }

    #[test]
    fn test_apply_weights_mismatched_lengths_pads_unweighted() {
        // Frame is longer than weights — extra dims copied unchanged
        let frame = vec![1.0, 2.0, 3.0, 4.0];
        let weights = vec![2.0, 0.5];
        let scaled = apply_weights_to_frame(&frame, &weights);

        assert_eq!(scaled, vec![2.0, 1.0, 3.0, 4.0]);
    }

    #[test]
    fn test_weights_with_many_examples() {
        // Simulate realistic scenario: 5 examples of 3-dim data
        let examples: Vec<Sequence> = (0..5)
            .map(|i| {
                (0..10)
                    .map(|t| {
                        vec![
                            (t as f32 + i as f32) * 10.0,        // dim 0: high variance
                            50.0 + (i as f32) * 0.1,             // dim 1: low variance
                            (t as f32).sin() * (1.0 + i as f32), // dim 2: medium variance
                        ]
                    })
                    .collect()
            })
            .collect();

        let weights = compute_joint_weights(&examples).unwrap();
        assert_eq!(weights.len(), 3);

        // dim 0 (high var) > dim 2 (medium) > dim 1 (low)
        assert!(
            weights[0] > weights[2],
            "dim0 ({}) should > dim2 ({})",
            weights[0],
            weights[2]
        );
        assert!(
            weights[2] > weights[1],
            "dim2 ({}) should > dim1 ({})",
            weights[2],
            weights[1]
        );
    }
}
