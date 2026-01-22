//! Dynamic Time Warping (DTW) algorithm for gesture recognition.
//!
//! DTW measures the similarity between two temporal sequences that may vary in speed.
//! This is ideal for gesture recognition where the same gesture may be performed
//! at different speeds.

/// A frame of data - a vector of floats representing one point in time
/// (e.g., 68 floats for 34 joints × XY coordinates)
pub type Frame = Vec<f32>;

/// A sequence of frames representing a gesture
pub type Sequence = Vec<Frame>;

/// Calculate the Euclidean distance between two frames.
///
/// Both frames must have the same number of dimensions.
/// Returns the square root of the sum of squared differences.
///
/// # Panics
/// Panics if frames have different lengths.
pub fn euclidean_distance(a: &Frame, b: &Frame) -> f32 {
    assert_eq!(a.len(), b.len(), "Frames must have the same dimensions");

    let sum_sq: f32 = a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    sum_sq.sqrt()
}

/// Calculate the DTW distance between two sequences.
///
/// Uses dynamic programming to find the optimal alignment between sequences,
/// allowing for time warping (stretching/compressing) to match patterns
/// performed at different speeds.
///
/// # Algorithm
///
/// 1. Build a cost matrix where `cost[i][j]` is the minimum cumulative distance
///    to align `seq1[0..=i]` with `seq2[0..=j]`
/// 2. Each cell considers three possible moves:
///    - Match: align current frames (diagonal move)
///    - Insert: skip a frame in seq1 (move down)
///    - Delete: skip a frame in seq2 (move right)
/// 3. Return the final cell value as the total alignment cost
///
/// # Returns
///
/// The DTW distance (lower = more similar). Returns 0.0 for identical sequences.
/// Returns `f32::INFINITY` if either sequence is empty.
pub fn dtw_distance(seq1: &Sequence, seq2: &Sequence) -> f32 {
    if seq1.is_empty() || seq2.is_empty() {
        return f32::INFINITY;
    }

    let n = seq1.len();
    let m = seq2.len();

    // Create cost matrix with infinity as initial values
    // We use n+1 and m+1 to handle the boundary conditions
    let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];

    // Base case: starting point
    cost[0][0] = 0.0;

    // Fill in the cost matrix
    for i in 1..=n {
        for j in 1..=m {
            let dist = euclidean_distance(&seq1[i - 1], &seq2[j - 1]);

            // Minimum of three possible moves:
            // - cost[i-1][j-1]: match (diagonal)
            // - cost[i-1][j]: insertion (vertical)
            // - cost[i][j-1]: deletion (horizontal)
            let min_prev = cost[i - 1][j - 1]
                .min(cost[i - 1][j])
                .min(cost[i][j - 1]);

            cost[i][j] = dist + min_prev;
        }
    }

    cost[n][m]
}

/// Calculate normalized DTW distance.
///
/// Normalizes the DTW distance by the length of the warping path,
/// making it easier to compare distances between sequences of different lengths.
///
/// # Returns
///
/// The normalized DTW distance. Returns `f32::INFINITY` if either sequence is empty.
pub fn dtw_distance_normalized(seq1: &Sequence, seq2: &Sequence) -> f32 {
    if seq1.is_empty() || seq2.is_empty() {
        return f32::INFINITY;
    }

    let distance = dtw_distance(seq1, seq2);

    // Normalize by the average length of the two sequences
    // This accounts for different sequence lengths
    let avg_len = (seq1.len() + seq2.len()) as f32 / 2.0;

    distance / avg_len
}

/// Find the best matching example from a set of examples.
///
/// Compares the input sequence against all examples and returns
/// the index and distance of the best match.
///
/// # Returns
///
/// `Some((index, distance))` of the best match, or `None` if examples is empty.
pub fn find_best_match(input: &Sequence, examples: &[Sequence]) -> Option<(usize, f32)> {
    if examples.is_empty() || input.is_empty() {
        return None;
    }

    let mut best_idx = 0;
    let mut best_dist = f32::INFINITY;

    for (idx, example) in examples.iter().enumerate() {
        let dist = dtw_distance(input, example);
        if dist < best_dist {
            best_dist = dist;
            best_idx = idx;
        }
    }

    Some((best_idx, best_dist))
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // Euclidean Distance Tests
    // =========================================================================

    #[test]
    fn test_euclidean_identical_frames() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(euclidean_distance(&a, &b), 0.0);
    }

    #[test]
    fn test_euclidean_simple() {
        // Distance from (0,0) to (3,4) should be 5
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert_eq!(euclidean_distance(&a, &b), 5.0);
    }

    #[test]
    fn test_euclidean_single_dimension() {
        let a = vec![0.0];
        let b = vec![10.0];
        assert_eq!(euclidean_distance(&a, &b), 10.0);
    }

    #[test]
    fn test_euclidean_high_dimensional() {
        // 68 dimensions (like skeleton data)
        let a: Vec<f32> = (0..68).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..68).map(|i| i as f32).collect();
        assert_eq!(euclidean_distance(&a, &b), 0.0);
    }

    #[test]
    #[should_panic(expected = "Frames must have the same dimensions")]
    fn test_euclidean_mismatched_dimensions() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        euclidean_distance(&a, &b);
    }

    // =========================================================================
    // DTW Distance Tests
    // =========================================================================

    #[test]
    fn test_dtw_identical_sequences() {
        let seq = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let distance = dtw_distance(&seq, &seq);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_single_frame_sequences() {
        let seq1 = vec![vec![0.0, 0.0]];
        let seq2 = vec![vec![3.0, 4.0]];
        let distance = dtw_distance(&seq1, &seq2);
        assert_eq!(distance, 5.0); // Euclidean distance
    }

    #[test]
    fn test_dtw_empty_sequence() {
        let seq1: Sequence = vec![];
        let seq2 = vec![vec![1.0, 2.0]];

        assert_eq!(dtw_distance(&seq1, &seq2), f32::INFINITY);
        assert_eq!(dtw_distance(&seq2, &seq1), f32::INFINITY);
        assert_eq!(dtw_distance(&seq1, &seq1), f32::INFINITY);
    }

    #[test]
    fn test_dtw_time_warping_slower() {
        // Same pattern, but one is "slower" (repeated frames)
        let fast = vec![
            vec![0.0],
            vec![1.0],
        ];
        let slow = vec![
            vec![0.0],
            vec![0.0],
            vec![1.0],
            vec![1.0],
        ];

        let distance = dtw_distance(&fast, &slow);
        // DTW should align these well - distance should be low
        assert!(distance < 1.0, "DTW should handle time warping, got {}", distance);
    }

    #[test]
    fn test_dtw_time_warping_faster() {
        // One sequence is compressed
        let slow = vec![
            vec![0.0],
            vec![0.5],
            vec![1.0],
            vec![1.5],
            vec![2.0],
        ];
        let fast = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];

        let distance = dtw_distance(&slow, &fast);
        // Should still match reasonably well
        assert!(distance < 2.0, "DTW should handle speed differences, got {}", distance);
    }

    #[test]
    fn test_dtw_completely_different() {
        let seq1 = vec![vec![0.0, 0.0]];
        let seq2 = vec![vec![100.0, 100.0]];

        let distance = dtw_distance(&seq1, &seq2);
        // Distance should be large (sqrt(100^2 + 100^2) ≈ 141.4)
        assert!(distance > 100.0, "Different sequences should have large distance, got {}", distance);
    }

    #[test]
    fn test_dtw_symmetry() {
        let seq1 = vec![vec![1.0], vec![2.0], vec![3.0]];
        let seq2 = vec![vec![1.5], vec![2.5]];

        let d1 = dtw_distance(&seq1, &seq2);
        let d2 = dtw_distance(&seq2, &seq1);

        assert_eq!(d1, d2, "DTW should be symmetric");
    }

    #[test]
    fn test_dtw_triangle_inequality_approximation() {
        // DTW doesn't strictly satisfy triangle inequality, but for similar sequences
        // it should roughly hold
        let a = vec![vec![0.0], vec![1.0]];
        let b = vec![vec![0.5], vec![1.5]];
        let c = vec![vec![1.0], vec![2.0]];

        let ab = dtw_distance(&a, &b);
        let bc = dtw_distance(&b, &c);
        let ac = dtw_distance(&a, &c);

        // ac should not be dramatically larger than ab + bc
        assert!(ac <= ab + bc + 0.1, "Rough triangle inequality violated");
    }

    #[test]
    fn test_dtw_multi_dimensional() {
        // Test with 4-dimensional frames (like 2 joints × XY)
        let seq1 = vec![
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0, 2.0],
        ];
        let seq2 = vec![
            vec![0.0, 0.0, 1.0, 1.0],
            vec![0.5, 0.5, 1.5, 1.5],
            vec![1.0, 1.0, 2.0, 2.0],
        ];

        let distance = dtw_distance(&seq1, &seq2);
        // Should be relatively small since the patterns are similar
        assert!(distance < 2.0, "Similar multi-dim sequences should match well, got {}", distance);
    }

    // =========================================================================
    // Normalized DTW Tests
    // =========================================================================

    #[test]
    fn test_dtw_normalized_identical() {
        let seq = vec![vec![1.0], vec![2.0], vec![3.0]];
        assert_eq!(dtw_distance_normalized(&seq, &seq), 0.0);
    }

    #[test]
    fn test_dtw_normalized_different_lengths() {
        let short = vec![vec![0.0], vec![1.0]];
        let long = vec![vec![0.0], vec![0.5], vec![1.0], vec![1.5], vec![2.0]];

        let norm_dist = dtw_distance_normalized(&short, &long);
        let raw_dist = dtw_distance(&short, &long);

        // Normalized should be less than raw for longer sequences
        assert!(norm_dist < raw_dist, "Normalized should be smaller than raw");
    }

    #[test]
    fn test_dtw_normalized_empty() {
        let seq: Sequence = vec![];
        let other = vec![vec![1.0]];
        assert_eq!(dtw_distance_normalized(&seq, &other), f32::INFINITY);
    }

    // =========================================================================
    // Best Match Tests
    // =========================================================================

    #[test]
    fn test_find_best_match_exact() {
        let examples = vec![
            vec![vec![0.0], vec![1.0]],  // example 0
            vec![vec![5.0], vec![6.0]],  // example 1
            vec![vec![10.0], vec![11.0]], // example 2
        ];

        // Input matches example 1 exactly
        let input = vec![vec![5.0], vec![6.0]];
        let result = find_best_match(&input, &examples);

        assert!(result.is_some());
        let (idx, dist) = result.unwrap();
        assert_eq!(idx, 1);
        assert_eq!(dist, 0.0);
    }

    #[test]
    fn test_find_best_match_closest() {
        let examples = vec![
            vec![vec![0.0]],
            vec![vec![10.0]],
            vec![vec![20.0]],
        ];

        // Input is closest to example 1 (10.0)
        let input = vec![vec![9.0]];
        let result = find_best_match(&input, &examples);

        assert!(result.is_some());
        let (idx, dist) = result.unwrap();
        assert_eq!(idx, 1);
        assert_eq!(dist, 1.0);
    }

    #[test]
    fn test_find_best_match_empty_examples() {
        let examples: Vec<Sequence> = vec![];
        let input = vec![vec![1.0]];

        assert!(find_best_match(&input, &examples).is_none());
    }

    #[test]
    fn test_find_best_match_empty_input() {
        let examples = vec![vec![vec![1.0]]];
        let input: Sequence = vec![];

        assert!(find_best_match(&input, &examples).is_none());
    }

    // =========================================================================
    // Real-world Scenario Tests
    // =========================================================================

    #[test]
    fn test_gesture_recognition_scenario() {
        // Simulate a simple "wave" gesture as a sequence of hand positions
        // Training examples
        let wave1 = vec![
            vec![0.0, 0.5], // hand at rest
            vec![0.2, 0.7], // hand moving up-right
            vec![0.4, 0.5], // hand at top
            vec![0.2, 0.3], // hand moving down-left
            vec![0.0, 0.5], // hand back to rest
        ];

        let wave2 = vec![
            vec![0.0, 0.5],
            vec![0.15, 0.65], // slightly different timing
            vec![0.35, 0.5],
            vec![0.15, 0.35],
            vec![0.0, 0.5],
        ];

        // A different gesture (jump)
        let jump = vec![
            vec![0.5, 0.0], // crouching
            vec![0.5, 0.5], // jumping up
            vec![0.5, 0.8], // at peak
            vec![0.5, 0.5], // coming down
            vec![0.5, 0.0], // landing
        ];

        // Test input - a wave performed slightly differently
        let test_wave = vec![
            vec![0.0, 0.5],
            vec![0.25, 0.72],
            vec![0.42, 0.48],
            vec![0.22, 0.32],
            vec![0.0, 0.5],
        ];

        let examples = vec![wave1, wave2, jump];
        let result = find_best_match(&test_wave, &examples);

        assert!(result.is_some());
        let (idx, _dist) = result.unwrap();
        // Should match one of the wave examples (0 or 1), not jump (2)
        assert!(idx < 2, "Should recognize as wave, not jump");
    }
}
