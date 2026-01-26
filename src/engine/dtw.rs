//! Dynamic Time Warping (DTW) algorithm for gesture recognition.
//!
//! DTW measures the similarity between two temporal sequences that may vary in speed.
//! This is ideal for gesture recognition where the same gesture may be performed
//! at different speeds.

/// A frame of data - a vector of floats representing one point in time
/// (e.g., 66 floats for 33 MediaPipe keypoints × XY coordinates)
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

/// Calculate DTW distance with Sakoe-Chiba band constraint.
///
/// Limits the warping path to stay within a diagonal band, reducing computation
/// from O(N×M) to O(N×W) where W is the band width.
///
/// This is appropriate for gesture recognition where we expect the input
/// to roughly follow the same timing as the template.
///
/// # Arguments
/// * `seq1` - First sequence
/// * `seq2` - Second sequence
/// * `band_width` - Maximum allowed deviation from diagonal (in frames)
///
/// # Returns
///
/// The constrained DTW distance. Returns `f32::INFINITY` if sequences are empty
/// or if no valid path exists within the band.
#[allow(dead_code)]
pub fn dtw_distance_constrained(seq1: &Sequence, seq2: &Sequence, band_width: usize) -> f32 {
    if seq1.is_empty() || seq2.is_empty() {
        return f32::INFINITY;
    }

    let n = seq1.len();
    let m = seq2.len();

    // Create cost matrix with infinity as initial values
    let mut cost = vec![vec![f32::INFINITY; m + 1]; n + 1];

    // Base case: starting point
    cost[0][0] = 0.0;

    // Fill in the cost matrix within the Sakoe-Chiba band
    for i in 1..=n {
        // Calculate the band bounds for this row
        // The diagonal would be at j = i * m / n
        // We allow band_width deviation on each side
        let diagonal = (i * m) / n;
        let j_min = diagonal.saturating_sub(band_width).max(1);
        let j_max = (diagonal + band_width + 1).min(m);

        for j in j_min..=j_max {
            let dist = euclidean_distance(&seq1[i - 1], &seq2[j - 1]);

            // Minimum of three possible moves (if within band):
            let min_prev = cost[i - 1][j - 1]
                .min(cost[i - 1][j])
                .min(cost[i][j - 1]);

            cost[i][j] = dist + min_prev;
        }
    }

    cost[n][m]
}

/// Calculate normalized DTW distance with Sakoe-Chiba band constraint.
///
/// Combines normalization with band constraint for efficient, length-independent matching.
///
/// # Arguments
/// * `seq1` - First sequence
/// * `seq2` - Second sequence
/// * `band_fraction` - Band width as fraction of sequence length (e.g., 0.2 = 20%)
#[allow(dead_code)]
pub fn dtw_distance_constrained_normalized(
    seq1: &Sequence,
    seq2: &Sequence,
    band_fraction: f32,
) -> f32 {
    if seq1.is_empty() || seq2.is_empty() {
        return f32::INFINITY;
    }

    // Calculate band width based on the longer sequence
    let max_len = seq1.len().max(seq2.len());
    let band_width = ((max_len as f32) * band_fraction).ceil() as usize;

    let distance = dtw_distance_constrained(seq1, seq2, band_width);

    // Normalize by average length
    let avg_len = (seq1.len() + seq2.len()) as f32 / 2.0;
    distance / avg_len
}

/// Calculate normalized DTW distance.
///
/// Normalizes the DTW distance by the length of the warping path,
/// making it easier to compare distances between sequences of different lengths.
///
/// # Returns
///
/// The normalized DTW distance. Returns `f32::INFINITY` if either sequence is empty.
#[allow(dead_code)]
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
#[allow(dead_code)]
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

// =========================================================================
// Prototype Computation
// =========================================================================

/// Resample a sequence to a target length using linear interpolation.
///
/// This allows averaging examples of different lengths.
#[allow(dead_code)]
fn resample_sequence(seq: &Sequence, target_len: usize) -> Sequence {
    if seq.is_empty() || target_len == 0 {
        return Vec::new();
    }

    if seq.len() == target_len {
        return seq.clone();
    }

    let dim = seq[0].len();
    let mut result = Vec::with_capacity(target_len);

    for i in 0..target_len {
        // Map target index to source position
        let src_pos = (i as f32) * ((seq.len() - 1) as f32) / ((target_len - 1) as f32).max(1.0);
        let src_idx = src_pos.floor() as usize;
        let frac = src_pos - src_idx as f32;

        // Interpolate between frames
        let mut frame = vec![0.0; dim];
        if src_idx + 1 < seq.len() {
            for d in 0..dim {
                frame[d] = seq[src_idx][d] * (1.0 - frac) + seq[src_idx + 1][d] * frac;
            }
        } else {
            frame = seq[src_idx].clone();
        }
        result.push(frame);
    }

    result
}

/// Compute a prototype sequence from multiple examples.
///
/// The prototype is computed by:
/// 1. Resampling all examples to the median length
/// 2. Averaging corresponding frames
///
/// This creates a single "canonical" example that can be matched against,
/// reducing N comparisons to 1 per gesture.
///
/// Note: Currently unused - Wekinator-style recognition compares against
/// all examples instead of a prototype. Kept for future optimization.
///
/// # Returns
/// The prototype sequence, or an empty sequence if examples is empty.
#[allow(dead_code)]
pub fn compute_prototype(examples: &[Sequence]) -> Sequence {
    if examples.is_empty() {
        return Vec::new();
    }

    if examples.len() == 1 {
        return examples[0].clone();
    }

    // Find median length
    let mut lengths: Vec<usize> = examples.iter().map(|e| e.len()).collect();
    lengths.sort();
    let target_len = lengths[lengths.len() / 2];

    if target_len == 0 {
        return Vec::new();
    }

    // Resample all examples to target length
    let resampled: Vec<Sequence> = examples
        .iter()
        .map(|e| resample_sequence(e, target_len))
        .collect();

    // Average corresponding frames
    let dim = resampled[0][0].len();
    let n_examples = resampled.len() as f32;
    let mut prototype = Vec::with_capacity(target_len);

    for t in 0..target_len {
        let mut avg_frame = vec![0.0; dim];
        for example in &resampled {
            for d in 0..dim {
                avg_frame[d] += example[t][d];
            }
        }
        for d in 0..dim {
            avg_frame[d] /= n_examples;
        }
        prototype.push(avg_frame);
    }

    prototype
}

// =========================================================================
// Motion Energy / Activity Detection
// =========================================================================

/// Compute motion energy between two consecutive frames.
///
/// Motion energy is the sum of squared differences between frames,
/// representing how much movement occurred. Used for activity gating -
/// skipping DTW computation when the user is standing still.
///
/// # Returns
/// The motion energy (sum of squared velocities). Returns 0.0 if frames
/// have different lengths.
pub fn motion_energy(prev_frame: &Frame, curr_frame: &Frame) -> f32 {
    if prev_frame.len() != curr_frame.len() {
        return 0.0;
    }

    prev_frame
        .iter()
        .zip(curr_frame.iter())
        .map(|(a, b)| (b - a).powi(2))
        .sum()
}

/// Compute average motion energy over a sequence of frames.
///
/// Returns 0.0 if sequence has fewer than 2 frames.
#[allow(dead_code)]
pub fn average_motion_energy(frames: &Sequence) -> f32 {
    if frames.len() < 2 {
        return 0.0;
    }

    let total: f32 = frames
        .windows(2)
        .map(|w| motion_energy(&w[0], &w[1]))
        .sum();

    total / (frames.len() - 1) as f32
}

/// Check if a sequence of frames shows enough activity for gesture matching.
///
/// # Arguments
/// * `frames` - Recent frames to check
/// * `threshold` - Minimum average motion energy to consider "active"
///
/// # Returns
/// `true` if the sequence shows enough movement for gesture matching.
#[allow(dead_code)]
pub fn is_active(frames: &Sequence, threshold: f32) -> bool {
    average_motion_energy(frames) >= threshold
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
        // 66 dimensions (like MediaPipe skeleton data)
        let a: Vec<f32> = (0..66).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..66).map(|i| i as f32).collect();
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
    // Sakoe-Chiba Constrained DTW Tests
    // =========================================================================

    #[test]
    fn test_dtw_constrained_identical() {
        let seq = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]];
        let distance = dtw_distance_constrained(&seq, &seq, 2);
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn test_dtw_constrained_similar_to_unconstrained() {
        let seq1: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
        let seq2: Vec<Vec<f32>> = (0..20).map(|i| vec![(i as f32) + 0.5]).collect();

        let unconstrained = dtw_distance(&seq1, &seq2);
        let constrained = dtw_distance_constrained(&seq1, &seq2, 4); // 20% band

        // For similar sequences, constrained should be close to unconstrained
        // Allow 10% error for approximation
        assert!(
            (constrained - unconstrained).abs() / unconstrained < 0.1,
            "Constrained {} should be close to unconstrained {}",
            constrained,
            unconstrained
        );
    }

    #[test]
    fn test_dtw_constrained_handles_warping() {
        // Same pattern, different speeds
        let fast = vec![vec![0.0], vec![1.0], vec![2.0]];
        let slow = vec![vec![0.0], vec![0.0], vec![1.0], vec![1.0], vec![2.0], vec![2.0]];

        // With sufficient band width, should still match well
        let distance = dtw_distance_constrained(&fast, &slow, 3);
        assert!(distance < 1.0, "Should handle time warping, got {}", distance);
    }

    #[test]
    fn test_dtw_constrained_empty() {
        let seq: Sequence = vec![];
        let other = vec![vec![1.0]];
        assert_eq!(dtw_distance_constrained(&seq, &other, 2), f32::INFINITY);
    }

    #[test]
    fn test_dtw_constrained_normalized() {
        let seq1: Vec<Vec<f32>> = (0..30).map(|i| vec![i as f32]).collect();
        let seq2: Vec<Vec<f32>> = (0..30).map(|i| vec![(i as f32) + 1.0]).collect();

        let normalized = dtw_distance_constrained_normalized(&seq1, &seq2, 0.2);

        // Should be reasonably small for similar sequences
        assert!(normalized < 2.0, "Normalized distance should be small, got {}", normalized);
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
    // Prototype Computation Tests
    // =========================================================================

    #[test]
    fn test_resample_same_length() {
        let seq = vec![vec![0.0], vec![1.0], vec![2.0]];
        let resampled = resample_sequence(&seq, 3);
        assert_eq!(resampled, seq);
    }

    #[test]
    fn test_resample_shorter() {
        let seq = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let resampled = resample_sequence(&seq, 3);
        assert_eq!(resampled.len(), 3);
        assert_eq!(resampled[0], vec![0.0]);
        assert_eq!(resampled[2], vec![4.0]);
    }

    #[test]
    fn test_resample_longer() {
        let seq = vec![vec![0.0], vec![2.0]];
        let resampled = resample_sequence(&seq, 3);
        assert_eq!(resampled.len(), 3);
        assert_eq!(resampled[0], vec![0.0]);
        assert_eq!(resampled[1], vec![1.0]); // Interpolated
        assert_eq!(resampled[2], vec![2.0]);
    }

    #[test]
    fn test_resample_empty() {
        let seq: Sequence = vec![];
        let resampled = resample_sequence(&seq, 3);
        assert!(resampled.is_empty());
    }

    #[test]
    fn test_compute_prototype_single_example() {
        let examples = vec![
            vec![vec![1.0], vec![2.0], vec![3.0]],
        ];
        let prototype = compute_prototype(&examples);
        assert_eq!(prototype, examples[0]);
    }

    #[test]
    fn test_compute_prototype_identical_examples() {
        let example = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let examples = vec![example.clone(), example.clone(), example.clone()];
        let prototype = compute_prototype(&examples);
        assert_eq!(prototype, example);
    }

    #[test]
    fn test_compute_prototype_averages_frames() {
        let examples = vec![
            vec![vec![0.0], vec![0.0]],
            vec![vec![2.0], vec![2.0]],
        ];
        let prototype = compute_prototype(&examples);
        assert_eq!(prototype.len(), 2);
        assert_eq!(prototype[0], vec![1.0]); // Average of 0 and 2
        assert_eq!(prototype[1], vec![1.0]);
    }

    #[test]
    fn test_compute_prototype_different_lengths() {
        // Examples of different lengths - should resample to median
        let examples = vec![
            vec![vec![0.0], vec![1.0]],           // 2 frames
            vec![vec![0.0], vec![0.5], vec![1.0]], // 3 frames
            vec![vec![0.0], vec![0.25], vec![0.5], vec![0.75], vec![1.0]], // 5 frames
        ];
        let prototype = compute_prototype(&examples);
        // Median length is 3
        assert_eq!(prototype.len(), 3);
    }

    #[test]
    fn test_compute_prototype_empty() {
        let examples: Vec<Sequence> = vec![];
        let prototype = compute_prototype(&examples);
        assert!(prototype.is_empty());
    }

    #[test]
    fn test_prototype_matches_examples_well() {
        // Create examples with slight variations
        let examples = vec![
            vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]],
            vec![vec![0.1], vec![1.1], vec![2.1], vec![3.1]],
            vec![vec![-0.1], vec![0.9], vec![1.9], vec![2.9]],
        ];

        let prototype = compute_prototype(&examples);

        // Prototype should be close to all examples
        for example in &examples {
            let dist = dtw_distance(&prototype, example);
            assert!(dist < 1.0, "Prototype should be close to example, got {}", dist);
        }
    }

    // =========================================================================
    // Motion Energy / Activity Detection Tests
    // =========================================================================

    #[test]
    fn test_motion_energy_stationary() {
        let frame1 = vec![1.0, 2.0, 3.0];
        let frame2 = vec![1.0, 2.0, 3.0]; // Identical
        assert_eq!(motion_energy(&frame1, &frame2), 0.0);
    }

    #[test]
    fn test_motion_energy_moving() {
        let frame1 = vec![0.0, 0.0];
        let frame2 = vec![1.0, 1.0]; // Moved by 1 in each dimension
        // Energy = 1^2 + 1^2 = 2.0
        assert_eq!(motion_energy(&frame1, &frame2), 2.0);
    }

    #[test]
    fn test_motion_energy_different_lengths() {
        let frame1 = vec![1.0, 2.0];
        let frame2 = vec![1.0, 2.0, 3.0];
        assert_eq!(motion_energy(&frame1, &frame2), 0.0);
    }

    #[test]
    fn test_average_motion_energy_stationary() {
        let frames = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        assert_eq!(average_motion_energy(&frames), 0.0);
    }

    #[test]
    fn test_average_motion_energy_moving() {
        let frames = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        // Energy between each pair: (1-0)^2 = 1, (2-1)^2 = 1
        // Average = 2 / 2 = 1.0
        assert_eq!(average_motion_energy(&frames), 1.0);
    }

    #[test]
    fn test_average_motion_energy_single_frame() {
        let frames = vec![vec![1.0]];
        assert_eq!(average_motion_energy(&frames), 0.0);
    }

    #[test]
    fn test_is_active_stationary() {
        let frames = vec![
            vec![1.0, 2.0],
            vec![1.0, 2.0],
            vec![1.0, 2.0],
        ];
        assert!(!is_active(&frames, 0.1));
    }

    #[test]
    fn test_is_active_moving() {
        let frames = vec![
            vec![0.0],
            vec![1.0],
            vec![2.0],
        ];
        assert!(is_active(&frames, 0.5)); // Average energy is 1.0
        assert!(!is_active(&frames, 2.0)); // Threshold too high
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
