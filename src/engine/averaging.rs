//! Template averaging for gesture recognition.
//!
//! Medoid selection picks the example closest to the center of the set,
//! reducing recognition from O(N×DTW) to O(1×DTW) per gesture.

use super::dtw::{dtw_distance_with_abandon, Sequence};

/// Find the medoid: the example with lowest mean distance to all others.
/// Returns the index of the medoid, or None if fewer than 3 examples.
pub fn compute_medoid(examples: &[Sequence]) -> Option<usize> {
    if examples.len() < 3 {
        return None;
    }

    // Compute mean distance for each example to all others
    let mut best_idx = 0;
    let mut best_mean = f32::MAX;

    for i in 0..examples.len() {
        let mut total = 0.0_f32;
        let mut count = 0;
        for j in 0..examples.len() {
            if i == j {
                continue;
            }
            // Use banded DTW matching recognizer params (band=0.15, no abandon)
            let band_width = ((examples[i].len().max(examples[j].len()) as f32) * 0.15).ceil()
                as usize;
            if let Some(d) =
                dtw_distance_with_abandon(&examples[i], &examples[j], band_width, f32::INFINITY)
            {
                total += d;
                count += 1;
            }
        }
        if count > 0 {
            let mean = total / count as f32;
            if mean < best_mean {
                best_mean = mean;
                best_idx = i;
            }
        }
    }

    Some(best_idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_medoid_returns_none_for_few_examples() {
        let examples: Vec<Sequence> = vec![vec![vec![1.0, 2.0]; 10]];
        assert!(compute_medoid(&examples).is_none());

        let examples: Vec<Sequence> = vec![vec![vec![1.0]; 10], vec![vec![2.0]; 10]];
        assert!(compute_medoid(&examples).is_none());
    }

    #[test]
    fn test_medoid_picks_central_example() {
        // Three examples: [1.0], [1.1], [5.0] — medoid should be [1.0] or [1.1]
        let examples: Vec<Sequence> = vec![
            (0..20).map(|i| vec![i as f32 * 1.0]).collect(),
            (0..20).map(|i| vec![i as f32 * 1.1]).collect(),
            (0..20).map(|i| vec![i as f32 * 5.0]).collect(), // outlier
        ];
        let idx = compute_medoid(&examples).unwrap();
        assert!(idx < 2, "medoid should be one of the two similar examples, got {}", idx);
    }

    #[test]
    fn test_medoid_with_identical_examples() {
        let examples: Vec<Sequence> = vec![
            vec![vec![1.0, 2.0]; 15],
            vec![vec![1.0, 2.0]; 15],
            vec![vec![1.0, 2.0]; 15],
        ];
        // All identical — any index is valid
        let idx = compute_medoid(&examples).unwrap();
        assert!(idx < 3);
    }
}
