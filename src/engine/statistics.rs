//! Statistical threshold computation for gesture recognition.
//!
//! Implements the GRT-style μ+σ approach for automatic threshold calibration.
//! Instead of manually tuning thresholds per gesture, this module computes
//! statistics from training examples and derives thresholds automatically.
//!
//! ## Algorithm
//!
//! For each gesture with N training examples:
//! 1. Compute DTW distances between all pairs of examples
//! 2. Calculate mean (μ) and standard deviation (σ) of these distances
//! 3. Set threshold = μ + σ × coefficient (default coefficient = 2.0)
//!
//! This approach adapts to each gesture's natural variability:
//! - Simple, consistent gestures get tighter thresholds
//! - Complex, variable gestures get looser thresholds
//!
//! ## Reference
//!
//! Based on the Gesture Recognition Toolkit (GRT) by Nick Gillian:
//! https://github.com/nickgillian/grt

use super::dtw::{dtw_distance, Sequence};

/// Statistics computed from training examples for threshold calibration.
#[derive(Debug, Clone)]
pub struct ThresholdStats {
    /// Mean distance between training examples
    pub mean: f32,
    /// Standard deviation of distances
    pub std: f32,
    /// Recommended threshold (μ + σ × coefficient)
    pub threshold: f32,
    /// Number of pairwise distances computed
    pub sample_count: usize,
}

/// Compute threshold statistics from a set of training examples.
///
/// Calculates the mean and standard deviation of DTW distances between
/// all pairs of training examples, then derives a threshold using the
/// GRT-style μ+σ approach.
///
/// # Arguments
/// * `examples` - Training examples (sequences of frames)
/// * `coefficient` - Multiplier for standard deviation (default: 2.0)
///                   Higher = more permissive, Lower = stricter
///
/// # Returns
/// * `Some(ThresholdStats)` if at least 2 examples are provided
/// * `None` if fewer than 2 examples (can't compute pairwise distances)
///
/// # Example
/// ```ignore
/// let examples = vec![seq1, seq2, seq3];
/// if let Some(stats) = compute_threshold_stats(&examples, 2.0) {
///     println!("Threshold: {} (μ={}, σ={})", stats.threshold, stats.mean, stats.std);
/// }
/// ```
pub fn compute_threshold_stats(examples: &[Sequence], coefficient: f32) -> Option<ThresholdStats> {
    if examples.len() < 2 {
        return None;
    }

    // Compute all pairwise DTW distances
    let mut distances: Vec<f32> = Vec::new();
    for i in 0..examples.len() {
        for j in (i + 1)..examples.len() {
            let dist = dtw_distance(&examples[i], &examples[j]);
            if dist.is_finite() {
                distances.push(dist);
            }
        }
    }

    if distances.is_empty() {
        return None;
    }

    // Compute mean
    let n = distances.len() as f32;
    let mean = distances.iter().sum::<f32>() / n;

    // Compute standard deviation
    let variance = distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    // Compute threshold: μ + σ × coefficient
    let threshold = mean + std * coefficient;

    Some(ThresholdStats {
        mean,
        std,
        threshold,
        sample_count: distances.len(),
    })
}

/// Compute threshold statistics with downsampling for performance.
///
/// For gestures with many examples or long sequences, this version
/// downsamples the examples before computing DTW distances.
///
/// # Arguments
/// * `examples` - Training examples (sequences of frames)
/// * `coefficient` - Multiplier for standard deviation (default: 2.0)
/// * `downsample_factor` - Take every Nth frame (e.g., 4 for 60fps -> 15fps)
///
/// # Returns
/// Same as `compute_threshold_stats`
pub fn compute_threshold_stats_downsampled(
    examples: &[Sequence],
    coefficient: f32,
    downsample_factor: usize,
) -> Option<ThresholdStats> {
    if examples.len() < 2 {
        return None;
    }

    // Downsample examples
    let downsampled: Vec<Sequence> = examples
        .iter()
        .map(|seq| {
            if downsample_factor <= 1 {
                seq.clone()
            } else {
                seq.iter().step_by(downsample_factor).cloned().collect()
            }
        })
        .collect();

    compute_threshold_stats(&downsampled, coefficient)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sequence(values: &[f32]) -> Sequence {
        values.iter().map(|&v| vec![v]).collect()
    }

    #[test]
    fn test_compute_threshold_stats_basic() {
        let examples = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.1, 1.1, 2.1]),
            make_sequence(&[0.0, 1.0, 2.0]),
        ];

        let stats = compute_threshold_stats(&examples, 2.0);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert!(stats.mean >= 0.0);
        assert!(stats.std >= 0.0);
        assert!(stats.threshold >= stats.mean);
        assert_eq!(stats.sample_count, 3); // C(3,2) = 3 pairs
    }

    #[test]
    fn test_compute_threshold_stats_identical_examples() {
        let example = make_sequence(&[1.0, 2.0, 3.0]);
        let examples = vec![example.clone(), example.clone(), example.clone()];

        let stats = compute_threshold_stats(&examples, 2.0);
        assert!(stats.is_some());

        let stats = stats.unwrap();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std, 0.0);
        assert_eq!(stats.threshold, 0.0);
    }

    #[test]
    fn test_compute_threshold_stats_insufficient_examples() {
        // 0 examples
        let stats = compute_threshold_stats(&[], 2.0);
        assert!(stats.is_none());

        // 1 example
        let examples = vec![make_sequence(&[1.0, 2.0])];
        let stats = compute_threshold_stats(&examples, 2.0);
        assert!(stats.is_none());
    }

    #[test]
    fn test_coefficient_affects_threshold() {
        let examples = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.5, 1.5, 2.5]),
            make_sequence(&[1.0, 2.0, 3.0]),
        ];

        let stats_low = compute_threshold_stats(&examples, 1.0).unwrap();
        let stats_high = compute_threshold_stats(&examples, 3.0).unwrap();

        // Same mean and std, but different thresholds
        assert!((stats_low.mean - stats_high.mean).abs() < 0.001);
        assert!((stats_low.std - stats_high.std).abs() < 0.001);
        assert!(stats_high.threshold > stats_low.threshold);
    }

    #[test]
    fn test_downsampled_computation() {
        let examples = vec![
            make_sequence(&[0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            make_sequence(&[0.1, 0.35, 0.6, 0.85, 1.1, 1.35, 1.6, 1.85, 2.1]),
        ];

        let stats_full = compute_threshold_stats(&examples, 2.0);
        let stats_ds = compute_threshold_stats_downsampled(&examples, 2.0, 2);

        assert!(stats_full.is_some());
        assert!(stats_ds.is_some());

        // Downsampled should produce different (but related) results
        let sf = stats_full.unwrap();
        let sd = stats_ds.unwrap();

        // Both should be non-negative
        assert!(sf.mean >= 0.0);
        assert!(sd.mean >= 0.0);
    }

    #[test]
    fn test_sample_count() {
        // 4 examples = C(4,2) = 6 pairs
        let examples = vec![
            make_sequence(&[1.0]),
            make_sequence(&[2.0]),
            make_sequence(&[3.0]),
            make_sequence(&[4.0]),
        ];

        let stats = compute_threshold_stats(&examples, 2.0).unwrap();
        assert_eq!(stats.sample_count, 6);
    }
}
