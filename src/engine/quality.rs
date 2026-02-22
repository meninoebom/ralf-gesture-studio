//! Example quality assessment for training feedback.
//!
//! After a training session completes, each new example is assessed against
//! existing examples to detect potential quality issues. Feedback is
//! informational and non-blocking — it preserves the dancer's flow state.
//!
//! Three checks run in priority order:
//! 1. **TooShort** — frame count < 50% of mean
//! 2. **TooStill** — total motion < 5% of mean motion
//! 3. **Outlier** — mean DTW distance to others > 3× inter-example mean

use serde::{Deserialize, Serialize};

use super::dtw::{dtw_distance_with_abandon, Sequence};

/// Quality issues detected in a training example.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum QualityIssue {
    /// The example has very little motion compared to others.
    TooStill { motion: f32, threshold: f32 },
    /// The example is statistically dissimilar to other examples.
    Outlier { distance: f32, threshold: f32 },
    /// The example has significantly fewer frames than others.
    TooShort {
        frame_count: usize,
        threshold: usize,
    },
}

impl QualityIssue {
    /// Human-readable summary for the frontend.
    pub fn message(&self) -> String {
        match self {
            QualityIssue::TooStill { .. } => "Very little movement detected".to_string(),
            QualityIssue::Outlier { .. } => "Looks different from other examples".to_string(),
            QualityIssue::TooShort { .. } => {
                "Significantly shorter than other examples".to_string()
            }
        }
    }

    /// Short label for the frontend.
    pub fn label(&self) -> &'static str {
        match self {
            QualityIssue::TooStill { .. } => "Low Motion",
            QualityIssue::Outlier { .. } => "Outlier",
            QualityIssue::TooShort { .. } => "Too Short",
        }
    }
}

/// Assess a single preprocessed example against existing preprocessed examples.
///
/// Returns the first quality issue detected (priority order), or `None` if OK.
/// Requires at least one existing example for comparison.
pub fn assess_example(new_example: &Sequence, existing: &[Sequence]) -> Option<QualityIssue> {
    if existing.is_empty() {
        return None;
    }

    // Check 1: TooShort — frame count < 50% of mean
    let mean_frames = existing.iter().map(|e| e.len() as f32).sum::<f32>() / existing.len() as f32;
    let short_threshold = (mean_frames * 0.5) as usize;
    if short_threshold > 0 && new_example.len() < short_threshold {
        return Some(QualityIssue::TooShort {
            frame_count: new_example.len(),
            threshold: short_threshold,
        });
    }

    // Check 2: TooStill — total motion < 5% of mean motion
    let new_motion = compute_total_motion(new_example);
    let mean_motion =
        existing.iter().map(compute_total_motion).sum::<f32>() / existing.len() as f32;
    let still_threshold = mean_motion * 0.05;
    if still_threshold > 0.0 && new_motion < still_threshold {
        return Some(QualityIssue::TooStill {
            motion: new_motion,
            threshold: still_threshold,
        });
    }

    // Check 3: Outlier — mean DTW distance > 3× inter-example mean (needs ≥2 existing)
    if existing.len() >= 2 {
        let mut inter_distances = Vec::new();
        for i in 0..existing.len() {
            for j in (i + 1)..existing.len() {
                let max_len = existing[i].len().max(existing[j].len());
                let band_width = ((max_len as f32) * 0.15).ceil() as usize;
                if let Some(d) = dtw_distance_with_abandon(
                    &existing[i], &existing[j], band_width, f32::INFINITY,
                ) {
                    if d.is_finite() {
                        inter_distances.push(d);
                    }
                }
            }
        }

        if !inter_distances.is_empty() {
            let inter_mean = inter_distances.iter().sum::<f32>() / inter_distances.len() as f32;
            let outlier_threshold = inter_mean * 3.0;

            let new_distances: Vec<f32> = existing
                .iter()
                .filter_map(|ex| {
                    let max_len = new_example.len().max(ex.len());
                    let band_width = ((max_len as f32) * 0.15).ceil() as usize;
                    dtw_distance_with_abandon(new_example, ex, band_width, f32::INFINITY)
                })
                .filter(|d| d.is_finite())
                .collect();

            if !new_distances.is_empty() {
                let new_mean = new_distances.iter().sum::<f32>() / new_distances.len() as f32;
                if new_mean > outlier_threshold {
                    return Some(QualityIssue::Outlier {
                        distance: new_mean,
                        threshold: outlier_threshold,
                    });
                }
            }
        }
    }

    None
}

/// Compute the consistency of a gesture's examples (σ/μ ratio of pairwise distances).
///
/// Low values (< 0.3) = consistent examples. High values (> 0.6) = noisy/inconsistent.
/// Returns `None` if fewer than 2 examples.
pub fn compute_gesture_consistency(examples: &[Sequence]) -> Option<f32> {
    if examples.len() < 2 {
        return None;
    }

    // Downsample for speed (matches threshold stats factor)
    let downsample_factor = 4;
    let downsampled: Vec<Sequence> = examples
        .iter()
        .map(|seq| seq.iter().step_by(downsample_factor).cloned().collect())
        .collect();

    let mut distances = Vec::new();
    for i in 0..downsampled.len() {
        for j in (i + 1)..downsampled.len() {
            let max_len = downsampled[i].len().max(downsampled[j].len());
            let band_width = ((max_len as f32) * 0.15).ceil() as usize;
            if let Some(d) = dtw_distance_with_abandon(
                &downsampled[i],
                &downsampled[j],
                band_width,
                f32::INFINITY,
            ) {
                if d.is_finite() {
                    distances.push(d);
                }
            }
        }
    }

    if distances.is_empty() {
        return None;
    }

    let mean = distances.iter().sum::<f32>() / distances.len() as f32;
    if mean < f32::EPSILON {
        return Some(0.0); // identical examples = perfect consistency
    }

    let variance = distances.iter().map(|d| (d - mean).powi(2)).sum::<f32>() / distances.len() as f32;
    let std = variance.sqrt();

    Some(std / mean)
}

/// Compute total Euclidean displacement across consecutive frames.
fn compute_total_motion(seq: &Sequence) -> f32 {
    if seq.len() < 2 {
        return 0.0;
    }
    let mut total = 0.0_f32;
    for pair in seq.windows(2) {
        let sum_sq: f32 = pair[0]
            .iter()
            .zip(pair[1].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        total += sum_sq.sqrt();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_moving_sequence(n_frames: usize, speed: f32) -> Sequence {
        (0..n_frames).map(|i| vec![i as f32 * speed, 0.0]).collect()
    }

    #[test]
    fn test_no_issues_with_good_example() {
        let existing = vec![make_moving_sequence(30, 1.0), make_moving_sequence(30, 1.1)];
        let new_ex = make_moving_sequence(30, 0.9);
        assert!(assess_example(&new_ex, &existing).is_none());
    }

    #[test]
    fn test_no_assessment_without_existing() {
        let new_ex = make_moving_sequence(30, 1.0);
        assert!(assess_example(&new_ex, &[]).is_none());
    }

    #[test]
    fn test_too_short_detected() {
        let existing = vec![make_moving_sequence(30, 1.0), make_moving_sequence(30, 1.0)];
        let short = make_moving_sequence(10, 1.0); // 10 < 30 * 0.5 = 15
        let result = assess_example(&short, &existing);
        assert!(matches!(result, Some(QualityIssue::TooShort { .. })));
    }

    #[test]
    fn test_too_still_detected() {
        let existing = vec![make_moving_sequence(30, 1.0), make_moving_sequence(30, 1.0)];
        let still: Sequence = vec![vec![5.0, 5.0]; 30]; // zero motion
        let result = assess_example(&still, &existing);
        assert!(matches!(result, Some(QualityIssue::TooStill { .. })));
    }

    #[test]
    fn test_outlier_detected() {
        let existing = vec![
            make_moving_sequence(30, 1.0),
            make_moving_sequence(30, 1.1),
            make_moving_sequence(30, 0.9),
        ];
        let outlier = make_moving_sequence(30, 100.0);
        let result = assess_example(&outlier, &existing);
        assert!(matches!(result, Some(QualityIssue::Outlier { .. })));
    }

    #[test]
    fn test_outlier_needs_two_existing() {
        let existing = vec![make_moving_sequence(30, 1.0)];
        let different = make_moving_sequence(30, 100.0);
        // With only 1 existing, outlier check requires >= 2, so it's skipped
        let result = assess_example(&different, &existing);
        assert!(!matches!(result, Some(QualityIssue::Outlier { .. })));
    }

    #[test]
    fn test_too_short_has_priority() {
        // An example that is both too short AND too still should report TooShort
        let existing = vec![make_moving_sequence(30, 1.0), make_moving_sequence(30, 1.0)];
        let short_still: Sequence = vec![vec![5.0, 5.0]; 5]; // 5 frames, no motion
        let result = assess_example(&short_still, &existing);
        assert!(matches!(result, Some(QualityIssue::TooShort { .. })));
    }

    #[test]
    fn test_compute_total_motion() {
        let still: Sequence = vec![vec![1.0, 2.0]; 10];
        assert_eq!(compute_total_motion(&still), 0.0);

        let moving: Sequence = (0..10).map(|i| vec![i as f32, 0.0]).collect();
        let motion = compute_total_motion(&moving);
        assert!((motion - 9.0).abs() < 0.001); // 9 steps of 1.0 each

        let empty: Sequence = Vec::new();
        assert_eq!(compute_total_motion(&empty), 0.0);
        let single: Sequence = vec![vec![1.0]];
        assert_eq!(compute_total_motion(&single), 0.0);
    }

    #[test]
    fn test_consistency_returns_none_for_few_examples() {
        assert!(compute_gesture_consistency(&[]).is_none());
        assert!(compute_gesture_consistency(&[make_moving_sequence(30, 1.0)]).is_none());
    }

    #[test]
    fn test_consistency_low_for_similar_examples() {
        let examples = vec![
            make_moving_sequence(30, 1.0),
            make_moving_sequence(30, 1.05),
            make_moving_sequence(30, 0.95),
        ];
        let c = compute_gesture_consistency(&examples).unwrap();
        assert!(c < 0.5, "similar examples should have low consistency ratio: got {}", c);
    }

    #[test]
    fn test_consistency_high_for_mixed_examples() {
        let examples = vec![
            make_moving_sequence(30, 1.0),
            make_moving_sequence(30, 1.0),
            make_moving_sequence(30, 50.0), // very different
        ];
        let c = compute_gesture_consistency(&examples).unwrap();
        assert!(c > 0.3, "mixed examples should have higher consistency ratio: got {}", c);
    }

    #[test]
    fn test_quality_issue_labels() {
        let ts = QualityIssue::TooShort {
            frame_count: 5,
            threshold: 15,
        };
        assert_eq!(ts.label(), "Too Short");

        let tst = QualityIssue::TooStill {
            motion: 0.1,
            threshold: 2.0,
        };
        assert_eq!(tst.label(), "Low Motion");

        let out = QualityIssue::Outlier {
            distance: 100.0,
            threshold: 30.0,
        };
        assert_eq!(out.label(), "Outlier");
    }
}
