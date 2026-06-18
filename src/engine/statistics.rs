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

use super::dtw::{dtw_distance, dtw_distance_with_abandon, sdtw_distance, Sequence};

/// Statistics computed from training examples for threshold calibration.
#[derive(Debug, Clone)]
pub struct ThresholdStats {
    /// Mean distance between training examples
    pub mean: f32,
    /// Standard deviation of distances
    pub std: f32,
    /// Recommended threshold (μ + σ × coefficient)
    #[allow(dead_code)]
    pub threshold: f32,
    /// Number of pairwise distances computed
    #[allow(dead_code)]
    pub sample_count: usize,
    /// Indices of examples flagged as outliers (mean distance > overall_mean + 2*sigma)
    pub outlier_indices: Vec<usize>,
    /// F1 score when threshold was computed via F1 optimization (None for μ+σ)
    pub f1_score: Option<f32>,
}

/// Compute threshold statistics from a set of training examples.
///
/// Calculates the mean and standard deviation of DTW distances between
/// all pairs of training examples, then derives a threshold using the
/// GRT-style μ+σ approach.
///
/// # Arguments
/// * `examples` - Training examples (sequences of frames)
/// * `coefficient` - Multiplier for standard deviation (default: 2.0).
///   Higher = more permissive, Lower = stricter
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

    let n_examples = examples.len();

    // Compute all pairwise DTW distances
    let mut all_distances: Vec<f32> = Vec::new();

    for i in 0..n_examples {
        for j in (i + 1)..n_examples {
            let dist = dtw_distance(&examples[i], &examples[j]);
            if dist.is_finite() {
                all_distances.push(dist);
            }
        }
    }

    if all_distances.is_empty() {
        return None;
    }

    // Compute mean
    let n = all_distances.len() as f32;
    let mean = all_distances.iter().sum::<f32>() / n;

    // Compute standard deviation
    let variance = all_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f32>()
        / n;
    let std = variance.sqrt();

    // Compute threshold: μ + σ × coefficient
    let threshold = mean + std * coefficient;

    Some(ThresholdStats {
        mean,
        std,
        threshold,
        sample_count: all_distances.len(),
        outlier_indices: vec![],
        f1_score: None,
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
#[allow(dead_code)]
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

/// Compute threshold statistics using the same DTW parameters as recognition.
///
/// This ensures thresholds are calibrated in the same distance domain as
/// real-time recognition (Sakoe-Chiba banded DTW on downsampled sequences).
/// Also performs outlier detection: examples whose mean distance to others
/// exceeds 2σ above the overall mean are flagged.
///
/// # Arguments
/// * `examples` - Training examples (sequences of frames)
/// * `coefficient` - Multiplier for standard deviation (default: 2.0)
/// * `downsample_factor` - Take every Nth frame (must match recognizer's downsample)
/// * `sakoe_chiba_band` - Fractional band width (must match recognizer's band)
pub fn compute_threshold_stats_banded(
    examples: &[Sequence],
    coefficient: f32,
    downsample_factor: usize,
    sakoe_chiba_band: f32,
) -> Option<ThresholdStats> {
    if examples.len() < 2 {
        return None;
    }

    // Downsample examples to match recognition resolution
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

    let n_examples = downsampled.len();

    // Compute all pairwise banded DTW distances (once, store in matrix)
    // distance_matrix[i][j] stores dist between example i and j (j > i)
    let mut all_distances: Vec<f32> = Vec::new();
    let mut per_example_sums: Vec<f32> = vec![0.0; n_examples];
    let mut per_example_counts: Vec<usize> = vec![0; n_examples];

    for i in 0..n_examples {
        for j in (i + 1)..n_examples {
            let max_len = downsampled[i].len().max(downsampled[j].len());
            let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;

            if let Some(dist) = dtw_distance_with_abandon(
                &downsampled[i],
                &downsampled[j],
                band_width,
                f32::INFINITY, // No early abandon during calibration
            ) {
                if dist.is_finite() {
                    all_distances.push(dist);
                    per_example_sums[i] += dist;
                    per_example_sums[j] += dist;
                    per_example_counts[i] += 1;
                    per_example_counts[j] += 1;
                }
            }
        }
    }

    if all_distances.is_empty() {
        return None;
    }

    // Per-example mean distances (for outlier detection)
    let per_example_means: Vec<f32> = (0..n_examples)
        .map(|i| {
            if per_example_counts[i] == 0 {
                f32::INFINITY
            } else {
                per_example_sums[i] / per_example_counts[i] as f32
            }
        })
        .collect();

    // Outlier detection: flag examples whose mean distance > overall_mean + 2*std
    let finite_means: Vec<f32> = per_example_means
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    let pem_mean = finite_means.iter().sum::<f32>() / finite_means.len() as f32;
    let pem_std = {
        let var = finite_means
            .iter()
            .map(|d| (d - pem_mean).powi(2))
            .sum::<f32>()
            / finite_means.len() as f32;
        var.sqrt()
    };

    let outlier_cutoff = pem_mean + 2.0 * pem_std;
    let outlier_indices: Vec<usize> = per_example_means
        .iter()
        .enumerate()
        .filter(|(_, &m)| m > outlier_cutoff && m.is_finite())
        .map(|(i, _)| i)
        .collect();

    // Compute final stats, excluding outlier pairs if any
    let final_distances: Vec<f32> = if outlier_indices.is_empty() {
        all_distances.clone()
    } else {
        // Recompute from stored distances, skipping pairs involving outliers
        let mut clean = Vec::new();
        let mut idx = 0;
        for i in 0..n_examples {
            for j in (i + 1)..n_examples {
                if idx < all_distances.len() {
                    if !outlier_indices.contains(&i) && !outlier_indices.contains(&j) {
                        clean.push(all_distances[idx]);
                    }
                    idx += 1;
                }
            }
        }
        if clean.is_empty() {
            all_distances.clone() // Fall back if all are outliers
        } else {
            clean
        }
    };

    let n = final_distances.len() as f32;
    let mean = final_distances.iter().sum::<f32>() / n;
    let variance = final_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f32>()
        / n;
    let std = variance.sqrt();
    let threshold = mean + std * coefficient;

    Some(ThresholdStats {
        mean,
        std,
        threshold,
        sample_count: all_distances.len(),
        outlier_indices,
        f1_score: None,
    })
}

/// Compute threshold statistics using subsequence DTW for pairwise distances.
///
/// Same as `compute_threshold_stats_banded` but uses sDTW, which finds the
/// best-matching subsequence between each pair. This should be used when
/// recognition will use sDTW for distance computation.
pub fn compute_threshold_stats_sdtw(
    examples: &[Sequence],
    coefficient: f32,
    downsample_factor: usize,
    sakoe_chiba_band: f32,
) -> Option<ThresholdStats> {
    if examples.len() < 2 {
        return None;
    }

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

    let n_examples = downsampled.len();
    let mut all_distances: Vec<f32> = Vec::new();
    let mut per_example_sums: Vec<f32> = vec![0.0; n_examples];
    let mut per_example_counts: Vec<usize> = vec![0; n_examples];

    for i in 0..n_examples {
        for j in (i + 1)..n_examples {
            let max_len = downsampled[i].len().max(downsampled[j].len());
            let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
            if let Some(dist) =
                sdtw_distance(&downsampled[i], &downsampled[j], band_width, f32::INFINITY)
            {
                if dist.is_finite() {
                    all_distances.push(dist);
                    per_example_sums[i] += dist;
                    per_example_sums[j] += dist;
                    per_example_counts[i] += 1;
                    per_example_counts[j] += 1;
                }
            }
        }
    }

    if all_distances.is_empty() {
        return None;
    }

    let per_example_means: Vec<f32> = (0..n_examples)
        .map(|i| {
            if per_example_counts[i] == 0 {
                f32::INFINITY
            } else {
                per_example_sums[i] / per_example_counts[i] as f32
            }
        })
        .collect();

    let finite_means: Vec<f32> = per_example_means
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    let pem_mean = finite_means.iter().sum::<f32>() / finite_means.len() as f32;
    let pem_std = {
        let var = finite_means
            .iter()
            .map(|d| (d - pem_mean).powi(2))
            .sum::<f32>()
            / finite_means.len() as f32;
        var.sqrt()
    };

    let outlier_cutoff = pem_mean + 2.0 * pem_std;
    let outlier_indices: Vec<usize> = per_example_means
        .iter()
        .enumerate()
        .filter(|(_, &m)| m > outlier_cutoff && m.is_finite())
        .map(|(i, _)| i)
        .collect();

    let final_distances: Vec<f32> = if outlier_indices.is_empty() {
        all_distances.clone()
    } else {
        let mut clean = Vec::new();
        let mut idx = 0;
        for i in 0..n_examples {
            for j in (i + 1)..n_examples {
                if idx < all_distances.len() {
                    if !outlier_indices.contains(&i) && !outlier_indices.contains(&j) {
                        clean.push(all_distances[idx]);
                    }
                    idx += 1;
                }
            }
        }
        if clean.is_empty() {
            all_distances.clone()
        } else {
            clean
        }
    };

    let n = final_distances.len() as f32;
    let mean = final_distances.iter().sum::<f32>() / n;
    let variance = final_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f32>()
        / n;
    let std = variance.sqrt();
    let threshold = mean + std * coefficient;

    Some(ThresholdStats {
        mean,
        std,
        threshold,
        sample_count: all_distances.len(),
        outlier_indices,
        f1_score: None,
    })
}

/// Compute F1-optimized threshold using positive and negative distance distributions.
///
/// Instead of the heuristic μ+σ approach, this sweeps candidate thresholds against
/// both positive distances (within-gesture pairwise DTW) and negative distances
/// (cross-gesture DTW), picking the threshold that maximizes F1-score.
///
/// # Arguments
/// * `positives` - Training examples for this gesture
/// * `negatives` - Examples from OTHER gestures (used as negative samples)
/// * `downsample_factor` - Take every Nth frame (must match recognizer's downsample)
/// * `sakoe_chiba_band` - Fractional band width (must match recognizer's band)
/// * `coefficient` - Fallback μ+σ coefficient if F1 optimization fails
///
/// # Returns
/// * `Some(ThresholdStats)` with F1-optimal threshold, or μ+σ fallback
/// * `None` if fewer than 2 positive examples
pub fn compute_threshold_f1(
    positives: &[Sequence],
    negatives: &[Sequence],
    downsample_factor: usize,
    sakoe_chiba_band: f32,
    coefficient: f32,
) -> Option<ThresholdStats> {
    if positives.len() < 2 {
        return None;
    }

    // Downsample all examples
    let ds_pos: Vec<Sequence> = positives
        .iter()
        .map(|seq| {
            if downsample_factor <= 1 {
                seq.clone()
            } else {
                seq.iter().step_by(downsample_factor).cloned().collect()
            }
        })
        .collect();

    let ds_neg: Vec<Sequence> = negatives
        .iter()
        .map(|seq| {
            if downsample_factor <= 1 {
                seq.clone()
            } else {
                seq.iter().step_by(downsample_factor).cloned().collect()
            }
        })
        .collect();

    // 1. Positive distances: pairwise sDTW within this gesture's examples
    let mut positive_distances: Vec<f32> = Vec::new();
    let mut per_example_sums: Vec<f32> = vec![0.0; ds_pos.len()];
    let mut per_example_counts: Vec<usize> = vec![0; ds_pos.len()];

    for i in 0..ds_pos.len() {
        for j in (i + 1)..ds_pos.len() {
            let max_len = ds_pos[i].len().max(ds_pos[j].len());
            let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
            if let Some(dist) = sdtw_distance(&ds_pos[i], &ds_pos[j], band_width, f32::INFINITY) {
                if dist.is_finite() {
                    positive_distances.push(dist);
                    per_example_sums[i] += dist;
                    per_example_sums[j] += dist;
                    per_example_counts[i] += 1;
                    per_example_counts[j] += 1;
                }
            }
        }
    }

    if positive_distances.is_empty() {
        return None;
    }

    // Outlier detection (same as existing functions)
    let per_example_means: Vec<f32> = (0..ds_pos.len())
        .map(|i| {
            if per_example_counts[i] == 0 {
                f32::INFINITY
            } else {
                per_example_sums[i] / per_example_counts[i] as f32
            }
        })
        .collect();

    let finite_means: Vec<f32> = per_example_means
        .iter()
        .copied()
        .filter(|d| d.is_finite())
        .collect();
    let pem_mean = finite_means.iter().sum::<f32>() / finite_means.len() as f32;
    let pem_std = {
        let var = finite_means
            .iter()
            .map(|d| (d - pem_mean).powi(2))
            .sum::<f32>()
            / finite_means.len() as f32;
        var.sqrt()
    };
    let outlier_cutoff = pem_mean + 2.0 * pem_std;
    let outlier_indices: Vec<usize> = per_example_means
        .iter()
        .enumerate()
        .filter(|(_, &m)| m > outlier_cutoff && m.is_finite())
        .map(|(i, _)| i)
        .collect();

    // Compute mean/std of positive distances (for stats reporting)
    let n = positive_distances.len() as f32;
    let mean = positive_distances.iter().sum::<f32>() / n;
    let variance = positive_distances
        .iter()
        .map(|d| (d - mean).powi(2))
        .sum::<f32>()
        / n;
    let std = variance.sqrt();

    // 2. If no negatives, fall back to μ+σ
    if ds_neg.is_empty() {
        return Some(ThresholdStats {
            mean,
            std,
            threshold: mean + std * coefficient,
            sample_count: positive_distances.len(),
            outlier_indices,
            f1_score: None,
        });
    }

    // 3. Negative distances: each negative example against each positive example
    let mut negative_distances: Vec<f32> = Vec::new();
    for neg in &ds_neg {
        for pos in &ds_pos {
            let max_len = neg.len().max(pos.len());
            let band_width = ((max_len as f32) * sakoe_chiba_band).ceil() as usize;
            if let Some(dist) = sdtw_distance(neg, pos, band_width, f32::INFINITY) {
                if dist.is_finite() {
                    negative_distances.push(dist);
                }
            }
        }
    }

    // If no valid negative distances, fall back to μ+σ
    if negative_distances.is_empty() {
        return Some(ThresholdStats {
            mean,
            std,
            threshold: mean + std * coefficient,
            sample_count: positive_distances.len(),
            outlier_indices,
            f1_score: None,
        });
    }

    // 4. Threshold sweep: 200 steps across the combined range
    let all_min = positive_distances
        .iter()
        .chain(negative_distances.iter())
        .cloned()
        .fold(f32::INFINITY, f32::min);
    let all_max = positive_distances
        .iter()
        .chain(negative_distances.iter())
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);

    let steps = 200;
    let step_size = (all_max - all_min) / steps as f32;

    let mut best_f1 = 0.0_f32;
    let mut best_threshold = mean + std * coefficient; // fallback

    for s in 0..=steps {
        let candidate = all_min + step_size * s as f32;

        // TP: positive distances below threshold (correctly accepted)
        let tp = positive_distances
            .iter()
            .filter(|&&d| d <= candidate)
            .count() as f32;
        // FP: negative distances below threshold (incorrectly accepted)
        let fp = negative_distances
            .iter()
            .filter(|&&d| d <= candidate)
            .count() as f32;
        // FN: positive distances above threshold (incorrectly rejected)
        let fn_ = positive_distances
            .iter()
            .filter(|&&d| d > candidate)
            .count() as f32;

        let denom = 2.0 * tp + fp + fn_;
        if denom > 0.0 {
            let f1 = 2.0 * tp / denom;
            if f1 > best_f1 {
                best_f1 = f1;
                best_threshold = candidate;
            }
        }
    }

    Some(ThresholdStats {
        mean,
        std,
        threshold: best_threshold,
        sample_count: positive_distances.len(),
        outlier_indices,
        f1_score: Some(best_f1),
    })
}

/// Downsample factor for confusion detection (matches recognition downsampling).
const CONFUSION_DOWNSAMPLE: usize = 4;

/// Recentred flag cutoff for confusion detection. A pair is flagged when the
/// cross-gesture distance is below `max_threshold * CONFUSION_FLAG_FACTOR`.
///
/// sDTW free-start matches the common idle/standing portions of any two
/// gestures, so cross-distances run systematically smaller than the
/// intra-gesture thresholds — the old `cross < max_threshold` cutoff would flag
/// nearly every pair. Calibrated empirically: on the well-separated benchmark
/// trio the tightest pair sits at ~0.98× its threshold (must NOT flag), while a
/// genuinely confusable vocabulary (Arm Sweep / jump / spin) sits at
/// 0.72–0.85× (must flag). 0.9 splits them cleanly.
const CONFUSION_FLAG_FACTOR: f32 = 0.9;

/// Detect pairs of gestures whose training examples are too similar.
///
/// For each pair (A, B), computes the mean cross-gesture distance between A's
/// and B's examples and flags the pair when that distance falls below
/// `max(threshold_A, threshold_B) * CONFUSION_FLAG_FACTOR`.
///
/// The distance metric matches recognition: examples are downsampled by
/// [`CONFUSION_DOWNSAMPLE`] and compared with subsequence DTW (sDTW), the same
/// metric the recognition thresholds are calibrated against. Using full-
/// resolution standard DTW here (the previous behaviour) compared a different
/// metric against sDTW-calibrated thresholds — the exact calibration/recognition
/// mismatch the system forbids elsewhere. sDTW is asymmetric (free start), so
/// each example pair is averaged over both directions. Downsampling also makes
/// each pairwise call ~16× cheaper.
///
/// # Arguments
/// * `gestures` - List of (preprocessed examples, threshold) per gesture
///
/// # Returns
/// Vec of (gesture_index_a, gesture_index_b, overlap_ratio) where
/// overlap_ratio = max_threshold / cross_distance (higher = more overlap).
pub fn detect_confusion_pairs(gestures: &[(Vec<Sequence>, f32)]) -> Vec<(usize, usize, f32)> {
    // Downsample every example once, up front.
    let downsampled: Vec<Vec<Sequence>> = gestures
        .iter()
        .map(|(examples, _)| {
            examples
                .iter()
                .map(|seq| seq.iter().step_by(CONFUSION_DOWNSAMPLE).cloned().collect())
                .collect()
        })
        .collect();

    let mut pairs = Vec::new();

    for i in 0..gestures.len() {
        for j in (i + 1)..gestures.len() {
            let examples_a = &downsampled[i];
            let examples_b = &downsampled[j];

            if examples_a.is_empty() || examples_b.is_empty() {
                continue;
            }

            // Mean cross-gesture sDTW distance, averaged over both directions.
            let mut sum = 0.0f32;
            let mut count = 0usize;
            for ea in examples_a.iter() {
                for eb in examples_b.iter() {
                    let max_len = ea.len().max(eb.len());
                    let band_width = ((max_len as f32) * 0.15).ceil() as usize;
                    let ab = sdtw_distance(ea, eb, band_width, f32::INFINITY);
                    let ba = sdtw_distance(eb, ea, band_width, f32::INFINITY);
                    if let (Some(d_ab), Some(d_ba)) = (ab, ba) {
                        if d_ab.is_finite() && d_ba.is_finite() {
                            sum += (d_ab + d_ba) / 2.0;
                            count += 1;
                        }
                    }
                }
            }

            if count == 0 {
                continue;
            }

            let cross_distance = sum / count as f32;
            let max_threshold = gestures[i].1.max(gestures[j].1);

            if cross_distance < max_threshold * CONFUSION_FLAG_FACTOR && cross_distance > 0.0 {
                let overlap_ratio = max_threshold / cross_distance;
                pairs.push((i, j, overlap_ratio));
            }
        }
    }

    pairs
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

    #[test]
    fn test_banded_produces_valid_results() {
        let examples = vec![
            make_sequence(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            make_sequence(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            make_sequence(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        ];

        let banded = compute_threshold_stats_banded(&examples, 2.0, 1, 0.15).unwrap();

        // Should produce valid non-negative statistics
        assert!(banded.mean >= 0.0);
        assert!(banded.std >= 0.0);
        assert!(banded.threshold >= banded.mean);
        assert_eq!(banded.sample_count, 3); // C(3,2) = 3 pairs
    }

    #[test]
    fn test_banded_with_downsampling() {
        let examples = vec![
            make_sequence(&[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            make_sequence(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
        ];

        let full = compute_threshold_stats_banded(&examples, 2.0, 1, 0.15).unwrap();
        let ds2 = compute_threshold_stats_banded(&examples, 2.0, 2, 0.15).unwrap();

        assert!(full.mean >= 0.0);
        assert!(ds2.mean >= 0.0);
        // Downsampled should produce different (but related) distances
    }

    #[test]
    fn test_outlier_detection() {
        // 5 similar examples + 1 outlier
        let examples = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.1, 1.1, 2.1]),
            make_sequence(&[0.0, 0.9, 2.0]),
            make_sequence(&[0.1, 1.0, 1.9]),
            make_sequence(&[0.0, 1.1, 2.1]),
            make_sequence(&[5.0, 6.0, 7.0]), // Outlier — very different
        ];

        let stats = compute_threshold_stats_banded(&examples, 2.0, 1, 0.15).unwrap();

        // The outlier (index 5) should be flagged
        assert!(
            stats.outlier_indices.contains(&5),
            "Expected example 5 to be flagged as outlier, got: {:?}",
            stats.outlier_indices
        );
    }

    #[test]
    fn test_no_outliers_when_all_similar() {
        let examples = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.1, 1.1, 2.1]),
            make_sequence(&[0.0, 0.9, 2.0]),
        ];

        let stats = compute_threshold_stats_banded(&examples, 2.0, 1, 0.15).unwrap();
        assert!(
            stats.outlier_indices.is_empty(),
            "No outliers expected for similar examples"
        );
    }

    #[test]
    fn test_confusion_detection_overlapping() {
        // Two gesture classes with very similar examples
        let gesture_a = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.1, 1.1, 2.1]),
        ];
        let gesture_b = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.2, 1.2, 2.2]),
        ];
        // Set a high threshold that the cross-distance will be below
        let gestures = vec![(gesture_a, 100.0), (gesture_b, 100.0)];
        let pairs = detect_confusion_pairs(&gestures);
        assert!(
            !pairs.is_empty(),
            "similar gestures with high threshold should be confused"
        );
        assert_eq!(pairs[0].0, 0);
        assert_eq!(pairs[0].1, 1);
    }

    #[test]
    fn test_confusion_detection_well_separated() {
        let gesture_a = vec![
            make_sequence(&[0.0, 1.0, 2.0]),
            make_sequence(&[0.1, 1.1, 2.1]),
        ];
        let gesture_b = vec![
            make_sequence(&[50.0, 60.0, 70.0]),
            make_sequence(&[51.0, 61.0, 71.0]),
        ];
        // Tight thresholds — cross-distance will be much larger
        let gestures = vec![(gesture_a, 1.0), (gesture_b, 1.0)];
        let pairs = detect_confusion_pairs(&gestures);
        assert!(
            pairs.is_empty(),
            "well-separated gestures should not be confused"
        );
    }
}
