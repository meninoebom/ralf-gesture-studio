# Phase 5: F1-Optimized Thresholds & Precompute Optimization — Handoff Plan

**Date:** 2026-02-22
**Branch:** `feature/f1-thresholds` (create from main)
**Status:** Not started. Phases 1-4 are complete and merged to main.

## Context

Phases 1-4 completed 10 of 12 brainstorm items from the accuracy hardening roadmap. The current threshold system uses a mu+sigma heuristic — it computes the mean and standard deviation of pairwise DTW distances within a gesture's examples and sets threshold = mean + 1*sigma. This works but has no awareness of inter-class distances. It can't adapt when a gesture's positive distribution overlaps with negatives from other classes.

F1-optimized thresholds (Jackknife-style) replace the heuristic with empirically optimized thresholds using synthetic positive/negative samples. Separately, downsampled examples are recomputed every DTW frame — precomputing them at `start()` eliminates hundreds of Vec allocations per second.

## Implementation Order

Do these in order — #1 is a pure performance win with no API changes, making it a safe first commit.

---

### 1. Precompute downsampled examples (performance)

**Why:** `find_best_distance()` calls `Self::downsample_seq(example, downsample)` per-example per-frame. At 15Hz DTW rate × N gestures × M examples, this is hundreds of throwaway Vec allocations per second.

**File:** `src/engine/recognizer.rs`

- Add `downsampled_examples: Vec<Sequence>` field to `GestureState` struct
- Populate it in `compute_envelopes()` — this method already iterates over examples and downsamples them for envelope computation. Just store the downsampled result instead of throwing it away.
- In `find_best_distance()`: replace `Self::downsample_seq(example, downsample)` with an index into `self.downsampled_examples[i]`
- Same change in `compute_consensus()` if it also downsamples on-the-fly

**Test:** Add a test that verifies precomputed downsampled examples match the on-the-fly computation (same values, same length).

---

### 2. F1-optimized threshold computation

**Why:** mu+sigma is a heuristic with no concept of negative examples. F1 optimization sweeps thresholds against both positive and negative distance distributions to find the threshold that maximizes F1-score. This is the Jackknife approach from gesture recognition literature.

**File:** `src/engine/statistics.rs`

Add a new public function:

```rust
pub fn compute_threshold_f1(
    positives: &[Sequence],
    negatives: &[Sequence],
    downsample_factor: usize,
    sakoe_chiba_band: f32,
) -> Option<ThresholdStats>
```

Algorithm:
1. **Positive distances:** Pairwise DTW within `positives` (reuse existing banded DTW logic from `compute_threshold_stats_banded`)
2. **Synthetic positives (GPSR):** For each real example, generate ~20 variants using `temporal_stretch` with wider range (±50% instead of the normal ±15%). Compute DTW of each variant against all real examples. This expands the positive distribution to be more robust.
3. **Negative distances:** DTW of each `negative` example against each `positive` example
4. **Threshold sweep:** 200 steps from min(all_negative_distances) to max(all_positive_distances). At each candidate threshold:
   - TP = positive distances below threshold
   - FP = negative distances below threshold
   - FN = positive distances above threshold
   - F1 = 2*TP / (2*TP + FP + FN)
5. Pick the threshold with maximum F1
6. Return `ThresholdStats` with the F1-optimal threshold, plus mean/std for display
7. Outlier detection: same as existing banded function

Also add `f1_score: Option<f32>` field to `ThresholdStats` struct.

**File:** `src/engine/augmentation.rs`
- Ensure `temporal_stretch()` is `pub` (it may currently be private)

**File:** `src/engine/mod.rs`
- Export `compute_threshold_f1`

**Tests:**
- `compute_threshold_f1` produces a valid threshold for known distributions
- F1 threshold is tighter than mu+sigma when negatives are close (the main benefit)
- F1 threshold falls back gracefully with no negatives (single gesture scenario)

---

### 3. Vocabulary flag and GUI integration

**File:** `src/model/vocabulary.rs`
- Add `#[serde(default)] pub f1_threshold: bool` to `Vocabulary` (default false)
- Initialize to `false` in `Vocabulary::new()`

**File:** `src/gui/mod.rs`
- In `compute_gesture_statistics()`: when `self.vocabulary.f1_threshold` is true AND vocabulary has 2+ gestures with examples, collect other gestures' preprocessed examples as negatives, call `compute_threshold_f1` instead of `compute_threshold_stats_banded`
- When only 1 gesture has examples, fall back to banded stats (no negatives available)
- Add `set_f1_threshold` Tauri command (follow the exact pattern of `set_complexity_correction` which was added in Phase 4)
- Add `f1_threshold: bool` to `VocabularyDto`
- Sync flag in `sync_recognizer`

**File:** `src/main.rs`
- Register `gui::set_f1_threshold` in the `invoke_handler`

**File:** `ui/index.html`
- Add an "F1:" toggle button next to the existing CID button in the toolbar

**File:** `ui/main.js`
- Cache the element, add event listener, render toggle state, implement toggle function (same pattern as CID toggle)

---

### 4. Display F1 score in monitor

**File:** `src/gui/mod.rs`
- Add `f1_score: Option<f32>` to `GestureMonitorDto`
- Populate from gesture stats when available

**File:** `ui/main.js`
- Show F1 score next to stats display when in AUTO threshold mode: `F1=0.94`

---

## Files Summary

| File | Changes |
|------|---------|
| `src/engine/recognizer.rs` | Precompute downsampled examples in `GestureState`, use stored copies in `find_best_distance` and `compute_consensus` |
| `src/engine/statistics.rs` | New `compute_threshold_f1()` function, add `f1_score` to `ThresholdStats` |
| `src/engine/augmentation.rs` | Make `temporal_stretch` pub |
| `src/engine/mod.rs` | Export `compute_threshold_f1` |
| `src/model/vocabulary.rs` | Add `f1_threshold: bool` field |
| `src/gui/mod.rs` | F1 gate in `compute_gesture_statistics`, new Tauri command, DTO updates |
| `src/main.rs` | Register `set_f1_threshold` command |
| `ui/index.html` | F1 toggle button |
| `ui/main.js` | Toggle logic + F1 score display in monitor |
| `tests/recognition_integration.rs` | Add `f1_threshold: false` to test config structs (if they exist) |

## Architecture Notes

- **NEVER compute DTW in a polling command.** `get_state()` runs at ~60Hz. All threshold computation must happen when data changes (after training examples are saved), not during the poll loop.
- **sDTW is now the default.** `use_subsequence_dtw` defaults to `true`. Threshold calibration must use `compute_threshold_stats_sdtw` (or the new F1 variant) to match the recognition distance metric.
- **Downsampling factor is 4.** This is used throughout for pairwise statistics to keep 10-example gestures (45 pairs) responsive.
- **The CID toggle (`complexity_correction`) was added in Phase 4** and is the exact pattern to follow for the F1 toggle.

## Verification

1. `cargo test` — all existing 196 tests pass (f1_threshold defaults to false, precompute is internal)
2. New tests for `compute_threshold_f1` (see test descriptions above)
3. New test: precomputed downsampled examples match on-the-fly computation
4. Manual: enable F1 thresholds, train 2+ gestures, verify thresholds differ from mu+sigma
5. Manual: verify F1 score displays in monitor panel
6. `cargo clippy` clean, `cargo fmt --check` clean
