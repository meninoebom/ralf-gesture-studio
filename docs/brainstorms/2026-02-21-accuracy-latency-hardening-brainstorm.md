# Accuracy, Latency & Hardening Brainstorm

**Date:** 2026-02-21
**Goal:** Make Gesture Studio production-ready for high-level dancers — accurate, responsive, no hallucinations, no false positives.

---

## The Smoking Gun: Threshold/Recognition Mismatch

The single most impactful finding from this brainstorm: **threshold calibration and recognition use different DTW configurations.**

| | Threshold Calibration (`statistics.rs`) | Recognition (`recognizer.rs`) |
|---|---|---|
| DTW variant | Unconstrained | Sakoe-Chiba band (0.15) |
| Resolution | Full | 4x downsampled |
| Preprocessing | `process_sequence()` batch | `process_frame()` streaming |

Thresholds are computed in one distance domain and applied in another. This is likely the root cause of many accuracy problems — thresholds may be systematically too loose or too tight depending on how banding and downsampling affect distances for each gesture.

**Fix:** Make `statistics.rs` use the same DTW parameters as recognition: Sakoe-Chiba banded, on downsampled sequences. This is a low-effort, high-impact change.

---

## Prioritized Recommendations

### Tier 1: Fix What's Broken (Do First)

#### 1. Fix threshold/recognition DTW mismatch
- **Files:** `statistics.rs:72-78`, `recognizer.rs:582-604`
- **Change:** Use `dtw_distance_with_abandon` (banded) in statistics, downsample examples before pairwise comparison
- **Impact:** Thresholds will finally be calibrated in the same distance domain as recognition
- **Effort:** Low

#### 2. Add outlier detection to threshold calibration
- **Files:** `statistics.rs:62-106`
- **Change:** After computing pairwise distances, flag examples whose mean distance to others exceeds 2*sigma. Expose per-example scores via `get_state`. Exclude outliers from threshold computation (or at least warn the user).
- **Impact:** Prevents one bad example from tripling the threshold
- **Effort:** Low-Medium

#### 3. Add margin-based rejection (dual-mode null rejection)
- **Files:** `recognizer.rs` (in `run_state_machines`)
- **Change:** After finding the best gesture match, check if the margin to the second-best is sufficient: `(second_best - best) / second_best > margin_threshold`. Reject if too close. This catches inter-class confusion that per-gesture thresholds miss.
- **Effort:** Low — the distances are already computed for all gestures each frame

### Tier 2: Improve Input Quality (Do Second)

#### 4. Add One Euro Filter for temporal smoothing
- **Files:** New `src/engine/smoothing.rs`, modify `src/osc/receiver.rs` or `preprocess.rs`
- **Change:** Apply per-joint, per-coordinate One Euro filtering before any preprocessing. Reduces MediaPipe jitter that inflates DTW distances. Industry standard for real-time pose data.
- **Params:** `min_cutoff=1.5`, `beta=0.007`, `d_cutoff=1.0` (tune for dance)
- **Crate:** [`one-euro-rs`](https://github.com/MichaelMauderer/one-euro-rs)
- **Latency cost:** Essentially zero (single-sample filter)
- **Effort:** Low

#### 5. Precompute downsampled examples at `start()` instead of per-frame
- **Files:** `recognizer.rs:581` (inside `find_best_distance` inner loop)
- **Change:** Downsample and weight-apply examples once when recognition starts, not on every DTW frame
- **Impact:** Eliminates N*gestures Vec allocations per DTW cycle. Also eliminates a latency source.
- **Effort:** Low

#### 6. Fix Recovery state to exit on distance, not just timeout
- **Files:** `recognizer.rs:391-404`
- **Change:** Exit Recovery when distance rises above threshold AND stays above for N frames (mirror of Building entry). Keep 5s timeout as safety valve only.
- **Impact:** After a hit, re-detection can happen as soon as the dancer resets, not after a forced 5s wait
- **Effort:** Low

### Tier 3: Better Features & Smarter Thresholds (Do Third)

#### 7. Add angular features (joint angles)
- **Files:** New computation in `preprocess.rs`, append to frame data
- **Change:** Compute angles at major joints (shoulders, elbows, hips, knees — ~12-16 angles). Append to existing position features (hybrid representation).
- **Why:** Rotation-invariant, scale-invariant, noise-resistant. Raptis/Hoppe 2011 achieved 96.9% on 28 dance gesture classes with angular skeleton representation.
- **Effort:** Medium

#### 8. DBA (DTW Barycenter Averaging) for template consolidation
- **Files:** New `src/engine/averaging.rs`, modify `recognizer.rs`
- **Change:** Compute a single averaged template per gesture using DBA. Use this for recognition instead of comparing against all N examples.
- **Benefits:** (a) Recognition cost drops from O(N*DTW) to O(1*DTW) per gesture, (b) Outlier examples are naturally down-weighted, (c) Solves "more data doesn't help" — the average converges even if individual examples are noisy.
- **Quick win variant:** Start with medoid selection (example closest to center) as the primary template.
- **Effort:** Medium

#### 9. F1-optimized thresholds via synthetic data (Jackknife-style)
- **Files:** `statistics.rs`
- **Change:** Generate positive samples via stochastic resampling (GPSR) and negative samples by splicing half-gestures from different classes. Score all synthetics, sweep thresholds, pick the point that maximizes F1-score. Replaces mu+sigma with an empirically optimized threshold.
- **Why:** mu+sigma is a heuristic; F1-optimization adapts to each gesture's discriminability. Jackknife and VKM both use this approach.
- **Effort:** Medium-High

### Tier 4: UX Improvements (Ongoing)

#### 10. Training data quality dashboard
- **Files:** `ui/main.js`, `gui/mod.rs` (expose via `get_state`)
- **Change:** Show per-example quality in the UI:
  - Distance to medoid (green/yellow/red traffic light)
  - Example length relative to mean
  - Overall gesture health score (σ/μ ratio — low = consistent, high = noisy)
  - Inter-gesture confusion warning when two gestures' example distributions overlap
- **Why:** Users currently add data blind. Showing quality metrics enables informed decisions about which examples to keep/remove.
- **Effort:** Medium

#### 11. Enable preprocessing by default
- **Files:** `preprocess.rs:38-46`
- **Change:** Default hip_centering and scale_normalization to ON. These are essential for position/scale invariance and should not be opt-in.
- **Effort:** Trivial

#### 12. Complexity correction factors (CID)
- **Files:** `recognizer.rs`
- **Change:** Normalize DTW scores by gesture complexity (total derivative path length) so simple and complex gestures produce comparable scores. Cheap to compute.
- **Effort:** Low

---

## What We're NOT Doing

| Idea | Why Skip |
|------|----------|
| Frequency domain features (FFT) | DTW already captures temporal patterns; FFT adds complexity for marginal gain at this stage |
| Adaptive sliding window | Interesting but complex; fix the threshold mismatch first — may resolve the warm-up issue |
| Full ODOT protocol | MediaPipe never drops landmarks; ODOT adds complexity for no benefit (researched in prior session) |
| OSCQuery discovery | Adds brittleness; manual config is fine for research tools |
| Replacing DTW with neural network | DTW's interpretability and few-shot capability are core strengths; keep it |

---

## Implementation Roadmap

**Phase 1 — Fix the foundations (1-2 sessions)**
- #1 Threshold/recognition DTW mismatch
- #2 Outlier detection in threshold calibration
- #5 Precompute downsampled examples
- #6 Recovery exits on distance
- #11 Enable preprocessing defaults

**Phase 2 — Improve signal quality (1-2 sessions)**
- #4 One Euro Filter
- #3 Margin-based rejection
- #10 Training data quality dashboard (basic version)

**Phase 3 — Advanced features (2-3 sessions)**
- #7 Angular features
- #8 DBA template averaging
- #12 Complexity correction factors

**Phase 4 — Production hardening (2-3 sessions)**
- #9 F1-optimized thresholds (Jackknife-style)
- #10 Full quality dashboard with inter-gesture confusion detection

---

## Key Sources

- [Jackknife gesture recognizer](https://github.com/ISUE/Jackknife) — F1-optimized thresholds, correction factors, direction-based DTW
- [VKM (2022, CHI)](https://mykola.io/publication/vkm/vkm.pdf) — Improved negatives via "Mincer", continuous stream optimization
- [GRT](https://github.com/nickgillian/grt) — Dual-mode null rejection, DTW implementation reference
- [DBA averaging](https://www.sciencedirect.com/science/article/abs/pii/S003132031000453X) — DTW Barycenter Averaging algorithm
- [One Euro Filter](https://github.com/casiez/OneEuroFilter) — Adaptive low-pass filter for real-time input
- [one-euro-rs](https://github.com/MichaelMauderer/one-euro-rs) — Rust implementation
- [Raptis/Hoppe 2011](https://hhoppe.com/dance.pdf) — 96.9% accuracy on dance gestures with angular features
- [DDTW paper](https://ics.uci.edu/~pazzani/Publications/sdm01.pdf) — Derivative DTW outperforms positional DTW
- [Sliding Adaptive DTW (2025)](https://link.springer.com/chapter/10.1007/978-3-031-82475-3_12) — Adaptive window bounds
- [MediaPipe accuracy study (2024)](https://dl.acm.org/doi/10.1145/3719384.3719453) — Real-world reliability by body region
