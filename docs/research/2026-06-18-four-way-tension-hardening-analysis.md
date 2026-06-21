# Gesture Recognition Reliability: The "When To Fire" Tradeoff (Verified Analysis)

**Date:** 2026-06-18
**Source issue:** ralf-docs #2 (Tier A, reasoning-bound)
**Status:** Analysis and design only. No source was edited. Every recommendation below is tagged by how it can be validated.
**How this was produced:** A multi-agent ultracode workflow (Triage to Stress to adversarial Cross-verify to Synthesize), grounded in the real diagnostic log corpus (`~/Documents/RALF/*.log`, 51 logs) and the runnable benchmark harness (`tests/recognition_integration.rs`). 36 candidate changes were generated, each stress-tested by three independent skeptic lenses (cross-mode regression, log evidence, sDTW correctness). 34 survived, 2 were refuted. A completeness critic then passed it as complete. The load-bearing finding (RC-1) was re-verified by hand against source.

---

## Bottom line first

The four-way tension this issue names (false positives, false negatives, echo, latency) does not split evenly. When you measure it against what the logs and code actually show, three of the four corners are already solved, and the real problem is concentrated in one place.

1. **Echo is solved and over-determined.** Production logs show 0% echo rate and a minimum gap between hits of 6.5 seconds, which is 6.5x the real 1000ms global cooldown. The layer doing the work is the buffer clear plus the roughly 2 to 3 second window refill after each hit, not the global cooldown.
2. **Above-threshold false positives are closed by construction.** In current code a hit cannot fire while distance is above threshold. The negative-margin fires seen in old logs came from a recovery scheme that no longer exists.
3. **Latency is better than the docs claim.** Steady-state time from movement to fire is about 175 to 200ms, not the roughly 400ms the skill documents. It already meets the target. The real lag a dancer feels is the window fill, not the confirmation count.
4. **The one genuinely open problem is the false-negative / inter-class-confusion corner**, and it is a threshold-and-data problem, not a state-machine problem. State-machine selection changes are off-limits because they regressed the benchmark from 93.3% to 37-50% in the past.

**The single highest-leverage finding:** the designated fix for that open corner, F1-optimized thresholds, is non-functional in production. It is dormant twice over. It is off by default, and even when turned on, the threshold it computes is thrown away before it reaches the recognizer. Fixing that wiring (RC-1) is the prerequisite for measuring whether the inter-class confusion is fixable at all.

---

## The corrected baseline (the issue body is partly stale)

Issue #2 was written against an older snapshot. Before any recommendation makes sense, here is what is actually true in the code today.

### Real config values

| Constant | Issue / older docs say | Actual (`RecognitionConfig::default`) | Where |
|---|---|---|---|
| `frames_to_fire` | 3 | **2** (about 133ms confirmation at 15Hz DTW) | recognizer.rs:95 |
| `global_cooldown_ms` | 1500 | **1000** | recognizer.rs:99 |
| `cooldown_ms` | 500 | 500 | recognizer.rs:91 |
| `max_recovery_ms` | 5000 (safety valve) | 5000 but **dead in the live path** | recognizer.rs:97 |
| `sakoe_chiba_band` | 0.15 | 0.15 | recognizer.rs:101 |
| `margin_rejection_ratio` | (not mentioned) | **0.0** (disabled, required for sDTW) | recognizer.rs:103 |
| `use_subsequence_dtw` | (not mentioned) | **true** (sDTW with wavefront banding is production) | recognizer.rs:104 |

### What is already done

The Feb 21 accuracy-latency hardening roadmap is largely shipped. Phases 1 through 4 are merged (10 of 12 brainstorm items), including the threshold/recognition metric-domain fix, outlier detection in calibration, the One Euro filter, preprocessing defaults on, recovery exit on distance, DBA averaging, the CID complexity-correction toggle, and the quality dashboard. The recovery deadlock and the echo storm documented in `RAW_LEARNINGS.md` are both closed.

So the stress phase was explicitly forbidden from re-proposing already-solved work, and the disproven approaches from prior sessions (prototype averaging, hysteresis, debounce, activity gate, and any state-machine selection change) were ruled out up front.

---

## The four-way tension, resolved by evidence

The analysis enumerated 12 failure modes (the issue's six plus six latent ones). Here is where each one actually stands, measured against the log corpus split into PRE-hardening (Jan 28 to Feb 1, 38 logs) and CURRENT-code (Feb 21 to 22, 13 logs).

| Mode | Status | Evidence |
|---|---|---|
| Echo (same gesture re-firing) | **Solved** | PRE 61.6% echo rate (513/833) to CURRENT 0% (0/43). Min inter-hit gap 6500ms. |
| Global-cooldown not enforced | **Solved** | PRE had 15/29 sessions under the cooldown floor; CURRENT has 0/4. Real floor is 1000ms, not 1500ms. |
| Negative-margin false positive | **Closed by construction** | 20/833 PRE hits fired above threshold; 0/43 CURRENT. Building aborts the moment distance rises (recognizer.rs:416-419). |
| Latency to fire | **Better than target** | PRE mean 284ms to CURRENT 175ms Building-to-Peak. Below the documented 400ms floor and under the 200ms target. |
| Warm-up effect | **Understood, not a bug** | First-3-hits margin 35.9% vs post-warm-up 48.7%, matching the skill's expected ~34% warm-up dip. |
| Recovery deadlock | **Solved** | Recovery now exits on `cooldown_complete` (recognizer.rs:444); the `max_recovery_ms` safety valve is a dead constant. |
| Metric drift (calibration vs recognition) | **Solved** | Phase 1 fixed it for the threshold path. One display-only function still drifts (see RC-3). |
| **False negative (hover above threshold)** | **OPEN** | 8 of 13 CURRENT sessions produced zero hits despite real attempts. 1784 above-threshold near-misses, mean -9.4% over threshold. |
| **Inter-class confusion (one gesture steals another)** | **OPEN** | Squat (16 hits) and Cross Arms run at 17-21% margin, min 1.8%; Reach Up fired once against 364 near-misses. |
| **Tight-margin / high-variance knife-edge** | **OPEN** | The negatives-blind mu+sigma threshold cannot separate confusable gestures. |
| **Variable-length window size** | **OPEN (but penalty unmeasured)** | Window is pinned to the first trained example's length. Real but its FN cost is asserted, not yet measured. |
| MediaPipe jitter false positives | **Partial** | One Euro is shipped; the slope gate's continued value is unmeasured (see RC-5). |

The three OPEN false-negative modes all collapse to the same root cause: the production threshold is `mu + sigma * coefficient`, which has no awareness of where other gestures sit. That is what F1 thresholds were built to fix.

---

## The headline finding: F1 thresholds are computed then discarded

This is verified against source, not inferred.

`compute_threshold_f1` (statistics.rs:445) does the real work. It sweeps 200 candidate thresholds against positive and negative distance distributions and returns the F1-optimal value in `ThresholdStats.threshold` (statistics.rs:622).

But the consumer drops it. In `gui/mod.rs`:

```rust
compute_threshold_f1(&examples, &negatives, 4, ..., coefficient)  // computes swept best_threshold
// ...
gesture.update_statistics(stats.mean, stats.std);                 // passes ONLY mean and std
self.recognizer.set_threshold(gesture_id, gesture.threshold);     // uses the recomputed value
```

And `update_statistics` (vocabulary.rs:182) recomputes the old formula:

```rust
self.threshold = mean + std * self.threshold_coefficient;         // swept value never used
```

So the swept F1 threshold and the F1 score are thrown away. Even if a user sets `vocabulary.f1_threshold = true`, recognition still runs the negatives-blind mu+sigma threshold. The feature is dormant twice: off by default (vocabulary.rs:366) and inert when on.

Every earlier proposal that leaned on F1 as "the FP fix" assumed it worked. It does not. This is why it became RC-1 (P0), and why the experiment that decides whether the confusion is even fixable (RC-2) depends on it.

---

## Recommended changes (verified set)

Ordered. Each tag says how you can check it: **validatable-on-logs** (grep the existing corpus), **harness-testable** (`cargo test`), or **rehearsal-gated** (needs a dancer producing real skeleton streams).

| ID | Pri | Change | Tag |
|---|---|---|---|
| RC-1 | P0 | Wire the F1-swept threshold through to recognition (currently discarded) | harness-testable |
| RC-2 | P0 | Add an F1-on vs F1-off holdout test reporting per-gesture precision/recall (depends on RC-1) | harness-testable |
| RC-3 | P1 | Align `detect_confusion_pairs` with the recognition metric (sDTW on downsampled, not full-res standard DTW) | harness-testable |
| RC-4 | P1 | Add executable no-echo / no-negative-margin-fire / NMS-floor regression guards | harness-testable |
| RC-6 | P1 | Correct the stale recovery / cooldown / latency health checks in the skill and docs | validatable-on-logs |
| RC-5 | P2 | Quantify the slope-gate FN cost vs FP defense with a jitter-injected experiment; make the gate abort observable | harness-testable |
| RC-7 | P2 | Make the buffer-clear / global-cooldown echo defense observable (logging only) | validatable-on-logs |

### RC-1 (P0): Wire the F1-swept threshold through. *harness-testable.*
When the F1 path is taken, set the gesture's recognition threshold to `stats.threshold` (the swept value), not to `mean + std * coefficient`. Add a dedicated path (a Gesture field or a manual-style set) before `recognizer.set_threshold`, and recompute it on `load_vocabulary` the same way `threshold_coefficient` is force-upgraded, or existing `.ralf` files never pick it up.
- **Effect:** Enables the only built negatives-aware defense to actually reach recognition. By itself it does not fix FN; whether it helps is RC-2's experiment.
- **Risk:** Zero runtime risk if landed with `f1_threshold` still defaulting false. Do not enable by default until RC-2 measures it.
- **Guard:** Add a test asserting the threshold the recognizer uses under `f1_threshold=true` equals the swept value, so this exact discard bug cannot silently return.

### RC-2 (P0, depends on RC-1): Per-gesture F1 holdout harness. *harness-testable.*
Today the benchmark reports only aggregate 93.3% / 3.3%, which masks the steal. Add a parameterized leave-one-out holdout that runs twice (mu+sigma vs F1-with-negatives) and prints per-gesture (hit rate, FP count, chosen threshold, best F1). Run at **both** coefficient 2.0 (production) and 3.0 (benchmark default) because the probe shows F1's threshold-delta direction flips between them. Copy `262102 jump-wave-spin-test.ralf` into `tests/fixtures/`.
- **Effect:** Converts "is the knife-edge a threshold problem (F1 fixes it) or a data problem (F1 cannot)" from speculation into a measured number. A low best-F1 per gesture is the direct "data problem" signal.
- **Caveat:** A green benchmark does not license enabling F1 for the dancer. The benchmark runs with no preprocessing at coefficient 3.0; production runs preprocessing on at 2.0 on a different vocabulary. Treat the harness result as a lower bound; production go/no-go is rehearsal-gated.

### RC-3 (P1): Fix the confusion detector's metric. *harness-testable.*
`detect_confusion_pairs` (statistics.rs:645-690) compares full-resolution standard banded DTW distances against sDTW-calibrated thresholds. That is the exact calibration-vs-recognition mismatch the hard constraints forbid, in the one function that still drifts. Downsample both gestures by 4 and use `sdtw_distance` with `band_width = ceil(max_len * 0.15)`. Net faster (about 16x per pairwise call) and makes the triage tool trustworthy.
- **Note:** sDTW cross-distances are systematically smaller (free-start), so the `overlap_ratio` alarm cutoff must be recentered or it flags nearly every pair. Validate on benchmark.ralf that the well-separated trio is not falsely flagged. Resolve sDTW asymmetry by averaging both directions.
- **Also:** `delete_example` recomputes stats but does not refresh confusion pairs, so the list goes stale after deletions. Fix in the same change.

### RC-4 (P1): Executable echo / FP regression guards. *harness-testable.*
Three invariants are enforced only by convention and absence in logs. Add: a no-negative-margin-fire guard (`distance <= threshold * threshold_high_factor`, that is margin >= 0, not a positive floor, because the +1.8% warm-up hit is legitimate); a no-same-gesture-re-fire guard; and a cross-gesture cooldown guard.
- **Critical design subtlety the verifiers caught:** the harness replays frames in a tight no-sleep loop, but global cooldown and Recovery dwell are wall-clock (`Instant::elapsed`). So the buffer-clear plus window-refill (frame-domain) is the only echo gate the tight loop actually exercises. Scope the no-re-fire guard to that, with a descending tail longer than `window_size`, and prove it fails if `buffer.clear` is removed. Exercise the wall-clock NMS only via a competing second gesture or a real `thread::sleep` variant, labeled separately. Do not assert return-to-Idle (Recovery is wall-clock and never exits in a tight loop).

### RC-6 (P1): Correct the stale diagnostic health checks. *validatable-on-logs.*
The skill and docs will currently false-flag healthy current code as broken, which incentivizes re-adding the very mechanisms that caused past death-spirals. Corrections, all tied to `RecognitionConfig::default` as source of truth:
- Recovery is 100% `cooldown_complete` on `cooldown_ms=500`, not `safety_valve_timeout` / `max_recovery_ms` (a dead constant). Bucket `safety_valve_timeout` as a pre-2026-02-22 reason.
- Inter-hit floor is `global_cooldown_ms=1000`, not 1500. The real-world floor is `max(1000ms, window_refill)`, which for the observed 122-frame window is about 2033ms.
- Latency baseline is about 175 to 200ms steady-state, not 400ms. Report per-gesture median, exclude session-start 0ms burst pairs.
- Fix CLAUDE.md's stale config block (`frames_to_fire:3`, `global_cooldown_ms:1500`).
- Optionally add a silent-FN health check: per gesture, post-warm-up above-threshold near-miss count with zero hits. Report the margin distribution (within-5% band vs deep tail), not a single closest value, so it does not over-claim "threshold too tight" when the real cause is a slope-gated held pose or a winner-take-all loss.

### RC-5 (P2): Measure the slope gate, do not ship a change. *harness-testable, then rehearsal-gated.*
The slope gate (require distance strictly falling to enter Building, recognizer.rs:331-341) suppresses resting-pose FPs, but One Euro may already remove the jitter it defends against. Two coordinated, measurement-first moves:
- **Observability:** surface a per-frame `slope_gated` flag and a NEAR reason `slope_gated` (edge-triggered, within a margin band). Do not emit a STATE self-transition (it would flood the STATE channel about 5-6x and pollute the Building-to-Peak metric).
- **Experiment:** in a test-only fork, compare gate policies (strict / relaxed / disabled) crossed with noise conditions, injecting Gaussian jitter into both the static lead-in and the gesture frames before `process_frame`, plus a noise-to-PoseSmoother arm (the harness currently bypasses the smoother and the fixtures were recorded post-smoothing, so without injected jitter the experiment understates both gate value and relaxed-gate FP risk).
- **Discipline:** a clean-data green result does not license relaxing the gate. Final feel is rehearsal-gated.

### RC-7 (P2): Make the echo defense observable. *validatable-on-logs.*
Echo is 0% in production, so this is regression insurance, not a fix. Emit a greppable marker when `buffer.clear` runs on a hit and during the post-hit refill blackout, and a NEAR with reason `global_cooldown_block` when the global cooldown would block an otherwise-eligible Idle-to-Building.
- **Honest framing:** the NMS layer is structurally shadowed by buffer-clear (the roughly 2 to 3 second refill outlasts the 1000ms cooldown), so expect about zero `global_cooldown_block` lines, and treat that null result as the finding. The buffer-clear / refill marker is the genuinely useful half. NEAR channel only, never a phantom Idle-to-Idle STATE transition.

---

## Variable-length window design sketch

**The problem.** `window_size` is fixed for the whole session to the first trained example's frame count (recognizer.rs:763). sDTW free-start absorbs speed, onset, and template-length variation *within* that fixed window, but the window size itself never adapts. A gesture longer than the window is truncated to its last `window_size` frames before any DTW runs, losing its lead-in. A gesture much shorter sits in a window padded with idle frames (partly mitigated by `trim_to_onset` plus sDTW free-start). So a vocabulary mixing a 1s clap and a 7s phrase cannot be served well by one window.

**Important caveat, verified:** on the current benchmark (all gestures about 120-123 frames) this is a no-op, and on a first mixed-length probe, `trim_to_onset` plus sDTW free-start absorbed more truncation than expected. The FN penalty is real in principle but **asserted, not yet measured**. So the design is gated behind a measurement.

### Options

| Option | Sketch | Pros | Cons |
|---|---|---|---|
| **A: Monotonic-max window** | `window_size = window_size.max(example.len())` in `add_example` (one line, recognizer.rs:763) | Trivial, order-independent, removes truncation for every gesture. No new DTW machinery. | Larger window means longer buffer fill before first detection and longer post-hit blackout (about 7s for a 7s gesture). A 1s clap sits in a 7s idle-padded window, raising its FP risk. Cap the recognition band at template length so calibration and recognition bands stay matched. |
| **B: Per-gesture windows** | Each `GestureState` gets its own `window_len` from its own examples; slice that many recent frames per gesture. | The real fix: serves a 1s clap (small window, tight FP) and a 7s phrase (no truncation) in one vocabulary. Keeps winner-take-all. | More per-frame work. Winner-take-all now compares distances over different window lengths; sDTW cumulative cost grows with length, biasing selection toward longer windows. Needs per-gesture distance normalization, done carefully to avoid the velocity-destroying resampling trap. |
| **C: SPRING continuous sDTW** | Maintain a continuously updated sDTW alignment over a growing buffer; emit a match when normalized cost dips below threshold at any subsequence ending now. | Fully length-adaptive; no window bookkeeping. The principled long-term answer. | Largest change. Touches selection, so it carries the documented winner-take-all regression risk (93.3% to 37-50%). Normalization across template lengths is exactly where past selection changes broke. Deferred. |

### Recommendation (two-step, gated)

- **Step 0 (do first, cheap):** build the varied-length harness probe (mixed short and long vocabulary). Assert the long gesture is missed at short-first-seed and hit at longest-seed, assert short-gesture FP stays under 5% in a window sized to the long gesture, and include a pure-idle no-fire segment. This quantifies the penalty that is currently only asserted.
- **Step 1:** if the probe shows a material truncation FN, ship Option A as the zero-risk band-aid (name it as a band-aid: it relaxes "window = first example" toward "window = longest example"; the principled path back is per-gesture windows). Validate it holds benchmark >= 93.3% and does not raise short-gesture FP.
- **Step 2:** only if Option A's short-gesture FP cost is unacceptable for a genuinely wide-spread vocabulary do you invest in Option B. Reserve Option C for when per-gesture windows prove insufficient; it must clear the full benchmark and holdout before any rehearsal.

---

## Decisions that need you (taste and risk, not technical)

These are genuine judgment calls a dancer-performer should make, not things the code can settle.

1. **Should F1 thresholds become the production default once wired (RC-1) and measured (RC-2)?** F1 is the only built defense against one tight gesture stealing another's detections, but it can tighten thresholds (fewer hits) at coefficient 3.0 and loosen them at 2.0, and the benchmark cannot prove production behavior. Recommendation: keep it opt-in, default off, and recommend it only for vocabularies the corrected confusion detector flags as overlapping, until rehearsal data exists.

2. **Latency vs false-positive posture: keep `frames_to_fire = 2`?** Lowering toward 1 cuts about 66ms but admits more single-frame noise as fires. Current 175-200ms already meets the target, and the real lag is the window fill, not the confirmation count. Recommendation: leave at 2; optionally expose it per gesture so a fast percussive gesture can use 1.

3. **For a real mixed-length vocabulary, how much warm-up and post-hit blackout is acceptable to avoid truncating the long gesture?** This decides Option A vs Option B above, and it depends entirely on whether 1s and 7s gestures actually coexist in one piece. Recommendation: answer it from real repertoire, or constrain authoring so gesture lengths stay within about 2x of each other.

4. **Is the strict slope gate worth keeping after One Euro smoothing?** The corpus suggests it is mostly catching held poses (working as intended). Recommendation: keep it strict until RC-5 plus a rehearsal show otherwise, and never relax it on clean-data harness evidence alone.

---

## What was ruled out

Adversarial verification killed 2 proposals outright and corrected several others:

- **F1 lowers thresholds at coefficient 3.0** (worsening, not fixing, the benchmark misses) was the refutation that led to discovering the deeper discard bug. The direction question is now deferred to RC-2 on a functional F1.
- **A window-mismatch warning heuristic** was refuted: it fires on benchmark.ralf itself, and the implied WARN channel does not exist.
- **An "effective floor" monitoring metric** was refuted on a field misread (it cited `buffer_len` as `window_size`). Its one good idea (floor = max(cooldown, refill)) was folded into the RC-6 doc fix.
- **Precompute downsampled examples** (Phase 5 item 1) is correct but inert: it is bit-for-bit identical output with no measurable latency win per the corpus. Excluded from the recommended set so it is not mistaken for a latency fix.
- **Re-asserted disproven approaches** (state-machine selection changes, prototype averaging, hysteresis, debounce, activity gate) were blocked at the stress phase.

---

## Appendix: log corpus profile

51 logs examined, split by recognizer generation.

**PRE-hardening (Jan 28 to Feb 1, 38 logs):** 833 hits. Echo rate 61.6% (declining 79.8% to 48.7% within the bucket as fixes landed). Min inter-hit gap under the cooldown floor in 15 of 29 multi-hit sessions (worst 204ms). All-hit margin mean 51.6% (min -45.8%). 20 negative-margin fires, all in one pathological session.

**CURRENT-code (Feb 21 to 22, 13 logs):** 43 hits. Echo rate 0%. Min inter-hit gap 6500ms in every multi-hit session. No negative-margin hits. Building-to-Peak conversion 96.2%. Margin mean 44.8%, but Cross Arms (17.9%, min 5.9%) and Squat (21.7%, min 1.8%) run extremely tight, and 8 of 13 sessions produced zero hits despite real attempts. That zero-hit pattern is the false-negative problem RC-1 and RC-2 target.

---

*Generated from an ultracode multi-agent analysis on 2026-06-18. All contested claims were verified against source (recognizer.rs, statistics.rs, gui/mod.rs, vocabulary.rs) and the log corpus before inclusion. The F1 discard finding (RC-1) was re-verified by hand.*
