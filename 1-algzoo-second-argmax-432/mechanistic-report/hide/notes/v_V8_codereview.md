# Verification bundle V8_codereview

## Summary

I reran all 16 experiment scripts and wrote 14 check scripts in v_V8_codereview. Every number in claims_for_verification.md that the reviewed scripts produce reproduces exactly on the lead's seeds, and the rechecks with independent code or fresh samples agree. But five scripts need corrections to how numbers are computed or quoted, and three readings in the notebook and report draft are not supported.

What is clean: no off-by-one error in ages, steps or logit indices; no misuse of the −inf placeholders; no division by non-positive M; no intervention at the wrong step or on the wrong units. running_stats matches a brute-force loop (0 mismatches), K0 holds to 4e-15, and fig_mechanism.npz is identical to a recomputation with the current code. Nothing wrong in common.py, modes.py, e12, e13, e15b, e16, e19, e20, e21, e24, e25 (only label and comment nits).

Corrections (the qualitative conclusion survives in each):
- **e17 (I1):** the pooled numbers include the wipe at the last step, although the script's header says pM ≤ 8 − k, and the split is shown only for the most favourable lag, k = 2. Over all lags k = 1..9, P(answer = max) is 0.986 for t0 ≤ 7, 0.855 for t0 = 8 and 0.495 for t0 = 9; single cells range from 0.42 to 1.00 at t0 = 8.
- **e15 (S2):** R² = 0.88–0.97 is pooled over all ten logit indices. Index 9 carries 33–49% of the variance and fits almost perfectly. For the indices the claim is about (i ≤ 7) it is 0.82–0.96 pooled, and 0.14–0.98 per index (≥ 0.85 where the wave is large).
- **e14 (D2, D1):** the persistence loop has no g ≥ 1 condition although the claim says "such records"; with it, c1 = 41.1, 42.7, 34.1, 29.5, 25.5, 20.2. D1 pools two orders of the positions whose c1 differ by up to 2.3×, and the Δ = 1 ratio is 66, not 65.
- **e18 (I2):** P(answer = p0 − k) is taken over all sequences, including those where that position does not exist or is the old runner-up. Conditioned properly, a one-step rotation moves the answer by one position in 36% (k = −1) and 40% (k = +1), against the reported 47% and 38%. For f = 1.5 and 2, clipping the unreachable negative activations gives 0.838 and 0.676 instead of 0.893 and 0.746.
- **fig_mechanism:** panel (a) bins by (M − max(S, 0))/M but the legend says (M − S)/M; panel (f) shows conditional rates without saying so.

Readings that do not hold:
1. Report §3.5, "wipe at step 8 or earlier → 99.7–100% answer the max": true only for k = 2. Over all lags with t0 ≤ 8 it is 0.957.
2. Notebook, "the lead decays about 10% per step" (D2): this is a cohort artefact. In fixed cohorts the lead grows by 20–60%.
3. Report §7 and §11, "errors at separations 7 and 8 follow from the single-harmonic deposit" (D4): they come from records at the last step and from one order. D = 8 is 0.708 against 0.005 by order; D = 7 with the record at step 8 is 0.022.

Two scope caveats:
- **T2:** R² = 0.985–0.992 is for a hypothetical free step. Real no-event steps give 0.80–0.91.
- **K5:** R² of activations is a lenient measure. A clock-free oracle baseline reaches 0.82–0.997.

Not reviewed: the scripts behind M3–M6 (e01–e09, e11). I only ran e07 and e11, and their output matches M3 and M4.

## Verdicts

### common.py: CONFIRMED

- my numbers: running_stats vs brute-force loop on 3000 sequences: 0 mismatches in M1..M3, p1..p3, a1..a3. run vs plain loop: max diff 0. traces identity h_t = Σ_p J[t,p] x_p: error 1e-13. Accuracy on 2^20 sequences: 0.9532 (N(0,1)), 0.9383 (Unif(-1,1)).
- comment: No error. Conventions are as stated: state after input, window x_0..x_t inclusive, M2/M3 = -inf and p = a = -1 when absent. Every caller guards those placeholders. Fragile: `from common import *` exports T, N, w, W, U into every script; e16 rebinds T (harmless there).

### modes.py: CONFIRMED

- my numbers: Eigenvalues 1.0099, 0.9735, 0.9509@42.75°, 0.9370@83.28°, -0.9283, 0.8373@124.78°, 0.7350@42.99°, 0.2310, 0.0010. LEFT·W[ORD,4] = 1, LEFT_i·RIGHT_j = δ_ij, LEFT·A = λ·LEFT, reconstruction error 1e-13. M1 and M2 reproduce (comparator shares 0.041, 0.506, 0.966, 0.606, 0.106, 0.347, 0.916, 0.976; level on unit 2: 0.852). With units rescaled by their rms activation the shares become 0.025, 0.428, 0.977, 0.741, 0.169, 0.476, 0.945, 0.988.
- comment: No error; the gauge and the edit formula 2·Re(dz·RIGHT) are correct. Fragile: mode names are assigned by sorted |λ| with only a length assert, so other weights would be mislabelled silently. The left-vector shares depend on unit scale (the draft says so).

### e10_tape_shift.py: CONFIRMED

- my numbers: Residual 0.0949. Σ[i,i+1] = 0.971, 0.933, 0.898, 0.886, 1.003, 0.925, 0.934, 0.339, -0.001. Largest other entry in rows 0–6: 0.387 (the 0.42 is Σ[7,1]). Row 9: 1.224–1.615. T2: 0.9917, 0.9911, 0.9846; also centred per position: 0.987, 0.984, 0.981; worst single position 0.87–0.92. Null residual for random or rotated readouts: mean 0.44–0.47, minimum 0.15. Real steps, same R² convention: 0.56–0.76 all states, 0.80–0.91 no-event steps, 0.92–0.99 when the gate pattern is the resting one (10.3% of states).
- comment: No code error; T1 and T2 are exact as defined. T1's bound '≤ 0.42' for rows 0–6 is loose (0.39). The comment on line 20 ('remove the mean ... subdiagonal') does not match the code, which prints the raw superdiagonal. Scope objection: T2 tests a hypothetical free step, see objections.

### e12_early_readout.py: CONFIRMED

- my numbers: Independent re-implementation without running_stats: 0.992, 0.987, 0.971, 0.966, 0.969, 0.970, 0.970, 0.967, 0.961 given S_t > 0; 0.687, 0.676, 0.777, 0.839, 0.893, 0.926, 0.945, 0.953, 0.953 over all states.
- comment: Nothing wrong. Index-to-position conversion and the valid-index mask are correct.

### e13_provisional_scores.py: CONFIRMED

- my numbers: Fresh sample, own code. Age 0: ratio median d / median v = 61.8, 61.8, 61.6, 61.4, 59.8, 55.8 in the bins checked. Age 1: medians -18.2, -15.9, -13.6, -12.8, -9.7, -4.7. The max at ages 2..9 with v > 0.2: medians from -14.5 to -4.9.
- comment: Nothing wrong in the code. T4's 'between -4.5 and -14.5' for v > 0.2 is loose: -4.5 belongs to the bin (0.1, 0.2]; the range is -4.9 to -14.5. The column headed 'median/v' is median(d)/median(v), as the claim says.

### e14_score_updates.py: CORRECTED

- my numbers: D1 reproduces. Exact c1/(1 − cos(42.75°·Δ)) = 66.1, 47.2, 45.6, 44.5, 44.5, 51.3, 81.8. Split by order (old max earlier | later than old runner-up): c1 = 21.3|11.2, 39.8|51.7, 53.2|109.5, 60.1|139.8, 63.1|116.4, 59.7|71.6, 46.3|28.8 for Δ = 1..7. D2 as coded: 40.7, 42.6, 35.3, 32.1, 28.6, 23.8. With g ≥ 1: 41.1, 42.7, 34.1, 29.5, 25.5, 20.2. Fixed cohort (top two unchanged through t+5): c1 = 26.3, 26.6, 17.2, 18.0, 19.4, 20.2, c2 from 0.0 to 7.6, median y(k)/y(0) = 1, 1.18, 1.35, 1.49, 1.53, 1.60. D3 reproduces (k1 = 34.5 … 60.4, R² 0.562–0.719).
- comment: Indexing is correct. Two population problems in the persistence block (lines 63–75). (a) It has no g ≥ 1 condition although the claim says 'such records'; c1 changes by up to 3.6. (b) The set of records changes with k, because t ≤ 9 − k means later k uses earlier records with smaller separations. That produces the apparent decay 40.7 → 23.8; within a fixed cohort the lead grows, and c1 at a fixed separation ≥ 2 is flat within about 10%. D1 pools the two orders of the positions, whose c1 differ by up to 2.3×. The Δ = 1 ratio is 66, not 65 (1 − cos was rounded to 0.27).

### e15_two_score_vectors.py: CORRECTED

- my numbers: Pooled R² over all 10 indices reproduces: 0.965, 0.963, 0.971, 0.956, 0.913, 0.897, 0.889, 0.880 for a1 = 2..9. Index 9 holds 33–49% of the total sum of squares with R² ≥ 0.994. Indices 0..7 pooled: 0.947, 0.945, 0.958, 0.926, 0.832, 0.832, 0.836, 0.816. Single index 0..7: 0.14–0.98, median 0.86; at d = 0: 0.855–0.972; at d = ±4: 0.879–0.976. A(d) table reproduces; d alone explains 97.2% of the variance of the 64 coefficients, and −60·cos(42.75°·d) explains 93.2% (free fit: amplitude 66.2, offset 3.5). B(d): amplitude 7.1, minimum at d ≈ 0, offset 4.2. Fitted C_9 = 58.2–58.7; A_9 ∈ [−59.2, −34.8]; B_9 ∈ [−58.4, −54.9].
- comment: The regression and the d-indexing are correct. The R² on line 43 is pooled over the ten indices, which the claim does not say, and index 9 (V[9] ≈ 58·x_9) inflates it. For i ≤ 7 the range is 0.82–0.96 instead of 0.88–0.97. Line 48 prints U[:, COMP]·w[COMP] under the label 'C (coefficient of x_9, from the a1=5 fit)'; the fitted C is never printed, though it agrees. Lines 62–63 hold a dead duplicate assignment. Parts (i) and (iii) are superseded by e15b.

### e15b_two_score_vectors.py: CONFIRMED

- my numbers: S1 reproduces. By age of the max the rate is 0.96–0.97 (age 1), 0.98–0.99 (age 2), 0.89 (age 3), 0.91–0.93 (age 4), 0.98–0.99 (ages 5–8). At t = 9 with the max at position 0 it is 0.305, which alone produces the drop to 0.891; the other positions give 0.888–0.993. S6 reproduces. All four comparators are off at 85–97% of records. m_t against shifted m_(t−1): 0.77–0.96 at records, 0.67–0.88 at no-event steps. Part (iv) with comparators on at both t and t−1, as its label says: 0.529, 0.684, 0.763, 0.849, 0.915, 0.932, 0.955.
- comment: No error in indices or masks; the claim text matches the code. Line 41's label says 'comparators all on at t and t-1' but the code conditions on t only. S1's value at t = 9 is a mixture, see my_numbers. S6's R² measures how well m itself shifts, because s_t = m_t exactly whenever all four comparators are off.

### e16_m_scores.py: CONFIRMED

- my numbers: All quoted cells reproduce. Split by order (runner-up earlier | later than the max): V-part 51.9|45.7 and s-margin 8.2|11.0 for D = 3, gap (0.3, 0.5]; V-part 73.6|63.1 for D = 3, gap (0.5, 0.8]; s-margin 13.7|24.5 for D = 5 and 13.6|21.3 for D = 6. P(margin > 0) ≥ 0.998 in both orders. Over all cells with gap > 0.05 the median V-part exceeds the 60·(M − 0.87·S)·(1 − cos) prediction by −3…+13% for D = 2–4, +10…+26% for D = 5, up to +35% for D = 6 and +38…+83% for D = 1.
- comment: Nothing wrong. At t = 9 the logit index equals the position, so no conversion is needed. Line 14 rebinds T (the sequence length from common) to the third-largest value; harmless here but fragile. The prediction is quoted for the cells where it fits best (D = 3, 4); it runs 25–80% low for D = 1 and D = 6.

### e17_wipe_comparators.py: CORRECTED

- my numbers: I1 reproduces. Pooled over t0 ≤ 8 only, P(max) = 0.973, 1.000, 0.881, 0.893, 0.977, 1.000 for k = 1..6 (e17: 0.868, 0.873, 0.753, 0.816, 0.977, 0.999). Full table on 1M sequences, P(answer = max): t0 ≤ 7: 0.889–1.000 by cell, pooled 0.986; t0 = 8: 0.794, 0.997, 0.418, 0.710, 0.931, 1.000, 1.000, 0.990 for k = 1..8, pooled 0.855; t0 = 9: 0.020, 0.000, 0.000, 0.435, 0.979, 0.996, 0.994, 1.000, 0.028 for k = 1..9, pooled 0.495. All k with t0 ≤ 8: max 0.957, runner-up 0.002, the wipe step itself 0.041. Control unchanged (P(max) ≤ 0.009).
- comment: The intervention is applied at the right step and to the right units. Line 23 says the population is pM ≤ 8 − k, but line 27 uses t0 ≤ 9, so the pooled numbers mix in the wipe at the last step, which is a direct readout ablation. The split by t0 is printed only for k = 2, the most favourable lag. In the failing cells the answer is the wipe step itself (58% for k = 3, t0 = 8; 98–100% for k ≤ 3, t0 = 9). The conclusion (wipe → max, almost never the runner-up) stands, but '≈100%' is k-dependent.

### e18_clock_before_record.py: CORRECTED

- my numbers: All I2 numbers reproduce (n = 152,690). Clipped at 0 after the edit, the f-scan is 0.019, 0.097, 0.342, 0.889, 0.990, 0.838, 0.676. States with a negative activation after the edit: 8–9% for f < 1 (one unit), 98–100% for f = 1.5 and 2 (2–3 units, min h ≈ −3 to −6·M0), 33–68% for rotations. P(answer = p0 − k | that position exists and is not q0) = 0.233, 0.363, 0.400, 0.365 for k = −2, −1, +1, +2, against the reported 0.321, 0.474, 0.381, 0.249. When the target is q0 it is answered in 96% (k = ±1). The target does not exist in 24% of sequences for k = +1 and 46% for k = +2. Setting f = 0 on other modes: P(p0) = 0.301 (level), 0.219 (runner), 0.828 (h2), 0.971 (alt), 0.928 (h3), 0.846 (short). A one-step rotation of h2, h3 or short gives p0+1 in 0.145, 0.073, 0.037.
- comment: The edit changes exactly the clock coordinate and is applied to h_(tr−1) as claimed; the selection matches the claim. Two things make the quoted probabilities misleading. The categories on lines 47–51 are not exclusive and their denominator includes sequences where the target position does not exist. And the edited states are unreachable for f > 1. The conclusion (radius and phase of the clock control the answer) stands.

### e19_runnerup_event.py: CONFIRMED

- my numbers: Own code: θ_1 = 1.274, θ_6 = 1.673, θ_7 = 1.381, θ_8 = 0.555, θ_4 = 1.366. All nine rows of O3 reproduce (gates, h_6[4], h_7[5], |z|/10.157 at t = 9, answer).
- comment: Nothing wrong. The thresholds are computed from h_5, so they do not depend on v, and the gate pattern is consistent with x < θ_j.

### e20_unit_orbits.py: CONFIRMED

- my numbers: O1 reproduces: fractions of exact zeros; h_12 maxima 15.3 and 31.1; half-ranges 2.3/6.3, 2.2/6.6, 1.8/5.7, 1.2/4.1 (two differ in the last digit by rounding at .25 and .15). 24% of the selected states have a1 = 1. With ages 2..9 only, the third-bin half-ranges of units 6 and 7 fall to 4.4 and 3.4. The third bin contains 31–67% states with S ≤ 0.
- comment: The code matches claim O1. The docstring says 'the runner-up has not changed for ≥ 2 steps', but a2 ≥ 2 does not ensure that at a1 = 1, where the old max was demoted one step earlier. Line 27 computes frac0 and never uses it. The bin (0.85, 1.0] of (M − max(S, 0))/M includes every state with S ≤ 0.

### e21_core_wiring.py: CONFIRMED

- my numbers: K0 checked on 14.4M threshold values: max error 4e-15. K1 reproduces from the weights with own code (Bc, Bc·1, eigenvalues 0.9375@±42.56° and 0.0432, b2, w4·b4, W[2,2], W[2,4], w_2, 0.0485, 0.0970). K3 and K4 reproduce with an independent running max and age; K4 at age 9 is 0.918. The least-squares constant in h_2 ≈ c·M is 17.3.
- comment: Nothing wrong. The threshold identity is exact, and the masks (max age ≥ 2, M > 0.3, t ≤ 8) are as claimed. No script of the lead checks K0 numerically; it holds.

### e22_separation_errors.py: CONFIRMED

- my numbers: Table reproduces exactly. With 16 seeds, gap bin (0.03, 0.08]: D = 7 gives 0.022 with the record at step 8 and 0.241 at step 9; D = 6 gives 0.053, 0.104, 0.183 for records at 7, 8, 9; D = 1 gives 0.005–0.053 for records at ≤ 8 and 0.189 at 9. D = 8 is 0.708 (old max at 8, third at 0) against 0.005 (reverse). D = 7 cells with the record at 9: 0.625, 0.263, 0.007, 0.097. Records at step ≤ 7: 0.024, 0.039, 0.033, 0.021, 0.036, 0.053 for D = 1..6.
- comment: No code error; D4's table is exact. But the table cannot serve as a test of the (1 − cos(ω·D)) prediction. The high rates at D ≥ 6 come from records at the last step and from one order of the two positions; a symmetric law in D predicts neither. See objections (affects the interpretation in report §7 and §11 item 7).

### e23_core_alone.py: CONFIRMED

- my numbers: A: 0.965, 0.981, 0.932, 0.964, 0.979, 0.877. B: 0.991, 0.995, 0.989, 0.995, 0.993, 0.991 (both reproduce). No recurrence at all: −1.63 to 0.34. Clock-free oracle (θ_j = c_j × running max): 0.979, 0.997, 0.879, 0.876, 0.988, 0.820. R² of θ_j/M for units 4, 1, 6, 7, 8: A: −3.99, 0.52, 0.81, −0.80, 0.44; B: −1.08, 0.92, 0.98, 0.38, 0.96 (rms error of B: 0.05–0.09·M). Core clock z_c, simulated against real, complex centred R²: A 0.87, B 0.92, oracle 0.01.
- comment: No code error; K5 is exact. The docstring announces a threshold comparison that the code does not make. R² of activations is lenient, because a model with no clock reaches 0.82–0.997. Variant B is clearly above that baseline for units 1, 6, 8 and reproduces the clock coordinate (0.92). Variant A is not better than the baseline for half of the units.

### e24_core_clock.py: CONFIRMED

- my numbers: K6 reproduces (phases, b_M, b_S, uncentred R² 0.9597–0.9971; |ℓ·1| = 0.0132). Centred R² of the same fit: 0.991, 0.986, 0.987, 0.976, 0.910, 0.870, 0.863, 0.882. The discarded component perpendicular to the median phase has rms 8–16% of the real part. Complex fit z_c ≈ α·M + β·S⁺: uncentred R² 0.940–0.996, centred 0.81–0.99; phase of α within 6° of the median phase.
- comment: Nothing wrong; the R² convention is the one the claim states (uncentred). The script uses two conventions under the same label 'R2' (centred on line 38, uncentred on line 48). λ is hard-coded on line 36, and LEFT is imported but unused.

### e25_numbers_for_report.py: CONFIRMED

- my numbers: T5, K2, O2 reproduce with own code: 5.01, 7.38, 49.90, −4.04, sum 58.25; −50.70; 3.398; 18.009; 84.335; max other 0.432; cosines 0.740 and 0.693, others ≤ 0.038. S4 reproduces. With the frequency left free the best fits are at 48.65°, 49.05°, 44.35°, 44.90° (R² 0.980, 0.996, 0.992, 0.994). Over all ten positions at 42.75°: R² 0.63, 0.77, 0.97, 0.72.
- comment: Nothing wrong. Line 9 labels W[[5,9], ORD]·w[ORD] as '(all four comparators on)', but the sum is over all 13 non-pulse units; the wave units contribute only −0.105 for unit 5. S4 is a 3-parameter fit to 8 points; the columns of units 1 and 6 prefer about 49° over 42.75°.

### fig_mechanism.py: CORRECTED

- my numbers: The cache is identical to a recomputation with the current compute() (all 8 arrays), although the .npz is one minute older than the .py. Panel (a), third bin: 32–78% of states have S ≤ 0. As coded, |z|/(10.157·M) = 0.996, 1.004, 0.959, 0.914, 0.873, 0.826, 0.784, 0.739, 0.697 for ages 1..9. With the literal (M − S)/M ∈ (0.85, 1]: 0.927, 0.935, 0.899, 0.868, 0.839, 0.805, 0.768, 0.732, 0.692; phases agree within 0.7°. In the first bin the radius is 1.4–1.7× the gap. Panel (f) as plotted: 0.961–0.992 and 0.891–0.972. Without the conditions: 0.675–0.953 (all units → runner-up) and 0.500–0.858 (no comparators → max, index 9 excluded). With index 9 allowed: 0.02–0.48. Panel (b) medians recomputed: identical.
- comment: Panels (c), (d), (e), (f) compute the same quantities as e15, e13, e10, e12/e15b. Two labels do not match the computation. In (a) the legend says 'gap (M − S)/M' but line 36 bins by (M − max(S, 0))/M. In (f) the curves are conditional on S_t > 0 (and for the second curve on the max not being the current input, with index 9 excluded), which neither the legend nor the axis says. The title of (e) describes real steps, but Σ is the hypothetical resting step. `--redraw` trusts the cache with no staleness check.

## Objections tested

- **e17 / I1: the wipe result is shown split by wipe step only for the lag k = 2, and the pooled numbers include the wipe at the last step. The report draft (§3.5, §10, §11) generalises this to 'wipe at step 8 or earlier → 99.7–100% answer the max' and '≈100%'.** (affects conclusion: partly)
  - test: Full table of P(answer = max / runner-up / the wipe step / other) by lag k = 1..9 and wipe step t0 = 1..9, on e17's sample and on a fresh 1M sample (c02, c14).
  - result: P(answer = max): 0.986 for t0 ≤ 7 (cells 0.889–1.000), 0.855 for t0 = 8 (0.79, 0.997, 0.42, 0.71, 0.93, 1.00, 1.00, 0.99 for k = 1..8), 0.495 for t0 = 9. All lags with t0 ≤ 8: 0.957 max, 0.002 runner-up, 0.041 the wipe step itself. The 99.7–100% holds for k = 2 (and k = 6, 7) only.
- **e17, e15b, figure panel (f): 'the wave units alone rank the maximum first' holds only among inputs older than the current one. V also carries the current input's own (negative) score, so m_t with index 9 allowed usually has the current input on top.** (affects conclusion: partly)
  - test: Passive argmax of m_t with index 9 allowed, by age of the max; compared with the wipe at t0 = 9; unconditional rates per t (c14, c09, c10).
  - result: P(argmax m_t = max) with index 9 allowed: 0.00–0.02 at ages 1–3, 0.31–0.59 at age 4, 0.97–1.00 at ages 5–8, 0.03 at age 9. Unconditional per t: 0.02–0.48. Even with index 9 excluded, S1's 0.891 at t = 9 is 0.305 when the max is at position 0. The report summary's '94–97% of states' omits t = 9 and the conditions.
- **e14 / D2: the trend of c1 with k compares different sets of records. A record that can be followed for k steps must occur at t ≤ 9 − k, so it has smaller separations, and c1 depends strongly on separation.** (affects conclusion: yes)
  - test: Fixed cohorts (top two unchanged through t+K, K = 5, 3, 2), followed over k; per-sequence ratios y(k)/y(0); c1 within one separation (c04, c11).
  - result: Within a fixed cohort the lead grows: median y(k)/y(0) = 1.18–1.21, 1.34–1.37, 1.47–1.49, 1.53, 1.60 for k = 1..5, and P(y > 0) rises from 0.972 to 0.995–0.998. c1 at fixed separation 2, 3, 4 is flat within about 10% (for example 41.6, 43.5, 39.7, 39.8); only separation 1 drops (21 → 13) while the M0 term grows. The notebook's 'decays about 10% per step (40.7 → 23.8)' is a cohort artefact.
- **e14 / D1: the fit pools the two orders of the positions (old max before or after the old runner-up). The near-constant ratio c1/(1 − cos(ω·Δ)) ≈ 45–47 for Δ = 2..5 is used as evidence for a first-harmonic deposit.** (affects conclusion: partly)
  - test: Same regression split by the sign of p_M0 − p_S0, and by record step (c04).
  - result: c1 differs by up to 2.3× between the orders: Δ = 3: 53 against 110; Δ = 4: 60 against 140; Δ = 5: 63 against 116. The ratio to (1 − cos) is 80, 43, 33, 30, 34, 48, 91 when the old max is earlier, and 42, 56, 68, 70, 64, 58, 56 when it is later (R² 0.86–0.98 there for Δ ≥ 2). The (1 − cos) shape with amplitude about 60 holds for one order only; the constant 45–47 is a property of the mixture.
- **e22 / D4: the separation D = |p_S − p_T| is confounded with the step of the record and with the order of the two positions. D = 8 forces positions {0, 8} and the record at step 9. The table is read as a confirmed prediction of the single-harmonic deposit (notebook e22; report §7 bullet 2 and §11 item 7).** (affects conclusion: yes)
  - test: Error rate by D × record step, by order, and by exact positions with the record at step 9; 16 seeds, 8M sequences (c07).
  - result: Gap bin (0.03, 0.08]. D = 8: 0.708 (old max at 8, third at 0) against 0.005 (reverse). D = 7: 0.022 with the record at step 8, 0.241 at step 9, cells from 0.007 to 0.625. D = 6: 0.053, 0.104, 0.183 for records at 7, 8, 9. D = 1: ≤ 0.053 for records at ≤ 8, 0.189 at 9. Records at step ≤ 7 show no (1 − cos) pattern: 0.024, 0.039, 0.033, 0.021, 0.036, 0.053 for D = 1..6. The elevated errors at D ≥ 6 are a last-step-record effect.
- **e18 / I2: the edit h[ORD] += 2·Re(dz·RIGHT) acts on a post-ReLU state and can leave negative activations, which the network cannot produce. The outcome 'answer = p0 − k' is counted over all sequences, including those where that position does not exist or is the old runner-up q0.** (affects conclusion: partly)
  - test: Same edits followed by clipping at 0; share and size of negative entries; conditional probabilities; the same edits on the other modes (c03, c11).
  - result: For f ≤ 1 clipping changes nothing (8–9% of states, one small negative unit). For f = 1.5 and 2, 98–100% of states are unreachable and clipping gives 0.838 and 0.676 instead of 0.893 and 0.746. For rotations clipping moves numbers by ≤ 0.05. Clean one-position shift: 0.363 (k = −1) and 0.400 (k = +1); the reported 0.474 includes 0.11 from sequences where p0+1 is the old runner-up, and 0.381 is diluted by the 24% of sequences with p0 = 0. The effect is specific to the clock among rotating modes (0.04–0.15 for the others), but setting the level or runner coordinate to 0 also destroys the answer (0.30, 0.22).
- **e23 / K5: centred R² of activations over all steps is a lenient measure of 'the core runs almost by itself', because most of the variance is the common level of the thresholds.** (affects conclusion: partly)
  - test: Two baselines (no recurrence; thresholds = c_j × oracle running max), R² of thresholds in units of M, and R² of the core clock coordinate, simulated against real (c05).
  - result: The clock-free oracle baseline reaches 0.979, 0.997, 0.879, 0.876, 0.988, 0.820, comparable to variant A (0.877–0.981). Variant B (0.989–0.995) is clearly above it for units 1, 6, 8. Thresholds/M under B: R² 0.92, 0.98, 0.96 for θ_1, θ_6, θ_8, but 0.38 for θ_7 and negative for θ_4 (rms error 0.07–0.09·M, as large as their whole spread). Core clock R²: A 0.87, B 0.92, baseline 0.01. The statement holds for variant B; 'without any feedback term R² is 0.88–0.98' says little.
- **e10 / T2 and figure panel (e): the R² of 0.985–0.992 is for a hypothetical step (all 13 ordinary units on, pulses ignored, no input) applied to real states. The panel title and report §3.2 read as statements about real steps without events.** (affects conclusion: partly)
  - test: Share of states with the resting gate pattern; the Σ-prediction against the real next scores U·h_(t+1) under e10's R² convention, for all states, no-event steps, and no-event steps with the resting pattern; a null for the residual 0.095 (c06, c12).
  - result: The resting pattern holds in 10.3% of states (21% of quiet states with a standing max). Real steps: R² 0.56–0.76 over all states, 0.80–0.91 on no-event steps (0.83–0.94 for positions 0–7), 0.92–0.99 when the pattern at t+1 is the resting one. The top-scored input survives a real no-event step in 97.5–97.7%. The residual 0.095 is far below random readouts (mean 0.44–0.47, minimum 0.15), so T1 is meaningful.
- **fig_mechanism panel (a) and e20: the gap is defined with max(S, 0), so the top bin (0.85, 1] contains every state with S ≤ 0, while the legend says (M − S)/M.** (affects conclusion: no)
  - test: Share of S ≤ 0 states per bin and age; medians recomputed with the literal definition and for S ≤ 0 only (c09).
  - result: The top bin is 32–78% S ≤ 0 states in the figure and 31–67% in e20. With the literal definition the radii are up to 7% smaller at young ages (0.927 against 0.996 at age 1) and the phases are unchanged (≤ 0.7°). O1's half-ranges change little (unit 7: 2.6 for S > 0, 3.7 for S ≤ 0).
- **e24 / K6: only the projection of z_c on the phase of its median is regressed, with an uncentred R², so a wandering phase or a poor fit of the variation could be hidden.** (affects conclusion: no)
  - test: Centred R² of the same fit; rms of the discarded perpendicular component; complex fit z_c ≈ α·M + β·S⁺ with both conventions (c08).
  - result: Centred R² 0.86–0.99; the perpendicular component is 8–16% of the real part; complex uncentred R² 0.940–0.996 (centred 0.81–0.99); phase of α within 6° of the median phase. The law holds; ages 5–8 are noticeably looser than ages 1–4.

## Bugs or issues

- e17_wipe_comparators.py lines 23/27: the printed header says the population is 'pM <= 8 - k', but the code uses t0 <= 9. The pooled P(answer = max) (0.868, 0.873, 0.753, 0.816, 0.977, 0.999) therefore includes the wipe at the last step. With t0 <= 8: 0.973, 1.000, 0.881, 0.893, 0.977, 1.000.
- e17_wipe_comparators.py lines 34-40: the split by wipe step is done only for k = 2, the most favourable lag. Other cells are much lower (k = 3, t0 = 8: 0.42; k = 1, t0 = 8: 0.79; k = 4, t0 = 7: 0.89); in those the network answers the wipe step itself.
- e15_two_score_vectors.py line 43: R² is pooled over the ten logit indices. Index 9 holds 33-49% of the sum of squares with R² >= 0.994. For indices 0..7: 0.82-0.96 pooled, 0.14-0.98 per index.
- e15_two_score_vectors.py line 48: the line labelled 'C(idx) (coefficient of x_9, from the a1=5 fit)' prints the weight product U[:, COMP] @ w[COMP]; the fitted C is never printed. Fitted C_9 = 58.2-58.7, so S3's 58.2 is fine.
- e15_two_score_vectors.py lines 62-63: dead duplicate assignment of `new`. Parts (i) and (iii) are superseded by e15b and should not be quoted.
- e14_score_updates.py lines 63-75: the persistence loop has no g >= 1 condition although claim D2 says 'such records'. With it, c1 = 41.1, 42.7, 34.1, 29.5, 25.5, 20.2.
- e14_score_updates.py lines 63-75: the cohort changes with k (t <= 9 - k), so the trend 40.7 -> 23.8 is a composition effect. In a fixed cohort the lead grows (median y(k)/y(0) = 1.18, 1.35, 1.49, 1.53, 1.60).
- e14 / claim D1: c1/(1 - cos(42.75°·Δ)) for Δ = 1 is 66.1, not 65 (1 - cos was rounded to 0.27).
- e14 / claim D1: the fit pools both orders of the two positions, whose c1 differ by up to 2.3× (Δ = 4: 60.1 vs 139.8).
- e18_clock_before_record.py lines 47-51: the printed outcome categories are not mutually exclusive (p0 ± j can equal q0), and the denominator includes sequences where p0 - k does not exist (24% for k = +1, 46% for k = +2). Conditioned: 0.233, 0.363, 0.400, 0.365 for k = -2, -1, +1, +2.
- e18_clock_before_record.py lines 36-38: the edited state is not clipped at zero. 98-100% of states are unreachable for f = 1.5 and 2 (clipped results 0.838, 0.676 instead of 0.893, 0.746); 33-68% for rotations (change <= 0.05); 8-9% for f < 1 (no change).
- e22_separation_errors.py: correct, but pooling over the record step and the order of p_S, p_T hides the source of the D >= 6 errors. D = 8 is the average of 0.708 and 0.005; D = 7 is 0.022 with the record at step 8 and 0.241 at step 9.
- e15b_two_score_vectors.py line 41: the label says 'comparators all on at t and t-1' but line 43 conditions on step t only (the claim text matches the code). With both steps: 0.529, 0.684, 0.763, 0.849, 0.915, 0.932, 0.955.
- e15b / claim S1: the value 0.891 at t = 9 is a mixture: 0.305 when the max is at position 0, 0.888-0.993 for the other positions. Ages 3 and 4 are systematically lower (0.89, 0.92) at every t.
- e20_unit_orbits.py docstring and line 14: 'the runner-up has not changed for >= 2 steps' is not ensured by a2 >= 2 at a1 = 1 (24% of the selected states). Line 27 computes frac0 and never uses it.
- e23_core_alone.py docstring: announces a comparison of thresholds; the code compares only activations and gates. Measured here: threshold R² under variant B is 0.92-0.98 for θ_1, θ_6, θ_8, 0.38 for θ_7 and negative for θ_4.
- e16_m_scores.py line 14: rebinds T (the sequence length imported from common) to the third-largest value. Harmless in this script, fragile if code is added below.
- e10_tape_shift.py lines 20-21: the comment says the mean over positions 0..8 is removed and the subdiagonal is shown; the code prints the raw superdiagonal Sig[i, i+1].
- Claim T1: 'all other entries in rows 0-6 <= 0.42' is loose; the largest is 0.387 (0.42 is Sig[7,1]).
- Claim T4: 'ages 2..9 and v > 0.2: medians between -4.5 and -14.5' is loose; the range is -4.9 to -14.5 (-4.5 belongs to the bin (0.1, 0.2]).
- e24_core_clock.py: two R² conventions under the same label (centred on line 38, uncentred on line 48); λ hard-coded on line 36 instead of LAM['clock']; LEFT imported but unused.
- e25_numbers_for_report.py line 9: the label '(all four comparators on)' is attached to a sum over all 13 non-pulse units.
- modes.py lines 25-28: mode names are assigned by descending |λ| with only a length assert. Correct for these weights; other weights (for example the replicas) would be mislabelled silently.
- fig_mechanism.py line 36 vs legend on line 164: the gap is binned as (M - max(S, 0))/M but labelled '(M − S)/M'; the top bin is 32-78% states with S <= 0. The title 'radius = gap' is approximate (the radius is 1.4-1.7× the gap in the first bin).
- fig_mechanism.py lines 97-110 and 236-237: the panel (f) curves are conditional (S_t > 0; second curve also max not current and index 9 excluded), while the axis says 'fraction of sequences'. Unconditional: 0.675-0.953 and 0.500-0.858; with index 9 allowed 0.02-0.48.
- fig_mechanism.py line 247: `--redraw` loads the cache with no check that it matches the code. The cache is currently identical to a recomputation, although the .npz is one minute older than the .py.
- Report draft §3.5 item 1 and §10/§11 ('wipe at step 8 or earlier → 99.7-100%', '≈100%'): supported only for k = 2. Over all lags it is 0.957 for t0 <= 8 and 0.986 for t0 <= 7.
- Report draft §1 ('m ranks the maximum first (94-97% of states)'): omits t = 9 (0.891) and the three conditions (older inputs only, S_t > 0, max not the current input).
- Lab notebook e14 ('the lead decays about 10% per step') and report §7/§11 item 7 ('errors at separations 7 and 8 as a consequence of the single-harmonic deposit'): not supported, see the objections.