# Round-1 arm D_geometry: structured result

## Summary

ARM D (geometry). NOTE FIRST: findings.md was NOT written. The Write tool refused it ("Subagents should return findings as text, not write report files") and I did not work around the block. This object carries the findings; every table is in the script outputs (*.txt) listed under files. All work is on N(0,1), float64 weights, fresh seeds 1000-1081. No figures. Nothing written outside my folder.

VERDICT ON THE HYPOTHESIS
- "Age of the max on a circle with fixed directions": SUPPORTED. Clock phase φ(a) = 36, 79, 123, 166, 209, 251, 294, 337° for a = 2..9, independent of M and of a2 (within ±8° in all 72 (a1,a2) cells). A nearest-phase decoder gets the age right in 98.9-99.9 % of states with S/M ≤ 0.4, 96 % for 0.4-0.6, 84 % for 0.6-0.8, 38 % for S/M > 0.9.
- "Centre depends on M and S": SUPPORTED, with a second term. centre ≈ 10.157·[ℓ·M·R_level + (r0·M + r1·S)·R_runner], ℓ = 0.90-1.01, r0 = 2.9→1.55 (a = 2→9), r1 ≈ 5.2. The level mode holds M only; the runner mode (λ = 0.9735, eigenvector negative on units 0, 10, 11, 12, 14, 15) lowers the centre of the six large wave units by 6-10 per unit of S.
- "Radius depends on M and S": SUPPORTED, but it is NOT ∝ M − S. Clock coordinate |z|/10.157 ≈ ρ(a)(M − S) + δ·S with ρ(a) ≈ 1.02·0.951^(a−2) and δ = 0.07-0.39 depending on (a1,a2). In 13-dim Euclidean norm the mean orbit has radius ≈ 23.5·(M − 0.63·S).
- (a) "off-centre so that ReLUs do not cut it": SUPPORTED WITH TWO QUALIFICATIONS. Without the centre 48-55 % of (unit, age) entries of the orbit are negative, with it 4-6 %; removing the centre in the running net clips 5.5-6 of 13 units at the next step and takes accuracy from 97 % to 11-21 %. But (i) the margin is zero: the orbit is tangent to the ReLU boundary (units 10 and 11 are OFF at max-age 6 and 5 in 67-98 % of states, pre-activation ≈ −0.5·M), and (ii) those grazing cuts cost the clock ≤ 2 %.
- (b) EXACT. (c) SUPPORTED: the centre is carried by the two real eigenvalues of A near 1 (1.0099 and 0.9735); the first coordinate is 0.92-0.94·10.157·M and is also 80 % of unit 4's threshold, so the "bias" is the stored max itself.

D1. Gram matrices of [g = M−S, S, T, x_t, h_t[ORD]] per (t,a1,a2) cell, seeds 1001-1008, 1.6M sequences, 11,218,990 states with S > 0. Centred R² of all 13 units within an age a1 = 2..9: (g,S) per a1: 0.44-0.54; + x_t: 0.82-0.92; + T: 0.84-0.92; (g,S,T,x_t) per (a1,a2): 0.94-0.98. The missing half with (g,S) is the current input in the comparators. Unit 2 ≥ 0.95 with (g,S) alone. u(a), a = 2..9: 99.73 % of its variance in a plane 5.1°/7.9° from the clock plane; rotation 42.1°/step (spiral fit 41.96°, |λ| 0.975); centre has cos 0.992 with R_level (level 0.951, runner 1.74, all others ≤ 0.03); 13-dim radius 26.2→22.0. v(a) (state per unit S at M = S) also rotates: 96.3 % planar, 17° from the clock plane, 40°/step, radius 12.0→13.8, centre cos 0.90 with R_level (runner 6.0). Keeping only level + runner + clock (+echo) parts of u and v reproduces 89.4 % of the variance over a1 = 2..9 (free u,v: 93.7 %); dropping the runner mode: 83.0 %; dropping the rotation of the S part: 84.9 %; no rotation: 45.8 %.

D2. Mean pre-input orbit W·h_(t−1) per unit M in narrow S/M bins (seeds 1041-1044). Clock radius / centre at S/M = 0: 0.82-1.22 for 9 of 13 units (0, 1, 3, 8, 10, 11, 12, 14, 15); ≤ 0.45 for units 2, 6, 7, 13, which stay positive. Orbit minima (per M) at S/M = 0, 0.5, 0.9: unit 10: −0.42, −0.47, −0.48 at age 6 (OFF 89, 98, 92 %); unit 11: 0.02, −0.59, −0.50 at age 5 (OFF 67, 96, 95 %); unit 12: 0.53, 0.33, 0.96 at age 4 (OFF 21, 42, 17 %); unit 15: −1.14 (age 4), −0.16 (age 3), −0.86 (age 2); unit 0: −4.40, −3.15, −2.06 at age 2 (always OFF; that is the transient after the unit-5 pulse, not the circle). The age of every unit's orbit minimum is the age of its largest measured clipping frequency. Centre removed (linear model): clock coordinate of ReLU(rotating part) is 0.30-0.90 (mean 0.56) of the original with phase errors up to ±50°.

D3. Exact: rotation about c ≠ 0 by a linear map needs A·c = c; real eigenvalues of A: 1.0099, 0.9735, 0.231, 0.001, −0.928; level eigenvector positive (A has 67 negative entries of 169); det(I − R_clock) = 0.508. Intervention (seed 1021, 200k; t0 = 3, 5, 7; a1 ≥ 2; n = 93,683 / 131,439 / 149,443): level × f. f = 0: 6.6-7.2 negative entries, 5.5-6.0 units newly OFF next step; clock ratio after one step 0.54 / 0.76 / 0.93 (autonomous flow), phase +26..+37°, then roughly constant; 0.87-0.98 if comparators are held on by negative inputs; level regenerated to 0.25-0.30. Accuracy, quiet continuation (later inputs −|x|): 97 % → 11 / 21 / 20 % (f = 0), 30-37 % (f = 0.5), 66-68 % (f = 0.75), 87-94 % (f = 1.25), 85-91 % (f = 1.5). Real inputs: 95 % → 35 / 22 / 19 % at f = 0. The clock is not destroyed as a rotation, but the answers are. Controls: clock × 0 costs 5-6 points and clips nothing; clock × 1.5 clips 1.6 extra units and costs 15-22 points.

D4. Impulse reproduced: |n| = 34.99, in-plane radius 22.10, off-plane rms 0.88 (0.94 with 1/(n−1); owner's 0.91 not exactly reproduced), rotation 41.47°/step. Centre = level mode (110 % of the projection on the mean, runner −8 %); rotation = clock mode (103 % of the variance, short −6 %). (1,0,0,0,2,…): level/M = 0.92-0.96 as in the impulse, clock restarts at 44, 87, 130, 174°, |z| = 1.22-1.23 × ρ(a)(M−S). (2,0,0,0,1,…): level/M 0.90-0.93, |z| = 1.51 × ρ(a)(M−S). Background −0.5: |n| = 42.04, radius 23.23, ρ 4.5 % larger, unit 5 no longer at threshold; ratios 1.17 and 1.45.

D5. A second rotating component exists in h2, h3, alt, short. Its coordinate is F_m(a1−a2)·λ_m^(min(a1,a2)−2): the phase tracks the more recent of the two top-2 arrivals, not the runner-up as such. |F_h2| = 0.53, 0.08, 0.52, 0.98, 1.53, 1.68, 1.35 for d = +1..+7 and 0.39, 0.11, 0.45, 1.08, 1.51, 1.74, 1.47 for d = −1..−7. Amplitude grows with S and falls with lower inputs (τ/v ≈ −0.65), closest to S − T; not ∝ M − S.

D6. No. For S/M ≤ 0.1 the median |z|/(10.157·M) is 1.009, 0.965, 0.924, 0.885, 0.838, 0.793, 0.746, 0.702 for a = 2..9 against free decay 1.020 … 0.717; removing the wave-unit ReLUs changes |z| by 0.9-1.1 %. Reason: |L_clock| ≤ 0.22 on wave units (96.6 % of its squared norm is on the comparators).

## Claims

### D-01 [EXACT, confidence high]

A linear map that rotates about a point c ≠ 0 must fix c (A·c = c). A = W[ORD,ORD] has exactly two real eigenvalues near 1: 1.0099 (level) and 0.9735 (runner); the others are 0.231, 0.001, −0.928 and four rotating pairs. The level eigenvector is positive on all 13 ordinary units although A has 67 negative entries of 169. Inside the clock plane no fixed point exists.

- numbers: |A·c − c|/|c| = 0.0099 for c = R_level; unit-norm entries of R_level 0.054-0.460; det(I − R_2x2) = |1 − λ_clock|² = 0.508; level grows ×1.071 over 7 resting steps while the clock decays to 0.703 (radius/centre ×0.9416 per step); L_level·w_ORD = −0.045, L_runner·w_ORD = 1.03.
- how checked: d3b_impulse.py: eigenvalues, residuals, sign count. Could have failed if the leading eigenvector had mixed sign or the eigenvalue were far from 1. Impulse run confirms level/10.157 = 0.927→0.990 for t = 2..9 (×1.010 per step).
- caveats: The centre is fixed only up to 1 %/step growth. The resting gate pattern (13 ordinary on, pulses off) holds in only 18.8 % of states with a1 ≥ 2, so A is a linearisation (rms ReLU correction 6.6 % of the state norm for a1 ≥ 3).

### D-02 [MIXED, confidence high]

The amplitude of the centre is a stored value: the level coordinate is b·10.157·M with b independent of S, and the same coordinate supplies most of the thresholds, so the 'bias' is the stored max itself.

- numbers: b = 0.918, 0.918, 0.914, 0.919, 0.924, 0.930, 0.934, 0.938 for a1 = 2..9 (seed 1000, 200k). Regression coefficients of the level on (M−S) and on S: 0.927/0.927 at a1 = 2, 0.961/0.982 at a1 = 9. Per unit of level/10.157 the level contributes 0.796 to θ4 and 0.631, 0.749, 0.782, 0.748 to θ1, θ6, θ7, θ8 (exact). θ4 ≈ 0.815·M + 0.18·S ± 0.16·|z|/10.157.
- how checked: d0_verify.py, d1_vectors.py, d3c_thresholds.py. Independent seeds 1041-1044 (d2_orbit.py) give level/10.157M = 0.91-1.01 at S/M = 0 and 0.90-0.94 at S/M = 0.9.
- caveats: Centred R² within an age cell is 0.9963 (a1 = 2) falling to 0.986 (a1 = 9); uncentred 0.998-0.9994. The lead's 'R² ≥ 0.997' holds only uncentred. Level/M drops by up to 0.07 at late ages when S/M → 0.9, and by 6-11 % after a second smaller pulse in exact two-pulse runs.

### D-03 [FIT, confidence high]

Regressing h_t[ORD] on (M−S)·u(a1) + S·v(a1) explains about half of the variance of the 13 ordinary units; almost all of the rest is the current input in the comparators. Splitting by a2 and adding T and x_t reaches 94-98 %.

- numbers: Centred R², all 13 units, within a1 = 2..9: (g,S) per a1 0.436-0.540; (g,S,x_t) 0.824-0.916; (g,S,T,x_t) 0.837-0.924; (g,S) per (a1,a2) 0.626-0.689; (g,S,T,x_t) per (a1,a2) 0.942-0.980. Per unit with (g,S): unit 2 0.950-0.999; comparators 0.01-0.35 (≥ 0.984 once x_t is added); unit 3 0.11-0.42; unit 15 0.11-0.83. Pooled over a1 = 2..9: (g,S) 0.620 (wave units 0.878); (g,S,T,x_t) 0.937 (wave 0.898). n = 11,218,990 states with S > 0.
- how checked: d1_gram.py + d1_r2.py (exact least squares from Gram matrices). Repeated on all M > 0 states with positive parts (14,400,219 states): same picture. Per-(t,a1,a2) fits add ≤ 0.01.
- caveats: R² near 0 for unit 0 at a1 = 2, unit 11 at a1 = 5 and unit 10 at a1 = 6 is because those units are clipped there (OFF 99.9 %, 92 %, 95 %) and have almost no variance. The pooled figure counts variation between ages as explained.

### D-04 [FIT, confidence high]

The fitted vectors u(a), a = 2..9 (state per unit of M − S), form a planar spiral about a centre: the plane is the clock plane, the centre is the level eigenvector, the rotation is the clock's.

- numbers: PC0+PC1 = 0.9973 of the variance about the mean; rms distance from mean 24.10, off-plane rms 1.25; principal angles to span(Re,Im R_clock) 5.1° and 7.9°; rotation per step 33.8, 44.5, 50.1, 46.0, 39.1, 38.5, 42.7 (mean 42.1°); spiral fit |λ| = 0.975 at 41.96° with 0.30 % unexplained; centre norm 39.07, cos(centre, R_level) = 0.992, centre coordinates level 0.951, runner 1.74, others ≤ 0.03; 13-dim radius 26.2 (a = 2) → 22.0 (a = 9). u's clock coordinate: 1.021@36°, 0.990@79°, 0.952@123°, 0.928@166°, 0.898@209°, 0.870@251°, 0.834@294°, 0.806@337°.
- how checked: d1_geometry.py, d1_vectors.py. Same within 0.01 amplitude and 1° phase with regressors (g,S) only and on the M > 0 data set. Per (a1,a2) cell the phase of u's clock coordinate is within ±6° of the list in 69 of 72 cells and within ±8° in all.
- caveats: u's decay 0.975/step is slower than the free 0.951 because one line in S/M is fitted to a slightly concave dependence: at S/M ≤ 0.1 the measured amplitude at a = 9 is 0.702, not 0.806. In cells with a2 > a1 u's clock amplitude is 1.037, 0.988, 0.946, 0.906, 0.853, 0.807, 0.763 for a1 = 2..8 (ratios 0.94-0.96, free decay).

### D-05 [FIT, confidence medium]

The S part v(a) (state per unit of S when M = S) does not reduce to a centre: it also rotates with the age of the max, with about half the 13-dim radius of u, in a plane 17° from the clock plane, and its centre contains a large runner-mode term.

- numbers: PC0+PC1 = 0.963; principal angles 16.8°, 17.7°; rotation mean 40.4°/step; spiral |λ| 1.020 at 39.5°, 4.3 % unexplained; radius 12.0 → 13.8; centre cos with R_level 0.900 (level 0.936, runner 6.01, short 0.35, h2 0.18); v's clock coordinate 0.22-0.34 lagging u by 8-52°; covariance share of the variance about the mean: clock 0.52, short 0.38.
- how checked: d1_geometry.py. With regressors (g,S) only the radius of v is 7.5 → 9.4 (T absorbs part of it), so the size depends on the controls; the existence of the rotation does not.
- caveats: v pooled over a2 is a mixture: per (a1,a2) cell the S coefficient of each oscillating mode is F_m(a1−a2)·λ_m^(min(a1,a2)−2), see D-17.

### D-06 [MEASURED, confidence high]

Centre and radius as functions of M and S. Centre ≈ 10.157·[ℓ·M·R_level + (r0·M + r1·S)·R_runner]; radius (clock coordinate) |z|/10.157 ≈ ρ(a)(M − S) + δ·S. In 13-dim Euclidean norm the mean orbit over a = 2..9 has radius ≈ 23.5·(M − 0.63·S) and its distance from the origin barely changes with S.

- numbers: Mean orbit per unit M in bins S/M ≤ 0.05, 0.45-0.55, 0.85-0.95 (a = 2..9): level 0.91→1.01 / 0.91→0.94 / 0.90→0.94; runner 2.91→1.55 / 5.05→4.18 / 6.84→6.04 (so r1 = 5.2 ± 0.3 for a ≥ 3); clock |z| 0.96→0.70 / 0.63→0.51 / 0.34→0.27 (δ ≈ 0.22-0.32); phases 36..336 / 35..333 / 29..325°. PCA of the mean orbit: |mean| 38.86 / 34.81 / 33.52; origin-plane distance 35.03 / 33.09 / 32.79; in-plane rms radius 23.53 / 16.26 / 10.19; cos(mean, R_level) 0.987 / 0.935 / 0.849; rms norm of clock part 24.01 / 15.92 / 7.88. Measured centres of units at the three S/M: unit 0: 13.6, 8.5, 4.8; unit 10: 10.9, 7.5, 5.1; unit 11: 11.4, 8.0, 5.3; unit 12: 16.7, 12.4, 9.2; unit 2: 14.4, 16.0, 17.7.
- how checked: d2_orbit.py, seeds 1041-1044 (4×200k; 1,087-219,656 states per age and bin). At S/M = 0 the origin-plane distance 35.03 and radius 23.5 match the impulse (34.99, 22.10). The independent regression (seeds 1001-1008) evaluated at S/M = 0, 0.5, 0.9 reproduces the binned orbits with rms difference 0.40, 0.08, 0.11 against orbit rms 12.6, 10.7, 9.8.
- caveats: These are mean orbits of the pre-input state W·h_(t−1); individual states scatter with a2, T and later inputs. The keep-only level+runner+clock(+echo) description reproduces 89.4 % of the variance over a1 = 2..9 (wave units 83.2 %) against 93.7 % for free u, v. It leaves out the a1 = 0, 1 transients, the harmonics tied to a2, the T dependence and the piecewise-linear dependence on S/M.

### D-07 [MIXED, confidence high]

'Radius ∝ M − S' is the weakest part of the hypothesis: the clock radius does not go to zero as S → M, and in exact two-pulse runs it exceeds ρ(a)(M − S) by a factor 1.2-1.5 at S = M/2 and 1.8-3.4 at S = 0.9·M.

- numbers: Median |z|/(10.157·M) at S/M in (0.9, 1] by (a1,a2) cell: 0.07-0.39 (0.07-0.12 when the runner-up arrived one step after the max; 0.23-0.31 when the max arrived one step after the old max; 0.31-0.39 for (9,1), (9,2), (8,0), (8,1), (2,0)); a pure law gives ≤ 0.05. By S/M bin at a = 2: 1.009, 0.841, 0.694, 0.537, 0.377, 0.252; at a = 9: 0.702, 0.641, 0.542, 0.441, 0.326, 0.176. Single-term fit z ≈ c(a)·10.157·(M−S⁺): |c| = 1.06, 1.09, 1.06, 1.03, 1.04, 1.03, 1.03, 1.01, 0.99 for a = 1..9 with uncentred R² 0.971 falling to 0.854. Two-pulse (exact): M=2@4, S=1@0: |z|/prediction 1.22-1.23; M=2@0, S=1@4: 1.51; S = 1.8: 1.81-1.88 and 3.40; S = 0.2: 1.08-1.12.
- how checked: d6_radius.py part C (seeds 1031-1034), d6b_radius_cells.py (seeds 1061-1064), d0_verify.py, d4b_twopulse.py (deterministic). The lead's e03 script rerun gives b_M = 1.01→0.78 and b_S = −0.85→−0.57 for a = 1..9, i.e. the S coefficient is 0.73-0.86 of the M coefficient.
- caveats: The remainder per unit of S is not a constant: 0.06-0.40 at S = M/2, 0.11-0.21 at S = 0.9M, 0.27-0.87 at S = 0.1M in two-pulse runs, depending on order, distance and background.

### D-08 [MEASURED, confidence high]

The directions on the circle have a fixed meaning: the age of the max can be decoded from the clock phase alone, except when S is close to M.

- numbers: Nearest-phase decoder of a1 (9 classes, chance 0.11), accuracy / median |phase error| by S⁺/M bin: ≤0.2: 0.989 / 0.6°; 0.2-0.4: 0.999 / 2.1°; 0.4-0.6: 0.962 / 4.3°; 0.6-0.8: 0.840 / 8.2°; 0.8-0.9: 0.653 / 14.0°; 0.9-1.0: 0.377 / 26.1°. n per bin 611,562-1,218,528 states.
- how checked: d1_phase.py, seeds 1071-1074; reference phases from the impulse with background −0.5 (0.1, 35.9, 78.9, 122.7, 165.2, 207.8, 250.2, 292.8, 335.8°). Could have failed and does fail near S = M, where the mean signed error reaches −17 to −29° for a1 = 2..4.
- caveats: Ages 1 and 9 are only 24° apart on the circle.

### D-09 [MEASURED, confidence high]

The orbit is tangent to the ReLU boundary rather than well inside it. At S/M = 0 the clock radius of 9 of the 13 ordinary units is within ±22 % of their centre; units 10 and 11 dip just below zero at max-age 6 and 5 for every S/M; units 2, 6, 7, 13 stay clearly positive. The age of each unit's orbit minimum is the age of its largest measured clipping frequency.

- numbers: radius/centre at S/M = 0: units 0, 1, 3, 8, 10, 11, 12, 14, 15: 0.82, 0.89, 1.16, 0.86, 1.22, 1.00, 0.96, 1.02, 1.19; units 2, 6, 7, 13: 0.26, 0.45, 0.25, 0.25. Orbit minimum per unit M at S/M = 0 / 0.5 / 0.9: unit 10: −0.42 / −0.47 / −0.48 (age 6), OFF 0.888 / 0.978 / 0.916; unit 11: 0.02 / −0.59 / −0.50 (age 5), OFF 0.666 / 0.959 / 0.949; unit 12: 0.53 / 0.33 / 0.96 (age 4), OFF 0.213 / 0.416 / 0.167; unit 15: −1.14 (age 4) / −0.16 (age 3) / −0.86 (age 2), OFF 0.708 / 0.494 / 0.568; unit 0: −4.40 / −3.15 / −2.06 (age 2), OFF 1.000. All M > 0 states (800k sequences): unit 0 OFF 0.999 at a1 = 2; unit 11 0.919 at a1 = 5; unit 10 0.950 at a1 = 6; unit 15 0.31-0.44 at a1 = 2..4; unit 12 0.294 at a1 = 4; units 2, 13, 7, 6 never above 0.001, 0.029, 0.047, 0.073 for a1 ≥ 2.
- how checked: d2_orbit.py (seeds 1041-1044) and d2_clip.py (seeds 1011-1014), independent samples. Impulse (exact): clipped pre-activations −0.45 (unit 14, age 2), −0.41 and −1.45 (unit 15, ages 3, 4), −0.02 (unit 11, age 5), −0.46 (unit 10, age 6), on swings of 20-31.
- caveats: Unit 0 at age 2 is far below zero and is the transient after the unit-5 pulse, not a grazing cut. For comparators 'OFF' is mostly caused by the current input. Whether the tangency has a function is not determined. The tangency of units 10, 11 at S/M = 0.5 and 0.9 relies on the centre falling with S (runner mode): with a centre ∝ M only, unit 10's minimum at S/M = 0.9 would be +5.3·M.

### D-10 [MIXED, confidence medium]

Linear-model answer to 'what if the centre were removed': about half of the (unit, age) entries of the rotating part are negative, half of its squared norm is cut by the ReLU, and one rectification leaves on average 56 % of the clock coordinate with phase errors of up to ±50°.

- numbers: S/M = 0 / 0.5 / 0.9: share of entries negative 0.481 / 0.519 / 0.548; share of squared norm in negative entries 0.527 / 0.527 / 0.577; clock coordinate of ReLU(y − centre) over that of y − centre, a = 2..9: 0.53, 0.32, 0.48, 0.74, 0.58, 0.69, 0.67, 0.49 (S/M = 0) with phase errors −27, −5, +35, +23, +16, +6, −15, −46°; level regenerated by the rectification 0.13-0.21 / 0.10-0.14 / 0.06-0.09 of the original. With the centre: 4.8 / 3.8 / 5.8 % of entries negative.
- how checked: d2_orbit.py: exact mode decomposition of the measured mean orbits (centre = level + runner + trans + echo parts). Compared with the running-network intervention D-11, which gives the same order of magnitude (clock 0.29-0.41 of the original after clipping the level-free state at t0 = 3).
- caveats: Uses mean orbits and removes the whole real-mode part; the intervention D-11 removes only the level.

### D-11 [INTERVENTION, confidence high]

Removing the level coordinate at a step with max-age ≥ 2 clips about six ordinary units at the next step, cuts the clock amplitude and advances its phase in one step, after which the clock keeps rotating; the ReLU regenerates about 0.3 of the level. The clock is not destroyed as a rotation.

- numbers: f = 0, t0 = 3 / 5 / 7 (n = 93,683 / 131,439 / 149,443): negative entries of the patched state 6.62 / 7.22 / 7.10 of 13; ordinary units newly OFF at the next step 6.04 / 5.52 / 5.49 (autonomous flow), 5.73 / 5.75 / 5.92 (real inputs), 4.40 / 4.42 / 4.66 (comparators held on by inputs −2M); then 1.5-3.0 per step. Clock |z'/z| median [IQR] after one step, autonomous: 0.54 [0.46, 0.65] / 0.76 [0.57, 1.04] / 0.93 [0.70, 1.05], phase +37° / +31° / +26°; at the last step 0.54 / 0.73 / 0.97. Comparators held on: 0.87 / 0.92 / 0.98, +26..+27°. Level'/level after one step 0.29 / 0.28 / 0.25. f = 0.5: 3.36 units newly OFF, clock 0.82, +8°; f = 0.75: 2.05, 0.92, +4°; f = 1.25 and 1.5: none newly OFF, clock ×1.03-1.04 after six steps. Units newly OFF at f = 0, t0 = 3: unit 11 0.99, 0 0.97, 14 0.90, 12 0.89, 3 0.48, 15 0.34, 8 0.27; units 2 and 13 ≤ 0.06.
- how checked: d3_intervene.py, seed 1021, 200k sequences; f = 1 reproduces the baseline exactly; other mode coordinates unchanged by construction (reconstruction error 1e-13). Deterministic impulse example (d3b_impulse.py): 7 ordinary units OFF at t = 4 instead of 1, |z'/z| = 0.559-0.596, phase +32..+41°, level'/level 0.268-0.299; of the level regained, 0.209 comes through unit 12.
- caveats: With real inputs the patched clock is scattered in about half of the sequences (phase IQR −79° to +93° at t = 9 for t0 = 3) because unit 4 fires spuriously in 45-53 % of sequences. In the impulse example comparator 8's clipping alone moves z by 0.43·|z| and the deeply clipped wave units 12, 15, 11 by 0.26, 0.15, 0.13, so the loss is not only through the comparators. If the patched state is first clipped at zero the clock ratio is 0.41 / 0.56 / 0.66 (autonomous).

### D-12 [INTERVENTION, confidence high]

Lowering the centre destroys the final answers with a monotone dose-response, even when no later input can change the top two.

- numbers: Quiet continuation (later inputs replaced by −|x_t|; scored on sequences with S(t0) > 0; unit 4 fires extra in ≤ 2.3 %), accuracy at t0 = 3 / 5 / 7: f = 1: 0.973 / 0.972 / 0.971; f = 0.75: 0.681 / 0.660 / 0.666; f = 0.5: 0.297 / 0.333 / 0.366; f = 0.25: 0.162 / 0.251 / 0.239; f = 0: 0.111 / 0.212 / 0.196. Real later inputs, all sequences: f = 1: 0.949 / 0.952 / 0.955; f = 0.75: 0.751 / 0.641 / 0.638; f = 0.5: 0.542 / 0.393 / 0.353; f = 0: 0.354 / 0.224 / 0.186.
- how checked: d3_intervene.py; the quiet continuation removes the confound of spurious new-max firings; the reference run under the same quiet inputs scores 97 %.
- caveats: Scaling the level also rescales the thresholds that read it (D-02), see the controls in D-13.

### D-13 [INTERVENTION, confidence high]

Controls separate the two roles of the level. Raising it clips nothing and costs far less than lowering it; enlarging the clock at fixed centre hurts more than erasing the clock. So a circle that is too large for its centre is harmful because it gets cut.

- numbers: Quiet continuation, accuracy at t0 = 3 / 5 / 7: level ×1.25: 0.873 / 0.869 / 0.944; level ×1.5: 0.847 / 0.846 / 0.913 (no unit newly OFF). Clock ×0: 0.920 / 0.914 / 0.924 (0.00 units newly OFF); clock ×0.5: 0.956 / 0.946 / 0.958; clock ×1.25: 0.915 / 0.901 / 0.908 (0.98 units newly OFF at t0 = 3); clock ×1.5: 0.753 / 0.816 / 0.817 (1.63 units newly OFF). Real inputs: level ×1.25: 0.788 / 0.780 / 0.836; ×1.5: 0.651 / 0.653 / 0.734, with unit 4 failing to fire in 13 % and 23 % of sequences at t0 = 3. Clock erased at t0 = 3 with real inputs: 0.684 overall, 0.911 if the max is kept (baseline 0.939), 0.52 by subtraction if the max is later replaced (baseline 0.957).
- how checked: d3_intervene.py, same sequences as D-11/D-12.
- caveats: Level ×0.75 (0.66-0.68) is worse than clock ×1.25-1.5 because it also makes the harmonic packet of D-17 relatively larger. There is no intervention that changes the centre without changing the thresholds that read the same coordinate.

### D-14 [EXACT, confidence high]

The owner's impulse plane is reproduced, and it decomposes cleanly: the centre is the level mode (reduced 8 % by the runner mode) and the rotation is the clock mode.

- numbers: x = (1,0,…,0), PCA of h_2..h_9 on 13 units: origin-plane distance 34.99; in-plane rms radius 22.10; off-plane rms 0.881 (0.941 with 1/(n−1)); PC0+PC1 = 99.84 %; rotation per step 38.4, 48.9, 48.7, 41.4, 36.3, 37.2, 39.5 (mean 41.47°). Mean state norm 38.18, cos with R_level 0.983. Share of the projection on the mean: level 1.100, runner −0.082, clock −0.013; of the perpendicular n: level 1.058, runner −0.056; of the variance about the mean: clock 1.027, short −0.058, level 0.014, trans 0.010. PCA plane 3.9° and 7.2° from the clock plane. Mode coordinates/10.157 for t = 2..9: level 0.927→0.990; runner 2.37→2.06; clock 0.964@36.4°, 0.908@79.6°, 0.862@123.4°, 0.820@166.2°, 0.774@209.1°, 0.736@251.8°, 0.699@294.6°, 0.665@337.3°; h2, h3 ≤ 0.11 from t = 2.
- how checked: d4_impulse.py. Owner's values: 34.99, 22.1, 0.91, about 41°.
- caveats: The off-plane rms 0.91 is not reproduced exactly (0.88 or 0.94 depending on normalisation). |n| < |mean| because R_level is 61.7° from the PCA plane (66.4° from the clock plane), not perpendicular.

### D-15 [EXACT, confidence high]

Two-pulse inputs: the centre scales with M (level per unit of the current max is the same as in the impulse after a larger second input) and the clock restarts at the new max's age; a smaller second input shrinks the radius by only 0.5-0.8·ρ·S and adds 4-6 per unit of S to the runner coordinate. The network answers correctly in all cases tried.

- numbers: (1,0,0,0,2,0,…), t = 6..9: level/M 0.923-0.957 (impulse 0.927-0.951); clock 1.176@44.1°, 1.118@86.8°, 1.063@129.6°, 1.009@173.8° = 1.22-1.23 × ρ(a)(M−S), +6..+8° ahead; residual 0.23-0.26 per unit S; (runner − r(a)·M)/S = 3.7-4.4. (2,0,0,0,1,0,…), t = 6..9: level/M 0.901-0.928 (impulse 0.961-0.990); clock 1.168@208.1° … 1.004@336.3° = 1.51 × prediction, in phase (−1°); residual 0.34-0.40; runner S part 4.6-4.9. Four-point PCA planes (t = 6..9): origin-plane distance 65.2 and 67.9 (1.86-1.94 × the impulse), circle radius 27.5 and 27.2 (1.23 × the impulse instead of 1.0).
- how checked: d4_impulse.py, d4b_twopulse.py, also for (M,S) positions (2,0), (0,2), (1,0), (0,1) and S = 0.2, 1.8: level/M 0.85-0.98 in all 20 runs.
- caveats: The two-pulse PCA planes have only four points and are tilted 20-63° from the clock plane by the harmonics; mode coordinates are the reliable numbers.

### D-16 [EXACT, confidence high]

A background of exactly 0 is special in three ways; none changes the conclusions.

- numbers: Impulse, background 0 vs −0.5: unit 5 pre-activation for t = 2..9: 0.81, −0.41, −0.64, −0.35, −0.11, −0.13, −0.37, −0.29 vs −7.75 … −8.66 (at threshold vs clearly off); comparators 6 and 8 OFF at t = 1 vs ON; clock ρ(2..9) 0.964→0.665 vs 1.007→0.700 (4.5 % larger); runner 2.37→2.06 vs 2.14→1.43; origin-plane distance 34.99 vs 42.04 (echo mode contributes 19 % of n), radius 22.10 vs 23.23, off-plane rms 0.88 vs 1.15, rotation 41.47° vs 41.07°; cos(mean, R_level) 0.983 vs 0.944. Two-pulse ratios 1.22 / 1.51 vs 1.17 / 1.45.
- how checked: d4_impulse.py, d4b_twopulse.py, d6_radius.py part B (backgrounds 0, −0.25, −0.5, −1).

### D-17 [FIT, confidence high]

There is a second rotating component tied to the runner-up, carried by the modes h2, h3, alt and short, but its phase tracks the age of the more recent of the two top-2 arrivals, min(a1,a2), not the runner-up's age as such: coordinate of the S coefficient = F_m(a1 − a2)·λ_m^(min(a1,a2) − 2), with F_m depending on d = a1 − a2 only and nearly symmetric in the sign of d for h2 and h3.

- numbers: F_h2 (amplitude@phase) for d = +1..+7: 0.53@343, 0.08@272, 0.52@267, 0.98@287, 1.53@283, 1.68@280, 1.35@284; for d = −1..−7: 0.39@352, 0.11@161, 0.45@251, 1.08@276, 1.51@287, 1.74@287, 1.47@289; spread across cells with the same d ≤ 0.09 for |d| ≥ 3. |F_h3|: 0.43/0.36, 0.62/0.52, 0.60/0.57, 0.78/0.60, 1.34/0.99, 1.68/1.44, 1.33/1.54. |F_alt|: 0.06/0.03, 0.88/1.25, 0.54/0.26, 0.56/0.77, 1.24/0.79, 1.80/1.27, 1.51/1.88. |F_short|: 0.85/0.87, 0.58/0.71, 0.28/0.38, 0.24/0.23, 0.65/0.47, 1.09/0.99, 1.09/1.31. Phase of F(d) vs F(−d): h2 within 16° for |d| ≠ 2, h3 within 19°. The max's own coefficient u has 0.02-0.2 of these modes. State-space size per unit S at min(a1,a2) = 2: h2 8.0, 1.3, 14.7, 25.2 for |d| = 1, 2, 4, 6 against 28.9 per unit of M − S for the clock; decay over 7 steps h2 0.63, alt 0.59, h3 0.29, short 0.12.
- how checked: d1_cells.py + d5_runner.py (cells with n ≥ 2000). Exact two-pulse runs agree: h2/S two steps after the later pulse = 1.10@292° (M@0, S@4) and 1.11@278° (M@4, S@0) vs F(±4) = 0.98@287°, 1.08@276°; 0.60@347° and 0.46@352° for d = ±1; then +82.6..83.4° and ×0.93-0.94 per step. Exact cell-mean traces (seed 1051, 60k) reproduce v within 8 % and 7° in the large cells.
- caveats: So the runner-up has no private circle whose phase means 'age of the runner-up'. When the runner-up is the old max the h2 phase restarts with the new max. F_m(d) is measured, not derived from the weights.

### D-18 [MIXED, confidence medium]

The amplitude of the harmonic packet grows with S and falls with the lower inputs; among the three options it is closest to S − T and it is not proportional to M − S.

- numbers: Regression, cells with |d| ≥ 4: τ/v = 0.65@177° (h2), 0.67@181° (h3), −0.59 (alt), 0.64@178° (short); u/v = 0.06, 0.09, 0.06, 0.10. Exact traces: the third-largest input alone carries −0.39 to −0.57 of v for h2 (the regression's T coefficient is 30-75 % larger because T stands in for lower inputs). Exact two-pulse scan (M = 2 at 0, S at 4, background −0.5): |h2|/10.157 two steps after S = 0.40, 1.10, 1.65 for S = 0.2, 1.0, 1.8.
- how checked: d5_runner.py parts B, C; d4b_twopulse.py. The regression-vs-trace comparison could have failed and did show the bias of the T coefficient, which is reported.
- caveats: Not an exact proportionality: with S = T the regression still leaves about 0.25-0.35 of v, and the amplitude also depends on |a1 − a2| through F.

### D-19 [EXACT, confidence high]

Clipping of wave units does not limit the clock radius. The clock's left vector is small on the wave units, so a clipped wave unit mostly excites the harmonics, not the clock.

- numbers: |L_clock| on comparators 1, 6, 7, 8: 0.769, 1.208, 1.062, 0.553; on wave units 0, 2, 3, 10, 11, 12, 13, 14, 15: 0.110, 0.046, 0.036, 0.139, 0.109, 0.119, 0.044, 0.217, 0.116; 96.6 % of |L_clock|² on comparators. 13-dim norm of the mode parts of a unit perturbation on unit 10 / 11 / 12: clock 0.39 / 0.30 / 0.34, level 0.06 / 0.22 / 0.47, h2 0.53 / 1.16 / 1.19, short 0.89 / 1.00 / 1.04. A wave unit clipped by its whole clock swing would change the clock by 14-33 % (units 0, 10, 11, 12, 14, 15); actual grazing depths are 1-6 % of the swing.
- how checked: d6_radius.py part A and an inline decomposition of unit vectors. Consistent with D-20 and D-21.
- caveats: L_level has 85 % of its squared norm on unit 2, which never clips; deep clipping (after the centre is removed) does move both level and clock.

### D-20 [MIXED, confidence high]

For small S/M the clock amplitude ρ(a) equals the free decay 0.951^a to within 2 %, on N(0,1) data and in the impulse.

- numbers: N(0,1), S⁺/M ≤ 0.1, median |z|/(10.157·M) for a1 = 1..9: 0.995, 1.009, 0.965, 0.924, 0.885, 0.838, 0.793, 0.746, 0.702; free decay 1.02·0.9509^(a−2) for a = 2..9: 1.020, 0.970, 0.922, 0.877, 0.834, 0.793, 0.754, 0.717. Per (a1,a2) cell at S/M ≤ 0.1: 1.00-1.01, 0.96-0.97, 0.92-0.93, 0.88-0.89, 0.83-0.84, 0.78-0.80, 0.72-0.76 for a1 = 2..8, phase within ±6° (one cell −12°). Impulse: ratio to free decay from a = 2: 0.991, 0.989, 0.989, 0.982, 0.982, 0.982, 0.982 (1.8 % lost over seven steps, at the steps after units 15 and 11 are clipped); with backgrounds −0.25, −0.5, −1 within 0.989-1.027.
- how checked: d6_radius.py parts B, C (seeds 1031-1034), d6b_radius_cells.py (seeds 1061-1064).
- caveats: The starting amplitude at a = 2 is 0.96-1.01 depending on the input at age 1 (comparators 6, 8 clip when that input is ≥ 0).

### D-21 [INTERVENTION, confidence medium]

Removing the ReLUs of the nine wave units from max-age 3 on changes the clock amplitude by about 1 % when S/M is small; the effect is larger at large S/M, where the clock is small.

- numbers: Median |z_linear|/|z_real| at t = 9, states with a1 = 2 at t0 = 2 and max kept, by final S⁺/M: ≤0.1: 1.009 (n = 778, phase −1.0°); 0.1-0.3: 1.017 (2,260); 0.3-0.6: 1.039 (11,194); 0.6-0.9: 1.086 (18,974); 0.9-1.0: 1.171 (6,858). From t0 = 4: 1.011, 1.022, 1.042, 1.078, 1.152. Real run at S/M ≤ 0.1: 0.16, 1.01, 0.65, 0.92, 0.03, 0.01, 0.02 wave units clipped per step. Independent measure: level ×1.25-1.5 with autonomous flow raises |z| by 3-4 % over six steps at t0 = 3 and 2-3 % at t0 = 5.
- how checked: d6_radius.py part D (seed 1035, 400k); d3_intervene.py.
- caveats: The rows with S/M > 0.3 include the deep clipping that follows a new runner-up, so they are an upper bound for the effect of grazing cuts. Only 778 sequences in the lowest bin.

### D-22 [MEASURED, confidence high]

The exactly resting gate pattern (13 ordinary units on, three pulses off) is the minority case on N(0,1) data, although the deviation from the resting map is small.

- numbers: Share of states with the resting pattern: all a1 ≥ 2: 0.188; by a1 = 2..9: 0.012, 0.287, 0.261, 0.042, 0.026, 0.539, 0.638, 0.609. All 13 ordinary units on: 0.248 for a1 ≥ 2. Unit 5 on in 0.21-0.29 of states with a1 ≥ 2. For a1 ≥ 3 the one-step linear prediction equals the state exactly in 0.339 of states; rms ReLU correction / rms state = 0.066.
- how checked: d3b_impulse.py part 3, seed 1081, 200k sequences.
- caveats: Relevant to F3: the eigenmodes of A are a linearisation around a pattern that is rarely exact.

### D-23 [MIXED, confidence medium]

The runner mode is the S-dependent part of the centre: its coordinate is about 10.157·(r0·M + 5.2·S), and its eigenvector has mixed sign, so it lowers the centre of units 0, 10, 11, 12, 14, 15 and raises that of unit 2 and the comparators as S grows.

- numbers: R_runner on units 0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15: −0.165, 0.054, 0.091, 0.035, 0.044, 0.051, 0.043, −0.113, −0.128, −0.149, 0.034, −0.116, −0.032 (exact). Runner/10.157M at S/M = 0 / 0.5 / 0.9 for a = 3: 2.09 / 4.56 / 6.70; a = 9: 1.55 / 4.18 / 6.04. Regression per a1: u_runner 1.6-2.0, v_runner 4.7-7.1; per cell (a1,a2 ≥ 2) v_runner 2.0-8.3, median 7.1, with the T coefficient taking over (4.8-6.7) in cells where the runner-up arrived 0-2 steps ago at late a1. Two-pulse: (runner − r(a)·M)/S = 3.7-6.3. Measured slope of the centre per unit S: unit 0 −10, unit 10 −6.5, unit 11 −6.8, unit 12 −8.3, unit 14 −5.5, unit 2 +3.6.
- how checked: d2_orbit.py, d1_vectors.py, d1_cells.py, d4b_twopulse.py. Dropping the runner mode from the description lowers R² over a1 = 2..9 from 0.894 to 0.830 (wave units 0.832 to 0.741).
- caveats: r1 is 4.3-4.4 at a = 2 and 5.0-5.6 for a ≥ 3; the mode updates to a new runner-up over about 2-3 steps at late a1, so 'runner ∝ S' is only true once the runner-up has aged.

## Contradictions with the lead

No outright contradiction with F1-F9; three refinements and one caveat.

1. F5: the coefficient is confirmed (level = 0.918-0.938 × 10.157·M for a1 = 2..9), but "R² ≥ 0.997 within age cells" holds only for the uncentred R² (0.998-0.9994). The centred R² within an age cell is 0.9963 at a1 = 2 and falls to 0.986 at a1 = 9.

2. F6: phases and ρ(a) are confirmed at small S/M (ρ = 1.01 → 0.70, the free decay; phases 0, 36, 79, 123, 166, 209, 251, 294, 337°). But the S-proportional remainder is 0.07-0.39 at S/M > 0.9 depending on the (a1,a2) cell, a bit wider than "0.1-0.35", and it is not a constant per unit of S (0.27-0.87 at S = 0.1·M in exact two-pulse runs). Read as a formula, z ≈ 10.157·(M − S)·ρ(a)·e^(iφ(a)) underestimates the radius by a factor 1.2-1.5 at S = M/2 and 1.8-3.4 at S = 0.9·M. A single-term fit in (M − S) has coefficient 0.99-1.09 at every age (not ρ(a)) with uncentred R² 0.97 → 0.85.

3. F3 caveat: the "resting" gate pattern is exact in only 18.8 % of states with a1 ≥ 2 (1.2 % at a1 = 2, 4 % at a1 = 5, 3 % at a1 = 6, 54-64 % at a1 = 7-9). The modes of A remain a good linearisation (rms ReLU correction 6.6 % of the state norm for a1 ≥ 3).

4. F4: cos(w_ORD, R_echo) = −0.998 in the modes.py gauge; the sign is a gauge convention, not a disagreement.

Relative to the task's own guess in D6 ("does clipping of wave units cap the clock amplitude when M − S is large relative to M"): the answer is no; the relative effect of the wave ReLUs is largest when S/M is large (8-17 %), not when it is small (1 %).

## Surprises

1. The orbit is tangent to the ReLU boundary, not inside it. Units 10 and 11 are OFF at max-age 6 and 5 in 89-98 % of states with S/M ≥ 0.5 (67-89 % at S/M ≈ 0), with a pre-activation of about −0.5·M that does not depend on S, T or x_t. At S = 0 the clock radius equals the centre within ±22 % for 9 of 13 units.

2. The centre has two components. Besides the level (∝ M), the runner mode (λ = 0.9735, coordinate ≈ 10.157·(r0·M + 5.2·S)) has an eigenvector of mixed sign and lowers the centre of the six large wave units by 6-10 per unit of S. This is what keeps units 10 and 11 tangent at every S/M.

3. The S part of the state rotates too. At S = M the mean orbit still has a 13-dim radius of about 10 per unit M (23.5 at S = 0).

4. Erasing the clock costs only 5-6 accuracy points when no later input can change the top two, but enlarging it by 50 % costs 15-22 points. With real inputs, erasing it costs 3 points if the max is kept and about 44 points if the max is later replaced: the clock matters at the moment its owner becomes the runner-up.

5. After the level is removed the ReLU regenerates 25-30 % of it within one step, and the clock keeps rotating with a constant phase offset of about +30°.

6. Raising the level by 25 % (nothing clipped) still costs 3-10 points in the quiet continuation and 12-17 points with real inputs, because the level is 80 % of unit 4's threshold.

7. The harmonics h2 and h3 respond almost identically to "runner-up arrives after the max" and "max arrives after the old max" (F(d) ≈ F(−d)), and their size depends on the distance between the two positions: for h2 it is 0.1 at |d| = 2 and 1.7 at |d| = 6. At t = 9 the inputs (1,0,0,0,2,…) and (2,0,0,0,1,…) have nearly the same h2 coordinate (0.66@166° and 0.82@175°) although the answers differ.

8. The wave units' coefficient on the third largest value T is large (up to −12 per unit on units 10 and 11) with the same age profile as u, so in wave units the oscillation scales roughly as M − 0.3·S − 0.45·T rather than as M − S.

9. The step from max-age 1 to 2 removes the harmonics that the unit-4 kick put in: all modes have coordinate ≈ 1 at age 1; at age 2 h2, h3, alt, short are 0.03-0.21 while level, clock stay near 0.93-0.96 and runner rises to 2.4.

## Not done

- findings.md was not written: the Write tool rejected it ("Subagents should return findings as text, not write report files") and I did not circumvent the block. The content is in this object; the tables are in the .txt outputs.
- Everything is on N(0,1); nothing was repeated on Unif(−1,1).
- Why the orbit is kept tangent was not determined (function for the readout, or side effect of making the circle as large as the centre allows).
- F_m(d) of the harmonic packet is measured, not derived from the weights. I did not trace how the readout uses it, nor how the two orders of the top two are told apart at the readout.
- The D2 orbits are means per S/M bin; the spread around them per (a1,a2) cell is not tabulated. The 13-dim geometry of v was analysed pooled over a2 and per mode per cell, not as a 13-dim object per cell.
- The dependence on the third largest value T is reported but not analysed.
- In D3 the centre role and the threshold role of the level are separated by controls (quiet continuation, f > 1, clock scaling), not by an intervention that changes one without the other.
- The owner's off-plane rms 0.91 for the impulse is not reproduced exactly (0.88 with 1/n, 0.94 with 1/(n−1)).
- Two-pulse PCA planes use only four points (t = 6..9) as the task specified; I did not repeat them with an earlier second pulse to get a longer orbit (mode coordinates were used instead).
- The wave-ReLU removal has only 778 sequences in the lowest S/M bin.

## Open questions

- Is the tangency functional? Units 10, 11, 12, 15 touch zero at max-age 6, 5, 4, 2-4 for every S/M. Since the readout is blind to level and clock only while all 13 units are on (F7), do these zero-touches write age-specific terms into the logits?
- What does the runner mode's sign pattern do for the readout? It moves the centre of units 0, 10, 11, 12, 14, 15 down and of unit 2 and the comparators up by 4-10 per unit of S.
- How does the readout tell 'runner-up arrived after the max' from 'the runner-up is the old max', given that the long-lived harmonics h2 and h3 are nearly the same in both cases (F(d) ≈ F(−d))? Candidates: the clock phase (which tracks a1), the runner coordinate, the pulses.
- Can F_m(d) be derived from the weights as the conversion of the old clock, at phase φ(d), into harmonics when the comparators are clipped at the second event?
- Why does the amplitude of h2 nearly vanish at |d| = 2 (0.08-0.11) and peak at |d| = 6 (1.7)?
- What is the role of the third largest value in the wave units (T coefficient up to −12 on units 10 and 11 with the age profile of u)?
- Is the residual clock radius at S ≈ M (0.07-0.39 of 10.157·M, with a phase offset of up to −85°) a source of the network's errors when S is close to M? The phase decoder falls to 38 % there.
- Do the same numbers hold on Unif(−1,1), the training distribution?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d0_verify.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d0_verify.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_gram.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_gram.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_lib.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_r2.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_r2_strict.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_r2_pos.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_vectors.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_vectors_strict_gSTx.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_vectors_strict_gS.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_vectors_pos_gSTx.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_geometry.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_geometry_strict_gSTx.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_geometry_strict_gS.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_geometry_pos_gSTx.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_geometry_pos_gS.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_cells.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_cells_strict.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_cells_strict.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_phase.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d1_phase.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d2_clip.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d2_clip.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d2_clip.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d2_orbit.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d2_orbit.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3_intervene.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3_intervene.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3_summary.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3_summary.md.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3b_impulse.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3b_impulse.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3c_thresholds.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d3c_thresholds.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d4_impulse.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d4_impulse.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d4b_twopulse.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d4b_twopulse.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d5_runner.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d5_runner.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d6_radius.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d6_radius.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d6b_radius_cells.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_D_geometry/d6b_radius_cells.txt