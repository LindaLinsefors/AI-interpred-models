# Round-1 arm B_max_circuit: structured result

## Summary

Arm B is complete for B1–B6, but findings.md was NOT written: the harness refused the write (subagents may not write report .md files) and I did not work around it. The full findings are in this object. All scripts and their raw printed outputs (*_output.txt) are in my folder. Everything rests on my own computations from the weights (N(0,1) samples of 200k–800k sequences, seeds 1000–1101).

Picture of the max circuit:
- Unit 4 computes h4 = 10.157·(x_t − θ_4)⁺ exactly. θ_4 is not "the level": θ_4 = (0.7956·level + 0.0353·runner + clock term + small)/10.157. The slow part is only 0.82·M + 0.16·S⁺; a clock term of amplitude 0.16·ρ(a)·(roughly M − S⁺) fills the rest at ages 1–2 and removes as much again at ages 5–6. So θ_4 is within 1–3% of M only for ages 0–2 and falls to 0.70·M at ages 5–6 for an isolated max.
- Detection: unit 4 fires on 99.94% of positive-part records and on 1.1% of other inputs. The false firings are new runner-ups in the upper part of the gap (28–29% of new runner-ups at ages 5–6). Records with x_t ≤ 0 never fire it.
- Level bookkeeping (exact): level(t) = 1.0099·level(t−1) + h4 − 0.0724·h5 − 0.1263·h9 − 0.0447·x_t + clip terms. The kick gives 1.006 from rest; units 5 and 9 then take off 0.087 + 0.019, which gives 0.93. For an isolated max nothing compensates the eigenvalue (0.929 at age 2 to 1.014 at age 9). On N(0,1) data the average stays near 0.92–0.94 only because unit 5 fires after each new runner-up (−0.76% per step on average).
- Negative inputs are not stored as values: accuracy is 11.9% with no positive input, 23.1% with one, 97.9% with two; k ≤ 1 gives 18% of all errors.
- Unit 2 holds 62–81% of the level coordinate (h_2 ≈ 1.73·10.157·M at ages 1–4) but is not a standalone integrator: cutting its other inputs lowers accuracy from 95.3% to 83.0%.
- Causal: scaling the level shifts all five input thresholds by 0.63–0.80 of the change and changes 12–31% of answers at steps 3–7, mostly without any change in unit 4's firing. Replacing unit 4 by the oracle gives 96.3% agreement and LOWERS accuracy to 93.1%: the oracle's gate helps (+0.34 pp), its size hurts, because the network's kick is measured from its own sagged threshold and resets it (θ_4 ← θ_4 + 0.986·(x − θ_4)).

## Claims

### V0 [MIXED, confidence high]

Re-verification of the facts I build on: F1 (h_t = Σ_p J[t,p]·x_p), F2 (gate = threshold rule for unit 4 and the four comparators), F3 (eigenvalues of A = W[ORD,ORD] and the gauge LEFT·W[ORD,4] = 1) and F4 (w[ORD] is the echo direction) hold on re-computation. Network accuracy is 95.30% on N(0,1) and 93.90% on Unif(−1,1).

- numbers: F1: max |h_t − Σ_p J[t,p]·x_p| = 2.7e−13 on 2000 sequences. F2: gate equals the threshold rule in 100.0000% of 4M states for units 4, 1, 6, 7, 8 (seed 1090). F3: level 1.00989, runner 0.97349, clock 0.95088 at 42.748°, h2 0.93701 at 83.281°, alt −0.92830, h3 0.83734 at 124.781°, short 0.73499 at 42.995°, trans 0.23098, echo 0.00099. F4: |cos| = 0.998 (−0.998 in this gauge), max |A·w[ORD]| = 0.432, W[PULSE,ORD]·w[ORD] = (0.01, 18.01, 84.34). Accuracy 0.95302 (seed 1000, n = 400k), 0.93903 (seed 1001, n = 400k). Eigenvector matrix condition number 17.9; cos(RIGHT_level, RIGHT_runner) = −0.57.
- how checked: b0_basics.py and b9_more_checks.py part 1: direct recomputation and asserts.
- caveats: F8 and F9 were not re-examined.

### B1.1 [EXACT, confidence high]

θ_4(t) = −W[4]·h_(t−1)/w_4 in mode coordinates of h_(t−1) is θ_4 = (1/10.157)·[0.7956·level + 0.0353·runner + 2Re(c_clock·z_clock) + 2Re(c_short·z_short) + 2Re(c_h2·z_h2) + 2Re(c_h3·z_h3) + 0.0034·alt + 0.0031·trans + 0.0002·echo + 0.9856·h4 − 0.0017·h5 − 0.0834·h9]. A record kick h4 therefore raises the threshold by 0.9856·h4/10.157 at the next step (through W[4,4] = −0.986) and by 0.9916·h4/10.157 one step later through the ordinary units, of which level 0.796, clock 0.159, runner 0.035, h3 0.020, h2 0.008, short −0.033: 16% of the fresh threshold sits in the clock mode, which then rotates away.

- numbers: 10.157·2|c| and arg c: clock 0.1603 at −6.57°; short 0.0784 at −115.0°; h2 0.0243 at −70.9°; h3 0.0206 at +10.1°. Sum of the mode coefficients (real + 2Re for pairs) = −W[4,ORD]·W[ORD,4] = 0.99165.
- how checked: The mode and pulse terms sum to θ_4 within 1e−9 on 7.2M states (assert in b1_theta4.py); the coefficient sum equals −W[4,ORD]·W[ORD,4].

### B1.2 [EXACT, confidence high]

After the impulse x = (1,0,…,0) the threshold for the next input is 0.988, 0.979, 0.963, 0.881, 0.780, 0.717, 0.726, 0.795, 0.885 at ages 0..8 of the max. The level + runner part is 0.82–0.85 at every age ≥ 2 and never reaches 1; the clock term is +0.154, +0.134, +0.043, −0.062, −0.123, −0.115, −0.049, +0.035 at ages 1..8 (period 360/42.75 = 8.4 steps). The sag of the threshold with age is the clock term rotating from +0.15 to −0.12.

- numbers: age: θ_4 | level | runner | clock | short. 1: 0.979 | 0.801 | 0.038 | 0.154 | −0.033. 2: 0.963 | 0.738 | 0.084 | 0.134 | 0.004. 3: 0.881 | 0.741 | 0.086 | 0.043 | 0.011. 4: 0.780 | 0.749 | 0.083 | −0.062 | 0.010. 5: 0.717 | 0.757 | 0.081 | −0.123 | 0.003. 6: 0.726 | 0.765 | 0.079 | −0.115 | −0.003. 7: 0.795 | 0.772 | 0.077 | −0.049 | −0.005. 8: 0.885 | 0.780 | 0.075 | 0.035 | −0.003. Age 0: 0.988, of which 0.986 is unit 4's self-weight term. With background −1 instead of 0: 0.963, 0.878, 0.772, 0.703, 0.710, 0.782, 0.878 at ages 2..8.
- how checked: Free-flow check that could have failed (b7 part H): taking the mode coordinates at age 2 and multiplying each by λ^(a−2) gives 0.9633, 0.8815, 0.7798, 0.7159, 0.7250, 0.7933, 0.8833 against the real 0.9631, 0.8807, 0.7801, 0.7175, 0.7262, 0.7947, 0.8854 (within 0.002), so the wave units clipped along the way (0, 14, 15, 11, 10) do not matter for θ_4.

### B1.3 [MEASURED, confidence high]

On N(0,1) data θ_4 lies below the standing max M_(t−1) at every age of the max (states with M > 0): mean θ_4 − M = −0.013, −0.025, −0.036, −0.095, −0.170, −0.229, −0.232, −0.179, −0.105 at ages 0..8, with sd 0.010, 0.016, 0.024, 0.065, 0.120, 0.159, 0.161, 0.129, 0.084. The median of θ_4/M is 0.988, 0.978, 0.968, 0.925, 0.875, 0.838, 0.840, 0.878, 0.928. Over all ages the mean is −0.075, the sd 0.106 and the median −0.031.

- numbers: n = 800k sequences × 9 steps, seeds 1010, 1011; 6,402,318 states with M > 0. n per age 0..8: 1,708,118; 1,308,352; 1,008,771; 775,798; 587,798; 433,153; 302,452; 188,958; 88,918. 5% quantiles: −0.030, −0.052, −0.081, −0.222, −0.404, −0.538, −0.542, −0.424, −0.261. 95% quantiles: −0.001, −0.003, −0.006, −0.017, −0.028, −0.037, −0.034, −0.015, +0.006. By step t = 1..9 the mean is −0.010, −0.016, −0.024, −0.044, −0.073, −0.100, −0.117, −0.121, −0.113.
- how checked: b1_theta4.py. The isolated-max column of the data (S ≤ 0: θ_4/M = 0.988, 0.979, 0.963, 0.878, 0.773, 0.704, 0.712, 0.784, 0.879) matches the exact impulse response of claim B1.2 with negative background.
- caveats: Absolute numbers are for N(0,1) inputs; the net is homogeneous, so the relative columns are the transferable ones.

### B1.4 [FIT, confidence high]

Within an age cell the sag is linear in the gap: M − θ_4 = ε(a)·M + κ(a)·(M − S⁺), with ε = 0.010, 0.020, 0.018, 0.027, 0.039, 0.058, 0.062, 0.046, 0.025 and κ = 0.003, 0.002, 0.023, 0.100, 0.196, 0.256, 0.253, 0.201, 0.124 at ages 0..8. So the threshold is short of the max by up to a quarter of the gap (ages 5–6) plus 1–6% of M. When S⁺ is within 10% of M the median θ_4/M is 0.987, 0.976, 0.984, 0.971, 0.960, 0.950, 0.954, 0.974, 0.995; for an isolated max it is 0.988, 0.979, 0.963, 0.878, 0.773, 0.704, 0.712, 0.784, 0.879.

- numbers: Fit: θ_4 = b_M·M + b_S·S⁺ through the origin per age, ε = 1 − b_M − b_S, κ = b_S. R² = 0.9999, 0.9998, 0.9998, 0.9995, 0.9977, 0.9917, 0.9873, 0.9891, 0.9926 at ages 0..8. Mean ± sd of (θ_4 − M)/M at age 5 by relative gap g = (M − S⁺)/M: −0.054 ± 0.026 (g < 0.1), −0.106 ± 0.040, −0.159 ± 0.041, −0.219 ± 0.032, −0.270 ± 0.020 (0.75–1), −0.297 ± 0.005 (S ≤ 0). At age 3: −0.030 ± 0.014, −0.044 ± 0.013, −0.064 ± 0.012, −0.090 ± 0.010, −0.112 ± 0.006, −0.122 ± 0.003. At age 1: −0.020 to −0.022 in every gap bin.
- how checked: Two independent samples (seeds 1010, 1011 and 1100, 1101; 800k sequences each) give ε and κ equal to 0.001. Splitting by whether the runner-up is older or younger than the max changes b_M by at most 0.04 and b_S by at most 0.03.
- caveats: S⁺ is the positive part of the second largest so far; a negative runner-up behaves as S⁺ = 0 (claim B4.5).

### B1.5 [MIXED, confidence high]

Two things pull θ_4 below M. (a) The slow part is short by construction: the level and runner terms together equal 0.816·M + 0.163·S⁺ (ages ≥ 2), that is M − 0.02·M − 0.16·(M − S⁺); the level term alone is 0.73–0.80·M and the runner term about 0.06·M + 0.2·S⁺. (b) The clock term, 2Re(c_clock·z_clock)/10.157 with amplitude 0.160·ρ(a) times roughly the gap, fills that 0.16 at ages 1–2 and doubles it at ages 5–6. Everything else (short, h2, h3, alt, trans, echo, pulses) has mean −0.011 and sd 0.028.

- numbers: n = 800k sequences, seeds 1100, 1101. Level + runner terms on (M, S⁺), pooled ages ≥ 2: 0.816, 0.163, R² = 0.998; by age 1..8: c_M = 0.833, 0.810, 0.813, 0.818, 0.822, 0.827, 0.827, 0.826 and c_S = 0.173, 0.175, 0.172, 0.163, 0.151, 0.138, 0.135, 0.137. Clock term on (M, S⁺) at ages 1..8: (+0.160, −0.135), (+0.141, −0.109), (+0.047, −0.022), (−0.068, +0.071), (−0.137, +0.114), (−0.131, +0.100), (−0.059, +0.039), (+0.040, −0.039). Short term at most 0.055 in either coefficient. Level term alone (slope on M) at ages 1..8: 0.790, 0.729, 0.729, 0.727, 0.731, 0.736, 0.739, 0.743; for an isolated max it drifts from 0.739 (age 2) to 0.798 (age 8) while the runner term decays from 0.074 to 0.051.
- how checked: Exact term split (assert) followed by regression of each term. Within age cells 4–7 the clock term correlates 0.91–0.98 with θ_4 − M. The clock coefficients agree with F6: 0.160·ρ(a)·cos(φ(a) − 6.6°) with the lead's φ(a) gives +0.159, +0.138ρ, +0.046ρ, −0.071ρ, −0.150ρ, −0.148ρ, −0.067ρ, +0.048ρ.
- caveats: The clock term is only roughly proportional to M − S⁺: its S⁺ coefficient is 0.76–0.84 of the M coefficient at ages 1, 2, 5, 6 where the term is largest (0.47–1.04 over all ages).

### B1.6 [MIXED, confidence high]

When all earlier inputs are negative, θ_4 is a small positive number: exactly 0.0015·|x_0| after one input, mean 0.001, 0.005, 0.005, 0.004, 0.005, 0.007, 0.011, 0.015, 0.017 for t = 1..9, positive in every sampled state, at most 0.042. Unit 4 therefore compares the next input with about 0, not with the largest negative input.

- numbers: EXACT: −W[4]·ReLU(−w)/w_4 = 0.001503; the four comparator contributions to W[4]·ReLU(−w) are 0.376, 1.026, −0.387, −1.004 (sum 0.011). MEASURED: 797,253 states with all earlier inputs negative (seeds 1100, 1101): fraction with θ_4 > 0 = 1.000000, mean 0.0031, max 0.0416. At t = 9 the level term of θ_4 is +0.0345 and the runner term −0.0200 (seed 1041, 200k all-negative sequences).
- how checked: b1_theta4.py, b10_summary_numbers.py, b4_negative.py part c. Consequence checked in B2.1: none of 554,421 true records with x_t ≤ 0 fires unit 4.

### B2.1 [MEASURED, confidence high]

Unit 4 is on for 99.936% of positive-part records (x_t > max(0, x_0..x_(t−1)); n = 1,788,439) and for 1.112% of all other inputs (n = 6,211,561); P(record | on) = 0.963. It never fires on a true record with x_t ≤ 0 (0 of 554,421). Of the 1137 missed records, 988 are first positive inputs smaller than the tiny threshold left by a negative prefix and 149 have θ_4 > M (largest missed margin 0.086).

- numbers: n = 800k sequences × 10 steps, seeds 1020, 1021. By step t = 0..9: P(on | not record) = 0, 0.0015, 0.0029, 0.0046, 0.0087, 0.0133, 0.0176, 0.0190, 0.0181, 0.0153; P(record | on) = 1.000, 0.9975, 0.9929, 0.9851, 0.9651, 0.9365, 0.9038, 0.8822, 0.8734, 0.8782; P(on | record) between 0.9983 and 1.0000. 69,082 false firings. Sequences with at least one false firing: 8.4%; their error rate is 10.5% against 4.2% for the rest (overall 4.70%).
- how checked: b2_detection.py. Check that could have failed: h4 = 10.157·ReLU(x_t − θ_4(t)) within 1e−9 on all 8M states (assert).
- caveats: The error-rate comparison is correlational; the causal number is in B6.4 (+0.34 pp when false firings are suppressed).

### B2.2 [MEASURED, confidence high]

False firings of unit 4 are new runner-ups in the upper part of the gap between S⁺ and M. By age 0..8 of the standing max, P(on | not a record) = 0.0026, 0.0045, 0.0062, 0.0163, 0.0286, 0.0382, 0.0361, 0.0256, 0.0141; P(on | S⁺ < x_t ≤ M) = 0.011, 0.022, 0.035, 0.100, 0.194, 0.278, 0.290, 0.218, 0.128; P(on | x_t ≤ S⁺) ≤ 0.0006. The firing point inside the gap, v* = (θ_4 − S⁺)/(M − S⁺), has median 0.985, 0.966, 0.943, 0.845, 0.714, 0.598, 0.578, 0.676, 0.812: at ages 5–6 a new runner-up in the upper 40% of the gap fires unit 4.

- numbers: Same sample (seeds 1020, 1021), states with a positive standing max. P(on) for new runner-ups by v = (x_t − S⁺)/(M − S⁺) in bins <0.5, 0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–0.95, 0.95–1: age 3: 0.002, 0.005, 0.015, 0.039, 0.437, 0.998, 1.000; age 5: 0.029, 0.143, 0.455, 0.987, 1.000, 1.000, 1.000; age 6: 0.040, 0.203, 0.528, 0.974, 1, 1, 1; age 8: 0.002, 0.032, 0.077, 0.205, 0.592, 0.972, 1.000. IQR of v* at age 5: [0.437, 0.664]. By relative gap of the standing max at age 5, P(on | not record) = 0.012 (g < 0.1), 0.025, 0.037, 0.049, 0.064, 0.073 (S ≤ 0). False firings are small: h4/(10.157·M) has median 0.044, 90% quantile 0.153, 99% 0.247, max 0.354 (true records over a positive max: median 0.78).
- how checked: b2_detection.py. Consistent with the fit of B1.4: the firing fraction of the gap is κ(a) + ε(a)·M/(M − S⁺), e.g. 1 − 0.598 = 0.40 at age 5 against κ + ε = 0.31 for an isolated max and more for smaller gaps.

### B2.3 [MEASURED, confidence high]

By relative margin u = (x_t − M)/M over a positive standing max: records with u ≥ 0.05 are detected in 100.0% of cases at every age; records with 0 ≤ u < 0.01 in 0.990, 0.992, 0.991, 0.999, 1.000, 1.000, 0.998, 0.983, 0.950 at ages 0..8; inputs 10–20% below the max fire unit 4 with probability 0.000, 0.000, 0.000, 0.037, 0.353, 0.522, 0.511, 0.333, 0.045; inputs 20–30% below with 0.026, 0.154, 0.154, 0.018 at ages 4..7 and 0.000 otherwise; inputs more than 30% below never (0.000).

- numbers: Seeds 1020, 1021. Further bins, ages 0..8: u in [−0.1, −0.05): 0, 0, 0, 0.476, 0.755, 0.843, 0.815, 0.706, 0.445; [−0.05, −0.02): 0, 0.087, 0.345, 0.859, 0.940, 0.962, 0.942, 0.881, 0.731; [−0.02, −0.01): 0.297, 0.920, 0.890, 0.987, 0.990, 0.999, 0.984, 0.946, 0.854; [−0.01, 0): 0.937, 0.974, 0.985, 0.996, 1.000, 1.000, 1.000, 0.976, 0.936; [0.01, 0.02): 0.997, 0.995, 0.997, 1.000, 1.000, 1.000, 1.000, 0.997, 0.979. Cell sizes 141 to 1.2M.
- how checked: b2_detection.py; cells with fewer than 100 states were not reported.
- caveats: Cells at age 8 near u = 0 have only 141–164 states.

### B2.4 [FIT, confidence high]

On a record h4 exceeds the oracle increment 10.157·(x_t − M) by exactly 10.157·(M − θ_4). Regressing h4 on 10.157·(x_t − M⁺) through the origin: slope 1.014, R² 0.9912 on all positive-part records; slope 0.999, R² 0.99998 on first positive inputs; slope 1.033, R² 0.983 on records over a positive max, where by age 0..8 of the old max the slope is 1.006, 1.013, 1.019, 1.054, 1.101, 1.136, 1.141, 1.111, 1.064 with R² 0.9998, 0.9993, 0.9985, 0.988, 0.958, 0.926, 0.920, 0.949, 0.978 and mean excess M − θ_4 = 0.008, 0.015, 0.022, 0.059, 0.107, 0.145, 0.147, 0.114, 0.066. On all 8M states against 10.157·ReLU(x_t − M⁺): slope 1.014, R² 0.9953.

- numbers: Seeds 1020, 1021. n: 1,788,439 records; 799,199 first positive inputs; 989,240 records over a positive max (by age: 319,471; 220,056; 153,727; 107,567; 75,157; 51,105; 33,645; 19,725; 8,787). Affine fits: all records slope 0.988, intercept 0.294, R² 0.9919; over a positive max slope 0.988, intercept 0.469, R² 0.9849; by age the affine slope is 0.998, 0.996, 0.995, 0.987, 0.979, 0.971, 0.971, 0.980, 0.987 and the intercept 0.089, 0.173, 0.254, 0.680, 1.212, 1.640, 1.660, 1.269, 0.734.
- how checked: The identity h4 = 10.157·ReLU(x_t − θ_4) is asserted on all states, so the excess is exact; the regression only summarises it.
- caveats: R² is centred, 1 − Var(resid)/Var(y), for fits through the origin.

### B3.1 [EXACT, confidence high]

The level coordinate obeys level(t) = 1.00989·level(t−1) + 1.0000·h4(t−1) − 0.0724·h5(t−1) − 0.1263·h9(t−1) − 0.0447·x_t + Σ_i L_i·clip_(t,i), with clip = max(−z, 0) on the ordinary units. The input is almost invisible to it: L·w[ORD] = −0.0447, of which the four comparators give −0.0037 (−0.294 − 0.706 + 0.557 + 0.439) and unit 14 gives −0.090, unit 12 +0.032. When a record clips all four comparators by the same margin, their clip terms cancel to +0.0037 per unit margin. So a record enters the level one step late, through unit 4, with coefficient exactly 1; units 5 and 9, which fire at s+1, take 0.0724·h5 + 0.1263·h9 off at s+2.

- numbers: Clip coefficients L_i (= LEFT_level): unit 2: 0.4066; 12: 0.1060; 14: 0.0679; 6: 0.0642; 11: 0.0509; 0: 0.0335; 1: 0.0279; 13: 0.0221; 10: 0.0132; 15: 0.0095; 3: −0.0220; 8: −0.0357; 7: −0.0423. LEFT_level·RIGHT_echo = 1e−18 (biorthogonality; w[ORD] is 99.8% echo, which is why L·w is small). Sequence across a record at s: level(s) = λ·level(s−1) − 0.0447·x_s + L·clip_s; level(s+1) = λ·level(s) + h4(s) − 0.0447·x_(s+1) + L·clip_(s+1); level(s+2) = λ·level(s+1) − 0.0724·h5(s+1) − 0.1263·h9(s+1) − 0.0447·x_(s+2) + L·clip_(s+2); later steps: λ·level − 0.0724·h5 − 0.0447·x + clip, plus h4 and −0.1263·h9 after a false firing.
- how checked: The update identity holds within 1e−8 on all 8M states of seeds 1030, 1031 (assert in b3_level.py).

### B3.2 [EXACT, confidence high]

For a record from rest (impulse x = (1,0,…,0)) the level in units of 10.157·M is 0.0047 at the record step, 1.0063 one step later and 0.9274 two steps later. 1.0063 = 1 (kick, by the gauge) + 1.0099 × 0.0047 (what the record step itself leaves: wave units 3, 13, 14, 15 clipped +0.0088, comparators +0.0004, input −0.0044) + 0.0015 (comparators 6, 8 clipped at s+1). 0.9274 = 1.0063 + 0.0099 (growth) − 0.0870 (0.0724 × h5, h5 = 1.2017) − 0.0186 (0.1263 × h9, h9 = 0.1470) + 0.0166 (wave units 0, 14 clipped). After that it only grows: 0.9312, 0.9417, 0.9511, 0.9611, 0.9706, 0.9802, 0.9899 at t = 3..9.

- numbers: h5(1)/10.157 = 1.2017 ≈ W[5,4] = 1.187 plus 0.015 from the ordinary units; h9(1)/10.157 = 0.1470 ≈ W[9,4] = 0.146. At t = 3 a weak second firing of unit 5 (h5(2) = 0.0798) takes off another 0.0058. With background −0.5 the level is 1.0070, 0.9262, …, 1.0102 at t = 1, 2, 9; with background −1 it is 1.0092, 0.9282, …, 1.0280, because each negative input adds +0.0044·|x|.
- how checked: Term-by-term table from the exact update (b3_level.py); terms sum to the level change by the asserted identity.

### B3.3 [FIT, confidence high]

On N(0,1) data the same accounting holds. One step after a record the median of level/(10.157·M) is 1.0067 and level/10.157 = 1.011·M_new − 0.046·M_old⁺: the kick h4/10.157 = 0.998·M_new − 0.946·M_old⁺ cancels the old level (λ²·level(s−1) ≈ 0.94·M_old), so the kick resets the level to the new max within ±0.08·M_old. Two steps after the record level/10.157 = 0.930·M_new − 0.035·M_old⁺ = 1.011 + 0.010 (growth) − 0.088 (unit 5: 0.0724 × 1.216) − 0.019 (unit 9: 0.1263 × 0.148) + 0.013 (wave clipping) + 0.002. Unit 5 fires after every record (P = 1.0000) and unit 9 after 99.87%.

- numbers: n = 800k sequences, seeds 1030, 1031; 855,594 records that still stand two steps later. Regressions through the origin on (M_new, M_old⁺, x_(s+1)): h5(s+1)/10.157 = 1.216, −0.427, −0.033 (R² 0.920); h9(s+1)/10.157 = 0.148, +0.004, −0.023 (R² 0.921); level(s+1)/10.157 = 1.011, −0.052, −0.001 (R² 0.995); level(s+2)/10.157 = 0.930, −0.035, 0.000 (R² 0.996). M_old coefficient of level(s+1) by age 0..7 of the old max: −0.075, −0.061, −0.079, −0.056, −0.001, +0.063, +0.083, +0.057 (R² 0.972–0.999); it turns positive when the threshold had sagged, because h4 is then larger. First positive input after a negative prefix: level(s+1)/10.157 = 1.014·M_new (R² 0.9998, n = 298,776). Age-1 IQR of level/(10.157·M): 0.979–1.017.
- how checked: b3_level.py; the M_new coefficients of the terms at s+2 add up to the fitted level(s+2) coefficient (1.0107 + 0.0100 − 0.0880 − 0.0187 + 0.0132 + 0.0024 = 0.9296).
- caveats: The lead's 1.006 is the from-rest value and the age-1 median; over an old max the level depends weakly on M_old (−0.08 to +0.08).

### B3.4 [MEASURED, confidence high]

Nothing compensates the eigenvalue 1.0099 in the level coordinate of an isolated max: with all other inputs ≤ 0, level/(10.157·M) is 0.929, 0.941, 0.954, 0.966, 0.979, 0.990, 1.003, 1.014 at ages 2..9 (+1.27% per step: λ plus +0.0044·|x_t| per negative input). On all N(0,1) data it is 0.918, 0.918, 0.914, 0.919, 0.924, 0.930, 0.934, 0.938, nearly flat, because unit 5 fires one step after each new runner-up (P = 0.9985, mean h5 = 0.48·10.157·M) and takes about 3–3.5% of 10.157·M off the level each time. Mean per-step terms at ages ≥ 3, in units of 10.157·M: growth +0.0091, unit 5 −0.0076 (of which −0.0065 after new runner-ups), false firings of unit 4 +0.0009, input +0.0004, comparator clipping −0.0007, wave clipping +0.0004, total +0.0023.

- numbers: n = 800k, seeds 1030, 1031 (isolated-max n by age 2..9: 199,144; 99,209; 49,017; 24,117; 11,791; 5,465; 2,401; 779). Class of the input two steps earlier at ages ≥ 3: new runner-up (20.2% of steps) unit-5 term −0.031; false firing (1.1%) −0.063 with unit-9 term −0.005; below the runner-up (78.7%) −0.0007. Seed 1035, n = 400k, one step after an input x_u over a positive standing max: below runner-up (67.5%): P(h5 > 0) = 0.066, level change from units 5, 9 = −0.001; new runner-up with unit 4 silent (15.2%): P(h5 > 0) = 0.9985, P(h9 > 0) = 0.003, −0.035; false firing (1.0%): 1.000, 0.526, −0.070; record (16.2%): 0.9999, 0.993, −0.092. In θ_4 the level's growth is partly offset by the runner term's decay: level + runner part 0.821 at age 2 and 0.854 at age 8 for the impulse.
- how checked: Exact term accounting (asserted identity) split by input class; the isolated-max drift 1.27% per step matches growth 0.99% plus the input term +0.0044·E|x|/M ≈ 0.3%, a check that could have failed.
- caveats: The balance on N(0,1) is statistical, not a correction: the ratio spreads with history (IQR 0.913–0.964 at age 9) and the level has an S⁺ coefficient of −0.037 to −0.068 at ages 3–9. Within-age R² of level on M: 0.9963, 0.9967, 0.9965, 0.9957, 0.9933, 0.9896, 0.9872, 0.9865 at ages 2..9.

### B4.1 [MEASURED, confidence high]

Accuracy by number of positive inputs k = 0..6 on N(0,1): 0.119, 0.231, 0.979, 0.969, 0.965, 0.961, 0.958. Sequences with k ≤ 1 are 1.07% of the data and contribute 0.84 pp of error, 18% of the 4.70% total. On Unif(−1,1) the pattern is the same: 0.119, 0.221, 0.985, 0.969 for k = 0..3.

- numbers: 200k sequences per k, sampled as |N(0,1)| with k random positions positive (seeds 1041–1047; uniform: seeds 1094–1097). P(k) = C(10,k)/1024: 0.00098, 0.00977, 0.04395, 0.11719, 0.20508, 0.24609, 0.20508. Error contributions P(k)·(1 − acc): 0.00086, 0.00751, 0.00093, 0.00367, 0.00722, 0.00952, 0.00869. With k = 0 the answer is position 0 in 51% and position 1 in 23%; P(answer = argmax) = 0.31.
- how checked: b4_negative.py part a, b9 part 5. Cross-check from an unconditioned sample (seeds 1061, 1062): accuracy 0.218 on the 8640 of 800k sequences with k ≤ 1, against (0.1185 + 10 × 0.2314)/11 = 0.221.

### B4.2 [MEASURED, confidence high]

With exactly one positive input (max at p, true answer q = largest negative) the network is right in 23.1% of cases (chance 11.1%); it answers position 0 in 42.7%, p in 20.2%, p−1 in 19.3%. Accuracy is continuous through S = 0 and depends on S/M: 0.071 (S/M < −1), 0.102, 0.129, 0.169, 0.258 (−0.1..−0.05), 0.404, 0.517, 0.602 (−0.01..0) with one positive input, then 0.682 (0..0.01), 0.753, 0.871 (0.02..0.05), 0.951 (0.05..0.1), 0.985, 0.998, 0.999, 0.998, 0.996, 0.894 (0.95..1) with two. The runner-up is located reliably only when S ≳ 0.05·M.

- numbers: One positive: n = 200k (seed 1042) and 400k (seed 1073); two positives: 400k (seed 1074). Bins for one positive: < −1, −1..−0.5, −0.5..−0.2, −0.2..−0.1, −0.1..−0.05, −0.05..−0.02, −0.02..−0.01, −0.01..0; for two: 0..0.01, 0.01..0.02, 0.02..0.05, 0.05..0.1, 0.1..0.2, 0.2..0.5, 0.5..0.8, 0.8..0.9, 0.9..0.95, 0.95..1. Exact split of the final logits into input terms (40k one-positive sequences), median sd across the 10 logits: term of the max 1.80, term of the last input 10.96, all other negative inputs together 2.12; the true runner-up's own term moves its logit by a median 1.13 per unit of x_q relative to the others; median winning margin 0.87.
- how checked: b4_negative.py part b, b7_followups.py part E. The two seeds for one positive input give the same S/M dependence (seed 1042: 0.568 for |S|/M < 0.02, 0.085 for > 0.5).
- caveats: Chance level 1/9 assumes the max position is excluded; the network answers the max position itself in 20%.

### B4.3 [MIXED, confidence high]

After an all-negative prefix the state is essentially the current input: the trace of an earlier input, ‖∂h_9/∂x_p‖, has median 0.48–0.73 against 23.65 for the current input (2–3%); unit 4 is never on; unit 2 is on in only 16–34% of states at t = 4..7. What accumulates is a sum, not a value: while all 13 ordinary units stay on, each input adds exactly −0.0447·x_t to the level and +1.0264·x_t to the runner coordinate (a tenth of a record kick), whose effects on θ_4, −0.0035·x_t and +0.0036·x_t, nearly cancel. At t = 9 the level is +0.047 ± 0.012 and the runner coordinate −0.62 ± 0.17 in units of 10.157.

- numbers: EXACT: LEFT_level·w[ORD] = −0.0447, LEFT_runner·w[ORD] = 1.0264; times the θ_4 coefficients 0.7956/10.157 and 0.0353/10.157. MEASURED: 200k all-negative sequences (seeds 1041, 1093), exact traces on 40k. Regression of the t = 9 coordinates (/10.157) on the ten inputs: level coefficients −0.004 to −0.007 (R² 0.93; free flow predicts −0.004 to −0.005), runner coefficients 0.028 for x_0 and 0.071–0.105 for x_1..x_9 (R² 0.85; free flow predicts 0.079–0.101). Gates: comparators on in 99–100% of states; units 5 and 9 on at t = 0 (negative input weights), afterwards unit 5 in 1–7% and unit 9 in ≤ 0.2%. P(unit 2 on) by t = 0..9: 0, 0, 0.93, 0.62, 0.19, 0.16, 0.27, 0.34, 0.80, 0.85. For comparison a unit record leaves runner ≈ +2.4 at age 2.
- how checked: b4_explore.py, b4_negative.py part c, b9 part 4: exact coefficients compared with regression on data.
- caveats: The free-flow coefficients are exact only while no ordinary unit is clipped; in all-negative prefixes several wave units are often off, which is why the measured coefficients differ by up to 50% for the level.

### B4.4 [INTERVENTION, confidence high]

The first positive input after a negative prefix is treated as a record over about 0: h4 = 10.157·(x_s − θ_4) with median h4/(10.157·x_s) from 0.9985 (s = 1) to 0.978 (s = 8). Resetting h_(s−1) to 0 before it changes few answers when at least two inputs are positive: agreement 0.998, 0.994, 0.991, 0.990, 0.982, 0.972, 0.953, 0.936 for s = 1..8, with accuracy real/reset 0.966/0.965 at s = 1 and 0.987/0.930 at s = 8. So the state left by a negative prefix is small but is used when the first positive input comes late.

- numbers: n = 100k per s, N(0,1) conditioned on x_0..x_(s−1) < 0 < x_s (seeds 1048, 1072). Accuracy real/reset with ≥ 2 positive inputs, s = 1..8: 0.966/0.965, 0.969/0.968, 0.970/0.968, 0.973/0.968, 0.970/0.959, 0.969/0.950, 0.977/0.943, 0.987/0.930. With exactly one positive input: 0.19/0.16, 0.25/0.19, 0.28/0.21, 0.24/0.14, 0.23/0.12, 0.23/0.12, 0.25/0.11, 0.26/0.11. Median relative state difference ‖h − h'‖/‖h‖ one step after the positive input: 0.011, 0.022, 0.030, 0.041, 0.057, 0.067, 0.073, 0.078 for s = 1..8. Mean θ_4(s): 0.001, 0.005, 0.005, 0.004, 0.005, 0.007, 0.011, 0.015.
- how checked: b4_negative.py part d and b7_followups.py part D (two seeds, same result). The reset run uses the same absolute positions, so the readout is aligned.
- caveats: With exactly one positive input the true answer often lies in the prefix; the real network is then right in 19–28% and the reset one in 11–21%.

### B4.5 [FIT, confidence high]

A negative runner-up is not stored in any of the five input thresholds. For states with M > 0 > S, regressing θ_4, θ_1, θ_6, θ_7, θ_8 on (M, S) gives S coefficients within ±0.05 at every age 1..8, against up to +0.94 when S > 0, while the M coefficients agree between the two cases within 0.07 (within 0.04 at ages 1–5). The thresholds of a max with a negative runner-up are those of an isolated max.

- numbers: n = 400k, seed 1049. S coefficients, S < 0 vs S > 0: age 1: θ_1 +0.004 vs +0.929, θ_6 +0.015 vs +0.243, θ_8 +0.012 vs +0.706; age 3: θ_4 +0.001 vs +0.103, θ_7 +0.006 vs +0.123, θ_8 +0.035 vs +0.917; age 5: θ_4 +0.006 vs +0.260, θ_1 −0.020 vs −0.307, θ_7 +0.017 vs +0.333; largest |S coefficient| for S < 0: 0.047 (θ_6 at age 8). M coefficients of θ_4 for S < 0 at ages 1..8: 0.980, 0.963, 0.879, 0.774, 0.706, 0.714, 0.785, 0.879.
- how checked: b4_negative.py part e. The comparison with the S > 0 case is the check: had negative runner-ups been stored, the two sets of S coefficients would match.
- caveats: Cells at ages 7 and 8 have 2335 and 752 states for S < 0.

### B5.1 [MEASURED, confidence high]

Unit 2 is the main carrier of the max. h_2/(10.157·M) has median 1.738, 1.730, 1.818, 1.787 at ages 1..4 (R² on M ≥ 0.996; the kick is W[2,4] = 1.731), then follows the clock: 1.665, 1.531, 1.450, 1.463, 1.563 at ages 5..9 (R² 0.977, 0.925, 0.872, 0.885, 0.938; with S⁺ as second regressor 0.953–0.994). A leaky integrator of unit 4 alone, g_t = 0.969·g_(t−1) + 1.731·h4(t−1), has squared correlation 1.0000, 0.9996, 0.995, 0.993, 0.989, 0.973, 0.953, 0.944, 0.942 with h_2 at t = 1..9.

- numbers: n = 800k, seeds 1050, 1051. IQR of h_2/(10.157·M) at ages 1..9: 1.732–1.782, 1.696–1.766, 1.804–1.845, 1.744–1.822, 1.574–1.748, 1.388–1.663, 1.280–1.608, 1.309–1.609, 1.453–1.679. Fit on (M, S⁺): age 5: 1.459, +0.342; age 7: 1.024, +0.658; age 9: 1.265, +0.452. Unit 2 is off in 83% of states with M ≤ 0 and in at most 0.05% of states with M > 0 and age ≥ 1 (8.7% at age 0). Pooled ages ≥ 1: h_2/10.157 = 1.704·M, R² 0.959. EXACT: W[2,4]·w_4 = 17.58 per unit of (x − θ_4); through unit 2 alone a record raises θ_4 by W[2,4]·(−W[4,2]) = 0.852 per unit.
- how checked: b5_unit2.py; the leaky-integrator comparison is a check that could have failed.

### B5.2 [MIXED, confidence high]

The level coordinate is mostly read from unit 2, but the level mode is not unit 2. LEFT_level has weight 0.4066 on unit 2 (85% of its squared norm, cosine 0.923 with e_2) and unit 2 supplies 62–81% of the level coordinate on data (74.5% pooled over ages ≥ 2). RIGHT_level is spread over all 13 ordinary units (unit 2 has 8.4% of its squared norm, cosine 0.290), and h_2 itself is only 64–83% level mode: per unit of 10.157·M it holds level 1.17–1.20, runner 0.43–0.47 and clock between +0.23 (age 3) and −0.19 (age 7). Unit 2's own term in θ_4 is 0.87, 0.86, 0.90, 0.88, 0.81, 0.74, 0.69, 0.70 of M at ages 1..8, of totals 0.98, 0.97, 0.92, 0.87, 0.83, 0.83, 0.87, 0.93.

- numbers: EXACT: LEFT_level[2] = 0.4066, RIGHT_level[2] = 1.2752, L·R = 0.5185; LEFT_runner[2] = 5.081 (43% of its squared norm), RIGHT_runner[2] = 0.091; 2|RIGHT_clock[2]| = 0.437. MEASURED (seeds 1050, 1051): L_2·h_2/level by age 1..9 = 0.72, 0.77, 0.81, 0.79, 0.73, 0.67, 0.62, 0.63, 0.67. Level coordinate by unit, ages ≥ 2: unit 2 74.5%, unit 12 8.4%, unit 6 7.7%, unit 14 7.3%, unit 11 4.0%, unit 1 2.8%, unit 0 2.5%, units 7 and 8 −6.1% and −3.9%. Remaining θ_4 terms at ages 1..8: other wave units 0.117, 0.111, 0.061, 0.046, 0.059, 0.097, 0.145, 0.173; comparators 0.007, 0.003, −0.035, −0.054, −0.042, −0.006, 0.033, 0.051.
- how checked: b5_unit2.py; sums over modes of L·R on unit 2 equal 1 (printed check).

### B5.3 [INTERVENTION, confidence high]

Unit 2 is not a standalone integrator. Its self-weight 0.969 leaks 3.1% per step (−0.044 to −0.057 of 10.157·M per step) while the level eigenvalue is 1.0099; the difference is supplied by the other units. Cutting row 2 of W down to (self 0.969, unit 4 1.731) lowers accuracy from 95.3% to 83.0%; with the self-weight set to 1 it is 86.8%; removing only the other wave units' weights gives 76.6%, only the comparators' 89.7%, only unit 9's 94.1%. Scaling h_2 by 0.8 or 1.2 moves θ_4 at the next step by ∓0.175·M, exactly 0.0485·(f−1)·h_2.

- numbers: Weight cuts: n = 400k, seed 1092, network 0.9528. θ_4/M at ages 1, 3, 5, 8: network 0.979, 0.925, 0.830, 0.926; self + unit 4 only: 0.963, 0.822, 0.768, 0.912 (h_2/(10.157·M) = 1.73, 1.62, 1.52, 1.41, pure 0.969^k decay; false-firing rate 1.95%); self = 1: 0.980, 0.889, 0.875, 1.090; no other wave units: 0.954, 0.781, 0.669, 0.789. MEASURED feeds (slopes on M, /10.157, seeds 1050, 1051) at ages 2..9: comparators −0.030, +0.011, −0.049, −0.086, −0.073, −0.015, +0.052, +0.088; other wave units +0.106, +0.131, +0.066, +0.014, −0.015, −0.016, +0.022, +0.082; unit 9 −0.035 at age 2. Largest sources at ages 3–9: unit 8 +0.148, unit 6 −0.114, unit 1 −0.093, unit 0 −0.056. Scaling h_2 (n = 200k, seed 1052, t0 = 3 or 5): shift ∓0.175·M at t0+1, 0.171–0.175 at t0+2, 0.11–0.16 at t0+4; the level-mode part alone predicts 0.116–0.119, because a unit-2 perturbation is 66% level and 36% runner in its effect on θ_4 (0.7956 × 0.4066 = 0.324 and 0.0353 × 5.081 = 0.179 of 0.492).
- how checked: b9_more_checks.py part 3 and b5_unit2.py; the predicted next-step shift 0.0485·(f−1)·h_2 was matched to 4 decimals.
- caveats: Weight cuts change the whole trajectory, so the accuracy losses measure how much the circuit depends on those inputs, not which downstream quantity breaks.

### B6.1 [MIXED, confidence high]

The level is the common reference of all five input thresholds. Per unit of level/10.157 the next-step thresholds move by 0.7956 (unit 4) and 0.6311, 0.7492, 0.7822, 0.7477 (comparators 1, 6, 7, 8). Scaling the level coordinate of h_t0 by f, with every other mode coordinate and the pulses unchanged, shifts θ_4(t0+1) by a median of ±0.073–0.080·M for f = 1 ± 0.1 and ±0.146–0.160·M for f = 1 ± 0.2, and the comparator thresholds by ±0.116–0.157·M for f = 1 ± 0.2. One step later the θ_4 shift is 1.008–1.009 times larger for t0 ≥ 3 (free flow: 1.0099).

- numbers: EXACT: dθ_4 = −W[4,ORD]·RIGHT_level; dθ_j = −λ·RIGHT_level[j]·10.157/w_j. The level also feeds the pulses: +0.512 per unit of level into unit 5 and −6.099 into unit 9. Readout: U[:,ORD]·RIGHT_level = (−0.452, −0.517, −0.531, −0.505, −0.308, −0.435, −0.476, −0.417, −0.168, +1.943). INTERVENTION: n = 400k, seed 1060, t0 in {1, 3, 5, 7, 8, 9}, f in {0.8, 0.9, 1.1, 1.2}. Median θ_4 shift/M for f = 1.2: 0.160 (t0 = 1), 0.148, 0.146, 0.146, 0.146, 0.147. A 20% level change is 22% of the norm of the ordinary state (the level-mode part has 1.11 times the norm of the whole ordinary state at t0 = 5).
- how checked: After the perturbation the level coordinate is f times the old one, every other mode coordinate changes by less than 1e−12 and the pulses are identical (printed check); the measured shift equals 0.7956·(f−1)·level/10.157 within 1e−9 (assert).
- caveats: For f < 1, 71–99% of the perturbed states contain a negative ordinary activation, which no real state has.

### B6.2 [INTERVENTION, confidence high]

Later firings of unit 4 respond as the threshold shift predicts. With f = 1.2 unit 4 misses records: P(on | positive-part record) over the remaining steps falls from 0.9995 to 0.956 (t0 = 3), 0.934 (t0 = 5), 0.899 (t0 = 7), 0.880 (t0 = 8), and false firings fall from 1.5–1.8% to 0.2–0.5%. With f = 0.8 the false-firing rate rises from 1.5–1.8% to 3.5%, 4.5%, 5.1%, 5.1% and no records are lost. For f = 1.1 record detection falls to 0.987, 0.980, 0.965, 0.958; for f = 0.9 false firings rise to 2.5%, 3.1%, 3.3%, 3.2%.

- numbers: n = 400k, seed 1060. Baseline over the steps after t0 = 3, 5, 7, 8: P(on | record) 0.9995, 0.9994, 0.9992, 0.9989; P(on | not record) 0.0154, 0.0175, 0.0169, 0.0157. Sequences with any changed unit-4 firing: 1.3–9.9% over all (t0, f); e.g. t0 = 3: 9.9%, 4.9%, 4.8%, 9.0% for f = 0.8, 0.9, 1.1, 1.2.
- how checked: b6_causal.py; at step t0+1 the changed firings are exactly the inputs between the old and the shifted threshold, since the shift is exact.
- caveats: Same caveat on f < 1 as B6.1.

### B6.3 [INTERVENTION, confidence high]

The final answer depends on the level far beyond unit 4. Scaling the level at t0 = 3, 5, 7 changes 11.6%, 15.4%, 15.3% of answers for f = 0.9; 13.2%, 16.5%, 14.4% for f = 1.1; 21.5%, 29.9%, 31.0% for f = 0.8; 20.2%, 25.1%, 20.6% for f = 1.2 (accuracy falls from 95.3% to 68–86%). Only 5–20% of the changed answers come with a changed unit-4 firing. For f > 1 the new answer is the position of the true max in 75–85% of changes, concentrated where the runner-up is older than the max; for f < 1 it is the third largest in 36–49%. At the last step (t0 = 9) only 2.1–4.2% of answers change, 86–87% of them to or from position 9, as the readout vector (+1.94 at position 9) predicts.

- numbers: n = 400k each, seeds 1060 and 1071 (P(changed) agrees to 0.001 between them). Accuracy after: t0 = 3: 0.768, 0.862, 0.849, 0.785; t0 = 5: 0.691, 0.829, 0.819, 0.738; t0 = 7: 0.679, 0.831, 0.839, 0.781; t0 = 9: 0.929, 0.947, 0.943, 0.927 for f = 0.8, 0.9, 1.1, 1.2. Among changed answers the gates that differ after t0: wave units 55–99%, comparators 16–64%, pulses 5/9 17–50%, unit 4 5–20%; with no gate change at all 2–21% of answers still change. Older vs younger runner-up (seed 1076, t0 = 5, q = runner-up position, p = max position): q < p ≤ t0: P(changed) = 0.451 (f = 1.1), 0.544 (f = 1.2), 96% of them to the max; p < q ≤ t0: 0.108, 0.152. Noise-free two-pulse inputs (M = 1.5 at p, S in {0.3, 1.0, 1.4} at q, background −1 or −0.3, p, q ≤ 4, t0 = 4, all 120 answered correctly at baseline): at f = 1.2 the answer flips to the max position in 43 of 60 cases with q < p and 9 of 60 with q > p; at f = 1.1 in 38 and 7; the two candidates' logits move by 13–16 while only one or two wave-unit gates differ.
- how checked: b6_causal.py, b7_followups.py parts B and C, b6_probe_twopulse.py, b6_probe_breakdown.py. Robustness: clipping negative activations after the f = 0.8 perturbation gives 19.1%, 23.8%, 27.1% changed answers instead of 21.6%, 29.9%, 31.0%.
- caveats: The path from the level to the logits was not traced; I only located which gates change. The f < 1 rows involve unphysical states (see B6.1).

### B6.4 [INTERVENTION, confidence high]

Replacing unit 4's activation at every step by the oracle 10.157·max(x_t − max(0, x_0..x_(t−1)), 0) gives the same answer as the network in 96.32% of sequences and a task accuracy of 93.12% (network 95.29%; second sample 93.17%). The oracle's two parts act in opposite directions: taking only its gate (unit 4 set to 0 when x_t is not a positive-part record, the network's own value otherwise) raises accuracy to 95.63% (+0.34 pp), whereas taking its size on true records costs 2.4 pp on sequences without false firings (96.71% → 94.30%). Accuracy is sensitive to the gain: oracle × 0.9 gives 84.4%, × 1.1 gives 85.7%. An oracle that also fires on negative records gives 71.6%; removing unit 4 gives 17.9%.

- numbers: n = 800k, seeds 1061, 1062. Oracle: right→wrong 2.84%, wrong→right 0.67%. Oracle gate only: agreement 99.40%, right→wrong 0.09%, wrong→right 0.44%; on the 8.5% of sequences with a false firing 89.45% → 93.53%, no change elsewhere. Full oracle by subset: sequences with a false firing 89.45% → 89.55%; ≤ 1 positive input 21.8% → 22.0%; the rest 96.71% → 94.30%. Agreement: × 0.9: 86.5%; × 1.1: 87.5%; true-max oracle 73.4%; unit 4 removed 18.5%. Oracle size on true records with false firings left in (seeds 1080, 1081, n = 800k): 92.41% (−2.90 pp), agreement 96.41%.
- how checked: A patch that writes back the network's own value reproduces the network exactly (assert). Two further independent implementations in b8_excess_paths.py give identical answers on all 800k sequences of seeds 1080, 1081.
- caveats: M is taken with its positive part, as specified; at t = 0 the oracle is 10.157·max(x_0, 0).

### B6.5 [INTERVENTION, confidence medium]

The oracle size hurts because the record update is a closed loop on the network's own threshold, θ_4 ← θ_4 + 0.986·(x_s − θ_4). In the network the threshold for the input right after a record is θ_4 = 0.985·M_new + 0.106·sag (sag = M_old − θ_4 before the record) and θ_7 = 1.018·M_new − 0.088·sag: the record resets both whatever the sag was. With the oracle size they come out as 0.985·M_new − 0.877·sag and 1.018·M_new − 1.109·sag (θ_1: sag coefficient −0.372 → −1.205); the differences equal unit 4's direct weights on those thresholds (0.986, 1.016, 0.838). Two steps after the record the oracle run still carries −0.82·sag in θ_4 against +0.15·sag in the network.

- numbers: n = 400k, seeds 1070 and 1091; records at s over an old max > 0.2; regression of the threshold computed from h_s on (sag, M_new, M_old), R² 0.996–0.9999. θ_6 and θ_8 are the same in both runs and sit near the old max minus the sag: θ_6 = 1.061·M_old − 1.222·sag − 0.048·M_new, θ_8 = 1.109·M_old − 1.070·sag − 0.035·M_new (direct kicks −0.044, −0.034). Mean (θ_4 − M)/M by age 0..8, network vs oracle run: −0.012/−0.029, −0.022/−0.036, −0.030/−0.041, −0.075/−0.083, −0.127/−0.132, −0.164/−0.169, −0.160/−0.168, −0.119/−0.131, −0.069/−0.082. Path-specific removal of the excess 10.157·sag (n = 800k, seeds 1080, 1081), change in accuracy: level only −2.26 pp, trans −2.57, short −1.38, clock −1.30, unit 5 −0.57, h2 −0.55, h3 −0.42, readout −0.42, unit 4 self-weight −0.11, runner −0.05, alt −0.03, echo and unit 9 −0.02, all nine ordinary modes −3.29, all paths −2.90.
- how checked: b7_followups.py part A, b9_more_checks.py part 2, b8_excess_paths.py. Checks that could have failed: the paths add up to W[:,4] (assert), and 'all paths, all firings' reproduces the oracle patch on every sequence.
- caveats: The carry-over of the sag into θ_4, θ_7, θ_1 is measured directly. That these mis-set thresholds are the whole cause of the 2.4–2.9 pp loss is not shown: the path effects are not additive, and removing the excess from a single mode (level, or the short-lived trans mode) costs almost as much as removing it everywhere.

## Contradictions with the lead

1. F5, R²: "R² ≥ 0.997 within age cells" is not reproduced for late ages. A through-origin fit of the level on M (states with M > 0, centred R²) gives 0.9963, 0.9967, 0.9965, 0.9957, 0.9933, 0.9896, 0.9872, 0.9865 at ages 2..9 (n = 800k, seeds 1030, 1031). Adding S⁺ as a regressor gives 0.9965–0.9981 at ages 2–5 and 0.987–0.995 at ages 6–9.

2. F5, meaning of 0.93: confirmed as an average (slopes 0.914–0.938 at ages 2–9), but it is not a constant of the level mode. For an isolated max the ratio grows from 0.929 (age 2) to 1.014 (age 9); on N(0,1) data it stays flat only because unit 5 fires after each new runner-up. The level also depends on S⁺ (coefficient −0.02 to −0.07). The 1.006 at age 1 is the from-rest value and the age-1 median; over an old max the level is 1.011·M_new − 0.046·M_old⁺.

3. F7, scope: the readout vector is reproduced, and a level change at the last step changes only 2–4% of answers (86–87% of them to or from position 9). But a level change at steps 3–7 of ±10% changes 12–17% of answers and ±20% changes 20–31%, mostly without any change in unit 4's firing. "Blind to the level" should not be read as "the answer does not depend on the level".

4. Framing of the task (θ_4 against M; oracle 10.157·max(x_t − M, 0)): unit 4 is not the oracle and θ_4 is not the level. θ_4 = 0.7956·level + 0.0353·runner + a clock term of amplitude 0.16 (all /10.157); replacing unit 4 by the oracle lowers accuracy from 95.3% to 93.1%.

No disagreement with F1–F4 (reproduced exactly) or F6 (my clock term of θ_4 matches 0.160·ρ(a)·cos(φ(a) − 6.6°) with the lead's phases). F8 and F9 were not examined.

## Surprises

1. The oracle makes the network worse (93.1% against 95.3%). Only removing the false firings helps (+0.34 pp); using the exact increment on true records costs 2.4–2.9 pp, and a ±10% gain error on unit 4 costs about 10 pp.

2. The threshold sags a lot: for an isolated max θ_4 falls to 0.70·M at ages 5–6 and returns to 0.88·M at age 8 (clock period 8.4 steps). At ages 5–6, 28–29% of new runner-ups fire unit 4.

3. A record resets several thresholds at once through unit 4's direct weights: θ_4 (0.986), θ_7 (1.016) and partly θ_1 (0.838) go to the new max, while θ_6 and θ_8 stay near the old max minus the sag.

4. Scaling the level up by 10% flips 45% of answers from an older runner-up to the newer max when both were already seen (54% at 20%), also on noise-free two-pulse inputs, with only one or two wave-unit gates changing.

5. Negative inputs are summed, not stored: the runner coordinate accumulates 1.03·x_t per sub-threshold input (a tenth of a record kick), and its effect on θ_4 cancels the level's within 3%.

6. The negative prefix is not fully forgotten: with the first positive input at step 8 and a second at step 9, resetting the state before step 8 lowers accuracy from 98.7% to 93.0%.

7. Two positive inputs give the highest accuracy (97.9%); one gives 23.1%, none 11.9%. Accuracy is continuous through S = 0 (0.60 just below, 0.68 just above), so the limit is a soft floor at S ≈ 0.05·M, not the sign.

8. Unit 5 fires one step after 99.85% of new runner-ups, unit 9 after 99.3% of records and 53% of false firings but only 0.3% of plain new runner-ups.

9. Turning unit 2 into a pure integrator of unit 4 costs 12 pp (8.5 pp even with a self-weight of exactly 1).

## Not done

findings.md was NOT written. The Write tool refused it ("Subagents should return findings as text, not write report files") and I did not work around that with another tool. The complete findings are in this object; the raw printed output of every script is saved beside it as *_output.txt, so the lead can assemble findings.md from them.

Not done or not determined:
- Why a level change flips the answer between the max and an older runner-up. I located which gates change (mostly wave units) but did not trace the path to the logits.
- Whether the mis-set thresholds after an oracle-size record are the whole cause of the accuracy loss (path effects are not additive).
- What sets ε(a), the 1–6% of M by which θ_4 is short even with no gap.
- Whether the sag has a use; I only showed that its false firings cost 0.34 pp and that its excess in the kick is needed given the other weights.
- Unif(−1,1) inputs beyond accuracy by number of positive inputs (k = 0..3).
- The thresholds of units 5 and 9, the step-9 special case (unit 4's direct readout U[:,4]) and the size of the clock kick. I only measured when units 5 and 9 fire.
- A derivation of the runner coordinate as a function of the input history (regressions only).
- F8 and F9 were not re-verified.
- The f < 1 level perturbations produce states with negative activations (71–99% of states); I reported them with that caveat and a clipped variant, but did not design a strictly physical alternative.

## Open questions

- Which readout path uses the level relative to the other modes? Scaling the level up by 10% moves 45% of answers from an older runner-up to the newer max with almost no gate changes outside the wave units.
- Is the clock leakage into unit 4's threshold (2|c| = 0.16) a cost of sharing unit 2 and the wave units with the clock, or does the sag do something useful? Removing only its false firings gains 0.34 pp.
- After a record θ_4 and θ_7 are reset to the new max while θ_6 and θ_8 keep the old max minus the sag: is that how the old max becomes the stored runner-up?
- Why does removing the kick excess from the short-lived trans mode alone (eigenvalue 0.23) cost 2.6 pp? It suggests the comparators' state one step after a record depends on the exact kick size.
- The runner coordinate integrates every sub-threshold input with weight 1.03 (decay 0.9735 per step). Does the runner-up circuit use this running sum, or is it a nuisance term it has to cancel?
- What produces the 23% accuracy (twice chance) with one positive input when |S| > 0.05·M, and the strong bias towards answering position 0 (43%)?
- Does the same picture hold on Unif(−1,1), the training distribution, where gaps are distributed differently?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/blib.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b0_basics.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b0_basics_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b1_theta4.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b1_theta4_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b2_detection.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b2_detection_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b3_level.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b3_level_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b3b_unit5_after_runner.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b3b_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b4_explore.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b4_explore_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b4_negative.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b4_negative_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b5_unit2.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b5_unit2_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b6_causal.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b6_causal_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b6_probe_twopulse.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b6_probe_twopulse_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b6_probe_breakdown.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b6_probe_breakdown_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b7_followups.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b7_followups_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b8_excess_paths.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b8_excess_paths_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b9_more_checks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b9_more_checks_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b10_summary_numbers.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_B_max_circuit/b10_output.txt