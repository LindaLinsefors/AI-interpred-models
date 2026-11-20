# Round-1 arm I_errors: structured result

## Summary

NOTE ON DELIVERY: findings.md was NOT written. The environment refused the file ("subagents should return findings as text, not write report files"), and I did not route around that block. All findings are in this object; every table is in the script outputs i*_out.txt in c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/ (markdown tables, ready to paste).

Notation: M > S > T are the three largest inputs, at p_M, p_S, p_T; a1 = 9 − p_M, a2 = 9 − p_S; relative gaps g1 = (M − S⁺)/M, g2 = (S⁺ − T⁺)/M with S⁺ = max(S,0); θ_j(t) = −W[j]·h_(t−1)/w_j.

I1 (census, n = 2^21, seeds 1000–1007). Accuracy 0.95290 ± 0.00015. Of 98,784 errors: 63.6% answer the third, 20.2% the max, 16.2% fourth or lower. 92.4% of errors are either a non-positive runner-up (17.8%; accuracy 0.233 with one positive input, 0.123 with none) or a relative gap below 5% of M (74.6%: 54.1% runner-up≈third, 15.7% max≈runner-up, 4.9% both). With T > 0 and both gaps ≥ 0.2M (22.4% of sequences) accuracy is 99.97%. Order of arrival barely matters (runner-up before max 0.9551, after 0.9507). Position pairs range from 0.849 at (p_M,p_S) = (0,7) to 0.985 at (7,4).

I2 (shift and scale). Exactly scale-equivariant. Accuracy under x + c: 0.116, 0.134, 0.547, 0.859, 0.953, 0.952, 0.940, 0.912, 0.881 for c = −3, −2, −1, −0.5, 0, 0.5, 1, 2, 3; Unif(−1,1) 0.9384. The reference is zero: θ_j(0) = 0 exactly, and a runner-up at S = sM (third clearly negative) is found with probability 0.15, 0.31, 0.51, 0.83, 0.93, 0.96 at s = −0.25, −0.10, −0.05, 0, +0.05, +0.10. Negative shifts lose accuracy because the non-positive-runner-up classes grow; positive shifts because relative gaps shrink: re-weighting the c = 0 error table E(g1, g2) predicts the T > 0 accuracy at c = 0.5..3 and on Unif(−1,1), Unif(0,1) to within 0.1–1.1 pp.

I3 (interventions, n = 2^21; fixed/new as shares of the net's errors). (a) unit 4 := oracle: accuracy 0.9314, agreement 0.9633, fixed 14.3%, new 60.1%. (b1) misfires removed: 0.9563, 9.3% / 2.0%. (b2) misses repaired: 0.9529, 0.05% / 0. (c) unit 5 off after x_t < S_prev: 0.9588, 15.8% / 3.3%. (d) = b1+b2+c: 0.9621, 23.8% / 4.2%. (b1) repairs 64% of the errors in the class "max≈runner-up, max first" and (c) 41% of "runner-up≈third, runner-up first"; the mirror classes (later input is the larger one) and the non-positive runner-up are not repaired (0–8%). The oracle hurts because the net's own record pulse is 10.157·(x_t − θ_4) = oracle + 10.157·(sag of θ_4), and the readout is calibrated to that.

I4 (near ties). Exact coefficients: for S − T < 0.05 the coefficients of S and T in logit[p_S] − logit[p_T] are nearly equal and opposite (medians +70..+80 and −69..−80; slope k ≈ 70–78 logits per unit gap; sum within 10% of k in 56–69% of sequences). Input sweeps show the comparison is linear in the gap over ±0.15M (62–85 logits per unit) but its zero is displaced: a later candidate takes over from an earlier one when it exceeds it by ε*·M, ε* quartiles −0.010 / −0.001 / +0.011, 5–95% −0.031 / +0.041. 70% of var(ε*) is a fixed function of the three positions (p_e, p_l, p_M) (cross-validated), and three-value inputs on a constant background reproduce the cell medians (r = 0.91, 360 configurations). The max-vs-later-input comparison is twice as sharp (record offset quartiles −0.003 / +0.001 / +0.006), steep at the tie (≈160 logits per unit within ±1% of M) and saturating; its zero does not sit at θ_4.

I5 (ages). Accuracy does not fall for an old runner-up (a2 = 5..8: 0.958–0.965, the best; a2 = 1, 2: 0.942, 0.940, the worst). It falls at both ends of the max's age: a1 = 0: 0.935, a1 = 9: 0.931, a1 = 8: 0.949 (mid ages 0.953–0.963). 59% of the between-cell variance of the (a1, a2) error table is interaction. Mean margin rises with a1 (8.3 → 12.9) and is lowest for adjacent pairs (4.3–6.8). Removing unit-4 misfires recovers only 0.3–0.7 pp at a1 = 7..9.

I6. Eight hand-checkable failing inputs (background −1, robust to backgrounds −0.3, −0.5, −2), each with the threshold that was crossed or missed: see the two I6 claims.

## Claims

### I1-a [MEASURED, confidence high]

On 2^21 N(0,1) sequences the net's accuracy is 0.95290 ± 0.00015; of the 98,784 errors 63.6% answer the position of the third-largest input, 20.2% the max and 16.2% the fourth or lower.

- numbers: n = 2,097,152 (8 batches of 2^18, seeds 1000-1007); per-seed accuracy 0.952-0.954. Share of errors by rank of the answered input: rank 1: 0.2018; 3: 0.6359; 4: 0.0545; 5: 0.0254; 6: 0.0224; 7: 0.0189; 8: 0.0162; 9: 0.0139; 10: 0.0109. Answer minus true position, share of errors: -9 .029, -8 .043, -7 .049, -6 .063, -5 .060, -4 .054, -3 .060, -2 .074, -1 .124, +1 .091, +2 .071, +3 .059, +4 .052, +5 .045, +6 .038, +7 .041, +8 .026, +9 .021. Adjacent answers are 21.5% of errors (chance 20%): 15.3% of max-answers, 21.6% of third-answers, 28.9% of fourth-or-lower answers. The wrong answer is position 0 in 19.1% of errors and position 9 in 15.3%. Of the fourth-or-lower answers 64.6% have S <= 0; they answer position 0 in 40.0% and p_M - 1 in 19.9%.
- how checked: i01_census.py. Rank of the answer and sign of the margin recomputed with a sort-based routine on 20,000 fresh sequences (seed 1090): exact agreement; accuracy equals common.accuracy (i99_checks.py).
- caveats: Binomial standard error only; ties in the inputs have probability zero.

### I1-b [MEASURED, confidence high]

92.4% of all errors fall into two classes: the runner-up is not positive (17.8% of errors) or one of the two scale-free gaps g1 = (M-S)/M, g2 = (S-T+)/M is below 0.05 (74.6% of errors). When both gaps are at least 0.2 the error rate is 0.04%.

- numbers: Mutually exclusive classes (share of sequences / error rate / share of errors / errors answering max-third-4th+): M<=0: 0.0010 / 0.877 / 0.0187 / .34-.12-.54. S<=0<M: 0.0098 / 0.767 / 0.1594 / .26-.14-.59. g2<0.05<=g1: 0.1531 / 0.166 / 0.5405 / .01-.92-.08. g1<0.05<=g2: 0.0780 / 0.095 / 0.1567 / .83-.16-.01. both<0.05: 0.0081 / 0.284 / 0.0486 / .35-.63-.01. min gap in [0.05,0.1): 0.2022 / 0.0131 / 0.0562. min gap in [0.1,0.2): 0.2967 / 0.0028 / 0.0178. both>=0.2: 0.2511 / 0.0004 / 0.0022. Error rate vs g2 (T>0, g1>=0.1) in bins [0,.005) [.005,.01) [.01,.02) [.02,.03) [.03,.05) [.05,.075) [.075,.1) [.1,.15) [.15,.2): 0.440, 0.318, 0.203, 0.116, 0.052, 0.0205, 0.0096, 0.0046, 0.0020. Error rate vs g1 (T>0, g2>=0.1), same bins: 0.360, 0.193, 0.096, 0.039, 0.0155, 0.0068, 0.0044, 0.0028, 0.0020. With T>0 and both gaps >= 0.2M (22.45% of sequences, n about 470,000) accuracy is 0.99972.
- how checked: i01b_relative_gaps.py, i05_ages.py on the saved census. Class conditions are asserted to be mutually exclusive and exhaustive in i03b_iv_by_class.py.
- caveats: Gaps use positive parts S+ = max(S,0), T+ = max(T,0); the gap classes require S > 0.

### I1-c [MEASURED, confidence high]

Error rate against the absolute gaps: it is governed by S-T when M-S is not tiny, and the row with large M-S is worse only because large M-S goes with non-positive S and with large M (the resolution is relative to M).

- numbers: Rows M-S in [0,.025) [.025,.05) [.05,.1) [.1,.2) [.2,.4) [.4,.8) >=.8; columns S-T in the same bins; error rates. Row 1: .421 .261 .219 .189 .171 .172 .196 (all .2005). Row 2: .306 .128 .075 .048 .031 .023 .038 (.0574). Row 3: .283 .093 .048 .024 .013 .005 .006 (.0350). Row 4: .274 .080 .034 .018 .008 .002 .000 (.0287). Row 5: .276 .085 .034 .015 .006 .001 .000 (.0289). Row 6: .308 .109 .042 .017 .007 .001 .000 (.0354). Row 7: .377 .198 .096 .040 .015 .004 .000 (.0647). Column marginals: .3186 .1296 .0612 .0301 .0163 .0100 .0116. Cells hold 4,014-151,158 sequences. In row 7, 24.4% of the errors have S <= 0 (2.1% of its sequences, error rate 0.756). For S>0, g1>=0.1 and absolute S-T<0.05 the error rate rises with M: 0.149 (M<1), 0.194 (1-1.5), 0.231 (1.5-2), 0.282 (M>=2).
- how checked: i01_census.py; the two explanations for row 7 were each tested by conditioning (S<=0 share; error rate by M at fixed absolute gap).

### I1-d [MEASURED, confidence high]

Accuracy by positions (p_M, p_S) ranges from 0.849 to 0.985; order of arrival matters little in aggregate; by age the worst cases are the max at the last step and at the first step.

- numbers: Accuracy, rows p_M = 0..9, columns p_S = 0..9 (. = diagonal). 0: . .973 .950 .943 .931 .945 .940 .849 .930 .918 | 1: .963 . .971 .942 .957 .952 .949 .933 .927 .948 | 2: .953 .958 . .957 .960 .943 .966 .963 .950 .944 | 3: .967 .959 .951 . .959 .965 .947 .971 .952 .969 | 4: .976 .970 .972 .962 . .958 .967 .947 .943 .957 | 5: .962 .974 .961 .967 .939 . .964 .966 .965 .972 | 6: .952 .970 .978 .951 .966 .909 . .958 .962 .969 | 7: .926 .966 .968 .971 .985 .967 .899 . .950 .948 | 8: .950 .966 .966 .968 .971 .966 .975 .957 . .952 | 9: .902 .948 .940 .962 .972 .951 .925 .917 .899 . (22,907-23,699 sequences per cell, s.e. about 0.0014). By p_M (0..9): .9310 .9491 .9549 .9601 .9613 .9634 .9572 .9535 .9634 .9352. By p_S: .9500 .9650 .9619 .9583 .9600 .9507 .9479 .9401 .9422 .9527. Runner-up before the max 0.9551, after 0.9507. By arrival order of the three (earliest to latest): MST .9489, MTS .9450, SMT .9514, STM .9601, TMS .9582, TSM .9538. With which of the top three is at position 9: max .9352, runner-up .9527, third .9306 (78% of those errors answer 9), none .9587. By number of positive inputs (share / accuracy / share of errors): 0: 0.00100 / 0.1227 / 0.0187; 1: 0.00978 / 0.2327 / 0.1594; 2: 0.04374 / 0.9786 / 0.0199; 3+: 0.94547 / 0.9600 / 0.8020.
- how checked: i01_census.py; same sequences as I1-a.
- caveats: The weak cells with the runner-up right before the max, (6,5), (7,6), (9,8), were not analysed further.

### I2-a [MEASURED, confidence high]

The net is exactly scale-equivariant (answers unchanged under x -> lambda x, lambda > 0) but not shift-invariant: accuracy under a constant shift c falls on both sides of c = 0, and Unif(-1,1) gives 0.9384.

- numbers: Accuracy for c = -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3: 0.1164, 0.1336, 0.5465, 0.8588, 0.9529, 0.9519, 0.9400, 0.9118, 0.8812 (n = 2^21 each). Unif(-1,1) (seeds 1100-1107): 0.9384, with ans=max 0.0181 against 0.0095 on N(0,1). Accuracy by sign class at c = 0: no positive input 0.123; exactly one (S<=0<M) 0.233; exactly two (S>0>=T) 0.979; three or more (T>0) 0.960. Across shifts the class with one positive input has accuracy 0.04-0.25 and the classes with S>0 keep 0.92-0.98. Class shares at c = -1: 0.178, 0.334, 0.284, 0.204 (formula for the first two: (1-Phi(c))^10 + 10 Phi(c)(1-Phi(c))^9 = 0.513). Scale: identical answers on 100,000 sequences at scales 0.01, 0.5, 7, 1000.
- how checked: i02_shift_scale.py. Homogeneity also checked on hidden states (max error 5e-13, i00_basics.py).

### I2-b [FIT, confidence high]

The loss under positive shifts and on uniform inputs is explained by the two relative gaps alone: re-weighting the c = 0 error table E(g1, g2) of the T > 0 class predicts the accuracy of other input distributions to within about 1 pp.

- numbers: E(g1,g2) from 1,982,794 N(0,1) sequences on a 24x24 log grid (477 cells with n>=50). Actual (in covered cells) vs predicted accuracy of the T>0 class: N(0,1): 0.9608 vs 0.9606; +0.5: 0.9532 vs 0.9489; +1: 0.9419 vs 0.9352; +2: 0.9153 vs 0.9082; +3: 0.8872 vs 0.8835; Unif(-1,1): 0.9464 vs 0.9451; Unif(0,1): 0.8999 vs 0.8888. Coverage 0.982-0.999. Fresh seeds 1200-1203 and 1300-1303, 4 x 2^18 each.
- how checked: i02b_reference.py part (1). Out-of-distribution prediction: it uses nothing but (g1, g2) and could have failed.
- caveats: The prediction is slightly low (0.1-1.1 pp) at positive shifts; inputs ranked fourth and lower are ignored by the table.

### I2-c [MIXED, confidence high]

The net's implicit reference is zero: all five thresholds are exactly 0 at t = 0 and stay within about 0.03 of zero until the first positive input, and a runner-up is found only if it lies above roughly -0.05 M.

- numbers: [EXACT] h_(-1) = 0 and no biases give theta_j(0) = 0 for units 4, 1, 6, 7, 8. [MEASURED n = 400,000, seed 1040] before any positive input the 1%/99% quantiles of theta_j(t) are 0.000/0.013 (unit 4), -0.019/0.023 (1), -0.032/0.006 (6), -0.003/0.008 (7), -0.029/0.014 (8). [MEASURED, input sweep, n = 3,301, seed 1060, third < -0.3 M, runner-up moved to s M] P(answer = p_S) at s = -0.25, -0.20, -0.15, -0.10, -0.05, -0.02, 0, +0.02, +0.05, +0.10, +0.25: 0.150, 0.168, 0.204, 0.308, 0.512, 0.702, 0.829, 0.889, 0.930, 0.956, 0.981; smallest s answered correctly has quartiles -0.12 / -0.05 / -0.01; with the runner-up at position 9, 0.914 already at s = -0.05. [EXACT, stated input] two-value inputs (M = 1 at p_M in {0,3}, v at a later position p, background -1 or -0.3): the net answers p iff v > v*, v* in [-0.119, +0.030]; with no runner-up stored the threshold of unit 5 on x_p is -0.04 to -0.11 (background -1) while the lowest comparator threshold is 0.31-0.63 when the max is 5-9 steps old. When exactly one input is positive (n = 20,518) the net answers p_S 0.233, p_M 0.202, p_M - 1 0.193, p_T 0.109; with no positive input (n = 2,103) it answers position 0 in 52.6% and position 1 in 22.0%.
- how checked: i02b_reference.py parts (2), (3); i06_thresholds.py; example E1 (wrong for x_4 in [-0.9, -0.031], correct above).
- caveats: The reference is a soft band around zero, not a sharp threshold; one two-value case (p_M = 3, p = 7, background -0.3) answers p for any v because 7 is the default answer there.

### I3-a [INTERVENTION, confidence high]

Interventions on the pulse units: removing unit-4 misfires and unit-5 pulses after sub-runner-up inputs repairs about a quarter of the net's errors and creates few new ones; replacing unit 4 by the oracle at every step makes the net worse.

- numbers: n = 2^21, net accuracy 0.9529, 98,784 errors. Columns: sequences touched / agreement with the net / task accuracy / net's errors fixed / new errors (both as shares of the net's errors). (a) unit 4 := 10.157 max(x_t - max(M_prev,0), 0) every step: 0.9172 / 0.9633 / 0.9314 / 0.1432 / 0.6005. (b1) unit 4 := 0 when x_t <= M_prev: 0.0846 / 0.9941 / 0.9563 / 0.0929 / 0.0199 (identical counts with the zero-floored condition). (b2) oracle where x_t is a positive record but unit 4 is silent: 0.0014 / 1.0000 / 0.9529 / 0.0005 / 0.0000. (b) = b1+b2: 0.0860 / 0.9941 / 0.9564 / 0.0934 / 0.0199. (c) unit 5 at t+1 := 0 when x_t < S_prev: 0.2517 / 0.9904 / 0.9588 / 0.1582 / 0.0328. (c') with max(S_prev,0): 0.2872 / 0.9891 / 0.9581 / 0.1604 / 0.0504. (d) = b1+b2+c: 0.3155 / 0.9856 / 0.9621 / 0.2384 / 0.0423. (a)+(c): 0.9454 / 0.9572 / 0.9376 / 0.2687 / 0.5934.
- how checked: i03_interventions.py. A plain per-sequence Python loop reproduces all nine interventions on 1,500 sequences (identical answers); the harness with no intervention reproduces the net (asserted on every batch).
- caveats: M_prev, S_prev are the largest and second largest of x_0..x_(t-1); interventions act on the running net, so 'silent' in (b2) is evaluated on the intervened trajectory.

### I3-b [INTERVENTION, confidence high]

Each pulse intervention repairs one specific error class: (b1) the case where the runner-up arrives after the max and slightly below it, (c) the case where the third arrives after the runner-up and slightly below it. The mirror cases (the later input is the larger one) and the non-positive runner-up are not repaired.

- numbers: n = 2^21. Class: share of errors; fixed/new as shares of the class's errors under (b1), (c), (d). M<=0: 0.0187; 0/0, 0/0, 0/0. S<=0<M: 0.1594; 0/0, .002/.001, .002/.001. runner-up~third (g2<0.05<=g1), runner-up first: 0.2806; .069/.012, .410/.020, .471/.025. Same, third first: 0.2600; .010/.013, .038/.044, .048/.055. max~runner-up (g1<0.05<=g2), max first: 0.0860; .643/.043, .097/.058, .664/.045. Same, runner-up first: 0.0707; .011/.001, .082/.073, .096/.074. both gaps<0.05: 0.0486; .158/.089, .202/.059, .306/.122. min gap in [0.05,0.1): 0.0562; .101/.081, .130/.040, .208/.090. min gap>=0.1: 0.0200; .082/.031, .079/.026, .157/.040.
- how checked: i03b_iv_by_class.py; the asymmetry between a class and its mirror (64% vs 1%, 41% vs 4%) is the check: a non-specific intervention would not show it.

### I3-c [INTERVENTION, confidence high]

The oracle hurts because unit 4's own pulse at a record is 10.157 (x_t - theta_4(t)) = oracle + 10.157 (max(M_prev,0) - theta_4(t)), i.e. the oracle plus the sag of its threshold, and the downstream readout needs that sum.

- numbers: n = 2^20 (seeds 1000-1003). Oracle only at true records (misfires untouched): accuracy 0.9242, fixes 6.4% of the net's errors, makes 67.3% new ones. Adding the sag term back reproduces the net's answers on 100.000% of sequences. Net vs oracle-at-records accuracy by age of the previous max when the final max arrives (1..9): .9536/.9256, .9686/.9403, .9683/.9429, .9736/.9192, .9731/.9156, .9629/.9031, .9490/.8981, .9528/.8996, .9030/.8832; median relative sag of theta_4 at that step: .012, .022, .032, .074, .123, .160, .156, .117, .069. When the final max is the first positive input, accuracy is unchanged (0.9193 vs 0.9203).
- how checked: i03c_oracle_detail.py; identity h4 = 10.157 max(x_t - theta_4, 0) verified to 1e-14 on 200,000 states (i99_checks.py).
- caveats: I did not identify which downstream weights cancel the sag term.

### I3-d [MIXED, confidence high]

Unit 4 fires on an input iff it exceeds a threshold theta_4 that sags below the running max and recovers with the age of the max; unit 5 is not a top-two detector but a graded score with a threshold about 0.13 M below the runner-up.

- numbers: [MEASURED n = 400,000 x 10 steps, seed 1020] unit 4 vs oracle: R^2 = 0.9950; P(on | record) = 0.99937; P(on | no record) = 0.0112; 8.4% of sequences contain a misfire (accuracy 0.893 vs 0.959 with none), 0.14% a miss. Median theta_4/M by age of the max at the step (1..9): .988 .978 .968 .925 .875 .837 .840 .878 .928; 1% quantile: .980 .965 .954 .873 .767 .697 .705 .773 .854; misfire rate per state: .0022 .0038 .0053 .0141 .0248 .0337 .0320 .0230 .0131. [FIT] at ages 6-7 (theta_4 - M)/M = -0.31 + 0.22 S+/M + 0.05 T+/M, R^2 0.35-0.44. Unit 5 at t+1 after a positive x_t of rank 1, 2, 3, 4+ among x_0..x_t fires with probability 0.998, 0.999, 0.388, 0.046; after non-positive inputs below 0.05. [EXACT + MEASURED n = 2,177,591 states, seed 1041] when comparators 7 and 8 are on and unit 4 is off at step t (68% of states, t = 1..8), unit 5 fires at t+1 iff x_t > theta_5(t), with gain W[5,7]w_7 + W[5,8]w_8 + ... = 18.0 (17.9-18.3) on x_t (rule holds in 100% of those states). (theta_5 - S+)/M quartiles: -0.184 / -0.131 / -0.076. [FIT] theta_5/M = a + b S+/M with a = -0.01..-0.09, b = 0.70..0.88, R^2 0.66-0.96 by age of the max. P(theta_5 > S) = 0.009; P(theta_5 < S - 0.1 M) = 0.77.
- how checked: i03a_pulse_events.py, i06b_unit5_threshold.py; the threshold rule for unit 5 was checked state by state.
- caveats: theta_5 is defined with the gates of step t held fixed and includes the small term w_5 x_(t+1) (w_5 = -0.28).

### I3-e [INTERVENTION, confidence medium]

Supplementary input-level intervention: replacing every input that is below the runner-up so far (it can never be the answer) by a low value repairs at most about 40-46% of the errors, mostly those where a later third beats the runner-up.

- numbers: n = 2^20. Replacement beta*max(M_prev,0): beta = 0: accuracy 0.9654, fixed 0.299, new 0.033; beta = -0.3: 0.9664, 0.330, 0.044; beta = -1: 0.9669, 0.389, 0.091; beta = -1 plus (b1): 0.9712, 0.459, 0.069; beta = -3: 0.9563, 0.462, 0.388. With beta = -1, fixed/new by class: runner-up~third with runner-up first 0.791/0.046; third first 0.295/0.098; max~runner-up with max first 0.210/0.296; runner-up first 0.239/0.166; S<=0 0.025/0.010.
- how checked: i03d_input_clamp.py; the label is asserted unchanged by the replacement.
- caveats: The result depends on the replacement value (very negative values add errors through leakage), so it is a bound, not a clean attribution.

### I4-a [MIXED, confidence high]

In near ties between runner-up and third, the exact coefficients of S and T in the margin logit[p_S] - logit[p_T] are nearly equal and opposite with a slope of about 70-78 logits per unit gap, but the margin has an offset of comparable size, so the offset rather than the gap usually decides.

- numbers: Coefficients exact per sequence (gates fixed), statistics over the 253,087 of 2^21 sequences with S-T<0.05 (answers: p_S 77.3%, p_T 18.9%, p_M 1.3%); clean subset T>0, M-S>=0.1: n = 212,601, p_S 79.2%, p_T 19.5%. Medians of c_S / c_T / k = (c_S - c_T)/2 by order and gap bin [0,.0125), [.0125,.025), [.025,.05): runner-up first: 80/-77/78, 73/-70/72, 72/-69/70; third first: 78/-80/79, 73/-74/73, 70/-72/71. 10-90% range of k about 20-38 to 126-333. c_S + c_T: median -0.9..+2.7, 10-90% about -13..+14; within 10% of k in 56-69% of sequences; k > 0 in 97.4-98.3%. Terms of the margin on the clean subset (logits): k(S-T) median 1.60 (10-90%: 0.23-4.10); offset b = margin - k(S-T): median 0.41, sd 3.33, mean abs 2.42. The three parts of b each have sd 10-19 logits and cancel (correlation -0.85 and -0.53). sign(margin) = sign(b) in 76.6%. For max vs runner-up near ties (M-S<0.05; 157,162 sequences; answers p_S 87.0%, p_M 9.3%) the coefficients of M and S are again opposite (sum within 10% of k in 58-92%), median k = 149-159 for gap<0.0125, 105-111 for [0.0125,0.025), 57-79 for [0.025,0.05).
- how checked: i04_near_ties.py; the backward-pass coefficients agree with traces() to 5e-13 and sum_p c_p x_p equals the margin to 1e-12.
- caveats: The decomposition is local (gates of the given input); b is the extrapolated margin at zero gap with gates held fixed.

### I4-b [MEASURED, confidence high]

Slope of the runner-up/third comparison by ages: it does not depend on the order of arrival, is 62-85 logits per unit gap for most age pairs, drops to 24-46 when the two candidates are adjacent and both at least 3-4 steps old, and rises to 100-155 when they are about four positions apart and old.

- numbers: Median k by age of S (rows 0..9) and age of T (columns 0..9), clean subset: 0: . 65 62 64 65 66 73 73 68 66 | 1: 65 . 63 72 73 73 74 94 81 69 | 2: 63 67 . 68 70 76 85 85 85 81 | 3: 64 73 71 . 46 63 90 110 113 83 | 4: 65 73 76 76 . 34 75 103 155 119 | 5: 65 72 76 84 73 . 34 77 122 154 | 6: 68 72 82 90 90 34 . 25 80 128 | 7: 69 71 78 99 94 71 31 . 24 77 | 8: 67 77 68 124 116 98 39 30 . 24 | 9: 66 69 71 69 145 140 101 32 28 . For max vs runner-up the median slope is near zero (-0.0 to 6.5) when the runner-up arrived two steps before the max and the max is at least 2 steps old (age pairs (2,4), (2,5), (3,5), (4,6), (5,7), (6,8), (7,9)) while the net is right in 97-100% there: the margin has already saturated.
- how checked: i04_near_ties.py (cells with n >= 100).

### I4-c [MEASURED, confidence high]

Input sweeps: the runner-up/third comparison is linear in the gap but its zero is displaced. A later candidate takes over from an earlier one when it exceeds it by eps* M, where eps* is spread around zero with a standard deviation of 0.0235; this distribution predicts the census error rates.

- numbers: n = 60,000 sequences (seed 1050; M-S>=0.35M, S>0.3M, fourth<S-0.2M); earlier of the two positions fixed at S, later one set to S + eps M, eps in [-0.15,0.15] step 0.0025; eps* = sign change of logit[p_l] - logit[p_e]. Exactly one sign change in 99.1%. Quantiles 5/10/25/50/75/90/95%: -0.0308 / -0.0214 / -0.0097 / -0.0008 / +0.0110 / +0.0279 / +0.0410; mean |eps*| = 0.0157. By position of the max: before both -0.025..+0.050 (median +0.004, mean abs 0.0188); between -0.041..+0.038 (median -0.004, 0.0184); after both -0.021..+0.028 (median -0.001, 0.0099). Later candidate at position 9: mean abs 0.0222 vs 0.0142 otherwise. Share answering the later one at eps = -0.10, -0.05, -0.03, -0.02, -0.01, 0, +0.01, +0.02, +0.03, +0.05, +0.10: .004 .016 .054 .113 .246 .522 .734 .841 .907 .965 .994. Median of (logit[p_l] - logit[p_e])/M at eps = -0.15, -0.10, -0.05, -0.01, 0, 0.01, 0.05, 0.10, 0.15: -10.4, -7.4, -4.0, -1.25, +0.10, +1.22, +3.9, +7.1, +10.2; local slope 62-71 logits per unit gap, 83-85 within +-0.01. Prediction of the census (T>0, g1>=0.1; less restricted sweep sample, seed 1081, 79% of sequences qualify): runner-up first, observed P(answer = p_T) vs predicted P(eps* < -g2) in g2 bins [0,.005) [.005,.01) [.01,.02) [.02,.03) [.03,.05): .454/.461, .315/.313, .192/.192, .101/.095, .039/.039; third first, observed vs P(eps* > g2): .383/.398, .279/.286, .184/.182, .110/.104, .054/.048.
- how checked: i04b_takeover.py, i04d_takeover_checks.py, i04g_margin_shape.py. The less restricted sample gives the same quartiles (-0.011 / -0.001 / +0.010); an offset is unchanged when the whole input is scaled by 3.7 (to 1e-9).
- caveats: eps* is defined on the two-position logit difference; along the sweep the argmax is one of the two positions in more than 99.9% of cases.

### I4-d [MIXED, confidence high]

What decides near ties between runner-up and third is mostly a fixed function of the three positions: cell means over (p_e, p_l, p_M) explain 70% of the variance of eps*, and three-value inputs on a constant background reproduce those cell medians.

- numbers: [FIT] cell means over (p_e, p_l, p_M), 360 cells: R^2 = 0.695 in sample and 0.695 cross-validated on a random half; (p_e, p_l) alone 0.077; sign of eps* equals the sign of the cell median from the other half in 84.3%. [EXACT, stated input] three-value inputs (1 at p_M, 0.5 at p_e, later candidate swept, background -1): correlation 0.908 with the N(0,1) cell medians over 360 configurations, median absolute difference 0.0054; over configurations eps* has quartiles -0.015 / -0.004 / +0.007, 5-95% -0.038 / +0.039, |eps*| > 0.05 in 5.6%. Largest cells (p_e, p_l, p_M): N(0,1) median / three-value: (0,7,9) -0.097 / -0.107; (0,8,9) +0.081 / +0.115; (0,9,4) +0.073 / +0.086; (0,9,6) -0.071 / -0.097; (3,4,0) +0.068 / +0.079; (8,9,6) +0.065 / +0.067; (1,7,0) +0.057 / +0.052; (7,8,9) -0.045 / -0.054. The offset is not proportional to M - S: across levels x_e/M = 0.1..0.9 mean |eps*| stays 0.016-0.023 and eps* = c (1 - x_e/M) explains none of the within-configuration variance. [MEASURED n = 902,555] predicting 'net answers the third' for census sequences (T>0, g1>=0.05) from the three-value table alone: predicted rate 0.0336 vs observed 0.0302; recall 0.61, precision 0.55; for g2<0.01 recall 0.71, precision 0.65 at a base rate of 0.36.
- how checked: i04c_takeover_fit.py, i04d_takeover_checks.py, i04e_three_value_model.py. The direct scan of example E5 gives +0.0522 for cell (1,7,0), equal to the table entry.
- caveats: The three-value table explains about 60% of these confusions; the other seven inputs still move the offset and that part is not modelled. No closed-form derivation of the offsets from the weights.

### I4-e [MEASURED, confidence high]

The max-versus-later-input comparison is about twice as sharp as runner-up versus third, is steep at the tie and saturates away from it, and its zero is not the firing threshold of unit 4: theta_4 sags by up to 20-30% of M while the input level at which a later input is treated as the new max stays within about 2% of M.

- numbers: n = 60,000 (seeds 1051/1052; max at p_e, a later input set to M(1+eps), all other inputs < 0.7M); eps* = sign change of logit[p_e] - logit[p_l]. Quantiles 5/10/25/50/75/90/95%: -0.0164 / -0.0109 / -0.0034 / +0.0009 / +0.0061 / +0.0158 / +0.0224; mean |eps*| = 0.0080; one sign change in 99.9%. Share answering the old max at eps = -0.05, -0.03, -0.02, -0.01, -0.005, 0, +0.005, +0.01, +0.02, +0.03, +0.05: .001 .009 .031 .109 .196 .438 .716 .822 .932 .975 .997. Slope of the logit difference: 22-27 logits per unit for eps < -0.03, 97-159 within +-0.02, 55 beyond +0.03. theta_4/M - 1 when the later input arrives: 5/50/95% = -0.219 / -0.024 / -0.006; eps* > theta_4/M - 1 in 98.2%; correlation of eps* with that sag 0.08. By d = age of the max at that step (1..9): median sag -.012 -.022 -.033 -.084 -.148 -.200 -.204 -.158 -.098; median eps* +.0025 +.0001 -.0003 -.0032 +.0011 +.0042 -.0086 +.0085 -.0209; mean |eps*| .0081 .0039 .0065 .0092 .0084 .0123 .0138 .0141 .0226. [FIT] cell means over (p_e, p_l) explain 38% of var(eps*); with the position and level of the runner-up so far 64%. [EXACT, stated input] two-value inputs (background -1): correlation 0.86 with the N(0,1) cell medians over 45 cells; mean |eps*| 0.0088; largest: (p_e,p_l) = (0,6) +0.039, (0,9) -0.038, (2,9) -0.031, (0,8) +0.030, (3,7) +0.028.
- how checked: i04b_takeover.py, i04c_takeover_fit.py, i04f_two_value_records.py, i04g_margin_shape.py, i05c_offsets_by_age.py; examples E2 and E3 are the cells (0,9) and (0,8).

### I5-a [MEASURED, confidence high]

Accuracy does not fall when the runner-up is old; it is lowest when the max arrives at the last step or was the first input. Most of the structure of the (a1, a2) table is interaction between the two ages, not either age alone.

- numbers: n = 2^21. Accuracy by runner-up age a2 = 0..9: .9527 .9422 .9401 .9479 .9507 .9600 .9583 .9619 .9650 .9500. By max age a1 = 0..9: .9352 .9634 .9535 .9572 .9634 .9613 .9601 .9549 .9491 .9310. Share of errors answering the max by a1: .20 .20 .22 .18 .14 .13 .11 .18 .24 .33. The 2-D accuracy table is the (p_M, p_S) table of I1-d with a = 9 - p; lowest cells (a1,a2): (9,2) .849, (0,1) .899, (2,3) .899, (0,9) .902, (3,4) .909, (0,2) .917, (9,0) .918; highest (2,5) .985, (3,7) .978, (5,9) .976. Additive fit of the error rate E = mu + r(a1) + c(a2): mu = 0.047; r = (+.018, -.011, -.001, -.004, -.011, -.008, -.007, -.002, +.004, +.022); c = (.000, +.011, +.013, +.005, +.002, -.007, -.005, -.009, -.012, +.003); shares of between-cell variance: a1 27%, a2 14%, interaction 59%. Mean margin (logit[p_S] minus best other) by a1 = 0..9: 8.3 9.4 7.8 8.5 9.0 11.0 11.6 11.6 12.1 12.9; by a2: 12.8 12.3 10.3 10.4 9.8 9.1 9.2 9.4 9.2 9.6; lowest cells are adjacent pairs at least two steps old, e.g. (2,1) 4.3, (3,2) 4.6, (2,3) 4.8, (3,4) 5.1, (4,3) 5.1, (8,9) 5.4, and (0,9) 5.8; highest are old max with recent runner-up, e.g. (6,0) 18.2, (9,4) 18.3. With T>0 and both gaps >= 0.2M accuracy is 0.99972 and the error rate is below 0.1% in every cell except (a1,a2) = (0,9) 1.27%, (9,0) 0.36%, (2,9) 0.17%, (8,9) 0.13%.
- how checked: i05_ages.py. The (0,9) easy errors inspected on a fresh seed (1070; 5 of 724) all answer a third at position 7, the configuration (p_e,p_l,p_M) = (0,7,9) which has the largest takeover offset in I4-d.
- caveats: The full 10x10 mean-margin and median-margin tables are in i05_ages_out.txt.

### I5-b [INTERVENTION, confidence high]

The sag of unit 4's threshold accounts for only a small part of the loss for an old max: removing misfires recovers 0.3-0.7 percentage points at a1 = 7..9.

- numbers: n = 2^20. By a1 = 0..9: share of sequences with a unit-4 misfire .079 .061 .046 .035 .037 .052 .083 .119 .150 .180; net accuracy .9361 .9637 .9532 .9572 .9637 .9610 .9602 .9545 .9487 .9307; under (b1) .9418 .9667 .9553 .9582 .9650 .9635 .9633 .9611 .9551 .9334; under (d) .9446 .9678 .9569 .9628 .9708 .9689 .9717 .9723 .9631 .9428; share of errors answering the max that (b1) repairs .04 .03 .09 .09 .17 .40 .44 .59 .51 .29. By a2 = 0..9, net vs (d): .9532/.9629, .9420/.9472, .9405/.9433, .9477/.9548, .9501/.9612, .9594/.9704, .9579/.9715, .9627/.9737, .9653/.9738, .9499/.9627.
- how checked: i05b_ages_iv.py (seeds 1000-1003).
- caveats: The remaining deficit at a1 = 0, 8, 9 is attributed to the larger comparison offsets (I4-e) by their size, not by a dedicated intervention.

### I6-a [EXACT, confidence high]

Five failing inputs with different mechanisms (background -1; same wrong answer with backgrounds -0.3, -0.5, -2). E1 zero reference; E2 false record through the sagged theta_4; E3 missed record; E4 a later third wins through the unit-5 pulse; E5 a late runner-up fails to displace an earlier third.

- numbers: E1: x = (-1,-1,1,-1,-0.2,-1,-1,-1,-1,-1); correct 4, net 1. At step 4 the thresholds are theta_4 .978, theta_1 -.016, theta_6 .675, theta_7 1.012, theta_8 .302; x_4 = -0.2 is below all of them and unit 5 does not fire at step 5; logits 3.56 (pos 1) vs 1.74 (pos 4). Wrong for x_4 in [-0.9, -0.031]. No pulse intervention changes it. E2: x = (1,-1,-1,-1,-1,-1,-1,-1,-1,0.98); correct 9, net 0. theta_4 runs .988 .979 .963 .878 .772 .703 .710 .782 .878 over steps 1..9; x_9 = 0.98 > .878, unit 4 fires with 10.157(0.98 - 0.878) = 1.04 and comparators 1, 6, 7 switch off, as for a new max; logits 50.52 (pos 0) vs 48.64 (pos 9), slopes in x_9: -56 (logit 9), +50 (logit 0). Wrong for x_9 in [0.962, 1.000). (b1) gives the correct answer 9. E3: x = (1,-1,-1,-1,-1,-1,-1,-1,1.02,-1); correct 0, net 8. At step 8 theta_4 = .782; unit 4 fires with 2.41, of which only 0.20 is the true increment (a runner-up of 0.9 would give 1.19); logits 48.84 (pos 8) vs 47.67 (pos 0), slopes in x_8: -25 and +89. Wrong for x_8 in (1.000, 1.030]; correct at 1.06. Not repaired by (b1), (c) or the oracle (a). E4: x = (1,-1,-1,0.5,-1,0.49,-1,-1,-1,-1); correct 3, net 5. x_5 = 0.49 crosses no comparator threshold (lowest theta_8 = .648) but unit 5 fires at step 6 for any x_5 > 0.316 (h_6[5] = 3.12); logits 15.35 (pos 5) vs 14.40 (pos 3), slopes in x_5: +69 and -26. Wrong for x_5 in [0.480, 0.500). (c) gives the correct answer 3. E5: x = (1,0.5,-1,-1,-1,-1,-1,0.54,-1,-1); correct 7, net 1. At step 7 all comparator thresholds are .753-1.194, above the stored 0.5, so x_7 crosses none; its only trace is the unit-5 pulse at step 8, 18.0(0.54 - 0.286) = 4.57, one step before readout; logit 7 = 17.27 (slope +61.5 in x_7), logit 1 = 18.01 (slope +1.3). Wrong for x_7 in (0.500, 0.552]. With the newcomer at step 4 instead the answer is right (margin 3.7). No pulse intervention repairs it.
- how checked: i06_examples.py (explain.py prints the full step tables). For each: a neighbouring input answered correctly, a scan of the free value in 4,000 steps, three other backgrounds, and the interventions (a), (b1), (c), (d).
- caveats: The wrong intervals are 2-5% of M wide; these are near ties by construction (except E1), which is where 75% of real errors are.

### I6-b [EXACT, confidence high]

Three further failing inputs: E6 a third at the last position crosses comparator thresholds lying below the runner-up; E7 a runner-up arriving immediately after the third gets a comparison with less than half the usual slope; E8 with no positive input nothing crosses zero and the net answers its default, position 0.

- numbers: E6: x = (0.5,-1,-1,-1,-1,-1,1,-1,-1,0.45); correct 0, net 9. At step 9: theta_4 .996, theta_1 .447, theta_6 1.006, theta_7 1.025, theta_8 .347; thresholds of units 8 and 1 lie below the runner-up 0.5 and x_9 = 0.45 crosses both; logits 29.63 (pos 9) vs 24.71 (pos 0), slopes in x_9: +56 and +16. Wrong for x_9 in [0.403, 0.500): a third 19% smaller than the runner-up still wins. No pulse intervention repairs it. E7: x = (1,-1,-1,0.5,0.55,-1,-1,-1,-1,-1); correct 4, net 3. x_3 = 0.5 switches off comparators 1 and 8 (theta_1 .298, theta_8 .046); x_4 = 0.55 arrives while unit 5 fires for x_3 (h_4[5] = 4.44), switches off comparator 8 only (theta_8 .477) and gets a unit-5 pulse of 2.73 at step 5 (threshold .357); logits 15.20 (pos 3) vs 13.99 (pos 4), slopes in x_4: -17 and +25. Wrong for x_4 in (0.500, 0.579]. E8: x = (-1,-0.6,-1.4,-0.1,-0.8,-0.3,-1.2,-0.5,-0.9,-0.7); correct 5, net 0. All five thresholds stay within 0.021 of zero at every step, no comparator switches off, no pulse fires after step 0; logits 0-8 lie between 1.2 and 5.0.
- how checked: i06_examples.py, same checks as I6-a.

### X-1 [MEASURED, confidence high]

The gate pattern with all 13 ordinary units on and the three pulse units off (the pattern of the resting map A) holds in only 9% of states; several wave units are off about a fifth of the time.

- numbers: n = 400,000 sequences (seed 1041): 9.3% of states at t = 1..8. n = 200,000 (seed 1042), by step t = 2..9: 0.059 0.084 0.074 0.054 0.082 0.158 0.235 0.288 (0 at t = 1). All 13 ordinary units on, pulses free, t = 1..9: 0.241 0.224 0.261 0.195 0.132 0.157 0.262 0.363 0.426. Share of states with the unit off at t >= 2: unit 0 0.208, 1 0.227, 3 0.218, 6 0.220, 7 0.179, 8 0.268, 15 0.251 (others below 0.09). All four comparators on, t = 1..9: 0.491 to 0.817. Number of ordinary units off (t >= 2): 0: 0.253, 1: 0.283, 2: 0.185, 3: 0.086, 4+: 0.19.
- how checked: Direct count of pre-activations > 0 from common.run(return_pre=True) on two seeds.
- caveats: This limits how many steps F3, F4 and F7 (stated for that pattern) describe; it does not contradict them.

### X-2 [EXACT, confidence high]

F8 re-derived from the weights holds with two qualifications: unit 9's kernel of about -3.1 one position before the input holds only for pulse ages of at least 2, and unit 5's negative lobe covers only the three or four positions just before its own and only for ages of at least 2; unit 5 also puts about +1.0 to +1.4 on the position right after its own.

- numbers: Kernels U[:,ORD] A^(a-1) W[ORD,u] (a = age of the pulse at readout; a = 0 is U[:,u]). Unit 4 at its own position: -4.99 (a=0), -6.63 (a=1), -3.48, -3.84, -3.82, -3.97, -4.12, -3.88, -3.32, -3.18 (a=2..9). Unit 5 at the position of the input before it fired: 3.40, 3.37, 3.66, 3.82, 3.90, 4.16, 3.95, 3.59, 3.67 (a=0..8); on the 3-4 positions before that: 0 to -0.9 at a=0,1 and -0.7 to -2.4 at a>=2; on the position after: 1.08, 1.13, 1.11, 1.09, 1.00, 1.07, 0.85, 1.37 (a=1..8). Unit 9 one position before the input: 0.22 (a=0), -1.92 (a=1), -2.80, -3.02, -3.37, -3.14, -3.07, -3.17 (a=2..7); at a=1 it also has -3.19 at position 0. Direct response of the logits to the last input when all 13 ordinary units are on: U[:,ORD] w[ORD] = (-4.3, -4.5, -4.5, -4.4, -4.6, -4.6, -4.7, -4.9, -3.8, +57.2).
- how checked: Direct matrix products from the weights (printed in the session; the kernel script is not saved as a file, the formula is given here).

## Contradictions with the lead

Re-verified and in agreement: F1 (trace sum error 2e-13, homogeneity 5e-13), F2 (100% of states on fresh data), F4 (A w[ORD] entries <= 0.43; W[PULSE,ORD] w[ORD] = (0.01, 18.01, 84.33)), F8 (with the qualifications in claim X-2). F5-F7 were not re-tested. Points where my results disagree with the framing of the task rather than with a numbered fact:

1. The oracle for unit 4 is the wrong counterfactual. Intervention (a) lowers accuracy from 0.9529 to 0.9314 (60% new errors relative to the net's error count). Unit 4's own pulse at a record is 10.157 (x_t - theta_4) = oracle + 10.157 x sag, and the readout is calibrated to that sum; restoring the sag term reproduces the net exactly.

2. 'Thresholds sag, therefore errors' is only weakly true. theta_4 sags to a median of 0.84 M (1% quantile 0.70) when the max is 6-7 steps old and unit 4 then fires on any later input above that, but the level at which the net actually treats a later input as the new max stays at M(1 + eps*) with mean |eps*| of 0.008 overall and 0.012-0.023 at ages 6-9; eps* lies above the firing threshold in 98.2% of sequences and its correlation with the sag is 0.08. Removing misfires repairs 9.3% of all errors, concentrated in one class.

3. Unit 5 is not a 'new runner-up or new max' detector. It fires after 38.8% of positive rank-3 inputs; its threshold is theta_5 ~ 0.8 S+ - 0.05 M, a median of 0.13 M below the runner-up, and it lies below S - 0.1 M in 77% of states. Its activation 18.0 (x_t - theta_5) acts as a graded score, so 'unit 5 off after every input below the runner-up' removes a working part of the comparison rather than a misfire (it still helps, because that part can only favour the wrong position).

4. The all-on gate pattern behind F3, F4 and F7 holds in 9.3% of states at t = 1..8 (all 13 ordinary units on in 13-43% per step); units 0, 1, 3, 6, 7, 8, 15 are each off in 18-27% of states. Those facts are correct as stated but describe a minority of steps.

5. F8, two qualifications: unit 9's -3.1 one position before the input holds for pulse ages >= 2 only (-1.9 at age 1, +0.2 at age 0); unit 5's -1..-2 lobe covers only the 3-4 positions just before its own and only for ages >= 2.

6. F9 holds for the gaps tested, but two-value inputs do fail inside the record offset: (1,-1,...,-1,1.02 at position 8,-1) is answered 8 instead of 0, and (1,-1,...,-1,0.98 at position 9) is answered 0 instead of 9.

## Surprises

1. The runner-up/third comparison is close to an ideal linear comparator (coefficients +70..80 and -69..-80, linear over +-0.15 M); what fails is the position of its zero, and that displacement is 70% a deterministic function of the three positions (p_e, p_l, p_M), reproducible with three-value inputs (r = 0.91). So most runner-up/third errors are systematic, not noise from the other inputs.
2. The oracle intervention on unit 4 makes the net worse (0.9314), and the damage is present even when the previous max is one step old (sag 1.2%): 0.9536 -> 0.9256.
3. A sag of 16-30% in unit 4's threshold costs almost nothing by itself: the record decision stays within about 2% of M.
4. Old runner-ups are the best case (a2 = 5..8: 0.958-0.965), and the mean margin rises with the age of the max (8.3 at a1 = 0 to 12.9 at a1 = 9) although accuracy is lowest at a1 = 9.
5. The class with exactly two positive inputs is the most accurate (0.979 against 0.960 with three or more): a non-positive third cannot be confused with the runner-up.
6. Specific configurations have offsets of 5-11% of M: a third at position 7 beats an old runner-up at 0 when the max comes last (eps* = -0.10); a runner-up at 8 must beat a third at 0 by 8-11% when the max is at 9; a runner-up right after the third at positions 3, 4 needs 7-8% extra.
7. With no positive input the net answers position 0 in 53% of cases; with one positive input it answers p_M - 1 in 19% and p_M itself in 20%.
8. Re-weighting a single error table over the two relative gaps predicts the accuracy on shifted Gaussian and on uniform inputs to within about 1 pp, including Unif(0,1) at 0.89-0.90.

## Not done

1. findings.md was not written. The Write call for it was refused by the environment ('subagents should return findings as text, not write report files'), and I did not write the same content through another tool. All results are in this object; all tables are in the i*_out.txt files listed under files. If the lead needs findings.md, it has to be assembled from these by an agent that is allowed to write it.
2. No closed-form derivation of the takeover and record offsets eps* from the weights (mode coordinates, kernels). They are tabulated numerically with two- and three-value inputs (i04e_three_value_table.npy holds eps* for 360 configurations x 9 levels).
3. The three-value table explains about 60% of the runner-up/third confusions (recall 0.61, precision 0.55). How the other seven inputs move the offset is not modelled.
4. The weak position cells were explained individually only for (p_M,p_S) = (0,7), the last-position cases and (9,0); the cells with the runner-up right before the max, (6,5), (7,6), (9,8), were not analysed (while (8,7) is not weak).
5. Unit 9 was not intervened on, and I did not identify which downstream weights cancel the sag term of unit 4's pulse.
6. Unif(-1,1) was not broken down beyond accuracy, answer type, sign class and the re-weighting prediction.
7. The input-clamp intervention (I3-e) depends on the replacement value and was not refined into a clean attribution.
8. F5, F6, F7 were not re-tested. The kernel recomputation for F8 was run inline and is not saved as a script.

## Open questions

- What sets the zero of the runner-up/third comparison as a function of (p_e, p_l, p_M)? The offsets are reproducible with three-value inputs, so they should be derivable from the mode kernels; the largest ones involve a max at the last step or a later candidate arriving when the max is 7-9 steps old.
- Which readout weights compensate the sag of theta_4, so that the record decision stays at M within about 2% while unit 4 fires from 0.70-0.84 M upward? The compensation error grows with the age of the max (mean |eps*| 0.004 at age 2 to 0.023 at age 9).
- Is unit 5's threshold, theta_5 ~ 0.8 S+ - 0.05 M, the intended reference of a graded score, and what balances the earlier candidate's logit against 18.0 (x_t - theta_5) times the unit-5 kernel so that the comparison is fair on average?
- Why are the cells with the runner-up immediately before the max weak for p_M = 6, 7, 9 (0.909, 0.899, 0.899) but not for p_M = 8 (0.957)?
- How should the resting-map picture (F3, F4, F7) be extended to the 91% of states in which at least one ordinary unit is off or a pulse is on? Wave units 0, 3 and 15 are off in 21-25% of states at t >= 2.
- Why does a runner-up at position 9 escape the zero reference (found with probability 0.91 at S = -0.05 M against 0.51 overall), while a third at position 9 is the most over-answered wrong position (78% of the errors when the third is last)?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/icommon.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/explain.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i00_basics.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i01_census.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i01_census_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i01_census.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i01b_relative_gaps.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i01b_relative_gaps_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i02_shift_scale.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i02_shift_scale_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i02b_reference.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i02b_reference_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03_interventions.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03_interventions_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03a_pulse_events.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03a_pulse_events_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03b_iv_by_class.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03b_iv_by_class_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03c_oracle_detail.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03c_oracle_detail_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03d_input_clamp.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i03d_input_clamp_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04_near_ties.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04_near_ties_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04b_takeover.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04b_takeover_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04b_RT.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04b_MR.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04c_takeover_fit.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04c_takeover_fit_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04d_takeover_checks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04d_takeover_checks_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04e_three_value_model.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04e_three_value_model_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04e_three_value_table.npy
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04f_two_value_records.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04f_two_value_records_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04g_margin_shape.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i04g_margin_shape_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i05_ages.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i05_ages_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i05b_ages_iv.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i05b_ages_iv_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i05c_offsets_by_age.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i05c_offsets_by_age_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i06_examples.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i06_examples_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i06_thresholds.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i06_thresholds_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i06b_unit5_threshold.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i06b_unit5_threshold_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i99_checks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_I_errors/i99_checks_out.txt