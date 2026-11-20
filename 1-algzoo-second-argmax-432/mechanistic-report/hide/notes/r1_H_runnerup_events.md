# Round-1 arm H_runnerup_events: structured result

## Summary

Arm H (runner-up events), all from my own computations on the weights; 36 scripts/outputs in r1_H_runnerup_events. NOTE: findings.md could not be written (the harness refuses report .md files from subagents), so the full findings are in this object and the tables are in the *_out.txt script outputs.

Conventions: M, S, T = largest/2nd/3rd of x_0..x_(t-1) (before x_t); a = t − p1 = age of the max at the event step (1..9); r = (x_t − S)/(M − S); ρ_j = (θ_j − S)/(M − S); K = 10.157. N-range event: S < x_t < M, S > 0, t ≥ 2; N event: N-range with unit 4 silent; class L: 0.5 ≤ x_t/S < 1.

H1. θ_j = S + ρ_j(a)(M − S): each comparator's threshold sweeps between about S and above M as the max ages, so the set of clipped comparators is decided by (a, r). At ages 5–8 most N events clip NO comparator (65, 91, 83, 55 %). Unit 4 misfires in 9.9 % of N-range events (ρ_4 sags to 0.64 at ages 6–7); removing misfires raises accuracy 95.33 → 95.67 %.
H2. Exact: z5(t+1) = w5·x_(t+1) + 18.009(x_t − Θ5) − 10.462(x_t − θ_8)_+ − 7.813(x_t − θ_7)_+ + 0.167(x_t − θ_6)_+ − 0.005(x_t − θ_1)_+ + wave clips. With per-age threshold laws it predicts z5 out of sample with R² 0.91. Unit 5 fires after 99.9 % of N events, unit 9 after 0.34 % (99.2 % after new maxima).
H3. An N event writes +5.7..6.5 K(x_t − S) into the runner mode at ages 1–4 (2.6–3.8 later), shrinks the clock by (0.84, 0.83, 0.75, 0.53)K(x_t − S) at ages 1–4 and not at ages ≥ 5, and writes the unit-5 vector into h2/alt/h3/short. The logit bump is at position t itself: +42..48 per unit x_t (ages 1–4), +57..61 (ages 5–8), 74–100 % from the unit-5 pulse; the old runner-up's position drops by 9–23 per unit x_t. Final margin (t over old runner-up) ≈ 58..72·(x_t − S).
H4. Unit 5 is a fifth comparator: Θ5 = 0.01265·m_runner − 0.02872·m_level + small ≈ (0.70..0.86)S − (0.01..0.07)M, blind to the clock. It fires for half of the inputs at x_t/S ≈ 0.75–0.8. Forcing it off after every below-runner-up input raises accuracy 95.30 → 95.89 % (uniform 93.83 → 94.59 %): these records are harmful on net.
H5. Controlled M, S: all 45 pairs correct for S/M ≤ 0.9, 1–2 failures at 0.97. Position is carried by h3, short, h2; the runner mode only penalises position 9. With a third value, margin(p2 over p3) ≈ c(S − T), c = 77 / 50 / 64–66 by order of arrival.
H6. No second clock: every written mode then rotates/decays freely with its own eigenvalue (median R² ≥ 0.998).

## Claims

### H1a [MEASURED, confidence high]

At an N-range event (S < x_t < M, S > 0, t >= 2) each comparator threshold is theta_j = S + rho_j(a)*(M - S), where rho_j depends mainly on a = steps since the max arrived (1..9); comparator j is OFF iff r = (x_t-S)/(M-S) > rho_j, and unit 4 is ON iff r > rho_4. The thresholds sweep between about S (rho = 0) and above M (rho = 1.45) as the max ages.

- numbers: Median rho_j at the events for a = 1..9. unit 1: 0.84, -0.04, 0.26, 0.82, 1.29, 1.40, 1.12, 0.61, 0.15. unit 6: -0.05, 0.67, 1.05, 1.19, 1.07, 0.81, 0.52, 0.37, 0.42. unit 7: 1.02, 1.02, 1.01, 0.87, 0.69, 0.58, 0.58, 0.69, 0.85. unit 8: -0.02, 0.32, 0.06, 0.05, 0.42, 0.97, 1.39, 1.45, 1.14. unit 4: 0.98, 0.97, 0.95, 0.86, 0.74, 0.65, 0.64, 0.72, 0.83. Interquartile widths 0.01-0.24. Regression theta_j ~ bM*M + bS*S per age: R2 0.92-0.9999, bM+bS = 0.85-1.08. n = 891,562 events from 1.2M N(0,1) sequences (seeds 1001-1003).
- how checked: Gates read directly from pre-activations (h1_tables.py). Check that could have failed: predicting 'j off' from r > median rho_j(a) alone matches the true gate in 96.0/97.3/98.1/95.2 % of events (units 1, 6, 7, 8) and 97.5 % for unit 4. With S <= 0 and S+ = 0 the medians of theta_j/M agree with the table within 0.07 (h1_extra.py). F2 gate rule re-verified on 2M steps (1.000000).
- caveats: rho_j also depends weakly on the age of the runner-up and on the third value (largest scatter at ages 8-9 and for units 1, 6). I measured these laws; I did not derive them from the level/clock laws.

### H1b [MEASURED, confidence high]

Which comparators an N event clips is set by the age of the max: at ages 1-4 at least one comparator has its threshold at about S and is clipped by nearly every N event, but at ages 5-8 most N events clip NO comparator. The excess of a clipped comparator is x_t - theta_j = (r - rho_j(a))*(M - S).

- numbers: P(off | N-range event), all r, a = 1..9. comparator 1: .112 .996 .674 .134 .000 .000 .013 .326 .796. comparator 6: .999 .269 .007 .000 .001 .149 .419 .573 .526. comparator 7: .000 .000 .002 .087 .237 .351 .356 .252 .111. comparator 8: .972 .575 .921 .905 .461 .032 .000 .000 .016. Commonest off-sets (unit 4 silent): a=1 {6,8} 87 %; a=2 {1} 42 %, {1,8} 33 %, {1,6,8} 24 %; a=3 {1,8} 65 %, {8} 27 %; a=4 {8} 84 %; a=5 none 65 %, {8} 30 %; a=6 none 91 %; a=7 none 83 %; a=8 none 55 %, {6} 30 %; a=9 {1,6} 44 %, {1} 31 %, none 23 %. Mean excess in gap units (r - rho_j when off): 0.03-0.52 depending on unit and age. Full r x age tables (5 r-bins) in h1_tables_out.txt. n = 891,562.
- how checked: Direct gate counts on 1.2M sequences; r-resolved tables behave as step functions at r = rho_j(a) (e.g. comparator 8 at a=2: P(off) = .014, .379, .961, .986, .992 across r bins, rho_8 = 0.32).
- caveats: Restricted to S > 0 and t >= 2.

### H1c [MIXED, confidence high]

Unit 4 misfires (fires although x_t < M) in 9.9 % of N-range events, because its threshold sags below M as the max ages (rho_4 = 0.64-0.65 at ages 6-7). Misfires are nearly certain for r > 0.8 at max ages 5-8. Removing them by intervention raises accuracy by 0.34 pp on N(0,1) and 0.64 pp on Unif(-1,1).

- numbers: P(misfire) by age a=1..9: .017 .030 .040 .104 .200 .287 .300 .226 .129; by r bin [0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1): .003 .008 .025 .160 .498. For r in [.8,1): .125 .209 .296 .745 1.000 1.000 1.000 .999 .807 (a=1..9); for r in [.6,.8): .002 .008 .006 .035 .368 .765 .757 .472 .130. Median size h_t[4]/(K(M-S)) when on: 0.015-0.195 (largest at ages 6-7). Sequences containing an N-range event are answered correctly 88.9 % of the time with a misfire vs 96.1 % without. INTERVENTION (n = 1.2M per distribution, seeds 1026-1028): unit 4 forced to 0 at every step where x_t is not a new prefix maximum: accuracy 95.33 -> 95.67 % (agreement 99.42 %, broke 0.09 %, fixed 0.43 %); uniform 93.80 -> 94.44 %. 8.4 % (uniform 13.5 %) of sequences contain a misfire.
- how checked: h1_tables.py, h1_extra.py. Independent recount with plain loops on seed 1040 (h_independent_check.py): misfire rate 0.0990 vs 0.0987.
- caveats: The 88.9 % vs 96.1 % comparison is an association; the causal number is the intervention. I did not explain why rho_4 sags.

### H2a [EXACT, confidence high]

Exact form of unit 5's pre-activation: z5(t+1) = w5*x_(t+1) + c5*(x_t - Theta5(t)) + sum_{j in comparators} kappa_j*(x_t - theta_j)_+ + sum_{i in wave} W[5,i]*c_t[i], with c5 = W[5,ORD].w[ORD] = 18.009, kappa_j = W[5,j]*|w_j| = (-0.005, +0.167, -7.813, -10.462) for j = (1, 6, 7, 8), w5 = -0.281, c_t[i] = ReLU(-z_t[i]), and c5*Theta5 = -W[5,ORD].(W[ORD,:] h_(t-1)) - W[5,PULSE].h_t[PULSE]. So for a set C of clipped comparators: z5 - w5*x_(t+1) = c_x*x_t - sum_{j in C} kappa_j*theta_j - c5*Theta5 + wave clips, with c_x = 18.009 + sum_{j in C} kappa_j.

- numbers: Slopes in x_t: 18.009 (neither 7 nor 8 clipped), 7.546 (8 clipped), 10.195 (7 clipped), -0.267 (both clipped: the pulse saturates), -0.105 (all four). W[5,PULSE] = (1.187, -0.421, 0.297) for units (4, 5, 9). Dropping the wave-clip term costs a median error of 0.07 (90 %: 0.85, 99 %: 2.1) against sd(z5) = 17.
- how checked: Identity verified to 4e-14 .. 6e-14 on 3.6M steps of N(0,1) data (h2_unit5.py seeds 1004-1006, h2_formula.py seed 1023). The equivalent form with [theta_j - x_t]_+ terms and wave/pulse activations was checked separately.

### H2b [FIT, confidence medium]

Reduced to (x_t, S, M, age): z5(t+1) - w5*x_(t+1) ~ 18.009*(x_t - Theta5) - 10.462*(x_t - theta_8)_+ - 7.813*(x_t - theta_7)_+ with per-age laws Theta5 = bS*S + bM*M, theta_7, theta_8 linear in (S, M). Fitted on one seed and tested on another it explains 91 % of the variance of z5 over N events; the resulting pulse h_(t+1)[5] has slope about 7.8 per unit of x_t at ages 1, 3, 4 (comparator 8 clipped) and about 18 at ages 6-8, and does not vanish at x_t = S.

- numbers: Laws (seed 1023), a = 1..8, as (coef of S, coef of M). Theta5: (.791,-.070) (.694,.053) (.797,-.010) (.815,-.035) (.855,-.050) (.828,-.065) (.745,-.036) (.697,-.013). theta_7: (-.009,1.018) (-.004,1.014) (.002,1.004) (.122,.872) (.256,.710) (.330,.611) (.321,.616) (.235,.719). theta_8: (1.051,-.031) (.712,.312) (.949,.048) (.926,.064) (.568,.437) (.097,.953) (-.254,1.342) (-.338,1.408). Out-of-sample R2 (seed 1024, n = 235,710 N events): 0.906 overall; 0.860 0.850 0.931 0.931 0.967 0.939 0.919 0.859 for a = 1..8; with each event's true thresholds 0.971 overall. Sign agreement 99.7 %. Composed vs directly fitted coefficients (c_x, c_S, c_M) for the dominant patterns, e.g. a=1 '7 on, 8 off' (97 %): 7.55/7.85, -3.25/-3.77, 0.95/0.98; a=6 '7 on, 8 on' (91 %): 18.01/17.82, -14.91/-14.69, 1.18/1.16. Direct fit h5 ~ alpha*(x_t-S) + beta*S + gamma*M (n = 235k): alpha = 7.85 11.41 7.86 8.07 14.88 17.88 17.71 18.04; beta = 4.01 3.43 3.50 2.14 1.95 3.22 3.96 4.39; gamma = 1.00 1.16 0.56 1.18 1.49 1.11 0.90 0.65; R2 0.85-0.95.
- how checked: h2_formula.py: laws fitted on seed 1023, tested on seed 1024; per-(age, pattern) regression compared with coefficients composed from the weights and the laws.
- caveats: Per-age R2 is only 0.85-0.86 at ages 1, 2, 8. The formula fails in the three minority patterns with share <= 0.11 ('7 on, 8 on' at a = 1, 3, 4: fitted c_x 28.7, 12.7, 13.1 instead of 18.0), where x_t - S has almost no range. Wave-unit clipping matters most at a = 2.

### H2c [MIXED, confidence high]

After an N event unit 5 fires essentially always and unit 9 essentially never. Unit 9 is vetoed by comparator 7 (weight 66 per unit of threshold distance) and fires only when x_t is within about 0.03*M of theta_7 or above it, which makes it a new-maximum detector.

- numbers: EXACT: kappa9_j = W[9,j]*|w_j| = (-3.775, -0.981, -66.26, -13.44) for j = (1, 6, 7, 8). MEASURED (seed 1024, 2 <= t <= 8, S > 0): P(unit 9 on at t+1) = 0.9922 after a new max (n = 259,014), 0.0034 after an N event (n = 235,710), 0.5051 after an N-range misfire (n = 23,849), 0.0001 after x_t < S (n = 1,498,587). P(unit 5 on at t+1) = 1.0000, 0.9993, 1.0000, 0.0742 for the same classes. Among N events P(unit 9 on) by (x_t - theta_7)/M: 0.000 below -0.1, 0.002 in [-0.1,-0.03), 0.105 in [-0.03,0), 0.077 in [0,0.05). By (a, r) it reaches 0.23 only at a = 4, r > 0.8 and 0.08 at a = 5, r in [0.6, 0.8); elsewhere <= 0.025.
- how checked: h2_unit5.py (seeds 1004-1006) and h2_formula.py (seed 1024) give 0.0037 and 0.0034; independent loop count on seed 1040 gives 0.0042 with a slightly different t range.

### H3a [MIXED, confidence medium]

What an N event writes into the mode coordinates two steps later (difference to the same sequence with x_t := 0, per unit of x_t - S, in units of K): runner +5.7..6.5 at max ages 1-4 and +2.6..3.8 at ages 5-7; the clock radius is reduced by 0.84, 0.83, 0.75, 0.53 at ages 1-4 (written 159-177 degrees away from the clock's own phase) and is left essentially unchanged at ages >= 5; h2, alt, h3, short receive the position record, largest at ages >= 5.

- numbers: alpha (coefficient of x_t - S in dm_k/K ~ alpha*(x_t-S) + beta*S + gamma*M), a = 1..7. level: -.04 -.09 -.09 -.12 -.13 -.13 -.12. runner: 5.74 6.34 6.50 5.69 3.83 2.97 2.58. clock (amp@deg at t+2): .84@-99 .83@-40 .75@-7 .53@30 .10@42 .12@-153 .17@-127. h2: .52@-13 .05@-143 .41@-105 .83@-75 1.50@-82 1.56@-84 1.28@-83. alt: .25 .57 .40 .22 1.46 1.88 1.48. h3: .45@-10 .59@-51 .55@-46 .67@-31 1.33@-46 1.64@-52 1.25@-51. short: .95@-98 .71@-75 .42@-30 .16@-72 .72@-142 1.06@-148 1.00@-140. trans: -.19 -.15 -.10 -.14 -.26 -.30 -.26. beta (per unit S): runner .60-1.28; h2 .18-.35; alt .15-.56; h3 .20-.41; short .10-.35; clock <= .15. R2: runner .87-.98; clock .92-.98 at a <= 4; h2 .51-.97; alt .56-.93; h3 .63-.93; short .69-.93. n = 12.6k, 12.1k, 10.2k, 7.3k, 4.6k, 2.4k, 1.1k events (t <= 7, two negative followers) out of 235,412 N events (seeds 1011, 1012).
- how checked: h3_event.py / h3_analyze.py (counterfactual runs). Check that could have failed (h3_check.py, 7,727 events with no comparator clipped): regressing the mode change on the unit-5 pulse alone gives c_k = -0.077, 1.806, 0.064@-136, 0.885@-84, 1.075, 0.926@-52, 0.604@-145, -0.173 (level..trans) against the weights' LEFT_k.W[ORD,5] = -0.072, 1.723, 0.074@-138, 0.872@-85, 1.094, 0.935@-53, 0.599@-146, -0.172, R2 >= 0.99 for every mode.
- caveats: 'Change' is defined as the counterfactual difference at t+2, not m(t+2) - lambda^3 m(t-1). The x_t := 0 replacement is itself quiet in 93.1 % of cases. For h2/alt/h3/short at ages 1-4 the three-regressor fits have R2 of only 0.5-0.8, so those alphas are rough. Events with S <= 0 not analysed.

### H3b [INTERVENTION, confidence medium]

The logit kernel of a whole N event is a bump at position t itself. With quiet followers (all later inputs negative) the final logit at t rises by 18-23 times M on average, i.e. +42..48 per unit of x_t at max ages 1-4 and +57..61 at ages 5-8; position t+1 gets +0.6..4.4 times M; the old runner-up's position falls by 9-23 per unit of x_t. The unit-5 pulse supplies 74-100 % of the bump; clipped comparators add to it only at ages 1-4, where they also RAISE the max's position by 46-62 per unit of x_t.

- numbers: Mean logit change / M at t (a = 1..8): 22.7 21.2 18.2 17.9 19.8 20.3 19.6 21.1; at t+1: 2.9 4.4 2.7 2.1 2.7 1.7 0.6 -2.6; at the old runner-up: -8.7 -7.4 -6.6 -5.0 -3.9 -3.5 -2.9 -2.8; at the max: 13.0 10.5 6.6 2.9 -1.7 0.2 0.0 1.7. Regression coefficient per unit x_t at t: 47.8 47.1 42.1 42.9 58.2 61.0 56.8 61.2 (R2 .70-.95); at the old runner-up: -22.9 -17.7 -21.5 -16.1 -14.2 -10.9 -9.4 -10.5 (R2 .08-.37); at the max: 45.6 43.2 43.4 21.5 -2.2 3.4 4.6 20.4. Exact source split of the coefficient at t: unit 5 27.6 40.2 28.0 28.8 51.6 61.7 60.2 61.5; clipped comparators 23.9 15.1 21.1 18.6 6.4 -0.8 -0.5 -1.3; sources at steps >= t+2: -2.3 -9.1 -5.8 -0.7 -0.6 -0.7 -3.8 0.0; clipped wave units at t, units 9 and 4: |.| <= 0.7. At the old runner-up: comparators -16.8 -13.6 -18.7 -11.9 -4.3 -0.3 -1.1 -4.0, unit 5 -4.8 -7.4 -4.1 -4.6 -8.7 -9.9 -7.7 -6.2. Dependence on steps to the end d = 9 - t of the slope at t: a=1: 56.4 47.6 43.6 38.8 29.6 (d = 1..5); a=3: 61.9 38.0 26.1 19.9 18.1; a=5: 62.7 55.8 53.0 55.8 (d = 1..4). n = 35,609 quiet events of 235,412.
- how checked: h3_analyze2.py. The split uses the exact source decomposition logits = sum_s [Kin(9-s) x_s + sum_u Kp_u(9-s) h_s[u] + sum_i Kc_i(9-s) c_s[i]], which reproduces the logits to 8e-13 (h3_kernels_exact.py). Worked example (h3_example.py): M=1.5 at 0, S=0.6 at 3, new 0.9 at t=6, background -1: change +39.5 at 6, -14.4 at 3, +1.7 at 0, of which unit 5 gives +38.9, -14.1, +2.0.
- caveats: Quiet followers only; with all followers the slope at t is 24-42 at ages 1-4 and 54-61 at ages 5-8 (h3_analyze_out.txt). The numbers at the old runner-up's position are averages over its varying distance from t (low R2). Records made at max ages 1-4 lose about half of their slope within four steps; records at ages >= 5 were only observed for d <= 4.

### H3c [MIXED, confidence medium]

The comparison of a new value with the old runner-up comes out as a graded margin: with quiet followers the final logit[t] - logit[old runner-up] is about 58..72 times (x_t - S), with the S- and M-proportional terms cancelling. Without the event the old runner-up leads by about (13..22)*S + (0.4..5)*M; the event's bump has S- and M-proportional parts of that size. Most of the swing is the new bump (42-61 per unit x_t), the smaller part is the lowering of the old position (9-23), which comes from the negative lobe of the unit-5 kernel at earlier positions and, at max ages 1-4, from the clipped comparators.

- numbers: Margin ~ alpha*(x_t-S) + beta*S + gamma*M, a = 1..8. Actual: alpha 70.3 63.7 62.7 58.1 71.7 72.2 67.2 71.6; beta 2.6 2.2 6.9 -0.4 -0.7 0.2 -0.5 -0.8; gamma -0.1 2.4 -1.9 1.4 0.9 -0.3 -0.9 -0.3; R2 .77 .55 .48 .56 .92 .95 .94 .93. With x_t := 0: beta -21.6 -21.6 -18.8 -16.6 -12.6 -13.0 -15.7 -19.2; gamma -4.3 -0.4 -1.0 -1.6 -3.5 -5.3 -5.4 -2.6. P(logit[t] > logit[old]) by r: 0.45-0.86 for r < 0.05, 0.86-0.98 for r in [0.05, 0.1), >= 0.98 for r >= 0.1; overall 0.92-0.98 by age. EXACT: unit-5 kernel per unit h5 is +3.37..+4.16 at its own position, -0.37..-2.39 at earlier positions, +0.85..+1.37 at the next position.
- how checked: h3_analyze2.py on the counterfactual data set (n = 1.9k-6.0k quiet events per age). Worked examples (h3_example.py): with S = 0.6 the margin of the new position is +28.6 for x_t = 0.9, +4.1 for 0.63, -1.3 for 0.57.
- caveats: R2 of the margin fit is only 0.48-0.77 at ages 1-4. With non-quiet followers P(margin > 0) drops to 0.75-0.94 because later events intervene.

### H4a [EXACT, confidence high]

Unit 5 is a fifth comparator on x_t. When all 13 ordinary units are on at step t, z5(t+1) = w5*x_(t+1) + 18.009*(x_t - Theta5(t)) exactly, and Theta5 is read from the state as Theta5 = 0.01265*m_runner - 0.02872*m_level + small terms: it is 'runner coordinate minus level coordinate' and does not see the clock.

- numbers: Coefficients of Theta5 per unit mode coordinate of h_(t-1) (complex modes: 2|.|): level -0.02872, runner +0.01265, clock 0.00365, h2 0.00958, alt -0.00171, h3 0.01132, short 0.00751, trans -0.00509, echo 0.00001. Pulse terms: (-0.562, +0.284, +0.005).h_(t-1)[4,5,9]/18.009 and (-1.187, +0.421, -0.297).h_t[4,5,9]/18.009. The clock can shift Theta5 by at most about 0.037*(M - S).
- how checked: h4_theta5.py: formula exact to 4e-14 on the 20.4 % of steps with all 13 on; the mode parts plus pulses sum to Theta5 to 1e-14. On below-runner-up inputs with all 13 units on, the rule 'unit 5 on iff x_t > Theta5 - (w5/c5) x_(t+1)' is right in 100.0 % of cases; on ALL below-runner-up inputs 'on iff x_t > Theta5' is right in 98.97 %.
- caveats: For the laws I use a state-only version (unit 4 dropped, units 5 and 9 at step t evaluated for x_t = 0); it differs from the exact threshold by a median of 0.0000 and a 99th percentile of 0.011 on unit-4-off steps.

### H4b [FIT, confidence high]

Theta5 is about (0.70..0.86)*S - (0.01..0.07)*M, weakly dependent on the age of the max: the level part is -0.27..-0.29 M, the runner part 0.64..0.82 S + 0.11..0.25 M, so the M terms cancel. Unit 5 therefore fires for x_t/S above about 0.70-0.79 when the gap is small and at lower ratios when S << M. Theta5 follows a new runner-up with gain 0.58-0.73 at max ages 1-4 but only 0.36-0.41 at ages 5-6.

- numbers: Theta5 ~ bS*S + bM*M, a = 1..8: bS .793 .695 .798 .816 .856 .828 .746 .697; bM -.071 .053 -.011 -.035 -.050 -.065 -.036 -.013; R2 .889 .927 .966 .971 .977 .944 .884 .808 (with T+: .903-.986; within single (a, a2) cells .98-.99). n = 6.05M steps (seeds 1007-1009). Underlying: m_level/K = .93-.96 M (R2 >= .99); m_runner/K = (5.0..6.4) S + (0.9..2.0) M (R2 .75-.96). Median Theta5/S by gap (M-S)/M: 0.70-0.79 for gap < 0.2, 0.69-0.79 for 0.35-0.5, 0.62-0.82 for 0.5-0.7, 0.33-0.73 (1.00 at a = 2) for gap > 0.7. Ratchet (seed 1035, quiet followers): Theta5(t+3) - Theta5(t) ~ g*(x_t - Theta5(t)) after an N event with g = .73 .70 .64 .58 .41 .36 for a = 1..6 (R2 .96 .94 .89 .91 .87 .76); a pure unit-5 record gives g = 18.009*0.01265*1.723 = 0.39 exactly.
- how checked: h4_theta5.py, h4_theta5_explore.py, h4_ratchet.py. The crude rule 'unit 5 on iff x_t > 0.75*S' is right for 97.4 % of below-runner-up inputs. Controlled example: S = 0.6 arriving at max age 2 gives Theta5 = 0.30-0.45 afterwards, arriving at max age 6 gives 0.13-0.25.
- caveats: An earlier version of my regression (R2 0.68-0.80) wrongly included the unit-4 term on new-max steps; the numbers here use the corrected state-only threshold. For class-L events the ratchet regression has R2 <= 0.55 and I claim no law there.

### H4c [MEASURED, confidence high]

Measured firing of unit 5 after inputs below the runner-up: 7.4 % of all such inputs fire it; P(on) is 0.1 % for negative inputs, 51 % for x_t/S in [0.7, 0.8), 89 % in [0.8, 0.9) and 99.5 % in [0.9, 1). The 50 % point is at x_t/S of about 0.75-0.8 for gaps below 0.5 and moves down to about 0.65 for gaps above 0.7; it is lowest at max ages 1 and 6-8.

- numbers: P(unit 5 on at t+1 | x_t < S) by x_t/S bins <0, 0-.3, .3-.5, .5-.6, .6-.7, .7-.8, .8-.9, .9-1. All: .001 .009 .034 .082 .182 .507 .894 .995 (shares of inputs .673 .114 .070 .033 .031 .029 .026 .024). By gap (M-S)/M: 0-.1: .000 .004 .022 .048 .089 .343 .832 .996; .1-.2: .000 .005 .021 .047 .096 .389 .860 .998; .2-.35: .000 .005 .019 .047 .111 .433 .891 .998; .35-.5: .000 .005 .022 .054 .157 .537 .917 .998; .5-.7: .000 .007 .031 .105 .294 .675 .935 .997; .7-1: .004 .063 .178 .360 .541 .757 .916 .973. By age: a=1: .003 .027 .079 .175 .347 .674 .946 .999; a=3: .000 .002 .010 .029 .096 .428 .865 .991; a=5: .001 .003 .012 .036 .103 .398 .868 .996; a=7: .000 .008 .062 .158 .309 .676 .961 1.000 (all ages in h4_theta5_out.txt). n = 4.5M inputs with S > 0, unit 4 silent, 2 <= t <= 8 (seeds 1007-1009).
- how checked: Independent recount with plain loops on seed 1040 reproduces the 'all' row within 0.004 (.001 .009 .034 .079 .181 .505 .890 .995).

### H4d [INTERVENTION, confidence high]

A class-L record (unit-5 pulse after an input just below the runner-up) raises the final logit at its own position by about 3.3 per unit of h5 and lowers the runner-up's position by 0.4-0.7 per unit; the pulse is 1.5-3.6 on average. Without the pulse the runner-up leads the L position by about 21*S regardless of how close x_t is; with it the lead shrinks linearly to about 2*S just below a tie, and the network then answers the L position in 21 % of the cases with x_t/S in [0.95, 1).

- numbers: By x_t/S bin (0.5-0.7, 0.7-0.8, 0.8-0.9, 0.9-0.95, 0.95-1): mean h5 1.80 1.47 2.04 2.91 3.60; logit change per unit h5 at own position 3.29 3.21 3.26 3.29 3.31 (exact direct kernel 3.47-3.62), at t+1 +0.13..+0.43, at t-1 -0.43..-0.50, at the runner-up -0.36 -0.51 -0.61 -0.65 -0.67, at the max -0.42..-0.69. Margin (L position over runner-up)/S with / without the pulse: -27.1/-37.9, -18.8/-27.0, -11.9/-22.4, -6.2/-21.1, -2.1/-20.7. P(answer = L position) with / without: .0016/.0001, .0017/.0001, .0052/.0002, .0270/.0002, .2133/.0001. P(correct) with / without: .9658/.9695, .9736/.9786, .9684/.9774, .9390/.9698, .7574/.9561. n = 211,807 L events with unit 5 on (seeds 1016, 1017); the margin rows use the 112,974 events whose runner-up stays the final one.
- how checked: h4_Lkernel.py: one pulse per sequence forced to zero, final logits compared; compared against the exact kernel Kp_5.
- caveats: When the runner-up changes later (98,833 events) removing the pulse changes 1.6 % of answers with no net effect on accuracy (broke 0.81 %, fixed 0.81 %).

### H4e [INTERVENTION, confidence high]

Requested intervention: forcing unit 5 to zero at t+1 after EVERY input below the runner-up so far raises accuracy from 95.30 % to 95.89 % on N(0,1) and from 93.83 % to 94.59 % on Unif(-1,1); agreement with the unmodified network is 99.05 % and 98.49 %. So these records are harmful on net; nearly all of the gain comes from inputs with 0.9 <= x_t/S < 1.

- numbers: N(0,1), n = 1.2M (seeds 1013-1015): every x_t < S: accuracy 0.9589, agreement 0.9905, broke 0.0015, fixed 0.0073. Class L (0.5 <= x_t/S < 1): 0.9588, 0.9908, 0.0014, 0.0072. 0.9-1 only: 0.9584, 0.9926, 0.0008, 0.0061. 0.5-0.9 only: 0.9536, 0.9979, 0.0007, 0.0013. Below S but not L: 0.9531, 0.9998. Controls: forced off after N-range inputs 0.5873; after new maxima 0.5621; at all steps 0.2079. Uniform: 0.9459 / agreement 0.9849 (every x_t < S), 0.9459 (L), 0.9455 (0.9-1), 0.9389 (0.5-0.9). 4.34 below-S inputs per sequence, 6.5 % of them fire unit 5, 25.2 % of sequences are touched.
- how checked: h4_intervention.py with a custom forward pass; the mask uses the true prefix runner-up, t >= 2.
- caveats: The intervention uses oracle knowledge of S. My reading that these records are the false positives of an analog comparison (Theta5 sits about 0.2*S below S, and that offset is what makes an N record at x_t = S just cancel the old runner-up's lead) is an interpretation of H3c and H4d, not a tested claim.

### H5a [MEASURED, confidence high]

Controlled inputs (M = 1.5 at p1, S at p2 > p1, constant background -1 or -0.3): the network answers p2 for all 45 pairs for S/M in {0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9}; at S/M = 0.97 it fails for (p1, p2) = (0, 9) and (2, 9) with background -1 and for (0, 9) with background -0.3, answering p1 (unit-4 misfire at the last step). Margins are smallest when the runner-up comes right after the max and when S/M is near 0.1 or 1, and they shrink with the age a2 = 9 - p2 of the runner-up.

- numbers: Margin of p2 over the best other, divided by M, as (min, median) over the 45 pairs, background -1: S/M=0.1: 1.63, 6.20; 0.2: 2.14, 9.28; 0.35: 3.47, 12.46; 0.5: 4.07, 12.58; 0.65: 4.38, 12.83; 0.8: 4.55, 10.80; 0.9: 2.79, 8.69; 0.97: -0.84, 3.72. Background -0.3: 1.13, 4.94; 1.82, 7.30; 3.39, 11.24; 4.00, 12.34; 4.24, 11.66; 4.42, 10.91; 2.83, 9.08; -0.67, 3.75. Diagonal p2 = p1+1 at S/M = 0.5: 4.1-6.2. Median margin/M by a2 = 0..8, background -1: S/M=0.2: 15.5 12.6 9.4 8.9 7.8 6.5 6.2 4.7 2.1; S/M=0.5: 23.3 25.0 13.9 13.5 11.8 10.5 10.7 8.1 4.7; S/M=0.8: 12.6 22.5 14.2 15.4 10.5 9.2 7.5 7.4 6.3; S/M=0.97: 2.7 3.2 3.5 4.1 4.8 4.9 4.4 4.3 3.3. Full (p1, p2) tables for S/M = 0.2, 0.5, 0.8, 0.97 and both backgrounds in h5_controlled_out.txt. 720 deterministic sequences.
- how checked: h5_controlled.py; agrees with the lead's F9 on its grid (S/M <= 0.933). For S/M >= 0.65 the best other position is almost always the max's.
- caveats: Only p1 < p2 (runner-up after the max), one value of M (the net is homogeneous, so only b/M and S/M matter).

### H5b [MIXED, confidence high]

Where the runner-up's record sits at the final step: at a2 = 0 in the direct read of x_9 (+57.2 per unit at position 9), at a2 = 1 in unit 5 read directly (3.40 per unit h5 at position 8), and from a2 = 2 on in the modes h3 (largest at a2 = 2-4), short (largest at a2 >= 3) and h2, with a little from alt. The runner mode holds the largest coordinate (about 6*K*S) but carries no position: its readout is 0.01-0.12 on positions 0-8 and -0.78 on position 9, so it only penalises position 9.

- numbers: Contribution to dlogit[p2] - mean over other non-max positions, per unit S, background -1, S/M = 0.5, a2 = 2..8. h3: 27.5 18.5 17.8 10.3 6.7 5.9 3.9; short: 9.3 23.6 19.5 14.2 10.6 15.2 20.2; h2: 10.6 9.8 5.5 11.4 13.5 9.4 15.3; alt: 3.0 1.0 2.1 2.5 4.3 3.1 -0.4; runner: 8.0 8.7 8.5 4.9 6.3 5.1 3.1; clock: -1.7 -1.1 0.0 -0.2 -1.3 -4.2 -9.0; total: 66.6 62.1 53.7 43.3 40.5 34.9 33.2 (a2 = 0: 147.8, of which echo 144.6; a2 = 1: 81.9, of which pulses 62.2). Record/(K*S) at a2 = 2 for a_ev = 1 (background -0.3, S/M = 0.5): runner 6.26, clock 0.91, h2 0.79, alt 0.18, h3 0.65, short 1.06; for a_ev = 5: runner 3.61, clock 0.07, h2 1.70, alt 1.98, h3 1.72, short 1.07. Logit bump at p2 per unit S: a_ev = 1: 54.9 at a2 = 2 falling to 25.3 at a2 = 8; a_ev = 5: 65.9, 67.6, 69.6 at a2 = 2, 3, 4. EXACT readout amplitude of the short mode 2|U.R_short| by position 0..9: 14.9 9.7 6.5 5.2 5.8 5.1 4.1 3.2 3.2 4.0 (it compensates the decay 0.735 per step).
- how checked: h5_controlled.py (mode-by-mode logit contributions sum to the logits exactly), h5d_record_table.py. Same ordering of modes for S/M = 0.2 and 0.8 and for background -0.3.
- caveats: The 'record' is the difference between the run with S and the run with the background value at p2, so it includes S's effect on the max's own record (the clock column).

### H5c [FIT, confidence medium]

Amplitude law of the record: it is h5 times the fixed write vector of unit 5 (runner 1.723, h2 0.872, alt 1.094, h3 0.935, short 0.599, clock 0.074 per unit h5) plus, when S arrives within 4 steps of the max, the clip corrections |w_j|*(S - theta_j)_+ of the comparators whose threshold is near S at that age. The pulse is about 7.7*S + M when comparator 8 is clipped (max ages 1, 3, 4) and about 18.3*S + 0.3*M when none is (ages 6, 7). The runner record is about 6*K*S at ages 1-4; the clock is reduced by 0.7-0.9*K*S at ages 1-4.

- numbers: Record at a2 = 2 fitted as cS*S + cM*M + cb*b over S/M in {0.1..0.97} and b in {-1,-0.6,-0.3,-0.1} (40 sequences per row), cS by a_ev = 1..7 (R2 in brackets). runner: 5.93 6.70 6.87 5.87 3.83 1.52 1.22 (1.00 1.00 1.00 1.00 1.00 .97 .97). clock: .85@-97 .89@-42 .82@-12 .71@27 .61@62 .39@104 .27@176 (1.00 .99 1.00 .98 .76 .72 .85). h2: .51 .17 .13 .74 1.38 1.54 1.11 (1.00 .72 .97 .99 1.00 1.00 1.00). alt: .17 .43 .14 .56 .71 1.48 1.26. h3: .37 .36 .41 .55 .93 1.47 1.13 (R2 >= .99). short: .99 .80 .50 .16 .39 1.04 1.10. Unit-5 pulse: 7.76S+0.83M-1.10b, 9.56S+2.20M-1.02b, 7.59S+0.60M-1.00b, 7.80S+1.30M-0.77b, 11.66S+2.32M-0.92b, 18.19S+0.52M-1.34b, 18.43S+0.17M-1.43b (R2 >= .995). |cM|, |cb| of the modes <= 0.62.
- how checked: h5b_amplitude_third.py; the pulse slopes match the exact slopes 7.546 and 18.009 of claim H2a; the N(0,1) counterfactual data (claim H3a) give the same runner and clock coefficients (5.7-6.5 and 0.84-0.53).
- caveats: The network is piecewise linear in S, so rows where the set of clipped comparators changes with S/M (a_ev = 2, 5, 6, 7) are averages over pieces (clock R2 0.72-0.85 there).

### H5d [MIXED, confidence high]

With a third value T < S at a third position, the final margin between S's and T's positions is proportional to S - T with almost no offset (tie at S = T within about 0.02*M), and the slope depends on the order of arrival: about 77 logits per unit when T came before the max, 50 when S landed on top of an already recorded T (order M, T, S), 64-66 when T came last. The same slopes hold on N(0,1) samples, where 'answered the third largest' is 2.98 % of all sequences, i.e. 64 % of all errors.

- numbers: Controlled (M = 1.5, S/M in {.3,.5,.7,.9}, T/S in {.2,.5,.7,.8,.9,.95,.98}, 1,440 sequences per T/S and background). margin ~ c1*(S-T) + c2*S + c3*M for T/S >= 0.7: order T,M,S: 76.9, 1.8, 0.2 (R2 .65) [b=-1], 77.5, 1.4, 0.2 [b=-0.3]; M,T,S: 49.8, 0.7, 0.4 (.59), 50.2, 0.5, 0.2; M,S,T: 66.2, -2.7, 1.6 (.81), 63.5, -2.4, 1.8. P(answer = p2), b = -1, at T/S = 0.8 / 0.9 / 0.95 / 0.98: T,M,S 1.00 / .97 / .94 / .83; M,T,S .97 / .92 / .84 / .70; M,S,T 1.00 / .96 / .88 / .65; all wrong answers go to T's position. N(0,1), n = 1.2M (seeds 1032-1034), about 101k per order with T/S >= 0.7: c1 = 77.6 (T,M,S), 54.0 (M,T,S), 63.9 (M,S,T), 72.7 (T,S,M), 51.8 (S,T,M), 67.3 (S,M,T); |c2|, |c3| <= 2.4; R2 .36-.81. Error budget: third-largest position 2.98 %, max's position 0.95 %, other 0.75 %, accuracy 95.32 %. P(answer = p3 | T/S in [0.95,1)) = .135-.258 by order.
- how checked: h5b_amplitude_third.py (controlled) and h5c_sampled_third.py (sampled) give the same slopes independently. Slope by time since the later arrival d (b = -0.3): T,M,S 76-85 for d <= 3 falling to 50 at d = 6; M,T,S 48-57 then 39; M,S,T rising 58 to 74.
- caveats: R2 of the margin regressions is 0.36-0.81; I did not explain why the slope depends on the order.

### H6 [FIT, confidence high]

There is no second clock for the runner-up. Every mode the event writes into (runner, clock, h2, alt, h3, short) carries the record at its own rate: after the record is complete (three steps) each coordinate follows c*lambda_k^a2 with the mode's own eigenvalue. Replacing free per-age vectors by this rotating-record law costs at most 0.023 of R2 on N(0,1) data. The runner-up's age is thus a bump made by the superposition of freely rotating records, read out by a kernel localised at p2.

- numbers: Controlled, 30 conditions per mode (a_ev 1..5, S/M .2/.5/.8, background -1/-0.3), fit over a2 = 2..9-a_ev. Fit with own lambda, R2 median (min): runner 1.000 (.998), clock 1.000 (.991), h2 .998 (.614), alt .998 (.106), h3 .998 (.926), short .999 (.983). Free complex rate, median [min, max]: runner .970 [.950, .975] vs .973; clock .949 @ 42.8 deg [41.8, 48.8] vs .951 @ 42.7; h2 .934 @ 83.0 [77.7, 84.3] vs .937 @ 83.3; alt .911 @ 180 vs .928; h3 .825 @ 124.7 [117.4, 127.2] vs .837 @ 124.8; short .730 @ 43.1 [41.2, 47.9] vs .735 @ 43.0. Exact source split: sources entering later than p2+2 contribute a median of 0.0-0.7 % of sum|dm|^2; stopping at p2+1 leaves 3-1800 % unexplained (the unit-5 pulse and the wave units it clips enter at p2+2). N(0,1), 791,103 sequences with the runner-up after the max (seeds 1019-1022): R2 of c*lambda^a2 on the per-age S-coefficients, a_ev = 1..5: runner 1.000 1.000 1.000 1.000 .999; clock 1.000 .998 .999 1.000 .999; h2 .981 .347 .942 .998 .999; alt .652 .964 .975 .992 .995; h3 .866 .971 .996 .997 1.000; short .998 .993 .989 .947 .991. Data R2 free / constrained, e.g. h3 a_ev=1: .3575/.3349 (largest loss), short a_ev=3: .5038/.5000, runner a_ev=1: .9910/.9910.
- how checked: h6_second_clock.py: two fits (own eigenvalue, free rate) on exact controlled sequences plus an exact attribution of the record to sources by entry step; then the constrained-versus-free comparison on sampled data.
- caveats: The poor fits are the cells where the record in that mode is tiny (h2 at a_ev = 2: 0.02-0.36 K*S; alt at a_ev = 1: 0.2-0.35 K*S). On sampled data (M, S, T+) explain only 3-58 % of the variance of h2/alt/h3/short at the final step: these modes also hold records of other inputs.

### F6ref [FIT, confidence high]

Refinement of the lead's F6: the clock radius is not K*(M - S) in general. The S part is written only by the comparators that the runner-up clips when it arrives (unit 5 writes 0.074 per unit into the clock), so a runner-up that arrives late leaves the clock radius almost unchanged. At the final step the S coefficient of the clock relative to its M coefficient falls from -0.9 (runner-up one step after the max) to -0.28 (seven steps after).

- numbers: Re(bS/bM) of the clock coordinate at t = 9 in fits z/K ~ bM*M + bS*S + bT*T+ per (a_ev, age of max) cell, n about 13k each, R2 .96-.995 (seeds 1029-1031): a_ev = 1: -0.90..-0.94; 2: -0.88..-0.90; 3: -0.80..-0.85; 4: -0.70..-0.74; 5: -0.47..-0.52; 6: -0.38..-0.39; 7: -0.28. Counterfactual N-event data: clock change per unit (x_t - S) = 0.84, 0.83, 0.75, 0.53 K at max ages 1-4 and 0.10-0.17 at ages 5-7. Share of N events that clip no comparator: 65, 91, 83, 55 % at ages 5-8.
- how checked: Three independent routes agree: final-step regression (h1_extra.py), counterfactual N events (h3_analyze.py), controlled inputs (h5b: clock cS = .85 .89 .82 .71 .61 .39 .27 for a_ev = 1..7). F5 and the F6 phases were reproduced (phases 0, 33, 78, 119, 164, 204, 246, 289, 331 deg for a = 1..9; level/(K*M) = 0.92-0.94).

## Contradictions with the lead

1. F6 (clock ~ K*(M-S)*rho(a)*exp(i*phi(a))): the phases are confirmed within 2-8 degrees, but the radius law needs a correction. S enters the clock only through comparators clipped when the runner-up arrives; unit 5's write vector has a clock component of only 0.074. The S coefficient relative to the M coefficient is -0.90..-0.94 when the runner-up arrived 1 step after the max, then -0.88, -0.80..-0.85, -0.70..-0.74, -0.47..-0.52, -0.38, -0.28 for 2..7 steps after (F6 implies -1). So "the S-proportional remainder has size 0.1-0.35" understates it for late runner-ups (up to about 0.7).
2. F4 ("an input that leaves all 13 ordinary units on leaves almost nothing in them one step later"): true only below unit 5's threshold Theta5 of about 0.75*S. 7.4 % of all below-runner-up inputs fire unit 5 one step later, and that pulse writes a full record (runner 1.72, h2 0.87, alt 1.09, h3 0.94, short 0.60 per unit h5). Also every all-on input adds exactly 1.026*x_t to the runner coordinate, which shifts Theta5 by 0.013*x_t per step.
3. F7 (readout blind to the clock): holds for positions >= 3. At positions 0, 1, 2 the clock readout is 2|R| = 1.63, 1.18, 0.75 per unit coordinate, and the clock part of a runner-up record lowers logit[p2] by 4-9 per unit S when p2 is 1 or 2.
4. F8: confirmed (+3.37..+4.16 at own position; earlier positions -0.37..-2.39, a bit wider than "-1..-2"); in addition the unit-5 kernel is +0.85..+1.37 at the NEXT position.
5. F9: confirmed on the lead's grid for p1 < p2; outside it, at S/M = 0.97, the network fails for (p1, p2) = (0, 9) and (2, 9) (background -1) and (0, 9) (background -0.3).
F1, F2, F3, F5 reproduced without disagreement.

## Surprises

- At max ages 5-8 most N events clip no comparator at all (65, 91, 83, 55 %): the whole record is then a single unit-5 pulse, and the clock is not updated.
- Unit 4 misfires in 9.9 % of N-range events (100 % for r > 0.8 at ages 5-8); this alone costs 0.34 pp accuracy on Gaussian and 0.64 pp on uniform inputs.
- Unit 5 is a comparator whose threshold sits at about 0.75-0.8 of the runner-up, not at the runner-up. Its pulse does not vanish at x_t = S (it is about 2..4.4*S + 0.6..1.5*M there), and inputs up to 25 % BELOW the runner-up are recorded. Removing those records improves accuracy by 0.59 pp (uniform 0.76 pp).
- The decision between two sub-max candidates is a graded analog comparison: final margin is about 50-78 logits per unit of (S - T), zero at equality. On N(0,1), 64 % of all errors are "answered the third largest".
- At max ages 1-4 an N event raises the logit at the MAX's position almost as much as at its own (+43..46 against +42..48 per unit x_t at ages 1-3), through the clipped comparators that shrink the clock.
- Records written at max ages 1-4 lose about half of their own-position slope within four steps (56-64 at d = 1 to 20-39 at d = 4); pure unit-5 records made at ages >= 5 keep theirs (53-63).
- The runner mode has the largest coordinate of the record (about 6*K*S) but is invisible to the readout on positions 0-8 (0.01-0.12) and reads -0.78 on position 9: it acts only as a penalty on the last position.
- Unit 5's threshold follows a new runner-up with gain 0.58-0.73 at max ages 1-4 but only 0.36-0.41 at ages 5-6, so after a late runner-up the threshold is too low.
- Unit 9 fires after 99.2 % of new maxima, 0.34 % of N events and 50.5 % of misfires.

## Not done

- findings.md was NOT written: the harness refuses report .md files from subagents ("return findings as text"). I did not work around it. Everything is in this object; every table is in the *_out.txt script outputs in my folder.
- I measured the threshold laws rho_j(a) but did not derive them from the level/clock mode laws, and did not explain why rho_4 sags.
- N-range events at t = 9 (no unit 5 afterwards) are covered only through the controlled inputs (a2 = 0 rows of H5).
- H3 is for events with S > 0; for S <= 0 only the thresholds were checked (they follow the same laws with S+ = 0).
- The knock-on when followers are not quiet is included in the "all followers" numbers of h3_analyze_out.txt but not analysed (slope at position t is then 24-42 instead of 42-48 at ages 1-4).
- Why the S-versus-T slope depends on the order of arrival (50 against 77) is not explained.
- H5/H6 controlled inputs use only p1 < p2, as the task asked.
- Apart from the two interventions (unit 5 after below-runner-up inputs, unit-4 misfires), everything is on Gaussian inputs only.
- The statement that the class-L band is the price of the analog comparison (offset of Theta5 needed for the tie point at x_t = S) is an interpretation, not tested.
- The 240 MB cache h3_event.npz was deleted; h3_event.py rebuilds it in about a minute when h3_analyze.py, h3_analyze2.py or h3_check.py are run.

## Open questions

- Can rho_j(a) be derived from the level law plus the clock phase (theta_j = level part + 2 Re(clock * readout_j)), and is the sag of rho_4 to 0.64 at ages 6-7 the unavoidable by-product of the same clock-based thresholds?
- Why is the S-versus-T margin slope 77 when T precedes the max, 50 when S lands on an already recorded T, and 64-66 when T comes last?
- Is the offset of unit 5's threshold (about 0.2*S below S) required for the tie point of the graded comparison to fall at x_t = S, i.e. would moving Theta5 to S break N records?
- Records written at max ages 1-4 lose half their own-position slope in four steps, pure unit-5 records do not: which part of the comparator-written record decays or rotates out of the readout?
- After a runner-up that arrives at max age >= 5 both the clock radius and Theta5 are under-updated (gain 0.36-0.41, clock coefficient 0.3-0.5): how much of the remaining error rate is caused by this, e.g. extra class-L records and misjudged gaps?
- What exactly happens at the last step: N-range events and misfires at t = 9, where the evidence is the direct read of x_9 (+57.2 per unit) and unit 5 cannot fire?
- The visible modes h2, alt, h3, short are only 3-58 % explained by (M, S, T+) at the final step: what else is stored there (class-L pulses, the max's own unit-5/9 pulses)?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/hcommon.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h0_basics.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h1_explore.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h1_tables.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h1_tables_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h1_extra.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h1_extra_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h2_unit5.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h2_unit5_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h2_formula.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h2_formula_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_kernels_exact.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_kernels_exact_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_event.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_analyze.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_analyze_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_analyze2.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_analyze2_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_check.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_check_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_example.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h3_example_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_theta5.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_theta5_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_theta5_explore.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_theta5_explore_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_intervention.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_intervention_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_Lkernel.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_Lkernel_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_ratchet.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h4_ratchet_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5_controlled.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5_controlled_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5b_amplitude_third.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5c_sampled_third.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5c_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5d_record_table.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h5d_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h6_second_clock.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h6_second_clock_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h_independent_check.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h_independent_check_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h_f6_check.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_H_runnerup_events/h_f6_check_out.txt