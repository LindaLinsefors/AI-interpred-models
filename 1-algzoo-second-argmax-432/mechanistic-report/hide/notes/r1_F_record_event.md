# Round-1 arm F_record_event: structured result

## Summary

Arm F (record anatomy and relabelling), all on N(0,1) inputs, seeds 1001-1095, 13 scripts in the arm folder with raw outputs sNN_out.txt. IMPORTANT: findings.md was NOT written - the harness rejected the Write call ("Subagents should return findings as text, not write report files"), and I did not work around that; the complete results are in this object and the tables are in the sNN_out.txt files.

Main results. (F1) A record at step s is a fixed four-step gate sequence: unit 4 on at s (99.997%) with all four comparators off (95.5%); units 5 and 9 fire at s+1 (100%, 99.7%); unit 0 is off at s+2 and only there (99.7% when the next three inputs are below the old max); resting pattern at s+3. h4(s) = 10.157 (x_s - theta4(s)) exactly; h5(s+1) ~ 12.2 (Mn-M0) + 10.4 M0 - 5.8 S0+ and h9(s+1) ~ 1.56 (Mn-M0) + 1.77 M0 - 0.43 S0+. (F2) Unit 5 is an anti-kick: in the readout-visible modes h2, alt, h3, short, trans its write is 168-189 degrees opposite to unit 4's kick and of similar size; with the clipping of unit 0 at s+2 the new gap ends up only in level (0.92 w4), runner (2.2 w4) and clock (1.04 w4 at 36 deg), with 4-11% of the kick left in h2, alt, h3. The clipped comparators erase the old clock coordinate (to 5-31%) and rewrite the runner mode from S0 to M0. (F3) H_dev is correct about the part created at the record and incomplete about the total: from the weights the comparator-zeroed clock peaks exactly at the old max's position in 35/35 cells (a>=2, j=0..6), 4.0-10.6 logits per unit radius; on data the record creates +28.3 logits at the old max's position (relative to bystander positions) and the clock part of the comparator clipping is +28.0. But the old max also needs a pre-existing +13.5 that exists only because its own unit-5 pulse (one step after it arrived) cancelled the kick's negative trace: removing that one pulse drops P(answer = old max) from 0.967 to 0.020 while the record still creates the same +28 bump. Rotating the clock alone moves the answer as predicted in only 28-40% (76% for large old gaps), all five rotating modes 55-74%; for g=1 the clock coordinate is irrelevant. (F4) Unit 5 off after every record: accuracy 0.953 -> 0.493; unit 9 off: 0.665 with 87.5% of changed answers going to the max; unit 0 forced on at s+2: 0.763 with 66% going to the max. (F5) Nothing erases the old runner-up: its logit falls by 4.2 on average (21%, sd 22, unrelated to its size), a random bystander is perturbed equally (sd 25, mean -0.2); it loses to the old max's +28. (F6) At s=9 no pulse and no unit-0 reset exist, at s=8 the pulses are only read directly; the network still answers (0.943, 0.973) because the comparator clipping acts at the record step itself (un-clipping the comparators at step 9 flips the choice among old positions from the old max, 0.952, to the old runner-up, 0.914) and unit 4 is read directly with U[9,4] = -4.99.

## Claims

### F1.a [MEASURED, confidence high]

Gate sequence of a record at step s (x_s > M0 = old max > 0, s = 2..6). At s: unit 4 on in 99.997% of records, comparators 1, 6, 7, 8 on in only 1.9%, 2.1%, 0.9%, 0.25% (all four off: 95.5%). At s+1: unit 5 on 100.0%, unit 9 on 99.7%. With the next three inputs below M0 (L1): comparators back on at s+1 (98.0-100%), unit 4 off; at s+2 unit 0 is OFF (on in 0.3%) while it is on in 98.9% at s+1 and 99.95% at s+3; pulses 5 and 9 off again at s+2 (on 9.1%, 0.0%). Only wave units 3 and 15 vary inside these classes (unit 3 on .74/.63/.54/.99, unit 15 on .64/.63/.37/.64 at s..s+3).

- numbers: n = 938,993 records (seeds 1001-1003, 3 x 400k sequences); L1 subset 394,221 (42.0%), stricter L2 (next inputs < max(S0,0)) 217,714 (23.2%). Coverage of pattern classes all/L1/L2: C_s (unit 4 on, comparators off) 0.9545/0.9341/0.9395; C_s+1 (5, 9 on; 4 off; comparators and unit 0 on) 0.7052/0.9688/0.9833; C_s+2 (unit 0 off, pulses off, comparators on) 0.4821/0.8837/0.9818; C_s+3 (unit 0 on, pulses off, comparators on) 0.4661/0.9133/0.9868; all four 0.3282/0.7737/0.9038 (by s=2..6 under L1: .820 .825 .794 .741 .693). Commonest full 16-unit patterns inside the classes: 19.0% at s, 45.8% at s+1, 36.3% at s+2, 60.2% at s+3.
- how checked: s02_anatomy.py, direct gate counts; could have failed if unit 0 were also off at s+1 or s+3 (it is on .989 and .9995). Without the L1 condition the step-s row is unchanged and later rows are diluted (unit 0 on at s+2 in 13.0%).
- caveats: 'Low following inputs' was read as below the runner-up after the record (= M0); under the stricter L2 the classes are cleaner. Records at s = 2..6 only (so that s+3 <= 9).

### F1.b [MIXED, confidence high]

h4(s) = w4 (x_s - theta4(s)) with theta4(s) = -W[4].h_(s-1)/w4, w4 = 10.157, exactly whenever unit 4 is on. theta4 sags from the old max towards the old runner-up as the old max ages: theta4 ~ a M0 + b S0+ with (a, b) = (0.987, -0.001), (0.979, -0.004), (0.961, 0.018), (0.876, 0.093), (0.765, 0.185), (0.684, 0.242) for age g = 1..6 of the old max at the record step.

- numbers: Exact formula: max |error| 1.4e-14 over all unit-4-on records. Fits of theta4: R2 = 1.0000, 1.0000, 0.9999, 0.9997, 0.9981, 0.9928 for g = 1..6. h4 ~ c1 Mn + c2 M0 + c3 S0+: c1 = 10.153..10.159 at every g; c2 = -10.02, -9.94, -9.75, -8.89, -7.76, -6.95; c3 = 0.01, 0.04, -0.18, -0.95, -1.88, -2.46; R2 >= 0.9956. Pooled: h4 ~ 10.138 (Mn-M0) + 0.518 M0 - 0.101 S0+ (R2 0.993).
- how checked: s02_anatomy.py: formula compared with simulated activations; regressions without intercept, n per g = 279k, 280k, 179k, 111k, 63k, 27k.

### F1.c [MIXED, confidence high]

On the class 'comparators off at s' (and the unit on at s+1): h5(s+1) = -0.281 x_(s+1) + 1.187 h4(s) + [0.224 h0 - 0.031 h2 - 0.200 h3 - 0.421 h5 + 0.297 h9 + 0.122 h10 + 0.328 h11 + 0.261 h12 - 0.009 h13 + 0.205 h14 + 0.074 h15](s) and h9(s+1) = -0.228 x_(s+1) + 0.146 h4(s) + [-0.032 h0 + 0.004 h2 - 0.044 h3 + 0.043 h5 - 0.553 h9 + 0.033 h10 + 0.053 h11 + 0.016 h12 - 0.026 h13 + 0.107 h14 - 0.010 h15](s), exactly. Both units are normally held off by the comparators (W[5,7] = -0.593, W[5,8] = -0.850, W[9,7] = -5.032, W[9,8] = -1.092) and fire because the comparators are zero at s. Regressions (L1): h5(s+1) ~ 12.23 (Mn-M0) + 10.43 M0 - 5.77 S0+; h9(s+1) ~ 1.56 (Mn-M0) + 1.77 M0 - 0.43 S0+.

- numbers: Exact: max |error| 2e-14; coverage 95.4% (unit 5) / 95.2% (unit 9) of all records, 93.4% under L1. Coefficient of x_s: W[5,4] w4 = 12.057 (plus -0.055 from wave units), W[9,4] w4 = 1.481. Mean (sd) of terms under L1: unit 5: 1.187 h4 = 7.25 (5.78), wave units 8.08 (4.77), pulses 5, 9 at s -1.31, total 14.12 (6.60); unit 9: 0.89, 1.39, -0.02, total 2.32 (0.94). Fit R2: h5 0.939 pooled, 0.987/0.994/0.992/0.991/0.984/0.974 for g = 1..6 with (Mn-M0) coefficient 12.20-12.33, M0 coefficient 9.08, 11.75, 8.83, 9.51, 13.85, 19.70 and S0+ coefficient -3.99, -7.06, -4.51, -5.78, -10.07, -14.42. h9: R2 0.864 pooled (0.905 with x_(s+1), coefficient -0.233), by g 0.924..0.748.
- how checked: s02_anatomy.py. The fitted (Mn-M0) coefficients (12.2, 1.56) agree with the exact W[.,4] w4 (12.06, 1.48) within 2-5%, which need not have happened.
- caveats: The M0 and S0 coefficients depend on g (they come from the wave units and the thresholds); I give fits by g, no closed form.

### F1.d [MIXED, confidence high]

Unit 0 is switched off at s+2 by the two pulses: z0(s+2) = w0 x_(s+2) + W[0].h_(s+1) with mean terms -0.913 h5(s+1) = -12.90, -1.424 h9(s+1) = -3.30, comparators -3.27, wave units +15.48, total -4.01 (sd 2.19). Without both pulse terms z0(s+2) would be positive in 100.00% of L1 records (median 11.5); without the unit-9 term only, in 30.7%; without the unit-5 term only, in 100%. The comparator part of unit 0's input does not contain the current input: sum_j W[0,j] w_j = 0.014.

- numbers: P(unit 0 on) under L1 at s, s+1, s+2, s+3 = 0.713, 0.989, 0.003, 0.9995. Fit: z0(s+2) ~ -4.52 (Mn-M0) - 1.81 M0 + 0.41 S0+ (R2 0.937). At s+3 the same terms are -0.22, 0.00, -4.53, +10.52 (total +5.75).
- how checked: s02_anatomy.py; exact formula checked to 3e-14; intervention F4 (unit 0 forced on at s+2) confirms the clipping matters.

### F1.e [FIT, confidence high]

One step after the record two comparators already compare with the new max and two with the new runner-up: theta7(s+1) ~ 1.018 Mn, theta4(s+1) ~ 0.985 Mn, theta1(s+1) ~ 0.83 Mn + 0.16 M0, but theta6(s+1) ~ 0.98 M0 and theta8(s+1) ~ 1.07 M0.

- numbers: R2 = 0.9999, 0.9999, 0.9968, 0.985, 0.988 (L2 records, n = 217,714, regressors Mn, M0, S0+, no intercept). At s+3: theta7 ~ 1.008 Mn, theta6 ~ 1.050 Mn, theta8 ~ 0.92 M0, theta1 ~ 0.29 Mn + 0.56 M0.
- how checked: s02_anatomy.py section 5.

### F2.a [EXACT, confidence high]

Mode coordinates of the pulse writes c_p,k = LEFT_k . W[ORD,p] (unit 4 = 1 by gauge). Unit 5: level -0.072, runner 1.723, clock 0.074@-137.9deg, h2 0.872@-84.9, alt 1.094, h3 0.935@-53.2, short 0.599@-146.2, trans -0.172, echo -0.673. Unit 9: level -0.126, runner -0.443, clock 0.912@-4.4, h2 0.504@-127.1, alt 1.488, h3 0.781@-39.0, short 0.551@47.3, trans 0.216, echo -0.280. Unit 5 fires one step after unit 4, when the kick has coordinate lambda_k; its write is opposite to that in every readout-visible mode: arg c5 - arg lambda = -168 deg (h2), 180 (alt), -178 (h3), -189 (short), 180 (trans), and nearly absent in level and clock. Unit 5 is an anti-kick.

- numbers: LEFT_k[unit 0] = 0.034 (level), -1.658 (runner), 0.110@15.7 (clock), 0.641@110.4 (h2), -1.209 (alt), 1.056@130.2 (h3), 0.461@4.2 (short), 0.155 (trans). W[ORD,5] is negative on the wave units it touches (unit 0: -0.913, 15: -0.663, 3: -0.518, 14: -0.424); W[ORD,9] has -1.424 on unit 0.
- how checked: s00_basics.py and s07_pulse_writes.py (same numbers from two scripts).

### F2.b [MIXED, confidence high]

Net change of each mode coordinate two steps after a record relative to free flow, N_k = m_k(s+2) - lambda_k^3 m_k(s-1), per unit of the new gap (Mn-M0), in mode-coordinate units (divide by 10.157 for units of w4). Pulses only (lambda_k h4 + c5 h5 + c9 h9 with fitted activations) / pulses plus the clipping of unit 0 at s+2 / measured: level 9.15 / 9.30 / 9.38; runner 30.27 / 22.78 / 22.27; clock 9.75@36.7 / 10.21@35.7 / 10.54@36.3; h2 2.41 / 1.05 / 0.38; alt 6.30 / 0.84 / 1.01; h3 4.18 / 0.69 / 0.66; short 1.63 / 2.69 / 2.75; trans 0.58 / 1.28 / 1.38. So after two steps the new gap is held in level (0.92 w4), runner (2.19 w4) and clock (1.04 w4 at 36 deg, the age-2 phase; the 6 deg lag behind the free 42.75 is unit 9's write 1.42@-4.4), and only 4%, 11%, 8% of unit 4's kick remain in h2, alt, h3 (37% in short).

- numbers: Individual writes per unit (Mn-M0): P4 = 10.24, 9.87, 9.64@42.7, 9.50@83.3, -9.41, 8.49@124.8, 7.45@43.0, 2.34; P5 = -0.89, 21.09, 0.91@-138, 10.67@-84.9, 13.39, 11.44@-53.2, 7.33@-146.2, -2.10; P9 = -0.20, -0.69, 1.42@-4.4, 0.79@-127, 2.32, 1.22@-39, 0.86@47, 0.34 (level, runner, clock, h2, alt, h3, short, trans). Unit-0 clipping term = LEFT_k[0] * 4.52. Activations used: h4 ~ 10.135 (Mn-M0) + 0.527 M0 - 0.106 S0+ (R2 0.988), h5 ~ 12.238, 10.419, -5.774 (0.939), h9 ~ 1.558, 1.766, -0.429 (0.866). At s+3 the measured coefficients are 9.48, 21.52, 9.99@79.5, 0.47, -1.04, 0.52, 2.07, 0.47. n = 263,366 L1 records, seeds 1050, 1051.
- how checked: s07b_pulse_writes_exact.py: an exact per-record identity (m_k(s+2) = free flow + P4 + P5 + P9 + comparator clipping at s + unit-0 clipping at s+2 + other clipping + older pulses + inputs) asserted to 1e-7 for every record; each term regressed on (Mn-M0, M0, S0+). Pulses-only misses the measurement by 8 (runner) and 5 (alt), adding the single unit-0 term closes it to <= 0.7; all other terms contribute <= 1.2 to these coefficients.
- caveats: Regressions pooled over g without intercept; R2 of the measured N_k is 0.86-0.95 for level, runner, clock but 0.07-0.74 for the small modes. The echo mode just mirrors the current input and is omitted.

### F2.c [FIT, confidence high]

The comparators clipped at the record erase the old clock coordinate and rewrite the runner mode. Clock (units of w4): the freely rotated old coordinate lambda^3 z(s-1) is 0.86@128 M0, 0.87@165, 0.84@-152, 0.81@-108 for g = 2..5; the comparator-clipping term is 0.91@-36, 0.86@7, 0.78@48, 0.70@87, i.e. 164, 158, 200, 195 degrees away with similar size, leaving 0.31, 0.19, 0.14, 0.05 M0, while the new clock is 1.03-1.06 (Mn-M0) at 36-38 deg whatever g was. Runner: before the record (0.8..2.0) M0 + (5.5..5.8) S0+; at s+2 it is 2.1 (Mn-M0) + 8.2 M0 + 0.4 S0+ for g <= 3 (7.05 M0 + 1.36 S0+ at g=4, 4.73 M0 + 3.21 S0+ at g=5); the comparator term supplies +4.1..4.7 (M0 - S0+) for g <= 3. Level at s+2 is 0.92-0.93 (Mn-M0) + 0.87-1.03 M0 (R2 1.000).

- numbers: Pooled over g: measured N_runner = 22.27 (Mn-M0) + 69.90 M0 - 53.45 S0+ (R2 0.95), of which comparator clipping 38.16 M0 - 41.27 S0+, unit 5's write 17.95 M0 - 9.95 S0+, older pulses 15.82 M0 - 3.88 S0+. 'After' fits by g have R2 >= 0.99 for level, runner, clock.
- how checked: s07b_pulse_writes_exact.py (by-g tables) and s07_pulse_writes.py (state before/after by g).
- caveats: For g >= 5 the runner rewrite is incomplete. M0 and S0 coefficients of the other rotating modes are g-dependent and were only tabulated, not modelled.

### F3.a [EXACT, confidence high]

H_dev part (i), from the weights only: take level*R_level + 2Re(r e^{i phi(a)} R_clock), zero the four comparators, propagate j resting steps, read with U. For every age a = 2..9 and every j = 0..6 (35 cells) the logit maximum of the clock part over all ten positions is exactly at the old max's position p* = 9 - j - a; its height is 3.97-10.64 per unit r (median 6.94), and 4.58-11.82 above the mean of the other nine positions (median 7.64). The bump therefore stays on the old max's position as it develops. Without zeroing the same state gives 0.12-1.56 at p* (max |logit| 1.6). For a = 1 the peak is at p* - 1 (5.7-7.3 there against -0.2..2.7 at p*).

- numbers: Logit at p* per unit r, rows a = 2..9, columns j = 0..: a2: 5.68 6.94 6.91 5.82 4.95 4.47 4.78; a3: 6.78 6.77 5.61 4.52 3.97 4.31 4.71; a4: 7.65 6.77 6.18 5.71 6.25 7.00; a5: 7.42 7.70 7.72 8.15 9.10; a6: 7.71 8.41 8.96 9.92; a7: 8.86 9.54 10.64; a8: 9.50 10.55; a9: 8.14. Second-best position is 0.4-4.0 lower (the bump is broad). phi(a) re-measured: 0.3, 36.3, 79.3, 123.1, 165.9, 208.6, 251.0, 294.0, 337.8 deg. Level part after zeroing at j=0: U E R_level = (-0.44, -0.50, -1.16, -1.74, -1.34, -0.97, -0.49, 0.09, -0.40, +6.30) per unit level (unzeroed: -0.17..-0.53 and +1.94). For the full free-flow trace of a record kick the zeroing changes the logit at p* by +3.4..+7.5 per unit (median 5.0), peak of the change at p* in 32/35 cells.
- how checked: s04_hdev_weights.py. On real clock coordinates (s10_checks.py, seed 1080, n = 136,233) the same idealised map predicts a mean bump of 32.3 at the old max's position (measured clock term 28.0; by s = 3..9: 17.9, 22.5, 28.8, 34.0, 36.8, 36.1, 31.6 predicted vs 16.0, 13.3, 23.4, 33.6, 33.4, 30.1, 27.4 measured), 6.6 per unit r, with the predicted peak among earlier positions at the old max in 78.5%.
- caveats: Level + clock alone do not reproduce the answer for small gaps (with M0 = 1: argmax = p* in 27/35 cells for S0 = 0.5, 0/35 for S0 = 0.9), because the zeroed level part is not uniform (+59 M0 on position 9 at j = 0, down to -16 M0 elsewhere); the rest of the state matters.

### F3.b [INTERVENTION, confidence high]

H_dev part (ii), zeroing: on sequences where a record at s is the final max and the old max (position p0, age g >= 2) is the true runner-up, setting the clock coordinate of h_(s-1) to zero drops P(answer = p0) from 0.97 to 0.103 (answers: new max s 0.520, old runner-up 0.190, other 0.188) and the centred logit at p0 from 34.7 to 4.6. Zeroing h2, h3, short or alt instead gives 0.835, 0.873, 0.832, 0.946 (0.849, 0.893, 0.852, 0.953 when restricted to cases where the record is still gated as a record); the four together 0.703. For g = 1 zeroing the clock does nothing (0.934 against 0.941): the old max's increment is still in unit 4.

- numbers: n = 387,101 relabel sequences, records at s = 2..9, seeds 1030, 1031; baseline P(p0) by g = 1..7: .941 .969 .970 .976 .975 .966 .953. Clock zeroed by g = 2..7: 0.258, 0.060, 0.054, 0.074, 0.046, 0.049. Record still gated as a record (unit 4 on, comparators off) in 0.970 after zeroing the clock, 0.85-0.87 after zeroing another mode.
- how checked: s05_hdev_rotation.py and s05b_controls.py (conditioning on gating).
- caveats: Intervened states are off the data manifold (24% have a negative entry after zeroing the clock, 55-67% for other modes).

### F3.c [INTERVENTION, confidence high]

H_dev part (ii), rotation: rotating the clock coordinate of h_(s-1) by k resting steps moves the final answer from p0 to p0 - k in 0.282 (k=-2), 0.322 (k=-1), 0.398 (k=+1), 0.369 (k=+2) of valid cases, never in the wrong direction (p0 + k: 0.000-0.001), against 0.0001-0.0016 without intervention. Rotating clock, h2, h3, short and alt together: 0.546, 0.704, 0.715, 0.616. Rotating only h2, h3, short, alt: 0.213 / 0.194 for k = -1 / +1 with P(p0) falling to 0.621 / 0.542. The clock rotation works when the old gap is large: by tercile of (M0 - S0+)/M0, P(answer = p0-1) = 0.115, 0.314, 0.765. So the clock carries the old max's position but not alone; the other rotating modes carry it too.

- numbers: g >= 2, n = 153k-246k per k; remaining answers for clock k=+1: p0 0.404, s 0.035, old runner-up 0.136; k=-1: p0 0.240, s 0.189, old runner-up 0.144. By g (k=+1): 0.001 (g=1), 0.225, 0.400, 0.391, 0.470, 0.546, 0.514 (g = 2..7). Direction control on the subset where both p0-k and p0+k are valid (n = 153,788): clock 0.379 vs 0.000 (k=+1), 0.294 vs 0.001 (k=-1); all five 0.740 vs 0.003, 0.690 vs 0.000.
- how checked: s05_hdev_rotation.py, s05b_controls.py; rotation = multiplication of the mode coordinate by exp(i k arg lambda) (sign flip for alt), then the network runs on the true x_s..x_9.
- caveats: Off-manifold states (32-88% have a negative entry); the record remains gated as a record in 80-90%. Not repeated with a repaired state.

### F3.d [MEASURED, confidence high]

H_dev part (iii): exact decomposition of the final logits on relabel sequences (g >= 2, records at s = 3..9), relative to the mean logit of bystander positions (positions before s other than old max and old runner-up). 'Existed' = logits of the no-record counterfactual (same sequence, x_s replaced by min(S0,0) - 1); 'created' = actual - counterfactual. Old max p0: existed 13.51, created +28.29, actual 41.80. Old runner-up q0: existed 20.08, created -4.19, actual 15.89. New max s: existed -16.81, actual 20.85. Writing the record as threshold excesses, h_s = h_s^lin + sum_j c_j e_j + h4(s) e_4 with c_j = |w_j| (x_s - theta_j(s)) for the clipped comparators, the created +28.29 at p0 splits exactly into: clock part of the comparator excess (= comparator half of the propagated clock mode, removed) +28.02; all other parts of the comparator excess -5.93; unit-4 pulse +0.95; other gate changes +5.25. At q0 the clock part is -0.42.

- numbers: n = 272,233 (g >= 2) of 323,790 sequences with S0 > 0, seeds 1040, 1041; network answers p0 in 0.963. By s = 3..9 at p0: existed 12.6, 13.8, 13.4, 13.4, 13.6, 13.5, 13.7; created 11.0, 17.7, 24.0, 28.8, 31.4, 33.0, 29.4; clock part 16.0, 13.3, 23.4, 33.6, 33.4, 30.1, 27.4. Per sequence, created vs clock part: correlation 0.76 (0.25 at s = 3, 4; 0.62-0.87 for s >= 5), sd of difference 15.4. Of the clock part, +2.9 runs through unit 9 at s+1 and +1.1 through unit 5. Pairwise P(logit p0 > logit q0): 0.103 without the record, 0.983 with it.
- how checked: s06c_decompose_excess.py (exactness asserted per sequence in the companion s06b); two other exact splits in s06_decompose.py and s06b_decompose_paths.py give the same clock-specific term (+34.8 at p0, 0.0 at q0 in the 'old state re-read' split). The counterfactual step is a resting step in 100% of cases, and lowering x_s' by another 1.0 changes the bystander-referenced logits at p0, q0 by 0.35, 0.39 on average.
- caveats: The split between 'other comparator excess' and 'other gate changes' is unstable (they cancel: -87 vs +80 at s = 3) and uninformative at the new max's and later positions; only their sum and the clock part are robust. For g = 1 the clock part is -0.9 while created is +28.4: there the bump comes from comparators 6 and 8 that have not yet moved (see F3.f).

### F3.e [FIT, confidence medium]

The bump created at the old max's position is mainly but not purely proportional to the old gap: created ~ 47.6 (M0-S0) + 11.8 S0 + 2.9 (Mn-M0) (bystander-referenced, g >= 2, pooled). The gap coefficient grows with the record step: about 18-40 for records at s <= 5 and 55-81 at s >= 7. The final margin of the old max over the old runner-up is ~ 52.1 (M0-S0) + 5.5 S0 + 2.7 (Mn-M0); without the record that margin is 0.0 (M0-S0) - 10.1 S0.

- numbers: Pooled R2: created 0.546, clock part 0.401 (44.7, 10.5, 6.2), actual margin 0.410, existed margin 0.111. Within (s, g) cells for the created logit: gap coefficient 17.6 (s=3, g=2) to 81.3 (s=8, g=2), S0 coefficient 3.2 to 18.5, R2 0.36-0.96 (0.83-0.96 for s >= 7 and g = 4, 6). Margin of old max over new max: 4.5-8.1 (Mn-M0) + 5.3-11.6 (M0-S0+) + 4.4-7.7 S0+ for s <= 7 (R2 0.36-0.63), 32.65 (Mn-M0) at s = 8 (R2 0.84), 58.03 (Mn-M0) at s = 9 (R2 0.94).
- how checked: s06c_decompose_excess.py, s09_late_records.py; no-intercept regressions.
- caveats: Low pooled R2: other inputs and gate patterns add large sequence-to-sequence variation.

### F3.f [FIT, confidence high]

The old gap sits in the comparator thresholds: at the record step M0 - theta_j(s) ~ a_j(g) (M0 - S0), with a swinging between 0 (threshold at the old max) and 1 (threshold at the old runner-up) as the old max ages. Unit 1: 0.17, 1.01, 0.70, 0.16, -0.27, -0.37, -0.12 for g = 1..7; unit 6: 1.05, 0.31, -0.05, -0.17, -0.08, 0.17, 0.43; unit 7: -0.02, -0.01, 0.00, 0.13, 0.29, 0.40, 0.39; unit 8: 1.03, 0.69, 0.96, 0.94, 0.57, 0.05, -0.35; unit 4: 0.02, 0.02, 0.04, 0.13, 0.24, 0.32, 0.33. So the excess a record leaves on comparator j is |w_j| [(Mn-M0) + a_j(g)(M0-S0)]; this is the clock seen from the comparators, and it also covers g = 1 (a_6 = 1.05, a_8 = 1.03).

- numbers: R2 of the large entries 0.82-0.98 (e.g. unit 8: 0.98, 0.97, 0.98, 0.97, 0.90 for g = 1..5; unit 1: 0.82, 0.98, 0.92); the S0 coefficient is |b| <= 0.09 everywhere. n = 25k-52k per g (relabel sequences, s = 3..9).
- how checked: s06c_decompose_excess.py threshold table.
- caveats: Small entries (|a| < 0.2) have low R2 and should be read as 'about zero'.

### F3.g [INTERVENTION, confidence high]

The old max's own unit-5 pulse (one step after it arrived) is necessary for its later relabelling, and supplies the part of the logit that exists before the record. Forcing unit 5 off only at step p0+1: P(answer = p0) 0.967 -> 0.020 (answers: new max 0.400, old runner-up 0.322, p0+-1 0.366). The same for unit 9: 0.900; unit 0 made linear at p0+2: 0.970 (no effect). With that one pulse removed the level and clock coordinates just before the record change by a median 5.6% and 14%, but h2, alt, h3, short by 170%, 230%, 150%, 100%; the old max's no-record logit goes from +12.3 above the bystanders to -12.6 below; the record still creates the same bump (+28.3 against +29.1), ending at 15.7 instead of 41.4 (old runner-up 20.8, new max 23.1). By contrast the NEW record's unit-5 pulse is not needed (unit 5 off at s+1 only: 0.927 against 0.973).

- numbers: s05b_controls.py part C: n = 307,719 relabel sequences with g >= 2 (seeds 1030, 1031). s05c_old_pulse.py: n = 107,349 (g >= 3, s = 4..9, seed 1095), P(p0) 0.968 -> 0.019, record still gated as a record in 0.822, h4(s) changed by a median factor 0.964. After the NEW record: unit 9 off at s+1 -> 0.529 (answers s: 0.445); units 5 and 9 both off -> 0.943 (answers s: 0.000); unit 0 linear at s+2 -> 0.587 (answers s: 0.327).
- how checked: Single-step clamps on otherwise unmodified runs; logits compared with and without the record (no-record counterfactual as in F3.d).
- caveats: In 18% of the cases with the old pulse removed the later record is no longer gated as a clean record, so a minor part of the accuracy loss may come from that.

### F3.h [MIXED, confidence medium]

Verdict on H_dev: correct for the part created at the record (clipping the comparators removes the comparator half of the clock; the remainder is read as a bump at the old max's position; +28.0 of +28.3 on average, right position in 35/35 cells from the weights), but only part of the story: (1) the final logit at the old max (41.8 above bystanders) is 13.5 existing + 28.3 created and both are needed (old runner-up ends at 15.9, new max at 20.9); the existing part depends on the old max's own unit-5 anti-kick; (2) h2, h3, short, alt also carry the old max's position (clock-only rotation steers 28-40% of answers, all five modes 55-74%); (3) for g = 1 the clock coordinate plays no role; (4) the bump is ~ 48 (M0-S0) + 12 S0, not purely the gap; (5) the level mode is also re-read when the comparators are clipped (+6.3 per unit level on position 9 at j = 0).

- numbers: See F3.a-F3.g.
- how checked: Synthesis of the weights calculation, the two families of interventions and the exact decomposition.
- caveats: What exactly builds the standing max's +13.5 beyond the unit-5 intervention was not traced path by path.

### F4.a [INTERVENTION, confidence high]

Interventions after every record (x_s > 0 and above all earlier inputs, s = 0 included; 2.24 records per sequence), unmodified accuracy 0.9530. Unit 5 off at s+1: agreement with the network 0.499, accuracy 0.493 (0.089 on sequences whose answer is the old max of the final record, 0.898 on the others); changed answers go to the third largest 37.0%, to +-1 of the unmodified answer 50.5%, to the max 5.5%. Unit 9 off at s+1: agreement 0.676, accuracy 0.665 (0.488 / 0.843); 87.5% of changed answers go to the max. Unit 0 forced on (ReLU removed) at s+2: agreement 0.775, accuracy 0.763 (0.790 / 0.736); 65.9% of changed answers go to the max, 18.1% to the third. Unit 0 on at s+1 is a no-op (agreement 0.9993).

- numbers: n = 800,000, seeds 1060, 1061. Other variants: unit 5 off at s+2: 0.787 / 0.761; unit 5 off at s+1 and s+2: 0.444 / 0.439; unit 9 off at s+2: 0.899 / 0.862; unit 9 off at both: 0.675 / 0.665; units 5 and 9 off at s+1: 0.475 / 0.470 (0.039 on relabel sequences); control, unit 0 off at s+3: 0.623 / 0.613. Of right-to-wrong answers under 'unit 5 off at s+1' (n = 378,717; 92% relabel sequences): third largest 38.6%, true position +1 27.9%, -1 24.3%, max 5.6%. Under 'unit 9 off at s+1': max 90.4%. Under 'unit 0 on at s+2': max 70.7%, third 19.6%. The clamp changes the unit in 99.8% (unit 5, s+1), 97.5% (unit 9, s+1), 81.0% (unit 0, s+2), 0.5% (unit 0, s+1) of clamped steps.
- how checked: s08_interventions.py; the single-record versions in s05b_controls.py separate the two roles: the unit-5 damage comes from the pulse after the OLD max (0.967 -> 0.020), not from the final record's pulse (0.927); unit 9 and the unit-0 clipping counter unit 5 at the CURRENT max's own position (unit 9 off after the final record -> the network answers the max in 44.5%, with 5 and 9 both off in 0.0%).
- caveats: 'Unit 5/9 off at s+2' mostly hits pulses caused by other events at s+1 (unit 5 is on at s+2 in 43%, unit 9 in 19% of clamped steps).

### F5.a [MEASURED, confidence high]

A record does not erase the old runner-up's evidence. Its logit relative to bystanders falls from 20.08 to 15.89 (-4.19, -21%; sd of the change 21.9; lower in 58.6% of sequences); the change is -13% for records at s = 9 and grows to -43% for records at s = 3. It does not scale with the runner-up's value or own logit: change ~ -4.4 (M0-S0) - 3.8 S0 + 0.15 (Mn-M0) with R2 0.006, and regressed on its own no-record logit the slope is +0.055 (erasure would be -1). A random bystander position is perturbed just as strongly (mean -0.18, sd 24.9, against mean -4.06, sd 23.3 for the old runner-up and mean +29.84, sd 22.0 for the old max). The old runner-up loses because the old max gains about +28: with the old max's new logit against the runner-up's unchanged logit the old max already wins in 95.7% (98.3% actual).

- numbers: n = 272,233 (s06c sample, g >= 2); by s = 3..9: no-record logit 12.6, 15.0, 16.4, 18.4, 20.2, 21.9, 23.7; change -5.44, -5.40, -5.07, -4.99, -4.73, -3.40, -3.09. Composition of -4.19: clock part of comparator excess -0.42, other parts -0.42, unit-4 pulse +0.51, other gate changes -3.87. Specificity test: seed 1090, s = 5..9, n = 120,919. Resulting errors: on relabel sequences the network answers the third largest in 1.6-3.9% (14.3% at s = 9 and 5.1% at s <= 7 in the lowest quintile of the old gap).
- how checked: s06c_decompose_excess.py, s10_checks.py, s11_f5_specificity.py (change at the old runner-up compared with the change at one random bystander, both against the remaining bystanders).
- caveats: What does change for the old runner-up is the stored value in the runner mode (from about 5.5 S0 to 8.2 M0 in units of w4), but that mode is nearly position-blind in the readout (0.0-0.1 on positions 0-8, -0.8 on 9).

### F6.a [MIXED, confidence high]

Record at the last step (s = 9): units 5 and 9 never fire for it, unit 0 is never reset, the kick never enters the ordinary units and nothing rotates; the readout reads h_9 directly. The old max's bump is created by the comparator clipping at step 9 itself (existed 13.66, created 29.38, of which clock part of the clipping 27.42). Reading the comparators unclipped (their negative pre-activations) flips the choice among the old positions 0..8 from the old max (0.952) to the old runner-up (0.914), mean margin +22.3 -> -9.9. The max's own position is suppressed by the direct readout of unit 4: U[p,4] - U[9,4] = 4.84..5.74, i.e. 49-58 logits per unit of (x_9 - theta4); margin of old max over position 9 = 33.75 (unit 4) - 1.96 (wave) + 4.20 (pulses left from earlier events) - 0.99 (comparators still on). Setting unit 4 to 0 at step 9: P(old max) 0.529, P(position 9) 0.423.

- numbers: Relabel sequences with the record at s = 9: n = 79,359 (seeds 1070, 1071), P(answer = old max) = 0.943 (by g = 1..7: .911 .925 .929 .957 .979 .973 .952) against 0.963-0.979 for s = 2..8; errors: third largest 3.9% (14.3% in the lowest quintile of the old gap, (M0-S0+)/M0 < 0.11), old max +-1 2.0%, the max itself 1.3% (6.3% in the lowest quintile of the new gap, (Mn-M0)/M0 < 0.13). All four comparators clipped at step 9 in 87.3%. Unit-4 part of the margin ~ 55.4 (Mn-M0) + 9.2 (M0-S0+) + 1.3 S0+ (R2 0.971). Over all ten positions the unclipped readout answers 9 in 0.986 (the unclipped comparator 7 adds +49.9 (x_9 - theta7) to position 9). Accuracy by position of the overall max 0..9: .930 .949 .956 .961 .962 .964 .958 .954 .964 .935.
- how checked: s09_late_records.py, s06c_decompose_excess.py (s = 9 table), s10_checks.py part 3 (seed 1080, n = 37,922).

### F6.b [MIXED, confidence high]

Record at the second-to-last step (s = 8): missing are the unit-0 reset (it would be at step 10) and any propagation of the pulses' writes. Units 5 and 9 fire at step 9 (100%, 97.2%; means 15.3, 2.07) and are read directly: U[8,5] = +3.40 and U[8,9] = -1.66 at the record's own position, the same signs their writes produce later. Unit 4's kick is one step old in the ordinary units with its age-1 kernel -6.63 at its own position (against -3.5..-4.1 at later ages). Removing unit 5 at step 9: P(answer = old max) 0.973 -> 0.940, P(answer = 8) 0.007 -> 0.000; removing unit 9: 0.903, P(8) = 0.053; both: 0.938, P(8) = 0.000. So here the pulses are worth 3-7 points, not the 40+ points they are worth for earlier records. The old max's bump is as large as for earlier records (existed 13.48, created 33.04, clock part of the clipping 30.09).

- numbers: n = 70,174 relabel sequences with the record at s = 8; direct readout at position 8: U[8,5] h5 = +52.0, U[8,9] h9 = -3.4 on average. Margin of old max over position 8 ~ 32.65 (Mn-M0) + 10.32 (M0-S0+) + 3.22 S0+ (R2 0.84); rough expectation from the kernels: (-6.63 + 3.40 * 1.21) * 10.14 = -26 per unit new gap at position 8, plus -2.6 from unit 9.
- how checked: s09_late_records.py (single-step removals at the readout), s06b_decompose_paths.py and s06c_decompose_excess.py (s = 8 tables).
- caveats: The -26 estimate is a back-of-envelope combination of kernels, only roughly matching the fitted 32.65.

### V.a [MIXED, confidence high]

Re-verification of the lead's facts used here: F2-F4 reproduce exactly (A w[ORD] max entry 0.432; W[PULSE,ORD] w[ORD] = (0.01, 18.01, 84.34); cosine of w[ORD] with the echo eigenvector 0.998). F5: level = c * 10.157 * M with c = 0.918-0.938 for max-age 2..9. F6: clock phases agree within 1 degree. F7: clock readout total is 5.8% of comparator part + wave part. F8 free-flow kernels reproduce as linear algebra (unit 4: -3.48..-4.12 at own position for ages 2-8, -6.63 at age 1, -4.99 at age 0).

- numbers: Seed 1010, 400k sequences. Level fit R2 by age 2..9: 0.9963, 0.9967, 0.9965, 0.9957, 0.9932, 0.9897, 0.9873, 0.9866. rho(a) for a = 1..9: 1.005, 1.018, 0.983, 0.944, 0.913, 0.881, 0.853, 0.820, 0.797; S-proportional remainder of the clock 0.14-0.22.
- how checked: s00_basics.py, s03_verify_lead.py.
- caveats: See contradictions_with_lead for the three places where my numbers or reading differ.

## Contradictions with the lead

1. F8 does not describe the running network after a record. The free-flow kernels are correct linear algebra (I reproduce -3.5..-4.1 at the record's own position for unit 4 and +3.4..+4.2 for unit 5), but within two steps units 5 and 9 plus the clipping of unit 0 cancel the kick's visible content: per unit of the new gap only 4% (h2), 11% (alt), 8% (h3), 37% (short) of unit 4's contribution remain, and unit 5's largest write (-0.913 into unit 0) lands on a unit that is clipped at s+2 in 99.7%. Measured centred own-position kernel of the max, d logit_s / d x_s, is -7..-13 per unit input for records at s <= 7 (about -1 per unit of h4, not -3.5..-4.1), and the standing max sits 13.5 logits ABOVE the bystander positions (6.6 below the runner-up), not below them.
2. F6: rho(9). I get the clock radius factor 0.797 at age 9 (falling from 1.02 at age 2), the lead states 0.72. My estimator is a complex least-squares fit of z/10.157 on (M - S+, S+) per age cell; the difference may be the estimator.
3. F5: 'R^2 >= 0.997 within age cells' holds in my re-check only for ages 2-5 (0.9957-0.9967 with M alone, 0.9966-0.9981 with S+ added); at ages 6-9 I get 0.993, 0.990, 0.987, 0.987. The coefficient 0.918-0.938 agrees with 'about 0.93'.
4. F7 holds as stated (all 13 ordinary units on), but note that it stops holding at a record: with the comparators clipped the level mode reads (-0.44, -0.50, -1.16, -1.74, -1.34, -0.97, -0.49, 0.09, -0.40, +6.30) per unit level instead of a uniform -0.3..-0.5 and +1.94.
5. Relative to the hypothesis H_dev handed to me: it is right about the created bump but omits that half of the old max's final logit (13.5 of 41.8, and a swing of 25 logits against the bare-kick baseline) is put there earlier by the old max's own unit-5 pulse; and for g = 1 the clock coordinate is not involved at all.

## Surprises

1. The old max's OWN unit-5 pulse, fired one step after it arrived, decides whether it can later be relabelled: removing that single pulse takes P(answer = old max) from 0.967 to 0.020 although the later record still creates the same +28 bump; removing the NEW record's unit-5 pulse costs only 5 points (0.973 -> 0.927). So unit 5's job at a record is to prepare the current max for a possible later demotion, by cancelling the kick's negative trace in the visible modes.
2. Units 5 and 9 form a balanced pair for the current max's own position: unit 9 off after the final record makes the network answer the max itself in 44.5%, but with units 5 and 9 both off it never does (0.0%) and the old max is still found in 94.3%.
3. Unit 5's write is almost exactly anti-parallel to unit 4's one-step-rotated kick in every readout-visible mode (168, 180, 178, 189, 180 degrees) while leaving level and clock alone; the 6-degree lag of the age-2 clock phase (36.3 instead of 42.75) is unit 9's write.
4. The comparator thresholds are the clock: at the record step M0 - theta_j = a_j(g)(M0 - S0) with a_j swinging between 0 and 1 as the old max ages (unit 8: 1.03, 0.69, 0.96, 0.94, 0.57, 0.05, -0.35 for g = 1..7), and one step after a record thresholds 7 and 4 already sit at the new max while 6 and 8 sit at the new runner-up.
5. At s = 9, simply reading the comparators unclipped flips the choice among the old positions from the old max (95.2%) to the old runner-up (91.4%).
6. A record perturbs the logit of every earlier position by about +-23 (sd) relative to the others, with no systematic sign; only the old max gets a systematic shift (+28 to +30).
7. Accuracy when the record is at s = 8 (0.973) is as good as for mid-sequence records even though the unit-0 reset never happens and the pulses are only read directly: the readout columns U[:,5] and U[:,9] reproduce at age 0 what the writes produce later.
8. The no-record margin between the standing max and the runner-up does not scale with their gap (0.0 (M0-S0) - 10.1 S0, R2 0.11); only after the record does the margin scale with the gap (52 per unit).

## Not done

- findings.md was NOT written: the Write call for c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/findings.md was rejected by the harness ("Subagents should return findings as text, not write report files"), and I did not bypass that through the shell. The full results are in this object; every table is in the sNN_out.txt files listed under files.
- Uniform(-1,1) inputs: everything is on N(0,1).
- Closed forms for the M0 and S0 coefficients of h5(s+1), h9(s+1) and of the mode changes in the rotating modes: only fits by g (they depend on the clock phase through the wave units and thresholds).
- g = 1 (two records in a row): thresholds and interventions are covered, but there is no clean mode-level account (the mode split has cancelling terms of +-200 there, because the old max's increment is still in unit 4).
- The rotation counterfactuals are off-manifold (19-88% of intervened states have a negative entry) and were not repeated with a repaired state; thresholds were not rotated directly.
- Not explained: why the created bump per unit old gap is smaller for early records (18-40 at s <= 5 against 55-81 at s >= 7), and why the per-sequence agreement between created bump and clock term is poor at s = 3, 4 (correlation 0.25 against 0.62-0.87 for s >= 5).
- Insertions (a new runner-up below the max), where unit 5 also fires, were not studied; nor the contribution of wave units 3 and 15, the only units that vary inside the pattern classes.
- The paths that build the standing max's existing +13.5 logit were not traced beyond the unit-5 intervention.
- F1 covers records at s = 2..6 only (s+3 <= 9); F3-F6 cover s up to 9.
- No figures (as instructed).

## Open questions

- Which paths turn the old max's unit-5 pulse into the +25-logit difference at its own position (which modes, read through which units), and is the standing max's logit really 'runner-up-sized' by design (existed at old max ~ 5.0 (M0-S0) + 16.3 S0, at old runner-up ~ 5.0 (M0-S0) + 26.4 S0)?
- Why is the created bump per unit old gap two to four times smaller for early records (s <= 5) than for late ones, when the from-weights conversion per unit clock radius falls only from 7.7 (j = 0) to 4.8 (j = 6)?
- Is the non-specific +-23-logit perturbation that a record adds to every earlier position the main source of 'answers the third largest' errors (1.6-3.9% of relabel sequences, 14% for small old gaps at s = 9)?
- For g = 1, is the complete mechanism just 'comparators 6 and 8 have not yet been raised' (a_6 = 1.05, a_8 = 1.03), and does it explain the lower accuracy of g = 1 at s = 6, 7, 9 (0.923, 0.907, 0.911)?
- Do insertions (new runner-up below the max) use the same unit-5 anti-kick without the unit-0 clipping, i.e. is unit 0's clipping the only thing that distinguishes a record from an insertion downstream of unit 5?
- How much of the error rate at records comes from the sag of theta4 below the old max (a_4 = 0.13, 0.24, 0.32, 0.33 of the old gap at g = 4..7), which makes the network treat inputs slightly below an old max as records?
- The rotation tests say h2, h3, short and alt carry the old max's position alongside the clock: is that the same content the old max's unit-5 pulse wrote, and does it explain why clock-only rotation succeeds only for large old gaps (76.5% in the top tercile against 11.5% in the bottom)?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/fcommon.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s00_basics.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s01_explore_gates.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s02_anatomy.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s02_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s03_verify_lead.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s03_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s03_clock.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s04_hdev_weights.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s04_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05_hdev_rotation.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05a_kernels.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05a_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05b_controls.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05c_old_pulse.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s05c_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06_decompose.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06b_decompose_paths.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06c_decompose_excess.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06c_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s06c_terms.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s07_pulse_writes.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s07_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s07b_pulse_writes_exact.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s07b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s08_interventions.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s08_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s09_late_records.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s09_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s10_checks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s10_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s11_f5_specificity.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_F_record_event/s11_out.txt