# Round-1 arm C_clock: structured result

## Summary

Arm C is complete for C1-C6, but findings.md was NOT written: the harness rejected the Write ("Subagents should return findings as text, not write report files") and I did not work around it. The findings are in this object; every table is in the *_out.txt files next to the scripts in the arm folder.

Notation: K = w_4 = 10.157; z = LEFT_clock . h[ORD]; zhat = z/K; M, S+ = max(S,0), T+ = three largest so far; a1, a2 = ages of max and runner-up; lam = 0.95088 e^{i 42.748 deg}; "position" of a threshold u = (theta - S+)/(M - S+) (0 = at the runner-up, 1 = at the max).

1. Phase (C1, C6). F6's phase law holds. The max's own trace in the clock has phase phi(a1) = 0.3, 36.5, 79.5, 123.3, 166.0, 208.6, 251.0, 294.0, 337.7 deg for a1 = 1..9 (F6 within 1 deg), the same in every (a1,a2) cell to 3.3 deg (a1 <= 5) or 6.8 deg (a1 = 6..9). Nearest-phase decoding of a1 from arg z is 84.0% correct (97.4% within one step; 99% when (M-S+)/M >= 0.5, 28% when < 0.05).

2. Radius (C1, C3). F6's radius law z = K (M - S) rho e^{i phi} is only a first approximation. alpha is not a function of a1 alone (|alpha| 0.68-0.90 across a2 at a1 = 9), |beta| is 0.04-0.64 and |gamma| 0.01-0.45 per cell. (M - S+) explains 94.9% of sum|zhat|^2, M alone 82.7%, free alpha M + beta S+ 97.1% with beta/alpha = -0.75..-0.88, per-cell 3-term 99.2%. Exact traces show why: the runner-up's coefficient is -(0.63..1.0) times the max's when the runner-up is the previous max, -(0.77..0.93) when it arrived 1-3 steps after the max, -0.64..-0.69 at 4 steps, and about 0 at 5-7 steps (no trace at all in 53-66% of states at the arrival step).

3. Record (C2). A record does not erase the clock. Zeroing the four comparators keeps s_c = LEFT_clock[wave].RIGHT_clock[wave] = 0.494 at -68 deg of the clock coordinate. Removing the pre-record clock changes the post-record clock by 0.341 e^{i 28 deg} z(s-1) (intervention). The kick is h4(s)/K = M_new - theta_4, and theta_4 lies below M_old by up to 43% of (M_old - S_old). The 37 deg first step is lam + c5 W[5,4] + c9 W[9,4] = 0.958 at 37.0 deg.

4. New runner-up (C3). An input v between S and M clips the comparators whose threshold is below v. At max-age 1-4 some threshold sits at S, and the radius moves 63-83% of the ideal amount at once. At max-age 5-8 the lowest threshold is 0.2-0.5 of the way to M, 42-66% of such inputs clip nothing and fire no unit 4, and the clock keeps radius (M - S_old). The answer is still right (91-95% on data in exactly those cases, 99-100% on controlled inputs); unit 5 fires on the next step in 98.5-99.2% of them.

5. Thresholds (C4). theta_j = A_j M + B_j S+ + (M - S+) C_j cos(phi(a) + delta_j) with delta = -154.4, -79.3, -3.9, +125.1, -6.6 deg and C/rho = 0.715, 0.397, 0.234, 0.726, 0.160 for units 1, 6, 7, 8, 4; A = 0.68-0.83, B = 0.13-0.27. Level + runner + clock explain 96.4-99.6% of the variance of the thresholds and 90-99.5% of theta_j - S+.

6. Causal tests (C5). After a phase shift by k steps the five thresholds match the natural ones of age a+k (mean deviation 0.030 vs 0.331). Answers change in 24-38% of runs. When the max of the edited step is later dethroned and ends as the runner-up, 55-94% of answers change and 34-50% (k = +-1) move by exactly -k positions; shifting h2, h3, short and alt instead changes only 31-48%. Editing h_9 alone changes 5.6-16% of answers (not zero). Radius x0: thresholds collapse to 0.55-0.83, 71% of new runner-ups clip nothing, 30% of answers change.

7. Unrequested but relevant: only 32.1% of states with a1 >= 1 have all 13 ordinary units on; wave units 0, 11, 10 are off at max-age 2, 5, 6.

## Claims

### C1-phase [FIT, confidence high]

The clock phase is a function of the age of the max only. In per-cell fits zhat ~ alpha (M-S+) + beta S+ + gamma T+, the phase of alpha is phi(a1) = 0.3, 36.5, 79.5, 123.3, 166.0, 208.6, 251.0, 294.0, 337.7 deg for a1 = 1..9 (F6's values within 1 deg). Across a2 cells at fixed a1 it varies by at most 3.5 deg for a1 <= 6 and by 5.4, 10.0, 10.7 deg for a1 = 7, 8, 9.

- numbers: Increments phi(a+1)-phi(a): 36.2, 43.0, 43.7, 42.7, 42.6, 42.4, 43.0, 43.7 deg (free rotation 42.75). Exact-trace medians of the max's own trace zeta_M: phase within 3.3 deg of phi(a1) in all cells with a1 <= 5 and within 6.8 deg for a1 = 6..9; interquartile range of the phase 0-2 deg in undisturbed cells.
- how checked: c1_cells.py, 1.25M sequences (seeds 1100-1104), complex least squares without intercept per (a1,a2) cell, states with M > 0. Coefficients from seeds 1100-1101 and 1102-1104 differ by at most 0.012. Independent check with exact Jacobian traces on 200k sequences (c1b_traces.py, seeds 1700-1707; identity zhat = sum_p zeta_p x_p verified to 1e-9). On quiet steps the measured one-step factor z(t)/z(t-1) is 0.93-0.96 at 41.9-45.0 deg for a1 = 3..9 (c2c_free.py).
- caveats: Cells with a2 = 0 or 1 at a1 >= 7 lead by 5-11 deg.

### C1-radius-cells [FIT, confidence high]

alpha is not a function of a1 only, and beta, gamma are not small. At fixed a1 the modulus of alpha ranges 0.98-1.04 (a1=1), 0.85-0.96 (a1=5), 0.76-0.93 (a1=7), 0.68-0.90 (a1=9). Per cell |beta| is 0.039-0.637 and |gamma| 0.010-0.451.

- numbers: With a1-only coefficients: rho = |alpha| = 1.000, 1.015, 0.981, 0.943, 0.912, 0.881, 0.853, 0.821, 0.796; |beta| = 0.22-0.33; |gamma| = 0.12-0.21 (this is F6's 0.1-0.35). Along the diagonals a2 = a1-1 the modulus decays by 0.95 per step: 0.992, 0.933, 0.888, 0.846, 0.800, 0.759, 0.720, 0.683 for cells (2,1)..(9,8). Largest |beta|: cells (8,1) 0.637, (7,0) 0.630, (6,0) 0.617, (9,1) 0.609, (1,9) 0.599. F6's rho = 0.72 at a = 9 matches only cells (9,7), (9,8).
- how checked: c1_cells.py, same data as C1-phase; split-half stability 0.012; per-cell model fitted on seeds 1100-1101 explains 0.9916 on seeds 1102-1104.
- caveats: zhat is exactly linear in the inputs for a fixed gate pattern, so a good linear fit in (M,S,T) is expected; see C1-traces for the sharper test.

### C1-explained [FIT, confidence high]

(M - S+) describes the clock far better than M alone, and a free S coefficient is better still: the clock holds about M - 0.75..0.88 S+, not M - S+.

- numbers: Fraction of sum|zhat|^2 explained, all states with a1 >= 1: coefficients depending on a1 only: alpha(M-S+) 0.949, alpha M + beta S+ 0.971, alpha M 0.827, 3-term 0.974; one coefficient set per (a1,a2) cell: 0.956, 0.988, 0.851, 0.992. Per-cell 3-term fit: 0.957-0.998 in every cell with a1 >= 1. Free fit beta/alpha (a1-only) = 0.861, 0.856, 0.877, 0.866, 0.821, 0.798, 0.770, 0.753, 0.766 for a1 = 1..9, phases within 9 deg of 180. Radius only: |zhat| ~ rho (M-S+) has R2 0.977 (a1=1) falling to 0.871 (a1=9); |zhat| ~ a M + b S+ has b/a = -0.83..-0.71, R2 0.992..0.934; |zhat| ~ a M has R2 0.851-0.864.
- how checked: c1_cells.py, 12.5M states; the M-only model could have won and did not; out-of-sample explained fraction 0.9916.
- caveats: States with M <= 0 (10.0%) are excluded; they carry 0.003% of sum|zhat|^2.

### C1-traces [MIXED, confidence high]

Exact traces: the max's trace in the clock is rho e^{i phi(a1)}, but the runner-up's trace is not minus that. Ratio r = -zeta_S/zeta_M (F6 says 1): 0.63-1.00 when the runner-up is the previous max; 0.77-0.93 when it arrived 1-3 steps after the max; 0.64-0.69 at 4 steps; about 0 at 5-7 steps.

- numbers: Median modulus of zeta_M by a1 = 1..9: 1.002, 1.014, 0.971, 0.939, 0.894, 0.842, 0.810, 0.777, 0.757. Median Re r / share of states with no trace (|r| < 0.1), by max-age at the runner-up's arrival and steps since arrival (0, 1, 2): age 1: 0.77/0.00, 0.78/0.00, 0.90/0.00; age 2: 0.84, 0.84, 0.87; age 3: 0.91/0.07, 0.93/0.06, 0.86; age 4: 0.69/0.10, 0.69/0.08, 0.64; age 5: 0.01/0.53, 0.31/0.50, 0.47/0.00; age 6: 0.00/0.66, 0.00/0.63, -0.05/0.00; age 7: -0.01/0.60, -0.01/0.56, 0.08/0.01; age 8: 0.52/0.44, -0.01/0.40. Old max as runner-up by lag a2-a1 = 1..8: 0.76-0.80, 0.63-0.80, 0.74-0.85, 0.85-0.89, 0.87-0.98, 0.92-1.00, 0.91-0.93, 0.74. Share of states with |r - 1| < 0.25: 94-98% only in cells a2 = a1-1 and (1,0); 0.3-85% elsewhere. A negative runner-up contributes rms 0.009 against rms |zhat| 0.867.
- how checked: c1b_traces.py: Jacobian traces from common.traces on 200k sequences (seeds 1700-1707), per state exact (identity checked to 1e-9), component-wise medians per cell. Agrees with the regression version in c3_runnerup.py (update fraction by age and delay) and with the per-cell beta - alpha of c1_cells.py.
- caveats: Medians are used because 1-23% of states per cell have |zeta_M| > 2 (unit 9 firing, gain 6.0). Tangential part Im r is +0.27..0.29 for arrival at max-age 2 and +0.21..0.25 at max-age 3 in the first two steps.

### C1-phase-error [MEASURED, confidence high]

The phase of z lags phi(a1), and the lag grows as the gap M - S+ closes.

- numbers: Error arg z - phi(a1), percentiles 5/25/50/75/95 in deg and share within 21.4 deg. By a1: 1: -15/-1/0/+2/+26, 0.901; 2: -45/-12/-3/0/+9, 0.829; 3: -51/-15/-3/+1/+11, 0.793; 4: -40/-14/-4/0/+9, 0.826; 5: -28/-10/-3/0/+8, 0.891; 6: -28/-10/-3/+1/+6, 0.896; 7: -30/-11/-3/0/+6, 0.886; 8: -35/-13/-5/-1/+4, 0.851; 9: -46/-17/-7/-2/+3, 0.798. By (M-S+)/M: 0-0.05 (5.6% of states): -86/-47/-26/-4/+70, 0.310; 0.05-0.10: -64/-30/-16/0/+38, 0.515; 0.10-0.25: -47/-20/-9/+2/+23, 0.709; 0.25-0.50: -27/-10/-4/+1/+11, 0.907; 0.50-0.75: -12/-5/-2/+1/+5, 0.996; 0.75-1: -4/-1/0/0/+3, 0.991.
- how checked: c1_cells.py, n = 1.25M sequences, states with M > 0 and a1 >= 1, reference phases from the 3-term a1-only fit.
- caveats: With the one-coefficient (M-S+) fit as reference the medians shift by +2 to +5 deg.

### C2-survival-exact [EXACT, confidence high]

Zeroing the four comparators does not remove the clock coordinate: it keeps s_c = LEFT_clock[wave] . RIGHT_clock[wave] = 0.494 at -68.0 deg of it, although 96.6% of |LEFT_clock|^2 is on the comparators. It also leaks 0.138 (at -138 deg) of the level coordinate, 0.465 of short and 0.681 of trans into the clock.

- numbers: s_n for level, runner, clock, h2, alt, h3, short, trans, echo: 0.138@-138.2, 0.011@36.2, 0.494@-68.0, 0.112@-176.6, 0.015@153.9, 0.067@-163.1, 0.465@126.1, 0.681@-0.4, 0.003. Conjugate part for the clock 0.035@145.7. LEFT_clock on wave units 0,2,3,10,11,12,13,14,15: 0.110@15.7, 0.046@23.6, 0.036@-90.1, 0.139@9.2, 0.109@-93.0, 0.119@-136.4, 0.044@8.6, 0.217@142.7, 0.116@-141.9. One free step later: lam^2 s_c = 0.447@17.5.
- how checked: c2_restart.py part A; check s_clock + LEFT_clock[comp].RIGHT_clock[comp] = 1 to 1e-6. The measured regression coefficient of zhat(s) on zhat(s-1) at records is 0.386@-11.8 against the free-flow value lam s_c = 0.470@-25.2.
- caveats: Applies when all four comparators are zero, which holds at 94.2% of records.

### C2-record-stats [MEASURED, confidence high]

At a record step unit 4 fires almost always, the comparators are usually but not always all zero, and one step later the clock is the kick plus a remainder of about a quarter of its size.

- numbers: 454,626 records (s = 1..8, M_old > 0). Unit 4 on: 99.995% (1.23% of non-record steps). All four comparators off: 94.2%; a comparator stays on in 1.5, 1.3, 3.1, 10.7, 17.3, 22.9, 23.6, 25.9% of records when the old max was 0..7 steps old. rms |zhat| at the record step 0.210 (0.467 one step before; kick rms 0.817). Exact terms of z(s+1) relative to rms h4(s), quiet next step (n = 334,427): lam z(s) 0.268, c5 h5(s) 0.049, c9 h9(s) 0.063, input 0.012, clips <= 0.032. Remainder r = zhat(s+1) - h4(s)/K: sum|r|^2 / sum h4^2 = 0.070 (quiet next step), 0.119 (all). Units 5, 9 are on at the record step itself in 48% and 30% of records.
- how checked: c2_restart.py, 400k sequences, seed 1200; the exact update z(t) = lam z(t-1) + sum_p c_p h_p(t-1) + e x_t + clips reproduces z to 1.8e-13.

### C2-survival-intervention [INTERVENTION, confidence high]

About a third of the old clock survives a record. Setting the clock coordinate to 0 one step before a record changes the clock one step after it by 0.341 e^{i 28 deg} z(s-1) + 0.072 e^{i 170 deg} conj z(s-1).

- numbers: n = 455,855 records, explained 0.902; rms change / rms zhat(s-1) = 0.356. Two paths: through unit 4's threshold exactly 2 Re(zhat(s-1) W[4,ORD].RIGHT_clock) with W[4,ORD].RIGHT_clock = 0.0801@173.4 (regression returns 0.0801@173.4, explained 0.9999); everything else 0.409 e^{i 21.8 deg} z(s-1) (explained 0.934; exact free-flow value lam^2 s_c = 0.447@17.5). Regression version: r ~ 0.385@34.0 zhat(s-1) + 0.070 conj + 0.196@-33.5 M_old + 0.232@153.9 S_old, explained 0.841 (quiet next step); zhat(s-1) alone 0.338@19.1, explained 0.649.
- how checked: c2b_survival.py, seed 1201: clock removed at s-1 with other modes and pulses unchanged, steps s and s+1 rerun with the real inputs; unit 4 still fires in 99.97%. Fixed-gate exact Jacobian of the non-unit-4 path averaged over records: 0.35-0.43 at +22..+30 deg for old-max ages 1..7.
- caveats: 54% of the edited states have a negative ordinary unit. For records right after another record (old-max age 0) the pre-record clock is tiny and the unweighted mean Jacobian differs (0.208@92).

### C2-kick-sag [FIT, confidence high]

The kick is h4(s)/K = M_new - theta_4(s), and theta_4 is below the old max by an amount that depends on the old max's age, so the kick overshoots M_new - M_old.

- numbers: h4/K against M_new - M_old: LS slope 1.031, R2 0.9927 with slope fixed at 1. Median (M_old - theta_4)/(M_old - S_old+) by age of the old max 0..7: 0.012, 0.028, 0.051, 0.149, 0.282, 0.406, 0.431, 0.330. Budget at a1 = 1 (a2 >= 2, n = 396,226): alpha = 1.013@0.5 of which the kick 0.996; beta = 0.251@3.7 of which carried old content 0.188@16.6 and kick 0.062; gamma = 0.232@177.9 of which carried 0.205.
- how checked: c2_restart.py and c2b_survival.py part (d); thresholds from theta_4 = -W[4].h/w_4. Consistent with C4's position of theta_4 by age (0.985, 0.966, 0.943, 0.845, 0.714, 0.598, 0.578, 0.675).

### C2-first-step [MIXED, confidence high]

The first step after the kick advances the clock by 37 deg instead of 42.75 because units 5 and 9 fire one step after the record: z(s+2)/z(s+1) = lam + c5 W[5,4] + c9 W[9,4] = 0.951@42.7 + 0.088@-137.9 + 0.133@-4.4 = 0.958@37.0. Unit 5's term points against the clock and only shortens it; unit 9's term lags it by 47 deg and turns it back.

- numbers: c5 = LEFT_clock.W[ORD,5] = 0.074@-137.9; c9 = LEFT_clock.W[ORD,9] = 0.912@-4.4; W[5,4] = 1.187, W[9,4] = 0.146. Real impulse x = (1,0,..): phases 0.1, 36.4, 79.6, 123.4 deg (first increment 36.3; with background -1: 35.7); moduli 0.965, 0.964, 0.908, 0.862. On N(0,1) data the (M-S+) coefficient at a1 = 2 (a2 >= 3, n = 246,947) is 0.968@43.2 (carried) + 0.090@-137.9 (unit 5) + 0.144@-4.4 (unit 9) + 0.059@22.4 (wave clips) = 1.037@36.5. Activations: h5(s+1)/K ~ 1.214 (M-S+) + 1.053 S+ - 0.456 T+ (explained 0.980), h9(s+1)/K ~ 0.157 (M-S+) + 0.185 S+ - 0.057 T+ (0.978); relative to the kick, h5/h4 median 2.05 (IQR 1.55-3.27), h9/h4 median 0.32 (IQR 0.22-0.56); both on in >= 99.8%.
- how checked: c2_restart.py (impulse and data), c2b_survival.py part (d): each exact term regressed on (M-S+, S+, T+), the parts sum to the total coefficient.
- caveats: The phase of the full z advances less than the phase of alpha: median 28.8 deg (IQR 21.9-32.7) from a1 = 1 to 2, because units 5 and 9 are also driven by S and T.

### C3-formula [EXACT, confidence high]

A comparator j is clipped iff the input v exceeds theta_j, and then adds c_j (v - theta_j) to zhat with c_j = LEFT_clock[j] |w_j| / K = 0.800@-123.9 (unit 1), 1.308@159.2 (unit 6), 1.377@28.7 (unit 7), 0.670@-44.6 (unit 8). The four gain directions leave a 130 deg gap between unit 7 and unit 6.

- numbers: sum_j c_j = 0.019 (clipping all four cancels the direct input term e/K = 0.0106@-51.5). LEFT[j] RIGHT[j] for the comparators: 0.301@39.0, 0.273@37.2, 0.169@-18.0, 0.256@37.7, sum 0.935@29.4. For an input at max-age 1 the clipped pair (6, 8) gives (c_6 + c_8)(v - S) = 0.746@-179.5 (v - S), i.e. 75% of the ideal radial update.
- how checked: c3_runnerup.py: on 4,080 controlled sequences the identity zhat_with_v - zhat_without = sum over clipped c_j (v - theta_j) + e (v - b)/K + wave-clip difference holds to 1.7e-15; the wave-clip part is 1.8% (rms) of the comparator part.

### C3-what-is-clipped [MEASURED, confidence high]

Which comparators a new runner-up clips depends on the age of the max, and at max-age 5-8 it usually clips none. Share of inputs with S+ < x_t < M that neither clip a comparator nor fire unit 4, by max-age 1..8: 0.0, 0.3, 6.7, 9.6, 53.0, 65.5, 58.2, 42.4%.

- numbers: 494,805 events on 400k sequences (seed 1300): unit 4 fires in 6.98%, no comparator clipped in 12.6%, all four in 0.01%. Unit 4 fires in 1.1, 2.2, 3.4, 10.1, 19.1, 28.0, 29.4, 23.0% by max-age 1..8 (seed 1301, n = 455,633). Typical clipped sets for u = (v-S)/(M-S) around 0.5: age 1: units 6, 8; age 2: 1, 8; age 3: 1, 8; age 4: 8; age 5: 8 in 73%; age 6: 7 in 23%; age 7: 6 in 42%, 7 in 26%; age 8: 6 in 87%; age 9: 1 and 6. Deterministic example: M = 1.5 at position 0, background -1, v = 0.6 at position p: clipped sets {6,8}, {1,8}, {1,8}, {8} for p = 1..4 and none for p = 5..8 (thresholds at p = 6: 2.00, 1.24, 0.95, 1.42; theta_4 = 1.06).
- how checked: c3_runnerup.py, c3b_false_records.py, c3c_examples.py; gates read from pre-activations. Consistent with the measured lowest threshold position by age (C4-positions).
- caveats: Max-age here is the age at the step the input arrives (state age + 1).

### C3-update-size [FIT, confidence high]

The clock after a new runner-up v is K (M - v) rho e^{i phi} only approximately and only when the max is young. Immediate radial update as a fraction of the ideal -alpha(a)(v - S_old), by max-age 1..9: 0.83, 0.72, 0.75, 0.63, 0.14, 0.04, 0.00, 0.20, 0.46; tangential part +0.01, +0.50, +0.35, +0.09, -0.09, -0.03, +0.02, +0.25, +0.63.

- numbers: After d = 0, 1, 2, 3 steps (update fraction, complex): age 1: 0.78+0.04i, 0.79+0.11i, 0.91+0.01i, 0.91+0.02i; age 2: 0.80+0.51i, 0.81+0.51i, 0.89+0.24i, 0.87+0.20i; age 3: 0.84+0.33i, 0.87+0.26i, 0.80+0.10i, 0.81+0.09i; age 4: 0.67+0.14i, 0.67, 0.62, 0.62; age 5: 0.21, 0.24, 0.13, 0.14; age 6: 0.05, 0.06, 0.01, 0.02; age 7: 0.07, 0.08, 0.12; age 8: 0.38+0.35i, 0.42+0.37i. Explained fraction of the immediate-update fits 0.989, 0.971, 0.943, 0.951, 0.595, 0.480, 0.064, 0.468, 0.693.
- how checked: c3_runnerup.py part B on N(0,1) events without unit-4 firing; independent confirmation without regression from the exact traces (C1-traces).
- caveats: The second-step change (+0.12 at age 1) is the only systematic delayed correction; at ages 5-7 a delayed trace appears after two steps but its radial part is 0.0-0.5.

### C3-controlled [INTERVENTION, confidence high]

Controlled inputs confirm it: with M = 1.5 at p1, an optional earlier runner-up, and an input v at p1 + a, the clock ratio R = z_with_v/z_without_v matches F6's prediction R = 1 - u only for a = 1; for a = 5..8 and u <= 0.5 the clock does not change at all.

- numbers: 4,080 sequences. u_eff - u (u_eff = 1 - |R|) / phase shift in deg at the event step for u = 0.1, 0.3, 0.5, 0.7: a=1: +0.02/+1, -0.05/0, -0.12/0, -0.19/-1; a=2: -0.01/-3, -0.09/-7, -0.22/-21, -0.35/-35; a=3: -0.08/-2, -0.18/-8, -0.20/-15, -0.23/-26; a=4: -0.08, -0.15, -0.23, -0.30 (phase -1..-10); a=5: -0.09, -0.29, -0.45, -0.53; a=6: -0.10, -0.30, -0.50, -0.59; a=7: -0.11, -0.31, -0.51, -0.65; a=8: -0.11, -0.32, -0.50, -0.56 (phases within 2 deg for u <= 0.5). Two steps later: a=1: +0.09, +0.04, 0.00, -0.06; a=2: +0.01, -0.04, -0.12, -0.19; a=5: -0.12, -0.35, -0.53, -0.61; a=6: -0.11, -0.32, -0.54, -0.65. Over u in [0.1,0.9]: mean |u_eff - u| = 0.230, 0.209, 0.178, 0.162 at delays 0..3; LS slope of u_eff on u 0.53, 0.58, 0.68, 0.71; mean |phase shift| 10.6, 11.0, 6.5, 6.0 deg (max 61). Example M = 1.5 at 0, v = 0.6 at p: |z| ratio 0.671, 0.712, 0.803, 0.762 (p = 1..4) and 0.990, 0.999, 1.007, 1.015 (p = 5..8); target 0.600.
- how checked: c3_runnerup.py part C and c3c_examples.py: backgrounds -1 and -0.3 (same result: 0.177 vs 0.179), five earlier-runner-up configurations, p1 = 0..7. Inputs below the old runner-up leave the clock alone (| |R| - 1 | = 0.006, phase 0.4 deg). Unit 4 fired in 492 sequences, all at u >= 0.7; those are excluded.
- caveats: M = 1.5 only (the network is scale-equivariant).

### C3-answer-unaffected [MEASURED, confidence high]

The answer does not need the radius update. When the final runner-up arrived 5-8 steps after the final max and its arrival clipped nothing and fired no unit 4, the network is still right in 91-95% of cases, and unit 5 is on at the next step in 98.5-99.2% of them (unit 9 in 0.0-0.1%).

- numbers: 400k sequences, seed 1302, overall accuracy 0.9536. Lag 5: such arrivals are 47.0% of cases, accuracy 0.946; lag 6: 62.4%, 0.950; lag 7: 56.7%, 0.911; lag 8: 43.3%, 0.931. Accuracy when a comparator was clipped and unit 4 silent: 0.970, 0.970, 0.975, 0.975, 0.992, 0.993, 0.945, 0.967 for lags 1..8; when unit 4 fired: 0.908, 0.888, 0.868, 0.915, 0.914, 0.936, 0.880, 0.927, and 0.701 at lag 9. Controlled sequences of C3-controlled: answer = position of v in 99-100% for u = 0.1..0.9.
- how checked: c3d_accuracy.py (gates at the runner-up's arrival step, final answers) and the final answers in c3_runnerup.py.
- caveats: At lags 1-2 the rare arrivals that clip nothing (0.9%) are answered correctly in only 6-11%; these are near-ties with the old runner-up.

### C3-false-records [FIT, confidence medium]

7% of new-runner-up inputs fire unit 4 because its threshold has sagged; these false records do not reset the clock.

- numbers: Median position of theta_4 in these events 0.73; kick v - theta_4 has median 0.11 (v - S_old); comparators all clipped in 0-1% of them (1.6-3.0 clipped on average). Regression zhat(t+1) ~ A lam^2 zhat(t-1) + B h4(t)/K: A = 0.28, 0.39, 0.70, 0.83, 0.91, 0.79 at max-age 3..8, against A = 0.38 after a true record (explained 0.91-0.95).
- how checked: c3b_false_records.py, 455,633 events, seed 1301.
- caveats: B is not 1 (0.5-2.0) because clipping terms correlate with the kick; only A is interpreted.

### C4-formula [MIXED, confidence high]

theta_j = A_j M + B_j S+ + (M - S+) C_j cos(phi(a) + delta_j) + other modes, with delta_j = -154.4, -79.3, -3.9, +125.1, -6.6 deg and C_j/rho = 0.715, 0.397, 0.234, 0.726, 0.160 for units 1, 6, 7, 8, 4. A_j = 0.681-0.696, 0.776-0.790, 0.803-0.822, 0.764-0.784, 0.810-0.833 and B_j = 0.218-0.266, 0.159-0.200, 0.150-0.192, 0.133-0.171, 0.135-0.175 over state ages 1..8. Units 7 and 4 have nearly the same phase and level weight, so their thresholds move together.

- numbers: T_jn = -K W[j,ORD].RIGHT_n / w_j: level 0.631, 0.749, 0.782, 0.748, 0.796; runner 0.050, 0.040, 0.038, 0.034, 0.035; clock 0.358@-154.4, 0.198@-79.3, 0.117@-3.9, 0.363@125.1, 0.080@-6.6; short 0.145, 0.067, 0.032, 0.208, 0.039; trans 0.206, -0.085, 0.005, -0.369, 0.003. Laws by state age: level/K = 1.009 M - 0.037 S+ (a=1), 0.929..0.964 M - 0.03..0.07 S+ (a = 2..8), R2 >= 0.998; runner/K = 1.7..2.0 M + 4.9..6.0 S+ for a >= 2 (R2 0.954-0.993); clock one-coefficient law rho = 1.058, 1.094, 1.060, 1.031, 1.035, 1.027, 1.026, 1.017 at 0.7, 33.5, 75.8, 120.0, 163.6, 206.3, 248.8, 291.0 deg. A_j + B_j = 0.91-1.01. Formula vs measured median positions (S+ > 0, (M-S+)/M > 0.3) agree to 0.00-0.07 for ages 1..8 except unit 6 at age 1 (0.87 vs 0.66).
- how checked: c4_thresholds.py, 400k sequences, seed 1400. Exact mode decomposition of the thresholds verified to 5e-15; the formula's R2 on raw thresholds without refitting: 0.989, 0.972, 0.995, 0.981, 0.997.
- caveats: Not valid at the record step itself (state age 0), where unit 4's own activation sets the thresholds of units 1, 7, 4 (pulse terms +0.84, +1.02, +0.99) and the level coordinate still holds the old max (0.925 S+).

### C4-positions [MEASURED, confidence high]

Measured threshold positions by age of the max in the state that sets them. A comparator threshold sits at the runner-up only for state ages 0-3 and 8; for ages 4-7 the lowest one is 0.2-0.5 of the way up to the max. Unit 4's threshold sits at 0.94-0.99 for ages 0-2 and sags to 0.58-0.71 for ages 4-7.

- numbers: Median positions (units 1, 6, 7, 8, 4) by state age a: 0: 0.84, -0.04, 1.02, -0.03, 0.985; 1: -0.05, 0.65, 1.02, 0.31, 0.966; 2: 0.24, 1.04, 1.01, 0.05, 0.943; 3: 0.81, 1.20, 0.87, 0.05, 0.845; 4: 1.32, 1.08, 0.67, 0.42, 0.714; 5: 1.46, 0.79, 0.53, 0.99, 0.598; 6: 1.13, 0.45, 0.52, 1.46, 0.578; 7: 0.54, 0.23, 0.63, 1.52, 0.675; 8: -0.06, 0.29, 0.83, 1.14, 0.812. Lowest comparator (median, which unit): -0.04 (6), -0.05 (1), +0.03 (8), +0.04 (8), +0.39 (8), +0.53 (7), +0.44 (6), +0.22 (6), -0.06 (1). Share of states whose lowest comparator threshold is above 0.25: 0.6, 0.0, 1.5, 7.6, 77.5, 80.5, 69.9, 46.2, 7.8%. Share of states with theta_4 position < 0.9: 6, 16, 21, 99, 100, 100, 100, 98, 85%.
- how checked: c4_thresholds.py, all states with M > 0 and t <= 8 (854k states at age 0 down to 45k at age 8). The pattern predicts exactly the clipped sets of C3-what-is-clipped, measured independently.
- caveats: Positions have wide interquartile ranges at ages 6-8 (e.g. unit 6 at age 7: -0.16..+0.38) because states with S <= 0 and small gaps are included.

### C4-variance [MEASURED, confidence high]

Level + runner + clock explain 96.4-99.6% of the variance of the five thresholds and 90-99.5% of their height above the runner-up; the remainder is almost entirely the short and trans modes.

- numbers: Centred R2 with exact mode coordinates, no refitting, units 1, 6, 7, 8, 4, states with a >= 1. Raw theta: level 0.353, 0.601, 0.735, 0.499, 0.793; level + runner 0.602, 0.876, 0.942, 0.515, 0.970; level + runner + clock 0.983, 0.975, 0.995, 0.964, 0.996; + short + trans 0.992, 0.998, 0.999, 0.997, 0.999. theta - S+: 0.967, 0.961, 0.993, 0.901, 0.995; + short + trans 0.985, 0.997, 0.999, 0.993, 0.999. Position u_j (states with M > 0.5 and (M-S+)/M > 0.3, n = 1.37M): 0.955, 0.772, 0.924, 0.896, 0.899, median absolute position error 0.051, 0.033, 0.024, 0.062, 0.015; + short + trans 0.978, 0.980, 0.996, 0.991, 0.992. Covariance share of the clock in var(theta - S+): 0.62, 0.28, 0.18, 0.56, 0.11; level 0.28, 0.49, 0.56, 0.48, 0.59.
- how checked: c4_thresholds.py; per-state contributions sum to the threshold exactly (5e-15).
- caveats: Within a single age the three-mode R2 can be negative where a threshold sits near S (unit 8 at state ages 2-3, unit 1 at age 1): there the other modes and pulses move the threshold by 0.13-0.21 of rms (M - S+).

### C5-thresholds [INTERVENTION, confidence high]

After multiplying the clock coordinate by e^{i k 42.75 deg} at a step where the max has age a, the comparator and unit-4 thresholds for the next input equal the natural thresholds of age a + k, and stay so on later steps.

- numbers: Mean over the five units of |median position after the edit - natural median|: 0.030 against age a + k, 0.331 against age a, over all 26 cells (a, k) with k in -2..2 and 1 <= a + k <= 8; closer to a + k in 26 of 26. One and two steps later: 0.035 vs 0.333 and 0.051 vs 0.338. Example a = 4, k = -2: edited 0.24, 1.05, 1.00, 0.05, 0.95; natural at age 2: 0.24, 1.04, 1.01, 0.05, 0.94. Largest deviations at a = 1 (0.06-0.09).
- how checked: c5_causal.py, 200k sequences, seed 1500, all states with M > 0 and a1 >= 1 at t = 1..8; k = 0 reproduces every baseline answer. With the edited state clipped at zero (c5e_clipped.py): 0.034 vs 0.325, 26 of 26.
- caveats: 36-79% of phase-shifted states contain a negative ordinary unit; the clipped variant gives the same result.

### C5-answers [INTERVENTION, confidence high]

A phase shift by k steps changes the final answer in 27.2% (k = -1), 24.1% (+1), 35.3% (-2), 38.0% (+2) of runs. If a new max arrives after the edit, 34-57% of answers change and 58-60% of those (k = +-1) move by exactly -k positions; with nothing arriving later, 16-32% change and 55-72% of those become the max's own position.

- numbers: n = 1,173,867 edited runs per k; accuracy 0.9555 before, 0.721, 0.750, 0.646, 0.616 after. Of the changed answers: new - old = +1 in 49% for k = -1; -1 in 38% for k = +1; +2 in 30% for k = -2; -2 in 15% for k = +2; new answer = the max in 24-39%. By what happens after the edit (k = -1 / +1): nothing above S+ arrives (39% of runs): changed 0.159 / 0.209; new runner-up arrives (22%): 0.083 / 0.121; new max arrives (39%): 0.495 / 0.342. Share changed by max-age at the edit falls from 0.29-0.47 (age 1) to 0.13-0.20 (age 8).
- how checked: c5_causal.py; 91-93% of the changed answers were correct before the edit. Clipped-state variant (c5e_clipped.py): 0.278, 0.243, 0.353, 0.400.
- caveats: Direction: a clock made older by k steps moves the answer k positions earlier.

### C5-carrier [INTERVENTION, confidence medium]

The clock phase is the main carrier of the position of the current max once it is dethroned. In runs where the max of the edited step ends as the runner-up, shifting the clock alone by one step changes 82.7% (k = -1) or 55.3% (k = +1) of answers and moves 50.3% or 33.7% of all such answers by exactly -k; shifting h2, h3, short and alt instead changes 30.6% or 40.4% and leaves 68.6% or 57.7% correct. For a runner-up that is already present the roles are reversed.

- numbers: n = 264,539 runs (max of step t ends as runner-up). Clock only, k = -2 / +2: changed 0.941 / 0.895, moved by -k 0.405 / 0.221, still correct 0.041 / 0.086. Other four modes, k = -2 / +2: changed 0.481 / 0.471, moved by -k 0.043 / 0.021. All oscillating modes, k = -1 / +1: changed 0.715 / 0.677, moved by -k 0.611 / 0.410. Runs where nothing above S+ arrives later (n = 454,241), k = -1 / +1: clock only changed 0.159 / 0.210, moved by -k 0.042 / 0.031; other four modes changed 0.686 / 0.710, moved by -k 0.464 / 0.452.
- how checked: c5b_exposure.py, 200k sequences, seed 1501; each complex mode coordinate multiplied by e^{i k angle_n}, alt by (-1)^k. Clipped variant for the clock-only shift: changed 0.822 / 0.510, moved by -k 0.499 / 0.292.
- caveats: Asymmetric in k and far from 100%: the clock is the largest single carrier, not the only one. The mechanism by which the surviving pre-record phase is read out was not derived.

### C5-final-state [INTERVENTION, confidence high]

Editing the clock in the final state h_9 only does change answers: 5.6% (k = -1), 5.9% (+1), 10.5% (-2), 11.1% (+2), 16.0% (half turn), 7.75% (radius x0), 3.7% (x0.5), 8.8% (x2). Removing the clock from h_9 lowers accuracy from 95.70% to 90.76%.

- numbers: n = 179,837 final states (M > 0, max-age >= 1). Accuracy after: 0.925, 0.924, 0.882, 0.877, 0.830, 0.908, 0.941, 0.895. Largest logit change, median: 4.5 (k = +-1), 8.5 (+-2), 12.9 (half turn), 6.4 (x0); median baseline margin between the top two logits 8.3. Rotating only the comparators' share of the clock changes 57.3-58.9% (k = +-1) and 75.2-77.2% (+-2). Readout vector 2|U[q,ORD].RIGHT_clock| for positions 0..9: 1.63, 1.18, 0.75, 0.47, 0.38, 0.40, 0.37, 0.47, 0.54, 0.62. Changed answers go to rank 3 (45-51%), rank >= 4 (22-39%), true runner-up (11-18%), the max (4-10%).
- how checked: c5_causal.py part (ii); k = 0 changes nothing. Clipped-state variant gives larger numbers (6.7%, 11.9%, 12.5%, 21.8%, 8.5%, 4.1%, 26.8%) because clipping also alters other modes.
- caveats: F7's cancellation removes about nine tenths of the effect; the rest is several logits because |z| is 5-10.

### C5-radius [INTERVENTION, confidence high]

Scaling the clock radius changes the thresholds' swing and the events that follow. Radius x0 puts all five thresholds at 0.55-0.83 for every age; then 70.7% of new runner-ups clip nothing at the next step (baseline 18.5%). Radius x2 doubles the swing; then 16.3% of inputs below the runner-up clip a comparator (baseline 0.9%) and 4.2% of true records fail to fire unit 4.

- numbers: Answers changed: 30.3% (x0), 19.9% (x0.5), 29.5% (x2); accuracy 0.692, 0.791, 0.697 (baseline 0.9555). Events at the next step (baseline / x0 / x0.5 / x2): record fires unit 4: 0.9999 / 0.9998 / 0.9998 / 0.958; all four comparators clipped at a record: 0.914 / 0.979 / 0.984 / 0.734; new runner-up fires unit 4: 0.098 / 0.149 / 0.121 / 0.101; new runner-up clips nothing: 0.185 / 0.707 / 0.516 / 0.074; low input clips something: 0.009 / 0.002 / 0.002 / 0.163. Threshold medians at state age 5: natural 1.46, 0.79, 0.53, 0.99, 0.60; x0: 0.62, 0.72, 0.79, 0.76, 0.77; x2: 2.33, 0.84, 0.27, 1.23, 0.42. For x0 and x0.5 the new answer is the max in 51% and 59% of changed cases; for x2 it is an input of rank >= 4 in 43%.
- how checked: c5_causal.py, n = 1,173,867 runs per edit; clipped variant 0.307, 0.200, 0.268.
- caveats: 95% of x2-edited states contain a negative ordinary unit.

### C6-decoding [MEASURED, confidence high]

The age of the max can be read from the angle of z alone with 84.0% accuracy (nearest of the nine reference phases), 85.0% if ages above t are excluded, and 97.4% within one step. Accuracy depends on the relative gap: 27.7%, 46.0%, 67.9%, 89.1%, 99.0%, 99.0% for (M-S+)/M in 0-0.05, 0.05-0.10, 0.10-0.25, 0.25-0.50, 0.50-0.75, 0.75-1.

- numbers: 6,764,934 states (M > 0, a1 >= 1) from 1M sequences, seeds 1600-1603. By a1 = 1..9: 0.858, 0.802, 0.795, 0.831, 0.891, 0.896, 0.885, 0.853, 0.798. Errors are mostly one step too young: 9-17% of states of each age >= 2 are decoded as a1 - 1; age 1 is decoded as 9 in 5.1% and as 2 in 6.2%. By step t = 1..9: 0.965, 0.887, 0.842, 0.826, 0.833, 0.841, 0.842, 0.821, 0.787. Always guessing age 1: 0.257.
- how checked: c6_decode.py; reference phases fitted on different seeds (1100-1104). Free-rotation references 36.5 + (a-2) 42.748 give 0.844; reversed phases give 0.228.
- caveats: Angle only; the modulus and the other units are not used.

### X-wave-gates [MEASURED, confidence medium]

The premise 'all 13 ordinary units are on' fails in two thirds of states. Only 32.1% of states with max-age >= 1 have all 13 on (38.3% of quiet steps); wave unit 0 is off at max-age 2 (on in 0.00), unit 11 at age 5 (0.08), unit 10 at age 6 (0.05), and units 3, 12, 15 are off in 20-44% of states at ages 1-6.

- numbers: Share with all 13 on by max-age 1..9: 0.529, 0.000, 0.440, 0.304, 0.047, 0.030, 0.698, 0.821, 0.777. All nine wave units on: 40.5%; all four comparators on: 83.0%. Effect on the clock coordinate itself is small: wave clips are 0.3-6.6% of rms |z| per step. Effect on the readout is large: the exact linear contribution of the clock component of h_t to the final logit at the max's own position, in runs where nothing arrives after t, is +3.9..+6.5 with free-flow gates and -9..-38 with the real gates; gating units 5 and 9 off changes it by under 0.5 and comparators are clipped in under 1% of those steps. After a clock-only shift by one step the gates of units 10 and 11 follow only halfway (unit 11 on-share 0.47 and 0.52 at the two candidate ages); with h2, h3, short, alt shifted too they follow (0.05, 0.00); unit 0's gate does not follow.
- how checked: c5d_wavegates.py (200k sequences, seed 1503) and c5c_diag.py (100k, seed 1502). The free response A^(a-1) W[ORD,4] alone does not make units 0, 10, 11 negative, so the gates come from units 5, 9 and the real history.
- caveats: Measured, not explained: I did not derive which modes set each gate nor how the gates enter the answer. The linear attribution is exact for the actual run but is not a counterfactual.

## Contradictions with the lead

F6 (radius). The phase part is confirmed, the radius part only partly. The runner-up's coefficient in the clock is not -1 times the max's: it is -(0.63..1.0) when the runner-up is the previous max, and about 0 when the runner-up arrives 5-7 steps after the max, because then no comparator threshold lies below it (53-66% of such inputs clip nothing and fire no unit 4). alpha depends on a2 as well as a1 (|alpha| 0.68-0.90 at a1 = 9). The S-proportional remainder is 0.04-0.64 per cell; F6's 0.1-0.35 is its a1-only average (0.22-0.33). F6's rho = 0.72 at a = 9 holds only for cells (9,7), (9,8); pooled it is 0.80.

F7 (readout blind to the clock). The two numbers are confirmed (cancellation ratio 0.0575; 96.6% of |LEFT_clock|^2 on comparators). But editing the clock in h_9 alone changes 5.6-16% of answers and removing it costs 4.9 points of accuracy, and the premise 'all 13 ordinary units on' holds in only 32.1% of states with max-age >= 1.

Implicit assumption in C2 that a record wipes the clock: it does not. Zeroing the comparators keeps 0.494 at -68 deg of the clock coordinate (exact), and 0.34 e^{i 28 deg} of the pre-record clock is in the post-record clock (intervention). The comparators are all zero at only 94.2% of records (74-83% when the old max is 4-7 steps old).

Unit 4's threshold is not the max: its position between S and M is 0.94-0.99 only for state ages 0-2 and 0.58-0.71 for ages 4-7, so 7.0% of new-runner-up inputs fire unit 4 (19-29% at max-age 5-8). This does not contradict F2 as stated.

No disagreement with F1, F2, F3, F4, F5 or F9 (all re-verified; F9 is 90/90 in all six settings; F4's cosine is -0.998 in the modes.py gauge, a sign convention only). F8 was not tested.

## Surprises

1. A new runner-up arriving 5-8 steps after the max usually leaves the clock untouched, yet the network still answers its position (91-95% on data in those cases; unit 5 fires on the next step in 98.5-99.2% of them). So the clock radius is not what stores the new runner-up.

2. The old clock survives a record, and that surviving phase is used: shifting the clock before a record moves the position later reported for the dethroned max by exactly -k in 34-50% of such runs, more than shifting all other oscillating modes together.

3. The lowest comparator threshold sits at the runner-up only for state ages 0-3 and 8. The four gain directions (-124, 159, 29, -45 deg) leave a 130 deg gap and unit 7's swing is small (0.23), so for ages 4-7 the lowest threshold is 0.2-0.5 of the way up to the max.

4. Units 7 and 4 have nearly identical threshold laws (clock phases -3.9 and -6.6 deg, level weights 0.78 and 0.80).

5. Wave units act as age detectors: unit 0 is off at max-age 2, unit 11 at age 5, unit 10 at age 6. Only 32% of states have all 13 ordinary units on.

6. At the record step the level coordinate still holds the old max (0.925 S+); it switches to the new max one step later (1.009 M - 0.037 S+).

7. The runner coordinate is large in this gauge: about 1.7-2.0 M + 4.9-6.0 S+ times K for state age >= 2.

8. False records cost accuracy: when the runner-up's arrival fired unit 4, accuracy is 87-94% (70% when it arrived 9 steps after the max), against 95-99% when it clipped a comparator.

9. With no later record, a wrong or missing clock mostly makes the network answer the max itself (55-72% of changed answers for phase shifts, 62-76% for radius x0 and x0.5).

## Not done

findings.md was not written. The Write call was rejected by the harness ("Subagents should return findings as text, not write report files") and I did not route around it; the full findings are in this object and the tables are in the *_out.txt files of the arm folder.

Not worked out: how a runner-up that leaves no trace in the clock is stored (only that unit 5 fires and the answer stays right); which modes set the wave-unit gates and how those gates enter the readout; why a wrong phase makes the network answer the max when no record follows; how the surviving pre-record phase is turned into the dethroned max's position (I measured the effect by intervention, not the path). The regression coefficients on M_old and S_old in C2 mix the level leak, the short and trans modes and unit 4's sag and were not separated.

Not tested: F8 (pulse logit kernels); anything on Unif(-1,1) inputs; controlled inputs with M other than 1.5. No figures were made, as instructed.

## Open questions

- Where is a runner-up stored when it arrives 5-8 steps after the max and clips no comparator? Unit 5 fires on the next step in 98.5-99.2% of those cases; which modes does that pulse write to, and is that the general runner-up marker also at young ages?
- How is the pre-record clock phase read out after the record? Zeroing the comparators maps the clock into clock (0.494 at -68 deg), short and other modes; does the resulting pattern mark the old max's position directly in the logits?
- What do the age-locked wave-unit gates do (unit 0 off at age 2, unit 11 at age 5, unit 10 at age 6)? With real gates the clock component contributes -9 to -38 logits at the max's own position instead of +4 to +6.5 in free flow; is this the mechanism that suppresses the max's position?
- Is unit 4's threshold sag (position 0.58-0.71 at state ages 4-7) functional or a defect? False records are 7% of runner-up events and are answered less accurately (87-94%, 70% at lag 9).
- Should the mode framework be restated around the actual gate patterns, given that all 13 ordinary units are on in only 32% of states?
- Why is the answer shift after a clock-only phase shift asymmetric (k = -1: 83% changed, 50% moved by +1; k = +1: 55% changed, 34% moved by -1)?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/cl.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c0_verify.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c0_verify_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1_cells.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1_cells_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1_laws.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1b_traces.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1b_traces_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1c_diag.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c1c_diag_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c2_restart.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c2_restart_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c2b_survival.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c2b_survival_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c2c_free.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c2c_free_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3_runnerup.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3_runnerup_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3b_false_records.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3b_false_records_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3c_examples.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3c_examples_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3d_accuracy.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c3d_accuracy_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c4_thresholds.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c4_thresholds_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c4_thresholds.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5_causal.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5_causal_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5b_exposure.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5b_exposure_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5c_diag.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5c_diag_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5d_wavegates.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5d_wavegates_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5e_clipped.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c5e_clipped_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c6_decode.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_C_clock/c6_decode_out.txt