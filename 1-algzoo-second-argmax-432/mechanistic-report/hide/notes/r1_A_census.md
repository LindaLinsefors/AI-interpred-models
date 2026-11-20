# Round-1 arm A_census: structured result

## Summary

Arm A census, on 2^20 N(0,1) sequences (4 x 2^18, seeds 1000-1003; accuracy 0.9529). All scripts and their printed tables are in r1_A_census/ (a00..a10 *.py with matching *_out.txt). IMPORTANT: findings.md was NOT written. The harness refused the write ('subagents should return findings as text, not write report files') and I did not route around it, so this object is the write-up and the *_out.txt files hold every full table.

Notation: M, S = largest and second largest of x_0..x_(t-1); M+ = max(M,0), S+ = max(S,0). Positive-part classes: R*: x_t > M+; N*: S+ < x_t < M; L*: x_t < S+. s_R, s_N = steps since the last R* / N* event. Age a = age of the max in h_(t-1).

A1. Gate-on frequency over all steps: comparators 0.685-0.770, unit 4 0.232, unit 9 0.262, unit 5 0.409, wave units 0.728 (unit 15) to 0.904 (unit 13). At t = 0 the gate is exactly [w_i x_0 > 0].

A2. Gates follow the positive-part classes, not the literal ones: a negative new maximum never fires unit 4 (0 of 202,996) and leaves all comparators on (0.977). R* gives unit 4 on (0.9992 at t >= 1) and all four comparators off (0.937 when x_t > M > 0). L* gives all comparators on (> 0.999 for x_t < S/2, 0.650 for x_t in (0.9S, S)). N* gives mixed patterns that depend on the age of the max; at ages 4-7 all four comparators stay on in 43-66% of N* events. History: unit 5 is on at t after R* or N* at t-1 (0.992-1.000; 0.03-0.07 after L*), unit 9 after R* at t-1 (0.906-1.000; 0.002-0.044 after N*), unit 0 is off at t iff R* at t-2 and not R* at t-1 (0.989-1.000 against at most 0.027).

A3. Best rule per unit (accuracy): unit 4 on iff R*(t) 0.9912, iff 7 off 0.9958; comparators and unit 4 by an age-dependent threshold x_t vs S+ + u_j(a)(M - S+): 0.991-0.997 out of sample; unit 9 on(t) iff 4 on(t-1) 0.9897 (with '8 off(t-1)' 0.9916); unit 5 on(t) iff x_(t-1) > 0.75 S+ 0.9832; unit 0 off(t) iff R*(t-2) and not R*(t-1) 0.9959 (candidate '7 and 9 on at t-1': 0.9739 for t >= 1, 0.9932 for t >= 2); units 2 and 13 on whenever a positive input occurred earlier (0.9997, 0.9966); unit 11 off iff s_R = 5 (0.9869), unit 10 off iff s_R = 6 (0.9801); unit 12 off only at s_R = 4 (29.6% of those steps); units 3 and 15 have no simple rule (best lookup 0.840, 0.828 against base rates 0.785, 0.739).

A4. The exact resting pattern holds on 10.3% of all steps (12.9% of t >= 2, 29.0% at t = 9). 1377 distinct patterns occur; 10 cover 50%, 75 cover 90%. 'Pulses off and comparators on' holds on 37.0% (58.5% at t = 9). Top 15 patterns are in claim C16.

A5. F2 holds exactly (0 mismatches in 52,428,800 evaluations). Units 5 and 9 also obey a threshold, on the previous input (0 exceptions in 18,874,368). Thresholds are not fixed points between S and M: the median position u = (theta - S)/(M - S) runs from -0.07 to 1.50 with the age of the max, as a sinusoid at the clock frequency 42.75 deg for ages >= 2. theta_j = alpha(a) M + beta(a) S fits with within-cell R^2 0.994-0.999. Unit 4's threshold sags to 0.61 at ages 5-6, so 3.74% of unit-4 firings are not new maxima (12.5% at t = 8, 9). Unit 5's threshold is about 0.72 S.

A6. Resampling one input below S/2 changes the answer in 0.55% of cases, the margin by a median 0.13 (median margin 8.5); all inputs below S/2 at once (3.9 per sequence): 1.32%. The exact resting kernel is 0.78-0.98 logits per unit input at the input's own position and the one before. Inputs in (0.7S, S): 5.5% (10.8% for the last input); class N inputs: 14.5% (5.5% with the true answer unchanged).

A7. Wave units are clipped at specific ages of the max (unit 0 at s_R = 2: 99.9%; 11 at 5: 91.9%; 10 at 6: 95.0%; 3 and 15 broadly). Median clip 2.0-2.6 for units 0, 3, 15, at most 0.9 for the rest. The wave clip exceeds 10% of the state norm on 20.7% of steps. Removing one unit's ReLU costs 19.0 pp (unit 0), 7.3 pp (15), 1.0 pp (3), 0.9 pp (10), 0.3 pp (11, 14), under 0.05 pp (2, 12, 13).

Also: accuracy is 0.220 when the final runner-up is negative (1.08% of sequences).

## Claims

### C01 [MEASURED, confidence high]

On 2^20 N(0,1) sequences the share of steps on which each gate is on (all 10 steps pooled) is: unit 0 0.752, 1 0.729, 2 0.897, 3 0.766, 4 0.232, 5 0.409, 6 0.723, 7 0.770, 8 0.685, 9 0.262, 10 0.885, 11 0.875, 12 0.902, 13 0.904, 14 0.861, 15 0.728. At t = 9: 0.877, 0.866, 0.9997, 0.874, 0.114, 0.289, 0.865, 0.885, 0.858, 0.113, 0.869, 0.894, 0.985, 0.991, 0.986, 0.898.

- numbers: n = 1,048,576 sequences per step (seeds 1000-1003). Per step t = 0..9: unit 4: .500 .375 .293 .237 .200 .175 .157 .141 .127 .114; unit 5: .500 .499 .512 .481 .435 .392 .354 .325 .306 .289; unit 9: .500 .476 .364 .287 .233 .198 .172 .150 .130 .113; unit 3: .500 .907 .867 .889 .751 .623 .657 .758 .838 .874; unit 15: .500 .789 .986 .795 .496 .485 .675 .796 .859 .898; unit 10: .500 .954 .996 .999 1.000 .997 .850 .835 .849 .869; unit 11: .500 .935 .983 .988 .985 .847 .861 .871 .881 .894. Full table: a01_out.txt.
- how checked: a01_gate_freq.py. Checks that could have failed: the t = 0 column is 0.4999/0.5001 as the sign rule requires; P(unit 4 on at t = 9) = 0.1142 equals P(R*) 0.1001 + P(false record) 0.0142 from separate counts (a09_extras.py); a plain-loop re-implementation on seed 1100 (a10_checks.py) reproduces accuracy 0.953 and the resting share.

### C02 [EXACT, confidence high]

At t = 0 the gate of unit i is exactly [w_i x_0 > 0]: units 0, 2, 4, 10, 11, 12 (positive input weight) are on iff x_0 > 0 and the other ten iff x_0 < 0. These are the two t = 0 patterns, each 5.0-6.0% of all steps.

- numbers: h_(-1) = 0, so z_0 = w x_0. w > 0 for units 0 (0.069), 2 (0.015), 4 (10.157), 10 (0.060), 11 (0.148), 12 (0.298).
- how checked: Algebra; the measured t = 0 column of C01 is 0.4999 for exactly these six units and 0.5001 for the others.

### C03 [MEASURED, confidence high]

By literal class of step t >= 1 (R: x_t > M; N: S < x_t < M; L: x_t < S): R (share 0.2142): unit 4 on 0.899, all four comparators off 0.858, all on 0.098. N (0.2145): unit 4 on 0.045, all comparators on 0.431, off-sets {6,8} 0.187, {1,8} 0.117, {8} 0.092. L (0.5713): unit 4 on 0.0001, all comparators on 0.989.

- numbers: n = 9,437,184 steps: R 2,021,649, N 2,024,284, L 5,391,251. P(on) of comparators 1/6/7/8: R 0.119/0.112/0.106/0.109; N 0.750/0.724/0.960/0.526; L 0.995/0.995/1.000/0.997. Most common full patterns per class and all conditional tables: a02_out.txt.
- how checked: a02_events.py. Check: for iid inputs P(R) = P(N) = mean over t = 1..9 of 1/(t+1) = 0.21433 and P(L) = 0.57134; measured 0.2142, 0.2145, 0.5713. Loop re-implementation (seed 1100): 0.2151/0.2141/0.5709.
- caveats: The literal classes mix positive and negative inputs, which the network treats differently (C04).

### C04 [MEASURED, confidence high]

The gates follow positive-part classes, not literal ones. New max above a positive max (x_t > M > 0): unit 4 on 0.9999, all four comparators off 0.937. First positive input: 0.9974 and 0.997. Negative new max (0 > x_t > M): unit 4 on 0.0000 (n = 202,996), all comparators on 0.977. Negative new runner-up: all comparators on 0.968 (M > 0) and 1.000 (M < 0). Low inputs with S > 0, by r = x_t/S: all four comparators on 1.000 (r < 0.25), 0.999 (0.25-0.5), 0.995 (0.5-0.7), 0.947 (0.7-0.9), 0.650 (0.9-1).

- numbers: Positive new runner-up by quarter of q = (x_t - S+)/(M - S+): P(4 on) = 0.002, 0.007, 0.053, 0.316; P(7 on) = 0.997, 0.990, 0.917, 0.765; P(8 on) = 0.486, 0.234, 0.120, 0.103; P(1 on) = 0.756, 0.641, 0.594, 0.317; P(6 on) = 0.676, 0.652, 0.537, 0.344. 17 fine classes with n, comparator off-sets and top-4 full patterns: a02_out.txt (A2.2, A2.3).
- how checked: a02_events.py on 9,437,184 steps; class shares sum to the literal shares of C03.

### C05 [MEASURED, confidence high]

Event cascade conditioned on earlier steps (positive-part classes, t >= 2): unit 5 is on at t with probability 0.992-1.000 when step t-1 was R* or N*, and 0.032-0.070 when it was L*. Unit 9 is on at t after R* at t-1 with probability 0.9998 (x_t is N*), 0.9964 (L*), 0.906 (R*); after N* 0.002-0.044; after L* at most 0.0002. Unit 0 is off at t with probability 0.989-1.000 in the six class triples with R* at t-2 and not R* at t-1, and 0.0000-0.027 in the other 21 triples.

- numbers: Pairs (class t-1, class t): P(5 on), P(9 on): R*R* .9952 .9063; R*N* 1.0000 .9998; R*L* 1.0000 .9964; N*R* .9921 .0279; N*N* .9981 .0019; N*L* .9999 .0436; L*R* .0323 .0000; L*N* .0582 .0000; L*L* .0701 .0002. P(0 off): R*N*R* .9888, R*N*N* .9999, R*N*L* .9989, R*L*R* .9946, R*L*N* 1.0000, R*L*L* .9981; R*R*L* .0272, N*L*L* .0269, N*L*R* .0200, R*R*R* .0131. By r = x/S of an earlier low input: P(5 on next step) = .001 (r<0), .008, .031, .130 (0.5-0.7), .694 (0.7-0.9), .995 (0.9-1). After an N+ input in the top quarter of (S+, M): P(9 on) = .189, P(0 off two steps later) = .070. Top-3 full patterns for all 9 pairs and 27 triples: a02_out.txt (A2.4, A2.5); a04_out.txt (A2.6).
- how checked: a02_events.py, a04_patterns.py; n per triple 16,118 to 2,341,755. The same rules scored as Boolean rules with confusion counts (C09-C11) and re-derived with plain loops on seed 1100.
- caveats: With literal classes the delayed responses are blurred (P(5 on | literal N at t-1) = 0.38-0.83) because literal N and R include negative inputs.

### C06 [MEASURED, confidence high]

Which comparators switch off at a new positive runner-up (N*) depends on the age a of the max in h_(t-1). Most common off-set: a=0 {6,8} 0.877; a=1 {1} 0.393, {1,8} 0.346, {1,6,8} 0.252; a=2 {1,8} 0.648, {8} 0.273; a=3 {8} 0.762; a=4 none 0.527, {8} 0.243, {7,8} 0.216; a=5 none 0.655, {7} 0.191; a=6 none 0.589, {6,7} 0.338; a=7 none 0.433, {1,6,7} 0.228, {6} 0.228; a=8 {1,6} 0.400, {1} 0.268. So at ages 4-7 a new runner-up leaves all four comparators on in 43-66% of cases and is then registered only by unit 5 one step later.

- numbers: n (N* steps) for a = 0..8: 420,540; 288,301; 201,256; 140,999; 98,610; 67,218; 44,219; 25,681; 11,598. P(unit 4 on | N*, a) = .011, .023, .035, .100, .193, .280, .294, .221, .127.
- how checked: a02b_ncode.py on all 2^20 sequences. Consistent with the threshold positions of C19 (for example at a = 0 thresholds 8 and 6 sit at S, so any input above S switches them off).

### C07 [MEASURED, confidence high]

Unit 4: 'on iff R*(t)' (x_t > max(M,0)) is right on 0.99118 of 10,485,760 steps; 'on iff comparator 7 off at t' on 0.99583. With the literal class R it is 0.96869. Of 2,432,302 unit-4 firings 90,962 (3.74%) are not new positive maxima, and of 2,342,880 R* steps 1,540 lack unit 4.

- numbers: Confusion counts (TP, TN, FP, FN; target 'unit 4 on'): R*(t): 2,341,340 / 8,051,918 / 1,540 / 90,962. 7 off(t): 2,401,252 / 8,040,818 / 12,640 / 31,050. All four comparators off: 0.98350 (2,259,588 / 8,053,113 / 345 / 172,714). Literal R (t >= 1): 1,817,113 / 7,324,573 / 204,536 / 90,962.
- how checked: a03_rules.py; loop re-implementation on seed 1100 gives 0.9910 and 0.9957.

### C08 [MIXED, confidence high]

Comparators: class rules are only moderately accurate (7 on iff not R*: 0.99107; 8 on iff L*: 0.96180; 1 on iff not R*: 0.94551; 6 on iff not R*: 0.94146), but the rule 'comparator j on iff x_t < S+ + u_j(a)(M - S+)', with u_j(a) the median threshold position at age a of the max, is right on 0.99124 (unit 8), 0.99351 (6), 0.99080 (1), 0.99742 (7) of steps, and the mirrored rule for unit 4 on 0.99686, on seeds not used for fitting.

- numbers: Out-of-sample confusion counts (TP, TN, FP, FN; n = 4,718,592 steps t >= 1 of seeds 1002-1003): 8: 3,312,395 / 1,364,860 / 23,879 / 17,458; 6: 3,514,799 / 1,173,158 / 17,204 / 13,431; 1: 3,536,155 / 1,139,006 / 18,397 / 25,034; 7: 3,768,915 / 937,520 / 6,841 / 5,316; 4: 946,524 / 3,757,265 / 7,790 / 7,013. Class rules (n = 10,485,760): 7 on iff not R*: 8,060,536 / 2,331,548 / 82,344 / 11,332; 8 on iff L*: 6,813,727 / 3,271,510 / 30,731 / 369,792; 1 on iff not R*: 7,608,976 / 2,305,452 / 533,904 / 37,428; 6 on iff not R*: 7,555,145 / 2,316,827 / 587,735 / 26,053. A single constant position per unit (u = 0.83, 0.68, 1.01, 0.08 for 1, 6, 7, 8) gives 0.949, 0.944, 0.990, 0.961.
- how checked: a03_rules.py (class rules, all seeds); a02b_ncode.py (u_j(a) = medians fitted on seeds 1000-1001 with S > 0, M - S > 0.2; scored on seeds 1002-1003; when no positive input has occurred the rule uses threshold 0).
- caveats: The exact rule is the threshold of C17; this rule replaces the state-dependent threshold by a 9-entry table per unit.

### C09 [MEASURED, confidence high]

Unit 9: 'on at t iff unit 4 on at t-1' is right on 0.98973 of 9,437,184 steps; almost all errors are 'unit 4 fired but unit 9 stays off' (91,861 against 5,103). 'On iff 4 on and 8 off at t-1' gives 0.99157; 'on iff R*(t-1)' 0.98936; 'on iff 7 and 8 off at t-1' 0.98989.

- numbers: TP, TN, FP, FN: 4 on(t-1): 2,220,711 / 7,119,509 / 91,861 / 5,103. 4 on and 8 off (t-1): 2,207,600 / 7,150,075 / 61,295 / 18,214. R*(t-1): 2,181,657 / 7,155,070 / 56,300 / 44,157. 7, 8 off (t-1): 2,187,695 / 7,154,069 / 57,301 / 38,119. 7 off (t-1): 0.98743.
- how checked: a03_rules.py; loop re-implementation on seed 1100: 0.9895 and 0.9915. A search over all one- and two-feature Boolean rules on 133 lagged gate and event features (x04_rule_search.py, 400,000 steps) found nothing better than 0.9939.
- caveats: The exact statement is the delayed threshold of C18.

### C10 [MEASURED, confidence high]

Unit 5 is a threshold on the previous input relative to the runner-up: 'on at t iff x_(t-1) > 0.75 max(S_(t-2), 0) and x_(t-1) > 0' is right on 0.98324 of 9,437,184 steps. 'On iff x_(t-1) > S+' (R* or N* at t-1) gives 0.96312, with 341,162 misses where unit 5 is on although the input was below S+.

- numbers: TP, TN, FP, FN: 0.75 S+: 3,647,740 / 5,631,286 / 39,064 / 119,094. Above S+: 3,425,672 / 5,663,508 / 6,842 / 341,162. Other constants: 0.6 S+ 0.97394; 0.7 S+ 0.98203; 0.8 S+ 0.98180; 0.9 S+ 0.97317. 'x_(t-1) > 0' 0.88846; '8 off at t-1' 0.93357; 'any comparator off at t-1' 0.95581.
- how checked: a03_rules.py; loop re-implementation on seed 1100: 0.9838 (0.75 S+) and 0.9635 (S+).
- caveats: The exact threshold theta_5 is about 0.72 S in the median but varies (quartiles 0.62-0.79 of S), see C23.

### C11 [MEASURED, confidence high]

Unit 0 is off at t iff a new positive maximum arrived at t-2 and none at t-1: right on 0.99587 of 8,388,608 steps (t >= 2). The candidate 'off iff units 7 and 9 were both on at t-1' is right on 0.97394 of steps t >= 1 and 0.99323 of steps t >= 2.

- numbers: TP, TN, FP, FN (target 'unit 0 off'): R*(t-2) and not R*(t-1): 1,711,072 / 6,642,902 / 3,893 / 30,741. 7 on and 9 on at t-1, t >= 1: 2,055,868 / 7,135,342 / 221,197 / 24,777; t >= 2: 1,718,885 / 6,612,964 / 33,831 / 22,928. 4 on(t-2) and 4 off(t-1): 0.99258. 9 on(t-1) alone: 0.93775. 's_R = 2' on steps with M_t > 0: 0.95593 (it misses the case 'R* now and R* two steps ago').
- how checked: a03_rules.py; loop re-implementation on seed 1100: 0.9959, 0.9740, 0.9935. The t = 1 failures of the candidate are explained by algebra: after x_0 < 0 units 7 and 9 are both on but z_1[0] = -0.03 |x_0| + 0.069 x_1, whose sign is not fixed.

### C12 [MEASURED, confidence high]

'Always on from some step' holds for wave units 2 and 13 only in the form 'on whenever a positive input occurred before t': unit 2 is off on 0.03% and unit 13 on 0.34% of such steps. It does not hold for unit 12 (off on 3.5%) or unit 14 (5.9%; 1.6% once two positive inputs have occurred).

- numbers: Unconditional: unit 2 on for all t >= 2: 0.98469 (off on 128,446 of 8,388,608 steps); unit 13 on for t >= 3: 0.99320; unit 12 on for t >= 2: 0.95522; unit 14 on for t >= 2: 0.92062, t >= 3: 0.95171. Given M_(t-1) > 0 (n = 8,388,784 steps): P(off) = unit 0 .2075, 2 .0003, 3 .2098, 10 .0748, 11 .0810, 12 .0351, 13 .0034, 14 .0591, 15 .2490.
- how checked: a03_rules.py and a09_extras.py (per-step table in a09_out.txt).

### C13 [MEASURED, confidence high]

Wave units 11, 10, 0 and 12 are off at one specific age of the last new positive maximum. On steps with t >= 1 and M_t > 0: P(unit 11 off | s_R = 5) = 0.919 and the rule 'off iff s_R = 5' is right on 0.98693; P(unit 10 off | s_R = 6) = 0.950, rule 0.98014; P(unit 0 off | s_R = 2) = 0.999; P(unit 12 off | s_R = 4) = 0.296 and unit 12 is off at no other age (at most 0.019).

- numbers: n = 8,912,067 steps. TP, TN, FP, FN: 11 off iff s_R = 5: 617,767 / 8,177,794 / 54,428 / 62,078. 10 off iff s_R = 6: 475,618 / 8,259,474 / 24,834 / 152,141. 12 off iff s_R = 4: 0.92703, below the base rate 0.9670 (259,383 / 8,002,380 / 615,603 / 34,701). P(off | s_R = 0..9): unit 10: .035 0 0 0 0 0 .950 .211 .059 .002; unit 11: .031 .001 0 0 .001 .919 .002 .001 .001 .002; unit 12: .003 .003 .001 .019 .296 0 0 0 0 0; unit 14: .378 0 .170 0 0 0 0 0 0 0. Unit 10 is also off at (s_R, s_N) = (7, 6) with 0.96 and (8, 6) with 0.42. For the single impulse x = (1, 0, ..., 0) the pre-activation minima are unit 0 -4.12 (age 2), 14 -0.45 (age 2), 15 -0.41 and -1.45 (ages 3, 4), 11 -0.02 (age 5), 10 -0.46 (age 6), 12 +0.42 (age 4).
- how checked: a03b_wave.py (tables P(off | s_R), P(off | s_N), P(off | s_R, s_N) in a03b_out.txt; P(off | a1, a2) in a03c_out.txt); lookup on s_R fitted on seeds 1000-1001 scores 0.9869 (unit 11) and 0.9802 (unit 10) on seeds 1002-1003. Impulse response: x05_impulse.py.
- caveats: The single-impulse minima of units 11, 10, 14 are close to zero, so these gates are grazing events (see clip sizes in C29).

### C14 [MIXED, confidence high]

Wave units 3 and 15 have no simple gate rule in terms of event ages. A lookup table on (s_R, s_N, decile of S+/M), fitted on two seeds and scored on two others, predicts unit 3's gate on 0.8401 of steps (always-on baseline 0.7853) and unit 15's on 0.8281 (baseline 0.7393). Unit 15 is off mainly at s_R = 2..5 and unit 3 at s_R = 0..2 and 4..6, with probabilities between 0 and 0.95 depending on s_N.

- numbers: Lookup accuracy on seeds 1002-1003 (always on / s_R / (s_R, s_N) / (s_R, s_N, S+/M decile)): unit 3: .7853 / .7853 / .8214 / .8401; unit 15: .7393 / .7393 / .7954 / .8281; unit 14: .8956 / .8956 / .8993 / .9484; unit 12: .9670 / .9670 / .9679 / .9750; unit 13: .9594 / .9594 / .9594 / .9693. P(off | s_R = 0..9): unit 3: .241 .183 .305 .020 .386 .322 .217 .041 .013 .055; unit 15: .316 .139 .313 .445 .438 .241 .006 0 0 0. Examples of P(off | s_R, s_N): unit 15: (3,2) 0.95, (4,2) 0.88, (4,1) 0.73, (0,2) 0.71; unit 3: (4,2) 0.91, (5,4) 0.68, (4,1) 0.66, and at most 0.03 for s_R = 3.
- how checked: a03b_wave.py, train/test split by seed (4,456,825 / 4,455,242 steps with t >= 1, M_t > 0).
- caveats: A better rule may exist in other variables; I did not find one.

### C15 [MEASURED, confidence high]

The exact resting pattern (units 4, 5, 9 off, the other 13 on) holds on 10.32% of all steps: 0 at t = 0 and 1, then 5.9%, 8.4%, 7.4%, 5.4%, 8.0%, 15.8%, 23.4%, 29.0% at t = 2..9 (12.9% of steps t >= 2). 'Pulses off and all four comparators on, wave units in any state' holds on 37.0% (58.5% at t = 9). 1377 distinct 16-bit patterns occur in the sample; 10 cover 50% of steps, 75 cover 90%, 237 cover 99%.

- numbers: n = 10,485,760 steps. Resting share by class of step t: R* 0.0000, N* 0.037, L* 0.164. When steps t-1 and t are both L*: resting 0.274, pulses-off-comparators-on 0.922. Pulses 4, 5, 9 off: 0.419 of all steps; four comparators on: 0.661; all nine wave units on: 0.337.
- how checked: a04_patterns.py and a01_gate_freq.py (same number from two scripts); loop re-implementation on seed 1100 gives 0.1041 on 300,000 steps.
- caveats: The count of distinct patterns is for this sample of 2^20 sequences.

### C16 [MEASURED, confidence high]

The 15 most common full gate patterns cover 59.1% of all steps. In order (share): (1) resting 0.1032, low input after a low input with no wave unit at a clipping age; (2) 5 and 9 on, rest as resting 0.0923, low input one step after a new max; (3) 4 on, comparators off, wave 3, 13, 14, 15 off 0.0600, first positive input, mostly x_0 > 0; (4) 5, 9 on, wave 0, 2, 10, 11, 12 off 0.0501, t = 0 with x_0 < 0; (5) resting but unit 15 off 0.0370, low input 3-4 steps after the last new max; (6) resting but 10 off 0.0339, 6-7 steps after; (7) resting but 0 off 0.0333, 2 steps after; (8) 5, 9 on and comparators 6, 8 off 0.0332, new runner-up one step after a new max; (9) only 5 on 0.0307, low input after a new runner-up or a near miss; (10) 4, 5, 9 on, comparators off 0.0267, second of two new maxima in a row; (11) resting but 11 off 0.0255, 5 steps after the last new max; (12) 4 on, comparators off, wave all on 0.0196, new max after two low inputs; (13) 5, 9 on and 3 off 0.0167, low input one step after a new max; (14) resting but 0, 15 off 0.0146 and (15) resting but 0, 3, 15 off 0.0142, both low inputs 2 steps after a new max.

- numbers: For each pattern a04_out.txt gives the class shares of steps t, t-1, t-2, the top values of s_R and s_N and the share per step t. Examples: pattern 5: s_R = 3 in 82%, 4 in 17%; pattern 6: s_R = 6 in 80%, 7 in 16%; pattern 7: s_R = 2 in 98%; pattern 11: s_R = 5 in 99%; pattern 8: N* at t in 95%, R* at t-1 in 100%; pattern 9: N* at t-1 in 73%, L* in 26%.
- how checked: a04_patterns.py on 10,485,760 steps; descriptions read off the class and age columns of that table.

### C17 [EXACT, confidence high]

F2 holds exactly. With theta_j(t) = -W[j].h_(t-1)/w_j, comparator j (1, 6, 7, 8) is on iff x_t < theta_j(t) and unit 4 is on iff x_t > theta_4(t): 0 mismatches in 52,428,800 gate evaluations and no ties. When on, h_t[j] = |w_j| (theta_j(t) - x_t) for a comparator and h_t[4] = w_4 (x_t - theta_4(t)).

- numbers: 5 units x 10,485,760 steps, 0 mismatches each.
- how checked: Algebra (z_t[j] = w_j x_t + W[j].h_(t-1), divide by w_j, sign flips for w_j < 0); a05_thresholds.py numerically; independent loop version on seed 1100: 0 mismatches of 1,500,000.

### C18 [MEASURED, confidence high]

Units 5 and 9 obey a threshold on the previous input: with h_(t-1) and x_(t+1) fixed, z_(t+1)[K](x_t) = W[K].ReLU(w x_t + W h_(t-1)) + w_K x_(t+1) changed sign exactly once in every tested case, at theta_K(t), and unit K was on at t+1 iff x_t > theta_K(t): 0 exceptions in 9,437,184 steps for each of K = 5, 9.

- numbers: Root by bisection on [-60, 60] (46 halvings); 0 steps without a sign change, 0 gate mismatches. Slope contributions W[K,i] w_i: unit 5: +12.06 (unit 4 on), +10.47 (8 on), +7.81 (7 on), negative terms sum to at most -0.51; unit 9: +66.3 (7 on), +13.4 (8), +3.8 (1), +1.5 (4), +1.0 (6), negative terms at most -0.16. theta_K depends weakly on x_(t+1) (w_5 = -0.281, w_9 = -0.228).
- how checked: a05_thresholds.py. The gate equality could have failed wherever the function is not monotone; it did not.
- caveats: Uniqueness of the sign change is a numerical finding, not a proof: the slope can be slightly negative when comparators 7, 8 (and 1, 6 for unit 9) and unit 4 are all off.

### C19 [MEASURED, confidence high]

Thresholds are not fixed points between S and M. Median position u = (theta - S)/(M - S) by age a = 0..8 of the max in h_(t-1): unit 8: -0.01, 0.33, 0.05, 0.04, 0.41, 0.98, 1.43, 1.50, 1.16; unit 6: -0.06, 0.65, 1.05, 1.20, 1.08, 0.80, 0.49, 0.31, 0.35; unit 1: 0.84, -0.07, 0.22, 0.80, 1.31, 1.44, 1.14, 0.59, 0.06; unit 7: 1.03, 1.02, 1.01, 0.87, 0.68, 0.55, 0.55, 0.65, 0.83; unit 4: 0.98, 0.96, 0.94, 0.85, 0.72, 0.62, 0.61, 0.69, 0.81. Thresholds lie above M on 15.0% (unit 8), 39.4% (6), 25.8% (1), 50.6% (7), 0.7% (4) of steps and below S on 20.9%, 20.1%, 20.8%, 0.4%, 0.2%.

- numbers: Selection: S > 0 and M - S > 0.2 for the prefix before the tested input; n = 4,755,708 steps (906,249 at a = 0 down to 84,266 at a = 8). Interquartile ranges are 0.01-0.07 wide for units 7 and 4 at a <= 3 and 0.05-0.27 wide elsewhere; full quartiles, the split by runner-up older or newer than the max, and medians by (a1, a2) are in a05_out.txt. The age of the runner-up shifts the medians by at most about 0.2. Delayed thresholds (a = 0..7): unit 5: -0.28, -0.21, -0.22, -0.24, -0.27, -0.35, -0.39, -0.40; unit 9: 0.98, 0.98, 0.97, 0.84, 0.74, 0.90, 1.16, 1.25.
- how checked: a05_thresholds.py on all 2^20 sequences; the same medians refitted on seeds 1000-1001 alone (a02b_ncode.py) agree to 0.002 and predict gates out of sample (C08).
- caveats: Positions are relative to the literal S and M of the prefix; steps with S <= 0 or M - S <= 0.2 are excluded.

### C20 [FIT, confidence high]

Each threshold is close to a linear function of the top two values given their ages: theta_j = alpha M + beta S fitted per (age of max, age of runner-up) cell explains 0.9940 (unit 8), 0.9969 (6), 0.9950 (1), 0.9989 (7), 0.9991 (4), 0.9986 (9) and 0.9683 (5) of the within-cell variance, with rms residuals of 0.016-0.038 input units (0.063 for unit 5). Everything else in the history moves the thresholds by that much.

- numbers: Per age of the max (pooled over runner-up age), alpha / beta: unit 4: a=0 0.985/0.006, a=2 0.957/0.026, a=3 0.873/0.101, a=4 0.764/0.194, a=5 0.691/0.243, a=6 0.695/0.228, a=7 0.768/0.168, a=8 0.865/0.091. Unit 7: 1.018/-0.009 (a=0), 0.620/0.313 (a=5), 0.883/0.080 (a=8). Unit 8: -0.027/1.033 (a=0), 0.317/0.693 (a=1), 0.055/0.924 (a=2), 0.446/0.539 (a=4), 0.955/0.084 (a=5), 1.379/-0.279 (a=7). Unit 6: -0.043/0.986 (a=0), 0.695/0.242 (a=1), 1.172/-0.146 (a=3), 0.471/0.388 (a=7). Unit 1: 0.836/0.155 (a=0), -0.001/0.916 (a=1), 1.352/-0.279 (a=5), 0.281/0.541 (a=8). Per-age R^2 0.91-1.00. Full table: a05_out.txt (A5.5).
- how checked: a05_tables.py: least squares without intercept; 72 cells with n >= 2000 (56 for units 5, 9); R^2 measured against the within-cell mean, so it is not inflated by between-cell differences.
- caveats: Same selection as C19 (S > 0, M - S > 0.2).

### C21 [FIT, confidence medium]

For ages a >= 2 of the max, the threshold positions oscillate at the clock frequency: u_j(a) = c_j + r_j cos(42.75 deg x a - phi_j) reaches the R^2 of the best arbitrary function of a to within 0.0005 for all five same-step thresholds. Amplitudes and phases: unit 7: r = 0.281, phi = 59.6 deg; unit 6: 0.506, 132.2; unit 1: 0.886, 204.7; unit 8: 0.806, 286.0; unit 4: 0.195, 63.1. The four comparator phases are 72.6, 72.5 and 81.3 degrees apart.

- numbers: n = 2,942,126 steps (ages 2..8, S > 0, M - S > 0.2). R^2 at 42.75 deg / ceiling for any function of a: unit 8 0.9268 / 0.9273; 6 0.8429 / 0.8431; 1 0.8826 / 0.8827; 7 0.8013 / 0.8013; 4 0.6835 / 0.6838. Means c: 0.773, 0.699, 0.593, 0.759, 0.753. Frequency left free (grid 20-90 deg, step 0.25): best 43.75, 43.50, 44.25, 42.75, 43.25 deg. With ages 0 and 1 included the single sinusoid fails for units 8 and 1 (R^2 0.61 and 0.67 against ceilings 0.91 and 0.92).
- how checked: a05c_clockfit.py: exact weighted least squares from per-age counts, sums and sums of squares; the free-frequency optimum landing at 42.75-44.25 deg is the check that could have failed.
- caveats: Seven ages and three parameters, covering 256 degrees of one period, so the frequency is only loosely constrained by this fit alone.

### C22 [MEASURED, confidence high]

Unit 4's threshold sags below the max as the max ages (median position 0.98 at age 0, 0.61-0.62 at ages 5-6, 0.81 at age 8), so unit 4 also fires for inputs that are not new maxima: 90,962 of 2,432,302 firings (3.74%), rising with t to 12.6% at t = 8 and 12.5% at t = 9. 8.4% of sequences contain such a false record; the accuracy on them is 0.895 against 0.958 on the others.

- numbers: False records per step t = 0..9: 0, 970, 2273, 3748, 7187, 11626, 15737, 17766, 16724, 14931; share of firings: 0, .0025, .0074, .0151, .0342, .0634, .0956, .1199, .1259, .1247. Sequences with at least one: 88,555 (accuracy 0.8948); without: 960,021 (0.9582). P(unit 4 on | N* event) by age of the max: .011, .023, .035, .100, .193, .280, .294, .221, .127.
- how checked: a09_extras.py (false record = unit 4 on and x_t <= max(M, 0)); consistent with the confusion counts of C07 and the threshold positions of C19.
- caveats: The accuracy difference is an association; I did not test whether the false record causes the errors.

### C23 [MIXED, confidence high]

Unit 5's delayed threshold is a fraction of the runner-up: theta_5/S has quartiles 0.620 / 0.721 / 0.786 and theta_5 = alpha M + beta S fits with alpha between -0.074 and -0.005 and beta between 0.72 and 0.87. Unit 9's delayed threshold follows unit 4's for ages 0-4 of the max (median positions 0.98, 0.98, 0.97, 0.84, 0.74 against 0.98, 0.96, 0.94, 0.85, 0.72) but is higher at ages 5-7 (0.90, 1.16, 1.25 against 0.62, 0.61, 0.69), so a false record at those ages usually does not fire unit 9.

- numbers: n = 4,005,096 steps (S > 0, M - S > 0.2). theta_5/S median by age 0..7: 0.631, 0.740, 0.750, 0.752, 0.736, 0.690, 0.683, 0.673; by S/M: 0.546 for S/M in [0.1, 0.2), 0.713 for [0.4, 0.5), 0.780 for [0.8, 0.9). Per-age R^2 of the theta_5 fit 0.84-0.98. theta_5 < S on 98.7% of steps; theta_9 > M on 8.6%.
- how checked: a05_thresholds.py and a05_tables.py; consistent with the gate statistics P(5 on at t+1) = 0.13, 0.69, 0.995 for x_t/S in [0.5, 0.7), [0.7, 0.9), [0.9, 1) (C05) and with the 0.75 S+ rule of C10.

### C24 [INTERVENTION, confidence high]

Resampling one input that is below half the runner-up (S > 0, x_t < S/2, redrawn from N(0,1) given x < S/2, median change 0.62): for inputs at least one step before the end the largest logit change is 0.37-0.61 in the median and 2.5-4.3 at the 99th percentile, the margin of the true answer changes by 0.08-0.20 in the median and 0.50-0.88 at the 90th percentile (median margin 8.5), and the network's answer changes in 0.45-0.73% of resamples (0.55% overall, n = 4,066,321). The effect is nearly flat in the distance k from the end.

- numbers: P(answer changes) for k = 0..7: .0045 .0073 .0062 .0053 .0052 .0047 .0046 .0048. Sub-cases: x_t < 0: 0.50% (n = 3,155,796); 0 < x_t < S/2: 0.13% (n = 910,525); S/2 < x_t < 0.7 S: 0.12% (n = 308,498). Total-variation distance of the softmax outputs: median 0.000, 90% 0.011, 99% 0.126; above 0.1 in 1.44%. Last input (k = 0): raw logit change median 36.6 but confined to logit 9; margin change median 0.118. Some later gate differs in 3.9% (k = 1) to 17% (k = 7) of resamples; per unit and lag the flip rate is at most 2.6% (unit 12 at the step itself), 0.5-1% for units 3, 15, 10, 11, 5.
- how checked: a06_resample.py (resampling seeds 2000-2003), a06d_gateflips.py. Checks: accuracy before and after is equal (0.9633 and 0.9634), 'correct to wrong' equals 'wrong to correct' (0.0026 and 0.0027), and the true answer never changed (asserted).
- caveats: Only steps t >= 2 with S_(t-1) > 0.

### C25 [EXACT, confidence high]

Exact linear trace of a low input k = 9 - t steps before the end, if the 13 ordinary units are on at t and the pattern is resting afterwards: d logits / d x_t = U[:, ORD] A^k w[ORD]. For k = 1..8 its largest entries are +0.63 to +1.01 at the input's own position and +0.65 to +0.89 at the position before it, and its size after removing the mean over logits is 0.78-0.98 per unit input. For k = 0 it is +57.19 at position 9 and -3.78 to -4.90 elsewhere. The measured median slope after resampling (0.66-0.98 for k = 1..7) agrees, and equals the kernel exactly at k = 0 (55.497) and k = 1 (0.981).

- numbers: Size by k = 0..8: 55.497, 0.981, 0.862, 0.780, 0.805, 0.821, 0.877, 0.826, 0.924. Measured median slope for x_t < S/2, k = 0..7: 55.497, 0.981, 0.868, 0.913, 0.906, 0.860, 0.802, 0.661. Kernel k = 1: (-0.13, -0.77, -0.37, +0.02, +0.14, +0.27, -0.18, +0.77, +1.01, -0.51).
- how checked: a06b_kernels.py (weights only) against a06_resample.py (data); the agreement at k = 0 and 1 to three decimals is the check.
- caveats: The kernel assumes the resting pattern after t; along real trajectories other patterns occur, which is why the measured slopes differ by up to 25% for k >= 3.

### C26 [INTERVENTION, confidence high]

An input just below the runner-up is not low for the network. Resampling x_t within (0.7 S, S) changes the answer in 5.5% of cases overall (10.8% for the last input, falling to 2.1% for k = 7); the slope is 38-62 logits per unit input against about 0.9 for inputs below S/2, because unit 5 fires one step later in 79% of these cases. Resampling a class-N input within (S, M) changes the answer in 14.5% of cases; the true answer changes in 10.4%; with the true answer unchanged the network's answer still changes in 5.5% (5.2% at k = 0, 7.8% at k = 1, 3.5% at k = 7).

- numbers: (0.7 S, S): n = 383,700; P(answer changes) k = 0..7: .1083 .0555 .0505 .0456 .0337 .0335 .0261 .0207; margin change median 0.45, 90% 6.3; total-variation distance above 0.1 in 14.6%; accuracy on these sequences 0.934 (0.895 when the near miss is the last input). (S, M): n = 779,497; P(answer changes) k = 0..7: .0521 .1266 .1545 .1622 .1718 .1714 .1666 .1493; margin change median 2.7, 90% 15.5. Exact kernel when unit 5 fires and the pattern is otherwise resting: +62.2 to +75.7 per unit input at the input's own position (k = 1..8), +15 to +21 at the next position, -15 to -44 at earlier ones; measured median slope 62.0 at k = 1 (exact 62.1), 58.1, 58.0, 58.5, 54.8, 52.6, 37.9 for k = 2..7.
- how checked: a06_resample.py and a06b_kernels.py. Checks: accuracy before and after equal (0.9337 and 0.9341; 0.9539 and 0.9540); for class N the answer changes are split by whether the true answer changed.
- caveats: The measured slopes fall below the exact kernel for larger k because not every resample keeps the input above theta_5 and later patterns are not resting.

### C27 [INTERVENTION, confidence high]

Resampling all low inputs of a sequence at once (every x_p with S_(p-1) > 0 and x_p < S_(p-1)/2; 3.88 inputs per sequence on average) changes the network's answer in 1.32% of sequences, changes the margin by a median 0.40 (99th percentile 3.2) and leaves all 16 gates at t = 9 unchanged in 94.6%. With only negative inputs resampled: 1.02%. With every input below the runner-up resampled: 3.65%.

- numbers: Sequences with at least one resampled input: 96.5% of 1,048,576. Threshold 0: 3.01 inputs, 1.02%, margin median 0.28; 0.5 S: 3.88, 1.32%, 0.40; 0.7 S: 4.17, 1.48%, 0.46; S: 4.54, 3.65%, 0.73. Total-variation distance above 0.1: 3.3%, 4.4%, 4.9%, 10.1%. P(answer changes) grows with the number resampled: 0.70% (one input) to 1.63% (six) at threshold 0.5 S. Accuracy before and after: 0.9611 and 0.9612.
- how checked: a06c_joint.py (resampling seeds 3000-3003); the script asserts that the running top two and the true answer are unchanged; accuracy invariance is the distribution check.

### C28 [MIXED, confidence high]

The trace of a low input in the hidden state is about 2-3% of its direct effect and does not decay further over the next steps. Measured median of |change of the 13 ordinary units| / |change of x_t| after resampling an input below S/2: 23.65 at the step itself, 0.47, 0.56, 0.50 one, two and three steps later; the pulse units change by 0 in the median at all later steps. Exactly, |A^k w[ORD]| = 23.652, 0.682, 0.682, 0.626, 0.604, 0.737, 0.691, 0.458, 0.530 for k = 0..8.

- numbers: For inputs in (0.7 S, S) the pulses change by 17.99 per unit input one step later (W[5, ORD].w[ORD] = 18.009) and the ordinary units by 21.3 and 17.9 two and three steps later. max_i |(A w[ORD])_i| = 0.432.
- how checked: a06_resample.py (trace table, steps t <= 6) and a06b_kernels.py (norms from the weights).
- caveats: This qualifies F4: what A leaves of the input direction is small (2.9% in norm) but lies in slow modes, so it persists to the end.

### C29 [MEASURED, confidence high]

How often and how much each wave unit is clipped (steps t >= 1, n = 9,437,184): unit 15 on 24.7% of steps, 0 on 22.0%, 3 on 20.4%, 14 on 9.9%, 11 on 8.4%, 10 on 7.2%, 2 on 5.9%, 12 on 5.3%, 13 on 5.1%. Clipped amount when clipped (50% / 90% / 99%): unit 0: 2.64 / 6.53 / 10.42; 3: 2.02 / 7.94 / 16.34; 15: 2.06 / 7.47 / 13.74; 14: 0.89 / 2.27 / 3.65; 10: 0.54 / 1.23 / 2.09; 11: 0.49 / 1.42 / 2.61; 12: 0.22 / 0.72 / 1.59; 13: 0.05 / 0.12 / 1.38; 2: 0.02 / 0.05 / 0.10.

- numbers: Mean activation when on: unit 0 9.13, 2 19.92, 3 3.68, 10 13.89, 11 10.74, 12 9.41, 13 6.79, 14 11.73, 15 5.48. Mean clip over all steps divided by that: unit 3 17.7%, 15 13.7%, 0 7.3%, 14 0.9%, 11 0.5%, 10 0.3%, 12 0.2%, 13 0.07%, 2 0.01%.
- how checked: a07_clips.py; check: P(clipped) per step equals 1 - P(on) of C01 (unit 15 at t = 4: 0.504 and 0.4961).

### C30 [MEASURED, confidence high]

Clipping by event history. By steps since the last new positive maximum: unit 0 is clipped at s_R = 2 (0.999, mean clip 3.96, about 3.2 times the running max in the median); unit 11 at s_R = 5 (0.919, mean 0.71); unit 10 at s_R = 6 (0.950, mean 0.71); unit 12 at s_R = 4 (0.296); unit 14 at s_R = 0 (0.378) and 2 (0.170); units 3 and 15 over several ages with mean clips of 1.2-1.9 at s_R = 2, 4, 5 and 2, 3, 4. By class triple (t-2, t-1, t): unit 0 is clipped in 0.989-1.000 of the six triples (R*, not R*, any); unit 15 in 0.49-0.72 of the triples starting with N* and containing no later R*-R* pair; unit 14 in 0.665 of (R*, L*, R*) and 0.635 of (R*, L*, N*).

- numbers: P(clipped | s_R = 0..9): unit 0: .200 .005 .999 .003 .001 .001 .001 .003 .015 .075; unit 3: .241 .183 .305 .020 .386 .322 .217 .041 .013 .055; unit 15: .316 .139 .313 .445 .438 .241 .006 0 0 0; unit 13: .190 0 .002 .002 0 0 .001 .008 .026 .028; unit 2: .112 at s_R = 0 and at most .001 otherwise. Before any positive input: unit 2 .661, 0 .506, 12 .401. By steps since the last new runner-up: unit 15 .204 .295 .647 .289 .136 for s_N = 0..4; unit 3 .126 .403 .432 .104 .406. Clip sizes per s_R, s_N, class triple and step t: a07_out.txt.
- how checked: a07_clips.py on all 2^20 sequences; the s_R profile matches the gate tables of a03b_wave.py computed separately.

### C31 [MEASURED, confidence high]

Relative to the whole state, the wave clip rho = |c_t| / |h_t| (2-norms; steps t >= 1 with running max above 0.5; n = 7,868,918) is zero on 37.9% of steps, below 1% on 46.9%, below 3% on 58.1%, below 10% on 79.3%; the 90th percentile is 0.129 and the 99th 0.326. So on about one step in five the wave-unit clipping changes the state by more than 10% of its norm.

- numbers: By class of step t: R*: P(rho < 0.1) = 0.628, median 0.056; N*: 0.765, 0.005; L*: 0.849, 0.012. With unit 0 left out of the clip: P(rho < 0.1) = 0.854.
- how checked: a07_clips.py.
- caveats: The ratio is to the norm of all 16 units; steps with running max below 0.5 are excluded because the ratio is unstable there.

### C32 [INTERVENTION, confidence high]

Removing the ReLU of single wave units (h = z at every step) changes the accuracy on 2^20 sequences from 0.9529 to: unit 0 0.7632 (-19.0 pp), unit 15 0.8803 (-7.3 pp), unit 3 0.9426 (-1.0 pp), unit 10 0.9441 (-0.9 pp), unit 11 0.9497 (-0.3 pp), unit 14 0.9502 (-0.3 pp), unit 12 0.9525 (-0.04 pp), unit 13 0.9528 (-0.01 pp), unit 2 0.9529 (0.00 pp). All nine together: 0.4436.

- numbers: Groups: units 2, 10, 11, 12, 13 together 0.9087 (-4.4 pp, against -1.2 pp for the sum of their single effects); units 3, 15: 0.8638; units 0, 3, 15: 0.6892; all wave units except 0: 0.7992. Share of sequences whose answer changes: unit 0 22.5%, 15 9.4%, 3 2.2%, 10 2.0%, 11 1.3%, 14 1.2%, 12 0.45%, 13 0.14%, 2 0.08%. Linear from t = 1 only: same to 0.001.
- how checked: a08_linearise.py. Controls: with no unit selected the code reproduces 0.9529; linearising comparator 8 or 7 gives 0.334 and 0.356, pulse 5 0.546, pulse 9 or 4 0.08.
- caveats: Like any ablation this takes the network off its trained distribution.

### C33 [MEASURED, confidence high]

When the second-largest input of the whole sequence is negative (at most one positive input; 11,356 of 1,048,576 sequences, 1.08%) the accuracy is 0.220, against 0.961 when it is positive. These sequences account for 0.84 percentage points of the 4.71% error.

- numbers: Expected share of such sequences: 11/1024 = 1.07%. Other splits: R* at t = 9: accuracy 0.9368; N* at t = 9: 0.9599; N* at t = 8: 0.9502; R* at t = 8: 0.9591.
- how checked: a09_extras.py; consistent with C04 (negative runner-ups and negative maxima do not change the comparator or unit-4 gates).

### C34 [INTERVENTION, confidence medium]

The gate pattern on random inputs is not reproduced by the two-pulse skeleton. Setting every input except the current largest and second largest (if positive) to 0 reproduces all 16 gates of step t on 51.3% of steps; keeping x_(t-1) and x_t as well, on 69.1% (75.3% on L* steps). Per unit, in the second variant, the agreement is 97.5-99.7% except unit 3 (88.8%), unit 15 (83.1%) and unit 5 (97.0%).

- numbers: n = 8,912,067 steps with t >= 1 and M_t > 0. Second variant per unit u0..u15: .9887 .9892 .9827 .8879 .9954 .9701 .9927 .9946 .9882 .9969 .9899 .9861 .9776 .9747 .9765 .8310. First variant: unit 5 .8720, 6 .9303, 8 .9257, 3 .8443, 15 .8121.
- how checked: a03c_skeleton.py.
- caveats: Zero is a special fill value: it sits exactly at S+ when there is no second positive input, and the skeleton also removes former maxima, which did fire unit 4. The numbers are a lower bound on how far the top two determine the gates.

### C35 [EXACT, confidence high]

F8's pulse kernels are reproduced from the weights, with side lobes that F8 does not list. Per unit of activation, propagated with A and read with U: unit 4: -4.99 at its own position at age 0, -6.63 at age 1, -3.32 to -4.12 at ages 2-8, plus -1.39 to -1.73 at the following position and +0.1 to +2.1 at the two preceding positions (ages >= 2). Unit 5 firing at step s: +3.37 to +4.16 at position s-1 (the input that triggered it), +0.85 to +1.37 at position s, -0.37 to -1.92 at s-2. Unit 9 firing at step s: -1.92 to -3.37 at position s-2 for ages >= 1, and +2.3 to +2.9 at early positions for ages 1-5.

- numbers: Unit 4 own position, ages 0..8: -4.99, -6.63, -3.48, -3.84, -3.82, -3.97, -4.12, -3.88, -3.32. Unit 5 at s-1, ages 0..8: +3.40, +3.37, +3.66, +3.82, +3.90, +4.16, +3.95, +3.59, +3.67. Unit 9 at s-2, ages 1..7: -1.92, -2.80, -3.02, -3.37, -3.14, -3.07, -3.17.
- how checked: a06e_pulse_kernels.py: U[:, P] at age 0 and U[:, ORD] A^(m-1) W[ORD, P] at age m.
- caveats: Free-flow kernels assume the resting pattern after the pulse, which holds exactly on a minority of steps (C15).

## Contradictions with the lead

Nothing contradicts F1-F4, which I re-verified: F1 max error 3.8e-13 on 20,000 sequences; F2 0 mismatches in 52,428,800 evaluations; F3 eigenvalues identical; F4 |cos| = 0.998, max |A w[ORD]| = 0.432, pulse drive (0.010, 18.009, 84.335). Accuracy: 0.9529 on the 2^20 sequences, 0.9531 +- 0.0002 on 2^21 more; uniform inputs 0.9382 +- 0.0002. F8's pulse kernels are reproduced (C35). F5, F6, F7 and F9 were not re-derived.

Four results qualify the lead's picture:
1. F3/F7/F8 rest on the resting map A, but the exact resting pattern holds on only 10.3% of steps (29.0% at t = 9). On a further 26.7% only wave-unit clips separate the step from resting, and those clips exceed 10% of the state norm on 20.7% of steps. Unit 0's clip two steps after every new maximum (mean 3.96) is needed for 19 pp of accuracy and unit 15's for 7 pp.
2. F2's thresholds are not comparison points inside (S, M). They sweep from below S to 1.5 times the way to M with the age of the max, at the clock frequency; they lie above M on 15-51% of steps and below S on about 21% (units 8, 6, 1). Unit 4's threshold sags to about 0.69 M + 0.24 S at ages 5-6, so 3.74% of its firings are not new maxima.
3. F4 says a low input leaves almost nothing one step later. In norm that is 2.9% (0.682 of 23.65), but it does not decay afterwards (|A^k w| stays 0.46-0.74 up to k = 8) and gives a structured logit trace of about +0.8 per unit input at the input's own position and the one before.
4. F8 omits side lobes: unit 5 also adds +0.85 to +1.37 at its own firing position, unit 4 adds -1.4 to -1.7 at the following position.
On F9: two-pulse inputs are answered correctly, but zeroing everything except the top two reproduces all 16 gates of real runs on only 51% of steps (69% keeping the last two inputs).

## Surprises

1. The resting pattern is rare (10.3% of steps); 1377 distinct gate patterns occur and it takes 75 to cover 90% of steps.
2. Comparator thresholds rotate with the age of the max instead of sitting between S and M. For ages >= 2 a sinusoid at 42.75 degrees explains all of the age dependence, with the four comparators about 73-81 degrees apart in phase.
3. At ages 4-7 of the max a new runner-up leaves all four comparators on in 43-66% of cases; the event is then visible in the gates only through unit 5 one step later.
4. Unit 4 fires for inputs that are not new maxima: 3.74% of its firings overall, 12.5% at t = 8 and 9. Unit 9's delayed threshold is stricter at ages 5-7 (position 0.90-1.25 against 0.61-0.69), so it usually does not follow a false record.
5. The network's boundary of 'low' is unit 5's threshold near 0.72 S, not S. Inputs in (0.7 S, S) change the answer in 5.5% of resamples (10.8% for the last input), inputs in (0.5 S, 0.7 S) in 0.12%.
6. Wave-unit clips look small but are not negligible: unit 0's ReLU is worth 19 pp, unit 15's 7.3 pp, and the five smallest-clip units together 4.4 pp, more than the sum of their single effects (1.2 pp).
7. Negative values are invisible to the network: a negative new maximum never fires unit 4, and when the final runner-up is negative the accuracy is 22.0%.
8. The delayed gates of units 5 and 9 are exact thresholds on the previous input (0 exceptions in 9.4 million steps each), so the network has seven thresholds per input: five applied at once and two read out one step later.

## Not done

- findings.md does not exist. The harness refused the write for this subagent ('return findings as text, not write report files') and I did not write it by another route. The write-up is this object; every table is in the *_out.txt files (script stdout). If the lead wants findings.md on disk it has to be assembled from these.
- One transient helper file was created in the Windows temp directory (outside my folder) and deleted immediately; nothing else was written outside r1_A_census/.
- Uniform inputs: only the accuracy was measured (0.9382 +- 0.0002); the census was not repeated.
- No rule better than 84% / 83% for wave units 3 and 15, and no exact rule for units 12, 13, 14.
- No test of whether false records cause the lower accuracy (0.895) on the sequences that contain them.
- No proof that z_(t+1)[5] and z_(t+1)[9] have a single sign change in x_t; it is a numerical finding.
- Full 16-bit patterns were conditioned on the positive-part class of the two earlier steps, not on the fine bins of x/S of earlier steps; only units 5, 9 and 0 were conditioned on those bins.
- Class R inputs were not resampled.
- F5, F6, F7 and F9 were not re-derived.

## Open questions

- How does the readout stay correct after a false record (unit 4 firing at 0.61-0.72 of the way from S to M at ages 4-7 of the max)? Unit 9 mostly does not follow at ages 5-7; is that the correction, and does the false record cause the 0.895 accuracy on those sequences?
- What is the functional role of unit 0's clip two steps after every new maximum (19 pp) and of unit 15's clips (7.3 pp)? Which logit or mode do they correct?
- What variables determine the gates of units 3 and 15? The ages of the last R* and N* events plus S/M give only 84% and 83%.
- At ages 4-7 a new runner-up often leaves all comparators on. Is unit 5's pulse (about 18 (x - theta_5), kernel +3.4 to +4.2 at the input's position) then the only carrier of the runner-up's position, and how does that fit the clock amplitude proportional to M - S (F6)?
- Thresholds overshoot M (position up to 1.5). When a new maximum lands between M and such a threshold one comparator stays on (6.3% of new maxima above a positive max). Does the network handle these cases differently?
- Is the 22% accuracy with a negative runner-up a necessary consequence of the bias-free ReLU design or just absent from training pressure (1.08% of sequences)?
- Why is the joint effect of linearising the five small-clip wave units (4.4 pp) so much larger than the sum of single effects (1.2 pp)? Units 12, 11, 10 are clipped at successive ages 4, 5, 6, which suggests a chain.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/cc.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a00_basics.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a00_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a01_gate_freq.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a01_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a02_events.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a02_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a02b_ncode.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a02b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a03_rules.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a03_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a03b_wave.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a03b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a03c_skeleton.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a03c_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a04_patterns.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a04_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a05_thresholds.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a05_tables.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a05_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a05c_clockfit.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a05c_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06_resample.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06b_kernels.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06b_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06c_joint.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06c_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06d_gateflips.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06d_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06e_pulse_kernels.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a06e_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a07_clips.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a07_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a08_linearise.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a08_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a09_extras.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a09_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a10_checks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/a10_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/x04_rule_search.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/x04_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/x05_impulse.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/x05_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/x06_explore.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_A_census/x06_out.txt