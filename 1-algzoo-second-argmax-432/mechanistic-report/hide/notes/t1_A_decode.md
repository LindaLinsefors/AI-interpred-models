# Tape investigation, arm A_decode

## Answer

Terms: $M \ge S \ge T$ are the three largest inputs so far, $a_M, a_S, a_T$ their ages (steps since arrival), $c$ the seven tape coordinates, "the rest" the nine other numbers of the state (level, runner, clock, trans, echo coordinates and the three pulse units). I decoded from $c/|c|$ (the tape alone, scale removed by its own norm), from the rest and from the full state, on quiet final states ($t = 9$, $a_M, a_S \ge 2$, $S > 0$; 246,144 states, a quarter held out), with a linear and a flexible (gradient-boosting or nearest-neighbour) decoder.

1. The tape stores where the top two inputs are, above all the age of the runner-up. $a_S$ is read from the tape alone with accuracy 0.90 (linear) and 0.95 (flexible), against 0.35 and 0.65 from the rest and 0.98 from the full state (majority class 0.13). $a_M$ is also in the tape, but only non-linearly (0.89 flexible, 0.35 linear) and redundantly, since the rest holds it linearly (0.90); the pair $(a_M, a_S)$, 56 classes, is read with 0.86.

2. No single coordinate is "the age". The phase of the short coordinate follows the runner-up, $\arg z_{\text{short}} \approx 43.0^\circ a_S + 155^\circ$ in both orders of arrival, and the phase of h3 follows the more recent of the top two, $\min(a_M, a_S)$ (phase concentration 0.82 and 0.81 on a scale where 1 is exact). But one phase alone gives $a_S$ with only 0.34 to 0.51, and all four coordinates together 0.94.

3. The code is used, and jointly. Making all four tape coordinates one resting step older, $c_m \to \lambda_m c_m$, moves the network's answer by exactly one position in 89% of states (unchanged in 0.6%); doing it to the short coordinate alone moves it in 5%. Putting the tape of a state with a different $a_S$ onto a state makes the network answer the tape owner's runner-up position in 78% and the rest owner's in 1.5%; with the tape set to zero it answers the runner-up in 6%.

4. Values are held weakly: $S/M$ has $R^2 = 0.56$ from the tape direction (0.00 linearly, 0.82 once the ages are given), against 0.95 from the rest; $M$ and $S$ themselves 0.45 and 0.77, against 0.996 and 0.95.

5. The tape also marks inputs close to the runner-up. Whether the input of a given age (not one of the top two) lies within $0.1M$ of $S$ is linearly readable (AUC 0.87 to 0.95; rest 0.55 to 0.63 for ages 3 and up), and the third's age is read with 0.54 (rest 0.34, both 0.71). How far below the runner-up a single input lies is recovered only roughly ($R^2$ 0.53 to 0.69 flexible, 0.14 to 0.24 linear), less than the true top-three summary alone gives (0.75).

6. Turned round: a tape rebuilt from three named quantities, $c/M = A(a_M, a_S) + B(a_M, a_S)\,S/M$ with one fitted pair of 7-vectors per age pair, explains 80% of the tape's variance. Substituted for the real tape at any step $t = 3..9$ with the run continued, it gives final accuracy 0.947 to 0.961, against 0.947 to 0.965 untouched.

So from the decoding angle: the tape stores $V = \{a_S, a_M, S/M\}$, with $a_S$ the element the rest lacks, plus side information about near-ties with the runner-up (mainly the third) that changes few answers.

Not established: full sufficiency. When the maximum or runner-up changes after the replacement, the rebuilt tape loses 2 to 4 points; $A$ and $B$ are a 56-entry lookup table, not a law; and the two simpler forms I tried (a sum of one mark per top input, or two marks that only turn with their eigenvalues) lose 1 to 10 points. I did not write `tape_v2.md`; these findings are for the synthesis step.

## Claims (11)

### 1. [strong] The age of the runner-up a_S is decodable from the tape alone and is not held by the rest of the state.

- evidence: Quiet final states (t = 9), 246,144 of 400,000 N(0,1) sequences, train 184,608 / test 61,536 (flexible decoder trained on 100,000). Accuracy for a_S (8 classes, majority 0.126): tape c/|c| 0.900 linear (multinomial logistic), 0.949 gradient boosting, 0.941 kNN; tape divided by the full-state norm 0.903 / 0.955; rest 0.348 / 0.648; full state 0.977 / 0.975. Shuffled-label control 0.124-0.126. All steps t = 3..9 pooled (429,590 quiet states from 150,000 sequences, one linear decoder not told t): tape 0.923, rest 0.405, full 0.981; per step from the tape 0.949, 0.951, 0.951, 0.953, 0.947, 0.918, 0.857.
- script: t1_decoding/d1_headline.py -> d1_headline.txt; t1_decoding/d5_where.py -> d5_where.txt (parts A, C)
- caveats: Linear accuracy falls to 0.72 for a_S = 9, to 0.64 when the third is within 0.02 M of the runner-up and to 0.82 when the maximum is within 0.05 M. Linear: 0.869 when S arrived before M, 0.932 after. Flexible decoders are lower bounds on what is present.

### 2. [strong] The network's answer follows the runner-up age stored in the tape, not anything in the rest: transplanting or shifting the tape moves the answer accordingly, and without the tape the network almost never answers the runner-up.

- evidence: Quiet final states, n = 246,144, answer = argmax U h (untouched: runner-up in 0.964). Transplant (rest of recipient + tape of donor scaled by M_rec/M_don), donor with the same a_M and a different a_S: answer = tape owner's runner-up position 0.785, rest owner's 0.015 (clipped at zero 0.763 / 0.020). Different a_M and a_S (n = 206,186): 0.748 / 0.027. Tape made k resting steps older (c_m -> lambda_m^k c_m): answer moves to p_S - k in 0.888 for k = +1 (n = 184,662; stays 0.006; clipped 0.876), 0.818 for k = +2, 0.808 for k = +3, 0.768 for k = -1, 0.662 for k = -2. When the shifted position is the maximum's own position (n = 30,587, k = +1) the network answers it in 0.959, so the rest does not veto the maximum. Tape set to zero (test states): runner-up answered in 0.057.
- script: t1_decoding/d6_functional.py -> d6_functional.txt; t1_decoding/d8_encoding.py -> d8_encoding.txt
- caveats: Final states only (immediate readout). 28-48% of the modified states have a unit below -0.01 M (off the reachable set); clipping at zero changes the numbers by 0.01-0.05, except where the shifted target is the maximum (k = +2, +3: 0.908 -> 0.750 and 0.966 -> 0.709).

### 3. [strong] A tape rebuilt from the two ages and one ratio, c/M = A(a_M,a_S) + B(a_M,a_S) S/M, is nearly functionally equivalent to the real tape; the two ages alone already give most of it.

- evidence: 300,000 sequences, quiet states of t = 3..9 pooled (858,636; fitted on the 644,033 states of the first 225,000 sequences, tested on the rest). Tape replaced at step t, run continued with the real later inputs, final answer = true second argmax, for t = 3..9: untouched 0.947, 0.953, 0.957, 0.959, 0.964, 0.965, 0.963; E3 (ages + S/M) 0.947, 0.949, 0.951, 0.953, 0.957, 0.961, 0.954; E1 (56 class-mean tapes, ages only) 0.910, 0.915, 0.921, 0.928, 0.935, 0.948, 0.949; a_S only 0.832 ... 0.902, 0.948; a_M only 0.742 ... 0.029; tape zero 0.688 ... 0.057; E4 (adds a_T and T+/M) 0.944-0.967. Variance of c/M explained on test states: E1 0.70, E3 0.80, E4 0.92 (final states alone: E1 0.59, E4 0.81). Clipping at zero changes results by under 0.01.
- script: t1_decoding/d9_encoding_continue.py -> d9_encoding_continue.txt; t1_decoding/d8_encoding.py -> d8_encoding.txt
- caveats: A and B are a lookup table (56 classes x 14 numbers), not a law. Agreement with the untouched run's own answer is only 0.93-0.98 (E3): rebuilt tapes fix some near-tie errors and add others. Split by what happens after step t (t = 3..8): no later input above S: untouched 0.967, E3 0.972; a later input between S and M: 0.947 vs 0.935; exactly one later input above M: 0.964 vs 0.925 (E4 0.936). So sufficiency is not complete when the top two change later. 27-31% of rebuilt final states have a unit below -0.01 M.

### 4. [strong] Geometry: the phase of the short coordinate is a clock for the runner-up's age in both orders of arrival; h3 (and more weakly h2 and the sign of alt) follow the more recent of the top two, min(a_M, a_S).

- evidence: Quiet final states, n = 246,144. lock = |mean of (z/|z|) e^{-i w a}| with w the mode's eigenvalue angle. short vs a_S: 0.822 (S before M 0.832, S after M 0.815); per-age resultant length 0.77-0.88; phase minus 43.0 deg x a_S stays within 145-168 deg. short vs a_M: 0.066. h3 vs min(a_M,a_S): 0.809 (0.740 / 0.893), vs a_S 0.554; h2 vs min: 0.624; alt: sign = (-1)^min(a_M,a_S) in 78% (lock 0.558). For comparison the clock coordinate (not tape) vs a_M: 0.944. Median |z_short|/M falls with a_S: 3.46 -> 1.09 (a_S = 3..9, S before M), 2.79 -> 0.71 (a_S = 2..8, S after M), about x0.8 per step.
- script: t1_decoding/d3_geometry.py -> d3_geometry.txt; t1_decoding/d4_which_age.py -> d4_which_age.txt
- caveats: The phase is concentrated, not exact: nearest-mean-phase decoding of a_S from the short phase alone gives 0.525 (majority 0.126). Amplitudes decay more slowly than the eigenvalue modulus 0.735, so the coordinate is not simply written once and left alone.

### 5. [strong] The age code is joint across the four tape modes; no single coordinate carries it, in decoding or in function.

- evidence: kNN (k = 25, 100,000 train, 61,536 test), each subset divided by its own norm, accuracy for a_S: short phase 0.514, h2 0.373, h3 0.341, alt sign 0.192; pairs 0.68-0.81; triples 0.84-0.93; all four 0.941. Pair (a_M,a_S) (majority 0.019): single 0.03-0.16, all four 0.862. Function (n = 184,662): advancing one resting step only short moves the answer one position in 0.047 (stays 0.744); only h2 0.134; only h3 0.233; h2+h3+alt 0.637; short+h2+h3 0.663; all four 0.888. By order: h2+h3+alt gives 0.776 when S arrived after M and 0.499 when before.
- script: t1_decoding/d5_where.py -> d5_where.txt (part A); t1_decoding/d6_functional.py -> d6_functional.txt (part C)
- caveats: The short coordinate has the best-concentrated phase for a_S yet the least effect when moved alone: decodable is not the same as used.

### 6. [moderate] The maximum's age and the order of arrival are also in the tape, non-linearly and redundantly with the rest; the final readout does not need them from the tape, later steps do.

- evidence: a_M (majority 0.124): tape 0.345 linear / 0.893 flexible; rest 0.904 / 0.909; full 0.992 / 0.993. S arrived before M (majority 0.499): tape 0.692 / 0.930; rest 0.759 / 0.879; full 0.824 / 0.982. Function: a tape rebuilt from a_S alone gives 0.948 at t = 9 (ages pair: 0.949) but 0.83-0.90 at t = 3..8 against 0.91-0.95 for the pair. Transplant with the same a_S and a different a_M keeps the answer in 0.806, against 0.855 with both the same.
- script: t1_decoding/d1_headline.py -> d1_headline.txt; t1_decoding/d9_encoding_continue.py -> d9_encoding_continue.txt; t1_decoding/d6_functional.py -> d6_functional.txt
- caveats: Pooled over steps the linear tape decoder of a_M gets 0.534 (0.89 at t = 3, 0.33 at t = 9).

### 7. [strong] Values are held weakly by the tape and well by the rest.

- evidence: Quiet final states, test R2. S/M: tape 0.002 linear / 0.563 flexible (0.696 with the full-state norm); rest 0.612 / 0.953; full 0.788 / 0.988. Raw M from raw coordinates (homogeneous flexible decoder): tape 0.448, rest 0.996; raw S: tape 0.769, rest 0.951; a multiple of the tape norm alone is worse than the mean (R2 -1.1 and -0.4). Given the class (a_M,a_S), per-class least squares on the raw tape coordinates (pooled t): M 0.747, S 0.855, against 1.000 and 0.995 from the rest; S/M from the tape direction per class 0.817 (median abs. error 0.065, sd of S/M 0.247).
- script: t1_decoding/d1_headline.py -> d1_headline.txt; t1_decoding/d10_structure.py -> d10_structure.txt (part B)
- caveats: The tape does depend on S/M (adding it raises the rebuilt tape from 0.91-0.95 to 0.947-0.961 final accuracy), but S/M is not well recoverable from the tape; it is a parameter of the tape rather than something the tape is the store of.

### 8. [moderate] The tape flags inputs that are close to the runner-up, mainly the third, and holds the age of an older third that the rest lacks; information about the third's value is held jointly by tape and rest.

- evidence: Per age a = 2..9, states where that input is neither maximum nor runner-up (about 46,000 test states per age, 4.6% positives): 'within 0.1 M of S' ROC AUC from the tape 0.946, 0.946, 0.946, 0.936, 0.942, 0.928, 0.920, 0.866 linear and 0.96-0.98 flexible; rest 0.55-0.63 linear for ages 3-9 (0.89 at age 2) and 0.71-0.78 flexible (0.95 at age 2); full 0.99. Age of the third (10 classes, majority 0.124): tape 0.119 linear / 0.536 flexible, rest 0.267 / 0.344, full 0.319 / 0.714; for a_T >= 3 the tape gives 0.47-0.68, the rest 0.10-0.25, the full state 0.55-0.75. T+/S: tape 0.420, rest 0.404, full 0.748 (R2, flexible). The linear a_S decoder's 6,126 errors name the third's age in 41% and the maximum's age in 36%.
- script: t1_decoding/d2_per_age.py -> d2_per_age.txt; t1_decoding/d1_headline.py -> d1_headline.txt; t1_decoding/d7_more.py -> d7_more.txt (part B); t1_decoding/d5_where.py -> d5_where.txt (part B)
- caveats: This content changes few answers: adding a_T and T+/M to the rebuilt tape raises variance explained from 0.80 to 0.92 but final accuracy by at most 0.006. A real donor tape matched on (a_M, a_S) does worse than the class-mean tape (0.85-0.88 against 0.91-0.95), and matched also on a_T, S/M in 5 bins and nearest (S-T+)/M it reaches only 0.912-0.916 at t = 9, so this content must match the rest it sits with.

### 9. [moderate] For a single older input the tape holds only a rough depth below the runner-up, less than the true top-three summary would give, though it adds to that summary for old inputs.

- evidence: Target (S - max(x,0))/M for the input of age a, states where it is neither maximum nor runner-up, test R2 for ages 2..9: tape 0.14-0.24 linear, 0.53-0.69 flexible; rest 0.42-0.54 flexible; full state 0.69-0.90 flexible; true summary V3 = (a_M, a_S, a_T, S/M, T+/S) 0.75; V3 plus tape 0.78-0.80 (ages 2-4), 0.82-0.90 (ages 5-9). Rank of the input at each age (all states): tape 0.57-0.66, rest 0.32-0.55, full 0.68-0.74 (flexible R2).
- script: t1_decoding/d2_per_age.py -> d2_per_age.txt; t1_decoding/d7_more.py -> d7_more.txt (part A)
- caveats: Consistent with the earlier finding that single tape entries are not a clean comparison with the runner-up. The gain over V3 at ages 5-9 shows some information beyond the top three exists; I did not identify what it is (the fourth largest adds 2-4 points of tape variance in d8).

### 10. [moderate] Counts, record history and controls: partly decodable counts, no decodable error signal, and the shuffled control sits exactly at baseline.

- evidence: Flexible accuracy tape / rest / full (majority): number of other inputs within 0.10 M of S (0/1/2+) 0.715 / 0.710 / 0.817 (0.555); within 0.25 M (0/1/2/3+) 0.527 / 0.527 / 0.665 (0.419); number of records 0.789 / 0.728 / 0.867 (0.325); age of the last overtaken maximum when S arrived after M (23,108 test states) 0.791 / 0.628 / 0.841 (0.531). Whether the network's answer is correct: 0.963-0.965 from every source (majority 0.963). Shuffled a_S and shuffled S/M: 0.124-0.126 and R2 between -0.001 and 0. Smallest input: its age from the tape 0.247-0.255 against 0.144 from the true top-three summary (majority 0.125); its value min/M from the tape 0.07-0.10, the same as from the summary (0.068), from the full state 0.53-0.55.
- script: t1_decoding/d1_headline.py -> d1_headline.txt; t1_decoding/d7_more.py -> d7_more.txt (part C)
- caveats: The 'smallest input' targets were meant as negative controls but are not clean nulls: the state holds some information about where the smallest input is and (in the rest) how low it is. Only the shuffled targets are true nulls.

### 11. [moderate] Negative result: the lookup table A(a_M,a_S), B(a_M,a_S) does not reduce to a sum of one mark per top input, nor to two marks that only turn with their eigenvalues.

- evidence: Same data and test as the rebuilt-tape claim, final accuracy for t = 3..9. E3 (full table): 0.947-0.961, variance explained 0.80. Additive A1(a_M) + A2(a_S) with S/M terms: 0.86-0.93, variance 0.72. Additive fitted separately per order of arrival: 0.911-0.947, variance 0.76. Two freely turning marks per order, z_m/M = lambda_m^a_S (u + u' S/M) + lambda_m^a_M (v + v' S/M): 0.872-0.939, variance 0.71.
- script: t1_decoding/d10_structure.py -> d10_structure.txt (part A)
- caveats: The simple forms still recover 87-94% correct answers, so they are a fair first approximation; the missing part is an interaction between the two ages that I did not characterise.

## Did not work

- Linear decoders across all states fail for everything except a_S and the per-age near-tie flags: from the tape direction S/M gets R2 0.002, a_M 0.345 and a_T 0.119 (majority 0.124), although flexible decoders reach 0.56, 0.89 and 0.54. The code is linear only within an age class.
- Reading a_S from a single phase: the short phase alone gives 0.51-0.53, h2 0.37, h3 0.34, the sign of alt 0.19.
- Estimating S/M as the ratio of per-class linear tape estimates of S and M blows up where the M estimate is near zero (median abs. error 0.063, R2 meaningless).
- A multiple of the tape norm as an estimate of M or S is worse than predicting the mean (R2 -1.1 and -0.4): the tape's size does not track the input scale.
- Transplanting a real donor tape matched on the decodable quantities is not a good sufficiency test: matched on (a_M, a_S) the answer is kept in 0.855, matched on (a_M, a_S, a_T), S/M in 5 bins and nearest (S-T+)/M only 0.912-0.916, against 0.964 untouched and 0.949-0.959 for class-mean or fitted tapes.
- The additive two-mark model and the freely-turning-marks model of the tape lose 1-10 points of final accuracy against the full age-pair table.
- The smallest input's age and value did not work as negative controls (above baseline from the full state); the shuffled targets did.
- The first kNN run hung on the degenerate 'alt alone' subset (only two distinct points) and was stopped and rewritten; d1_headline.py takes about 13 min and d7_more.py about 6 min because of multi-class gradient boosting, over the 3-minute guideline.

## Open

- What the remaining tape variance (20% after ages + S/M, 8-15% after adding the third and fourth) encodes, and why rebuilt tapes lose 2-4 points when the maximum or runner-up changes after the replacement (0.925-0.939 against 0.947-0.964). Candidates not tested: the precision of the later comparison with S, or third-related content needed at the handover.
- Why a class-mean tape works better than a real donor tape of the same class. The decoding results suggest the third is held jointly by tape and rest, so a donor's tape mismatches the recipient's rest, but I did not test this directly.
- A compact law for A(a_M, a_S) and B(a_M, a_S): the interaction between the two ages beyond 'two marks' is uncharacterised.
- Transplants and tape-advance tests were done on final states only; the rebuilt-tape tests cover all steps. Non-quiet states (a new maximum or runner-up in the last two steps) were not studied.
- All decoders were trained and tested on iid N(0,1) inputs; nothing was checked on uniform inputs or on controlled sequences.
- The amplitude of the short coordinate falls about x0.8 per step of a_S, more slowly than its eigenvalue 0.735, and its phase stays on a_S even after a later new maximum: what re-drives it is a question for the accounting of the writes.
- tape_v2.md was not written by this investigator; these findings are returned for the synthesis step.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/dec.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d1_headline.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d1_headline.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d2_per_age.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d2_per_age.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d3_geometry.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d3_geometry.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d4_which_age.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d4_which_age.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d5_where.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d5_where.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d6_functional.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d6_functional.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d7_more.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d7_more.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d8_encoding.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d8_encoding.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d9_encoding_continue.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d9_encoding_continue.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d10_structure.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_decoding/d10_structure.txt
