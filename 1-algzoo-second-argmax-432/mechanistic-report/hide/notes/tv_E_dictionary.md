# Tape investigation, verifier of arm E_dictionary

## Summary

All eight claims of E_dictionary reproduce with my own code, my own tape projector (agrees with tape_common.py to 1e-13) and other seeds; I found no error in its scripts (same tape definition, no fitted quantity leaks into its tests, negative-activation shares reported). Three claims need corrections of interpretation, not of numbers. (1) The functional numbers of the zero-filled "skeleton" test at t = 9 are confounded: zeroing the two newest inputs removes their direct write (about 3% of answers), while a bare (M, S) skeleton repairs 2.6% of states the network got wrong, so "answers changed" stays flat (7.2% to 4.9%) while R2 goes from 0.56 to 0.97. Redrawing the dropped inputs from their conditional distribution gives a clean ordering: (M, S) explain 75% of the tape's variance (not 56%) but a swap changes 9-11% of answers; the candidate set explains 98.9% and changes 1.7% (accuracy 96.0% against 96.4%) when the two newest inputs are kept. (2) The candidate set is not minimal: the negative early inputs in it do not matter; "positive inputs that exceeded 0.75 of the then runner-up" (3.56 per state instead of 4.79) does equally well. (3) The clean dictionary entry holds in random quiet states only with unit 5's real pulse as amplitude and only for young writes (uncentred R2 0.98, 0.95, 0.92 at ages 2-4; 0.81, 0.51, below 0 at ages 5-7); with the named amplitude 19.1 (x - 0.75 S)_+ the median per-case error is 57%. The negative results (no amplitude x pattern x eigenvalue^age form for the maximum, runner-up and dethroned inputs; non-additivity) hold at every step t = 3..9 and for uniform inputs. The changed answers concentrate in the 16% of states where runner-up and third are within 0.05 M. This angle shows what the tape depends on, not that each element can be read back, and gives no closed form for the dominant part. I did not write tape_v2.md (this step allows no .md files and only my own folder); note v2_random.py ran 5.5 minutes, above the 3-minute guideline.

## Corrected answer

Terms. A state is $h_t$, reached after the inputs $x_0..x_t$. $M_t \ge S_t$ are the largest and second largest of these inputs (maximum and runner-up), and the age of $x_p$ is $a = t - p$. A quiet state has both of them at age $\ge 2$ and $S_t > 0$. The tape is written as four complex coordinates $z_m$, $m \in \{\text{alt}, \text{short}, \text{h2}, \text{h3}\}$, with eigenvalues $\lambda_m$ (7 real numbers, since alt is real), in units of $M_t$. For an input $x_p$ let $S_{p-1}$ be the runner-up of the inputs before it. I call $x_p$ a contender if $x_p > 0.75\,\max(S_{p-1}, 0)$, with the maximum taken as 0 when fewer than two inputs came before. A quiet final state has 3.56 contenders on average, and the maximum and the runner-up are always among them.

1. What the tape depends on. The tape of a quiet state is, to 98.9% of its variance, a function of the values and ages of the contenders. This was measured by redrawing every other input from its conditional distribution. Putting the tape of such a redrawn sequence into the real state changes 1.7% of the answers (accuracy 96.0% against 96.4%), provided the two newest inputs $x_t, x_{t-1}$ are left as they are. If they are redrawn too, 4.7% change. The newest input enters linearly, $\Delta z_m = (\ell_m^\top w[o])\,x_t$ with $|\ell_m^\top w[o]| = 0.17$ to $0.31$: 0.3% of the variance, about 3% of the answers.

Smaller sets do worse:

| Set | Variance explained | Answers changed |
|---|---|---|
| maximum and runner-up | 75% | 9.0% |
| top three by value | 92% | 4.9% |
| top four by value | 97% | 2.7% |
| positive inputs in the top two on arrival (3.21 per state) | 96% | 3.9% |
| contenders (3.56 per state) | 98.9% | 1.7% |

Negative inputs older than one step do not matter, wherever they sit: setting them to 0 costs at most 0.02 of $R^2$. The share explained by maximum and runner-up alone falls with the number of other inputs: $R^2$ of the zero-filled test is 0.88 at $t = 4$ and 0.56 at $t = 9$. Changed answers concentrate where runner-up and third are within $0.05\,M_t$ (16% of quiet final states, 10 to 21% changed there).

2. The one part with a known encoding. A contender below the runner-up that arrives after both maximum and runner-up writes

$$\Delta z_m = u_5\,k_m\,\lambda_m^{\,a-2}, \qquad k_m = \ell_m^\top W[o,5] \propto (1,\ 0.55e^{-i146^\circ},\ 0.80e^{-i85^\circ},\ 0.85e^{-i53^\circ}),$$

where $u_5$ is the pulse of unit 5 one step after the input. So its age is held as the factor $\lambda_m^{a}$ and one amplitude is held on a fixed pattern.

- In controlled sequences the pattern carries 97.5% of the squared norm, and $u_5$ rises linearly above a threshold, $A = g\,(x_p - \Theta)_+$ with $g = 19.1$.
- In random quiet final states the form holds for young writes only: uncentred $R^2$ 0.98, 0.95, 0.92 at ages 2, 3, 4, then 0.81, 0.51 and below 0 at ages 5, 6, 7.
- The threshold is not a named quantity. Over random prefixes $\Theta / S_{p-1}$ has median 0.74 and a 10% to 90% range of 0.45 to 0.85. It is lower when the runner-up is small against the maximum (median 0.62 for $S/M < 0.4$). Each earlier write raises it by about 0.41 of that write's excess (10% to 90%: 0.26 to 0.76).
- With $\Theta = 0.75\,S$ put in, the formula misses single writes by 57% in the median.
- These writes are 3.5% of the tape's variance in quiet final states.

3. The rest has no dictionary. The part due to the maximum, the runner-up and the dethroned contenders (in the top two on arrival, no longer so) is not a sum of terms amplitude $\times$ fixed pattern $\times\ \lambda^{a}$.

- A lone maximum's writes cancel from $19.8\,M$ at age 1 to $3.0\,M$ at age 2, and the remainder does not follow the eigenvalues before age 6, because one wave unit is clipped at every step until then.
- The tape of a pair (maximum, runner-up) differs from the sum of the two single tapes by about its own norm; a single input leaves 5 to 17% of what the pair leaves.
- In random quiet final states the exact write terms (unit 5's pulses caused by the maximum $13.7\,M$ and by the runner-up $9.2\,M$, unit 4's pulse $8.6\,M$, wave-unit clips $7.0\,M$) are each larger than the tape ($7.9\,M$) and cancel.
- The real pulses of units 4, 5, 9 with fixed patterns give $R^2 = 0.23$ at $t = 9$ and negative values for $t \le 7$, for Gaussian and for uniform inputs. 58% of answers change at $t = 9$.
- Two late contenders are not additive either: the second difference is 0.6 to 0.9 of the smaller single write.

4. What this angle does not show. It shows sufficiency: which inputs the tape depends on. It does not show that each contender's value and age can be read back from the 7 coordinates (3.56 values and ages in units of $M_t$ are about 6 numbers, so counting does not decide it), and it gives no closed form for the dominant part of section 3. Even the contender description misses a single state's tape by 9% of its norm in the median and 26% at the 90th percentile in its zero-filled form.

In one sentence: the tape is a function of the values and ages of the three to four positive inputs that exceeded about three quarters of the then runner-up, plus a small linear trace of the newest input. That function is a sum of age-rotated writes only for late contenders below the runner-up (amplitude = unit 5's pulse). For the maximum, the runner-up and the dethroned contenders it is an order-dependent, non-additive function that has been evaluated only by running the network.

## Verdicts (8)

### [confirmed_with_correction] Claim 1 (strong). Clean dictionary entry: an input below the runner-up that arrives after both maximum and runner-up writes one fixed pattern k (unit 5's, k_m = l_m.W[o,5]) times A = g (x - Theta)_+, and from age 2 on the write is multiplied by each mode's eigenvalue per step; below Theta essentially nothing is written.

- my numbers: Controlled sequences (42 configurations, grid 0.005, my filler draws): share of squared norm on one pattern 0.974-0.975 (n = 9968 filler 0; 159488 random fillers), cosine with k5 0.996-0.998, pattern (1, 0.65@-140, 0.79@-81, 0.87@-51); slope g median 19.1 (10%..90% 15.3..19.4); Theta/S medians 0.60 / 0.72 / 0.75 (filler 0) and 0.45 / 0.65 / 0.69 (filler -1) for S = 0.3 / 0.6 / 0.9, minimum -0.10 (filler 0) and -0.41 (filler -1); write just below onset 3-7% of the write at S; free-evolution error median 0.000 from age 2, share below 0.1 0.75-1.00, 90th percentile up to 0.21. Same with S = 0.45, 0.75. RANDOM sequences: Theta/S_prev median 0.74, 10%..90% 0.45..0.85 (100940 prefixes), 64% within 0.65..0.85; median 0.62 / 0.74 / 0.78 for S/M below 0.4 / 0.4-0.7 / above 0.7. Random quiet final states, 26723 late inputs that fired unit 5: real pulse x k5 x lam^(a-2) + direct write has uncentred R2 0.954, per-case error median 12%, 90th percentile 59%; by age 2..7: R2 0.983, 0.946, 0.918, 0.811, 0.506, -0.55. Formula 19.1 (x - 0.75 S)_+ kA lam^(a-2): uncentred R2 0.566, per-case error median 57%, within 20% in 14% of cases.
- note: Every controlled-sequence number reproduces. Corrections: (a) the range of Theta extends below 0 in corner configurations, not down to 0.07 S; (b) 'multiplied by the eigenvalue from age 2 on' was shown only with non-positive fillers after the write; in random quiet states the total effect of the input is the freely evolved pulse write only for ages 2-4; (c) the amplitude is unit 5's pulse; expressed by named quantities (x_p and 0.75 S) it is a median with wide scatter and fails on single cases.

### [confirmed] Claim 2 (strong). On random quiet states the form 'amplitude x fixed pattern x eigenvalue^age' does not describe the tape; the exact writes are several large cancelling terms, and wave-unit clips are as large as the tape itself.

- my numbers: 300000 sequences, seed 4242, 184628 quiet final states. Exact split sums to the tape within 5.9e-13. rms / M: tape 7.95; unit 5's pulse caused by the maximum 13.71, runner-up 9.16, dethroned 4.91, below 1.47; unit 4 8.64; unit 9 2.36; wave clips 6.97; comparator clips 4.61; direct 0.86. Three pulse units with fixed patterns: R2 0.232 (per coordinate -0.10..0.57), answers changed 57.8% (53.2% clipped at 0; negative activation in 69%); + direct 0.264, 57.8%; all but wave clips 0.209, 62.2%. References: mean tape 91.3%, other quiet state 87.3%, zero tape 93.2%. t = 6 (133751 states): R2 -0.60, -0.53, -0.57; changed 36.0%, 35.7%, 26.1% against 48.2%. Leaving out the 'below' writes: R2 0.965, 2.1% (t = 9), 0.981, 0.6% (t = 6); leaving out the dethroned writes: 0.607, 15.3%. New: at every step t = 3..8 the three-pulse R2 is negative or 0.03 (-4.5, -1.45, -0.84, -0.60, -0.24, 0.03), and for uniform inputs on [-1, 1] it is -5.5 .. 0.30 with 30-58% of answers changed.
- note: Holds in both input distributions and at all steps. The split is relative to the map with all 13 non-pulse units on, which is the stated reference.

### [confirmed_with_correction] Claim 3 (strong). The tape of a quiet state is nearly a function of the values and ages of the inputs that were top-two candidates at their arrival (4.8 per state), plus a small linear leak of all other inputs; the top two alone give about half.

- my numbers: Zero-filled skeletons, t = 9 (184628 states): M, S R2 0.558, changed 7.2%; top 3 0.852, 6.5%; top 4 0.942, 5.5%; top 5 0.965, 5.0%; positive inputs 0.970, 5.0%; top two at arrival 0.944, 6.0%; candidates 0.972, 4.9%, with leak 0.975, 2.8%; candidates + ages 0, 1: 0.979, 2.0%; all but the newest + leak 0.999, 0.4%; after / before / between: 0.580, 0.701, 0.847. t = 6: M, S 0.787, 3.7%; top 3 0.908, 2.2%; top 5 0.952, 1.5%; mean tape 48.2%. REDRAW test (others redrawn from N(0,1) conditioned on leaving V unchanged; share of variance explained, answers changed, accuracy; real accuracy 96.4%): keeping the inputs of age 0, 1: M, S 0.755, 9.0%, 92.4%; top 3 0.916, 4.9%, 94.4%; top 4 0.971, 2.7%, 95.6%; candidates 0.989, 1.6%, 96.0%; x_p > 0.75 max(S_prev, 0) (3.56 inputs) 0.989, 1.7%, 96.0%; x_p > max(S_prev, 0) (3.21 inputs) 0.961, 3.9%, 95.0%; all positive inputs (5.05) 0.995, 1.1%, 96.3%. Redrawing the inputs of age 0, 1 as well: M, S 0.751, 11.4%, 90.0%; candidates 0.985, 4.6%, 93.7%. Zeroing only the negative candidates of age >= 2 (1.23 per state): R2 0.993, 1.0% changed; zeroing positive candidates below the then runner-up (0.21 per state, present in 19% of states): R2 0.802 on those states, 7.9% changed.
- note: Numbers reproduce. Corrections: (a) the set is not minimal, the negative early inputs in it are irrelevant; 3.56 inputs per state suffice; so the remark that the set holds more numbers than 7 coordinates can carry does not follow (3.56 values and ages in units of M are about 6 numbers). (b) 'M, S give R2 0.56' is the zero-filled predictor; values and ages of M, S explain 75% of the variance, yet a swap between states sharing them changes 9-11% of answers, more than the skeleton's 7.2%. (c) The share for M, S alone depends on the step: 0.88 at t = 4, 0.79 at t = 6, 0.56 at t = 9 (uniform inputs: 0.90 .. 0.44); 'between matter most' is true at t = 9 only (t = 6: after 0.834, before 0.874, between 0.868). (d) The leak matters functionally only for the two newest inputs; for older dropped negative inputs adding it changes R2 from 0.983 to 0.985 and answers changed from 1.9% to 2.3%. (e) Sufficiency only: nothing here shows the candidates can be read back from the tape.

### [confirmed] Claim 4 (strong). Writes are not additive: a written input raises the threshold for every later input by about 0.41 of its own excess with the slope unchanged; a pair (maximum, runner-up) is far from the sum of the two single tapes.

- my numbers: Second difference / smaller single contribution: median 0.640 (adjacent), 0.685 (2 apart), 0.926 (3+), filler 0 (n = 245, 197, 299); 0.579, 0.696, 0.915 filler -1; 0.62, 0.70, 0.90 my random fillers (n = 12930); minimum over all cases 0.28. Theta_2/S 0.779, 0.813, 0.852 for a first input at 0.8, 0.9, 0.99 S against 0.746 without, slope 19.2; beta = 0.410, R2 0.445, n = 453, ratio median 0.434 (10%..90% 0.257..0.757); ratio by gap 1..5: 0.45, 0.38, 0.43, 0.43, 0.43. Pair against sum of singles / |tape(S alone)|: 10.35, 7.37, 6.10 (S after M, S = 0.3, 0.6, 0.9) and 16.76, 13.94, 12.72 (before). Added: the same difference / |tape(M, S)| is 1.00-1.02 (10%..90% 0.87..1.26), and |tape(S alone)| / |tape(M, S)| is 0.05-0.17.
- note: The '6 to 17 times' is a small-denominator number, as the investigator says. The plainer statement: a lone input leaves 5-17% of what the pair leaves, so the pair's tape is almost entirely an interaction term. The 0.41 is a median with wide scatter (R2 of the law 0.445), measured with filler 0 and M, S in the first two positions only.

### [confirmed_with_correction] Claim 5 (strong). The maximum and the runner-up have no entry of the dictionary form: a lone maximum's pulse writes cancel within two steps and the remainder does not follow the eigenvalues until age 6; the runner-up's write has the pattern k only when it arrives at least 5 steps after the maximum.

- my numbers: Lone maximum, filler 0: |tape| = 0.23, 19.82, 3.04, 1.86, 1.35, 1.01, 0.97, 0.75, 0.58, 0.46 at ages 0..9; free-evolution error 0.691, 0.741, 0.038, 0.581 from ages 2..5, 0.000 from age 6; pulses 10.16, 12.21, 1.49; cosine of the age-2 tape with k 0.022; three pulses with fixed patterns miss by 2.17 times its norm. Cause: non-pulse units clipped at steps 1..6 are [6, 8], [0, 14], [15], [15], [11], [10], none after. Fillers -1 / random: error 0.13-0.69 at all ages. Runner-up g steps after the maximum, projection on the measured pattern, A/S and residual: g >= 5 and S <= 0.6: 16.9-18.8, 6-11%; S = 0.9: residual 34%, 19%, 12% at g = 5, 6, 7; S = 0.1: residual 9% and 7% already at g = 3, 4; g = 1, 2: residual 47-87% for every S.
- note: The lone-maximum numbers reproduce exactly. Correction to the second half: the distance at which the runner-up's write takes the pattern depends on S/M: 3 steps for S = 0.1 M, 5 steps for S = 0.3-0.6 M, and for S = 0.9 M the residual is still 12-34% at 5-7 steps. I did not recompute section B (overtaking maximum).

### [confirmed] Claim 6 (strong). For a quiet state that stays quiet, one more step is exactly 'multiply by the eigenvalues, add the direct write and the pulse writes' on 47% of steps; otherwise one wave unit is clipped, mostly when the maximum is 4 or 5 steps old.

- my numbers: 200000 sequences, seed 6161, 296088 pairs (t = 5..8): no non-pulse unit clipped 47.2%, one 44.1%, a comparator 0.5%. Pure eigenvalue multiplication: error median 9.5%, 90th percentile 42.7%; after subtracting direct and pulse writes: 1.9%, 31.5%; with one unit clipped: median 14.5%, 90th percentile 41.7%. By age of the maximum 2..8: median 0.000, 0.000, 0.238, 0.125, 0.000, 0.000, 0.000.
- note: Applies only to steps on which the state stays quiet, as stated.

### [confirmed] Claim 7 (moderate). The explicit formula for later writes (skeleton (M, S) plus g (x_p - Theta_p)_+ k lam^(age-2), Theta = 0.75 S raised by 0.44 of each excess) recovers most of those writes in R2 but does not pass the functional test.

- my numbers: 184628 quiet final states; one later write (n = 21899): R2 0.379 (skeleton M, S), 0.616 (formula), 0.695 (skeleton with the real later inputs); answers changed 11.2%, 14.1%, 8.4%; 59.3% of the later inputs' contribution reproduced. Two or more (n = 1697): 0.486, 0.770, 0.841; 15.1%, 24.5%, 9.2%; 77.9%. All quiet states: R2 0.575 against 0.558, changed 7.6% against 7.2%. t = 6, one write (n = 8328): 0.541, 0.842, 0.883; 8.1%, 7.9%, 3.3%. Per case (v5): formula error median 57%; with unit 5's real pulse as amplitude 12%.
- note: The failure is in the amplitude law (the threshold is not 0.75 S in single cases), not in the pattern. The investigator's cells with two or more writes differ a little from mine (17.4%, 25.6%, 12.1% there), consistent with n near 1100-1700.

### [confirmed] Claim 8 (moderate). Capacity under the dictionary: the 7 coordinates fix the amplitudes on the 8 age slots up to one null direction, the map is well conditioned, so a few sparse writes can be read back; the network's own readout reads back far fewer; real tapes do not follow the dictionary.

- my numbers: Singular values 2.329, 1.500, 1.331, 1.015, 0.844, 0.556, 0.383, condition number 6.1, null direction 0.19, 0.04, 0.17, 0.16, 0.40, 0.49, 0.36, 0.61. Ages read back by subset least squares for N = 1..5: 100% without noise; 99, 97, 92, 82, 59% at noise 0.3; 74, 52, 35, 21, 12% at noise 1.0. Top-N entries of the tape piece: 87, 68, 39, 25, 20% without noise. Real quiet final states (122918): correlation of slot amplitude with unit 5's pulse 0.38-0.79; largest slot is the runner-up's age in 73.7% against 84.1% for the tape piece and 97.4% for the full score; tape = Phi u has uncentred R2 -0.57.
- note: A statement about formula-generated tapes only, as the investigator says; amplitude range and noise model are arbitrary choices. It says nothing about what real tapes can hold.

## Attacks (7)

### Sufficiency by redraw instead of the filler 0. For V = (M, S), top 3, top 4, candidates: every input outside V redrawn from N(0,1) conditioned on leaving V unchanged; the redrawn sequence's tape replaces the real one. Two seeds (5151, 6161), about 123000 quiet final states each; I checked that V is unchanged after the redraw.

- result: Share of the tape's variance explained by V (1 - (1 - R2_pair)/2): M, S 0.75; top 3 0.92; top 4 0.97; candidates 0.989. Answers changed with the inputs of age 0, 1 kept / also redrawn: M, S 9.0% / 11.4%; top 3 4.9% / 7.2%; top 4 2.7% / 4.8%; candidates 1.6% / 4.6%. Accuracy (real 96.4%): 92.4 / 90.0; 94.4 / 92.3; 95.6 / 93.9; 96.0 / 93.7%.
- consequence: The claim that the tape is nearly a function of the candidates survives the test that could have refuted it, and is stronger than the skeleton suggested (0.989 against 0.972). The zero-filled skeleton understated what M and S determine in variance (0.56 against 0.75) and overstated how well they do functionally (7.2% against 9-11%).

### Split of the pooled functional number by the gap (S - T)/M between runner-up and third, with accuracy besides 'answers changed' (t = 9, zero-filled skeletons).

- result: States with gap below 0.05 (16% of quiet states; the network itself is right in 82.7% of them): changed 21.4% (M, S), 20.5% (top 3), 14.9% (candidates), 10.5% (candidates + leak). Other bins: 3.5-6.3%, 2.6-6.5%, 2.2-4.9%, 1.1-2.3%. The M, S skeleton turns 4.3% of states from right to wrong and 2.6% from wrong to right (the network errs in 3.6-3.7%); candidates 4.0% and 0.8%. Accuracy after substitution: M, S 94.7%, top 3 93.0%, candidates 93.2%, real 96.4%.
- consequence: The pooled 5-7% hides that about half of the changes sit in near-ties of runner-up and third. With filler 0, 'answers changed' is not a monotone measure of how good a description is: the bare (M, S) skeleton scores best on accuracy because it removes the competitor. The investigator's ordering of the skeletons by R2 stands; its functional column at t = 9 should be replaced by the redraw numbers.

### Other populations: every step t = 3..9 with the run continued, and uniform inputs on [-1, 1] (300000 sequences each).

- result: N(0,1), t = 3..9: M, S skeleton R2 0.84, 0.88, 0.83, 0.79, 0.73, 0.65, 0.56; candidates 0.84, 0.945, 0.953, 0.958, 0.970, 0.976, 0.972; changed (candidates + leak) 0.9-2.8% against 14-91% for the mean tape. Uniform: M, S 0.90 .. 0.44; candidates 0.90, 0.97, 0.97, 0.92, 0.94, 0.98, 0.98. Three pulses with fixed patterns: R2 negative for t <= 7 in both.
- consequence: Claims 2 and 3 are not artefacts of t = 9 or of Gaussian inputs. 'The top two alone give about half' holds only at t = 9; earlier, with fewer other inputs, they give 0.8-0.9.

### Is the candidate set the right set? Leave one class of inputs out of the real sequence (ages >= 2), and a sweep of the criterion x_p > f max(S_prev, 0).

- result: Zeroing negative candidates (1.23 per state): R2 0.993, 1.0% changed. Zeroing positive non-candidates (0.86 per state): 0.995, 0.8%. Zeroing dethroned inputs: 0.642. Zeroing candidates that were below the then runner-up (0.21 per state): R2 0.971 overall, 0.802 on the 19% of states that have one, 7.9% changed there. Sweep (inputs kept; variance explained; changed, newest two kept): f = 1.0: 3.21, 0.961, 3.9%; 0.9: 3.34, 0.977, 2.4%; 0.75: 3.56, 0.989, 1.7%; 0.6: 3.81, 0.992, 1.5%; 0.5: 3.99, 0.993, 1.4%; 0: 5.05, 0.995, 1.1%. Top 4 by final value: 4.00, 0.971, 2.7%.
- consequence: The investigator's set can be cut from 4.79 to 3.56 inputs without loss. The criterion 'at arrival above about 0.75 of the then runner-up' is more economical than rank by final value (3.56 inputs beat the top 4), and the knee of the sweep sits near 0.75, matching unit 5's median threshold. Its counting argument (more numbers than 7 coordinates) falls.

### The clean dictionary entry in random quiet final states (58434 late inputs between 0.5 S and S, arrived after both M and S; write = tape(x) - tape(x with that input set to 0)).

- result: Unit 5 fired for 46%: by x_p/S 6% (0.5-0.6), 13%, 33% (0.7-0.75), 58% (0.75-0.8), 89%, 100% (0.9-1). Where it fired: real pulse x k5 x lam^(a-2) + direct write gives uncentred R2 0.954, per-case error median 12%, 90th percentile 59%; by age 2..7: 0.983, 0.946, 0.918, 0.811, 0.506, -0.55. With a later input above 0.5 S: 0.921, median error 21%; without: 0.971, 9%. Formula with Theta = 0.75 S: R2 0.566, median error 57%.
- consequence: The pattern and the eigenvalue evolution hold outside controlled sequences for young writes (ages 2-4) but not for old ones, so 'from age 2 on' is too strong for real states. The amplitude is only as good as unit 5's real pulse; 'excess over three quarters of the runner-up' is a median, not a law.

### Unit 5's threshold in random sequences (sweep of x_p in 100940 random prefixes; firing rate of unit 5 for 749358 natural inputs below the then runner-up).

- result: Theta/S_prev median 0.74, 10%..90% 0.45..0.85; 64% within 0.65..0.85; never on below S_prev in 1.2%, already on at -0.5 S_prev in 1.7%. By S_prev/M_prev below 0.4 / 0.4-0.7 / above 0.7: median 0.62 (10%..90% 0.00..0.82) / 0.74 / 0.78. Firing rate by x_p/S_prev: 0.6-0.7 17.5%; 0.7-0.75 38%; 0.75-0.8 61%; 0.8-0.9 89%; 0.9-1 99.4%.
- consequence: Confirms 'about 0.75 S in the median, not a fixed multiple' in the natural population; the transition is soft, between 0.6 and 0.9 of the runner-up, and lower when the runner-up is small against the maximum.

### Controls on the metric: is the result special to the tape, and does the 1/M normalisation drive the R2?

- result: Same skeletons scored on the rest piece of the scores (entries of age >= 2, centred): candidates 0.785, positive inputs 0.790, against 0.970 for the tape piece. R2 without dividing by M: 0.556 (M, S), 0.860 (top 3), 0.987 (candidates); states with M above its median: 0.548, 0.859, 0.989. Per-state |error| / |tape|: median 0.49, 0.23, 0.09; 90th percentile 1.24, 0.73, 0.26.
- consequence: The R2 values are not produced by small-M states. The tape is more nearly a function of the candidates than the rest of the state is; that comparison is confounded by the newest input, which the comparators in the rest read directly and the skeleton zeroes, so it is weak evidence. Even the best skeleton misses a single state's tape by 9% in the median and 26% at the 90th percentile.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v_common.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v_common.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v1_controlled.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v1_controlled.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v1_patterns.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v2_random.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v2_random.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v3_attacks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v3_attacks.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v4_more.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v4_more.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v5_entry_in_random_states.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_E_dictionary/v5_entry_in_random_states.txt