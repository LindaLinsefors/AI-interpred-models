# Round-1 arm E_logit_sources: structured result

## Summary

**findings.md was NOT written**: the harness refused the write ("subagents should return findings as text, not write report files"), so this object is the findings. All scripts and their raw outputs are in `c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/`; ready-made markdown tables are in `e2_markdown_output.txt`, `e3_markdown_output.txt`, `e3_composite_output.txt`. Seeds 1000–1150, all N(0,1) unless stated.

## Conventions
Amplitudes: raw $x_s$; pulses $h_s[u]$, $u\in\{4,5,9\}$; clips $c_s[i]=\max(-z_s[i],0)$ for the 13 ordinary units. Age $a=9-s$. Kernels $K^x(a)=U_OA^aw_O$, $K^u(0)=U_{\cdot u}$, $K^u(a)=U_OA^{a-1}W_{Ou}$, $K^{c,i}(a)=U_OA^ae_i$. Classes: X, P4, P5, P9, C1, C6, C7, C8 (comparator clips), CW (wave-unit clips). Role of an event = the input that caused it ($x_s$ for X, P4, clips; $x_{s-1}$ for P5, P9). RM = runner-up arrived before the max; MR = after. Composites: CLIP = X + C1 + C6 + C7 + C8, MARK = P4 + CW, TAPE = P5.

## E1
$h_9[O]=\sum_s A^{9-s}[w_Ox_s+W_{OP}p_{s-1}+c_s]$ and logits $=\sum$ kernel × amplitude hold to $1.5\times10^{-13}$ ($h_9$) and $8.9\times10^{-13}$ (logits), n = 200k; also on Unif(−1,1). Without the clip term the error is 48.9. $c_s[j]=|w_j|(x_s-\theta_j(s))_+$ to $4\times10^{-14}$.

## E2 (exact; full tables in `e2_kernels_output.txt`)
- Tapes (extreme entry at the causing input's position at every age, no decay): pulse 4: $-4.99,-6.63$ at ages 0, 1, then $-3.18$ to $-4.12$; pulse 5: $+3.37$ to $+4.16$ (ages 0–8).
- Raw input: $+57.19$ on position 9 at age 0 ($-3.8$ to $-4.9$ elsewhere), then $\le1.01$ (97% lost in one step).
- Pulse 9: not local at own; trough $-2.8$ to $-3.4$ one position before the causing input for ages 2–7.
- Single comparator clips per unit excess: travelling waves, max $|K|$ 40–120 at ages 0–4 and 18–31 at age 9, pattern fixed relative to the event (comparator 7: trough at own−2, peak at own−6; comparator 1 the reverse; 6: trough own−4/−5; 8: peak own−4, trough own−8). Not local.
- Sum of the four: $-58.25$ on position 9 at age 0, $\le2.08$ for ages $\ge1$; exactly $\sum_j\tilde K^j+K^x=U_OA^aw_{wave}$. With all four clipped, raw + clips $=x\,U_OA^aw_{wave}-\sum_jb_jU_OA^ae_j$, $b_j=W_jh_{s-1}$: the size of a new max leaves no trace; the clip leaves minus the comparators' stored drive (threshold differences).
- Wave-unit clips: troughs at fixed offsets: unit 0 at own−2 ($-2.8$ to $-3.5$), 12 at own−4, 11 at own−5 ($-4.6$ to $-5.4$), 10 at own−6.

## E3 (n = 524288, seeds 1010–1017; classes sum to the exact logits, error $8.3\times10^{-13}$)
No single class creates a margin: rms of each class's contribution to the margin vs best other is 18–48 (P9: 5.2) against 13.7 for the margin. Mean CLIP / MARK / TAPE / P9 = total:

| case | runner-up − max | runner-up − third | runner-up − best other |
|---|---|---|---|
| all | 5.4 / 36.3 / −23.3 / −0.6 = 17.8 | 16.2 / −12.7 / 17.3 / 0.5 = 21.3 | 6.4 / 4.3 / −0.6 / 0.2 = 10.2 |
| RM | 25.7 / 5.0 / −14.3 / 1.8 = 18.2 | 29.3 / −27.4 / 18.5 / 1.6 = 22.0 | 20.8 / −16.7 / 3.6 / 1.4 = 9.0 |
| MR | −14.8 / 67.3 / −32.2 / −3.0 = 17.3 | 3.1 / 2.0 / 16.1 / −0.6 = 20.6 | −8.0 / 25.2 / −4.8 / −1.0 = 11.4 |
| RM, max last | −5.4 / −21.4 / 60.1 / 1.6 = 34.9 | | |
| RM, max age 2–4 | 33.8 / 5.1 / −29.5 / 2.5 = 11.8 | | |
| MR, runner-up last | 21.6 / 90.3 / −89.8 / −2.2 = 19.8 | | |
| MR, runner-up age 2–4 | −22.6 / 63.2 / −19.2 / −3.2 = 18.3 | | |

- MR: the record mark decides (MARK $+67.3$, positive in 99.7% of sequences; max carries $-71.5$, runner-up $-4.2$).
- RM: the comparator clips at the new max's arrival decide (CLIP $+25.7$, positive in 85.4%); they put $+26.5$ on the old max's position. MARK is neutral on average.
- vs third: TAPE and CLIP, against MARK in RM.
- P4 + P5 at the max's own position is positive ($+15.6$ RM, $+21.0$ MR): pulse 4 alone does not push the max down.
Checks: (a) RM, max at steps 2–8 (n = 101778): the largest entry of the max-step X + C events among earlier positions is at the old max in 88.9% (chance 20%); mean $+27.4$ there. (b) Bump size: $R^2=0.002$ on $M_{new}-M_{old}$, slope 42.2 with $R^2=0.53$ on old max − old runner-up. (c) Clock read through wave units alone: $+35$ to $+61$ logits at the max's position; through comparators $-35$ to $-54$. (d) Wave unit 0 is clipped with $P=0.998$ when the running max has age exactly 2 ($\le0.007$ at ages 1, 3, 4, 5); CW at the max's position: $-0.6,-12.7,-15.1,-20.3,-26.1,-29.6$ at max ages 1–6. (e) Removing terms from the exact sum: CLIP of the max → RM accuracy 0.070; MARK → MR accuracy 0.003; TAPE of the runner-up → 0.003 (RM), 0.19 (MR).

## E4
$$L_9=-49.90(\theta_7-x_9)_+-7.38(\theta_6-x_9)_+-5.01(\theta_1-x_9)_+ +4.04(\theta_8-x_9)_+-50.70(x_9-\theta_4)_++\rho_9$$
exact ($2.5\times10^{-13}$), $\rho_9$ = wave units and pulses 5, 9. $\partial(L_9-L_p)/\partial x_9=61.0$–$62.1$ for all $p\le8$ below the thresholds; $-54.6$ to $-57.8$ above all of them. Position 8: $h_9[5]$ gains $18.3\,x_8$ (7.8 between $\theta_8,\theta_7$; 12.1 for a record), times $U_{8,5}=3.40$ = 62.1 logits per unit: same currency as position 9. Sweeps (20000 prefixes): the network answers 9 on one interval whose edges are $S_8$ (slope 1.013, $R^2=0.990$, median error 0.021) and $M_8$ (slope 0.998, $R^2=0.996$, 0.013); same for position 8 (edges at $g=0.00$ and $0.99$).

| intervention | all | max last | runner-up last | runner-up at 8 |
|---|---|---|---|---|
| none | 0.953 | 0.935 | 0.953 | 0.944 |
| $U_{9,4}=0$ | 0.938 | 0.775 | 0.964 | 0.913 |
| $U_{9,4}=+0.5$ | 0.915 | 0.543 | 0.965 | 0.854 |
| $U_{8,5}=0$ | 0.864 | 0.840 | 0.960 | 0.017 |
| $U_{9,7}=0$ | 0.548 | 0.934 | 0.983 | 0.525 |

## E5
Exact kernels (n = 262144): all but P9: agreement 0.886, task 0.875; all but X 0.869; all but CW 0.456; all but P4 0.353; all but P5 0.118; X + P4 + P5 + comparators 0.489. Not reducible to 3–5 classes. Events of every age matter (age $\le A$: 0.23, 0.32, 0.40, 0.48, 0.55, 0.62, 0.70, 0.74, 0.80, 1.00). Refitted kernels, held out: X + P4 + P5 + 4 comparators: agreement 0.909, task 0.897; plus wave clips: 0.984, 0.950. Effective algorithm: (1) tape write, unit 5: $h_{s+1}[5]=18.0(x_s-\theta_5)_+$, $\theta_5\approx0.74M_2^++0.13M_3^+$ ($R^2=0.958$), $+3.4$ to $+4.2$ per unit on that position; (2) record mark, unit 4 ($-3.2$ to $-4.1$ per unit, $h[4]\approx9.90(x_s-M_{s-1}^+)+0.44$) plus wave-unit clips from age 2; (3) clock-addressed bump from comparator clips; (4) current input read directly ($+57.2x_9$). No closed-form model was built.

## E6 (n = 524288, seeds 1060–1067)
Rms of contributions to logits centred over positions 0–8 (total 22.96): level 1.68 (7.3%), clock 4.19 (18.2%), level + clock 4.43 (19.3%; 3.7% of the sum of squares). All 13 ordinary units on at $t=9$ (42.9% of sequences): 22.8% / 5.2%; otherwise 17.5% / 3.1%. Removing level + clock changes the best position among 0–8 in 9.4% of sequences (clock 8.0%, level 3.6%). Carriers: short (rms ratio 1.26), trans (0.91), h2 (0.54), h3 (0.42), pulses at $t=9$ (0.33). Position 9 sees the level: $R_{level}[9]$ is 2.37 above the mean of positions 0–8.

## Claims

### E1.1 [MIXED, confidence high]

The source identity is exact: h_9[ORD] = sum_s A^(9-s) [w[ORD] x_s + W[ORD,PULSE] p_(s-1) + c_s], and the logits equal the sum over events of (fixed kernel) x (amplitude), where all amplitudes except the raw inputs are >= 0. Comparator clip amounts equal |w_j| max(x_s - theta_j(s), 0) and unit 4 equals w_4 max(x_s - theta_4(s), 0).

- numbers: N(0,1), n=200k, seed 1000: max abs error 1.5e-13 on h_9 (max |h_9| 122), 8.9e-13 on logits (max |logit| 312). Unif(-1,1), seed 1001: 5.7e-14 and 3.5e-13. Closed forms of comparator clips and unit 4: <= 3.7e-14. Control without the clip term: error 48.9. Rms of class sums: raw 18.6, pulses 19.0, clips 20.7; logits 30.0.
- how checked: e1_identity.py (state identity, logit identity through the kernel arrays, and the position-selecting code path). The same kernel arrays are re-tested by the additivity checks in e3_aggregate.py (8.3e-13), e5_*.py and e6_modes.py (7.7e-13).

### E2.1 [EXACT, confidence high]

Raw-input kernel: at age 0 the current input is written on position 9 only (+57.19 per unit x_9; -3.78 to -4.90 on positions 0-8; 49.9 of it is U[9,7] w_7). One step later 97% is gone (centred norm 58.5 -> 1.65); for ages 1-9 all entries are <= 1.01 in magnitude, with +0.63 to +1.01 at the own position.

- numbers: K_x(0) = (-4.32,-4.47,-4.45,-4.39,-4.60,-4.61,-4.69,-4.90,-3.78,+57.19); centred norms by age 58.5, 1.65, 1.64, 1.50, 1.47, 1.54, 1.70, 1.59, 1.58, 1.45; max |A w[ORD]| = 0.43, cosine(w[ORD], echo eigenvector) = 0.998.
- how checked: e2_kernels.py, e2_markdown.py from the weights; kernel array is the one used in the E1 identity.

### E2.2 [EXACT, confidence high]

Pulse-4 kernel is a negative tape: per unit h_s[4], the position of the input that fired it is the most negative of the ten entries at every age, with no decay over 9 steps. Side structure: -1.4 to -1.8 at own+1 (ages >= 2), +0.3 to +2.1 on earlier positions.

- numbers: Own-position values for ages 0..9: -4.99, -6.63, -3.48, -3.84, -3.82, -3.97, -4.12, -3.88, -3.32, -3.18. Centred norm 4.1-6.9 at all ages. Largest other entry in magnitude <= 2.12.
- how checked: e2_kernels.py (rank of the own entry among the 10 positions is 'smallest' at all 10 ages); used in E1.
- caveats: Ages 8 and 9 (-3.32, -3.18) fall slightly outside the lead's stated range -3.5..-4.1.

### E2.3 [EXACT, confidence high]

Pulse-5 kernel is a positive tape: per unit h_s[5], the position of the input one step before the pulse (s-1) is the largest of the ten entries at every age 0-8, with no decay. +0.85 to +1.37 at own+1; -0.4 to -2.4 on the two to four positions before.

- numbers: Own-position values for ages 0..8: 3.40, 3.37, 3.66, 3.82, 3.90, 4.16, 3.95, 3.59, 3.67. Centred norm 3.6-5.5.
- how checked: e2_kernels.py; used in E1. Timing checked on data (s01_amplitudes.py, n=400k, seed 1002): unit 5 fires at s+1 after 95.4% of rank-1, 77.8% of rank-2, 19.2% of rank-3 and 0.9% of lower inputs (positive inputs only: 100%, 99.9%, 38.9%, 4.5%).

### E2.4 [EXACT, confidence high]

Pulse-9 kernel is not local at the causing input's position (own entry between -1.66 and +1.62). Its stable feature is a trough one position BEFORE the causing input for ages 2-7, with +1.9 to +2.9 three to four positions before; it fades after age 5.

- numbers: Entry at own-1 for ages 0..7: +0.22, -1.92, -2.80, -3.02, -3.37, -3.14, -3.07, -3.17. Centred norm 5.9 (age 1) -> 2.1 (age 8).
- how checked: e2_kernels.py, s06_verify_lead_facts.py. Unit 9 fires one step after 93.5% of rank-1 inputs and 3.0% of rank-2 inputs (s01_amplitudes.py).
- caveats: The lead's 'about -3.1 one position before' holds only for ages 2-7.

### E2.5 [EXACT, confidence high]

Single comparator clip kernels (per unit of excess x_s - theta_j) are not local: each is a travelling wave over positions, fixed relative to the event position, with magnitudes far above every other kernel. Comparator 7: trough at own-2, peak at own-6; comparator 1: peak at own-2, trough at own-6; comparator 6: trough at own-4/-5; comparator 8: peak at own-4, trough at own-8. The entry at the own position is small relative to the wave and changes sign with age.

- numbers: Max |K| over positions, ages 0-4: comp 1: 61-70, comp 6: 58-66, comp 7: 97-120, comp 8: 40-96; at age 9: 24, 18, 31, 29. Centred norms 122-184 at age 0, 33-45 at age 9. Own-position values comp 7: -50,-33,-33,-24,-6,+3,+16,+14,+17,+19; comp 8: +4,+35,+19,+3,-3,+5,+18,+19,+20,+23.
- how checked: e2_kernels.py, e2_markdown.py, e2_summary.py (mode content: dominated by the 'short' mode at young ages, h2 and runner at old ages).

### E2.6 [EXACT, confidence high]

The sum of the four comparator kernels (all four clipped by a common excess) is minus the raw-input kernel up to the wave units' small input weights: it cancels the raw input's +57.19 on position 9 at age 0 and is negligible afterwards. Consequently, when all four are clipped, raw + comparator-clip sources of a step equal x_s U_O A^a w_wave - sum_j b_j U_O A^a e_j with b_j = W[j].h_(s-1): the size of the new input leaves almost no trace; the clip leaves minus the comparators' stored drive.

- numbers: Sum kernel at age 0: -58.25 on position 9, +2.7 to +11.5 elsewhere; ages >= 1: max |entry| 2.08 (<= 4% of a single comparator kernel). Identity sum_j K_j + K_x = U_O A^a w_wave: error 5.7e-14. On data (89.2% of 102044 RM max-arrival steps have all four clipped): identity error 1.8e-13; rms of the x-term 1.7, of the stored-drive term 20.2.
- how checked: e2_summary.py (kernel identity), e3_bump.py (data identity, seeds 1150-1151).

### E2.7 [EXACT, confidence high]

Wave-unit clip kernels are not local at the own position (|K_own| <= 1.7) but have a trough at a fixed offset before the clip step: unit 0 at own-2, unit 12 at own-4, unit 11 at own-5, unit 10 at own-6. Unit 2's largest entry sits on position 9 at every age >= 1.

- numbers: Unit 0: -2.8 to -3.5 at own-2 (ages 0-7). Unit 12: -2.5 to -2.8 at own-4. Unit 11: -4.6 to -5.4 at own-5 (ages 0-4). Unit 10: -3.9 to -4.8 at own-6 (ages 0-3). Unit 2: -2.2 to -3.0 on position 9.
- how checked: e2_kernels.py, e2_markdown.py.

### E3.1 [MEASURED, confidence high]

No single source class creates a margin. Each class's contribution to the margin (runner-up minus best other position) fluctuates more than the margin itself, and single comparator classes change sign with the ages. The nine classes sum exactly to the logits.

- numbers: n=524288, seeds 1010-1017, accuracy 0.9535. Rms of contribution to margin vs best other: X 26.4, P4 29.6, P5 39.5, P9 5.2, C1 35.9, C6 31.3, C7 48.2, C8 44.9, CW 18.1; margin itself: mean 10.2, median 8.2, rms 13.7. Means for (runner-up - max): X -3.8, P4 +25.2, P5 -23.3, P9 -0.6, C1 +6.7, C6 +3.0, C7 +7.4, C8 -7.9, CW +11.1, total 17.8. Additivity error 8.3e-13.
- how checked: e3_aggregate.py, e3_tables.py, e3_markdown.py (mean and median per class for L[runner-up], L[max], L[third] and three margins; split RM/MR and by ages).
- caveats: Full per-class tables with medians and by-age splits are in e3_markdown_output.txt and e3_tables_output.txt, not repeated here.

### E3.2 [MEASURED, confidence high]

When the runner-up arrives AFTER the max (MR), the margin runner-up minus max is created by the record mark (pulse 4 plus wave-unit clips), which only the max carries; the pulse-5 tape and the comparator clips work against it.

- numbers: MR, n=262833: mean contributions to L[runner-up]-L[max]: MARK (P4+CW) +67.3 (P4 +44.5, CW +22.8), TAPE (P5) -32.2, CLIP (X + comparator clips) -14.8, P9 -3.0, total +17.3. MARK is positive in 99.7% of MR sequences. MARK at the max's position -71.5, at the runner-up's -4.2. TAPE: max's own pulse-5 bump 66.7, runner-up's 25.3. Stable over runner-up ages 1-8: MARK +59 to +63.
- how checked: e3_composite.py; additivity of composites checked (3.7e-5 in float32 cache). Ablation in the exact sum (e3_ablate.py, n=262144): removing MARK gives MR accuracy 0.003 (answers the max in 95.9%); removing only CW: 0.234; removing P4 of the max: 0.056.
- caveats: The grouping into composites is my choice; the ablation holds later amplitudes fixed (direct effects).

### E3.3 [MEASURED, confidence high]

When the runner-up arrived BEFORE the max (RM, it was the max until then), the margin runner-up minus max is created by the raw-input + comparator-clip sources (CLIP): the new max's clipping deposits a bump on the old max's position. The record mark is neutral on average (both are marked records) and the pulse-5 tape works against it.

- numbers: RM, n=261455: CLIP +25.7 (positive in 85.4% of sequences), MARK +5.0, TAPE -14.3, P9 +1.8, total +18.2. Events of the max's step put +26.5 of CLIP on the runner-up's position. By max age: CLIP +33.0, +33.8, +33.5 for ages 1, 2-4, 5-8; MARK +10.8, +5.1, +24.3; TAPE -26.8, -29.5, -44.0.
- how checked: e3_composite.py. Ablation (e3_ablate.py): removing the CLIP events caused by the max drops RM accuracy 0.956 -> 0.070 (MR stays 0.919).
- caveats: Exception at the last step: see E3.9.

### E3.4 [MEASURED, confidence high]

The runner-up beats the third mainly through the pulse-5 tape and (in RM) the comparator-clip bump, against the record mark that the runner-up carries in RM.

- numbers: L[runner-up]-L[third], CLIP / MARK / TAPE / P9: all +16.2 / -12.7 / +17.3 / +0.5 = 21.3; RM +29.3 / -27.4 / +18.5 / +1.6 = 22.0; MR +3.1 / +2.0 / +16.1 / -0.6 = 20.6. Margin vs best other: RM +20.8 / -16.7 / +3.6 / +1.4 = 9.0; MR -8.0 / +25.2 / -4.8 / -1.0 = 11.4.
- how checked: e3_composite.py, n=524288.

### E3.5 [MEASURED, confidence high]

Pulse 4 and pulse 5 nearly cancel at a record's own position, with pulse 5 the larger: the max's own pulses give the max's position a net positive logit. Pulse 4's negative own-position kernel therefore does not by itself put the max below the runner-up.

- numbers: n=262144 (seeds 1080-1081). P4+P5 caused by the max, read at the max's position: RM -27.8 + 43.4 = +15.6; MR -45.7 + 66.7 = +21.0. Caused by the runner-up at the runner-up's position: RM -25.4 + 42.0 = +16.5; MR -0.5 + 25.3 = +24.7. For records h5(s+1) ~ 1.21 h4(s) + 7.0 M_prev+ + 0.4 (R^2 0.915, n=503958).
- how checked: e3_checks.py (check 3), s05_tape_amplitude.py.

### E3.6 [MEASURED, confidence high]

In RM the bump from the max-step sources (raw input + comparator clips at the step where the max arrives) lands on the position of the old max (the final runner-up). In MR the same sources at the runner-up's step point at the max's position: the bump goes to the position of the current max, whoever clips.

- numbers: RM, max at steps 2-8, n=101778: among earlier positions the largest entry is at the runner-up's position in 88.9% (chance 20.0%), within +-1 position in 97.9%; mean value there +27.4 (median +23.4), -2.9 on other earlier positions, +2.0 on the max's own position. By distance p1-p2 = 1..8: 96.8, 96.4, 84.5, 77.3, 87.7, 82.8, 85.4, 95.4%. MR control: points at the max in 71.7%, mean +18.4.
- how checked: e3_checks.py (check 1), seeds 1080-1081.

### E3.7 [FIT, confidence medium]

The size of that bump is set by the stored state (the gap between the old max and the old runner-up, i.e. the clock amplitude), not by the size of the new max.

- numbers: RM steps with all four comparators clipped (about 91k sequences, seeds 1150-1151). Bump = value at old max's position minus mean over other earlier positions: mean 28.9. Regression on M_new - M_old: coefficient -2.5, R^2 = 0.002. On M_old - S_old+: slope 42.2, intercept 9.2, R^2 = 0.533; by distance d=1..8 (no intercept) slopes 37, 60, 50, 57, 69, 75, 89, 105 with R^2 0.21, 0.53, 0.55, 0.89, 0.91, 0.75, 0.72, 0.68. Correlation with |z_clock| before the max: 0.68.
- how checked: e3_bump.py; consistent with the exact identity in E2.6.
- caveats: R^2 is only 0.53 pooled; I did not derive the bump in closed form from the clock coordinate.

### E3.8 [MIXED, confidence medium]

The clock is read through the wave units as a bump on the max's position and through the comparators as the opposite; the two cancel. Clipping the comparators removes their stored part and leaves the wave-unit part, which explains why the clip bump sits on the max's position.

- numbers: Exact: clock readout via wave units 2|R| = 8.2, 9.5, 9.2, 7.7, 7.5, 7.8, 6.8, 5.7 at positions 0-7 with phase advancing 44.2 deg/position (clock rotation 42.75 deg/step); via comparators same size, opposite phase; |sum|/(|a|+|b|) = 0.0575. Measured (n=400k, seed 1110), max age a=2..9: wave-unit part at the max's position +35, +40, +46, +47, +50, +57, +61, +51; comparator part -35, -38, -44, -45, -47, -53, -54, -42; sum +0.3 to +9.4. Wave-unit part has its maximum at the max's position in 74% of sequences with a >= 2 (87-93% for a = 4..7).
- how checked: e6b_clock_split.py. Clock phase per age refitted on the same data: 0.6, 36.7, 80.1, 123.3, 166.3, 208.9, 250.9, 293.8, 337.7 deg for a=1..9.
- caveats: The link 'clipping un-cancels the clock' is supported by the exact identity (E2.6), the bump location (E3.6) and this split, but I did not show quantitatively that the bump equals the wave-unit clock term.

### E3.9 [MEASURED, confidence high]

The last step is an exception because the newest input has no pulse-5 bump yet (unit 5 fires one step later). With the max last (RM) the pulse-5 tape is the source of the margin over the max; with the runner-up last (MR) the tape is strongly against and the record mark carries the margin.

- numbers: RM, max age 0 (n=52442): CLIP -5.4, MARK -21.4, TAPE +60.1, P9 +1.6 = +34.9. MR, runner-up age 0 (n=52314): CLIP +21.6, MARK +90.3, TAPE -89.8, P9 -2.2 = +19.8.
- how checked: e3_composite.py (by-age rows).

### E3.10 [MEASURED, confidence high]

The wave-unit clips are delayed record marks, phase-locked to the age of the running max: unit 0 is clipped when the running max is exactly two steps old and (almost) never at other ages, and its kernel trough is two positions back, i.e. on the max. Units 12, 11, 10 do the same at ages 4, 5, 6. The mark deepens with the max's age.

- numbers: n=400k, seed 1140. Unit 0: P(clipped | running max age 2) = 0.998 (n=427k states, mean amount 3.98); ages 1, 3, 4, 5: 0.007, 0.003, 0.001, 0.001; age 0: 0.18 (0.84 if x_(t-2) was a record, 0.002 otherwise). Amount ~ 0.17 h5 + 1.05 h9 - 0.81 of the record's next-step pulses (R^2 0.76). Unit 12 at age 4: 0.32; unit 11 at age 5: 0.94; unit 10 at age 6: 0.96 (other ages <= 0.04). Wave-clip contribution at the max's position (MR): -0.6, -12.7, -15.1, -20.3, -26.1, -29.6 for max ages 1-6, about -27 for 7-9. Its minimum over positions is at the max in 86.2% of MR sequences. Mean at record positions -6.6, at non-record positions +2.9.
- how checked: s07_wave_clips.py, e3_checks.py (check 2). Independent confirmation: in the refitted model without wave clips (E5.3) the pulse-4 own-position kernel deepens to -5.6..-12.1 for ages 2-7.
- caveats: With the role rule used in E3, these delayed clips are filed under 'other inputs' (CW 'other' = +16.2 in the MR margin).

### E3.11 [MEASURED, confidence high]

Removing single (composite, role) terms from the exact sum confirms which terms are necessary in each scenario.

- numbers: n=262144, seeds 1120-1121; baseline RM 0.956, MR 0.952. Accuracy RM / MR after removal: CLIP caused by max 0.070 / 0.919; CLIP caused by runner-up 0.892 / 0.657; MARK all 0.594 / 0.003; P4 of max 0.227 / 0.056; P4 of runner-up 0.989 / 0.960; CW all 0.672 / 0.234; TAPE of runner-up 0.003 / 0.192; TAPE of max 0.914 / 0.463; P9 all 0.822 / 0.930.
- how checked: e3_ablate.py part (i).
- caveats: Direct-effect ablation: later amplitudes are held at their measured values; this is not a rerun of the network.

### E3.12 [INTERVENTION, confidence medium]

In the running network, not rectifying the four comparators at the single step where the final max arrives destroys the answer in RM.

- numbers: Max at steps 1-8, n about 105k per scenario: RM accuracy 0.961 -> 0.125, mean L[runner-up]-L[max] 14.1 -> -4.5; MR accuracy 0.956 -> 0.430 (mean margin vs max 16.7 -> 75.4). Comparator 7 only: RM 0.271, MR 0.589. Comparators 1, 6, 8 only: RM 0.202, MR 0.621.
- how checked: e3_ablate.py part (ii).
- caveats: Blunt: it lets large negative activations into the recurrence, so only the direction of the RM effect should be read.

### E4.1 [EXACT, confidence high]

Exact last-step form: with b = W h_8, theta_j = b_j/|w_j| (comparators), theta_4 = -b_4/w_4, every logit is L_p = sum_j U[p,j]|w_j| (theta_j - x_9)_+ + U[p,4] w_4 (x_9 - theta_4)_+ + sum over wave units and pulses 5, 9 of U[p,i] (b_i + w_i x_9)_+. For position 9: L_9 = -49.90 (theta_7 - x_9)_+ - 7.38 (theta_6 - x_9)_+ - 5.01 (theta_1 - x_9)_+ + 4.04 (theta_8 - x_9)_+ - 50.70 (x_9 - theta_4)_+ + (wave and pulse terms): a tent in x_9.

- numbers: Identity error 2.5e-13 on n=200k (seed 1043). L_9 rises 58.25 per unit through the four comparators (49.90 via U[9,7] = -3.79) and falls 50.70 per unit above theta_4 (U[9,4] = -4.99). On data theta_7 - theta_4 at step 9 has median 0.004, quartiles -0.023 / +0.025.
- how checked: e4_last_two.py part A.

### E4.2 [MIXED, confidence high]

Position 9 is scored in a common currency: while x_9 is below the thresholds, L_9 - L_p rises by the same amount per unit x_9 for every other position p; above all thresholds it falls at a nearly uniform rate.

- numbers: Exact slopes d(L_9 - L_p)/dx_9, p=0..8: 61.5, 61.7, 61.6, 61.6, 61.8, 61.8, 61.9, 62.1, 61.0 (all 13 ordinary units on, unit 4 off); -57.4, -55.1, -56.1, -56.9, -57.5, -57.8, -57.4, -54.6, -56.8 (comparators clipped, unit 4 on). Measured in the sweep: 61.8 (quartiles 61.6-62.1) at g=-0.5; -57.4 (quartiles -57.8..-56.0) at g=1.7. In between each comparator still on adds (U[9,j]-U[p,j]) w_j, between -87 and +146.
- how checked: e4_last_two.py parts A and B (20000 prefixes, seed 1040).

### E4.3 [INTERVENTION, confidence high]

For a fixed prefix x_0..x_8 the network answers 9 on a single interval of x_9 whose lower edge is the stored runner-up S_8 and whose upper edge is the stored max M_8. At the lower edge the competitor is the old runner-up's position and the comparison is made by the readout with gain 61.7; at the upper edge the competitor is the old max's position.

- numbers: 19627 prefixes with S_8 > 0 (seed 1040), 1201 values of x_9. Answers 9 somewhere in 95.4%; single interval inside the sweep range in 95.1% of those. Lower edge vs S_8: slope 1.013, intercept -0.022, R^2 0.990, median abs error 0.021; in units g=(x_9-S_8)/(M_8-S_8): median -0.02, quartiles -0.07/+0.02. Upper edge vs M_8: slope 0.998, R^2 0.996, median abs error 0.013; g median 1.00, quartiles 0.98/1.03. Competitor below the lower edge: old runner-up 95.6%; above the upper edge: old max 90.6%. Zero of the low-regime line of L_9 - L[old runner-up] vs S_8: slope 1.009, R^2 0.981, median abs deviation 0.026. Slope of L_9 - L[old max] near g=1: median -105 to -112.
- how checked: e4_last_two.py part B (input swept, network rerun at the last step).
- caveats: 5% tails of the edges are wide (lower edge 5th percentile g=-0.46, upper 95th percentile 1.26): these are the error cases.

### E4.4 [MIXED, confidence high]

Position 8 is scored through unit 5 at the last step: h_9[5] = (W[5].h_8 + w_5 x_9)_+ rises with x_8 because comparators 7 and 8 inhibit unit 5, and U[8,5] = +3.40 converts it into the same currency as position 9. The network answers 8 on the interval between the runner-up and the max of the other nine inputs.

- numbers: Exact: W[5,7]|w_7| = -7.81, W[5,8]|w_8| = -10.46, W[5,4] w_4 = +12.06; dh_9[5]/dx_8 = 18.3 (x_8 below theta_7, theta_8), 7.8 (between), 12.1 (record); times 3.40 = 62.1, 26.5, 41.0 logits per unit (W[5,ORD] w[ORD] = 18.01 gives 61.2). Measured (20000 prefixes, seed 1041): dh_9[5]/dx_8 median 18.0 (quartiles 13.1-18.1) at g=-0.1; d(L_8 - L[runner-up of others])/dx_8 median 69.1 (63.6-74.3), predicted 63.9-74.6; lower edge g median 0.00 (quartiles -0.05/+0.04), upper edge 0.99 (0.96/1.01); single interval in 93.9%. Far above the max the slope of L_8 - L[max of others] is -30.5. Pulse 4 at age 1 puts -6.63 per unit on position 8.
- how checked: e4_last_two.py parts A and C; s05_tape_amplitude.py (finite-difference slopes 18.0, 7.6, 12.0 at steps 4 and 7).
- caveats: The predicted range 63.9-74.6 ignores the small input weights of units 5 and 9 at step 8.

### E4.5 [INTERVENTION, confidence high]

U[8,5] = +3.40 is indispensable for position 8, whereas U[9,4] = -4.99 is only part of what rejects a new max at position 9; U[9,7] = -3.79 is what makes position 9 comparable at all.

- numbers: n=400k, seed 1042. Accuracy (all / max last / runner-up last / max at 8 / runner-up at 8 / neither): none 0.953 / 0.935 / 0.953 / 0.964 / 0.944 / 0.955; U[9,4]=0: 0.938 / 0.775 / 0.964 / 0.964 / 0.913 / 0.955 (answers 9 in 18.4% of max-last cases); U[9,4]=+0.5: 0.915 / 0.543 / 0.965 / 0.964 / 0.854 / 0.955; U[8,5]=0: 0.864 / 0.840 / 0.960 / 0.970 / 0.017 / 0.958 (answers max 41%, third 49% when the runner-up is at 8); U[9,7]=0: 0.548 / 0.934 / 0.983 / 0.430 / 0.525 / 0.452; U[8,9]=0: 0.952 / 0.940 / 0.946 / 0.942 / 0.953 / 0.955.
- how checked: e4_last_two.py part D (one entry of U changed, logits recomputed from the same h_9).

### E5.1 [MEASURED, confidence high]

With the exact kernels and measured amplitudes the logit sum does not reduce to 3-5 source classes: dropping even the smallest class changes many answers, because the margin is a small residual of large cancelling terms.

- numbers: n=262144, seeds 1050-1053. Agreement with the network's argmax / task accuracy: all but P9 0.886 / 0.875 (centred R^2 0.985); all but X 0.869 / 0.855; all but CW 0.456 / 0.452; all but P4 0.353 / 0.348; all but P5 0.118 / 0.113; all but C7 0.261; X+P4+P5+four comparators 0.489 / 0.485; P4+P5 0.410; X+P4+P5 0.430; without CLIP 0.422; without MARK 0.302. Best subset of size 1-5 of six classes: 0.377, 0.432, 0.483, 0.801, 0.886.
- how checked: e5_reduced.py (all subsets enumerated).

### E5.2 [MEASURED, confidence high]

Events of every age contribute; the readout is not carried by recent events only, and the events of step 0 alone are worth 20 points of agreement.

- numbers: n=131072, seeds 1070-1073. Keeping only events of age <= A: agreement 0.232, 0.324, 0.399, 0.480, 0.548, 0.622, 0.700, 0.743, 0.802, 1.000 for A=0..9. One class restricted to age <= 3: X 0.968, P9 0.931, CW 0.730, comparators 0.667, P4 0.454, P5 0.327. X only at age 0: 0.960. Wave clips of units 0, 3, 15 only: 0.920; units 0, 3, 15, 14, 10, 11: 0.992.
- how checked: e5_thinning.py.

### E5.3 [FIT, confidence high]

If the kernels of the kept sources are refitted by least squares (so they absorb what is correlated with the dropped sources), raw inputs + pulse 4 + pulse 5 + the four comparator clips reproduce about 91% of the network's answers on held-out data; adding the wave-unit clips gives 98%. In the refit the pulse-4 own-position kernel deepens with age, absorbing the wave-unit clips.

- numbers: Train 262144 (seeds 1090-1093), test 131072 (seeds 1094-1095). X+P4+P5+C (70 features): agreement 0.909, task accuracy 0.897, centred R^2 0.989. X(age 0 only)+P4+P5+C: 0.898 / 0.888. +P9: 0.932 / 0.917. X+P4+P5+C+CW (160 features): 0.984 / 0.950. P4+P5 alone: 0.329. X+C: 0.621. Refitted P4 own-position kernel ages 2-7: -5.6, -6.2, -7.6, -10.0, -10.5, -12.1 (exact -3.5 to -4.1); refitted P5: +3.1 to +3.4 for ages 0-5.
- how checked: e5_refit.py (free kernel per source, step and position; scored on fresh seeds).
- caveats: A regression, not a mechanism; the features are measured amplitudes of the running network.

### E5.4 [FIT, confidence medium]

Effective algorithm in terms of sources: logit_p = TAPE + MARK + CLIP (+ a small direct P9 term). TAPE: unit 5 writes +3.4 to +4.2 per unit on the position of an input near the top, with amplitude 18.0 (x_s - theta_5)_+ in the low regime, theta_5 about 0.17 below the running runner-up. MARK: a new max fires unit 4 (-3.2 to -4.1 per unit on its position) and, from two steps later, wave-unit clips add -13 to -30. CLIP: the current input is read directly (+57.2 x_9 on position 9); for older inputs the comparator clips leave a bump on the position of the max of that moment.

- numbers: theta_5 := x_s - pre5(s+1)/18.01 ~ -0.025 M1+ + 0.740 M2+ + 0.125 M3+ - 0.015, R^2 0.958 (n=1.98M states with comparators 7, 8 on, seed 1100); unit 5 fires after 100% / 99.9% / 38.7% / 7.1% of positive inputs of prefix rank 1 / 2 / 3 / 4. h4_s ~ 9.90 (x_s - max(0, M_(s-1))) + 0.44, R^2 0.988 (n=928k firings, seed 1002). Gain of the tape: 3.4-4.2 x 18.0 = 61-75 logits per unit input. TAPE+MARK at the max in MR: 55.9 - 71.5 = -15.6; at the runner-up: 23.7 - 4.2 = +19.5. Exact-kernel sum of the three composites: agreement 0.886.
- how checked: s05_tape_amplitude.py, s01_amplitudes.py, e3_composite.py, e5_reduced.py.
- caveats: Descriptive: no closed-form model with amplitudes replaced by order statistics was built or scored.

### E6.1 [MEASURED, confidence high]

On data the level and clock coordinates of h_9 contribute little to the logit differences among positions 0-8, slightly more when all 13 ordinary units are on at t=9 than otherwise; the readout among positions 0-8 is carried by the short, transient, h2 and h3 modes and the pulses present at t=9.

- numbers: n=524288, seeds 1060-1067; all 13 ordinary units on at t=9 in 42.9%. Rms of contribution to logits centred over positions 0-8 (ratio to total): all sequences (total 22.96): level 1.68 (0.073), clock 4.19 (0.182), level+clock 4.43 (0.193; 3.7% of the sum of squares), runner 3.15 (0.137). All on (total 19.55): level 1.76 (0.090), clock 4.44 (0.227), level+clock 4.46 (0.228; 5.2%). Otherwise (total 25.23): 1.62 (0.064), 3.99 (0.158), 4.40 (0.175; 3.1%). Carriers (all): short 28.9 (1.26), trans 20.8 (0.91), h2 12.3 (0.54), h3 9.6 (0.42), pulses at t=9 7.5 (0.33). Mode decomposition sums to the logits within 7.7e-13.
- how checked: e6_modes.py.
- caveats: Ratios above 1 reflect cancellation between modes.

### E6.2 [MEASURED, confidence high]

The readout is not exactly blind to level and clock: removing their contributions changes a noticeable fraction of answers, the clock's direct contribution slightly favours the max's position, and position 9 sees the level strongly.

- numbers: Best position among 0-8 changes when removing: level+clock 9.4% (all on 8.4%, otherwise 10.2%), clock 8.0%, level 3.6%. Overall argmax (0-9) changes: level+clock 17.0%, level 12.5%, clock 7.4%. Clock contribution to L[runner-up]-L[max]: mean -3.3, rms 7.3; to margin vs best other: mean -0.9, rms 4.8 (total margin mean 10.2, rms 13.7). Exact: R_level = (-0.452,-0.517,-0.531,-0.505,-0.308,-0.435,-0.476,-0.417,-0.168,+1.943), position 9 is 2.37 above the mean of 0-8; mean level coordinate 14.1, mean |clock| 6.1.
- how checked: e6_modes.py (contribution subtracted from the logits, argmax recomputed).

### V.1 [MIXED, confidence high]

The lead's facts F1, F2, F3, F4, F6, F9 reproduce; F7 and F8 reproduce with small deviations at the edges of their stated ranges.

- numbers: F1: trace identity error 2.8e-13. F2: 0 gate mismatches for units 1, 6, 7, 8, 4 over 50000 x 10 unit-steps each. F3: eigenvalues 1.0099, 0.9735, 0.9509@42.75, 0.9370@83.28, -0.9283, 0.8373@124.78, 0.7350@42.99, 0.2310, 0.0010; gauge coordinates of W[ORD,4] all 1. F4: cosine 0.998, max |A w| 0.432, W[PULSE,ORD] w[ORD] = (0.01, 18.01, 84.33). F6: phases 0.6, 36.7, 80.1, 123.3, 166.3, 208.9, 250.9, 293.8, 337.7 deg; rho 1.01, 1.03, 0.99, 0.95, 0.92, 0.88, 0.85, 0.82, 0.80. F7: clock cancellation ratio 0.0575; R_level over positions 0-8 from -0.531 to -0.168. F9: 540/540 two-pulse inputs correct.
- how checked: s06_verify_lead_facts.py (seed 1130), e6b_clock_split.py (seed 1110), s00_weights.py.
- caveats: F5 was not re-verified.

## Contradictions with the lead

No outright contradiction; four deviations and one caution.
(1) F7: R_level on positions 0-8 ranges from -0.531 to -0.168; position 8 (-0.168) is outside the stated -0.31..-0.53. The 6% clock cancellation reproduces (0.0575).
(2) F7 on data: "nearly blind" is quantitatively 19% of the rms (3.7% of the sum of squares) of the logit differences among positions 0-8 for level + clock; removing them changes the best position among 0-8 in 9.4% of sequences, and the overall answer in 17% (position 9 reads the level with weight 2.37 above the other positions).
(3) F8: the unit-4 own-position kernel is -3.32 and -3.18 at ages 8 and 9 (outside -3.5..-4.1). The unit-9 trough "one position before" is -2.8..-3.4 only for ages 2-7 (-1.92 at age 1, +0.22 at age 0).
(4) F6: reproduced, but rho(9) = 0.80 in my fit (regression with an S term) against 0.72.
(5) Caution on reading F8: pulse 4's negative own-position kernel does not make the max lose. At the max's own position pulse 5 gives more than pulse 4 takes (P4 + P5 = +15.6 in RM, +21.0 in MR). The max loses through the wave-unit clips that follow a record (MR) and through the comparator-clip bump on the old max (RM).

## Surprises

1. Wave unit 0 is clipped with probability 0.998 when the running max is exactly two steps old and almost never otherwise, and its clip kernel has its trough exactly two positions back. Together with units 12, 11, 10 (ages 4, 5, 6) the wave-unit clips form a second, delayed record mark worth -13 at max age 2 and about -27 to -30 from age 6. Without them MR accuracy is 23% (direct-effect ablation).
2. The comparator-clip bump is addressed by the stored clock, not by the input that clips: its size does not depend on the new max (R^2 = 0.002) and it lands on the old max's position in 89% of RM sequences.
3. U[9,4] = -4.99 is not necessary for most new-max-at-the-end cases (accuracy 0.935 -> 0.775 without it, 0.543 with unit 4 read like at other positions), whereas U[8,5] = +3.40 is indispensable for position 8 (0.944 -> 0.017).
4. The decomposition is not sparse: dropping the smallest class (P9, rms contribution 5 logits) costs 11 points of agreement; dropping only the events of step 0 costs 20 points.
5. Positions 8 and 9 use the same gain, 61-62 logits per unit input, through different paths (comparator 7 read by U[9,7] for position 9; unit 5 read by U[8,5] for position 8).
6. Single comparator clip kernels are 15-30 times larger than the pulse kernels (up to 120 logits per unit excess) yet their sum over the four comparators is <= 2.1 after one step.

## Not done

- findings.md was not written: the harness refused the write for subagents ("return findings as text, not write report files"). The findings are in this object; markdown tables are in e2_markdown_output.txt, e3_markdown_output.txt and e3_composite_output.txt.
- No closed-form reduced model (amplitudes replaced by functions of the order statistics) was built or scored; E5.4 is a description with fitted pieces.
- The bump size was not derived from the clock coordinate (only a regression, R^2 0.53 pooled, 0.89-0.91 at distances 4-5).
- The individual comparator classes by age (tables in e3_tables_output.txt) were not interpreted one by one.
- Why the margin runner-up minus max does not scale with M - S for older positions was not analysed (fits in s03_margin_fits_output.txt: slope 55-69 with R^2 0.82-1.00 only when the max is last).
- The 4.7% error cases were not decomposed.
- Unif(-1,1) inputs: only E1 was run on them.
- F5 was not re-verified.
- e3_cache.npz (420 MB) was deleted after use; e3_tables.py, e3_markdown.py and e3_composite.py need e3_aggregate.py to be run first (about 15 s).

## Open questions

- How do the thresholds of comparators 1, 6, 8 (spread at step 9 from below the runner-up to above the max) encode the clock phase, and can the clip bump be written in closed form from (age of the max, M - S)? The readout weights U[p,j]|w_j| of the four comparators are shifted copies of one wave over positions.
- Pulse 9 has a small direct kernel (rms contribution 5 logits) but dropping it costs 11 points of agreement, and it drives the unit-0 clip (amount ~ 0.17 h5 + 1.05 h9 - 0.81, R^2 0.76). Is its main role to make the delayed record mark specific to records?
- Why do the events of step 0 matter so much (20 points of agreement)? At step 0 every threshold is 0, so every positive x_0 is a record and clips all comparators.
- Does the 61-62 logits-per-unit currency hold for older positions? The margin runner-up minus third is about 57 (S - T) overall (R^2 0.53) but its slope varies from 21 to 105 with the relative ages of the two positions.
- Which composite (tape, record mark, clip bump) fails in the error cases, and is the wide tail of the answer-9 interval edges (5th percentile of the lower edge at g = -0.46) the same failure?
- What sets the margin of the runner-up over the max for old positions, given that it does not scale with M - S?

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/src.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e1_identity.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e1_identity_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e2_kernels.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e2_kernels_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e2_summary.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e2_summary_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e2_markdown.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e2_markdown_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_aggregate.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_tables.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_tables_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_markdown.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_markdown_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_composite.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_composite_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_checks.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_checks_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_bump.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_bump_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_ablate.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e3_ablate_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e4_last_two.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e4_last_two_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e5_reduced.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e5_reduced_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e5_thinning.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e5_thinning_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e5_refit.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e5_refit_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e6_modes.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e6_modes_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e6b_clock_split.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/e6b_clock_split_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s00_weights.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s01_amplitudes.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s01_amplitudes_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s02_profiles.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s02_profiles_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s03_margin_fits.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s03_margin_fits_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s05_tape_amplitude.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s05_tape_amplitude_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s06_verify_lead_facts.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s06_verify_lead_facts_output.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s07_wave_clips.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/r1_E_logit_sources/s07_wave_clips_output.txt