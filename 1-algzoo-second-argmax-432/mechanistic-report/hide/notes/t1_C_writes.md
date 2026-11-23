# Tape investigation, arm C_writes

## Answer

Angle: follow the writes. (I did not write tape_v2.md: this step returns findings only, and subagents cannot write .md files.)

Definitions. A "write" at step $s$ is one term of the exact update of the 13 non-pulse units, $h_s[o] = A\,h_{s-1}[o] + q_s$, with $q_s = w[o]\,x_s + W[o,P]\,h_{s-1}[P] + [-a_s[o]]_+$, where $a_s = w\,x_s + W h_{s-1}$ are the pre-activations and $P=\{4,5,9\}$. There are 17 write channels: the input $x_s$; last step's pulses of units 4, 5, 9; the clip $[-a_s[k]]_+ = g_k[x_s-\theta_k(s)]_+$ of each comparator $k\in\{1,6,7,8\}$ ("comparator switched off"); and the clip of each of the nine wave units.

What is established. (1) The tape coordinates are exactly a fixed linear blur of these write amplitudes with the exact eigenvalue powers: $c(t) = \sum_{s\le t}\sum_{ch} \mathrm{Amp}_{ch}(s)\,K_{ch}[t-s]$, $K_{ch}[n] = \lambda_m^{\,n}(\ell_m\cdot D_{ch})$ (error $5\times10^{-13}$). The earlier pulse-only attempt fails because the comparator and clip channels carry as much as the pulses and partly cancel them, not because the kernel was fitted. (2) Every amplitude except the raw input is a clipped comparison of ONE input with a threshold supplied by the previous state, and for units 4, 5, 9, 6, 7 that threshold does not come from the tape (the tape part of $h_{s-1}$ gives at most 0.06 of the threshold's standard deviation; 0.19 to 0.23 for comparators 1 and 8). So the tape is a ledger of threshold crossings: for each past input $x_j$, by how much it exceeded each threshold when it arrived, turned and shrunk by $\lambda_m^{\text{age}}$. (3) The thresholds are weighted means of the maximum $M$ and runner-up $S$ of the earlier inputs, with weights set by the age of that maximum: $\theta_u(j) \approx \alpha_u(a)\,M_{j-1} + \beta_u(a)\,S_{j-1}$, $\alpha+\beta \approx 0.85$ to $1.08$, rms error 0.016 to 0.041 $M$ (0.009 to 0.020 $M$ if the weights also depend on the ages of runner-up and third). So each stored amount is a difference between an input and a mix of the then-current top two, roughly $x_j - S_{j-1} - c_u(a)\,(M_{j-1}-S_{j-1})$. (4) Unit 5's pulse, the largest and most sensitive write, is $[k_x x_j - \dots]_+$: after an input below the runner-up $18.0\,(x_j - \Theta_5)$ with $\Theta_5 \approx (0.49\text{–}0.82)\,S + (0.09\text{–}0.32)\,T - (0\text{–}0.07)\,M$; after a new runner-up the slope is 7.8 to 18.7 depending on the age of the maximum; after a new maximum 12.1 to 13.0. (5) By causing input, on quiet final states: the maximum's arrival carries 53% of the variance of the tape scores of ages $\ge 2$, the runner-up's arrival 34%, an earlier runner-up that was later displaced 9%, everything else under 4% each. Which one dominates depends on the order: runner-up after the maximum 77% runner-up and 10% maximum; runner-up before the maximum 81% maximum and 6% runner-up.

So the tape stores, per past event, the excess of the arriving input over thresholds built from the top two at that time: for the maximum, how far it rose above the earlier maximum and runner-up; for the runner-up, its lead over the earlier runner-up and its position between the old runner-up and the maximum; for inputs just below the runner-up, their excess over about $0.75\,S$. Ages are encoded only through the fixed factor $\lambda_m^{\text{age}}$.

What is not established. No description in named quantities passes the functional test cleanly. The best one (thresholds from $M^+, S^+, T^+$, the latest input and their three ages, exact kernel) gives $R^2$ 0.88 to 0.96 per coordinate and 0.925 for the scores, yet changes 7.2% of answers at the last step (22.6% when the runner-up leads the third by less than $0.05\,M$). The tape is therefore not a function of the current $M, S$ and their ages alone: it keeps amounts fixed at event time (old maximum, old runner-up, displaced runners-up), and the wave-unit clips, which partly depend on the tape itself, are not reduced to a simple formula.

## Claims (9)

### 1. [strong] The tape coordinates are exactly a fixed linear blur of 17 write amplitudes with the exact factors lam^age: c(t) = sum over s <= t and channels of Amp_ch(s) K_ch[t-s], K_ch[n] = lam_m^n (l_m . D_ch). Channels: input, pulses of units 4, 5, 9 from the previous step, clips of comparators 1, 6, 7, 8, clips of the nine wave units.

- evidence: 300,000 N(0,1) sequences (seed 101). One-step rule for the four tape modes: max error 4.0e-13 over all steps. Unrolled sum at t = 3, 6, 9: max error 4.2e-13, 5.2e-13, 5.5e-13 (coordinates are of order 10).
- script: c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w1_verify_kinds.py -> w1_verify_kinds.txt (kernel in wcommon.py)
- caveats: An identity. It restates the network; the content is in the claims about the amplitudes.

### 2. [strong] The kinds of write cancel strongly, and every kind except the raw input and unit 9's pulse is functionally necessary. A pulse-only description cannot work, whatever kernel is used.

- evidence: Quiet final states, n = 184,653 of 300,000. Std of each kind's part of the tape scores (ages >= 2, units of M_t) relative to the std of the whole: pulse 5 1.60, comparators off 1.18, pulse 4 1.12, wave clips 0.71, pulse 9 0.25, input 0.16. Covariance shares (sum to 1): +0.84, +0.79, -0.42, -0.28, +0.01, +0.05. Answers changed when one kind is dropped and the tape rebuilt (set_tape on h_9): pulse 5 97.0%, pulse 4 67.5%, wave clips 62.2% (clip of unit 0 alone 48.0%, the other wave clips 22.6%), comparators off 54.6%, pulse 9 12.2%, input 5.3%; whole tape deleted 93.2%. R2 of the tape score entry of age 2..9 on 123,299 quiet final states (seed 304): pulses only with the exact kernel 0.61, 0.42, 0.17, 0.04, -0.05, -0.39, -0.47, -0.16; pulses with a free least-squares kernel over all lags 0.89, 0.75, 0.60, 0.59, 0.64, 0.51, 0.38, 0.57; pulses + comparators 0.61, 0.69, 0.67, 0.56, 0.30, 0.22, 0.32, 0.60; + clip of unit 0 0.82 to 0.90 at ages 2-5 and 0.61 to 0.79 at ages 6-9 (23.2% of answers still change); all but the input term 0.944 to 0.994 (5.1% change).
- script: w1_verify_kinds.py -> w1_verify_kinds.txt; w4_reduced_whole.py -> w4_reduced_whole.txt (part 1)
- caveats: Final states only for the leave-out numbers. Covariance shares can be negative because the parts are anticorrelated; they are an accounting, not a causal measure.

### 3. [moderate] Grouped by the input that caused the writes, the tape is mostly the record of two arrivals, the maximum's and the runner-up's, and which of the two dominates depends on their order. A displaced earlier runner-up is the third contributor.

- evidence: 300,000 sequences (seed 202), quiet final states n = 184,212. Covariance share in the tape scores of ages >= 2 (sums to 1): maximum 0.533, runner-up arrived after the maximum 0.302, runner-up arrived before the maximum 0.037, earlier runner-up now third or lower 0.090, earlier maximum now third or lower 0.009, input within (0.7 S, S] at arrival 0.012, positive input further below 0.004, non-positive inputs 0.034, wave clips at steps without a recent event -0.021. Split by order: runner-up after the maximum (n = 92,014): runner-up 0.773, maximum 0.098, earlier runner-up 0.098; runner-up before the maximum (n = 92,198): maximum 0.808, runner-up 0.062, earlier runner-up 0.087. Inside the maximum's chain the kinds still cancel (std ratios pulse 5 1.41, pulse 4 0.96, comparators 0.86, wave clips 0.46).
- script: w2_by_cause.py -> w2_by_cause.txt
- caveats: Attribution rules are mine: input term and comparator clips to x_s, pulse 4 to x_(s-1), pulses 5 and 9 to x_(s-2); wave clips by a priority list of recent events (a heuristic; clips matching no rule are 'spontaneous'). 'Near' is defined as x > 0.7 S at arrival. Pooled over the ages of maximum and runner-up.

### 4. [strong] Leave-out test by cause: the writes of the maximum and of the runner-up are indispensable; the tape of the two alone is not enough; far-below inputs leave nothing that matters.

- evidence: Same 184,212 quiet final states, baseline accuracy 96.27%. Answers changed when one class is dropped (among states where it is present): maximum 42.8%; runner-up after the maximum 96.3%; runner-up before the maximum 62.6%; earlier runner-up 19.9% (present in 48%); near 10.6% (present in 35%); earlier maximum 4.3% (48%); far below 0.9% (76%); non-positive 4.0%; spontaneous wave clips 5.6% (80%). Keeping only the writes of maximum and runner-up: 20.1% changed (71% of the changed answers become the maximum's position), R2 of the scores 0.755; adding earlier maxima and runners-up: 11.1%, R2 0.931; everything except far-below and non-positive inputs: 4.3%, R2 0.968. With the run continued from quiet states at t = 4, 6, 7 (seed 304; n = 48,399, 89,190, 103,354): maximum and runner-up only 2.8%, 9.5%, 13.0% (clipped at zero 2.3%, 6.8%, 9.7%); without far and non-positive 0.9%, 1.5%, 1.8%; tape deleted 36.6%, 56.5%, 67.9%.
- script: w2_by_cause.py -> w2_by_cause.txt (part 4); w4_reduced_whole.py -> w4_reduced_whole.txt (part 3)
- caveats: In the continued runs 40 to 80% of modified states have some activation below -1e-9 (mostly tiny); the clipped repeats are given. The 4.0% for non-positive inputs is mostly the direct input term of the last one or two steps.

### 5. [strong] Every write amplitude except the raw input is a clipped comparison of one input with a threshold from the previous state, and the thresholds that matter are weighted means of the maximum and runner-up of the earlier inputs, with weights set by the age of that maximum. Comparator and unit-4 amounts are therefore differences of named quantities and survive the functional test.

- evidence: Fit on 300,000 sequences, tested on 200,000 others (123,299 quiet final states). L1 (weights depend on the age a of the earlier maximum; regressors M+, S+, latest input): rms threshold error in units of the running maximum 0.016 (unit 4), 0.040 (1), 0.032 (6), 0.018 (7), 0.041 (8); with weights depending on ages of maximum and runner-up (L2) 0.010 to 0.023; with the third's age too (L3) 0.009 to 0.020. Examples (a = age of the maximum at the step before): theta_4 = 0.98 M at a = 1, 0.68 M + 0.26 S at a = 5-6; theta_7 = 1.01 M at a = 1, 0.61 M + 0.34 S at a = 5; theta_1 = 0.92 S at a = 1, 0.84 M + 0.14 S at a = 3, 1.36 M - 0.31 S at a = 5, 0.25 M + 0.60 S at a = 8; theta_6 = 0.70 M + 0.23 S at a = 1, 1.17 M - 0.15 S at a = 3, 0.45 M + 0.43 S at a = 7; theta_8 = 0.05 M + 0.94 S at a = 2, 0.44 M + 0.55 S at a = 4, 1.41 M - 0.34 S at a = 7. Functional test with only that kind of write modelled (others real): comparators off 3.9% / 1.5% / 1.3% of answers changed at L1 / L2 / L3; pulse 4 0.4% / 0.3% / 0.2%.
- script: w3_named_writes.py -> w3_named_writes.txt
- caveats: The weights are fitted per age, not derived. alpha + beta is 0.85 to 1.08, so 'difference' is approximate. M and S here are those of the inputs before the event, not the current ones.

### 6. [moderate] The thresholds that set the pulses and comparator writes are supplied by the rest of the state, not by the tape; the tape feeds back on itself only through the clips of wave units 3, 15, 13, 14.

- evidence: 200,000 test sequences, steps 1..9. Std of the tape part's contribution to the drive (W h_(s-1))[u] relative to the std of the drive: unit 4 0.03, unit 5 0.06, unit 9 0.01, comparators 6 and 7 0.04, unit 2 0.04, comparators 1 and 8 0.19 and 0.23; wave units 3, 15, 13, 14: 0.98, 0.67, 0.54, 0.51. Covariance shares: at most 0.02 in size for units 2, 4, 5, 6, 7, 9; 0.06 for units 1 and 8; 0.60 for unit 3 and 0.23 for unit 15.
- script: w3_named_writes.py -> w3_named_writes.txt (last block)
- caveats: A variance statement over all states, not an intervention. The 'rest' includes the pulse units.

### 7. [strong] Unit 5's pulse is the largest write and the bottleneck of any named description. Its meaning is clear by event, but a fit with R2 above 0.99 of its amplitude still changes about 5% of answers.

- evidence: Pre-activation one step after x_j fitted per kind of x_j and per age of the earlier maximum on (x_j, Mp, Sp, Tp, x_(j+1)); 200,000 train, 200,000 test. Below the runner-up: slope 17.9 to 18.2 in x_j, implied threshold (0.49 to 0.82) Sp + (0.09 to 0.32) Tp - (0 to 0.07) Mp, fires after 19 to 29% of positive inputs below the runner-up. New runner-up: slope 7.8 to 8.1 when the maximum was 0, 2 or 3 steps old at the step before, 11.1 at 1, 13.2 at 4, 18.3 to 18.7 at 5 to 7; R2 of the pulse 0.93 to 0.97. New maximum: slope 12.1 to 13.0, R2 0.94 to 0.995. Rms error of the modelled pulse 0.36 M_t against 4.65 M_t rms of the real pulse. Functional test, only unit 5's pulses replaced by the formula: 5.4% of answers changed (after new maxima only 1.8%, new runners-up only 2.8%, below the runner-up only 2.0%). With weights depending only on the age of the maximum (L1): 34.6%; L2 6.0%; L3 5.3%. Removing the pulses after new maxima changes 74.9% of answers, after new runners-up 50.3%, after inputs below the runner-up 2.1%. One unit of pulse puts +2.5 to +4.0 into the tape score at its own input's index and -0.9 to -1.9 at the index of the input one step earlier.
- script: w5_pulse5_meaning.py -> w5_pulse5_meaning.txt; w3_named_writes.py -> w3_named_writes.txt
- caveats: The 'rms/scale' column of w5's table is in units of max(x_j, Mp) per event and is less useful than the summary rms. The per-event slopes agree with report section 5.3 (18.0 and 7.5).

### 8. [strong] The reduced description as a whole (all writes replaced by clipped comparisons with thresholds from the three largest earlier inputs, the latest input and their ages; exact kernel) reproduces the tape to R2 about 0.9 but does not pass the functional test cleanly.

- evidence: 123,299 quiet final states of 200,000 test sequences, baseline accuracy 96.37%. L1 (10 age configurations): R2 per coordinate 0.53 to 0.76, scores 0.658, 30.0% of answers changed. L2 (82 configurations): 0.83 to 0.94, scores 0.885, 10.4% changed. L3 (586 configurations): 0.88 to 0.96, scores 0.925, 7.2% changed, accuracy 92.3%. L3 using only the writes of maximum, runner-up, earlier maxima and runners-up and near inputs: scores 0.893, 13.9% changed. L3 failures concentrate in close calls: 22.6% changed when the runner-up leads the third by less than 0.05 M, 7.9% for 0.05-0.1, 2.9 to 4.2% above 0.1; by order 7.7% (runner-up after the maximum) and 6.7% (before). Reference: the tape of a random other quiet state, rescaled by M, changes 87.4%. With the run continued from quiet states at t = 4, 6, 7, L3 changes 1.4%, 3.3%, 4.5% (clipped 1.4%, 3.1%, 4.1%); L2 2.9%, 5.3%, 7.1%.
- script: w3_named_writes.py -> w3_named_writes.txt; w4_reduced_whole.py -> w4_reduced_whole.txt
- caveats: L3 has about 586 x 4 coefficients per unit, so it is an upper bound on what 'top three values and their ages at event time' can describe, not a compact law. By kind at L3: pulse 5 5.3%, wave clips 4.5%, pulse 9 1.7%, comparators 1.3%, pulse 4 0.2%.

### 9. [moderate] The wave-unit clips are the part with no simple named meaning. The one regular piece is the clip of unit 0 two steps after a new maximum.

- evidence: 300,000 sequences: unit 0 is clipped at 100% of steps two steps after a positive new maximum (mean size 3.5 running maxima); unit 11 at 92% of steps five steps after, unit 10 at 95% six steps after, unit 12 at 30% four steps after; units 3 and 15 at 14 to 44% of steps over lags 0 to 6. Wave-clip amplitudes modelled at L1: R2 0.33, 14.0% of answers changed; L2 0.82, 7.1%; L3 0.93, 4.5%. Wave clips at steps with no event in the last two steps: covariance share -0.021, 5.6% of answers changed when dropped (where present).
- script: w1_verify_kinds.py -> w1_verify_kinds.txt (part d); w2_by_cause.py (part 0); w3_named_writes.py
- caveats: The mean-size column of the lag table is distorted in some cells by steps with a tiny running maximum; I use only the clipped shares and the unit-0 entry.

## Did not work

- Thresholds with weights that depend only on the age of the maximum (L1) are fine for comparators and unit 4 but fail for unit 5's pulse (34.6% of answers changed) and for the wave clips (amplitude R2 0.33): after an event the pulse depends on the old runner-up and on which comparators were off.
- Pulses of units 4, 5, 9 alone with the exact kernel give R2 from 0.61 down to -0.47 for tape entries of ages 2 to 9; a free least-squares kernel gives 0.38 to 0.89. The tape is not a blur of the pulses.
- A tape built from the writes of the current maximum and runner-up only changes 20% of final answers (71% of those become the maximum), so the tape is not determined by the two arrivals alone.
- No named description reached the real tape functionally: the best (L3) still changes 7.2% of final answers, and an amplitude R2 of 0.997 for unit 5's pulse still costs about 5%.
- The first version of the unit-5 functional test was off by one step (the pulse at step j+1 is written at step j+2) and gave 91% changed; fixed and guarded by an assertion in w5.

## Open

- A compact closed form for the age-dependent threshold weights alpha_u(a), beta_u(a) (they look like a level plus a rotating term, the clock) was not fitted or tested.
- What the 7.2% residual of L3 consists of: I did not test whether adding the fourth largest input, or the values at the time of earlier events, closes it.
- The wave clips of units 3 and 15 depend on the tape itself (0.98 and 0.67 of their drive's std); their role was not reduced to named quantities.
- The accounting is pooled over the ages of maximum and runner-up within each order; I did not split the shares by the full age configuration.
- The attribution of wave clips to causing inputs is a heuristic priority list; a counterfactual attribution (rerun with the input lowered) was not done.
- Why keeping only the maximum's and runner-up's writes sends 71% of the changed answers to the maximum: probably the missing writes of earlier runners-up, not tested.
- The leave-out and reduced-description tests were run on N(0,1) inputs only, and the main numbers are for final states (t = 9); t = 4, 6, 7 with the run continued were checked for a subset of the tests.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/wcommon.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w1_verify_kinds.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w1_verify_kinds.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w2_by_cause.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w2_by_cause.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w3_named_writes.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w3_named_writes.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w3_cache.npz
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w4_reduced_whole.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w4_reduced_whole.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w5_pulse5_meaning.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_writes/w5_pulse5_meaning.txt
