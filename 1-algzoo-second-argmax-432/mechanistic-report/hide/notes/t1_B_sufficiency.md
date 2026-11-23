# Tape investigation, arm B_sufficiency

## Answer

Method and terms. A candidate $V$ is the step $t$ plus a list of inputs whose age and value (in units of the current maximum $M_t$) are held fixed; every other input is redrawn from $N(0,1)$, truncated so that $V$ stays exactly the same (asserted in code). $FU(V) = E[\mathrm{Var}(c/M_t \mid V)] / \mathrm{Var}(c/M_t)$ is the share of the variance of the seven tape coordinates $c$ over quiet states (steps 3..9) that $V$ leaves unexplained, summed over the seven. The tape transplant puts the tape part of a redrawn state into the original state, continues with the original later inputs, and asks whether the final answer changes.

1. The tape is first of all a positional record. The ages of maximum and runner-up alone leave $FU = 0.31$; the value $S_t/M_t$ alone leaves 0.93; both together ($V_0$) leave 0.18. All values without any ages leave 0.92.
2. The best single further quantity is the displaced runner-up $D$: the largest input that arrived before the last record event (the arrival of the younger of maximum and runner-up). $V_0 + D$ leaves 0.049. No input of a fixed age does this (0.15-0.18), and the third largest does slightly worse (0.058). $D$ needs both its age (alone 0.129) and its value (alone 0.117).
3. The smallest set I found that reaches about 1% is the "contenders": every input $x_j$ with $x_j > 0.75\,\max(R_{j-1}, 0)$, where $R_{j-1}$ is the second largest of the inputs before it. This is 3.0 inputs on average, maximum and runner-up included, out of 7.8. It leaves $FU = 0.014$ over all quiet states and 0.007 on the 96% with $M_t > 0.5$. The threshold is not sharp: 1 (records only) gives 0.031, 0.5 gives 0.011, 0 (all positive inputs) gives 0.010.
4. Not needed, in variance terms: the other 4.8 inputs per state (78% negative), including most of the 3.3 inputs that arrived after both maximum and runner-up, and recency as such (fixing the two newest inputs moves FU from 0.180 to 0.172).
5. There is no floor. Each further input fixed lowers FU by a factor 2-3 (top 4: 0.020, top 5: 0.008, top 6: 0.003), so the tape keeps a faint trace of every input. For non-contenders this trace does not scale with $M_t$: FU is 0.09-0.13 where $M_t \le 0.5$ and 0.004 where $M_t > 2$.
6. Functionally, sufficiency is only partial. The tape of an unrelated quiet state changes the answer in 64% of states; same $V_0$: 6.7%; $V_0 + D$: 4.0%; same contenders: 2.3%; same positive inputs: 1.6%. Hybrid accuracy is 93.1%, 94.7%, 95.3% and 95.6%, against 96.2% unswapped. At $t = 9$ alone the contender transplant changes 4.8% of answers. Changes concentrate in near ties: 11.8% where the final runner-up and third differ by less than $0.02M$, about 1% above $0.1M$.
7. Part of the reason is that the tape's trace of the non-contender inputs is cancelled by the rest of the state. Redrawing them changes the tape piece of the scores (relative to the runner-up's entry) by rms $3.5M_t$ and the rest by $3.0M_t$, but their sum by only $1.8M_t$ (correlation $-0.86$). A tape-only transplant breaks that cancellation.

Not established: how each contender is encoded in the seven coordinates (not tested here), and whether the contender set is minimal beyond the families I tried. I did not write tape_v2.md; the brief routes findings through this return value.

## Claims (10)

### 1. [strong] The tape (coordinates divided by M_t) is mainly a positional record of maximum and runner-up. Their two ages alone explain 69% of its variance; adding S_t/M_t brings it to 81-82%; values without ages explain almost nothing.

- evidence: 115,368 quiet states (steps 3..9 of 40,000 N(0,1) sequences, seed 2), K = 4 exact conditional redraws per state, step t fixed in every line. FU summed over the 7 coordinates: nothing fixed 0.968; ages of max and runner-up only 0.309; S/M only (ages redrawn) 0.927; V0 = both 0.188; ages of all inputs by rank without values 0.227; all values without ages 0.918. Tape-transplant answer change: 62.5%, 13.7%, 60.2%, 6.7%, 12.7%, 60.1%. V0 on five other samples: 0.178-0.193.
- script: hide/t1_resampling/s2_values_vs_ages.py -> s2_values_vs_ages.txt; s7_check_samplers.py -> s7_check_samplers.txt
- caveats: FU is relative to the variance over all quiet states at steps 3..9, which includes differences between steps. States from the same sequence at different steps are correlated; no bootstrap errors.

### 2. [strong] The most compact sufficient set found is the contenders: every input that on arrival exceeded 0.75 x max(runner-up of the earlier inputs, 0). It averages 3.0 inputs including maximum and runner-up and leaves about 1% of the tape's variance unexplained.

- evidence: 114,899 quiet states (seed 5), K = 3. FU per coordinate (alt, short re/im, h2 re/im, h3 re/im): 0.012, 0.016, 0.004, 0.023, 0.015, 0.018, 0.013; sum 0.014; tape scores at ages >= 2: 0.021. Kept inputs 3.03 of 7.81 (2 kept in 35.5% of states, 3 in 35.9%, 4 in 20.3%, 5+ in 8.3%). Threshold scan, FU sum / kept: c=1 (records above zero) 0.031 / 2.78; 0.9: 0.020 / 2.87; 0.75: 0.014 / 3.03; 0.5: 0.011 / 3.34; 0.25: 0.010 / 3.69; 0: 0.010 / 4.07. On states with M_t > 0.5 (96%): 0.0072 and 0.0073 on two samples. Compare at similar size: V1 (top 3) 0.057-0.059, V2 (top 4) 0.019-0.020, V3(0.5) = inputs above 0.5 S_t 0.046 at 2.88 kept.
- script: hide/t1_resampling/s5_final.py -> s5_final.txt; s1_candidates.py -> s1_candidates.txt; s8_small_max.py -> s8_small_max.txt
- caveats: The threshold 0.75 is not sharp; FU falls smoothly with c. Minimality is only relative to the families tried (final ranks, thresholds on S_t, records, thresholds on the runner-up at arrival, segments, single ages). The pooled value over all states is unstable between samples (0.012-0.036 on six samples of 20,000 sequences) because of states with small M_t; the M_t > 0.5 value is stable.

### 3. [strong] The best single quantity to add to V0 is the displaced runner-up D, the largest input that arrived before the last record event. Both its age and its value are needed. No input of a fixed age helps.

- evidence: Greedy search, 115,025 quiet states (seed 4), K = 2 for the search. Step 1 from V0 (0.179): D 0.048 (2.87 kept); third largest overall 0.058; largest input after the last record event 0.161; input of any single age 0..9: 0.148-0.176. Later steps: + largest after the last record event 0.028; + 2nd largest before 0.010; + 2nd largest after 0.006; + 3rd largest before 0.003. Separately (seed 5, K = 3): V0 + age of D only 0.129, V0 + value of D only 0.117, both 0.049; D exists in 86.6% of states. Same split for the third largest (seed 2): age only 0.117, value only 0.156, both 0.065.
- script: hide/t1_resampling/s4_greedy.py -> s4_greedy.txt; s5_final.py -> s5_final.txt; s2_values_vs_ages.py
- caveats: At step 2 three candidates are within 0.002 of each other (0.028-0.030), so the order of the path after D is not meaningful.

### 4. [strong] Sufficiency is partial in the functional test. Transplanting the tape between states that share V changes far fewer answers than an unrelated tape, but still costs accuracy, and more so at the final step.

- evidence: Seed 5, 114,899 states, K = 3, donor rescaled to the recipient's M. Answer changed / hybrid accuracy (unswapped 96.18%): unrelated quiet state 63.8% / 35.8%; V0 6.70% / 93.07%; V0 + D 4.04% / 94.68%; records 3.63% / 94.81%; contenders c=0.75 2.29% / 95.28%; all positive inputs 1.56% / 95.63%; top 6 inputs (seed 3) 0.99% / 95.85%. Sham (own tape) 0.00%. At t = 9 only (24,683 states): unrelated 87.4%, V0 11.5%, contenders 4.78% (accuracy 93.53% against 96.17%), all positive 3.00%. Clipping the hybrid at zero changes these by at most 0.7 points (V0 at t = 9: 12.2%); 44-48% of hybrids have a negative activation. Contender-transplant changes by gap (S - T)/M of the whole sequence: 11.8% below 0.02, 5.6% at 0.02-0.05, 1.75% at 0.05-0.1, 0.9-1.2% above 0.1. By margin decile of the unswapped answer: 15.7% in the lowest, at most 0.7% from the fourth up.
- script: hide/t1_resampling/s5_final.py -> s5_final.txt; s3_segments_configs.py -> s3_segments_configs.txt
- caveats: For t < 9 the run continues with the original later inputs, which can overwrite the effect; t = 9 is the stricter test.

### 5. [moderate] The tape's residual dependence on the non-contender inputs is largely cancelled by the rest of the state, so the tape is not a self-contained store of them. A tape-only transplant breaks that cancellation.

- evidence: 36,585 final quiet states (t = 9, 60,000 sequences, seed 6), one redraw each, score entries of age >= 2 relative to the runner-up's entry, units of M. Contenders c=0.75 fixed: change of the tape piece rms 3.53, of the rest of the score 3.03, of their sum 1.78, correlation -0.86. All positive inputs fixed: 2.63, 2.18, 1.37, -0.85. Under V0: 13.25, 4.64, 12.17, -0.40 (little cancellation). Consistent with the transplants (seed 5, all steps): contenders fixed, tape only 2.29%, rest only 2.17%, whole redrawn state 1.38%; V0: 6.70%, 5.93%, 4.33%.
- script: hide/t1_resampling/s6_compensation.py -> s6_compensation.txt; s5_final.py -> s5_final.txt
- caveats: One redraw per state and t = 9 only for the correlation. Which of the rest's modes or pulse units do the cancelling was not examined.

### 6. [strong] Inputs that arrived after both maximum and runner-up carry little of the tape's variance although they are most of the other inputs; inputs between and before them carry most of it.

- evidence: 114,389 quiet states (seed 3), K = 3. Mean inputs per state: older than both 1.27, between 1.26, younger than both 3.27. FU sum: V0 0.178; V0 + all older 0.117; V0 + all between 0.091; V0 + all younger 0.155; older + between fixed (only the younger redrawn) 0.025; older + younger fixed 0.094; between + younger fixed 0.066. Records above zero alone (0.78 extra inputs) give 0.030, nearly the same as fixing all 2.5 older and between inputs. Recency: V0 + inputs of age 0, 1: 0.172; V0 + ages 0..3: 0.138 (seed 1).
- script: hide/t1_resampling/s3_segments_configs.py -> s3_segments_configs.txt; s1_candidates.py -> s1_candidates.txt
- caveats: Small in variance, not in function: redrawing only the younger inputs still changes the answer in 3.4% of tape transplants (2.8% for the whole redrawn state), because some of them are near ties with the runner-up.

### 7. [strong] There is no floor: the tape holds a faint trace of every input, including negative ones, and for non-contenders this trace does not scale with M_t.

- evidence: Seed 1, 228,596 states, K = 4. FU sum by number of largest inputs fixed: top 2: 0.180, top 3: 0.059, top 4: 0.020, top 5: 0.008, top 6: 0.003. All positive inputs fixed, only negative ones redrawn: 0.011, answer changed 1.52%. By M_t (seeds 2 and 5, contenders fixed, FU within the bin): M <= 0.5: 0.131 / 0.086; 0.5-1: 0.012; 1-1.5: 0.0067; 1.5-2: 0.005; above 2: 0.004. The 1% of states with the largest within-V variance carry 45-65% of the numerator.
- script: hide/t1_resampling/s1_candidates.py -> s1_candidates.txt; s8_small_max.py -> s8_small_max.txt
- caveats: The M_t dependence is a consequence of dividing by M_t while the other inputs are N(0,1); in absolute units the trace is small but present.

### 8. [strong] By configuration, V0 fails where many inputs lie between maximum and runner-up; the contender set is uniform across configurations.

- evidence: Seed 3, K = 3, FU relative to all quiet states. V0 by step: t = 3: 0.013, rising to t = 9: 0.307. V0 by pair of ages: 0.06-0.13 when the two are 1-2 steps apart, 0.87-1.14 for (a_M, a_S) = (2, 8), (2, 9), (8, 2), (9, 2), (9, 3), (3, 9). Runner-up older than maximum 0.164, younger 0.192. Contenders c=0.75: 0.010-0.014 at every step, 0.009-0.019 for every a_M and for a_S <= 8, 0.030 at a_S = 9; 0.012 for both orders. Answer change under contenders rises with age: 1.5% at a_M = 2 to 7.8% at a_M = 9; 1.7% at a_S = 2 to 10.3% at a_S = 9.
- script: hide/t1_resampling/s3_segments_configs.py -> s3_segments_configs.txt
- caveats: Cells with fewer than 200 states are not shown. The a_M = 9 and a_S = 9 groups have about 3,060 states each.

### 9. [moderate] At the final step the tape carries most of the answer by itself: keeping a state's own tape and taking everything else from an unrelated quiet state preserves the answer in 75% of states, while the reverse gives chance.

- evidence: 24,683 final quiet states (seed 5), K = 3, donor rescaled to M. Own tape + unrelated rest: answer changed 24.8%. Own rest + unrelated tape: 87.4% changed, hybrid accuracy 12.1% (chance 10%). Over all steps 3..9: 35.9% and 63.8%.
- script: hide/t1_resampling/s5_final.py -> s5_final.txt
- caveats: A side result of the control, not followed up. The rest includes the pulse units, which are zero in most quiet states.

### 10. [moderate] The set of inputs that fired unit 5 one step after arriving (V5) is no better than the simple threshold on the runner-up at arrival, and costs more inputs.

- evidence: Seed 1, 228,596 states, K = 4. V5 = record history + inputs that fired unit 5: 3.96 kept, FU 0.011, answer changed 2.37%. Inputs above 0.75 x runner-up on arrival: 3.03 kept, FU 0.015, 2.26%. Above 0.5 x: 3.33 kept, 0.012, 1.98%. Unit 5 fired after 3.00 inputs per state, of which 0.21 were not already records. Rejection sampler failed for 0.01% of states.
- script: hide/t1_resampling/s1_candidates.py -> s1_candidates.txt
- caveats: V5 was built on the plain record history (3.75 kept), which includes negative early records that the zero-floored version drops (2.78 kept, FU 0.032 against 0.030).

## Did not work

- Fixing inputs by age (recency) explains almost nothing beyond V0: any single age gives FU 0.148-0.176, ages 0 and 1 together 0.172.
- Thresholds relative to the final runner-up, V3(c), are less efficient than thresholds relative to the runner-up at arrival: V3(0.75) leaves 0.093 at 2.39 inputs and V3(0.5) leaves 0.046 at 2.88, against 0.014 at 3.03 for the contenders.
- Earlier maxima alone (inputs that were the running maximum) leave 0.126; the displaced runners-up are what matters.
- The unit-5 firing set (V5) did not improve on the plain threshold rule.
- I could not get the tape-transplant answer-change rate below about 1% without fixing nearly every input (top 6 of 7.8: 0.99%), so no small V is sufficient to the precision the answer needs.
- The plain variance metric is unreliable for FU below about 0.03 over all states: it moved between 0.012 and 0.036 across samples until I restricted to M_t > 0.5.
- A 'last k records' chain was not tested, because its conditional distribution needs rejection sampling that I did not build.

## Open

- How each contender is encoded in the seven coordinates was not tested here; this angle only shows which inputs the tape depends on.
- Which part of the rest (level, runner, clock, trans, echo or pulse units) cancels the tape's trace of the non-contender inputs, and whether the cancellation is structural.
- Whether the contender threshold reflects unit 5's threshold (about 0.75 S in the project notes) or is just a point on a smooth curve; the scan over c shows no kink.
- Why the contender transplant still changes 1.4% of answers at M_t > 2 and 0.9-1.2% at gaps above 0.1 M, where near ties should not matter.
- No bootstrap errors; states from one sequence at several steps are correlated. Per-coordinate FU for M_t > 0.5 was not computed, only the sum.
- tape_v2.md was not written by this investigator; the brief says findings go in this return value.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/rs.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s1_candidates.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s1_candidates.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s2_values_vs_ages.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s2_values_vs_ages.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s3_segments_configs.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s3_segments_configs.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s4_greedy.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s4_greedy.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s5_final.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s5_final.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s6_compensation.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s6_compensation.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s7_check_samplers.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s7_check_samplers.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s8_small_max.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_resampling/s8_small_max.txt
