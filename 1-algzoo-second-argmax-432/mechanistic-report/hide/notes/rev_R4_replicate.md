# Report review R4_replicate

## Summary

Spot replication of sub-investigation claims in sections 4.3, 5.2, 5.3, 5.4, 6, 7 and 8, with my own code and fresh seeds 7001–7112 (scripts and raw outputs in hide/rev_R4_replicate; only common.py and modes.py reused). No claim failed. Four numbers or formulas differ from what the report states, or agree only under a condition the report does not state (severity minor). The rest are ambiguities of wording or conditioning (nit).

The ten claims (report's statement → my number → verdict):

1. §4.3, clock and level in the readout. Level + clock = 19% of the rms of logit differences over positions 0–8 → 19.3% (level 7.3%, clock 18.2%; n = 2²¹): agrees. Clock zeroed in h₉ changes 7.8% of answers and costs 4.9 points → 7.85% and 4.94 points on sequences with M > 0 and the maximum not last; 7.4% and 4.6 points over all sequences: agrees on the arm's subset, which the report does not state (nit). Also confirmed: resting pattern on 10.3% of steps, 29.0% at t = 9, 18.8% when the maximum is ≥ 2 steps old; 10 patterns cover half of all steps and 75 cover 90%. I count 1352 distinct patterns in my 2²⁰ sequences against the report's 1377; the count depends on the sample. ReLU correction 6.55% of the state norm.

2. §5.2, unit 5's input and threshold. 18.009 and −0.281 → 18.0085 and −0.2813; Θ₅ coefficients 0.01265 (runner) and −0.02872 (level) → identical, per unit coordinate of h_(t−1); clock at most 0.037 of the gap → 0.00365 × 10.157 = 0.037. With all clip terms the identity holds to 6×10⁻¹⁴ on 3.6M steps. Firing after 0.1% / 51% / 89% / 99.5% of inputs (negative, x/S in [0.7,0.8), [0.8,0.9), [0.9,1)) → 0.09% / 50.9% / 89.3% / 99.5%: agrees. Differs: the M coefficient of Θ₅ is +0.05 at max-age 2, outside the stated −(0.01–0.07); "+ small" is about 0.4·S at age 2 (minor); "exactly" omits the comparator-1 and comparator-6 terms (nit).

3. §5.3, new runners-up that cut no comparator. 42–66% at max-ages 5–8 → 52.1, 64.7, 57.9, 42.3% (52.3, 65.4, 58.8, 43.2% with S⁺); 0.1, 0.4, 6.8, 9.3% at ages 1–4: agrees. "Still right in 91–95%" → 0.948, 0.952, 0.912, 0.928 for lags 5–8: agrees.

4. §5.4, gate sequence and amplitudes (new maximum at s = 2..6, M₀ > 0, n = 940k). Unit 4 fires 99.997% → 99.998%; all four comparators cut 94–96% → 95.5% pooled (99% at s = 1–3 falling to 87% at s = 8; nit); units 5 and 9 at s+1 100% / 99.7% → 99.997% / 99.68%; h₅ ≈ 12.2, 10.4, −5.8 → 12.22, 10.44, −5.79; h₉ ≈ 1.56, 1.77, −0.43 → 1.56, 1.76, −0.43 (regressor is S₀⁺; nit); 12.06 and 1.48 → 12.057 and 1.481; −12.9 and −3.3 → −12.91 and −3.30; thresholds at s+1 → 1.018 Mₙ, 0.985 Mₙ, 0.831 Mₙ + 0.159 M₀, 0.981 M₀, 1.067 M₀: all agree. Differs: unit 0 cut at s+2 "99.7%" → 99.65% only when the next inputs are low; 86.9% over all new maxima, 5.5% when x_(s+1) is another new maximum (minor).

5. §5.4, the old maximum's own unit-5 pulse. 0.967 → 0.020 → 0.9675 → 0.0199 (n = 317k): agrees. This claim is marked "not repeated" in section 9; it is now repeated. New maximum's pulse 0.973 → 0.927 → 0.9735 → 0.9273: agrees. "12.3 above / 12.6 below" → +12.0 / −12.3 on the arm's subset (final logits of a no-overtaking counterfactual; wording nit). "Raises its score by the same 28" → 28.9 against 27.95 with the pulse removed. 13.5 → 41.8 and 20.1 → 15.9 → 13.50 → 41.87 and 20.07 → 15.89. Clock share +28.0 of +28.3 → +28.03 of +28.31: agrees, with a caveat on cancelling terms (nit).

6. §5.4, units 9 and 0 after a new maximum. Unit 9 off: answers the maximum in 44.5% → 44.6% on the arm's subset (final maximum at steps 3–8 that overtook a runner-up ≥ 2 steps old), but 29.7% over all sequences and 11.2% when the runner-up arrives after the maximum (minor: subset not stated). Unit 0 forced on at s+2: 66% of changed answers are the maximum → 65.9% (after every positive new maximum; accuracy 0.762): agrees. Units 5 and 9 both off: never the maximum → 0.0000 in every subset: agrees.

7. §6, last step. Coefficients −49.90, −7.38, −5.01, +4.04, −50.70 → identical; identity error 1.4×10⁻¹³. Slopes 61.0–62.1 and −54.6 to −57.8 → identical from the weights; on data 5–95% = 61.0–62.1. Interval: one interval in 99.66% of the prefixes that ever answer 9 (95.8% of prefixes; nit). Lower edge slope 1.013, R² 0.990 → 1.008, 0.986 (median error 0.022); upper edge 0.998, 0.996 → 0.998, 0.996: agrees. U[8,5] = 0: 0.944 → 0.017 → 0.9425 → 0.0158; U[9,7] = 0: 0.548 → 0.550: agrees. Differs: "h₉[5] rises by 18.3 per unit of x₈ … 62.1" → 18.01 and 61.2 (minor).

8. §7, census and positive inputs (n = 2²¹). Accuracy 0.9529 ± 0.0002 → 0.95309 ± 0.00015; errors 63.6 / 20.2 / 16.2% → 63.5 / 20.3 / 16.2%; runner-up not positive 17.8% → 17.8%; gap < 0.05M 74.6% (54.1 / 15.7 / 4.9) → 74.5% (54.0 / 15.6 / 4.8); two classes 92.4% → 92.3%; 99.97% → 99.971%; 0.119 / 0.231 / 0.979 with 0 / 1 / 2 positive inputs → 0.1194 / 0.2317 / 0.9785 (conditional sampling, 2²⁰ each); the nine shift accuracies agree within 0.001; 0.935 / 0.931 for the maximum last / first → 0.935 / 0.931: all agree.

9. §7, unit-4 detector and repairs (n = 2²¹). Exact detector 0.9314 → 0.9315; false firings removed 0.9563 → 0.9567 (baseline 0.9532); unit 5 off after inputs below the runner-up 0.9588 → 0.9590 (0.9531 → 0.9589 on a second sample; 0.9584 from x/S ≥ 0.9 alone); both 0.9621 → 0.9623; repairs 64% → 65.1% and 65.5% (two samples); 41% → 39.8% and 40.8%; 0.986 → −W[4,4] = 0.9856; unit 4 fires for 19–29% of new runners-up at max-ages 5–6 → 20% and 28–29%: all agree within sampling noise.

10. §8, level removal (quiet continuation, t₀ = 3 / 5 / 7). 97% → 11–21% → 0.972 → 0.110 / 0.210 / 0.198; half level 30–37% → 0.293 / 0.331 / 0.369; 1.25× level 87–94% → 0.872 / 0.870 / 0.942; clock erased costs 5–6 points and cuts nothing → 4.7–6.0 points, 0.00–0.03 units; clock × 1.5 costs 15–22 points → 15.5–22.4: agrees. "About six units cut" → 5.0 in the quiet continuation, 5.7–5.9 with real inputs; "1.6 more units" → 0.7–1.7 depending on t₀ (nit).

## Issues (15)

### [minor] section 5.4 (Step s+2), and the gate table at the top of section 5

- quote: Unit 0 is cut at this step (99.7%), by unit 5 ($-0.913\,h_5 \approx -12.9$) and unit 9 ($-1.424\,h_9 \approx -3.3$).
- problem: The 99.7% holds only under a condition the report does not state. Arm F measured it for new maxima at s = 2..6 followed by three inputs below the old maximum, which is 42% of new maxima; I get 99.65% there. Over all new maxima with M0 > 0, unit 0 is off at s+2 in 86.9% (s = 2..6, n = 940k) and 86.8% (s = 1..7). The split is clean: 99.7% when x_(s+1) is not another new maximum, and 5.5% when it is (13.7% of new maxima), because the sequence then restarts one step later. A later new runner-up, or a new maximum at s+2, does not disturb it (99.6–99.99%). The means -12.91 and -3.30 are confirmed under the arm's condition.
- suggested fix: Write: "Unit 0 is cut at this step (99.7%, unless x_{s+1} is itself a new maximum, which happens in 14% of cases and restarts the sequence; 87% over all new maxima)". Add the same qualifier to "unit 0 cut at t+2" in the table of section 5, and say that the percentages of steps s+1 and s+2 are for new maxima at s = 2..6.

### [minor] section 5.4 (second bullet under "Two things about this sequence")

- quote: With unit 9 forced off after the final maximum, the network answers the maximum itself in 44.5% of cases. With unit 0 forced on at $s+2$, 66% of the changed answers are the maximum.
- problem: The two numbers come from different protocols and subsets, neither stated. (a) 44.5% is for sequences whose final maximum, at steps 3–8, overtook an old maximum that was at least 2 steps old and is the true runner-up; I get 44.6% there (n = 238k), and 46.3% if s = 2 is included. Over all sequences with a positive final maximum at step <= 8 the share is 29.7% (accuracy 0.956 -> 0.674): 51.7% when the runner-up is an overtaken maximum, and 11.2% when the runner-up arrives after the maximum (accuracy 0.960 -> 0.855). (b) 66% is from clamping after every positive new maximum in all sequences (I get 65.9%, accuracy 0.762). On the subset of (a), unit 0 forced on at s+2 makes the network answer the maximum in 32.5%. So the bullet's headline, that unit 9 keeps the standing maximum below the runner-up, is supported mainly when the runner-up is the overtaken old maximum.
- suggested fix: State the subset and protocol of each number, and add the all-sequence figures: unit 9 off after the final maximum gives "answers the maximum" in 30% overall, 52% when the runner-up is the overtaken maximum and 11% when the runner-up arrives later. For unit 0 either give 32.5% on the same subset or say that the 66% is from clamping after every new maximum.

### [minor] section 6 (first bold paragraph) and the section 9 table row "score laws; slope 61 by three routes"

- quote: The input before last is scored through unit 5: $h_9[5]$ rises by 18.3 per unit of $x_8$ and $U[8,5] = 3.40$, which gives 62.1 per unit.
- problem: 18.3 is not the derivative. It is the sum of the comparator-7 and comparator-8 terms only: W[5,7]g_7 + W[5,8]g_8 = 7.81 + 10.46 = 18.28. With all 13 non-pulse units on, dh_9[5]/dx_8 = W[5,ORD]·w[ORD] = 18.009, which gives 61.19 logits per unit. Finite differences on 200k sequences give a median of 18.01 (5–95%: 17.96–18.18; exactly 18.009 when all 13 are on and the pulses are off at step 8). Sections 3.3 ("+18.01 ... 61.19") and 5.2 ("18.009", "61.2") give the correct value, so the report contradicts itself. The section 9 table repeats "62.1 at index 8".
- suggested fix: Replace with "rises by 18.0 per unit of x_8 ..., which gives 61.2 per unit" in section 6, and change "62.1 at index 8" to 61.2 in the section 9 table. If the comparator-only figure is wanted, label it as the contribution of comparators 7 and 8.

### [minor] section 5.2 (display formula and the sentence after it)

- quote: \Theta_5 = 0.01265\,m_{\text{runner}} - 0.02872\,m_{\text{level}} + \text{small}. ... On data $\Theta_5 \approx (0.70\text{–}0.86)\,S - (0.01\text{–}0.07)\,M$
- problem: The two coefficients are confirmed exactly (per unit mode coordinate of h_(t-1), which the report does not say), and the S coefficient is 0.695–0.855 as stated. Two things differ. (1) The remainder is not small soon after a new maximum. On steps with comparators 7 and 8 on and S > 0 (n = 1.56M) the two terms explain 77% of the variance of Θ5; by age of the maximum 1..8 the share is 0.82, 0.22, 0.91, 0.93, 0.95, 0.92, 0.85, 0.78. The median remainder is 0.13 S at age 1 and 0.39 S at age 2, about half of Θ5 there (Θ5 ≈ 0.81 S), and at most 0.09 S from age 3 on. Its source is the pulse units, mainly unit 5's pulse of the previous step: 90th percentile 0.19 M for the pulses of step t-1 and 0.22 M for those of step t, against a median Θ5 of 0.44 M. (2) The fitted M coefficient is -0.071, +0.053, -0.011, -0.035, -0.050, -0.065, -0.035, -0.013 for ages 1..8 (the arm's own table has the same values). At age 2 it is +0.05, outside the stated range and of opposite sign.
- suggested fix: Write "+ pulse terms" instead of "+ small", and say that the two modes give 91–95% of the variance of Θ5 from three steps after a new maximum on, while one and two steps after it the pulses of units 4, 5, 9 add about 0.1 S and 0.4 S. Give the M coefficient as "between -0.07 and +0.05 (positive only two steps after a new maximum)". State that m_runner and m_level are coordinates of h_{t-1}.

### [nit] section 5.2 (sentence introducing the display formula)

- quote: When neither comparator 7 nor 8 is cut at step $t$, unit 5's input at the next step is exactly
- problem: As written (wave-unit clips only), the formula is exact only if comparators 1 and 6 are also uncut. The full identity, which I verified to 6e-14 on 3.6M steps, has two more terms: +0.167 [x_t - θ_6]_+ - 0.005 [x_t - θ_1]_+. With 7 and 8 uncut these are nonzero in 2.6% of steps (largest error 0.12, mean 0.00006, against a spread of z_5 of about 17). Also, Θ5 as defined contains the pulse units of step t. If unit 4 fires at step t while 7 and 8 are uncut (0.03% of such steps), the slope in x_t is 18.009 + 12.057 = 30.07, not 18.009.
- suggested fix: Write "When no comparator is cut and unit 4 is silent at step t", or add the two small terms. Alternatively replace "exactly" by "to within 0.17 times the excess over θ_6".

### [nit] section 5.4 (Step s)

- quote: All four comparators are cut in 94–96%; a small increment can leave one on, as in the last rows of the sweep.
- problem: "94–96%" reads like a range but is a pooled value: 95.5% for s = 2..6 and 94.3% for s = 1..8. By step of the new maximum s = 1..8 the share falls steadily: 98.9, 98.8, 98.2, 95.8, 92.9, 90.3, 88.5, 86.9% (n = 131k–206k per step). In the same way, unit 9 fires at s+1 in 99.5–99.8% for s <= 6 but in 98.5% at s = 7 and 97.2% at s = 8, while the text gives 99.7%.
- suggested fix: Write "in 95% overall (99% for a new maximum at steps 1–3, falling to 87% at step 8)", and note that the 100% / 99.7% for units 5 and 9 are for s = 2..6 (97.2% for unit 9 at s = 8, as section 5.4 says later).

### [nit] section 5.4 (Step s+1)

- quote: $h_5 \approx 12.2\,(M_n - M_0) + 10.4\,M_0 - 5.8\,S_0$ and $h_9 \approx 1.56\,(M_n - M_0) + 1.77\,M_0 - 0.43\,S_0$
- problem: The coefficients are reproduced (12.22, 10.44, -5.79 and 1.56, 1.76, -0.43), but only with the regressor S0+ = max(S0, 0) and on new maxima at s = 2..6 followed by three inputs below M0. With the raw S0, as the formula is written, the fit is 11.9, 8.9, -2.5 for unit 5 and 1.53, 1.67, -0.23 for unit 9. The M0 and S0 coefficients of unit 5 are averages over a strong dependence on the age g of the old maximum: M0 coefficient 9.1, 11.8, 8.8, 9.5, 13.8, 19.7 and S0+ coefficient -4.0, -7.1, -4.5, -5.8, -10.1, -14.4 for g = 1..6. Only the 12.2 is stable (12.20–12.33).
- suggested fix: Write S_0^+ in both formulas and add "pooled over the age of the old maximum; the M_0 and S_0^+ coefficients of unit 5 range over 9–20 and -4 to -14 by age".

### [nit] section 5.4 (first bullet under "Two things about this sequence")

- quote: Without that pulse its score before the overtaking is 12.6 below the other old positions instead of 12.3 above.
- problem: The numbers are not scores "before the overtaking". They are final logits at t = 9 of a counterfactual run in which the new maximum is replaced by a low input, on sequences with the old maximum at least 3 steps old and the new maximum at s >= 4. On that subset I get +12.0 and -12.3; on g >= 2, s >= 3 I get +12.9 and -11.5. The provisional score just before the overtaking (U h_{s-1} read at the old maximum's index, relative to the other old positions) is +18.5 with the pulse and -9.9 without. The sign change, which is the point, holds either way.
- suggested fix: Reword to "Without that pulse, and if it were not overtaken, its final logit would be 12.6 below the other old positions instead of 12.3 above", or quote the provisional scores (+18.5 and -9.9).

### [nit] section 5.4 (Step s, first bullet)

- quote: In an exact decomposition of that rise, the removal of the comparators' share of the clock accounts for $+28.0$ of $+28.3$.
- problem: Reproduced: +28.03 of +28.31 (n = 272k). The near-equality of the two means is partly a cancellation of the other three terms of the same decomposition: -6.1 (non-clock part of the comparators' cut), +0.95 (unit 4's pulse) and +5.4 (gate changes). Per sequence, the rise and the clock term correlate with r = 0.76. By step of the new maximum s = 3..9 the rise is 11.2, 17.6, 24.1, 28.8, 31.6, 32.8, 29.4 and the clock term 15.8, 13.0, 23.4, 33.8, 33.5, 29.9, 27.4. The clock term has its largest value among the earlier positions at the old maximum in 63% of sequences. A reader will take "28.0 of 28.3" as 99% explained.
- suggested fix: Add "on average; the other terms (-6.1, +1.0, +5.4) cancel, and per sequence the rise and the clock term correlate with r = 0.76".

### [nit] section 6 (first bold paragraph)

- quote: For a fixed prefix the network answers "9" on a single interval of $x_9$.
- problem: This is not true for every prefix. In my sweep (20,000 prefixes with S_8 > 0, x_9 from -4 to 7 in steps of 0.004) the network answers 9 for some x_9 in 95.8% of prefixes. Of those, 99.66% have one interval and 0.34% have two. For 4.2% of prefixes no value of x_9 gives the answer 9 (4.0% without the S_8 > 0 filter). The edges agree with the report: lower edge against S_8 has slope 1.008, intercept -0.019, R² 0.986 and median error 0.022 (report: 1.013, 0.990); upper edge against M_8 has slope 0.998, R² 0.996 and median error 0.013. The lower-edge R² depends on the tail: its 5th percentile is at g = -0.46.
- suggested fix: Write "In 96% of prefixes the network answers 9 on one interval of x_9 (two intervals in 0.3% of those; in 4% it never answers 9)".

### [nit] section 4.3 (fourth bullet)

- quote: Setting the clock to zero in the final state changes 7.8% of answers and lowers accuracy by 4.9 points.
- problem: These are the values for final states with M > 0 whose maximum is not the last input, which is arm C's population: I get 7.85% and 95.59% -> 90.65% there. Over all sequences it is 7.4% of answers and 4.6 points (95.31% -> 90.72%). The subset is not stated.
- suggested fix: Either add "(sequences whose maximum is not the last input)" or give the all-sequence numbers, 7.4% and 4.6 points.

### [nit] section 8, point (a), third bullet

- quote: With the level removed from the state at one step, about six units are cut at the next step and accuracy falls from 97% to 11–21%, even if no later input can change the top two. ... enlarging it by half cuts 1.6 more units and costs 15–22 points.
- problem: All accuracies are reproduced, but the two unit counts come from other variants of the experiment. In the quiet continuation, where the accuracies are measured, removing the level switches off 5.0 / 4.9 / 5.0 non-pulse units at t0+1 for t0 = 3 / 5 / 7. The figure 5.7–5.9 is with the real later inputs. "1.6 more units" for the enlarged clock is the t0 = 3 value only: I get 1.3 / 1.0 / 0.7 in the quiet continuation and 1.7 / 1.1 / 0.7 with real inputs.
- suggested fix: Write "five to six units" and "0.7–1.7 more units, depending on the step of the edit".

### [nit] section 7, class (2), and section 1 ("75% have two of the top three within 5% of M of each other")

- quote: **(2) A relative gap below $0.05\,M$ (74.6% of errors: runner-up close to third 54.1%, maximum close to runner-up 15.7%, both 4.9%).**
- problem: Reproduced as 74.5% (54.0 / 15.6 / 4.8), but only with gaps defined on positive parts, (S - max(T,0))/M. Errors amounting to 1.5% of all errors are counted as "runner-up close to third" although the third is not positive; there the runner-up is within 0.05 M of zero, not of the third. With raw values (two of the top three within 0.05 M of each other, S > 0) the share is 73.7%. The report does not say that negative values are set to zero in the gaps.
- suggested fix: Add "(gaps computed with negative values set to zero)" in section 7, and write 74% or "about three quarters" in section 1.

### [nit] section 5.4 (first bullet under "Two things about this sequence")

- quote: Removing the pulse after the *new* maximum costs little (0.973 to 0.927).
- problem: Reproduced on overtaking sequences: 0.9735 -> 0.9273. The sentence can be read as "the unit-5 pulse after the final maximum is unimportant", which is not so. When the runner-up arrives after the final maximum, removing that maximum's own pulse lowers accuracy from 0.960 to 0.860. Over all sequences with a positive final maximum at step <= 8 it goes from 0.956 to 0.877.
- suggested fix: Add "when the answer is the overtaken maximum; if the runner-up arrives later, removing the final maximum's pulse costs 10 points (0.960 to 0.860)".

### [nit] section 3.5, item 3 (outside the sections I was given; an arm-D number measured as a by-product)

- quote: This costs 3 points of accuracy; when the maximum is later overtaken it costs about 44 points.
- problem: Both numbers are for an edit at step 3 only. Erasing the clock at t0 = 3 / 5 / 7 with the real later inputs: if the maximum stays, accuracy goes 0.941 -> 0.911, 0.950 -> 0.913, 0.954 -> 0.913 (3.0, 3.7, 4.1 points). If the maximum is later overtaken, it goes 0.956 -> 0.517, 0.957 -> 0.358, 0.961 -> 0.177 (44, 60, 78 points). The 44 is the smallest of the three.
- suggested fix: Write "3–4 points; 44, 60 and 78 points when the maximum is later overtaken, for an edit at step 3, 5 and 7".

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/rlib.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r01_census.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r01_census_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r02_unit5.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r02_unit5_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r03_newmax.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r03_newmax_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r04_laststep.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r04_laststep_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r05_interventions.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r05_interventions_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r06_level.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r06_level_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r07_extras.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r07_extras_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r08_theta5_terms.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r08_theta5_terms_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r09_repair_shares.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r09_repair_shares_out.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r10_clock_share.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/rev_R4_replicate/r10_clock_share_out.txt