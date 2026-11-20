# Report review N2_sections4to5

## Summary

Sections 4 and 5 checked number by number against v_V4, v_V5, v_V7, rev_R4, arms A, B, F, H and their raw output files. I recomputed what was cheap with my own scripts (fresh seeds 9201–9231, 300k–600k sequences) in c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\rev2_N2\ (n2_weights.py, n2_example.py, n2_sampled.py, n2_extra.py, n2_firststep.py, n2_level.py, n2_splits.py, n2_example_phase.py). Nothing else was edited; no lead script was run.

No major error: nothing found changes the account of the mechanism. 8 moderate, 12 minor, 6 nits.

Reproduced without change: every weight-derived number of 4.1, 4.3, 5.2, 5.4 (B_c, b_2, w_4·b_4, eigenvalues, level vector, 67 of 169, ratio 0.0575, Θ_5 coefficients, pulse writes 168°–189°); the threshold table of 4.1; the mode table of 4.3; the whole table of 5.5 (every cell); every row of the 5.3 sweep, including the margins 1.7 and 13.9 and the two error windows; the shares 53 / 4 / 21 / 21%; 99.9%, 3.7%, 6.5%; unit 0's cut 99.7% / 87% / 14%.

Moderate issues:
1. 5.3: "one, two or three comparators in 11%, 47%, 29%" has one and three swapped (V4's raw table counts comparators ON).
2. 5.3: "74–100% from unit 5" is the share of the mean bump, not of the quoted slope (58–85% at ages 1–4).
3. 5.3: the bump and margin slopes hold only when all later inputs are negative; that condition is missing.
4. 4.3: "no random vector below 0.29" is false (0.29 is the 0.1% quantile; minima 0.22–0.24), and the rank-1 split wins by 0.0575 against 0.0608.
5. 5.2: "91–95% from three steps on" holds for ages 3–6 only (85%, 78% at ages 7, 8).
6. 5.4: the "exact" first-step factor 0.958@37.0° omits the cut of unit 0; with it the factor is 1.00@36.0°, which is what is measured and what 4.1 says.
7. 4.1: 0.82–0.997 belongs to V8's oracle (fitted multiple of the running max), not to the model described (0.64–0.995).
8. 4.1: unit 4's precision 0.99 / recall 0.98 are for the compensated variant; in the plain cut precision is 0.91.

Numbers with no source in the notes, computed here: runner law (1.86 M + 5.57 S⁺, so "2M + 5S" is a little off); 53% and 4% of the section 5 table (53.4%, 3.7%); sweep margins (1.66, 13.87); level leak sign (−0.0447).

## Issues (26)

### [moderate] section 5.3

- quote: A positive new runner-up cuts one, two or three comparators in 11%, 47% and 29% of cases, and none in 14%.
- problem: One and three are swapped. V4's raw output (hide/v_V4_core/v9_followup.txt, t = 2..9, old max > 0.3) is P(number of comparators ON = 0..4) = 0.0001, 0.1061, 0.4665, 0.2853, 0.1421. So three are cut in 10.6%, two in 46.7%, one in 28.5%, none in 14.2%. V4's note wrote 'switches off 1, 2, 3' for these 'on' counts and the report copied it. My own count (600k sequences, all positive new runners-up, t ≥ 1): none 12.6%, one 25.6%, two 51.0%, three 10.8%, four 0%. The per-age 'none' rates of the same bullet reproduce (0.0, 0.2, 6.6, 9.5, 52.8, 66.2, 58.6, 42.9%).
- suggested fix: A positive new runner-up cuts three, two or one comparators in 11%, 47% and 29% of cases, and none in 14%.

### [moderate] section 5.3

- quote: 74–100% of it comes from unit 5
- problem: Number attached to the wrong quantity. In arm H's output (hide/r1_H_runnerup_events/h3_analyze2_out.txt) 74–100% is unit 5's share of the MEAN logit change at position t (16.8 of 22.7 M at age 1, 20.1 of 21.2, 14.6 of 18.2, 14.2 of 17.9, then 94–100% at ages 5–8). The sentence quotes the slope per unit of x_t. Of that slope unit 5 gives 27.6 of 47.8, 40.2 of 47.1, 28.0 of 42.1, 28.8 of 42.9 at ages 1–4 (58%, 85%, 67%, 67%) and 51.6 of 58.2, 61.7 of 61.0, 60.2 of 56.8, 61.5 of 61.2 at ages 5–8 (89–106%). The cut comparators give 15–24 per unit at ages 1–4.
- suggested fix: Of this slope unit 5 supplies 58–85% when the maximum is 1–4 steps old (the cut comparators add 15–24 per unit) and all of it at ages 5–8. Of the mean bump (18–23 M) it supplies 74–100%.

### [moderate] section 5.3

- quote: The result is a bump at position $t$ of $+42$ to $+48$ per unit of $x_t$ when the maximum is 1–4 steps old
- problem: Missing condition. These slopes, the fall of 9–23 at the old runner-up and the final margin (58–72)(x_t − S) are arm H's numbers for events whose later inputs are all negative (35,609 of 235,412 events, 15%; claims H3b, H3c). Arm H's caveat: with all followers the slope at position t is 24–42 at ages 1–4 and 54–61 at ages 5–8, and P(margin > 0) is 0.75–0.94 instead of 0.92–0.98. The runner-mode and clock numbers of points 2 and 3 are likewise for events followed by two negative inputs.
- suggested fix: Add: "(measured when all later inputs are negative; over all continuations the slope at position t is 24–42 at ages 1–4 and 54–61 at ages 5–8)", and put the same qualifier on the final margin (58–72)(x_t − S).

### [moderate] section 4.3

- quote: It is special: it is the smallest among all 715 ways of splitting the 13 units into 4 and 9, and no random vector in 20,000 comes below 0.29.
- problem: (a) Wrong: 0.29 is the 0.1% quantile, not the minimum. V5's detail line gives 0.1% quantiles 0.290 (vectors with the clock's magnitudes and random phases) and 0.355 (random complex vectors) and says only 'none at or below 0.0575'. Recomputed with 20,000 draws each: 18 and 5 vectors are below 0.29; the minima are 0.22 and 0.24; medians 0.71 and 0.75. (b) The rank-1 statement is true but lacks its context: all 715 splits give 0.0575–0.260 (median 0.106). The next ones are 0.0608 for {0,1,6,8}, 0.0630 for {0,1,7,8}, 0.0632 for {10,12,14,15} (no comparator), and 28 splits are below 0.07. What is special is the clock vector, which is nearly in the kernel of U_o: |U_o v| / Σ_i |U_o[:,i] v_i| = 0.031.
- suggested fix: The clock vector is special: for random vectors the ratio has median 0.75, and the smallest of 20,000 is 0.22 (0.1% quantile 0.29–0.35). The split matters less. The comparators give the smallest ratio of the 715 splits into 4 and 9, but every split gives 0.058–0.26 (median 0.11, next best 0.061), because the clock vector as a whole is nearly invisible to the readout (0.031 of its per-unit pieces).

### [moderate] section 5.2

- quote: From three steps after a new maximum on, these two modes give 91–95% of the variance of
- problem: Wrong range. R4's raw output (hide/rev_R4_replicate/r08_theta5_terms_out.txt) gives the share by age of the maximum 1..8 as 0.82, 0.22, 0.91, 0.93, 0.95, 0.92, 0.85, 0.78. So 91–95% holds for ages 3–6 only; at ages 7 and 8 it is 85% and 78% (median remainder 0.07 S at age 8). R4's own suggested wording had this slip and the report copied it.
- suggested fix: Three to six steps after a new maximum these two modes give 91–95% of the variance of Θ_5 (85% and 78% at seven and eight steps).

### [moderate] section 5.4

- quote: = 0.958@37.0°$ for the factor from $s+1$ to $s+2$, exactly.
- problem: The arithmetic is right, but this is not the factor. It leaves out the cut of unit 0 at s+2, which the previous paragraph says happens in 99.7% of cases. That cut adds LEFT_clock[unit 0] · clip / h_4 = 0.110@15.7° × 4.1 / 10.157 ≈ 0.045@15.7° (0.049 with arm F's mean clip 4.52). With it the factor is 1.00@36.0°. Measured: 0.998@36.3° for x = (1, 0, …, 0), 1.017@35.7° for x = (1, −1, …, −1). Arm F (F2.b), per unit of new gap: pulses only 9.75@36.7°, pulses plus unit 0's cut 10.21@35.7°, measured 10.54@36.3°. So the report gives two values for the same step: 0.958@37° here, and in 4.1 α(1) = 1.01 → α(2) = 1.03 with 'a first step of 36°'.
- suggested fix: Its first step is 36° because of the two pulses and the cut of unit 0: 0.951@42.7° + 0.088@−137.9° + 0.133@−4.4° + 0.045@15.7° = 1.00@36.0° for the factor from s+1 to s+2 (the pulses alone give 0.958@37.0°; measured on x = (1, 0, …, 0): 0.998@36.3°).

### [moderate] section 4.1

- quote: A model with no dynamics at all (every threshold equal to the running maximum) already reaches $R^2$ = 0.82–0.997 for the six activations.
- problem: Number attached to the wrong model. 0.82–0.997 is V8's clock-free oracle with one fitted constant per unit, θ_j = c_j × running max: 0.979, 0.997, 0.879, 0.876, 0.988, 0.820 for units 2, 4, 1, 6, 7, 8. The model described in the sentence is V4's yardstick (all thresholds equal to the running max), which gives 0.979, 0.995, 0.833, 0.826, 0.984, 0.636.
- suggested fix: Either "(every threshold a fixed multiple of the running maximum) already reaches R² = 0.82–0.997", or "(every threshold equal to the running maximum) already reaches R² = 0.98–0.995 for units 2, 4, 7 and 0.64–0.83 for units 1, 6, 8".

### [moderate] section 4.1

- quote: Unit 4 still fires correctly (precision 0.99, recall 0.98).
- problem: Missing condition. These are V4's numbers for the variant with the fixed multiple of unit 2 as replacement (49.0%): P(new max | fire) = 0.990, P(fire | new max) = 0.979. For the plain cut (50.6%), whose core equals the six-unit simulation, V4 gives recall 0.9999 but precision 0.913 overall, falling to 0.729 at t = 9 (27% of the firings at the last step are false). The conclusion stands on V4's separate ablations: cutting the wave units' weights into unit 4 alone gives 92.4%, into the comparators alone 54.9%.
- suggested fix: The damage is in the comparator thresholds: cutting only the wave units' weights into unit 4 costs 3 points, into the comparators 40. With the replacement, unit 4 still fires correctly (precision 0.99, recall 0.98); in the plain cut its precision is 0.91 (0.73 at the last step).

### [minor] section 4.1

- quote: sags to 0.84 when it is 5–6 steps old
- problem: Off by one against the table three paragraphs below, where unit 4's threshold is lowest at ages 6 and 7 (0.64, 0.63), and against arm H and V7 (ages 6–7). V4's 0.837, 0.840 are indexed by the age of the maximum in the state that produces the threshold; when the input meets that threshold the maximum is one step older. Recomputed, θ_4 / M_old by age of the old maximum at the comparison, 1..9: 0.99, 0.98, 0.97, 0.93, 0.87, 0.84, 0.84, 0.88, 0.93.
- suggested fix: θ_4 is 0.98 of the old maximum when that is young and sags to 0.84 when it is 6–7 steps old (ages counted as in the table below).

### [minor] section 5.1

- quote: per unit of input the level coordinate changes by 0.045 and the runner coordinate by 1.026
- problem: Sign and unit. Recomputed: LEFT_level · w[ORD] = −0.0447 and LEFT_runner · w[ORD] = +1.0264 (arm B gives the same). The two leaks have opposite signs. In units of w_4 they are −0.0044 and +0.101; section 4.3 quotes the runner leak as 0.10 in those units, so 1.026 here and 0.10 there read as two values. In the gauge fixed in 4.1 the echo entry of section 4.3 is −15.3, not +15.3.
- suggested fix: per unit of input the level coordinate changes by −0.045 and the runner coordinate by +1.026 (−0.004 and +0.10 in units of w_4); the second is a tenth of what a new maximum writes. In 4.3: "adds −15.3 to the echo coordinate".

### [minor] section 4.1

- quote: is between 0.02 and 0.38. It is largest just after a new runner-up (0.30) and decays with the runner-up's age (0.05 at age 8).
- problem: (a) In V5's fit β is complex, and 0.02–0.38 is its modulus. Its phase is not φ(a), so the formula z ≈ w_4[α(M − S⁺) + βS⁺]e^{iφ(a)} with one common phase is not what was fitted. Examples from hide/v_V5_modes/v5_b_m5m6.txt: a_1 = 2, a_2 = 0: α = 1.02@37°, β = 0.36@−26°; a_1 = 1, a_2 = 5: α = 1.05@1°, β = 0.17@85°; a_1 = 3, a_2 = 1: α = 0.98@80°, β = 0.33@16°. (b) The decay 0.30 → 0.05 is V5's list for cells a_2 < a_1, a runner-up that arrived after the maximum. When the runner-up is the overtaken old maximum, |β| does not decay: for a_1 = 1 it is 0.19, 0.24, 0.21, 0.17, 0.08, 0.13, 0.25, 0.33 at a_2 = 2..9.
- suggested fix: Write the law as z ≈ w_4[α(a)e^{iφ(a)}(M − S⁺) + βS⁺] and: "β is a complex number with its own phase and a modulus between 0.02 and 0.38. For a runner-up that arrived after the maximum, |β| is largest just after its arrival (0.30) and decays with its age (0.05 at age 8)."

### [minor] section 5.3

- quote: because the lowest threshold is then 0.15–0.19 $M$ above $S$
- problem: True for three of the four ages named. V4's medians (hide/v_V4_core/v8_extra.txt, all states with M > 0.3) are 0.159, 0.189, 0.146, 0.066 M for a maximum that is 5, 6, 7, 8 steps old when the input arrives. Recomputed: 0.16, 0.19, 0.15, 0.07.
- suggested fix: because the lowest threshold is then a median 0.15–0.19 M above S (0.07 M at age 8)

### [minor] section 4.1

- quote: It holds in the running network with all side paths included. With all four comparators on,
- problem: The bound ≤ 0.011 that follows is the largest of four mean absolute values (V4: means 0.007, 0.006, 0.001, 0.011 with all four comparators on and unit 4 off), not a bound over states, and the condition 'unit 4 silent' is missing. Recomputed on 3.47M such steps: means 0.007, 0.007, 0.001, 0.011; maxima 0.013, 0.016, 0.002, 0.014. On the 0.005% of steps with all four comparators on and unit 4 firing, the derivative is 0.84 for θ_1 and 1.02 for θ_7.
- suggested fix: With all four comparators on and unit 4 silent, |∂θ_j(t+1)/∂x_t| is 0.001–0.011 on average and at most 0.016.

### [minor] section 5.4 (same numbers in 3.4)

- quote: All four comparators are cut in 95% of new maxima: 99% at steps 1–3, falling to 87% at step 8.
- problem: Selections are mixed. 95.5% is for new maxima at steps 2–6 (arm F, R4). Over all steps it is 94%: R4 94.3% for s = 1..8, arm A 93.7%; recomputed 93.7% for s = 1..9, by step 98.8, 98.9, 98.1, 95.7, 93.0, 90.3, 88.5, 86.6, 87.0%. The 23–50% for an excess of at most 5% is V2's range over steps 3–9 for an old maximum above 0.3 that is at least two steps old on arrival (reproduced: 0.48, 0.36, 0.31, 0.27, 0.25, 0.23, 0.25). For all new maxima over a positive old one it is 30–69% by step (42% pooled). It depends mainly on the old maximum's age on arrival: 68%, 73%, 43% at ages 1, 2, 3 and 1–7% at ages 4–8.
- suggested fix: All four comparators are cut in 94% of new maxima (95.5% at steps 2–6): 98–99% at steps 1–3, falling to 87% at steps 8 and 9. A small increment can leave some on: with an excess of at most 5%, all four are cut in about 70% of cases when the old maximum is 1–2 steps old, in 43% at 3 steps and in under 10% when it is older; above 30% excess, in at least 99%.

### [minor] section 5.5

- quote: Both come from what is left of the previous clock
- problem: Not true for the phases at steps 2 and 5. The maximum at position 0 has no previous clock, yet its phase at step 2 is 25° instead of 36°. Replacing x_2 = 0.3 by −0.5 (no new runner-up) gives 0.81@36°; replacing x_5 = 0.9 by −1.0 gives 0.56@39° instead of 0.44@31°. So the two new runners-up, which cut comparators, pull the phase back by 11° and 8°. Only the 9° and 13° one step after the second and third maximum, and the two larger radii, come from the previous clock (maximum 1.2 at step 3 with nothing positive before it: 1.20@0°, then 1.21@36°). This also qualifies "with little change of phase" in 5.3: V5 found the phase advance of that step cut to 19° and 30° when the gap shrinks to 5% and 18% of its old size.
- suggested fix: The 9° and 13° one step after the second and third maximum, and their larger radii, come from what is left of the previous clock. The 25° and 31° at steps 2 and 5 (law: 36°) come from the new runners-up: without them the phases are 36° and 39°. In 5.3 replace "with little change of phase" by "and the phase falls back by about 10° (13–24° when the gap shrinks to less than a fifth)".

### [minor] section 4.1

- quote: What the satellites add to a threshold is small: 0.04–0.11 $M$ on average
- problem: The numbers of this bullet (0.04–0.11 M, 0.003–0.175 M, 23–66%, 54.5%, 58.0%, 83.1%) are V4's for the eight other wave units (0, 3, 10–15), not for the ten satellites as section 2 defines them. The two pulse satellites are a separate, smaller item in V4: cutting the weights from units 5 and 9 into the core gives 84.6%.
- suggested fix: What the eight wave satellites add to a threshold is small: … Removing it gives 54.5% accuracy … (Cutting the weights from units 5 and 9 into the core instead gives 84.6%.)

### [minor] section 4.2

- quote: for every age of the runner-up: unit 0 at age 2 (98–100% of states), unit 11 at age 5 (95–99%), unit 10 at age 6 (88–99%)
- problem: Missing conditions. V7's ranges are for M > 0.3 and a runner-up at least one step old. When the runner-up is the current input, unit 11 is zero in 74% (V7 objection 1c). For M in (0.1, 0.3] the rates are 70% (unit 11) and 64% (unit 10). When the maximum is the first positive input they are 100%, 88%, 95% for units 0, 11, 10, so "This also happens" is at a lower rate for unit 11.
- suggested fix: for every age ≥ 1 of the runner-up (maximum above 0.3): unit 0 at age 2 (98–100% of states), unit 11 at age 5 (95–99%; 74% when the runner-up is the current input), unit 10 at age 6 (88–99%). It also happens when the maximum is the first positive input (100%, 88%, 95%), and for x = (1, 0, …, 0).

### [minor] section 4.1

- quote: and then grows by 1% per step (the eigenvalue 1.0099)
- problem: True only on steps not preceded by a pulse of unit 5 or 9. Recomputed (500k sequences, maximum at least 2 steps old, no new maximum): the factor is 1.011 after a step with units 5 and 9 silent (73% of steps) and 0.98 after a step with one of them on (27%); median 1.0095, mean 1.003. So on data the level does not grow: median level/(w_4 M) by age 2..9 is 0.93, 0.92, 0.92, 0.92, 0.92, 0.93, 0.93, 0.94 (V5: 0.918 … 0.935), and the same inside a fixed cohort (final maximum at position 0: 0.931 at age 2, 0.936 at age 9). Only the lone impulse grows from 0.93 to 0.99. A reader of the sentence would expect 0.99 w_4 M at age 9.
- suggested fix: It is 0.99 w_4 M one step after a new maximum and 0.92–0.94 w_4 M from age 2 on. It grows by 1% on a step without pulses (the eigenvalue 1.0099) and falls by 2% after a pulse of unit 5 or 9.

### [minor] section 5.2

- quote: unit 5's input at the next step is exactly
- problem: The first equation is exact, but the expression given for Θ_5 under the same "exactly" leaves out the other seven modes. Arm H (H4a), per unit coordinate: third harmonic 0.01132, second harmonic 0.00958, short clock 0.00751, transient −0.00509, clock 0.00365, alternating −0.00171. R4's table gives their median sizes as 0.022, 0.011, 0.016, 0.004, 0.006, 0.006 M (90th percentiles up to 0.056 M) against a median Θ_5 of 0.44 M. They are the missing 5–9% of variance of the next sentence.
- suggested fix: Θ_5 = 0.01265 m_runner − 0.02872 m_level + (pulse terms) + (terms of the other seven modes, each at most 0.02 M in the median)

### [minor] section 5.4

- quote: the old maximum's score rises by 28 on average: from 13.5 to 41.8
- problem: Placed under "Three consequences follow at once" at step s, but arm F's numbers (F3.d) are not scores before and after step s. They compare the final logits at t = 9 of the real run with those of a run in which the new maximum is replaced by a low input, on sequences where the new maximum (steps 3–9) is the final one and the old maximum (at least 2 steps old) is the true runner-up. R4 corrected the same wording for 12.3 / −12.6 further down; this sentence was left.
- suggested fix: In the final logits, relative to the other old positions, the old maximum stands 28 higher than in a run without the new maximum: 41.8 against 13.5, while the old runner-up stands at 15.9 against 20.1 (new maximum at steps 3–9, old maximum at least 2 steps old).

### [nit] section 4.3

- quote: Their sum is 0.4–1.6, which is 5.75% of the two sizes added.
- problem: The sizes and the sum are for positions 0–7, but 5.75% is the ratio of the norms over all ten positions. Over positions 0–7 it is 5.50% (V5 M3; recomputed 0.0550 and 0.0575).
- suggested fix: Their sum is 0.4–1.6, which is 5.5% of the two sizes added (5.75% over all ten positions).

### [nit] section 4.2

- quote: between 7% and 21% of it remains as $S$ approaches $M$
- problem: Inconsistent with the c of the same sentence: c = 0.79–0.95 means 5–21% remains (V7's straight-line shares: 0.11, 0.08, 0.21, 0.05 for units 1, 6, 7, 8). The 63% of the next sentence is not in V7's list either (0.20–0.59 for units 0, 10, 11, 12, 14, 15).
- suggested fix: between 5% and 21% of it remains as S approaches M. In the wave units more remains (20–59%).

### [nit] section 5.1

- quote: One step later its slot holds the floor of an unrecorded input, 10 to 20 $M$ below the runner-up.
- problem: Three values for one quantity: section 3.2 says −12 M, the table of 3.3 says −10 to −18, here 10 to 20. Recomputed for inputs below the runner-up with unit 5 silent: median −12.0 M, medians by age of the maximum −10 to −15, 10–90% range −6 to −20.
- suggested fix: One step later its slot holds the floor of an unrecorded input, 10 to 15 M below the runner-up (median 12).

### [nit] section 4.1 (and the table of 4.3)

- quote: (2M + 5S)$, roughly. It sets unit 5's threshold (section 5.2).
- problem: Unsourced as written; no verifier number. Arms C, D, H give (0.9–2.0) M + (4.9–6.4) S. Recomputed: runner/w_4 = 1.86 M + 5.57 S⁺ for maxima at least 2 steps old (by age 2..9: 1.7–2.0 M + 4.9–6.0 S⁺; 0.9 M + 5.6 S⁺ at age 1; centred R² 0.89). With 2M + 5S the formula of 5.2 gives Θ_5 = 0.64 S − 0.02 M; with 1.9M + 5.6S it gives 0.72 S − 0.03 M, which is what 5.2 reports.
- suggested fix: Runner ≈ w_4 (1.9 M + 5.6 S⁺), roughly (in the table of 4.3: about w_4 (1.9M + 5.6S⁺)).

### [nit] section 5.3 (sweep table)

- quote: | none | 0 | 0 | 0.90 | 0, correct |
- problem: At v = 0.30 unit 5 already fires: 0.084 at the next step (V7: 0.0841; recomputed). It switches on at v = 0.296, so the row label ≤ 0.3 does not match the entry 0.
- suggested fix: Label the row "≤ 0.29" (unit 5 is 0.1 at v = 0.3).

### [nit] section 5.5

- quote: The top of $m_t$ is the maximum so far (0, 0, 0, 3, 3, 3, 3, 7, 7).
- problem: At steps 3 and 7 the maximum so far is x_t itself (positions 3 and 7). The list gives the largest of the inputs before x_t, as the table header says. The table itself reproduces exactly.
- suggested fix: The top of m_t is the largest of the inputs before x_t (0, 0, 0, 3, 3, 3, 3, 7, 7).

## Files
