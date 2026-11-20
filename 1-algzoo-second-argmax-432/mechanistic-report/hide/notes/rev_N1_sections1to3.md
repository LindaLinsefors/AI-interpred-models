# Report review N1_sections1to3

## Summary

Checked every number and quantitative statement of sections 1, 2, 3 and the figure caption against v_V1, v_V2, v_V3, v_V8, v_V4, v_V5, rev_R4, the arms' digest, the lead scripts' saved outputs and the figure cache, and recomputed the doubtful or unsourced ones (scripts and outputs in c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\rev2_N1\: n1_weights.py, n1_data.py, n1_extra.py, n1_wipe.py, n1_figcache.py, each with an _out.txt). Almost all numbers trace to a verifier and agree. I list 19 issues: 6 moderate, 7 minor, 6 nits, and no major one. The moderate ones: (1) the summary says the clock phase is the same 'whatever the values', but it lags by 8–14° at a relative gap of 0.1–0.3 and by 12–32° below 0.1; (2) the caption of panel (f) omits the condition 'the maximum is not the current input', without which the orange rate is 0.65–0.86; (3) in 3.4 the rates 'margin positive' hold for D ≤ 6 only; (4) in 3.4 the rates of 'all four comparators zero at a new maximum' mix two selections (pooled value is 94%, and 23–50% belongs to a narrower selection; under the stated one it is 30–67%); (5) in 3.4 the per-sequence ratio −1.03 leaves out V2's companion number, a median residual of 0.36; (6) in 3.1 the running answer equals the accuracy on zero-prefixed inputs only with the answer restricted to the real inputs (unrestricted: 0.932 instead of 0.992 at t = 1). Statements with no number in the notes that I computed and found correct, so not listed: the input weights of section 2 (wave units at most 0.298 in size, unit 14 at −1.321); the gate shares 0.661 / 0.2155 / 0.1235 (0.4998 at t = 0, 0.8185 at t = 9), where all-off states are new maxima in 99.99% and states with one to three on are new runners-up in 89%; the table values −55 and −47 of section 3.3 (figure cache: −55.3 and −47.3); the Σ entries, cosines and relative errors. Third-decimal differences between the lead's 400k-sample values and the verifiers' larger samples (for example 0.966 against 0.965 at t = 4 in the table of 3.1, 0.942 against 0.943 in 3.4) are sampling noise and are not listed.

## Issues (19)

### [moderate] section 1

- quote: 338° when the maximum is 1..9 steps old, whatever the values.
- problem: The list is the phase of the gap's coefficient α (V5: phase of the pooled α per age, S > 0.05), which is the phase of z only when the gap is large. The phase of z itself lags as S approaches M. V5: median |arg z − φ(a)| is 0.6–2.6° for a relative gap above 0.6, 3.2–6.5° for 0.3–0.6, 8–15° for 0.1–0.3 and 18–33° below 0.1. My check (hide/rev2_N1/n1_extra.py, 400k sequences, M > 0.3) shows a systematic lag, not scatter: median phases at ages 2..9 are 23°, 66°, 112°, 157°, 201°, 241°, 285°, 324° for a relative gap of 0.1–0.3 and 8°, 47°, 98°, 146°, 193°, 239°, 273°, 315° below 0.1. Panel (a) of the figure shows the same (inner ring at 24°, 67°, 113°, 157°, 201°, 241°, 285°, 325°). The cause is that the S⁺ term of the clock law has its own phase (β is complex in V5's fit); the formula in section 4.1 puts both terms under one phase factor and hides this.
- suggested fix: The phase of z is 0°, 36°, 79°, 123°, 166°, 209°, 251°, 294°, 338° when the maximum is 1..9 steps old and the gap is large (within about 3° for a relative gap above 0.6). It lags as the runner-up approaches the maximum: by 3–7° for a relative gap of 0.3–0.6, by 8–14° for 0.1–0.3 and by 12–32° below 0.1.

### [moderate] section 1 (figure caption)

- quote: in which the largest entry of $m_t$ among the inputs before $x_t$ is the maximum so far (orange)
- problem: Missing condition. fig_mechanism.py (lines 107–111) computes the orange curve only over states in which, besides S_t > 0, the maximum is not the current input (p1 < t). The caption and the y-axis label give only 'states with a positive runner-up'. Read literally, over all states with S_t > 0, the fraction is 0.65, 0.71, 0.76, 0.79, 0.82, 0.84, 0.86, 0.80 for t = 2..9 (n1_data.py, 1.2M sequences), not the plotted 0.972, 0.943, 0.948, 0.953, 0.958, 0.964, 0.966, 0.891. Section 3.4 states both conditions.
- suggested fix: (f) Fraction of the states with a positive runner-up in which the largest entry of s_t is the runner-up so far (blue); and, among those states whose maximum is not the current input, the fraction in which the largest entry of m_t among the inputs before x_t is the maximum so far (orange).

### [moderate] section 3.4

- quote: The margin is positive in at least 99.7% of cases for relative gaps between 0.05 and 0.8, in 98.6–99.5% above 0.8, and in 85–96% below 0.05.
- problem: These ranges hold for D ≤ 6 only (V2's S5 covers D = 1..6), but the sentence has no such limit and the next one quotes D = 1..8. V2 for D = 7 and 8: 0.994 and 0.996 at relative gaps of 0.05–0.15, and 0.845 and 0.794 below 0.05. My recomputation (n1_data.py, 1.2M sequences): D = 7 gives 0.995, 0.999, 0.998, 0.997 in the four bins from 0.05 to 0.8, 0.976 above 0.8 (n = 574) and 0.845 below 0.05; D = 8 gives 0.994, 1.000, 1.000, 0.999, then 0.991 (n = 319) and 0.803. Also, '98.6–99.5% above 0.8' is V2's range for D = 2..6; for D = 1 it is 100.0% (V2 1.0000, mine 1.0000).
- suggested fix: For D ≤ 6 the margin is positive in at least 99.7% of cases for relative gaps between 0.05 and 0.8, in 98.6–100% above 0.8, and in 85–96% below 0.05. For D = 7 and 8 it is 99.4–99.6% at gaps of 0.05–0.15, 98–99% above 0.8, and 84% and 79–80% below 0.05.

### [moderate] section 3.4

- quote: All four comparators are exactly zero in 95% of new maxima over a positive old maximum: 99% at steps 1–3, falling to 87% at step 8. The rate is 23–50% when the new maximum exceeds the old one by at most 5%
- problem: Two selections are mixed. (a) 99% and 87% are R4's by-step values for all new maxima over a positive old maximum (98.9, 98.8, 98.2, 95.8, 92.9, 90.3, 88.5, 86.9% for steps 1..8). Pooled over those steps that is 94.3% (R4), and 93.7% with step 9 (arm A: 0.937; mine: 0.937). 95.5% is R4's value for steps 2–6 only. (b) 23–50% is V2's range under the narrower selection of claim S6 (old maximum above 0.3 and at least two steps old, steps 3–9), where the overall rates are 84–97%. Under the selection the sentence states, the rate for an excess of at most 5% is 63, 67, 54, 45, 38, 34, 32, 30, 32% at steps 1..9 and 42% pooled (n1_data.py, 1.2M sequences; under V2's selection I reproduce 48, 37, 31, 27, 25, 23, 26% for steps 3..9). Above 30% it is at least 99.8% under both. The sentence is repeated in section 5.4, and the table of section 9 quotes the third range (84–97%).
- suggested fix: All four comparators are exactly zero in 94% of new maxima over a positive old maximum: 99%, 99%, 98% at steps 1–3, falling to 87% at steps 8 and 9. When the new maximum exceeds the old one by at most 5% the rate is 42% (63–67% at steps 1–2, about 30% at steps 7–9), and it is at least 99.8% when the excess is above 30%. If the old maximum is at least two steps old the rates are lower: 84–97% overall and 23–50% for an excess of at most 5% (steps 3–9).

### [moderate] section 3.4

- quote: the wave in $m$ is $-1.03$ times the wave in $V$ (median ratio of their complex amplitudes)
- problem: −1.03 is the median of the real part of Z_m/Z_V (V2; I get −1.028, with a median modulus of 1.11). V2 gave a second number that the sentence leaves out: the median of |Z_m + Z_V|/|Z_V| is 0.36 (I get 0.360; n1_data.py, 860k sequences, t = 9, all four comparators on, S > 0.1, maximum 2–9 steps old). So within a single sequence the two first-harmonic amplitudes over indices 0–7 cancel to about a third, not to 3%. The cancellation to within 8 per unit of gap is a property of the regression coefficients.
- suggested fix: Sequence by sequence the cancellation is rougher. The ratio Z_m/Z_V of the two complex amplitudes over indices 0–7 has a median real part of −1.03, but the median of |Z_m + Z_V|/|Z_V| is 0.36, because s itself has a first-harmonic component.

### [moderate] section 3.1

- quote: By fact 3 of section 2, this is the network's accuracy on zero-prefixed inputs.
- problem: Not exactly. The table uses the largest entry among the indices of existing inputs, while the network's answer on the zero-prefixed sequence is the largest of all ten logits. V1: the unrestricted largest entry is on a position that does not exist yet in 6.1%, 5.0%, 2.2%, 1.3%, 0.5% of the S_t > 0 states at t = 1..5, and V1 asked for this to be stated next to the numbers. My check (n1_extra.py, 400k sequences): given S_t > 0, the unrestricted accuracy is 0.932, 0.940, 0.951, 0.958, 0.966, 0.968, 0.968, 0.966, 0.961 for t = 1..9, against 0.992, 0.987, 0.972, 0.965, 0.969, 0.970, 0.970, 0.967, 0.961 restricted. The same applies to '96–99%' in the summary ('the final readout can be applied at any step'): applied literally it gives 93–97%.
- suggested fix: By fact 3 of section 2, this is the network's accuracy on zero-prefixed inputs when its answer is restricted to the real inputs. Unrestricted, the largest logit is on one of the zeros in 6%, 5%, 2%, 1% of these states at t = 1..4, and the accuracy is 0.932, 0.940, 0.951, 0.958 (from t = 5 on the two differ by at most 0.003). In the summary: (96–99% of the states whose runner-up is positive, with the answer restricted to the inputs so far).

### [minor] section 1

- quote: The network then answers the maximum in 98.6% of sequences and the runner-up almost never.
- problem: The number holds for the selection of claim I1, which the summary does not state: final runner-up above 0.2 and final gap above 0.2 (69.6% of sequences). Section 3.5 gives only the gap condition ('sequences with a final gap above 0.2'). I reproduce the number under that selection (0.9859, runner-up 0.0018; n1_wipe.py, 400k sequences). Over all sequences with a positive maximum the same wipe at t_0 ≤ 7 gives the maximum in 94.4% and the runner-up in 2.0%; for a runner-up above 0.2 with a gap of at most 0.2 it gives 83.5% and 6.7%. V3's table by gap agrees.
- suggested fix: In sequences whose runner-up and gap both exceed 0.2 (70% of all), the network then answers the maximum in 98.6% and the runner-up in 0.2% (94% and 2% over all sequences). In section 3.5: (sequences with a final runner-up above 0.2 and a final gap above 0.2).

### [minor] section 1

- quote: Removing only the comparators' share of the clock pattern does the same (94–100%).
- problem: The range is V3's for a maximum 2 to 7 steps old (0.997, 0.977, 0.968, 0.983, 0.964, 0.943). One step after a new maximum the same removal gives the maximum in 0.32 (V3: 'at k = 1 the split does not work'; its outputs a6 and a9 give 0.317–0.318, with the runner-up answered in 24%). The summary gives no age condition, and section 3.5 says 'for k = 2..7' without saying what happens at k = 1.
- suggested fix: Removing only the comparators' share of the clock pattern does the same when the maximum is at least two steps old (94–100%; one step after a new maximum only 32%). In section 3.5 add '(at k = 1: 0.32)' after 'for k = 2..7'.

### [minor] section 3.5

- quote: against 0.95–0.97 (maximum 2–7 steps old)
- problem: The restriction hides the one age at which the clock mode is not silent. V3: with the whole clock mode removed one step after a new maximum and no new maximum following, accuracy is 0.561 against 0.969, and the maximum is answered in 35.9%. A reader of 'held in reserve' will assume that 0.91–0.95 holds from the first step.
- suggested fix: With no new maximum following, removing it leaves accuracy at 0.91–0.95, against 0.95–0.97, when the maximum is 2–7 steps old. One step after a new maximum it is not silent: accuracy falls to 0.56 (from 0.97) and the maximum is answered in 36%.

### [minor] section 1

- quote: The ten feed small corrections into the thresholds (0.003 to 0.175 $M$, depending on the age of the maximum)
- problem: Number attached to the wrong set of units. 0.003–0.175 M is V4's range for the part of the thresholds that comes from the eight wave units other than unit 2 (claim K3: means by age 1..8 for θ_1, θ_7, θ_8), not from all ten satellites. Pulse units 5 and 9 are not in it; from V4's totals their mean part in those states is between −0.002 and +0.014 M. Section 4.1 has the same attribution ('What the satellites add to a threshold').
- suggested fix: Among them, the eight other wave units feed small corrections into the thresholds (0.003 to 0.175 M on average, depending on the threshold and the age of the maximum)

### [minor] section 3.4

- quote: At the newest index, $V[9]
- problem: Unsourced as a fit: −50 M − 7 S appears only as the lead's rounded reading in the lab notebook. V2 (claim S3) gives, for ages 2..9 of the maximum, A_9 = −54.2, −59.2, −56.0, −47.7, −39.3, −34.9, −36.7, −43.5 on (M − S) and B_9 = −57.8 to −54.8 on S. That is a coefficient of M between −35 and −59 and of S between +1 and −20. A pooled least-squares fit (n1_data.py) gives −45.9 M − 10.7 S + 58.8 x_9. Only C_9 = 58.2 and B_9 ≈ −57 are stable across ages.
- suggested fix: At the newest index, V[9] ≈ 58.2 x_9 − 57 S − a (M − S), with a between 35 and 59 depending on the age of the maximum (pooled: −46 M − 11 S + 58.8 x_9): the current input's own score.

### [minor] section 3.2

- quote: plus three real ones (0.85, 0.68, 0.26). They match the faster modes of $A$ (section 4.3).
- problem: Only part of the list matches. From the weights (n1_weights.py): Σ has −0.932, 0.925∠±82.6°, 0.834∠±124.6°, 0.694∠±41.2°, 0.845, 0.676, 0.257. Besides its three slow modes, A has −0.928, 0.937∠±83.3°, 0.837∠±124.8°, 0.735∠±43.0°, 0.231, 0.001. So the alternating mode and the two harmonics match within 0.012 and 0.7°, the short clock roughly (0.69 against 0.735), and 0.26 is near the transient 0.231. The real eigenvalues 0.85 and 0.68 have no counterpart in A, whose other real eigenvalues are 1.0099, 0.9735 and 0.001.
- suggested fix: The first three match the alternating mode and the two harmonics of A (section 4.3), the fourth is close to the short clock (0.735 e^{±i 43.0°}), and 0.26 is close to the transient (0.231). The real eigenvalues 0.85 and 0.68 have no counterpart in A.

### [minor] section 3.5

- quote: and the old maximum ends as the runner-up (unperturbed: correct in 99.1%)
- problem: The 99.1% and all rates of this item are for the narrower selection of claim I2, which the text does not give: the final maximum arrives at step 3 to 8; the old maximum is then at least two steps old; the old runner-up is above 0.2, at least 0.1 below the old maximum, and ends as the third (10.2% of sequences, V3).
- suggested fix: Take sequences in which the final maximum arrives at a step t_r from 3 to 8, the old maximum is then at least two steps old and ends as the runner-up, and the old runner-up (above 0.2 and at least 0.1 below the old maximum) ends third (10% of sequences; unperturbed: correct in 99.1%).

### [nit] section 3.2

- quote: (random or rotated $A$, random, permuted or rotated $U_o$; 300 draws each) it never exceeds 0.65
- problem: 0.65 is the largest value, among the null draws, of the mean cosine over i = 0..6 (V1: network 0.987, nulls at most 0.65). It is not a bound on the single cosines that the sentence has just quoted (0.968–0.995).
- suggested fix: Their mean is 0.987. In five kinds of random control (random or rotated A, random, permuted or rotated U_o; 300 draws each) that mean never exceeds 0.65.

### [nit] section 1

- quote: (in 95% of cases; less often when it exceeds the old maximum only slightly)
- problem: 95.5% is the rate for new maxima at steps 2–6 only (arm F, R4). Over all new maxima above a positive old maximum it is 93.7% (arm A: 0.937; my n1_data.py: 0.937), or 94.3% for steps 1–8 (R4).
- suggested fix: (in 94% of cases; less often when it exceeds the old maximum only slightly)

### [nit] section 1

- quote: 15–20% of it remains when the runner-up ties with the maximum
- problem: V5 gives the remainder in units of w_4 M: 0.213 at age 1 and 0.147 at age 8 for S/M above 0.95 (section 4.1 quotes 0.21 and 0.15). As a fraction of the radius without a positive runner-up at the same age (0.95–1.0 and 0.74) that is 20–22%.
- suggested fix: about a fifth of it remains (0.15–0.21 w_4 M) when the runner-up ties with the maximum

### [nit] section 1

- quote: with a radius equal to its excess over unit 4's threshold
- problem: Only approximately. 5–35% of the old clock survives a new maximum (section 5.4), and in the worked example of section 5.5 the radius one step after the second maximum is 0.61 against an excess of 0.43.
- suggested fix: with a radius close to its excess over unit 4's threshold (plus what survives of the old clock)

### [nit] section 3.3

- quote: | two or more steps old, below | slope 60–85 near zero
- problem: Two values for the same quantity. The text of the same section gives 60–80 for this slope (V1's value); 60–85 is the lead's earlier e13 figure. In the same way the row above gives −10 to −18 for an input that is one step old, and section 5.1 says '10 to 20 M below the runner-up' for that slot.
- suggested fix: Write 'slope 60–80 near zero' in the table, and '10 to 18 M below the runner-up' in section 5.1.

### [nit] section 3.4

- quote: | 4 | 0.30–0.50 | 61.7 | −47.7 | 13.8 |
- problem: The same cell (D = 4, gap 0.3–0.5) is 13.7 in the list two lines below the table (V2: 13.72 on 4M sequences; the table has the lead's 13.8).
- suggested fix: | 4 | 0.30–0.50 | 61.7 | −47.7 | 13.7 |

## Files
