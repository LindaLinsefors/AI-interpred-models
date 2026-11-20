# Lab notebook: mechanistic report on M_16,10 (second-argmax RNN)

Running record of what has been established, by which script, and what is still open.
Scripts are in `hide/` (lead's own: `e01`–`e09`, `common.py`, `modes.py`) and `hide/r1_*/` (parallel arms).
All numbers are from my own computations on the weights unless marked otherwise.

## Task from the user (2026-11-19)
Explain mechanistically how the network works: how information is stored in the hidden state, how it is
updated; a report with evidence ("how I know") and remaining uncertainties. Investigate the user's
hypothesis: the age of the max is stored on a circle with fixed directions whose centre and radius depend on
the max and second-max values (off-centre because of ReLUs; no bias, so an ersatz bias from stored values).
Constraints: everything outside `mechanistic-report/` is read-only (so I have NOT edited the project
CLAUDE.md). The colleague's papers are "not well written": use as hypotheses only, do not trust or emulate.

## Conventions
- $h_t = \mathrm{ReLU}(w x_t + W h_{t-1})$, logits $= U h_9$. Units: comparators 1, 6, 7, 8; pulses 4, 5, 9; wave 0, 2, 3, 10–15. ORD = 13 non-pulse units.
- $M_t, S_t, T_t$: largest, second, third of $x_0..x_t$; ages $a_1, a_2$ = steps since they arrived (window includes $x_t$).
- Resting map $A = W[\mathrm{ORD},\mathrm{ORD}]$. Mode coordinates $m = L\,h[\mathrm{ORD}]$, gauge: $L\,W[\mathrm{ORD},4] = 1$ for every mode (`modes.py`).

## Established (lead's scripts)
- **e01** eigenvalues of $A$: level 1.0099, runner 0.9735, clock 0.951∠42.75°, h2 0.937∠83.28°, alt −0.928, h3 0.837∠124.78°, short 0.735∠42.99°, trans 0.231, echo 0.001.
  Level eigenvector positive on all 13 ordinary units. Input direction $w[\mathrm{ORD}]$ ≈ echo eigenvector (cos 0.998); $A\,w[\mathrm{ORD}]$ entries ≤ 0.43; $W[\mathrm{PULSE},\mathrm{ORD}]\,w[\mathrm{ORD}] = (0.01, 18.0, 84.3)$.
- **e02/e03/e05** on N(0,1) data: level ≈ 0.93·10.157·M (age ≥ 2; 1.006 at age 1; old max at age 0). Clock $z \approx 10.157\,(M-S)\,\rho(a_1)\,e^{i\varphi(a_1)}$, φ = 0, 37, 80, 123, 166, 209, 252, 294, 338° (a = 1..9), ρ 1.0 → 0.72; S-remainder 0.1–0.35; R² ≥ 0.95 per (a1, a2) cell.
  After a record, every mode coordinate ≈ h4 at age 1 (kick = 1 in the gauge). h2/h3 components of the increment are gone by age 2 (cancelled by pulses 5/9).
- **e04** LEFT vectors: clock 96.6% on comparators, short 91.6%, trans 97.6%, level 4%. Zeroing comparators maps the clock mostly into h2/alt/h3/short/runner (matrix P in `e04_real_basis.npz`).
- **e06** readout per mode $R_k = U[:,\mathrm{ORD}]\,\mathrm{RIGHT}_k$; pulse kernels under free flow: unit 4 ≈ −3.5..−4.1 at own position at all ages; unit 5 ≈ +3.4..+4.2 at the recorded input's position, −1..−2 at earlier positions; unit 9 ≈ −3.1 one position before.
- **e07** level and clock are (nearly) invisible to the readout: level reads −0.31..−0.53 uniformly on positions 0–8 and +1.94 on position 9; clock: comparator half and wave half are 180° apart and cancel to 6%.
- **e08** two-pulse inputs (M = 1.5; S ∈ {0.3, 1.0, 1.4}; background −1 or −0.3): 90/90 correct.
- **e09** trace × mode decomposition of two-pulse logits: no single visible mode marks the runner-up; the short mode flips sign between "latest event = new max" and "latest event = new runner-up".

## Working model (to be verified by round 1 and adversarial checks)
1. Hidden variables (invisible to the readout while comparators are on): level ∝ M (centre / ersatz bias), clock = gap × phase(age of max), stored in the comparators.
2. Visible tape (h2, h3, alt, short, trans, runner): localized ± bumps at the positions of recorded events; free flow shifts them one slot per step.
3. Writes: unit 4 (−bump at new max, ∝ increment), unit 5 (+bump at previous input's position), unit 9, comparator clipping ((x − θ_j)⁺ records), wave clipping.
4. A new max zeroes the comparators: wipes the clock, exposes the wave half → (hypothesis H_dev) positive bump at the old max's position ∝ old gap.
5. Low inputs: null direction of A, only pulses 5 and 9 see them.

## Open
- How exactly the runner-up's position is stored when it arrived after the max.
- Whether H_dev is right; what else carries the old max's position.
- Exact logic by which the runner-up beats the max and the third.

## New results (lead, e10–e14) — the "tape of provisional scores" picture
- **e10 (EXACT + measured)**: define provisional logits $\ell_t = U h_t$ (index $9-a$ = input of age $a$). Under one free resting step the ordinary units obey $U_o A \approx \Sigma U_o$ with $\Sigma = U_o A U_o^{+}$ (relative residual 0.095); $\Sigma$ is a shift: $\Sigma[i, i+1] = 0.97, 0.93, 0.90, 0.89, 1.00, 0.93, 0.93$ for $i = 0..6$ (0.34 for $i=7$, 0 for $i=8$). On real states the next-step provisional logits are predicted with $R^2 = 0.985$–$0.992$ (positions 0–8, mean removed). The eigenmodes are the Fourier modes of this shift register.
- **e11**: the readout barely sees level/runner/clock/echo among positions 0–8 (per unit coordinate: 0.33, 0.11, 0.91/0.75, 0.04 vs h2 2.6, h3 2.6, alt 1.4, short 9.4/5.6, trans 24). "Hidden" must be stated per unit coordinate: the clock coordinate is large (≈ 10 × gap), so its 6% leak is a few logits × gap, mostly at positions 0–2.
- **e12 (measured, n = 400k)**: reading $h_t$ at intermediate $t$ with the final readout gives the runner-up of the prefix: accuracy 0.992, 0.987, 0.971, 0.966, 0.969, 0.970, 0.970, 0.967, 0.961 for $t = 1..9$ on states with $S_t > 0$. So the network maintains a running answer in age coordinates.
- **e13 (measured)**: provisional score relative to the runner-up's, in units of $M$:
  - current input below $S$: $-61.6\,(S - x_t)$, linear over the whole range (age 0);
  - previous input (age 1): $\approx -60\,(S - x)$ for deficits $< 0.15 M$, saturating at $-10..-18\,M$;
  - ages $\ge 2$: slope 60–85 near 0, saturating near $-20 M$;
  - the max: age 0: $\approx -60\,(M - S)$; age 1: $\approx -28\,(M-S)$; ages $\ge 2$: $-5..-14\,M$, nearly independent of the gap unless the gap is $< 0.05 M$.
  - Same slope from three routes: age 0 via comparators ($U[9,\mathrm{COMP}]\cdot w[\mathrm{COMP}] = 58.2$, of which unit 7 gives 49.9) and unit 4 ($U[9,4] w_4 = -50.7$); age 1 via unit 5 ($U[8,5]\times 18.0 = 61.2$).
- **e14 (measured, n = 600k)**: at a new max, afterwards score(old max) − score(old runner-up) $\approx c(\Delta)\,(M_0 - S_0) + (0.4..5.7) M_0$ with $c = 17.6, 43.5, 73.8, 88.5, 81.6, 63.2, 41.8$ for separation $\Delta = |p_{M_0} - p_{S_0}| = 1..7$; $c(\Delta)/(1 - \cos 42.75°\Delta) \approx 65, 47, 46, 45, 45, 51, 82$. So the deposit is a first-harmonic bump $\approx 45\,(M_0 - S_0)\cos(\omega(p - p_{M_0}))$: the hidden clock (radius = gap, phase = age) developed into visible scores. Prediction: weak separation at $\Delta = 1$ and near $\Delta = 8$.
  The lead decays about 10% per step afterwards (40.7 → 23.8 per unit gap over 5 steps).
  New runner-up's lead over the old one: $\approx (27..62)(x_t - S_0) + 2M$, $R^2 \approx 0.6$ (noisier).

## Working model v2
State = (i) a tape of provisional scores by age (10-dim image of $U$; shifts one slot per free step), such that the argmax is always the current runner-up, and scores of near candidates are comparisons by value against the runner-up with slope ≈ 61 logits per unit; (ii) comparison variables nearly invisible to the readout: level ∝ M and clock = gap × phase(age of max), the latter being the "pending deposit" that is developed at the old max's position when it is dethroned; (iii) one-step registers: unit 4 (increment), unit 5 (previous input's excess over ≈ 0.7 S), unit 9 (previous input was a record), echo (current input).

## e15 / e15b: two score vectors (confirms and sharpens Zihao Chen's 2024 picture; ARC note, see sources_arc.md)
- $s_t = U h_t$ (all units), $m_t$ = the same without the comparator columns, $V_t = s_t - m_t$ = comparators' vote.
- (measured, n = 500k) $\arg\max m_t$ over inputs older than the current one = position of the max of the prefix: 0.972, 0.942, 0.948, 0.953, 0.958, 0.964, 0.966, 0.891 for $t = 2..9$ (given $S_t > 0$, max not the current input). $\arg\max s_t$ = runner-up (e12).
- (fit, $t = 9$, all four comparators on, $R^2$ 0.88–0.97) $V[\text{idx}] \approx A\,(M - S) + B\,S + C\,x_9$ with $A$ depending only on the distance $d$ from the max's index: $A(d) = -62, -48/-42, -6/+7, +40/+59, +60/+82, +47/+61, +14, -25/-33$ for $d = 0, \pm1, .., \pm7$, i.e. $A(d) \approx -60 \cos(42.75° d)$. $B$ has the same shape with amplitude ≈ 8 ⇒ $V \approx -60\,(M - 0.87 S)\cos(\omega (p - p_M))$. At index 9: $V[9] \approx -50 M - 7 S + 58.2\,x_9$.
  ⇒ **the comparators hold $s - m$ as a first-harmonic wave centred on the max's position with amplitude ∝ gap: this is the clock.** (Chen: "the trained net has 4 negative input neurons which seem very entangled and this has been a major obstacle to understanding how $(s_k - m_k)$ is represented".)
- (measured) at a new max at step $t$: $s_t$ on inputs $0..t-2$ ≈ shifted $m_{t-1}$: $R^2$ = 0.96, 0.77, 0.77, 0.88, 0.91, 0.89, 0.87 ($t = 3..9$), versus shifted $s_{t-1}$: $R^2 \le 0.29$. The old max tops $s_t$ among older inputs with probability 0.964–0.996. With no event ($x_t < S$): $s_t$ ≈ shifted $s_{t-1}$, $R^2$ 0.67–0.96.

## e16: the runner-up's margin over the max = m-part + V-part (t = 9, all comparators on, n = 800k)
- V-part $= V[p_S] - V[p_M]$ matches $60\,(M - 0.87 S)(1 - \cos(42.75° D))$ ($D = |p_M - p_S|$): e.g. $D=3$: 10.3, 20.3, 32.8, 49.0, 68.7 (units of $M$) for gap bins (0,.05], (.05,.15], (.15,.3], (.3,.5], (.5,.8] vs predicted 14.7, 21.0, 31.5, 45.7, 63.4.
- m-part $= m[p_S] - m[p_M]$ is negative and nearly cancels it ($-4.8, -12.8, -24.1, -39.3, -58.7$ for $D = 3$): in $m$ the max leads by about the same cosine-shaped amount.
- The sum (s-margin) is small and nearly gap-independent: 3–5 $M$ ($D=1$), 4–8 ($D=2$), 5–10 ($D=3$), 4–14 ($D=4,5$); positive in 99.7–100% of cases for gap $> 0.05 M$, 85–96% for gap $\le 0.05 M$.
- So $m = s + 60\,(M - 0.87 S)\cos(\omega(p - p_M))$: the wave units hold the runner-up scores plus a bump at the max; the comparators subtract the bump. Both halves are the two halves of the clock mode.

## Prior art (see notes/sources_arc.md, notes/sources_thirdparty.md) — must be credited in the report
- Zihao Chen (ARC, 2024): logits shift one position per step; intermediate logits give the running answer; "max sub-circuit" = readout without units 1, 6, 7, 8; on a new max "the old max logit gets a boost"; distilled algorithm with two score vectors $m_k$, $s_k$ and $s \leftarrow m$ on a record; open problem: how $(s_k - m_k)$ is represented by the four entangled comparators.
- Goldowsky-Dill (LW, Jan 2026): comparator pre-activations ≈ $a_n(\delta) M + b_n(\delta) S^+$, $R^2$ 0.992; input cancels over {1,6,7,8,2}.
- DraggonZ (Oct 2026): store/detector/keep gate/late gate/hiding neurons/memory; "hiding neurons cancel the vote of the current largest; a bigger number switches them off"; "blindness" of non-gate units to the input direction; readout shift cosines; design argument that a bias-free ReLU net must put the wave on a positive level (close to the user's (a), (c)).
- Snider (Oct 2026): two equal-speed clocks (0.951 and 0.735 at 43°), n5 as write pointer, rotating the max clock changes only 5% of outputs.
- Panahi (Oct 2026): $h \approx m_2 U[p_2] + (m_1 - m_2) V[p_1]$; "damped discrete Fourier memory".
- Skyway1111: "the clock is the comparators"; Keane: phase wheel, reset-and-rebuild; Khachaturov: exact 16-coordinate program, false record at $z = 511/512$ after $[1, 0]$.
- Nobody (per the digest) derives the level as a substitute for a bias from the trained weights, or gives the comparators' vote as $-60\,(M - 0.87S)\cos(\omega(p - p_M))$.

## e17 (INTERVENTION, n ≈ 100–190k per row): wiping the comparators once makes the net answer the max
Set $h_{t_0}[1,6,7,8] = 0$ at one step $t_0 = p_M + k$ (no record there), continue with the real inputs (final gap > 0.2, S > 0.2).
- For $t_0 \le 8$ (k = 2): P(answer = max) = 0.997–1.000, P(answer = runner-up) = 0.000. For $t_0 = 9$ (wipe at the last step) the answer is neither (the comparators also hold the current input's own score).
- Pooled over $t_0$: k = 1..6 → 0.868, 0.873, 0.753, 0.816, 0.977, 0.999 answer the max.
- Control (wave units 10, 11, 12, 15 zeroed instead): P(max) ≤ 0.008; accuracy collapses to 0.22–0.54.
⇒ the wave units alone rank the max first; the comparators hold exactly what hides it; once wiped it is not rebuilt without a new record.

## e18 (INTERVENTION, n = 152,690): the clock is the pending deposit for the old max
Sequences where a record at step $t_r$ turns the old max (position $p_0$, age ≥ 2) into the final runner-up and the old runner-up $q_0$ into the final third. One step before the record only the clock coordinate $z$ of $h_{t_r-1}$ is changed (state moved along RIGHT["clock"]); the run continues with the real inputs.
- unmodified: answer $= p_0$ 0.990.
- radius $z \to f z$: P(answer $= p_0$) = 0.019, 0.098, 0.342, 0.889, 0.990, 0.893, 0.746 for $f$ = 0, 0.25, 0.5, 0.75, 1, 1.5, 2. With $f = 0$ the answers are the new max (0.775) or $q_0$ (0.164).
- phase $z \to z e^{i k \omega}$ ($\omega$ = 42.75°): $k=-1$: answers $p_0+1$ 0.474, $p_0+2$ 0.154, $p_0$ 0.093; $k=-2$: $p_0+2$ 0.321, $p_0+1$ 0.153, $p_0$ 0.002; $k=+1$: $p_0$ 0.458, $p_0-1$ 0.381; $k=+2$: $p_0-1$ 0.298, $p_0-2$ 0.249, $p_0$ 0.071. Mean shift of answers within ±3 of $p_0$: +1.36, +2.15, +2.73 for $k=-1,-2,-3$ and −0.54, −1.26, −1.61 for $k=+1,+2,+3$.
⇒ the direction of the clock has a fixed meaning (age of the max ⇒ position that gets promoted), and its radius sets how much.

## e19 (controlled sweep): x = (0.5, b, 1.5, b, b, b, v, b, b, b), b = −1, v swept
State before v: θ_1 = 1.274, θ_6 = 1.673, θ_7 = 1.381, θ_8 = 0.555, θ_4 = 1.366 (S = 0.5, M = 1.5, max age 3). Clock |z|/10.157 = 1.112 (M − 0.87 S = 1.065).
- v ≤ 0.3: nothing clipped, unit 5 silent, clock untouched, answer 0 (correct), margin ≈ 7.
- v = 0.45 (= 0.9 S): nothing clipped but unit 5 fires 2.77 at the next step; margin of the true answer drops to 1.7.
- v = 0.55 (just above S, below θ_8): nothing clipped, unit 5 = 4.57, answer switches to 6 (correct).
- 0.7 ≤ v ≤ 1.15: comparator 8 clipped; v = 1.3: comparators 1 and 8; unit 5 = 5.7..10.3 (slope 7.6 per unit v); s[6] − s[0] at the end = 17.0, 29.5, 42.1, 54.6, 68.6 (slope ≈ 85 per unit v).
  Clock radius at t = 9 falls linearly with v: 0.851, 0.770, 0.688, 0.607, 0.517 (slope −0.56 per unit v; radius ∝ M − 0.87 S would need −0.74); phase moves by ≤ 11°.
- v = 1.40..1.49 (below M = 1.5 but above θ_4 = 1.366): unit 4 misfires (0.34..1.25), unit 9 fires, clock collapses (0.26 → 0.12), margin shrinks 24.7 → 5.0 (still correct).
- v = 1.55: new max; comparator 6 stays ON (θ_6 = 1.673 > v): partial wipe; answer 2 (correct, margin 9.0). v = 1.8: all four off; margin 30.

## e20: raw unit activations by age of the max, three gap bins (quiet steps, n = 1.5M sequences)
- **Comparators** (medians of $h_j/M$ over ages 1..9): swing scales with the gap: unit 1 ±2.3 (gap 0.15–0.3) vs ±6.2 (gap 0.85–1.0); unit 8 ±2.2 vs ±6.6; unit 6 ±1.8 vs ±5.7; unit 7 ±1.2 vs ±4.2. Ratios 0.29–0.37 ≈ $(M - 0.87S)$ ratio 0.35. So in the comparators both centre and swing follow $S + (M-S) f_j(a)$.
- **Wave units do NOT shrink to a point when the gap closes.** Each is cut to exactly 0 at a fixed age of the max for every gap size: unit 0 at age 2 (100%), unit 12 ≈ 0 at age 4 (21–42% exactly 0, median 0.1–0.3), unit 11 at age 5 (86–99%), unit 10 at age 6 (97–98%), unit 15 at ages 2–4 (23–83%). Their maxima scale with the gap (unit 12: 15.3 → 31.1; unit 10: 13.5 → 23.2 in units of $M$).
  So in the wave units the max's age is a half-wave-rectified travelling wave ("a zero that travels 0 → 12 → 11 → 10"), cut by the ReLU at every cycle; the cut is part of the code (it does not perturb the clock, whose left vector is 97% on the comparators).
- Readout matches the holes: each of these units has its most negative readout weight at the position where it is zero when the max is there: $U[7,0] = -2.90$ (age 2), $U[5,12] = -2.79$ (age 4), $U[4,11] = -4.56$ (age 5), $U[3,10] = -3.87$ (age 6). ⇒ in $m$ the max's position is "the one that is not voted against".
- Consequence for the user's hypothesis: (a) "off-centre so the ReLUs do not cut it" is right for the comparators (the core, where the clock is stored: thresholds stay at or above $S$ between records; a new max cuts all four = wipe). In the wave units the wave is cut once per cycle by design.

## e21 (EXACT + measured): wiring of the core (units 2, 4, comparators 1, 6, 7, 8)
With $g_j = |w_j|$ and comparator activations $h_k = g_k(\theta_k - x_t)$ (when on), the thresholds obey exactly
$$\theta(t+1) = B_c\,(\theta(t) - x_t \mathbf 1) + b_2\,h_2(t) + b_4\,h_4(t) + (\text{other units}),\quad B_c[j,k] = W[j,k]\,g_k/g_j .$$
- $B_c$ rows (order 1, 6, 7, 8): (0.409, 1.036, −1.306, −0.140); (−0.130, 0.361, −0.151, −0.092); (−0.081, −0.152, 0.078, 0.156); (0.544, 0.000, −1.170, 0.615).
- **$B_c \mathbf 1 = (-0.001, -0.012, 0.001, -0.011)$**: a common shift of the four thresholds — and therefore the input $x_t$ — is annihilated. Eigenvalues of $B_c$: $0.9375\angle\pm42.56°$ and $0.043$ (pair). So the comparator block is a 2-D rotator acting on threshold *differences*; swing per comparator (rel. to $\theta_7$): $\theta_8$ 1.00@72°, $\theta_1$ 0.80@143°, $\theta_6$ 0.21@−144°, $\theta_7$ 0.20@0°.
- **Level supply (the ersatz bias):** $b_2 = -W[j,2]/w_j = (0.048, 0.049, 0.051, 0.051)$; with $h_2 \approx 17.3 M$ this is $0.83$–$0.89\,M$ to every threshold (measured mean parts: unit 2 0.827–0.887 $M$; comparators' own part mean −0.26..+0.03 with sd 0.05–0.42 (the swing); other wave units +0.04..+0.11 $M$). Because $B_c$ is blind to common shifts, the common level cannot live in the comparators; it is re-supplied every step from unit 2.
- Unit 2: $h_2' = 0.969 h_2 + 1.731 h_4 - 0.102 h_1 - 0.103 h_6 + 0.023 h_7 + 0.155 h_8 - 0.229 h_9 + \dots$, input weight 0.015. Measured $h_2 / (17.3 M_{t-1})$ by age of max 1..8: 1.02, 1.02, 1.07, 1.05, 0.98, 0.90, 0.85, 0.86 (it sags at ages 6–8).
- **Record supply:** $b_4 \cdot 10.157 = (0.838, -0.044, 1.016, -0.034)$ per unit increment $d$ for $\theta_1, \theta_6, \theta_7, \theta_8$: a record lifts $\theta_7$ and $\theta_1$ by ≈ $d$ and leaves $\theta_6, \theta_8$ at the old max. So one step after a record the thresholds are spread over $[S, M]$ with spread $= d = M_{new} - M_{old}$ = the new gap: **this is why the clock's radius is the gap.**
- Unit 4's threshold: $\theta_4' = 0.0485 h_2 + 0.0970 h_4 + \dots$ (measured parts: unit 2 0.841 $M$, wave units 0.085 $M$).

## e22 (prediction test, n = 4M sequences): errors against the separation of runner-up and third
Order "max arrives after both the runner-up (= old max) and the third (= old runner-up)", $T > 0.2$. P(answer = third):

| $D = |p_S - p_T|$ | $1-\cos(\omega D)$ | gap $(S-T)/S \in (0, .03]$ | $(.03, .08]$ | $(.08, .2]$ | $(.2, .5]$ |
|---|---|---|---|---|---|
| 1 | 0.27 | 0.183 | 0.066 | 0.029 | 0.005 |
| 2 | 0.92 | 0.147 | 0.038 | 0.010 | 0.003 |
| 3 | 1.62 | 0.189 | 0.029 | 0.003 | 0.000 |
| 4 | 1.99 | 0.245 | 0.020 | 0.001 | 0.000 |
| 5 | 1.83 | 0.276 | 0.028 | 0.002 | 0.000 |
| 6 | 1.23 | 0.338 | 0.137 | 0.023 | 0.002 |
| 7 | 0.51 | 0.371 | 0.171 | 0.125 | 0.066 |
| 8 | 0.05 | 0.447 | 0.346 | 0.123 | 0.004 |

As predicted by a first-harmonic deposit (period 8.42 positions), errors are lowest at $D = 3..5$, higher at $D = 1$ and much higher at $D = 7, 8$ (12–35% at gaps of 3–20% of $S$). $D = 6$ is worse than $1-\cos$ alone predicts: large $D$ also means old records (age confound, not separated).

## e23 (simulation): the six-unit core is nearly autonomous
Simulating only units {2, 4, 1, 6, 7, 8} on the same inputs (n = 300k):
- A (other ten units removed): $R^2$ of activations 0.965 (2), 0.981 (4), 0.932 (1), 0.964 (6), 0.979 (7), 0.877 (8); unit 4 fires on 4.3% of non-records (real 1.2%).
- B (the other units' feedback replaced by a fixed multiple of unit 2, taken from the level eigenvector: extra weights from unit 2 of +0.031, −0.128, +0.085, +0.100, +0.109, +0.191 into units 2, 4, 1, 6, 7, 8): $R^2$ = 0.991, 0.995, 0.989, 0.995, 0.993, 0.991; gate agreement 0.962–0.993; unit 4 fires on 87.6% of new maxima (real 89.9%; the rest are negative new maxima) and 0.3% of others (real 1.2%).
⇒ max, increment and clock form a closed subsystem; the satellites feed back essentially only a level term.

## e24: the clock defined on the thresholds alone
$z_c = \ell\cdot\theta$ with $\ell$ the left eigenvector of $B_c$ for $0.9375\angle42.56°$, normalised to $\ell\cdot(b_4 w_4) = 1$: $|\ell\cdot\mathbf 1| = 0.013$ (level supply does not excite it). On data (n = 300k), thresholds read one step after max-age $a$: phase 33.5°, 72.0°, 120.5°, 167.3°, 218.6°, 262.8°, 308.8°, 346.2° ($a = 1..8$); radial part $b_M M + b_S S^+$ with $b_M$ = 0.93, 0.83, 0.75, 0.68, 0.62, 0.62, 0.66, 0.73 and $b_S/b_M$ = −0.91, −0.94, −0.96, −0.91, −0.84, −0.80, −0.77, −0.78; $R^2$ 0.96–0.997. Agreement with the 13-unit clock coordinate: $R^2$ = 0.93.

## Notes from the digests of the colleague's papers (independent agents re-verified their numbers; see notes/digest_*.md)
- Records are written as dips: $W[:,5] \le 0$ on the tape units (−0.91 on unit 0, −0.66 on 15, −0.52 on 3, −0.42 on 14, −0.14 on 12) and the readout weight at the carrier is negative; a dip needs a positive baseline. Unit 0 is clipped two steps after a new max mainly because the max's own record ($\approx 12 M$ through $W[0,5] = -0.91$) exceeds unit 0's baseline (with unit 5 forced off there, unit 0 is never off; with unit 9 forced off it is still off 71%).
- A record's own effect at ages 2–6 is carried by the 83°, 125° and short 43° modes, not by the main clock (< 1 per unit).
- $W[5,4] w_4 = 12.06$, $W[9,4] w_4 = 1.48$, $W[5,7] w_7 = 7.81$, $W[5,8] w_8 = 10.46$; four comparators together 18.1.
- Unit 5 is on one step after 99.9% of positive new maxima and 99.9% of positive new runners-up (5.0% / 7.2% of negative ones).

## e25 (EXACT): numbers for the report
- $U[9,\mathrm{COMP}]\cdot w[\mathrm{COMP}] = (5.01, 7.38, 49.90, -4.04)$, sum 58.25; $U[9,4] w_4 = -50.70$; $U[8,5] = 3.398$; $U[8,5]\times 18.009 = 61.19$.
- One unit of input (four comparators on) leaves in every unit one step later: $\le 0.43$ in magnitude, except unit 5 (+18.009 = 7.81 via unit 7 + 10.46 via unit 8) and unit 9 (+84.335).
- "Blindness": $\cos$ between row $W[u,\mathrm{COMP}]$ and $w[\mathrm{COMP}]$ is within $\pm 0.04$ of 0 for every unit except 5 (+0.74) and 9 (+0.69).
- $W[5,4] w_4 = 12.06$, $W[9,4] w_4 = 1.48$, $W[2,4] w_4 = 17.58$; $W[2,2] = 0.9688$, $W[4,2] = -0.4923$, $W[4,4] = -0.9856$.
- The comparators' readout columns are sinusoids over positions 0–7 at the clock frequency: $U[p,j] \approx a_j\cos(42.75° p - \varphi_j) + c_j$ with $a$ = 5.81, 5.09, 6.21, 6.08, $\varphi$ = −29°, 26°, 131°, −130°, $R^2$ = 0.974, 0.943, 0.991, 0.988 (units 1, 6, 7, 8).
- From e07: the comparator part of the clock's readout, $2|R|$@phase = 6.59@−161°, 8.40@−117°, 8.42@−85°, 7.26@−33°, 7.10@20°, 7.39@65°, 6.44@99°, 5.56@145° (positions 0..7): phase advances ≈ 44° per position. With $\varphi(a) = 37° + 42.75°(a-2)$ the vote at the max's own position has phase 175°–186°, i.e. it is most negative there, with size $\approx 7.5\,|z| \approx 76\,\rho\,(M - 0.87S)$.
- Wave units' most negative readout weight: unit 0 at position 7, 10 at 3, 11 at 4, 12 at 5, 13 at 2, 14 at 0, 2 at 1.

## Verification round (2026-11-19, evening): what changed

Eight verifiers recomputed the 33 claims of `claims_for_verification.md` with their own code (`hide/v_*/`, verdicts in `notes/v_*.md`). All numbers reproduced: 31 as stated, 2 too wide in scope (S5: the bound 0.997 holds for gap/M up to 0.8 only; D2: my numbers were for $g \ge 0$, not $g \ge 1$). The entries above are kept as written at the time. Where a reading below contradicts one above, the one below holds.

**Readings above that are wrong or too strong.**

- **e10 (shift).** The $R^2$ of 0.985–0.992 is for a hypothetical resting step. On real no-event steps it is 0.80–0.91, and the shift holds for entries two or more steps old only (0.50, 0.56, 0.74, 0.89, 0.93, 0.94, 0.96 for $t = 3..9$). The residual 0.095 is almost all row 9 (rows 0–8: 0.027); row 9 of $\Sigma$ rides on the smallest singular direction of $U_o$ and should not be read.
- **e12 (running answer).** Exact identity: $h_t(x_0..x_t) = h_9(0,\dots,0,x_0..x_t)$. So this is the accuracy on zero-prefixed inputs, and 354 of 379 retrained networks (accuracy ≥ 0.90) show it too.
- **e13 (score laws).** Slope 61.6 is a property of the weights for the newest input: $(U[9] - U[i])\cdot w[\mathrm{ORD}] = 61.0$–$62.1$. Older scores are clipped comparisons whose floor depends on arrival order ($-15$, $-25$, $-33\,M$).
- **e15, e15b, e16 (two score vectors).** $m$ carries the mirror image of $V$'s wave (amplitude 62.6, crest on the max; $V$: 66.2, trough on the max); net coefficient of the gap at most 8. The cosine shape of $V$ is forced by the readout columns. The $R^2$ of 0.88–0.97 in e15 was pooled over all ten indices; on indices 0–7 it is 0.82–0.96. With the max removed, the runner-up is first in $s$ 97.7%, $m$ 42%, $V$ 13%. Replacing $V$ by the cosine formula leaves the runner-up first in 61% (true $V$: 97.4%).
- **e17 (wipe).** The pooled rates included the wipe at the last step. For $t_0 \le 7$: 0.999, 1.000, 0.975, 0.941, 0.992, 1.000 for $k = 1..6$; for $t_0 = 8$: 0.80, 1.00, 0.42, 0.71, 0.93, 1.00; misses answer the wipe-step input. The comparators are rank 1 of 1820 four-unit sets. What hides the max is their clock-mode share. With 2% of their activity left the max is answered in only 10–32% ($k \le 4$): unit 9 must fire at $t_0+1$.
- **e18 (clock before a record).** The rotation rates were unconditional. Conditional on the target existing and not being the old runner-up: 0.36, 0.23, 0.40, 0.37 for $k = -1, -2, +1, +2$. For $f > 1$ nearly all edited states are unreachable. At $f = 0$ the network answers the new max (78%), not the old runner-up (16%). The causal carrier is the wave units' share of the clock pattern (wave-only removal: 0.022; comparators moved so that $z = 0$: 0.987).
- **e14 (lead at a record).** $c_1$ depends on the two ages $(a_M, a_S)$, not on the separation alone. The cosine law holds for $a_S \ge 2$ ($c_1/(1-\cos)$ = 45.0–54.9, $K$ = 65.7 or 42.5 by order); it fails for $a_S = 1$. The "decay of about 10% per step" is a cohort artefact: in fixed cohorts the lead grows.
- **e22 (errors by separation).** The reading "errors at separations 7–8 follow from the single harmonic" is wrong. $D = 8$: 0.708 for $(p_S, p_T, p_M) = (8, 0, 9)$ against 0.005 for $(0, 8, 9)$. $D = 7$: carried by $(0, 7, 9)$ and $(7, 0, 9)$. Records at the last step, or with the runner-up right before the max, carry 65–92% of these errors. Separation explains 8–19% of the between-configuration variance, the pair of ages 57–94%.
- **e23 (core alone).** The conclusion "closed subsystem" is false. Cutting the 60 weights from the ten satellites into the core: accuracy 50.6% (49.0% with compensation). A model with no dynamics reaches $R^2$ 0.82–0.997 for the activations, so that measure was uninformative. The satellites' term in the thresholds (0.003–0.175 $M$ by age) is mostly differences, not level; removing it gives 54.5%.
- **e21 (unit 2).** Unit 2's supply is indispensable (19% without) but not uniform: making $b_2$ uniform gives 65.7%. $h_2 \approx 17.3\max(M_{t-1}, 0)$; for old maxima it sags towards $S^+$ ($\kappa$ = 0.21–0.40).
- **e02, e05 (level, clock).** Level $\approx 0.936\,w_4\max(M_{t-1}, 0)$: it lags by one step. Clock radius: best single law $M - cS^+$, $c$ = 0.87 falling to 0.75; 0.15–0.2 $M$ remains at a tie; $\alpha(a)$ = 1.01, 1.03, 0.99, 0.95, 0.92, 0.88, 0.85, 0.82, 0.81.
- **e07 (cancellation).** 0.0575 is invariant under unit rescaling and rank 1 of 715 splits, but deleting the clock piece from $h_9$ still changes 7.4% of answers.
- **e20 (zeros).** Unit 12 is not cut (zero in 14% of final states at age 4). Only unit 0 is clipped deeply; units 11, 10 graze zero. "Not voted against" holds at max ages 1, 2, 5, 6 only.

**Script issues found by the code review (V8)** and fixed afterwards: e17 header and pooling; e15 pooled $R^2$ and a mislabelled print; e14 missing $g \ge 1$ in the persistence block and $\Delta = 8$ never printed; e18 non-exclusive outcome categories; labels in e15b, e20, e23, e24, e25, e10; `fig_mechanism.py` panel labels.

**Report review, round 1 (on the first draft).** R4 replicated ten arm-derived claims from scratch (sections 4.3, 5.2, 5.3, 5.4, 6, 7, 8): none failed; four numbers were corrected (the cut of unit 0 fails when the next input is another new max; the unit-9 rates depend on the subset; 18.0 not 18.3 in section 6; the pulse terms of $\Theta_5$). The other three reviewers of that round (R1, R2, R3) were lost to a 45-minute session stall; their partial scripts are in `hide/rev_R1_numbers`, `rev_R2_sceptic`, `rev_R3_reader`.

**Report review, round 2 (on the revised text; `notes/rev_N1_*.md`, `rev_N2_*.md`, `rev_N3_*.md`, `rev_R_reader.md`, `rev_S_sceptic.md`).** No wrong mechanism. What changed:

- *Sceptic (major):* the exact zeros of units 0, 11, 10 are not needed at a new maximum; their low values are. With the ReLU of that unit removed at that step the old maximum is still answered in 0.997 / 0.982 / 0.982. Only the clipping of unit 0 has a measured function, at later steps (unit 0 linear throughout: 0.82).
- *Phase:* the list 0°, 36°, 79°, … is the phase of the gap term. $\beta$ (the $S^+$ term) is complex with its own phase, so the phase of $z$ lags by 3–7° / 8–15° / 18–33° for relative gaps of 0.3–0.6 / 0.1–0.3 / below 0.1. A new runner-up holds the phase back.
- *Comparators at a new maximum:* all four off in 94% overall; 0.99, 0.99, 0.97, 0.89, 0.83, 0.77, 0.77, 0.75 by age 1..8 of the old maximum. When one stays on, the old maximum is still answered in 94.5% (97.7% when all are off).
- *Launch radius:* 1.04 times unit 4's excess on average, but 0.86–2.3 times case by case; 36–69% of the old $|z|$ is still in the wave units at the step of the new maximum (5–35% two steps later).
- *Level and runner removal* (3.5) are not threshold effects in the same way: the runner removal goes through unit 5 (restoring unit 5: 72%); the level removal is the loss of the offset (restoring the wave units: 77%).
- *Numbers attached to the wrong selection or range*, about thirty; the important ones: 5.3 "one, two, three comparators off" was reversed (three 11%, two 47%, one 29%); 4.3 "no random vector below 0.29" was a quantile (minimum 0.22); 6 "−32" is unit 5 alone (others −15, −3; total +17); 6 "3–14 M" replaced by the distance dependence; 7 "worst arrangements" were the worst among separations 7–8 only ((7, 8, 9) is worse at small gaps); 5.4 first clock step is $1.00\,e^{i36.0°}$ with the clipping of unit 0 included (pulses alone $0.958\,e^{i37.0°}$); runner law $1.9\,M + 5.5\,S^+$.
- *Reader:* mode coordinates, $z$ and the clock pattern are now defined in section 2; one exact update rule heads section 5; the two ways of splitting the logits (by unit read, by source) are stated in 5.4; notation unified ($h_t[j]$, index against position, $t_r$, "written score" instead of "record").

**Final consistency check** (one more reader, after the rewrite): every major and moderate review issue handled, no number copied wrongly; about thirty partial propagations and notation leftovers, all applied. `hide/lint_report.py` checks the Markdown of `report.md` (dollar signs, display equations, tables).

## Update rule for the mode coordinates (exact; added in the review, coefficients recomputed by the lead)

$\zeta_m(t) = \lambda_m\zeta_m(t-1) + (\ell_m^\top w_o)x_t + \sum_{u\in\{4,5,9\}}(\ell_m^\top W[o,u])h_{t-1}[u] + \sum_{k\in\{1,6,7,8\}}\ell_m[k]g_k[x_t-\theta_k]_+ + \sum_{u\in\text{wave}}\ell_m[u][-a_t[u]]_+$. In units of $w_4$: input −0.004 (level), +0.101 (runner), 0.011∠−52° (clock); unit 4: 1, 1, 1; unit 5: −0.072, +1.72, 0.074∠−138°; unit 9: −0.126, −0.44, 0.91∠−4°; comparators 1, 6, 7, 8 off, per unit of $x_t-\theta_k$: level +0.03, +0.07, −0.05, −0.04; runner +1.41, −0.20, −5.38, +4.10 (comparator 7 corrected on 2026-11-20 from −0.06 and −5.39: the exact values are −0.0548 and −5.3848); clock 0.80∠−124°, 1.31∠159°, 1.38∠29°, 0.67∠−45°. Sums of the four comparator terms: 0.00, −0.08, 0.02∠−35° ($B_c\mathbf 1 \approx 0$ in mode coordinates). $|r_{\text{runner}}|/|r_{\text{level}}| = 0.076$.

## e29 (intervention, not independently verified): what the wave units feed back into the comparator thresholds

`e29_satellite_feedback.py`, 400k sequences. The eight wave units other than unit 2 add $\varepsilon_j = \sum_u W[j,u]h_u$ to each comparator's pre-activation. Split $h[o]$ exactly into the nine mode patterns and remove one mode's part of $\varepsilon$ at every step. Accuracy: intact 0.953; all removed 0.550; level 0.598, runner 0.712, short 0.709, transient 0.736, h3 0.813, h2 0.877, clock 0.894, echo 0.930, alt 0.947. Keeping one part only: 0.42–0.60; level + runner + clock kept: 0.803. Size of each part in threshold units (mean level / rms difference between the four thresholds, in $M$): level +0.125 / 0.037; transient −0.031 / 0.034; runner −0.031 / 0.029 (level part ≈ −0.010 − 0.042 $S^+/M$, $R^2$ 0.82); h3 +0.019 / 0.021; short −0.005 / 0.021; h2 +0.007 / 0.013; clock +0.010 / 0.008; echo +0.003 / 0.008; alt +0.004 / 0.004. The cost of removing a part follows the size of the differences it makes between the thresholds.
⇒ the feedback is the wave units' share of every mode, the level first. The level is a loop through all 13 units (unit 2 alone: 0.969 per step; eigenvalue 1.0099).

## Figures round (2026-11-20): 20 extra figures, their checks, and e30

On request ("all the plots you think I might find helpful"): 20 figures in `plots/`, listed with captions in `plots/index.md`, drawn by seven scripts `hide/plot_<group>.py` (update_rule, worked_example, two_waves, clock_laws, interventions, errors_units, modes; each takes `--redraw` and caches in `plot_<group>.npz`). One agent drew each group and a second checked it with separate code; full results in `notes/plots_check_results.md` (dumped by `hide/dump_plot_results.py`). No plotted value was wrong; 15 of 20 figures had a title, label or layout fixed by the checker.

The checks found places where the report was off. I recomputed each with my own code (`hide/e30_after_figures.py`, 1M sequences, seed 3030, plus six more samples for three last digits; output in `outputs/e30_after_figures.txt`) and corrected `report.md`. Where an entry above contradicts this section, this section holds.

**Three readings.**

- *Margin against distance (3.4, 6).* "The runner-up's margin over the maximum grows with the distance" held only for pooled pairs. Pairs that include position 8 (an input one step old) are 12–33% of the pairs at $D \le 6$, 50% at $D = 7$ and 100% at $D = 8$. Without them the margin, in units of $M$, is 4.8, 7.3, 10.1, 13.4, 15.6, 14.0, 13.4 for $D = 1..7$ at relative gaps of 0.3–0.5 (all pairs: 4.8, 7.8, 10.2, 13.6, 16.9, 16.4, 18.1, 21.8 for $D = 1..8$). Over the four gap bins it peaks at $D = 4$–5 (12–18) and is 8–13 at $D = 7$. The $V$ part at $D = 7$ is 0.22–0.25 of its value at $D = 4$; $1 - \cos(42.75° D)$ gives 0.26.
- *Phase lag (1, 4.1, 8).* Signed median of $\varphi(a) - \arg z$ ($M > 0.3$), relative gap 0.1–0.3, ages 1..9: 0.6, 12.6, 11.4, 12.0, 7.1, 6.0, 7.2, 8.4, 11.8°; gap 0.3–0.6: 0.8, 5.7, 4.7, 5.0, 3.3, 2.5, 2.7, 4.0, 5.6°. Median absolute deviation at age 1: 9.4° and 3.2°. So the lag starts at age 2; one step after a new maximum the deviation has the same size and no sign.
- *"Other troughs that matter" (4.2).* Value at the step of a new maximum against one step before, by age of the old maximum at that step (medians, units of the old maximum): unit 12 at age 3: 9.6 → 3.2; unit 14 at age 8: 15.4 → 7.2; unit 15 at age 7: 8.5 → 13.2 with $U[2,15] = +2.62$; unit 13 at age 4: 6.1 → 8.2 with $U[5,13] = +2.09$. The last two are rising, positive votes, so the forcing test (the unit keeps its earlier value) shows there that the high value is needed. For comparison, units 0, 11, 10 at ages 2, 5, 6: 6.2 → 0, 3.3 → 0, 4.1 → 0 (exactly zero in 100%, 65%, 87% with this selection: final maximum, old maximum above 0.3).

**Statements that were not exact.**

- Resting gate pattern (4.3): 10.3% of states, 29.1% at the last step. By age 0..9 of a positive maximum: 0, 0, 0.005, 28.8, 26.3, 4.1, 2.6, 54.0, 64.0, 61.0 % (with a runner-up above 0.05: 0, 0, 0.01, 22.4, 27.3, 2.4, 2.2, 53.1, 63.5, 60.7 %). The report had "never at ages 1 or 2" and "2% at ages 5 and 6".
- Longest run of consecutive resting steps in 1M sequences: 0: 333,645; 1: 474,787; 2: 164,714; 3: 26,191; 4: 621; 5: 31; 6: 11. In 6M more: 296 with five or more (longest 7). The report had "no sequence has more than four".
- Wipe at $t_0 = 8$ (3.5): P(maximum) = 0.793, 0.997, 0.418, 0.707, 0.934, 1.000 for $k = 1..6$; of the other answers, position 8 in 1.000, 0.921, 0.997, 0.988, 0.930 (pooled 0.991). The report had "every miss".

**Numbers.**

- Update rule, comparator 7 off: level $-0.0548$, runner $-5.3848$ (the report had rounded twice: $-0.06$, $-5.39$).
- $\alpha(9) = 0.8048 \pm 0.0004$ (7M sequences): 0.80, not 0.81. All nine: 1.011, 1.032, 0.991, 0.951, 0.917, 0.882, 0.852, 0.823, 0.803.
- Radius at a tie ($S/M > 0.95$, $M > 0.3$): 0.213 $w_4 M$ at age 1, 0.161 at age 4, 0.144 at age 8 (6M sequences; a checker got 0.1451 ± 0.0004 on 8M): 0.14, not 0.15.
- Level one step after a new maximum, level$(t)/(w_4 M_{t-1})$: median 1.008 over all such states, 1.015 after the first input, 0.984 with a runner-up above 0.05 stored at $t - 1$ (0.992 with V5's selection, $S_t > 0.05$). From age 2 on: 0.93, 0.92, 0.93 (ages 2, 3–5, 6–9). The report had 0.99.
- Arrangements (7), exact conditional samples (400k sequences rearranged per arrangement, third above 0.2), P(answer = third) in the four bins: (0,7,9) 0.635, 0.617, 0.547, 0.371; (8,0,9) 0.863, 0.699, 0.230, 0.005; (7,0,9) 0.263, 0.272, 0.215, 0.012; (7,8,9) 0.960, 0.716, 0.141, 0.001; (2,1,9) 0.583, 0.393, 0.269, 0.056; (6,5,9) 0.169, 0.270, 0.246, 0.050; (7,6,9) 0.181, 0.236, 0.215, 0.046. The report had 0.23–0.25 for the last three in the third bin.
- Rates of naming the third by where the maximum is (third above 0.2): after both candidates 0.205, 0.050, 0.018, 0.004; at position 9: 0.278, 0.100, 0.040, 0.010; before both 0.313, 0.086, 0.022, 0.004; at position 0: 0.341, 0.136, 0.038, 0.007. The report's "maximum last" meant "after both candidates".

**Conditions now stated in the report.** Readout-column sinusoids have the frequency fixed at 42.75°; left free the best fits are 48.63°, 49.07°, 44.36°, 44.88° per index ($R^2$ 0.980, 0.996, 0.992, 0.994). In e18 the old maximum is at least two steps old one step before the new maximum ($t_r - 1 - p_0 \ge 2$). e22 requires the third above 0.2. $\Sigma$ also lacks the echo mode: its eigenvalue nearest to 0.001 is 0.257.

**Not changed.** The figure `accuracy_by_gaps.png` prints 99.98% for both differences above $0.2\,M$ with a positive third; its seeds are a low draw (460 errors in 1.88M), and 48 other seeds give 99.972%, so the report keeps 99.97%. Found by checkers and not in the report: unit 2's sag is deepest at ages 7–8 (median ratio 0.85–0.86) and smaller at age 9 (0.92); the level coordinate does not sag; in the impulse response wave unit 14 is clipped as well as unit 0 at step 2 (0.010 of the clock coordinate).
