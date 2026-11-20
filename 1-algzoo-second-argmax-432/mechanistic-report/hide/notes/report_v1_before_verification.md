# How the 432-parameter second-argmax RNN works

Mechanistic report on ARC's AlgZoo model $M_{16,10}$ (`example_2nd_argmax()`), 19 November 2026.

Every number below was computed from the weights with the scripts in `hide/`. Other people's write-ups were used as hypotheses only; section 11 says who found what first. Section 9 lists the evidence for each claim and how it was checked. Section 10 lists what remains uncertain.

## 1. Summary

The network runs an online algorithm. It never looks back at earlier inputs, and it does not know the step number. After each input its 16 units hold three things.

**(1) A tape of provisional scores.** Apply the final readout to the state at any step: $s_t = U h_t$. Entry $s_t[9-a]$ scores the input that arrived $a$ steps ago. The largest entry marks the second-largest input so far (96–99% of states with a positive runner-up, at every step). A step in which nothing special happens moves every score one place: $s_{t+1}[i] \approx (0.89\text{–}1.00)\,s_t[i+1]$. Scores compare values: an input $x_t$ below the runner-up $S$ arrives with the score $s[\text{runner-up}] - 61.6\,(S - x_t)$.

**(2) Three numbers that the readout barely sees, used for comparing.**

- The largest value so far, $M$. Unit 2 holds it ($h_2 \approx 17.3\,M$).
- The second-largest so far, $S$, held in a second slow mode. It sets the threshold of unit 5, $\Theta_5 \approx 0.75\,S$.
- A clock: a two-dimensional quantity that rotates by 42.75° per step. Its angle counts the steps since the maximum arrived. Its radius is close to the gap $M - S$. It lives in the four comparators (units 1, 6, 7, 8) as the differences between their thresholds; unit 2 supplies the thresholds' common level. This is the circle of your hypothesis.

**(3) Three one-step registers.** Unit 4 fires by how far the current input exceeds the maximum. Unit 5 fires, one step late, by how far the previous input exceeded $\Theta_5$. Unit 9 fires when the previous input was a new maximum.

One identity ties (1) and (2) together. Split the scores as $s = m + V$, where $V$ is the part of the readout that passes through the four comparators and $m$ is the rest. Then:

- $m$ ranks the **maximum** first (89–97% of states, by step);
- $V[p] \approx -60\,(M - 0.87\,S)\cos\!\big(42.75°\,(p - p_M)\big)$ is a wave that is most negative at the maximum's position $p_M$, and it pushes the maximum below the runner-up;
- a new maximum exceeds all four thresholds, so all four comparators switch off, $V = 0$ and $s = m$. The old maximum has just become the runner-up, and it is now on top. Unit 4's pulse then restarts the rotation for the new maximum, with a radius equal to the new gap.

So the clock is a pending correction to the scores: how much the maximum has to be hidden, and at which position. Interventions agree. Setting the four comparators to zero once makes the network answer the maximum from then on (99.7–100% of cases). Scaling the clock's radius one step before the maximum is overtaken by 0, 0.5, 1, 2 leaves the correct answer in 2%, 34%, 99%, 75% of cases, and rotating it by one step moves the answer by one position in 38–47%.

**Your hypothesis** (section 8) is right in substance. The directions of the circle have a fixed meaning. The centre is set by $M$ (plus a smaller term in $S$) and the radius by the gap. The network has no bias, so the centre is a stored value: unit 2 re-supplies it to the comparators at every step, because the comparator block itself forgets any common level ($B_c\mathbf 1 \approx 0$). Two refinements: the circle is not there for the readout but for the comparisons, and in the wave units the orbit is not kept clear of the ReLUs; it touches them at fixed ages, by design.

**The network is wrong on 4.7% of inputs.** 18% of the errors have a runner-up that is not positive (zero is the network's reference; negative values are not stored). 75% have two of the top three within 5% of $M$ of each other. With a positive third and both gaps above $0.2\,M$ the accuracy is 99.97%.

![Six panels of evidence](mechanism_summary.png)

*Figure. (a) The clock coordinate $z$ divided by $10.157\,M$, medians by age of the maximum, for three sizes of the relative gap: same directions, radius set by the gap. (b) Positions of the four comparator thresholds and of unit 4's threshold between runner-up (0) and maximum (1), medians by age of the maximum. (c) The comparators' vote per unit of gap against the distance from the maximum's position (dots: fits for each age of the maximum), with $-60\cos(42.75°\,d)$. (d) Score minus the runner-up's score, in units of $M$, against the input's value relative to the runner-up, by the input's age. (e) The matrix $\Sigma$ that maps the scores to the scores one free step later. (f) Fraction of states in which the largest entry of $s_t$ is the runner-up so far (blue), and in which the largest entry of $m_t$ is the maximum so far (orange).*

## 2. Setting and exact facts

$$h_t = \mathrm{ReLU}(w\,x_t + W h_{t-1}),\qquad h_{-1} = 0,\qquad \text{logits} = U h_9 .$$

Here $w = W_{ih}$ (16 numbers), $W = W_{hh}$ ($16\times16$) and $U = W_{out}$ ($10\times16$). Inputs are iid $\mathcal N(0,1)$ unless stated. $M_t \ge S_t \ge T_t$ are the three largest of $x_0..x_t$. The *age* of a value is the number of steps since it arrived. The *gap* is $M - S$. $S^+ = \max(S, 0)$.

Unit groups (names from earlier work): **comparators** 1, 6, 7, 8 (input weights $-10.57, -11.00, -13.17, -12.31$; I write $g_j = -w_j$); **pulse units** 4 ($w_4 = +10.157$), 5, 9; **wave units** 0, 2, 3, 10–15 ($|w| \le 0.3$, except unit 14 with $-1.32$). I call units 2, 4, 1, 6, 7, 8 the **core** and the other ten the **satellites**.

Three facts hold exactly.

1. **Piecewise linear, no constants.** $h_t = D_t(w x_t + W h_{t-1})$ with $D_t$ the 0/1 matrix of active units. Scaling all inputs by $c > 0$ scales every $h_t$ by $c$. So no direction of state space can hold a constant, and every offset has to be a stored input value.
2. **Thresholds.** For a comparator, $h_j(t) = g_j\,[\theta_j(t) - x_t]_+$ with $\theta_j(t) = (W h_{t-1})_j / g_j$. For unit 4, $h_4(t) = w_4\,[x_t - \theta_4(t)]_+$ with $\theta_4(t) = -(W h_{t-1})_4 / w_4$. The thresholds are linear readouts of the previous state, in input units.
3. **Age coordinates.** The same $U$ is applied whatever happened before, and the recurrence does not know $t$. So $U h_t$ can be read at any step, with index $9 - a$ meaning "the input of age $a$".

## 3. What the state means: a tape of provisional scores

### 3.1 The state holds a running answer

Take the provisional scores $s_t = U h_t$, pick the largest entry among the indices of inputs that exist ($9 - t \le i \le 9$), and convert it to a position.

| $t$ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| P(it is the runner-up of $x_0..x_t$), given $S_t > 0$ | 0.992 | 0.987 | 0.971 | 0.966 | 0.969 | 0.970 | 0.970 | 0.967 | 0.961 |
| the same over all states | 0.688 | 0.676 | 0.777 | 0.839 | 0.893 | 0.927 | 0.945 | 0.953 | 0.953 |

The second row is lower at early steps because the runner-up is then often negative, and the network does not store negative values (section 7).

### 3.2 A step without an event shifts the tape

Let $A$ be $W$ restricted to the 13 non-pulse units (the "resting map": pulses off, the others on) and $U_o$ the matching columns of $U$. The scores obey a closed linear rule to good accuracy:

$$U_o A \approx \Sigma\,U_o,\qquad \Sigma = U_o A\,U_o^{+},\qquad \frac{\lVert U_o A - \Sigma U_o\rVert}{\lVert U_o A\rVert} = 0.095 .$$

$\Sigma$ is a shift (panel e). Its entries $\Sigma[i, i+1]$ are $0.97, 0.93, 0.90, 0.89, 1.00, 0.93, 0.93$ for $i = 0..6$; every other entry in rows 0–6 is at most 0.42 in size. The two newest slots are different: $\Sigma[7,8] = 0.34$ and $\Sigma[8,9] \approx 0$, because units 5 and 4 write them (section 6). Row 9 is about 1.4 times the sum of all scores, a level term. On real states the scores one free step later are predicted by $\Sigma$ with $R^2 = 0.985$–$0.992$ (positions 0–8, mean removed).

The rotating eigenmodes of $A$ (42.75°, 83.28°, 124.78°, 180° per step, radii 0.84–0.95) are the Fourier modes of this shift register.

### 3.3 Scores compare values with the runner-up

Scores relative to the runner-up's, in units of $M$, against the input's value relative to the runner-up, $v = (x_p - S)/M$ (panel d):

| input | law (medians) |
|---|---|
| arriving now, below the runner-up | $-61.6\,(S - x_t)$, linear over the whole range ($-55$ at $v = -0.9$) |
| arriving now, a new maximum | about $-60\,(x_t - M_{\text{old}})$ for a small excess; $-47$ at $v = 0.9$ |
| one step old, below the runner-up | $-60\,(S - x)$ for deficits below $0.15\,M$, then saturates at $-10$ to $-18\,M$ |
| two or more steps old, below | slope 60–85 near zero, saturates near $-20\,M$ |
| the maximum, two or more steps old | $-5$ to $-14\,M$, almost independent of the gap unless the gap is below $0.05\,M$ |

The same slope is produced by three routes, one per age:

- age 0, through the comparators: $U[9,\mathrm{COMP}]\cdot w[\mathrm{COMP}] = 5.01 + 7.38 + 49.90 - 4.04 = 58.25$ per unit of $x_t$ (unit 7 alone gives 49.9); for a new maximum, through unit 4: $U[9,4]\,w_4 = -50.70$;
- age 1, through unit 5: one unit of input drives unit 5 by $+18.01$ one step later and $U[8,5] = 3.398$, which gives $61.19$;
- ages $\ge 2$: unit 5's pulse enters the wave units and is shifted along the tape.

### 3.4 Two score vectors

Split the readout: $V_t = U[:,\mathrm{COMP}]\,h_t[\mathrm{COMP}]$ is the comparators' vote and $m_t = s_t - V_t$ is the rest.

**$m$ ranks the maximum first.** Among the inputs older than the current one, the largest entry of $m_t$ is the maximum of the prefix in 0.972, 0.942, 0.948, 0.953, 0.958, 0.964, 0.966, 0.891 of states for $t = 2..9$ (given $S_t > 0$ and a maximum that is not the current input).

**$V$ is a wave centred on the maximum.** At $t = 9$, on steps where all four comparators are on, a regression $V[i] \approx A\,(M - S) + B\,S + C\,x_9$ per age of the maximum has $R^2$ of 0.88–0.97. $A$ depends on the index only through the distance $d$ from the maximum's index (panel c):

| $d$ | 0 | $\pm1$ | $\pm2$ | $\pm3$ | $\pm4$ | $\pm5$ | $\pm6$ | $\pm7$ |
|---|---|---|---|---|---|---|---|---|
| $A(d)$, $d \ge 0$ | −62 | −48 | −6 | +40 | +60 | +47 | +14 | −25 |
| $A(d)$, $d \le 0$ | −62 | −42 | +7 | +59 | +82 | +61 | +14 | −33 |
| $-60\cos(42.75°\,d)$ | −60 | −44 | −5 | +37 | +59 | +50 | +14 | −29 |

$B$ has the same shape with an amplitude of about 8, so for $p \le 8$

$$V[p] \approx -60\,(M - 0.87\,S)\,\cos\!\big(42.75°\,(p - p_M)\big).$$

At the newest index, $V[9] \approx -50\,M - 7\,S + 58.2\,x_9$: the current input's own score.

**Why it is a cosine.** The readout columns of the four comparators are sinusoids over positions at the clock's frequency: $U[p,j] \approx a_j\cos(42.75°\,p - \varphi_j) + c_j$ for $p = 0..7$, with $a = 5.81, 5.09, 6.21, 6.08$, $\varphi = -29°, 26°, 131°, -130°$ and $R^2 = 0.974, 0.943, 0.991, 0.988$ for units 1, 6, 7, 8. The thresholds carry a pattern that advances 42.75° per step (section 4.1). The two are aligned: with the measured phase of the clock, the vote at the maximum's own position has phase 166°–188° for every age 2–9, so it is most negative there.

**The two parts nearly cancel in the margin between runner-up and maximum.** At $t = 9$, with $D = |p_M - p_S|$, medians in units of $M$:

| $D$ | gap/$M$ | $V[p_S] - V[p_M]$ | $60(M - 0.87S)(1 - \cos 42.75°D)$ | $m[p_S] - m[p_M]$ | sum: runner-up's margin |
|---|---|---|---|---|---|
| 3 | 0.05–0.15 | 20.3 | 21.0 | −12.8 | 7.8 |
| 3 | 0.15–0.30 | 32.8 | 31.5 | −24.1 | 8.8 |
| 3 | 0.30–0.50 | 49.0 | 45.7 | −39.3 | 10.2 |
| 3 | 0.50–0.80 | 68.7 | 63.4 | −58.7 | 10.5 |
| 4 | 0.30–0.50 | 61.7 | 56.2 | −47.7 | 13.8 |
| 1 | 0.30–0.50 | 11.4 | 7.5 | −6.2 | 4.8 |

The margin is positive in 99.7–100% of cases when the gap exceeds $0.05\,M$, and in 85–96% when it does not.

**At a new maximum, $s$ becomes $m$.** For new maxima at step $t$, the new scores of the older inputs equal the previous step's $m$, shifted: $R^2$ = 0.96, 0.77, 0.77, 0.88, 0.91, 0.89, 0.87 for $t = 3..9$, against at most 0.29 for the previous step's $s$. The old maximum is then the top entry among the older inputs in 96.4–99.6% of cases. Without an event, $s_t$ is the shifted $s_{t-1}$ ($R^2$ 0.67–0.96).

### 3.5 Interventions on the scores

1. **Switch the comparators off once.** At a step with no new maximum, set $h[1,6,7,8] = 0$ and continue with the real inputs. If this is done at step 8 or earlier, the network answers the *maximum* in 99.7–100% of sequences and the runner-up in 0.0% (final gap above 0.2). Setting four wave units to zero instead never makes it answer the maximum (at most 0.8%). So the other units alone rank the maximum first, the comparators hold what hides it, and it is not rebuilt without a new maximum.
2. **Change the clock one step before the maximum is overtaken** (152,690 sequences in which the old maximum ends as the runner-up; only the clock coordinate of the state is changed). Scaling the radius by 0, 0.25, 0.5, 0.75, 1, 1.5, 2 leaves the correct answer in 1.9%, 9.8%, 34%, 89%, 99.0%, 89%, 75%. Rotating the phase by $k$ steps moves the answer: for $k = -1$ the network answers one position later in 47% of cases (the old position in 9%); for $k = -2$, two positions later in 32% (old position 0.2%); for $k = +1$, one position earlier in 38% (old position 46%); for $k = +2$, one or two positions earlier in 55% (old position 7%).
3. **Erase the clock when nothing overtakes the maximum.** This costs 3 points of accuracy; when the maximum is later overtaken it costs about 44 points.

## 4. Where it is stored

### 4.1 The core: maximum, increment and clock in six units

For the core the recurrence can be written exactly in terms of the thresholds. Since $h_k = g_k[\theta_k - x_t]_+$ for the comparators,

$$\boxed{\;\theta(t+1) = B_c\,\big[\theta(t) - x_t\mathbf 1\big]_+ + b_2\,h_2(t) + b_4\,h_4(t) + \varepsilon(t)\;}$$

with $\theta = (\theta_1, \theta_6, \theta_7, \theta_8)$, $B_c[j,k] = W[j,k]\,g_k/g_j$, $b_2[j] = W[j,2]/g_j$, $b_4[j] = W[j,4]/g_j$, and $\varepsilon$ the contribution of the ten satellites. From the weights:

$$B_c = \begin{pmatrix} 0.409 & 1.036 & -1.306 & -0.140\\ -0.130 & 0.361 & -0.151 & -0.092\\ -0.081 & -0.152 & 0.078 & 0.156\\ 0.544 & 0.000 & -1.170 & 0.615 \end{pmatrix},\quad b_2 = (0.048, 0.049, 0.051, 0.051),\quad w_4 b_4 = (0.838, -0.044, 1.016, -0.034),$$

$$h_2(t+1) = 0.969\,h_2(t) + 1.731\,h_4(t) + \dots,\qquad h_4(t) = 10.157\,[x_t - \theta_4(t)]_+,\qquad \theta_4(t+1) = 0.0485\,h_2(t) + 0.0970\,h_4(t) + \dots$$

Four properties of these numbers carry the mechanism.

1. **The comparator block forgets a common shift.** $B_c\mathbf 1 = (-0.001, -0.012, 0.001, -0.011)$. Subtracting the same number from all four thresholds changes nothing one step later, and the input does exactly that. So an input below all four thresholds leaves no trace in the core. The same holds for every unit except 5 and 9: the cosine between a unit's weights on the comparators and the comparators' input weights is within $\pm0.04$ of zero for 14 units, and $0.74$ and $0.69$ for units 5 and 9. One unit of such an input sends $+18.0$ to unit 5 and $+84.3$ to unit 9 one step later, and at most 0.43 to any other unit.
2. **What is left is a rotation.** The eigenvalues of $B_c$ are $0.9375\,e^{\pm i\,42.56°}$ and a pair of size 0.043. $B_c$ rotates the *differences* between the thresholds by 42.6° per step and shrinks them by 6%. (In the 13-unit resting map the same mode is $0.951\,e^{\pm i\,42.75°}$.)
3. **Unit 2 supplies the common level.** Because the block forgets common shifts, it cannot hold the thresholds' own level. Unit 2 re-supplies it at every step: $b_2 h_2 \approx 0.05 \times 17.3\,M$, measured as 0.83–0.89 $M$ for each threshold (the satellites add 0.04–0.11 $M$). The weights are matched for this: $W[j,2]$ is $-0.048, -0.049, -0.051, -0.051$ times the input weight $w_j$, so unit 2 moves all four thresholds by the same amount. Unit 2 itself integrates unit 4's pulses (0.969 on itself, 1.731 from unit 4) and has an input weight of 0.015.
4. **A new maximum launches the rotation with the size of its increment.** If $x_t$ exceeds the old maximum by $d$, unit 4 fires $10.157\,d$ and the comparators are zero. One step later the thresholds are the level plus $w_4 b_4\,d = (0.84, -0.04, 1.02, -0.03)\,d$: $\theta_7$ and $\theta_1$ sit near the new maximum, $\theta_6$ and $\theta_8$ at the old one, which is now the runner-up. The spread is $d = M - S$, and $B_c$ rotates this spread from then on. That is why the radius of the rotation is the gap.

**The core runs almost by itself.** Simulating only the six core units on the same inputs, with the satellites' feedback replaced by a fixed multiple of unit 2 (taken from the level eigenvector), reproduces their activations with $R^2$ = 0.991, 0.995, 0.989, 0.995, 0.993, 0.991 for units 2, 4, 1, 6, 7, 8. With no feedback term at all, $R^2$ is 0.88–0.98.

**Measured laws of the thresholds** (panel b). The level, runner and clock modes of section 4.3 account for 96.4–99.6% of the variance of each threshold. With the measured laws of those modes this reads

$$\theta_j \approx A_j M + B_j S^+ + (M - S^+)\,C_j\,\rho(a)\cos\!\big(\varphi(a) + \delta_j\big)$$

with $A_j$ = 0.68–0.83, $B_j$ = 0.13–0.27, $C_j$ = 0.715, 0.397, 0.234, 0.726, 0.160 and $\delta_j$ = −154°, −79°, −4°, +125°, −7° for units 1, 6, 7, 8, 4. Here $a$ is the age of the maximum, $\varphi(a)$ = 0°, 37°, 80°, 123°, 166°, 209°, 251°, 294°, 338° for $a = 1..9$, and $\rho(a) \approx 1.02 \times 0.951^{a-2}$. Units 7 and 4 follow nearly the same law, close to $M$ while the maximum is young. Units 1, 6, 8 swing between about $S$ and well above $M$.

**Measured laws of the three hidden numbers**, in the eigen-coordinates of section 4.3 (gauge: one unit of unit 4's activation writes 1 into every mode):

- level $\approx 0.93 \times 10.157\,M$ from age 2 on ($1.006$ at age 1; within-age $R^2$ 0.987–0.997). For an isolated maximum it grows by 1.3% per step. On data it stays flat because unit 5 takes 3% off after every new runner-up.
- runner $\approx 10.157\,(2M + 5S)$; it sets unit 5's threshold (section 5.2).
- clock $z \approx 10.157\,\big[\rho(a)\,(M - S) + \delta\,S\big]\,e^{i\varphi(a)}$ with $\delta$ between 0.07 and 0.39. The phase is the same in every (age of max, age of runner-up) cell to within 3° (ages ≤ 5) or 7° (ages 6–9). Pooled, $M - S^+$ explains 94.9% of $|z|^2$, $M$ alone 82.7%, and the best two-term fit is $M - (0.75..0.88)\,S^+$ (panel a). Section 5 explains the departures from $M - S$.

### 4.2 The satellites: the tape

The eight wave units other than unit 2 hold the tape of section 3. Three properties are worth knowing.

1. **A zero travels through them with the age of the maximum.** Each of units 0, 12, 11, 10 is cut to zero by the ReLU at one particular age of the maximum, for every size of the gap: unit 0 at age 2 (100% of states), unit 12 near zero at age 4 (exactly zero in 21–42%), unit 11 at age 5 (86–99%), unit 10 at age 6 (97–98%). Each has its most negative readout weight at the matching position: $U[7,0] = -2.90$, $U[5,12] = -2.79$, $U[4,11] = -4.56$, $U[3,10] = -3.87$. So in $m$ the maximum's position is the one that these units do not vote against. The cut itself matters for unit 0 and not for the others: letting one unit go negative instead of zero lowers accuracy by 19.0 points for unit 0, 7.3 for unit 15, 1.0 for unit 3, 0.9 for unit 10 and 0.3 or less for units 11, 14, 12, 13, 2 (all nine together: accuracy 0.44).
2. **Their swing does not vanish when the gap closes.** Between the zeros the activations scale with $M$ and with the gap (unit 12 peaks at $15\,M$ for gaps of 0.15–0.3 $M$ and at $31\,M$ for gaps above $0.85\,M$). In the comparators the swing follows the gap alone (ratio of swings between those two gap ranges: 0.29–0.37; ratio of $M - 0.87S$: 0.35).
3. **Records are dips.** Unit 5 writes into the wave units with negative weights ($W[0,5] = -0.91$, $W[15,5] = -0.66$, $W[3,5] = -0.52$, $W[14,5] = -0.42$), and the readout reads those units negatively at the matching position, which gives a positive score. A dip needs a positive baseline. The level, proportional to $M$, is that baseline.

### 4.3 The same thing in eigen-coordinates

The resting map $A$ has these modes. "Seen by the readout" is the size of $U_o v$ over positions 0–8 after removing their mean, per unit coordinate, in the gauge above.

| mode | eigenvalue | what it holds on data | read from (left vector) | seen by the readout |
|---|---|---|---|---|
| level | 1.0099 | $0.93 \times 10.157\,M$ | unit 2 (85% of its squared norm) | 0.33 |
| runner | 0.9735 | $10.157\,(2M + 5S)$ | unit 2 and comparators | 0.11 |
| clock | $0.951\,e^{\pm i 42.75°}$ | gap × phase of the maximum's age | comparators (97%) | 0.91, 0.75 |
| second harmonic | $0.937\,e^{\pm i 83.28°}$ | records on the tape | mixed (61% comparators) | 2.6, 2.4 |
| alternating | −0.928 | records on the tape | wave units (89%) | 1.4 |
| third harmonic | $0.837\,e^{\pm i 124.78°}$ | records on the tape | wave units (65%) | 2.6, 2.1 |
| short clock | $0.735\,e^{\pm i 42.99°}$ | the most recent event | comparators (92%) | 9.4, 5.6 |
| transient | 0.231 | one-step event flag | comparators (98%) | 24 |
| echo | 0.001 | the current input | | 0.04 |

Four points.

- **The level vector is positive on all 13 units** (relative to unit 2: 1.35, 0.51, 1, 0.19, 0.63, 0.79, 0.70, 1.06, 1.14, 1.59, 0.43, 1.23, 0.81 for units 0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15), although $A$ has 67 negative entries of 169.
- **The input direction is the echo mode.** The cosine between $w$ on these units and the eigenvector with eigenvalue 0.001 is 0.998.
- **The readout does not see the clock because two halves cancel.** The comparators' part and the wave units' part of the clock's readout have sizes 5.6–8.4 and 5.7–9.5 per unit coordinate (positions 0–7) and are 178°–180° out of phase. Their sum is 0.4–1.6, which is 6% of the two sizes added. The comparator half is $V$; the wave half is the bump at the maximum in $m$. A new maximum removes the first and leaves the second.
- **"Does not see" is approximate.** On data, level and clock together make up 19% of the rms of the logit differences among positions 0–8. Setting the clock to zero in the final state changes 7.8% of answers and lowers accuracy by 4.9 points. The leak is largest at positions 0–2.

This linear picture describes the gate pattern "13 units on, pulses off". That exact pattern holds on 10% of all steps (29% at the last step) and in 19% of states whose maximum is at least two steps old. 1377 different gate patterns occur in $2^{20}$ sequences; 10 of them cover half of all steps and 75 cover 90%. The departures are the events of section 5 and the age-locked cuts of the wave units. The rms size of the ReLU corrections to the resting map is 6.6% of the state norm once the maximum is three or more steps old.

## 5. How it is updated

Every input is compared with seven thresholds that the state supplies. Five act at once: $\theta_4$, slightly below $M$, and the four comparator thresholds, which lie between about $S$ and somewhat above $M$ and swing with the clock. Two act one step late: $\Theta_5 \approx 0.75\,S$ for unit 5, and a threshold for unit 9 that follows $\theta_4$ while the maximum is young and lies above it later. All seven are exact thresholds: each gate switches at one value of the input (no exception in $5 \times 10^7$ and $2 \times 10^7$ evaluations). What happens depends only on which thresholds the input exceeds.

| input $x_t$ | share of steps 1–9 | what fires | what changes |
|---|---|---|---|
| below $\Theta_5$ | 53% | nothing | nothing lasting |
| between $\Theta_5$ and $S$ | 4% | unit 5 at $t+1$ | a score is written on the tape at position $t$ |
| between $S$ and $M$ (new runner-up) | 21% | comparators with a threshold below $x_t$ are cut at $t$; unit 5 at $t+1$ | score at $t$ above the old runner-up's; stored $S$ raised; clock radius reduced |
| above $M$ (new maximum) | 21% | unit 4 at $t$, all four comparators cut; units 5 and 9 at $t+1$; unit 0 cut at $t+2$ | old maximum promoted; stored $M$ raised; clock restarted with radius = new gap |

(The shares of new maxima and new runners-up are $1/(t+1)$ each at step $t$. Unit 4 fires on 99.94% of positive new maxima and never on a negative one. Unit 5 fires after 99.9% of positive new maxima and positive new runners-up, and after 6.5% of inputs below the runner-up.)

### 5.1 An input below everything

All four comparators stay on and hold $g_j(\theta_j - x_t)$. By section 4.1 this leaves nothing in the core one step later, and at most 0.43 per unit of input in any unit other than 5 and 9. Those two receive $+18.0\,x_t$ and $+84.3\,x_t$ on top of large negative terms from the state, and stay off. Two small leaks exist: per unit of input the level coordinate changes by $-0.0447$ and the runner coordinate by $+1.026$, a tenth of what a new maximum writes.

The input's own score exists for one step only, at index 9: $s[9] \approx s[\text{runner-up}] - 61.6\,(S - x_t)$. One step later its slot holds the baseline of an unrecorded input, 10 to 20 $M$ below the runner-up.

Tested by resampling: replacing one input that is below $S/2$ by another such value changes the answer in 0.55% of cases and the margin of the answer by a median of 0.13 logits (the median margin is 8.5). Resampling all such inputs of a sequence at once (3.9 on average) changes 1.3% of answers. What a low input leaves in the 13 non-pulse units is 2–3% of its direct effect. The network's boundary for "low" is $\Theta_5$, not $S$: resampling an input within $(0.7\,S, S)$ changes 5.5% of answers.

### 5.2 An input just below the runner-up: unit 5 is a fifth comparator

When neither comparator 7 nor 8 is cut at step $t$, unit 5's input at the next step is exactly

$$z_5(t+1) = 18.009\,\big(x_t - \Theta_5(t)\big) - 0.281\,x_{t+1} + (\text{clipped wave units}),\qquad \Theta_5 = 0.01265\,m_{\text{runner}} - 0.02872\,m_{\text{level}} + \text{small}.$$

On data $\Theta_5 \approx (0.70\text{–}0.86)\,S - (0.01\text{–}0.07)\,M$, and it does not see the clock (at most 0.037 of the gap). This is the job of the runner mode: it holds the runner-up's value as the threshold of unit 5. Unit 5 fires after 0.1% of negative inputs, 51% of inputs with $x/S \in [0.7, 0.8)$, 89% for $[0.8, 0.9)$ and 99.5% for $[0.9, 1)$.

Its pulse, about $18.0\,(x_t - \Theta_5)$, is read directly at index 8 ($U[8,5] = 3.40$, so 61.2 per unit of $x_t$). It is also written into the wave units, where it becomes a bump at position $t$ that the tape then shifts: about $+3.3$ per unit of pulse at its own position and $-0.4$ to $-0.7$ at the runner-up's.

These records keep the score linear in the value near the runner-up, which a comparison by value needs. They also cost accuracy, because an input slightly below the runner-up gets a score slightly below the runner-up's. Forcing unit 5 off after every input below the runner-up raises accuracy from 95.30% to 95.89%; almost all of the gain comes from inputs with $0.9 \le x/S < 1$.

### 5.3 A new runner-up

Take $S < x_t < \theta_4$. Three things happen.

1. **Its score.** At step $t$ its score at index 9 is above the old runner-up's. At $t+1$ unit 5 fires (99.9% of cases), with slope 18.0 per unit of $x_t$ if no comparator is cut and 7.5 if comparator 8 is cut. The result is a bump at position $t$ of $+42$ to $+48$ per unit of $x_t$ when the maximum is 1–4 steps old, and $+57$ to $+61$ when it is 5–8 steps old; 74–100% of it comes from unit 5. The old runner-up's position falls by 9–23 per unit. The final margin of the new over the old runner-up is $(58\text{–}72)\,(x_t - S)$, with no separate term in $S$ or $M$.
2. **The stored runner-up value.** The runner coordinate rises by $(5.7\text{–}6.5) \times 10.157\,(x_t - S)$ when the maximum is 1–4 steps old, and by $(2.6\text{–}3.8) \times 10.157\,(x_t - S)$ when it is older. $\Theta_5$ follows with a gain of 0.58–0.73 at ages 1–4 and only 0.36–0.41 at ages 5–6.
3. **The clock.** The comparators whose thresholds lie below $x_t$ are cut ($[\theta_k - x_t]_+ = 0$ in the boxed equation). Which ones depends on the age of the maximum, because the thresholds swing. At ages 1–4 at least one threshold sits near $S$ and nearly every new runner-up cuts it. At ages 5–8, 42–66% of new runners-up cut none. When comparators are cut, the clock radius falls by $(0.84, 0.83, 0.75, 0.53) \times 10.157\,(x_t - S)$ at ages 1–4, with little change of phase. From age 5 it hardly changes.

So "radius = gap" is exact at the launch (up to the sag of $\theta_4$) and is afterwards maintained only for runners-up that arrive within about four steps of the maximum. In a fit $z \propto b_M M + b_S S$ the ratio $b_S/b_M$ is $-0.90$ to $-0.94$ when the runner-up arrived one step after the maximum and $-0.88, -0.83, -0.72, -0.50, -0.38, -0.28$ for 2 to 7 steps after. The answer does not depend on this update: when the final runner-up arrived 5–8 steps after the maximum and cut nothing, the network is still right in 91–95% of cases.

A controlled sweep shows all of this in one sequence, $x = (0.5, b, 1.5, b, b, b, v, b, b, b)$ with $b = -1$. Before $v$ arrives the thresholds are $\theta_8 = 0.555$, $\theta_1 = 1.274$, $\theta_4 = 1.366$, $\theta_7 = 1.381$, $\theta_6 = 1.673$.

| $v$ | comparators cut | unit 4 | unit 5 at next step | clock radius at $t = 9$ (/10.157) | answer |
|---|---|---|---|---|---|
| $\le 0.3$ | none | 0 | 0 | 0.90 | 0, correct |
| 0.45 | none | 0 | 2.8 | 0.92 | 0, correct (margin 1.7) |
| 0.55 | none | 0 | 4.6 | 0.93 | 6, correct |
| 0.85 | 8 | 0 | 6.9 | 0.77 | 6 |
| 1.15 | 8 | 0 | 9.1 | 0.61 | 6 |
| 1.30 | 1, 8 | 0 | 10.3 | 0.52 | 6 |
| 1.45 | 1, 7, 8 | 0.85 (false firing) | 11.9 | 0.18 | 6 (margin 13.9) |
| 1.55 | 1, 7, 8 | 1.86 | 13.0 | 0.07 | 2, correct: new maximum |
| 1.80 | all | 4.40 | 16.0 | 0.40 | 2 |

### 5.4 A new maximum

Let the new maximum $M_n$ exceed the old one $M_0$ (age $g$) at step $s$, with old runner-up $S_0$. The sequence of gates is fixed.

**Step $s$.** Unit 4 fires, $h_4 = 10.157\,(x_s - \theta_4)$, in 99.997% of such steps. All four comparators are cut in 94–96%; a small increment can leave one on, as in the last rows of the sweep. Three consequences follow at once.

- $V = 0$, so the scores of the older inputs are the previous $m$, shifted (section 3.4). Relative to the other old positions, the old maximum's score rises by 28 on average: from 13.5 to 41.8, while the old runner-up goes from 20.1 to 15.9. In an exact decomposition of that rise, the removal of the comparators' share of the clock accounts for $+28.0$ of $+28.3$. From the weights alone, a clock state with the comparators set to zero has its largest logit exactly at the old maximum's position for every age 2–9 and every number of later steps 0–6 (35 of 35 cases), 4.0–10.6 logits per unit of clock radius. The rise is roughly $48\,(M_0 - S_0) + 12\,S_0$.
- The new maximum's own score is low: $U[9,4]\,w_4 = -50.7$ per unit of excess.
- Most of the old clock is removed, because the clock coordinate is read mostly from the comparators. Between 5% and 35% of it survives, depending on the old maximum's age: the wave units carry the rest of the coordinate.

**Step $s+1$.** The comparators are back on. Their thresholds are the level plus unit 4's supply: $\theta_7 \approx 1.02\,M_n$, $\theta_4 \approx 0.985\,M_n$, $\theta_1 \approx 0.83\,M_n + 0.16\,M_0$, $\theta_6 \approx 0.98\,M_0$, $\theta_8 \approx 1.07\,M_0$. Units 5 and 9 fire (100%, 99.7%), because the comparators that normally hold them down were zero: $h_5 \approx 12.2\,(M_n - M_0) + 10.4\,M_0 - 5.8\,S_0$ and $h_9 \approx 1.56\,(M_n - M_0) + 1.77\,M_0 - 0.43\,S_0$. The exact parts are $W[5,4]\,w_4 = 12.06$ and $W[9,4]\,w_4 = 1.48$.

**Step $s+2$.** Unit 4's pulse has entered the 13 other units with coordinate 1 in every mode: level, clock, and the modes of the tape. On the tape this is a negative bump at the new maximum's position. Unit 5's pulse arrives one step behind and is its mirror image on the tape: in every mode that the readout sees, its write is 168°–189° away from unit 4's (rotated by one step) and of similar size, and it puts almost nothing into level or clock. Unit 0 is cut at this step (99.7%), by unit 5 ($-0.913\,h_5 \approx -12.9$) and unit 9 ($-1.424\,h_9 \approx -3.3$). After this step the new gap $d$ is held in the level ($0.92 \times 10.157\,d$), the runner mode, and the clock ($1.04 \times 10.157\,d$ at phase 36°). Only 4%, 11% and 8% of unit 4's kick remain in the three long-lived tape modes.

**Step $s+3$ onward.** No special gates. The level is $0.93 \times 10.157\,M_n$: one step after the new maximum it was 1.006, and unit 5 took off 0.087 and unit 9 0.019. The clock turns 42.75° per step. Its first step is 37° because of the two pulses: $0.951@42.7° + 0.088@{-137.9°} + 0.133@{-4.4°} = 0.958@37.0°$, exactly.

Two things about this sequence were not obvious and were tested.

- **Unit 5's pulse after a maximum prepares that maximum for being overtaken later.** Removing only the pulse that followed the *old* maximum's arrival drops the probability that it is answered after being overtaken from 0.967 to 0.020, although the later new maximum still raises its score by the same 28. Without that pulse its score before the overtaking is 12.6 below the other old positions instead of 12.3 above. Removing the pulse after the *new* maximum costs little (0.973 to 0.927).
- **Unit 9 and the cut of unit 0 keep the standing maximum below the runner-up.** With unit 9 forced off after the final maximum, the network answers the maximum itself in 44.5% of cases. With unit 0 forced on at $s+2$, 66% of the changed answers are the maximum. With units 5 and 9 both off it never answers the maximum. So unit 5 raises the maximum's own score and unit 9 with unit 0 lower it again.

**New maxima at the last two steps** have no time for this sequence. At $s = 9$ everything is read directly: unit 4 through $U[9,4] = -4.99$, and the cut comparators through the absence of their vote (reading them as if uncut flips the choice among the old positions from the old maximum, 95.2%, to the old runner-up, 91.4%). At $s = 8$ units 5 and 9 are read directly ($U[8,5] = +3.40$, $U[8,9] = -1.66$), with the same signs that their writes produce later. The old maximum is answered in 94.3% and 97.3% of these two cases.

## 6. How the answer comes out

The final readout is nothing special: it is the same reading of the tape that gives the running answer at every earlier step. Three points complete the picture.

**The two newest inputs are scored by separate routes, in the same currency.** For the last input the logit is exactly

$$L_9 = -49.90\,[\theta_7 - x_9]_+ - 7.38\,[\theta_6 - x_9]_+ - 5.01\,[\theta_1 - x_9]_+ + 4.04\,[\theta_8 - x_9]_+ - 50.70\,[x_9 - \theta_4]_+ + \rho_9,$$

where $\rho_9$ collects the wave units and units 5 and 9. This is a tent in $x_9$. It rises by 61.0–62.1 per unit relative to every other logit while $x_9$ is below the thresholds, and falls by 54.6–57.8 per unit once $x_9$ is above all of them. For a fixed prefix the network answers "9" on a single interval of $x_9$. Its lower edge is the stored runner-up (slope 1.013 against the true one, $R^2 = 0.990$) and its upper edge is the stored maximum (slope 0.998, $R^2 = 0.996$). The input before last is scored through unit 5: $h_9[5]$ rises by 18.3 per unit of $x_8$ and $U[8,5] = 3.40$, which gives 62.1 per unit. Setting $U[8,5]$ to zero drops the accuracy for a runner-up at position 8 from 0.944 to 0.017. Setting $U[9,7]$ to zero drops overall accuracy to 0.548.

**The runner-up beats the maximum by a small margin of fixed size, not by the gap.** The runner-up's score exceeds the standing maximum's by 3–14 $M$ (medians; less when the two are adjacent), almost independently of the gap unless the gap is below $0.05\,M$. The maximum's side of this margin is the balance of the writes at its own position described in section 5.4, followed by the cuts of units 12, 11, 10 at ages 4, 5, 6. An exact bookkeeping of the final logits by source (inputs, pulses, and every ReLU cut, each propagated linearly) reads the same way. When the runner-up arrived after the maximum, the margin over the maximum is carried by the maximum's own marks (unit 4 plus the wave-unit cuts: $+67$ on average, positive in 99.7% of sequences) against the tape ($-32$). When the runner-up is an overtaken maximum, it is carried by the comparator cuts at the moment of overtaking ($+26$).

**The runner-up beats the third by a comparison of values.** The coefficients of $S$ and $T$ in the margin between their logits are nearly equal and opposite: $+70$ to $+80$ and $-69$ to $-80$ for near ties, and the margin is linear in the gap over $\pm 0.15\,M$. The slope depends on the order of arrival (77 per unit when the third came before the maximum, 50 when the runner-up landed on a third that was already recorded, 64–66 when the third came last). How the comparison is made depends on the order too:

- if both arrived after the maximum, each was scored against the stored runner-up when it arrived (sections 5.2 and 5.3);
- if the runner-up is an overtaken maximum and the third is the previous runner-up, the comparison was made at the moment of overtaking by the clock's deposit, whose size is the old gap. Because the deposit is a first harmonic in position, the old maximum's lead is $c(\Delta)\,(M_0 - S_0)$ with $c$ = 17.6, 43.5, 73.8, 88.5, 81.6, 63.2, 41.8 for separations $\Delta = 1..7$ between the two positions, and $c(\Delta)/(1 - \cos 42.75°\Delta)$ = 65, 47, 46, 45, 45, 51, 82.

No single part of the state holds the answer. In the linear bookkeeping, dropping any one class of sources changes many answers (without unit 9's pulses: 88.6% agreement with the network; without the wave-unit cuts: 45.6%; without unit 5: 11.8%). Each margin is a small remainder of large terms.

## 7. Where it fails

Accuracy is $0.9529 \pm 0.0002$ on $2^{21}$ sequences. Of the errors, 63.6% name the third largest, 20.2% the maximum and 16.2% a smaller input. 92.4% of all errors fall in two classes.

**(1) The runner-up is not positive (17.8% of errors).** The state starts at zero and all thresholds are exactly zero at the first step, so zero is the network's reference. Negative inputs are summed weakly, not stored as values. Accuracy is 0.119 with no positive input, 0.231 with one and 0.979 with two. The limit is soft: a runner-up at $S = sM$, with a clearly negative third, is found with probability 0.15, 0.31, 0.51, 0.83, 0.93, 0.96 for $s$ = −0.25, −0.10, −0.05, 0, 0.05, 0.10. The same reference explains what happens when a constant $c$ is added to all inputs: accuracy 0.116, 0.134, 0.547, 0.859, 0.953, 0.952, 0.940, 0.912, 0.881 for $c$ = −3, −2, −1, −0.5, 0, 0.5, 1, 2, 3. Negative shifts make the runner-up negative; positive shifts shrink the gaps relative to $M$.

**(2) A relative gap below $0.05\,M$ (74.6% of errors: runner-up close to third 54.1%, maximum close to runner-up 15.7%, both 4.9%).** With a positive third and both gaps above $0.2\,M$ the accuracy is 99.97%. Four mechanisms were separated.

- *The zero of the value comparison is displaced.* A later candidate takes over from an earlier one when it exceeds it by $\varepsilon^* M$, with quartiles of $\varepsilon^*$ at −0.010, −0.001, +0.011. 70% of the variance of $\varepsilon^*$ is a fixed function of the three positions (earlier candidate, later candidate, maximum), and inputs with only three non-background values reproduce it ($r = 0.91$). So most confusions of runner-up and third are systematic and tied to positions, not noise from the other inputs.
- *The deposit is a single harmonic.* For "maximum arrives after both", the rate of answering the third at a relative gap $(S-T)/S$ of 3–8% is 6.6%, 3.8%, 2.9%, 2.0%, 2.8% for separations 1–5, and 13.7%, 17.1%, 34.6% for separations 6, 7, 8. At a gap of 8–20% it is at most 1% for separations 2–5 and 12% for separations 7 and 8. A first harmonic with a period of 8.4 positions cannot separate positions that are 7 or 8 apart, and separates neighbours poorly. Large separations also mean old records; the two effects were not separated.
- *Records the network does not need.* An input slightly below the runner-up fires unit 5 and gets a score slightly below the runner-up's (section 5.2). Forcing unit 5 off after every input below the runner-up raises accuracy to 0.9588 and removes 41% of the errors in the class "runner-up close to third, runner-up first".
- *Unit 4's threshold sags.* $M - \theta_4 = \varepsilon(a) M + \kappa(a)(M - S)$ with $\kappa$ up to 0.26 when the maximum is 5–6 steps old, because 16% of the threshold rides on the clock. Unit 4 then fires for 19–29% of new runners-up at those ages. Removing these false firings raises accuracy to 0.9563 and repairs 64% of the errors in the class "maximum close to runner-up, maximum first". The damage is limited because the rest of the network is calibrated to the sag: the level at which a later input is treated as the new maximum stays within about 2% of $M$. For the same reason, replacing unit 4 by an exact detector of new maxima *lowers* accuracy to 0.9314. The network's own pulse is $10.157\,(x - \theta_4)$, the increment plus the sag, and that sum is what resets the thresholds: $\theta_4 \leftarrow \theta_4 + 0.986\,(x - \theta_4)$.

Both repairs together give 0.9621. The remaining errors include the mirror cases (the later input is the larger one), which neither repair touches.

Accuracy is not lower for an old runner-up (0.958–0.965 at ages 5–8). It is lowest when the maximum is the last input (0.935) or the first (0.931).

## 8. Your hypothesis, point by point

> The age of the current max is stored in a circle, where the directions have a fixed meaning, but the centre and radius of the circle depend on the current max and second max values. Probably because (a) the circle has to be off-centre, to not be cut off by ReLUs; (b) to rotate around a point off centre, you need a bias; (c) the network has no bias, so it makes an ersatz bias from the values it has stored.

**Verdict: right in substance. The sharper statements follow.**

**The directions have a fixed meaning.** The clock coordinate has phase 0°, 37°, 80°, 123°, 166°, 209°, 251°, 294°, 338° when the maximum is 1..9 steps old, the same in every cell of (age of max, age of runner-up) and independent of $M$. The age can be decoded from the angle alone in 99% of states when the gap is at least half of $M$, in 89% for gaps of 0.25–0.5 $M$, and in 28% when the gap is below $0.05\,M$, where the circle is small.

It is causal. Multiplying the clock coordinate by $e^{ik\,42.75°}$ at some step makes all five thresholds follow the law of a maximum $k$ steps older (mean deviation 0.030, against 0.331 from the law of the true age). If that maximum is later overtaken and ends as the runner-up, a shift by one step moves the network's answer by exactly one position in 34–50% of runs. Shifting the other four rotating modes instead does much less.

**The centre is set by $M$, with a second, smaller term set by $S$.** In eigen-coordinates the centre is $10.157\,[\,\ell M\,r_{\text{level}} + (r_0 M + r_1 S)\,r_{\text{runner}}\,]$ with $\ell \approx 0.93$, $r_0 \approx 2$, $r_1 \approx 5$. The level vector is positive on all 13 units. The runner vector has mixed signs: it lowers the centre of units 0, 10, 11, 12, 14, 15 by 6–10 per unit of $S$ and raises unit 2 and the comparators. For the impulse input of your `rotate/` analysis, the plane's offset from the origin (34.99) is the level mode and the in-plane rotation (radius 22.1) is the clock mode.

**The radius is set by the gap, not by $M$ and $S$ separately.** At launch it is unit 4's pulse, $M_n - \theta_4 \approx M - S$, because unit 4 feeds two of the four thresholds and not the other two. Afterwards $|z|/10.157 \approx \rho(a)(M - S) + \delta S$ with $\rho(a)$ the free decay and $\delta$ between 0.07 and 0.39. The $\delta$ term has two sources: 5–35% of the previous clock survives a new maximum, and a runner-up that arrives five or more steps after the maximum usually leaves the radius at the old gap. In two exact examples: $x = (1,0,0,0,2,0,\dots)$ gives $1.22\,\rho(a)(M - S)$ and $x = (2,0,0,0,1,0,\dots)$ gives $1.51\,\rho(a)(M-S)$.

The radius is the gap because the radius is the size of the correction that the readout applies at the maximum's position (section 3.4), and it becomes the old maximum's lead over the old runner-up when a new maximum arrives (section 5.4).

**(a) Off-centre because of the ReLUs: yes for the comparators, where the circle lives. In the wave units the circle is cut on purpose.**

- In the comparators the offset is the threshold itself. $h_j = g_j[\theta_j - x_t]_+$ and the thresholds are $S + \rho_j(a)(M - S)$ with $\rho_j$ between about 0 and 1.45. For any input below the runner-up all four stay on and the rotation continues untouched. A new maximum pushes all four below zero at once, and that is how most of the old rotation is removed. The ReLU boundary is the reset mechanism.
- In the wave units the orbit is tangent to the boundary. At $S = 0$ the clock's amplitude is within ±22% of the centre for 9 of the 13 units. Units 0, 12, 11, 10 are cut when the maximum is 2, 4, 5, 6 steps old for every size of the gap (section 4.2). These cuts change the clock by at most 2%, because the clock coordinate is read from the comparators.
- The offset is needed. With the level removed from the state at one step, about six units are cut at the next step and accuracy falls from 97% to 11–21%, even if no later input can change the top two. With half the level it is 30–37%; with 1.25 times the level, 87–94%. In a linear model, about half of the entries of the orbit would be negative without the centre and 4–6% with it. The controls point the same way: erasing the clock at a fixed centre cuts nothing and costs 5–6 points, while enlarging it by half cuts 1.6 more units and costs 15–22 points.

**(b) A rotation about a point off the origin needs a bias: exact.** Within one gate pattern the map is linear, $h' = A h$. A rotation about $c \ne 0$ requires $A c = c$. So the centre has to be an eigenvector with eigenvalue 1, and its length is a free stored number. $A$ has two real eigenvalues near 1: 1.0099 and 0.9735.

**(c) The stored maximum is the ersatz bias: yes, and the weights show how.**

- In the comparator block $B_c\mathbf 1 \approx 0$: the block forgets any common shift of the four thresholds. It has to, because the input enters as a common shift and must be forgotten. So the block cannot hold its own centre.
- Unit 2 supplies the centre at every step. Its weights into the comparators are $-0.048, -0.049, -0.051, -0.051$ times their input weights, so it shifts all four thresholds equally, by $0.05\,h_2 \approx 0.85\,M$. Unit 2 integrates unit 4's pulses and has an input weight of 0.015, so the input cannot disturb it.
- By homogeneity nothing else is possible. Scaling the inputs scales every activation, so no unit can hold a constant. An offset has to be proportional to something stored, and the largest value so far is the natural choice, since every threshold has to sit near it.

**Two things the hypothesis does not say.**

1. The circle is not for the readout. The readout is arranged not to see it (section 4.3). The clock is a correction held in reserve, used in full only when its owner is overtaken.
2. A runner-up that arrived after the maximum has no circle of its own. Its position is a bump on the tape, written by unit 5. Each mode that the bump is made of then turns at its own rate (fits with the modes' own eigenvalues have a median $R^2$ of at least 0.998). The 83° and 125° modes carry a packet whose phase tracks the more recent of the two top-two arrivals.

## 9. How I know

Evidence types: **W** = follows from the weights by linear algebra; **M** = measured on sampled inputs; **F** = regression; **I** = intervention on the running network. "Lead" scripts are `hide/e01`–`e26`. "Arm X" is one of eight independent sub-investigations (`hide/r1_X/`, results in `hide/notes/r1_X.md`). The last column says who re-derived the claim independently.

| claim | type | source | independent check |
|---|---|---|---|
| exact structure, threshold identities, homogeneity | W | lead e21; arms B, E, I | all arms (errors ≤ $10^{-12}$) |
| eigenvalues of the resting map; positive level vector; input = echo mode | W | lead e01 | arms B, C, D, F |
| running answer at every step (3.1) | M | lead e12 | PENDING |
| shift matrix $\Sigma$ (3.2) | W, M | lead e10 | PENDING |
| score laws; slope 61 by three routes (3.3) | M, W | lead e13, e25 | arm E (61.0–62.1 at index 9; 62.1 at index 8); arm H (58–72 for new runners-up) |
| $m$ ranks the maximum first; $V$ is a cosine at the maximum (3.4) | M, F, W | lead e15, e15b, e16, e25 | PENDING; arm E (clock read through wave units alone: +35 to +61 at the maximum; through comparators −35 to −54) |
| $s$ becomes $m$ at a new maximum | M | lead e15b | arm F (rise of 28 at the old maximum, of which 28.0 from the comparators' share of the clock) |
| wiping the comparators makes the network answer the maximum | I | lead e17 | PENDING |
| clock radius and phase before an overtaking steer the answer | I | lead e18 | arm F (zeroing: 0.97 to 0.10; rotation moves the answer by $-k$ in 28–40%, never the wrong way); arm C (34–50%) |
| threshold recurrence; $B_c\mathbf 1 \approx 0$; unit 2 supplies the level; launch pattern $b_4$ (4.1) | W | lead e21 | PENDING; arm F (thresholds one step after a new maximum) |
| six-unit core simulation | M | lead e23 | PENDING |
| phase law of the clock | F | lead e02, e05 | arms C, D, F, H (within 1–8°) |
| radius law and its departures | F, I | lead e03, e05, e24; arms C, D, H | three arms agree on $\rho(a)(M-S) + \delta S$ |
| level law; level bookkeeping | W, F | arm B | arms D, F |
| readout halves of the clock cancel to 6% | W | lead e07 | arms C, E (0.0575) |
| travelling zeros in units 0, 12, 11, 10 (4.2) | M | lead e20 | arms C, D, E; PENDING |
| unit 5 as fifth comparator; $\Theta_5$ from the runner mode (5.2) | W, F | arm H | arm I ($\theta_5 \approx 0.8 S^+ - 0.05 M$) |
| new runner-up: score, runner mode, clock update (5.3) | M, F, I | arms H, C; lead e19 | arms C and H agree; arm D (two-pulse runs) |
| new maximum: gate sequence, pulse amplitudes, anti-kick, unit 0 (5.4) | W, M, I | arm F | arm B (level steps 1.006 and 0.927); arm E (unit 0 cut at age 2 with P = 0.998) |
| the old maximum's own unit-5 pulse is necessary | I | arm F | not repeated |
| last-step formula; interval of $x_9$; readout-weight ablations (6) | W, I | arm E | lead e25 (weights) |
| value comparison between runner-up and third | F | arms H, I | arms H and I agree (slopes 50–80) |
| deposit is a first harmonic; errors by separation | F, M | lead e14, e22 | PENDING; arm I (slope 24–46 for adjacent old candidates, 100–155 at four apart; offsets of 5–11% of $M$ at separations 7–8) |
| error census; shift; repairs; oracle for unit 4 lowers accuracy (7) | M, I | arm I | arm B (oracle 93.1%), arm H (repairs) |
| centre: level removal and scaling (8) | I | arm D | arm B (level scaling shifts thresholds and answers) |

## 10. Remaining uncertainties

**Measured but not derived.**

1. *The calibration.* Several balances make the algorithm work, and I can only report them. The coefficients of the runner-up and the third in their margin are equal and opposite. The standing maximum ends just below the runner-up through the balance of unit 5 against unit 9 and the cut of unit 0. The comparators' vote is aligned to within a few degrees with the maximum's position. I have no short calculation that produces these numbers from the weights.
2. *The position-dependent offsets.* The zero of the comparison between runner-up and third is displaced by an amount that is 70% determined by the three positions involved. The single-harmonic deposit explains the pattern for "maximum last", but I have not derived the table of offsets, and at large separations the effect is confounded with the age of the records.
3. *The order dependence of the slope* of that comparison (77, 50, 64–66 logits per unit by order of arrival).

**Where the description is incomplete.**

4. *Gate patterns.* The linear picture (the eigenmodes of the resting map, the shift matrix $\Sigma$) describes a gate pattern that holds in a minority of states. The cuts of the wave units are locked to the age of the maximum and they matter: without the wave-unit cuts the linear bookkeeping agrees with the network on 46% of inputs. I describe what the cuts of units 0, 10, 11, 12 do. I have no account of units 3, 13, 15 and only a partial one of unit 14.
5. *Two bookkeepings.* The split $s = m + V$ attributes the maximum's suppression to the comparators' vote; the linear bookkeeping by source attributes it to unit 4's pulse and the later wave-unit cuts. Both are exact identities, and the interventions agree with both. I have not reduced one to the other.
6. *Leftovers of the clock.* Part of the old clock (5–35%) survives a new maximum, and a runner-up that arrives late does not shrink the radius. Their consequences for errors were not measured.
7. *Negative values.* The network sums inputs below threshold weakly instead of storing them. How it reaches 23% (twice chance) with one positive input, and why it prefers position 0 then, is not explained.

**What limits the evidence.**

8. *No accuracy estimate from the weights.* ARC's standard for an explanation is a deductive estimate of the 95.3%. This report classifies the errors and attributes about a quarter of them by intervention. It does not predict the error rate.
9. *Input distribution.* Almost everything is on $\mathcal N(0,1)$. The network was trained on $\mathrm{Unif}(-1,1)$, where its accuracy is 0.938. Only accuracy, the two pulse repairs and the shift experiment were repeated on uniform inputs.
10. *Interventions leave the data manifold.* Editing a mode coordinate gives states with negative activations (20–90% of edited states, depending on the edit). A few edits were repeated with the state clipped at zero; most were not.
11. *Gauge.* "How much the readout sees a mode" and "what fraction of a left vector sits on the comparators" depend on how the units are scaled. The cancellation of the clock's two readout halves, the phase laws and all interventions do not.
12. *One network.* Outside analyses report that other seeds of this architecture use different solutions. I did not check whether the ersatz-bias structure appears in the retrained replicas in `replica/`.

## 11. Earlier work, and what is new here

I read the colleague's three write-ups in `reverse-engineered-networks/`, ARC's own material and the outside analyses listed in `hide/notes/sources_arc.md` and `hide/notes/sources_thirdparty.md`.

Already known, and confirmed here:

- Zihao Chen (ARC, 2024): units 2, 4, 7 as running maximum, increment and drawdown; the logits shift one position per step and the intermediate logits give the running answer; the readout without units 1, 6, 7, 8 finds the maximum; on a new maximum "the old max logit gets a boost"; an algorithm with two score vectors $m$, $s$ and $s \leftarrow m$ on a new maximum. Chen's stated obstacle was how the four "very entangled" negative-input units represent $s - m$.
- Jacob Hilton (ARC, 2026): formulas for units 2, 4, 6, 7, 1.
- N. Goldowsky-Dill (2026): comparator thresholds as $a(\delta) M + b(\delta) S^+$ with $\delta$ the age of the maximum; the input cancels in the block {1, 6, 7, 8, 2}.
- The colleague's write-ups: unit roles; the eigenvalues of the resting map; unit 5 as recorder with a cut-off near 0.7 of the runner-up; the three-step sequence after a new maximum; near-tie errors; the false firings of unit 4.
- Outside repositories of October 2026 (DraggonZ, J. Snider, Skyway1111, A. Panahi, D. Khachaturov, M. Keane): "hiding neurons" that cancel the vote of the current largest and are switched off by a larger number; the 43° rotation inside the comparator block; unit 5 as a write pointer; the state as $S\,U[p_S] + (M - S)\,V[p_M]$; a design argument that a bias-free ReLU network must put a wave on a positive level; a false new maximum at $z = 511/512$.

New here, as far as I can tell from those sources:

1. The representation of $s - m$: the comparators' vote is $-60\,(M - 0.87 S)\cos(42.75°(p - p_M))$, traced to the sinusoidal readout columns of the four comparators and the rotating thresholds. This answers Chen's question.
2. The threshold recurrence with $B_c\mathbf 1 \approx 0$, which shows in the weights why the centre has to be supplied from outside the comparators and why the radius is the increment. The six-unit simulation.
3. The reading of the clock as a correction held in reserve, with the interventions that support it.
4. The unit-5 pulse after a maximum as preparation for that maximum's later demotion, and unit 5's write as the mirror image of unit 4's kick on the tape.
5. The runner mode as the threshold of unit 5, and the absence of a second clock.
6. The shift matrix $\Sigma$ and the laws of the scores: slope 61 by three routes, and the tent at the newest position with edges at the stored runner-up and maximum.
7. Errors at separations 7 and 8 as a consequence of the single-harmonic deposit; the position-determined offsets of the comparison; the finding that an exact detector of new maxima makes the network worse.
8. The assessment of your hypothesis in section 8, including the tangency of the wave units' orbit to the ReLU boundary and the level-removal experiment.

## 12. Files

- `report.md`, `mechanism_summary.png`: this report and its figure.
- `hide/common.py`, `hide/modes.py`: the network, sampling, order statistics, exact traces; the eigenmodes in a fixed gauge.
- `hide/e01`–`e26`: the lead's experiments, one script per result; `hide/fig_mechanism.py` draws the figure.
- `hide/r1_*/`: scripts and raw outputs of the eight sub-investigations; `hide/notes/r1_*.md`: their results; `hide/notes/integration_round1.md`: a digest.
- `hide/v_*/`: scripts of the independent verification; `hide/notes/claims_for_verification.md`: the claims they were given.
- `hide/notes/sources_arc.md`, `sources_thirdparty.md`, `digest_*.md`: what ARC, outside authors and the colleague's write-ups say, with checks of their numbers.
- `hide/notes/lab_notebook.md`: the running record of the lead's results.
