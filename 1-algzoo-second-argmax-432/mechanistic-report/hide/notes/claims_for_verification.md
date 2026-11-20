# Claims to be verified independently (lead's experiments e10–e25)

Conventions. $h_t = \mathrm{ReLU}(w x_t + W h_{t-1})$, logits $= U h_9$. COMP = [1, 6, 7, 8], PULSE = [4, 5, 9], ORD = the 13 other units, WAVE = [0, 2, 3, 10, 11, 12, 13, 14, 15]. $M_t \ge S_t \ge T_t$: the three largest of $x_0..x_t$ (window includes $x_t$), positions $p_1, p_2, p_3$, ages $a_k = t - p_k$. Inputs iid N(0,1). "Score" $s_t = U h_t$; index $9 - a$ refers to the input of age $a$ (position $t - a$). $A = W[\mathrm{ORD},\mathrm{ORD}]$, $U_o = U[:,\mathrm{ORD}]$. Mode coordinates are those of `modes.py` (gauge: `LEFT[k] @ W[ORD, 4] == 1`), clock = the eigenpair $0.951\,e^{\pm i 42.75°}$, $\omega$ = 42.75°.

## Bundle V1 — the tape

- **T1.** $\Sigma = U_o A\,\mathrm{pinv}(U_o)$. Frobenius $\lVert U_o A - \Sigma U_o\rVert / \lVert U_o A\rVert = 0.095$. $\Sigma[i,i+1] = 0.97, 0.93, 0.90, 0.89, 1.00, 0.93, 0.93$ for $i = 0..6$; $\Sigma[7,8] = 0.34$; $|\Sigma[8,9]| < 0.05$; all other entries in rows 0–6 have magnitude ≤ 0.42; row 9 entries lie in 1.22–1.62.
- **T2.** For real states at $t = 3, 5, 7$: with $l_0 = U_o h[\mathrm{ORD}]$ and $l_1 = U_o A\,h[\mathrm{ORD}]$, the prediction $\Sigma l_0$ explains $l_1$ with $R^2 = 0.992, 0.991, 0.985$ (positions 0–8 only; for each state the mean over those positions is removed from both; $R^2 = 1 - \mathrm{SS_{res}}/\mathrm{SS_{tot}}$ pooled).
- **T3.** For $t = 1..9$: take $\arg\max$ of $(U h_t)[i]$ over $i \ge 9 - t$ and convert to position $i + t - 9$. It equals the second argmax of $x_0..x_t$ with probability 0.992, 0.987, 0.971, 0.966, 0.969, 0.970, 0.970, 0.967, 0.961 among states with $S_t > 0$, and 0.688, 0.676, 0.777, 0.839, 0.893, 0.927, 0.945, 0.953, 0.953 over all states.
- **T4.** Score laws, states with $S_t > 0.2$, $t = 2..9$, in units of $M_t$: $d = (s[p] - s[p_2])/M_t$ against $v = (x_p - S_t)/M_t$.
  - inputs that are neither max nor runner-up, age 0: median $d \approx 61.6\,v$ for $v \in [-2, -0.05]$ (ratio median $d$ / median $v$ between 55.9 and 61.8 in every bin of that range);
  - age 1: median $d$ = −4.7 for $v \in (-0.10, -0.05]$, −9.7 for $(-0.25, -0.15]$, −12.8 for $(-0.7, -0.5]$, −18.3 for $v < -2$;
  - ages ≥ 2: −1.25 for $v \in (-0.02, 0]$, −5.3 for $(-0.10, -0.05]$, −10.8 for $(-0.25, -0.15]$, −17.7 for $(-0.7, -0.5]$, −23.7 for $v < -2$;
  - the max ($p = p_1$, here $v = (M - S)/M$): age 0: −9.5 for $v \in (0.1, 0.2]$, −25.0 for $(0.35, 0.5]$, −48.8 for $(0.85, 1]$; age 1: −26.4 for $(0.85, 1]$; ages 2..9 and $v > 0.2$: medians between −4.5 and −14.5.
- **T5.** Exact: $U[9,\mathrm{COMP}] * w[\mathrm{COMP}] = (5.01, 7.38, 49.90, -4.04)$, sum 58.25; $U[9,4]\,w_4 = -50.70$; $U[8,5] = 3.398$; $W[5,\mathrm{ORD}]\cdot w[\mathrm{ORD}] = 18.009$; $W[9,\mathrm{ORD}]\cdot w[\mathrm{ORD}] = 84.335$; $|W[u,\mathrm{ORD}]\cdot w[\mathrm{ORD}]| \le 0.432$ for every other unit $u$.

## Bundle V2 — two score vectors

Definitions: $V_t = U[:,\mathrm{COMP}]\,h_t[\mathrm{COMP}]$, $m_t = s_t - V_t$.

- **S1.** Among inputs older than the current one (indices $9-t..8$), given $S_t > 0$ and $p_1 < t$: P($\arg\max m_t$ is the position of the max of $x_0..x_t$) = 0.972, 0.942, 0.948, 0.953, 0.958, 0.964, 0.966, 0.891 for $t = 2..9$.
- **S2.** At $t = 9$, on sequences where all four comparators are on at $t = 9$ and $S_9 > 0.1$, for each age $a_1 = 2..9$ of the max, least squares $V[i] \approx A_i (M - S) + B_i S + C_i x_9$ (no intercept) has $R^2$ between 0.88 and 0.97. As a function of $d = i - (9 - a_1)$, restricted to $i \le 7$ and averaged over $a_1$: $A$ = −62 ($d=0$), −48 ($+1$), −42 ($-1$), −6 ($+2$), +7 ($-2$), +40 ($+3$), +59 ($-3$), +60 ($+4$), +82 ($-4$), +47 ($+5$), +61 ($-5$), +14 ($\pm6$), −25 ($+7$), −33 ($-7$). Compare with $-60\cos(42.75°\,d)$.
- **S3.** $B_i$ has the same dependence on $d$ with amplitude about 8 (minimum near $d = 0$). At index 9: $C_9 = 58.2$, $A_9 \in [-59, -35]$, $B_9 \in [-58.4, -54.9]$.
- **S4.** Readout columns of the comparators over positions 0..7: least squares $U[p,j] \approx a_j\cos(42.75°\,p - \varphi_j) + c_j$: $a$ = 5.81, 5.09, 6.21, 6.08; $\varphi$ = −29°, 26°, 131°, −130°; $R^2$ = 0.974, 0.943, 0.991, 0.988 for $j$ = 1, 6, 7, 8.
- **S5.** At $t = 9$, sequences with all four comparators on, $S > 0.2$, $p_1 < 9$, $p_2 < 9$; $D = |p_1 - p_2|$, gap$/M = (M-S)/M$; medians in units of $M$ of $V[p_2] - V[p_1]$, $m[p_2] - m[p_1]$, $s[p_2] - s[p_1]$: $D=3$: (20.3, −12.8, 7.8) for gap/$M \in (0.05, 0.15]$; (32.8, −24.1, 8.8) for $(0.15, 0.3]$; (49.0, −39.3, 10.2) for $(0.3, 0.5]$; (68.7, −58.7, 10.5) for $(0.5, 0.8]$. $D=4$, $(0.3, 0.5]$: (61.7, −47.7, 13.8). $D=1$, $(0.3, 0.5]$: (11.4, −6.2, 4.8). The median of $60\,(M - 0.87 S)(1 - \cos(42.75° D))/M$ in the same cells is 21.0, 31.5, 45.7, 63.4, 56.2, 7.5. P($s[p_2] > s[p_1]$) ≥ 0.997 in every cell with gap/$M > 0.05$ and between 0.85 and 0.96 in the cells with gap/$M \le 0.05$ ($D = 1..6$).
- **S6.** New max at step $t$ ($x_t > M_{t-1}$, $M_{t-1} > 0.3$, $p_1(t-1) < t - 1$): compare $s_t$ on inputs $0..t-2$ with $m_{t-1}$ on the same inputs (and with $s_{t-1}$): pooled $R^2$ with a common slope after removing each state's mean over those inputs: 0.96, 0.77, 0.77, 0.88, 0.91, 0.89, 0.87 for $t = 3..9$ (for $s_{t-1}$: ≤ 0.29). P(the old max is the largest entry of $s_t$ among inputs $0..t-2$) = 0.996, 0.994, 0.991, 0.990, 0.988, 0.982, 0.964. Without an event ($x_t < S_{t-1}$, $S_{t-1} > 0.1$, all four comparators on at $t$): $s_t$ vs shifted $s_{t-1}$: $R^2$ = 0.67, 0.72, 0.80, 0.89, 0.93, 0.94, 0.96 for $t = 3..9$.

## Bundle V3 — interventions

- **I1 (wipe).** Let $p_M$ be the position of the largest of the ten inputs. Condition: second largest $> 0.2$ and largest − second $> 0.2$. At one step $t_0 = p_M + k \le 9$ set $h_{t_0}[1,6,7,8] = 0$ (after computing $h_{t_0}$), then continue with the real inputs. With $k = 2$: for $t_0 \le 7$, P(answer $= p_M$) = 1.000 and P(answer = true runner-up) = 0.000; for $t_0 = 8$: 0.997 and 0.000; for $t_0 = 9$: 0.000 and 0.109. Pooled over valid $t_0$: P(answer $= p_M$) = 0.868, 0.873, 0.753, 0.816, 0.977, 0.999 for $k = 1..6$. Control (zero units 10, 11, 12, 15 instead): P(answer $= p_M$) ≤ 0.008, P(runner-up) between 0.22 and 0.54.
- **I2 (clock before a record).** Selection: the largest input arrived at step $t_r \in [3, 8]$; $p_0$ = position of the max of $x_0..x_{t_r - 1}$ with age $t_r - 1 - p_0 \ge 2$; $q_0$ = position of the runner-up of that prefix, value $S_0 > 0.2$, and $M_0 - S_0 > 0.1$; the true answer for the whole sequence is $p_0$ and the third largest of the whole sequence is at $q_0$. Intervention on $h_{t_r - 1}$: $z = \texttt{LEFT["clock"]}\cdot h[\mathrm{ORD}]$, replace by $z'$ via $h[\mathrm{ORD}] \mathrel{+}= 2\,\mathrm{Re}((z' - z)\,\texttt{RIGHT["clock"]})$; continue with the real inputs.
  - $z' = f z$: P(answer $= p_0$) = 0.019, 0.098, 0.342, 0.889, 0.990, 0.893, 0.746 for $f$ = 0, 0.25, 0.5, 0.75, 1, 1.5, 2. For $f = 0$: P(answer $= t_r$) = 0.775, P(answer $= q_0$) = 0.164.
  - $z' = z\,e^{i k \omega}$: $k = -1$: P($p_0 + 1$) = 0.474, P($p_0 + 2$) = 0.154, P($p_0$) = 0.093; $k = -2$: P($p_0 + 2$) = 0.321, P($p_0 + 1$) = 0.153, P($p_0$) = 0.002; $k = +1$: P($p_0$) = 0.458, P($p_0 - 1$) = 0.381, P($p_0 - 2$) = 0.077; $k = +2$: P($p_0 - 1$) = 0.298, P($p_0 - 2$) = 0.249, P($p_0$) = 0.071.

## Bundle V4 — the core

With $g_j = -w_j$ for $j \in$ COMP (order 1, 6, 7, 8): $B_c[j,k] = W[j,k]\,g_k/g_j$, $b_2[j] = W[j,2]/g_j$, $b_4[j] = W[j,4]/g_j$, so that exactly $\theta(t+1) = B_c[\theta(t) - x_t\mathbf 1]_+ + b_2 h_2(t) + b_4 h_4(t) + (\text{terms from the other ten units})$ where $\theta_j(t) = (W h_{t-1})_j / g_j$.

- **K0.** That identity holds exactly on data.
- **K1.** $B_c$ rows: (0.409, 1.036, −1.306, −0.140); (−0.130, 0.361, −0.151, −0.092); (−0.081, −0.152, 0.078, 0.156); (0.544, 0.000, −1.170, 0.615). $B_c\mathbf 1 = (-0.001, -0.012, 0.001, -0.011)$. Eigenvalues: $0.9375\,e^{\pm i 42.56°}$ and a pair of modulus 0.043. $b_2 = (0.048, 0.049, 0.051, 0.051)$; $w_4 b_4 = (0.838, -0.044, 1.016, -0.034)$. $W[2,2] = 0.969$, $W[2,4] = 1.731$, $w_2 = 0.015$; $-W[4,2]/w_4 = 0.0485$, $-W[4,4]/w_4 = 0.0970$.
- **K2.** The cosine between the row $W[u,\mathrm{COMP}]$ and $w[\mathrm{COMP}]$ is within ±0.04 of 0 for all 16 units except unit 5 (0.74) and unit 9 (0.69).
- **K3.** On steps $t \le 8$ with max age $\ge 2$ and $M_t > 0.3$, the part of $\theta_j(t+1)/M_t$ that comes from unit 2 has mean 0.827, 0.844, 0.887, 0.881 for $j$ = 1, 6, 7, 8 (and 0.841 for unit 4's threshold); the part from the wave units other than unit 2: 0.094, 0.103, 0.043, 0.113 (0.085); the part from the comparators: mean −0.042, 0.029, −0.022, −0.259 with standard deviation 0.305, 0.068, 0.054, 0.415.
- **K4.** Median of $h_2(t)/(17.3\,M_{t-1})$ by age of the max at $t$ = 1..8 (states with $M_{t-1} > 0.3$): 1.02, 1.02, 1.07, 1.05, 0.98, 0.90, 0.85, 0.86.
- **K5.** Simulating only units [2, 4, 1, 6, 7, 8] ($h' = \mathrm{ReLU}(W_{cc} h + w_c x)$): centred $R^2$ of each unit's activation against the real network over all steps: 0.965, 0.981, 0.932, 0.964, 0.979, 0.877. If in addition the weight from unit 2 into each core unit $j$ is increased by $\sum_{u \in \mathrm{WAVE}\setminus\{2\}} W[j,u]\,r_u / r_2$ with $r$ the level eigenvector of $A$ (values +0.031, −0.128, +0.085, +0.100, +0.109, +0.191 for $j$ = 2, 4, 1, 6, 7, 8): $R^2$ = 0.991, 0.995, 0.989, 0.995, 0.993, 0.991.
- **K6.** $z_c = \ell\cdot\theta(t+1)$, $\ell$ = left eigenvector of $B_c$ for the eigenvalue with angle +42.56°, normalised so that $\ell\cdot(w_4 b_4) = 1$: $|\ell\cdot\mathbf 1| = 0.013$. By age $a$ of the max at $t$ ($a = 1..8$, states with $M > 0.3$): phase of the median of $z_c/M$: 33.5°, 72.0°, 120.5°, 167.3°, 218.6°, 262.8°, 308.8°, 346.2°; regression of $\mathrm{Re}(z_c e^{-i\,\text{phase}})$ on $(M, S^+)$ without intercept: $b_M$ = 0.93, 0.83, 0.75, 0.68, 0.62, 0.62, 0.66, 0.73; $b_S/b_M$ = −0.91, −0.94, −0.96, −0.91, −0.84, −0.80, −0.77, −0.78; uncentred $R^2$ between 0.96 and 0.997.

## Bundle V5 — eigenmodes and what the readout sees

- **M1.** Eigenvalues of $A$: 1.0099, 0.9735, $0.9509\,e^{\pm i 42.75°}$, $0.9370\,e^{\pm i 83.28°}$, −0.9283, $0.8373\,e^{\pm i 124.78°}$, $0.7350\,e^{\pm i 42.99°}$, 0.2310, 0.0010. The eigenvector for 1.0099 has entries of one sign on all 13 units; relative to unit 2: 1.35, 0.51, 1, 0.19, 0.63, 0.79, 0.70, 1.06, 1.14, 1.59, 0.43, 1.23, 0.81 for units 0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15. $|\cos|$ between $w[\mathrm{ORD}]$ and the eigenvector for 0.0010 is 0.998.
- **M2.** Fraction of each left (dual) vector's squared norm on the comparators: level 0.041, runner 0.506, clock 0.966, 83° mode 0.606, −0.928 mode 0.106, 125° mode 0.347, short (0.735) 0.916, 0.231 mode 0.976. For the level, the fraction on unit 2 is 0.85.
- **M3.** Readout of the clock mode per unit coordinate, $2|R|$ with $R = U[p, \cdot]\,\texttt{RIGHT["clock"]}$ restricted to a unit group: comparators: 6.59, 8.40, 8.42, 7.26, 7.10, 7.39, 6.44, 5.56 at $p = 0..7$ with phases −161°, −117°, −85°, −33°, 20°, 65°, 99°, 145°; wave units: 8.18, 9.54, 9.16, 7.74, 7.48, 7.75, 6.79, 5.70 with phases 17°, 61°, 94°, 147°, −159°, −114°, −83°, −40°; all 13: 1.63, 1.18, 0.75, 0.47, 0.38, 0.40, 0.37, 0.47. Ratio of 10-vector norms |total| / (|comparator part| + |wave part|) = 0.058.
- **M4.** Size of $U_o v$ over positions 0–8 after removing their mean, per unit coordinate in the `modes.py` gauge ($v$ = RIGHT real/imaginary parts): level 0.33, runner 0.11, clock 0.91 / 0.75, 83° mode 2.61 / 2.37, alternating 1.40, 125° mode 2.64 / 2.12, short 9.44 / 5.57, 0.231 mode 24.3, echo 0.04.
- **M5.** Per cell $(a_1, a_2)$ with $S > 0.05$: complex least squares $z_{clock}/10.157 \approx \alpha (M - S) + \beta S$. $\alpha$ depends on $a_1$ only (spread over $a_2$ within about ±0.1 in modulus and ±5° in phase): about 1.0@0°, 1.0@37°, 1.0@80°, 0.95@123°, 0.90@166°, 0.87@209°, 0.82@252°, 0.78@294°, 0.7–0.9@338° for $a_1 = 1..9$. $|\beta|$ between 0.02 and 0.37. Uncentred complex $R^2 \ge 0.945$ in every cell with $a_1 \ge 1$.
- **M6.** Level coordinate$/10.157 \approx \alpha (M - S) + \beta S$ with $\alpha \in [0.91, 1.01]$ and $\beta \in [0.86, 1.10]$ for $a_1 \ge 1$; uncentred $R^2 \ge 0.996$.

## Bundle V6 — deposits at a new maximum, and separation

- **D1.** New max at step $t \ge 2$ with $M_0 = M_{t-1} > 0.3$, $S_0 = S_{t-1} > 0.1$; $p_{M0}, p_{S0}$ their positions; age of the old max $g = t - 1 - p_{M0} \ge 1$. After the record, $y = s_t[\text{old max}] - s_t[\text{old runner-up}]$. Least squares $y \approx c_1 (M_0 - S_0) + c_2 M_0$ by separation $\Delta = |p_{M0} - p_{S0}|$ = 1..7: $c_1$ = 17.6, 43.5, 73.8, 88.5, 81.6, 63.2, 41.8; $c_2$ = 0.58, 2.00, 5.03, 5.66, 4.34, 0.43, 2.05; centred $R^2$ = 0.65, 0.72, 0.67, 0.67, 0.73, 0.81, 0.36; P($y > 0$) = 0.960, 0.985, 0.979, 0.989, 0.968, 0.946, 0.893. $c_1 / (1 - \cos(42.75°\Delta))$ = 65, 47, 46, 45, 45, 51, 82.
- **D2.** For such records with the top two unchanged for the next $k$ steps, the same $y$ read at $t + k$: $c_1$ = 40.7, 42.6, 35.3, 32.1, 28.6, 23.8 for $k = 0..5$ (pooled over separations).
- **D3.** New runner-up at $t$ ($S_{t-1} < x_t < M_{t-1}$, $S_{t-1} > 0.2$): $s_t[t] - s_t[\text{old runner-up}] \approx k_1 (x_t - S_{t-1}) + k_2 M$ with $k_1$ = 34.5, 27.3, 29.2, 39.3, 50.5, 56.9, 61.8, 60.4 for $t = 2..9$, $R^2$ 0.56–0.72.
- **D4.** Full sequences; $M > S > T$ the top three at positions $p_M, p_S, p_T$ with $p_M > p_S$, $p_M > p_T$, $T > 0.2$. P(network answers $p_T$) by $D = |p_S - p_T|$ and relative gap $(S - T)/S$:

| $D$ | (0, .03] | (.03, .08] | (.08, .2] | (.2, .5] |
|---|---|---|---|---|
| 1 | 0.183 | 0.066 | 0.029 | 0.005 |
| 2 | 0.147 | 0.038 | 0.010 | 0.003 |
| 3 | 0.189 | 0.029 | 0.003 | 0.000 |
| 4 | 0.245 | 0.020 | 0.001 | 0.000 |
| 5 | 0.276 | 0.028 | 0.002 | 0.000 |
| 6 | 0.338 | 0.137 | 0.023 | 0.002 |
| 7 | 0.371 | 0.171 | 0.125 | 0.066 |
| 8 | 0.447 | 0.346 | 0.123 | 0.004 |

## Bundle V7 — raw unit activations by age of the max, and a controlled sweep

- **O1.** States with $M_t > 0.3$, $a_1 \ge 1$, $a_2 \ge 2$, binned by $g = (M - \max(S,0))/M$ in (0.15, 0.30], (0.45, 0.60], (0.85, 1.0]. Fraction of states in which the unit is exactly 0: unit 0 at $a_1 = 2$: 1.00 in all three bins; unit 11 at $a_1 = 5$: 0.98, 0.99, 0.86; unit 10 at $a_1 = 6$: 0.97, 0.98, 0.98; unit 12 at $a_1 = 4$: 0.28, 0.42, 0.21. Largest median of $h_{12}/M$ over ages 1..9: 15.3 in the first bin, 31.1 in the third. Half-range over ages 1..9 of the median of $h_j/M$ for the comparators, first bin vs third bin: unit 1: 2.3 vs 6.2; unit 8: 2.2 vs 6.6; unit 6: 1.8 vs 5.7; unit 7: 1.2 vs 4.2.
- **O2.** Most negative entry of each wave unit's readout column: unit 0: −2.90 at position 7; unit 10: −3.87 at 3; unit 11: −4.56 at 4; unit 12: −2.79 at 5; unit 14: −5.14 at 0.
- **O3.** Sweep $x = (0.5, -1, 1.5, -1, -1, -1, v, -1, -1, -1)$. Thresholds for step 6: $\theta_1$ = 1.274, $\theta_6$ = 1.673, $\theta_7$ = 1.381, $\theta_8$ = 0.555, $\theta_4$ = 1.366. Rows ($v$; comparators on at $t=6$; $h_6[4]$; $h_7[5]$; clock $|z|/10.157$ at $t = 9$; answer): (0.30; all; 0; 0.08; 0.899; 0), (0.45; all; 0; 2.77; 0.915; 0), (0.55; all; 0; 4.57; 0.926; 6), (0.85; 1, 6, 7; 0; 6.86; 0.770; 6), (1.15; 1, 6, 7; 0; 9.12; 0.607; 6), (1.30; 6, 7; 0; 10.26; 0.517; 6), (1.45; 6 only; 0.85; 11.85; 0.176; 6), (1.55; 6 only; 1.86; 13.03; 0.071; 2), (1.80; none; 4.40; 16.00; 0.404; 2).
