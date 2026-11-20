# PCA of the rows of $W_{out}$

`w_out_pca.png` (script `hide/w-out-analasys/w_out_pca.py`, runs in about 2 s). Row $i$ of $W_{out}$ ($10 \times 16$) is the readout of output position $i$, so it is a point in unit space. Each figure row is one of eight versions: centred (mean row subtracted) or not, all 16 units or only the comparators 1, 6, 7, 8, and positions 0–9 or 0–7. Panels plot PC$k$ against PC0 for every PC with nonzero variance. Uncentred fractions are of the second moment.

Variance fractions:

| version | PC0 | PC1 | PC2 | PC3 | rest |
|---|---|---|---|---|---|
| uncentred, all units, pos 0–9 | 41.8% | 36.4% | 8.5% | 4.3% | 9.0% |
| uncentred, all units, pos 0–7 | 47.9% | 38.2% | 6.4% | 3.1% | 4.4% |
| uncentred, units 1,6,7,8, pos 0–9 | 50.8% | 40.8% | 7.4% | 1.0% | |
| uncentred, units 1,6,7,8, pos 0–7 | 55.5% | 42.4% | 2.1% | 0.0% | |
| centred, all units, pos 0–9 | 41.8% | 36.4% | 8.5% | 4.3% | 9.0% |
| centred, all units, pos 0–7 | 48.9% | 38.5% | 6.4% | 3.1% | 3.2% |
| centred, units 1,6,7,8, pos 0–9 | 50.8% | 40.8% | 7.4% | 1.0% | |
| centred, units 1,6,7,8, pos 0–7 | 56.4% | 42.8% | 0.8% | 0.0% | |

Observations:
- The rows of $W_{out}$ nearly sum to zero (largest $|$column mean$|$ 0.11, against entries up to 7.3), so centring changes almost nothing when all ten positions are used.
- PC0–PC1 hold 78–99% of the variance. In that plane, positions 0–7 lie on a loop in order, like a clock. Positions 8 and 9 fall inside the loop near 7 instead of continuing it.
- On units 1, 6, 7, 8 with positions 0–7, the rows are almost exactly planar: PC0 + PC1 hold 98–99%.

# Ablating columns of $W_{out}$

`w_out_ablate.png` (script `hide/w-out-analasys/w_out_ablate.py`, about 1 min). Column $u$ of $W_{out}$ is the readout of unit $u$; zeroing it removes $h_9[u]$ from every logit. Accuracy is that of the second argmax on $2^{20}$ samples, split by the target $y$ (the position of the second max). The figure shows the drop in percentage points (pp) for Unif(−1, 1) inputs. Gaussian inputs give the same pattern; the numbers below are uniform / Gaussian. "Removable" means a greedy search (remove, one at a time, the column that costs least) keeps the total drop below 0.1 pp.

**All data: no column can be removed.** The cheapest is unit 9 (0.8 / 0.55 pp), then unit 4 (4.3 / 4.0 pp) and unit 5 (8.9 / 9.5 pp). Every comparator and wave-unit column costs 20–70 pp.

**Targets 0–7 only ($y \le 7$): column 5 is free, and column 9 almost is.**
- Unit 5 is needed only for $y = 8$: without it, accuracy at $y = 8$ falls from 0.925 to 0.000, and at every other target it changes by at most 1.3 pp. On $y \le 7$ the drop is −0.2 / +0.1 pp.
- Unit 9 alone costs 1.2 / 0.8 pp on $y \le 7$ (mostly at $y = 4$–7), but once column 5 is gone it costs only 0.1 pp more. Removing both gives −0.2 / +0.1 pp on $y \le 7$. Greedy removes $\{5, 9\}$ for uniform inputs and $\{5\}$ for Gaussian, where 9 then costs 0.12 pp, just over the threshold.
- Unit 4 is still needed (4.6 / 4.0 pp, up to 10 pp at $y = 6$–8). Without it, $y = 9$ gets slightly *better*.

Zero ablation is the right test here: mean ablation of column 5 is worse (it costs 5.5 / 11 pp on $y \le 7$), because adding the mean of $h_9[5]$ acts as a bias.

# Hessian with respect to column scales

Script `hide/w-out-analasys/w_out_hessian.py` (about 1 min). Uniform inputs and $2^{20}$ samples unless stated. Every quantity is given for all data and for $y \le 7$ (second-argmax position 8 or 9 left out).

## Setup

Each column of $W_{out}$ gets a scale $s_u$:

$$W_{out}^{mod} = W_{out}\,\mathrm{diag}(s), \qquad s = \mathbf{1} + v_{mod}.$$

So $s = \mathbf{1}$ is the trained network, and $s_u = 0$ ablates column $u$. The logits are linear in $s$:

$$\ell = \sum_u s_u\, c_u = C s, \qquad c_u = W_{out}[:, u]\; h_9[u], \qquad C = [c_0, \dots, c_{15}] \in \mathbb{R}^{10 \times 16}.$$

With $p = \mathrm{softmax}(\ell)$ and loss $\mathcal{L}(s) = \mathbb{E}[-\log p_y]$, the gradient and Hessian are exact:

$$g = \mathbb{E}\left[C^\top (p - e_y)\right], \qquad H = \mathbb{E}\left[C^\top \left(\mathrm{diag}(p) - p\,p^\top\right) C\right].$$

## Spectrum at $s = \mathbf{1}$

| | loss | flat eigenvalues | remaining eigenvalues |
|---|---|---|---|
| uniform, all | 0.187 | 0.015, 0.022, 0.069 | 1.35, 2.44, 2.80, 3.84, 5.28, 5.71, 7.15, 8.16, 15.6, 20.3, 74.4, 361, 463 |
| uniform, $y \le 7$ | 0.174 | 0.014, 0.023, 0.053 | 1.33, 1.50, 2.88, 3.07, 5.35, 5.94, 7.29, 8.27, 14.7, 20.7, 58.1, 408, 516 |
| Gaussian, all | 0.146 | 0.008, 0.024, 0.085 | 1.72, 2.91, 4.16, 5.92, 8.59, 9.78, 15.1, 20.0, 25.4, 43.5, 99.1, 473, 600 |
| Gaussian, $y \le 7$ | 0.138 | 0.008, 0.023, 0.072 | 1.50, 1.81, 3.70, 6.00, 8.75, 10.4, 16.2, 20.0, 26.0, 43.2, 83.6, 532, 660 |

Three eigenvalues are flat, at least 20× below the rest, in every case.

## The flat eigenvectors (uniform)

| eigenvalue (all / $y \le 7$) | weight on unit 9 | weight on unit 4 | every other unit | reading |
|---|---|---|---|---|
| 0.015 / 0.014 | 0.86 / 0.89 | 0.15 / 0.12 | ≈ 0.13 / 0.11 | $\mathbf{1}$ + unit 9 |
| 0.022 / 0.023 | 0.51 / 0.45 | −0.30 / −0.33 | ≈ −0.22 | unit 9 − $\mathbf{1}$ |
| 0.069 / 0.053 | 0.03 / 0.04 | 0.94 / 0.94 | ≈ −0.09 | unit 4 |

So the flat subspace is about

$$\mathrm{span}\{\mathbf{1},\, e_9,\, e_4\}.$$

Here $\mathbf{1}$ scales all logits together. Unit 5 is not in it, in either case.

## Per-unit curvature and gradient (uniform)

| unit | group | $H_{uu}$, all | $H_{uu}$, $y \le 7$ | $g_u$, all | $g_u$, $y \le 7$ |
|---|---|---|---|---|---|
| 1 | comparator | 149.4 | 168.3 | 0.489 | 0.303 |
| 6 | comparator | 149.8 | 161.5 | 0.423 | 0.950 |
| 7 | comparator | 331.0 | 357.0 | −0.434 | −1.353 |
| 8 | comparator | 239.9 | 264.0 | −0.524 | −0.067 |
| 4 | pulse | 0.067 | 0.050 | 0.001 | −0.012 |
| 5 | pulse | **5.19** | **2.30** | **−0.019** | **+0.181** |
| 9 | pulse | 0.022 | 0.021 | −0.001 | −0.007 |
| 0 | wave | 11.0 | 11.6 | −0.042 | −0.049 |
| 2 | wave | 16.9 | 17.8 | −0.160 | −0.000 |
| 3 | wave | 3.36 | 3.33 | 0.042 | −0.066 |
| 10 | wave | 10.2 | 10.6 | −0.054 | 0.028 |
| 11 | wave | 4.94 | 5.37 | 0.055 | 0.052 |
| 12 | wave | 15.3 | 15.6 | 0.146 | 0.057 |
| 13 | wave | 4.92 | 5.43 | −0.031 | −0.034 |
| 14 | wave | 16.8 | 17.7 | −0.081 | 0.023 |
| 15 | wave | 13.0 | 12.9 | 0.134 | −0.071 |

## What changes when $y \in \{8, 9\}$ is left out

- **Unit 5** changes most. Its curvature halves, $H_{55}$: 5.19 → 2.30, and its gradient changes sign, $g_5$: −0.019 → +0.181. To second order, ablating column 5 ($\Delta s = -e_5$) changes the loss by
  $$\Delta \mathcal{L} \approx -g_5 + \tfrac12 H_{55} = \begin{cases} 0.019 + 2.60 = 2.61 & \text{all} \ -0.181 + 1.15 = 0.97 & y \le 7. \end{cases}$$
  The quadratic estimate is still positive on $y \le 7$, but the ablation found accuracy unchanged there. The full step $s_5: 1 \to 0$ is far outside the quadratic regime. In the eigenbasis, unit 5 moves from the λ = 2.44 and 3.84 eigenvectors (weights 0.54, 0.63) to mostly the λ = 1.50 eigenvector (weight 0.81).
- **Units 4 and 9** stay flat in both cases. Yet ablating them cost about 4 pp and 1 pp, so here too the local picture underestimates the full step.
- **The comparator gradients** shift. $g_7$ grows from −0.43 to −1.35 and $g_6$ from 0.42 to 0.95, while $g_8$ nearly vanishes (−0.52 → −0.07).
- **The trained network is not stationary in $s$:** $|g| \approx 1.0$ (all) and $1.7$ ($y \le 7$), almost all on the comparators.

## At $s = 0$ ($W_{out}^{mod} = 0$)

All logits are 0, $p$ is uniform and $\mathcal{L} = \ln 10$. No direction is flat:

| | smallest eigenvalue | largest eigenvalue |
|---|---|---|
| uniform, all | 0.156 | 4280 |
| uniform, $y \le 7$ | 0.152 | 4753 |
| Gaussian, all | 0.616 | 14445 |
| Gaussian, $y \le 7$ | 0.621 | 15934 |
