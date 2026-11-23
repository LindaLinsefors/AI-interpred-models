I computed it (`hide/e32_tape_transform.py`). The transform is a damped Fourier series in the age, with the modes' own eigenvalues as the frequencies and decay rates. It is exact in one form and approximate in the clean form.

**Exact form.** With $o$ the 13 non-pulse units, $r_m$ the mode patterns and $z_m$ the mode coordinates,

$$s_t[i] = \sum_{\text{real}} \zeta_m\, p_m[i] + \sum_{\text{pairs}} 2\,\mathrm{Re}\big(z_m\, p_m[i]\big) + (\text{pulse units' terms}), \qquad p_m = U_o\, r_m .$$

So $p_m[i]$ is mode $m$'s readout pattern over the logit index $i = 9 - a$.

**Clean form.** A resting step multiplies $z_m$ by $\lambda_m$. For that to move every entry one index down, the pattern must be $p_m[i] = c_m\,\lambda_m^{\,i}$. Substituting $i = 9 - a$:

$$f_t(a) + \text{const} \approx \sum_m 2\,\mathrm{Re}\big( C_m\, \lambda_m^{-a}\, z_m(t) \big), \qquad C_m = c_m\,\lambda_m^{9},$$

with a single real term $C\,\lambda^{-a}\,\zeta$ for the alternating mode. The factor $\lvert\lambda_m\rvert^{-a}$ grows with age and exactly undoes the decay of $z_m$, so an entry keeps its size as it ages.

**The numbers** (fit on indices 0–7, i.e. ages 2–9):

| mode | $\lambda_m$ | $c_m$ | share of $p_m$ explained by $c_m\,\lambda_m^{\,i}$ | measured phase step per index |
|---|---|---|---|---|
| short clock | $0.735 \angle 43.0°$ | $7.16 \angle 172°$ | 0.94 | 40°–57° for $i \le 4$ |
| second harmonic | $0.937 \angle 83.3°$ | $1.41 \angle {-164°}$ | 0.86 | 78°–93° for $i \le 3$ |
| third harmonic | $0.837 \angle 124.8°$ | $1.77 \angle {-134°}$ | 0.87 | 124°–150° for $i \le 5$ |
| alternating | $-0.928$ | $-0.59$ | 0.89 | sign alternates throughout |

Units: logits per unit of mode coordinate, where a pulse of unit 4 of size $h$ adds $h$ to every coordinate.

**The other direction.** A written score of size $d$ at age $a_e$ has put $\kappa_m\, d\, \lambda_m^{\,a_e}$ into each $z_m$, where $\kappa_m$ is how strongly the writing event (unit 5's pulse) feeds mode $m$. Inserting this above gives a convolution:

$$f_t(a) \approx \sum_e d_e\, K(a_e - a), \qquad K(n) = \sum_m 2\,\mathrm{Re}\big( C_m\, \kappa_m\, \lambda_m^{\,n} \big) .$$

For the tape to work, $K$ must be a peak at $n = 0$. I have not computed the $\kappa_m$ or $K$; that is a few lines if you want it.

**Limits.**

- The clean form misses 6–14% of each pattern on indices 0–7 and fails on indices 8 and 9.
- It is a resting-step statement, and only 10% of states are in the resting gate pattern.
- This is not an orthogonal Fourier basis: four frequencies with different decay rates, on 8 positions.
- I have not checked on data how much of $f_t(a)$ these four modes reproduce without the others.
