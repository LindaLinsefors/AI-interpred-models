"""V5. What one unit of unit 5's pulse puts into the tape scores (exact kernel), and where the tape's part of unit 5's
drive comes from.   Run:  python -B v5_pulse5_kernel.py > v5_pulse5_kernel.txt"""
from vcommon import *

# the pulse h_(j+1)[5] answers x_j and is written at step j+2; n steps later x_j is n+2 steps old, score index 7-n
S = G[:, 2] @ Uo.T                                           # (10 lags, 10 score entries) per unit of pulse 5
print("tape score per unit of unit 5's pulse, at the entry of the input x_j it answers, when x_j is 2..9 steps old: " + " ".join(f"{S[n, 7 - n]:.2f}" for n in range(8)))
print("the same at the entry of the input one step older than x_j (3..9 steps old):                              " + " ".join(f"{S[n, 6 - n]:.2f}" for n in range(7)))
print("the same at the entry of the input one step younger than x_j (2..9 steps old):                            " + " ".join(f"{S[n, 8 - n]:.2f}" for n in range(1, 9)))

x = sample(200_000, 7501); H, Z, Amp = amplitudes(x)
vals, ages = prefix_before(x)
# tape's part of unit 5's drive at step s (s = 1..9), against named quantities of x_0..x_(s-1), in units of the running maximum
dt5 = (H[:, :-1][:, :, ORD] @ P.T) @ W[5, ORD]
ok = vals[0][:, 1:] > 0.3
Mr = vals[0][:, 1:][ok]; y = dt5[ok] / Mr
F = np.stack([vals[1][:, 1:][ok] / Mr, vals[2][:, 1:][ok] / Mr, x[:, :-1][ok] / Mr, np.ones(ok.sum())], 1)
cfg = ages[0][:, 1:][ok] * 10 + ages[1][:, 1:][ok]
res = np.zeros_like(y)
for c in np.unique(cfg):
    r = cfg == c
    res[r] = y[r] - F[r] @ np.linalg.lstsq(F[r], y[r], rcond=None)[0]
print(f"\ntape's part of unit 5's drive / running maximum, steps with running maximum > 0.3 (n = {ok.sum()}): std {y.std():.2f}; "
      f"left after a fit on (S/M, T/M, latest input/M, 1) per configuration of the ages of maximum and runner-up: std {res.std():.2f}")
print(f"for scale: unit 5's pre-activation / running maximum where it fires: std {(Z[:, 1:, 5][ok] / Mr)[Z[:, 1:, 5][ok] > 0].std():.2f}")
