"""A6 extra [INTERVENTION]: which gates does a resampled low input flip, and when?

Condition 'L: x<.5S' of a06_resample.py (S_{t-1} > 0, x_t < 0.5 S_{t-1}, redrawn from N(0,1) | x < 0.5 S_{t-1}), t = 2..8.
For every unit: share of resamples in which its gate differs at step t (lag 0), t+1 (lag 1), t+2 (lag 2), any step >= t+3.
Same resampling seeds as a06_resample.py would give different draws (other conditions are skipped), so seeds 4000 + batch."""
from cc import *
from a06_resample import trunc_normal

cnt = np.zeros((4, N)); nres = np.zeros(4)
for bi, (seed, x, H, Z) in enumerate(batches()):
    rng = np.random.default_rng(4000 + bi)
    ev = events(x); G0 = Z > 0
    for t in range(2, T - 1):
        S = ev["Sp"][:, t]
        idx = np.flatnonzero((S > 0) & (x[:, t] < 0.5 * S))
        xn = x[idx].copy()
        xn[:, t] = trunc_normal(np.full(len(idx), -np.inf), 0.5 * S[idx], rng)
        D = (run(xn, return_pre=True)[1] > 0) != G0[idx]            # (n, T, 16)
        for lag in range(3):
            if t + lag < T:
                cnt[lag] += D[:, t + lag].sum(0); nres[lag] += len(idx)
        if t + 3 < T:
            cnt[3] += D[:, t + 3:].any(1).sum(0); nres[3] += len(idx)
rows = [[lab, int(nres[k])] + [f"{cnt[k, i] / nres[k]:.4f}" for i in range(N)]
        for k, lab in enumerate(["step t (lag 0)", "step t+1", "step t+2", "any step >= t+3"])]
print("## share of resampled low inputs (x_t < S/2, t = 2..8) after which the gate of each unit differs\n")
print(md_table(["at", "n"] + [f"u{i}" for i in range(N)], rows))
