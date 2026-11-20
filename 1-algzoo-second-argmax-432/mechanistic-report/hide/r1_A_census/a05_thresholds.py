"""A5: verify F2 exactly and measure where each threshold sits between S and M.

Same-step thresholds (h_{t-1} given):  theta_j(t) = -W[j].h_{t-1} / w_j   for j in 1, 6, 7, 8, 4.
Delayed thresholds of units 5 and 9 on the PREVIOUS input: unit K is on at t+1 iff x_t > theta_K(t), where theta_K(t)
is the root in x_t of  z_{t+1}[K] = W[K].ReLU(w x_t + W h_{t-1}) + w_K x_{t+1}  (found by bisection; x_{t+1} at its actual value).
Position:  u = (theta - S) / (M - S),  M, S = largest and second largest of x_0..x_{t-1}.
Selection for the position statistics: M - S > 0.2 and S > 0.   Age a = age of the max in h_{t-1} = (t-1) - argmax.
"""
from cc import *

UN = [8, 6, 1, 7, 4, 5, 9]
mism = {j: 0 for j in UN}; ntest = {j: 0 for j in UN}
nonmono = {5: 0, 9: 0}; nmono = 0
pos = {j: [] for j in UN}          # (u, a1, a2, t)
meta = []
exact_ties = 0
fit = {j: [] for j in UN}


def delayed_theta(K, hprev, xnext, iters=46):
    """Root in x of z_next[K](x) = W[K].ReLU(w x + W hprev) + w_K xnext, by bisection on [-60, 60]."""
    base = hprev @ W.T                                   # (n, 16)
    f = lambda xv: np.maximum(xv[:, None] * w + base, 0) @ W[K] + w[K] * xnext
    lo, hi = np.full(len(base), -60.0), np.full(len(base), 60.0)
    flo, fhi = f(lo), f(hi)
    ok = (flo < 0) & (fhi > 0)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        up = f(mid) > 0
        hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
    return 0.5 * (lo + hi), ok


for seed, x, H, Z in batches():
    ev = events(x); G = Z > 0
    B = len(x)
    Hprev = np.concatenate([np.zeros((B, 1, N)), H[:, :-1]], 1)          # h_{t-1}, zero at t = 0
    th = {}
    for j in COMP + [4]:
        th[j] = -(Hprev @ W[j]) / w[j]                                      # (B, T); theta(0) = 0
        pred = (x < th[j]) if j in COMP else (x > th[j])
        mism[j] += (pred != G[:, :, j]).sum(); ntest[j] += pred.size
        exact_ties += (x == th[j]).sum()
    # delayed thresholds, defined for t = 0..8 (read out at t+1)
    for K in (5, 9):
        thK = np.full((B, T), np.nan)
        for t in range(T - 1):
            thK[:, t], ok = delayed_theta(K, Hprev[:, t], x[:, t + 1])
            thK[~ok, t] = np.nan
        th[K] = thK
        valid = ~np.isnan(thK[:, :-1])
        pred = x[:, :-1] > thK[:, :-1]
        mism[K] += (pred != G[:, 1:, K])[valid].sum(); ntest[K] += valid.sum()
        nonmono[K] += (~valid).sum()
    # positions
    Mp, Sp = ev["Mp"], ev["Sp"]
    a1p = np.concatenate([np.full((B, 1), -1), ev["a1"][:, :-1]], 1)    # age of the max in h_{t-1}
    a2p = np.concatenate([np.full((B, 1), -1), ev["a2"][:, :-1]], 1)
    sel = (Sp > 0) & (Mp - Sp > 0.2)
    sel[:, 0] = False
    tt = np.broadcast_to(np.arange(T), (B, T))
    for j in UN:
        s = sel & ~np.isnan(th[j])
        if j in (5, 9):
            s[:, -1] = False
        u = (th[j][s] - Sp[s]) / (Mp[s] - Sp[s])
        pos[j].append(np.stack([u, a1p[s], a2p[s], tt[s], th[j][s], Mp[s], Sp[s]], 1))

print("## A5.1 exact check of the threshold form of the gates (2^20 sequences)\n")
rows = [[j, "x_t < theta_j(t)" if j in COMP else "x_t > theta_4(t)" if j == 4 else f"x_(t-1) > theta_{j}(t-1) (bisection)",
         ntest[j], mism[j], nonmono.get(j, "-")] for j in UN]
print(md_table(["unit", "gate on iff", "steps tested", "mismatches", "steps without a sign change on [-60, 60]"], rows))
print("\nexact ties x_t == theta:", exact_ties)

pos = {j: np.concatenate(v) for j, v in pos.items()}
q = lambda a: " / ".join(f"{v:+.3f}" for v in np.percentile(a, [25, 50, 75])) if len(a) else "-"
print("\n## A5.2 threshold position u = (theta - S)/(M - S): quartiles 25% / 50% / 75%, by age of the max in h_(t-1)\n")
print("selection: S > 0 and M - S > 0.2 (prefix before the tested input); n =", len(pos[8]), "steps for the same-step thresholds\n")
rows = []
for a in range(0, 9):
    rows.append([a, a + 1, int((pos[8][:, 1] == a).sum())] + [q(pos[j][pos[j][:, 1] == a, 0]) for j in UN])
rows.append(["all", "", len(pos[8])] + [q(pos[j][:, 0]) for j in UN])
print(md_table(["age of max in h_(t-1)", "steps from max to tested input", "n"] + [f"unit {j}" for j in UN], rows))

print("\n## A5.3 the same, split by whether the runner-up arrived before or after the max\n")
for label, cmp in (("runner-up older than the max (a2 > a1)", np.greater), ("runner-up newer than the max (a2 < a1)", np.less)):
    print(f"\n{label}\n")
    rows = []
    for a in range(0, 9):
        m8 = (pos[8][:, 1] == a) & cmp(pos[8][:, 2], pos[8][:, 1])
        if m8.sum() < 500:
            continue
        rows.append([a, int(m8.sum())] + [q(pos[j][(pos[j][:, 1] == a) & cmp(pos[j][:, 2], pos[j][:, 1]), 0]) for j in UN])
    print(md_table(["age of max in h_(t-1)", "n"] + [f"unit {j}" for j in UN], rows))

import a05_tables                      # tables A5.4 - A5.7
a05_tables.main(pos)
