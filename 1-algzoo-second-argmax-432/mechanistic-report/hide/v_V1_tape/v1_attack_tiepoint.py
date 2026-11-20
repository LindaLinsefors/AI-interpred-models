"""Verifier V1: precision of the comparison "newest input against the runner-up" (follow-up of E3e).
Fixed history h_{t-1} (N(0,1), seed 6301, 80k sequences, one random t in 2..9 each, S_{t-1} > 0.2).
x_t = S + v M is swept over v in [-0.3, +0.3] but never above the max (states with (M-S)/M > 0.32 only, so the
whole sweep stays a 'no new max' input).  Tie point v*: where s_t[9] = s_t[index of the old runner-up].
Run: python -B v1_attack_tiepoint.py -> v1_attack_tiepoint.txt
"""
from vlib import *

P = Tee("v1_attack_tiepoint.txt")
x = sample(80_000, 6301)
tt = np.random.default_rng(6302).integers(2, 10, len(x))
H = run(x)
grid = np.linspace(-0.3, 0.3, 241)
ties, cls, ts, a2s, gaps = [], [], [], [], []
for t in range(2, 10):
    sel = np.where(tt == t)[0]
    p, val = top3(x[sel], t - 1)
    p2, M, S = p[:, 1], val[:, 0], val[:, 1]
    ok = (S > 0.2) & ((M - S) / M > 0.32)
    sel, p2, M, S = sel[ok], p2[ok], M[ok], S[ok]
    h0 = H[sel, t - 1]; idx = 9 - t + p2; r = np.arange(len(sel))
    D = np.empty((len(sel), len(grid)))
    for j, vv in enumerate(grid):
        h = np.maximum((S + vv * M)[:, None] * w + h0 @ W.T, 0)
        s = h @ U.T
        D[:, j] = (s[:, 9] - s[r, idx]) / M
    pos = D > 0
    first = pos.argmax(1)
    c = np.where(~pos.any(1), 2, np.where(first == 0, 0, 1))        # 0: already positive at -0.3; 1: crossing inside; 2: never positive
    tie = np.full(len(sel), np.nan)
    rr = np.where(c == 1)[0]; i1 = first[rr]
    y0, y1 = D[rr, i1 - 1], D[rr, i1]
    tie[rr] = grid[i1 - 1] + (grid[i1] - grid[i1 - 1]) * (-y0) / (y1 - y0)
    ties.append(tie); cls.append(c); ts.append(np.full(len(sel), t)); a2s.append(t - p2); gaps.append((M - S) / M)
ties, cls, ts, a2s, gaps = map(np.concatenate, (ties, cls, ts, a2s, gaps))
P(f"states: {len(ties)}  (S > 0.2 and (M-S)/M > 0.32)")
P("share: tie below -0.3: %.4f | tie inside [-0.3, 0.3]: %.4f | newest input still below the old runner-up's score at v = +0.3: %.4f" % tuple(np.mean(cls == k) for k in (0, 1, 2)))
ok = cls == 1
P("tie point v* (units of M; 0 = exact comparison with the runner-up): median %+.4f  mean %+.4f  sd %.4f  pct 1,5,25,75,95,99: %s"
  % (np.median(ties[ok]), ties[ok].mean(), ties[ok].std(), np.round(np.percentile(ties[ok], [1, 5, 25, 75, 95, 99]), 4)))
P("   share with |v*| < 0.01: %.3f ; < 0.02: %.3f ; < 0.05: %.3f ; < 0.1: %.3f" % tuple(np.mean(np.abs(ties[ok]) < q) for q in (0.01, 0.02, 0.05, 0.1)))
P("   by t (median / sd): " + "  ".join(f"t{t}: {np.median(ties[ok & (ts == t)]):+.4f}/{ties[ok & (ts == t)].std():.3f}" for t in range(2, 10)))
P("   by age of the runner-up at t (median / sd): " + "  ".join(f"a{a}: {np.median(ties[ok & (a2s == a)]):+.4f}/{ties[ok & (a2s == a)].std():.3f}" for a in range(1, 10)))
P("   in units of S instead of M: median %+.4f, sd %.4f" % (np.median(ties[ok] / (1 - gaps[ok])), (ties[ok] / (1 - gaps[ok])).std()))
