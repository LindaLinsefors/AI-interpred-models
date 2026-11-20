"""Verifier V1, attack E4b.
(1) For entries that are neither max nor runner-up: how much of the variance of d = (s[p]-s[p2])/M is explained
    by the input's own value v = (x_p - S)/M, and how much by WHERE the input sits (t, its age, age of the max,
    age of the runner-up)?   eta^2 = between-cell variance / total variance.
(2) Smearing onto neighbours, restricted to overwritten inputs that arrived after both top-2 inputs (no event
    can have happened at their arrival for any v' < 0).
(3) Off-by-one check of the T3 index convention.
Seeds 5801 (1), 5802 (2), 5803 (3); 200k / 150k / 200k sequences.
Run: python -B v1_attack_value_vs_config.py -> v1_attack_value_vs_config.txt
"""
from vlib import *

P = Tee("v1_attack_value_vs_config.txt")

# ---------------------------------------------------------------- (1)
x = sample(200_000, 5801)
H = run(x)
cols = {k: [] for k in ("d", "v", "age", "t", "a1", "a2")}
for t in range(2, 10):
    sub, _ = scores(H, t)
    p, val = top3(x, t)
    p1, p2, M, S = p[:, 0], p[:, 1], val[:, 0], val[:, 1]
    keep = S > 0.2
    sub, xx, p1, p2, M, S = sub[keep], x[keep, : t + 1], p1[keep], p2[keep], M[keep], S[keep]
    r = np.arange(len(M))
    d = (sub - sub[r, p2][:, None]) / M[:, None]
    v = (xx - S[:, None]) / M[:, None]
    other = np.ones(d.shape, bool); other[r, p1] = False; other[r, p2] = False
    pos = np.broadcast_to(np.arange(t + 1), d.shape)
    cols["d"].append(d[other]); cols["v"].append(v[other]); cols["age"].append((t - pos)[other])
    cols["t"].append(np.full(other.sum(), t))
    cols["a1"].append(np.broadcast_to((t - p1)[:, None], d.shape)[other]); cols["a2"].append(np.broadcast_to((t - p2)[:, None], d.shape)[other])
c = {k: np.concatenate(v) for k, v in cols.items()}


def eta2(d, labels):
    """between-cell variance share for integer labels"""
    _, inv = np.unique(labels, return_inverse=True)
    cnt = np.bincount(inv); sm = np.bincount(inv, d)
    mean = sm / cnt
    return ((d - mean[inv]) ** 2).sum(), cnt.size


def qbin(v, nb):
    e = np.quantile(v, np.linspace(0, 1, nb + 1)[1:-1])
    return np.searchsorted(e, v)


P("=== (1) variance of d for 'other' entries: own value vs configuration (seed 5801, 200k seq, S_t>0.2, t=2..9) ===")
P("    eta^2 = 1 - SS_within/SS_total;  cells: v = 40 quantile bins of v;  config = (t, age, a1, a2);  both = config x 8 quantile bins of v")
for name, m in (("age 0", c["age"] == 0), ("age 1", c["age"] == 1), ("ages>=2", c["age"] >= 2),
                ("ages>=2, v > -0.3", (c["age"] >= 2) & (c["v"] > -0.3)), ("ages>=2, v <= -0.3", (c["age"] >= 2) & (c["v"] <= -0.3)),
                ("ages>=2, t = 9 only", (c["age"] >= 2) & (c["t"] == 9))):
    d, v = c["d"][m], c["v"][m]
    tot = ((d - d.mean()) ** 2).sum()
    cfg = c["t"][m] * 1000 + c["age"][m] * 100 + c["a1"][m] * 10 + c["a2"][m]
    w_v, n_v = eta2(d, qbin(v, 40))
    w_c, n_c = eta2(d, cfg)
    w_a, n_a = eta2(d, c["age"][m])
    w_b, n_b = eta2(d, cfg * 10 + qbin(v, 8))
    P(f"   {name:22s} n={m.sum():8d} sd(d)={d.std():6.2f} | eta2 value {1-w_v/tot:.3f} | age only {1-w_a/tot:.3f} | config {1-w_c/tot:.3f} ({n_c} cells) | value+config {1-w_b/tot:.3f} ({n_b} cells)"
          f" | Spearman(d, v) {np.corrcoef(np.argsort(np.argsort(d)), np.argsort(np.argsort(v)))[0,1]:.3f}")
P("   median d for ages>=2 and v <= -1 by (offset of the entry from the max, p - p1), pooled t (a pure value comparison would be flat):")
m = (c["age"] >= 2) & (c["v"] <= -1)
off1 = c["a1"][m] - c["age"][m]                      # p - p1
off2 = c["a2"][m] - c["age"][m]                      # p - p2
P("      p-p1: " + "  ".join(f"{o:+d}: {np.median(c['d'][m][off1 == o]):6.1f}" for o in range(-7, 8) if o != 0 and (off1 == o).sum() > 500))
P("      p-p2: " + "  ".join(f"{o:+d}: {np.median(c['d'][m][off2 == o]):6.1f}" for o in range(-7, 8) if o != 0 and (off2 == o).sum() > 500))
del H

# ---------------------------------------------------------------- (2)
P("\n=== (2) smearing, overwritten input arrived AFTER both top-2 (seed 5802/5803 rng, 150k seq, read at t=9) ===")
x = sample(150_000, 5802)
rng = np.random.default_rng(5803)
p3, val = top3(x, 9)
p1, p2, M, S = p3[:, 0], p3[:, 1], val[:, 0], val[:, 1]
late_ok = (S > 0.2) & (np.maximum(p1, p2) <= 6)        # at least 3 positions after both
x, p1, p2, M, S = x[late_ok], p1[late_ok], p2[late_ok], M[late_ok], S[late_ok]
n = len(x); r = np.arange(n)
lo = np.maximum(p1, p2) + 1
p = (lo + np.floor(rng.random(n) * (10 - lo))).astype(int)        # uniform in lo..9
outs = {}
for vv in (-2.5, -0.075):
    xm = x.copy(); xm[r, p] = S + vv * M
    s = run(xm)[:, 9] @ U.T
    outs[vv] = (s - s[r, p2][:, None]) / M[:, None]
dB = outs[-0.075] - outs[-2.5]
own = dB[r, p]
by = np.ones((n, 10), bool); by[r, p] = False; by[r, p1] = False; by[r, p2] = False
off = np.arange(10)[None, :] - p[:, None]
for name, m in (("age 0", p == 9), ("age 1", p == 8), ("ages>=2", p <= 7)):
    line = f"   {name:8s} n={m.sum():6d} own change median {np.median(own[m]):7.2f} | bystander median change by offset q-p: "
    for o in (-3, -2, -1, 1, 2, 3):
        sel = by & (off == o) & m[:, None]
        if sel.sum() > 200:
            line += f"{o:+d}: {np.median(dB[sel]):6.2f}  "
    P(line)
m = p <= 7
sel_n = by & (np.abs(off) == 1) & m[:, None]
sel_f = by & (np.abs(off) >= 2) & m[:, None]
P(f"   ages>=2: neighbours (|q-p|=1) median change {np.median(dB[sel_n]):.2f} = {np.median(dB[sel_n])/np.median(own[m]):.2f} of the own change; "
  f"farther entries median {np.median(dB[sel_f]):.2f}, median |.| {np.median(np.abs(dB[sel_f])):.2f}")
P(f"   the max's entry: median change {np.median(dB[r, p1][m]):.2f}")

# ---------------------------------------------------------------- (3)
P("\n=== (3) T3 index convention: accuracy given S_t>0 with position = i + t - 9 + k (seed 5804, 200k) ===")
x = sample(200_000, 5804)
H = run(x)
for t in (2, 4, 6, 8):
    full = H[:, t] @ U.T
    p, val = top3(x, t)
    pos = val[:, 1] > 0
    row = []
    for k in (-1, 0, 1):
        lo_i = 9 - t - k                                 # valid indices so that positions 0..t map into range
        idxs = np.arange(max(lo_i, 0), min(lo_i + t + 1, 10))
        g = idxs[full[:, idxs].argmax(1)] + t - 9 + k
        row.append(f"k={k:+d}: {np.mean(g[pos] == p[pos, 1]):.4f}")
    g0 = full[:, 9 - t:].argmax(1)
    row.append(f"| P(argmax = max position) {np.mean(g0[pos] == p[pos, 0]):.4f}  P(= third) {np.mean(g0[pos] == p[pos, 2]):.4f}")
    P(f"   t={t}: " + "  ".join(row))
