"""V6 causal test of 'the promotion is a first-harmonic (clock-mode) bump'.

D1 records (g >= 1). On h_{t-1}, scale one eigenmode coordinate of the resting map by f
(h[ORD] += 2 Re((f - 1) z RIGHT) for a complex pair, h[ORD] += (f - 1) m RIGHT for a real mode),
recompute h_t = ReLU(w x_t + W h_{t-1}) and y = s_t[old max] - s_t[old runner-up].
If the lead is carried by the 42.75-degree clock mode, c1 should scale like f for that mode and be
insensitive to the others.  Seeds 5200..5203, 400k sequences each.
"""
from v6_data import *

out = open(HERE / "a6_clock_scaling.txt", "w", encoding="utf-8")


def P(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    out.write(s + "\n")


SEEDS6 = [5200, 5201, 5202, 5203]
FS = [0.0, 0.5, 1.0, 1.5]
MODES = ["clock", "h2", "h3", "short", "alt", "level", "runner"]
rows = {k: [] for k in ("t", "aM", "aS", "gap", "M0")}
ys = {(m, f): [] for m in MODES for f in FS}
frac_on = []
for sd in SEEDS6:
    x = sample(NB, sd)
    H = run(x)
    vals, pos = top3(x)
    for t in range(2, 10):
        M0, S0 = vals[:, t - 1, 0], vals[:, t - 1, 1]
        pM, pS = pos[:, t - 1, 0], pos[:, t - 1, 1]
        i = np.flatnonzero((x[:, t] > M0) & (M0 > 0.3) & (S0 > 0.1) & (t - 1 - pM >= 1))
        h = H[i, t - 1]
        for k_, a in zip(("t", "aM", "aS", "gap", "M0"), (np.full(len(i), t), t - pM[i], t - pS[i], (M0 - S0)[i], M0[i])):
            rows[k_].append(a)
        ia, ib = 9 - (t - pM[i]), 9 - (t - pS[i])
        ar = np.arange(len(i))
        for m in MODES:
            c = h[:, ORD] @ LEFT[m]
            for f in FS:
                h2 = h.copy()
                if IS_COMPLEX[m]:
                    h2[:, ORD] += 2 * np.real(np.multiply.outer((f - 1) * c, RIGHT[m]))
                else:
                    h2[:, ORD] += np.real(np.multiply.outer((f - 1) * np.real(c), RIGHT[m]))
                ht = np.maximum(x[i, t, None] * w + h2 @ W.T, 0)
                s = ht @ U.T
                ys[(m, f)].append(s[ar, ia] - s[ar, ib])
                if m == "clock" and f == 1.0:
                    frac_on.append((ht[:, COMP] > 0).mean(0))
r = {k: np.concatenate(v) for k, v in rows.items()}
ys = {k: np.concatenate(v) for k, v in ys.items()}
D = np.abs(r["aM"] - r["aS"])
n = len(D)
P(f"{n} D1 records, seeds {SEEDS6}")
P("fraction of records with each comparator (1, 6, 7, 8) ON at the record step:", np.mean(frac_on, 0).round(4))


def c1c2(m, y):
    return np.linalg.lstsq(np.column_stack([r["gap"][m], r["M0"][m]]), y[m], rcond=None)[0]


fams = {"all": np.ones(n, bool), "both ages >= 2": r["aS"] >= 2, "aS = 1": r["aS"] == 1}
for mode in MODES:
    P(f"\nmode '{mode}' (lambda = {abs(LAM[mode]):.4f} @ {np.degrees(np.angle(LAM[mode])):.2f} deg) scaled by f at t-1:")
    for fn, fm in fams.items():
        P(f"  [{fn}] rows f; columns Delta = 1..8: c1   | P(y>0)")
        for f in FS:
            c, p = [], []
            for d in range(1, 9):
                m = fm & (D == d)
                if m.sum() < 300:
                    c.append("     -"); p.append("    -")
                    continue
                c.append(f"{c1c2(m, ys[(mode, f)])[0]:6.1f}"); p.append(f"{np.mean(ys[(mode, f)][m] > 0):.3f}")
            P(f"    f={f:3.1f}: " + " ".join(c) + "   | " + " ".join(p))
    # one-number summary: slope of c1(f) relative to c1(1), both ages >= 2, pooled over Delta via the cosine regressor
    m = r["aS"] >= 2
    cosf = (1 - np.cos(OMEGA * D[m])) * r["gap"][m]
    Ks = [np.linalg.lstsq(np.column_stack([cosf, r["M0"][m]]), ys[(mode, f)][m], rcond=None)[0] for f in FS]
    P("  cosine-law amplitude K (both ages >= 2; y ~ K (1-cos)(M0-S0) + c2 M0) at f = 0, 0.5, 1, 1.5: "
      + ", ".join(f"{k[0]:.1f}" for k in Ks) + "   | c2: " + ", ".join(f"{k[1]:.2f}" for k in Ks))
out.close()
