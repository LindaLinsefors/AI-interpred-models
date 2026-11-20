"""Verifier V1, attack E5: controlled no-event steps ("hold" test) and a few supplementary checks.

(1) HOLD: take x_0..x_3 from N(0,1) with S_3 > 0.2, then feed 6 filler inputs that are no events
    (below the runner-up), all equal to f.  If a step without an event only shifts the tape, then at every
    later step the largest valid entry is still the runner-up, and the entries of the four original inputs,
    relative to the runner-up's entry, stay what they were at t = 3.
    Fillers: f = -2M, -0.5M, 0, 0.5S, 0.9S.   Seed 5901, 200k sequences.
(2) Sigma row 9 and the conditioning of pinv(U_o).
(3) T3 with only part of the state read out (which units carry the early readout).
(4) T4 medians for ages >= 1 under other iid input distributions (are those 'laws' distribution-free?).
(5) T3 with tiny random values instead of the implicit zero padding (continuity; seed 5905).
Run: python -B v1_attack_hold.py -> v1_attack_hold.txt
"""
from vlib import *

P = Tee("v1_attack_hold.txt")

# ---------------------------------------------------------------- (1)
P("=== (1) hold test: x_0..x_3 real (S_3 > 0.2), then six equal fillers f below the runner-up (seed 5901) ===")
x4 = sample(200_000, 5901)[:, :4]
p, val = top3(x4, 3)
keep = val[:, 1] > 0.2
x4, p, val = x4[keep], p[keep], val[keep]
p1, p2, M, S = p[:, 0], p[:, 1], val[:, 0], val[:, 1]
n = len(x4); r = np.arange(n)
P(f"sequences: {n}")
fillers = {"f = -2M": -2 * M, "f = -0.5M": -0.5 * M, "f = 0": 0 * M, "f = 0.5S": 0.5 * S, "f = 0.9S": 0.9 * S}
for name, f in fillers.items():
    x = np.concatenate([x4, np.repeat(f[:, None], 6, 1)], 1)
    H = run(x)
    acc_all, acc_orig, r2_step, r2_tot, ratio, relmax = [], [], [], [], [], []
    sub3 = (H[:, 3] @ U.T)[:, 6:10]                          # entries of the 4 original inputs at t=3
    rel3 = (sub3 - sub3[r, p2][:, None]) / M[:, None]
    prev = sub3
    for t in range(3, 10):
        full = H[:, t] @ U.T
        valid = full[:, 9 - t:]
        acc_all.append(np.mean(valid.argmax(1) == p2))       # among all valid positions (fillers included)
        orig = full[:, 9 - t: 13 - t]                        # the 4 original inputs sit at indices 9-t..12-t
        acc_orig.append(np.mean(orig.argmax(1) == p2))
        rel = (orig - orig[r, p2][:, None]) / M[:, None]
        if t > 3:
            y = orig - orig.mean(1, keepdims=True); pr = prev - prev.mean(1, keepdims=True)
            r2_step.append(1 - ((y - pr) ** 2).sum() / (y ** 2).sum())
            p0 = sub3 - sub3.mean(1, keepdims=True)
            r2_tot.append(1 - ((y - p0) ** 2).sum() / (y ** 2).sum())
            oth = np.ones((n, 4), bool); oth[r, p1] = False; oth[r, p2] = False
            ratio.append(np.median(rel[oth] / rel3[oth]))
            relmax.append(np.median(rel[r, p1]))
        prev = orig
    P(f"--- {name}")
    P("    t =                                   " + "".join(f"{t:8d}" for t in range(3, 10)))
    P("    P(largest valid entry = runner-up)    " + "".join(f"{a:8.4f}" for a in acc_all))
    P("    P(largest of the 4 original = r.-up)  " + "".join(f"{a:8.4f}" for a in acc_orig))
    P("    R2 one-step shift (4 orig., centred)          " + "".join(f"{a:8.3f}" for a in r2_step))
    P("    R2 against the t=3 entries (cumulative)       " + "".join(f"{a:8.3f}" for a in r2_tot))
    P("    median (rel. score now / rel. score at t=3), the two non-top-2 originals " + "".join(f"{a:7.2f}" for a in ratio))
    P(f"    median rel. score of the max: t=3 {np.median(rel3[r, p1]):.2f} -> " + "".join(f"{a:7.2f}" for a in relmax))

# ---------------------------------------------------------------- (2)
P("\n=== (2) Sigma row 9 and the conditioning of pinv(U_o) ===")
Uo = U[:, ORD]; A = W[np.ix_(ORD, ORD)]
uu, ss, vt = np.linalg.svd(Uo, full_matrices=False)
P("singular values of U_o:", np.round(ss, 3))
P("left singular vector of the smallest singular value (0.293):", np.round(uu[:, -1], 3), " cos with the all-ones vector: %.4f" % abs(uu[:, -1].sum() / np.sqrt(10)))
P("column sums of U (sum over positions, per unit):", np.round(U.sum(0), 3), " |1^T U_o| =", round(float(np.linalg.norm(Uo.sum(0))), 3), " vs typical row norm", round(float(np.linalg.norm(Uo, axis=1).mean()), 2))
Sig = Uo @ A @ np.linalg.pinv(Uo)
resid = Uo @ A - Sig @ Uo
P("share of the squared Frobenius residual in each row 0..9:", np.round((resid ** 2).sum(1) / (resid ** 2).sum(), 3))
pinv9 = (vt[:9].T / ss[:9]) @ uu[:, :9].T
Sig9 = Uo @ A @ pinv9
P("Sigma with the smallest singular direction dropped (rank-9 pinv): superdiagonal i=0..8:", np.round([Sig9[i, i + 1] for i in range(9)], 3))
P("   row 9:", np.round(Sig9[9], 3))
P("   relative residual all rows: %.4f ; rows 0-8: %.4f" % (np.linalg.norm(Uo @ A - Sig9 @ Uo) / np.linalg.norm(Uo @ A),
                                                             np.linalg.norm((Uo @ A - Sig9 @ Uo)[:9]) / np.linalg.norm((Uo @ A)[:9])))
P("   Sigma row 9 minus its mean:", np.round(Sig[9] - Sig[9].mean(), 3), " (mean %.3f)" % Sig[9].mean())

# ---------------------------------------------------------------- (3)
P("\n=== (3) T3 (S_t > 0) with only part of the state read out (seed 5903, 200k) ===")
x = sample(200_000, 5903)
H = run(x)
parts = {"all 16 units": list(range(16)), "13 ordinary": ORD, "comparators only": COMP, "all but comparators": [u for u in range(16) if u not in COMP],
         "wave only": WAVE, "pulse only": PULSE}
P("   t   " + "  ".join(f"{k:>20s}" for k in parts))
for t in range(1, 10):
    pp, vv = top3(x, t)
    pos = vv[:, 1] > 0
    row = []
    for k, units in parts.items():
        s = (H[:, t][:, units] @ U[:, units].T)[:, 9 - t:]
        row.append(f"{np.mean(s.argmax(1)[pos] == pp[pos, 1]):20.4f}")
    P(f"   {t}   " + "  ".join(row))

# ---------------------------------------------------------------- (4)
P("\n=== (4) T4 medians for ages >= 1 under other iid distributions (200k each; S_t > 0.2, t = 2..9) ===")
bins = [(-0.10, -0.05), (-0.25, -0.15), (-0.7, -0.5), (-np.inf, -2)]


def t4_other(xx):
    Hh = run(xx)
    D, V, AG = [], [], []
    for t in range(2, 10):
        sub, _ = scores(Hh, t)
        pp, vv = top3(xx, t)
        kp = vv[:, 1] > 0.2
        sub, xw, q1, q2, Mm, Ss = sub[kp], xx[kp, : t + 1], pp[kp, 0], pp[kp, 1], vv[kp, 0], vv[kp, 1]
        rr = np.arange(len(Mm))
        d = (sub - sub[rr, q2][:, None]) / Mm[:, None]
        v = (xw - Ss[:, None]) / Mm[:, None]
        oth = np.ones(d.shape, bool); oth[rr, q1] = False; oth[rr, q2] = False
        D.append(d[oth]); V.append(v[oth]); AG.append(np.broadcast_to(t - np.arange(t + 1), d.shape)[oth])
    return np.concatenate(D), np.concatenate(V), np.concatenate(AG)


rngd = np.random.default_rng
dd = {"N(0,1) seed 5911": rngd(5911).standard_normal((200_000, 10)), "Unif(-1,1) seed 5912": rngd(5912).uniform(-1, 1, (200_000, 10)),
      "Laplace seed 5913": rngd(5913).laplace(0, 1, (200_000, 10)), "Exp(1)-0.5 seed 5914": rngd(5914).exponential(1, (200_000, 10)) - 0.5}
P("   median d in bins (-0.10,-0.05] | (-0.25,-0.15] | (-0.7,-0.5] | v<-2      [claims: age 1: -4.7, -9.7, -12.8, -18.3; ages>=2: -5.3, -10.8, -17.7, -23.7]")
for name, xx in dd.items():
    D, V, AG = t4_other(xx)
    for lab, m in (("age 1  ", AG == 1), ("ages>=2", AG >= 2)):
        row = []
        for lo, hi in bins:
            k = m & (V > lo) & (V <= hi)
            row.append(f"{np.median(D[k]):7.2f} (n={k.sum()})" if k.sum() > 200 else "    n/a")
        P(f"   {name:22s} {lab}: " + " | ".join(row))

# ---------------------------------------------------------------- (5)
P("\n=== (5) T3 by zero padding vs padding with tiny values N(0, 0.01^2) (seed 5905, 200k; S>0.05 so that the pads cannot be top-2) ===")
x = sample(200_000, 5905)
rng = np.random.default_rng(5906)
H = run(x)
for t in (1, 2, 3, 5, 7):
    pp, vv = top3(x, t)
    ok = vv[:, 1] > 0.05
    early = (H[:, t] @ U.T)[:, 9 - t:].argmax(1)
    xp = 0.01 * rng.standard_normal((len(x), 10)); xp[:, 9 - t:] = x[:, : t + 1]
    lg = run(xp)[:, 9] @ U.T
    P(f"   t={t}: early readout {np.mean(early[ok] == pp[ok, 1]):.4f} | tiny-padded full run, argmax over last t+1: {np.mean(lg[:, 9 - t:].argmax(1)[ok] == pp[ok, 1]):.4f}"
      f" | tiny-padded, argmax over all 10 (= plain network answer): {np.mean((lg.argmax(1) - (9 - t))[ok] == pp[ok, 1]):.4f}")
