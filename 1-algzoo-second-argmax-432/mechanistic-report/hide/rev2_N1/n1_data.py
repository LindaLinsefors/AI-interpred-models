"""Reviewer N1: sampled numbers of sections 1-3 of report.md that are unsourced or disagree between notes.
Run: python -B n1_data.py      (N(0,1) inputs, fresh seeds 8101.., batches of 200k)"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
from collections import defaultdict

NB, BATCH = 6, 200_000
OMEGA = np.radians(42.748)
acc = defaultdict(float)          # name -> running sum


def add(name, num, den=None):
    acc[name + "|n"] += float(np.sum(num))
    acc[name + "|d"] += float(np.sum(den)) if den is not None else float(np.size(num))


def rate(name):
    return acc[name + "|n"] / max(acc[name + "|d"], 1)


reg = defaultdict(lambda: [0, 0])  # normal equations for regressions
ratios = []
for b in range(NB):
    x = sample(BATCH, 8101 + b)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    M, S, p1, p2, a1 = st["M1"], st["M2"], st["p1"], st["p2"], st["a1"]
    ar = np.arange(BATCH)
    non = (Z[:, :, COMP] > 0).sum(-1)                       # number of comparators on, (B, T)

    # ---- section 2: gate configurations
    add("all4on", non == 4); add("all4off", non == 0); add("1to3", (non > 0) & (non < 4))
    for t in (0, 9):
        add(f"all4on_t{t}", non[:, t] == 4)
    newmax = p1 == np.arange(10)[None, :]
    newrun = p2 == np.arange(10)[None, :]
    add("newmax|alloff", newmax & (non == 0), non == 0)
    add("posnewmax_or_t0|alloff", (newmax & (x > 0)) & (non == 0), non == 0)
    add("newrun|1to3", newrun & (non > 0) & (non < 4), (non > 0) & (non < 4))
    add("newmax|1to3", newmax & (non > 0) & (non < 4), (non > 0) & (non < 4))

    # ---- section 3.4 / summary: all four comparators zero at a new maximum over a positive old maximum
    for t in range(1, 10):
        ev = newmax[:, t] & (M[:, t - 1] > 0)
        add(f"nm_alloff_t{t}", (non[:, t] == 0) & ev, ev)
        add("nm_alloff_pooled1to9", (non[:, t] == 0) & ev, ev)
        if t <= 8:
            add("nm_alloff_pooled1to8", (non[:, t] == 0) & ev, ev)
        if 2 <= t <= 6:
            add("nm_alloff_pooled2to6", (non[:, t] == 0) & ev, ev)
        exc = (x[:, t] - M[:, t - 1]) / np.where(M[:, t - 1] > 0, M[:, t - 1], np.nan)
        add(f"nm_alloff_exc<=5%_t{t}", (non[:, t] == 0) & ev & (exc <= 0.05), ev & (exc <= 0.05))
        add("nm_alloff_exc<=5%_pooled", (non[:, t] == 0) & ev & (exc <= 0.05), ev & (exc <= 0.05))
        add(f"nm_alloff_exc>30%_t{t}", (non[:, t] == 0) & ev & (exc > 0.30), ev & (exc > 0.30))
        add("nm_alloff_exc>30%_pooled", (non[:, t] == 0) & ev & (exc > 0.30), ev & (exc > 0.30))
        # V2's selection (claim S6): M_{t-1} > 0.3 and old max not the previous input
        if t >= 2:
            ev2 = newmax[:, t] & (M[:, t - 1] > 0.3) & (p1[:, t - 1] < t - 1)
            add(f"nmS6_alloff_t{t}", (non[:, t] == 0) & ev2, ev2)
            add("nmS6_alloff_pooled", (non[:, t] == 0) & ev2, ev2)
            exc2 = (x[:, t] - M[:, t - 1]) / np.where(M[:, t - 1] > 0.3, M[:, t - 1], np.nan)
            add(f"nmS6_alloff_exc<=5%_t{t}", (non[:, t] == 0) & ev2 & (exc2 <= 0.05), ev2 & (exc2 <= 0.05))
            add(f"nmS6_alloff_exc>30%_t{t}", (non[:, t] == 0) & ev2 & (exc2 > 0.30), ev2 & (exc2 > 0.30))

    # ---- scores
    L = H @ U.T
    V = H[:, :, COMP] @ U[:, COMP].T
    m = L - V

    # ---- 3.1 running answer; 3.4 m ranks the max first (by t and by age of the max)
    for t in range(1, 10):
        valid = np.arange(10) >= 9 - t
        sp = S[:, t] > 0
        ps = np.where(valid, L[:, t], -np.inf).argmax(1) + t - 9
        add(f"T3_pos_t{t}", (ps == p2[:, t]) & sp, sp)
        add(f"T3_all_t{t}", ps == p2[:, t])
        if t >= 2:
            ok = sp & (p1[:, t] < t)
            pm = np.where(valid & (np.arange(10) <= 8), m[:, t], -np.inf).argmax(1) + t - 9
            add(f"S1_t{t}", (pm == p1[:, t]) & ok, ok)
            for a in range(1, 10):
                oka = ok & (a1[:, t] == a)
                add(f"S1_age{a}", (pm == p1[:, t]) & oka, oka)
            # unconditional on "max is not the current input" (caption of panel f)
            add(f"S1_incl_current_max_t{t}", (pm == p1[:, t]) & sp, sp)

    # ---- 3.4: V[9] regression and complex amplitudes, t = 9, all four on, S > 0.1, a1 = 2..9
    t = 9
    sel = (non[:, t] == 4) & (S[:, t] > 0.1) & (a1[:, t] >= 2)
    X = np.column_stack([M[sel, t], S[sel, t], x[sel, t]])
    reg["V9_pooled"][0] += X.T @ X; reg["V9_pooled"][1] += X.T @ V[sel, t, 9]
    for a in range(2, 10):
        s_ = sel & (a1[:, t] == a)
        Xa = np.column_stack([M[s_, t], S[s_, t], x[s_, t]])
        reg[f"V9_a{a}"][0] += Xa.T @ Xa; reg[f"V9_a{a}"][1] += Xa.T @ V[s_, t, 9]
    P = np.arange(8)
    D3 = np.column_stack([np.cos(OMEGA * P), np.sin(OMEGA * P), np.ones(8)])
    cV = np.linalg.lstsq(D3, V[sel, t, :8].T, rcond=None)[0]
    cm = np.linalg.lstsq(D3, m[sel, t, :8].T, rcond=None)[0]
    ZV = cV[0] - 1j * cV[1]; Zm = cm[0] - 1j * cm[1]
    ratios.append(Zm / ZV)

    # ---- 3.4: P(margin of the runner-up over the max > 0) by D and gap, t = 9, all on, S > 0.2, both older than x_9
    sel = (non[:, t] == 4) & (S[:, t] > 0.2) & (p1[:, t] < 9) & (p2[:, t] < 9)
    D = np.abs(p1[:, t] - p2[:, t]); g = (M[:, t] - S[:, t]) / M[:, t]
    ds = L[ar, t, p2[:, t]] - L[ar, t, p1[:, t]]
    for lo, hi in [(0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, .8), (.8, 1.0)]:
        for d in range(1, 9):
            s_ = sel & (D == d) & (g > lo) & (g <= hi)
            add(f"marg_D{d}_g{lo}-{hi}", (ds > 0) & s_, s_)

print("== section 2: gates")
print("all four on %.4f  (t=0: %.4f, t=9: %.4f)   all off %.4f   one to three %.4f" % (
    rate("all4on"), rate("all4on_t0"), rate("all4on_t9"), rate("all4off"), rate("1to3")))
print("P(new max | all off) = %.3f  (positive new max: %.3f)   P(new runner-up | 1-3 on) = %.3f   P(new max | 1-3 on) = %.3f" % (
    rate("newmax|alloff"), rate("posnewmax_or_t0|alloff"), rate("newrun|1to3"), rate("newmax|1to3")))

print("\n== all four comparators exactly zero at a new maximum over a positive old maximum (M_{t-1} > 0)")
print("by step 1..9:", " ".join("%.3f" % rate(f"nm_alloff_t{t}") for t in range(1, 10)))
print("pooled steps 1-9: %.4f   1-8: %.4f   2-6: %.4f" % (rate("nm_alloff_pooled1to9"), rate("nm_alloff_pooled1to8"), rate("nm_alloff_pooled2to6")))
print("excess <= 5%% by step:", " ".join("%.3f" % rate(f"nm_alloff_exc<=5%_t{t}") for t in range(1, 10)), " pooled %.3f" % rate("nm_alloff_exc<=5%_pooled"))
print("excess > 30%% by step:", " ".join("%.3f" % rate(f"nm_alloff_exc>30%_t{t}") for t in range(1, 10)), " pooled %.3f" % rate("nm_alloff_exc>30%_pooled"))
print("V2/S6 selection (M_{t-1} > 0.3, old max not at t-1), steps 2..9:", " ".join("%.3f" % rate(f"nmS6_alloff_t{t}") for t in range(2, 10)), " pooled %.3f" % rate("nmS6_alloff_pooled"))
print("   excess <= 5%%:", " ".join("%.3f" % rate(f"nmS6_alloff_exc<=5%_t{t}") for t in range(2, 10)))
print("   excess > 30%%:", " ".join("%.3f" % rate(f"nmS6_alloff_exc>30%_t{t}") for t in range(2, 10)))

print("\n== 3.1 running answer, t = 1..9")
print("given S_t > 0:", " ".join("%.4f" % rate(f"T3_pos_t{t}") for t in range(1, 10)))
print("all states   :", " ".join("%.4f" % rate(f"T3_all_t{t}") for t in range(1, 10)))
print("\n== 3.4 top of m among older inputs = max (S_t > 0, max not current), t = 2..9")
print(" ".join("%.4f" % rate(f"S1_t{t}") for t in range(2, 10)))
print("by age of the max 1..9:", " ".join("%.3f" % rate(f"S1_age{a}") for a in range(1, 10)))
print("same without the condition 'max is not the current input' (only S_t > 0), t = 2..9:")
print(" ".join("%.4f" % rate(f"S1_incl_current_max_t{t}") for t in range(2, 10)))

print("\n== 3.4 V[9] ~ cM*M + cS*S + cx*x9 (t = 9, all four on, S > 0.1, max age 2..9)")
print("pooled:", np.round(np.linalg.solve(*reg["V9_pooled"]), 2))
for a in range(2, 10):
    print(f"  age {a}:", np.round(np.linalg.solve(*reg[f"V9_a{a}"]), 2))

r = np.concatenate(ratios)
print("\n== 3.4 per-sequence complex amplitudes over indices 0-7 (same selection), r = Z_m / Z_V")
print("median Re r = %.3f   median |r| = %.3f   median |1 + r| = |Z_m + Z_V|/|Z_V| = %.3f   IQR of Re r: %.2f..%.2f   n = %d" % (
    np.median(r.real), np.median(np.abs(r)), np.median(np.abs(1 + r)), *np.percentile(r.real, [25, 75]), len(r)))

print("\n== 3.4 P(s[p_S] > s[p_M]) at t = 9 by D (rows 1..8) and gap/M bin (0,.05] (.05,.15] (.15,.3] (.3,.5] (.5,.8] (.8,1]")
for d in range(1, 9):
    row = []
    for lo, hi in [(0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, .8), (.8, 1.0)]:
        k = f"marg_D{d}_g{lo}-{hi}"
        row.append("%.4f (n=%d)" % (rate(k), acc[k + "|d"]))
    print(f"D={d}: " + "  ".join(row))
