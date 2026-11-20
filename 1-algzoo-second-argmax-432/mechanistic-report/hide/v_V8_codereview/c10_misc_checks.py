"""Code review, smaller checks of common.py, modes.py, e12, e13, e15b, e16, e19, e25.

 (1) common.running_stats against a brute-force loop; common.run against a plain re-implementation;
     accuracy on N(0,1) and Unif(-1,1); common.traces identity.
 (2) e12 / T3 re-implemented without running_stats.
 (3) e13 / T4: the range quoted for 'the max, ages 2..9, v > 0.2'.
 (4) e15b / S1: why is t = 9 lower (0.891)?  By position of the max; what wins instead.
 (5) e15b / S6: (iv) is labelled 'comparators all on at t and t-1' but only t is conditioned; and at a
     record, how often are all four comparators off (then s_t = m_t exactly)?
 (6) modes.py: gauge, bi-orthogonality, M1 and M2 numbers.
 (7) e25 / S4: the sinusoid fit of the comparators' readout columns with the frequency left free.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import *
import modes
from modes import LEFT, RIGHT, LAM, NAMES, IS_COMPLEX

# ---------------------------------------------------------------- (1)
x = sample(3000, 7)
st = running_stats(x)
bad = 0
for b in range(len(x)):
    for t in range(T):
        pre = x[b, :t + 1]
        o = sorted(range(t + 1), key=lambda i: -pre[i])
        for k in range(3):
            if k <= t:
                bad += (st[f"p{k+1}"][b, t] != o[k]) + (st[f"M{k+1}"][b, t] != pre[o[k]]) + (st[f"a{k+1}"][b, t] != t - o[k])
            else:
                bad += (st[f"p{k+1}"][b, t] != -1) + (st[f"M{k+1}"][b, t] != -np.inf) + (st[f"a{k+1}"][b, t] != -1)
print("(1) running_stats vs brute force on 3000 sequences: mismatches =", int(bad))
h = np.zeros(N); Hloop = np.empty((T, N))
for t in range(T):
    h = np.maximum(w * x[0, t] + W @ h, 0); Hloop[t] = h
print("    run vs plain loop (one sequence): max abs diff =", float(np.abs(run(x[:1])[0] - Hloop).max()))
Hh, J = traces(x[:200])
print("    traces identity h_t = sum_p J[t,p] x_p: max abs error =", float(np.abs(np.einsum("btpn,bp->btn", J, x[:200]) - Hh).max()))
xb = sample(1 << 20, 123)
print("    accuracy on 2^20 N(0,1) sequences:", round(float(accuracy(xb)), 4), "  on 2^20 Unif(-1,1):", round(float(accuracy(sample(1 << 20, 124, 'uniform'))), 4),
      "  (project notes: 0.9531 / 0.9378)")
del xb

# ---------------------------------------------------------------- (2)
x = sample(400_000, 1212)
H = run(x)
acc_all, acc_pos = [], []
for t in range(1, 10):
    pre = x[:, :t + 1]
    srt = np.sort(pre, axis=1)
    second_pos = np.argsort(pre, axis=1)[:, -2]
    s2 = srt[:, -2]
    sc = H[:, t] @ U.T
    best = np.full(len(x), -1); bestv = np.full(len(x), -np.inf)
    for p in range(t + 1):
        v = sc[:, 9 - (t - p)]
        up = v > bestv
        best[up] = p; bestv[up] = v[up]
    acc_all.append(np.mean(best == second_pos)); acc_pos.append(np.mean(best[s2 > 0] == second_pos[s2 > 0]))
print("\n(2) T3 re-implemented: given S_t > 0:", np.round(acc_pos, 3), "\n                       all states:  ", np.round(acc_all, 3))

# ---------------------------------------------------------------- (3)
x = sample(400_000, 1313)
H = run(x)
st = running_stats(x)
B = len(x); ar = np.arange(B)
gb = [0.2, 0.35, 0.5, 0.7, 0.85, 1.0]
meds = []
for A in range(2, 10):
    dd, vv = [], []
    for t in range(max(2, A), 10):
        l = H[:, t] @ U.T
        Mx, Sx = st["M1"][:, t], st["M2"][:, t]
        s = (Sx > 0.2) & (st["a1"][:, t] == A)
        d = (l[ar, 9 - A] - l[ar, 9 - (t - st["p2"][:, t])]) / Mx
        dd.append(d[s]); vv.append(((Mx - Sx) / Mx)[s])
    dd, vv = np.concatenate(dd), np.concatenate(vv)
    for lo, hi in zip(gb[:-1], gb[1:]):
        q = (vv > lo) & (vv <= hi)
        if q.sum() > 200:
            meds.append(np.median(dd[q]))
print("\n(3) T4, the max at ages 2..9, bins with v > 0.2: medians range from", round(min(meds), 1), "to", round(max(meds), 1),
      "  (claim: 'between -4.5 and -14.5'; -4.5 belongs to the bin (0.1, 0.2] at age 2)")

# ---------------------------------------------------------------- (4)
x = sample(500_000, 1516)
H, Z = run(x, return_pre=True)
st = running_stats(x)
Uc = np.zeros_like(U); Uc[:, COMP] = U[:, COMP]
S_all = H @ U.T
M_all = S_all - H @ Uc.T
print("\n(4) S1: P(argmax m_t over indices 9-t..8 = position of the max | S_t > 0, max not current), by AGE of the max (columns a1 = 1..9)")
for t in (6, 7, 8, 9):
    valid = (np.arange(10) >= 9 - t) & (np.arange(10) <= 8)
    ok = (st["M2"][:, t] > 0) & (st["p1"][:, t] < t)
    pm = np.where(valid, M_all[:, t], -np.inf).argmax(1) + t - 9
    row = []
    for a in range(1, t + 1):
        q = ok & (st["a1"][:, t] == a)
        row.append(f"{np.mean(pm[q] == st['p1'][q, t]):.3f}")
    print(f"    t={t}: all {np.mean(pm[ok] == st['p1'][ok, t]):.3f} | " + " ".join(row))
t = 9
valid = (np.arange(10) <= 8)
ok = (st["M2"][:, t] > 0) & (st["p1"][:, t] < t)
pm = np.where(valid, M_all[:, t], -np.inf).argmax(1)
wrong = ok & (pm != st["p1"][:, t])
print("    t=9, the wrong choices: P(choice = position 0) =", round(float(np.mean(pm[wrong] == 0)), 3), "  P(choice = 8) =", round(float(np.mean(pm[wrong] == 8)), 3),
      "  P(choice = runner-up) =", round(float(np.mean(pm[wrong] == st['p2'][wrong, t])), 3))
print("    t=9 with position 0 also excluded from the candidates and from the population (p1 >= 1):",
      round(float(np.mean((np.where(valid & (np.arange(10) >= 1), M_all[:, t], -np.inf).argmax(1) == st['p1'][:, t])[ok & (st['p1'][:, t] >= 1)])), 3))
print("    P(argmax m_t = max) at t = 9 by position of the max 0..8:",
      " ".join(f"{np.mean(pm[ok & (st['p1'][:, t] == p)] == p):.3f}" for p in range(9)))

# ---------------------------------------------------------------- (5)
def r2(a, b):
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    sl = (a * b).sum() / (b * b).sum()
    return 1 - ((a - sl * b) ** 2).sum() / (a ** 2).sum(), sl


print("\n(5) S6 (iv), no event: R2 of s_t vs shifted s_(t-1) on inputs 0..t-2;  as coded (comparators on at t) | as labelled (on at t and t-1) | also requiring x_(t-1) < S_(t-2) (no event at t-1 either)")
on = (Z[:, :, COMP] > 0).all(2)
for t in range(3, 10):
    base = (x[:, t] < st["M2"][:, t - 1]) & (st["M2"][:, t - 1] > 0.1)
    row = []
    for s in (base & on[:, t], base & on[:, t] & on[:, t - 1], base & on[:, t] & on[:, t - 1] & (x[:, t - 1] < st["M2"][:, t - 2])):
        c, d = r2(S_all[s, t][:, 9 - t: 8], S_all[s, t - 1][:, 10 - t: 9])
        row.append(f"{c:.3f} (slope {d:.2f}, n={s.sum():6d})")
    print(f"    t={t}: " + " | ".join(row))
print("    S6 (iii), record at t: share of records with ALL four comparators off at t (then s_t = m_t exactly); R2 of s_t vs shifted m_(t-1) for all records | all-off records | not-all-off records")
for t in range(3, 10):
    s = (x[:, t] > st["M1"][:, t - 1]) & (st["M1"][:, t - 1] > 0.3) & (st["p1"][:, t - 1] < t - 1)
    off = ~(Z[:, t][:, COMP] > 0).any(1)
    row = []
    for q in (s, s & off, s & ~off):
        row.append(f"{r2(S_all[q, t][:, 9 - t: 8], M_all[q, t - 1][:, 10 - t: 9])[0]:.3f}")
    # the same comparison for m_t itself (no comparators at either step): is the 'shift' property of m, or of the record?
    mm = r2(M_all[s, t][:, 9 - t: 8], M_all[s, t - 1][:, 10 - t: 9])[0]
    nb = (x[:, t] < st["M2"][:, t - 1]) & (st["M2"][:, t - 1] > 0.1) & on[:, t]
    mm_noevent = r2(M_all[nb, t][:, 9 - t: 8], M_all[nb, t - 1][:, 10 - t: 9])[0]
    print(f"    t={t}: all off {np.mean(off[s]):.3f} | " + " | ".join(row) + f" |  m_t vs shifted m_(t-1): at records {mm:.3f}, at no-event steps {mm_noevent:.3f}")

# ---------------------------------------------------------------- (6)
print("\n(6) modes.py")
A = W[np.ix_(ORD, ORD)]
Lm = np.array([LEFT[n] for n in NAMES]); Rm = np.array([RIGHT[n] for n in NAMES])
print("    LEFT @ KICK4 == 1:", np.allclose(Lm @ W[ORD, 4], 1), "   LEFT_i . RIGHT_j = delta_ij:", np.allclose(Lm @ Rm.T, np.eye(len(NAMES))),
      "   LEFT A = lam LEFT:", np.allclose(Lm @ A, np.array([LAM[n] for n in NAMES])[:, None] * Lm),
      "   LEFT . conj(RIGHT) = 0 for complex modes:", np.allclose([LEFT[n] @ np.conj(RIGHT[n]) for n in NAMES if IS_COMPLEX[n]], 0))
print("    eigenvalues:", "  ".join(f"{n} {abs(LAM[n]):.4f}@{np.degrees(np.angle(LAM[n])):.2f}" for n in NAMES))
ci = [ORD.index(u) for u in COMP]
print("    M2, share of |LEFT|^2 on the comparators:", "  ".join(f"{n} {np.sum(np.abs(LEFT[n][ci]) ** 2) / np.sum(np.abs(LEFT[n]) ** 2):.3f}" for n in NAMES),
      "  ; level on unit 2:", round(float(np.abs(LEFT['level'][ORD.index(2)]) ** 2 / np.sum(np.abs(LEFT['level']) ** 2)), 3))
print("    the same after rescaling every unit by 1/(its rms activation on data)  [the share depends on the unit scale]:")
Hd = run(sample(200_000, 5))
rms = np.sqrt((Hd[:, 2:, :][:, :, ORD] ** 2).mean((0, 1)))
print("        ", "  ".join(f"{n} {np.sum(np.abs(LEFT[n][ci] * rms[ci]) ** 2) / np.sum(np.abs(LEFT[n] * rms) ** 2):.3f}" for n in NAMES))
lev = RIGHT["level"].real
print("    M1, level eigenvector relative to unit 2:", np.round(lev / lev[ORD.index(2)], 2), " all one sign:", bool((lev > 0).all() or (lev < 0).all()))
ech = RIGHT["echo"].real
print("    M1, |cos(w[ORD], echo eigenvector)| =", round(float(abs(w[ORD] @ ech) / np.linalg.norm(w[ORD]) / np.linalg.norm(ech)), 4))

# ---------------------------------------------------------------- (7)
print("\n(7) S4, comparator readout columns over positions 0..7: fixed frequency 42.75 deg (as e25) | frequency left free (grid search)")
p = np.arange(8)
for u in COMP:
    def fit(omd):
        X = np.column_stack([np.cos(np.radians(omd) * p), np.sin(np.radians(omd) * p), np.ones(8)])
        c, *_ = np.linalg.lstsq(X, U[:8, u], rcond=None)
        return 1 - ((U[:8, u] - X @ c) ** 2).sum() / ((U[:8, u] - U[:8, u].mean()) ** 2).sum(), np.hypot(c[0], c[1]), np.degrees(np.arctan2(c[1], c[0]))
    grid = np.arange(20, 80, 0.05)
    best = grid[int(np.argmax([fit(o)[0] for o in grid]))]
    f0, f1 = fit(42.75), fit(best)
    X2 = np.column_stack([np.cos(np.radians(42.75) * np.arange(10)), np.sin(np.radians(42.75) * np.arange(10)), np.ones(10)])
    c2, *_ = np.linalg.lstsq(X2, U[:, u], rcond=None)
    r10 = 1 - ((U[:, u] - X2 @ c2) ** 2).sum() / ((U[:, u] - U[:, u].mean()) ** 2).sum()
    print(f"    unit {u}: 42.75 deg: a = {f0[1]:.2f}, phi = {f0[2]:.0f}, R2 = {f0[0]:.3f} | best frequency {best:.2f} deg: R2 = {f1[0]:.3f} | all ten positions at 42.75 deg: R2 = {r10:.3f}")
