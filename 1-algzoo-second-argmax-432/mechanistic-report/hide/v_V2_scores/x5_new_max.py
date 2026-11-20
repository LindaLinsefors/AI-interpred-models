"""Attacks on S6 ("at a new maximum the comparators are zero, so s becomes the shifted m").

Events: new max at t (x_t > M_{t-1}, M_{t-1} > 0.3, p1(t-1) < t-1), t = 3..9; inputs 0..t-2.
 1. Are the comparators zero?  By relative margin r = (x_t - M_{t-1}) / M_{t-1}; size of V_t when they are not.
 2. How much of R^2(s_t ~ shifted m_{t-1}) is a generic profile?  (a) R^2 without the old max's position;
    (b) template baseline: M_{t-1} x (profile that depends only on t and the age g of the old max);
    (c) R^2 between the residuals of s_t and of shifted m_{t-1} after removing their templates;
    (d) shifted m_{t-1} of ANOTHER event with the same (t, g), rescaled by the ratio of M_{t-1}.
 3. Intervention: set h_{t-1}[COMP] = 0, recompute step t with the real x_t. If s_t at a new max were a function of
    m_{t-1} alone, nothing would change.

Usage: python -B x5_new_max.py [seed0] [n_batches]
"""
import sys

sys.dont_write_bytecode = True
from vbase import *

seed0 = int(sys.argv[1]) if len(sys.argv) > 1 else 5400
nb = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ts = range(3, 10)
RB = [0, 0.05, 0.15, 0.3, 0.5, 1.0, np.inf]

ev = {t: {k: [] for k in ("Y", "X", "Xs", "g", "M", "r", "non", "V", "Yint", "Yint_m", "non_int", "pS")} for t in ts}
for x, H, M, S, p1, p2 in batches(nb, seed0):
    for t in ts:
        new = (x[:, t] > M[:, t - 1]) & (M[:, t - 1] > 0.3) & (p1[:, t - 1] < t - 1)
        s_t, V_t, m_t = scores(H[new, t]); s_p, V_p, m_p = scores(H[new, t - 1])
        e = ev[t]
        e["Y"].append(s_t[:, 9 - t:8]); e["X"].append(m_p[:, 10 - t:9]); e["Xs"].append(s_p[:, 10 - t:9])
        e["g"].append(t - 1 - p1[new, t - 1]); e["M"].append(M[new, t - 1]); e["pS"].append(p2[new, t - 1])
        e["r"].append((x[new, t] - M[new, t - 1]) / M[new, t - 1])
        e["non"].append((H[new, t][:, COMP] > 0).sum(1)); e["V"].append(V_t[:, 9 - t:8])
        # intervention: comparators wiped at t-1
        hp = H[new, t - 1].copy(); hp[:, COMP] = 0.0
        ht = np.maximum(x[new, t, None] * w + hp @ W.T, 0.0)
        e["Yint"].append((ht @ U.T)[:, 9 - t:8]); e["non_int"].append((ht[:, COMP] > 0).sum(1))
        # control intervention: four wave units (10, 11, 12, 15) wiped at t-1
        hp = H[new, t - 1].copy(); hp[:, [10, 11, 12, 15]] = 0.0
        e["Yint_m"].append((np.maximum(x[new, t, None] * w + hp @ W.T, 0.0) @ U.T)[:, 9 - t:8])
for t in ts:
    for k in ev[t]:
        ev[t][k] = np.concatenate(ev[t][k])


def cen(a):
    return a - a.mean(1, keepdims=True)


def r2_slope(X, Y):
    return (X * Y).sum() ** 2 / ((X * X).sum() * (Y * Y).sum()), (X * Y).sum() / (X * X).sum()


print(f"seeds {seed0}..{seed0 + nb - 1}; {nb * B_DEFAULT} sequences")
print("\n=== 1. comparators at the step of a new maximum ===")
print("P(all four comparators exactly zero at t) by relative margin r = (x_t - M_{t-1})/M_{t-1}; bins", RB)
for t in ts:
    e = ev[t]; rb = np.digitize(e["r"], RB, right=True) - 1
    print(f"t={t}: all {np.mean(e['non'] == 0):.4f} | " + " ".join(f"{np.mean(e['non'][rb == k] == 0):.3f}" for k in range(len(RB) - 1))
          + "   share of events per bin: " + " ".join(f"{np.mean(rb == k):.3f}" for k in range(len(RB) - 1)))
print("when at least one comparator is on: rms over old inputs of centred V_t relative to centred s_t (pooled), and P(old max = argmax s_t) with / without a comparator on")
for t in ts:
    e = ev[t]; on = e["non"] > 0
    Vc, Yc = cen(e["V"][on]), cen(e["Y"][on])
    oldmax = t - 2 - (e["g"] - 1)                       # position of the old max = its column among inputs 0..t-2
    hit = e["Y"].argmax(1) == oldmax
    print(f"t={t}: n_on = {on.sum():6d}  rms(V)/rms(s) = {np.sqrt((Vc ** 2).sum() / (Yc ** 2).sum()):.3f}   "
          f"P(hit | some comparator on) = {hit[on].mean():.4f}   P(hit | all zero) = {hit[~on].mean():.4f}")

print("\n=== 2. how much of R^2(s_t ~ shifted m_{t-1}) is a generic profile? ===")
print("t | R^2 all old inputs (slope) | without the old max's position (slope) | template (t, g, M) alone | residual-vs-residual R^2 (slope)"
      " | other event's m, same (t, g) | old max entry only: corr(s_t[old max] - mean, m_{t-1}[old max] - mean)")
rng = np.random.default_rng(seed0 + 99)
for t in ts:
    e = ev[t]
    Yc, Xc = cen(e["Y"]), cen(e["X"])
    r_all = r2_slope(Xc, Yc)
    n, k = Yc.shape
    col = t - 2 - (e["g"] - 1)                          # column of the old max
    if k >= 3:
        mask = np.ones((n, k), bool); mask[np.arange(n), col] = False
        Y2 = cen(e["Y"][mask].reshape(n, k - 1)); X2 = cen(e["X"][mask].reshape(n, k - 1))
        r_wo = r2_slope(X2, Y2)
    else:
        r_wo = (np.nan, np.nan)
    RY = np.empty_like(Yc); RX = np.empty_like(Xc); Xperm = np.empty_like(Xc)
    for g in np.unique(e["g"]):
        kk = np.flatnonzero(e["g"] == g)
        Mg = e["M"][kk, None]
        PY = (Mg * Yc[kk]).sum(0) / (Mg ** 2).sum(); PX = (Mg * Xc[kk]).sum(0) / (Mg ** 2).sum()
        RY[kk] = Yc[kk] - Mg * PY; RX[kk] = Xc[kk] - Mg * PX
        perm = rng.permutation(len(kk))
        Xperm[kk] = Xc[kk][perm] * (Mg / Mg[perm])
    r_tmpl = 1 - (RY ** 2).sum() / (Yc ** 2).sum()
    r_res = r2_slope(RX, RY)
    r_perm = r2_slope(Xperm, Yc)
    cm = np.corrcoef(Yc[np.arange(n), col], Xc[np.arange(n), col])[0, 1]
    print(f"t={t} | {r_all[0]:.3f} ({r_all[1]:.2f}) | {r_wo[0]:.3f} ({r_wo[1]:.2f}) | {r_tmpl:.3f} | {r_res[0]:.3f} ({r_res[1]:.2f}) | {r_perm[0]:.3f} | {cm:.3f}")

print("\n=== 3. intervention at a new maximum: h_{t-1}[COMP] := 0, then the real step t ===")
print("t | R^2(s_t intervened vs s_t real; 1 - SS_diff/SS_tot, centred over old inputs) | slope-free corr^2 | P(old max = argmax) real -> intervened"
      " | P(all comparators zero at t) real -> intervened | R^2(intervened s_t ~ shifted m_{t-1}) | control: wipe units 10,11,12,15 instead: R^2 vs real, P(old max = argmax)")
for t in ts:
    e = ev[t]
    Yc, Ic, Xc, Cc = cen(e["Y"]), cen(e["Yint"]), cen(e["X"]), cen(e["Yint_m"])
    col = t - 2 - (e["g"] - 1)
    print(f"t={t} | {1 - ((Ic - Yc) ** 2).sum() / (Yc ** 2).sum():.3f} | {r2_slope(Ic, Yc)[0]:.3f} | {(e['Y'].argmax(1) == col).mean():.4f} -> {(e['Yint'].argmax(1) == col).mean():.4f}"
          f" | {np.mean(e['non'] == 0):.3f} -> {np.mean(e['non_int'] == 0):.3f} | {r2_slope(Xc, Ic)[0]:.3f} (slope {r2_slope(Xc, Ic)[1]:.2f})"
          f" | {1 - ((Cc - Yc) ** 2).sum() / (Yc ** 2).sum():.3f}, {(e['Yint_m'].argmax(1) == col).mean():.4f}")
print("\nintervened runs: where does argmax of s_t (old inputs) go when it is not the old max?  P(= old runner-up position)")
for t in ts:
    e = ev[t]; col = t - 2 - (e["g"] - 1)
    am = e["Yint"].argmax(1); miss = am != col
    # old runner-up of x_0..x_{t-1}: p2(t-1); it can be t-1 itself (not among inputs 0..t-2)
    print(f"t={t}: P(miss) = {miss.mean():.4f};  of the misses, argmax = old runner-up: {np.mean(am[miss] == e['pS'][miss]):.3f}")
