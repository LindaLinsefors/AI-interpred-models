"""Verifier V5, part J: how much GAP-PROPORTIONAL signal reaches the final logits, and through which modes?

At t = 9, sequences with S > 0.2, age of the max a1 >= 2, age of the runner-up a2 >= 1 (so x_9 is neither of the two):
    D_k = (contribution of part k to logit[p2]) - (contribution of part k to logit[p1])
    least squares  D_k ~ a_k (M - S) + b_k S + c_k x_9   (no intercept), pooled and by separation |p1 - p2|.
If the readout were blind to the clock, a_clock would be 0.  The parts add up exactly: sum_k D_k = logit[p2] - logit[p1].
Seeds 5070-5073 (4 x 200k sequences).   Run:  python -B v5_j_gap_slope_by_mode.py > v5_j_gap_slope_by_mode.txt
"""
from v5_lib import *

rows = []
for seed in [5070, 5071, 5072, 5073]:
    x, H, Z, st, m = batch(seed)
    h9 = H[:, 9]
    C = {}
    for n in NAMES:
        R = Uo @ RIGHT[n]
        C[n] = 2 * np.real(m[n][:, 9, None] * R[None, :]) if IS_COMPLEX[n] else np.real(m[n][:, 9, None] * R[None, :])
    C["pulses"] = h9[:, PULSE] @ U[:, PULSE].T
    zc = m["clock"][:, 9]
    C["clock_comp"] = 2 * np.real(zc[:, None] * (U[:, COMP] @ RIGHT["clock"][CI])[None, :])
    C["clock_wave"] = 2 * np.real(zc[:, None] * (U[:, WAVE] @ RIGHT["clock"][WI])[None, :])
    C["all_comp_units"] = h9[:, COMP] @ U[:, COMP].T          # V of bundle V2
    C["all_wave_units"] = h9[:, WAVE] @ U[:, WAVE].T
    C["TOTAL"] = h9 @ U.T
    pM, pS = st["pM"][:, 9], st["pS"][:, 9]
    idx = np.arange(len(pM))
    sel = (st["S"][:, 9] > 0.2) & (st["a1"][:, 9] >= 2) & (st["a2"][:, 9] >= 1)
    rows.append(dict(D={k: (v[idx, pS] - v[idx, pM])[sel] for k, v in C.items()}, M=st["M"][sel, 9], S=st["S"][sel, 9], x9=x[sel, 9],
                     sep=np.abs(pM - pS)[sel], a1=st["a1"][sel, 9]))
cat = lambda k: np.concatenate([r[k] for r in rows])
M, S, x9, sep, a1 = (cat(k) for k in ("M", "S", "x9", "sep", "a1"))
Dk = {k: np.concatenate([r["D"][k] for r in rows]) for k in rows[0]["D"]}
X = np.stack([M - S, S, x9], 1)
order = ["TOTAL", "all_comp_units", "all_wave_units", "pulses"] + list(NAMES) + ["clock_comp", "clock_wave"]
print(f"seeds 5070-5073, 200k sequences each; selected sequences: {len(M)}")
print("coefficient a_k of the gap (M - S) in  D_k = part of logit[p2] - logit[p1]   [and b_k (S), c_k (x_9), centred R2 of the 3-regressor fit]")
print(f"   {'part':15s}  {'pooled a_k':>10s} {'b_k':>8s} {'c_k':>8s} {'R2c':>6s}   |  a_k by separation |p1 - p2| = 1, 2, 3, 4, 5, 6")
for k in order:
    c, ru, rc = lsq(X, Dk[k])
    out = f"   {k:15s}  {c[0]:10.2f} {c[1]:8.2f} {c[2]:8.2f} {rc:6.3f}   | "
    for d in range(1, 7):
        s = sep == d
        cd, _, _ = lsq(X[s], Dk[k][s])
        out += f" {cd[0]:8.2f}"
    print(out)
print("   check: sum over modes + pulses of pooled a_k =", round(sum(lsq(X, Dk[k])[0][0] for k in list(NAMES) + ["pulses"]), 2))
print("\n   the same by age of the max a1 (pooled over separations): a_k for TOTAL, clock, short, h2, h3, pulses, clock_comp, clock_wave")
for A1 in range(2, 10):
    s = a1 == A1
    print(f"      a1={A1}: " + "  ".join(f"{k} {lsq(X[s], Dk[k][s])[0][0]:7.2f}" for k in ("TOTAL", "clock", "short", "h2", "h3", "pulses", "clock_comp", "clock_wave")))
