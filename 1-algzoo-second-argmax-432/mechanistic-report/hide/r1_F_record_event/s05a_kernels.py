"""s05a. Exact logit kernels (Jacobians) around a record.  K[p -> q] = d logit_q / d x_p (gates fixed);
logit_q = sum_p K[p -> q] x_p exactly.

Sequences: record at s with M0 > 0, x_s the final max, the old max (p0) the final runner-up,
the old runner-up (q0) the final third (so every later input is below S0), S0 > 0.
Seed 1020, 400k sequences.
"""
from fcommon import *

x = sample(400_000, 1020)
st = running_stats(x)
H, Z, J = final_jac(x)
lg = H[:, 9] @ U.T
K = J @ U.T                                  # (B, source p, logit q)
B = x.shape[0]
ar = np.arange(B)
order = np.argsort(-x, axis=1)
print("overall accuracy:", (lg.argmax(1) == order[:, 1]).mean())

for s in (3, 5, 7, 8, 9):
    rec, q = records(x, s, st)
    ok = rec & (order[:, 0] == s) & (order[:, 1] == q["p0"]) & (order[:, 2] == q["q0"]) & (q["S0"] > 0)
    for g in (1, 2, 3, 4):
        m = ok & (q["g"] == g)
        if m.sum() < 300:
            continue
        i = np.flatnonzero(m)
        p0, q0 = q["p0"][i], q["q0"][i]
        Kc = K[i] - K[i].mean(2, keepdims=True)          # centred over logit positions
        acc = (lg[i].argmax(1) == p0).mean()
        print(f"\n=== record at s = {s}, old max age g = {g} (p0 = {s-g}): n = {len(i)}, network answers p0 in {acc:.3f} ===")
        src = {"x_s (new max)": np.full(len(i), s), "x_p0 (old max)": p0, "x_q0 (old 2nd)": q0}
        tgt = {"s": np.full(len(i), s), "p0": p0, "q0": q0}
        print("   mean centred kernel d logit[q]/d x_p:      q = s        p0        q0")
        for sn, sp in src.items():
            print(f"     p = {sn:16s}            " + " ".join(f"{Kc[np.arange(len(i)), sp, tq].mean():9.2f}" for tq in tgt.values()))
        # contribution of each source to the margins
        xi = x[i]
        cont = K[i] * xi[:, :, None]                      # (n, p, q)
        def margin(qa, qb):
            return cont[np.arange(len(i)), :, qa] - cont[np.arange(len(i)), :, qb]      # (n, p)
        for lab, qa, qb in (("p0 - q0", p0, q0), ("p0 - s ", p0, np.full(len(i), s))):
            mg = margin(qa, qb)
            parts = [mg[np.arange(len(i)), np.full(len(i), s)], mg[np.arange(len(i)), p0], mg[np.arange(len(i)), q0]]
            rest = mg.sum(1) - sum(parts)
            print(f"   margin logit[{lab}]: mean total {mg.sum(1).mean():7.2f} = from x_s {parts[0].mean():7.2f} + from x_p0 {parts[1].mean():7.2f}"
                  f" + from x_q0 {parts[2].mean():7.2f} + from all other inputs {rest.mean():7.2f}")
        Mn, M0, S0 = q["Mn"][i], q["M0"][i], q["S0"][i]
        for lab, qa, qb in (("p0 - q0", p0, q0), ("p0 - s ", p0, np.full(len(i), s))):
            y = lg[i][np.arange(len(i)), qa] - lg[i][np.arange(len(i)), qb]
            b, r2 = ols(y, [Mn - M0, M0 - S0, S0])
            print(f"   [FIT] logit[{lab}] ~ {b[0]:7.2f} (Mn-M0) + {b[1]:7.2f} (M0-S0) + {b[2]:7.2f} S0    R2 = {r2:.3f}")
