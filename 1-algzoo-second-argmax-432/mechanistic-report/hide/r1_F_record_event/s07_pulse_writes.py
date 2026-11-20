"""s07 (task F2). What the three pulses write, in mode coordinates, and the net change of every mode
coordinate from just before a record (s-1) to two and three steps after it (s+2, s+3).

Records at s = 2..6, M0 > 0, following inputs low (L1: x_{s+1..s+3} < M0).  Seeds 1050, 1051 (400k each).

Net change relative to the free resting flow:   N_k(d) = m_k(s+d) - lam_k^(d+1) m_k(s-1),   d = 2, 3.
Exact split of h_{s+2}[ORD] by source, each propagated through the ACTUAL gates (D = 0/1 gate matrices on ORD):
   T4   = D2 A D1 K4 h4(s)                      unit 4's write            (K_p = W[ORD, p])
   T5   = D2 K5 h5(s+1),  T9 = D2 K9 h9(s+1)    the writes of units 5 and 9
   Told = D2 A D1 A D0 (A h_{s-1}[ORD] + K h_{s-1}[PULSE])   the old state (comparators clipped at s inside D0)
   Tps  = D2 A D1 (K5 h5(s) + K9 h9(s))         pulses 5, 9 that were already active at the record step
   Tx   = inputs x_s, x_{s+1}, x_{s+2} entering the ordinary units directly
Prediction 'pulses only, no gates':  lam_k h4 + c5_k h5 + c9_k h9  with c_p,k = LEFT_k . K_p  and the fitted activations.
"""
from fcommon import *

Kp = {p: W[ORD, p] for p in PULSE}
cP = {p: {n: LEFT[n] @ Kp[p] for n in NAMES} for p in PULSE}
print("=== [EXACT] mode coordinates of the pulse writes, c_p,k = LEFT_k . W[ORD, p]  (complex: |.|@deg) ===")
print(f"  {'mode':7s} {'lam':>16s} {'unit 4':>16s} {'unit 5':>16s} {'unit 9':>16s} | unit 5 and 9 without their unit-0 entry | LEFT_k[unit 0]")
i0 = OI[0]
for n in NAMES:
    f = (lambda c: fmtc(c)) if IS_COMPLEX[n] else (lambda c: f"{np.real(c):8.3f}      ")
    c5n, c9n = cP[5][n] - LEFT[n][i0] * Kp[5][i0], cP[9][n] - LEFT[n][i0] * Kp[9][i0]
    print(f"  {n:7s} {f(LAM[n]):>16s} {f(cP[4][n]):>16s} {f(cP[5][n]):>16s} {f(cP[9][n]):>16s} | {f(c5n):>16s} {f(c9n):>16s} | {f(LEFT[n][i0]):>16s}")
print("  W[ORD,5] =", np.round(Kp[5], 3), "\n  W[ORD,9] =", np.round(Kp[9], 3), "  (ORD =", ORD, ")")

acc = {}
def put(k, v):
    acc.setdefault(k, []).append(v)

for seed in (1050, 1051):
    x = sample(400_000, seed)
    H, Z = run(x, return_pre=True)
    st = running_stats(x)
    for s in range(2, 7):
        rec, q = records(x, s, st)
        ok = rec & (x[:, s + 1:s + 4] < q["M0"][:, None]).all(1)
        i = np.flatnonzero(ok)
        Hm, H0, H1, H2, H3 = (H[i, s + d] for d in (-1, 0, 1, 2, 3))
        D0, D1, D2, D3 = ((Z[i, s + d][:, ORD] > 0).astype(float) for d in (0, 1, 2, 3))
        step = lambda v, D: (v @ A.T) * D
        old0 = (Hm[:, ORD] @ A.T + Hm[:, PULSE] @ W[np.ix_(ORD, PULSE)].T) * D0
        x0, x1, x2, x3 = (x[i, s + d, None] * w[ORD] for d in (0, 1, 2, 3))
        k4 = H0[:, 4, None] * Kp[4] * D1
        kps = (H0[:, 5, None] * Kp[5] + H0[:, 9, None] * Kp[9]) * D1
        parts2 = {"T4": step(k4, D2), "T5": H1[:, 5, None] * Kp[5] * D2, "T9": H1[:, 9, None] * Kp[9] * D2,
                  "Told": step(step(old0, D1), D2), "Tps": step(kps, D2),
                  "Tx": step(step(x0 * D0, D1), D2) + step(x1 * D1, D2) + x2 * D2,
                  "T4late": H1[:, 4, None] * Kp[4] * D2}
        assert np.abs(sum(parts2.values()) - H2[:, ORD]).max() < 1e-8
        late3 = (H2[:, PULSE] @ W[np.ix_(ORD, PULSE)].T) * D3
        parts3 = {k: step(v, D3) for k, v in parts2.items()}
        parts3["Tx"] = parts3["Tx"] + x3 * D3
        parts3["Tlate"] = late3
        assert np.abs(sum(parts3.values()) - H3[:, ORD]).max() < 1e-8
        for nm, v in (("Mn", q["Mn"][i]), ("M0", q["M0"][i]), ("S0p", q["S0p"][i]), ("g", q["g"][i]),
                      ("h4", H0[:, 4]), ("h5", H1[:, 5]), ("h9", H1[:, 9]), ("mprev", Hm[:, ORD]),
                      ("m2", H2[:, ORD]), ("m3", H3[:, ORD])):
            put(nm, v)
        for k, v in parts2.items():
            put("2" + k, v)
        for k, v in parts3.items():
            put("3" + k, v)
R = {k: np.concatenate(v) for k, v in acc.items()}
n = len(R["g"])
X = [R["Mn"] - R["M0"], R["M0"], R["S0p"]]
print(f"\nrecords used: n = {n}")
fit = {}
for pu in ("h4", "h5", "h9"):
    b, r2 = ols(R[pu], X)
    fit[pu] = b
    print(f"[FIT] {pu} ~ {b[0]:.3f} (Mn-M0) + {b[1]:.3f} M0 + {b[2]:.3f} S0+   R2 = {r2:.4f}")


def creg(y):
    """regress (complex or real) y on X; returns coefs (3,), uncentred R2."""
    b, r2 = cols_ols(y, X)
    return b, r2


def show(c, n):
    return fmtc(c) if IS_COMPLEX[n] else f"{np.real(c):7.2f}      "


for d, lab in ((2, "s+2"), (3, "s+3")):
    print(f"\n=== net change of the mode coordinates at {lab} relative to free flow, N_k = m_k({lab}) - lam^{d+1} m_k(s-1), "
          f"as a (Mn-M0) + b M0 + c S0+  (all g pooled) ===")
    for vi, vn in enumerate(("(Mn-M0)", "M0", "S0+")):
        print(f"  --- coefficient of {vn} ---")
        print(f"  {'mode':7s} {'pulses only':>14s} {'measured N_k':>14s} (R2)  | exact parts through the actual gates: {'T4':>14s} {'T5':>14s} {'T9':>14s} "
              f"{'Told - free':>14s} {'Tps':>14s} {'Tx':>14s}" + ("" if d == 2 else f" {'Tlate':>14s}"))
        for nme in NAMES:
            L = LEFT[nme]
            lam = LAM[nme]
            pred = lam ** (d - 1) * (fit["h4"][vi] + 0) * 1.0 + lam ** (d - 2) * (cP[5][nme] * fit["h5"][vi] + cP[9][nme] * fit["h9"][vi])
            mfin = R[f"m{d}"] @ L
            free = lam ** (d + 1) * (R["mprev"] @ L)
            b, r2 = creg(mfin - free)
            row = f"  {nme:7s} {show(pred, nme):>14s} {show(b[vi], nme):>14s} ({r2:.2f}) |" + " " * 38
            for part in ("T4", "T5", "T9"):
                row += f" {show(creg(R[f'{d}{part}'] @ L)[0][vi], nme):>14s}"
            row += f" {show(creg(R[f'{d}Told'] @ L - free)[0][vi], nme):>14s}"
            row += f" {show(creg(R[f'{d}Tps'] @ L)[0][vi], nme):>14s}"
            row += f" {show(creg((R[f'{d}Tx'] + R[f'{d}T4late']) @ L)[0][vi], nme):>14s}"
            if d == 3:
                row += f" {show(creg(R['3Tlate'] @ L)[0][vi], nme):>14s}"
            print(row)

print("\n=== the state after the record: m_k(s+2)/10.157 and m_k(s+3)/10.157 ~ a (Mn-M0) + b M0 + c S0+ by g (complex LSQ) ===")
for d in (2, 3):
    for nme in NAMES:
        L = LEFT[nme]
        print(f"  mode {nme}, step s+{d}:")
        for g in (1, 2, 3, 4, 5):
            m = R["g"] == g
            y = (R[f"m{d}"][m] @ L) / w[4]
            Xg = np.column_stack([v[m] for v in X]).astype(complex)
            b, *_ = np.linalg.lstsq(Xg, y.astype(complex), rcond=None)
            r2 = 1 - (np.abs(y - Xg @ b) ** 2).sum() / (np.abs(y) ** 2).sum()
            yb = (R["mprev"][m] @ L) / w[4]
            Xb = np.column_stack([R["M0"][m], R["S0p"][m]]).astype(complex)
            bb, *_ = np.linalg.lstsq(Xb, yb.astype(complex), rcond=None)
            r2b = 1 - (np.abs(yb - Xb @ bb) ** 2).sum() / (np.abs(yb) ** 2).sum()
            print(f"     g={g}: after: {show(b[0], nme)} (Mn-M0)  {show(b[1], nme)} M0  {show(b[2], nme)} S0+  (R2 {r2:.3f})"
                  f"   | before (s-1): {show(bb[0], nme)} M0  {show(bb[1], nme)} S0+ (R2 {r2b:.3f})")
