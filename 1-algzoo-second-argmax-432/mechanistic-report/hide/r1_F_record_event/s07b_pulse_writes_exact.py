"""s07b (task F2). Exact 'writes + clipping corrections' identity for the mode coordinates after a record.

Each step:  h_t[ORD] = v_t + r_t,  v_t = A h_{t-1}[ORD] + K h_{t-1}[PULSE] + w[ORD] x_t  (linear),
            r_t = -min(v_t, 0) >= 0  (what the ReLU adds on the units that are off).
With m = LEFT . h[ORD] (LEFT A = lam LEFT) this gives, exactly, for every mode k:

  m_k(s+2) = lam^3 m_k(s-1)                                   free flow of the old coordinate
           + lam h4(s) + c5 h5(s+1) + c9 h9(s+1)              P4, P5, P9: the three pulse writes (c_p = LEFT_k.W[ORD,p])
           + lam^2 LEFT_k.r_s[comparators]                    R0c: the comparators clipped at the record step
           + LEFT_k[unit 0] r_{s+2}[unit 0]                   R2u0: unit 0 clipped at s+2 (its 'reset')
           + other clipping (wave units at s, s+1, s+2; comparators at s+1, s+2)          Rother
           + pulses that were already active (h_{s-1}[PULSE], h5(s), h9(s), h4(s+1))      Pold
           + inputs lam^2 e x_s + lam e x_{s+1} + e x_{s+2},  e = LEFT_k.w[ORD]           X
Records at s = 2..6, M0 > 0, L1 (x_{s+1..s+3} < M0); seeds 1050, 1051.  Each term is regressed on
(Mn-M0, M0, S0+) without intercept (complex LSQ for the rotating modes).
"""
from fcommon import *

KP = W[np.ix_(ORD, PULSE)]                 # (13, 3) columns for units 4, 5, 9
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
        r = [-np.minimum(Z[i, s + d][:, ORD], 0) for d in (0, 1, 2, 3)]          # clipping corrections on ORD
        for nm, v in (("Mn", q["Mn"][i]), ("M0", q["M0"][i]), ("S0p", q["S0p"][i]), ("g", q["g"][i]),
                      ("Hm", Hm), ("H0p", H0[:, PULSE]), ("H1p", H1[:, PULSE]), ("H2p", H2[:, PULSE]),
                      ("m2", H2[:, ORD]), ("m3", H3[:, ORD]), ("r0", r[0]), ("r1", r[1]), ("r2", r[2]), ("r3", r[3]),
                      ("xs", x[i, s:s + 4])):
            put(nm, v)
R = {k: np.concatenate(v) for k, v in acc.items()}
n = len(R["g"])
X = [R["Mn"] - R["M0"], R["M0"], R["S0p"]]
print(f"records used: n = {n}")
iu0 = OI[0]
cmask = np.zeros(13); cmask[CI] = 1


def terms(nme, d):
    """dict of per-record terms of m_k(s+d), d in (2, 3)."""
    L, lam = LEFT[nme], LAM[nme]
    c = L @ KP                                   # (3,) for units 4, 5, 9
    e = L @ w[ORD]
    t = {}
    t["free"] = lam ** 3 * (R["Hm"][:, ORD] @ L)
    t["P4"] = lam * R["H0p"][:, 0] * c[0]
    t["P5"] = R["H1p"][:, 1] * c[1]
    t["P9"] = R["H1p"][:, 2] * c[2]
    t["R0c"] = lam ** 2 * ((R["r0"] * cmask) @ L)
    t["R2u0"] = R["r2"][:, iu0] * L[iu0]
    r2o = R["r2"].copy(); r2o[:, iu0] = 0
    t["Rother"] = lam ** 2 * ((R["r0"] * (1 - cmask)) @ L) + lam * (R["r1"] @ L) + r2o @ L
    t["Pold"] = lam ** 2 * (R["Hm"][:, PULSE] @ c) + lam * (R["H0p"][:, 1] * c[1] + R["H0p"][:, 2] * c[2]) + R["H1p"][:, 0] * c[0]
    t["X"] = e * (lam ** 2 * R["xs"][:, 0] + lam * R["xs"][:, 1] + R["xs"][:, 2])
    tot = sum(t.values())
    assert np.abs(tot - R["m2"] @ L).max() < 1e-7, np.abs(tot - R["m2"] @ L).max()
    if d == 3:
        t = {k: lam * v for k, v in t.items()}
        t["Rother"] = t["Rother"] + R["r3"] @ L
        t["Pold"] = t["Pold"] + R["H2p"] @ c
        t["X"] = t["X"] + e * R["xs"][:, 3]
        assert np.abs(sum(t.values()) - R["m3"] @ L).max() < 1e-7
    return t


def show(c, nme):
    return fmtc(c) if IS_COMPLEX[nme] else f"{np.real(c):7.2f}      "


COLSN = ["P4", "P5", "P9", "R0c", "R2u0", "Rother", "Pold", "X"]
for d in (2, 3):
    for vi, vn in enumerate(("(Mn-M0)", "M0", "S0+")):
        print(f"\n=== step s+{d}: coefficient of {vn} in  N_k = m_k(s+{d}) - lam^{d+1} m_k(s-1)  and in each exact term (all g pooled) ===")
        print(f"  {'mode':7s} {'N_k measured':>14s} (R2)   {'P4+P5+P9':>14s} |" + "".join(f" {c:>14s}" for c in COLSN))
        for nme in NAMES:
            t = terms(nme, d)
            N = sum(v for k, v in t.items() if k != "free")
            b, r2 = cols_ols(N, X)
            bp = cols_ols(t["P4"] + t["P5"] + t["P9"], X)[0]
            row = f"  {nme:7s} {show(b[vi], nme):>14s} ({r2:.2f})   {show(bp[vi], nme):>14s} |"
            for cn in COLSN:
                row += f" {show(cols_ols(t[cn], X)[0][vi], nme):>14s}"
            print(row)

print("\n=== clock mode by g, step s+2 (units of 10.157): old coordinate after free flow, the comparator term R0c, and the result ===")
print("   g | free flow lam^3 z(s-1): M0, S0+ coefs | R0c: (Mn-M0), M0, S0+ coefs | m(s+2): (Mn-M0), M0, S0+ coefs")
for g in range(1, 7):
    m = R["g"] == g
    Xg = [v[m] for v in X]
    t = terms("clock", 2)
    bf = cols_ols(t["free"][m] / w[4], Xg[1:])[0]
    br = cols_ols(t["R0c"][m] / w[4], Xg)[0]
    bt = cols_ols((R["m2"][m] @ LEFT["clock"]) / w[4], Xg)[0]
    print(f"   {g} | {fmtc(bf[0])} {fmtc(bf[1])} | {fmtc(br[0])} {fmtc(br[1])} {fmtc(br[2])} | {fmtc(bt[0])} {fmtc(bt[1])} {fmtc(bt[2])}")
print("\n=== runner mode by g, step s+2 (units of 10.157): free flow, R0c and result ===")
for g in range(1, 7):
    m = R["g"] == g
    Xg = [v[m] for v in X]
    t = terms("runner", 2)
    bf = cols_ols(t["free"][m] / w[4], Xg[1:])[0].real
    br = cols_ols(t["R0c"][m] / w[4], Xg)[0].real
    bt = cols_ols((R["m2"][m] @ LEFT["runner"]) / w[4], Xg)[0].real
    print(f"   {g} | free: {bf[0]:6.2f} M0 {bf[1]:+6.2f} S0+ | R0c: {br[0]:6.2f} (Mn-M0) {br[1]:+6.2f} M0 {br[2]:+6.2f} S0+ | m(s+2): {bt[0]:6.2f} (Mn-M0) {bt[1]:+6.2f} M0 {bt[2]:+6.2f} S0+")
