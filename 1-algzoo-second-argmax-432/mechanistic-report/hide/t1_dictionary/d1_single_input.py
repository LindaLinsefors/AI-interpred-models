"""D1. Dictionary entry of ONE input that arrives after both the maximum and the runner-up.

Controlled sequences (M = 1): the test input of value v sits at position p0 = max(gM, gS); the maximum 1 arrived gM
steps before it, the runner-up S arrived gS steps before it; every other input (between and after) is a filler:
"zero" (0), "minus1" (-1) or "random" (-|N(0,1)|, 16 draws).  Nothing precedes the older of M and S.
Contribution of the test input at age a:  D(a) = tape(sequence) - tape(same sequence with the filler in its place),
read at step p0 + a, in complex tape coordinates (alt, short, h2, h3), units of M.
Run:  python -B d1_single_input.py > d1_single_input.txt"""
from dict_common import *

rng = np.random.default_rng(1)
VS = np.round(np.arange(-0.5, 1.0, 0.01), 2)
SS = (0.3, 0.6, 0.9)
COMBOS = [(gM, gS) for gM in range(1, 8) for gS in range(1, 8) if gM != gS]
FILLS = {"zero": np.zeros((1, 10)), "minus1": -np.ones((1, 10)), "random": -np.abs(rng.standard_normal((16, 10)))}

res = {}
for name, F in FILLS.items():
    for S in SS:
        for gM, gS in COMBOS:
            p0 = max(gM, gS)
            X = np.repeat(F[:, None, :], len(VS), 1)                      # (R, V, 10)
            X[..., :p0 - max(gM, gS)] = 0.0                               # nothing before the older of M, S (no-op: p0 = max)
            X[..., p0 - gM], X[..., p0 - gS] = 1.0, S
            Xr = X[:, :1].copy()                                          # reference: filler at p0
            X[..., p0] = VS
            H = run(X.reshape(-1, 10)).reshape(*X.shape[:2], 10, 16)
            Hr = run(Xr.reshape(-1, 10)).reshape(len(F), 1, 10, 16)
            res[name, S, gM, gS] = dict(D=zc(H) - zc(Hr), u=H[:, :, p0 + 1, 5] - Hr[:, :, p0 + 1, 5], p0=p0,
                                        d4=H[:, :, p0, 4], comp_on=(H[:, :, p0][..., COMP] > 0).sum(-1))

nrm = lambda D: np.sqrt((np.abs(D) ** 2).sum(-1))                        # 7-real norm of a complex 4-vector with real alt
first = lambda mask: np.where(mask.any(-1), mask.argmax(-1), -1)         # first True along the value axis, -1 if none

print("1. THRESHOLD.  n(v) = |D(v)| / |D(S - 0.01)| at age 2.  v_10 = smallest v with n > 0.1;  v_on = smallest v at which")
print("   unit 5's pulse one step after the input exceeds its value in the reference by 0.05.  Both in units of S.  Only v < S.")
for name in FILLS:
    for S in SS:
        iS = int(np.argmin(np.abs(VS - (S - 0.01))))
        v10, von, leak = [], [], []
        for gM, gS in COMBOS:
            r = res[name, S, gM, gS]; n_ = nrm(r["D"][:, :, r["p0"] + 2]); n_ = n_ / n_[:, iS:iS + 1]
            i10 = first(n_[:, :iS + 1] > 0.1); ion = first(r["u"][:, :iS + 1] > 0.05)
            v10.append(VS[i10] / S); von.append(np.where(ion >= 0, VS[ion] / S, np.nan))
            leak.append(n_[np.arange(len(n_)), np.maximum(ion - 1, 0)])
        v10, von, leak = np.array(v10), np.array(von), np.array(leak)
        print(f"   filler {name:7s} S = {S}:  v_10/S median {np.median(v10):.2f}  (5%..95%: {np.quantile(v10, .05):.2f}..{np.quantile(v10, .95):.2f}) | "
              f"v_on/S median {np.nanmedian(von):.2f} ({np.nanquantile(von, .05):.2f}..{np.nanquantile(von, .95):.2f}), never fires {np.isnan(von).mean():.2f} | "
              f"n just below v_on: median {np.median(leak):.3f}, max {leak.max():.3f}")
print("\n   v_on / S for the filler 0 and S = 0.6, rows gM = 1..7 (steps since the maximum at arrival), columns gS = 1..7 (steps since the runner-up)")
for S in SS:
    print(f"   S = {S}")
    for gM in range(1, 8):
        row = []
        for gS in range(1, 8):
            if gM == gS: row.append("    . "); continue
            r = res["zero", S, gM, gS]; iS = int(np.argmin(np.abs(VS - (S - 0.01)))); ion = first(r["u"][:, :iS + 1] > 0.05)[0]
            row.append(f"{VS[ion] / S:6.2f}" if ion >= 0 else "  none")
        print("     gM =", gM, " ".join(row))

print("\n2. DIRECTION.  Contribution at age a turned back to age 2:  D(a) lam^-(a-2), a = 2..9-p0, as 7 real numbers.")
print("   Uncentred SVD over all cases: share of the squared norm in the leading components; cosine of component 1 with")
print("   the pattern written by unit 5, k5 = l.W[o,5], and with the pattern of a unit-4 pulse (1,1,1,1) turned forward one step.")
k5 = to7(LEFTM @ W[ORD, 5]); k4 = to7(LAMV * (LEFTM @ W[ORD, 4]))
cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
SV = {}
for name in FILLS:
    for rng_name, sel in (("0.75 S < v < S", lambda S: (VS > 0.75 * S) & (VS < S)), ("S < v < 1", lambda S: (VS > S) & (VS < 1)),
                          ("v < 0.5 S", lambda S: VS < 0.5 * S)):
        rows = []
        for S in SS:
            for gM, gS in COMBOS:
                r = res[name, S, gM, gS]
                for a in range(2, 10 - r["p0"]):
                    rows.append(to7(r["D"][:, sel(S), r["p0"] + a] * LAMV ** (-(a - 2))).reshape(-1, 7))
        Rr = np.concatenate(rows); sv = np.linalg.svd(Rr, full_matrices=False)
        sh = sv[1] ** 2 / (sv[1] ** 2).sum(); SV[name, rng_name] = sv[2][0] * np.sign(sv[2][0] @ k5)
        print(f"   filler {name:7s} {rng_name:15s} n = {len(Rr):7d}  rms |D| = {np.sqrt((Rr ** 2).sum(1).mean()):5.2f}  shares {np.round(sh[:4], 3)}  "
              f"cos(comp 1, k5) = {cos(sv[2][0], k5):.3f}  cos(comp 1, k4 lam) = {cos(sv[2][0], k4):.3f}")
kA = SV["zero", "0.75 S < v < S"]; kB = SV["zero", "S < v < 1"]
print("   component 1 (filler 0), as complex numbers per mode, scaled so that alt = 1:")
print("     0.75 S < v < S :", fmtc(from7(kA) / from7(kA)[0].real), "\n     S < v < 1      :", fmtc(from7(kB) / from7(kB)[0].real),
      "\n     k5 = l.W[o,5]  :", fmtc(from7(k5) / from7(k5)[0].real), f"\n     cosine between the two measured patterns: {cos(kA, kB):.3f}")

print("\n3. FREE EVOLUTION.  Does one more step (a filler) multiply the contribution by the eigenvalues?")
print("   err(a) = |D(a+1) - lam D(a)| / |D(a+1)|, over all cases with |D(a+1)| > 0.3; median, 90th percentile, share with err < 0.1")
for name in FILLS:
    for rng_name, sel in (("0.75 S < v < S", lambda S: (VS > 0.75 * S) & (VS < S)), ("S < v < 1", lambda S: (VS > S) & (VS < 1))):
        out = []
        for a in range(0, 7):
            e = []
            for S in SS:
                for gM, gS in COMBOS:
                    r = res[name, S, gM, gS]; p0 = r["p0"]
                    if p0 + a + 1 > 9: continue
                    D0, D1 = r["D"][:, sel(S), p0 + a], r["D"][:, sel(S), p0 + a + 1]
                    ok = nrm(D1) > 0.3
                    e.append((nrm(D1 - LAMV * D0) / np.maximum(nrm(D1), 1e-12))[ok])
            e = np.concatenate(e)
            out.append(f"a={a}->{a + 1}: {np.median(e):.3f}/{np.quantile(e, .9):.3f}/{np.mean(e < 0.1):.2f}")
        print(f"   filler {name:7s} {rng_name:15s} " + "  ".join(out))

print("\n4. AMPLITUDE.  A(v) = projection of the age-2 contribution on the measured pattern of its value range (section 2, filler 0),")
print("   normalised so that A = the alt coordinate when the pattern fits exactly.  resid = |D - A k| / |D|.")
print("   For v < S, straight-line fit A = g (v - Theta) on v_on <= v < S:  g, Theta / S, R2 of the line; for S < v < 1: slope g2 and resid.")
kAn = kA / kA[0]; kBn = kB / kB[0]
proj = lambda D7, k: (D7 @ k) / (k @ k)
for name in ("zero", "minus1", "random"):
    for S in SS:
        g, th, R2, rs, g2, rs2, Aat = [], [], [], [], [], [], []
        for gM, gS in COMBOS:
            r = res[name, S, gM, gS]; D7 = to7(r["D"][:, :, r["p0"] + 2])            # (R, V, 7)
            for i in range(len(D7)):
                lo = (VS < S) & (r["u"][i] > 0.05)
                if lo.sum() >= 3:
                    A = proj(D7[i, lo], kAn); b = np.polyfit(VS[lo], A, 1)
                    g.append(b[0]); th.append(-b[1] / b[0] / S); R2.append(r2(A, np.polyval(b, VS[lo])))
                    rs.append(np.median(np.linalg.norm(D7[i, lo] - np.outer(A, kAn), axis=1) / np.linalg.norm(D7[i, lo], axis=1)))
                    Aat.append(proj(D7[i, int(np.argmin(np.abs(VS - (S - 0.01))))], kAn))
                hi = (VS > S + 0.02) & (VS < 0.98)
                if hi.sum() >= 3:
                    A = proj(D7[i, hi], kBn); g2.append(np.polyfit(VS[hi], A, 1)[0])
                    rs2.append(np.median(np.linalg.norm(D7[i, hi] - np.outer(A, kBn), axis=1) / np.linalg.norm(D7[i, hi], axis=1)))
        q = lambda v_: f"{np.median(v_):6.2f} ({np.quantile(v_, .1):6.2f}..{np.quantile(v_, .9):6.2f})"
        print(f"   filler {name:7s} S = {S}: g {q(g)}  Theta/S {q(th)}  R2 {q(R2)}  resid {q(rs)}  A(v = S-) {q(Aat)} | g2 {q(g2)}  resid {q(rs2)}")

print("\n5. EXAMPLES (filler 0, S = 0.6): A(v) on the v < S pattern at age 2, and unit 5's extra pulse u, for some (gM, gS)")
ex = [(1, 2), (2, 1), (3, 5), (5, 3), (7, 2), (2, 7)]
vsel = [int(np.argmin(np.abs(VS - v))) for v in (0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.59, 0.65, 0.7, 0.8, 0.9, 0.99)]
print("   v:            " + " ".join(f"{VS[i]:6.2f}" for i in vsel))
for gM, gS in ex:
    r = res["zero", 0.6, gM, gS]; D7 = to7(r["D"][0, :, r["p0"] + 2])
    print(f"   gM={gM} gS={gS}  A: " + " ".join(f"{proj(D7[i], kAn):6.2f}" for i in vsel))
    print(f"              u: " + " ".join(f"{r['u'][0, i]:6.2f}" for i in vsel))
    print(f"   comparators on: " + " ".join(f"{r['comp_on'][0, i]:6d}" for i in vsel) + "    unit 4: " + " ".join(f"{r['d4'][0, i]:5.2f}" for i in vsel[-4:]))
np.savez(HERE / "d1_patterns.npz", kA=kAn, kB=kBn, k5=k5)
