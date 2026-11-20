"""Verifier V5, part G: does the readout see the AGE of the max through other modes than the clock?

 G1  the other modes fitted like the clock: coordinate/w4 ~ alpha(a1) (M - S) + beta(a1) S per age (S > 0.05, pooled over a2)
 G2  which modes separate the runner-up from the max in the final logits?  logit[p2] - logit[p1] at t = 9, split by mode
 G3  does the third largest value enter the clock radius?
 G4  phase spread on resting vs non-resting states
Seeds 5050, 5051 (2 x 200k sequences).   Run:  python -B v5_g_harmonics_margin.py > v5_g_harmonics_margin.txt
"""
from v5_lib import *

SEEDS_G = [5050, 5051]
acc = {}
def add(k, a):
    acc.setdefault(k, []).append(a)
fin = []
for seed in SEEDS_G:
    x, H, Z, st, m = batch(seed)
    rest = (Z[:, :, ORD] > 0).all(-1) & (Z[:, :, PULSE] <= 0).all(-1)
    for k, a in (("M", st["M"]), ("S", st["S"]), ("T3", st["T3"]), ("a1", st["a1"]), ("a2", st["a2"]), ("rest", rest)):
        add(k, a.reshape(-1))
    for n in NAMES:
        add("m_" + n, (m[n] / W4).reshape(-1))
    h9 = H[:, 9]
    C = {}
    for n in NAMES:
        R = Uo @ RIGHT[n]
        C[n] = 2 * np.real(m[n][:, 9, None] * R[None, :]) if IS_COMPLEX[n] else np.real(m[n][:, 9, None] * R[None, :])
    C["pulses"] = h9[:, PULSE] @ U[:, PULSE].T
    zc = m["clock"][:, 9]
    C["clock_comp"] = 2 * np.real(zc[:, None] * (U[:, COMP] @ RIGHT["clock"][CI])[None, :])
    C["clock_wave"] = 2 * np.real(zc[:, None] * (U[:, WAVE] @ RIGHT["clock"][WI])[None, :])
    fin.append(dict(C=C, lg=h9 @ U.T, pM=st["pM"][:, 9], pS=st["pS"][:, 9], pT=st["pT"][:, 9], M=st["M"][:, 9], S=st["S"][:, 9], T3=st["T3"][:, 9]))
D = {k: np.concatenate(v) for k, v in acc.items()}
M, S, T3, a1, a2, rest = (D[k] for k in ("M", "S", "T3", "a1", "a2", "rest"))
ok = S > 0.05
X = np.stack([M - S, S], 1)
print(f"seeds {SEEDS_G}, {NB} sequences each")

print("=" * 120)
print("G1  every mode fitted like the clock, per age a1 (S > 0.05, pooled over a2):  coordinate/w4 ~ alpha (M - S) + beta S   (complex LS for pairs)")
print("    entries: |alpha|@phase, |beta|, uncentred R2 / centred R2")
for n in NAMES:
    y = D["m_" + n]
    print(f"   {n} (|lam| = {abs(LAM[n]):.3f}, angle = {np.degrees(np.angle(LAM[n])):.1f} deg; rms |coord|/w4 = {np.sqrt(np.mean(np.abs(y[ok & (a1 >= 1)]) ** 2)):.3f})")
    phs = []
    for A1 in range(1, 10):
        s = ok & (a1 == A1)
        c, ru, rc = lsq(X[s], y[s])
        phs.append(np.angle(c[0]))
        if IS_COMPLEX[n]:
            print(f"      a1={A1}: alpha {pol(c[0])}  beta {pol(c[1])}  R2u {ru:.3f} / R2c {rc:.3f}")
        else:
            print(f"      a1={A1}: alpha {c[0].real:+.3f}  beta {c[1].real:+.3f}  R2u {ru:.3f} / R2c {rc:.3f}")
    if IS_COMPLEX[n]:
        print(f"      phase steps of alpha (deg): {np.round(np.degrees(wrap(np.diff(phs))), 1)}")

print("=" * 120)
print("G2  final logits: logit[p2] - logit[p1] (runner-up minus max) split by mode; sequences with S > 0.2, age of the max a1 >= 2 at t = 9, p2 < 9")
cat = lambda k: np.concatenate([f[k] for f in fin])
pM, pS, pT, Mf, Sf, Tf, lg = (cat(k) for k in ("pM", "pS", "pT", "M", "S", "T3", "lg"))
C = {n: np.concatenate([f["C"][n] for f in fin]) for n in fin[0]["C"]}
idx = np.arange(len(pM))
base = (Sf > 0.2) & (pM <= 7) & (pS < 9)
gr = (Mf - Sf) / Mf
parts = list(NAMES) + ["pulses", "clock_comp", "clock_wave"]
print("   medians in units of M.   gap/M bin:        " + "   ".join(f"{b!s:>12s}" for b in ((0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8))))
tot_rows = {}
for n in ["TOTAL"] + parts:
    out = f"   {n:11s}"
    for lo, hi in ((0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8)):
        s = base & (gr > lo) & (gr <= hi)
        src = lg if n == "TOTAL" else C[n]
        d = (src[idx, pS] - src[idx, pM])[s] / Mf[s]
        out += f"   {np.median(d):8.2f} (mean {d.mean():7.2f})"
    print(out)
print("   same for logit[p2] - logit[p3] (runner-up minus third), sequences with T > 0.2, p2 < 9, p3 < 9, binned by (S - T)/S:")
base3 = (Tf > 0.2) & (pS < 9) & (pT < 9)
g3 = (Sf - Tf) / Sf
print("   medians in units of S.   (S-T)/S bin:      " + "   ".join(f"{b!s:>12s}" for b in ((0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8))))
for n in ["TOTAL"] + parts:
    out = f"   {n:11s}"
    for lo, hi in ((0.05, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8)):
        s = base3 & (g3 > lo) & (g3 <= hi)
        src = lg if n == "TOTAL" else C[n]
        d = (src[idx, pS] - src[idx, pT])[s] / Sf[s]
        out += f"   {np.median(d):8.2f} (mean {d.mean():7.2f})"
    print(out)

print("=" * 120)
print("G3  third largest value in the clock law?  z/w4 ~ alpha (M - S) + beta S + delta max(T, 0), per a1 (S > 0.05, pooled over a2; states where a third value exists)")
z = D["m_clock"]
for A1 in range(1, 10):
    s = ok & (a1 == A1) & np.isfinite(T3)
    X3 = np.stack([M[s] - S[s], S[s], np.maximum(T3[s], 0)], 1)
    c, ru, rc = lsq(X3, z[s])
    c2, ru2, rc2 = lsq(X3[:, :2], z[s])
    print(f"   a1={A1}: alpha {pol(c[0])}  beta {pol(c[1])}  delta {pol(c[2])}   R2u {ru:.4f} (without T: {ru2:.4f})   R2c {rc:.4f} (without T: {rc2:.4f})")

print("=" * 120)
print("G4  phase spread |arg z - phi(a1)| on resting vs non-resting states (M > 0.3, relative gap (M - max(S,0))/M > 0.3); phi from the pooled fit of this script")
r = (M - np.maximum(S, 0)) / np.where(M > 0, M, np.nan)
for A1 in range(1, 10):
    s = ok & (a1 == A1)
    c, _, _ = lsq(X[s], z[s])
    s0 = (a1 == A1) & (M > 0.3) & (r > 0.3)
    out = f"   a1={A1}: frac resting {rest[s0].mean():.3f}"
    for nm, sub in (("resting", s0 & rest), ("non-resting", s0 & ~rest)):
        if sub.sum() > 300:
            d = np.degrees(wrap(np.angle(z[sub]) - np.angle(c[0])))
            out += f" | {nm}: median delta {np.median(d):+5.1f}, median |delta| {np.median(np.abs(d)):4.1f}, 90th pct {np.percentile(np.abs(d), 90):5.1f}, median |z|/(w4 (M-S+)) {np.median(np.abs(z[sub]) / (M[sub] - np.maximum(S[sub], 0))):.3f}"
    print(out)
