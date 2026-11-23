"""V1. Controlled sequences: recomputation of the investigator's claims 1 (clean entry), 4 (non-additivity), 5 (maximum, runner-up).
Own code, own grids and filler draws (seed 901).  M = 1 throughout; complex tape coordinates (alt, short, h2, h3).
Run:  python -B v1_controlled.py > v1_controlled.txt"""
from v_common import *

rng = np.random.default_rng(901)
VS = np.round(np.arange(-0.5, 1.0, 0.005), 3)
k5 = LEFTM @ W[ORD, 5]
cos7 = lambda a, b: abs(to7(a) @ to7(b)) / np.linalg.norm(to7(a)) / np.linalg.norm(to7(b))
q3 = lambda v: f"{np.nanmedian(v):.3f} ({np.nanquantile(v, .1):.3f}..{np.nanquantile(v, .9):.3f})"
fmt = lambda z: " ".join(f"{abs(v):.2f}@{np.degrees(np.angle(v)):4.0f}" for v in z)
FILL = {"zero": np.zeros((1, 10)), "minus1": -np.ones((1, 10)), "random": -np.abs(rng.standard_normal((16, 10)))}
CONF = [(gM, gS) for gM in range(1, 8) for gS in range(1, 8) if gM != gS]


def sweep(F, gM, gS, S):
    """Write of a test input v at position p0 = max(gM, gS): D (R, V, 10, 4), unit-5 extra pulse u (R, V), p0."""
    p0 = max(gM, gS)
    X = np.repeat(F[:, None, :], len(VS), 1).copy()
    X[..., p0 - gM], X[..., p0 - gS] = 1.0, S
    ref = run(X[:, 0])                                                     # filler at p0
    X[..., p0] = VS
    H = run(X.reshape(-1, 10)).reshape(len(F), len(VS), 10, 16)
    return zc(H) - zc(ref)[:, None], H[:, :, min(p0 + 1, 9), 5] - ref[:, None, min(p0 + 1, 9), 5], p0


print("=== CLAIM 1: one input v < S arriving after both M = 1 and S ===")
for SS, lab in (((0.3, 0.6, 0.9), "investigator's S values"), ((0.45, 0.75), "other S values")):
    for name, F in FILL.items():
        rows, g_, th_, on_, r2_, res_, free, below = [], [], [], [], [], [], {a: [] for a in range(1, 7)}, []
        store = []
        for S in SS:
            for gM, gS in CONF:
                D, u, p0 = sweep(F, gM, gS, S)
                band = (VS > 0.75 * S) & (VS < S)
                for a in range(2, 10 - p0):
                    rows.append(to7(D[:, band, p0 + a] * LAMV ** (-(a - 2))).reshape(-1, 7))
                for a in range(1, 7):
                    if p0 + a + 1 <= 9:
                        d0, d1 = D[:, band, p0 + a], D[:, band, p0 + a + 1]
                        ok = nrm(d1) > 0.3
                        free[a].append((nrm(d1 - LAMV * d0) / np.maximum(nrm(d1), 1e-12))[ok])
                store.append((S, D[:, :, p0 + 2], u))
        Rr = np.concatenate(rows); sv = np.linalg.svd(Rr, full_matrices=False)
        k = from7(sv[2][0]); k = k / k[0].real
        for S, D2, u in store:
            for i in range(len(D2)):
                lo = (VS < S) & (u[i] > 0.05)
                if lo.sum() < 3: continue
                Aall = to7(D2[i]) @ to7(k) / (to7(k) @ to7(k))
                b = np.polyfit(VS[lo], Aall[lo], 1)
                g_.append(b[0]); th_.append(-b[1] / b[0] / S); on_.append(VS[lo].min() / S)
                r2_.append(1 - ((Aall[lo] - np.polyval(b, VS[lo])) ** 2).sum() / ((Aall[lo] - Aall[lo].mean()) ** 2).sum())
                res_.append(np.median(nrm(D2[i, lo] - Aall[lo, None] * k) / nrm(D2[i, lo])))
                j = np.flatnonzero(lo)[0]; iS = np.flatnonzero(VS < S)[-1]
                below.append(nrm(D2[i, max(j - 1, 0)]) / nrm(D2[i, iS]))
        print(f" {lab}, filler {name:7s}: n = {len(Rr)}  share on component 1: {sv[1][0] ** 2 / (sv[1] ** 2).sum():.3f}  cos with k5 {cos7(k, k5):.3f}  pattern: {fmt(k)}")
        print(f"      slope g {q3(g_)}   Theta/S (line intercept) {q3(th_)}   onset of unit 5 / S {q3(on_)}   R2 line {q3(r2_)}   residual off pattern {q3(res_)}")
        print(f"      write just below onset / write at S-: {q3(below)}")
        print("      free evolution err(a -> a+1) median/90%/share<0.1: " + "  ".join(
            f"a={a}: {np.median(np.concatenate(v)):.3f}/{np.quantile(np.concatenate(v), .9):.3f}/{np.mean(np.concatenate(v) < 0.1):.2f}" for a, v in free.items() if v))
        if lab.startswith("inv") and name == "zero": kA = k
print("   Theta/S medians by S (filler 0 / filler -1), from the line fit:")
for S in (0.3, 0.6, 0.9):
    out = []
    for name in ("zero", "minus1"):
        th = []
        for gM, gS in CONF:
            D, u, p0 = sweep(FILL[name], gM, gS, S)
            lo = (VS < S) & (u[0] > 0.05)
            Aall = to7(D[0, :, p0 + 2]) @ to7(kA) / (to7(kA) @ to7(kA)); b = np.polyfit(VS[lo], Aall[lo], 1); th.append(-b[1] / b[0] / S)
        out.append(f"{np.median(th):.2f} (min {np.min(th):.2f}, max {np.max(th):.2f})")
    print(f"     S = {S}: " + " / ".join(out))
print("   k5 = l.W[o,5] scaled to alt = 1:", fmt(k5 / k5[0].real), "   measured kA:", fmt(kA))
np.savez(HERE / "v1_patterns.npz", kA=to7(kA))

print("\n=== CLAIM 5: the maximum alone (1 at position 0) ===")
for name, F in FILL.items():
    X = F.copy(); X[:, 0] = 1.0; H = run(X); c = zc(H)
    err = [np.median(nrm(c[:, a + 1] - LAMV * c[:, a]) / nrm(c[:, a + 1])) for a in range(9)]
    print(f" filler {name:7s} |tape| at ages 0..9: " + " ".join(f"{v:.2f}" for v in nrm(c).mean(0)) + "\n      free-evolution err a -> a+1, a = 0..8: " + " ".join(f"{e:.3f}" for e in err))
    if name == "zero":
        PM = c[0, 2]; pul = LAMV * H[0, 0, 4] + k5 * H[0, 1, 5] + (LEFTM @ W[ORD, 9]) * H[0, 1, 9]
        print(f"      pulses: unit 4 at age 0 {H[0, 0, 4]:.2f}, unit 5 at age 1 {H[0, 1, 5]:.2f}, unit 9 at age 1 {H[0, 1, 9]:.2f};  cos(age-2 tape, kA) {cos7(PM, kA):.3f};"
              f"  |tape - three pulses with fixed patterns| / |tape| = {np.linalg.norm(to7(PM - pul)) / np.linalg.norm(to7(PM)):.3f}")
        Z_ = run(X, return_pre=True)[1][0]
        print("      non-pulse units with negative pre-activation at steps 1..9 (the clips that break free evolution):", [[ORD[j] for j in np.flatnonzero(Z_[s, ORD] < 0)] for s in range(1, 10)])
print(" the runner-up S placed g steps after the maximum (filler 0), write = tape(1, S) - tape(1) at its age 2: A/S on kA, residual share of the norm")
for g in range(1, 8):
    row = []
    for S in (0.1, 0.3, 0.6, 0.9):
        X = np.zeros((2, 10)); X[:, 0] = 1.0; X[0, g] = S; c = zc(run(X)); D = c[0, g + 2] - c[1, g + 2]
        A = to7(D) @ to7(kA) / (to7(kA) @ to7(kA)); row.append(f"S={S}: {A / S:6.2f} {nrm(D - A * kA) / nrm(D):.2f}")
    print(f"      g = {g}:  " + "   ".join(row))

print("\n=== CLAIM 4: additivity ===")
tp = lambda X: zc(run(X)[:, 9])
print(" (a) two inputs below S (0.8, 0.9, 0.99 S), both after M and S at positions 0, 1; NA = |c12 - c1 - c2 + c0| / min(|c1 - c0|, |c2 - c0|) at step 9")
for name, F in FILL.items():
    out = {}
    for pM, pS in ((0, 1), (1, 0)):
        for S in (0.3, 0.6, 0.9):
            base = F.copy(); base[:, pM], base[:, pS] = 1.0, S; c0 = tp(base)
            for p1 in range(2, 8):
                for p2 in range(p1 + 1, 8):
                    for f1 in (0.8, 0.9, 0.99):
                        for f2 in (0.8, 0.9, 0.99):
                            X1, X2, X12 = base.copy(), base.copy(), base.copy()
                            X1[:, p1] = f1 * S; X2[:, p2] = f2 * S; X12[:, p1], X12[:, p2] = f1 * S, f2 * S
                            c1, c2, c12 = tp(X1), tp(X2), tp(X12); sm = np.minimum(nrm(c1 - c0), nrm(c2 - c0)); ok = sm > 0.3
                            out.setdefault(min(p2 - p1, 3), []).append((nrm(c12 - c1 - c2 + c0) / np.maximum(sm, 1e-9))[ok])
    print(f"     filler {name:7s} " + "   ".join(f"gap {k}{'+' if k == 3 else ''}: {q3(np.concatenate(v))} min {np.concatenate(v).min():.2f} n = {len(np.concatenate(v))}" for k, v in sorted(out.items())))
print(" (b) threshold of a second input after a first one v1 = f S (filler 0): Theta_2 from the line fit on kA, all (pM, pS) in ((0,1),(1,0)), S in 0.3, 0.6, 0.9")
V2 = np.round(np.arange(0.0, 1.0, 0.005), 3)


def theta2(pM, pS, S, p1, v1, p2):
    X = np.zeros((len(V2), 10)); X[:, pM], X[:, pS] = 1.0, S
    if v1 is not None: X[:, p1] = v1
    ref = run(X[:1]); X[:, p2] = V2; H = run(X)
    D = zc(H[:, p2 + 2]) - zc(ref[:, p2 + 2]); A = to7(D) @ to7(kA) / (to7(kA) @ to7(kA)); u = H[:, p2 + 1, 5] - ref[:, p2 + 1, 5]
    lo = (V2 < S) & (u > 0.05)
    if lo.sum() < 3: return np.nan, np.nan
    b = np.polyfit(V2[lo], A[lo], 1); return -b[1] / b[0] / S, b[0]


rec = []
for S in (0.3, 0.6, 0.9):
    for pM, pS in ((0, 1), (1, 0)):
        for p1 in range(2, 7):
            for p2 in range(p1 + 1, 8):
                t0 = theta2(pM, pS, S, p1, None, p2)[0]
                for f in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99):
                    t2, g = theta2(pM, pS, S, p1, f * S, p2); rec.append((f, t0, t2, g, p2 - p1))
rec = np.array(rec)
for f in (0.8, 0.9, 0.99):
    s = rec[:, 0] == f
    print(f"     f = {f}: Theta_2/S {q3(rec[s, 2])}   Theta_0/S median {np.nanmedian(rec[s, 1]):.3f}   slope {np.nanmedian(rec[s, 3]):.1f}")
ok = np.isfinite(rec[:, 1]) & np.isfinite(rec[:, 2]); e = rec[ok, 0] - rec[ok, 1]; d = rec[ok, 2] - rec[ok, 1]; s = e > 0.02
beta = e[s] @ d[s] / (e[s] @ e[s])
print(f"     Theta_2 = Theta_0 + beta (v1 - Theta_0)_+ : beta = {beta:.3f}, R2 {1 - ((d[s] - beta * e[s]) ** 2).sum() / ((d[s] - d[s].mean()) ** 2).sum():.3f}, n = {s.sum()}, ratio {q3(d[s] / e[s])}")
print("     ratio (Theta_2 - Theta_0) / (v1 - Theta_0) by the gap p2 - p1: " + "  ".join(f"gap {k}: {np.median((d[s] / e[s])[rec[ok][s, 4] == k]):.2f}" for k in range(1, 6)))
print(" (c) maximum 1 and runner-up S alone (filler 0), step 9: |c(M,S) - c(M) - c(S)| / |c(S alone)|  and the same / |c(M,S)|")
for order in ("S after M", "S before M"):
    for S in (0.3, 0.6, 0.9):
        e1, e2, n1 = [], [], []
        for p1 in range(0, 7):
            for p2 in range(p1 + 1, 8):
                pM, pS = (p1, p2) if order == "S after M" else (p2, p1)
                X = np.zeros((3, 10)); X[0, pM] = 1; X[1, pS] = S; X[2, pM], X[2, pS] = 1, S; c = tp(X)
                e1.append(nrm(c[2] - c[0] - c[1]) / nrm(c[1])); e2.append(nrm(c[2] - c[0] - c[1]) / nrm(c[2])); n1.append(nrm(c[1]) / nrm(c[2]))
        print(f"     {order}, S = {S}: / |c(S)| {q3(e1)}   / |c(M,S)| {q3(e2)}   |c(S alone)| / |c(M,S)| median {np.median(n1):.3f}")
