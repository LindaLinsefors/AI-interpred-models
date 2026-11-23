"""D2. Dictionary entries of the maximum and of the runner-up, and additivity.  Controlled sequences, M = 1.
Complex tape coordinates (alt, short, h2, h3).  Patterns kA (an input below the runner-up) and kB (a new runner-up)
come from d1 (age-2 patterns, alt = 1).  Fillers: 0, -1, or 16 draws of -|N(0,1)|.
Run:  python -B d2_max_runner_additivity.py > d2_max_runner_additivity.txt"""
from dict_common import *

rng = np.random.default_rng(2)
pat = np.load(HERE / "d1_patterns.npz"); kA, kB = pat["kA"], pat["kB"]
FILLS = {"zero": np.zeros((1, 10)), "minus1": -np.ones((1, 10)), "random": -np.abs(rng.standard_normal((16, 10)))}
nrm = lambda D: np.sqrt((np.abs(D) ** 2).sum(-1))
cos = lambda a, b: abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b)
q3 = lambda e: f"{np.median(e):.3f} ({np.quantile(e, .1):.3f}..{np.quantile(e, .9):.3f})"


def tapes(F, placed):
    """Tape coordinates at every step, (R, 10, 4), of the sequences: fillers F (R, 10) with {position: value} placed."""
    X = F.copy()
    for p, v in placed.items():
        X[:, p] = v
    return zc(run(X))


print("A. THE MAXIMUM ALONE: 1 at position 0, fillers after.  c(a) = tape at age a.")
k4 = LAMV * 1.0; k5 = LEFTM @ W[ORD, 5]; k9 = LEFTM @ W[ORD, 9]
for name, F in FILLS.items():
    c = tapes(F, {0: 1.0})                                                # (R, 10, 4)
    H = run(np.where(np.arange(10) == 0, 1.0, F))
    err = [np.median(nrm(c[:, a + 1] - LAMV * c[:, a]) / nrm(c[:, a + 1])) for a in range(0, 9)]
    print(f"   filler {name:7s} |c(a)| a=0..9: " + " ".join(f"{v:5.2f}" for v in nrm(c).mean(0)))
    print(f"                  err(a -> a+1) = |c(a+1) - lam c(a)| / |c(a+1)|, a = 0..8 (median over draws): " + " ".join(f"{e:.3f}" for e in err))
    if name == "zero":
        P_M = c[0, 2]
        B = np.stack([to7(k4 * H[0, 0, 4]), to7(k5 * H[0, 1, 5]), to7(k9 * H[0, 1, 9])], 1)
        print(f"     pulses: unit 4 at age 0 = {H[0, 0, 4]:.2f}, unit 5 at age 1 = {H[0, 1, 5]:.2f}, unit 9 at age 1 = {H[0, 1, 9]:.2f}")
        print("     pattern at age 2, P_M (abs@deg):       ", fmtc(P_M))
        print("     from the three pulses with fixed k:    ", fmtc(from7(B.sum(1))), f"  relative difference {np.linalg.norm(to7(P_M) - B.sum(1)) / np.linalg.norm(to7(P_M)):.3f}")
        print("     P_M scaled to alt = 1:                 ", fmtc(P_M / P_M[0].real), f"  cos with kA {cos(to7(P_M), kA):.3f}, with kB {cos(to7(P_M), kB):.3f}")
        print("     unit 4's pulse turned on one step, lam:", fmtc(k4), f"  cos with kA {cos(to7(k4), kA):.3f}")
P_M7 = to7(P_M)

print("\nB. A MAXIMUM THAT OVERTAKES AN EARLIER ONE: r at position 0, 1 at position g, filler 0 after.  Read at age 2 of the new maximum.")
print("   D_new = tape(r, 1) - tape(r alone);  D_old = tape(r, 1) - tape(1 alone);  amplitudes by projection on P_M (the lone maximum's age-2 pattern)")
print("   and the same turned on g steps, P_M lam^g;  resid = share of the norm not on that pattern;  d4 = unit 4's pulse / w_4 = excess over its threshold")
print("     g    r   | D_new: A on P_M  resid   d4   1-r  | D_old: A on P_M lam^g (in units of r)  resid  | |D_old| / |P_M lam^g| r")
F0 = np.zeros((1, 10))
for g in (1, 2, 3, 5, 7):
    for r in (0.2, 0.5, 0.8, 0.95):
        both, old, new = tapes(F0, {0: r, g: 1.0})[0, g + 2], tapes(F0, {0: r})[0, g + 2], tapes(F0, {g: 1.0})[0, g + 2]
        d4 = run(seq(g + 1, {g: r, 0: 1.0}))[0, g, 4] / w[4]
        Dn, Do = to7(both - old), to7(both - new); Pg = to7(P_M * LAMV ** g)
        An, Ao = Dn @ P_M7 / (P_M7 @ P_M7), Do @ Pg / (Pg @ Pg)
        print(f"     {g}  {r:4.2f}  |  {An:7.3f}  {np.linalg.norm(Dn - An * P_M7) / np.linalg.norm(Dn):6.3f}  {d4:5.3f} {1 - r:5.2f} |  {Ao / r:7.3f}  "
              f"{np.linalg.norm(Do - Ao * Pg) / np.linalg.norm(Do):6.3f}  |  {np.linalg.norm(Do) / np.linalg.norm(Pg) / r:6.3f}")

print("\nC. THE RUNNER-UP AFTER THE MAXIMUM: 1 at position 0, S at position g, filler after.  D_S = tape(1, S) - tape(1, filler), at age 2 of S.")
print("   A = projection on kB (alt = 1 pattern of a new runner-up);  resid as above;  u = unit 5's extra pulse one step after S")
for name in ("zero", "minus1"):
    F = FILLS[name]
    print(f"   filler {name}:   g   | S = 0.1: A / S  resid | S = 0.3 | S = 0.6 | S = 0.9 | S = 0.99      (A / S, resid, u / S)")
    for g in range(1, 8):
        row = []
        for S in (0.1, 0.3, 0.6, 0.9, 0.99):
            D = to7(tapes(F, {0: 1.0, g: S})[0, g + 2] - tapes(F, {0: 1.0})[0, g + 2]); A = D @ kB / (kB @ kB)
            Hs = run(np.array([seq(10, {9: 1.0, 9 - g: S}, F[0]), seq(10, {9: 1.0}, F[0])]))
            row.append(f"{A / S:6.2f} {np.linalg.norm(D - A * kB) / np.linalg.norm(D):5.2f} {(Hs[0, g + 1, 5] - Hs[1, g + 1, 5]) / S:6.2f}")
        print(f"                   {g}   | " + " | ".join(row))

print("\nD. ADDITIVITY.  NA = |tape(both) - tape(only first) - tape(only second) + tape(neither)| / (the smaller of the two single contributions),")
print("   single contribution = |tape(only that one) - tape(neither)|; read at the final step 9; both inputs have final age >= 2.")
print("   D1. Two inputs below the runner-up (values v1, v2 in units of S), both after the maximum (position 0 or 1) and the runner-up (the other).")
for name, F in FILLS.items():
    out = {}
    for pM, pS in ((0, 1), (1, 0)):
        for S in (0.3, 0.6, 0.9):
            base = {pM: 1.0, pS: S}
            c0 = tapes(F, base)[:, 9]
            for p1 in range(2, 8):
                for p2 in range(p1 + 1, 8):
                    for f1 in (0.8, 0.9, 0.99):
                        for f2 in (0.8, 0.9, 0.99):
                            c1, c2 = tapes(F, {**base, p1: f1 * S})[:, 9], tapes(F, {**base, p2: f2 * S})[:, 9]
                            c12 = tapes(F, {**base, p1: f1 * S, p2: f2 * S})[:, 9]
                            small = np.minimum(nrm(c1 - c0), nrm(c2 - c0))
                            ok = small > 0.3
                            key = "adjacent" if p2 - p1 == 1 else "2 apart" if p2 - p1 == 2 else "3+ apart"
                            out.setdefault(key, []).append((nrm(c12 - c1 - c2 + c0) / np.maximum(small, 1e-9))[ok])
    print(f"      filler {name:7s} " + "   ".join(f"{k}: {q3(np.concatenate(v))}, share < 0.1: {np.mean(np.concatenate(v) < 0.1):.2f} (n = {len(np.concatenate(v))})" for k, v in out.items()))

print("   D2. Runner-up S and third T = f S, both after the maximum at position 0; 'neither' = the maximum with fillers.")
for name in ("zero", "minus1"):
    F = FILLS[name]
    for order in ("S first", "T first"):
        row = []
        for f in (0.3, 0.6, 0.8, 0.9, 0.99):
            e = []
            for S in (0.3, 0.6, 0.9):
                for p1 in range(1, 8):
                    for p2 in range(p1 + 1, 8):
                        v1, v2 = (S, f * S) if order == "S first" else (f * S, S)
                        c0, c1, c2 = tapes(F, {0: 1.0})[:, 9], tapes(F, {0: 1.0, p1: v1})[:, 9], tapes(F, {0: 1.0, p2: v2})[:, 9]
                        c12 = tapes(F, {0: 1.0, p1: v1, p2: v2})[:, 9]
                        e.append(nrm(c12 - c1 - c2 + c0) / np.minimum(nrm(c1 - c0), nrm(c2 - c0)))
            row.append(f"f={f}: {q3(np.concatenate(e))}")
        print(f"      filler {name:7s} {order}:  " + "   ".join(row))

print("   D3. Maximum 1 and runner-up S, nothing else (filler 0): tape(1, S) against tape(1 alone) + tape(S alone).  NA relative to |tape(S alone)|.")
for order in ("S after the maximum", "S before the maximum"):
    row = []
    for S in (0.3, 0.6, 0.9):
        e = []
        for p1 in range(0, 7):
            for p2 in range(p1 + 1, 8):
                pM, pS = (p1, p2) if order.startswith("S after") else (p2, p1)
                c1, c2, c12 = tapes(F0, {pM: 1.0})[:, 9], tapes(F0, {pS: S})[:, 9], tapes(F0, {pM: 1.0, pS: S})[:, 9]
                e.append(nrm(c12 - c1 - c2) / nrm(c2))
        row.append(f"S={S}: {q3(np.concatenate(e))}")
    print(f"      {order}:  " + "   ".join(row))
np.savez(HERE / "d2_patterns.npz", P_M=to7(P_M))
