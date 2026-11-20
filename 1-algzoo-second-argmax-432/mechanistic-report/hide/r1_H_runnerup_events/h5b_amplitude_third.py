"""H5 parts 3-4.  (3) amplitude law of the S record in (S, M, b);  (4) a third value T < S."""
from hcommon import *
from h5_controlled import seq, Mv

# ---------- 3. amplitude law of the record just after it is written (a2 = 2), per age of the max at the event ----------
print("=== 3. S record at a2 = 2 (final step = p2 + 2):  dm_k / K ~ cS * S + cM * M + cb * b   over S/M in 0.1..0.97 and b in {-1,-0.6,-0.3,-0.1} ===")
SRs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97]
BGs = [-1.0, -0.6, -0.3, -0.1]
for n in ["runner", "clock", "h2", "alt", "h3", "short"]:
    cplx = IS_COMPLEX[n]
    print(f"  mode {n}:   a_ev | cS | cM | cb | R2 | record/(K S) at S/M = 0.2, 0.5, 0.8 (b=-0.3)")
    for a_ev in range(1, 8):
        p2 = 7; p1 = p2 - a_ev
        rows, ys = [], []
        for sr in SRs:
            for b in BGs:
                S = sr * Mv
                h1 = run(seq(p1, p2, S, b))[0, -1]; h0 = run(seq(p1, p2, b, b))[0, -1]
                ys.append((coords(h1)[n] - coords(h0)[n]) / K); rows.append([S, Mv, b])
        X = np.array(rows); y = np.array(ys)
        coef, *_ = np.linalg.lstsq(X.astype(complex) if cplx else X, y if cplx else y.real, rcond=None)
        res = (y if cplx else y.real) - X @ coef
        R2 = 1 - np.sum(np.abs(res) ** 2) / np.sum(np.abs(y) ** 2)
        f = (lambda c: f"{abs(c):5.2f}@{np.degrees(np.angle(c)):6.1f}") if cplx else (lambda c: f"{np.real(c):6.2f}")
        ex = [y[SRs.index(sr) * len(BGs) + BGs.index(-0.3)] / (sr * Mv) for sr in (0.2, 0.5, 0.8)]
        print(f"            {a_ev}   | {f(coef[0])} | {f(coef[1])} | {f(coef[2])} | {R2:.3f} | " + "  ".join(f(v) for v in ex))

# the same for the unit-5 pulse itself
print("\n  unit-5 pulse h_{p2+1}[5] ~ cS S + cM M + cb b (same grid), and number of comparators clipped at p2 for S/M = 0.2, 0.5, 0.8:")
for a_ev in range(1, 8):
    p2 = 7; p1 = p2 - a_ev
    rows, ys, nc = [], [], []
    for sr in SRs:
        for b in BGs:
            H, Z = run(seq(p1, p2, sr * Mv, b), return_pre=True)
            ys.append(H[0, p2 + 1, 5]); rows.append([sr * Mv, Mv, b])
            if b == -0.3 and sr in (0.2, 0.5, 0.8): nc.append("{" + ",".join(str(j) for j in COMP if Z[0, p2, j] <= 0) + "}")
    X = np.array(rows); y = np.array(ys)
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    R2 = 1 - np.sum((y - X @ coef) ** 2) / np.sum(y ** 2)
    print(f"    a_ev={a_ev}: h5 = {coef[0]:6.2f} S + {coef[1]:6.2f} M + {coef[2]:6.2f} b   (R2 {R2:.3f});  clipped comparators: {nc}")

# ---------- 4. third value ----------
print("\n=== 4. a third value T < S at a third position p3 (M at p1 < p2 = position of S) ===")
SR3 = [0.3, 0.5, 0.7, 0.9]
TR = [0.2, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98]
rec = []
for b in (-1.0, -0.3):
    for sr in SR3:
        for tr in TR:
            S, Tv = sr * Mv, tr * sr * Mv
            xs, meta = [], []
            for p1 in range(9):
                for p2 in range(p1 + 1, 10):
                    for p3 in range(10):
                        if p3 in (p1, p2): continue
                        xs.append(seq(p1, p2, S, b, extra=(p3, Tv)))
                        order = "T,M,S" if p3 < p1 else ("M,T,S" if p3 < p2 else "M,S,T")
                        meta.append((p1, p2, p3, order))
            lg = logits(run(np.array(xs)))
            for (p1, p2, p3, order), l in zip(meta, lg):
                rec.append((b, sr, tr, p1, p2, p3, order, l[p2] - l[p3], l[p2] - l[p1], int(l.argmax()), S - Tv, S, Tv))
import collections
orders = ["T,M,S", "M,T,S", "M,S,T"]
arr = lambda k: np.array([r[k] for r in rec])
B_, SR_, TR_, P1, P2, P3 = arr(0), arr(1), arr(2), arr(3), arr(4), arr(5)
ORD_ = np.array([r[6] for r in rec]); MG, MGmax, ANS, DST, Sv, Tvv = arr(7), arr(8), arr(9), arr(10), arr(11), arr(12)
for b in (-1.0, -0.3):
    print(f"\n background {b}: P(answer = p2) | P(answer = p3) | median margin(p2 over p3)/M, by order of arrival and T/S (pooled over S/M in {SR3} and positions)")
    print("   order  | " + " | ".join(f"   T/S={tr:4.2f}      " for tr in TR))
    for o in orders:
        row = ""
        for tr in TR:
            s = (B_ == b) & (ORD_ == o) & (TR_ == tr)
            row += f" {np.mean(ANS[s] == P2[s]):.2f} {np.mean(ANS[s] == P3[s]):.2f} {np.median(MG[s]) / Mv:6.1f} |"
        print(f"   {o:6s} | {row}  n per cell = {((B_ == b) & (ORD_ == o) & (TR_ == TR[0])).sum()}")
print("\n regression  margin(p2 over p3) ~ c1 (S - T) + c2 S + c3 M   for T/S >= 0.7 (T's record exists), by order and background:")
for b in (-1.0, -0.3):
    for o in orders:
        s = (B_ == b) & (ORD_ == o) & (TR_ >= 0.7)
        coef, R2 = lsq(MG[s], [DST[s], Sv[s], np.full(s.sum(), Mv)])
        print(f"   b={b:4.1f} {o}: {coef[0]:7.1f} (S-T) + {coef[1]:6.1f} S + {coef[2]:6.1f} M    R2 = {R2:.3f}   n = {s.sum()}")
print("\n the same by time since the later of the two arrivals (d = 9 - max(p2, p3)), b = -0.3, T/S >= 0.7: c1 (per unit of S - T), c2, c3, R2")
for o in orders:
    row = ""
    for dd in range(0, 8):
        s = (B_ == -0.3) & (ORD_ == o) & (TR_ >= 0.7) & (9 - np.maximum(P2, P3) == dd)
        if s.sum() < 30: row += "      .      |"; continue
        coef, R2 = lsq(MG[s], [DST[s], Sv[s], np.full(s.sum(), Mv)])
        row += f" {coef[0]:6.1f} {coef[1]:5.1f} {coef[2]:5.1f} ({R2:.2f}) |"
    print(f"   {o}: {row}")
print("\n wrong answers at T/S = 0.95 and 0.98 by order (share answering p3 / p1 / other), b = -0.3:")
for o in orders:
    for tr in (0.95, 0.98):
        s = (B_ == -0.3) & (ORD_ == o) & (TR_ == tr)
        print(f"   {o} T/S={tr}: p3 {np.mean(ANS[s] == P3[s]):.3f}  p1 {np.mean(ANS[s] == P1[s]):.3f}  other {np.mean((ANS[s] != P3[s]) & (ANS[s] != P1[s]) & (ANS[s] != P2[s])):.3f}")
